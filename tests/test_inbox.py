import chromadb
import pytest

import rag


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        # Match the dim used by the fixture below.
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def vault_with_corpus(tmp_path, monkeypatch, fake_embed):
    """Tmp vault + collection populated with a typed corpus so folder
    suggestion has somewhere reasonable to point.
    """
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    (vault / "02-Areas/Coaching").mkdir(parents=True)
    (vault / "02-Areas/Música").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="inbox_test", metadata={"hnsw:space": "cosine"}
    )
    # Three Coaching notes (the "winning" folder for any incoming similar note),
    # one Música note, one Inbox note (skipped from folder suggestions).
    rows = [
        ("02-Areas/Coaching/Ikigai.md", "Ikigai", "02-Areas/Coaching",
         "coaching"),
        ("02-Areas/Coaching/Liderazgo.md", "Liderazgo", "02-Areas/Coaching",
         "coaching,liderazgo"),
        ("02-Areas/Coaching/Modelos.md", "Modelos", "02-Areas/Coaching",
         "coaching,modelos"),
        ("02-Areas/Música/Letras.md", "Letras", "02-Areas/Música", "musica"),
        ("00-Inbox/old.md", "old", "00-Inbox", ""),
    ]
    for file_, note, folder, tags in rows:
        col.add(
            ids=[f"{file_}::0"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            documents=[f"chunk for {note}"],
            metadatas=[{
                "file": file_, "note": note, "folder": folder, "tags": tags,
                "outlinks": "",
            }],
        )
    rag._invalidate_corpus_cache()
    return vault, col


# ── _suggest_folder_for_note ─────────────────────────────────────────────────


def test_folder_suggestion_picks_majority(vault_with_corpus):
    vault, col = vault_with_corpus
    # Drop a fresh note in Inbox — it'll match the Coaching cluster (most chunks).
    (vault / "00-Inbox" / "new.md").write_text("notas sobre coaching y propósito.")
    folder, conf = rag._suggest_folder_for_note(col, "00-Inbox/new.md")
    assert folder == "02-Areas/Coaching"
    assert conf > 0.0


def test_folder_suggestion_skips_inbox(vault_with_corpus):
    vault, col = vault_with_corpus
    (vault / "00-Inbox" / "new.md").write_text("contenido")
    folder, _ = rag._suggest_folder_for_note(col, "00-Inbox/new.md")
    # Even though Inbox notes are in the candidate pool, they're skipped.
    assert not folder.startswith("00-")


def test_folder_suggestion_returns_empty_for_missing_file(vault_with_corpus):
    _, col = vault_with_corpus
    folder, conf = rag._suggest_folder_for_note(col, "nope.md")
    assert (folder, conf) == ("", 0.0)


def test_folder_suggestion_empty_body_returns_empty(vault_with_corpus):
    vault, col = vault_with_corpus
    (vault / "00-Inbox" / "blank.md").write_text("")
    folder, _ = rag._suggest_folder_for_note(col, "00-Inbox/blank.md")
    assert folder == ""


# ── _suggest_tags_for_note ───────────────────────────────────────────────────


def test_suggest_tags_filters_to_vocab(vault_with_corpus, monkeypatch):
    _, col = vault_with_corpus
    # Helper "returns" three candidates: two real, one made-up.
    monkeypatch.setattr(
        rag.ollama, "chat",
        lambda **kw: _FakeResponse("- coaching\n- liderazgo\n- inventado"),
    )
    picked = rag._suggest_tags_for_note(col, "body about leadership coaching", "Note")
    assert "coaching" in picked
    assert "liderazgo" in picked
    assert "inventado" not in picked  # not in vault vocab


def test_suggest_tags_empty_body_returns_empty(vault_with_corpus, monkeypatch):
    _, col = vault_with_corpus
    called = {"n": 0}
    def _chat(**kw):
        called["n"] += 1
        return _FakeResponse("- coaching")
    monkeypatch.setattr(rag.ollama, "chat", _chat)
    out = rag._suggest_tags_for_note(col, "", "Note")
    assert out == []
    assert called["n"] == 0  # short-circuits without calling helper


def test_suggest_tags_handles_helper_exception(vault_with_corpus, monkeypatch):
    _, col = vault_with_corpus
    def _boom(**kw):
        raise RuntimeError("helper down")
    monkeypatch.setattr(rag.ollama, "chat", _boom)
    assert rag._suggest_tags_for_note(col, "body", "Note") == []


# ── _apply_frontmatter_tags ──────────────────────────────────────────────────


def test_apply_frontmatter_replaces_tag_block(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("---\ntitle: foo\ntags:\n- old\n---\n\nbody")
    assert rag._apply_frontmatter_tags(p, ["new1", "new2"]) is True
    out = p.read_text()
    assert "tags:\n- new1\n- new2" in out
    assert "title: foo" in out  # other fields preserved
    assert "body" in out


def test_apply_frontmatter_inserts_when_absent(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("just a body without frontmatter")
    rag._apply_frontmatter_tags(p, ["x"])
    out = p.read_text()
    assert out.startswith("---\ntags:\n- x\n---\n\n")
    assert "just a body" in out


def test_apply_frontmatter_returns_false_for_missing_file(tmp_path):
    assert rag._apply_frontmatter_tags(tmp_path / "missing.md", ["x"]) is False


def test_apply_frontmatter_handles_malformed_yaml(tmp_path):
    p = tmp_path / "n.md"
    # Opens with --- but never closes
    p.write_text("---\ntitle: foo\nNO CLOSING")
    assert rag._apply_frontmatter_tags(p, ["x"]) is False


# ── triage_inbox_note (composition) ──────────────────────────────────────────


def test_triage_returns_full_plan(vault_with_corpus, monkeypatch):
    vault, col = vault_with_corpus
    monkeypatch.setattr(
        rag.ollama, "chat",
        lambda **kw: _FakeResponse("- coaching"),
    )
    (vault / "00-Inbox" / "fresh.md").write_text("Hablando de Ikigai en coaching.")
    rag._invalidate_corpus_cache()
    t = rag.triage_inbox_note(col, "00-Inbox/fresh.md")
    assert t["path"] == "00-Inbox/fresh.md"
    assert t["folder_suggested"]  # something
    assert "current_folder" in t
    # wikilinks: should pick up "Ikigai" since it's a note title
    assert any(w["title"] == "Ikigai" for w in t["wikilinks"])


def test_triage_missing_note_returns_error(vault_with_corpus):
    _, col = vault_with_corpus
    t = rag.triage_inbox_note(col, "nope.md")
    assert t.get("error") == "not found"
