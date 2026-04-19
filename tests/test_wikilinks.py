from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


# ── _wikilink_skip_spans ─────────────────────────────────────────────────────


def test_skip_frontmatter():
    text = "---\ntitle: foo\ntags: [a,b]\n---\nbody Ikigai mentioned"
    spans = rag._wikilink_skip_spans(text)
    fm_end = text.find("\n---") + 4
    assert any(s == 0 and e >= fm_end for s, e in spans)


def test_skip_fenced_code():
    text = "Ikigai antes\n```\nIkigai en código\n```\nIkigai después"
    spans = rag._wikilink_skip_spans(text)
    code_start = text.find("```")
    code_end = text.rfind("```") + 3
    assert any(s <= code_start and e >= code_end for s, e in spans)


def test_skip_inline_code():
    text = "ver `Ikigai` literal y Ikigai libre"
    spans = rag._wikilink_skip_spans(text)
    code_start = text.find("`")
    assert any(s == code_start for s, e in spans)


def test_skip_existing_wikilinks():
    text = "ya hay [[Ikigai]] acá"
    spans = rag._wikilink_skip_spans(text)
    wl_start = text.find("[[")
    assert any(s == wl_start for s, e in spans)


def test_skip_markdown_links():
    text = "ver [Ikigai](https://x.com/y) externamente"
    spans = rag._wikilink_skip_spans(text)
    ml_start = text.find("[")
    assert any(s == ml_start for s, e in spans)


def test_skip_html_tags():
    text = "<span>Ikigai</span> con tags"
    spans = rag._wikilink_skip_spans(text)
    assert spans  # at least the tags


# ── find_wikilink_suggestions ─────────────────────────────────────────────────


@pytest.fixture
def vault_with_titles(tmp_path, monkeypatch):
    """Set up a tmp vault with three titled notes whose paths feed
    `_load_corpus`'s `title_to_paths` index. Patches VAULT_PATH and gives
    a real chroma collection populated with the metadata structure that
    `_load_corpus` expects.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Ikigai.md").write_text("Ikigai es un concepto japonés.\n")
    (vault / "Moka.md").write_text("Moka es una empresa.\n")
    (vault / "TDD.md").write_text("test driven development.\n")  # short title
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    client = _TestVecClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="wl_test", metadata={"hnsw:space": "cosine"}
    )
    # Add one chunk per note with the title in metadata.
    for i, (note, file_) in enumerate([
        ("Ikigai", "Ikigai.md"),
        ("Moka", "Moka.md"),
        ("TDD", "TDD.md"),
    ]):
        col.add(
            ids=[f"{file_}::0"],
            embeddings=[[0.1 * i, 0.2, 0.3, 0.4]],
            documents=[f"chunk for {note}"],
            metadatas=[{
                "file": file_, "note": note, "folder": "",
                "tags": "", "outlinks": "", "hash": "x",
            }],
        )
    rag._invalidate_corpus_cache()
    return vault, col


def test_suggests_unlinked_title(vault_with_titles):
    vault, col = vault_with_titles
    new_path = vault / "thoughts.md"
    new_path.write_text("Hoy pensé sobre Ikigai y su impacto.")
    sugs = rag.find_wikilink_suggestions(col, "thoughts.md")
    assert any(s["title"] == "Ikigai" for s in sugs)


def test_skips_already_linked(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "linked.md").write_text("Hablando de [[Ikigai]] que es importante.")
    sugs = rag.find_wikilink_suggestions(col, "linked.md")
    assert not any(s["title"] == "Ikigai" for s in sugs)


def test_skips_inside_code(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "code.md").write_text("normal text\n```\nIkigai en bloque\n```\nfin")
    sugs = rag.find_wikilink_suggestions(col, "code.md")
    assert not any(s["title"] == "Ikigai" for s in sugs)


def test_skips_inside_frontmatter(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "fm.md").write_text("---\ntitle: Ikigai related stuff\n---\nbody normal\n")
    sugs = rag.find_wikilink_suggestions(col, "fm.md")
    assert not any(s["title"] == "Ikigai" for s in sugs)


def test_skips_short_title_by_default(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "test.md").write_text("Trabajamos con TDD a diario.")
    sugs = rag.find_wikilink_suggestions(col, "test.md")
    # TDD is 3 chars, default min_title_len is 4 → suppressed
    assert not any(s["title"] == "TDD" for s in sugs)


def test_min_len_lowered_includes_short(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "test.md").write_text("Trabajamos con TDD a diario.")
    sugs = rag.find_wikilink_suggestions(col, "test.md", min_title_len=3)
    assert any(s["title"] == "TDD" for s in sugs)


def test_skips_self_link(vault_with_titles):
    vault, col = vault_with_titles
    sugs = rag.find_wikilink_suggestions(col, "Ikigai.md")
    # The note about Ikigai shouldn't propose linking to itself
    assert not any(s["title"] == "Ikigai" for s in sugs)


def test_one_suggestion_per_title(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "many.md").write_text(
        "Ikigai aquí.\nIkigai allá.\nMás Ikigai por todos lados."
    )
    sugs = rag.find_wikilink_suggestions(col, "many.md")
    titles = [s["title"] for s in sugs]
    assert titles.count("Ikigai") == 1


def test_word_boundary_avoids_partial(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "partial.md").write_text("Imokaisa no es palabra, ni Mokaesa.")
    sugs = rag.find_wikilink_suggestions(col, "partial.md")
    # No standalone "Moka" here — only embedded inside larger words.
    assert not any(s["title"] == "Moka" for s in sugs)


def test_returns_line_and_context(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "ctx.md").write_text("linea 1\nlinea 2 con Ikigai mencionado\nlinea 3")
    sugs = rag.find_wikilink_suggestions(col, "ctx.md")
    s = next(x for x in sugs if x["title"] == "Ikigai")
    assert s["line"] == 2
    assert "Ikigai" in s["context"]


def test_nonexistent_file_returns_empty(vault_with_titles):
    _, col = vault_with_titles
    assert rag.find_wikilink_suggestions(col, "nope.md") == []


def test_path_escape_returns_empty(vault_with_titles):
    _, col = vault_with_titles
    assert rag.find_wikilink_suggestions(col, "../escape.md") == []


# ── apply_wikilink_suggestions ───────────────────────────────────────────────


def test_apply_wraps_title(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "apply.md").write_text("Hablamos de Ikigai en la reunión.")
    sugs = rag.find_wikilink_suggestions(col, "apply.md")
    n, titles = rag.apply_wikilink_suggestions("apply.md", sugs)
    assert n == 1
    assert "Ikigai" in titles
    assert "[[Ikigai]]" in (vault / "apply.md").read_text()


def test_apply_preserves_other_text(vault_with_titles):
    vault, col = vault_with_titles
    text = "## Header\n\nVarias líneas.\nIkigai mencionado.\n\nFin."
    (vault / "preserve.md").write_text(text)
    sugs = rag.find_wikilink_suggestions(col, "preserve.md")
    rag.apply_wikilink_suggestions("preserve.md", sugs)
    new = (vault / "preserve.md").read_text()
    assert "## Header" in new
    assert "Fin." in new
    assert "[[Ikigai]] mencionado" in new


def test_apply_handles_multiple_suggestions(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "multi.md").write_text("Ikigai y Moka son temas distintos.")
    sugs = rag.find_wikilink_suggestions(col, "multi.md")
    n, titles = rag.apply_wikilink_suggestions("multi.md", sugs)
    assert n == 2
    assert set(titles) == {"Ikigai", "Moka"}
    new = (vault / "multi.md").read_text()
    assert "[[Ikigai]]" in new and "[[Moka]]" in new


def test_apply_skips_stale_offset(vault_with_titles):
    vault, col = vault_with_titles
    (vault / "stale.md").write_text("Ikigai está acá.")
    sugs = rag.find_wikilink_suggestions(col, "stale.md")
    # Mutate the file behind the suggestion's back so the offset no longer
    # points at "Ikigai".
    (vault / "stale.md").write_text("XXX está acá.")
    n, titles = rag.apply_wikilink_suggestions("stale.md", sugs)
    assert n == 0
    assert titles == []
    assert "[[" not in (vault / "stale.md").read_text()


def test_apply_empty_returns_zero(vault_with_titles):
    n, titles = rag.apply_wikilink_suggestions("Ikigai.md", [])
    assert n == 0
    assert titles == []
