import io
import json
from pathlib import Path

import chromadb
import pytest
from click.testing import CliRunner

import rag


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


_SUMMARY = (
    "Leí un artículo sobre retrieval-augmented generation. El autor sostiene "
    "que combinar búsqueda densa con reranking cruzado sube la precisión. "
    "Se conecta con mis notas sobre [[rag-basics]]: confirma la intuición "
    "de priorizar el reranker. Me queda abierta la pregunta de cuánto "
    "pesa el chunking en el resultado final."
)


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def fake_chat(monkeypatch):
    """Default chat stub: summary for command-r, empty YAML list for helper
    (tag picker). Tests that need richer tag picking override this.
    """
    def _chat(model, messages, options=None, keep_alive=None, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        if "VOCABULARIO" in prompt and "TAGS:" in prompt:
            return _FakeResponse("- rag\n- ml")
        return _FakeResponse(_SUMMARY)
    monkeypatch.setattr(rag.ollama, "chat", _chat)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed, fake_chat):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "queries.jsonl")
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="read_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(
        rag, "_index_single_file", lambda *a, **kw: "indexed",
    )
    monkeypatch.setattr(
        rag, "resolve_chat_model", lambda: "fake-model",
    )
    rag._invalidate_corpus_cache()
    return vault, col


def _add_note_chunk(col, path, note, folder="02-Areas", tags="", outlinks=""):
    col.add(
        ids=[f"{path}::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=[f"chunk of {note}"],
        metadatas=[{
            "file": path, "note": note, "folder": folder,
            "tags": tags, "outlinks": outlinks, "hash": "x",
        }],
    )


def _fake_html(title="Intro to RAG", body_paragraphs=None):
    body_paragraphs = body_paragraphs or [
        "Retrieval-augmented generation combines a retriever with a "
        "generator to ground answers in documents.",
        "The retriever embeds the query and finds the nearest chunks "
        "in an index. A cross-encoder reranks the top candidates.",
        "Chunking matters: too small loses context, too big dilutes "
        "the reranker signal. Typical sizes are 200-800 characters.",
        "This post is long enough to clear the 500-char minimum the "
        "reader enforces, so it should pass ingestion without complaint. "
        "Lorem ipsum dolor sit amet consectetur adipiscing elit.",
    ]
    body = "\n".join(f"<p>{p}</p>" for p in body_paragraphs)
    return (
        "<!DOCTYPE html><html><head>"
        f"<title>{title}</title>"
        "<script>var tracker = 1;</script>"
        "<style>body{color:red}</style>"
        "</head><body>"
        "<nav>Home | About</nav>"
        "<header>site header</header>"
        f"<main>{body}</main>"
        "<footer>copyright 2026</footer>"
        "</body></html>"
    )


# ── _read_extract ────────────────────────────────────────────────────────────


def test_extract_strips_script_style_nav_header_footer():
    html = _fake_html()
    title, text = rag._read_extract(html)
    assert title == "Intro to RAG"
    assert "tracker" not in text
    assert "color:red" not in text
    assert "site header" not in text
    assert "Home | About" not in text
    assert "copyright 2026" not in text
    assert "Retrieval-augmented" in text


def test_extract_handles_entities():
    html = "<html><head><title>X &amp; Y</title></head><body>" + \
           ("<p>Tom &amp; Jerry &mdash; foo bar baz. " * 20) + "</p></body></html>"
    title, text = rag._read_extract(html)
    assert title == "X & Y"
    assert "Tom & Jerry" in text
    assert "&amp;" not in text


def test_extract_returns_empty_title_when_missing():
    html = "<html><body>" + ("<p>short content only " * 60) + "</p></body></html>"
    title, text = rag._read_extract(html)
    assert title == ""
    assert "short content only" in text


# ── _read_slug_from ──────────────────────────────────────────────────────────


def test_slug_from_title():
    assert "intro-to-rag" in rag._read_slug_from("Intro to RAG!", "https://x.io/a")


def test_slug_falls_back_to_host():
    assert "example-com" in rag._read_slug_from("", "https://www.example.com/path")


# ── ingest_read_url (dry-run + save) ─────────────────────────────────────────


def test_ingest_dry_run_does_not_write(tmp_vault):
    vault, col = tmp_vault
    html = _fake_html()
    result = rag.ingest_read_url(
        col, "https://example.com/post",
        save=False,
        fetcher=lambda u: (html, {}),
    )
    assert result["path"] is None
    assert list((vault / "00-Inbox").iterdir()) == []
    assert "RAG" in result["summary"] or "rag" in result["summary"].lower()


def test_ingest_save_writes_to_inbox(tmp_vault):
    vault, col = tmp_vault
    html = _fake_html(title="My Article")
    result = rag.ingest_read_url(
        col, "https://example.com/post",
        save=True,
        fetcher=lambda u: (html, {}),
    )
    assert result["path"] is not None
    assert result["path"].parent == vault / "00-Inbox"
    assert result["path"].is_file()
    assert "-read-" in result["path"].name
    assert "my-article" in result["path"].name


def test_ingest_save_frontmatter_shape(tmp_vault):
    vault, col = tmp_vault
    html = _fake_html(title="RAG Primer")
    result = rag.ingest_read_url(
        col, "https://example.com/rag",
        save=True,
        fetcher=lambda u: (html, {}),
    )
    txt = result["path"].read_text(encoding="utf-8")
    assert txt.startswith("---\n")
    assert "type: read" in txt
    assert "source: https://example.com/rag" in txt
    assert 'title: "RAG Primer"' in txt
    assert "related:" in txt
    assert "- read" in txt
    assert "# RAG Primer" in txt


def test_ingest_rejects_short_content(tmp_vault):
    _, col = tmp_vault
    html = "<html><head><title>tiny</title></head><body><p>too short</p></body></html>"
    with pytest.raises(RuntimeError, match="insuficiente"):
        rag.ingest_read_url(
            col, "https://example.com/tiny",
            save=False,
            fetcher=lambda u: (html, {}),
        )


def test_ingest_rejects_short_content_no_write(tmp_vault):
    vault, col = tmp_vault
    html = "<html><body><p>short</p></body></html>"
    with pytest.raises(RuntimeError):
        rag.ingest_read_url(
            col, "https://example.com/x",
            save=True,
            fetcher=lambda u: (html, {}),
        )
    assert list((vault / "00-Inbox").iterdir()) == []


def test_ingest_empty_summary_raises(tmp_vault, monkeypatch):
    _, col = tmp_vault

    def _chat(model, messages, options=None, keep_alive=None, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        if "VOCABULARIO" in prompt:
            return _FakeResponse("")
        return _FakeResponse("   ")

    monkeypatch.setattr(rag.ollama, "chat", _chat)
    html = _fake_html()
    with pytest.raises(RuntimeError, match="vacío"):
        rag.ingest_read_url(
            col, "https://example.com/x", save=False,
            fetcher=lambda u: (html, {}),
        )


# ── tags from existing vocab only ────────────────────────────────────────────


def test_ingest_tags_only_from_existing_vocab(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    # Vault has limited vocab.
    _add_note_chunk(col, "02-Areas/a.md", "note-a", tags="rag,ml")
    _add_note_chunk(col, "02-Areas/b.md", "note-b", tags="coaching")
    rag._invalidate_corpus_cache()

    # Helper returns a hallucinated tag PLUS a real one.
    def _chat(model, messages, options=None, keep_alive=None, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        if "VOCABULARIO" in prompt and "TAGS:" in prompt:
            return _FakeResponse("- invented-tag\n- rag\n- madeup")
        return _FakeResponse(_SUMMARY)

    monkeypatch.setattr(rag.ollama, "chat", _chat)
    html = _fake_html()
    result = rag.ingest_read_url(
        col, "https://example.com/post",
        save=False,
        fetcher=lambda u: (html, {}),
    )
    assert "invented-tag" not in result["tags"]
    assert "madeup" not in result["tags"]
    assert "rag" in result["tags"]


def test_ingest_related_uses_vault_titles(tmp_vault):
    _, col = tmp_vault
    _add_note_chunk(col, "02-Areas/rag-basics.md", "rag-basics", tags="rag")
    _add_note_chunk(col, "02-Areas/coaching.md", "coaching", tags="coaching")
    rag._invalidate_corpus_cache()
    html = _fake_html()
    result = rag.ingest_read_url(
        col, "https://example.com/post",
        save=True,
        fetcher=lambda u: (html, {}),
    )
    # related should reference actual vault notes.
    assert result["related"]
    for t in result["related"]:
        assert t in {"rag-basics", "coaching"}
    txt = result["path"].read_text(encoding="utf-8")
    assert "[[rag-basics]]" in txt or "[[coaching]]" in txt


# ── _read_fetch_url network errors ───────────────────────────────────────────


def test_fetch_wraps_http_error(monkeypatch):
    import urllib.error

    class _Ctx:
        def __enter__(self_inner): raise urllib.error.HTTPError(
            "https://x.io", 404, "Not Found", {}, None,
        )
        def __exit__(self_inner, *a): return False

    def _urlopen(req, timeout=None):
        raise urllib.error.HTTPError("https://x.io", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    with pytest.raises(RuntimeError, match="404"):
        rag._read_fetch_url("https://x.io/")


def test_fetch_wraps_url_error(monkeypatch):
    import urllib.error

    def _urlopen(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    with pytest.raises(RuntimeError, match="Error de red"):
        rag._read_fetch_url("https://x.io/")


# ── CLI surface (`rag read`) ─────────────────────────────────────────────────


def test_cli_rejects_bad_scheme(tmp_vault):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["read", "ftp://nope.invalid/"])
    assert result.exit_code == 1
    assert "URL inválida" in result.output or "inválida" in result.output.lower()


def test_cli_dry_run_prints_preview(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    html = _fake_html(title="CLI Test")
    monkeypatch.setattr(
        rag, "_read_fetch_url", lambda url, timeout=None: (html, {}),
    )
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["read", "https://example.com/post", "--plain"],
    )
    assert result.exit_code == 0
    assert list((vault / "00-Inbox").iterdir()) == []
    assert "type: read" in result.output


def test_cli_save_writes_file(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    html = _fake_html(title="Saved Post")
    monkeypatch.setattr(
        rag, "_read_fetch_url", lambda url, timeout=None: (html, {}),
    )
    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["read", "https://example.com/saved", "--save", "--plain"],
    )
    assert result.exit_code == 0
    files = list((vault / "00-Inbox").iterdir())
    assert len(files) == 1
    assert "-read-saved-post" in files[0].name
    txt = files[0].read_text(encoding="utf-8")
    assert "source: https://example.com/saved" in txt


def test_cli_short_content_exits_nonzero(tmp_vault, monkeypatch):
    monkeypatch.setattr(
        rag, "_read_fetch_url",
        lambda url, timeout=None: (
            "<html><body><p>x</p></body></html>", {},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["read", "https://example.com/short", "--plain"],
    )
    assert result.exit_code == 1
    assert "insuficiente" in result.output.lower()


def test_cli_network_failure_exits_nonzero(tmp_vault, monkeypatch):
    def _boom(url, timeout=None):
        raise RuntimeError("Error de red: unreachable")
    monkeypatch.setattr(rag, "_read_fetch_url", _boom)
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["read", "https://example.com/down", "--plain"],
    )
    assert result.exit_code == 1
    assert "red" in result.output.lower()
