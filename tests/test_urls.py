import chromadb
import pytest

import rag


# ── extract_urls (pure) ──────────────────────────────────────────────────────


def test_extract_markdown_link():
    text = "Mira la [doc oficial](https://docs.example.com/x) que explica todo."
    out = rag.extract_urls(text)
    assert len(out) == 1
    assert out[0]["url"] == "https://docs.example.com/x"
    assert out[0]["anchor"] == "doc oficial"
    assert out[0]["line"] == 1
    assert "doc oficial" in out[0]["context"]


def test_extract_bare_url():
    text = "Bookmark: https://example.com/repo es el repo."
    out = rag.extract_urls(text)
    assert len(out) == 1
    assert out[0]["url"] == "https://example.com/repo"
    assert out[0]["anchor"] == ""


def test_extract_strips_trailing_punctuation():
    text = "Ver https://example.com/x. Y también https://example.com/y, ok?"
    urls = [u["url"] for u in rag.extract_urls(text)]
    assert "https://example.com/x" in urls
    assert "https://example.com/y" in urls


def test_extract_dedups_within_file():
    text = (
        "Ver [doc](https://docs.example.com/x) y también https://docs.example.com/x "
        "que aparece dos veces."
    )
    out = rag.extract_urls(text)
    # Same URL appears twice; markdown-style is consumed first, bare second is skipped.
    assert len(out) == 1
    assert out[0]["url"] == "https://docs.example.com/x"
    assert out[0]["anchor"] == "doc"


def test_extract_preserves_line_numbers():
    text = "linea uno\n[doc](https://example.com/y)\nlinea tres"
    out = rag.extract_urls(text)
    assert out[0]["line"] == 2


def test_extract_no_urls():
    assert rag.extract_urls("solo prosa, sin links.") == []


def test_extract_handles_multiple_links_same_line():
    text = "[a](https://a.com/x) y [b](https://b.com/y) lado a lado."
    out = rag.extract_urls(text)
    assert {u["url"] for u in out} == {"https://a.com/x", "https://b.com/y"}


def test_extract_ignores_url_inside_markdown_link():
    # The bare URL https://docs.example.com/x is inside a markdown link.
    # Bare-URL pass must not double-flag it.
    text = "[oficial](https://docs.example.com/x)"
    out = rag.extract_urls(text)
    assert len(out) == 1


# ── find_urls (with stubbed embed/reranker/chroma) ──────────────────────────


class _FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        return [1.0 - i * 0.01 for i in range(len(pairs))]


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def fake_reranker(monkeypatch):
    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())


@pytest.fixture
def urls_col(tmp_path, fake_embed, fake_reranker, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    c = client.get_or_create_collection(
        name="urls_test", metadata={"hnsw:space": "cosine"}
    )
    main = client.get_or_create_collection(
        name="urls_test_main", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_urls_db", lambda: c)
    # The auto-backfill in find_urls peeks at get_db() — point it at an
    # empty tmp collection too so backfill never touches the real vault.
    monkeypatch.setattr(rag, "get_db", lambda: main)
    # Reset the per-process backfill latch so each test starts clean.
    monkeypatch.setattr(rag, "_URLS_BACKFILL_DONE", False)
    return c


def _add_url(col, idx, url, context, file_, note, anchor="", line=1):
    col.add(
        ids=[f"{file_}::url::{idx}"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=[context],
        metadatas=[{
            "file": file_, "note": note, "folder": "",
            "tags": "", "url": url, "anchor": anchor, "line": line,
        }],
    )


def test_find_urls_empty_collection_returns_empty(urls_col):
    assert rag.find_urls("anything") == []


def test_find_urls_returns_indexed_urls(urls_col):
    _add_url(urls_col, 0, "https://docs.example.com/x",
             "doc oficial sobre X", "notes/x.md", "X")
    _add_url(urls_col, 1, "https://example.com/repo",
             "repo en github", "notes/y.md", "Y")
    out = rag.find_urls("documentación de X", k=5)
    assert len(out) == 2
    assert any(it["url"] == "https://docs.example.com/x" for it in out)


def test_find_urls_dedups_by_url_across_files(urls_col):
    _add_url(urls_col, 0, "https://shared.com/x", "primer mención",
             "a.md", "A")
    _add_url(urls_col, 1, "https://shared.com/x", "segunda mención",
             "b.md", "B")
    out = rag.find_urls("x", k=10)
    urls = [it["url"] for it in out]
    assert urls.count("https://shared.com/x") == 1


def test_find_urls_respects_k(urls_col):
    for i in range(8):
        _add_url(urls_col, i, f"https://example.com/{i}",
                 f"contexto número {i}", f"n{i}.md", f"N{i}")
    out = rag.find_urls("contexto", k=3)
    assert len(out) == 3


def test_find_urls_filters_by_folder(urls_col):
    urls_col.add(
        ids=["a.md::url::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=["alpha"],
        metadatas=[{
            "file": "alpha/a.md", "note": "A", "folder": "alpha",
            "tags": "", "url": "https://a.com", "anchor": "", "line": 1,
        }],
    )
    urls_col.add(
        ids=["b.md::url::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=["beta"],
        metadatas=[{
            "file": "beta/b.md", "note": "B", "folder": "beta",
            "tags": "", "url": "https://b.com", "anchor": "", "line": 1,
        }],
    )
    out = rag.find_urls("query", k=10, folder="alpha")
    assert all("alpha" in it["path"] for it in out)


def test_find_urls_returns_anchor_and_context(urls_col):
    _add_url(urls_col, 0, "https://docs.example.com/x",
             "tutorial completo de X paso a paso",
             "notes/x.md", "X", anchor="tutorial X", line=12)
    out = rag.find_urls("tutorial X", k=1)
    assert out[0]["anchor"] == "tutorial X"
    assert out[0]["line"] == 12
    assert "tutorial completo" in out[0]["context"]


# ── _index_urls (idempotent replace) ─────────────────────────────────────────


def test_index_urls_writes_and_replaces(urls_col):
    text1 = "[primer link](https://a.com/1) en el body."
    n = rag._index_urls(urls_col, "n.md", text1, "n", "", [])
    assert n == 1
    out = rag.find_urls("primer", k=5)
    assert any(it["url"] == "https://a.com/1" for it in out)

    # Re-index with different content — old URL is gone, new one appears.
    text2 = "Ahora [otro link](https://b.com/2) reemplaza al primero."
    n2 = rag._index_urls(urls_col, "n.md", text2, "n", "", [])
    assert n2 == 1
    urls_now = {it["url"] for it in rag.find_urls("link", k=10)}
    assert "https://b.com/2" in urls_now
    assert "https://a.com/1" not in urls_now


def test_index_urls_zero_when_none(urls_col):
    n = rag._index_urls(urls_col, "n.md", "sin links de ningún tipo.", "n", "", [])
    assert n == 0


# ── detect_link_intent ──────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "donde está el link a la documentación de claude code",
    "dame la url de docs",
    "url del repo",
    "donde tengo el doc de obsidian",
    "documentación de bge-m3",
    "link al paper",
    "/links docs claude",
])
def test_link_intent_matches(text):
    matched, _ = rag.detect_link_intent(text)
    assert matched, f"should match: {text}"


@pytest.mark.parametrize("text", [
    "qué dice X sobre Y",
    "explicame ikigai",
    "hace cuánto que escribí sobre música",
    "guardá esto",
    "reindexá las notas",
])
def test_link_intent_does_not_match(text):
    matched, _ = rag.detect_link_intent(text)
    assert not matched, f"should NOT match: {text}"


def test_link_intent_slash_passes_residual_query():
    matched, q = rag.detect_link_intent("/links docs claude")
    assert matched
    assert q == "docs claude"


def test_link_intent_slash_no_query():
    matched, q = rag.detect_link_intent("/links")
    assert matched
    assert q is None
