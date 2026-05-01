"""Tests para el boost de ranking basado en filename/title match.

Motivación medida en terreno 2026-04-21: nota `03-Resources/dev cycles.md`
con title `# Dev Cycles` aparecía en el retrieve pero con score rerank 0.0
(body = link + embed de imagen, el reranker no ve signal semántica), y el
gate de confianza la filtraba. Para queries como "en qué cycle estamos"
que son semánticamente idénticas al filename/title, la nota quedaba afuera.

El boost `title_match` suma al score rerank una fracción proporcional a
cuántas palabras del query aparecen en (filename ∪ title), tokenizadas
con stopword-removal ES/EN y prefix-match de 4 chars (handlea plurales
simples: "cycle" ≈ "cycles"). Weight default 0.15; tuneable via
`rag tune`. Con ese default, query "dev cycles" contra dev cycles.md
suma +0.15 al rerank score — suficiente para rescatar la nota del gate
sin dominar queries donde el body sí tiene signal.
"""
import pytest

import rag


# ── _title_match_score: pure function over (query, meta) ──────────────────────


def _meta(file: str, note: str | None = None) -> dict:
    """Shortcut para metas del reranker. `note` default al stem del filename
    (que es cómo el indexer computa `note_title` via `Path(file).stem`)."""
    from pathlib import Path
    return {"file": file, "note": note if note is not None else Path(file).stem}


def test_title_match_exact_filename():
    """Query idéntico al filename stem → 1.0."""
    assert rag._title_match_score("dev cycles", _meta("dev cycles.md")) == 1.0


def test_title_match_exact_title():
    """Query idéntico al title (note) → 1.0 aunque el filename sea distinto."""
    score = rag._title_match_score(
        "performance review",
        _meta("02-Areas/misc.md", note="Performance Review"),
    )
    assert score == 1.0


def test_title_match_query_subset_of_title():
    """Query "cycles" (1 token significativo) contra filename "dev cycles"
    (2 tokens) → todos los tokens del query matchean → 1.0 (coverage sobre
    query, no sobre title)."""
    assert rag._title_match_score("cycles", _meta("dev cycles.md")) == 1.0


def test_title_match_partial_coverage():
    """Query "cycle estamos" (2 tokens significativos, 1 matchea) → 0.5.
    "estamos" no está en el título, "cycle" sí (prefix match con "cycles")."""
    score = rag._title_match_score("cycle estamos", _meta("dev cycles.md"))
    assert score == pytest.approx(0.5, abs=0.01)


def test_title_match_prefix_handles_plurals():
    """Prefix matching de 4 chars: "cycle" y "cycles" deben matchear."""
    assert rag._title_match_score("cycle", _meta("dev cycles.md")) == 1.0
    assert rag._title_match_score("cycles", _meta("dev cycle.md")) == 1.0


def test_title_match_case_insensitive():
    """Matching es case-insensitive en ambos lados."""
    assert rag._title_match_score("DEV CYCLES", _meta("dev cycles.md")) == 1.0
    assert rag._title_match_score(
        "dev cycles", _meta("02-Areas/foo.md", note="DEV CYCLES"),
    ) == 1.0


@pytest.mark.parametrize("query", [
    "en qué",                # solo stopwords ES
    "what is the",           # solo stopwords EN
    "cómo cuándo dónde",     # solo WH stopwords
])
def test_title_match_stopword_only_query_returns_zero(query):
    """Query que son solo stopwords → no signal, 0.0. Evita que
    palabras funcionales inflen el boost artificialmente."""
    assert rag._title_match_score(query, _meta("dev cycles.md")) == 0.0


def test_title_match_spanish_stopwords_ignored():
    """Query "en qué cycle estamos" → tokens significativos = {cycle, estamos};
    matchea "cycle" → coverage 1/2 = 0.5. Las stopwords en/qué no cuentan."""
    score = rag._title_match_score(
        "en qué cycle estamos", _meta("dev cycles.md"),
    )
    assert score == pytest.approx(0.5, abs=0.01)


def test_title_match_english_stopwords_ignored():
    """Query "what is the cycle" → tokens = {cycle}; matchea → 1.0."""
    assert rag._title_match_score(
        "what is the cycle", _meta("dev cycles.md"),
    ) == 1.0


def test_title_match_no_overlap():
    """Query sin overlap con filename/title → 0.0."""
    assert rag._title_match_score(
        "xyz unrelated", _meta("dev cycles.md"),
    ) == 0.0


def test_title_match_filename_path_segments_matter():
    """Folder segments en el path NO deben contarse como parte del título
    (evita que `03-Resources/dev.md` matchee "resources")."""
    # "resources" está en el path pero NO en el filename stem ni en note_title
    assert rag._title_match_score(
        "resources", _meta("03-Resources/dev.md"),
    ) == 0.0


def test_title_match_filename_dashes_underscores_split():
    """Filenames con `-` o `_` deben tokenizarse como palabras separadas."""
    assert rag._title_match_score(
        "aws tagging",
        _meta("01-Projects/aws-tagging/_index.md", note="AWS Tagging"),
    ) == 1.0
    assert rag._title_match_score(
        "aws",
        _meta("01-Projects/finops_aws_report/Reporte.md", note="Reporte"),
    ) == 1.0


def test_title_match_handles_empty_meta():
    """Meta sin `file` ni `note` → 0.0, no crash."""
    assert rag._title_match_score("anything", {}) == 0.0
    assert rag._title_match_score("anything", {"file": "", "note": ""}) == 0.0


def test_title_match_handles_empty_query():
    """Query vacío → 0.0, no crash."""
    assert rag._title_match_score("", _meta("dev cycles.md")) == 0.0
    assert rag._title_match_score("   ", _meta("dev cycles.md")) == 0.0


# ── Weight defaults + RankerWeights shape ────────────────────────────────────


def test_ranker_weights_has_title_match_slot():
    """`title_match` está en __slots__ del dataclass."""
    assert "title_match" in rag.RankerWeights.__slots__


def test_ranker_weights_title_match_default_is_positive():
    """Default ≠ 0 (a diferencia de `recency_always` / `tag_literal` que
    arrancan en 0 esperando `rag tune`). Razón: title match es signal
    estructural, no aprendido — debe andar desde el día uno."""
    w = rag.RankerWeights()
    assert w.title_match > 0.0


def test_ranker_weights_title_match_loads_from_dict():
    """`from_dict` respeta title_match override."""
    w = rag.RankerWeights.from_dict({"title_match": 0.25})
    assert w.title_match == pytest.approx(0.25, abs=1e-6)


def test_ranker_weights_title_match_missing_falls_back_to_default():
    """Old ranker.json sin la key → default, sin crash."""
    default = rag.RankerWeights().title_match
    w = rag.RankerWeights.from_dict({})
    assert w.title_match == pytest.approx(default, abs=1e-6)


def test_weight_ranges_includes_title_match():
    """`_TUNE_SPACE` incluye `title_match` para que rag tune lo explore."""
    assert "title_match" in rag._TUNE_SPACE
    lo, hi = rag._TUNE_SPACE["title_match"]
    assert lo == 0.0
    assert hi > 0.0


# ── Integración en retrieve(): el boost se aplica y promueve la nota ─────────


class _FakeReranker:
    """Reranker mock que devuelve el mismo score para todos los pairs.
    Aísla el test de la señal semántica real — solo ejercitamos el boost."""
    def predict(self, pairs, show_progress_bar=False, **_):
        return [0.0] * len(pairs)


def test_retrieve_applies_title_match_boost_to_final_score(tmp_path, monkeypatch):
    """Caso medido: sin el boost, chunks con título match + body pobre
    perdían contra chunks con título sin match + body relevante. Con el
    boost, la nota titulada debería promoverse."""
    # Corpus sintético: 2 notas, misma body length, distinto título-match.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "embed", lambda ts: [[1.0, 0.0, 0.0, 0.0] for _ in ts])
    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())
    vault = tmp_path / "v"; vault.mkdir()
    col = rag.get_db_for(vault)

    col.add(
        ids=["titled::0", "body_only::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        documents=["short body", "short body sobre cycles del producto"],
        metadatas=[
            {
                "file": "03-Resources/dev cycles.md",
                "note": "dev cycles",
                "folder": "03-Resources",
                "tags": "", "outlinks": "", "hash": "x",
            },
            {
                "file": "01-Projects/algo.md",
                "note": "algo",
                "folder": "01-Projects",
                "tags": "", "outlinks": "", "hash": "y",
            },
        ],
    )
    rag._invalidate_corpus_cache()

    result = rag.retrieve(
        col, "dev cycles", k=2, folder=None,
        multi_query=False, auto_filter=False,
    )
    paths = [m["file"] for m in result["metas"]]
    # Con title_match boost, la nota con match en filename/title va primero,
    # aunque el reranker les dé el mismo score (0.0 para ambas).
    assert paths[0] == "03-Resources/dev cycles.md", (
        f"Expected title-match note first; got order: {paths}. "
        f"scores: {result['scores']}"
    )


def test_retrieve_without_title_match_weight_is_neutral(tmp_path, monkeypatch):
    """Con `title_match=0.0` explícito en RankerWeights, el BOOST EXPLÍCITO
    no aplica. El test compara la nota con título match vs la nota sin
    match: con el boost activo (default 0.15) la diferencia incluye una
    contribución de `coverage_query × 0.15`; con weight=0 esa
    contribución desaparece y la diferencia residual es solo la señal
    indirecta que BM25 aporta via RRF (filename match) — mucho menor.

    2026-05-01: el assert exacto `==` era flaky porque BM25 sigue
    aportando una señal lexical para "dev cycles" matcheando el
    filename `03-Resources/dev cycles.md` aunque el peso de
    `title_match` esté en 0. Con weight default (0.15) la nota titulada
    sale ~0.07-0.15 más alta; con weight=0 sale solo ~0.005-0.05 más
    alta (BM25 únicamente). Adjusted: assert que la diferencia con
    weight=0 es CHICA (≤0.06) — bound conservador que detecta una
    regresión donde el boost se aplique a pesar del weight=0, sin
    fallarse por la señal residual de BM25 que es indirecta.
    """
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "embed", lambda ts: [[1.0, 0.0, 0.0, 0.0] for _ in ts])
    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())
    # Forzamos weight a 0 para este test.
    zero_weights = rag.RankerWeights(title_match=0.0)
    monkeypatch.setattr(rag, "get_ranker_weights", lambda: zero_weights)

    vault = tmp_path / "v"; vault.mkdir()
    col = rag.get_db_for(vault)
    col.add(
        ids=["titled::0", "body_only::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        documents=["short body", "short body"],
        metadatas=[
            {
                "file": "03-Resources/dev cycles.md", "note": "dev cycles",
                "folder": "03-Resources", "tags": "", "outlinks": "", "hash": "x",
            },
            {
                "file": "01-Projects/algo.md", "note": "algo",
                "folder": "01-Projects", "tags": "", "outlinks": "", "hash": "y",
            },
        ],
    )
    rag._invalidate_corpus_cache()

    # Con weight=0 el boost explícito desaparece pero BM25 sigue aportando
    # señal indirecta. La diferencia residual debería ser pequeña (≤0.06).
    result = rag.retrieve(
        col, "dev cycles", k=2, folder=None,
        multi_query=False, auto_filter=False,
    )
    diff = abs(result["scores"][0] - result["scores"][1])
    assert diff <= 0.06, (
        f"Con title_match=0 la diferencia residual (BM25 lexical only) "
        f"debería ser ≤0.06; got {result['scores']} (diff={diff:.4f}). "
        f"Si esto crece, el boost de title_match se está aplicando "
        f"a pesar del weight=0."
    )
