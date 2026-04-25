"""Tests for the `gap` anticipatory signal.

Detecta clusters de queries recurrentes sin coverage en el vault. Todos los
inputs externos (SQL via `_scan_queries_log`, embeddings via
`_cluster_queries`, vault via `retrieve`) se mockean — los tests validan
la lógica de decisión (threshold de cluster size, cutoff de score de
retrieve, escala de score, stability del dedup_key) sin tocar disk ni red.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import rag
from rag_anticipate.signals.gap import gap_signal


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path y devuelve una sqlite-vec
    collection usable como retorno de `get_db` en los mocks.

    Espejo del fixture homónimo en `tests/test_anticipate_agent.py` — la
    idea es que cada test_ que necesite un `col` dummy pueda simplemente
    inyectar el fixture y mockear `get_db` retornándolo."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    from rag import SqliteVecClient
    client = SqliteVecClient(path=str(db_path))
    col = client.get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )
    with rag._ragvec_state_conn() as _conn:
        pass
    return col


# Helper: construye events en el shape que devuelve `_scan_queries_log`.
def _mk_events(queries: list[str]) -> list[dict]:
    return [{"q": q} for q in queries]


# ── 1. Sin queries en ventana → []  ──────────────────────────────────────────

def test_gap_no_queries_returns_empty(monkeypatch, state_db):
    monkeypatch.setattr(rag, "_scan_queries_log", lambda **kw: [])
    # _cluster_queries no debería ser llamado, pero lo stubeamos por las dudas.
    monkeypatch.setattr(rag, "_cluster_queries", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [], "scores": []})
    out = gap_signal(datetime.now())
    assert out == []


def test_gap_queries_all_too_short_returns_empty(monkeypatch, state_db):
    """Queries de <6 chars se filtran antes del clustering."""
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(["ok", "no", "?", "hi"]))
    called = {"clustered": False}

    def _cluster_spy(qs, **kw):
        called["clustered"] = True
        return []

    monkeypatch.setattr(rag, "_cluster_queries", _cluster_spy)
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [], "scores": []})
    out = gap_signal(datetime.now())
    assert out == []
    # Como no queda nada post-filtro, _cluster_queries ni se llama.
    assert called["clustered"] is False


# ── 2. Cluster <3 → []  ──────────────────────────────────────────────────────

def test_gap_cluster_below_min_size_returns_empty(monkeypatch, state_db):
    queries = ["como deployo a fly io", "fly io setup"]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    # Un único cluster con 2 elementos → no cumple ≥3.
    monkeypatch.setattr(rag, "_cluster_queries", lambda qs, **kw: [[0, 1]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    retrieve_called = {"n": 0}

    def _retrieve_spy(*a, **kw):
        retrieve_called["n"] += 1
        return {"metas": [], "scores": []}

    monkeypatch.setattr(rag, "retrieve", _retrieve_spy)
    out = gap_signal(datetime.now())
    assert out == []
    # No debería invocar retrieve — el cluster no pasa el size filter.
    assert retrieve_called["n"] == 0


# ── 3. Cluster ≥3 + score alto (cubierto) → []  ──────────────────────────────

def test_gap_cluster_with_coverage_returns_empty(monkeypatch, state_db):
    queries = [
        "como configuro caddy",
        "caddy tls internal",
        "configurar caddy reverse proxy",
    ]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [[0, 1, 2]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    # scores[0] = 0.75 >= 0.30 → HAY coverage → no es gap.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "02-Areas/Caddy.md", "note": "Caddy"}],
            "scores": [0.75],
        },
    )
    out = gap_signal(datetime.now())
    assert out == []


# ── 4. Cluster ≥3 + score bajo → emit candidate  ─────────────────────────────

def test_gap_cluster_without_coverage_emits_candidate(monkeypatch, state_db):
    queries = [
        "como conecto obsidian a spotify",
        "spotify plugin obsidian",
        "sincronizar tracks spotify obsidian",
        "obsidian spotify integration",
    ]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [[0, 1, 2, 3]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    # scores[0] = 0.15 < 0.30 → gap real.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "x.md", "note": "x"}],
            "scores": [0.15],
        },
    )
    out = gap_signal(datetime.now())
    assert len(out) == 1
    c = out[0]
    assert c.kind == "anticipate-gap"
    assert c.snooze_hours == 168
    assert c.dedup_key.startswith("gap:")
    # El representante es la query más corta del cluster.
    rep = min(queries, key=len)
    assert rep in c.message
    # Mensaje con el conteo y el hint de /capture.
    assert "4 veces" in c.message
    assert "14d" in c.message
    # Score = min(1.0, 4/10) = 0.4
    assert c.score == pytest.approx(0.4)


def test_gap_empty_retrieve_scores_is_a_gap(monkeypatch, state_db):
    """Si retrieve no devuelve scores (vault vacío / query raro), tratamos
    como gap — no hay cobertura por definición."""
    queries = [
        "tema exotico uno muy raro",
        "tema exotico dos muy raro",
        "tema exotico tres muy raro",
    ]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [[0, 1, 2]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [], "scores": []})
    out = gap_signal(datetime.now())
    assert len(out) == 1
    assert out[0].kind == "anticipate-gap"


# ── 5. Score escala con cluster size  ────────────────────────────────────────

def test_gap_score_scales_with_cluster_size(monkeypatch, state_db):
    """score = min(1.0, n/10). Para n=3 → 0.3; n=7 → 0.7; n=15 → 1.0."""
    def _run_for_n(n: int):
        queries = [f"query numero {i:03d} larga" for i in range(n)]
        monkeypatch.setattr(rag, "_scan_queries_log",
                            lambda **kw: _mk_events(queries))
        monkeypatch.setattr(rag, "_cluster_queries",
                            lambda qs, **kw: [list(range(n))])
        monkeypatch.setattr(rag, "get_db", lambda: state_db)
        monkeypatch.setattr(rag, "retrieve",
                            lambda *a, **kw: {"metas": [], "scores": [0.10]})
        return gap_signal(datetime.now())

    out3 = _run_for_n(3)
    out7 = _run_for_n(7)
    out15 = _run_for_n(15)
    assert out3[0].score == pytest.approx(0.3)
    assert out7[0].score == pytest.approx(0.7)
    assert out15[0].score == pytest.approx(1.0)
    # Monotonía
    assert out3[0].score < out7[0].score <= out15[0].score


# ── 6. dedup_key estable entre runs ──────────────────────────────────────────

def test_gap_dedup_key_stable_across_calls(monkeypatch, state_db):
    queries = [
        "es el mismo tema recurrente de siempre",
        "pregunta sobre el tema recurrente",
        "el mismo tema que pregunto siempre",
    ]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [[0, 1, 2]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [], "scores": [0.05]})

    out1 = gap_signal(datetime.now())
    out2 = gap_signal(datetime.now())
    assert out1[0].dedup_key == out2[0].dedup_key
    # Y el dedup_key depende del representante (la query más corta).
    rep = min(queries, key=len)
    import hashlib
    expected_suffix = hashlib.sha256(rep.encode("utf-8")).hexdigest()[:12]
    assert out1[0].dedup_key == f"gap:{expected_suffix}"


# ── 7. Varios clusters → pick el más grande con gap ──────────────────────────

def test_gap_picks_largest_cluster_with_gap(monkeypatch, state_db):
    """Tres clusters: [size=3 cubierto], [size=6 gap], [size=4 gap]. Debe
    elegir el de size=6 (el más grande con gap)."""
    # Distribuimos las queries por posición — los índices son lo que importa.
    queries = [
        # cluster A: 3 items, cubierto (rep = "aaa corto cubierto")
        "aaa corto cubierto",
        "aaa cubierto pregunta dos",
        "aaa cubierto pregunta tres",
        # cluster B: 6 items, gap (rep = "bbb corto gap")
        "bbb corto gap",
        "bbb gap pregunta numero dos larga",
        "bbb gap pregunta numero tres larga",
        "bbb gap pregunta numero cuatro larga",
        "bbb gap pregunta numero cinco larga",
        "bbb gap pregunta numero seis larga",
        # cluster C: 4 items, gap (rep = "ccc corto gap")
        "ccc corto gap",
        "ccc gap pregunta numero dos larga",
        "ccc gap pregunta numero tres larga",
        "ccc gap pregunta numero cuatro larga",
    ]
    a_idx = [0, 1, 2]
    b_idx = [3, 4, 5, 6, 7, 8]
    c_idx = [9, 10, 11, 12]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [a_idx, b_idx, c_idx])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)

    # retrieve devuelve score alto (0.80) solo para el rep de A, bajo para B y C.
    def _retrieve(col, q, k, **kw):
        if "aaa" in q:
            return {"metas": [{"file": "a.md"}], "scores": [0.80]}
        return {"metas": [{"file": "x.md"}], "scores": [0.10]}

    monkeypatch.setattr(rag, "retrieve", _retrieve)
    out = gap_signal(datetime.now())
    assert len(out) == 1
    c = out[0]
    assert c.kind == "anticipate-gap"
    # El rep del cluster B = "bbb corto gap" (query más corta del cluster).
    assert "bbb corto gap" in c.message
    # Score para cluster de 6 = 0.6.
    assert c.score == pytest.approx(0.6)


# ── 8. Silent-fail si algo explota ───────────────────────────────────────────

def test_gap_scan_queries_exception_returns_empty(monkeypatch, state_db):
    """El contract de las signals obliga silent-fail: retornar [] si crashe
    cualquier input externo."""
    def _boom(**kw):
        raise RuntimeError("sql down")

    monkeypatch.setattr(rag, "_scan_queries_log", _boom)
    out = gap_signal(datetime.now())
    assert out == []


def test_gap_retrieve_exception_skips_cluster(monkeypatch, state_db):
    """Si `retrieve` falla en un cluster, pasamos al siguiente en lugar de
    tumbar la signal entera."""
    queries = [
        # cluster A (3): retrieve crashea
        "aaa primera pregunta larga",
        "aaa segunda pregunta larga",
        "aaa tercera pregunta larga",
        # cluster B (3): retrieve ok, gap
        "bbb primera pregunta larga",
        "bbb segunda pregunta larga",
        "bbb tercera pregunta larga",
    ]
    monkeypatch.setattr(rag, "_scan_queries_log",
                        lambda **kw: _mk_events(queries))
    monkeypatch.setattr(rag, "_cluster_queries",
                        lambda qs, **kw: [[0, 1, 2], [3, 4, 5]])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)

    def _retrieve(col, q, k, **kw):
        if "aaa" in q:
            raise RuntimeError("retrieve crashed on A")
        return {"metas": [{"file": "x.md"}], "scores": [0.05]}

    monkeypatch.setattr(rag, "retrieve", _retrieve)
    out = gap_signal(datetime.now())
    assert len(out) == 1
    assert "bbb" in out[0].message
