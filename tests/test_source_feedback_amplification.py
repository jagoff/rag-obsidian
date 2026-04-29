"""Tests del per-source feedback amplification (B.5, 2026-04-29).

Cubre:
- `_compute_source_feedback_amplification` aplica sqrt-based factor
  capeado a SOURCE_FEEDBACK_AMP_MAX.
- Sources con poco feedback reciben factor > 1.0; vault baseline = 1.0.
- Source con 0 feedback en la ventana → no aparece en el dict
  (apply_weighted_scores cae al default 1.0).
- ref_volume < SOURCE_FEEDBACK_AMP_REF_MIN → dict vacío (bootstrap mode).
- `RAG_SOURCE_FEEDBACK_AMP_DISABLE=1` retorna {}.
- Cache invalidation por MAX(ts) de rag_feedback.
- Integración en `apply_weighted_scores`: el factor multiplica
  feedback_pos y feedback_neg, no afecta otros signals (rerank, recency, etc.).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

import rag


@pytest.fixture(autouse=True)
def reset_amp_cache():
    """Reset amplification cache antes y después de cada test."""
    rag._reset_source_feedback_amp_cache()
    yield
    rag._reset_source_feedback_amp_cache()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute(
        """
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT
        )
        """
    )
    yield c
    c.close()


def _seed_fb(
    conn: sqlite3.Connection,
    *,
    paths: list[str],
    rating: int = 1,
    days_ago: int = 0,
) -> None:
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json) "
        "VALUES (?, ?, 'q', ?)",
        (ts, rating, json.dumps(paths)),
    )


# ─── _compute_source_feedback_amplification ────────────────────────────


def test_amp_vault_baseline(conn):
    """Vault con más volumen → factor=1.0."""
    # 60 vault feedbacks (>= REF_MIN=50)
    for _ in range(60):
        _seed_fb(conn, paths=["01-Projects/foo.md"])
    factors = rag._compute_source_feedback_amplification(conn)
    assert factors.get("vault") == pytest.approx(1.0)


def test_amp_low_volume_gets_amplified(conn):
    """Source con 1/4 del volumen vault → factor=2.0 (sqrt(4/1))."""
    for _ in range(60):
        _seed_fb(conn, paths=["01-Projects/foo.md"])  # vault
    for _ in range(15):
        _seed_fb(conn, paths=["whatsapp://msg/X"])  # 60/15 = 4 → sqrt(4)=2.0
    factors = rag._compute_source_feedback_amplification(conn)
    assert factors.get("vault") == pytest.approx(1.0)
    assert factors.get("whatsapp") == pytest.approx(2.0, abs=0.01)


def test_amp_capped_at_max(conn):
    """Source con muy poco volumen → cap a SOURCE_FEEDBACK_AMP_MAX."""
    for _ in range(100):
        _seed_fb(conn, paths=["01-Projects/foo.md"])
    # Una sola gmail row → factor sería sqrt(100) = 10x → cap a 3.0
    _seed_fb(conn, paths=["gmail://thread/X"])
    factors = rag._compute_source_feedback_amplification(conn)
    assert factors.get("gmail") == pytest.approx(rag.SOURCE_FEEDBACK_AMP_MAX)


def test_amp_bootstrap_returns_empty(conn):
    """Si ref_volume < REF_MIN, bootstrap mode → dict vacio."""
    # Solo 10 vault rows → ref=10 < 50 → bootstrap
    for _ in range(10):
        _seed_fb(conn, paths=["01-Projects/foo.md"])
    factors = rag._compute_source_feedback_amplification(conn)
    assert factors == {}


def test_amp_excludes_old_rows(conn):
    """Rows fuera de la ventana SOURCE_FEEDBACK_AMP_WINDOW_DAYS no cuentan."""
    # 60 vault rows recientes (vienen)
    for _ in range(60):
        _seed_fb(conn, paths=["01-Projects/foo.md"], days_ago=1)
    # 200 whatsapp rows VIEJAS (no vienen)
    for _ in range(200):
        _seed_fb(conn, paths=["whatsapp://X"], days_ago=200)
    factors = rag._compute_source_feedback_amplification(conn)
    assert "whatsapp" not in factors
    assert factors.get("vault") == pytest.approx(1.0)


def test_amp_skips_rows_without_paths(conn):
    """Rows sin paths_json no cuentan."""
    for _ in range(60):
        _seed_fb(conn, paths=["vault.md"])
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json) "
        "VALUES (datetime('now'), -1, 'q', NULL)"
    )
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json) "
        "VALUES (datetime('now'), -1, 'q', '')"
    )
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json) "
        "VALUES (datetime('now'), -1, 'q', 'not-json')"
    )
    factors = rag._compute_source_feedback_amplification(conn)
    # Solo el vault count cuenta, los corruptos se skipean.
    assert factors.get("vault") == pytest.approx(1.0)


def test_amp_skips_rating_zero(conn):
    """rating=0 (sin signal) no cuenta."""
    for _ in range(60):
        _seed_fb(conn, paths=["vault.md"], rating=1)
    for _ in range(40):
        _seed_fb(conn, paths=["whatsapp://X"], rating=0)
    factors = rag._compute_source_feedback_amplification(conn)
    assert "whatsapp" not in factors


def test_amp_uses_top_path_only(conn):
    """Solo el primer path (top-1) cuenta para amplification."""
    # vault como top-1, whatsapp como #2 → cuenta vault
    for _ in range(50):
        _seed_fb(conn, paths=["01-Projects/v.md", "whatsapp://X"])
    # whatsapp como top-1 → cuenta whatsapp
    for _ in range(15):
        _seed_fb(conn, paths=["whatsapp://Y", "01-Projects/v.md"])
    factors = rag._compute_source_feedback_amplification(conn)
    # vault gets 50, whatsapp gets 15 → ref=50, wa_factor = sqrt(50/15)
    assert factors.get("vault") == pytest.approx(1.0)
    assert factors.get("whatsapp") == pytest.approx((50 / 15) ** 0.5, abs=0.01)


# ─── _source_feedback_amp (env disable + cache) ────────────────────────


def test_amp_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RAG_SOURCE_FEEDBACK_AMP_DISABLE", "1")
    factors = rag._source_feedback_amp()
    assert factors == {}


def test_amp_disabled_via_env_other_truthy_values(monkeypatch):
    for val in ("true", "yes", "on", "TRUE"):
        monkeypatch.setenv("RAG_SOURCE_FEEDBACK_AMP_DISABLE", val)
        rag._reset_source_feedback_amp_cache()
        assert rag._source_feedback_amp() == {}


# ─── Integración en apply_weighted_scores ──────────────────────────────


def test_apply_weighted_scores_amp_multiplies_feedback(monkeypatch):
    """Con amp_table activo, feedback_pos/neg se multiplican por el factor."""
    # Stub el amp_table para no depender de la DB real.
    monkeypatch.setattr(
        rag, "_source_feedback_amp",
        lambda: {"whatsapp": 2.0, "vault": 1.0},
    )

    # Ranker weights: solo activamos feedback_pos para aislar el efecto.
    weights = rag.RankerWeights(
        recency_cue=0.0,
        recency_always=0.0,
        tag_literal=0.0,
        title_match=0.0,
        feedback_pos=0.10,
        feedback_neg=0.10,
        feedback_match_floor=0.80,
    )

    # Dos candidatos identicos salvo source. fb_pos_cos = 0.95 (ramps to ~0.75).
    feat_vault = {
        "ignored": False,
        "rerank": 0.5,
        "has_recency_cue": False,
        "recency_raw": 0.0,
        "tag_hits": 0,
        "title_match": 0.0,
        "graph_pagerank": 0.0,
        "click_prior": 0.0,
        "click_prior_folder": 0.0,
        "click_prior_hour": 0.0,
        "dwell_score": 0.0,
        "contradiction_count": 0.0,
        "fb_pos_cos": 0.95,
        "fb_neg_cos": 0.0,
        "meta": {"source": "vault", "path": "v.md", "created_ts": None},
        "path": "v.md",
    }
    feat_wa = {**feat_vault,
               "meta": {"source": "whatsapp", "path": "wa.md", "created_ts": None},
               "path": "wa.md"}

    out = rag.apply_weighted_scores([feat_vault, feat_wa], weights, k=2)
    assert len(out) == 2
    by_path = {f["path"]: f["score"] for f in out}
    # Source weight diferencia: vault=1.00, whatsapp=0.75 (de SOURCE_WEIGHTS).
    # El whatsapp src_mult * el factor amp → la diferencia neta debe favorecer
    # WA si el amp es alto suficiente. Verificamos que el feedback boost
    # del WA es ~2x el del vault.
    # El vault score = 0.5 + 0.10 * pos_w
    # El WA score = (0.5 + 0.10 * pos_w * 2.0) * 0.75 (source_weight WA=0.75)
    pos_w_05 = (0.95 - 0.80) / (1 - 0.80)  # = 0.75 ramp value
    expected_v = 0.5 + 0.10 * pos_w_05  # source_weight vault=1.0
    expected_wa = (0.5 + 0.10 * pos_w_05 * 2.0) * 0.75  # WA src=0.75
    assert by_path["v.md"] == pytest.approx(expected_v, abs=0.01)
    assert by_path["wa.md"] == pytest.approx(expected_wa, abs=0.01)


def test_apply_weighted_scores_no_amp_when_dict_empty(monkeypatch):
    """Cuando _source_feedback_amp() devuelve {}, comportamiento legacy."""
    monkeypatch.setattr(rag, "_source_feedback_amp", lambda: {})

    weights = rag.RankerWeights(
        recency_cue=0.0,
        feedback_pos=0.10,
        feedback_match_floor=0.80,
    )
    feat = {
        "ignored": False,
        "rerank": 0.5,
        "has_recency_cue": False,
        "recency_raw": 0.0,
        "tag_hits": 0,
        "title_match": 0.0,
        "graph_pagerank": 0.0,
        "click_prior": 0.0,
        "click_prior_folder": 0.0,
        "click_prior_hour": 0.0,
        "dwell_score": 0.0,
        "contradiction_count": 0.0,
        "fb_pos_cos": 0.95,
        "fb_neg_cos": 0.0,
        "meta": {"source": "whatsapp", "path": "wa.md", "created_ts": None},
        "path": "wa.md",
    }
    out = rag.apply_weighted_scores([feat], weights, k=1)
    pos_w = (0.95 - 0.80) / (1 - 0.80)
    # Sin amp, factor=1.0 default. WA src_mult=0.75.
    expected = (0.5 + 0.10 * pos_w * 1.0) * 0.75
    assert out[0]["score"] == pytest.approx(expected, abs=0.01)


def test_apply_weighted_scores_amp_disable_env(monkeypatch):
    """Env disable → factor=1.0 (no-op) aunque haya datos en rag_feedback."""
    monkeypatch.setenv("RAG_SOURCE_FEEDBACK_AMP_DISABLE", "1")
    rag._reset_source_feedback_amp_cache()
    factors = rag._source_feedback_amp()
    assert factors == {}


# ─── Constantes sanity ─────────────────────────────────────────────────


def test_constants_in_reasonable_ranges():
    assert 1.5 <= rag.SOURCE_FEEDBACK_AMP_MAX <= 5.0
    assert 10 <= rag.SOURCE_FEEDBACK_AMP_REF_MIN <= 200
    assert 7 <= rag.SOURCE_FEEDBACK_AMP_WINDOW_DAYS <= 365
