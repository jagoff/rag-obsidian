"""Tests para el retry + stale-cache fallback de los SQL readers críticos.

### El bug que estos tests cierran

Audit 2026-04-24: `sql_state_errors.jsonl` mostró:

  211 feedback_golden_sql_read_failed
  177 behavior_priors_sql_read_failed

Esos son **readers** (SELECT) que fallaron con `database is locked` u
otra OperationalError transient. Pre-fix:

- `_load_behavior_priors`: try/except bare sin retry → caía al primer
  intento, devolvía empty snapshot, ranker-vivo perdía priors.
- `load_feedback_golden`: tenía retry (5 attempts) PERO al exhausto
  el path "error" devolvía un payload empty que sobrescribía el memo
  previo en `_feedback_golden_memo`. La próxima call veía
  `_feedback_golden_memo is not None` y devolvía el empty cacheado
  como si fuera memo hit válido — feedback signal quedaba envenenado
  hasta que rag_feedback recibiera una entry nueva.

### Fix verificado por estos tests

- Retry attempts: 8 (match writers) en vez de 5.
- En retry exhausto: devolver el cache previo SIN tocarlo.
- En fresh process sin cache: empty fallback (mantiene retrieval funcional).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

import rag


# ── _load_behavior_priors ────────────────────────────────────────────────────


@pytest.fixture
def reset_behavior_cache(monkeypatch):
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)


def test_behavior_priors_returns_stale_cache_on_sql_retry_exhausted(
    monkeypatch, reset_behavior_cache,
):
    """Cuando todos los retries fallan, devolvemos el cache previo en vez
    de empty. Stale priors >> sin priors para el ranker."""
    # Pre-cargar un cache "stale".
    stale = {"click_prior": {"vault/x.md": 0.5}, "n_events": 100, "hash": "sql:stale"}
    monkeypatch.setattr(rag, "_behavior_priors_cache", stale)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", "old_ts")

    # Forzar fail persistente del SQL read.
    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):  # acelera el retry backoff
        result = rag._load_behavior_priors()

    assert result is stale, (
        "expected stale cache returned (object identity) — got fresh empty"
    )
    # El hash del stale cache se preserva — confirma que NO se reseteó.
    assert result["hash"] == "sql:stale"


def test_behavior_priors_returns_empty_on_fresh_process_with_db_locked(
    monkeypatch, reset_behavior_cache,
):
    """Fresh process (cache=None) + DB locked → fallback a empty snapshot.
    Mantiene retrieval funcional aunque sin priors."""
    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):
        result = rag._load_behavior_priors()

    # Empty snapshot shape contract.
    assert result["hash"] == "sql:error"
    assert result["click_prior"] == {} or "click_prior" in result
    # n_events es 0 cuando viene de _compute_behavior_priors_from_rows([]).
    assert result.get("n_events", 0) == 0


def test_behavior_priors_uses_8_retry_attempts(
    monkeypatch, reset_behavior_cache,
):
    """Audit bump 5→8: el retry debe intentar 8 veces antes de degradar."""
    attempts = {"n": 0}

    def counting_open(*a, **kw):
        attempts["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", counting_open), \
         patch("time.sleep"):
        rag._load_behavior_priors()

    assert attempts["n"] == 8, (
        f"expected 8 retry attempts (post audit 2026-04-24), got {attempts['n']}"
    )


# ── load_feedback_golden ─────────────────────────────────────────────────────


@pytest.fixture
def reset_feedback_memo(monkeypatch):
    monkeypatch.setattr(rag, "_feedback_golden_memo", None)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", "")


def test_feedback_golden_does_not_poison_memo_on_error(
    monkeypatch, reset_feedback_memo,
):
    """**El bug clásico de audit**: pre-fix, retry exhausto sobrescribía
    `_feedback_golden_memo` con `{positives:[], negatives:[]}` (default
    del retry), envenenando el cache. Próxima call → "memo hit" devolvía
    empty hasta que rag_feedback recibiera nueva entry y triggereara
    rebuild. Post-fix: en error, NO tocamos el memo."""
    # Pre-cargar un memo válido con feedback real.
    real_memo = {
        "positives": [{"path": "Coaching/Ikigai.md", "embedding": [0.1] * 8}],
        "negatives": [],
    }
    monkeypatch.setattr(rag, "_feedback_golden_memo", real_memo)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", "2026-04-24T10:00:00")

    # Forzar fail del SQL read.
    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):
        result = rag.load_feedback_golden()

    # El memo previo se devuelve SIN modificar.
    assert result is real_memo, (
        "expected previous memo returned on error (preserves feedback signal)"
    )
    assert rag._feedback_golden_memo is real_memo, (
        "memo NO debe ser sobrescrito con el empty default del retry"
    )
    # source_ts también queda intacto.
    assert rag._feedback_golden_source_ts_sql == "2026-04-24T10:00:00"


def test_feedback_golden_empty_on_fresh_process_with_db_locked(
    monkeypatch, reset_feedback_memo,
):
    """Fresh process sin memo + DB locked → empty snapshot fallback."""
    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):
        result = rag.load_feedback_golden()

    assert result == {"positives": [], "negatives": []}


def test_feedback_golden_uses_8_retry_attempts(
    monkeypatch, reset_feedback_memo,
):
    """Audit bump 5→8."""
    attempts = {"n": 0}

    def counting_open(*a, **kw):
        attempts["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", counting_open), \
         patch("time.sleep"):
        rag.load_feedback_golden()

    assert attempts["n"] == 8, (
        f"expected 8 retry attempts (post audit 2026-04-24), got {attempts['n']}"
    )


def test_non_transient_error_short_circuits_retry(
    monkeypatch, reset_behavior_cache,
):
    """Errores no-transient (ej. tabla no existe, schema drift) NO se
    reintentan — el _is_transient_sql_error filtra. Esto valida que el
    retry no esconde bugs de schema bajo retries inútiles."""
    attempts = {"n": 0}

    def schema_drift(*a, **kw):
        attempts["n"] += 1
        raise sqlite3.OperationalError("no such table: rag_behavior")

    with patch.object(rag, "_ragvec_state_conn", schema_drift), \
         patch("time.sleep"):
        result = rag._load_behavior_priors()

    # Solo 1 intento — fail-fast en schema drift.
    assert attempts["n"] == 1, (
        f"expected 1 attempt for non-transient error, got {attempts['n']}"
    )
    # Y degrada limpiamente al empty fallback.
    assert result["hash"] == "sql:error"
