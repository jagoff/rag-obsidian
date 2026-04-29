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
    # TTL gate (audit 2026-04-29): reset el timestamp así cada test
    # arranca con TTL "expirado" — los tests que no testean el TTL
    # explícitamente esperan SQL hits cada call, no cache hits.
    monkeypatch.setattr(rag, "_behavior_priors_loaded_ts", 0.0)


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
    # TTL gate (audit 2026-04-29): reset el timestamp así cada test
    # arranca con TTL "expirado".
    monkeypatch.setattr(rag, "_feedback_golden_loaded_ts", 0.0)


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


# ── TTL memoization (Task B 2026-04-29) ──────────────────────────────────────
#
# Tests para el TTL gate de 60s de los SQL readers de retrieve():
#   - _load_behavior_priors() y load_feedback_golden().
# Bajo carga del web server (5-20 retrieve()/s) cada call pegaba SQL.
# Con TTL, calls dentro del window devuelven cache directo sin tocar
# DB. SQL se pega solo cuando TTL expiró → reduce contención WAL
# proporcionalmente.


def test_behavior_priors_ttl_gate_skips_sql_within_window(
    monkeypatch, reset_behavior_cache,
):
    """Dos calls dentro del TTL window: solo la PRIMERA pega SQL, la segunda
    devuelve cache directo sin abrir conexión."""
    sql_opens = {"n": 0}

    class _DummyConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, *a, **kw):
            class _Cur:
                def fetchall(self): return []
                def fetchone(self): return None
            return _Cur()
        @property
        def row_factory(self): return None
        @row_factory.setter
        def row_factory(self, v): pass

    def counting_open(*a, **kw):
        sql_opens["n"] += 1
        return _DummyConn()

    # Mock _sql_max_ts para que devuelva None (DB vacía) — fuerza el "empty"
    # path que es el más rápido y no requiere mocks complejos del rebuild.
    monkeypatch.setattr(rag, "_ragvec_state_conn", counting_open)
    monkeypatch.setattr(rag, "_sql_max_ts", lambda *a, **kw: None)

    # Primera call: pega SQL, llena cache.
    r1 = rag._load_behavior_priors()
    assert sql_opens["n"] == 1
    assert r1 is not None

    # Segunda call inmediata: TTL gate hit, NO pega SQL.
    r2 = rag._load_behavior_priors()
    assert sql_opens["n"] == 1, (
        f"expected SQL hit count UNCHANGED on cache hit; got {sql_opens['n']}"
    )
    assert r2 is r1, "cached snapshot debe ser el mismo objeto (identity)"


def test_behavior_priors_ttl_gate_expires_after_window(
    monkeypatch, reset_behavior_cache,
):
    """Tras TTL_S segundos sin tocar el cache, la próxima call pega SQL
    de nuevo. Sin esto el cache nunca se refrescaría."""
    sql_opens = {"n": 0}

    class _DummyConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, *a, **kw):
            class _Cur:
                def fetchall(self): return []
                def fetchone(self): return None
            return _Cur()
        @property
        def row_factory(self): return None
        @row_factory.setter
        def row_factory(self, v): pass

    def counting_open(*a, **kw):
        sql_opens["n"] += 1
        return _DummyConn()

    monkeypatch.setattr(rag, "_ragvec_state_conn", counting_open)
    monkeypatch.setattr(rag, "_sql_max_ts", lambda *a, **kw: None)

    # Primera call: SQL hit.
    rag._load_behavior_priors()
    assert sql_opens["n"] == 1

    # Forzar expiración del TTL: bajar loaded_ts a hace TTL+10s.
    monkeypatch.setattr(
        rag, "_behavior_priors_loaded_ts",
        rag._behavior_priors_loaded_ts - rag._BEHAVIOR_PRIORS_TTL_S - 10.0,
    )

    # Segunda call tras "expirar" el TTL: pega SQL de nuevo.
    rag._load_behavior_priors()
    assert sql_opens["n"] == 2, (
        f"expected SQL hit on TTL expiration; got {sql_opens['n']}"
    )


def test_behavior_priors_ttl_does_not_bump_on_sql_failure(
    monkeypatch, reset_behavior_cache,
):
    """Si el SQL read falla (retry exhausto), NO bumpeamos loaded_ts —
    así la próxima call reintenta SQL inmediatamente. Sin esto, un
    fail transitorio congelaría el ranker sin priors por 60s.

    Coopera con el invariante #3 de CLAUDE.md (stale-cache fallback):
    devolvemos el cache previo si existe, pero NO marcamos el read
    como exitoso."""
    import sqlite3

    # Pre-cargar cache stale + loaded_ts viejo (simulando que ya hubo
    # un read exitoso hace mucho).
    stale = {"click_prior": {"x.md": 0.5}, "n_events": 10, "hash": "sql:old"}
    monkeypatch.setattr(rag, "_behavior_priors_cache", stale)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", "old_ts")
    # Forzar TTL expirado para que el gate NO corte; la call debe
    # intentar SQL → fallar → devolver stale + NO bumpear loaded_ts.
    import time as _time
    very_old = _time.monotonic() - 1000.0
    monkeypatch.setattr(rag, "_behavior_priors_loaded_ts", very_old)

    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):
        result = rag._load_behavior_priors()

    # Devuelve stale.
    assert result is stale
    # loaded_ts SIGUE siendo el viejo (no se bumpeó).
    assert rag._behavior_priors_loaded_ts == very_old, (
        "loaded_ts NO debe bumpearse en error path — sino congela 60s"
    )


def test_feedback_golden_ttl_gate_skips_sql_within_window(
    monkeypatch, reset_feedback_memo,
):
    """Dos calls dentro del TTL window: solo la PRIMERA pega SQL."""
    sql_opens = {"n": 0}

    class _DummyConn:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, *a, **kw):
            class _Cur:
                def fetchall(self): return []
                def fetchone(self): return None
            return _Cur()

    def counting_open(*a, **kw):
        sql_opens["n"] += 1
        return _DummyConn()

    # Pre-cargar memo válido + loaded_ts reciente para que el TTL gate
    # corte. (Sin pre-load, el gate ve memo=None y va a SQL.)
    real_memo = {"positives": [{"path": "x.md", "emb": [0.1]}], "negatives": []}
    monkeypatch.setattr(rag, "_feedback_golden_memo", real_memo)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", "ts1")
    import time as _time
    monkeypatch.setattr(rag, "_feedback_golden_loaded_ts", _time.monotonic())

    monkeypatch.setattr(rag, "_ragvec_state_conn", counting_open)

    # Call 1: TTL gate hit, devuelve memo SIN tocar SQL.
    r1 = rag.load_feedback_golden()
    assert sql_opens["n"] == 0, (
        f"TTL hit debe skipear SQL; got {sql_opens['n']} opens"
    )
    assert r1 is real_memo


def test_feedback_golden_ttl_does_not_bump_on_sql_failure(
    monkeypatch, reset_feedback_memo,
):
    """Si SQL read falla, NO bumpeamos loaded_ts → próxima call
    reintenta SQL inmediatamente."""
    import sqlite3

    real_memo = {"positives": [{"path": "x.md", "emb": [0.1]}], "negatives": []}
    monkeypatch.setattr(rag, "_feedback_golden_memo", real_memo)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", "ts1")
    # TTL expirado para que el gate no corte.
    import time as _time
    very_old = _time.monotonic() - 1000.0
    monkeypatch.setattr(rag, "_feedback_golden_loaded_ts", very_old)

    def fail_open(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(rag, "_ragvec_state_conn", fail_open), \
         patch("time.sleep"):
        result = rag.load_feedback_golden()

    # Devuelve el memo real (stale-cache fallback).
    assert result is real_memo
    # loaded_ts NO bumpeado.
    assert rag._feedback_golden_loaded_ts == very_old, (
        "loaded_ts NO debe bumpearse en error path"
    )


def test_feedback_golden_ttl_concurrent_access_safe(
    monkeypatch, reset_feedback_memo,
):
    """Multiple threads concurrentes leyendo: el lock garantiza que el
    state (memo + loaded_ts + source_ts) queda consistente."""
    import threading

    real_memo = {"positives": [{"path": "x.md", "emb": [0.1]}], "negatives": []}
    monkeypatch.setattr(rag, "_feedback_golden_memo", real_memo)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", "ts1")
    import time as _time
    monkeypatch.setattr(rag, "_feedback_golden_loaded_ts", _time.monotonic())

    # No mock del SQL — el TTL gate debe cortar antes del SQL.
    results = []
    errors = []

    def worker():
        try:
            results.append(rag.load_feedback_golden())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"concurrent calls raised: {errors}"
    assert len(results) == 10
    # Todos los results son el mismo memo (TTL gate hit en todos).
    assert all(r is real_memo for r in results)


def test_behavior_priors_ttl_concurrent_access_safe(
    monkeypatch, reset_behavior_cache,
):
    """Idem para behavior priors."""
    import threading

    stale = {"click_prior": {"x.md": 0.5}, "n_events": 10, "hash": "sql:fresh"}
    monkeypatch.setattr(rag, "_behavior_priors_cache", stale)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", "ts1")
    import time as _time
    monkeypatch.setattr(rag, "_behavior_priors_loaded_ts", _time.monotonic())

    results = []
    errors = []

    def worker():
        try:
            results.append(rag._load_behavior_priors())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"concurrent calls raised: {errors}"
    assert len(results) == 10
    assert all(r is stale for r in results)
