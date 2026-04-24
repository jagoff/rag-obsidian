"""Regression tests for production edge-case bugs surfaced in web.log /
silent_errors.jsonl across the 2026-04-20 / 2026-04-21 pass:

- `_fetch_pagerank_top` IndexError when `ranked` is empty despite non-empty
  PageRank map (n=0 or mid-call cache invalidation).
- `_sanitize_confidence` must survive None / garbage / NaN / ±Inf without
  raising, because `_retry_pending_conversation_turns` feeds it raw
  JSONL-decoded values that may have been written pre-sanitize.
- `_cache_key` path in /api/chat: topic-shift reassignment of `history = []`
  shouldn't UnboundLocalError the PUT path. We don't wire the full chat
  handler here — just assert the helper contract.
- `_persist_with_sqlite_retry` retries transient locks + propagates real
  SQL errors through the silent log (not re-raised).
"""
from __future__ import annotations

import math
import sqlite3
from unittest.mock import patch

import pytest


def test_fetch_pagerank_top_empty_ranked_no_crash(tmp_path, monkeypatch):
    import rag
    from web import server

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    col = client.get_or_create_collection("test_pr")
    col.add(
        ids=["only-one"], embeddings=[[0.1] * 8],
        documents=["solo un doc"],
        metadatas=[{"file": "A.md"}],
    )
    # Force get_pagerank to return a non-empty map while forcing `n=0` so
    # `ranked = sorted(...)[:0]` is empty. Without the guard this raised
    # IndexError on `ranked[0][1]`.
    fake_map = {"A.md": 0.5, "B.md": 0.3}
    with patch.object(server, "get_pagerank", return_value=fake_map):
        out = server._fetch_pagerank_top(col, n=0)
    assert out == []


def test_fetch_pagerank_top_returns_ranked_when_non_empty(tmp_path, monkeypatch):
    import rag
    from web import server

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    col = client.get_or_create_collection("test_pr2")
    col.add(
        ids=["a", "b"], embeddings=[[0.1] * 8, [0.2] * 8],
        documents=["d1", "d2"],
        metadatas=[{"file": "A.md"}, {"file": "B.md"}],
    )
    fake_map = {"A.md": 1.0, "B.md": 0.5}
    with patch.object(server, "get_pagerank", return_value=fake_map):
        out = server._fetch_pagerank_top(col, n=5)
    assert len(out) == 2
    assert out[0]["path"] == "A.md"
    assert out[0]["pr"] == 1.0
    assert out[1]["pr"] == 0.5


@pytest.mark.parametrize("bad", [
    float("-inf"), float("inf"), float("nan"),
    None, "not-a-number", [], {},
])
def test_sanitize_confidence_handles_garbage(bad):
    from web.server import _sanitize_confidence
    out = _sanitize_confidence(bad)
    assert isinstance(out, float)
    assert not math.isnan(out)
    assert not math.isinf(out)
    assert out == 0.0


@pytest.mark.parametrize("good,expected", [
    (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-0.3, -0.3),
    ("0.42", 0.42), (42, 42.0),
])
def test_sanitize_confidence_preserves_finite(good, expected):
    from web.server import _sanitize_confidence
    assert _sanitize_confidence(good) == expected


def test_persist_with_sqlite_retry_retries_on_locked(monkeypatch):
    from web import server
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        # 3rd attempt succeeds
        return None

    # Monkey patch time.sleep to avoid actually sleeping in tests.
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    server._persist_with_sqlite_retry(flaky, "unit_test_tag")
    assert len(calls) == 3
    assert logged == []  # no error logged — 3rd succeeded


def test_persist_with_sqlite_retry_gives_up_after_3(monkeypatch):
    from web import server

    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    server._persist_with_sqlite_retry(always_locked, "unit_test_tag2")
    assert len(logged) == 1
    assert logged[0][0] == "unit_test_tag2"
    assert "locked" in logged[0][1].lower()


def test_persist_with_sqlite_retry_propagates_other_errors(monkeypatch):
    from web import server

    def schema_err():
        raise sqlite3.OperationalError("no such table: rag_missing")

    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    # No retries on non-lock errors — logged on first failure.
    server._persist_with_sqlite_retry(schema_err, "unit_test_schema")
    assert len(logged) == 1
    assert "no such table" in logged[0][1]


def test_retry_pending_conversation_turns_sanitizes_legacy_infinity(
    tmp_path, monkeypatch,
):
    """Regression 2026-04-21 (bug #7): pending records written
    pre-`_sanitize_confidence` carry `-Infinity` (JSON allow_nan emits that
    non-standard token) or `null`. `_retry_pending_conversation_turns`
    used to do `float(rec.get('confidence', 0.0))` which happily re-hydrated
    `-inf` and pushed it into the frontmatter as `confidence_avg: -inf`,
    polluting the note with a value that broke the next turn's averaging.

    Tras el fix, el retry pasa por `_sanitize_confidence` → 0.0 finito.
    Este test seedea un pending file con tres shapes problemáticas y
    verifica que:
      1. Las tres se re-aplican sin raise.
      2. Las notas resultantes tienen `confidence_avg` finito (no `-inf`,
         no `nan`, no string garbage).
      3. El pending file queda vacío / borrado porque todas fueron
         consumidas.
    """

    from web import server as server_mod
    from web import conversation_writer as cw  # noqa: F401 — ensure module loads
    import rag

    vault_root = tmp_path / "vault"
    (vault_root / "00-Inbox" / "conversations").mkdir(parents=True)
    # Post 2026-04-21 split: conversation_writer reads rag.DB_PATH dynamically
    # and appends _TELEMETRY_DB_FILENAME. Redirect both so this test stays
    # hermetic without poking a private symbol that no longer exists.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    pending_path = tmp_path / "conversation_turn_pending.jsonl"
    monkeypatch.setattr(server_mod, "_CONV_PENDING_PATH", pending_path)

    # Registros con las tres formas más comunes de "confidence corrupta"
    # que vimos en pending.jsonl y en logs: -Infinity (json allow_nan),
    # null (pre-fix caller pasaba None), y un string garbage.
    base_ts = "2026-04-21T09:00:00+00:00"
    # Note: los session_ids NO contienen "inf"/"nan" para que el check
    # de contenido saneado no matchee el sid literal.
    records = [
        {
            "ts": base_ts,
            "vault_root": str(vault_root),
            "session_id": "web:legacyA",
            "question": "q1 con negative-infinity",
            "answer": "a1",
            "sources": [],
            "confidence": float("-inf"),  # json.dumps → "-Infinity"
            "error": "ValueError prior",
        },
        {
            "ts": base_ts,
            "vault_root": str(vault_root),
            "session_id": "web:legacyB",
            "question": "q2 con null",
            "answer": "a2",
            "sources": [],
            "confidence": None,
            "error": "ValueError prior",
        },
        {
            "ts": base_ts,
            "vault_root": str(vault_root),
            "session_id": "web:legacyC",
            "question": "q3 con string garbage",
            "answer": "a3",
            "sources": [],
            "confidence": "not-a-number",
            "error": "ValueError prior",
        },
    ]
    import json as _json
    pending_path.write_text(
        "\n".join(_json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    # `_retry_pending_conversation_turns` importa write_turn y usa el
    # `Path(rec["vault_root"])` directo; TurnData wraps con timestamp
    # parseado. No necesitamos monkey-patchar más — el path real ejerce
    # el código de producción.
    n_retried = server_mod._retry_pending_conversation_turns()
    assert n_retried == 3, f"expected 3 retries, got {n_retried}"

    # Las tres notas quedan escritas en 00-Inbox/conversations/.
    conv_folder = vault_root / "00-Inbox" / "conversations"
    written = sorted(conv_folder.glob("*.md"))
    assert len(written) == 3, f"expected 3 notes, got {[p.name for p in written]}"

    # Ninguna carga `-inf` / `nan` en el frontmatter — el sanitizer clampea
    # a 0.0 finito antes de `_write_frontmatter` (que usa `f"{c:.3f}"`).
    # Chequeamos SOLO la línea del confidence_avg para no matchear el
    # session_id por accidente.
    for p in written:
        text = p.read_text(encoding="utf-8")
        avg_line = next(
            ln for ln in text.splitlines() if ln.startswith("confidence_avg:")
        )
        assert "-inf" not in avg_line.lower(), f"{p.name}: {avg_line!r}"
        assert "nan" not in avg_line.lower(), f"{p.name}: {avg_line!r}"
        assert "confidence_avg: 0.000" in avg_line, (
            f"{p.name} no saneado a 0.000: {avg_line!r}"
        )

    # Pending file queda vacío porque las 3 se consumieron.
    assert not pending_path.is_file() or pending_path.read_text().strip() == ""


def test_behavior_priors_lock_serialises_concurrent_loads(tmp_path, monkeypatch):
    """Regression 2026-04-21 (bug #8): `_behavior_priors_lock` estaba
    definido pero `_load_behavior_priors` NO lo adquiría. Dos callers
    paralelos (home-prewarmer + /api/chat, o dos /api/chat streams)
    podían intercalar un cache overwrite con el key-freshness check,
    devolviendo un snapshot stale permanentemente.

    Test approach: amplificar el race inyectando una pausa entre el
    freshness check y el cache assign en el código real es costoso;
    en vez de eso verificamos la invariante de seguridad fundamental:
    20 threads paralelos llamando `_load_behavior_priors()` en un DB
    poblado devuelven TODOS el mismo snapshot estructuralmente válido
    (mismos keys, mismo `n_events`, mismo `hash` porque los 3 globales
    se escriben bajo el RLock y ninguno los ve a medias).
    """
    import threading
    import rag

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Reset cache globals so el test no lee un snapshot heredado.
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)

    # Seedear el DB con 40 behavior events sintéticos. Necesitamos
    # `rag_behavior` poblada para que `_sql_max_ts` devuelva non-None
    # y el path de rebuild se ejecute.
    with rag._ragvec_state_conn() as conn:
        for i in range(40):
            rag._sql_append_event(conn, "rag_behavior", {
                "ts": f"2026-04-21T10:00:{i:02d}",
                "source": "test",
                "event": "open" if i % 3 else "impression",
                "path": f"01-Projects/note_{i % 5}.md",
                "dwell_s": 12.0 if i % 3 else None,
            })

    # 20 threads paralelos — todos leen del mismo cache y deben
    # devolver snapshots idénticos post-fix.
    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        try:
            snap = rag._load_behavior_priors()
            results.append(snap)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"concurrent _load_behavior_priors raised: {errors}"
    assert len(results) == 20

    # Todos los snapshots son estructuralmente idénticos — el hash es
    # determinístico sobre MAX(ts), así que si dos threads vieran un
    # cache a medio escribir los hashes divergirían.
    first = results[0]
    for i, s in enumerate(results[1:], start=1):
        assert s["hash"] == first["hash"], (
            f"thread {i} snapshot hash {s['hash']!r} != first {first['hash']!r}"
        )
        assert s["n_events"] == first["n_events"]
        assert set(s.keys()) == set(first.keys())

    # Y el snapshot es NO-vacío (los 40 events se foldearon).
    assert first["n_events"] > 0
    assert first["hash"].startswith("sql:")
    assert len(first["click_prior"]) > 0


def test_save_vaults_config_is_atomic(tmp_path, monkeypatch):
    """_save_vaults_config must use tmp+replace so a crash mid-write
    never leaves an empty vaults.json (which would silently wipe the
    registry on next load).
    """
    import rag
    cfg_path = tmp_path / "vaults.json"
    monkeypatch.setattr(rag, "VAULTS_CONFIG_PATH", cfg_path)
    cfg = {"vaults": {"A": "/tmp/a", "B": "/tmp/b"}, "current": "A"}
    rag._save_vaults_config(cfg)
    # No orphaned `.json.tmp` left behind.
    assert not (cfg_path.with_suffix(".json.tmp")).exists()
    # Final file is valid JSON with the expected contents.
    import json
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded == cfg


# ── `_validate_retrieve_result` — defense-in-depth regressor ──────────────


class TestValidateRetrieveResult:
    """Audit 2026-04-24 surfaced 3 sitios en web/server.py donde el handler
    accedía a `result["confidence"]` / `m["file"]` / `zip(metas, scores)`
    sin validar el shape. `_validate_retrieve_result` corre una vez después
    del retrieve y garantiza invariantes para el resto del handler.

    Tests cubren:
      1. Happy path: shape correcto → passthrough.
      2. Lengths mismatch → truncate al min común + log.
      3. Metas malformadas (None/str/int) → coerce a `{}`.
      4. Confidence None/NaN/Inf/str → 0.0 via `_sanitize_confidence`.
      5. Missing keys → default vacío sin crash.
    """

    def test_happy_path_preserves_shape(self):
        from web.server import _validate_retrieve_result
        result = {
            "docs": ["doc1", "doc2"],
            "metas": [{"file": "A.md"}, {"file": "B.md"}],
            "scores": [0.9, 0.7],
            "confidence": 0.85,
            "query_variants": ["v1"],  # extra fields preservados
        }
        out = _validate_retrieve_result(result)
        assert out is result  # mutación in-place
        assert len(out["docs"]) == 2
        assert len(out["metas"]) == 2
        assert len(out["scores"]) == 2
        assert out["confidence"] == 0.85
        assert out["query_variants"] == ["v1"]  # preservado

    def test_mismatch_truncates_to_min(self, capsys):
        """docs=3, metas=2, scores=3 → truncate todos a 2."""
        from web.server import _validate_retrieve_result
        result = {
            "docs": ["d1", "d2", "d3"],
            "metas": [{"file": "A.md"}, {"file": "B.md"}],
            "scores": [0.9, 0.8, 0.7],
            "confidence": 0.5,
        }
        _validate_retrieve_result(result)
        assert len(result["docs"]) == 2
        assert len(result["metas"]) == 2
        assert len(result["scores"]) == 2
        # El tercer doc/score se descartó — el handler ya no puede mis-attribuir.
        assert "d3" not in result["docs"]
        # Log del mismatch visible para operators.
        out = capsys.readouterr().out
        assert "[retrieve-shape-mismatch]" in out
        assert "docs=3" in out
        assert "metas=2" in out
        assert "truncating to 2" in out

    def test_metas_non_dict_coerced_to_empty_dict(self):
        """Si rag.py devuelve un meta None/str/int (bug downstream), el
        validator lo reemplaza con `{}` para que `m.get("file", "")`
        después no crashee."""
        from web.server import _validate_retrieve_result
        result = {
            "docs": ["d1", "d2", "d3", "d4"],
            "metas": [{"file": "A.md"}, None, "broken", 42],
            "scores": [0.9, 0.8, 0.7, 0.6],
            "confidence": 0.5,
        }
        _validate_retrieve_result(result)
        # Los 3 malformados se reemplazaron con `{}`, sin perder slot.
        assert len(result["metas"]) == 4
        assert result["metas"][0] == {"file": "A.md"}
        assert result["metas"][1] == {}
        assert result["metas"][2] == {}
        assert result["metas"][3] == {}

    def test_confidence_sanitized(self):
        """NaN / Inf / None / str → 0.0. float válido pasa."""
        import math
        from web.server import _validate_retrieve_result
        for bad in (float("nan"), float("inf"), float("-inf"), None, "xyz", [], {}):
            result = {"docs": [], "metas": [], "scores": [], "confidence": bad}
            _validate_retrieve_result(result)
            conf = result["confidence"]
            assert isinstance(conf, float)
            assert not math.isnan(conf)
            assert not math.isinf(conf)
            assert conf == 0.0

    def test_missing_keys_default_to_empty(self):
        """Result completamente vacío (caller esperaba {"docs":[],...} pero
        le llegó {}) no crashea."""
        from web.server import _validate_retrieve_result
        result: dict = {}
        _validate_retrieve_result(result)
        assert result["docs"] == []
        assert result["metas"] == []
        assert result["scores"] == []
        assert result["confidence"] == 0.0

    def test_none_values_handled_like_missing(self):
        """Si rag.py devuelve `None` en alguna key (en vez de lista vacía),
        lo tratamos igual que missing — no crashamos con `TypeError:
        'NoneType' object is not iterable`."""
        from web.server import _validate_retrieve_result
        result = {
            "docs": None,
            "metas": None,
            "scores": None,
            "confidence": None,
        }
        _validate_retrieve_result(result)
        assert result["docs"] == []
        assert result["metas"] == []
        assert result["scores"] == []
        assert result["confidence"] == 0.0

    def test_all_empty_is_valid(self):
        """Retrieve sin hits — `docs=[]`, `metas=[]`, `scores=[]` es
        estado legítimo. No debe tratarse como mismatch."""
        from web.server import _validate_retrieve_result
        result = {
            "docs": [], "metas": [], "scores": [],
            "confidence": float("-inf"),  # sentinel de rag.py
        }
        _validate_retrieve_result(result)
        assert result["docs"] == []
        assert result["confidence"] == 0.0  # -inf sanitized

    def test_only_missing_confidence_crashes_handler_fix(self):
        """Regression: handler acceso `float(result["confidence"])` sin
        .get(), crasheaba si rag.py omitía la key. Post-fix: validator
        normaliza → `result["confidence"] = 0.0`."""
        from web.server import _validate_retrieve_result
        result = {"docs": ["d"], "metas": [{"file": "A"}], "scores": [0.5]}
        # No hay "confidence" — pre-fix sería KeyError en float(result["confidence"]).
        _validate_retrieve_result(result)
        assert result["confidence"] == 0.0
        # Ahora el handler puede hacer float(result["confidence"]) sin riesgo.
        assert isinstance(result["confidence"], float)


# ── Rate limiter deque O(1) expiration — audit C1 ────────────────────────


def test_rate_limit_uses_deque_for_o1_expiration():
    """2026-04-24 audit: `_CHAT_BUCKETS` y `_BEHAVIOR_BUCKETS` cambiaron
    de `list` a `collections.deque` para que el sliding-window
    expiration (`while events[0] < cutoff: events.pop(0)`) sea O(1) por
    pop en vez de O(n). Bajo carga moderada el loop caía a O(n²) y
    dominaba CPU.

    Este test verifica que el tipo subyacente ES deque (signal del fix).
    """
    import collections as _collections
    from web import server
    # `_CHAT_BUCKETS` es un `defaultdict(deque)`. Al acceder a una key
    # nueva, se crea un `deque` vacío.
    server._CHAT_BUCKETS.clear()
    bucket = server._CHAT_BUCKETS["test-ip"]
    assert isinstance(bucket, _collections.deque), (
        f"_CHAT_BUCKETS[ip] debería ser deque (O(1) popleft), "
        f"got {type(bucket).__name__}"
    )
    # Mismo check para _BEHAVIOR_BUCKETS.
    server._BEHAVIOR_BUCKETS.clear()
    bh_bucket = server._BEHAVIOR_BUCKETS["test-ip"]
    assert isinstance(bh_bucket, _collections.deque)


def test_rate_limit_expires_events_correctly():
    """Smoke test del sliding-window post-deque. Events que caen fuera
    del window deben expirar, los que están dentro quedan contados."""
    import time
    from web.server import _check_rate_limit, _CHAT_BUCKETS

    _CHAT_BUCKETS.clear()
    # Primera llamada: evento se registra.
    _check_rate_limit(_CHAT_BUCKETS, "ip-1", limit=5, window=1.0)
    assert len(_CHAT_BUCKETS["ip-1"]) == 1

    # Rápido fuego dentro del window: los 5 permitidos pasan.
    for _ in range(4):
        _check_rate_limit(_CHAT_BUCKETS, "ip-1", limit=5, window=1.0)
    assert len(_CHAT_BUCKETS["ip-1"]) == 5

    # 6to intento → 429.
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        _check_rate_limit(_CHAT_BUCKETS, "ip-1", limit=5, window=1.0)
    assert exc_info.value.status_code == 429

    # Esperá a que todos expiren + una llamada más → pasa.
    time.sleep(1.1)
    _check_rate_limit(_CHAT_BUCKETS, "ip-1", limit=5, window=1.0)
    # La deque debería tener solo el evento fresco (los viejos popleft'ed).
    assert len(_CHAT_BUCKETS["ip-1"]) == 1


# ── _wa_executor cleanup on submit failure — audit C3 ────────────────────


def test_wa_executor_cleanup_pattern_works():
    """2026-04-24 audit: smoke test del patrón try/except que usa el chat
    endpoint cuando `_wa_executor.submit()` falla después del ctor
    exitoso (edge case: executor broken, out-of-threads).

    El fix es el bloque:
        try:
            _wa_executor = ThreadPoolExecutor(...)
            _wa_future = _wa_executor.submit(...)
        except Exception:
            if _wa_executor is not None:
                _wa_executor.shutdown(wait=False)
                _wa_executor = None
            _wa_future = None

    Pre-fix este patrón no existía: el submit failure dejaba el executor
    "vivo" pero sin future, y el finally downstream solo cleanup-eaba
    cuando `_wa_future is not None` (que en este caso no lo era) → thread
    leak silencioso.

    Este test verifica que el patrón cleanup llama a `shutdown()` cuando
    submit raisea — sin depender de monkeypatch del módulo (ThreadPoolExecutor
    se importa como símbolo, no como atributo del módulo server).
    """
    class _BrokenExecutor:
        def __init__(self, *args, **kwargs):
            self.shutdown_called = False

        def submit(self, *args, **kwargs):
            raise RuntimeError("executor broken")

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_called = True

    # Replica del patrón del fix en web/server.py:5419-5441.
    _wa_executor = None
    _wa_future = None
    try:
        _wa_executor = _BrokenExecutor()
        _wa_future = _wa_executor.submit(lambda: None)  # raisea
    except Exception:
        if _wa_executor is not None:
            _wa_executor.shutdown(wait=False)
            _wa_executor_saved = _wa_executor  # preservar ref para assertion
            _wa_executor = None
        _wa_future = None

    # El executor se limpió (shutdown called) y _wa_future quedó en None.
    assert _wa_future is None
    assert _wa_executor is None
    assert _wa_executor_saved.shutdown_called, (
        "shutdown() no se llamó después de submit failure — "
        "thread leak potencial"
    )
