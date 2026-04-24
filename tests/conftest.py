import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# Module-level ollama availability check — evaluated ONCE at collection
# time. Avoids paying the subprocess + HTTP cost per test when 100+ tests
# share the same skip condition.
def _has_chat_model() -> bool:
    """True if `resolve_chat_model()` finds an installed model in ollama.
    False when ollama is down, not installed, or no CHAT_MODEL_PREFERENCE
    model is pulled — CI, fresh dev boxes, sandboxed environments."""
    try:
        import rag
        rag.resolve_chat_model()
        return True
    except Exception:
        return False


_HAS_CHAT_MODEL: bool = _has_chat_model()


@pytest.fixture(autouse=True)
def _skip_if_no_ollama(request):
    """Skip tests decorated with `@pytest.mark.requires_ollama` when
    no ollama chat model is available (CI on ubuntu-latest, machines
    without ollama pulled).

    The marker is applied at either test or module scope:
        pytestmark = pytest.mark.requires_ollama      # file-wide
        @pytest.mark.requires_ollama                  # per-test
    """
    if request.node.get_closest_marker("requires_ollama") and not _HAS_CHAT_MODEL:
        pytest.skip(
            "requires ollama with a CHAT_MODEL_PREFERENCE model installed "
            "(e.g. `ollama pull qwen2.5:7b`); not available in this env"
        )


@pytest.fixture(autouse=True)
def _isolate_apple_integrations(request, monkeypatch):
    """Safety-critical: bloquear escrituras a Apple Reminders / Calendar /
    Contacts reales durante los tests.

    `rag._apple_enabled()` devuelve `True` por default a menos que
    `OBSIDIAN_RAG_NO_APPLE=1` esté seteada — y `propose_reminder` /
    `propose_calendar_event` usan eso como único gate antes de llamar
    `osascript` con `make new reminder` / `make new event`. Si un test
    ejercita esos code paths sin mockear `_create_reminder` /
    `_create_calendar_event` / `_osascript`, termina creando entidades
    reales en Reminders.app / Calendar.app del user.

    Gap documentado 2026-04-24 Fer F. report: durante una pytest run
    de 2h el test `test_semantic_skipped_when_propose_intent`
    (tests/test_web_chat_semantic_cache.py) disparó
    `_post_chat("recordame llamar a mamá a las 18")` con el propose
    path "parcialmente stubbed" (su propio comentario) — cada corrida
    creaba un `llamar a mamá @ 18:00 hoy` en Reminders del user. Idem
    `test_propose_intent_blocks_bypass` con `recordame comprar pan mañana`.
    La única razón de que no explote en los ~30+ tests que mockean
    `_osascript` explícitamente es que ellos saben que hay que hacerlo;
    los que usan el propose path end-to-end olvidan.

    Fix defensivo: setear `OBSIDIAN_RAG_NO_APPLE=1` por default ANTES
    de cada test. `_apple_enabled` lee `os.environ.get` puro sin
    caching, así que la env var se aplica en cada call. Cualquier flow
    que pase por ese guard hace early-return con mensaje `"Apple
    integration deshabilitada"` y cero osascript invocation — seguro.

    Opt-out: tests que genuinamente quieren ejercitar la integración
    real (hoy no hay ninguno en la suite) pueden decorarse con
    `@pytest.mark.real_apple` para saltar el override.

    Compatible con los tests que ya hacen
    `monkeypatch.setattr(rag, "_apple_enabled", lambda: True)` por
    sí mismos: ese patch corre DESPUÉS del env var set, y es
    function-scoped, así que gana. Los tests tipo
    `test_reminder_create_extended.py` siguen viendo `_apple_enabled()`
    devolver True y sus osascript mockeados se ejecutan como antes.
    """
    if request.node.get_closest_marker("real_apple"):
        yield
        return
    monkeypatch.setenv("OBSIDIAN_RAG_NO_APPLE", "1")
    yield


@pytest.fixture(autouse=True)
def _isolate_vault_path(tmp_path_factory, request):
    """Safety-critical: `rag.VAULT_PATH` resuelve en import-time a
    `_DEFAULT_VAULT` (el iCloud Obsidian real del usuario) si no hay
    override en env o vaults.json. Sin este fixture, cualquier test que
    llame a código que toque `VAULT_PATH / ...` (p.ej. el loop agéntico
    de `rag do`, `_index_single_file`, ingesters, `rag capture`) escribiría
    al vault de producción — ~34 tests puntuales usan
    `monkeypatch.setattr(rag, "VAULT_PATH", ...)` per-test, pero los 2k+
    restantes no. Gap auditado 2026-04-21 en el hardening pass.

    Esta autouse apunta `rag.VAULT_PATH` a un tmp dir único por test
    (via `tmp_path_factory.mktemp`, no session-wide para evitar state
    leak entre tests) y restaura el valor original al teardown.

    Opt-out: tests que genuinamente necesitan leer el vault real (p.ej.
    validación de queries.yaml contra paths vivos, benchmarks de retrieval
    contra el corpus real) deben decorarse con `@pytest.mark.real_vault`.
    Estos tests NO reciben el tmp dir — `rag.VAULT_PATH` queda apuntando
    al vault real. Solo READ-ONLY tests deberían usar este marker;
    cualquier test que escriba al vault DEBE usar monkeypatch.

    Compatibilidad con los ~34 tests que ya monkeypatchean VAULT_PATH:
    al ser pytest's `monkeypatch` function-scoped y al desarmarse en
    LIFO antes de las autouse-no-deps (esta fixture no pide monkeypatch
    como arg), la secuencia es:
      1. Setup: VAULT_PATH = tmp (direct assign)
      2. Test opcionalmente hace `monkeypatch.setattr(rag, "VAULT_PATH", vault_test)`
      3. Test corre con VAULT_PATH = vault_test
      4. monkeypatch.undo() → VAULT_PATH vuelve a tmp (NO a `snap_original`,
         porque monkeypatch snapshoteó el valor tmp cuando el test llamó
         a setattr)
      5. Esta fixture's finally → VAULT_PATH = snap_original

    Drift detection (mismo patrón que `_stabilize_rag_state`): si un test
    mutó `rag.VAULT_PATH` directamente (bypasseando monkeypatch), lo
    detectamos porque después del undo la VAULT_PATH no coincide con
    `tmp`. Emitimos warning + restauramos. En la práctica ningún test
    hace esto hoy, pero el guard previene regresiones silenciosas.
    """
    # Opt-out: tests marcados con `real_vault` NO reciben tmp redirect.
    # Se confía en el autor del test para no escribir; el marker es
    # equivalente a "I know what I'm doing".
    #
    # 2026-04-24: antes el opt-out hacía `yield None` a secas, heredando
    # cualquier VAULT_PATH stale que un test anterior hubiera dejado
    # (drift por mutación directa + monkeypatch.undo se desarma al
    # tmp capturado, no al _DEFAULT_VAULT). Los tests real_vault
    # assumen que VAULT_PATH apunta al vault real del user —
    # forzamos esa invariante explícitamente antes de ceder el
    # control, y restauramos a snap_original al teardown (mismo patrón
    # que la branch normal). Fix para falla intermitente de
    # `test_queries_yaml_all_paths_exist_or_placeholder` en full suite
    # run (passa en isolation, fallaba solo tras que otros tests
    # movieran VAULT_PATH).
    if request.node.get_closest_marker("real_vault"):
        import rag as _rag
        snap_real = _rag.VAULT_PATH
        # Re-resolve al default real (el que había en import-time).
        _rag.VAULT_PATH = _rag._DEFAULT_VAULT
        try:
            yield None
        finally:
            _rag.VAULT_PATH = snap_real
        return

    import rag as _rag
    snap_original = _rag.VAULT_PATH
    tmp = tmp_path_factory.mktemp("vault_isolated")
    _rag.VAULT_PATH = tmp
    try:
        yield tmp
    finally:
        # monkeypatch.undo() ya corrió (LIFO: esta fixture no depende de
        # monkeypatch → tears down después de él). Cualquier drift vs
        # `tmp` es mutación directa que bypasseó monkeypatch.
        current = _rag.VAULT_PATH
        if current != tmp:
            warnings.warn(
                f"rag.VAULT_PATH leaked from test (expected {tmp}, "
                f"now {current}); restoring to module default",
                stacklevel=2,
            )
        _rag.VAULT_PATH = snap_original


@pytest.fixture(autouse=True)
def _isolate_silent_errors_log(tmp_path_factory):
    """Evita que los tests que ejercen paths con `_silent_log` (session
    JSON corrupto, ranker.json corrupto, synthetic_q_cache corrupto,
    etc.) contaminen el `~/.local/share/obsidian-rag/silent_errors.jsonl`
    real del usuario. Pre-fix, 100+ entries aparecían ahí después de
    cada corrida de suite (session_load_json JSONDecodeError, etc.),
    haciendo imposible distinguir errores reales de ruido de tests.

    Redirige `SILENT_ERRORS_LOG_PATH` a un archivo en tmp por la
    duración del test. El `_LOG_QUEUE` worker (daemon thread) sigue
    drainando al path monkeypatched; el real user dir queda intacto.
    """
    import rag as _rag
    tmp = tmp_path_factory.mktemp("silent_errors") / "silent_errors.jsonl"
    original = _rag.SILENT_ERRORS_LOG_PATH
    _rag.SILENT_ERRORS_LOG_PATH = tmp
    try:
        yield
    finally:
        # Drenar el queue ANTES de restaurar, para que los writes pendientes
        # que encolan `(SILENT_ERRORS_LOG_PATH, line)` con el path viejo
        # (ya bound en la tuple) aterricen en tmp y no en el real.
        try:
            _rag._LOG_QUEUE.join()
        except Exception:
            pass
        _rag.SILENT_ERRORS_LOG_PATH = original


@pytest.fixture(autouse=True)
def _isolate_sql_state_error_log(tmp_path_factory):
    """Gemelo de `_isolate_silent_errors_log`: evita que tests que disparan
    `_log_sql_state_error` (SQL writer/reader fails con monkeypatched DB,
    `test_tag` events, etc.) contaminen el
    `~/.local/share/obsidian-rag/sql_state_errors.jsonl` real del usuario.

    Bug medido en el audit 2026-04-24: **161 entries de `test_tag`** en el
    sql_state_errors.jsonl de producción, mezcladas con 1595 errores reales
    — imposible distinguir ruido de test vs señal de ops real sin filtrar a
    mano. Pre-fix este path no se monkeypatcheaba en ningún lado.

    Redirige `_SQL_STATE_ERROR_LOG` a tmp por la duración del test. El file
    append es sync (no hay queue intermedio), así que al restore el path
    viejo no tiene pendings colgados — simplemente reponemos el valor
    original y las writes futuras vuelven al path real.
    """
    import rag as _rag
    tmp = tmp_path_factory.mktemp("sql_state_errors") / "sql_state_errors.jsonl"
    original = _rag._SQL_STATE_ERROR_LOG
    _rag._SQL_STATE_ERROR_LOG = tmp
    try:
        yield
    finally:
        _rag._SQL_STATE_ERROR_LOG = original


@pytest.fixture(autouse=True)
def _clear_query_caches():
    """Evita que los LRU de embed/expand_queries contaminen tests.

    Las paraphrases son determinísticas en producción (seed=42), pero en los
    tests el mismo string se mockea con respuestas distintas entre cases.
    """
    import rag as _rag
    _rag._embed_cache.clear()
    _rag._expand_cache.clear()
    yield
    _rag._embed_cache.clear()
    _rag._expand_cache.clear()


@pytest.fixture(autouse=True)
def _disable_semantic_response_cache(monkeypatch):
    """GC#1 (2026-04-22): semantic response cache is ENABLED by default. In
    tests this causes cross-contamination — test A stores a stubbed response,
    test B (same query via monkeypatched chat) gets a cache hit from A and
    skips its own chat stub entirely.

    Disable the cache for every test by default. Tests that specifically
    exercise the cache (test_semantic_cache.py) override via
    `monkeypatch.setenv('RAG_CACHE_ENABLED', '1')` — env precedence means
    the per-test setenv wins over this autouse setting.
    """
    monkeypatch.setenv("RAG_CACHE_ENABLED", "0")
    yield


@pytest.fixture(autouse=True)
def _force_log_query_event_sync(monkeypatch):
    """Perf audit 2026-04-22: ``log_query_event`` pasó a async por default
    (queue daemon `_BACKGROUND_SQL_QUEUE`) para eliminar el 1.3s retry
    budget del hot path del usuario. Tests como `test_rag_writers_sql`,
    `test_telemetry_intent_logged`, `test_vaults`, etc. asumen contract
    sincrónico — llaman `log_query_event(evt)` y luego leen
    `rag_queries` en la misma secuencia.

    Forzamos sync para TODOS los tests por default. Tests que
    específicamente ejercen el path async (si alguno futuro lo hace)
    pueden override con `monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "1")`.
    """
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")
    yield


@pytest.fixture(autouse=True)
def _force_behavior_and_metrics_sync(monkeypatch):
    """Audit 2026-04-24: `log_behavior_event`, `log_impressions`, y los
    samplers de cpu/memory metrics pasaron a async por default (mismo
    queue que log_query_event) para aliviar la contención WAL contra
    telemetry.db. Mismo problema que `_force_log_query_event_sync`:
    los tests asumen contract sincrónico y leen `rag_behavior` +
    `rag_cpu_metrics` + `rag_memory_metrics` inmediatamente después
    de escribir.

    Forzamos sync para TODOS los tests. Override per-test con
    `monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "1")` o
    `RAG_METRICS_ASYNC=1`.
    """
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")
    monkeypatch.setenv("RAG_METRICS_ASYNC", "0")
    yield


@pytest.fixture(autouse=True)
def _snapshot_rag_local_embed_env():
    """`_maybe_auto_enable_local_embed` (rag.py:6970) mutates `os.environ`
    directly when the CLI group runs for query-like subcommands. Any test that
    calls `CliRunner().invoke(rag.cli, ["query", ...])` leaks `RAG_LOCAL_EMBED=1`
    into the process env — and since monkeypatch only reverts values *it* set,
    the leak contaminates subsequent tests that assume the flag is unset (e.g.
    `test_retrieve_source_filter.py` with 8-dim mock vec collections).

    Snapshot the flag before each test and restore it after.
    """
    before = os.environ.get("RAG_LOCAL_EMBED")
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("RAG_LOCAL_EMBED", None)
        else:
            os.environ["RAG_LOCAL_EMBED"] = before


@pytest.fixture(autouse=True)
def _stabilize_rag_state():
    """Flake root cause — the T4 branch introduced writer/reader tests that
    mutate module-level globals (`RAG_STATE_SQL`, `DB_PATH`). Pytest's
    monkeypatch reverts those correctly, BUT:

    1. `log_behavior_event()` and `log_query_event()` enqueue onto the module
       `_LOG_QUEUE`, which is drained by a daemon thread (`_LOG_THREAD`) — so
       tests that call these writers + assert on the JSONL output have an
       implicit race. Under heavier pytest load (output capture + many prior
       tests), the assertion fires before the background write lands. This
       manifested as intermittent failures in `test_brief_diff_signal.py` and
       similar readers that indirectly enqueue via `_diff_brief_signal`.

    2. If a test crashes mid-way after setting `RAG_STATE_SQL=True` but before
       `monkeypatch` unwinds, the flag leaks to later tests. Those tests would
       then hit the SQL branch of writers and accidentally touch the LIVE DB
       (`~/.local/share/obsidian-rag/ragvec/ragvec.db`) when `DB_PATH` wasn't
       redirected in parallel. This fixture asserts both are restored.

    Fix: after every test,
      (a) drain `_LOG_QUEUE` so filesystem state is quiescent before teardown;
      (b) snapshot & restore `RAG_STATE_SQL` + `DB_PATH` if they drifted,
          emitting a warning so the offending test is obvious.
    """
    import rag as _rag
    snap_sql_flag = _rag.RAG_STATE_SQL
    snap_db_path = _rag.DB_PATH
    snap_telemetry_db = _rag._TELEMETRY_DB_FILENAME
    try:
        yield
    finally:
        # (a) Drain the writer queue so assertions in the next test aren't
        # contaminated by writes enqueued by the previous one.
        try:
            _rag._LOG_QUEUE.join()
        except Exception:
            pass

        # (b) Detect flag/DB_PATH/_TELEMETRY_DB_FILENAME drift — restore + warn.
        if _rag.RAG_STATE_SQL is not snap_sql_flag:
            warnings.warn(
                f"RAG_STATE_SQL leaked from test (was {snap_sql_flag}, "
                f"now {_rag.RAG_STATE_SQL}); restoring",
                stacklevel=2,
            )
            _rag.RAG_STATE_SQL = snap_sql_flag
        if _rag.DB_PATH != snap_db_path:
            warnings.warn(
                f"rag.DB_PATH leaked from test (was {snap_db_path}, "
                f"now {_rag.DB_PATH}); restoring",
                stacklevel=2,
            )
            _rag.DB_PATH = snap_db_path
        if _rag._TELEMETRY_DB_FILENAME != snap_telemetry_db:
            warnings.warn(
                f"rag._TELEMETRY_DB_FILENAME leaked from test (was {snap_telemetry_db!r}, "
                f"now {_rag._TELEMETRY_DB_FILENAME!r}); restoring",
                stacklevel=2,
            )
            _rag._TELEMETRY_DB_FILENAME = snap_telemetry_db
