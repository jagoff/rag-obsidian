import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


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
    try:
        yield
    finally:
        # (a) Drain the writer queue so assertions in the next test aren't
        # contaminated by writes enqueued by the previous one.
        try:
            _rag._LOG_QUEUE.join()
        except Exception:
            pass

        # (b) Detect flag/DB_PATH drift — restore + warn.
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
