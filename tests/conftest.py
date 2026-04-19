import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


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
