import sys
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
