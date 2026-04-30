"""Audit 2026-04-30 — Fix 2: handle_recent ahora filtra por ventana temporal
cuando el caller pasa `question`. Pre-fix devolvía top-20 by modified desc
ignorando "esta semana"/"hoy"/etc.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag  # noqa: E402


def _meta(file: str, modified: datetime) -> dict:
    return {
        "file": file,
        "note": file.replace(".md", ""),
        "folder": file.split("/")[0] if "/" in file else "",
        "modified": modified.isoformat(timespec="seconds"),
        "tags": "",
    }


def _stub_col(metas: list[dict]) -> MagicMock:
    """Stub de SqliteVecCollection: _load_corpus devuelve {"metas": ...}."""
    return MagicMock()


@pytest.fixture
def patched_load_corpus(monkeypatch):
    state = {"metas": []}
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": state["metas"]})
    return state


def test_handle_recent_legacy_no_question(patched_load_corpus):
    """Sin `question` → comportamiento legacy: top-N by modified desc."""
    now = datetime.now()
    patched_load_corpus["metas"] = [
        _meta("a.md", now - timedelta(days=90)),
        _meta("b.md", now - timedelta(days=1)),
        _meta("c.md", now - timedelta(days=30)),
    ]
    files = rag.handle_recent(_stub_col([]), {}, limit=20)
    # Top by modified desc — sin filtro temporal
    assert [f["file"] for f in files] == ["b.md", "c.md", "a.md"]


def test_handle_recent_filters_to_this_week(patched_load_corpus):
    """`question='qué hice esta semana'` → solo notas dentro del rango."""
    now = datetime.now()
    patched_load_corpus["metas"] = [
        _meta("hoy.md", now - timedelta(hours=3)),
        _meta("ayer.md", now - timedelta(days=1)),
        _meta("hace_un_mes.md", now - timedelta(days=30)),
        _meta("hace_dos_meses.md", now - timedelta(days=60)),
    ]
    files = rag.handle_recent(_stub_col([]), {}, limit=20, question="qué hice esta semana")
    paths = {f["file"] for f in files}
    # esta semana = [lunes 00:00, próx lunes 00:00). hoy y ayer SIEMPRE
    # caen adentro salvo que hoy sea lunes muy temprano. Si es lunes
    # antes de las ~03:00, ayer (domingo) cae fuera de la semana actual.
    assert "hoy.md" in paths
    assert "hace_un_mes.md" not in paths
    assert "hace_dos_meses.md" not in paths


def test_handle_recent_filters_to_today(patched_load_corpus):
    """`question='notas modificadas hoy'` → solo hoy."""
    now = datetime.now()
    patched_load_corpus["metas"] = [
        _meta("hoy.md", now - timedelta(hours=2)),
        _meta("ayer.md", now - timedelta(days=1, hours=2)),
        _meta("hace_semana.md", now - timedelta(days=7)),
    ]
    files = rag.handle_recent(_stub_col([]), {}, limit=20, question="notas modificadas hoy")
    paths = {f["file"] for f in files}
    assert "hoy.md" in paths
    assert "ayer.md" not in paths
    assert "hace_semana.md" not in paths


def test_handle_recent_question_without_temporal_returns_all(patched_load_corpus):
    """`question` sin temporal cue → comportamiento legacy (sin filtro)."""
    now = datetime.now()
    patched_load_corpus["metas"] = [
        _meta("a.md", now - timedelta(days=90)),
        _meta("b.md", now - timedelta(days=1)),
    ]
    files = rag.handle_recent(_stub_col([]), {}, limit=20, question="qué hice")
    assert [f["file"] for f in files] == ["b.md", "a.md"]


def test_handle_recent_invalid_modified_skipped_when_filtering(patched_load_corpus):
    """Metas con `modified` inválido se skipean cuando hay filtro temporal."""
    now = datetime.now()
    patched_load_corpus["metas"] = [
        _meta("hoy.md", now - timedelta(hours=2)),
        {"file": "broken.md", "note": "broken", "modified": "not-a-date", "tags": ""},
    ]
    files = rag.handle_recent(_stub_col([]), {}, limit=20, question="notas de hoy")
    paths = {f["file"] for f in files}
    assert "hoy.md" in paths
    assert "broken.md" not in paths


def test_handle_recent_signature_accepts_kwarg():
    """Sanity-check: la firma acepta `question` como kwarg explícito."""
    import inspect

    sig = inspect.signature(rag.handle_recent)
    assert "question" in sig.parameters
    p = sig.parameters["question"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None
