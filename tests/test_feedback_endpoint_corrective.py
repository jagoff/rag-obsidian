"""Tests for the /api/feedback endpoint corrective_path flow (2026-04-22).

Web-side counterpart of test_corrective_path_chat.py (CLI flow). When the
user picks 👎 on a web answer the UI now shows the top-5 source cards as
selectable, plus a "ninguna" option. The selection posts to /api/feedback
with `corrective_path` alongside the rating.

Invariants tested here:
  1. corrective_path is passed through to record_feedback()
  2. presence of corrective_path forces reason='corrective' (sentinel for
     `_feedback_augmented_cases()` mining in `rag tune`)
  3. absence of corrective_path preserves the user's free-text reason
  4. cross-source native ids (calendar://, whatsapp://, gmail://) are
     silent-dropped server-side — they are not vault-relative paths and
     `_feedback_augmented_cases()` cannot consume them as positives
  5. empty-string corrective_path is normalised to None (same as missing)

These are mocked tests — the real record_feedback() writes to SQL and is
validated in test_feedback.py / test_rag_writers_sql.py. Here we verify
only the glue between the HTTP boundary and record_feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skip cleanly if web deps aren't importable (pure-rag test run).
_web_server = pytest.importorskip("web.server")
_fastapi = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


def _base_payload(rating: int = -1) -> dict:
    return {
        "turn_id": "abc123",
        "rating": rating,
        "q": "cuál es mi postura al andar en skate",
        "paths": ["02-Areas/Salud/postura.md", "other.md"],
        "session_id": None,
    }


def test_corrective_path_forwarded_to_record_feedback(client):
    """Happy path: user picks a card, server forwards path + flips reason."""
    payload = {**_base_payload(),
               "corrective_path": "02-Areas/Salud/postura.md"}
    with patch.object(_web_server, "record_feedback") as mock_rf:
        resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mock_rf.assert_called_once()
    call_kwargs = mock_rf.call_args.kwargs
    assert call_kwargs["corrective_path"] == "02-Areas/Salud/postura.md"
    # Reason gets overridden to the sentinel even though the client didn't
    # send a reason — this is what `_feedback_augmented_cases()` SQL filter
    # keys on (reason='corrective').
    assert call_kwargs["reason"] == "corrective"
    assert call_kwargs["rating"] == -1


def test_corrective_overrides_free_text_reason(client):
    """When both arrive, corrective_path wins — reason becomes 'corrective'.
    The free-text reason is discarded because that's the signal the mining
    SQL looks for in `_feedback_augmented_cases()`."""
    payload = {**_base_payload(),
               "corrective_path": "02-Areas/Salud/postura.md",
               "reason": "muy genérico, buscaba la nota sobre skate"}
    with patch.object(_web_server, "record_feedback") as mock_rf:
        resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    assert mock_rf.call_args.kwargs["reason"] == "corrective"
    assert mock_rf.call_args.kwargs["corrective_path"] == "02-Areas/Salud/postura.md"


def test_free_text_reason_preserved_without_corrective(client):
    """Client sent a reason but no corrective_path — reason flows through."""
    payload = {**_base_payload(),
               "reason": "falta la nota de 2024 sobre peso en skate"}
    with patch.object(_web_server, "record_feedback") as mock_rf:
        resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    assert mock_rf.call_args.kwargs["reason"] == "falta la nota de 2024 sobre peso en skate"
    assert mock_rf.call_args.kwargs["corrective_path"] is None


def test_cross_source_corrective_path_rejected(client):
    """calendar://, whatsapp://, gmail:// are not vault paths — the mining
    SQL in `_feedback_augmented_cases()` can't turn them into positive
    training pairs. Server drops them silently but keeps the rating."""
    for bad in ("calendar://EVENT-abc",
                "whatsapp://chat/123",
                "gmail://inbox/xyz"):
        payload = {**_base_payload(), "corrective_path": bad}
        with patch.object(_web_server, "record_feedback") as mock_rf:
            resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 200
        assert mock_rf.call_args.kwargs["corrective_path"] is None, (
            f"expected {bad!r} to be dropped, got it forwarded"
        )
        # Without a valid corrective, reason stays None (since we didn't
        # send one).
        assert mock_rf.call_args.kwargs["reason"] is None


def test_empty_corrective_path_normalised_to_none(client):
    """Empty string == missing. Whitespace-only too."""
    for empty in ("", "   "):
        payload = {**_base_payload(), "corrective_path": empty}
        with patch.object(_web_server, "record_feedback") as mock_rf:
            resp = client.post("/api/feedback", json=payload)
        assert resp.status_code == 200
        assert mock_rf.call_args.kwargs["corrective_path"] is None


def test_corrective_works_on_positive_rating_too(client):
    """Edge case: +1 + corrective_path (e.g. user 👍 but still picked the
    best source to boost it). Server forwards verbatim — the intent is
    "this path is high-quality", even more so than the rating alone."""
    payload = {**_base_payload(rating=1),
               "corrective_path": "02-Areas/Salud/postura.md"}
    with patch.object(_web_server, "record_feedback") as mock_rf:
        resp = client.post("/api/feedback", json=payload)
    assert resp.status_code == 200
    assert mock_rf.call_args.kwargs["rating"] == 1
    assert mock_rf.call_args.kwargs["corrective_path"] == "02-Areas/Salud/postura.md"
    assert mock_rf.call_args.kwargs["reason"] == "corrective"
