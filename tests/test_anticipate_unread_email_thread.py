"""Tests for the 'unread_email_thread' Anticipatory Agent signal.

Mockea `rag.integrations.gmail._fetch_gmail_evidence` para evitar
network/OAuth — patch retorna `{"awaiting_reply": [...]}` con shape
controlada.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rag.integrations import gmail as gmail_pkg
from rag_anticipate.signals.unread_email_thread import (
    _truncate,
    unread_email_thread_signal,
)


_REF_NOW = datetime(2026, 5, 10, 12, 0, 0)


def _make_thread(
    *,
    thread_id: str = "t1",
    sender: str = "Juan Pérez <juan@example.com>",
    subject: str = "Re: presupuesto",
    snippet: str = "te paso el adjunto cuando pueda",
    days_old: float = 3.0,
) -> dict:
    return {
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "snippet": snippet,
        "days_old": days_old,
    }


@pytest.fixture
def mock_gmail(monkeypatch):
    """Returns a setter `set_threads(list)` to control the fixture."""
    state: dict = {"threads": []}

    def _fake_fetch(_now):
        return {"awaiting_reply": list(state["threads"])}

    monkeypatch.setattr(gmail_pkg, "_fetch_gmail_evidence", _fake_fetch)
    return state


# ── _truncate helper ─────────────────────────────────────────────────────────


def test_truncate_short_unchanged():
    assert _truncate("hola", 10) == "hola"


def test_truncate_exact_length():
    assert _truncate("hola", 4) == "hola"


def test_truncate_long_with_ellipsis():
    s = "esta es una linea bastante larga que va a ser truncada"
    out = _truncate(s, 30)
    assert len(out) <= 30
    assert out.endswith("…")


def test_truncate_strips_newlines():
    out = _truncate("linea1\nlinea2", 100)
    assert "\n" not in out
    assert "linea1 linea2" == out


def test_truncate_word_boundary():
    """Trunc a 20 chars debe cortar en espacio si está cerca del corte."""
    s = "una frase con muchas palabras cortas"
    out = _truncate(s, 20)
    assert len(out) <= 20
    assert not out.endswith(" …")  # No espacio antes de elipsis


# ── Empty / silent-fail ──────────────────────────────────────────────────────


def test_empty_threads_returns_empty(mock_gmail):
    mock_gmail["threads"] = []
    assert unread_email_thread_signal(_REF_NOW) == []


def test_gmail_fetch_returns_none(monkeypatch):
    monkeypatch.setattr(gmail_pkg, "_fetch_gmail_evidence", lambda _now: None)
    assert unread_email_thread_signal(_REF_NOW) == []


def test_gmail_fetch_raises(monkeypatch):
    def _broken(_now):
        raise RuntimeError("api down")
    monkeypatch.setattr(gmail_pkg, "_fetch_gmail_evidence", _broken)
    assert unread_email_thread_signal(_REF_NOW) == []


def test_gmail_returns_non_dict(monkeypatch):
    monkeypatch.setattr(gmail_pkg, "_fetch_gmail_evidence", lambda _now: "garbage")
    assert unread_email_thread_signal(_REF_NOW) == []


# ── Window filtering ─────────────────────────────────────────────────────────


def test_too_recent_skipped(mock_gmail):
    """<1d → no emit."""
    mock_gmail["threads"] = [_make_thread(days_old=0.5)]
    assert unread_email_thread_signal(_REF_NOW) == []


def test_too_old_skipped(mock_gmail):
    """>30d → no emit."""
    mock_gmail["threads"] = [_make_thread(days_old=45.0)]
    assert unread_email_thread_signal(_REF_NOW) == []


# ── Score bucketing ──────────────────────────────────────────────────────────


def test_age_1d_score_0_5(mock_gmail):
    mock_gmail["threads"] = [_make_thread(days_old=1.5)]
    result = unread_email_thread_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.5


def test_age_5d_score_0_7(mock_gmail):
    mock_gmail["threads"] = [_make_thread(days_old=5.0)]
    result = unread_email_thread_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.7


def test_age_10d_score_1_0(mock_gmail):
    mock_gmail["threads"] = [_make_thread(days_old=10.0)]
    result = unread_email_thread_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 1.0


# ── Malformed threads skipped ────────────────────────────────────────────────


def test_missing_sender_skipped(mock_gmail):
    mock_gmail["threads"] = [_make_thread(sender="")]
    assert unread_email_thread_signal(_REF_NOW) == []


def test_missing_subject_skipped(mock_gmail):
    mock_gmail["threads"] = [_make_thread(subject="")]
    assert unread_email_thread_signal(_REF_NOW) == []


def test_missing_thread_id_skipped(mock_gmail):
    mock_gmail["threads"] = [_make_thread(thread_id="")]
    assert unread_email_thread_signal(_REF_NOW) == []


def test_invalid_days_old_skipped(mock_gmail):
    mock_gmail["threads"] = [_make_thread(days_old="not-a-number")]
    assert unread_email_thread_signal(_REF_NOW) == []


# ── Sender extraction ────────────────────────────────────────────────────────


def test_sender_with_name_extracted(mock_gmail):
    """`Juan Pérez <juan@example.com>` → 'Juan Pérez' en el message."""
    mock_gmail["threads"] = [_make_thread(
        sender='"Juan Pérez" <juan@example.com>', days_old=3.0,
    )]
    result = unread_email_thread_signal(_REF_NOW)
    assert "Juan Pérez" in result[0].message


def test_sender_email_only_extracts_local_part(mock_gmail):
    """`<juan@example.com>` (sin name) → 'juan' (local-part) extraído."""
    mock_gmail["threads"] = [_make_thread(
        sender="<juan@example.com>", days_old=3.0,
    )]
    result = unread_email_thread_signal(_REF_NOW)
    assert "juan" in result[0].message


# ── Sort + max ───────────────────────────────────────────────────────────────


def test_sorts_by_age_desc(mock_gmail):
    """3 hilos: 8d, 2d, 5d → emit en orden 8d, 5d, 2d."""
    mock_gmail["threads"] = [
        _make_thread(thread_id="t_2d", days_old=2.0),
        _make_thread(thread_id="t_8d", days_old=8.0),
        _make_thread(thread_id="t_5d", days_old=5.0),
    ]
    result = unread_email_thread_signal(_REF_NOW)
    assert len(result) == 3
    # First = oldest
    assert "t_8d" in result[0].dedup_key
    assert "t_5d" in result[1].dedup_key
    assert "t_2d" in result[2].dedup_key


def test_max_3_candidates(mock_gmail):
    """5 hilos → emit solo 3."""
    mock_gmail["threads"] = [
        _make_thread(thread_id=f"t{i}", days_old=2.0 + i)
        for i in range(5)
    ]
    result = unread_email_thread_signal(_REF_NOW)
    assert len(result) == 3


# ── Message format ───────────────────────────────────────────────────────────


def test_message_contains_emoji_and_when(mock_gmail):
    mock_gmail["threads"] = [_make_thread(
        sender="Maria <maria@example.com>",
        subject="urgente",
        snippet="por favor confirmar",
        days_old=4.0,
    )]
    result = unread_email_thread_signal(_REF_NOW)
    msg = result[0].message
    assert "📧" in msg
    assert "Maria" in msg
    assert "hace 4d" in msg
    assert "urgente" in msg
    assert "por favor confirmar" in msg


def test_message_yesterday_label(mock_gmail):
    """days_old < 2 → 'ayer'."""
    mock_gmail["threads"] = [_make_thread(days_old=1.5)]
    result = unread_email_thread_signal(_REF_NOW)
    assert "ayer" in result[0].message


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_gmail):
    mock_gmail["threads"] = [_make_thread(thread_id="abc123", days_old=3.0)]
    result = unread_email_thread_signal(_REF_NOW)
    assert result[0].dedup_key == "email-awaiting:abc123"


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "unread_email_thread" in names
