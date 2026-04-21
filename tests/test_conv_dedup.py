"""Tests for `_conv_dedup_window` — conversational dedup post-rerank.

Collapses WhatsApp/messages chunks from the same chat within a time
window, keeping the highest-scored. Non-WA sources pass through.
"""
from __future__ import annotations

import pytest

import rag


def _mk_pair(
    jid: str, first_ts: float, score: float,
    *, source: str = "whatsapp", file_suffix: str = "m",
) -> tuple:
    """Build a (candidate, expanded_text, score) tuple matching what
    retrieve()'s final_pairs looks like."""
    meta = {
        "source": source,
        "chat_jid": jid,
        "first_ts": first_ts,
        "file": f"{source}://{jid}/{file_suffix}",
    }
    candidate = ("id", meta, "raw_doc")
    return (candidate, "expanded_text", score)


# ── Basic dedup ─────────────────────────────────────────────────────────

def test_dedup_empty_input():
    assert rag._conv_dedup_window([]) == []


def test_dedup_collapses_same_chat_within_window():
    # 3 chunks from same chat, all within 30min → keep only highest-scored.
    pairs = [
        _mk_pair("ana@jid", 1000.0, score=3.0),
        _mk_pair("ana@jid", 1200.0, score=2.0),
        _mk_pair("ana@jid", 2500.0, score=1.8),  # 1500s from #1 < 1800s
    ]
    out = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(out) == 1
    assert out[0][2] == 3.0  # the highest-scored survivor


def test_dedup_keeps_chunks_outside_window():
    pairs = [
        _mk_pair("ana@jid", 1000.0, score=3.0),
        _mk_pair("ana@jid", 5000.0, score=2.0),   # 4000s apart > 1800 → keep
    ]
    out = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(out) == 2


def test_dedup_preserves_input_order():
    """Caller expects the list sorted by score desc; dedup must not reorder."""
    pairs = [
        _mk_pair("a@jid", 1000.0, score=3.0),
        _mk_pair("b@jid", 1000.0, score=2.5),
        _mk_pair("c@jid", 1000.0, score=2.0),
    ]
    out = rag._conv_dedup_window(pairs)
    assert [p[2] for p in out] == [3.0, 2.5, 2.0]


def test_dedup_different_chats_survive_together():
    pairs = [
        _mk_pair("ana@jid",   1000.0, score=3.0),
        _mk_pair("grupo@g",   1000.0, score=2.5),
        _mk_pair("juan@jid",  1000.0, score=2.0),
    ]
    out = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(out) == 3


# ── Non-WA passthrough ──────────────────────────────────────────────────

def test_dedup_vault_chunks_pass_through():
    # Multiple vault chunks with identical chat_jid (legacy) — should NOT
    # be deduped because source != whatsapp/messages.
    pairs = [
        _mk_pair("chat", 1000.0, score=3.0, source="vault"),
        _mk_pair("chat", 1100.0, score=2.5, source="vault"),
        _mk_pair("chat", 1200.0, score=2.0, source="vault"),
    ]
    out = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(out) == 3


def test_dedup_messages_source_also_applies():
    """The dedup rule covers `messages` (Apple iMessage/SMS) too — same
    conversational failure mode."""
    pairs = [
        _mk_pair("num@jid", 1000.0, score=3.0, source="messages"),
        _mk_pair("num@jid", 1100.0, score=2.0, source="messages"),
    ]
    out = rag._conv_dedup_window(pairs)
    assert len(out) == 1


def test_dedup_mixed_sources_independent():
    """WA chunks dedup amongst themselves; vault in the mix untouched."""
    pairs = [
        _mk_pair("chat@jid", 1000.0, score=3.0),                        # WA, keep
        _mk_pair("chat@jid", 1200.0, score=2.5),                        # WA, dropped
        _mk_pair("some/note.md", 1000.0, score=2.2, source="vault"),    # vault, pass
    ]
    out = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(out) == 2
    sources = [p[0][1]["source"] for p in out]
    assert sources == ["whatsapp", "vault"]


# ── Defensive ──────────────────────────────────────────────────────────

def test_dedup_missing_chat_jid_keeps_chunk():
    """WA chunk without chat_jid (shouldn't happen, but defend) → passthrough."""
    meta = {"source": "whatsapp", "first_ts": 1000.0, "file": "whatsapp://x"}
    candidate = ("id", meta, "doc")
    pairs = [(candidate, "exp", 2.0)]
    out = rag._conv_dedup_window(pairs)
    assert len(out) == 1


def test_dedup_missing_first_ts_keeps_chunk():
    meta = {"source": "whatsapp", "chat_jid": "a@jid", "file": "whatsapp://a"}
    candidate = ("id", meta, "doc")
    pairs = [(candidate, "exp", 2.0)]
    out = rag._conv_dedup_window(pairs)
    assert len(out) == 1


def test_dedup_invalid_first_ts_keeps_chunk():
    meta = {
        "source": "whatsapp", "chat_jid": "a@jid",
        "first_ts": "not-a-number", "file": "whatsapp://a",
    }
    candidate = ("id", meta, "doc")
    pairs = [(candidate, "exp", 2.0)]
    out = rag._conv_dedup_window(pairs)
    assert len(out) == 1


def test_dedup_custom_window_size():
    # Tighter window → fewer dedups
    pairs = [
        _mk_pair("chat", 1000.0, score=3.0),
        _mk_pair("chat", 1200.0, score=2.0),  # 200s apart
    ]
    # With window=100s both survive; with default they dedup.
    tight = rag._conv_dedup_window(pairs, window_s=100.0)
    assert len(tight) == 2
    loose = rag._conv_dedup_window(pairs, window_s=1800.0)
    assert len(loose) == 1
