"""Tests for the corrective_path prompt in the `rag chat` feedback loop
(2026-04-22).

When a user rates a turn negative (-/👎/`/mal`), the chat loop now offers
an optional prompt to mark which path was the correct one. The selection is
persisted as `corrective_path` in `rag_feedback.extra_json`, feeding the
reranker fine-tune pipeline (GC#2.B) with clean (query, positive, negative)
triplets.

Pre-2026-04-22 the prompt didn't exist — `rag_feedback` had 0 rows with
`corrective_path` in production, so the fine-tune regressed (−3.3pp chains
hit@5).

Tests here focus on the *selection logic* (parse numeric / parse free text /
skip) since the full chat loop is complex to exercise end-to-end. The
record_feedback plumbing is validated separately in test_feedback.py.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

import rag


# ── Helpers ──────────────────────────────────────────────────────────────────


def _candidate_paths_from_sources(last_sources: list[dict]) -> list[str]:
    """Replicate the candidate-build logic from the chat loop. Kept in sync
    with rag.py:~18444."""
    out: list[str] = []
    for m in last_sources:
        p = m.get("file", "")
        if p and "://" not in p and p not in out:
            out.append(p)
        if len(out) >= 5:
            break
    return out


# ── 1. candidate-build logic ─────────────────────────────────────────────────


def test_candidate_paths_skip_cross_source_native_ids():
    """Native cross-source ids (calendar://, whatsapp://, etc) no son paths
    vault-relativos y no deberían aparecer como opción de corrective_path."""
    srcs = [
        {"file": "01-Projects/note.md"},
        {"file": "calendar://foo/bar"},
        {"file": "02-Areas/other.md"},
        {"file": "whatsapp://chat/123"},
    ]
    cands = _candidate_paths_from_sources(srcs)
    assert cands == ["01-Projects/note.md", "02-Areas/other.md"]


def test_candidate_paths_dedup():
    """Mismo path repetido en múltiples chunks → una sola opción."""
    srcs = [
        {"file": "01-Projects/note.md"},
        {"file": "01-Projects/note.md"},
        {"file": "02-Areas/other.md"},
        {"file": "01-Projects/note.md"},
    ]
    cands = _candidate_paths_from_sources(srcs)
    assert cands == ["01-Projects/note.md", "02-Areas/other.md"]


def test_candidate_paths_cap_at_5():
    """Solo top-5 paths (suficiente elección; más opciones agregan ruido)."""
    srcs = [{"file": f"folder/note-{i}.md"} for i in range(10)]
    cands = _candidate_paths_from_sources(srcs)
    assert len(cands) == 5
    assert cands[0] == "folder/note-0.md"
    assert cands[-1] == "folder/note-4.md"


def test_candidate_paths_skip_empty_file():
    """Metas sin `file` (ej. graph chunks crudos) no son candidatos."""
    srcs = [
        {"file": ""},
        {"file": "good.md"},
        {},
    ]
    cands = _candidate_paths_from_sources(srcs)
    assert cands == ["good.md"]


def test_candidate_paths_empty_sources():
    assert _candidate_paths_from_sources([]) == []


# ── 2. selection parsing (numeric vs. free text vs. skip) ────────────────────


def _parse_answer(ans: str, candidates: list[str]) -> str | None:
    """Replicate the selection-parse logic. Kept in sync with rag.py:~18469."""
    ans = ans.strip()
    if not ans:
        return None
    if ans.isdigit():
        idx = int(ans) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
        return None  # out-of-range digit → treat as skip
    return ans  # free-text path


def test_parse_numeric_selection_first():
    cands = ["a.md", "b.md", "c.md"]
    assert _parse_answer("1", cands) == "a.md"


def test_parse_numeric_selection_last():
    cands = ["a.md", "b.md", "c.md"]
    assert _parse_answer("3", cands) == "c.md"


def test_parse_numeric_out_of_range():
    """'99' con 3 candidatos → skip (None) en vez de crashear."""
    cands = ["a.md", "b.md", "c.md"]
    assert _parse_answer("99", cands) is None


def test_parse_empty_skip():
    """Enter (string vacío) → skip."""
    cands = ["a.md"]
    assert _parse_answer("", cands) is None
    assert _parse_answer("   ", cands) is None


def test_parse_free_text_path():
    """Path que el usuario escribió manualmente (no está en candidates)."""
    cands = ["shown.md"]
    result = _parse_answer("03-Resources/actual.md", cands)
    assert result == "03-Resources/actual.md"


def test_parse_free_text_pregunta_abierta_marker():
    """Path que marca 'faltaba esta nota' — no validar que exista."""
    cands = ["other.md"]
    result = _parse_answer("@pregunta-abierta/falta.md", cands)
    assert result == "@pregunta-abierta/falta.md"


# ── 3. record_feedback call-time contract ────────────────────────────────────


def test_record_feedback_with_corrective_path_persists_extra_json(monkeypatch):
    """El corrective_path debe persistir en extra_json para que el tune
    lo lea via json_extract downstream."""
    # Use an in-memory collection so the test doesn't touch the real telemetry.db
    import sqlite3
    conn = sqlite3.connect(":memory:")
    rag._ensure_telemetry_tables(conn)

    # Stub _ragvec_state_conn to return our in-memory conn (context manager-compatible)
    class _FakeCtx:
        def __init__(self, c):
            self._c = c
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(rag, "_ragvec_state_conn", lambda: _FakeCtx(conn))

    rag.record_feedback(
        "turn-abc", -1, "que dice el vault sobre X",
        ["bad-path-shown.md"],
        corrective_path="good/correct.md",
        reason="corrective",
    )

    # Verify the row was inserted with the corrective_path embedded
    row = conn.execute(
        "SELECT rating, q, extra_json FROM rag_feedback WHERE turn_id = 'turn-abc'"
    ).fetchone()
    assert row is not None
    rating, q, extra_json = row
    assert rating == -1
    assert q == "que dice el vault sobre X"
    import json
    extra = json.loads(extra_json)
    assert extra.get("corrective_path") == "good/correct.md"
    assert extra.get("reason") == "corrective"


def test_record_feedback_without_corrective_path_omits_field(monkeypatch):
    """Sin corrective_path, la key NO debe aparecer en extra_json (evita
    queries downstream con `WHERE corrective_path IS NOT NULL` que fallan
    por string vacío).
    """
    import sqlite3
    conn = sqlite3.connect(":memory:")
    rag._ensure_telemetry_tables(conn)

    class _FakeCtx:
        def __init__(self, c):
            self._c = c
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(rag, "_ragvec_state_conn", lambda: _FakeCtx(conn))

    rag.record_feedback(
        "turn-plain", 1, "q ok", ["good.md"],
    )

    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE turn_id = 'turn-plain'"
    ).fetchone()
    # Si no había extras relevantes el campo puede ser None o un dict sin la key
    if row[0] is not None:
        import json
        extra = json.loads(row[0])
        assert "corrective_path" not in extra
