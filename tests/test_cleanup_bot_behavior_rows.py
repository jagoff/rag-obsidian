"""Tests para `scripts/cleanup_bot_behavior_rows.py` — heurística post-hoc
para identificar rows bot-initiated en `rag_behavior` que pre-2026-04-28 se
loguearon con `source="cli"` hardcoded.

Cubre las 3 heurísticas:
  A. Orphan impressions (sin positive event compañero en ±N min).
  B. Eval set matches (queries.yaml).
  C. Short queries (<N chars).

Validación crítica: cada heurística debe NO tirar user-legítimas (queries
con click follow-up dentro de la ventana, queries largas, queries no en
eval set).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rag  # noqa: E402
import cleanup_bot_behavior_rows as cleanup  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    return tmp_path


def _seed(events: list[dict]) -> None:
    with rag._ragvec_state_conn() as conn:
        for ev in events:
            ev = {"ts": ev.get("ts") or datetime.now().isoformat(timespec="seconds"),
                  **ev}
            rag._sql_append_event(conn, "rag_behavior",
                                    rag._map_behavior_row(ev))


def _iso(offset_seconds: int) -> str:
    """ISO timestamp `offset_seconds` from now."""
    return (datetime.now() + timedelta(seconds=offset_seconds)).isoformat(
        timespec="seconds",
    )


# ── Heurística A: orphan impressions ────────────────────────────────────────


def test_heuristic_a_orphan_impressions_flags_isolated_rows(tmp_db):
    """Impressions sin positive companion en ±N min → flagged."""
    _seed([
        # Orphan: bot-initiated impression, ningún click después.
        {"source": "cli", "event": "impression",
         "query": "calendar event title", "path": "bot-target.md",
         "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert len(ids) == 1


def test_heuristic_a_skips_impressions_with_click_followup(tmp_db):
    """Impressions con positive event compañero (open/copy/save/kept) en la
    ventana NO se flagean — son user-legítimas."""
    _seed([
        # User searches → impression ts=-1800s, click ts=-1700s (100s after).
        {"source": "cli", "event": "impression",
         "query": "user query", "path": "user-note.md",
         "ts": _iso(-1800)},
        {"source": "cli", "event": "open",
         "query": "user query", "path": "user-note.md",
         "ts": _iso(-1700)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert ids == []


def test_heuristic_a_window_boundary(tmp_db):
    """Click fuera de la ventana (>60 min después de la impression) NO
    salva la impression — la heurística asume que clicks tardíos no son
    señal del mismo turn de retrieval."""
    # Impression a 3h ago, open a 30min ago = 2.5h DESPUÉS = fuera de
    # la window de 60min → orphaned → flagged.
    _seed([
        {"source": "cli", "event": "impression",
         "query": "old query", "path": "old.md",
         "ts": _iso(-3 * 3600)},  # 3h ago
        {"source": "cli", "event": "open",
         "query": "old query", "path": "old.md",
         "ts": _iso(-30 * 60)},   # 30 min ago = 2.5h after impression
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert len(ids) == 1, (
        "click 2.5h post-impression debería estar fuera de la 60min window"
    )


def test_heuristic_a_skips_non_cli_sources(tmp_db):
    """source="brief", "anticipate-*", "web", "whatsapp" NO entran al delete
    set — solo "cli" (que es donde la pollution histórica vive)."""
    _seed([
        {"source": "brief", "event": "impression",
         "query": "brief curated", "path": "brief.md",
         "ts": _iso(-3600)},
        {"source": "anticipate-calendar", "event": "impression",
         "query": "calendar event", "path": "calendar.md",
         "ts": _iso(-3600)},
        {"source": "web", "event": "impression",
         "query": "web user", "path": "web.md",
         "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert ids == [], "non-cli sources should be left alone"


def test_heuristic_a_skips_non_impression_events(tmp_db):
    """Events distintos a `impression` (open/copy/save/explore/kept/deleted)
    son señal real, NO se tocan."""
    _seed([
        {"source": "cli", "event": "open", "query": "q",
         "path": "open.md", "ts": _iso(-3600)},
        {"source": "cli", "event": "copy", "query": "q",
         "path": "copy.md", "ts": _iso(-3600)},
        {"source": "cli", "event": "kept", "query": "q",
         "path": "kept.md", "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert ids == []


def test_heuristic_a_skips_empty_query(tmp_db):
    """Rows sin query (NULL o "") no se tocan — defensive (raros pero
    posibles)."""
    _seed([
        {"source": "cli", "event": "impression", "query": "",
         "path": "empty.md", "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=60,
        )
    assert ids == []


# ── Heurística B: eval set matches ──────────────────────────────────────────


def test_heuristic_b_matches_eval_questions(tmp_db, tmp_path):
    """Queries cuyo texto matchea (lower + whitespace collapse) cualquier
    question del eval set son flagged."""
    eval_yaml = tmp_path / "queries.yaml"
    eval_yaml.write_text(
        "singles:\n"
        "  - question: 'qué tengo sobre postura?'\n"
        "  - question: 'reglas de la oficina'\n",
        encoding="utf-8",
    )
    _seed([
        {"source": "cli", "event": "impression",
         "query": "QUÉ tengo SOBRE   postura?",  # case + whitespace variants
         "path": "p.md", "ts": _iso(-3600)},
        {"source": "cli", "event": "impression",
         "query": "user-genuine query no-eval",
         "path": "u.md", "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_b_eval_queries(conn, eval_yaml)
    assert len(ids) == 1


def test_heuristic_b_missing_yaml_returns_empty(tmp_db, tmp_path):
    """Si queries.yaml no existe, return [] sin crash."""
    _seed([
        {"source": "cli", "event": "impression", "query": "q",
         "path": "p.md", "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_b_eval_queries(
            conn, tmp_path / "doesnt-exist.yaml",
        )
    assert ids == []


# ── Heurística C: short queries ─────────────────────────────────────────────


def test_heuristic_c_short_queries(tmp_db):
    """Queries < N chars son flagged. Anticipatory-calendar manda títulos
    cortos como queries (ej. 'almuerzo')."""
    _seed([
        {"source": "cli", "event": "impression",
         "query": "almuerzo", "path": "p1.md",  # 8 chars
         "ts": _iso(-3600)},
        {"source": "cli", "event": "impression",
         "query": "user query mucho mas larga", "path": "p2.md",  # 26 chars
         "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        ids = cleanup._heuristic_c_short_queries(conn, min_chars=10)
    assert len(ids) == 1


def test_normalise():
    """Helper de normalización: lowercase + whitespace collapse."""
    assert cleanup._normalise("HOLA   Mundo") == "hola mundo"
    assert cleanup._normalise("  trim  ") == "trim"
    assert cleanup._normalise("") == ""
    assert cleanup._normalise(None) == ""  # type: ignore


# ── Backup before delete ─────────────────────────────────────────────────────


def test_backup_rows_writes_jsonl(tmp_db, tmp_path, monkeypatch):
    """`_backup_rows` dumpea las rows a un JSONL antes del DELETE para
    rollback manual si fuera necesario."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _seed([
        {"source": "cli", "event": "impression",
         "query": "q1", "path": "a.md", "ts": _iso(-3600)},
        {"source": "cli", "event": "impression",
         "query": "q2", "path": "b.md", "ts": _iso(-3600)},
    ])
    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM rag_behavior WHERE event = 'impression'"
        ).fetchall()
        ids = [r[0] for r in rows]
        backup_path = cleanup._backup_rows(conn, ids)
    assert backup_path.is_file()
    lines = backup_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    import json
    parsed = [json.loads(line) for line in lines]
    queries = sorted(p["query"] for p in parsed)
    assert queries == ["q1", "q2"]
