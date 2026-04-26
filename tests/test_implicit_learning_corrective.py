"""Tests para `rag_implicit_learning.corrective_paths`.

Cubre el flujo completo del implicit corrective_path inference:

- Happy path: 👎 + open de otra source en la ventana → corrective_path inferido.
- Skip: feedback ya tiene corrective_path → idempotente.
- Skip: feedback sin session_id → no se puede correlacionar.
- Skip: feedback sin paths_json → no hay top a comparar.
- Skip: no hay opens en la ventana → no se infiere nada.
- Skip: opened path == top path → user reaccionó al 👎 abriendo la #1.
- Window respetada: open fuera de la ventana NO cuenta.
- Session match estricto: open de OTRA session NO cuenta.
- in_top_k metadata: True si corrective ∈ paths_json, False si externa.
- dry_run: no muta la DB.
- only_feedback_id: filter aplica correctamente.
- Multi-feedback batch: procesa varios candidatos en una sola pasada.

Estructura de datos in-memory en sqlite3 ':memory:' para velocidad —
test isolation total, no toca la DB de producción.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from rag_implicit_learning.corrective_paths import (
    DEFAULT_WINDOW_SECONDS,
    infer_corrective_paths_from_behavior,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _seed_schema(conn: sqlite3.Connection) -> None:
    """Crea las tablas mínimas para el test (mismo schema que telemetry.db)."""
    conn.executescript(
        """
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT
        );
        CREATE TABLE rag_behavior (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            event TEXT NOT NULL,
            path TEXT,
            query TEXT,
            rank INTEGER,
            dwell_s REAL,
            extra_json TEXT,
            trace_id TEXT
        );
        """
    )


def _insert_feedback(
    conn: sqlite3.Connection,
    *,
    ts: str,
    turn_id: str = "t1",
    rating: int = -1,
    q: str = "test query",
    paths: list[str] | None = None,
    session: str | None = "web:abc123",
    existing_corrective: str | None = None,
) -> int:
    """Inserta un feedback row, devuelve el id."""
    paths_json = json.dumps(paths or [])
    extra: dict = {}
    if session:
        extra["session_id"] = session
    if existing_corrective:
        extra["corrective_path"] = existing_corrective
    extra_json = json.dumps(extra) if extra else None
    cur = conn.execute(
        "INSERT INTO rag_feedback (ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, turn_id, rating, q, paths_json, extra_json),
    )
    return cur.lastrowid


def _insert_behavior_open(
    conn: sqlite3.Connection,
    *,
    ts: str,
    path: str,
    session: str = "web:abc123",
    source: str = "web",
) -> int:
    """Inserta un evento `open` en rag_behavior."""
    extra_json = json.dumps({"session": session})
    cur = conn.execute(
        "INSERT INTO rag_behavior (ts, source, event, path, extra_json) "
        "VALUES (?, ?, 'open', ?, ?)",
        (ts, source, path, extra_json),
    )
    return cur.lastrowid


def _get_extra(conn: sqlite3.Connection, fb_id: int) -> dict:
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE id = ?", (fb_id,)
    ).fetchone()
    return json.loads(row[0] or "{}")


@pytest.fixture
def conn() -> sqlite3.Connection:
    """Connection in-memory con schema sembrado, una por test."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    _seed_schema(c)
    yield c
    c.close()


# ── Happy path ──────────────────────────────────────────────────────────────

def test_infers_corrective_when_user_opens_other_path_in_window(conn):
    """Caso canónico: 👎 + open de otra source dentro de 60s → corrective."""
    fb_id = _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "alex-pago.md", "other.md"],
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:23", path="alex-pago.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert result["n_inferred"] == 1
    assert result["n_candidates"] == 1
    assert len(result["updates"]) == 1
    assert result["updates"][0]["corrective_path"] == "alex-pago.md"
    assert result["updates"][0]["top_path"] == "wrong.md"
    assert result["updates"][0]["in_top_k"] is True

    # DB persistencia
    extra = _get_extra(conn, fb_id)
    assert extra["corrective_path"] == "alex-pago.md"
    assert extra["corrective_source"] == "implicit_behavior_inference"
    assert extra["corrective_in_top_k"] is True
    assert "corrective_inferred_at" in extra


def test_infers_corrective_for_path_outside_top_k_with_in_top_k_false(conn):
    """Si el user navegó a una nota que NO estaba en los top-k, también
    inferimos pero marcamos `in_top_k: false`."""
    fb_id = _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong-1.md", "wrong-2.md"]
    )
    _insert_behavior_open(
        conn, ts="2026-04-25T18:00:30", path="navegacion-externa.md"
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert result["n_inferred"] == 1
    assert result["updates"][0]["in_top_k"] is False
    extra = _get_extra(conn, fb_id)
    assert extra["corrective_path"] == "navegacion-externa.md"
    assert extra["corrective_in_top_k"] is False


# ── Idempotencia ────────────────────────────────────────────────────────────

def test_skips_feedback_already_with_corrective_path(conn):
    """Si el feedback ya tiene `corrective_path` (manual o de un run previo),
    NO lo procesamos otra vez."""
    fb_id = _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "right.md"],
        existing_corrective="manually-marked.md",
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="right.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert result["n_skip_already_corrective"] == 1
    assert result["n_inferred"] == 0
    extra = _get_extra(conn, fb_id)
    assert extra["corrective_path"] == "manually-marked.md"


def test_rerun_is_idempotent(conn):
    """Correr 2 veces seguidas: la segunda no infiere nada nuevo."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "right.md"],
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="right.md")

    first = infer_corrective_paths_from_behavior(conn, dry_run=False)
    second = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert first["n_inferred"] == 1
    assert second["n_inferred"] == 0
    assert second["n_skip_already_corrective"] == 1


# ── Skips defensivos ────────────────────────────────────────────────────────

def test_skips_feedback_without_session(conn):
    """Sin session_id en extra_json → no podemos correlacionar con behavior."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "right.md"],
        session=None,
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="right.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_session"] == 1


def test_skips_feedback_without_paths(conn):
    """Sin paths_json → no hay top a contradecir."""
    _insert_feedback(conn, ts="2026-04-25T18:00:00", paths=[])
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="some.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_paths"] == 1


def test_skips_when_no_open_in_window(conn):
    """Sin opens posteriores → el user abandonó, no podemos inferir."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "alex-pago.md"]
    )
    # No insertamos opens.

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1


def test_skips_when_user_opened_top_path(conn):
    """Si el user abre la #1 (probablemente curiosidad post-thumbsdown), NO
    inferimos corrective — no hay disconfirmación clara."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "other.md"]
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:10", path="wrong.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_opened_top"] == 1


# ── Boundary: ventana temporal ──────────────────────────────────────────────

def test_open_outside_window_does_not_count(conn):
    """Open 90s después del feedback NO cuenta con default window=60s."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "right.md"]
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:01:30", path="right.md")  # +90s

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1


def test_custom_window_extends_lookback(conn):
    """Con window=120s, el open a +90s sí cuenta."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "right.md"]
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:01:30", path="right.md")

    result = infer_corrective_paths_from_behavior(
        conn, window_seconds=120, dry_run=False
    )
    assert result["n_inferred"] == 1
    assert result["window_seconds"] == 120


def test_open_at_same_timestamp_does_not_count(conn):
    """Open al MISMO ts que el feedback no cuenta — usamos `ts > ?`,
    estricto. Evita race conditions de inserts en el mismo segundo."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "right.md"]
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:00", path="right.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0


# ── Boundary: session match ─────────────────────────────────────────────────

def test_open_from_other_session_does_not_count(conn):
    """Aislación por session: opens de OTRA session no se asocian al
    feedback aunque caigan en la ventana temporal."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "right.md"],
        session="web:session-A",
    )
    _insert_behavior_open(
        conn,
        ts="2026-04-25T18:00:15",
        path="right.md",
        session="web:session-B",
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1


# ── dry_run ─────────────────────────────────────────────────────────────────

def test_dry_run_does_not_mutate_db(conn):
    """En dry_run reportamos los updates pero NO los persistimos."""
    fb_id = _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["wrong.md", "right.md"]
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="right.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=True)
    assert result["n_inferred"] == 1
    assert result["dry_run"] is True
    assert len(result["updates"]) == 1

    # DB no mutada — extra_json sigue sin corrective_path.
    extra = _get_extra(conn, fb_id)
    assert "corrective_path" not in extra


# ── Filtering ───────────────────────────────────────────────────────────────

def test_only_feedback_id_processes_just_that_one(conn):
    """`only_feedback_id` filtra a un solo fb, ignora el resto."""
    fb1 = _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["w1.md", "r1.md"], turn_id="t1"
    )
    fb2 = _insert_feedback(
        conn, ts="2026-04-25T18:05:00", paths=["w2.md", "r2.md"], turn_id="t2"
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:15", path="r1.md")
    _insert_behavior_open(conn, ts="2026-04-25T18:05:15", path="r2.md")

    result = infer_corrective_paths_from_behavior(
        conn, dry_run=False, only_feedback_id=fb2
    )
    assert result["n_candidates"] == 1
    assert result["n_inferred"] == 1
    assert result["updates"][0]["feedback_id"] == fb2

    # fb1 no se tocó.
    extra1 = _get_extra(conn, fb1)
    assert "corrective_path" not in extra1


# ── Batch processing ────────────────────────────────────────────────────────

def test_processes_multiple_feedbacks_in_one_pass(conn):
    """Batch: 3 feedbacks negativos, 2 con opens válidos, 1 sin → 2 inferidos."""
    _insert_feedback(
        conn, ts="2026-04-25T18:00:00", paths=["w1.md", "r1.md"], turn_id="t1"
    )
    _insert_feedback(
        conn, ts="2026-04-25T18:05:00", paths=["w2.md", "r2.md"], turn_id="t2"
    )
    _insert_feedback(
        conn, ts="2026-04-25T18:10:00", paths=["w3.md", "r3.md"], turn_id="t3"
    )
    # Solo opens para los primeros dos.
    _insert_behavior_open(conn, ts="2026-04-25T18:00:20", path="r1.md")
    _insert_behavior_open(conn, ts="2026-04-25T18:05:30", path="r2.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_candidates"] == 3
    assert result["n_inferred"] == 2
    assert result["n_skip_no_open"] == 1


def test_first_open_wins_over_later_one(conn):
    """Si hay 2 opens en la ventana, el más cercano al feedback es el
    corrective. Garantiza determinismo."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        paths=["wrong.md", "first.md", "later.md"],
    )
    _insert_behavior_open(conn, ts="2026-04-25T18:00:10", path="first.md")
    _insert_behavior_open(conn, ts="2026-04-25T18:00:40", path="later.md")

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 1
    assert result["updates"][0]["corrective_path"] == "first.md"


# ── Defaults ────────────────────────────────────────────────────────────────

def test_default_window_is_60_seconds():
    """Sanity: el default no cambió silenciosamente. Si baja el default, el
    behavior post-feedback queda fuera; si sube, capturamos opens que ya
    no son reactivos al thumbsdown."""
    assert DEFAULT_WINDOW_SECONDS == 60
