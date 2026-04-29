"""Tests para la rama paráfrasis-fallback de
`rag_implicit_learning.corrective_paths.infer_corrective_paths_from_behavior`.

Esta rama (introducida 2026-04-29 junto al bump de window 60→600s)
captura un corrective_path implícito cuando el user dio 👎 a una
respuesta y, en lugar de abrir otra nota, simplemente reformuló la
pregunta. Si la nueva corrida del retrieve devolvió un top-1 distinto
con score decente, asumimos que ESE es el path que el ranker original
debería haber elegido.

Casos cubiertos:
- Happy path: paráfrasis con top-1 distinto + score ≥0.5 → corrective
  con `corrective_source = "implicit_paraphrase_inference"`.
- Mismo top path: la paráfrasis insiste en el mismo top → no es
  corrective (el ranker no cambió de opinión).
- Top score bajo (< 0.5) → skip por confianza insuficiente.
- Query distinto sin overlap → no es paráfrasis, no se infiere.
- Out of window: paráfrasis a +700s → fuera del default 600 → no se
  infiere; con override de window sí.
- Backwards compat: si hay open Y paráfrasis válida, gana el open
  (señal más fuerte) y el `corrective_source` queda como
  `implicit_behavior_inference`.
- `n_inferred_via_paraphrase` métrica en el resultado.

Mimetiza el style de [`test_implicit_learning_corrective.py`](test_implicit_learning_corrective.py)
y [`test_implicit_learning_requery.py`](test_implicit_learning_requery.py): in-memory sqlite, schema mínimo
sembrado, fixtures por test.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from rag_implicit_learning.corrective_paths import (
    DEFAULT_PARAPHRASE_TOP_SCORE_MIN,
    DEFAULT_WINDOW_SECONDS,
    infer_corrective_paths_from_behavior,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _seed_schema(conn: sqlite3.Connection) -> None:
    """Schema mínimo: feedback + behavior + queries.

    Es el mismo set que `_TELEMETRY_DDL` define en producción pero
    reducido a las columnas que el inference usa. Abre/cierra rápido
    en `:memory:`.
    """
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
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            q TEXT NOT NULL,
            session TEXT,
            mode TEXT,
            top_score REAL,
            t_retrieve REAL,
            t_gen REAL,
            answer_len INTEGER,
            paths_json TEXT,
            extra_json TEXT
        );
        """
    )


def _insert_feedback(
    conn: sqlite3.Connection,
    *,
    ts: str,
    q: str = "qué sabes de Grecia",
    paths: list[str] | None = None,
    session: str | None = "web:abc123",
    rating: int = -1,
    turn_id: str = "t1",
) -> int:
    """Inserta un row en `rag_feedback`. Devuelve el id."""
    paths_json = json.dumps(paths or [])
    extra: dict = {}
    if session:
        extra["session_id"] = session
    extra_json = json.dumps(extra) if extra else None
    cur = conn.execute(
        "INSERT INTO rag_feedback (ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, turn_id, rating, q, paths_json, extra_json),
    )
    return cur.lastrowid


def _insert_query(
    conn: sqlite3.Connection,
    *,
    ts: str,
    q: str,
    paths: list[str],
    top_score: float,
    session: str = "web:abc123",
    cmd: str = "chat",
) -> int:
    """Inserta un row en `rag_queries` simulando una corrida del retrieve.

    `paths_json` y `top_score` son los campos que el inference lee
    para determinar si la corrida follow-up devolvió un candidate
    útil.
    """
    paths_json = json.dumps(paths)
    cur = conn.execute(
        "INSERT INTO rag_queries "
        "(ts, cmd, q, session, top_score, paths_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, cmd, q, session, top_score, paths_json),
    )
    return cur.lastrowid


def _insert_behavior_open(
    conn: sqlite3.Connection,
    *,
    ts: str,
    path: str,
    session: str = "web:abc123",
) -> int:
    """Inserta un evento `open` en `rag_behavior` (mismo formato que
    test_implicit_learning_corrective.py)."""
    extra_json = json.dumps({"session": session})
    cur = conn.execute(
        "INSERT INTO rag_behavior (ts, source, event, path, extra_json) "
        "VALUES (?, 'web', 'open', ?, ?)",
        (ts, path, extra_json),
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


def test_paraphrase_with_higher_score_top_becomes_corrective(conn):
    """Caso canónico: 👎 + paráfrasis follow-up con top-1 distinto y
    `top_score >= 0.5` dentro del window → corrective_path inferido
    con source `implicit_paraphrase_inference`."""
    fb_id = _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md", "alex-pago.md"],
    )
    # Misma session, +30s después, paráfrasis (overlap "Grecia"), top-1
    # distinto al original, score > 0.5.
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md", "other.md"],
        top_score=0.72,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert result["n_candidates"] == 1
    assert result["n_inferred"] == 1
    assert result["n_inferred_via_paraphrase"] == 1
    assert len(result["updates"]) == 1
    update = result["updates"][0]
    assert update["corrective_path"] == "right-grecia.md"
    assert update["top_path"] == "wrong-grecia.md"
    assert update["corrective_source"] == "implicit_paraphrase_inference"
    # in_top_k=False: el path correcto NO estaba en los paths_json del
    # feedback original (signal de exploración fuera del top-k mostrado).
    assert update["in_top_k"] is False

    extra = _get_extra(conn, fb_id)
    assert extra["corrective_path"] == "right-grecia.md"
    assert extra["corrective_source"] == "implicit_paraphrase_inference"
    assert extra["corrective_in_top_k"] is False
    assert "corrective_inferred_at" in extra


def test_paraphrase_top_already_in_original_paths_marks_in_top_k(conn):
    """Si la paráfrasis devuelve un path que SÍ estaba en los top-k del
    feedback original, marcamos `in_top_k=True` (el ranker tenía la nota
    en el top-k pero no la priorizó)."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="cuánto debe Alex de la macbook",
        paths=["moka-foda.md", "alex-pago.md", "other.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="cuánto me debe Alex de macbook",
        paths=["alex-pago.md", "moka-foda.md"],  # top-1 ahora es alex-pago
        top_score=0.81,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 1
    assert result["updates"][0]["corrective_path"] == "alex-pago.md"
    assert result["updates"][0]["in_top_k"] is True


# ── Mismo top path: no hay corrección ──────────────────────────────────────


def test_paraphrase_with_same_top_does_not_infer(conn):
    """Si la paráfrasis cae en el MISMO top que el original → el ranker
    no cambió de opinión → no es corrective real."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md", "other.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["wrong-grecia.md", "other.md"],  # Mismo top-1.
        top_score=0.81,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_inferred_via_paraphrase"] == 0
    # Como no hubo opens NI paráfrasis útil, el bucket "no_open"
    # incrementa (semánticamente "no encontramos signal post-feedback").
    assert result["n_skip_no_open"] == 1


# ── Top score bajo: confianza insuficiente ─────────────────────────────────


def test_paraphrase_with_low_top_score_is_skipped(conn):
    """Paráfrasis con `top_score < threshold` → no confiamos en su top-1, skip.

    2026-04-29: threshold bajado de 0.5 a 0.1 — ahora el "low score" es
    <0.1 (rangos típicos del cross-encoder en queries fallidas).
    """
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.05,  # Por debajo del threshold default 0.1.
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1


def test_paraphrase_with_null_top_score_is_skipped(conn):
    """`top_score IS NULL` → asumimos 0.0, queda bajo threshold, skip."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, session, top_score, paths_json) "
        "VALUES (?, 'chat', ?, 'web:abc123', NULL, ?)",
        (
            "2026-04-25T18:00:30",
            "dame info sobre Grecia",
            json.dumps(["right-grecia.md"]),
        ),
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0


def test_paraphrase_threshold_constant_is_01():
    """Sanity sobre el threshold default. Si baja, agrandamos el dataset
    pero metemos más ruido al fine-tune. Si sube, el dataset se achica.
    Ambos cambios deberían ser deliberados.

    2026-04-29: bajado de 0.5 a 0.1 después de validar contra DB live.
    El cross-encoder bge-reranker-v2-m3 produce scores muy bajos (<0.15)
    para queries de WhatsApp/voz por su calibración absoluta — con 0.5
    ningún paraphrase WA pasaba el gate. 0.1 deja pasar matches con
    overlap léxico fuerte sin abrir compuertas para top_score~0.
    """
    assert DEFAULT_PARAPHRASE_TOP_SCORE_MIN == 0.1


# ── No es paraphrase: distinto tema ─────────────────────────────────────────


def test_unrelated_followup_query_is_not_a_paraphrase(conn):
    """Follow-up query con tema completamente distinto → `is_paraphrase`
    devuelve False, no se infiere."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="cuál es la capital de Italia",  # Sin overlap con "Grecia".
        paths=["roma.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1


# ── Boundary: ventana temporal ──────────────────────────────────────────────


def test_paraphrase_outside_default_window_is_skipped(conn):
    """Paráfrasis a +700s del feedback → fuera del default 600 → skip."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:11:40",  # +700s = 11min 40s, fuera del default.
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["window_seconds"] == DEFAULT_WINDOW_SECONDS


def test_paraphrase_inside_extended_window_is_picked_up(conn):
    """Con window=900 explícito, una paráfrasis a +700s sí cuenta."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:11:40",  # +700s.
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(
        conn, window_seconds=900, dry_run=False
    )
    assert result["n_inferred"] == 1
    assert result["n_inferred_via_paraphrase"] == 1
    assert result["window_seconds"] == 900


def test_paraphrase_in_other_session_does_not_count(conn):
    """Aislación por session: una paráfrasis en OTRA session no se
    asocia al feedback aunque caiga en la ventana."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
        session="web:session-A",
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
        session="web:session-B",
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0


# ── Backwards compat: opens gana sobre paraphrase ───────────────────────────


def test_open_wins_when_both_open_and_paraphrase_present(conn):
    """Cuando hay Y open Y paráfrasis válida, la rama opens-based
    (señal más fuerte) gana — el `corrective_source` queda como
    `implicit_behavior_inference`, no `paraphrase`."""
    fb_id = _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md", "open-pick.md"],
    )
    # Open dentro del window: este debería ganar.
    _insert_behavior_open(
        conn, ts="2026-04-25T18:00:15", path="open-pick.md"
    )
    # Paráfrasis válida también dentro del window con OTRO top-1 — pero
    # el opens-based ya cerró el caso, esta nunca se evalúa.
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["paraphrase-pick.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert result["n_inferred"] == 1
    assert result["n_inferred_via_paraphrase"] == 0
    assert result["updates"][0]["corrective_path"] == "open-pick.md"
    assert result["updates"][0]["corrective_source"] == "implicit_behavior_inference"
    extra = _get_extra(conn, fb_id)
    assert extra["corrective_path"] == "open-pick.md"
    assert extra["corrective_source"] == "implicit_behavior_inference"


def test_paraphrase_is_skipped_when_opens_match_only_top(conn):
    """Si los opens existen pero son TODOS al top-path original (no
    descalifica el ranking), entramos al bucket `n_skip_opened_top` y
    NO disparamos la rama paráfrasis (esa solo se prueba cuando NO hay
    opens en absoluto). Decisión de diseño: una vez que el user abrió
    el top, asumimos curiosidad/confirmación, no contradicción."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md", "other.md"],
    )
    # Open al MISMO top → entra a `n_skip_opened_top`.
    _insert_behavior_open(
        conn, ts="2026-04-25T18:00:10", path="wrong-grecia.md"
    )
    # Paráfrasis válida que SÍ daría un corrective si entráramos a la
    # rama 2 — pero no entramos (rama 1 ya consumió el caso).
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_opened_top"] == 1
    assert result["n_inferred_via_paraphrase"] == 0


# ── Multi-candidate: primer match útil gana ────────────────────────────────


def test_first_paraphrase_in_window_wins(conn):
    """Si hay 2 paráfrasis follow-up en la ventana, la PRIMERA en orden
    temporal con top-1 distinto + score OK gana (determinismo)."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:20",
        q="dame info sobre Grecia",
        paths=["first-pick.md"],
        top_score=0.78,
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:50",
        q="contame de Grecia antigua",
        paths=["later-pick.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 1
    assert result["updates"][0]["corrective_path"] == "first-pick.md"


def test_skips_low_score_paraphrase_then_picks_next_valid(conn):
    """Si la 1ra paráfrasis tiene score bajo (skip) y la 2da pasa el
    threshold, usamos la 2da — no nos quedamos colgados con la 1ra."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:20",
        q="dame info sobre Grecia",
        paths=["bad-confidence.md"],
        top_score=0.05,  # Por debajo del threshold (0.1 default 2026-04-29).
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:50",
        q="contame de Grecia antigua",
        paths=["good-confidence.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=False)
    assert result["n_inferred"] == 1
    assert result["updates"][0]["corrective_path"] == "good-confidence.md"
    assert result["updates"][0]["corrective_source"] == "implicit_paraphrase_inference"


# ── dry_run via paráfrasis ──────────────────────────────────────────────────


def test_dry_run_via_paraphrase_does_not_mutate(conn):
    """En dry_run la rama paráfrasis reporta el update pero no persiste."""
    fb_id = _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
    )

    result = infer_corrective_paths_from_behavior(conn, dry_run=True)
    assert result["dry_run"] is True
    assert result["n_inferred"] == 1
    assert result["n_inferred_via_paraphrase"] == 1
    extra = _get_extra(conn, fb_id)
    assert "corrective_path" not in extra


# ── Idempotencia ────────────────────────────────────────────────────────────


def test_paraphrase_inference_is_idempotent(conn):
    """Re-correr el inferencer no re-procesa feedbacks ya marcados via
    paráfrasis (skipea por `corrective_path` already present)."""
    _insert_feedback(
        conn,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:30",
        q="dame info sobre Grecia",
        paths=["right-grecia.md"],
        top_score=0.85,
    )

    first = infer_corrective_paths_from_behavior(conn, dry_run=False)
    second = infer_corrective_paths_from_behavior(conn, dry_run=False)

    assert first["n_inferred_via_paraphrase"] == 1
    assert second["n_inferred"] == 0
    assert second["n_skip_already_corrective"] == 1


# ── Defensa: missing rag_queries table ─────────────────────────────────────


def test_handles_missing_rag_queries_table_silently():
    """Si la conn no tiene `rag_queries` (DB minimal-schema, edge case),
    el fallback hace silent-fail y NO levanta excepción — el feedback
    queda sin corrective y se reintenta en el próximo run."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    # Schema sin rag_queries.
    c.executescript(
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
            extra_json TEXT
        );
        """
    )
    _insert_feedback(
        c,
        ts="2026-04-25T18:00:00",
        q="qué sabes de Grecia",
        paths=["wrong-grecia.md"],
    )
    # No opens, no rag_queries → fallback corre, hace silent-fail por
    # tabla missing, devuelve None → bucket no_open.
    result = infer_corrective_paths_from_behavior(c, dry_run=False)
    assert result["n_inferred"] == 0
    assert result["n_skip_no_open"] == 1
    c.close()
