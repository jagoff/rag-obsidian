"""Tests for the `question_awaiting` Anticipatory Agent signal.

Cubre:
1. No preguntas en tabla → []
2. Pregunta hace 2 días → [] (below min 3d)
3. Pregunta hace 5 días sin respuesta → emit
4. Pregunta hace 5 días CON respuesta posterior del user → []
5. Score escala con edad (3d→~0.21, 7d→0.5, 14d+→1.0)
6. Multiple preguntas → max 2, oldest first
7. dedup_key estable cross-runs
8. Tabla no existe → silent-fail []
9. Fila posterior del chat pero NO del user='me' → igual "awaiting"
10. Registry check (signal registrada en SIGNALS + _ANTICIPATE_SIGNALS)
11. Preview corto no rompe el message

El signal consulta SQL (rag_wa_tasks). Aislamos el telemetry DB con
`monkeypatch.setattr(rag, "DB_PATH", tmp_path)` y creamos la tabla con
el schema extendido (kind, source_chat, message_preview, user) que el
signal espera.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import rag
from rag_anticipate.signals.question_awaiting import question_awaiting_signal


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path + crea la tabla rag_wa_tasks con
    el schema extendido que el signal espera (kind, source_chat,
    message_preview, user). Retorna tmp_path para eventual inspección.

    El signal corre con `_ragvec_state_conn()` directo; no necesitamos
    mockear nada más — el monkeypatch del DB_PATH hace que todas las
    conns apunten al tmp_path.
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)

    # Cargar sqlite-vec collection para que `_ensure_telemetry_tables`
    # corra al primer open; igual vamos a sobrescribir rag_wa_tasks con
    # nuestro schema extendido manualmente.
    from rag import SqliteVecClient
    client = SqliteVecClient(path=str(db_path))
    client.get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )

    with rag._ragvec_state_conn() as conn:
        # Drop la tabla que _ensure_telemetry_tables haya creado con el
        # schema "production" (id, ts, since, chats, items, path,
        # extra_json) y recrearla con el schema extendido que el signal
        # consulta.
        try:
            conn.execute("DROP TABLE IF EXISTS rag_wa_tasks")
        except Exception:
            pass
        conn.execute(
            "CREATE TABLE rag_wa_tasks ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL,"
            " kind TEXT,"
            " source_chat TEXT,"
            " message_preview TEXT,"
            " user TEXT"
            ")"
        )
    return tmp_path


def _insert(ts_dt: datetime, *, kind: str, chat: str,
            preview: str = "", user: str = "them") -> None:
    """Inserta una fila en rag_wa_tasks con el schema extendido."""
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_wa_tasks (ts, kind, source_chat, message_preview, user) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts_dt.isoformat(timespec="seconds"), kind, chat, preview, user),
        )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_signal_registered_in_registry():
    """Sanity: el decorator registra la signal en SIGNALS."""
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "question_awaiting" in names


def test_signal_in_anticipate_tuple():
    """La signal aparece en el tuple global `rag._ANTICIPATE_SIGNALS`."""
    names = [n for (n, _fn) in rag._ANTICIPATE_SIGNALS]
    assert "question_awaiting" in names


def test_empty_table_returns_empty(state_db):
    """Tabla sin filas → []."""
    out = question_awaiting_signal(datetime.now())
    assert out == []


def test_question_below_min_age_returns_empty(state_db):
    """Pregunta hace 2 días (<3d threshold) → []."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    _insert(now - timedelta(days=2), kind="question", chat="chat_A",
            preview="¿Venís mañana?")
    out = question_awaiting_signal(now)
    assert out == []


def test_question_exactly_3_days_emits(state_db):
    """Pregunta hace exactamente 3 días sin respuesta → emit con score ~0.21."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    _insert(now - timedelta(days=3), kind="question", chat="chat_A",
            preview="¿Cuando nos vemos?")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    c = out[0]
    assert c.kind == "anticipate-question_awaiting"
    # 3/14 ≈ 0.2143
    assert c.score == pytest.approx(3 / 14.0, abs=0.01)
    assert c.snooze_hours == 168


def test_question_5d_no_reply_emits(state_db):
    """Pregunta hace 5 días sin respuesta del user → emit."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="¿Qué onda el proyecto X?")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    c = out[0]
    assert c.score == pytest.approx(5 / 14.0, abs=0.01)
    assert "pregunta sin respuesta" in c.message
    assert "5 días" in c.message
    assert "¿Qué onda el proyecto X?" in c.message
    assert c.dedup_key == f"awaiting:chat_A:{q_ts.date().isoformat()}"


def test_question_5d_with_reply_returns_empty(state_db):
    """Pregunta hace 5 días pero el user respondió 2 días después → []."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="¿Venís al asado?")
    # User respondió 2 días después de la pregunta (3 días atrás).
    _insert(q_ts + timedelta(days=2), kind="fact", chat="chat_A",
            preview="Dale, voy", user="me")
    out = question_awaiting_signal(now)
    assert out == []


def test_reply_from_other_user_does_not_count(state_db):
    """Respuesta posterior pero NO del user='me' → sigue awaiting."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="¿Vamos al cine?")
    # El mismo contact sigue hablando, pero NO es 'me' → no cuenta como respuesta.
    _insert(q_ts + timedelta(days=1), kind="fact", chat="chat_A",
            preview="dale avisame", user="them")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    assert out[0].dedup_key == f"awaiting:chat_A:{q_ts.date().isoformat()}"


def test_reply_outside_window_does_not_count(state_db):
    """Respuesta del user PERO después de la ventana de 3 días → awaiting.

    Si el user contestó recién al día 4 (después de la ventana de reply
    de 3d), la pregunta cuenta como awaiting. Edge-case-defensivo: esto
    es lo que diseñamos — el user tardó demasiado, igual vale recordárselo.
    """
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=7)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="¿Confirmás?")
    # Reply al día 4 (fuera de la ventana de 3d).
    _insert(q_ts + timedelta(days=4), kind="fact", chat="chat_A",
            preview="listo", user="me")
    out = question_awaiting_signal(now)
    assert len(out) == 1


def test_score_scales_with_age(state_db):
    """Score sube linealmente con la edad de la pregunta."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    # 14 días ≥ cap → score 1.0
    _insert(now - timedelta(days=14), kind="question", chat="chat_A",
            preview="vieja")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    assert out[0].score == pytest.approx(1.0, abs=0.01)


def test_score_capped_at_1(state_db):
    """Preguntas más viejas que 14d igual saturan a 1.0."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    # Justo dentro de la ventana de 14d (13.9 días).
    _insert(now - timedelta(days=13, hours=23), kind="question", chat="chat_A",
            preview="casi al límite")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    assert out[0].score <= 1.0
    assert out[0].score == pytest.approx(13 / 14.0, abs=0.05)


def test_question_outside_lookback_window_ignored(state_db):
    """Pregunta hace 20 días (>14d lookback) → ignorada, []."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    _insert(now - timedelta(days=20), kind="question", chat="chat_A",
            preview="prehistórica")
    out = question_awaiting_signal(now)
    assert out == []


def test_multiple_questions_max_2_oldest_first(state_db):
    """Con 5 preguntas awaiting, emite solo las 2 más viejas en orden."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    # 5 preguntas awaiting con edades distintas.
    _insert(now - timedelta(days=3), kind="question", chat="chat_C",
            preview="nueva")
    _insert(now - timedelta(days=4), kind="question", chat="chat_D",
            preview="cuarta")
    _insert(now - timedelta(days=10), kind="question", chat="chat_A",
            preview="vieja-1")
    _insert(now - timedelta(days=8), kind="question", chat="chat_B",
            preview="vieja-2")
    _insert(now - timedelta(days=5), kind="question", chat="chat_E",
            preview="intermedia")

    out = question_awaiting_signal(now)
    assert len(out) == 2
    # Primera = más vieja (10 días)
    assert "vieja-1" in out[0].message
    # Segunda = 8 días
    assert "vieja-2" in out[1].message
    # Scores descendentes (más vieja → score más alto).
    assert out[0].score >= out[1].score


def test_dedup_key_stable_across_runs(state_db):
    """Dos runs seguidos producen el mismo dedup_key (idempotente)."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="estable?")
    out1 = question_awaiting_signal(now)
    out2 = question_awaiting_signal(now)
    assert len(out1) == 1
    assert len(out2) == 1
    assert out1[0].dedup_key == out2[0].dedup_key
    # Formato esperado
    expected = f"awaiting:chat_A:{q_ts.date().isoformat()}"
    assert out1[0].dedup_key == expected


def test_dedup_key_differs_per_chat(state_db):
    """Dos preguntas en chats distintos → dedup_keys distintos."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A", preview="a")
    _insert(q_ts, kind="question", chat="chat_B", preview="b")
    out = question_awaiting_signal(now)
    assert len(out) == 2
    keys = {c.dedup_key for c in out}
    assert len(keys) == 2


def test_table_does_not_exist_silent_fails(tmp_path, monkeypatch):
    """Si la tabla rag_wa_tasks no existe → signal retorna [] sin tirar."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    from rag import SqliteVecClient
    SqliteVecClient(path=str(db_path)).get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )
    # Explícitamente DROP la tabla si `_ensure_telemetry_tables` la creó.
    with rag._ragvec_state_conn() as conn:
        try:
            conn.execute("DROP TABLE IF EXISTS rag_wa_tasks")
        except Exception:
            pass

    out = question_awaiting_signal(datetime.now())
    assert out == []


def test_production_schema_silent_fails(tmp_path, monkeypatch):
    """Con el schema de producción (sin columnas `kind`, `source_chat`, etc.)
    el SELECT tira OperationalError → signal retorna []."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    from rag import SqliteVecClient
    SqliteVecClient(path=str(db_path)).get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )
    # Tabla con el schema "producción" (no tiene kind/source_chat/user).
    with rag._ragvec_state_conn() as conn:
        conn.execute("DROP TABLE IF EXISTS rag_wa_tasks")
        conn.execute(
            "CREATE TABLE rag_wa_tasks ("
            " id INTEGER PRIMARY KEY,"
            " ts TEXT,"
            " since TEXT,"
            " chats INTEGER,"
            " items INTEGER,"
            " path TEXT,"
            " extra_json TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO rag_wa_tasks (ts, since, chats, items) VALUES (?, ?, ?, ?)",
            ("2026-05-15T10:00:00", "2026-05-15T09:00:00", 3, 10),
        )

    out = question_awaiting_signal(datetime.now())
    assert out == []


def test_candidate_shape(state_db):
    """Validación de shape del AnticipatoryCandidate retornado."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    _insert(now - timedelta(days=5), kind="question", chat="chat_A",
            preview="validación shape")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    c = out[0]
    assert isinstance(c.kind, str) and c.kind.startswith("anticipate-")
    assert isinstance(c.score, float)
    assert 0.0 <= c.score <= 1.0
    assert isinstance(c.message, str) and c.message
    assert isinstance(c.dedup_key, str) and c.dedup_key.startswith("awaiting:")
    assert isinstance(c.snooze_hours, int)
    assert c.snooze_hours == 168
    assert isinstance(c.reason, str)


def test_empty_preview_does_not_crash(state_db):
    """Pregunta con preview vacío o NULL → no rompe, message sigue válido."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    _insert(now - timedelta(days=5), kind="question", chat="chat_A",
            preview="")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    assert out[0].message  # string no vacío
    assert "sin preview" in out[0].message or "pregunta sin respuesta" in out[0].message


def test_long_preview_is_truncated(state_db):
    """Preview muy largo se trunca a 120 chars en el message."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    long_preview = "x" * 500
    _insert(now - timedelta(days=5), kind="question", chat="chat_A",
            preview=long_preview)
    out = question_awaiting_signal(now)
    assert len(out) == 1
    # El message incluye el preview truncado, no los 500 chars completos.
    assert "x" * 500 not in out[0].message
    # Pero sí incluye al menos 120 chars de 'x'.
    assert "x" * 120 in out[0].message


def test_question_from_other_chat_does_not_block(state_db):
    """Reply del user en OTRO chat NO debe contar como respuesta a la
    pregunta del chat_A (scope por source_chat)."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    q_ts = now - timedelta(days=5)
    _insert(q_ts, kind="question", chat="chat_A",
            preview="¿chat A pregunta?")
    # El user respondió en chat_B, no en chat_A.
    _insert(q_ts + timedelta(days=1), kind="fact", chat="chat_B",
            preview="respondí en otro chat", user="me")
    out = question_awaiting_signal(now)
    assert len(out) == 1
    assert "chat_A" in out[0].dedup_key


def test_kind_filter_excludes_non_questions(state_db):
    """Filas con kind≠'question' se ignoran aunque sean viejas."""
    now = datetime(2026, 5, 20, 10, 0, 0)
    # Solo commitments/facts, no questions.
    _insert(now - timedelta(days=5), kind="commitment", chat="chat_A",
            preview="me comprometí a algo")
    _insert(now - timedelta(days=10), kind="fact", chat="chat_A",
            preview="un hecho")
    _insert(now - timedelta(days=7), kind="action", chat="chat_A",
            preview="una acción")
    out = question_awaiting_signal(now)
    assert out == []
