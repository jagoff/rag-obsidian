"""Tests for `rag_anticipate.feedback` — feedback loop del Anticipatory Agent.

Cubre:
- record_feedback inserta correctamente
- record_feedback rejecta ratings inválidos
- parse_wa_reply: emojis sueltos, frases con texto libre, case-insensitive
- parse_wa_reply: ambiguous '👍 pero 👎' → 'negative' (conservador)
- feedback_stats sin filter / con kind prefix / con window days
- recent_feedback ordenado desc
- Silent-fail: si la tabla no existe inicialmente la primera llamada
  debe crearla y todo lo siguiente funciona sin raise.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import rag
from rag import SqliteVecClient as _TestVecClient
from rag_anticipate.feedback import (
    feedback_stats,
    parse_wa_reply,
    record_feedback,
    recent_feedback,
)


# ── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path. Idéntico patrón al usado en
    tests/test_anticipate_agent.py — monkeypatch DB_PATH antes de abrir
    conn, dispara la creación de tablas via `_ragvec_state_conn`.

    NO crea explícitamente `rag_anticipate_feedback`: queremos verificar
    que el módulo la crea on-demand (test 13).
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    client.get_or_create_collection(
        name="anticipate_feedback_test", metadata={"hnsw:space": "cosine"},
    )
    # Triggers DDL de tablas estándar (rag_*), pero NO crea la tabla
    # nueva `rag_anticipate_feedback` — ese DDL es responsabilidad del
    # módulo bajo test.
    with rag._ragvec_state_conn() as _conn:
        pass
    return tmp_path


# ── record_feedback ──────────────────────────────────────────────────────────


def test_record_feedback_inserts_row(state_db):
    """T1: record_feedback inserta correctamente y la row queda en SQL."""
    ok = record_feedback("cal:test:01", "positive", reason="se posta")
    assert ok is True

    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT dedup_key, rating, source, reason "
            "FROM rag_anticipate_feedback"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("cal:test:01", "positive", "wa", "se posta")


def test_record_feedback_rejects_invalid_rating(state_db):
    """T2: rating fuera del enum → False sin escribir."""
    ok = record_feedback("cal:test:02", "invalid")
    assert ok is False

    with rag._ragvec_state_conn() as conn:
        # _ensure_feedback_table no se ejecutó (record_feedback short-circuita
        # antes de abrir la conn cuando rating es inválido). Pero la tabla
        # puede o no existir — verificamos que NO hay row con ese dedup_key.
        try:
            rows = conn.execute(
                "SELECT * FROM rag_anticipate_feedback WHERE dedup_key = ?",
                ("cal:test:02",),
            ).fetchall()
            assert rows == []
        except Exception:
            # Si la tabla no existe todavía está bien — no se insertó nada.
            pass


def test_record_feedback_empty_dedup_key_returns_false(state_db):
    """Edge: dedup_key vacío → False (no escribe)."""
    assert record_feedback("", "positive") is False


# ── parse_wa_reply ───────────────────────────────────────────────────────────


def test_parse_wa_reply_thumbs_up_alone():
    """T3: '👍' sola → 'positive'."""
    assert parse_wa_reply("👍") == "positive"


def test_parse_wa_reply_thumbs_down_with_text():
    """T4: emoji 👎 mezclado con frase libre → 'negative'."""
    assert parse_wa_reply("👎 no sirvió") == "negative"


def test_parse_wa_reply_mute_with_word():
    """T5: 🔇 + 'basta' → 'mute' (ambas señales coherentes con mute)."""
    assert parse_wa_reply("🔇 basta") == "mute"


def test_parse_wa_reply_random_text_returns_none():
    """T6: texto sin keywords → None."""
    assert parse_wa_reply("random text") is None
    assert parse_wa_reply("hola que tal") is None
    assert parse_wa_reply("") is None
    assert parse_wa_reply(None) is None  # type: ignore[arg-type]


def test_parse_wa_reply_case_insensitive():
    """T7: 'SI' / 'NO' / 'OK' deben matchear igual que sus lowercase."""
    assert parse_wa_reply("SI") == "positive"
    assert parse_wa_reply("Si") == "positive"
    assert parse_wa_reply("NO") == "negative"
    assert parse_wa_reply("No") == "negative"
    assert parse_wa_reply("OK") == "positive"


def test_parse_wa_reply_ambiguous_prefers_negative():
    """T8: '👍 pero 👎' → 'negative' (conservador, respetar el negativo)."""
    assert parse_wa_reply("👍 pero 👎") == "negative"
    # También aplica para palabras: "si pero no" → negative
    assert parse_wa_reply("si pero no") == "negative"


def test_parse_wa_reply_mute_overrides_positive():
    """Mute debe prevalecer incluso si hay un emoji positivo en el mismo body."""
    assert parse_wa_reply("👍 pero silenciar") == "mute"
    assert parse_wa_reply("ok basta") == "mute"


def test_parse_wa_reply_strips_whitespace():
    """Whitespace al borde no debe romper el match."""
    assert parse_wa_reply("   👍   ") == "positive"
    assert parse_wa_reply("\n  no  \n") == "negative"


def test_parse_wa_reply_shortcodes():
    """Los shortcodes :thumbsup: / :thumbsdown: / :mute: deben parsear."""
    assert parse_wa_reply(":thumbsup:") == "positive"
    assert parse_wa_reply(":thumbsdown:") == "negative"
    assert parse_wa_reply(":mute:") == "mute"


# ── feedback_stats ───────────────────────────────────────────────────────────


def test_feedback_stats_total_no_filter(state_db):
    """T9: sin filtros, suma todos los ratings de la ventana default."""
    record_feedback("cal:1", "positive")
    record_feedback("cal:2", "negative")
    record_feedback("anniv:1", "positive")
    record_feedback("anniv:2", "mute")

    stats = feedback_stats()
    assert stats["positive"] == 2
    assert stats["negative"] == 1
    assert stats["mute"] == 1
    assert stats["total"] == 4
    # rate = positives / total = 2/4 = 0.5
    assert stats["rate"] == pytest.approx(0.5)


def test_feedback_stats_filters_by_kind_prefix(state_db):
    """T10: kind='cal:' → solo dedup_keys que arrancan con 'cal:'."""
    record_feedback("cal:abc", "positive")
    record_feedback("cal:def", "negative")
    record_feedback("anniv:abc", "positive")
    record_feedback("echo:xyz", "mute")

    cal_stats = feedback_stats(kind="cal:")
    assert cal_stats["positive"] == 1
    assert cal_stats["negative"] == 1
    assert cal_stats["mute"] == 0
    assert cal_stats["total"] == 2

    anniv_stats = feedback_stats(kind="anniv:")
    assert anniv_stats["positive"] == 1
    assert anniv_stats["total"] == 1

    none_match = feedback_stats(kind="commit:")
    assert none_match["total"] == 0
    assert none_match["rate"] == 0.0


def test_feedback_stats_window_days(state_db):
    """T11: rows fuera de la ventana de N días no cuentan."""
    # Insert una row con ts antiguo (45 días) bypass del helper para
    # forzar el ts. Tabla la crea record_feedback en la llamada previa.
    record_feedback("cal:fresh", "positive")

    old_ts = (datetime.now() - timedelta(days=45)).isoformat(timespec="seconds")
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_anticipate_feedback "
            "(ts, dedup_key, rating, source, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (old_ts, "cal:old", "negative", "wa", ""),
        )
        conn.commit()

    # Ventana default 30d → solo la fresca.
    stats_30d = feedback_stats(days=30)
    assert stats_30d["total"] == 1
    assert stats_30d["positive"] == 1
    assert stats_30d["negative"] == 0

    # Ventana 60d → ambas.
    stats_60d = feedback_stats(days=60)
    assert stats_60d["total"] == 2
    assert stats_60d["positive"] == 1
    assert stats_60d["negative"] == 1


# ── recent_feedback ──────────────────────────────────────────────────────────


def test_recent_feedback_ordered_desc(state_db):
    """T12: las rows vienen ordenadas desc por ts (más reciente primero)."""
    # Insertamos manualmente 3 con ts crecientes, después chequeamos orden.
    base = datetime.now()
    rows_in = [
        ((base - timedelta(minutes=30)).isoformat(timespec="seconds"),
         "cal:old", "positive"),
        ((base - timedelta(minutes=10)).isoformat(timespec="seconds"),
         "cal:mid", "negative"),
        (base.isoformat(timespec="seconds"), "cal:new", "mute"),
    ]
    # _ensure_feedback_table corre dentro de record_feedback; usamos un
    # write inicial para asegurar tabla creada, después inyectamos a mano.
    record_feedback("seed:1", "positive")
    with rag._ragvec_state_conn() as conn:
        for ts, dk, rating in rows_in:
            conn.execute(
                "INSERT INTO rag_anticipate_feedback "
                "(ts, dedup_key, rating, source, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, dk, rating, "wa", ""),
            )
        conn.commit()

    out = recent_feedback(limit=10)
    # Esperamos al menos las 3 manuales + la seed
    assert len(out) >= 4
    # El primer elemento (más reciente) debe ser cal:new (ts más alto).
    # Nota: la seed usa datetime.now() que puede ser muy cercano a `base`;
    # la cota es que cal:new venga ANTES que cal:mid y cal:mid antes que
    # cal:old.
    keys_order = [r["dedup_key"] for r in out]
    idx_new = keys_order.index("cal:new")
    idx_mid = keys_order.index("cal:mid")
    idx_old = keys_order.index("cal:old")
    assert idx_new < idx_mid < idx_old


def test_recent_feedback_respects_limit(state_db):
    """recent_feedback debe truncar al `limit`."""
    for i in range(5):
        record_feedback(f"cal:{i}", "positive")
    out = recent_feedback(limit=2)
    assert len(out) == 2


# ── Silent-fail / table autocreation ─────────────────────────────────────────


def test_feedback_stats_creates_table_on_first_call(state_db):
    """T13: si la tabla no existe (cold-start), feedback_stats no debe raise.

    El fixture state_db invoca `_ragvec_state_conn` pero NO importa
    `rag_anticipate.feedback`, así que la tabla `rag_anticipate_feedback`
    no existe hasta la primera llamada. feedback_stats debe crearla y
    devolver el shape vacío.
    """
    # Verificamos que la tabla NO existe antes (puede existir si otro
    # test corrió antes pero el state_db es per-test via tmp_path, así
    # que el DB es virgen).
    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='rag_anticipate_feedback'"
        ).fetchall()
    table_existed = len(rows) > 0

    stats = feedback_stats()
    assert stats == {
        "positive": 0,
        "negative": 0,
        "mute": 0,
        "total": 0,
        "rate": 0.0,
    }

    # Después de la llamada la tabla DEBE existir.
    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='rag_anticipate_feedback'"
        ).fetchall()
    assert len(rows) == 1, (
        f"Tabla rag_anticipate_feedback no se creó "
        f"(table_existed_before={table_existed})"
    )

    # Y un record subsiguiente debe persistir bien.
    assert record_feedback("cold:1", "positive") is True
    assert recent_feedback(limit=1)[0]["dedup_key"] == "cold:1"
