"""WhatsApp promise tracker — regex pre-filter, time parser, LLM extractor, DDL.

Sin LLM real ni Ollama: el extractor se testea stubbeando `rag._summary_client`
con un fake que devuelve JSON pre-armado. Patrón cribado de `test_today.py`.
La DDL se valida abriendo una conn nueva contra una tmp_path DB y
chequeando con PRAGMA table_info / index_list.

Cobertura:
  - `_has_promise_hint`: matches positivos rioplatenses + non-matches small talk.
  - `_PROMISE_REGEX_HINTS`: cardinality + compilación.
  - `_parse_promise_when`: when_text vacío, parseable, no-parseable, en pasado.
  - `_wa_extract_promises`: skip-no-hint, anti-loop U+200B, JSON inválido,
    promesas válidas con direction in/out, dedup, msg_id enrichment.
  - DDL: tabla rag_promises existe con shape esperado + 4 índices.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag


# ─────────────────────────────────────────────────────────────────────────────
# Helpers comunes (cribados de test_today.py)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeClient:
    """Stub para `_summary_client()` — devuelve siempre el mismo JSON
    pre-armado al .chat(). Si `raise_on_call=True`, simula error LLM."""

    def __init__(self, response_content: str = "{}", raise_on_call: bool = False) -> None:
        self._content = response_content
        self._raise = raise_on_call

    def chat(self, **kwargs):
        if self._raise:
            raise RuntimeError("LLM down (simulated)")
        return _FakeResponse(self._content)


def _install_fake_llm(monkeypatch, *, content: str = "{}", raise_on_call: bool = False) -> _FakeClient:
    fake = _FakeClient(content, raise_on_call=raise_on_call)
    monkeypatch.setattr(rag, "_summary_client", lambda: fake)
    return fake


# ─────────────────────────────────────────────────────────────────────────────
# 1. _has_promise_hint — regex pre-filter
# ─────────────────────────────────────────────────────────────────────────────

class TestHasPromiseHint:
    """Pre-filter de regex: descarta el ~80% de msgs sin hints."""

    @pytest.mark.parametrize("text", [
        "después te aviso lo del proyecto",
        "Despues te aviso (sin tilde)",
        "te llamo en un rato",
        "te paso la info más tarde",
        "más tarde lo veo",
        "luego te paso",
        "mañana te aviso",
        "el lunes te paso",
        "te prometo que sí",
        "te confirmo después",
        "después arreglamos",
        "en 10 min te llamo",
        "en 2 horas te confirmo",
        "ahora te mando",
        "voy a avisarte mañana",
        "te lo paso enseguida",
        "esta tarde lo reviso",
        "esta noche te confirmo",
        "lo reviso y te digo",
        "ya te aviso",
    ])
    def test_positive_matches(self, text: str) -> None:
        assert rag._has_promise_hint(text) is True, f"esperaba match para: {text!r}"

    @pytest.mark.parametrize("text", [
        "hola, todo bien?",
        "jaja qué loco",
        "🎉",
        "buenas",
        "ok dale",
        "perfecto",
        "te avisé ayer",  # pasado, no es promesa
        # NOTA: "siempre te aviso temprano" matchea "te aviso" — pero ese
        # false-positive es aceptable, el LLM filtra después. El pre-filter
        # está deliberadamente sesgado a recall (no missed promises) sobre
        # precision (no extra LLM calls). Si querés cubrir ese caso en el
        # futuro, agregar un negative-lookbehind \b(?<!siempre )te\s+aviso\b.
        "",
        " ",
    ])
    def test_negative_matches(self, text: str) -> None:
        assert rag._has_promise_hint(text) is False, f"esperaba NO-match para: {text!r}"

    def test_none_returns_false(self) -> None:
        assert rag._has_promise_hint(None) is False  # type: ignore

    def test_non_string_returns_false(self) -> None:
        assert rag._has_promise_hint(12345) is False  # type: ignore
        assert rag._has_promise_hint([]) is False  # type: ignore

    def test_patterns_compiled(self) -> None:
        """Las regex están compiladas a tiempo de import (lazy en runtime sería peor perf)."""
        import re
        assert isinstance(rag._PROMISE_REGEX_HINTS, tuple)
        assert len(rag._PROMISE_REGEX_HINTS) >= 10, "patterns demasiado pocos — recall bajo"
        for p in rag._PROMISE_REGEX_HINTS:
            assert isinstance(p, re.Pattern)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _parse_promise_when — wrapper sobre _parse_natural_datetime
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePromiseWhen:
    """Resolución de when_text → (datetime, confidence)."""

    _ANCHOR = datetime(2026, 4, 25, 14, 0, 0)  # viernes 14:00

    def test_empty_returns_default_2h(self) -> None:
        dt, conf = rag._parse_promise_when("", anchor=self._ANCHOR)
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_whitespace_returns_default_2h(self) -> None:
        dt, conf = rag._parse_promise_when("   ", anchor=self._ANCHOR)
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_none_returns_default_2h(self) -> None:
        dt, conf = rag._parse_promise_when(None, anchor=self._ANCHOR)  # type: ignore
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_explicit_relative_in_2hs(self) -> None:
        dt, conf = rag._parse_promise_when("en 2 horas", anchor=self._ANCHOR)
        # Permitimos algo de slop — dateparser puede interpretar como "en 2hs desde ahora"
        # con drift de segundos. Validamos que esté en la ventana correcta.
        delta = (dt - self._ANCHOR).total_seconds()
        assert 6900 < delta < 7500, f"esperaba ~2hs, got {delta}s"
        assert conf == 0.9

    def test_unparseable_returns_default(self) -> None:
        dt, conf = rag._parse_promise_when("xyzasdf qué dice esto", anchor=self._ANCHOR)
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_past_time_corrected_to_default(self, monkeypatch) -> None:
        """Si el parser devuelve algo en el pasado relativo al anchor, fallback."""
        # Forzamos al parser a devolver un past datetime
        monkeypatch.setattr(rag, "_parse_natural_datetime",
                            lambda *a, **kw: datetime(2020, 1, 1, 0, 0, 0))
        dt, conf = rag._parse_promise_when("hace 6 años", anchor=self._ANCHOR)
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_parser_exception_returns_default(self, monkeypatch) -> None:
        """Si _parse_natural_datetime levanta excepción, fallback silencioso."""
        def _boom(*a, **kw):
            raise RuntimeError("parser internals broke")
        monkeypatch.setattr(rag, "_parse_natural_datetime", _boom)
        dt, conf = rag._parse_promise_when("mañana", anchor=self._ANCHOR)
        assert dt == self._ANCHOR + timedelta(hours=2)
        assert conf == 0.3

    def test_anchor_default_is_now(self, monkeypatch) -> None:
        """Sin anchor explícito, usa datetime.now()."""
        # No quiero depender del reloj real → mockeo el parser para que retorne future_dt
        future_dt = datetime(2099, 1, 1, 12, 0, 0)
        monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: future_dt)
        dt, conf = rag._parse_promise_when("alguna fecha")
        assert dt == future_dt
        assert conf == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 3. _wa_extract_promises — extractor LLM con stubbing
# ─────────────────────────────────────────────────────────────────────────────

class TestWaExtractPromises:
    """Extractor end-to-end (LLM stubbed)."""

    def _msg(self, *, msg_id: str, who: str, text: str, ts: str = "2026-04-25T14:30:00") -> dict:
        return {"msg_id": msg_id, "who": who, "text": text, "ts": ts}

    def test_empty_messages_returns_empty(self, monkeypatch) -> None:
        # No debe llamar al LLM
        called = {"chat": 0}
        monkeypatch.setattr(rag, "_summary_client",
                            lambda: type("_", (), {"chat": lambda **kw: (called.__setitem__("chat", called["chat"]+1) or _FakeResponse("{}"))})())
        result = rag._wa_extract_promises("Juan", False, [])
        assert result == []
        assert called["chat"] == 0

    def test_no_hint_skips_llm(self, monkeypatch) -> None:
        """Si NINGÚN msg matchea regex, NO se llama al LLM (cost saving)."""
        called = {"chat": 0}
        class _CountingClient:
            def chat(self, **kw):
                called["chat"] += 1
                return _FakeResponse('{"promises": []}')
        monkeypatch.setattr(rag, "_summary_client", lambda: _CountingClient())

        msgs = [
            self._msg(msg_id="1", who="Juan", text="hola"),
            self._msg(msg_id="2", who="yo", text="ok"),
            self._msg(msg_id="3", who="Juan", text="jaja"),
        ]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert result == []
        assert called["chat"] == 0, "LLM no debería haberse llamado"

    def test_anti_loop_drops_u200b_messages(self, monkeypatch) -> None:
        """Msgs que empiecen con U+200B son nuestros propios reminders → drop."""
        called = {"chat": 0}
        class _CountingClient:
            def chat(self, **kw):
                called["chat"] += 1
                return _FakeResponse('{"promises": []}')
        monkeypatch.setattr(rag, "_summary_client", lambda: _CountingClient())

        # El único msg con hint tiene U+200B → debe descartarse
        msgs = [
            self._msg(msg_id="1", who="yo", text="\u200bdespués te aviso (este es nuestro reminder)"),
        ]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert result == []
        assert called["chat"] == 0, "msg con U+200B debe descartarse pre-LLM"

    def test_llm_failure_returns_empty(self, monkeypatch) -> None:
        """LLM exception → silent-fail con [] (no propagación)."""
        _install_fake_llm(monkeypatch, raise_on_call=True)
        msgs = [self._msg(msg_id="1", who="yo", text="después te aviso")]
        assert rag._wa_extract_promises("Juan", False, msgs) == []

    def test_llm_invalid_json_returns_empty(self, monkeypatch) -> None:
        _install_fake_llm(monkeypatch, content="not json at all {{")
        msgs = [self._msg(msg_id="1", who="yo", text="después te aviso")]
        assert rag._wa_extract_promises("Juan", False, msgs) == []

    def test_valid_promise_outbound(self, monkeypatch) -> None:
        _install_fake_llm(monkeypatch, content=json.dumps({
            "promises": [
                {"text": "te llamo en un rato", "when_text": "en un rato",
                 "direction": "out", "msg_id": "msg-1"}
            ]
        }))
        msgs = [self._msg(msg_id="msg-1", who="yo", text="te llamo en un rato")]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert len(result) == 1
        p = result[0]
        assert p["text"] == "te llamo en un rato"
        assert p["when_text"] == "en un rato"
        assert p["direction"] == "out"
        assert p["msg_id"] == "msg-1"
        assert p["msg_ts"] == "2026-04-25T14:30:00"
        assert p["speaker"] == "yo"

    def test_valid_promise_inbound(self, monkeypatch) -> None:
        _install_fake_llm(monkeypatch, content=json.dumps({
            "promises": [
                {"text": "después te paso el link", "when_text": "",
                 "direction": "in", "msg_id": "msg-7"}
            ]
        }))
        msgs = [self._msg(msg_id="msg-7", who="Maria", text="después te paso el link")]
        result = rag._wa_extract_promises("Maria", False, msgs)
        assert len(result) == 1
        assert result[0]["direction"] == "in"
        assert result[0]["speaker"] == "Maria"

    def test_dedup_same_text_same_direction(self, monkeypatch) -> None:
        """Dos promesas idénticas en (text, direction) → dedup, solo 1."""
        _install_fake_llm(monkeypatch, content=json.dumps({
            "promises": [
                {"text": "te llamo", "when_text": "", "direction": "out", "msg_id": "1"},
                {"text": "TE LLAMO", "when_text": "", "direction": "out", "msg_id": "2"},
            ]
        }))
        msgs = [
            self._msg(msg_id="1", who="yo", text="te llamo después"),
            self._msg(msg_id="2", who="yo", text="te llamo en un rato"),
        ]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert len(result) == 1

    def test_filter_invalid_direction(self, monkeypatch) -> None:
        """Direction != 'in'/'out' → descartar el item."""
        _install_fake_llm(monkeypatch, content=json.dumps({
            "promises": [
                {"text": "te llamo", "when_text": "", "direction": "lateral", "msg_id": "1"},
                {"text": "te paso info", "when_text": "", "direction": "out", "msg_id": "2"},
            ]
        }))
        msgs = [
            self._msg(msg_id="1", who="yo", text="te llamo después"),
            self._msg(msg_id="2", who="yo", text="te paso info más tarde"),
        ]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert len(result) == 1
        assert result[0]["text"] == "te paso info"

    def test_filter_too_short_or_long(self, monkeypatch) -> None:
        """text < 4 o > 240 chars → descartar."""
        _install_fake_llm(monkeypatch, content=json.dumps({
            "promises": [
                {"text": "ok", "when_text": "", "direction": "out", "msg_id": "1"},        # too short
                {"text": "x" * 300, "when_text": "", "direction": "out", "msg_id": "2"},   # too long
                {"text": "te paso info", "when_text": "", "direction": "out", "msg_id": "3"},  # OK
            ]
        }))
        msgs = [self._msg(msg_id="3", who="yo", text="te paso info más tarde")]
        result = rag._wa_extract_promises("Juan", False, msgs)
        assert len(result) == 1
        assert result[0]["text"] == "te paso info"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DDL — tabla rag_promises se crea con shape correcto
# ─────────────────────────────────────────────────────────────────────────────

class TestPromisesDDL:
    """La DDL crea la tabla con todas las columnas + 4 índices esperados."""

    def _setup_db(self, tmp_path: Path) -> sqlite3.Connection:
        """Abre una conn limpia y corre el DDL completo."""
        db_path = tmp_path / "test_telemetry.db"
        conn = sqlite3.connect(str(db_path))
        rag._ensure_telemetry_tables(conn)
        return conn

    def test_table_exists(self, tmp_path: Path) -> None:
        conn = self._setup_db(tmp_path)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_promises'"
        ).fetchone()
        assert row is not None, "tabla rag_promises no se creó"

    def test_schema_columns(self, tmp_path: Path) -> None:
        conn = self._setup_db(tmp_path)
        cols = {row[1]: row for row in conn.execute("PRAGMA table_info(rag_promises)").fetchall()}
        expected = {
            "id", "ts", "contact_jid", "contact_name", "promise_text",
            "direction", "due_ts", "due_confidence", "source_msg_id",
            "source_chat_jid", "status", "reminder_sent_ts", "closed_ts",
            "closed_reason", "extra_json",
        }
        assert expected.issubset(cols.keys()), \
            f"columnas faltantes: {expected - cols.keys()}"

    def test_indexes_created(self, tmp_path: Path) -> None:
        conn = self._setup_db(tmp_path)
        idx = {row[1] for row in conn.execute("PRAGMA index_list(rag_promises)").fetchall()}
        for expected_idx in (
            "ix_rag_promises_ts",
            "ix_rag_promises_due_ts",
            "ix_rag_promises_status",
            "ix_rag_promises_contact",
        ):
            assert expected_idx in idx, f"índice {expected_idx} no se creó"

    def test_status_default_pending(self, tmp_path: Path) -> None:
        """Insertar sin especificar status → DEFAULT 'pending'."""
        conn = self._setup_db(tmp_path)
        conn.execute(
            "INSERT INTO rag_promises (ts, contact_jid, promise_text, direction) "
            "VALUES (?, ?, ?, ?)",
            ("2026-04-25T14:30:00", "5491234567890@s.whatsapp.net", "te llamo", "out"),
        )
        row = conn.execute("SELECT status FROM rag_promises").fetchone()
        assert row[0] == "pending"

    def test_required_fields_not_null(self, tmp_path: Path) -> None:
        """Las 4 columnas NOT NULL rechazan inserts sin esos valores."""
        conn = self._setup_db(tmp_path)
        # Sin ts
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rag_promises (contact_jid, promise_text, direction) "
                "VALUES (?, ?, ?)",
                ("jid", "txt", "out"),
            )
        # Sin contact_jid
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rag_promises (ts, promise_text, direction) "
                "VALUES (?, ?, ?)",
                ("2026-04-25T14:30:00", "txt", "out"),
            )
        # Sin promise_text
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rag_promises (ts, contact_jid, direction) "
                "VALUES (?, ?, ?)",
                ("2026-04-25T14:30:00", "jid", "out"),
            )
        # Sin direction
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rag_promises (ts, contact_jid, promise_text) "
                "VALUES (?, ?, ?)",
                ("2026-04-25T14:30:00", "jid", "txt"),
            )
