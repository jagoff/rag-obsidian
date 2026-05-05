"""Tests para Sprint 3-A: replay payload en rag_queries.extra_json.

Cubre:
- Shape del extra_json post-write (con/sin flags de privacidad)
- Truncation cuando text supera el cap de bytes
- Hashes determinísticos (misma input → mismo hash)
- Back-compat: rows sin campos de replay siguen legibles
- Env var gates: RAG_LOG_REPLAY_PAYLOAD y RAG_LOG_RERANK_RAW
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import rag


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Aisla DB_PATH para no escribir a la prod telemetry.db."""
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        rag.DB_PATH = snap


@pytest.fixture(autouse=True)
def _sync_writes(monkeypatch):
    """Fuerza writes síncronos para read-after-write en tests."""
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _expected_hash(text: str) -> str:
    """sha256[:16] del texto — replica la lógica de _replay_hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_last_extra(conn=None) -> dict:
    """Lee el extra_json de la última fila de rag_queries."""
    import sqlite3
    db_path = rag.DB_PATH / "telemetry.db"
    with sqlite3.connect(str(db_path)) as c:
        row = c.execute(
            "SELECT extra_json FROM rag_queries ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row, "No rows in rag_queries"
    return json.loads(row[0] or "{}")


def _write_event(**kwargs) -> dict:
    """Escribe un evento mínimo y devuelve el extra_json persistido."""
    event = {
        "cmd": "test",
        "q": "test query",
        **kwargs,
    }
    rag.log_query_event(event)
    return _read_last_extra()


# ─── Tests: _replay_hash ────────────────────────────────────────────────────

def test_replay_hash_deterministic():
    """La misma string siempre produce el mismo hash de 16 chars."""
    h1 = rag._replay_hash("hola mundo")
    h2 = rag._replay_hash("hola mundo")
    assert h1 == h2
    assert len(h1) == 16


def test_replay_hash_empty():
    """String vacía → retorna ''."""
    assert rag._replay_hash("") == ""
    assert rag._replay_hash(None) == ""  # type: ignore[arg-type]


def test_replay_hash_matches_sha256():
    """El hash coincide con sha256(text)[:16]."""
    text = "respuesta del LLM"
    assert rag._replay_hash(text) == _expected_hash(text)


def test_replay_hash_different_inputs():
    """Inputs distintos producen hashes distintos (probabilísticamente)."""
    assert rag._replay_hash("texto A") != rag._replay_hash("texto B")


# ─── Tests: _truncate_for_replay ────────────────────────────────────────────

def test_truncate_no_op_when_within_limit():
    """Texto que entra en el cap no se trunca."""
    text = "hola"
    result, truncated = rag._truncate_for_replay(text, max_bytes=100)
    assert result == text
    assert truncated is False


def test_truncate_at_cap():
    """Texto que supera el cap se trunca; el flag es True."""
    text = "a" * 200
    result, truncated = rag._truncate_for_replay(text, max_bytes=100)
    assert len(result.encode("utf-8")) <= 100
    assert truncated is True


def test_truncate_preserves_utf8_codepoints():
    """No corta en medio de un codepoint multibyte (ej. emoji 4 bytes)."""
    # 🎸 es 4 bytes en UTF-8. 10 emojis = 40 bytes.
    text = "🎸" * 10
    result, _ = rag._truncate_for_replay(text, max_bytes=15)
    # El resultado debe decodificar limpio (sin reemplazos U+FFFD)
    assert "�" not in result
    assert len(result.encode("utf-8")) <= 15


def test_truncate_empty_input():
    """String vacía → '' sin truncar."""
    result, truncated = rag._truncate_for_replay("", max_bytes=100)
    assert result == ""
    assert truncated is False


# ─── Tests: _build_replay_fields ────────────────────────────────────────────

def test_build_replay_fields_always_on_hashes(monkeypatch):
    """corpus_hash + response_hash + prompt_hash siempre presentes."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)

    fields = rag._build_replay_fields(
        response="respuesta",
        prompt="prompt",
        corpus_hash="abc123",
    )
    assert fields["corpus_hash"] == "abc123"
    assert fields["response_hash"] == _expected_hash("respuesta")
    assert fields["prompt_hash"] == _expected_hash("prompt")
    # Sin flag: no payload raw
    assert "response_text" not in fields
    assert "response_truncated" not in fields


def test_build_replay_fields_empty_response(monkeypatch):
    """Response vacía → no hay response_hash."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    fields = rag._build_replay_fields(response="", corpus_hash="abc")
    assert "response_hash" not in fields


def test_build_replay_fields_history_hash_always_on(monkeypatch):
    """history_hash SIEMPRE presente cuando hay history."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    history = [{"role": "user", "content": "pregunta anterior"}]
    fields = rag._build_replay_fields(history=history)
    assert "history_hash" in fields
    assert len(fields["history_hash"]) == 16
    # Sin flag: no snapshot
    assert "history_snapshot" not in fields


def test_build_replay_fields_payload_flag_on(monkeypatch):
    """Con RAG_LOG_REPLAY_PAYLOAD=1, response_text + history_snapshot presentes."""
    monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", "1")
    history = [{"role": "user", "content": "q prev"}]
    fields = rag._build_replay_fields(
        response="respuesta larga",
        history=history,
        corpus_hash="hash123",
    )
    assert "response_text" in fields
    assert "response_truncated" in fields
    assert "history_snapshot" in fields
    assert "history_truncated" in fields
    assert fields["response_text"] == "respuesta larga"
    assert fields["response_truncated"] is False


def test_build_replay_fields_rerank_logits_gated(monkeypatch):
    """rerank_logits_raw solo presente con RAG_LOG_RERANK_RAW=1."""
    monkeypatch.delenv("RAG_LOG_RERANK_RAW", raising=False)
    fields = rag._build_replay_fields(rerank_logits=[0.9, 0.7])
    assert "rerank_logits_raw" not in fields

    monkeypatch.setenv("RAG_LOG_RERANK_RAW", "1")
    fields2 = rag._build_replay_fields(rerank_logits=[0.9, 0.7])
    assert "rerank_logits_raw" in fields2
    assert fields2["rerank_logits_raw"] == [0.9, 0.7]


def test_build_replay_fields_rerank_logits_rounded(monkeypatch):
    """Los logits se redondean a 6 decimales."""
    monkeypatch.setenv("RAG_LOG_RERANK_RAW", "1")
    fields = rag._build_replay_fields(rerank_logits=[0.123456789])
    assert fields["rerank_logits_raw"] == [0.123457]


def test_build_replay_fields_no_corpus_hash(monkeypatch):
    """corpus_hash vacío → no incluye el campo."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    fields = rag._build_replay_fields(response="r", corpus_hash="")
    assert "corpus_hash" not in fields


def test_build_replay_fields_empty_history(monkeypatch):
    """history=None o [] → sin history_hash."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    assert "history_hash" not in rag._build_replay_fields(history=None)
    assert "history_hash" not in rag._build_replay_fields(history=[])


def test_build_replay_fields_response_truncated_at_cap(monkeypatch):
    """Con payload ON y response > 8 KB, response_truncated=True."""
    monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", "1")
    big_response = "x" * 10_000  # 10 KB > 8 KB cap
    fields = rag._build_replay_fields(response=big_response)
    assert fields["response_truncated"] is True
    assert len(fields["response_text"].encode("utf-8")) <= 8_192


def test_build_replay_fields_history_truncated_at_cap(monkeypatch):
    """Con payload ON e history > 4 KB, history_truncated=True."""
    monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", "1")
    big_history = [{"role": "user", "content": "a" * 5_000}]
    fields = rag._build_replay_fields(history=big_history)
    assert fields["history_truncated"] is True


# ─── Tests: persistencia en rag_queries.extra_json ─────────────────────────

def test_log_query_event_persists_corpus_hash(monkeypatch):
    """corpus_hash en extra_json cuando se pasa al evento."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    extra = _write_event(**rag._build_replay_fields(corpus_hash="myhash"))
    assert extra.get("corpus_hash") == "myhash"


def test_log_query_event_persists_response_hash(monkeypatch):
    """response_hash en extra_json."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    extra = _write_event(**rag._build_replay_fields(response="mi respuesta"))
    assert extra.get("response_hash") == _expected_hash("mi respuesta")


def test_log_query_event_no_payload_without_flag(monkeypatch):
    """Sin RAG_LOG_REPLAY_PAYLOAD=1 no hay response_text en extra_json."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    extra = _write_event(**rag._build_replay_fields(
        response="una respuesta", corpus_hash="h1"
    ))
    assert "response_text" not in extra
    # Pero sí hay hashes
    assert "response_hash" in extra
    assert "corpus_hash" in extra


def test_log_query_event_payload_with_flag(monkeypatch):
    """Con RAG_LOG_REPLAY_PAYLOAD=1 hay response_text en extra_json."""
    monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", "1")
    extra = _write_event(**rag._build_replay_fields(
        response="respuesta completa", corpus_hash="h2"
    ))
    assert extra.get("response_text") == "respuesta completa"
    assert extra.get("response_truncated") is False


def test_log_query_event_history_hash_always(monkeypatch):
    """history_hash en extra_json sin necesidad de flag."""
    monkeypatch.delenv("RAG_LOG_REPLAY_PAYLOAD", raising=False)
    history = [{"role": "user", "content": "turno previo"}]
    extra = _write_event(**rag._build_replay_fields(history=history))
    assert "history_hash" in extra
    assert "history_snapshot" not in extra  # flag OFF


def test_log_query_event_history_snapshot_with_flag(monkeypatch):
    """history_snapshot en extra_json con RAG_LOG_REPLAY_PAYLOAD=1."""
    monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", "1")
    history = [{"role": "assistant", "content": "respuesta previa"}]
    extra = _write_event(**rag._build_replay_fields(history=history))
    assert "history_snapshot" in extra
    # El snapshot puede ser list (JSON parseado) o str (si la truncation dejó JSON inválido)
    assert isinstance(extra["history_snapshot"], (list, str))


def test_log_query_event_without_replay_fields_backward_compat():
    """Rows sin campos de replay siguen siendo legibles (back-compat)."""
    # Simula una row legacy sin hashes
    rag.log_query_event({"cmd": "legacy", "q": "vieja query"})
    extra = _read_last_extra()
    # Debe ser un dict (posiblemente vacío), sin errores
    assert isinstance(extra, dict)
    # No deben estar los campos de replay
    assert "response_hash" not in extra
    assert "corpus_hash" not in extra


def test_replay_hash_stable_across_calls():
    """El mismo texto produce el mismo hash en múltiples llamadas."""
    text = "texto de prueba para estabilidad"
    hashes = {rag._replay_hash(text) for _ in range(10)}
    assert len(hashes) == 1  # todos iguales


def test_build_replay_fields_truthy_variants_for_payload_flag(monkeypatch):
    """RAG_LOG_REPLAY_PAYLOAD acepta '1', 'true', 'yes'."""
    for val in ("1", "true", "True", "TRUE", "yes", "YES"):
        monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", val)
        fields = rag._build_replay_fields(response="r")
        assert "response_text" in fields, f"Flag '{val}' debería activar payload"


def test_build_replay_fields_falsy_variants_for_payload_flag(monkeypatch):
    """RAG_LOG_REPLAY_PAYLOAD no activa con '', '0', 'false', 'no'."""
    for val in ("", "0", "false", "False", "no", "NO"):
        monkeypatch.setenv("RAG_LOG_REPLAY_PAYLOAD", val)
        fields = rag._build_replay_fields(response="r")
        assert "response_text" not in fields, f"Flag '{val}' NO debería activar payload"
