"""Tests para el campo `backend` en rag_queries.extra_json.

Post-Ola 7 (2026-05-06): `OllamaBackend` retirado. Los dispatch points
activos son `_mlx_chat`, `_mlx_chat_via_backend`, `_chat_stream_dispatch`
— todos setean el ContextVar `_ACTIVE_BACKEND_CTX` a "mlx". Los tests
de fallback ollama->mlx siguen midiendo el ContextVar manualmente
(via `_mark_backend`) porque `fallback_reason` sigue siendo parte del
schema operativo (telemetría histórica + futuras versiones de backend).
"""
import json

import pytest

import rag


# ---------------------------------------------------------------------------
# Fixtures de aislamiento
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Aislar DB_PATH por test para no contaminar telemetry.db de prod."""
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield tmp_path
    finally:
        rag.DB_PATH = snap


@pytest.fixture(autouse=True)
def _sync_writes(monkeypatch):
    """Forzar writes sincronas para poder leer rag_queries inmediatamente."""
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")


@pytest.fixture()
def _reset_backend_cv():
    """Resetea el ContextVar de backend antes y despues de cada test."""
    rag._reset_backend_telemetry()
    yield
    rag._reset_backend_telemetry()


# ---------------------------------------------------------------------------
# Tests del ContextVar (_mark_backend / _get_backend_telemetry)
# ---------------------------------------------------------------------------


class TestBackendContextVar:
    """Verifica la mecanica del ContextVar en aislamiento."""

    def test_default_is_none(self, _reset_backend_cv):
        assert rag._get_backend_telemetry() is None

    def test_mark_mlx(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        bk = rag._get_backend_telemetry()
        assert bk is not None
        assert bk["backend"] == "mlx"
        assert bk["fallback_reason"] is None
        assert bk["backend_active"] == "mlx"

    def test_mark_ollama_native(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
        rag._mark_backend("ollama")
        bk = rag._get_backend_telemetry()
        assert bk["backend"] == "ollama"
        assert bk["fallback_reason"] is None

    def test_mark_ollama_tools_fallback(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("ollama", fallback_reason="tools")
        bk = rag._get_backend_telemetry()
        assert bk["backend"] == "ollama"
        assert bk["fallback_reason"] == "tools"

    def test_mark_ollama_stream_fallback(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("ollama", fallback_reason="stream")
        bk = rag._get_backend_telemetry()
        assert bk["fallback_reason"] == "stream"

    def test_reset_clears_value(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        assert rag._get_backend_telemetry() is not None
        rag._reset_backend_telemetry()
        assert rag._get_backend_telemetry() is None

    def test_overwrite_overwrites(self, _reset_backend_cv, monkeypatch):
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        rag._mark_backend("ollama", fallback_reason="tools")
        bk = rag._get_backend_telemetry()
        assert bk["backend"] == "ollama"
        assert bk["fallback_reason"] == "tools"


# ---------------------------------------------------------------------------
# Tests de log_query_event + extra_json
# ---------------------------------------------------------------------------


def _read_last_query(tmp_path) -> dict:
    """Lee el extra_json del ultimo row de rag_queries en la DB aislada."""
    import sqlite3
    db = tmp_path / "ragvec" / "telemetry.db"
    if not db.exists():
        return {}
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT extra_json FROM rag_queries ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    if row is None or not row["extra_json"]:
        return {}
    return json.loads(row["extra_json"]) or {}


class TestLogQueryEventBackendField:
    """Verifica que log_query_event lee el CV y agrega backend a extra_json."""

    def test_backend_field_written_when_cv_set(self, _isolate_db, _reset_backend_cv, monkeypatch):
        """Caso happy path: CV seteado antes de log_query_event -> backend en extra_json."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
        rag._mark_backend("ollama")
        rag.log_query_event({
            "cmd": "test_backend_field",
            "q": "test query backend",
        })
        extra = _read_last_query(_isolate_db)
        assert extra.get("backend") == "ollama", f"extra_json={extra}"
        assert "fallback_reason" in extra

    def test_backend_missing_when_cv_not_set(self, _isolate_db, _reset_backend_cv):
        """Sin CV seteado, no se agrega backend al extra_json."""
        rag.log_query_event({
            "cmd": "test_no_backend",
            "q": "test sin backend cv",
        })
        extra = _read_last_query(_isolate_db)
        assert "backend" not in extra, f"Unexpected backend in extra_json={extra}"

    def test_mlx_backend_written(self, _isolate_db, _reset_backend_cv, monkeypatch):
        """Simula dispatch MLX: backend=mlx en extra_json."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        rag.log_query_event({
            "cmd": "web",
            "q": "pregunta via mlx",
        })
        extra = _read_last_query(_isolate_db)
        assert extra.get("backend") == "mlx"
        assert extra.get("fallback_reason") is None
        assert extra.get("backend_active") == "mlx"

    def test_tools_fallback_written(self, _isolate_db, _reset_backend_cv, monkeypatch):
        """Simula fallback MLX->Ollama por tools=: fallback_reason=tools en extra_json."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("ollama", fallback_reason="tools")
        rag.log_query_event({
            "cmd": "web",
            "q": "pregunta con tools bajo mlx",
        })
        extra = _read_last_query(_isolate_db)
        assert extra.get("backend") == "ollama"
        assert extra.get("fallback_reason") == "tools"
        assert extra.get("backend_active") == "mlx"

    def test_stream_fallback_written(self, _isolate_db, _reset_backend_cv, monkeypatch):
        """Simula fallback MLX->Ollama por stream=True."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("ollama", fallback_reason="stream")
        rag.log_query_event({
            "cmd": "web",
            "q": "streaming bajo mlx",
        })
        extra = _read_last_query(_isolate_db)
        assert extra.get("fallback_reason") == "stream"

    def test_backend_not_overwritten_if_already_in_event(
        self, _isolate_db, _reset_backend_cv, monkeypatch
    ):
        """Si el evento ya trae 'backend', no se pisa con el CV."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        rag.log_query_event({
            "cmd": "test",
            "q": "con backend explicito",
            "backend": "explicit_override",
        })
        extra = _read_last_query(_isolate_db)
        assert extra.get("backend") == "explicit_override"

    def test_ollama_native_no_fallback_reason(self, _isolate_db, _reset_backend_cv, monkeypatch):
        """Backend Ollama nativo: fallback_reason=None."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
        rag._mark_backend("ollama")
        rag.log_query_event({"cmd": "cli", "q": "ollama nativo"})
        extra = _read_last_query(_isolate_db)
        assert extra.get("backend") == "ollama"
        assert extra.get("fallback_reason") is None


# ---------------------------------------------------------------------------
# Tests de dispatch points (sin LLM real: verifica que el CV se setea)
# ---------------------------------------------------------------------------


@pytest.mark.no_auto_mlx_stub
class TestDispatchPointsMarkBackend:
    """Verifica que los dispatch points setean el CV correctamente.

    No se invoca LLM real — solo verifica el CV via _get_backend_telemetry()
    despues de mockear `_mlx_chat_via_backend` para que no toque MLX.

    `no_auto_mlx_stub` marker: la fixture autouse del conftest stubea
    `_mlx_chat` para tests indirectos. Acá necesitamos ejercitar el
    `_mlx_chat` real (con su `_mark_backend` side-effect), así que
    optamos out.
    """

    def test_mlx_chat_marks_backend(self, monkeypatch, _reset_backend_cv):
        """`_mlx_chat` setea backend=mlx en el ContextVar."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        mock_resp = type("R", (), {"message": type("M", (), {"content": "ok"})()})()
        # Bypass MLX backend real — mockear el via_backend directamente.
        monkeypatch.setattr(rag, "_mlx_chat_via_backend", lambda **kw: mock_resp)

        rag._mlx_chat(model="qwen2.5:3b", messages=[])
        bk = rag._get_backend_telemetry()
        assert bk is not None
        assert bk["backend"] == "mlx"

    def test_mark_backend_called_directly_in_mlx_path(self, monkeypatch, _reset_backend_cv):
        """Simula el dispatch MLX llamando `_mark_backend` directo."""
        monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
        rag._mark_backend("mlx")
        bk = rag._get_backend_telemetry()
        assert bk is not None
        assert bk["backend"] == "mlx"
        assert bk["backend_active"] == "mlx"
