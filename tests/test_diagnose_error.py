"""Tests para `/api/diagnose-error` + `/api/diagnose-error/execute`.

Foco: la safety del endpoint execute. Si dejamos pasar `rm -rf` o
`sudo` por accidente, el LLM puede dañar el sistema. Estos tests son
defensa-en-profundidad contra regressions.

Cubre:
  - Model resolution: command-r preferido, fallback a chat default.
  - /diagnose-error rechaza payloads malformados (422 / 4xx).
  - /execute bloquea comandos fuera del whitelist (403).
  - /execute bloquea metachars peligrosos (;, |, >, $, backticks).
  - /execute valida arguments por command (rag stats OK, rag index NO).
  - /execute corre comandos válidos con env mínimo (mocked subprocess).
  - /execute audit log se escribe (best-effort, no bloquea ejecución).
  - SSE stream emite eventos type=model/token/done/error.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


# ── Model resolution ─────────────────────────────────────────────────────


def test_resolve_diagnose_model_prefers_command_r(monkeypatch):
    """Con command-r:latest instalado lo elige antes que cualquier otro."""
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None  # reset cache

    class FakeModel:
        def __init__(self, m): self.model = m
    class FakeList:
        models = [FakeModel("command-r:latest"), FakeModel("qwen2.5:7b")]

    monkeypatch.setattr(_server.ollama, "list", lambda: FakeList())
    assert _server._resolve_diagnose_model() == "command-r:latest"
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None


def test_resolve_diagnose_model_falls_back_to_chat_model(monkeypatch):
    """Sin ningún command-r, cae a resolve_chat_model() del rag.py."""
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None

    class FakeModel:
        def __init__(self, m): self.model = m
    class FakeList:
        models = [FakeModel("phi4:latest")]

    monkeypatch.setattr(_server.ollama, "list", lambda: FakeList())
    monkeypatch.setattr(_server, "resolve_chat_model", lambda: "phi4:latest")
    assert _server._resolve_diagnose_model() == "phi4:latest"
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None


def test_resolve_diagnose_model_caches_per_process(monkeypatch):
    """Cacheado tras el primer call — `ollama.list()` es ~50ms."""
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None
    call_count = {"n": 0}

    class FakeModel:
        def __init__(self, m): self.model = m
    class FakeList:
        models = [FakeModel("command-r:latest")]

    def fake_list():
        call_count["n"] += 1
        return FakeList()

    monkeypatch.setattr(_server.ollama, "list", fake_list)
    _server._resolve_diagnose_model()
    _server._resolve_diagnose_model()
    _server._resolve_diagnose_model()
    assert call_count["n"] == 1
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None


# ── /api/diagnose-error: validación de payload ───────────────────────────


def test_diagnose_rejects_missing_required_fields():
    """Sin error_text → 422."""
    resp = _client.post("/api/diagnose-error", json={})
    assert resp.status_code == 422


def test_diagnose_rejects_empty_error_text():
    """error_text vacío → 422 (validator rechaza después de strip)."""
    resp = _client.post("/api/diagnose-error", json={
        "error_text": "   ",
        "service": "test",
    })
    assert resp.status_code == 422


def test_diagnose_truncates_long_error_text():
    """error_text > 4000 chars no se rechaza pero se trunca con marker."""
    long_text = "x" * 5000
    # No esperamos 422 — el validator trunca, no rechaza.
    # Para verificar el truncado tenemos que mockear ollama.chat.
    captured = {"prompt": None}

    class FakeChunk:
        def __init__(self, content, done=False):
            self.message = {"content": content}
            self.done = done

    def fake_chat(**kwargs):
        # Capturar el user prompt para inspeccionar.
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "user":
                captured["prompt"] = msg["content"]
        return iter([{"message": {"content": "ok"}, "done": True}])

    class FakeList:
        models = [type("M", (), {"model": "command-r:latest"})()]

    with patch.object(_server.ollama, "list", return_value=FakeList()), \
         patch.object(_server.ollama, "chat", fake_chat):
        _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None
        with _client.stream(
            "POST", "/api/diagnose-error",
            json={"error_text": long_text, "service": "test"},
        ) as resp:
            assert resp.status_code == 200
            list(resp.iter_bytes())  # consume stream
        _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None
    assert captured["prompt"] is not None
    assert "(truncado)" in captured["prompt"]


# ── /api/diagnose-error/execute: whitelist ──────────────────────────────


def test_execute_rejects_empty_command():
    resp = _client.post("/api/diagnose-error/execute", json={"command": ""})
    assert resp.status_code == 403
    assert "vacío" in resp.json()["detail"].lower()


def test_execute_rejects_command_outside_whitelist():
    """`whoami` no está en la whitelist → 403."""
    resp = _client.post("/api/diagnose-error/execute", json={"command": "whoami"})
    assert resp.status_code == 403
    detail = resp.json()["detail"].lower()
    assert "whitelist" in detail or "no está" in detail


def test_execute_rejects_python_command():
    """`python -c 'print(1)'` no está whitelisted → 403."""
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "python -c 'print(1)'"},
    )
    assert resp.status_code == 403


# ── /api/diagnose-error/execute: bloqueo de metachars peligrosos ────────


@pytest.mark.parametrize("dangerous", [
    "rag stats; ls /",            # ; (chain) blocked
    "rag stats | tee /tmp/x",     # | (pipe) blocked
    "rag stats > /tmp/x",         # > (redirect) blocked
    "rag stats < /tmp/x",         # < (redirect) blocked
    "rag stats && rm -rf",        # & (background/chain) blocked
    "rag stats `whoami`",         # ` (command sub) blocked
    "rag stats $(whoami)",        # $ (command sub) blocked
    "rag stats\nls",              # \n (newline) blocked
])
def test_execute_blocks_dangerous_metachars(dangerous):
    """Cada metachar peligroso rebota con 403 — ninguno llega a subprocess."""
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": dangerous},
    )
    assert resp.status_code == 403, (
        f"comando peligroso pasó: {dangerous!r} → {resp.status_code} {resp.text}"
    )


def test_execute_rejects_path_in_binary():
    """`/usr/bin/whoami` o `./rag` rechazados — el binario debe ser por nombre."""
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "/usr/bin/whoami"},
    )
    assert resp.status_code == 403
    assert "path" in resp.json()["detail"].lower() or "nombre" in resp.json()["detail"].lower()


# ── /api/diagnose-error/execute: validators por comando ─────────────────


def test_execute_rag_stats_passes_validation(monkeypatch):
    """`rag stats` sin args → válido. Mockeamos subprocess para no
    depender del binario instalado."""
    class FakeResult:
        returncode = 0
        stdout = "vault: foo\nchunks: 100\n"
        stderr = ""

    monkeypatch.setattr(_server.subprocess, "run", lambda *a, **kw: FakeResult())
    monkeypatch.setattr(
        _server, "_SAFE_RAG", "/Users/fer/.local/bin/rag",
    )
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/Users/fer/.local/bin/rag", _server._validate_rag_args),
    )
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag stats"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exit_code"] == 0
    assert "vault" in body["stdout"]


def test_execute_rejects_rag_index(monkeypatch):
    """`rag index` NO está en READ_ONLY_SUBCMDS — escribe al corpus."""
    monkeypatch.setattr(
        _server, "_SAFE_RAG", "/Users/fer/.local/bin/rag",
    )
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/Users/fer/.local/bin/rag", _server._validate_rag_args),
    )
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag index"},
    )
    assert resp.status_code == 403


def test_execute_rejects_rag_with_path_arg(monkeypatch):
    """`rag stats /foo` rechazado — paths arbitrarios no permitidos."""
    monkeypatch.setattr(
        _server, "_SAFE_RAG", "/Users/fer/.local/bin/rag",
    )
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/Users/fer/.local/bin/rag", _server._validate_rag_args),
    )
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag stats /etc"},
    )
    assert resp.status_code == 403


# ── /api/diagnose-error/execute: subprocess wiring ──────────────────────


def test_execute_uses_minimal_env(monkeypatch):
    """El subprocess se llama con env restringido (PATH/HOME/LANG only)."""
    captured = {"env": None}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return FakeResult()

    monkeypatch.setattr(_server.subprocess, "run", fake_run)
    monkeypatch.setattr(_server, "_SAFE_RAG", "/usr/bin/rag")
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/usr/bin/rag", _server._validate_rag_args),
    )
    _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag stats"},
    )
    assert captured["env"] is not None
    # Solo las 3 keys mínimas — NO leak de OBSIDIAN_RAG_*, OPENAI_API_KEY, etc.
    assert set(captured["env"].keys()) == {"PATH", "HOME", "LANG"}


def test_execute_handles_timeout(monkeypatch):
    """TimeoutExpired → exit_code 124 + flag `timed_out`."""
    import subprocess as _sp

    def fake_run(*args, **kwargs):
        raise _sp.TimeoutExpired(cmd="rag", timeout=15)

    monkeypatch.setattr(_server.subprocess, "run", fake_run)
    monkeypatch.setattr(_server, "_SAFE_RAG", "/usr/bin/rag")
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/usr/bin/rag", _server._validate_rag_args),
    )
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag stats"},
    )
    # Implementación devuelve 200 con exit_code=124 (no 504), porque
    # el endpoint NO falla — el comando sí. El user ve "✗ exit 124".
    assert resp.status_code == 200
    assert resp.json()["exit_code"] == 124
    assert resp.json().get("timed_out") is True


def test_execute_truncates_huge_stdout(monkeypatch):
    """stdout > _DIAGNOSE_OUTPUT_TRUNCATE se trunca al final."""
    huge = "x" * 50_000
    cap = _server._DIAGNOSE_OUTPUT_TRUNCATE

    class FakeResult:
        returncode = 0
        stdout = huge
        stderr = ""

    monkeypatch.setattr(_server.subprocess, "run", lambda *a, **kw: FakeResult())
    monkeypatch.setattr(_server, "_SAFE_RAG", "/usr/bin/rag")
    monkeypatch.setitem(
        _server._DIAGNOSE_COMMAND_REGISTRY, "rag",
        ("/usr/bin/rag", _server._validate_rag_args),
    )
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rag stats"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["stdout"]) == cap


# ── SSE smoke test ──────────────────────────────────────────────────────


def test_diagnose_stream_emits_expected_events(monkeypatch):
    """SSE smoke — verificamos shape del stream sin probar contenido del LLM."""
    class FakeChunk(dict):
        pass

    def fake_chat(**kwargs):
        return iter([
            {"message": {"content": "hello "}, "done": False},
            {"message": {"content": "world"}, "done": True},
        ])

    class FakeList:
        models = [type("M", (), {"model": "command-r:latest"})()]

    monkeypatch.setattr(_server.ollama, "list", lambda: FakeList())
    monkeypatch.setattr(_server.ollama, "chat", fake_chat)
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None

    payload = {"error_text": "boom", "service": "test"}
    with _client.stream("POST", "/api/diagnose-error", json=payload) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()

    assert '"type": "model"' in body
    assert "command-r:latest" in body
    assert '"type": "token"' in body
    assert "hello" in body
    assert '"type": "done"' in body
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None


def test_diagnose_stream_emits_error_event_on_llm_failure(monkeypatch):
    """ollama.chat tira → stream emite type=error con el detail."""
    def fake_chat(**kwargs):
        raise RuntimeError("ollama unreachable")

    class FakeList:
        models = [type("M", (), {"model": "command-r:latest"})()]

    monkeypatch.setattr(_server.ollama, "list", lambda: FakeList())
    monkeypatch.setattr(_server.ollama, "chat", fake_chat)
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None

    payload = {"error_text": "boom", "service": "test"}
    with _client.stream("POST", "/api/diagnose-error", json=payload) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()

    assert '"type": "error"' in body
    assert "ollama unreachable" in body
    _server.__dict__["_DIAGNOSE_MODEL_RESOLVED"] = None
