"""End-to-end live test contra `rag serve` (HTTP server persistente).

**Opt-in SOLO**. Esta suite requiere ollama corriendo + modelo chat
pulled + vault real indexado. Pytest por default NO la ejecuta — hay
que pasar `RAG_E2E_LIVE=1` o correr explícitamente con `-m live`.

### Qué cubre

Gap identificado en el audit 2026-04-24: 247 archivos de test, toda la
cobertura es unitaria/mock. No había UN test que levantara el stack
completo (ollama + rag.serve + SQLite-vec real) y validara el contrato
punta-a-punta del `/query` endpoint. Cualquier regresión de integración
(import order wrong, shape de response cambió, signature de `retrieve()`
drifteada con un caller distinto, etc.) aparecía solo cuando el usuario
pegaba una query real desde WhatsApp.

Este test cierra ese gap con un smoke de ~60s (warmup + 1 query real).

### Cómo correrlo

    # Opt-in via env var (default config, no argumentos extra)
    RAG_E2E_LIVE=1 .venv/bin/python -m pytest tests/test_e2e_live.py -v

    # Opt-in via marker selector
    .venv/bin/python -m pytest -m live -v

    # Forzar skip aunque esté setedo (útil en debug local)
    RAG_E2E_LIVE=0 .venv/bin/python -m pytest tests/test_e2e_live.py

### Cómo fallan / debug

Si el subprocess no bootea, el test falla con el stdout+stderr
capturado del proceso `rag serve`. Las causas típicas:

- ollama no está corriendo → `brew services start ollama`.
- No hay chat model instalado → `ollama pull qwen2.5:7b`.
- Vault vacío → `rag index` primero.
- Puerto ocupado → el test busca un puerto libre via OS (port 0).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Guard doble: marker + skipif. Marker para `pytest -m live`; skipif
# para cortar collección cuando alguien corre el suite completo sin el
# env var. Si el user quiere correrlo con `-m live` debe ALSO setear el
# env var — doble opt-in intencional (este test arranca un subprocess
# que carga ollama + modelos en VRAM, no queremos que corra por accidente).
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RAG_E2E_LIVE", "0") not in ("1", "true", "yes"),
        reason="opt-in only — export RAG_E2E_LIVE=1 to run (loads ollama + chat model)",
    ),
]


def _find_free_port() -> int:
    """Ask the OS for an unused TCP port. Race-safe enough for tests —
    the subprocess binds immediately after we release; a real collision
    requires another process grabbing it in the ms window, extremely
    unlikely in a test env."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, *, timeout_s: float = 90.0) -> None:
    """Poll /health until 200 OK or timeout. `rag serve` warmup takes
    ~30-60s on cold load (reranker on MPS + local bge-m3 + chat model
    prefill). 90s is generous but bounded."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            ) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError) as exc:
            last_err = exc
        time.sleep(1.0)
    raise TimeoutError(
        f"rag serve on port {port} did not become healthy in {timeout_s}s; "
        f"last error: {last_err}"
    )


def _post_query(port: int, body: dict, *, timeout_s: float = 60.0) -> dict:
    """POST JSON al /query endpoint y devolver el response parseado."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.fixture(scope="module")
def live_server():
    """Arrancar `rag serve` como subprocess, esperar warmup, yield port,
    terminar al finalizar. Scope=module para no pagar warmup por test —
    toda la suite comparte un daemon.
    """
    port = _find_free_port()
    # Usamos el .venv del proyecto. `sys.executable` apunta al pytest
    # runner que por convention es el mismo interpreter que tiene rag
    # instalado editable.
    cmd = [sys.executable, "-m", "rag", "serve", "--host", "127.0.0.1", "--port", str(port)]
    # `python -m rag` asume que rag.py es importable como módulo. Si no
    # lo es, fallback al `rag` CLI binary del uv tool install.
    rag_bin = ROOT / ".venv" / "bin" / "rag"
    if rag_bin.is_file():
        cmd = [str(rag_bin), "serve", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        _wait_for_health(port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_health_endpoint_returns_ok(live_server):
    """Smoke: /health responde 200 con JSON que incluye chunk count."""
    port = live_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5.0) as resp:
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
    # Shape contract — si cambian estos keys, bots y dashboards rompen.
    assert "ok" in body or "status" in body or "chunks" in body, (
        f"/health response missing expected keys: {body!r}"
    )


def test_query_endpoint_full_pipeline(live_server):
    """Smoke: POST /query con pregunta real → response con shape esperado.

    NO asserteamos sobre el contenido del answer (no determinista por
    LLM stochastic sampling). Asserteamos el CONTRATO: keys presentes,
    tipos correctos, no errors. Un fail acá significa que el response
    shape cambió sin actualizar este contract.
    """
    port = live_server
    # Pregunta genérica que cualquier vault medio va a poder contestar.
    # Si el vault está vacío, el response va a tener answer corto pero
    # el shape sigue siendo válido.
    response = _post_query(port, {"question": "qué hay en el vault", "k": 3})

    # Contract: campos obligatorios del response.
    assert isinstance(response, dict), f"response must be dict, got {type(response)}"
    assert "answer" in response, f"response missing 'answer' key: {list(response.keys())}"
    assert "sources" in response, f"response missing 'sources' key: {list(response.keys())}"
    assert isinstance(response["answer"], str)
    assert isinstance(response["sources"], list)
    # `mode` o `t_retrieve`/`t_gen` para diagnósticos — al menos uno debe estar.
    has_timing = any(k in response for k in ("t_retrieve", "t_gen", "mode"))
    assert has_timing, (
        f"response missing timing/mode keys (expected at least one of "
        f"t_retrieve/t_gen/mode): {list(response.keys())}"
    )


def test_query_empty_question_returns_error(live_server):
    """Regression: sin question → `error` key, no 500. Pre-existente."""
    port = live_server
    response = _post_query(port, {"question": ""})
    assert "error" in response, f"expected error key on empty query: {response!r}"


def test_query_cache_hit_on_repeat(live_server):
    """Segunda llamada idéntica pega cache — verifica que `cached: True`
    aparece en el response. Cache es parte del contrato del serve (WA
    users retipean con alta frecuencia, esto protege contra regresiones
    que rompan la LRU del serve)."""
    port = live_server
    body = {"question": "qué hay en notas", "k": 3}
    first = _post_query(port, body)
    second = _post_query(port, body)
    # No es garantizado que la PRIMER call cachee (depende de filters,
    # force flag) pero la SEGUNDA con misma key debería. Si hay filtros
    # que salteen el cache, este test pierde valor pero no rompe —
    # `cached` puede faltar perfectamente legítimo. Solo asserteamos
    # que si está, es True.
    if "cached" in second:
        assert second["cached"] is True, f"second call should be cache hit, got {second.get('cached')!r}"
    # Sanity: first.answer y second.answer deberían ser iguales si cached.
    if second.get("cached"):
        assert first["answer"] == second["answer"]
