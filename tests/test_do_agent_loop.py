"""Tests para el loop agéntico de `rag do` (CLI tool-calling).

Cubre la integración end-to-end del comando `rag do` definido en
`rag.py:19692-19822`: parsea una instrucción, corre un loop con
`_chat_capped_client().chat(tools=...)`, ejecuta los tool_calls que
devuelve el modelo (search/read/list/propose_write/weather) y
termina cuando el modelo responde sin tool_calls.

Estos son los primeros tests del agent loop (pre-fix: CERO cobertura
e2e). No llamamos ollama real ni tocamos filesystem real — todo via
monkeypatch + CliRunner. `VAULT_PATH` queda aislado por la autouse
`_isolate_vault_path` de conftest.

Casos mínimos:
  1. one-shot tool_call → final answer (happy path)
  2. command-r `{"parameters": {...}}` arg unwrapping
  3. tool raisea Exception → loop sigue (no crashea el comando)
  4. max-iterations hit → termina con warning sin crash
"""
from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner

import rag


# ── Helpers ─────────────────────────────────────────────────────────────────


def _fake_ollama_response(tool_calls: list, content: str = "") -> MagicMock:
    """Shape mínimo de una ollama.ChatResponse: .message.{content,tool_calls}.

    El loop agéntico espera además que cada tool_call tenga `.model_dump()`
    (serializable — el loop lo persiste en `messages` para el siguiente
    turno). MagicMock devuelve MagicMock default, que es OK porque
    `ollama.chat` también está mockeado y nunca inspecciona el payload.
    """
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


def _fake_tool_call(name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _install_tool_mocks(monkeypatch, *, capture: dict):
    """Reemplaza los 5 agent tools con mocks que registran llamadas en
    `capture`. Usa funciones reales (no MagicMock) porque el loop usa
    `fn.__name__` como key del dispatch dict — MagicMock.__name__ = "mock"
    por default y rompería el lookup por nombre.
    """

    def fake_search(query, k=5):
        capture.setdefault("search", []).append({"query": query, "k": k})
        return f"(mock) resultados para '{query}'"
    fake_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", fake_search)

    def fake_read_note(path):
        capture.setdefault("read_note", []).append({"path": path})
        return f"(mock) contenido de {path}"
    fake_read_note.__name__ = "_agent_tool_read_note"
    monkeypatch.setattr(rag, "_agent_tool_read_note", fake_read_note)

    def fake_list_notes(folder=None, tag=None, limit=30):
        capture.setdefault("list_notes", []).append(
            {"folder": folder, "tag": tag, "limit": limit}
        )
        return "(mock) listado"
    fake_list_notes.__name__ = "_agent_tool_list_notes"
    monkeypatch.setattr(rag, "_agent_tool_list_notes", fake_list_notes)

    def fake_weather(location=None):
        capture.setdefault("weather", []).append({"location": location})
        return "(mock) clima"
    fake_weather.__name__ = "_agent_tool_weather"
    monkeypatch.setattr(rag, "_agent_tool_weather", fake_weather)

    def fake_propose_write(path, content, rationale=""):
        capture.setdefault("propose_write", []).append(
            {"path": path, "content": content, "rationale": rationale}
        )
        return f"(mock) propuesta {path}"
    fake_propose_write.__name__ = "_agent_tool_propose_write"
    monkeypatch.setattr(rag, "_agent_tool_propose_write", fake_propose_write)


def _install_base_mocks(monkeypatch):
    """Silencia warmup + resolve_chat_model (no queremos warmup real ni
    llamadas a ollama para descubrir el chat model)."""
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")


# ── Caso 1: one-shot tool_call → final answer ───────────────────────────────


def test_do_one_shot_tool_call_then_final_answer(monkeypatch):
    """El LLM llama search una vez, recibe el resultado y responde con
    texto plano (sin tool_calls). Verificamos que:
      - el tool fue invocado con los args del tool_call,
      - el CLI termina con exit_code 0,
      - la respuesta final del LLM aparece en stdout.
    """
    _install_base_mocks(monkeypatch)
    captured: dict = {}
    _install_tool_mocks(monkeypatch, capture=captured)

    FINAL_ANSWER = "Encontré tres notas sobre ikigai."

    # Script de 2 turnos del LLM:
    #   turn 0 → tool_call(search, query="ikigai")
    #   turn 1 → final answer (no tool_calls)
    responses = [
        _fake_ollama_response(
            [_fake_tool_call("_agent_tool_search", {"query": "ikigai", "k": 5})]
        ),
        _fake_ollama_response([], content=FINAL_ANSWER),
    ]

    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "4", "buscá notas de ikigai"],
    )

    assert result.exit_code == 0, result.output
    # El tool fue invocado con los args correctos
    assert captured.get("search") == [{"query": "ikigai", "k": 5}]
    # El LLM fue llamado exactamente 2 veces (1 tool-round + 1 final)
    assert fake_client.chat.call_count == 2
    # La respuesta final aparece en stdout (renderizada por render_response)
    assert "ikigai" in result.output.lower()


# ── Caso 2: command-r arg unwrapping ────────────────────────────────────────


def test_do_unwraps_command_r_parameters_shape(monkeypatch):
    """command-r envuelve los args como `{"tool_name": "...", "parameters": {...}}`.
    El loop en rag.py:19757-19764 desenvuelve esa shape antes de llamar al
    tool real. Verificamos que el mock de search recibe `{"query": "foo"}`,
    NO `{"parameters": {"query": "foo"}}`.
    """
    _install_base_mocks(monkeypatch)
    captured: dict = {}
    _install_tool_mocks(monkeypatch, capture=captured)

    responses = [
        _fake_ollama_response([
            _fake_tool_call("_agent_tool_search", {
                "tool_name": "_agent_tool_search",
                "parameters": {"query": "foo", "k": 3},
            }),
        ]),
        _fake_ollama_response([], content="OK listo."),
    ]

    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "4", "test command-r"],
    )

    assert result.exit_code == 0, result.output
    # El tool fue invocado con los args DESENVUELTOS (sin "parameters")
    assert captured.get("search") == [{"query": "foo", "k": 3}]


# ── Caso 3: tool raisea → loop sigue, no crashea ───────────────────────────


def test_do_tool_raises_loop_continues(monkeypatch):
    """Si el tool raisea, el `except Exception` del loop (rag.py:19776)
    convierte el error en un mensaje `tool` role y el loop continúa.
    Verificamos que:
      - el comando NO crashea (exit_code 0),
      - el LLM recibe un 2do turno (prueba de que el loop siguió),
      - la final answer del 2do turno aparece en stdout.
    """
    _install_base_mocks(monkeypatch)
    captured: dict = {}
    _install_tool_mocks(monkeypatch, capture=captured)

    # Override search para que raisee
    def exploding_search(query, k=5):
        captured.setdefault("search", []).append({"query": query, "k": k})
        raise RuntimeError("db corrupta")
    exploding_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", exploding_search)

    responses = [
        _fake_ollama_response([
            _fake_tool_call("_agent_tool_search", {"query": "ikigai"}),
        ]),
        _fake_ollama_response([], content="Disculpá, tool falló pero respondo igual."),
    ]

    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "4", "exploding tool test"],
    )

    assert result.exit_code == 0, result.output
    # Search fue llamado (y raiseó) — k default es 5
    assert captured.get("search") == [{"query": "ikigai", "k": 5}]
    # Loop siguió → LLM fue invocado 2 veces
    assert fake_client.chat.call_count == 2
    # Final answer presente en stdout (evidencia de que el loop no crasheó)
    assert "respondo igual" in result.output


# ── Caso 4: max-iterations hit ──────────────────────────────────────────────


def test_do_max_iterations_hit(monkeypatch):
    """El LLM siempre devuelve tool_calls (nunca final answer). El loop
    debe salir por `else` del `for it in range(max_iterations)` cuando
    `--max-iterations 3` se agota, sin crashear.

    Verificamos:
      - exit_code 0 (no crash),
      - LLM fue llamado exactamente 3 veces (max_iterations=3),
      - el tool fue invocado 3 veces (una por iteración),
      - stdout contiene el warning de cap alcanzado.
    """
    _install_base_mocks(monkeypatch)
    captured: dict = {}
    _install_tool_mocks(monkeypatch, capture=captured)

    # LLM que SIEMPRE devuelve un tool_call de search
    def always_tool_calls(**kwargs):
        return _fake_ollama_response([
            _fake_tool_call("_agent_tool_search", {"query": "loop forever"}),
        ])

    fake_client = MagicMock()
    fake_client.chat.side_effect = always_tool_calls
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "3", "loop forever"],
    )

    assert result.exit_code == 0, result.output
    # Exactamente 3 rounds (cap respetado)
    assert fake_client.chat.call_count == 3
    # El tool corrió 3 veces (una por iteración)
    assert len(captured.get("search", [])) == 3
    # El warning de cap aparece en stdout (rag.py:19784)
    assert (
        "iteraciones alcanzado" in result.output.lower()
        or "iterations" in result.output.lower()
    )
