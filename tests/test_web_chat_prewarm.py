"""Tests para el chat-model prewarm del web server.

Guard contra una regresión que pasó el 2026-05-03: `[chat-stream-error]
phase=synthesis exc=timed out ttft_ms=90005 got_first_token=False` con
modelo `qwen2.5:3b` cuando el chat fast-path llega después de un helper
(paraphrases / HyDE / intent / NLI). El helper carga el modelo con
`HELPER_OPTIONS["num_ctx"]=1024` y el fast-path lo pide con
`_LOOKUP_NUM_CTX=4096` → mismatch num_ctx en el mismo modelo → ollama
reinicializa la KV cache → en MPS bajo memory pressure tarda 60-120s →
timeout 90s.

Fix: el prewarm pinea AMBOS modelos (chat principal Y `_LOOKUP_MODEL`)
con el `num_ctx` correcto, así el modelo arranca cargado con
`context_length=4096` y el fast-path no fuerza reinit.

Tests cubren:

  1. `_chat_prewarm_targets()` devuelve el modelo principal con
     `_WEB_CHAT_NUM_CTX` y el lookup model con `_LOOKUP_NUM_CTX`.
  2. Si el operador setea `OBSIDIAN_RAG_WEB_CHAT_MODEL == _LOOKUP_MODEL`
     (mismo modelo en ambos roles), targets dedupea a uno solo (no
     queremos pingear dos veces el mismo modelo con el mismo num_ctx).
  3. `_run_chat_prewarm_cycle()` invoca `_OLLAMA_STREAM_CLIENT.chat`
     una vez por target, con el num_ctx correcto.
  4. Si un target falla (modelo no disponible), el OTRO target se sigue
     intentando — guard del fail-mode "skip-cycle entero".
"""

from __future__ import annotations

import pytest

from web import server as server_mod


def test_targets_default_pins_main_and_lookup(monkeypatch):
    """Con la config default (main=qwen2.5:7b, lookup=qwen2.5:3b),
    targets devuelve los dos pares con los `num_ctx` correctos."""
    monkeypatch.setattr(server_mod, "_resolve_web_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(server_mod, "_LOOKUP_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_NUM_CTX", 4096)
    monkeypatch.setattr(server_mod, "_WEB_CHAT_NUM_CTX", 4096)

    targets = server_mod._chat_prewarm_targets()

    assert targets == [("qwen2.5:7b", 4096), ("qwen2.5:3b", 4096)], (
        f"prewarm tiene que pinear main + lookup; got {targets}. "
        "Si esta lista no incluye al lookup model, vuelve el bug del 2026-05-03 "
        "(synthesis falló: timed out)."
    )


def test_targets_dedup_when_main_equals_lookup(monkeypatch):
    """Si el operador setea OBSIDIAN_RAG_WEB_CHAT_MODEL=qwen2.5:3b
    (mismo que `_LOOKUP_MODEL`), no pingueamos dos veces el mismo modelo."""
    monkeypatch.setattr(server_mod, "_resolve_web_chat_model", lambda: "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_NUM_CTX", 4096)
    monkeypatch.setattr(server_mod, "_WEB_CHAT_NUM_CTX", 4096)

    targets = server_mod._chat_prewarm_targets()

    assert targets == [("qwen2.5:3b", 4096)], (
        f"dedup falló cuando main == lookup; got {targets}"
    )


def test_targets_use_lookup_num_ctx_for_lookup_model(monkeypatch):
    """El lookup model usa `_LOOKUP_NUM_CTX` (no `_WEB_CHAT_NUM_CTX`).

    Si alguna vez los defaults divergen — ej. operador setea
    `RAG_LOOKUP_NUM_CTX=2048` para ahorrar memoria — el target del lookup
    tiene que reflejar ESE valor, no el del chat principal. Caso contrario
    el prewarm pinea con num_ctx=`_WEB_CHAT_NUM_CTX` pero el fast-path
    pide num_ctx=`_LOOKUP_NUM_CTX` → mismatch → reinit → bug original
    pero al revés.
    """
    monkeypatch.setattr(server_mod, "_resolve_web_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(server_mod, "_LOOKUP_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_NUM_CTX", 2048)
    monkeypatch.setattr(server_mod, "_WEB_CHAT_NUM_CTX", 4096)

    targets = server_mod._chat_prewarm_targets()

    assert targets == [("qwen2.5:7b", 4096), ("qwen2.5:3b", 2048)]


def test_run_cycle_calls_chat_for_every_target(monkeypatch):
    """`_run_chat_prewarm_cycle` invoca `_OLLAMA_STREAM_CLIENT.chat` una
    vez por target, con el `num_ctx` correcto en `options`."""
    calls: list[dict] = []

    class FakeStreamClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"content": "."}}

    monkeypatch.setattr(server_mod, "_OLLAMA_STREAM_CLIENT", FakeStreamClient())
    monkeypatch.setattr(server_mod, "_resolve_web_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(server_mod, "_LOOKUP_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_NUM_CTX", 4096)
    monkeypatch.setattr(server_mod, "_WEB_CHAT_NUM_CTX", 4096)

    server_mod._run_chat_prewarm_cycle()

    assert len(calls) == 2, f"esperaba 2 calls (main + lookup), got {len(calls)}"
    assert calls[0]["model"] == "qwen2.5:7b"
    assert calls[0]["options"]["num_ctx"] == 4096
    assert calls[0]["options"]["num_predict"] == 1
    assert calls[0]["stream"] is False
    assert calls[1]["model"] == "qwen2.5:3b"
    assert calls[1]["options"]["num_ctx"] == 4096
    assert calls[1]["options"]["num_predict"] == 1


def test_run_cycle_one_target_failure_does_not_block_other(monkeypatch):
    """Si el primer target falla (ollama down, modelo no descargado, network
    blip), el segundo target SE SIGUE INTENTANDO. Pre-fix con un solo
    try/except externo, una falla en el primer modelo cancelaba el
    segundo del mismo cycle."""
    calls: list[dict] = []

    class FakeStreamClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            if kwargs["model"] == "qwen2.5:7b":
                raise RuntimeError("simulated ollama 503")
            return {"message": {"content": "."}}

    monkeypatch.setattr(server_mod, "_OLLAMA_STREAM_CLIENT", FakeStreamClient())
    monkeypatch.setattr(server_mod, "_resolve_web_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(server_mod, "_LOOKUP_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(server_mod, "_LOOKUP_NUM_CTX", 4096)
    monkeypatch.setattr(server_mod, "_WEB_CHAT_NUM_CTX", 4096)

    # No debe propagar la excepción del primer modelo.
    server_mod._run_chat_prewarm_cycle()

    models_called = [c["model"] for c in calls]
    assert "qwen2.5:7b" in models_called, "el primer target se intentó"
    assert "qwen2.5:3b" in models_called, (
        f"el segundo target tiene que intentarse aunque el primero falle; "
        f"models_called={models_called}"
    )
