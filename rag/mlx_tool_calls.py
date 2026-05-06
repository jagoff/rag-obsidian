"""Tool-calling adapter para Qwen2.5/3 via MLX (Ola 5).

Qwen2.5-Instruct y Qwen3-Instruct soportan tool calling nativo via el
chat template estándar de Hugging Face: pasás `tools=[...]` a
`tokenizer.apply_chat_template(...)` y el modelo emite uno o más bloques
`<tool_call>{...}</tool_call>` con JSON `{"name": "...", "arguments": {...}}`.

Este módulo expone:

- `parse_tool_calls(text)` — parsea los bloques `<tool_call>` del output
  bruto y devuelve una lista `Message.ToolCall` (shape Ollama) o `None`.
- `strip_tool_call_blocks(text)` — quita los marcadores del texto para
  devolver la prosa visible al user (puede ser `None` si solo había
  tool calls).

Uso desde `MLXBackend.chat()` (cuando se cablea la integración):

```python
from rag.mlx_tool_calls import parse_tool_calls, strip_tool_call_blocks

text = mlx_lm.generate(...)
calls = parse_tool_calls(text)
if calls:
    content = strip_tool_call_blocks(text)
    return ChatResponse(message=Message(content=content, tool_calls=calls), ...)
```

Diseñado para ser robusto a:
- Múltiples bloques `<tool_call>` en un mismo response.
- Whitespace + newlines alrededor del JSON.
- JSON con trailing commas / single quotes (best-effort repair).
- Modelos que emiten arguments como string en vez de dict.
- Algunos chat templates emiten `args` en vez de `arguments`.

Si el parse falla en TODOS los bloques, devuelve `None` y el caller trata
el response como texto plano.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["parse_tool_calls", "strip_tool_call_blocks"]

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL,
)


def _repair_json(raw: str) -> str:
    """Best-effort cleanup: single quotes → double, drop trailing commas."""
    cleaned = raw.replace("'", '"')
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def parse_tool_calls(text: str | None) -> list[Any] | None:
    """Parse Qwen `<tool_call>{...}</tool_call>` blocks → ollama-shape list.

    Returns None when no parseable tool calls are found.
    """
    if not text or "<tool_call>" not in text:
        return None

    from rag.llm_backend import Message

    out: list[Any] = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            try:
                obj = json.loads(_repair_json(raw))
            except json.JSONDecodeError:
                continue
        name = obj.get("name")
        if not name or not isinstance(name, str):
            continue
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        out.append(
            Message.ToolCall(
                function=Message.ToolCall.Function(name=name, arguments=args)
            )
        )
    return out or None


def strip_tool_call_blocks(text: str) -> str | None:
    """Remove `<tool_call>...</tool_call>` markers. Returns None when only
    whitespace remains (caller should set content=None on the response)."""
    stripped = re.sub(
        r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL,
    ).strip()
    return stripped or None
