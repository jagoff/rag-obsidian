"""Tests de `render_response`: markdown → rich Text con gutter + links.

El rendering vive en terminal — markdown raw (``` y backticks) no se parsea
solo. Este test fija la expectativa: los fences se despliegan con gutter
(`  │ `) y bold white, inline code con bold cyan, y los patterns viejos de
link/ext siguen andando sin regresión.
"""
import rag


def _plain(rendered) -> str:
    """Texto sin estilos — para assertions de contenido."""
    return rendered.plain


# ── Code fences ──────────────────────────────────────────────────────────────


def test_fence_strips_backticks_and_lang():
    text = "Corré esto:\n\n```bash\nls -la\n```\n\nListo."
    r = _plain(rag.render_response(text))
    assert "```" not in r
    assert "bash" not in r
    assert "ls -la" in r
    assert "Corré esto:" in r
    assert "Listo." in r


def test_fence_content_gets_gutter_prefix():
    text = "```\necho hola\n```"
    r = _plain(rag.render_response(text))
    # Cada línea del bloque va prefijada con el gutter.
    assert "  │ echo hola" in r


def test_fence_multiline_each_line_gutter():
    text = "```python\na = 1\nb = 2\n```"
    r = _plain(rag.render_response(text))
    assert "  │ a = 1" in r
    assert "  │ b = 2" in r


def test_fence_without_language_works():
    text = "```\nclaude --flag\n```"
    r = _plain(rag.render_response(text))
    assert "  │ claude --flag" in r
    assert "```" not in r


def test_multiple_fences_all_rendered():
    text = "Paso 1:\n```\na\n```\nPaso 2:\n```\nb\n```"
    r = _plain(rag.render_response(text))
    assert "  │ a" in r
    assert "  │ b" in r
    assert "Paso 1:" in r
    assert "Paso 2:" in r


def test_unclosed_fence_falls_back_to_literal():
    # Si el LLM se olvida de cerrar, que no se coma el resto del texto.
    # Actualmente CODE_FENCE_RE es greedy-non-greedy, sin closing no matchea.
    text = "```bash\nls -la"
    r = _plain(rag.render_response(text))
    # Como no hay close, el texto queda tal cual (incluye los backticks).
    assert "```bash" in r or "ls -la" in r


# ── Inline code ───────────────────────────────────────────────────────────────


def test_inline_code_strips_backticks():
    text = "Usá `ls -la` para listar."
    r = _plain(rag.render_response(text))
    assert "`" not in r
    assert "ls -la" in r
    assert "Usá " in r
    assert " para listar." in r


def test_inline_code_multiple_per_line():
    text = "Flags: `-v` o `--verbose`."
    r = _plain(rag.render_response(text))
    assert "`" not in r
    assert "-v" in r
    assert "--verbose" in r


def test_inline_code_inside_fence_is_not_reparsed():
    # Los backticks dentro de un fence son literales del código.
    text = "```bash\necho `date`\n```"
    r = _plain(rag.render_response(text))
    # El gutter aplica, y los backticks internos no se interpretan como inline.
    assert "  │ echo `date`" in r


# ── Regresión: features existentes siguen andando ────────────────────────────


def test_markdown_link_still_renders():
    text = "Mirá [Mi nota](02-Areas/mi-nota.md) para más."
    r = _plain(rag.render_response(text))
    assert "Mi nota" in r
    assert "02-Areas/mi-nota.md" in r


def test_bare_path_link_still_renders():
    text = "Ver [02-Areas/nota.md] para detalle."
    r = _plain(rag.render_response(text))
    assert "02-Areas/nota.md" in r


def test_ext_block_still_renders_with_warning():
    text = "Base: X. <<ext>>Contexto agregado<</ext>> Fin."
    r = _plain(rag.render_response(text))
    assert "⚠" in r
    assert "Contexto agregado" in r
    assert "<<ext>>" not in r
    assert "<</ext>>" not in r


# ── Combinaciones (el caso real que motivó esto) ─────────────────────────────


def test_prose_plus_fence_plus_link_roundtrip():
    # Mezcla típica de respuesta del RAG con comando bash + cita de nota.
    text = (
        "Ejecuta el siguiente comando:\n\n"
        "```bash\n"
        "claude --dangerously-skip-permissions --dangerously-load-development-channels server:claude-peers\n"
        "```\n\n"
        "Este comando se encuentra en [Claude-peers](00-Inbox/Claude-peers.md)."
    )
    r = _plain(rag.render_response(text))
    # El comando aparece limpio, con gutter, sin los backticks.
    assert "```" not in r
    assert "  │ claude --dangerously-skip-permissions" in r
    # La prosa antes y después sobrevive.
    assert "Ejecuta el siguiente comando:" in r
    assert "Este comando se encuentra en" in r
    # Link sigue funcionando.
    assert "Claude-peers" in r
    assert "00-Inbox/Claude-peers.md" in r
