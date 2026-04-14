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


# ── Bold ─────────────────────────────────────────────────────────────────────


def test_bold_strips_asterisks():
    text = "Mirá **esto** con atención."
    r = _plain(rag.render_response(text))
    assert "**" not in r
    assert "esto" in r
    assert "Mirá " in r
    assert " con atención." in r


def test_bold_label_pattern_like_list_item():
    # Caso real que pediste arreglar: "**Amplificadores:**" en listas.
    text = "- **Amplificadores:** Marshall, Fender\n- **Pedales:** Boss"
    r = _plain(rag.render_response(text))
    assert "**" not in r
    assert "Amplificadores:" in r
    assert "Pedales:" in r
    assert "Marshall" in r


def test_bold_multiple_per_line():
    text = "Entre **foo** y **bar** hay diferencia."
    r = _plain(rag.render_response(text))
    assert "**" not in r
    assert "foo" in r
    assert "bar" in r


def test_bold_inside_inline_code_stays_literal():
    # Si los ** están dentro de `code`, son parte del código — no tocar.
    text = "El operador `**` eleva al cuadrado."
    r = _plain(rag.render_response(text))
    # Inline code strip: los backticks se van, pero los ** quedan.
    assert "**" in r
    assert "eleva al cuadrado" in r


def test_bold_inside_fence_stays_literal():
    text = "```py\nx = a ** 2\n```"
    r = _plain(rag.render_response(text))
    # Dentro del fence, los ** son código literal.
    assert "x = a ** 2" in r
    # Y el wrapper ``` no aparece.
    assert "```" not in r


def test_bold_empty_marker_is_not_consumed():
    # `****` vacío — la regex exige contenido no-vacío, queda literal.
    text = "Nada **** acá."
    r = _plain(rag.render_response(text))
    assert "****" in r


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


# ── External URLs ────────────────────────────────────────────────────────────
# Real case from a "bookmarks de películas" chat: command-r emits external
# links as `[Label](https://...)`. Render symmetric with note links —
# `label (url)` both visible, OSC 8 hyperlinks on each span so both are
# Cmd/Ctrl-clickable. Preserves the markdown-source look of the screenshot.


def _styles(rendered) -> list[tuple[str, str]]:
    """List of (text, style) spans — for asserting that styles include link."""
    return [(span.text, str(span.style)) for span in rendered.divide([len(rendered)])[0].split()]


def test_url_link_renders_label_and_url():
    text = "Mirá [Cliver.tv](https://www2.cliver.me/) para ver pelis."
    r = _plain(rag.render_response(text))
    assert "Cliver.tv" in r
    # URL visible al lado del label, entre paréntesis dim — mismo shape
    # que note-md. OSC 8 hyperlinks hacen ambos clickeables.
    assert "https://www2.cliver.me/" in r
    assert "(https://www2.cliver.me/)" in r
    # Markdown wrappers de corchetes no aparecen (reemplazados por paren dim).
    assert "[" not in r
    assert "](" not in r


def test_url_link_with_special_chars_in_url():
    # Real case: query strings con `=` y `&` no deben romper el match.
    text = "[Search](https://example.com/q?x=1&y=2)"
    r = _plain(rag.render_response(text))
    assert "Search" in r
    assert "https://example.com/q?x=1&y=2" in r


def test_bare_url_renders_as_clickable():
    text = "Visitá https://example.com hoy."
    r = _plain(rag.render_response(text))
    # URL bare se imprime tal cual (no hay label que mostrar).
    assert "https://example.com" in r
    assert "Visitá " in r
    assert " hoy." in r


def test_url_link_does_not_consume_md_link_after():
    # Mezcla: URL externa primero, nota .md después — ambas se rendereen.
    text = "[Web](https://x.com) y [nota](02-Areas/X.md)"
    r = _plain(rag.render_response(text))
    assert "Web" in r
    assert "nota" in r
    assert "02-Areas/X.md" in r
    assert "https://x.com" in r


def test_url_attaches_link_style():
    # OSC 8 hyperlink sale via Rich — style del span debe incluir 'link'.
    rendered = rag.render_response("[Anchor](https://example.com/foo)")
    full_style = " ".join(str(span.style) for span in rendered.spans)
    assert "link" in full_style
    assert "https://example.com/foo" in full_style


def test_md_link_with_md_extension_in_https_url():
    # Edge: GitHub raw link a un .md externo. URL debe ganar a NOTE_LINK_RE
    # (URL es más específica — exige `://`). Si NOTE_LINK_RE ganara,
    # _file_link_style trataría la URL como path relativo al vault.
    text = "[README](https://raw.githubusercontent.com/x/y/main/README.md)"
    rendered = rag.render_response(text)
    full_style = " ".join(str(span.style) for span in rendered.spans)
    # El link debe apuntar al https, no a file://.../README.md.
    assert "link https://" in full_style
    assert "link file://" not in full_style


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
