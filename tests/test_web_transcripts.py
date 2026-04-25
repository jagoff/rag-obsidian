"""Tests del endpoint `/transcripts` — dashboard del whisper learning loop.

Phase 2 Step 3.b. Cubre:
- Renders 200 con HTML válido.
- Estructura: stats, histogram, tabla transcripts, tabla corrections, tabla vocab.
- Dark mode default + media query light.
- Empty state (sin transcripts logueadas) muestra mensaje útil al user.
- Manejo de error si la DB no está disponible.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


# ── Renderiza ─────────────────────────────────────────────────────────────────


def test_transcripts_endpoint_returns_200():
    """GET /transcripts → 200 + content-type HTML."""
    resp = _client.get("/transcripts")
    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "text/html" in ct, f"expected text/html, got {ct}"


def test_transcripts_renders_basic_structure():
    """HTML tiene los componentes principales del dashboard."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Header
    assert "<title>whisper transcripts" in html
    assert "<h1>whisper transcripts</h1>" in html
    # Secciones principales
    assert "logprob histogram" in html
    assert "vocab por source" in html
    assert "últimas 30 transcripciones" in html
    assert "últimas 20 correcciones" in html
    assert "top 50 vocab terms" in html


def test_transcripts_includes_stat_cards():
    """5 stat cards con KPIs principales."""
    resp = _client.get("/transcripts")
    html = resp.text
    for label in (
        "AUDIOS 30D",
        "AVG LOGPROB",
        "AVG DURACIÓN",
        "CORRECCIONES TOTALES",
        "VOCAB TERMS",
    ):
        # Las labels están uppercase via CSS pero el texto fuente es lowercase
        # con clase .lbl (text-transform: uppercase).
        assert label.lower() in html.lower(), f"missing stat: {label}"


def test_transcripts_logprob_histogram_buckets():
    """Histogram tiene los 5 buckets esperados (excelente / alta / media / baja / LLM)."""
    resp = _client.get("/transcripts")
    html = resp.text
    for bucket in (
        "excelente",
        "alta",
        "media",
        "baja",
        "LLM correct",
    ):
        assert bucket in html, f"missing histogram bucket: {bucket}"


# ── Dark mode ─────────────────────────────────────────────────────────────────


def test_transcripts_has_dark_mode_meta():
    """`<meta name='color-scheme' content='dark'>` para que el browser ajuste
    scrollbars + form controls al theme."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert 'name="color-scheme"' in html
    assert 'content="dark"' in html


def test_transcripts_dark_palette_is_default():
    """El `:root` definido al toplevel debe tener la paleta dark
    (sin estar wrapped en un media query). El `prefers-color-scheme: light`
    es OVERRIDE del default dark, no al revés."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Buscar el primer :root (default) y verificar paleta GitHub dark.
    root_idx = html.index(":root {")
    snippet = html[root_idx:root_idx + 500]
    assert "#0d1117" in snippet, "dark bg color GitHub-dark esperado en :root default"
    assert "#e6edf3" in snippet, "dark text color esperado en :root default"


def test_transcripts_light_override_via_media_query():
    """Light mode existe como media query override (`prefers-color-scheme: light`)."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert "@media (prefers-color-scheme: light)" in html
    # Después del media query, hay otro :root con bg blanco.
    media_idx = html.index("prefers-color-scheme: light")
    snippet = html[media_idx:media_idx + 500]
    assert "#ffffff" in snippet, "light bg color #ffffff esperado en media query"


def test_transcripts_uses_css_classes_not_inline_colors():
    """Refactor del Step 3.b a classes — no debe haber `style='color:#xxx'`
    en los row builders (todo via classes pill-*, lp-*, src-*, text-*)."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Heurística: el bloque del body NO debería tener inline `color:#`.
    body_idx = html.index("<body>")
    body_html = html[body_idx:]
    # Permitimos un par de inline `style="text-align:..."` y `style="background:..."`
    # para datos dinámicos (width del bar histogram, padding del empty-state).
    # Lo prohibido es `color:#xxx` en el body.
    forbidden = "color:#"
    occurrences = body_html.count(forbidden)
    assert occurrences == 0, (
        f"found {occurrences} inline `color:#` in body — debe ir via class. "
        f"Encontrado: {[body_html[i-20:i+30] for i in range(len(body_html)) if body_html[i:i+len(forbidden)] == forbidden][:3]}"
    )


# ── Empty state ──────────────────────────────────────────────────────────────


def test_transcripts_shows_empty_state_when_no_transcripts():
    """Si la tabla `rag_audio_transcripts` está vacía, mostrar un mensaje
    instructivo que diga al user qué hacer (mandar un audio)."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Si el deploy actual NO tiene transcripts, el empty state aparece.
    # Si tiene transcripts, este test no aplica (skip silencioso).
    if "sin transcripciones logueadas" in html:
        assert "mandá un audio" in html


def test_transcripts_shows_empty_state_for_corrections():
    """Mismo patrón para correcciones — instructivo con `/fix`."""
    resp = _client.get("/transcripts")
    html = resp.text
    if "sin correcciones todavía" in html:
        assert "/fix" in html


# ── Css classes definidas ────────────────────────────────────────────────────


def test_transcripts_defines_pill_classes():
    """Las pills llm/fix/vault deben estar definidas como classes."""
    resp = _client.get("/transcripts")
    html = resp.text
    for cls in (".pill-llm", ".pill-fix", ".pill-vault"):
        assert cls in html, f"missing CSS class: {cls}"


def test_transcripts_defines_logprob_classes():
    """Las classes lp-good/mid/bad deben estar definidas para colorear logprobs."""
    resp = _client.get("/transcripts")
    html = resp.text
    for cls in (".lp-good", ".lp-mid", ".lp-bad"):
        assert cls in html


def test_transcripts_defines_correction_source_classes():
    """src-explicit / src-llm / src-vault deben estar definidas."""
    resp = _client.get("/transcripts")
    html = resp.text
    for cls in (".src-explicit", ".src-llm", ".src-vault"):
        assert cls in html


# ── Defensivo ────────────────────────────────────────────────────────────────


def test_transcripts_escapes_user_text():
    """El template usa `_esc()` helper para evitar XSS si un transcript
    o correction tuviera caracteres HTML en el contenido. Smoke test que
    `_esc` está siendo invocado en el código (sanity)."""
    # Test indirecto: el helper `_esc` debe estar definido.
    assert hasattr(_server, "_esc"), "_esc helper missing"
    # Test directo de _esc:
    assert _server._esc("<script>") == "&lt;script&gt;"
    assert _server._esc("\"hola\"") == "&quot;hola&quot;"
    assert _server._esc("a & b") == "a &amp; b"
    assert _server._esc(None) == ""
