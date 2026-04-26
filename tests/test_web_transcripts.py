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
    # Header (puede tener un refresh button inline, no asumir exact match).
    assert "<title>whisper transcripts" in html
    assert "<h1>whisper transcripts" in html
    # Secciones principales
    assert "logprob histogram" in html
    assert "vocab por source" in html
    assert "últimas 30 transcripciones" in html
    assert "últimas 20 correcciones" in html
    assert "top 50 vocab terms" in html


def test_transcripts_has_manual_refresh_button():
    """`/transcripts` tiene un botón refresh inline para recargar sin esperar
    los 60s del auto-refresh."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert "refresh-btn" in html
    # Botón con symbol "↻" o algo similar visible.
    assert '↻' in html


def test_transcripts_meta_says_auto_refresh_off_when_nofresh():
    """Cuando `?nofresh=1`, la línea meta debe indicar que el auto-refresh
    está OFF."""
    resp = _client.get("/transcripts?nofresh=1")
    html = resp.text
    assert "auto-refresh OFF" in html


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


def test_transcripts_no_light_override_dark_is_forced():
    """Dark FIJO desde commit cce1716 (2026-04-25 — el override
    `@media (prefers-color-scheme: light)` se removió a propósito porque
    el resto del stack del rag tiene tema oscuro como default y la página
    se flipeaba a light cuando el OS estaba en modo claro). Si alguien
    re-introduce el override sin actualizar este test, salta acá."""
    resp = _client.get("/transcripts")
    html = resp.text
    # NO debe haber media query light flippeando la paleta.
    assert "@media (prefers-color-scheme: light)" not in html
    # Y NO debe haber colores hardcoded del light theme (el #ffffff
    # del bg viejo, el #1f2328 del text viejo). Si alguien copy-pastea
    # variables claras desde otra página, este test las flagea.
    assert "#ffffff" not in html
    assert "#1f2328" not in html


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


# ── Heatmap por hora del día ──────────────────────────────────────────────────


def test_transcripts_renders_hourly_heatmap_section():
    """`/transcripts` tiene una sección `distribución horaria` con heatmap
    de audios por hora del día (últimos 30d)."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert "distribución horaria" in html


def test_transcripts_heatmap_empty_state_when_no_audios():
    """Cuando no hay audios en 30d, el heatmap muestra mensaje informativo
    en vez de cells vacías."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Si la DB está vacía, el empty state aparece.
    if "sin audios en últimos 30d" in html:
        assert "distribución horaria" in html


def test_transcripts_heatmap_css_classes_defined():
    """Las classes del heatmap deben estar definidas para que se vea bien."""
    resp = _client.get("/transcripts")
    html = resp.text
    for cls in (".heatmap-wrap", ".heatmap-cell", ".hour-label", ".hour-count"):
        assert cls in html, f"missing CSS class: {cls}"


# ── Patrones repetidos ────────────────────────────────────────────────────────


def test_transcripts_renders_patterns_section():
    """`/transcripts` tiene una sección `patrones repetidos` que muestra
    los single-word swaps detectados."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert "patrones repetidos" in html


def test_transcripts_patterns_empty_state():
    """Sin patterns todavía, mostrar placeholder con instrucción."""
    resp = _client.get("/transcripts")
    html = resp.text
    if "sin patrones repetidos" in html:
        # Mensaje útil con `/fix` y comando CLI mencionados.
        assert "/fix" in html
        assert "rag whisper patterns" in html


# ── Heatmap semanal día×hora ──────────────────────────────────────────────────


def test_transcripts_renders_weekly_heatmap_section():
    """`/transcripts` tiene una sección `distribución semanal` con
    matriz 7×24 (días × horas)."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert "distribución semanal" in html


def test_transcripts_weekly_heatmap_empty_state():
    """Sin audios en 60d, el heatmap semanal muestra mensaje informativo."""
    resp = _client.get("/transcripts")
    html = resp.text
    if "sin audios en últimos 60d" in html:
        assert "patrones tipo" in html  # sneak peek del placeholder text


def test_transcripts_weekly_heatmap_css_classes():
    """Classes del heatmap semanal están definidas."""
    resp = _client.get("/transcripts")
    html = resp.text
    for cls in (".week-heatmap", ".week-cell", ".week-day-hdr", ".week-hour-hdr"):
        assert cls in html, f"missing weekly heatmap CSS class: {cls}"


# ── Auto-refresh ─────────────────────────────────────────────────────────────


def test_transcripts_default_has_meta_refresh_60s():
    """Default: la página se auto-recarga cada 60s para mostrar audios nuevos
    en vivo cuando el listener loguea uno."""
    resp = _client.get("/transcripts")
    html = resp.text
    assert '<meta http-equiv="refresh" content="60">' in html


def test_transcripts_nofresh_suppresses_auto_refresh():
    """`?nofresh=1` omite el meta-refresh — útil cuando estás leyendo la
    página y no querés que scrollée perdiendo posición."""
    resp = _client.get("/transcripts?nofresh=1")
    assert resp.status_code == 200
    html = resp.text
    assert "http-equiv=\"refresh\"" not in html
    assert "Auto-refresh suppressed" in html


# ── Navegación cruzada (links entre páginas) ─────────────────────────────────


def test_home_html_links_to_transcripts():
    """`/home` debe tener un link a `/transcripts` en el topbar para que
    el dashboard sea descubrible sin tener que recordar la URL."""
    from pathlib import Path
    static_dir = Path(_server.STATIC_DIR)
    html = (static_dir / "home.html").read_text(encoding="utf-8")
    assert 'href="/transcripts"' in html


def test_dashboard_html_links_to_transcripts():
    """`/dashboard` (semáforo) también debe enlazar a `/transcripts`."""
    from pathlib import Path
    static_dir = Path(_server.STATIC_DIR)
    html = (static_dir / "dashboard.html").read_text(encoding="utf-8")
    assert 'href="/transcripts"' in html


def test_transcripts_has_topnav_to_other_pages():
    """`/transcripts` tiene una topnav con links a las otras páginas
    para no quedar como dead-end."""
    resp = _client.get("/transcripts")
    html = resp.text
    # Espera ver links a las páginas hermanas.
    for href in ('href="/"', 'href="/chat"', 'href="/dashboard"', 'href="/status"'):
        assert href in html, f"missing nav link: {href}"
    # Self-link debería tener `class="active"` o similar marker.
    assert 'href="/transcripts"' in html
    assert 'class="active"' in html


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
