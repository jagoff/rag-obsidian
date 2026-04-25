"""Tests para `_format_forced_tool_output` y el datos_block estructurado
del pre-router de /api/chat (2026-04-22 tarde, Fer F. report).

Contexto del fix:
  - Pre-fix, cuando el pre-router disparaba `reminders_due` + `calendar_ahead`
    para "qué tengo para hacer esta semana?", el datos_block se armaba como
    ```
    ## reminders_due
    {"dated": [...], "undated": [...]}
    ## calendar_ahead
    [...]
    ```
    Raw JSON → LLM droppeaba los items `undated`, inventaba items ausentes,
    y sembraba citation artifacts (`[[calendar_ahead]]`).
  - Post-fix: el helper rinde cada tool como markdown estructurado con
    fecha/hora visible, dedup por (name, due), y secciones explícitas
    "con fecha" / "sin fecha" para reminders, "(sin eventos)" cuando
    calendar viene vacío. Además concat un header de anclaje temporal
    ("HOY: martes 22/04/2026", "ESTA SEMANA: lun 20 → dom 26").

Invariants esperados:
  1. JSON shapes oficiales renderean como listas markdown limpias.
  2. Dedup de reminders con mismo (name, due) → 1 bullet.
  3. Bucket labels traducidos al español (overdue → "vencido" etc.)
  4. Empty state declara explícitamente "(sin X)" en vez de sección vacía.
  5. Tool names nunca leak como wikilinks ni headers.
  6. Calendar respeta date_label cuando es presente; fallback a "start".
  7. Helper es total: nunca raisea (malformed JSON → fallback a str(raw)).
"""
from __future__ import annotations

import json


def _import_helper():
    from web.server import _format_forced_tool_output
    return _format_forced_tool_output


# ── reminders_due ─────────────────────────────────────────────────────


def test_reminders_due_dated_and_undated_render_as_sections():
    fmt = _import_helper()
    raw = json.dumps({
        "dated": [
            {"id": "r1", "name": "llamar al dentista",
             "due": "2026-04-24T10:00", "list": "Recordatorios",
             "bucket": "upcoming"},
        ],
        "undated": [
            {"id": "r2", "name": "Mudanza: Arreglar mosquiteros",
             "due": "", "list": "Recordatorios", "bucket": "undated"},
            {"id": "r3", "name": "Comprar entradas Demos",
             "due": "", "list": "Recordatorios", "bucket": "undated"},
        ],
    }, ensure_ascii=False)

    out = fmt("reminders_due", raw)

    # Sección principal
    assert "Recordatorios" in out
    # Con fecha — date + hora
    assert "2026-04-24" in out
    assert "10:00" in out
    assert "llamar al dentista" in out
    # Sin fecha — TODOS los items
    assert "Mudanza: Arreglar mosquiteros" in out
    assert "Comprar entradas Demos" in out
    # No debe filtrar los undated (el bug principal)
    assert out.count("- ") >= 3, f"esperaba ≥3 bullets, got:\n{out}"


def test_reminders_due_deduplicates_same_name_and_due():
    """AppleScript a veces retorna duplicados — dedup explícito por
    (name, due) para que el LLM no los trate como items distintos."""
    fmt = _import_helper()
    raw = json.dumps({
        "dated": [
            {"id": "a", "name": "llamar al dentista",
             "due": "2026-04-24T10:00", "bucket": "upcoming"},
            {"id": "b", "name": "llamar al dentista",
             "due": "2026-04-24T10:00", "bucket": "upcoming"},
        ],
        "undated": [],
    }, ensure_ascii=False)

    out = fmt("reminders_due", raw)
    # Una sola aparición del nombre
    assert out.count("llamar al dentista") == 1, (
        f"esperaba 1 aparición (dedup), got {out.count('llamar al dentista')}:\n{out}"
    )


def test_reminders_due_empty_declares_explicit_empty_state():
    """Empty state debe aparecer COMO TEXTO en el markdown; sin esto
    el LLM interpreta 'sección ausente' como 'no se buscó' y inventa."""
    fmt = _import_helper()
    raw = json.dumps({"dated": [], "undated": []})
    out = fmt("reminders_due", raw)
    # Al menos alguna forma explícita de "sin recordatorios"
    assert any(phrase in out.lower() for phrase in (
        "sin recordator", "no hay recordator", "ningún recordator",
        "0 recordator",
    )), f"esperaba mensaje explícito de empty, got:\n{out}"


def test_reminders_due_bucket_labels_in_spanish():
    """overdue/today/upcoming/undated → vencido/hoy/próximo/sin fecha."""
    fmt = _import_helper()
    raw = json.dumps({
        "dated": [
            {"id": "r1", "name": "tarea vencida",
             "due": "2026-04-10T09:00", "bucket": "overdue"},
            {"id": "r2", "name": "tarea hoy",
             "due": "2026-04-22T15:00", "bucket": "today"},
        ],
        "undated": [],
    }, ensure_ascii=False)
    out = fmt("reminders_due", raw).lower()
    # overdue → "vencido" y today → "hoy" en alguna forma
    assert "vencid" in out or "atrasad" in out or "[overdue]" not in out
    # No rendereamos los labels en inglés
    assert "overdue" not in out
    assert "upcoming" not in out


def test_reminders_due_no_tool_name_leak():
    """Nunca el tool name (`reminders_due`) debe aparecer como header
    ni wikilink ni parte del output — los usuarios lo ven como título
    raro."""
    fmt = _import_helper()
    raw = json.dumps({"dated": [], "undated": []})
    out = fmt("reminders_due", raw)
    assert "reminders_due" not in out
    assert "[[reminders_due]]" not in out
    assert "## reminders_due" not in out


# ── calendar_ahead ────────────────────────────────────────────────────


def test_calendar_ahead_renders_list_with_dates():
    fmt = _import_helper()
    raw = json.dumps([
        {"title": "cumpleaños de Astor", "date_label": "day after tomorrow",
         "time_range": ""},
        {"title": "cumpleaños de Astor", "date_label": "25 abr 2026",
         "time_range": ""},
    ], ensure_ascii=False)

    out = fmt("calendar_ahead", raw)
    # Sección Calendar
    assert "calendar" in out.lower() or "calendario" in out.lower()
    # Ambos eventos listados (distintas fechas, NO dedup)
    assert out.count("cumpleaños de Astor") >= 2, (
        f"esperaba 2 apariciones (fechas distintas), got:\n{out}"
    )
    # Fechas visibles
    assert "day after tomorrow" in out or "2026-04-24" in out or "24/04" in out
    assert "25 abr 2026" in out or "2026-04-25" in out or "25/04" in out


def test_calendar_ahead_empty_declares_explicit_empty_state():
    fmt = _import_helper()
    out = fmt("calendar_ahead", "[]")
    assert any(phrase in out.lower() for phrase in (
        "sin eventos", "no hay eventos", "ningún evento",
        "0 eventos", "sin calendar",
    )), f"esperaba mensaje explícito de empty, got:\n{out}"


def test_calendar_ahead_no_tool_name_leak():
    fmt = _import_helper()
    out = fmt("calendar_ahead", "[]")
    assert "calendar_ahead" not in out
    assert "[[calendar_ahead]]" not in out


# ── defensive: malformed input ────────────────────────────────────────


def test_malformed_json_falls_through_without_raising():
    """Si el tool devuelve algo raro (no JSON), el helper no debe
    crashear el request — devolver el raw como fallback."""
    fmt = _import_helper()
    out = fmt("reminders_due", "Error: timeout")
    # No raisea; retorna algo útil
    assert isinstance(out, str)
    assert len(out) > 0


def test_unknown_tool_falls_back_to_raw_string():
    fmt = _import_helper()
    raw = '{"foo": "bar"}'
    out = fmt("unknown_tool_xyz", raw)
    # Fallback: el raw JSON + un header genérico (tool name permitido solo
    # como header de la sección si el helper no lo reconoce)
    assert isinstance(out, str)
    # El contenido debe estar presente (aunque sea crudo)
    assert "foo" in out or "bar" in out


# ── counts explícitos en header (2026-04-23) ─────────────────────────


def test_reminders_due_header_includes_dated_and_undated_counts():
    """El header debe expresar N con fecha / M sin fecha para anclar al
    LLM y evitar que cuente mal ('tres tareas' cuando había una).
    """
    fmt = _import_helper()
    raw = json.dumps({
        "dated": [
            {"id": "r1", "name": "A", "due": "2026-04-24T10:00", "bucket": "upcoming"},
        ],
        "undated": [
            {"id": "r2", "name": "B", "due": "", "bucket": "undated"},
            {"id": "r3", "name": "C", "due": "", "bucket": "undated"},
        ],
    }, ensure_ascii=False)
    out = fmt("reminders_due", raw)
    # Expect "1 con fecha, 2 sin fecha" en el header.
    assert "1 con fecha" in out, f"header sin count `con fecha`:\n{out}"
    assert "2 sin fecha" in out, f"header sin count `sin fecha`:\n{out}"


def test_reminders_due_header_only_dated():
    fmt = _import_helper()
    raw = json.dumps({
        "dated": [{"name": "A", "due": "2026-04-24T10:00", "bucket": "upcoming"}],
        "undated": [],
    }, ensure_ascii=False)
    out = fmt("reminders_due", raw)
    assert "1 con fecha" in out
    # No debe mencionar "sin fecha" cuando no hay items de esa categoría.
    assert "sin fecha" not in out.lower().replace("_sin recordatorios pendientes._", "")


def test_calendar_ahead_header_includes_event_count():
    fmt = _import_helper()
    raw = json.dumps([
        {"title": "A", "date_label": "hoy", "time_range": ""},
        {"title": "B", "date_label": "mañana", "time_range": ""},
        {"title": "C", "date_label": "pasado", "time_range": ""},
    ], ensure_ascii=False)
    out = fmt("calendar_ahead", raw)
    assert "3 eventos" in out, f"esperaba '3 eventos' en header:\n{out}"


def test_calendar_ahead_header_singular_for_one_event():
    fmt = _import_helper()
    raw = json.dumps([
        {"title": "Solo uno", "date_label": "hoy", "time_range": ""},
    ], ensure_ascii=False)
    out = fmt("calendar_ahead", raw)
    # Singular "evento" no "eventos".
    assert "1 evento)" in out, f"esperaba '1 evento)' en header (singular):\n{out}"


def test_gmail_recent_header_includes_thread_count():
    fmt = _import_helper()
    raw = json.dumps({
        "unread_count": 12,
        "threads": [
            {"kind": "awaiting_reply", "from": "a@b.com", "subject": "s1"},
            {"kind": "starred", "from": "c@d.com", "subject": "s2"},
        ],
    }, ensure_ascii=False)
    out = fmt("gmail_recent", raw)
    assert "2 hilos" in out
    assert "12 no leídos" in out


# ── integration: compose multiple blocks ──────────────────────────────


def test_compose_multiple_tools_reminders_and_calendar():
    """Smoke test — combinar los 2 tools disparados por "qué tengo
    esta semana?" debe producir un bloque cohesivo con secciones
    separadas y sin mezcla."""
    fmt = _import_helper()
    rem_raw = json.dumps({
        "dated": [{"id": "r1", "name": "llamar dentista",
                   "due": "2026-04-24T10:00", "bucket": "upcoming"}],
        "undated": [{"id": "r2", "name": "Arreglar mosquiteros",
                     "due": "", "bucket": "undated"}],
    }, ensure_ascii=False)
    cal_raw = json.dumps([
        {"title": "cumpleaños Astor", "date_label": "24 abr 2026",
         "time_range": ""},
    ], ensure_ascii=False)

    block = fmt("reminders_due", rem_raw) + "\n" + fmt("calendar_ahead", cal_raw)
    assert "llamar dentista" in block
    assert "Arreglar mosquiteros" in block
    assert "cumpleaños Astor" in block
    # Secciones claramente separadas
    assert "ecordator" in block
    assert "calendar" in block.lower() or "calendario" in block.lower()
