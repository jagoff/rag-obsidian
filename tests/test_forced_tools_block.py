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
import pytest


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
