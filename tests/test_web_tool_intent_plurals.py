"""Regression tests for the pre-router regex (`_TOOL_INTENT_RULES`) and the
source-specific intent hint (`_build_source_intent_hint`).

Origin: user report 2026-04-24 (Fer F.) — "cuales son mis ultimos mails?"
en el web chat devolvía un resumen de un hilo de WhatsApp (Joana Gonzales
sobre plantas) en lugar de información sobre mails o al menos un
reconocimiento explícito tipo "busqué en tus mails y no encontré nada".

Root cause identificado:

  1. **Regex bug en plurales españoles**: el pre-router usaba `\\bmail\\b`
     (word boundary a ambos lados) lo que NO matchea "mails" porque la
     `s` sigue siendo word-char → no hay transición word→non-word en la
     segunda `\\b`. Mismo problema con "correos", "eventos", "citas",
     "agendas", "días", "semanas". Confirmado en `[chat-timing]` del turn
     fallido: `tool_rounds=0 tool_names=` — `gmail_recent` nunca fired.

  2. **Sin hint source-specific al LLM**: incluso cuando el pre-router
     engancha `gmail_recent` y la sección "### Mails" entra en CONTEXTO,
     si esa sección está vacía (Gmail desync, nada reciente) el LLM
     obedece REGLA 1 ("engancháte SIEMPRE con el CONTEXTO") y responde
     sobre otra sección (WhatsApp, notas) como si fuera la respuesta
     principal, sin reconocer que el user pidió mails específicamente.

Este archivo cubre ambas partes del fix — regex plurals + hint builder —
como tests puros (no tocan TestClient ni monkeypatch pesado). La
integración end-to-end (que el hint efectivamente se inyecta en
`tool_messages` durante un turn real) vive en `test_web_chat_tools.py`
junto a los demás tests SSE.
"""
from __future__ import annotations

import pytest

from web.server import (
    _SOURCE_INTENT_LABEL,
    _build_source_intent_hint,
    _detect_tool_intent,
    _is_empty_tool_output,
)


# ── 1. Regex plurals — el bug reportado ─────────────────────────────────────


# Cada entrada: (query, expected tool name que DEBE matchear).
# Las queries son mix de singular/plural en español rioplatense — la
# versión pre-fix solo enganchaba los singulares.
_PLURAL_CASES: list[tuple[str, str]] = [
    # gmail_recent — el caso original del user report.
    ("cuales son mis ultimos mails?", "gmail_recent"),
    ("dame mis mails", "gmail_recent"),
    ("ultimos mails", "gmail_recent"),
    ("tengo mails nuevos?", "gmail_recent"),
    ("revisar correos", "gmail_recent"),
    ("mis correos de hoy", "gmail_recent"),
    ("emails sin leer", "gmail_recent"),
    # Singulares que ya andaban — no regresión.
    ("revisar mail", "gmail_recent"),
    ("tengo mail?", "gmail_recent"),
    ("abrí gmail", "gmail_recent"),
    ("bandeja de entrada", "gmail_recent"),

    # calendar_ahead — plurales de "evento", "cita", "agenda".
    ("que eventos tengo", "calendar_ahead"),
    ("mis proximos eventos", "calendar_ahead"),
    ("hay citas agendadas?", "calendar_ahead"),
    ("proximas citas", "calendar_ahead"),
    ("mis agendas de la semana", "calendar_ahead"),
    # Singulares — no regresión.
    ("tengo un evento hoy?", "calendar_ahead"),
    ("agendá una cita", "calendar_ahead"),
    ("mi agenda", "calendar_ahead"),
    ("reunión con Juan", "calendar_ahead"),

    # reminders_due / calendar_ahead — _PLANNING_PAT con "semanas", "días".
    # Nota: queries de planning disparan AMBOS tools (por diseño), alcanza
    # con que matcheen al menos el primero.
    ("que tengo esta semana", "reminders_due"),
    ("que tengo estas semanas", "reminders_due"),
    ("que tengo en los proximos dias", "reminders_due"),
    ("que tengo en los próximos días", "reminders_due"),
    # Singulares — no regresión.
    ("que tengo esta semana", "reminders_due"),
    ("como viene el dia", "reminders_due"),

    # weather — no hay "climas"/"tiempos" comunes pero validamos que `s?`
    # no rompe los singulares.
    ("cómo está el clima", "weather"),
    ("va a llover mañana?", "weather"),
    ("temperatura para hoy", "weather"),
]


@pytest.mark.parametrize("query,expected_tool", _PLURAL_CASES)
def test_pre_router_matches_plurals(query: str, expected_tool: str):
    """Cada query debe matchear AL MENOS el tool esperado. Si el pre-router
    engancha tools adicionales (p.ej. planning queries disparan
    reminders_due + calendar_ahead) está bien, mientras el esperado esté
    entre los matcheados."""
    matched = [name for name, _args in _detect_tool_intent(query)]
    assert expected_tool in matched, (
        f"Query {query!r} debería haber enganchado {expected_tool}, "
        f"pero el pre-router devolvió {matched}"
    )


def test_pre_router_false_positives_guarded():
    """El sufijo `s?` no debe convertir palabras singulares derivadas en
    matches. El `\\b` de apertura preserva el word-boundary al inicio —
    palabras que CONTIENEN el token como substring pero en otro contexto
    no deben enganchar.
    """
    # "mailbox" tiene "mail" al inicio pero sin boundary al final del match.
    # La regex es `\b(mail|...)s?\b` — para "mailbox" matchearía "mail"
    # hasta "l", pero después viene "b" (word-char) → falla el `\b` final.
    # Agregar `s?` no cambia eso: "mail" + "s?" sigue requiriendo boundary.
    false_positive_queries = [
        "mailbox lleno",       # mail + "box" → no match
        "eventualmente",       # evento + "...almente" → no match
        "citadino",            # cita + "dino" → no match
        "semanal",             # semana + "l" → no match
    ]
    for q in false_positive_queries:
        matched = _detect_tool_intent(q)
        # Ninguno de estos queries menciona planning tokens aislados, así
        # que el expected es "no pre-router match" (lista vacía). Si alguna
        # regex más amplia (finance_summary con "plata" etc.) engancha por
        # razones legítimas, documentar acá.
        assert matched == [], (
            f"Query {q!r} NO debería haber enganchado ningún tool, "
            f"pero matcheó {matched}"
        )


# ── 2. `_build_source_intent_hint` — output shape ───────────────────────────


def test_hint_none_when_no_tools():
    assert _build_source_intent_hint([]) is None


def test_hint_none_for_non_source_specific_tools():
    """weather y finance_summary no son source-specific: el user no pregunta
    'busqué en el clima y no encontré nada', así que el hint no aplica.
    """
    assert _build_source_intent_hint(["weather"]) is None
    assert _build_source_intent_hint(["finance_summary"]) is None
    assert _build_source_intent_hint(["weather", "finance_summary"]) is None


def test_hint_for_single_source_gmail():
    hint = _build_source_intent_hint(["gmail_recent"])
    assert hint is not None
    # Etiqueta humana + sección CONTEXTO correctas.
    assert "tus mails/correos" in hint
    assert "### Mails" in hint
    # Instrucción de reconocer vacío ANTES de fallback.
    assert "Busqué en tus mails/correos" in hint
    # Prohibición explícita de derivar a otras fuentes.
    assert "PROHIBIDO" in hint
    assert "WhatsApp" in hint


def test_hint_for_single_source_calendar():
    hint = _build_source_intent_hint(["calendar_ahead"])
    assert hint is not None
    assert "tu calendario/agenda/eventos" in hint
    assert "### Calendario" in hint


def test_hint_for_single_source_reminders():
    hint = _build_source_intent_hint(["reminders_due"])
    assert hint is not None
    assert "tus recordatorios/pendientes" in hint
    assert "### Recordatorios" in hint


def test_hint_mixes_two_sources_with_y():
    """Cuando el user dispara 2 tools source-specific (ej "que tengo esta
    semana" → reminders_due + calendar_ahead) el hint los une con "y".
    """
    hint = _build_source_intent_hint(["reminders_due", "calendar_ahead"])
    assert hint is not None
    assert "tus recordatorios/pendientes y tu calendario/agenda/eventos" in hint
    assert "### Recordatorios y ### Calendario" in hint


def test_hint_mixes_three_sources_with_comma_and_y():
    hint = _build_source_intent_hint(
        ["gmail_recent", "calendar_ahead", "reminders_due"]
    )
    assert hint is not None
    # El join es "A, B y C".
    assert ", " in hint
    assert " y " in hint
    # Las 3 secciones aparecen.
    assert "### Mails" in hint
    assert "### Calendario" in hint
    assert "### Recordatorios" in hint


def test_hint_ignores_non_source_tools_when_mixed():
    """Si el user pide algo con mix de source-specific + no-source-specific
    (ej "que tengo esta semana y cómo está el clima"), el hint solo menciona
    las source-specific — weather no tiene sentido como fallback.
    """
    hint = _build_source_intent_hint(
        ["reminders_due", "calendar_ahead", "weather"]
    )
    assert hint is not None
    # reminders + calendar están; weather NO debe aparecer en el hint
    # (solo tiene entrada en _SOURCE_INTENT_LABEL los 3 source-specific).
    assert "tus recordatorios/pendientes" in hint
    assert "tu calendario/agenda/eventos" in hint
    assert "clima" not in hint.lower()


# ── 3. Config sanity ────────────────────────────────────────────────────────


def test_source_intent_label_covers_exactly_source_specific_tools():
    """`_SOURCE_INTENT_LABEL` debe tener entradas para los 3 tools que
    buscan data real del usuario (gmail, calendar, reminders) y NO tener
    entradas para weather / finance_summary (no son "fuentes" buscables).
    """
    assert set(_SOURCE_INTENT_LABEL.keys()) == {
        "gmail_recent",
        "calendar_ahead",
        "reminders_due",
    }
    # Cada entrada tiene la forma (etiqueta_humana, section_header).
    for name, (label, section) in _SOURCE_INTENT_LABEL.items():
        assert isinstance(label, str) and label, f"{name}: label vacío"
        assert section.startswith("### "), (
            f"{name}: section header debe ser '### X' (matchea el output de "
            f"_format_forced_tool_output), got {section!r}"
        )


# ── 4. `_is_empty_tool_output` — shape-aware empty detection ───────────────


class TestIsEmptyToolOutput:
    """Cubre los 5 tools del pre-router + edge cases de JSON malformado."""

    # gmail_recent
    def test_gmail_empty_when_no_threads_no_unread(self):
        import json
        raw = json.dumps({"unread_count": 0, "threads": []})
        assert _is_empty_tool_output("gmail_recent", raw) is True

    def test_gmail_not_empty_when_has_threads(self):
        import json
        raw = json.dumps({
            "unread_count": 0,
            "threads": [{"kind": "starred", "subject": "x", "from": "a@b.com"}],
        })
        assert _is_empty_tool_output("gmail_recent", raw) is False

    def test_gmail_not_empty_when_has_unread(self):
        import json
        raw = json.dumps({"unread_count": 5, "threads": []})
        assert _is_empty_tool_output("gmail_recent", raw) is False

    def test_gmail_malformed_unread_falls_back_to_zero(self):
        """unread_count no-numérico se trata como 0 — es defensivo, no
        debería pasar en prod pero si pasa, no queremos crashear."""
        import json
        raw = json.dumps({"unread_count": "whatever", "threads": []})
        assert _is_empty_tool_output("gmail_recent", raw) is True

    # calendar_ahead
    def test_calendar_empty_when_list_empty(self):
        assert _is_empty_tool_output("calendar_ahead", "[]") is True

    def test_calendar_not_empty_when_has_events(self):
        import json
        raw = json.dumps([{"title": "reunión", "date_label": "mañana"}])
        assert _is_empty_tool_output("calendar_ahead", raw) is False

    # reminders_due
    def test_reminders_empty_when_both_empty(self):
        import json
        raw = json.dumps({"dated": [], "undated": []})
        assert _is_empty_tool_output("reminders_due", raw) is True

    def test_reminders_not_empty_when_dated(self):
        import json
        raw = json.dumps({
            "dated": [{"name": "pagar luz", "due": "2026-05-01"}],
            "undated": [],
        })
        assert _is_empty_tool_output("reminders_due", raw) is False

    def test_reminders_not_empty_when_undated(self):
        import json
        raw = json.dumps({
            "dated": [],
            "undated": [{"name": "comprar café"}],
        })
        assert _is_empty_tool_output("reminders_due", raw) is False

    # finance_summary / weather → no empty-state semantics.
    def test_finance_always_not_empty(self):
        """finance_summary con todos los campos en 0 sigue siendo data
        válida ("tu mes fue cero gastos" es una respuesta útil), no
        empty-state."""
        assert _is_empty_tool_output("finance_summary", "{}") is False
        import json
        raw = json.dumps({"total_month": 0, "top_categories": []})
        assert _is_empty_tool_output("finance_summary", raw) is False

    def test_weather_always_not_empty(self):
        """weather es passthrough string — siempre tiene output utilizable."""
        assert _is_empty_tool_output("weather", '"cielo despejado, 20°C"') is False
        assert _is_empty_tool_output("weather", '""') is False

    # Edge cases.
    def test_malformed_json_returns_false(self):
        """JSON roto → conservador, return False (no asumir empty)."""
        assert _is_empty_tool_output("gmail_recent", "not json") is False
        assert _is_empty_tool_output("gmail_recent", "") is False

    def test_unknown_tool_returns_false(self):
        """Tool no mapeado: conservador, return False."""
        assert _is_empty_tool_output("unknown_tool", "{}") is False

    def test_gmail_non_dict_returns_false(self):
        """JSON parsea pero no es dict (p.ej. lista)."""
        assert _is_empty_tool_output("gmail_recent", "[]") is False

    def test_reminders_non_dict_returns_false(self):
        """Igual que gmail, reminders espera dict."""
        assert _is_empty_tool_output("reminders_due", "null") is False
