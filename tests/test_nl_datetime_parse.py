"""Tests para `_parse_natural_datetime` y `_parse_natural_recurrence`.

Ambas helpers alimentan los tools `propose_reminder` / `propose_calendar_event`:
el LLM ya entrega `title` + `when` + opcional `recurrence_text`; nuestra tarea
es convertir los strings a `datetime` / RRULE dict deterministas.

Estrategia:
- dateparser (es+en, prefer_dates_from=future) resuelve el 80% de los casos;
  esos tests corren sin LLM.
- Para fechas que dateparser no agarra (ej. "el jueves que viene a las 3"),
  cae a helper LLM (mockeado).
- Recurrencia: regex hand-rolled sobre patrones comunes ES/EN; no LLM.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


# ── _parse_natural_datetime: dateparser path ─────────────────────────────────


def test_parse_manana_10am():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("mañana a las 10am", now=now)
    assert dt is not None
    assert dt.date().isoformat() == "2026-04-21"
    assert dt.hour == 10
    assert dt.minute == 0


def test_parse_en_2_horas():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("en 2 horas", now=now)
    assert dt is not None
    # Tolerancia de ±1 minuto — dateparser computa desde "now" real.
    delta = abs((dt - datetime(2026, 4, 20, 17, 0, 0)).total_seconds())
    assert delta < 120


def test_parse_iso_string():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("2026-04-25 14:30", now=now)
    assert dt is not None
    assert dt.isoformat(timespec="minutes") == "2026-04-25T14:30"


def test_parse_english_monday_3pm():
    """'next monday 3pm' confunde a dateparser; 'monday 3pm' con prefer_future
    rinde el próximo lunes. El caso 'next X' cae al LLM fallback (ver más abajo).
    """
    now = datetime(2026, 4, 20, 15, 0, 0)  # Monday
    dt = rag._parse_natural_datetime("monday 3pm", now=now)
    assert dt is not None
    # Next monday = 27, 15:00
    assert dt.weekday() == 0
    assert dt.hour == 15


def test_parse_pasado_manana_via_preprocessor(monkeypatch):
    """Post-preprocessor 'pasado mañana' → 'day after tomorrow', que
    dateparser sí maneja — no hace falta el LLM fallback."""
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("no debería llamarse")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("pasado mañana 9:30", now=now)
    assert dt is not None
    assert dt.date().isoformat() == "2026-04-22"
    assert dt.hour == 9
    assert dt.minute == 30


def test_parse_prefers_future_for_bare_weekday():
    """'el jueves' sin referencia semanal → PRÓXIMO jueves, no el pasado."""
    # Lunes 20-abr-2026 → jueves futuro es 23-abr.
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("jueves 14:00", now=now)
    assert dt is not None
    assert dt >= now


def test_parse_empty_returns_none():
    assert rag._parse_natural_datetime("", now=datetime(2026, 4, 20)) is None
    assert rag._parse_natural_datetime("   ", now=datetime(2026, 4, 20)) is None


def test_parse_garbage_returns_none(monkeypatch):
    """Texto no-parseable + LLM fallback que también devuelve null → None."""
    # Forzar que el fallback LLM también devuelva null.
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": null}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime("asdf xyz qwerty", now=datetime(2026, 4, 20))
    assert dt is None


# ── _parse_natural_datetime: LLM fallback ───────────────────────────────────


def test_parse_llm_fallback_invoked_when_dateparser_fails(monkeypatch):
    now = datetime(2026, 4, 20, 15, 0, 0)
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": "2026-04-30T11:00"}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    # Input que dateparser no resuelve bien (ni con es ni en).
    dt = rag._parse_natural_datetime("la próxima reunión del comité", now=now)
    assert dt == datetime(2026, 4, 30, 11, 0, 0)
    fake_client.chat.assert_called_once()


def test_parse_llm_fallback_malformed_json_returns_none(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content="not json at all")
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime(
        "algo super raro que no es fecha", now=datetime(2026, 4, 20)
    )
    assert dt is None


def test_parse_llm_fallback_error_returns_none(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("ollama caído")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime(
        "algo raro que no parsea", now=datetime(2026, 4, 20)
    )
    assert dt is None


def test_parse_llm_fallback_skipped_when_dateparser_succeeds(monkeypatch):
    """Performance guard: dateparser primero, LLM no se toca si ya resolvió."""
    fake_client = MagicMock()
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime("mañana a las 10", now=datetime(2026, 4, 20))
    assert dt is not None
    fake_client.chat.assert_not_called()


# ── _parse_natural_recurrence ───────────────────────────────────────────────


def test_recurrence_todos_los_dias():
    r = rag._parse_natural_recurrence("todos los días")
    assert r == {"freq": "DAILY", "interval": 1}


def test_recurrence_diariamente():
    assert rag._parse_natural_recurrence("diariamente") == {"freq": "DAILY", "interval": 1}


def test_recurrence_todos_los_lunes():
    r = rag._parse_natural_recurrence("todos los lunes")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]}


def test_recurrence_cada_martes():
    r = rag._parse_natural_recurrence("cada martes")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["TU"]}


def test_recurrence_semanalmente():
    assert rag._parse_natural_recurrence("semanalmente") == {"freq": "WEEKLY", "interval": 1}


def test_recurrence_cada_2_semanas():
    r = rag._parse_natural_recurrence("cada 2 semanas")
    assert r == {"freq": "WEEKLY", "interval": 2}


def test_recurrence_mensualmente():
    assert rag._parse_natural_recurrence("mensualmente") == {"freq": "MONTHLY", "interval": 1}


def test_recurrence_todos_los_meses():
    assert rag._parse_natural_recurrence("todos los meses") == {"freq": "MONTHLY", "interval": 1}


def test_recurrence_anualmente():
    assert rag._parse_natural_recurrence("anualmente") == {"freq": "YEARLY", "interval": 1}


def test_recurrence_every_day_english():
    assert rag._parse_natural_recurrence("every day") == {"freq": "DAILY", "interval": 1}


def test_recurrence_weekly_english():
    assert rag._parse_natural_recurrence("weekly") == {"freq": "WEEKLY", "interval": 1}


def test_recurrence_every_monday_english():
    r = rag._parse_natural_recurrence("every monday")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]}


def test_recurrence_none_for_non_recurring():
    assert rag._parse_natural_recurrence("mañana a las 10") is None
    assert rag._parse_natural_recurrence("") is None
    assert rag._parse_natural_recurrence(None) is None


# ── Rioplatense preprocessing (Argentinian idioms) ──────────────────────────


def test_preprocess_hs_suffix():
    """'18hs' / '18 hs' / '18h' → '18:00'. Si hay 'a las' antes, las dos
    reglas combinadas ('a las N' + 'Nhs') colapsan a sólo el tiempo."""
    assert rag._preprocess_rioplatense_datetime("hoy a las 18hs") == "hoy 18:00"
    assert rag._preprocess_rioplatense_datetime("hoy 18hs") == "hoy 18:00"
    assert rag._preprocess_rioplatense_datetime("a las 9 hs") == "9:00"
    assert rag._preprocess_rioplatense_datetime("18h") == "18:00"


def test_preprocess_mediodia():
    assert rag._preprocess_rioplatense_datetime("al mediodía") == "12:00"
    assert rag._preprocess_rioplatense_datetime("al medio día") == "12:00"
    assert rag._preprocess_rioplatense_datetime("mañana al mediodía") == "mañana 12:00"


def test_preprocess_que_viene():
    """'que viene' se normaliza a inglés. Para weekdays el bare form
    (no 'next X') porque dateparser 1.4 rechaza 'next <weekday>'; el bare
    'monday' con PREFER_DATES_FROM=future rolls-forward determinista."""
    assert rag._preprocess_rioplatense_datetime(
        "la semana que viene",
    ) == "next week"
    assert rag._preprocess_rioplatense_datetime(
        "el lunes que viene",
    ).lower() == "monday"
    assert rag._preprocess_rioplatense_datetime(
        "el mes que viene",
    ).lower() == "next month"


def test_preprocess_bare_weekday_rewrite():
    """'el jueves' / 'este jueves' / 'próximo jueves' → 'thursday' (bare)."""
    assert rag._preprocess_rioplatense_datetime("el jueves").lower() == "thursday"
    assert rag._preprocess_rioplatense_datetime("este viernes").lower() == "friday"
    assert rag._preprocess_rioplatense_datetime("próximo miércoles").lower() == "wednesday"


def test_preprocess_a_la_tarde_and_friends():
    """'a la mañana/tarde/noche/tardecita/nochecita' → hora default."""
    assert rag._preprocess_rioplatense_datetime("a la mañana") == "09:00"
    assert rag._preprocess_rioplatense_datetime("a la tarde") == "16:00"
    assert rag._preprocess_rioplatense_datetime("a la noche") == "20:00"
    assert rag._preprocess_rioplatense_datetime("a la tardecita") == "17:00"


def test_preprocess_tipo_N():
    """'tipo 10' → '10:00'. 'a eso de las 4' → '4:00'. Idioms aproximativos
    muy usados en rioplatense."""
    assert rag._preprocess_rioplatense_datetime("tipo 10") == "10:00"
    assert rag._preprocess_rioplatense_datetime("tipo 18:30") == "18:30"
    assert rag._preprocess_rioplatense_datetime("a eso de las 4") == "4:00"
    # Combinado con weekday normalización.
    out = rag._preprocess_rioplatense_datetime("el sábado tipo 10")
    assert "10:00" in out.lower()
    assert "saturday" in out.lower()


def test_preprocess_a_las_bare():
    """'a las N' → 'N:00' (dateparser interpreta 'a las 10' como día 10)."""
    assert rag._preprocess_rioplatense_datetime("a las 10") == "10:00"
    assert rag._preprocess_rioplatense_datetime("a las 18:30") == "18:30"
    assert "10:00" in rag._preprocess_rioplatense_datetime("mañana a las 10")


def test_preprocess_de_la_manana():
    """'a las N de la mañana' → 'N:00 am'."""
    assert rag._preprocess_rioplatense_datetime(
        "a las 10 de la mañana",
    ) == "10:00 am"
    assert rag._preprocess_rioplatense_datetime(
        "a las 10:30 de la mañana",
    ) == "10:30 am"


def test_preprocess_de_la_tarde():
    """'a las N de la tarde' → '(N+12):00' si N < 12."""
    assert rag._preprocess_rioplatense_datetime(
        "a las 4 de la tarde",
    ) == "16:00"
    # 12 de la tarde = 12:00 (mediodía), no 24:00.
    assert rag._preprocess_rioplatense_datetime(
        "a las 12 de la tarde",
    ) == "12:00"


def test_preprocess_diminutivos():
    assert rag._preprocess_rioplatense_datetime("en 2 horitas") == "en 2 horas"
    assert "minutos" in rag._preprocess_rioplatense_datetime("en 10 minutitos")


def test_preprocess_finde():
    """'el finde' → 'saturday' (bare). PREFER_DATES_FROM=future rollea al
    próximo sábado. 'next saturday' NO funciona en dateparser 1.4."""
    assert rag._preprocess_rioplatense_datetime("el finde") == "saturday"


def test_preprocess_pasado_manana():
    """'pasado mañana' se normaliza a 'day after tomorrow' (inglés)."""
    assert rag._preprocess_rioplatense_datetime("pasado mañana") == "day after tomorrow"
    assert rag._preprocess_rioplatense_datetime(
        "pasado mañana a las 10",
    ) == "day after tomorrow 10:00"


def test_preprocess_passthrough_unchanged():
    """Input sin idioms no debe cambiar."""
    s = "2026-04-25 14:30"
    assert rag._preprocess_rioplatense_datetime(s) == s


def test_parse_rioplatense_hs(monkeypatch):
    """Integration: '18hs' sin LLM debe resolver vía normalización."""
    # Forzar que el LLM fallback falle (no debe llegar a invocarse).
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("no debería llamarse")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("hoy a las 18hs", now=now)
    assert dt is not None
    assert dt.date().isoformat() == "2026-04-20"
    assert dt.hour == 18
    assert dt.minute == 0


def test_parse_rioplatense_mediodia(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("no debería llamarse")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("mañana al mediodía", now=now)
    assert dt is not None
    assert dt.date().isoformat() == "2026-04-21"
    assert dt.hour == 12


def test_parse_rioplatense_semana_que_viene(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("no debería llamarse")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    now = datetime(2026, 4, 20, 15, 0, 0)  # lunes
    dt = rag._parse_natural_datetime("la semana que viene", now=now)
    assert dt is not None
    # "próxima semana" → la semana siguiente (cualquier día de esa semana ok)
    assert dt >= now


def test_parse_anchor_time_echo_triggers_fallback(monkeypatch):
    """Dateparser a veces devuelve la hora del anchor cuando matcheó sólo
    la parte de fecha (bug conocido en 1.4 con ciertos time markers raros).
    El guard debe detectar eco exacto minuto-a-minuto y caer al LLM fallback.

    El normalizador rioplatense resuelve la mayoría de casos reales, así
    que forzamos el escenario monkeypatcheando dateparser directo.
    """
    import dateparser as _dp
    anchor = datetime(2026, 4, 20, 15, 0, 0)
    # Simular que dateparser devolvió el anchor exacto (echo).
    monkeypatch.setattr(_dp, "parse", lambda *a, **kw: anchor)
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": "2026-04-20T10:00"}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    # Input con time marker "las" → guard debe detectar eco y tirar a LLM.
    dt = rag._parse_natural_datetime("las fechas anchor-weird", now=anchor)
    assert dt is not None
    assert dt.hour == 10
    fake_client.chat.assert_called_once()
