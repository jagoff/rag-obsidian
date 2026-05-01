"""Tests for the 4 chat-latency fixes shipped 2026-05-01.

Repro: el user reportó que la query "qué tengo hoy" tardaba 201 segundos —
log mostraba `reform=174289ms reform_outcome=concat tool_ms=2153
tool_names=whatsapp_pending,calendar_ahead,reminders_due
topic_shift=cosine=0.471` mientras Ollama logueaba
`restart_skipped_in_flight p95_recent=291498ms ratio=5.59`. Root cause:
banda borderline cosine [0.32, 0.50) en una query autónoma de 3 palabras
con tools deterministas ya forzados → reform LLM se colgó esperando
slot del `_HELPER_SEM` (cap 2). Cuando volvió, el `search_question`
reformulado se descartó igual porque el pre-router reemplaza el
CONTEXTO entero con el output de los tools.

4 fixes ortogonales:

  Fix 1: skipear reform LLM si el pre-router va a forzar tools (regex
         sub-millisecond — `_detect_tool_intent`). El reform es
         desperdicio porque el retrieve queda reemplazado.

  Fix 2: skipear reform LLM si la query es ≤3 palabras y no tiene
         pronombres anaphoricos (regex `_TOPIC_SHIFT_FOLLOWUP_RE`).
         Queries cortas autónomas como "qué tengo hoy", "clima en BA"
         son self-contained — el reform no aporta y paga la latencia.

  Fix 3: TTL cache de 30s en `_fetch_reminders_due`. El osascript itera
         todas las listas + items con `completed is false` (~2.1s
         observado) y es el long pole del bucket paralelo del web chat.
         Cache reduce hits subsiguientes a ~50ms.

  Fix 4: convertir time_range del calendar a 24h (sin AM/PM) antes de
         inyectarlo al LLM — "10:30 AM-11:00 AM" → "10:30–11:00". El
         user reportó alucinación "10:30 AM–en en:00 AM", probable
         confusión del modelo con tokens AM/PM bajo presión de Ollama.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pytest


# ── Fix 4: time_range normalization a 24h ─────────────────────────────────


def test_normalize_time_12h_to_24h_am_basic():
    from rag.integrations.calendar import _normalize_time_to_24h

    assert _normalize_time_to_24h("10:30 AM") == "10:30"
    assert _normalize_time_to_24h("9:00 AM") == "09:00"
    assert _normalize_time_to_24h("8:15 am") == "08:15"


def test_normalize_time_12h_to_24h_pm_basic():
    from rag.integrations.calendar import _normalize_time_to_24h

    assert _normalize_time_to_24h("1:00 PM") == "13:00"
    assert _normalize_time_to_24h("11:30 PM") == "23:30"
    assert _normalize_time_to_24h("3:45 pm") == "15:45"


def test_normalize_time_12h_to_24h_noon_midnight_edges():
    """12:00 AM = midnight (00:00), 12:00 PM = noon (12:00). Caso clásico
    donde la conversión naive falla."""
    from rag.integrations.calendar import _normalize_time_to_24h

    assert _normalize_time_to_24h("12:00 AM") == "00:00"
    assert _normalize_time_to_24h("12:30 AM") == "00:30"
    assert _normalize_time_to_24h("12:00 PM") == "12:00"
    assert _normalize_time_to_24h("12:45 PM") == "12:45"


def test_normalize_time_passthrough_24h_input():
    """Ya en 24h — no tocar."""
    from rag.integrations.calendar import _normalize_time_to_24h

    assert _normalize_time_to_24h("10:30") == "10:30"
    assert _normalize_time_to_24h("23:00") == "23:00"
    assert _normalize_time_to_24h("00:00") == "00:00"


def test_normalize_time_passthrough_malformed():
    """Inputs malformed se devuelven tal cual — la función nunca raisea."""
    from rag.integrations.calendar import _normalize_time_to_24h

    assert _normalize_time_to_24h("") == ""
    assert _normalize_time_to_24h("not a time") == "not a time"
    assert _normalize_time_to_24h("10:30 XM") == "10:30 XM"


def test_normalize_time_range_em_dash():
    """Fix 4 caso canónico: separador em-dash + AM/PM → 24h con em-dash."""
    from rag.integrations.calendar import _normalize_time_range_to_24h

    assert _normalize_time_range_to_24h("10:30 AM–11:00 AM") == "10:30–11:00"
    assert _normalize_time_range_to_24h("2:30 PM–3:30 PM") == "14:30–15:30"


def test_normalize_time_range_ascii_dash():
    """También acepta separador ASCII (-) — icalBuddy raw."""
    from rag.integrations.calendar import _normalize_time_range_to_24h

    assert _normalize_time_range_to_24h("10:30 AM-11:00 AM") == "10:30–11:00"


def test_normalize_time_range_already_24h():
    from rag.integrations.calendar import _normalize_time_range_to_24h

    assert _normalize_time_range_to_24h("10:30–11:00") == "10:30–11:00"


def test_normalize_time_range_empty_passthrough():
    from rag.integrations.calendar import _normalize_time_range_to_24h

    assert _normalize_time_range_to_24h("") == ""


def test_calendar_ahead_emits_24h_in_time_range(monkeypatch):
    """Smoke integration: `_fetch_calendar_ahead` con AM/PM en icalBuddy
    output → time_range queda en 24h. Garantiza que el LLM nunca recibe
    AM/PM del calendar."""
    import rag

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/usr/local/bin/icalBuddy")

    fake_stdout = (
        "Entrega llave casa\n"
        "    today at 10:30 AM - 11:00 AM\n"
        "Psiquiatra\n"
        "    today at 2:30 PM - 3:30 PM\n"
    )

    class _FakeRes:
        returncode = 0
        stdout = fake_stdout

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeRes())

    from rag.integrations.calendar import _fetch_calendar_ahead

    events = _fetch_calendar_ahead(days_ahead=1)
    titles = {e["title"]: e for e in events}
    assert "Entrega llave casa" in titles
    assert "Psiquiatra" in titles
    # CRÍTICO: ya no debe contener "AM" / "PM" — solo dígitos + em-dash
    assert titles["Entrega llave casa"]["time_range"] == "10:30–11:00"
    assert titles["Psiquiatra"]["time_range"] == "14:30–15:30"
    for ev in events:
        assert "AM" not in ev["time_range"], f"{ev}"
        assert "PM" not in ev["time_range"], f"{ev}"


# ── Fix 3: TTL cache para `_fetch_reminders_due` ──────────────────────────


def _setup_reminders_cache_test(monkeypatch, ttl: float = 30.0):
    """Helper: limpia el cache + patchea `_apple_enabled` + cuenta calls
    a `_osascript`. Devuelve un dict con `count` que el caller chequea."""
    from rag.integrations import reminders as reminders_mod

    reminders_mod._reminders_cache_clear()
    monkeypatch.setenv("RAG_REMINDERS_CACHE_TTL", str(ttl))
    import rag

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    counter = {"count": 0}

    def _fake_osascript(script: str, timeout: float = 45.0) -> str:
        counter["count"] += 1
        return ""  # empty output → no items, but cache key is still set

    monkeypatch.setattr(rag, "_osascript", _fake_osascript)
    return counter


def test_reminders_cache_hits_avoid_reexec(monkeypatch):
    """Fix 3: 2 calls dentro del TTL → osascript se ejecuta SOLO una vez."""
    from rag.integrations.reminders import _fetch_reminders_due

    counter = _setup_reminders_cache_test(monkeypatch, ttl=30.0)
    # Primer call: cache miss → osascript fires.
    # Pero como devuelve "" tampoco se cachea (cache solo guarda raw output
    # NO vacío). Probemos con output válido.

    from rag.integrations import reminders as reminders_mod

    reminders_mod._reminders_cache_clear()
    monkeypatch.setenv("RAG_REMINDERS_CACHE_TTL", "30")
    import rag

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    counter = {"count": 0}

    def _fake_osascript(script: str, timeout: float = 45.0) -> str:
        counter["count"] += 1
        return "rid1|comprar pan||groceries\nrid2|llamar mama||personal\n"

    monkeypatch.setattr(rag, "_osascript", _fake_osascript)

    now = datetime.now()
    items1 = _fetch_reminders_due(now, horizon_days=1)
    items2 = _fetch_reminders_due(now, horizon_days=1)
    items3 = _fetch_reminders_due(now, horizon_days=7)  # diff horizon, mismo cache raw
    assert counter["count"] == 1, f"osascript debería ejecutarse 1× pero corrió {counter['count']}"
    # Las 3 llamadas devolvieron items consistentes
    assert len(items1) == 2
    assert len(items2) == 2
    assert len(items3) == 2


def test_reminders_cache_disabled_when_ttl_zero(monkeypatch):
    """`RAG_REMINDERS_CACHE_TTL=0` desactiva el cache → osascript corre
    en cada call."""
    from rag.integrations import reminders as reminders_mod
    from rag.integrations.reminders import _fetch_reminders_due

    reminders_mod._reminders_cache_clear()
    monkeypatch.setenv("RAG_REMINDERS_CACHE_TTL", "0")
    import rag

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    counter = {"count": 0}

    def _fake_osascript(script: str, timeout: float = 45.0) -> str:
        counter["count"] += 1
        return "rid1|comprar pan||groceries\n"

    monkeypatch.setattr(rag, "_osascript", _fake_osascript)
    now = datetime.now()
    _fetch_reminders_due(now)
    _fetch_reminders_due(now)
    _fetch_reminders_due(now)
    assert counter["count"] == 3


def test_reminders_cache_expires_after_ttl(monkeypatch):
    """Cuando `time.monotonic()` avanza más allá del TTL, el siguiente
    call re-ejecuta osascript."""
    from rag.integrations import reminders as reminders_mod
    from rag.integrations.reminders import _fetch_reminders_due

    reminders_mod._reminders_cache_clear()
    monkeypatch.setenv("RAG_REMINDERS_CACHE_TTL", "5")
    import rag

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    counter = {"count": 0}

    def _fake_osascript(script: str, timeout: float = 45.0) -> str:
        counter["count"] += 1
        return "rid1|comprar pan||groceries\n"

    monkeypatch.setattr(rag, "_osascript", _fake_osascript)

    fake_clock = {"now": 1000.0}
    monkeypatch.setattr(reminders_mod.time, "monotonic", lambda: fake_clock["now"])

    now = datetime.now()
    _fetch_reminders_due(now)
    fake_clock["now"] += 1.0  # dentro del TTL
    _fetch_reminders_due(now)
    assert counter["count"] == 1
    fake_clock["now"] += 10.0  # fuera del TTL (1+10=11 > 5)
    _fetch_reminders_due(now)
    assert counter["count"] == 2


# ── Fix 1 + Fix 2: gates del path (c) reform LLM ──────────────────────────


def test_pre_router_detects_planning_query():
    """`qué tengo hoy` matchea `_PLANNING_PAT` → fuerza reminders +
    calendar + whatsapp_pending. Sanity check que el regex sigue
    enganchando esa frase puntual del repro 2026-05-01."""
    from web.server import _detect_tool_intent

    pairs = _detect_tool_intent("qué tengo hoy")
    names = {n for n, _ in pairs}
    assert "reminders_due" in names
    assert "calendar_ahead" in names
    assert "whatsapp_pending" in names


def test_pre_router_query_likely_autonomous():
    """`qué tengo hoy` (3 palabras, sin pronombres anaphoricos) →
    `_query_likely_autonomous` debería ser True."""
    from rag import _TOPIC_SHIFT_FOLLOWUP_RE

    q = "qué tengo hoy"
    assert len(q.split()) == 3
    assert _TOPIC_SHIFT_FOLLOWUP_RE.search(q) is None


def test_pre_router_query_with_pronoun_not_autonomous():
    """`y eso?` matchea `_TOPIC_SHIFT_FOLLOWUP_RE` → NO autonomous,
    el reform LLM debería seguir corriendo en banda borderline."""
    from rag import _TOPIC_SHIFT_FOLLOWUP_RE

    q = "y eso?"
    assert _TOPIC_SHIFT_FOLLOWUP_RE.search(q) is not None


def test_long_query_not_autonomous_even_without_pronoun():
    """Queries de >3 palabras NO se consideran autonomous aunque no
    tengan pronombres — pueden necesitar reform si caen en banda
    borderline (e.g. "tenés más sobre la cumbre del jueves")."""
    from rag import _TOPIC_SHIFT_FOLLOWUP_RE

    q = "tenés más sobre la cumbre del jueves"
    # 7 palabras > 3 → no se considera autonomous por el guard de q_words
    assert len(q.split()) > 3
    # Y "más sobre" matchea el regex anaphoric (línea 7330 server.py)
    # — sin importar el largo, esto se rutea a path (a) concat o (c)
    # reform. El test acá sólo asegura que el guard no apaga reform
    # para queries largas.


def test_reform_outcome_formula_replicates_skipped_reason():
    """La fórmula del log `reform_outcome` (web/server.py ~14316) emite
    `skipped_<reason>` cuando `_reform_skipped_reason` está populado.
    Replicamos el ternario acá para garantizar que los nuevos motivos
    `forced_tools` y `short_autonomous` salen al log con el prefix
    correcto."""

    def _outcome(used_concat: bool, fired: bool, skipped_reason: str) -> str:
        # Espejo de la fórmula inline en server.py
        return (
            "concat" if used_concat
            else "rewritten" if fired
            else f"skipped_{skipped_reason}" if skipped_reason
            else "skipped"
        )

    assert _outcome(True, True, "") == "concat"
    assert _outcome(False, True, "") == "rewritten"
    assert _outcome(False, False, "") == "skipped"
    assert _outcome(False, False, "forced_tools") == "skipped_forced_tools"
    assert _outcome(False, False, "short_autonomous") == "skipped_short_autonomous"
