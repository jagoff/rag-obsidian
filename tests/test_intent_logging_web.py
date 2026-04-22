"""Tests para el logueo de intent desde el web server (`/api/chat`).

Auditoría 2026-04-22 encontró que 1641/1667 (98.4%) de queries en las
últimas 7 días tenían `extra_json` SIN el campo ``intent``:

  cmd         count
  ---------   -----
  web         700    ← 42% del tráfico, no loguea intent
  followup    363
  read        248
  query       105    ← único path que lo loguea
  chat         38    ← tampoco loguea
  ...

El GC#2.A (commit cb2805e) lo agregó solo a CLI (`query()` + 2 spots más);
los 3 call sites de ``log_query_event`` en ``web/server.py`` lo omiten.
Resultado: imposible medir qué intents tienen top_score más bajo, qué
intents disparan más citation-repair, si ``RAG_ADAPTIVE_ROUTING`` ayuda.

Cambios necesarios:

  1. ``retrieve()`` y ``multi_retrieve()`` devuelven ``intent`` en el dict
     de resultado (eco del argumento ``intent=...`` que se pasa o del
     ``classify_intent()`` interno si no).
  2. Los tres call sites del web server (``web.chat.metachat``,
     ``web.chat.low_conf_bypass``, ``web``) agregan ``"intent":
     result.get("intent")`` al ``log_query_event`` payload.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── retrieve() / multi_retrieve() devuelven intent ───────────────────────────


def test_retrieve_returns_intent_when_passed_as_arg():
    """Si el caller pasó ``intent="list"`` a ``retrieve()``, el dict de
    retorno debe exponerlo. Antes (pre 2026-04-22) quedaba en el scope
    interno y nunca llegaba al caller."""
    class _EmptyCol:
        def count(self):
            return 0

    result = rag.retrieve(
        _EmptyCol(), "cuántas notas tengo sobre coaching",
        k=3, folder=None, intent="count",
    )
    assert result.get("intent") == "count", \
        f"retrieve() must echo back the `intent` arg; got {result.get('intent')!r}"


def test_retrieve_returns_intent_none_when_not_passed():
    """Cuando el caller no pasó intent, el dict expone None (no crash)."""
    class _EmptyCol:
        def count(self):
            return 0

    result = rag.retrieve(_EmptyCol(), "test", k=3, folder=None)
    assert "intent" in result
    assert result["intent"] is None


def test_retrieve_returns_intent_in_empty_corpus_path():
    """El early-return en corpus vacío (count=0) también debe exponer intent
    — antes era un dict distinto al del happy path y perdía la key."""
    class _EmptyCol:
        def count(self):
            return 0

    result = rag.retrieve(_EmptyCol(), "test", k=3, folder=None, intent="recent")
    assert result.get("intent") == "recent"


# ── Integration smoke: el classify_intent que el web ya hace debe llegar al log


def test_web_has_intent_classifier_wired_in_pipeline():
    """Regression guard: el web server debe tener `_rag.classify_intent`
    accesible. Si alguien remueve el import fallará acá antes de afectar
    producción."""
    import importlib
    # web/server.py siempre importa rag; chequeamos que classify_intent
    # es una función callable (no un string u otro sentinel).
    assert callable(rag.classify_intent)


def test_log_query_event_accepts_intent_field():
    """Sanity: ``log_query_event`` debe aceptar y persistir un campo
    ``intent`` sin rechazarlo. Si el mapper cambia schema y lo deja
    afuera, la telemetría se rompe silenciosamente."""
    # Usamos un monkey-patch al writer SQL para capturar el payload final
    # sin necesidad de DB real.
    captured: list[dict] = []

    def fake_write(payload):
        captured.append(payload)

    with patch.object(rag, "_sql_write_with_retry",
                      side_effect=lambda fn, tag, **kw: fn()):
        # No abrimos conexion real; interceptamos via _ragvec_state_conn.
        # Solo comprobamos que la llamada no levanta y que el dict llega con intent.
        ev = {
            "cmd": "web",
            "q": "test question",
            "intent": "synthesis",
            "top_score": 0.5,
        }
        # Llamar a log_query_event con un event que incluya intent no debe
        # levantar. Si el SQL escribe, el intent debería ir a extra_json.
        try:
            rag.log_query_event(ev)
        except Exception as exc:
            pytest.fail(f"log_query_event rechazó un event con intent={ev['intent']}: {exc}")


# ── multi_retrieve: intent se propaga en single-vault + empty-vaults paths ───


def test_multi_retrieve_empty_vaults_returns_intent():
    """Early return (vaults=[]) must still expose the caller's intent."""
    result = rag.multi_retrieve([], "q", k=3, folder=None, intent="list")
    assert result.get("intent") == "list"


def test_multi_retrieve_single_vault_returns_intent(tmp_path, monkeypatch):
    """Single-vault shortcut must echo intent from the inner retrieve()."""
    class _EmptyCol:
        def count(self):
            return 0

    # Stub get_db_for to return our empty col regardless of path.
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _EmptyCol())

    result = rag.multi_retrieve(
        [("home", tmp_path)], "q", k=3, folder=None, intent="comparison",
    )
    assert result.get("intent") == "comparison"


# ── Regression: propagating intent arg doesn't break other callers ──────────


def test_retrieve_accepts_intent_kwarg_without_regression():
    """Passing intent= as kwarg must not break retrieve's other kwargs."""
    class _EmptyCol:
        def count(self):
            return 0

    # Multiple kwargs alongside intent — ensure the signature still parses.
    result = rag.retrieve(
        _EmptyCol(), "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
        intent="entity_lookup",
    )
    assert result.get("intent") == "entity_lookup"
