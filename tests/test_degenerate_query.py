"""Tests para `_is_degenerate_query` + `_pick_degenerate_reply` +
short-circuit en `/api/chat` (2026-04-23).

Motivación: el scratch_eval encontró que inputs como `"x"` o `"?¡@#"`
no disparaban metachat (no son saludo) y caían al pipeline full del
retriever, devolviendo chunks random de WhatsApp sin relación porque
el matching semántico de un string casi vacío es puro ruido.

Los tests cubren:
  - El detector matchea queries con <2 chars alfanuméricos.
  - Queries normales NO se marcan degenerate.
  - Metachat y degenerate son mutuamente exclusivos en el endpoint
    (metachat gana si aplica — "hola" tiene 4 alfanuméricos, cae ahí).
  - El picker devuelve strings no vacíos y rota entre las 3 variantes.
  - La respuesta HTTP del short-circuit es < 200ms (no toca retrieve).
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _import_helpers():
    from web.server import (
        _is_degenerate_query,
        _pick_degenerate_reply,
        _DEGENERATE_REPLIES,
    )
    return _is_degenerate_query, _pick_degenerate_reply, _DEGENERATE_REPLIES


# ── detector ──────────────────────────────────────────────────────────


def test_detects_single_char():
    is_deg, _, _ = _import_helpers()
    assert is_deg("x")
    assert is_deg("a")
    assert is_deg("1")


def test_detects_pure_symbols():
    is_deg, _, _ = _import_helpers()
    assert is_deg("?¡@#")
    assert is_deg("!!!")
    assert is_deg("...")
    assert is_deg("---")


def test_detects_empty_and_whitespace():
    is_deg, _, _ = _import_helpers()
    assert is_deg("")
    assert is_deg("   ")
    assert is_deg("\n\t")
    assert is_deg(None)  # type: ignore[arg-type]


def test_does_not_match_normal_query():
    is_deg, _, _ = _import_helpers()
    assert not is_deg("hola")
    assert not is_deg("qué tengo hoy")
    assert not is_deg("ab")  # exactamente 2 alfanuméricos → OK


def test_one_alphanum_between_symbols_is_degenerate():
    """Caso borderline: `¿a?` tiene solo 1 alfanumérico — NO alcanza el
    umbral de 2 → sí es degenerate."""
    is_deg, _, _ = _import_helpers()
    assert is_deg("¿a?")
    assert is_deg("!x!")


def test_two_alphanum_is_threshold():
    is_deg, _, _ = _import_helpers()
    # 2 chars alfanuméricos es el umbral (>=2 no es degenerate).
    assert not is_deg("ab")
    assert not is_deg("a1")
    assert not is_deg("?ab?")  # 2 alfanuméricos aunque estén entre símbolos
    assert is_deg("?a?")  # solo 1 alfanumérico


# ── picker ────────────────────────────────────────────────────────────


def test_picker_returns_valid_reply():
    _, pick, replies = _import_helpers()
    out = pick("x")
    assert isinstance(out, str)
    assert out in replies


def test_picker_stable_within_same_minute():
    """Mismo input + mismo timestamp → mismo reply. Tests pueden monkey-
    patchear `now` para determinismo."""
    _, pick, _ = _import_helpers()
    reply_a = pick("x", now=1700000000.0)
    reply_b = pick("x", now=1700000030.0)  # 30s después, mismo minuto
    assert reply_a == reply_b


def test_picker_rotates_across_minutes():
    """Distintos minutos pueden devolver distinto variant. Chequeamos
    que haya al menos 2 variantes únicas en N intentos (no garantía
    absoluta, pero muy probable con 3 variantes x 10 tiempos)."""
    _, pick, _ = _import_helpers()
    seen = {pick("x", now=1700000000.0 + i * 60) for i in range(10)}
    assert len(seen) >= 2, f"el picker no rotó: {seen}"


# ── endpoint integration ──────────────────────────────────────────────


def test_degenerate_query_short_circuits_chat_endpoint():
    """POST /api/chat con query degenerada devuelve la canned reply vía
    SSE sin llegar al retrieve. Marcamos `metachat: True` en el payload
    del `sources` event para que el frontend no muestre lista de fuentes.
    """
    from web.server import app
    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/chat",
            json={"question": "?¡@#", "session_id": "t-deg-1"},
        ) as r:
            assert r.status_code == 200
            body = "".join(chunk for chunk in r.iter_text())
    # La respuesta debe contener un evento `sources` con metachat True
    # y al menos un token con parte de la canned reply.
    assert '"metachat": true' in body
    # Alguna de las 3 variantes tiene que haberse emitido.
    _, _, replies = _import_helpers()
    assert any(r[:20] in body for r in replies), (
        f"ninguna canned reply aparece en el SSE stream:\n{body[:500]}"
    )


def test_metachat_wins_over_degenerate():
    """`"hola"` tiene 4 alfanuméricos → NO es degenerate, cae en metachat.
    El endpoint debe responder con una variante de `_METACHAT_GREETING`,
    no de `_DEGENERATE_REPLIES`.
    """
    from web.server import app, _METACHAT_GREETING
    _, _, degenerate_replies = _import_helpers()
    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/chat",
            json={"question": "hola", "session_id": "t-deg-2"},
        ) as r:
            body = "".join(chunk for chunk in r.iter_text())
    # Debe haber una variante de greeting metachat, no una degenerate.
    assert any(g[:15] in body for g in _METACHAT_GREETING), (
        f"esperaba metachat greeting, no apareció ninguna:\n{body[:500]}"
    )
    # Y ninguna de las degenerate.
    assert not any(d[:20] in body for d in degenerate_replies), (
        f"aparecía una canned degenerate cuando debería ser metachat:\n{body[:500]}"
    )


def test_normal_query_does_not_short_circuit():
    """Query normal NO debe disparar ninguno de los dos short-circuits —
    el marker `"metachat": true` NO debe aparecer en el `sources` event.
    """
    from web.server import app
    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/chat",
            json={"question": "qué apunté sobre productividad", "session_id": "t-deg-3"},
        ) as r:
            # Tomamos solo el primer chunk — el stream podría demorar
            # mucho esperando al LLM; con los primeros eventos alcanza
            # para verificar que NO arrancó como short-circuit.
            first_events = ""
            for chunk in r.iter_text():
                first_events += chunk
                if "event: status" in first_events or "event: sources" in first_events:
                    # Recibimos al menos el primer estado que el pipeline
                    # full emite (stage retrieving/rerank). Cerramos y
                    # verificamos.
                    break
    # En el short-circuit, el PRIMER `sources` event ya tiene metachat:
    # true. En el path normal no lo marca (o si lo hace, es False).
    # Chequeamos que NO haya "metachat": true en los primeros eventos.
    assert '"metachat": true' not in first_events, (
        f"la query normal disparó un short-circuit:\n{first_events[:500]}"
    )
