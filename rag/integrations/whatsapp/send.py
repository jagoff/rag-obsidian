"""WhatsApp send paths — POST al bridge HTTP local + ambient/draft mode.

Tres surfaces:

- ``_whatsapp_send_to_jid(jid, text, *, anti_loop, reply_to)`` — low-level POST
  a `http://localhost:8080/api/send`. Retorna True en 2xx, False en cualquier
  otra cosa. Hard kill-switch via `RAG_DISABLE_WHATSAPP_SEND=1` /
  `RAG_TESTING=1` (usado por `tests/conftest.py`).
- ``_ambient_whatsapp_send(jid, text)`` — fire-and-forget wrapper con anti-loop
  marker (U+200B) prefixed para que el listener bot no procese su propio
  output como query. Cuando ``RAG_DRAFT_VIA_RAGNET`` está prendida, redirige
  TODOS los ambient sends al grupo RagNet con header ``📨 *RagNet draft*``.
- ``_GROUP_PREFIX_RE`` (NO LIVE acá) → ver `resolve.py`.

Invariantes:
- Silent-fail: cualquier excepción → False, nunca raise out.
- Anti-loop marker U+200B se prefixa SOLO cuando ``anti_loop=True``.
- Bridge HTTP soporta ``ContextInfo``/quoted messages desde 2026-05-09.
  ``reply_to`` se propaga al payload y el bridge construye el
  ``ExtendedTextMessage`` con ContextInfo (cita boxed nativa de WhatsApp
  visible al destinatario).

Why deferred imports (`from rag import ...` adentro del cuerpo):
``AMBIENT_WHATSAPP_BRIDGE_URL`` y ``_AMBIENT_ANTILOOP_MARKER`` viven en
`rag/__init__.py`. Module-level imports acá deadlock-ean el package load;
function-body imports respetan runtime monkey-patches.
"""

from __future__ import annotations

import json
import os


def _ambient_whatsapp_send(jid: str, text: str) -> bool:
    """Fire-and-forget al bridge local de WhatsApp. Retorna True en 2xx.

    POSTea a `http://localhost:8080/api/send` con body
    `{recipient: <jid>, message: <text>}`. El listener del bot RAG
    filtra mensajes que arrancan con U+200B (anti-loop) — se prefixa
    acá para evitar que nuestro propio output se procese como query.

    ── DRAFT MODE (testing flag, 2026-04-28) ────────────────────────────
    Cuando ``RAG_DRAFT_VIA_RAGNET`` está prendida (``1``/``true``/``yes``),
    TODOS los ambient sends (morning brief, archive push, reminder push,
    contradicciones digest, etc.) se redirigen al grupo RagNet
    (``WHATSAPP_BOT_JID``) en vez de mandarse al destinatario original.
    El header ``📨 *RagNet draft* → <jid_original>`` se prepende para que
    se vea claro qué iba a salir y a dónde.

    Sirve para revisar qué dispararía el sistema sin spamear al user
    (típico antes de enable/disable de una feature nueva, o cuando tunás
    el morning brief y querés iterar sin notificar).

    El flag NO afecta sends user-initiated (``propose_whatsapp_send``,
    replies, scheduled messages, contact cards) — esos van directo via
    ``_whatsapp_send_to_jid`` y son explícitamente confirmados por el
    user con [Enviar] en la proposal card.

    Para apagarlo: ``unset RAG_DRAFT_VIA_RAGNET`` y reiniciar los daemons
    afectados (``launchctl bootout`` + ``bootstrap`` de los plists tipo
    ``com.fer.obsidian-rag-*``).
    """
    from rag.integrations.whatsapp._constants import WHATSAPP_BOT_JID
    if jid != WHATSAPP_BOT_JID and os.environ.get(
        "RAG_DRAFT_VIA_RAGNET", "",
    ).lower() in ("1", "true", "yes"):
        text = f"📨 *RagNet draft* → `{jid}`\n\n{text}"
        jid = WHATSAPP_BOT_JID
    # Re-resolve `_whatsapp_send_to_jid` via package namespace so that
    # tests `monkeypatch.setattr(_waint, "_whatsapp_send_to_jid", ...)`
    # propagate to the call site.
    from rag.integrations.whatsapp import _whatsapp_send_to_jid
    return _whatsapp_send_to_jid(jid, text, anti_loop=True)


def _whatsapp_send_to_jid_detailed(
    jid: str,
    text: str,
    *,
    anti_loop: bool = True,
    reply_to: dict | None = None,
) -> tuple[bool, str]:
    """Variante de `_whatsapp_send_to_jid` que devuelve `(ok, error_kind)`
    en lugar de solo bool. `error_kind` es uno de:

      - `"sent"` — 2xx del bridge.
      - `"disabled"` — RAG_DISABLE_WHATSAPP_SEND / RAG_TESTING activo.
      - `"unreachable"` — bridge no responde (connection refused, timeout
        a nivel TCP). El daemon probablemente está caído o el puerto cambió.
      - `"rejected"` — bridge devolvió 4xx/5xx (e.g., JID inválido, jurar
        inactiva). Reintentar no va a ayudar sin intervención.
      - `"timeout"` — request fue aceptada pero excedió 3s. Puede haber
        sido entregada o no — ambiguo.
      - `"error:<repr>"` — cualquier otra excepción no catalogada.

    Endpoints HTTP usan esto para devolver un detail más útil al user
    (e.g., 502 con detail "bridge unreachable" vs 502 con detail "bridge
    rejected" → distinto remediation).
    """
    from rag import AMBIENT_WHATSAPP_BRIDGE_URL, _AMBIENT_ANTILOOP_MARKER
    import urllib.request
    import urllib.error
    if (os.environ.get("RAG_DISABLE_WHATSAPP_SEND", "").lower() in ("1", "true", "yes")
        or os.environ.get("RAG_TESTING", "").lower() in ("1", "true", "yes")):
        return False, "disabled"
    payload_text = text
    if anti_loop and not text.startswith(_AMBIENT_ANTILOOP_MARKER):
        payload_text = _AMBIENT_ANTILOOP_MARKER + text
    body: dict = {
        "recipient": jid,
        "message": payload_text,
    }
    if reply_to and isinstance(reply_to, dict):
        rt_id = reply_to.get("message_id") or reply_to.get("id")
        if rt_id:
            body["reply_to"] = {
                "message_id": str(rt_id),
                "original_text": str(reply_to.get("original_text") or reply_to.get("text") or "")[:1024],
                "sender_jid": str(reply_to.get("sender_jid") or reply_to.get("from_jid") or ""),
            }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        AMBIENT_WHATSAPP_BRIDGE_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if 200 <= resp.status < 300:
                return True, "sent"
            return False, "rejected"
    except urllib.error.HTTPError:
        return False, "rejected"
    except urllib.error.URLError as exc:
        # Differential: TimeoutError es timeout-after-connect; otros URLError
        # tipo ConnectionRefusedError son unreachable.
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError):
            return False, "timeout"
        return False, "unreachable"
    except TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, f"error:{exc!r}"[:120]


def _whatsapp_send_to_jid(
    jid: str,
    text: str,
    *,
    anti_loop: bool = True,
    reply_to: dict | None = None,
) -> bool:
    """Low-level POST al bridge local. Dos modos:

    - ``anti_loop=True`` (default, usado por ``_ambient_whatsapp_send``):
      prefixa U+200B para que el listener del bot RAG ignore el mensaje
      como query entrante. Necesario cuando el bot se manda cosas a su
      propio grupo (briefs matutinos, archive pushes, etc.).
    - ``anti_loop=False``: texto literal. Usalo cuando el destinatario
      es un contacto tercero (mensajes iniciados desde el chat del user
      vía ``propose_whatsapp_send``), porque el prefix se vería como un
      char raro en el WhatsApp del contacto.

    ``reply_to`` (optional): cuando el caller quiere responder a un
    mensaje específico con quote nativo de WhatsApp. Shape esperado:
    ``{"message_id": str, "original_text": str, "sender_jid": str?}``.

    Soporte nativo desde 2026-05-09: el bridge Go (main.go) acepta
    ``reply_to`` en el payload y construye ``ExtendedTextMessage`` con
    ``ContextInfo`` (StanzaID + Participant + QuotedMessage). El receptor
    ve la cita boxed nativa de WhatsApp ("respondiendo a..."). Para
    grupos el ``sender_jid`` es required (el JID full del autor del
    quoted msg); para 1:1 puede ser el chat_jid mismo.

    Retorna True en 2xx del bridge, False en cualquier otra cosa
    (unreachable, 4xx, 5xx, timeout 10s).
    """
    from rag import AMBIENT_WHATSAPP_BRIDGE_URL, _AMBIENT_ANTILOOP_MARKER
    import urllib.request
    # Hard kill-switch para tests / dry-run. 2026-04-30 (bug fix tras
    # leak del NARRATIVE_STUB de tests/test_today.py al grupo RagNet
    # real: 4 evening briefs duplicados con texto placeholder posteados
    # al WhatsApp del user). Defensa en profundidad — si algún test
    # olvida mockear, este guard previene el blast radius.
    #
    # ENV vars que activan el guard (cualquiera de las dos basta):
    #   RAG_DISABLE_WHATSAPP_SEND=1   — explícito, override máximo
    #   RAG_TESTING=1                 — alias semántico
    #
    # `tests/conftest.py` setea RAG_DISABLE_WHATSAPP_SEND=1 por default
    # vía autouse session-scoped fixture; los tests que necesitan
    # ejercitar el path real del send (mockeando `urlopen` directamente)
    # hacen `monkeypatch.delenv("RAG_DISABLE_WHATSAPP_SEND")` para opt-out.
    # NO chequeamos PYTEST_CURRENT_TEST acá porque hay tests legítimos
    # del wire format del send que se rompen — el opt-in vía conftest +
    # delenv selectivo es más explícito y trackeable.
    if (os.environ.get("RAG_DISABLE_WHATSAPP_SEND", "").lower() in ("1", "true", "yes")
        or os.environ.get("RAG_TESTING", "").lower() in ("1", "true", "yes")):
        return False
    payload_text = text
    if anti_loop and not text.startswith(_AMBIENT_ANTILOOP_MARKER):
        payload_text = _AMBIENT_ANTILOOP_MARKER + text
    body: dict = {
        "recipient": jid,
        "message": payload_text,
    }
    if reply_to and isinstance(reply_to, dict):
        # Bridge ContextInfo desde 2026-05-09: estos campos disparan la
        # construcción de ExtendedTextMessage con cita boxed nativa.
        # Ver docstring arriba.
        rt_id = reply_to.get("message_id") or reply_to.get("id")
        if rt_id:
            body["reply_to"] = {
                "message_id": str(rt_id),
                "original_text": str(reply_to.get("original_text") or reply_to.get("text") or "")[:1024],
                "sender_jid": str(reply_to.get("sender_jid") or reply_to.get("from_jid") or ""),
            }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        AMBIENT_WHATSAPP_BRIDGE_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    # Audit 2026-04-26 (BUG #46): bajar timeout 10→3s. Bridge HTTP en
    # localhost responde en <100ms warm; 10s es excesivo y bloquea
    # el ambient hook (sync-call desde indexing) si el bridge crashea
    # / restartea. 3s sigue tolerando spike normal.
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


__all__ = [
    "_ambient_whatsapp_send",
    "_whatsapp_send_to_jid",
    "_whatsapp_send_to_jid_detailed",
]
