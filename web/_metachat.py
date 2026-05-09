"""Meta-chat + degenerate-query short-circuits — canned replies para
turnos que NO ameritan un round-trip al LLM.

Extraído de ``web/server.py`` (Phase W2, 2026-05-08). Funciones puras
sin thread state. Stdlib only (`hashlib`, `time`).

## Surface

Metachat (saludos / agradecimientos / despedidas / "qué podés hacer"):
- ``_METACHAT_GREETING``, ``_METACHAT_THANKS``, ``_METACHAT_META``,
  ``_METACHAT_BYE`` — tuples de variantes canned por bucket.
- ``_metachat_bucket(q)`` — clasifica el input en bucket → tuple de variantes.
- ``_pick_metachat_reply(q, *, now)`` — elige variante con seed estable
  por minute-bucket (retries dentro del mismo minuto devuelven la misma
  frase; entre minutos rotan).

Degenerate queries (`"x"`, `"?"`, `"?¡@#"`, vacías):
- ``_DEGENERATE_REPLIES`` — tuple de variantes canned.
- ``_is_degenerate_query(q)`` — True si la query tiene <2 caracteres
  alfanuméricos totales.
- ``_pick_degenerate_reply(q, *, now)`` — misma seeding que metachat.

## Por qué módulo separado

Estos helpers se llaman ANTES de tocar retrieve / LLM. Leerlos sin
cargar el FastAPI app entero acelera tests del short-circuit
(`test_metachat_intent`, etc.) — pure-functional units sin
side-effects.
"""

from __future__ import annotations

import hashlib
import time as _time


# Canned replies for the meta-chat short-circuit. Buckets keyed by the
# class of input; within a bucket we pick one variant by hashing the
# message + the current minute so the same phrase in a tight window is
# stable (no user surprise if they re-send) but repeat visits pick
# different variants (feels alive, not scripted).
_METACHAT_GREETING = (
    "¡Hola! Preguntame lo que quieras sobre tus notas, o decime *recordame …* / *agendá …* si querés crear algo.",
    "Hola 👋 ¿en qué te ayudo? Probá una pregunta sobre tus notas o pedime *recordame …* / *agendá …*.",
    "¡Buenas! Tirame una pregunta, o decime *recordame X* / *el viernes 20hs X* para crear un recordatorio o evento.",
)
_METACHAT_THANKS = (
    "¡De nada!",
    "¡Cuando quieras!",
    "👌",
)
_METACHAT_META = (
    "Puedo responder sobre tus notas, crear recordatorios (*recordame X*) y eventos de calendar (*el viernes 20hs …*). Probá algo.",
    "Consulto tu vault de Obsidian y creo recordatorios / eventos desde texto libre. ¿Qué necesitás?",
    "Leo tus notas y armo recordatorios o eventos cuando se los pedís en lenguaje natural. Tirame algo concreto.",
)
_METACHAT_BYE = (
    "¡Hasta luego!",
    "¡Nos vemos!",
    "👋",
)


def _metachat_bucket(q: str) -> tuple[str, ...]:
    """Classify the meta-chat message into a response bucket."""
    s = q.strip().lower().lstrip("¿ ").lstrip()
    if s.startswith(("gracias", "muchas gracias", "mil gracias",
                     "dale gracias", "thanks", "thx")):
        return _METACHAT_THANKS
    if s.startswith(("chau", "bye", "adiós", "adios", "nos vemos")):
        return _METACHAT_BYE
    if s.startswith(("qué podés", "que podes", "qué sabés", "que sabes",
                     "cómo funcion", "como funcion", "cómo te us",
                     "como te us", "qué comandos", "que comandos",
                     "ayuda", "help", "quién sos", "quien sos",
                     "quién es este", "quien es este")):
        return _METACHAT_META
    return _METACHAT_GREETING


def _pick_metachat_reply(q: str, *, now: float | None = None) -> str:
    """Pick a canned reply for a meta-chat turn.

    Variation seed = hash(q) XOR minute-bucket. Same input within the
    same minute returns the same variant (stable on retry); different
    inputs or different minutes rotate. Tests monkey-patch with fixed
    `now` for determinism.
    """
    bucket = _metachat_bucket(q)
    ts = now if now is not None else _time.time()
    minute = int(ts // 60)
    seed = int(hashlib.sha256(f"{q}|{minute}".encode()).hexdigest()[:8], 16)
    return bucket[seed % len(bucket)]


# ── Degenerate query short-circuit (2026-04-23) ──────────────────────
# Queries con <2 caracteres alfanuméricos (ej. "x", "?", "?¡@#") caían
# al retrieve + rerank y devolvían chunks random de WhatsApp porque el
# matching semántico sobre un input casi vacío es puro ruido. Medido en
# scratch_eval: `?¡@#` → 395 chars de contenido WA sin relación. Ahora
# devolvemos una respuesta canned antes de tocar retrieve/LLM,
# invitando al usuario a reformular.
_DEGENERATE_REPLIES: tuple[str, ...] = (
    "No entendí tu pregunta. Podés reformularla con más detalle?",
    "Necesito un poco más de contexto. Qué querés consultar de tus notas?",
    "Tu mensaje parece muy corto o sin contenido. Preguntame algo concreto sobre el vault.",
)


def _is_degenerate_query(q: str) -> bool:
    """True si la query tiene <2 caracteres alfanuméricos totales.

    Evita que `"x"`, `"?"`, `"?¡@#"`, strings de puro símbolo, o cadenas
    vacías disparen el pipeline full — no hay suficiente señal para que
    el retrieve devuelva algo útil ni para que el LLM produzca una
    respuesta honesta. Metachat tiene su propio short-circuit; esta
    función se encarga de lo que no alcanza ni siquiera a ser saludo.
    """
    if not q or not q.strip():
        return True
    alphanum = sum(1 for c in q if c.isalnum())
    return alphanum < 2


def _pick_degenerate_reply(q: str, *, now: float | None = None) -> str:
    """Pick a canned reply for a degenerate-query turn. Same seeding as
    metachat (hash(q) XOR minute-bucket) so retries stay stable.
    """
    ts = now if now is not None else _time.time()
    minute = int(ts // 60)
    seed = int(hashlib.sha256(f"{q}|{minute}".encode()).hexdigest()[:8], 16)
    return _DEGENERATE_REPLIES[seed % len(_DEGENERATE_REPLIES)]


__all__ = [
    "_METACHAT_GREETING",
    "_METACHAT_THANKS",
    "_METACHAT_META",
    "_METACHAT_BYE",
    "_DEGENERATE_REPLIES",
    "_metachat_bucket",
    "_pick_metachat_reply",
    "_is_degenerate_query",
    "_pick_degenerate_reply",
]
