"""Tone shifter pre-send — reescribe el draft del composer en otro tono.

User typea un mensaje en `/wzp` → 4 chips abajo del composer (`+formal`,
`+casual`, `+corto`, `+cariñoso`). Click → qwen2.5:3b MLX reescribe el
texto manteniendo el contenido y la intención, ajustando solo el
registro. El composer reemplaza el input con el shifted; el original
queda en memoria por si quiere undo.

Innovador porque ningún cliente de chat hace adaptación contextual del
tono in-place. Util porque el user puede ser directo escribiendo a
clientes (suaviza), formal con familia (relaja), prolijo en sms cortos
(condensa).

Diseño:

- `shift_tone(text, tone)` → `{"shifted": str, "tone": str}` o None.
- Cache process-level LRU 512 keyed por `(text, tone)` — re-click
  entre tonos para back/forward instant.
- Few-shot prompt con 1 ejemplo por tono — verificado en `translate.py`
  que qwen2.5:3b 4bit necesita few-shot para no leakear instrucciones
  al output. Sin examples produce traducciones tipo "Sos un vaso para
  stor" (caso real).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger("rag.wa.tone_shift")

_CACHE: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 512

# Tonos soportados — keys que el frontend pasa. El prompt usa
# descripciones en español rioplatense para anclar el LLM al registro.
_TONE_DESCRIPTORS = {
    "formal": (
        "más formal y prolijo, manteniendo distancia profesional pero "
        "sin sonar acartonado"
    ),
    "casual": (
        "más relajado e informal, voseo argentino fluido, como hablás "
        "con un amigo"
    ),
    "short": (
        "más corto y directo, sin perder lo esencial — apuntar a la "
        "mitad de longitud o menos"
    ),
    "warm": (
        "más cálido y cariñoso, con calidez emocional pero sin sonar "
        "empalagoso, manteniendo voseo argentino"
    ),
}

# Examples por tono — input/output reales para anclar el modelo.
_EXAMPLES = {
    "formal": (
        ("dale, ya te paso eso", "Perfecto, te lo envío en breve."),
        ("no entiendo qué querés", "¿Podrías aclararme un poco más a qué te referís?"),
    ),
    "casual": (
        ("Solicito que me confirme la disponibilidad para el martes 14.",
         "Che, ¿estás el martes 14? avisame."),
        ("Me dirijo a usted para informarle que el pago está procesado.",
         "Ya te mandé el pago, fijate."),
    ),
    "short": (
        ("dale, te paso los archivos por mail cuando llegue a casa esta noche",
         "Te paso los archivos cuando llegue."),
        ("estoy pensando si puedo ir o no, depende un poco del trabajo",
         "Depende del laburo, te confirmo."),
    ),
    "warm": (
        ("ok, recibido",
         "Dale, gracias por avisarme ❤️"),
        ("no voy a poder ir",
         "Uff, ojalá pudiera ir — me hubiera encantado verte."),
    ),
}


def _cache_get(key: tuple[str, str]) -> dict | None:
    with _CACHE_LOCK:
        v = _CACHE.get(key)
        if v is not None:
            _CACHE.move_to_end(key)
        return v


def _cache_put(key: tuple[str, str], value: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def shift_tone(text: str, tone: str) -> dict | None:
    """Reescribe `text` en `tone`. Devuelve `{shifted, tone}` o None.

    Cache por `(text, tone)`. Si tone no soportado, None. Si text
    vacío o >1500 chars, None.
    """
    text = (text or "").strip()
    if not text or len(text) > 1500:
        return None
    if tone not in _TONE_DESCRIPTORS:
        return None
    key = (text, tone)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        from rag import (  # noqa: PLC0415
            HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client,
        )
    except Exception as exc:
        logger.warning("imports failed: %s", exc)
        return None

    descriptor = _TONE_DESCRIPTORS[tone]
    examples = _EXAMPLES.get(tone, ())
    example_block = "\n\n".join(
        f"INPUT: {ex_in}\nOUTPUT: {{\"shifted\": \"{ex_out}\"}}"
        for ex_in, ex_out in examples
    )

    prompt = (
        f"Reescribí texto en español rioplatense (voseo argentino) "
        f"hacia un tono {descriptor}. Mantené el CONTENIDO y la "
        f"INTENCIÓN del original, solo cambiá el registro. NO agregues "
        f"comentarios, explicaciones, prefijos como \"Reescritura:\", "
        f"ni cambies el sentido del mensaje. Solo la reescritura.\n\n"
        f"EJEMPLOS:\n{example_block}\n\n"
        f"Ahora reescribí este texto:\n"
        f"INPUT: {text}\n"
        f"OUTPUT:"
    )

    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 400, "num_ctx": 2048},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("llm failed: %s", exc)
        return None

    if not isinstance(data, dict):
        return None
    shifted = (data.get("shifted") or "").strip()
    if not shifted:
        return None
    # Defensa contra el LLM que devuelva el mismo input — eso significa
    # que entendió mal o que el texto ya está en ese tono. Devolvemos
    # igual pero marcamos `noop` para que el frontend muestre un hint.
    noop = shifted.lower() == text.lower()
    result = {"shifted": shifted, "tone": tone, "noop": noop}
    _cache_put(key, result)
    return result


__all__ = ["shift_tone"]
