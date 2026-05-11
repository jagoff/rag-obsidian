"""Inline translate de mensajes de WhatsApp.

Long-press / menu en una burbuja inbound → endpoint LLM helper devuelve
traducción al español rioplatense. qwen2.5:3b ya está warm in-process,
costo ~0.5-2s warm, sin red. Cache process-level por (msg_id, target)
para que re-pegar al mismo msg sea instant.

Detección de idioma: heurística simple por chars/palabras antes del LLM
call — saltea si ya parece español. El LLM la auto-detecta de todas
formas, esto solo evita la call innecesaria en >90% de los casos.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger("rag.wa.translate")

_CACHE: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 512

# Heurística rápida para "ya está en español rioplatense". Si matchea
# alguno de estos markers, NO llamamos al LLM — devolvemos None y el
# frontend muestra "ya está en español" o no muestra nada.
_ES_MARKERS = (
    "vos ",
    " vos ",
    " che ",
    "dale",
    "boludo",
    " está ",
    " está\n",
    "están ",
    "él ",
    "ella ",
    "qué ",
    "cómo ",
    "está bien",
    "estoy ",
    "ahora ",
)

# Markers que delatan otro idioma. Si vemos esto, vale la pena llamar
# al LLM. Cubre los casos más frecuentes en el corpus del user (pt-BR,
# inglés, italiano).
_NON_ES_MARKERS = (
    "você",
    "obrigad",
    "agora",
    "porque ",
    "muito ",
    "hello",
    "sorry",
    "thanks",
    "please",
    "ciao",
    "grazie",
)


def detect_likely_non_spanish(text: str) -> bool:
    """True si el texto parece NO estar en español rioplatense — vale
    la pena traducir. False si parece español o si es muy corto para
    decidir (<6 chars).
    """
    s = (text or "").strip().lower()
    if len(s) < 6:
        return False
    # Match explícito de otro idioma → traducir.
    for m in _NON_ES_MARKERS:
        if m in s:
            return True
    # Match español → no traducir.
    for m in _ES_MARKERS:
        if m in s:
            return False
    # CJK / arabic / cyrillic → traducir.
    if re.search(r"[一-鿿぀-ヿ؀-ۿЀ-ӿ]", s):
        return True
    # Sino, dejamos al LLM auto-detectar — devolvemos True para llamar.
    return True


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


def translate(content: str, *, msg_id: str = "", target: str = "es-AR") -> dict | None:
    """Traduce `content` al `target`. Devuelve:

      {"translated": "...", "source_lang": "pt|en|...", "skipped": False}
      {"skipped": True, "reason": "already_spanish"} si la heurística
      decide que no hace falta.

    None si el LLM falla. Cache por `(msg_id, target)` cuando msg_id
    se provee — para que re-click no re-traduzca.
    """
    text = (content or "").strip()
    if not text:
        return None
    if len(text) > 2000:
        text = text[:2000]
    if msg_id:
        cached = _cache_get((msg_id, target))
        if cached is not None:
            return cached

    if not detect_likely_non_spanish(text):
        result = {"skipped": True, "reason": "already_spanish"}
        if msg_id:
            _cache_put((msg_id, target), result)
        return result

    try:
        from rag import (  # noqa: PLC0415
            HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client,
        )
    except Exception as exc:
        logger.warning("imports failed: %s", exc)
        return None

    # Prompt con few-shot + delimiters claros para forzar literal
    # translation sin que el modelo (qwen2.5:3b 4bit) leakee instrucciones
    # al output o invente texto. Verificado 2026-05-11 que sin few-shot
    # producía traducciones rotas estilo "Sos un vaso para stor".
    prompt = (
        "Traducí texto al español rioplatense (voseo argentino, "
        "informal). Detectá el idioma origen. PROHIBIDO: agregar "
        "comentarios, explicaciones, prefijos como \"Traducción:\", "
        "o cualquier texto fuera del JSON. Solo traducción literal "
        "manteniendo tono.\n\n"
        "EJEMPLOS:\n"
        "INPUT: Você sabe que eu te amo muito\n"
        'OUTPUT: {"translated": "Sabés que te quiero mucho", "source_lang": "pt"}\n\n'
        "INPUT: Sorry for the late reply, will get back to you tomorrow\n"
        'OUTPUT: {"translated": "Perdón por la demora, te respondo mañana", "source_lang": "en"}\n\n'
        "INPUT: Ciao bella, come stai?\n"
        'OUTPUT: {"translated": "Hola linda, ¿cómo andás?", "source_lang": "it"}\n\n'
        "Ahora traducí este texto:\n"
        f"INPUT: {text}\n"
        "OUTPUT:"
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
    translated = (data.get("translated") or "").strip()
    if not translated:
        return None
    source_lang = (data.get("source_lang") or "").strip().lower()[:6]

    result = {
        "translated": translated,
        "source_lang": source_lang or "unknown",
        "skipped": False,
    }
    if msg_id:
        _cache_put((msg_id, target), result)
    return result


__all__ = ["translate", "detect_likely_non_spanish"]
