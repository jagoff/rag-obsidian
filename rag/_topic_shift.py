"""Topic-shift detection — gate combinado para descartar history irrelevante.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer la detección de cambio de tema desde `rag/__init__.py`.

## Contexto

Protege a la conversación de contaminación cross-tópico cuando el
user cambia de tema abruptamente sin cerrar el hilo anterior.
Reportado 2026-04-20: session `web:b03ec059db32` → turno T-1 "cual
es mi password de avature?" → turno T "busca informacion sobre mi
mama" produjo respuesta mezclada porque los 6 últimos mensajes de
history incluían el hilo de Avature y el chat LLM arrastró la
inercia temática.

## Gate combinado

  1. Anafóricos explícitos ("eso", "ella", "y eso?", "más sobre X")
     → NO shift (el follow-up necesita la history para resolver
     el antecedente).
  2. `person_fired` → SHIFT (nombrar a alguien de 99-Mentions
     re-frame el turno; el dossier inyectado por
     `build_person_context` es autoritativo).
  3. cosine(current, last_user_q) < `TOPIC_SHIFT_COSINE` → SHIFT.
     Umbral 0.32 empírico tras feedback de continuidad multi-turn.

## Re-export

`rag/__init__.py` hace `from rag._topic_shift import *  # noqa`.
Preserva 100% compat con call sites históricos.
"""

from __future__ import annotations

import re

__all__ = [
    "_TOPIC_SHIFT_FOLLOWUP_RE",
    "TOPIC_SHIFT_COSINE",
    "detect_topic_shift",
]


# Regex local — el gate vive en rag.py (CLI + web ambos lo usan) así que
# duplicamos el patrón en vez de importar de web/server.py (que a su vez
# importa rag). Mantener sincronizado con _FOLLOWUP_CUES en web/server.py si
# cambia.
_TOPIC_SHIFT_FOLLOWUP_RE = re.compile(
    r"\b(eso|esa|ese|esto|esta|este|aquello|ella|[eé]l\b(?!\s+[A-ZÁÉÍÓÚ])|"
    r"ah[ií]|all[íáa]|it\b|this\b|that\b|he\b|she\b|"
    r"y\s+(de|sobre|con|para|en|c[oó]mo|qu[eé])|"
    r"c[oó]mo\s+(lo|la|los|las)\s+\w+|"  # "como lo desactivo", "como la uso"
    r"profundiz[aá]|ampl[ií]a|segu[ií]|contin[uú]a|"
    r"m[aá]s\s+(sobre|de|al\s+respecto)|"
    # 2026-04-28 wave-6: follow-up phrases comunes en multi-turn detectadas
    # via Playwright. "qué otros X me recomendarías" / "cuál era X" /
    # "dame un ejemplo" / "explicame mejor" son CLARAMENTE referencias al
    # turno anterior, pero el cosine cae en 0.36-0.39 (debajo del umbral
    # 0.40 actual) porque las palabras vacías ("otros", "cuál", "dame") no
    # cargan signal. Whitelist explícita.
    r"qu[eé]\s+otros?|qu[eé]\s+otras?|"        # "qué otros materiales"
    # 2026-05-04: sacamos `es` del verb-set. Era falso-positivo masivo —
    # "cuál es el/la X" es el patrón estándar de pregunta factual en
    # español ("cuál es la capital de Francia", "cuál es el nro serial
    # de la guitarra fender") y matcheaba como anáfora arrastrando la
    # history del turno anterior. Past/conditional (era/fue/sería/serían)
    # SÍ son señal fuerte de referencia hacia atrás. Las anáforas legítimas
    # con presente ("cuál es el último que dijiste") las atrapa la rule
    # de ordinales abajo.
    r"cu[aá]l\s+(?:era|fue|ser[ií]a|ser[ií]an)\s+(?:el|la|los|las)?|"  # "cuál era el primer punto"
    r"dame\s+(?:un|el|los|otro)|"               # "dame un ejemplo"
    r"un\s+ejemplo|otro\s+ejemplo|"             # "un ejemplo más"
    r"explic[aá]me\s+(?:mejor|bien|m[aá]s|otra\s+vez)|"
    r"resum[ií]me|"                             # "resumime eso"
    r"tradu[cz]i?[ií]?(?:me|lo|la)?|"           # "traducíme/traducilo"
    r"y\s+despu[eé]s|y\s+entonces|y\s+ahora|"  # "y después?"
    r"recomend[aá]ri[aá]s|sugerir[ií]as|"      # "qué recomendarías"
    # 2026-04-28 P3: extensiones rioplatenses que el regex anterior no
    # cubría — cosine-an bajo (0.20-0.35) pero son CLARAMENTE referencias
    # al turno anterior. Detectados en conversaciones reales con el bot.
    r"contame\s+(?:m[aá]s|otra|c[oó]mo|qu[eé])|"   # "contame más", "contame otra cosa"
    r"cont[aá]\s+(?:m[aá]s|otra)|"                  # "contá más" (más informal)
    r"del\s+(?:primer[oa]?|segund[oa]|tercer[oa]?|"
    r"otr[oa]|anterior|[uú]ltim[oa])\b|"            # "del primero", "del último", "del anterior"
    r"\b(?:el|la)\s+(?:primer[oa]?|segund[oa]|"
    r"[uú]ltim[oa])\s+(?:que|de|del)|"              # "el primero que dijiste"
    r"alguna?\s+(?:otra|m[aá]s)\s+(?:cosa|opci[oó]n|idea|forma|manera)|"  # "alguna otra opción"
    r"algo\s+m[aá]s\s+(?:sobre|de|que)|"            # "algo más sobre eso"
    r"otra\s+vez\s+(?:eso|el|la|los|las)|"          # "otra vez eso?"
    r"explic[aá](?:lo|la|me)\b|"                    # "explicalo", "explicame"
    r"d[aá]le\s+(?:m[aá]s|otra|otro)|"              # "dale más", "dale otro"
    r"y\s+(?:vos|usted|tu|tú)\s+qu[eé]|"            # "y vos qué pensás?" (continuación)
    r"justo\s+(?:eso|esto|por\s+eso))\b",           # "justo eso quería preguntar"
    # NOTA: "claro" / "bien" / "aha" sueltos NO se incluyen acá. Como acuses
    # son anáforas reales, pero matchearlas en ANY position genera too
    # many false-keeps ("entendí bien, qué es X" mantendría history aunque
    # X sea unrelated). Lo dejo sin cubrir; si emerge como problema real,
    # mejor agregar gate "starts-with acuse" + cosine combinado.
    re.IGNORECASE,
)

# 2026-04-28 P3: bajado 0.40 → 0.32 tras feedback de continuidad multi-turn.
# Caso reportado por el user: pregunta 1 + pregunta 2 relacionadas pero con
# vocabulario distinto (paráfrasis sin overlap léxico) cosine-an en
# 0.32-0.39 → con threshold 0.40 caía como "shift" y descartaba history,
# las dos preguntas se respondían como conversaciones separadas.
#
# Trade-off del cambio:
#   - Antes (0.40): false-shifts en paráfrasis = MAYORÍA de los problemas.
#     False-keeps en cambios reales = pocos.
#   - Ahora (0.32): false-shifts ↓ (paráfrasis con cosine 0.32-0.40 ahora
#     mantienen history, fix correcto). False-keeps ↑ (cambios reales con
#     cosine 0.32-0.40 mantienen history y traen contexto irrelevante al
#     LLM). El balance neto se eligió tras ver que la mayoría de cambios
#     REALES de tema en este vault caen en cosine < 0.30 (medido en
#     conversaciones reales de RagNet 2026-04-26 a 2026-04-28).
#
# El borderline LLM reformulator del web server cubre [0.32, 0.70] —
# cuando el cosine es ambigüo el LLM reformula la query con el contexto
# del turn anterior antes del retrieve, mitigando los false-keeps.
#
# Si volvemos a ver "no encuentra relación entre turns": subir window
# antes de bajar el cosine de nuevo. Si vemos "trae contexto random":
# considerar volver a 0.40 y mejorar el regex de follow-ups en su lugar.
TOPIC_SHIFT_COSINE = 0.32


def detect_topic_shift(
    current_q: str,
    history: list[dict],
    *,
    person_fired: bool,
) -> tuple[bool, str, float | None]:
    """¿El turno actual cambia de tema vs el anterior? Devuelve (shift, razón, cosine).

    - `shift=True` → el caller debería descartar history para este turno.
    - `razón` es un string corto para logging/observabilidad.
    - `cosine` es la similitud bge-m3 entre current_q y last_user_q cuando
      el cosine gate corrió, o None cuando un gate previo cortó (short,
      anaphoric, person, embed-failure). Lo expone el web chat handler
      para decidir search-query strategy: cosine ≥ 0.7 → query autónoma
      (raw), [0.4-0.7) → borderline (reformulate LLM), < 0.4 → ya hubo
      shift (history dropeado, raw).

    No muta nada. Pure function, testeable sin vault/session.
    """
    from rag import _match_mentions_in_query, cosine_sim, embed  # noqa: PLC0415

    if not history:
        return False, "no-history", None
    # (1) Anafóricos: pronombres, demostrativos, "y X?", "como lo Y". Short
    # follow-ups (≤2 tokens) también entran acá por construcción — una
    # query de 1-2 palabras es casi siempre elipsis del turno anterior
    # ("y?", "más?", "ella?"). 3+ tokens ya puede ser standalone (ej.
    # "notas de coaching") y cae al cosine gate.
    if len(current_q.split()) <= 2:
        return False, "short", None
    if _TOPIC_SHIFT_FOLLOWUP_RE.search(current_q):
        return False, "anaphoric", None
    # (2) Person-mention gate: build_person_context ya inyectó el dossier de
    # la persona; history de otros temas es ruido puro. Pero si el turno
    # anterior menciona a LA MISMA persona, es follow-up mismo-tema
    # (ej.: "contame sobre mi mama" → "y cuándo cumple años?") → keep.
    last_user_q = next(
        (m.get("content") for m in reversed(history) if m.get("role") == "user"),
        None,
    )
    if person_fired:
        if last_user_q:
            try:
                current_people = set(_match_mentions_in_query(current_q))
                last_people = set(_match_mentions_in_query(last_user_q))
            except Exception:
                current_people = last_people = set()
            if current_people and current_people == last_people:
                return False, "same-person", None
        return True, "person", None
    # (3) Cosine gate contra el último user turn.
    if not last_user_q:
        return False, "no-last-user", None
    try:
        vecs = embed([current_q, last_user_q])
        if len(vecs) < 2:
            return False, "embed-empty", None
        sim = cosine_sim(vecs[0], vecs[1])
    except Exception as exc:
        # Silent fallback — embed puede fallar si el modelo no carga; en ese
        # caso preferimos mantener history (fail-safe para follow-ups) antes
        # que dropearla agresivamente.
        return False, f"embed-failed:{type(exc).__name__}", None
    if sim < TOPIC_SHIFT_COSINE:
        return True, f"cosine={sim:.3f}", float(sim)
    return False, f"cosine={sim:.3f}", float(sim)
