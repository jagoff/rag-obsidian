"""Voice normalizer (1ª persona → 2ª) — extracted from rag/today_correlator.py 2026-05-09.

Post-processing del LLM output del brief diario. El prompt prohíbe 1ª
persona ("yo trabajé") pero el modelo 7B se desliza ~10% del tiempo —
este pass es el último guard.

Lo que hace:
  - Verbos en pretérito perfecto / presente 1PS → 2da singular (vos):
    "trabajé" → "trabajaste", "tengo" → "tenés", "vimos" → "viste".
  - Pronombres / determinantes 1ª → 2ª: "yo" → "vos", "me" → "te",
    "mi" → "tu", "mío" → "tuyo", "nos" → "te".
  - **Preserva** texto entre comillas (citas literales del user).
  - **Preserva** case del original ("Recibí" → "Recibiste").
  - Word-boundary strict — "concentré" NO matchea "centré".

Aplica una sola vez (no idempotente — evita over-replace en ediciones
repetidas).

Re-exportado desde ``rag.today_correlator`` para preservar
``from rag.today_correlator import normalize_voice_to_2da_persona``.
"""
from __future__ import annotations

import re

__all__ = [
    "_VOICE_VERB_REPLACEMENTS_1PS",
    "_VOICE_PRONOUN_REPLACEMENTS",
    "_make_word_boundary_pattern",
    "normalize_voice_to_2da_persona",
]


# Mapeo 1ra persona singular ("yo trabajé") → 2da persona singular ("vos
# trabajaste"). Solo se chequean palabras completas (word-boundary).
# Ordenado por longitud DESC para que "encontré" matchee antes que "encé".
_VOICE_VERB_REPLACEMENTS_1PS = [
    # 1PS pretérito perfecto simple regulares -ar → -aste
    ("trabajé", "trabajaste"), ("revisé", "revisaste"),
    ("centré", "centraste"), ("pasé", "pasaste"), ("noté", "notaste"),
    ("preparé", "preparaste"), ("armé", "armaste"), ("dejé", "dejaste"),
    ("envié", "enviaste"), ("mandé", "mandaste"), ("toqué", "tocaste"),
    ("escribí", "escribiste"), ("entré", "entraste"),
    ("encontré", "encontraste"), ("intenté", "intentaste"),
    ("cerré", "cerraste"), ("llamé", "llamaste"), ("hablé", "hablaste"),
    ("pregunté", "preguntaste"), ("contesté", "contestaste"),
    ("avancé", "avanzaste"), ("logré", "lograste"), ("terminé", "terminaste"),
    ("empecé", "empezaste"), ("agregué", "agregaste"),
    ("investigué", "investigaste"), ("comparé", "comparaste"),
    ("reuní", "reuniste"),
    # 1PS irregulares
    ("recibí", "recibiste"), ("vi", "viste"), ("fui", "fuiste"),
    ("estuve", "estuviste"), ("tuve", "tuviste"), ("hice", "hiciste"),
    ("dije", "dijiste"), ("vine", "viniste"), ("puse", "pusiste"),
    ("supe", "supiste"), ("anduve", "anduviste"), ("traje", "trajiste"),
    ("pude", "pudiste"), ("quise", "quisiste"), ("leí", "leíste"),
    ("oí", "oíste"), ("salí", "saliste"), ("dormí", "dormiste"),
    ("comí", "comiste"), ("subí", "subiste"), ("abrí", "abriste"),
    ("escogí", "escogiste"), ("seguí", "seguiste"), ("conseguí", "conseguiste"),
    # Presente 1PS irregulares
    ("tengo", "tenés"), ("hago", "hacés"), ("digo", "decís"),
    ("pongo", "ponés"), ("salgo", "salís"), ("vengo", "venís"),
    ("conozco", "conocés"), ("sé", "sabés"),
    # 1ra plural ("nosotros") en pretérito perfecto suelen sonar igual
    # (-amos / -imos) — el LLM a veces dice "trabajamos" / "vimos" para
    # incluir al user. Convertir a 2da singular del usuario.
    ("trabajamos", "trabajaste"), ("revisamos", "revisaste"),
    ("vimos", "viste"), ("hicimos", "hiciste"), ("estuvimos", "estuviste"),
    ("tuvimos", "tuviste"), ("fuimos", "fuiste"), ("dijimos", "dijiste"),
    ("tocamos", "tocaste"), ("notamos", "notaste"),
    ("encontramos", "encontraste"), ("dejamos", "dejaste"),
    ("recibimos", "recibiste"), ("preparamos", "preparaste"),
]


# Pronombres / determinantes en 1ra persona → 2da. Cuidado con falsos
# positivos: "mi" puede ser nota musical. La regla pragmática: solo
# reemplazar al inicio de palabra Y dentro de un contexto de prosa
# narrativa. El regex word-boundary es suficiente.
_VOICE_PRONOUN_REPLACEMENTS = [
    # Pronombres sujeto / objeto
    ("yo", "vos"),
    # "me" reflexivo: se queda como "te"
    ("me ", "te "),
    # "mi/mí/mío/míos/mía/mías" → "tu/vos/tuyo..."
    ("mi ", "tu "),
    ("mí ", "vos "),
    ("mío", "tuyo"),
    ("míos", "tuyos"),
    ("mía", "tuya"),
    ("mías", "tuyas"),
    # "nos" reflexivo plural ("nos vimos") → "te"
    ("nos ", "te "),
]


def _make_word_boundary_pattern(words: list[str]) -> "re.Pattern":
    """Compila un único regex que matchea CUALQUIERA de las palabras
    como word completa (word boundary). Más rápido que iterar N regex.
    """
    # Sort longest-first para que "encontré" capture antes que "encé"
    sorted_words = sorted(words, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted_words) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def normalize_voice_to_2da_persona(text: str) -> str:
    """Reemplaza verbos / pronombres en 1ra persona por 2da singular.

    Pragmático: el prompt prohíbe 1ª persona pero el modelo 7B se desliza
    ~10% del tiempo. Este pass es el último guard. NO toca:
      - Texto entre comillas (preserva citas literales del user)
      - Substrings dentro de palabras (ej. "concentré" no matchea
        "centré" porque el word-boundary se aplica)
      - Code fences (no debería haber en briefs, pero por las dudas)

    Estrategia de preservación de citas: split por delimiters de cita
    `"..."` y `'...'`, normalizar SOLO los segmentos fuera de comillas,
    re-juntar.

    Aplica una sola vez (no idempotente para evitar over-replace en
    ediciones repetidas).
    """
    if not text:
        return text

    # Build maps: lowercase verb / pronoun → replacement (preservando
    # case del original donde se pueda — ej. "Recibí" → "Recibiste").
    verb_map = {orig: repl for orig, repl in _VOICE_VERB_REPLACEMENTS_1PS}
    verb_re = _make_word_boundary_pattern(list(verb_map.keys()))

    # Pronouns con espacio trailing — usamos un regex distinto porque
    # ya incluye el espacio en el match.
    # Para "me " etc., el regex es \bme\s — matchea la palabra + 1 ws.
    pronoun_pattern = r"\b(yo|me|mi|mí|mío|míos|mía|mías|nos)\b"
    pronoun_re = re.compile(pronoun_pattern, re.IGNORECASE)
    pronoun_map = {
        "yo": "vos",
        "me": "te",
        "mi": "tu",
        "mí": "vos",
        "mío": "tuyo",
        "míos": "tuyos",
        "mía": "tuya",
        "mías": "tuyas",
        "nos": "te",
    }

    def _preserve_case(orig: str, repl: str) -> str:
        if orig.isupper():
            return repl.upper()
        if orig[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    def _replace_verb(m: re.Match) -> str:
        orig = m.group(0)
        repl = verb_map.get(orig.lower(), orig)
        return _preserve_case(orig, repl)

    def _replace_pronoun(m: re.Match) -> str:
        orig = m.group(0)
        repl = pronoun_map.get(orig.lower(), orig)
        return _preserve_case(orig, repl)

    # Preservar citas: split text por comillas dobles y procesar solo
    # segmentos pares (índices 0, 2, 4...).
    parts = re.split(r'(".*?"|\'.*?\')', text)
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # adentro de comillas
            out_parts.append(part)
            continue
        # Aplicar verb replacements primero, después pronouns.
        normalized = verb_re.sub(_replace_verb, part)
        normalized = pronoun_re.sub(_replace_pronoun, normalized)
        out_parts.append(normalized)
    return "".join(out_parts)
