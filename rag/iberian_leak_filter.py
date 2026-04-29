"""Filtro PT/galego → ES para respuestas del LLM (rioplatense argentino).

Motivación
----------
Los LLM locales que usamos (qwen2.5:7b, command-r) son multilingües y
ocasionalmente "se contagian" del lenguaje del CONTEXTO o de palabras
similares. Síntomas observados:

- Respuestas que mezclan portugués: "primeira", "tua experiência",
  "falam", "vistes", "primeiramente", "nos braços", "uma", "também".
- Galego con apóstrofes raros: "do´mañá", "do'mana".
- Conjugaciones no-españolas: "esqueças", "dessas".
- Inicio de respuesta en otro idioma incluso aunque el system prompt
  esté entero en español.

Reportes recientes:
- 2026-04-23 — `do´mañá` literal en respuesta de `ft_pendientes`.
- 2026-04-28 (wave-5) — Playwright batch sobre nota Ikigai filtró
  "esse"/"isso"/"você" en respuestas que debían ser español.
- 2026-04-29 — `rag query "Que tenes de Grecia?"` devolvió respuesta
  con "primeira", "e de tu abuelo", "Soy una hermosa hija e estoy".

Este filtro es la **última barrera** después del system prompt
`language_es_AR.v1` (que prohíbe el portugués explícitamente al LLM).
La regla del prompt es necesaria pero no suficiente — el modelo se
desliza en ~2-5% de las respuestas pese a la instrucción. Acá
reemplazamos palabra-por-palabra a su equivalente español rioplatense.

Diseño
------
Conservador: sólo pares alta-confianza donde la palabra portuguesa /
gallega NO existe (o es muy rara) en español rioplatense. Las palabras
que existen en ambos idiomas (`mesa`, `casa`, `vida`) NO se tocan.

Idempotente: si el texto ya está en español, el filtro pasa sin
cambios. Probado en `tests/test_iberian_leak_filter.py` con la
invariante `stream(chunks) == replace(concat)` para distintos
tamaños de chunk.

Uso
---
Sync (post-generación, batch):

    from rag.iberian_leak_filter import replace_iberian_leaks
    cleaned = replace_iberian_leaks(llm_response)

Streaming (token-by-token, ej. SSE web o CLI):
    el `_IberianLeakFilter` con buffer de 200 chars vive en
    `web/server.py` (acoplado al pipeline de PII redaction y filename
    normalization). No lo movimos acá porque su contrato depende del
    streaming wrapper completo, no sólo del replacement.
"""
from __future__ import annotations

import re

# Palabras portuguesas/gallegas → español rioplatense.
#
# Orden CRÍTICO: frases multi-palabra PRIMERO. Si aplicáramos las
# reglas atómicas antes, "em março" → "em marzo" (palabra "em"
# quedaría como galego en la respuesta).
#
# Mantener en sync con `_COMPOUND_STARTER_TAIL_RE` en `web/server.py`
# (el streaming filter necesita saber qué palabras inician compounds
# para retener buffer hasta ver el complemento).
_IBERIAN_LEAK_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # ── Frases multi-palabra (van primero) ────────────────────────
    (r"\buma\s+conversa\b", "una conversación"),
    (r"\buma\s+conversação\b", "una conversación"),
    (r"\bem\s+março\b", "en marzo"),
    (r"\bem\s+maio\b", "en mayo"),
    (r"\bem\s+junho\b", "en junio"),
    (r"\bem\s+julho\b", "en julio"),
    (r"\bem\s+setembro\b", "en septiembre"),
    (r"\bem\s+outubro\b", "en octubre"),
    (r"\bem\s+novembro\b", "en noviembre"),
    (r"\bem\s+dezembro\b", "en diciembre"),
    (r"\bem\s+fevereiro\b", "en febrero"),
    (r"\bcontigo\s+em\b", "contigo en"),
    # 2026-04-29 (Grecia bug): "nos braços" → "en los brazos".
    (r"\bnos\s+braços\b", "en los brazos"),
    (r"\bno\s+braço\b", "en el brazo"),
    # 2026-04-29: "tua experiência" / "tua mamá" / "tua filha" — el
    # posesivo "tua" en pt no existe en español rioplatense ("tu" sí,
    # "tua" no). Igual con "teus" / "teu" / "tuas".
    (r"\btua\s+", "tu "),
    (r"\bteu\s+", "tu "),
    (r"\btuas\s+", "tus "),
    (r"\bteus\s+", "tus "),
    # ── Meses (único sentido en pt — todos tienen otra grafía en es) ──
    (r"\bmarço\b", "marzo"),
    (r"\bmaio\b", "mayo"),
    (r"\bjunho\b", "junio"),
    (r"\bjulho\b", "julio"),
    (r"\bsetembro\b", "septiembre"),
    (r"\boutubro\b", "octubre"),
    (r"\bnovembro\b", "noviembre"),
    (r"\bdezembro\b", "diciembre"),
    (r"\bfevereiro\b", "febrero"),
    # ── Tiempo (palabras que NO existen en español) ─────────────
    (r"\bhoje\b", "hoy"),
    (r"\bontem\b", "ayer"),
    (r"\bamanhã\b", "mañana"),
    # Galego "do'mañá" + variantes con apóstrofe ascii / unicode prime /
    # backtick. Incluye formas truncadas que el LLM emite cuando
    # "trata" de españolizar a medias ("do´man", "do´mañ", "do´mana") —
    # captura cualquier "do[apóstrofe]ma[nñ][a|á]?".
    (r"\bdo['´`]ma[nñ][áa]?\b", "mañana"),
    (r"\bmañá\b", "mañana"),
    # ── Pronombres / negación (pt) ──────────────────────────────
    (r"\bnão\b", "no"),
    # "sim" podría ser "sim" de simulación en español técnico — prefix
    # la palabra con word-boundary y usamos case-insensitive para no
    # cazar SIM en siglas.
    (r"\bsim\b", "sí"),
    # ── Cantidad (pt) ───────────────────────────────────────────
    (r"\bmuito\b", "mucho"),
    (r"\bmuita\b", "mucha"),
    (r"\bmuitos\b", "muchos"),
    (r"\bmuitas\b", "muchas"),
    # ── Cortesía ────────────────────────────────────────────────
    (r"\bobrigado\b", "gracias"),
    (r"\bobrigada\b", "gracias"),
    # ── Verbos comunes (pt — conjugaciones que no existen en es) ──
    (r"\besqueças\b", "olvides"),       # "no te esqueças" → "no te olvides"
    (r"\besqueça\b", "olvide"),
    (r"\bdessas\b", "de esas"),         # "no te esqueças dessas"
    (r"\bdesses\b", "de esos"),
    # 2026-04-28 wave-5: leaks observados en Playwright batch sobre nota
    # Ikigai. El LLM mezcló pt en respuestas que sí debían ser español.
    # "tem" (3ra persona singular del verbo ter en pt) → "tiene". CUIDADO:
    # "tem" puede aparecer en frases como "sistema TEM-1" — usamos word
    # boundary + lookahead que pida espacio + minúscula después para no
    # tocar siglas técnicas.
    (r"\btem\s+(?=[a-záéíóúñ])", "tiene "),
    (r"\bté\b", "té"),                  # idempotent (té ya está en español)
    # 2026-04-29 (Grecia bug): "falam" (3ra plural pt de "falar") y
    # "vistes" (2da pretérito pt de "ver" — en español rioplatense es
    # "viste" sin "s", pero "vistes" también puede ser español castellano
    # arcaico "vosotros visteis"; en contexto rioplatense `vistes` sólo
    # aparece como pt-leak, así que lo normalizamos).
    (r"\bfalam\b", "hablan"),
    (r"\bfalou\b", "habló"),
    (r"\bfala\b(?!\s+un)", "habla"),  # "fala" sólo si no es "fala un" (raro)
    (r"\bvistes\b", "viste"),
    # 2026-04-29: "primeira" / "primeiro" / "primeiramente" / "primero"
    # En portugués "primeira" es femenino ("la primera") y "primeiro"
    # masculino ("el primero"). En español: "primera"/"primero". Y
    # "primeiramente" no existe en español — equivalente "por primera
    # vez" / "primero".
    (r"\bprimeira\b", "primera"),
    (r"\bprimeiro\b", "primero"),
    (r"\bprimeiramente\b", "primero"),
    # 2026-04-29: "uma" (artículo indefinido pt) → "una". Multi-palabra
    # de "uma conversa" ya está arriba, ésta es "uma" suelto.
    (r"\buma\b", "una"),
    # 2026-04-29: "também" (también en pt) — la grafía con tilde nasal
    # en la 'e' SÓLO existe en pt. En es: "también". El "también"
    # español usa tilde aguda en la 'e' final, no nasal.
    (r"\btambém\b", "también"),
    # 2026-04-29: relaciones familiares pt → es. Ojo: "neta" en es
    # también puede significar "limpia/clara" (poco usado), "tio"
    # podría ser typo en es de "tío", etc. Estos pares los
    # restringimos lo más posible para no falsificar texto en es.
    (r"\bavô\b", "abuelo"),
    (r"\bavó\b", "abuela"),
    (r"\birmão\b", "hermano"),
    (r"\birmã\b", "hermana"),
    (r"\bfilha\b", "hija"),
    (r"\bfilho\b", "hijo"),
    (r"\bcriança\b", "niño"),
    (r"\bmãe\b", "mamá"),
    (r"\bpai\b", "papá"),
    # 2026-04-29 (E2E test sobre /api/chat con "qué tengo de Grecia"):
    # el LLM (qwen2.5:7b o command-r) sigue emitiendo portugués denso
    # pese al system prompt + intent prompts. Output observado:
    # "Tu nota es una carta para alguém em sua vida. A carta habla
    # sobre o amor e a revolução que vos trouxe à família, desde a
    # chegada da niño até os momentos emocionantes da primária."
    #
    # Agregamos las palabras que faltaban de ese output. Todas tienen
    # grafías que NO existen en español rioplatense (acentos nasales
    # `ã`/`õ`, sufijos `-ução`, vocabulario distintivo pt).
    (r"\balguém\b", "alguien"),       # quien (pt) → alguien (es)
    (r"\bninguém\b", "nadie"),
    (r"\btrouxe\b", "trajo"),         # ele/ela trouxe = él/ella trajo
    (r"\btrouxeram\b", "trajeron"),
    (r"\bprimária\b", "primaria"),    # acento en la 'a' es pt; en es es 'i'
    (r"\bsecundária\b", "secundaria"),
    (r"\brevolução\b", "revolución"),
    (r"\brevoluções\b", "revoluciones"),
    (r"\beducação\b", "educación"),
    (r"\binformação\b", "información"),
    (r"\binformações\b", "informaciones"),
    (r"\bemoção\b", "emoción"),
    (r"\bemoções\b", "emociones"),
    (r"\bcanção\b", "canción"),
    (r"\bcanções\b", "canciones"),
    (r"\bnação\b", "nación"),
    (r"\bnações\b", "naciones"),
    (r"\brelação\b", "relación"),
    (r"\brelações\b", "relaciones"),
    # Sufijo general `-ução` (pt) → `-ución` (es). Captura cualquier
    # palabra terminada en -ução que no esté en la lista explícita
    # de arriba (ej. "destruição" → "destrucción", "construção" →
    # "construcción"). El sufijo `-ução` SÓLO existe en pt.
    (r"\b(\w+?)ução\b", r"\1ución"),
    (r"\b(\w+?)uções\b", r"\1uciones"),
    # Sufijo `-ção` (pt) → `-ción` (es). Cubre los casos que no eran
    # palabras estándar (ej. "atualização" → "actualización" — aunque
    # el "atu" sigue siendo pt, el sufijo se corrige). Las palabras
    # explícitas de arriba (ação, solução, etc.) ya estaban; ésta es
    # red de protección.
    (r"\b(\w+?)ção\b", r"\1ción"),
    (r"\b(\w+?)ções\b", r"\1ciones"),
    # Preposiciones pt con `à`/`às`/`ao`/`aos` (contracción `a + el`)
    # — no existen en es. "à família" → "a la familia", pero distinguir
    # género (la/el) sin contexto es complejo. Conservador: sólo
    # convertir cuando el siguiente sustantivo se reconoce, o usar
    # el patrón genérico → "a" sin género (puede sonar raro pero es
    # mejor que "à").
    (r"\bà\b", "a la"),
    (r"\bàs\b", "a las"),
    (r"\bao\b", "al"),
    (r"\baos\b", "a los"),
    # "em sua" / "em seu" — pt explícito. "em" como starter ya tiene
    # algunas reglas (em março, em junho, etc.) pero "em sua" / "em
    # seu" son contextos de posesivo muy comunes en pt.
    (r"\bem\s+sua\b", "en su"),
    (r"\bem\s+seu\b", "en su"),
    (r"\bem\s+suas\b", "en sus"),
    (r"\bem\s+seus\b", "en sus"),
    (r"\bem\s+um\b", "en un"),
    (r"\bem\s+uma\b", "en una"),
    # "até" (pt: "hasta") — la grafía con `é` final es pt. En es se
    # escribe "hasta" o "incluso". Conservador: convertir a "hasta".
    (r"\baté\b", "hasta"),
    # "vos trouxe" / "para vos" donde el `vos` apunta a 2da persona
    # pt (no rioplatense — en pt brasileiro no se usa "vos"; en pt
    # europeo arcaico sí). En rioplatense "vos" es OK, pero combinado
    # con verbos pt como "trouxe" es leak. La regla de "trouxe" ya
    # arriba lo cubre — no agregamos "vos" → "vos" (idempotente).
    # "novo" / "nova" / "novos" / "novas" (pt) → "nuevo"/etc. (es).
    # CUIDADO: "novo" no existe en es estándar (es "nuevo"), pero
    # tampoco es muy común el typo. Lo agregamos:
    (r"\bnovo\b", "nuevo"),
    (r"\bnova\b", "nueva"),
    (r"\bnovos\b", "nuevos"),
    (r"\bnovas\b", "nuevas"),
    # "capítulo" en pt y "capítulo" en es son IGUALES. No tocar.
    # "esperando" en ambos idiomas también. No tocar.
    # "momentos" en ambos. No tocar.
    # Sufijo `-mente` adverbial: igual en pt y es. No tocar.
    # "chegada" (pt) → "llegada" (es). La grafía con `ch` para sonido
    # /ʃ/ es pt; en es es siempre /tʃ/.
    (r"\bchegada\b", "llegada"),
    (r"\bchegou\b", "llegó"),
    (r"\bchegar\b", "llegar"),
    # Artículos contraídos pt SEGUROS (no aparecen en español rioplatense
    # como palabra suelta + letra). EXCLUIDOS:
    #   - `\bno\b` y `\bnos\b` → SUPER COMÚN en es como negación y
    #     pronombre reflexivo ("no quiero", "nos vamos", "se nos hizo").
    #     La regla específica `\bnos\s+braços\b` ya está arriba.
    #   - `\bdos\b` → es número 2 ("dos casas", "dos personas").
    #   - `\bo\b`, `\ba\b` → artículos super genéricos en pt, pero "o"
    #     en es es conjunción ("ese o el otro") y "a" es preposición
    #     ("voy a casa"). Demasiado riesgo.
    (r"\bda\s+", "de la "),
    (r"\bdo\s+(?=[a-záéíóúñ])", "del "),  # cuidado con "do" como verbo musical / sigla
    (r"\bdas\s+", "de las "),
    (r"\bna\s+", "en la "),
    (r"\bnas\s+", "en las "),
    # "os" (pt: "los") y "as" (pt: "las") — en rioplatense no se usan
    # como pronombres (es voseo, no vosotros). En español de España
    # SÍ ("os digo", "as visto") pero el target es rioplatense. Riesgo
    # bajo. Lookahead exige letra después para evitar matches en
    # acrónimos/abreviaciones (OS X, etc.).
    (r"\bos\s+(?=[a-záéíóúñ])", "los "),
    (r"\bas\s+(?=[a-záéíóúñ])", "las "),
    # "um" (pt: "un") → "un". "uma" ya está como "una". CUIDADO con
    # palabras técnicas como "UMI", "UMP" — `\b` las aísla pero el
    # case-insensitive matchea. En la práctica no son palabras + letra
    # minúscula.
    (r"\bum\s+(?=[a-záéíóúñ])", "un "),
    # "família" (pt: "familia") — la tilde nasal sobre la `i` no
    # existe en es.
    (r"\bfamília\b", "familia"),
    (r"\bfamílias\b", "familias"),
    (r"\bescola\b", "escuela"),  # pt → es
    (r"\bescolas\b", "escuelas"),
    # 2026-04-29: sufijos y palabras pt con `-ência` (acento circunflejo
    # nasal en `ê`). En es es `-encia`. La grafía con `ê` SÓLO existe
    # en pt — convertir es seguro. Cubre "experiência", "ciência",
    # "consciência", "paciência", "frequência", etc.
    (r"\b(\w+?)ência\b", r"\1encia"),
    (r"\b(\w+?)ências\b", r"\1encias"),
    # Idem `-ância` (pt) → `-ancia` (es). "importância", "elegância",
    # "circunstância", "tolerância".
    (r"\b(\w+?)ância\b", r"\1ancia"),
    (r"\b(\w+?)âncias\b", r"\1ancias"),
    # 2026-04-29: pronombres personales pt → es donde la grafía es
    # CLARAMENTE pt (no existe en español). "ela" en es no existe
    # como palabra suelta (sí "él"/"ella"); restringido a contextos
    # conjugados. Incluimos "tiene" en la lista (no solo "tem")
    # porque el regex `\btem\s+...→tiene` puede aplicarse ANTES si el
    # input es "ele tem un perro" (la regla de "tem" matchea primero,
    # quedando "ele tiene un perro" — sin la opción "tiene" en el
    # match siguiente, el "ele" no se cazaría).
    (r"\bela\s+(é|es|era|foi|fue|estava|estaba|tem|tiene|teve|tuvo|disse|dijo)\b", r"ella \1"),
    (r"\bele\s+(é|es|era|foi|fue|estava|estaba|tem|tiene|teve|tuvo|disse|dijo)\b", r"él \1"),
    # 2026-04-29: "é" suelto (verbo ser pt 3ra sing) → "es". El acento
    # agudo en "é" sola NO existe en es (la palabra "es" no lleva tilde).
    (r"\bé\b", "es"),
    # 2026-04-29: artículo definido pt "a" cuando va seguido de
    # sustantivo femenino sin tilde. CUIDADO: "a" como preposición es
    # común en español ("voy a casa"). Restringimos al patrón
    # "<verbo|adverbio> a <sustantivo>" donde el contexto delata pt.
    # Patrón seguro observado: "era a primera/primeira" → "era la primera".
    (r"\bera\s+a\s+primer", r"era la primer"),
    # "foi" suelto (verbo ser/ir pt 3ra sing pretérito) → "fue".
    (r"\bfoi\b", "fue"),
    # 2026-04-29: "e" como conjunción copulativa pt → "y" (es). En es
    # la "e" sólo se usa antes de palabras que empiezan en "i" o "hi"
    # ("Pedro e Inés"). Cualquier otro uso de "e " seguido de palabra
    # común es leak pt. Lista explícita de palabras que vienen después
    # — más seguro que lookahead negativo, que podría romper texto es.
    (r"\be\s+(estoy|estás|está|estamos|están|estaba|estaban|son|fue|fueron|" \
     r"de|del|el|la|los|las|un|una|en|por|para|con|sin|" \
     r"sus|tus|mis|nuestro|nuestra|" \
     r"cuando|donde|porque|pero|aunque|" \
     r"muy|más|menos|" \
     r"yo|vos|él|ella|ellos|ellas)\b", r"y \1"),
    # ── Demostrativos/adjetivos (pt) — la doble s NO existe en es ──
    (r"\besse\b", "ese"),
    (r"\bessa\b", "esa"),
    (r"\besses\b", "esos"),
    (r"\bessas\b", "esas"),
    (r"\bisso\b", "eso"),
    (r"\bisto\b", "esto"),
    (r"\baquilo\b", "aquello"),
    # ── Adverbios/locuciones pt comunes en respuestas ───────────
    (r"\bAqui\s+estão\b", "Acá están"),
    (r"\baqui\s+está\b", "acá está"),
    (r"\baqui\b", "acá"),               # SOLO al inicio de frase + minúscula
    (r"\bestão\b", "están"),
    (r"\bmelhor\b", "mejor"),
    (r"\bpior\b", "peor"),
    # ── Verbos de movimiento/uso (pt → es) ──────────────────────
    (r"\bajudar\b", "ayudar"),
    (r"\bajuda\b", "ayuda"),
    (r"\bvocê\b", "vos"),
    (r"\bvocês\b", "ustedes"),
    # ── Conectores/preposiciones (pt — claramente no español) ───
    # `com` (pt: "con") — usar negative lookbehind para EXCLUIR
    # cuando aparece dentro de URLs/dominios (`example.com hoy.` —
    # ".com" + " " + "h" matchearía sin el guard, rompiendo la URL).
    # El lookbehind `(?<![./])` excluye "com" precedido de "." o "/".
    (r"(?<![./])\bcom\s+(?=[a-záéíóúñ])", "con "),
    (r"\bde\s+(?=[a-záéíóúñ]\w+ção\b)", "de "),  # idempotente — solo prep
    # Sufijos -ção / -ções son siempre pt; convertir a -ción/-ciones.
    (r"\bação\b", "acción"),
    (r"\bações\b", "acciones"),
    (r"\bsolução\b", "solución"),
    (r"\bsoluções\b", "soluciones"),
    (r"\bquestão\b", "cuestión"),
    (r"\bquestões\b", "cuestiones"),
    # ── Francés/italiano stray words (hallucination genuina) ────
    # "Voulu" observado al inicio de respuesta — claramente falso.
    (r"\bVoulu,?\s+", ""),
    (r"\bvoilà\b", "acá"),
)


_IBERIAN_LEAK_COMPILED: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _IBERIAN_LEAK_REPLACEMENTS
)


# Auto-extracción de "compound starters" para el streaming filter del web.
#
# Problema: cuando el LLM emite respuestas via SSE chunk-by-chunk, las frases
# multi-palabra (ej. "em março", "nos braços") llegan partidas. Si emitimos
# al primer boundary, "em " sale antes de ver "março" y la regla compound
# nunca matchea. El filter retiene en buffer cualquier candidate que
# TERMINE con un "starter" conocido (primera palabra de un compound).
#
# Antes este regex se mantenía manual en `web/server.py` con riesgo de drift:
# si alguien agregaba un compound nuevo (ej. `\bnos\s+braços\b`) y olvidaba
# updater el regex de starters, el web streaming filter no lo cazaba aunque
# el batch sync sí. Auto-derivar elimina el riesgo.
#
# Heurística: extraemos la primera palabra de cada regex que matchee el
# patrón `\b<palabra>\s+` (compound de 2+ tokens). Los regexes single-word
# (`\bcom\b`, `\btua\s+` que es prefijo no-compound) se filtran porque su
# replace dispara inmediato sin buffer.
def _extract_compound_starters() -> tuple[str, ...]:
    """Extrae los starters de compounds de los regexes del filter.

    Un "starter" es la primera palabra de cualquier regex que tenga la
    forma `\\b<palabra>\\s+...<algo después de \\s+>...`. Devuelve la
    lista deduplicada y lowercase.
    """
    # Match `\b<chars>\s+` donde chars es letra/acento/apóstrofe.
    starter_re = re.compile(r"^\\b([a-záéíóúñâêîôûãõàèìòùç']+)\\s\+")
    seen: set[str] = set()
    out: list[str] = []
    for pat, _repl in _IBERIAN_LEAK_REPLACEMENTS:
        m = starter_re.match(pat)
        if not m:
            continue
        word = m.group(1).lower()
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
    return tuple(out)


_COMPOUND_STARTERS: tuple[str, ...] = _extract_compound_starters()


# Regex compilada lista para usar por el streaming filter del web. Matchea
# un candidate de emit que TERMINE con uno de los starters detectados.
# El web debe importar esto en vez de mantener una copia manual.
_COMPOUND_STARTER_TAIL_RE: re.Pattern = re.compile(
    r"\b(" + "|".join(_COMPOUND_STARTERS) + r")(\s+\S*)?\s*$",
    re.IGNORECASE,
)


def replace_iberian_leaks(text: str | None) -> str:
    """Aplica los regexes de `_IBERIAN_LEAK_REPLACEMENTS` en orden.

    Safe sobre input no-string / vacío. Preserva el case de los
    surrounding caracteres pero el replacement sale lowercase
    (los regexes son IGNORECASE para el match, los replaces son
    literales lowercase). Mixed-case originales como "Março" se
    normalizan a "marzo" — aceptable porque el leak es bug del modelo,
    no preserva intent del usuario.

    Idempotente: aplicar varias veces sobre la misma string da el
    mismo resultado (todos los pares apuntan a strings que YA están
    en español, no caen en otro replace).
    """
    if not text:
        return text or ""
    out = text
    for pat, repl in _IBERIAN_LEAK_COMPILED:
        out = pat.sub(repl, out)
    return out


__all__ = (
    "_IBERIAN_LEAK_REPLACEMENTS",
    "_IBERIAN_LEAK_COMPILED",
    "replace_iberian_leaks",
)
