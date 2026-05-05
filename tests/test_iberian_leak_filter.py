"""Tests para `_replace_iberian_leaks` + `_IberianLeakFilter` (2026-04-23).

Motivación (scratch_eval 2026-04-23 run):
  - `ft_pendientes` devolvió "Llamar al dentista, a la hora de 10h
    do´mañá." — `do´mañá` es galego, palabra no-española que el LLM
    qwen2.5:7b dejó pasar pese a REGLA 0 endurecida.
  - `wa_charla_maria` (pre-Fix #5 del commit anterior) devolvía
    respuestas mitad portugués: "uma conversa em março", "contigo em".

REGLA 0 en el prompt es necesaria pero no suficiente — el modelo se
contagia del lenguaje del CONTEXTO (WhatsApp con contactos brasileros,
notas scrapeadas de blogs en portugués). Este filter es la última
barrera: reemplaza palabra-por-palabra. Conservador por diseño — sólo
pares alta-confianza donde la palabra portuguesa/gallega NO existe (o es
muy rara) en español rioplatense.
"""
from __future__ import annotations


def _import_helpers():
    from web.server import (
        _replace_iberian_leaks,
        _IberianLeakFilter,
    )
    return _replace_iberian_leaks, _IberianLeakFilter


# ── _replace_iberian_leaks: simple reemplazos ────────────────────────


def test_meses_en_portugues_se_traducen():
    replace, _ = _import_helpers()
    assert replace("em março") == "en marzo"
    assert replace("maio") == "mayo"
    assert replace("em junho próximo") == "en junio próximo"
    assert replace("fevereiro 2026") == "febrero 2026"


def test_tiempo_hoje_ontem_amanha():
    replace, _ = _import_helpers()
    assert replace("hoje") == "hoy"
    assert replace("ontem") == "ayer"
    assert replace("amanhã") == "mañana"


def test_galego_do_mana_variantes_apostrofe():
    """El galego emite la palabra compound con apóstrofe ASCII,
    unicode prime (´) o backtick. Todas deben normalizar a 'mañana'.
    También formas truncadas que el LLM emite cuando intenta
    españolizar a medias ('do´man' dropping 'ñá', 'do´mañ' partial).
    """
    replace, _ = _import_helpers()
    assert replace("a las 10h do'mañá") == "a las 10h mañana"
    assert replace("do´mañá") == "mañana"      # prime unicode
    assert replace("do`mañá") == "mañana"      # backtick
    # Formas truncadas (LLM emitiendo parcialmente).
    assert replace("10h do´man.") == "10h mañana."
    assert replace("do´mañ") == "mañana"
    assert replace("do'mana") == "mañana"
    # "mañá" suelto también.
    assert replace("ate mañá") == "ate mañana"


def test_verbos_portugues_esquecas():
    """`no te esqueças dessas` (pt) → `no te olvides de esas` (es).
    El LLM a veces emite estas conjugaciones pt bajo contexto WA.
    """
    replace, _ = _import_helpers()
    assert replace("no te esqueças") == "no te olvides"
    assert replace("no te esqueças dessas") == "no te olvides de esas"
    assert replace("não te esqueças dessas!") == "no te olvides de esas!"


def test_negacion_y_afirmacion_portugues():
    replace, _ = _import_helpers()
    assert replace("não quiero") == "no quiero"
    assert replace("sim") == "sí"


def test_cantidades_muito_muita():
    replace, _ = _import_helpers()
    assert replace("muito bien") == "mucho bien"
    assert replace("muita gente") == "mucha gente"
    assert replace("muitos amigos") == "muchos amigos"


def test_obrigado_se_traduce():
    replace, _ = _import_helpers()
    assert replace("obrigado") == "gracias"
    assert replace("obrigada") == "gracias"


def test_frases_multi_palabra_se_traducen():
    replace, _ = _import_helpers()
    assert replace("uma conversa importante") == "una conversación importante"
    assert replace("uma conversação") == "una conversación"
    assert replace("hablé contigo em marzo") == "hablé contigo en marzo"


def test_texto_ya_en_espanol_no_se_toca():
    """Idempotencia sobre texto limpio. Este es el caso más común en
    producción (el filter pasa sobre ~todo el stream sin matches).
    """
    replace, _ = _import_helpers()
    clean = "Tenés un cumpleaños de Astor mañana. Llamar al dentista."
    assert replace(clean) == clean


def test_empty_string_y_none_safe():
    replace, _ = _import_helpers()
    assert replace("") == ""
    # None → passthrough (helper debe tolerar).
    assert replace(None) in (None, "")


def test_case_insensitive():
    """Los matches son case-insensitive (el LLM a veces Capitaliza inicio
    de oración). La replacement sale lowercase — es aceptable porque el
    leak es bug del modelo, no preserva intent del usuario.
    """
    replace, _ = _import_helpers()
    # "Março" al inicio → "marzo" (lowercase OK).
    assert "marzo" in replace("Março fue intenso")
    assert "hoy" in replace("Hoje te digo")


def test_word_boundary_no_rompe_palabras_compuestas():
    """`\\b` previene que `março` matchee dentro de otras palabras.
    Caso hipotético: no deberíamos tocar una hipotética palabra
    `submarço` (no real, pero sirve de test de aislamiento)."""
    replace, _ = _import_helpers()
    assert replace("submarço") == "submarço"
    assert replace("marzote") == "marzote"  # no-match


# ── _IberianLeakFilter: streaming ─────────────────────────────────────


def test_stream_passthrough_texto_limpio():
    _, Filter = _import_helpers()
    f = Filter()
    out = f.feed("hola ")
    out += f.feed("qué tal ")
    out += f.feed("todo bien")
    out += f.flush()
    assert out == "hola qué tal todo bien"


def test_stream_reemplaza_hoje_en_un_solo_chunk():
    _, Filter = _import_helpers()
    f = Filter()
    out = f.feed("hoje tengo reunión ")
    out += f.flush()
    assert "hoy" in out
    assert "hoje" not in out


def test_stream_reemplaza_palabra_partida_entre_chunks():
    """Caso crítico: el token `hoj` llega en un chunk y `e ` en el
    siguiente. El filter tiene que buffear `hoj` hasta ver el boundary.
    """
    _, Filter = _import_helpers()
    f = Filter()
    out = f.feed("hoj")    # Sin boundary, queda en buffer.
    # Probablemente vacío (nada emitido porque no hay word boundary).
    out += f.feed("e ")    # Ahora sí: `hoje ` completo.
    out += f.flush()
    assert "hoy" in out
    assert "hoje" not in out


def test_stream_maneja_varias_palabras_por_chunk():
    _, Filter = _import_helpers()
    f = Filter()
    out = f.feed("em março te veo uma conversa tranquila ")
    out += f.flush()
    assert "en marzo" in out
    assert "una conversación" in out
    assert "em março" not in out
    assert "uma conversa" not in out


def test_stream_flush_sin_boundary_final():
    """Si el stream termina sin un boundary final (ej. el LLM emitió
    `marzo` como último token sin espacio), flush() drena el buffer
    aplicando replace.
    """
    _, Filter = _import_helpers()
    f = Filter()
    out = f.feed("hola ")
    out += f.feed("março")  # último token, sin boundary trailing.
    # En este punto el feed probablemente no emitió todavía.
    out += f.flush()
    assert out.endswith("marzo")


def test_stream_empty_chunk_es_noop():
    _, Filter = _import_helpers()
    f = Filter()
    assert f.feed("") == ""
    assert f.feed(None) == ""  # type: ignore[arg-type]
    assert f.flush() == ""


def test_stream_forced_emit_en_buffer_grande():
    """Si el modelo emite un token gigante sin espacios (caso raro),
    el filter no debe quedarse colgado indefinidamente — forza emit
    cuando el buffer supera _MAX_HOLD."""
    _, Filter = _import_helpers()
    f = Filter()
    huge = "a" * 300  # > 200 _MAX_HOLD
    out = f.feed(huge)
    out += f.flush()
    assert out == huge  # passthrough sin match


def test_stream_total_output_matches_replace_over_concat():
    """Invariante: stream(chunk1 + chunk2 + ...) == replace(concat).
    Cualquier partition del input tiene que producir el mismo output
    que una sola llamada con todo el texto.
    """
    replace, Filter = _import_helpers()
    text = (
        "em março hablamos de não sé qué "
        "uma conversa re tranquila contigo em junho "
        "obrigado por muito tiempo hoje ontem amanhã"
    )
    expected = replace(text)
    # Partición arbitraria.
    for chunk_size in (1, 3, 7, 13, 50):
        f = Filter()
        out = ""
        for i in range(0, len(text), chunk_size):
            out += f.feed(text[i:i + chunk_size])
        out += f.flush()
        assert out == expected, (
            f"chunk_size={chunk_size}: got={out!r} expected={expected!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# 2026-04-29: tests para las palabras nuevas agregadas al filter después
# del bug "Que tenes de Grecia?" — respuesta del LLM con "primeira",
# "tua", "falam", "vistes", "primeiramente", "nos braços". Estas reglas
# son seguras (las grafías no existen en español rioplatense) — los
# tests las fijan como spec contra futuras refactorizaciones.
# ─────────────────────────────────────────────────────────────────────


def test_grecia_bug_repro_completo():
    """El reporte original: respuesta mezclando pt/galego en una nota
    personal. Después del filter debe quedar 100% es rioplatense.
    """
    replace, _ = _import_helpers()
    pt_leak = (
        "A primera vista la tua experiência con tu hija parece falam de algo "
        "grande. Estabas nos braços de tu mamá. Primeiramente, ela era a "
        "primeira filha. Soy una hermosa hija e estoy muy agradecido."
    )
    out = replace(pt_leak)
    # Palabras pt prohibidas — todas tienen que haber desaparecido.
    for forbidden in [
        "primeira", "primeiramente", "tua experi", "falam", "nos braços",
        "ela era", "e estoy",
    ]:
        assert forbidden not in out.lower(), f"leak {forbidden!r} sigue en {out!r}"
    # Y las equivalentes es tienen que estar.
    for required in ["primera", "primero", "hablan", "en los brazos", "ella era", "y estoy"]:
        assert required in out.lower(), f"falta {required!r} en {out!r}"


def test_primeira_y_primeiramente():
    replace, _ = _import_helpers()
    assert replace("a primeira filha") == "a primera hija"
    assert replace("o primeiro día") == "o primero día"  # "o" es pt para "el", no la tocamos por ahora
    assert "primero" in replace("Primeiramente, te digo")


def test_falam_falou_fala():
    replace, _ = _import_helpers()
    assert replace("ellos falam de eso") == "ellos hablan de eso"
    assert replace("Maria falou ayer") == "Maria habló ayer"


def test_vistes_pt_a_viste_es():
    replace, _ = _import_helpers()
    assert replace("vistes la nota?") == "viste la nota?"


def test_uma_y_tambem():
    replace, _ = _import_helpers()
    assert replace("uma persona") == "una persona"
    assert replace("também voy") == "también voy"


def test_posesivos_tua_teu_tuas_teus():
    replace, _ = _import_helpers()
    assert replace("tua experiência") == "tu experiencia"
    assert replace("teu hermano") == "tu hermano"
    assert replace("tuas notas") == "tus notas"
    assert replace("teus libros") == "tus libros"


def test_nos_bracos_y_no_braco():
    replace, _ = _import_helpers()
    assert replace("nos braços") == "en los brazos"
    assert replace("no braço") == "en el brazo"


def test_familia_pt_a_es():
    replace, _ = _import_helpers()
    assert replace("avô") == "abuelo"
    assert replace("avó") == "abuela"
    assert replace("irmão") == "hermano"
    assert replace("irmã") == "hermana"
    assert replace("filha") == "hija"
    assert replace("filho") == "hijo"
    assert replace("mãe") == "mamá"
    assert replace("pai") == "papá"


def test_sufijos_encia_y_ancia():
    """`-ência` (pt) → `-encia` (es). La grafía con `ê` SOLO existe en
    pt — convertir es seguro. Cubre experiência, ciência, paciência,
    consciência, frequência en una sola regla.
    """
    replace, _ = _import_helpers()
    assert replace("experiência única") == "experiencia única"
    assert replace("ciência aplicada") == "ciencia aplicada"
    assert replace("paciência infinita") == "paciencia infinita"
    assert replace("consciência plena") == "consciencia plena"
    assert replace("importância grande") == "importancia grande"  # -ância
    assert replace("circunstâncias varias") == "circunstancias varias"


def test_ela_ele_con_verbo_conjugado():
    replace, _ = _import_helpers()
    assert replace("ela era una niña") == "ella era una niña"
    assert replace("ele tem un perro") == "él tiene un perro"  # "tem" también se filtra
    assert replace("ela disse algo") == "ella disse algo"
    # Sin verbo conjugado pt, no se toca.
    assert replace("ela aria") == "ela aria"  # no match — palabra es


def test_e_acento_y_foi():
    replace, _ = _import_helpers()
    assert replace("ele é mi amigo") == "él es mi amigo"
    assert replace("foi un día largo") == "fue un día largo"


def test_e_conjuncion_a_y():
    """`e` como conjunción copulativa pt → `y` (es). Lista explícita de
    palabras que vienen después para no romper texto en es válido
    (Pedro e Inés, amor único e infinito).
    """
    replace, _ = _import_helpers()
    assert replace("hija e estoy") == "hija y estoy"
    assert replace("mamá, e de tu abuelo") == "mamá, y de tu abuelo"
    assert replace("vine e cuando") == "vine y cuando"
    assert replace("yo e vos") == "yo y vos"  # 'vos' está en la lista
    # No tocar "e" antes de palabras que empiezan con "i" o "hi" (uso
    # legítimo en es para evitar cacofonía).
    assert replace("Pedro e Inés") == "Pedro e Inés"
    assert replace("amor único e infinito") == "amor único e infinito"


def test_url_no_se_rompe():
    """Bug pre-existente revelado al activar filter en CLI. El regex
    `com` matcheaba dentro de URLs (example.com hoy → example.con hoy).
    Fix con negative lookbehind `(?<![./])`.
    """
    replace, _ = _import_helpers()
    assert "https://example.com hoy" in replace("Visitá https://example.com hoy.")
    assert "github.com con detalles" in replace("docs en github.com con detalles")
    # Pero "com" pt suelto al inicio de oración SIGUE corrigiéndose.
    assert replace("com vos hablamos") == "con vos hablamos"


def test_idempotencia_corpus_completo():
    """Aplicar el filter 2 veces == aplicarlo 1 vez. Sin loops infinitos
    ni cambios sucesivos. Crítico porque el filter se llama en
    `render_response()` que puede dispararse varias veces sobre el
    mismo texto (citation_repaired, critique_changed re-renders).
    """
    replace, _ = _import_helpers()
    samples = [
        "em março hablamos uma conversa contigo em junho",
        "tua experiência com não muito tiempo hoje ontem amanhã",
        "ela disse a primeira filha de tu mamá nos braços",
        "Pedro e Inés vinieron a la fiesta",  # solo es, no debe tocar
    ]
    for s in samples:
        once = replace(s)
        twice = replace(once)
        assert once == twice, f"no idempotente: {s!r} → {once!r} → {twice!r}"


def test_leaks_2026_05_04_proyectos_query():
    """Smoke E2E del 2026-05-04: `rag query "qué proyectos tengo activos"`
    devolvió respuesta con `tá no folder`, `mencionou`, `los tus projetos`,
    `em diferentes`, `ou proyectos`, `detalhes lá`. Locked acá para que
    el filter no regrese a tener gaps en estas palabras.
    """
    replace, _ = _import_helpers()
    src = (
        'la nota tá no archivo. mencionou los tus projetos en em '
        "diferentes tipos ou proyectos. más detalhes lá."
    )
    out = replace(src)
    # Palabras pt que NO deben quedar en el output (con word boundary
    # para no caer en substrings como "tá" dentro de "está").
    import re
    for forbidden in ("tá", "mencionou", "projetos", "ou", "detalhes",
                      "lá"):
        assert not re.search(rf"\b{re.escape(forbidden)}\b", out), (
            f"{forbidden!r} no se filtró: {out!r}"
        )
    assert "los tus" not in out, f"'los tus' no se filtró: {out!r}"
    # Reemplazos correctos esperados:
    for expected in ("está", "mencionó", "proyectos", "detalles",
                     "allá"):
        assert expected in out, f"falta {expected!r} en {out!r}"


def test_negacion_genuina_no_se_corrompe():
    """Crítico: el `\\bno\\b` solo NO se reemplaza para no romper la
    negación española genuina ("no quiero", "el código no funciona").
    El leak residual `está no <noun>` se acepta — el riesgo de
    falsificar negación supera al beneficio.
    """
    replace, _ = _import_helpers()
    casos = (
        "No quiero ir, no me gusta esa idea.",
        "El código no funciona y no compila tampoco.",
        "Está, no sé qué decirte.",
        "la pelota no rueda — está, no funciona el motor",
    )
    for s in casos:
        out = replace(s)
        # `no` como palabra debe sobrevivir tal cual al menos una vez
        assert " no " in f" {out} " or out.startswith("No "), (
            f"negación genuina rota: {s!r} → {out!r}"
        )


def test_em_compound_no_pisa_em_substring():
    """Sanity: el regex `\\bem\\s+(?=[a-z])` NO debe tocar substrings
    como "tema", "iemma", "BMC-EM-1". Sólo `em` como palabra suelta
    seguida de letra minúscula.
    """
    replace, _ = _import_helpers()
    safe = (
        "el tema central del proyecto BMC-EM-1 es claro",
        "Sistema EMI tiene 3 módulos",
        "EM-2 es la versión actualizada",
    )
    for s in safe:
        assert replace(s) == s, f"`em` substring rompió: {s!r} → {replace(s)!r}"


def test_stream_corpus_expandido_2026_04_29():
    """Misma invariante que `test_stream_total_output_matches_replace_over_concat`
    pero usando el corpus extendido con las palabras nuevas. Cubre los
    starters auto-derivados (uma|em|contigo|nos|no|tua|teu|...).

    NO incluye "ela era a primer..." porque hay una limitación conocida
    del streaming filter cuando 2 compound starters son adyacentes (ej.
    "ela" + "era"): el filter retiene "ela" como starter, y al ver "era"
    también starter, los emite separados → la regla `\\bela\\s+era\\b`
    no matchea bajo `chunk_size=1`. El sync (post-generation) sí los caza.
    Ver `test_stream_limitacion_starters_adyacentes` abajo para el caso
    documentado.
    """
    replace, Filter = _import_helpers()
    text = (
        "A primera vista la tua experiência con tu hija parece falam de algo "
        "grande. Estabas nos braços de tu mamá. Primeiramente, fue a "
        "primeira filha de teu pai. Soy una hermosa hija e estoy muy "
        "agradecido. Pedro e Inés también vinieron e cuando lleguen vamos."
    )
    expected = replace(text)
    for chunk_size in (1, 2, 5, 11, 23, 100):
        f = Filter()
        out = ""
        for i in range(0, len(text), chunk_size):
            out += f.feed(text[i:i + chunk_size])
        out += f.flush()
        assert out == expected, (
            f"chunk_size={chunk_size}: got={out!r} expected={expected!r}"
        )


def test_stream_starters_adyacentes_arreglado():
    """Refactor 2026-04-29: el streaming filter ahora usa "safe-to-emit
    check" — verifica que aplicar el filter al buffer completo == a las
    partes separadas. Si difieren, un compound atraviesa el borde y
    retiene todo. Esto cubre starter chains (`ela era`, `ele tem`) que
    el algoritmo previo no manejaba bajo `chunk_size=1`.

    Antes de este refactor existía `test_stream_limitacion_starters_adyacentes`
    que DOCUMENTABA la limitación. Ahora la limitación está resuelta y
    este test verifica que el comportamiento esperado se mantiene
    (sync == streaming para cualquier chunk_size).
    """
    replace, Filter = _import_helpers()
    text = "ela era una niña."
    sync_output = replace(text)
    assert sync_output == "ella era una niña.", f"sync should fix: got {sync_output!r}"

    # Ahora streaming con chunk_size=1 SÍ caza "ela era" adyacente.
    f = Filter()
    out = ""
    for ch in text:
        out += f.feed(ch)
    out += f.flush()
    assert out == sync_output, (
        f"streaming chunk_size=1 debería matchear sync ahora: got {out!r}"
    )

    # Y con chunks grandes también, obviamente.
    f = Filter()
    out = f.feed(text) + f.flush()
    assert out == sync_output, (
        f"streaming chunk_size=large debería matchear sync: got {out!r}"
    )


def test_starters_auto_derivados_estan_completos():
    """Validación de la auto-derivación: para cada compound multi-palabra
    en `_IBERIAN_LEAK_REPLACEMENTS`, la primera palabra debe estar en
    `_COMPOUND_STARTERS`. Si alguien agrega un compound nuevo al regex
    y olvida que el starter se auto-deriva, este test confirma que se
    extrajo bien.
    """
    from rag.iberian_leak_filter import (
        _COMPOUND_STARTERS,
        _IBERIAN_LEAK_REPLACEMENTS,
    )
    import re as _re

    starter_re = _re.compile(r"^\\b([a-záéíóúñâêîôûãõàèìòùç']+)\\s\+")
    for pat, _repl in _IBERIAN_LEAK_REPLACEMENTS:
        m = starter_re.match(pat)
        if not m:
            continue
        word = m.group(1).lower()
        assert word in _COMPOUND_STARTERS, (
            f"compound starter {word!r} (de regex {pat!r}) no está en "
            f"_COMPOUND_STARTERS = {_COMPOUND_STARTERS!r}"
        )
