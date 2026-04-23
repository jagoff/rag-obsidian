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
