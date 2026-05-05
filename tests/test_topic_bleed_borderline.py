"""Regression tests for the borderline-band history-drop fix (2026-05-04).

Bug: el user reportó que en una sesión web hizo dos preguntas seguidas de
temas distintos:

    T-1: "dame el número de trámite de mi DNI"
    T  : "decime que ideas de tech app tengo?"

La respuesta T mezcló los dos temas: empezó con "Según tus notas, no hay
información específica sobre tu número de trámite DNI. Sin embargo, en la
nota Ideas.md, mencionaste una idea para crear una aplicación...".

Root cause: cosine similitud entre los dos turnos cae en la banda
borderline `[TOPIC_SHIFT_COSINE=0.32, REFORM_COSINE_HIGH)`. Antes del fix,
esa banda mantenía la history viva y disparaba `reformulate_query()` con
LLM helper, que mezclaba los dos topics en la query reescrita y además el
LLM final también veía los turnos previos en su prompt.

Fix: bajar `REFORM_COSINE_HIGH` 0.50 → 0.40 + en la banda residual
[0.32, 0.40), si la query NO matchea el regex de follow-up
(`_looks_like_followup`), tratar como topic-shift débil y dropear history
(igual que sub-0.32 cosine). Esto elimina el bleed tanto en retrieve
(no reform LLM, raw query) como en synthesis (history vacía → el prompt
del LLM final no ve los turnos previos).

Estos tests son *unit* — no levantan FastAPI. Cubren las dos partes
verificables sin TestClient: (1) el constante quedó a 0.40, (2) el regex
`_looks_like_followup` NO matchea queries autónomas tipo "decime que
ideas de tech app tengo?", garantizando que la nueva rama de drop se
ejecuta cuando corresponde.
"""
from __future__ import annotations


def test_reform_cosine_high_is_0_40():
    """Garantiza que la banda borderline [0.32, 0.40) está activa.

    Si alguien sube el threshold a >= 0.50 sin actualizar el branch de
    borderline-drop, el bug del DNI puede volver. Este test es la canary.
    """
    from web.server import REFORM_COSINE_HIGH

    assert REFORM_COSINE_HIGH == 0.40, (
        f"REFORM_COSINE_HIGH={REFORM_COSINE_HIGH}. Si lo cambiás, "
        "actualizá también el branch de borderline-drop en /api/chat."
    )


def test_followup_regex_misses_standalone_questions():
    """Las dos queries del bug report NO deben ser detectadas como
    follow-ups por el regex — porque si lo fueran, el branch de
    borderline-drop NO dispararía y la history se mantendría.

    `_looks_like_followup` busca conectores ("y", "pero"), demostrativos
    ("eso", "esa"), comparativos ("también"), elipsis ("y vos?"). Las
    queries del bug son standalone: tienen verbos y sujetos completos.
    """
    from web.server import _looks_like_followup

    # El turno problemático del bug report — query autónoma con verbo +
    # objeto explícito ("ideas de tech app"). El borderline-drop chequea
    # `_looks_like_followup(question)` sobre la query CURRENT, así que
    # alcanza con que esta no matchee como follow-up.
    assert _looks_like_followup("decime que ideas de tech app tengo?") is False
    # Sanity: una query realmente follow-up SÍ matchea (control test).
    assert _looks_like_followup("y eso por qué?") is True


def test_followup_regex_keeps_real_followups():
    """Sanity: queries que SÍ son follow-ups deben matchear el regex
    para que el borderline-drop NO dispare y se preserve la history.
    """
    from web.server import _looks_like_followup

    # Estas son ellipsis del turno previo — necesitan history.
    assert _looks_like_followup("y los gastos de la visa?") is True
    assert _looks_like_followup("y eso?") is True
