"""Source-specific intent hint builder — turn-scoped LLM directive cuando
el pre-router fuerza un tool source-specific (mails / calendario / drive /
recordatorios / WhatsApp).

Extraído de ``web/server.py`` (Phase W2, 2026-05-08). Pure data + función
formateadora — sin thread state, sin imports del FastAPI app.

## Surface

- ``_SOURCE_INTENT_META`` — tabla por tool con `label`, `live_section`,
  `digest_hint`, `item_shape`, `empty_phrase`. Source of truth.
- ``_SOURCE_INTENT_LABEL`` — vista derivada `(label, live_section)` para
  back-compat con tests viejos.
- ``_build_source_intent_hint(forced_tool_names)`` — compone el system
  message turn-scoped que se inyecta antes del LLM call. Devuelve None
  cuando ninguna tool del forced set es source-specific (solo weather /
  finance_summary que no tienen "concepto de ausencia").

## Motivación histórica (2026-04-24, user report iter 1-3)

- Iter 1: regex plurales faltaba → `gmail_recent` no disparaba → el
  sistema respondía con WhatsApp sin reconocer intent.
- Iter 2: con el tool disparando pero vacío, el CONTEXTO se reemplazaba
  → LLM sin material para fallback → "te dejo otras fuentes" abstracto.
- Iter 3: con el CONTEXTO preservado, el LLM tenía notas de
  `99-obsidian/99-AI/external-ingest/Gmail/*.md` disponibles, PERO hablaba
  de "tu nota del 22 de abril" en vez de extraer los asuntos. Este hint
  ahora explicita el formato deseado.
"""

from __future__ import annotations


# Metadata por tool source-specific, usada para componer el hint de
# "intención explícita" que se le pasa al LLM cuando el pre-router
# disparó un tool. Campos:
#   label         — cómo nombramos la fuente al user ("tus mails/correos").
#   live_section  — header de la sección fresca en CONTEXTO que renderea
#                   `_format_forced_tool_output` (p.ej. "### Mails").
#   digest_hint   — dónde más buscar items indexados en el CONTEXTO si la
#                   live section está vacía. Para mails: las notas de
#                   99-obsidian/99-AI/external-ingest/Gmail/YYYY-MM-DD.md usan un `## <asunto>`
#                   por mail con From/Date/Snippet debajo — extraer esos
#                   H2 da un listado crudo de "mis últimos mails" que el
#                   user espera. Otros sources tienen su propio formato.
#   item_shape    — ejemplo del formato que debe usar cada bullet en la
#                   respuesta final, para que el LLM no invente prosa
#                   cuando el user pidió un listado.
#   empty_phrase  — frase explícita cuando NO hay nada ni live ni en
#                   digest. Reemplaza el vago "te dejo otras fuentes".
#
# Weather / finance_summary NO están acá porque no son "fuentes" que el
# user busca — son resúmenes autogenerados sin concepto de ausencia.
_SOURCE_INTENT_META: dict[str, dict[str, str]] = {
    "gmail_recent": {
        "label": "tus mails/correos",
        "live_section": "### Mails",
        "digest_hint": (
            "Si en el CONTEXTO hay notas del vault del tipo "
            "`99-obsidian/99-AI/external-ingest/Gmail/YYYY-MM-DD.md` (cada `## <asunto>` "
            "dentro es UN mail, con su **From:**, **Date:** y **Snippet:**), "
            "extraé esos asuntos y listálos uno por línea — son LITERALMENTE "
            "los últimos mails del usuario. NO digas 'en tu nota' ni "
            "menciones la ruta de la nota: los asuntos SON los mails."
        ),
        "item_shape": "- <asunto> (de <remitente>)",
        "empty_phrase": "No encontré mails recientes en tu corpus",
    },
    "calendar_ahead": {
        "label": "tu calendario/agenda/eventos",
        "live_section": "### Calendario",
        "digest_hint": (
            "Si en el CONTEXTO hay notas con eventos (morning brief, "
            "agenda del día), extraé los títulos de los eventos y listálos."
        ),
        "item_shape": "- <título> (<fecha/hora>)",
        "empty_phrase": "No tenés eventos en el horizonte",
    },
    "reminders_due": {
        "label": "tus recordatorios/pendientes",
        "live_section": "### Recordatorios",
        "digest_hint": (
            "Si en el CONTEXTO hay notas que mencionan tareas pendientes "
            "(morning/evening brief, PARA projects), extraelas y listálas."
        ),
        "item_shape": "- <tarea> (<fecha si tiene>)",
        "empty_phrase": "No tenés recordatorios pendientes",
    },
    "drive_search": {
        "label": "tu Google Drive",
        "live_section": "### Google Drive",
        "digest_hint": (
            "La sección live trae los archivos encontrados con su body "
            "exportado. Si el user pidió un dato concreto (precio, deuda, "
            "cantidad), citálo TEXTUAL del body si está; si no aparece, "
            "decí explícitamente que buscaste y no encontraste ese dato."
        ),
        "item_shape": "- <nombre del archivo> (<tipo>) · <dato relevante o 'sin match'>",
        "empty_phrase": "No encontré nada en tu Google Drive que matchee",
    },
    "whatsapp_pending": {
        "label": "tus chats de WhatsApp esperando respuesta",
        "live_section": "### WhatsApp",
        "digest_hint": (
            "La sección live trae los chats donde el user debe el próximo "
            "mensaje (último inbound sin reply). Si en el CONTEXTO hay "
            "notas de `99-obsidian/99-AI/external-ingest/WhatsApp/<contacto>/YYYY-MM.md` con "
            "más contexto de esos chats, podés complementar. NUNCA "
            "inventes conversaciones de WhatsApp — si la sección live "
            "está vacía decilo explícitamente en vez de citar otras "
            "fuentes como si fueran WA."
        ),
        "item_shape": "- <contacto> (hace <Xh/d>): <último mensaje>",
        "empty_phrase": "No hay chats de WhatsApp esperando tu respuesta",
    },
    "whatsapp_search": {
        "label": "tus mensajes de WhatsApp (búsqueda por contenido)",
        "live_section": "### WhatsApp",
        "digest_hint": (
            "La sección live trae los mensajes WhatsApp matcheantes a la "
            "query, ordenados por relevancia. Cada bullet tiene "
            "`[<contacto> · <fecha>] <snippet>`; si arranca con `yo →` el "
            "mensaje lo mandó el user, no el contacto. NUNCA inventes "
            "conversaciones — citá TEXTUAL de los snippets que aparecen, "
            "y si la sección live está vacía decilo explícitamente."
        ),
        "item_shape": "- <contacto> (<fecha>): <cita textual del snippet>",
        "empty_phrase": "No encontré mensajes de WhatsApp que matcheen tu búsqueda",
    },
}


# Retrocompatibilidad: el helper previo `_SOURCE_INTENT_LABEL` es un
# mapping más chico (label + section) que usan los tests directos.
# Lo mantenemos derivado de `_SOURCE_INTENT_META` para no romper imports
# viejos ni tener dos sources of truth que puedan divergir.
_SOURCE_INTENT_LABEL: dict[str, tuple[str, str]] = {
    name: (meta["label"], meta["live_section"])
    for name, meta in _SOURCE_INTENT_META.items()
}


def _build_source_intent_hint(forced_tool_names: list[str]) -> str | None:
    """Compone un system message turn-scoped que le dice al LLM cómo
    responder cuando el user preguntó explícitamente por una fuente
    concreta (mails / calendario / recordatorios).

    El hint combina:

    1. Dónde buscar primero (la sección live del tool: "### Mails").
    2. Dónde buscar si la live está vacía (notas indexadas del vault con
       formato conocido — p.ej. 99-obsidian/99-AI/external-ingest/Gmail/YYYY-MM-DD.md tiene
       un H2 por mail, listar esos H2 == listar los últimos mails).
    3. Formato de respuesta esperado (viñetas, shape por item).
    4. Qué PROHIBIR explícitamente (decir "tus notas" / "otras fuentes" /
       "te dejo esto por si ayuda" — vocabulario abstracto que no le
       sirve al user cuando pidió una lista concreta).
    5. Frase canned cuando NO hay nada (reemplaza el vago "te dejo
       otras fuentes").

    Motivación histórica (2026-04-24, user report iter 1-3):

    - Iter 1: regex plurales faltaba → `gmail_recent` no disparaba → el
      sistema respondía con WhatsApp sin reconocer intent.
    - Iter 2: con el tool disparando pero vacío, el CONTEXTO se reemplazaba
      → LLM sin material para fallback → "te dejo otras fuentes" abstracto.
    - Iter 3 (este): con el CONTEXTO preservado, el LLM tiene notas de
      `99-obsidian/99-AI/external-ingest/Gmail/*.md` disponibles, PERO hablaba de "tu nota
      del 22 de abril" y "fuentes" en lugar de extraer los asuntos de
      los mails. User feedback textual: "en vez de fuentes (que no tiene
      sentido porque son notas de obsidian) trae los titulos de los
      mails". Este hint ahora explicita el formato deseado.

    Devuelve None si ninguna tool es source-specific (solo weather o
    finance_summary). En ese caso no hay hint que agregar y el system
    prompt default alcanza.
    """
    metas = [_SOURCE_INTENT_META[n] for n in forced_tool_names
             if n in _SOURCE_INTENT_META]
    if not metas:
        return None

    def _join(parts: list[str]) -> str:
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} y {parts[1]}"
        return ", ".join(parts[:-1]) + f" y {parts[-1]}"

    labels = [m["label"] for m in metas]
    sections = [m["live_section"] for m in metas]
    joined_labels = _join(labels)
    joined_sections = _join(sections)
    digest_block = "\n".join(f"  • {m['digest_hint']}" for m in metas)
    shape_block = "\n".join(f"  • {m['item_shape']}" for m in metas)
    empty_phrase = _join([m["empty_phrase"] for m in metas])

    return (
        f"INTENCIÓN EXPLÍCITA DEL USUARIO: pidió {joined_labels}. "
        f"Tu tarea es devolver un LISTADO CONCRETO, no un resumen abstracto "
        f"ni una referencia a 'fuentes' del sistema.\n\n"
        f"ORDEN DE BÚSQUEDA en el CONTEXTO:\n"
        f"  1. Sección live {joined_sections}: si tiene items, listálos.\n"
        f"  2. Si la sección live está vacía, buscá en el resto del "
        f"CONTEXTO data indexada de esta fuente:\n{digest_block}\n\n"
        f"FORMATO DE RESPUESTA — lista con viñetas, un item por línea:\n"
        f"{shape_block}\n\n"
        f"PROHIBIDO:\n"
        f"  • Decir 'en tu nota X', 'en tus fuentes', 'te dejo otras "
        f"fuentes que podrían ayudarte', 'revisá tus notas' — el user "
        f"pidió los items directamente, no una meta-referencia.\n"
        f"  • Mencionar rutas del vault (`03-Resources/...`, "
        f"`04-Archive/...`) ni el sistema PARA.\n"
        f"  • Resumir en prosa cuando la pregunta exige un listado.\n"
        f"  • Responder sobre WhatsApp u otras fuentes como si fueran la "
        f"respuesta principal cuando el usuario pidió {joined_labels}.\n\n"
        f"SI NO HAY DATA NI LIVE NI INDEXADA: respondé exactamente "
        f"'{empty_phrase}' — sin agregar sugerencias de fallback."
    )


__all__ = [
    "_SOURCE_INTENT_META",
    "_SOURCE_INTENT_LABEL",
    "_build_source_intent_hint",
]
