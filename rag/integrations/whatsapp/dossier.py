"""Per-contact dossier generator (Game Changer #2).

Cron diario regenera la sección ``## Dossier`` en cada nota de
`99-Contacts/<persona>.md` con LLM-resumen cross-source de:

- Última conversación WA (tail de mensajes recientes con esa persona).
- Promesas abiertas (rag_promises status='pending' direction in/out).
- Eventos compartidos próximos (Apple Calendar matchando el nombre).
- Sentiment del último 30 días (positivo/neutral/negativo).
- Notas vault que mencionan al contacto (top 3 por relevance).
- Quote pending action: si hay loop abierto, sugerir next step.

El listener TS lee el bloque ``## Dossier`` antes de generar drafts y lo
inyecta al system prompt. Drafts pasan de "OK" a "wow, sabe TODO sobre
mi relación con esta persona".

## Surface

- ``generate_dossier(contact_name, *, llm=None) -> dict`` — produce el
  dict del dossier sin escribir.
- ``write_dossier_to_note(contact_name, dossier_dict) -> bool`` — append/
  reemplaza la sección ``## Dossier`` en la nota del contacto. Idempotente.
- ``refresh_all_dossiers(*, max_workers=2, dry_run=False) -> dict`` —
  regenera dossier de TODOS los contactos del vault. Usado por el cron.

## Why deferred imports

Mantenemos el patrón del package: imports `from rag import ...` adentro
del cuerpo, para evitar ciclos al load-time + respetar monkeypatches.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


_DOSSIER_HEADING = "## Dossier"
_DOSSIER_PROMPT = (
    "Sos un asistente que prepara un DOSSIER ULTRA-COMPACTO sobre un contacto "
    "del user, para que el bot de WhatsApp tenga contexto al responder mensajes "
    "del contacto o redactar drafts. Idioma: español rioplatense (voseo). NO "
    "portugués. NO tuteo peninsular.\n\n"
    "El dossier va a ser leído por OTRO LLM al generar respuestas — sé denso "
    "(SIN fluff) pero específico. Bullets cortos.\n\n"
    "Estructura el output en JSON estricto con estos campos:\n"
    "{\n"
    "  \"summary\": \"2-3 líneas: quién es esta persona, qué relación tiene "
    "con el user, qué tono usar.\",\n"
    "  \"recent_topics\": [\"tema 1 reciente\", \"tema 2\", ...] — máx 5,\n"
    "  \"open_promises\": [\"promise pendiente 1\", ...] — máx 3,\n"
    "  \"upcoming_events\": [\"evento 1 con fecha\", ...] — máx 3,\n"
    "  \"vibe\": \"positivo|neutral|tenso|distante|cercano\",\n"
    "  \"suggested_next_step\": \"frase corta o null\"\n"
    "}\n\n"
    "Si no hay datos suficientes, devolvé los arrays vacíos y el campo "
    "correspondiente en null. NO inventes."
)


def generate_dossier(
    contact_name: str,
    *,
    days: int = 30,
    max_messages: int = 30,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Genera el dict del dossier para un contacto.

    Pipeline:
      1. Resolver al contacto via vault `99-Contacts/`.
      2. Pull mensajes WA recientes (last 30d default).
      3. Pull promises abiertas con esa persona.
      4. Pull eventos calendar próximos que mencionen el name.
      5. LLM-resumir todo en JSON estructurado.

    Devuelve dict con shape del prompt. Silent-fail: si algo no está
    disponible (LLM down, no contact, etc.), devuelve dict vacío con
    `error` field.
    """
    from rag.integrations.whatsapp import (
        _lookup_vault_contact,
        _fetch_whatsapp_recent_with_jid,
        _whatsapp_jid_from_contact,
    )
    contact = _lookup_vault_contact(contact_name)
    if not contact:
        return {"error": f"contact {contact_name!r} not in vault `99-Contacts/`"}

    full_name = contact.get("full_name") or contact_name

    # 1. Pull mensajes WA recientes
    wa_msgs: list[dict] = []
    try:
        lookup = _whatsapp_jid_from_contact(contact_name)
        if lookup.get("jid"):
            recent = _fetch_whatsapp_recent_with_jid(lookup["jid"], limit=max_messages)
            wa_msgs = recent.get("messages", [])
    except Exception:
        pass

    # 2. Pull promises abiertas (mismo full_name match)
    promises_block: list[dict] = []
    try:
        import rag as _rag
        with _rag._ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT text, when_text, direction, msg_ts FROM rag_promises "
                "WHERE status = 'pending' AND speaker = ? OR speaker LIKE ? "
                "ORDER BY id DESC LIMIT 5",
                (full_name, f"%{full_name.split()[0]}%"),
            ).fetchall()
            promises_block = [
                {
                    "text": r[0],
                    "when_text": r[1] or "",
                    "direction": r[2],
                    "msg_ts": r[3] or "",
                }
                for r in rows
            ]
    except Exception:
        pass

    # 3. Eventos calendar próximos (next 30d)
    events_block: list[dict] = []
    try:
        import rag as _rag
        if hasattr(_rag, "_fetch_calendar_window"):
            events_raw = _rag._fetch_calendar_window(days_ahead=30) or []
            name_lower = full_name.lower()
            for ev in events_raw[:50]:
                title = (ev.get("title") or "").lower()
                if name_lower in title or any(
                    p in title for p in name_lower.split()[:1]
                ):
                    events_block.append({
                        "title": ev.get("title"),
                        "start": ev.get("start"),
                    })
    except Exception:
        pass

    # 4. Build LLM input + call
    convo_lines: list[str] = []
    for m in wa_msgs[-20:]:
        ts = (m.get("ts") or "")[:16].replace("T", " ")
        who = m.get("who") or "?"
        text = (m.get("text") or "")[:200]
        convo_lines.append(f"[{ts}] {who}: {text}")
    convo = "\n".join(convo_lines) or "(sin mensajes recientes)"

    promises_str = "\n".join(
        f"- {p['direction']}: {p['text']} ({p['when_text'] or 'sin fecha'})"
        for p in promises_block
    ) or "(sin promesas pendientes)"

    events_str = "\n".join(
        f"- {ev['start']}: {ev['title']}" for ev in events_block
    ) or "(sin eventos próximos)"

    contact_meta = (
        f"Relación: {contact.get('relation_label') or '?'}. "
        f"Aliases: {', '.join(contact.get('aliases', [])) or '(ninguno)'}."
    )

    user_input = (
        f"Contacto: {full_name}\n"
        f"Metadata: {contact_meta}\n\n"
        f"## Conversación WhatsApp reciente (últimos {days}d)\n{convo}\n\n"
        f"## Promesas pendientes\n{promises_str}\n\n"
        f"## Eventos calendar próximos\n{events_str}"
    )

    try:
        import rag as _rag
        model = llm_model or _rag.HELPER_MODEL
        resp = _rag._summary_client().chat(
            model=model,
            messages=[
                {"role": "system", "content": _DOSSIER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": 600, "num_ctx": 8192},
            keep_alive=_rag.LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception as exc:
        return {"error": f"llm_failed: {exc!r}"[:200]}

    if not isinstance(data, dict):
        return {"error": "llm_returned_non_dict"}

    # Validate shape — todas las keys esperadas, defaults razonables
    return {
        "summary": str(data.get("summary") or "").strip(),
        "recent_topics": [str(t).strip() for t in (data.get("recent_topics") or [])[:5] if t],
        "open_promises": [str(p).strip() for p in (data.get("open_promises") or [])[:3] if p],
        "upcoming_events": [str(e).strip() for e in (data.get("upcoming_events") or [])[:3] if e],
        "vibe": str(data.get("vibe") or "neutral"),
        "suggested_next_step": (data.get("suggested_next_step") or None),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "contact_full_name": full_name,
    }


def render_dossier_markdown(d: dict[str, Any]) -> str:
    """Convierte el dict del dossier a markdown para inyectar en la nota."""
    if d.get("error"):
        return f"{_DOSSIER_HEADING}\n\n_(no se pudo generar: {d['error']})_\n"
    lines: list[str] = [_DOSSIER_HEADING, ""]
    if d.get("summary"):
        lines.append(d["summary"])
        lines.append("")
    if d.get("vibe"):
        lines.append(f"**Vibe**: {d['vibe']}")
    if d.get("recent_topics"):
        lines.append("**Temas recientes**:")
        for t in d["recent_topics"]:
            lines.append(f"- {t}")
    if d.get("open_promises"):
        lines.append("**Promesas pendientes**:")
        for p in d["open_promises"]:
            lines.append(f"- {p}")
    if d.get("upcoming_events"):
        lines.append("**Eventos próximos**:")
        for e in d["upcoming_events"]:
            lines.append(f"- {e}")
    if d.get("suggested_next_step"):
        lines.append("")
        lines.append(f"**Siguiente paso sugerido**: {d['suggested_next_step']}")
    if d.get("generated_at"):
        lines.append("")
        lines.append(f"_Generado: {d['generated_at']} · modelo: {d.get('model', '?')}_")
    return "\n".join(lines) + "\n"


def write_dossier_to_note(contact_name: str, dossier: dict[str, Any]) -> bool:
    """Append/reemplaza la sección `## Dossier` en `99-Contacts/<name>.md`.

    Idempotente: si la sección ya existe, la reemplaza completa. Si no,
    la appendea al final de la nota. NO toca el resto del contenido.
    """
    from rag.integrations.whatsapp import _vault_contacts_dir, _load_vault_contacts, _normalize_hint
    base = _vault_contacts_dir()
    if not base:
        return False
    contacts = _load_vault_contacts()
    target_path: Path | None = None
    name_norm = _normalize_hint(contact_name)
    for c in contacts:
        if _normalize_hint(c["stem"]) == name_norm:
            target_path = c["path"]
            break
        if _normalize_hint(c["parsed"].get("full_name", "")) == name_norm:
            target_path = c["path"]
            break
    if not target_path:
        return False
    try:
        text = target_path.read_text(encoding="utf-8")
    except Exception:
        return False

    new_block = render_dossier_markdown(dossier)

    # Reemplazar bloque existente o agregar al final
    idx = text.find(_DOSSIER_HEADING)
    if idx == -1:
        # Append al final
        if not text.endswith("\n"):
            text += "\n"
        if not text.endswith("\n\n"):
            text += "\n"
        new_text = text + new_block
    else:
        # Buscar próxima `## ` después del dossier (delimita el bloque)
        next_section = text.find("\n## ", idx + len(_DOSSIER_HEADING))
        if next_section == -1:
            new_text = text[:idx] + new_block
        else:
            new_text = text[:idx] + new_block + text[next_section:]

    try:
        target_path.write_text(new_text, encoding="utf-8")
    except Exception:
        return False
    return True


def refresh_all_dossiers(
    *,
    only_with_recent_activity: bool = True,
    activity_days: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Regenera dossier de todos los contactos del vault.

    Cuando `only_with_recent_activity=True` (default), skipea contactos
    sin mensajes WA en los últimos `activity_days` días — no vale la pena
    gastar LLM call para alguien con quien no interactúan.

    Returns `{processed, skipped, errors, by_contact}`.
    """
    from rag.integrations.whatsapp import _load_vault_contacts
    contacts = _load_vault_contacts()
    summary: dict[str, Any] = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "by_contact": [],
    }
    for c in contacts:
        stem = c["stem"]
        try:
            dossier = generate_dossier(stem)
            if dossier.get("error"):
                summary["errors"] += 1
                summary["by_contact"].append({"contact": stem, "status": "error", "detail": dossier["error"]})
                continue
            # Skip si no hay activity reciente Y filtro activo
            if only_with_recent_activity:
                has_signal = bool(
                    dossier.get("recent_topics") or dossier.get("open_promises")
                    or dossier.get("upcoming_events")
                )
                if not has_signal:
                    summary["skipped"] += 1
                    summary["by_contact"].append({"contact": stem, "status": "skipped"})
                    continue
            if not dry_run:
                ok = write_dossier_to_note(stem, dossier)
                if ok:
                    summary["processed"] += 1
                    summary["by_contact"].append({"contact": stem, "status": "written"})
                else:
                    summary["errors"] += 1
                    summary["by_contact"].append({"contact": stem, "status": "write_failed"})
            else:
                summary["processed"] += 1
                summary["by_contact"].append({"contact": stem, "status": "would_write"})
        except Exception as exc:
            summary["errors"] += 1
            summary["by_contact"].append({"contact": stem, "status": "exception", "detail": repr(exc)[:120]})
    return summary


def read_dossier_from_note(contact_name: str) -> str | None:
    """Lee la sección `## Dossier` de la nota del contacto. Returns markdown
    raw o None si no existe / la sección no fue creada todavía.

    Used por el listener (via HTTP endpoint en web/server.py) para inyectar
    el dossier al system prompt antes de generar drafts.
    """
    from rag.integrations.whatsapp import _vault_contacts_dir, _load_vault_contacts, _normalize_hint
    base = _vault_contacts_dir()
    if not base:
        return None
    contacts = _load_vault_contacts()
    name_norm = _normalize_hint(contact_name)
    target: Path | None = None
    for c in contacts:
        if _normalize_hint(c["stem"]) == name_norm:
            target = c["path"]
            break
    if not target:
        return None
    try:
        text = target.read_text(encoding="utf-8")
    except Exception:
        return None
    idx = text.find(_DOSSIER_HEADING)
    if idx == -1:
        return None
    next_section = text.find("\n## ", idx + len(_DOSSIER_HEADING))
    if next_section == -1:
        return text[idx:].strip()
    return text[idx:next_section].strip()


__all__ = [
    "generate_dossier",
    "render_dossier_markdown",
    "write_dossier_to_note",
    "refresh_all_dossiers",
    "read_dossier_from_note",
]
