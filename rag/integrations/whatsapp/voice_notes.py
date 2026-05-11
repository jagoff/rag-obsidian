"""Voice notes inbound — writer al vault, asociado al contact note.

Cuando una nota de voz de WhatsApp se transcribe (vía endpoint
`/api/wa/voice/transcript/{msg_id}` o el daemon de batch que la pre-
cachea), escribimos el transcript como una nota .md bajo

  `99-obsidian/99-AI/external-ingest/whatsapp-voice/<contact-slug>/<date>-<msg-id>.md`

con frontmatter `related: ["[[<Contact Name>]]"]` y un wikilink
`[[<Contact Name>]]` en el body para que **el backlinks panel del
contact note muestre todas las voice notes asociadas**. Si el jid
no resuelve a un contact note conocido, fallback a `<jid-slug>/` y
nombre genérico (legacy behavior).

`rag watch` indexa el folder → `rag query` retorna snippets cuando
matchean, y desde Obsidian abrís el contacto y ves todos los
transcripts en los backlinks.

Naming idempotente: msg_id ya es único globalmente. Si la nota existe,
solo se sobrescribe si el body cambió (modelo whisper distinto al
re-correr con `use_cache=False`).
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

# Sub-folder bajo el vault donde van todas las notas de voz transcritas.
# `99-obsidian/99-AI/` es la convención del user para "infra del
# sistema" (ver `~/.claude/CLAUDE.md` § "Artefactos del sistema → vault
# Obsidian"). Sub-folder por feature.
_VOICE_SUBPATH = "99-obsidian/99-AI/external-ingest/whatsapp-voice"


def _slug_jid(jid: str) -> str:
    """JID → folder name safe.

    `5493425153999@s.whatsapp.net` → `5493425153999`.
    `123456789012345@g.us`         → `group-123456789012345`.
    `abc@lid`                      → `lid-abc`.
    """
    if not jid or "@" not in jid:
        return "unknown"
    local, _, domain = jid.partition("@")
    if domain == "g.us":
        return f"group-{local}"
    if domain == "lid":
        return f"lid-{local}"
    return local


def _slug_msg_id(msg_id: str) -> str:
    """msg_id puede contener caracteres no aptos para filename (ej. `/`)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", msg_id)[:80]


def _slug_name(name: str) -> str:
    """Display name → folder slug safe (lowercase, alphanum + hyphens)."""
    s = re.sub(r"[^A-Za-z0-9\s_-]", "", (name or "").strip().lower())
    s = re.sub(r"\s+", "-", s).strip("-_")
    return s or "unknown"


# Cache jid → (contact_name, mtime_max). Re-build cuando algún .md de
# 99-Contacts cambia. Evita escanear el folder en cada transcripción.
_CONTACT_INDEX_LOCK = threading.Lock()
_CONTACT_INDEX: dict[str, str] = {}
_CONTACT_INDEX_MTIME: float = 0.0


def _contacts_dir() -> Path | None:
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return None
    from rag.integrations.whatsapp._constants import VAULT_CONTACTS_SUBPATH  # noqa: PLC0415

    d = VAULT_PATH / VAULT_CONTACTS_SUBPATH
    return d if d.is_dir() else None


def _rebuild_contact_index() -> None:
    """Reconstruye el cache jid → contact name escaneando 99-Contacts/."""
    global _CONTACT_INDEX, _CONTACT_INDEX_MTIME
    d = _contacts_dir()
    if not d:
        return
    idx: dict[str, str] = {}
    latest_mtime = 0.0
    wa_jid_re = re.compile(
        r"^-\s*\*\*\s*wa_jid\s*\*\*\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
    )
    for note_path in d.glob("*.md"):
        if note_path.name.startswith("_"):
            continue
        try:
            mt = note_path.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
            text = note_path.read_text(encoding="utf-8")
        except OSError:
            continue
        m = wa_jid_re.search(text)
        if not m:
            continue
        # Una nota puede tener múltiples jids separados por coma (group + lid).
        raw = m.group(1).strip()
        if not raw or raw.startswith("<"):
            continue
        # Name del file (sin .md) es lo que va al wikilink.
        name = note_path.stem
        for jid in re.split(r"[,\s]+", raw):
            jid = jid.strip()
            if "@" in jid:
                idx[jid] = name
    with _CONTACT_INDEX_LOCK:
        _CONTACT_INDEX = idx
        _CONTACT_INDEX_MTIME = latest_mtime


def _lookup_contact_name(jid: str) -> str | None:
    """Resuelve jid → display name del contact note (ej. 'Astor').

    None si no existe contact note con ese wa_jid. Cache mtime-aware:
    si alguna nota cambió, re-escanea automáticamente.
    """
    if not jid:
        return None
    d = _contacts_dir()
    if not d:
        return None
    # Detectar si algún .md cambió desde la última build.
    try:
        current_max = max((p.stat().st_mtime for p in d.glob("*.md")), default=0.0)
    except OSError:
        current_max = 0.0
    with _CONTACT_INDEX_LOCK:
        stale = current_max > _CONTACT_INDEX_MTIME or not _CONTACT_INDEX
    if stale:
        _rebuild_contact_index()
    with _CONTACT_INDEX_LOCK:
        return _CONTACT_INDEX.get(jid)


def _vault_voice_dir(jid: str, *, contact_name: str | None = None) -> Path | None:
    """Folder destino para los transcripts de un jid.

    Si `contact_name` se pasa o se resuelve, usamos su slug ('astor').
    Si no, fallback a jid-slug ('lid-242253...'). En ambos casos el
    folder vive bajo `99-obsidian/99-AI/external-ingest/whatsapp-voice/`.
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return None
    name = contact_name or _lookup_contact_name(jid)
    slug = _slug_name(name) if name else _slug_jid(jid)
    base = VAULT_PATH / _VOICE_SUBPATH / slug
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_voice_note(
    *,
    msg_id: str,
    jid: str,
    sender: str | None,
    text: str,
    audio_ts: str | None = None,
) -> Path | None:
    """Persiste el transcript como nota .md en el vault. Idempotente.

    El folder destino y el wikilink al contacto se resuelven via
    `_lookup_contact_name(jid)`. Si el jid corresponde a una nota de
    `99-Contacts/<Name>.md`:
      - Folder: `99-AI/external-ingest/whatsapp-voice/<name-slug>/`
      - Frontmatter `related: ["[[<Name>]]"]` (backlinks panel del
        contact note muestra todos los transcripts vinculados).
      - Wikilink `[[<Name>]]` en el body.
    Si no se resuelve, fallback a jid-slug folder + sender plain text.

    Devuelve el path escrito o `None` si falla (vault inaccesible,
    transcript vacío, etc.) — el caller tolera el `None`.
    """
    if not msg_id or not text or not text.strip():
        return None
    contact_name = _lookup_contact_name(jid)
    base = _vault_voice_dir(jid, contact_name=contact_name)
    if base is None:
        return None

    # Date prefix para que el listing del folder quede cronológico.
    date_prefix = ""
    if audio_ts:
        try:
            dt = datetime.fromisoformat(str(audio_ts).split("+")[0].split("T")[0])
            date_prefix = dt.strftime("%Y-%m-%d") + "-"
        except (ValueError, TypeError):
            try:
                dt = datetime.fromtimestamp(int(float(audio_ts)))
                date_prefix = dt.strftime("%Y-%m-%d") + "-"
            except (ValueError, TypeError):
                date_prefix = ""
    if not date_prefix:
        date_prefix = datetime.now().strftime("%Y-%m-%d") + "-"

    fname = f"{date_prefix}{_slug_msg_id(msg_id)}.md"
    path = base / fname

    sender_clean = (sender or "").split("@")[0] or "unknown"
    speaker_display = contact_name or sender_clean
    fm_audio_ts = (audio_ts or "").replace('"', "'")
    related_line = (
        f'related: ["[[{contact_name}]]"]\n' if contact_name else ""
    )
    backlink = f"[[{contact_name}]]" if contact_name else speaker_display
    body = (
        "---\n"
        "type: wa-voice\n"
        f"jid: {jid}\n"
        f"sender: {sender_clean}\n"
        f"contact: {contact_name or ''}\n"
        f"msg_id: {msg_id}\n"
        f"audio_ts: \"{fm_audio_ts}\"\n"
        f"transcribed_at: {datetime.now().isoformat(timespec='seconds')}\n"
        + related_line
        + "tags: [wa-voice, voice-note]\n"
        "---\n\n"
        f"# Voice note de {backlink}\n\n"
        f"{text.strip()}\n"
    )

    # Idempotencia: solo overwrite si el contenido cambia.
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
            if existing == body:
                _cleanup_legacy_jid_slug(jid, contact_name, fname, path)
                return path
        except OSError:
            pass
    try:
        path.write_text(body, encoding="utf-8")
    except OSError:
        return None
    # Después de escribir en el folder canonical, eliminamos el
    # duplicate legacy en `<jid-slug>/` si quedó de una transcripción
    # anterior. Si no, el corpus indexa el msg dos veces.
    _cleanup_legacy_jid_slug(jid, contact_name, fname, path)
    return path


def _cleanup_legacy_jid_slug(
    jid: str, contact_name: str | None, fname: str, current_path: Path
) -> None:
    """Si el path canonical es contact-slug, borra el duplicate jid-slug.

    No hace nada si:
    - No se resolvió contact_name (canonical YA es jid-slug).
    - El path legacy no existe.
    - El path legacy y el canonical son el mismo (shouldn't happen).
    """
    if not contact_name:
        return
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return
    legacy = VAULT_PATH / _VOICE_SUBPATH / _slug_jid(jid) / fname
    if legacy == current_path:
        return
    if legacy.is_file():
        try:
            legacy.unlink()
        except OSError:
            return
        # Si el folder legacy quedó vacío, lo barremos también.
        legacy_dir = legacy.parent
        try:
            if legacy_dir.is_dir() and not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
        except OSError:
            pass


__all__ = ["write_voice_note"]
