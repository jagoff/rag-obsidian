"""Voice notes inbound — writer al vault.

Cuando una nota de voz de WhatsApp se transcribe (vía endpoint
`/api/wa/voice/transcript/{msg_id}` o el daemon de batch que la pre-
cachea), escribimos el transcript como una nota .md bajo

  `99-obsidian/99-AI/external-ingest/whatsapp-voice/<jid-slug>/<date>-<msg-id>.md`

para que `rag watch` la pickee y entre al corpus. A partir de ahí,
`rag query` puede retornar el snippet del audio cuando matchea
semánticamente con la pregunta del user.

Naming idempotente: el msg_id ya es único globalmente. Si la nota
existe ya, lo overwriteamos solo si el transcript cambió (ej.
modelo whisper distinto al re-correr con `use_cache=False`).
"""

from __future__ import annotations

import re
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


def _vault_voice_dir(jid: str) -> Path | None:
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return None
    base = VAULT_PATH / _VOICE_SUBPATH / _slug_jid(jid)
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

    Devuelve el path escrito, o `None` si no se pudo (vault inaccesible,
    transcript vacío, etc.) — el caller debe tolerar el `None`.
    """
    if not msg_id or not text or not text.strip():
        return None
    base = _vault_voice_dir(jid)
    if base is None:
        return None

    # Date prefix para que el listing del folder quede cronológico.
    date_prefix = ""
    if audio_ts:
        # audio_ts puede venir como ISO o como string raw del bridge
        # (numérico unix). Intentamos parsear ISO primero.
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
    fm_audio_ts = (audio_ts or "").replace('"', "'")
    body = (
        "---\n"
        "type: wa-voice\n"
        f"jid: {jid}\n"
        f"sender: {sender_clean}\n"
        f"msg_id: {msg_id}\n"
        f"audio_ts: \"{fm_audio_ts}\"\n"
        f"transcribed_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "tags: [wa-voice, voice-note]\n"
        "---\n\n"
        f"# Voice note de {sender_clean}\n\n"
        f"{text.strip()}\n"
    )

    # Idempotencia: solo overwrite si el contenido cambia.
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
            if existing == body:
                return path
        except OSError:
            pass
    try:
        path.write_text(body, encoding="utf-8")
    except OSError:
        return None
    return path


__all__ = ["write_voice_note"]
