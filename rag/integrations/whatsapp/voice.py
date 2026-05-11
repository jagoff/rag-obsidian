"""Voice notes — record (browser) → transcribe (Whisper MLX) → send PTT.

Pipeline:

1. Browser graba con MediaRecorder, sube blob `audio/webm;codecs=opus`.
2. ``transcribe_blob(bytes)`` escribe a tmp + corre Whisper MLX vía
   ``rag.whisper.transcribe_audio`` (con cache SQL).
3. ``to_opus(in_path)`` re-encodea a ``.ogg/opus 32kbps mono`` con
   ``ffmpeg`` (formato esperado por WhatsApp para que el receptor lo vea
   como PTT bubble con waveform en lugar de "audio file").
4. Caller manda al bridge con ``bridge_client.send_ptt``.

`transcribe_only=True` permite que el frontend pida transcript sin
mandar — útil para preview-then-confirm UX y para "dictá texto" sin
publicar voice.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger("rag.wa.voice")


def _temp_dir() -> Path:
    p = Path.home() / ".local/share/obsidian-rag/wa-media/tmp-voice"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_blob_to_tmp(data: bytes, suffix: str = ".webm") -> Path:
    """Persiste el blob recibido del browser a un archivo temporal.
    El caller debe llamar ``cleanup(path)`` cuando termine.
    """
    if not data:
        raise ValueError("empty audio blob")
    name = f"{uuid.uuid4().hex}{suffix}"
    p = _temp_dir() / name
    p.write_bytes(data)
    return p


def cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            if p and p.is_file():
                p.unlink()
        except OSError:
            pass


def transcribe_blob(data: bytes, *, language: str = "es", suffix: str = ".webm") -> dict:
    """Transcribe un blob de audio crudo del browser.

    Devuelve ``{text, language, duration_s, model, cached}`` igual que
    ``rag.whisper.transcribe_audio``. Limpia el tmp file al final.
    """
    from rag.whisper import transcribe_audio  # noqa: PLC0415

    tmp = write_blob_to_tmp(data, suffix=suffix)
    try:
        return transcribe_audio(tmp, language=language)
    finally:
        cleanup(tmp)


def to_opus(in_path: Path) -> Path | None:
    """Convierte un archivo audio a `.ogg/opus 32kbps mono` (formato PTT
    de WhatsApp). Devuelve el path del `.ogg` o ``None`` si ffmpeg falla.
    """
    out_path = _temp_dir() / f"{uuid.uuid4().hex}.ogg"
    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(in_path),
        "-c:a", "libopus",
        "-b:a", "32k",
        "-ac", "1",
        "-ar", "48000",
        "-application", "voip",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("ffmpeg to_opus failed: %s", e)
        return None
    if proc.returncode != 0:
        logger.warning("ffmpeg returncode=%s stderr=%s", proc.returncode, proc.stderr[:300])
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    if not out_path.is_file() or out_path.stat().st_size < 200:
        return None
    return out_path
