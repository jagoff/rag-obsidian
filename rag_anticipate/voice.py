"""Voice brief utility — convierte text a audio file usando /api/tts.

Phase 2 del Anticipatory Agent: en lugar de WhatsApp text, podemos
mandar voice notes con el brief leído. Stub utility, no integrado al
orchestrator todavía. La integración futura sería: en lugar de pasar
`message: str` a proactive_push, pasarlo a este helper, recibir un
audio_path, y mandar el voice note.

Requirements:
- /api/tts endpoint del web server debe estar corriendo (default localhost).
- ffmpeg disponible para conversiones (opcional, fallback a wav).
"""

from __future__ import annotations
from pathlib import Path


VOICE_OUT_DIR = Path.home() / ".local/share/obsidian-rag/voice_briefs"


def _tts_endpoint() -> str:
    """URL del endpoint TTS local."""
    import os
    base = os.environ.get("RAG_WEB_BASE_URL", "http://127.0.0.1:8765")
    return f"{base}/api/tts"


def text_to_audio(text: str, *, voice: str = "Monica", out_dir: Path | None = None) -> Path | None:
    """Convierte texto a audio file via /api/tts.

    Args:
        text: el texto a convertir.
        voice: voz a usar (default 'Monica' = español ES). Otras opciones según endpoint.
        out_dir: dir de output. Default ~/.local/share/obsidian-rag/voice_briefs/.

    Returns:
        Path al archivo audio generado, o None si falla.
    """
    if not text or not text.strip():
        return None
    out_dir = out_dir or VOICE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        import json as _json
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        out_path = out_dir / f"brief-{h}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path  # cache hit
        req = urllib.request.Request(
            _tts_endpoint(),
            data=_json.dumps({"text": text, "voice": voice}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            audio_bytes = resp.read()
        if not audio_bytes:
            return None
        out_path.write_bytes(audio_bytes)
        return out_path
    except Exception:
        return None


def is_tts_available() -> bool:
    """True si el web server con /api/tts está accesible. Probe rápido (timeout 2s)."""
    try:
        import urllib.request
        req = urllib.request.Request(_tts_endpoint().rsplit("/api/tts", 1)[0] + "/")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status < 500
    except Exception:
        return False


def cleanup_old_briefs(max_age_days: int = 7, *, out_dir: Path | None = None) -> int:
    """Elimina audio briefs >max_age_days. Returns count eliminados."""
    out_dir = out_dir or VOICE_OUT_DIR
    if not out_dir.is_dir():
        return 0
    import time
    cutoff = time.time() - max_age_days * 86400
    count = 0
    for p in out_dir.glob("brief-*.wav"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                count += 1
        except Exception:
            continue
    return count
