"""Morning voice brief — text → audio file → WhatsApp media push.

The morning brief is generated as text by `morning()` in
``rag/__init__.py``. This module adds an optional audio rendering path so
the user can listen to the brief while getting ready instead of reading
it. Anticipatory Phase 2.C (see ``plans/anticipatory-agent.md §2.C``).

Pipeline (default, all-local on macOS — sin deps Python nuevas):

  1. Strip markdown noise (frontmatter, wikilinks, code, citations,
     headings, bullets, the ``_brief:<path>_`` footer).
  2. Render with ``say -v <voice> --file-format=AIFF`` to a temp AIFF.
  3. Convert AIFF → OGG/Opus mono 24 kbps via ``ffmpeg`` — el formato que
     usa WhatsApp para voice notes; el bridge (``whatsapp-mcp/whatsapp-bridge``)
     acepta cualquier media via ``media_path`` y manda OGG como audio
     nativo.
  4. Cache to ``~/.local/share/obsidian-rag/voice_briefs/YYYY-MM-DD-<kind>.ogg``
     idempotente — un segundo ``rag morning --voice`` el mismo día reusa
     el archivo (útil para retries / debugging).

Falls back to ``None`` silently if ``say`` no está disponible (CI Linux),
si ``ffmpeg`` falta y AIFF no se puede entregar, o si el audio resulta
mayor a 5 MB (cap de WhatsApp para voice notes). El text brief siempre
llega regardless — la integración en ``_brief_push_to_whatsapp`` es
"audio if-available, text always".

Env vars:
  - ``TTS_VOICE`` — voz para ``say`` (default ``Mónica``, la rioplatense
    de macOS). Cualquier voz instalada funciona; ``say -v ?`` lista las
    disponibles.
  - ``RAG_VOICE_BRIEF_DIR`` — override del directorio de cache (default
    ``~/.local/share/obsidian-rag/voice_briefs/``). Útil en tests.
  - ``RAG_VOICE_BRIEF_TTL_DAYS`` — TTL para ``cleanup_old_voice_briefs``
    (default 30 días). ``rag maintenance`` los borra.
  - ``RAG_MORNING_VOICE`` — si ``1``/``true``/``yes``, ``rag morning``
    actúa como si llevara ``--voice``. Pensado para el plist (que
    configura el daemon launchd).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

_VOICE_BRIEF_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "voice_briefs"

#: Hard cap on speech text. Past this point ``say`` toma minutos y el
#: audio supera fácil los 5 MB que WhatsApp permite. Se trim al primer
#: ``_MAX_TEXT_CHARS`` y se concatena ``...``.
_MAX_TEXT_CHARS = 4000

#: WhatsApp voice note hard cap. Si el OGG resulta más grande, devolvemos
#: ``None`` y el caller cae a text-only — mejor que un send que el bridge
#: rechaza.
_MAX_AUDIO_BYTES = 5 * 1024 * 1024

#: Voz rioplatense default de macOS. ``say -v ?`` muestra alternativas
#: (``Diego`` también es es-AR pero masculino, ``Paulina`` es es-MX).
_DEFAULT_VOICE = "Mónica"

# ── Markdown stripping for speech ────────────────────────────────────────────
# El brief lleva markdown que ``say`` lee literal: backticks, headings con
# ``#``, wikilinks ``[[Foo]]``, bullets ``-``, links ``[label](path.md)``,
# y el footer ``_brief:<path>_``. Sacamos todo y dejamos prosa limpia.
_RE_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_RE_INLINE_CODE = re.compile(r"`[^`]+`")
_RE_WIKILINK = re.compile(r"\[\[([^\]\|]+?)(?:\|([^\]]+))?\]\]")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+|[^)]+\.md)\)")
_RE_BARE_BRACKET_PATH = re.compile(r"\[([^\]\n]+\.md)\]")
_RE_HEADING_HASH = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_RE_BOLD_ITALIC = re.compile(r"(?<!\w)([*_]{1,3})([^*_\n]+)\1(?!\w)")
_RE_HORIZONTAL_RULE = re.compile(r"^-{3,}$", re.MULTILINE)
_RE_BRIEF_FOOTER = re.compile(r"_brief:[^_\n]+_", re.MULTILINE)
_RE_OBSIDIAN_URL = re.compile(r"\(?obsidian://[^)\s]+\)?")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")


def _silent_log(where: str, exc: BaseException) -> None:
    """Wrapper lazy-import alrededor de ``rag._silent_log`` para evitar
    circular imports — este módulo se importa desde ``rag/__init__.py``
    y al revés."""
    try:
        from rag import _silent_log as _sl  # noqa: PLC0415
        _sl(where, exc)
    except Exception:
        pass


def get_voice_brief_dir() -> Path:
    """Directorio de cache de los voice briefs. Honor ``RAG_VOICE_BRIEF_DIR``."""
    override = os.environ.get("RAG_VOICE_BRIEF_DIR")
    if override:
        return Path(override).expanduser()
    return _VOICE_BRIEF_DIR


def get_voice() -> str:
    """Voz configurada (env ``TTS_VOICE``, default ``Mónica``)."""
    v = (os.environ.get("TTS_VOICE") or "").strip()
    return v or _DEFAULT_VOICE


def strip_markdown_for_speech(text: str) -> str:
    """Limpia el markdown del brief para que ``say`` lo lea como prosa.

    Removes:
      - frontmatter YAML al inicio (``---\\n...\\n---``)
      - fenced + inline code blocks (no querés escuchar backticks)
      - wikilinks ``[[Title]]`` / ``[[path|Alias]]`` → keeps el label
      - markdown links ``[Label](url-or-path)`` → keeps el label
      - bare bracket paths ``[path.md]`` → drop
      - obsidian:// URLs → drop
      - ``_brief:<path>_`` footer → drop (es metadata para el listener)
      - heading markers ``#`` / ``##`` / ...
      - bullet markers ``-`` / ``*`` / ``+``
      - bold/italic emphasis ``**foo**`` / ``_foo_`` → keep contenido
      - horizontal rules ``---`` standalone
      - colapsa whitespace múltiple
    """
    out = text or ""
    out = _RE_FRONTMATTER.sub("", out)
    out = _RE_FENCED_CODE.sub(" ", out)
    out = _RE_INLINE_CODE.sub(" ", out)
    out = _RE_WIKILINK.sub(lambda m: m.group(2) or m.group(1), out)
    out = _RE_MD_LINK.sub(lambda m: m.group(1), out)
    out = _RE_BARE_BRACKET_PATH.sub("", out)
    out = _RE_OBSIDIAN_URL.sub("", out)
    out = _RE_BRIEF_FOOTER.sub("", out)
    out = _RE_HEADING_HASH.sub("", out)
    out = _RE_BULLET.sub("", out)
    out = _RE_BOLD_ITALIC.sub(lambda m: m.group(2), out)
    out = _RE_HORIZONTAL_RULE.sub("", out)
    out = _RE_MULTI_SPACE.sub(" ", out)
    out = _RE_MULTI_NEWLINE.sub("\n\n", out)
    return out.strip()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _say_available() -> bool:
    return shutil.which("say") is not None


def _say_to_aiff(text: str, voice: str, out: Path) -> bool:
    """Llama a ``say`` para escribir AIFF. Silent-fail → ``False``."""
    if not _say_available():
        return False
    try:
        subprocess.run(
            ["say", "-v", voice, "--file-format=AIFF", "-o", str(out), text],
            check=True, capture_output=True, timeout=180,
        )
        return out.is_file() and out.stat().st_size > 0
    except Exception as exc:
        _silent_log("voice_brief_say", exc)
        return False


def _aiff_to_ogg(aiff_path: Path, ogg_path: Path) -> bool:
    """Convert AIFF → OGG/Opus mono 24kbps. ``ffmpeg``-driven; silent-fail."""
    if not _ffmpeg_available():
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff_path),
             "-c:a", "libopus", "-b:a", "24k", "-ac", "1", "-ar", "16000",
             str(ogg_path)],
            check=True, capture_output=True, timeout=120,
        )
        return ogg_path.is_file() and ogg_path.stat().st_size > 0
    except Exception as exc:
        _silent_log("voice_brief_ffmpeg", exc)
        return False


def _audio_path_for(date_str: str, kind: str = "morning",
                    output_dir: Path | None = None) -> Path:
    base = output_dir if output_dir is not None else get_voice_brief_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{date_str}-{kind}.ogg"


def synthesize_brief_audio(
    text: str,
    output_dir: Path | None = None,
    *,
    kind: str = "morning",
    date_str: str | None = None,
    voice: str | None = None,
) -> Path | None:
    """Render ``text`` a un audio file. Devuelve ``Path`` o ``None`` si TTS falló.

    Output cacheado en ``<output_dir>/<YYYY-MM-DD>-<kind>.ogg``. Si el
    archivo ya existe con contenido, se devuelve sin regenerar
    (idempotente — útil cuando ``morning`` se reruns el mismo día durante
    debugging).

    Caps:
      - ``len(text) > 4000`` → trim al primer 4000 chars + ``...``.
      - audio resultante > 5 MB → log warning + return ``None`` (WA limit).

    Silent-fail: cualquier subprocess error (``say`` missing, ``ffmpeg``
    missing, codec error, FS write error) → ``None`` y log via
    ``_silent_log``.

    Args:
      text: el cuerpo del brief — markdown completo, se limpia internamente.
      output_dir: override del directorio (default ``get_voice_brief_dir()``).
      kind: sufijo del filename (``morning``/``evening``/``digest``).
      date_str: ``YYYY-MM-DD``; default ``today``.
      voice: voz override; default ``get_voice()``.

    Returns:
      ``Path`` al ``.ogg`` (o ``.aiff`` fallback si no hay ffmpeg) o ``None``.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if not _say_available():
        return None

    voice = voice or get_voice()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = _audio_path_for(date_str, kind, output_dir)

    # Idempotency: reuse cached file when present + non-empty.
    try:
        if out_path.is_file() and out_path.stat().st_size > 0:
            return out_path
    except OSError:
        pass

    clean = strip_markdown_for_speech(text)
    if len(clean) > _MAX_TEXT_CHARS:
        clean = clean[:_MAX_TEXT_CHARS].rstrip() + "..."
    if not clean:
        return None

    # AIFF temp file — al lado del .ogg final para que cleanup sea simple.
    aiff_path = out_path.with_suffix(".aiff")
    if not _say_to_aiff(clean, voice, aiff_path):
        try:
            aiff_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    # AIFF → OGG/Opus. Si ffmpeg falta, fallback a AIFF (.aiff) — el
    # bridge igual lo manda como media; pesa más pero funciona.
    final_path: Path | None = None
    if _ffmpeg_available():
        if _aiff_to_ogg(aiff_path, out_path):
            final_path = out_path
        try:
            aiff_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        # Sin ffmpeg → dejamos el AIFF como artifact final.
        final_path = aiff_path

    if final_path is None or not final_path.is_file():
        return None

    # 5 MB cap — over the limit → drop, fall back to text-only.
    try:
        size = final_path.stat().st_size
    except OSError:
        return None
    if size > _MAX_AUDIO_BYTES:
        try:
            final_path.unlink(missing_ok=True)
        except Exception:
            pass
        _silent_log(
            "voice_brief_oversize",
            ValueError(f"audio {size} > {_MAX_AUDIO_BYTES} bytes"),
        )
        return None

    return final_path


def send_audio_to_whatsapp(jid: str, audio_path: Path) -> bool:
    """POSTea ``audio_path`` al bridge local de WhatsApp como ``media_path``.

    Returns ``True`` en 2xx. El bridge acepta
    ``{recipient, message?, media_path}`` (ver docstring de
    ``_whatsapp_send_to_jid`` en ``rag/integrations/whatsapp.py``); cuando
    ``media_path`` está set + apunta a un OGG/Opus, el mensaje sale como
    voice note nativo.

    No usamos ``_ambient_whatsapp_send`` porque ese helper hardcodea text
    payload + prefixa el U+200B anti-loop (que un audio no necesita y no
    se puede inyectar en bytes binarios). Hacemos el POST directo acá.

    Silent-fail en cualquier excepción (bridge down, archivo missing,
    timeout) → ``False``. El caller manda solo texto en ese caso.
    """
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    if not audio_path or not audio_path.is_file():
        return False
    try:
        from rag import AMBIENT_WHATSAPP_BRIDGE_URL  # noqa: PLC0415
    except Exception:
        AMBIENT_WHATSAPP_BRIDGE_URL = "http://localhost:8080/api/send"
    body = json.dumps({
        "recipient": jid,
        "media_path": str(audio_path),
    }).encode("utf-8")
    req = urllib.request.Request(
        AMBIENT_WHATSAPP_BRIDGE_URL, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        _silent_log("voice_brief_send_audio", exc)
        return False


def cleanup_old_voice_briefs(*, ttl_days: int | None = None,
                             output_dir: Path | None = None) -> dict:
    """Borra audios > ``ttl_days``. Returns ``{deleted, bytes_freed, errors}``.

    Default TTL: 30 días (override env ``RAG_VOICE_BRIEF_TTL_DAYS``). Lo
    llama ``run_maintenance`` para que los audios efímeros no se acumulen
    — un brief de 2 minutos pesa ~100 KB pero después de 6 meses son 60+
    archivos huérfanos sin valor histórico (el texto en el vault es la
    fuente de verdad).
    """
    if ttl_days is None:
        try:
            ttl_days = int(os.environ.get("RAG_VOICE_BRIEF_TTL_DAYS", "30"))
        except ValueError:
            ttl_days = 30
    base = output_dir if output_dir is not None else get_voice_brief_dir()
    if not base.is_dir():
        return {"deleted": 0, "bytes_freed": 0, "errors": []}
    cutoff = time.time() - ttl_days * 86400
    deleted = 0
    bytes_freed = 0
    errors: list[str] = []
    for f in base.iterdir():
        if not f.is_file():
            continue
        try:
            if f.stat().st_mtime < cutoff:
                size = f.stat().st_size
                f.unlink()
                deleted += 1
                bytes_freed += size
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f.name}: {e}")
    return {"deleted": deleted, "bytes_freed": bytes_freed, "errors": errors}
