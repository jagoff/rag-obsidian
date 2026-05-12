"""Avatares para `/wa` sacados de Apple Contacts.app.

Flujo en 3 capas con caches:

1. **Contact index** (``~/.local/share/obsidian-rag/wa-contacts.json``) —
   snapshot de ``name`` + ``phones[]`` de todos los Apple Contacts que
   tienen foto. Refresh on-demand cuando: (a) no existe, (b) tiene >24h,
   (c) los lookups recientes empiezan a fallar y se fuerza. ~50-200ms
   para 1000 contactos.

2. **Match en Python** — fold de acentos + lowercase + substring. Se
   intenta primero por ``digits`` del teléfono (matchea aunque Contacts
   guarde el número con espacios/dashes/+); si no, por nombre.

3. **Image fetch** — segundo AppleScript específico que pide
   ``first person whose name is "<exacto>"`` y escribe la imagen a
   ``~/.local/share/obsidian-rag/wa-avatars/<safe_jid>.jpg``.

Por qué 2 scripts en lugar de uno: el AppleScript con ``use framework
"Foundation"`` activado rompe ``POSIX file`` (error `-1700` "Can't make
current application into type «class fsrf»"). Separando enum (sin
framework, salida stdout) del fetch (sin framework, image-to-disk)
evitamos esa colisión.

Cache de avatares: hit 7 días, miss 24h.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import unicodedata
from pathlib import Path

logger = logging.getLogger("rag.wa.avatars")

_AVATAR_DIR = Path.home() / ".local/share/obsidian-rag/wa-avatars"
_INDEX_PATH = _AVATAR_DIR / "wa-contacts.json"
_CACHE_TTL_S = 7 * 24 * 3600
_MISS_TTL_S = 24 * 3600
_INDEX_TTL_S = 24 * 3600

_SAFE_NAME_RE = re.compile(r"^[\w\s\-_.,áéíóúüñÁÉÍÓÚÜÑ()]+$")


def _safe_jid(jid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]", "_", jid)


def _ensure_dir() -> None:
    _AVATAR_DIR.mkdir(parents=True, exist_ok=True)


def _fold(s: str) -> str:
    """NFD fold + drop combining marks + lowercase. "Mónica" → "monica"."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def _strip_emoji(s: str) -> str:
    out: list[str] = []
    for ch in s:
        cat = ord(ch)
        if ch.isalpha() or ch.isspace() or ch in {"-", "'", "."}:
            out.append(ch)
        elif cat < 0x2000:
            pass
    return "".join(out).strip()


def _digits_from_jid(jid: str) -> str | None:
    if not jid or "@" not in jid:
        return None
    user, server = jid.split("@", 1)
    if server != "s.whatsapp.net":
        return None
    digits = re.sub(r"\D", "", user)
    return digits if len(digits) >= 7 else None


# ─────────────────────────────────────────────────────────────
# Step 1: enum contacts → snapshot JSON
# ─────────────────────────────────────────────────────────────

# Enum sólo names — agregar phones cuadruplica el tiempo (`phones of p` es
# slow on Contacts.app). Para match por phone usamos un fetch on-demand
# por contacto candidato si el name match falla.
_ENUM_SCRIPT = '''tell application "Contacts"
  set _out to ""
  repeat with p in every person
    if image of p is not missing value then
      set _out to _out & (name of p) & linefeed
    end if
  end repeat
  return _out
end tell'''


def _enum_contacts() -> list[dict]:
    """Devuelve [{name}] de todos los contacts con foto.

    Silent-fail: timeout / permission denied / parse error → []. Demora
    ~15s en una contact book de ~3k personas — es la operación más cara
    del módulo, por eso cacheamos 24h.
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", _ENUM_SCRIPT],
            capture_output=True, text=True, timeout=45,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("enum contacts failed: %s", e)
        return []
    if proc.returncode != 0:
        logger.warning("enum returncode=%s stderr=%s", proc.returncode, proc.stderr[:200])
        return []
    out: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        name = line.strip()
        if name:
            out.append({"name": name})
    return out


def _index_load() -> list[dict] | None:
    if not _INDEX_PATH.is_file():
        return None
    try:
        age = time.time() - _INDEX_PATH.stat().st_mtime
        if age > _INDEX_TTL_S:
            return None
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _index_save(rows: list[dict]) -> None:
    _ensure_dir()
    try:
        _INDEX_PATH.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("index save failed: %s", e)


def _index_get(force_refresh: bool = False) -> list[dict]:
    """Carga el index cached o lo refresca."""
    if not force_refresh:
        cached = _index_load()
        if cached is not None:
            return cached
    fresh = _enum_contacts()
    if fresh:
        _index_save(fresh)
    return fresh


# ─────────────────────────────────────────────────────────────
# Step 2: matching en Python
# ─────────────────────────────────────────────────────────────


def _match_contact(index: list[dict], digits: str | None, name: str | None) -> str | None:
    """Encuentra el name exacto del contact que matchea por name.

    El index no guarda phones (es lento de enumerar); para JIDs
    ``s.whatsapp.net`` confiamos en que el ``chat_name`` del bridge —
    que viene del push-name de WhatsApp — coincida con el name del
    Apple Contact en la mayoría de los casos. Si no, fallback iniciales.

    Devuelve el ``name`` exacto (listo para pasar al script fetch) o None.
    """
    if not index or not name:
        return None
    cleaned = _strip_emoji(name).strip()
    if not cleaned:
        return None
    q = _fold(cleaned)
    # Exact fold-match primero (más estricto)
    for c in index:
        if _fold(c["name"]) == q:
            return c["name"]
    # Substring match después (solo si la query tiene 4+ chars
    # para evitar matches accidentales tipo "Ana" en "Mariana")
    if len(q) >= 4:
        for c in index:
            if q in _fold(c["name"]):
                return c["name"]
    return None


# ─────────────────────────────────────────────────────────────
# Step 3: image fetch
# ─────────────────────────────────────────────────────────────


def _build_fetch_script(name_exact: str, out_path: str) -> str | None:
    if not _SAFE_NAME_RE.match(name_exact):
        return None
    esc = name_exact.replace("\\", "\\\\").replace('"', '\\"')
    return f'''tell application "Contacts"
  set _img to missing value
  try
    set _people to (every person whose name is "{esc}")
    if (count of _people) > 0 then
      set _p to first item of _people
      if image of _p is not missing value then
        set _img to image of _p
      end if
    end if
  end try
end tell

if _img is missing value then return "MISS"

try
  set _f to open for access (POSIX file "{out_path}") with write permission
  set eof of _f to 0
  write _img to _f
  close access _f
  return "OK"
on error errMsg
  try
    close access (POSIX file "{out_path}")
  end try
  return "ERR:" & errMsg
end try'''


def _fetch_image(name_exact: str, out_path: Path) -> bool:
    """Fetch image vía AppleScript a un path temporal (.raw) y después
    recomprime con ``sips`` a JPEG 256x256.

    Apple Contacts a veces devuelve TIFF sin compresión (11 MB para una
    foto de 1024x1024). Sin downscale el sidebar tarda 30s en cargar.
    ``sips`` viene con macOS y es ~50ms por imagen.
    """
    raw_path = out_path.with_suffix(".raw")
    script = _build_fetch_script(name_exact, str(raw_path))
    if script is None:
        return False
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("fetch image timeout: %s", e)
        return False
    if proc.returncode != 0 or (proc.stdout or "").strip() != "OK":
        # cleanup posible archivo parcial
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    if not raw_path.is_file() or raw_path.stat().st_size < 100:
        return False

    # Convert + downscale.
    try:
        conv = subprocess.run(
            [
                "/usr/bin/sips",
                "-Z", "256",
                "-s", "format", "jpeg",
                "-s", "formatOptions", "75",
                str(raw_path),
                "--out", str(out_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("sips conversion failed: %s", e)
        # Fallback: usar la raw como está (browser puede manejar TIFF? no
        # con todos los browsers, pero mejor que nada).
        try:
            raw_path.rename(out_path)
            return True
        except OSError:
            return False
    finally:
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
    if conv.returncode != 0:
        logger.warning("sips returncode=%s stderr=%s", conv.returncode, conv.stderr[:200])
        return False
    try:
        return out_path.is_file() and out_path.stat().st_size > 100
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def get_avatar_path(jid: str, chat_name: str | None = None) -> Path | None:
    """Devuelve el path local de la foto del contacto, o ``None``.

    Cache: hits 7 días, misses 24h. El index de contactos se refresca
    cada 24h.
    """
    if not jid:
        return None
    _ensure_dir()
    safe = _safe_jid(jid)
    out_path = _AVATAR_DIR / f"{safe}.jpg"
    miss_path = _AVATAR_DIR / f"{safe}.miss"
    now = time.time()

    # Migración: si existe archivo con @ en el nombre (bug anterior), renombrarlo
    legacy_path = _AVATAR_DIR / f"{jid}.jpg"
    if legacy_path.is_file() and not out_path.is_file():
        try:
            legacy_path.rename(out_path)
            logger.info("Migrated avatar: %s -> %s", legacy_path.name, out_path.name)
        except OSError as e:
            logger.warning("Migration failed for %s: %s", legacy_path, e)

    if out_path.is_file():
        try:
            if now - out_path.stat().st_mtime < _CACHE_TTL_S:
                return out_path
        except OSError:
            pass

    # Match contra Apple Contacts ANTES de chequear el .miss cache:
    # el .miss queda keyed por jid sin chat_name, así que un miss
    # viejo (de cuando el lookup no tenía chat_name) bloqueaba
    # fetches válidos cuando el chat_name aparecía después. Si HAY
    # match en el index, forzamos el fetch ignorando el miss.
    digits = _digits_from_jid(jid)
    index = _index_get()
    name_exact = _match_contact(index, digits, chat_name)

    if name_exact and _fetch_image(name_exact, out_path):
        try:
            miss_path.unlink(missing_ok=True)
        except OSError:
            pass
        return out_path

    # Solo respetar el .miss cache cuando NO había match — si nunca
    # vamos a poder resolver Apple Contacts, evitar re-correr osascript
    # cada 200ms del scroll.
    if not name_exact and miss_path.is_file():
        try:
            if now - miss_path.stat().st_mtime < _MISS_TTL_S:
                return None
        except OSError:
            pass

    try:
        miss_path.touch()
    except OSError:
        pass
    if out_path.is_file() and out_path.stat().st_size <= 100:
        try:
            out_path.unlink()
        except OSError:
            pass
    return None
