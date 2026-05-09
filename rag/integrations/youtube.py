"""YouTube transcripts ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Walks YouTube daily notes under ``99-obsidian/99-AI/external-ingest/YouTube/``
(populated by ``_sync_chrome_history``), pulls (video_id, title) pairs, and
fetches transcripts via ``youtube-transcript-api`` into per-video markdown
under ``YouTube/transcripts/<video_id>.md`` so the regular ``_run_index``
rglob absorbs them.

Silent-fail contract: helpers return ``None`` / ``{ok: False, reason: "..."}``
instead of raising. ``_atomic_write_if_changed`` + ``_etl_log_swallow`` are
lazy-imported from ``rag.cross_source_etls`` to avoid circular import.

Circuit-breaker: when YouTube blocks the IP (``IpBlocked``), the cooldown
state is persisted to ``ragvec.db::rag_yt_transcript_cooldown`` with an
exponential backoff (4h → 8h → 16h → 24h). Subsequent runs short-circuit
with ``reason: "yt_ip_cooldown_active"`` until the window expires.

Tests (``tests/test_external_etls.py``) monkeypatch
``rag._fetch_yt_transcript_for_index`` on the top-level ``rag`` module —
``_sync_youtube_transcripts`` re-resolves the symbol at call time via
``sys.modules.get("rag")`` so the patch propagates regardless of where the
function lives.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_YOUTUBE_VAULT_SUBPATH",
    "_YOUTUBE_TRANSCRIPTS_SUBPATH",
    "_YOUTUBE_WATCH_RE",
    "_YT_TRANSCRIPT_LANG_PRIORITY",
    "_YT_TRANSCRIPT_BATCH",
    "_YT_VIDEO_ID_RE",
    "_YT_IP_BLOCKED_COOLDOWNS_SECONDS",
    "_check_yt_ip_cooldown",
    "_set_yt_ip_cooldown",
    "_collect_youtube_video_ids",
    "_fetch_yt_transcript_for_index",
    "_sync_youtube_transcripts",
]

_YOUTUBE_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/YouTube"
_YOUTUBE_TRANSCRIPTS_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/YouTube/transcripts"
_YOUTUBE_WATCH_RE = re.compile(r"^https?://(www\.|m\.)?youtube\.com/watch\?(?:.*&)?v=([\w\-]+)")

_YT_TRANSCRIPT_LANG_PRIORITY = ("es", "es-419", "en", "en-US")
_YT_TRANSCRIPT_BATCH = 10
_YT_VIDEO_ID_RE = re.compile(r"youtube\.com/watch\?v=([\w\-]{6,})")

# Circuit-breaker backoff schedule (exponencial): 4h → 8h → 16h → 24h
_YT_IP_BLOCKED_COOLDOWNS_SECONDS = (4 * 3600, 8 * 3600, 16 * 3600, 24 * 3600)


def _check_yt_ip_cooldown() -> bool | dict:
    """Retorna True si cooldown está activo (skipar fetch), dict de status si no.
    Retorna `{"active": False}` si no hay cooldown o expiró."""
    try:
        from rag import _ragvec_state_conn
    except ImportError:
        return False  # Sin DB disponible, asumir cooldown inactivo (laxo)

    try:
        with _ragvec_state_conn() as conn:
            # Crear tabla si no existe (idempotent)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_yt_transcript_cooldown ("
                " id INTEGER PRIMARY KEY CHECK (id = 0),"
                " blocked_until_ts REAL NOT NULL,"
                " retry_count INTEGER DEFAULT 0,"
                " last_blocked_at REAL NOT NULL"
                ")"
            )
            row = conn.execute(
                "SELECT blocked_until_ts, retry_count FROM rag_yt_transcript_cooldown WHERE id = 0"
            ).fetchone()

            if not row:
                return {"active": False}

            blocked_until_ts, retry_count = row
            now = time.time()
            if now < blocked_until_ts:
                return {"active": True, "blocked_until_ts": blocked_until_ts, "retry_count": retry_count}
            return {"active": False}
    except Exception:  # pragma: no cover
        # Error al leer DB → asumir cooldown inactivo (laxo, mejor fail-open)
        return False


def _set_yt_ip_cooldown() -> None:
    """Registra un bloqueo IpBlocked y aumenta el cooldown exponencialmente.

    Idempotente dentro de un mismo cooldown window: si el cooldown ya está
    activo, NO bumpear retry_count ni reescribir blocked_until_ts.
    Solo bumpear cuando el cooldown anterior expiró y se entra a uno nuevo.
    """
    try:
        from rag import _ragvec_state_conn
    except ImportError:
        return  # Sin DB, nada que registrar

    try:
        with _ragvec_state_conn() as conn:
            # Crear tabla si no existe (idempotent)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_yt_transcript_cooldown ("
                " id INTEGER PRIMARY KEY CHECK (id = 0),"
                " blocked_until_ts REAL NOT NULL,"
                " retry_count INTEGER DEFAULT 0,"
                " last_blocked_at REAL NOT NULL"
                ")"
            )
            now = time.time()
            row = conn.execute(
                "SELECT blocked_until_ts, retry_count FROM rag_yt_transcript_cooldown WHERE id = 0"
            ).fetchone()

            # Si el cooldown anterior sigue activo, NO bumpear counter
            if row:
                blocked_until_ts, retry_count = row
                if now < blocked_until_ts:
                    # Cooldown activo — retornar sin cambios (idempotente)
                    return
                # Cooldown expiró — bumpear para el próximo nivel
                retry_count = retry_count + 1
            else:
                retry_count = 0

            # Clamp a max cooldown (24h)
            cooldown_idx = min(retry_count, len(_YT_IP_BLOCKED_COOLDOWNS_SECONDS) - 1)
            cooldown_seconds = _YT_IP_BLOCKED_COOLDOWNS_SECONDS[cooldown_idx]
            blocked_until_ts = now + cooldown_seconds

            conn.execute(
                """
                INSERT OR REPLACE INTO rag_yt_transcript_cooldown (id, blocked_until_ts, retry_count, last_blocked_at)
                VALUES (0, ?, ?, ?)
                """,
                (blocked_until_ts, retry_count, now),
            )
            conn.commit()
    except Exception:  # pragma: no cover
        pass


def _collect_youtube_video_ids(vault_root: Path) -> list[tuple[str, str]]:
    """Read recent YouTube daily notes, pull (video_id, title) pairs."""
    yt_dir = vault_root / _YOUTUBE_VAULT_SUBPATH
    if not yt_dir.is_dir():
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for md in sorted(yt_dir.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            m = _YT_VIDEO_ID_RE.search(line)
            if not m:
                continue
            vid = m.group(1)
            if vid in seen:
                continue
            seen.add(vid)
            title_match = re.search(r"\[([^\]]+)\]\(", line)
            title = title_match.group(1).strip() if title_match else vid
            out.append((vid, title))
    return out


def _fetch_yt_transcript_for_index(video_id: str) -> tuple[str, str] | None:
    """Returns (lang, transcript_text) or None on miss.

    Si detecta IpBlocked, registra el cooldown y retorna None.
    """
    from rag.cross_source_etls import _etl_log_swallow

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import IpBlocked
    except ImportError:
        return None
    try:
        api = YouTubeTranscriptApi()
        listing = api.list(video_id)
    except IpBlocked as exc:
        # Circuit-breaker: YouTube está bloqueando esta IP
        _set_yt_ip_cooldown()
        _etl_log_swallow("yt_transcript_list_ip_blocked", exc)
        return None
    except Exception as exc:
        _etl_log_swallow("yt_transcript_list", exc)
        return None
    transcript = None
    chosen_lang = None
    for lang in _YT_TRANSCRIPT_LANG_PRIORITY:
        try:
            transcript = listing.find_transcript([lang])
            chosen_lang = lang
            break
        except Exception:
            # Per-lang miss esperado — el video no tiene ese idioma. NO
            # loggear (sino se llena el log con noise por cada lang miss).
            continue
    if transcript is None:
        try:
            transcript = next(iter(listing))
            chosen_lang = transcript.language_code
        except Exception as exc:
            _etl_log_swallow("yt_transcript_iter_fallback", exc)
            return None
    try:
        fetched = transcript.fetch()
    except Exception as exc:
        _etl_log_swallow("yt_transcript_fetch", exc)
        return None
    snippets = getattr(fetched, "snippets", None) or fetched
    parts = [getattr(s, "text", None) or s.get("text", "") for s in snippets]
    text = " ".join(p for p in parts if p).strip()
    if not text:
        return None
    return chosen_lang or "?", text


def _sync_youtube_transcripts(vault_root: Path, batch: int = _YT_TRANSCRIPT_BATCH) -> dict:
    """For each video referenced in recent YouTube daily notes, fetch its
    transcript once. Caps at `batch` per run.

    Si YouTube está bloqueando la IP (circuit-breaker activo), retorna early
    sin tocar la red.
    """
    from rag.cross_source_etls import _atomic_write_if_changed

    # Check circuit-breaker antes de hacer fetch
    cooldown_status = _check_yt_ip_cooldown()
    if cooldown_status is True or (isinstance(cooldown_status, dict) and cooldown_status.get("active")):
        return {
            "ok": True,
            "files_written": 0,
            "fetched_this_run": 0,
            "failed_this_run": 0,
            "videos_known": 0,
            "reason": "yt_ip_cooldown_active",
            "cooldown_status": cooldown_status if isinstance(cooldown_status, dict) else None,
        }

    _yt_fetch = getattr(sys.modules.get("rag"), "_fetch_yt_transcript_for_index", _fetch_yt_transcript_for_index)
    videos = _collect_youtube_video_ids(vault_root)
    if not videos:
        return {"ok": True, "files_written": 0, "reason": "no_videos"}
    target_dir = vault_root / _YOUTUBE_TRANSCRIPTS_SUBPATH
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    fetched = 0
    failed = 0
    for vid, title in videos:
        target = target_dir / f"{vid}.md"
        if target.is_file():
            continue
        if fetched >= batch:
            break
        result = _yt_fetch(vid)
        fetched += 1
        if not result:
            failed += 1
            # Chequear si el circuit-breaker se activó tras el fetch fallido
            # (IpBlocked activaría el cooldown). Si está activo, romper el loop.
            cooldown_status = _check_yt_ip_cooldown()
            if cooldown_status is True or (isinstance(cooldown_status, dict) and cooldown_status.get("active")):
                break
            continue
        lang, text = result
        url = f"https://www.youtube.com/watch?v={vid}"
        body = (
            "---\n"
            "source: youtube-transcript\n"
            f"video_id: {vid}\n"
            f"language: {lang}\n"
            f"url: {url}\n"
            "tags:\n"
            "- youtube-transcript\n"
            "- system-snapshot\n"
            "---\n\n"
            f"# {title}\n\n"
            f"{url}\n\n"
            f"{text}\n"
        )
        if _atomic_write_if_changed(target, body):
            written += 1
    return {
        "ok": True,
        "files_written": written,
        "fetched_this_run": fetched,
        "failed_this_run": failed,
        "videos_known": len(videos),
        "target": _YOUTUBE_TRANSCRIPTS_SUBPATH,
    }
