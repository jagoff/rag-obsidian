"""Spotify ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Snapshots ``current_user_recently_played`` (daily) plus a weekly-refreshed
``_top.md`` (top tracks + top artists, ``short_term`` window) via spotipy
OAuth. Outputs land in ``99-obsidian/99-AI/external-ingest/Spotify/`` so
the regular ``_run_index`` rglob absorbs them.

Silent-fail contract: every helper returns ``None`` /
``{ok: False, reason: "..."}`` instead of raising. ``_atomic_write_if_changed``
and ``_etl_log_swallow`` are lazy-imported from ``rag.cross_source_etls`` to
avoid circular import.

Naming: file is ``spotify_etl.py`` (not ``spotify.py``) to avoid shadowing
the ``spotipy`` import / the existing top-level ``spotify_local`` integration
when grepping by basename.

Tests (``tests/test_external_etls.py``) monkeypatch ``rag._SPOTIFY_TOKEN_PATH``,
``rag._SPOTIFY_CREDS_PATH``, and ``rag._spotify_client`` on the top-level
``rag`` module — the helpers re-resolve those symbols at call time via
``sys.modules.get("rag")`` so the patches propagate regardless of where the
binding lives.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_SPOTIFY_VAULT_SUBPATH",
    "_SPOTIFY_CREDS_PATH",
    "_SPOTIFY_TOKEN_PATH",
    "_SPOTIFY_SCOPES",
    "_SPOTIFY_TOP_TTL_DAYS",
    "_spotify_client",
    "_sync_spotify_notes",
]

_SPOTIFY_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Spotify"
_SPOTIFY_CREDS_PATH = Path.home() / ".config/obsidian-rag/spotify_client.json"
_SPOTIFY_TOKEN_PATH = Path.home() / ".config/obsidian-rag/spotify_token.json"
_SPOTIFY_SCOPES = "user-read-recently-played user-top-read"
_SPOTIFY_TOP_TTL_DAYS = 7  # weekly refresh of _top.md


def _spotify_client(allow_interactive: bool = True) -> "spotipy.Spotify | None":
    """Return an authenticated `spotipy.Spotify` instance, or None."""
    from rag.cross_source_etls import _etl_log_swallow

    _rag = sys.modules.get("rag")
    _creds_path = getattr(_rag, "_SPOTIFY_CREDS_PATH", _SPOTIFY_CREDS_PATH)
    _token_path = getattr(_rag, "_SPOTIFY_TOKEN_PATH", _SPOTIFY_TOKEN_PATH)
    if not _creds_path.is_file():
        return None
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        return None
    try:
        creds = json.loads(_creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cid = creds.get("client_id")
    secret = creds.get("client_secret")
    redirect = creds.get("redirect_uri", "http://localhost:8888/callback")
    if not (cid and secret):
        return None
    open_browser = bool(allow_interactive and sys.stdin.isatty())
    auth = SpotifyOAuth(
        client_id=cid, client_secret=secret, redirect_uri=redirect,
        scope=_SPOTIFY_SCOPES, cache_path=str(_token_path),
        open_browser=open_browser,
    )
    try:
        token = auth.get_cached_token()
        if not token or auth.is_token_expired(token):
            if not open_browser:
                return None
            token = auth.get_access_token(as_dict=True)
        if not token:
            return None
        try:
            if _token_path.is_file():
                os.chmod(_token_path, 0o600)
        except OSError as exc:
            _etl_log_swallow("spotify_token_chmod", exc)
        return spotipy.Spotify(auth=token["access_token"])
    except Exception as exc:
        _etl_log_swallow("spotify_oauth_token", exc)
        return None


def _sync_spotify_notes(vault_root: Path, max_recent: int = 50) -> dict:
    """Snapshot Spotify recently-played + (weekly) top tracks."""
    from rag.cross_source_etls import _atomic_write_if_changed

    _rag = sys.modules.get("rag")
    _token_path = getattr(_rag, "_SPOTIFY_TOKEN_PATH", _SPOTIFY_TOKEN_PATH)
    _sp_client = getattr(_rag, "_spotify_client", _spotify_client)
    if not _token_path.is_file():
        return {"ok": False, "reason": "no_spotify_token"}
    sp = _sp_client(allow_interactive=False)
    if sp is None:
        return {"ok": False, "reason": "no_spotify_credentials"}

    try:
        recent = sp.current_user_recently_played(limit=max_recent)
    except Exception as exc:
        return {"ok": False, "reason": f"recent_failed: {str(exc)[:120]}"}

    items = recent.get("items") or []
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    fm = [
        "---",
        "source: spotify",
        f"snapshot_date: {today}",
        f"track_count: {len(items)}",
        "tags:",
        "- spotify",
        "- system-snapshot",
        "---",
        "",
        f"# Spotify recently played — {today}",
        "",
    ]
    for it in items:
        track = it.get("track") or {}
        name = track.get("name") or "(sin título)"
        artists = ", ".join(a.get("name", "?") for a in (track.get("artists") or []))
        album = (track.get("album") or {}).get("name", "")
        played = (it.get("played_at") or "").replace("T", " ").split(".")[0]
        url = (track.get("external_urls") or {}).get("spotify", "")
        link = f"[{name}]({url})" if url else name
        fm.append(f"- `{played}` {link} — {artists}{f' · _{album}_' if album else ''}")
    body = "\n".join(fm) + "\n"
    target = vault_root / _SPOTIFY_VAULT_SUBPATH / f"{today}.md"
    written_recent = _atomic_write_if_changed(target, body)

    top_target = vault_root / _SPOTIFY_VAULT_SUBPATH / "_top.md"
    written_top = 0
    needs_top = (
        not top_target.is_file()
        or (time.time() - top_target.stat().st_mtime) > _SPOTIFY_TOP_TTL_DAYS * 86400
    )
    if needs_top:
        try:
            top_tracks = sp.current_user_top_tracks(limit=20, time_range="short_term")
            top_artists = sp.current_user_top_artists(limit=20, time_range="short_term")
        except Exception:
            top_tracks = top_artists = None
        if top_tracks and top_artists:
            t_items = top_tracks.get("items") or []
            a_items = top_artists.get("items") or []
            tfm = [
                "---",
                "source: spotify-top",
                f"refreshed_date: {today}",
                "window: short_term (4 weeks)",
                "tags:",
                "- spotify",
                "- system-snapshot",
                "---",
                "",
                "# Spotify Top — últimas 4 semanas",
                "",
                f"## Top tracks ({len(t_items)})",
                "",
            ]
            for t in t_items:
                artists = ", ".join(a.get("name", "?") for a in (t.get("artists") or []))
                url = (t.get("external_urls") or {}).get("spotify", "")
                name = t.get("name", "?")
                link = f"[{name}]({url})" if url else name
                tfm.append(f"- {link} — {artists}")
            tfm += ["", f"## Top artists ({len(a_items)})", ""]
            for a in a_items:
                url = (a.get("external_urls") or {}).get("spotify", "")
                name = a.get("name", "?")
                genres = ", ".join((a.get("genres") or [])[:3])
                link = f"[{name}]({url})" if url else name
                tfm.append(f"- {link}{f' · {genres}' if genres else ''}")
            top_body = "\n".join(tfm) + "\n"
            if _atomic_write_if_changed(top_target, top_body):
                written_top = 1

    return {
        "ok": True,
        "files_written": (1 if written_recent else 0) + written_top,
        "recently_played": len(items),
        "refreshed_top": bool(written_top),
        "target": _SPOTIFY_VAULT_SUBPATH,
    }
