"""Spotify integration via AppleScript on the macOS desktop app.

Reemplaza el path de la Web API ([`/me/player/recently-played`](https://developer.spotify.com/documentation/web-api/reference/get-recently-played))
que rompió 2026-04-30 (Spotify devuelve 403 "Active premium subscription
required for the owner of the app" para apps de devs sin Premium —
[gate documentado en developer.spotify.com](https://developer.spotify.com/documentation/web-api/concepts/quota-modes#extension-requests)).

## Surfaces

- `now_playing()` — query AppleScript del desktop app. Devuelve dict con
  `{name, artist, album, state, track_id, duration_ms, position_s, art_url?}`
  o None si Spotify no está corriendo. NO lanza Spotify si no está abierto
  (guard `is running`).
- `record_now_playing()` — captura `now_playing()` y persiste en
  `rag_spotify_log` (telemetry.db). Idempotente: si el último insert es
  el mismo track_id dentro de 5min, hace UPDATE last_seen en vez de un
  nuevo INSERT. Devuelve `{recorded, updated, track?, reason?}` para el
  caller (poller / log).
- `recent_tracks_today(limit)` — lee `rag_spotify_log` filtrado por
  `date = today` ordenado por `first_seen DESC`. Para el panel del home
  + el bucket del evening brief.

## Invariantes

- Silent-fail: AppleScript timeout / Spotify cerrado / DB locked → return
  vacío o `{ok: False, reason: ...}`. Nunca raise — el poller debe poder
  fallar 1 tick sin matar el daemon.
- Sin auth: usa solo el desktop app. NO hay OAuth ni API keys aquí.
- Sin lanzar Spotify: el AppleScript usa `if application "Spotify" is running`
  para evitar abrir el app si el user lo cerró deliberadamente. Sin este
  guard, cada poll de 60s reabriría Spotify infinitamente.

## Por qué deferred imports
`_ragvec_state_conn` vive en `rag.__init__`. Module-level `from rag import _ragvec_state_conn`
crea ciclo. Function-body imports corren después de que el package terminó
de cargar.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional


# AppleScript que devuelve `name|artist|album|state|track_id|duration_ms|position_s`
# o `NOT_RUNNING` si Spotify no está abierto. Usar `|` como separador es safe
# porque Spotify no acepta `|` en metadata (album/artist/track names lo strippean).
# Edge case: si el track no tiene album/artist (raro, podcasts a veces),
# AppleScript devuelve "" para esos campos — el parser lo maneja.
_APPLESCRIPT = '''\
if application "Spotify" is running then
  tell application "Spotify"
    try
      set s to (name of current track) & "|" & (artist of current track) & "|" & (album of current track) & "|" & (player state as string) & "|" & (id of current track) & "|" & (duration of current track) & "|" & (player position as string) & "|" & (artwork url of current track)
      return s
    on error
      return "NO_TRACK"
    end try
  end tell
else
  return "NOT_RUNNING"
end if
'''


def now_playing(timeout: float = 5.0) -> Optional[dict]:
    """Query Spotify desktop app via osascript. Returns None if not running,
    no track loaded, or AppleScript failed.

    Performance: ~50ms cold de un terminal user, ~500ms-3s desde un launchd
    daemon (el primer AppleEvent paga TCC permission check + warm del proxy
    AppleScriptObjC). Por eso el default es 5s, no 2s — desde el web
    daemon (`com.fer.obsidian-rag-web`) consistentemente hitea 2s con el
    cap viejo y devuelve None aunque Spotify esté playing.

    Safe to call from request hot path: si Spotify no está corriendo, el
    `is running` check en AppleScript falla rápido (<10ms) sin lanzar la
    app — el costo solo aplica cuando el app está vivo.
    """
    try:
        res = subprocess.run(
            ["osascript", "-e", _APPLESCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    out = (res.stdout or "").strip()
    if not out or out in ("NOT_RUNNING", "NO_TRACK"):
        return None
    parts = out.split("|")
    # Defensive: the AppleScript returns 8 fields. Fewer = parse error
    # (Spotify might be in a transient state between tracks). Bail.
    if len(parts) < 5:
        return None
    name = parts[0].strip()
    artist = parts[1].strip()
    if not name or not artist:
        # Silent skip — Spotify sometimes returns empty during transitions.
        return None
    duration_ms: Optional[int] = None
    position_s: Optional[float] = None
    art_url: Optional[str] = None
    try:
        if len(parts) > 5 and parts[5]:
            duration_ms = int(float(parts[5]))
    except (ValueError, TypeError):
        pass
    try:
        if len(parts) > 6 and parts[6]:
            position_s = float(parts[6])
    except (ValueError, TypeError):
        pass
    if len(parts) > 7 and parts[7]:
        art_url = parts[7].strip() or None
    return {
        "name": name,
        "artist": artist,
        "album": parts[2].strip(),
        "state": parts[3].strip(),  # "playing" / "paused" / "stopped"
        "track_id": parts[4].strip(),
        "duration_ms": duration_ms,
        "position_s": position_s,
        "art_url": art_url,
    }


# Window dentro del cual un poll repetido del mismo track_id se considera
# "la misma sesión" (UPDATE last_seen) en vez de un re-listen (INSERT
# nuevo). 5 min cubre pausas largas + skips intermedios sin meter ruido,
# y es menor que el track más largo plausible para el use case (canciones,
# no podcasts de 1h+).
_SESSION_GAP_S = 300.0


def record_now_playing() -> dict:
    """Captura el track actual y persiste en `rag_spotify_log`.

    Llamado por el poller (`scripts/spotify_poll.py` cada 60s via launchd).
    Idempotente: si el último row tiene el mismo `track_id` y `last_seen` está
    dentro de `_SESSION_GAP_S`, UPDATE last_seen. Sino INSERT nuevo.

    Solo registra cuando state == "playing" — pausas no cuentan como
    actividad (sino el panel mostraría 30 rows con el mismo track parado
    durante una reunión).

    Devuelve dict con shape `{recorded: bool, updated: bool?, track: str?,
    reason: str?}` para que el poller lo loguee.
    """
    np = now_playing()
    if not np:
        return {"recorded": False, "reason": "spotify_not_running_or_no_track"}
    if np.get("state") != "playing":
        return {"recorded": False, "reason": f"state_{np.get('state', 'unknown')}"}

    # Deferred import para evitar ciclo (rag.__init__ → integrations → rag.__init__).
    from rag import _ragvec_state_conn  # noqa: PLC0415

    now = time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    track_id = np["track_id"]

    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT id, track_id, last_seen FROM rag_spotify_log "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[1] == track_id and (now - float(row[2])) < _SESSION_GAP_S:
                conn.execute(
                    "UPDATE rag_spotify_log SET last_seen=? WHERE id=?",
                    (now, row[0]),
                )
                return {"recorded": True, "updated": True, "track": np["name"]}
            conn.execute(
                "INSERT INTO rag_spotify_log "
                "(track_id, name, artist, album, state, duration_ms, "
                " first_seen, last_seen, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    track_id, np["name"], np["artist"], np["album"],
                    np["state"], np["duration_ms"],
                    now, now, today,
                ),
            )
            return {"recorded": True, "updated": False, "track": np["name"]}
    except Exception as exc:
        # No crashear el poller — el próximo tick reintenta. El daemon
        # supervisor (launchd KeepAlive=false con StartInterval=60) sigue
        # firing cada 60s sin importar este error.
        return {"recorded": False, "reason": f"db_error: {exc!r}"}


def recent_tracks_today(limit: int = 20) -> list[dict]:
    """Lee tracks de HOY desde `rag_spotify_log` ordenados por first_seen DESC.

    Usado por el panel `p-spotify` del home v2 + (TODO futuro) bucket
    `spotify_today` del evening brief. Si la tabla está vacía o el día
    apenas empezó, devuelve [].

    Returns: lista de dicts `{track_id, name, artist, album, first_seen,
    last_seen, duration_played_s}`. `duration_played_s = last_seen - first_seen`
    aproxima cuánto sonó el track (no exacto — el poller mide cada 60s,
    así que la resolución es ±60s).
    """
    from rag import _ragvec_state_conn  # noqa: PLC0415

    today = time.strftime("%Y-%m-%d")
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT track_id, name, artist, album, first_seen, last_seen "
                "FROM rag_spotify_log WHERE date=? "
                "ORDER BY first_seen DESC LIMIT ?",
                (today, int(limit)),
            ).fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        first_seen = float(r[4])
        last_seen = float(r[5])
        out.append({
            "track_id": r[0],
            "name": r[1],
            "artist": r[2],
            "album": r[3] or "",
            "first_seen": first_seen,
            "last_seen": last_seen,
            "duration_played_s": int(last_seen - first_seen),
        })
    return out
