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
- `control(action)` — ejecuta playpause / next / previous / play / pause /
  stop sobre Spotify desktop via AppleScript. Determinístico, ~50ms-500ms.
  Devuelve `{ok, state?, error?}`.
- `play_uri(uri)` — reproduce un URI de Spotify (`spotify:track:XXX` o
  `spotify:album:YYY` o `spotify:playlist:ZZZ`) en el desktop. Lanza
  Spotify si está cerrado (a propósito — el user pidió poner música).
- `search_track(query)` — busca un track via Web API con Client Credentials
  (NO requiere Premium ni OAuth user — el gate de Premium aplica a
  `/me/*`, no a `/search`). Devuelve `{name, artist, uri}` o None.
- `get_volume()` / `set_volume(N)` / `adjust_volume(delta)` — control del
  volumen del app Spotify (0-100, separado del system volume de macOS).
  AppleScript `sound volume` lee/setea el del app, no el master.

## Invariantes

- Silent-fail: AppleScript timeout / Spotify cerrado / DB locked → return
  vacío o `{ok: False, reason: ...}`. Nunca raise — el poller debe poder
  fallar 1 tick sin matar el daemon.
- Sin auth para queries pasivas: `now_playing` / `recent_tracks_today` /
  `control` usan solo el desktop app, sin OAuth.
- `search_track` SÍ requiere Client Credentials (`~/.config/obsidian-rag/
  spotify_client.json` con `client_id` + `client_secret`). Silent-fails
  si no están.
- `now_playing` y `control(playpause/next/previous/pause)` NO lanzan
  Spotify si está cerrado (guard `is running`). Pero `play_uri` y
  `control(play)` SÍ lo lanzan — el user explicit quiere música.

## Por qué deferred imports
`_ragvec_state_conn` vive en `rag.__init__`. Module-level `from rag import _ragvec_state_conn`
crea ciclo. Function-body imports corren después de que el package terminó
de cargar.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
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


# Acciones soportadas por `control()`. Mapean 1:1 a comandos AppleScript
# del Spotify desktop app ([docs](https://developer.spotify.com/documentation/applescript)
# — sí, Spotify mantiene una scripting dictionary pública). `playpause`
# es el toggle (más útil para "pausa" cuando podría estar paused ya).
_CONTROL_ACTIONS = {
    "play": "play",
    "pause": "pause",
    "playpause": "playpause",
    "next": "next track",
    "previous": "previous track",
}


# Step default para subir/bajar volumen — 10 puntos (escala 0-100). Da
# ~10 niveles entre mute y max, suficiente granularidad sin requerir
# muchos comandos para llegar a un nivel deseado.
_VOLUME_STEP = 10


def get_volume(timeout: float = 2.0) -> int | None:
    """Devuelve el volumen actual de Spotify (0-100) o None si Spotify
    está cerrado / AppleScript falla.

    Spotify mantiene su propio volumen separado del system volume — el
    AppleScript `sound volume` consulta el del app, no el master de
    macOS. Eso es lo que el user típicamente quiere modificar (subir
    Spotify sin tocar el resto del audio).
    """
    script = '''\
if application "Spotify" is running then
  tell application "Spotify"
    return sound volume as string
  end tell
else
  return "NOT_RUNNING"
end if
'''
    try:
        res = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    out = (res.stdout or "").strip()
    if not out or out == "NOT_RUNNING":
        return None
    try:
        return int(float(out))
    except (ValueError, TypeError):
        return None


def set_volume(level: int, timeout: float = 2.0) -> dict:
    """Setea el volumen de Spotify a `level` (0-100). Clampea fuera de
    rango para no fallar silencioso.

    Devuelve `{ok, level?, error?}`. Si Spotify está cerrado, NO lo
    lanza (el user pidió ajustar volumen, no abrir el app).
    """
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"invalid_level: {level!r}"}
    lvl = max(0, min(100, lvl))
    script = f'''\
if application "Spotify" is running then
  tell application "Spotify"
    set sound volume to {lvl}
    return "OK"
  end tell
else
  return "NOT_RUNNING"
end if
'''
    try:
        res = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "error": f"applescript_failed: {exc!r}"}
    out = (res.stdout or "").strip()
    if out == "NOT_RUNNING":
        return {"ok": False, "error": "spotify_not_running"}
    if res.returncode != 0:
        return {"ok": False, "error": (res.stderr or "").strip()[:200]}
    return {"ok": True, "level": lvl}


def adjust_volume(delta: int, timeout: float = 2.0) -> dict:
    """Sube/baja el volumen en `delta` puntos (positivo o negativo).
    Lee el actual + setea el nuevo. Clampea a [0, 100].

    Devuelve `{ok, level?, prev?, error?}`. Si Spotify está cerrado o
    el get_volume falla, devuelve `{ok: False, error: ...}`.
    """
    current = get_volume(timeout=timeout)
    if current is None:
        return {"ok": False, "error": "spotify_not_running_or_unreadable"}
    new_level = max(0, min(100, current + int(delta)))
    res = set_volume(new_level, timeout=timeout)
    if not res.get("ok"):
        return res
    return {"ok": True, "level": new_level, "prev": current}


def control(action: str, timeout: float = 3.0) -> dict:
    """Ejecuta un comando de control de playback sobre Spotify desktop.

    Args:
        action: uno de play/pause/playpause/next/previous.
        timeout: AppleScript timeout (default 3s — más alto que `now_playing`
          porque `next`/`previous` cargan track nuevo y pueden tardar).

    Devuelve:
        `{ok: bool, action: str, state?: str, name?, artist?, error?: str}`.
        En `ok=True`, incluye el `now_playing()` post-acción para que el
        caller pueda armar respuesta tipo "⏭ Saltando — ahora suena: X / Y"
        sin un round-trip extra.

    NO lanza Spotify si está cerrado para `pause`/`playpause`/`next`/
    `previous` (no tiene sentido pausar nada). Para `play`, sí lanza —
    "play" sin nada cargado es ambiguo, pero al menos abre el app.
    """
    a = action.strip().lower()
    cmd = _CONTROL_ACTIONS.get(a)
    if cmd is None:
        return {"ok": False, "action": a, "error": f"unknown_action: {a}"}

    # Para `play` permitimos lanzar Spotify (user pidió música).
    # Para el resto, guard `is running` — sino reabriríamos el app.
    if a == "play":
        script = f'''\
tell application "Spotify"
  {cmd}
end tell
return "OK"
'''
    else:
        script = f'''\
if application "Spotify" is running then
  tell application "Spotify"
    {cmd}
  end tell
  return "OK"
else
  return "NOT_RUNNING"
end if
'''
    try:
        res = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "action": a, "error": f"applescript_failed: {exc!r}"}
    out = (res.stdout or "").strip()
    if out == "NOT_RUNNING":
        return {"ok": False, "action": a, "error": "spotify_not_running"}
    if res.returncode != 0:
        return {"ok": False, "action": a, "error": (res.stderr or "").strip()[:200]}

    # Post-acción: pequeño settle (Spotify tarda ~100-300ms en actualizar el
    # track después de un `next`) y leer el now_playing actual.
    if a in ("next", "previous"):
        time.sleep(0.4)
    np = now_playing(timeout=2.0)
    if np:
        return {
            "ok": True, "action": a,
            "state": np.get("state"),
            "name": np.get("name"),
            "artist": np.get("artist"),
            "album": np.get("album"),
        }
    return {"ok": True, "action": a}


def play_uri(uri: str, timeout: float = 4.0) -> dict:
    """Reproduce un URI de Spotify en el desktop app.

    Args:
        uri: URI de Spotify, formato `spotify:track:XXX`, `spotify:album:YYY`
          o `spotify:playlist:ZZZ`.
        timeout: AppleScript timeout (default 4s — load de track + arranque).

    Devuelve `{ok, name?, artist?, error?}`. Si Spotify está cerrado, lo
    lanza (a diferencia de `now_playing`) — el user explícitamente pidió
    música.
    """
    if not uri or not uri.startswith("spotify:"):
        return {"ok": False, "error": f"invalid_uri: {uri!r}"}
    # Escapamos la URI para AppleScript (paranoid — los URIs de Spotify
    # son ASCII alphanumeric + `:` así que no debería haber edge cases).
    safe = uri.replace('"', '')
    script = f'''\
tell application "Spotify"
  play track "{safe}"
end tell
return "OK"
'''
    try:
        res = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "error": f"applescript_failed: {exc!r}"}
    if res.returncode != 0:
        return {"ok": False, "error": (res.stderr or "").strip()[:200]}

    # Settle: Spotify tarda ~300-700ms en arrancar el track nuevo.
    time.sleep(0.6)
    np = now_playing(timeout=2.0)
    if np:
        return {
            "ok": True,
            "name": np.get("name"),
            "artist": np.get("artist"),
            "album": np.get("album"),
        }
    return {"ok": True}


# Cache del token de Client Credentials. Spotify devuelve tokens válidos
# por 1h (3600s); cacheamos in-process porque el server es persistente.
_CC_TOKEN_CACHE: dict = {"token": None, "expires_at": 0.0}

_SPOTIFY_CREDS_PATH = Path.home() / ".config/obsidian-rag/spotify_client.json"


def _client_credentials_token() -> Optional[str]:
    """Devuelve un access token via Client Credentials flow.

    [Docs](https://developer.spotify.com/documentation/web-api/tutorials/client-credentials-flow).
    Cachea in-process. Silent-fails (devuelve None) si:
      - No hay `~/.config/obsidian-rag/spotify_client.json`.
      - `client_id` o `client_secret` faltan.
      - urllib request falla (red, cred inválido, etc.).
    """
    now = time.time()
    cached = _CC_TOKEN_CACHE.get("token")
    expires = float(_CC_TOKEN_CACHE.get("expires_at") or 0.0)
    # Refrescamos 60s antes para evitar carrera con expiración.
    if cached and now < (expires - 60.0):
        return cached  # type: ignore[return-value]

    if not _SPOTIFY_CREDS_PATH.is_file():
        return None
    try:
        creds = json.loads(_SPOTIFY_CREDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cid = (creds.get("client_id") or "").strip()
    secret = (creds.get("client_secret") or "").strip()
    if not (cid and secret):
        return None

    # Usamos urllib para no agregar otra dep — `requests` no está garantizado
    # en este path y `spotipy` arrastra OAuth user flow (que no queremos).
    import base64
    import urllib.parse
    import urllib.request

    auth_b64 = base64.b64encode(f"{cid}:{secret}".encode("ascii")).decode("ascii")
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("ascii")
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=body,
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    token = payload.get("access_token")
    expires_in = float(payload.get("expires_in") or 3600.0)
    if not token:
        return None
    _CC_TOKEN_CACHE["token"] = token
    _CC_TOKEN_CACHE["expires_at"] = now + expires_in
    return token


def search_track(query: str, limit: int = 1, timeout: float = 4.0) -> Optional[dict]:
    """Busca un track via Spotify Web API (Client Credentials).

    [Docs](https://developer.spotify.com/documentation/web-api/reference/search).
    NO requiere Premium ni user OAuth — el gate de Premium aplica a
    `/me/*` (`/me/player/recently-played` etc.), pero `/v1/search` es
    público para apps en development mode con Client Credentials.

    Args:
        query: texto de búsqueda (ej. "bohemian rhapsody queen", "spinetta").
        limit: cuántos resultados pedir (devuelve solo el top — el limit
          es para la API, que por default ya es 1).
        timeout: HTTP timeout (default 4s).

    Devuelve `{name, artist, album, uri, track_id}` del top result o None
    si no hay match / API falla / no hay credenciales.
    """
    q = (query or "").strip()
    if not q:
        return None
    token = _client_credentials_token()
    if not token:
        return None

    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({
        "q": q, "type": "track", "limit": int(limit),
        # `market=AR` ayuda a Spotify a devolver versiones regionales con
        # mejor disponibilidad (algunos tracks tienen múltiples releases).
        "market": "AR",
    })
    req = urllib.request.Request(
        f"https://api.spotify.com/v1/search?{params}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    items = (((payload or {}).get("tracks") or {}).get("items") or [])
    if not items:
        return None
    top = items[0]
    artists = top.get("artists") or []
    artist_names = ", ".join((a.get("name") or "") for a in artists if a.get("name"))
    album = (top.get("album") or {}).get("name") or ""
    return {
        "name": top.get("name") or "",
        "artist": artist_names,
        "album": album,
        "uri": top.get("uri") or "",
        "track_id": top.get("id") or "",
    }
