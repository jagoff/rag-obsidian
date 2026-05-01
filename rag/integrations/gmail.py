"""Gmail integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Three surfaces:

- `_gmail_service()` — read/list-only client. Tries the canonical token at
  `~/.config/obsidian-rag/google_token.json` first (managed by
  `_load_google_credentials`, scope `gmail.readonly + drive.readonly`), and
  falls back to the legacy `~/.gmail-mcp/credentials.json` (scope
  `gmail.modify`). Returns `None` if neither path produces valid creds.
- `_gmail_send_service()` — send-capable client. Uses the legacy MCP path
  (scope `gmail.modify` includes `send`); the primary token typically has
  only `gmail.readonly` and would 403 on send. Returns `None` if no
  send-capable creds exist (caller silent-fails).
- `_gmail_thread_last_meta(svc, thread_id)` — fetch metadata of the LAST
  message of a thread. Returns `None` on error. Pure helper; doesn't touch
  shared state.
- `_fetch_gmail_evidence(now)` — Gmail signals for the morning brief +
  on-demand "últimos mails" queries. Surfaces `unread_count`, `starred`,
  `awaiting_reply`, `recent`. Hits Gmail API ~5-15 times (<3s with cached
  discovery). Silent-fail → `{}`.

## Invariants
- Silent-fail: every code path that can raise (missing deps, missing creds,
  expired token, API 403/500, network error) returns `{}` / `None` / `[]`.
  Errors get logged via `_silent_log` (deferred import) for postmortem
  observability — never propagated.
- The OAuth fallback in `_gmail_service`/`_gmail_send_service` writes the
  refreshed `access_token` back to `credentials.json` so subsequent calls
  skip the refresh hop.
- Scopes: `gmail.modify + gmail.settings.basic` for the legacy path.
  `gmail.readonly` (via `_load_google_credentials`) for the primary path.
  See `docs/design-cross-source-corpus.md §10.6` for scope policy.

## Why deferred imports
`_silent_log` and `_load_google_credentials` live in `rag.__init__`.
Module-level imports here would deadlock the package load. Function-body
imports run after `rag.__init__` finishes, so they always succeed.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


# ── Gmail evidence (via OAuth shared with gmail MCP / google-auth) ──────────
GMAIL_CREDS_DIR = Path.home() / ".gmail-mcp"
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.settings.basic",
]


def _gmail_service():
    """Authed Gmail API client, or None.

    Intenta 2 paths de OAuth en orden:

    1. **`~/.config/obsidian-rag/google_token.json`** (vía
       `_load_google_credentials`). Este es el token que usan el
       ingester de Gmail + Calendar + Drive + morning brief — se
       refresca solo via google-auth-oauthlib y es el path que el user
       re-autentica con `rag auth google` / primera corrida
       interactiva. Scopes: `gmail.readonly + drive.readonly` — suficiente
       para LECTURA (que es todo lo que hacen `_fetch_gmail_evidence` y
       `_sync_gmail_notes`).

    2. **`~/.gmail-mcp/credentials.json`** (legacy, shared con gmail-mcp
       NPM). Scopes más amplios (`gmail.modify`) pero el refresh token
       tiende a caducar porque Google lo revoca si la OAuth app está en
       "Testing" mode y pasaron >7d. Lo mantenemos como fallback para no
       romper setups viejos que tengan solo este token, pero ya no es el
       path canónico.

    Retorna None si ambos fallan (caller maneja silent-fail). Rollback
    al comportamiento pre-iter5: borrar el bloque (1) y dejar solo (2).
    Motivo del cambio (2026-04-24, user report iter 5): el token en (2)
    se revocó y `_fetch_gmail_evidence` devolvía vacío silenciosamente
    → el user preguntaba "cuales son mis ultimos mails?" y el sistema
    decía "no encontré mails recientes" aunque el inbox estaba lleno.
    """
    from rag import _load_google_credentials, _silent_log
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return None

    # Path primario: google_token.json (self-refreshing, alive).
    try:
        creds = _load_google_credentials(allow_interactive=False)
        if creds is not None and creds.valid:
            return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as exc:
        _silent_log("gmail_service_primary", exc)

    # Fallback: ~/.gmail-mcp/credentials.json (legacy, may be revoked).
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        return None
    creds_path = GMAIL_CREDS_DIR / "credentials.json"
    oauth_path = GMAIL_CREDS_DIR / "gcp-oauth.keys.json"
    if not creds_path.is_file() or not oauth_path.is_file():
        return None
    try:
        stored = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
        installed = oauth.get("installed") or oauth.get("web") or {}
        creds = Credentials(
            token=stored.get("access_token"),
            refresh_token=stored.get("refresh_token"),
            token_uri=installed.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=installed.get("client_id"),
            client_secret=installed.get("client_secret"),
            scopes=GMAIL_SCOPES,
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            stored["access_token"] = creds.token
            creds_path.write_text(json.dumps(stored), encoding="utf-8")
        return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as exc:
        _silent_log("gmail_service_fallback", exc)
        return None


def _gmail_send_service():
    """Authed Gmail client para ENVIAR (scope `gmail.modify`).

    Similar a `_gmail_service()` pero prioriza invertido: prefiere el
    fallback en `~/.gmail-mcp/credentials.json` (scope `gmail.modify` —
    incluye send) porque el primary (`~/.config/obsidian-rag/
    google_token.json`) típicamente tiene sólo `gmail.readonly` + `drive.
    readonly` y fallaría con 403 al enviar.

    Retorna None si no hay creds con el scope correcto. Caller silent-
    fail. El fallback re-escribe el access token refrescado al archivo,
    como hace `_gmail_service()`, así el próximo call evita el refresh.
    """
    from rag import _silent_log
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        return None
    creds_path = GMAIL_CREDS_DIR / "credentials.json"
    oauth_path = GMAIL_CREDS_DIR / "gcp-oauth.keys.json"
    if not creds_path.is_file() or not oauth_path.is_file():
        return None
    try:
        stored = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
        installed = oauth.get("installed") or oauth.get("web") or {}
        creds = Credentials(
            token=stored.get("access_token"),
            refresh_token=stored.get("refresh_token"),
            token_uri=installed.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=installed.get("client_id"),
            client_secret=installed.get("client_secret"),
            scopes=GMAIL_SCOPES,
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            stored["access_token"] = creds.token
            creds_path.write_text(json.dumps(stored), encoding="utf-8")
        return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as exc:
        _silent_log("gmail_send_service", exc)
        return None


def _gmail_thread_last_meta(svc, thread_id: str) -> dict | None:
    """Fetch metadata of the LAST message of a thread. Returns None on error.
    Shape: {subject, from, snippet, internal_date_ms}.
    """
    try:
        t = svc.users().threads().get(
            userId="me", id=thread_id, format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
    except Exception:
        return None
    msgs = t.get("messages") or []
    if not msgs:
        return None
    last = msgs[-1]
    headers = {
        h["name"]: h["value"]
        for h in (last.get("payload", {}).get("headers") or [])
    }
    try:
        internal_ms = int(last.get("internalDate") or 0)
    except Exception:
        internal_ms = 0
    return {
        "subject": headers.get("Subject", "(sin asunto)"),
        "from": headers.get("From", ""),
        "snippet": (last.get("snippet") or "").strip()[:140],
        "internal_date_ms": internal_ms,
    }


def _fetch_gmail_today(now: datetime, max_items: int = 8) -> list[dict]:
    """Mails recibidos HOY (today 00:00 local → now). Distinto de
    `_fetch_gmail_evidence`: ese devuelve "unread total + starred 7d +
    awaiting 3-14d + recent 30d". Este corta exactamente al inicio del
    día local — para el evening brief que quiere "qué llegó hoy".

    Returns: list of {subject, from, snippet, thread_id, internal_date_ms}
    sorted DESC by internal_date_ms (most recent first), max `max_items`.

    Silent-fail: si la API falla devuelve []. Sin auth → []. El brief
    se renderea sin la sección de gmail-today.
    """
    from rag import _silent_log
    svc = _gmail_service()
    if svc is None:
        return []
    today_start_ms = int(now.replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).timestamp() * 1000)
    out: list[dict] = []
    try:
        # `newer_than:1d` agarra desde "now - 24h" (window relative).
        # Filtramos después por internal_date_ms >= today_start_ms para
        # cortar exacto al 00:00 local. Si el cron corre a las 22hs,
        # `newer_than:1d` puede mezclar 22hs de ayer; filter cliente
        # arregla.
        r = svc.users().threads().list(
            userId="me", q="in:inbox newer_than:1d",
            maxResults=max(int(max_items) * 2, 10),
        ).execute()
        for th in r.get("threads", []) or []:
            tid = th.get("id") or ""
            meta = _gmail_thread_last_meta(svc, tid)
            if not meta:
                continue
            ms = int(meta.get("internal_date_ms") or 0)
            if ms < today_start_ms:
                continue
            out.append({
                "subject": meta["subject"],
                "from": meta["from"],
                "snippet": meta["snippet"],
                "thread_id": tid,
                "internal_date_ms": ms,
            })
            if len(out) >= max_items:
                break
    except Exception as exc:
        _silent_log('gmail_today_list', exc)
        return []
    out.sort(key=lambda x: x.get("internal_date_ms") or 0, reverse=True)
    return out


# Cache para _fetch_gmail_evidence (2026-05-01 audit fix). El fetcher
# pega 5-15 veces a la Gmail API por call (~2.7s warm medido en SSE
# stream, p99 mucho peor en cold con OAuth refresh). El panel home se
# refrescá cada 5min pero el user puede hacer manual refreshes — cache
# corto (90s) absorbe esos hits sin que la data quede stale (gmail no
# cambia mucho en 90s y el morning brief que es el otro consumer se
# corre 1x/día).
#
# Lock para single-flight (2 fetchers concurrentes en cold = 2 OAuth
# refreshes paralelos = peor). El primer caller hace el HTTP, los demás
# esperan el lock + leen del cache.
_GMAIL_EVIDENCE_CACHE: dict = {"ts": 0.0, "payload": None}
_GMAIL_EVIDENCE_TTL = 90  # seconds
_GMAIL_EVIDENCE_LOCK = threading.Lock()


def _clear_gmail_evidence_cache() -> None:
    """Test helper — drop the gmail evidence cache."""
    with _GMAIL_EVIDENCE_LOCK:
        _GMAIL_EVIDENCE_CACHE["ts"] = 0.0
        _GMAIL_EVIDENCE_CACHE["payload"] = None


def _fetch_gmail_evidence(now: datetime) -> dict:
    """Gmail signals for the morning brief + on-demand "últimos mails"
    queries. Hits Gmail API ~5-15 times (<3s total with cached discovery).
    Silent-fail on any error.

    Cached 90s in-memory. The cache is keyed only by the function
    (not by `now`) — `now` is used only for date-relative queries that
    don't drift meaningfully in 90s. If the user is at midnight rollover
    there's a 90s window where stale "today" buckets can show, but the
    next refresh covers it.

    Returns:
        {
          "unread_count": int — total unread in inbox (all-time),
          "starred": [{subject, from, snippet, thread_id, internal_date_ms}]
             up to 3 recent starred threads,
          "awaiting_reply": [{subject, from, snippet, days_old, thread_id,
             internal_date_ms}] up to 5 threads where the last message is
             NOT from me and is 3-14d old.
          "recent": [{subject, from, snippet, thread_id, internal_date_ms}]
             up to 8 most-recent inbox threads regardless of flags. 2026-04-24
             (user report iter 5): el tool original solo devolvía starred +
             awaiting — para un user con inbox-zero-ish ambos vienen vacíos
             aunque tenga mails perfectamente navegables en el inbox. "últimos
             mails" significa "los más recientes", no "los flagueados". Este
             bucket tapa ese gap.
        }

        `thread_id` + `internal_date_ms` are emitted so downstream consumers
        (web tools → `gmail_recent`) don't have to re-query Gmail to enrich
        each item. `internal_date_ms` is int milliseconds since epoch;
        callers convert to ISO timestamp at render time.
    """
    # Cache check + single-flight: si el payload está fresh, return inmediato.
    # Si no, tomamos el lock y dejamos que solo un caller pegue a la API.
    # Los demás esperan el lock; cuando entran, re-checkean el cache (que ya
    # fue populado por el primer caller) y retornan sin pegar a Gmail.
    now_ts = time.time()
    cached = _GMAIL_EVIDENCE_CACHE
    if cached["payload"] is not None and now_ts - cached["ts"] < _GMAIL_EVIDENCE_TTL:
        return cached["payload"]

    with _GMAIL_EVIDENCE_LOCK:
        # Re-check dentro del lock para soportar single-flight real.
        cached = _GMAIL_EVIDENCE_CACHE
        if cached["payload"] is not None and time.time() - cached["ts"] < _GMAIL_EVIDENCE_TTL:
            return cached["payload"]
        # Compute uncached + persist. El uncached call es pesado (~2.7s warm,
        # mucho más en cold con OAuth refresh) → sostenemos el lock durante
        # todo el HTTP así otros callers no disparan requests redundantes.
        result = _fetch_gmail_evidence_uncached(now)
        if result:
            _GMAIL_EVIDENCE_CACHE["ts"] = time.time()
            _GMAIL_EVIDENCE_CACHE["payload"] = result
        return result


def _fetch_gmail_evidence_uncached(now: datetime) -> dict:
    """Internal helper — does the actual HTTP/API work for `_fetch_gmail_evidence`.
    Extracted so the cache wrapper can compose it without the cache logic
    leaking into the API call paths.
    """
    from rag import _silent_log
    svc = _gmail_service()
    if svc is None:
        return {}
    try:
        profile = svc.users().getProfile(userId="me").execute()
    except Exception:
        return {}
    me_lower = (profile.get("emailAddress") or "").lower()
    out: dict = {
        "unread_count": 0,
        "starred": [],
        "awaiting_reply": [],
        "recent": [],
    }

    # Unread count — INBOX label metadata gives exact thread/message counts
    # without scanning messages. Cheaper and accurate vs resultSizeEstimate.
    try:
        label = svc.users().labels().get(userId="me", id="INBOX").execute()
        out["unread_count"] = int(label.get("threadsUnread") or 0)
    except Exception as exc:
        _silent_log('gmail_unread_count', exc)

    # Starred recent — explicit user-flagged threads.
    try:
        r = svc.users().threads().list(
            userId="me", q="is:starred in:inbox newer_than:7d", maxResults=3,
        ).execute()
        for th in r.get("threads", []) or []:
            tid = th.get("id") or ""
            meta = _gmail_thread_last_meta(svc, tid)
            if meta:
                out["starred"].append({
                    "subject": meta["subject"],
                    "from": meta["from"],
                    "snippet": meta["snippet"],
                    "thread_id": tid,
                    "internal_date_ms": meta["internal_date_ms"],
                })
    except Exception as exc:
        _silent_log('gmail_unread_list', exc)

    # Awaiting reply — threads stuck, last sender != me. Exclude auto-generated
    # categories to keep signal/noise high.
    try:
        q = (
            "in:inbox newer_than:14d older_than:3d "
            "-category:promotions -category:social "
            "-category:updates -category:forums"
        )
        r = svc.users().threads().list(
            userId="me", q=q, maxResults=15,
        ).execute()
        for th in r.get("threads", []) or []:
            if len(out["awaiting_reply"]) >= 5:
                break
            tid = th.get("id") or ""
            meta = _gmail_thread_last_meta(svc, tid)
            if not meta:
                continue
            sender = meta["from"].lower()
            if me_lower and me_lower in sender:
                continue  # last reply was mine — not awaiting me
            days_old = (
                (now.timestamp() - meta["internal_date_ms"] / 1000) / 86400.0
                if meta["internal_date_ms"] else 0.0
            )
            out["awaiting_reply"].append({
                "subject": meta["subject"],
                "from": meta["from"],
                "snippet": meta["snippet"],
                "days_old": round(days_old, 1),
                "thread_id": tid,
                "internal_date_ms": meta["internal_date_ms"],
            })
    except Exception as exc:
        _silent_log('gmail_followup_list', exc)

    # Recent inbox — los últimos 8 hilos del inbox sin filtrar por flag y
    # SIN exclusiones de categoría. Es el bucket que responde literal
    # "cuales son mis ultimos mails?" — el user quiere ver todo lo que el
    # Gmail le muestra en la pantalla de inbox, no una vista filtrada.
    #
    # 2026-04-24 iter 6 (Fer F. user decision): antes excluíamos
    # promotions/social/updates/forums (heredado del morning brief donde
    # esconder newsletters + GitHub notifications hace sentido). Pero
    # para "últimos mails" el user quiere literal los 8 más recientes
    # del inbox tal cual Gmail los ordena. `category:updates` en particular
    # atrapa GitHub notifications, Stripe receipts, Anthropic updates —
    # todo "mail real" para un dev. Si eventualmente se siente ruidoso
    # (promotions puras de Mercadolibre), rollback poniendo
    # `-category:promotions` nuevamente. Rollback parcial no pasa tests
    # nuevos porque los fake_ev no filtran por esas labels.
    #
    # Mantenemos `newer_than:30d` para acotar el scan: si el user no
    # tocó Gmail en un mes, los 8 más viejos no son "últimos" en el
    # sentido útil.
    try:
        q_recent = "in:inbox newer_than:30d"
        r = svc.users().threads().list(
            userId="me", q=q_recent, maxResults=8,
        ).execute()
        # Dedup contra starred/awaiting — si el user tiene un mail starred
        # que también es el más reciente, no queremos listarlo dos veces.
        # Set de thread_ids ya agregados en las secciones previas.
        _seen_tids: set[str] = {
            str(it.get("thread_id") or "")
            for bucket in ("starred", "awaiting_reply")
            for it in out[bucket]
        }
        for th in r.get("threads", []) or []:
            tid = th.get("id") or ""
            if tid in _seen_tids:
                continue
            meta = _gmail_thread_last_meta(svc, tid)
            if not meta:
                continue
            out["recent"].append({
                "subject": meta["subject"],
                "from": meta["from"],
                "snippet": meta["snippet"],
                "thread_id": tid,
                "internal_date_ms": meta["internal_date_ms"],
            })
            _seen_tids.add(tid)
    except Exception as exc:
        _silent_log('gmail_recent_list', exc)

    return out
