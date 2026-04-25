"""Google Drive integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Two surfaces:

- `_fetch_drive_evidence(now, days, max_items)` — files modified in the last N
  days (default 5), used by morning briefs to mention "qué docs/sheets/slides
  tocaste" (defensive read-only — `drive.readonly` scope).
- `_drive_search_tokens(query, max_tokens)` — extracts up to N meaningful
  keywords from a natural-language query, filtering Spanish/English stopwords
  + Drive-self-reference noise. Used by the `_agent_tool_drive_search` chat
  tool (which lives in `rag.__init__` and uses these tokens to build the
  `fullText contains` query).
- `_drive_service()` — authed Drive API client. Shares OAuth creds with the
  [google-drive MCP](https://github.com/modelcontextprotocol/servers) at
  `~/.config/google-drive-mcp/`. Refreshes access tokens in place and persists
  back to `tokens.json` on expiry (so both the brief and the MCP stay authed
  through the same refresh token).

## Invariants
- Silent-fail: missing deps (`googleapiclient` not installed),
  missing creds, expired refresh token, API error, network error → return
  `{}` / `None` / `[]`. Never raise.
- The OAuth token cache shape supports both `access_token` (Gmail/Drive MCP
  format) and `token` (some Google libs) — both are written back so the cache
  stays compatible with whichever client touched it last.
- `GDRIVE_SCOPES` is `drive.readonly` only — Phase 1.b cross-source corpus
  contract (see `docs/design-cross-source-corpus.md §10.6`). NEVER widen
  without an explicit user override.

## Why `_GDRIVE_MIME_LABEL` lives here too
It's used by `_fetch_drive_evidence` AND by `_agent_tool_drive_search` (which
stays in `rag.__init__` because it's a chat tool, not a leaf integration).
Tests don't patch the constant directly. Re-exported at the bottom of
`rag.__init__` so the agent-tool side keeps resolving it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path


# ── Google Drive evidence (via OAuth shared with google-drive MCP) ──────────
# Surface files modified in the last N days — gives the morning brief context
# about what docs/sheets/slides were touched. Skips the Drive.app daily churn
# by showing only the top-N most-recent items. Silent-fail: missing deps, missing
# creds, or API error → {}. OAuth reused from the `google-drive` MCP setup
# (~/.config/google-drive-mcp/). Drive API exact-count is free; we don't need
# to scan folders — modifiedTime query is O(log n).
GDRIVE_CREDS_DIR = Path.home() / ".config/google-drive-mcp"
GDRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]
# Human-readable Drive mimeTypes for the rendered section. Everything else
# falls back to a generic "📄 archivo" label.
_GDRIVE_MIME_LABEL = {
    "application/vnd.google-apps.document": "Doc",
    "application/vnd.google-apps.spreadsheet": "Sheet",
    "application/vnd.google-apps.presentation": "Slide",
    "application/vnd.google-apps.folder": "Folder",
    "application/vnd.google-apps.form": "Form",
    "application/pdf": "PDF",
}


def _drive_service():
    """Authed Drive API client, or None. Refreshes access_token in place and
    persists back to tokens.json on expiry. Shares creds with the google-drive
    MCP — so both paths stay authed through the same refresh token.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        return None
    tokens_path = GDRIVE_CREDS_DIR / "tokens.json"
    oauth_path = GDRIVE_CREDS_DIR / "gcp-oauth.keys.json"
    if not tokens_path.is_file() or not oauth_path.is_file():
        return None
    try:
        stored = json.loads(tokens_path.read_text(encoding="utf-8"))
        oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
        installed = oauth.get("installed") or oauth.get("web") or {}
        # The google-drive MCP tokens file uses `access_token` / `refresh_token`
        # at top level (same shape as the Gmail MCP). Some Google libs write
        # `token` instead — accept both.
        token = stored.get("access_token") or stored.get("token")
        creds = Credentials(
            token=token,
            refresh_token=stored.get("refresh_token"),
            token_uri=installed.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=installed.get("client_id"),
            client_secret=installed.get("client_secret"),
            scopes=GDRIVE_SCOPES,
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            stored["access_token"] = creds.token
            stored["token"] = creds.token
            tokens_path.write_text(json.dumps(stored), encoding="utf-8")
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


def _fetch_drive_evidence(now: datetime, days: int = 5, max_items: int = 5) -> dict:
    """Files modified in Drive in the last `days`. Filters trashed + picks the
    `max_items` most-recent. Returns `{"files": [{name, modified, link,
    mime_label, days_ago}]}`. `days_ago` is a float for the renderer to format.
    Silent-fail → {}.
    """
    svc = _drive_service()
    if svc is None:
        return {}
    cutoff_dt = now - timedelta(days=days)
    # Drive query wants RFC3339; strip tz for simplicity (local vs UTC mismatch
    # is at most a few hours — negligible at a 5-day window).
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    q = f"modifiedTime > '{cutoff_iso}' and trashed = false"
    try:
        resp = svc.files().list(
            q=q,
            orderBy="modifiedTime desc",
            pageSize=max_items,
            fields=(
                "files(id,name,modifiedTime,webViewLink,mimeType,"
                "lastModifyingUser(displayName))"
            ),
            spaces="drive",
        ).execute()
    except Exception:
        return {}
    files_out: list[dict] = []
    for f in resp.get("files") or []:
        mtime_iso = f.get("modifiedTime") or ""
        try:
            # Drive returns `2026-04-14T18:23:00.000Z`
            mtime_dt = datetime.fromisoformat(mtime_iso.replace("Z", "+00:00"))
            days_ago = max(0.0, (now.astimezone(mtime_dt.tzinfo) - mtime_dt).total_seconds() / 86400.0)
        except Exception:
            days_ago = 0.0
        mime = f.get("mimeType") or ""
        label = _GDRIVE_MIME_LABEL.get(mime, "archivo")
        files_out.append({
            "name": f.get("name") or "(sin nombre)",
            "modified": mtime_iso,
            "link": f.get("webViewLink") or "",
            "mime_label": label,
            "days_ago": round(days_ago, 1),
            "modifier": (f.get("lastModifyingUser") or {}).get("displayName") or "",
        })
    return {"files": files_out, "window_days": days}


# ── Drive on-demand search (chat tool) ───────────────────────────────────────
# Motivación (2026-04-24, user report Fer F.): el user pidió "busca en mi
# google drive y decime cuánto adeuda alexis de la macbook pro" y el chat
# respondió sobre la única planilla snapshoteada (`Lista de precios
# Online`, modificada <48h) — sin buscar realmente en Drive. El snapshot
# diario (`_sync_gdrive_notes`) sólo trae los 4 docs más recientes con
# 8000 chars de body — insuficiente para responder queries sobre archivos
# viejos o grandes. Este helper expone una búsqueda on-demand en la API
# de Drive: dado un query en lenguaje natural, filtra stopwords, arma un
# `fullText contains` y exporta el body de los top-N archivos.
#
# Comparte `_drive_service()` con el brief evidence path — mismas creds
# (`~/.config/google-drive-mcp/`), misma refresh-token semantics, silent-
# fail cuando falta auth (y el chat cae de vuelta al retrieve del vault).
_GDRIVE_SEARCH_STOPWORDS: frozenset[str] = frozenset({
    # Articles / conjunctions / prepositions (ES + EN, deduped into one set).
    "a", "al", "an", "and", "ante", "at", "bajo", "but", "by", "con",
    "contra", "de", "del", "desde", "durante", "e", "el", "en", "entre",
    "for", "from", "hacia", "hasta", "in", "la", "las", "lo", "los",
    "mediante", "o", "of", "on", "or", "para", "pero", "por", "segun",
    "sin", "sobre", "the", "to", "tras", "un", "una", "unas", "unos",
    "with", "y",
    # Pronouns / interrogatives that show up in command phrasing (ES + EN).
    "mi", "mis", "tu", "tus", "me", "te", "se", "le", "les", "nos", "yo",
    "que", "qué", "como", "cómo", "cuando", "cuándo", "cuanto", "cuánto",
    "donde", "dónde", "quien", "quién", "cual", "cuál",
    "my", "your", "our", "his", "her", "its", "their", "is", "are", "was",
    "were", "i", "you", "we", "they", "he", "she", "it",
    # Command / filler verbs (Spanish rioplatense + neutral).
    "busca", "buscá", "buscar", "buscame", "decime", "deci", "dime",
    "contame", "quiero", "necesito", "quisiera", "saber", "ver", "mirar",
    "revisar", "chequear", "chequeame", "favor", "fijate", "fijame",
    # "tener/haber/ser/estar" conjugations frecuentes — son verbos
    # genéricos que inflan name-OR sin aportar (ej. "tengo la planilla
    # de alexis" → name contains 'tengo' matchea "Tengo que firmar...").
    "tengo", "tenés", "tiene", "tienen", "tenemos", "hay", "había",
    "hubo", "habría", "era", "eran", "fue", "fueron", "estaba",
    "estaban", "está", "están", "estamos", "son", "soy", "eres",
    # Drive-self-reference (redundant as a search token when the query is
    # already scoped to Drive — "busca X en mi drive" → tokens=[X]).
    "drive", "gdrive", "google", "doc", "docs", "documento", "documentos",
    "sheet", "sheets", "planilla", "planillas", "spreadsheet",
    "spreadsheets", "slide", "slides", "presentacion", "presentación",
    "presentaciones", "archivo", "archivos", "file", "files",
})


def _drive_search_tokens(query: str, max_tokens: int = 6) -> list[str]:
    """Extract up to `max_tokens` meaningful keywords from a natural-language
    query. Lowercases, drops punctuation, filters `_GDRIVE_SEARCH_STOPWORDS`,
    de-duplicates preserving order. Empty → []."""
    # Normalize: lowercase, strip punctuation except inner apostrophes/dashes.
    cleaned = re.sub(r"[^\w\s\-']", " ", query.lower(), flags=re.UNICODE)
    seen: set[str] = set()
    out: list[str] = []
    for tok in cleaned.split():
        tok = tok.strip("-'")
        if len(tok) < 2:
            continue
        if tok in _GDRIVE_SEARCH_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_tokens:
            break
    return out
