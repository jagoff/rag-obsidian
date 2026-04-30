#!/usr/bin/env python3
"""Bootstrap OAuth para Google Calendar — abre browser, el user autoriza,
escribe ~/.calendar-mcp/credentials.json fresco.

Uso (un solo paso):
    .venv/bin/python scripts/bootstrap_calendar_oauth.py

Requiere ~/.calendar-mcp/gcp-oauth.keys.json (las creds de la app GCP).
Sobrescribe ~/.calendar-mcp/credentials.json si existe (tras revocación).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

CREDS_DIR = Path.home() / ".calendar-mcp"
OAUTH_KEYS = CREDS_DIR / "gcp-oauth.keys.json"
CREDENTIALS = CREDS_DIR / "credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
]


def main() -> int:
    if not OAUTH_KEYS.is_file():
        print(f"ERROR: {OAUTH_KEYS} no existe.", file=sys.stderr)
        print("Necesitás copiar las credenciales de la app GCP ahí primero.",
              file=sys.stderr)
        return 1

    print(f"Bootstrapping OAuth → {CREDENTIALS}")
    print("Abriendo browser. Aceptá los scopes y volvé a la terminal.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(OAUTH_KEYS), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    payload = {
        "access_token": creds.token,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": list(creds.scopes or SCOPES),
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }

    tmp = CREDENTIALS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(CREDENTIALS)

    print(f"\nOK — {CREDENTIALS} escrito.")
    print(f"  scopes: {payload['scopes']}")
    print(f"  refresh_token: {'set' if payload['refresh_token'] else 'MISSING'}")
    print(f"  expiry: {payload['expiry']}")
    print("\nProbalo: rag index --source calendar --reset --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
