#!/usr/bin/env python3
"""Bootstrap OAuth para Google (Gmail + Drive readonly) — abre browser,
el user autoriza, escribe ~/.config/obsidian-rag/google_token.json fresco.

Uso:
    .venv/bin/python scripts/bootstrap_google_oauth.py

Requiere ~/.gmail-mcp/gcp-oauth.keys.json (las creds de la app GCP).
Sobrescribe ~/.config/obsidian-rag/google_token.json si existe (tras revocación).
"""
from __future__ import annotations

import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

OAUTH_KEYS = Path.home() / ".gmail-mcp" / "gcp-oauth.keys.json"
TOKEN_PATH = Path.home() / ".config" / "obsidian-rag" / "google_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def main() -> int:
    if not OAUTH_KEYS.is_file():
        print(f"ERROR: {OAUTH_KEYS} no existe.", file=sys.stderr)
        return 1

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Bootstrapping Google OAuth (Gmail + Drive) → {TOKEN_PATH}")
    print("Abriendo browser. Aceptá los scopes y volvé a la terminal.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(OAUTH_KEYS), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    TOKEN_PATH.write_text(creds.to_json())
    TOKEN_PATH.chmod(0o600)
    print(f"\n✓ Token escrito en {TOKEN_PATH}")
    print(f"  Scopes: {', '.join(creds.scopes or SCOPES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
