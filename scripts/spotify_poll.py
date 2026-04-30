#!/usr/bin/env python3
"""Spotify poller — fires `record_now_playing()` once per invocation.

Driven by `~/Library/LaunchAgents/com.fer.obsidian-rag-spotify-poll.plist`
con `StartInterval=60` (corre cada 60s mientras el laptop está despierto).
Exit codes:
  0 — record succeeded (insert o update) o estado esperado (Spotify cerrado,
      track paused). NO error condition.
  1 — error inesperado (DB lock crónico, AppleScript misbehaving, etc).
      Launchd lo loguea pero no reintenta — el próximo tick (60s) reintenta solo.

Output: una línea JSON-ish a stdout cuando RECORD ocurre. Cuando state=paused
o spotify=closed, NO escribe nada (evita inflar el log con ruido).
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        from rag.integrations.spotify_local import record_now_playing
    except Exception as exc:
        # Bootstrap error (rag no instalado, env mal). Los siguientes ticks
        # van a fallar igual hasta que el user fixe — no spamear el log.
        print(f"import_error: {exc}", file=sys.stderr)
        return 1
    try:
        result = record_now_playing()
    except Exception as exc:
        # Defensive: spotify_local.record_now_playing() ya hace silent-fail
        # internamente. Este except es belt-and-suspenders.
        print(f"unexpected_error: {exc}", file=sys.stderr)
        return 1
    if result.get("recorded"):
        # Solo loguear cuando ALGO se grabó (insert o update) — los polls
        # idle (Spotify cerrado, paused) no aportan ruido al log.
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
