#!/usr/bin/env python3
"""Mood poller — fires `rag.mood.run_poll_cycle()` once per invocation.

Driven by `~/Library/LaunchAgents/com.fer.obsidian-rag-mood-poll.plist`
con `StartInterval=1800` (30 min mientras el laptop está despierto).

Exit codes:
  0 — cycle completed (incluso si el feature está off / daemon disabled
      / scorers no encontraron señales). NO error condition.
  1 — error inesperado (DB lock crónico, ImportError, etc). Launchd lo
      loguea pero no reintenta — el próximo tick (30min) reintenta.

Output a stdout: 1 línea JSON con el summary del cycle. Si el feature
está off o el daemon está disabled, NO escribe nada (evita inflar el
log con ruido cuando el user no opt-inó).

Toggle:
  rag mood enable    # crea ~/.local/share/obsidian-rag/mood_enabled
  rag mood disable   # borra el state file
  rag mood status    # muestra si está enabled

El plist tiene `RAG_MOOD_ENABLED=1` cableado, así que prender el daemon
es solo el toggle del state file. No requiere editar el plist ni
recargar launchd.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        from rag.mood import run_poll_cycle
    except Exception as exc:
        # Bootstrap error (rag no instalado, env mal). Los próximos
        # ticks van a fallar igual hasta que el user fixe — no spamear
        # el log con la misma exception cada 30min.
        print(f"import_error: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_poll_cycle()
    except Exception as exc:
        # Defensive: run_poll_cycle ya hace silent-fail interno por
        # scorer. Este except es belt-and-suspenders.
        print(f"unexpected_error: {exc}", file=sys.stderr)
        return 1

    # Si el daemon está disabled o el feature off, no escribir nada
    # — el log se mantendría vacío hasta que el user opt-in. Si hay
    # signals reales, sí logueamos el summary 1 línea.
    reason = result.get("reason")
    if reason in ("feature_off", "daemon_disabled"):
        return 0

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
