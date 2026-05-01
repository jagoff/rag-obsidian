#!/usr/bin/env python3
"""Wake hook — kickstart-overdue post-Mac-wake.

launchd `StartCalendarInterval` no dispara retroactivamente cuando el
Mac estuvo dormido a la hora del slot — los daemons calendar-driven
(morning 7am, today 22hs, archive weekly, etc) se pierden runs cuando
la laptop estuvo cerrada. El watchdog cada 5min mitiga, pero igual hay
hasta 5min de lag post-wake.

Este script corre como daemon long-running (KeepAlive=true) en un loop
sleep 60s. Cada tick lee `pmset -g log | grep "Display is turned on"`
y se queda con la fecha de la última. Compara contra el state file
`~/.local/share/obsidian-rag/wake-hook-state.json`. Si hay un wake nuevo
desde el último check, dispara `rag daemons kickstart-overdue` para
catchup inmediato.

"Display is turned on" se elige sobre "Wake from Normal Sleep" porque
representa wakes user-visible (no DarkWake de mantenimiento), que es
el caso donde el user razonablemente espera que los daemons se hayan
ejecutado.

Silent-fail por todos lados: si pmset falla, log un warning y seguir;
si el state file está corrupto, recrearlo; si el subprocess de
`rag daemons kickstart-overdue` rebota, log + seguir. Nunca crashear el
loop principal.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LOG = logging.getLogger("rag.wake_hook")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [wake-hook] %(levelname)s %(message)s",
    stream=sys.stderr,
)

STATE_PATH = Path.home() / ".local/share/obsidian-rag/wake-hook-state.json"
POLL_SECONDS = int(os.environ.get("RAG_WAKE_HOOK_POLL_SECONDS", "60"))
RAG_BIN = os.environ.get("RAG_WAKE_HOOK_RAG_BIN", "/Users/fer/.local/bin/rag")

_DISPLAY_ON_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})\s+Notification\s+Display is turned on"
)


def _read_state() -> dict:
    """Lee el state file. Si no existe / está corrupto → dict vacío."""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(state: dict) -> None:
    """Escribe el state file. Silent-fail."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        LOG.warning("write_state failed: %s", exc)


def _last_display_on() -> str | None:
    """Última fecha de "Display is turned on" en pmset log. None si no hay."""
    # Timeout 30s: en idle `pmset -g log` tarda ~4.5s sobre 116k líneas,
    # pero post-wake con IO contention y daemons fork-bombing en paralelo
    # puede pasar de 15s. Subir a 30s mantiene el silent-fail como red
    # de seguridad sin que se gatille por load transitorio normal.
    try:
        proc = subprocess.run(
            ["pmset", "-g", "log"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOG.warning("pmset failed: %s", exc)
        return None
    if proc.returncode != 0:
        LOG.warning("pmset exit=%s stderr=%s", proc.returncode, proc.stderr[:200])
        return None
    last_match: str | None = None
    for line in proc.stdout.splitlines():
        m = _DISPLAY_ON_RE.match(line.strip())
        if m:
            last_match = m.group(1)
    return last_match


def _kickstart_overdue() -> bool:
    """Llama `rag daemons kickstart-overdue`. Devuelve True si exit 0."""
    try:
        proc = subprocess.run(
            [RAG_BIN, "daemons", "kickstart-overdue"],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOG.warning("kickstart-overdue failed: %s", exc)
        return False
    if proc.returncode != 0:
        LOG.warning(
            "kickstart-overdue exit=%s stderr=%s",
            proc.returncode, proc.stderr[:300],
        )
        return False
    LOG.info("kickstart-overdue ok: %s", (proc.stdout or "").strip()[:300])
    return True


def check_once() -> bool:
    """Una iteración del check. Devuelve True si disparó kickstart."""
    last_wake = _last_display_on()
    if not last_wake:
        return False
    state = _read_state()
    prev_wake = state.get("last_wake")
    if prev_wake == last_wake:
        return False
    LOG.info("wake detected: %s (prev=%s)", last_wake, prev_wake or "none")
    fired = _kickstart_overdue()
    state["last_wake"] = last_wake
    state["last_check_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["last_kickstart_ok"] = fired
    _write_state(state)
    return fired


def main() -> int:
    """Loop forever: check_once + sleep POLL_SECONDS."""
    LOG.info(
        "starting wake hook (poll=%ss state=%s rag=%s)",
        POLL_SECONDS, STATE_PATH, RAG_BIN,
    )
    # Initial bootstrap: store current wake without firing (evita kickstart
    # spurious al instalar el daemon).
    state = _read_state()
    if "last_wake" not in state:
        initial = _last_display_on()
        if initial:
            state["last_wake"] = initial
            state["last_check_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state["bootstrap"] = True
            _write_state(state)
            LOG.info("bootstrap: anchored to last_wake=%s", initial)
    while True:
        try:
            check_once()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("check_once unexpected exception: %s", exc)
        time.sleep(POLL_SECONDS)
    # unreachable
    return 0


if __name__ == "__main__":
    sys.exit(main())
