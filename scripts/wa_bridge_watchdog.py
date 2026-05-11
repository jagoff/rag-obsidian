#!/usr/bin/env python3
"""WhatsApp bridge watchdog — detecta websocket-down y reinicia.

El binary `./whatsapp-bridge` se mantiene vivo via launchd plist
(`com.fer.whatsapp-bridge`, `KeepAlive=true`), pero whatsmeow no
auto-reconnect cuando el websocket muere con `EOF`. Síntoma observado
(2026-05-11): proceso vivo, log muestra
``Error reading from websocket: failed to read frame header: EOF``
seguido de ``Message sent false Not connected to WhatsApp``, nada más
se procesa. El user perdió 3 horas de inbound silenciosamente.

Este script tail-ea el log del bridge y cuando ve el pattern de
desconexión sin auto-recovery en N segundos, hace

    launchctl kickstart -k gui/$UID/com.fer.whatsapp-bridge

para forzar restart. La sesión persiste en
``/Users/fer/repos/whatsapp-mcp/whatsapp-bridge/store/`` así que el
restart NO requiere re-pairing con QR (a menos que la sesión expire
por inactividad larga, caso en el que el watchdog igual restartea y
el user verá el QR en el log).

Cada `CHECK_INTERVAL_S` segundos:
  1. Lee últimas 200 líneas del log.
  2. Encuentra timestamp del último ``Error reading from websocket``
     o ``Not connected to WhatsApp``.
  3. Encuentra timestamp del último evento "healthy" (Connected to
     WhatsApp / Stored message / Successfully downloaded).
  4. Si último error > último healthy AND error > `SETTLE_DELAY_S`
     segundos antiguo → dispara kickstart.
  5. ``cooldown_until`` evita restart-storms (mínimo 5min entre
     kickstarts consecutivos).

Logs propios → ``~/.local/share/obsidian-rag/wa-bridge-watchdog.log``
(rotación responsibility del plist).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


LOG_PATH = Path("/Users/fer/.local/share/obsidian-rag/wa-bridge.log")
STATE_PATH = Path.home() / ".local/share/obsidian-rag/wa-bridge-watchdog.state.json"
BRIDGE_LABEL = "com.fer.whatsapp-bridge"
CHECK_INTERVAL_S = 60       # cuánto duerme entre pasadas
SETTLE_DELAY_S = 120        # cuánto esperar tras un EOF antes de actuar (deja chance a auto-reconnect)
COOLDOWN_S = 300            # mínimo entre kickstarts consecutivos

# Patterns que marcan "websocket caído". Lista whitelist — solo lo que
# confirmamos en logs reales del bridge.
DISCONNECT_PATTERNS = [
    re.compile(r"Error reading from websocket"),
    re.compile(r"Not connected to WhatsApp"),
]
# Eventos que confirman que el bridge está procesando msgs.
HEALTHY_PATTERNS = [
    re.compile(r"\[Client INFO\] Connected to WhatsApp"),
    re.compile(r"Stored message:"),
    re.compile(r"Successfully downloaded"),
    re.compile(r"Message sent true"),
]

# Timestamp parser. Bridge log empieza cada línea con "HH:MM:SS.fff" en
# tiempo local (no incluye fecha). Asumimos fecha == hoy local; si hay
# rollover de medianoche entre dos eventos, los timestamps pueden volver
# atrás — el watchdog tolera eso porque solo nos interesa "qué tan
# viejo es el último error" en magnitud absoluta y `now()` siempre va
# para adelante.
_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")


def _parse_log_ts(line: str) -> datetime | None:
    m = _TS_RE.match(line)
    if not m:
        return None
    today = datetime.now().replace(microsecond=0)
    try:
        return today.replace(
            hour=int(m.group(1)),
            minute=int(m.group(2)),
            second=int(m.group(3)),
        )
    except ValueError:
        return None


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state))
    except OSError as e:
        print(f"[watchdog] state save failed: {e}", flush=True)


def _detect_state(log_path: Path) -> tuple[str, datetime | None]:
    """Devuelve ('healthy', None) o ('disconnected', error_ts).

    Lee últimas 200 líneas del log via `tail -n` (más rápido que abrir
    archivo de varios MB en Python).
    """
    if not log_path.is_file():
        return "healthy", None  # log inexistente todavía — bridge no arrancó
    try:
        proc = subprocess.run(
            ["tail", "-n", "200", str(log_path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[watchdog] tail failed: {e}", flush=True)
        return "healthy", None  # fail open — no actuar si no podemos leer
    lines = proc.stdout.splitlines()
    last_disconnect: datetime | None = None
    last_healthy: datetime | None = None
    for line in lines:
        ts = _parse_log_ts(line)
        if not ts:
            continue
        if any(p.search(line) for p in DISCONNECT_PATTERNS):
            last_disconnect = ts
        elif any(p.search(line) for p in HEALTHY_PATTERNS):
            last_healthy = ts
    if last_disconnect and (last_healthy is None or last_disconnect > last_healthy):
        return "disconnected", last_disconnect
    return "healthy", None


def _kickstart() -> bool:
    """Dispara `launchctl kickstart -k` sobre el plist del bridge.

    Devuelve True si el comando devolvió rc=0.
    """
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{BRIDGE_LABEL}"],
            capture_output=True, text=True, timeout=10,
        )
        ok = proc.returncode == 0
        print(
            f"[watchdog] kickstart rc={proc.returncode} "
            f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()!r}",
            flush=True,
        )
        return ok
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[watchdog] kickstart failed: {e}", flush=True)
        return False


def main() -> int:
    print(
        f"[watchdog] starting · log={LOG_PATH} · "
        f"check_interval={CHECK_INTERVAL_S}s · settle={SETTLE_DELAY_S}s · "
        f"cooldown={COOLDOWN_S}s",
        flush=True,
    )
    state = _load_state()
    while True:
        try:
            tick_state, err_ts = _detect_state(LOG_PATH)
            now = datetime.now()
            if tick_state == "disconnected" and err_ts is not None:
                age_s = (now - err_ts).total_seconds()
                last_action_iso = state.get("last_kickstart_at")
                in_cooldown = False
                if last_action_iso:
                    try:
                        last_action = datetime.fromisoformat(last_action_iso)
                        in_cooldown = (now - last_action).total_seconds() < COOLDOWN_S
                    except ValueError:
                        pass
                if age_s >= SETTLE_DELAY_S and not in_cooldown:
                    print(
                        f"[watchdog] disconnect detected · error_age={age_s:.0f}s · "
                        f"firing kickstart",
                        flush=True,
                    )
                    if _kickstart():
                        state["last_kickstart_at"] = now.isoformat(timespec="seconds")
                        state["last_kickstart_reason"] = "disconnect"
                        _save_state(state)
                elif in_cooldown:
                    print(
                        f"[watchdog] disconnect but in cooldown ({COOLDOWN_S}s) — skip",
                        flush=True,
                    )
                else:
                    print(
                        f"[watchdog] disconnect age {age_s:.0f}s < settle {SETTLE_DELAY_S}s — wait",
                        flush=True,
                    )
            # tick "healthy" no log para no ruido — el watchdog runs cada minuto
        except Exception as e:
            print(f"[watchdog] tick error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
