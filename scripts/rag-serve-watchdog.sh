#!/usr/bin/env bash
# Watchdog externo de `rag serve` — detecta cuelgues silenciosos donde el
# proceso sigue vivo (launchctl list muestra PID, exit 0) pero el HTTP
# server dejó de responder.
#
# ¿Qué detecta que KeepAlive=true NO detecta?
#   KeepAlive reinicia si el proceso MUERE. Pero `rag serve` puede colgarse
#   bajo carga sostenida (queries pesadas en sucesión rápida → MPS deadlock,
#   queue interna full, GIL bloqueado en algún C extension). En ese caso
#   launchctl lista un PID pero ningún cliente recibe respuesta. El listener
#   de WhatsApp queda esperando 120s (RAG_TIMEOUT_MS) en cada query, después
#   cae al fallback subprocess que también es lento → toda la pipeline
#   degradada.
#
# Estrategia:
#   1. HTTP GET a /health con timeout corto (3s).
#   2. Si responde 2xx → state.fails = 0 (sano).
#   3. Si NO responde → state.fails++ . Loguea.
#   4. Si fails >= FAIL_THRESHOLD (default 2) Y no estamos en cooldown
#      (>5min desde el último kickstart) → kickstart -k. Reset fails.
#   5. Si fails >= threshold pero cooldown activo → loguea "would restart"
#      sin actuar (evita restart loops cuando el serve necesita warm-up).
#
# Estado persistente: ~/.local/share/obsidian-rag/serve-watchdog.state
#   formato: KEY=VALUE por línea. Variables: fails, last_restart.
#
# Logs: ~/.local/share/obsidian-rag/serve-watchdog.log
#
# Instalación: corre via launchd plist propio
# (com.fer.obsidian-rag-serve-watchdog) con StartInterval=60.
#
# Diseñado el 2026-04-27 después de un test E2E donde rag serve se colgó
# 3 veces durante una batería de queries pesadas (/search docker, brief
# de cierre, etc.). Ver
# `00-Inbox/RAG - Flujo WhatsApp - auditoría.md` del vault.

set -u

URL="${RAG_SERVE_HEALTH_URL:-http://127.0.0.1:7832/health}"
TIMEOUT="${RAG_SERVE_HEALTH_TIMEOUT:-3}"
FAIL_THRESHOLD="${RAG_SERVE_FAIL_THRESHOLD:-2}"
COOLDOWN_SEC="${RAG_SERVE_COOLDOWN_SEC:-300}"
LABEL="com.fer.obsidian-rag-serve"

UID_VAL="$(id -u)"
DOMAIN="gui/${UID_VAL}"
STATE_DIR="${HOME}/.local/share/obsidian-rag"
STATE="${STATE_DIR}/serve-watchdog.state"
LOG="${STATE_DIR}/serve-watchdog.log"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$1" >> "${LOG}"
}

# Read previous state (defaults to fails=0, last_restart=0 si no existe).
fails=0
last_restart=0
if [ -f "${STATE}" ]; then
  # shellcheck disable=SC1090
  source "${STATE}" 2>/dev/null || true
  fails="${fails:-0}"
  last_restart="${last_restart:-0}"
fi

# Health check. -o /dev/null descarta el body, -w '%{http_code}' devuelve solo
# el status code. Cualquier 2xx cuenta como sano.
http_code=$(curl -sS -m "${TIMEOUT}" -o /dev/null -w '%{http_code}' "${URL}" 2>/dev/null || echo "000")

if [[ "${http_code}" =~ ^2 ]]; then
  if [ "${fails}" -gt 0 ]; then
    log "RECOVERED  — http=${http_code} (fails reset desde ${fails})"
  else
    # Señal de vida del watchdog: cada tick loguea "OK". Mismo patrón que
    # `whatsapp-listener-healthcheck.sh`. Ayuda a detectar si el watchdog
    # mismo dejó de correr (mtime del log stale).
    log "OK  — http=${http_code}"
  fi
  fails=0
else
  fails=$(( fails + 1 ))
  log "UNHEALTHY  — http=${http_code} fails=${fails}/${FAIL_THRESHOLD} url=${URL}"
fi

# Decide restart.
now=$(date +%s)
since_last=$(( now - last_restart ))

if [ "${fails}" -ge "${FAIL_THRESHOLD}" ]; then
  if [ "${since_last}" -gt "${COOLDOWN_SEC}" ]; then
    log "RESTARTING — kickstart -k ${DOMAIN}/${LABEL} (fails=${fails}, since_last=${since_last}s)"
    if launchctl kickstart -k "${DOMAIN}/${LABEL}" 2>>"${LOG}"; then
      last_restart=$now
      fails=0
      log "RESTART_OK  — kickstart issued, sleeping fails=0 + cooldown ${COOLDOWN_SEC}s"
    else
      log "RESTART_FAIL — kickstart returned non-zero (label registered? domain ok?)"
    fi
  else
    log "WOULD_RESTART_BUT_COOLDOWN — since_last=${since_last}s < ${COOLDOWN_SEC}s; serve probablemente warmeando todavía"
  fi
fi

# Persist state. Sobreescribe atómicamente via temp+rename.
tmp="${STATE}.tmp.$$"
{
  printf 'fails=%s\n' "${fails}"
  printf 'last_restart=%s\n' "${last_restart}"
} > "${tmp}"
mv -f "${tmp}" "${STATE}"
