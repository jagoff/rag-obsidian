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
# FAIL_THRESHOLD subido de 2→8 el 2026-04-30 después de un episodio donde
# el warmup en frío (bge-m3 + BM25 corpus + qwen2.5:7b) tardaba ~5min y el
# watchdog mataba el proceso justo antes de que termine, generando un loop
# de 8 restarts en 30min. 8 ticks × 60s = 8min de paciencia antes de
# considerar un restart. El cooldown subió en paralelo (300→900).
FAIL_THRESHOLD="${RAG_SERVE_FAIL_THRESHOLD:-8}"
COOLDOWN_SEC="${RAG_SERVE_COOLDOWN_SEC:-900}"
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

# ── Nightly catchup — plists con StartCalendarInterval que se saltearon ────
# macOS launchd se saltea schedules con StartCalendarInterval cuando la
# Mac está dormida en el horario programado (no hay catch-up retroactivo).
# Para mac laptops esto pasa típicamente con plists nightly como wake-up
# (4am) o consolidate (lunes 6am) — observado el 2026-04-27 con `runs=0`
# después de 17h+ instalado.
#
# Estrategia: este watchdog ya corre cada 60s. Aprovechamos la cadencia
# para detectar plists nightly cuyo log no se actualizó hace > su window
# esperada + buffer. Si pasa, kickstart manual.
#
# Cada entry: LABEL_SUFFIX:LOG_BASENAME:MAX_AGE_SECONDS
#   - LABEL_SUFFIX se concatena con `com.fer.obsidian-rag-`
#   - LOG_BASENAME es el archivo en ~/.local/share/obsidian-rag/
#   - MAX_AGE_SECONDS antes de considerarlo missed
#
# Schedules conocidos:
#   wake-up:     daily 4am           → 26h max age (24h ciclo + 2h buffer)
#   archive:     daily 23h           → 26h max age
#   consolidate: weekly mon 6am      → 8d max age (7d ciclo + 1d buffer)
#   digest:      weekly sun 22h      → 8d max age
#   patterns:    weekly sun 20h      → 8d max age
#   emergent:    weekly fri 10am     → 8d max age
#
# Si querés pausar el catchup: setear RAG_WATCHDOG_CATCHUP_DISABLED=1.

if [ -z "${RAG_WATCHDOG_CATCHUP_DISABLED:-}" ]; then
  # Schedule sensitivity por plist. Lista expandida 2026-04-27 después
  # del audit que detectó 8 plists schedule-sensitive faltantes (audit
  # subagent 2297bb6e). El bug original eran wake-up + consolidate
  # missing por Mac dormida, pero el problema es general: TODOS los
  # plists con StartCalendarInterval pueden saltearse si la Mac no
  # está despierta a esa hora.
  #
  # Schedules cubiertos (LABEL_SUFFIX:LOG_BASENAME:MAX_AGE_SECONDS):
  #   Diarios (max 26h = 24h ciclo + 2h buffer):
  #     wake-up         daily 4am
  #     archive         daily 23h (NOTA: schedule real es día 1 del mes
  #                     según audit, pero el log es diario por activity
  #                     interna; usamos 26h = funciona aunque sobre-trigger)
  #     vault-cleanup   daily 2am
  #     auto-harvest    daily 3am
  #     implicit-feedback daily 3:25am
  #     online-tune     daily 3:30am (fix del timeout 1200→2400 también
  #                     necesita esto para correr esta noche)
  #     maintenance     daily 4am
  #     calibrate       daily 4:30am
  #
  #   Lunes-viernes (max 50h = 48h fin de semana + buffer):
  #     morning         L-V 7am
  #     today           L-V 22h
  #
  #   Semanales (max 8d = 7d ciclo + 1d buffer):
  #     consolidate     weekly mon 6am
  #     digest          weekly sun 22h
  #     patterns        weekly sun 20h
  #     emergent        weekly fri 10am
  NIGHTLY_PLISTS=(
    # Diarios
    "wake-up:wake-up.log:93600"
    "archive:archive.log:93600"
    "vault-cleanup:vault-cleanup.log:93600"
    "auto-harvest:auto-harvest.log:93600"
    "implicit-feedback:implicit-feedback.log:93600"
    "online-tune:online-tune.log:93600"
    "maintenance:maintenance.log:93600"
    "calibrate:calibrate.log:93600"
    # Lunes-viernes
    "morning:morning.log:180000"
    "today:today.log:180000"
    # Semanales
    "consolidate:consolidate.log:691200"
    "digest:digest.log:691200"
    "patterns:patterns.log:691200"
    "emergent:emergent.log:691200"
  )

  catchup_state="${STATE_DIR}/serve-watchdog.catchup.state"
  # Cooldown por plist — evita kickstart repetido si el plist en sí está
  # tardando mucho. Si ya kickstart en los últimos N segundos, skip.
  CATCHUP_COOLDOWN_SEC="${RAG_CATCHUP_COOLDOWN_SEC:-3600}"  # 1h
  catchup_now=$(date +%s)

  for entry in "${NIGHTLY_PLISTS[@]}"; do
    suffix="${entry%%:*}"
    rest="${entry#*:}"
    log_basename="${rest%%:*}"
    max_age="${rest##*:}"
    nightly_label="com.fer.obsidian-rag-${suffix}"
    nightly_log="${STATE_DIR}/${log_basename}"

    # ¿Ya kickstart-eamos este plist recientemente? Skip si sí.
    last_kick_var="last_catchup_${suffix//-/_}"
    last_kick=0
    if [ -f "${catchup_state}" ]; then
      last_kick=$(grep "^${last_kick_var}=" "${catchup_state}" 2>/dev/null \
                  | tail -1 | cut -d'=' -f2)
      last_kick="${last_kick:-0}"
    fi
    if [ $(( catchup_now - last_kick )) -lt "${CATCHUP_COOLDOWN_SEC}" ]; then
      continue
    fi

    # Determinar la "edad" del último run. Si el log no existe O su mtime
    # es viejo, considerar missed. NOTA: si no hay log, seteamos age a
    # max_age + 1 para que el `-gt` triggee (con `-ge` también andaría
    # pero `-gt` deja un buffer natural — un plist que JUSTO terminó
    # hace `max_age` segundos no se kickstart hasta el próximo tick).
    if [ ! -f "${nightly_log}" ]; then
      age=$(( max_age + 1 ))  # asegurar que > max_age → trigger
    else
      log_mtime=$(stat -f "%m" "${nightly_log}" 2>/dev/null || echo "0")
      age=$(( catchup_now - log_mtime ))
    fi

    if [ "${age}" -gt "${max_age}" ]; then
      log "CATCHUP    — ${suffix} last_log_mtime=${age}s > ${max_age}s → kickstart"
      if launchctl kickstart "${DOMAIN}/${nightly_label}" 2>>"${LOG}"; then
        # Append al state file (read all entries, replace this one)
        tmp_state="${catchup_state}.tmp.$$"
        if [ -f "${catchup_state}" ]; then
          grep -v "^${last_kick_var}=" "${catchup_state}" > "${tmp_state}" 2>/dev/null || true
        fi
        printf '%s=%s\n' "${last_kick_var}" "${catchup_now}" >> "${tmp_state}"
        mv -f "${tmp_state}" "${catchup_state}"
        log "CATCHUP_OK — kickstart ${nightly_label}"
      else
        log "CATCHUP_FAIL — ${nightly_label} (label registered? domain ok?)"
      fi
    fi
  done
fi

# Persist state. Sobreescribe atómicamente via temp+rename.
tmp="${STATE}.tmp.$$"
{
  printf 'fails=%s\n' "${fails}"
  printf 'last_restart=%s\n' "${last_restart}"
} > "${tmp}"
mv -f "${tmp}" "${STATE}"
