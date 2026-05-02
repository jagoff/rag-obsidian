#!/usr/bin/env bash
# Ollama health watchdog — detecta cuelgues silenciosos del daemon ollama
# y los recupera automáticamente, sin tocar al rag-serve ni al web server.
#
# ── Por qué existe ──────────────────────────────────────────────────────
# 2026-05-01 18:21–18:30: Ollama se atascó silenciosamente (cola gigante
# de embeddings, /api/embed bge-m3 tardaba 30-58s, cualquier modelo de
# chat timeouteaba). El web server (`:8765/api/query`) se cuelga porque
# upstream Ollama no responde, y el listener de WhatsApp no puede generar
# drafts (`LLM call failed (todos los modelos): The operation timed out`).
# El usuario ve `rag query falló — code 1` y `❌ no pude generar draft`.
#
# Hubo un watchdog viejo (`rag-serve-watchdog.sh` + plist
# `com.fer.obsidian-rag-serve-watchdog`) que YA tenía la lógica correcta
# para esto: probe directo de `/api/embed` y kill de runners zombies.
# Pero ese plist se deprecó cuando se quitó el viejo `rag serve --port
# 7832` del stack (commit 1326d85), y la lógica de Ollama se fue al
# drain con él. Este script restaura esa defensa, dedicado y desacoplado
# del rag-serve.
#
# Hay también un watchdog interno en `web/server.py` (`_ollama_health.py`)
# que dispara con `p95_recent / p95_baseline >= 3x`. NO cubre el caso de
# Ollama totalmente caído (no hay datos recientes para calcular p95) ni
# las llamadas que hace el WhatsApp listener directo a `/api/embed` sin
# pasar por la FastAPI. Este watchdog es complementario, no redundante.
#
# ── Estrategia ──────────────────────────────────────────────────────────
# Cada tick (StartInterval 60s):
#   0. lsof :11434. Si hay >1 listener (típico: brew bindea IPv4 +
#      Ollama.app bindea IPv6 al mismo puerto), kill los PIDs de
#      Ollama.app + remover de login items. Sin cooldown — `pgrep` da
#      idempotencia. Bug recurrente: ambas instancias duplican modelos
#      en VRAM → embeds 30-90s, drafts WhatsApp timeoutean.
#   1. GET /api/tags timeout 5s. Si no responde 2xx → Ollama serve está
#      caído o pegado a TCP-level. Después de TAGS_FAIL_THRESHOLD ticks
#      consecutivos (default 3 = 3min), `brew services restart ollama`.
#      Cooldown 15min entre restarts.
#   2. POST /api/embed bge-m3 "ping" timeout 10s. Un embed warm corre en
#      <500ms; si tarda >10s o devuelve no-2xx, contador `embed_fails++`.
#      Después de EMBED_FAIL_THRESHOLD ticks (default 3 = 3min),
#      `pkill -9 -f 'ollama runner'`. Ollama serve respawnea los runners
#      al próximo call. Cooldown 5min entre kills.
#   3. POST /api/generate qwen2.5:7b "ok" timeout 15s, num_predict=1.
#      Cubre el caso donde embed anda pero chat se atascó. Mismo
#      tratamiento que embed.
#
# Grace period: durante POST_KILL_GRACE_SEC (120s) post-kill, los
# timeouts de embed/generate son cold-load esperado y NO cuentan como
# fail. Evita el bug donde el watchdog mata runners que ESTÁN cargando.
#
# Escalada: si kill_runners se dispara ESCALATE_KILLS_THRESHOLD veces
# consecutivas sin que embed vuelva a 200, el problema NO es solo
# runners zombies — Ollama serve mismo está roto. Saltamos kill y
# disparamos restart_serve completo (`brew services restart ollama`).
#
# Trade-off: el kill -9 de los runners cierra cualquier conexión TCP en
# vuelo (chat streaming, embed batch). Por eso los thresholds son
# conservadores (3-4 ticks consecutivos = 3-4 min) y los cooldowns son
# largos (5-15 min). El watchdog interno de _ollama_health.py YA
# respeta `in_flight_chats`, que es la defensa principal contra cortar
# streams. Este watchdog externo es la red de seguridad para casos donde
# el interno no se entera (Ollama totalmente caído, listener-only
# traffic, etc.).
#
# ── Knobs ──────────────────────────────────────────────────────────────
# Variables de entorno para tunear sin tocar el script:
#   OLLAMA_WATCHDOG_DISABLED=1                desactiva todo el watchdog
#   OLLAMA_TAGS_TIMEOUT=5                     timeout GET /api/tags
#   OLLAMA_EMBED_TIMEOUT=10                   timeout POST /api/embed
#   OLLAMA_GENERATE_TIMEOUT=15                timeout POST /api/generate
#   OLLAMA_TAGS_FAIL_THRESHOLD=3              ticks antes de restart serve
#   OLLAMA_EMBED_FAIL_THRESHOLD=3             ticks antes de kill runners
#   OLLAMA_GENERATE_FAIL_THRESHOLD=4          ticks antes de kill runners
#   OLLAMA_KILL_COOLDOWN_SEC=300              cooldown entre kill runners
#   OLLAMA_RESTART_COOLDOWN_SEC=900           cooldown entre restart serve
#   OLLAMA_POST_KILL_GRACE_SEC=120            grace post-kill (cold-load)
#   OLLAMA_ESCALATE_KILLS_THRESHOLD=2         kills sin recovery → restart
#   OLLAMA_GENERATE_PROBE_DISABLED=1          desactiva probe de chat
#   OLLAMA_DUP_DETECTION_DISABLED=1           desactiva probe 0 (kill Ollama.app)
#   OLLAMA_WATCHDOG_CATCHUP_DISABLED=1        desactiva el catchup nightly
#
# ── Estado ──────────────────────────────────────────────────────────────
# State file: ~/.local/share/obsidian-rag/ollama-watchdog.state (KEY=VALUE)
#   tags_fails, embed_fails, generate_fails  (contadores)
#   last_serve_restart, last_runners_kill    (timestamps unix)
#   kills_without_recovery                    (kills consecutivos sin OK)
#
# Catchup state: ~/.local/share/obsidian-rag/ollama-watchdog.catchup.state
#   last_catchup_<plist_label_suffix> = timestamp
#
# Logs: ~/.local/share/obsidian-rag/ollama-watchdog.log
#       ~/.local/share/obsidian-rag/ollama-watchdog.{stdout,stderr}.log

set -u

if [ -n "${OLLAMA_WATCHDOG_DISABLED:-}" ]; then
  exit 0
fi

STATE_DIR="${HOME}/.local/share/obsidian-rag"
STATE="${STATE_DIR}/ollama-watchdog.state"
LOG="${STATE_DIR}/ollama-watchdog.log"
CATCHUP_STATE="${STATE_DIR}/ollama-watchdog.catchup.state"

mkdir -p "${STATE_DIR}"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
TAGS_TIMEOUT="${OLLAMA_TAGS_TIMEOUT:-5}"
EMBED_TIMEOUT="${OLLAMA_EMBED_TIMEOUT:-10}"
GENERATE_TIMEOUT="${OLLAMA_GENERATE_TIMEOUT:-15}"
TAGS_FAIL_THRESHOLD="${OLLAMA_TAGS_FAIL_THRESHOLD:-3}"
EMBED_FAIL_THRESHOLD="${OLLAMA_EMBED_FAIL_THRESHOLD:-3}"
GENERATE_FAIL_THRESHOLD="${OLLAMA_GENERATE_FAIL_THRESHOLD:-4}"
KILL_COOLDOWN_SEC="${OLLAMA_KILL_COOLDOWN_SEC:-300}"
RESTART_COOLDOWN_SEC="${OLLAMA_RESTART_COOLDOWN_SEC:-900}"
POST_KILL_GRACE_SEC="${OLLAMA_POST_KILL_GRACE_SEC:-120}"
ESCALATE_KILLS_THRESHOLD="${OLLAMA_ESCALATE_KILLS_THRESHOLD:-2}"
EMBED_MODEL="${OLLAMA_WATCHDOG_EMBED_MODEL:-bge-m3}"
GENERATE_MODEL="${OLLAMA_WATCHDOG_GENERATE_MODEL:-qwen2.5:7b}"

log() {
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$1" >> "${LOG}"
}

# ── Probe 0: Duplicate-instance detection ──────────────────────────────
# Si hay >1 listener en :11434 (típicamente IPv4 + IPv6 separados),
# significa que están corriendo brew `ollama serve` Y `Ollama.app` al
# mismo tiempo. Cada uno carga sus propios modelos en VRAM → unified
# memory pressure → embeds 30-90s, drafts WhatsApp timeoutean
# (`LLM call failed (todos los modelos): The operation timed out`).
#
# Bug recurrente — primera vez 2026-04-30 (memoria
# `ollama_saturated_60s_embed_timeout_culprit_2_instancias_paralela`),
# volvió 2026-05-02 cuando el user reabrió Ollama.app sin querer y
# rompió 3 drafts seguidos (Fer F, Maria, Seba Serra). El watchdog
# detectaba "ollama lento" pero no la causa raíz, restartear brew no
# alcanzaba porque Ollama.app seguía consumiendo VRAM.
#
# Estrategia: el de brew es la instancia headless confiable (LaunchAgent
# que reinicia solo si crashea); la app es solo botón menubar / settings
# GUI. Si detectamos ambos, matamos la app y dejamos brew. La idempotencia
# está dada por `pgrep`: si la app ya no existe, no hay nada que matar.
#
# Override: `OLLAMA_DUP_DETECTION_DISABLED=1` apaga el probe (útil si
# alguien temporalmente quiere ambos para debug).
if [ -z "${OLLAMA_DUP_DETECTION_DISABLED:-}" ]; then
  listeners=$(lsof -nP -iTCP:11434 -sTCP:LISTEN 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
  if [ "${listeners:-0}" -gt 1 ]; then
    # Identificar PIDs de Ollama.app (Electron menubar + su `ollama serve` hijo + runners)
    app_pids=$(pgrep -f "Applications/Ollama.app" 2>/dev/null | tr '\n' ' ' | sed 's/ $//')
    if [ -n "${app_pids}" ]; then
      log "DUP_INSTANCE — listeners=${listeners} en :11434 (esperado=1). Ollama.app pids=[${app_pids}] — kill (queda brew solo)"
      # shellcheck disable=SC2086
      kill -9 ${app_pids} 2>>"${LOG}" || true
      # Bonus: removerla de login items para que no vuelva al próximo boot.
      # Idempotente — si ya está fuera, no falla.
      osascript -e 'tell application "System Events" to delete every login item whose name is "Ollama"' \
        >>"${LOG}" 2>&1 || true
      log "DUP_INSTANCE_OK — Ollama.app killed + removida de login items"
    else
      log "DUP_INSTANCE_UNKNOWN — listeners=${listeners} pero no encontré pids de Applications/Ollama.app (revisar manual)"
    fi
  fi
fi

# ── Read previous state ────────────────────────────────────────────────
tags_fails=0
embed_fails=0
generate_fails=0
last_serve_restart=0
last_runners_kill=0
kills_without_recovery=0
if [ -f "${STATE}" ]; then
  # shellcheck disable=SC1090
  source "${STATE}" 2>/dev/null || true
  tags_fails="${tags_fails:-0}"
  embed_fails="${embed_fails:-0}"
  generate_fails="${generate_fails:-0}"
  last_serve_restart="${last_serve_restart:-0}"
  last_runners_kill="${last_runners_kill:-0}"
  kills_without_recovery="${kills_without_recovery:-0}"
fi

now=$(date +%s)

# ── Probe 1: Ollama serve responde a /api/tags ─────────────────────────
tags_code=$(curl -sS -m "${TAGS_TIMEOUT}" -o /dev/null -w '%{http_code}' \
  "${OLLAMA_HOST}/api/tags" 2>/dev/null || echo "000")

if [[ "${tags_code}" =~ ^2 ]]; then
  if [ "${tags_fails}" -gt 0 ]; then
    log "TAGS_RECOVERED — http=${tags_code} (fails reset desde ${tags_fails})"
  fi
  tags_fails=0
else
  tags_fails=$(( tags_fails + 1 ))
  log "TAGS_UNHEALTHY — http=${tags_code} fails=${tags_fails}/${TAGS_FAIL_THRESHOLD} url=${OLLAMA_HOST}/api/tags"
fi

# ── Recovery escalation 1: restart Ollama serve ────────────────────────
since_serve_restart=$(( now - last_serve_restart ))
if [ "${tags_fails}" -ge "${TAGS_FAIL_THRESHOLD}" ] \
   && [ "${since_serve_restart}" -gt "${RESTART_COOLDOWN_SEC}" ]; then
  log "RESTARTING_SERVE — tags_fails=${tags_fails} since_last=${since_serve_restart}s"
  if /opt/homebrew/bin/brew services restart ollama 2>>"${LOG}" >>"${LOG}"; then
    log "RESTART_SERVE_OK — brew services restart ollama"
    last_serve_restart=${now}
    tags_fails=0
    embed_fails=0
    generate_fails=0
  else
    log "RESTART_SERVE_FAIL — brew services returned non-zero"
  fi
elif [ "${tags_fails}" -ge "${TAGS_FAIL_THRESHOLD}" ]; then
  log "WOULD_RESTART_SERVE_BUT_COOLDOWN — since_last=${since_serve_restart}s < ${RESTART_COOLDOWN_SEC}s"
fi

# Si Ollama serve está caído, no tiene sentido seguir con embed/generate.
if ! [[ "${tags_code}" =~ ^2 ]]; then
  log "TICK_OK — tags_fails=${tags_fails} (skip embed/generate probes)"
else
  # ── Grace period post-kill/restart ──────────────────────────────────
  in_grace=0
  grace_since=0
  last_recovery_action=$(( last_runners_kill > last_serve_restart ? last_runners_kill : last_serve_restart ))
  if [ "${last_recovery_action}" -gt 0 ]; then
    grace_since=$(( now - last_recovery_action ))
    if [ "${grace_since}" -lt "${POST_KILL_GRACE_SEC}" ]; then
      in_grace=1
    fi
  fi

  # ── Probe 2: /api/embed ───────────────────────────────────────────────
  embed_code=$(curl -sS -m "${EMBED_TIMEOUT}" -o /dev/null -w '%{http_code}' \
    -X POST "${OLLAMA_HOST}/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED_MODEL}\",\"input\":\"watchdog ping\"}" \
    2>/dev/null || echo "000")

  if [[ "${embed_code}" =~ ^2 ]]; then
    if [ "${embed_fails}" -gt 0 ]; then
      log "EMBED_RECOVERED — http=${embed_code} (fails reset desde ${embed_fails})"
    fi
    embed_fails=0
    if [ "${kills_without_recovery}" -gt 0 ]; then
      log "KILLS_RECOVERED — kills_without_recovery=${kills_without_recovery} → 0 (embed sano)"
      kills_without_recovery=0
    fi
  elif [ "${in_grace}" = "1" ]; then
    log "EMBED_GRACE — http=${embed_code} grace_since=${grace_since}s/${POST_KILL_GRACE_SEC}s (cold-load esperado, no cuento fail)"
  else
    embed_fails=$(( embed_fails + 1 ))
    log "EMBED_UNHEALTHY — http=${embed_code} fails=${embed_fails}/${EMBED_FAIL_THRESHOLD} model=${EMBED_MODEL}"
  fi

  # ── Probe 3: /api/generate ────────────────────────────────────────────
  generate_resp="skipped"
  if [ -z "${OLLAMA_GENERATE_PROBE_DISABLED:-}" ]; then
    generate_resp=$(curl -sS -m "${GENERATE_TIMEOUT}" -o /dev/null -w '%{http_code}' \
      -X POST "${OLLAMA_HOST}/api/generate" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${GENERATE_MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
      2>/dev/null || echo "000")

    if [[ "${generate_resp}" =~ ^2 ]]; then
      if [ "${generate_fails}" -gt 0 ]; then
        log "GENERATE_RECOVERED — http=${generate_resp} (fails reset desde ${generate_fails})"
      fi
      generate_fails=0
    elif [ "${in_grace}" = "1" ]; then
      log "GENERATE_GRACE — http=${generate_resp} grace_since=${grace_since}s/${POST_KILL_GRACE_SEC}s (cold-load esperado, no cuento fail)"
    else
      generate_fails=$(( generate_fails + 1 ))
      log "GENERATE_UNHEALTHY — http=${generate_resp} fails=${generate_fails}/${GENERATE_FAIL_THRESHOLD} model=${GENERATE_MODEL}"
    fi
  fi

  # ── Recovery escalation 2: kill runners zombies o restart_serve ──────
  since_kill=$(( now - last_runners_kill ))
  embed_triggered=0
  generate_triggered=0
  [ "${embed_fails}" -ge "${EMBED_FAIL_THRESHOLD}" ] && embed_triggered=1
  [ "${generate_fails}" -ge "${GENERATE_FAIL_THRESHOLD}" ] && generate_triggered=1

  if [ "${embed_triggered}" = "1" ] || [ "${generate_triggered}" = "1" ]; then
    reason=""
    [ "${embed_triggered}" = "1" ] && reason="embed_fails=${embed_fails}"
    [ "${generate_triggered}" = "1" ] && reason="${reason:+${reason}, }generate_fails=${generate_fails}"

    if [ "${kills_without_recovery}" -ge "${ESCALATE_KILLS_THRESHOLD}" ] \
       && [ "${since_serve_restart}" -gt "${RESTART_COOLDOWN_SEC}" ]; then
      log "ESCALATE_RESTART_SERVE — kills_without_recovery=${kills_without_recovery} (kill no resuelve, restart serve completo)"
      if /opt/homebrew/bin/brew services restart ollama 2>>"${LOG}" >>"${LOG}"; then
        log "RESTART_SERVE_OK — brew services restart ollama (escalado)"
        last_serve_restart=${now}
        tags_fails=0
        embed_fails=0
        generate_fails=0
        kills_without_recovery=0
      else
        log "RESTART_SERVE_FAIL — brew services returned non-zero (escalado)"
      fi
    elif [ "${since_kill}" -le "${KILL_COOLDOWN_SEC}" ]; then
      log "WOULD_KILL_RUNNERS_BUT_COOLDOWN — since_last=${since_kill}s < ${KILL_COOLDOWN_SEC}s reason=${reason}"
    else
      killed=$(pgrep -f 'ollama runner' | tr '\n' ' ' | sed 's/ $//')
      if [ -n "${killed}" ]; then
        # shellcheck disable=SC2086
        kill -9 ${killed} 2>>"${LOG}" || true
        log "KILL_RUNNERS_OK — pids=${killed} reason=${reason} (Ollama serve respawn auto)"
        last_runners_kill=${now}
        embed_fails=0
        generate_fails=0
        kills_without_recovery=$(( kills_without_recovery + 1 ))
      else
        # Runners-less zombie state — `ollama serve` responde HTTP pero
        # los runners están muertos (caso típico: kill -9 manual de
        # runners, app crash externa, OOM). `ollama ps` puede mentir
        # diciendo que los modelos siguen cargados, pero embed/generate
        # timeoutean. Sin escalation, el watchdog quedaría infinitamente
        # logueando NOOP y nunca salvaría la situación.
        #
        # Tratamos esto como un "kill sin recovery" virtual: incrementamos
        # el contador para que el próximo tick (con embed/generate
        # todavía rotos) escale a restart_serve completo.
        #
        # 2026-05-02 root cause: descubierto durante el cleanup de
        # duplicate-instance — al matar Ollama.app sus runners murieron,
        # brew `ollama serve` quedó solo y zombie (HTTP up, runners
        # gone). El watchdog detectó duplicate-instance pero no logró
        # auto-recover el zombie state hasta que parcheamos esto.
        log "KILL_RUNNERS_NOOP — pgrep no encontró ollama runner procs (zombie state — runners muertos pero serve up). reason=${reason}"
        kills_without_recovery=$(( kills_without_recovery + 1 ))
        embed_fails=0
        generate_fails=0
      fi
    fi
  fi

  if [ "${tags_fails}" -eq 0 ] && [ "${embed_fails}" -eq 0 ] && [ "${generate_fails}" -eq 0 ]; then
    log "OK — tags=${tags_code} embed=${embed_code} generate=${generate_resp}"
  fi
fi

# ── Nightly catchup — plists con StartCalendarInterval que se saltearon ─
# Heredado del watchdog viejo (rag-serve-watchdog.sh). macOS launchd se
# saltea schedules con StartCalendarInterval cuando la Mac está dormida
# en el horario programado (no hay catch-up retroactivo). Aprovechamos
# la cadencia de este watchdog para detectar plists nightly cuyo log no
# se actualizó hace > su window esperada + buffer; si pasa, kickstart
# manual.
#
# Cada entry: LABEL_SUFFIX:LOG_BASENAME:MAX_AGE_SECONDS
#   - LABEL_SUFFIX se concatena con `com.fer.obsidian-rag-`
#   - LOG_BASENAME es el archivo en ~/.local/share/obsidian-rag/
#   - MAX_AGE_SECONDS antes de considerarlo missed
#
# Si querés pausar el catchup: setear OLLAMA_WATCHDOG_CATCHUP_DISABLED=1.

if [ -z "${OLLAMA_WATCHDOG_CATCHUP_DISABLED:-}" ]; then
  UID_VAL="$(id -u)"
  DOMAIN="gui/${UID_VAL}"

  NIGHTLY_PLISTS=(
    # Diarios (max 26h = 24h ciclo + 2h buffer)
    "wake-up:wake-up.log:93600"
    "archive:archive.log:93600"
    "vault-cleanup:vault-cleanup.log:93600"
    "auto-harvest:auto-harvest.log:93600"
    "implicit-feedback:implicit-feedback.log:93600"
    "online-tune:online-tune.log:93600"
    "maintenance:maintenance.log:93600"
    "calibrate:calibrate.log:93600"
    # Lunes-viernes (max 50h = 48h fin de semana + buffer)
    "morning:morning.log:180000"
    "today:today.log:180000"
    # Semanales (max 8d = 7d ciclo + 1d buffer)
    "consolidate:consolidate.log:691200"
    "digest:digest.log:691200"
    "patterns:patterns.log:691200"
    "emergent:emergent.log:691200"
  )

  CATCHUP_COOLDOWN_SEC="${OLLAMA_WATCHDOG_CATCHUP_COOLDOWN_SEC:-3600}"  # 1h
  catchup_now="${now}"

  for entry in "${NIGHTLY_PLISTS[@]}"; do
    suffix="${entry%%:*}"
    rest="${entry#*:}"
    log_basename="${rest%%:*}"
    max_age="${rest##*:}"
    nightly_label="com.fer.obsidian-rag-${suffix}"
    nightly_log="${STATE_DIR}/${log_basename}"

    last_kick_var="last_catchup_${suffix//-/_}"
    last_kick=0
    if [ -f "${CATCHUP_STATE}" ]; then
      last_kick=$(grep "^${last_kick_var}=" "${CATCHUP_STATE}" 2>/dev/null \
                  | tail -1 | cut -d'=' -f2)
      last_kick="${last_kick:-0}"
    fi
    if [ $(( catchup_now - last_kick )) -lt "${CATCHUP_COOLDOWN_SEC}" ]; then
      continue
    fi

    if [ ! -f "${nightly_log}" ]; then
      age=$(( max_age + 1 ))
    else
      log_mtime=$(stat -f "%m" "${nightly_log}" 2>/dev/null || echo "0")
      age=$(( catchup_now - log_mtime ))
    fi

    if [ "${age}" -gt "${max_age}" ]; then
      log "CATCHUP — ${suffix} last_log_mtime=${age}s > ${max_age}s → kickstart"
      if launchctl kickstart "${DOMAIN}/${nightly_label}" 2>>"${LOG}"; then
        tmp_state="${CATCHUP_STATE}.tmp.$$"
        if [ -f "${CATCHUP_STATE}" ]; then
          grep -v "^${last_kick_var}=" "${CATCHUP_STATE}" > "${tmp_state}" 2>/dev/null || true
        fi
        printf '%s=%s\n' "${last_kick_var}" "${catchup_now}" >> "${tmp_state}"
        mv -f "${tmp_state}" "${CATCHUP_STATE}"
        log "CATCHUP_OK — kickstart ${nightly_label}"
      else
        log "CATCHUP_FAIL — ${nightly_label} (label registered? domain ok?)"
      fi
    fi
  done
fi

# ── Persist state ──────────────────────────────────────────────────────
tmp="${STATE}.tmp.$$"
{
  printf 'tags_fails=%s\n' "${tags_fails}"
  printf 'embed_fails=%s\n' "${embed_fails}"
  printf 'generate_fails=%s\n' "${generate_fails}"
  printf 'last_serve_restart=%s\n' "${last_serve_restart}"
  printf 'last_runners_kill=%s\n' "${last_runners_kill}"
  printf 'kills_without_recovery=%s\n' "${kills_without_recovery}"
} > "${tmp}"
mv -f "${tmp}" "${STATE}"
