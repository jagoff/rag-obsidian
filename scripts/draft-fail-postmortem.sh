#!/usr/bin/env bash
# Captura snapshot completo del sistema cuando un draft fail (o zombie
# state de ollama) es detectado. Llamado por ollama-watchdog.sh en
# eventos críticos.
#
# Por qué existe: el user reportaba "drafts WA fallan muy seguido" y
# para diagnosticar yo necesitaba estar online en el momento del fail
# (procesos vivos, vmmap reciente, etc). Este script captura todo
# automáticamente para que post-incident pueda inspeccionar sin
# depender de timing humano.
#
# Output: ~/.local/share/obsidian-rag/draft-postmortems/<timestamp>/
# Auto-rotación: mantiene los últimos 20 (más viejos se borran).
#
# Llamado con: bash draft-fail-postmortem.sh "<reason>"

set -u
REASON="${1:-unknown}"
TS="$(date +%Y%m%d_%H%M%S)"
DIR="${HOME}/.local/share/obsidian-rag/draft-postmortems/${TS}_${REASON// /_}"
mkdir -p "${DIR}"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $1" >> "${DIR}/_meta.log"; }
log "REASON=${REASON}"

# 1) Memory state global
{
  echo "=== vm_stat ==="
  vm_stat
  echo
  echo "=== swap ==="
  sysctl vm.swapusage
  echo
  echo "=== top -o mem (top 30) ==="
  top -l 1 -o mem -stats pid,mem,rsize,vsize,command -n 30
} > "${DIR}/00-mem-state.txt" 2>&1

# 2) Procesos críticos
{
  echo "=== ollama serve + runners ==="
  ps -ef | grep -E "ollama (serve|runner)" | grep -v grep
  echo
  echo "=== web/server.py + rag procs ==="
  ps -ef | grep -E "web/server\.py|rag (watch|anticipate|chat|do)" | grep -v grep
  echo
  echo "=== whatsapp-listener ==="
  ps -ef | grep -E "whatsapp-listener" | grep -v grep
  echo
  echo "=== whisper-server ==="
  ps -ef | grep -E "whisper-server" | grep -v grep
  echo
  echo "=== launchctl list relevantes ==="
  launchctl list | grep -iE "ollama|fer\.obsidian|whatsapp|whisper"
} > "${DIR}/01-procs.txt" 2>&1

# 3) Ollama state
{
  echo "=== lsof :11434 ==="
  lsof -nP -iTCP:11434 -sTCP:LISTEN
  echo
  echo "=== ollama ps ==="
  /opt/homebrew/bin/ollama ps 2>&1
  echo
  echo "=== /api/tags (rapid health) ==="
  curl -s -m 3 http://127.0.0.1:11434/api/tags | head -c 1000
  echo
  echo
  echo "=== /api/ps (que dice ollama de los modelos) ==="
  curl -s -m 3 http://127.0.0.1:11434/api/ps | head -c 2000
} > "${DIR}/02-ollama.txt" 2>&1

# 4) Ollama internal log (últimas 100 líneas)
tail -n 100 /opt/homebrew/var/log/ollama.log > "${DIR}/03-ollama-log.txt" 2>&1

# 5) Listener error log (últimas 30 líneas)
tail -n 30 ~/.local/share/whatsapp-listener/listener.error.log > "${DIR}/04-listener-errors.txt" 2>&1
tail -n 50 ~/.local/share/whatsapp-listener/listener.log > "${DIR}/05-listener-stdout.txt" 2>&1

# 6) vmmap del web/server.py (la fuente más sospechosa por el leak MPS)
WEB_PID=$(pgrep -f "web/server\.py" | head -1)
if [ -n "${WEB_PID}" ]; then
  /usr/bin/vmmap --summary "${WEB_PID}" > "${DIR}/06-vmmap-web.txt" 2>&1
fi

# 7) vmmap de rag watch
WATCH_PID=$(pgrep -f "rag watch" | head -1)
if [ -n "${WATCH_PID}" ]; then
  /usr/bin/vmmap --summary "${WATCH_PID}" > "${DIR}/07-vmmap-watch.txt" 2>&1
fi

# 8) Watchdog state + log
cp ~/.local/share/obsidian-rag/ollama-watchdog.state "${DIR}/08-watchdog-state.txt" 2>/dev/null
tail -n 50 ~/.local/share/obsidian-rag/ollama-watchdog.log > "${DIR}/09-watchdog-log.txt" 2>&1

# 9) macOS jetsam / kernel events (últimos 5 min)
log show --last 5m --predicate 'eventMessage CONTAINS "ollama" OR eventMessage CONTAINS "memorystatus" OR eventMessage CONTAINS "killed" OR eventMessage CONTAINS "jetsam"' 2>/dev/null | head -50 > "${DIR}/10-kernel-events.txt"

# 10) Latencia probe — cuánto tarda /api/embed AHORA
{
  echo "=== probe embed bge-m3 (timeout 30s) ==="
  time curl -s -m 30 -o /dev/null -w 'http=%{http_code} t=%{time_total}\n' \
    -X POST http://127.0.0.1:11434/api/embed \
    -H "Content-Type: application/json" \
    -d '{"model":"bge-m3","input":"postmortem"}' 2>&1
  echo
  echo "=== probe generate qwen2.5:7b (timeout 30s, num_predict=1) ==="
  time curl -s -m 30 -o /dev/null -w 'http=%{http_code} t=%{time_total}\n' \
    -X POST http://127.0.0.1:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:7b","prompt":"ok","stream":false,"options":{"num_predict":1,"num_ctx":4096}}' 2>&1
} > "${DIR}/11-probes.txt" 2>&1

log "DONE — captured to ${DIR}"

# Auto-rotation: mantener solo últimos 20
POSTMORTEM_DIR="${HOME}/.local/share/obsidian-rag/draft-postmortems"
if [ -d "${POSTMORTEM_DIR}" ]; then
  ls -t "${POSTMORTEM_DIR}" | tail -n +21 | while read old; do
    rm -rf "${POSTMORTEM_DIR}/${old}"
  done
fi

echo "${DIR}"
