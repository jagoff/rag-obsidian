#!/usr/bin/env bash
# Watcher del quick-tunnel de cloudflared: detecta cambios de URL en el log
# y reacciona automáticamente sin que el user tenga que grepear a mano.
#
# Qué hace cuando ve una URL nueva:
#   1. La escribe a ~/.local/share/obsidian-rag/cloudflared-url.txt
#      (el alias `rag-url` del ~/.zshrc lee ese archivo).
#   2. La copia al clipboard del Mac (`pbcopy`) — lista para pegar en
#      Safari del iPhone o en un chat, sin fricción.
#   3. Manda una notificación macOS via osascript con la URL visible.
#   4. Escribe una línea timestampeada a su propio log para observability.
#
# Corre como servicio launchd (com.fer.obsidian-rag-cloudflare-tunnel-watcher).
# El gate "URL nueva" usa un state file — si cloudflared reinicia pero
# obtiene la MISMA URL (raro, pero puede pasar), no molesta al user.
#
# No modifica el PWA del iPhone — eso requiere Safari manual. Lo que sí
# hace es dejar la URL list-to-paste al instante para re-abrir el PWA.
set -euo pipefail

LOG_FILE="$HOME/.local/share/obsidian-rag/cloudflared.error.log"
URL_FILE="$HOME/.local/share/obsidian-rag/cloudflared-url.txt"
WATCHER_LOG="$HOME/.local/share/obsidian-rag/cloudflared-watcher.log"

log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') $*" >> "$WATCHER_LOG"
}

notify() {
    local url="$1"
    # macOS notification. `osascript` es la vía nativa sin dep externa.
    # `display notification` requiere permisos Notifications la primera vez
    # — el user los acepta una vez y queda persistido.
    # Escapar comillas simples: usamos dobles en el AppleScript.
    osascript -e "display notification \"$url\" with title \"RAG Tunnel\" subtitle \"Nueva URL (copiada al portapapeles)\" sound name \"Tink\"" 2>/dev/null || true
    # Copy to clipboard — sin newline final para que al pegar en Safari
    # no haya un salto fantasma. `printf %s` en vez de `echo` justamente
    # por eso.
    printf %s "$url" | pbcopy 2>/dev/null || true
}

# On startup: si el log ya tiene una URL (caso típico cuando el watcher
# arranca DESPUÉS de cloudflared), inicializá el state file con la
# última URL vista. Evita spamear una notificación al arrancar el
# watcher cuando la URL no cambió realmente.
current_url_in_log=""
if [ -f "$LOG_FILE" ]; then
    current_url_in_log=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_FILE" | tail -1 || true)
fi

# Last seen URL state. Persiste entre restarts del watcher.
last_url=""
if [ -f "$URL_FILE" ]; then
    last_url=$(cat "$URL_FILE" 2>/dev/null || true)
fi

# Si hay URL en el log pero el state está desactualizado, updeatela + notifica.
# Típico: user recién reinició cloudflared y el watcher arrancó después.
if [ -n "$current_url_in_log" ] && [ "$current_url_in_log" != "$last_url" ]; then
    printf %s "$current_url_in_log" > "$URL_FILE"
    notify "$current_url_in_log"
    log "Initial sync — URL: $current_url_in_log"
    last_url="$current_url_in_log"
fi

log "Watcher started, tailing $LOG_FILE"

# `tail -F` sigue el archivo aunque se rote (launchd puede cortar el log).
# `-n 0` arranca desde el final — el bootstrap de arriba ya cubrió el
# estado inicial; acá sólo escuchamos líneas nuevas.
tail -n 0 -F "$LOG_FILE" 2>/dev/null | while read -r line; do
    url=$(printf %s "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' || true)
    [ -z "$url" ] && continue
    [ "$url" = "$last_url" ] && continue
    # URL nueva. Actualizar state + notificar.
    printf %s "$url" > "$URL_FILE"
    notify "$url"
    log "URL changed → $url"
    last_url="$url"
done
