#!/usr/bin/env bash
# whisper_tail — tail los logs filtrados a eventos del whisper learning loop.
#
# Útil cuando estás debuggeando o calibrando el sistema y querés ver SOLO
# las líneas relevantes del listener.log + whisper-server.log + ingest-whatsapp.log
# sin el ruido del resto del sistema.
#
# Uso:
#   ./scripts/whisper_tail.sh                # default: tail -f con filtro
#   ./scripts/whisper_tail.sh --no-follow    # imprimir últimas N y salir
#   ./scripts/whisper_tail.sh -n 50          # últimas 50 líneas matching
#
# Filtros aplicados:
# - [whisper]  → prefix de logs del listener (mode, duration, logprob).
# - [voice]    → transcripts del voice flow.
# - whisper_   → logs del whisper-server (vad, model load, etc.).
# - LLM correct → eventos del LLM auto-correct.
# - whatsapp:  → ingestion stats del ETL.

set -euo pipefail

LISTENER_LOG="${HOME}/.local/share/whatsapp-listener/listener.log"
WHISPER_SERVER_LOG="${HOME}/.local/share/whatsapp-listener/whisper-server.log"
INGEST_LOG="${HOME}/.local/share/obsidian-rag/ingest-whatsapp.log"
VOCAB_LOG="${HOME}/.local/share/obsidian-rag/whisper-vocab.log"

# Pattern de interés
PATTERN='\[whisper\]|\[voice\]|whisper_|LLM correct|whatsapp:|warm-up|ggml-|VAD|telemetry'

FOLLOW=1
NLINES=0  # 0 = sin límite

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-follow|-N)
      FOLLOW=0
      shift
      ;;
    -n)
      NLINES="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "uso: $0 [--no-follow] [-n N]" >&2
      exit 2
      ;;
  esac
done

# Verify que los logs existen.
LOGS=()
[[ -f "$LISTENER_LOG" ]] && LOGS+=("$LISTENER_LOG")
[[ -f "$WHISPER_SERVER_LOG" ]] && LOGS+=("$WHISPER_SERVER_LOG")
[[ -f "$INGEST_LOG" ]] && LOGS+=("$INGEST_LOG")
[[ -f "$VOCAB_LOG" ]] && LOGS+=("$VOCAB_LOG")

if [[ ${#LOGS[@]} -eq 0 ]]; then
  echo "no se encontraron logs del whisper learning loop. revisar:" >&2
  echo "  $LISTENER_LOG" >&2
  echo "  $WHISPER_SERVER_LOG" >&2
  echo "  $INGEST_LOG" >&2
  echo "  $VOCAB_LOG" >&2
  exit 1
fi

if [[ "$FOLLOW" == "1" ]]; then
  # tail -F para survive logrotate. Filtros via grep -E con --line-buffered
  # para que el output sea live (no buffered hasta el flush).
  tail -F "${LOGS[@]}" 2>/dev/null | grep -E --line-buffered --color=auto "$PATTERN"
else
  # Sin follow — imprimir últimas N líneas matching de cada file.
  N="${NLINES:-100}"
  if [[ "$N" -le 0 ]]; then N=100; fi
  for log in "${LOGS[@]}"; do
    echo "=== $(basename "$log") ==="
    grep -E --color=auto "$PATTERN" "$log" | tail -n "$N" || true
    echo ""
  done
fi
