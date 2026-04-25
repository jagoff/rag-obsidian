#!/bin/bash
# .devin/hooks/post-edit-syntax-check.sh
#
# PostToolUse hook que valida que rag.py y web/server.py siguen importando
# después de cada edit/write. Catch instantáneo de syntax errors + import
# errors antes de que el agent siga editando otra cosa basado en un estado
# roto.
#
# Triggered desde .devin/config.json:
#   PostToolUse → matcher: "edit|write" → command: este script.
#
# Stdin format (PostToolUse contract):
#   { "tool_name": "edit", "tool_input": { "file_path": "...", ... },
#     "tool_response": { "success": bool, "output": "...", "error": null } }
#
# Behavior:
#   - Si el file_path es rag.py o web/server.py: corre `python -c "import X"`.
#   - Si import falla: imprime el error a stderr (visible al agent) pero
#     exit 0 para no bloquear (el agent decide si seguir editando o corregir).
#   - Si el file_path es cualquier otra cosa: exit 0 silencioso.
#
# Costo: ~0.5-2s en cold cache, <0.5s warm. Negligible vs el ahorro de
# detectar syntax errors temprano.
#
# Cómo desactivar: borrar la entrada PostToolUse del .devin/config.json.

set -u

input=$(cat)

# Validar que tenemos jq disponible (instalado por default en macOS via brew)
if ! command -v jq >/dev/null 2>&1; then
    # Sin jq no podemos parsear el stdin — exit silencioso para no romper.
    exit 0
fi

file=$(echo "$input" | jq -r '.tool_input.file_path // empty')
tool_success=$(echo "$input" | jq -r '.tool_response.success // true')

# Si el tool falló, no corremos check (no hay edit que validar).
if [ "$tool_success" != "true" ]; then
    exit 0
fi

# Resolver path absoluto si es relativo (el agent puede pasar relative paths).
if [[ "$file" != /* ]] && [ -n "$file" ]; then
    file="$(pwd)/$file"
fi

# Filtrar: solo nos importan rag.py y web/server.py.
case "$file" in
    */rag.py)
        module="rag"
        ;;
    */web/server.py)
        module="web.server"
        ;;
    *)
        exit 0
        ;;
esac

# Cambiar al repo root para que `import` resuelva contra .venv/sys.path.
repo_root="/Users/fer/repositories/obsidian-rag"
cd "$repo_root" 2>/dev/null || exit 0

# Verificar que .venv/bin/python existe — si no, skip silencioso.
if [ ! -x "$repo_root/.venv/bin/python" ]; then
    exit 0
fi

# Import check con timeout duro de 10s (defensivo contra import-time side effects).
# `timeout` no viene por default en macOS — usamos `gtimeout` (brew coreutils) si
# está, sino corremos sin timeout (el module-level code de rag.py es rápido).
if command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd="gtimeout 10"
elif command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout 10"
else
    timeout_cmd=""
fi

output=$($timeout_cmd "$repo_root/.venv/bin/python" -c "import $module" 2>&1)
exit_code=$?

if [ $exit_code -eq 124 ]; then
    # Timeout — el módulo está haciendo I/O lento o loop en import-time.
    echo "⚠️  post-edit-syntax-check: import de '$module' tardó >10s (timeout)." >&2
    echo "   Probablemente hay I/O o trabajo en import-time. Ver $file." >&2
    exit 0
fi

if [ $exit_code -ne 0 ]; then
    echo "" >&2
    echo "❌ post-edit-syntax-check: import de '$module' falló después de editar $file" >&2
    echo "─────────────────────────────────────────────────────────────────────" >&2
    echo "$output" | head -40 >&2
    echo "─────────────────────────────────────────────────────────────────────" >&2
    echo "Probable causa: syntax error o import error introducido por el edit." >&2
    # exit 0: no bloqueamos el flow, solo informamos. El agent ve el stderr
    # y puede corregir antes de seguir.
    exit 0
fi

# Import OK — silencioso (no spameamos en el happy path).
exit 0
