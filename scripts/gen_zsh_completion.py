"""Genera un completion script de zsh (`_rag`) introspeccionando el árbol de
Click del CLI `rag`. El resultado es un archivo hand-written fast (no spawnea
Python en cada Tab — todos los subcomandos, flags, choices y descriptions
quedan inlined). Regenerar cuando cambia la superficie del CLI:

    .venv/bin/python scripts/gen_zsh_completion.py > completions/_rag

Install:

    cp completions/_rag ~/.oh-my-zsh/custom/completions/
    # O cualquier dir dentro de $fpath. Reiniciar zsh.

Este generador soporta grupos anidados (ambient, session, vault, etc.),
choices (emite `(a b c)`), tipos `Path`/`File` (emite `_files` / `_path_files -/`),
descripciones con quoting seguro, y un helper dinámico `_rag_vaults` que
consulta `rag vault list` para auto-completar nombres de vault.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import cli as root_cli  # noqa: E402


def _zsh_quote(s: str) -> str:
    """Escape una string para meterla dentro de single-quotes en zsh.

    zsh single-quotes no interpretan nada salvo `'` mismo — que se cierra,
    mete un `\\'` literal, y se re-abre.
    """
    return s.replace("'", "'\\''")


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0].strip()


def _option_choices(opt: click.Option) -> list[str] | None:
    t = opt.type
    if isinstance(t, click.Choice):
        return list(t.choices)
    return None


def _option_takes_path(opt: click.Option) -> str | None:
    """Devuelve el snippet zsh `:descr:action` para el valor de la opción,
    o None si la opción es un flag booleano sin valor."""
    if opt.is_flag or opt.count:
        return None
    descr = opt.metavar or (opt.name or "value").upper()
    t = opt.type
    if isinstance(t, click.Path):
        if t.file_okay and not t.dir_okay:
            return f":{descr}:_files"
        if t.dir_okay and not t.file_okay:
            return f":{descr}:_path_files -/"
        return f":{descr}:_files"
    if isinstance(t, click.File):
        return f":{descr}:_files"
    choices = _option_choices(opt)
    if choices:
        quoted = " ".join(choices)
        return f":{descr}:({quoted})"
    long_flags = {o for o in opt.opts if o.startswith("--")}
    if "--vault" in long_flags:
        return f":{descr}:_rag_vaults"
    if "--source" in long_flags:
        return ":SOURCE:(vault whatsapp gmail calendar reminders contacts bookmarks chrome claude drive github mail moze music reviews spotify youtube)"
    if "--session" in long_flags or opt.name in ("session", "session_id"):
        return ":SESSION:_rag_sessions"
    return f":{descr}:"


def _render_option(opt: click.Option) -> list[str]:
    """Emite uno o más fragmentos `_arguments` para la opción.

    Tres casos comunes:
      - flag boolean (--flag)             → '--flag[help]'
      - flag con alias corto              → '(-h --help)'{-h,--help}'[help]'
      - opción con valor                  → '--name=[help]:DESCR:action'
      - flag boolean con --no-flag pair   → dos entradas mutuamente excluyentes
    """
    help_text = _first_line(opt.help or "")
    help_esc = _zsh_quote(help_text)
    value_suffix = _option_takes_path(opt) or ""

    longs = [o for o in opt.opts if o.startswith("--")]
    shorts = [o for o in opt.opts if o.startswith("-") and not o.startswith("--")]
    secondary_longs = [o for o in (opt.secondary_opts or []) if o.startswith("--")]

    entries: list[str] = []

    if opt.is_flag and secondary_longs:
        group = " ".join(longs + secondary_longs + shorts)
        for name in longs + shorts:
            entries.append(f"'({group}){name}[{help_esc}]'")
        for name in secondary_longs:
            entries.append(f"'({group}){name}[{help_esc}]'")
        return entries

    all_flags = longs + shorts
    if len(all_flags) > 1:
        group = " ".join(all_flags)
        joined = "{" + ",".join(all_flags) + "}"
        if value_suffix:
            entries.append(f"'({group}){joined}=[{help_esc}]{value_suffix}'")
        else:
            entries.append(f"'({group}){joined}[{help_esc}]'")
        return entries

    name = all_flags[0]
    if value_suffix:
        entries.append(f"'{name}=[{help_esc}]{value_suffix}'")
    else:
        entries.append(f"'{name}[{help_esc}]'")
    return entries


def _render_arguments(cmd: click.Command) -> list[str]:
    """Emite las posicionales del comando como entradas `_arguments`.

    Para `click.Argument(nargs=-1)` usamos `*:…:…`; para posicionales simples
    usamos el índice numérico. Si el tipo es Path/File, delegamos a `_files`.
    """
    entries: list[str] = []
    positional_index = 1
    for param in cmd.params:
        if not isinstance(param, click.Argument):
            continue
        metavar = param.metavar or (param.name or "arg").upper()
        action = ":"
        t = param.type
        if isinstance(t, click.Path):
            if t.dir_okay and not t.file_okay:
                action = ":_path_files -/"
            else:
                action = ":_files"
        elif isinstance(t, click.File):
            action = ":_files"
        elif isinstance(t, click.Choice):
            action = ":(" + " ".join(t.choices) + ")"
        if param.nargs == -1:
            entries.append(f"'*:{metavar}{action}'")
        else:
            entries.append(f"'{positional_index}:{metavar}{action}'")
            positional_index += 1
    return entries


def _func_name(path: list[str]) -> str:
    slug = "_".join(p.replace("-", "_") for p in path)
    return f"_rag_cmd_{slug}" if slug else "_rag_cmd"


def _emit_command_func(cmd: click.Command, path: list[str], out: list[str]) -> None:
    fname = _func_name(path)
    out.append(f"{fname}() {{")
    arg_entries: list[str] = []
    for param in cmd.params:
        if isinstance(param, click.Option):
            arg_entries.extend(_render_option(param))
    if not isinstance(cmd, click.Group):
        arg_entries.extend(_render_arguments(cmd))
    if isinstance(cmd, click.Group):
        out.append('    local curcontext="$curcontext" state line')
        out.append("    typeset -A opt_args")
        out.append("    _arguments -C \\")
        for entry in arg_entries:
            out.append(f"        {entry} \\")
        out.append("        '1: :->subcmd' \\")
        out.append("        '*::arg:->args'")
        out.append("    case $state in")
        out.append("        subcmd)")
        out.append("            local -a subcommands")
        out.append("            subcommands=(")
        for sub_name, sub_cmd in sorted(cmd.commands.items()):
            if sub_cmd.hidden:
                continue
            descr = _zsh_quote(_first_line(sub_cmd.help or sub_cmd.short_help or ""))
            out.append(f"                '{sub_name}:{descr}'")
        out.append("            )")
        out.append('            _describe -t subcommands "rag ' + " ".join(path) + ' subcommand" subcommands')
        out.append("            ;;")
        out.append("        args)")
        out.append("            case $line[1] in")
        for sub_name, sub_cmd in sorted(cmd.commands.items()):
            if sub_cmd.hidden:
                continue
            sub_path = path + [sub_name]
            out.append(f"                {sub_name}) {_func_name(sub_path)} ;;")
        out.append("            esac")
        out.append("            ;;")
        out.append("    esac")
    else:
        if arg_entries:
            out.append("    _arguments \\")
            for i, entry in enumerate(arg_entries):
                suffix = " \\" if i < len(arg_entries) - 1 else ""
                out.append(f"        {entry}{suffix}")
        else:
            out.append("    :")
    out.append("}")
    out.append("")
    if isinstance(cmd, click.Group):
        for sub_name, sub_cmd in sorted(cmd.commands.items()):
            if sub_cmd.hidden:
                continue
            _emit_command_func(sub_cmd, path + [sub_name], out)


PREAMBLE = """#compdef rag
# Zsh completion for the `rag` CLI (obsidian-rag).
# Auto-generated by scripts/gen_zsh_completion.py — do not edit by hand.
# Regenerate after touching the Click command tree:
#   .venv/bin/python scripts/gen_zsh_completion.py > completions/_rag

_rag_vaults() {
    local -a vaults
    local line
    # `rag vault list` imprime filas tipo:  "* name  path  (active)"
    # extraemos la segunda columna ignorando la marca y el header.
    if (( $+commands[rag] )); then
        vaults=(${(f)"$(rag vault list 2>/dev/null | awk 'NR>1 && $1!~/^-/ {sub(/^\\*/,\"\",$1); print $1}')"})
    fi
    if (( ${#vaults} )); then
        compadd -a vaults
    fi
}

_rag_sessions() {
    local -a sessions
    if (( $+commands[rag] )); then
        sessions=(${(f)"$(rag session list 2>/dev/null | awk 'NR>1 {print $1}')"})
    fi
    if (( ${#sessions} )); then
        compadd -a sessions
    fi
}

"""

POSTAMBLE = """
_rag() {
    _rag_cmd "$@"
}

# When sourced directly (not autoloaded from fpath), register and dispatch.
if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    _rag "$@"
else
    compdef _rag rag
fi
"""


def main() -> None:
    out: list[str] = [PREAMBLE]
    _emit_command_func(root_cli, [], out)
    out.append(POSTAMBLE)
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
