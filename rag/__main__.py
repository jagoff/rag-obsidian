"""Permite `python -m rag <subcommand>` (usado por el Makefile dev loop).

El entry point canónico via `uv tool install` sigue siendo el binario
`rag` (definido en `pyproject.toml::[project.scripts]` como `rag:cli`).
Este módulo solo wrappea el mismo `cli` para soportar la invocación
`python -m rag` que requiere el target `make eval`/`make tune`/etc.
"""

from rag import cli

if __name__ == "__main__":
    cli()
