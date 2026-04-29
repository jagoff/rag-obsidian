"""Regression test — todo filtro definido en web/server.py debe estar
cableado al menos una vez en la pipeline de streaming o de cache replay.

Motivación (wave-8 2026-04-28): `_strip_foreign_scripts` quedó "definido
pero olvidado" durante semanas. Tenía regex completo + docstring +
comentario explicando el bug que arreglaba pero ningún call site la
invocaba, así que el CJK leak en respuestas de weather siguió en
producción hasta wave-8 cuando lo descubrí por casualidad.

Este test atrapa el patrón: cualquier `class _XxxFilter` o función
`_strip_*`, `_redact_*`, `_normalize_*` definida en `web/server.py`
TIENE QUE tener al menos un call site fuera de su propia definición.
Si no, falla con instrucciones claras al dev.

Si una función está intencionalmente definida pero todavía no cableada
(work in progress legítimo), agregala a `_INTENTIONALLY_UNUSED` con un
comentario explicando por qué + tracking issue.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = REPO_ROOT / "web" / "server.py"

# Pattern para clases _XxxFilter y funciones _strip_*/_redact_*/_normalize_*.
# Excluye los que tienen "_block" / "_section" / "_intent" / "_handler" en
# el nombre — esos son helpers de render/routing, no filtros del pipeline.
_FILTER_DEF_RE = re.compile(
    r"^(?:def|class)\s+"
    r"(_(?:strip|redact|normalize)_(?!.*(?:block|section|intent|handler)\b)\w+|"
    r"_\w+Filter|_\w+Stripper)\b",
    re.MULTILINE,
)

# Si agregás un filtro acá, dejá un comentario diciendo por qué todavía
# no está cableado (e.g. "shadowed por un PR en review", "feature gated
# por env var", etc.). Cuando lo cableés, removelo.
_INTENTIONALLY_UNUSED: dict[str, str] = {
    # name → reason
}


def _all_filter_defs(source: str) -> list[str]:
    """Devuelve nombres de filtros definidos en el source."""
    return _FILTER_DEF_RE.findall(source)


def _is_called(name: str, source: str, definition_line: int) -> bool:
    """True si `name` aparece como call site (no la línea de definición)."""
    # Buscamos `name(` en cualquier línea EXCEPTO la de definición.
    # También aceptamos `name.flush()`, `name.feed()` (instance methods de
    # clases _XxxFilter / _XxxStripper).
    pattern = re.compile(
        r"\b" + re.escape(name) + r"\s*(?:\(|\.\w+\s*\()"
    )
    for i, line in enumerate(source.splitlines(), start=1):
        if i == definition_line:
            continue
        if pattern.search(line):
            return True
    return False


def _definition_line(name: str, source: str) -> int:
    pattern = re.compile(
        r"^(?:def|class)\s+" + re.escape(name) + r"\b",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return -1
    return source.count("\n", 0, m.start()) + 1


@pytest.fixture(scope="module")
def server_source() -> str:
    return SERVER_PY.read_text(encoding="utf-8")


def test_all_filters_in_server_are_wired(server_source: str) -> None:
    """Cada `_XxxFilter` / `_*Stripper` / `_strip_*` / `_redact_*` /
    `_normalize_*` definido en web/server.py debe tener al menos un call
    site OTHER que su línea de definición.

    Si este test falla:
    1. Buscá la definición — `grep -n "def <name>\\|class <name>" web/server.py`.
    2. Identificá el pipeline donde DEBERÍA correr (streaming `_emit`,
       cache replay, post-process del tool output, etc.).
    3. Cableá el call site. Para filters streaming: dentro de `gen()`,
       buscá la chain `raw_tool_filter → stripper → iberian → pii_filter`
       y agregá el tuyo en el orden correcto.
    4. Para cache replay: buscá `_redact_pii(_sem_text)` en línea ~9887
       y aplicá el mismo patrón.
    5. Si el filter está intencionalmente WIP, agregalo a
       `_INTENTIONALLY_UNUSED` en este archivo con razón + tracking issue.
    """
    filters = _all_filter_defs(server_source)
    assert filters, "Sanity: deberíamos detectar al menos un filtro"
    unused: list[tuple[str, int]] = []
    for name in filters:
        if name in _INTENTIONALLY_UNUSED:
            continue
        line = _definition_line(name, server_source)
        if line < 0:
            continue
        if not _is_called(name, server_source, line):
            unused.append((name, line))

    if unused:
        msg_lines = [
            "Filtros definidos en web/server.py SIN call site detectado:",
            "",
        ]
        for name, line in unused:
            msg_lines.append(f"  - {name} (línea {line})")
        msg_lines += [
            "",
            "Causa habitual: 'lo definí pero olvidé wirear'. Wave-8 (2026-04-28)",
            "encontró `_strip_foreign_scripts` en este estado durante semanas.",
            "",
            "Cómo arreglar:",
            "  1. Identificá la pipeline correcta (streaming `_emit`, cache replay, etc.).",
            "  2. Agregá el call site.",
            "  3. Si es intencionalmente WIP, agregalo a `_INTENTIONALLY_UNUSED`",
            "     en `tests/test_filter_wiring.py` con razón + tracking issue.",
        ]
        pytest.fail("\n".join(msg_lines))
