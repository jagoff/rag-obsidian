#!/usr/bin/env python3
"""Sanitiza el `rag_response_cache` aplicando `replace_iberian_leaks` a
cada entrada existente. Idempotente: re-correr no cambia nada después
de la primera pasada (el filter es idempotente sobre texto en es).

Por qué: hay entradas en el cache con leaks pt (`primeira`, `tua`,
`falam`, etc.) generadas antes del fix de los system prompts +
auto-aplicación del filter en `render_response()`. Aunque
`render_response()` ahora filtra al renderear, hay paths que devuelven
el cached `response` directo (ej. `mcp_server.py.rag_query`, integración
WhatsApp via `serve`) sin pasar por el filter. Sanitizar el cache una
vez los limpia para esos paths también.

Uso:
    .venv/bin/python scripts/sanitize_response_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import _ragvec_state_conn  # noqa: E402
from rag.iberian_leak_filter import replace_iberian_leaks  # noqa: E402


def main():
    n_total = 0
    n_changed = 0
    chars_before = 0
    chars_after = 0

    with _ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT id, response FROM rag_response_cache ORDER BY ts ASC"
        ).fetchall()
        n_total = len(rows)

        for row_id, response in rows:
            if not response:
                continue
            chars_before += len(response)
            cleaned = replace_iberian_leaks(response)
            chars_after += len(cleaned)
            if cleaned != response:
                conn.execute(
                    "UPDATE rag_response_cache SET response = ? WHERE id = ?",
                    (cleaned, row_id),
                )
                n_changed += 1

        conn.commit()

    print(f"Sanitización del rag_response_cache:")
    print(f"  Entries totales: {n_total}")
    print(f"  Entries modificadas: {n_changed}")
    print(f"  Chars before: {chars_before}")
    print(f"  Chars after:  {chars_after}")
    print(f"  Delta: {chars_after - chars_before:+d} chars")
    if n_total > 0:
        print(f"  % cambios: {n_changed/n_total*100:.1f}%")


if __name__ == "__main__":
    main()
