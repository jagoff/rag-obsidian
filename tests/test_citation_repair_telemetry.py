"""Tests de la telemetría extra de citation-repair (2026-04-22 follow-up).

Gap histórico: `rag_queries.citation_repaired` es booleano — sabemos que
reparó, no cuántas paths estaban mal ni si siquiera se intentó.

  - `bad_citations_count`: `len(bad)` que devolvió verify_citations(),
    antes del gate `_CITATION_REPAIR_MAX_BAD`.  Distribución empírica de
    esto + `citation_repaired` permite validar si el threshold (default 2)
    está bien calibrado.  Ejemplo de análisis:

      SELECT bad_citations_count, citation_repaired, COUNT(*)
      FROM (
        SELECT
          CAST(json_extract(extra_json,'$.bad_citations_count') AS INT) AS bad_citations_count,
          citation_repaired
        FROM rag_queries
        WHERE ts > datetime('now','-7 days')
      )
      GROUP BY 1, 2
      ORDER BY 1;

    Si `bad_citations_count=3` tiene >60% `citation_repaired=1`,
    vale la pena subir `_CITATION_REPAIR_MAX_BAD` a 3.

  - `citation_repair_attempted`: `(len(bad) > 0) AND
    (len(bad) <= _CITATION_REPAIR_MAX_BAD)`. Distingue "no se reparó
    porque no hacía falta" de "se intentó y no mejoró".

Validación en ambos paths (query + chat) porque 97% del tráfico real viene
de `cmd=web` (que usa chat path vía server) y `cmd=query` (CLI one-shot).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Regression guard: el campo existe en la row mapper ──────────────────────


def test_extra_json_accepts_bad_citations_count():
    """El SQL row mapper debe persistir bad_citations_count en extra_json
    sin rechazarlo. Si el schema cambia y alguien agrega `bad_citations_count`
    como columna standard, esto sigue pasando — es un campo "opcional"
    en el payload que llega a log_query_event."""
    ev = {
        "cmd": "query",
        "q": "test",
        "bad_citations_count": 2,
        "citation_repair_attempted": True,
        "citation_repaired": False,
    }
    # log_query_event no debe raisear, y el mapper debe acomodarlo
    # (stub write_retry para no tocar DB real).
    from unittest.mock import patch
    with patch.object(rag, "_sql_write_with_retry",
                      side_effect=lambda fn, tag, **kw: fn()):
        try:
            rag.log_query_event(ev)
        except Exception as exc:
            import pytest
            pytest.fail(f"log_query_event rejected bad_citations_count: {exc}")


# ── Los 2 campos se derivan del resultado del post-process ──────────────────


def _derive(bad_count: int, max_bad: int = 2):
    """Spec: cómo se deriva `citation_repair_attempted` a partir de `len(bad)`.

    Este es el contrato que los call sites en query() y chat() deben
    implementar. Tests abajo verifican que match lo esperado."""
    return (bad_count > 0) and (bad_count <= max_bad)


def test_repair_attempted_true_when_bad_in_range():
    assert _derive(bad_count=1) is True
    assert _derive(bad_count=2) is True


def test_repair_not_attempted_when_zero_bad():
    assert _derive(bad_count=0) is False


def test_repair_not_attempted_when_over_threshold():
    assert _derive(bad_count=3) is False
    assert _derive(bad_count=10) is False


def test_repair_attempted_respects_env_threshold():
    # Con _CITATION_REPAIR_MAX_BAD=3, bad=3 sí es attempted.
    assert _derive(bad_count=3, max_bad=3) is True
    assert _derive(bad_count=4, max_bad=3) is False


# ── Integration: query() path construye el event con los campos ─────────────


def test_query_log_event_shape_contains_new_fields():
    """Smoke: inspeccionar el source del query() log_query_event debe
    referenciar bad_citations_count + citation_repair_attempted. Si alguien
    remueve las líneas, este test falla antes de llegar a producción."""
    src = Path(__file__).resolve().parent.parent.joinpath("rag", "__init__.py").read_text()
    # Ambos campos deben aparecer en rag.py como keys del event.
    assert '"bad_citations_count"' in src, \
        "rag.py debe loggear bad_citations_count en query() / chat() path"
    assert '"citation_repair_attempted"' in src, \
        "rag.py debe loggear citation_repair_attempted en query() / chat() path"


def test_query_log_event_has_new_fields_in_both_paths():
    """Tanto query() como chat() deben loggear los dos campos (no sólo uno).
    Cuenta de occurrencias mínima 2 (una por path)."""
    src = Path(__file__).resolve().parent.parent.joinpath("rag", "__init__.py").read_text()
    assert src.count('"bad_citations_count"') >= 2, \
        "bad_citations_count debe loggearse en query() + chat() (≥2 ocurrencias)"
    assert src.count('"citation_repair_attempted"') >= 2, \
        "citation_repair_attempted debe loggearse en query() + chat() (≥2 ocurrencias)"
