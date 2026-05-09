"""Drift watcher job — F1 shadow mode.

Reemplazo in-process del plist ``com.fer.obsidian-rag-drift-watcher`` que
hoy invoca ``scripts/drift_watcher.py`` cada 6h.

Estrategia SHADOW MODE:
- En F1, el plist viejo SIGUE corriendo (no se bootea). Este job en el
  supervisor también corre cada 6h. Comparamos resultados durante 24-48h
  via ``rag_supervisor_jobs`` vs ``drift_alerts.jsonl`` para validar que
  detecta los mismos eventos.
- En F2 (post-validación), bootout del plist viejo + remove factory.

Reuse del código existente: importamos ``evaluate()`` desde
``scripts/drift_watcher.py`` directo. NO duplicamos la lógica.

Cadencia preservada: cada 6h (matchea el plist viejo). El primer tick
post-supervisor-start es a las 6h — para forzar una corrida inmediata
usar ``rag supervisor trigger drift_watcher``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from rag.runtime.scheduler import interval

logger = logging.getLogger(__name__)


def _import_evaluate():
    """Import lazy de ``scripts/drift_watcher.py:evaluate``.

    El script está fuera del package ``rag/`` así que tenemos que
    agregar ``scripts/`` al sys.path. Idempotente.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import drift_watcher as _dw  # noqa: PLC0415
    return _dw


@interval(
    hours=6,
    label="drift_watcher",
    description="Alerta si singles_hit5/chains_hit5 cae entre runs de rag eval (F1 shadow).",
)
def drift_watcher_job() -> dict[str, Any]:
    """Wrapper del ``evaluate()`` del script.

    Retorna un dict ``{"alerts": int, "kinds": [...]}`` para que
    ``rag_supervisor_jobs.signals`` capture si algo se disparó esta
    corrida.
    """
    try:
        dw = _import_evaluate()
    except ImportError as exc:
        logger.warning("drift_watcher: cannot import evaluate(): %s", exc)
        return {"alerts": 0, "kinds": [], "error": str(exc)}

    try:
        # ``evaluate()`` retorna list[dict] de alertas nuevas (después de
        # filtrar dedupes), o [] si no hubo regresión.
        alerts = dw.evaluate()
    except Exception as exc:  # noqa: BLE001
        logger.exception("drift_watcher: evaluate() raised")
        return {"alerts": 0, "kinds": [], "error": str(exc)}

    if not isinstance(alerts, list):
        return {"alerts": 0, "kinds": []}

    kinds = sorted({a.get("kind", "?") for a in alerts if isinstance(a, dict)})
    return {
        "alerts": len(alerts),
        "kinds": kinds,
    }
