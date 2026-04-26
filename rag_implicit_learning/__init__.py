"""Implicit feedback learning — extrae signal del comportamiento del user
sin que tenga que tocar botones explícitamente.

Diseñado para cerrar el loop del aprendizaje del ranker sin depender de
inputs explícitos (👍/👎 + corrective_path en el chat). El primer
componente — `corrective_paths.py` — infiere `corrective_path` para
feedback negativo histórico revisando qué nota abrió el user después
de dar 👎.

Diseño:
- Lee de `rag_feedback` + `rag_behavior` (telemetry.db). NO escribe nuevos
  rows: actualiza `extra_json` de feedbacks existentes con
  `corrective_path` + `corrective_source = 'implicit_behavior_inference'`.
- Idempotente: skipea feedbacks que ya tienen `corrective_path` (manual o
  inferido). Re-correrlo NO duplica trabajo.
- Correlación por `session_id` + ventana temporal (default 60s) — si el
  user no abrió nada después del 👎, el feedback queda sin corrective.

El loop completo del sistema cerrado de auto-aprendizaje vive en la nota
`Sistema cerrado de aprendizaje automatizado.md` del vault. Este módulo
es solo el primer componente (Sprint 1, behavioral inference).

Llamadores:
- CLI: `rag feedback infer-implicit` (dry-run + apply).
- Daemon nightly: invocado pre-tune para enriquecer la señal antes del
  fine-tune del ranker en `com.fer.obsidian-rag-online-tune`.
"""

from rag_implicit_learning.corrective_paths import (
    infer_corrective_paths_from_behavior,
)

__all__ = ["infer_corrective_paths_from_behavior"]
