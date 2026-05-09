"""Job modules para el supervisor.

Cada submódulo aquí registra jobs vía decorators de
``rag.runtime.scheduler``. Importarlo es suficiente para que el
scheduler los conozca — el supervisor entrypoint hace
``importlib.iter_modules(rag.runtime.jobs.__path__)`` al startup para
descubrirlos automáticamente.

Layout proyectado (F1-F3):

- ``drift_watcher.py`` — F1, en shadow mode primero
- ``nightly.py`` — F2: auto_harvest, whisper_vocab, implicit_feedback,
  online_tune, maintenance, calibrate
- ``proactive.py`` — F3: emergent, patterns, archive, distill,
  brief_auto_tune, anticipate
- ``ingest.py`` — F3: ingest_whatsapp, ingest_cross_source
- ``briefs.py`` — F3: morning, today, digest
- ``poll.py`` — F3: mood (luego F4 on-demand), spotify (luego F4 event),
  wa_tasks (luego F4 SQL hook)
- ``learning.py`` — F3: routing_rules, active_learning_*
- ``housekeeping.py`` — F3: consolidate, vault_cleanup, wake_up
"""
from __future__ import annotations

__all__: list[str] = []
