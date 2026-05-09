"""Launchd plist factories — sub-paquete post-split 2026-05-09.

Layout por dominio (target post-Fase 3):

  persistent.py    → _watch_plist + _web_plist (KeepAlive=true, hot path)
  briefs.py        → _digest_plist + _morning_plist + _today_plist
  proactive.py     → emergent / patterns / archive / distill / anticipate /
                      active-learning-nudge / brief-auto-tune
  learning.py      → auto-harvest / online-tune / calibration /
                      implicit-feedback / routing-rules / whisper-vocab
  maintenance.py   → maintenance / vault-cleanup / consolidate
  ingest.py        → ingest-whatsapp + ingest-cross-source
  wa.py            → wa-fast (worker WA time-sensitive)
  poll.py          → mood-poll + spotify-poll (uv-tool venv polls)
  control.py       → wake-up + daemon-watchdog + wake-hook
  _render.py       → _render_plist + helpers + constantes + _repo_root()
  _spec.py         → _services_spec + _services_spec_manual + gates +
                      _DEPRECATED_LABELS + _INSTALL_GATES

Estado actual (post-commit-1, bootstrap):
  - persistent.py + briefs.py ya viven en su sub-módulo final.
  - proactive / learning / maintenance / ingest / wa / poll / control
    siguen TEMPORALMENTE en `_legacy.py` — se splittean en commits 2-4.

Back-compat: `from rag.plists import *` re-exporta los 38 nombres del
`__all__` original (el viejo `rag/plists.py` pre-split). `rag._watch_plist`,
`rag._services_spec`, `rag._LAUNCH_AGENTS_DIR` siguen funcionando porque
`rag/__init__.py` ya hace `from rag.plists import *  # noqa: F401, F403`.
"""
from __future__ import annotations

# Helpers + constantes
from rag.plists._render import (  # noqa: F401
    _DEFAULT_PATH,
    _GOOGLE_TOKEN_PATH,
    _LAUNCH_AGENTS_DIR,
    _PLIST_HEADER,
    _RAG_LOG_DIR,
    _logs,
    _rag_binary,
    _render_plist,
    _repo_root,
)

# Factories ya migradas a sub-módulos finales
from rag.plists.briefs import (  # noqa: F401
    _digest_plist,
    _morning_plist,
    _today_plist,
)
from rag.plists.learning import (  # noqa: F401
    _auto_harvest_plist,
    _calibration_plist,
    _implicit_feedback_plist,
    _online_tune_plist,
    _routing_rules_plist,
    _whisper_vocab_plist,
)
from rag.plists.ingest import (  # noqa: F401
    _ingest_cross_source_plist,
    _ingest_whatsapp_plist,
)
from rag.plists.maintenance import (  # noqa: F401
    _consolidate_plist,
    _maintenance_plist,
    _vault_cleanup_plist,
)
from rag.plists.persistent import _watch_plist, _web_plist  # noqa: F401
from rag.plists.proactive import (  # noqa: F401
    _active_learning_nudge_plist,
    _anticipate_plist,
    _archive_plist,
    _brief_auto_tune_plist,
    _distill_plist,
    _emergent_plist,
    _patterns_plist,
)
from rag.plists.wa import _wa_fast_plist  # noqa: F401

# Factories TEMPORALMENTE en _legacy.py (se splittean en commit 4)
from rag.plists._legacy import (  # noqa: F401
    _daemon_watchdog_plist,
    _mood_poll_plist,
    _spotify_poll_plist,
    _wake_hook_plist,
    _wake_up_plist,
)

# Spec orquestación + gates + deprecated
from rag.plists._spec import (  # noqa: F401
    _DEPRECATED_LABELS,
    _INSTALL_GATES,
    _calendar_creds_exist,
    _google_token_exists,
    _mood_daemon_opted_in,
    _services_spec,
    _services_spec_manual,
)

# `__all__` literal — idéntico al `rag/plists.py` pre-split (38 nombres).
# NO auto-generar con `list(globals())` (incluiría basura como `Path`).
__all__ = [
    "_LAUNCH_AGENTS_DIR", "_RAG_LOG_DIR", "_GOOGLE_TOKEN_PATH",
    "_rag_binary", "_watch_plist", "_web_plist",
    "_digest_plist", "_morning_plist", "_today_plist", "_wa_fast_plist",
    "_emergent_plist", "_patterns_plist", "_archive_plist",
    "_distill_plist",
    "_consolidate_plist", "_vault_cleanup_plist", "_anticipate_plist",
    "_maintenance_plist", "_calibration_plist", "_auto_harvest_plist",
    "_active_learning_nudge_plist", "_online_tune_plist",
    "_implicit_feedback_plist", "_ingest_whatsapp_plist",
    "_ingest_cross_source_plist",
    "_mood_poll_plist", "_routing_rules_plist",
    "_whisper_vocab_plist", "_wake_up_plist",
    "_brief_auto_tune_plist", "_daemon_watchdog_plist", "_wake_hook_plist",
    "_services_spec", "_google_token_exists", "_calendar_creds_exist",
    "_mood_daemon_opted_in", "_DEPRECATED_LABELS", "_INSTALL_GATES",
    "_services_spec_manual",
]
