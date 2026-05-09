"""WhatsApp tasks state — high-water mark + dedup ring.

Estado del extractor `rag wa-tasks` persistido en
``~/.local/share/obsidian-rag/wa_tasks_state.json``:

- ``last_run_ts``: ISO8601 del último run exitoso. Próximo run fetchea
  estrictamente después.
- ``processed_ids``: ring de message_ids recientes (cap 2000). Dedup
  cheap entre ventanas que solapen.

Why deferred imports (``import rag as _rag``):
``WA_TASKS_STATE_PATH`` vive en `rag/__init__.py` (re-export desde
`_constants.py`). Resolver via `rag` en runtime permite que tests con
`monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", ...)` re-direcccionen
el path a un tmp.
"""

from __future__ import annotations

import json


def _wa_tasks_load_state() -> dict:
    """Returns `{last_run_ts: iso|null, processed_ids: [id, ...]}`.

    `processed_ids` is a ring of recent message ids (cap 2000) — cheap dedup
    across overlapping windows. `last_run_ts` is the high-water mark; next
    run fetches strictly after it.
    """
    # Deferred lookup so tests `monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", ...)`
    # are honored — the patch lives on `rag.__init__`, not on this module.
    import rag as _rag
    state_path = _rag.WA_TASKS_STATE_PATH
    if not state_path.is_file():
        return {"last_run_ts": None, "processed_ids": []}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_ts": None, "processed_ids": []}
    if not isinstance(data, dict):
        return {"last_run_ts": None, "processed_ids": []}
    data.setdefault("last_run_ts", None)
    data.setdefault("processed_ids", [])
    if not isinstance(data["processed_ids"], list):
        data["processed_ids"] = []
    return data


def _wa_tasks_save_state(state: dict) -> None:
    import rag as _rag
    state_path = _rag.WA_TASKS_STATE_PATH
    ids = state.get("processed_ids") or []
    if len(ids) > 2000:
        state["processed_ids"] = ids[-2000:]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )


__all__ = [
    "_wa_tasks_load_state",
    "_wa_tasks_save_state",
]
