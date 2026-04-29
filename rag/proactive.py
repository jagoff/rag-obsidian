"""Proactive nudges — extracted from `rag/__init__.py` (Phase 4a of monolith split, 2026-04-25).

Pattern: features proactivas (emergent themes, feedback patterns, anticipate
agent, reminder push, etc.) comparten un pipeline común que maneja:

- **Rate-limit global** — `PROACTIVE_DAILY_CAP=3` mensajes por día, compartido
  entre TODOS los kinds. Evita inundar al usuario con WA pings.
- **Silencio per-kind** — `rag silence <kind>` agrega a `state["silenced"]`,
  `_proactive_can_push` skipea el kind hasta que se quite.
- **Snooze per-kind** — al enviar, el caller puede pasar `snooze_hours=N` y
  ese kind se snoozea por N horas (evita re-pingear el mismo trigger).
- **Append-only log** — toda intent (sent + skipped) va a la tabla
  `rag_proactive_log` para analytics.

El objetivo es que cada feature (emergent / patterns / followup / calendar /
anticipate) solo declare `kind` + construya el mensaje; la infra se encarga
del resto.

## Surfaces (re-exportadas en `rag.<X>` via shim al final de `rag/__init__.py`)

Constantes:
- `PROACTIVE_STATE_PATH` — `~/.local/share/obsidian-rag/proactive.json`.
- `PROACTIVE_LOG_PATH` — `~/.local/share/obsidian-rag/proactive.jsonl`.
- `PROACTIVE_DAILY_CAP` — int (default 3).

Funciones:
- `_proactive_load_state()` — lee state file con reset diario automático.
- `_proactive_save_state(state)` — escribe atómico via tmp + rename.
- `_proactive_log(event)` — append a `rag_proactive_log` (silent-fail).
- `_proactive_can_push(kind)` — `(ok, reason)` decision; chequea silenced,
  snooze, daily_cap, ambient config.
- `proactive_push(kind, message, *, snooze_hours)` — entry point. Llama
  `_ambient_whatsapp_send` y aplica state updates.

## Why deferred imports

`_proactive_can_push` y `proactive_push` necesitan `_ambient_config` y
`_ambient_whatsapp_send`, que viven en `rag/__init__.py`. Tests hacen
`monkeypatch.setattr(rag, "_ambient_config", ...)` y
`monkeypatch.setattr(rag, "_ambient_whatsapp_send", ...)` — para que esos
patches propaguen, hacemos `from rag import X` adentro de los function
bodies. Igual con `_ragvec_state_conn`, `_sql_append_event`,
`_sql_write_with_retry`, `_map_proactive_row`.

## Tests-friendly: PROACTIVE_LOG_PATH es overridable

`test_rag_writers_sql.py` patches `rag.PROACTIVE_LOG_PATH`. La constante se
re-exporta tal cual en el shim, así que la asignación en el módulo y el
`monkeypatch.setattr(rag, "PROACTIVE_LOG_PATH", ...)` apuntan al mismo
objeto. Si en el futuro alguna función referencia `PROACTIVE_LOG_PATH`
directamente, va a leer el valor original; tendría que resolver via
`rag.PROACTIVE_LOG_PATH` para ver el patch — pero hoy ningún consumer lo
referencia internamente, solo el test lo lee tras patchearlo.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


# ── Constants ───────────────────────────────────────────────────────────────
PROACTIVE_STATE_PATH = Path.home() / ".local/share/obsidian-rag/proactive.json"
PROACTIVE_LOG_PATH = Path.home() / ".local/share/obsidian-rag/proactive.jsonl"
PROACTIVE_DAILY_CAP = 3


def _proactive_load_state() -> dict:
    """Carga {date, daily_count, silenced:[], snooze:{kind: iso_ts}}.

    Si el date guardado no es hoy, resetea daily_count.

    Resolves PROACTIVE_STATE_PATH via `rag.PROACTIVE_STATE_PATH` so tests can
    `monkeypatch.setattr(rag, "PROACTIVE_STATE_PATH", tmp_path / "p.json")`
    and have the override take effect here without rewriting state files
    in the user's real ~/.local/share/.
    """
    import rag as _rag
    default = {"date": datetime.now().strftime("%Y-%m-%d"),
               "daily_count": 0, "silenced": [], "snooze": {}}
    state_path = _rag.PROACTIVE_STATE_PATH
    if not state_path.is_file():
        return default
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    today = datetime.now().strftime("%Y-%m-%d")
    if data.get("date") != today:
        data["date"] = today
        data["daily_count"] = 0
    return {**default, **data}


def _proactive_save_state(state: dict) -> None:
    import rag as _rag
    state_path = _rag.PROACTIVE_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _proactive_log(event: dict) -> None:
    from rag import (
        _map_proactive_row,
        _ragvec_state_conn,
        _sql_append_event,
        _sql_write_with_retry,
    )
    full = {"ts": datetime.now().isoformat(timespec="seconds"), **event}

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_proactive_log",
                               _map_proactive_row(full))
    _sql_write_with_retry(_do, "proactive_sql_write_failed")


def _proactive_can_push(kind: str) -> tuple[bool, str]:
    """Devuelve (ok, reason). 'reason' explica por qué no se envía."""
    from rag import _ambient_config
    cfg = _ambient_config()
    if not cfg:
        return (False, "ambient WA no habilitado (/enable_ambient desde el bot)")
    state = _proactive_load_state()
    if kind in state.get("silenced", []):
        return (False, f"{kind} silenciado (rag silence off {kind})")
    snooze_ts = state.get("snooze", {}).get(kind)
    if snooze_ts:
        try:
            until = datetime.fromisoformat(snooze_ts)
            if datetime.now() < until:
                return (False, f"{kind} en snooze hasta {until.isoformat(timespec='minutes')}")
        except Exception:
            pass
    if state.get("daily_count", 0) >= PROACTIVE_DAILY_CAP:
        return (False, f"daily cap alcanzado ({PROACTIVE_DAILY_CAP})")
    return (True, "")


def proactive_push(
    kind: str, message: str, *,
    snooze_hours: int | None = None,
    dedup_key: str | None = None,
) -> tuple[bool, str | None]:
    """Push proactivo a WA con rate-limit + silencio + snooze compartidos.

    Si `snooze_hours` se pasa, tras enviar el kind entra en snooze por ese
    tiempo — evita repetir el mismo trigger (ej: emergent theme sobre 'X'
    ya se pingeó, no repetir hasta snooze_hours después).

    Si `dedup_key` se pasa, el body se sufija con
    `\\n\\n_anticipate:<dedup_key>_` (markdown italic, WA lo renderiza
    como cursiva pequeña). El listener TS lee este footer cuando el user
    responde 👍/👎/🔇 al push y lo postea a `/api/anticipate/feedback`
    con el `dedup_key` parseado — cierra el loop de feedback del
    Anticipatory Agent. Sin `dedup_key`, el body queda intacto (back-
    compat con `emergent` y `patterns` que no tienen dedup_key estable).

    Returns `(sent, reason)`:
      - sent=True, reason=None   → mensaje enviado al WA jid.
      - sent=False, reason=str   → razón EXACTA del skip. Una de:
          * "ambient WA no habilitado (/enable_ambient desde el bot)"
              (gate 1 — no existe ambient.json o enabled=false)
          * "{kind} silenciado (rag silence off {kind})"  (gate 2)
          * "{kind} en snooze hasta {iso_ts}"             (gate 3)
          * "daily cap alcanzado ({N})"                   (gate 4)
          * "WA bridge send failed"   (los 4 gates pasaron pero el HTTP
              POST al bridge falló — cluster down, bridge crasheado, etc.)

    Pre-2026-04-26: devolvía solo `bool`. Los callers imprimían un mensaje
    genérico "no pusheado (cap diario, silencio o snooze)" que ocultaba el
    motivo real (ej. ambient deshabilitado, gate 1, NO uno de los 3 que
    nombraba el mensaje). Cambio motivado por debugging del loop de
    anticipatory roto: 72 candidates con sent=0 que parecían rate-limit
    pero eran ambient.json missing. La razón exacta ahorra horas.
    """
    from rag import _ambient_config, _ambient_whatsapp_send
    ok, reason = _proactive_can_push(kind)
    if not ok:
        _proactive_log({"kind": kind, "sent": False, "reason": reason})
        return (False, reason)
    cfg = _ambient_config()
    if not cfg:
        # Defensive: _proactive_can_push ya chequea esto, pero por si la
        # config se borra entre el chequeo y el send.
        return (False, "ambient WA no habilitado (/enable_ambient desde el bot)")
    # Sufijar el footer ANTES de mandar al bridge. Markdown italic
    # discreto pero visible para audit; el listener parsea
    # `_anticipate:<key>_` con un regex simple en el reply.
    body = message
    if dedup_key:
        body = f"{message}\n\n_anticipate:{dedup_key}_"
    sent = _ambient_whatsapp_send(cfg["jid"], body)
    state = _proactive_load_state()
    if sent:
        state["daily_count"] = state.get("daily_count", 0) + 1
        if snooze_hours:
            state.setdefault("snooze", {})[kind] = (
                datetime.now() + timedelta(hours=snooze_hours)
            ).isoformat(timespec="seconds")
        _proactive_save_state(state)
    _proactive_log({"kind": kind, "sent": sent, "message_preview": body[:120]})
    return (sent, None if sent else "WA bridge send failed")
