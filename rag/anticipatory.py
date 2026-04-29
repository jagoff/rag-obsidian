"""Anticipatory agent — extracted from `rag/__init__.py` (Phase 3 of monolith split, 2026-04-25).

Game-changer 2026-04-24: el RAG deja de ser puramente "pull" (yo pregunto, él
responde) y empieza a "hablarte primero" cuando tiene algo timely que decir.

## Diseño

- Un scheduler (`anticipate_run_impl`) corre N scorers (signals).
- Cada signal devuelve `list[AnticipatoryCandidate]` — kind + score [0,1] +
  mensaje + dedup_key + snooze_hours.
- Se loguean TODOS los candidates a `rag_anticipate_candidates` (analytics).
- Filtra por threshold (`RAG_ANTICIPATE_MIN_SCORE`, default 0.35) y dedup
  (no repetir el mismo dedup_key dentro de 24h vía SQL lookup).
- Pickea top-1 por score y empuja vía `proactive_push()` existente
  (que aporta silence + snooze + daily_cap=3 — el agente comparte el cap
  con `emergent` y `patterns`).
- launchd `com.fer.obsidian-rag-anticipate.plist` corre cada 10min.

## Señales activas

1. **anticipate-calendar** — eventos próximos 15-90min con contexto en vault.
2. **anticipate-echo** — nota de hoy que resuena con una vieja (>60d, cosine ≥0.70).
3. **anticipate-commitment** — open loops stale ≥7d (delegación a followup).

Señales NUEVAS van en el package externo `rag_anticipate.signals.<kind>` y
se auto-registran vía el decorator `@register_signal` — leemos la lista
`rag_anticipate.SIGNALS` al import-time y la concatenamos con los core
signals. Silent-fail si el package no carga (ej. tests sin el folder):
core sigue funcionando.

## Kill-switches

- `rag silence anticipate-calendar` (per-kind via _proactive_load_state).
- `RAG_ANTICIPATE_DISABLED=1` (global, `anticipate_run_impl` early-return).
- `rm ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist + bootout`.

## Surfaces (re-exportadas en `rag.<X>` via shim al final de `rag/__init__.py`)

Constantes (env-tunable):
- `_ANTICIPATE_MIN_SCORE`, `_ANTICIPATE_DEDUP_WINDOW_HOURS`,
  `_ANTICIPATE_CALENDAR_MIN_MIN`, `_ANTICIPATE_CALENDAR_MAX_MIN`,
  `_ANTICIPATE_ECHO_MIN_AGE_DAYS`, `_ANTICIPATE_ECHO_MIN_COSINE`,
  `_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS`.

Tipos:
- `AnticipatoryCandidate` — frozen dataclass (kind, score, message,
  dedup_key, snooze_hours, reason).

Helpers (callables desde Python):
- `_anticipate_dedup_seen(dedup_key, window_hours)` — SQL lookup en
  `rag_anticipate_candidates` para skipping.
- `_anticipate_log_candidate(c, *, selected, sent)` — append a la tabla
  `rag_anticipate_candidates` (silent-fail).
- `_anticipate_signal_calendar/echo/commitment(now)` — los 3 signals core.
- `anticipate_run_impl(*, dry_run, explain, force, now)` — orchestrator.
- `_anticipate_candidate_to_dict(c)` — serializer para tests + --explain.
- `_anticipate_fetch_log(*, limit, only_sent)` — last-N rows.

Click commands (registrados en `cli` global vía decorators):
- `anticipate` — `rag anticipate`, group con default → `anticipate run`.
- `anticipate run [--dry-run] [--explain] [--force]` — main entrypoint.
- `anticipate log [-n N] [--only-sent]` — historial.
- `anticipate explain` — todas las señales del momento, sin pushear ni dedup.

## Why deferred imports

`rag/anticipatory.py` se carga al final de `rag/__init__.py` (después de que
`cli`, `console`, `_ragvec_state_conn`, `proactive_push`, `_fetch_calendar_today`
[via integrations re-export], `retrieve`, `get_db`, `_resolve_vault_path`,
`is_excluded`, `find_followup_loops`, `_silent_log`, `_sql_append_event`,
`_sql_write_with_retry`, `_map_anticipate_row` están todos definidos).

Los 2 imports module-level (`from rag import cli, console`) funcionan porque
en ese punto el parent package está fully loaded.

Los helpers internos del parent se importan **dentro de cada función body**
con `from rag import X`. Esto es crítico para que tests que hacen
`monkeypatch.setattr(rag, "_fetch_calendar_today", ...)` o
`monkeypatch.setattr(rag, "proactive_push", ...)` propaguen al call site —
`from rag import X` re-evalúa `rag.X` cada vez que la función corre.

## Tabla del log (schema en `rag_anticipate_candidates`)

Definida en `rag/__init__.py` (junto al resto del schema SQL); este módulo
solo lee/escribe via `_ragvec_state_conn` + `_sql_append_event`. Ver
`tests/test_anticipate_agent.py` para fixtures que setean una DB vacía con
ese schema antes de invocar las funciones de aquí.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import click

from rag import cli, console

if TYPE_CHECKING:
    pass


# ── Env-tunable thresholds ──────────────────────────────────────────────────
_ANTICIPATE_MIN_SCORE = float(os.environ.get("RAG_ANTICIPATE_MIN_SCORE", "0.35"))
_ANTICIPATE_DEDUP_WINDOW_HOURS = int(
    os.environ.get("RAG_ANTICIPATE_DEDUP_WINDOW_HOURS", "24")
)
_ANTICIPATE_CALENDAR_MIN_MIN = int(
    os.environ.get("RAG_ANTICIPATE_CALENDAR_MIN_MIN", "15")
)
_ANTICIPATE_CALENDAR_MAX_MIN = int(
    os.environ.get("RAG_ANTICIPATE_CALENDAR_MAX_MIN", "90")
)
_ANTICIPATE_ECHO_MIN_AGE_DAYS = int(
    os.environ.get("RAG_ANTICIPATE_ECHO_MIN_AGE_DAYS", "60")
)
_ANTICIPATE_ECHO_MIN_COSINE = float(
    os.environ.get("RAG_ANTICIPATE_ECHO_MIN_COSINE", "0.70")
)
_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS = int(
    os.environ.get("RAG_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS", "7")
)


@dataclass(frozen=True)
class AnticipatoryCandidate:
    """Un candidate de push proactivo. Inmutable; producido por una signal,
    consumido por el orchestrator."""
    kind: str            # ej. "anticipate-calendar", "anticipate-echo", ...
    score: float         # [0, 1]; threshold default 0.35
    message: str         # WA body completo, listo para `proactive_push()`
    dedup_key: str       # estable cross-runs (ej. event_uid, source_path)
    snooze_hours: int    # default por kind: cal=2, echo=72, commit=168
    reason: str          # debug — por qué este candidate (mostrado en --explain)


def _anticipate_dedup_seen(dedup_key: str,
                            window_hours: int = _ANTICIPATE_DEDUP_WINDOW_HOURS) -> bool:
    """True si este dedup_key fue ENVIADO con éxito en la ventana."""
    from rag import _ragvec_state_conn, _silent_log
    cutoff_dt = datetime.now() - timedelta(hours=window_hours)
    cutoff = cutoff_dt.isoformat(timespec="seconds")
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM rag_anticipate_candidates"
                " WHERE dedup_key = ? AND ts >= ? AND sent = 1 LIMIT 1",
                (dedup_key, cutoff),
            ).fetchone()
        return row is not None
    except Exception as exc:
        _silent_log("anticipate_dedup_lookup", exc)
        return False


def _anticipate_log_candidate(c: "AnticipatoryCandidate", *,
                                selected: bool, sent: bool) -> None:
    """Append candidate row a rag_anticipate_candidates. Silent-fail."""
    from rag import (
        _map_anticipate_row,
        _ragvec_state_conn,
        _sql_append_event,
        _sql_write_with_retry,
    )
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": c.kind,
        "score": float(c.score),
        "dedup_key": c.dedup_key,
        "selected": int(bool(selected)),
        "sent": int(bool(sent)),
        "reason": c.reason,
        "message_preview": (c.message or "")[:120],
    }

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_anticipate_candidates",
                                _map_anticipate_row(row))
    _sql_write_with_retry(_do, "anticipate_log_failed")


# ── Calendar proximity signal ────────────────────────────────────────────────

def _anticipate_parse_event_start(ev: dict, now: datetime) -> datetime | None:
    """Parsea el `start` HH:MM de un evento de hoy a un datetime concreto.

    `_fetch_calendar_today()` devuelve `start` como `"09:30"` (24h) o
    `"9:30 AM"`. Si el formato no matchea, devolvemos None.
    """
    raw = (ev.get("start") or "").strip()
    if not raw:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])?", raw)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ampm = (m.group(3) or "").upper()
    if ampm == "PM" and hh != 12:
        hh += 12
    elif ampm == "AM" and hh == 12:
        hh = 0
    if hh > 23 or mm > 59:
        return None
    return now.replace(hour=hh, minute=mm, second=0, microsecond=0)


def _anticipate_format_calendar_brief(
    title: str, delta_min: int, top_meta: dict, top_score: float
) -> str:
    note = (top_meta.get("note") or top_meta.get("file") or "?").rsplit("/", 1)[-1]
    file_rel = top_meta.get("file") or ""
    snippet = (top_meta.get("preview") or top_meta.get("text") or "").strip()[:140]
    snippet_line = f"  > {snippet}" if snippet else ""
    return (
        f"📅 En {delta_min} min: {title}\n"
        f"\n"
        f"Contexto en el vault:\n"
        f"  · [[{note}]] ({file_rel}, score {int(top_score * 100)}%)\n"
        f"{snippet_line}".rstrip()
    )


def _anticipate_signal_calendar(now: datetime) -> list["AnticipatoryCandidate"]:
    """Eventos de hoy que arrancan en [MIN, MAX] minutos con contexto en vault."""
    from rag import _fetch_calendar_today, _silent_log, get_db, retrieve
    try:
        events = _fetch_calendar_today(max_events=20)
    except Exception as exc:
        _silent_log("anticipate_calendar_fetch", exc)
        return []
    out: list[AnticipatoryCandidate] = []
    for ev in events:
        title = (ev.get("title") or "").strip()
        if not title:
            continue
        start = _anticipate_parse_event_start(ev, now)
        if start is None:
            continue
        delta_min = (start - now).total_seconds() / 60.0
        if delta_min < _ANTICIPATE_CALENDAR_MIN_MIN:
            continue
        if delta_min > _ANTICIPATE_CALENDAR_MAX_MIN:
            continue
        # Retrieve contexto del vault para el evento.
        # `caller="anticipate-calendar"` (2026-04-28): etiqueta este
        # impression con el caller real para que no se mezcle con user
        # signal en `rag_behavior` (loop estructural cerrado en el commit
        # `fd97829`). El ranker training filtra esto out.
        try:
            col = get_db()
            result = retrieve(
                col, title, 3, folder=None, tag=None,
                precise=False, multi_query=False, auto_filter=False,
                caller="anticipate-calendar",
            )
        except Exception as exc:
            _silent_log("anticipate_calendar_retrieve", exc)
            continue
        scores = result.get("scores") or []
        metas = result.get("metas") or []
        if not scores or scores[0] < 0.25:
            continue
        score = max(0.0, min(1.0, 1.0 - (delta_min / float(_ANTICIPATE_CALENDAR_MAX_MIN))))
        msg = _anticipate_format_calendar_brief(title, int(delta_min), metas[0], scores[0])
        # Sin uid disponible desde icalBuddy → dedup por title + start clock-time.
        dedup_key = f"cal:{title[:60]}:{start.strftime('%Y-%m-%dT%H:%M')}"
        out.append(AnticipatoryCandidate(
            kind="anticipate-calendar",
            score=score,
            message=msg,
            dedup_key=dedup_key,
            snooze_hours=2,
            reason=(
                f"event in {int(delta_min)}min,"
                f" top_score={scores[0]:.2f}"
            ),
        ))
    return out


# ── Temporal echo signal ─────────────────────────────────────────────────────

def _anticipate_find_recent_notes(vault: Path, *, within_hours: int,
                                    min_chars: int, limit: int) -> list[Path]:
    """Notas modificadas dentro de las últimas `within_hours` horas, sorted
    desc por mtime. Excluye carpetas según `is_excluded` y notas chicas."""
    from rag import is_excluded
    cutoff = time.time() - within_hours * 3600
    candidates: list[tuple[float, Path]] = []
    if not vault.is_dir():
        return []
    for p in vault.rglob("*.md"):
        try:
            rel = str(p.relative_to(vault))
        except ValueError:
            continue
        try:
            if is_excluded(rel):
                continue
        except Exception:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff:
            continue
        if st.st_size < min_chars:
            continue
        candidates.append((st.st_mtime, p))
    candidates.sort(reverse=True)
    return [p for _, p in candidates[:limit]]


def _anticipate_note_age_days(file_rel: str, vault: Path) -> float:
    try:
        return (time.time() - (vault / file_rel).stat().st_mtime) / 86400.0
    except Exception:
        return 0.0


def _anticipate_note_first_chars(p: Path, n: int) -> str:
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]
    return text[:n].strip()


def _anticipate_format_echo_brief(today_note: Path, old_meta: dict,
                                    old_score: float, age_days: float) -> str:
    today_title = today_note.stem
    old_title = (old_meta.get("note") or old_meta.get("file") or "?").rsplit("/", 1)[-1]
    old_file = old_meta.get("file") or ""
    months_ago = max(1, int(age_days / 30))
    return (
        f"🔮 Lo que escribiste hoy resuena con algo de hace ~{months_ago} meses:\n"
        f"\n"
        f"Hoy: [[{today_title}]]\n"
        f"Entonces: [[{old_title}]] ({old_file})\n"
        f"\n"
        f"Cosine {int(old_score * 100)}%. ¿Mergear, archivar o solo releer?"
    )


def _anticipate_signal_echo(now: datetime) -> list["AnticipatoryCandidate"]:
    """Última nota tocada hoy (≥6h ventana) que resuena con una nota vieja
    (>60d). Empuja si cosine ≥ 0.70."""
    # `_anticipate_note_age_days` is resolved via `rag` (not local) so tests
    # can monkeypatch.setattr(rag, "_anticipate_note_age_days", ...) and
    # have the override take effect inside this signal.
    import rag as _rag
    from rag import _resolve_vault_path, _silent_log, get_db, retrieve
    try:
        vault = _resolve_vault_path()
    except Exception as exc:
        _silent_log("anticipate_echo_vault", exc)
        return []
    recent = _anticipate_find_recent_notes(
        vault, within_hours=6, min_chars=500, limit=3,
    )
    if not recent:
        return []
    out: list[AnticipatoryCandidate] = []
    for note in recent:
        snippet = _anticipate_note_first_chars(note, n=500)
        if not snippet or len(snippet) < 50:
            continue
        # `caller="anticipate-echo"` (2026-04-28): etiqueta el impression
        # con el caller real para que el ranker training pueda filtrarlo.
        # Ver doc en rag.retrieve() y commit `fd97829`.
        try:
            col = get_db()
            result = retrieve(
                col, snippet, 5, folder=None, tag=None,
                precise=False, multi_query=False, auto_filter=False,
                caller="anticipate-echo",
            )
        except Exception as exc:
            _silent_log("anticipate_echo_retrieve", exc)
            continue
        try:
            today_rel = str(note.relative_to(vault))
        except ValueError:
            today_rel = note.name
        candidates: list[tuple[dict, float, float]] = []
        for m, s in zip(result.get("metas") or [], result.get("scores") or []):
            file_rel = m.get("file") or ""
            if not file_rel or file_rel == today_rel:
                continue
            age = _rag._anticipate_note_age_days(file_rel, vault)
            if age < _ANTICIPATE_ECHO_MIN_AGE_DAYS:
                continue
            candidates.append((m, float(s), age))
        if not candidates:
            continue
        old_meta, old_score, age = candidates[0]
        if old_score < _ANTICIPATE_ECHO_MIN_COSINE:
            continue
        msg = _anticipate_format_echo_brief(note, old_meta, old_score, age)
        dedup_key = f"echo:{today_rel}:{old_meta.get('file','?')}"
        out.append(AnticipatoryCandidate(
            kind="anticipate-echo",
            score=float(old_score),
            message=msg,
            dedup_key=dedup_key,
            snooze_hours=72,
            reason=f"cosine={old_score:.2f}, age={int(age)}d",
        ))
    return out


# ── Stale commitment signal ──────────────────────────────────────────────────

def _anticipate_format_commitment_brief(loop: dict) -> str:
    quote = (loop.get("loop_text") or "").strip()[:200]
    source = (loop.get("source_note") or "?").rsplit("/", 1)[-1]
    age = int(loop.get("age_days") or 0)
    return (
        f"⏰ Hace {age} días dijiste que ibas a hacer algo y no veo señal:\n"
        f"\n"
        f"  > {quote}\n"
        f"\n"
        f"Fuente: [[{source}]]\n"
        f"\n"
        f"¿Avance? Si ya está hecho, `rag fix` con la nota resolutoria."
    )


def _anticipate_signal_commitment(now: datetime) -> list["AnticipatoryCandidate"]:
    """Open loops stale ≥7d. Reusa `find_followup_loops()`. Push 1 por run."""
    from rag import _resolve_vault_path, _silent_log, find_followup_loops, get_db
    try:
        col = get_db()
        vault = _resolve_vault_path()
    except Exception as exc:
        _silent_log("anticipate_commitment_db", exc)
        return []
    try:
        loops = find_followup_loops(
            col, vault, days=60, stale_days=_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS,
            now=now,
        )
    except Exception as exc:
        _silent_log("anticipate_commitment_scan", exc)
        return []
    stale = [
        l for l in loops
        if l.get("status") == "stale"
        and (l.get("age_days") or 0) >= _ANTICIPATE_COMMITMENT_MIN_AGE_DAYS
    ]
    if not stale:
        return []
    stale.sort(key=lambda l: l.get("age_days") or 0, reverse=True)
    top = stale[0]
    age = float(top.get("age_days") or 0)
    score = max(0.0, min(1.0, age / 30.0))
    h = hashlib.sha256(
        ((top.get("loop_text") or "") + "|" + (top.get("source_note") or ""))
        .encode("utf-8")
    ).hexdigest()[:12]
    msg = _anticipate_format_commitment_brief(top)
    return [AnticipatoryCandidate(
        kind="anticipate-commitment",
        score=score,
        message=msg,
        dedup_key=f"commit:{h}",
        snooze_hours=168,
        reason=f"age={int(age)}d, kind={top.get('kind')}",
    )]


# ── Orchestrator ─────────────────────────────────────────────────────────────

# Tuple de (kind_label_corto, signal_fn). El orden NO importa para el outcome
# porque después se filtra por score; pero el log respeta este orden.
#
# Las 3 señales originales (calendar / echo / commitment) viven acá por
# histórico. Señales NUEVAS van en `rag_anticipate/signals/<kind>.py` y se
# auto-registran vía el decorator `@register_signal` — leemos la lista
# `rag_anticipate.SIGNALS` al final y la concatenamos. Silent-fail si el
# package no carga (ej. tests sin el folder): core sigue funcionando.
_ANTICIPATE_CORE_SIGNALS: tuple[tuple[str, "Callable[[datetime], list[AnticipatoryCandidate]]"], ...] = (
    ("calendar", _anticipate_signal_calendar),
    ("echo", _anticipate_signal_echo),
    ("commitment", _anticipate_signal_commitment),
)

try:
    import rag_anticipate as _rag_anticipate_pkg
    _ANTICIPATE_EXTRA_SIGNALS: tuple = tuple(_rag_anticipate_pkg.SIGNALS)
except Exception as _exc:
    _ANTICIPATE_EXTRA_SIGNALS = ()
    try:
        from rag import _silent_log as _sl
        _sl("anticipate_extra_signals_load", _exc)
    except Exception:
        pass

_ANTICIPATE_SIGNALS: tuple[tuple[str, "Callable[[datetime], list[AnticipatoryCandidate]]"], ...] = (
    *_ANTICIPATE_CORE_SIGNALS,
    *_ANTICIPATE_EXTRA_SIGNALS,
)


def anticipate_run_impl(
    *, dry_run: bool = False, explain: bool = False, force: bool = False,
    now: datetime | None = None,
) -> dict:
    """Una pasada del scheduler. Devuelve un dict con:
        - selected: dict del candidate elegido (o None)
        - sent: bool — si proactive_push lo envió de verdad
        - all: list[dict] de TODOS los candidates (para tests + --explain)

    `force=True` bypassa dedup + daily_cap (debug/manual). `dry_run=True`
    evalúa + loguea pero NO empuja a WA.
    """
    # Resolve via `rag` so tests can monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS", ...)
    # and (rag, "_anticipate_dedup_seen", ...) and (rag, "proactive_push", ...).
    import rag as _rag
    from rag import _silent_log
    if os.environ.get("RAG_ANTICIPATE_DISABLED", "").strip() in ("1", "true", "yes"):
        return {"selected": None, "sent": False, "all": [], "disabled": True}
    now = now or datetime.now()
    all_candidates: list[AnticipatoryCandidate] = []
    for label, fn in _rag._ANTICIPATE_SIGNALS:
        try:
            all_candidates.extend(fn(now))
        except Exception as exc:
            _silent_log(f"anticipate_signal_{label}_failed", exc)

    # Log all (selected=False, sent=False inicialmente).
    for c in all_candidates:
        try:
            _anticipate_log_candidate(c, selected=False, sent=False)
        except Exception as exc:
            _silent_log("anticipate_log_initial", exc)

    if not all_candidates:
        return {"selected": None, "sent": False, "all": []}

    viable = [
        c for c in all_candidates
        if c.score >= _ANTICIPATE_MIN_SCORE
        and (force or not _rag._anticipate_dedup_seen(c.dedup_key))
    ]
    viable.sort(key=lambda c: c.score, reverse=True)

    if not viable:
        return {
            "selected": None, "sent": False,
            "all": [_anticipate_candidate_to_dict(c) for c in all_candidates],
        }

    # Iterate through viable candidates (already sorted by score DESC) and
    # return the first one that proactive_push actually delivers.  Before this
    # fix, the orchestrator picked top-1 unconditionally; if that kind was
    # snoozed, proactive_push returned (False, reason) and the whole run was
    # a no-op even when lower-ranked kinds had no snooze.  The new loop falls
    # through to the next candidate transparently.
    top: AnticipatoryCandidate | None = None
    sent = False
    skip_reason: str | None = None

    for candidate in viable:
        if dry_run:
            # In dry-run mode we never call proactive_push, so just pick the
            # highest-score candidate (first in the already-sorted list).
            top = candidate
            break

        try:
            # Pasamos `dedup_key` para que `proactive_push` sufije el body
            # con `_anticipate:<key>_` — el listener TS lo lee al detectar
            # un reply 👍/👎/🔇 y lo postea a /api/anticipate/feedback.
            # Sin esto, el feedback loop quedaría desconectado (no hay
            # otra forma de mapear "el user reaccionó a este push" →
            # "qué dedup_key era").
            sent, skip_reason = _rag.proactive_push(
                candidate.kind, candidate.message,
                snooze_hours=candidate.snooze_hours,
                dedup_key=candidate.dedup_key,
            )
        except Exception as exc:
            _silent_log("anticipate_proactive_push", exc)
            sent = False
            skip_reason = f"exception: {exc}"

        if sent:
            # Successfully pushed — record which candidate won and stop.
            top = candidate
            break
        # Not sent (snoozed / daily-cap / silenced) — try the next candidate.

    if top is None:
        # All viable candidates were snoozed / exhausted (or viable was empty
        # in a non-dry-run pass — the dry_run branch always sets top).
        import logging
        logging.getLogger(__name__).info(
            "anticipate: all %d viable kinds snoozed — nothing pushed", len(viable)
        )
        return {
            "selected": None, "sent": False,
            "skip_reason": "all kinds snoozed",
            "all": [_anticipate_candidate_to_dict(c) for c in all_candidates],
        }

    try:
        _anticipate_log_candidate(top, selected=True, sent=sent)
    except Exception as exc:
        _silent_log("anticipate_log_selected", exc)

    return {
        "selected": _anticipate_candidate_to_dict(top),
        "sent": sent,
        "skip_reason": skip_reason,
        "all": [_anticipate_candidate_to_dict(c) for c in all_candidates],
    }


def _anticipate_candidate_to_dict(c: AnticipatoryCandidate) -> dict:
    return {
        "kind": c.kind,
        "score": c.score,
        "message": c.message,
        "dedup_key": c.dedup_key,
        "snooze_hours": c.snooze_hours,
        "reason": c.reason,
    }


def _anticipate_fetch_log(*, limit: int = 20, only_sent: bool = False) -> list[dict]:
    """Lee últimas N rows de rag_anticipate_candidates."""
    from rag import _ragvec_state_conn, _silent_log
    where = "WHERE sent = 1" if only_sent else ""
    sql = (
        "SELECT ts, kind, score, dedup_key, selected, sent, reason,"
        " message_preview FROM rag_anticipate_candidates "
        f"{where} ORDER BY id DESC LIMIT ?"
    )
    try:
        with _ragvec_state_conn() as conn:
            cursor = conn.execute(sql, (int(limit),))
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
    except Exception as exc:
        _silent_log("anticipate_fetch_log", exc)
        return []
    return [dict(zip(cols, r)) for r in rows]


# ── CLI ──────────────────────────────────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def anticipate(ctx):
    """Anticipatory agent — el vault te habla sin que preguntes.

    Sin subcomando = `rag anticipate run` (default action). El daemon
    `com.fer.obsidian-rag-anticipate` corre esto cada 10 min via launchd.

    Subcomandos:
      run        Evalúa señales y empuja top-1 a WhatsApp.
      log        Últimos N candidates (sent + skipped).
      explain    Muestra TODAS las señales del momento sin pushear.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(anticipate_run, dry_run=False, explain=False, force=False)


@anticipate.command("run")
@click.option("--dry-run", is_flag=True,
                help="Evalúa + loguea pero NO pushea a WA")
@click.option("--explain", is_flag=True,
                help="Imprime tabla de TODOS los candidates con scores")
@click.option("--force", is_flag=True,
                help="Bypassea dedup_key + daily_cap (debug)")
def anticipate_run(dry_run: bool, explain: bool, force: bool):
    """Una pasada del scheduler — evalúa señales y empuja top-1."""
    result = anticipate_run_impl(dry_run=dry_run, explain=explain, force=force)
    if result.get("disabled"):
        console.print("[yellow]anticipate disabled (RAG_ANTICIPATE_DISABLED=1)[/yellow]")
        return
    all_c = result.get("all") or []
    if explain or dry_run:
        if not all_c:
            console.print("[dim]ninguna señal activa ahora[/dim]")
        else:
            from rich.table import Table
            t = Table(title=f"Anticipate — {len(all_c)} candidate(s)")
            t.add_column("kind"); t.add_column("score", justify="right")
            t.add_column("dedup_key"); t.add_column("reason")
            for c in sorted(all_c, key=lambda c: c["score"], reverse=True):
                t.add_row(c["kind"], f"{c['score']:.2f}",
                            c["dedup_key"][:40], (c["reason"] or "")[:50])
            console.print(t)
    sel = result.get("selected")
    if sel:
        console.print(f"\n[bold]Selected:[/bold] {sel['kind']} (score {sel['score']:.2f})")
        if dry_run:
            console.print("[dim]dry-run — no se envió[/dim]")
        elif result.get("sent"):
            console.print("[green]✓ pusheado a WA[/green]")
        else:
            reason = result.get("skip_reason") or "razón desconocida"
            console.print(f"[yellow]no pusheado: {reason}[/yellow]")
    elif not all_c:
        console.print("[dim]ninguna señal activa[/dim]")
    else:
        console.print(
            f"[dim]ningún candidate pasó el threshold "
            f"({_ANTICIPATE_MIN_SCORE:.2f}) o todos están dedup'd[/dim]"
        )


@anticipate.command("log")
@click.option("-n", "--limit", type=int, default=20)
@click.option("--only-sent", is_flag=True,
                help="Solo los que efectivamente se enviaron a WA")
def anticipate_log_cmd(limit: int, only_sent: bool):
    """Últimos N candidates de rag_anticipate_candidates."""
    rows = _anticipate_fetch_log(limit=limit, only_sent=only_sent)
    if not rows:
        console.print("[dim]sin registros todavía[/dim]")
        return
    from rich.table import Table
    t = Table(title=f"anticipate log ({len(rows)} rows)")
    t.add_column("ts"); t.add_column("kind"); t.add_column("score", justify="right")
    t.add_column("sel"); t.add_column("sent"); t.add_column("reason")
    for r in rows:
        ts_short = (r.get("ts") or "")[11:19]
        t.add_row(
            ts_short, r.get("kind") or "",
            f"{r.get('score', 0):.2f}",
            "✓" if r.get("selected") else "",
            "✓" if r.get("sent") else "",
            (r.get("reason") or "")[:55],
        )
    console.print(t)


@anticipate.command("explain")
def anticipate_explain_cmd():
    """Evalúa señales y muestra todas con su score, sin pushear ni filtrar dedup."""
    result = anticipate_run_impl(dry_run=True, explain=True, force=True)
    if result.get("disabled"):
        console.print("[yellow]anticipate disabled[/yellow]")
        return
    all_c = result.get("all") or []
    if not all_c:
        console.print("[dim]ninguna señal activa ahora[/dim]")
        return
    from rich.table import Table
    t = Table(title=f"All candidates ({len(all_c)})")
    t.add_column("kind"); t.add_column("score", justify="right")
    t.add_column("dedup_key"); t.add_column("snooze_h", justify="right")
    t.add_column("reason")
    for c in sorted(all_c, key=lambda c: c["score"], reverse=True):
        t.add_row(c["kind"], f"{c['score']:.2f}",
                    c["dedup_key"][:40], str(c["snooze_hours"]),
                    (c["reason"] or "")[:50])
    console.print(t)
    console.print(
        f"\n[dim]Threshold actual: {_ANTICIPATE_MIN_SCORE:.2f}. "
        f"Override con RAG_ANTICIPATE_MIN_SCORE.[/dim]"
    )
