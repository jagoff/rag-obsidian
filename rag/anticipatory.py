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


def _phase2_apply_kind_weight(kind: str, score: float) -> float:
    """Phase 2.D wire-up: multiplica `score` por el weight per-kind
    (default 1.0). Silent-fail si el módulo no carga (tests aislados,
    schema corrupto): retorna `score` sin tocar."""
    try:
        from rag_anticipate.kind_weights import apply_kind_weight
        return apply_kind_weight(kind, score)
    except Exception:
        return score


def _phase2_kind_threshold(kind: str) -> float:
    """Phase 2.A wire-up: threshold base + delta per-kind del feedback
    tuning. Silent-fail al threshold base si el módulo no carga."""
    try:
        from rag_anticipate.feedback_tuning import (
            compute_kind_threshold_adjustment,
        )
        delta = compute_kind_threshold_adjustment(kind)
    except Exception:
        delta = 0.0
    return float(_ANTICIPATE_MIN_SCORE) + float(delta)


def anticipate_run_impl(
    *, dry_run: bool = False, explain: bool = False, force: bool = False,
    now: datetime | None = None,
) -> dict:
    """Una pasada del scheduler. Devuelve un dict con:
        - selected: dict del candidate elegido (o None)
        - sent: bool — si proactive_push lo envió de verdad
        - all: list[dict] de TODOS los candidates (para tests + --explain)
        - skip_reason: si no se envió, motivo (quiet hours, no viable, etc.)

    `force=True` bypassa dedup + daily_cap (debug/manual). `dry_run=True`
    evalúa + loguea pero NO empuja a WA.

    Phase 2 changes (2026-04-29):
      - Aplica `apply_kind_weight(kind, score)` antes de filtrar por
        threshold (Phase 2.D, weights configurables).
      - Aplica `compute_kind_threshold_adjustment(kind)` para subir/bajar
        el threshold según feedback acumulado del user (Phase 2.A).
      - Gate global de `is_in_quiet_hours(now)` antes de pushear; si
        retorna `(True, reason)` no se envía y se setea
        `skip_reason="quiet_hours: <reason>"` (Phase 2.B).
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

    # Phase 2.D: apply per-kind weight to score IN-PLACE (re-build
    # candidates with weighted score). The weight defaults to 1.0 when
    # no override is configured, so this is a no-op for users who
    # haven't tuned anything.
    weighted_candidates: list[AnticipatoryCandidate] = []
    for c in all_candidates:
        weighted_score = _phase2_apply_kind_weight(c.kind, c.score)
        if weighted_score == c.score:
            weighted_candidates.append(c)
        else:
            weighted_candidates.append(AnticipatoryCandidate(
                kind=c.kind,
                score=weighted_score,
                message=c.message,
                dedup_key=c.dedup_key,
                snooze_hours=c.snooze_hours,
                reason=f"{c.reason} | weight*={weighted_score / max(c.score, 1e-9):.2f}",
            ))
    all_candidates = weighted_candidates

    # Log all (selected=False, sent=False inicialmente).
    for c in all_candidates:
        try:
            _anticipate_log_candidate(c, selected=False, sent=False)
        except Exception as exc:
            _silent_log("anticipate_log_initial", exc)

    if not all_candidates:
        return {"selected": None, "sent": False, "all": []}

    # Phase 2.A: el threshold ya no es global; sube/baja per-kind según
    # la ratio de mute/positive del user. Default 0 si no hay feedback.
    viable = [
        c for c in all_candidates
        if c.score >= _phase2_kind_threshold(c.kind)
        and (force or not _rag._anticipate_dedup_seen(c.dedup_key))
    ]
    viable.sort(key=lambda c: c.score, reverse=True)

    # Bug 2026-04-30 fix: skipear kinds que están en snooze o silenciados
    # ANTES de seleccionar el top. Pre-fix, el scheduler elegía el top por
    # score (siempre el mismo, ej. anticipate-commitment 1.00), llamaba a
    # proactive_push, este chequeaba snooze y devolvía sent=False — nunca
    # se evaluaba viable[1], viable[2]. Resultado observado en producción:
    # 96 entries/día en rag_proactive_log con
    # `anticipate-commitment en snooze hasta 2026-05-03`, y los otros
    # kinds (calendar/echo/gap/dupes_pressure) NUNCA llegaban al user
    # aunque su snooze ya hubiera vencido. Con el filtro acá, el ranking
    # avanza al siguiente kind viable cuyo snooze esté libre.
    #
    # Filtramos solo snooze + silenced (state-aware), NO daily_cap ni
    # ambient_config: esos son globales y self-resolving / orthogonales
    # al ranking. `force=True` bypasea esto igual que bypasea dedup.
    if not force and viable:
        try:
            from rag.proactive import _proactive_load_state
            state = _proactive_load_state()
            silenced = set(state.get("silenced", []) or [])
            snooze_map = state.get("snooze", {}) or {}

            def _kind_available(kind: str) -> bool:
                if kind in silenced:
                    return False
                until_iso = snooze_map.get(kind)
                if not until_iso:
                    return True
                try:
                    until = datetime.fromisoformat(until_iso)
                except Exception:
                    return True
                return until <= now

            viable = [c for c in viable if _kind_available(c.kind)]
        except Exception as exc:
            _silent_log("anticipate_filter_snooze", exc)

    if not viable:
        return {
            "selected": None, "sent": False,
            "all": [_anticipate_candidate_to_dict(c) for c in all_candidates],
        }

    top = viable[0]
    sent = False
    skip_reason: str | None = None

    # Phase 2.B: quiet hours gate. Si estamos en horario "no molestar"
    # no se envía pero igual logueamos el candidate como `selected=True,
    # sent=False` para que `rag anticipate log` lo muestre.
    if not dry_run and not force:
        try:
            from rag_anticipate.quiet_hours import is_in_quiet_hours
            quiet, reason = is_in_quiet_hours(now)
        except Exception as exc:
            _silent_log("anticipate_quiet_hours", exc)
            quiet, reason = False, None
        if quiet:
            try:
                _anticipate_log_candidate(top, selected=True, sent=False)
            except Exception as exc:
                _silent_log("anticipate_log_selected", exc)
            return {
                "selected": _anticipate_candidate_to_dict(top),
                "sent": False,
                "skip_reason": f"quiet_hours: {reason}",
                "all": [_anticipate_candidate_to_dict(c) for c in all_candidates],
            }

    if not dry_run:
        try:
            # Pasamos `dedup_key` para que `proactive_push` sufije el body
            # con `_anticipate:<key>_` — el listener TS lo lee al detectar
            # un reply 👍/👎/🔇 y lo postea a /api/anticipate/feedback.
            # Sin esto, el feedback loop quedaría desconectado (no hay
            # otra forma de mapear "el user reaccionó a este push" →
            # "qué dedup_key era").
            sent, skip_reason = _rag.proactive_push(
                top.kind, top.message,
                snooze_hours=top.snooze_hours,
                dedup_key=top.dedup_key,
            )
        except Exception as exc:
            _silent_log("anticipate_proactive_push", exc)
            sent = False
            skip_reason = f"exception: {exc}"

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


# ── Phase 2 CLI subcommands ──────────────────────────────────────────────────
# `feedback stats`, `quiet-hours status`, `weights {set, list, reset}`.
# Cada uno delega al módulo correspondiente en `rag_anticipate.*` y
# renderiza vía Rich. Todos importan deferred — si los módulos no están
# disponibles (env no instalado, schema corrupto), el import-time del CLI
# no se rompe y solo el subcomando específico falla con un mensaje claro.


@anticipate.group("feedback")
def anticipate_feedback_group():
    """Feedback loop del agent (Phase 2.A) — engagement del user.

    Subcomandos:
      stats   Distribución mute/positive/negative por kind + delta del threshold.
    """


@anticipate_feedback_group.command("stats")
@click.option("--days", type=int, default=30,
                help="Ventana hacia atrás en días (default 30).")
@click.option("--plain", is_flag=True,
                help="Output sin tabla Rich (1 línea per kind, parseable).")
def anticipate_feedback_stats(days: int, plain: bool):
    """Distribución de feedback por kind con delta de threshold computed."""
    from rag_anticipate.feedback_tuning import all_kinds_feedback_summary
    summaries = all_kinds_feedback_summary()
    if not summaries:
        console.print(
            "[dim]sin feedback registrado todavía. "
            "El user empieza a reaccionar con 👍/👎/🔇 a los pushes "
            "y este comando va a mostrar la distribución.[/dim]"
        )
        return

    if plain:
        for s in summaries:
            console.print(
                f"{s['kind']}\tpos={s['positive']}\tneg={s['negative']}\t"
                f"mute={s['mute']}\tratio={s['mute_ratio']:.2f}\t"
                f"delta={s['delta']:+.2f}"
            )
        return

    from rich.table import Table
    t = Table(title=f"Feedback por kind (últimos {days} días)")
    t.add_column("kind")
    t.add_column("positive", justify="right")
    t.add_column("negative", justify="right")
    t.add_column("mute", justify="right")
    t.add_column("total", justify="right")
    t.add_column("mute_ratio", justify="right")
    t.add_column("Δ threshold", justify="right")
    for s in summaries:
        delta = s["delta"]
        delta_str = f"{delta:+.2f}"
        if delta > 0:
            delta_str = f"[red]{delta_str}[/red]"  # más estricto = menos volumen
        elif delta < 0:
            delta_str = f"[green]{delta_str}[/green]"  # más volumen
        t.add_row(
            s["kind"],
            str(s["positive"]),
            str(s["negative"]),
            str(s["mute"]),
            str(s["total"]),
            f"{s['mute_ratio']:.2f}",
            delta_str,
        )
    console.print(t)
    console.print(
        f"\n[dim]Threshold base: {_ANTICIPATE_MIN_SCORE:.2f}. "
        f"Δ se suma al base por kind. "
        f"Disable con RAG_ANTICIPATE_FEEDBACK_TUNING=0.[/dim]"
    )


@anticipate.group("quiet-hours")
def anticipate_quiet_hours_group():
    """Quiet hours del agent (Phase 2.B) — gate de no-molestar.

    Subcomandos:
      status   Muestra si AHORA estás en quiet hours y por qué.
    """


@anticipate_quiet_hours_group.command("status")
def anticipate_quiet_hours_status():
    """Imprime el estado actual del gate quiet hours."""
    from datetime import datetime as _dt
    from rag_anticipate.quiet_hours import is_in_quiet_hours
    now = _dt.now()
    quiet, reason = is_in_quiet_hours(now)
    if quiet:
        console.print(
            f"[yellow]🤫 quiet hours[/yellow] @ {now.strftime('%H:%M:%S')} — "
            f"reason: [bold]{reason}[/bold]"
        )
        if reason == "nighttime":
            band = os.environ.get("RAG_QUIET_HOURS_NIGHTTIME", "23-7")
            console.print(f"   ventana nocturna: {band} (override con RAG_QUIET_HOURS_NIGHTTIME)")
        elif reason == "in_meeting":
            console.print("   evento de calendario en curso (icalBuddy)")
        elif reason == "focus_code":
            console.print(
                "   proceso IDE recién abierto (<2 min). "
                "Disable con RAG_QUIET_HOURS_FOCUS_CODE=0"
            )
    else:
        console.print(
            f"[green]✓ open[/green] @ {now.strftime('%H:%M:%S')} — "
            f"el agent puede pushear ahora."
        )
    bypass = os.environ.get("RAG_ANTICIPATE_BYPASS_QUIET", "").strip().lower()
    if bypass in ("1", "true", "yes", "on"):
        console.print("[dim](bypass activo: RAG_ANTICIPATE_BYPASS_QUIET=1)[/dim]")


@anticipate.group("weights")
def anticipate_weights_group():
    """Multiplicadores per-kind del score base (Phase 2.D).

    Subcomandos:
      set     Set/update el weight de un kind.
      list    Lista todos los overrides configurados.
      reset   Borra el override de un kind (vuelve al default 1.0).
    """


@anticipate_weights_group.command("set")
@click.option("--kind", required=True,
                help="Ej. anticipate-calendar, anticipate-echo, anticipate-commitment")
@click.option("--weight", type=float, required=True,
                help="Float ∈ [0.0, 5.0]. 1.0 = no-op. <1 desprioriza, >1 prioriza.")
def anticipate_weights_set_cmd(kind: str, weight: float):
    """UPSERT del weight de `kind` en SQL (Phase 2.D)."""
    from rag_anticipate.kind_weights import set_kind_weight
    if not set_kind_weight(kind, weight):
        console.print(
            f"[red]error[/red]: weight {weight} fuera de rango [0.0, 5.0] "
            f"o write a SQL falló para kind={kind!r}."
        )
        raise SystemExit(2)
    console.print(f"[green]✓[/green] {kind} → weight {weight:.2f}")


@anticipate_weights_group.command("list")
def anticipate_weights_list_cmd():
    """Tabla de todos los overrides configurados."""
    from rag_anticipate.kind_weights import list_kind_weights
    rows = list_kind_weights()
    if not rows:
        console.print(
            "[dim]sin overrides — todos los kinds usan weight 1.0 (default).[/dim]\n"
            "Set con: [bold]rag anticipate weights set --kind <X> --weight <W>[/bold]"
        )
        return
    from rich.table import Table
    t = Table(title=f"kind weights overrides ({len(rows)})")
    t.add_column("kind")
    t.add_column("weight", justify="right")
    t.add_column("last_updated")
    for r in rows:
        t.add_row(r["kind"], f"{r['weight']:.2f}", r["last_updated"])
    console.print(t)


@anticipate_weights_group.command("reset")
@click.option("--kind", required=True,
                help="Kind cuyo weight querés borrar (vuelve a default 1.0).")
def anticipate_weights_reset_cmd(kind: str):
    """Borra el override del kind. Idempotente — si no había, igual OK."""
    from rag_anticipate.kind_weights import reset_kind_weight
    if not reset_kind_weight(kind):
        console.print(f"[red]error[/red]: write falló para kind={kind!r}")
        raise SystemExit(2)
    console.print(f"[green]✓[/green] {kind} → weight reset a 1.0 (default)")
