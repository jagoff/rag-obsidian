"""Dashboard analytics del Anticipatory Agent.

Render de métricas operacionales sobre la tabla `rag_anticipate_candidates`
(append-only log que escribe `_anticipate_log_candidate` en `rag.py`):

- `fetch_metrics(days)` — agregados en ventana: total evaluated/selected/sent,
  send_rate, selection_rate, breakdown por `kind` con avg_score.
- `render_dashboard(days)` — render texto plano (no rich) listo para CLI/log.
- `top_reasons_skipped(days, limit)` — agrupa los `reason` de los candidates
  que NO se enviaron (sent=0). Útil para tunear thresholds: ver qué razones
  más comunes hay para descartar candidates.
- `signal_health(days)` — health check per-signal:
    - `silent`  — el signal NUNCA emitió (0 rows en toda la tabla).
    - `stale`   — emitió alguna vez pero hace >= `days` días.
    - `noisy`   — >= 10 emits en ventana con avg_score < 0.3 (sub-threshold).
    - `healthy` — caso normal.

Diseño:
- Silent-fail end-to-end: si la tabla `rag_anticipate_candidates` no existe
  todavía (caso de un DB fresh sin ningún candidate evaluado), o el DB está
  inaccesible, las funciones devuelven shapes vacíos consistentes (no
  raise). Es un dashboard, no debe tumbar el llamador.
- Lecturas read-only sobre `rag._ragvec_state_conn()` — la misma conn que
  usa el resto del agent. No carga sqlite-vec extension, no escribe.
- "kinds known" en `signal_health` = unión de:
    1. los 3 signals originales del MVP (`anticipate-calendar`,
       `anticipate-echo`, `anticipate-commitment`),
    2. los signals registrados en `rag_anticipate.signals.base.SIGNALS`
       (`anticipate-{name}` para cada uno),
    3. los `kind` distintos que aparecen en la tabla.
  Así un signal recién agregado aparece como `silent` aunque todavía no
  haya emitido nunca, y un signal viejo eliminado del registry pero con
  rows históricas aparece como `stale` (no se "pierde" la observabilidad).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import rag


# ── Shape constants ──────────────────────────────────────────────────────────

# Los 3 signals originales del MVP que viven hardcoded en `rag.py` bajo
# `# ── ANTICIPATORY AGENT ──`. Cualquier signal nuevo va al registry de
# `rag_anticipate.signals` y se descubre dinámicamente vía SIGNALS.
_MVP_SIGNAL_KINDS: tuple[str, ...] = (
    "anticipate-calendar",
    "anticipate-echo",
    "anticipate-commitment",
)


# ── Empty shapes (silent-fail returns) ───────────────────────────────────────

def _empty_metrics(days: int) -> dict:
    return {
        "window_days": int(days),
        "total_evaluated": 0,
        "total_selected": 0,
        "total_sent": 0,
        "by_kind": {},
        "send_rate": 0.0,
        "selection_rate": 0.0,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cutoff_iso(days: int) -> str:
    """ISO timestamp para `WHERE ts >= ?`. days <= 0 → epoch (sin filtro)."""
    if days <= 0:
        return "0001-01-01T00:00:00"
    cutoff = datetime.now() - timedelta(days=days)
    return cutoff.isoformat(timespec="seconds")


def _registry_signal_kinds() -> set[str]:
    """Lee `rag_anticipate.signals.base.SIGNALS` y devuelve el set de kinds
    `anticipate-{name}`. Silent-fail si el módulo no carga."""
    out: set[str] = set()
    try:
        from rag_anticipate.signals.base import SIGNALS as _SIGNALS
    except Exception:
        return out
    try:
        # SIGNALS puede ser dict, list de tuples (name, fn), o iterable de
        # nombres. Tolerar las 3 formas para no atarnos a la representación.
        if isinstance(_SIGNALS, dict):
            iterable = _SIGNALS.keys()
        else:
            iterable = _SIGNALS
        for entry in iterable:
            name = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(name, str) and name and not name.startswith("_"):
                out.add(f"anticipate-{name}")
    except Exception:
        pass
    return out


def _known_kinds(conn) -> set[str]:
    """Unión de los 3 MVP + los del registry + los que aparecen en DB."""
    kinds: set[str] = set(_MVP_SIGNAL_KINDS)
    kinds |= _registry_signal_kinds()
    try:
        rows = conn.execute(
            "SELECT DISTINCT kind FROM rag_anticipate_candidates"
        ).fetchall()
        for (k,) in rows:
            if k:
                kinds.add(k)
    except Exception:
        # Tabla missing → solo MVP + registry
        pass
    return kinds


# ── Public API ───────────────────────────────────────────────────────────────


def fetch_metrics(days: int = 7) -> dict:
    """Agregados sobre `rag_anticipate_candidates` en los últimos `days` días.

    Returns:
        {
            "window_days": int,
            "total_evaluated": int,    # filas totales en ventana
            "total_selected": int,     # SUM(selected)
            "total_sent": int,         # SUM(sent)
            "by_kind": {
                "<kind>": {"evaluated": N, "selected": N, "sent": N,
                           "avg_score": float},
                ...
            },
            "send_rate": float,        # total_sent / total_evaluated
            "selection_rate": float,   # total_selected / total_evaluated
        }

    Silent-fail: si la tabla no existe todavía o el DB está inaccesible,
    devuelve el shape con todos los contadores en 0. Nunca raise.
    """
    cutoff = _cutoff_iso(days)
    try:
        with rag._ragvec_state_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT kind, COUNT(*), SUM(selected), SUM(sent), AVG(score) "
                    "FROM rag_anticipate_candidates "
                    "WHERE ts >= ? "
                    "GROUP BY kind",
                    (cutoff,),
                ).fetchall()
            except Exception:
                # Tabla missing o columna distinta → empty result
                return _empty_metrics(days)
    except Exception:
        return _empty_metrics(days)

    by_kind: dict[str, dict] = {}
    total_evaluated = 0
    total_selected = 0
    total_sent = 0
    for kind, count, sel_sum, sent_sum, avg_score in rows:
        evaluated = int(count or 0)
        selected = int(sel_sum or 0)
        sent = int(sent_sum or 0)
        by_kind[str(kind)] = {
            "evaluated": evaluated,
            "selected": selected,
            "sent": sent,
            "avg_score": float(avg_score) if avg_score is not None else 0.0,
        }
        total_evaluated += evaluated
        total_selected += selected
        total_sent += sent

    send_rate = (total_sent / total_evaluated) if total_evaluated else 0.0
    selection_rate = (total_selected / total_evaluated) if total_evaluated else 0.0

    return {
        "window_days": int(days),
        "total_evaluated": total_evaluated,
        "total_selected": total_selected,
        "total_sent": total_sent,
        "by_kind": by_kind,
        "send_rate": send_rate,
        "selection_rate": selection_rate,
    }


def render_dashboard(days: int = 7) -> str:
    """Render texto plano (no rich) con las métricas. Output para CLI/log.

    Si no hay candidates evaluados en la ventana, devuelve un mensaje
    "no data" en lugar de un dashboard con todos los ceros (más claro
    para el operador y mejor para grep).
    """
    metrics = fetch_metrics(days)
    title = f"Anticipate Dashboard ({days} days)"
    underline = "=" * len(title)

    if metrics["total_evaluated"] == 0:
        return (
            f"{title}\n{underline}\n"
            f"(no data — 0 candidates evaluated in the last {days} days)\n"
        )

    send_rate_pct = metrics["send_rate"] * 100.0
    sel_rate_pct = metrics["selection_rate"] * 100.0

    lines: list[str] = []
    lines.append(title)
    lines.append(underline)
    lines.append(f"Total evaluated: {metrics['total_evaluated']}")
    lines.append(f"Total selected:  {metrics['total_selected']}")
    lines.append(f"Total sent:      {metrics['total_sent']}")
    lines.append(f"Send rate:       {send_rate_pct:.1f}%")
    lines.append(f"Selection rate:  {sel_rate_pct:.1f}%")
    lines.append("")
    lines.append("By signal:")

    # Orden estable: por evaluated desc, kind asc para empates.
    by_kind = metrics["by_kind"]
    ordered = sorted(
        by_kind.items(),
        key=lambda kv: (-kv[1]["evaluated"], kv[0]),
    )
    if not ordered:
        lines.append("  (none)")
    else:
        # Padding del nombre del signal para alineación visual.
        max_kind_len = max(len(k) for k, _ in ordered)
        for kind, stats in ordered:
            kind_pad = kind.ljust(max_kind_len)
            lines.append(
                f"  {kind_pad}  "
                f"{stats['evaluated']} eval / "
                f"{stats['selected']} sel / "
                f"{stats['sent']} sent  "
                f"avg_score={stats['avg_score']:.2f}"
            )

    return "\n".join(lines) + "\n"


def top_reasons_skipped(days: int = 7, limit: int = 10) -> list[tuple[str, int]]:
    """Razones más frecuentes por las que se DESCARTÓ un candidate (sent=0).

    Útil para tunear thresholds — si el top reason es "score < 0.35"
    quizás bajar el threshold ayuda; si es "dedup_seen" quizás bajar el
    snooze; si es "max_per_day" quizás subir el cap.

    Returns:
        Lista de tuples `(reason, count)` ordenada desc por count.
        `reason` viene tal cual está en la columna; rows con reason
        NULL/"" quedan agrupadas bajo `"(no reason)"` para no perderlas.
        Lista vacía si la tabla no existe o no hay rows skipped.
    """
    if limit <= 0:
        return []
    cutoff = _cutoff_iso(days)
    try:
        with rag._ragvec_state_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT COALESCE(NULLIF(reason, ''), '(no reason)') AS r, "
                    "COUNT(*) AS c "
                    "FROM rag_anticipate_candidates "
                    "WHERE ts >= ? AND sent = 0 "
                    "GROUP BY r "
                    "ORDER BY c DESC, r ASC "
                    "LIMIT ?",
                    (cutoff, int(limit)),
                ).fetchall()
            except Exception:
                return []
    except Exception:
        return []
    return [(str(r), int(c)) for r, c in rows]


def signal_health(days: int = 7) -> dict:
    """Health check per-signal sobre los últimos `days` días.

    Returns:
        {
            "<kind>": {
                "status": "healthy" | "stale" | "noisy" | "silent",
                "last_emit": ISO_ts | None,    # MAX(ts) de toda la tabla
                "avg_score_7d": float,         # AVG(score) en ventana
                "send_rate": float,            # SUM(sent)/COUNT en ventana
            },
            ...
        }

    Reglas de status (priority en orden, primer match gana):
      1. `silent` — last_emit is None Y 0 emits en ventana
                    (signal nunca emitió en toda la tabla).
      2. `noisy`  — emits en ventana >= 10 Y avg_score < 0.3
                    (genera ruido sub-threshold consistente).
      3. `stale`  — 0 emits en ventana Y last_emit existe pero
                    age_days >= `days` (signal estaba activo y se cayó).
      4. `healthy` — caso normal.

    Silent-fail: dict vacío si DB inaccesible o tabla missing.
    """
    cutoff = _cutoff_iso(days)
    out: dict[str, dict] = {}
    try:
        with rag._ragvec_state_conn() as conn:
            kinds = _known_kinds(conn)
            now = datetime.now()
            for kind in sorted(kinds):
                window_count = 0
                avg_score = 0.0
                send_count = 0
                last_emit: str | None = None

                try:
                    row = conn.execute(
                        "SELECT COUNT(*), AVG(score), SUM(sent) "
                        "FROM rag_anticipate_candidates "
                        "WHERE kind = ? AND ts >= ?",
                        (kind, cutoff),
                    ).fetchone()
                    if row:
                        window_count = int(row[0] or 0)
                        avg_score = float(row[1]) if row[1] is not None else 0.0
                        send_count = int(row[2] or 0)
                except Exception:
                    # Si la tabla no existe, salimos completos con dict vacío.
                    return {}

                try:
                    last_row = conn.execute(
                        "SELECT MAX(ts) FROM rag_anticipate_candidates WHERE kind = ?",
                        (kind,),
                    ).fetchone()
                    if last_row and last_row[0]:
                        last_emit = str(last_row[0])
                except Exception:
                    last_emit = None

                send_rate = (send_count / window_count) if window_count else 0.0

                # Status decision
                if window_count == 0 and last_emit is None:
                    status = "silent"
                elif window_count >= 10 and avg_score < 0.3:
                    status = "noisy"
                elif window_count == 0 and last_emit is not None:
                    # 0 emits en ventana, pero existe last_emit → stale.
                    # (No discriminamos por age exacta — basta con que el
                    # último haya quedado fuera de la ventana de `days`).
                    status = "stale"
                else:
                    # Tiene emits en ventana y no es noisy → healthy.
                    # Adicional: si last_emit es muy viejo respecto a hoy
                    # (>= days) lo marcamos stale incluso con emits, pero
                    # eso no puede pasar por construcción (last_emit >= ts
                    # de algún row en ventana).
                    status = "healthy"

                # Sanity extra: si last_emit existe y es muy viejo respecto
                # a now, forzamos stale (cubre el edge case de filas con
                # ts manualmente inyectado fuera de ventana — el COUNT en
                # ventana lo deja en 0 pero queremos status explícito).
                if status == "healthy" and last_emit is not None:
                    try:
                        last_dt = datetime.fromisoformat(last_emit)
                        if (now - last_dt) >= timedelta(days=days):
                            status = "stale"
                    except Exception:
                        pass

                out[kind] = {
                    "status": status,
                    "last_emit": last_emit,
                    "avg_score_7d": avg_score,
                    "send_rate": send_rate,
                }
    except Exception:
        return {}
    return out


__all__ = [
    "fetch_metrics",
    "render_dashboard",
    "top_reasons_skipped",
    "signal_health",
]
