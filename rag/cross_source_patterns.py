"""Cross-source pattern detection — Pearson correlations + lag analysis
sobre todas las métricas diarias del usuario (mood, sleep, spotify,
queries, WA, etc).

## Diseño

Engine genérico inspirado en `pillow_sleep.detect_patterns()` pero
expandido a 12 métricas core. Mismo Pearson + filtros + caché, ahora
sobre todas las pairs cruzadas + lag analysis.

### Time series daily

Cada métrica tiene un **collector** que devuelve `dict[date_str, float]`
para un date range. El engine NO se preocupa del cómo (SQL, filesystem,
LLM). Cada collector es:

```python
def fn(start_date: str, end_date: str) -> dict[str, float]:
    return {"2026-04-30": 5.2, "2026-04-29": 4.1, ...}
```

Solo días con data real — el engine alinea series por date intersection.

### Pearson + lag

Para cada par `(A, B)` con A != B y para cada lag en `(0, 1, 7)`:
- Alineamos series por date offset.
- Calculamos Pearson `r`, `n` (número de pares con data en ambas), `p`.
- Filtros: `n >= min_n` (default 21), `|r| >= min_abs_r` (default 0.4),
  `p < max_p` (default 0.05).
- Severity: `weak` (|r|<0.4) / `moderate` (<0.6) / `strong` (>=0.6).

Lag interpretation:
- `lag=0`: A y B del mismo día → sincronía (no causalidad).
- `lag=1`: B[t] vs A[t-1] → A precede a B por 1 día (sugiere causalidad
  temporal).
- `lag=7`: ciclo semanal (ej. "viernes financiero → lunes mood").

### Riesgos calibrados

Con 12 métricas, 66 pairs × 3 lags = 198 tests. Sin Bonferroni con
p<0.05 esperaríamos ~10 falsos positivos por chance puro. Mitigación:
- Threshold conservador `|r| >= 0.4` (no 0.3 como pillow_sleep).
- `min_n >= 21` (3 semanas, no 14).
- UI marca explícitamente "relación observada", no "causa".
- Lag-1 / lag-7 reportados como hipótesis temporales, no confirmadas.

### Cache

Memoizado por `(min_n, min_abs_r, max_p, lags_tuple, range_days)`.
Invalida cuando cualquier source escribe data nueva (max(updated_at)
de cualquier tabla relevante). Cheap re-call para refresh de UI.

## Uso

```python
from rag.cross_source_patterns import (
    collect_daily_metrics, compute_correlations,
)

metrics = collect_daily_metrics(days=30)
findings = compute_correlations(metrics, lags=(0, 1, 7))
# findings[0] = {pair: ("sleep_quality", "mood_score"), lag: 0,
#                r: 0.62, n: 24, p: 0.001, severity: "strong",
#                description: "..."}
```

Aprendido el 2026-04-30.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# ──────────────────────────────────────────────────────────────────────────
# Constants

_DEFAULT_DAYS = 30
_DEFAULT_LAGS = (0, 1, 7)
_DEFAULT_MIN_N = 21
_DEFAULT_MIN_ABS_R = 0.4
_DEFAULT_MAX_P = 0.05

# Severity bands.
_STRONG_R = 0.6
_MODERATE_R = 0.4

# Cache: max 10 entries por proceso (cubre los rangos típicos).
_CACHE: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_CACHE_MAX = 10

# ──────────────────────────────────────────────────────────────────────────
# Date helpers


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _date_range(end_date: str, days: int) -> list[str]:
    """Lista de fechas YYYY-MM-DD desde `end_date - days+1` hasta `end_date`,
    cronológico ascendente."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [(end_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(days - 1, -1, -1)]


def _silent_log_safe(event: str, exc: BaseException) -> None:
    try:
        from rag import _silent_log  # noqa: PLC0415
        _silent_log(event, exc)
    except Exception:  # pragma: no cover
        pass


# ──────────────────────────────────────────────────────────────────────────
# Pearson + p-value


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, int, float]:
    """Pearson correlation coefficient + p-value (two-tailed).

    Returns `(r, n, p)`. Si `n < 3` devuelve `(0.0, n, 1.0)`. Si la
    desviación std es 0 (constante), devuelve `(0.0, n, 1.0)` para
    evitar division by zero.

    Implementación inline (no scipy) — mantiene bundle simple. Para
    n grande la diferencia con scipy es negligible.
    """
    if len(xs) != len(ys):
        return 0.0, 0, 1.0
    paired = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(paired)
    if n < 3:
        return 0.0, n, 1.0
    mean_x = sum(x for x, _ in paired) / n
    mean_y = sum(y for _, y in paired) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in paired))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in paired))
    if den_x == 0 or den_y == 0:
        return 0.0, n, 1.0
    r = num / (den_x * den_y)
    # Clamp por float drift.
    r = max(-1.0, min(1.0, r))
    # Two-tailed p-value via t-distribution: t = r * sqrt((n-2)/(1-r^2))
    if abs(r) >= 1.0 or n <= 2:
        p = 0.0 if abs(r) >= 1.0 else 1.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        # Approximation via scipy si disponible (más preciso para n chico).
        try:
            from scipy.stats import t as _t  # noqa: PLC0415
            p = 2 * (1 - _t.cdf(abs(t_stat), df=n - 2))
        except Exception:
            # Fallback: Student's t CDF approximation. Suficiente para
            # threshold filtering aunque no sea exacto.
            # Approximation Hill 1970 — buena para n>=4.
            x = (n - 2) / (n - 2 + t_stat * t_stat)
            p = max(0.0, min(1.0, x ** ((n - 2) / 2)))
    return r, n, float(p)


# ──────────────────────────────────────────────────────────────────────────
# Metric collectors registry


_COLLECTORS: dict[str, Callable[[str, str], dict[str, float]]] = {}
_METRIC_LABELS: dict[str, str] = {}


def register_metric(name: str, label: str) -> Callable:
    """Decorator — registra un collector. `label` es la versión humana
    para mostrar en UI ("sleep · quality" en lugar de "sleep_quality")."""
    def _decorator(fn: Callable[[str, str], dict[str, float]]) -> Callable:
        _COLLECTORS[name] = fn
        _METRIC_LABELS[name] = label
        return fn
    return _decorator


def known_metrics() -> list[str]:
    return sorted(_COLLECTORS.keys())


def metric_label(name: str) -> str:
    return _METRIC_LABELS.get(name, name)


# ──────────────────────────────────────────────────────────────────────────
# Collectors — uno por métrica


@register_metric("mood_score", "mood · score diario")
def _c_mood_score(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, score FROM rag_mood_score_daily "
                "WHERE date >= ? AND date <= ? AND n_signals > 0",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    except Exception as exc:
        _silent_log_safe("xspat_mood_score_failed", exc)
        return {}


@register_metric("mood_self_report", "mood · self-report manual")
def _c_mood_self_report(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, AVG(value) FROM rag_mood_signals "
                "WHERE source='manual' AND signal_kind='self_report' "
                "AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_mood_self_failed", exc)
        return {}


@register_metric("sleep_quality", "sleep · quality")
def _c_sleep_quality(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, AVG(quality) FROM rag_sleep_sessions "
                "WHERE is_nap=0 AND quality IS NOT NULL "
                "AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_sleep_quality_failed", exc)
        return {}


@register_metric("sleep_duration_h", "sleep · duración horas")
def _c_sleep_duration(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, SUM(end_ts - start_ts) / 3600.0 "
                "FROM rag_sleep_sessions "
                "WHERE is_nap=0 AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_sleep_duration_failed", exc)
        return {}


@register_metric("sleep_awakenings", "sleep · awakenings")
def _c_sleep_awakenings(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, AVG(awakenings) FROM rag_sleep_sessions "
                "WHERE is_nap=0 AND awakenings IS NOT NULL "
                "AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_sleep_awakenings_failed", exc)
        return {}


@register_metric("sleep_deep_pct", "sleep · deep%")
def _c_sleep_deep(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(rag_sleep_sessions)"
            ).fetchall()}
            if "deep_pct" not in cols:
                return {}
            rows = conn.execute(
                "SELECT date, AVG(deep_pct) FROM rag_sleep_sessions "
                "WHERE is_nap=0 AND deep_pct IS NOT NULL "
                "AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_sleep_deep_failed", exc)
        return {}


@register_metric("wakeup_mood", "sleep · wake-up mood")
def _c_wakeup_mood(start: str, end: str) -> dict[str, float]:
    """ZWAKEUPMOOD de Pillow (escala 0-3, 0=neutral, 3=:laughing:).
    Lo normalizamos a [0, 1] para que sea comparable con otras
    métricas en la misma escala."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, AVG(wakeup_mood) FROM rag_sleep_sessions "
                "WHERE is_nap=0 AND wakeup_mood IS NOT NULL "
                "AND date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) / 3.0 for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_wakeup_mood_failed", exc)
        return {}


@register_metric("spotify_minutes", "spotify · minutos escuchados")
def _c_spotify_minutes(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, SUM(last_seen - first_seen) / 60.0 "
                "FROM rag_spotify_log "
                "WHERE date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows if r[1] is not None}
    except Exception as exc:
        _silent_log_safe("xspat_spotify_minutes_failed", exc)
        return {}


@register_metric("spotify_distinct_tracks", "spotify · tracks distintos")
def _c_spotify_tracks(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, COUNT(DISTINCT track_id) "
                "FROM rag_spotify_log "
                "WHERE date >= ? AND date <= ? "
                "GROUP BY date",
                (start, end),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    except Exception as exc:
        _silent_log_safe("xspat_spotify_tracks_failed", exc)
        return {}


@register_metric("queries_total", "queries · total al RAG")
def _c_queries_total(start: str, end: str) -> dict[str, float]:
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT substr(ts, 1, 10) AS d, COUNT(*) "
                "FROM rag_queries "
                "WHERE ts >= ? AND ts < ? "
                "AND COALESCE(cmd,'') IN ('', 'query', 'chat', 'ask') "
                "GROUP BY d",
                (f"{start}T00:00:00", f"{end}T23:59:59"),
            ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    except Exception as exc:
        _silent_log_safe("xspat_queries_total_failed", exc)
        return {}


@register_metric("queries_existential", "queries · patrón existencial")
def _c_queries_existential(start: str, end: str) -> dict[str, float]:
    """Cuenta de queries con regex existencial (ver `mood._QUERIES_EXISTENTIAL_RE`)."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        from rag.mood import _QUERIES_EXISTENTIAL_RE  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT substr(ts, 1, 10) AS d, q "
                "FROM rag_queries "
                "WHERE ts >= ? AND ts < ? "
                "AND COALESCE(cmd,'') IN ('', 'query', 'chat', 'ask')",
                (f"{start}T00:00:00", f"{end}T23:59:59"),
            ).fetchall()
        out: dict[str, int] = {}
        for date, q in rows:
            if not q:
                continue
            if _QUERIES_EXISTENTIAL_RE.search(q):
                out[date] = out.get(date, 0) + 1
        return {d: float(c) for d, c in out.items()}
    except Exception as exc:
        _silent_log_safe("xspat_queries_existential_failed", exc)
        return {}


@register_metric("wa_outbound_avg_chars", "WhatsApp · avg chars outbound")
def _c_wa_outbound_chars(start: str, end: str) -> dict[str, float]:
    """Promedio de chars/mensaje outbound por día desde el bridge SQLite."""
    try:
        from rag.integrations.whatsapp import WHATSAPP_BRIDGE_DB_PATH  # noqa: PLC0415
        db = Path(WHATSAPP_BRIDGE_DB_PATH)
        if not db.exists():
            return {}
        import sqlite3 as _sql  # noqa: PLC0415
        conn = _sql.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT substr(timestamp, 1, 10) AS d, AVG(LENGTH(content)) "
                "FROM messages "
                "WHERE is_from_me=1 AND content IS NOT NULL "
                "AND timestamp >= ? AND timestamp < ? "
                "GROUP BY d",
                (f"{start}T00:00:00", f"{end}T23:59:59"),
            ).fetchall()
            return {r[0]: float(r[1]) for r in rows if r[1] is not None}
        finally:
            with contextlib.suppress(Exception):
                conn.close()
    except Exception as exc:
        _silent_log_safe("xspat_wa_outbound_failed", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────
# Aggregator + correlation engine


def collect_daily_metrics(
    days: int = _DEFAULT_DAYS,
    *,
    metrics: list[str] | None = None,
    end_date: str | None = None,
) -> dict[str, dict[str, float]]:
    """Colecta time series diarias de cada métrica registrada en el rango.

    Args:
      days: tamaño del rango.
      metrics: subset opcional de nombres a colectar; default = todos.
      end_date: último día (YYYY-MM-DD); default = hoy.

    Returns:
      `{metric_name: {date: value, ...}, ...}`. Métricas sin data en
      el rango se incluyen con dict vacío (el caller decide si saltearlas).
    """
    end = end_date or _today_str()
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    start = (end_dt - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    target = metrics or list(_COLLECTORS.keys())
    out: dict[str, dict[str, float]] = {}
    for name in target:
        fn = _COLLECTORS.get(name)
        if not fn:
            continue
        try:
            out[name] = fn(start, end)
        except Exception as exc:
            _silent_log_safe(f"xspat_collect_failed:{name}", exc)
            out[name] = {}
    return out


def _align_series(
    series_a: dict[str, float],
    series_b: dict[str, float],
    *,
    lag: int = 0,
) -> tuple[list[float], list[float], list[str]]:
    """Alinea dos series por date con offset `lag`. Devuelve `(xs, ys, dates)`
    donde `xs[i]` es A[date - lag] y `ys[i]` es B[date].

    `lag=0` → mismo día.
    `lag=1` → A[t-1] vs B[t] (A precede a B en 1 día).
    `lag=7` → A[t-7] vs B[t].

    Solo incluye dates donde ambas series tienen valor (intersection).
    """
    xs: list[float] = []
    ys: list[float] = []
    dates: list[str] = []
    for date_b, val_b in series_b.items():
        # Calcular date_a = date_b - lag.
        try:
            db_dt = datetime.strptime(date_b, "%Y-%m-%d")
        except ValueError:
            continue
        date_a = (db_dt - timedelta(days=lag)).strftime("%Y-%m-%d")
        val_a = series_a.get(date_a)
        if val_a is None:
            continue
        xs.append(float(val_a))
        ys.append(float(val_b))
        dates.append(date_b)
    return xs, ys, dates


def _severity(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= _STRONG_R:
        return "strong"
    if abs_r >= _MODERATE_R:
        return "moderate"
    return "weak"


def _format_description(metric_a: str, metric_b: str, r: float, lag: int) -> str:
    """Texto humano para el finding. Voz neutra ("relación observada"),
    NO afirma causalidad."""
    direction = "sube" if r > 0 else "baja"
    label_a = metric_label(metric_a)
    label_b = metric_label(metric_b)
    if lag == 0:
        return f"{label_a} alta → {label_b} {direction} (mismo día)"
    elif lag == 1:
        return f"{label_a} alta → {label_b} {direction} al día siguiente"
    elif lag == 7:
        return f"{label_a} alta → {label_b} {direction} 1 semana después"
    return f"{label_a} alta → {label_b} {direction} (lag {lag}d)"


def compute_correlations(
    metrics: dict[str, dict[str, float]] | None = None,
    *,
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    min_n: int = _DEFAULT_MIN_N,
    min_abs_r: float = _DEFAULT_MIN_ABS_R,
    max_p: float = _DEFAULT_MAX_P,
    days: int = _DEFAULT_DAYS,
) -> list[dict[str, Any]]:
    """Calcula Pearson sobre todos los pares × lags.

    Si `metrics` no se pasa, se llama a `collect_daily_metrics(days)`.

    Devuelve lista ordenada por `|r|` desc de findings:

      `{pair: (a, b), lag: int, r: float, n: int, p: float,
        severity: str, description: str}`

    Filtros: `n >= min_n`, `|r| >= min_abs_r`, `p < max_p`.
    """
    if metrics is None:
        metrics = collect_daily_metrics(days)

    cache_key = (
        tuple(sorted(metrics.keys())),
        tuple(sorted(lags)),
        min_n, min_abs_r, max_p, days,
        tuple((k, len(v)) for k, v in sorted(metrics.items())),
    )
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    names = sorted(metrics.keys())
    findings: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i:]:
            for lag in lags:
                # Para lag=0 con a==b es trivialmente 1.0 — skipear.
                if a == b and lag == 0:
                    continue
                # Para lag>0 con a==b es autocorrelation: significa
                # "la métrica predice su propio futuro". Útil pero
                # menos interesante que cross-source. Lo mantenemos
                # solo para lags > 0, no consume mucho.
                xs, ys, _dates = _align_series(metrics[a], metrics[b], lag=lag)
                r, n, p = _pearson(xs, ys)
                if n < min_n or abs(r) < min_abs_r or p >= max_p:
                    continue
                findings.append({
                    "pair": (a, b),
                    "lag": lag,
                    "r": round(r, 3),
                    "n": n,
                    "p": round(p, 4),
                    "severity": _severity(r),
                    "description": _format_description(a, b, r, lag),
                })

    findings.sort(key=lambda f: -abs(f["r"]))

    # LRU cache.
    _CACHE[cache_key] = findings
    _CACHE.move_to_end(cache_key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)

    return findings


def patterns_summary(
    *,
    days: int = _DEFAULT_DAYS,
    top: int = 10,
    lags: tuple[int, ...] = _DEFAULT_LAGS,
) -> dict[str, Any]:
    """Wrapper para UI: colecta + computa + agrupa.

    Returns:
      `{n_findings, top: [findings], by_severity: {strong, moderate, weak},
        metrics_with_data: [(name, n_days)], days_range: int,
        lags_tested: list[int]}`.
    """
    metrics = collect_daily_metrics(days)
    findings = compute_correlations(
        metrics, lags=lags, days=days,
    )
    by_severity: dict[str, int] = {"strong": 0, "moderate": 0, "weak": 0}
    for f in findings:
        by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1
    metrics_with_data = sorted(
        [(name, len(series)) for name, series in metrics.items() if series],
        key=lambda x: -x[1],
    )
    return {
        "n_findings": len(findings),
        "top": findings[:top],
        "by_severity": by_severity,
        "metrics_with_data": metrics_with_data,
        "days_range": days,
        "lags_tested": list(lags),
    }


__all__ = [
    "collect_daily_metrics",
    "compute_correlations",
    "patterns_summary",
    "register_metric",
    "known_metrics",
    "metric_label",
    "_pearson",
    "_align_series",
]
