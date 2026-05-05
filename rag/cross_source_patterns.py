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
import math
import re
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


@register_metric("gmail_received", "gmail · mensajes recibidos")
def _c_gmail_received(start: str, end: str) -> dict[str, float]:
    """Count de mensajes gmail recibidos por día.

    Source: notas en `<vault>/04-Archive/99-obsidian-system/99-AI/external-ingest/Gmail/<YYYY-MM-DD>.md` que
    el ingester de gmail genera 1×/día con un dump de las últimas 48h.
    El frontmatter trae `message_count: N` que es el count exacto.
    Más confiable que parsear el body buscando subjects (formato puede
    cambiar). Skipea snapshots overlap por window_hours — solo
    contamos cada YYYY-MM-DD una vez (la del archivo con ese nombre).

    Devuelve dict vacío si:
      - vault no resoluble
      - folder Gmail/ no existe (usuario sin gmail integration)
      - parse del frontmatter falla (silent-fail)
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return {}
    folder = VAULT_PATH / "04-Archive" / "99-obsidian-system" / "99-AI" / "external-ingest" / "Gmail"
    if not folder.exists() or not folder.is_dir():
        return {}
    out: dict[str, float] = {}
    # Filename pattern: YYYY-MM-DD.md (matchea el ingester actual)
    name_re = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
    fm_re = re.compile(r"^message_count:\s*(\d+)\s*$", re.MULTILINE)
    try:
        for path in folder.iterdir():
            m = name_re.match(path.name)
            if not m:
                continue
            date = m.group(1)
            if date < start or date > end:
                continue
            try:
                # Solo leer primeras 30 líneas (el frontmatter es chico).
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    head = "".join(f.readline() for _ in range(30))
            except OSError:
                continue
            mm = fm_re.search(head)
            if mm:
                try:
                    out[date] = float(mm.group(1))
                except ValueError:
                    continue
    except Exception as exc:
        _silent_log_safe("xspat_gmail_received_failed", exc)
        return {}
    return out


@register_metric("vault_notes_touched", "vault · notas tocadas")
def _c_vault_notes_touched(start: str, end: str) -> dict[str, float]:
    """Count de notas .md del vault con mtime en cada día del rango.

    rglob completo del vault filtrando:
      - Solo `.md`
      - NO system files (`.obsidian/`, files que arrancan con `_`)
      - NO bajo `04-Archive/99-obsidian-system/` (auto-generated)
      - mtime dentro del rango pedido

    Cuesta más que las queries SQL (~200ms para vault chico,
    ~2s para uno grande). Tolerable porque el engine tiene cache
    LRU + el endpoint solo se llama on-demand desde el panel.

    Métrica útil para correlar productividad / engagement con el vault
    vs mood / sleep. Patron esperado: días con muchas notas tocadas
    suelen ser días "productivos" — vale ver si correlaciona con mood.
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return {}
    if not VAULT_PATH.exists():
        return {}

    # Convert range to epoch for fast comparison.
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
    except ValueError:
        return {}

    out: dict[str, int] = {}
    try:
        for path in VAULT_PATH.rglob("*.md"):
            # Skip system folders.
            rel = path.relative_to(VAULT_PATH)
            parts = rel.parts
            if any(p.startswith(".") or p.startswith("_") for p in parts):
                continue
            if len(parts) >= 2 and parts[0] == "04-Archive" and parts[1] == "99-obsidian-system":
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < start_ts or mtime >= end_ts:
                continue
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            out[date] = out.get(date, 0) + 1
    except Exception as exc:
        _silent_log_safe("xspat_vault_notes_failed", exc)
        return {}
    return {d: float(c) for d, c in out.items()}


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


# ──────────────────────────────────────────────────────────────────────────
# Mood prediction (LinearRegression con lag-1)


_PREDICT_MIN_DAYS = 21
_PREDICT_TARGET = "mood_score"
# Features a usar como predictoras (lag-1). NO incluyen mood_score
# mismo (eso seria autocorrelation y daria 1.0 trivial).
_PREDICT_FEATURES = [
    "sleep_quality", "sleep_duration_h", "sleep_awakenings",
    "wakeup_mood", "spotify_minutes", "spotify_distinct_tracks",
    "queries_total", "queries_existential", "wa_outbound_avg_chars",
    "mood_self_report",
]


def _build_training_data(
    metrics: dict[str, dict[str, float]],
    *,
    target: str = _PREDICT_TARGET,
    features: list[str] | None = None,
    lag: int = 1,
) -> tuple[list[list[float]], list[float], list[str], list[str]]:
    """Arma matriz X (rows = días, cols = features) + vector y (target).

    Para cada día `t` donde `target[t]` y TODAS las features en `t-lag`
    existen, agregamos una row a X y un valor a y. Días con missing
    en cualquier feature se saltean (no imputamos para no contaminar
    con guesses).

    Returns: `(X, y, feature_names, target_dates)`.
    """
    feature_names = features or _PREDICT_FEATURES
    # Una feature solo es "available" si está en metrics Y tiene
    # data en al menos algún día. Features registradas pero
    # ENTERAMENTE vacías se saltean — sino bloqueamos el training
    # cuando agregamos una métrica nueva que aún no recolectó nada.
    available_features = [
        f for f in feature_names
        if f in metrics and len(metrics[f]) > 0
    ]
    X: list[list[float]] = []
    y: list[float] = []
    target_dates: list[str] = []
    target_series = metrics.get(target, {})
    for date_b, val_y in target_series.items():
        try:
            db_dt = datetime.strptime(date_b, "%Y-%m-%d")
        except ValueError:
            continue
        date_lag = (db_dt - timedelta(days=lag)).strftime("%Y-%m-%d")
        row: list[float] = []
        complete = True
        for fname in available_features:
            v = metrics[fname].get(date_lag)
            if v is None:
                complete = False
                break
            row.append(float(v))
        if not complete:
            continue
        X.append(row)
        y.append(float(val_y))
        target_dates.append(date_b)
    return X, y, available_features, target_dates


def predict_mood_tomorrow(
    *,
    days: int = 60,
    metrics: dict[str, dict[str, float]] | None = None,
    target: str = _PREDICT_TARGET,
    features: list[str] | None = None,
) -> dict[str, Any] | None:
    """Predice `mood_score` de mañana usando RidgeCV sobre las features
    (lag-1) de los últimos `days` días + features de hoy como input.

    Algoritmo:
      1. Colectar métricas de los últimos `days` días.
      2. Build training data: cada row es (features[t-1] → mood[t]).
      3. Si tenemos < `_PREDICT_MIN_DAYS` rows entrenables → None.
      4. Fit `RidgeCV` (alpha auto-seleccionada por LOO interno) con
         fallback a `LinearRegression` si Ridge falla.
      5. Cross-validation R² out-of-sample con `TimeSeriesSplit`
         (2-5 folds según n) → eso es `confidence`. NO usamos R²
         in-sample como antes — eso era overconfident con n cercano
         al mínimo.
      6. Tomar features de HOY (que predicen MAÑANA).
      7. Predecir + clamp a [-1, +1].
      8. Calcular SHAP-style importance (`coef * (value_today - mean_train)`)
         que mide cuánto la feature de HOY desvía la predicción
         respecto a un día promedio. Más interpretable que el
         producto crudo `coef * value_today`.

    Returns:
      `{
        prediction: float (-1..+1) | None,
        confidence: float (CV R², puede ser <0 si modelo peor que media),
        confidence_in_sample: float (R² del fit, retrocompat / debug),
        model: "ridge_cv" | "linear",
        alpha: float | None (alpha de Ridge, None si fallback a Linear),
        cv_n_splits: int (0 si CV no se pudo correr),
        n_training_days: int,
        target_date: str (mañana, YYYY-MM-DD),
        based_on_date: str (hoy, YYYY-MM-DD),
        top_features: [{
          feature, coef, value_today, value_baseline,
          contribution,                # legacy: coef*value_today
          deviation_contribution,      # SHAP-style: coef*(today-mean)
        }, ...] (ordenadas por |deviation_contribution|, top 5)
      }`
      o `None` si no hay enough data o sklearn no está disponible.

    Importante: NO interpretar como verdad — la regresión asume
    estabilidad de patrones que pueden cambiar. UI debe marcar como
    "estimación basada en patrones recientes" y mostrar `confidence`
    (CV R²) prominentemente — si está cerca de 0 o negativo, el
    modelo no aprendió nada útil.
    """
    if metrics is None:
        metrics = collect_daily_metrics(days=days)

    X, y, feature_names, _dates = _build_training_data(
        metrics, target=target, features=features, lag=1,
    )
    if len(X) < _PREDICT_MIN_DAYS:
        return None

    try:
        import numpy as np  # noqa: PLC0415
        from sklearn.linear_model import (  # noqa: PLC0415
            LinearRegression,
            RidgeCV,
        )
        from sklearn.model_selection import (  # noqa: PLC0415
            TimeSeriesSplit,
            cross_val_score,
        )
    except Exception as exc:
        _silent_log_safe("xspat_predict_sklearn_failed", exc)
        return None

    # 1. Modelo: RidgeCV con grid de alphas razonables — auto-selecciona
    #    el mejor por leave-one-out CV interno. Para 21-30 days y ~10
    #    features, regularizar evita overfit cuando alguna feature es
    #    casi-constante o colineal con otra. Si por algún motivo Ridge
    #    falla (raro: matriz singular extrema), caemos a LinearRegression.
    try:
        X_np = np.array(X, dtype=float)
        y_np = np.array(y, dtype=float)
        try:
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            model.fit(X_np, y_np)
            model_name = "ridge_cv"
            alpha = float(getattr(model, "alpha_", 0.0))
        except Exception:
            model = LinearRegression()
            model.fit(X_np, y_np)
            model_name = "linear"
            alpha = None
        # R² in-sample (lo que devolvíamos antes como confidence). Lo
        # mantenemos para retrocompat / debug — pero NO es la métrica
        # principal de honestidad.
        confidence_in_sample = float(model.score(X_np, y_np))
    except Exception as exc:
        _silent_log_safe("xspat_predict_fit_failed", exc)
        return None

    # 2. Cross-validation R² (out-of-sample, time-aware). TimeSeriesSplit
    #    respeta orden temporal: cada fold entrena en pasado y testea en
    #    futuro. Para n=21 → 2 folds; para n=42 → ~5 folds. Nunca menos
    #    de 2 (si n<10 saltamos CV y devolvemos in-sample como confidence,
    #    pero esto es muy raro porque _PREDICT_MIN_DAYS=21).
    n_splits = max(2, min(5, len(X) // 7))
    cv_r2: float | None = None
    try:
        if len(X) >= 2 * n_splits + 1:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(
                model, X_np, y_np, cv=tscv, scoring="r2",
            )
            # cross_val_score puede devolver R² negativos cuando el
            # modelo es peor que predecir la media. Es signal valid:
            # confidence baja → UI debe mostrar warning.
            cv_r2 = float(cv_scores.mean())
    except Exception as exc:
        _silent_log_safe("xspat_predict_cv_failed", exc)
        cv_r2 = None

    # `confidence` ahora es CV R² (out-of-sample, honest). Si CV no se
    # pudo correr por algún motivo, fallback a in-sample para no romper
    # el contrato (callers asumen `confidence: float`).
    confidence = cv_r2 if cv_r2 is not None else confidence_in_sample

    # 3. Features de HOY (las del último día disponible). Esas predicen
    #    MAÑANA bajo la asumción de que los patrones de los últimos
    #    días se mantienen.
    today_str = _today_str()
    target_date = (datetime.strptime(today_str, "%Y-%m-%d")
                   + timedelta(days=1)).strftime("%Y-%m-%d")
    today_features = []
    for fname in feature_names:
        v = metrics.get(fname, {}).get(today_str)
        if v is None:
            # Sin features de hoy completas, no podemos predecir mañana.
            return {
                "prediction": None,
                "confidence": round(confidence, 3),
                "confidence_in_sample": round(confidence_in_sample, 3),
                "model": model_name,
                "alpha": alpha,
                "cv_n_splits": n_splits if cv_r2 is not None else 0,
                "n_training_days": len(X),
                "target_date": target_date,
                "based_on_date": today_str,
                "top_features": [],
                "reason": f"missing_feature_today:{fname}",
            }
        today_features.append(float(v))

    try:
        prediction = float(model.predict(np.array([today_features]))[0])
        # Clamp a [-1, +1] que es el rango natural del mood_score.
        prediction = max(-1.0, min(1.0, prediction))
    except Exception as exc:
        _silent_log_safe("xspat_predict_inference_failed", exc)
        return None

    # 4. Importance estilo SHAP (lineal exacto):
    #    contribution        = coef * value_today                (legacy)
    #    deviation_contribution = coef * (value_today - mean(value_train))
    #
    # La deviation es más interpretable porque mide "cuánto la feature
    # de HOY desvía la predicción respecto a un día promedio". Si
    # value_today == mean → contribución 0 (no es noticia). Si
    # value_today >> mean y coef > 0 → push positivo fuerte. Esto es
    # el SHAP value exacto para modelos lineales (cuando el baseline
    # es la expectativa marginal de la feature).
    coefs = model.coef_.tolist()
    feature_means = X_np.mean(axis=0).tolist()
    contributions = []
    for fname, coef, value, mean_v in zip(
        feature_names, coefs, today_features, feature_means,
    ):
        contrib = coef * value
        deviation_contrib = coef * (value - mean_v)
        contributions.append({
            "feature": fname,
            "coef": round(float(coef), 4),
            "value_today": round(value, 3),
            "value_baseline": round(float(mean_v), 3),
            "contribution": round(float(contrib), 3),
            "deviation_contribution": round(float(deviation_contrib), 3),
        })
    # Ordenamos por |deviation_contribution| — qué tan inusual es hoy
    # respecto al baseline + cuánto pesa esa feature en el modelo.
    contributions.sort(key=lambda x: -abs(x["deviation_contribution"]))

    return {
        "prediction": round(prediction, 3),
        "confidence": round(confidence, 3),
        "confidence_in_sample": round(confidence_in_sample, 3),
        "model": model_name,
        "alpha": alpha,
        "cv_n_splits": n_splits if cv_r2 is not None else 0,
        "n_training_days": len(X),
        "target_date": target_date,
        "based_on_date": today_str,
        "top_features": contributions[:5],
    }


__all__ = [
    "collect_daily_metrics",
    "compute_correlations",
    "patterns_summary",
    "predict_mood_tomorrow",
    "register_metric",
    "known_metrics",
    "metric_label",
    "_pearson",
    "_align_series",
    "_build_training_data",
]
