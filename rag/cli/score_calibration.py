"""`rag calibrate` — score calibration per-source.

Phase 3 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer la sub-API de score calibration desde `rag/__init__.py`
(que ya bajó a 60.6k LOC tras Phase 3 health + synth-queries).

## Por qué calibration

Cross-encoder scores del bge-reranker-v2-m3 son incomparables entre
sources (vault: 0.8-1.2, whatsapp: 0.02-0.10, calendar: 0.3-0.7).
Esto forzó hacks como `RAG_WA_FAST_PATH_THRESHOLD=0.05` + branch 1/2.
La solución estructural: isotonic regression per-source que mapea
raw → calibrated en [0, 1], haciendo el gate de fast-path + ranking
cross-source uniforme.

## Isotonic regression

- Monotonic non-parametric: preserva el orden del cross-encoder
  (mayor raw → mayor calibrated) pero permite ajustar la escala.
- Entrenado con feedback: (raw_score, 1) para corrective_paths o
  paths que ganaron thumbs-up; (raw_score, 0) para paths que no.
- Persistido como 2 arrays float (JSON) en rag_score_calibration,
  aplicado en retrieve() via piecewise-linear interpolation.

## Feature flag

`RAG_SCORE_CALIBRATION` — default OFF inicialmente hasta que
(a) haya data suficiente del auto-harvest (Feature #1) y (b) el
`rag eval` confirme no-regresión. Activar con:
  export RAG_SCORE_CALIBRATION=1
Rollback: unset the env var — retrieve() vuelve a usar raw scores.

## Lazy imports

`_ragvec_state_conn`, `console` viven en `rag/__init__.py`. Lazy
adentro de las funciones que los usan.

## Re-export

`rag/__init__.py` hace `from rag.cli.score_calibration import *`.
`calibrate_cli` se registra al final del módulo via
`cli.add_command(calibrate_cli)`.
"""

from __future__ import annotations

import json
import os
import threading

import click

__all__ = [
    "_SCORE_CALIBRATION_ENABLED",
    "_CALIBRATION_SOURCES",
    "SYNTH_FALLBACK_THRESHOLD",
    "_calibration_cache",
    "_calibration_cache_lock",
    "_reset_calibration_cache",
    "_load_calibration",
    "calibrate_score",
    "_fit_isotonic_from_pairs",
    "_gather_calibration_pairs",
    "_classify_source_from_path",
    "_gather_synthetic_calibration_pairs",
    "train_calibration",
    "calibrate_cli",
]


_SCORE_CALIBRATION_ENABLED = os.environ.get(
    "RAG_SCORE_CALIBRATION", ""
).strip().lower() in ("1", "true", "yes")

# Sources que entrenamos por separado. Todo lo que no esté acá cae al
# bucket "vault" (fallback conservador — si no reconocemos el source,
# usamos la calibración del vault ya que es la distribución más ancha).
_CALIBRATION_SOURCES = (
    "vault", "whatsapp", "calendar", "gmail",
    "drive", "reminders", "safari", "contacts", "calls",
)

# In-process cache del modelo por source: {source: (raw_knots, cal_knots)}
# Invalidate con `_reset_calibration_cache()` post-training.
_calibration_cache: dict[str, tuple[list[float], list[float]]] | None = None
_calibration_cache_lock = threading.Lock()


def _reset_calibration_cache() -> None:
    """Invalidate the in-process calibration cache.

    Called post `train_calibration()` so the next retrieve sees fresh knots.
    Idempotent; safe to call from any thread.
    """
    global _calibration_cache
    with _calibration_cache_lock:
        _calibration_cache = None


def _load_calibration() -> dict[str, tuple[list[float], list[float]]]:
    """Lazy-load the calibration models from rag_score_calibration.

    Returns a dict {source: (raw_knots, cal_knots)} where both arrays are
    sorted parallel floats defining the piecewise-linear map raw→cal.
    Empty dict when the table doesn't exist or no rows — calibrate_score()
    then falls back to the raw value.
    """
    from rag import _ragvec_state_conn  # noqa: PLC0415

    global _calibration_cache
    with _calibration_cache_lock:
        if _calibration_cache is not None:
            return _calibration_cache
    out: dict[str, tuple[list[float], list[float]]] = {}
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, raw_knots_json, cal_knots_json "
                "FROM rag_score_calibration"
            ).fetchall()
    except Exception:
        rows = []
    for source, raw_json, cal_json in rows:
        try:
            raw_k = [float(x) for x in json.loads(raw_json)]
            cal_k = [float(x) for x in json.loads(cal_json)]
        except Exception:
            continue
        if len(raw_k) < 2 or len(raw_k) != len(cal_k):
            continue
        out[str(source)] = (raw_k, cal_k)
    with _calibration_cache_lock:
        _calibration_cache = out
    return out


def calibrate_score(source: str | None, raw: float) -> float:
    """Map a raw cross-encoder score to calibrated probability in [0, 1].

    When the feature flag is OFF or no model exists for `source`, returns
    `raw` unchanged — safe to call unconditionally from retrieve().

    Uses piecewise-linear interpolation between trained knots, clamped
    at the boundaries. source=None or unknown → tries the `vault` bucket
    (most common) before falling back to raw. Preserves ordering within
    a source (isotonic = monotonic).

    Never raises. Returns a finite float.
    """
    try:
        raw_f = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not _SCORE_CALIBRATION_ENABLED:
        return raw_f
    models = _load_calibration()
    if not models:
        return raw_f
    key = (source or "").strip().lower() if isinstance(source, str) else ""
    knots = models.get(key) if key else None
    if knots is None:
        knots = models.get("vault")
    if knots is None:
        return raw_f
    raw_k, cal_k = knots
    # Clamp at boundaries.
    if raw_f <= raw_k[0]:
        return cal_k[0]
    if raw_f >= raw_k[-1]:
        return cal_k[-1]
    # Binary search for the bracket. Small arrays — linear scan is fine
    # and branch-predictable; keep it simple.
    for i in range(1, len(raw_k)):
        if raw_f <= raw_k[i]:
            lo, hi = raw_k[i - 1], raw_k[i]
            lo_cal, hi_cal = cal_k[i - 1], cal_k[i]
            if hi == lo:
                return (lo_cal + hi_cal) / 2.0
            t = (raw_f - lo) / (hi - lo)
            return lo_cal + t * (hi_cal - lo_cal)
    return cal_k[-1]


def _fit_isotonic_from_pairs(
    pairs: list[tuple[float, int]],
) -> tuple[list[float], list[float]] | None:
    """Fit isotonic regression on (raw_score, label ∈ {0,1}) pairs.

    Returns (raw_knots, cal_knots) as parallel sorted arrays defining
    the piecewise-linear map, or None when the input is too small /
    degenerate (ex. all-same labels).

    Uses scikit-learn's IsotonicRegression(out_of_bounds='clip'). The
    output is simplified: we keep only the breakpoints (consecutive
    equal cal_knots get collapsed) to produce a compact JSON payload.
    """
    if len(pairs) < 5:
        return None
    labels = {p[1] for p in pairs}
    if len(labels) < 2:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    except Exception:
        return None
    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)
    # Clip raw scores to a sane range — extreme outliers (NaN, inf)
    # would break the fit.
    x = np.clip(x, -10.0, 20.0)
    model = IsotonicRegression(
        out_of_bounds="clip", y_min=0.0, y_max=1.0,
    )
    try:
        model.fit(x, y)
    except Exception:
        return None
    # Sample at ~50 quantiles of x to get a compact piecewise rep.
    try:
        x_sorted = np.sort(np.unique(x))
        if len(x_sorted) <= 50:
            probe = x_sorted
        else:
            probe = np.quantile(x_sorted, np.linspace(0, 1, 50))
            probe = np.unique(probe)
        cal = model.predict(probe)
    except Exception:
        return None
    raw_k = [float(v) for v in probe]
    cal_k = [float(v) for v in cal]
    # Collapse flat plateaus (keep first + last of each run) for compactness.
    if len(raw_k) > 2:
        keep_idx = [0]
        for i in range(1, len(raw_k) - 1):
            if abs(cal_k[i] - cal_k[i - 1]) > 1e-6 or abs(cal_k[i] - cal_k[i + 1]) > 1e-6:
                keep_idx.append(i)
        keep_idx.append(len(raw_k) - 1)
        raw_k = [raw_k[i] for i in keep_idx]
        cal_k = [cal_k[i] for i in keep_idx]
    return (raw_k, cal_k)


def _gather_calibration_pairs(
    conn, source: str, since_days: int = 90,
) -> list[tuple[float, int]]:
    """Extract (raw_score, label ∈ {0,1}) training pairs for a source.

    Positive labels come from rag_feedback rows with rating=1 joined
    back to the originating rag_queries.scores_json (via q + ts window).
    Negative labels come from the same rag_queries' OTHER paths (those
    NOT in rag_feedback positives for that query).

    The path→source mapping uses prefix heuristics on the vault-relative
    path. Paths with `<scheme>://` URIs map to their scheme (whatsapp://,
    gmail://, calendar://, reminders://, etc). `.md` paths under the
    WhatsApp folder get `source='whatsapp'`; everything else is `vault`.
    """
    pairs: list[tuple[float, int]] = []
    # Collect positives: rag_feedback rating=1 with matched path.
    try:
        pos_rows = conn.execute(
            "SELECT f.q, f.paths_json, "
            "       json_extract(f.extra_json, '$.corrective_path') AS cp, "
            "       q.paths_json AS qp, q.scores_json "
            " FROM rag_feedback f "
            " LEFT JOIN rag_queries q "
            "   ON q.q = f.q "
            "   AND ABS(julianday(q.ts) - julianday(f.ts)) < 1 "
            f" WHERE f.rating = 1 "
            f"   AND f.ts > datetime('now', '-{int(since_days)} days')"
        ).fetchall()
    except Exception:
        pos_rows = []

    # Quickly map (q, path) → raw_score from rag_queries on-the-fly.
    for q, fb_paths_json, cp, q_paths_json, q_scores_json in pos_rows:
        try:
            fb_paths = json.loads(fb_paths_json) if fb_paths_json else []
            q_paths = json.loads(q_paths_json) if q_paths_json else []
            q_scores = json.loads(q_scores_json) if q_scores_json else []
        except Exception:
            continue
        # Prefer corrective_path (known golden). Fall back to fb.paths[0]
        # when corrective_path is missing (older data) — rating=1 on a
        # single-path set still is reliable signal.
        golden_paths: list[str] = []
        if cp and isinstance(cp, str):
            golden_paths.append(cp)
        elif len(fb_paths) == 1:
            golden_paths.append(fb_paths[0])
        for gp in golden_paths:
            if gp in q_paths:
                idx = q_paths.index(gp)
                if idx < len(q_scores):
                    try:
                        raw = float(q_scores[idx])
                    except (TypeError, ValueError):
                        continue
                    if _classify_source_from_path(gp) == source:
                        pairs.append((raw, 1))
        # Negatives: every OTHER path in the q's candidate list that was
        # NOT marked as golden for that query, IF the path belongs to
        # this source. This is key — a top-ranked whatsapp path on a
        # vault-intended query is a good negative for the WA model.
        for i, p in enumerate(q_paths):
            if p in golden_paths:
                continue
            if i >= len(q_scores):
                continue
            if _classify_source_from_path(p) != source:
                continue
            try:
                raw = float(q_scores[i])
            except (TypeError, ValueError):
                continue
            pairs.append((raw, 0))
    return pairs


def _classify_source_from_path(path: str) -> str:
    """Map a path to its source bucket. Best-effort — conservative on unknowns."""
    if not path:
        return "vault"
    if "://" in path:
        scheme = path.split("://", 1)[0].strip().lower()
        # Normalize scheme aliases.
        if scheme in ("whatsapp", "whats_app", "wa"):
            return "whatsapp"
        # `gdrive://` es el scheme histórico del ingester de Drive
        # (gdrive://file/<id>); `_CALIBRATION_SOURCES` usa "drive".
        if scheme == "gdrive":
            return "drive"
        if scheme in _CALIBRATION_SOURCES:
            return scheme
        return "vault"
    if path.startswith("99-obsidian/99-AI/external-ingest/WhatsApp/"):
        return "whatsapp"
    return "vault"


# Quick Win #4 (2026-04-29): umbral minimo de pares de FEEDBACK REAL
# por debajo del cual fallback-eamos a synthetic. Igual al
# `min_pairs_per_source` del CLI (default 20). Conservador a proposito:
# 19 reales + 50 sinteticos es peor que 19 reales puros (el cosine
# bge-m3 introduce un sesgo que la calibracion isotonic absorbe). Solo
# disparamos fallback cuando el real signal es claramente insuficiente.
SYNTH_FALLBACK_THRESHOLD = 20


def _gather_synthetic_calibration_pairs(
    conn, source: str,
) -> list[tuple[float, int]]:
    """Pull (raw_score, label) pairs from synthetic queries + negatives.

    Quick Win #4 fallback cuando el feedback real para una source no
    alcanza `SYNTH_FALLBACK_THRESHOLD`. Las "fuentes" de raw score son
    los cosines del bge-m3 embedding (no el cross-encoder score que usa
    el feedback real), por lo que la calibracion resultante NO es
    directamente comparable — el row se persiste con
    `model_version='isotonic-v1-synth'` para que el lector (ej.
    `calibrate_score`) sepa que cosa absorbio.

    Positives: cada synthetic query tiene un `positive_path`. El cosine
    query->positive lo captura `mine_hard_negatives_for_synthetic` en
    la column `cosine_to_positive` de `rag_synthetic_negatives` (mismo
    valor para todas las rows del mismo synth_id). DISTINCT para no
    contar el mismo positive N veces.

    Negatives: cada row de `rag_synthetic_negatives` con `neg_path`
    matcheando la source contribuye un par (cosine_to_query, 0).

    Args:
        conn: connection a `telemetry.db`.
        source: source target (vault/whatsapp/gmail/...).

    Returns:
        Lista de pares (raw_score, label) — vacia si la tabla no existe
        o no hay datos para la source.
    """
    pairs: list[tuple[float, int]] = []

    # Positives: distinct synth_id con cosine_to_positive populated.
    try:
        pos_rows = conn.execute(
            "SELECT DISTINCT synthetic_query_id, cosine_to_positive, "
            "       positive_path "
            "FROM rag_synthetic_negatives "
            "WHERE cosine_to_positive IS NOT NULL"
        ).fetchall()
    except Exception:
        pos_rows = []
    for _synth_id, cos_pos, positive_path in pos_rows:
        if cos_pos is None:
            continue
        try:
            score = float(cos_pos)
        except (TypeError, ValueError):
            continue
        if _classify_source_from_path(positive_path) == source:
            pairs.append((score, 1))

    # Negatives: every hard negative row whose neg_path matches source.
    try:
        neg_rows = conn.execute(
            "SELECT cosine_to_query, neg_path FROM rag_synthetic_negatives"
        ).fetchall()
    except Exception:
        neg_rows = []
    for cos_q, neg_path in neg_rows:
        if cos_q is None:
            continue
        try:
            score = float(cos_q)
        except (TypeError, ValueError):
            continue
        if _classify_source_from_path(neg_path) == source:
            pairs.append((score, 0))

    return pairs


def train_calibration(
    *,
    since_days: int = 90,
    min_pairs_per_source: int = 20,
    dry_run: bool = False,
    use_synthetic_fallback: bool = True,
) -> dict:
    """Train isotonic regression per-source and persist to rag_score_calibration.

    Returns per-source stats:
      {sources: {src: {status, n_pairs, n_pos, n_neg, knots,
                       used_synthetic, n_synth_pairs}, ...},
       total_pairs, trained_sources}

    Never raises (wraps everything defensively). dry_run=True computes
    knots but skips the DB write.

    Quick Win #4 (2026-04-29): si `use_synthetic_fallback=True` (default)
    y una source tiene < SYNTH_FALLBACK_THRESHOLD pares de feedback real,
    suplementa con pares sinteticos derivados de
    `rag_synthetic_queries`/`rag_synthetic_negatives` (cosine bge-m3 como
    proxy de raw_score). El row de calibracion resultante se persiste
    con `model_version='isotonic-v1-synth'` para distinguirlo del
    `isotonic-v1` puro de feedback real.
    """
    from datetime import datetime  # noqa: PLC0415

    from rag import _ragvec_state_conn  # noqa: PLC0415

    result: dict = {
        "sources": {},
        "total_pairs": 0,
        "trained_sources": 0,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with _ragvec_state_conn() as conn:
            for src in _CALIBRATION_SOURCES:
                real_pairs = _gather_calibration_pairs(
                    conn, src, since_days=since_days,
                )
                pairs = list(real_pairs)
                synth_pairs: list[tuple[float, int]] = []
                used_synthetic = False
                # Quick Win #4: synthetic fallback. Solo si feedback real
                # cae por debajo del umbral — preferimos siempre real
                # signal cuando haya. El threshold es el mismo que el
                # `min_pairs_per_source` del CLI (default 20).
                if (
                    use_synthetic_fallback
                    and len(real_pairs) < SYNTH_FALLBACK_THRESHOLD
                ):
                    try:
                        synth_pairs = _gather_synthetic_calibration_pairs(
                            conn, src,
                        )
                    except Exception:
                        synth_pairs = []
                    if synth_pairs:
                        pairs.extend(synth_pairs)
                        used_synthetic = True
                n_pos = sum(1 for _, y in pairs if y == 1)
                n_neg = len(pairs) - n_pos
                n_real_pos = sum(1 for _, y in real_pairs if y == 1)
                n_real_neg = len(real_pairs) - n_real_pos
                entry = {
                    "status": "skipped",
                    "n_pairs": len(pairs),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    # Subset de los counts: cuantos vinieron de synthetic
                    # vs feedback real. Util para diagnostics — el
                    # dashboard de calibration muestra el split.
                    "n_real_pos": n_real_pos,
                    "n_real_neg": n_real_neg,
                    "n_synth_pairs": len(synth_pairs),
                    "used_synthetic": used_synthetic,
                }
                result["total_pairs"] += len(pairs)
                if len(pairs) < min_pairs_per_source or n_pos == 0 or n_neg == 0:
                    entry["status"] = (
                        "insufficient" if len(pairs) < min_pairs_per_source
                        else "no-positive" if n_pos == 0
                        else "no-negative"
                    )
                    result["sources"][src] = entry
                    continue
                fit = _fit_isotonic_from_pairs(pairs)
                if fit is None:
                    entry["status"] = "fit-failed"
                    result["sources"][src] = entry
                    continue
                raw_k, cal_k = fit
                entry["status"] = "trained"
                entry["knots"] = len(raw_k)
                entry["raw_range"] = [raw_k[0], raw_k[-1]]
                entry["cal_range"] = [cal_k[0], cal_k[-1]]
                result["sources"][src] = entry
                result["trained_sources"] += 1
                if dry_run:
                    continue
                # Quick Win #4: model_version refleja si la calibracion
                # se baso en feedback real puro (`isotonic-v1`) o si
                # absorbio pares sinteticos como fallback
                # (`isotonic-v1-synth`). Lectores como `calibrate_score`
                # NO discriminan hoy, pero el flag esta disponible para
                # heuristicas futuras (ej. mostrar warning en el dashboard
                # si una source critica esta siendo calibrada con synth).
                model_version = (
                    "isotonic-v1-synth" if used_synthetic else "isotonic-v1"
                )
                try:
                    conn.execute(
                        "INSERT INTO rag_score_calibration "
                        "(source, raw_knots_json, cal_knots_json, "
                        " n_pos, n_neg, trained_at, model_version) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(source) DO UPDATE SET "
                        " raw_knots_json=excluded.raw_knots_json, "
                        " cal_knots_json=excluded.cal_knots_json, "
                        " n_pos=excluded.n_pos, "
                        " n_neg=excluded.n_neg, "
                        " trained_at=excluded.trained_at, "
                        " model_version=excluded.model_version",
                        (
                            src,
                            json.dumps(raw_k),
                            json.dumps(cal_k),
                            n_pos, n_neg,
                            result["trained_at"],
                            model_version,
                        ),
                    )
                except Exception as exc:
                    entry["status"] = f"persist-failed: {exc!r}"
    except Exception as exc:
        result["error"] = repr(exc)
    if not dry_run and result["trained_sources"] > 0:
        _reset_calibration_cache()
    return result


@click.command("calibrate")
@click.option("--since", default=90, show_default=True,
              help="Ventana en días de feedback a usar para entrenar.")
@click.option("--min-pairs", "min_pairs", default=20, show_default=True,
              help="Mínimo de pares (pos+neg) para entrenar una source.")
@click.option("--dry-run", is_flag=True,
              help="Calcular los knots pero no persistir.")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON machine-readable (para launchd logs).")
def calibrate_cli(since: int, min_pairs: int, dry_run: bool, as_json: bool):
    """Entrenar calibración de scores per-source desde rag_feedback.

    Por cada source (vault, whatsapp, calendar, gmail, reminders, safari,
    contacts, calls) fittea una isotonic regression sobre los pares
    (raw_score, label) extraídos de los últimos --since días de feedback.
    Persiste a rag_score_calibration; lo aplica retrieve() al próximo call
    cuando RAG_SCORE_CALIBRATION=1 está activo.

    Sin --apply implícito: si no pasás --dry-run, la escritura ocurre.
    """
    from rag import console  # noqa: PLC0415

    result = train_calibration(
        since_days=since, min_pairs_per_source=min_pairs, dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(result, default=str))
        return
    console.print()
    mode = " [yellow](dry-run)[/yellow]" if dry_run else ""
    console.print(f"[bold]Calibration training{mode}[/bold]")
    console.print(f"  Window: últimos {since}d  "
                  f"| min pairs: {min_pairs}  "
                  f"| total pairs: {result['total_pairs']}")
    console.print()
    for src, entry in result["sources"].items():
        status = entry["status"]
        if status == "trained":
            color = "green"
            detail = (f"{entry['n_pos']}+ / {entry['n_neg']}− → "
                      f"{entry.get('knots', '?')} knots  "
                      f"raw[{entry['raw_range'][0]:.3f}..{entry['raw_range'][1]:.3f}]")
        elif status.startswith("persist-failed"):
            color = "red"
            detail = status
        else:
            color = "dim"
            detail = f"{entry['n_pos']}+ / {entry['n_neg']}− ({status})"
        console.print(f"  [{color}]{src:12s}[/{color}]  {detail}")
    console.print()
    console.print(f"[bold]Trained: {result['trained_sources']} / "
                  f"{len(_CALIBRATION_SOURCES)} sources[/bold]")
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
    if not _SCORE_CALIBRATION_ENABLED and result["trained_sources"] > 0:
        console.print()
        console.print("[yellow]⚠[/yellow]  RAG_SCORE_CALIBRATION=0 — los modelos "
                      "quedaron guardados pero NO se aplican en retrieve().")
        console.print("   Activar con: [cyan]export RAG_SCORE_CALIBRATION=1[/cyan]")
    console.print()
