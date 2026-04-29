"""Backend del dashboard de aprendizaje (`/learning`).

Una función pública por sección — todas reciben `days: int` y devuelven un
``dict`` con shape estable que el frontend consume sin checks defensivos.
Cada función envuelve sus reads con ``_sql_read_with_retry`` y ante cualquier
error retorna un shape "vacío" pero válido (`insufficient: True` en cada
serie cuando hay menos de 5 puntos). El JS del dashboard nunca recibe
``null`` en una key esperada — siempre listas vacías o dicts con ceros.

Tablas leídas:
    rag_eval_runs, rag_tune, rag_queries, rag_feedback, rag_behavior,
    rag_learned_paraphrases, rag_response_cache, rag_anticipate_candidates,
    rag_routing_decisions, rag_routing_rules, rag_whisper_vocab,
    rag_audio_corrections, rag_audio_transcripts, rag_entities,
    rag_entity_mentions, rag_contradictions, rag_filing_log, rag_surface_log,
    rag_archive_log, rag_score_calibration.

Files leídos (read-only):
    ~/.local/share/obsidian-rag/ranker.json (current weights + baseline en
    .metadata.baseline; NO existe ranker.json.baseline separado).
    ~/.local/share/obsidian-rag/anticipate_weights.json (puede no existir).

Nota sobre timestamps: algunas tablas (rag_queries, rag_feedback,
rag_behavior, rag_eval_runs, rag_tune, rag_anticipate_candidates,
rag_contradictions, rag_filing_log, rag_surface_log, rag_archive_log)
usan ``ts TEXT`` ISO8601. Otras (rag_routing_decisions, rag_audio_*,
rag_entities, rag_entity_mentions, rag_whisper_vocab) usan ``ts REAL/INTEGER``
UNIX. Las funciones que mezclan ambas normalizan a ISO8601 antes de
devolver al frontend.

Performance budget: <100ms warm por función. Todas las agregaciones se
hacen en SQL (GROUP BY DATE(ts), COUNT, SUM, AVG) — no se rehidratan rows
para sumar en Python.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

# Threshold universal para badge "Datos insuficientes para tendencia".
_INSUFFICIENT_THRESHOLD = 5

# Paths de configs leídas (read-only). NO se asume que existan.
_DATA_DIR = Path.home() / ".local/share/obsidian-rag"
_RANKER_PATH = _DATA_DIR / "ranker.json"
_ANTICIPATE_WEIGHTS_PATH = _DATA_DIR / "anticipate_weights.json"

# Orden canonical de weight keys del ranker. El frontend pinta una línea
# por key en este orden — si alguna key no existe en una row de rag_tune
# o en ranker.json, se rellena con None (el chart la skipea).
_RANKER_WEIGHT_KEYS = (
    "recency_always",
    "recency_cue",
    "tag_literal",
    "title_match",
    "graph_pagerank",
    "click_prior",
    "click_prior_folder",
    "click_prior_hour",
    "dwell_score",
    "feedback_pos",
    "feedback_neg",
    "feedback_match_floor",
)

# Buckets de routing — orden estable para que el chart pinte siempre las
# mismas 5 líneas. Buckets nuevos caen al "other" implícito (no se trackean).
_ROUTING_BUCKETS = ("task", "note", "transcribe", "respond", "schedule")

# Folders del PARA — orden estable. Los demás folders del filing log caen
# en "other" implícito (no se trackean en el over-time chart).
_PARA_FOLDERS = ("01-Projects", "02-Areas", "03-Resources", "04-Archive", "00-Inbox")

# Entity types — el schema actual usa lowercase ("person", "organization",
# "event", "location"). El frontend espera UPPERCASE: normalizamos al output.
# Tipos no contemplados ("event", anything else) caen a "OTHER" — por eso
# no agregamos "EVENT" al map: queremos que el getter use el default.
_ENTITY_TYPE_MAP = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "LOCATION",
}
_ENTITY_TYPES = ("PERSON", "ORG", "LOCATION", "OTHER")


def _insufficient(n: int) -> bool:
    return n < _INSUFFICIENT_THRESHOLD


def _empty_series(extra: dict | None = None) -> dict:
    base = {"series": [], "n_samples": 0, "insufficient": True}
    if extra:
        base.update(extra)
    return base


def _date_range(days: int) -> list[str]:
    """Lista de fechas YYYY-MM-DD desde days-1 atrás hasta hoy (inclusive)."""
    today = date.today()
    return [(today - timedelta(days=days - 1 - i)).isoformat() for i in range(days)]


def _cutoff_iso(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")


def _cutoff_unix(days: int) -> float:
    return (datetime.now() - timedelta(days=days)).timestamp()


def _safe_json(raw: str | bytes | None) -> dict | list | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _read_ranker_files() -> tuple[dict | None, dict | None]:
    """Devuelve (current_weights_dict, baseline_weights_dict). Cualquiera puede
    ser None si el archivo no existe / está corrupto."""
    current = None
    baseline = None
    try:
        if _RANKER_PATH.exists():
            data = json.loads(_RANKER_PATH.read_text())
            if isinstance(data, dict):
                w = data.get("weights")
                if isinstance(w, dict):
                    current = w
                meta = data.get("metadata") or {}
                if isinstance(meta, dict):
                    b = meta.get("baseline")
                    if isinstance(b, dict):
                        baseline = b
    except Exception:
        pass
    return current, baseline


def _read_anticipate_weights() -> dict:
    try:
        if _ANTICIPATE_WEIGHTS_PATH.exists():
            data = json.loads(_ANTICIPATE_WEIGHTS_PATH.read_text())
            if isinstance(data, dict):
                # El archivo puede tener estructura {kind: weight} o
                # {"weights": {kind: weight}, ...}. Aceptar ambos.
                w = data.get("weights") if isinstance(data.get("weights"), dict) else data
                return {str(k): float(v) for k, v in w.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. KPIs
# ─────────────────────────────────────────────────────────────────────────────

def kpis(days: int = 30) -> dict:
    """Strip de KPIs hero del dashboard. 8 métricas con valor + delta vs
    la ventana anterior del mismo tamaño + flag ``insufficient`` cuando
    hay menos de 5 samples. Cada métrica tira de una tabla distinta — el
    error en una NO contamina las otras (cada `_kpi_*` helper hace su
    propio try/except y devuelve el shape vacío)."""
    return {
        "eval_hit5_singles": _kpi_eval_hit5(days, column="singles_hit5"),
        "eval_hit5_chains": _kpi_eval_hit5(days, column="chains_hit5"),
        "feedback_total": _kpi_feedback_total(days),
        "behavior_per_query": _kpi_behavior_per_query(days),
        "cache_hit_rate": _kpi_cache_hit_rate(days),
        "paraphrases_count": _kpi_paraphrases_count(days),
        "entities_count": _kpi_entities_count(days),
        "contradictions_resolved_pct": _kpi_contradictions_resolved_pct(days),
    }


def _kpi_empty(extra_keys: tuple[str, ...] = ("delta_30d",)) -> dict:
    base: dict = {"value": 0, "n_samples": 0, "insufficient": True}
    for k in extra_keys:
        base[k] = 0
    return base


def _kpi_eval_hit5(days: int, *, column: str) -> dict:
    """Promedio de singles_hit5/chains_hit5 en la ventana actual y delta vs
    la ventana anterior del mismo tamaño."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            row_now = conn.execute(
                f"SELECT AVG({column}) AS v, COUNT({column}) AS n "
                "FROM rag_eval_runs WHERE ts >= ?",
                (cutoff_now,),
            ).fetchone()
            row_prev = conn.execute(
                f"SELECT AVG({column}) AS v, COUNT({column}) AS n "
                "FROM rag_eval_runs WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()
        v_now = float(row_now[0]) if row_now and row_now[0] is not None else 0.0
        n_now = int(row_now[1]) if row_now and row_now[1] is not None else 0
        v_prev = float(row_prev[0]) if row_prev and row_prev[0] is not None else 0.0
        n_prev = int(row_prev[1]) if row_prev and row_prev[1] is not None else 0
        # Sin baseline confiable (ventana previa < threshold) NO mostramos
        # delta — el frontend interpreta None como "primera medición" y
        # evita el falso "+97% subió" cuando antes simplemente no medíamos.
        delta = (
            round(v_now - v_prev, 4)
            if n_now and n_prev >= _INSUFFICIENT_THRESHOLD
            else None
        )
        return {
            "value": round(v_now, 4),
            "delta_30d": delta,
            "n_samples": n_now,
            "insufficient": _insufficient(n_now),
        }

    return _sql_read_with_retry(_do, "learning_kpi_eval_hit5_failed", default=_kpi_empty())


def _kpi_feedback_total(days: int) -> dict:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            n_now = conn.execute(
                "SELECT COUNT(*) FROM rag_feedback WHERE ts >= ?", (cutoff_now,),
            ).fetchone()[0] or 0
            n_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_feedback WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
        return {
            "value": int(n_now),
            "delta_30d": int(n_now - n_prev),
            "n_samples": int(n_now),
            "insufficient": _insufficient(int(n_now)),
        }

    return _sql_read_with_retry(_do, "learning_kpi_feedback_total_failed", default=_kpi_empty())


def _kpi_behavior_per_query(days: int) -> dict:
    """Eventos rag_behavior por query en la ventana — proxy de "señales
    implícitas por interacción". Mismo cálculo para current/previous."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            n_b_now = conn.execute(
                "SELECT COUNT(*) FROM rag_behavior WHERE ts >= ?", (cutoff_now,),
            ).fetchone()[0] or 0
            n_q_now = conn.execute(
                "SELECT COUNT(*) FROM rag_queries WHERE ts >= ?", (cutoff_now,),
            ).fetchone()[0] or 0
            n_b_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_behavior WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
            n_q_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_queries WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
        ratio_now = (n_b_now / n_q_now) if n_q_now else 0.0
        ratio_prev = (n_b_prev / n_q_prev) if n_q_prev else 0.0
        # Delta = None cuando la ventana previa no tiene queries suficientes
        # — evita "+21 vs 30d" cuando hace 60 días no se medía nada.
        delta = (
            round(ratio_now - ratio_prev, 3)
            if int(n_q_now) and int(n_q_prev) >= _INSUFFICIENT_THRESHOLD
            else None
        )
        return {
            "value": round(ratio_now, 3),
            "delta_30d": delta,
            "n_samples": int(n_q_now),
            "insufficient": _insufficient(int(n_q_now)),
        }

    return _sql_read_with_retry(_do, "learning_kpi_behavior_per_query_failed", default=_kpi_empty())


def _kpi_cache_hit_rate(days: int) -> dict:
    """Hit rate del response cache. Lee `extra_json` de rag_queries — el flag
    ``cache_hit: true|false`` se setea en cada query (también las que no usan
    cache, donde queda false). Si no hay `cache_hit` key en la row, no cuenta
    para el denominador."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            # cache_hit:true / cache_hit:false como sub-string match — más
            # rápido que parsear JSON en Python row a row. Falsos positivos
            # casi nulos (la key es JSON-escaped igual en todos los writers).
            n_hit_now = conn.execute(
                "SELECT COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND extra_json LIKE '%\"cache_hit\": true%'",
                (cutoff_now,),
            ).fetchone()[0] or 0
            n_total_now = conn.execute(
                "SELECT COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND extra_json LIKE '%\"cache_hit\":%'",
                (cutoff_now,),
            ).fetchone()[0] or 0
            n_hit_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND ts < ? AND extra_json LIKE '%\"cache_hit\": true%'",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
            n_total_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND ts < ? AND extra_json LIKE '%\"cache_hit\":%'",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
        rate_now = (n_hit_now / n_total_now) if n_total_now else 0.0
        rate_prev = (n_hit_prev / n_total_prev) if n_total_prev else 0.0
        # Sin baseline confiable → None (frontend muestra "primera medición").
        delta = (
            round(rate_now - rate_prev, 4)
            if int(n_total_now) and int(n_total_prev) >= _INSUFFICIENT_THRESHOLD
            else None
        )
        return {
            "value": round(rate_now, 4),
            "delta_30d": delta,
            "n_samples": int(n_total_now),
            "insufficient": _insufficient(int(n_total_now)),
        }

    return _sql_read_with_retry(_do, "learning_kpi_cache_hit_rate_failed", default=_kpi_empty())


def _kpi_paraphrases_count(days: int) -> dict:
    """Conteo total de paraphrases creadas en la ventana — el delta es vs
    la ventana anterior."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            n_now = conn.execute(
                "SELECT COUNT(*) FROM rag_learned_paraphrases WHERE created_ts >= ?",
                (cutoff_now,),
            ).fetchone()[0] or 0
            n_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_learned_paraphrases "
                "WHERE created_ts >= ? AND created_ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
        return {
            "value": int(n_now),
            "delta_30d": int(n_now - n_prev),
            "n_samples": int(n_now),
            "insufficient": _insufficient(int(n_now)),
        }

    return _sql_read_with_retry(_do, "learning_kpi_paraphrases_count_failed", default=_kpi_empty())


def _kpi_entities_count(days: int) -> dict:
    """Snapshot total de entities — first_seen_ts es UNIX. El delta cuenta
    cuántas se descubrieron en la ventana actual."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_unix(days)
    cutoff_prev = _cutoff_unix(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            n_total = conn.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0] or 0
            n_now = conn.execute(
                "SELECT COUNT(*) FROM rag_entities WHERE first_seen_ts >= ?",
                (cutoff_now,),
            ).fetchone()[0] or 0
            n_prev = conn.execute(
                "SELECT COUNT(*) FROM rag_entities "
                "WHERE first_seen_ts >= ? AND first_seen_ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()[0] or 0
        return {
            "value": int(n_total),
            "delta_30d": int(n_now - n_prev),
            "n_samples": int(n_total),
            "insufficient": _insufficient(int(n_total)),
        }

    return _sql_read_with_retry(_do, "learning_kpi_entities_count_failed", default=_kpi_empty())


def _kpi_contradictions_resolved_pct(days: int) -> dict:
    """Proxy: una contradicción se considera "no resuelta" si su row tiene
    ``skipped IS NULL OR skipped = ''``. Lo demás (skipped tiene valor) se
    considera resuelta/dismissed/etc. Si la tabla está vacía, devuelve
    insufficient."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff_now = _cutoff_iso(days)
    cutoff_prev = _cutoff_iso(days * 2)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            row_now = conn.execute(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN skipped IS NOT NULL AND skipped != '' THEN 1 ELSE 0 END) AS resolved "
                "FROM rag_contradictions WHERE ts >= ?",
                (cutoff_now,),
            ).fetchone()
            row_prev = conn.execute(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN skipped IS NOT NULL AND skipped != '' THEN 1 ELSE 0 END) AS resolved "
                "FROM rag_contradictions WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()
        total_now = int(row_now[0] or 0) if row_now else 0
        resolved_now = int(row_now[1] or 0) if row_now else 0
        total_prev = int(row_prev[0] or 0) if row_prev else 0
        resolved_prev = int(row_prev[1] or 0) if row_prev else 0
        pct_now = (resolved_now / total_now) if total_now else 0.0
        pct_prev = (resolved_prev / total_prev) if total_prev else 0.0
        delta = (
            round(pct_now - pct_prev, 4)
            if total_now and total_prev >= _INSUFFICIENT_THRESHOLD
            else None
        )
        return {
            "value": round(pct_now, 4),
            "delta_30d": delta,
            "n_samples": total_now,
            "insufficient": _insufficient(total_now),
        }

    return _sql_read_with_retry(
        _do, "learning_kpi_contradictions_resolved_pct_failed", default=_kpi_empty(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Quality (rag_eval_runs + rag_tune + rag_queries)
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_quality(days: int = 30) -> dict:
    return {
        "eval_over_time": _retrieval_eval_over_time(days),
        "tune_deltas": _retrieval_tune_deltas(days),
        "latency_vs_score": _retrieval_latency_vs_score(),
    }


def _retrieval_eval_over_time(days: int) -> dict:
    """Una row por run de eval — no agregamos por día porque el run típico
    es 1/día y ya tenés `singles_hit5`, `chains_hit5`, etc. directos."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT ts, singles_hit5, chains_hit5, singles_mrr, chains_mrr, "
                "       singles_n, chains_n "
                "FROM rag_eval_runs WHERE ts >= ? ORDER BY ts ASC",
                (cutoff,),
            ).fetchall()
        series = []
        for ts, s_hit5, c_hit5, s_mrr, c_mrr, s_n, c_n in rows:
            series.append({
                "ts": ts,
                "hit5_singles": float(s_hit5) if s_hit5 is not None else None,
                "hit5_chains": float(c_hit5) if c_hit5 is not None else None,
                "mrr_singles": float(s_mrr) if s_mrr is not None else None,
                "mrr_chains": float(c_mrr) if c_mrr is not None else None,
                "n_singles": int(s_n) if s_n is not None else 0,
                "n_chains": int(c_n) if c_n is not None else 0,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_retrieval_eval_over_time_failed", default=_empty_series(),
    )


def _retrieval_tune_deltas(days: int) -> dict:
    """Bar chart: % de mejora vs línea base por iteración de tune.

    Filtra runs con `delta` NULL o exactamente 0 — esos son tunes que NO
    encontraron mejora (espacio de búsqueda saturado, ej. 378 de 392 runs en
    el snapshot del 2026-04-29). Mostrar 392 barras de altura 0 hace que la
    chart parezca vacía: el usuario no ve los 14 ajustes que realmente
    movieron la aguja. Filtramos al backend y agregamos `n_total` /
    `n_zero_filtered` para que la UI pueda mencionar el contexto si quiere.
    """
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT ts, delta, rolled_back, n_cases "
                "FROM rag_tune WHERE ts >= ? ORDER BY ts ASC",
                (cutoff,),
            ).fetchall()
        series = []
        n_total = len(rows)
        n_zero_filtered = 0
        for ts, delta, rolled_back, n_cases in rows:
            if delta is None:
                n_zero_filtered += 1
                continue
            try:
                delta_f = float(delta)
            except Exception:
                n_zero_filtered += 1
                continue
            if delta_f == 0.0:
                n_zero_filtered += 1
                continue
            delta_pct = delta_f * 100.0
            series.append({
                "ts": ts,
                "delta_pct": round(delta_pct, 4),
                "rolled_back": bool(rolled_back) if rolled_back is not None else False,
                "n_cases": int(n_cases) if n_cases is not None else 0,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "n_total": n_total,
            "n_zero_filtered": n_zero_filtered,
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_retrieval_tune_deltas_failed", default=_empty_series(),
    )


def _retrieval_latency_vs_score() -> dict:
    """Scatter plot: 1 punto por query reciente. Cap a 1000 más recientes
    para no inflar el JSON (~30KB worst case).

    IMPORTANTE: filtramos NaN/inf con `math.isfinite` — sqlite a veces guarda
    `-inf` en `top_score` (visto en producción 2026-04-29) y eso se serializa
    como literal `-Infinity` en JSON, que NO es JSON válido. El endpoint
    REST `/api/learning` lo limpia FastAPI, pero el SSE
    `/api/learning/stream` usa `json.dumps` directo y rompía el `JSON.parse`
    del cliente cada 30s ("No number after minus sign at position 83609"),
    cortando el live update. Filtramos al origen.
    """
    import math
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT t_retrieve, top_score FROM rag_queries "
                "WHERE t_retrieve IS NOT NULL AND top_score IS NOT NULL "
                "ORDER BY id DESC LIMIT 1000"
            ).fetchall()
        points = []
        for t, s in rows:
            if not isinstance(t, (int, float)) or not isinstance(s, (int, float)):
                continue
            t_f = float(t)
            s_f = float(s)
            if not (math.isfinite(t_f) and math.isfinite(s_f)):
                continue
            points.append({"t_retrieve": t_f, "top_score": s_f})
        return {
            "points": points,
            "n_samples": len(points),
            "insufficient": _insufficient(len(points)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_retrieval_latency_vs_score_failed",
        default={"points": [], "n_samples": 0, "insufficient": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ranker Weights (rag_tune.best_json + ranker.json)
# ─────────────────────────────────────────────────────────────────────────────

def ranker_weights(days: int = 30) -> dict:
    """Evolución de los pesos del ranker. Cada row de rag_tune con
    ``best_json.weights`` aporta un punto en la serie. El frontend pinta
    una línea por key del weight vector — usamos el orden canonical de
    ``_RANKER_WEIGHT_KEYS``. Las rows que NO tienen `weights` (modelos
    LightGBM puros, que solo guardan feature_importance_gain) se skipean.

    Baseline + current se leen de ``ranker.json`` (file en disco), no de
    rag_tune — la baseline vive en `metadata.baseline` del archivo y el
    "current" es el `weights` top-level."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT ts, best_json FROM rag_tune "
                "WHERE ts >= ? AND best_json IS NOT NULL ORDER BY ts ASC",
                (cutoff,),
            ).fetchall()
        series = []
        for ts, best_json in rows:
            payload = _safe_json(best_json)
            if not isinstance(payload, dict):
                continue
            weights = payload.get("weights")
            if not isinstance(weights, dict):
                continue
            values = [
                float(weights[k]) if isinstance(weights.get(k), (int, float)) else None
                for k in _RANKER_WEIGHT_KEYS
            ]
            series.append({"ts": ts, "values": values})
        return series

    series = _sql_read_with_retry(_do, "learning_ranker_weights_failed", default=[]) or []

    current_dict, baseline_dict = _read_ranker_files()
    current = (
        [
            float(current_dict[k]) if isinstance(current_dict.get(k), (int, float)) else None
            for k in _RANKER_WEIGHT_KEYS
        ]
        if current_dict
        else None
    )
    baseline = (
        [
            float(baseline_dict[k]) if isinstance(baseline_dict.get(k), (int, float)) else None
            for k in _RANKER_WEIGHT_KEYS
        ]
        if baseline_dict
        else None
    )

    return {
        "weight_keys": list(_RANKER_WEIGHT_KEYS),
        "series": series,
        "baseline": baseline,
        "current": current,
        "n_samples": len(series),
        "insufficient": _insufficient(len(series)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Score Calibration (rag_score_calibration)
# ─────────────────────────────────────────────────────────────────────────────

def score_calibration() -> dict:
    """rag_score_calibration es snapshot — hay 1 row por source. No hay
    historia de re-fits, así que no hay over-time. La tabla tiene un PK
    sobre `source` así que cada source aparece como mucho una vez."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> list[dict]:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, raw_knots_json, cal_knots_json, n_pos, n_neg, trained_at "
                "FROM rag_score_calibration ORDER BY source"
            ).fetchall()
        out = []
        for source, raw_j, cal_j, n_pos, n_neg, trained_at in rows:
            raw = _safe_json(raw_j)
            cal = _safe_json(cal_j)
            n_pos_i = int(n_pos or 0)
            n_neg_i = int(n_neg or 0)
            total = n_pos_i + n_neg_i
            out.append({
                "source": source,
                "raw_knots": raw if isinstance(raw, list) else [],
                "cal_knots": cal if isinstance(cal, list) else [],
                "trained_at": trained_at,
                "n_pos": n_pos_i,
                "n_neg": n_neg_i,
                "insufficient": _insufficient(total),
            })
        return out

    curves = _sql_read_with_retry(_do, "learning_score_calibration_failed", default=[]) or []
    return {"curves": curves}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feedback Explícito (rag_feedback)
# ─────────────────────────────────────────────────────────────────────────────

def feedback_explicit(days: int = 30) -> dict:
    return {
        "thumbs_over_time": _feedback_thumbs_over_time(days),
        "corrective_paths_cumulative": _feedback_corrective_cumulative(days),
        "by_scope": _feedback_by_scope(days),
    }


def _feedback_thumbs_over_time(days: int) -> dict:
    """Stack chart por día: positive (rating=1), negative (rating=-1),
    corrective (rating cualquiera con `corrective_path` en extra_json).
    El total positive+negative incluye los corrective — corrective es una
    sub-categoría, no excluyente."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day_pos: dict[str, int] = {d: 0 for d in dates}
        per_day_neg: dict[str, int] = {d: 0 for d in dates}
        per_day_corr: dict[str, int] = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts) AS d, rating, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? GROUP BY d, rating",
                (cutoff,),
            ).fetchall()
            for d, rating, n in rows:
                if d not in per_day_pos:
                    continue
                if rating == 1:
                    per_day_pos[d] += int(n)
                elif rating == -1:
                    per_day_neg[d] += int(n)
            corr_rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? AND extra_json LIKE '%corrective_path%' "
                "GROUP BY d",
                (cutoff,),
            ).fetchall()
            for d, n in corr_rows:
                if d in per_day_corr:
                    per_day_corr[d] = int(n)
        series = [
            {
                "date": d,
                "positive": per_day_pos[d],
                "negative": per_day_neg[d],
                "corrective": per_day_corr[d],
            }
            for d in dates
        ]
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_feedback_thumbs_over_time_failed", default=_empty_series(),
    )


def _feedback_corrective_cumulative(days: int) -> dict:
    """Curva acumulada de corrective paths — por día acumulamos count desde
    el principio de la tabla hasta esa fecha (no solo dentro de la ventana,
    para que el chart muestre crecimiento real). Para no escanear todo el
    historial, calculamos el "carry" pre-ventana en una sola query."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            carry = conn.execute(
                "SELECT COUNT(*) FROM rag_feedback "
                "WHERE ts < ? AND extra_json LIKE '%corrective_path%'",
                (cutoff,),
            ).fetchone()[0] or 0
            rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? AND extra_json LIKE '%corrective_path%' "
                "GROUP BY d ORDER BY d ASC",
                (cutoff,),
            ).fetchall()
        per_day = {d: 0 for d in dates}
        for d, n in rows:
            if d in per_day:
                per_day[d] = int(n)
        running = int(carry)
        series = []
        for d in dates:
            running += per_day[d]
            series.append({"date": d, "cumulative": running})
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_feedback_corrective_cumulative_failed", default=_empty_series(),
    )


def _feedback_by_scope(days: int) -> dict:
    """Bucketing por origen × signo del feedback.

    El spec original asumía scope="answer"/"retrieval"/"both", pero el writer
    real nunca setea esos valores — los rows tienen scope="turn" (manual
    explícito vía thumbs/harvester) o scope=NULL (señal implícita derivada
    del comportamiento de sesión, ej. session_outcome_win/loss). Auditado el
    2026-04-29: 100% de los 556 rows caían en "unknown" → la dona quedaba
    inservible.

    Re-categorizamos por **fuente × signo**, que es lo que realmente nos
    interesa al mirar la dona: ¿el feedback vino del usuario explícito o
    inferido de la sesión, y fue positivo o negativo?

        scope IN ("turn", "answer", "retrieval", "both") + rating>0  → manual_pos
        scope IN ("turn", "answer", "retrieval", "both") + rating<=0 → manual_neg
        scope IS NULL + rating>0                                     → implicit_pos
        scope IS NULL + rating<=0                                    → implicit_neg

    Las 4 keys están siempre presentes (default 0)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        out = {"manual_pos": 0, "manual_neg": 0, "implicit_pos": 0, "implicit_neg": 0}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT scope, rating, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? GROUP BY scope, rating",
                (cutoff,),
            ).fetchall()
        for scope, rating, n in rows:
            try:
                r = int(rating) if rating is not None else 0
            except Exception:
                r = 0
            is_manual = scope is not None  # turn / answer / retrieval / both → manual
            if is_manual:
                key = "manual_pos" if r > 0 else "manual_neg"
            else:
                key = "implicit_pos" if r > 0 else "implicit_neg"
            out[key] += int(n)
        return out

    return _sql_read_with_retry(
        _do,
        "learning_feedback_by_scope_failed",
        default={"manual_pos": 0, "manual_neg": 0, "implicit_pos": 0, "implicit_neg": 0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feedback Implícito (rag_feedback con extra_json.implicit_loss_source)
# ─────────────────────────────────────────────────────────────────────────────

def feedback_implicit(days: int = 30) -> dict:
    return {
        "by_source": _feedback_implicit_by_source(days),
        "implicit_signal_rate_over_time": _feedback_implicit_rate(days),
    }


_IMPLICIT_SOURCES = (
    "requery_detection",
    "session_outcome_loss",
    "session_outcome_win",
    "corrective_paths",
)


def _feedback_implicit_by_source(days: int) -> dict:
    """Cuenta por source dentro de extra_json.implicit_loss_source. Usamos
    LIKE con el valor literal — los sources están definidos en el código
    de RAG (no son free-text del user) así que el set es cerrado."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        out = {src: 0 for src in _IMPLICIT_SOURCES}
        out["other"] = 0
        with _ragvec_state_conn() as conn:
            for src in _IMPLICIT_SOURCES:
                n = conn.execute(
                    "SELECT COUNT(*) FROM rag_feedback "
                    "WHERE ts >= ? AND extra_json LIKE ?",
                    (cutoff, f'%"implicit_loss_source": "{src}"%'),
                ).fetchone()[0] or 0
                out[src] = int(n)
            n_total = conn.execute(
                "SELECT COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? AND extra_json LIKE '%implicit_loss_source%'",
                (cutoff,),
            ).fetchone()[0] or 0
        out["other"] = max(0, int(n_total) - sum(out[s] for s in _IMPLICIT_SOURCES))
        return out

    return _sql_read_with_retry(
        _do,
        "learning_feedback_implicit_by_source_failed",
        default={**{s: 0 for s in _IMPLICIT_SOURCES}, "other": 0},
    )


def _feedback_implicit_rate(days: int) -> dict:
    """Por día: count de feedback rows con `implicit_loss_source` / total
    de queries del día. Aproxima la tasa de "señal implícita por turn"."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day_impl: dict[str, int] = {d: 0 for d in dates}
        per_day_q: dict[str, int] = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            impl_rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? AND extra_json LIKE '%implicit_loss_source%' "
                "GROUP BY d",
                (cutoff,),
            ).fetchall()
            for d, n in impl_rows:
                if d in per_day_impl:
                    per_day_impl[d] = int(n)
            q_rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_queries "
                "WHERE ts >= ? GROUP BY d",
                (cutoff,),
            ).fetchall()
            for d, n in q_rows:
                if d in per_day_q:
                    per_day_q[d] = int(n)
        series = []
        for d in dates:
            n_q = per_day_q[d]
            n_impl = per_day_impl[d]
            pct = (n_impl / n_q) if n_q else 0.0
            series.append({
                "date": d,
                "implicit_pct": round(pct, 4),
                "n_turns": n_q,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_feedback_implicit_rate_failed", default=_empty_series(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Behavior (rag_behavior)
# ─────────────────────────────────────────────────────────────────────────────

_BEHAVIOR_SOURCES = ("web", "cli", "wa")


def behavior(days: int = 30) -> dict:
    return {
        "ctr_by_source_over_time": _behavior_ctr_over_time(days),
        "dwell_distribution": _behavior_dwell_distribution(days),
        "top_paths": _behavior_top_paths(days),
        "heatmap_dow_hour": _behavior_heatmap(days),
    }


def _behavior_ctr_over_time(days: int) -> dict:
    """CTR = open events / impression events, por día y source. Los sources
    canonical son web/cli/wa — `whatsapp` lo mapeamos a `wa`. Otros sources
    (brief, etc.) caen en "other" implícito (no se trackean)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        # Inicializar matriz vacía: para cada fecha, una list de len(sources).
        impressions: dict[str, dict[str, int]] = {d: {s: 0 for s in _BEHAVIOR_SOURCES} for d in dates}
        opens: dict[str, dict[str, int]] = {d: {s: 0 for s in _BEHAVIOR_SOURCES} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts) AS d, source, event, COUNT(*) "
                "FROM rag_behavior WHERE ts >= ? GROUP BY d, source, event",
                (cutoff,),
            ).fetchall()
        for d, src, event, n in rows:
            if d not in impressions:
                continue
            src_norm = "wa" if src == "whatsapp" else src
            if src_norm not in _BEHAVIOR_SOURCES:
                continue
            if event == "impression":
                impressions[d][src_norm] += int(n)
            elif event == "open":
                opens[d][src_norm] += int(n)
        series = []
        for d in dates:
            values = []
            for s in _BEHAVIOR_SOURCES:
                imp = impressions[d][s]
                op = opens[d][s]
                ctr = (op / imp) if imp else 0.0
                values.append(round(ctr, 4))
            series.append({"date": d, "values": values})
        return {
            "sources": list(_BEHAVIOR_SOURCES),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_behavior_ctr_over_time_failed",
        default={"sources": list(_BEHAVIOR_SOURCES), **_empty_series()},
    )


def _behavior_dwell_distribution(days: int) -> dict:
    """Buckets fijos de dwell time. Solo eventos con `dwell_s` no-null."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    bucket_defs = [
        ("0-5s", 0.0, 5.0),
        ("5-15s", 5.0, 15.0),
        ("15-60s", 15.0, 60.0),
        ("60-300s", 60.0, 300.0),
        ("300s+", 300.0, float("inf")),
    ]

    def _do() -> dict:
        counts = {label: 0 for label, _, _ in bucket_defs}
        # Single CASE-based aggregation — 1 SQL roundtrip vs 5 (consolidated
        # 2026-04-26 audit, cold path bajó de ~150ms a ~30ms warm).
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT "
                "  SUM(CASE WHEN dwell_s >= 0   AND dwell_s < 5    THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN dwell_s >= 5   AND dwell_s < 15   THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN dwell_s >= 15  AND dwell_s < 60   THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN dwell_s >= 60  AND dwell_s < 300  THEN 1 ELSE 0 END), "
                "  SUM(CASE WHEN dwell_s >= 300                    THEN 1 ELSE 0 END)  "
                "FROM rag_behavior WHERE ts >= ? AND dwell_s IS NOT NULL",
                (cutoff,),
            ).fetchone()
        if row:
            for (label, _, _), v in zip(bucket_defs, row):
                counts[label] = int(v or 0)
        total = sum(counts.values())
        return {
            "buckets": [{"label": label, "count": counts[label]} for label, _, _ in bucket_defs],
            "n_samples": total,
            "insufficient": _insufficient(total),
        }

    return _sql_read_with_retry(
        _do,
        "learning_behavior_dwell_distribution_failed",
        default={
            "buckets": [{"label": label, "count": 0} for label, _, _ in bucket_defs],
            "n_samples": 0,
            "insufficient": True,
        },
    )


def _behavior_top_paths(days: int) -> list[dict]:
    """Top 20 paths por cantidad de opens en la ventana. Avg dwell calculado
    sobre eventos con dwell_s no-null."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> list[dict]:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT path, "
                "       COUNT(*) FILTER (WHERE event = 'open') AS clicks, "
                "       AVG(dwell_s) AS avg_dwell "
                "FROM rag_behavior "
                "WHERE ts >= ? AND path IS NOT NULL "
                "GROUP BY path "
                "HAVING clicks > 0 "
                "ORDER BY clicks DESC LIMIT 20",
                (cutoff,),
            ).fetchall()
        return [
            {
                "path": p,
                "clicks": int(c),
                "avg_dwell_s": round(float(d), 2) if d is not None else 0.0,
            }
            for p, c, d in rows
        ]

    return _sql_read_with_retry(_do, "learning_behavior_top_paths_failed", default=[]) or []


def _behavior_heatmap(days: int) -> list[list[int]]:
    """Matrix 7x24 [dow][hour]. dow: 0=Sunday, 6=Saturday (sqlite strftime
    convention).
    """
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> list[list[int]]:
        matrix = [[0 for _ in range(24)] for _ in range(7)]
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT CAST(strftime('%w', ts) AS INTEGER) AS dow, "
                "       CAST(strftime('%H', ts) AS INTEGER) AS hour, "
                "       COUNT(*) "
                "FROM rag_behavior "
                "WHERE ts >= ? AND event = 'open' "
                "GROUP BY dow, hour",
                (cutoff,),
            ).fetchall()
        for dow, hour, n in rows:
            if dow is None or hour is None:
                continue
            try:
                d = int(dow)
                h = int(hour)
            except Exception:
                continue
            if 0 <= d < 7 and 0 <= h < 24:
                matrix[d][h] = int(n)
        return matrix

    return _sql_read_with_retry(
        _do,
        "learning_behavior_heatmap_failed",
        default=[[0 for _ in range(24)] for _ in range(7)],
    ) or [[0 for _ in range(24)] for _ in range(7)]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Query Learning (rag_learned_paraphrases + rag_response_cache + rag_queries)
# ─────────────────────────────────────────────────────────────────────────────

def query_learning(days: int = 30) -> dict:
    return {
        "paraphrases_count_over_time": _query_paraphrases_cumulative(days),
        "top_paraphrases": _query_top_paraphrases(),
        "cache_hit_rate_over_time": _query_cache_hit_rate_over_time(days),
    }


def _query_paraphrases_cumulative(days: int) -> dict:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            carry = conn.execute(
                "SELECT COUNT(*) FROM rag_learned_paraphrases WHERE created_ts < ?",
                (cutoff,),
            ).fetchone()[0] or 0
            rows = conn.execute(
                "SELECT DATE(created_ts) AS d, COUNT(*) FROM rag_learned_paraphrases "
                "WHERE created_ts >= ? GROUP BY d ORDER BY d ASC",
                (cutoff,),
            ).fetchall()
        per_day = {d: 0 for d in dates}
        for d, n in rows:
            if d in per_day:
                per_day[d] = int(n)
        running = int(carry)
        series = []
        for d in dates:
            running += per_day[d]
            series.append({"date": d, "cumulative": running})
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_query_paraphrases_cumulative_failed", default=_empty_series(),
    )


def _query_top_paraphrases() -> list[dict]:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> list[dict]:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT q_normalized, paraphrase, hit_count "
                "FROM rag_learned_paraphrases "
                "ORDER BY hit_count DESC LIMIT 20"
            ).fetchall()
        return [
            {
                "q_normalized": q or "",
                "paraphrase": p or "",
                "hit_count": int(h or 0),
            }
            for q, p, h in rows
        ]

    return _sql_read_with_retry(_do, "learning_query_top_paraphrases_failed", default=[]) or []


def _query_cache_hit_rate_over_time(days: int) -> dict:
    """Por día: hit_rate = count(cache_hit:true) / count(extra_json tiene cache_hit)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day_hit: dict[str, int] = {d: 0 for d in dates}
        per_day_total: dict[str, int] = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            hit_rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND extra_json LIKE '%\"cache_hit\": true%' "
                "GROUP BY d",
                (cutoff,),
            ).fetchall()
            total_rows = conn.execute(
                "SELECT DATE(ts) AS d, COUNT(*) FROM rag_queries "
                "WHERE ts >= ? AND extra_json LIKE '%\"cache_hit\":%' "
                "GROUP BY d",
                (cutoff,),
            ).fetchall()
        for d, n in hit_rows:
            if d in per_day_hit:
                per_day_hit[d] = int(n)
        for d, n in total_rows:
            if d in per_day_total:
                per_day_total[d] = int(n)
        series = []
        for d in dates:
            tot = per_day_total[d]
            hit = per_day_hit[d]
            rate = (hit / tot) if tot else 0.0
            series.append({
                "date": d,
                "hit_rate": round(rate, 4),
                "n_queries": tot,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_query_cache_hit_rate_failed", default=_empty_series(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Anticipatory (rag_anticipate_candidates + anticipate_weights.json)
# ─────────────────────────────────────────────────────────────────────────────

_ANTICIPATE_KINDS = ("anticipate-calendar", "anticipate-echo", "anticipate-commitment")


def anticipatory(days: int = 30) -> dict:
    return {
        "candidates_by_kind_over_time": _anticipate_by_kind(days),
        "selection_send_rate": _anticipate_selection_rate(days),
        "user_reactions": _anticipate_user_reactions(days),
        "weights_current": _read_anticipate_weights(),
    }


def _anticipate_by_kind(days: int) -> dict:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: {k: 0 for k in _ANTICIPATE_KINDS} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts) AS d, kind, COUNT(*) FROM rag_anticipate_candidates "
                "WHERE ts >= ? GROUP BY d, kind",
                (cutoff,),
            ).fetchall()
        for d, kind, n in rows:
            if d in per_day and kind in per_day[d]:
                per_day[d][kind] = int(n)
        series = [
            {"date": d, "values": [per_day[d][k] for k in _ANTICIPATE_KINDS]}
            for d in dates
        ]
        return {
            "kinds": list(_ANTICIPATE_KINDS),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_anticipate_by_kind_failed",
        default={"kinds": list(_ANTICIPATE_KINDS), **_empty_series()},
    )


def _anticipate_selection_rate(days: int) -> dict:
    """Por día: selection_rate = selected/total candidates, send_rate =
    sent/selected (si selected > 0; si no, 0)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day_total = {d: 0 for d in dates}
        per_day_sel = {d: 0 for d in dates}
        per_day_sent = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts) AS d, "
                "       COUNT(*) AS total, "
                "       SUM(selected) AS sel, "
                "       SUM(sent) AS sent "
                "FROM rag_anticipate_candidates WHERE ts >= ? GROUP BY d",
                (cutoff,),
            ).fetchall()
        for d, total, sel, sent in rows:
            if d in per_day_total:
                per_day_total[d] = int(total or 0)
                per_day_sel[d] = int(sel or 0)
                per_day_sent[d] = int(sent or 0)
        series = []
        for d in dates:
            tot = per_day_total[d]
            sel = per_day_sel[d]
            sent = per_day_sent[d]
            sel_rate = (sel / tot) if tot else 0.0
            send_rate = (sent / sel) if sel else 0.0
            series.append({
                "date": d,
                "selection_rate": round(sel_rate, 4),
                "send_rate": round(send_rate, 4),
                "n_candidates": tot,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_anticipate_selection_rate_failed", default=_empty_series(),
    )


def _anticipate_user_reactions(days: int) -> dict:
    """No existe una tabla `rag_anticipate_feedback` en el schema actual.
    Como proxy: contamos eventos de rag_behavior con source=brief y events
    relacionados (positive_implicit, kept, deleted). Si no hay match,
    devolvemos todos en 0 con insufficient implícito por la cuenta total.

    Mapping:
        rag_behavior.event == 'positive_implicit'  → positive
        rag_behavior.event == 'deleted'            → negative
        rag_behavior.event == 'kept'               → unknown (no es ni + ni -)
        rag_behavior.event == 'mute_*'             → mute (no existe hoy, defensivo)
    """
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        out = {"positive": 0, "negative": 0, "mute": 0, "unknown": 0}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT event, COUNT(*) FROM rag_behavior "
                "WHERE ts >= ? AND source = 'brief' GROUP BY event",
                (cutoff,),
            ).fetchall()
        for event, n in rows:
            n_i = int(n or 0)
            if event == "positive_implicit":
                out["positive"] += n_i
            elif event == "deleted":
                out["negative"] += n_i
            elif event and "mute" in str(event).lower():
                out["mute"] += n_i
            else:
                out["unknown"] += n_i
        return out

    return _sql_read_with_retry(
        _do,
        "learning_anticipate_user_reactions_failed",
        default={"positive": 0, "negative": 0, "mute": 0, "unknown": 0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Routing Learning (rag_routing_decisions + rag_routing_rules)
# ─────────────────────────────────────────────────────────────────────────────

def routing_learning(days: int = 30) -> dict:
    return {
        "decisions_by_bucket_over_time": _routing_by_bucket(days),
        "active_rules_count_over_time": _routing_active_rules_over_time(days),
        "evidence_ratio_distribution": _routing_evidence_distribution(),
    }


def _routing_by_bucket(days: int) -> dict:
    """rag_routing_decisions.ts es UNIX integer — convertimos a date con
    DATETIME(ts, 'unixepoch')."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = int(_cutoff_unix(days))
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: {b: 0 for b in _ROUTING_BUCKETS} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts, 'unixepoch') AS d, bucket_final, COUNT(*) "
                "FROM rag_routing_decisions WHERE ts >= ? "
                "GROUP BY d, bucket_final",
                (cutoff,),
            ).fetchall()
        for d, bucket, n in rows:
            if d in per_day and bucket in per_day[d]:
                per_day[d][bucket] = int(n)
        series = [
            {"date": d, "values": [per_day[d][b] for b in _ROUTING_BUCKETS]}
            for d in dates
        ]
        return {
            "buckets": list(_ROUTING_BUCKETS),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_routing_by_bucket_failed",
        default={"buckets": list(_ROUTING_BUCKETS), **_empty_series()},
    )


def _routing_active_rules_over_time(days: int) -> dict:
    """rag_routing_rules.promoted_at es UNIX. Reglas "activas en fecha D" =
    rules con promoted_at <= D AND active = 1. Acumulado simple desde la
    promoción inicial."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    dates = _date_range(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT promoted_at FROM rag_routing_rules WHERE active = 1"
            ).fetchall()
        promotions = sorted(int(r[0] or 0) for r in rows if r and r[0] is not None)
        series = []
        for d in dates:
            try:
                end_ts = int(datetime.fromisoformat(d).timestamp()) + 86400
            except Exception:
                end_ts = 0
            count = sum(1 for p in promotions if p <= end_ts)
            series.append({"date": d, "count": count})
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_routing_active_rules_failed", default=_empty_series(),
    )


def _routing_evidence_distribution() -> dict:
    """Buckets de evidence_ratio en pasos de 0.1, 0.5-1.0."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    bucket_defs = [
        ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90-1.00", 0.90, 1.0001),
    ]

    def _do() -> dict:
        counts = {label: 0 for label, _, _ in bucket_defs}
        with _ragvec_state_conn() as conn:
            for label, lo, hi in bucket_defs:
                n = conn.execute(
                    "SELECT COUNT(*) FROM rag_routing_rules "
                    "WHERE evidence_ratio >= ? AND evidence_ratio < ?",
                    (lo, hi),
                ).fetchone()[0] or 0
                counts[label] = int(n)
        total = sum(counts.values())
        return {
            "buckets": [{"range": label, "count": counts[label]} for label, _, _ in bucket_defs],
            "n_samples": total,
            "insufficient": _insufficient(total),
        }

    return _sql_read_with_retry(
        _do,
        "learning_routing_evidence_distribution_failed",
        default={
            "buckets": [{"range": label, "count": 0} for label, _, _ in bucket_defs],
            "n_samples": 0,
            "insufficient": True,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Whisper Learning (rag_whisper_vocab + rag_audio_corrections + rag_audio_transcripts)
# ─────────────────────────────────────────────────────────────────────────────

_WHISPER_CORRECTION_SOURCES = ("explicit", "llm", "vault_diff")


def whisper_learning(days: int = 30) -> dict:
    return {
        "vocab_size_by_source": _whisper_vocab_size(),
        "corrections_by_source_over_time": _whisper_corrections_over_time(days),
        "avg_logprob_over_time": _whisper_logprob_over_time(days),
        "top_vocab": _whisper_top_vocab(),
    }


def _whisper_vocab_size() -> dict:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) FROM rag_whisper_vocab GROUP BY source"
            ).fetchall()
        by_source = {str(s): int(n) for s, n in rows if s}
        return {"by_source": by_source}

    return _sql_read_with_retry(
        _do, "learning_whisper_vocab_size_failed", default={"by_source": {}},
    )


def _whisper_corrections_over_time(days: int) -> dict:
    """rag_audio_corrections.ts es UNIX REAL."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_unix(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: {s: 0 for s in _WHISPER_CORRECTION_SOURCES} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts, 'unixepoch') AS d, source, COUNT(*) "
                "FROM rag_audio_corrections WHERE ts >= ? GROUP BY d, source",
                (cutoff,),
            ).fetchall()
        for d, source, n in rows:
            if d in per_day and source in per_day[d]:
                per_day[d][source] = int(n)
        series = [
            {"date": d, "values": [per_day[d][s] for s in _WHISPER_CORRECTION_SOURCES]}
            for d in dates
        ]
        return {
            "sources": list(_WHISPER_CORRECTION_SOURCES),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_whisper_corrections_failed",
        default={"sources": list(_WHISPER_CORRECTION_SOURCES), **_empty_series()},
    )


def _whisper_logprob_over_time(days: int) -> dict:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_unix(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: {"sum": 0.0, "n": 0} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(transcribed_at, 'unixepoch') AS d, "
                "       AVG(avg_logprob) AS avg_lp, "
                "       COUNT(avg_logprob) AS n "
                "FROM rag_audio_transcripts "
                "WHERE transcribed_at >= ? AND avg_logprob IS NOT NULL "
                "GROUP BY d",
                (cutoff,),
            ).fetchall()
        per_day_avg: dict[str, tuple[float, int]] = {}
        for d, avg_lp, n in rows:
            if d:
                per_day_avg[d] = (float(avg_lp or 0.0), int(n or 0))
        series = []
        for d in dates:
            avg_lp, n = per_day_avg.get(d, (0.0, 0))
            series.append({
                "date": d,
                "avg_logprob": round(avg_lp, 4) if n else None,
                "n_transcripts": n,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_whisper_logprob_failed", default=_empty_series(),
    )


def _whisper_top_vocab() -> list[dict]:
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> list[dict]:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT term, weight, source FROM rag_whisper_vocab "
                "ORDER BY weight DESC LIMIT 20"
            ).fetchall()
        return [
            {
                "term": term or "",
                "weight": round(float(w or 0.0), 4),
                "source": source or "",
            }
            for term, w, source in rows
        ]

    return _sql_read_with_retry(_do, "learning_whisper_top_vocab_failed", default=[]) or []


# ─────────────────────────────────────────────────────────────────────────────
# 12. Vault Intelligence
#     (rag_entities, rag_entity_mentions, rag_contradictions, rag_filing_log,
#      rag_surface_log, rag_archive_log)
# ─────────────────────────────────────────────────────────────────────────────

def vault_intelligence(days: int = 30) -> dict:
    return {
        "entities_by_type_over_time": _vault_entities_over_time(days),
        "mentions_per_day": _vault_mentions_per_day(days),
        "contradictions_per_week": _vault_contradictions_per_week(days),
        "filing_by_folder_over_time": _vault_filing_over_time(days),
        "surface_archive_over_time": _vault_surface_archive_over_time(days),
    }


def _vault_entities_over_time(days: int) -> dict:
    """Acumulado: para cada fecha D, count de entities con first_seen_ts <= end_of(D),
    grouped by entity_type. El schema usa entity_type lowercase
    (person/organization/location/event); normalizamos a UPPERCASE
    (PERSON/ORG/LOCATION/OTHER). "OTHER" agrupa todo lo que no matchea
    a las 3 categorías canonicalmente trackeadas (incluye event)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    dates = _date_range(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT entity_type, first_seen_ts FROM rag_entities "
                "WHERE first_seen_ts IS NOT NULL"
            ).fetchall()
        # Bucket entities por tipo + ordered list de first_seen.
        by_type: dict[str, list[float]] = {t: [] for t in _ENTITY_TYPES}
        for raw_type, fst in rows:
            mapped = _ENTITY_TYPE_MAP.get(str(raw_type or "").lower(), "OTHER")
            try:
                by_type[mapped].append(float(fst))
            except Exception:
                continue
        for k in by_type:
            by_type[k].sort()
        series = []
        for d in dates:
            try:
                end_ts = datetime.fromisoformat(d).timestamp() + 86400
            except Exception:
                end_ts = 0.0
            values = []
            for t in _ENTITY_TYPES:
                xs = by_type[t]
                # Binary search-ish: count xs <= end_ts.
                lo, hi = 0, len(xs)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if xs[mid] <= end_ts:
                        lo = mid + 1
                    else:
                        hi = mid
                values.append(lo)
            series.append({"date": d, "values": values})
        return {
            "types": list(_ENTITY_TYPES),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_vault_entities_over_time_failed",
        default={"types": list(_ENTITY_TYPES), **_empty_series()},
    )


def _vault_mentions_per_day(days: int) -> dict:
    """rag_entity_mentions.ts es UNIX REAL."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_unix(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts, 'unixepoch') AS d, COUNT(*) "
                "FROM rag_entity_mentions WHERE ts >= ? GROUP BY d",
                (cutoff,),
            ).fetchall()
        for d, n in rows:
            if d in per_day:
                per_day[d] = int(n)
        series = [{"date": d, "count": per_day[d]} for d in dates]
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_vault_mentions_per_day_failed", default=_empty_series(),
    )


def _vault_contradictions_per_week(days: int) -> dict:
    """Bucketing weekly: week_start = lunes ISO de cada fecha. detected =
    rows con `skipped IS NULL OR skipped = ''`. resolved = rows con
    skipped no-null/no-empty (mismo proxy que el KPI)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT ts, "
                "  CASE WHEN skipped IS NULL OR skipped = '' THEN 0 ELSE 1 END AS resolved "
                "FROM rag_contradictions WHERE ts >= ?",
                (cutoff,),
            ).fetchall()
        per_week: dict[str, dict[str, int]] = {}
        for ts, resolved in rows:
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            week_start = (dt.date() - timedelta(days=dt.weekday())).isoformat()
            bucket = per_week.setdefault(week_start, {"detected": 0, "resolved": 0})
            bucket["detected"] += 1
            if resolved:
                bucket["resolved"] += 1
        series = sorted(
            ({"week_start": k, "detected": v["detected"], "resolved": v["resolved"]}
             for k, v in per_week.items()),
            key=lambda x: x["week_start"],
        )
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_vault_contradictions_per_week_failed", default=_empty_series(),
    )


def _vault_filing_over_time(days: int) -> dict:
    """Por día, count de filings agrupado por la primera componente del
    folder (lo que matchea con el PARA top-level)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day = {d: {f: 0 for f in _PARA_FOLDERS} for d in dates}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DATE(ts) AS d, folder, COUNT(*) "
                "FROM rag_filing_log WHERE ts >= ? AND folder IS NOT NULL "
                "GROUP BY d, folder",
                (cutoff,),
            ).fetchall()
        for d, folder, n in rows:
            if d not in per_day:
                continue
            top = str(folder or "").split("/", 1)[0]
            if top in per_day[d]:
                per_day[d][top] += int(n)
        series = [
            {"date": d, "values": [per_day[d][f] for f in _PARA_FOLDERS]}
            for d in dates
        ]
        return {
            "folders": list(_PARA_FOLDERS),
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do,
        "learning_vault_filing_over_time_failed",
        default={"folders": list(_PARA_FOLDERS), **_empty_series()},
    )


def _vault_surface_archive_over_time(days: int) -> dict:
    """Por día: surface = SUM(n_pairs) de rag_surface_log; archive =
    SUM(n_applied) de rag_archive_log."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)
    dates = _date_range(days)

    def _do() -> dict:
        per_day_surf = {d: 0 for d in dates}
        per_day_arch = {d: 0 for d in dates}
        with _ragvec_state_conn() as conn:
            for d, n in conn.execute(
                "SELECT DATE(ts) AS d, COALESCE(SUM(n_pairs), 0) "
                "FROM rag_surface_log WHERE ts >= ? GROUP BY d",
                (cutoff,),
            ).fetchall():
                if d in per_day_surf:
                    per_day_surf[d] = int(n or 0)
            for d, n in conn.execute(
                "SELECT DATE(ts) AS d, COALESCE(SUM(n_applied), 0) "
                "FROM rag_archive_log WHERE ts >= ? GROUP BY d",
                (cutoff,),
            ).fetchall():
                if d in per_day_arch:
                    per_day_arch[d] = int(n or 0)
        series = [
            {"date": d, "surface": per_day_surf[d], "archive": per_day_arch[d]}
            for d in dates
        ]
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_vault_surface_archive_failed", default=_empty_series(),
    )


# ── /api/learning/health ─────────────────────────────────────────
# Semáforo del sistema. Devuelve un nivel global (green/yellow/red) +
# señales individuales con texto en lenguaje plain. Pensado para que un
# usuario sin conocimiento técnico abra el dashboard y entienda en 2s si
# algo está mal o no.
#
# Filosofía:
#   - "Worst-case wins": si una sola señal está roja, el banner va rojo.
#   - Cada señal trae 4 datos: nivel propio, valor crudo (para el dev),
#     texto plain (para el usuario), y tooltip técnico (para el dev en hover).
#   - Thresholds basados en p50 histórico (30d) — ver constantes ``_HEALTH_*``
#     más abajo. Son ajustables sin romper el shape del payload.
#
# Aprendido el 2026-04-26: el dashboard previo tenía 8 KPIs + 11 secciones
# + 43 charts pero ningún indicador de "todo OK / algo mal". Un usuario
# sin contexto necesitaba interpretar números técnicos para saber si el
# sistema estaba andando. Este endpoint resuelve esa pregunta primero.

# Thresholds — verde es ≥, rojo es ≤, amarillo en el medio.
_HEALTH_RETRIEVAL_SINGLES_GREEN = 0.75   # ≥75% del eval set en top-5
_HEALTH_RETRIEVAL_SINGLES_RED = 0.60     # <60% es señal de regresión
_HEALTH_RETRIEVAL_CHAINS_GREEN = 0.80    # ≥80% multi-hop OK
_HEALTH_RETRIEVAL_CHAINS_RED = 0.65      # <65% multi-hop está roto
_HEALTH_VAULT_GREEN_HOURS = 2.0          # último cambio indexado <2h
_HEALTH_VAULT_RED_HOURS = 12.0           # >12h sin actividad de indexación
_HEALTH_ERRORS_GREEN_PCT = 0.0           # 0% degraded runs últimas 24h
_HEALTH_ERRORS_RED_PCT = 0.05            # >5% es señal real
_HEALTH_SPEED_GREEN_S = 20.0             # p95 latencia <20s (datos reales: p50=9s, p95=32s)
_HEALTH_SPEED_RED_S = 45.0               # >45s p95 ya es problema operativo
_HEALTH_SPEED_MIN_SAMPLES = 20           # n<20 → "sin suficientes datos" (yellow)

# Servicios "críticos" — si UNO de estos está caído, level=red. Servicios
# secundarios (digest, archive, etc.) son tolerables → max amarillo.
_HEALTH_CRITICAL_SERVICES = {
    "com.fer.obsidian-rag-web",     # web UI + chat SSE
    "com.fer.obsidian-rag-watch",   # vault file watcher
    "com.fer.obsidian-rag-serve",   # query server hot path
}


def _level_worst(levels: list[str]) -> str:
    """Worst-case wins: red > yellow > green. Default green si lista vacía."""
    if "red" in levels:
        return "red"
    if "yellow" in levels:
        return "yellow"
    return "green"


def _signal(*, key: str, label: str, level: str, value_text: str,
            value_raw: float | int | str | None, tooltip: str,
            explanation: str = "") -> dict:
    """Shape canonical de una señal individual. Todas las funciones
    ``_health_*`` devuelven exactamente este dict."""
    return {
        "key": key,
        "label": label,
        "level": level,
        "value_text": value_text,
        "value_raw": value_raw,
        "tooltip": tooltip,
        "explanation": explanation,
    }


def _health_retrieval(column: str, *, key: str, label_simple_or_complex: str,
                      tooltip_kind: str,
                      green_th: float, red_th: float) -> dict:
    """Una señal de calidad de retrieval (singles o chains). Reusa el
    pattern de ``_kpi_eval_hit5`` pero devuelve el shape de health."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(30)

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                f"SELECT AVG({column}) AS v, COUNT({column}) AS n "
                "FROM rag_eval_runs WHERE ts >= ?",
                (cutoff,),
            ).fetchone()
        v = float(row[0]) if row and row[0] is not None else 0.0
        n = int(row[1]) if row and row[1] is not None else 0
        if n < _INSUFFICIENT_THRESHOLD:
            return _signal(
                key=key,
                label=f"Acierto en {label_simple_or_complex}",
                level="yellow",
                value_text="Sin suficientes datos todavía",
                value_raw=None,
                tooltip=f"hit@5 sobre {tooltip_kind} del eval set",
                explanation=f"Solo hay {n} corridas de eval en los últimos 30 días.",
            )
        # Texto en "X de cada 10 veces" para que sea intuitivo.
        out_of_10 = round(v * 10, 1)
        # Si es .0 lo mostramos sin decimal: "9 de cada 10". Si no: "8.5 de cada 10".
        out_of_10_text = (
            f"{int(out_of_10)}" if out_of_10 == int(out_of_10) else f"{out_of_10:.1f}"
        )
        if v >= green_th:
            level = "green"
        elif v <= red_th:
            level = "red"
        else:
            level = "yellow"
        return _signal(
            key=key,
            label=f"Acierto en {label_simple_or_complex}",
            level=level,
            value_text=f"{out_of_10_text} de cada 10 veces",
            value_raw=round(v, 4),
            tooltip=f"hit@5 sobre {tooltip_kind} del eval set",
            explanation=f"Promedio últimos 30 días sobre {n} corridas de eval.",
        )

    return _sql_read_with_retry(
        _do, f"learning_health_{key}_failed",
        default=_signal(
            key=key,
            label=f"Acierto en {label_simple_or_complex}",
            level="yellow",
            value_text="No pude leer la base",
            value_raw=None,
            tooltip=f"hit@5 sobre {tooltip_kind} del eval set",
            explanation="Falla al leer rag_eval_runs.",
        ),
    )


def _health_services() -> dict:
    """Lee `launchctl list` y verifica que los daemons críticos del RAG
    tengan PID asignado (≠ "-"). Servicios programados (no-keepalive) que
    están en exit 0 sin PID se consideran OK — son cron-style.

    Tolerancia:
      - Todos críticos arriba → green.
      - 1 secundario caído (no en la lista crítica) → yellow.
      - ≥1 crítico caído O ≥1 con last_exit != 0 → red.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(f"launchctl exit {result.returncode}")
    except Exception:
        return _signal(
            key="services",
            label="Servicios del sistema",
            level="yellow",
            value_text="No pude verificar el estado",
            value_raw=None,
            tooltip="`launchctl list` falló o no está disponible",
            explanation="No pudimos consultar a macOS por el estado de los servicios.",
        )

    rag_lines = [
        line for line in result.stdout.splitlines()
        if "com.fer.obsidian-rag-" in line
    ]
    critical_down: list[str] = []
    nonzero_exit: list[str] = []
    n_running_critical = 0
    seen_labels: set[str] = set()
    for line in rag_lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        pid, status, label = parts[0], parts[1], parts[2]
        seen_labels.add(label)
        is_critical = label in _HEALTH_CRITICAL_SERVICES
        try:
            status_int = int(status)
        except ValueError:
            status_int = 0
        has_pid = pid != "-"
        # Si tiene PID, está corriendo AHORA. El status que reporta launchctl
        # es del último exit ANTERIOR (post-bootout/restart típicamente -15
        # SIGTERM), no del estado actual. Daemon con PID = vivo, regardless.
        if has_pid:
            if is_critical:
                n_running_critical += 1
            continue
        # Sin PID — puede ser un crítico que se cayó O un job programado
        # (cron-style con KeepAlive=false) que está esperando su próximo tick.
        if is_critical:
            # Crítico se espera que esté corriendo siempre (KeepAlive=true).
            # Sin PID = caído.
            critical_down.append(label.replace("com.fer.obsidian-rag-", ""))
        elif status_int != 0:
            # Secundario sin PID con last_exit != 0 (y != -15 SIGTERM): crash real.
            # -15 SIGTERM es restart limpio, no crash.
            if status_int != -15:
                nonzero_exit.append(label.replace("com.fer.obsidian-rag-", ""))

    # Si falta algún crítico de la lista esperada (no aparece en launchctl list).
    for crit in _HEALTH_CRITICAL_SERVICES:
        if crit not in seen_labels:
            critical_down.append(crit.replace("com.fer.obsidian-rag-", "") + " (no instalado)")

    if critical_down or nonzero_exit:
        items_red = critical_down + [f"{s} (crasheó)" for s in nonzero_exit]
        return _signal(
            key="services",
            label="Servicios del sistema",
            level="red",
            value_text="Hay servicios caídos: " + ", ".join(items_red[:3]),
            value_raw=len(items_red),
            tooltip="critical down: " + ",".join(critical_down) +
                    " | nonzero exit: " + ",".join(nonzero_exit),
            explanation="Si la búsqueda no responde, lo más probable es esto.",
        )
    return _signal(
        key="services",
        label="Servicios del sistema",
        level="green",
        value_text="Todos respondiendo bien",
        value_raw=n_running_critical,
        tooltip=f"{n_running_critical} críticos arriba: " +
                ",".join(s.replace("com.fer.obsidian-rag-", "")
                         for s in _HEALTH_CRITICAL_SERVICES),
        explanation="Web, búsqueda e indexador están todos arriba.",
    )


def _health_vault_freshness() -> dict:
    """¿Cuándo fue el último cambio que el sistema procesó? Mira el último
    ts de ``rag_filing_log`` (acción de archiving) o ``rag_queries`` (uso
    activo). Tomamos el MÁS reciente — si hay queries pero no filings, el
    vault sigue activo aunque el indexer no haya tenido nada que archivar."""
    from rag import _ragvec_state_conn, _sql_read_with_retry
    import time as _time

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            # rag_queries ts es ISO8601 — comparamos directamente con strftime.
            row_q = conn.execute(
                "SELECT MAX(ts) FROM rag_queries"
            ).fetchone()
            row_f = conn.execute(
                "SELECT MAX(ts) FROM rag_filing_log"
            ).fetchone()
        last_q = row_q[0] if row_q and row_q[0] else None
        last_f = row_f[0] if row_f and row_f[0] else None
        candidates = [t for t in (last_q, last_f) if t]
        if not candidates:
            return _signal(
                key="vault_freshness",
                label="Vault al día",
                level="yellow",
                value_text="Sin actividad reciente",
                value_raw=None,
                tooltip="MAX(ts) de rag_queries y rag_filing_log: ambas vacías",
                explanation="No hay queries ni archivos procesados todavía.",
            )
        last_iso = max(candidates)
        # Parse ISO8601 sin timezone — asumimos local.
        try:
            last_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
        except Exception:
            return _signal(
                key="vault_freshness",
                label="Vault al día",
                level="yellow",
                value_text="No pude leer la fecha",
                value_raw=last_iso,
                tooltip=f"datetime.fromisoformat falló sobre: {last_iso}",
                explanation="Fecha del último evento en formato inesperado.",
            )
        # Convertimos a aware si no lo es (ts naive = local time del sistema).
        if last_dt.tzinfo is None:
            last_dt = last_dt.astimezone()
        now = datetime.now(last_dt.tzinfo)
        age_s = (now - last_dt).total_seconds()
        age_h = age_s / 3600.0
        if age_h < _HEALTH_VAULT_GREEN_HOURS:
            level = "green"
            text = _humanize_age(age_s) + " desde el último cambio"
            expl = "Tu vault está siendo procesado al día."
        elif age_h > _HEALTH_VAULT_RED_HOURS:
            level = "red"
            text = "Sin actividad hace " + _humanize_age(age_s)
            expl = "El indexer puede estar caído o el vault sin uso."
        else:
            level = "yellow"
            text = "Última actividad hace " + _humanize_age(age_s)
            expl = "Todo OK pero hace un rato que no procesa nada."
        return _signal(
            key="vault_freshness",
            label="Vault al día",
            level=level,
            value_text=text,
            value_raw=round(age_h, 2),
            tooltip="MAX(ts) de rag_queries ∪ rag_filing_log",
            explanation=expl,
        )

    return _sql_read_with_retry(
        _do, "learning_health_vault_freshness_failed",
        default=_signal(
            key="vault_freshness",
            label="Vault al día",
            level="yellow",
            value_text="No pude leer la base",
            value_raw=None,
            tooltip="DB read failed",
            explanation="Falla al consultar telemetry.db.",
        ),
    )


def _humanize_age(seconds: float) -> str:
    """7200 → '2 horas'. 90 → '1 minuto'. 86400 → '1 día'. Castellano."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} segundos" if seconds != 1 else "1 segundo"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minutos" if minutes != 1 else "1 minuto"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} horas" if hours != 1 else "1 hora"
    days = hours // 24
    return f"{days} días" if days != 1 else "1 día"


def _health_errors_24h() -> dict:
    """% de runs del home-compute que terminaron degraded en las últimas 24h.
    Es el indicador más confiable de "algo se está rompiendo en silencio"
    porque el home-compute corre cada vez que alguien abre /home (~50/hr) y
    el flag `degraded` se setea cuando alguna sub-task tiró excepción."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT SUM(degraded) AS d, COUNT(*) AS n "
                "FROM rag_home_compute_metrics "
                "WHERE ts >= datetime('now', '-1 day')"
            ).fetchone()
        n_degraded = int(row[0] or 0) if row else 0
        n_total = int(row[1] or 0) if row else 0
        if n_total == 0:
            return _signal(
                key="errors_24h",
                label="Errores recientes",
                level="yellow",
                value_text="Sin actividad para evaluar",
                value_raw=None,
                tooltip="rag_home_compute_metrics vacío en últimas 24h",
                explanation="Nadie abrió la home en 24h, no hay datos para chequear.",
            )
        pct = n_degraded / n_total
        if pct <= _HEALTH_ERRORS_GREEN_PCT:
            level = "green"
            text = "Sin errores"
            expl = "Ninguna corrida del sistema falló en las últimas 24 horas."
        elif pct >= _HEALTH_ERRORS_RED_PCT:
            level = "red"
            text = f"{n_degraded} corridas fallaron de {n_total}"
            expl = "Más del 5% del sistema falló — revisar los logs."
        else:
            level = "yellow"
            text = f"{n_degraded} corridas degradadas de {n_total}"
            expl = "Algunos errores aislados — vale la pena revisar."
        return _signal(
            key="errors_24h",
            label="Errores recientes",
            level=level,
            value_text=text,
            value_raw=round(pct, 4),
            tooltip=f"SUM(degraded)/COUNT(*) de rag_home_compute_metrics 24h: "
                    f"{n_degraded}/{n_total}",
            explanation=expl,
        )

    return _sql_read_with_retry(
        _do, "learning_health_errors_24h_failed",
        default=_signal(
            key="errors_24h",
            label="Errores recientes",
            level="yellow",
            value_text="No pude leer la base",
            value_raw=None,
            tooltip="DB read failed",
            explanation="Falla al consultar telemetry.db.",
        ),
    )


def _health_response_speed() -> dict:
    """p95 de latencia (t_retrieve + t_gen) en los últimos 7 días.
    NTILE(100) sobre los queries con t > 0 nos da percentiles sin
    necesidad de window functions complejas."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            row = conn.execute("""
                WITH ranked AS (
                  SELECT (COALESCE(t_retrieve,0)+COALESCE(t_gen,0)) AS t,
                         NTILE(100) OVER (ORDER BY (COALESCE(t_retrieve,0)+COALESCE(t_gen,0))) AS pct
                  FROM rag_queries
                  WHERE ts >= datetime('now','-7 days')
                    AND (COALESCE(t_retrieve,0)+COALESCE(t_gen,0)) > 0
                )
                SELECT MAX(t), COUNT(*) FROM ranked WHERE pct = 95
            """).fetchone()
        p95 = float(row[0]) if row and row[0] is not None else 0.0
        n = int(row[1]) if row and row[1] is not None else 0
        if n < _HEALTH_SPEED_MIN_SAMPLES:
            return _signal(
                key="response_speed",
                label="Velocidad de respuesta",
                level="yellow",
                value_text=f"Sin suficientes consultas para medir (n={n})",
                value_raw=None,
                tooltip=f"NTILE(100) p95 de t_retrieve+t_gen en 7d, "
                        f"n={n} < min={_HEALTH_SPEED_MIN_SAMPLES}",
                explanation="Hacen falta más consultas en los últimos 7 días "
                            "para que el p95 sea representativo.",
            )
        if p95 < _HEALTH_SPEED_GREEN_S:
            level = "green"
            text = f"95% de respuestas en menos de {p95:.0f}s"
            expl = "El sistema responde rápido casi siempre."
        elif p95 > _HEALTH_SPEED_RED_S:
            level = "red"
            text = f"Hay respuestas que tardan más de {p95:.0f}s"
            expl = "El sistema está lento — revisar carga del LLM o de Ollama."
        else:
            level = "yellow"
            text = f"95% de respuestas en menos de {p95:.0f}s"
            expl = "Velocidad aceptable pero podría ser mejor."
        return _signal(
            key="response_speed",
            label="Velocidad de respuesta",
            level=level,
            value_text=text,
            value_raw=round(p95, 2),
            tooltip=f"p95 latencia (t_retrieve+t_gen) últimos 7d sobre n={n}",
            explanation=expl,
        )

    return _sql_read_with_retry(
        _do, "learning_health_response_speed_failed",
        default=_signal(
            key="response_speed",
            label="Velocidad de respuesta",
            level="yellow",
            value_text="No pude leer la base",
            value_raw=None,
            tooltip="DB read failed",
            explanation="Falla al consultar telemetry.db.",
        ),
    )


def system_health() -> dict:
    """Computa el semáforo del sistema. Devuelve nivel global + lista de
    señales individuales con texto en lenguaje plain.

    Cada señal corre en un try/except independiente (vía
    ``_sql_read_with_retry``); la falla de una NO contamina las otras —
    devuelve nivel `yellow` y texto explicando que no pudo leer la base.

    Side effect: ``_health_services`` invoca subprocess `launchctl list`
    (timeout 5s). Si launchctl no está disponible (no-macOS) la señal
    devuelve yellow.

    El resultado se cachea en `web/server.py` con TTL=15s — es lo que
    determina con qué frecuencia se actualiza el banner. Más corto que
    el `/api/learning` (60s) porque queremos que el usuario
    vea cambios de salud rápido."""
    signals = [
        _health_retrieval(
            "singles_hit5",
            key="retrieval_singles",
            label_simple_or_complex="preguntas simples",
            tooltip_kind="singles",
            green_th=_HEALTH_RETRIEVAL_SINGLES_GREEN,
            red_th=_HEALTH_RETRIEVAL_SINGLES_RED,
        ),
        _health_retrieval(
            "chains_hit5",
            key="retrieval_chains",
            label_simple_or_complex="preguntas con varios pasos",
            tooltip_kind="chains",
            green_th=_HEALTH_RETRIEVAL_CHAINS_GREEN,
            red_th=_HEALTH_RETRIEVAL_CHAINS_RED,
        ),
        _health_services(),
        _health_vault_freshness(),
        _health_errors_24h(),
        _health_response_speed(),
    ]
    overall = _level_worst([s["level"] for s in signals])
    headlines = {
        "green": "Todo funcionando bien",
        "yellow": "Hay algo para vigilar",
        "red": "Algo no está bien",
    }
    summaries = {
        "green": "El sistema responde bien y está al día.",
        "yellow": "Hay alguna señal que conviene revisar — todo lo crítico anda.",
        "red": "Algo importante no está funcionando — mirá los detalles.",
    }
    return {
        "level": overall,
        "headline": headlines[overall],
        "summary": summaries[overall],
        "signals": signals,
        "checked_at": datetime.now().astimezone().isoformat(),
    }


# ── Veredicto: ¿aprende cada sistema? ──────────────────────────────────────
#
# Una mirada honesta del estado del aprendizaje. Cada sistema está en uno de
# tres estados: 🟢 alive (actividad reciente + cambios), 🟡 stale (hubo data
# pero ya no), 🔴 dormant (nunca aprendió o loop roto).
#
# Origen: el diagnóstico manual que hicimos el 2026-04-26 detectó que de los
# 11 sistemas de aprendizaje del RAG, 5 estaban vivos, 3 stale, 3 dormidos
# (paraphrases vacío, routing rules sin data, audio corrections sin uso). El
# loop de anticipatory estaba ROTO (72 candidates sent=0) porque ambient.json
# no existía. Esta función automatiza ese diagnóstico para que no haya que
# correrlo a mano nunca más.
#
# Importante: la regla de "alive" es flexible por sistema. Un sistema que
# corre nightly se considera alive si tuvo update en las últimas 48h; uno
# continuo (behavior, queries) en las últimas 6h. Si en el futuro cambian
# las cadencias, ajustar `_VERDICT_THRESHOLDS_HOURS`.

# Thresholds por sistema (en horas) para el corte alive/stale.
# stale → alive si último update < alive_h. dormant si nunca o > stale_h.
_VERDICT_THRESHOLDS_HOURS: dict[str, tuple[int, int]] = {
    # Continuo (events constantes mientras el user usa el sistema)
    "feedback":            (48, 24 * 14),    # 2d alive, 14d stale-cap
    "behavior":            (6,  24 * 7),     # 6h alive, 7d stale-cap
    "contradictions":      (24, 24 * 14),
    "entities":            (24 * 7, 24 * 30),  # crece con indexing
    # Nightly / cron
    "ranker":              (24,      24 * 7),
    "eval":                (24,      24 * 7),
    "score_calibration":   (24 * 14, 24 * 60),
    "whisper_vocab":       (24 * 14, 24 * 60),
    # On-demand / depende de actividad del user
    "anticipatory":        (48,      24 * 7),
    "paraphrases":         (24 * 30, 24 * 90),
    "routing_rules":       (24 * 30, 24 * 90),
    "whisper_corrections": (24 * 30, 24 * 90),
}


def _hours_ago(ts_iso: str | None) -> float | None:
    """Horas transcurridas desde un ISO8601 (o None). Si el ts no parsea,
    devuelve None — el caller trata None como "nunca"."""
    if not ts_iso:
        return None
    try:
        # Soporta naive (lo más común en este DB) y aware. Comparamos contra
        # naive UTC porque las tablas guardan localtime ISO sin tz.
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        delta = datetime.now() - ts
        return delta.total_seconds() / 3600.0
    except Exception:
        return None


def _verdict_status(system_id: str, hours_since: float | None) -> str:
    """Devuelve 'alive' | 'stale' | 'dormant' según `_VERDICT_THRESHOLDS_HOURS`.

    Si `hours_since is None` (nunca tuvo update), devuelve 'dormant'.
    """
    if hours_since is None:
        return "dormant"
    alive_h, stale_h = _VERDICT_THRESHOLDS_HOURS.get(system_id, (24, 24 * 7))
    if hours_since <= alive_h:
        return "alive"
    if hours_since <= stale_h:
        return "stale"
    return "dormant"


def _format_hours_ago(hours: float | None) -> str:
    """'hace 47min' | 'hace 3h' | 'hace 5d' | 'nunca'."""
    if hours is None:
        return "nunca"
    if hours < 1:
        mins = max(1, int(hours * 60))
        return f"hace {mins}min"
    if hours < 48:
        return f"hace {int(round(hours))}h"
    days = int(round(hours / 24))
    return f"hace {days}d"


def _verdict_ranker() -> dict:
    """Ranker: comparar weights actual vs baseline embebido en metadata."""
    out: dict = {
        "id": "ranker",
        "name": "Cómo prioriza resultados",
        "status": "dormant",
        "last_active_ts": None,
        "last_active_human": "nunca",
        "metric": None,
        "note": None,
    }
    # Last tune timestamp + estadística reciente
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM rag_tune"
            ).fetchone()
            last_ts = row[0] if row else None
            stats = conn.execute(
                "SELECT COUNT(*) AS total,"
                " SUM(CASE WHEN delta > 0 THEN 1 ELSE 0 END) AS positive"
                " FROM rag_tune WHERE ts >= datetime('now', '-7 days')"
            ).fetchone()
    except Exception:
        return out
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("ranker", hours_since)
    # Cuánto cambió: # weights modificados + total absolute delta
    try:
        if _RANKER_PATH.is_file():
            r = json.loads(_RANKER_PATH.read_text())
            weights = r.get("weights") or {}
            baseline = (r.get("metadata") or {}).get("baseline") or {}
            n_changed = 0
            total_delta = 0.0
            for k in set(list(weights.keys()) + list(baseline.keys())):
                cur = float(weights.get(k, 0) or 0)
                base = float(baseline.get(k, 0) or 0)
                if abs(cur - base) > 1e-6:
                    n_changed += 1
                    total_delta += abs(cur - base)
            if weights:
                out["metric"] = (
                    f"{n_changed} de {len(weights)} factores ajustados · "
                    f"cambio total = {total_delta:.3f}"
                )
    except Exception:
        pass
    if stats and stats[0]:
        total, positive = int(stats[0] or 0), int(stats[1] or 0)
        if total > 5 and positive < total * 0.1:
            out["note"] = (
                f"{total} runs últimos 7d, solo {positive} mejoraron — "
                "espacio de búsqueda saturado"
            )
    return out


def _verdict_eval() -> dict:
    """Eval: comparar hit5 últimos 7d vs ventana 7-30d para mostrar mejora."""
    out: dict = {
        "id": "eval", "name": "Calidad de los resultados",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            last = conn.execute("SELECT MAX(ts) FROM rag_eval_runs").fetchone()
            recent = conn.execute(
                "SELECT AVG(singles_hit5), AVG(chains_hit5), COUNT(*) "
                "FROM rag_eval_runs WHERE ts >= datetime('now', '-7 days')"
            ).fetchone()
            prior = conn.execute(
                "SELECT AVG(singles_hit5), COUNT(*) "
                "FROM rag_eval_runs "
                "WHERE ts < datetime('now', '-7 days') "
                "  AND ts >= datetime('now', '-30 days')"
            ).fetchone()
    except Exception:
        return out
    last_ts = last[0] if last else None
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("eval", hours_since)
    if recent and recent[0] is not None:
        cur_hit5 = float(recent[0])
        n_recent = int(recent[2] or 0)
        prior_hit5 = float(prior[0]) if prior and prior[0] is not None else None
        if prior_hit5 is not None and prior[1] and int(prior[1]) >= 5:
            delta_pts = (cur_hit5 - prior_hit5) * 100
            out_of_10 = round(cur_hit5 * 10, 1)
            out_of_10_text = (
                f"{int(out_of_10)}" if out_of_10 == int(out_of_10) else f"{out_of_10:.1f}"
            )
            if abs(delta_pts) < 0.1:
                trend = "igual que hace 30 días"
            elif delta_pts >= 0:
                trend = f"subió {delta_pts:.1f} puntos en 30 días"
            else:
                trend = f"bajó {abs(delta_pts):.1f} puntos en 30 días"
            out["metric"] = (
                f"Acierto: {out_of_10_text} de cada 10 ({n_recent} pruebas) · {trend}"
            )
        else:
            out_of_10 = round(cur_hit5 * 10, 1)
            out_of_10_text = (
                f"{int(out_of_10)}" if out_of_10 == int(out_of_10) else f"{out_of_10:.1f}"
            )
            out["metric"] = f"Acierto: {out_of_10_text} de cada 10 ({n_recent} pruebas)"
    return out


def _verdict_feedback() -> dict:
    out: dict = {
        "id": "feedback", "name": "Lo que evaluaste con 👍/👎",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            last = conn.execute("SELECT MAX(ts) FROM rag_feedback").fetchone()
            stats = conn.execute(
                "SELECT "
                " SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END), "
                " SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) "
                "FROM rag_feedback WHERE ts >= datetime('now', '-7 days')"
            ).fetchone()
    except Exception:
        return out
    last_ts = last[0] if last else None
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("feedback", hours_since)
    if stats:
        pos = int(stats[0] or 0)
        neg = int(stats[1] or 0)
        if pos + neg > 0:
            out["metric"] = f"{pos+neg} reacciones esta semana ({pos} 👍 / {neg} 👎)"
            if neg > pos * 1.5 and (pos + neg) >= 5:
                out["note"] = "Más 👎 que 👍 — convendría revisar la calidad de las respuestas"
    return out


def _verdict_behavior() -> dict:
    out: dict = {
        "id": "behavior", "name": "Cómo interactuás con los resultados",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            last = conn.execute("SELECT MAX(ts) FROM rag_behavior").fetchone()
            count = conn.execute(
                "SELECT COUNT(*) FROM rag_behavior "
                "WHERE ts >= datetime('now', '-7 days')"
            ).fetchone()
    except Exception:
        return out
    last_ts = last[0] if last else None
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("behavior", hours_since)
    if count and count[0]:
        out["metric"] = f"{int(count[0]):,} interacciones esta semana".replace(",", ".")
    return out


def _verdict_anticipatory() -> dict:
    out: dict = {
        "id": "anticipatory", "name": "Predicciones proactivas",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            last = conn.execute(
                "SELECT MAX(ts) FROM rag_anticipate_candidates"
            ).fetchone()
            stats = conn.execute(
                "SELECT COUNT(*), SUM(sent) FROM rag_anticipate_candidates "
                "WHERE ts >= datetime('now', '-2 days')"
            ).fetchone()
    except Exception:
        return out
    last_ts = last[0] if last else None
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("anticipatory", hours_since)
    if stats and stats[0]:
        total = int(stats[0] or 0)
        sent = int(stats[1] or 0)
        out["metric"] = f"{sent} de {total} predicciones enviadas en 48h"
        # Loop roto: hay candidates pero sent=0 → bajar a dormant aunque
        # haya update reciente, porque el feedback loop está cortado.
        if total >= 10 and sent == 0:
            out["status"] = "dormant"
            out["note"] = (
                "loop cortado: genera candidates pero no envía "
                "(probable: ambient.json missing o cap)"
            )
    return out


def _verdict_paraphrases() -> dict:
    out: dict = {
        "id": "paraphrases", "name": "Variantes de preguntas aprendidas",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_used_ts), COUNT(*), COALESCE(SUM(hit_count), 0) "
                "FROM rag_learned_paraphrases"
            ).fetchone()
    except Exception:
        return out
    last_ts = row[0] if row else None
    count = int(row[1] or 0) if row else 0
    hits = int(row[2] or 0) if row else 0
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("paraphrases", hours_since)
    if count > 0:
        out["metric"] = f"{count} variantes aprendidas · {hits} usos"
    else:
        out["note"] = "Hace falta entrenar las variantes desde el feedback positivo (`rag paraphrases train`)"
    return out


def _verdict_routing_rules() -> dict:
    out: dict = {
        "id": "routing_rules",
        "name": "Reglas de derivación de mensajes",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            rules = conn.execute(
                "SELECT MAX(promoted_at), "
                " COALESCE(SUM(active), 0), COUNT(*) "
                "FROM rag_routing_rules"
            ).fetchone()
            decisions = conn.execute(
                "SELECT COUNT(*) FROM rag_routing_decisions"
            ).fetchone()
    except Exception:
        return out
    last_unix = rules[0] if rules else None
    last_ts = (
        datetime.fromtimestamp(int(last_unix)).isoformat(timespec="seconds")
        if last_unix else None
    )
    active = int(rules[1] or 0) if rules else 0
    n_decisions = int(decisions[0] or 0) if decisions else 0
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("routing_rules", hours_since)
    decisiones_word = "decisión" if n_decisions == 1 else "decisiones"
    reglas_word = "regla activa" if active == 1 else "reglas activas"
    if active > 0:
        out["metric"] = f"{active} {reglas_word} · {n_decisions} {decisiones_word} registradas"
    elif n_decisions < 10:
        out["note"] = (
            f"Sin datos suficientes ({n_decisions} {decisiones_word}, hace falta ~50 "
            "para detectar patrones)"
        )
    return out


def _verdict_whisper_vocab() -> dict:
    out: dict = {
        "id": "whisper_vocab", "name": "Vocabulario para transcripciones",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(refreshed_at), COUNT(*) FROM rag_whisper_vocab"
            ).fetchone()
    except Exception:
        return out
    last_unix = row[0] if row else None
    count = int(row[1] or 0) if row else 0
    last_ts = (
        datetime.fromtimestamp(float(last_unix)).isoformat(timespec="seconds")
        if last_unix else None
    )
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("whisper_vocab", hours_since)
    if count > 0:
        out["metric"] = f"{count} términos en el vocabulario"
    return out


def _verdict_whisper_corrections() -> dict:
    out: dict = {
        "id": "whisper_corrections",
        "name": "Correcciones de transcripciones",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts), COUNT(*) FROM rag_audio_corrections"
            ).fetchone()
    except Exception:
        return out
    last_unix = row[0] if row else None
    count = int(row[1] or 0) if row else 0
    last_ts = (
        datetime.fromtimestamp(float(last_unix)).isoformat(timespec="seconds")
        if last_unix else None
    )
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("whisper_corrections", hours_since)
    if count > 0:
        out["metric"] = f"{count} correcciones manuales"
    else:
        out["note"] = "Sin transcripciones para corregir todavía"
    return out


def _verdict_score_calibration() -> dict:
    out: dict = {
        "id": "score_calibration",
        "name": "Confianza calibrada por fuente",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(trained_at), COUNT(*), "
                " COALESCE(SUM(n_pos), 0) "
                "FROM rag_score_calibration"
            ).fetchone()
    except Exception:
        return out
    last_ts = row[0] if row else None
    n_curves = int(row[1] or 0) if row else 0
    n_pos = int(row[2] or 0) if row else 0
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("score_calibration", hours_since)
    if n_curves > 0:
        fuentes_word = "fuente calibrada" if n_curves == 1 else "fuentes calibradas"
        out["metric"] = f"{n_curves} {fuentes_word} · {n_pos} ejemplos positivos"
    return out


def _verdict_contradictions() -> dict:
    out: dict = {
        "id": "contradictions",
        "name": "Detector de contradicciones",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts), COUNT(*) FROM rag_contradictions"
            ).fetchone()
            recent = conn.execute(
                "SELECT COUNT(*) FROM rag_contradictions "
                "WHERE ts >= datetime('now', '-7 days')"
            ).fetchone()
    except Exception:
        return out
    last_ts = row[0] if row else None
    total = int(row[1] or 0) if row else 0
    n_recent = int(recent[0] or 0) if recent else 0
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("contradictions", hours_since)
    if total > 0:
        total_fmt = f"{total:,}".replace(",", ".")
        out["metric"] = f"{total_fmt} detectadas en total · {n_recent} esta semana"
    return out


def _verdict_entities() -> dict:
    out: dict = {
        "id": "entities", "name": "Personas y lugares detectados",
        "status": "dormant", "last_active_ts": None,
        "last_active_human": "nunca", "metric": None, "note": None,
    }
    try:
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_seen_ts), COUNT(*), "
                " COALESCE(SUM(mention_count), 0) "
                "FROM rag_entities"
            ).fetchone()
    except Exception:
        return out
    last_unix = row[0] if row else None
    n_entities = int(row[1] or 0) if row else 0
    n_mentions = int(row[2] or 0) if row else 0
    last_ts = (
        datetime.fromtimestamp(float(last_unix)).isoformat(timespec="seconds")
        if last_unix else None
    )
    out["last_active_ts"] = last_ts
    hours_since = _hours_ago(last_ts)
    out["last_active_human"] = _format_hours_ago(hours_since)
    out["status"] = _verdict_status("entities", hours_since)
    if n_entities > 0:
        ent_fmt = f"{n_entities:,}".replace(",", ".")
        men_fmt = f"{n_mentions:,}".replace(",", ".")
        out["metric"] = f"{ent_fmt} personas/lugares · {men_fmt} menciones"
    return out


def verdict() -> dict:
    """Estado actual de los 11 sistemas de aprendizaje del RAG.

    Devuelve summary (counts por status) + lista de sistemas con
    status/last_active/metric/note. Cada función `_verdict_*` es silent-fail
    (try/except + retorna shape vacío) → un sistema roto no rompe el dashboard.

    Las funciones individuales hacen sus propios SQL reads (no comparten conn
    porque cada `_ragvec_state_conn()` ya hace pool internamente y prefiero
    aislamiento sobre micro-optimización: si una corrupta o lockea, las otras
    siguen).

    Performance: ~50-100ms warm (12 queries SQL muy chicas, todas con índice
    sobre `ts`/`trained_at`/etc).
    """
    systems = [
        _verdict_ranker(),
        _verdict_eval(),
        _verdict_feedback(),
        _verdict_behavior(),
        _verdict_anticipatory(),
        _verdict_paraphrases(),
        _verdict_routing_rules(),
        _verdict_whisper_vocab(),
        _verdict_whisper_corrections(),
        _verdict_score_calibration(),
        _verdict_contradictions(),
        _verdict_entities(),
    ]
    counts = {"alive": 0, "stale": 0, "dormant": 0}
    for s in systems:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    return {
        "summary": {
            "alive": counts["alive"],
            "stale": counts["stale"],
            "dormant": counts["dormant"],
            "total": len(systems),
        },
        "systems": systems,
    }
