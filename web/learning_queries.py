"""Backend del dashboard de aprendizaje (`/dashboard/learning`).

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
                f"SELECT AVG({column}) AS v "
                "FROM rag_eval_runs WHERE ts >= ? AND ts < ?",
                (cutoff_prev, cutoff_now),
            ).fetchone()
        v_now = float(row_now[0]) if row_now and row_now[0] is not None else 0.0
        n_now = int(row_now[1]) if row_now and row_now[1] is not None else 0
        v_prev = float(row_prev[0]) if row_prev and row_prev[0] is not None else 0.0
        delta = v_now - v_prev if n_now else 0.0
        return {
            "value": round(v_now, 4),
            "delta_30d": round(delta, 4),
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
        return {
            "value": round(ratio_now, 3),
            "delta_30d": round(ratio_now - ratio_prev, 3),
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
        return {
            "value": round(rate_now, 4),
            "delta_30d": round(rate_now - rate_prev, 4),
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
        return {
            "value": round(pct_now, 4),
            "delta_30d": round(pct_now - pct_prev, 4),
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
        for ts, delta, rolled_back, n_cases in rows:
            try:
                delta_pct = float(delta) * 100.0 if delta is not None else None
            except Exception:
                delta_pct = None
            series.append({
                "ts": ts,
                "delta_pct": round(delta_pct, 4) if delta_pct is not None else None,
                "rolled_back": bool(rolled_back) if rolled_back is not None else False,
                "n_cases": int(n_cases) if n_cases is not None else 0,
            })
        return {
            "series": series,
            "n_samples": len(series),
            "insufficient": _insufficient(len(series)),
        }

    return _sql_read_with_retry(
        _do, "learning_retrieval_tune_deltas_failed", default=_empty_series(),
    )


def _retrieval_latency_vs_score() -> dict:
    """Scatter plot: 1 punto por query reciente. Cap a 1000 más recientes
    para no inflar el JSON (~30KB worst case)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    def _do() -> dict:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT t_retrieve, top_score FROM rag_queries "
                "WHERE t_retrieve IS NOT NULL AND top_score IS NOT NULL "
                "ORDER BY id DESC LIMIT 1000"
            ).fetchall()
        points = [
            {"t_retrieve": float(t), "top_score": float(s)}
            for t, s in rows
            if isinstance(t, (int, float)) and isinstance(s, (int, float))
        ]
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
    """Bucketing por scope. El schema actual usa "turn" o NULL — no
    "answer"/"retrieval"/"both" como en el spec. Mapeamos:
        scope == "answer" → answer
        scope == "retrieval" → retrieval
        scope == "both" → both
        cualquier otro valor / NULL → unknown (incluye "turn")
    Las 4 keys están siempre presentes (default 0)."""
    from rag import _ragvec_state_conn, _sql_read_with_retry

    cutoff = _cutoff_iso(days)

    def _do() -> dict:
        out = {"answer": 0, "retrieval": 0, "both": 0, "unknown": 0}
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT scope, COUNT(*) FROM rag_feedback "
                "WHERE ts >= ? GROUP BY scope",
                (cutoff,),
            ).fetchall()
        for scope, n in rows:
            key = scope if scope in ("answer", "retrieval", "both") else "unknown"
            out[key] += int(n)
        return out

    return _sql_read_with_retry(
        _do,
        "learning_feedback_by_scope_failed",
        default={"answer": 0, "retrieval": 0, "both": 0, "unknown": 0},
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
