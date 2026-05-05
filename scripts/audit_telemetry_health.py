#!/usr/bin/env python3
"""Audit del estado de telemetry de obsidian-rag — diagnóstico data-first.

Agrega los 5 queries que usé en el audit 2026-04-24 para encontrar bugs
reales en producción (DB lock contention, readers degradados, cache
inoperante, test pollution, latency outliers). Output diseñado para
consumir antes de cualquier sesión de "auditá el sistema".

Uso:
    python scripts/audit_telemetry_health.py [--days N] [--json]

Sin argumentos imprime un reporte legible. `--json` emite el mismo
contenido como dict para encadenar con jq u otros agentes.

Por qué este script existe:

El audit del 2026-04-24 encontró 5 bugs reales (alerting roto, 1756
errores SQL silenciosos, readers sin retry, cache miss path sin
telemetry, test pollution) usando estos mismos queries SQL en ~5
segundos. Sin esta consolidación, cada audit re-tipea los queries en
sqlite3 a mano y se olvidan invariantes (ej. cruzar la curva diaria
de errores con `git log` para encontrar el commit que rompió cosas).

Memoria asociada:
- `feedback_telemetry_first_audit.md` (project memory): pattern
  general "data-first antes de leer código".
- `project_async_writer_package_invariant.md`: invariante de los 4
  cambios coordinados que requiere agregar async a un writer.
- `project_silent_log_unified_counter.md`: invariante del counter
  unificado para alerting.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path.home() / ".local/share/obsidian-rag"
TELEMETRY_DB = DATA_DIR / "ragvec/telemetry.db"
RAGVEC_DB = DATA_DIR / "ragvec/ragvec.db"
SQL_ERRORS_LOG = DATA_DIR / "sql_state_errors.jsonl"
SILENT_ERRORS_LOG = DATA_DIR / "silent_errors.jsonl"


def _open_db(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _audit_sql_errors(days: int, *, since_ts: str | None = None) -> dict:
    """Cuenta + distribuye los errores swallowed de los últimos N días.

    Cruza ambos logs (silent_errors + sql_state_errors). Devuelve top
    causas + curva diaria — para detectar cuándo empezó la degradación
    cruzando contra `git log --since=...`.

    `since_ts` (ISO 8601) es un cutoff adicional — útil para filtrar
    pollution histórica pre-fix y ver solo señal post-deploy. Si está
    presente, eventos con ts < since_ts se ignoran AUNQUE estén dentro
    de la ventana de days. Útil para `--since '2026-04-24T17:53:00'`
    (cuando deployó la fixture de aislamiento de logs).
    """
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_iso = cutoff.isoformat(timespec="seconds")
    effective_cutoff = max(cutoff_iso, since_ts) if since_ts else cutoff_iso
    out = {
        "total_errors": 0,
        "by_event": Counter(),
        "by_day": Counter(),
        "by_log_file": {"sql_state": 0, "silent_errors": 0},
        "test_pollution_hits": 0,
        "files_missing": [],
        "since_ts_applied": since_ts,
    }
    for log_path, label in (
        (SQL_ERRORS_LOG, "sql_state"),
        (SILENT_ERRORS_LOG, "silent_errors"),
    ):
        if not log_path.is_file():
            out["files_missing"].append(str(log_path))
            continue
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("ts", "")
                if ts < effective_cutoff:
                    continue
                out["total_errors"] += 1
                out["by_log_file"][label] += 1
                day = ts[:10] if len(ts) >= 10 else "unknown"
                out["by_day"][day] += 1
                event = rec.get("event") or rec.get("where") or "unknown"
                out["by_event"][event] += 1
                # Test pollution: production logs deberían tener 0 entries
                # con `test_tag` event. Pre-fix audit 2026-04-24 había
                # 161; post-fix conftest los aísla a tmp.
                if "test" in event.lower():
                    out["test_pollution_hits"] += 1
    return out


def _audit_query_latency(conn: sqlite3.Connection, days: int) -> dict:
    """Distribución de latencia por cmd + outliers >30s en los últimos N días.

    El `_DEEP_MAX_SECONDS=30s` cap (post 2026-04-22) garantiza que ningún
    retrieve real exceda eso. Si hay outliers post-cap, señal de bug en
    deep_retrieve o un caller que bypassea el guard.
    """
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    rows = conn.execute(
        """
        SELECT cmd,
               COUNT(*) as n,
               ROUND(AVG(t_retrieve), 2) as avg_retrieve,
               MAX(t_retrieve) as max_retrieve,
               ROUND(AVG(t_gen), 2) as avg_gen,
               MAX(t_gen) as max_gen
        FROM rag_queries
        WHERE ts >= ? AND cmd IS NOT NULL
        GROUP BY cmd
        ORDER BY n DESC
        LIMIT 15
        """,
        (cutoff_iso,),
    ).fetchall()
    outliers = conn.execute(
        """
        SELECT ts, cmd, substr(q, 1, 50) as q, t_retrieve, t_gen
        FROM rag_queries
        WHERE ts >= ?
          AND (t_retrieve > 30 OR t_gen > 60)
        ORDER BY (COALESCE(t_retrieve,0) + COALESCE(t_gen,0)) DESC
        LIMIT 10
        """,
        (cutoff_iso,),
    ).fetchall()
    return {
        "by_cmd": [dict(r) for r in rows],
        "outliers": [dict(r) for r in outliers],
        "outliers_count": len(outliers),
    }


def _audit_cache_health(conn: sqlite3.Connection, days: int) -> dict:
    """Distribución de cache_probe en `rag_queries.extra_json`.

    Sin esto el `rag cache stats` puede mentir — pre-fix 2026-04-24 el
    miss path NO loggeaba cache_probe → 998 web queries quedaban fuera
    del cache stats que solo veía 5 eligible.
    """
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    rows = conn.execute(
        """
        SELECT
          json_extract(extra_json, '$.cache_probe.result') as result,
          json_extract(extra_json, '$.cache_probe.reason') as reason,
          COUNT(*) as n
        FROM rag_queries
        WHERE ts >= ? AND cmd = 'web'
        GROUP BY result, reason
        ORDER BY n DESC
        """,
        (cutoff_iso,),
    ).fetchall()
    cache_total = conn.execute(
        "SELECT COUNT(*) FROM rag_response_cache"
    ).fetchone()[0]
    return {
        "by_probe": [dict(r) for r in rows],
        "cache_table_rows": cache_total,
    }


def check_anticipate_health(conn: sqlite3.Connection, days: int = 7) -> dict:
    """Health check del Anticipatory Agent (rag_anticipate_candidates).

    Detecta automáticamente las cuatro patologías que importan en prod:

    1. **Send rate global bajo (<5%)** con suficiente muestra (≥30 rows
       evaluated) → status="degraded". Significa que el agente está
       evaluando candidates pero el orchestrator descarta casi todos
       (threshold mal calibrado, dedup demasiado agresivo, quiet hours
       mal seteado).
    2. **Última emit muy vieja (>24h)** → status="stale". El daemon
       parece down — ningún signal disparó en >1 día. El nombre
       "last_emit_age_hours" sigue la convención del Anticipatory
       Agent: una "emit" es una row con sent=1.
    3. **Signals "silent"** (≥1 evaluated en la ventana, 0 emits) — la
       signal corre pero nunca dispara. Threshold mal puesto, signal
       rota, o feature deshabilitada. Se loguea como issue per-signal.
    4. **Signals "noisy"** (>10 emits con avg_score <0.3) — la signal
       dispara mucho con poca confianza. Likely false positive farm —
       hay que subir el threshold o agregar un dedup más estricto.

    Ver `# ── ANTICIPATORY AGENT ──` en rag.py para el diseño del
    sistema de signals + scoring + threshold.

    Returns un dict JSON-serializable. Status mapping:
    - "unknown": tabla missing o 0 rows totales (agente nunca corrió)
    - "stale":   last_emit_age_hours > 24 (daemon parece down)
    - "degraded": send_rate <0.05 con ≥30 evaluated en la ventana
    - "healthy": caso normal
    """
    out: dict = {
        "status": "unknown",
        "window_days": days,
        "total_evaluated": 0,
        "total_sent": 0,
        "send_rate": 0.0,
        "last_emit_age_hours": None,
        "issues": [],
        "by_signal": {},
    }

    # 1) Tabla existe?
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_anticipate_candidates'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["issues"].append(f"sql_error: {exc!r}")
        return out
    if exists is None:
        out["issues"].append("table missing: rag_anticipate_candidates")
        return out

    # 2) ¿Alguna vez corrió? (separado del window check para distinguir
    # "nunca corrió" de "no corrió en los últimos N días").
    total_ever = conn.execute(
        "SELECT COUNT(*) FROM rag_anticipate_candidates"
    ).fetchone()[0]
    if not total_ever:
        out["issues"].append(
            "0 rows in rag_anticipate_candidates — agent never ran"
        )
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    # 3) Totales en la ventana
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(sent), 0) "
        "FROM rag_anticipate_candidates WHERE ts >= ?",
        (cutoff_iso,),
    ).fetchone()
    total_evaluated = int(row[0] or 0)
    total_sent = int(row[1] or 0)
    out["total_evaluated"] = total_evaluated
    out["total_sent"] = total_sent
    out["send_rate"] = (
        round(total_sent / total_evaluated, 4) if total_evaluated else 0.0
    )

    # 4) Última emit (sent=1) — across all time, así detectamos un daemon
    # down aunque la ventana sea chica.
    last_emit_ts = conn.execute(
        "SELECT MAX(ts) FROM rag_anticipate_candidates WHERE sent = 1"
    ).fetchone()[0]
    if last_emit_ts:
        try:
            last_dt = datetime.fromisoformat(last_emit_ts)
            age_h = (datetime.now() - last_dt).total_seconds() / 3600.0
            out["last_emit_age_hours"] = round(age_h, 2)
        except (ValueError, TypeError):
            # ts mal formado — silent fail, dejamos None y seguimos.
            pass

    # 5) Per-signal breakdown — emits, avg_score, send_rate y status
    sig_rows = conn.execute(
        "SELECT kind, COUNT(*), COALESCE(SUM(sent), 0), AVG(score), "
        "       MAX(CASE WHEN sent = 1 THEN ts END) "
        "FROM rag_anticipate_candidates "
        "WHERE ts >= ? "
        "GROUP BY kind "
        "ORDER BY COUNT(*) DESC",
        (cutoff_iso,),
    ).fetchall()

    by_signal: dict = {}
    for r in sig_rows:
        kind = r[0]
        n = int(r[1] or 0)
        emits = int(r[2] or 0)
        avg_score = float(r[3] or 0.0)
        last_ts_signal = r[4]

        if emits == 0:
            sig_status = "silent"
        elif emits > 10 and avg_score < 0.3:
            sig_status = "noisy"
        else:
            sig_status = "healthy"
            if last_ts_signal:
                try:
                    age_signal = (
                        datetime.now()
                        - datetime.fromisoformat(last_ts_signal)
                    ).total_seconds() / 3600.0
                    if age_signal > 24:
                        sig_status = "stale"
                except (ValueError, TypeError):
                    pass

        by_signal[kind] = {
            "evaluated": n,
            "emits": emits,
            "avg_score": round(avg_score, 4),
            "send_rate": round(emits / n, 4) if n else 0.0,
            "status": sig_status,
        }

        if sig_status == "silent":
            out["issues"].append(
                f"silent signal: {kind} (0 emits / {n} evaluated in last {days}d)"
            )
        elif sig_status == "noisy":
            out["issues"].append(
                f"noisy signal: {kind} ({emits} emits, avg_score={avg_score:.2f} <0.3)"
            )

    out["by_signal"] = by_signal

    # 6) Status global — orden de precedencia: stale > degraded > healthy
    if (
        out["last_emit_age_hours"] is not None
        and out["last_emit_age_hours"] > 24
    ):
        out["status"] = "stale"
        out["issues"].append(
            f"last emit {out['last_emit_age_hours']:.1f}h ago — daemon may be down"
        )
    elif total_evaluated >= 30 and out["send_rate"] < 0.05:
        out["status"] = "degraded"
        out["issues"].append(
            f"low send rate: {out['send_rate'] * 100:.1f}% with "
            f"{total_evaluated} evaluated (threshold: <5% with ≥30 rows)"
        )
    else:
        out["status"] = "healthy"

    return out


def check_retrieval_health(conn: sqlite3.Connection, days: int = 7) -> dict:
    """Health check del pipeline de retrieval (cache + ranker + latency).

    Audit 2026-04-25 R2-Telemetry #5: agregamos este check porque el
    audit detectó que la degradation silenciosa del retrieval (cache
    invalidation suelta, ranker devolviendo basura, latency creep) pasa
    desapercibida hasta que el user la experimenta. Mismo patrón que
    `check_anticipate_health` — recibe conn, calcula thresholds,
    devuelve dict JSON-serializable.

    Detecta cuatro patologías:

    1. **stale**: 0 queries en la ventana → system idle o corrupto.
    2. **cache hit rate < 50%** (con muestra ≥20 web queries que
       loguearon `cache_probe`): cache invalidation suelta o config
       rota — el LRU cosine no está devolviendo hits en queries que
       repiten.
    3. **median top_score < 0.4**: ranker degradado. Si la mediana de
       los scores top-1 es bajísima, el embed/rerank está devolviendo
       contexto irrelevante para casi todas las queries.
    4. **p95 t_retrieve > baseline × 1.3** (baseline 1500ms → trip a
       >1950ms): latency creep — alguno de bm25/sem/rerank se estancó.

    Reglas adicionales (no `degraded` por sí solas, solo informativas):

    - **NLI grounding bajo** (`claims_supported / claims_total < 0.5`)
      cuando claims_total > 0. Marca degraded si está pero no hace
      degraded el caso `claims_total == 0` — el NLI puede no haber
      corrido y eso no es per se un bug del retrieval.

    Returns dict con shape:

        {
            "status": "healthy" | "degraded" | "stale" | "unknown",
            "issues": ["..."],
            "details": {
                "queries_count": N,
                "cache_hit_rate_pct": X.X | None,
                "p95_retrieve_ms": Y | None,
                "median_top_score": Z.Z | None,
                "nli_supported_rate": W | None,  # opcional
            }
        }

    TODO (audit 2026-04-25 R2-Telemetry #5): el threshold de cache hit
    rate (50%) y p95 baseline (1500ms) son numbers tomados de la
    distribución observada en abril 2026. Cuando subamos el ranker o
    cambiemos modelo de embeddings, estos baselines necesitan revisarse
    — convendría persistirlos en una tabla `rag_telemetry_baselines`
    en lugar de hardcodear acá.
    """
    out: dict = {
        "status": "unknown",
        "issues": [],
        "details": {
            "queries_count": 0,
            "cache_hit_rate_pct": None,
            "p95_retrieve_ms": None,
            "median_top_score": None,
        },
    }

    # 1) Tabla rag_queries existe?
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_queries'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["issues"].append(f"sql_error: {exc!r}")
        return out
    if exists is None:
        out["issues"].append("table missing: rag_queries")
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    # 2) Total queries en la ventana — los 3 cmds que disparan retrieve
    # real ('chat' CLI, 'query' CLI, 'web' endpoint principal). Los
    # sub-cmds 'web.chat.*' los excluímos porque son re-logs de stages
    # post-retrieval (metachat/degenerate/low_conf_bypass) y no
    # representan un retrieve nuevo.
    queries_count = conn.execute(
        "SELECT COUNT(*) FROM rag_queries "
        "WHERE ts >= ? AND cmd IN ('chat', 'query', 'web')",
        (cutoff_iso,),
    ).fetchone()[0]
    out["details"]["queries_count"] = queries_count

    if queries_count == 0:
        out["status"] = "stale"
        out["issues"].append(
            f"0 retrieval queries en ventana de {days}d (system idle o corrupto)"
        )
        return out

    # 3) Cache hit rate — leemos `extra_json.cache_hit` que el web
    # endpoint loguea desde el fix 2026-04-24 (commit 3dcbe81). Antes
    # del fix solo se logueaba en el hit path; ahora también el miss.
    # Solo evaluamos la regla si tenemos una muestra mínima — sino los
    # primeros días post-deploy mostrarían "cache hit rate 0% degraded"
    # cuando en realidad recién empezó a poblarse.
    #
    # 2026-05-01: excluimos del denominador las queries `skipped`
    # (con history, multi-vault, forced/propose intent — todas
    # legítimamente NO elegibles para cache). Pre-fix el rate quedaba
    # en 0% en todo deploy donde el chat es multi-turn (caso WhatsApp
    # listener: ~90% de queries tienen history). El nuevo numerator
    # son los hits sobre queries que SÍ pasaron por el cache layer
    # (result IN ('hit', 'miss')).
    cache_rows = conn.execute(
        "SELECT json_extract(extra_json, '$.cache_probe.result') "
        "FROM rag_queries "
        "WHERE ts >= ? AND cmd = 'web' "
        "  AND json_extract(extra_json, '$.cache_probe.result') IS NOT NULL",
        (cutoff_iso,),
    ).fetchall()
    cache_total_eligible = sum(1 for r in cache_rows if r[0] in ("hit", "miss"))
    cache_total_skipped = sum(1 for r in cache_rows if r[0] == "skipped")
    cache_hits = sum(1 for r in cache_rows if r[0] == "hit")
    cache_hit_rate_pct: float | None = None
    if cache_total_eligible > 0:
        cache_hit_rate_pct = round(100.0 * cache_hits / cache_total_eligible, 1)
        out["details"]["cache_hit_rate_pct"] = cache_hit_rate_pct
        out["details"]["cache_eligible_count"] = cache_total_eligible
        out["details"]["cache_skipped_count"] = cache_total_skipped
    # cache_total se mantiene para back-compat de la regla de issues.
    cache_total = cache_total_eligible

    # 4) p95 t_retrieve (en ms). Nearest-rank percentile — para muestras
    # chicas no usamos interpolación (sería sobreingeniería). Convertimos
    # de segundos a ms acá.
    retrieve_rows = conn.execute(
        "SELECT t_retrieve FROM rag_queries "
        "WHERE ts >= ? AND cmd IN ('chat', 'query', 'web') "
        "  AND t_retrieve IS NOT NULL "
        "ORDER BY t_retrieve ASC",
        (cutoff_iso,),
    ).fetchall()
    p95_retrieve_ms: float | None = None
    if retrieve_rows:
        n = len(retrieve_rows)
        # nearest-rank: ceil(0.95 * n) - 1 con guard de bounds
        idx = max(0, min(n - 1, int(0.95 * n) - 1 if 0.95 * n == int(0.95 * n) else int(0.95 * n)))
        p95_retrieve_ms = round(retrieve_rows[idx][0] * 1000.0, 1)
        out["details"]["p95_retrieve_ms"] = p95_retrieve_ms

    # 5) Median top_score — mediana clásica (índice n//2 sobre lista
    # ordenada). Si hay 0 rows con top_score, queda None.
    score_rows = conn.execute(
        "SELECT top_score FROM rag_queries "
        "WHERE ts >= ? AND cmd IN ('chat', 'query', 'web') "
        "  AND top_score IS NOT NULL "
        "ORDER BY top_score ASC",
        (cutoff_iso,),
    ).fetchall()
    median_top_score: float | None = None
    if score_rows:
        n = len(score_rows)
        median_top_score = round(score_rows[n // 2][0], 3)
        out["details"]["median_top_score"] = median_top_score

    # 6) NLI grounding — claims_supported/claims_total. En prod actualmente
    # `extra_json.grounding_summary` está mayormente null (NLI corre
    # solo en ciertos paths). Si no hay claims, este check queda no-op.
    nli_rows = conn.execute(
        "SELECT json_extract(extra_json, '$.grounding_summary.claims_total'), "
        "       json_extract(extra_json, '$.grounding_summary.supported') "
        "FROM rag_queries "
        "WHERE ts >= ? AND cmd IN ('chat', 'query', 'web') "
        "  AND json_extract(extra_json, '$.grounding_summary.claims_total') > 0",
        (cutoff_iso,),
    ).fetchall()
    nli_claims_total = sum(int(r[0] or 0) for r in nli_rows)
    nli_claims_supported = sum(int(r[1] or 0) for r in nli_rows)
    nli_supported_rate: float | None = None
    if nli_claims_total > 0:
        nli_supported_rate = round(nli_claims_supported / nli_claims_total, 3)
        out["details"]["nli_supported_rate"] = nli_supported_rate

    # 7) Aplicar reglas de degradation
    issues: list[str] = []
    P95_BASELINE_MS = 1500.0
    P95_THRESHOLD_MS = P95_BASELINE_MS * 1.3  # 1950ms

    # Cache hit rate threshold: bajo a 10% (sobre queries elegibles).
    # 50% era irrealista — la mayoría de queries semánticas son únicas
    # (paráfrasis del user nunca pegan exactamente). 10% sobre queries
    # elegibles indica al menos algo de re-uso. Si el cache quedó
    # inservible (queries todas únicas), 0% por mucho tiempo dispara
    # alerta — pero el threshold real lo debería ajustar el operador
    # mirando varios días.
    if (
        cache_hit_rate_pct is not None
        and cache_total >= 50
        and cache_hit_rate_pct < 10.0
    ):
        issues.append(
            f"cache hit rate {cache_hit_rate_pct:.1f}% < 10% threshold "
            f"(n_eligible={cache_total}, n_skipped={cache_total_skipped})"
        )
    if median_top_score is not None and median_top_score < 0.4:
        issues.append(
            f"median top_score {median_top_score:.2f} < 0.4 (ranker degradado)"
        )
    if p95_retrieve_ms is not None and p95_retrieve_ms > P95_THRESHOLD_MS:
        issues.append(
            f"p95 t_retrieve {p95_retrieve_ms:.0f}ms (baseline "
            f"{P95_BASELINE_MS:.0f}ms × 1.3 = {P95_THRESHOLD_MS:.0f}ms)"
        )
    if nli_supported_rate is not None and nli_supported_rate < 0.5:
        issues.append(
            f"NLI grounding bajo: {nli_claims_supported}/{nli_claims_total} "
            f"claims supported ({nli_supported_rate * 100:.0f}% < 50%)"
        )

    out["issues"] = issues
    out["status"] = "degraded" if issues else "healthy"
    return out


def check_chat_health(conn: sqlite3.Connection, days: int = 7) -> dict:
    """Health check del pipeline de chat / LLM (post-retrieval).

    Audit 2026-04-25 R2-Telemetry #5: complementa
    `check_retrieval_health` cubriendo el lado generation. Detecta
    degradation que `check_retrieval_health` no ve — por ejemplo, el
    ranker puede estar perfecto pero el LLM tardar 6s en generar (cap
    de t_gen creció), o el critique nunca dispara porque alguien rompió
    el hook al refactorear.

    Detecta:

    1. **stale**: 0 chats en ventana → endpoint /chat caído o nadie
       lo está usando.
    2. **p95 t_gen > baseline × 1.3** (baseline 3000ms → trip a
       >3900ms): latency creep en generation.
    3. **critique_fired_rate == 0** con muestra ≥20 → critique nunca
       dispara, hook roto. (Nota: critique_fired = 1 cuando el critique
       loop detectó algo y modificó la respuesta; rate sano ronda
       5-15%.)
    4. **refusal_rate > 50%**: el bot devuelve `web.chat.degenerate`
       canned reply en >50% de las queries → o routing roto, o el LLM
       perdió capacidad, o el dataset cambió drásticamente. Solo se
       evalúa con muestra ≥20 chats para evitar ruido en days bajos.

    TODO (audit 2026-04-25 R2-Telemetry #5): no hay un flag explícito
    `error: true` en `extra_json` para LLM crashes — los timeouts se
    propagan a `silent_errors.jsonl` con `where=...` que no incluye
    `cmd`, así que atribuir errores a chat vs otros sub-sistemas
    requiere parsear el JSONL aparte. Por ahora `error_rate_pct` queda
    out de las reglas de `degraded` (no se calcula). Si en R3 agregamos
    un flag dedicado `extra_json.error` o `extra_json.exc_type`,
    levantamos esta TODO y agregamos la regla `error_rate > 5%`.

    Returns dict con shape:

        {
            "status": "healthy" | "degraded" | "stale" | "unknown",
            "issues": ["..."],
            "details": {
                "chats_count": N,
                "p95_gen_ms": Y | None,
                "critique_fired_rate": F | None,
                "refusal_rate_pct": R | None,
            }
        }
    """
    out: dict = {
        "status": "unknown",
        "issues": [],
        "details": {
            "chats_count": 0,
            "p95_gen_ms": None,
            "critique_fired_rate": None,
            "refusal_rate_pct": None,
        },
    }

    # 1) Tabla rag_queries existe?
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_queries'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["issues"].append(f"sql_error: {exc!r}")
        return out
    if exists is None:
        out["issues"].append("table missing: rag_queries")
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    # 2) Total chats — agarramos `chat` (CLI), `web` (endpoint
    # principal del web server) y todos los sub-cmds `web.chat.%`
    # (metachat / degenerate / low_conf_bypass / cached_semantic).
    # Ver `web/server.py:7119` para la convención de naming.
    chats_count = conn.execute(
        "SELECT COUNT(*) FROM rag_queries "
        "WHERE ts >= ? "
        "  AND (cmd = 'chat' OR cmd = 'web' OR cmd LIKE 'web.chat.%')",
        (cutoff_iso,),
    ).fetchone()[0]
    out["details"]["chats_count"] = chats_count

    if chats_count == 0:
        out["status"] = "stale"
        out["issues"].append(
            f"0 chats en ventana de {days}d (endpoint idle o caído)"
        )
        return out

    # 3) p95 t_gen (ms) — solo sobre chats que efectivamente generaron
    # (t_gen > 0). Cache hits y degenerate replies tienen t_gen NULL y
    # no entran al cálculo de latency.
    gen_rows = conn.execute(
        "SELECT t_gen FROM rag_queries "
        "WHERE ts >= ? "
        "  AND (cmd = 'chat' OR cmd = 'web' OR cmd LIKE 'web.chat.%') "
        "  AND t_gen IS NOT NULL AND t_gen > 0 "
        "ORDER BY t_gen ASC",
        (cutoff_iso,),
    ).fetchall()
    p95_gen_ms: float | None = None
    if gen_rows:
        n = len(gen_rows)
        idx = max(0, min(n - 1, int(0.95 * n) - 1 if 0.95 * n == int(0.95 * n) else int(0.95 * n)))
        p95_gen_ms = round(gen_rows[idx][0] * 1000.0, 1)
        out["details"]["p95_gen_ms"] = p95_gen_ms

    # 4) Critique fired rate — sobre chats con `critique_fired` no-null
    # (la columna admite NULL para queries pre-feature). Si el rate
    # cae a 0 con muestra significativa, el hook se rompió.
    crit_rows = conn.execute(
        "SELECT critique_fired FROM rag_queries "
        "WHERE ts >= ? "
        "  AND (cmd = 'chat' OR cmd = 'web' OR cmd LIKE 'web.chat.%') "
        "  AND critique_fired IS NOT NULL",
        (cutoff_iso,),
    ).fetchall()
    crit_total = len(crit_rows)
    crit_fired = sum(1 for r in crit_rows if r[0])
    critique_fired_rate: float | None = None
    if crit_total > 0:
        critique_fired_rate = round(crit_fired / crit_total, 3)
        out["details"]["critique_fired_rate"] = critique_fired_rate

    # 5) Refusal rate — `web.chat.degenerate` es la canned reply
    # "no puedo responder eso, reformulá". Si supera 50% de los chats
    # totales, el bot se está rindiendo demasiado.
    refusals = conn.execute(
        "SELECT COUNT(*) FROM rag_queries "
        "WHERE ts >= ? AND cmd = 'web.chat.degenerate'",
        (cutoff_iso,),
    ).fetchone()[0]
    refusal_rate_pct = (
        round(100.0 * refusals / chats_count, 1) if chats_count else 0.0
    )
    out["details"]["refusal_rate_pct"] = refusal_rate_pct

    # 6) Reglas de degradation
    issues: list[str] = []
    P95_BASELINE_MS = 3000.0
    P95_THRESHOLD_MS = P95_BASELINE_MS * 1.3  # 3900ms

    if p95_gen_ms is not None and p95_gen_ms > P95_THRESHOLD_MS:
        issues.append(
            f"p95 t_gen {p95_gen_ms:.0f}ms (baseline "
            f"{P95_BASELINE_MS:.0f}ms × 1.3 = {P95_THRESHOLD_MS:.0f}ms)"
        )
    if (
        critique_fired_rate is not None
        and crit_total >= 20
        and critique_fired_rate == 0.0
    ):
        issues.append(
            f"critique nunca dispara: 0/{crit_total} chats con "
            f"critique_fired=1 (hook roto?)"
        )
    if refusal_rate_pct > 50.0 and chats_count >= 20:
        issues.append(
            f"refusal rate {refusal_rate_pct:.1f}% > 50% — bot se rinde "
            f"demasiado ({refusals}/{chats_count} `web.chat.degenerate`)"
        )

    out["issues"] = issues
    out["status"] = "degraded" if issues else "healthy"
    return out


def _audit_feedback_corrective_gap(conn: sqlite3.Connection) -> dict:
    """Gap de corrective_path en feedbacks negativos — gate del LoRA fine-tune.

    El fine-tune del reranker (GC#2.C en CLAUDE.md) requiere ≥20 filas con
    `corrective_path` limpio en `rag_feedback`. Sin eso el gate de
    `scripts/finetune_reranker.py` aborta con exit 5. Este check dice
    exactamente cuántos negativos tienen CP y cuántos faltan para abrir el gate.

    Columnas de output:

    - total_neg: feedbacks con rating < 0
    - has_cp: de esos, cuántos tienen corrective_path no-null y no-vacío
    - missing_cp: total_neg - has_cp
    - pct_covered: has_cp / total_neg * 100 (0.0 si total_neg == 0)
    - gate_threshold: 20 (hardcoded, mismo que RAG_FINETUNE_MIN_CORRECTIVES)
    - gate_open: has_cp >= gate_threshold
    - rows_to_close_gate: max(0, gate_threshold - has_cp)
    """
    GATE_THRESHOLD = 20
    out: dict = {
        "total_neg": 0,
        "has_cp": 0,
        "missing_cp": 0,
        "pct_covered": 0.0,
        "gate_threshold": GATE_THRESHOLD,
        "gate_open": False,
        "rows_to_close_gate": GATE_THRESHOLD,
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_feedback'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_feedback"
        return out

    row = conn.execute(
        """
        SELECT
          COUNT(*) AS total_neg,
          SUM(CASE WHEN json_extract(extra_json, '$.corrective_path') IS NOT NULL
                    AND json_extract(extra_json, '$.corrective_path') <> '' THEN 1 ELSE 0 END) AS has_cp
        FROM rag_feedback
        WHERE rating < 0
        """
    ).fetchone()

    total_neg = int(row[0] or 0)
    has_cp = int(row[1] or 0)
    missing_cp = total_neg - has_cp
    pct_covered = round(has_cp / total_neg * 100, 2) if total_neg > 0 else 0.0
    gate_open = has_cp >= GATE_THRESHOLD
    rows_to_close = max(0, GATE_THRESHOLD - has_cp)

    out.update({
        "total_neg": total_neg,
        "has_cp": has_cp,
        "missing_cp": missing_cp,
        "pct_covered": pct_covered,
        "gate_open": gate_open,
        "rows_to_close_gate": rows_to_close,
    })
    return out


def _audit_harvest_candidates(conn: sqlite3.Connection, days: int = 7) -> dict:
    """Queries recientes con score bajo que todavía no tienen thumbs.

    Detecta el input listo para `rag feedback harvest`: queries con
    top_score entre CONFIDENCE_RERANK_MIN (0.015) y 0.35 (zona de
    respuesta dudosa, no refuse), sin feedback explícito en
    `rag_feedback`. Son los candidatos más valiosos para labeling activo
    porque el sistema respondió algo pero sin confianza — y sin
    corrección humana el ranker nunca aprende.

    Join contra `rag_feedback` por `lower(q)` (no hay columna `session`
    en `rag_feedback`). Un query que aparece N veces en `rag_queries`
    pero tiene UNA entrada en `rag_feedback` queda fuera — el usuario
    ya lo puntuó alguna vez con esa misma query.

    Shape de output:

        {
            "window_days": 7,
            "count": 58,          # total sin LIMIT
            "top_candidates": [
                {
                    "query_id": 4374,
                    "q": "cancelar el del médico",
                    "top_score": 0.04,
                    "ts": "2026-04-28T19:49:55",
                    "cmd": "web",
                },
                ...
            ],
            "alert": True,        # True si count >= 10
            "harvest_command": "rag feedback harvest --since 7 --limit 20 --confidence-below 0.35",
        }

    Thresholds:

    - `top_score >= 0.015`: abajo de esto el sistema se niega a
      responder (CONFIDENCE_RERANK_MIN). Sin sentido labelear refuses.
    - `top_score <= 0.35`: arriba de esto la confianza es aceptable —
      el labeling aporta menos. 0.35 coincide con
      `CONFIDENCE_DEEP_THRESHOLD` (0.10) × 3.5; elegido para capturar
      la zona que dispara `deep_retrieve` pero termina con score bajo.
    - `alert=True` desde count >= 10: arbitrario pero accionable.
      10 candidatos sin label en 7 días = una sesión de harvest de ~5min.
    """
    SCORE_MIN = 0.015
    SCORE_MAX = 0.35
    ALERT_THRESHOLD = 10
    LIMIT = 20
    CMDS = ("web", "query", "serve.chat")

    out: dict = {
        "window_days": days,
        "count": 0,
        "top_candidates": [],
        "alert": False,
        "harvest_command": (
            f"rag feedback harvest --since {days} --limit {LIMIT} "
            f"--confidence-below {SCORE_MAX}"
        ),
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_queries'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_queries"
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    cmds_placeholder = ", ".join("?" * len(CMDS))

    # Conteo total sin LIMIT (separado para no contaminar el shape con
    # los 20 rows del top_candidates).
    count_row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM rag_queries rq
        LEFT JOIN rag_feedback rf
          ON lower(rf.q) = lower(rq.q)
        WHERE rq.ts >= ?
          AND rq.top_score BETWEEN ? AND ?
          AND rq.cmd IN ({cmds_placeholder})
          AND rq.q != ''
          AND rf.id IS NULL
        """,
        (cutoff_iso, SCORE_MIN, SCORE_MAX, *CMDS),
    ).fetchone()
    total_count = int(count_row[0] or 0)
    out["count"] = total_count
    out["alert"] = total_count >= ALERT_THRESHOLD

    # Top-N candidatos ordenados por score ascendente (los menos
    # confiables primero) y ts descendente (más recientes en empate).
    rows = conn.execute(
        f"""
        SELECT
          rq.id AS query_id,
          rq.q,
          ROUND(rq.top_score, 4) AS top_score,
          rq.ts,
          rq.cmd
        FROM rag_queries rq
        LEFT JOIN rag_feedback rf
          ON lower(rf.q) = lower(rq.q)
        WHERE rq.ts >= ?
          AND rq.top_score BETWEEN ? AND ?
          AND rq.cmd IN ({cmds_placeholder})
          AND rq.q != ''
          AND rf.id IS NULL
        ORDER BY rq.top_score ASC, rq.ts DESC
        LIMIT {LIMIT}
        """,
        (cutoff_iso, SCORE_MIN, SCORE_MAX, *CMDS),
    ).fetchall()

    out["top_candidates"] = [
        {
            "query_id": r[0],
            "q": r[1],
            "top_score": r[2],
            "ts": r[3],
            "cmd": r[4],
        }
        for r in rows
    ]
    return out




def _audit_cross_source_single_source(
    telemetry_conn: sqlite3.Connection,
    ragvec_conn: sqlite3.Connection,
    days: int = 7,
) -> dict:
    """Detector: queries con múltiples sources DISTINTOS en top-k pero dominio de uno.

    **Blocker conocido (2026-05-04)**: `rag_queries.filters_json` siempre está vacío,
    así que no sabemos QUÉ sources pidió el user vía CLI flags (`--source X,Y`).

    **Workaround**: miramos los sources REALES en los chunks devueltos (via `paths_json`
    → lookup en meta_obsidian_notes_v11). Si hay ≥2 sources DISTINTOS pero uno domina
    >80% de los resultados, es probable que el user pidiera múltiples sources pero el
    ingester de los otros está roto/ausente. Heurística imperfecta, pero accionable.

    Returns:
        {
            "window_days": 7,
            "queries_analyzed": N,
            "imbalanced_queries": [
                {
                    "ts": "...",
                    "q": "...",
                    "sources": {"vault": 4, "whatsapp": 1},
                    "dominant_source": "vault",
                    "imbalance_pct": 80.0,
                }
            ],
            "count": M,
            "alert": bool,
            "blocker_message": "...",
            "suggestion": "Verificar si los ingesters de fuentes minoritarias corrieron..."
        }
    """
    out: dict = {
        "window_days": days,
        "queries_analyzed": 0,
        "imbalanced_queries": [],
        "count": 0,
        "alert": False,
        "blocker_message": (
            "⚠️  BLOCKER OPERACIONAL: `rag_queries.filters_json` siempre está vacío, "
            "así que no podemos determinar explícitamente qué sources pidió el user. "
            "Este detector usa heurística basada en fuentes reales de los chunks devueltos."
        ),
        "suggestion": (
            "FIX PROPUESTO: agregar logging de `filters_applied` a "
            "`retrieve()` en rag/__init__.py para registrar sources pedidos vs. obtenidos. "
            "Sin esto, el detector sigue siendo heurístico y puede tener falsos positivos."
        ),
    }

    try:
        exists = ragvec_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='meta_obsidian_notes_v11_741d239c'"
        ).fetchone()
    except sqlite3.OperationalError:
        out["error"] = "metadata table inaccesible"
        return out
    if exists is None:
        out["error"] = "metadata table missing"
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    # Queries con paths_json no-null y >1 path (top-k > 1).
    try:
        queries = telemetry_conn.execute(
            """
            SELECT ts, q, paths_json
            FROM rag_queries
            WHERE ts >= ? AND paths_json IS NOT NULL
              AND json_array_length(paths_json) >= 2
            ORDER BY ts DESC
            LIMIT 50
            """,
            (cutoff_iso,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out

    out["queries_analyzed"] = len(queries)

    # Para cada query, lookupeá los sources de sus paths en metadata.
    meta_table = "meta_obsidian_notes_v11_741d239c"
    imbalanced: list = []

    for ts, query_text, paths_json_str in queries:
        try:
            paths = (
                json.loads(paths_json_str)
                if isinstance(paths_json_str, str)
                else paths_json_str
            )
        except (json.JSONDecodeError, TypeError):
            continue

        # Lookup sources para cada path
        sources_counter: Counter = Counter()
        for path in paths:
            try:
                result = ragvec_conn.execute(
                    f"SELECT source FROM {meta_table} WHERE file = ? LIMIT 1",
                    (path,),
                ).fetchone()
                if result and result[0]:
                    sources_counter[result[0]] += 1
            except sqlite3.OperationalError:
                continue

        # ¿Multi-source con imbalance?
        if len(sources_counter) >= 2:
            total = sum(sources_counter.values())
            dominant_source = sources_counter.most_common(1)[0][0]
            dominant_count = sources_counter.most_common(1)[0][1]
            imbalance_pct = round(100.0 * dominant_count / total, 1)

            # Alert threshold: una fuente domina >80% del top-k
            if imbalance_pct > 80.0:
                imbalanced.append(
                    {
                        "ts": ts,
                        "q": query_text[:60] if query_text else "",
                        "sources": dict(sources_counter),
                        "dominant_source": dominant_source,
                        "imbalance_pct": imbalance_pct,
                    }
                )

    out["imbalanced_queries"] = imbalanced
    out["count"] = len(imbalanced)
    out["alert"] = out["count"] >= 3

    return out


def _audit_abandon_high_score(conn, days: int) -> dict:
    """Queries donde el ranker estaba seguro pero el user se fue igual.

    Detecta el patrón "content gap, no ranking gap": ``top_score >= 0.4``
    (el pipeline encontró algo relevante) pero el outcome fue
    ``session_outcome_weak_negative`` (el user abandonó sin interactuar).
    Esto señala que el problema está en la **respuesta del LLM** —
    vacía, alucinada, o que no responde la pregunta — y NO en el ranking.

    Diferencia vs ranking gap: un ranking gap tiene ``top_score < 0.4``
    — el pipeline no encontró nada relevante desde el vamos. Un content
    gap tiene score alto pero el usuario igual se fue.

    El JOIN usa ``turn_id`` de ``rag_feedback`` que tiene formato
    ``<session_id>:<query_id>`` — se extrae el id numérico con substr
    y se cruza contra ``rag_queries.id``.

    Returns dict con shape::

        {
            "window_days": 14,
            "count": 7,
            "samples": [{"q": "...", "top_score": 0.55, "ts": "...", "session": "..."}, ...],
            "alert": True,
            "suggestion": "...",
        }

    ``alert=True`` cuando ``count >= 5``. Devuelve ``count=0, alert=False``
    cuando no hay datos (no explota).
    """
    ALERT_THRESHOLD = 5
    out: dict = {
        "window_days": days,
        "count": 0,
        "samples": [],
        "alert": False,
        "suggestion": (
            "Top queries donde el ranker acertó (top_score >= 0.4) "
            "pero el user se fue igual (weak_negative). Revisar la "
            "respuesta del LLM — ¿era vacía, alucinada, o no respondía "
            "la pregunta? El problema está en generation, no en retrieval."
        ),
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type=\'table\' AND name=\'rag_feedback\'"
        ).fetchone()
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_feedback"
        return out

    cutoff_param = f"-{days} days"

    count_row = conn.execute(
        """
        SELECT COUNT(*)
        FROM rag_feedback rf
        JOIN rag_queries rq
          ON rq.id = CAST(substr(rf.turn_id, instr(rf.turn_id, ':') + 1) AS INTEGER)
        WHERE json_extract(rf.extra_json, '$.implicit_loss_source')
              = 'session_outcome_weak_negative'
          AND rq.top_score >= 0.4
          AND rq.ts >= datetime('now', ?)
        """,
        (cutoff_param,),
    ).fetchone()
    count = int(count_row[0] or 0)
    out["count"] = count
    out["alert"] = count >= ALERT_THRESHOLD

    if count > 0:
        sample_rows = conn.execute(
            """
            SELECT
              substr(rq.q, 1, 80)  AS q_short,
              rq.top_score,
              rq.ts,
              rq.session
            FROM rag_feedback rf
            JOIN rag_queries rq
              ON rq.id = CAST(substr(rf.turn_id, instr(rf.turn_id, ':') + 1) AS INTEGER)
            WHERE json_extract(rf.extra_json, '$.implicit_loss_source')
                  = 'session_outcome_weak_negative'
              AND rq.top_score >= 0.4
              AND rq.ts >= datetime('now', ?)
            ORDER BY rq.ts DESC
            LIMIT 20
            """,
            (cutoff_param,),
        ).fetchall()
        out["samples"] = [
            {
                "q": r[0],
                "top_score": round(float(r[1] or 0.0), 3),
                "ts": r[2],
                "session": r[3],
            }
            for r in sample_rows
        ]

    return out


def _audit_cache_hit_by_intent(conn, days: int) -> dict:
    """Cache hit rate por intent — detecta intents con eligible>=20 y hit_rate<5%."""
    import sqlite3 as _sqlite3
    ALERT_MIN_ELIGIBLE = 20
    ALERT_MAX_HIT_RATE = 5.0
    from datetime import datetime, timedelta
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        rows = conn.execute("""
            SELECT
              COALESCE(json_extract(extra_json, '$.intent'), 'unknown') AS intent,
              COUNT(*) AS eligible,
              SUM(CASE WHEN json_extract(extra_json, '$.cache_probe.result') = 'hit'
                       THEN 1 ELSE 0 END) AS hits,
              SUM(CASE WHEN json_extract(extra_json, '$.cache_probe.result') = 'miss'
                       THEN 1 ELSE 0 END) AS misses,
              ROUND(100.0 * SUM(CASE WHEN json_extract(extra_json, '$.cache_probe.result') = 'hit'
                               THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS hit_rate_pct
            FROM rag_queries
            WHERE ts >= ?
              AND cmd IN ('web', 'query', 'serve.chat')
              AND json_extract(extra_json, '$.cache_probe.result') IN ('hit', 'miss')
            GROUP BY intent
            HAVING eligible >= 5
            ORDER BY eligible DESC
        """, (cutoff_iso,)).fetchall()
    except _sqlite3.OperationalError as exc:
        return {"window_days": days, "by_intent": [], "alerts_count": 0,
                "global_hit_rate_pct": None, "suggestion": None, "error": repr(exc)}
    by_intent = []; total_eligible = 0; total_hits = 0; alerts_count = 0
    for r in rows:
        eligible = int(r[1] or 0); hits = int(r[2] or 0)
        hit_rate_pct = float(r[4] or 0.0)
        is_alert = eligible >= ALERT_MIN_ELIGIBLE and hit_rate_pct < ALERT_MAX_HIT_RATE
        if is_alert:
            alerts_count += 1
        by_intent.append({"intent": r[0], "eligible": eligible, "hits": hits,
                           "misses": int(r[3] or 0), "hit_rate_pct": hit_rate_pct, "alert": is_alert})
        total_eligible += eligible; total_hits += hits
    global_hit_rate_pct = round(100.0 * total_hits / total_eligible, 2) if total_eligible > 0 else None
    suggestion = (
        "Cache cosine threshold alto (RAG_CACHE_COSINE=0.93) o corpus_hash bucket "
        "muy chico — el cache nunca pega en los intents con alerta. "
        "Prob\u00e1 bajar RAG_CACHE_COSINE a 0.90."
    ) if alerts_count > 0 else None
    return {"window_days": days, "by_intent": by_intent, "alerts_count": alerts_count,
            "global_hit_rate_pct": global_hit_rate_pct, "suggestion": suggestion}


def _audit_ranker_blind_spots(conn: sqlite3.Connection, days: int = 30) -> dict:
    """Paths que el user marcó como `corrective_path` pero el ranker NUNCA trae como top-1.

    Pregunta accionable: ¿qué notas el user tuvo que corregir manualmente
    (porque el ranker se equivocó de top-1) que el ranker NUNCA pone en
    posición 1 dentro de la ventana? Esos son blind spots concretos —
    el `feedback_pos` weight del scoring formula está mal calibrado para
    esos paths.

    Cualquier hit es señal de blind spot real: el user ya dijo "esta es
    la nota correcta" y el ranker la sigue ignorando. Cualquier
    `count > 0` dispara `alert=True`.

    Acción sugerida cuando dispara: subir `feedback_pos` weight en
    `ranker.json` o re-correr `rag tune --apply` para que esos paths
    pesen más en el scoring formula post-rerank.

    Defensivo contra `paths_json` con JSON malformado (rows viejos)
    usando `json_valid()`. Top-1 = `j.key = 0` sobre el array — el shape
    de `paths_json` es lista de strings, no de objetos (verificado
    contra telemetry.db real al implementar).

    Returns:
        {
          "window_days": 30,
          "blind_spots": [{"path": "...", "times_corrected": N}, ...],
          "count": N,
          "alert": bool,
          "suggestion": "..." (sólo presente cuando alert=True),
        }
    """
    out: dict = {
        "window_days": days,
        "blind_spots": [],
        "count": 0,
        "alert": False,
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_feedback'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_feedback"
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    try:
        rows = conn.execute(
            """
            WITH cp AS (
              SELECT json_extract(extra_json, '$.corrective_path') AS path,
                     COUNT(*) AS times
              FROM rag_feedback
              WHERE rating < 0
                AND json_extract(extra_json, '$.corrective_path') IS NOT NULL
                AND json_extract(extra_json, '$.corrective_path') <> ''
              GROUP BY path
            ),
            top1_paths AS (
              SELECT DISTINCT j.value AS path
              FROM rag_queries, json_each(rag_queries.paths_json) j
              WHERE j.key = 0
                AND rag_queries.ts >= ?
                AND json_valid(rag_queries.paths_json)
            )
            SELECT cp.path, cp.times
            FROM cp
            LEFT JOIN top1_paths ON cp.path = top1_paths.path
            WHERE top1_paths.path IS NULL
            ORDER BY cp.times DESC
            LIMIT 20
            """,
            (cutoff_iso,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out

    blind_spots = [
        {"path": r[0], "times_corrected": int(r[1] or 0)} for r in rows
    ]
    out["blind_spots"] = blind_spots
    out["count"] = len(blind_spots)
    out["alert"] = out["count"] > 0
    if out["alert"]:
        out["suggestion"] = (
            "Subir feedback_pos weight en ranker.json o re-correr "
            "`rag tune --apply` con estos paths como signal"
        )
    return out


def _audit_corpus_coverage_gaps(conn, days: int) -> dict:
    """Temas que el user busca repetidamente y el corpus no cubre bien.

    Pregunta accionable: ¿qué debería indexar o agregar al vault?

    Identifica queries que:
    - Aparecen ≥2 veces en la ventana (no es un one-off accidental).
    - Tienen top_score entre 0.015 y 0.3 — el pipeline encontró algo pero
      con muy baja confianza. Scores < 0.015 son refusals directos (la
      query ni merece intentarlo); scores > 0.3 son cobertura aceptable.
    - Vienen de los canales principales (chat, query, web, serve.chat).

    Output ordenado por recurrencia DESC, avg_top_score ASC — las más
    buscadas y peor servidas al frente.

    Thresholds calibrados con distribución real de producción
    (commit 2026-05-04, SQL validado en DB real → 129 queries únicas
    con ≥2 ocurrencias en 30 días):
    - top_score < 0.3: rango donde el reranker no tiene señal semántica
      suficiente — respuestas vagas, citas poco relevantes.
    - unique_queries_repeated ≥ 5: alerta conservadora; menos de 5 tiene
      demasiado falso positivo (tema nicho de una semana atípica).
    """
    import sqlite3 as _sqlite3
    from datetime import datetime, timedelta
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    out: dict = {
        "window_days": days,
        "total_low_score_queries": 0,
        "unique_queries_repeated": 0,
        "top_candidates": [],
        "alert": False,
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_queries'"
        ).fetchone()
    except _sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_queries"
        return out

    total_row = conn.execute(
        """
        SELECT COUNT(*)
        FROM rag_queries
        WHERE ts >= ?
          AND top_score < 0.3
          AND top_score > 0.015
          AND cmd IN ('chat', 'query', 'web', 'serve.chat')
          AND q != ''
        """,
        (cutoff_iso,),
    ).fetchone()
    out["total_low_score_queries"] = int(total_row[0] or 0)

    unique_row = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
          SELECT lower(q) AS q_normalized
          FROM rag_queries
          WHERE ts >= ?
            AND top_score < 0.3
            AND top_score > 0.015
            AND cmd IN ('chat', 'query', 'web', 'serve.chat')
            AND q != ''
          GROUP BY q_normalized
          HAVING COUNT(*) >= 2
        )
        """,
        (cutoff_iso,),
    ).fetchone()
    out["unique_queries_repeated"] = int(unique_row[0] or 0)

    rows = conn.execute(
        """
        SELECT
          lower(q) AS q_normalized,
          COUNT(*) AS occurrences,
          ROUND(AVG(top_score), 3) AS avg_top_score,
          MIN(ts) AS first_seen,
          MAX(ts) AS last_seen
        FROM rag_queries
        WHERE ts >= ?
          AND top_score < 0.3
          AND top_score > 0.015
          AND cmd IN ('chat', 'query', 'web', 'serve.chat')
          AND q != ''
        GROUP BY q_normalized
        HAVING COUNT(*) >= 2
        ORDER BY occurrences DESC, avg_top_score ASC
        LIMIT 20
        """,
        (cutoff_iso,),
    ).fetchall()
    out["top_candidates"] = [dict(r) for r in rows]
    out["alert"] = out["unique_queries_repeated"] >= 5
    return out



def _audit_draft_decision_health(conn: sqlite3.Connection, days: int = 30) -> dict:
    """Health check del loop de aprendizaje del bot WA (drafts).

    Pregunta accionable: ¿el loop está acumulando pares gold (approved_editar)
    a buen ritmo para alimentar el DPO fine-tune del modelo de drafts?

    Gate DPO: requiere >=100 pares gold all-time. Un par gold es una row con
    decision='approved_editar' AND sent_text IS NOT NULL AND sent_text != bot_draft.

    Alerta cuando approved_editar_pct < 10% con muestra >=10 en la ventana.
    """
    DPO_GATE_THRESHOLD = 100
    out: dict = {
        "window_days": days,
        "total": 0,
        "by_decision": {
            "approved_si": 0,
            "approved_editar": 0,
            "rejected": 0,
            "expired": 0,
        },
        "approved_editar_pct": 0.0,
        "gold_pairs_total": 0,
        "dpo_gate_threshold": DPO_GATE_THRESHOLD,
        "dpo_gate_open": False,
        "alert": False,
        "suggestion": None,
    }

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name='rag_draft_decisions'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        out["error"] = repr(exc)
        return out
    if exists is None:
        out["error"] = "table missing: rag_draft_decisions"
        return out

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    rows = conn.execute(
        "SELECT decision, COUNT(*) AS n"
        " FROM rag_draft_decisions"
        " WHERE ts >= ?"
        " GROUP BY decision"
        " ORDER BY n DESC",
        (cutoff_iso,),
    ).fetchall()

    total = 0
    by_decision: dict = {"approved_si": 0, "approved_editar": 0, "rejected": 0, "expired": 0}
    for r in rows:
        decision = r[0] or "unknown"
        n = int(r[1] or 0)
        total += n
        if decision in by_decision:
            by_decision[decision] = n

    out["total"] = total
    out["by_decision"] = by_decision

    approved_editar_pct = (
        round(by_decision["approved_editar"] / total * 100, 1) if total > 0 else 0.0
    )
    out["approved_editar_pct"] = approved_editar_pct

    gold_pairs_total = conn.execute(
        "SELECT COUNT(*) FROM rag_draft_decisions"
        " WHERE decision = 'approved_editar'"
        "   AND sent_text IS NOT NULL"
        "   AND sent_text != bot_draft"
    ).fetchone()[0]
    out["gold_pairs_total"] = int(gold_pairs_total or 0)
    out["dpo_gate_open"] = out["gold_pairs_total"] >= DPO_GATE_THRESHOLD

    if total >= 10 and approved_editar_pct < 10.0:
        out["alert"] = True
        out["suggestion"] = (
            f"Solo {by_decision['approved_editar']} de {total} drafts fueron editados"
            f" ({approved_editar_pct:.1f}%) en los ultimos {days}d."
            " El modelo DPO no acumula pares gold."
            " Si los drafts son mediocres y el user los aprueba igual,"
            " el loop de aprendizaje esta roto."
        )

    return out


def _audit_db_size() -> dict:
    """Tamaño físico de las DBs + WAL files. Detección temprana de bloat."""
    out = {}
    for db_path, label in (
        (TELEMETRY_DB, "telemetry"),
        (RAGVEC_DB, "ragvec"),
    ):
        if not db_path.is_file():
            out[label] = {"missing": True}
            continue
        size_mb = db_path.stat().st_size / 1024 / 1024
        wal = db_path.with_suffix(db_path.suffix + "-wal")
        wal_mb = wal.stat().st_size / 1024 / 1024 if wal.is_file() else 0.0
        out[label] = {
            "path": str(db_path),
            "size_mb": round(size_mb, 1),
            "wal_mb": round(wal_mb, 1),
        }
    return out


def _render_text(report: dict) -> str:
    out = []
    out.append("=" * 72)
    title = f"obsidian-rag telemetry health audit — últimos {report['days']} días"
    if report.get("since"):
        title += f" (desde {report['since']})"
    out.append(title)
    out.append("=" * 72)
    out.append("")

    # Errors
    err = report["sql_errors"]
    if err.get("files_missing"):
        out.append(f"⚠️  Logs missing: {', '.join(err['files_missing'])}")
        out.append("")
    out.append(f"📊 Silent errors loggeados: {err['total_errors']}")
    if err["total_errors"] > 0:
        out.append(f"   • silent_errors.jsonl: {err['by_log_file']['silent_errors']}")
        out.append(f"   • sql_state_errors.jsonl: {err['by_log_file']['sql_state']}")
        if err["test_pollution_hits"]:
            out.append(
                f"   ⚠️  TEST POLLUTION: {err['test_pollution_hits']} entries con "
                f"event~test (deberían ser 0 post 2026-04-24)"
            )
        out.append("")
        out.append("   Top causas:")
        for ev, n in err["by_event"].most_common(8):
            out.append(f"     {n:>5} × {ev}")
        out.append("")
        out.append("   Curva diaria (busca el día que explotó):")
        for day, n in sorted(err["by_day"].items()):
            bar = "█" * min(40, n // 20)
            out.append(f"     {day}  {n:>4}  {bar}")
    out.append("")

    # Query latency
    lat = report.get("query_latency")
    if lat is not None:
        out.append("📈 Query latency (ms):")
        out.append(
            f"     {'cmd':<28}  {'n':>5}  {'avg_retr':>9}  {'max_retr':>9}  {'avg_gen':>8}  {'max_gen':>8}"
        )
        for r in lat["by_cmd"][:10]:
            out.append(
                f"     {r['cmd']:<28}  {r['n']:>5}  "
                f"{r['avg_retrieve'] or 0:>9}  {r['max_retrieve'] or 0:>9}  "
                f"{r['avg_gen'] or 0:>8}  {r['max_gen'] or 0:>8}"
            )
        out.append("")
        if lat["outliers"]:
            out.append(
                f"   ⚠️  {lat['outliers_count']} outliers >30s retrieve o >60s gen "
                f"(deberían ser 0 post _DEEP_MAX_SECONDS cap):"
            )
            for o in lat["outliers"][:5]:
                out.append(
                    f"     {o['ts']}  {o['cmd']}  retrieve={o['t_retrieve']}s  gen={o['t_gen']}s  q={o['q']!r}"
                )
        else:
            out.append("   ✓ Sin outliers >30s — el cap _DEEP_MAX_SECONDS funciona.")
        out.append("")

    # Cache health
    ch = report.get("cache_health")
    if ch is not None:
        out.append(f"💾 Semantic cache: {ch['cache_table_rows']} rows en rag_response_cache")
        if ch["by_probe"]:
            out.append("   Cache probe distribution (web queries):")
            total_with_probe = sum(
                r["n"] for r in ch["by_probe"] if r["result"] is not None
            )
            for r in ch["by_probe"]:
                result = r["result"] or "(no_probe_logged)"
                reason = r["reason"] or "-"
                out.append(f"     {r['n']:>5} × result={result:<10} reason={reason}")
            if total_with_probe == 0:
                out.append(
                    "   ⚠️  0 queries con cache_probe — verificar el fix 2026-04-24 "
                    "(commit 3dcbe81) está deployado."
                )
        out.append("")

    # Anticipatory agent
    ant = report.get("anticipate")
    if ant is not None:
        status = (ant.get("status") or "unknown").upper()
        sent = ant.get("total_sent", 0)
        ev = ant.get("total_evaluated", 0)
        rate_pct = (ant.get("send_rate") or 0.0) * 100
        last_age = ant.get("last_emit_age_hours")
        if status == "STALE":
            last_str = (
                f"last emit {last_age:.0f}h ago — daemon may be down"
                if last_age is not None
                else "no emits ever — daemon may be down"
            )
            out.append(f"🤖 Anticipate: {status} | {last_str}")
        elif status == "UNKNOWN":
            issues = "; ".join(ant.get("issues") or []) or "no data"
            out.append(f"🤖 Anticipate: {status} | {issues}")
        else:
            last_str = (
                f"last emit {last_age:.0f}h ago"
                if last_age is not None
                else "no emits yet"
            )
            out.append(
                f"🤖 Anticipate: {status} | {ev} evaluated, "
                f"{sent} sent ({rate_pct:.0f}%), {last_str}"
            )
        # Per-signal one-liner para signals con problemas
        for kind, info in (ant.get("by_signal") or {}).items():
            sig_st = info.get("status")
            if sig_st in ("silent", "noisy", "stale"):
                out.append(
                    f"     ⚠️  {kind}: {sig_st} "
                    f"(emits={info.get('emits')}, evaluated={info.get('evaluated')}, "
                    f"avg_score={info.get('avg_score')})"
                )
        out.append("")

    # Retrieval + Chat health — audit 2026-04-25 R2-Telemetry #5.
    # Render compacto: ícono + status + 1 línea per issue.
    def _icon_for(status: str) -> str:
        s = status.lower()
        if s == "healthy":
            return "✓"
        if s == "degraded":
            return "⚠"
        if s == "stale":
            return "✗"
        return "?"

    out.append("=== Health checks ===")
    if ant is not None:
        ant_status = (ant.get("status") or "unknown").lower()
        ant_age = ant.get("last_emit_age_hours")
        suffix = (
            f"last_emit {ant_age:.0f}h ago"
            if isinstance(ant_age, (int, float))
            else "no emits yet"
        )
        out.append(f"{_icon_for(ant_status)} Anticipate: {ant_status} ({suffix})")
    ret = report.get("retrieval_health")
    if ret is not None:
        ret_status = (ret.get("status") or "unknown").lower()
        details = ret.get("details") or {}
        n = details.get("queries_count") or 0
        out.append(
            f"{_icon_for(ret_status)} Retrieval: {ret_status} "
            f"({n} queries en {report['days']}d)"
        )
        for issue in (ret.get("issues") or []):
            out.append(f"  - {issue}")
    chat = report.get("chat_health")
    if chat is not None:
        chat_status = (chat.get("status") or "unknown").lower()
        details = chat.get("details") or {}
        n = details.get("chats_count") or 0
        out.append(
            f"{_icon_for(chat_status)} Chat: {chat_status} "
            f"({n} chats en {report['days']}d)"
        )
        for issue in (chat.get("issues") or []):
            out.append(f"  - {issue}")
    out.append("")

    # Feedback corrective_path gap
    fcg = report.get("feedback_corrective_gap")
    if fcg is not None:
        if fcg.get("error"):
            out.append(f"🎯 Feedback corrective_path gap: ERROR — {fcg['error']}")
        else:
            total = fcg["total_neg"]
            has_cp = fcg["has_cp"]
            missing = fcg["missing_cp"]
            pct = fcg["pct_covered"]
            gate_open = fcg["gate_open"]
            rows_left = fcg["rows_to_close_gate"]
            threshold = fcg["gate_threshold"]
            gate_icon = "✅" if gate_open else "🔒"
            out.append(
                f"🎯 Feedback corrective_path gap: "
                f"{has_cp}/{total} negativos con CP ({pct:.1f}%) "
                f"— gate LoRA {gate_icon} {'ABIERTO' if gate_open else 'CERRADO'} "
                f"(threshold: {threshold})"
            )
            if not gate_open:
                out.append(
                    f"   Faltan {rows_left} corrective_paths para abrir el gate. "
                    f"Corré `rag feedback backfill` o `rag feedback infer-implicit`."
                )
        out.append("")

    # Cross-source single-source detector
    css = report.get("cross_source_single_source")
    if css is not None:
        if css.get("error"):
            out.append(f"🔀 Cross-source imbalance: ERROR — {css['error']}")
        elif css.get("blocker_message"):
            out.append(css["blocker_message"])
        if css.get("count", 0) > 0:
            out.append(
                f"🔀 Cross-source imbalance: {css['count']} queries con "
                f">80% dominio de una fuente (de {css['queries_analyzed']} analizadas)"
            )
            if css["alert"]:
                out.append("   ⚠️  ALERT TRIGGER: ≥3 queries imbalanceadas")
            for q_info in css.get("imbalanced_queries", [])[:3]:
                sources_str = ", ".join(
                    f"{s}:{c}" for s, c in sorted(q_info["sources"].items())
                )
                out.append(
                    f"     • {q_info['ts']} | {q_info['q']!r} | "
                    f"sources=[{sources_str}] | domina {q_info['dominant_source']} "
                    f"({q_info['imbalance_pct']:.0f}%)"
                )
            if css.get("suggestion"):
                out.append(f"   Sugerencia: {css['suggestion'][:90]}...")
        out.append("")

    # Cache hit por intent
    chi = report.get("cache_hit_by_intent")
    if chi is not None:
        if chi.get("error"):
            out.append(f"🔍 Cache hit por intent: ERROR — {chi['error']}")
        else:
            ghr = chi.get("global_hit_rate_pct")
            ghr_str = f"{ghr:.1f}%" if ghr is not None else "n/d"
            out.append(f"🔍 Cache hit por intent ({chi['window_days']}d) — global: {ghr_str}")
            if chi["by_intent"]:
                out.append(f"   {'intent':<18} {'eligible':>8} {'hits':>6} {'misses':>7} {'hit_rate':>9}  alerta")
                out.append("   " + "-" * 58)
                for row in chi["by_intent"]:
                    alert_str = "⚠️ " if row["alert"] else "  "
                    out.append(
                        f"   {alert_str}{row['intent']:<16} "
                        f"{row['eligible']:>8} {row['hits']:>6} {row['misses']:>7} "
                        f"{row['hit_rate_pct']:>8.1f}%"
                    )
            if chi.get("alerts_count", 0) > 0 and chi.get("suggestion"):
                out.append(f"   → {chi['suggestion']}")
        out.append("")

    # Harvest candidates
    hc = report.get("harvest_candidates")
    if hc is not None:
        if hc.get("error"):
            out.append(f"🌾 Harvest candidates: ERROR — {hc['error']}")
        else:
            count = hc["count"]
            days_w = hc["window_days"]
            alert_icon = "⚠️ " if hc["alert"] else "✓ "
            out.append(
                f"{alert_icon}Harvest candidates (últimos {days_w}d, "
                f"score 0.015–0.35, sin thumbs): {count}"
            )
            if count > 0:
                out.append("   Comando copy-pasteable:")
                out.append(f"     {hc['harvest_command']}")
                out.append("")
                out.append(f"   {'#':>5}  {'score':>6}  {'cmd':<12}  query")
                for i, c in enumerate(hc["top_candidates"][:10], 1):
                    q_truncated = (
                        (c["q"][:55] + "…") if len(c["q"]) > 55 else c["q"]
                    )
                    out.append(
                        f"   {i:>5}. {c['top_score']:>6.4f}  "
                        f"{c['cmd']:<12}  {q_truncated!r}"
                    )
                if count > len(hc["top_candidates"]):
                    remaining = count - len(hc["top_candidates"])
                    out.append(
                        f"          ... y {remaining} más — "
                        "corré el comando de arriba para labelear todos."
                    )
        out.append("")

    # Ranker blind spots — paths con corrective_path que NUNCA fueron top-1
    rbs = report.get("ranker_blind_spots")
    if rbs is not None:
        if rbs.get("error"):
            out.append(f"🎯 Ranker blind spots: ERROR — {rbs['error']}")
        else:
            count = rbs["count"]
            days_w = rbs["window_days"]
            alert_icon = "⚠️ " if rbs["alert"] else "✓ "
            out.append(
                f"{alert_icon}Ranker blind spots (paths con CP "
                f"que nunca fueron top-1, últimos {days_w}d): {count}"
            )
            if count > 0:
                out.append("   El user marcó estos paths como correctos pero el ranker los ignora:")
                for bs in rbs["blind_spots"][:10]:
                    out.append(
                        f"     • {bs['times_corrected']}× corregido → {bs['path']}"
                    )
                if rbs.get("suggestion"):
                    out.append(f"   Sugerencia: {rbs['suggestion']}")
        out.append("")

    # Draft decision health (DPO fine-tune gate del bot WA)
    ddh = report.get("draft_decision_health")
    if ddh is not None:
        if ddh.get("error"):
            out.append(f"📝 Drafts WA: ERROR — {ddh['error']}")
        else:
            total_w = ddh["total"]
            ae_pct = ddh["approved_editar_pct"]
            gold = ddh["gold_pairs_total"]
            gate_open = ddh["dpo_gate_open"]
            threshold = ddh["dpo_gate_threshold"]
            gate_icon = "✅" if gate_open else "🔒"
            by_d = ddh["by_decision"]
            out.append(
                f"📝 Drafts WA (últimos {ddh['window_days']}d): "
                f"{total_w} decisiones | "
                f"aprobado={by_d['approved_si']} "
                f"editado={by_d['approved_editar']} ({ae_pct:.1f}%) "
                f"rechazado={by_d['rejected']} expirado={by_d['expired']}"
            )
            out.append(
                f"   Gold pairs all-time: {gold} "
                f"— gate DPO {gate_icon} {'ABIERTO' if gate_open else 'CERRADO'} "
                f"(threshold: {threshold})"
            )
            if ddh["alert"] and ddh["suggestion"]:
                out.append(f"   ⚠️  {ddh['suggestion']}")
        out.append("")

    # DB size
    db = report.get("db_size", {})
    out.append("💽 DB sizes:")
    for label, info in db.items():
        if info.get("missing"):
            out.append(f"     {label}: MISSING ({info!r})")
        else:
            wal_str = f" + WAL {info['wal_mb']}MB" if info["wal_mb"] > 1 else ""
            out.append(f"     {label}: {info['size_mb']} MB{wal_str}")
    out.append("")

    # Hints
    out.append("─" * 72)
    out.append("Próximos pasos sugeridos:")
    if err.get("test_pollution_hits"):
        out.append("  • Test pollution detectada → revisar `_isolate_sql_state_error_log`")
        out.append("    autouse fixture en tests/conftest.py — debe estar activa.")
    err_count = err.get("total_errors", 0)
    if err_count > 100:
        out.append(
            f"  • {err_count} errores en {report['days']}d — investigar top causas. "
            "Cruzar curva diaria con `git log --since='YYYY-MM-DD'`."
        )
    if (
        ch is not None
        and ch.get("by_probe")
        and ch["cache_table_rows"] < 10
    ):
        out.append(
            "  • Cache table casi vacía — chequear gates en run_chat_turn / "
            "_semantic_eligible. ¿Demasiado restrictivo?"
        )
    if ant is not None:
        ant_status = (ant.get("status") or "").lower()
        if ant_status == "stale":
            out.append(
                "  • Anticipate STALE — el daemon parece down. Revisar "
                "`launchctl list | grep anticipate` y los logs en "
                "~/Library/Logs/com.fer.obsidian-rag-anticipate.log."
            )
        elif ant_status == "degraded":
            out.append(
                "  • Anticipate DEGRADED — send_rate <5%. Revisar threshold "
                "(_ANTICIPATE_THRESHOLD), dedup window y quiet hours."
            )
        elif ant_status == "unknown":
            out.append(
                "  • Anticipate UNKNOWN — tabla missing o agente nunca corrió. "
                "Validar que el daemon esté instalado."
            )
        elif (ant.get("issues") or []):
            # healthy global pero con per-signal issues
            n_issues = len(ant.get("issues") or [])
            out.append(
                f"  • Anticipate healthy global, pero {n_issues} signal "
                f"issue(s) — revisar la sección 🤖 arriba (silent / noisy)."
            )
    # Audit 2026-04-25 R2-Telemetry #5: hints para retrieval + chat.
    if ret is not None:
        ret_status = (ret.get("status") or "").lower()
        if ret_status == "stale":
            out.append(
                "  • Retrieval STALE — 0 queries en la ventana. ¿El web "
                "server está down o nadie está usando el sistema?"
            )
        elif ret_status == "degraded":
            n_issues = len(ret.get("issues") or [])
            out.append(
                f"  • Retrieval DEGRADED — {n_issues} issue(s). Revisar "
                "cache hit rate, ranker scores, y p95 t_retrieve arriba."
            )
    if chat is not None:
        chat_status = (chat.get("status") or "").lower()
        if chat_status == "stale":
            out.append(
                "  • Chat STALE — 0 chats en la ventana. ¿El endpoint "
                "/chat está respondiendo? ¿Ollama está corriendo?"
            )
        elif chat_status == "degraded":
            n_issues = len(chat.get("issues") or [])
            out.append(
                f"  • Chat DEGRADED — {n_issues} issue(s). Revisar p95 "
                "t_gen, critique_fired_rate, y refusal_rate arriba."
            )
    if fcg is not None and not fcg.get("error") and not fcg["gate_open"]:
        rows_left = fcg["rows_to_close_gate"]
        has_cp = fcg["has_cp"]
        threshold = fcg["gate_threshold"]
        out.append(
            f"  • LoRA fine-tune BLOQUEADO: {has_cp}/{threshold} corrective_paths "
            f"disponibles (faltan {rows_left}). Corré `rag feedback backfill` o "
            "`rag feedback infer-implicit --window-seconds 600` para destrabar el gate."
        )
    ahs_hints = report.get("abandon_high_score")
    if ahs_hints is not None and ahs_hints.get("alert") and not ahs_hints.get("error"):
        out.append(
            f"  • Content gap detectado: {ahs_hints['count']} queries con "
            "top_score \u2265 0.4 donde el user se fue igual. "
            "Revisar respuestas del LLM (vacias / alucinaciones)."
        )
    if (
        not err.get("total_errors")
        and not (lat and lat.get("outliers"))
        and (ant is None or ant.get("status") == "healthy")
        and (ret is None or ret.get("status") == "healthy")
        and (chat is None or chat.get("status") == "healthy")
        and (fcg is None or fcg.get("gate_open") or fcg.get("error"))
        and (ahs_hints is None or not ahs_hints.get("alert"))
    ):
        out.append("  • ✅ Sistema sano. No se requiere acción inmediata.")
    out.append("=" * 72)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--days", type=int, default=7,
        help="Ventana de análisis en días (default 7)",
    )
    parser.add_argument(
        "--since",
        help=("Cutoff adicional (ISO8601) — ignora eventos antes de este ts "
              "aunque estén dentro de la ventana de days. Útil para excluir "
              "pollution histórica pre-fix. Ej: --since 2026-04-24T17:53:00"),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON en lugar de texto legible",
    )
    args = parser.parse_args()

    report: dict = {"days": args.days, "since": args.since}
    report["sql_errors"] = _audit_sql_errors(args.days, since_ts=args.since)
    report["db_size"] = _audit_db_size()

    conn = _open_db(TELEMETRY_DB)
    ragvec_conn = _open_db(RAGVEC_DB)
    if conn is None:
        report["query_latency"] = None
        report["cache_health"] = None
        report["anticipate"] = None
        report["retrieval_health"] = None
        report["chat_health"] = None
        report["feedback_corrective_gap"] = None
        report["harvest_candidates"] = None
        report["cross_source_single_source"] = None
        report["cache_hit_by_intent"] = None
        report["abandon_high_score"] = None
        report["ranker_blind_spots"] = None
        report["draft_decision_health"] = None
        report["db_unavailable"] = str(TELEMETRY_DB)
    else:
        try:
            report["query_latency"] = _audit_query_latency(conn, args.days)
            report["cache_health"] = _audit_cache_health(conn, args.days)
            # rag_anticipate_candidates vive en la misma telemetry.db,
            # así que reusamos la conexión.
            report["anticipate"] = check_anticipate_health(conn, args.days)
            # Audit 2026-04-25 R2-Telemetry #5: agregamos retrieval +
            # chat health para detectar degradation silenciosa que el
            # `check_anticipate_health` no cubría.
            report["retrieval_health"] = check_retrieval_health(conn, args.days)
            report["chat_health"] = check_chat_health(conn, args.days)
            report["feedback_corrective_gap"] = _audit_feedback_corrective_gap(conn)
            report["harvest_candidates"] = _audit_harvest_candidates(conn, args.days)
            # Cross-source single-source detector (requiere ragvec_conn)
            if ragvec_conn is not None and conn is not None:
                report["cross_source_single_source"] = _audit_cross_source_single_source(
                    conn, ragvec_conn, args.days
                )
            else:
                report["cross_source_single_source"] = None
            report["cache_hit_by_intent"] = _audit_cache_hit_by_intent(conn, args.days)
            report["abandon_high_score"] = _audit_abandon_high_score(conn, args.days)
            report["ranker_blind_spots"] = _audit_ranker_blind_spots(conn, days=30)
            report["draft_decision_health"] = _audit_draft_decision_health(conn, args.days)
        except sqlite3.OperationalError as exc:
            report["query_latency"] = None
            report["cache_health"] = None
            report["anticipate"] = None
            report["retrieval_health"] = None
            report["chat_health"] = None
            report["feedback_corrective_gap"] = None
            report["harvest_candidates"] = None
            report["cross_source_single_source"] = None
            report["cache_hit_by_intent"] = None
            report["abandon_high_score"] = None
            report["ranker_blind_spots"] = None
            report["draft_decision_health"] = None
            report["db_error"] = repr(exc)
        finally:
            conn.close()
            if ragvec_conn is not None:
                ragvec_conn.close()

    if args.json:
        # Counter no es JSON-serializable directo.
        if "sql_errors" in report:
            report["sql_errors"]["by_event"] = dict(report["sql_errors"]["by_event"])
            report["sql_errors"]["by_day"] = dict(report["sql_errors"]["by_day"])
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
