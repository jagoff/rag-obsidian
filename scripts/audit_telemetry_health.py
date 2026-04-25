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
    if (
        not err.get("total_errors")
        and not (lat and lat.get("outliers"))
        and (ant is None or ant.get("status") == "healthy")
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
    if conn is None:
        report["query_latency"] = None
        report["cache_health"] = None
        report["anticipate"] = None
        report["db_unavailable"] = str(TELEMETRY_DB)
    else:
        try:
            report["query_latency"] = _audit_query_latency(conn, args.days)
            report["cache_health"] = _audit_cache_health(conn, args.days)
            # rag_anticipate_candidates vive en la misma telemetry.db,
            # así que reusamos la conexión.
            report["anticipate"] = check_anticipate_health(conn, args.days)
        except sqlite3.OperationalError as exc:
            report["query_latency"] = None
            report["cache_health"] = None
            report["anticipate"] = None
            report["db_error"] = repr(exc)
        finally:
            conn.close()

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
