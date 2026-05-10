"""Per-metric data collectors — extracted from rag/cross_source_patterns.py 2026-05-09.

Each ``_c_*`` function reads a single source (sqlite-vec table, vault note,
WhatsApp bridge) and returns ``dict[date_str, float]`` for the requested
range. Each collector is silent-fail (returns ``{}`` on any error).

The ``@register_metric(name, label)`` decorator (defined in
``rag.cross_source_patterns``) populates a module-level registry
(``_COLLECTORS`` + ``_METRIC_LABELS``). When this module is imported, the
registry gets populated as a side-effect.

## Why this module is imported by cross_source_patterns

``rag/cross_source_patterns.py`` ends with
``from rag.cross_source_collectors import *  # noqa: F401, F403`` which
triggers the side-effect registration. Without that explicit import, the
registry stays empty and ``compute_correlations()`` would have nothing to
correlate.

Order matters: ``register_metric`` + ``_silent_log_safe`` must be defined in
``cross_source_patterns.py`` BEFORE the trigger import at the bottom of
that file. They are (lines ~114 and ~178), so the chain works.
"""
from __future__ import annotations

import contextlib
import re
from datetime import datetime, timedelta
from pathlib import Path

from rag.cross_source_patterns import _silent_log_safe, register_metric

__all__ = [
    "_c_mood_score",
    "_c_mood_self_report",
    "_c_sleep_quality",
    "_c_sleep_duration",
    "_c_sleep_awakenings",
    "_c_sleep_deep",
    "_c_wakeup_mood",
    "_c_spotify_minutes",
    "_c_spotify_tracks",
    "_c_queries_total",
    "_c_queries_existential",
    "_c_gmail_received",
    "_c_vault_notes_touched",
    "_c_wa_outbound_chars",
]


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
    """Cuenta de queries con regex existencial (ver `mood._QUERIES_EXISTENTIAL_RE`).

    Audit 2026-05-10 (U5): único collector del módulo que materializa filas
    SIN agregación SQL (el regex existencial se aplica en Python). Con vault
    de uso intensivo + rango grande (90d), `rag_queries` puede contener
    decenas de miles de filas y `fetchall()` ahogaba la RAM. Mitigación:

    1. Iterar el cursor (no fetchall) → memoria O(1) por fila.
    2. `LIMIT` defensivo (50k filas) para acotar peores casos; el regex
       en Python es lo barato, lo caro es materializar todo en lista.
    3. Filtro por `cmd` ya selectivo en SQL — la query promedio devuelve
       ~10-30k filas para 90d en uso real.
    """
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        from rag.mood import _QUERIES_EXISTENTIAL_RE  # noqa: PLC0415
        out: dict[str, int] = {}
        with _ragvec_state_conn() as conn:
            cursor = conn.execute(
                "SELECT substr(ts, 1, 10) AS d, q "
                "FROM rag_queries "
                "WHERE ts >= ? AND ts < ? "
                "AND COALESCE(cmd,'') IN ('', 'query', 'chat', 'ask') "
                "LIMIT 50000",
                (f"{start}T00:00:00", f"{end}T23:59:59"),
            )
            for date, q in cursor:
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

    Source: notas en `<vault>/99-obsidian/99-AI/external-ingest/Gmail/<YYYY-MM-DD>.md` que
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
    folder = VAULT_PATH / "99-obsidian" / "99-AI" / "external-ingest" / "Gmail"
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
      - NO bajo `99-obsidian/` (auto-generated)
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
            if len(parts) >= 2 and parts[0] == "99-obsidian":
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
        from rag.integrations.whatsapp.fetch import _bridge_ts_bound  # noqa: PLC0415
        db = Path(WHATSAPP_BRIDGE_DB_PATH)
        if not db.exists():
            return {}
        import sqlite3 as _sql  # noqa: PLC0415
        # 2026-05-09: bounds con offset matching bridge format. Antes pasábamos
        # `f"{start}T00:00:00"` con `T` → lex compare contra `... HH:MM:SS-03:00`
        # con espacio → `T` (84) > ` ` (32), así que ALL rows quedaban excluídas
        # por el `>= ?` upper bound. Métrica devolvía dict vacío permanente.
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        start_bound = _bridge_ts_bound(start_dt)
        end_bound = _bridge_ts_bound(end_dt)
        conn = _sql.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT substr(timestamp, 1, 10) AS d, AVG(LENGTH(content)) "
                "FROM messages "
                "WHERE is_from_me=1 AND content IS NOT NULL "
                "AND timestamp >= ? AND timestamp < ? "
                "GROUP BY d",
                (start_bound, end_bound),
            ).fetchall()
            return {r[0]: float(r[1]) for r in rows if r[1] is not None}
        finally:
            with contextlib.suppress(Exception):
                conn.close()
    except Exception as exc:
        _silent_log_safe("xspat_wa_outbound_failed", exc)
        return {}
