"""Pillow (iOS sleep tracker) integration via iCloud-synced data dump.

Pillow Pro sincroniza un export de Core Data a iCloud Drive. El archivo
vive en `~/Library/Mobile Documents/com~apple~CloudDocs/Sueño/PillowData.txt`
(formato custom: cada entidad de Core Data en una sección, una fila por
línea con `ZFIELD -> value` separados por las propias keys uppercase).

## Surfaces

- `parse_pillow_dump(path)` — parser puro, devuelve list[dict] de sesiones
  nocturnas (excluye naps por default).
- `ingest()` — corre el ingester completo: parsea el archivo, UPSERT a
  `rag_sleep_sessions` por `uuid`, y emite señales de wake-up mood a
  `rag_mood_signals`. Idempotente; correr 1×/día via launchd.
- `last_night()` — devuelve la última sesión nocturna (no-nap) más reciente.
- `recent_nights(limit)` — devuelve las últimas N noches con campos clave.
- `weekly_stats()` — promedio últimos 7 días vs histórico (delta).
- `record_self_report_mood(mood, notes)` — endpoint helper para `/api/mood`,
  inserta una fila a `rag_mood_signals` con `source="manual"`.

## Invariantes

- Silent-fail: si el archivo no existe (Pillow no instalado o iCloud
  no sync) → ingester no-op, fetcher devuelve `None`. Nunca raise.
- UPSERT por uuid (ZUNIQUEIDENTIFIER del Core Data) → re-ingestar es free.
- Apple Core Data epoch = 2001-01-01 UTC. Convertimos a unix epoch al
  guardar para que los queries SQL usen las funciones standard.
- `date` field = YYYY-MM-DD del **end** de la sesión, en zona horaria
  local del usuario (typical Argentina/Cordoba). El user piensa en
  "anoche" como "el día en que me desperté", no "el día en que me acosté".

## Fuente del export

El user tiene Pillow Pro con sync automático a iCloud. El archivo se
actualiza cada noche post wake-up (~30min después). Si el sync falla
(iCloud quota, network), el ingester lee el archivo viejo — no es un
problema, las filas están idempotentes por uuid.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import statistics as _stats
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("rag.integrations.pillow_sleep")

# Default location of the iCloud-synced Pillow export. Overridable via env
# `RAG_PILLOW_DUMP_PATH` for testing or non-default iCloud setups.
_DEFAULT_DUMP_PATH = (
    Path.home()
    / "Library/Mobile Documents/com~apple~CloudDocs/Sueño/PillowData.txt"
)

# Apple Core Data epoch (CFAbsoluteTime reference date).
_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

# Wakeup mood scale en Pillow (ZWAKEUPMOOD): 0=neutral / not-set,
# 1-3 escala creciente. Mapeo a 0-1 para `rag_mood_signals.value`:
#   0 → omitir (no señal)
#   1 → 0.33  (mal)
#   2 → 0.66  (normal)
#   3 → 1.0   (bien)
_WAKEUP_MOOD_NORMALIZED = {1: 0.33, 2: 0.66, 3: 1.0}


def _dump_path() -> Path:
    """Resolves the Pillow dump path, honoring RAG_PILLOW_DUMP_PATH env override."""
    env = os.environ.get("RAG_PILLOW_DUMP_PATH")
    return Path(env) if env else _DEFAULT_DUMP_PATH


def _apple_to_unix(value: float) -> float:
    """Convert Apple CFAbsoluteTime (sec since 2001-01-01 UTC) to unix epoch."""
    return value + _APPLE_EPOCH.timestamp()


def _parse_row(line: str) -> dict[str, str]:
    """Each row is a sequence of `ZFIELD -> value` separated by the next field's
    key. We split with a regex that captures keys (uppercase Z-prefixed)
    followed by ` -> ` and zip the alternating list into a dict.
    """
    parts = re.split(r"(Z[A-Z0-9_]+)\s+->\s+", line.strip())
    out: dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        out[parts[i]] = parts[i + 1].strip()
        i += 2
    return out


@dataclass
class SleepSession:
    pk: int
    uuid: str
    start_ts: float          # unix epoch UTC
    end_ts: float
    date: str                # YYYY-MM-DD (local TZ del end)
    is_nap: bool
    is_edited: bool
    quality: float | None
    fatigue: float | None
    wakeup_mood: int | None
    awakenings: int | None
    snoozes: int | None
    time_awake_s: float | None
    deep_s: float | None
    light_s: float | None
    rem_s: float | None
    time_to_sleep_s: float | None
    device: str
    used_watch: bool
    tz: str

    @property
    def sleep_total_s(self) -> float:
        """Suma de stages dormido (deep + light + REM). Si no hay stages,
        cae al `session_dur - time_awake`."""
        stages = (self.deep_s or 0) + (self.light_s or 0) + (self.rem_s or 0)
        if stages > 0:
            return stages
        return max(0.0, (self.end_ts - self.start_ts) - (self.time_awake_s or 0))

    def to_row(self, source_file: str, ingested_at: float) -> tuple:
        return (
            self.pk,
            self.uuid,
            self.start_ts,
            self.end_ts,
            self.date,
            int(self.is_nap),
            int(self.is_edited),
            self.quality,
            self.fatigue,
            self.wakeup_mood,
            self.awakenings,
            self.snoozes,
            self.time_awake_s,
            self.deep_s,
            self.light_s,
            self.rem_s,
            self.time_to_sleep_s,
            self.device,
            int(self.used_watch),
            self.tz,
            source_file,
            ingested_at,
        )


def _local_date_from_ts(ts: float, tz_name: str) -> str:
    """Convert unix ts to YYYY-MM-DD in the session's local TZ. Falls back
    to system local TZ if the named tz isn't loadable (e.g., zoneinfo
    not available on Windows or rare Pillow tz string)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        tz = None
    dt = datetime.fromtimestamp(ts, tz=tz) if tz else datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d")


def parse_pillow_dump(
    path: Path | str | None = None,
    *,
    include_naps: bool = False,
) -> list[SleepSession]:
    """Parse the Pillow export file and return a list of SleepSession.

    Returns empty list if the file doesn't exist or has zero parseable rows.
    Never raises on bad rows — we skip them silently with a debug log.
    """
    p = Path(path) if path else _dump_path()
    if not p.is_file():
        logger.debug("pillow dump not found at %s", p)
        return []

    sessions: list[SleepSession] = []
    current_entity: str | None = None

    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                stripped = line.strip()
                # Entity headers son tokens uppercase con prefijo Z, en
                # línea propia (sin ` -> `). Ej. ZSLEEPSESSION, ZALARM.
                # Las link tables tipo Z_5SLEEPSESSION también caen acá
                # cuando aparecen como header — el regex permite dígitos
                # tras el guión bajo. Las propias filas de datos arrancan
                # con `Z_PK -> N...` y NO matchean fullmatch (tienen `->`).
                if "->" not in stripped and re.fullmatch(r"Z[A-Z0-9_]+", stripped):
                    current_entity = stripped
                    continue
                if current_entity != "ZSLEEPSESSION":
                    continue

                row = _parse_row(line)
                try:
                    s_apple = float(row.get("ZSTARTTIME", "0"))
                    e_apple = float(row.get("ZENDTIME", "0"))
                except (ValueError, TypeError):
                    continue
                if s_apple <= 0 or e_apple <= 0:
                    continue

                start_ts = _apple_to_unix(s_apple)
                end_ts = _apple_to_unix(e_apple)
                tz = row.get("ZTIMEZONEIDENTIFIER", "")
                is_nap = row.get("ZISNAP") == "1"
                if is_nap and not include_naps:
                    continue

                def _f(key: str) -> float | None:
                    v = row.get(key)
                    if v is None or v == "":
                        return None
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None

                def _i(key: str) -> int | None:
                    v = row.get(key)
                    if v is None or v == "":
                        return None
                    try:
                        return int(float(v))
                    except (ValueError, TypeError):
                        return None

                try:
                    pk = int(row.get("Z_PK", "0"))
                except (ValueError, TypeError):
                    continue
                uuid = row.get("ZUNIQUEIDENTIFIER", "")
                if not uuid:
                    continue

                sessions.append(
                    SleepSession(
                        pk=pk,
                        uuid=uuid,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        date=_local_date_from_ts(end_ts, tz),
                        is_nap=is_nap,
                        is_edited=row.get("ZISEDITED") == "1",
                        quality=_f("ZSLEEPQUALITY"),
                        fatigue=_f("ZFATIGUE"),
                        wakeup_mood=_i("ZWAKEUPMOOD"),
                        awakenings=_i("ZNUMBEROFAWAKENINGS"),
                        snoozes=_i("ZNUMBEROFSNOOZES"),
                        time_awake_s=_f("ZTIMEAWAKE"),
                        deep_s=_f("ZTIMEINDEEPSLEEP"),
                        light_s=_f("ZTIMEINLIGHTSLEEP"),
                        rem_s=_f("ZTIMEINREMSLEEP"),
                        time_to_sleep_s=_f("ZTIMETOSLEEP"),
                        device=row.get("ZDEVICEUSED", ""),
                        used_watch=row.get("ZUSEDAPPLEWATCH") == "1",
                        tz=tz,
                    )
                )
    except OSError as exc:
        logger.warning("failed to read pillow dump: %s", exc)
        return []

    sessions.sort(key=lambda s: s.start_ts)
    return sessions


# ─── Ingester (writes to telemetry.db) ─────────────────────────────────


def _telemetry_conn():
    """Helper — reuses the canonical connector defined in `rag/__init__.py`.
    Lazy import to avoid circulars."""
    from rag import _ragvec_state_conn  # type: ignore
    return _ragvec_state_conn()


def _upsert_session(conn, session: SleepSession, source_file: str, now: float) -> bool:
    """Insert or update one session. Returns True if it was a new insert."""
    row = session.to_row(source_file, now)
    cur = conn.execute(
        "INSERT INTO rag_sleep_sessions ("
        " pk, uuid, start_ts, end_ts, date, is_nap, is_edited,"
        " quality, fatigue, wakeup_mood, awakenings, snoozes,"
        " time_awake_s, deep_s, light_s, rem_s, time_to_sleep_s,"
        " device, used_watch, tz, source_file, ingested_at"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(uuid) DO UPDATE SET"
        " pk=excluded.pk,"
        " start_ts=excluded.start_ts,"
        " end_ts=excluded.end_ts,"
        " date=excluded.date,"
        " is_nap=excluded.is_nap,"
        " is_edited=excluded.is_edited,"
        " quality=excluded.quality,"
        " fatigue=excluded.fatigue,"
        " wakeup_mood=excluded.wakeup_mood,"
        " awakenings=excluded.awakenings,"
        " snoozes=excluded.snoozes,"
        " time_awake_s=excluded.time_awake_s,"
        " deep_s=excluded.deep_s,"
        " light_s=excluded.light_s,"
        " rem_s=excluded.rem_s,"
        " time_to_sleep_s=excluded.time_to_sleep_s,"
        " device=excluded.device,"
        " used_watch=excluded.used_watch,"
        " tz=excluded.tz,"
        " source_file=excluded.source_file,"
        " ingested_at=excluded.ingested_at"
        " WHERE excluded.ingested_at > rag_sleep_sessions.ingested_at",
        row,
    )
    # rowcount semantics for UPSERT in SQLite: 1 always (insert OR update).
    # Better signal: SELECT before to know if it was new. For perf we skip
    # — caller doesn't need this distinction in the common path.
    return cur.rowcount > 0


def _emit_wakeup_mood_signal(conn, session: SleepSession) -> None:
    """If the session has a wakeup_mood ≥ 1, write it as a `rag_mood_signals`
    row with source="pillow", signal_kind="wakeup_mood". Idempotent: dedupe
    by (date, source, signal_kind) — re-ingesting the same night doesn't
    duplicate the signal.
    """
    if session.wakeup_mood is None or session.wakeup_mood not in _WAKEUP_MOOD_NORMALIZED:
        return
    value = _WAKEUP_MOOD_NORMALIZED[session.wakeup_mood]
    evidence = json.dumps({
        "session_uuid": session.uuid,
        "raw_score": session.wakeup_mood,
        "quality": session.quality,
        "duration_h": round(session.sleep_total_s / 3600, 2),
    })
    # Dedupe: delete any prior signal for the same (date, source, kind)
    # before inserting. Cheaper than ON CONFLICT pattern for the
    # already-existing schema (no UNIQUE on those cols by design — the
    # table is append-only for other sources).
    conn.execute(
        "DELETE FROM rag_mood_signals "
        " WHERE date=? AND source='pillow' AND signal_kind='wakeup_mood'",
        (session.date,),
    )
    conn.execute(
        "INSERT INTO rag_mood_signals (ts, date, source, signal_kind, value, weight, evidence) "
        "VALUES (?, ?, 'pillow', 'wakeup_mood', ?, 1.0, ?)",
        (session.end_ts, session.date, value, evidence),
    )

    # Fatigue signal (if reported and >0). Inverted: low fatigue = high mood.
    if session.fatigue is not None and session.fatigue > 0:
        fatigue_value = max(0.0, min(1.0, 1.0 - session.fatigue))
        fatigue_ev = json.dumps({
            "session_uuid": session.uuid,
            "raw_fatigue": session.fatigue,
        })
        conn.execute(
            "DELETE FROM rag_mood_signals "
            " WHERE date=? AND source='pillow' AND signal_kind='fatigue'",
            (session.date,),
        )
        conn.execute(
            "INSERT INTO rag_mood_signals (ts, date, source, signal_kind, value, weight, evidence) "
            "VALUES (?, ?, 'pillow', 'fatigue', ?, 0.5, ?)",
            (session.end_ts, session.date, fatigue_value, fatigue_ev),
        )


def ingest(path: Path | str | None = None) -> dict[str, Any]:
    """Run the full Pillow → SQLite ingester. Idempotent; safe to run repeatedly.

    Returns a dict with counts:
      `{file, total_parsed, ingested, mood_signals, elapsed_ms}`
    """
    t0 = time.time()
    p = Path(path) if path else _dump_path()
    sessions = parse_pillow_dump(p)
    if not sessions:
        return {
            "file": str(p),
            "total_parsed": 0,
            "ingested": 0,
            "mood_signals": 0,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "skipped": True,
            "reason": "file_not_found_or_empty",
        }

    now = time.time()
    ingested = 0
    moods = 0
    with _telemetry_conn() as conn:
        # BEGIN/COMMIT around the whole batch — single fsync at the end.
        conn.execute("BEGIN")
        try:
            for s in sessions:
                _upsert_session(conn, s, str(p), now)
                ingested += 1
                # Pre-emit cleanup: drop any cached mood signals for this
                # session before re-emitting (handles the case where the
                # user changed their wake-up mood in Pillow after the
                # initial ingest).
                _emit_wakeup_mood_signal(conn, s)
                if s.wakeup_mood and s.wakeup_mood in _WAKEUP_MOOD_NORMALIZED:
                    moods += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return {
        "file": str(p),
        "total_parsed": len(sessions),
        "ingested": ingested,
        "mood_signals": moods,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


# ─── Read-side helpers (used by web/server.py fetcher) ──────────────────


def _row_to_dict(row) -> dict[str, Any]:
    """Convert sqlite Row to plain dict + computed `sleep_total_h`."""
    d = dict(row)
    deep = d.get("deep_s") or 0
    light = d.get("light_s") or 0
    rem = d.get("rem_s") or 0
    stages = deep + light + rem
    if stages > 0:
        d["sleep_total_s"] = stages
    else:
        d["sleep_total_s"] = max(0.0, (d["end_ts"] - d["start_ts"]) - (d.get("time_awake_s") or 0))
    d["sleep_total_h"] = d["sleep_total_s"] / 3600
    if stages > 0:
        d["deep_pct"] = deep / stages * 100
        d["rem_pct"] = rem / stages * 100
        d["light_pct"] = light / stages * 100
    else:
        d["deep_pct"] = d["rem_pct"] = d["light_pct"] = None
    # Local-time start/end strings for UI rendering — we already stored
    # date in local TZ, just format the clock part from the unix ts using
    # the same tz we saved.
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(d.get("tz") or "")
    except Exception:
        tz = None
    bed = datetime.fromtimestamp(d["start_ts"], tz=tz) if tz else datetime.fromtimestamp(d["start_ts"])
    wake = datetime.fromtimestamp(d["end_ts"], tz=tz) if tz else datetime.fromtimestamp(d["end_ts"])
    d["bedtime_local"] = bed.strftime("%H:%M")
    d["waketime_local"] = wake.strftime("%H:%M")
    return d


def last_night() -> dict[str, Any] | None:
    """Returns the most recent non-nap sleep session as a dict, or None
    if the table is empty."""
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM rag_sleep_sessions "
            " WHERE is_nap=0 ORDER BY end_ts DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row) if row else None


def recent_nights(limit: int = 14) -> list[dict[str, Any]]:
    """Last N nights ordered most-recent-first."""
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM rag_sleep_sessions "
            " WHERE is_nap=0 ORDER BY end_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def weekly_stats() -> dict[str, Any]:
    """Compute averages for last 7 days vs all-time, and return delta.

    Returns:
      `{week: {duration_h, quality, deep_pct, awakenings, n},
        hist: {...},
        delta: {duration_h, quality, deep_pct, awakenings},
        spark_quality_7d: [floats],
        spark_duration_7d: [floats],
        worst_recent: dict | None}`
    """
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        all_rows = conn.execute(
            "SELECT * FROM rag_sleep_sessions WHERE is_nap=0 ORDER BY end_ts ASC"
        ).fetchall()
    nights = [_row_to_dict(r) for r in all_rows]
    if not nights:
        return {"week": {}, "hist": {}, "delta": {}, "spark_quality_7d": [], "spark_duration_7d": [], "worst_recent": None}

    # Bucket: last 7 calendar days vs everything before.
    last_end = nights[-1]["end_ts"]
    seven_days_ago = last_end - 7 * 86400
    week = [n for n in nights if n["end_ts"] >= seven_days_ago]
    hist = nights  # incluyo histórico completo (incluye los últimos 7d) para
                   # baseline estable; el delta sale de week vs hist.

    def _avg(rows: Iterable[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return _stats.mean(vals) if vals else None

    def _summary(rows: list[dict]) -> dict:
        durs = [r["sleep_total_h"] for r in rows if r["sleep_total_s"] > 0]
        qs = [r["quality"] for r in rows if r.get("quality")]
        deeps = [r["deep_pct"] for r in rows if r.get("deep_pct") is not None]
        rems = [r["rem_pct"] for r in rows if r.get("rem_pct") is not None]
        aws = [r["awakenings"] for r in rows if r.get("awakenings") is not None]
        return {
            "n": len(rows),
            "duration_h": _stats.mean(durs) if durs else None,
            "quality": _stats.mean(qs) if qs else None,
            "deep_pct": _stats.mean(deeps) if deeps else None,
            "rem_pct": _stats.mean(rems) if rems else None,
            "awakenings": _stats.mean(aws) if aws else None,
        }

    week_s = _summary(week)
    hist_s = _summary(hist)
    delta = {
        k: ((week_s.get(k) or 0) - (hist_s.get(k) or 0))
        for k in ("duration_h", "quality", "deep_pct", "rem_pct", "awakenings")
        if week_s.get(k) is not None and hist_s.get(k) is not None
    }

    # Sparklines: una entry por día calendario en los últimos 7 días.
    # Si el user no durmió un día, el slot queda en None (el frontend lo
    # muestra como gap visual).
    spark_q: list[float | None] = []
    spark_d: list[float | None] = []
    spark_dates: list[str] = []
    today = datetime.fromtimestamp(last_end).date()
    for d_offset in range(6, -1, -1):
        d = today - timedelta(days=d_offset)
        dstr = d.strftime("%Y-%m-%d")
        spark_dates.append(dstr)
        n = next((x for x in week if x["date"] == dstr), None)
        spark_q.append(round(n["quality"], 2) if n and n.get("quality") else None)
        spark_d.append(round(n["sleep_total_h"], 2) if n and n["sleep_total_s"] > 0 else None)

    # Worst recent: peor noche por quality en los últimos 7 días.
    worst = None
    bad = [r for r in week if r.get("quality")]
    if bad:
        worst = min(bad, key=lambda r: r["quality"])

    return {
        "week": week_s,
        "hist": hist_s,
        "delta": delta,
        "spark_quality_7d": spark_q,
        "spark_duration_7d": spark_d,
        "spark_dates_7d": spark_dates,
        "worst_recent": worst,
    }


def latest_self_report_mood(hours: int = 12) -> dict[str, Any] | None:
    """Read the most recent self-reported mood (manual input via /api/mood)
    within the last `hours`. Returns None if no recent self-report.
    """
    cutoff = time.time() - hours * 3600
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT ts, value, evidence FROM rag_mood_signals "
            " WHERE source='manual' AND signal_kind='self_report' AND ts >= ? "
            " ORDER BY ts DESC LIMIT 1",
            (cutoff,),
        ).fetchone()
    if not row:
        return None
    ev = {}
    with contextlib.suppress(Exception):
        ev = json.loads(row["evidence"]) if row["evidence"] else {}
    return {"ts": row["ts"], "value": row["value"], "label": ev.get("label"), "notes": ev.get("notes")}


def record_self_report_mood(mood: str, notes: str | None = None) -> dict[str, Any]:
    """Persist a manual mood report. `mood` must be one of {good, meh, bad}.

    Writes to `rag_mood_signals` with source="manual", signal_kind="self_report".
    Multiple reports in the same day are all recorded (timeline of mood
    through the day) — the panel only shows the most recent one.

    Returns: `{ok, ts, value, label}` or `{ok: False, error}`.
    """
    mapping = {"good": 1.0, "meh": 0.5, "bad": 0.0}
    if mood not in mapping:
        return {"ok": False, "error": f"mood must be one of {list(mapping)}"}
    value = mapping[mood]
    now = time.time()
    date = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    evidence = json.dumps({"label": mood, "notes": notes or ""})
    with _telemetry_conn() as conn:
        conn.execute(
            "INSERT INTO rag_mood_signals (ts, date, source, signal_kind, value, weight, evidence) "
            "VALUES (?, ?, 'manual', 'self_report', ?, 1.0, ?)",
            (now, date, value, evidence),
        )
    return {"ok": True, "ts": now, "value": value, "label": mood}


# ─── Correlation / pattern detection ──────────────────────────────────


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Pearson correlation coefficient r ∈ [-1, +1]. Returns (r, n_pairs).
    `n_pairs` is the count of (x, y) tuples where neither is None.
    Returns (0.0, n) when n < 3 or variance is zero — caller should
    interpret r=0 with n<3 as "not enough data" rather than "no
    correlation".
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return 0.0, n
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    sx = sum((p[0] - mean_x) ** 2 for p in pairs)
    sy = sum((p[1] - mean_y) ** 2 for p in pairs)
    if sx <= 0 or sy <= 0:
        return 0.0, n
    cov = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
    return cov / (sx * sy) ** 0.5, n


def _bedtime_normalized(start_ts: float, tz_name: str) -> float:
    """Bedtime as hours-after-noon (so 22:00 → 22, 01:00 → 25). Local TZ.

    Returning a continuous scalar makes Pearson tractable across the
    midnight wraparound — without it, a 23:30 / 00:30 pair would look
    23-hour-apart instead of 1-hour-apart.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        tz = None
    dt = datetime.fromtimestamp(start_ts, tz=tz) if tz else datetime.fromtimestamp(start_ts)
    h = dt.hour + dt.minute / 60.0
    return h + 24 if h < 12 else h


def detect_patterns(min_n: int = 14, min_abs_r: float = 0.3) -> list[dict[str, Any]]:
    """Compute correlations on the user's whole sleep history and return
    the strongest findings as a list of dicts ready to render in the UI
    or feed the brief.

    Each finding has:
      `{kind, description, r, n, severity}`
    where severity is "weak" (|r|<0.3), "moderate" (<0.5) or "strong" (≥0.5).

    Filters:
      - n ≥ `min_n` (default 14 — at least 2 weeks of data)
      - |r| ≥ `min_abs_r` (default 0.3 — drop weak noise)

    Returns empty list if there aren't enough sessions yet.
    """
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM rag_sleep_sessions WHERE is_nap=0 ORDER BY end_ts ASC"
        ).fetchall()
    if len(rows) < min_n:
        return []
    nights = [_row_to_dict(r) for r in rows]

    # Para bedtime usamos el normalizado continuo en lugar del raw clock.
    bedtimes = [_bedtime_normalized(n["start_ts"], n.get("tz", "")) for n in nights]
    qualities = [n.get("quality") for n in nights]
    deeps = [n.get("deep_pct") for n in nights]
    rems = [n.get("rem_pct") for n in nights]
    awakenings = [n.get("awakenings") for n in nights]
    durations = [n.get("sleep_total_h") for n in nights]
    moods = [n.get("wakeup_mood") for n in nights]

    # Cross-source mood signals — manual self-reports en `rag_mood_signals`.
    # Asociamos cada sesión a su date para correlar wakeup_quality vs
    # mood manual del mismo día.
    with _telemetry_conn() as conn:
        import sqlite3
        conn.row_factory = sqlite3.Row
        manual_rows = conn.execute(
            "SELECT date, AVG(value) AS avg_mood, COUNT(*) AS n "
            " FROM rag_mood_signals "
            " WHERE source='manual' AND signal_kind='self_report' "
            " GROUP BY date"
        ).fetchall()
    manual_by_date: dict[str, float] = {r["date"]: r["avg_mood"] for r in manual_rows}
    manual_moods = [manual_by_date.get(n["date"]) for n in nights]

    candidates = [
        ("bedtime↔quality", "bedtime más tarde → quality {direction}",
         bedtimes, qualities),
        ("bedtime↔deep_pct", "bedtime más tarde → deep% {direction}",
         bedtimes, deeps),
        ("duration↔quality", "más horas dormidas → quality {direction}",
         durations, qualities),
        ("awakenings↔quality", "más awakenings → quality {direction}",
         awakenings, qualities),
        ("deep_pct↔wakeup_mood", "más deep → mood al despertar {direction}",
         deeps, moods),
        ("quality↔wakeup_mood", "quality alta → mood al despertar {direction}",
         qualities, moods),
        ("rem_pct↔wakeup_mood", "más REM → mood al despertar {direction}",
         rems, moods),
        ("quality↔manual_mood", "quality alta → mood manual del día {direction}",
         qualities, manual_moods),
    ]

    findings: list[dict[str, Any]] = []
    for kind, template, xs, ys in candidates:
        r, n = _pearson(xs, ys)
        if n < min_n or abs(r) < min_abs_r:
            continue
        direction = "sube" if r > 0 else "baja"
        severity = (
            "strong" if abs(r) >= 0.5
            else "moderate" if abs(r) >= 0.4
            else "weak"
        )
        findings.append({
            "kind": kind,
            "description": template.format(direction=direction),
            "r": round(r, 2),
            "n": n,
            "severity": severity,
        })

    # Sort by strength (|r|) descending — strongest patterns first.
    findings.sort(key=lambda f: -abs(f["r"]))
    return findings


def patterns_summary(top: int = 3) -> dict[str, Any]:
    """Lightweight wrapper for the home panel: top N patterns + counts.

    Returns:
      `{n_findings, top: list[finding]}`
    or `{n_findings: 0, top: []}` if there aren't enough nights yet.
    """
    findings = detect_patterns()
    return {"n_findings": len(findings), "top": findings[:top]}


__all__ = [
    "SleepSession",
    "parse_pillow_dump",
    "ingest",
    "last_night",
    "recent_nights",
    "weekly_stats",
    "latest_self_report_mood",
    "record_self_report_mood",
    "detect_patterns",
    "patterns_summary",
]
