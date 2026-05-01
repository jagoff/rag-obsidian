"""Pattern detectors for the today brief — surface signals that don't
fit the existing buckets (gmail/wa/cal/yt) but emerge from comparing
TODAY against the recent baseline.

Three detectors, all consumed by `web/server.py` and exposed in
`payload.today.patterns` for the home dashboard:

- `detect_wa_spikes(today_chats, db_path, today_iso, ...)` — chats
  whose msg count today is sharply above their 7-day baseline. Useful
  to flag "che, María te escribió 35 veces hoy, normalmente son 5 —
  algo está pasando".

- `detect_wa_silences(db_path, today_iso, ...)` — people who used to
  message a lot in the last 7 days and TODAY haven't appeared. Useful
  for "hace 4 días que X no escribe — fijate si todo bien".

- `detect_topic_concentrations(correlations, min_sources=3)` — topics
  that appear in 3+ different sources (mails + WA + notes + YouTube,
  etc.). Already in `correlations.topics` but with default min=2; this
  bumps the bar so only "real" concentrations make it.

All detectors are read-only against the WhatsApp bridge SQLite (no
writes ever). They silent-fail if the DB is missing — `payload.today.
patterns` just lacks the corresponding key, frontend hides the chip.

The "spike" threshold is intentionally conservative: ratio ≥ 2.5×
baseline AND today_count ≥ 5. That filters out noise like "1 msg today
vs 0.3 avg" which is mathematically a 3× spike but uninteresting.

Aprendido el 2026-04-30: el correlator ya marcaba personas + temas
cross-source, pero el dashboard nunca los rendeaba como pattern
estructurado — quedaban dentro del prose del LLM. Esta capa los expone
como chips/cards independientes.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

# Mínimo de mensajes hoy para considerar un chat como "spike candidate".
# Un chat con 1-2 msgs hoy nunca es spike aunque el baseline sea 0,
# porque la métrica se rompe (división por casi-cero) y el ruido es
# más alto que la señal.
_SPIKE_MIN_TODAY = 5
# Ratio today/baseline para flagear como spike. 2.5× = "más del doble"
# del promedio. Empíricamente, valores menores generan demasiado
# ruido — un chat con baseline 4 msgs/día y 8 msgs hoy NO es noticia.
_SPIKE_RATIO = 2.5
# Umbral de "activity sostenida" para que un silencio cuente. Si una
# persona te escribió 3 msgs en 7 días y hoy 0, no es silencio: es lo
# normal. Solo flageamos los chats que SÍ tenían ritmo activo.
_SILENCE_MIN_7D = 15
# Cap de resultados para no inundar el dashboard.
_MAX_SPIKES = 5
_MAX_SILENCES = 5


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    """Open the WA bridge in read-only mode. Returns None on any error
    (missing file, corrupt, locked) — callers treat the bucket as empty.
    """
    if not db_path.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def _start_of_today(now: datetime | None = None) -> str:
    """ISO local-naive string at 00:00 of the current day. Matches how
    the WA bridge stores timestamps (no TZ suffix, local time). Same
    pattern used by `_fetch_whatsapp_today` in rag.integrations.whatsapp.
    """
    n = now or datetime.now()
    return n.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def detect_wa_spikes(
    today_chats: list[dict],
    db_path: Path,
    bot_jid: str,
    now: datetime | None = None,
) -> list[dict]:
    """Detect WA chats whose msg count today is ≥2.5× their 7-day baseline.

    Args:
        today_chats: output of `_fetch_whatsapp_today` or `_fetch_whatsapp_unread`
            — list of `{jid, name, count, last_snippet}`.
        db_path: WA bridge SQLite path.
        bot_jid: skip this JID (the bot's own group).
        now: anchor time, default `datetime.now()`. For tests.

    Returns: list of `{jid, name, today, avg_7d, ratio, last_snippet}`
    sorted by ratio desc, capped at `_MAX_SPIKES`.
    """
    if not today_chats:
        return []
    candidates = [
        c for c in today_chats
        if int(c.get("count") or 0) >= _SPIKE_MIN_TODAY
    ]
    if not candidates:
        return []
    con = _connect_ro(db_path)
    if con is None:
        return []
    today_start = _start_of_today(now)
    seven_d_ago = (now or datetime.now()).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    seven_d_ago = seven_d_ago.replace(day=seven_d_ago.day) if False else seven_d_ago  # noqa
    # Compute "7 days before today 00:00" using SQLite (avoids tz/dst foot-guns).
    out: list[dict] = []
    try:
        for c in candidates:
            jid = c.get("jid")
            if not jid or jid == bot_jid:
                continue
            row = con.execute(
                """
                SELECT count(*) AS n
                FROM messages
                WHERE chat_jid = ?
                  AND is_from_me = 0
                  AND datetime(timestamp) >= datetime(?, '-7 days')
                  AND datetime(timestamp) < datetime(?)
                """,
                (jid, today_start, today_start),
            ).fetchone()
            n_7d = int(row["n"]) if row and row["n"] is not None else 0
            avg_per_day = n_7d / 7.0 if n_7d > 0 else 0.0
            today_count = int(c.get("count") or 0)
            # Ratio: cuántas veces más mensajes hoy que un día promedio
            # de la semana pasada. Si avg=0 (chat nuevo), ratio infinito
            # → lo capeamos con un floor de 0.5 para que un chat de 6
            # msgs hoy con cero baseline cuente pero no domine la lista.
            denom = max(avg_per_day, 0.5)
            ratio = today_count / denom
            if ratio >= _SPIKE_RATIO:
                out.append({
                    "jid": jid,
                    "name": c.get("name") or "",
                    "today": today_count,
                    "avg_7d": round(avg_per_day, 1),
                    "ratio": round(ratio, 1),
                    "last_snippet": c.get("last_snippet") or "",
                })
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out.sort(key=lambda d: -d["ratio"])
    return out[:_MAX_SPIKES]


def detect_wa_silences(
    db_path: Path,
    bot_jid: str,
    now: datetime | None = None,
) -> list[dict]:
    """Detect WA chats with sustained 7-day activity but ZERO inbound today.

    Useful to flag "X te escribía todos los días y hoy nada, fijate".
    Threshold: ≥15 inbound msgs in last 7d (excluding today) AND 0
    today.

    Returns: list of `{jid, name, msgs_7d, last_msg_iso, hours_silent}`
    sorted by msgs_7d desc, capped at `_MAX_SILENCES`.
    """
    con = _connect_ro(db_path)
    if con is None:
        return []
    today_start = _start_of_today(now)
    out: list[dict] = []
    try:
        rows = con.execute(
            """
            SELECT
              m.chat_jid AS jid,
              (SELECT name FROM chats WHERE jid = m.chat_jid) AS name,
              count(*) AS msgs_7d,
              max(datetime(m.timestamp)) AS last_msg_iso
            FROM messages m
            WHERE m.is_from_me = 0
              AND datetime(m.timestamp) >= datetime(?, '-7 days')
              AND datetime(m.timestamp) < datetime(?)
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
              AND m.chat_jid NOT IN (
                SELECT chat_jid FROM messages
                WHERE is_from_me = 0
                  AND datetime(timestamp) >= datetime(?)
              )
            GROUP BY m.chat_jid
            HAVING msgs_7d >= ?
            ORDER BY msgs_7d DESC
            LIMIT ?
            """,
            (today_start, today_start, bot_jid, today_start,
             _SILENCE_MIN_7D, _MAX_SILENCES * 2),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    n = now or datetime.now()
    for r in rows:
        raw_name = (r["name"] or "").strip()
        jid_prefix = (r["jid"] or "").split("@")[0]
        display = raw_name or jid_prefix
        # Filter out unresolved @lid participants whose name is just
        # digits (same convention as `_fetch_whatsapp_unread`).
        if not any(ch.isalpha() for ch in display):
            continue
        last_iso = r["last_msg_iso"] or ""
        hours_silent = None
        if last_iso:
            try:
                last_dt = datetime.fromisoformat(last_iso.replace(" ", "T"))
                hours_silent = round((n - last_dt).total_seconds() / 3600.0, 1)
            except ValueError:
                pass
        out.append({
            "jid": r["jid"],
            "name": display,
            "msgs_7d": int(r["msgs_7d"]),
            "last_msg_iso": last_iso,
            "hours_silent": hours_silent,
        })
        if len(out) >= _MAX_SILENCES:
            break
    return out


def detect_topic_concentrations(
    correlations: dict | None,
    min_sources: int = 3,
) -> list[dict]:
    """Filter `correlations.topics` to only those with ≥`min_sources`
    sources. The correlator already returns topics ≥2 sources by default;
    this bumps the bar to 3 so only "real" cross-source concentrations
    surface as patterns. Caller passes the full correlations object.
    """
    if not correlations:
        return []
    topics = correlations.get("topics") or []
    out: list[dict] = []
    for t in topics:
        sources = t.get("sources") or []
        if len(sources) >= min_sources:
            out.append(t)
    return out


def detect_today_patterns(
    today_correlations: dict | None,
    today_chats: list[dict],
    db_path: Path,
    bot_jid: str,
    now: datetime | None = None,
) -> dict:
    """Top-level facade — runs all 3 detectors + lifts gaps from the
    correlator. Returns `{spikes, silences, concentrations, gaps}`.
    """
    spikes = detect_wa_spikes(today_chats, db_path, bot_jid, now)
    silences = detect_wa_silences(db_path, bot_jid, now)
    concentrations = detect_topic_concentrations(
        today_correlations, min_sources=3,
    )
    gaps = (today_correlations or {}).get("gaps") or []
    return {
        "spikes": spikes,
        "silences": silences,
        "concentrations": concentrations,
        "gaps": gaps,
    }
