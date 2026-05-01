"""Brief schedule auto-tuning (2026-04-29).

Reads `rag_brief_feedback` and recommends whether the morning/today/digest
brief plist schedules should shift forward when the user has been muting
them consistently in the first hour after delivery. Overrides are persisted
in `rag_brief_schedule_prefs` (consumed by `_services_spec()` at plist
generation time).

The full loop:

  1. The brief daemons (morning/today/digest) write the brief and push
     to WhatsApp with a footer `_brief:<vault_relpath>_`.
  2. The user reacts in RagNet → listener TS → `POST /api/brief/feedback`
     → `_record_brief_feedback` inserts a row in `rag_brief_feedback`.
  3. This module's `analyze_brief_feedback(brief_kind)` reads recent rows,
     filters by brief kind via the path heuristic in `_classify_brief_kind`,
     computes mute/positive ratio and (if mutes concentrate around the
     current schedule) suggests a +30min shift capped to a safe band.
  4. The auto-tune daemon (`rag brief schedule auto-tune --apply`,
     scheduled via `com.fer.obsidian-rag-brief-auto-tune.plist` Sunday
     03:00) writes the override into `rag_brief_schedule_prefs` and
     re-bootstraps the affected plist via launchctl. Next morning's run
     honours the new schedule. `rag setup` regenerates plists from spec;
     the spec reads the prefs table at generation time.

Safe bands prevent runaway shifts (e.g. shifting morning so far that it
becomes a midday brief):

  morning ∈ [06:30, 09:00]
  today   ∈ [18:00, 21:00]
  digest  ∈ [21:00, 23:30]

Defaults (when no override exists) match the historical hardcoded values
in `_morning_plist`/`_today_plist`/`_digest_plist`:

  morning: Mon-Fri 07:00
  today:   Mon-Fri 22:00
  digest:  Sunday  22:00

Note: `today`'s default (22:00) lives outside its safe band, so the
auto-tune logic treats it as "already as late as we'd ever want" and
won't shift it without an explicit reset to a band-internal slot first.
This is intentional — bands are normative for SHIFTS, not for the
hardcoded baseline that the user may have manually picked outside
the band.

This module is import-side-effect-free: it does not open SQL connections
nor mutate state at import time. All IO happens inside the public
functions, which lazy-import `_ragvec_state_conn` and `_silent_log` from
`rag/__init__.py` to avoid the circular import that would otherwise
pull this module into the rag bootstrap path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

# ── Safe bands ─────────────────────────────────────────────────────────────
# (hour, minute) inclusive bounds. Auto-tune NEVER shifts a schedule
# outside these. Hardcoded by design — these are operational invariants,
# not user-tunable preferences. If the user wants morning at 5am, they
# edit the plist by hand (and break the auto-tune contract knowingly).
SAFE_BANDS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "morning": ((6, 30), (9, 0)),
    "today":   ((18, 0), (21, 0)),
    "digest":  ((21, 0), (23, 30)),
}

# Default schedules (when no `rag_brief_schedule_prefs` row exists).
# MUST match the hardcoded values in the plist functions in
# `rag/__init__.py` — drift here would silently change the spec.
DEFAULT_SCHEDULES: dict[str, tuple[int, int]] = {
    "morning": (7, 0),
    "today":   (22, 0),
    "digest":  (22, 0),
}

VALID_BRIEF_KINDS: tuple[str, ...] = ("morning", "today", "digest")

# Threshold for the "should_shift" heuristic. Hardcoded for now; if we
# end up wanting per-kind tuning we'd promote them to env vars or rows
# in a meta table — but the math is simple enough that one set works.
MIN_MUTES_FIRST_HOUR = 3
MUTE_RATIO_THRESHOLD = 0.5
SHIFT_STEP_MINUTES = 30


# ── Path → brief_kind classification ──────────────────────────────────────

# Brief vault paths follow one of three patterns (observed in production
# 2026-04-29):
#
#   morning: 04-Archive/99-obsidian-system/99-AI/reviews/YYYY-MM-DD.md
#   today:   04-Archive/99-obsidian-system/99-AI/reviews/YYYY-MM-DD-evening.md
#   digest:  04-Archive/99-obsidian-system/99-AI/reviews/YYYY-WNN.md
#
# Tests (and earlier docs) sometimes use explicit `-morning.md` /
# `-digest.md` suffixes — the matcher below handles both naming styles.
_RE_DIGEST_DATE = re.compile(r"\d{4}-W\d{2}\.md$", re.IGNORECASE)
_RE_MORNING_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\.md$")


def _classify_brief_kind(dedup_key: str) -> str | None:
    """Return `morning|today|digest` for a brief vault path, or None.

    Heuristic in priority order:
      1. Explicit `-morning`/`-evening`/`-today`/`-digest` infix.
      2. ISO week pattern `YYYY-Wnn.md` → digest.
      3. ISO date pattern `YYYY-MM-DD.md` → morning.

    The ordering matters: `2026-04-29-evening.md` matches the `-evening`
    case BEFORE the date-only branch (which would also match the
    `2026-04-29` prefix), so today wins over morning for that file.
    """
    if not dedup_key:
        return None
    name = dedup_key.rsplit("/", 1)[-1].lower()
    # Explicit suffix infix wins.
    if "-morning" in name:
        return "morning"
    if "-evening" in name or "-today" in name:
        return "today"
    if "-digest" in name:
        return "digest"
    if _RE_DIGEST_DATE.search(name):
        return "digest"
    if _RE_MORNING_DATE.search(name):
        return "morning"
    return None


# ── Slots within a band ────────────────────────────────────────────────────

def _slots_in_band(band: tuple[tuple[int, int], tuple[int, int]]) -> list[tuple[int, int]]:
    """Return all (hour, minute) slots within `band` (inclusive) at
    `SHIFT_STEP_MINUTES` granularity. Used to enumerate candidate
    schedules when looking for a quieter slot."""
    start, end = band
    out: list[tuple[int, int]] = []
    h, m = start
    while (h, m) <= end:
        out.append((h, m))
        m += SHIFT_STEP_MINUTES
        while m >= 60:
            h += 1
            m -= 60
    return out


def _is_in_band(brief_kind: str, hour: int, minute: int) -> bool:
    """Return True iff (hour, minute) falls inside the safe band for
    `brief_kind`. Used by `set_brief_schedule_pref` to refuse writes
    that would shift outside the band — the public API is opinionated
    about this so a buggy caller can't silently break the contract."""
    band = SAFE_BANDS.get(brief_kind)
    if band is None:
        return False
    start, end = band
    return start <= (hour, minute) <= end


# ── Prefs IO (rag_brief_schedule_prefs) ────────────────────────────────────

def get_brief_schedule_pref(brief_kind: str) -> dict | None:
    """Read the override row for `brief_kind` from `rag_brief_schedule_prefs`.

    Returns `{hour, minute, last_updated, reason}` or None if no row.
    Silent-fail: if the table doesn't exist (e.g. fresh DB) or any other
    SQL error, returns None — callers (notably `_services_spec()`) treat
    that as "no override, use default".
    """
    if brief_kind not in VALID_BRIEF_KINDS:
        return None
    try:
        from rag import _ragvec_state_conn  # lazy: avoid circular import
    except Exception:
        return None
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT hour, minute, last_updated, reason "
                "FROM rag_brief_schedule_prefs WHERE brief_kind = ?",
                (brief_kind,),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {
        "hour": int(row[0]),
        "minute": int(row[1]),
        "last_updated": row[2],
        "reason": row[3],
    }


def set_brief_schedule_pref(
    brief_kind: str,
    hour: int,
    minute: int,
    *,
    reason: str | None = None,
) -> bool:
    """Upsert an override into `rag_brief_schedule_prefs`. Refuses writes
    outside the safe band for `brief_kind`. Returns True on success,
    False on validation/SQL failure.

    The `reason` column is human-readable diagnostic text (ej. "shift
    +30min: 5 mutes en hour=7 vs 1 en hour=8"). Surfaced in
    `rag brief schedule status` so the user can audit why a given
    schedule moved.
    """
    if brief_kind not in VALID_BRIEF_KINDS:
        return False
    if not _is_in_band(brief_kind, hour, minute):
        return False
    try:
        from rag import _ragvec_state_conn  # lazy
    except Exception:
        return False
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT INTO rag_brief_schedule_prefs"
                " (brief_kind, hour, minute, last_updated, reason)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(brief_kind) DO UPDATE SET"
                " hour=excluded.hour,"
                " minute=excluded.minute,"
                " last_updated=excluded.last_updated,"
                " reason=excluded.reason",
                (brief_kind, int(hour), int(minute), now, reason),
            )
            conn.commit()
        return True
    except Exception:
        return False


def reset_brief_schedule_pref(brief_kind: str) -> bool:
    """Delete the override row for `brief_kind`. The next plist
    regeneration falls back to `DEFAULT_SCHEDULES`. Returns True even
    if no row was present (idempotent reset). False only on SQL error.
    """
    if brief_kind not in VALID_BRIEF_KINDS:
        return False
    try:
        from rag import _ragvec_state_conn  # lazy
    except Exception:
        return False
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "DELETE FROM rag_brief_schedule_prefs WHERE brief_kind = ?",
                (brief_kind,),
            )
            conn.commit()
        return True
    except Exception:
        return False


def current_schedule(brief_kind: str) -> tuple[int, int]:
    """Return the (hour, minute) currently in effect for `brief_kind`:
    override if present, otherwise `DEFAULT_SCHEDULES[brief_kind]`.

    Centralised so the analyzer + the CLI status command + the plist
    spec all agree on what "current" means without each rolling their
    own fallback logic.
    """
    pref = get_brief_schedule_pref(brief_kind)
    if pref is not None:
        return (pref["hour"], pref["minute"])
    return DEFAULT_SCHEDULES.get(brief_kind, (0, 0))


# ── Analysis ───────────────────────────────────────────────────────────────

@dataclass
class _Counts:
    mute: int = 0
    negative: int = 0
    positive: int = 0
    hour_distribution_of_mutes: dict[int, int] = field(default_factory=dict)


def _read_feedback_rows(lookback_days: int) -> Iterable[tuple[str, str, str]]:
    """Yield (ts, dedup_key, rating) rows from `rag_brief_feedback` within
    the lookback window. Silent-fail to empty list if the table doesn't
    exist (fresh DB) or any SQL error.
    """
    try:
        from rag import _ragvec_state_conn
    except Exception:
        return []
    cutoff_iso = (datetime.now() - timedelta(days=int(lookback_days))).isoformat(timespec="seconds")
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT ts, dedup_key, rating FROM rag_brief_feedback "
                "WHERE ts >= ? ORDER BY ts ASC",
                (cutoff_iso,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []


def _hour_of_iso(ts: str) -> int | None:
    """Parse an ISO-8601 ts and return its hour (0-23), or None on bad
    input. Tolerant of the `'YYYY-MM-DDTHH:MM:SS'` and the lenient
    space-separated variant — both shapes appear in production rows
    depending on whether the writer used `isoformat()` with default
    `sep` or not.
    """
    if not ts:
        return None
    try:
        # `fromisoformat` accepts both 'T' and ' ' as separator since 3.11.
        return datetime.fromisoformat(ts).hour
    except Exception:
        # Fallback: regex-yank the HH chunk.
        m = re.search(r"[T ](\d{2}):", ts)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None


def _aggregate(rows: Iterable[tuple[str, str, str]], brief_kind: str) -> _Counts:
    """Filter rows to `brief_kind` (via path heuristic) and bucket by
    rating. Mutes additionally bucket by feedback hour for the
    `hour_distribution_of_mutes` field.
    """
    c = _Counts()
    for ts, dedup_key, rating in rows:
        kind = _classify_brief_kind(dedup_key or "")
        if kind != brief_kind:
            continue
        if rating == "mute":
            c.mute += 1
            h = _hour_of_iso(ts or "")
            if h is not None:
                c.hour_distribution_of_mutes[h] = (
                    c.hour_distribution_of_mutes.get(h, 0) + 1
                )
        elif rating == "negative":
            c.negative += 1
        elif rating == "positive":
            c.positive += 1
    return c


def _suggest_shift(
    brief_kind: str,
    current: tuple[int, int],
    mutes_by_hour: dict[int, int],
) -> tuple[tuple[int, int], str] | None:
    """Find the next slot within the safe band that has strictly fewer
    mutes-in-hour than `current`. Returns ((hour, minute), reason) or
    None if no such slot exists (band exhausted with no improvement).

    The "mutes in this slot" count uses the hour-of-feedback as a proxy
    for "the user disliked the brief at this hour". If the user mutes a
    7am brief at 7:15, that counts as a mute at hour=7. Shifting to
    7:30 doesn't change the hour bucket; this is fine — the heuristic
    is intentionally coarse, only graduating to the next hour when the
    +30min step crosses an hour boundary.
    """
    band = SAFE_BANDS.get(brief_kind)
    if band is None:
        return None
    slots = _slots_in_band(band)
    # Filter to slots strictly AFTER current — we never auto-tune
    # earlier (the heuristic is "user mutes morning briefs because
    # they're too early, push later").
    candidates = [s for s in slots if s > current]
    if not candidates:
        return None
    current_mute_cnt = mutes_by_hour.get(current[0], 0)
    for slot in candidates:
        slot_mute_cnt = mutes_by_hour.get(slot[0], 0)
        if slot_mute_cnt < current_mute_cnt:
            reason = (
                f"shift +30min: {current_mute_cnt} mute(s) en hour={current[0]} "
                f"vs {slot_mute_cnt} en hour={slot[0]}"
            )
            return (slot, reason)
    return None


def analyze_brief_feedback(brief_kind: str, lookback_days: int = 30) -> dict:
    """Analyse recent feedback for `brief_kind` and return a recommendation.

    Shape:
        {
          "brief_kind": str,
          "lookback_days": int,
          "mute_count": int,
          "negative_count": int,
          "positive_count": int,
          "hour_distribution_of_mutes": dict[int, int],
          "current_hour": int,
          "current_minute": int,
          "recommendation": {
            "should_shift": bool,
            "current_hour": int,
            "suggested_hour": int | None,
            "suggested_minute": int | None,
            "reason": str,
          },
        }

    Decision rule (matches the user spec 2026-04-29):
      - If `mute_count_first_hour` (mutes whose ts hour falls inside
        [current_hour, current_hour+1)) >= `MIN_MUTES_FIRST_HOUR` AND
      - mute_total / (mute_total + positive_total) > `MUTE_RATIO_THRESHOLD`
      → suggest shifting +30min iteratively until reaching a slot with
        fewer mutes (capped to `SAFE_BANDS[brief_kind]`).

    Returns a recommendation with `should_shift=False` (and a textual
    reason) when there's no data, when the ratio gate fails, or when no
    in-band slot improves on current. Never raises.
    """
    if brief_kind not in VALID_BRIEF_KINDS:
        return {
            "brief_kind": brief_kind,
            "lookback_days": int(lookback_days),
            "mute_count": 0,
            "negative_count": 0,
            "positive_count": 0,
            "hour_distribution_of_mutes": {},
            "current_hour": 0,
            "current_minute": 0,
            "recommendation": {
                "should_shift": False,
                "current_hour": 0,
                "suggested_hour": None,
                "suggested_minute": None,
                "reason": f"brief_kind inválido: {brief_kind!r}",
            },
        }

    rows = list(_read_feedback_rows(lookback_days))
    counts = _aggregate(rows, brief_kind)
    cur_h, cur_m = current_schedule(brief_kind)

    # Mutes within the first hour after the brief was sent. We use the
    # hour of the feedback ts, NOT the hour of the brief — this is the
    # signal the user actually emits ("I muted at 7:15am the brief that
    # arrived at 7:00am").
    mutes_first_hour = counts.hour_distribution_of_mutes.get(cur_h, 0)
    total_pos_mute = counts.positive + counts.mute
    ratio = (counts.mute / total_pos_mute) if total_pos_mute > 0 else 0.0

    rec_reason = ""
    suggested: tuple[int, int] | None = None

    if counts.mute == 0 and counts.positive == 0 and counts.negative == 0:
        rec_reason = "sin feedback en la ventana"
    elif mutes_first_hour < MIN_MUTES_FIRST_HOUR:
        rec_reason = (
            f"sólo {mutes_first_hour} mute(s) en hour={cur_h} "
            f"(< {MIN_MUTES_FIRST_HOUR}); no shift"
        )
    elif ratio <= MUTE_RATIO_THRESHOLD:
        rec_reason = (
            f"ratio mute/(mute+positive)={ratio:.2f} "
            f"(<= {MUTE_RATIO_THRESHOLD}); no shift"
        )
    else:
        suggestion = _suggest_shift(
            brief_kind, (cur_h, cur_m), counts.hour_distribution_of_mutes,
        )
        if suggestion is None:
            rec_reason = (
                f"sin slot mejor en banda {SAFE_BANDS[brief_kind]} "
                f"(actual {cur_h:02d}:{cur_m:02d})"
            )
        else:
            suggested, rec_reason = suggestion
            # Enrich rec_reason with panel feedback comments if available
            ft_comments = []
            try:
                from rag import _ragvec_state_conn  # lazy
                with _ragvec_state_conn() as conn:
                    cur = conn.execute("""
                        SELECT comment FROM rag_ft_panel_ratings
                        WHERE stream='brief' AND rating=-1
                          AND ts >= datetime('now', ?)
                          AND comment IS NOT NULL AND length(comment) > 0
                          AND item_id LIKE ?
                        ORDER BY ts DESC LIMIT 5
                    """, (f'-{lookback_days} days', f'%-{brief_kind}%'))
                    ft_comments = [row[0] for row in cur.fetchall()]
            except Exception:
                pass  # silent-fail
            if ft_comments:
                rec_reason = f"{rec_reason} | feedback panel: {' / '.join(ft_comments[:3])}"

    should_shift = suggested is not None
    return {
        "brief_kind": brief_kind,
        "lookback_days": int(lookback_days),
        "mute_count": counts.mute,
        "negative_count": counts.negative,
        "positive_count": counts.positive,
        "hour_distribution_of_mutes": dict(counts.hour_distribution_of_mutes),
        "current_hour": cur_h,
        "current_minute": cur_m,
        "recommendation": {
            "should_shift": should_shift,
            "current_hour": cur_h,
            "suggested_hour": suggested[0] if suggested else None,
            "suggested_minute": suggested[1] if suggested else None,
            "reason": rec_reason,
        },
    }


def analyze_all(lookback_days: int = 30) -> dict[str, dict]:
    """Convenience: run `analyze_brief_feedback` for every kind. Used by
    `rag brief schedule status` and the auto-tune daemon."""
    return {k: analyze_brief_feedback(k, lookback_days) for k in VALID_BRIEF_KINDS}
