"""Unified pendientes dashboard — extracted from `rag/__init__.py` (Phase 5 of monolith split, 2026-04-25).

Single mid-day view combining every "awaiting my attention" signal across
the integrations: Gmail awaiting-reply + Apple Mail unread + WhatsApp chats
sin responder + Reminders overdue/today/undated + vault followup loops +
new contradictions + today's calendar + low-confidence queries.

Morning brief already touches these but dispersed across sections and
diluted by the LLM narrative. `rag pendientes` is code-owned, deterministic,
fast (<3s on warm caches, no LLM call) — designed for a mid-day check-in.

## Architecture

- `_pendientes_collect(col, now, days, *, progress)` — orchestrator. Spawns
  9 fetchers concurrently via `ThreadPoolExecutor` (each is independent
  blocking I/O — osascript, subprocess, SQLite, HTTP, file reads — so the
  GIL releases and we get ~9× parallelism). Per-future silent-fail.
- 4 `_pendientes_*` helpers for the data shaping (`extract_loops_fast`,
  `recent_contradictions`, `low_conf_queries`, `urgent`).
- `pendientes` — Click command `rag pendientes [--days N] [--plain]`.
- 2 renderers: `_pendientes_render_plain` (newline-separated for bots) and
  `_pendientes_render_rich` (Rich-styled for terminal).

## Why deferred imports

This module references *a lot* of helpers that live in `rag/__init__.py`:
- Integrations (already re-exported): `_fetch_mail_unread`,
  `_fetch_reminders_due`, `_fetch_calendar_today`, `_fetch_whatsapp_unread`,
  `_fetch_weather_rain`, `_fetch_gmail_evidence`.
- Vault helpers: `_resolve_vault_path`, `is_excluded`, `_note_created_ts`,
  `_extract_followup_loops`, `_format_gmail_from`.
- SQL: `_ragvec_state_conn`, `_sql_query_window`, `_sql_read_with_retry`.
- Module-level: `VAULT_PATH`, `LOG_PATH`, `CONTRADICTION_LOG_PATH`, `get_db`,
  `log_query_event`, `_round_timing_ms`, `CONFIDENCE_RERANK_MIN`.

All resolved via `from rag import X` inside function bodies so:
1. Tests can `monkeypatch.setattr(rag, "_pendientes_extract_loops_fast", ...)`
   and similar — `from rag import X` re-evaluates on every call.
2. We avoid import-cycle issues when this module loads.

## Tests-friendly: lambdas resolve through `rag.<X>`

`test_home_progress_stream.py` patches `rag._pendientes_extract_loops_fast`,
`rag._pendientes_recent_contradictions`, and `rag._pendientes_low_conf_queries`
to assert the progress callback fires for each fetcher. The lambdas inside
`_pendientes_collect`'s task dict resolve those names via `rag.<X>` (not
local references) so the monkey-patches propagate at call time.
"""

from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import click
from rich.rule import Rule

from rag import cli, console


def _pendientes_collect(
    col,
    now: datetime,
    days: int,
    *,
    progress: "Callable[[str, str, float, str | None], None] | None" = None,
) -> dict:
    """Pure evidence collection — caller renders. Easier to test.

    Fetches dispatch concurrently via ThreadPoolExecutor: each source is
    independent I/O (osascript, subprocess, SQLite, HTTP, file reads) so the
    GIL releases during each call and 9 ~1s fetches drop from ~10s serial
    to the slowest single fetch (~2s). Per-future silent-fail preserved.

    `progress` (optional): callback used by the SSE stream endpoint to
    surface sub-stage timings. Invoked as
    `progress(stage_name, status, elapsed_ms, error_message)` where
    `status ∈ {"start","done","error"}`. The web-side wrapper namespaces
    these as `signals.<task>` (e.g. `signals.gmail`, `signals.reminders`)
    so the UI's chip strip can show which inner fetcher of `signals` is
    the actual bottleneck (typically `gmail` cold + `whatsapp` cold).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Resolve through `rag` so test monkey-patches on these symbols
    # (rag._fetch_mail_unread, rag._pendientes_extract_loops_fast, etc.)
    # propagate to the lambdas below.
    import rag as _rag

    stale_days = 14

    def _loops() -> dict:
        loops_all = _rag._pendientes_extract_loops_fast(
            _rag.VAULT_PATH, days=days, max_items=40,
        )
        loops_all.sort(key=lambda x: x.get("age_days", 0), reverse=True)
        return {
            "loops_stale": [x for x in loops_all if x.get("age_days", 0) >= stale_days][:5],
            "loops_activo": [x for x in loops_all if x.get("age_days", 0) < stale_days][:5],
        }

    tasks: dict[str, callable] = {
        "mail_unread":    _rag._fetch_mail_unread,
        "reminders":      lambda: _rag._fetch_reminders_due(now, horizon_days=1, max_items=30),
        "calendar":       _rag._fetch_calendar_today,
        "whatsapp":       lambda: _rag._fetch_whatsapp_unread(hours=24, max_chats=8),
        "weather":        _rag._fetch_weather_rain,
        "gmail":          lambda: _rag._fetch_gmail_evidence(now),
        "loops":          _loops,
        "contradictions": lambda: _rag._pendientes_recent_contradictions(
            _rag.CONTRADICTION_LOG_PATH, now, days=days,
        ),
        "low_conf":       lambda: _rag._pendientes_low_conf_queries(
            _rag.LOG_PATH, now, days=days,
        ),
    }

    def _timed(key: str, fn):
        """Wrap a fetcher so the progress callback (if any) sees it
        start + finish. Errors swallow per-source (preserve historical
        silent-fail contract) but emit an `error` progress event so the
        UI can show ⚠ on the chip."""
        if progress is not None:
            with contextlib.suppress(Exception):
                progress(key, "start", 0.0, None)
        t0 = time.time()
        try:
            result = fn()
            if progress is not None:
                elapsed_ms = (time.time() - t0) * 1000.0
                with contextlib.suppress(Exception):
                    progress(key, "done", elapsed_ms, None)
            return result
        except Exception as exc:
            if progress is not None:
                elapsed_ms = (time.time() - t0) * 1000.0
                with contextlib.suppress(Exception):
                    progress(key, "error", elapsed_ms, str(exc))
            raise

    ev: dict = {}
    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="pendientes") as pool:
        futures = {pool.submit(_timed, key, fn): key for key, fn in tasks.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                result = fut.result()
            except Exception:
                continue  # silent-fail per source (already emitted via progress)
            if key == "loops":
                ev.update(result)  # expands to loops_stale + loops_activo
            else:
                ev[key] = result
    return ev


def _pendientes_extract_loops_fast(
    vault: Path, days: int = 14, max_items: int = 40,
) -> list[dict]:
    """Walk vault for notes modified in last `days`, extract loops via
    `_extract_followup_loops`, attach age_days. No LLM. Silent-fail per note.
    """
    from rag import _extract_followup_loops, _note_created_ts, is_excluded
    if not vault.is_dir():
        return []
    now = datetime.now()
    start = now - timedelta(days=days)
    out: list[dict] = []
    for p in vault.rglob("*.md"):
        try:
            rel = str(p.relative_to(vault))
        except ValueError:
            continue
        if is_excluded(rel):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        mtime = st.st_mtime
        if datetime.fromtimestamp(mtime) < start:
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        extracted_ts = _note_created_ts(raw, mtime)
        loops = _extract_followup_loops(raw, rel, extracted_ts)
        for loop in loops:
            try:
                ex_dt = datetime.fromisoformat(loop.get("extracted_at", ""))
                age_days = max(0, (now - ex_dt).days)
            except Exception:
                age_days = 0
            loop["age_days"] = age_days
            out.append(loop)
        if len(out) >= max_items * 3:
            break
    return out


def _pendientes_recent_contradictions(
    log_path: Path, now: datetime, days: int = 14, max_items: int = 5,
) -> list[dict]:
    """Index-time contradictions from the last `days`. Newest-first.

    Reads from rag_contradictions (SQL-only since T10). The `log_path` arg
    is retained for call-site compatibility but no longer consulted. On SQL
    error: empty list (after retry budget — `_sql_read_with_retry` swallows
    transient `database is locked` + `disk I/O error` for up to 5 attempts
    with jittered backoff, so a brief writer holding the WAL doesn't blank
    the home page's contradictions panel).
    """
    from rag import _ragvec_state_conn, _sql_query_window, _sql_read_with_retry
    cutoff = now - timedelta(days=days)
    cutoff_iso = cutoff.isoformat(timespec="seconds")
    out: list[dict] = []

    def _do_read():
        with _ragvec_state_conn() as conn:
            return _sql_query_window(conn, "rag_contradictions", cutoff_iso)

    rows = _sql_read_with_retry(
        _do_read,
        "contradictions_sql_read_failed",
        default=None,
    )
    if rows is None:
        return []

    # Newest-first: rows come ordered by ts ASC; reverse.
    for r in reversed(rows):
        try:
            contradicts = json.loads(r["contradicts_json"]) \
                if r["contradicts_json"] else []
        except Exception:
            contradicts = []
        if not contradicts:
            continue
        out.append({
            "subject_path": r["subject_path"] or "",
            "targets": contradicts[:3],
            "ts": r["ts"],
        })
        if len(out) >= max_items:
            break
    return out


def _pendientes_low_conf_queries(
    log_path: Path, now: datetime, days: int = 7, max_items: int = 5,
) -> list[dict]:
    """Queries below `CONFIDENCE_RERANK_MIN` in last `days`. Dedup by q
    (keep lowest score). Sorted ascending.
    """
    from rag import CONFIDENCE_RERANK_MIN
    if not log_path.is_file():
        return []
    cutoff = now - timedelta(days=days)
    best: dict[str, float] = {}
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("cmd") != "query":
            continue
        try:
            ts = datetime.fromisoformat(e.get("ts", ""))
        except Exception:
            continue
        if ts < cutoff:
            continue
        score = e.get("top_score")
        if not isinstance(score, (int, float)) or score > CONFIDENCE_RERANK_MIN:
            continue
        q = (e.get("q") or "").strip()
        if not q:
            continue
        if q not in best or score < best[q]:
            best[q] = float(score)
    ranked = sorted(best.items(), key=lambda x: x[1])[:max_items]
    return [{"q": q, "top_score": s} for q, s in ranked]


def _pendientes_urgent(ev: dict, now: datetime) -> list[str]:
    """Surface items that jump off the screen — miss them = blown SLA."""
    from rag import _format_gmail_from
    urgent: list[str] = []
    for r in ev.get("reminders") or []:
        if r.get("bucket") == "overdue":
            lst = f" [{r['list']}]" if r.get("list") else ""
            urgent.append(f"⏰ overdue · {r.get('name','')}{lst}")
    for m in (ev.get("gmail") or {}).get("awaiting_reply") or []:
        if (m.get("days_old") or 0) >= 7:
            who = _format_gmail_from(m.get("from") or "")
            urgent.append(
                f"⏳ {m.get('days_old')}d · {m.get('subject','')} — {who}"
            )
    for m in ev.get("mail_unread") or []:
        if m.get("is_vip"):
            urgent.append(f"📧 VIP · {m.get('subject','')} — {m.get('sender','')}")
    return urgent


@cli.command("pendientes")
@click.option("--days", default=14, show_default=True,
              help="Ventana para loops/queries/contradicciones")
@click.option("--plain", is_flag=True, help="Salida plana (para bots)")
def pendientes(days: int, plain: bool):
    """Dashboard unificado: todo lo que espera tu atención ahora mismo.

    Gmail awaiting-reply + WhatsApp chats activos + Apple Reminders
    (overdue/today/undated) + loops del vault + contradicciones recientes +
    queries sin respuesta + agenda del día. Mid-day check-in: morning brief
    cubre el arranque, pendientes te reubica.

    Sin LLM, <3s en warm caches. Secciones vacías se omiten.
    """
    from rag import _round_timing_ms, get_db, log_query_event
    col = get_db()
    now = datetime.now()
    t0 = time.perf_counter()
    ev = _pendientes_collect(col, now, days=days)
    urgent = _pendientes_urgent(ev, now)

    log_query_event({
        "cmd": "pendientes", "days": days,
        "counts": {
            "urgent": len(urgent),
            "mail_unread": len(ev.get("mail_unread") or []),
            "gmail_awaiting": len((ev.get("gmail") or {}).get("awaiting_reply") or []),
            "whatsapp": len(ev.get("whatsapp") or []),
            "reminders": len(ev.get("reminders") or []),
            "loops_stale": len(ev.get("loops_stale") or []),
            "loops_activo": len(ev.get("loops_activo") or []),
            "contradictions": len(ev.get("contradictions") or []),
            "low_conf": len(ev.get("low_conf") or []),
            "calendar": len(ev.get("calendar") or []),
        },
        "timing": _round_timing_ms({"total_ms": (time.perf_counter() - t0) * 1000}),
    })

    if plain:
        _pendientes_render_plain(ev, urgent, now)
        return
    _pendientes_render_rich(ev, urgent, now)


def _pendientes_render_plain(ev: dict, urgent: list[str], now: datetime) -> None:
    from rag import _format_gmail_from
    out: list[str] = [f"📋 Pendientes · {now.strftime('%Y-%m-%d %H:%M')}"]
    if urgent:
        out.append("")
        out.append(f"🚨 Urgente ({len(urgent)})")
        out.extend(f"  {line}" for line in urgent)
    gm = ev.get("gmail") or {}
    awaiting = gm.get("awaiting_reply") or []
    if awaiting or gm.get("unread_count"):
        out.append("")
        out.append(f"📧 Gmail ({len(awaiting)} awaiting · {gm.get('unread_count',0)} unread)")
        for m in awaiting[:5]:
            who = _format_gmail_from(m.get("from") or "")
            out.append(f"  ⏳ {m.get('days_old',0)}d · {m.get('subject','')} — {who}")
    mail = ev.get("mail_unread") or []
    if mail:
        out.append("")
        out.append(f"📬 Apple Mail ({len(mail)} no leídos 36h)")
        for m in mail[:5]:
            prefix = "VIP · " if m.get("is_vip") else ""
            out.append(f"  · {prefix}{m.get('subject','')} — {m.get('sender','')}")
    wa = ev.get("whatsapp") or []
    if wa:
        out.append("")
        total = sum(int(w.get("count", 0)) for w in wa)
        out.append(f"💬 WhatsApp ({len(wa)} chats · {total} msgs 24h)")
        for w in wa[:5]:
            snip = (w.get("last_snippet") or "").strip()[:60]
            snip_part = f" — \"{snip}\"" if snip else ""
            out.append(f"  · {w.get('name','(?)')} ({w.get('count',0)}){snip_part}")
    rem = ev.get("reminders") or []
    if rem:
        out.append("")
        out.append(f"📌 Reminders ({len(rem)})")
        for bucket in ("overdue", "today", "upcoming", "undated"):
            for r in [x for x in rem if (x.get("bucket") or "") == bucket][:5]:
                lst = f" [{r['list']}]" if r.get("list") else ""
                due = f" · {r['due']}" if r.get("due") else ""
                tag = f"({bucket}) " if bucket != "undated" else "📌 "
                out.append(f"  {tag}{r.get('name','')}{due}{lst}")
    stale = ev.get("loops_stale") or []
    activo = ev.get("loops_activo") or []
    if stale or activo:
        out.append("")
        out.append(f"📁 Vault loops ({len(stale)} stale · {len(activo)} activo)")
        for it in stale[:3]:
            out.append(f"  🕸 {it.get('age_days',0)}d · {it.get('loop_text','')[:80]} [[{Path(it['source_note']).stem}]]")
        for it in activo[:3]:
            out.append(f"  🔥 {it.get('age_days',0)}d · {it.get('loop_text','')[:80]} [[{Path(it['source_note']).stem}]]")
    contrad = ev.get("contradictions") or []
    if contrad:
        out.append("")
        out.append(f"⚠ Contradicciones recientes ({len(contrad)})")
        for c in contrad[:3]:
            tgt = ", ".join(t.get("path","") for t in c.get("targets", [])[:2])
            out.append(f"  · {c.get('subject_path','')} ↔ {tgt}")
    low = ev.get("low_conf") or []
    if low:
        out.append("")
        out.append(f"❓ Queries sin respuesta ({len(low)})")
        for q in low[:3]:
            out.append(f"  · \"{q['q']}\" (score {q['top_score']:+.2f})")
    cal = ev.get("calendar") or []
    if cal:
        out.append("")
        out.append(f"📅 Hoy en agenda ({len(cal)})")
        for e in cal[:8]:
            out.append(f"  · {e.get('start','')} {e.get('title','')}")
    weather = ev.get("weather")
    if weather:
        out.append("")
        out.append(f"🌧 {weather.get('summary', 'lluvia')}")
    total_items = (
        len(urgent) + len(awaiting) + len(mail) + len(wa) + len(rem)
        + len(stale) + len(activo) + len(contrad) + len(low) + len(cal)
    )
    if total_items == 0:
        out.append("")
        out.append("✨ Todo limpio — sin pendientes trackeados.")
    click.echo("\n".join(out))


def _pendientes_render_rich(ev: dict, urgent: list[str], now: datetime) -> None:
    from rag import _format_gmail_from
    console.print()
    console.print(Rule(
        f"📋 [bold]Pendientes[/bold] · [dim]{now.strftime('%Y-%m-%d %H:%M')}[/dim]",
        style="bold cyan",
    ))
    if urgent:
        console.print()
        console.print(f"[bold red]🚨 Urgente ({len(urgent)})[/bold red]")
        for line in urgent:
            console.print(f"  [red]{line}[/red]")

    gm = ev.get("gmail") or {}
    awaiting = gm.get("awaiting_reply") or []
    unread_total = int(gm.get("unread_count") or 0)
    if awaiting or unread_total:
        console.print()
        console.print(
            f"[bold]📧 Gmail[/bold] "
            f"[dim]({len(awaiting)} awaiting · {unread_total} unread)[/dim]"
        )
        for m in awaiting[:5]:
            who = _format_gmail_from(m.get("from") or "")
            days_old = m.get("days_old") or 0
            console.print(
                f"  ⏳ [yellow]{days_old}d[/yellow] · "
                f"{m.get('subject','')} [dim]— {who}[/dim]"
            )

    mail = ev.get("mail_unread") or []
    if mail:
        console.print()
        console.print(f"[bold]📬 Apple Mail[/bold] [dim]({len(mail)} no leídos 36h)[/dim]")
        for m in mail[:5]:
            prefix = "[bold magenta]VIP[/bold magenta] · " if m.get("is_vip") else ""
            console.print(f"  · {prefix}{m.get('subject','')} [dim]— {m.get('sender','')}[/dim]")

    wa = ev.get("whatsapp") or []
    if wa:
        total = sum(int(w.get("count", 0)) for w in wa)
        console.print()
        console.print(f"[bold]💬 WhatsApp[/bold] [dim]({len(wa)} chats · {total} msgs 24h)[/dim]")
        for w in wa[:5]:
            snip = (w.get("last_snippet") or "").strip()[:60]
            snip_part = f' [dim]— "{snip}"[/dim]' if snip else ""
            console.print(
                f"  · {w.get('name','(?)')} "
                f"[yellow]({w.get('count',0)})[/yellow]{snip_part}"
            )

    rem = ev.get("reminders") or []
    if rem:
        console.print()
        console.print(f"[bold]📌 Reminders[/bold] [dim]({len(rem)})[/dim]")
        bucket_style = {"overdue": "red", "today": "yellow", "upcoming": "cyan", "undated": "white"}
        for bucket in ("overdue", "today", "upcoming", "undated"):
            for r in [x for x in rem if (x.get("bucket") or "") == bucket][:5]:
                lst = f" [dim][{r['list']}][/dim]" if r.get("list") else ""
                due = f" [dim]· {r['due']}[/dim]" if r.get("due") else ""
                if bucket == "undated":
                    console.print(f"  📌 {r.get('name','')}{lst}")
                else:
                    style = bucket_style.get(bucket, "white")
                    console.print(f"  [{style}]({bucket})[/{style}] {r.get('name','')}{due}{lst}")

    stale = ev.get("loops_stale") or []
    activo = ev.get("loops_activo") or []
    if stale or activo:
        console.print()
        console.print(f"[bold]📁 Vault loops[/bold] [dim]({len(stale)} stale · {len(activo)} activo)[/dim]")
        for it in stale[:3]:
            stem = Path(it["source_note"]).stem
            console.print(f"  [red]🕸 {it.get('age_days',0)}d[/red] · {it.get('loop_text','')[:80]} [dim]← [[{stem}]][/dim]")
        for it in activo[:3]:
            stem = Path(it["source_note"]).stem
            console.print(f"  [yellow]🔥 {it.get('age_days',0)}d[/yellow] · {it.get('loop_text','')[:80]} [dim]← [[{stem}]][/dim]")

    contrad = ev.get("contradictions") or []
    if contrad:
        console.print()
        console.print(f"[bold]⚠ Contradicciones recientes[/bold] [dim]({len(contrad)})[/dim]")
        for c in contrad[:3]:
            tgt = ", ".join(t.get("path", "") for t in c.get("targets", [])[:2])
            console.print(f"  · {c.get('subject_path','')} [dim]↔ {tgt}[/dim]")

    low = ev.get("low_conf") or []
    if low:
        console.print()
        console.print(f"[bold]❓ Queries sin respuesta[/bold] [dim]({len(low)})[/dim]")
        for q in low[:3]:
            console.print(f"  · \"{q['q']}\" [dim](score {q['top_score']:+.2f})[/dim]")

    cal = ev.get("calendar") or []
    if cal:
        console.print()
        console.print(f"[bold]📅 Hoy en agenda[/bold] [dim]({len(cal)})[/dim]")
        for e in cal[:8]:
            console.print(f"  · [cyan]{e.get('start','')}[/cyan] {e.get('title','')}")

    weather = ev.get("weather")
    if weather:
        console.print()
        console.print(f"[dim]🌧 {weather.get('summary', 'lluvia')}[/dim]")

    total_items = (
        len(urgent) + len(awaiting) + len(mail) + len(wa) + len(rem)
        + len(stale) + len(activo) + len(contrad) + len(low) + len(cal)
    )
    if total_items == 0:
        console.print()
        console.print("[bold green]✨ Todo limpio — sin pendientes trackeados.[/bold green]")
