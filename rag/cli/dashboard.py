"""`rag dashboard` — terminal analytics dashboard.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 200).

Terminal-based analytics over queries.jsonl. Shows query patterns,
retrieval quality distribution (score percentiles + histogram),
latency breakdown (retrieve / generate / total), activity heatmap by
hour, hot topics, feedback ratio (positive/negative/corrective), and
index stats (chunks, tags, folders, top PageRank).

## API

- `_dashboard_data(days)` → dict con buckets agregados.
- `dashboard_cmd` Click command — registrado al final de
  `rag/__init__.py` via `cli.add_command(dashboard_cmd, name="dashboard")`.

## Lazy imports

`_scan_queries_log`, `_load_corpus`, `console`, `FEEDBACK_PATH`,
`get_db`, `get_pagerank` viven en `rag/__init__.py`. Lazy adentro
para evitar circular import.

## Re-export

`rag/__init__.py` re-exporta `_dashboard_data` para tests/scripts
que llamen `rag._dashboard_data(...)`.
"""

from __future__ import annotations

from datetime import datetime

import click
from rich.rule import Rule
from rich.table import Table

__all__ = ["_dashboard_data", "dashboard_cmd"]


def _dashboard_data(days: int = 30) -> dict:
    """Parse queries.jsonl into dashboard metrics."""
    from rag import _scan_queries_log  # noqa: PLC0415

    entries = _scan_queries_log(days=days)
    if not entries:
        return {"n": 0}

    scores: list[float] = []
    t_retrieves: list[float] = []
    t_gens: list[float] = []
    gated = 0
    answered = 0
    topics: dict[str, int] = {}
    hours: dict[int, int] = {}
    cmds: dict[str, int] = {}

    for e in entries:
        ts_raw = e.get("ts") or e.get("timestamp")
        if ts_raw:
            try:
                hour = datetime.fromisoformat(ts_raw).hour
                hours[hour] = hours.get(hour, 0) + 1
            except Exception:
                pass

        cmd = e.get("cmd", "?")
        cmds[cmd] = cmds.get(cmd, 0) + 1

        top = e.get("top_score")
        if isinstance(top, (int, float)):
            scores.append(float(top))

        tr = e.get("t_retrieve")
        if isinstance(tr, (int, float)):
            t_retrieves.append(float(tr))

        tg = e.get("t_gen")
        if isinstance(tg, (int, float)) and tg > 0:
            t_gens.append(float(tg))

        if e.get("gated_low_confidence"):
            gated += 1
        if e.get("answered"):
            answered += 1

        q = e.get("q", "")
        if q:
            words = q.lower().split()[:3]
            topic = " ".join(words)
            topics[topic] = topics.get(topic, 0) + 1

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    return {
        "n": len(entries),
        "days": days,
        "scores": scores,
        "t_retrieves": t_retrieves,
        "t_gens": t_gens,
        "gated": gated,
        "answered": answered,
        "topics": topics,
        "hours": hours,
        "cmds": cmds,
        "pct": pct,
    }


@click.command("dashboard")
@click.option("--days", default=30, show_default=True, help="Ventana temporal en días")
def dashboard_cmd(days: int):
    """Analytics dashboard — métricas del pipeline sobre queries.jsonl."""
    from rag import (  # noqa: PLC0415
        FEEDBACK_PATH,
        _load_corpus,
        console,
        get_db,
        get_pagerank,
    )

    data = _dashboard_data(days)
    if data["n"] == 0:
        console.print("[yellow]Sin queries en el período.[/yellow]")
        return

    console.print()
    console.print(Rule(title="[bold cyan]Dashboard[/bold cyan]", style="cyan"))
    console.print(f"  [dim]{data['n']} queries · últimos {data['days']} días[/dim]")
    console.print()

    t = Table(title="Comandos", show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Comando", style="cyan")
    t.add_column("Count", justify="right")
    for cmd, count in sorted(data["cmds"].items(), key=lambda x: -x[1])[:8]:
        t.add_row(cmd, str(count))
    console.print(t)
    console.print()

    scores = data["scores"]
    pct = data["pct"]
    if scores:
        console.print(Rule(title="[dim]Retrieval Quality[/dim]", style="dim", characters="╌"))
        gate_rate = data["gated"] / data["n"] * 100 if data["n"] else 0
        answer_rate = data["answered"] / data["n"] * 100 if data["n"] else 0
        console.print(f"  Score p50: [bold]{pct(scores, 50):.3f}[/bold] · "
                      f"p95: [bold]{pct(scores, 95):.3f}[/bold] · "
                      f"min: {min(scores):.3f} · max: {max(scores):.3f}")
        console.print(f"  Gate rate: [{'red' if gate_rate > 30 else 'green'}]"
                      f"{gate_rate:.1f}%[/] · Answer rate: [green]{answer_rate:.1f}%[/]")
        buckets = [0] * 10
        for s in scores:
            clamped = max(0.0, min(s, 1.0))
            idx = min(int(clamped * 10), 9)
            buckets[idx] += 1
        max_b = max(buckets) or 1
        console.print("  Score distribution:")
        for i, b in enumerate(buckets):
            bar = "█" * int(b / max_b * 20) if b else ""
            label = f"  {i/10:.1f}-{(i+1)/10:.1f}"
            console.print(f"  {label} [{('green' if i >= 5 else 'yellow' if i >= 2 else 'red')}]{bar}[/] {b}")
        console.print()

    t_r = data["t_retrieves"]
    t_g = data["t_gens"]
    if t_r:
        console.print(Rule(title="[dim]Latency[/dim]", style="dim", characters="╌"))
        console.print(f"  Retrieve: p50 [bold]{pct(t_r, 50):.2f}s[/bold] · "
                      f"p95 [bold]{pct(t_r, 95):.2f}s[/bold]")
        if t_g:
            console.print(f"  Generate: p50 [bold]{pct(t_g, 50):.2f}s[/bold] · "
                          f"p95 [bold]{pct(t_g, 95):.2f}s[/bold]")
            total = [r + g for r, g in zip(t_r[:len(t_g)], t_g)]
            if total:
                console.print(f"  Total:    p50 [bold]{pct(total, 50):.2f}s[/bold] · "
                              f"p95 [bold]{pct(total, 95):.2f}s[/bold]")
        console.print()

    hours = data["hours"]
    if hours:
        console.print(Rule(title="[dim]Activity by Hour[/dim]", style="dim", characters="╌"))
        max_h = max(hours.values()) or 1
        for h in range(24):
            count = hours.get(h, 0)
            bar = "█" * int(count / max_h * 30) if count else ""
            console.print(f"  {h:02d}:00 [cyan]{bar}[/] {count if count else ''}")
        console.print()

    topics = data["topics"]
    if topics:
        console.print(Rule(title="[dim]Hot Topics[/dim]", style="dim", characters="╌"))
        for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:10]:
            if count >= 2:
                console.print(f"  [bold]{count}×[/bold] {topic}")
        console.print()

    try:
        if FEEDBACK_PATH.is_file():
            fb_lines = FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
            pos = sum(1 for l in fb_lines if '"rating": 1' in l or '"rating":1' in l)
            neg = sum(1 for l in fb_lines if '"rating": -1' in l or '"rating":-1' in l)
            corrective = sum(1 for l in fb_lines if '"corrective_path"' in l)
            console.print(Rule(title="[dim]Feedback[/dim]", style="dim", characters="╌"))
            total_fb = pos + neg
            ratio = pos / total_fb * 100 if total_fb else 0
            console.print(f"  👍 {pos} · 👎 {neg} · ratio: [{'green' if ratio > 70 else 'yellow'}]{ratio:.0f}%[/]")
            if corrective:
                console.print(f"  🎯 {corrective} corrective paths")
            console.print()
    except Exception:
        pass

    try:
        col = get_db()
        n_chunks = col.count()
        console.print(Rule(title="[dim]Index[/dim]", style="dim", characters="╌"))
        console.print(f"  {n_chunks} chunks indexed")
        corpus = _load_corpus(col)
        console.print(f"  {len(corpus['tags'])} tags · "
                      f"{len(corpus['folders'])} folders · "
                      f"{len(corpus['title_to_paths'])} notes")
        pr = get_pagerank(col)
        if pr:
            top_pr = sorted(pr.items(), key=lambda x: -x[1])[:5]
            console.print("  Top PageRank:")
            for path, score in top_pr:
                console.print(f"    {score:.4f} {path}")
    except Exception:
        pass
    console.print()
