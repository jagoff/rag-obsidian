#!/usr/bin/env python3
"""Harness temporal de eval end-to-end contra el servidor web del RAG.

Dispara las queries definidas en `queries.yaml` contra `POST /api/chat`,
consume el stream SSE, junta la respuesta completa, corre checks
automáticos (latency caps, substrings esperados, no-leak de tool names,
refusal detection) y escribe reports timestamped en `reports/`.

Diseñado para correr sin interacción humana:
  - Exit 0 si todos los checks pasan.
  - Exit 1 si al menos uno falla (pero siempre completa TODO el batch
    antes de devolver el código — así un fail temprano no esconde
    fails posteriores).

Uso:
    .venv/bin/python scratch_eval/run_eval.py
    .venv/bin/python scratch_eval/run_eval.py --category forced_tools
    .venv/bin/python scratch_eval/run_eval.py --base-url http://localhost:8765

ESTE HARNESS ES TEMPORAL. Borrar la carpeta `scratch_eval/` entera
cuando la validación de fixes (2026-04-23) se dé por cerrada.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from rich.console import Console
from rich.table import Table


HERE = Path(__file__).resolve().parent
DEFAULT_QUERIES_PATH = HERE / "queries.yaml"
REPORTS_DIR = HERE / "reports"

# Frases que delatan una refusal del LLM. El match es case-insensitive y
# busca substring — no regex — porque los modelos varían la redacción.
REFUSAL_PHRASES = (
    "no tengo esa información",
    "no tengo esa info",
    "no encuentro información",
    "no encontré información",
    "no puedo responder",
    "no hay información",
    "sin resultados",
    "no hay notas",
    "no figura",
    "no aparece",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QueryRun:
    id: str
    category: str
    question: str
    latency_ms: int = 0
    answer: str = ""
    sources_count: int = 0
    confidence: float | None = None
    metachat: bool = False
    turn_id: str = ""
    error: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.error and all(c.passed for c in self.checks)


def load_queries(path: Path, category_filter: str | None = None) -> tuple[list[dict], list[str]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    leak_patterns = list(doc.get("_leak_patterns_global", []))
    queries = list(doc.get("queries", []))
    if category_filter:
        queries = [q for q in queries if q.get("category") == category_filter]
    return queries, leak_patterns


def _parse_sse_event(block: str) -> tuple[str, dict]:
    """Parse a single SSE event block (lines separated by \\n, block by
    \\n\\n). Returns (event_name, data_dict) or ("", {}) if malformed.
    """
    event = ""
    data = ""
    for line in block.split("\n"):
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data += line.split(":", 1)[1].strip()
    if not event:
        return "", {}
    try:
        parsed = json.loads(data) if data else {}
    except json.JSONDecodeError:
        parsed = {"_raw": data}
    return event, parsed


def run_query(base_url: str, q: dict, timeout_s: float, console: Console) -> QueryRun:
    """POST a /api/chat con streaming SSE. Acumula tokens hasta 'done'.

    Un session_id único por query garantiza que las respuestas no se
    contaminen entre loops (sino el pre-router vería historia y podría
    saltearse tools).
    """
    qid = q["id"]
    run = QueryRun(id=qid, category=q.get("category", "?"), question=q["question"])
    payload = {
        "question": q["question"],
        "session_id": f"scratch-eval-{qid}-{uuid.uuid4().hex[:6]}",
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            with client.stream(
                "POST",
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as r:
                if r.status_code != 200:
                    run.error = f"HTTP {r.status_code}: {r.text[:500]}"
                    run.latency_ms = int((time.perf_counter() - t0) * 1000)
                    return run
                buf = ""
                for chunk in r.iter_text():
                    if not chunk:
                        continue
                    buf += chunk
                    while "\n\n" in buf:
                        block, buf = buf.split("\n\n", 1)
                        ev, data = _parse_sse_event(block)
                        if ev == "token":
                            run.answer += str(data.get("delta") or "")
                        elif ev == "sources":
                            items = data.get("items") or []
                            run.sources_count = len(items)
                            conf = data.get("confidence")
                            run.confidence = float(conf) if conf is not None else None
                            run.metachat = bool(data.get("metachat", False))
                        elif ev == "done":
                            run.turn_id = str(data.get("turn_id") or "")
                            run.metachat = run.metachat or bool(data.get("metachat", False))
                            break
                        elif ev == "error":
                            run.error = str(data.get("message") or data.get("_raw") or "")
                            break
    except httpx.TimeoutException:
        run.error = f"timeout tras {timeout_s}s"
    except httpx.HTTPError as exc:
        run.error = f"http error: {exc}"
    except Exception as exc:  # pragma: no cover — defensive
        run.error = f"{type(exc).__name__}: {exc}"

    run.latency_ms = int((time.perf_counter() - t0) * 1000)
    return run


def _contains_any(haystack: str, needles: list[str]) -> list[str]:
    lo = haystack.lower()
    return [n for n in needles if n.lower() in lo]


def apply_checks(run: QueryRun, q: dict, global_leak_patterns: list[str]) -> None:
    """Ejecuta los checks declarativos del YAML y anota resultados en
    `run.checks`. No raisea — todo failure va al report.
    """
    checks_cfg = q.get("checks") or {}
    answer = run.answer or ""

    # Error de transporte → fail todos los checks implícitamente.
    if run.error:
        run.checks.append(CheckResult("no_transport_error", False, run.error))
        return
    run.checks.append(CheckResult("no_transport_error", True))

    # No-empty answer.
    non_empty = len(answer.strip()) > 0
    run.checks.append(
        CheckResult(
            "non_empty_answer",
            non_empty,
            f"len={len(answer)}" if non_empty else "answer vacío",
        )
    )

    # Latency cap (hard) y target (soft).
    hard = int(checks_cfg.get("max_latency_ms", 0) or 0)
    if hard > 0:
        ok = run.latency_ms <= hard
        run.checks.append(
            CheckResult(
                "latency_within_hard_cap",
                ok,
                f"{run.latency_ms}ms vs cap {hard}ms",
            )
        )

    target = int(checks_cfg.get("target_latency_ms", 0) or 0)
    # Soft target: anotado como check pero NO falla la query si se excede.
    # El report lo muestra con warning. Así vemos degradaciones sin
    # romper el batch.
    if target > 0 and hard > 0 and run.latency_ms > target:
        # No lo marcamos como fail — sólo registro informativo.
        run.checks.append(
            CheckResult(
                "latency_within_soft_target",
                True,  # siempre pass, es soft
                f"WARN {run.latency_ms}ms > target {target}ms",
            )
        )

    # Substrings requeridos.
    req = checks_cfg.get("require_substrings") or []
    if req:
        missing = [s for s in req if s.lower() not in answer.lower()]
        run.checks.append(
            CheckResult(
                "required_substrings_present",
                not missing,
                f"missing={missing}" if missing else "ok",
            )
        )

    # Substrings prohibidos. Permite banear cosas como URLs que el
    # prompt prohibe (`omnifocus.com` en queries person), cross-source
    # contamination ("es tu hermano"), idiomas foráneos ("em março"),
    # o tokens de control que no deberían leakear.
    forbid = checks_cfg.get("forbid_substrings") or []
    if forbid:
        found = [s for s in forbid if s.lower() in answer.lower()]
        run.checks.append(
            CheckResult(
                "no_forbidden_substrings",
                not found,
                f"leaked={found}" if found else "ok",
            )
        )

    # No-leak de tool names (unless opt-out por la query).
    if checks_cfg.get("forbid_leak_patterns", True):
        leaked = _contains_any(answer, global_leak_patterns)
        run.checks.append(
            CheckResult(
                "no_tool_name_leak",
                not leaked,
                f"leaked={leaked}" if leaked else "ok",
            )
        )

    # Refusal detection. Si allow_refusal=false, la presencia de una
    # frase de refusal marca fail. Si true, la frase no afecta.
    allow_ref = bool(checks_cfg.get("allow_refusal", True))
    ref_found = [p for p in REFUSAL_PHRASES if p in answer.lower()]
    if not allow_ref:
        run.checks.append(
            CheckResult(
                "not_refusing",
                not ref_found,
                f"refusal={ref_found}" if ref_found else "ok",
            )
        )
    # Si allow_ref=true, lo registramos como info.
    elif ref_found:
        run.checks.append(
            CheckResult(
                "refusal_noted",
                True,
                f"refusal={ref_found}",
            )
        )


def render_table(runs: list[QueryRun], console: Console) -> None:
    tbl = Table(title="scratch_eval — resumen", header_style="bold cyan")
    tbl.add_column("id", overflow="fold")
    tbl.add_column("cat")
    tbl.add_column("ms", justify="right")
    tbl.add_column("srcs", justify="right")
    tbl.add_column("meta", justify="center")
    tbl.add_column("refs?", justify="center")
    tbl.add_column("status")
    tbl.add_column("failures", overflow="fold")
    for r in runs:
        refusal_noted = any(c.name == "refusal_noted" for c in r.checks)
        failures = [c for c in r.checks if not c.passed]
        if r.error:
            status = "[red]ERR[/red]"
        elif failures:
            status = f"[red]FAIL ({len(failures)})[/red]"
        else:
            status = "[green]OK[/green]"
        fails_fmt = "; ".join(f"{c.name}: {c.detail}" for c in failures)[:120]
        tbl.add_row(
            r.id,
            r.category,
            str(r.latency_ms),
            str(r.sources_count),
            "✓" if r.metachat else "",
            "✓" if refusal_noted else "",
            status,
            fails_fmt,
        )
    console.print(tbl)


def write_reports(runs: list[QueryRun], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"run_{ts}.json"
    md_path = reports_dir / f"run_{ts}.md"

    payload = {
        "timestamp": ts,
        "total": len(runs),
        "passed": sum(1 for r in runs if r.passed),
        "failed": sum(1 for r in runs if not r.passed),
        "runs": [
            {
                **asdict(r),
                "checks": [asdict(c) for c in r.checks],
                "passed": r.passed,
            }
            for r in runs
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        f"# scratch_eval run — {ts}",
        "",
        f"- **Total**: {payload['total']}",
        f"- **Pass**: {payload['passed']}",
        f"- **Fail**: {payload['failed']}",
        "",
        "## Tabla resumen",
        "",
        "| id | cat | ms | srcs | meta | status | failures |",
        "|---|---|---:|---:|:---:|---|---|",
    ]
    for r in runs:
        failures = [c for c in r.checks if not c.passed]
        status = "ERR" if r.error else ("FAIL" if failures else "OK")
        fails_fmt = "; ".join(f"`{c.name}`: {c.detail}" for c in failures)
        meta = "✓" if r.metachat else ""
        lines.append(
            f"| `{r.id}` | {r.category} | {r.latency_ms} | {r.sources_count} | {meta} | {status} | {fails_fmt} |"
        )
    lines.append("")
    lines.append("## Respuestas completas")
    lines.append("")
    for r in runs:
        lines.append(f"### `{r.id}` · {r.category}")
        lines.append("")
        lines.append(f"**Pregunta**: {r.question}")
        lines.append("")
        lines.append(f"- **Latency**: {r.latency_ms}ms")
        lines.append(f"- **Sources count**: {r.sources_count}")
        if r.confidence is not None:
            lines.append(f"- **Confidence**: {r.confidence:.3f}")
        if r.metachat:
            lines.append("- **Metachat**: sí (short-circuit antes del LLM)")
        if r.turn_id:
            lines.append(f"- **turn_id**: `{r.turn_id}`")
        if r.error:
            lines.append("")
            lines.append(f"**Error de transporte**: `{r.error}`")
        lines.append("")
        lines.append("**Respuesta**:")
        lines.append("")
        lines.append("```")
        lines.append(r.answer.strip() or "(vacío)")
        lines.append("```")
        lines.append("")
        if r.checks:
            lines.append("**Checks**:")
            for c in r.checks:
                mark = "✓" if c.passed else "✗"
                lines.append(f"- {mark} `{c.name}` — {c.detail}")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8765")
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES_PATH))
    ap.add_argument("--category", default=None, help="Filtrar por categoría.")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="Timeout por query en segundos (default 120).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Correr solo las primeras N queries (debug).")
    args = ap.parse_args()

    console = Console()

    # Sanity check del servidor antes de quemar 20 queries.
    try:
        probe = httpx.get(f"{args.base_url.rstrip('/')}/", timeout=5.0)
        if probe.status_code >= 500:
            console.print(f"[red]Servidor devolvió {probe.status_code}. Abortando.[/red]")
            return 2
    except httpx.HTTPError as exc:
        console.print(f"[red]No se pudo contactar {args.base_url}: {exc}[/red]")
        console.print("[yellow]Tip: arrancá el server con `launchctl start com.fer.obsidian-rag-web` o `rag web`.[/yellow]")
        return 2

    queries, leak_patterns = load_queries(Path(args.queries), args.category)
    if args.limit > 0:
        queries = queries[: args.limit]

    console.print(f"[cyan]Corriendo {len(queries)} queries contra {args.base_url}...[/cyan]")

    runs: list[QueryRun] = []
    for i, q in enumerate(queries, 1):
        console.print(f"[dim]  [{i}/{len(queries)}] {q['id']}: {q['question'][:60]}...[/dim]")
        run = run_query(args.base_url, q, args.timeout, console)
        apply_checks(run, q, leak_patterns)
        runs.append(run)

    render_table(runs, console)
    json_path, md_path = write_reports(runs, REPORTS_DIR)

    total = len(runs)
    passed = sum(1 for r in runs if r.passed)
    failed = total - passed
    console.print(
        f"\n[bold]{passed}/{total} pass, {failed} fail[/bold] · "
        f"JSON: {json_path} · MD: {md_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
