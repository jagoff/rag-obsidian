"""Archive subsystem — extracted from rag/__init__.py 2026-05-04.

Cyclical cleanup: mueve las notas detectadas por `find_dead_notes` a
`04-Archive/` preservando la jerarquía PARA original. Todas las funciones
públicas siguen disponibles en el namespace `rag.*` via el re-export en
`__init__.py`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import rag as _rag

__all__ = [
    "_ARCHIVE_ROOT",
    "ARCHIVE_GATE_DEFAULT",
    "ARCHIVE_LOG_PATH",
    "_ARCHIVE_OPT_OUT_TYPES",
    "CITATION_PROMOTE_THRESHOLD",
    "CITATION_WINDOW_DAYS",
    "_archive_target_path",
    "_archive_resolve_collision",
    "_is_archive_opt_out",
    "_archive_stamp_frontmatter",
    "_open_archive_batch",
    "_append_archive_batch",
    "_log_archive_event",
    "_archive_move_one",
    "_load_conversation_citations",
    "archive_dead_notes",
    "_render_archive_result",
    "_write_archive_report",
    "_push_archive_notification",
]

# ── PROMOTE-ON-CITE GUARDRAIL (added 2026-05-04) ────────────────────────────
# Una nota citada como source en N+ conversations recientes está sosteniendo
# conocimiento operativo activo (procedimientos, URLs, runbooks). Si el
# detector de dead notes la propone para archivar, la frenamos: el user la
# está usando en chats aunque no la haya editado. Sin este check, las notas
# útiles de `00-Inbox/` se evaporaban por inactividad de edición y al día
# siguiente RAG decía "no tengo nada" sobre el mismo tema (caso AWS
# Organizations 2026-05-04).
CITATION_PROMOTE_THRESHOLD = int(os.getenv("RAG_ARCHIVE_CITE_THRESHOLD", "2"))
CITATION_WINDOW_DAYS = int(os.getenv("RAG_ARCHIVE_CITE_WINDOW_DAYS", "30"))
_CONVERSATIONS_REL = "04-Archive/99-obsidian-system/99-AI/conversations"

# ── ARCHIVER (rag archive) ───────────────────────────────────────────────────
# Cyclical cleanup: mueve las notas detectadas por `find_dead_notes` a
# `04-Archive/` preservando la jerarquía PARA original (01-Projects/X/nota.md
# → 04-Archive/01-Projects/X/nota.md). Frontmatter stamp archived_at /
# archived_from / archived_reason para que el move sea reversible. Gate de
# confirmación si hay más de N candidatos (evita masacres silenciosas cuando
# la ventana o la edad afloja). Reusa el detector de `rag dead` — mismos
# criterios AND (0 outlinks + 0 backlinks + no retrieved + age > N + fuera
# de Inbox/Archive/Reviews). Opt-out por nota con `archive: never` o
# `type: moc|index|permanent` en frontmatter.

_ARCHIVE_ROOT = "04-Archive"
ARCHIVE_GATE_DEFAULT = 20
ARCHIVE_LOG_PATH = Path.home() / ".local/share/obsidian-rag/archive.jsonl"
_ARCHIVE_OPT_OUT_TYPES = ("moc", "index", "permanent")


def _archive_target_path(src_rel: str) -> str:
    """Mirror source path under 04-Archive/. Preserves the PARA folder below
    the root so `01-Projects/app-X/nota.md` lands at
    `04-Archive/01-Projects/app-X/nota.md`. Already-archived paths are
    returned unchanged (detector should skip them anyway).
    """
    if src_rel.startswith(_ARCHIVE_ROOT + "/") or src_rel == _ARCHIVE_ROOT:
        return src_rel
    return f"{_ARCHIVE_ROOT}/{src_rel}"


def _archive_resolve_collision(vault: Path, dst_rel: str) -> str:
    """If destination exists, append `-archived-YYYY-MM` to the stem. Walks
    a counter if that variant also collides (rare).
    """
    dst = vault / dst_rel
    if not dst.exists():
        return dst_rel
    parent_rel = str(Path(dst_rel).parent)
    stem = Path(dst_rel).stem
    suffix = Path(dst_rel).suffix
    tag = datetime.now().strftime("%Y-%m")
    for i in range(1, 50):
        extra = f"-archived-{tag}" if i == 1 else f"-archived-{tag}-{i}"
        candidate = f"{parent_rel}/{stem}{extra}{suffix}" if parent_rel != "." else f"{stem}{extra}{suffix}"
        if not (vault / candidate).exists():
            return candidate
    return dst_rel  # give up; caller will fail explicitly on the move


def _is_archive_opt_out(raw: str) -> bool:
    """True if the note frontmatter opts out of archiving via `archive: never`
    or `type: moc|index|permanent`. Permanent-knowledge notes (MOCs, indexes,
    evergreen references) shouldn't die just for lacking edits.

    `type:` may be a scalar (`type: moc`) or a list (`type: [moc, reference]`).
    Both forms are honored — list form was silently unmatched before because
    stringifying the list yielded `"['moc', 'reference']"`.
    """
    fm = _rag.parse_frontmatter(raw)
    if str(fm.get("archive", "")).strip().lower() == "never":
        return True
    raw_type = fm.get("type")
    if isinstance(raw_type, list):
        types = [str(x).strip().lower() for x in raw_type]
    else:
        types = [str(raw_type or "").strip().lower()]
    return any(t in _ARCHIVE_OPT_OUT_TYPES for t in types)


def _archive_stamp_frontmatter(raw: str, orig_rel: str,
                                reason: str = "dead") -> str:
    """Inject `archived_at / archived_from / archived_reason` into the
    frontmatter. Creates a fresh YAML block if the note has none. Idempotent:
    re-stamping overwrites prior archived_* fields.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    stamps = {
        "archived_at": today,
        "archived_from": orig_rel,
        "archived_reason": reason,
    }
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end >= 0:
            fm_text = raw[4:end]
            rest = raw[end + 5:]
            kept_lines: list[str] = []
            for line in fm_text.splitlines():
                key = line.split(":", 1)[0].strip()
                if key in stamps:
                    continue  # drop prior stamp; we're rewriting
                kept_lines.append(line)
            extra = "\n".join(f"{k}: {v}" for k, v in stamps.items())
            body = "\n".join(kept_lines).rstrip()
            block = f"{body}\n{extra}" if body else extra
            return f"---\n{block}\n---\n{rest}"
    # No frontmatter (or malformed): prepend a fresh one.
    fresh = "\n".join(f"{k}: {v}" for k, v in stamps.items())
    return f"---\n{fresh}\n---\n\n{raw}"


def _open_archive_batch() -> Path:
    """One-batch-per-run audit log for rollback. Lives alongside filing
    batches but with an `archive-` prefix so `rag file --undo` doesn't pick
    it up accidentally.
    """
    _rag.FILING_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _rag.FILING_BATCHES_DIR / f"archive-{ts}.jsonl"
    path.touch()
    return path


def _append_archive_batch(batch_path: Path, entry: dict) -> None:
    with batch_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _log_archive_event(event: dict) -> None:
    e = {"ts": datetime.now().isoformat(timespec="seconds"), **event}

    def _do() -> None:
        with _rag._ragvec_state_conn() as conn:
            _rag._sql_append_event(conn, "rag_archive_log",
                                   _rag._map_archive_row(e))
    if _rag._log_archive_event_background_default():
        _rag._enqueue_background_sql(_do, "archive_sql_write_failed")
    else:
        _rag._sql_write_with_retry(_do, "archive_sql_write_failed")


def _archive_move_one(
    col: "object", src_rel: str, dst_rel: str,
) -> dict:
    """Move a single note. Stamps frontmatter in place FIRST (so the moved
    file carries the stamp), then `shutil.move`, then two reindex calls:
    old path (now gone → sqlite-vec deletes its chunks) + new path (fresh chunks
    under the new `file=` metadata key).
    """
    src = (_rag.VAULT_PATH / src_rel).resolve()
    src.relative_to(_rag.VAULT_PATH.resolve())
    if not src.is_file():
        raise FileNotFoundError(src_rel)
    dst = (_rag.VAULT_PATH / dst_rel).resolve()
    dst.relative_to(_rag.VAULT_PATH.resolve())
    if dst.exists():
        raise FileExistsError(dst_rel)
    raw = src.read_text(encoding="utf-8", errors="ignore")
    stamped = _archive_stamp_frontmatter(raw, src_rel)
    src.write_text(stamped, encoding="utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    # Clean old path from sqlite-vec + add new path. Both best-effort — the next
    # full `rag index` would converge regardless.
    try:
        _rag._index_single_file(col, src, skip_contradict=True)
    except Exception:
        pass
    try:
        _rag._index_single_file(col, dst, skip_contradict=True)
    except Exception:
        pass
    return {
        "src": src_rel,
        "dst": dst_rel,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def _load_conversation_citations(
    vault: Path,
    window_days: int = CITATION_WINDOW_DAYS,
) -> dict[str, int]:
    """Escanea ``99-AI/conversations/*.md`` y devuelve un counter
    ``{vault_relative_path: citation_count}`` con cuántas conversations
    recientes citan cada nota como source.

    Window basado en ``created`` del frontmatter; si falta, en mtime.
    Solo se cuentan paths que parecen notas markdown (descarta pseudo-URIs
    como ``whatsapp://...`` que no son archivables).

    Cache no persistente — se invoca una vez por corrida de
    ``archive_dead_notes`` y muere con el proceso. Para 121 conversations
    típicas, la pasada cuesta < 200 ms.
    """
    counter: dict[str, int] = {}
    conv_dir = vault / _CONVERSATIONS_REL
    if not conv_dir.is_dir():
        return counter
    cutoff = datetime.now() - timedelta(days=window_days)
    fm_re = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
    for f in conv_dir.rglob("*.md"):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = fm_re.match(txt)
        if not m:
            continue
        # Window check via frontmatter `created`; fallback al mtime del file.
        created_str = ""
        for line in m.group(1).splitlines():
            if line.startswith("created:"):
                created_str = line.split(":", 1)[1].strip()
                break
        try:
            if created_str:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created.tzinfo is not None:
                    created = created.replace(tzinfo=None)
            else:
                created = datetime.fromtimestamp(f.stat().st_mtime)
        except Exception:
            created = datetime.fromtimestamp(f.stat().st_mtime)
        if created < cutoff:
            continue
        # Parse `sources:` block — lista YAML simple, una entrada por línea.
        in_sources = False
        for line in m.group(1).splitlines():
            stripped = line.rstrip()
            if stripped.startswith("sources:"):
                in_sources = True
                continue
            if in_sources:
                if stripped.startswith("  - ") or stripped.startswith("- "):
                    src = stripped.split("-", 1)[1].strip().strip('"').strip("'")
                    if src and not src.startswith(("whatsapp://", "http://", "https://")):
                        counter[src] = counter.get(src, 0) + 1
                elif stripped and not stripped.startswith(" "):
                    in_sources = False
    return counter


def archive_dead_notes(
    col: "object",
    vault: Path,
    candidates: list[dict],
    apply: bool,
    force: bool,
    gate: int = ARCHIVE_GATE_DEFAULT,
) -> dict:
    """Plan (and optionally execute) the archive moves. Pure function — takes
    precomputed candidates from `find_dead_notes`. Returns:
        {"plan": [...], "applied": [...], "skipped": [...], "gated": bool,
         "batch_path": str | None}
    """
    plan: list[dict] = []
    skipped: list[dict] = []
    seen_dst: set[str] = set()
    # Promote-on-cite: pre-cargo el counter una sola vez por corrida. Si la
    # carpeta de conversations no existe (vault recién inicializado, tests),
    # el dict queda vacío y el check abajo es no-op.
    citations = _load_conversation_citations(vault)
    for c in candidates:
        src_rel = c["path"]
        src_full = vault / src_rel
        if not src_full.is_file():
            skipped.append({"path": src_rel, "reason": "missing"})
            continue
        try:
            raw = src_full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped.append({"path": src_rel, "reason": "unreadable"})
            continue
        if _is_archive_opt_out(raw):
            skipped.append({"path": src_rel, "reason": "opt-out"})
            continue
        cite_count = citations.get(src_rel, 0)
        if cite_count >= CITATION_PROMOTE_THRESHOLD:
            # Knowledge being actively cited in chat → no archive. The user
            # is using this note even if it hasn't been edited recently.
            skipped.append({
                "path": src_rel,
                "reason": f"cited-{cite_count}x-in-{CITATION_WINDOW_DAYS}d",
            })
            continue
        dst_rel = _archive_target_path(src_rel)
        if dst_rel == src_rel:
            skipped.append({"path": src_rel, "reason": "already-archived"})
            continue
        dst_rel = _archive_resolve_collision(vault, dst_rel)
        # Collision across this same batch (two src stems mapping to the same
        # dst after suffixing) — add a second suffix to the later one.
        if dst_rel in seen_dst:
            stem = Path(dst_rel).stem
            suffix = Path(dst_rel).suffix
            parent_rel = str(Path(dst_rel).parent)
            for i in range(2, 50):
                tag = datetime.now().strftime("%Y-%m")
                candidate = (
                    f"{parent_rel}/{stem}-{i}{suffix}" if parent_rel != "."
                    else f"{stem}-{i}{suffix}"
                )
                if candidate not in seen_dst and not (vault / candidate).exists():
                    dst_rel = candidate
                    break
        seen_dst.add(dst_rel)
        plan.append({
            "src": src_rel, "dst": dst_rel,
            "age_days": c.get("age_days", 0),
        })

    gated = apply and bool(plan) and not force and len(plan) > gate
    applied: list[dict] = []
    batch_path: Path | None = None
    if apply and not gated and plan:
        batch_path = _open_archive_batch()
        for entry in plan:
            try:
                result = _archive_move_one(col, entry["src"], entry["dst"])
                result["age_days"] = entry["age_days"]
                _append_archive_batch(batch_path, result)
                applied.append(result)
            except Exception as e:
                skipped.append({"path": entry["src"], "reason": str(e)})
    return {
        "plan": plan,
        "applied": applied,
        "skipped": skipped,
        "gated": gated,
        "batch_path": str(batch_path) if batch_path else None,
    }


def _render_archive_result(result: dict, apply: bool, plain: bool) -> None:
    import click
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table
    console = Console()

    plan = result["plan"]
    applied = result["applied"]
    skipped = result["skipped"]
    gated = result["gated"]

    if plain:
        for e in (applied if apply and not gated else plan):
            click.echo(f"{e['src']}\t→\t{e['dst']}")
        return

    if not plan and not skipped:
        console.print("[green]Sin candidatos a archivar.[/green]")
        return

    title = (
        f"[bold yellow]📦 {len(plan)} nota(s) a archivar"
        f"{' (dry-run)' if not apply else ''}[/bold yellow]"
    )
    console.print()
    console.print(Rule(title=title, style="yellow"))

    if plan:
        tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        tbl.add_column("edad", style="yellow", justify="right")
        tbl.add_column("origen", style="cyan")
        tbl.add_column("→", style="dim")
        tbl.add_column("destino", style="magenta")
        for e in plan:
            tbl.add_row(
                f"{e['age_days']}d",
                e["src"],
                "→",
                e["dst"],
            )
        console.print(tbl)

    if gated:
        console.print(
            f"\n[yellow]⚠ Gate activo[/yellow]: {len(plan)} > gate. "
            "Corré con [bold]--force[/bold] para aplicar (o achicá la ventana)."
        )
    elif apply:
        console.print(
            f"\n[green]✓[/green] Aplicados: {len(applied)}. "
            f"Batch log: [dim]{result.get('batch_path') or '—'}[/dim]"
        )

    if skipped:
        console.print(
            f"\n[dim]{len(skipped)} skipped — "
            + ", ".join(f"{s['reason']}" for s in skipped[:5])
            + (" …" if len(skipped) > 5 else "")
            + "[/dim]"
        )


def _write_archive_report(result: dict, apply: bool) -> Path | None:
    """Write a human-readable Markdown report to 04-Archive/99-obsidian-system/99-AI/reviews/YYYY-MM-archive.md.
    Appends a dated section if the file already exists (monthly cadence can
    hit the same file twice if triggered manually between cycles).
    """
    plan = result["plan"]
    skipped = result["skipped"]
    if not plan and not skipped:
        return None
    reviews_dir = _rag.VAULT_PATH / "04-Archive/99-obsidian-system/99-AI/reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    ym = datetime.now().strftime("%Y-%m")
    path = reviews_dir / f"{ym}-archive.md"
    header = (
        f"\n\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
        f"{'apply' if apply and not result['gated'] else 'dry-run'}\n\n"
    )
    lines = [header]
    if plan:
        lines.append(f"**{len(plan)} nota(s) planeadas:**\n")
        for e in plan:
            lines.append(f"- `{e['src']}` → `{e['dst']}` ({e['age_days']}d)")
        lines.append("")
    if result["gated"]:
        lines.append("> ⚠ Gate activo — no se aplicó. Revisá y corré `rag archive --apply --force`.\n")
    elif apply:
        lines.append(f"**Aplicadas**: {len(result['applied'])}")
        if result.get("batch_path"):
            lines.append(f"**Batch log**: `{result['batch_path']}`")
        lines.append("")
    if skipped:
        lines.append(f"**{len(skipped)} skipped:**\n")
        for s in skipped[:20]:
            lines.append(f"- `{s['path']}` — {s['reason']}")
        if len(skipped) > 20:
            lines.append(f"- _… {len(skipped) - 20} más_")
    body = "\n".join(lines) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        path.write_text(existing + body, encoding="utf-8")
    else:
        frontmatter = (
            "---\n"
            f"type: archive-report\n"
            f"created: {datetime.now().isoformat(timespec='seconds')}\n"
            "tags:\n- archive\n- review\n"
            "---\n\n"
            f"# Archive report — {ym}\n"
        )
        path.write_text(frontmatter + body, encoding="utf-8")
    return path


def _push_archive_notification(result: dict, apply: bool) -> bool:
    """Fire-and-forget WhatsApp push via the ambient bridge. Silently no-ops
    if ambient isn't configured. Message is compact: counts + top-3 folders.
    """
    from collections import Counter
    cfg = _rag._ambient_config()
    if cfg is None:
        return False
    plan = result["plan"]
    if not plan and not result["skipped"]:
        return False
    folder_counts = Counter(str(Path(e["src"]).parent) for e in plan)
    top = folder_counts.most_common(3)
    verb = "archivadas" if apply and not result["gated"] else "candidatas"
    lines = [f"📦 *Archive* — {len(plan)} {verb}"]
    if result["gated"]:
        lines.append(f"⚠ gate activo (> {ARCHIVE_GATE_DEFAULT}) — revisar y correr `rag archive --apply --force`")
    elif apply:
        lines.append(f"✓ aplicadas: {len(result['applied'])}")
    if top:
        lines.append("\nPor carpeta:")
        for folder, n in top:
            lines.append(f"• `{folder}` — {n}")
    if result["skipped"]:
        lines.append(f"\n_{len(result['skipped'])} skipped_")
    msg = "\n".join(lines)
    sent = _rag._ambient_whatsapp_send(cfg["jid"], msg)
    _rag._ambient_log_event({
        "cmd": "archive_push",
        "n_plan": len(plan),
        "n_applied": len(result["applied"]),
        "gated": result["gated"],
        "whatsapp_sent": sent,
    })
    return sent
