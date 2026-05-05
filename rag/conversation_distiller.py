"""Conversation distiller — recover knowledge from chat answers whose
original sources have evaporated.

When ``rag chat`` answers a question well by citing notes from
``00-Inbox/`` (or anywhere else in the vault), the answer is archived
under ``99-AI/conversations/<slug>.md`` with the cited paths in the
frontmatter ``sources:`` list. If those source notes are later
archived/deleted, the operational knowledge survives only inside the
conversation log — but conversations are intentionally NOT indexed
(decision 2026-04-20, anti-session-contamination).

This distiller closes that gap. It scans conversations for answers
whose sources are partially or fully missing, and rewrites the bot
turns as standalone runbook notes under
``03-Resources/runbooks/from-conversations/<slug>.md``. The runbooks
ARE indexed (regular ``03-Resources/`` path), so the next time the
user asks the same question, RAG retrieves the distilled answer
directly even if the original notes are gone.

Idempotent: a runbook only gets written once per conversation. Re-runs
skip conversations that already have a distilled counterpart (tracked
via the ``distilled_to:`` frontmatter field stamped on the original
conversation).

Triggered manually via ``rag distill-conversations`` or weekly via the
``com.fer.obsidian-rag-distill`` launchd plist (added 2026-05-04).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


__all__ = [
    "RUNBOOKS_DIR",
    "MIN_CONFIDENCE_DEFAULT",
    "find_orphaned_conversations",
    "distill_conversation",
    "run_distillation",
]

# Donde aterrizan las notas destiladas. Bajo
# `04-Archive/99-obsidian-system/99-AI/runbooks/from-conversations/`
# para cumplir la regla CLAUDE.md global del user: "todo lo que se
# genere por fuera de Obsidian y se genere por medio de AI o con fines
# de AI tiene que ir a 99-AI". `is_excluded()` whitelistea
# `99-AI/runbooks/` (subcarpeta agregada al lado de `99-AI/memory/`,
# `99-AI/external-ingest/`).
RUNBOOKS_DIR = "04-Archive/99-obsidian-system/99-AI/runbooks/from-conversations"

# Floor de confianza para considerar destilable. Las conversations con
# confidence_avg < threshold tienen respuestas dudosas y no las queremos
# canonizar. 0.5 era el promedio de las queries que efectivamente
# acertaban en el corpus 2026-04-30 → 2026-05-04.
MIN_CONFIDENCE_DEFAULT = 0.5

_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_TURN_HEADER_RE = re.compile(r"^## Turn \d+", re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return ``(fm_dict, body)``. Permissive parser — only flat keys and
    simple lists, which is all conversations write."""
    m = _FM_RE.match(raw)
    if not m:
        return {}, raw
    body = raw[m.end():].lstrip("\n")
    fm: dict = {}
    in_list_key: str | None = None
    for line in m.group(1).splitlines():
        if line.startswith(("  - ", "- ")):
            if in_list_key:
                val = line.split("-", 1)[1].strip().strip('"').strip("'")
                fm.setdefault(in_list_key, []).append(val)
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                in_list_key = key
                fm.setdefault(key, [])
            else:
                in_list_key = None
                fm[key] = val.strip('"').strip("'")
    return fm, body


def _split_turns(body: str) -> list[str]:
    """Split body por `## Turn N` headers. Devuelve cada turn con su header."""
    parts = _TURN_HEADER_RE.split(body)
    headers = _TURN_HEADER_RE.findall(body)
    if not headers:
        return [body] if body.strip() else []
    turns: list[str] = []
    # parts[0] es el preámbulo antes del primer header.
    iter_parts = iter(parts[1:])
    for header in headers:
        chunk = next(iter_parts, "")
        turns.append(header + chunk)
    return turns


def _extract_bot_answer(turn_text: str) -> tuple[str, str]:
    """Return ``(user_query, bot_answer)``. User query = primera línea
    blockquote (``> ...``); bot answer = todo lo no-blockquote post-header,
    sin la línea final ``**Sources**:``.
    """
    lines = turn_text.splitlines()
    # Header line + skip blank
    user_q = ""
    bot_lines: list[str] = []
    seen_header = False
    for line in lines:
        if not seen_header and line.startswith("## Turn"):
            seen_header = True
            continue
        if line.startswith("> ") or line.strip() == ">":
            if not user_q:
                user_q = line.lstrip("> ").strip()
            continue
        if line.startswith("**Sources**"):
            break  # corta el bot answer
        bot_lines.append(line)
    answer = "\n".join(bot_lines).strip()
    return user_q, answer


def _slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] or "conversation"


def find_orphaned_conversations(
    vault: Path,
    min_confidence: float = MIN_CONFIDENCE_DEFAULT,
    require_missing_source: bool = True,
) -> list[dict]:
    """List conversations cuyo `confidence_avg ≥ min_confidence` y que tienen
    al menos una source que ya no existe en el filesystem (= conocimiento
    en riesgo de evaporación).

    Si ``require_missing_source=False``, devuelve también conversations con
    sources presentes — útil para destilar proactivamente antes de que se
    pierdan.

    Skip de conversations que ya fueron destiladas (frontmatter
    ``distilled_to: <runbook_path>``).
    """
    conv_dir = vault / "04-Archive/99-obsidian-system/99-AI/conversations"
    if not conv_dir.is_dir():
        return []
    out: list[dict] = []
    for f in conv_dir.rglob("*.md"):
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fm, body = _parse_frontmatter(raw)
        if not fm:
            continue
        if fm.get("distilled_to"):
            continue
        try:
            conf = float(fm.get("confidence_avg", "0") or 0)
        except ValueError:
            conf = 0.0
        if conf < min_confidence:
            continue
        sources = fm.get("sources") or []
        # Solo paths reales (descarta whatsapp:// y http(s)://)
        md_sources = [
            s for s in sources
            if isinstance(s, str)
            and not s.startswith(("whatsapp://", "http://", "https://"))
        ]
        if not md_sources:
            continue
        missing = [s for s in md_sources if not (vault / s).is_file()]
        present = [s for s in md_sources if (vault / s).is_file()]
        if require_missing_source and not missing:
            continue
        out.append({
            "conv_path": str(f.relative_to(vault)),
            "conv_full": f,
            "confidence_avg": conf,
            "sources_missing": missing,
            "sources_present": present,
            "frontmatter": fm,
            "body": body,
        })
    # Ordeno por confianza descendente — destilamos las más fuertes primero.
    out.sort(key=lambda x: x["confidence_avg"], reverse=True)
    return out


def distill_conversation(
    conv: dict,
    vault: Path,
    apply: bool = False,
) -> dict:
    """Destila los bot answers de una conversation a una nota canónica.

    Si la conversation tiene N turns, todos los bot answers se concatenan
    en orden, separados por headers ``## Pregunta original`` con el query
    del user. Cada answer preserva el markdown original (URLs, listas,
    code blocks).

    Returns dict con ``{runbook_path, runbook_full, written, reason?}``.
    Si ``apply=False``, escribe sólo el plan (no toca filesystem).
    """
    body = conv["body"]
    conv["frontmatter"]
    turns = _split_turns(body)
    if not turns:
        return {"runbook_path": None, "written": False, "reason": "no-turns"}

    bot_answers: list[str] = []
    for i, t in enumerate(turns, 1):
        user_q, answer = _extract_bot_answer(t)
        if not answer.strip():
            continue
        if user_q:
            bot_answers.append(f"## Pregunta original {i}\n\n> {user_q}\n\n{answer}")
        else:
            bot_answers.append(f"## Respuesta {i}\n\n{answer}")
    if not bot_answers:
        return {"runbook_path": None, "written": False, "reason": "no-answers"}

    # Slug = primer query del user, o stem del archivo.
    first_user_q = ""
    for t in turns:
        uq, _ = _extract_bot_answer(t)
        if uq:
            first_user_q = uq
            break
    slug_seed = first_user_q or Path(conv["conv_path"]).stem
    slug = _slugify(slug_seed)

    runbook_rel = f"{RUNBOOKS_DIR}/{slug}.md"
    runbook_full = vault / runbook_rel
    # Collision: si ya existe un runbook con el mismo slug (otra conversation
    # destilada con query similar), sufijo con timestamp del original.
    if runbook_full.exists():
        ts_tag = Path(conv["conv_path"]).stem[:15]  # YYYY-MM-DD-HHMM
        runbook_rel = f"{RUNBOOKS_DIR}/{slug}-{ts_tag}.md"
        runbook_full = vault / runbook_rel

    today = datetime.now().strftime("%Y-%m-%d")
    fm_lines = [
        "---",
        "type: runbook",
        f"distilled_at: {today}",
        f"distilled_from: {conv['conv_path']}",
        f"confidence_avg: {conv['confidence_avg']:.3f}",
    ]
    if conv["sources_missing"]:
        fm_lines.append("original_sources_missing:")
        for s in conv["sources_missing"]:
            fm_lines.append(f"  - {s}")
    if conv["sources_present"]:
        fm_lines.append("original_sources_present:")
        for s in conv["sources_present"]:
            fm_lines.append(f"  - {s}")
    fm_lines.extend([
        "tags:",
        "  - runbook",
        "  - distilled",
        "---",
        "",
        f"# {first_user_q or slug_seed}",
        "",
        "> **Nota destilada automáticamente** desde una conversación previa "
        f"({conv['conv_path']}). Las fuentes originales fueron archivadas o "
        "borradas, pero la respuesta operativa quedó preservada acá.",
        "",
    ])
    content = "\n".join(fm_lines) + "\n" + "\n\n".join(bot_answers) + "\n"

    if not apply:
        return {
            "runbook_path": runbook_rel,
            "runbook_full": str(runbook_full),
            "written": False,
            "reason": "dry-run",
        }

    runbook_full.parent.mkdir(parents=True, exist_ok=True)
    runbook_full.write_text(content, encoding="utf-8")

    # Stamp del original para idempotencia: la próxima corrida saltea esta
    # conversation. Idempotent — re-escribir el mismo distilled_to es no-op.
    conv_full: Path = conv["conv_full"]
    try:
        raw = conv_full.read_text(encoding="utf-8", errors="ignore")
        m = _FM_RE.match(raw)
        if m:
            fm_text = m.group(1)
            if "distilled_to:" not in fm_text:
                new_fm = fm_text + f"\ndistilled_to: {runbook_rel}"
                new_raw = f"---\n{new_fm}\n---" + raw[m.end():]
                conv_full.write_text(new_raw, encoding="utf-8")
    except OSError:
        pass

    return {
        "runbook_path": runbook_rel,
        "runbook_full": str(runbook_full),
        "written": True,
    }


def run_distillation(
    vault: Path,
    apply: bool = False,
    min_confidence: float = MIN_CONFIDENCE_DEFAULT,
    limit: int | None = None,
    require_missing_source: bool = True,
) -> dict:
    """Top-level driver. Returns ``{candidates, distilled, skipped}``."""
    candidates = find_orphaned_conversations(
        vault,
        min_confidence=min_confidence,
        require_missing_source=require_missing_source,
    )
    if limit:
        candidates = candidates[:limit]
    distilled: list[dict] = []
    skipped: list[dict] = []
    for c in candidates:
        try:
            res = distill_conversation(c, vault, apply=apply)
            if res.get("written") or res.get("reason") == "dry-run":
                distilled.append({
                    "conv": c["conv_path"],
                    "runbook": res.get("runbook_path"),
                    "missing": len(c["sources_missing"]),
                    "confidence": c["confidence_avg"],
                })
            else:
                skipped.append({
                    "conv": c["conv_path"],
                    "reason": res.get("reason", "unknown"),
                })
        except Exception as e:
            skipped.append({"conv": c["conv_path"], "reason": str(e)[:120]})
    return {
        "candidates": len(candidates),
        "distilled": distilled,
        "skipped": skipped,
        "apply": apply,
    }
