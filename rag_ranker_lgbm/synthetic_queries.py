"""Generación sintética de queries desde el corpus para entrenar el ranker.

Sprint 2.A del cierre del loop de auto-aprendizaje (2026-04-26). Resuelve
el problema central del Sprint 2: con solo 73 queries de feedback real
el LightGBM lambdarank perdió en el A/B test (-6.35pp hit@5). Synthetic
queries augmentan ese training set con miles de pairs (query, positive_doc)
sin tocar al user.

Diseño:

  1. Para cada nota del vault, qwen2.5:7b genera 3-5 queries que esa nota
     respondería bien. Prompt en español rioplatense con ejemplos para
     forzar diversidad (factual / exploratory / conversational).
  2. Cada par (synthetic_query, source_note_path) se persiste a la tabla
     `rag_synthetic_queries` (idempotente — re-runs no duplican; se skipea
     notas ya procesadas con el mismo content hash).
  3. Hard negative mining (hard_negatives.py): para cada synthetic, los
     top-k chunks por similitud de embedding que NO son la source nota
     se marcan como hard negatives — los más informativos para entrenar.

Costos:

  - Generación: ~3-5s por nota con qwen2.5:7b. Con 1500 notas = ~75-125 min.
    Diseñado para ejecutar overnight como daemon nightly (con --resume).
  - Storage: ~5KB por synthetic. 7500 synthetics = ~37MB en SQL.
  - Re-generation: si una nota cambia (content hash distinto), re-genera
    selectivamente — no rebuild completo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_GEN_MODEL = "qwen2.5:7b"
DEFAULT_QUERIES_PER_NOTE = 4
DEFAULT_NUM_PREDICT = 400
DEFAULT_BODY_CHARS = 2500


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_synthetic_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            note_path TEXT NOT NULL,
            note_hash TEXT NOT NULL,
            query TEXT NOT NULL,
            query_kind TEXT,
            gen_model TEXT,
            gen_meta_json TEXT,
            UNIQUE(note_path, query)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_synth_queries_path "
        "ON rag_synthetic_queries(note_path)"
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end > 0:
        return text[end + 4:].lstrip("\n")
    return text


def _truncate_body(text: str, max_chars: int = DEFAULT_BODY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_break = cut.rfind("\n\n")
    if last_break > max_chars // 2:
        return cut[:last_break]
    return cut


def _generation_prompt(note_path: str, body: str, n: int) -> str:
    return f"""Sos un asistente que ayuda a mejorar un sistema de búsqueda sobre un vault de Obsidian.

Tenés esta nota:

PATH: {note_path}
CONTENIDO:
---
{body}
---

Tu tarea: generar {n} queries en español rioplatense que un usuario REAL haría buscando esta nota.

Variá deliberadamente entre:
- "factual": preguntas concretas con respuesta corta. Ej: "cuándo nació X", "cuánto cuesta Y".
- "exploratory": preguntas abiertas que invitan a explorar. Ej: "qué tengo sobre productividad", "info de mi proyecto".
- "conversational": preguntas naturales como las haría hablando. Ej: "che, qué onda con eso de X", "¿hay algo de Y?".

Reglas:
- Las queries DEBEN ser distintas entre sí (no parafrasear la misma cosa N veces).
- NO copies frases textuales de la nota — formulá como si NO supieras qué hay adentro.
- Naturales, casuales — el usuario no escribe formal en su sistema personal.

Respondé JSON estricto sin preámbulo:
{{"queries": [
  {{"q": "...", "kind": "factual"}},
  {{"q": "...", "kind": "exploratory"}},
  ...
]}}"""


def _parse_generation_response(raw: str) -> list[dict[str, str]]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    queries = data.get("queries") or []
    if not isinstance(queries, list):
        return []
    out: list[dict[str, str]] = []
    for item in queries:
        if not isinstance(item, dict):
            continue
        q = item.get("q") or item.get("query")
        if not q or not isinstance(q, str):
            continue
        kind = item.get("kind") or "unknown"
        out.append({"q": q.strip(), "kind": str(kind).strip().lower()})
    return out


def _default_llm_call(prompt: str, *, model: str = DEFAULT_GEN_MODEL) -> str:
    import rag

    resp = rag._summary_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            **rag.HELPER_OPTIONS,
            "num_ctx": 4096,
            "num_predict": DEFAULT_NUM_PREDICT,
        },
        keep_alive=rag.OLLAMA_KEEP_ALIVE,
        format="json",
    )
    return (resp.message.content or "").strip()


def _iter_vault_notes(
    vault: Path,
    *,
    is_excluded_fn: Callable[[str], bool] | None = None,
) -> list[tuple[Path, str]]:
    if is_excluded_fn is None:
        try:
            import rag
            is_excluded_fn = rag.is_excluded
        except Exception:
            def is_excluded_fn(p):  # type: ignore[no-redef]
                return False

    out: list[tuple[Path, str]] = []
    for md in vault.rglob("*.md"):
        try:
            rel = str(md.relative_to(vault))
        except ValueError:
            continue
        if any(seg.startswith(".") for seg in rel.split("/") if seg):
            continue
        if is_excluded_fn(rel):
            continue
        out.append((md, rel))
    return out


def _already_generated_for_hash(
    conn: sqlite3.Connection, note_path: str, note_hash: str
) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_queries "
        "WHERE note_path = ? AND note_hash = ?",
        (note_path, note_hash),
    ).fetchone()
    return (row[0] or 0) > 0


def generate_synthetic_queries(
    conn: sqlite3.Connection,
    *,
    vault: Path | None = None,
    limit: int | None = None,
    queries_per_note: int = DEFAULT_QUERIES_PER_NOTE,
    model: str = DEFAULT_GEN_MODEL,
    body_chars: int = DEFAULT_BODY_CHARS,
    dry_run: bool = False,
    llm_call: Callable[..., str] | None = None,
    is_excluded_fn: Callable[[str], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if vault is None:
        import rag
        vault = rag._resolve_vault_path()

    _ensure_table(conn)

    if llm_call is None:
        llm_call = _default_llm_call

    notes = _iter_vault_notes(vault, is_excluded_fn=is_excluded_fn)
    if limit is not None:
        notes = notes[:limit]

    metrics: dict[str, int] = {
        "n_notes_seen": len(notes),
        "n_notes_skipped_unchanged": 0,
        "n_notes_skipped_empty": 0,
        "n_notes_processed": 0,
        "n_notes_llm_failed": 0,
        "n_queries_inserted": 0,
        "n_queries_skipped_duplicate": 0,
    }

    started = time.time()
    pairs_generated: list[dict[str, str]] = []

    for i, (abs_path, rel_path) in enumerate(notes):
        if progress_callback is not None:
            progress_callback(i, len(notes), rel_path)

        try:
            raw = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            metrics["n_notes_skipped_empty"] += 1
            continue
        body = _strip_frontmatter(raw).strip()
        if not body:
            metrics["n_notes_skipped_empty"] += 1
            continue

        note_hash = _content_hash(body)

        if _already_generated_for_hash(conn, rel_path, note_hash):
            metrics["n_notes_skipped_unchanged"] += 1
            continue

        body_truncated = _truncate_body(body, max_chars=body_chars)
        prompt = _generation_prompt(rel_path, body_truncated, queries_per_note)

        try:
            raw_resp = llm_call(prompt, model=model)
        except Exception as exc:
            logger.warning("LLM call failed for %s: %s", rel_path, exc)
            metrics["n_notes_llm_failed"] += 1
            continue

        queries = _parse_generation_response(raw_resp)
        if not queries:
            metrics["n_notes_llm_failed"] += 1
            continue

        metrics["n_notes_processed"] += 1
        gen_meta = {
            "model": model,
            "queries_per_note": queries_per_note,
            "body_chars_used": len(body_truncated),
            "n_queries_returned": len(queries),
        }
        gen_meta_json = json.dumps(gen_meta)
        now_iso = datetime.now().isoformat(timespec="seconds")

        for q_obj in queries[:queries_per_note]:
            q_text = q_obj["q"]
            q_kind = q_obj.get("kind", "unknown")
            pair = {"note_path": rel_path, "query": q_text, "kind": q_kind}
            pairs_generated.append(pair)

            if dry_run:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO rag_synthetic_queries "
                    "(ts, note_path, note_hash, query, query_kind, gen_model, gen_meta_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        now_iso, rel_path, note_hash, q_text, q_kind,
                        model, gen_meta_json,
                    ),
                )
                if conn.total_changes > 0:
                    metrics["n_queries_inserted"] += 1
                else:
                    metrics["n_queries_skipped_duplicate"] += 1
            except sqlite3.IntegrityError:
                metrics["n_queries_skipped_duplicate"] += 1

    elapsed = time.time() - started
    return {
        **metrics,
        "duration_s": round(elapsed, 1),
        "pairs_sample": pairs_generated[:20],
        "n_pairs_total": len(pairs_generated),
        "dry_run": dry_run,
    }


def get_synthetic_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    _ensure_table(conn)
    n_total = conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_queries"
    ).fetchone()[0]
    n_unique_notes = conn.execute(
        "SELECT COUNT(DISTINCT note_path) FROM rag_synthetic_queries"
    ).fetchone()[0]
    by_kind = dict(conn.execute(
        "SELECT query_kind, COUNT(*) FROM rag_synthetic_queries "
        "GROUP BY query_kind"
    ).fetchall())
    by_model = dict(conn.execute(
        "SELECT gen_model, COUNT(*) FROM rag_synthetic_queries "
        "GROUP BY gen_model"
    ).fetchall())
    return {
        "n_total": n_total,
        "n_unique_notes": n_unique_notes,
        "by_kind": by_kind,
        "by_model": by_model,
    }
