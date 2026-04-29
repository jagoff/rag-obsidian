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

Cross-source extension (Quick Win #4, 2026-04-29):

  - `generate_synthetic_queries_for_cross_source(conn, source, ...)` lee
    items desde la collection vectorial (`meta_obsidian_notes_v11`)
    filtrando por `source` field (gmail / calendar / drive / safari /
    whatsapp / reminders / contacts / calls). Usa el `file` URI como
    `note_path` (ej: `gmail://thread/19dd...`).
  - Mismo prompt + parsing, distinta fuente de datos. La unique constraint
    `(note_path, query)` previene colisiones entre vault paths y URIs.
  - Calibración cross-source (`_gather_calibration_pairs` en rag.py)
    consume estas synthetic pairs como fallback cuando una source tiene
    < min_pairs de feedback real, marcando el row con
    `model_version='isotonic-v1-synth'`.

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

    # Quick Win #4 (2026-04-29): per-source distribution. Inferimos el
    # source desde `note_path` — URIs (gmail://, calendar://, ...) → ese
    # scheme; cualquier otro → "vault". El infer es el mismo que aplica
    # `_classify_source_from_path` en rag.py para el calibrate.
    by_source: dict[str, int] = {}
    for (note_path,) in conn.execute(
        "SELECT note_path FROM rag_synthetic_queries"
    ):
        if "://" in (note_path or ""):
            scheme = note_path.split("://", 1)[0].strip().lower()
            if scheme in ("wa", "whats_app"):
                scheme = "whatsapp"
            if scheme in ("gdrive",):
                scheme = "drive"
            by_source[scheme] = by_source.get(scheme, 0) + 1
        else:
            by_source["vault"] = by_source.get("vault", 0) + 1

    return {
        "n_total": n_total,
        "n_unique_notes": n_unique_notes,
        "by_kind": by_kind,
        "by_model": by_model,
        "by_source": by_source,
    }


# ─── Cross-source generation (Quick Win #4, 2026-04-29) ────────────────


# Sources soportadas para generación cross-source. Mismo set que
# `_CALIBRATION_SOURCES` en rag.py. Vault excluido a propósito —
# `generate_synthetic_queries` ya cubre ese caso desde el FS.
CROSS_SOURCE_SOURCES: tuple[str, ...] = (
    "whatsapp", "calendar", "gmail", "drive",
    "reminders", "safari", "contacts", "calls",
)

# Caracteres mínimos del documento para considerarlo procesable. WhatsApp
# messages cortos ("ok", "sí") no aportan signal útil para queries
# sintéticas — preferimos skipearlos antes que pedirle al LLM que invente.
MIN_DOC_CHARS_CROSS_SOURCE = 50


def _iter_cross_source_items(
    state_conn: sqlite3.Connection, source: str
) -> list[tuple[str, str, str]]:
    """Lista (file_uri, title, document) para todos los items de la
    collection vectorial cuyo `source` matchee.

    Args:
        state_conn: connection a `ragvec.db` (NO al telemetry — los items
            viven en `meta_obsidian_notes_v11`).
        source: nombre normalizado (whatsapp, gmail, etc.).

    Returns:
        Lista de tuples ordenadas por file URI. Un mismo file con varios
        chunks aparece UNA vez con el chunk más largo (proxy de "el chunk
        más representativo del item completo"). Vacía si la tabla no
        existe o el source no tiene items.
    """
    try:
        rows = state_conn.execute(
            """
            SELECT file, title, document
            FROM meta_obsidian_notes_v11
            WHERE source = ?
              AND file IS NOT NULL
              AND file != ''
            ORDER BY file ASC, length(document) DESC
            """,
            (source,),
        ).fetchall()
    except sqlite3.OperationalError:
        # Collection no creada aún (pre-ingest). Silent-fail con lista
        # vacía — el caller reporta n_notes_seen=0 en metrics.
        return []

    # Dedup por file: el primer row con file=X (el de mayor length por
    # el ORDER BY) gana. Los demás son chunks distintos del mismo item.
    seen: dict[str, tuple[str, str, str]] = {}
    for file_uri, title, document in rows:
        if file_uri in seen:
            continue
        title_clean = (title or "").strip()
        document_clean = (document or "").strip()
        seen[file_uri] = (file_uri, title_clean, document_clean)
    return list(seen.values())


def _cross_source_prompt(
    source: str, file_uri: str, title: str, body: str, n: int
) -> str:
    """Prompt LLM para generar queries que traerían un item cross-source.

    Diferente al prompt de vault porque la naturaleza del item cambia:
    un email no se busca igual que una nota personal. Damos contexto del
    source para que el LLM ajuste el tono y formato de las queries.
    """
    source_hints = {
        "whatsapp": "un mensaje de WhatsApp (puede ser de un chat individual o grupo)",
        "gmail": "un thread de email (puede tener varios mensajes — el body es el del último o un resumen)",
        "calendar": "un evento del calendario (título + fecha + descripción)",
        "drive": "un documento de Google Drive (Doc/Sheet/Slide)",
        "reminders": "un recordatorio (Apple Reminders) — task con fecha opcional",
        "safari": "una página web visitada (título + URL + opcionalmente snippet)",
        "contacts": "un contacto (nombre + datos de contacto + notas)",
        "calls": "un registro de llamada (contacto + duración + dirección)",
    }
    hint = source_hints.get(source, f"un item de tipo `{source}`")

    title_line = f"TÍTULO: {title}\n" if title else ""

    return f"""Sos un asistente que ayuda a mejorar un sistema de búsqueda sobre múltiples fuentes (vault Obsidian + integraciones).

Tenés {hint}:

ID: {file_uri}
{title_line}CONTENIDO:
---
{body}
---

Tu tarea: generar {n} queries en español rioplatense que un usuario REAL haría buscando ESTE item específico. El usuario NO sabe que el resultado va a venir de "{source}" — solo describe qué información necesita.

Variá deliberadamente entre:
- "factual": preguntas concretas con respuesta corta. Ej: "cuándo es la reunión con X", "cuánto pagó Y".
- "exploratory": preguntas abiertas. Ej: "qué pasó con el proyecto Z", "info de mi cliente W".
- "conversational": preguntas naturales. Ej: "che, qué onda con eso de A", "hay algo de B?".

Reglas:
- Las queries DEBEN ser distintas entre sí.
- NO copies frases textuales del item — formulá como si NO supieras qué hay adentro.
- Naturales, casuales — el usuario no escribe formal en su sistema personal.
- NO menciones la fuente ("encontrá en mi gmail" / "buscá en whatsapp") — el ranker tiene que decidir solo.

Respondé JSON estricto sin preámbulo:
{{"queries": [
  {{"q": "...", "kind": "factual"}},
  {{"q": "...", "kind": "exploratory"}},
  ...
]}}"""


def generate_synthetic_queries_for_cross_source(
    conn: sqlite3.Connection,
    *,
    source: str,
    state_conn: sqlite3.Connection | None = None,
    limit: int | None = None,
    queries_per_note: int = DEFAULT_QUERIES_PER_NOTE,
    model: str = DEFAULT_GEN_MODEL,
    body_chars: int = DEFAULT_BODY_CHARS,
    dry_run: bool = False,
    llm_call: Callable[..., str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Generar queries sintéticas para items cross-source.

    Análogo a `generate_synthetic_queries` (que lee del FS del vault),
    pero pull desde la collection vectorial (`meta_obsidian_notes_v11`)
    filtrando por `source` field. El `file` URI del meta se usa como
    `note_path` en `rag_synthetic_queries` — la unique constraint
    `(note_path, query)` previene colisiones con vault paths reales.

    Args:
        conn: connection a `telemetry.db` (donde vive
            `rag_synthetic_queries`).
        source: source target — debe estar en `CROSS_SOURCE_SOURCES`.
            Llamarlo con "vault" raise — usá la otra función para vault.
        state_conn: connection a `ragvec.db` (collection vectorial). Si
            None, abre con `rag._ragvec_state_conn()`.
        limit: máximo items a procesar. None = todos.
        queries_per_note: cuántas queries pedir al LLM por item.
        model: modelo LLM (default qwen2.5:7b — mismo que vault).
        body_chars: truncamiento del body al pasar al LLM.
        dry_run: no escribir, solo reportar.
        llm_call: override callable para tests.
        progress_callback: callback (i, total, file_uri).

    Returns:
        dict con métricas + sample. Forma compatible con
        `generate_synthetic_queries` (vault) más un campo `source`.

    Raises:
        ValueError: si `source` no está en `CROSS_SOURCE_SOURCES`.
    """
    if source not in CROSS_SOURCE_SOURCES:
        raise ValueError(
            f"source={source!r} no es cross-source válido. "
            f"Valores aceptados: {CROSS_SOURCE_SOURCES}. "
            f"Para vault, usá `generate_synthetic_queries` (FS-based)."
        )

    _ensure_table(conn)

    if llm_call is None:
        llm_call = _default_llm_call

    # Lazy connection a ragvec.db si no nos pasaron una. Soportamos
    # `_ragvec_state_conn` siendo un context manager (producción) Y
    # un callable que retorne un objeto con `close()` (tests
    # sintéticos), así que abrimos defensivamente.
    # Quick Win #4 fix: lazy conn a ragvec.db (donde viven los meta).
    # NO usamos _ragvec_state_conn — apunta a telemetry.db.
    own_state_conn = False
    if state_conn is None:
        import rag
        state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
        own_state_conn = True

    try:
        items = _iter_cross_source_items(state_conn, source)
    finally:
        if own_state_conn:
            try:
                state_conn.close()
            except Exception:
                pass

    if limit is not None:
        items = items[:limit]

    metrics: dict[str, int] = {
        "n_notes_seen": len(items),
        "n_notes_skipped_unchanged": 0,
        "n_notes_skipped_empty": 0,
        "n_notes_processed": 0,
        "n_notes_llm_failed": 0,
        "n_queries_inserted": 0,
        "n_queries_skipped_duplicate": 0,
    }

    started = time.time()
    pairs_generated: list[dict[str, str]] = []

    for i, (file_uri, title, body) in enumerate(items):
        if progress_callback is not None:
            progress_callback(i, len(items), file_uri)

        if not body or len(body) < MIN_DOC_CHARS_CROSS_SOURCE:
            metrics["n_notes_skipped_empty"] += 1
            continue

        note_hash = _content_hash(body)

        if _already_generated_for_hash(conn, file_uri, note_hash):
            metrics["n_notes_skipped_unchanged"] += 1
            continue

        body_truncated = _truncate_body(body, max_chars=body_chars)
        prompt = _cross_source_prompt(
            source, file_uri, title, body_truncated, queries_per_note,
        )

        try:
            raw_resp = llm_call(prompt, model=model)
        except Exception as exc:
            logger.warning(
                "LLM call failed for %s/%s: %s", source, file_uri, exc
            )
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
            # Marcamos el source en el meta para que el calibrate /
            # stats lo lean sin re-parsear el URI scheme.
            "source": source,
        }
        gen_meta_json = json.dumps(gen_meta)
        now_iso = datetime.now().isoformat(timespec="seconds")

        for q_obj in queries[:queries_per_note]:
            q_text = q_obj["q"]
            q_kind = q_obj.get("kind", "unknown")
            pair = {
                "note_path": file_uri,
                "query": q_text,
                "kind": q_kind,
                "source": source,
            }
            pairs_generated.append(pair)

            if dry_run:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO rag_synthetic_queries "
                    "(ts, note_path, note_hash, query, query_kind, gen_model, gen_meta_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        now_iso, file_uri, note_hash, q_text, q_kind,
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
        "source": source,
    }
