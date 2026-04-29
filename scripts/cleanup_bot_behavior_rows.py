"""Delete bot-initiated impression rows from `rag_behavior` con heurística post-hoc.

Pre-2026-04-28 el `retrieve()` siempre logueaba `log_impressions(source="cli")`
hardcodeado. Bot-initiated retrieves (anticipatory cada 10min, followup
daemon, rag eval) terminaban contaminando la tabla con cientos/miles de
impressions/día atribuidas al user — el ranker training las consumía como
signal del user, aprendiendo a despriorizar paths que el bot mismo busca
seguido.

Los commits del 2026-04-28 (`fd97829`, `16df67e`, `4750041`, `2c794b6`,
`bb10504`) cierran el loop FUTURO — bot-initiated retrieves van con
`source` correcto. Pero las rows históricas siguen en la tabla.

Este script las identifica por heurística + las borra (dry-run por
default, `--apply` para ejecutar).

## Heurísticas de detección

Las heurísticas tienen recall imperfecto (algunas bot rows escapan, algunas
user rows se confunden) — son lo mejor que se puede hacer sin un caller
field histórico. El user puede ajustar los thresholds o correr en modo
"audit-only" para inspeccionar sin borrar.

### Heurística A: orphan impressions (default + más conservativa)

Una `event="impression"` row es bot-initiated si NO tiene NINGÚN positive
event compañero (open / copy / save / kept) en la misma `(query, path)`
dentro de una ventana de ±N minutos (default 60).

Lógica: el user genera ~10-30% click-through rate sobre lo que ve. Una
impresión orfaneada después de 1h es señal de "bot disparó retrieve, no
hubo click humano". El bot dispara muchísimas (anticipatory cada 10 min ×
top-3-5 paths × 144 ticks/día = 600-2000 impressions/día sin clicks).

Falsa positivos: queries donde el user vio la respuesta inline en el chat
y NO necesitó abrir la nota (los embeddings + rerank ya bastaron para
generar respuesta correcta). Esto es ~30-50% de los user queries reales —
significa que vamos a tirar algunos legítimos. El damage es: el ranker
pierde signal de paths que el user vio "exitosamente" pero no abrió. Es
aceptable para limpiar el bulk de pollution; mejor perder ese signal débil
que mantener 100k+ rows fake.

### Heurística B: eval set matches (opt-in con `--purge-eval`)

Carga `queries.yaml` (default `eval/queries.yaml`) y filtra rows cuya
`query` (case-insensitive, normalised whitespace) matchea cualquier
question del eval set. Esto pesca eval runs históricos que pasaron por
`source="cli"` antes del kill-switch `RAG_SKIP_BEHAVIOR_LOG`.

### Heurística C: very-short queries (opt-in con `--purge-short-queries`)

Filtra rows con queries < N chars (default 10). El anticipatory-calendar
manda títulos cortos (ej. "reunión", "almuerzo Maria") como queries; el
user típico tipea queries de 20-80 chars. Heurística más agresiva, more
false positives — usar con cuidado.

## Output dry-run

Por cada heurística aplicada, imprime: cuántas rows matchea, distribution
por path / event, y un sample de 10 queries. NO borra nada hasta `--apply`.

## Backup automático

Antes de cualquier DELETE con `--apply`, dumpea las rows que se van a
borrar a `~/.local/share/obsidian-rag/cleanup_backup_<ts>.jsonl`. Si te
arrepentís, podés re-importar con `sqlite3 ... ".import ..."` (manual).

## Uso

    # Dry-run con heurística A (default — más conservadora):
    python scripts/cleanup_bot_behavior_rows.py

    # Dry-run con todas las heurísticas:
    python scripts/cleanup_bot_behavior_rows.py --purge-eval --purge-short-queries

    # Aplicar (CON confirmación interactiva):
    python scripts/cleanup_bot_behavior_rows.py --apply

    # Aplicar sin confirm (cron / scripts):
    python scripts/cleanup_bot_behavior_rows.py --apply --yes

    # Custom orphan window (más agresivo: 5 min en vez de 60):
    python scripts/cleanup_bot_behavior_rows.py --orphan-window-min 5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


_POSITIVE_EVENTS = ("open", "copy", "save", "kept", "open_external")


def _heuristic_a_orphan_impressions(
    conn: sqlite3.Connection,
    *,
    orphan_window_minutes: int = 60,
) -> list[int]:
    """Identifica rowids de impressions sin positive event compañero en la
    misma `(query, path)` dentro de la ventana ±`orphan_window_minutes`.

    SQL strategy: LEFT JOIN rag_behavior consigo misma; filter rows donde
    el JOIN no encontró pareja positiva. Mantiene queries vacías fuera del
    delete set por defensividad.
    """
    placeholders = ",".join("?" for _ in _POSITIVE_EVENTS)
    sql = (
        "SELECT b1.id "
        "FROM rag_behavior b1 "
        "WHERE b1.event = 'impression' "
        "  AND b1.source = 'cli' "
        "  AND b1.query IS NOT NULL AND b1.query != '' "
        "  AND NOT EXISTS ("
        "    SELECT 1 FROM rag_behavior b2 "
        "    WHERE b2.path = b1.path "
        f"      AND b2.event IN ({placeholders}) "
        "      AND ABS(strftime('%s', b2.ts) - strftime('%s', b1.ts)) <= ? "
        "  )"
    )
    params = (*_POSITIVE_EVENTS, orphan_window_minutes * 60)
    cur = conn.execute(sql, params)
    return [row[0] for row in cur.fetchall()]


def _heuristic_b_eval_queries(
    conn: sqlite3.Connection,
    queries_yaml: Path,
) -> list[int]:
    """Identifica rowids de impressions cuya `query` matchea el eval set.

    Carga `queries_yaml`, normaliza (lower + whitespace collapse), y hace
    SQL match contra rag_behavior.query (también normalizada con LOWER + TRIM).
    """
    if not queries_yaml.is_file():
        return []
    try:
        import yaml  # PyYAML — opt dep, fallback simple
        data = yaml.safe_load(queries_yaml.read_text(encoding="utf-8"))
    except Exception:
        # Fallback parser: regex `^- question:` lines.
        import re
        text = queries_yaml.read_text(encoding="utf-8", errors="ignore")
        questions = re.findall(r"^\s*-?\s*question:\s*['\"]?(.+?)['\"]?\s*$",
                                text, re.MULTILINE)
        data = {"singles": [{"question": q} for q in questions]}

    eval_questions: set[str] = set()
    for section in ("singles", "chains"):
        items = (data or {}).get(section) or []
        for item in items:
            if isinstance(item, dict):
                q = item.get("question") or ""
                if q:
                    eval_questions.add(_normalise(q))
            elif isinstance(item, list):
                # Chain: list of turns.
                for turn in item:
                    if isinstance(turn, dict):
                        q = turn.get("question") or ""
                        if q:
                            eval_questions.add(_normalise(q))

    if not eval_questions:
        return []

    # Pull all impression queries; do the match in Python (sqlite3 doesn't
    # support `IN (set of normalised strings)` cleanly when normalisation
    # is non-trivial). Volume bound by impression count — 190k rows × ~50
    # bytes = 10MB, fine.
    cur = conn.execute(
        "SELECT id, query FROM rag_behavior "
        "WHERE event = 'impression' AND source = 'cli' "
        "  AND query IS NOT NULL AND query != ''"
    )
    matches: list[int] = []
    for row in cur.fetchall():
        rid, q = row
        if _normalise(q) in eval_questions:
            matches.append(rid)
    return matches


def _heuristic_c_short_queries(
    conn: sqlite3.Connection,
    min_chars: int = 10,
) -> list[int]:
    """Identifica rowids de impressions cuya `query` es muy corta. El
    anticipatory-calendar usa títulos de eventos como query (típicamente
    1-3 palabras). User queries reales suelen ser 20+ chars.
    """
    cur = conn.execute(
        "SELECT id FROM rag_behavior "
        "WHERE event = 'impression' AND source = 'cli' "
        "  AND query IS NOT NULL AND length(trim(query)) < ?",
        (min_chars,),
    )
    return [row[0] for row in cur.fetchall()]


def _normalise(s: str) -> str:
    """Lowercase + whitespace collapse for query matching."""
    return " ".join((s or "").lower().split())


def _summarise_rows(
    conn: sqlite3.Connection,
    ids: list[int],
    label: str,
) -> dict:
    """Render summary of a row set: count, top paths, top queries, sample.

    SQLite limita ~999 variables por query, así que con ID lists grandes
    (>10k rows típicos) hay que chunkear el SELECT — sino tira
    `OperationalError: too many SQL variables`.
    """
    from collections import Counter
    if not ids:
        return {"label": label, "count": 0, "top_paths": [], "top_queries": [],
                "sample": []}
    paths = Counter()
    queries = Counter()
    sample: list[dict] = []
    chunk_size = 500  # safe under SQLite's 999 default
    rows_seen = 0
    sample_target = 10
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(
            f"SELECT path, query, ts FROM rag_behavior WHERE id IN ({placeholders})",
            chunk,
        )
        for path, query, ts in cur.fetchall():
            paths[path] += 1
            queries[_normalise(query or "")[:80]] += 1
            if len(sample) < sample_target:
                sample.append({"path": path, "query": (query or "")[:80],
                               "ts": ts})
            rows_seen += 1
    return {
        "label": label,
        "count": len(ids),
        "top_paths": paths.most_common(10),
        "top_queries": queries.most_common(10),
        "sample": sample,
    }


def _print_summary(summary: dict) -> None:
    print(f"\n── {summary['label']} ──")
    print(f"  count: {summary['count']:,}")
    if not summary["count"]:
        return
    print(f"  top paths:")
    for p, n in summary["top_paths"]:
        print(f"    {n:>6,}  {p[:80]}")
    print(f"  top queries:")
    for q, n in summary["top_queries"]:
        print(f"    {n:>6,}  {q[:80]}")
    print(f"  sample (10):")
    for s in summary["sample"]:
        print(f"    [{s['ts']}] {s['query'][:60]:<60}  → {s['path'][:50]}")


def _backup_rows(conn: sqlite3.Connection, ids: list[int]) -> Path:
    """Dump rows to a JSONL file before delete. Returns the path.

    Chunked SELECT (500/batch) para evitar el SQLite variable limit con
    ID lists grandes.
    """
    if not ids:
        return Path()
    state_dir = Path.home() / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = state_dir / f"cleanup_backup_{ts}.jsonl"
    cols_query = ("id", "ts", "trace_id", "source", "event", "path", "query",
                  "rank", "dwell_s", "extra_json")
    chunk_size = 500
    with backup_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(
                f"SELECT {', '.join(cols_query)} FROM rag_behavior "
                f"WHERE id IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                d = dict(zip(cols_query, row))
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    return backup_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true",
                    help="Ejecutar el DELETE (default: dry-run)")
    p.add_argument("--yes", action="store_true",
                    help="Skipear confirm interactiva (use solo en cron)")
    p.add_argument("--orphan-window-min", type=int, default=60,
                    help="Ventana en minutos para heurística A (default: 60)")
    p.add_argument("--purge-eval", action="store_true",
                    help="Aplicar heurística B (eval set matches)")
    p.add_argument("--queries-yaml", type=Path,
                    default=Path(__file__).resolve().parent.parent / "eval/queries.yaml",
                    help="Path al eval set (default: eval/queries.yaml)")
    p.add_argument("--purge-short-queries", action="store_true",
                    help="Aplicar heurística C (queries <10 chars)")
    p.add_argument("--short-min-chars", type=int, default=10,
                    help="Threshold para heurística C (default: 10)")
    args = p.parse_args()

    print(f"[cleanup] DB: {rag.DB_PATH}")
    print(f"[cleanup] mode: {'APPLY' if args.apply else 'dry-run'}")

    with rag._ragvec_state_conn() as conn:
        # Snapshot total para context.
        total = conn.execute(
            "SELECT COUNT(*) FROM rag_behavior WHERE event = 'impression' "
            "AND source = 'cli'"
        ).fetchone()[0]
        print(f"[cleanup] total cli|impression rows: {total:,}")

        all_ids: set[int] = set()

        # Heurística A — siempre activa.
        ids_a = _heuristic_a_orphan_impressions(
            conn, orphan_window_minutes=args.orphan_window_min,
        )
        _print_summary(_summarise_rows(
            conn, ids_a,
            f"Heurística A: orphan impressions (window ±{args.orphan_window_min}min)",
        ))
        all_ids.update(ids_a)

        # Heurística B — opt-in.
        if args.purge_eval:
            ids_b = _heuristic_b_eval_queries(conn, args.queries_yaml)
            _print_summary(_summarise_rows(
                conn, ids_b,
                f"Heurística B: eval set matches (yaml={args.queries_yaml.name})",
            ))
            all_ids.update(ids_b)

        # Heurística C — opt-in.
        if args.purge_short_queries:
            ids_c = _heuristic_c_short_queries(
                conn, min_chars=args.short_min_chars,
            )
            _print_summary(_summarise_rows(
                conn, ids_c,
                f"Heurística C: short queries (<{args.short_min_chars} chars)",
            ))
            all_ids.update(ids_c)

        delete_count = len(all_ids)
        print(f"\n[cleanup] union de rowids a borrar: {delete_count:,} "
              f"({100 * delete_count / max(total, 1):.1f}% de cli|impression)")
        print(f"[cleanup] rows que sobreviven: {total - delete_count:,}")

        if not args.apply:
            print(f"\n[cleanup] DRY-RUN — re-correr con --apply para ejecutar")
            return 0

        if not all_ids:
            print(f"\n[cleanup] nada para borrar")
            return 0

        if not args.yes:
            confirm = input(f"\nConfirmar DELETE de {delete_count:,} rows? [y/N] ")
            if confirm.lower() not in ("y", "yes", "s", "si", "sí"):
                print("[cleanup] abortado por el user")
                return 1

        # Backup antes del DELETE.
        ids_list = sorted(all_ids)
        backup_path = _backup_rows(conn, ids_list)
        print(f"[cleanup] backup escrito: {backup_path}")

        # DELETE en chunks de 500 (SQLite soporta hasta 999 args por default).
        deleted = 0
        chunk_size = 500
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM rag_behavior WHERE id IN ({placeholders})",
                chunk,
            )
            deleted += len(chunk)
            if deleted % 5000 == 0:
                print(f"[cleanup]   borrados: {deleted:,}/{len(ids_list):,}")
        conn.commit()
        print(f"[cleanup] DELETE completo: {deleted:,} rows")

    print(f"[cleanup] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
