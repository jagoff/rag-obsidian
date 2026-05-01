"""Data augmentation para el fine-tune de drafts WhatsApp (fase B, 2026-05-01).

Contrato:
  Input:  bridge DB de WhatsApp (~/repositories/whatsapp-mcp/whatsapp-bridge/
          store/messages.db) — todos los outgoing reales del user
          (`is_from_me=1`).
  Output: rows en `rag_draft_decisions` con `decision='approved_editar'`,
          `extra_json={"synthetic": true, "review_only": true,
          "augment_v": 1}`. Cada row es un par (bot_draft sintético,
          sent_text real) que el script `finetune_drafts.py` puede consumir.
  Goal:   subir el sample size del fine-tune de 5 → ~5000 pares para que
          el LoRA capture el TONO real del user (rioplatense, voseo, sin
          "¡"/"¿", acks atómicos) en lugar de overfitear a 4 frases.

Diseño deliberado — bot_drafts NO usan LLM:
  Generamos bot_drafts sintéticos por TEMPLATES rotativos (~12 variantes
  corporate-style aburridas). Razones:

  (1) El LLM call (qwen2.5:7b en Ollama) tardaría ~2-3s por draft × 6116
      outgoing = ~5h. Templates son instantáneos.
  (2) El signal que el FT necesita aprender NO es "transformar bot_draft
      X específico → output Y". Es "dado un contexto conversacional, mandar
      como Fer". El bot_draft sintético funciona como SCAFFOLD genérico que
      el modelo eventualmente aprende a IGNORAR a favor del contexto + tono
      target. Con menos diversidad de bot_draft, el modelo no se distrae
      memorizando templates.
  (3) Reproducible y auditable. Un LLM call introduce variance no-determinista.

  Trade-off conocido: el modelo aprende a "siempre rechazar" cualquier
  estructura corporate genérica, lo cual es lo que queremos en producción
  (el listener TS usa qwen2.5:14b sin FT y a veces sale corporate).

Filtros aplicados:
  - chat_jid NOT IN [status@broadcast, RagNet group, Notes inbox] —
    mismo set que el ingester de WhatsApp (`scripts/ingest_whatsapp.py`).
  - len(content) ∈ [10, 500] — descarta acks atómicos ("ok", "👍") y
    monólogos largos (no son drafts típicos).
  - content NO empieza con U+200B (anti-loop marker del bot).
  - Tiene contexto previo: ≥1 mensaje en el mismo chat dentro de 30min
    ANTES del outgoing target (sin contexto no hay nada útil que aprender).

Idempotencia:
  - El `draft_id` se compone como `synthetic-{wa_msg_id}` para garantizar
    unicidad por mensaje source.
  - Antes de insertar verificamos si ya existe una row con ese `draft_id`
    + `extra_json.synthetic=true` y la skipeamos (resume-friendly).
  - `--reset-synthetic` borra TODAS las rows synthetic previas antes de
    re-procesar (uso: cambio de versión del augmenter).

Uso:
  uv run python scripts/augment_drafts_dataset.py --dry-run --limit 50
  uv run python scripts/augment_drafts_dataset.py --limit 1000
  uv run python scripts/augment_drafts_dataset.py            # full corpus
  uv run python scripts/augment_drafts_dataset.py --reset-synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────

BRIDGE_DB = (
    Path.home() / "repositories" / "whatsapp-mcp"
    / "whatsapp-bridge" / "store" / "messages.db"
)

# Mismo set de exclusión que el ingester de WhatsApp — ver
# `scripts/ingest_whatsapp.py:97-107` para justificación detallada de
# cada JID. Hardcoded acá (en vez de importar) porque este script
# vive aislado del módulo `rag.integrations.whatsapp`.
EXCLUDE_JIDS = frozenset({
    "status@broadcast",
    "120363426178035051@g.us",         # RagNet (bot's own group)
    "5493425153999-1539438783@g.us",   # Notes inbox (notes-to-self)
})

# Anti-loop marker (zero-width space) que el listener bot prefija a sus
# outputs. Cualquier outgoing que empiece con esto es output del bot,
# no del user — descartar.
ANTILOOP_MARKER = "\u200B"

# Filtros de contenido.
MIN_CONTENT_LEN = 10           # acks atómicos (<10 chars) no aportan
MAX_CONTENT_LEN = 500          # mensajes largos (>500) no son drafts típicos
MAX_CONTEXT_GAP_MIN = 30       # contexto previo válido si <=30min antes
MIN_PRIOR_MSGS = 1             # exigir al menos 1 msg previo para aprender
MAX_PRIOR_MSGS = 5             # cap arriba — más no aporta y satura el prompt

# Bot-draft templates (corporate / aburrido / genérico). El target real
# (sent_text del user) es lo que el FT aprende. La diversidad acá es
# baja A PROPÓSITO — queremos que el modelo aprenda a ignorar el draft
# y apoyarse en el contexto + tono target.
BOT_DRAFT_TEMPLATES = (
    "Entendido, te respondo en breve.",
    "De acuerdo, lo confirmo apenas pueda.",
    "Recibido, ahora lo veo y te aviso.",
    "Claro, lo coordinamos sin problema.",
    "Perfecto, gracias por avisar.",
    "Hola, gracias por tu mensaje. ¿En qué te puedo ayudar?",
    "Sí, dale, lo vemos en el día.",
    "Entendido. Te confirmo a la brevedad.",
    "Genial, muchas gracias por la actualización.",
    "Hola! Sí, claro, sin problema.",
    "Buenas, recibido. Lo reviso y te respondo.",
    "Listo, gracias. Cualquier duda te aviso.",
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _pick_template(msg_id: str) -> str:
    """Pseudo-random template pick by msg id — determinista para idempotencia."""
    h = hashlib.md5(msg_id.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(BOT_DRAFT_TEMPLATES)
    return BOT_DRAFT_TEMPLATES[idx]


def _parse_ts(raw: str) -> datetime | None:
    """Parsea el timestamp del bridge DB ('2025-10-21 13:36:56-03:00')."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace(" ", "T"))
    except Exception:
        return None


def fetch_outgoing_with_context(
    bridge_path: Path,
    *,
    limit: int | None = None,
) -> list[dict]:
    if not bridge_path.exists():
        print(f"[error] bridge DB no encontrada: {bridge_path}", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(f"file:{bridge_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        excl_q = ",".join(["?"] * len(EXCLUDE_JIDS))
        sql = f"""
            SELECT id, chat_jid, content, timestamp
            FROM messages
            WHERE is_from_me = 1
              AND content IS NOT NULL
              AND content != ''
              AND length(content) BETWEEN ? AND ?
              AND chat_jid NOT IN ({excl_q})
              AND substr(content, 1, 1) != ?
            ORDER BY timestamp ASC
        """
        params: list = [MIN_CONTENT_LEN, MAX_CONTENT_LEN]
        params.extend(EXCLUDE_JIDS)
        params.append(ANTILOOP_MARKER)
        if limit:
            sql += " LIMIT ?"
            params.append(limit * 3)
        rows = list(conn.execute(sql, params))

        out: list[dict] = []
        for r in rows:
            target_ts = _parse_ts(r["timestamp"])
            if target_ts is None:
                continue
            min_ts = (target_ts - timedelta(minutes=MAX_CONTEXT_GAP_MIN)).isoformat(sep=" ")
            ctx_rows = list(conn.execute(
                """
                SELECT content, is_from_me, timestamp
                FROM messages
                WHERE chat_jid = ?
                  AND timestamp < ?
                  AND timestamp >= ?
                  AND content IS NOT NULL AND content != ''
                  AND substr(content, 1, 1) != ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (r["chat_jid"], r["timestamp"], min_ts, ANTILOOP_MARKER, MAX_PRIOR_MSGS),
            ))
            if len(ctx_rows) < MIN_PRIOR_MSGS:
                continue
            original_msgs = [
                {
                    "text": cr["content"],
                    "from_me": bool(cr["is_from_me"]),
                    "ts": cr["timestamp"],
                }
                for cr in reversed(ctx_rows)
            ]
            out.append({
                "id": r["id"],
                "chat_jid": r["chat_jid"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "original_msgs": original_msgs,
            })
            if limit and len(out) >= limit:
                break
        return out
    finally:
        conn.close()


def fetch_contact_names(bridge_path: Path, jids: set[str]) -> dict[str, str]:
    if not jids:
        return {}
    conn = sqlite3.connect(f"file:{bridge_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join(["?"] * len(jids))
        rows = conn.execute(
            f"SELECT jid, name FROM chats WHERE jid IN ({placeholders})",
            list(jids),
        ).fetchall()
        return {jid: (name or "") for jid, name in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def upsert_synthetic_pair(conn: sqlite3.Connection, item: dict, contact_name: str) -> bool:
    draft_id = f"synthetic-{item['id']}"
    existing = conn.execute(
        "SELECT id FROM rag_draft_decisions WHERE draft_id = ? LIMIT 1",
        (draft_id,),
    ).fetchone()
    if existing:
        return False

    bot_draft = _pick_template(item["id"])
    extra = {"synthetic": True, "review_only": True, "augment_v": 1}
    conn.execute(
        """
        INSERT INTO rag_draft_decisions
            (ts, draft_id, contact_jid, contact_name,
             original_msgs_json, bot_draft, decision, sent_text, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, 'approved_editar', ?, ?)
        """,
        (
            item["timestamp"],
            draft_id,
            item["chat_jid"],
            contact_name,
            json.dumps(item["original_msgs"], ensure_ascii=False),
            bot_draft,
            item["content"],
            json.dumps(extra),
        ),
    )
    return True


def reset_synthetic(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "DELETE FROM rag_draft_decisions "
        "WHERE draft_id LIKE 'synthetic-%' "
        "AND json_extract(extra_json, '$.synthetic') IN (1, 'true', 'True')"
    )
    return cur.rowcount


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment rag_draft_decisions con outgoing reales del user."
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Limitar cantidad de pares (debug).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build pairs + print stats, NO inserta.")
    ap.add_argument("--reset-synthetic", action="store_true",
                    help="Borrar TODAS las rows synthetic previas antes de procesar.")
    args = ap.parse_args()

    print("== augment drafts dataset (synthetic templates) ==", file=sys.stderr)
    print(f"  Bridge DB: {BRIDGE_DB}", file=sys.stderr)
    print(f"  Limit: {args.limit if args.limit else 'sin límite'}", file=sys.stderr)

    t0 = time.time()
    items = fetch_outgoing_with_context(BRIDGE_DB, limit=args.limit)
    elapsed = time.time() - t0
    print(f"  Outgoing válidos con contexto: {len(items)} ({elapsed:.1f}s)",
          file=sys.stderr)
    if not items:
        print("[warn] no hay outgoing con contexto suficiente. Bajá MIN_PRIOR_MSGS o "
              "MAX_CONTEXT_GAP_MIN si querés más samples.", file=sys.stderr)
        return

    print("\n== Preview (primeros 3 pares) ==", file=sys.stderr)
    for it in items[:3]:
        ctx = " | ".join(
            f"{'me' if m['from_me'] else 'them'}: {m['text'][:50]}"
            for m in it["original_msgs"]
        )
        print(f"  [{it['timestamp']}] chat={it['chat_jid'][:25]}", file=sys.stderr)
        print(f"    contexto: {ctx[:200]}", file=sys.stderr)
        print(f"    bot_draft: {_pick_template(it['id'])}", file=sys.stderr)
        print(f"    sent_text: {it['content'][:200]}", file=sys.stderr)
        print("", file=sys.stderr)

    jids = {it["chat_jid"] for it in items}
    names = fetch_contact_names(BRIDGE_DB, jids)
    print(f"  Resolved contact_name para {sum(1 for n in names.values() if n)}/"
          f"{len(jids)} JIDs distintos.", file=sys.stderr)

    if args.dry_run:
        print(f"\n[dry-run] no inserto. Procesaría {len(items)} pares.",
              file=sys.stderr)
        return

    n_inserted = 0
    n_skipped = 0
    n_reset = 0
    with rag._ragvec_state_conn() as conn:
        if args.reset_synthetic:
            n_reset = reset_synthetic(conn)
            print(f"  Reset synthetic: {n_reset} rows borradas", file=sys.stderr)
        for i, it in enumerate(items, 1):
            inserted = upsert_synthetic_pair(conn, it, names.get(it["chat_jid"], ""))
            if inserted:
                n_inserted += 1
            else:
                n_skipped += 1
            if i % 500 == 0:
                conn.commit()
                print(f"  [{i}/{len(items)}] inserted={n_inserted} skipped={n_skipped}",
                      file=sys.stderr)
        conn.commit()

    total_elapsed = time.time() - t0
    print(f"\n== DONE ({total_elapsed:.1f}s) ==", file=sys.stderr)
    print(f"  Inserted: {n_inserted}", file=sys.stderr)
    print(f"  Skipped (ya existían): {n_skipped}", file=sys.stderr)
    if args.reset_synthetic:
        print(f"  Reset previo: {n_reset}", file=sys.stderr)
    print(f"\nNext: re-correr `uv run --extra finetune python "
          f"scripts/finetune_drafts.py` con threshold ≥ 100", file=sys.stderr)


if __name__ == "__main__":
    main()
