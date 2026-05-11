#!/usr/bin/env python3
"""Backfill `rag_draft_decisions` desde `~/.local/share/whatsapp-listener/draft.jsonl`.

Contexto (2026-05-10):
  El listener TS llama POST `/api/draft/decision` cada vez que el user
  decide sobre un draft (approve / reject / edit) o el draft expira.
  Hasta 2026-05-10 (later) el path era `/si /no /editar` standalone;
  desde entonces es reply-by-quote al draft posteado en RagNet (mismas
  decisions persistidas: `approved_si`, `approved_editar`, `rejected`,
  `expired`). Eso persiste en la tabla SQL `rag_draft_decisions`
  (telemetry.db). Pero el endpoint y/o el helper Python
  `_record_draft_decision` empezaron a estar disponibles bastante
  después de que el listener arrancara — el JSONL local `draft.jsonl`
  tiene 13 días de eventos (2026-04-27 → 2026-05-10), mientras que la
  tabla SQL tiene solo rows de 2026-05-10.

Resultado: 30+ pares de preference (approved_editar, rejected,
expired) no se contabilizan para el path de fine-tune (DPO necesita
≥100 pares). Backfill desde el JSONL los recupera y normaliza la
pipeline para que adelante todo se sume al mismo bucket.

Algoritmo:
  1. Indexar todos los `kind="generated"` del JSONL por `id` (draft_id).
  2. Iterar `kind` ∈ {approved, rejected, expired}:
     · Buscar el `generated` correspondiente para reconstruir
       `bot_draft` + `original_msgs` (que el evento de decisión NO
       incluye — solo trae `sent_text` y `approval_kind`).
     · Mapear `kind` + `approval_kind` a `decision`:
         approved + approval_kind=si    → approved_si
         approved + approval_kind=editar → approved_editar
         rejected                        → rejected
         expired                         → expired
  3. INSERT con `ts` literal del JSONL (no `now()`) + dedupe por
     `(draft_id, decision)` (alguien podría haber loggeado dos
     decisiones para el mismo draft_id — quedamos con la primera).
  4. Print stats: nuevas filas insertadas + skipped (dedupe hits).

Uso:
  uv run python scripts/backfill_draft_decisions.py
  uv run python scripts/backfill_draft_decisions.py --dry-run

Re-runnable: idempotente. Usa INSERT OR IGNORE con UNIQUE constraint
implícita via composite key check pre-INSERT.

Rollback: `DELETE FROM rag_draft_decisions WHERE ts < '2026-05-10T13:00';`
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

JSONL = Path.home() / ".local/share/whatsapp-listener/draft.jsonl"
DB = Path.home() / ".local/share/obsidian-rag/ragvec/telemetry.db"


def _approval_to_decision(kind: str, approval_kind: str | None) -> str | None:
    if kind == "approved":
        ak = (approval_kind or "si").strip().lower()
        if ak in ("si", "sí"):
            return "approved_si"
        if ak == "editar":
            return "approved_editar"
        return None
    if kind == "rejected":
        return "rejected"
    if kind == "expired":
        return "expired"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="contar sin escribir")
    args = p.parse_args()

    if not JSONL.exists():
        print(f"jsonl not found: {JSONL}", file=sys.stderr)
        return 2
    if not DB.exists():
        print(f"db not found: {DB}", file=sys.stderr)
        return 2

    # Indexar generated por draft_id (último gana — overwrites del bot
    # mantienen el último bot_draft visto).
    generated: dict[str, dict] = {}
    decision_events: list[dict] = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind", "")
            if kind == "generated":
                if ev.get("id"):
                    generated[ev["id"]] = ev
            elif kind in ("approved", "rejected", "expired"):
                decision_events.append(ev)

    print(f"jsonl: {len(generated)} generated · {len(decision_events)} decisions")

    # Agrupar decisions por (draft_id, decision) — la primera gana.
    seen: set[tuple[str, str]] = set()
    rows_to_insert: list[dict] = []
    skipped_no_gen = 0
    skipped_dup = 0
    skipped_invalid = 0
    for ev in decision_events:
        did = ev.get("id", "")
        if not did:
            skipped_invalid += 1
            continue
        decision = _approval_to_decision(ev["kind"], ev.get("approval_kind"))
        if not decision:
            skipped_invalid += 1
            continue
        if (did, decision) in seen:
            skipped_dup += 1
            continue
        seen.add((did, decision))
        gen = generated.get(did)
        if not gen:
            skipped_no_gen += 1
            continue

        # Construir original_msgs minimal — el JSONL `generated` event
        # tiene solo `original` (texto del incoming, single string), no
        # la lista completa. Lo metemos como un msg único.
        orig_text = str(gen.get("original", "")).strip()
        original_msgs = [{"id": "", "text": orig_text, "ts": gen.get("ts", "")}] if orig_text else []

        rows_to_insert.append({
            "ts": ev.get("ts", "")[:19],  # ISO8601 sin Z, formato igual que el resto
            "draft_id": did,
            "contact_jid": gen.get("jid", ev.get("jid", "")),
            "contact_name": gen.get("name", ""),
            "original_msgs_json": json.dumps(original_msgs, ensure_ascii=False),
            "bot_draft": str(gen.get("draft", "")),
            "decision": decision,
            "sent_text": ev.get("sent_text"),
            "extra_json": json.dumps({"backfilled_from": "draft.jsonl"}),
        })

    print(
        f"to insert: {len(rows_to_insert)} · "
        f"skipped: invalid={skipped_invalid} dup_key={skipped_dup} no_generated={skipped_no_gen}"
    )

    # Breakdown by decision
    by_dec: dict[str, int] = defaultdict(int)
    for r in rows_to_insert:
        by_dec[r["decision"]] += 1
    for k in sorted(by_dec):
        print(f"  {k}: {by_dec[k]}")

    if args.dry_run:
        print("(dry-run — no writes)")
        return 0

    if not rows_to_insert:
        print("nothing to insert")
        return 0

    # Pre-check: existing keys en la tabla, no duplicar.
    conn = sqlite3.connect(DB, isolation_level=None)
    try:
        existing: set[tuple[str, str]] = set()
        for did, dec in conn.execute(
            "SELECT draft_id, decision FROM rag_draft_decisions"
        ):
            existing.add((did, dec))

        new_rows = [r for r in rows_to_insert if (r["draft_id"], r["decision"]) not in existing]
        print(f"existing in DB: {len(existing)} · actually inserting: {len(new_rows)}")

        if not new_rows:
            print("all already in DB — no-op")
            return 0

        conn.execute("BEGIN")
        for r in new_rows:
            conn.execute(
                """
                INSERT INTO rag_draft_decisions
                  (ts, draft_id, contact_jid, contact_name, original_msgs_json,
                   bot_draft, decision, sent_text, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["ts"], r["draft_id"], r["contact_jid"], r["contact_name"],
                    r["original_msgs_json"], r["bot_draft"], r["decision"],
                    r["sent_text"], r["extra_json"],
                ),
            )
        conn.execute("COMMIT")
        print(f"inserted {len(new_rows)} rows")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
