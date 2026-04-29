"""Fine-tune bge-reranker-v2-m3 on user feedback (GC#2.B/2.C, 2026-04-22).

Two modes are supported, dispatched via `--mode`:

  --mode lora  (DEFAULT, GC#2.C 2026-04-23)
    Trains a tiny PEFT/LoRA adapter (r=8, alpha=16, dropout=0.1) on top of
    the base bge-reranker-v2-m3. Output: a ~5 MB adapter dir at
    `~/.local/share/obsidian-rag/reranker_ft/`. Loaded at runtime when the
    operator sets `RAG_RERANKER_FT=1` (default OFF).
    Requires the `[finetune]` extra (`peft`, `transformers`, `datasets`,
    `accelerate`). Install with:
        uv tool install --reinstall --editable '.[finetune]'

  --mode full  (GC#2.B 2026-04-22, kept for backwards-compat)
    Full CrossEncoder fine-tune. Output: ~2 GB checkpoint dir at
    `~/.cache/obsidian-rag/reranker-ft-{ts}/`. Promoted via the
    `reranker-ft-current` symlink + `RAG_RERANKER_FT_PATH` env when the
    `rag eval` gate passes (no regression on singles+chains hit@5).

Pipeline (shared between both modes):
  1. Load positive pairs from `rag_feedback` (rating=+1) + `rag_behavior`
     (`event='positive_implicit'`) filtered by query field.
  2. Mine hard negatives:
     - Default: top-K retrieve() results NOT in the rated-positive turn
       (mode=full path).
     - With --pairs-from: pre-mined from impression history by
       `scripts/export_training_pairs.py`.
     - LoRA path: also samples from rag_behavior — chunks that appeared
       in top-5 (`event='impression'` with rank ≥3) on a query whose
       user-clicked path was different.
  3. Stratified 80/20 split by query for held-out validation.
  4. Train (LoRA adapter or full CrossEncoder fine-tune).
  5. Print held-out metrics: pos/neg margin, nDCG@5 (LoRA mode), pair-rank
     correlation.
  6. Optional: run `rag eval` and gate-promote (full mode) / report deltas
     vs baseline (LoRA mode).

Usage:
  # GC#2.C — LoRA fine-tune (default mode)
  uv run python scripts/finetune_reranker.py
  uv run python scripts/finetune_reranker.py --epochs 3 --lr 2e-5
  uv run python scripts/finetune_reranker.py --dry-run     # prepare data only

  # GC#2.B — full fine-tune (legacy)
  python scripts/finetune_reranker.py --mode full
  python scripts/finetune_reranker.py --mode full --no-eval

  # Alternative: consume pairs pre-exported by scripts/export_training_pairs.py
  # (2026-04-22). That miner uses rag_behavior events (copies, opens, dwells)
  # as additional positives — 10x more signal than rag_feedback alone once
  # the user has used the system for a few days. The JSONL shape is
  # {query, positive, negatives, source, ...} — see export_training_pairs.py.
  python scripts/export_training_pairs.py --days 30 -o /tmp/pairs.jsonl
  python scripts/finetune_reranker.py --pairs-from /tmp/pairs.jsonl

Caveats:
  - Uses `corrective_path` from `rag_feedback.extra_json` as clean single
    positive when available. Falls back to all turn paths when absent
    (noisy but usable).
  - Gated on `RAG_FINETUNE_MIN_CORRECTIVES` (default 20). Below this,
    aborts with exit 5 — the signal is too weak for meaningful fine-tune.
    NOTE: gate only applies to the default (SQL) source. When using
    --pairs-from, the user already curated the signal in the miner so the
    gate is bypassed (tuning the min-negatives flag there instead).
  - Hard negatives mined via retrieve() top-K, excluding the corrective_path
    when present. When using --pairs-from, negatives come pre-mined from
    impression history in the miner (no runtime retrieve needed).
  - The eval gate is strict: no regression on either singles or chains hit@5.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Run the script with the venv python — it imports rag (which needs every dep).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


CACHE_ROOT = Path.home() / ".cache" / "obsidian-rag"
# GC#2.C — LoRA adapter output dir. Picked under XDG data home (not cache)
# because the user feedback signal is not regeneratable; we don't want the
# OS to wipe a curated adapter when low on disk. Loader at runtime reads
# from the same path (`rag.RERANKER_FT_ADAPTER_DIR`).
LOCAL_FT_ROOT = Path.home() / ".local" / "share" / "obsidian-rag" / "reranker_ft"
MODEL_TAG_BASE = "BAAI/bge-reranker-v2-m3"
BASELINE_SINGLES_MIN = 0.0  # overwritten by actual baseline at runtime
BASELINE_CHAINS_MIN = 0.0

# GC#2.C — LoRA defaults. r=8 keeps the adapter ~5 MB; alpha=16 sets
# scaling factor 2.0; dropout=0.1 to dampen overfit on the small (~hundreds
# of pairs) dataset. Targets the attention Q/V projections only — the
# convention that empirically works for cross-encoder ranking heads.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ("query", "value")  # XLM-RoBERTa attention proj names


def _fetch_feedback_pairs() -> list[dict]:
    """Return positive feedback rows joined with their query paths.

    Row shape: {rating, turn_id, q, paths: [vault-rel], extra}
    Keeps only positives (rating=1) with non-empty paths. Negatives drop
    because the query-turn path column is blank when the retrieve returned
    nothing (exactly why the user rated it -1) — unusable as pairs.
    """
    rows: list[dict] = []
    with rag._ragvec_state_conn() as conn:
        cursor = conn.execute(
            """
            SELECT f.rating, f.turn_id, f.q, q.paths_json, q.scores_json,
                   json_extract(f.extra_json, '$.corrective_path') AS cp
            FROM rag_feedback f
            LEFT JOIN rag_queries q ON json_extract(q.extra_json, '$.turn_id') = f.turn_id
            WHERE f.rating = 1
            ORDER BY f.ts DESC
            """
        )
        for rating, turn_id, q, paths_json, scores_json, cp in cursor.fetchall():
            if not q or not paths_json:
                continue
            try:
                paths = json.loads(paths_json)
            except Exception:
                continue
            try:
                scores = json.loads(scores_json) if scores_json else []
            except Exception:
                scores = []
            vault_paths = [p for p in paths if "://" not in p]
            if not vault_paths:
                continue
            corrective_path = cp.strip() if isinstance(cp, str) and cp.strip() else None
            rows.append({
                "rating": int(rating),
                "turn_id": turn_id or "",
                "q": q,
                "paths": vault_paths,
                "scores": scores,
                "corrective_path": corrective_path,
            })
    return rows


def _path_to_doc(path: str, vault_root: Path) -> str | None:
    """Read the first chunk-sized slice of the vault file as the document
    text for the cross-encoder input. 800 chars matches MAX_CHUNK so the
    reranker sees something representative of the real scoring target.
    """
    try:
        full = (vault_root / path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return full[:800] if full else None


def _mine_hard_negatives(
    query: str,
    positive_paths: set[str],
    col,
    *,
    k_pool: int = 10,
) -> list[str]:
    """Run retrieve() on the query and return up to k paths NOT in
    positive_paths. Those are "the system surfaced them but the user picked
    a different one" → hard negatives.
    """
    try:
        result = rag.retrieve(col, query, k=k_pool, folder=None,
                              multi_query=False, auto_filter=False)
    except Exception as exc:
        print(f"  [warn] retrieve failed for {query!r}: {exc}", file=sys.stderr)
        return []
    candidates = [
        m.get("file", "") for m in (result.get("metas") or [])
        if m.get("file") and "://" not in m.get("file", "")
    ]
    return [p for p in candidates if p not in positive_paths]


def _build_training_pairs(
    feedback_rows: list[dict],
    col,
    vault_root: Path,
    *,
    hard_neg_k: int = 5,
) -> list[dict]:
    """Materialise {text1, text2, label} triples from the feedback rows.

    text1: user question
    text2: document excerpt (first 800 chars of the path)
    label: 1.0 (positive) or 0.0 (mined hard negative)

    Branching on `corrective_path`:
      - present: that single path is the ONLY positive (label=1.0). Hard negs
        mined from retrieve() top-K excluding the corrective_path.
      - absent: every path in the rated-positive turn is a positive (noisy
        fallback). Hard negs exclude all turn paths.
    """
    pairs: list[dict] = []
    seen = 0
    skipped_unreadable = 0
    for row in feedback_rows:
        seen += 1
        corrective = row.get("corrective_path")
        if corrective:
            positives = [corrective]
            exclude = {corrective}
        else:
            positives = list(row["paths"])
            exclude = set(row["paths"])
        for p in positives:
            doc = _path_to_doc(p, vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 1.0})
        hard_negs = _mine_hard_negatives(row["q"], exclude, col)[:hard_neg_k]
        for p in hard_negs:
            doc = _path_to_doc(p, vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 0.0})
    print(f"  Built {len(pairs)} pairs from {seen} feedback rows "
          f"({skipped_unreadable} skipped unreadable).", file=sys.stderr)
    return pairs


def _split_train_val(pairs: list[dict], val_frac: float = 0.2, seed: int = 42):
    """Stratified split by query to avoid leaking same-query examples across
    train/val.
    """
    import random
    rng = random.Random(seed)
    # Group by query
    by_q: dict[str, list[dict]] = {}
    for p in pairs:
        by_q.setdefault(p["text1"], []).append(p)
    queries = sorted(by_q.keys())
    rng.shuffle(queries)
    n_val = max(1, int(len(queries) * val_frac))
    val_queries = set(queries[:n_val])
    train, val = [], []
    for q, ps in by_q.items():
        (val if q in val_queries else train).extend(ps)
    return train, val


def _train_crossencoder(
    train_pairs: list[dict],
    val_pairs: list[dict],
    *,
    out_dir: Path,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 8,
):
    """Fine-tune using sentence-transformers' CrossEncoder API.

    Saves the fine-tuned model to out_dir. Returns the final val loss
    approximation (BCE-style) and pairs of (pred, label) on val.
    """
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from torch.utils.data import DataLoader

    # Device: MPS is faster but the system often has ollama keep_alive=-1
    # pinning command-r / qwen2.5 (~30 GiB), leaving insufficient headroom
    # for the reranker's gradients during training. CPU fallback is 3-5×
    # slower but reliable. Override with RAG_FT_DEVICE=mps if you've
    # unloaded ollama first.
    device = os.environ.get("RAG_FT_DEVICE", "cpu").lower()
    print(f"  Loading base model {MODEL_TAG_BASE} on device={device} …",
          file=sys.stderr)
    model = CrossEncoder(MODEL_TAG_BASE, num_labels=1, device=device)

    # sentence-transformers 5.x API: CrossEncoder.fit() with a HF Dataset.
    from datasets import Dataset
    train_ds = Dataset.from_list(train_pairs)
    val_ds = Dataset.from_list(val_pairs) if val_pairs else None
    loss_fn = BinaryCrossEntropyLoss(model=model)

    args_kwargs = dict(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        seed=42,
        # MPS-friendly: no fp16 (known score-collapse bug with bge-reranker
        # on MPS, see CLAUDE.md §Model stack).
        fp16=False,
        bf16=False,
        report_to="none",
        # Force CPU when RAG_FT_DEVICE=cpu so accelerate doesn't grab MPS
        # on its own — the 30 GiB of ollama-pinned memory makes MPS OOM-prone
        # for gradient-accumulation during training.
        use_cpu=(device == "cpu"),
    )
    try:
        from sentence_transformers import CrossEncoderTrainer, CrossEncoderTrainingArguments
        training_args = CrossEncoderTrainingArguments(**args_kwargs)
    except ImportError:
        # Older API fallback — pre sentence-transformers 3.4 had no dedicated
        # CrossEncoder trainer; use model.fit() directly.
        print("  Using CrossEncoder.fit() legacy API", file=sys.stderr)
        examples = [
            {"texts": [p["text1"], p["text2"]], "label": p["label"]}
            for p in train_pairs
        ]
        from sentence_transformers import InputExample
        train_samples = [
            InputExample(texts=ex["texts"], label=ex["label"]) for ex in examples
        ]
        train_loader = DataLoader(train_samples, shuffle=True,
                                  batch_size=batch_size)
        model.fit(
            train_dataloader=train_loader,
            epochs=epochs,
            optimizer_params={"lr": lr},
            output_path=str(out_dir),
            save_best_model=False,
            use_amp=False,
        )
        return _predict_val(model, val_pairs)

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss_fn,
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    return _predict_val(model, val_pairs)


def _predict_val(model, val_pairs):
    if not val_pairs:
        return []
    preds = model.predict([(p["text1"], p["text2"]) for p in val_pairs])
    return list(zip(preds.tolist() if hasattr(preds, "tolist") else list(preds),
                    [p["label"] for p in val_pairs]))


def _run_eval(model_path: str | None = None) -> dict | None:
    """Invoke `rag eval` in a subprocess and parse the Singles/Chains lines.

    When model_path is given, set RAG_RERANKER_FT_PATH so the reranker loader
    picks up the fine-tuned model. When None, runs the baseline.
    """
    env = os.environ.copy()
    env["RAG_EXPLORE"] = "0"  # eval command already strips this, belt+suspenders
    env["RAG_CACHE_ENABLED"] = "0"  # eval should miss the response cache
    if model_path:
        env["RAG_RERANKER_FT_PATH"] = model_path
    else:
        env.pop("RAG_RERANKER_FT_PATH", None)
    try:
        proc = subprocess.run(
            ["rag", "eval"],
            env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        print(f"  [eval] rc={proc.returncode} stderr={proc.stderr[:200]}",
              file=sys.stderr)
        return None

    import re
    out = proc.stdout
    singles = re.search(r"Singles:\s+hit@5\s+([\d.]+)%.+?MRR\s+([\d.]+)", out)
    chains = re.search(r"Chains:\s+hit@5\s+([\d.]+)%.+?MRR\s+([\d.]+)", out)
    return {
        "singles_hit5": float(singles.group(1)) / 100 if singles else None,
        "singles_mrr": float(singles.group(2)) if singles else None,
        "chains_hit5": float(chains.group(1)) / 100 if chains else None,
        "chains_mrr": float(chains.group(2)) if chains else None,
    }


def _print_val_summary(val_predictions):
    if not val_predictions:
        print("  No validation pairs.", file=sys.stderr)
        return
    import statistics
    pos_preds = [p for p, l in val_predictions if l == 1.0]
    neg_preds = [p for p, l in val_predictions if l == 0.0]
    print("  Validation:", file=sys.stderr)
    print(f"    positives n={len(pos_preds)} mean={statistics.fmean(pos_preds):.3f}"
          if pos_preds else "    positives n=0", file=sys.stderr)
    print(f"    negatives n={len(neg_preds)} mean={statistics.fmean(neg_preds):.3f}"
          if neg_preds else "    negatives n=0", file=sys.stderr)
    if pos_preds and neg_preds:
        margin = statistics.fmean(pos_preds) - statistics.fmean(neg_preds)
        print(f"    margin={margin:+.3f}  "
              f"(positive > negative by this much; should be > 0)", file=sys.stderr)


def _load_pairs_from_jsonl(
    jsonl_path: Path, vault_root: Path, hard_neg_k: int,
) -> list[dict]:
    """Convert export_training_pairs.py-style JSONL rows into the
    `{text1, text2, label}` format the CrossEncoder trainer consumes.

    Input shape (one per line, matches scripts/export_training_pairs.py):
      {"query": str, "positive": "<vault-rel>", "negatives": [str, ...],
       "source": str, "turn_id": str|None, "ts": str}

    Output per row yields up to 1 + hard_neg_k training pairs:
      - 1 positive: {text1: query, text2: first-800-chars-of-positive, label: 1.0}
      - up to hard_neg_k negatives: same text1, each negative's doc, label 0.0

    Paths whose vault file is unreadable (gone, permissions) are skipped
    with a tally. Counting skips explicitly so the operator sees how much
    of the export landed in the training set.
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"pairs file not found: {jsonl_path}")

    pairs: list[dict] = []
    rows_seen = 0
    skipped_unreadable = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [skip] line {line_idx}: bad JSON ({exc})",
                      file=sys.stderr)
                continue
            query = (row.get("query") or "").strip()
            positive = (row.get("positive") or "").strip()
            if not query or not positive:
                continue
            rows_seen += 1
            pos_doc = _path_to_doc(positive, vault_root)
            if pos_doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": query, "text2": pos_doc, "label": 1.0})
            # Cap negatives per row at hard_neg_k so the ratio stays sane.
            for neg_path in (row.get("negatives") or [])[:hard_neg_k]:
                neg_doc = _path_to_doc(str(neg_path).strip(), vault_root)
                if neg_doc is None:
                    skipped_unreadable += 1
                    continue
                pairs.append({"text1": query, "text2": neg_doc, "label": 0.0})
    print(f"  JSONL rows seen: {rows_seen} "
          f"(skipped unreadable paths: {skipped_unreadable})",
          file=sys.stderr)
    return pairs


# ── GC#2.C — LoRA-specific data + training ────────────────────────────────


def _fetch_behavior_positive_pairs(window_days: int = 90) -> list[dict]:
    """Return implicit positives from `rag_behavior` events.

    Schema-aware: rag_behavior has a `query` field (added 2026-04-22 as
    part of GC#2.A "telemetry honesty"). When present + the event is a
    positive signal (`positive_implicit`/`open`/`save`/`kept`) we have a
    clean (query, path) tuple ready to feed the trainer with label=1.0.

    Defensive: rows without query are skipped (they predate the schema
    change). The `vault path` filter is the same as the feedback path —
    pseudo-URI sources (whatsapp://, gmail://, ...) require their own
    `_path_to_doc` reader, not implemented in this script yet.

    Used only in `--mode lora` — the legacy `--mode full` path keeps the
    feedback-only signal source unchanged for parity with prior runs.
    """
    rows: list[dict] = []
    try:
        with rag._ragvec_state_conn() as conn:
            cursor = conn.execute(
                """
                SELECT query, path
                FROM rag_behavior
                WHERE event = 'positive_implicit'
                  AND query IS NOT NULL AND query != ''
                  AND path  IS NOT NULL AND path != ''
                  AND ts >= datetime('now', 'localtime', ?)
                ORDER BY ts DESC
                """,
                (f"-{int(window_days)} days",),
            )
            for q, p in cursor.fetchall():
                if "://" in p:
                    # Cross-source pseudo-URI — not readable as vault file.
                    # Future: extend `_path_to_doc` to fetch from the
                    # source-specific store. For now skip silently.
                    continue
                rows.append({"q": q, "path": p})
    except Exception as exc:
        print(f"  [warn] _fetch_behavior_positive_pairs: {exc}", file=sys.stderr)
    return rows


def _fetch_behavior_hard_negatives(window_days: int = 90) -> list[dict]:
    """Mine hard negatives from rag_behavior impressions.

    Heuristic: when the user issued a query and `event='impression'` shows
    a top-5 with rank ≥3, but a subsequent `event='open'` (close in time,
    same query) hit a different path → the impression at rank ≥3 is a
    hard negative for that query. The user saw it, the system thought it
    was relevant enough to surface, but the user didn't pick it.

    Output rows: {q, path, opened_path}. The opened_path is the positive
    that anchored this triple — it's used downstream to dedup against
    positives so the same (q, path) doesn't end up in both buckets.

    Conservative: requires both impression AND open events to coexist for
    the same (query, ts-window). Without the open event we can't
    distinguish "didn't pick" from "didn't see / abandoned the query".
    """
    rows: list[dict] = []
    try:
        with rag._ragvec_state_conn() as conn:
            # Step 1: opens with a query field (the positive anchor).
            opens = conn.execute(
                """
                SELECT ts, query, path
                FROM rag_behavior
                WHERE event = 'open'
                  AND query IS NOT NULL AND query != ''
                  AND path  IS NOT NULL AND path != ''
                  AND ts >= datetime('now', 'localtime', ?)
                """,
                (f"-{int(window_days)} days",),
            ).fetchall()
            # Step 2: for each open, find impressions w/ same query + rank≥3
            # within ±10 minutes. SQLite's `datetime()` window math is
            # tolerable here because we're scanning hundreds of rows max.
            for ts, q, opened in opens:
                impressions = conn.execute(
                    """
                    SELECT path, rank
                    FROM rag_behavior
                    WHERE event = 'impression'
                      AND query = ?
                      AND rank >= 3
                      AND path  IS NOT NULL AND path != ''
                      AND path != ?
                      AND ABS(julianday(ts) - julianday(?)) * 86400 < 600
                    """,
                    (q, opened, ts),
                ).fetchall()
                for path, _rank in impressions:
                    if "://" in path:
                        continue
                    rows.append({"q": q, "path": path, "opened_path": opened})
    except Exception as exc:
        print(f"  [warn] _fetch_behavior_hard_negatives: {exc}", file=sys.stderr)
    return rows


def _build_lora_training_pairs(
    feedback_rows: list[dict],
    vault_root: Path,
    *,
    include_behavior: bool = True,
) -> list[dict]:
    """Materialise {text1, text2, label} triples for the LoRA path.

    Sources, in priority order:
      1. rag_feedback rating=+1 with `corrective_path` (cleanest signal).
      2. rag_feedback rating=+1 without corrective (noisy fallback — every
         turn path is treated as positive).
      3. rag_behavior `positive_implicit` events with a query field (when
         `include_behavior=True`).
    Hard negatives are ALSO mined from rag_behavior — impressions at
    rank ≥3 with no subsequent open. This is the GC#2.C-specific signal
    requested in the plan: chunks the system surfaced but the user did
    NOT click. The full-FT path (mode=full) keeps the runtime retrieve()
    miner instead.

    Dedup contract: a (query, path) tuple appears at most once across the
    whole output, with positive winning if the same (q, path) shows up in
    both buckets (avoids the model learning contradictory labels).
    """
    pairs: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    skipped_unreadable = 0

    # ── 1+2. rag_feedback positives ──
    for row in feedback_rows:
        corrective = row.get("corrective_path")
        positives = [corrective] if corrective else list(row.get("paths", []))
        for p in positives:
            if not p or "://" in p:
                continue
            key = (row["q"], p)
            if key in seen_keys:
                continue
            doc = _path_to_doc(p, vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 1.0})
            seen_keys.add(key)

    # ── 3. rag_behavior positive_implicit ──
    if include_behavior:
        for row in _fetch_behavior_positive_pairs():
            key = (row["q"], row["path"])
            if key in seen_keys:
                continue
            doc = _path_to_doc(row["path"], vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 1.0})
            seen_keys.add(key)

    # ── Hard negatives from impressions ──
    if include_behavior:
        for row in _fetch_behavior_hard_negatives():
            key = (row["q"], row["path"])
            if key in seen_keys:
                continue  # positive wins
            doc = _path_to_doc(row["path"], vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 0.0})
            seen_keys.add(key)

    print(f"  Built {len(pairs)} LoRA training pairs "
          f"({skipped_unreadable} skipped unreadable).", file=sys.stderr)
    return pairs


def _ndcg_at_k(scores_with_labels: list[tuple[float, float]], k: int = 5) -> float:
    """Compute nDCG@k for a single ranked list.

    `scores_with_labels` is a list of (predicted_score, ground_truth_label)
    tuples — labels are binary 0.0/1.0 in our setup. Returns 0.0 when the
    list is empty or has no positives (degenerate but expected for some
    held-out queries with only negatives).
    """
    import math
    if not scores_with_labels:
        return 0.0
    ranked = sorted(scores_with_labels, key=lambda x: x[0], reverse=True)[:k]
    dcg = sum(
        (rel / math.log2(i + 2))  # i+2 because positions are 1-indexed (log2(1+1)=1)
        for i, (_, rel) in enumerate(ranked)
    )
    ideal = sorted([rel for _, rel in scores_with_labels], reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _ndcg_per_query(
    val_pairs: list[dict],
    predictor,
    k: int = 5,
) -> dict[str, float]:
    """Compute mean nDCG@k across queries in val_pairs.

    `predictor` is a callable `[(text1, text2), ...] -> [score, ...]`.
    Returns `{n_queries, mean_ndcg, n_no_pos}` where `n_no_pos` counts
    queries that ended up without any positive in held-out (they get
    nDCG=0 in the mean which is expected — they're degenerate tests).
    """
    by_q: dict[str, list[tuple[float, float]]] = {}
    if not val_pairs:
        return {"n_queries": 0, "mean_ndcg": 0.0, "n_no_pos": 0}
    inputs = [(p["text1"], p["text2"]) for p in val_pairs]
    preds = predictor(inputs)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    for pair, score in zip(val_pairs, preds):
        by_q.setdefault(pair["text1"], []).append((float(score), float(pair["label"])))
    ndcgs: list[float] = []
    n_no_pos = 0
    for _q, items in by_q.items():
        if not any(lbl == 1.0 for _s, lbl in items):
            n_no_pos += 1
        ndcgs.append(_ndcg_at_k(items, k=k))
    return {
        "n_queries": len(by_q),
        "mean_ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "n_no_pos": n_no_pos,
    }


def _pair_ranking_correlation(val_predictions: list[tuple[float, float]]) -> float | None:
    """Spearman-ish ranking correlation between predicted scores and labels.

    Since labels are binary, "correlation" reduces to: do positives tend
    to outrank negatives? We compute the AUC-style stat: probability that
    a random positive scores higher than a random negative. 0.5 is chance,
    1.0 is perfect, <0.5 is anti-correlated (bad).

    Returns None if either bucket is empty (no signal).
    """
    pos = [s for s, lbl in val_predictions if lbl == 1.0]
    neg = [s for s, lbl in val_predictions if lbl == 0.0]
    if not pos or not neg:
        return None
    wins = 0
    ties = 0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                ties += 1
    total = len(pos) * len(neg)
    return (wins + 0.5 * ties) / total if total else None


def _train_lora_adapter(
    train_pairs: list[dict],
    val_pairs: list[dict],
    *,
    out_dir: Path,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
) -> dict:
    """Fine-tune a PEFT/LoRA adapter on top of bge-reranker-v2-m3.

    Uses transformers + peft directly (vs sentence-transformers'
    CrossEncoderTrainer): we need fine-grained control over the LoRA
    config and the adapter save path. The trainer wraps an
    `AutoModelForSequenceClassification` (num_labels=1, regression head)
    so it matches the cross-encoder's contract; the LoRA adapter wraps
    Q/V projections of the XLM-RoBERTa backbone.

    Output: writes a PEFT adapter dir to `out_dir`. Returns a metrics
    dict for the held-out val set:
      - `pos_mean`, `neg_mean`: predicted score means per bucket
      - `margin`: pos_mean - neg_mean (should be > 0)
      - `auc`: pair-ranking probability (0.5 chance, >0.5 better)
      - `ndcg_at_5`: mean nDCG@5 across val queries
      - `ndcg_at_5_baseline`: same metric with the BASE model (no LoRA)
        for an apples-to-apples delta the operator can read at a glance.
    """
    try:
        import torch  # noqa: F401
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as exc:
        print(
            "  [error] missing dep for LoRA mode: "
            f"{exc}\n"
            "  Install with: uv tool install --reinstall --editable '.[finetune]'",
            file=sys.stderr,
        )
        sys.exit(6)

    device = os.environ.get("RAG_FT_DEVICE", "cpu").lower()
    print(
        f"  Loading {MODEL_TAG_BASE} (transformers={transformers.__version__}) on {device} …",
        file=sys.stderr,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TAG_BASE)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_TAG_BASE, num_labels=1, problem_type="regression",
    )

    # Reference predictions BEFORE wrapping in PEFT — gives the operator
    # the "without fine-tune" reading on the same val pairs, eliminating
    # noise from val-split selection.
    print("  Computing baseline (pre-LoRA) val metrics …", file=sys.stderr)
    base_preds = _predict_with_hf_model(base_model, tokenizer, val_pairs)
    base_metrics = _summarise_predictions(base_preds, val_pairs)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    try:
        # peft >=0.5 exposes a printable summary of trainable parameter
        # counts. Best-effort — older versions silently skip.
        model.print_trainable_parameters()
    except Exception:
        pass

    # ── Tokenization ──
    def _tok(batch):
        enc = tokenizer(
            batch["text1"], batch["text2"],
            truncation=True, max_length=512, padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    train_ds = Dataset.from_list(train_pairs).map(_tok, batched=True)
    val_ds = Dataset.from_list(val_pairs).map(_tok, batched=True) if val_pairs else None
    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in keep_cols]
    )
    if val_ds is not None:
        val_ds = val_ds.remove_columns(
            [c for c in val_ds.column_names if c not in keep_cols]
        )

    args = TrainingArguments(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=10,
        seed=42,
        fp16=False,
        bf16=False,
        report_to=[],
        use_cpu=(device == "cpu"),
    )

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    # PEFT save: writes adapter_config.json + adapter_model.safetensors.
    # NOT the full base model — we want the ~5 MB adapter, not the 2 GB
    # checkpoint.
    model.save_pretrained(str(out_dir))
    # Also drop the tokenizer in the adapter dir so the runtime loader
    # has everything it needs in one place. Cheap (~10 MB) compared to
    # the gain in operator simplicity.
    try:
        tokenizer.save_pretrained(str(out_dir))
    except Exception as exc:
        print(f"  [warn] tokenizer.save_pretrained failed: {exc}", file=sys.stderr)

    # ── Held-out val metrics POST-LoRA ──
    ft_preds = _predict_with_hf_model(model, tokenizer, val_pairs)
    ft_metrics = _summarise_predictions(ft_preds, val_pairs)

    return {
        "baseline": base_metrics,
        "fine_tuned": ft_metrics,
        "delta_ndcg_at_5": ft_metrics.get("ndcg_at_5", 0.0)
                           - base_metrics.get("ndcg_at_5", 0.0),
        "delta_margin": ft_metrics.get("margin", 0.0)
                        - base_metrics.get("margin", 0.0),
    }


def _predict_with_hf_model(model, tokenizer, val_pairs: list[dict]) -> list[float]:
    """Score `val_pairs` with a HF AutoModel head. Returns plain Python floats.

    Conservative batching (batch=8) so we don't OOM on CPU; for typical
    held-out sizes (<200 pairs) this is well under a second on M-series.
    """
    import torch
    if not val_pairs:
        return []
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(val_pairs), 8):
            batch = val_pairs[i:i + 8]
            enc = tokenizer(
                [p["text1"] for p in batch],
                [p["text2"] for p in batch],
                truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            )
            out = model(**enc)
            # Sigmoid because the cross-encoder output is a logit; we
            # want the [0,1] score that matches the loader's contract.
            sig = torch.sigmoid(out.logits.squeeze(-1))
            scores.extend(sig.cpu().tolist())
    return scores


def _summarise_predictions(preds: list[float], val_pairs: list[dict]) -> dict:
    """Aggregate held-out predictions into operator-readable metrics."""
    if not preds or not val_pairs:
        return {
            "n_pairs": 0, "n_pos": 0, "n_neg": 0,
            "pos_mean": 0.0, "neg_mean": 0.0, "margin": 0.0,
            "auc": None, "ndcg_at_5": 0.0, "n_queries": 0,
        }
    import statistics
    pairs_with_preds = list(zip(preds, [p["label"] for p in val_pairs]))
    pos_preds = [s for s, lbl in pairs_with_preds if lbl == 1.0]
    neg_preds = [s for s, lbl in pairs_with_preds if lbl == 0.0]
    pos_mean = statistics.fmean(pos_preds) if pos_preds else 0.0
    neg_mean = statistics.fmean(neg_preds) if neg_preds else 0.0
    auc = _pair_ranking_correlation(pairs_with_preds)
    # nDCG: group by query
    by_q: dict[str, list[tuple[float, float]]] = {}
    for pair, score in zip(val_pairs, preds):
        by_q.setdefault(pair["text1"], []).append((score, float(pair["label"])))
    ndcgs = [_ndcg_at_k(items, k=5) for items in by_q.values()]
    return {
        "n_pairs": len(val_pairs),
        "n_pos": len(pos_preds),
        "n_neg": len(neg_preds),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "margin": pos_mean - neg_mean,
        "auc": auc,
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "n_queries": len(by_q),
    }


def _print_lora_report(metrics: dict) -> None:
    """Pretty-print the before/after metrics block for the LoRA mode."""
    base = metrics["baseline"]
    ft = metrics["fine_tuned"]
    print("  ── Held-out validation (before vs after LoRA) ──", file=sys.stderr)
    print(
        f"    n_queries={ft['n_queries']}  n_pairs={ft['n_pairs']}  "
        f"(pos={ft['n_pos']} neg={ft['n_neg']})",
        file=sys.stderr,
    )
    print(
        f"    nDCG@5      base={base['ndcg_at_5']:.4f}  "
        f"ft={ft['ndcg_at_5']:.4f}  Δ={metrics['delta_ndcg_at_5']:+.4f}",
        file=sys.stderr,
    )
    print(
        f"    pos mean    base={base['pos_mean']:.3f}  "
        f"ft={ft['pos_mean']:.3f}",
        file=sys.stderr,
    )
    print(
        f"    neg mean    base={base['neg_mean']:.3f}  "
        f"ft={ft['neg_mean']:.3f}",
        file=sys.stderr,
    )
    print(
        f"    margin      base={base['margin']:+.3f}  "
        f"ft={ft['margin']:+.3f}  Δ={metrics['delta_margin']:+.3f}",
        file=sys.stderr,
    )
    if base["auc"] is not None and ft["auc"] is not None:
        print(
            f"    pair-rank   base AUC={base['auc']:.3f}  "
            f"ft AUC={ft['auc']:.3f}  (>0.5 = pos>neg)",
            file=sys.stderr,
        )


def _run_lora_mode(args) -> None:
    """Top-level dispatch for the LoRA fine-tune (GC#2.C).

    Steps:
      1. Build pairs from rag_feedback + rag_behavior.
      2. Stratified 80/20 split (held-out is for OUR metrics, not the
         repo's `rag eval` — that gate runs separately downstream).
      3. Train LoRA adapter, save under
         `~/.local/share/obsidian-rag/reranker_ft/`.
      4. Print before/after nDCG@5 + ranking correlation.

    Conservative gates:
      - <10 positives → abort (signal too weak for a meaningful fine-tune).
        Lower than the full-FT path because LoRA tolerates small data
        better.
      - --dry-run → exit after pair build, no training.
    """
    print("== GC#2.C — LoRA fine-tune of bge-reranker-v2-m3 ==", file=sys.stderr)
    vault_root = rag._resolve_vault_path()

    if args.pairs_from:
        # Same JSONL ingestion as the full-FT path. The labels are
        # already 1.0/0.0 — no rebuild against rag_behavior.
        jsonl_path = Path(args.pairs_from).expanduser().resolve()
        print(f"  Loading pre-mined pairs from: {jsonl_path}", file=sys.stderr)
        pairs = _load_pairs_from_jsonl(
            jsonl_path, vault_root, hard_neg_k=args.hard_neg_k,
        )
    else:
        rows = _fetch_feedback_pairs()
        print(f"  Feedback positives (rating=+1): {len(rows)}", file=sys.stderr)
        pairs = _build_lora_training_pairs(
            rows, vault_root, include_behavior=True,
        )

    pos = sum(1 for p in pairs if p["label"] == 1.0)
    neg = sum(1 for p in pairs if p["label"] == 0.0)
    print(
        f"  Training pairs: total={len(pairs)} pos={pos} neg={neg}",
        file=sys.stderr,
    )
    if pos < 10:
        print(
            f"  Not enough positives ({pos} < 10). Need more rag_feedback "
            "rating=+1 turns or rag_behavior positive_implicit events with a "
            "query field. Aborting.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dry_run:
        print("  [dry-run] exiting before training.", file=sys.stderr)
        return

    train_pairs, val_pairs = _split_train_val(pairs, val_frac=0.2)
    print(
        f"  Split: train={len(train_pairs)} val={len(val_pairs)}",
        file=sys.stderr,
    )

    LOCAL_FT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"  Training → {LOCAL_FT_ROOT}", file=sys.stderr)
    t0 = time.time()
    metrics = _train_lora_adapter(
        train_pairs, val_pairs,
        out_dir=LOCAL_FT_ROOT,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    print(f"  Trained in {time.time() - t0:.1f}s", file=sys.stderr)
    _print_lora_report(metrics)

    # Persist a JSON next to the adapter so `rag stats` / a future
    # `rag feedback status` can surface the last train run without
    # re-loading the model. Best-effort.
    try:
        meta = {
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "metrics": metrics,
        }
        (LOCAL_FT_ROOT / "ft_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
    except Exception as exc:
        print(f"  [warn] writing ft_meta.json failed: {exc}", file=sys.stderr)

    print(
        "  Adapter saved. To activate: `export RAG_RERANKER_FT=1` and "
        "restart any long-running rag processes (web, serve).",
        file=sys.stderr,
    )


# ── End of GC#2.C LoRA helpers ────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("lora", "full"),
        default="lora",
        help=("Training mode. `lora` (default, GC#2.C 2026-04-23) trains a "
              "PEFT adapter saved to ~/.local/share/obsidian-rag/reranker_ft/. "
              "`full` (GC#2.B) does a full CrossEncoder fine-tune saved to "
              "~/.cache/obsidian-rag/reranker-ft-{ts}/ with eval-gate promotion."),
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Build pairs and print counts; no training or eval.")
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip the `rag eval` gate (debug only). "
                         "Mode=lora ignores this — eval gate is mode=full only.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hard-neg-k", type=int, default=5)
    ap.add_argument("--pairs-from", type=str, default=None,
                    metavar="JSONL",
                    help=("Consume pre-exported training pairs from a JSONL "
                          "file (produced by scripts/export_training_pairs.py). "
                          "Skips the default SQL-based _fetch_feedback_pairs() + "
                          "runtime hard-neg mining; the JSONL already carries "
                          "history-based hard negs."))
    args = ap.parse_args()

    # Dispatch to the LoRA path early — it has its own pre-flight + data
    # build (LoRA-specific positives from rag_behavior + impression-based
    # hard negs). The legacy full-FT path stays inline below.
    if args.mode == "lora":
        _run_lora_mode(args)
        return

    print("== GC#2.B — full fine-tune reranker ==", file=sys.stderr)

    col = rag.get_db()
    vault_root = rag._resolve_vault_path()

    # Two code paths depending on the data source:
    #   - --pairs-from <jsonl>: skip the SQL fetch + runtime retrieve hard-neg
    #     mining. The miner has already done that work offline with richer
    #     signal (rag_behavior events + impression history).
    #   - default: the original rag_feedback-only path with runtime retrieve.
    if args.pairs_from:
        jsonl_path = Path(args.pairs_from).expanduser().resolve()
        print(f"  Loading pre-mined pairs from: {jsonl_path}", file=sys.stderr)
        pairs = _load_pairs_from_jsonl(
            jsonl_path, vault_root, hard_neg_k=args.hard_neg_k,
        )
        if len(pairs) < 20:
            # Lower bar than the SQL path's 10-row gate because each JSONL
            # row already yields 1 + up to N negatives, so 20 training
            # pairs = ~4-6 queries worth — still too low, abort.
            print(f"  Not enough training pairs from JSONL ({len(pairs)} < 20). "
                  f"Re-export with --days wider or --min-negatives 0. Aborting.",
                  file=sys.stderr)
            sys.exit(2)
    else:
        # 1. Feedback rows
        rows = _fetch_feedback_pairs()
        print(f"  Feedback positives (with paths): {len(rows)}", file=sys.stderr)
        if len(rows) < 10:
            print("  Not enough feedback data — need ≥10 rated-positive turns with "
                  "paths. Aborting.", file=sys.stderr)
            sys.exit(2)

        min_correctives = int(os.environ.get("RAG_FINETUNE_MIN_CORRECTIVES", "20"))
        rows_with_corrective = [r for r in rows if r.get("corrective_path")]
        print(f"  Rows with corrective_path: {len(rows_with_corrective)} "
              f"(min required: {min_correctives})", file=sys.stderr)
        if len(rows_with_corrective) < min_correctives:
            print(f"  Not enough corrective_path signal — need ≥{min_correctives}, "
                  f"got {len(rows_with_corrective)}.", file=sys.stderr)
            print("  Use `rag chat` + thumbs-down on failed turns to generate more. "
                  "Aborting.", file=sys.stderr)
            sys.exit(5)

        # 2. Build pairs
        pairs = _build_training_pairs(rows, col, vault_root, hard_neg_k=args.hard_neg_k)

    pos = sum(1 for p in pairs if p["label"] == 1.0)
    neg = sum(1 for p in pairs if p["label"] == 0.0)
    print(f"  Training pairs: total={len(pairs)} pos={pos} neg={neg}",
          file=sys.stderr)

    if args.dry_run:
        print("  [dry-run] exiting before training.", file=sys.stderr)
        return

    train_pairs, val_pairs = _split_train_val(pairs, val_frac=0.2)
    print(f"  Split: train={len(train_pairs)} val={len(val_pairs)}",
          file=sys.stderr)

    # 3. Train
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = CACHE_ROOT / f"reranker-ft-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Training → {out_dir}", file=sys.stderr)
    t0 = time.time()
    val_preds = _train_crossencoder(
        train_pairs, val_pairs,
        out_dir=out_dir,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    print(f"  Trained in {time.time()-t0:.1f}s", file=sys.stderr)
    _print_val_summary(val_preds)

    if args.no_eval:
        print("  [--no-eval] skipping gate; model saved at "
              f"{out_dir} but NOT promoted.", file=sys.stderr)
        return

    # 4. Eval gate
    print("  Running baseline eval …", file=sys.stderr)
    base = _run_eval(None)
    if not base:
        print("  [gate] baseline eval failed; aborting.", file=sys.stderr)
        sys.exit(3)
    print(f"  Baseline: singles={base['singles_hit5']:.4f} "
          f"MRR={base['singles_mrr']:.3f}  "
          f"chains={base['chains_hit5']:.4f} MRR={base['chains_mrr']:.3f}",
          file=sys.stderr)

    print("  Running fine-tuned eval …", file=sys.stderr)
    ft = _run_eval(str(out_dir))
    if not ft:
        print("  [gate] fine-tuned eval failed; aborting.", file=sys.stderr)
        sys.exit(3)
    print(f"  Fine-tuned: singles={ft['singles_hit5']:.4f} "
          f"MRR={ft['singles_mrr']:.3f}  "
          f"chains={ft['chains_hit5']:.4f} MRR={ft['chains_mrr']:.3f}",
          file=sys.stderr)

    # Gate: no regression on either hit@5 metric (tolerate exact equality).
    singles_ok = ft["singles_hit5"] >= base["singles_hit5"]
    chains_ok = ft["chains_hit5"] >= base["chains_hit5"]
    if singles_ok and chains_ok:
        current = CACHE_ROOT / "reranker-ft-current"
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(out_dir.name)  # relative symlink
        print(f"  [gate] ✓ passed. Promoted {out_dir.name} → reranker-ft-current",
              file=sys.stderr)
        print(f"    Singles delta: {(ft['singles_hit5']-base['singles_hit5'])*100:+.1f}pp",
              file=sys.stderr)
        print(f"    Chains  delta: {(ft['chains_hit5']-base['chains_hit5'])*100:+.1f}pp",
              file=sys.stderr)
    else:
        print("  [gate] ✗ regression detected. NOT promoting.", file=sys.stderr)
        print(f"    singles_ok={singles_ok} chains_ok={chains_ok}", file=sys.stderr)
        print(f"    Model kept at {out_dir} for inspection — remove manually.",
              file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
