"""Fine-tune bge-reranker-v2-m3 on user feedback (GC#2.B, 2026-04-22).

Pipeline:
  1. Load feedback pairs from rag_feedback ⋈ rag_queries (via turn_id).
  2. Mine hard negatives from the corpus (top-K retrieve() results that weren't
     in the user-positively-rated turn).
  3. Fine-tune sentence-transformers/CrossEncoder on pairwise data.
  4. Save to ~/.cache/obsidian-rag/reranker-ft-{ts}/.
  5. Run `rag eval` with the fine-tuned model and the baseline. Promote the
     symlink `~/.cache/obsidian-rag/reranker-ft-current` ONLY if both singles
     hit@5 AND chains hit@5 beat the baseline by >= +0pp (no regression).

Usage:
  python scripts/finetune_reranker.py               # training + eval + gate
  python scripts/finetune_reranker.py --dry-run     # prepare data only
  python scripts/finetune_reranker.py --no-eval     # skip the gate (debug only)
  python scripts/finetune_reranker.py --epochs 3 --lr 2e-5

Caveats:
  - 55 positive pairs × 4 paths avg = ~220 noisy positive examples.
  - `corrective_path` would yield cleaner signals — current data has 0 rows
    with it. A rating=1 on a 5-chunk turn means "I liked the answer" not
    "chunk X was the golden one".
  - The gate is strict: no regression on either singles or chains hit@5.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Run the script with the venv python — it imports rag (which needs every dep).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


CACHE_ROOT = Path.home() / ".cache" / "obsidian-rag"
MODEL_TAG_BASE = "BAAI/bge-reranker-v2-m3"
BASELINE_SINGLES_MIN = 0.0  # overwritten by actual baseline at runtime
BASELINE_CHAINS_MIN = 0.0


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
            SELECT f.rating, f.turn_id, f.q, q.paths_json, q.scores_json
            FROM rag_feedback f
            LEFT JOIN rag_queries q ON json_extract(q.extra_json, '$.turn_id') = f.turn_id
            WHERE f.rating = 1
            ORDER BY f.ts DESC
            """
        )
        for rating, turn_id, q, paths_json, scores_json in cursor.fetchall():
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
            # Filter cross-source native ids — they aren't filesystem paths
            # and fetching their document text requires the retrieve pipeline;
            # keep only vault-style paths (no `://` scheme).
            vault_paths = [p for p in paths if "://" not in p]
            if not vault_paths:
                continue
            rows.append({
                "rating": int(rating),
                "turn_id": turn_id or "",
                "q": q,
                "paths": vault_paths,
                "scores": scores,
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
    """
    pairs: list[dict] = []
    seen = 0
    skipped_unreadable = 0
    for row in feedback_rows:
        seen += 1
        positive_paths = set(row["paths"])
        # Positives: each path in the rated-positive turn
        for p in positive_paths:
            doc = _path_to_doc(p, vault_root)
            if doc is None:
                skipped_unreadable += 1
                continue
            pairs.append({"text1": row["q"], "text2": doc, "label": 1.0})
        # Hard negatives: top-10 retrieved that AREN'T in positive_paths
        hard_negs = _mine_hard_negatives(row["q"], positive_paths, col)[:hard_neg_k]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Build pairs and print counts; no training or eval.")
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip the `rag eval` gate (debug only).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hard-neg-k", type=int, default=5)
    args = ap.parse_args()

    print("== GC#2.B — fine-tune reranker ==", file=sys.stderr)

    # 1. Feedback rows
    rows = _fetch_feedback_pairs()
    print(f"  Feedback positives (with paths): {len(rows)}", file=sys.stderr)
    if len(rows) < 10:
        print("  Not enough feedback data — need ≥10 rated-positive turns with "
              "paths. Aborting.", file=sys.stderr)
        sys.exit(2)

    # 2. Build pairs
    col = rag.get_db()
    vault_root = rag._resolve_vault_path()
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
        print(f"  [gate] ✗ regression detected. NOT promoting.", file=sys.stderr)
        print(f"    singles_ok={singles_ok} chains_ok={chains_ok}", file=sys.stderr)
        print(f"    Model kept at {out_dir} for inspection — remove manually.",
              file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
