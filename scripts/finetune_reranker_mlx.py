#!/usr/bin/env python3
"""Reranker LoRA fine-tune via [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

Fase 4 del plan MLX-full-migration (Ola 10, 2026-05-07). Migra el flow
`peft + transformers + bge-reranker-v2-m3` de
[`scripts/finetune_reranker.py`](scripts/finetune_reranker.py) al runtime
MLX nativo, usando Qwen3-Reranker como base prompt-style scoring model.

## Bloqueador

**Esta fase depende de Fase 2** ([`scripts/eval_reranker_mlx_tiers.py`](scripts/eval_reranker_mlx_tiers.py)).
Antes de ejecutar este script, hay que:

1. Correr el sweep de Fase 2 para encontrar el tier MLX que pasa floor.
2. Hacer el cutover (cambiar `DEFAULT_MLX_RERANKER` + `is_mlx_reranker_enabled`).
3. Acumular `RAG_FINETUNE_MIN_CORRECTIVES` (default 20) feedback pairs en
   `rag_feedback` para que el dataset sea entrenable.

## Qué hace este script

1. Re-usa `_fetch_feedback_pairs()` + `_build_lora_training_pairs()` del
   script viejo para construir el dataset de pares (query, doc, label).
2. **Reformatea como completion examples para SFT**: cada par positivo
   se mappea a `{"prompt": "<query>...<doc>", "completion": "yes"}` y
   cada hard negative a `{"prompt": ..., "completion": "no"}`. Esto
   matchea el formato que Qwen3-Reranker espera ya nativo para scoring
   (mismo prompt template que usa `MLXReranker._build_prompt()`).
3. Escribe JSONL en `~/.local/share/obsidian-rag/reranker_ft_mlx/data/{train,valid}.jsonl`.
4. Invoca `python -m mlx_lm lora --train --mode lora` (SFT, no DPO —
   el reranker es classification yes/no, no preferencia entre dos
   completions).
5. Adapter resultante en `~/.local/share/obsidian-rag/reranker_ft_mlx/`.

## Por qué SFT + no DPO

Para drafts (Fase 1.2) usamos DPO porque tenemos preference pairs
genuinos (`bot_draft` rejected, `sent_text` chosen). Para el reranker
tenemos labels binarias (relevant=yes, irrelevant=no) — eso es classification
clásica. SFT con prompt completion templates entrena más rápido y
preserva mejor el zero-shot baseline del modelo.

## Cómo correr (post Fase 2 cutover)

```bash
.venv/bin/python scripts/finetune_reranker_mlx.py train \\
    --base mlx-community/Qwen3-Reranker-0.6B-mxfp8 \\
    --iters 200
```

(El `--base` debe matchear el tier que ganó en Fase 2.)

## Runtime adapter loading

Cuando se quiera A/B el adapter MLX vs el base, el runtime
[`rag/mlx_reranker.py`](rag/mlx_reranker.py) `MLXReranker._ensure_loaded()`
puede recibir `adapter_path=...` via env `RAG_MLX_RERANKER_ADAPTER_PATH`.
**TODO**: agregar ese hook en `mlx_reranker.py` cuando se decida prender
el adapter (solo después de validar que aporta sobre el base, mismo
patrón que `RAG_RERANKER_FT=1` del path PEFT histórico).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Constantes ─────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
DEFAULT_OUTPUT_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "reranker_ft_mlx"
DEFAULT_ITERS = 200
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_LAYERS = 16
DEFAULT_BATCH_SIZE = 1

MIN_CORRECTIVES = int(os.environ.get("RAG_FINETUNE_MIN_CORRECTIVES", "20"))


# ── Dataset ─────────────────────────────────────────────────────────────


def _import_legacy_helpers():
    """Reusar las funciones de mining + pair building del script viejo."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finetune_reranker import (  # type: ignore[import-not-found]
        _fetch_feedback_pairs,
        _build_lora_training_pairs,
        _split_train_val,
    )
    return _fetch_feedback_pairs, _build_lora_training_pairs, _split_train_val


# Plantilla del prompt — mismo shape que `MLXReranker._build_prompt()`
# para que el adapter aprenda en el mismo distribution sobre el cual
# corre el inference.
_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _format_prompt(query: str, doc: str) -> str:
    """Formato compatible con `MLXReranker._build_prompt()`. Sin chat template
    (el SFT entrena al modelo a producir "yes"/"no" directo después de este
    prompt).
    """
    return (
        f"<Instruct>: {_RERANKER_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


def build_dataset(*, max_chars_doc: int = 1500) -> tuple[list[dict], list[dict]]:
    """Extrae pairs de `rag_feedback` (positivos + hard negatives) y los
    convierte al formato SFT de mlx_lm.lora (`{"prompt", "completion"}`).
    """
    fetch_feedback, build_pairs, split_tv = _import_legacy_helpers()
    feedback = fetch_feedback()
    if len(feedback) < MIN_CORRECTIVES:
        raise RuntimeError(
            f"Dataset demasiado chico: {len(feedback)} feedback pairs < "
            f"floor {MIN_CORRECTIVES}. Acumulá más signal en rag_feedback "
            f"(`rag harvest` o reactions del usuario)."
        )
    pairs = build_pairs(feedback)  # [{query, doc, label, ...}]
    examples = []
    for p in pairs:
        prompt_text = _format_prompt(
            (p.get("query") or "")[:300],
            (p.get("doc") or "")[:max_chars_doc],
        )
        completion = "yes" if p.get("label", 0) >= 0.5 else "no"
        examples.append({"prompt": prompt_text, "completion": completion})

    # Stratified split por label — mismo balance positivo/negativo en train/val
    pos = [e for e in examples if e["completion"] == "yes"]
    neg = [e for e in examples if e["completion"] == "no"]
    train_pos, val_pos = split_tv(pos)
    train_neg, val_neg = split_tv(neg)
    train = train_pos + train_neg
    val = val_pos + val_neg
    return train, val


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ── Training ───────────────────────────────────────────────────────────


def train_with_mlx_lora(
    *,
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    iters: int,
    learning_rate: float,
    num_layers: int,
    batch_size: int,
) -> None:
    """Invocar `mlx_lm.lora --train --mode lora` (SFT) como subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--data", str(data_dir),
        "--model", base_model,
        "--adapter-path", str(output_dir),
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--train-type", "lora",
        "--mode", "lora",  # SFT (no DPO)
    ]
    config_path = Path.home() / ".local/share/obsidian-rag/finetune_mlx_reranker.yml"
    if config_path.is_file():
        cmd.extend(["--config", str(config_path)])

    print(f"[mlx-lora] Ejecutando: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


# ── CLI ────────────────────────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[mlx-lora] Extrayendo feedback pairs…", flush=True)
    train, val = build_dataset()

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, val)
    print(
        f"[mlx-lora] Dataset escrito: train={len(train)} val={len(val)} → {data_dir}",
        flush=True,
    )

    train_with_mlx_lora(
        base_model=args.base,
        data_dir=data_dir,
        output_dir=output_dir,
        iters=args.iters,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
    )

    print(
        f"[mlx-lora] Entrenamiento OK. Adapter en {output_dir}\n"
        f"  TODO: wirear `RAG_MLX_RERANKER_ADAPTER_PATH=$ADAPTER` en\n"
        f"        `MLXReranker._ensure_loaded()` para activar runtime.",
        flush=True,
    )


def _cmd_dump_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    data_dir = output_dir / "data"
    train, val = build_dataset()
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "valid.jsonl", val)
    print(
        f"[mlx-lora] Dataset only — train={len(train)} val={len(val)} → {data_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reranker LoRA fine-tune via mlx_lm.lora (Qwen3-Reranker SFT).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train LoRA SFT via mlx_lm.lora")
    p_train.add_argument("--base", default=DEFAULT_BASE_MODEL,
                         help=f"Base reranker MLX (default: {DEFAULT_BASE_MODEL})")
    p_train.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Adapter output dir (default: {DEFAULT_OUTPUT_DIR})")
    p_train.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p_train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p_train.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.set_defaults(func=_cmd_train)

    p_dump = sub.add_parser("dump-dataset", help="Solo dump del dataset JSONL")
    p_dump.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    p_dump.set_defaults(func=_cmd_dump_dataset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
