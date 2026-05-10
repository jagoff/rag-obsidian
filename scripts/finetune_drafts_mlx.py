#!/usr/bin/env python3
"""Drafts DPO fine-tune via [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

Fase 1.2 del plan MLX-full-migration (Ola 10, 2026-05-07). Reemplaza el
flow `trl + peft + transformers` de [`scripts/finetune_drafts.py`](scripts/finetune_drafts.py)
por un wrapper alrededor del CLI nativo `mlx_lm.lora` con `--mode dpo`.

## Qué hace

1. Extrae el dataset de gold preference pairs (`{prompt, chosen, rejected}`)
   reusando la lógica existente de [`fetch_draft_pairs()`](finetune_drafts.py)
   + `build_dpo_example()` + `split_train_val()`.
2. Escribe `train.jsonl` y `valid.jsonl` en el directorio de output con el
   shape exacto que `mlx_lm.lora` espera para DPO
   (un objeto JSON por línea con keys `prompt`, `chosen`, `rejected`).
3. Invoca `python -m mlx_lm lora --train --data <data_dir>
   --model <base-mlx-id> --adapter-path <out_dir> --num-layers 16
   --batch-size 1 --iters <N> --learning-rate 1e-5 --train-type lora`
   con `--config` opcional (tomado de `~/.local/share/obsidian-rag/finetune_mlx_dpo.yml`
   si existe). El adapter resultante queda en
   `~/.local/share/obsidian-rag/drafts_ft_mlx/` (separado del adapter
   PEFT histórico para coexistencia + rollback).

## Por qué importa

`mlx_lm.lora` corre nativo sobre Apple Silicon GPU (Metal), elimina la
dep `trl` + `peft` + `transformers` + `torch` para fine-tuning, y carga
el adapter resultante directo en runtime via `mlx_lm.load(model,
adapter_path=...)`. Un solo runtime tanto para train como para inference.

## Cómo usar

```bash
.venv/bin/python scripts/finetune_drafts_mlx.py train \\
    --base mlx-community/Qwen2.5-7B-Instruct-4bit \\
    --iters 200 \\
    --output ~/.local/share/obsidian-rag/drafts_ft_mlx
```

Después, el runtime loader (gestionado por `_load_drafts_ft_model_mlx()`)
levanta el adapter cuando `RAG_DRAFTS_FT=1` + adapter dir existe.

## Rollback

Mantener el script viejo `finetune_drafts.py` + el adapter dir
`~/.local/share/obsidian-rag/drafts_ft/` permite volver al path PEFT
histórico sin perder el modelo entrenado. Para forzar el path viejo en
runtime: `RAG_DRAFTS_FT_BACKEND=peft`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ── Constantes ─────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_OUTPUT_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "drafts_ft_mlx"
DEFAULT_ITERS = 200
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_LAYERS = 16
DEFAULT_BATCH_SIZE = 1
MIN_EXAMPLES = 20  # Floor: con menos no vale la pena entrenar


# ── Dataset extraction ─────────────────────────────────────────────────


def _import_legacy_helpers():
    """Importar `fetch_draft_pairs`, `build_dpo_example`, `split_train_val`
    del script viejo sin duplicar 866 líneas. El módulo legacy expone
    todas las funciones públicas — solo lo importamos en demand para no
    cargar trl/peft acá (que no necesitamos)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finetune_drafts import (  # type: ignore[import-not-found]
        fetch_draft_pairs,
        build_dpo_example,
        split_train_val,
    )
    return fetch_draft_pairs, build_dpo_example, split_train_val


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    """Escribir lista de dicts como JSONL (una línea por record).

    `mlx_lm.lora --mode dpo` espera EXACTAMENTE las keys `prompt`,
    `chosen`, `rejected` (sin extras). Filtramos draft_id u otras
    metadatas del shape de `build_dpo_example()`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            record = {
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(*, exclude_review_only: bool = False) -> tuple[list[dict], list[dict]]:
    """Extrae el dataset de `rag_draft_decisions` y devuelve (train, val).

    Re-usa la lógica del script viejo para preservar el formato de prompt
    + el split estratificado por `draft_id`.

    Returns:
        (train, val): listas de dicts con keys `prompt`, `chosen`,
        `rejected`, `draft_id`. El draft_id se usa solo para el split y
        se descarta antes de escribir el JSONL final.

    Raises:
        RuntimeError si el dataset es demasiado chico para entrenar
        (`< MIN_EXAMPLES`). Best-practice de TRL es ≥50 pairs para que
        DPO converja sin overfitting masivo; con <20 los gradients son
        demasiado ruidosos.
    """
    fetch, build_one, split = _import_legacy_helpers()
    bundle = fetch(exclude_review_only=exclude_review_only)
    pairs = bundle.get("gold", [])
    examples = [build_one(item) for item in pairs]
    if len(examples) < MIN_EXAMPLES:
        raise RuntimeError(
            f"Dataset demasiado chico: {len(examples)} < {MIN_EXAMPLES} pairs. "
            f"Acumulá más rows en `rag_draft_decisions` antes de entrenar."
        )
    train, val = split(examples)
    return train, val


# ── Training (mlx_lm.lora subprocess) ──────────────────────────────────


def train_with_mlx_lora(
    *,
    base_model: str,
    train_path: Path,
    valid_path: Path,
    output_dir: Path,
    iters: int,
    learning_rate: float,
    num_layers: int,
    batch_size: int,
) -> None:
    """Invocar `mlx_lm.lora` CLI como subprocess.

    `mlx_lm.lora` es el flow oficial de Apple para fine-tuning con LoRA
    (incluido DPO). Doc: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

    El CLI espera que `--data` apunte a un dir conteniendo `train.jsonl`
    + `valid.jsonl` con el formato DPO (un dict por línea con
    `prompt`/`chosen`/`rejected`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Los JSONLs deben vivir en el mismo dir que pasamos a --data.
    data_dir = train_path.parent
    if valid_path.parent != data_dir:
        raise ValueError(
            f"train ({train_path.parent}) y valid ({valid_path.parent}) "
            "tienen que vivir en el mismo dir; mlx_lm.lora --data lo asume."
        )

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
        # `--mode dpo` activa el DPO loss path (vs SFT por default).
        "--mode", "dpo",
    ]

    # Override config opcional (~/.local/share/obsidian-rag/finetune_mlx_dpo.yml)
    config_path = Path.home() / ".local/share/obsidian-rag/finetune_mlx_dpo.yml"
    if config_path.is_file():
        cmd.extend(["--config", str(config_path)])

    print(f"[mlx-lora] Ejecutando: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


# ── CLI ────────────────────────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[mlx-lora] Extrayendo dataset de rag_draft_decisions…", flush=True)
    train, val = build_dataset(exclude_review_only=args.exclude_review_only)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, val)
    print(
        f"[mlx-lora] Dataset escrito: {train_path.name} ({len(train)} ex), "
        f"{valid_path.name} ({len(val)} ex)",
        flush=True,
    )

    train_with_mlx_lora(
        base_model=args.base,
        train_path=train_path,
        valid_path=valid_path,
        output_dir=output_dir,
        iters=args.iters,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
    )

    print(
        f"[mlx-lora] Entrenamiento OK. Adapter en {output_dir}\n"
        f"  Activar runtime: export RAG_DRAFTS_FT=1 RAG_DRAFTS_FT_BACKEND=mlx",
        flush=True,
    )


def _cmd_dump_dataset(args: argparse.Namespace) -> None:
    """Solo extrae el dataset y escribe los JSONLs — útil para inspección
    sin disparar training."""
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val = build_dataset(exclude_review_only=args.exclude_review_only)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, val)
    print(
        f"[mlx-lora] Dataset only — train={len(train)} val={len(val)} → {output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drafts DPO fine-tune via mlx_lm.lora (MLX-native).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train DPO LoRA via mlx_lm.lora")
    p_train.add_argument("--base", default=DEFAULT_BASE_MODEL,
                         help=f"Modelo base MLX (default: {DEFAULT_BASE_MODEL})")
    p_train.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Adapter output dir (default: {DEFAULT_OUTPUT_DIR})")
    p_train.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                         help=f"DPO iterations (default: {DEFAULT_ITERS})")
    p_train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                         help=f"LR (default: {DEFAULT_LEARNING_RATE})")
    p_train.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS,
                         help=f"Layers a finetuner (default: {DEFAULT_NUM_LAYERS})")
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    p_train.add_argument("--exclude-review-only", action="store_true",
                         help="Excluir rows con review_only=True (default: incluir)")
    p_train.set_defaults(func=_cmd_train)

    p_dump = sub.add_parser("dump-dataset", help="Solo dump del dataset JSONL")
    p_dump.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output dir")
    p_dump.add_argument("--exclude-review-only", action="store_true")
    p_dump.set_defaults(func=_cmd_dump_dataset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
