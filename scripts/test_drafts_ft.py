"""Quick smoke test del LoRA adapter de drafts (post-fine-tune).

Uso:
  uv run --extra finetune python scripts/test_drafts_ft.py \\
      "Hola fer, mañana a las 10 vamos a la cancha con los pibes, te sumas?"

Carga el adapter desde ~/.local/share/obsidian-rag/drafts_ft/ y genera
una respuesta. Compara visualmente contra el base model (sin adapter)
para ver si el FT cambió el output.

NO depende del listener TS ni del endpoint /api/draft/preview — para
ese flujo el endpoint ya existe en `web/server.py`. Este script es solo
para iteración rápida durante el desarrollo del FT.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ADAPTER_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "drafts_ft"
DEFAULT_BASE_MODEL = os.environ.get(
    "RAG_DRAFTS_FT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)


def build_prompt(message: str, contact: str = "amigo") -> str:
    """Mismo template que `finetune_drafts.build_training_example`."""
    return (
        f"Conversación con {contact}:\n"
        f"- {message}\n\n"
        f"## Borrador del bot:\n"
        f"Entendido, te respondo en breve.\n\n"
        f"## Mensaje final que mandó Fer:\n"
    )


def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 100) -> str:
    import torch
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", help="Mensaje del contacto al cual querés que el modelo responda como Fer.")
    ap.add_argument("--contact", default="amigo")
    ap.add_argument("--no-baseline", action="store_true",
                    help="No correr el modelo base sin adapter (más rápido).")
    args = ap.parse_args()

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[error] no hay adapter en {ADAPTER_DIR}. Corré finetune_drafts.py primero.",
              file=sys.stderr)
        sys.exit(1)

    print(f"== Test de adapter en {ADAPTER_DIR} ==", file=sys.stderr)
    print(f"  Base: {DEFAULT_BASE_MODEL}", file=sys.stderr)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("  Loading tokenizer …", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(args.message, args.contact)

    print("  Loading base model …", file=sys.stderr, flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(DEFAULT_BASE_MODEL, torch_dtype="auto")

    if not args.no_baseline:
        print("\n--- BASELINE (sin adapter) ---")
        baseline_out = generate(base_model, tokenizer, prompt)
        print(baseline_out)

    print("  Loading LoRA adapter …", file=sys.stderr, flush=True)
    model_ft = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))

    print("\n--- FINE-TUNED (con adapter) ---")
    ft_out = generate(model_ft, tokenizer, prompt)
    print(ft_out)


if __name__ == "__main__":
    main()
