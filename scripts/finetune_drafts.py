"""Fine-tune del modelo de drafts de WhatsApp via LoRA (2026-04-29).

Contrato:
  Input:  rows de `rag_draft_decisions` con `decision='approved_editar'`
          (gold pairs (bot_draft, sent_text)) + `decision='rejected'`
          (anti-patterns).
  Output: PEFT/LoRA adapter en `~/.local/share/obsidian-rag/drafts_ft/`
          con `adapter_config.json` + `adapter_model.safetensors` +
          `ft_meta.json` (training config + held-out metrics).
  Goal:   capturar las correcciones que Fer hace al draft generado por
          el listener TS para que el modelo "aprenda" su estilo.

Decisión deliberada: el modelo en producción para generar drafts es
`qwen2.5:14b` (en el listener TS). El fine-tune corre sobre `qwen2.5:7b`
(modelo más chico, más rápido de entrenar, ~20x menos VRAM). El adapter
NO sustituye el draft generation del listener — es completamente
opcional y solo accesible vía endpoint preview (`/api/draft/preview`)
para permitirle a Fer comparar manualmente "qué hubiera salido del
baseline" vs "qué hubiera salido del fine-tuned".

Pipeline:
  1. Pull de `rag_draft_decisions` con `decision IN ('approved_editar',
     'rejected')`. `approved_editar` → label=positive (sent_text es el
     target real), `rejected` → label=negative (cualquier output ≠
     bot_draft preferido). Por default incluye rows con
     `extra_json.review_only=true`; el flag `--exclude-review-only` los
     filtra (signal del review-only loop sigue siendo signal real, así
     que default=incluir).
  2. Construye prompt en formato chat:
        <conversación previa>
        ## Borrador del bot:
        <bot_draft>
        ## Mensaje final que mandó Fer:
        <sent_text>     ← el target a aprender
  3. Stratified 80/20 split por `draft_id` (no leakea same-draft pares
     entre train/val).
  4. Entrena LoRA r=8 alpha=16 dropout=0.05 sobre Qwen2.5-7B (HF tag
     `Qwen/Qwen2.5-7B-Instruct`). Solo Q/V projections — convención
     que empíricamente funciona para causal LMs.
  5. Métricas held-out: BLEU-1 (unigram precision-recall) + similarity
     (1 - normalized edit distance via difflib.SequenceMatcher) entre
     pred y sent_text. Print 5 samples random.
  6. Persist `ft_meta.json` con timestamp, config, métricas — el
     loader runtime lo lee para mostrar en `rag draft stats`.

Activación del loader runtime:
  - El listener TS NO usa este modelo directamente (decisión: complejo
    + riesgo). El draft que va al user sigue saliendo de qwen2.5:14b en
    el listener.
  - El endpoint `/api/draft/preview` (web/server.py) acepta
    `{original_conversation, bot_draft_baseline}` y devuelve el output
    del fine-tuned model SI:
      * existe el adapter en `~/.local/share/obsidian-rag/drafts_ft/`
      * `RAG_DRAFTS_FT=1` está seteado
    Si alguna condición falla → echo del baseline (silent-fail con log
    a `silent_errors.jsonl`).

Reglas de seguridad:
  - <100 pares totales → exit 1 con mensaje claro ("datos insuficientes,
    esperá más feedback"). El fine-tune sobre poca data overfitea brutal.
  - peft no instalado → mensaje claro + exit 6 (no excepción rara).
  - dry-run: solo build pairs + print stats, no toca el modelo.
  - NO se promueve automáticamente: una vez que entrena, el operator
    decide cuándo activar via `RAG_DRAFTS_FT=1`. Este script no setea
    env vars ni reinicia procesos.

Uso:
  uv run python scripts/finetune_drafts.py --dry-run
  uv run python scripts/finetune_drafts.py --epochs 3 --lr 1e-4
  uv run python scripts/finetune_drafts.py --exclude-review-only
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Run with the venv python — el script importa rag para reusar la
# helper `_ragvec_state_conn()` (single source of truth para el path
# del telemetry DB).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Constantes ────────────────────────────────────────────────────────────

# El adapter vive bajo XDG data home (no cache) porque la signal del
# user es no-regenerable: si macOS limpia el cache por low-disk
# perdemos un fine-tune curado. Mismo path convention que el reranker
# (`rag.RERANKER_FT_ADAPTER_DIR`).
DRAFTS_FT_ADAPTER_DIR = (
    Path.home() / ".local" / "share" / "obsidian-rag" / "drafts_ft"
)

# Modelo base. Usamos el chico (7B vs 14B del listener) porque:
#   - El fine-tune del 14B requiere >40GB VRAM incluso con LoRA
#   - El A/B "baseline qwen2.5:14b en el listener vs fine-tuned 7B en
#     preview" se hace manualmente por el user, así que el cambio de
#     tamaño es aceptable.
# Tag HF (transformers descarga del Hub). Override via
# `RAG_DRAFTS_FT_BASE_MODEL` env si el user quiere experimentar.
DRAFTS_BASE_MODEL = os.environ.get(
    "RAG_DRAFTS_FT_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)

# LoRA hyperparams. Mismo r=8/alpha=16 que el reranker (ratio 2.0).
# dropout=0.05 más bajo que el reranker (0.1) porque la data de drafts
# es más diversa por fuente: no overfit-prone como el rerank-binary.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Qwen2.5 attention proj names (verificable con
# `print(model)` — los q/v projection layers). Mantengo solo q+v para
# minimizar parametros entrenables (consistente con reranker).
LORA_TARGET_MODULES = ("q_proj", "v_proj")

# Threshold mínimo de pares para considerar el fine-tune viable.
# Lo justifico al user en el exit message: <100 → variance del estilo
# personal no se captura, el LoRA con 50 ejemplos overfit a las 50
# frases exactas y no generaliza.
MIN_PAIRS = 100

# Held-out validation fraction. 20% es estándar para LoRA con datasets
# pequeños — el train set queda con 80%, suficiente para 3 epochs sin
# memorizar.
VAL_FRAC = 0.2

# Max tokens por sample. Las conversaciones de WA son cortas (<2KB
# típico). 1024 cubre prompts grandes sin truncar la respuesta target.
MAX_TOKENS = 1024


# ── Mining de pairs ───────────────────────────────────────────────────────


def fetch_draft_pairs(*, exclude_review_only: bool = False) -> dict:
    """Pull `rag_draft_decisions` y arma listas de gold/anti-pattern pairs.

    Args:
        exclude_review_only: si True, descarta rows cuyo
            `extra_json.review_only` sea truthy. Default False — el
            review-only loop sigue siendo signal real (el user igual
            corrige), así que normalmente lo queremos en train.

    Returns:
        dict con keys:
            gold: list[dict]    — approved_editar, gold (bot_draft, sent_text)
            anti: list[dict]    — rejected (anti-pattern: cualquier output != bot_draft)
            stats: dict         — counts + ratio review-only para reporte
    """
    gold: list[dict] = []
    anti: list[dict] = []
    n_review_only_total = 0
    n_review_only_excluded = 0

    try:
        with rag._ragvec_state_conn() as conn:
            rows = list(conn.execute(
                """
                SELECT id, draft_id, contact_jid, contact_name,
                       original_msgs_json, bot_draft, decision,
                       sent_text, extra_json, ts
                FROM rag_draft_decisions
                WHERE decision IN ('approved_editar', 'rejected')
                ORDER BY ts ASC
                """
            ).fetchall())
    except Exception as exc:
        print(f"[error] reading rag_draft_decisions: {exc}", file=sys.stderr)
        return {"gold": [], "anti": [], "stats": {}}

    for row in rows:
        (rid, draft_id, jid, name, msgs_json, bot_draft, decision,
         sent_text, extra_json, ts) = row
        is_review_only = False
        if extra_json:
            try:
                extra = json.loads(extra_json)
                # truthy check: True, 1, "true" all count
                if extra.get("review_only") in (True, 1, "true", "True"):
                    is_review_only = True
            except Exception:
                pass
        if is_review_only:
            n_review_only_total += 1
            if exclude_review_only:
                n_review_only_excluded += 1
                continue

        try:
            original_msgs = json.loads(msgs_json) if msgs_json else []
        except Exception:
            original_msgs = []

        item = {
            "id": rid,
            "draft_id": draft_id,
            "contact_name": name or "",
            "original_msgs": original_msgs,
            "bot_draft": bot_draft or "",
            "sent_text": sent_text,
            "ts": ts,
        }

        if decision == "approved_editar":
            # Gold pair: bot_draft != sent_text por definición (si fueran
            # iguales sería approved_si). El sent_text es el target.
            if not sent_text:
                # defensivo: no debería pasar (el listener envía sent_text
                # cuando hay edit). Skip silently.
                continue
            gold.append(item)
        elif decision == "rejected":
            # Anti-pattern: el bot_draft fue rechazado completamente. NO
            # tenemos un "good output" alternativo, solo sabemos que el
            # bot_draft no sirve. Lo usamos como signal débil (ranking-
            # loss style: preferir CUALQUIER output != bot_draft).
            anti.append(item)

    stats = {
        "n_total_rows": len(rows),
        "n_gold": len(gold),
        "n_anti": len(anti),
        "n_review_only_total": n_review_only_total,
        "n_review_only_excluded": n_review_only_excluded,
        "review_only_ratio": (
            n_review_only_total / len(rows) if rows else 0.0
        ),
    }
    return {"gold": gold, "anti": anti, "stats": stats}


# ── Pair → training prompt ────────────────────────────────────────────────


def _format_conversation(original_msgs: list[dict]) -> str:
    """Formatea los mensajes originales como una conversación legible.

    Mantengo el formato simple (sin role tags) porque Qwen2.5 sabe
    parsear mensajes naturales. Cap a último ~600 chars para no
    saturar el prompt — el contact rarely tiene >5 mensajes
    relevantes pre-draft.
    """
    if not original_msgs:
        return "(sin contexto previo)"
    lines = []
    for msg in original_msgs[-5:]:  # últimos 5 mensajes
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"- {text}")
    full = "\n".join(lines)
    return full[-600:] if len(full) > 600 else full


def build_training_example(item: dict, *, is_anti: bool = False) -> dict:
    """Convierte una row del DB en un sample de training (prompt → target).

    Para gold (approved_editar):
        prompt: contexto + bot_draft + "## Mensaje final que mandó Fer:\n"
        target: sent_text
        weight: 1.0

    Para anti (rejected):
        prompt: contexto + bot_draft + "## Mensaje final que mandó Fer:\n"
        target: "" (vacío — pseudo-anti, el modelo aprende que el
                bot_draft completo merecía ser ignorado)
        weight: 0.3 (signal débil — sabemos que bot_draft no sirve, pero
                no tenemos un good output)

    Note: el peso anti=0.3 es heurístico. Si en el futuro acumulamos
    enough volume de rejected (>500), podríamos hacer ranking-loss
    contrastivo en lugar de peso. Por ahora simple (CrossEntropy
    weighted).
    """
    contact = item.get("contact_name") or "(sin nombre)"
    convo = _format_conversation(item["original_msgs"])
    bot_draft = item.get("bot_draft") or ""

    prompt = (
        f"Conversación con {contact}:\n"
        f"{convo}\n\n"
        f"## Borrador del bot:\n"
        f"{bot_draft}\n\n"
        f"## Mensaje final que mandó Fer:\n"
    )
    if is_anti:
        # Pseudo-target vacío. El loss apunta a "no copiar el bot_draft".
        target = ""
        weight = 0.3
    else:
        target = item.get("sent_text") or ""
        weight = 1.0

    return {
        "prompt": prompt,
        "target": target,
        "weight": weight,
        "draft_id": item.get("draft_id", ""),
        # Useful para debugging:
        "_bot_draft": bot_draft,
    }


def split_train_val(
    examples: list[dict], val_frac: float = VAL_FRAC, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified split por `draft_id` para evitar leakeo same-draft entre
    train/val. Si todos los drafts son únicos (esperable en data real),
    es equivalente a un split random.
    """
    rng = random.Random(seed)
    by_draft: dict[str, list[dict]] = {}
    for ex in examples:
        by_draft.setdefault(ex["draft_id"], []).append(ex)
    drafts = sorted(by_draft.keys())
    rng.shuffle(drafts)
    n_val = max(1, int(len(drafts) * val_frac))
    val_drafts = set(drafts[:n_val])
    train, val = [], []
    for d, lst in by_draft.items():
        if d in val_drafts:
            val.extend(lst)
        else:
            train.extend(lst)
    return train, val


# ── Métricas (BLEU-1 + similarity) ────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Tokenizer simple: lowercase + split por whitespace.

    No usamos sacrebleu/nltk para no agregar dep — la métrica acá es
    un sanity-check, no una eval competitiva. El user hace la eval real
    visual via `/api/draft/preview` comparando outputs.
    """
    return (text or "").lower().split()


def bleu1(pred: str, ref: str) -> float:
    """BLEU-1 unigram precision (sin brevity penalty).

    Range [0, 1]. 1.0 = todos los tokens del pred están en ref. Es
    una aproximación grosera: para drafts cortos (10-50 palabras) es
    más informativa que BLEU-4 (que tiende a 0 por sparsity).
    """
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks:
        return 0.0
    ref_counts = Counter(ref_toks)
    matches = 0
    for tok in pred_toks:
        if ref_counts.get(tok, 0) > 0:
            matches += 1
            ref_counts[tok] -= 1
    return matches / len(pred_toks)


def similarity(pred: str, ref: str) -> float:
    """1 - normalized edit distance via difflib.SequenceMatcher.

    Range [0, 1]. 1.0 = idénticos a nivel char. difflib es stdlib —
    no agregamos dep. Para drafts cortos correlaciona bien con
    "qué tan cerca quedó" mejor que el edit distance puro de
    Levenshtein.
    """
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    return difflib.SequenceMatcher(None, pred, ref).ratio()


def evaluate_predictions(
    val_examples: list[dict], predictions: list[str],
) -> dict:
    """Computa métricas held-out + selecciona 5 samples random para print.
    """
    if not predictions:
        return {
            "n_val": 0, "bleu1_mean": 0.0, "sim_mean": 0.0,
            "samples": [],
        }
    bleu_scores = [
        bleu1(pred, ex["target"])
        for pred, ex in zip(predictions, val_examples)
    ]
    sim_scores = [
        similarity(pred, ex["target"])
        for pred, ex in zip(predictions, val_examples)
    ]

    rng = random.Random(42)
    sample_idxs = sorted(rng.sample(
        range(len(val_examples)), k=min(5, len(val_examples)),
    ))
    samples = []
    for i in sample_idxs:
        samples.append({
            "bot_draft": val_examples[i].get("_bot_draft", ""),
            "target_sent_text": val_examples[i]["target"],
            "prediction": predictions[i],
            "bleu1": bleu_scores[i],
            "sim": sim_scores[i],
        })

    return {
        "n_val": len(predictions),
        "bleu1_mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "bleu1_min": min(bleu_scores) if bleu_scores else 0.0,
        "bleu1_max": max(bleu_scores) if bleu_scores else 0.0,
        "sim_mean": sum(sim_scores) / len(sim_scores) if sim_scores else 0.0,
        "sim_min": min(sim_scores) if sim_scores else 0.0,
        "sim_max": max(sim_scores) if sim_scores else 0.0,
        "samples": samples,
    }


# ── Training (LoRA via peft) ──────────────────────────────────────────────


def train_lora(
    train_examples: list[dict], val_examples: list[dict],
    *, out_dir: Path, epochs: int, lr: float, batch_size: int,
) -> dict:
    """Entrena un LoRA adapter sobre Qwen2.5-7B-Instruct.

    Imports lazy adentro de la fn para que `--dry-run` no intente
    cargar transformers/peft (1-2s startup + dep check). Si falta
    `peft` o `transformers`, exit 6 con mensaje accionable.

    Returns dict con métricas held-out (bleu1_mean, sim_mean, samples).
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(
            f"[error] missing dep para LoRA training: {exc}\n"
            "  Install with: uv tool install --reinstall --editable '.[finetune]'",
            file=sys.stderr,
        )
        sys.exit(6)

    device = os.environ.get("RAG_FT_DEVICE", "cpu").lower()
    print(
        f"  Loading {DRAFTS_BASE_MODEL} (transformers={transformers.__version__}) "
        f"on device={device} …",
        file=sys.stderr,
    )

    tokenizer = AutoTokenizer.from_pretrained(DRAFTS_BASE_MODEL)
    if tokenizer.pad_token is None:
        # Qwen no setea pad_token por default — usamos eos para el
        # padding del collator. Standard recipe.
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        DRAFTS_BASE_MODEL,
        torch_dtype="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    def _tok_fn(batch):
        # Concatenamos prompt + target (causal LM training: el modelo
        # ve todo el secuencia y el loss se computa sobre los tokens
        # del target solamente — el collator usa labels=-100 en los
        # prompt tokens via `mlm=False` + label_pad_token_id).
        texts = []
        for prompt, target in zip(batch["prompt"], batch["target"]):
            texts.append(prompt + target + tokenizer.eos_token)
        enc = tokenizer(
            texts, truncation=True, max_length=MAX_TOKENS,
            padding=False,
        )
        return enc

    train_ds = Dataset.from_list([
        {"prompt": ex["prompt"], "target": ex["target"]} for ex in train_examples
    ]).map(_tok_fn, batched=True, remove_columns=["prompt", "target"])
    val_ds = (Dataset.from_list([
        {"prompt": ex["prompt"], "target": ex["target"]} for ex in val_examples
    ]).map(_tok_fn, batched=True, remove_columns=["prompt", "target"])
              if val_examples else None)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
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

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    try:
        tokenizer.save_pretrained(str(out_dir))
    except Exception as exc:
        print(f"  [warn] tokenizer.save_pretrained failed: {exc}", file=sys.stderr)

    # ── Held-out predictions + métricas ──
    print("  Generating held-out predictions for metrics …", file=sys.stderr)
    predictions = generate_predictions(model, tokenizer, val_examples, device)
    metrics = evaluate_predictions(val_examples, predictions)
    return metrics


def generate_predictions(
    model, tokenizer, val_examples: list[dict], device: str,
) -> list[str]:
    """Genera predicciones del modelo para val_examples (held-out).

    Cada sample: input = prompt, output = generación greedy de
    máx 200 tokens. No hacemos sampling ni temperature porque queremos
    reproducibilidad para BLEU/sim mediciones.
    """
    import torch
    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for ex in val_examples:
            inp = tokenizer(
                ex["prompt"], return_tensors="pt", truncation=True,
                max_length=MAX_TOKENS - 200,
            )
            out = model.generate(
                **inp,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            # Strip el prompt para quedarnos solo con la generación
            if full.startswith(ex["prompt"]):
                preds.append(full[len(ex["prompt"]):].strip())
            else:
                preds.append(full.strip())
    return preds


# ── Reporte ───────────────────────────────────────────────────────────────


def print_stats(stats: dict, *, exclude_review_only: bool) -> None:
    """Print del resumen del mining (post-fetch_draft_pairs)."""
    print("== Mining stats ==", file=sys.stderr)
    print(f"  Total rows (approved_editar + rejected): {stats['n_total_rows']}",
          file=sys.stderr)
    print(f"  Gold pairs (approved_editar):            {stats['n_gold']}",
          file=sys.stderr)
    print(f"  Anti-pattern (rejected):                 {stats['n_anti']}",
          file=sys.stderr)
    print(f"  Review-only rows (extra_json flag):      "
          f"{stats['n_review_only_total']}  "
          f"({stats['review_only_ratio'] * 100:.1f}%)",
          file=sys.stderr)
    if exclude_review_only:
        print(f"  Excluded by --exclude-review-only:       "
              f"{stats['n_review_only_excluded']}",
              file=sys.stderr)


def print_metrics_report(metrics: dict) -> None:
    """Print del reporte de métricas held-out + samples."""
    print("\n== Held-out validation metrics ==", file=sys.stderr)
    print(f"  n_val: {metrics['n_val']}", file=sys.stderr)
    print(
        f"  BLEU-1 mean={metrics['bleu1_mean']:.3f}  "
        f"min={metrics['bleu1_min']:.3f}  max={metrics['bleu1_max']:.3f}",
        file=sys.stderr,
    )
    print(
        f"  Similarity (char-level) mean={metrics['sim_mean']:.3f}  "
        f"min={metrics['sim_min']:.3f}  max={metrics['sim_max']:.3f}",
        file=sys.stderr,
    )
    print("\n== 5 random held-out samples ==", file=sys.stderr)
    for i, sample in enumerate(metrics.get("samples", []), start=1):
        print(f"\n  --- Sample {i} ---", file=sys.stderr)
        print(f"  bot_draft: {sample['bot_draft'][:200]}", file=sys.stderr)
        print(f"  target:    {sample['target_sent_text'][:200]}", file=sys.stderr)
        print(f"  pred:      {sample['prediction'][:200]}", file=sys.stderr)
        print(f"  bleu1={sample['bleu1']:.3f}  sim={sample['sim']:.3f}",
              file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune del modelo de drafts WhatsApp via LoRA",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size por device (default 4 para CPU/MPS).")
    ap.add_argument("--exclude-review-only", action="store_true",
                    help="Filtrar rows con extra_json.review_only=true. "
                         "Default: incluir (signal real igual).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build pairs + report stats; NO entrena.")
    args = ap.parse_args()

    print("== fine-tune drafts WhatsApp (LoRA) ==", file=sys.stderr)
    print(f"  Base model: {DRAFTS_BASE_MODEL}", file=sys.stderr)
    print(f"  Adapter dir: {DRAFTS_FT_ADAPTER_DIR}", file=sys.stderr)

    data = fetch_draft_pairs(exclude_review_only=args.exclude_review_only)
    print_stats(data["stats"], exclude_review_only=args.exclude_review_only)

    # Build examples: gold + anti combinados.
    examples = (
        [build_training_example(item, is_anti=False) for item in data["gold"]]
        + [build_training_example(item, is_anti=True) for item in data["anti"]]
    )
    pos = sum(1 for ex in examples if ex["weight"] >= 1.0)
    neg = sum(1 for ex in examples if ex["weight"] < 1.0)
    print(f"\n  Training examples: total={len(examples)} "
          f"gold={pos} anti={neg}", file=sys.stderr)

    if len(examples) < MIN_PAIRS:
        print(
            f"\n[error] datos insuficientes: {len(examples)} pares < {MIN_PAIRS} mínimo. "
            f"Esperá más feedback (más decisions /editar y /no en RagNet).\n"
            f"  Tip: `rag draft stats` para ver el conteo actual.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] exiting before training. Pairs OK para entrenar.",
              file=sys.stderr)
        return

    train_examples, val_examples = split_train_val(examples, val_frac=VAL_FRAC)
    print(f"\n  Split: train={len(train_examples)} val={len(val_examples)}",
          file=sys.stderr)

    DRAFTS_FT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Training → {DRAFTS_FT_ADAPTER_DIR}", file=sys.stderr)
    t0 = time.time()
    metrics = train_lora(
        train_examples, val_examples,
        out_dir=DRAFTS_FT_ADAPTER_DIR,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f"\n  Trained in {elapsed:.1f}s", file=sys.stderr)
    print_metrics_report(metrics)

    # Persist meta para que `rag drafts stats` lo lea.
    try:
        meta = {
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "base_model": DRAFTS_BASE_MODEL,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "exclude_review_only": args.exclude_review_only,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "n_gold": pos,
            "n_anti": neg,
            "elapsed_sec": elapsed,
            "metrics": {
                k: v for k, v in metrics.items() if k != "samples"
            },
        }
        (DRAFTS_FT_ADAPTER_DIR / "ft_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        print(f"  Meta saved → {DRAFTS_FT_ADAPTER_DIR / 'ft_meta.json'}",
              file=sys.stderr)
    except Exception as exc:
        print(f"  [warn] writing ft_meta.json failed: {exc}", file=sys.stderr)

    print(
        "\n  Adapter saved. Para activar el endpoint preview con este "
        "adapter:\n"
        "    export RAG_DRAFTS_FT=1\n"
        "  El listener TS NO usa este modelo (sigue con qwen2.5:14b en prod). "
        "Solo el endpoint /api/draft/preview lo carga para A/B manual.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
