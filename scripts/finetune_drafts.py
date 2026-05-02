"""Fine-tune del modelo de drafts de WhatsApp via DPO + LoRA (2026-05-01).

Contrato:
  Input:  rows de `rag_draft_decisions` con `decision='approved_editar'`
          — gold preference pairs `(bot_draft=rejected, sent_text=chosen)`.
  Output: PEFT/LoRA adapter en `~/.local/share/obsidian-rag/drafts_ft/`
          con `adapter_config.json` + `adapter_model.safetensors` +
          `ft_meta.json` (training config + held-out metrics).
  Goal:   alinear el modelo al tono Fer (rioplatense, voseo, sin
          "¡"/"¿", acks atómicos) usando DPO sobre los pares reales
          (corporate bot_draft → estilo Fer sent_text) que el listener
          captura cuando vos editás un borrador.

Por qué DPO y no SFT:
  El refactor anterior usaba SFT (Supervised Fine-Tuning) con un truco
  feo: para `decision='rejected'` ponía `target=""` con `weight=0.3`
  como "pseudo-anti-pattern". Eso es un parche — el modelo igual ve
  todos los pares con la misma cross-entropy y aprende mediocre.

  DPO (Direct Preference Optimization, [Rafailov et al. 2023]) es la
  herramienta natural para este problema: tus pares son LITERALMENTE
  preference pairs (sent_text > bot_draft). DPO optimiza directamente
  log p(chosen) - log p(rejected) con una KL penalty contra el modelo
  base, evitando catastrophic forgetting.

  Caso de éxito comparable: [`RigoChat-7b-v2`](https://huggingface.co/IIC/RigoChat-7b-v2)
  es Qwen2.5-7B-Instruct fine-tuned con DPO para español por el IIC.
  Mejora 79.55 vs 77.17 en Spanish benchmark MMLU vs el base, sin
  perder capabilities en otros idiomas. Mismo playbook acá.

Decisión deliberada — base model 7B vs el 14B en producción:
  El listener TS usa qwen2.5:7b en producción (con fallback). El
  fine-tune corre sobre `Qwen/Qwen2.5-7B-Instruct` (HF Hub, no Ollama).
  El adapter NO sustituye el draft generation del listener — es
  completamente opcional y solo accesible vía endpoint preview
  (`/api/draft/preview`) para A/B manual del user.

Pipeline:
  1. Pull de `rag_draft_decisions` con `decision='approved_editar'`. Por
     default incluye rows con `extra_json.review_only=true`; el flag
     `--exclude-review-only` los filtra. La data sintética del
     `augment_drafts_dataset.py` (review_only=true, synthetic=true) se
     incluye también — ayuda al volumen para que el LoRA generalice.
  2. Construye DPO triplets:
        prompt:   "Conversación con <contacto>:\n<últimos 5 msgs>\n\n## Tu respuesta:\n"
        chosen:   sent_text          ← lo que vos escribiste (target)
        rejected: bot_draft          ← lo que el bot propuso (corporate)
  3. Stratified 80/20 split por `draft_id` (no leakea same-draft entre
     train/val).
  4. DPO sobre Qwen2.5-7B con LoRA r=8 alpha=16 dropout=0.05, target
     modules q+v projections, beta=0.1 (KL penalty estándar).
  5. Métricas held-out: BLEU-1 (unigram precision-recall) +
     similarity (1 - normalized edit distance) + preference win rate
     (cuántas veces la generación se parece más al chosen que al
     rejected) entre pred y sent_text. Print 5 samples random.
  6. Persist `ft_meta.json` con timestamp, config, métricas.

Activación del loader runtime (sin cambios vs SFT):
  - El listener TS NO usa este modelo directamente.
  - El endpoint `/api/draft/preview` (web/server.py) acepta
    `{original_conversation, bot_draft_baseline}` y devuelve el output
    del fine-tuned model SI `RAG_DRAFTS_FT=1` + adapter existe.
    Si alguna condición falla → echo del baseline.

Reglas de seguridad:
  - <100 pares con preference completo → exit 1 con mensaje claro.
  - peft / trl no instalados → mensaje claro + exit 6.
  - dry-run: solo build pairs + print stats, no toca el modelo.
  - NO se promueve automáticamente.

Uso:
  uv run python scripts/finetune_drafts.py --dry-run
  uv run python scripts/finetune_drafts.py --epochs 1 --lr 5e-6
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

# Modelo base. Override via `RAG_DRAFTS_FT_BASE_MODEL` env si el user
# quiere experimentar con RigoChat-7b-v2 / llama3.1:8b, etc.
DRAFTS_BASE_MODEL = os.environ.get(
    "RAG_DRAFTS_FT_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)

# LoRA hyperparams. r=8/alpha=16 dropout=0.05 mismo que el reranker
# (ratio 2.0). Solo q+v projections — minimiza parametros entrenables
# y es la convención que empíricamente funciona para causal LMs.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "v_proj")

# Threshold mínimo de PREFERENCE PAIRS COMPLETOS para considerar el
# fine-tune viable. <100 → DPO con LoRA overfit-prone (gradient noise
# alto + few preference signals).
MIN_PAIRS = 100

# Held-out validation fraction. 20% es estándar para LoRA con datasets
# pequeños — train queda con 80%, suficiente para 1-3 epochs sin memo.
VAL_FRAC = 0.2

# Max tokens por sample. Las conversaciones de WA son cortas (<2KB
# típico). 1024 cubre prompts grandes sin truncar la respuesta target.
MAX_TOKENS = 1024

# DPO beta (KL penalty coefficient). 0.1 es el default de TRL y un
# sweet spot empírico — más alto (0.5+) anchorea demasiado al modelo
# base y limita el aprendizaje del estilo.
DPO_BETA = 0.1


# ── Mining de pairs ───────────────────────────────────────────────────────


def fetch_draft_pairs(*, exclude_review_only: bool = False) -> dict:
    """Pull `rag_draft_decisions` y arma listas de gold preference pairs.

    Args:
        exclude_review_only: si True, descarta rows cuyo
            `extra_json.review_only` sea truthy. Default False — los
            sintéticos del augmenter Y los reales review-only siguen
            siendo signal real.

    Returns:
        dict con keys:
            gold: list[dict]    — preference pairs (bot_draft=rejected, sent_text=chosen)
            anti: list[dict]    — VACÍA siempre (legacy compat: el SFT viejo
                                  usaba 'rejected' rows como pseudo-anti, DPO
                                  no las puede usar sin un chosen alternativo)
            stats: dict         — counts + ratio review-only para reporte
    """
    gold: list[dict] = []
    # `anti` queda vacía — el legacy SFT usaba decision='rejected' rows
    # como pseudo-anti-patterns con target="". DPO requiere AMBOS chosen
    # y rejected, y no tenemos un "chosen alternativo" para esas rows.
    # Las skipeamos limpiamente. Las contamos para el reporte.
    anti: list[dict] = []
    n_review_only_total = 0
    n_review_only_excluded = 0
    n_rejected_skipped = 0

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

        # decision='rejected' rows: skip silently. Legacy SFT las
        # contaba como anti, DPO no puede usarlas.
        if decision == "rejected":
            n_rejected_skipped += 1
            continue

        # Only approved_editar reaches here. Validamos sent_text ≠ bot_draft.
        if not sent_text:
            # defensivo: no debería pasar (si fueran iguales sería
            # approved_si). Skip silently.
            continue
        if sent_text == bot_draft:
            # par degenerado: chosen == rejected → DPO log-ratio = 0,
            # gradient = 0. Skip para no diluir el batch.
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
        gold.append(item)

    stats = {
        "n_total_rows": len(rows),
        "n_gold": len(gold),
        # `n_anti` queda en 0 siempre. Mantenemos la key por
        # compat con scripts/observers que la leen.
        "n_anti": 0,
        "n_rejected_skipped": n_rejected_skipped,
        "n_review_only_total": n_review_only_total,
        "n_review_only_excluded": n_review_only_excluded,
        "review_only_ratio": (
            n_review_only_total / len(rows) if rows else 0.0
        ),
    }
    return {"gold": gold, "anti": anti, "stats": stats}


# ── Pair → DPO triplet ────────────────────────────────────────────────────


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


def build_dpo_example(item: dict) -> dict:
    """Convierte una row gold en un DPO preference triplet.

    El prompt MATCHEA lo que el modelo va a ver en producción
    (`/api/draft/preview`): solo el contexto conversacional, sin el
    bot_draft incluido. El bot_draft es el "rejected" porque
    representa el output corporate-aburrido que NO querés. El
    sent_text es el "chosen" porque es lo que VOS realmente
    mandaste — tono Fer, voseo rioplatense, conciso.

    Output schema (TRL DPOTrainer compatible):
        {
            "prompt": str,       # contexto + intro
            "chosen": str,       # sent_text (tu respuesta real)
            "rejected": str,     # bot_draft (la corporate)
            "draft_id": str,     # para stratified split
        }
    """
    contact = item.get("contact_name") or "(sin nombre)"
    convo = _format_conversation(item["original_msgs"])
    bot_draft = item.get("bot_draft") or ""
    sent_text = item.get("sent_text") or ""

    prompt = (
        f"Conversación con {contact}:\n"
        f"{convo}\n\n"
        f"## Tu respuesta:\n"
    )

    return {
        "prompt": prompt,
        "chosen": sent_text,
        "rejected": bot_draft,
        "draft_id": item.get("draft_id", ""),
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


# ── Métricas (BLEU-1 + similarity + preference win rate) ──────────────────


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


def preference_win(pred: str, chosen: str, rejected: str) -> bool:
    """¿La predicción se parece más al chosen que al rejected?

    Métrica única de DPO: queremos que el modelo, cuando genera
    libremente, produzca algo más cercano al sent_text (chosen) que
    al bot_draft (rejected). Usamos similarity como proxy de
    "cercanía". Si chosen y rejected están ambos lejos de pred (caso
    creativo), el modelo devuelve "neutral" → contamos como TIE
    (no win, no loss).

    Returns:
        True si sim(pred, chosen) > sim(pred, rejected).
        False si sim(pred, chosen) <= sim(pred, rejected).
    """
    s_chosen = similarity(pred, chosen)
    s_rejected = similarity(pred, rejected)
    return s_chosen > s_rejected


def evaluate_predictions(
    val_examples: list[dict], predictions: list[str],
) -> dict:
    """Computa métricas held-out + selecciona 5 samples random para print.

    Métricas:
      - bleu1 / sim contra el chosen (sent_text) — qué tan cerca está
        la generación del target preferido.
      - preference win rate — % de val_examples donde sim(pred, chosen)
        > sim(pred, rejected). Métrica directa de la calidad DPO.
    """
    if not predictions:
        return {
            "n_val": 0, "bleu1_mean": 0.0, "sim_mean": 0.0,
            "pref_win_rate": 0.0,
            "samples": [],
        }
    bleu_scores = [
        bleu1(pred, ex["chosen"])
        for pred, ex in zip(predictions, val_examples)
    ]
    sim_scores = [
        similarity(pred, ex["chosen"])
        for pred, ex in zip(predictions, val_examples)
    ]
    pref_wins = [
        preference_win(pred, ex["chosen"], ex["rejected"])
        for pred, ex in zip(predictions, val_examples)
    ]

    rng = random.Random(42)
    sample_idxs = sorted(rng.sample(
        range(len(val_examples)), k=min(5, len(val_examples)),
    ))
    samples = []
    for i in sample_idxs:
        samples.append({
            "rejected_bot_draft": val_examples[i]["rejected"],
            "chosen_sent_text": val_examples[i]["chosen"],
            "prediction": predictions[i],
            "bleu1": bleu_scores[i],
            "sim": sim_scores[i],
            "pref_win": pref_wins[i],
        })

    return {
        "n_val": len(predictions),
        "bleu1_mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "bleu1_min": min(bleu_scores) if bleu_scores else 0.0,
        "bleu1_max": max(bleu_scores) if bleu_scores else 0.0,
        "sim_mean": sum(sim_scores) / len(sim_scores) if sim_scores else 0.0,
        "sim_min": min(sim_scores) if sim_scores else 0.0,
        "sim_max": max(sim_scores) if sim_scores else 0.0,
        "pref_win_rate": sum(pref_wins) / len(pref_wins) if pref_wins else 0.0,
        "samples": samples,
    }


# ── Training (DPO + LoRA via trl + peft) ──────────────────────────────────


def train_dpo(
    train_examples: list[dict], val_examples: list[dict],
    *, out_dir: Path, epochs: int, lr: float, batch_size: int,
) -> dict:
    """Entrena un LoRA adapter sobre el modelo base via DPO.

    Imports lazy adentro de la fn para que `--dry-run` no intente
    cargar transformers/peft/trl. Si falta cualquiera, exit 6 con
    mensaje accionable.

    Returns dict con métricas held-out (bleu1_mean, sim_mean,
    pref_win_rate, samples).
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            f"[error] missing dep para DPO+LoRA training: {exc}\n"
            "  Install with: uv tool install --reinstall --editable '.[finetune]'",
            file=sys.stderr,
        )
        sys.exit(6)

    device = os.environ.get("RAG_FT_DEVICE", "cpu").lower()
    print(
        f"  Loading {DRAFTS_BASE_MODEL} (transformers={transformers.__version__}) "
        f"on device={device} …",
        file=sys.stderr, flush=True,
    )

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(DRAFTS_BASE_MODEL)
    if tokenizer.pad_token is None:
        # Qwen no setea pad_token por default — usamos eos para el
        # padding del collator. Standard recipe.
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  [{time.time()-t_load:.1f}s] tokenizer loaded",
          file=sys.stderr, flush=True)

    t_load = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        DRAFTS_BASE_MODEL,
        torch_dtype="auto",
    )
    print(f"  [{time.time()-t_load:.1f}s] base model loaded",
          file=sys.stderr, flush=True)

    # LoRA config — DPOTrainer lo inyecta vía peft_config. NO necesita
    # un ref_model separado: TRL usa el modelo base congelado como
    # referencia automáticamente cuando hay peft_config.
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )

    # Datasets en formato TRL preference: {prompt, chosen, rejected}.
    train_ds = Dataset.from_list([
        {"prompt": ex["prompt"], "chosen": ex["chosen"],
         "rejected": ex["rejected"]}
        for ex in train_examples
    ])
    val_ds = (Dataset.from_list([
        {"prompt": ex["prompt"], "chosen": ex["chosen"],
         "rejected": ex["rejected"]}
        for ex in val_examples
    ]) if val_examples else None)

    # DPOConfig — superset de TrainingArguments con los kwargs DPO-
    # specific (beta, loss_type, max_length, precompute_ref_log_probs).
    args = DPOConfig(
        output_dir=str(out_dir / "ckpts"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        seed=42,
        fp16=False,
        bf16=False,
        report_to=[],
        use_cpu=(device == "cpu"),
        disable_tqdm=True,
        # ── DPO-specific ────────────────────────────────────────
        beta=DPO_BETA,
        loss_type="sigmoid",  # default DPO loss; "ipo" / "kto_pair" disponibles
        max_length=MAX_TOKENS,
        max_prompt_length=MAX_TOKENS // 2,
        # precompute_ref_log_probs=True ahorra memoria al precomputar
        # log-probs del modelo de referencia (frozen base) ANTES del
        # loop. Importante en CPU/MPS donde el ref forward de cada
        # batch es caro.
        precompute_ref_log_probs=True,
        remove_unused_columns=False,
    )

    # Callback de progreso con flush — el logger default de
    # transformers bufferea hasta el final, dejándonos a ciegas
    # durante runs largos. Esto printea cada step.
    from transformers import TrainerCallback

    class FlushCallback(TrainerCallback):
        def __init__(self):
            self._t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self._t0
            print(
                f"    step {state.global_step}/{state.max_steps} "
                f"({elapsed:.0f}s elapsed, "
                f"{elapsed/max(state.global_step,1):.1f}s/step)",
                file=sys.stderr, flush=True,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            # DPO emite rewards/chosen, rewards/rejected, rewards/margins
            # — métricas únicas del DPO loss, las exponemos.
            if "loss" in logs:
                loss = logs["loss"]
                lr = logs.get("learning_rate", 0)
                ep = logs.get("epoch", 0)
                margin = logs.get("rewards/margins", None)
                acc = logs.get("rewards/accuracies", None)
                extra = ""
                if margin is not None:
                    extra += f" margin={margin:+.3f}"
                if acc is not None:
                    extra += f" acc={acc:.2f}"
                print(
                    f"    loss={loss:.3f} lr={lr:.2e} epoch={ep:.2f}"
                    f"{extra}",
                    file=sys.stderr, flush=True,
                )

    trainer = DPOTrainer(
        model=base_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[FlushCallback()],
    )

    print(
        f"  Training start (DPO, epochs={epochs}, batch={batch_size}, "
        f"train_n={len(train_examples)}, beta={DPO_BETA}, lr={lr})",
        file=sys.stderr, flush=True,
    )
    trainer.train()
    print("  Training done. Saving adapter …", file=sys.stderr, flush=True)
    trainer.save_model(str(out_dir))
    try:
        tokenizer.save_pretrained(str(out_dir))
    except Exception as exc:
        print(f"  [warn] tokenizer.save_pretrained failed: {exc}",
              file=sys.stderr)

    # ── Held-out predictions + métricas ──
    print("  Generating held-out predictions for metrics …",
          file=sys.stderr)
    # `trainer.model` es el peft-wrapped model con el adapter entrenado.
    predictions = generate_predictions(
        trainer.model, tokenizer, val_examples, device,
    )
    metrics = evaluate_predictions(val_examples, predictions)
    return metrics


def generate_predictions(
    model, tokenizer, val_examples: list[dict], device: str,
) -> list[str]:
    """Genera predicciones del modelo para val_examples (held-out).

    Cada sample: input = prompt, output = generación greedy de
    máx 200 tokens. No hacemos sampling ni temperature porque queremos
    reproducibilidad para BLEU/sim/pref_win mediciones.
    """
    import torch
    model.eval()
    # Detectar device del modelo. En MPS / CUDA, el tokenizer retorna
    # tensores en CPU por default → hay que .to(model.device) antes de
    # generate() para evitar `RuntimeError: Placeholder storage has not
    # been allocated on MPS device`. Bug observado el 2026-05-01 con
    # transformers 5.1.0 + MPS en M3 Max.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    preds: list[str] = []
    with torch.no_grad():
        for ex in val_examples:
            inp = tokenizer(
                ex["prompt"], return_tensors="pt", truncation=True,
                max_length=MAX_TOKENS - 200,
            )
            inp = {k: v.to(model_device) for k, v in inp.items()}
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
    print(f"  Total rows (approved_editar + rejected): "
          f"{stats['n_total_rows']}", file=sys.stderr)
    print(f"  Gold preference pairs (approved_editar):  "
          f"{stats['n_gold']}", file=sys.stderr)
    print(f"  Rejected rows skipped (no chosen):        "
          f"{stats.get('n_rejected_skipped', 0)}", file=sys.stderr)
    print(f"  Review-only rows (extra_json flag):       "
          f"{stats['n_review_only_total']}  "
          f"({stats['review_only_ratio'] * 100:.1f}%)",
          file=sys.stderr)
    if exclude_review_only:
        print(f"  Excluded by --exclude-review-only:        "
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
        f"  Similarity (char-level, vs chosen) mean={metrics['sim_mean']:.3f}  "
        f"min={metrics['sim_min']:.3f}  max={metrics['sim_max']:.3f}",
        file=sys.stderr,
    )
    print(
        f"  Preference win rate (pred more like chosen than rejected): "
        f"{metrics['pref_win_rate'] * 100:.1f}%",
        file=sys.stderr,
    )
    print("\n== 5 random held-out samples ==", file=sys.stderr)
    for i, sample in enumerate(metrics.get("samples", []), start=1):
        win_marker = "✓" if sample.get("pref_win") else "✗"
        print(f"\n  --- Sample {i} {win_marker} ---", file=sys.stderr)
        print(f"  rejected (bot_draft):  {sample['rejected_bot_draft'][:200]}",
              file=sys.stderr)
        print(f"  chosen (sent_text):    {sample['chosen_sent_text'][:200]}",
              file=sys.stderr)
        print(f"  prediction:            {sample['prediction'][:200]}",
              file=sys.stderr)
        print(f"  bleu1={sample['bleu1']:.3f}  sim={sample['sim']:.3f}  "
              f"pref_win={sample['pref_win']}",
              file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune del modelo de drafts WhatsApp via DPO + LoRA",
    )
    # DPO con LoRA converge en menos epochs que SFT — default 1 (vs 3
    # del SFT viejo). Más epochs sin más data overfit-prone.
    ap.add_argument("--epochs", type=int, default=1)
    # 5e-6 es la recipe TRL para DPO+LoRA. SFT viejo usaba 1e-4 — DPO
    # con LoRA necesita lr ~10x más bajo porque el contrastive loss
    # tiene gradients más fuertes que cross-entropy.
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size por device (default 4 para CPU/MPS).")
    ap.add_argument("--exclude-review-only", action="store_true",
                    help="Filtrar rows con extra_json.review_only=true. "
                         "Default: incluir (signal real igual).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build pairs + report stats; NO entrena.")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Cap del training set después del split "
                         "(debug/perf). Default: usar todo.")
    args = ap.parse_args()

    print("== fine-tune drafts WhatsApp (DPO + LoRA) ==", file=sys.stderr)
    print(f"  Base model: {DRAFTS_BASE_MODEL}", file=sys.stderr)
    print(f"  Adapter dir: {DRAFTS_FT_ADAPTER_DIR}", file=sys.stderr)

    data = fetch_draft_pairs(exclude_review_only=args.exclude_review_only)
    print_stats(data["stats"], exclude_review_only=args.exclude_review_only)

    # Build DPO triplets desde gold (approved_editar). Las rows
    # 'rejected' (sin chosen alternativo) ya se descartaron en
    # fetch_draft_pairs.
    examples = [build_dpo_example(item) for item in data["gold"]]
    print(f"\n  DPO training examples: {len(examples)} preference pairs",
          file=sys.stderr)

    if len(examples) < MIN_PAIRS:
        print(
            f"\n[error] datos insuficientes: {len(examples)} pares < "
            f"{MIN_PAIRS} mínimo. "
            f"Esperá más feedback (más decisions /editar en RagNet).\n"
            f"  Tip: `rag drafts stats` para ver el conteo actual.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] exiting before training. Pairs OK para entrenar.",
              file=sys.stderr)
        return

    train_examples, val_examples = split_train_val(examples, val_frac=VAL_FRAC)
    print(f"\n  Split: train={len(train_examples)} val={len(val_examples)}",
          file=sys.stderr, flush=True)
    if args.max_train_samples and len(train_examples) > args.max_train_samples:
        rng_cap = random.Random(42)
        rng_cap.shuffle(train_examples)
        train_examples = train_examples[:args.max_train_samples]
        val_cap = max(20, args.max_train_samples // 10)
        if len(val_examples) > val_cap:
            val_examples = val_examples[:val_cap]
        print(f"  CAPPED via --max-train-samples: train={len(train_examples)} "
              f"val={len(val_examples)}", file=sys.stderr, flush=True)

    DRAFTS_FT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Training → {DRAFTS_FT_ADAPTER_DIR}", file=sys.stderr)
    t0 = time.time()
    metrics = train_dpo(
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
            "method": "dpo+lora",
            "base_model": DRAFTS_BASE_MODEL,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "dpo_beta": DPO_BETA,
            "exclude_review_only": args.exclude_review_only,
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "n_gold": len(examples),
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
        print(f"  [warn] writing ft_meta.json failed: {exc}",
              file=sys.stderr)

    print(
        "\n  Adapter saved. Para activar el endpoint preview con este "
        "adapter:\n"
        "    export RAG_DRAFTS_FT=1\n"
        "  El listener TS NO usa este modelo (sigue con qwen2.5:7b en prod). "
        "Solo el endpoint /api/draft/preview lo carga para A/B manual.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
