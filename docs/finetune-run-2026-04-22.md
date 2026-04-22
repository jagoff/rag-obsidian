# Fine-tune reranker — run del 2026-04-22 (noisy data, 3 epochs)

**Outcome**: ✗ gate rechazó (chains −3.3pp). Model kept at
`~/.cache/obsidian-rag/reranker-ft-20260422-182127/` para inspección.

**Config**:

- `scripts/finetune_reranker.py --epochs 3 --lr 2e-5 --batch-size 8`
- `RAG_FINETUNE_MIN_CORRECTIVES=0` (override del gate pre-training)
- `RAG_FT_DEVICE=cpu`
- Base model: `BAAI/bge-reranker-v2-m3`

**Data**:

- 55 feedback positives (rating=1), 0 con `corrective_path`
- 347 training pairs tras build (101 pos + 246 neg; 75 skipped unreadable)
- Split 80/20: train=287, val=60

**Training**:

- Runtime: 615.9s (~10 min en CPU)
- Loss curve: 0.96 → 0.13 (converge en epoch 2, overfitting claro en epoch 3)
- Validation margin **+0.455** (pos mean 0.515 vs neg mean 0.060) — el modelo
  aprendió muy bien lo que le dimos.

**Eval gate**:

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Singles hit@5 | 71.67% | 71.67% | 0.0pp |
| Singles MRR | 0.681 | 0.657 | −0.024 |
| Chains hit@5 | **86.67%** | **83.33%** | **−3.3pp** |
| Chains MRR | 0.790 | 0.794 | +0.004 |

Gate criterion: no regression en ambos hit@5 (singles_ok=True, chains_ok=False).
→ NOT promoting.

## Análisis

El val margin altísimo (+0.455) confirma que el fine-tune **aprendió bien lo que
le dimos**. El problema es que **lo que le dimos era ruidoso**: cada turno con
rating=1 tiene ~4 paths como positivos cuando en realidad sólo uno era el
golden. El modelo aprendió a empujar hacia arriba también a los chunks
no-golden, lo que tumba los chains (queries multi-hop que dependen del chunk
exacto).

Singles hit@5 bit-idéntico sugiere que el top-1 rerank es estable — la
regresión ocurre en rank 2-5 donde los chunks no-golden del turn positivo
compiten con chunks relevantes de otras fuentes.

## Conclusión firme

**Sin `corrective_path` limpios, el fine-tune no supera baseline con esta
config.** 3 epochs (vs el run previo con 1 epoch) no alcanzan a romper el techo
de −3.3pp chains — la causa raíz es la señal ruidosa, no el undertraining.

## ¿Qué validó este run (pese a fallar)?

1. ✅ **Eval gate funcional E2E**: detectó la regresión y NO promovió. Safety
   net validada — el sistema es seguro aunque la data futura siga ruidosa.
2. ✅ **Pipeline completo funciona**: `_fetch_feedback_pairs` + `_build_training_pairs`
   + `_mine_hard_negatives` + `CrossEncoderTrainer` + subprocess `rag eval` +
   gate → todos operan sin crashes.
3. ✅ **Performance**: 10 min training CPU para 287 pairs × 3 epochs es razonable
   en la 36 GB unified memory sin tocar MPS (evita conflicto con ollama pineado).

## Próximo intento

Re-correr cuando haya ≥20 rows con `corrective_path` en `rag_feedback`. Generar
data:

- `rag chat` + 👎 en turnos malos → prompt pide el path correcto
- Web UI: picker de corrective_path en thumbs-down (commit `33ed3f0`)
- Skill `rag-feedback-harvester` para labeleo batch sobre queries low-confidence

Monitoreo del count:

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/ragvec.db \
  "SELECT COUNT(*) FROM rag_feedback
   WHERE json_extract(extra_json, '\$.corrective_path') IS NOT NULL
     AND json_extract(extra_json, '\$.corrective_path') <> ''"
```

Cuando pase de 20:

```bash
python scripts/finetune_reranker.py --epochs 2 --lr 2e-5
# (2 epochs > 3 — el loss convergió a 0.22 en epoch 2; epoch 3 es overfit)
```

## Limpieza pendiente

Modelos rechazados en cache, **4.2 GB total**, ninguno symlinkeado:

```bash
rm -rf ~/.cache/obsidian-rag/reranker-ft-20260422-124112/  # 2.1 GB (run anterior, 1 epoch)
rm -rf ~/.cache/obsidian-rag/reranker-ft-20260422-182127/  # 2.1 GB (este run, 3 epochs)
```

El usuario decide borrar manualmente — el script los deja a propósito para
inspección.
