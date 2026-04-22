# Miner `scripts/export_training_pairs.py` — análisis del JSONL

**Fecha**: 2026-04-22
**Miner**: commit [`5f33d44`](https://github.com/jagoff/rag-obsidian/commit/5f33d44)
**Trigger**: tras validar el fine-tune con data noisy (commit `2ebeee9`), el miner nuevo
debería proveer signal cualitativamente superior para el próximo intento.

## TL;DR

- **176 pairs** extraídos de 48 queries únicas (60 días).
- 100% tienen ≥1 hard-neg, 91% ≥3, **74% ≥5**.
- Avg 6.1 negs/pair. Total 1067 negativos.
- **Todos los hard-negs vienen de `impression` events reales**: son paths
  que el retriever SURFACED al user y el user NO seleccionó. Contraste
  mucho más crisp que el run anterior (negs re-retrievados post-hoc).
- **Próximo paso**: abp2vvvw tarea (B) — wrapper `lee_jsonl()` en
  `scripts/finetune_reranker.py` que consuma este JSONL.

## Comparación vs run anterior del finetune

| Métrica | Run 2 (feedback direct) | Miner JSONL |
|---|---|---|
| Pairs totales | 347 | **176** |
| Positivos únicos | 101 (duplicados por path) | **72** (deduped) |
| Negativos totales | 246 | **1067** |
| Neg/Pos ratio | 2.4 : 1 | **6.1 : 1** |
| Origen negativos | `retrieve()` post-hoc | Impressions reales del turn |
| Queries distintas | ~55 | **48** |
| Corpus drift risk | Alto (re-retrieve usa corpus actual) | **Bajo** (impressions fijas en el tiempo) |

Números absolutos bajaron pero **calidad subió** — cada pair está más
"ganado" en señal. El ratio 6.1:1 (vs 2.4:1) significa que el modelo
tiene 2.5× más ejemplos contrastivos por positive.

## Distribución por source

```
rating_pos                     176
```

Signals implícitos (`behavior_copy`, `behavior_open`, `behavior_save`)
todavía NO aparecen. Causa: los 21 `open` events en `rag_behavior` no
tienen `original_query_id` llenado (lo vimos en el análisis previo).
La instrumentación de `copy` events arrancó hoy (commit `db2a169`) —
en 7-14 días de uso normal debería generar signal masiva según el
commit message del miner.

## Distribución por folder PARA (positives)

```
03-Resources      69   (+ WhatsApp + web articles)
04-Archive        58
02-Areas          32
01-Projects       13
05-Reviews         3
00-Inbox           1
```

Heavy weight en 03-Resources (WhatsApp chats + recursos externos) y
04-Archive (notas viejas). Eso refleja el patrón real de queries del
user: buscar referencias históricas, conversaciones pasadas. El fine-tune
debería aprender preferencias en ese tipo de corpus.

## Muestra representativa

```json
{
  "query": "Que tenes de Grecia?",
  "positive": "03-Resources/WhatsApp/Maria/2026-03.md",
  "negatives": [
    "03-Resources/WhatsApp/RagNet/2026-04.md",
    "03-Resources/WhatsApp/Maria/2026-02.md",
    "04-Archive/Grecia/Grecia - Carta campamento escolar.md",
    "04-Archive/Grecia/Grecia - Viaje 1ero.md",
    "04-Archive/Grecia/Pagar escuela Grecia.md"
  ],
  "source": "rating_pos",
  "turn_id": "f9801c5af035",
  "ts": "2026-04-15T17:02:46"
}
```

Observación: los hard-negs son **semánticamente cercanos** (mencionan
Grecia, WhatsApp Maria, cosas de pagar escuela) pero NO son el golden
elegido. Exactly el tipo de contraste que un cross-encoder puede
aprender — baseline bge-reranker-v2-m3 probablemente los ranking
similarmente hoy, pero con fine-tune puede separarlos.

## Cuándo integrar al finetune

**Decisión de abp2vvvw** (tarea B de su summary actual). El peer
escribió en el commit message del miner:

> Yo no lo hago ahora porque el finetune es zona que el peer acaba
> de validar con su safety-net (rollback −3.3pp chains) y no quiero
> interferir.

Acuerdo: dejamos el miner como export standalone. La integración
natural sería:

1. Agregar flag `--input-jsonl PATH` a `scripts/finetune_reranker.py`.
2. Cuando el flag está, skipear `_fetch_feedback_pairs()` y leer
   directo del JSONL, convirtiendo cada row a la shape de `_build_training_pairs`.
3. Bypass del gate `RAG_FINETUNE_MIN_CORRECTIVES` (el gate es para
   corrective_paths explícitos; desde JSONL ya hay signal rica sin ese
   bucket).
4. Eval gate normal sigue aplicando — promoción solo si ≥ baseline.

## Comando exacto para re-correr el fine-tune con JSONL (cuando esté integrado)

```bash
# Regenerar el JSONL (inputs pueden cambiar día a día):
python scripts/export_training_pairs.py --days 60 -o training-pairs.jsonl

# Correr fine-tune con el JSONL:
python scripts/finetune_reranker.py \
  --input-jsonl training-pairs.jsonl \
  --epochs 2 \
  --lr 2e-5
```

Esperable: con 6.1× más hard-negs de calidad y 0.7× los pairs totales,
el modelo debería converger con señal más fuerte. Si la loss cae
suavemente en 2 epochs sin overfitting (vs epoch 2 del run noisy que
mostró 0.22), y el val margin se mantiene positivo, el eval gate
tiene chances reales de pasar.

## Monitoreo del crecimiento del JSONL

```bash
# Cada vez que hay sesión de harvest / backfill / thumbs en chat, el
# JSONL crece. Ejecutar periódicamente:
python scripts/export_training_pairs.py --stats-only --days 60

# Esperable en 2-4 semanas:
#   behavior_copy / behavior_open / behavior_save con double-digit counts
#   (hoy 0 — copy tracking arrancó HOY con db2a169, y open no tiene
#    original_query_id linkeado pre-hoy).
#   corrective (rescatados vía `rag feedback backfill`) con double-digit
#   counts (hoy 0 — comando nuevo commit 8931296).
```

## Rollback

Si el JSONL nunca vale, el miner queda como script standalone y no
rompe nada:

```bash
git rm scripts/export_training_pairs.py tests/test_export_training_pairs.py
```

Pero dado que el análisis muestra signal de calidad superior al run
previo, la recomendación es ir adelante con la integración cuando
abp2vvvw la tome.
