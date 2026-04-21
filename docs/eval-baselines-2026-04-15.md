# Eval baseline — 2026-04-15

Lock-in pre-track #4 (compresión de sesiones). Snapshot del comportamiento
actual de `retrieve()` + `reformulate_query()` para que cualquier cambio futuro
pueda medirse contra estos números.

## Contexto

- **Commit**: master @ `1ae12b0` (post `feat(insights)` + `fix(followup)` + design doc cross-source).
- **Comando**: `rag eval` ejecutado contra `queries.yaml` (golden set checked-in).
- **Modelos**: `bge-m3` (embed), `BAAI/bge-reranker-v2-m3` (cross-encoder, MPS+fp16),
  `qwen2.5:3b` (helper para reformulate), `command-r:latest` (chat — no se usa
  acá porque eval es retrieval-only).
- **Index**: colección `obsidian_notes_v7` (vigente al 2026-04-15; hoy está en `obsidian_notes_v11`). Vault personal del user; size != CI.

## Singles

```
hit@5 82.76%  ·  MRR 0.741  ·  recall@5 82.76%  ·  n=29
```

## Chains

```
hit@5 56.25%  ·  MRR 0.385  ·  recall@5 56.25%  ·  chain_success 16.67%
turns=16, chains=6
```

## Observación: gap vs baseline documentado en `CLAUDE.md`

El `CLAUDE.md` cita un baseline anterior (probablemente del momento del v6→v7
bump): `singles hit@5 90.48% · MRR 0.786`, `chains hit@5 75.00% · MRR 0.656 ·
chain_success 50.00%`.

Diff actual vs documentado:

| Métrica            | Doc baseline | Actual  | Δ          |
|--------------------|--------------|---------|------------|
| Singles hit@5      | 90.48%       | 82.76%  | -7.72pp    |
| Singles MRR        | 0.786        | 0.741   | -0.045     |
| Chains hit@5       | 75.00%       | 56.25%  | -18.75pp   |
| Chains MRR         | 0.656        | 0.385   | -0.271     |
| chain_success      | 50.00%       | 16.67%  | -33.33pp   |

Causas plausibles del drop:

- Singles: el set creció de 21 → 29 queries (queries más duras agregadas
  en commits posteriores). Comparación no apples-to-apples.
- Chains: mismo 16 turns / 6 chains, así que la regresión es real. Posibles
  vectores: re-chunking en algún commit reciente, drift de `qwen2.5:3b` en
  reformulate, o cambio de prompt no medido.

**No bloquea #4.** Estos números son la referencia contra la que el compressor
debe medirse — el objetivo del compressor es subir chains sin bajar singles,
arrancando desde acá.

## Cómo re-correr

```bash
cd /Users/fer/repositories/obsidian-rag
rag eval
```

Tarda ~5 min en M-series con LLM local. Output completo es estable suficiente
para diff (los `top-k` mostrados son determinísticos en el reranker, no en el
helper).
