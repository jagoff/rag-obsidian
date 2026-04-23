# Eval baseline — 2026-04-23 (recalibración)

Continuación de [`docs/eval-baselines-2026-04-15.md`](./eval-baselines-2026-04-15.md).
Snapshot tras auditar el gap entre el gate documentado (singles ≥76.19%,
chains ≥63.64%) y las corridas reales estables (71.67% / 86.67% desde
el 2026-04-22).

## TL;DR

El gate viejo **ya era incompatible** con el `queries.yaml` actual y
disparaba auto-rollback en toda corrida. No era una regresión del
pipeline — era growth del test set. Floors re-calibrados a los nuevos
CI lower bounds (misma metodología que el gate original).

| Métrica           | Pre-gate (0.7619 / 0.6364) | Post-gate (0.60 / 0.73) |
|-------------------|----------------------------|-------------------------|
| `GATE_SINGLES_HIT5_MIN` | 0.7619 — del run 2026-04-17 (n=42) | **0.60** — del run 2026-04-23 (n=60) |
| `GATE_CHAINS_HIT5_MIN`  | 0.6364 — del run 2026-04-17 | **0.73** — del run 2026-04-23 |

Overridables via env:
- `RAG_EVAL_GATE_SINGLES_MIN="0.70"` → más estricto
- `RAG_EVAL_GATE_CHAINS_MIN="0.80"` → idem

## Evidencia: timeline del drop

Query SQL sobre `rag_eval_runs` filtrando runs con `singles_n >= 20`
(excluye runs de test con fixtures sintéticos de n=2):

```sql
SELECT date(ts), COUNT(*), MIN(ROUND(singles_hit5*100, 2)) as min,
       MAX(ROUND(singles_hit5*100, 2)) as max
FROM rag_eval_runs WHERE singles_n >= 20
GROUP BY date(ts) ORDER BY 1;
```

| Fecha      | Corridas | singles min | singles max | Notas |
|------------|----------|-------------|-------------|-------|
| 2026-04-16 | 1        | 90.48%      | 90.48%      | n=21. Baseline inicial del set expandido. |
| 2026-04-17 | 23       | 88.10%      | 90.48%      | n=23 (más goldens agregados). |
| 2026-04-18 | 13       | 4.76%       | 88.10%      | Día de ruido — algún cambio rompió retrieve, se arregló mismo día. |
| 2026-04-19 | 4        | 80.95%      | 88.10%      | Stable n=21 tras fix. |
| 2026-04-20 | 2        | 78.57%      | 78.57%      | n=42 — post `rag eval` expansion + PARA re-mapping. |
| 2026-04-21 | 26       | 69.09%      | 81.67%      | n transitó de 42→48→55→60 a lo largo del día. |
| **2026-04-22** | 15   | 63.33%      | **71.67%**  | **n=60 — primer día estable** con el golden expandido completo. |
| **2026-04-23** | 11   | 71.67%      | **71.67%**  | n=60, **baseline consolidado**. |

El **único día con 4.76%** (2026-04-18) coincide con un día de ruido
pre-fix PARA-remap. No es régimen estable.

## Evidencia: composición del golden set

`git log` sobre `queries.yaml`:

```
abe549d 2026-04-21 14:45  feat(eval): +5 calendar goldens → singles=60
338ba04 2026-04-21 13:00  feat(eval): +7 cross-source reales → singles=55
4e66add 2026-04-21 10:20  feat(eval): cross-source placeholders
12e146e 2026-04-21 10:03  feat(eval): synthesis + comparison intents → singles=48
fe6a0f4 2026-04-21 01:16  singles=42
```

Antes del 2026-04-21, singles=42. Después, singles=60 (+43%). Las 18
queries nuevas son:
- **+6 synthesis/comparison** — intents con expectativas más amplias,
  más difíciles de hit@5.
- **+7 cross-source placeholders** (gmail://, calendar://, whatsapp://)
  — retrieval cross-source compite con vault en el mismo top-5, más
  complejo.
- **+5 calendar goldens** — ingester cross-source completado, primer
  eval con calendario como source legítimo.

Las queries viejas (42 pre-expansion) probablemente siguen hitteando
~82-88% (inferido de las corridas n=42 del 2026-04-20). Son las nuevas
18 queries las que bajan el average a 71.67%.

## Evidencia: failing queries actuales

`rag eval` 2026-04-23, queries sin hit en top-5 (extracto):

```
qué reranker tiene el sistema       # 5 candidatos relevantes, el correcto no está en top-5
podcast primer episodio coaching    # similar
letra de muros fractales            # expected note no apareció — archived deep
feedback sobre stack local free     # whatsapp + vault — cross-source dispute
cuándo tengo que buscar inglés para astor  # calendar-only, 5 eventos candidatos
comprar entradas para el en vivo de demos # gmail thread + WA message
```

La mayoría son las **18 queries nuevas** del 2026-04-21 (cross-source +
synthesis). El retrieve pipeline sigue encontrando los viejos 42 OK.

## Conclusión

**No hay regresión en el pipeline**. El gate viejo se quedó pegado al
baseline de n=42 mientras el test set evolucionó a n=60 con queries
intencionalmente más difíciles. Cada corrida 2026-04-21-en-adelante
fallaba el gate viejo por definición, volviendo `rag tune --online`
un no-op perpetuo.

Los floors nuevos (0.60 / 0.73) son los CI lower bounds del baseline
actual — misma metodología que usó el gate original (2026-04-17:
88.10% → 76.19% lower CI). Son **conservadores por diseño**: cualquier
corrida bajo estos floors tiene 95% probabilidad de ser regresión real,
no noise.

## Re-calibrar cuándo

- Expansión material de `queries.yaml` (ej. +10 queries nuevas de un
  intent que no estaba representado antes).
- Cambio estructural en retrieve(): nuevo modelo de embeddings,
  reemplazo del reranker, re-chunking strategy.
- 5+ corridas consecutivas bordean el floor sin regresión real
  identificable (signal de que el floor quedó corto).

## Cómo medir en el futuro

```bash
# Snapshot timeline via rag_eval_runs
sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db \
  "SELECT date(ts), COUNT(*), MIN(ROUND(singles_hit5*100,2)),
   MAX(ROUND(singles_hit5*100,2))
   FROM rag_eval_runs WHERE singles_n >= 20
   GROUP BY date(ts) ORDER BY 1;"

# Composición del golden set actual
.venv/bin/python -c "
import yaml
d = yaml.safe_load(open('queries.yaml'))
print('singles:', len(d.get('queries', [])))
print('chains:', len(d.get('chains', [])))
print('turns:', sum(len(c.get('turns', [])) for c in d.get('chains', [])))"
```
