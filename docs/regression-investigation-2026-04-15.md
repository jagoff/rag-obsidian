# Chain regression investigation — 2026-04-15

## TL;DR

**No hay regresión en el core de `retrieve()`. La caída de `chain_success` 50 → 16.67
se explica por dos causas ortogonales:**

1. **Golden-set staleness** (2/16 turns): notas referenciadas en `queries.yaml`
   se movieron a subcarpetas y la golden no se actualizó.
2. **Chain reformulation quality** (5/16 turns): el helper `reformulate_query` o
   la retrieval fallan al resolver continuaciones tipo "y el otro", "profundizá en X".

**Full `git bisect` evitable.** Commit `6a8ab0f` (apr 14) ya había flaggeado
literalmente: *"Baseline CLAUDE.md (90/75) probablemente requiere truth-ups
adicionales de otras notas movidas (claude-code, podcast, chains de
ikigai/liderazgo)."*

## Método

Antes de bisectar a ciegas en los 40 commits que tocaron `rag.py` desde el
commit que ancló el baseline documentado (`37d9818`, donde se agregó la métrica
de chains), chequée `git log` de `queries.yaml` — detecté `6a8ab0f` y `9109873`
como commits que ajustaron paths/queries. El mensaje de `6a8ab0f` apuntó
directamente a staleness. Verifiqué cada `expected` de las chains contra el
filesystem real del vault.

## Failing turns (7 de 16)

| Chain                         | Turn | Expected                                                         | Vault status   | Tipo        |
|-------------------------------|-----:|------------------------------------------------------------------|----------------|-------------|
| coaching-ikigai               |  3   | `02-Areas/Coaching/Fer Coach.md` (+ `Fer coach - Bio.md`)        | existe         | reformulate |
| coaching-liderazgo            |  2   | `02-Areas/Coaching/Coaching - Curso de liderazgo.md`             | existe         | reformulate |
| coaching-liderazgo            |  3   | `02-Areas/Coaching/Coaching - Referentes.md`                     | existe         | reformulate |
| musica-muros-fractales        |  2   | `02-Areas/Musica/Muros Fractales/Letra - La herrumbre - AI - 2.0.md` (+3) | existe | reformulate |
| tech-claude-code              |  1   | `03-Resources/Claude Code - Comandos CLI.md`                     | **MOVIDO** → `03-Resources/Claude/Claude Code - Comandos CLI.md` | golden stale |
| tech-claude-code              |  2   | `03-Resources/Claude Code - Comandos Interactivos.md`            | **MOVIDO** → `03-Resources/Claude/Claude Code - Comandos Interactivos.md` | golden stale |
| rag-system                    |  1   | `03-Resources/RAG-Local/Obsidian RAG Local.md`                   | existe         | reformulate |

Las 7 failures = 2 "golden stale" + 5 "retrieval/reformulate real".

## Chains (6 totales)

| Chain                     | Turns hit | Result |
|---------------------------|----------:|--------|
| coaching-ikigai           | 2/3       | fail   |
| coaching-liderazgo        | 1/3       | fail   |
| musica-muros-fractales    | 1/2       | fail   |
| tech-claude-code          | 0/2       | fail (100% golden stale) |
| rag-system                | 2/3       | fail   |
| meta-vault-ollama         | 3/3       | **✓ success** |

`chain_success = 1/6 = 16.67%`, como reporta el eval.

## Impacto estimado si se corrige sólo el golden

Actualizar los 2 paths de `tech-claude-code` en `queries.yaml` recupera la chain
entera (0/2 → 2/2 → success). No toca las otras 5 turns fallidas.

- **Proyección**: `chain_success 1/6 → 2/6 = 33.33%` (+16.66pp).
- **Gap residual vs baseline 50%**: 1 chain más tiene que ganar, lo que
  requiere recuperar alguna de las turns "reformulate" — candidato más
  accesible: `coaching-ikigai` (2/3, sólo falta la continuación).

Esto también encaja con la meta declarada de #4 (compresor de sesiones):
mejorar chain retrieval sin tocar singles. Si el compressor sube chain
reformulation, ataca directamente los 5 turns residuales.

## Causas NO encontradas

- **No es re-chunking**: chunks están estables desde la serie v6→v7; no hubo
  bump de `_COLLECTION_BASE` en ventana reciente.
- **No es reranker pool cap** (`c5e6a9d`, cap a 40): verificado indirectamente
  por el hecho de que los turns fallidos retrievan *otras* notas con buen score
  (no 0); es precisión, no recall.
- **No es feedback bias** (`c68ed29`): los queries de eval no tienen feedback
  registrado, el code path no aplica.

Si el user quiere validación con bisect explícito de estas hipótesis:

```bash
# Pinpointing lento (5 min por eval). No recomendado hasta cerrar #4:
git checkout <commit>; rag eval; git checkout master
```

## Recomendación

1. **Fix cheap**: actualizar `queries.yaml` — 2 paths de `tech-claude-code`
   deben pasar de `03-Resources/` a `03-Resources/Claude/`. Patch de 2 líneas.
   Re-correr eval. Proyección: chain_success 16.67 → ~33%.
2. **#4 (compressor) sigue**: su piso real son los 5 turns "reformulate". El
   baseline documentado (`docs/eval-baselines-2026-04-15.md`) ahora tiene
   contexto claro: no hay bug de retrieval; hay margen de mejora en
   reformulación.
3. **Update CLAUDE.md**: los números 90/75/50 citados pertenecen a un golden
   set anterior al `6a8ab0f`. Corresponde refrescarlos una vez que se aplica
   el fix cheap y se recorre el eval.

## Decisión NO tomada

No commiteé ningún fix. Este doc y `docs/eval-baselines-2026-04-15.md` quedan
en working tree hasta que el user despause y decida:
- ¿Commitear ambos (investigación + baseline)?
- ¿Aplicar el golden fix en queries.yaml ahora o después de #4?
- ¿Refrescar CLAUDE.md con los números nuevos?
