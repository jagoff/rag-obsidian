# Brief — fine_tunning panel audit + wiring (2026-05-04)

Sos un remote agent en cloud. **No tenés acceso al Mac del user**, sólo al repo `jagoff/rag-obsidian` (clonado). NO podés leer `~/.local/share/obsidian-rag/ragvec/telemetry.db` — proponé queries SQL para que el user las corra a mano. Respondé al user en **español rioplatense voseo**.

## Contexto

El 2026-05-01 shippeamos el panel `/fine_tunning` (master commits `4d3df8a` wave-1 + `316c612` wave-2 + `18b959a` fix WA + merge `a8772b2`). Es un panel web FastAPI (`web/server.py`) con frontend `web/static/fine_tunning.html` que muestra una cola unificada para puntuar 👍/👎 desde 6 streams: `retrieval`, `retrieval_answer`, `brief`, `anticipate`, `draft_wa`, `proactive_push` (+ `whatsapp_msg` infra-ready pero hoy 0 items).

Todo rating cae a `rag_ft_panel_ratings` en `telemetry.db`. Snoozes a `rag_ft_active_queue_state`.

**Loop abierto que cerrás vos**: thumbs sobre `stream='retrieval_answer'` (queries CON respuesta del modelo) hoy quedan SOLO en `rag_ft_panel_ratings`. El ranker-vivo nightly (`rag tune --online --apply` cron `com.fer.obsidian-rag-online-tune`) lee únicamente `rag_feedback`. Resultado: el user puede puntuar 100 respuestas en el panel y NO mover el ranker. Hay que bridgear.

Para más contexto leé `CLAUDE.md` (sección "fine_tunning" si existe; si no, grep "fine_tunning" en todo el archivo).

## Task 1 — Audit adoption (read-only, NO edits)

Generá un reporte markdown con queries SQL para que el user las corra contra `telemetry.db` (vos NO podés ejecutarlas). Path destino:

```
04-Archive/99-obsidian-system/99-AI/system/fine-tunning-audit-2026-05-04.md
```

Usá `Write` tool — **NO commitees** este archivo (vive sólo en el filesystem del workspace temporal del agent; el user lo recupera del session output).

Estructura del reporte:

1. **Summary ejecutivo arriba** — qué medir, qué mirar.
2. **Total ratings desde 2026-05-01 por stream**:
   ```sql
   SELECT stream, rating, COUNT(*)
   FROM rag_ft_panel_ratings
   WHERE ts >= '2026-05-01'
   GROUP BY stream, rating
   ORDER BY 1, 2;
   ```
   + interpretación esperada (qué quiere ver el user).
3. **Distribución temporal**:
   ```sql
   SELECT date(ts) AS dia, stream, COUNT(*)
   FROM rag_ft_panel_ratings
   WHERE ts >= '2026-05-01'
   GROUP BY 1, 2
   ORDER BY 1 DESC, 3 DESC;
   ```
   detectar si fue burst inicial o uso sostenido.
4. **Cobertura del comment textarea**:
   ```sql
   SELECT
     SUM(CASE WHEN comment IS NOT NULL AND length(comment) > 0 THEN 1 ELSE 0 END) AS con_comment,
     SUM(CASE WHEN comment IS NULL OR length(comment) = 0 THEN 1 ELSE 0 END) AS sin_comment
   FROM rag_ft_panel_ratings
   WHERE rating = -1 AND ts >= '2026-05-01';
   ```
5. **Top 5 items rateados múltiples veces** (debug signal de UI rota o duplicates):
   ```sql
   SELECT stream, item_id, COUNT(*) AS n_ratings, MIN(label) AS sample_label
   FROM rag_ft_panel_ratings
   WHERE ts >= '2026-05-01'
   GROUP BY stream, item_id
   HAVING COUNT(*) > 1
   ORDER BY n_ratings DESC
   LIMIT 5;
   ```
6. **Comparación cuantitativa panel vs `rag_feedback` directo** (chat thumbs):
   ```sql
   SELECT 'panel' AS origen, rating, COUNT(*)
     FROM rag_ft_panel_ratings WHERE ts >= '2026-05-01' GROUP BY rating
   UNION ALL
   SELECT 'rag_feedback' AS origen, rating, COUNT(*)
     FROM rag_feedback WHERE ts >= '2026-05-01' GROUP BY rating;
   ```
   ¿el panel está absorbiendo signal nuevo o redundando con el chat?

Cada sección lleva la query + interpretación esperada de cada resultado posible.

## Task 2 — Wiring retrieval_answer → rag_feedback (branch + push, NO merge)

Objetivo: cuando el user puntúa un item de `stream='retrieval_answer'` en el panel, además de la row a `rag_ft_panel_ratings`, persistir row equivalente a `rag_feedback` para que `rag tune --online` nightly lo levante.

**Pasos**:

1. `git checkout -b feat/fine-tunning-wire-retrieval-answer` desde `master`.
2. Editar `web/server.py` handler `fine_tunning_rate` (alrededor de líneas 23029-23080). Cuando `req.stream == 'retrieval_answer'`:
   - Leer la row de `rag_queries` con `id = int(req.item_id)` capturando `q`, `paths_json`, `top_score`, `ts`, `session`.
   - Llamar `record_feedback()` (existe en `rag/__init__.py` — buscalo con `grep -n "def record_feedback" rag/__init__.py`) con shape:
     ```python
     record_feedback(
         q=row.q,
         paths=json.loads(row.paths_json or '[]'),
         rating=req.rating,
         scope='turn',
         reason=req.comment,
         ts=row.ts,
         source='fine_tunning_panel',
     )
     ```
     (ajustá kwargs a la signature real si difiere — el grep te lo dice).
   - **Idempotencia**: si ya existe row en `rag_feedback` con misma `q` + `ts` cercano (< 1 día), skip silent.
   - Wrap todo en `try / except` con fallback a `_silent_log('fine_tunning_rate_bridge_to_feedback_failed', str(exc))` para que NUNCA tumbe el endpoint principal.

3. Sumar **3 tests** a `tests/test_fine_tunning_panel.py` (NO crear archivo nuevo):
   - `test_rate_retrieval_answer_pos_writes_both_tables`: rate retrieval_answer +1 → row en `rag_ft_panel_ratings` AND row en `rag_feedback` con `rating=+1`.
   - `test_rate_retrieval_answer_neg_with_comment_persists_reason`: rate -1 con comment → row en `rag_feedback` con `rating=-1` y `reason` matchea el comment.
   - `test_rate_other_streams_does_not_touch_rag_feedback`: rate brief / draft_wa / anticipate → row en `rag_ft_panel_ratings` pero **ningún row nuevo** en `rag_feedback`.

4. Correr `.venv/bin/python -m pytest tests/test_fine_tunning_panel.py tests/test_fine_tunning_brief_queue.py -q` — todos pasan.

5. Commit con subject `feat(fine_tunning): wire retrieval_answer thumbs → rag_feedback (cierra loop al ranker-vivo)` + body explicativo (por qué / qué cambia / cómo lo medí / cómo revertir) + trailer Devin co-author.

6. Push a origin de la branch (`git push -u origin feat/fine-tunning-wire-retrieval-answer`). **NO mergear a master**. El user audita el diff y mergea a mano.

**Restricciones**:
- NO tocar otros endpoints / streams. Solo `retrieval_answer` en este pase.
- NO modificar `rag tune --online`.
- NO mergear a master.
- Si encontrás un bug pre-existente que bloquea el wiring, reportalo en el output y NO arreglés nada extra.

## Output final

En el message final al user, incluí:

1. Path absoluto del audit markdown que escribiste (debe ser `04-Archive/99-obsidian-system/99-AI/system/fine-tunning-audit-2026-05-04.md` — si lo escribiste ahí, OK).
2. URL del PR creado en `https://github.com/jagoff/rag-obsidian/pull/...` (cuando hagas `git push -u`, GitHub responde con la URL para crear el PR — incluila).
3. Summary del wiring: líneas tocadas, tests añadidos, output del pytest.

Todo en español rioplatense voseo.
