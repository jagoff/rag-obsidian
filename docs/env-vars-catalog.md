# Env vars — catálogo completo

**Audit 2026-04-24**: `rag/__init__.py` consulta **85+ env vars únicas** via `os.environ.get()`. El CLAUDE.md documenta ~38 de ellas (las "críticas", con rollback paths + historia). Este doc complementa con las **47+ restantes** que el CLAUDE.md no cubre.

Para cambiar cualquier default: setear el env var en el shell antes de invocar el CLI, o agregarlo al `<key>EnvironmentVariables</key>` del plist correspondiente en `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist`. Tras editar un plist, aplicar con `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/...plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/...plist` (kickstart no relee env).

## Config / Defaults ajustables

| Variable | Default | Ubicación | Qué hace |
|---|---|---|---|
| `RAG_ADAPTIVE_K_MIN` | `2` | `rag/__init__.py:22849` | Floor del adaptive-k (mínimo de results devuelto por `retrieve()` cuando la ventana de scoring es estrecha). |
| `RAG_ADAPTIVE_K_GAP` | `0.35` | `rag/__init__.py:22850` | Gap ratio entre el top-score y la cutoff — define cuándo cortar el adaptive-k. |
| `RAG_AGENT_UNPRODUCTIVE_CAP` | `3` | `rag/__init__.py:38075` | Cuántas iteraciones sin tool-call útil tolera el `rag do` agent loop antes de abortar. |
| `RAG_AUTO_HARVEST_MIN_CONF` | `0.8` | `rag/__init__.py:61663` | Confidence mínimo para que el auto-harvest LLM-as-judge (nightly 03:30) acepte una query low-conf como label para el gate GC#2.C. |
| `RAG_AUTO_HARVEST_SNIPPET_CHARS` | `400` | `rag/__init__.py:61666` | Budget de context text que el judge ve por candidate path en el auto-harvest. |
| `RAG_CACHE_MAX_ROWS` | `2000` | `rag/__init__.py:5814` | Cap total de filas en `rag_response_cache` (semantic cache). LRU cuando supera. |
| `RAG_CACHE_TTL_DEFAULT` | `86400` (24h) | `rag/__init__.py:5812` | TTL default para entries del semantic cache. |
| `RAG_CACHE_TTL_RECENT` | `600` (10min) | `rag/__init__.py:5813` | TTL corto para queries con intent `recent`/`agenda` (data time-sensitive). |
| `RAG_COMPARISON_POOL` | `30` | `rag/__init__.py:281` | Rerank pool size cuando intent es `comparison` (más candidates para cubrir ambos polos). |
| `RAG_SYNTHESIS_POOL` | `30` | `rag/__init__.py:282` | Idem para `synthesis` (cross-source). |
| `RAG_CONTEXT_BUDGET_WARN` | `0.80` | `rag/__init__.py:59237` | % del `num_ctx` que dispara un warning `_silent_log` cuando el prompt real se acerca al límite. Instrumentación. |
| `RAG_CONTEXT_CACHE_MAX` | `5000` | `rag/__init__.py:3728` | Cap del context-summary cache (LRU). |
| `RAG_SYNTHETIC_Q_CACHE_MAX` | `5000` | `rag/__init__.py:4038` | Cap del synthetic-questions cache. |
| `RAG_DEDUP_PRE_RERANK` | `1` (on) | `rag/__init__.py:22249` | Dedup por file antes de truncar al rerank pool: cuando una nota tiene múltiples chunks, el RRF puede llenar los 25 slots con chunks de una sola nota, expulsando notas con 1-2 chunks de match perfecto. Conserva solo el primer chunk por file para dar más diversidad al reranker. `=0` vuelve al pool sin dedup (comportamiento previo a 2026-05-08). |
| `RAG_ENTITY_CONFIDENCE_MIN` | `0.70` | `rag/__init__.py:63272` | Umbral para el score de GLiNER al extraer entidades (persons/orgs). <0.7 se descarta. |
| `RAG_EXPAND_TIMEOUT_S` | `3.0` | `rag/__init__.py:15762` | Timeout del `expand_queries()` (qwen2.5:3b paraphrase call). Post-timeout → devolver [query] sin paraphrases. |
| `RAG_LEARNED_PARA_MIN_HITS` | `2` | `rag/__init__.py:15766` | Hits mínimos para que una paraphrase aprendida entre al reranker. Evita noise de one-offs. |
| `RAG_LLM_INTENT_TIMEOUT` | `2.0` | `rag/__init__.py:16381` | Timeout del classifier LLM-based intent (opt-in via `RAG_LLM_INTENT=1`). |
| `RAG_LOOKUP_THRESHOLD` | `0.6` | `rag/__init__.py:263` | Top-1 rerank score mínimo para activar el fast-path dispatch (qwen2.5:3b + num_ctx=4096, skip citation-repair). |
| `RAG_LOOKUP_MODEL` | `qwen2.5:3b` | `rag/__init__.py:264` | Modelo LLM del fast-path. Swap por otro model si querés probar. |
| `RAG_MMR_LAMBDA` | `0.7` | `rag/__init__.py:22697` | Lambda del MMR (Maximal Marginal Relevance) — 0 = max diversity, 1 = max relevance. |
| `RAG_MMR_POOL_MULTIPLIER` | `3.0` | `rag/__init__.py:22700` | Pool de candidatos que el MMR recibe (k * multiplier). |
| `RAG_NLI_CONTRADICTS_THRESHOLD` | `0.7` | `rag/__init__.py:238` | Score de la clase "contradiction" de mDeBERTa-NLI por encima del cual el claim se flagea. |
| `RAG_NLI_MAX_CLAIMS` | `20` | `rag/__init__.py:250` | Cap de claims extraídos del LLM response para chequear NLI. |
| `RAG_NLI_NEVER_UNLOAD` | `""` (off) | `rag/__init__.py:63171` | Setear `1` para pinear el NLI model en memoria (evita idle-unload). Análogo a `RAG_RERANKER_NEVER_UNLOAD`. |
| `RAG_MEMORY_PRESSURE_SWAP_GB` | `1.5` (CLI) / `8.0` (web plist) | `rag/_memory_pressure_watchdog.py:363` | Swap (GB) usado por el kernel que dispara `_handle_memory_pressure` (unload chat + reranker). Default bajado de `4.0`→`1.5` el 2026-05-12: Mac 16GB ya thrashea NAND con 2GB swap, esperar 4GB era muy tarde. El daemon web preserva 8.0 (long-running, más tolerante). |
| `RAG_MEMORY_PRESSURE_COOLDOWN_S` | `300` (default) / `60` (override en `rag index`) | `rag/_memory_pressure_watchdog.py:367` | Segundos entre acciones del watchdog (evita unload→reload→unload thrash). `rag index` baja el override a 60s en `_run_index` porque el indexer no re-carga modelos inmediato — preferible 10 acciones en 30min que 3. |
| `RAG_CONTRADICTION_MAX_WORKERS` | `2` | `rag/__init__.py` (`_CONTRA_MAX_WORKERS`) | Cap de daemon threads que corren contradiction check concurrentemente. Pre-fix (2026-05-12): unbounded — 681 archivos = potencial 681 threads cada uno cargando reranker bge MPS (~600MB). Semaphore adquirido DENTRO del wrapper, threads pendientes esperan (~1MB stack c/u sin presión GPU). |
| `RAG_PPR_SEED_K` | `5` | `rag/__init__.py:22432` | Cantidad de top-k candidates usados como seeds del Personalized PageRank (opt-in via `RAG_PPR_TOPIC=1`). |
| `RAG_SILENT_LOG_ALERT_THRESHOLD` | `20` | `rag/__init__.py:1048` | N silent-errors / ventana que dispara el alert a stderr (evita que los logs SQL fail queden invisible). |
| `RAG_TEMPORAL_LOOKUP_BOOST` | `3.0` | `rag/__init__.py:22399` | Multiplier de recency para queries con anchor temporal detectado (`hoy`, `esta semana`). |
| `RAG_TITLE_EXACT_BONUS` | `0.5` | `rag/__init__.py:22554` | Bonus sumado al score post-rerank cuando la query coincide exactamente con el título de una nota (title_match ≥0.999, 1-4 tokens). Cubre el gap típico (0.1-0.3) que el weight `title_match` solo no puede cerrar. `=0` desactiva. Override: `RAG_TITLE_EXACT_BONUS=0.3` para afinar. |
| `RAG_TOKENS_PER_CHAR` | `0.25` | `rag/__init__.py:59236` | Factor de conversión char→token para estimar el budget del prompt sin llamar al tokenizer real. 0.25 asume ~4 chars/token (promedio qwen2.5). |
| `RAG_VLM_CAPTION_MAX_PER_RUN` | `500` | `rag/__init__.py` | Cap de imágenes captioned por corrida del VLM. Bound para evitar runs multi-hora en vaults grandes. |
| `RAG_WEB_RERANK_POOL` | `3` | `web/server.py:12763` | Override del rerank pool en el web `/api/chat` para queries no-list-intent. Default bajado a 3 (de 5 legacy) para ahorrar ~40% del rerank time bajo RAM pressure (35/36 GB usados). List-intent queries ignoran este valor y usan pool=15. Override: `RAG_WEB_RERANK_POOL=5` vuelve al comportamiento legacy; `=2` para A/B más agresivo. |

## Feature flags (opt-in, default OFF)

| Variable | Activar con | Ubicación | Qué hace |
|---|---|---|---|
| `RAG_ADAPTIVE_K` | `1`/`true`/`yes` | `rag/__init__.py` | Activa adaptive-k (dinámicamente reduce `k` cuando el score gap cae). Default off, ver `RAG_ADAPTIVE_K_MIN`/`GAP`. |
| `RAG_CITA_DETECT` | `1` | `rag/__init__.py` | Activa detector de citas médicas/judiciales en chunks para flaggear fechas de vencimiento. |
| `RAG_CONTEXT_SUMMARY` | `0` | `rag/__init__.py:3944` | Opt-OUT del context summary prefix en chunks (default ON — ver `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY`). |
| `RAG_LLM_INTENT` | `1` | `rag/__init__.py` | Reemplaza el regex-based intent classifier por un LLM call (qwen2.5:3b). Experimental — eval drift +CI. |
| `RAG_MMR_DIVERSITY` | `1` | `rag/__init__.py:3077` | Activa el MMR diversity re-order post-rerank. Default OFF — afecta scoring. |
| `RAG_PARALLEL_POSTPROCESS` | `0` | `rag/__init__.py` | Default ON — setear a `0` desactiva paralelización de stages de postprocess (citation repair + NLI + diversity en threads). Debug. |
| `RAG_PEEKABOO_ENABLE` | `1`/`true`/`yes`/`on` (case-insensitive) | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py) | Activa la captura on-demand de pantalla via [Peekaboo CLI](https://github.com/openclaw/Peekaboo) + caption con granite MLX-VLM. Default OFF — la tool MCP `rag_screen_capture` y el CLI `rag screen` retornan `peekaboo_disabled` sin tocar el subprocess. Pull only en Fase 1 (sin daemon). Requiere `brew install steipete/tap/peekaboo` + Screen Recording TCC en la app terminal. |
| `RAG_PEEKABOO_BIN` | `""` | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py) | Override del path al binario `peekaboo`. Si no seteado, usa `shutil.which("peekaboo")`. Útil para tests con binario fake. |
| `RAG_PEEKABOO_TIMEOUT_SECS` | `10` | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py) | Timeout (segundos) del subprocess Peekaboo. Beyond → error `peekaboo_timeout`. Floor 1.0s. |
| `RAG_SCREEN_OBSERVE` | `1`/`true`/`yes`/`on` (case-insensitive) | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py), [`rag/runtime/jobs/frequent.py`](../rag/runtime/jobs/frequent.py) | Activa el observer pasivo Peekaboo (Fase 2). Cuando ON + `RAG_PEEKABOO_ENABLE=1` + TCC concedido: el supervisor job `screen_observer` corre `rag screen observe-once` cada 15min, captura el frontmost, captiona con granite, dedupea por (app,title) en últimos 60s, e inserta en `rag_screen_observations`. Default OFF — sin esto, el daemon corre pero observa_once retorna `skipped_reason: observe_disabled` en <100ms. Doble opt-in con `RAG_PEEKABOO_ENABLE`. |
| `RAG_SCREEN_QUIET_HOURS` | `""` (sin quiet) | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py) | Ventana horaria donde el observer NO captura. Formato `"HH:MM-HH:MM"` (24h local, soporta wrap medianoche). Ej. `"22:00-07:00"`. `observe_once` retorna `skipped_reason: quiet_hours` dentro de la ventana. Útil para no capturar cuando el user duerme o cuando hay riesgo de leak (pantalla bloqueada, contenido privado abierto). |
| `RAG_SCREEN_APP_DENY` | `""` | [`rag/integrations/peekaboo.py`](../rag/integrations/peekaboo.py) | CSV case-insensitive de nombres de apps que el observer NUNCA observa. Ej. `"1Password,Banking,Messages"`. Cuando la captura post-subprocess identifica el `app_name` en la denylist → borra la PNG, skip VLM, retorna `skipped_reason: app_denied`. Defensa contra leaks de pantalla sensibles. |
| `RAG_PPR_TOPIC` | `1` | `rag/__init__.py:22479` | Activa Personalized PageRank sobre el grafo de wikilinks usando los top-k results como seeds. Experimental. |
| `RAG_SCORE_CALIBRATION` | `1` | `rag/__init__.py:58608` | Activa el isotonic regression per-source (usa `rag_score_calibration`). Mejora rerank bajo mixed-source retrieval. |
| `RAG_UNIFIED_CHAT` | `1` | `rag/__init__.py:21121` | Activa el unified chat pipeline (`run_chat_turn`) para CLI `rag chat` — alineación con web/serve. |
| `RAG_VLM_CAPTION` | `1` (queries) / **`0` durante `rag index`** | `rag/__init__.py`, `rag/ocr.py:413` | Activa captioning de imágenes embebidas via VLM granite-vision. Apple Vision OCR sigue siendo el path default — VLM es fallback. **Default OFF durante `rag index`** (override en `_run_index` 2026-05-12 audit): cargar granite (~3 GB Metal) en simultáneo con embedder MLX + reranker + web daemon dispara swap NAND y reboot. Para captionar vault completo correr [`rag vlm-backfill`](../rag/__init__.py) en pase aislado (web daemon bajado recomendado). |
| `RAG_VLM_MODEL` | `mlx-community/granite-vision-3.2-2b-4bit` | [`rag/ocr.py:386`](../rag/ocr.py) | Override del modelo VLM cargado por [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm). Default granite-vision (~3 GB MPS). Alternativas testeables: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (~2GB, train data UI-aware más reciente) — usar si el caption quality del screen observer (Fase 2 Peekaboo) no mejora con prompt tuning. Eval real es user-initiated: cambiar var, correr `rag screen capture --keep --mode screen`, comparar caption A/B. Swap es ZERO-COST si NO se hace `rag index` con el nuevo modelo — solo afecta captioning en runtime (no embed dims). |
| `RAG_WIKILINK_EXPANSION` | `1` | `rag/__init__.py:11397` | Expande wikilinks en el CONTEXT del retrieve (resolver link targets a body). Experimental. |

## MLX backend

| Variable | Default | Ubicación | Qué hace |
|---|---|---|---|
| `RAG_MLX_IDLE_TTL` | `1800` (30min) | `rag/llm_backend.py` | Segundos de inactividad antes de que el watchdog evicte un modelo de `_loaded` + libere Metal cache. Setear `0` para deshabilitar. Análogo a `RAG_RERANKER_IDLE_TTL`. |
| `RAG_MLX_IDLE_DISABLE` | `""` (off) | `rag/llm_backend.py` | Setear `1` para deshabilitar el watchdog thread sin cambiar el TTL (útil para tests / debug). |
| `RAG_EMBED_VIA_SHIM` | `""` (off) | `rag/mlx_embed.py:253` | Setear `1` para routear el embedder MLX a un servidor dedicado (`mlx_embed_server`) en vez del path in-process. Razón: el embedder in-process puede trippear el watchdog Metal bajo bursts (chat + embed + rerank + gen en serie), causando crashes del rag-web. El server dedicado tiene su propio Metal context, sin contención. Si el shim no responde, cae automáticamente a in-process. Ver `RAG_EMBED_SHIM_URL`. |
| `RAG_EMBED_SHIM_URL` | `http://127.0.0.1:8085` | `rag/mlx_embed.py:202` | URL base del servidor embedder dedicado (OpenAI-compat `/v1/embeddings`). Solo relevante cuando `RAG_EMBED_VIA_SHIM=1`. |

## Infraestructura / internal

| Variable | Default | Ubicación | Qué hace |
|---|---|---|---|
| `_CONTRADICTION_ASYNC` | `1` (on) | `rag/__init__.py:24758` | Async default para el contradiction detector. Setear `0` para sync (útil si debugueás un falso positivo). |
| `OBSIDIAN_RAG_STATE_DIR` | `~/.local/share/obsidian-rag` | `rag/__init__.py` | Override del directorio de estado operativo (logs JSONL, caches, ranker, silent errors). Usado por tests/sandboxes para no escribir en el home real durante import. |
| `OBSIDIAN_RAG_CONFIG_DIR` | `~/.config/obsidian-rag` | `web/_admin.py` | Override del directorio de config para el admin token web. Si no está seteado pero `OBSIDIAN_RAG_STATE_DIR` sí, los tests guardan el token bajo ese state dir. |
| `OBSIDIAN_RAG_DB_PATH` | `$OBSIDIAN_RAG_STATE_DIR/ragvec` | `rag/__init__.py` | Override del directorio sqlite (`ragvec.db`, `telemetry.db`, locks). Debe apuntar a un directorio writable. |
| `OBSIDIAN_RAG_TEST_ADMIN_BYPASS` | `""` | `web/_admin.py`, `tests/conftest.py` | Bypass de auth solo para tests: requiere además `PYTEST_CURRENT_TEST`, por lo que no se activa en runtime normal. Los tests de auth lo desactivan explícitamente. |
| `OBSIDIAN_RAG_WEB_CHAT_MODEL` | `qwen2.5:7b` | `web/server.py:2148` | Override del chat model del web server. Generado al plist en `rag setup`. |
| `RAG_CACHE_ENABLED` | `1` (on) | `rag/__init__.py:5867` | Master switch del semantic cache. Tests autouse lo fuerzan a `0`. |
| `RAG_NO_WARMUP` | `""` (off) | `rag/__init__.py` | Debug — skippea el warmup del reranker + bge-m3 + corpus cache en startup. |
| `RAG_ANTICIPATE_LOCK_PATH` | `~/.local/share/obsidian-rag/anticipate.lock` | `rag_anticipate/lockfile.py` | Override del lockfile cooperativo del anticipatory agent. Tests/sandboxes lo usan para no chocar con el daemon launchd real. |
| `RAG_INDEX_NICE` | `10` | `rag/__init__.py` | Prioridad CPU del proceso durante `rag index`. Valor nice absoluto (0–20; mayor = menos prioridad). `0` desactiva el cambio de prioridad. Permite que el web server y el WA listener sigan respondiendo durante reindex largo. |
| `RAG_INDEX_SAFE` | `1` | `rag/__init__.py` | Perfil conservador automático para todo `rag index`: embed slices, batching, shard de notas grandes, entities off, preflight/memory guard y abort limpio por RSS del proceso. `0` desactiva las guardas. |
| `RAG_INDEX_LLM_ENRICHMENTS` | `0` | `rag/__init__.py` | Opt-in para enrichment LLM en indexing (synthetic questions/context summary/contextual retrieval/contradictions). Default off en safe mode para que `rag index` sea parse + embed. Context summary requiere además `RAG_CONTEXT_SUMMARY=1`; contextual retrieval requiere `--contextual` o `RAG_CONTEXTUAL_RETRIEVAL=1`. |
| `RAG_INDEX_SKIP_CONTRADICTIONS` | `1` en safe mode | `rag/__init__.py` | Saltea los checks de contradicción index-time en `rag index` normal. Seteá `0` o `RAG_INDEX_LLM_ENRICHMENTS=1` para reactivarlos. |
| `RAG_INDEX_FULL_SAFE` | `1` para `--full` | `rag/__init__.py` | Perfil adicional para full rebuild: salta GLiNER entities salvo override explícito y conserva el batching seguro rápido de `RAG_INDEX_SAFE`. Los enriquecimientos LLM se controlan con `RAG_INDEX_LLM_ENRICHMENTS`. |
| `RAG_INDEX_EMBED_CACHE` | `1` | `rag/embedding_cache.py` | Cache persistente exact-match de embeddings de index por hash de texto + modelo/backend. Permite que un rebuild lógico `--full` re-use embeddings idénticos sin perder chunks ni metadatos. `0` desactiva. |
| `RAG_INDEX_EMBED_CACHE_MAX_ROWS` | `200000` | `rag/embedding_cache.py` | Límite LRU aproximado de filas en `embedding_cache.db`; al pasar el límite borra las menos usadas recientemente. |
| `RAG_INDEX_ETL_FRESHNESS` | `1` | `rag/__init__.py`, `rag/index_etl_state.py` | Saltea ETLs cross-source recientemente exitosos cuando el snapshot `.md` todavía existe en el vault. El estado vive en `index_etl_state.json`; perderlo solo fuerza un pre-sync fresco. `0` fuerza todos los ETLs. |
| `RAG_INDEX_ETL_FRESH_TTL_S` | `600` | `rag/__init__.py`, `rag/index_etl_state.py` | Ventana de frescura para ETLs de index, en segundos. `0` desactiva skips por TTL y sirve para medir una corrida fresca completa. |
| `RAG_INDEX_ETL_PARALLEL` | `1` | `rag/__init__.py` | Corre los pre-syncs externos independientes (Reminders, Calendar, Chrome history, Gmail, GitHub, Claude, YouTube, Spotify, Screentime) en paralelo. `0` vuelve al orden serial para debug. |
| `RAG_INDEX_ETL_MAX_WORKERS` | `6` | `rag/__init__.py` | Límite de workers para `RAG_INDEX_ETL_PARALLEL`. Subirlo solo ayuda si los ETLs están esperando I/O externo; bajarlo reduce presión concurrente. |
| `RAG_INDEX_EMBED_SLICE_SIZE` | `auto` | `rag/__init__.py` | Cap externo de textos por llamada a `embed()` dentro del indexer. En MLX se ajusta por RAM (`48` en Macs de ~36 GB). `0` desactiva el slicing. |
| `RAG_INDEX_LOCAL_EMBED_BATCH` | `auto` | `rag/__init__.py` | Batch interno del forward local de embeddings. En MLX se ajusta por RAM (`48` en Macs de ~36 GB), alineado con `RAG_INDEX_EMBED_SLICE_SIZE`. |
| `RAG_INDEX_FILE_CHUNK_SLICE_SIZE` | `128` | `rag/__init__.py` | Sharding de una nota grande antes del flush. Evita que un único archivo genere miles de chunks/embeddings en una sola unidad de trabajo. `0` desactiva el sharding. |
| `RAG_INDEX_BATCH_EMBEDS` | `1` | `rag/__init__.py` | Agrupa chunks de varias notas antes de llamar a `embed()`. Reduce overhead por archivo; safe mode lo combina con slices chicos para acotar memoria. |
| `RAG_INDEX_BATCH_SIZE` | `auto` | `rag/__init__.py` | Objetivo de chunks acumulados antes de `_flush_batch()`; en MLX se ajusta por RAM (`96` en Macs de ~36 GB). |
| `RAG_INDEX_BATCH_SLEEP_MS` | `50` | `rag/__init__.py` | Pausa en ms después de cada `_flush_batch()` durante `rag index`. Ejemplo: `150` → ~5 batches/s máximo, GPU respira entre embeddings. No afecta archivos skippeados por hash. |
| `RAG_INDEX_PREFLIGHT_MEMORY_GUARD` | `1` | `rag/__init__.py` | Check de memory pressure antes del index y después de ETLs cross-source. Puede descargar modelos y abortar limpio si la presión sigue en zona peligrosa. |
| `RAG_INDEX_MEMORY_GUARD_INTERVAL_S` | `5` | `rag/__init__.py` | Chequeo síncrono de memory pressure entre slices/flushes. Si cruza threshold, unload in-process + clear cache + pausa corta. Complementa al watchdog thread. |
| `RAG_INDEX_MEMORY_PRESSURE_SLEEP_S` | `1` | `rag/__init__.py` | Pausa aplicada por el memory guard cuando detecta presión durante indexado. |
| `RAG_INDEX_ABORT_ON_MEMORY_PRESSURE` | `1` | `rag/__init__.py` | Si el propio proceso pasa `RAG_INDEX_ABORT_SELF_RSS_GB`, aborta el index antes de colgar la máquina. También respeta aborts de used%/swap si los seteás explícitamente. |
| `RAG_INDEX_ABORT_SELF_RSS_GB` | `18.0` | `rag/__init__.py` | Límite duro de RSS del propio proceso `rag index`. En una Mac de 36 GB, que el indexer pase ~18 GB indica fuga/pico patológico y se aborta limpio. |
| `RAG_INDEX_ALLOW_EMPTY_ORPHAN_CLEANUP` | `""` (off) | `rag/__init__.py` | Opt-in para limpiar huérfanos cuando el scan del vault encontró 0 notas. Default off para evitar que un vault mal resuelto o vacío borre una colección completa. |
| `RAG_REMINDERS_OSASCRIPT_TIMEOUT_S` | `45` (`5` durante safe index) | `rag/integrations/reminders.py`, `rag/__init__.py` | Timeout del AppleScript de Reminders. Safe index lo baja a 5s para que Reminders sin respuesta no bloquee todo el pre-sync durante 45s. |
| `RAG_START_SAFE` | `1` | `rag/cli/setup.py` | Perfil conservador automático para `rag start` / `rag start --full`: fuerza safe env en el catch-up index, lo corre sin checks de contradicción y sin context/synthetic/entities durante bootstrap, chequea memory pressure antes de levantar servicios y aplica stagger entre bootstraps. `0` desactiva. |
| `RAG_START_MEMORY_GUARD` | `1` | `rag/cli/setup.py` | Gate de memoria para `rag start`: si `memory_pressure`/swap están en zona de aborto, pausa y luego saltea el tramo en vez de cargar más procesos. |
| `RAG_START_ABORT_USED_PCT` | `92` | `rag/cli/setup.py` | Umbral de memoria usada del sistema para saltear catch-up/daemons durante `rag start`. |
| `RAG_START_ABORT_SWAP_GB` | `0` | `rag/cli/setup.py` | Opt-in para saltear catch-up/daemons durante `rag start` por swap usado. Default off porque macOS puede dejar swap viejo asignado aunque `memory_pressure` ya muestre mucha memoria libre. |
| `RAG_START_BOOTSTRAP_STAGGER_S` | `2` | `rag/cli/setup.py` | Pausa entre bootstraps launchd en `rag start`, para evitar cargar supervisor/watch/web/RagNet en una ráfaga. |
| `RAG_OCR` | `""` (on) | `rag/__init__.py` | Opt-out del OCR de imágenes embebidas (Apple Vision). `0` desactiva. |
| `RAG_POSTPROCESS_MODEL` | `""` | `rag/__init__.py:204` | Override del model LLM usado en stages de postprocess (critique, NLI repair). Default = chat model. |
| `RAG_RERANKER_FT_PATH` | `""` (off) | `rag/__init__.py:14325` | Path al cross-encoder fine-tuned (gate GC#2.C). Cuando hay model promovido, apunta al symlink `~/.cache/obsidian-rag/reranker-ft-current`. |
| `RAG_RETRIEVE_TIMING` | `""` (off) | `rag/__init__.py` | Debug — dump per-stage timing breakdown de `retrieve()` a stderr. |
| `RAG_LLM_BACKEND` | `mlx` | `rag/llm_backend.py` | Dispatch backend para chat/generation. Valor soportado en el runtime actual: `mlx` (Apple Silicon MLX native). El path Ollama quedó legacy y ya no es rollback soportado por `get_backend()`. Cada invocación de dispatch setea el ContextVar `_ACTIVE_BACKEND_CTX` (`rag/__init__.py`); `log_query_event` lo lee y lo persiste en `rag_queries.extra_json` como campos `backend`, `fallback_reason`, `backend_active`. Query de verificación: `SELECT json_extract(extra_json,'$.backend'), COUNT(*) FROM rag_queries WHERE ts > datetime('now','-1 day') GROUP BY 1`. |
| `RAG_STATE_SQL` | `1` | `rag/__init__.py` | Historicamente activaba el SQL path. Post-T10 es no-op, setear en plists para trail. |
| `RAG_TRACK_OPENS` | `""` (off) | `rag/__init__.py` | Switch del OSC-8 terminal link scheme de `file://` a `x-rag-open://`. Opt-in para trackeo de clicks. |
| `RAG_BEHAVIOR_IMPLICIT` | `1` (on) | `rag/__init__.py:_behavior_augmented_cases` | Master switch del loop implícito que feedea `rag tune --online` con eventos de `rag_behavior` (open/save/kept/copy/positive_implicit/negative_implicit/deleted). `0` → la función devuelve `[]` y `rag tune --online` se queda solo con feedback explícito + sintético + implícito (rag_feedback sin corrective). Dedup automático contra explicit feedback: si `(norm_q, path)` ya tiene rating ±1 o `corrective_path` en `rag_feedback`, el evento de behavior se descarta (explicit > implícito). |
| `YOUTUBE_API_KEY` | `""` | `web/server.py:5654` | API key opcional para YouTube transcript fetching. Sin esto, el fallback es `yt-dlp --write-auto-sub`. |

---

## Cómo se generó este catálogo

```bash
grep -oh 'os.environ.get("[A-Z_]\+' rag/__init__.py | sed 's/.*"//' | sort -u
# → 85+ env vars
grep -oE '`RAG_[A-Z_]+`|`OBSIDIAN_RAG_[A-Z_]+`|...' CLAUDE.md | tr -d '`' | sort -u
# → 38 documented
# → diff = 47+ missing (este doc)
```

Script exacto del audit en `scripts/audit_env_vars.py`.

---

## Importantes (ya documentadas en CLAUDE.md)

Las siguientes env vars tienen explicación extensa + historia + rollback paths en [`CLAUDE.md § Env vars`](../CLAUDE.md). Mencionadas aquí solo para cross-reference:

`OBSIDIAN_RAG_VAULT`, `OBSIDIAN_RAG_NO_APPLE`, `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY`, `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q`, `OBSIDIAN_RAG_MOZE_FOLDER`, `OBSIDIAN_RAG_INDEX_WA_MONTHLY`, `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS`, `OBSIDIAN_RAG_BIND_HOST`, `OBSIDIAN_RAG_ALLOW_LAN`, `OLLAMA_KEEP_ALIVE`, `RAG_TIMEZONE`, `RAG_KEEP_ALIVE_LARGE_MODEL`, `RAG_MEMORY_PRESSURE_DISABLE`/`THRESHOLD`/`INTERVAL`, `RAG_EXPLORE`, `RAG_RERANKER_IDLE_TTL`/`NEVER_UNLOAD`, `RAG_LOCAL_EMBED`/`WAIT_MS`, `RAG_FAST_PATH_KEEP_WITH_TOOLS`, `RAG_ENTITY_LOOKUP`/`EXTRACT_ENTITIES`, `RAG_NLI_GROUNDING`/`IDLE_TTL`/`SKIP_INTENTS`, `RAG_ADAPTIVE_ROUTING`, `RAG_LOOKUP_NUM_CTX`, `RAG_EXPAND_MIN_TOKENS`, `RAG_CITATION_REPAIR_MAX_BAD`, `RAG_WA_FAST_PATH`/`THRESHOLD`, `RAG_WA_SKIP_PARAPHRASE`, `RAG_DEEP_MAX_SECONDS`, `RAG_CACHE_COSINE`, `RAG_EVAL_GATE_SINGLES_MIN`/`CHAINS_MIN`, `RAG_DEBUG`, `RAG_LOG_QUERY_ASYNC`, `RAG_LOG_BEHAVIOR_ASYNC`, `RAG_METRICS_ASYNC`.
