# Variables de entorno

Lista completa de env vars que el RAG respeta. Última actualización **2026-04-25** (audit del catálogo + nuevas vars del audit R2-Security).

> Este doc es la **referencia exhaustiva** generada cruzando `os.environ.get()` en `rag/__init__.py`, `rag/ocr.py`, `rag/anticipatory.py`, `rag_anticipate/`, `web/server.py`, `scripts/finetune_reranker.py`. Los path:line apuntan al sitio de **lectura primaria** (cuando una var se lee en >1 lugar, el path:line del default canónico).
>
> Para discusión más narrativa de las "críticas" (con rollback paths e historia), ver [`CLAUDE.md`](../CLAUDE.md). Para guía orientada a usuarios (cuándo tocarlas), ver [`docs/variables-entorno.md`](./variables-entorno.md). Para el catálogo curado del audit del 2026-04-24, ver [`docs/env-vars-catalog.md`](./env-vars-catalog.md).
>
> Para aplicar un override persistente: setear en el shell antes de invocar `rag`, o agregarlo al `<key>EnvironmentVariables</key>` del plist en `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist` y aplicar con `launchctl bootout gui/$(id -u) ~/...plist && launchctl bootstrap gui/$(id -u) ~/...plist` (`kickstart -k` re-lanza pero NO relee env).

---

## Recientes (audit 2026-04-25)

Las del último audit de seguridad/perf. Citadas explícitamente porque son las más probables de aparecer como "qué es esto" en el log.

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_RERANK_FASTPATH_DIST` | `0.10` | `rag/__init__.py:12517` | Cosine-distance threshold del fast-path del reranker. Si el top-1 candidato del retrieval semántico está a `<= 0.10` cosine distance del query embedding, se skipea la pasada del cross-encoder MPS y se devuelve el orden semántico tal cual. Win: ~80-200ms en queries fáciles. Subir a `0.05` para más rerank (más preciso, más lento), bajar a `0.20` para más fast-path (más rápido, ruido en borderlines). |
| `RAG_AMBIENT_DISABLED` | unset | `rag/__init__.py:20903` | Kill switch global del ambient hook. Si `1`/`true`/`yes`, el hook que dispara sobre saves del Inbox NO corre — útil para desactivar el agent desde un launchd plist sin tocar `ambient.json`. Complementa `rag ambient disable` (que escribe el flag persistente). |
| `RAG_CHAT_UPLOADS_TTL_DAYS` | `30` | `rag/__init__.py:43415`, `rag/__init__.py:44586` | TTL de las imágenes subidas por `/api/chat/upload-image` que viven en `~/.local/share/obsidian-rag/chat-uploads/`. `rag maintenance` borra archivos con `mtime < now - TTL_DAYS * 86400`. Subir a `90` si querés conservar más historia, bajar a `7` si es un Mac con disco chico. |
| `RAG_OCR_HISTORICAL_MAX_DAYS` | `_CHAT_UPLOAD_HISTORICAL_DAYS_DEFAULT` (90) | `web/server.py:2528` | Tope de cuántos días hacia atrás el endpoint `/api/chat/upload-image` reprocesa OCR sobre uploads históricos al backfillear el índice. Anti-DoS: evita que un user pida re-OCR de uploads antiguos y sature la cola de Apple Vision. |
| `RAG_SSE_MAX_PER_IP` | `_SSE_MAX_PER_IP_DEFAULT` (4) | `web/server.py:12446` | Cap de conexiones SSE simultáneas por IP. El web server mantiene un counter por client IP; cuando un nuevo SSE supera el cap, se rechaza con HTTP 429. Defensive contra fork-bombs de tabs abiertos en el mismo host. |
| `RAG_CROSS_SOURCE_DEDUP_THRESHOLD` | `0.7` (jaccard) | `rag/__init__.py:2123` | Threshold del Jaccard sobre los primeros 600 chars de chunks pareados en el dedup cross-source. `>= 0.7` → considerar duplicates y mantener solo el de mayor score. Setear a `1.0` para deshabilitar (escape hatch); bajo `0.5` empieza a colapsar temas relacionados pero distintos. |

---

## Core (vault, timezone, Apple integrations)

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `OBSIDIAN_RAG_VAULT` | iCloud Notes path legacy | `rag/__init__.py:608` (también 19757, 25504, 42335, 42716, 42743, 42753) | Path absoluto del vault. **Gana sobre `rag vault use <name>`**. Las collections sqlite-vec se namespace por sha8 del path → switch no contamina datos. |
| `RAG_TIMEZONE` | `America/Argentina/Buenos_Aires` | `rag/__init__.py:33421` | Zona horaria IANA para parsear fechas ISO con tz aware. Único punto donde se asume Argentina; cambiar acá afecta digest/morning/today timestamps. |
| `OBSIDIAN_RAG_NO_APPLE` | unset (off) | `rag/__init__.py:32881` | Si `1`/`true`/`yes`, desactiva integraciones con Apple (Calendar, Reminders, Mail, Screen Time). Imprescindible en Linux o Macs sin Full Disk Access. |
| `OBSIDIAN_RAG_MOZE_FOLDER` | `02-Areas/Personal/Finanzas/MOZE` | (ver `rag/integrations`) | Path del vault donde cae la ingesta MOZE. Override si reorganizaste PARA. |
| `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` | unset | (ver `rag.watch`) | Lista coma-separada de carpetas que `rag watch` ignora. Útil para mantener `03-Resources/WhatsApp` o `04-Archive` fuera del re-index continuo. |
| `OBSIDIAN_RAG_WIKI_ENABLED` | `1` | `rag/__init__.py:8405` | Master switch del wikilink resolver/expansion. `0` desactiva todo el camino que materializa `[[link]]` → body. |
| `OBSIDIAN_RAG_BIND_HOST` | `127.0.0.1` | `web/server.py:13380` | Host de bind del FastAPI. Solo loopback por default; cambiar requiere también setear `OBSIDIAN_RAG_ALLOW_LAN`. |
| `OBSIDIAN_RAG_ALLOW_LAN` | unset (off) | `web/server.py:1246` | Si `1`/`true`/`yes`, permite que el web server acepte conexiones de la LAN (CORS + bind). Necesario para abrir desde el iPhone en la misma red. |
| `OBSIDIAN_RAG_STATIC_NO_CACHE` | unset (off) | `web/server.py:1221` | Si `1`, sirve los assets estáticos con `Cache-Control: max-age=0`. Útil cuando estás iterando sobre el frontend para evitar el cache del browser. |
| `OBSIDIAN_RAG_HOME_PREWARM` | `1` (on) | `web/server.py:5990` | Si `0`/`false`/`no`, desactiva el prewarm del home page (`/`) que carga el corpus + reranker en startup. Apagarlo si el server arranca demasiado lento en cold-boot. |
| `OBSIDIAN_RAG_CHAT_PREWARM_INTERVAL` | `240` (s) | `web/server.py:6034` | Cada cuántos segundos el web server hace un keep-alive call al chat model para evitar que Ollama lo descargue por inactividad. |

---

## Modelos / memoria / Ollama

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `-1` (forever) | `rag/__init__.py:2581` | Cuánto Ollama mantiene los modelos en memoria entre llamadas. Acepta int (segundos) o duration string (`30m`, `24h`). `-1` = pinear forever (default), `0` = liberar inmediato. |
| `RAG_KEEP_ALIVE_LARGE_MODEL` | unset (clamp automático) | `rag/__init__.py:2633` | Override del auto-clamp que el sistema aplica a modelos grandes (command-r) cuando `OLLAMA_KEEP_ALIVE=-1` y la RAM es justa. Solo lo necesitás con >64 GB y querer pinear "forever" sin clamping. |
| `RAG_MEMORY_PRESSURE_DISABLE` | unset (off) | `rag/__init__.py:13048` | Si `1`, apaga el watchdog que libera modelos cuando macOS reporta memory pressure. Setear en CI/tests donde no querés flapping. |
| `RAG_MEMORY_PRESSURE_THRESHOLD` | `85` (%) | `rag/__init__.py:13055` | % de RAM usada que dispara el watchdog. Bajalo (75) para liberar antes; subilo (90) si tenés RAM de sobra. |
| `RAG_MEMORY_PRESSURE_INTERVAL` | `60` (s) | `rag/__init__.py:13059` | Cada cuántos segundos el watchdog mira la presión de memoria. |
| `RAG_RERANKER_IDLE_TTL` | `900` (15 min) | `rag/__init__.py:12638` | Segundos que el cross-encoder se queda en MPS sin uso antes de descargarse. Idle-unload reduce ~2-3 GB de VRAM ocupada cuando no estás queriando. |
| `RAG_RERANKER_NEVER_UNLOAD` | unset (off) | `rag/__init__.py` (ver `web/server.py:1605`, 1662) | Si truthy, pinea el reranker en memoria para siempre. Cuesta ~2-3 GB pero elimina el ~9s de re-load en cold queries. |
| `RAG_RERANKER_FT_PATH` | unset (off) | `rag/__init__.py:12654` | Path al cross-encoder fine-tuned (gate GC#2.C). Cuando hay model promovido apunta al symlink `~/.cache/obsidian-rag/reranker-ft-current`. |
| `RAG_LOCAL_EMBED` | unset (auto-on en CLI one-shot) | `rag/__init__.py:10952` | Si truthy, embebe queries en-proceso con sentence-transformers (10-30 ms) en vez de Ollama (~140 ms). Requiere bge-m3 cacheado en HuggingFace. |
| `RAG_LOCAL_EMBED_WAIT_MS` | `6000` | `rag/__init__.py:18530` | Milisegundos que `retrieve()` espera a que el warmup del embedder local dispare el `Event` antes de fallback a Ollama. Bumped 4000→6000 el 2026-04-23 tras observar embed_ms=4005 exacto en producción. |
| `OBSIDIAN_RAG_WEB_CHAT_MODEL` | `qwen2.5:7b` | `rag/__init__.py:39220`, `web/server.py:1280` | Override del chat model del web server. Generado al plist en `rag setup`. |
| `RAG_POSTPROCESS_MODEL` | unset (= chat model) | `rag/__init__.py:136` | Override del LLM usado en stages de postprocess (citation-repair, NLI repair, critique). `helper` = qwen2.5:3b, `chat`/`legacy` = command-r. |
| `RAG_LOOKUP_MODEL` | `qwen2.5:3b` | `rag/__init__.py:196` | Modelo del fast-path dispatch. Swap por otro model si querés probar. |
| `RAG_LOOKUP_NUM_CTX` | `4096` | `rag/__init__.py:205` | num_ctx del fast-path. Bumped 2048→4096 el 2026-04-22 por refuses falsos cuando el CONTEXTO se inflaba con tools. |
| `RAG_LOOKUP_THRESHOLD` | `0.6` | `rag/__init__.py:195` | Top-1 rerank score mínimo para activar el fast-path (qwen2.5:3b + num_ctx=4096, skip citation-repair). |
| `YOUTUBE_API_KEY` | `""` | `rag/__init__.py:39215`, `web/server.py:3732` | API key opcional para YouTube transcript fetching. Sin esto, fallback a `yt-dlp --write-auto-sub`. |
| `RAG_VLM_MODEL` | `qwen2.5vl:3b` | `rag/ocr.py:364` | Modelo VLM para captioning de imágenes embebidas (complementario a OCR). |

---

## Embedding + Reranker + Cross-Source

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_EMBED_CACHE_MAX` | `512` | `rag/__init__.py:10773` | Cap del LRU cache de embeddings (query-side). Subilo a 2048 si tenés RAM y queries con mucha repetición. |
| `RAG_RERANK_FASTPATH_DIST` | `0.10` | `rag/__init__.py:12517` | Cosine-distance threshold del fast-path del reranker (ver sección Recientes). |
| `RAG_CONTEXT_CACHE_MAX` | `5000` | `rag/__init__.py:2908` | Cap del cache LRU de context summaries (per-document). |
| `RAG_SYNTHETIC_Q_CACHE_MAX` | `5000` | `rag/__init__.py:3159` | Cap del cache de synthetic questions. |
| `RAG_WIKI_CACHE_MAX` | `5000` | `rag/__init__.py:8390` | Cap del cache del wikilink resolver. |
| `RAG_CROSS_SOURCE_DEDUP_THRESHOLD` | `0.7` | `rag/__init__.py:2123` | Threshold Jaccard del dedup cross-source (ver sección Recientes). |

---

## Retrieval / Pipeline / Routing

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_ADAPTIVE_ROUTING` | `1` (on tras 2026-04-22) | `rag/__init__.py:252` | Activa el pipeline adaptativo (skip reformulate en metadata-only intents + fast-path dispatch). Sin regresión eval — ambas runs ON/OFF bit-idénticas. |
| `RAG_ENTITY_LOOKUP` | `1` (on tras 2026-04-21) | `rag/__init__.py:277` | Activa el dispatch a `handle_entity_lookup()` para queries tipo "todo sobre Astor". Requiere `rag_entities` poblada por `scripts/backfill_entities.py`. |
| `RAG_EXTRACT_ENTITIES` | `1` (on tras 2026-04-21) | `rag/__init__.py:49974` | Activa la extracción de entidades (GLiNER) durante el indexing. Apagar en reindex grande para acelerar. |
| `RAG_ENTITY_CONFIDENCE_MIN` | `0.70` | `rag/__init__.py:49887` | Umbral del score de GLiNER al extraer entidades. <0.7 se descarta. |
| `RAG_ADAPTIVE_K` | unset (off) | `rag/__init__.py:18968` | Activa adaptive-k (reduce `k` cuando el score gap cae). |
| `RAG_ADAPTIVE_K_MIN` | `2` | `rag/__init__.py:18976` | Floor del adaptive-k. |
| `RAG_ADAPTIVE_K_GAP` | `0.35` | `rag/__init__.py:18977` | Gap ratio entre top-score y cutoff que define cuándo cortar el adaptive-k. |
| `RAG_MMR_DIVERSITY` | unset (off) | `rag/__init__.py:47487` | Activa el MMR diversity re-order post-rerank. |
| `RAG_MMR_LAMBDA` | `0.7` | `rag/__init__.py:18956` | Lambda del MMR — 0=max diversity, 1=max relevance. |
| `RAG_MMR_POOL_MULTIPLIER` | `3.0` | `rag/__init__.py:18959` | Pool de candidatos del MMR (k * multiplier). |
| `RAG_PPR_TOPIC` | unset (off) | `rag/__init__.py:18762`, `47494` | Activa Personalized PageRank sobre el grafo de wikilinks usando los top-k results como seeds. Experimental. |
| `RAG_PPR_SEED_K` | `5` | `rag/__init__.py:18767` | Cantidad de top-k candidates usados como seeds del PPR. |
| `RAG_TEMPORAL_LOOKUP_BOOST` | `3.0` | `rag/__init__.py:18734` | Multiplier de recency para queries con anchor temporal detectado (`hoy`, `esta semana`). |
| `RAG_COMPARISON_POOL` | `30` | `rag/__init__.py:213` | Rerank pool size cuando intent es `comparison`. |
| `RAG_SYNTHESIS_POOL` | `30` | `rag/__init__.py:214` | Rerank pool size cuando intent es `synthesis`. |
| `RAG_SCORE_CALIBRATION` | unset (off) | `rag/__init__.py:47470` | Activa isotonic regression per-source (usa tabla `rag_score_calibration`). |
| `RAG_LLM_INTENT` | unset (off) | `rag/__init__.py:47477` | Reemplaza el intent classifier regex por un LLM call (qwen2.5:3b). Experimental. |
| `RAG_LLM_INTENT_TIMEOUT` | `2.0` (s) | `rag/__init__.py:13843` | Timeout del LLM intent classifier. |
| `RAG_LLM_INTENT_SHADOW` | unset (off) | (ver `rag.classify_intent`) | Shadow mode del LLM intent classifier — loguea ambas predicciones sin cambiar routing. |
| `RAG_EXPAND_MIN_TOKENS` | `6` | `rag/__init__.py:13355` | Queries más cortas que N tokens saltean la paráfrasis. Bumpeado de 4→6 para evitar paraphrases ruidosas en queries cortas. |
| `RAG_EXPAND_TIMEOUT_S` | `3.0` | `rag/__init__.py:13369` | Timeout del `expand_queries()` (qwen2.5:3b). Post-timeout → devolver [query] sola. |
| `RAG_LEARNED_PARA_MIN_HITS` | `2` | `rag/__init__.py:13373` | Hits mínimos para que una paraphrase aprendida entre al reranker. Evita noise de one-offs. |
| `RAG_CITATION_REPAIR_MAX_BAD` | `2` | `rag/__init__.py:123` | Si la respuesta tiene >N paths inventados, saltea el citation-repair (que tarda ~5-8s). |
| `RAG_PARALLEL_POSTPROCESS` | `1` (on) | `rag/__init__.py:49639` | Paraleliza stages de postprocess (citation repair + NLI + diversity en threads). `0` desactiva — debug. |
| `RAG_WIKILINK_EXPANSION` | unset (off) | `rag/__init__.py:10452` | Expande wikilinks en el CONTEXT del retrieve (resolver link targets a body). Experimental. |
| `RAG_UNIFIED_CHAT` | `1` (on) | `rag/__init__.py:17852` | Activa el unified chat pipeline (`run_chat_turn`) para CLI `rag chat` — alineación con web/serve. |
| `RAG_CONTEXT_SUMMARY` | `1` (on) | `rag/__init__.py:3075` | Master switch del context summary prefix en chunks. `0` desactiva. |
| `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY` | unset | `rag/__init__.py:3073` | Override binario — usado por tests para desactivar el context summary sin tocar `RAG_CONTEXT_SUMMARY`. |
| `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q` | unset | `rag/__init__.py:3246` | Idem para synthetic questions. |
| `RAG_TOKENS_PER_CHAR` | `0.25` | `rag/__init__.py:46552` | Factor char→token para estimar prompt budget sin llamar al tokenizer real (~4 chars/token, qwen2.5 promedio). |
| `RAG_CONTEXT_BUDGET_WARN` | `0.80` | `rag/__init__.py:46553` | % del num_ctx que dispara warning `_silent_log` cuando el prompt se acerca al límite. |
| `RAG_EXPLORE` | unset (off) | `rag/__init__.py:19080` | ε-exploration: 10% chance de meter un resultado de menor rank entre los top-3. Sirve para counterfactuals. **No activar durante `rag eval`**. |
| `RAG_TRACK_OPENS` | unset (off) | `rag/__init__.py:330` | Cambia el scheme de OSC-8 links de `file://` a `x-rag-open://` para trackear clicks. |
| `RAG_RETRIEVE_TIMING` | unset (off) | `rag/__init__.py:19127` | Debug — dump per-stage timing breakdown de `retrieve()` a stderr. |
| `RAG_NO_WARMUP` | unset (off) | `rag/__init__.py:13113` | Skippea el warmup del reranker + bge-m3 + corpus cache en startup. |
| `RAG_DEBUG` | unset (off) | `rag/__init__.py:11030` | Verbose stderr en path de embed local + otras subsystems. |
| `RAG_FAST_PATH_KEEP_WITH_TOOLS` | unset (off) | (ver `web/server.py /api/chat`) | Rollback del auto-downgrade en `/api/chat` cuando el pre-router matchea tools estando en fast-path. |
| `RAG_WA_FAST_PATH` | unset (off) | `rag/__init__.py:19208` | Activa fast-path WhatsApp (qwen2.5:3b + low threshold) para queries del listener. |
| `RAG_WA_FAST_PATH_THRESHOLD` | `0.05` | `rag/__init__.py:19219` | Threshold del fast-path WhatsApp. Más bajo que el general (0.6) porque las queries de WA suelen ser cortas y ambiguas. |
| `RAG_WA_SKIP_PARAPHRASE` | unset (off) | (ver `rag.expand_queries`) | Skippea paraphrase para queries con marker `wa:`. |
| `RAG_DEEP_MAX_SECONDS` | `30` | `rag/__init__.py:19279` | Cap absoluto de tiempo total para queries `--deep` antes de cortar y devolver lo que tenga. |
| `RAG_DEEP_HIGH_CONF_BYPASS` | `0.8` | `rag/__init__.py:19289` | Si la confidence del rerank top-1 supera este valor, deep-mode bypassea las refinement loops adicionales. |
| `RAG_DEEP_LOW_CONF_BYPASS` | `0.0` | `rag/__init__.py:19304` | Si la confidence top-1 cae por debajo, deep-mode también bypassea (no vale la pena gastar más LLM). |
| `RAG_AGENT_UNPRODUCTIVE_CAP` | `3` | `rag/__init__.py:29893` | Cuántas iteraciones sin tool-call útil tolera el `rag do` agent loop antes de abortar. |
| `RAG_EVAL_GATE_SINGLES_MIN` | `0.60` | `rag/__init__.py:27553` | hit@5 mínimo aceptable en `rag eval` sobre singles antes de fallar el gate. |
| `RAG_EVAL_GATE_CHAINS_MIN` | `0.73` | `rag/__init__.py:27556` | chain_success mínimo en `rag eval` sobre chains. |

---

## Semantic response cache (GC#1, 2026-04-23)

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_CACHE_ENABLED` | `1` (on) | `rag/__init__.py:4469` | Master switch del semantic response cache. `0` desactiva lookup + store. Tests autouse lo fuerzan a `0`. |
| `RAG_CACHE_COSINE` | `0.93` | `rag/__init__.py:4465` | Umbral de cosine similarity sobre embeddings bge-m3 para considerar un cache hit. |
| `RAG_CACHE_TTL_DEFAULT` | `86400` (24h) | `rag/__init__.py:4466` | TTL para entries con intents `semantic`/`synthesis`/`count`/`list`/`comparison`. |
| `RAG_CACHE_TTL_RECENT` | `600` (10 min) | `rag/__init__.py:4467` | TTL para entries con intents `recent`/`agenda` (data time-sensitive). |
| `RAG_CACHE_MAX_ROWS` | `2000` | `rag/__init__.py:4468` | Cap de filas a escanear por lookup (lineal, cheap hasta unas miles). Cleanup manual via `rag cache clear`. |
| `RAG_WEB_CHAT_CACHE_TTL` | `86400` (24h) | `web/server.py:6172` | TTL específico del cache de respuestas del web chat (separado del semantic cache CLI). |

---

## Ambient Agent (anticipatory + ambient hook)

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_AMBIENT_DISABLED` | unset | `rag/__init__.py:20903` | Kill switch global del ambient hook (ver sección Recientes). |
| `RAG_ANTICIPATE_DISABLED` | unset | `rag/anticipatory.py:525` | Kill switch del anticipatory agent (sub-feature del ambient). |
| `RAG_ANTICIPATE_MIN_SCORE` | `0.35` | `rag/anticipatory.py:110` | Score mínimo para que una señal anticipatoria entre al digest. |
| `RAG_ANTICIPATE_DEDUP_WINDOW_HOURS` | `24` | `rag/anticipatory.py:112` | Ventana de dedup de señales — la misma signal no dispara 2 veces en N horas. |
| `RAG_ANTICIPATE_CALENDAR_MIN_MIN` | `15` | `rag/anticipatory.py:115` | Minutos mínimos antes del evento de calendar para empezar a emitir prep signals. |
| `RAG_ANTICIPATE_CALENDAR_MAX_MIN` | `90` | `rag/anticipatory.py:118` | Minutos máximos — más allá no emite (ruido). |
| `RAG_ANTICIPATE_ECHO_MIN_AGE_DAYS` | `60` | `rag/anticipatory.py:121` | Edad mínima de la nota para que califique como "echo" (no es novedoso si es reciente). |
| `RAG_ANTICIPATE_ECHO_MIN_COSINE` | `0.70` | `rag/anticipatory.py:124` | Cosine similarity mínimo para echo matching. |
| `RAG_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS` | `7` | `rag/anticipatory.py:127` | Edad mínima de un commitment para empezar a recordarlo. |
| `RAG_ANTICIPATE_QUIET_NIGHT_START` | `22:00` | `rag_anticipate/quiet_hours.py:49` | Inicio del quiet period (no notificaciones). |
| `RAG_ANTICIPATE_QUIET_NIGHT_END` | `08:00` | `rag_anticipate/quiet_hours.py:53` | Fin del quiet period. |
| `RAG_ANTICIPATE_BYPASS_QUIET` | unset (off) | `rag_anticipate/quiet_hours.py:156` | Si `1`/`true`, ignora el quiet period. Debug. |
| `RAG_ANTICIPATE_SMOKE` | unset (off) | `rag_anticipate/signals/_smoke_test.py:20` | Habilita el smoke-test runner del anticipatory. |
| `RAG_WEB_BASE_URL` | `http://127.0.0.1:8765` | `rag_anticipate/voice.py:24` | URL base del web server desde el anticipatory voice subsystem. |

---

## OCR / VLM / detección de citas

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_OCR` | unset (on) | `rag/ocr.py:272` | Master switch de OCR de imágenes (Apple Vision). `0` desactiva. |
| `RAG_VLM_CAPTION` | unset (off) | `rag/ocr.py:393` | Activa captioning con qwen2.5-vl en imágenes embebidas. Complementa OCR (no lo reemplaza). |
| `RAG_VLM_MODEL` | `qwen2.5vl:3b` | `rag/ocr.py:364` | Modelo VLM. |
| `RAG_VLM_CAPTION_MAX_PER_RUN` | `500` | `rag/ocr.py:379` | Cap de imágenes captioned por corrida. Bound contra runs multi-hora en vaults grandes. |
| `RAG_CITA_DETECT` | unset (off) | `rag/ocr.py:757` | Activa detector de citas médicas/judiciales en chunks (flag de fechas de vencimiento). |
| `RAG_OCR_HISTORICAL_MAX_DAYS` | `90` (default constant) | `web/server.py:2528` | Tope de días hacia atrás para reprocesar OCR sobre uploads históricos (ver sección Recientes). |

---

## NLI grounding + contradiction radar

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_NLI_GROUNDING` | unset (off) | `rag/__init__.py:165` | Activa NLI grounding post-citation-repair. Carga mDeBERTa-NLI + extrae claims + clasifica. |
| `RAG_NLI_CONTRADICTS_THRESHOLD` | `0.7` | `rag/__init__.py:170` | Score de la clase `contradiction` por encima del cual el claim se flagea. |
| `RAG_NLI_MAX_CLAIMS` | `20` | `rag/__init__.py:182` | Cap de claims extraídos del LLM response para chequear NLI. |
| `RAG_NLI_SKIP_INTENTS` | `count,list,recent,agenda` | `rag/__init__.py:176` | Intents para los que skipea NLI (no aplica a metadata-only). |
| `RAG_NLI_IDLE_TTL` | `900` | `rag/__init__.py:49253` | TTL idle-unload del modelo NLI (mismo patrón que reranker). |
| `RAG_NLI_NEVER_UNLOAD` | unset (off) | `rag/__init__.py:49254` | Pinea el NLI model en memoria (evita idle-unload). |
| `_CONTRADICTION_ASYNC` | `1` (on) | `rag/__init__.py:20361` | Async default para el contradiction detector. `0` para sync (debug). |

---

## Telemetry + logging async

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_LOG_QUERY_ASYNC` | unset (off) | `rag/__init__.py:1174` | Si truthy, loguea queries a `rag_queries` en un thread async (no bloquea la respuesta). |
| `RAG_LOG_BEHAVIOR_ASYNC` | unset (off) | `rag/__init__.py:1196` | Idem para `rag_behavior` (clicks, opens, copies). |
| `RAG_METRICS_ASYNC` | unset (off) | `web/server.py:12853` | Idem para métricas del web server (latencia, SSE counts). |
| `RAG_SILENT_LOG_ALERT_THRESHOLD` | `20` | `rag/__init__.py:902` | N silent-errors por ventana que dispara alert a stderr. Evita que SQL fail queden invisible. |
| `RAG_WAL_CHECKPOINT_DISABLE` | unset (off) | `rag/__init__.py:12880` | Apaga el checkpoint manual del WAL del telemetry.db. |
| `RAG_WAL_CHECKPOINT_INTERVAL` | `30` (s) | `rag/__init__.py:12883` | Cada cuántos segundos el background checkpoint del WAL corre. |
| `RAG_STATE_SQL` | unset (no-op) | `rag/__init__.py:4947` | **[deprecated]** Históricamente activaba el SQL store. Post-T10 (2026-04-20) es no-op — SQL es el único path. Setear en plists para trail. |

---

## WhatsApp / Listener / Scheduled send

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `RAG_SSE_MAX_PER_IP` | `4` | `web/server.py:12446` | Cap de SSE simultáneas por IP (ver sección Recientes). |
| `RAG_CHAT_UPLOADS_TTL_DAYS` | `30` | `rag/__init__.py:43415`, `44586` | TTL del dir `chat-uploads/` (ver sección Recientes). |
| `RAG_CRITIQUE_ENABLED` | `0` (off) | `web/server.py:2210` | Activa critique post-respuesta en el web chat (auto-revisión de la respuesta). |
| `RAG_WEB_PROMPT_VERSION` | `v2` | `web/server.py:1479` | Selecciona la versión del system prompt del web chat (`v1`/`v2`). Rollback path si `v2` introduce regresión. |

---

## Privacy / offline / CI

| Variable | Default | Archivo | Descripción |
|---|---|---|---|
| `HF_HUB_OFFLINE` | (recomendado `1`) | (HuggingFace lib) | Reranker se carga del caché local de HuggingFace, sin tocar la red. |
| `TRANSFORMERS_OFFLINE` | (recomendado `1`) | (HuggingFace lib) | Idem para `transformers`. |
| `FASTEMBED_CACHE_PATH` | `~/.cache/fastembed` | `rag/__init__.py:44`, plists | Directorio persistente para los modelos ONNX que carga `fastembed` (usado por `mem0` para BM25 sparse vectors / hybrid search en Qdrant). El default upstream es `tempfile.gettempdir()/fastembed_cache` → en macOS resuelve a `/var/folders/.../T/fastembed_cache`, que el SO limpia periódicamente. Pinearlo a `$HOME` evita que `HF_HUB_OFFLINE=1` + cache miss tras GC dejen al encoder sin poder cargar (ver web.error.log 2026-04-29). Población inicial: `python -c 'from fastembed import SparseTextEmbedding; SparseTextEmbedding("Qdrant/bm25")'` con offline mode desactivado. |
| `RAG_FT_DEVICE` | `cpu` | `scripts/finetune_reranker.py:238` | Device para entrenar el cross-encoder fine-tuned. `cpu`/`mps`/`cuda`. |
| `RAG_FINETUNE_MIN_CORRECTIVES` | `20` | `scripts/finetune_reranker.py:481` | Cap mínimo de feedback rows correctivos para que el fine-tune dispare. Gate GC#2.C. |
| `RAG_AUTO_HARVEST_MIN_CONF` | `0.8` | `rag/__init__.py:48936` | Confidence mínimo del LLM-as-judge para aceptar una query low-conf como label en el auto-harvest nightly. |
| `RAG_AUTO_HARVEST_SNIPPET_CHARS` | `400` | `rag/__init__.py:48939` | Budget de context text que el judge ve por candidate path en el auto-harvest. |

---

## Cómo se generó este catálogo

```bash
grep -rE 'os\.environ\.get\("[A-Z_]+|os\.getenv\("[A-Z_]+' \
  rag/ web/ scripts/ rag_anticipate/ \
  | grep -oE '"[A-Z_]+(_[A-Z_]+)*"' \
  | sort -u
```

Resultado al 2026-04-25: **~95 env vars únicas** distribuidas en `rag/__init__.py` (~85), `web/server.py` (~12), `rag/ocr.py` (5), `rag/anticipatory.py` (8), `rag_anticipate/` (5), `scripts/finetune_reranker.py` (2). El conteo varía según iteramos — re-correr el grep antes de comparar.

---

## Notas sobre líneas

Los `path:line` apuntan al sitio de **lectura primaria** (donde se define el default canónico). Cuando una variable se lee en >1 sitio (típico: `OBSIDIAN_RAG_VAULT` se lee 6+ veces), agregamos las otras ubicaciones entre paréntesis. Si modificás esos números (refactor, agregar lógica), **regenerá el catálogo** o mantené esta tabla en sync — sin sync, los path:line se vuelven mentiras.
