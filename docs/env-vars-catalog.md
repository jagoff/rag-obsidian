# Env vars — catálogo completo

**Audit 2026-04-24**: `rag.py` consulta **85 env vars únicas** via `os.environ.get()`. El CLAUDE.md documenta ~38 de ellas (las "críticas", con rollback paths + historia). Este doc complementa con las **47 restantes** que el CLAUDE.md no cubre.

Para cambiar cualquier default: setear el env var en el shell antes de invocar el CLI, o agregarlo al `<key>EnvironmentVariables</key>` del plist correspondiente en `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist`. Tras editar un plist, aplicar con `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/...plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/...plist` (kickstart no relee env).

## Config / Defaults ajustables

| Variable | Default | Ubicación | Qué hace |
|---|---|---|---|
| `RAG_ADAPTIVE_K_MIN` | `2` | `rag.py:16944` | Floor del adaptive-k (mínimo de results devuelto por `retrieve()` cuando la ventana de scoring es estrecha). |
| `RAG_ADAPTIVE_K_GAP` | `0.35` | `rag.py:16945` | Gap ratio entre el top-score y la cutoff — define cuándo cortar el adaptive-k. |
| `RAG_AGENT_UNPRODUCTIVE_CAP` | `3` | `rag.py:26577` | Cuántas iteraciones sin tool-call útil tolera el `rag do` agent loop antes de abortar. |
| `RAG_AUTO_HARVEST_MIN_CONF` | `0.8` | `rag.py:46705` | Confidence mínimo para que el auto-harvest LLM-as-judge (nightly 03:30) acepte una query low-conf como label para el gate GC#2.C. |
| `RAG_AUTO_HARVEST_SNIPPET_CHARS` | `400` | `rag.py:46708` | Budget de context text que el judge ve por candidate path en el auto-harvest. |
| `RAG_CACHE_MAX_ROWS` | `2000` | `rag.py:4057` | Cap total de filas en `rag_response_cache` (semantic cache). LRU cuando supera. |
| `RAG_CACHE_TTL_DEFAULT` | `86400` (24h) | `rag.py:4055` | TTL default para entries del semantic cache. |
| `RAG_CACHE_TTL_RECENT` | `600` (10min) | `rag.py:4056` | TTL corto para queries con intent `recent`/`agenda` (data time-sensitive). |
| `RAG_COMPARISON_POOL` | `30` | `rag.py:216` | Rerank pool size cuando intent es `comparison` (más candidates para cubrir ambos polos). |
| `RAG_SYNTHESIS_POOL` | `30` | `rag.py:217` | Idem para `synthesis` (cross-source). |
| `RAG_CONTEXT_BUDGET_WARN` | `0.80` | `rag.py:44319` | % del `num_ctx` que dispara un warning `_silent_log` cuando el prompt real se acerca al límite. Instrumentación. |
| `RAG_CONTEXT_CACHE_MAX` | `5000` | `rag.py:2578` | Cap del context-summary cache (LRU). |
| `RAG_SYNTHETIC_Q_CACHE_MAX` | `5000` | `rag.py:2829` | Cap del synthetic-questions cache. |
| `RAG_ENTITY_CONFIDENCE_MIN` | `0.70` | `rag.py:47656` | Umbral para el score de GLiNER al extraer entidades (persons/orgs). <0.7 se descarta. |
| `RAG_EXPAND_TIMEOUT_S` | `3.0` | `rag.py:11395` | Timeout del `expand_queries()` (qwen2.5:3b paraphrase call). Post-timeout → devolver [query] sin paraphrases. |
| `RAG_LEARNED_PARA_MIN_HITS` | `2` | `rag.py:11399` | Hits mínimos para que una paraphrase aprendida entre al reranker. Evita noise de one-offs. |
| `RAG_LLM_INTENT_TIMEOUT` | `2.0` | `rag.py:11865` | Timeout del classifier LLM-based intent (opt-in via `RAG_LLM_INTENT=1`). |
| `RAG_LOOKUP_THRESHOLD` | `0.6` | `rag.py:196` | Top-1 rerank score mínimo para activar el fast-path dispatch (qwen2.5:3b + num_ctx=4096, skip citation-repair). |
| `RAG_LOOKUP_MODEL` | `qwen2.5:3b` | `rag.py:197` | Modelo LLM del fast-path. Swap por otro model si querés probar. |
| `RAG_MMR_LAMBDA` | `0.7` | `rag.py:16924` | Lambda del MMR (Maximal Marginal Relevance) — 0 = max diversity, 1 = max relevance. |
| `RAG_MMR_POOL_MULTIPLIER` | `3.0` | `rag.py:16927` | Pool de candidatos que el MMR recibe (k * multiplier). |
| `RAG_NLI_CONTRADICTS_THRESHOLD` | `0.7` | `rag.py:169` | Score de la clase "contradiction" de mDeBERTa-NLI por encima del cual el claim se flagea. |
| `RAG_NLI_MAX_CLAIMS` | `20` | `rag.py:181` | Cap de claims extraídos del LLM response para chequear NLI. |
| `RAG_NLI_NEVER_UNLOAD` | `""` (off) | `rag.py:47023` | Setear `1` para pinear el NLI model en memoria (evita idle-unload). Análogo a `RAG_RERANKER_NEVER_UNLOAD`. |
| `RAG_PPR_SEED_K` | `5` | `rag.py:16747` | Cantidad de top-k candidates usados como seeds del Personalized PageRank (opt-in via `RAG_PPR_TOPIC=1`). |
| `RAG_SILENT_LOG_ALERT_THRESHOLD` | `20` | `rag.py:899` | N silent-errors / ventana que dispara el alert a stderr (evita que los logs SQL fail queden invisible). |
| `RAG_TEMPORAL_LOOKUP_BOOST` | `3.0` | `rag.py:16714` | Multiplier de recency para queries con anchor temporal detectado (`hoy`, `esta semana`). |
| `RAG_TOKENS_PER_CHAR` | `0.25` | `rag.py:44318` | Factor de conversión char→token para estimar el budget del prompt sin llamar al tokenizer real. 0.25 asume ~4 chars/token (promedio qwen2.5). |
| `RAG_VLM_CAPTION_MAX_PER_RUN` | `500` | `rag.py:19292` | Cap de imágenes captioned por corrida del VLM. Bound para evitar runs multi-hora en vaults grandes. |

## Feature flags (opt-in, default OFF)

| Variable | Activar con | Ubicación | Qué hace |
|---|---|---|---|
| `RAG_ADAPTIVE_K` | `1`/`true`/`yes` | `rag.py:16936` | Activa adaptive-k (dinámicamente reduce `k` cuando el score gap cae). Default off, ver `RAG_ADAPTIVE_K_MIN`/`GAP`. |
| `RAG_CITA_DETECT` | `1` | `rag.py:19640` | Activa detector de citas médicas/judiciales en chunks para flaggear fechas de vencimiento. |
| `RAG_CONTEXT_SUMMARY` | `0` | `rag.py:2745` | Opt-OUT del context summary prefix en chunks (default ON — ver `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY`). |
| `RAG_LLM_INTENT` | `1` | `rag.py:45243` | Reemplaza el regex-based intent classifier por un LLM call (qwen2.5:3b). Experimental — eval drift +CI. |
| `RAG_MMR_DIVERSITY` | `1` | `rag.py:45253` | Activa el MMR diversity re-order post-rerank. Default OFF — afecta scoring. |
| `RAG_PARALLEL_POSTPROCESS` | `0` | `rag.py:47408` | Default ON — setear a `0` desactiva paralelización de stages de postprocess (citation repair + NLI + diversity en threads). Debug. |
| `RAG_PPR_TOPIC` | `1` | `rag.py:16742` | Activa Personalized PageRank sobre el grafo de wikilinks usando los top-k results como seeds. Experimental. |
| `RAG_SCORE_CALIBRATION` | `1` | `rag.py:45236` | Activa el isotonic regression per-source (usa `rag_score_calibration`). Mejora rerank bajo mixed-source retrieval. |
| `RAG_UNIFIED_CHAT` | `1` | `rag.py:15873` | Activa el unified chat pipeline (`run_chat_turn`) para CLI `rag chat` — alineación con web/serve. |
| `RAG_VLM_CAPTION` | `1` | `rag.py:19306` | Activa captioning de imágenes embebidas via VLM (qwen2.5-vl). OCR via Apple Vision sigue siendo el path default — VLM es complementario. |
| `RAG_WIKILINK_EXPANSION` | `1` | `rag.py:9092` | Expande wikilinks en el CONTEXT del retrieve (resolver link targets a body). Experimental. |

## Infraestructura / internal

| Variable | Default | Ubicación | Qué hace |
|---|---|---|---|
| `_CONTRADICTION_ASYNC` | `1` (on) | `rag.py:18215` | Async default para el contradiction detector. Setear `0` para sync (útil si debugueás un falso positivo). |
| `OBSIDIAN_RAG_WEB_CHAT_MODEL` | `qwen2.5:7b` | `rag.py:36775` | Override del chat model del web server. Generado al plist en `rag setup`. |
| `RAG_CACHE_ENABLED` | `1` (on) | `rag.py:4058` | Master switch del semantic cache. Tests autouse lo fuerzan a `0`. |
| `RAG_NO_WARMUP` | `""` (off) | `rag.py` | Debug — skippea el warmup del reranker + bge-m3 + corpus cache en startup. |
| `RAG_OCR` | `""` (on) | `rag.py` | Opt-out del OCR de imágenes embebidas (Apple Vision). `0` desactiva. |
| `RAG_POSTPROCESS_MODEL` | `""` | `rag.py:135` | Override del model LLM usado en stages de postprocess (critique, NLI repair). Default = chat model. |
| `RAG_RERANKER_FT_PATH` | `""` (off) | `rag.py:10878` | Path al cross-encoder fine-tuned (gate GC#2.C). Cuando hay model promovido, apunta al symlink `~/.cache/obsidian-rag/reranker-ft-current`. |
| `RAG_RETRIEVE_TIMING` | `""` (off) | `rag.py` | Debug — dump per-stage timing breakdown de `retrieve()` a stderr. |
| `RAG_STATE_SQL` | `1` | `rag.py` | Historicamente activaba el SQL path. Post-T10 es no-op, setear en plists para trail. |
| `RAG_TRACK_OPENS` | `""` (off) | `rag.py` | Switch del OSC-8 terminal link scheme de `file://` a `x-rag-open://`. Opt-in para trackeo de clicks. |
| `YOUTUBE_API_KEY` | `""` | `rag.py:36770` | API key opcional para YouTube transcript fetching. Sin esto, el fallback es `yt-dlp --write-auto-sub`. |

---

## Cómo se generó este catálogo

```bash
grep -oh 'os.environ.get("[A-Z_]\+' rag.py | sed 's/.*"//' | sort -u
# → 85 env vars
grep -oE '`RAG_[A-Z_]+`|`OBSIDIAN_RAG_[A-Z_]+`|...' CLAUDE.md | tr -d '`' | sort -u
# → 38 documented
# → diff = 47 missing (este doc)
```

Script exacto del audit en `scripts/audit_env_vars.py` (TODO — no committed yet).

---

## Importantes (ya documentadas en CLAUDE.md)

Las siguientes env vars tienen explicación extensa + historia + rollback paths en [`CLAUDE.md § Env vars`](../CLAUDE.md). Mencionadas aquí solo para cross-reference:

`OBSIDIAN_RAG_VAULT`, `OBSIDIAN_RAG_NO_APPLE`, `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY`, `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q`, `OBSIDIAN_RAG_MOZE_FOLDER`, `OBSIDIAN_RAG_INDEX_WA_MONTHLY`, `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS`, `OBSIDIAN_RAG_BIND_HOST`, `OBSIDIAN_RAG_ALLOW_LAN`, `OLLAMA_KEEP_ALIVE`, `RAG_TIMEZONE`, `RAG_KEEP_ALIVE_LARGE_MODEL`, `RAG_MEMORY_PRESSURE_DISABLE`/`THRESHOLD`/`INTERVAL`, `RAG_EXPLORE`, `RAG_RERANKER_IDLE_TTL`/`NEVER_UNLOAD`, `RAG_LOCAL_EMBED`/`WAIT_MS`, `RAG_FAST_PATH_KEEP_WITH_TOOLS`, `RAG_ENTITY_LOOKUP`/`EXTRACT_ENTITIES`, `RAG_NLI_GROUNDING`/`IDLE_TTL`/`SKIP_INTENTS`, `RAG_ADAPTIVE_ROUTING`, `RAG_LOOKUP_NUM_CTX`, `RAG_EXPAND_MIN_TOKENS`, `RAG_CITATION_REPAIR_MAX_BAD`, `RAG_WA_FAST_PATH`/`THRESHOLD`, `RAG_WA_SKIP_PARAPHRASE`, `RAG_DEEP_MAX_SECONDS`, `RAG_CACHE_COSINE`, `RAG_EVAL_GATE_SINGLES_MIN`/`CHAINS_MIN`, `RAG_DEBUG`, `RAG_LOG_QUERY_ASYNC`, `RAG_LOG_BEHAVIOR_ASYNC`, `RAG_METRICS_ASYNC`.
