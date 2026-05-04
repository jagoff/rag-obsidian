# Variables de entorno

Cosas que podés setear con `export` o poner adelante de un comando para cambiar el comportamiento sin editar código.

**Formato**: `EXPORT VAR=valor` en tu shell, o `VAR=valor rag <comando>` solo para una invocación.

---

## Las que seguramente vas a tocar

### `OBSIDIAN_RAG_VAULT`
Ruta absoluta al vault que querés usar, override del default.

```bash
export OBSIDIAN_RAG_VAULT=~/ruta/a/mi/vault
# O solo para esta invocación:
OBSIDIAN_RAG_VAULT=/tmp/test-vault rag index
```

**Precedencia**: esta env var gana sobre lo que hayas seteado con `rag vault use`. Para cambiar el vault persistentemente sin tocar env vars, usá `rag vault use <nombre>`.

### `RAG_TIMEZONE`
Zona horaria IANA para parsear strings ISO con tz. Default: `America/Argentina/Buenos_Aires`.

```bash
export RAG_TIMEZONE=America/New_York
```

### `OBSIDIAN_RAG_NO_APPLE=1`
Desactiva integraciones con Apple (Calendar, Reminders, Mail, Screen Time). Útil en Linux o si no diste Full Disk Access.

```bash
export OBSIDIAN_RAG_NO_APPLE=1
```

### `RAG_OCR=0`
Apaga OCR en imágenes embebidas (`![[captura.png]]`) durante el indexing. Por default está prendido en macOS.

```bash
export RAG_OCR=0          # apagar OCR
```

El OCR es lento la primera pasada (cacheado después). Si te importa el tiempo de indexing inicial, apagalo, pero perdés el texto dentro de las imágenes.

---

## Control de modelos y memoria

### `OLLAMA_KEEP_ALIVE`
Cuánto tiempo Ollama mantiene los modelos cargados entre llamadas. Default: `-1` (para siempre).

```bash
export OLLAMA_KEEP_ALIVE=-1       # (default) modelos cargados para siempre
export OLLAMA_KEEP_ALIVE=20m      # liberar después de 20 minutos idle
export OLLAMA_KEEP_ALIVE=0        # liberar inmediatamente (modo "pobre en RAM")
```

Si tu Mac se queda sin memoria con modelos grandes, subí este valor o bajalo a `20m`.

### `RAG_KEEP_ALIVE_LARGE_MODEL`
Override del auto-clamp que el sistema hace para modelos grandes (command-r). Solo lo necesitás si tenés >64 GB y querés pinearlos "forever".

```bash
export RAG_KEEP_ALIVE_LARGE_MODEL=4h
```

### `RAG_MEMORY_PRESSURE_DISABLE=1`
Desactiva el watchdog que libera modelos cuando el Mac está con presión de memoria. Usalo en tests/CI.

```bash
export RAG_MEMORY_PRESSURE_DISABLE=1
```

### `RAG_MEMORY_PRESSURE_THRESHOLD`
% de RAM usada que dispara el watchdog. Default: `85`.

```bash
export RAG_MEMORY_PRESSURE_THRESHOLD=80    # más agresivo
export RAG_MEMORY_PRESSURE_THRESHOLD=90    # más tolerante
```

### `RAG_MEMORY_PRESSURE_INTERVAL`
Cada cuántos segundos el watchdog mira la memoria. Default: `60`.

```bash
export RAG_MEMORY_PRESSURE_INTERVAL=30
```

### `RAG_RERANKER_IDLE_TTL`
Segundos que el reranker se queda en memoria sin usar antes de descargarse. Default: `900` (15 min).

```bash
export RAG_RERANKER_IDLE_TTL=300     # liberar después de 5 min sin usar
```

### `RAG_RERANKER_NEVER_UNLOAD=1`
Nunca descarga el reranker (pineado en VRAM). Cuesta ~2-3 GB de RAM pero elimina el ~9s de re-load.

```bash
export RAG_RERANKER_NEVER_UNLOAD=1
```

### `RAG_LOCAL_EMBED=1`
Embebe queries en-proceso con sentence-transformers (10-30 ms) en vez de Ollama (~140 ms). Requiere bge-m3 cacheado.

```bash
# Primero cacheá el modelo:
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
export RAG_LOCAL_EMBED=1
```

Para CLI one-shot (`rag query`, `rag chat`, `rag do`, `rag prep`, etc.) ya se activa solo. Usalo para `rag serve` / web server.

### `RAG_FAST_PATH_KEEP_WITH_TOOLS` (default off)
Rollback del auto-downgrade en `/api/chat` cuando el pre-router matchea tools estando en fast-path. Medido el 2026-04-23: query "qué pendientes tengo" disparaba `fast_path=True` (qwen2.5:3b + 4096 ctx) y además matchea el `_PLANNING_PAT` → `reminders_due + calendar_ahead` que inflan el CONTEXTO a 2-4K tokens. qwen2.5:3b prefillea eso a ~2.5ms/tok → `llm_prefill=11595ms, total=16.3s`.

El downgrade runtime cambia al modelo full (qwen2.5:7b, ~0.5ms/tok prefill) cuando detecta `_fast_path=True AND _forced_tools!=[]`. El marker `fast_path` en telemetry no cambia; se agrega `fast_path_downgraded=True` cuando dispara.

```bash
# Restaurar comportamiento pre-fix (mantener qwen2.5:3b aunque el
# pre-router haya inyectado tool output).
export RAG_FAST_PATH_KEEP_WITH_TOOLS=1
```

Solo afecta `/api/chat` (web). El CLI y `rag serve` no usan pre-router regex con tools, así que no necesitan el gate.

### `RAG_LOCAL_EMBED_WAIT_MS` (default `6000`)
Milisegundos que `retrieve()` espera para que el warmup del embedder local dispare el Event antes de caer a Ollama. **Default bumped 4000→6000 el 2026-04-23** tras observar producción: 4 CLI `query` consecutivas (2026-04-23T15:14-15:15) mostraron `embed_ms=4005` exacto — el wait timeaba justo antes del Event fire (~5s cold load en M3 Max), y el user pagaba 4s gratis de espera + 5ms de fallback Ollama warm.

Con 6000ms el Event dispara dentro del budget en ~5s, el user recibe el path MPS (~30ms encode) en la misma query. Si el warmup también timea (disk frío), el fallback Ollama warm sigue siendo ~140ms → cap final ~6.14s.

```bash
# Rollback a pre-2026-04-23
export RAG_LOCAL_EMBED_WAIT_MS=4000

# Legacy non-blocking (siempre fallback Ollama, sin esperar local)
export RAG_LOCAL_EMBED_WAIT_MS=0

# Macs con HDD externa o spinning disk (SentenceTransformer load >6s)
export RAG_LOCAL_EMBED_WAIT_MS=10000
```

Self-contained: no cambia warmup_async ni el path long-running (serve/web), donde el Event ya está set pre-query.

---

## Pipeline tuning

### `RAG_ADAPTIVE_ROUTING`
Activa el pipeline adaptativo: saltea pasos caros en queries simples. Default: ON desde 2026-04-22.

```bash
export RAG_ADAPTIVE_ROUTING=0        # desactivar
export RAG_ADAPTIVE_ROUTING=1        # activar explícito (default)
```

### `RAG_ENTITY_LOOKUP`
Activa el dispatch a `handle_entity_lookup()` para queries tipo "todo sobre Astor". Default: ON desde 2026-04-21.

```bash
export RAG_ENTITY_LOOKUP=0       # desactivar
```

### `RAG_EXTRACT_ENTITIES`
Extrae entidades (personas, orgs, lugares) con GLiNER durante el indexing. Default: ON desde 2026-04-21.

```bash
export RAG_EXTRACT_ENTITIES=0    # desactivar (útil en reindex grande para ir más rápido)
```

### `RAG_NLI_GROUNDING`
Verifica que cada claim de la respuesta esté soportada por los chunks recuperados. Default: OFF (es lento).

```bash
export RAG_NLI_GROUNDING=1       # activar
```

Si lo activás, vale la pena setear también:

```bash
export RAG_NLI_IDLE_TTL=900      # TTL idle-unload del modelo NLI
export RAG_NLI_SKIP_INTENTS=count,list,recent,agenda
export RAG_NLI_CONTRADICTS_THRESHOLD=0.7
export RAG_NLI_MAX_CLAIMS=20
```

### `RAG_EXPAND_MIN_TOKENS`
Queries más cortas que este número de tokens saltean la paráfrasis. Default: `4`.

```bash
export RAG_EXPAND_MIN_TOKENS=2   # volver al comportamiento viejo
```

### `RAG_CITATION_REPAIR_MAX_BAD`
Si la respuesta tiene más de N paths inventados, saltea el citation-repair (que tarda ~5-8s). Default: `3`.

```bash
export RAG_CITATION_REPAIR_MAX_BAD=0   # desactivar el repair completo
```

### `RAG_POSTPROCESS_MODEL`
Modelo a usar para citation-repair + critique. Default: `helper` (qwen2.5:3b).

```bash
export RAG_POSTPROCESS_MODEL=chat      # usa command-r (más lento, más preciso)
export RAG_POSTPROCESS_MODEL=legacy    # alias de "chat"
```

### `RAG_EXPLORE=1`
Activa ε-exploration en retrieve: 10% de chance de meter un resultado de menor rank entre los top-3. Sirve para generar counterfactuals.

```bash
export RAG_EXPLORE=1
```

Las plists de `morning`/`today` ya lo tienen. **No lo actives durante `rag eval`** — el comando lo apaga automáticamente.

### `RAG_TRACK_OPENS=1`
Cambia el scheme de los links OSC 8 de `file://` a `x-rag-open://` para que los clicks pasen por `rag open` y registren el evento.

```bash
export RAG_TRACK_OPENS=1
```

Requiere registrar `rag open` como handler de `x-rag-open://` en macOS.

### `RAG_STATE_SQL=1`
Históricamente activaba el SQL store para telemetría. **No-op desde 2026-04-20** — SQL es el único path.

### `RAG_LLM_INTENT_SHADOW=1`
Opt-in shadow mode del LLM intent classifier (Feature #3) — **sin cambiar routing**. Cuando está ON, cada query dispara la clasificación LLM además del regex, y loguea ambas predicciones a `rag_queries.extra_json.intent_shadow = {llm, regex, agree, latency_ms, llm_timed_out}`. El routing sigue usando el regex; shadow es puro measurement para evaluar si vale la pena cablear el router LLM full de Opción C.

**Uso típico** (pre-decisión de Opción C):

```bash
export RAG_LLM_INTENT_SHADOW=1
# correr el uso normal por ~1 semana → juntar N~500 queries
sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db \
  "SELECT json_extract(extra_json, '\$.intent_shadow.agree'), COUNT(*) \
   FROM rag_queries WHERE extra_json LIKE '%intent_shadow%' GROUP BY 1"
# si agree >95%: LLM no agrega valor, skip C
# si agree 80-95%: analizar disagreements por bucket, C probablemente útil
```

**Costo**: +150-500ms por query (qwen2.5:3b helper, primera vez). El `_intent_llm_cache` LRU-500 hace que queries repetidas sean gratis. Solo se ejecuta en callers de `run_chat_turn` + `rag query` CLI — no aplica a serve.chat/tasks/weather/metachat (esos no pasan por el classifier).

**Rollback**: apagar la var (`unset RAG_LLM_INTENT_SHADOW`). Sin ella el código es no-op.

---

## Semantic response cache (GC#1, post-2026-04-23)

El cache guarda respuestas completas indexadas por embedding de la query + `corpus_hash`. Queries repetidas (o near-paraphrases) sobre un vault sin cambios recientes devuelven en <100ms en vez de 5-20s. Las siguientes vars tunean su comportamiento — por default está prendido con umbrales conservadores.

### `RAG_CACHE_ENABLED=1`
Master switch. `0`/`false`/`no` desactiva lookup + store. Por default prendido.

```bash
export RAG_CACHE_ENABLED=0    # apagar el cache entero
```

### `RAG_CACHE_COSINE`
Umbral de cosine similarity sobre embeddings bge-m3 para considerar un hit. Default **0.93** — paraphrases del mismo concepto ("qué es ikigai" vs "qué es el ikigai") caen en 0.93-0.96. Más estricto (0.97) = menos hits pero menos riesgo de servir una respuesta de un concepto distinto; más permisivo (0.85) = más hits pero potencial falsos positivos.

```bash
export RAG_CACHE_COSINE=0.95   # más conservador
```

### `RAG_CACHE_TTL_DEFAULT` / `RAG_CACHE_TTL_RECENT`
TTL en segundos. `DEFAULT` (86400 = 24h) aplica a intents `semantic`/`synthesis`/`count`/`list`/`comparison`; `RECENT` (600 = 10 min) aplica a intents `recent`/`agenda` (queries tipo "qué pasó esta semana" expiran rápido porque el corpus cambia).

```bash
export RAG_CACHE_TTL_DEFAULT=3600    # 1h para queries "estables"
export RAG_CACHE_TTL_RECENT=120      # 2 min para queries temporales
```

### `RAG_CACHE_MAX_ROWS`
Máximo de filas a escanear por lookup. Default 2000 (lineal — cheap hasta unas miles de filas). El cleanup manual es via `rag cache clear`.

---

## Paths / folders

### `OBSIDIAN_RAG_MOZE_FOLDER`
Dónde cae la ingesta MOZE dentro del vault. Default: `02-Areas/Personal/Finanzas/MOZE`.

```bash
export OBSIDIAN_RAG_MOZE_FOLDER="02-Areas/Finanzas/MOZE"
```

### `OBSIDIAN_RAG_MOZE_DIR`
Carpeta iCloud donde el app Tally4 guarda los backups `MOZE_*.zip`. Default:
`~/Library/Mobile Documents/iCloud~amoos~Tally4/Documents` (separado de
`Finances/` desde el 2026-05-04).

Tally4 4.x no exporta CSV — el zip contiene un `moze.realm` (Realm
Object Database). El bridge `rag/integrations/tally4_realm.py` corre un
script Node.js (`scripts/tally4_realm_to_csv.js`) la primera vez que se
necesita, instala `realm` npm de forma lazy en
`~/.local/share/obsidian-rag/realm-extractor/`, y deja el CSV
regenerado en `~/.local/share/obsidian-rag/moze_cache/`. Si node/npm no
están instalados, falla en silencio y el resto del pipeline cae a
buscar CSVs históricos.

```bash
export OBSIDIAN_RAG_MOZE_DIR="~/Library/Mobile Documents/iCloud~amoos~Tally4/Documents"
```

### `OBSIDIAN_RAG_FINANCE_DIR`
Carpeta iCloud donde el banco deja los `Último resumen*.xlsx` (tarjetas)
y los PDFs de transferencias. Default:
`~/Library/Mobile Documents/com~apple~CloudDocs/Finances`.

```bash
export OBSIDIAN_RAG_FINANCE_DIR="~/Library/Mobile Documents/com~apple~CloudDocs/Finances"
```

### `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS`
Carpetas del vault que `rag watch` ignora. Coma-separadas.

```bash
export OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS="03-Resources/WhatsApp,04-Archive"
```

---

## Debug / desarrollo

Estos no los querés en producción.

### `RAG_DEBUG=1`
Verbose stderr en el path de embed local.

### `RAG_RETRIEVE_TIMING=1`
Imprime breakdown por etapa al final de cada `retrieve()`.

### `RAG_NO_WARMUP=1`
Saltea el warmup de reranker + bge-m3 + corpus. Útil para comandos livianos (`rag stats`, `rag session list`).

### `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1`
No genera el context summary del documento (usado en el prefix de embeddings). Lo usan los tests.

### `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q=1`
No genera synthetic questions. Mismo uso que el anterior.

### `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
El reranker se carga del caché local de HuggingFace, sin tocar la red.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### `FASTEMBED_CACHE_PATH=$HOME/.cache/fastembed`
Pinea el cache de [fastembed](https://github.com/qdrant/fastembed) (usado por
`mem0` para BM25 sparse / hybrid search en Qdrant) a un directorio persistente.
El default upstream es `tempfile.gettempdir()/fastembed_cache` que en macOS cae
en `/var/folders/.../T/fastembed_cache` y el SO lo limpia cada tanto. Combinado
con `HF_HUB_OFFLINE=1`, una limpieza del tmpdir deja al encoder sin poder
descargar el modelo de nuevo y `mem0` cae a búsqueda sólo-semántica:

```text
ERROR fastembed.common.model_management:retrieve_model_gcs:362 - Could not
find the model tar.gz file at /var/folders/.../T/fastembed_cache/bm25 and
local_files_only=True.
Failed to load BM25 encoder: Could not load model Qdrant/bm25 from any source.
```

Población inicial (corré una vez con offline mode apagado):

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python -c 'from fastembed import SparseTextEmbedding; SparseTextEmbedding("Qdrant/bm25")'
```

`rag/__init__.py` ya hace `setdefault` con `~/.cache/fastembed` y los plists
(`com.fer.obsidian-rag-web`, `com.fer.obsidian-rag-serve`) lo exportan
explícitamente para evitar la race con el módulo init.

---

## Resumen: qué setear según tu caso

### Primera vez que instalás
Nada. Los defaults andan.

### Tu Mac tiene poca RAM (<32 GB)
```bash
export OLLAMA_KEEP_ALIVE=20m
export RAG_MEMORY_PRESSURE_THRESHOLD=75
```

### Tu Mac tiene mucha RAM (>64 GB)
```bash
export RAG_RERANKER_NEVER_UNLOAD=1
```

### Querés máxima velocidad de query
```bash
export RAG_LOCAL_EMBED=1
export RAG_ADAPTIVE_ROUTING=1     # ya es el default
```

### Tu vault no está en iCloud Notes
```bash
rag vault add personal ~/ruta/al/vault
rag vault use personal
# (o export OBSIDIAN_RAG_VAULT=~/ruta/al/vault para override temporal)
```

### Corriendo en Linux o sin Full Disk Access
```bash
export OBSIDIAN_RAG_NO_APPLE=1
```

### Haciendo tests
```bash
export RAG_MEMORY_PRESSURE_DISABLE=1
export RAG_NO_WARMUP=1
```
