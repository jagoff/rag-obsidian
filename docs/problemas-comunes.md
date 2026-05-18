# Problemas comunes

Qué hacer cuando algo se rompe. Ordenados por "qué me está pasando", no por causa técnica.

---

## El sistema está lento o tarda mucho

### Una query tarda 60s+ (primera pregunta del día)

**Causa**: los modelos de Ollama están cold (no cargados en memoria). La primera carga es lenta.

**Fix**:
```bash
ollama ps                           # ver qué está cargado
# si está vacío, precalentalo:
echo "hola" | ollama run qwen2.5:7b --keep-alive -1
```

Si el problema persiste, chequeá que `OLLAMA_KEEP_ALIVE=-1` (default):

```bash
echo $OLLAMA_KEEP_ALIVE
```

### Todas las queries tardan 3× lo normal

**Causa posible**: el reranker cayó a CPU en vez de usar la GPU (Apple Silicon MPS).

**Fix**: verificar que `sentence-transformers` esté instalado con soporte MPS:

```bash
.venv/bin/python -c "from sentence_transformers import CrossEncoder; import torch; print(torch.backends.mps.is_available())"
# debe imprimir True
```

Si imprime False, reinstalar torch:

```bash
uv pip install --upgrade torch sentence-transformers
```

### El Mac se queda sin memoria / hay beachballs

**Causa**: algún modelo grande está pineado en RAM y bajo presión el sistema swappa.

**Fix rápido**: bajar el keep-alive a algo más humilde y dejar que Ollama libere memoria:

```bash
export OLLAMA_KEEP_ALIVE=20m
```

Si el watchdog de memoria no está funcionando, lo podés apagar/ajustar:

```bash
export RAG_MEMORY_PRESSURE_THRESHOLD=75    # más agresivo (default 85)
```

Ver más en [variables-entorno.md](./variables-entorno.md#oLLAMA_KEEP_ALIVE).

---

## Las respuestas son malas o alucinadas

### El LLM inventa paths que no existen

**Causa**: el citation-repair no alcanzó a corregir todo. Puede pasar si la respuesta tiene muchas citas inventadas.

**Fix**: hacer la pregunta con `--force` desactivado (para que respete la confidence gate) y revisar con `--raw`:

```bash
rag query "tu pregunta" --raw                   # ver los chunks crudos
rag query "tu pregunta" --critique              # que el LLM se autorevise
```

### Dice "no encontré nada sobre X" aunque sé que tengo notas

**Causa posible**: la confianza del reranker quedó baja (< 0.015) y el sistema se negó a responder.

**Fix**: forzá que igual llame al LLM:

```bash
rag query "tu pregunta" --force
rag query "tu pregunta" --force --loose         # permite más prosa del LLM
```

Si el problema es recurrente con un tema, tenés un "gap":

```bash
rag gaps                    # ver temas con muchas queries low-confidence
```

### El reranker no está trayendo las notas correctas

**Fix A**: decile al sistema cuál era la correcta:

```bash
rag fix 02-Areas/X.md       # marca X.md como positive para la última query
```

Con el tiempo, el ranker tuneado las va a traer mejor.

**Fix B**: correr el tune offline:

```bash
rag tune                     # dry-run, ver si cambió algo
rag tune --apply             # persistir el winner
```

**Fix C**: agregar la query al golden set (`queries.yaml`) con las notas esperadas, así entra al eval.

---

## La indexación no está andando

### `rag watch` no re-indexa cuando guardo

**Fix 1**: ver si el servicio está corriendo:

```bash
launchctl list | grep obsidian-rag-watch
```

**Fix 2**: ver si crasheó:

```bash
tail -n 50 ~/.local/share/obsidian-rag/watch.error.log
```

**Fix 3**: reiniciar el servicio:

```bash
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-watch
```

**Fix 4**: revisar que el folder no esté excluido. Algunos folders tipo `99-obsidian/99-AI/external-ingest/WhatsApp` se saltean porque generan ruido. La lista está en la env var:

```bash
echo $OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS
```

### `rag index --full` tarda horas o sube la memoria

**Esperable hasta cierto punto**. Un full reconstruye embeddings para todos los chunks del vault. En un vault de ~5000 chunks puede tardar 10-30 minutos (según el modelo y tu Mac).

Desde 2026-05-16, todo `rag index` activa safe mode por default: shardea notas grandes, saltea entities, chequea memory pressure antes y durante la corrida, y aborta limpio si el proceso crece demasiado. Desde 2026-05-17, safe mode también saltea enriquecimientos LLM opcionales (synthetic questions, context summary, contextual retrieval y contradictions) para que el index sea parse + embed por default.

El index mantiene además una cache persistente exact-match de embeddings (`RAG_INDEX_EMBED_CACHE=1`). Un primer full llena esa cache; los full posteriores pueden reconstruir la base y las URLs reutilizando vectores idénticos por hash de texto + modelo/backend. El batching MLX queda en `auto` por default, así que en Macs de ~36 GB usa batches más grandes (`batch=96`, `slice=48`, `local_batch=48`) sin cambiar el contenido indexado.

También mide y acota los pre-syncs cross-source antes del scan del vault. El output muestra `Cross-source timing` y el `Timing index` separa `etl`, `embed`, `write` y `urls`. Por default los ETLs externos corren en paralelo, Reminders tiene timeout corto durante safe index, y los ETLs que ya escribieron snapshots recientes se saltean por `RAG_INDEX_ETL_FRESH_TTL_S=600`. Para medir una pasada fresca completa sin cache de ETL:

```bash
RAG_INDEX_ETL_FRESH_TTL_S=0 rag index --full
```

Para reactivar esos enriquecimientos explícitamente:

```bash
RAG_INDEX_LLM_ENRICHMENTS=1 RAG_CONTEXT_SUMMARY=1 rag index --full --contextual
```

Para desactivar las guardas base existe `RAG_INDEX_SAFE=0`, pero no es recomendable en una Mac que ya se colgó por presión de memoria.

Si querés recuperar entidades después de un full seguro, corré el backfill aislado:

```bash
python scripts/backfill_entities.py
```

### `rag start --full` sube memoria o deja la Mac pesada

`--full` en `rag start` significa "levantar todo el spec de daemons", no "hacer `rag index --full`". Aun así, desde 2026-05-16 `rag start` tiene su propia safe mode: el catch-up incremental fuerza `RAG_INDEX_SAFE=1`, corre sin checks de contradicción y sin context/synthetic/entities durante bootstrap, chequea memory pressure antes de seguir y escalona el bootstrap de servicios.

Si la presión sigue alta, `rag start` prefiere saltear ese tramo y avisar antes que cargar más procesos. El opt-out existe para debug:

```bash
RAG_START_SAFE=0 rag start --full
```

No usar ese opt-out en la Mac que ya se colgó por swap/memoria salvo para aislar un bug con monitoreo manual.

### Hice `rag index` pero `rag query` dice "vault vacío"

**Fix**: chequear que estés indexando el vault correcto:

```bash
rag vault current                # muestra el vault activo
rag stats                        # muestra cuántos chunks tiene el índice
```

Si indexaste un vault y estás queriando otro, cambiá:

```bash
rag vault use <nombre>
```

---

## Problemas con Ollama

### `rag query` tira error "connection refused"

**Causa**: Ollama no está corriendo.

**Fix**:

```bash
# Arrancar Ollama (si lo instalaste como app)
open -a Ollama

# Si lo instalaste con Homebrew como servicio:
brew services start ollama

# O manualmente:
ollama serve &
```

### Falta un modelo

Error típico: `model "qwen2.5:7b" not found`.

**Fix**:

```bash
ollama pull qwen2.5:7b
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:3b
```

### `rag stats` dice "modelo chat: command-r" pero quiero qwen2.5:7b

El resolver mira `command-r:latest` → `qwen2.5:14b` → `phi4:latest` en ese orden. Si tenés command-r instalado, gana por default.

**Fix**: remové command-r o cambiá el orden en `resolve_chat_model()` en `rag.py`.

O instalá qwen2.5:7b y dejá que command-r no sea la primera opción (ver código).

---

## Problemas con las sesiones de chat

### `--resume` no me trae la última sesión

**Causa**: el pointer `last_session` quedó apuntando a una sesión borrada.

**Fix**: listar sesiones vivas y retomar explícitamente:

```bash
rag session list
rag chat --session <id-que-salió-en-list>
```

### Quiero borrar una sesión específica

```bash
rag session list                 # ver los ids
rag session clear <id>
rag session clear <id> --yes     # sin prompt
```

### Tengo muchas sesiones viejas

```bash
rag session cleanup              # default: purga >30 días
rag session cleanup --days 7     # más agresivo
```

---

## Problemas con launchd / servicios

### Después de reinstalar, los servicios no arrancan

**Fix**: re-correr el setup, es idempotente:

```bash
rag setup
```

### Un servicio está loopeando (se relanza sin parar)

**Causa**: el comando está crasheando al arrancar, y KeepAlive lo relanza.

**Fix**:

1. Ver el error:
   ```bash
   tail -n 100 ~/.local/share/obsidian-rag/<nombre>.error.log
   ```
2. Correr el comando a mano para reproducir:
   ```bash
   rag <nombre>
   ```
3. Arreglar la causa (modelo faltante, permisos, env var).
4. Recargar:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   launchctl load ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   ```

### Quiero parar un servicio temporalmente

```bash
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
# Para volver a activarlo:
launchctl load ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
```

---

## Problemas con el vault

### `rag dead` devuelve 0 candidatos aunque tengo notas viejas

**Causa**: iCloud bumpea los `mtime` de las notas constantemente por sync, así que el criterio "vieja por mtime" no discrimina.

**Fix**: `rag dead` usa `frontmatter.created:` cuando existe. Asegurate de que tus notas tengan ese campo. Obsidian tiene plugins que lo escriben automáticamente al crear una nota.

### `rag links` devuelve solo imágenes

**Causa**: el filtro de media quedó desactualizado.

**Fix**:

```bash
rag links --rebuild
```

### `rag wikilinks suggest --apply` rompió un archivo

Muy improbable (las regexs son conservadoras), pero por las dudas:

- Si el vault está versionado con git: `git diff tu-vault/` y revertí.
- Si no, los `[[wraps]]` son no destructivos — los podés editar manualmente en Obsidian.

---

## Problemas con el eval / tune

### Los baselines de `rag eval` cayeron de repente

**Causas posibles**:
- Bumpeaste el schema de la collection (v10 → v11) → corré `rag index --reset`.
- Cambió el pipeline (flag nuevo, adaptive routing, etc.).
- Cambiaste de vault (el `queries.yaml` está calibrado contra un vault específico).

**Fix para aislar**:

```bash
rag eval --no-multi                 # aislar efecto de multi-query
rag eval --hyde                     # probar con HyDE
rag eval --latency                  # ver si es un problema de latencia
```

### `rag eval` se está yendo a OpenAI o Anthropic

**No debería** — el sistema es 100% local. Si esto pasa es un bug.

Ver reglas en `.devin/config.json` — las Fetch a `openai.com` / `anthropic.com` están en la lista de ask, deberían pedir confirmación.

---

## Errores silenciosos

Hay un helper `_silent_log()` que captura excepciones en subsistemas secundarios (context cache corrupto, embed feedback offline, etc). Para verlos:

```bash
rag log --silent-errors              # últimas 20 excepciones silenciosas
rag log --silent-errors --summary    # agrupadas por (where × exc_type)
rag log --silent-errors -n 100       # más de 100
```

Si algo se loguea ahí repetidamente, abrí el archivo que indica `where` y arreglalo.

---

## Qué hacer si nada de esto funcionó

1. Mirá el log del comando que falló:
   ```bash
   tail -n 200 ~/.local/share/obsidian-rag/<nombre>.error.log
   ```
2. Corré con `RAG_DEBUG=1`:
   ```bash
   RAG_DEBUG=1 rag query "tu pregunta"
   ```
3. Abrí el breakdown de retrieve:
   ```bash
   RAG_RETRIEVE_TIMING=1 rag query "tu pregunta"
   ```
4. Buscá tu síntoma en:
   - [`../README.md`](../README.md#troubleshooting) — tabla de troubleshooting más técnica
   - [`../CLAUDE.md`](../CLAUDE.md) — si vas a tocar código
   - Los logs: `~/.local/share/obsidian-rag/*.log`
5. Si todo falla: commiteá lo que tenés, hacé un `git reset --hard` al último commit que andaba, y abrí un issue.
