# Migración Ollama → MLX

> Doc técnica de referencia. PM doc + estado vivo en el vault: [`99-AI/system/mlx-migration/dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch).
> Código de la abstracción: [`rag/llm_backend.py`](../rag/llm_backend.py).
> Iniciado: **2026-05-05**. Estado: **Ola 2 (en curso) — `MLXBackend` funcional, 4 modelos smoke-testeados OK, streaming implementado, dispatch de raw call sites en progreso**.

## Resumen ejecutivo

`obsidian-rag` está migrando los 4 LLMs locales de [Ollama](https://ollama.com) a [Apple MLX](https://github.com/ml-explore/mlx) vía [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Es un **reemplazo total**: post-cutover (Ola 5) el repo deja de depender del Python client de Ollama. La motivación es performance en Apple Silicon (kernels nativos Metal, sin pasar por la API HTTP de Ollama) + acceso directo a la familia [Qwen3](https://huggingface.co/collections/Qwen/qwen3) en formatos cuantizados que Ollama todavía no empaqueta oficialmente.

**Embeddings (`bge-m3`) NO entran en este scope.** Esa migración corre por separado en [`99-AI/system/embedding-swap-qwen3-8b/`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

### Gates de éxito

- **Ola 1 (cerrada)**: scaffold mergeable a master en compatibility mode. Backend abstraction + extra opcional `[mlx]` + docs base. ✅
- **Ola 2 (en curso)**: `MLXBackend.chat()` + `chat_stream()` funcionales. 4 modelos smoke-testeados OK. ~29 call sites que pasan por `_TimedOllamaProxy.chat()` ya ruteados via `_mlx_chat_via_backend()`. 14 raw `ollama.chat(stream=True, ...)` sites siendo patcheados. Ver sección [Dispatch Ola 2](#dispatch-ola-2) abajo.
- **Ola 3**: `keep_alive` emulado con resident-process + LRU. Plists actualizados. Métricas MLX expuestas.
- **Ola 4**: `rag eval` con bootstrap CIs **no-overlapping bajo floor** = ROLLBACK automático. Floor actual (2026-04-27, post-golden-remap): singles `hit@5 53.70% [40.74, 66.67] · MRR 0.528 [0.407, 0.657]`, chains `hit@5 72.00% [52.00, 88.00] · chain_success 33.33% [11.11, 66.67]`. Determinismo: dos corridas consecutivas → números idénticos. Backends Ollama y MLX deben compararse con CIs; si MLX cae bajo floor de Ollama con CIs no-overlapping → ROLLBACK.
- **Ola 5 (cutover)**: `import ollama` desaparece del repo (`rag/`, `tests/`). `ollama>=0.6.1` sale de [`pyproject.toml`](../pyproject.toml). `OllamaBackend` se retira (o se conserva en `legacy/` por insurance de rollback histórico).

## Arquitectura del backend

Toda la abstracción vive en [`rag/llm_backend.py`](../rag/llm_backend.py) (~555 LOC). En Ola 2 `MLXBackend` está funcional (no-stub) y los ~29 call sites que pasan por `_TimedOllamaProxy` ya rutean a MLX cuando `RAG_LLM_BACKEND=mlx`. El backend default sigue siendo `ollama`.

### Diagrama

```
                    rag/__init__.py (60k LOC)
                          │
                          │ get_backend().chat(model=..., messages=...)
                          ▼
              ┌──────────────────────────────┐
              │   get_backend()  [singleton] │
              │   env: RAG_LLM_BACKEND       │
              └──────────────┬───────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  OllamaBackend   │      │   MLXBackend     │
        │  (default hoy)   │      │  (Ola 2 ✅)      │
        ├──────────────────┤      ├──────────────────┤
        │ ollama.chat      │      │ mlx_lm.generate  │
        │ ollama.generate  │      │ + chat template  │
        │ keep_alive=-1    │      │ resident + LRU   │
        │ to_ollama(model) │      │ to_mlx(model)    │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
       Ollama daemon (HTTP)         mlx-lm in-process
       :11434                       (Apple Silicon only)
```

### Interfaz `LLMBackend`

ABC con tres métodos:

```python
class LLMBackend(ABC):
    name: str

    def chat(self, model, messages, options=None,
             keep_alive=-1, format=None, **kwargs) -> dict: ...

    def generate(self, model, prompt, options=None,
                 keep_alive=-1, **kwargs) -> dict: ...

    def list_available(self) -> list[str]: ...
```

El shape del return imita el dict de Ollama (`{"message": {"content": "..."}, ...}` para `chat`, `{"response": "..."}` para `generate`) — así los call sites no cambian al swappear backend.

### `ChatOptions`

Dataclass que mirrorea el `options` dict de Ollama. Defaults coinciden con `HELPER_OPTIONS`:

```python
@dataclass
class ChatOptions:
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 768
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
```

Call sites que necesitan chat-tier sampling pasan overrides (en general `CHAT_OPTIONS = {num_ctx: 4096, num_predict: 768}` — los mismos defaults; lo que cambia para chat es que el modelo es de mayor capacidad, no las options).

### `OllamaBackend` (legacy passthrough)

Wrappea el client `ollama`. Identity passthrough: traduce `ChatOptions` → `options` dict, llama a `ollama.chat` / `ollama.generate` / `ollama.list`, y resuelve nombres MLX-style → Ollama-style con `to_ollama()` (por si un call site ya migró a HF ID antes de Ola 5).

### `MLXBackend` (funcional desde Ola 2)

`chat()`, `chat_stream()` y `generate()` están implementados. El `__init__` valida que `mlx_lm` esté importable; si no está → `RuntimeError("mlx-lm not installed. Run \`uv pip install '.[mlx]'\` or set RAG_LLM_BACKEND=ollama.")`. `list_available()` scanea `~/.cache/huggingface/hub/` por carpetas `models--mlx-community--*`.

Lo que Ola 2 cableó:

- `mlx_lm.load(model_id)` con caché de modelos cargados (`self._loaded`, `OrderedDict` LRU).
- Chat template via `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`.
- `mlx_lm.generate(model, tokenizer, prompt, max_tokens=num_predict, sampler=make_sampler(temp, top_p), ...)`.
- LRU eviction: modelos `_BIG_MODELS` (Qwen3-30B-A3B) son single-tenant (evictan todo); pequeños LRU-cap a `_MAX_SMALL_LOADED=3`.
- JSON mode: `_apply_chat_template` agrega instrucción de sistema; `_extract_json` hace strip de fences y aísla el primer bloque `{...}`. Sin grammar-constrained decode nativo (MLX no lo tiene aún) — el parser en `rag/__init__.py` hace el repair final.
- `chat_stream()`: `mlx_lm.stream_generate` generator, retorna objetos `ChatResponse(done=False)` con `.message.content` incremental + terminal `done=True, done_reason='stop'`. Mismo `_apply_chat_template` que `chat()`.

Lo que Ola 2 **no** implementa (sin soporte bajo MLX, ver sección abajo):

- `tools=[...]` — no hay tool-calling nativo MLX. Las llamadas con `tools=` bypasean al path Ollama o se stripean.
- `ollama.generate(prompt='', keep_alive=0)` — trick de unload de Ollama. En MLX es no-op silencioso.

### Dispatch Ola 2: `_TimedOllamaProxy` + raw call sites

El mecanismo de dispatch tiene dos capas:

**Capa 1 — `_TimedOllamaProxy.chat()`** (línea ~3848 de `rag/__init__.py`): cuando `RAG_LLM_BACKEND=mlx`, cada `proxy.chat(...)` detecta el env var y llama a `_mlx_chat_via_backend(**kwargs)` en lugar de `ollama.Client.chat()`. Esta capa cubre todos los call sites que usan `_helper_client`, `_chat_client`, `_summary_client`, etc. — ~29 sitios en total.

`_mlx_chat_via_backend()` traduce el `options` dict Ollama → `ChatOptions`, droppea kwargs Ollama-only (`tools`, `think`, `logprobs`, `stream=True` — collapsa a no-streaming en esta capa).

**Capa 2 — raw `ollama.chat(stream=True, ...)`**: 14 call sites que llaman `ollama.chat` directamente (no via proxy), usados para SSE streaming en `rag query` y `rag chat`. Estos se están migrando en la sesión actual para usar `get_backend().chat_stream(...)` con fallback al path Ollama cuando el backend es `ollama`.

### `get_backend()` singleton

```python
choice = os.environ.get("RAG_LLM_BACKEND", "ollama").lower()
```

Singleton process-wide. Reset via `reset_backend()` (sólo tests). Valores válidos: `ollama` | `mlx`. Cualquier otra cosa → `ValueError`.

## Mapping de modelos

| MLX HF ID | Tamaño | Reemplaza Ollama | Tier | Smoke test | Use-cases |
|---|---|---|---|---|---|
| [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit) | ~1.9 GB | `qwen2.5:3b` | **HELPER** (det.) | ✅ 2026-05-05 | `_reformulate_query`, contextual summary, lookup, postprocess, followup judge, tag suggester, typo correction, history summarisation, datetime fallback. `temp=0, seed=42`. |
| [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | ~4.3 GB | `qwen2.5:7b` | **CHAT default** | ✅ 2026-05-05 | `rag query`, narrative brief fallback, read summary, prep brief, inbox triage, whisper LLM correct. |
| [`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) | ~17 GB | `command-r:latest`, `qwen2.5:14b` | **HQ tier** | ✅ 2026-05-05 (chat + JSON mode) | Contradiction detector, `_render_morning_structured_prompt` (brief JSON), `rag do` tool-loop, **re-test HyDE**. |
| [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) | ~2.5 GB | — | Experimental | ✅ 2026-05-05 | A/B vs el 3B helper. **NO default** hasta CIs no-overlapping arriba del floor en `rag eval`. |

### Justificación

- **Helper queda en familia Qwen 3B**. Saltar a un modelo más grande en el helper-tier ya falló: la memoria [`project_reformulate_helper_vs_chat`](mem-vault) registra que poner `command-r` en `_reformulate_query` regresó chains −11pp + 5× latencia el 2026-04-17. Mantenerlo en Qwen 3B (mismo training family, sólo runtime distinto) es bajo-riesgo.
- **Chat default Qwen2.5-7B** mantiene paridad con la versión Ollama (`qwen2.5:7b` ya estaba en uso como fallback de `command-r`). Cambia el runtime, no el modelo.
- **HQ tier sube a Qwen3-30B-A3B-2507**. Reemplaza `command-r:latest` (35B dense, comparable VRAM en Ollama Q4) y `qwen2.5:14b`. Justificación:
  - **MoE A3B = 3B parámetros activos por forward pass** sobre 30B totales → throughput cercano a un 3B denso una vez cargado, con la calidad de un modelo grande. Ideal para tasks chat-tier que no se pueden delegar al helper (contradiction detector, brief JSON, `rag do`).
  - El invariante `contradiction detector MUST use chat-tier` (memoria: qwen2.5:3b probó non-determinístico + JSON malformado en esa task) sigue intacto: ahora "chat-tier" significa Qwen3-30B-A3B en lugar de command-r.
  - **Re-test HyDE gratis**: el invariante "HyDE OFF por default" se midió contra qwen2.5:3b (drop singles −5pp). Con un modelo MoE 30B vale la pena re-medir; puede flippear de "hurts" a "useful" en singles.
  - Tool-calling: command-r usa un format propio de tool calls (XML-ish), Qwen3 usa el format estándar con `<tool_call>...</tool_call>` JSON inline. **Es un parser nuevo, no una migración 1:1** — coordinado en Ola 2 (`rag-llm` slot D).
- **Qwen3-4B-2507 queda experimental**. Está bajado pero NO mapeado como default a ningún tier. Sirve para A/B vs el 3B helper: si `rag eval` con Qwen3-4B en lugar de Qwen2.5-3B muestra **CIs no-overlapping arriba del floor** en singles+chains, se promueve. Hasta entonces, sólo se invoca con override manual (alias `qwen3:4b` → `mlx-community/Qwen3-4B-Instruct-2507-4bit`).

## Aliasing de nombres

Para no tocar 28 call sites en Ola 1, `LLMBackend` acepta tanto el nombre Ollama (`qwen2.5:3b`) como el HF ID MLX (`mlx-community/Qwen2.5-3B-Instruct-4bit`). Las funciones `to_mlx()` y `to_ollama()` resuelven en cualquier dirección.

### Tabla `MLX_MODEL_ALIAS`

Definida en [`rag/llm_backend.py`](../rag/llm_backend.py):

```python
MLX_MODEL_ALIAS: dict[str, str] = {
    # Helper tier
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    # Chat default
    "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # HQ tier (contradicciones, brief JSON, rag do, HyDE re-test)
    "command-r:latest": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "command-r": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen2.5:14b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    # Experimental (no default)
    "qwen3:4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
}
OLLAMA_MODEL_ALIAS = {v: k for k, v in MLX_MODEL_ALIAS.items()}
```

Notar que tres claves Ollama (`command-r:latest`, `command-r`, `qwen2.5:14b`) mapean al mismo HF ID. La inversión es lossy: `OLLAMA_MODEL_ALIAS["mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"]` → `"qwen2.5:14b"` (last-write-wins en el dict comprehension). Para call sites que pasen el HF ID a `OllamaBackend` y necesiten el nombre Ollama exacto, eso puede ser un gotcha — pero como la migración va en una sola dirección (Ollama → MLX, no al revés), no debería llegar a importar.

### `to_mlx()` / `to_ollama()`

```python
def to_mlx(model: str) -> str:
    if model.startswith("mlx-community/"):
        return model  # ya es MLX HF ID
    return MLX_MODEL_ALIAS.get(model, model)  # passthrough si no está en la tabla

def to_ollama(model: str) -> str:
    return OLLAMA_MODEL_ALIAS.get(model, model)  # passthrough si no está en la tabla
```

**Modelos que no estén en la tabla → fall through (passthrough)**. Esto es importante: si alguien llama `get_backend().chat(model="phi4", ...)` y el backend es MLX, el HF ID resultante es `"phi4"` (literal) y `mlx_lm.load()` va a fallar. **Es by design**: forzar al call site a usar nombres canónicos. `phi4` ya no está instalado de todos modos (verificado 2026-04-17).

## Invariantes preservadas

Todo lo que estaba garantizado con Ollama tiene que seguir garantizado con MLX. Lista cerrada:

1. **HELPER_OPTIONS = `{temperature: 0, seed: 42}`** — eval reproducibility floor. Cualquier helper LLM que rompa determinismo invalida `rag eval`.
2. **CHAT_OPTIONS = `{num_ctx: 4096, num_predict: 768}`** — VRAM-budgeted. No subir sin re-medir headroom.
3. **`keep_alive=-1` semantics** → MLX no tiene `keep_alive` nativo (no hay HTTP daemon afuera). Emulado con **resident-process + LRU eviction** dentro de `MLXBackend`. Política de eviction:
   - Modelos en `_BIG_MODELS` (`{Qwen3-30B-A3B}`) **NUNCA coexisten con `qwen2.5:7b`** en Macs con unified RAM <32 GB. Cargar el grande evictea el chico (y vice-versa).
   - Helper Qwen2.5-3B (~1.9 GB) puede coexistir con cualquiera.
4. **Contradiction detector NUNCA usa helper-tier**. Probado non-determinístico + JSON malformado en esa task con qwen2.5:3b. Hoy: Qwen3-30B-A3B.
5. **`reformulate_query` MUST use HELPER**, no chat-tier. Memoria 2026-04-17: command-r en helper regresó chains −11pp + 5× latencia.
6. **HyDE OFF por default**. `--hyde` opt-in. qwen2.5:3b drop singles −5pp. **Re-evaluar con Qwen3-30B-A3B** — puede flippear.
7. **Local-first**. Memoria `feedback_local_free_stack`: jamás OpenAI/Anthropic/Google/ElevenLabs APIs. MLX corre 100% local en Apple Silicon, así que esta invariante refuerza (no rompe).
8. **No caches stale por rename**. Contextual summary cache por **file hash**, no path. (Esto es del runtime, no del backend, pero queda acá porque MLX no debe alterarlo.)

## Streaming (`chat_stream`)

`MLXBackend.chat_stream()` usa `mlx_lm.stream_generate` y retorna un generator de objetos `ChatResponse` con la misma interfaz que `OllamaBackend.chat_stream()`:

```python
for chunk in backend.chat_stream(model="qwen2.5:7b", messages=[...]):
    piece = chunk.message.content   # token incremental (puede ser sub-word)
    if chunk.done:
        break                       # done=True, done_reason='stop', content=''
```

Diferencias vs Ollama streaming:

- **Granularidad de tokens**: MLX puede flushear sub-words o varios tokens por yield dependiendo del tokenizer; Ollama suele flushear un token por SSE event. El texto final es equivalente.
- **Primer token latency**: MLX no tiene daemon HTTP overhead → primer token más rápido en cold (no hay round-trip HTTP), pero el primer yield puede tardar más si el modelo no estaba en `_loaded` (load time ~3-8s para modelos pequeños, ~30-90s para el 30B).
- `stream=True` pasado via `_mlx_chat_via_backend` es stripeado (no hay flag `stream` en MLX). La diferencia es que `_TimedOllamaProxy.chat()` **no** usa `chat_stream()` — colapsa streaming a non-streaming. Los call sites que hacen SSE real (`for chunk in ollama.chat(stream=True, ...)`) necesitan migrar explícitamente a `get_backend().chat_stream(...)` (trabajo de la capa 2 de Ola 2).

## Limitaciones bajo MLX (sin soporte actualmente)

| Feature | Comportamiento bajo MLX |
|---|---|
| `tools=[...]` (tool-calling nativo) | `_mlx_chat_via_backend` droppea el kwarg. Las llamadas que requieren tool-calling JSON schema siguen yendo por Ollama o necesitan un parser custom Qwen3. El `rag do` tool-loop necesita este parser — coordinado en Ola 2. |
| `ollama.generate(prompt='', keep_alive=0)` | Trick de Ollama para descargar un modelo de VRAM. MLX no tiene ese mecanismo. En `MLXBackend.generate()` el `keep_alive=0` es ignorado (no-op silencioso). Para forzar eviction: `backend._loaded.clear()` manual. |
| JSON mode con constrained decode | MLX no tiene grammar-constrained decode nativo (a diferencia del `format="json"` de Ollama que fuerza tokens válidos). El `MLXBackend` usa la heurística de inyectar instrucción en el system prompt + `_extract_json()` post-gen. Funciona para HQ tier (Qwen3-30B es bueno en JSON); puede fallar con el helper 3B en outputs complejos. Si falla: downstream repair en `rag/__init__.py` completa el fix. |
| Multi-process model sharing | Ollama es daemon compartido; cada proceso MLX carga su propia copia. Ver gotcha en sección "Multi-process". |
| `keep_alive="5m"` (TTL custom) | Parseado pero no enforced todavía. Ola 3 implementa el watchdog thread. |

## Cómo flipear al backend MLX

El backend por default sigue siendo `ollama` durante la ventana de migración. Para probar MLX:

### Ad-hoc (shell / desarrollo)

```bash
export RAG_LLM_BACKEND=mlx
rag query "qué notas tengo sobre X" --plain
# o directo sin exportar:
RAG_LLM_BACKEND=mlx rag query "qué notas tengo sobre X" --plain
```

Para verificar que el backend está activo:

```python
from rag.llm_backend import get_backend, reset_backend
reset_backend()  # fuerza re-resolución con el env actual
b = get_backend()
print(b.name)           # → "mlx"
print(b.list_available())  # lista modelos MLX en ~/.cache/huggingface/hub/
```

### Daemons launchd (post-Ola 4)

Agregar al `EnvironmentVariables` dict de cada plist:

```xml
<key>RAG_LLM_BACKEND</key>
<string>mlx</string>
```

Luego recargar:

```bash
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
launchctl load   ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
```

O via `rag setup` si el factory `_services_spec()` ya incluye la variable.

**No hacer esto antes de Ola 4** (eval gate) salvo para testing puntual con `rag stop` previo.

### Tests

```python
def test_algo(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
    from rag.llm_backend import reset_backend
    reset_backend()
    # ... tu test
```

### Smoke test rápido

```bash
PYTHONPATH=. RAG_LLM_BACKEND=mlx .venv/bin/python scripts/smoke_mlx_models.py
# saltear el 30B (~17 GB) si querés correr rápido:
PYTHONPATH=. RAG_LLM_BACKEND=mlx .venv/bin/python scripts/smoke_mlx_models.py --skip-big
# probar un solo modelo:
PYTHONPATH=. RAG_LLM_BACKEND=mlx .venv/bin/python scripts/smoke_mlx_models.py --only qwen2.5:3b
```

## Rollback a Ollama

No requiere ningún cambio de código. Basta con dessetear o poner `ollama`:

```bash
# opción 1: dessetear (default ya es ollama)
unset RAG_LLM_BACKEND

# opción 2: explícito
export RAG_LLM_BACKEND=ollama

# reiniciar daemons si estaban con mlx seteado en plists:
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist
# editar plists: sacar RAG_LLM_BACKEND o ponerlo en "ollama"
launchctl load   ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist
```

Pre-requisito: `ollama serve` corriendo + modelos pulled:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull command-r:latest
```

El floor de eval asegura que rollback siempre es viable mientras `OllamaBackend` esté en el repo (hasta Ola 5).

## VRAM management

Apple Silicon usa **unified memory**: GPU + CPU comparten RAM. Cargar 4 modelos = ~26 GB en peak. En Macs 16 GB → imposible. En 32 GB → viable con LRU. En 64 GB+ → todos pueden quedar resident.

### Estrategia LRU (implementada en Ola 2)

```
slot 0: helper (~1.9 GB)         LRU cap=3 small, puede coexistir con cualquiera
slot 1: chat-default (~4.3 GB)   LRU cap=3 small
slot 2: HQ tier (~17 GB)         BIG_MODELS → single-tenant, evicta todo lo demás
slot 3: experimental (~2.5 GB)   LRU cap=3 small, on-demand
```

`_BIG_MODELS` en `MLXBackend` lleva el set de los modelos "grandes" (hoy sólo Qwen3-30B-A3B). Eviction policy (implementada en `_evict_for()`):

```
on load(M):
    if M in BIG_MODELS:
        evict ALL other models  # pure single-tenant
    else:
        evict any big first
        LRU-trim while len(loaded) >= MAX_SMALL_LOADED (=3)
    self._loaded[M] = mlx_lm.load(M)  # OrderedDict LRU, newest=last
```

### Idle unload

Env var `RAG_MLX_IDLE_TTL` (default `1800` segundos = 30 min). El valor se guarda en `self._idle_ttl` pero el watchdog thread todavía no está implementado (Ola 3). Eviction hoy es puramente capacity-driven. Excepciones planificadas para Ola 3: helper Qwen2.5-3B nunca se descarga por idle (cold-load penalty no vale la pena).

Equivalencia con Ollama:

| Ollama | MLX |
|---|---|
| `keep_alive=-1` (resident hasta que daemon muera) | resident + LRU (capacity-driven, Ola 2). Idle TTL watchdog: Ola 3. |
| `keep_alive=0` (descarga después del request) | no implementado — no-op silencioso |
| `keep_alive="5m"` | TTL custom por request (Ola 3) |

## Plan de cutover (5 olas)

Detalle vivo en [`dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch). Resumen:

| Ola | Scope | Agentes | Worktree | Estado |
|---|---|---|---|---|
| **0** | PM bootstrap (dispatch.md, worktrees, claude-peers announce) | `pm` | — | ✅ |
| **1** | Foundation: `rag/llm_backend.py`, benchmark harness, `pyproject.toml`, docs base | `developer-1`, `developer-2`, `developer-3`, `rag-doc-curator` | `foundation` | ✅ |
| **2** | `MLXBackend` funcional + streaming. ~29 call sites via `_TimedOllamaProxy` ya ruteados. 14 raw `ollama.chat(stream=True)` sites siendo patcheados. Smoke tests 4 modelos OK. | `developer-3` (esta sesión), `rag-llm` | `experimental/mlx-migration-foundation` | **en curso** |
| **3** | Runtime: launchd plists sin deps Ollama, `keep_alive` LRU eviction, métricas MLX (`/api/metrics/mlx`) | `rag-infra` (2 slots), `rag-telemetry` | `mlx-infra` | pending |
| **4** | Tests + eval. `rag eval` con bootstrap CIs vs floor. Determinismo x2 corridas. | `developer-{1,2,3}`, `rag-eval` | `mlx-tests` | pending |
| **5** | Cutover: sacar `import ollama` del repo, sacar `ollama` de `pyproject.toml`, mem-vault `reference_ollama.md` → `reference_mlx.md`. Verificar plists post-flip. | `pm` + 1-2 specialists | master | pending |

**Gate Ola 4 → Ola 5**: si `rag eval` post-MLX muestra CIs **no-overlapping bajo floor** en cualquier métrica clave (singles hit@5, chains hit@5, chain_success), ROLLBACK automático: `RAG_LLM_BACKEND=ollama` queda como default y la migración pausa hasta diagnóstico (probablemente regresión en Q4 quant del Qwen2.5-3B MLX vs el GGUF Q4_K_M de Ollama, o JSON robustness regression en MLX).

## Troubleshooting

### `RuntimeError: mlx-lm not installed`

Setear `RAG_LLM_BACKEND=mlx` sin tener `mlx-lm` instalado. Fix:

```bash
uv tool install --reinstall --editable '.[mlx]'
# o
uv pip install 'mlx-lm>=0.18'
```

`mlx-lm` sólo está en el extra `mlx`, marker `sys_platform == 'darwin' and platform_machine == 'arm64'`. En Linux/Intel el extra no se resuelve y los tests con marker `requires_mlx` se auto-skipean en CI.

### "El backend sigue siendo Ollama"

Ver sección [Cómo flipear al backend MLX](#cómo-flipear-al-backend-mlx) y [Rollback a Ollama](#rollback-a-ollama) arriba — instrucciones completas con comandos concretos.

### Modelo MLX no descargado

Si `MLXBackend.list_available()` no incluye el modelo target, descargarlo con [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli):

```bash
huggingface-cli download mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
huggingface-cli download mlx-community/Qwen2.5-3B-Instruct-4bit
huggingface-cli download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Cache default: `~/.cache/huggingface/hub/`. El 30B-A3B son ~17 GB — la primera descarga tarda. Re-tries son idempotentes (resume).

### CI Linux skipea tests MLX

By design: el extra `mlx` tiene marker Apple-Silicon-only, así que en runners Linux `mlx-lm` no se instala. Los tests marcados `@pytest.mark.requires_mlx` corren localmente (Mac) y se skipean en CI. Para forzar local: `pytest -m requires_mlx`.

## Comparación Ollama vs MLX

| Dimensión | Ollama | MLX |
|---|---|---|
| **Runtime** | Daemon HTTP (`ollama serve` :11434) | In-process (`mlx_lm.load` + `generate`) |
| **Cuantización default** | GGUF Q4_K_M | 4-bit MLX (group-wise) |
| **Keep alive** | Nativo (`keep_alive=-1`) | Emulado (resident + LRU + TTL) |
| **JSON mode** | `format="json"` (constrained decode) | Manual: parser + repair en post (Ola 2 wiring) |
| **Tool-calling** | Format por modelo (command-r XML, Qwen JSON) | Mismo problema, parser nuevo para Qwen3 |
| **Streaming** | SSE nativo del daemon | `mlx_lm.stream_generate` generator |
| **Modelos disponibles** | Catalogo Ollama (curado, lag vs HF) | HuggingFace `mlx-community/*` (acceso directo a Qwen3-30B-A3B) |
| **Multi-process** | Sí (daemon compartido) | No (cada proceso carga su copia) — gotcha si daemon `web` y daemon `morning` corren simultáneo |
| **Latencia esperada** | Baseline | Ver `benchmarks/bench_mlx_vs_ollama.py` (números reales post-Ola 2) |

### Latencia: medición pendiente

[`benchmarks/bench_mlx_vs_ollama.py`](../benchmarks/bench_mlx_vs_ollama.py) define la harness: 7 queries representativas (historial drift, RAG simple, tech ref, vault-specific, entity lookup, chitchat, one-word) corridas vs `/api/chat`, midiendo P50/P95 latency, tokens/s, drift detection (CJK + portugués/italiano hints). Output: tabla markdown comparable. Los números reales se llenan en Ola 4 con el eval gate. **No reportar latencias acá hasta tener corrida real.**

### Multi-process gotcha (importante)

Ollama corre como daemon → varios procesos (CLI `rag`, daemon `web`, daemon `morning`, listener WA) comparten el modelo cargado en VRAM una sola vez. **MLX no tiene daemon**: cada proceso que llama `get_backend()` con `RAG_LLM_BACKEND=mlx` carga su propia copia del modelo en su propio espacio de memoria. Cargar Qwen3-30B-A3B en 3 procesos = 51 GB. **Mitigación Ola 3**: que sólo el daemon `web` (FastAPI) tenga MLX backend, y los demás procesos hagan request HTTP a `/api/chat` en lugar de cargar modelos por su cuenta. CLI `rag query` también debería pegarle al daemon en lugar de cargar MLX local. Detalle en plan Ola 3.

## FAQ / decisiones explícitas

### ¿Por qué reemplazo total y no coexistencia permanente?

Mantener dos backends para siempre es deuda. El código de Ola 1 (`LLMBackend` ABC + dos subclases) está pensado como **insurance de rollback durante la migración**, no como abstracción permanente. Una vez que Ola 4 valide eval CIs, Ola 5 retira `OllamaBackend` y el repo queda con un solo runtime. Si alguna vez aparece un tercer backend (vLLM, llama.cpp directo, etc.) la abstracción se puede revivir.

### ¿Por qué embeddings quedan separados?

`bge-m3` es un modelo de embedding (no LLM generativo). El call path es completamente distinto: `sentence-transformers` + `transformers` directo, no `ollama`. Migrar embeddings a MLX requiere portear `bge-m3` a formato MLX (no está en `mlx-community` al 2026-05-05) o swappear por otro modelo de embedding. Es un proyecto independiente con su propio plan: [`99-AI/system/embedding-swap-qwen3-8b/`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

### ¿Por qué `OllamaBackend` se mantiene durante la migración?

**Insurance de rollback**. Si Ola 2 mergea call sites pero Ola 3/4 descubren regressions:
- VRAM blow-up en Mac 32 GB que no se mitigó con LRU.
- JSON robustness regression en MLX vs Ollama `format="json"`.
- Tool-calling parser nuevo introdujo bugs en `rag do`.
- Eval CIs no-overlapping bajo floor.

…flippear `RAG_LLM_BACKEND=ollama` recupera el comportamiento previo sin revertir commits. Recién en Ola 5, con eval verde sostenido por al menos una semana, se retira.

### ¿Por qué Qwen3-30B-A3B y no command-r en MLX?

[`mlx-community`](https://huggingface.co/mlx-community) no tiene un port de command-r 35B (license + arquitectura `cohere2` que `mlx-lm` todavía no soporta tan bien). Y aun si lo tuviera, Qwen3-30B-A3B-2507 es **MoE con 3B activos**: en Apple Silicon eso significa throughput ~3× sobre un dense 30B y ~10× sobre command-r 35B dense. La calidad medida en eval externos (MMLU, GSM8K) sale par o mejor que command-r en tasks reasoning + tool-calling. Para tasks JSON-strict y `rag do`, es upgrade no downgrade.

### ¿Por qué Qwen3-4B-2507 queda experimental?

El helper-tier es ultra-sensible (memoria `project_reformulate_helper_vs_chat`). Cualquier swap requiere `rag eval` con CIs **no-overlapping arriba del floor**, no sólo "parece similar". Hasta que esa medición se haga, el default seguro es Qwen2.5-3B. Qwen3-4B-2507 está bajado y mapeado en `MLX_MODEL_ALIAS` para correr A/B fácil, pero ningún call site lo usa por default.

## Referencias

### Internas

- Código: [`rag/llm_backend.py`](../rag/llm_backend.py), [`benchmarks/bench_mlx_vs_ollama.py`](../benchmarks/bench_mlx_vs_ollama.py).
- PM doc: [`99-AI/system/mlx-migration/dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch).
- Agent profile: [`.claude/agents/rag-llm.md`](../.claude/agents/rag-llm.md) (sección "MLX migration").
- Repo CLAUDE.md: [`CLAUDE.md`](../CLAUDE.md) (sección "MLX migration").
- Embedding migration paralela: [`99-AI/system/embedding-swap-qwen3-8b/plan`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

### Memorias relevantes (mem-vault)

- `project_reformulate_helper_vs_chat` (2026-04-17) — command-r en helper regresó chains −11pp + 5× latencia. Justifica mantener helper en familia 3B.
- `feedback_local_free_stack` — jamás cloud APIs para LLM/STT/TTS. MLX refuerza esta invariante.
- `reference_ollama.md` — referencia técnica del runtime actual. Se renombra a `reference_mlx.md` en Ola 5.

### Externas

- [Apple MLX framework](https://github.com/ml-explore/mlx)
- [`mlx-lm` package](https://github.com/ml-explore/mlx-lm)
- [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-66e81a666513e518adb90d9e)
- Model cards: [Qwen2.5-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit), [Qwen2.5-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit), [Qwen3-30B-A3B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit), [Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit).
- [Ollama Python client](https://github.com/ollama/ollama-python) (lo que se está reemplazando).
