# Migración Ollama → MLX — Retrospectiva post-cutover

> Doc técnica de referencia. PM doc + estado vivo en el vault: [`99-AI/system/mlx-migration/dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch).
> Código de la abstracción: [`rag/llm_backend.py`](../rag/llm_backend.py).
> Iniciado: **2026-05-05**. Estado: **Ola 5 cerrada — Migración MLX completada 2026-05-06. Default `RAG_LLM_BACKEND=mlx` en producción. Modelos chat Ollama purgados del disco.**

## Resumen ejecutivo

`obsidian-rag` migró los LLMs locales de [Ollama](https://ollama.com) a [Apple MLX](https://github.com/ml-explore/mlx) vía [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Es un **reemplazo total**: `RAG_LLM_BACKEND=mlx` es el default de producción y los modelos chat Ollama fueron purgados del disco (commit `69f1884`). La motivación fue performance en Apple Silicon (kernels nativos Metal, sin overhead de API HTTP de Ollama) + acceso directo a la familia [Qwen3](https://huggingface.co/collections/Qwen/qwen3) en formatos cuantizados.

**Eval post-cutover (floor MLX 2026-05-05, commit `48ababf`):**
- Singles: `hit@5 56.60% [43.40, 69.81] · MRR 0.535 [0.403, 0.667] · n=53`
- Chains: `hit@5 72.00% [56.00, 88.00] · MRR 0.617 [0.447, 0.773]`
- Floor PRE-MLX (archivado): singles `53.70% [40.74, 66.67]`, chains `72.00% [52.00, 88.00]`. MLX supera o iguala ambos.

### Excepciones: Ollama sigue activo (scope narrow)

Ollama sigue siendo dependencia para dos componentes fuera del scope de esta migración:

- **`qwen3-embedding:0.6b`** — embedder activo. El path de embedding usa `ollama.embed()` directamente (no el `LLMBackend` ABC). Migración separada en [`99-AI/system/embedding-swap-qwen3-8b/`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).
- **`OllamaBackend`** en [`rag/llm_backend.py`](../rag/llm_backend.py) se preserva como insurance de rollback histórico. Si no se necesita en ~6 meses (≈ octubre 2026), es candidate a borrar.

**Embeddings (`bge-m3`) no entran en este scope.** Ese path usa `sentence-transformers` directamente, no Ollama.

## Mapping de modelos

| MLX HF ID | Tamaño | Reemplaza Ollama | Tier | Smoke test | Use-cases |
|---|---|---|---|---|---|
| [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit) | ~1.9 GB | `qwen2.5:3b` | **HELPER** (det.) | ✅ 2026-05-05 | `_reformulate_query`, contextual summary, lookup, postprocess, followup judge, tag suggester, history summarisation, datetime fallback. `temp=0, seed=42`. |
| [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) | ~4.3 GB | `qwen2.5:7b` | **CHAT default** | ✅ 2026-05-05 | `rag query`, narrative brief fallback, read summary, prep brief, inbox triage, whisper LLM correct. |
| [`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) | ~17 GB | `command-r:latest`, `qwen2.5:14b` | **HQ tier** | ✅ 2026-05-05 (chat + JSON mode) | Contradiction detector, `_render_morning_structured_prompt` (brief JSON), `rag do` tool-loop. |
| [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) | ~2.5 GB | — | Experimental | ✅ 2026-05-05 | A/B vs el 3B helper. **NO default** hasta CIs no-overlapping arriba del floor en `rag eval`. |

### Justificación

- **Helper queda en familia Qwen 3B**. La memoria [`project_reformulate_helper_vs_chat`](mem-vault) registra que poner `command-r` en `_reformulate_query` regresó chains −11pp + 5× latencia el 2026-04-17. Mismo modelo, runtime distinto.
- **Chat default Qwen2.5-7B** mantiene paridad con la versión Ollama. Cambia el runtime, no el modelo.
- **HQ tier sube a Qwen3-30B-A3B-2507**. Reemplaza `command-r:latest` y `qwen2.5:14b`. **MoE con 3B parámetros activos por forward pass** → throughput cercano a un 3B denso una vez cargado, calidad de modelo grande. El invariante "contradiction detector MUST use chat-tier" sigue intacto: "chat-tier" ahora significa Qwen3-30B-A3B.
- **Qwen3-4B-2507 queda experimental**. En disco y mapeado en `MLX_MODEL_ALIAS` para A/B fácil, pero ningún call site lo usa por default. Requiere `rag eval` con CIs no-overlapping arriba del floor antes de promover.

## Arquitectura del backend

Toda la abstracción vive en [`rag/llm_backend.py`](../rag/llm_backend.py). Default actual (post-Ola 5): `RAG_LLM_BACKEND=mlx`. Rollback a Ollama requiere re-pull de modelos chat (~24 GB, purgados del disco) — ver sección [Rollback a Ollama](#rollback-a-ollama).

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
        │  (rollback/legcy)│      │  (default ✅)    │
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

### `OllamaBackend` (legacy — rollback insurance)

Wrappea el client `ollama`. Identity passthrough: traduce `ChatOptions` → `options` dict, llama a `ollama.chat` / `ollama.generate` / `ollama.list`, y resuelve nombres MLX-style → Ollama-style con `to_ollama()`. Se preserva como insurance de rollback; candidate a borrar en ~octubre 2026 si no se activa.

### `MLXBackend` (default desde Ola 5)

`chat()`, `chat_stream()` y `generate()` implementados. El `__init__` valida que `mlx_lm` esté importable; si no está → `RuntimeError("mlx-lm not installed. Run \`uv pip install '.[mlx]'\` or set RAG_LLM_BACKEND=ollama.")`. `list_available()` scanea `~/.cache/huggingface/hub/` por carpetas `models--mlx-community--*`.

Funcionalidades implementadas:

- `mlx_lm.load(model_id)` con caché de modelos cargados (`self._loaded`, `OrderedDict` LRU).
- Chat template via `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`.
- `mlx_lm.generate(model, tokenizer, prompt, max_tokens=num_predict, sampler=make_sampler(temp, top_p), ...)`.
- LRU eviction: modelos `_BIG_MODELS` (Qwen3-30B-A3B) son single-tenant (evictan todo); pequeños LRU-cap a `_MAX_SMALL_LOADED=3`.
- JSON mode: `_apply_chat_template` agrega instrucción de sistema; `_extract_json` hace strip de fences y aísla el primer bloque `{...}`. Sin grammar-constrained decode nativo — el parser en `rag/__init__.py` hace el repair final.
- `chat_stream()`: `mlx_lm.stream_generate` generator, retorna objetos `ChatResponse(done=False)` con `.message.content` incremental + terminal `done=True, done_reason='stop'`.
- **Tool-calling nativo** via [`rag/mlx_tool_calls.py`](../rag/mlx_tool_calls.py) (Ola 5, commits `82d27d5`, `121008c`). Parser Qwen `<tool_call>{...}</tool_call>` → `Message.ToolCall` formato Ollama. Wireado en [`rag/llm_backend.py:591`](../rag/llm_backend.py).
- **Idle-unload watchdog** daemon thread (commit `91ca89b`). Ver sección [VRAM management](#vram-management).

### `get_backend()` singleton

```python
choice = os.environ.get("RAG_LLM_BACKEND", "mlx").lower()
```

Singleton process-wide. Reset via `reset_backend()` (sólo tests). Valores válidos: `ollama` | `mlx`. Cualquier otra cosa → `ValueError`.

## Aliasing de nombres

`LLMBackend` acepta tanto el nombre Ollama (`qwen2.5:3b`) como el HF ID MLX (`mlx-community/Qwen2.5-3B-Instruct-4bit`). Las funciones `to_mlx()` y `to_ollama()` resuelven en cualquier dirección.

### Tabla `MLX_MODEL_ALIAS`

Definida en [`rag/llm_backend.py`](../rag/llm_backend.py):

```python
MLX_MODEL_ALIAS: dict[str, str] = {
    # Helper tier
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    # Chat default
    "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # HQ tier (contradicciones, brief JSON, rag do)
    "command-r:latest": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "command-r": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen2.5:14b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    # Experimental (no default)
    "qwen3:4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
}
OLLAMA_MODEL_ALIAS = {v: k for k, v in MLX_MODEL_ALIAS.items()}
```

Notar que tres claves Ollama (`command-r:latest`, `command-r`, `qwen2.5:14b`) mapean al mismo HF ID. La inversión es lossy: `OLLAMA_MODEL_ALIAS["mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"]` → `"qwen2.5:14b"` (last-write-wins). No importa en la práctica: la migración va en una sola dirección.

### `to_mlx()` / `to_ollama()`

```python
def to_mlx(model: str) -> str:
    if model.startswith("mlx-community/"):
        return model  # ya es MLX HF ID
    return MLX_MODEL_ALIAS.get(model, model)  # passthrough si no está en la tabla

def to_ollama(model: str) -> str:
    return OLLAMA_MODEL_ALIAS.get(model, model)  # passthrough si no está en la tabla
```

**Modelos que no estén en la tabla → fall through (passthrough)**. Comportamiento by design: forzar al call site a usar nombres canónicos.

## Invariantes preservadas

1. **HELPER_OPTIONS = `{temperature: 0, seed: 42}`** — eval reproducibility floor.
2. **CHAT_OPTIONS = `{num_ctx: 4096, num_predict: 768}`** — VRAM-budgeted. No subir sin re-medir headroom.
3. **`keep_alive=-1` semantics** → emulado con resident-process + LRU eviction + idle watchdog thread dentro de `MLXBackend`. Política LRU: modelos en `_BIG_MODELS` (Qwen3-30B-A3B) son single-tenant (evictan todo); helper Qwen2.5-3B (~1.9 GB) puede coexistir con cualquiera.
4. **Contradiction detector NUNCA usa helper-tier**. Probado non-determinístico + JSON malformado con qwen2.5:3b. Hoy: Qwen3-30B-A3B.
5. **`reformulate_query` MUST use HELPER**, no chat-tier. Memoria 2026-04-17: command-r en helper regresó chains −11pp + 5× latencia.
6. **HyDE OFF por default**. `--hyde` opt-in. qwen2.5:3b drop singles −5pp con Ollama. Re-evaluar con Qwen3-30B-A3B cuando aplique.
7. **Local-first**. Jamás OpenAI/Anthropic/Google APIs. MLX corre 100% local en Apple Silicon.
8. **No caches stale por rename**. Contextual summary cache por **file hash**, no path.
9. **Typo correction default OFF bajo MLX** (`RAG_TYPO_CORRECTION` default ON con Ollama / OFF con MLX). Razón: bug 2026-05-05 (commit `48ababf`) — qwen2.5:3b parafrasea agresivo bajo MLX runtime. Drop singles en eval: `54.72%` (Ollama) → `5.66%` (MLX) cuando typo corrector está ON. Override `RAG_TYPO_CORRECTION=1` siempre gana.

## Streaming (`chat_stream`)

`MLXBackend.chat_stream()` usa `mlx_lm.stream_generate` y retorna un generator de objetos `ChatResponse` con la misma interfaz que `OllamaBackend.chat_stream()`:

```python
for chunk in backend.chat_stream(model="qwen2.5:7b", messages=[...]):
    piece = chunk.message.content   # token incremental (puede ser sub-word)
    if chunk.done:
        break                       # done=True, done_reason='stop', content=''
```

Diferencias vs Ollama streaming:

- **Granularidad de tokens**: MLX puede flushear sub-words o varios tokens por yield; Ollama suele flushear un token por SSE event. El texto final es equivalente.
- **Primer token latency**: MLX no tiene daemon HTTP overhead → primer token más rápido en warm. Cold load puede tardar ~3-8s (modelos pequeños) o ~30-90s (30B).
- `stream=True` pasado via `_mlx_chat_via_backend` es stripeado (no hay flag `stream` en MLX). Los call sites SSE real usan `get_backend().chat_stream(...)` directamente.

## Limitaciones bajo MLX

| Feature | Comportamiento bajo MLX |
|---|---|
| `ollama.generate(prompt='', keep_alive=0)` | Trick de Ollama para descargar un modelo de VRAM. MLX no tiene ese mecanismo. En `MLXBackend.generate()` el `keep_alive=0` es ignorado (no-op silencioso). Para forzar eviction: `backend._loaded.clear()` manual. |
| JSON mode con constrained decode | MLX no tiene grammar-constrained decode nativo (a diferencia del `format="json"` de Ollama). `MLXBackend` usa heurística: instrucción en system prompt + `_extract_json()` post-gen. Funciona para HQ tier (Qwen3-30B es bueno en JSON); puede fallar con el helper 3B en outputs complejos. Downstream repair en `rag/__init__.py` cubre los casos residuales. |
| Multi-process model sharing | Ollama es daemon compartido; cada proceso MLX carga su propia copia. Ver gotcha en sección [Multi-process](#multi-process-gotcha-importante). |
| Tool-calling | Implementado nativo via [`rag/mlx_tool_calls.py`](../rag/mlx_tool_calls.py) (Ola 5). Parser Qwen `<tool_call>{...}</tool_call>` → `Message.ToolCall` formato Ollama. |

## Cómo flipear al backend MLX

`RAG_LLM_BACKEND=mlx` ya es el default. Los 15 plists de producción lo tienen seteado explícitamente en su bloque `EnvironmentVariables`.

### Ad-hoc (shell / desarrollo)

```bash
export RAG_LLM_BACKEND=mlx
rag query "qué notas tengo sobre X" --plain
```

Para verificar que el backend está activo:

```python
from rag.llm_backend import get_backend, reset_backend
reset_backend()  # fuerza re-resolución con el env actual
b = get_backend()
print(b.name)           # → "mlx"
print(b.list_available())  # lista modelos MLX en ~/.cache/huggingface/hub/
```

### Daemons launchd

Los plists ya incluyen `RAG_LLM_BACKEND=mlx`. Para regenerar si se modificó `_services_spec()`:

```bash
rag setup
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
launchctl load   ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
```

### Tests

Los tests usan la fixture autouse `_force_ollama_backend_for_tests` en `tests/conftest.py` que fuerza `RAG_LLM_BACKEND=ollama` por test. Como Ollama-chat no está en disco, los tests que asumen el backend deben monkeypatchear `ollama.chat` directamente:

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

> **Pre-requisito obligatorio: re-pull de los 3 modelos chat (~24 GB, purgados del disco, 15-30 min)**. Sin re-pull previo, el rollback falla con `model 'X' not found`.

```bash
# Paso 1: re-pull modelos chat (purgados en commit 69f1884)
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen3:30b-a3b

# Paso 2: deshabilitar mlx en shell
export RAG_LLM_BACKEND=ollama

# Paso 3: actualizar plists (sacar RAG_LLM_BACKEND o ponerlo en "ollama")
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist
# editar plists manualmente o via rag setup con backend revertido
launchctl load   ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist
```

`OllamaBackend` sigue en [`rag/llm_backend.py`](../rag/llm_backend.py) como insurance de rollback. Candidate a borrar en ~octubre 2026.

## VRAM management

Apple Silicon usa **unified memory**: GPU + CPU comparten RAM. Cargar 4 modelos = ~26 GB en peak. En Macs 16 GB → imposible. En 32 GB → viable con LRU. En 64 GB+ → todos pueden quedar resident.

### Estrategia LRU

```
slot 0: helper (~1.9 GB)         LRU cap=3 small, puede coexistir con cualquiera
slot 1: chat-default (~4.3 GB)   LRU cap=3 small
slot 2: HQ tier (~17 GB)         BIG_MODELS → single-tenant, evicta todo lo demás
slot 3: experimental (~2.5 GB)   LRU cap=3 small, on-demand
```

`_BIG_MODELS` lleva el set de los modelos "grandes" (hoy sólo Qwen3-30B-A3B). Eviction policy (implementada en `_evict_for()`):

```
on load(M):
    if M in BIG_MODELS:
        evict ALL other models  # pure single-tenant
    else:
        evict any big first
        LRU-trim while len(loaded) >= MAX_SMALL_LOADED (=3)
    self._loaded[M] = mlx_lm.load(M)  # OrderedDict LRU, newest=last
```

### Idle-unload watchdog (implementado, commit `91ca89b`)

Daemon thread que evicta modelos no usados por más de `RAG_MLX_IDLE_TTL` segundos (default `1800` = 30 min). Se spawnea en `MLXBackend.__init__()`. Checa idle-ness cada `max(60, _idle_ttl // 4)` segundos.

```bash
# Disable total:
RAG_MLX_IDLE_TTL=0 rag query "..."
# o:
RAG_MLX_IDLE_DISABLE=1 rag query "..."
```

Equivalencia con Ollama:

| Ollama | MLX |
|---|---|
| `keep_alive=-1` (resident hasta que daemon muera) | resident + LRU + idle watchdog (capacity-driven + TTL) |
| `keep_alive=0` (descarga después del request) | no implementado — no-op silencioso |
| `keep_alive="5m"` | TTL custom por request (parseado, usa `_idle_ttl` del objeto) |

## Plan de cutover (5 olas)

Detalle vivo en [`dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch). Resumen:

| Ola | Scope | Estado |
|---|---|---|
| **0** | PM bootstrap (dispatch.md, worktrees, claude-peers announce) | ✅ |
| **1** | Foundation: `rag/llm_backend.py`, benchmark harness, `pyproject.toml`, docs base | ✅ |
| **2** | `MLXBackend` funcional + streaming. ~29 call sites via `_TimedOllamaProxy` ruteados. 14 raw `ollama.chat(stream=True)` sites migrados a `get_backend().chat_stream(...)`. Smoke tests 4 modelos OK. | ✅ |
| **3** | Runtime: launchd plists con `RAG_LLM_BACKEND=mlx`, LRU eviction, métricas MLX (`/api/metrics/mlx`). Telemetría `backend` field en `rag_queries` (commit `673fae1`). | ✅ |
| **4** | Tests + eval. `rag eval` con bootstrap CIs vs floor. Determinismo x2 corridas. | ✅ |
| **5** | Cutover (2026-05-06): default `mlx`, modelos chat Ollama purgados del disco (commit `69f1884`), tool-calling nativo via `rag/mlx_tool_calls.py` (commits `82d27d5`, `121008c`), retiro watchdog `_ollama_health.py`, retiro `_ollama_alive`/`_ollama_chat_probe`/`_ollama_restart_if_stuck`, simplificación `/api/ollama/unload` a MLX-only, simplificación `_check_ollama_health` a solo `EMBED_MODEL`. | ✅ |

## Troubleshooting

### `RuntimeError: mlx-lm not installed`

Setear `RAG_LLM_BACKEND=mlx` sin tener `mlx-lm` instalado. Fix:

```bash
uv tool install --reinstall --editable '.[mlx]'
# o
uv pip install 'mlx-lm>=0.18'
```

`mlx-lm` sólo está en el extra `mlx`, marker `sys_platform == 'darwin' and platform_machine == 'arm64'`. En Linux/Intel el extra no se resuelve y los tests con marker `requires_mlx` se auto-skipean en CI.

### MLX GPU Hang bajo memory pressure o coexistencia con Ollama

Dos escenarios:

1. **Memory pressure ≥83% + cold-load de modelo grande** — el Metal allocator puede colgar esperando VRAM libre. Fix: bajar otras apps, esperar que el watchdog evicte modelos idle, o `RAG_MEMORY_PRESSURE_DISABLE=1` + manejo manual.
2. **Ollama VRAM ocupada** (aunque `rag stop` haya corrido, Ollama puede retener VRAM con `keep_alive` activo). Fix: `OLLAMA_KEEP_ALIVE=5m` en el plist de Ollama (o `OLLAMA_KEEP_ALIVE=1m` para testing), verificar con `ollama ps` que no haya modelos cargados antes de cargar uno grande en MLX.

Memoria `project_mlx_gpu_hang_under_memory_pressure` tiene más contexto.

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

By design: el extra `mlx` tiene marker Apple-Silicon-only. Tests marcados `@pytest.mark.requires_mlx` corren localmente (Mac) y se skipean en CI. Para forzar local: `pytest -m requires_mlx`.

### "El backend no está en MLX"

Verificar con:

```python
from rag.llm_backend import get_backend, reset_backend
reset_backend()
print(get_backend().name)  # debe ser "mlx"
```

Si sale `"ollama"`, verificar que `RAG_LLM_BACKEND` no esté seteado a `ollama` en el entorno o en el plist. Los 15 plists de producción tienen `RAG_LLM_BACKEND=mlx` explícito.

## Comparación Ollama vs MLX

| Dimensión | Ollama | MLX |
|---|---|---|
| **Runtime** | Daemon HTTP (`ollama serve` :11434) | In-process (`mlx_lm.load` + `generate`) |
| **Cuantización default** | GGUF Q4_K_M | 4-bit MLX (group-wise) |
| **Keep alive** | Nativo (`keep_alive=-1`) | Emulado (resident + LRU + idle watchdog thread) |
| **JSON mode** | `format="json"` (constrained decode) | Manual: parser + repair en post |
| **Tool-calling** | Format por modelo (command-r XML, Qwen JSON) | Nativo via [`rag/mlx_tool_calls.py`](../rag/mlx_tool_calls.py) — parser Qwen `<tool_call>` |
| **Streaming** | SSE nativo del daemon | `mlx_lm.stream_generate` generator |
| **Modelos disponibles** | Catálogo Ollama (curado, lag vs HF) | HuggingFace `mlx-community/*` (acceso directo a Qwen3-30B-A3B) |
| **Multi-process** | Sí (daemon compartido) | No (cada proceso carga su copia) — gotcha si daemon `web` y daemon `morning` corren simultáneo |
| **Status** | Legacy (modelos chat purgados del disco 2026-05-06) | **Default de producción** |

### Multi-process gotcha (importante)

Ollama corre como daemon → varios procesos comparten el modelo cargado en VRAM una sola vez. **MLX no tiene daemon**: cada proceso que llama `get_backend()` con `RAG_LLM_BACKEND=mlx` carga su propia copia del modelo en su propio espacio de memoria. Cargar Qwen3-30B-A3B en 3 procesos simultáneos = ~51 GB. Mitigación actual: que sólo el daemon `web` (FastAPI) tenga MLX backend con modelos grandes; el CLI `rag query` pega al daemon en lugar de cargar MLX local cuando es posible.

## FAQ / decisiones explícitas

### ¿Por qué reemplazo total y no coexistencia permanente?

Mantener dos backends para siempre es deuda. El código (`LLMBackend` ABC + dos subclases) está pensado como **insurance de rollback durante la migración**, no como abstracción permanente. Con Ola 5 cerrada y eval verde, `OllamaBackend` es candidate a borrar en ~octubre 2026.

### ¿Por qué embeddings quedan separados?

`bge-m3` y `qwen3-embedding:0.6b` son modelos de embedding (no LLM generativos). El call path usa `sentence-transformers` / `ollama.embed()` directamente, no el `LLMBackend` ABC. Migrar embeddings a MLX requiere portear a formato MLX o swappear por otro modelo. Es un proyecto independiente: [`99-AI/system/embedding-swap-qwen3-8b/`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

### ¿Por qué Qwen3-30B-A3B y no command-r en MLX?

[`mlx-community`](https://huggingface.co/mlx-community) no tiene un port de command-r 35B (license + arquitectura `cohere2` que `mlx-lm` no soporta bien). Y Qwen3-30B-A3B-2507 es **MoE con 3B activos**: throughput ~3× sobre un dense 30B y calidad superior en tasks reasoning + JSON. Para tasks JSON-strict y `rag do`, es upgrade no downgrade.

### ¿Por qué `OllamaBackend` se mantiene post-cutover?

**Insurance de rollback**. Si emerge una regresión grave (VRAM blow-up, JSON robustness regression, bug en tool-calling parser), flippear `RAG_LLM_BACKEND=ollama` recupera el comportamiento previo sin revertir commits — siempre que se hayan re-pulleado los modelos Ollama primero (ver sección [Rollback a Ollama](#rollback-a-ollama)).

### ¿Por qué Qwen3-4B-2507 queda experimental?

El helper-tier es ultra-sensible (memoria `project_reformulate_helper_vs_chat`). Cualquier swap requiere `rag eval` con CIs **no-overlapping arriba del floor**, no sólo "parece similar". Qwen3-4B-2507 está en disco y mapeado en `MLX_MODEL_ALIAS` para A/B fácil, pero ningún call site lo usa por default.

## Referencias

### Internas

- Código backend: [`rag/llm_backend.py`](../rag/llm_backend.py).
- Tool-calling MLX: [`rag/mlx_tool_calls.py`](../rag/mlx_tool_calls.py).
- Benchmark harness: [`benchmarks/bench_mlx_vs_ollama.py`](../benchmarks/bench_mlx_vs_ollama.py).
- PM doc: [`99-AI/system/mlx-migration/dispatch.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch).
- Repo CLAUDE.md: [`CLAUDE.md`](../CLAUDE.md) (sección "MLX migration").
- Embedding migration paralela: [`99-AI/system/embedding-swap-qwen3-8b/plan`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

### Memorias relevantes (mem-vault)

- `project_reformulate_helper_vs_chat` (2026-04-17) — command-r en helper regresó chains −11pp + 5× latencia. Justifica mantener helper en familia 3B.
- `project_mlx_gpu_hang_under_memory_pressure` — dos escenarios de GPU hang + mitigaciones.
- `feedback_local_free_stack` — jamás cloud APIs para LLM/STT/TTS. MLX refuerza esta invariante.

### Externas

- [Apple MLX framework](https://github.com/ml-explore/mlx)
- [`mlx-lm` package](https://github.com/ml-explore/mlx-lm)
- [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-66e81a666513e518adb90d9e)
- Model cards: [Qwen2.5-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit), [Qwen2.5-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit), [Qwen3-30B-A3B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit), [Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit).
- [Ollama Python client](https://github.com/ollama/ollama-python) (lo que se reemplazó).
