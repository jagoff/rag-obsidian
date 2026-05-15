# ADR-003: HuggingFace Offline + Fastembed Cache Bootstrap

## Contexto

`huggingface_hub`, `transformers` y `sentence_transformers` leen variables de entorno al **import time** y cachean la flag `offline` globalmente. Si las variables se setean tarde (dentro de un lazy loader), el cliente intenta HEAD requests y falla en setups air-gapped.

Adicionalmente, `mem0`'s Qdrant vector store lazy-carga `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")`. El cache default de fastembed es `tempfile.gettempdir()/fastembed_cache`, que en macOS resuelve a `/var/folders/.../T/fastembed_cache` — un path que macOS **periodicamente garbage-collecta**.

Combinado con `HF_HUB_OFFLINE=1`, un cache miss post-GC significa que fastembed no puede redescargar el modelo y el encoder falla al inicializar:

```
Could not find the model tar.gz file at /var/folders/.../T/fastembed_cache/bm25
and local_files_only=True
```

## Fix

Setear **antes de cualquier import** que transitivamente tire de `huggingface_hub`:

```python
for k, v in {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TQDM_DISABLE": "1",
    "FASTEMBED_CACHE_PATH": os.path.join(os.path.expanduser("~"), ".cache", "fastembed"),
}.items():
    os.environ.setdefault(k, v)
```

`os.environ.setdefault` respeta un operator que quiera online mode: si ya exportó la variable explícitamente antes de lanzar, no la sobreescribimos.

## Estado

**Accepted** — aplicado en `rag/__init__.py` desde 2026-04-20 (consolidado de dos bloques previos).
