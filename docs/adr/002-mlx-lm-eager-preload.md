# ADR-002: MLX-LM Eager Preload

## Contexto

`transformers` 5.x + `accelerate` 1.13 tienen una race condition de inicialización: cuando un **thread no-main** (ej. prewarm del reranker) inicia `from transformers import AutoTokenizer` antes de que el **main thread** haya tocado `mlx_lm`, `accelerate.state` queda en estado parcialmente inicializado en `sys.modules`.

Todo import posterior de `mlx_lm` desde cualquier thread raisa:

```
cannot import name 'AcceleratorState' from partially initialized module 'accelerate.state'
```

Esto decapita `MLXBackend()` con un `RuntimeError("mlx-lm not installed")` **engañoso** (el cause real es el circular import, no la ausencia del paquete).

## Fix

Forzar el import de `mlx_lm` en el **main thread**, **antes** de que cualquier prewarm thread arranque:

```python
try:
    import mlx_lm  # noqa: F401
except ImportError:
    pass
```

Una vez `transformers` y `accelerate` quedan inicializados limpios en `sys.modules`, los handlers de las requests pueden hacer `import mlx_lm` libremente desde cualquier thread.

## Aplicación

- `web/server.py` (top-level import, 2026-05-08).

## Estado

**Accepted** — aplicado en producción desde 2026-05-08.
