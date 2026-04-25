---
name: profile
description: Profile any function in rag.py / web/server.py empirically (cProfile + monkey-patch counter) before believing any "win" claim from a static audit. Use BEFORE applying any performance fix to validate the bottleneck is real and the impact estimate is correct. Triggers on `/profile <function>`, `profilá <función>`, `dame el perfil de <función>`, `cuántas veces se llama X`, `cuánto tarda X realmente`.
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(.venv/bin/python)
    - Exec(.venv/bin/python -m cProfile *)
    - Exec(sqlite3)
    - Exec(rg)
---

# /profile — empirical profiling before optimization

Esta skill existe porque el 2026-04-25 aprendimos a la mala que un audit
estático del rag-perf-auditor declaró un "win de 450ms en `_load_corpus`"
que en realidad eran 0.18ms (la función ya tenía cache module-level
desde un commit del día anterior). Profilar empírico costó 5min y
salvó horas de implementar un fix fantasma.

**Regla de oro**: nunca apliques un fix de performance basado solo en
code-reading. Profilá primero, números reales, después decidís.

## Inputs

El user típicamente invoca con uno de estos patrones:
- `/profile retrieve` — función específica
- `/profile retrieve --query "productividad y deep work"` — con caller
- `profilá _load_behavior_priors` — lenguaje natural

Si el user no especifica un caller representativo, preguntale:
"¿con qué query / argumentos típicos debería profilar `<función>`?"
o usa un default razonable de `queries.yaml` si la función está en
el path de retrieve.

## Workflow

### Paso 1 — Localizá la función

```bash
rg -n "^def {function_name}|^    def {function_name}" rag.py web/server.py
```

Si hay más de una match (ej. métodos de distintas classes), preguntá al
user cuál querés profilar.

### Paso 2 — Generá un script efímero de profiling

Crea un script en `/tmp/devin-profile-<function>.py` con:
1. Monkey-patch counter sobre la función (cuenta # invocaciones + ms total)
2. Caller que ejerce la función (con args razonables)
3. cProfile sobre todo el bloque

Template:

```python
#!/usr/bin/env python3
"""Devin profile — {function_name} on {date}."""
import sys, time, cProfile, pstats, io
from pstats import SortKey

sys.path.insert(0, "/Users/fer/repositories/obsidian-rag")

# Monkey-patch counter
import rag
_original = getattr(rag, "{function_name}")
_stats = {"n": 0, "total_ms": 0.0, "min_ms": 1e9, "max_ms": 0.0}

def _wrapper(*args, **kwargs):
    _stats["n"] += 1
    t0 = time.perf_counter()
    result = _original(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _stats["total_ms"] += elapsed_ms
    _stats["min_ms"] = min(_stats["min_ms"], elapsed_ms)
    _stats["max_ms"] = max(_stats["max_ms"], elapsed_ms)
    return result

setattr(rag, "{function_name}", _wrapper)

# Profile
pr = cProfile.Profile()
pr.enable()

# === CALLER (ajustar según función) ===
{caller_block}
# =======================================

pr.disable()

# Output
print(f"\n=== Counter for {function_name} ===")
print(f"Invocations:   {_stats['n']}")
print(f"Total ms:      {_stats['total_ms']:.2f}")
print(f"Avg ms/call:   {_stats['total_ms']/max(1,_stats['n']):.4f}")
print(f"Min ms:        {_stats['min_ms']:.4f}")
print(f"Max ms:        {_stats['max_ms']:.4f}")

print(f"\n=== cProfile top 30 (cumtime) ===")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE).print_stats(30)
print(s.getvalue())

print(f"\n=== cProfile top 15 (tottime) ===")
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).sort_stats(SortKey.TIME).print_stats(15)
print(s2.getvalue())
```

Ejemplos de `{caller_block}` por función:

| Función | Caller |
|---|---|
| `retrieve` | `col = rag.get_db(); rag.retrieve(col, "test query", k=5)` |
| `_load_corpus` | `col = rag.get_db(); rag.retrieve(col, "test", k=5)` (se llama internamente) |
| `_load_behavior_priors` | `rag._load_behavior_priors()` directamente |
| `query_embed_local` | `rag.query_embed_local("test query")` |
| `_wa_extract_actions` | construir messages dummy + invocar |

Si el user no especificó caller, inferí del contexto (qué función es
+ dónde se invoca).

### Paso 3 — Ejecutá el script

```bash
cd /Users/fer/repositories/obsidian-rag
.venv/bin/python /tmp/devin-profile-{function_name}.py 2>&1 | head -100
```

Timeout suggerido: 60s. Si supera, el caller que elegimos es demasiado
lento — preguntá al user.

### Paso 4 — Cross-reference con telemetría existente

Si la función está instrumentada con `rag_log_sql` o similar, query
directo:

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/ragvec.db "
  SELECT op, COUNT(*) as n, ROUND(AVG(duration_ms), 2) as avg_ms,
         ROUND(MAX(duration_ms), 2) as max_ms
  FROM rag_log_sql
  WHERE op LIKE '%{function_name}%' AND ts > strftime('%s','now')-86400
  GROUP BY op
  ORDER BY n DESC LIMIT 20;
"
```

Esto te da la realidad de producción (no solo el profile sintético).

### Paso 5 — Devolveme reporte estructurado

```markdown
# Profile de `{function_name}` — {fecha}

## Counter (monkey-patched)

- Invocaciones por llamada típica: N
- Tiempo total: NN.NNms
- Promedio por call: NN.NNms
- Min/max: NN.NN / NN.NN ms

## cProfile top hits

| ncalls | cumtime | function |
|---|---|---|
| ... | ... | ... |

**Hot path real**: `<función>` aporta X% del cumtime de la pipeline.

## Telemetría producción (último 24h, si aplica)

- Count: N invocaciones reales
- p50: NNms / p95: NNms / p99: NNms

## Veredicto

[basado en los números — claro y accionable]:
- "Vale la pena optimizar — es N% del hot path crítico"
- "No vale la pena — ya está cacheado / es 0.X% del tiempo"
- "Vale la pena pero el fix correcto es Y, no Z (el LIMIT que sugirió X audit)"
```

## Anti-patterns

- ❌ NO repliques los números del audit estático sin validar — la única
  fuente de verdad son los counters reales.
- ❌ NO afirmar "vale la pena optimizar" si el % del hot path es <5% (a
  menos que sea un path muy frecuente — en ese caso quantificar).
- ❌ NO declarar OK algo que está cacheado solo por la presencia del
  cache — verificá que el hit rate sea alto en producción (telemetry).
- ✅ SÍ distinguir warm/cold cache empíricamente (correr 2 invocaciones
  warmup + medir las siguientes).
- ✅ SÍ proponer el siguiente paso después del profile (qué fix
  implementar, qué medir post-fix para validar).

## Limitaciones conocidas

- Esta skill **no** reemplaza load testing realista. Profile sintético
  con 1-3 invocaciones puede no exponer concurrency issues.
- En funciones con efectos remotos (Ollama, sqlite-vec serve), el
  network/IO domina y el cProfile en sí da menos info útil.
- Si la función NO es importable directo desde `rag` (módulos privados),
  ajustá el path o agregá un wrapper temporal.
