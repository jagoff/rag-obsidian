# scratch_eval — harness temporal de eval end-to-end

Batería automatizada de preguntas reales contra el sistema RAG para
validar que:

- El fix de `_format_forced_tool_output` (pre-router del `/api/chat`) no
  leakea tool names ni droppea items.
- El fix de entity-lookup regex (`qué sabés de X`, `contame de X`, etc.)
  dispara el handler correcto y responde en tiempo razonable.
- No hay regresiones obvias en metachat, semantic vault, WhatsApp,
  calendar/reminders, edge cases.

**ESTO ES TEMPORAL.** Cuando terminemos de validar los fixes, la carpeta
entera se borra con `rm -rf scratch_eval/`. No lo versionamos como
suite oficial — para eso está `tests/` + el comando `rag eval`.

## Uso

Requiere el servidor web corriendo (`launchctl start com.fer.obsidian-rag-web`
o `rag web`), escuchando en `http://localhost:8765`.

```bash
# Dispara todas las queries del YAML y escribe reports timestamped:
.venv/bin/python scratch_eval/run_eval.py

# Solo una categoría:
.venv/bin/python scratch_eval/run_eval.py --category forced_tools

# Override del endpoint (por si arrancás el server en otro puerto):
.venv/bin/python scratch_eval/run_eval.py --base-url http://localhost:8787
```

Output:
- `scratch_eval/reports/run_<timestamp>.json` — resultados estructurados.
- `scratch_eval/reports/run_<timestamp>.md` — resumen legible (tabla +
  failures destacados).
- Stdout: tabla coloreada con pass/fail por query.

Exit code: `0` si todas pasan, `1` si alguna falla.

## Estructura de una query (queries.yaml)

```yaml
- id: entity_astor
  category: entity_lookup
  question: qué sabés de Astor
  checks:
    max_latency_ms: 15000       # hard ceiling
    target_latency_ms: 5000     # soft target (warning only)
    require_substrings: ["astor"]  # case-insensitive
    forbid_leak_patterns: true  # común a casi todos
    allow_refusal: false        # si true, "no tengo esa información" no es failure
  notes: >
    Dispara handle_entity_lookup vía el regex ampliado.
    Esperamos info cross-source (vault + WhatsApp + Calendar).
```

## Borrar cuando termine

```bash
rm -rf scratch_eval/
```
