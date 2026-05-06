# Wave-8 gotchas (2026-04-28) — pipeline de filtros + carry-over

## Filtros definidos pero no cableados

Síntoma: `_XxxFilter` o función `_strip_*` / `_redact_*` con docstring + regex completo, ningún call site la invoca. Bug que se suponía fixeada sigue ahí.

Caso real: `_strip_foreign_scripts` ([`web/server.py:1504-1531`](../web/server.py)) existía con docstring "Remove CJK/Cyrillic/Hebrew/Arabic". Nunca se llamaba. CJK leak en weather siguió hasta wave-8.

Cómo evitarlo:
1. Cuando agregás filtro, también editá `_emit()` helper en `gen()` (~línea 11631) Y pipeline de cache replay (~línea 9887, `_redact_pii(_sem_text)`).
2. Antes de "queda para wirear después", ya escribí el call site.
3. Test de regresión [`tests/test_filter_wiring.py`](../tests/test_filter_wiring.py) falla si clase `_*Filter` o función `_strip_*`/`_redact_*` está sin call site. Si false-positive intencional → allowlist, no borrar.

## Carry-over del pre-router silenciosamente sobrescrito por fast-path

Síntoma: lógica al inicio de `gen()` computa `_forced_tool_pairs`. Log dice que se computó. Tool nunca corre porque otro branch downstream re-llama `_detect_tool_intent(question)` y descarta tu carry-over.

Caso real: pre-router seteaba `_forced_tool_pairs = [('weather', {'location': 'Barcelona'})]` por carry-over de "y en Barcelona?". Línea 10996 hacía `_forced_tools = [] if _propose_intent else _detect_tool_intent(question)` — la query sola no matchea keyword, retornaba `[]`. Fix: `_forced_tools = list(_forced_tool_pairs)`.

Cómo evitarlo:

```bash
grep -n '_detect_tool_intent\|_forced_tools\s*=' web/server.py
```

Regla: pre-router corre UNA vez al inicio de `gen()`, todo el resto del flow LEE de `_forced_tool_pairs`.

## Bumpear `_FILTER_VERSION` es parte del fix

Síntoma: arreglaste filtro / system prompt / regex. Validás Playwright. Test reporta bug sigue. La causa: semantic cache sirviendo respuestas pre-fix porque cache key no incluye nada que tu fix haya cambiado.

Mecanismo: `_FILTER_VERSION` ([`rag/__init__.py:6017`](../rag/__init__.py)) está horneado en `_hash_chunk_count` y usado en corpus_hash → cache key. Bumpear la string invalida TODAS las entries pre-fix.

Cuándo bumpear:
- Cambia regex que afecta tools_fired (PII redact, raw tool stripper, iberian leaks, foreign scripts).
- Cambia `_WEB_SYSTEM_PROMPT` o cualquier REGLA N.
- Cambia traducción de descriptions inyectada al CONTEXTO.

Cuándo NO: perf/refactors sin output change, features off-by-default, herramientas administrativas.

Naming: `wave<N>-<YYYY-MM-DD>` ej. `wave8-2026-04-28`.
