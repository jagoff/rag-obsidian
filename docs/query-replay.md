# Query replay (`rag replay`)

Sprint 3 Tarea B (2026-05-04). `rag replay` rerunea queries históricas de `rag_queries` y diffa el resultado nuevo contra el output original.

## Modos

- `rag replay <id>` — diff de un query puntual (exit 0 = sin regresión, 1 = drift, 2 = id no found / q vacío, 3 = corpus drift sin --force)
- `rag replay <id> --explain` — muestra los paths nuevos sin comparar (exit 0 siempre)
- `rag replay --bulk [--since 7d] [--limit 20] [--filter-cmd CMD]` — batch sobre el historial

## Flags

- `--skip-gen` — solo comparar paths, sin LLM gen (más rápido, útil para CI)
- `--no-cache` — disable semantic cache durante el replay (default ON — replay debe ser reproducible)
- `--force` — continuar aunque haya corpus drift
- `--json` / `--plain` — output alternativo al Rich default

## Métricas de diff

- `path_jaccard` — Jaccard@5 entre paths originales y nuevos (1.0 = idénticos)
- `top3_changed` — cambió alguno de los 3 primeros paths
- `response_cosine` — cosine entre respuesta cacheada y nueva (cuando `response_text` disponible via Sprint 3 Tarea A)
- `response_hash_match` — hash comparison cuando solo hay `response_hash` en `extra_json`
- `corpus_drift` — flag cuando `corpus_hash` del row no matchea el corpus actual

## Verdicts

- `equivalent` — paths y respuesta dentro de los umbrales (jaccard ≥ 0.4 O top3 igual, cosine ≥ 0.85 si hay texto)
- `path_drift` — jaccard < 0.4 O top3 cambió, respuesta no comparada
- `response_drift` — respuesta divergente (cosine < 0.85 o hash mismatch)
- `regression` — error durante el replay o q vacío

## Invariantes

- `RAG_EXPLORE` scrubbed durante replay (misma invariante que `rag eval`)
- `RAG_SKIP_BEHAVIOR_LOG=1` durante replay — no contamina telemetría
- `auto_filter=False` — usa filtros del `filters_json` log, no re-infiere
- Forward-compatible con Sprint 3 Tarea A: funciona sin `response_text`/`response_hash` (solo compara paths en ese caso)

## Implementación

`_replay_load_row`, `_replay_cosine`, `_replay_query_row`, `_replay_render_single` en [`rag/__init__.py`](../rag/__init__.py). Tests: [`tests/test_rag_replay.py`](../tests/test_rag_replay.py) (27 casos).
