# Microservices-Oriented Refactor Plan

Estado al 2026-05-17. El proyecto sigue siendo local-first y single-process en
varios caminos calientes, asi que "microservicios" aca significa primero
limites de dominio claros, modulos chicos y contratos testeables. Convertir
cada limite en proceso/daemon separado queda como segunda etapa.

## Archivos mas grandes

Medido con `rg --files | xargs wc -l | sort -nr | head`:

| Archivo | Lineas aprox. | Lectura arquitectonica |
|---|---:|---|
| `rag/__init__.py` | 53k | CLI, retrieval, indexing, storage, telemetry y agentes mezclados |
| `web/server.py` | 24k | Gateway HTTP + chat + WhatsApp + dashboard + status + uploads |
| `web/static/app.js` | 6.8k | Chat UI monolitica, aunque ya convive con modulos ESM |
| `web/static/wa.css` | 3.6k | UI WhatsApp con estilos de varias responsabilidades |
| `web/static/home.v2.css` | 3.4k | Home/dashboard visual en un solo stylesheet |
| `web/learning_queries.py` | 3.1k | Queries/analytics de aprendizaje con mucho SQL embebido |

## Limites de servicio propuestos

| Servicio logico | Responsabilidad | Entry points actuales | Estado |
|---|---|---|---|
| Web Gateway | FastAPI, auth, SSE, static files, request validation | `web/server.py`, `web/basic_routes.py`, `web/action_routes.py` | En progreso |
| Chat/Retrieval API | Orquestar `retrieve`, rerank, postprocess, streaming de respuesta | `rag.__init__.py`, `web/server.py` | A separar detras de un contrato Python primero |
| Indexer | Chunking, embeddings, entity extraction, URL/wiki indexing | `rag.__init__.py`, `rag/vector_store.py`, `rag/contextual_retrieval.py` | Mayor deuda |
| Integrations ETL | Gmail, Calendar, Reminders, WhatsApp, Chrome, Drive, finance | `rag/integrations/**`, scripts `ingest_*` | Bastante modular, falta contrato comun |
| Runtime Scheduler | Jobs frecuentes/nocturnos, supervisor, IPC/eventos | `rag/runtime/**` | Buen limite inicial |
| Telemetry/State | SQLite state, logs, behavior, feedback, cache | `rag.__init__.py`, `rag/runtime/_telemetry.py` | A aislar para cortar dependencias |
| Local Model Runtime | MLX chat/embed/rerank/NLI lifecycle | `rag/llm_backend.py`, `rag/mlx_*`, partes de `rag.__init__.py` | Parcialmente modular |
| WhatsApp Surface | Web routes + bridge client + drafts + memory + voice | `web/server.py`, `rag/integrations/whatsapp/**` | Dominio claro, routes aun en server |

## Primer corte aplicado

- `web/chat_schemas.py`: modelos y validaciones de `ChatRequest` y
  `ChatAttachment`.
- `web/chat_uploads.py`: sanitizacion EXIF/HEIC, extraccion de texto de
  uploads y copia al vault.
- `rag/pipeline_flags.py`: flags de adaptive routing, NLI grounding,
  fast-path lookup y pools por intent.

Los nombres historicos privados siguen re-exportados desde `web.server` y
`rag` para mantener compatibilidad con tests y call sites existentes.

## Siguiente secuencia recomendada

1. Extraer `web/whatsapp_routes.py` desde el bloque `/api/wa/**`.
   Es el mayor corte seguro dentro de `web/server.py`, pero necesita un
   objeto de dependencias para no arrastrar globals.
2. Extraer `rag/indexing_service.py` con `_index_single_file`,
   `_run_index_inner`, `_do_index` y helpers de chunk/hash. El contrato debe
   aceptar `vault_path`, `collection`, `embed_fn`, `telemetry`.
3. Extraer `rag/retrieval_service.py` con `retrieve`, `multi_retrieve`,
   `run_chat_turn` y modelos de resultado. Es el paso que mas reduce
   acoplamiento, pero requiere primero estabilizar tipos.
4. Partir `web/static/app.js` por flujo de UI: composer/uploads, SSE chat,
   sources/rendering, settings, contact commands.
5. Convertir integraciones ETL en workers con contrato uniforme:
   `discover -> normalize -> write_notes -> index_delta`.

## Regla practica

No pasar a procesos separados hasta que el limite funcione como modulo Python
con contrato chico y tests focalizados. Si no se puede importar sin levantar
todo `rag`, todavia no esta listo para ser microservicio.
