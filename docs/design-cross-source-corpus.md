# Cross-Source Corpus — Design Document (fase 0)

**Estado:** exploratorio · **Autor:** peer uy49o1ap (track #2 del roadmap de 4 tracks) · **Fecha:** 2026-04-15

## 0. Contexto

El RAG actual (`obsidian-rag` v7) indexa únicamente el vault de Obsidian en `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` (521 notas, ~3.5k chunks). La hipótesis detrás del track #2 es que el valor del sistema crece no-linealmente si el mismo `retrieve()` puede devolver, para una query como *"qué decidimos con Juan sobre la migración"*, un mix coherente de:

- una nota de `02-Areas/` donde se tomó la decisión
- el thread de Gmail donde Juan mandó los requerimientos
- el mensaje de WhatsApp donde se cerró la fecha
- el evento de Calendar al que quedó amarrada la reunión

Hoy cada una de esas fuentes vive aislada. Este documento define cómo unificarlas **sin romper el pipeline actual** y qué preguntas abiertas necesita responder el usuario antes de escribir una sola línea de código de ingesta.

Este doc es *survey + diseño*, no un plan de implementación. No toca `rag.py` ni `mcp_server.py`. Depende del track #1 (feedback loop sobre `queries.jsonl`) para poder medir regresiones cuando se mezclen ontologías en el pool de retrieval.

---

## 1. Inventario de fuentes

### 1.1 Estado actual de MCPs en el host

| Fuente | MCP server | Estado integración obsidian-rag | Ruta/command |
|---|---|---|---|
| Obsidian vault | n/a (FS directo) | **Activa (única)** — indexing en `_index_single_file` | `OBSIDIAN_RAG_VAULT` env |
| Apple Mail | `mcp__apple-mcp__mail` | **No integrada** | apple-mcp (stdio, nativo) |
| Apple Calendar | `mcp__apple-mcp__calendar` | No integrada | apple-mcp |
| Apple Reminders | `mcp__apple-mcp__reminders` | No integrada (sí consumida por `rag agenda`) | apple-mcp |
| Apple Notes | `mcp__apple-mcp__notes` | No integrada (probable *out-of-scope*: redundante con vault) | apple-mcp |
| Apple Messages | `mcp__apple-mcp__messages` | No integrada | apple-mcp |
| WhatsApp | `mcp__whatsapp__*` + SQLite bridge | **No indexada, pero el listener consume los mismos mensajes** | `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db` |
| Gmail (OAuth) | `mcp__claude_ai_Gmail__*` | No integrada | OAuth cloud (Claude harness-hosted) |
| Google Calendar | `mcp__claude_ai_Google_Calendar__*` | No integrada | OAuth cloud |

Dos observaciones tácticas:

- **Gmail/Calendar tienen dos caminos**: el MCP nativo de Apple (si el usuario tiene esas cuentas agregadas a Mail.app / Calendar.app de macOS, lo cual es el caso habitual) y el MCP de Google vía OAuth. El nativo es local-first puro (cumple el principio de `CLAUDE.md`: "LLMs, embeddings, TTS y STT corren en la máquina"). El OAuth es cloud-hosted por el harness de Claude — usable como fallback pero rompe el *local-only* del sistema. **Recomendación:** priorizar Apple-native en v1.
- **WhatsApp ya está semi-indexado**, en el sentido de que el listener (`whatsapp-listener/listener.ts`) puede escribir mensajes al vault via `/note`, `/ob`, o captura de voz → `00-Inbox/`. Pero eso es *curated ingestion*, no barrido automático. El cross-source corpus haría el barrido.

### 1.2 Scope propuesto para fase 1 (implementación)

| Prioridad | Fuente | Justificación |
|---|---|---|
| P0 | **Apple Mail** | Volumen medio, metadata rica (thread, sender, subject, date), alta señal semántica. |
| P0 | **Apple Calendar** | Volumen bajo, metadata estructurada, útil para resolver queries "cuándo fue..." / "qué hicimos en...". |
| P1 | **WhatsApp** | Volumen alto (150k+ mensajes estimados), señal diluida, pero alto valor práctico para decisiones informales. |
| P2 | **Apple Messages (SMS/iMessage)** | Extensión natural post-WhatsApp. Mismo patrón. |
| P3 | **Apple Reminders** | Ya consumido por `rag agenda` para lectura — duplicarlo como corpus no es obvio. Skip v1. |
| P3 | **Apple Notes** | Redundante con el vault para la mayoría de usuarios. Skip v1. |

La priorización asume que el usuario valida los casos de uso P0 antes de que el peer (o quien implemente fase 1) escriba código. Pregunta abierta #1.

---

## 2. Schema propuesto

### 2.1 Discriminador de fuente

El principio guía es **"el corpus es uno, el metadata discrimina"**. En lugar de múltiples colecciones paralelas (una per fuente), todo convive en una sola colección con un campo de metadata `source` que permite filtrado + ponderación en `retrieve()`.

Ventajas de una colección única:
- El cross-encoder reranker ve todos los candidatos en el mismo pool → ranking global coherente.
- `find_related` (que ya existe para wikilinks) puede atravesar fuentes: una nota del vault con `[[Juan]]` → emails de Juan → WhatsApps de Juan.
- Un sólo lugar para aplicar el cache BM25 (que hoy cold→warm baja de 341ms a 2ms).

Desventajas:
- Schema migration es costosa (todo el corpus se re-embedea si cambia el prefijo de `embed_text`).
- Rerank calibration se acopla: el threshold 0.015 que hoy es "irrelevante vs legítimo" en vault puro puede no transferirse. Ver §4.

### 2.2 Collection naming

- Propuesto: `_COLLECTION_BASE = "obsidian_corpus_v8"` (bump desde `obsidian_notes_v7`, rename para reflejar el alcance).
- URL sub-index queda paralelo: `obsidian_urls_v1` — los URLs son cross-source por diseño (pueden venir de notas, emails, WhatsApp — el sub-index ya es source-agnostic hoy).
- Multi-vault suffix (`sha256(VAULT_PATH)[:8]`) se preserva. Para fuentes cross-source, el suffix refleja *la identidad del usuario*, no del vault puntual — queda como está porque la identidad es el vault-owner.

### 2.3 Metadata común (mínimo, todas las fuentes)

Extiende lo que ya hay en `_index_single_file:4965`:

```python
common_meta = {
    # Discriminador
    "source": "vault" | "gmail" | "calendar" | "whatsapp" | "messages",

    # Identidad global (reemplaza "file" que era vault-specific)
    "doc_id": "<source>://<native_id>",  # ej: "gmail://thread:1890f3...", "whatsapp://chat:120363...@g.us/msg:ABC"
    "native_id": "<source-specific opaque id>",

    # Tiempo (unificado, UTC ISO 8601)
    "created_ts": "2026-04-15T12:22:10+00:00",   # cuándo ocurrió (email sent, event start, msg sent, note created)
    "modified_ts": "2026-04-15T14:30:00+00:00",  # última edición conocida (algunas fuentes no tienen; fallback = created_ts)
    "ingested_ts": "2026-04-15T15:00:00+00:00",  # cuándo entró al corpus (para churn)

    # Hash de contenido (gate de re-embed) — igual que hoy
    "hash": "<sha256 del contenido>",

    # Conexión con el grafo
    "outlinks": "",  # CSV de [[wikilinks]] si los hay (vault only por ahora; emails podrían parsear firma/referencias pero eso es fase 2)
    "tags": "",      # CSV; vault-native en v7; gmail → labels; calendar → calendar name
}
```

### 2.4 Metadata específica por fuente

**Vault** (sin cambios respecto a v7, movido bajo discriminador):
```python
{**common_meta, "source": "vault",
 "file": "02-Areas/...",   # ruta relativa al vault root
 "note": "<stem>", "folder": "<folder>",
 "parent": "<1200 chars de sección envolvente>"}
```

**Gmail / Apple Mail** (unificado — el MCP subyacente cambia, el schema no):
```python
{**common_meta, "source": "gmail",
 "thread_id": "...",       # conversación
 "message_id": "...",      # mensaje individual (native_id lo duplica para lookups)
 "subject": "...",
 "sender_email": "juan@ejemplo.com",
 "sender_name": "Juan Pérez",
 "to": "csv de destinatarios",
 "cc": "csv",
 "folder": "INBOX | [Gmail]/Sent | ...",   # el "folder" de Mail
 "parent": "<subject + first 1200 chars del thread, para contexto>"}
```

**Calendar**:
```python
{**common_meta, "source": "calendar",
 "event_id": "...",
 "title": "...",
 "start_ts": "2026-04-20T15:00:00-03:00",
 "end_ts": "2026-04-20T16:00:00-03:00",
 "location": "...",
 "attendees": "csv de emails",
 "calendar_name": "Trabajo | Personal | ...",
 "parent": "<title + descripción + agenda inline si existe>"}
```

**WhatsApp**:
```python
{**common_meta, "source": "whatsapp",
 "chat_jid": "120363...@g.us",
 "chat_name": "RagNet | Juan Pérez | ...",
 "chat_type": "group" | "dm",
 "message_id": "...",
 "sender": "<sender JID o nombre resuelto via contacts>",
 "is_from_me": True | False,
 "media_type": None | "image" | "audio" | "document",   # sin indexar media todavía, sólo el flag
 "parent": "<ventana de ±N mensajes del mismo chat para contexto conversacional, cap 1200 chars>"}
```

### 2.5 `embed_text` y `display_text` — el prefijo discriminador

Hoy: `embed_text = "[folder | title | area=... | #tags] [related=...] {chunk body}"`

Propuesta: anteponer un `[source=...]` tag al prefijo para que bge-m3 aprenda a condicionar el embedding por tipo de fuente. Esto es barato (el prefijo se prepend antes del embed, un token extra) y reversible (si empíricamente empeora el recall, se quita).

Ejemplos:
```
[source=gmail | from=juan@ejemplo.com | subject=migración DB] <body del email>
[source=whatsapp | chat=RagNet | from=Fer] <body del mensaje>
[source=calendar | title=Standup semanal | 2026-04-20] <descripción>
[source=vault | folder=02-Areas/Tech | title=Decisiones-DB | #migration] <body del chunk>
```

`display_text` (lo que ve reranker + LLM) se mantiene raw — sin prefijo — para que la evaluación cruzada sea sobre contenido, no metadata sintética.

### 2.6 Chunking per-source

| Fuente | Estrategia de chunk | Tamaño | Notas |
|---|---|---|---|
| Vault | Headers + blank lines, merge <150, split >800 (actual) | 150–800 chars | Sin cambios |
| Gmail | Por mensaje del thread; si mensaje >800 chars, split por párrafos | 150–800 chars | Drop quoted replies (`> `) y firmas por regex. `parent` = thread_subject + primeros 1200 chars |
| Calendar | 1 chunk por evento (corto) | <800 chars | No split. `parent` = self |
| WhatsApp | **Agrupamiento conversacional**: mensajes contiguos del mismo sender en <5min → 1 chunk; cambio de hablante o pausa >5min → split | 150–800 chars | Mensajes aislados <30 chars se mergean con vecinos. Ver §3.3 |

### 2.7 ID scheme

Hoy: `"<doc_id_prefix>::<i>"` donde `doc_id_prefix` es la ruta del archivo y `i` es el índice del chunk.

Propuesta cross-source:
```
"<source>://<native_id>::<chunk_index>"
```

Ejemplos:
- `vault://02-Areas/Tech/Decisiones-DB.md::3`
- `gmail://thread:1890f3abcd::0`
- `whatsapp://120363...@g.us/msg:ABC123::0`
- `calendar://event:XYZ789::0`

Esto permite:
- `col.get(where={"source": "gmail"})` para auditoría/debugging por fuente.
- Orphan cleanup por fuente: cuando se reindexa WhatsApp, sólo se miran IDs con prefijo `whatsapp://`.
- Lookups O(1) por native_id via metadata `where`.

---

## 3. Churn strategy

El vault de Obsidian tiene churn bajísimo: 521 notas, ~5 modificaciones/día. Gmail y WhatsApp están en otro régimen. Este es el problema de diseño más delicado.

### 3.1 Volumen estimado

| Fuente | Volumen (orden de magnitud) | Churn |
|---|---|---|
| Vault | ~500 notas, ~3.5k chunks | ~5 mods/día |
| Gmail | ~50k emails históricos, ~20k "relevantes" (no-transactional) | ~50 emails/día |
| Calendar | ~5k eventos históricos, ~3 eventos/día | ~3 altas/día |
| WhatsApp | ~150k mensajes (ya en SQLite), ~30-100 mensajes/día | Alto |

Un full-sync ingenuo de WhatsApp a 800 chars/chunk = ~150k chunks nuevos → ~1.5× el corpus actual, re-embedding 150k items a bge-m3 en MPS = ~2-3 horas. Viable para un one-shot inicial, inviable como operación diaria.

### 3.2 Retention

**Propuesta:** ventana rodante configurable por fuente.

```python
RETENTION = {
    "vault": None,          # sin expiración
    "calendar": None,       # eventos son chicos, se queda todo
    "gmail": 365,           # último año de emails
    "whatsapp": 180,        # últimos 6 meses de WA
    "messages": 180,
}
```

Expiración = el chunk queda en la colección hasta que:
1. El delta `(now - created_ts).days > RETENTION[source]`, y
2. El chunk no fue recuperado (en `retrieve()`) en los últimos 30 días → `last_retrieved_ts < now - 30d`.

La condición combinada evita que emails viejos pero activamente útiles sean borrados. Requiere logging de `retrieve()` por `doc_id` — que ya existe parcialmente via `queries.jsonl` (track #1 lo formaliza).

Pregunta abierta #2 (retention en §6).

### 3.3 Dedup y colapso conversacional

El caso más complicado es WhatsApp. Un hilo de 50 mensajes entre dos personas sobre un solo tema genera 50 chunks que compiten entre sí en el pool de rerank. Dos estrategias:

**Opción A — Chunks fine-grained + post-rerank collapse:**
- Chunkeo por mensaje individual (con agrupamiento de §2.6).
- En `retrieve()`, después del rerank, se colapsan chunks del mismo `chat_jid` en ventana ±30min → uno sólo sobrevive (el de mayor score).
- Ventaja: mantiene granularidad para queries específicas ("el mensaje donde Juan dijo X").
- Desventaja: rerank procesa chunks redundantes (desperdicia compute).

**Opción B — Pre-chunking por "conversación saliente":**
- Al ingestar WA, se agrupa por chat_jid + ventana temporal (ej: todos los mensajes de un chat en un mismo día = 1 "conversación"), y se chunkifica eso como documento unitario.
- Ventaja: rerank más eficiente, menos noise.
- Desventaja: pierde la granularidad de "el mensaje puntual".

**Recomendación:** empezar con **A** porque es reversible (decisión se aplica en `retrieve()`, no en indexing) y preserva la información. Si empíricamente el rerank se satura con WA, mover a B.

Gmail tiene el mismo problema en grado menor (threads): por thread, chunks del mismo thread pueden competir. Misma solución A.

### 3.4 Full-sync vs incremental

- **Vault** hoy: incremental hash-based (re-embed sólo si content hash cambió) + full rebuild opt-in (`rag index --reset`). Mantener.
- **Gmail / Calendar / WhatsApp** propuesta:
  - First-time: full-sync de la ventana de retención (ej: últimos 365 días para Gmail).
  - Steady-state: incremental vía cursor (`MAX(created_ts) WHERE source='gmail'` → fetch nuevos desde ese ts).
  - Re-index opcional: `rag index --source gmail --reset` re-embedea todo Gmail sin tocar el resto.

### 3.5 Hooks de ingesta

Propuesta de comandos nuevos (fase 1, no fase 0):
- `rag index --source <gmail|calendar|whatsapp>` → full-sync o incremental según estado.
- `rag index --all-sources` → todas las fuentes configuradas.
- `rag watch` (ya existe para vault) → extender para escuchar: Gmail poll cada 15min, WA via SQLite change notifier (el bridge ya lo escribe), Calendar EventKit subscription.

---

## 4. Retrieval considerations

Este es el cambio más subtle. El pipeline actual está calibrado para vault puro. Mezclar ontologías rompe supuestos que hoy funcionan implícitamente.

### 4.1 Reranker calibration

**Problema:** `CONFIDENCE_RERANK_MIN = 0.015` fue calibrado en vault (queries irrelevantes ~0.005-0.015, legítimas ≥0.02). Un email corto ("ok, dale, nos vemos mañana") puede scorear 0.01 contra la query "qué le dije a Juan sobre el viernes" — *debería* ser relevante pero cae bajo el threshold.

**Propuesta:**
- Re-calibrar con `queries.yaml` extendido: agregar 20-30 golden queries cross-source después de una ingesta parcial.
- Considerar **threshold per source**:
  ```python
  CONFIDENCE_RERANK_MIN = {"vault": 0.015, "gmail": 0.010, "whatsapp": 0.008, "calendar": 0.012}
  ```
- Esto rompe la elegancia del threshold global pero refleja que texto conversacional corto tiene densidad semántica distinta a prosa de nota Obsidian.

### 4.2 Source weighting en el scoring final

Post-rerank, aplicar un multiplicador por fuente que refleje la "trust" del usuario:

```python
SOURCE_WEIGHTS = {
    "vault": 1.00,       # alta: curado manualmente
    "calendar": 0.95,    # alta: estructurada, factual
    "gmail": 0.85,       # media: incluye ruido (newsletters, notifs)
    "whatsapp": 0.75,    # media-baja: conversacional, efímero
    "messages": 0.75,
}
final_score = rerank_score * SOURCE_WEIGHTS[source] * recency_boost(created_ts)
```

Con `recency_boost` (exponential decay, half-life por fuente):
- Vault: half-life 5 años (las notas viejas siguen siendo relevantes)
- Gmail: half-life 180 días
- WhatsApp: half-life 30 días (un WA de hace 2 años es casi ruido)
- Calendar: half-life 90 días (eventos viejos útiles sólo si explícitamente preguntados)

**Pregunta abierta #3**: estos pesos son *guess*, no datos. Calibrar requiere feedback del track #1.

### 4.3 Filtrado explícito por intent

Extender `classify_intent()` para detectar señales de fuente:
- "el email de Juan" → force `source=gmail`
- "el WhatsApp donde..." → `source=whatsapp`
- "la reunión del martes" → `source=calendar`

Ya hay precedente: `infer_filters()` detecta `#tag` y folder. Misma arquitectura.

### 4.4 Multi-query expansion

La expansión a 3 paráfrasis via qwen2.5:3b hoy asume lenguaje formal de notas. Para WhatsApp podría generar paráfrasis demasiado formales que no matchean el registro conversacional del corpus.

**Mitigación sugerida:** si el `source_hint` es WhatsApp/messages, condicionar el prompt de paráfrasis con un ejemplo de registro informal. Si no hay hint, paráfrasis neutrales (default).

### 4.5 Parent expansion cross-source

Hoy `parent` en vault = sección envolvente del header Markdown. En cross-source:
- Gmail `parent` = thread subject + primeros 1200 chars del thread (contexto conversacional).
- Calendar `parent` = descripción completa del evento + attendees (cap 1200).
- WhatsApp `parent` = ventana ±10 mensajes del mismo chat alrededor del chunk, cap 1200 chars.

Este campo ya existe en el schema de v7. Es el lugar natural para que el reranker vea contexto pese al chunk pequeño.

### 4.6 Citaciones en el render

`NOTE_LINK_RE` hoy matchea `[path.md]` y `[Label](path.md)`. Para cross-source:
- Vault: sin cambios — `[[wikilinks]]` clickeables.
- Gmail: `[email:juan@... 2026-04-12 subject]` → OSC 8 link a `message://<Message-Id>` (Apple Mail URL scheme abre el mensaje).
- Calendar: `[event:Standup 2026-04-20]` → URL `ical://...` o `x-apple-calendar://...`.
- WhatsApp: `[wa:RagNet 2026-04-15]` → sin URL scheme universal, queda como texto plano o link a la nota si `/ob` la guardó.

`verify_citations()` extender para reconocer el nuevo formato.

---

## 5. Privacy, cost, ética

### 5.1 Local-first

Todo el pipeline de embedding + LLM ya corre local (bge-m3, bge-reranker, command-r vía Ollama). El único punto que potencialmente rompe local-first son los **MCPs Gmail/Calendar vía OAuth** (cloud-hosted por el harness de Claude). La alternativa: apple-mcp usa EventKit + Mail.app locales → datos nunca salen del host. **Recomendación fuerte:** usar apple-mcp, no el OAuth MCP, para P0.

### 5.2 Encriptación en reposo

ChromaDB embebe los `documents` en sqlite cleartext. Para vault eso es aceptable (el vault ya es cleartext en iCloud). Para WhatsApp/Gmail es más sensible — el corpus indexado *es* un corpus de mensajes privados.

**Opciones:**
1. No encriptar (aceptar riesgo, confiar en FileVault del host).
2. Encriptar la colección Chroma a nivel FS (sparse bundle cifrado, mount on demand).
3. Persistir sólo embeddings + metadata, re-fetch el documento original por `native_id` al momento de display (caro en latencia).

**Recomendación:** (1) si el host está en FileVault (default macOS moderno). El usuario valida esto.

### 5.3 Opt-out por fuente/filtro

Algunos emails/chats no deben indexarse nunca:
- Emails con labels `banking`, `financial`, `2FA` → skip.
- Chats de WhatsApp con etiqueta "private" o números sin nombre en contactos → skip (opt-in por chat).

Propuesta: archivo de config `~/.local/share/obsidian-rag/cross-source.yaml`:
```yaml
gmail:
  exclude_labels: [banking, 2fa]
  exclude_senders: ["*@bank.com"]
whatsapp:
  exclude_chats: ["+5491112345678"]
  exclude_keywords_in_chat_name: [bank, personal, confidential]
```

---

## 6. Open questions para el user

Antes de pasar a fase 1 (implementación), el usuario decide:

1. **Priorización de fuentes P0.** ¿Gmail + Calendar primero, o WhatsApp (que ya vive en SQLite local y es "lo más cercano")? La recomendación técnica: Gmail+Calendar (más limpio, menor volumen, MCPs ya robustos via apple-mcp).

2. **Retention windows.** ¿365 días Gmail, 180 WhatsApp son razonables para tus hábitos? Si querés todo desde siempre → bump a `None` pero aceptar corpus ~200k chunks (re-embed inicial = ~4h).

3. **Source weights.** ¿Estás cómodo con que WhatsApp rankée 0.75× vault? ¿O confiás en tus WA tanto como en tus notas curadas?

4. **Encriptación.** ¿FileVault está activo en este host? Si no, ¿querés ChromaDB en un sparse bundle cifrado?

5. **Opt-out.** ¿Hay labels de Gmail / chats de WhatsApp que explícitamente NO querés indexar? (El default propuesto es indexar todo).

6. **OAuth vs apple-mcp para Gmail/Calendar.** Confirmar preferencia por apple-mcp (local) sobre el MCP OAuth del harness (cloud).

7. **Colección única vs per-source.** El diseño propone única. Alternativa: colecciones separadas con rerank cruzado en memoria. ¿Preferencia?

8. **Timing.** Fase 1 (implementación) depende del track #1 (feedback loop) para calibrar pesos/threshold. ¿OK encolar fase 1 detrás de track #1, o arrancar en paralelo con pesos hardcoded y calibrar after?

---

## 7. Baselines a preservar

Cualquier implementación de fase 1 debe mantener (o re-baselinear explícitamente):

- **Singles eval** (`queries.yaml`): `hit@5 90.48% · MRR 0.786 · recall@5 90.48%` (v7 actual).
- **Chains eval**: `hit@5 75.00% · MRR 0.656 · chain_success 50.00%`.
- **Query latency p50**: <2.5s.
- **Index incremental** (cuando el vault cambia 5 notas): <10s.
- **BM25 cache warm-up**: 341ms cold, 2ms warm.

El eval harness tiene que extenderse con queries cross-source *antes* de mergear fase 1, para detectar regresiones de mixing. Concretamente: 10-15 queries donde la respuesta correcta vive en Gmail/Calendar/WA, y 5-10 queries del set actual que deben *no* bajar de accuracy con el corpus extendido.

---

## 8. Anti-scope (lo que explícitamente no hace este track)

- No reemplaza `rag prep` ni `rag morning` — esos consumen fuentes en *read-time* via apple-mcp. Seguirán existiendo.
- No toca el ambient agent (`_ambient_config`).
- No cambia el formato de `queries.jsonl` (eso es track #1).
- No toca el MCP server (`mcp_server.py`) — las tools `rag_query` / `rag_read_note` seguirán funcionando igual (filtrarán automáticamente por `source=vault` si el caller no especifica, para backward-compat).
- No indexa adjuntos binarios (imágenes WA, PDFs en Gmail). Eso es fase 2 (requiere OCR pipeline unificado).

---

## 9. Siguientes pasos (fuera de fase 0)

Después de que el usuario responda §6:

1. **Fase 1.a** — un spike de 3-5 días: ingestar Apple Calendar (volumen bajo, schema simple) como *proof of concept* del discriminador de fuente y el source weighting. Un comando `rag index --source calendar`. Re-correr eval. Medir impacto en baselines.
2. **Fase 1.b** — si 1.a valida el approach: Apple Mail. Mayor volumen, mismo patrón.
3. **Fase 1.c** — WhatsApp. Ultimo por complejidad de dedup conversacional.
4. **Fase 1.d** — apagado del workaround actual (el listener con `/note` deja de ser necesario para capturas reactive de WA; el corpus las tiene por barrido).

---

*Fin del documento. Feedback en /Users/fer/repositories/obsidian-rag/docs/design-cross-source-corpus.md.*
