# Cómo funciona por dentro

Sin tecnicismos innecesarios. Diagramas con palabras simples de lo que pasa cuando hacés una pregunta o indexás una nota.

## Las piezas grandes

```
┌───────────────────────────────────────────────────────────────────────┐
│                          TU MAC (todo local)                          │
│                                                                       │
│  ┌──────────────┐         ┌────────────────┐       ┌───────────────┐  │
│  │   VAULT      │ ──save─▶│  rag watch     │──────▶│    ÍNDICE     │  │
│  │  Obsidian    │         │ (background)   │       │ (sqlite-vec)  │  │
│  │  tus .md     │         │ re-indexa al   │       │  ~/.local/    │  │
│  └──────────────┘         │  guardar       │       │  share/       │  │
│         ▲                 └────────────────┘       │  obsidian-rag │  │
│         │                                          └───────┬───────┘  │
│         │ leé respuesta + clickeá links                    │          │
│         │                                                  │          │
│  ┌──────┴─────────┐          ┌──────────────┐       ┌──────▼───────┐  │
│  │  rag query     │─────────▶│   OLLAMA     │◀──────│  retrieve    │  │
│  │  rag chat      │          │  localhost   │       │  pipeline    │  │
│  │  rag do …      │          │  :11434      │       │              │  │
│  └────────────────┘          └──────────────┘       └──────────────┘  │
│                                      ▲                                │
│                                      │                                │
│                              ┌───────┴────────┐                       │
│                              │  reranker      │                       │
│                              │  bge-reranker  │                       │
│                              │  (en memoria)  │                       │
│                              └────────────────┘                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Ollama** es un programa que corre modelos de lenguaje en tu Mac. Lo usa el sistema para tres cosas:
- Convertir texto a embeddings (números que representan el significado).
- Tareas rápidas con un modelo chico (helper).
- Generar las respuestas con un modelo más grande (chat).

**sqlite-vec** es la base de datos donde se guardan los chunks de tus notas con sus embeddings. Es un archivo en `~/.local/share/obsidian-rag/ragvec/ragvec.db`.

**El reranker** es un modelo aparte que vive en la memoria de tu Mac. Toma una lista de candidatos y los re-ordena por "qué tan relacionados están con la pregunta". No pasa por Ollama.

---

## Qué pasa cuando indexás una nota

Pensalo como "cortar la nota en pedacitos, resumir, convertir a números, guardar".

```
 nota.md (1 archivo)
      │
      ▼
 ┌─────────────────────┐
 │ leer el archivo     │
 │ + frontmatter tags  │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │ cambió desde la     │──── igual ──▶ saltear (no hace nada)
 │ última vez? (hash)  │
 └─────────┬───────────┘
           │ sí, distinta
           ▼
 ┌─────────────────────┐
 │ (opcional) ¿alguna  │──sí─▶ anota "contradicts:"
 │ otra nota contradice│       en el frontmatter
 │ a esta?             │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │ partir en pedacitos │      cada pedacito tiene
 │ (150-800 caracteres)│ ───▶ entre 1-5 párrafos,
 │ por secciones       │      respeta los headers
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │ para cada pedacito: │
 │ embeber con bge-m3  │ ───▶ 1024 números por chunk
 │ (vía Ollama)        │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │ guardar en sqlite   │
 │ (chunks + URLs      │
 │  aparte)            │
 └─────────────────────┘
```

### Por qué se parten las notas

Una nota entera es muy grande para embeber de una. El sistema las corta en "chunks" más chicos (150-800 caracteres) para que cada pedazo tenga un tema concreto y se pueda buscar mejor.

### Qué es un embedding

Un embedding es una lista de 1024 números que representa el significado de un texto. Dos textos con significado parecido tienen embeddings parecidos (cosine similarity alta). Así es como el sistema "busca por significado" en vez de "buscar por palabra exacta".

---

## Qué pasa cuando hacés una pregunta

Pensalo como "entender la pregunta, buscar candidatos, ordenarlos por relevancia, pasarlos al LLM, responder".

```
 "¿qué sé sobre X?"
        │
        ▼
 ┌──────────────────────┐
 │ ¿qué tipo de         │  count  → contar cosas sin LLM
 │ pregunta es?         │  list   → listar notas sin LLM
 │ (classify_intent)    │  recent → ordenar por fecha
 └──────┬───────────────┘  agenda → calendar/reminders
        │ semantic         entity → lookup por persona/cosa
        ▼ (lo más común)
 ┌──────────────────────┐
 │ ¿hay filtros obvios  │
 │ en la pregunta?      │  "mis notas de abril"  →  --since 1m
 │ (infer_filters)      │  "en coaching"         →  --tag coaching
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ embeber la pregunta  │ ───▶ 1024 números que representan
 │ (bge-m3)             │      el significado de la pregunta
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐      cosine similarity entre
 │ buscar los N chunks  │ ───▶ la pregunta y cada chunk
 │ más parecidos en el  │      del vault
 │ índice (sqlite-vec)  │
 │ + matching léxico    │
 │ por keywords (BM25)  │
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ mergear ambos        │ ───▶ dedupear, expandir a la
 │ resultados y limpiar │      sección padre de cada chunk
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ rerankear con el     │ ───▶ un modelo más preciso
 │ cross-encoder        │      re-ordena los top 15
 │ (bge-reranker-v2-m3) │      por relevancia real
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ agregar vecinos del  │ ───▶ las notas que tienen
 │ grafo (wikilinks)    │      [[links]] a las top-3
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ ¿la confianza es     │──baja──▶ retrieval profundo:
 │ suficiente?          │          el LLM hace sub-preguntas
 └──────┬───────────────┘          y busca más rondas
        │ sí
        ▼
 ┌──────────────────────┐
 │ pasar los top-K al   │ ───▶ streaming con command-r
 │ LLM grande para que  │      (o qwen2.5:7b según stack)
 │ arme la respuesta    │
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │ verificar que las    │ ───▶ si el LLM inventó un path
 │ citas apunten a      │      que no se recuperó, lo
 │ archivos que existan │      corrige en una segunda pasada
 └──────┬───────────────┘
        │
        ▼
 respuesta + sources + links clickeables
```

### Las 5 etapas clave en 1 línea

1. **Entender** la pregunta (intent + filtros).
2. **Buscar** candidatos en el índice (vecinos semánticos + keywords).
3. **Re-ordenar** con un modelo más preciso (reranker).
4. **Expandir** con vecinos del grafo (wikilinks).
5. **Generar** la respuesta con el LLM + verificar citas.

---

## El radar de contradicciones

Detecta cuando una nota se contradice con otra del vault. Tres fases:

```
 FASE 1 — al hacer una pregunta (query-time)
 ──────────────────────────────────────────
   rag query "X" --counter
      │
      ▼
   después de la respuesta normal, el LLM busca en
   el vault chunks que CONTRADIGAN la respuesta
      │
      ▼
   te los muestra en un bloque "⚡ counter-evidence"


 FASE 2 — al guardar una nota (index-time)
 ──────────────────────────────────────────
   guardás una nota nueva/modificada
      │
      ▼
   el sistema busca si hay otras notas que la contradigan
      │
      ▼
   si encuentra: escribe `contradicts: [otra.md]` en el frontmatter
   + loguea a contradictions.jsonl


 FASE 3 — en el brief semanal
 ──────────────────────────────────────────
   domingo a las 22:00 corre `rag digest` automáticamente
      │
      ▼
   junta contradicciones de las fases 1 y 2
      │
      ▼
   command-r arma un brief narrativo sobre lo que cambió
   de opinión en la semana → escribe a 04-Archive/99-obsidian-system/99-Claude/reviews/YYYY-WNN.md
```

---

## El ranker-vivo

"Ranker-vivo" es la mecánica por la que el sistema aprende de vos con el tiempo y va mejorando el orden de los resultados sin que lo entrenes de cero.

### Qué es

El reranker (ese modelo que reordena candidatos por relevancia) tiene unos pesos configurables. Por default viene con pesos base. Pero a medida que usás el sistema, se van registrando "señales" sobre qué resultados fueron útiles y cuáles no — y periódicamente (`rag tune` automático, diario a las 3:30am) re-calibra los pesos con esas señales.

### Las señales que se acumulan

Cada interacción con una respuesta del RAG genera un evento en el archivo `~/.local/share/obsidian-rag/behavior.jsonl` (y la tabla SQL `rag_behavior`):

```
┌──────────────────────────────────────────────────────────────────────┐
│                 SEÑALES POSITIVAS (suman al CTR)                     │
├──────────────────────────────────────────────────────────────────────┤
│  copy        │ copiaste texto de la respuesta (Cmd+C ≥20 chars)      │
│  open        │ abriste una nota citada (clickeaste un link)          │
│  save        │ guardaste la respuesta como nota                      │
│  kept        │ aceptaste una propuesta del ambient agent             │
│  positive_   │ otras señales positivas implícitas                    │
│    implicit  │                                                       │
├──────────────────────────────────────────────────────────────────────┤
│                 SEÑALES NEGATIVAS (restan)                           │
├──────────────────────────────────────────────────────────────────────┤
│  negative_   │ señales negativas implícitas                          │
│    implicit  │                                                       │
│  deleted     │ borraste una propuesta del ambient agent              │
├──────────────────────────────────────────────────────────────────────┤
│                 DENOMINATOR (el "mostraste")                         │
├──────────────────────────────────────────────────────────────────────┤
│  impression  │ un chunk apareció en los resultados mostrados al user │
└──────────────────────────────────────────────────────────────────────┘
```

También hay señales **explícitas** que vos disparás a mano:
- `rag rate +` o `rag rate -` (o `👍` / `👎` en chat/web)
- `rag fix <path>` — decile al sistema cuál era la nota correcta

### El loop completo

```
  hacés una pregunta
       │
       ▼
  retrieve te muestra 5-8 chunks  ─────▶  evento IMPRESSION × N
       │                                  (denominator)
       │
       ▼
  vos hacés algo:
  ┌─── clickeás una cita        ─────▶  evento OPEN
  ├─── copiás texto (Cmd+C)     ─────▶  evento COPY
  ├─── /save <título>           ─────▶  evento SAVE
  ├─── 👍 / 👎                  ─────▶  evento POSITIVE / NEGATIVE
  └─── nada (ignoraste)         ─────▶  no hay evento, pero hay impression
       │
       ▼
  cada evento va a behavior.jsonl + tabla rag_behavior
       │
       ▼
  diario a las 3:30am:
  `rag tune --apply` toma los eventos + queries.yaml + feedback.jsonl
  y corre un random search sobre los pesos del ranker — el winner se
  persiste a ~/.config/obsidian-rag/ranker.json
       │
       ▼
  próxima query usa los pesos nuevos → mejor ranking
```

### Cómo verlo funcionar

El dashboard web (`/dashboard`) tiene un panel **Señales al ranker-vivo · últimos N días** con los counts por tipo de evento. Si está vacío, significa que todavía no hay tráfico suficiente. Podés "sembrar" el loop:

1. Hacer una query en chat o web.
2. Copiar al menos 20 caracteres de la respuesta con Cmd+C.
3. Recargar `/dashboard` → debería aparecer `copy 1 · web 1`.

### Rollback

Si un tune nuevo empeoró las cosas:

```bash
rag tune --rollback          # restaura el backup más reciente de ranker.json
```

---

## Los modelos que usa (y para qué)

```
┌─────────────────────────────────────────────────────────────────┐
│                       OLLAMA (localhost:11434)                  │
│                                                                 │
│   ┌──────────────┐  convertir texto a embeddings                │
│   │   bge-m3     │  (1024 números por chunk / query)            │
│   └──────────────┘                                              │
│                                                                 │
│   ┌──────────────┐  tareas rápidas y baratas:                   │
│   │  qwen2.5:3b  │  reformular preguntas, generar variantes,    │
│   │  (helper)    │  autotag, clasificar intent, sugerir filtros │
│   └──────────────┘                                              │
│                                                                 │
│   ┌──────────────┐  el "grande": genera respuestas,             │
│   │  qwen2.5:7b  │  escribe briefs, juzga contradicciones,      │
│   │  o command-r │  agent loop con tools                        │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   sentence-transformers (en memoria)            │
│                                                                 │
│   ┌────────────────────────┐  re-ordenar los candidatos         │
│   │ bge-reranker-v2-m3     │  después de la primera búsqueda    │
│   │ (MPS + fp32)           │  por relevancia precisa            │
│   └────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dónde vive cada cosa en tu Mac

```
~/.local/share/obsidian-rag/
 │
 ├── ragvec/
 │   └── ragvec.db              ◀── la base sqlite-vec
 │                                  (chunks + embeddings + URLs)
 │
 ├── sessions/
 │   └── <id>.json              ◀── conversaciones guardadas
 │
 ├── last_session               ◀── pointer a la última usada
 ├── queries.jsonl              ◀── log de cada query/chat
 ├── contradictions.jsonl       ◀── contradicciones detectadas
 ├── behavior.jsonl             ◀── eventos del ranker-vivo
 │                                   (copy/open/save/kept/impression/...)
 ├── feedback.jsonl             ◀── 👍/👎 y correcciones
 ├── ambient.json               ◀── config del Ambient Agent
 ├── ambient.jsonl              ◀── eventos del Ambient
 ├── surface.jsonl              ◀── puentes propuestos
 ├── filing.jsonl               ◀── decisiones de `rag file`
 ├── watch.log                  ◀── logs del servicio watch
 ├── morning.log / .error.log
 └── digest.log  / .error.log

~/.config/obsidian-rag/
 ├── vaults.json                ◀── registry multi-vault
 ├── ranker.json                ◀── pesos del ranker tuneado
 └── spotify_token.json         ◀── (opcional) OAuth de Spotify

~/Library/LaunchAgents/
 └── com.fer.obsidian-rag-*.plist   ◀── los 11 servicios de fondo

tu-vault/                       ◀── tu vault de Obsidian
 ├── 00-Inbox/                  ◀── donde caen las capturas
 ├── 01-Projects/               ◀── PARA
 ├── 02-Areas/
 ├── 03-Resources/
 ├── 04-Archive/
 └── 04-Archive/99-obsidian-system/99-Claude/reviews/                ◀── donde caen morning/today/digest
```

---

## Por qué está todo local

- **Privacidad**: tus notas nunca salen de tu Mac. Solo pasan por Ollama (que también corre local).
- **Velocidad**: sin red, sin latencia de API. Las queries son más rápidas.
- **Sin cuotas**: no pagás por token. Podés hacer las preguntas que quieras.
- **Off-line**: funciona sin internet (excepto integraciones externas como Gmail o Calendar).

La **única excepción** son algunas integraciones opcionales (Gmail, Google Calendar) que usan OAuth. WhatsApp va via el bridge local; Reminders y Calendar via Apple EventKit local.

---

## Dónde ver los diagramas oficiales

Los diagramas "serios" (mermaid + SVG) viven en [`./diagrams/`](./diagrams/):

- [system-overview](./diagrams/system-overview.svg) — cómo se conectan todas las piezas
- [retrieval](./diagrams/retrieval.svg) — pipeline de búsqueda detallado
- [indexing](./diagrams/indexing.svg) — pipeline de indexing detallado
- [ollama-interactions](./diagrams/ollama-interactions.svg) — qué modelo se usa para qué
- [services-topology](./diagrams/services-topology.svg) — launchd + CLIs + bots
- [contradiction-radar](./diagrams/contradiction-radar.svg) — las 3 fases del radar
- [inbox-triage](./diagrams/inbox-triage.svg) — composición de `rag inbox`

Para regenerar los SVGs:

```bash
cd docs/diagrams
for f in *.mmd; do
  npx -y @mermaid-js/mermaid-cli -i "$f" -o "${f%.mmd}.svg" -t dark -b transparent
done
```
