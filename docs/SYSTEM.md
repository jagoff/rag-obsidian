# obsidian-rag — System documentation

Documento canónico de cómo funciona el sistema completo: qué hay, cómo se comunica, qué pasa cuando pasa cada cosa, y por qué está diseñado así.

**Lector objetivo**: vos dentro de 6 meses, otro Claude continuando el proyecto, o alguien migrando el sistema a otra Mac. Asume que ya leíste `README.md` (comandos + flags) y opcionalmente `CLAUDE.md` (decisiones de diseño).

Complementa pero no duplica:
- **[CLAUDE.md](../CLAUDE.md)** — decisiones arquitecturales, trade-offs, findings empíricos.
- **[README.md](../README.md)** — referencia operativa, comandos, recetas.
- **Este doc** — modelo conceptual, data flows end-to-end, lifecycle de una nota.

---

## Tabla de contenidos

1. [Modelo conceptual](#1-modelo-conceptual)
2. [Mapa de componentes](#2-mapa-de-componentes)
3. [Data flows principales](#3-data-flows-principales)
4. [Lifecycle de una nota](#4-lifecycle-de-una-nota)
5. [State management](#5-state-management)
6. [Superficies de uso](#6-superficies-de-uso)
7. [Concurrencia y fallo](#7-concurrencia-y-fallo)
8. [Performance envelope](#8-performance-envelope)
9. [Extensión: cómo agregar una feature](#9-extensión-cómo-agregar-una-feature)

---

## 1. Modelo conceptual

El sistema trata al **vault de Obsidian** como **fuente de verdad inmutable** y construye capas de análisis encima sin reescribirlo (salvo escrituras explícitas y auditables como `rag wikilinks --apply`, `rag inbox --apply`, o el flag `contradicts:` en frontmatter).

Tres niveles conceptuales:

```
┌─────────────────────────────────────────────────┐
│  NIVEL 3: Aplicaciones                          │
│  rag chat · rag prep · rag morning · rag do    │
│  WhatsApp listener · MCP tools · weekly digest  │
│  ← aquí el usuario HABLA con el vault           │
├─────────────────────────────────────────────────┤
│  NIVEL 2: Primitivas composables                │
│  retrieve · find_related · find_urls            │
│  find_contradictions · find_wikilink_           │
│  suggestions · find_dead_notes · find_          │
│  duplicate_notes · capture_note · reformulate   │
│  ← pura función, sin side-effects de negocio    │
├─────────────────────────────────────────────────┤
│  NIVEL 1: Índices                               │
│  obsidian_notes_v11 (chunks + metadata)         │
│  obsidian_urls_v1 (URL + contexto)              │
│  title_to_paths / outlinks / backlinks          │
│  queries.jsonl · contradictions.jsonl           │
│  ← estado derivado del vault (regenerable)      │
├─────────────────────────────────────────────────┤
│  NIVEL 0: Vault de Obsidian (fuente)            │
│  ~/Library/Mobile Documents/                    │
│  iCloud~md~obsidian/Documents/Notes/            │
└─────────────────────────────────────────────────┘
```

**Invariante fundamental**: el Nivel 1 se puede tirar y reconstruir desde el Nivel 0 en ~30 min (`rag index --reset`). El Nivel 2 es puro código. El Nivel 3 es la cara pública.

---

## 2. Mapa de componentes

```mermaid
graph TB
    subgraph "User surfaces"
        CLI[rag CLI]
        MCP["obsidian-rag-mcp<br/>(stdio)"]
        WA["WhatsApp listener<br/>(Whisper → /rag · /note · /read · /followup · /today)"]
        LAUNCHD["launchd services<br/>watch · morning · digest"]
    end

    subgraph "Python core (rag.py + mcp_server.py)"
        Retrieve[retrieve pipeline]
        Index[indexing pipeline]
        Prims["primitivas:<br/>find_* · capture_* · apply_*"]
    end

    subgraph "Storage local"
        Vec[("sqlite-vec (ragvec.db)<br/>notes_v11 + urls_v1")]
        JSONL["jsonl logs<br/>queries + contradictions"]
        Sess["sessions/*.json"]
        Plists["launchd plists"]
    end

    subgraph "Modelos (Ollama localhost:11434)"
        Embed["bge-m3<br/>(embeddings)"]
        Helper["qwen2.5:3b<br/>(paraphrase, autotag,<br/>reformulate)"]
        Chat["command-r<br/>(answers, judgment,<br/>drafting)"]
    end

    Reranker["bge-reranker-v2-m3<br/>(MPS+fp16)"]

    subgraph "Vault (iCloud)"
        Notes["*.md notes"]
    end

    CLI --> Retrieve & Index & Prims
    MCP --> Retrieve & Prims
    WA --> CLI
    LAUNCHD --> CLI

    Index --> Embed
    Index --> Vec
    Retrieve --> Embed & Helper & Chat & Reranker
    Retrieve --> Vec
    Prims --> Chat & Helper

    Retrieve --> Sess & JSONL
    Index --> JSONL

    Notes --> Index
    Prims -.-> Notes

    classDef external fill:#f9f,stroke:#333
    class Notes,Embed,Helper,Chat,Reranker external
```

### Responsabilidades

| Componente | Responsabilidad | Lo que NUNCA hace |
|---|---|---|
| `rag.py` | CLI + todas las primitivas + pipelines | Llamadas de red externas (solo localhost) |
| `mcp_server.py` | Wrapper fino sobre las primitivas para Claude Code | Lógica nueva — delega a `rag.py` |
| sqlite-vec (`ragvec.db`) | Almacén vectorial persistente | Ser fuente de verdad — reconstructible del vault |
| Ollama | Servir embeddings + LLMs locales | Persistir estado de negocio |
| WhatsApp listener | Adapter texto/voz → `rag` CLI vía bridge local | Mantener historial del RAG (usa `rag` para eso) |
| launchd | Mantener `watch`/`morning`/`digest` vivos | Saber qué hace cada comando |

---

## 3. Data flows principales

### 3.1 Indexing (cambio de nota → estado)

Dos triggers: `rag index` (bulk) o `rag watch` (watchdog, continuo). Ambos llaman a `_index_single_file` por nota.

```mermaid
sequenceDiagram
    participant FS as filesystem<br/>(Obsidian save)
    participant Watch as rag watch<br/>(launchd)
    participant ISF as _index_single_file
    participant CR as find_contradictions_for_note
    participant Chat as command-r
    participant Vec as sqlite-vec
    participant URLs as URL sub-index
    participant FM as frontmatter
    participant Log as contradictions.jsonl

    FS->>Watch: FileModifiedEvent
    Watch->>ISF: path
    ISF->>Vec: get(where={file: path})<br/>→ existing chunks + hash
    Note over ISF: hash diff? → si no cambió, return "skipped"
    ISF->>ISF: parse_frontmatter, clean_md, extract_wikilinks
    alt skip_contradict == False
        ISF->>CR: body, exclude={self}
        CR->>Vec: embed + rerank
        CR->>Chat: JSON prompt estricto
        Chat-->>CR: contradictions: [...]
        CR->>FM: write `contradicts: [...]`
        CR->>Log: append entry
        CR-->>ISF: updated raw (si FM cambió)
    end
    ISF->>Vec: delete old chunks + insert new
    ISF->>URLs: _index_urls (extract + embed context)
    URLs->>Vec: upsert urls rows
    ISF->>ISF: _invalidate_corpus_cache
    opt path en 00-Inbox/
        ISF->>Ambient: _ambient_hook<br/>(wikilinks auto + dupes/related ping)
    end
```

**Guardrail clave**: el chequeo de contradicción solo corre incremental. `rag index --reset` lo skipea automáticamente (sería O(n²) calls a command-r).

### 3.2 Retrieval (pregunta → respuesta)

```mermaid
sequenceDiagram
    participant User
    participant Cmd as rag query/chat
    participant Sess as session
    participant H as helper (qwen2.5:3b)
    participant R as retrieve()
    participant Vec as sqlite-vec
    participant BM25 as BM25
    participant Rer as reranker (MPS)
    participant LLM as command-r
    participant Log as queries.jsonl

    User->>Cmd: question
    opt session_id presente
        Cmd->>Sess: load turns
        Sess->>H: reformulate_query(q, history)
        H-->>Cmd: effective_q
    end
    Cmd->>R: effective_q
    R->>R: classify_intent
    alt intent == count/list/recent
        R->>Chr: metadata scan
        Chr-->>User: resultado directo (NO LLM)
    else intent == semantic
        R->>R: infer_filters (auto folder/tag)
        R->>H: expand_queries (3 paraphrases)
        H-->>R: variants
        R->>Chr: batch embed(variants)
        loop por variante
            R->>Chr: sem query
            R->>BM25: BM25 search
            R->>R: RRF merge
        end
        R->>Chr: fetch candidates + expand_to_parent
        R->>Rer: cross-encoder rerank (top-k)
        Rer-->>R: scored chunks
        alt top_score < 0.015 AND not --force
            R-->>User: "no tengo esa información"<br/>(LLM NO se llama)
            R->>Log: gated_low_confidence: true
        else confidence OK
            R->>LLM: system + history + context + q
            LLM-->>User: streamed answer
            R->>R: verify_citations
            opt --counter
                R->>LLM: find_contradictions<br/>(answer vs otros chunks)
            end
        end
    end
    Cmd->>Log: append query event
    opt session_id
        Cmd->>Sess: append turn
    end
```

### 3.3 Contradiction Radar (3 fases)

```mermaid
graph LR
    subgraph "Phase 1: query-time"
        direction TB
        Q1[rag query/chat<br/>--counter] --> A1[answer generated]
        A1 --> F1[find_contradictions]
        F1 --> B1["⚡ Counter-evidence<br/>block"]
        F1 --> L1[log contradictions<br/>to queries.jsonl]
    end

    subgraph "Phase 2: index-time"
        direction TB
        E2[note edit]
        E2 --> H2[_index_single_file]
        H2 --> F2[find_contradictions<br/>_for_note]
        F2 --> FM2["contradicts: [...]<br/>in frontmatter"]
        F2 --> L2[contradictions.jsonl]
    end

    subgraph "Phase 3: weekly digest"
        direction TB
        C3["cron sunday 22:00<br/>(launchd)"] --> G3[_collect_week_evidence]
        FM2 --> G3
        L2 --> G3
        L1 --> G3
        G3 --> D3[command-r drafting]
        D3 --> N3["05-Reviews/YYYY-WNN.md<br/>auto-indexed"]
    end

    L1 -. "al próximo indexing" .-> FM2
    FM2 -. feed next .-> G3
```

**Decisión clave**: las 3 fases usan `command-r` como judge (NO el helper). qwen2.5:3b dio false positives + JSON malformado — command-r hugs source text y responde parseable. Ver `CLAUDE.md` finding #2.

### 3.3.5 Ambient Agent (hook reactivo de Inbox)

Se dispara al final de `_index_single_file` cuando el path está en `00-Inbox/` y el hash cambió. Composición de primitivas existentes, **sin LLM extra**:

```mermaid
sequenceDiagram
    participant ISF as _index_single_file<br/>(después de indexar)
    participant H as _ambient_hook
    participant Cfg as ambient.json
    participant St as ambient_state.jsonl
    participant WL as find_wikilink_<br/>suggestions
    participant Ap as apply_wikilink_<br/>suggestions
    participant DP as find_near_<br/>duplicates_for
    participant RL as find_related
    participant WA as WhatsApp bridge<br/>(localhost:8080)

    ISF->>H: (col, path, doc_id, hash)
    H->>Cfg: read
    alt no config o enabled=false
        H-->>ISF: no-op
    end
    H->>St: check (path+hash ya<br/>analizado <5min?)
    alt dedup hit
        H-->>ISF: skip
    end
    H->>H: parse_frontmatter<br/>(chequea ambient:skip /<br/>type: morning|digest|prep)
    H->>WL: get suggestions
    WL-->>H: [{title, target, offset}]
    H->>Ap: auto-apply<br/>(regex determinística)
    H->>DP: threshold=0.85
    DP-->>H: [{note, similarity}]
    H->>RL: self_meta
    RL-->>H: [(meta, score, reason)]
    H->>H: build U+200B prefix + compact msg
    alt hay findings
        H->>WA: POST /api/send<br/>(urllib, 10s timeout)
    end
    H->>St: record analyzed_at
    H->>H: log event to ambient.jsonl
```

Ping de ejemplo en el self-chat *RagNet* de WhatsApp:

```
🤖 Ambient: [[caminata-ideas]]
🔗 Linkeé 2: Ikigai, Moka
⚠ Posibles duplicados:
  · [[Ideas - caminata 2026-02]]  sim 0.88
📎 Relacionadas:
  · [[Coaching - Propósito]]  ×8 ↔#
  · [[Músicos y ikigai]]  ×4 #
```

**Desacople clave**: rag.py hace POST directo al bridge local `http://localhost:8080/api/send` (urllib, 10s timeout). El mensaje arranca con U+200B (anti-loop — el listener lo ignora para no procesar sus propios outputs como queries entrantes). NO depende del listener estando up: si el listener muere, el análisis corre y queda en `ambient.jsonl`; solo se pierde el ping hasta que el bridge lo entregue.

### 3.4 Capture → Morning → Digest (el loop diario/semanal)

```mermaid
graph TB
    subgraph "Captura continua (todo el día)"
        U1[usuario CLI] -->|rag capture| CAP[capture_note]
        WA1["WhatsApp /note"] -->|stdin| CAP
        WA2["WhatsApp voz (default)"] -->|Whisper → stdin| CAP
        CAP --> IBX["00-Inbox/<br/>YYYY-MM-DD-HHMM-*.md"]
        IBX --> IDX1[auto-index]
    end

    subgraph "Morning brief (lun-vie 7:00 launchd)"
        CRON1[cron mañana] --> MORN[rag morning]
        IBX --> MORN
        MOD["notas modificadas<br/>últimas 36h"] --> MORN
        TODOS["frontmatter todo:/due:"] --> MORN
        L1["contradictions.jsonl<br/>(últimas 36h)"] --> MORN
        LC["queries.jsonl<br/>low-conf últimas 36h"] --> MORN
        MORN --> DRAFT1[command-r draft<br/>120-280 palabras]
        DRAFT1 --> OUT1["05-Reviews/<br/>YYYY-MM-DD.md"]
        OUT1 --> IDX2[auto-index]
    end

    subgraph "Weekly digest (dom 22:00 launchd)"
        CRON2[cron semanal] --> DIG[rag digest]
        MOD -.semana.-> DIG
        L1 -.semana.-> DIG
        FMC["frontmatter<br/>contradicts:"] --> DIG
        LC -.semana.-> DIG
        DIG --> DRAFT2[command-r draft<br/>narrativo]
        DRAFT2 --> OUT2["05-Reviews/<br/>YYYY-WNN.md"]
        OUT2 --> IDX3[auto-index]
    end

    IDX2 -. consumible por .-> DIG
    IDX3 -. consumible por .-> MORN
```

El loop se auto-alimenta: un brief matutino es una nota que podría aparecer en el digest semanal; un digest semanal es contexto para el próximo morning.

### 3.5 Chat intent routing (texto → ¿qué hacer?)

```mermaid
flowchart TD
    Input[user input en rag chat / bot]
    Input --> Q{/exit?}
    Q -->|sí| Exit[salir]
    Q -->|no| LnkI{detect_link_intent?}
    LnkI -->|sí| L[find_urls<br/>render URLs<br/>skip LLM]
    LnkI -->|no| RxI{detect_reindex_intent?}
    RxI -->|sí, strong verb<br/>o weak+object| RX["_run_index<br/>+ reset? reindex completo"]
    RxI -->|no| SvI{detect_save_intent?}
    SvI -->|sí, strong<br/>o neutral+nota| SV["save_note<br/>(last_assistant → Inbox)"]
    SvI -->|no| Default["retrieve → LLM<br/>→ render + sources + related"]
    Default --> Persist[append turn to session]
    Persist --> Loop[volver al prompt]
    L --> Loop
    RX --> Loop
    SV --> Loop

    style LnkI fill:#ffe
    style RxI fill:#fee
    style SvI fill:#eef

    Note1["ORDEN IMPORTA:<br/>link antes que save para que<br/>'link a la nota X' no triggerea save.<br/>Reindex antes que save para que<br/>'reindexá las notas' no triggerea save"]
```

Todos los detectores son regex + word-boundary. Ningún LLM se llama para clasificar intención — determinístico y rápido.

---

## 4. Lifecycle de una nota

El viaje completo de una idea, desde capture hasta posible archivo. Ilustra cómo casi todas las primitivas del sistema tocan una nota en algún momento.

```mermaid
stateDiagram-v2
    [*] --> Captured: rag capture / /note<br/>Obsidian mobile / desktop

    Captured --> InInbox: escrita a 00-Inbox/
    InInbox --> Indexed: rag watch / rag index

    Indexed --> Triaged: rag inbox --apply<br/>(mueve + taggea + linkifica)
    Indexed --> InInbox: usuario edita

    Triaged --> InArea: mueve a 02-Areas/ u otro
    InArea --> Retrievable: queryeable vía RAG

    Retrievable --> Linked: rag wikilinks --apply<br/>densifica outlinks
    Retrievable --> Flagged: index-time contradiction<br/>detected

    Flagged --> Retrievable: después de revisar<br/>(contradicts: aún en FM)

    Linked --> Cited: aparece en queries.jsonl
    Cited --> Linked: sigue viva

    Linked --> Stale: mtime > 1yr<br/>sin edges + sin citas
    Stale --> DeadCandidate: rag dead la lista
    DeadCandidate --> InArchive: usuario la mueve manual<br/>a 04-Archive/
    DeadCandidate --> Retrievable: usuario la revive<br/>(edita, wikilink, etc.)

    InArchive --> [*]: excluida de rag dead<br/>(excluded_folders)
```

### Qué primitiva entra en cada paso

| Transición | Comando / primitiva | Escribe |
|---|---|---|
| `∅ → Captured` | `rag capture`, `/note`, voice + caption | `00-Inbox/YYYY-MM-DD-HHMM-*.md` |
| `Captured → Indexed` | `_index_single_file` (via `rag watch` o `rag index`) | sqlite-vec chunks + URLs |
| `InInbox → Triaged` | `rag inbox --apply` | Mueve archivo + frontmatter tags + wikilinks |
| `Indexed → Flagged` | `find_contradictions_for_note` (phase 2) | `contradicts:` en frontmatter + `contradictions.jsonl` |
| `Retrievable → Linked` | `rag wikilinks suggest --apply` | `[[wrapped]]` menciones en body |
| `* → Cited` | `rag query/chat` retrieves it | `paths: [...]` en queries.jsonl |
| `* → DeadCandidate` | `rag dead` | Nada (solo lista — el usuario decide) |

---

## 5. State management

### 5.1 sqlite-vec collections

```mermaid
graph LR
    V[vault path] --> H["sha8 hash<br/>(solo si no-default)"]
    H --> B1[obsidian_notes_v11<br/>main collection]
    H --> B2[obsidian_urls_v1<br/>URL sub-index]
    B1 --> C1["~4608 chunks<br/>en vault actual"]
    B2 --> C2["~1887 URLs<br/>en vault actual"]

    Notes["nota .md"] --> MC["múltiples chunks<br/>150-800 chars"]
    MC --> B1
    Notes --> MU["múltiples URLs<br/>(markdown + bare)"]
    MU --> B2

    style B1 fill:#bfb
    style B2 fill:#bbf
```

**Invariantes**:
- Bump de `_COLLECTION_BASE` = rebuild obligatorio (fresh collection).
- `OBSIDIAN_RAG_VAULT` distinto del default → sha8 suffix automático. Aisla vaults.
- Orphan cleanup: archivos removidos del disk → sus chunks se borran en el próximo `rag index`.

### 5.2 Session lifecycle

```mermaid
stateDiagram-v2
    [*] --> Fresh: ensure_session(None)<br/>o sesión nueva
    [*] --> Reloaded: ensure_session(id)<br/>y sesión existe

    Fresh --> Active: primer turno persistido
    Reloaded --> Active: continúa turnos

    Active --> Active: append_turn<br/>(hasta SESSION_MAX_TURNS=50)
    Active --> Pruned: turn > 50<br/>→ drop oldest

    Pruned --> Active: siguiente turno

    Active --> Stale: mtime > SESSION_TTL_DAYS=30
    Stale --> Deleted: rag session cleanup<br/>o manual clear

    Active --> Deleted: rag session clear <id>
    Deleted --> [*]
```

IDs opacos: `[A-Za-z0-9_.:-]{1,64}`. El listener de WhatsApp pasa `wa:<jid>` literal (ej: `wa:120363426178035051@g.us`); CLI usa autogen `<unixhex>-<rand6>`. Sesiones `tg:<chat_id>` viejas siguen válidas (formato idéntico) pero nunca se reactivan desde el nuevo listener.

### 5.3 Launchd services (automatización)

```mermaid
gantt
    title Servicios launchd (horario local)
    dateFormat HH:mm
    axisFormat %H:%M
    section watch
    continuo (KeepAlive)  :active, w, 00:00, 24h
    section morning (lun-vie)
    morning brief       :m1, 07:00, 10m
    section digest (domingo)
    weekly digest       :d1, 22:00, 15m
```

Los tres servicios son **independientes**. Si `watch` muere, no afecta `morning`/`digest`. Cada uno loggea a `~/.local/share/obsidian-rag/<name>.{log,error.log}`.

### 5.4 Log files (append-only, never rotated automatically)

| Log | Escribe | Lee |
|---|---|---|
| `queries.jsonl` | Cada `rag query`/`chat`/`links`/`dead` | `rag log`, `rag gaps`, `rag morning`, `rag digest`, `find_dead_notes` |
| `contradictions.jsonl` | `_check_and_flag_contradictions` en indexing | `rag morning`, `rag digest` |
| `ambient.jsonl` | `_ambient_hook` en cada save de Inbox | `rag ambient log` |
| `ambient_state.jsonl` | `_ambient_state_record` tras cada hook | `_ambient_should_skip` (dedup 5min) |
| `ambient.json` | `/enable_ambient` desde el listener WhatsApp | `_ambient_config` en rag.py (schema: `{jid, enabled}`; schema viejo `{chat_id, bot_token}` rechazado con hint) |
| `watch.log` | stdout de `rag watch` | `tail` manual |
| `morning.log` | stdout de `rag morning` | `tail` manual |
| `digest.log` | stdout de `rag digest` | `tail` manual |

**Sin rotación automática** — si crecen mucho (no lo han hecho en meses de uso), `rag log -n 1000 > tmp && mv tmp queries.jsonl` manualmente. TODO: rotación auto.

---

## 6. Superficies de uso

### CLI

Canonical interface. Todas las primitivas están expuestas. Ver `rag --help` o `README.md` para detalle.

### MCP (Claude Code)

5 tools expuestos vía stdio por `obsidian-rag-mcp`:

```mermaid
graph LR
    Claude[Claude Code]
    Claude -->|"rag_query(q, session?)"| RQ[rag.retrieve]
    Claude -->|"rag_links(q)"| RL[rag.find_urls]
    Claude -->|rag_read_note| RN["vault/path.md"]
    Claude -->|rag_list_notes| RLN[corpus metadata]
    Claude -->|rag_stats| RS[col.count + models]

    RQ --> Ret[return chunks]
    RL --> URL[return URLs]
```

Cada call a `rag_query` con `session_id` extiende la misma sesión que la CLI usa — estado compartido.

### WhatsApp listener — bot unificado

Un solo listener consolida los roles que antes ocupaban los 3 bots de Telegram (`@ffeerrr_bot`, `@ragsystemobs_bot`, `@rauuuliiitoo_bot`). Vive en `~/whatsapp-listener/listener.ts`, arranca vía launchd (`com.fer.whatsapp-listener`), y polea el SQLite del bridge local (`com.fer.whatsapp-bridge`, `http://localhost:8080`). Anti-loop con U+200B prefix (ignora mensajes que arrancan con ZWSP — sus propios outputs vía bridge).

| Trigger | Comportamiento | Backend |
|---|---|---|
| texto libre / `/rag <q>` | query RAG | `rag query --session wa:<jid> --plain` |
| `/read <url>` | ingesta externa → 00-Inbox | `rag read` |
| `/note <t>` | captura manual → 00-Inbox | `rag capture` |
| `/followup [N]` | loops abiertos del vault | `rag followup --days N` |
| `/today` | end-of-day closure | `rag today` |
| voz (default) | Whisper → captura auto | `rag capture --stdin` |
| `/enable_ambient` · `/disable_ambient` · `/ambient_status` | CRUD de `ambient.json` con el JID actual | escribe/lee `~/.local/share/obsidian-rag/ambient.json` |

### WhatsApp — voice + routing detallado

```mermaid
flowchart TB
    Msg[mensaje entrante en self-chat RagNet]
    Msg --> Loop{arranca con U+200B?}
    Loop -->|sí| Drop[drop — es mi propio output]
    Loop -->|no| Type{tipo?}
    Type -->|texto /rag q| Rag["rag query --session wa:<jid>"]
    Type -->|texto /read url| Read["rag read"]
    Type -->|texto /note t| Cap["rag capture"]
    Type -->|texto /followup| Fol["rag followup"]
    Type -->|texto /today| Tod["rag today"]
    Type -->|texto libre| RagT["rag query (default)"]
    Type -->|voz| V[transcribeVoice via Whisper]
    V --> CapV["rag capture --stdin (default)"]

    style Rag fill:#bfb
    style RagT fill:#bfb
    style Cap fill:#ffb
    style CapV fill:#ffb
```

El listener NO mantiene estado conversacional propio — todo vive en el RAG. La session `wa:<jid>` hace que retomes el hilo en el chat y lo encuentres también vía `rag session show wa:<jid>` desde la CLI.

### Automatización (launchd)

Ver §5.3 arriba. Principio: **todo lo que puede ser automático lo es**. El usuario solo interviene cuando quiere generar on-demand (`rag morning --dry-run`, `rag digest --week X`).

---

## 7. Concurrencia y fallo

### Concurrencia

- **sqlite-vec**: single-writer (no concurrent writes desde múltiples procesos al mismo DB). En la práctica: `rag watch` puede chocar con `rag index` manual. Mitigación: evitar correr `rag index` mientras `watch` está procesando un cambio. Lo deseable: poner un file lock; TODO futuro.
- **BM25 + sqlite-vec + GIL**: serializados por el GIL de Python. Paralelizar con `ThreadPoolExecutor` dio 3× MÁS LENTO en M3 Max (medido). No paralelizar.
- **Ollama**: thread-safe del lado servidor; queremos un solo modelo en VRAM (por eso `OLLAMA_KEEP_ALIVE=-1`).
- **Sessions**: write atómico (tmp + replace). Múltiples procesos pueden leer simultáneamente sin issue.

### Fallo y recuperación

| Fallo | Síntoma | Recuperación |
|---|---|---|
| Ollama muere | `rag query` tira error de connection | `brew services start ollama` → reintentar |
| Reranker cae a CPU | Query 3× más lenta | Verificar `get_reranker()` fuerza `device="mps"+fp16`; no removerlo |
| sqlite-vec corrupt | `col.count()` tira excepción | `rag index --reset` desde cero (~30 min) |
| Vault desaparece (iCloud disconnect) | `rag` dice "índice vacío" aunque `ragvec.db` tiene data | Verificar `$OBSIDIAN_RAG_VAULT`; esperar que iCloud monte |
| launchd service crashea | No auto-fire del digest/morning | `launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-<name>` |
| vec collection mismatch (vault movido) | Queries devuelven 0 resultados | Reindex; `_vault_slug` cambió con el path |
| Tests fallan después de edit | `_URLS_BACKFILL_DONE` latch persistente entre tests | Fixture ya resetea; si aparece en test nuevo, monkeypatch a False |

### Data flow en fallo parcial

Cada log write es best-effort (try/except pass). El pipeline de indexing NO rompe porque falló logging. El pipeline de queries NO rompe porque falló session save. Grado alto de robustez ante estado corrupto parcial.

---

## 8. Performance envelope

**Medidas reales en M3 Max sobre vault de 526 notas / 4608 chunks**:

| Operación | Cold | Warm | Notas |
|---|---|---|---|
| `rag query "X"` total | 12-15s | 3-5s | Cold: Ollama carga modelos en VRAM |
| retrieve() interno | ~350ms | ~2ms | Corpus cache (_load_corpus) 341ms → 2ms |
| bge-m3 embed (batch 3) | ~200ms | ~80ms | Un solo call para multi-query |
| BM25 search | <50ms | <5ms | Cache invalidate por `col.count()` delta |
| cross-encoder rerank (20 chunks) | ~1.5s | ~400ms | MPS+fp16; con CPU: 3× |
| command-r gen (500 tok) | ~8s | ~2s | Cold incluye modelo load |
| `rag index` incremental (sin cambios) | ~2s | — | Hash gate evita re-embed |
| `rag index` con 10 notas modificadas | ~15s | — | Sin contradiction check |
| `rag index` con contradiction check | +5-10s por nota | — | command-r call |
| `rag index --reset` full vault | ~30-45min | — | 526 notas desde cero |
| `rag links --rebuild` | ~75s | — | Solo URL extract + embed (no chunks) |
| `rag dupes` | <1s | <1s | Numpy pairwise sobre centroides |
| `rag wikilinks suggest` full vault | ~10s | — | Puro regex contra title_to_paths |
| `rag morning` | ~30s | ~15s | File scan + command-r draft |
| `rag digest --week X` | ~45s | ~20s | Idem + más evidencia |

**Modelos residentes** (con `OLLAMA_KEEP_ALIVE=-1`): ~15GB VRAM (command-r 20GB q4 + bge-m3 1.2GB + qwen2.5:3b 2.4GB). Sin keep-alive: reload penalty ~8s por modelo distinto al anterior.

---

## 9. Extensión: cómo agregar una feature

Patrón que emergió de las últimas 10 features:

1. **Identificar la primitiva nueva** — ¿es composición de lo existente o algo fundamentalmente distinto? Si es composición, casi seguro ya hay el building block (retrieve, find_*, capture_note, _index_single_file).

2. **Escribir la función pura primero** — `find_<something>(col, ...) -> list[dict]` o similar. Sin `console.print`, sin CLI hooks. Testeable en aislamiento.

3. **Si usa LLM para judgment**: usar `resolve_chat_model()` (command-r), NO el helper. qwen2.5:3b es no-determinista para juicio. El helper solo para paraphrase/reformulate (textualmente predecible).

4. **Agregar CLI wrapper** con Click. Flags: siempre ofrecer `--plain` si hay output consumible por scripts/bots, `--dry-run` si es destructivo. `--apply` para comandos "propose y ejecuta".

5. **Tests en `tests/test_<feature>.py`** usando los patrones existentes:
   - Fixture con tmp_path vault + sqlite-vec stub.
   - Monkeypatch `rag.embed`, `rag.get_reranker`, `rag.ollama.chat` para evitar LLM real.
   - Cobertura: happy path, edge case vacío, error handling.

6. **Si es proactivo (auto-fire)**: agregar plist en `_services_spec` de `rag setup`. Reload con `rag setup`.

7. **Actualizar `README.md`** (tabla de comandos) + `CLAUDE.md` (si es decisión de diseño no obvia) + este `SYSTEM.md` (si cambia el modelo conceptual).

8. **Commit cohesivo** con mensaje detallado explicando el "why", no solo el "what".

### Anti-patrones observados

- ❌ Agregar un LLM call en el hot path del retrieve sin medir — termina siendo un cuello de botella.
- ❌ Usar helper para judgment — FP rate lo hace inservible.
- ❌ Paralelizar BM25 + sqlite-vec — GIL serializa igual.
- ❌ Tocar `_invalidate_corpus_cache()` heuristics sin medir.
- ❌ Docs que dupliquen código — apuntar a `rag.py` con línea si hace falta; dejar que el doc explique el POR QUÉ.
- ❌ Remover `device="mps"+fp16` de `get_reranker()` — cae a CPU en uv venvs.

---

## Apéndice: cross-reference rápida

| Pregunta | Ir a |
|---|---|
| "¿Qué comandos hay?" | [README §Comandos](../README.md#comandos--referencia-completa) |
| "¿Cómo instalo esto?" | [README §Instalación](../README.md#instalación--setup) |
| "¿Dónde guarda cada cosa?" | [README §Storage layout](../README.md#storage-layout-dónde-vive-cada-cosa) |
| "¿Cómo resuelvo error X?" | [README §Troubleshooting](../README.md#troubleshooting) |
| "¿Por qué está diseñado así?" | [CLAUDE.md](../CLAUDE.md) |
| "¿Qué aprendimos empíricamente?" | [CLAUDE.md §Findings](../CLAUDE.md) / [README §Findings](../README.md#findings-empíricos-clave-no-olvidar) |
| "¿Cómo migro a otra Mac?" | Conversación con Claude — Opción A/B/C |
| "¿Cómo extiendo?" | §9 de este doc |
| "¿Qué primitivas hay?" | §2 de este doc (Component map) |
