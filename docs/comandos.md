# Comandos y modificadores

Todos los comandos de `rag` en un solo lugar. Agrupados por qué hacen, no por orden alfabético.

> Para ayuda corta en la terminal: `rag <comando> --help`.

## Índice rápido

1. [Lo básico](#lo-básico)
2. [Capturar cosas](#capturar-cosas)
3. [Organizar el Inbox](#organizar-el-inbox)
4. [Navegar el grafo y explorar](#navegar-el-grafo-y-explorar)
5. [Briefs automáticos](#briefs-automáticos)
6. [Limpiar y mantener el vault](#limpiar-y-mantener-el-vault)
7. [Ver cómo va el sistema](#ver-cómo-va-el-sistema)
8. [Sesiones conversacionales](#sesiones-conversacionales)
9. [Múltiples vaults](#múltiples-vaults)
10. [Servicios y automatización](#servicios-y-automatización)
11. [Otros](#otros)
12. [Convenciones de flags](#convenciones-de-flags)

---

## Lo básico

### `rag index`
Indexa tu vault. Incremental — solo re-procesa notas que cambiaron desde la última corrida.

```bash
rag index                           # incremental
rag index --reset                   # borra y reconstruye todo (lento)
rag index --no-contradict           # saltea el check de contradicciones
rag index --source whatsapp         # indexa WhatsApp en vez del vault
rag index --source calendar         # idem para calendario
rag index --source gmail            # idem para mails
rag index --source reminders        # idem para Apple Reminders
rag index --vault work              # usa otro vault sin cambiar el default
rag index --since 2026-04-01        # solo desde esta fecha (cross-source)
rag index --dry-run                 # no escribe (solo cross-source)
```

### `rag query "tu pregunta"`
Hacé una pregunta y te devuelve la respuesta + links a las notas.

```bash
rag query "qué sé sobre X?"
rag query "X" -k 8                             # traer 8 chunks (default: 5)
rag query "X" --folder 02-Areas/Musica         # limitar a una carpeta
rag query "X" --tag coaching                   # limitar a un tag
rag query "X" --since 7d                       # solo notas de los últimos 7 días
                                               # (acepta 7d/2w/3m/1y o YYYY-MM-DD)
rag query "X" --source vault,gmail             # elegir fuentes (coma-separadas)
rag query "X" --raw                            # mostrar chunks crudos (sin LLM)
rag query "X" --counter                        # además buscar contradicciones
rag query "X" --critique                       # que el LLM revise su respuesta
rag query "X" --no-cache                       # saltear el cache semántico
rag query "X" --plain                          # sin colores (para bots/scripts)
rag query "X" --continue                       # retoma la última sesión
rag query "X" --session mi-id                  # guardar/retomar por id explícito
rag query "X" --force                          # forzar LLM aun si confianza baja
rag query "X" --loose                          # permite prosa externa del LLM
rag query "X" --hyde                           # activar HyDE (solo con LLMs grandes)
rag query "X" --no-auto-filter                 # no inferir folder/tag de la pregunta
rag query "X" --no-deep                        # desactivar retrieval iterativo
```

### `rag chat`
Chat interactivo. Los pronombres absorben contexto entre turnos.

```bash
rag chat                                       # abre el prompt
rag chat --resume                              # retoma la última sesión
rag chat --session mi-id                       # por id explícito
rag chat --precise                             # HyDE + reformulación (+5s)
rag chat --counter                             # contradicciones en cada respuesta
rag chat --deep                                # retrieval iterativo (más lento)
rag chat --critique                            # auto-review del LLM
rag chat --vault work                          # otro vault solo para esta sesión
rag chat --vault work,personal                 # buscar en varios vaults
rag chat --vault all                           # buscar en todos los registrados
rag chat --folder / --tag / --since            # mismos filtros que query
rag chat -k 8                                  # chunks por turno
```

Dentro del chat se reconocen comandos tipo slash:
- `/help`, `/exit`, `/quit`
- `/save <título>` — guarda la última respuesta como nota al `00-Inbox/`
- `/reindex` — dispara un reindex incremental
- `/sources`, `/open <n>`, `/copy`, `/stats`

### `rag stats`
Muestra el estado del índice: cuántos chunks, qué modelos, qué vault.

```bash
rag stats
```

---

## Capturar cosas

### `rag capture "texto"`
Guarda una nota rápida al `00-Inbox/` del vault.

```bash
rag capture "idea suelta sobre X"
rag capture "idea" --tag voice --source whatsapp-voice
rag capture --stdin --tag transcript          # lee texto de stdin
echo "transcript largo" | rag capture --stdin
rag capture "texto" --title "mi-titulo"       # slug custom del archivo
rag capture "texto" --plain                   # solo imprime el path
```

### `rag read <url>`
Baja una URL, la resume con el LLM y te propone guardarla como nota.

```bash
rag read https://ejemplo.com/articulo            # dry-run: te muestra el preview
rag read https://ejemplo.com/articulo --save     # guarda a 00-Inbox/ + indexa
rag read <url> --plain                           # sin colores (para bots)
```

### `rag bookmarks`
Indexa los bookmarks de Chrome al sub-índice de URLs.

```bash
rag bookmarks sync            # parsea + indexa
rag bookmarks stats           # cuántos indexados por profile
rag bookmarks clear           # borra todos (no toca notas)
```

### `rag wa-tasks`
Extrae tareas/preguntas de tus chats de WhatsApp y las guarda al Inbox.

```bash
rag wa-tasks                         # delta desde último run
rag wa-tasks --dry-run               # solo mostrar
rag wa-tasks --hours 48              # ventana custom
rag wa-tasks --force                 # ignora el state file
```

---

## Organizar el Inbox

### `rag inbox`
Triage masivo: para cada nota del Inbox sugiere carpeta destino + tags + wikilinks + dupes.

```bash
rag inbox                              # dry-run, mostrar plan
rag inbox --apply                      # aplicar todo
rag inbox --folder 00-Inbox            # folder a triar
rag inbox --limit 20                   # cuántas notas por corrida
rag inbox --max-tags 5                 # tope de tags por nota
rag inbox --folder-min-conf 0.4        # confianza mínima para mover
rag inbox --no-folder                  # no sugerir carpeta
rag inbox --no-tags                    # no sugerir tags
rag inbox --no-wikilinks               # no sugerir wikilinks
```

### `rag file`
Filing asistido: más lento que `inbox`, pero con confirmación humana por nota.

```bash
rag file                                 # dry-run
rag file <path-a-nota>                   # una sola nota
rag file --apply                         # interactivo (y/n/e/s/q)
rag file --apply --one                   # solo la nota más vieja
rag file --undo                          # revertir el último batch
rag file --folder 01-Projects -k 8       # scope + cantidad de vecinos
rag file --limit 20                      # tope por corrida
rag file --plain                         # salida plana
```

### `rag autotag <path>`
Sugiere tags para una nota usando el vocabulario existente.

```bash
rag autotag 00-Inbox/mi-nota.md
rag autotag 00-Inbox/mi-nota.md --apply        # escribir al frontmatter
rag autotag <path> --max-tags 6
```

### `rag fix <path>`
Decile al sistema que el path correcto para el último turn era este. Mejora el ranker.

```bash
rag fix 02-Areas/X.md                   # usa la última sesión
rag fix <path> --session mi-id
rag fix <path> --plain
```

---

## Navegar el grafo y explorar

### `rag wikilinks suggest`
Busca menciones a títulos de notas que NO están escritas como `[[wikilink]]`.

```bash
rag wikilinks suggest                                # dry-run, todo el vault
rag wikilinks suggest --note <path>                  # solo una nota
rag wikilinks suggest --folder 02-Areas/Coaching
rag wikilinks suggest --apply                        # escribir + reindexar
rag wikilinks suggest --min-len 5
rag wikilinks suggest --max-per-note 20
rag wikilinks suggest --show 10
```

### `rag graph <titulo>`
Exporta un canvas de Obsidian con la vecindad de una nota.

```bash
rag graph "mi-nota"
rag graph "mi-nota" --depth 2                      # 2 niveles de vecindad
rag graph "mi-nota" --output mi.canvas             # path custom
```

### `rag surface`
Detecta "puentes no hechos": pares cercanos en significado pero lejanos en el grafo.

```bash
rag surface                                        # dry-run top 5
rag surface --sim-threshold 0.82 --min-hops 4
rag surface --top 10
rag surface --skip-young-days 7                    # ignorar notas recientes
rag surface --no-llm                               # sin "por qué" (más rápido)
rag surface --plain
```

### `rag dupes`
Encuentra pares de notas potencialmente duplicadas.

```bash
rag dupes                                           # default 0.85 cosine
rag dupes --threshold 0.90
rag dupes --folder 00-Inbox
rag dupes --limit 20
rag dupes --plain
```

### `rag timeline`
Notas ordenadas por fecha de modificación.

```bash
rag timeline
rag timeline "busqueda semantica"
rag timeline --tag coaching --folder 02-Areas
rag timeline --limit 50
```

### `rag links "query"`
Busca URLs en tu vault por contexto semántico — sin LLM, bien rápido.

```bash
rag links "documentación claude code"
rag links "ollama" -k 20
rag links "X" --folder / --tag             # filtros
rag links "X" --open 1                     # abre el rank 1 en el browser
rag links "X" --source note                # solo URLs de notas
rag links "X" --source bookmark            # solo bookmarks de Chrome
rag links "X" --plain
rag links --rebuild                        # re-extraer todas las URLs
```

### `rag prep "tema"`
Brief estructurado sobre una persona/proyecto/tema, armado con contexto del vault + URLs + notas relacionadas.

```bash
rag prep "Maria coaching liderazgo"
rag prep "X" --save                        # escribir al 00-Inbox/
rag prep "X" -k 8
rag prep "X" --folder 02-Areas/Coaching
rag prep "X" --no-urls --no-related
rag prep "X" --plain
```

---

## Briefs automáticos

### `rag morning`
Brief matutino: qué pasó ayer + qué enfocar hoy. Escribe a `05-Reviews/YYYY-MM-DD.md`.

```bash
rag morning                                # escribe
rag morning --dry-run                      # solo muestra
rag morning --date 2026-04-22              # fecha custom
rag morning --lookback-hours 36            # ventana hacia atrás
```

### `rag today`
Cierre del día: qué pasó hoy + cabos sueltos + semillas para mañana.

```bash
rag today                                  # escribe 05-Reviews/YYYY-MM-DD-evening.md
rag today --dry-run
rag today --date 2026-04-22
rag today --plain
```

### `rag digest`
Weekly digest narrativo del vault.

```bash
rag digest                                 # semana actual
rag digest --week 2026-W15                 # semana ISO específica
rag digest --days 7
rag digest --dry-run
```

### `rag pendientes`
Dashboard unificado: Gmail pending + WhatsApp activos + Reminders + loops del vault + agenda del día.

```bash
rag pendientes                             # vista completa
rag pendientes --days 14
rag pendientes --plain
```

### `rag followup`
Open loops del vault: lo que dijiste que harías y no cerraste.

```bash
rag followup
rag followup --days 30
rag followup --status stale                # solo los que quedaron viejos
rag followup --status activo               # solo los activos
rag followup --status resolved             # solo los resueltos
rag followup --stale-days 14
rag followup --json
rag followup --plain
```

---

## Limpiar y mantener el vault

### `rag dead`
Lista notas candidatas a archivar (sin links entrantes, sin salientes, viejas, no usadas).

```bash
rag dead                                   # default 365 días
rag dead --min-age-days 180
rag dead --query-window-days 90
rag dead --folder 03-Resources
rag dead --limit 100
rag dead --plain
```

### `rag archive`
Mueve las notas muertas a `04-Archive/` preservando la jerarquía.

```bash
rag archive                                # dry-run
rag archive --apply                        # ejecutar (con gate de confirmación)
rag archive --apply --force                # bypass del gate
rag archive --gate 20                      # límite de notas que requiere --force
rag archive --min-age-days 365
rag archive --folder / --limit
rag archive --notify / --no-notify         # push a WhatsApp si ambient está on
rag archive --report / --no-report         # reporte a 05-Reviews/
rag archive --plain
```

### `rag ignore`
Lista de notas hard-ignore que nunca aparecen en retrieve.

```bash
rag ignore                                 # listar
rag ignore add <path>
rag ignore rm <path>
rag ignore clear
```

### `rag maintenance`
Mantenimiento integral: reindex + limpia sesiones + rota logs + detecta dead notes.

```bash
rag maintenance                            # ejecutar todo
rag maintenance --dry-run                  # reportar sin tocar
rag maintenance --skip-reindex
rag maintenance --skip-logs
rag maintenance -v                         # verbose
rag maintenance --json                     # output para cron/scripts
```

### `rag consolidate`
Agrupa conversaciones del Inbox semánticamente similares y las promueve a PARA.

```bash
rag consolidate                            # dry-run
rag consolidate --apply                    # aplicar
rag consolidate --window-days 14
rag consolidate --threshold 0.75
rag consolidate --min-cluster 3
rag consolidate --json
```

---

## Ver cómo va el sistema

### `rag log`
Inspeccioná el log de queries.

```bash
rag log                                     # últimas 20
rag log -n 50
rag log --low-confidence                    # solo queries con confianza baja
rag log --feedback                          # solo turnos con 👍/👎
rag log --silent-errors                     # errores que fallaron en silencio
rag log --silent-errors --summary           # agrupados
```

### `rag dashboard`
Métricas del pipeline sobre `queries.jsonl`.

```bash
rag dashboard
rag dashboard --days 30
```

Si además tenés el web server corriendo (`rag serve` o el servicio launchd `com.fer.obsidian-rag-web`), entrá a `/dashboard` en el browser y vas a ver:

- **KPIs** arriba (queries totales, latencia, feedback positivo, etc).
- **Señales al ranker-vivo** (panel nuevo): cuántos eventos implícitos llegaron en la ventana — `copy`, `open`, `save`, `kept`, `positive_implicit` (suman al CTR), `negative_implicit`, `deleted` y `impression` (el denominator). Si está vacío, te dice qué hacer para empezar a alimentarlo (copiar/guardar/ratear una respuesta).
- **Charts** abajo (queries por día, top folders, etc.).

Más sobre el ranker-vivo en [como-funciona.md](./como-funciona.md#el-ranker-vivo).

### `rag eval`
Evalúa el retriever contra `queries.yaml` (golden set).

```bash
rag eval                                    # hit@k, MRR, recall@k
rag eval --file otro.yaml
rag eval -k 10
rag eval --hyde                             # con HyDE
rag eval --no-multi                         # sin multi-query
rag eval --latency                          # + P50/P95/P99 de retrieve()
rag eval --max-p95-ms 3000                  # gate para CI (exit 1 si se pasa)
```

### `rag tune`
Auto-calibra los pesos del ranker.

```bash
rag tune                                    # dry-run, mostrar winner
rag tune --apply                            # persistir
rag tune --yes                              # idem sin prompt (para cron)
rag tune --samples 500 --seed 42
rag tune --no-chains                        # solo singles + feedback
rag tune --no-feedback                      # solo queries.yaml
rag tune --online --days 14                 # incluir behavior.jsonl
rag tune --rollback                         # restaurar el backup más reciente
```

### `rag rate <rating>`
Aplicá feedback al último turn.

```bash
rag rate +                                  # 👍
rag rate -                                  # 👎
rag rate 👍
rag rate "0 genérico"                       # rating + reason
rag rate --session mi-id
rag rate --reason "falta X.md"
rag rate --plain
```

### `rag gaps`
Detecta temas consultados repetidamente sin respuesta en el vault.

```bash
rag gaps
rag gaps --threshold 0.015
rag gaps --min-count 2
rag gaps --days 60
```

### `rag emergent`
Detecta temas emergentes en `queries.jsonl` y propone capturarlos.

```bash
rag emergent
rag emergent --dry-run                      # default
rag emergent --push                         # pingear WhatsApp
rag emergent --days 14
rag emergent --min-size 5
rag emergent --threshold 0.75
```

### `rag insights`
Patrones en el log: gaps + hot queries + notas huérfanas.

```bash
rag insights                                # output texto
rag insights --days 30
rag insights --min-gap 2
rag insights --min-hot 3
rag insights --json
rag insights --plain
rag insights telemetry-health               # salud de la telemetría
```

### `rag patterns`
Alerta cuando una razón de feedback domina.

```bash
rag patterns
rag patterns --last 30
rag patterns --min-share 0.4
rag patterns --dry-run / --push
```

---

## Sesiones conversacionales

Cada chat/query con `--session` se guarda como archivo JSON. Sobreviven al cierre del proceso.

```bash
rag session list                            # ver sesiones recientes
rag session list -n 50
rag session show <id>                       # ver todos los turns de una sesión
rag session clear <id>                      # borrar una
rag session clear <id> --yes                # sin prompt
rag session cleanup                         # purgar viejas (default: 30 días)
rag session cleanup --days 14
```

---

## Múltiples vaults

Registrá distintos vaults y cambiá entre ellos sin que se mezclen los índices.

```bash
rag vault list                              # ver los registrados (→ marca el activo)
rag vault add personal ~/ruta/al/vault
rag vault use personal                      # cambia el activo (persistente)
rag vault current                           # muestra el activo y por qué
rag vault remove personal                   # desregistrar (no borra chunks)
```

**Precedencia para resolver el vault activo**:
1. `OBSIDIAN_RAG_VAULT` env var (override por invocación).
2. `rag vault use <nombre>` (persistente en `vaults.json`).
3. Default legacy (iCloud Notes).

---

## Servicios y automatización

### `rag setup`
Instala o desinstala los servicios launchd que mantienen el sistema vivo.

```bash
rag setup                                   # instalar/recargar (idempotente)
rag setup --remove                          # desinstalar todos
```

### `rag watch`
Daemon que re-indexa cuando guardás en Obsidian (lo maneja launchd, pero lo podés correr a mano).

```bash
rag watch
rag watch --debounce 3                      # segundos antes de reindexar
rag watch --all-vaults                      # vigilar todos los registrados
                                            # (ahorra ~3-4 GB swap por vault extra)
```

### `rag serve`
Servidor HTTP persistente que mantiene modelos en memoria (para bots/integraciones).

```bash
rag serve                                   # default 127.0.0.1:7832
rag serve --host 0.0.0.0 --port 9000
```

Endpoints:
- `GET /health` — liveness + cantidad de chunks
- `POST /query` — pipeline completo, devuelve JSON

### `rag ambient`
El Ambient Agent reacciona a saves en `00-Inbox/` con sugerencias automáticas.

```bash
rag ambient status                          # ¿habilitado? ¿a qué jid?
rag ambient disable                         # desactivar (deja config)
rag ambient test <path>                     # simular análisis sobre una nota
rag ambient log                             # tail del log de eventos
rag ambient log -n 50
rag ambient folders list                    # folders vigilados
rag ambient folders add <folder>
rag ambient folders remove <folder>
```

La activación se hace desde WhatsApp con `/enable_ambient` en el grupo RagNet.

### `rag silence [kind]`
Silencia notificaciones proactivas por tipo.

```bash
rag silence --list                          # qué está silenciado
rag silence emergent                        # silenciar "emergent"
rag silence emergent --off                  # reactivar
```

Kinds conocidos: `emergent`, `patterns`, `followup`, `calendar`, `anniversary`.

### `rag state [texto]`
Lee/escribe tu estado actual (cansado, inspirado, focus-code…). TTL de 24h. Se inyecta al system prompt automáticamente.

```bash
rag state                                   # ver el actual (si no expiró)
rag state cansado                           # setear
rag state --clear                           # borrar
rag state --plain
```

### `rag open <path>`
Abre una nota y registra un evento de open (para el ranker-vivo).

```bash
rag open 01-Projects/foo.md
rag open <path> --query "la pregunta" --rank 1
rag open <path> --source cli / whatsapp / web / brief
rag open <path> --session mi-id
```

---

## Otros

### `rag do "instrucción"`
Agente con tools: puede buscar, leer notas y proponer writes. Te pide confirmación antes de escribir.

```bash
rag do "armá un resumen sobre ikigai"
rag do "listame referentes en coaching y proponé una nota índice"
rag do "..." --yes                          # aplicar writes sin prompt
rag do "..." --max-iterations 12            # tope de iteraciones
```

### `rag cache`
Cache semántico de respuestas (para queries repetidas sobre el vault sin cambios).

```bash
rag cache stats                             # hits, rows, corpus_hashes
rag cache clear                             # limpiar
```

### `rag weather ["pregunta"]`
Clima sin agent loop ni writes.

```bash
rag weather
rag weather "va a llover mañana en BA?"
```

### `rag spotify-auth`
One-shot para autorizar Spotify. Después, `rag index` sincroniza automáticamente.

```bash
rag spotify-auth
```

---

## Convenciones de flags

Flags que aparecen repetidos con el mismo significado en muchos comandos:

| Flag | Qué hace |
|---|---|
| `--dry-run` | Mostrar sin escribir |
| `--apply` | Ejecutar cambios (la mayoría default a dry-run) |
| `--plain` | Sin colores ni ANSI, para bots/scripts |
| `--json` | Output JSON estructurado |
| `--folder <X>` | Acotar a una subcarpeta del vault |
| `--tag <X>` | Acotar a notas con este tag |
| `--since <X>` | Acotar por fecha (`7d`, `2w`, `3m`, `1y`, o `YYYY-MM-DD`) |
| `-k <N>` | Cantidad de chunks/vecinos/resultados a usar |
| `--limit <N>` | Tope de notas que procesa el comando |
| `--days <N>` | Ventana temporal en días |
| `--vault <nombre>` | Usar otro vault registrado solo para esta invocación |
| `--session <id>` | Sesión explícita (reanuda si existe, crea si no) |
| `--resume` / `--continue` | Retomar la última sesión usada |
| `--force` | Bypass de algún gate de seguridad |
| `--help` | Ayuda del comando |
