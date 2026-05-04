# `rag-harness` — toggle de MCPs / tools / CLI groups

Setup para activar y desactivar dinámicamente el conjunto de MCP servers
(Claude Code + Devin CLI) y las tools que expone el MCP de `obsidian-rag`,
con el objetivo de **reducir el tamaño del harness** (cantidad de tools que
el agente carga al boot de la sesión).

## Por qué

Cada MCP server registrado en `~/.claude.json` o `~/.config/devin/config.json`
expone N tools al agente. Esas tools entran al system prompt como
descriptores y consumen contexto incluso si nunca se llaman. Con 23 MCPs ×
varias tools cada uno, el harness pesa varios miles de tokens.

`rag-harness` permite tener **profiles** versionados con un subset de
MCPs y tools, y switchear entre ellos antes de abrir una sesión nueva.

## Estructura

```
.devin/mcp-profiles/
├── README.md                      ← este archivo
├── inventory.json                 ← snapshot del set completo de MCPs
├── profiles/
│   ├── full.json                  ← todo activo (baseline)
│   └── rag-only.json              ← solo obsidian-rag + obsidian + filesystem + time
├── bin/
│   └── rag-harness                ← CLI Click (uv run inline-deps)
└── .active                        ← (auto) nombre del profile activo
```

## Instalación

El script `bin/rag-harness` usa `uv run` con dependencias inline (PEP 723).
Para tenerlo en el `$PATH`:

```bash
ln -sf "$(pwd)/.devin/mcp-profiles/bin/rag-harness" ~/.local/bin/rag-harness
```

## Uso

```bash
rag-harness list                         # qué profiles existen + activo
rag-harness show rag-only                # contenido del profile
rag-harness use rag-only --dry-run       # qué cambiaría
rag-harness use rag-only                 # aplicar (con backup automático)
rag-harness diff full rag-only           # comparar dos profiles
rag-harness new mio --from rag-only      # crear profile nuevo
rag-harness cost                         # estimar tokens del profile activo + comparativa
rag-harness wrap rag-only -- <comando>   # aplicar profile, correr comando, restaurar
rag-harness sync-inventory               # regenerar inventory desde ~/.claude.json
rag-harness sync-full                    # regenerar profiles/full.json
rag-harness restore                      # volver al backup más reciente
```

**Después de cada `use`, hay que reabrir la sesión** (Claude Code / Devin)
para que el harness se recompile. Esto es un gotcha conocido de los CLIs:
los MCPs y custom agents se cargan UNA sola vez al boot.

## Tres niveles de granularidad

### 1. MCP servers (qué servers están activos)

El profile lista los keys del inventory que querés tener prendidos.
Los demás se quitan de `~/.claude.json` y `~/.config/devin/config.json`.

**MCPs ajenos al inventory se preservan**: por ejemplo `engram` y `mem-vault`
viven en `~/.config/devin/config.json` pero no en el inventory; el harness
los deja como están y solo gestiona los keys que conoce.

### 2. Tools del MCP `obsidian-rag` (qué tools expone)

`obsidian-rag-mcp` lee `RAG_MCP_TOOLS` (CSV) al boot y solo registra las
tools listadas. Acepta nombre completo (`rag_query`) o sufijo corto
(`query`). Default (var ausente) = todas.

```json
{
  "rag_mcp_tools": ["query", "read_note", "list_notes", "stats"]
}
```

El harness inyecta `RAG_MCP_TOOLS` en `env` de la entry `obsidian-rag` al
aplicar el profile. Si el set vacío matchea ninguna tool, el MCP loggea
WARN al stderr y arranca sin tools (no crash).

**Modo terse** (`rag_mcp_terse: true` en el profile → inyecta
`RAG_MCP_TOOLS_TERSE=1`): recorta los docstrings al primer párrafo
antes de registrar cada tool. Reduce ~50%-90% los tokens del descriptor
sin afectar funcionalidad. Ej: `rag_query` baja de 952 a 58 chars de
docstring. Recomendado prender en profiles de read-only; dejarlo en
`false` cuando habilitás write-tools (`capture`/`save_note`/etc) para
que el agente vea las advertencias y formatos completos.

### 3. Top-level commands del CLI `rag` (qué subcommands se ven)

Análogo: el CLI `rag` lee `RAG_CLI_KEEP` (CSV de nombres de comandos
top-level) y oculta del Click tree todos los demás. **Esto NO reduce el
harness del agente** — los MCP no llaman al CLI. Sirve para limpiar
`rag --help` (de ~100 commands a un puñado) y reducir distracciones en
sesiones acotadas.

Implementado como allowlist (no como "grupos") porque el CLI es flat:
~100 `@cli.command()` sin metadata de grupo. Mantener una lista
explícita de comandos a conservar es más simple y preciso.

```json
{
  "rag_cli_keep": ["chat", "query", "index", "stats", "start", "stop", "status"]
}
```

**Dot-notation** para subcomandos de grupos: `vault.add,vault.list`
mantiene solo esos sub del grupo `vault`. Mezclable con top-level:
`query,vault.add,chat` deja `query` y `chat` enteros, y `vault` con
solo `add` visible. Si listás `vault` sin dot, todos sus sub quedan
intactos.

Default (var ausente) = todos los comandos visibles.

## Profiles disponibles

- **`full`**: todo el inventory activo, ninguna restricción. Baseline /
  restore. Auto-regenerable: `rag-harness sync-full`.
- **`rag-only`**: foco mínimo en query/edición de vault. 4 MCPs, 4 tools
  del RAG (lectura), modo terse, 10 CLI commands. Para "leer y consultar
  el vault".
- **`dev`**: sesiones de coding. 7 MCPs (RAG + github + context7 +
  filesystem + claude-peers + sequential-thinking), 5 tools de RAG
  (lectura + links). Para laburar en otros repos con contexto del vault.
- **`comms`**: componer mensajes con contexto. 6 MCPs (RAG + whatsapp +
  gmail-send + apple-mcp + obsidian + time), 2 tools de RAG (`query`,
  `read_note`).
- **`writing`**: redacción de notas largas + captura activa. 5 MCPs +
  TODAS las tools del RAG incluyendo write-tools (`capture`, `save_note`,
  `create_reminder`, `create_event`). Sin terse para que las firmas de
  write-tools queden completas.
- **`debug-web`**: debugging del frontend. 5 MCPs (RAG + playwright +
  filesystem + github + time), 2 tools mínimas del RAG.

**Para ver el costo de cada profile** (estimación de tokens):

```bash
$ rag-harness cost
profile         mcps  rag-tools   tokens_est     delta_vs_full
----------------------------------------------------------------
 * full           23         10       19,805
   writing         5          9        5,204    -14,601 (-74%)
   dev             7          5        4,959    -14,846 (-75%)
   comms           6          2        4,083    -15,722 (-80%)
   debug-web       5          2        3,297    -16,508 (-84%)
   rag-only        4          4        2,540    -17,265 (-88%)
```

(Estimación grosera, útil para comparar profiles entre sí.)

Para crear más: `rag-harness new <nombre> --from rag-only` y editar a mano.

## `wrap` — profile temporal por comando

```bash
rag-harness wrap rag-only -- claude --print "explicame X"
rag-harness wrap dev -- bash ./scripts/codegen.sh
```

Aplica el profile, ejecuta el comando, y restaura el profile previo
cuando el comando termina (también si el comando falla — `try/finally`).
Útil para correr scripts puntuales con un harness distinto sin
switchear permanentemente.

**Limitación**: solo afecta procesos NUEVOS. Una sesión de Claude Code
ya abierta no ve el cambio (su harness se compiló al boot). El wrap es
útil para spawneo de procesos nuevos durante su ventana de validez.

## Backups

Cada `rag-harness use` y `rag-harness sync-*` hacen backup automático del
archivo previo con sufijo `.bak-harness-<timestamp>`. `rag-harness restore`
recupera el backup más reciente de cada target.

## Gotchas

1. **Reabrir la sesión**: el harness no toma efecto en sesiones abiertas
   (los profiles de MCP se compilan al boot). Cerrá Claude Code / Devin y
   reabrí.
2. **`~/.claude.json` lo escriben otras herramientas**: si Claude Code
   regenera el archivo después del switch, perdés los cambios. Si pasa
   seguido, considerá hacer el `use` solo cuando Claude Code está cerrado.
3. **Inventory drift**: si agregás un MCP nuevo por fuera del harness, no
   está en el inventory y los profiles no lo ven. Corré `sync-inventory`
   y luego editá los profiles que correspondan.
