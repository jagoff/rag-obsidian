# Empezar con obsidian-rag

Guía para arrancar de cero. Si ya tenés el proyecto instalado, salteá al paso 3.

## ¿Qué es esto?

Un buscador inteligente para tu vault de Obsidian. En vez de buscar por palabras exactas como Ctrl+F, le hacés preguntas en lenguaje natural ("qué dije sobre X la semana pasada?") y te devuelve las notas relevantes con una respuesta resumida.

**Todo corre en tu Mac.** Sin cloud, sin enviar tus notas a ningún lado.

## Lo que necesitás antes

1. **Ollama** corriendo en tu Mac. Instalalo desde [ollama.com](https://ollama.com) y arrancá el servicio.
2. **Los modelos** que usa el sistema. Bajalos con:
   ```bash
   ollama pull bge-m3           # embeddings
   ollama pull qwen2.5:3b       # helper (chico y rápido)
   ollama pull qwen2.5:7b       # chat principal (recomendado)
   ```
3. **Python 3.13** y [`uv`](https://docs.astral.sh/uv/) instalados.
4. Un **vault de Obsidian** con notas en markdown.

## Paso 1 — Instalar

```bash
cd ~/repositories/obsidian-rag
uv tool install --reinstall --editable .
```

Esto te deja dos comandos disponibles globalmente:
- `rag` → la CLI principal
- `obsidian-rag-mcp` → servidor MCP (lo usa Claude Code automáticamente, no lo corrés a mano)

## Paso 2 — Indexar tu vault

```bash
rag index
```

La primera vez tarda unos minutos (leé el vault, parte en chunks, calcula embeddings). Después es incremental: solo re-procesa las notas que cambiaron.

**¿Tu vault no está en la ruta default de iCloud?** Decile cuál usar:

```bash
rag vault add personal ~/ruta/a/tu/vault
rag vault use personal
rag index
```

## Paso 3 — Tu primera pregunta

```bash
rag query "qué tengo anotado sobre productividad?"
```

Pipeline simplificado de lo que pasa adentro:

```
Tu pregunta
    ↓
Busca chunks relevantes en la base local
    ↓
Los ordena con un reranker (el que está más relacionado, primero)
    ↓
Le pasa los mejores al LLM para armar la respuesta
    ↓
Te muestra la respuesta + links clickeables a las notas
```

## Paso 4 — Chat interactivo

Para conversar sin perder contexto entre preguntas:

```bash
rag chat
```

Te abre un prompt donde podés seguir haciendo preguntas y las siguientes "absorben" el contexto de las anteriores. Salí con `Ctrl+D` o `/exit`.

## Paso 5 — Dejar que se actualice solo

```bash
rag setup
```

Esto instala 11 servicios en segundo plano (launchd). Los más importantes:

- **watch** → re-indexa cuando guardás una nota nueva en Obsidian
- **morning** → a las 7:00 de lun a vie te escribe un brief del día en tu vault
- **digest** → los domingos 22:00 te escribe un resumen semanal

Más detalle en [automatizaciones.md](./automatizaciones.md).

## ¿Y ahora qué?

- Ver [comandos.md](./comandos.md) para todos los comandos que existen.
- Ver [como-funciona.md](./como-funciona.md) si querés entender el pipeline.
- Ver [problemas-comunes.md](./problemas-comunes.md) si algo no anda.

## Cheatsheet mínima

```bash
rag index                          # indexá (incremental)
rag query "tu pregunta"            # pregunta única
rag chat                           # chat interactivo
rag chat --resume                  # retomá la última sesión
rag stats                          # ver estado del índice
rag capture "idea suelta"          # apuntá algo al Inbox del vault
rag morning                        # brief del día ahora
rag digest                         # brief semanal ahora
```
