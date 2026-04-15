# web

UI web mínima para `rag chat`. Dark mode, sin build step, SSE streaming.
Espeja el pipeline del CLI (`multi_retrieve` → command-r → sources)
reusando las primitivas de `rag.py`.

## Correr

```bash
# desde el root del proyecto
uv pip install fastapi uvicorn
.venv/bin/python web/server.py
# → http://127.0.0.1:8765
```

Las sesiones se guardan en el mismo store que el CLI
(`~/.local/share/obsidian-rag/sessions/`) con prefijo `web:<uuid>`.

## Alcance actual

- Chat multi-turno con streaming token-a-token.
- Sources chips clickeables (abren `obsidian://open?file=…`).
- Persistencia de sesión vía `localStorage` — refrescá y seguís el hilo.

## Pendiente

- Intents (`/save`, `/reindex`, `/links`) — por ahora sólo CLI.
- Counter-evidence (`--counter`).
- Feedback 👍/👎.
- Auth (el server escucha en `127.0.0.1` — no exponer a red sin proxy).
