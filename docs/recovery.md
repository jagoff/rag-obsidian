# Recovery — qué hacer cuando algo se rompe

Procedures para los 7 modos de falla más probables del RAG. Pensados como **runbook**: cada bloque arranca con el síntoma, después la causa, después el fix verificado.

> Para diagnóstico exploratorio (síntomas variados) ver [`docs/problemas-comunes.md`](./problemas-comunes.md). Para tabla de troubleshooting técnico de una línea ver el [README §Troubleshooting](../README.md#troubleshooting). Este doc cubre **escenarios destructivos** (data loss, daemons rotos, dirs llenos) — el problema ya pasó y necesitás recover.

---

## 1. Borraste accidentalmente `state.db` o `telemetry.db`

**Síntoma**: `rag query` tira `sqlite3.OperationalError: no such table` al loguear o consultar telemetría. El corpus está intacto pero el dashboard / `rag log` / `rag feedback status` están vacíos.

**Causa**: una rm errónea contra `~/.local/share/obsidian-rag/ragvec/state.db` o `telemetry.db` (típicamente al limpiar `*.db-shm`/`*.db-wal` con un glob demasiado agresivo).

**Fix**: borrar también los siblings (`-shm`, `-wal`) si quedaron, y dejar que `rag index` recree los schemas idempotentes.

```bash
# Asegurate de que no queden los WAL files huérfanos
rm -f ~/.local/share/obsidian-rag/ragvec/state.db-shm
rm -f ~/.local/share/obsidian-rag/ragvec/state.db-wal
rm -f ~/.local/share/obsidian-rag/ragvec/telemetry.db-shm
rm -f ~/.local/share/obsidian-rag/ragvec/telemetry.db-wal

# Si querés empezar de cero (también borra lo que sí estaba):
rm -f ~/.local/share/obsidian-rag/ragvec/{state,telemetry}.db

# rag index recrea ambos automáticamente con DDL idempotente
rag index
```

**Qué se pierde**:
- `rag_queries` — log completo de queries (típico ~3 meses si corre `rag maintenance` semanal).
- `rag_behavior` — opens/clicks/copies de la UI web.
- `rag_feedback` — los 👍/👎 del usuario, **incluido el progreso hacia el gate GC#2.C** (mín. 20 correctives para disparar fine-tune).
- `rag_score_calibration` — la calibración isotonic per-source.
- `rag_ambient` — log estructurado del ambient hook (`whatsapp_sent`, `wikilinks_applied`, etc.).
- `rag_wa_scheduled_messages` — **mensajes WhatsApp programados pendientes** (si los había → no se mandan).
- `rag_response_cache` — semantic cache de respuestas (regenerable, costo: las primeras queries vuelven al pipeline lento).

**Qué se conserva**:
- El corpus se reconstruye desde el vault — `rag index` reembebe los chunks que detecta nuevos (`hash != stored_hash`). En un vault de ~5000 chunks tarda 10-30 min según modelo y Mac.
- `~/.config/obsidian-rag/vaults.json` (registry multi-vault) NO está en `state.db`.
- Las sessions (`~/.local/share/obsidian-rag/sessions/<id>.json`) son archivos sueltos — sobreviven.

**Decisión consciente**: NO hay backup automático de telemetry porque es regenerable. Si querés persistir, scheduleá un `rsync` propio a algún disco externo.

---

## 2. El daemon `wa-scheduled-send` no está enviando mensajes programados

**Síntoma**: programaste un mensaje de WhatsApp via la UI o el endpoint, llegó la hora, **no salió**. El dashboard muestra el row en `pending` con `attempt_count > 0` o `attempt_count = 0` (el worker ni lo pickeó).

**Causa más común**: el plist `com.fer.obsidian-rag-wa-scheduled-send` no está cargado en launchd. Ocurre típicamente después de un `git pull` que agregó la feature pero no se corrió `rag setup` (anti-patrón documentado en [`CLAUDE.md §Features con daemons`](../CLAUDE.md)).

**Diagnóstico**:

```bash
# ¿Está cargado?
launchctl list | grep wa-scheduled-send
# Si NO aparece, ese es el problema. Ver fix abajo.

# ¿Está corriendo y arrojó algo recientemente?
tail -50 ~/.local/share/obsidian-rag/wa-scheduled-send.log
tail -50 ~/.local/share/obsidian-rag/wa-scheduled-send.error.log

# ¿Qué hay en la cola?
sqlite3 ~/.local/share/obsidian-rag/ragvec/state.db \
  "SELECT id, scheduled_for_local, contact_name, status, attempt_count
     FROM rag_wa_scheduled_messages
     WHERE status = 'pending' ORDER BY scheduled_for_utc LIMIT 20;"
```

**Fix**: reinstalar con bootout + bootstrap. `kickstart -k` solo re-lanza el process; si el plist nunca estuvo cargado, hay que hacer bootstrap.

```bash
# Si el plist existe pero no está cargado:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-wa-scheduled-send.plist

# Si querés forzar full refresh (releer env vars del plist):
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-wa-scheduled-send.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-wa-scheduled-send.plist

# Forzar un tick inmediato para validar que arranca:
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-wa-scheduled-send
tail -f ~/.local/share/obsidian-rag/wa-scheduled-send.log

# Si el plist NO existe, regenerarlo:
rag setup
```

**Si el bridge está caído** (whatsapp-listener corriendo pero no logueado): el worker intenta el envío, falla, incrementa `attempt_count` y deja el row `pending`. Tras 5 reintentos pasa a `failed` y deja de pickearse. Para recuperar: arreglar el bridge y resetear los `failed` a `pending`:

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/state.db \
  "UPDATE rag_wa_scheduled_messages SET status='pending', attempt_count=0
     WHERE status='failed' AND scheduled_for_utc > strftime('%s','now') - 3600;"
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-wa-scheduled-send
```

Política Mac dormida: el worker manda igual los mensajes que se atrasaron por sleep, marca `sent_late` con `delta_minutes`. NO se cancela el envío por estar tarde.

---

## 3. El web server está colgado (lentitudes, freezes, 504s)

**Síntoma**: la PWA en `localhost:8765/chat` tarda 30+ segundos en responder, o tira 504 timeout, o la barra de carga se queda quieta. El SSE no llega.

**Causa más común**: Ollama liberó algún modelo y la próxima query lo está cargando cold (~9s para command-r en M3 Max + ~5s para bge-m3). En segundo lugar: el reranker cayó a CPU. En tercer lugar: el FastAPI process colgó por un deadlock o GIL contention en sqlite-vec.

**Diagnóstico**:

```bash
# ¿Ollama está respondiendo?
ollama ps         # qué modelos están cargados ahora
ollama list       # qué modelos hay disponibles
curl -m 5 http://localhost:11434/api/tags    # health check del daemon

# ¿El web está vivo?
curl -m 5 http://localhost:8765/api/health
launchctl list | grep com.fer.obsidian-rag-web

# Logs recientes
tail -50 ~/.local/share/obsidian-rag/web.log
tail -50 ~/.local/share/obsidian-rag/web.error.log
```

**Fix por orden de probabilidad**:

```bash
# 1. Reiniciar el web server (re-lanza pero mantiene env del plist)
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-web

# 2. Si Ollama no responde a /api/tags, reiniciarlo:
osascript -e 'quit app "Ollama"'
sleep 2
open -a Ollama
# o, si está como brew service:
brew services restart ollama

# 3. Si después de reiniciar las queries siguen lentas (~3× normal), el reranker
#    puede haber caído a CPU en lugar de MPS:
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
# Debe imprimir True. Si imprime False:
uv pip install --upgrade torch sentence-transformers

# 4. Si el web server arranca pero el chat queda colgado en stream, la cola SSE
#    puede estar saturada. El cap es RAG_SSE_MAX_PER_IP (default 4). Cerrar
#    tabs/PWAs duplicadas en el mismo dispositivo y reintentar.
```

Si el reinicio de web + Ollama no arregla el problema, probablemente sea cold-load de un modelo recién pulled. Esperar 1-2 min más y reintentar.

---

## 4. El cloudflare tunnel está corriendo y NO querés exposición pública

**Síntoma**: el RAG está expuesto en `https://*.trycloudflare.com` o similar y querés cortarlo (sospecha de uso indebido, vas a hacer upgrade del web server con downtime, reorganizando el setup).

**Causa**: están cargados los plists `com.fer.obsidian-rag-cloudflare-tunnel` (corre `cloudflared`) y `com.fer.obsidian-rag-cloudflare-tunnel-watcher` (`tail -F` del log + `pbcopy` de la URL nueva). Bootout total apaga ambos sin desinstalar — los plists quedan en disco para `bootstrap` futuro.

**Fix**:

```bash
# Bootout primero el watcher (tail -F sobre el log del tunnel)
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel-watcher.plist

# Después el tunnel
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel.plist

# Verificar que ya no están
launchctl list | grep cloudflare        # debe devolver vacío
pgrep -lf cloudflared                    # debe devolver vacío

# Si querés desactivarlo permanentemente (no se recargue con un `rag setup`):
mv ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel.plist{,.disabled}
mv ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel-watcher.plist{,.disabled}
```

Para volver a activar más tarde: `launchctl bootstrap gui/$(id -u) <plist>` (ó renombrar de vuelta sin `.disabled` si los moviste).

> El web server **sigue escuchando en `127.0.0.1:8765`** después del bootout. Solo se corta el túnel público — no la PWA local ni el chat desde la mac/iPhone en LAN (si tenés `OBSIDIAN_RAG_ALLOW_LAN=1`).

---

## 5. Se llenó el dir `chat-uploads/` (>1GB)

**Síntoma**: `du -sh ~/.local/share/obsidian-rag/chat-uploads/` reporta varios GB. El user subió muchas imágenes al chat (drag & drop) y nunca se limpiaron.

**Causa**: el endpoint `/api/chat/upload-image` guarda con naming hash-based en `~/.local/share/obsidian-rag/chat-uploads/`. **Antes del audit 2026-04-25 R2-Security #6 NO había TTL** — el dir crecía sin bound. Post-audit, `rag maintenance` corre cleanup automático con TTL `RAG_CHAT_UPLOADS_TTL_DAYS=30` (override via env var).

**Fix**:

```bash
# Opción 1 (recomendada): ejecutar maintenance que ya incluye el cleanup
rag maintenance                     # corre dry-run + reporta qué borraría
rag maintenance --as-json | jq .chat_uploads_cleaned   # ver el resultado

# Opción 2: manual con find si querés un TTL distinto al default
find ~/.local/share/obsidian-rag/chat-uploads -type f -mtime +30 -delete
du -sh ~/.local/share/obsidian-rag/chat-uploads/

# Opción 3: nuclear (borrar todo, perdés conversaciones con imágenes recientes)
rm -rf ~/.local/share/obsidian-rag/chat-uploads/
mkdir -p ~/.local/share/obsidian-rag/chat-uploads/
```

**Override del TTL** (si querés conservar más historia):

```bash
# Ad-hoc:
RAG_CHAT_UPLOADS_TTL_DAYS=90 rag maintenance

# Persistente: editar el plist com.fer.obsidian-rag-maintenance,
# agregar la env var, bootout + bootstrap.
```

El cleanup es defensive — si algo explota dentro de `_cleanup_chat_uploads()`, el resto del `rag maintenance` sigue (se loguea como `chat_uploads_cleaned_error`).

---

## 6. Tests fallan después de un `git pull`

**Síntoma**: `pytest tests/ -q` después de un pull fresh tira errores como:

- `sqlite3.OperationalError: database is locked`
- `sqlite3.OperationalError: no such table: rag_queries`
- Tests de telemetría que pasaban antes ahora fallan en CI local pero pasan en remote.

**Causa más común**: el WAL del telemetry.db quedó corrupto / lockeado por un proceso anterior (típicamente el web server o un `rag query` que crasheó dejando los `-shm`/`-wal` orphans). El test entra al schema, encuentra estado inconsistente, falla.

Causa secundaria: el schema de la collection bumpeó (v10→v11) y los tests usan un fixture que asumía v10.

**Fix**:

```bash
# Limpiar WAL/SHM huérfanos del telemetry.db (cheap, recreables)
rm -f ~/.local/share/obsidian-rag/ragvec/telemetry.db-shm
rm -f ~/.local/share/obsidian-rag/ragvec/telemetry.db-wal
rm -f ~/.local/share/obsidian-rag/ragvec/state.db-shm
rm -f ~/.local/share/obsidian-rag/ragvec/state.db-wal

# Limpiar pycache (a veces tests cachean fixtures)
rm -rf tests/__pycache__/ rag/__pycache__/

# Re-correr la suite
.venv/bin/python -m pytest tests/ -q
```

Si después de eso siguen fallando, chequear:

```bash
# ¿Hay un proceso real que mantiene el lock?
lsof ~/.local/share/obsidian-rag/ragvec/telemetry.db
lsof ~/.local/share/obsidian-rag/ragvec/state.db

# Si el web server está corriendo, los tests con autouse=False sobre la DB
# real pueden ver locks. Soluciones:
#   (a) parar el web server: launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
#   (b) correr los tests con env apuntando a un tmp:
HOME=/tmp/test-rag .venv/bin/python -m pytest tests/test_X.py -q
```

Si el schema de la collection cambió, hay tests que asumen `v11` y fallarán contra `v12`. Solución: reset y reindex:

```bash
rag index --reset          # rebuild colecciones a la versión actual del código
.venv/bin/python -m pytest tests/ -q
```

---

## 7. Necesitás restaurar de backup

**Síntoma**: borraste algo importante (vault entero, una nota, las DBs juntas) y querés volver atrás.

**Diagnóstico — qué hay y qué no hay**:

| Recurso | Backup | Cómo restaurar |
|---|---|---|
| Vault (notas, frontmatter) | **Sí** — iCloud Drive auto-sync de Apple | iCloud Web (`icloud.com/iclouddrive`) → sección "Recently Deleted" (30 días). Para notas individuales en Obsidian: `Cmd+Z` si es muy reciente, o pedir al peer del vault (otro device) que sincronice de vuelta. |
| `~/.local/share/obsidian-rag/ragvec/state.db` | **No** — decisión consciente | Reconstruir con `rag index` (corpus) + se pierde telemetría/feedback. Ver escenario #1. |
| `~/.local/share/obsidian-rag/ragvec/telemetry.db` | **No** | Idem. |
| `~/.local/share/obsidian-rag/sessions/<id>.json` | **No** (archivos sueltos) | Si el filesystem no fue wipeado, Time Machine los tiene. Sino, perdidos. |
| `~/.config/obsidian-rag/vaults.json` | **No** — pero es trivial reescribir | `rag vault add <name> <path>` para cada vault que tenías; `rag vault use <name>` para el activo. |
| `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist` | **No** — regenerables | `rag setup` los recrea idempotentemente. |
| Plists de Cloudflare tunnel | **No** | Reinstalar manualmente o regenerar (ver [CLAUDE.md §Cloudflare tunnel](../CLAUDE.md)). |
| Configs ad-hoc (`ambient.json`, `auto_index_state.json`, etc.) | **No** | Los flujos default los regeneran (`rag ambient` para config nueva, watcher recrea state al primer save). |

**Política**:

- El **vault SÍ** está en iCloud (backup automático Apple) — el corpus es regenerable desde ahí.
- Las **DBs locales NO** tienen backup automático — decisión consciente porque la telemetría es regenerable (vuelve a poblarse con el uso) y el costo de mantener un backup criptográficamente seguro de info que incluye paths privados del vault no compensa.
- Si querés cambiar esa política: scheduleá un `rsync` o un `Time Machine` selectivo (excluyendo `*.db-wal`/`*.db-shm` para evitar transient state).

---

## Si nada de lo anterior funcionó

1. Mirar los logs de error de los servicios:
   ```bash
   tail -100 ~/.local/share/obsidian-rag/*.error.log
   ```
2. Correr el comando que falla con `RAG_DEBUG=1`:
   ```bash
   RAG_DEBUG=1 rag <comando>
   ```
3. Pedir el breakdown del retrieve:
   ```bash
   RAG_RETRIEVE_TIMING=1 rag query "tu pregunta"
   ```
4. Health dashboard:
   ```bash
   rag health --since 24
   ```
5. Como último recurso: `git reset --hard <último-commit-que-andaba>` y reportar.
