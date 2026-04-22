# Automatizaciones (launchd)

El sistema instala servicios en segundo plano que corren solos, sin que tengas que acordarte. Los maneja `launchd`, el scheduler de macOS (equivalente a cron pero mejor).

**Cómo se instalan todos**:

```bash
rag setup             # instalar/recargar (idempotente)
rag setup --remove    # desinstalar todos
```

Los plists se escriben a `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist`.

## Diagrama mental

```
                     ┌─────────────────────────┐
                     │     LAUNCHD (macOS)     │
                     └─────────────┬───────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐       ┌────────▼────────┐      ┌──────▼──────┐
    │ siempre     │       │ por horario     │      │ por intervalo│
    │ corriendo   │       │ (hora del día)  │      │ (cada N seg) │
    └─────────────┘       └─────────────────┘      └──────────────┘
         │                         │                       │
         ▼                         ▼                       ▼
      watch               morning (lun-vie 7am)     ingest-gmail (1h)
      serve               today   (lun-vie 22pm)    ingest-calendar (1h)
      web                 digest  (dom 22pm)        ingest-reminders (1h)
                          archive (día 1, 23pm)     ingest-whatsapp (15min)
                          consolidate (lun 6am)     wa-tasks (30min)
                          emergent (vie 10am)
                          patterns (dom 20pm)
                          maintenance (diario 4am)
                          online-tune (diario 3:30am)
```

---

## La tabla completa

| Servicio | Qué hace | Cuándo corre |
|---|---|---|
| **watch** | Re-indexa el vault cuando guardás notas en Obsidian | Todo el tiempo (KeepAlive) |
| **serve** | Servidor HTTP persistente para bots/integraciones | Todo el tiempo (KeepAlive) |
| **web** | Servidor web del chat UI | Todo el tiempo (RunAtLoad) |
| **morning** | Brief matutino (`rag morning`) | Lun-Vie 7:00 |
| **today** | Cierre del día (`rag today`) | Lun-Vie 22:00 |
| **digest** | Resumen semanal (`rag digest`) | Dom 22:00 |
| **archive** | Archiva notas muertas (`rag archive --apply`) | Día 1 de cada mes, 23:00 |
| **consolidate** | Promueve clusters de Inbox a PARA (`rag consolidate --apply`) | Lun 6:00 |
| **emergent** | Detecta temas emergentes en queries | Vie 10:00 |
| **patterns** | Alerta si un reason de feedback domina | Dom 20:00 |
| **maintenance** | Mantenimiento integral (`rag maintenance`) | Diario 4:00 |
| **online-tune** | Re-tunea pesos del ranker con behavior reciente (`rag tune --apply`) | Diario 3:30 |
| **ingest-gmail** | Sincroniza Gmail al índice | Cada 1h + al boot |
| **ingest-calendar** | Sincroniza Google Calendar al índice | Cada 1h + al boot |
| **ingest-reminders** | Sincroniza Apple Reminders al índice | Cada 1h + al boot |
| **ingest-whatsapp** | Sincroniza WhatsApp al índice | Cada 15 min + al boot |
| **wa-tasks** | Extrae tareas/compromisos de WhatsApp | Cada 30 min |

Total: **17 servicios**.

---

## Comandos útiles para ver qué pasa

### Ver cuáles están cargados
```bash
launchctl list | grep obsidian-rag
```

### Mirar los logs en vivo
```bash
tail -f ~/.local/share/obsidian-rag/watch.log
tail -f ~/.local/share/obsidian-rag/morning.log
tail -f ~/.local/share/obsidian-rag/digest.log
# etc — cada servicio escribe a su propio <nombre>.log + <nombre>.error.log
```

Para ver todos al mismo tiempo:

```bash
tail -f ~/.local/share/obsidian-rag/*.log
```

### Forzar que un servicio corra ya
```bash
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-morning
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-watch
# etc
```

### Descargar y volver a cargar un servicio (si quedó raro)
```bash
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-watch.plist
launchctl load   ~/Library/LaunchAgents/com.fer.obsidian-rag-watch.plist
```

### Ver qué error tuvo un servicio
```bash
cat ~/.local/share/obsidian-rag/watch.error.log
tail -n 100 ~/.local/share/obsidian-rag/morning.error.log
```

### Parar solo un servicio temporalmente
```bash
launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-watch.plist
# Para volver a activarlo:
launchctl load ~/Library/LaunchAgents/com.fer.obsidian-rag-watch.plist
```

---

## Cómo funcionan los distintos tipos de trigger

### KeepAlive — "siempre corriendo"

```
  systema booteó
       │
       ▼
  launchd lanza el proceso
       │
       ▼
  si el proceso muere ──▶ launchd lo relanza solo
```

Usado por: `watch`, `serve`, `web`.

### StartCalendarInterval — "por horario"

```
  mira el reloj
       │
       ▼
  ¿es la hora configurada?  (ej: Hour=7, Weekday=1-5)
       │
       └── sí ──▶ corre el comando una vez
                  ▼
                  termina cuando termina el comando
                  ▼
                  espera hasta el próximo horario
```

Usado por: `morning`, `today`, `digest`, `archive`, `consolidate`, `emergent`, `patterns`, `maintenance`, `online-tune`.

**Weekday en launchd**: 0=domingo, 1=lunes, 2=martes… 6=sábado.

### StartInterval — "cada N segundos"

```
  systema booteó
       │
       ▼ (RunAtLoad=true → corre ya)
  corre el comando
       │
       ▼
  espera N segundos
       │
       ▼
  corre otra vez
       │
       └─▶ loop infinito
```

Usado por los ingesters: `ingest-gmail` (3600s), `ingest-calendar` (3600s), `ingest-reminders` (3600s), `ingest-whatsapp` (900s), `wa-tasks` (1800s).

---

## Qué esperar de cada servicio

### watch
Se queda silencioso el 99% del tiempo. Cuando guardás una nota, en 3 segundos (debounce) la re-indexa. Si hay un problema, mirá `watch.error.log`.

### serve / web
Mantienen los modelos calientes (bge-m3, qwen2.5:3b, reranker) para que las queries del bot/web respondan rápido. Consumen ~8-10 GB de RAM todo el tiempo.

### morning (7am lun-vie)
Genera un archivo nuevo en `tu-vault/05-Reviews/YYYY-MM-DD.md`. Si tu Mac estuvo apagado a esa hora, no corre (no hay "catch-up"). Lo podés correr a mano: `rag morning`.

### today (22pm lun-vie)
Genera `tu-vault/05-Reviews/YYYY-MM-DD-evening.md`. Si no hubo actividad en el día, imprime "sin actividad hoy" y no escribe nada.

### digest (domingo 22pm)
Genera `tu-vault/05-Reviews/YYYY-WNN.md` (semana ISO). Es un brief narrativo largo.

### archive (día 1 de cada mes, 23pm)
Corre `rag archive --apply --gate 20` — si detecta más de 20 notas a archivar, no lo aplica sin `--force`. Escribe un reporte a `05-Reviews/YYYY-MM-archive.md`.

### consolidate (lunes 6am)
Agrupa conversaciones del Inbox semánticamente similares y las promueve a PARA. Escribe notas nuevas en `01-Projects/` o `03-Resources/`.

### emergent (viernes 10am)
Busca temas consultados repetidamente que no tienen una nota cubriéndolos. Pingea WhatsApp con `/capture` prefilled.

### patterns (domingo 20pm)
Si una razón de feedback negativo domina los últimos 30 eventos, pingea WhatsApp con sugerencia de acción.

### maintenance (diario 4am)
Reindex incremental + limpia sesiones viejas + rota logs + detecta dead notes. Bajo overhead.

### online-tune (diario 3:30am)
Re-tunea los pesos del ranker con los events de behavior de los últimos 14 días. Guarda backup antes. Rollback: `rag tune --rollback`.

### ingest-* (cada 1h / 15min)
Sincronizan datos externos al índice. Cada uno escribe a su `.log` respectivo. El de WhatsApp corre más frecuente (15 min) porque los chats cambian más rápido.

### wa-tasks (cada 30min)
Escanea mensajes de WhatsApp no respondidos y extrae acciones al Inbox (`00-Inbox/WA-YYYY-MM-DD.md`).

---

## Si algo no arranca

1. Ver si está cargado:
   ```bash
   launchctl list | grep obsidian-rag | grep <nombre>
   ```
2. Ver el error:
   ```bash
   cat ~/.local/share/obsidian-rag/<nombre>.error.log
   ```
3. Intentar correr el comando a mano con las mismas env vars y ver qué pasa:
   ```bash
   rag <nombre>
   ```
4. Recargar el servicio:
   ```bash
   rag setup   # reinstala todo
   ```

Más tips en [problemas-comunes.md](./problemas-comunes.md).
