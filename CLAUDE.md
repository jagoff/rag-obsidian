# CLAUDE.md

Local RAG over an Obsidian vault. Single-file: `rag.py` (54.5k lines as of 2026-04-27 — drift +56% vs prior 32.7k snapshot, package-split is now an open discussion not a settled "no") + `mcp_server.py` (thin wrapper, 662 lines) + `web/` (FastAPI `web/server.py` 17.6k lines + ~7.7k JS/HTML/CSS) + `tests/` (6,072 tests, 337 files).

Entry points (both installed via `uv tool install --editable '.[entities]'` — incluye el extra `entities` para que gliner se instale en el uv tool venv y la feature de entity-aware retrieval quede activa de fábrica; sin el extra los 5 ingesters loggean `dep \`gliner\` not available` cada corrida y la feature corre desactivada silenciosamente):
- `rag` — CLI for indexing, querying, chat, productivity, automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Fully local: Sqlite-vec + Ollama + sentence-transformers. **Exception**: Gmail + Calendar cross-source ingesters (Phase 1.b/c, pending) use OAuth Google via the Claude harness MCP — user override 2026-04-20, see `docs/design-cross-source-corpus.md §10.6`. WhatsApp + Reminders stay local (bridge SQLite + EventKit).

## Idioma — español rioplatense (Argentina) por default

**Toda comunicación con el usuario va en español rioplatense (Argentina).** Esto aplica a:

- Respuestas en chat / terminal del agente.
- Mensajes de commit (subject + body).
- Comentarios nuevos en código que el agente escribe.
- Notas de Obsidian que el agente crea bajo `99-AI/...` o `00-Inbox/`.
- Output de scripts y daemons que produce el agente (briefs WA/morning/evening, summaries, etc.).
- Mensajes de error visibles al usuario.

**Voseo argentino**, no tuteo neutro: "vos podés" / "fijate" / "agarrá" / "mirá" — no "tú puedes" / "fíjate" / "agarra" / "mira". Vocabulario rioplatense natural ("dale", "tranqui", "che", "laburar") cuando aporta claridad sin sonar forzado.

**Excepciones que SÍ van en inglés** (no traducir):

- Tecnicismos estándar de software: `commit`, `pull request`, `branch`, `rebase`, `endpoint`, `payload`, `cache`, `streaming`, `SSE`, `LoRA`, `embedding`, `chunk`, `daemon`, `plist`, `launchctl`, `query`, `index`, `vault`, `scope`, `re-rank`, `retrieve`, `tune`, `eval`, etc.
- Nombres propios de productos / herramientas / paquetes: `Obsidian`, `FastAPI`, `Chart.js`, `Ollama`, `sqlite-vec`, `Devin`, `Claude Code`, `iCloud`, `LaunchAgents`.
- Stack traces, output literal de comandos, snippets de código, JSON de respuesta — son output mecánico, no se traducen.
- Output de `git`, `gh`, `pytest`, `uv`, etc. — se cita literal.
- Si el usuario explícitamente escribe en otro idioma o pide la respuesta en otro idioma, contestar en ese idioma.

**Lo que NO se permite** (problema observado y razón de existir esta regla):

- Respuestas o partes de respuestas en **portugués** ("você", "estamos fazendo", "esses", "isso", "obrigado") — el modelo a veces se desliza al portugués cuando el contexto tiene mucho input portugués cerca, especialmente con palabras parecidas. Si te encontrás escribiendo "esse"/"essa"/"isso"/"você" en una respuesta para el user, es bug — corregir a español rioplatense.
- Respuestas o partes de respuestas en **chino**, **japonés**, **coreano** o cualquier otro idioma no-español que no haya pedido explícitamente el user.
- Español neutro plano tipo manual de microondas ("usted puede ejecutar el siguiente comando") en vez de rioplatense — suena impersonal y no es lo que pide el user.

**Verificación rápida antes de mandar la respuesta**: si tu output contiene "você", "obrigad", "esses", "essa", "isso", "sim/não", caracteres CJK (汉字, 日本語, 한국어), o "tú" / "ustedes" en un contexto donde correspondería "vos" / "ustedes-rioplatense" — algo está roto, corregilo antes de enviar.

Esta regla NO cambia entre sesiones — es el comportamiento default. El usuario sólo escribe en inglés cuando cita doc inglesa o nombre técnico inglés; eso no es señal para cambiar de idioma.

## Agent dispatch rule

**Any task that will edit ≥3 files MUST go through `pm` first.** No preguntas — invocar directamente:

```
Agent(subagent_type: "pm", prompt: "<goal + context + ruled-out + invariants at risk>")
```

The PM returns a dispatch plan (tasks, dependencies, parallel-safe flags, risks, validation). The main session executes the plan by spawning the named agents in the prescribed order — never silently skips PM and improvises.

Tasks that touch ≤2 files go directly to the owning agent (`rag-retrieval`, `rag-llm`, `rag-brief-curator`, `rag-ingestion`, `rag-vault-health`, `rag-integrations`, `developer-{1,2,3}`). Roster + ownership lives in `.claude/agents/README.md`.

When peers are active (`mcp__claude-peers__list_peers(scope: "repo")` returns >1), even ≤2-file tasks may need PM coordination — flag overlapping zones before editing.

### Custom agent profiles requieren reload de la sesión

Los profiles definidos en [`.claude/agents/*.md`](.claude/agents/) se cargan **una sola vez al iniciar la sesión** de Devin / Claude Code. Si creás un agent nuevo durante una sesión activa (ej. agregás `.claude/agents/foo.md`), **esa sesión NO lo ve** — la lista de profiles disponibles se compiló al boot y no se refresca.

Síntoma: invocar `run_subagent(profile="foo", ...)` (Devin) o `Agent(subagent_type: "foo", ...)` (Claude Code) rebota con "Subagent failed to start" o "subagent type unknown".

Workarounds en orden de preferencia:

1. **Reabrir la sesión**: `Ctrl-D` (o `exit`) y volver a lanzar `devin` / `claude`. La nueva sesión escanea `.claude/agents/` y carga el profile. Es lo correcto a largo plazo — siempre que no estés en medio de algo importante.
2. **Inyectar el system prompt inline en `subagent_explore` o `subagent_general`**: pegás el body del nuevo `AGENT.md` como brief en `run_subagent(profile="subagent_explore", task="<system prompt completo>")`. Output equivalente, pero perdés el archivo como source of truth — cada invocación re-pegás el prompt.
3. **Adelantarse**: si vas a crear un agent nuevo durante una task, creá el archivo **antes** de empezar la sesión y arrancá Devin después. Sirve cuando el flujo es predecible.

Este gotcha aplica también a **skills** custom en `.devin/skills/` y `.claude/skills/`. Hooks en `.devin/config.json` SÍ se refrescan en runtime (al menos en versiones recientes), no hay que reabrir.

Aprendido el 2026-04-25 cuando creé `rag-perf-auditor.md` durante una sesión y `run_subagent(profile="rag-perf-auditor", ...)` rebotó.

## Auto-pull + commit + push rule

**Regla universal: cuando termino ALGO — feature, fix, refactor, limpieza, ajuste — el ciclo es siempre `git pull → git commit → git push origin master`. Sin preguntar.**

1. **Pull primero** (`git pull --rebase origin master` si hay cambios locales sin push, o `git pull` si el working tree está limpio). Evita que otro proceso — el user en su Mac, un cron, un Devin en paralelo — genere merge conflicts silenciosos.
2. **Commit** con un mensaje **completo y en palabras simples** de lo que hice, no jerga técnica seca. Formato esperado:
   - **Subject line**: `tipo(scope): resumen corto en español rioplatense (1 línea, ~70 chars)`
   - **Cuerpo**: explicación en párrafos cortos que cualquier humano (no solo yo en 6 meses) pueda leer: **qué cambié**, **por qué** lo cambié, **cómo lo medí** si aplica, y **cómo revertir** si rompe algo. Evitar tecnicismos gratuitos — si uso un término especializado (LoRA, RRF, idle-unload), que quede claro qué es en contexto.
   - **Trailer estándar Devin** al final:
     ```
     Generated with [Devin](https://cli.devin.ai/docs)
     Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
     ```
3. **Push** (`git push origin master`) inmediatamente después del commit. Si el pull del paso 1 hizo rebase y hay conflicts, los resolvés vos sin preguntar — nunca push con conflicts sin resolver.

Si los tests fallan o el build rompe, **NO commiteás** — arreglás primero.

Excepciones obvias: tareas exploratorias (investigar, responder preguntas, revisar diffs), cambios que el usuario explicitamente pidió no commitear, trabajo a medio camino (fix parcial + deferred-follow-up). Ante duda genuina, commit pero no push.

Esta regla NO cambia — es el comportamiento default para siempre. Si el user dice "hace el commit", "commit + push", "cerrá esto", o cualquier cosa que signifique "terminaste → guardá", ejecutás el ciclo completo pull → commit → push sin confirmar cada paso.

### Gotcha: commits locales en `master` se pushean solos (auto-pusher en paralelo)

**Cualquier commit que hagas en `master` localmente aparece en `origin/master` en segundos, aunque no llames `git push` explícitamente.** Observado en sesiones del 2026-04-25 donde agentes intentaron commits "experimentales" para evaluar y revertir, y los commits ya estaban en remote antes del rollback.

Causa raíz no confirmada al 100% pero hipótesis principal: **otra sesión de Devin/Claude Code corriendo en paralelo** (el repo tiene MCP `claude-peers` activo, [`.claude/skills/claude-peers-mcp/`](.claude/skills/claude-peers-mcp/), que permite a varias instancias coordinarse via broker). Esa otra sesión detecta los commits locales y los pushea con su propio auto-pull+commit+push rule (la de arriba). NO se encontró cron, git hook, FS watcher, ni `git push` literal en plists del repo o `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist` — descartado todo eso como culpable directo.

**Implicaciones operativas**:

1. **Asumí que `git commit` en `master` = `git push origin master` casi inmediato.** No hay ventana para "hago el commit, lo evalúo localmente, y si no me gusta lo deshago con `git reset HEAD~1`". Para cuando reseteaste, ya está en remote.
2. **Si querés experimentar sin pushear → branch dedicada.** `git checkout -b experimental/<slug>` y commitea ahí. Las ramas no-master no las agarra el auto-pusher (no se confirmó pero es la asunción razonable hasta que se demuestre lo contrario).
3. **Si pushiaste algo malo y no podés force-push** (la deny-list de [`.devin/config.json`](.devin/config.json) bloquea `git push --force`): único camino es `git revert <sha> && git push origin master`. Acepta que el commit malo + el revert quedan ambos en el log para siempre.
4. **No se puede "desactivar temporalmente"** el auto-pusher desde esta sesión — es una sesión paralela con sus propias reglas. Si necesitás coordinarlo: usar [`mcp__claude-peers__send_message`](.claude/skills/claude-peers-mcp/SKILL.md) para pedirle a la otra instancia que pause sus pushes, o avisarle al user humano para que cierre la otra sesión.

**Lección**: pensá dos veces antes de commitear en master. El commit es público en segundos.

Aprendido el 2026-04-25 durante una sesión de fix de CI donde 2 commits ("ruff cleanup masivo" + "destrabar 40 tests") aparecieron en remote antes de que la sesión activa los pushiera. El reflog mostraba `pull: Fast-forward` post-commit en lugar de `push:`, confirmando que el push vino de otra fuente.

## Autonomous mode — empezar y terminar una feature sin interrupciones

Devin for Terminal tiene 4 [permission modes](https://docs.devin.ai/reference/permissions): **Normal** (default, pide permiso para writes + exec), **Accept Edits** (auto-aprueba edits dentro del workspace), **Bypass** (auto-aprueba TODO sin prompts), y **Autonomous** (Bypass + sandbox OS). Este proyecto está configurado para minimizar interrupciones aun en modo Normal — y cuando querés zero prompts absolutos, hay un interruptor global.

**Dos niveles de autonomía disponibles:**

1. **`.devin/config.json` — permissions pre-aprobadas por default** (siempre activas)
   - Allow-list (~80 reglas): todo el workflow normal del RAG auto-aprobado — `git *`, `rag *`, `uv *`, `pytest`, `sqlite3`, `launchctl *` (incluye `remove`/`unload`/`kickstart`/`bootout`/`bootstrap` para que los scripts de rotación de daemons no pidan permiso), `tail/head/cat/ls/find/grep/rg/awk/sed` (observabilidad), `.venv/bin/python`, `ollama ls/list/ps/show` (solo read), `curl localhost`, writes dentro del repo (`Write(**)` relativo al cwd).
   - Deny-list (6 reglas): cosas verdaderamente irrecuperables sin red — `sudo`, `git reset --hard`, `git push --force` (`-f`), `git branch -D`. Nota histórica: (a) `launchctl remove/unload` estaban acá hasta 2026-04-24 — los sacamos porque el flujo normal de "refresh el web server después de cambiar el plist o el código" requiere un unload+load, bloquearlos forzaba pedir permiso cada vez, y el riesgo es bajo (si desactivo un daemon por error, lo vuelvo a cargar con `launchctl load` y listo). (b) `rm -rf` y `rm -fr` también estaban acá hasta 2026-04-28 — los sacamos porque pedirme permiso para cada cleanup de tmpdir/test rompía flow constantemente, el repo vive en `origin/master` (recuperable con `git clone` + un `git pull` de cualquier máquina), y el vault sigue protegido por la ask-rule. El blast radius real queda acotado a "trabajo no committed del momento" — aceptable.
   - Ask-list (7 reglas): operaciones sensibles que SIEMPRE preguntan aunque estemos en Bypass mode — `.env*`, `~/.ssh/`, `~/.aws/`, `~/.config/devin/`, writes al vault iCloud real, fetch a OpenAI/Anthropic APIs (el proyecto es local-first; si alguna vez tocamos esas URLs es un bug). `rm`/`rm -rf` ya NO está en ask desde 2026-04-28 — pasó a allow.
   - Resultado: en modo Normal, ~todo el flujo habitual corre sin pedir permiso. Las pausas que quedan son el handful de ops sensibles de la ask-list.

2. **Bypass mode — cero prompts absolutos** (explícito, por sesión)
   - Arrancar Devin con la flag: `devin --permission-mode bypass`
   - O durante una sesión: presionar **Shift+Tab** para alternar entre Normal → Accept Edits → Bypass → Plan (la barra de estado del terminal muestra el modo activo).
   - En Bypass mode todos los tool calls se auto-aprueban sin excepción — **salvo las reglas `deny` del `.devin/config.json`, que siempre ganan**. Por eso la deny-list está pensada como el safety net no-negociable: incluso en Bypass, no vas a hacer `sudo` ni `git push --force` por accidente. (Ojo: `rm -rf` SÍ se permite desde 2026-04-28 — el agente puede borrar archivos en bulk sin preguntar. El repo en GitHub es la red de seguridad para este caso. Si querés bloquearlo de nuevo, agregá `Exec(rm -rf)` a `deny` en [`.devin/config.json`](.devin/config.json).)
   - Las reglas `ask` también siguen preguntando en Bypass (por diseño del sistema de precedencia: deny > ask > allow). Si querés que también se auto-aprueben en Bypass, moverlas a `allow` en el config.
   - Ideal para: "empezá esta feature, corré tests, commit + push, seguí con la siguiente" sin babysitting.

**Precedencia de permissions** (de mayor a menor prioridad, del doc oficial):
1. Org/team settings (si hay enterprise).
2. Session-level grants (lo que aprobaste interactivamente).
3. `.devin/config.local.json` (override local no commiteado).
4. `.devin/config.json` (este archivo, committeado al repo).
5. `~/.config/devin/config.json` (user-level, aplicado a todos los proyectos).

**Agregar overrides sin commitear** (ej. permisos extra que solo valen en tu Mac):
```bash
# ~/.config/devin/config.json o .devin/config.local.json (ignorado por git)
{
  "permissions": {
    "allow": ["Exec(docker)", "Fetch(https://*.my-company.com/*)"]
  }
}
```

**Rollback rápido si el config rompe algo**:
```bash
# Desactivar todas las permissions del proyecto:
mv .devin/config.json .devin/config.json.disabled
# O solo una regla específica: editar y quitar la línea.
```

## Zsh tab-completion (`_rag`)

Hay un completion script hand-written en [`completions/_rag`](completions/_rag) con descriptions por subcomando, flags, sub-grupos anidados (ambient/session/vault/...), y helpers dinámicos (`_rag_vaults` lee `rag vault list`, `_rag_sessions` lee `rag session list`, `--source` abre el set cross-source como choices, `--vault` sugiere los registrados). Startup nativo de zsh (~10-50ms por Tab), no spawnea Python como haría el completion automático de Click (~350ms warm / 1s cold).

**Instalación one-shot** (una vez por máquina):

```bash
cp completions/_rag ~/.oh-my-zsh/custom/completions/_rag
rm -f ~/.zcompdump* /tmp/_zcomp*       # invalidar el cache de compinit
exec zsh                                # o abrir una nueva terminal
```

Cualquier dir dentro de `$fpath` sirve (no hace falta OMZ — Homebrew zsh usa `/usr/local/share/zsh/site-functions/`, macOS stock usa `/usr/share/zsh/site-functions/`).

**Regenerar tras cambiar el árbol de Click** (nuevo subcomando, nuevo flag, nuevo choice):

```bash
.venv/bin/python scripts/gen_zsh_completion.py > completions/_rag
cp completions/_rag ~/.oh-my-zsh/custom/completions/_rag
rm -f ~/.zcompdump*
exec zsh
```

El generador ([`scripts/gen_zsh_completion.py`](scripts/gen_zsh_completion.py)) camina el árbol de Click, emite `_arguments`/`_describe` nativos, detecta `click.Choice`/`click.Path`/`click.File` → actions correctas, y escapa single-quotes con el truco `'\''`. No regenera automáticamente en CI — quedó a ojo del dev que toca el CLI; si el completion queda stale, peor caso: Tab no sugiere el flag nuevo (no rompe nada).

## PWA (iOS add-to-home-screen)

El web server sirve una [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) instalable en iPhone que corre en pantalla completa (sin chrome de Safari) con splash screen custom y shell cacheado offline. El wiring está en:

- [`web/static/manifest.webmanifest`](web/static/manifest.webmanifest) — `start_url=/chat`, `display=standalone`, icons (192/512 any + maskable), shortcuts a home/chat/dashboard/learning.
- [`web/static/sw.js`](web/static/sw.js) — service worker. Estrategia: stale-while-revalidate para el shell (/, /chat, /dashboard), cache-first para `/static/**` con refresh oportunista, network-only para `/api/**` (no cacheamos streams SSE ni respuestas privadas del RAG).
- [`web/static/pwa/register-sw.js`](web/static/pwa/register-sw.js) — registra el SW desde los 3 HTML + muestra banner de "Agregar a pantalla de inicio" en iOS la primera vez (dismisseable, persiste en `localStorage`).
- FastAPI routes [`/manifest.webmanifest`](web/server.py) y [`/sw.js`](web/server.py) — servidos desde root (no desde `/static/`) porque el SW scope debe ser `/` para controlar todas las páginas; [`web/server.py:1034-1059`](web/server.py).
- [`scripts/gen_pwa_assets.py`](scripts/gen_pwa_assets.py) — genera icons + splash screens para 10 modelos de iPhone (X → 16 Pro Max) usando Pillow.

**Instalar en iPhone** (el user, no el dev):

1. Abrir [ra.ai/chat](https://ra.ai/chat) en **Safari** (no Chrome — Chrome iOS no soporta PWA install).
2. Tocar el botón **Compartir** (↑ con flechita) en la barra inferior.
3. Scrollear hasta **"Agregar a pantalla de inicio"** → tocar.
4. Confirmar el nombre ("rag") y tocar **Agregar**.
5. Cerrar Safari. El icono aparece en home screen como una app nativa.

Al abrir el icono el user ve: splash screen minimal con `rag·` centrado (el mismo logo del icon) durante el boot (~300ms), luego el chat directo, sin barra de Safari. Safe-area respetada (notch / Dynamic Island) vía `viewport-fit=cover` + `env(safe-area-inset-*)` en el CSS.

**Regenerar assets** (cuando cambie branding, o Apple saque un iPhone nuevo):

```bash
.venv/bin/python scripts/gen_pwa_assets.py [--print-html]
```

Genera 17 PNGs en `web/static/pwa/` (icons + maskable + apple-touch-icon + favicons + 10 splash screens). `--print-html` imprime el snippet de `<link rel="apple-touch-startup-image">` para pegar en los 3 HTML si se agregó/cambió un device.

**Forzar update del SW** (cuando rompe algo o querés wipe del cache en un user):

En la consola del browser (DevTools → Application → Service Workers):
```js
navigator.serviceWorker.getRegistrations().then(rs => rs.forEach(r => r.unregister()))
caches.keys().then(ks => ks.forEach(k => caches.delete(k)))
```

O bumpear `CACHE_VERSION` en [`web/static/sw.js`](web/static/sw.js) — el nuevo SW borra los caches viejos en el `activate` handler automáticamente.

**Limitaciones conocidas iOS**:
- No hay `beforeinstallprompt` en Safari → no se puede triggerear el flow de install programáticamente. Por eso mostramos el banner manual.
- Push notifications sólo desde iOS 16.4 y sólo cuando la PWA está instalada en home screen.
- Safari agresivo matando el SW (~20-30s de idle) → stale-while-revalidate es la estrategia correcta (no cache-first exclusivo).

Tests: [`tests/test_web_pwa.py`](tests/test_web_pwa.py) (9 casos: manifest mime+body+cache, SW headers+body, files-on-disk, 3 HTML wiring, shortcut routes válidas).

### Exponer la PWA al LAN (iPhone accede por IP del Mac)

`ra.ai` está mapeado a `127.0.0.1` en `/etc/hosts` + Caddy con `tls internal` → funciona sólo desde el Mac local. Para que el iPhone instale la PWA desde el mismo WiFi hay que exponer el server uvicorn al LAN:

**Dos env vars emparejadas** (ambas deben estar seteadas, o ninguna):

- `OBSIDIAN_RAG_BIND_HOST=0.0.0.0` — uvicorn bindea a todas las interfaces (default `127.0.0.1`). Ver [`web/server.py`](web/server.py) en el `__main__`.
- `OBSIDIAN_RAG_ALLOW_LAN=1` — extiende el regex de CORS a los 3 rangos privados [RFC1918](https://datatracker.ietf.org/doc/html/rfc1918): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`. Acepta tanto `http://` como `https://` (Caddy `tls internal`). Sin este flag, aunque el puerto esté accesible el browser bloquea el CORS porque el Origin no matchea localhost. Ver [`web/server.py`](web/server.py) sobre el bloque `CORSMiddleware`.

Ambas están seteadas en [`~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist`](~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist). Para aplicar tras editar el plist: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist`. (`kickstart -k` re-lanza el process pero NO relee las env vars — hay que hacer bootout+bootstrap para que agarre cambios del plist).

**Tradeoff importante sobre HTTPS y el SW**:

iOS registra service workers sólo en "secure contexts" (HTTPS o `localhost`). Una IP LAN HTTP (`http://192.168.x.x:8765`) **no es secure context** → el SW no se registra → **se pierde el offline cache** y el "instant-on" desde cache al reabrir.

Lo que **sí funciona sobre HTTP LAN** (el 90% del feel nativo): icon en home screen, splash screen, fullscreen standalone (sin chrome de Safari), Dynamic Island respetado. En red doméstica con el Mac prendido el impacto es nulo porque el fetch al Mac es ~20ms.

Para **full PWA con SW** sobre LAN, la ruta es: (1) agregar un bloque Caddy para la IP del Mac (además del `ra.ai`) con `tls internal`, (2) exportar el root CA de Caddy (`~/Library/Application Support/Caddy/pki/authorities/local/root.crt`) al iPhone vía AirDrop, (3) Settings → General → VPN & Device Management → Install Profile → Trust. Post-trust el iPhone acepta el cert self-signed y el SW registra como si fuera HTTPS público.

**Seguridad**: el server **no tiene auth**. En modo LAN-exposed cualquiera en el mismo WiFi puede leer el vault. Safe en red doméstica con WiFi privado; NUNCA activar en café/coworking/airport. Rollback a localhost-only:

```bash
# Sacar las dos env vars del plist y recargar
sed -i '' '/OBSIDIAN_RAG_ALLOW_LAN\|OBSIDIAN_RAG_BIND_HOST/,/<\/string>/d' ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist
```

Los defaults del código (`127.0.0.1` + regex localhost-only) se preservan si las env vars no están — sin estas dos variables setedas el server se comporta idéntico a antes.

### HTTPS público vía Cloudflare Tunnel (quick, sin dominio, sin cert local)

Para que iOS registre el Service Worker + PWA full (offline cache + instant-on) sin instalar root CA local ni comprar dominio, usamos un [Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/). Cero cuenta, cero config, HTTPS válido con cert público de Cloudflare.

**Cómo funciona**: `cloudflared tunnel --url http://localhost:8765` abre un túnel QUIC salida hacia un edge de Cloudflare y te asigna una URL aleatoria tipo `https://word-word-random.trycloudflare.com`. Todo el tráfico iPhone → Cloudflare → cloudflared → `localhost:8765`. No expone tu IP, no necesita abrir puertos en el router.

**CORS para el túnel — env var adicional requerida** (bug fix 2026-04-27):

El browser del iPhone envía el Origin `https://word-word-random.trycloudflare.com` y el server necesita aceptarlo. Sin la env var el CORS regex es localhost-only y el browser bloquea la primera petición API.

- `OBSIDIAN_RAG_ALLOW_TUNNEL=1` — extiende el regex de CORS con `^https://[a-z0-9-]+\.trycloudflare\.com$`. Default OFF. Agregar al plist del web server junto con las otras env vars. **Solo activar cuando el túnel está corriendo** — expone el server al internet público.

Para activar, agregar al plist `com.fer.obsidian-rag-web.plist`:

```xml
<key>OBSIDIAN_RAG_ALLOW_TUNNEL</key>
<string>1</string>
```

Y recargar: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist`.

**Trade-off clave**: la URL es **random y cambia cada vez que cloudflared reinicia** (launchctl restart, reboot, crash + auto-restart). El PWA guardado en el iPhone se rompe al cambiar la URL — hay que re-abrir en Safari con la URL nueva y re-guardar. Si querés URL estable, hay que migrar a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) con un dominio propio agregado a Cloudflare.

**Dos servicios launchd emparejados** (el segundo es opcional pero altamente recomendado):

1. `com.fer.obsidian-rag-cloudflare-tunnel` → corre `cloudflared tunnel --url http://localhost:8765`. RunAtLoad + KeepAlive. Logs en `~/.local/share/obsidian-rag/cloudflared.log` + `.error.log` (cloudflared escribe todo al stderr por diseño — los "INF" son logs normales).

2. `com.fer.obsidian-rag-cloudflare-tunnel-watcher` → corre [`scripts/cloudflared_watcher.sh`](scripts/cloudflared_watcher.sh). Hace `tail -F` del log de cloudflared y cuando detecta una URL nueva: **(a)** la escribe a `~/.local/share/obsidian-rag/cloudflared-url.txt` (state file estable), **(b)** la copia al clipboard con `pbcopy`, **(c)** manda una macOS notification via `osascript` con sonido Tink. Idempotente — si cloudflared re-emite la misma URL no molesta. State file persiste entre restarts del watcher.

**Helpers zsh** (agregados al final de `~/.zshrc`):

```zsh
alias rag-url='cat ~/.local/share/obsidian-rag/cloudflared-url.txt 2>/dev/null; echo'
alias rag-url-c='cat ~/.local/share/obsidian-rag/cloudflared-url.txt 2>/dev/null | tee >(pbcopy); echo " (copied)"'
```

`rag-url` devuelve la URL activa al instante sin grepear el log. `rag-url-c` la copia al clipboard además de imprimirla.

**Management**:

```bash
# Ver URL actual
rag-url                             # via alias
cat ~/.local/share/obsidian-rag/cloudflared-url.txt

# Forzar URL nueva (útil post cambio de red)
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-cloudflare-tunnel
# El watcher detecta el cambio en ~5-15s y notifica.

# Estado de ambos servicios
launchctl print gui/$(id -u)/com.fer.obsidian-rag-cloudflare-tunnel         | grep -E 'state|pid'
launchctl print gui/$(id -u)/com.fer.obsidian-rag-cloudflare-tunnel-watcher | grep -E 'state|pid'

# Log del watcher (historia de URLs vistas)
tail -n 20 ~/.local/share/obsidian-rag/cloudflared-watcher.log

# Detener todo (rollback a LAN-only HTTP)
launchctl bootout gui/$(id -u)/com.fer.obsidian-rag-cloudflare-tunnel-watcher
launchctl bootout gui/$(id -u)/com.fer.obsidian-rag-cloudflare-tunnel
mv ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel.plist{,.disabled}
mv ~/Library/LaunchAgents/com.fer.obsidian-rag-cloudflare-tunnel-watcher.plist{,.disabled}
```

**Seguridad**: el túnel expone el web server a **internet público**. No tiene auth — cualquiera que sepa la URL random (ej. alguien que ve la URL en el log de Cloudflare si hubiera una filtración) puede leer todo el vault. Las URLs de `trycloudflare.com` son unguessable en la práctica (entropy alta, no indexadas), pero no hay SLA — cloudflare [explícitamente](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) recomienda NO usar quick tunnels para producción. Para paranoia extra: migrar a named tunnel + [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/applications/configure-apps/) (requiere cuenta + dominio propio, gratis hasta 50 usuarios).

**Caveats operacionales**:

- La primera `display notification` pide permisos al user una vez (System Settings → Notifications → "Script Editor" permitir). Si la notificación no aparece, probablemente el user canceló el prompt. Fix: reabrir el prompt desde Script Editor.app o editar los permisos directo en Notifications Settings.
- `cloudflared` instalado vía `brew install cloudflared` (plist apunta a `/opt/homebrew/bin/cloudflared`). Si el user instaló via `npm install -g cloudflared` (wrapper Node), el plist se rompe — actualizar `ProgramArguments` al path del binary nativo o reinstalar vía brew.
- Si hay OTRO proceso cloudflared corriendo en el Mac (ej. para un port distinto), el watcher sólo monitorea el log de este servicio — no toca otros túneles.

## Anticipatory Agent — el vault habla primero (2026-04-24)

**Game-changer**: el RAG deja de ser puramente "pull" (vos preguntás, él responde) y pasa a "push" proactivo. Un daemon ([`com.fer.obsidian-rag-anticipate`](~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist)) corre cada 10 min y, cuando detecta algo timely, te escribe por WhatsApp sin que preguntes. Full doc: [`docs/anticipatory-agent.md`](docs/anticipatory-agent.md).

**3 señales activas** (todas en [`rag.py`](rag.py) bajo `# ── ANTICIPATORY AGENT ──`):

1. **Calendar proximity** (`anticipate-calendar`) — evento de hoy que arranca en [15, 90] min con contexto en el vault → push tipo "📅 En 30 min: call con Juan — [[Coaching - Juan]] score 87%". Reutiliza `_fetch_calendar_today()` (icalBuddy). Snooze 2h por evento.
2. **Temporal echo** (`anticipate-echo`) — nota tocada hoy (≥500 chars) que resuena con una nota >60d (cosine ≥0.70) → push tipo "🔮 Lo que escribiste hoy resuena con algo de hace ~8 meses: [[2025-08-15 - coaching insights]]". Snooze 72h por par.
3. **Stale commitment** (`anticipate-commitment`) — reutiliza `find_followup_loops()` del comando `rag followup`. Stale ≥7d → push tipo "⏰ Hace 11 días dijiste que ibas a hacer X y no veo señal". Snooze 168h (1 semana) por loop.

**Orchestrator** (`anticipate_run_impl`): recoge candidates de las 3 signals, filtra por score ≥ `RAG_ANTICIPATE_MIN_SCORE` (default 0.35), dedupe via SQL lookup en `rag_anticipate_candidates` (ventana 24h, solo `sent=1` bloquea), pickea top-1, empuja vía `proactive_push()` — que a su vez aplica silence list + per-kind snooze + daily_cap=3 (compartido con `emergent` y `patterns` — el budget global no se infla).

**Tabla nueva**: `rag_anticipate_candidates` (append-only, analytics; TODOS los candidates se loguean aunque no se envíen → permite tunear thresholds mirando `rag anticipate log -n 100`).

**CLI**:
```bash
rag anticipate              # = rag anticipate run (default: evalúa + push top-1)
rag anticipate explain      # ver scoring de todas las señales ahora (dry-run + force)
rag anticipate log [-n 20 --only-sent]
rag silence anticipate-calendar          # silenciar una señal (persistente)
rag silence anticipate-calendar --off    # re-activar
```

**Kill switches**:
- Per-kind: `rag silence anticipate-{calendar,echo,commitment}`.
- Global: `RAG_ANTICIPATE_DISABLED=1` → `anticipate_run_impl` early-returns.
- Nuclear: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist && rm ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist`.

**Env tuning** (todas en `rag.py` como module-level, requieren reinstalar daemon para tomar efecto): `RAG_ANTICIPATE_MIN_SCORE` (threshold 0.35), `RAG_ANTICIPATE_DEDUP_WINDOW_HOURS` (24), `RAG_ANTICIPATE_CALENDAR_MIN_MIN/MAX_MIN` (15/90), `RAG_ANTICIPATE_ECHO_MIN_AGE_DAYS/MIN_COSINE` (60/0.70), `RAG_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS` (7). Catálogo completo en [`docs/anticipatory-agent.md`](docs/anticipatory-agent.md).

**Tests**: [`tests/test_anticipate_agent.py`](tests/test_anticipate_agent.py) (38 casos: dataclass + dedup + 3 signals con mocks + orchestrator dedup/threshold/force/error-isolation + CLI). Drift guards: `test_services_spec_total_count` bumpeado 21→22, `test_anticipate_plist_valid_xml`, table count drift (33→34) en `test_sql_state_primitives.py`.

**Fase 2 (roadmap documentado, NO implementado)**: feedback loop (replies 👍/👎 → ajustes thresholds per-kind), quiet hours contextuales (22h→8h + "en reunión" via calendar), voice briefs matinales (morning via TTS en WA), user-configurable weights. Ver [`plans/anticipatory-agent.md`](plans/anticipatory-agent.md) y [`docs/anticipatory-agent.md`](docs/anticipatory-agent.md) para detalles.

### Footer `_anticipate:<dedup_key>_` en pushes anticipate (2026-04-29)

Cuando `proactive_push(...)` recibe `dedup_key=<key>` (lo pasa el orchestrator de anticipate; los callers `emergent` / `patterns` NO pasan dedup_key estable y siguen sin footer), el body se sufija con un párrafo italic markdown:

```
📅 En 30 min: call con Juan — [[Coaching - Juan]] score 87%

_anticipate:cal:event-uuid-123_
```

WhatsApp lo renderiza como cursiva pequeña, discreto pero visible para audit. El footer es **lo que cierra el loop de feedback**: el listener TS, cuando el user responde 👍/👎/🔇 al push, parsea `_anticipate:<key>_` del mensaje quoted y postea a [`POST /api/anticipate/feedback`](web/server.py) con `{dedup_key, rating, reason}`. La row se persiste a `rag_anticipate_feedback` para tunear thresholds + scoring per-kind.

Sin este footer, no hay forma de mapear "el user reaccionó" → "qué dedup_key era" (el orchestrator descarta el state después del push). Por eso lo ponemos inline en el body.

## Bot WA draft loop — auto-aprendizaje del modelo de respuestas (2026-04-29)

Loop completo: incoming WhatsApp → bot draft → user puntúa `/si` `/no` `/editar` en el RagNet group → reply al contacto + feedback al modelo.

**Cómo cierra el loop**:

1. Llega un mensaje al user (Mac de Fer) → el [listener TS](https://github.com/Fonsoide/whatsapp-listener) (otro repo, [`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener)) genera un `bot_draft` con un LLM y lo postea al RagNet group para puntuar.
2. El user responde `/si` (aprueba tal cual), `/no` (rechaza), o `/editar <texto corregido>` (aprueba con edits). Si el draft expira sin respuesta → `expired`.
3. El listener postea a [`POST /api/draft/decision`](web/server.py) con shape:
   ```json
   {
     "draft_id": "abc123",
     "contact_jid": "5491155555555@s.whatsapp.net",
     "contact_name": "Juan",
     "original_msgs": [{"id": "m1", "text": "hola", "ts": "..."}],
     "bot_draft": "todo bien, vos?",
     "decision": "approved_si | approved_editar | rejected | expired",
     "sent_text": "todo bien, vos?",
     "extra": {"draft_score": 0.85, "model": "qwen2.5:7b"}
   }
   ```
4. El web server persiste a `rag_draft_decisions` (silent-fail si DB inaccesible — la UX del listener nunca se rompe por telemetría).

**Tabla `rag_draft_decisions`** (append-only, retention infinita; el dataset histórico vale oro para fine-tunes futuros):

```sql
CREATE TABLE rag_draft_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,                    -- ISO local
  draft_id TEXT NOT NULL,              -- short hex del listener TS, estable across updates del mismo draft
  contact_jid TEXT NOT NULL,
  contact_name TEXT,
  original_msgs_json TEXT,             -- JSON array de {id, text, ts}
  bot_draft TEXT,                      -- texto que el LLM generó
  decision TEXT NOT NULL CHECK(decision IN ('approved_si','approved_editar','rejected','expired')),
  sent_text TEXT,                      -- lo que finalmente se mandó al contacto (NULL si rejected/expired)
  extra_json TEXT                      -- libre: draft_score, model, latency, etc.
);
```

Indices: `ix_rag_draft_decisions_ts`, `ix_rag_draft_decisions_decision`, `ix_rag_draft_decisions_jid`.

**Pares (bot_draft, sent_text) cuando `decision='approved_editar'` son gold humano** para fine-tunear el modelo de drafts. Por ahora NO se consumen en el training loop (eso es cambio bigger); cuando llegue el momento, la query es:

```sql
SELECT bot_draft, sent_text FROM rag_draft_decisions
WHERE decision='approved_editar' AND sent_text IS NOT NULL;
```

**CLI**:

```bash
rag draft                          # = rag draft stats (default)
rag draft stats                    # conteos por tipo + última fecha
rag draft stats --plain            # output plano (piping/scripts)
```

Ejemplo de output:
```
drafts: total 47
  approved_si: 12  (25.5%)
  approved_editar: 18  (38.3%)
  rejected: 9  (19.1%)
  expired: 8  (17.0%)
última decisión: 2026-04-29T14:32:18
```

Health check rápido: si `total` no crece después de un día de uso, o todo se va a `expired`, el flujo está roto (listener no postea, draft nunca llega al RagNet, etc.).

**Activación del loop — flag `WA_DRAFT_ALL_CONTACTS`**:

Por default el listener requiere whitelist explícita (`/draft on <contacto>` desde RagNet) para auto-draftear. La env var `WA_DRAFT_ALL_CONTACTS=1` en el plist del listener (`~/Library/LaunchAgents/com.fer.whatsapp-listener.plist`) activa **bypass global**: todo chat 1:1 (no `@g.us`) genera draft sin pasar por la whitelist. Grupos siguen bloqueados siempre — auto-draftear a 30 personas a la vez no es deseable. El SELF_CHAT_JID (chat 1:1 del user consigo mismo) también queda excluido — el user no se draftea a sí mismo. Mensajes con prefix `\u200B` (BOT_MARKER, outgoing del rag al bridge) se ignoran row-by-row para no loopear.

Para flipearlo: editar el plist + `launchctl bootout/bootstrap`. La env var no se hot-reloadea; el listener la pickup recién al próximo arranque.

**Bug pattern: flag respetada en helper público pero NO en el path real (2026-04-29)**:

Hasta el commit [`whatsapp-listener@29141d2`](https://github.com/jagoff/whatsapp-listener/commit/29141d2) había un bug silencioso. `isDraftWhitelisted(jid, wl)` (helper público, 1 línea: `if (isDraftAllContactsOn() && !jid.endsWith('@g.us')) return true`) sí respetaba el bypass. Pero `processDraftIncoming` (el handler que cada tick mira la bridge SQLite y dispara generación de drafts) construía el SQL siempre con `chat_jid IN (whitelistedJids)` — la whitelist explícita ganaba aunque `WA_DRAFT_ALL_CONTACTS=1` estuviera ON, así que el bypass quedaba muerto en el único path que importa. Síntoma: la flag estaba seteada en el plist hace días, el user veía mensajes entrar al bridge pero ningún draft aparecía en RagNet para contactos NO whitelisteados.

**Lección generalizable**: cuando agregás una flag de comportamiento, no alcanza con que el helper público (`isXEnabled()`) la respete — hay que **auditar TODOS los call sites donde el feature decide qué procesar** y asegurarse que cada uno consulte la flag. Especialmente cuando hay paths que precomputan listas (SQL builders, batch fetchers, cron job consumers) que NO llaman al helper sino que reusan la fuente de datos original (whitelist file, allowlist DB, etc.). Patrón de detección: si el helper se exporta `isXEnabled(arg) → bool` pero hay un call site que filtra por `Object.keys(getXSource())` directamente, el helper no se va a invocar y la flag muere.

Fix en este caso: extraer `buildDraftIncomingQuery({bypassOn, whitelistedJids, selfChatJid, sinceTs}) → {sql, params}` (puro/testeable) y ramificar el SQL según `bypassOn`. Tests cubren bypass on/off, ignora whitelist con bypass, params correctos, SQL invariants.

**Latencia del enriquecimiento de draft con vault context (2026-04-29)**:

`loadVaultContextForDraft(query, contactName)` en el listener llama a `POST /query retrieve_only=true` del rag serve para inyectar sources del corpus indexado al prompt del LLM (qwen2.5:14b genera el draft). Default ts'd timeout era 8s, pero medimos en producción (corpus actual, warm):

| Query | Latencia |
|---|---|
| Q1 cold post-restart | 11.2s |
| Q2 warm | 8.9s |
| Q3 warm | 8.2s |
| Listener-shape (300 chars + nombre) | 7.96s |

`/query retrieve_only=true` p50 ~9s con el rerank cross-encoder bge-reranker-v2-m3 + BM25 + embed bge-m3 secuenciales. Con timeout 8s el call abortaba ~50% de las veces (visible en `~/.local/share/whatsapp-listener/listener.error.log`: `[draft] loadVaultContextForDraft falló: The operation timed out`) y los drafts caían al fallback `[]` silenciosamente — el bot generaba drafts SIN sources del corpus, quedaban genéricos. Bumpeo a 12s en commit [`whatsapp-listener@c160079`](https://github.com/jagoff/whatsapp-listener/commit/c160079). Latencia total del draft sigue dominada por el LLM call (qwen2.5:14b, 30-90s); +4s en retrieve son ~6% del total y compensan con drafts ricos en context. Si el rag serve baja a <6s p99 en el futuro (perf audit pendiente del path `/query retrieve_only=true`), bajamos de nuevo.

**Endpoints relacionados**:

- [`POST /api/draft/decision`](web/server.py) — escribir una decisión (4 decision types válidas, Pydantic valida pre-handler).
- [`POST /api/anticipate/feedback`](web/server.py) — escribir un feedback de push proactivo (rating ∈ `positive|negative|mute`). El listener lo llama cuando parsea `_anticipate:<key>_` en un reply (ver footer pattern arriba).

**Tests**: [`tests/test_draft_decisions_table.py`](tests/test_draft_decisions_table.py) (DDL + helper, 10 casos), [`tests/test_draft_decisions_endpoint.py`](tests/test_draft_decisions_endpoint.py) (4 decision types + validation + persistencia, 9 casos), [`tests/test_anticipate_feedback_endpoint.py`](tests/test_anticipate_feedback_endpoint.py) (3 ratings + validation + silent-fail, 7 casos), [`tests/test_proactive_push_dedup_key_footer.py`](tests/test_proactive_push_dedup_key_footer.py) (footer agrega `_anticipate:<key>_` + back-compat sin dedup_key, 7 casos). Drift guard: table count en [`tests/test_sql_state_primitives.py`](tests/test_sql_state_primitives.py) bumpeado 44→45.

### Fine-tune del modelo de drafts WA — LoRA sobre approved_editar (2026-04-29)

Loop completo del bot WA (sección anterior) acumula gold pairs `(bot_draft, sent_text)` cuando el user hace `/editar`. Esa data hasta ahora se acumulaba sin consumirse. **El fine-tune cierra el loop**: entrena un adapter PEFT/LoRA sobre Qwen2.5-7B-Instruct usando los pares como ground-truth de "qué corrige Fer y por dónde". El adapter NO sustituye el modelo del listener (sigue siendo `qwen2.5:14b`) — es accesible solo vía endpoint preview para A/B manual.

**Cómo correr**:

```bash
rag drafts finetune --dry-run                          # reporta stats sin entrenar
rag drafts finetune --epochs 3 --lr 1e-4               # entrena con defaults
rag drafts finetune --exclude-review-only              # filtra rows con extra_json.review_only=true

# Equivalente directo al script:
uv run python scripts/finetune_drafts.py --dry-run
```

**Cuándo correr**: requiere ≥100 pares totales (`approved_editar` + `rejected`). Por debajo de ese umbral el script falla con `exit 1` y mensaje claro ("datos insuficientes, esperá más feedback"). Razón: con <100 ejemplos el LoRA overfit-ea al estilo exacto de las pocas frases vistas y no generaliza. Para ver el conteo actual: `rag drafts stats`.

Estimación de tiempo: con ~500 pares y 3 epochs en CPU del M-series tarda ~10-30min (LoRA r=8 alpha=16 sobre Q/V projections, ~5 MB de adapter). El base model (~14 GB) se descarga del HF Hub la primera vez.

**Métricas held-out** (split estratificado 80/20 por `draft_id` para no leakear): BLEU-1 unigram + similaridad char-level (1 - normalized edit distance via [`difflib.SequenceMatcher`](https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher)). Print de 5 samples random con bot_draft / target / prediction. Ambas métricas son sanity-check (no eval competitiva) — la eval real la hace el user comparando outputs vía el endpoint preview.

**Activación del adapter — env var `RAG_DRAFTS_FT=1`**:

Default OFF. Cuando ON + el adapter existe en `~/.local/share/obsidian-rag/drafts_ft/`, el endpoint [`POST /api/draft/preview`](web/server.py) usa el modelo fine-tuned. Sin esto (default), el endpoint devuelve `bot_draft_baseline` sin modificar (echo). El listener TS NUNCA usa este modelo — sigue con `qwen2.5:14b` para los drafts en producción.

**Endpoint `/api/draft/preview`**:

```bash
curl -X POST http://localhost:8765/api/draft/preview \
  -H 'Content-Type: application/json' \
  -d '{"original_conversation": "...", "bot_draft_baseline": "..."}'
# → {"ok": true, "preview": "<output>", "ft_active": true|false}
```

`ft_active=false` indica que el caller recibió el baseline tal cual (flag OFF, adapter missing, o generación falló). `ft_active=true` indica que el modelo fine-tuned realmente corrió. Útil para distinguir A/B real de echo silencioso.

**Subcomandos `rag drafts ...`** (alias de `rag draft` singular — ambos funcionan):

```bash
rag drafts stats              # extendido: total + breakdown + review-only + 30d + adapter info
rag drafts stats --plain      # output plano (piping/scripts)
rag drafts finetune --dry-run # bypass al script, reporta stats sin entrenar
rag drafts finetune           # entrena (requires ≥100 pares)
```

**Reglas de seguridad**:

- Si `peft` no está instalado, el script imprime mensaje accionable (`uv tool install --reinstall --editable '.[finetune]'`) y `exit 6` — no excepción rara.
- El loader runtime ([`rag.generate_draft_preview`](rag/__init__.py) + [`rag._load_drafts_ft_model`](rag/__init__.py)) hace silent-fail con [`_silent_log`](rag/__init__.py) en TODO error path: peft missing, adapter dir missing, adapter_config.json malformado, OOM en load, OOM en generate. Worst case el endpoint devuelve el baseline echo + una línea en `~/.local/share/obsidian-rag/silent_errors.jsonl`.
- El adapter dir vive bajo XDG data home (`~/.local/share/obsidian-rag/drafts_ft/`), NO bajo cache (`~/.cache/`). La signal del user (sus correcciones) no es regenerable; macOS limpia el cache automáticamente cuando hay low disk, no queremos perder un fine-tune curado.
- La `stats` extendida lee `extra_json.review_only` con `json_extract(...) IN (1, 'true', true)` — back-compat con tres formas de booleano truthy que pueden llegar del listener.

**Tests**: [`tests/test_finetune_drafts.py`](tests/test_finetune_drafts.py) (11 casos: mining gold+anti, filtro review-only, exit 1 con <100 pares, dry-run happy path, endpoint flag OFF echoes, endpoint flag ON adapter missing → silent fallback, endpoint helper never raises, CLI smoke `rag drafts stats --plain` plural+singular alias, BLEU/sim sanity, split sin leak por draft_id).

**Flujo recomendado** una vez que se acumulen los pares:

1. `rag drafts stats` periódicamente para ver cuándo cruzo los 100 pares.
2. `rag drafts finetune --dry-run` para confirmar el split y los counts antes de invertir tiempo de training.
3. `rag drafts finetune --epochs 3` para entrenar.
4. `export RAG_DRAFTS_FT=1` + reiniciar `rag serve` (`launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-web`) para que el endpoint pickup el adapter.
5. Levantar el listener TS con la conversación de un contacto real, capturar el `bot_draft` y el `original_conversation`, y `curl POST /api/draft/preview` para ver el A/B. Si el output del fine-tuned está más cerca del estilo de Fer → keep el adapter activo. Si peor → `unset RAG_DRAFTS_FT` y volver a iterar con más data.

### Brief feedback loop — los briefs aprenden de las reactions (2026-04-29)

Mismo shape que el loop de anticipate (footer markdown italic + listener TS parsea reactions y postea acá), aplicado a los morning / evening / digest briefs que el daemon escribe + pushea por WhatsApp.

**Cómo cierra el loop**:

1. El daemon arma el brief y lo escribe al vault (ej. `02-Areas/Briefs/2026-04-29-morning.md`).
2. `_brief_push_to_whatsapp(title, vault_relpath, narrative)` formatea el msg WA y le agrega un footer `\n\n_brief:<vault_relpath>_` ANTES de mandarlo. El `vault_relpath` siempre está disponible y es único por brief (lleva la fecha en el nombre), así que sirve directo como `dedup_key`. A diferencia de `proactive_push`, acá el footer NO es opcional — todo brief lleva el suyo.
3. El user reacciona en RagNet con 👍/👎/🔇 (o tokens "ok"/"no"/"basta") dentro de los 30min siguientes. Window más largo que anticipate (10min) porque los briefs se leen más relax, no son urgentes.
4. El listener TS detecta la reaction, busca el push reciente con el footer en la bridge SQLite, parsea el rating con la misma precedencia que anticipate (mute > negative > positive) y postea a [`POST /api/brief/feedback`](web/server.py).
5. El web server persiste a `rag_brief_feedback` (silent-fail si la DB está inaccesible — el listener nunca rompe por telemetría).

**Tabla `rag_brief_feedback`** (append-only):

```sql
CREATE TABLE rag_brief_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,                    -- ISO local
  dedup_key TEXT NOT NULL,             -- vault_relpath del brief
  rating TEXT NOT NULL CHECK(rating IN ('positive','negative','mute')),
  reason TEXT,                         -- texto libre del reply (debug)
  source TEXT DEFAULT 'wa'             -- forward-compatible (PWA, CLI, ...)
);
```

Indices: `ix_rag_brief_feedback_ts`, `ix_rag_brief_feedback_dedup_key`.

**Footer pattern**: el msg WA queda con shape:

```
📓 *Morning 2026-04-29* — `02-Areas/Briefs/2026-04-29-morning.md`

<body con citations renderizadas a obsidian:// URIs>

_brief:02-Areas/Briefs/2026-04-29-morning.md_
```

El listener TS extrae el `vault_relpath` del footer con regex (última línea no-vacía debe matchear `^_brief:([^_]+)_$`). Mismo formato visual que el footer `_anticipate:<key>_` — WA renderiza ambos como cursiva pequeña.

**Hook en el listener**: chequea brief PRIMERO, después anticipate. Razón: el window del brief (30min) es mayor que el de anticipate (10min), pero un push de anticipate puede aparecer DESPUÉS del brief en la misma sesión. Al chequear brief primero solo cuando hay un dedup_key reciente Y rating reconocible, el flow degrada gracefully al anticipate cuando no hay brief candidato.

**CLI**:

```bash
rag brief                          # = rag brief stats (default)
rag brief stats                    # conteos por rating + última fecha
rag brief stats --plain            # output plano (piping/scripts)
```

Ejemplo de output:
```
briefs: total 12
  positive: 8  (66.7%)
  negative: 1  (8.3%)
  mute: 3  (25.0%)
último feedback: 2026-04-29T08:31:42
```

Health check: si `total` no crece después de unos días de uso, o todo se va a `mute`, el flujo está roto (listener no postea, footer del brief roto, briefs no se están enviando, etc.).

**Endpoint relacionado**:

- [`POST /api/brief/feedback`](web/server.py) — escribir un feedback de brief (rating ∈ `positive|negative|mute`, dedup_key = vault_relpath). El listener lo llama cuando parsea `_brief:<key>_` en una reaction.

**Tests**: [`tests/test_brief_feedback_endpoint.py`](tests/test_brief_feedback_endpoint.py) (3 ratings + validation + silent-fail + persistencia E2E, 9 casos), [`tests/test_brief_push_dedup_key_footer.py`](tests/test_brief_push_dedup_key_footer.py) (footer agrega `_brief:<vault_relpath>_` para morning/evening/digest, 8 casos). Drift guard: table count en [`tests/test_sql_state_primitives.py`](tests/test_sql_state_primitives.py) bumpeado 45→46.

### Brief schedule auto-tuning — el horario de los briefs se mueve solo (2026-04-29)

Cierra el ciclo del brief feedback loop: si el usuario mutea consistentemente el morning brief en la primera hora (07:00–07:59), el sistema mueve el plist morning a un horario más tardío automáticamente — sin pedir nada. Mismo principio para today / digest dentro de bandas seguras hardcoded.

**Lógica** (en [`rag/brief_schedule.py`](rag/brief_schedule.py)):

- `analyze_brief_feedback(brief_kind, lookback_days=30)` lee `rag_brief_feedback`, filtra por kind via path heuristic (`_classify_brief_kind` matchea `-morning.md` / `-evening.md` / `-digest.md` o el patrón date-only/`YYYY-Wnn.md`), bucketiza ratings + cuenta mutes por hora-de-feedback.
- **Decision rule**: si `mutes_first_hour ≥ 3` (mutes cuyo `ts.hour == current_brief_hour`) AND `mute / (mute + positive) > 0.5` → sugerir shift `+30min` iterativo dentro de la banda hasta encontrar un slot con menos mutes.
- **Bandas seguras** (hardcoded — invariante operativo, no user-tunable): `morning ∈ [06:30, 09:00]`, `today ∈ [18:00, 21:00]`, `digest ∈ [21:00, 23:30]`. `set_brief_schedule_pref()` rechaza writes fuera de banda; nunca puede landear ahí ni con un caller buggy.

**Tabla `rag_brief_schedule_prefs`** (single-row-per-kind, upsert):

```sql
CREATE TABLE rag_brief_schedule_prefs (
  brief_kind TEXT PRIMARY KEY,    -- 'morning' | 'today' | 'digest'
  hour INTEGER NOT NULL,
  minute INTEGER NOT NULL DEFAULT 0,
  last_updated TEXT NOT NULL,
  reason TEXT                      -- diagnóstico humano: por qué shifteó
);
```

**Cómo se aplica el override**: [`_services_spec()`](rag/__init__.py) consulta `get_brief_schedule_pref(kind)` para morning/today/digest ANTES de generar cada plist. Si hay row, sustituye el `(hour, minute)` en `StartCalendarInterval`; si no, usa los defaults hardcoded de `_morning_plist` (07:00 Mon-Fri) / `_today_plist` (22:00 Mon-Fri) / `_digest_plist` (22:00 Sun). Lookup silent-fail — `rag setup` en una install fresca nunca bloquea por telemetría.

**CLI** (todo en `rag brief schedule *`):

```bash
rag brief schedule                       # = status (default)
rag brief schedule status [--plain]      # current schedule + recommendations por kind
rag brief schedule reset --kind morning  # borra el override (vuelve al default)
rag brief schedule reset --kind all      # borra los 3 overrides
rag brief schedule auto-tune             # dry-run: muestra qué shifts haría
rag brief schedule auto-tune --apply     # escribe overrides + re-bootstrap del plist afectado
```

`auto-tune --apply` escribe la pref Y re-bootstrappea sólo el plist del kind afectado (`launchctl bootout` + `bootstrap`) — no toca los otros plists ni espera al próximo `rag setup`. `--dry-run` siempre gana sobre `--apply` (sanity-check pre-cron).

**Daemon nuevo** [`com.fer.obsidian-rag-brief-auto-tune.plist`](~/Library/LaunchAgents/com.fer.obsidian-rag-brief-auto-tune.plist):

- Schedule: Domingo 03:00 (Sunday=0, Hour=3, Minute=0).
- Comando: `rag brief schedule auto-tune --apply`.
- Logs: `~/.local/share/obsidian-rag/brief-auto-tune.{log,error.log}`.
- `RunAtLoad=false` — no dispara al install (no hay nada que tunear todavía). Se activa solo en el próximo Domingo 03:00.

Por qué Domingo 03:00 específicamente: el digest semanal corre Domingo 22:00, así que cualquier shift toma efecto en el mismo digest del día. El cron de auto-tune sneakea ANTES del online-tune diario (03:30) — sin contención, ya que sólo upsertea una row PK en `rag_brief_schedule_prefs`.

**Tests**: [`tests/test_brief_schedule.py`](tests/test_brief_schedule.py) (11 casos: dry-run / apply / reset / band-respect / classifier / pref-roundtrip / `_services_spec` lee la pref / daemon registrado en spec).

### Voice brief — escuchar el morning en lugar de leerlo (2026-04-29)

Anticipatory Phase 2.C: el morning brief se sintetiza a un voice note (OGG/Opus) y se manda al WhatsApp ambient ANTES del texto. Pensado para escucharlo mientras te preparás (manejando, auriculares, cocinando) en vez de tener que abrir el chat y leer.

**Pipeline** (todo local, cero cloud, cero deps Python nuevas — ver [`rag/voice_brief.py`](rag/voice_brief.py)):

1. `synthesize_brief_audio(text)` strippea markdown del brief (frontmatter, wikilinks, code, footers, headings, bullets) → prosa limpia.
2. `say -v Mónica --file-format=AIFF -o tmp.aiff "<texto>"` (macOS nativo, voz rioplatense default).
3. `ffmpeg -i tmp.aiff -c:a libopus -b:a 24k -ac 1 -ar 16000 final.ogg` → OGG/Opus mono 16 kHz, mismo formato que voice notes nativos de WhatsApp.
4. Cachea en `~/.local/share/obsidian-rag/voice_briefs/YYYY-MM-DD-morning.ogg`. Idempotente — re-runs el mismo día reusan el archivo.
5. POST al bridge con `{recipient: <jid>, media_path: <path>}` (vía `voice_brief.send_audio_to_whatsapp` — NO usa el helper text-only `_ambient_whatsapp_send` porque ese hardcodea payload de texto + prefix anti-loop U+200B que un audio binario no soporta).
6. Después manda el texto del brief con un marker `(audio arriba ↑)` intercalado entre el body y el footer. **El footer `_brief:<vault_relpath>_` queda intacto en la última línea** — el feedback loop sigue funcionando porque el listener TS lee el footer del texto, no del audio.

**Caps de seguridad**:

- Texto > 4000 chars → trim al primer 4000 + `...`.
- Audio > 5 MB → log silent + return `None`, fallback a text-only.
- Sin `say` (CI Linux) → return `None` silent.
- Sin `ffmpeg` → fallback a entregar el `.aiff` directo (más pesado, pero el bridge lo manda igual).
- Send del audio falla → texto sale solo sin marker (transparent fallback).

**Cómo usarlo**:

```bash
# CLI manual: generá el audio de un brief existente.
rag voice-brief generate --date 2026-04-29              # del brief en disco
rag voice-brief generate --text "hola test"             # texto custom
rag voice-brief generate --date 2026-04-29 --apply      # también lo manda al WhatsApp
rag voice-brief generate --voice "Diego" --text "..."   # voz override

# CLI morning: agregar --voice al run habitual para activarlo manualmente.
rag morning --voice
```

**Activar el voice brief en el daemon (opt-in)**: el plist [`com.fer.obsidian-rag-morning.plist`](rag/__init__.py) lleva un env var `RAG_MORNING_VOICE` vacío por default — comportamiento actual sin cambios. Para activarlo, editá el plist seteando `<string>1</string>` en lugar de vacío:

```xml
<key>RAG_MORNING_VOICE</key><string>1</string>
```

Después: `launchctl bootout gui/$UID/com.fer.obsidian-rag-morning && launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.fer.obsidian-rag-morning.plist`. O alternativa: re-correr `rag setup` (no toca el value default vacío del plist regenerado, así que tenés que hacerlo a mano post-setup).

**Decisión de diseño**: env var en el plist existente vs. plist separado. Elegí env var porque (a) el audio + texto son una unidad lógica que sale del mismo run del comando — un plist separado tendría que leer el último brief generado y reabrir el LLM context para sintetizarlo, agregando complejidad, (b) menos servicios launchd que mantener, (c) la activación es un flag, no un schedule diferente.

**Env vars relevantes**:

- `RAG_MORNING_VOICE` — `1`/`true`/`yes` activa el voice brief en el daemon morning. Vacío/ausente = text-only (default histórico).
- `TTS_VOICE` — voz para `say` (default `Mónica`, la rioplatense de macOS). `Diego` para masculina rioplatense, `Paulina` para mexicana, etc. — `say -v ?` lista las disponibles.
- `RAG_VOICE_BRIEF_DIR` — override del cache dir (default `~/.local/share/obsidian-rag/voice_briefs/`). Útil en tests.
- `RAG_VOICE_BRIEF_TTL_DAYS` — TTL para cleanup (default 30 días).

**Cleanup**: `rag maintenance` borra audios > 30 días automáticamente (ver `_cleanup_voice_briefs` en `run_maintenance`). El texto en `04-Archive/99-obsidian-system/99-AI/reviews/<date>.md` es la fuente de verdad histórica; el audio es transiente — siempre re-generable on-demand con `rag voice-brief generate --date <YYYY-MM-DD>`.

**Desactivar**: borrá `RAG_MORNING_VOICE` del plist (o seteá vacío) + bootout/bootstrap. O simpler: kill switch global vía `unset RAG_MORNING_VOICE` cuando corrés `rag morning` manual.

**Tests**: [`tests/test_voice_brief.py`](tests/test_voice_brief.py) (12 casos: synthesis básica + idempotencia + trim 4000 chars + CLI generate / morning --voice / fallback no-say + strip markdown + send_audio_to_whatsapp posts media_path + cleanup + brief_push con audio).

## Whisper learning loop — la transcripción mejora con el uso (2026-04-25)

El sistema de transcripción de audios de WhatsApp aprende del corpus + correcciones manuales del usuario. Cada audio se loguea, las correcciones se acumulan, y un job nightly extrae vocabulario del corpus para sesgar el `--prompt` de whisper. Plan completo en el vault: [`04-Archive/99-obsidian-system/99-AI/system/whatsapp-whisper-learning/plan.md`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fwhatsapp-whisper-learning%2Fplan).

**3 surfaces de aprendizaje**:

1. **Pasivo (corpus + nightly)**: job [`com.fer.obsidian-rag-whisper-vocab`](~/Library/LaunchAgents/com.fer.obsidian-rag-whisper-vocab.plist) corre 03:15 → `refresh_vocab()` extrae top-N términos del vault (notas) + WhatsApp (chats 30d) + Apple Contacts + correcciones explícitas → escribe `rag_whisper_vocab`. Listener lee SQL cada 6h → arma `--prompt` dinámico.
2. **Explícito (`/fix` por WhatsApp)**: el usuario manda `/fix <texto correcto>` y la última transcripción se marca como gold signal con `source='explicit'` en `rag_audio_corrections`.
3. **Confidence-gated LLM correct**: si `avg_logprob < -0.8` (whisper transcribió con baja confianza), el listener pasa el output por `qwen2.5:7b` con sysprompt estructurado + few-shot de correcciones explícitas + vocab hints (top-30 nombres propios). El LLM corrige y la corrección queda con `source='llm'`.

**3 tablas SQL nuevas en `telemetry.db`** (Step 1 del plan):

- `rag_audio_transcripts` (extendida): `audio_path PK`, `text`, `corrected_text`, `correction_source ∈ {explicit, llm, vault_diff}`, `avg_logprob`, `audio_hash`, `chat_id`, `note_path/initial_hash` (placeholder Step 2.d).
- `rag_audio_corrections`: `(audio_hash, original, corrected, source, ts, chat_id, context)`. Append-only. Sin UNIQUE constraint, idempotencia manual en `rag whisper import`.
- `rag_whisper_vocab`: `term PK`, `weight`, `source ∈ {corrections, contacts, notes, chats}`, `last_seen_ts`, `refreshed_at`. Refresh full nightly (DELETE + INSERT). Caps por source: 100 corrections, 100 contacts, 200 notes, 100 chats = 500 total.

**CLI** (todos en `rag whisper *`):

```bash
rag whisper                         # = rag whisper stats (default)
rag whisper stats                   # resumen general
rag whisper vocab refresh           # forzar refresh manual (no esperar al nightly)
rag whisper vocab show [--source X --limit N]
rag whisper patterns [--min-count 2]    # detecta single-word swaps repetidos
rag whisper export [-o FILE] [--source S]   # backup correcciones a JSON
rag whisper import FILE [--dry-run]         # restore/migrate
```

**WhatsApp commands**:

- `/fix <texto correcto>` — corrige la última transcripción global.
- `/whisper` o `/whisper stats` — resumen agregado inline (audios 24h, vocab, correcciones).
- `/whisper recent [N]` — últimas N transcripciones (default 5, max 10) con markers de correction.

**Dashboard**: [`/transcripts`](https://ra.ai/transcripts) — server-rendered en `web/server.py:transcripts_dashboard()`. Stats cards + logprob histogram + heatmap horario (24 cells, 30d) + heatmap semanal día×hora (7×24 = 168 cells, 60d) + tabla últimas 30 transcripts (con duración + markers) + tabla últimas 20 correcciones (diff visual rojo→verde) + top 50 vocab terms. Auto-refresh 60s (override `?nofresh=1`). Dark mode default + light auto via `prefers-color-scheme`. Cross-linkeado desde `/home` y `/dashboard`.

**Listener integration** (en `~/whatsapp-listener/listener.ts`):

- `transcribe(audioPath, chatJid?)` → `TranscribeOutput {text, llmCorrected, originalText?, avgLogprob}`. Server path usa `verbose_json` para capturar `avg_logprob` weighted por duración.
- Logging: `logTranscription()` UPSERT a `rag_audio_transcripts` via `bun:sqlite` (WAL + busy_timeout=5000).
- LLM correct: `attemptLlmCorrect()` con threshold + sanity checks (no-op detect, ratio guard 0.3-2.5x, timeout 15s).
- Reply transparency: cuando hubo LLM correct, el reply al user incluye sufijo `_✨ corregido por llm (original: "...")_`.

**Env tuning** (todas en `~/Library/LaunchAgents/com.fer.whatsapp-listener.plist` o env del shell):

- `WHISPER_MODEL` — default auto-detect `ggml-large-v3-turbo.bin` con fallback a `ggml-small.bin`.
- `WHISPER_VAD_DISABLE=1` — desactiva el VAD pre-pass del silero v5.
- `WHISPER_AF_DISABLE=1` — desactiva el ffmpeg loudnorm + bandpass pre-process.
- `WHISPER_LLM_CORRECT_THRESHOLD=-0.8` — bajar a -0.6 si pocos audios disparan LLM.
- `WHISPER_LLM_CORRECT_MODEL=qwen2.5:7b` — probar `qwen3:30b-a3b` para más quality (slower).
- `WHISPER_LLM_CORRECT_DISABLE=1` — kill switch del LLM correct.
- `WHISPER_TELEMETRY_DISABLE=1` — kill switch del logging a SQL.
- `WHISPER_VOCAB_CACHE_TTL_MS=21600000` — TTL del cache in-mem del vocab (6h default).

**Kill switches**:
- LLM correct: `WHISPER_LLM_CORRECT_DISABLE=1` en plist + `launchctl kickstart -k`.
- Vocab refresh nightly: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-whisper-vocab.plist`.
- Logging: `WHISPER_TELEMETRY_DISABLE=1` en plist del listener.
- Nuclear: borrar las 3 tablas + `WHISPER_TELEMETRY_DISABLE=1` + revertir commits Phase 2 con `git revert c597932 fd3ff44 d747432 ...`.

**Tests**: 47+ tests del learning loop:
- [`tests/test_web_transcripts.py`](tests/test_web_transcripts.py) — 25 tests del dashboard (render, dark mode, auto-refresh, heatmaps, navegación).
- [`tests/test_whisper_patterns.py`](tests/test_whisper_patterns.py) — 11 tests del pattern detection.
- [`tests/test_whisper_export_import.py`](tests/test_whisper_export_import.py) — 11 tests del export/import roundtrip.

**Calibración futura**: después de unos días de uso real, mirar histogram en `/transcripts` y ajustar threshold; correr `rag whisper patterns` para ver errores sistemáticos del modelo; backup periódico con `rag whisper export -o ~/Backups/corrections-$(date +%F).json`.

## Implicit feedback — reward shaping con negativos débiles (2026-04-29)

`rag feedback classify-sessions` (en el plist [`com.fer.obsidian-rag-implicit-feedback`](rag/__init__.py)) backpropaga el outcome de una session entera (win/loss/abandon/partial) como signal a cada turn. La rutina vive en [`rag_implicit_learning/reward_shaping.py:apply_reward_from_session_outcomes`](rag_implicit_learning/reward_shaping.py).

**Branches por outcome**:

- `win` → `rating=+1`, `extra_json.implicit_loss_source='session_outcome_win'`. Aplica gate `confidence ≥ DEFAULT_MIN_CONFIDENCE` (0.7).
- `loss` → `rating=-1`, source `session_outcome_loss`. Mismo gate.
- `partial` → skip total (n_skip_ambiguous_outcome).
- `abandon` → branch nuevo (Quick Win #3, 2026-04-29):
  - Lee `top_score` del primer turn de la session (de `rag_queries.top_score`).
  - Si `top_score < WEAK_NEGATIVE_TOP_SCORE_THRESHOLD` (0.4) → `rating=-1`, source `session_outcome_weak_negative`. **El confidence gate NO aplica** (el feature absorbe data débil de propósito; `effective_confidence` se capa a min(0.5, session_confidence) para señalizar la duda).
  - Si `top_score >= 0.4` o no hay top_score → skip ambiguous (legacy).

**Treatment en training del ranker** ([`rag_ranker_lgbm/features.py:feedback_to_training_data`](rag_ranker_lgbm/features.py)): los candidates de feedback rows con `implicit_loss_source='session_outcome_weak_negative'` reciben `weight=0.3` (constante `WEAK_NEGATIVE_TRAINING_WEIGHT`). El resto pesa 1.0. El array `weight` se pasa a `lgb.Dataset(weight=...)` en `train_lambdarank` → el lambdarank gradient penaliza la mitad para esos rows. La columna `rating` queda como -1 (compatibilidad con downstream code que solo lee `rating`); la atenuación es solo en TRAINING.

**Por qué importa**: pre-fix había 542 sessions con outcome=abandon en la ventana de 14d → solo 18 loss confirmados → asimetría 30:1 en signal negativa. El sistema descartaba abandon como "ambiguo" por defecto, perdiendo la signal de queries que claramente no encontraron nada (top_score bajo + user se fue). Post-fix se espera absorber ~50-100 negativos débiles por semana sin contaminar los positivos (weight 0.3 acota el blast radius si la inferencia "abandon=loss" estuvo errada).

**Configuración** (in-code, no env var por ahora — cambios requieren edit + redeploy del CLI):

- `WEAK_NEGATIVE_TOP_SCORE_THRESHOLD = 0.4` en [`rag_implicit_learning/reward_shaping.py`](rag_implicit_learning/reward_shaping.py). Subir → más sessions abandon caen como weak_negative; bajar → solo las queries con top_score muy bajo. 0.4 es el equilibrio observado en el dataset (cae mid-distribution post-rerank).
- `WEAK_NEGATIVE_TRAINING_WEIGHT = 0.3` en [`rag_ranker_lgbm/features.py`](rag_ranker_lgbm/features.py). Subir a 1.0 desactiva la atenuación; bajar a 0.0 desactiva el feature.

**Tests**: [`tests/test_reward_shaping_weak_negative.py`](tests/test_reward_shaping_weak_negative.py) (11 casos: happy path, threshold boundary, no top_score, win/loss/partial intactos, confidence override, idempotencia, dry-run, training weight end-to-end). Pre-existentes [`tests/test_implicit_learning_session_outcome.py`](tests/test_implicit_learning_session_outcome.py) (17 casos del flow original, todos siguen pasando).

**Telemetry rápida**:

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db <<'EOF'
SELECT COUNT(*) FROM rag_feedback WHERE json_extract(extra_json, '$.implicit_loss_source')='session_outcome_loss';
SELECT COUNT(*) FROM rag_feedback WHERE json_extract(extra_json, '$.implicit_loss_source')='session_outcome_weak_negative';
EOF
```

## Commands

```bash
uv tool install --reinstall --editable '.[entities,stt,spotify]'   # reinstall after code changes (incluye extras: GLiNER, faster-whisper, spotipy)

# Core
rag index [--reset] [--no-contradict] [--vault NAME]  # incremental hash-based; --reset rebuilds; --vault override
rag index --source whatsapp [--reset] [--since ISO] [--dry-run] [--max-chats N]  # WA ingester (Phase 1.a)
rag index --source contacts [--reset] [--dry-run]    # Apple Contacts ingester (Phase 1.e) — corpus + phone→name helper
rag index --source calls [--reset] [--since ISO] [--dry-run]  # CallHistory ingester (Phase 1.f) — llamadas perdidas/entrantes/salientes, enriquecidas con Contacts
rag index --source safari [--reset] [--since ISO] [--dry-run] [--max-urls N] [--skip-bookmarks]  # Safari History + Bookmarks + Reading List (Phase 2)
rag watch                                  # watchdog auto-reindex (debounce 3s)
rag query "text" [--hyde --no-multi --raw --loose --force --counter --no-deep --session ID --continue --plain --source S[,S2] --vault NAME]
rag chat [--counter --no-deep --session ID --resume] # /save /reindex (or NL) work; create-intent tool-calling (`recordame X`, `cumple de Y el viernes`)
rag do "instrucción" [--yes --max-iterations 8]  # tool-calling agent loop
rag stats                                  # models + index status
rag session list|show|clear|cleanup

# Productivity
rag capture "texto" [--tag X --source Y --stdin --title T --plain]
rag inbox [--apply]                        # triage 00-Inbox: folder + tags + wikilinks + dupes
rag prep "tema" [--save]                   # context brief → optionally 00-Inbox/
rag read <url> [--save --plain]            # ingest article → 00-Inbox/ w/ auto-wikilinks
rag dupes [--threshold 0.85 --folder X]
rag links "query" [--open N --rebuild]     # semantic URL finder, no LLM
rag wikilinks suggest [--folder X --apply] # graph densifier, no LLM
rag followup [--days 30 --status stale|activo|resolved --json]
rag dead [--min-age-days 365]              # candidates to archive (read-only)
rag archive [--apply --force --gate 20]    # move dead → 04-Archive/ (dry-run default)

# Daily automation
rag morning [--dry-run]                    # daily brief → 04-Archive/99-obsidian-system/99-AI/reviews/YYYY-MM-DD.md
rag today [--dry-run]                      # EOD closure → 04-Archive/99-obsidian-system/99-AI/reviews/YYYY-MM-DD-evening.md
rag digest [--week YYYY-WNN --days N]      # weekly narrative → 04-Archive/99-obsidian-system/99-AI/reviews/YYYY-WNN.md
rag consolidate [--window-days 14 --threshold 0.75 --min-cluster 3 --dry-run --json]  # episodic memory Phase 2 → PARA

# Ambient agent (reactive — dispara al cambiar una nota)
rag ambient status|disable|test [path]|log [-n N]
rag ambient folders list|add <F>|remove <F>

# Anticipatory agent (proactive — el vault te habla sin que preguntes)
rag anticipate [run]                       # default: evalúa señales + push top-1 a WA
rag anticipate run [--dry-run --explain --force]  # dry-run no pushea, --explain muestra scoring
rag anticipate explain                     # ver scoring de todas las señales del momento
rag anticipate log [-n 20 --only-sent]     # últimas N entries de rag_anticipate_candidates
rag silence anticipate-{calendar,echo,commitment} [--off]  # silenciar per-kind

# Quality
rag eval [--latency --max-p95-ms N]        # queries.yaml → hit@k, MRR, recall@k (+ bootstrap CI); gate on P95
rag tune [--samples 500] [--apply] [--online --days 14] [--rollback]  # offline + online ranker-vivo loop
rag log [-n 20] [--low-confidence]
rag dashboard [--days 30]                  # analytics: scores, latency, topics, PageRank
rag feedback status                        # progress hacia los 20 corrective_paths del gate GC#2.C
rag feedback backfill [--limit N --rating pos|neg|both --since DAYS]  # agregar corrective_path a turns existentes
rag feedback infer-implicit [--window-seconds 600 --dry-run --json]   # derivar corrective_path desde opens + paráfrasis follow-up
rag behavior backfill [--dry-run --window-minutes N --limit N]  # linkea opens huérfanos (original_query_id NULL) al rag_queries.id — +training signal
rag feedback harvest [--limit N --since DAYS --confidence-below F]    # labelear queries low-conf sin thumbs
rag open <path> [--query Q --rank N --source cli]  # emits behavior event + `open` path (ranker-vivo click tracking)
rag open --nth N [--session ID]  # shortcut: abre el N-ésimo source del último rag_queries + auto-fill original_query_id (sin setup de x-rag-open handler)

# Maintenance
rag maintenance [--dry-run --skip-reindex --skip-logs --json]  # all-in-one housekeeping
rag free [--apply --yes --force --json --min-age-days N --ranker-keep N --skip-{tables,baks,logs,ranker}]  # liberar espacio sin romper: dropea tablas legacy en ragvec.db (sanity-checked contra telemetry.db), borra .bak.<ts> de T10, logs .archived* viejos, snapshots ranker.<ts>.json redundantes
python scripts/backfill_entities.py [--dry-run --limit N --vault NAME]  # one-shot GLiNER entity extraction
python scripts/audit_telemetry_health.py [--days 7] [--json]  # data-first health audit: errores SQL/silent, latency outliers, cache probe distribution, DB sizes; primer comando antes de cualquier "auditá el sistema"

rag implicit-feedback [--days 14 --json]  # recolecta feedback implícito de interacciones
rag routing-rules [--reset --debug --json]  # descriptor de rutas + patterns detectados
rag whisper-vocab [--refresh --show --source X --limit N]  # manejo de vocabulario de transcripción WhatsApp
rag vault-cleanup [--dry-run --apply --force]  # limpia carpetas transitorias del vault
rag ingest-drive [--reset --dry-run --json]  # Google Drive ingester — busca DAO + documentos compartidos


# Automation
rag setup [--remove]                       # install/remove 31 launchd services (anticipate incluido 2026-04-24)

# Tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_foo.py::test_bar -q   # single test
```

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

### Env vars

**Catálogo completo**: [docs/env-vars-catalog.md](docs/env-vars-catalog.md) lista las **47 env vars adicionales** que `rag.py` consulta pero no están documentadas aquí (config defaults, feature flags, infra). Esta sección cubre solo las críticas con rollback paths + historia detallada.

- `OBSIDIAN_RAG_VAULT` — override default vault path. Collections are namespaced per resolved path (sha256[:8]). En la precedencia multi-vault, gana sobre el `current` del registry. `rag query --vault NAME` y `rag index --vault NAME` son equivalentes por-invocación sin mutar el env. Single-vault only en ambos comandos; para cross-vault query usar `rag chat --vault a,b`. Los cross-source ETLs (MOZE, WhatsApp, Gmail, Reminders, Calendar, Chrome, Drive, GitHub, Claude, YouTube, Spotify) se gatean por `_is_cross_source_target(vault_path)` — por default solo el `_DEFAULT_VAULT` (iCloud Notes) los recibe. Para opt-inear a otro vault agregar `"cross_source_target": "<name>"` al `~/.config/obsidian-rag/vaults.json`. Sin opt-in, `rag index --vault work` skippea los 11 ETLs con un log `[dim]Cross-source syncs: skip[/dim]` y solo indexa las `.md` reales del vault — evita la contaminación medida 2026-04-21 en que los ETLs copiaron 19 archivos MOZE al vault `work`. Tests: `tests/test_vaults.py` (10 casos sobre guard + flag).
- `RAG_OCR=0` — desactiva OCR en imágenes embebidas durante el indexing (default ON cuando `ocrmac` está disponible). El indexer extrae `![[img.png]]` + `![alt](img.png)` de cada nota (`_extract_embedded_images` en `rag.py`), llama `_ocr_image` vía Apple Vision (es-ES + en-US), y concatena el texto extraído al body antes de chunkear con un marker `<!-- OCR: path/to/img -->`. Cache persistente en `rag_ocr_cache` SQL table (key = abs path, invalidación por mtime — sin TTL). El hash del chunk (`_file_hash_with_images`) suma los mtimes de las imágenes embebidas, así que una screenshot actualizada fuerza reindex aunque el .md no haya cambiado. Soft-dep: ocrmac + pyobjc son macOS-only — en Linux el `try: import ocrmac` falla y el OCR skippea silently. Motivación (2026-04-21): notas tipo link-hub (`dev cycles.md` = un link + un `![[captura.png]]` con la tabla informativa en la PNG) eran invisibles al retrieval porque el body textual tenía ~0 signal; post-OCR el rerank score pasó de +0.0 a +1.1 y el LLM responde queries específicas ("cuándo termina el cycle 10.59"). Tests: `tests/test_ocr_embedded_images.py` (26 casos).
- `OLLAMA_KEEP_ALIVE` — passed to every ollama chat/embed call. Code default `-1` (forever, `rag.py:1608`); launchd plists set the same `-1` for symmetry. Accepts int seconds or duration string. **Historia**: default fue `"20m"` entre 2026-04-17 y 2026-04-21 por un Mac-freeze bug cuando `-1` pineaba command-r (~19 GB) como wired memory en 36 GB unificados. Vuelto a `-1` el 2026-04-21 tras el cambio del chat model default a qwen2.5:7b (~4.7 GB, bench 2026-04-18) — stack pinned ahora ≈ 8 GB (qwen2.5:7b + qwen2.5:3b helper + bge-m3), holgado en 36 GB. **Guard automático** (`chat_keep_alive()` en `rag.py:1626`): si el chat model efectivo está en `_LARGE_CHAT_MODELS` (command-r, qwen3:30b-a3b), el keep_alive se clampea a `_LARGE_KEEP_ALIVE="20m"` automáticamente — no más Mac freeze aunque `resolve_chat_model()` caiga a command-r por falta de qwen2.5:7b. **Rollback**: si vuelven los beachballs, exportá `OLLAMA_KEEP_ALIVE=20m` en tu shell antes de la primera invocación de `rag` — override sigue funcionando en ambas direcciones. Tests: [`tests/test_chat_keep_alive_guard.py`](tests/test_chat_keep_alive_guard.py) (10 casos cubren clamp, passthrough, override por env, degradación graciosa).
- `RAG_KEEP_ALIVE_LARGE_MODEL` — opt-out del auto-clamp de `chat_keep_alive()` para modelos grandes. Útil si tenés >64 GB unified memory y querés pinear command-r "forever" igualmente. Setear a una duración (`"4h"`) o entero segundos (`"14400"`). Sin esta variable, los modelos en `_LARGE_CHAT_MODELS` usan `"20m"`.
- `RAG_MEMORY_PRESSURE_DISABLE=1` — desactiva el memory-pressure watchdog (`start_memory_pressure_watchdog()` en `rag.py:8816`). El watchdog es un daemon thread que arranca desde `rag serve` y desde el startup del web server; cada `RAG_MEMORY_PRESSURE_INTERVAL` segundos muestrea `vm_stat` + `sysctl hw.memsize` vía `_system_memory_used_pct()` y si `(wired + active + compressed) / total >= RAG_MEMORY_PRESSURE_THRESHOLD` (default 85%) descarga proactivamente el chat model vía `ollama.chat(keep_alive=0)` y — si la presión sigue ≥ threshold tras eso — fuerza el unload del cross-encoder con `maybe_unload_reranker(force=True)` (bypasseando tanto el TTL idle check como el `RAG_RERANKER_NEVER_UNLOAD=1` del operador: bajo presión real de memoria preferimos el 5s cold-reload antes que un Mac freeze). **Motivación**: post `chat_keep_alive()` auto-clamp, queda un edge case — usuario corriendo `rag do` + web server + Claude Code + Chrome pesado puede saturar los 36 GB igualmente, porque `OLLAMA_MAX_LOADED_MODELS=2` NO es VRAM-aware y helper + embed + chat + reranker ya ocupan ~8 GB pinned. El watchdog es la red de seguridad activa. **No-op en Linux** (vm_stat es macOS-only) y en CLI one-shot (no se arranca en `rag query/chat/do`). Setear `RAG_MEMORY_PRESSURE_DISABLE=1` en tests/CI. Tests: [`tests/test_memory_pressure_watchdog.py`](tests/test_memory_pressure_watchdog.py) (17 casos: parse de vm_stat con 16 KB pages Apple Silicon, zero-total defensive, escalación chat→reranker, idempotencia, env overrides, `maybe_unload_reranker(force=True)` bypass del TTL).
- `RAG_MEMORY_PRESSURE_THRESHOLD` — umbral % para disparar el watchdog (default 85). Valores razonables: 80 (más agresivo, previene antes), 90 (más tolerante, menos unloads inútiles). Bajarlo si tenés <32 GB; subirlo si tenés >64 GB.
- `RAG_MEMORY_PRESSURE_INTERVAL` — intervalo de sampling en segundos (default 60). Menos = detecta pressure más rápido pero gasta más CPU en `vm_stat` forks. Más = reacción más lenta pero menos overhead.
- `RAG_STATE_SQL=1` — historically enabled the SQL telemetry store. Post-T10 (2026-04-20) + split (2026-04-21, ragvec.db → ragvec.db + telemetry.db) the JSONL fallback is gone and the flag is a **no-op** — neither writers nor readers consult it, SQL is the only path. Still set on every launchd plist for deployment-config symmetry / faster rollback if needed.
- `RAG_LOG_QUERY_ASYNC=0` — opt-out del async-default de `log_query_event()`. Default ON desde 2026-04-22 (el writer va al `_BACKGROUND_SQL_QUEUE` daemon en vez de pegar sync + retry budget). Tests que leen `rag_queries` post-write lo fuerzan a `0` en conftest.
- `RAG_LOG_BEHAVIOR_ASYNC=0` — opt-out para `log_behavior_event()` + `log_impressions()`. Default ON desde 2026-04-24 tras audit de contención (156 `impression_sql_write_failed` + 34 `behavior_sql_write_failed` en 6 días). Mismo pattern que `RAG_LOG_QUERY_ASYNC`.
- `RAG_METRICS_ASYNC=0` — opt-out para los samplers `_memory_persist` / `_cpu_persist` del web server. Default ON desde 2026-04-24 (66 `memory_sql_write_failed` + 34 `cpu_sql_write_failed` en 6 días). Los samplers daemon cada 60s no necesitan confirmación sync.
- `RAG_TRACK_OPENS=1` — switches OSC 8 link scheme from `file://` to `x-rag-open://` so CLI clicks route through `rag open` (ranker-vivo signal capture). Absent = no behavior change.
- `RAG_EXPLORE=1` — enable ε-exploration in `retrieve()` (10% chance to swap a top-3 result with a rank-4..7 candidate). Set on `morning`/`today` plists to generate counterfactuals. MUST be unset during `rag eval` — the command actively `os.environ.pop`s it and asserts, as a belt-and-suspenders guard.
- `RAG_RERANKER_IDLE_TTL` — seconds the cross-encoder stays resident before idle-unload (default 900).
- `RAG_RERANKER_NEVER_UNLOAD` — set to `1` in the web + serve launchd plists to pin the reranker in MPS VRAM permanently; sweeper loop still runs but skips `maybe_unload_reranker()`. Eliminates the 9s cold-reload hit after idle eviction. Cost: ~2-3 GB unified memory pinned. Safe on 36 GB with command-r + qwen3:8b resident.
- `RAG_RERANKER_FT=1` — **GC#2.C (2026-04-23)**: opt-in al LoRA adapter del cross-encoder. Cuando está prendida, `get_reranker()` carga el adapter desde [`~/.local/share/obsidian-rag/reranker_ft/`](file:///Users/fer/.local/share/obsidian-rag/reranker_ft/) on top del base bge-reranker-v2-m3 vía [peft](https://github.com/huggingface/peft) (`PeftModel.from_pretrained`). El adapter se entrena con [`scripts/finetune_reranker.py --mode lora`](scripts/finetune_reranker.py) sobre los pares positivos/negativos de `rag_feedback` (rating=+1) + `rag_behavior` (`event='positive_implicit'` como positivos, `event='impression'` con rank ≥3 sin `open` posterior como hard-negs). LoRA config: `r=8, alpha=16, dropout=0.1`, target `query`/`value` projections de XLM-RoBERTa. **Default OFF** durante el rollout — el operador opt-inea explícitamente. Failure modes (todos silent_fail con log a `silent_errors.jsonl`, NUNCA tumban el flow): (a) flag prendida + dir vacío → fallback a base, (b) `peft` no instalado → fallback a base, (c) `adapter_config.json` missing/inválido → fallback a base. **Distinto del `RAG_RERANKER_FT_PATH`** (GC#2.B, full fine-tune via symlink): este es un overlay LoRA de ~5 MB, no un checkpoint completo de 2 GB. Pueden coexistir — el path resuelto por `_resolve_reranker_model_path()` es la base sobre la que se aplica el LoRA. **Setup**: `uv tool install --reinstall --editable '.[finetune]'` para los deps (peft + transformers + datasets + accelerate); después `python scripts/finetune_reranker.py --mode lora --epochs 3 --lr 2e-5` para entrenar. Tests: [`tests/test_finetune_reranker_gate.py`](tests/test_finetune_reranker_gate.py) (19 casos: env-unset → no peft import, dir vacío → fallback + warning, adapter válido → load OK, scores stay en [0,1], smoke con flag ON sin adapter, helpers de nDCG@5 + AUC).
- `RAG_LOCAL_EMBED` — set to `1` in the web + serve launchd plists to use in-process `SentenceTransformer("BAAI/bge-m3")` for query embedding instead of ollama HTTP (~10-30ms vs ~140ms). Requires BAAI/bge-m3 cached in `~/.cache/huggingface/hub/` — download once with `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` before enabling. Verify cosine >0.999 vs ollama embeddings of same text before enabling in production. Do NOT set for indexing/watch/ingest processes — bulk chunk embedding stays on ollama. Uses CLS pooling (same as ollama gguf). Post 2026-04-21 the CLI group (`cli()` in `rag.py:11894`) auto-sets this to `1` when invoking query-like subcommands (set in `_LOCAL_EMBED_AUTO_CMDS`: `query`, `chat`, `do`, `pendientes`, `prep`, `links`, `dupes`) unless the user already set it explicitly (both truthy + falsy overrides respected). Bulk paths (`index`, `watch`, ingesters) stay off the allow-list per the same invariant. **Cold-load warmup**: loading `SentenceTransformer` on MPS takes ~5.6s end-to-end (imports + weights + first encode JIT). `_warmup_local_embedder()` (rag.py, next to `query_embed_local`) centralises the preload and is invoked from `warmup_async()` (background daemon thread for CLI query-like subcommands) and from `rag serve`'s eager warmup. Before this, only `web/server.py:_do_warmup` preloaded it — `rag serve` and one-shot CLI invocations paid the 5.6s on the critical path of the first retrieve (confirmed 2026-04-21 in `rag_queries.extra_json`: embed_ms 3455/4137/4898 on the first few serve turns, dropping to 46ms post-warmup). Helper self-gates on `_local_embed_enabled()`, swallows exceptions, and is a no-op when the flag is falsy — safe to call unconditionally. Tests: `tests/test_warmup_local_embed.py` (16 cases). **Non-blocking gate (2026-04-21 evening)**: `_local_embedder_ready: threading.Event` fires *only* after `_warmup_local_embedder()` completes load + first successful encode. `query_embed_local()` checks `Event.is_set()` **before** entering the lock — if clear, returns None immediately so the caller falls back to ollama embed (~150ms). Pre-fix the main thread blocked on the lock for 5-12s on one-shot CLI when warmup_async raced the query (measured embed_ms up to 12014 in production). Long-running processes warm up at startup so the Event is set before the first user query, keeping the fast in-process path. Tests: `tests/test_local_embed_nonblocking.py` (9 cases: Event-clear bail timing, slow-loader doesn't block, post-warmup path works, concurrent warmup+5 queries stay <100ms).
- `RAG_FAST_PATH_KEEP_WITH_TOOLS` — rollback del auto-downgrade en `/api/chat` cuando el pre-router matchea tools deterministas estando en fast-path. **Default OFF** (downgrade activo). Motivación (2026-04-24, medido 2026-04-23): queries como "qué pendientes tengo" disparaban `fast_path=1` en `retrieve()` (intent `recent` + top-score alto) → web switchaba a `_LOOKUP_MODEL` (qwen2.5:3b) + `_LOOKUP_NUM_CTX` (4096). Pero ese mismo query matchea el pre-router (`_PLANNING_PAT`) → dispara `reminders_due` + `calendar_ahead` y **reemplaza** el CONTEXTO entero con la salida formateada (2-4K tokens de listas). qwen2.5:3b en M3 Max prefillea esas listas a ~2.5ms/tok → 7-11s sólo de prefill (medido: `"qué pendientes tengo"` llm_prefill=11595ms total=16.3s). qwen2.5:7b prefillea a ~0.5ms/tok → 1.5-2s en el mismo ctx, total estimado ~5s. El gate nuevo hace downgrade runtime: si `_fast_path=True` AND `_forced_tools` no está vacío, switch a `_resolve_web_chat_model()` + `_WEB_CHAT_NUM_CTX`. El marker `fast_path=True` en telemetry se mantiene (refleja la decisión de `retrieve()`); el marker nuevo `fast_path_downgraded=True` registra cuántas veces el downgrade disparó. Setear `export RAG_FAST_PATH_KEEP_WITH_TOOLS=1` para restaurar pre-fix. Tests: [`tests/test_web_fast_path_downgrade.py`](tests/test_web_fast_path_downgrade.py) (9 casos: contratos source-level + 3 functional con TestClient). El downgrade sólo aplica a `/api/chat` (web endpoint); `rag serve` y CLI no lo necesitan porque su fast-path no usa pre-router regex con tools.
- `RAG_LOCAL_EMBED_WAIT_MS` — budget (milliseconds) para esperar que `_local_embedder_ready` dispare antes de caer a `embed()` ollama en `retrieve()`. **Default 6000ms tras 2026-04-23** (bumped desde 4000ms). El cold load de `SentenceTransformer('BAAI/bge-m3')` en MPS toma ~5s extremo-a-extremo (imports + weights + first encode JIT). Con el default histórico de 4000ms, el wait timeaba **exactamente** al 4s antes del Event fire — producción mostró un patrón repetido de `embed_ms=4005` exacto en 4 CLI `query` consecutivas (2026-04-23T15:14-15:15): el user pagaba 4s de wait + 5ms de fallback ollama (warm) para un total de 4005ms de embed per query — pura latencia gratis. Con 6000ms el Event dispara dentro del budget, el user recibe el path MPS (~30ms encode) en la misma query. Si el warmup también timea (disk frío, HF cache distante), el fallback ollama sigue siendo 140ms warm → cap final ~6.14s (vs 4.14s + potencial ollama cold = peor). Rollback `export RAG_LOCAL_EMBED_WAIT_MS=4000` para restaurar pre-2026-04-23. `RAG_LOCAL_EMBED_WAIT_MS=0` restaura non-blocking legacy (Event.wait returns inmediato, fallback siempre). `RAG_LOCAL_EMBED_WAIT_MS=10000` para Macs lentas (HDD externa o spinning disk). Self-contained — no cambia warmup_async ni el path long-running (serve/web), donde el Event ya está set pre-query. Tests: [`tests/test_warmup_parallel.py`](tests/test_warmup_parallel.py) + [`tests/test_query_embed_local_wait.py`](tests/test_query_embed_local_wait.py) (contratos sobre el nuevo default 6.0s).
- `RAG_EXPAND_MIN_TOKENS` — threshold for the `expand_queries()` short-query perf gate (default `4`, `rag.py:7821`). Queries shorter than this token count (split by whitespace) skip the qwen2.5:3b paraphrase call (~1-3s saved). Raise to be more aggressive about skipping paraphrase; lower to restore pre-2026-04-21 behaviour (`<= 2` tokens skipped). Web server already forced `multi_query=False` globally (`web/server.py:3648`), this makes CLI caller-by-caller.
- `RAG_CITATION_REPAIR_MAX_BAD` — threshold for the citation-repair perf gate (default `3`, `rag.py:109`). When `verify_citations()` returns more than this many invented paths, the repair round-trip is skipped entirely (rationale: heavily hallucinated answers rarely recover under a single-shot repair and the 5-8s non-streaming call dominates latency). Set to `0` to disable citation-repair completely. Applies to both `rag query` (`rag.py:13924`) and `rag chat` (`rag.py:15043`) paths.
- `OBSIDIAN_RAG_NO_APPLE=1` — disables Apple integrations (Calendar, Reminders, Mail, Screen Time) entirely. Useful on non-macOS hosts or when Full Disk Access is not granted.
- `RAG_TIMEZONE` — IANA tz string used by `_parse_natural_datetime` for ISO-with-tzinfo inputs (ISO strings with `Z` / offset). Default `America/Argentina/Buenos_Aires` (UTC-3 / UTC-2 depending on DST, but AR stays UTC-3 year-round as of 2019). Naive datetimes (user typing "mañana 10am") are interpreted relative to anchor and don't hit the TZ conversion path; only IS0-8601 inputs with tzinfo do. Week-start follows dateparser's ISO default (Monday).
- `OBSIDIAN_RAG_MOZE_FOLDER` — override MOZE ETL target folder inside the vault (default `02-Areas/Personal/Finanzas/MOZE`).
- `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` — comma-separated vault-relative folders `rag watch` must ignore. Default `"03-Resources/WhatsApp"` (WA dumps re-fire the handler dozens of times per minute via periodic ETL; they're picked up by manual/periodic `rag index` instead).
- `OBSIDIAN_RAG_INDEX_WA_MONTHLY` — rollback para la exclusión permanente de `03-Resources/WhatsApp/<chat>/<YYYY-MM>.md` en `is_excluded()` (`rag.py:2201`). **Default OFF = monthly rollups NO se indexan como `source=vault`** (medido 2026-04-22). El directorio sigue recibiendo los `.md` desde `scripts/ingest_whatsapp.py` (para leer en Obsidian), pero los mensajes individuales ya están indexados como ~4150 chunks `source="whatsapp"` (pseudo-URI doc_id `whatsapp://<jid>/<msg_id>`, ~143 chars each). Pre-fix la duplicación agregaba +1355 chunks `source="vault"` con sizes hasta 21808 chars → el retrieve pescaba la misma conversación en 2 representaciones que competían por spots en top-k, y el LLM recibía context redundante que inflaba prefill + gen. Queries donde top-1 era WA tenían avg gen 16.3s / P95 37.9s. Setear `OBSIDIAN_RAG_INDEX_WA_MONTHLY=1` + `rag index --reset` para rehabilitar el double-indexing (útil sólo si se detectan regresiones de recall en queries WA específicas post-rollout). Tests: [`tests/test_wa_perf_fast_path.py`](tests/test_wa_perf_fast_path.py) (4 casos sobre `is_excluded`).
- `RAG_WA_SKIP_PARAPHRASE` — rollback para el auto-skip de `expand_queries()` cuando el caller explícita `source="whatsapp"` (único valor). **Default ON** (`val not in ("0","false","no")` → skip paraphrase). Motivación (2026-04-22): cuando el caller filtra por WA, las 3 paraphrases de qwen2.5:3b + 3× embed bge-m3 cuestan ~600ms para near-zero recall gain — WA chunks son mensajes literales de ~143 chars donde los tokens lexicales originales ya dominan BM25 + semantic. Medido en bench sintético: retrieve avg 3001ms → 2404ms (-20%) sin pérdida de hit@5. Skipear sólo cuando el único source es whatsapp — multi-source callers (`source={"whatsapp","calendar"}`) mantienen paraphrase porque calendar todavía se beneficia. Rollback: `export RAG_WA_SKIP_PARAPHRASE=0`. Tests: [`tests/test_wa_perf_fast_path.py`](tests/test_wa_perf_fast_path.py) (7 casos sobre single/set/list/multi-source/no-source/explicit-false/rollback).
- `RAG_CONTRADICTION_PENALTY` — gating del **contradiction penalty post-rerank** ([`rag/contradictions_penalty.py`](rag/contradictions_penalty.py), 2026-04-29). **Default ON** (`val not in ("0","false","no","off","")` → enabled). Cuando está prendida, después del rerank pero antes del cap top-k, el pipeline consulta `rag_contradictions` (rows con `skipped IS NULL OR skipped = ''` — o `resolved_at IS NULL` si esa columna existe en futuros schemas) y baja el `score` por una magnitud configurable (default `-0.05`) a cualquier chunk cuyo `path` figure como `subject_path` o dentro del JSON `contradicts_json`. NO filtra — sólo demote suave + re-orden DESC, así que si el chunk con contradicción es genuinamente lo único relevante sigue saliendo. Skipeado cuando el caller pasa `--counter` (radar explícito de contradicciones, ahí el user QUIERE verlas). Cache TTL 5min in-process para evitar un SQL hit por retrieve(). Telemetry: counter `contradiction_penalty_applied` se loggea en `rag_queries.extra_json` (cuántos chunks fueron demoted). Rollback: `export RAG_CONTRADICTION_PENALTY=0` desactiva (helper ni se importa). Cuando triage marca una contradicción como resuelta (skill `rag-contradict-triage`), su row queda con `skipped` no-null y deja de bajar el score automáticamente al expirar el cache TTL. Tests: [`tests/test_contradiction_penalty.py`](tests/test_contradiction_penalty.py) (12 casos: empty table, paths aggregation, resolved exclusion, no-match no-op, single match demote, reorder cuando el gap se supera, env disable, counter skip, cache TTL hit/expiry, count helper, return-same-list contract).
- `RAG_CONTRADICTION_PENALTY_MAGNITUDE` — magnitud del demote (positivo, default `0.05`). El offset aplicado al score es `-magnitude`. Subir a `0.10` para penalty más agresivo (típico gap entre candidates consecutivos en rerank_logit ≈ 0.1, así que `0.10` mueve dos slots de promedio). Bajar a `0.02` para casi-no-mover. **Cuidado** con la eval gate (CLAUDE.md sección eval): magnitudes >0.10 pueden regresar singles/chains hit@5 si la tabla `rag_contradictions` tiene paths legítimamente relevantes (ej. una nota tiene 3 contradicciones pero sigue siendo la fuente más útil para esa query). El default 0.05 fue calibrado para mantener el régimen "ni se nota" en el eval. Tests: cobertura compartida con `RAG_CONTRADICTION_PENALTY` arriba.
- `RAG_DRAFT_VIA_RAGNET` — **legacy override** (introducida 2026-04-28): cuando está prendida (`1`/`true`/`yes`), TODOS los ambient sends a WhatsApp (morning brief, archive push, reminder push, contradicciones digest — todo lo que entra por `_ambient_whatsapp_send` en [`rag/integrations/whatsapp.py`](rag/integrations/whatsapp.py)) se redirigen al grupo **RagNet** (el grupo del bot, `WHATSAPP_BOT_JID`) en vez de mandarse al destinatario original. El payload mantiene el U+200B anti-loop al inicio y se prepende un header `📨 *RagNet draft* → \`<jid_original>\`` para que sea obvio qué iba a salir y a dónde. **NO afecta** sends user-initiated (`propose_whatsapp_send`, replies, scheduled, contact cards) — esos van directo via `_whatsapp_send_to_jid` y son explícitamente confirmados con [Enviar] en la proposal card. **Cuándo usarla (2026-04-29 en adelante)**: legacy override útil si querés redirigir TODOS los ambient sends a RagNet — no sólo los drafts del listener TS. Ejemplos típicos: testing previo a un rollout (querés ver qué dispararía el morning brief / archive push / reminder push sin spamear contactos reales por 1-2 días), debugging de un fetcher nuevo, o auditoría manual de qué payloads salen en una ventana acotada. **Cuándo NO usarla**: el flow normal (drafts del listener TS) ya está cubierto desde 2026-04-29 por el **review-only mode default** del listener WhatsApp en [`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener) (commit `8c16c42`). Esa flag complementaria — `WA_DRAFT_AUTO_FORWARD` del listener TS — gobierna el comportamiento default de los drafts proactivos: con la flag OFF (default) los drafts quedan sólo en RagNet para review, con la flag ON salen automáticamente al destinatario. Si tu única necesidad es "ver los drafts del listener antes de que se manden", `WA_DRAFT_AUTO_FORWARD=0` (el default) ya lo cubre — `RAG_DRAFT_VIA_RAGNET` queda redundante para ese flow específico. **Para flipearla**: `export RAG_DRAFT_VIA_RAGNET=1` en tu shell + reiniciar los daemons relevantes (`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist && launchctl bootstrap gui/$(id -u) ...`) para que hereden la env var. Para apagarla: `unset RAG_DRAFT_VIA_RAGNET` + bootout/bootstrap idem. Tests: [`tests/test_whatsapp_send_draft.py`](tests/test_whatsapp_send_draft.py) (5 casos cubren flag off, flag on con redirect + header, idempotencia cuando ya iba a RagNet, truthy/falsy variants, y que `_whatsapp_send_to_jid(anti_loop=False)` queda intacto).
- `RAG_WA_FAST_PATH` / `RAG_WA_FAST_PATH_THRESHOLD` — extiende el fast-path de `RAG_ADAPTIVE_ROUTING` a queries dominadas por WhatsApp. El default gate (inherited) exige `top_score > _LOOKUP_THRESHOLD` (0.6) — WA queries **NUNCA** alcanzan ese umbral porque la calibración absoluta del cross-encoder bge-reranker-v2-m3 sobre chats cortos (~143 chars) pone los scores en rango **0.02-0.10** incluso cuando los matches son perfectos. Medido 2026-04-22 post-reindex: "qué charlamos con María en marzo sobre laburo" → 0.056, "mensajes con Juli sobre su mudanza" → 0.092. Sin el gate WA → todas caían al pipeline full (qwen2.5:7b + num_ctx default + citation repair, gen 40-55s). **Gate con 2 ramas** (post-2026-04-22 tarde): (1) caller explícito `source="whatsapp"` → fast-path incondicional, bypassea score gate (el caller ya declaró intent, no necesitamos umbral — branch 1); (2) detección implícita: ≥2 de los top-3 metas tienen `source="whatsapp"` AND top-1 score > `RAG_WA_FAST_PATH_THRESHOLD` (default **0.05**, bajado de 0.3 porque a ese umbral NINGUNA query real disparaba branch 2). La majority check de branch 2 protege contra un chunk WA incidental flipeando el pipeline. **Default ON** (`RAG_WA_FAST_PATH != "0"`). Rollback: `export RAG_WA_FAST_PATH=0` desactiva ambas ramas; `RAG_WA_FAST_PATH_THRESHOLD=0.3` deshabilita branch 2 efectivamente (no afecta branch 1). Medición bench con data real (5 queries WA, post-reindex): retrieve avg 3751ms → 2584ms (-31%) — Track 2+3 en conjunto. Branch 1 dispara correctamente en 5/5 queries; branch 2 requiere scores que aparecen menos seguido pero también dispara cuando hay. Tests: [`tests/test_wa_perf_fast_path.py`](tests/test_wa_perf_fast_path.py) (21 casos, incluye branch 1 vs 2, score gate bypass, threshold override, rollback, interacciones con default gate).

- `RAG_ADAPTIVE_ROUTING` — activa el pipeline adaptativo (Mejora #3). Cuando está ON: (a) `_should_skip_reformulate()` saltea el helper call de reformulación para intents metadata-only (`count`, `list`, `recent`, `agenda`, `entity_lookup`) — ahorra ~1-2s; (b) fast-path dispatch cuando `top_score >= _LOOKUP_THRESHOLD` (0.6) — usa `qwen2.5:3b` con `num_ctx=4096` (bumped desde 2048 el 2026-04-22 por refuses falsos, ver `RAG_LOOKUP_NUM_CTX`) y saltea citation-repair, pure perf win para queries simples. **Default ON tras 2026-04-22** (`os.environ.get("RAG_ADAPTIVE_ROUTING", "").strip().lower() not in ("0","false","no")`). Sin regresión eval: ambas runs ON y OFF producen resultados bit-idénticos en `rag eval` (validado 2026-04-21, ver §Eval baselines). Rollback: `export RAG_ADAPTIVE_ROUTING=0` → pipeline legacy bit-idéntico. Ver `docs/improvement-3-adaptive-routing-design.md`.
- `RAG_LOOKUP_NUM_CTX` — tamaño de context window del fast-path LLM (`qwen2.5:3b`). **Default 4096 tras 2026-04-22** (pre-fix era 2048). El bump salió por un bug medido en producción: queries con alto top_score (>1.0) devolvían "No tengo esa información" aunque el chunk relevante estaba en el top-5 del rerank. Causa: system prompt + `SYSTEM_RULES*` + 5 chunks (cada uno con parent + neighbors) + graph expansion pueden superar los 2048 tokens, y qwen2.5:3b ve un contexto truncado al primer chunk del batch — típicamente NO el más relevante. 4096 cabe el contexto típico sin truncation + qwen2.5:3b todavía es ~2x más rápido que qwen2.5:7b full. Reproducible pre-fix: `RAG_LOOKUP_NUM_CTX=2048 rag query "charla sobre ansiedad y estrés laboral"` → refuse falso; con 4096 → responde bien. Tests: [`tests/test_adaptive_fast_path.py`](tests/test_adaptive_fast_path.py) (2 casos sobre default + env override). Para operadores con presión extrema de memoria que acepten el tradeoff: `export RAG_LOOKUP_NUM_CTX=2048`.
- `RAG_ENTITY_LOOKUP` — activa el dispatch de `handle_entity_lookup()` para el intent `entity_lookup` en `query()` (Mejora #2). **Default ON tras 2026-04-21** (`os.environ.get("RAG_ENTITY_LOOKUP", "").strip().lower() not in ("0","false","no")`). Motivación del flip: `scripts/backfill_entities.py` corrió sobre el corpus activo (2026-04-21 evening, 4150 chunks procesados en ~6 min) y pobló `rag_entities` con 2022 entidades + `rag_entity_mentions` con 6520 mentions (cobertura 71% = 2953/4150 chunks con al menos una entidad). Top persons: Astor (129 mentions), Juli (113), Maria (75), Fernando (51). Top orgs: Moka (94), BBI (35). Sources: vault (5784 mentions) + calendar (672) + gmail (47) + whatsapp (17). Smoke-test validado: `rag query "todo lo que tengo de Astor"` devuelve síntesis cross-source agregada (biografía + conversaciones María + Flor + WhatsApp + Calendar cumpleaños). Handler tiene fall-through a semantic si la entidad no resuelve o la tabla está vacía → sin riesgo de regresión (peor caso: +1 SQL query ~10 ms antes de caer al pipeline legacy). Para rollback: `export RAG_ENTITY_LOOKUP=0`. Instalar la dep: `uv pip install obsidian-rag[entities]` (GLiNER + transitivos, ~500 MB modelo cacheado en `~/.cache/huggingface/hub/`). Ver `docs/improvement-2-entity-retrieval-plan.md`.
- `RAG_EXTRACT_ENTITIES` — **default ON tras 2026-04-21** (permissive: `val not in ("0","false","no")`). Popula `rag_entities` + `rag_entity_mentions` automáticamente durante indexing — evita que la tabla quede stale cuando el corpus crece. Wired en 5 sitios: `rag.py:_index_single_file` (incremental vault + `rag watch`), `rag.py` bulk vault path (`rag index --reset`), y los 4 ingesters cross-source (`scripts/ingest_whatsapp.py`, `ingest_reminders.py`, `ingest_gmail.py`, `ingest_calendar.py`). Cada caller pasa el `source` correspondiente (`"vault"` / `"whatsapp"` / etc.) así las mentions quedan atribuidas. Silent-fail si `gliner` no está instalado (sticky-fail via `_gliner_load_failed`: primer ImportError/load-fail marca el flag y ningún retry en la vida del proceso — sin log spam). Costo: ~0.16s/chunk cuando gliner está disponible + cold-load GLiNER ~5s una vez por proceso; para `rag index --reset` sobre 4150 chunks eso son ~11 min extras. Para skipear durante un reindex grande y correr el backfill después: `RAG_EXTRACT_ENTITIES=0 rag index --reset && python scripts/backfill_entities.py`. El one-shot [`scripts/backfill_entities.py`](scripts/backfill_entities.py) queda útil para re-popular tras cambios de schema (bump `_COLLECTION_BASE`) o para llenar la tabla en hosts donde `gliner` se instaló después del indexing inicial.
- `RAG_NLI_GROUNDING` — activa NLI grounding post-citation-repair (Mejora #1). Carga `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (~400 MB en MPS fp32), extrae claims del LLM response, y agrega un bloque NLI-grounded al final con porcentaje de claims soportados/neutros/contradictorios. En web: emite SSE event `nli_grounding` + panel UI coloreado. **Default OFF** (truthy check: `os.environ.get("RAG_NLI_GROUNDING", "").strip() not in ("", "0", "false", "no")`). No aplica para intents `count`/`list`/`recent`/`agenda` (controlado por `RAG_NLI_SKIP_INTENTS`). **No activar en producción aún**: cold-load mDeBERTa ~2s + inference ~300-1000ms por query impactan UX antes de medir en ≥50 queries reales. Activar con `RAG_NLI_IDLE_TTL` (default 900s para idle-unload). Ver `docs/improvement-1-nli-integration-plan.md`.

Dev/debug toggles (not set in production):

- `RAG_DEBUG=1` — verbose stderr in the local embed path (`rag.py:6593`).
- `RAG_RETRIEVE_TIMING=1` — per-stage timing breakdown printed to stderr at the end of `retrieve()`.
- `RAG_NO_WARMUP=1` — skip the background reranker + bge-m3 + corpus warmup (shaves startup for lightweight commands like `rag stats`, `rag session list`; first query pays the cold-load cost).
- `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1` — short-circuit `get_context_summary()` to empty string. Used by tests + emergency fallback if qwen2.5:3b is unavailable; leaves embeddings without contextual prefix.
- `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q=1` — same, for `get_synthetic_questions()`.

## Architecture — invariants

### Retrieval pipeline (`retrieve()`)

```
query → classify_intent → infer_filters [auto]
      → [RAG_ADAPTIVE_ROUTING: skip reformulate_query if metadata-only intent]
      → expand_queries (3 paraphrases, ONE qwen2.5:3b call)
      → embed(variants) batched bge-m3
      → per variant: sqlite-vec sem + BM25 (accent-normalised, GIL-serialised — do NOT parallelise)
      → RRF merge → dedup → expand to parent section (O(1) metadata)
      → cross-encoder rerank (bge-reranker-v2-m3, MPS+fp32)
      → [RAG_ADAPTIVE_ROUTING + score≥0.6: fast-path → qwen2.5:3b num_ctx=2048, skip citation-repair]
      → graph expansion (1-hop wikilink neighbors, always on)
      → [auto-deep: if confidence < 0.10, iterative sub-query retrieval]
      → top-k → LLM (streamed)
      → citation-repair [if needed + score<threshold]
      → [RAG_NLI_GROUNDING: NLI claim verification, skip count/list/recent/agenda]

Intent dispatch: semantic | synthesis | comparison | count | list | recent | agenda | entity_lookup
```

**Graph expansion** (always on): after rerank, top-3 results expand via 1-hop wikilink neighbors (`_build_graph_adj` + `_hop_set`). Up to 3 graph neighbors added as supplementary LLM context marked `[nota relacionada (grafo)]`. Cost: in-memory graph lookups, negligible.

**Auto-deep retrieval**: when top rerank score < `CONFIDENCE_DEEP_THRESHOLD` (0.10), `deep_retrieve()` auto-triggers: helper model judges sufficiency → generates focused sub-query → second retrieve pass → merge results. Max 3 iterations (`_DEEP_MAX_ITERS`) + wall-time cap `_DEEP_MAX_SECONDS` (default 30s, env `RAG_DEEP_MAX_SECONDS`, added 2026-04-22). Disable with `--no-deep`. **Motivación del timeout** (audit 2026-04-22): el peor query en 7d tuvo `t_retrieve=202.6s` — factor 5x sobre el P99 normal (~38s). Causa: bajo contención intermitente de Ollama (reranker idle-unload + cold-load del chat model + 3 pases de `retrieve()` cada uno con `expand_queries()` + rerank) cada iteración puede pasar de 5s nominal a 60-70s → 3 iters = 200s. El guard absoluto garantiza que ninguna query bloquee más de 30s en retrieve aunque el juez siga pidiendo sub-queries. Tests: [`tests/test_deep_retrieve.py`](tests/test_deep_retrieve.py) (3 casos nuevos: timeout dispara, default es 30s, env override funciona).

**Rerank pool** (`RERANK_POOL_MAX = 15`, dropped from 30 on 2026-04-21): cap on candidates scored by the cross-encoder. `rag eval --latency` bench sweep (pool=30/25/20/15 × 60 singles + 30 chains):

| pool | singles hit@5 | singles MRR | chains MRR | singles P50 | singles P95 |
|---|---|---|---|---|---|
| 30 | 71.67% | 0.679 | 0.740 | 2157 ms | 4704 ms |
| 25 | 71.67% | 0.679 | 0.740 | 1899 ms | 2333 ms |
| 20 | 71.67% | 0.681 | 0.757 | 1458 ms | 1914 ms |
| **15** | **71.67%** | **0.681** | **0.790** | **1163 ms** | **1577 ms** |

Pool=15 domina: hit@5 bit-identical, MRR chains **+5pp** (less noise in the rerank head), P95 singles **−66%** (−3.1s). The pre-existing comment claiming "pool=25 pierde 1 hit (90.48% vs 95.24%)" was measured against a smaller/older `queries.yaml` and does not reproduce on the current corpus. Web `/api/chat` still passes `rerank_pool=5` explicitly; CLI inherits the default. `rag tune` now fits weights on pool=15 feature vectors (old weights fit on pool=30 should be re-tuned — default weights are unaffected).

**Corpus cache** (`_load_corpus`): BM25 + vocab built once, invalidated by `col.count()` delta. Cold 341ms → warm 2ms. Do not touch without re-measuring.

**Cache locks (concurrency invariants)** — the web server hits rag from multiple threads concurrently, so every module-level cache that gets written more than once is protected:

| Cache | Lock | Guards against |
|---|---|---|
| `_context_cache` / `_context_cache_dirty` | `_context_cache_lock` (Lock) | Double lazy-init + `json.dumps` during mutation |
| `_synthetic_q_cache` / `_synthetic_q_cache_dirty` | `_synthetic_q_cache_lock` (Lock) | Same |
| `_mentions_cache` | `_mentions_cache_lock` (Lock) | Parallel folder re-scan + overwrite race |
| `_embed_cache` | `_embed_cache_lock` (Lock) | LRU eviction race |
| `_corpus_cache` + `_pagerank_cache` + `_pagerank_cache_cid` | `_corpus_cache_lock` (RLock) | Partial-build reads from watchdog invalidation |
| `_contacts_cache` | `_contacts_cache_lock` (Lock) | Apple Contacts lookup race |

LLM calls (`_generate_context_summary`, `_generate_synthetic_questions`) run **outside** the lock so concurrent requests don't serialise on helper-model latency. Tests: `tests/test_cache_concurrency.py` (8 cases — presence, lazy-init uniqueness, save-during-mutation safety for each cache).

### Indexing

Chunks 150–800 chars, split on headers + blank lines, merged if < MIN_CHUNK. Each chunk: `embed_text` (prefixed `[folder|title|area|#tags]` + contextual summary), `display_text` (raw), `parent` metadata (enclosing section, ≤1200 chars). Hash per file → re-embed only on change. `is_excluded()` skips `.`-prefixed segments.

**Contextual embeddings** (v8→v9): `get_context_summary()` generates a 1-2 sentence document-level summary per note via qwen2.5:3b, prepended to each chunk's `embed_text` as `Contexto: ...`. Cached by file hash in `~/.local/share/obsidian-rag/context_summaries.json`. Notes < 300 chars skip summarization. The original commit claimed "+11% chain_success" but that figure was never replicated against the current queries.yaml — treat as unverified.

**Temporal tokens** (removed 2026-04-20, v10→v11): `temporal_token()` was defined in commit d6e1073 to append `[recent]`/`[this-month]`/`[this-quarter]`/`[older]` to the embedding prefix but was never actually wired into `build_prefix()` (dead code). The 2026-04-20 A/B wired it in (v10) + reindexed + re-ran `rag eval`: singles hit@5 / MRR / chains hit@5 / chain_success all within noise vs the v9 baseline (singles MRR −0.011, others bit-identical). Feature removed (v11) along with the `temporal_token()` function. If recency ever matters empirically, resurrect from git history rather than reintroducing the dead code.

**`created_ts` backfill — SQL-persisted marker** (2026-04-21): `_maybe_backfill_created_ts()` populates `created_ts` metadata en chunks que no la tenían (pre-temporal-feature indexes). Antes: el guard `_CREATED_TS_BACKFILL_DONE` era solo in-process — cada restart del web daemon (medido en `web.log`: **149 restarts** en ~3 días por `[idle-sweep] ... disabled` prints) re-escaneaba 3600+ chunks + 580 file reads en la primera query con `date_range`. Ahora el marker se persiste en `rag_schema_version` (table_name sentinel `_created_ts_backfill_complete`, version=1) tras un `col.update()` exitoso o cuando el escaneo determina que todos los chunks ya tienen el campo. `_created_ts_backfill_persisted()` lee la fila antes de escanear → skip instantáneo tras el primer proceso exitoso. Relevancia perf: el pre-fix del `col.update` (commit `d41bfe7`) dejaba el `AttributeError` causar **queries de hasta 200s** (retrieve=202652ms medido en web.log) — tras el fix el escaneo vale ~1s pero era gratis en restarts. Ahora es ~0s en restarts. Tests: [`tests/test_created_ts_backfill_persist.py`](tests/test_created_ts_backfill_persist.py) (9 casos: marker presence/absence, round-trip, idempotencia, skip-si-persistido, empty vault no-mark, mark-on-no-missing, defensivos contra DB broken, sentinel name no colide con tablas reales).

**Graph PageRank**: `_graph_pagerank()` computes authority scores over the wikilink adjacency graph (power iteration, <10ms). Cached per corpus. Used as a tuneable ranking signal (`graph_pagerank` weight) and to sort graph expansion neighbors.

**Schema changes**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v11`). Per-vault suffix = sha256[:8] of resolved path.

### Model stack

| Role | Model | Notes |
|------|-------|-------|
| Chat | `resolve_chat_model()`: qwen2.5:7b > qwen3:30b-a3b > command-r > qwen2.5:14b > phi4 | qwen2.5:7b default tras bench 2026-04-18 (total P50 5.9s vs 37s de command-r); fallbacks high-quality disponibles. |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float32` forced — do NOT switch to fp16 on MPS. Two A/Bs verifican el desastre por motivos distintos (collapse 2026-04-13, overhead 2× con calidad equivalente 2026-04-22). Detalle en §"Rerank fp16 A/B 2026-04-22 — NO PROMOTED" abajo. CPU fallback = 3× slower. |
| NLI (opt-in) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | ~400 MB MPS fp32, idle-unload via `RAG_NLI_IDLE_TTL` (default 900s). Gate: `RAG_NLI_GROUNDING`. |

All ollama calls use `keep_alive=OLLAMA_KEEP_ALIVE` — default `-1` (forever) in code (`rag.py:1608`) since 2026-04-21; launchd plists use the same value for symmetry. `CHAT_OPTIONS`: `num_ctx=4096, num_predict=384` — don't bump unless prompts grow.

**Pattern**: helper for cheap rewrites, chat model for judgment. Contradiction detector MUST use chat model (qwen2.5:3b proved non-deterministic + malformed JSON on this task).

### Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse without LLM call. Calibrated for bge-reranker-v2-m3 on this corpus. Re-calibrate if reranker changes.

### Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query`, `semantic` intent): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always in chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.
- `SYSTEM_RULES_LOOKUP` (intent `count`/`list`/`recent`/`agenda`): terse 1-2 sentences, exact "No encontré esto en el vault." refusal.
- `SYSTEM_RULES_SYNTHESIS` (intent `synthesis`): cross-reference ≥2 overlapping sources, must surface tension. Fires via `_INTENT_SYNTHESIS_RE` — triggers on `resumí/resumime/síntesis/sintetizame/integrame todo lo que hay sobre X`, `qué dice el vault sobre X`, `summary of|synthesis of|synthesize X`. Plain `qué es X` stays `semantic`.
- `SYSTEM_RULES_COMPARISON` (intent `comparison`): explicit `X dice A / Y dice B / Diferencia clave: …` structure. Fires via `_INTENT_COMPARISON_RE` — triggers on `diferencia(s)? entre X y Y`, `comparame X con Y`, `X vs/versus Y`, `en qué se diferencian X y Y`, `qué distingue X de Y`, `contraste entre X y Y`. Checked BEFORE synthesis (precedence) because `X vs Y` is inherently comparative. 49 tests in [`tests/test_classify_intent.py`](tests/test_classify_intent.py); golden queries in [`queries.yaml`](queries.yaml) at the "Comparison intent" + "Synthesis intent" sections.
- Routed through `system_prompt_for_intent(intent, loose)` at generation time (both `query()` and `chat()` paths). `--loose` always maps to `SYSTEM_RULES` for every intent.

### Agenda intent (2026-04-21 evening)

Fired by `_INTENT_AGENDA_RE` in `classify_intent`, checked **before `recent`** because both share temporal tokens ("hoy" / "esta semana" / "este mes"). Pre-fix, queries like `"qué tengo esta semana"` fell into `handle_recent` → listed vault notes sorted by `modified` desc. Post-calendar-ingest (368 events + 37 reminders), that was a user-facing bug — calendar was invisible to browsing queries.

Regex patterns (positive fires):
- Possessives: `mi agenda` / `mis eventos|calendario|reuniones|turnos|citas`
- `qué tengo|hay|tenemos` + temporal anchor (hoy/mañana/esta semana/etc)
- `qué eventos|reuniones|citas|turnos tengo|hay|tenemos` (event noun carries agenda semantic even without temporal)
- `agenda de|del|para X`

Negative guards (stays semantic/recent): specific lookups like `"cuándo es el workshop de X"`, `"turno con Y"` (source-weighted retrieve already resolves) + `"notas modificadas hoy"` / `"últimas notas"` stay on `recent`.

`handle_agenda(col, params, limit=20, *, question=None)` in `rag.py` filters metas by `source ∈ _AGENDA_SOURCES = frozenset({"calendar", "reminders"})` via `normalize_source`, sorts by `created_ts` desc (start_ts for calendar events, creation anchor for reminders). Gracefully handles missing/string/malformed `created_ts` (coerce + try/except → 0.0).

**Temporal window filter (v2)**: `_parse_agenda_window(question, *, now=None) → (ts_start, ts_end) | None` parses the anchor and returns a half-open epoch interval. Dispatch order (narrowest first, critical for regex overlaps):
1. Day anchors (`pasado mañana` before `mañana`, then `hoy`)
2. Weekend (`el finde` / `este finde` / `próximo finde` — via dateparser for "finde" = saturday)
3. Week (`esta semana` / `la semana que viene` / `la próxima semana`)
4. Month (`este mes` / `el próximo mes` / `el mes que viene`)
5. Year (`este año` / `el próximo año` / `el año que viene`)
6. Weekday-specific (`el viernes` / `el próximo lunes` — delegated to `_parse_natural_datetime` so rollforward math matches the propose tool)

All windows snap to 00:00 local (naive datetime, `RAG_TIMEZONE` default `America/Argentina/Buenos_Aires`). December wraparound (`el próximo mes` from Dec → Jan next year) tested. Half-open semantics: event at exactly `end_ts` belongs to the NEXT window.

When `question` is passed to `handle_agenda` and the parser returns a window, post-source-filter metas are narrowed to `ts_start <= created_ts < ts_end`. No anchor (e.g. plain `"mi agenda"`) → fallback to top-20 by ts desc.

Tests: `tests/test_classify_intent.py` (+25 agenda cases — positives across possessives/tengo/hay/eventos/agenda shapes, negatives to guard against over-matching), `tests/test_handle_agenda.py` (13 cases: source filter + sort + limit + null ts + dedup + `_AGENDA_SOURCES` lock + 4 window-filter integration cases), `tests/test_agenda_window.py` (25 cases: every anchor type + ordering guards + half-open + December wraparound).

### Prompt-injection defence (passive, 2026-04-21)

Cross-source corpus (Gmail / WhatsApp, user override §10.5 = indexá-todo) means a hostile email or WhatsApp can land in the index and reach the LLM context through a legitimate semantic match. Two passive layers in `rag.py` (right above `SYSTEM_RULES`):

- **Redaction** — `_redact_sensitive(text)` strips OTPs, tokens, passwords, CBU / card / account numbers, CVV/CVC/CCV *before* the chunk body hits the LLM. Cue-gated (value must sit next to `code|token|password|cbu|cvv|...` within ~20 chars) with a digit-presence lookahead to avoid false positives (the regex `cue="code"` alone tripped on prose like "the code base is large"; the `(?=[A-Z0-9]*[0-9])` lookahead drops those matches). Embeddings stay indexed with the raw value — this defence only hides values from the LLM at generation time. NOT a barrier against a motivated attacker with vault write access.

- **Context isolation** — `_format_chunk_for_llm(doc, meta, role)` centralises the per-chunk wrapping: header `[{role}: {note}] [ruta: {file}]` stays identical (citation-repair + path-extraction rules in every `SYSTEM_RULES*` keep working unchanged), body goes between `<<<CHUNK>>>...<<<END_CHUNK>>>` fences after redaction. Paired with `_CHUNK_AS_DATA_RULE` — a prepended `REGLA 0` in every `SYSTEM_RULES*` variant that tells the model content between fences is DATA to cite, NEVER instructions. Hint to the classifier, not a cryptographic barrier; a sufficiently capable model may still follow injected instructions.

Callers of `_format_chunk_for_llm`: `build_progressive_context` (primary + multi-vault), `query()` graph section, `chat()` graph section, `rag serve` generation block. All four legacy inline formats `f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"` were migrated in the 2026-04-21 pass — any new caller assembling chunks for an LLM prompt MUST go through the helper so redaction + fencing stay centralised.

Tests: [`tests/test_prompt_injection_defence.py`](tests/test_prompt_injection_defence.py) (61 cases: OTP positives en ES + EN + unaccented, bank secret positives, negative cases for version strings / dates / commit SHAs / prose like "code base", chunk wrapper contract, `REGLA 0` presence + ordering in every `SYSTEM_RULES*`).

### Name-preservation guardrail (2026-04-21)

Separate from prompt-injection, a second always-on clause `_NAME_PRESERVATION_RULE` in `rag.py` (right below `_CHUNK_AS_DATA_RULE`) blocks a distinct failure mode: the LLM "correcting" proper nouns it doesn't recognise. Regression seed: user asked about "Bizarrap" (Argentine producer), the vault had no musical info, the model answered refusing about **"Bizarra"** — silently swapping a rare proper noun for a commoner dictionary word. Rule is prepended right after `_CHUNK_AS_DATA_RULE` in every `SYSTEM_RULES*` (ordering: chunks-as-data → names → variant body) so the model copies user-supplied names TEXTUAL and treats unknown terms as valid proper nouns it doesn't know. Verify with `python -c "import rag; print(rag._NAME_PRESERVATION_RULE[:80])"`. Tests: [`tests/test_name_preservation_rule.py`](tests/test_name_preservation_rule.py) (46 cases: constant presence, per-variant inclusion + ordering + no-duplication, `system_prompt_for_intent()` coverage for every intent incl. `loose=True`).

### Response-quality post-pipeline

**Citation-repair** (always-on): after generation, `verify_citations(full, metas)` flags invented paths. If non-empty, ONE repair call runs (`resolve_chat_model()` + `CHAT_OPTIONS`, non-streaming, `keep_alive=-1`) with system prompt `"Solo puedes citar las siguientes rutas: [...]. ... No inventes otras."` If repair output also has bad citations or is empty → keep original. On success → replace `full` silently (interactive: reprints via `render_response`; plain: single `click.echo` deferred until AFTER repair + critique). Logs `citation_repaired: bool` to `queries.jsonl`.

**`--critique` flag** (opt-in, both `rag query` and `rag chat`, plus `/critique` chat toggle): after citation-repair, second non-streaming chat-model call evaluates + regenerates if needed. Whitespace-normalized diff vs original → replace + `critique_changed=True`. Logs `critique_fired` (always equals flag state) + `critique_changed` to `queries.jsonl`. Adds one extra ollama round-trip only when flag is set — off by default so no latency cost.

### Rendering

OSC 8 `file://` hyperlinks for both `[Label](path.md)` and `[path.md]` formats. `NOTE_LINK_RE` handles single-level balanced parens. `verify_citations()` flags unknown paths post-response (feeds citation-repair loop).

### Scoring formula (post-rerank)

```
score = rerank_logit
      + w.recency_cue            * recency_raw         [if has_recency_cue]
      + w.recency_always         * recency_raw         [always]
      + w.tag_literal            * n_tag_matches
      + w.graph_pagerank         * (pr/max_pr)         [wikilink authority signal]
      + w.click_prior            * ctr_path            [behavior: path CTR, Laplace-smoothed]
      + w.click_prior_folder     * ctr_folder          [behavior: top-level folder CTR]
      + w.click_prior_hour       * ctr_path_hour       [behavior: path × current-hour CTR]
      + w.click_prior_dayofweek  * ctr_path_weekday    [behavior: path × weekday CTR]
      + w.dwell_score            * log1p(dwell_s)      [behavior: mean dwell time per path]
      - w.contradiction_penalty  * log1p(n_contrad_ts) [contradicciones recientes en rag_contradictions, 90d window]
      + w.feedback_pos                                 [if path in feedback+ cosine≥0.80]
      - w.feedback_neg                                 [if path in feedback- cosine≥0.80]
```

Weights in `~/.local/share/obsidian-rag/ranker.json` (written by `rag tune --apply`). Defaults `recency_always=0, tag_literal=0, click_*=0, dwell_score=0, contradiction_penalty=0` preserve pre-tune behavior. Behavior + contradiction knobs are inert until `rag_behavior` / `rag_contradictions` accumulate signal y `rag tune` finds non-zero weights.

Behavior priors (`_load_behavior_priors()`): read from `rag_behavior` (SQL), cached per MAX(ts). Positive events: `open`, `positive_implicit`, `save`, `kept`. Negative: `negative_implicit`, `deleted`. CTR uses Laplace smoothing `(clicks+1)/(impressions+10)`.

### GC#2.C — Reranker fine-tune (infra ready, gated on data)

- **Estado**: infra completa + gate E2E validado, esperando ≥20 rows con `corrective_path` en `rag_feedback` antes de re-correr con chances de promover.
- **Run anterior fallido** (`~/.cache/obsidian-rag/reranker-ft-20260422-124112/`, 2.1 GB, cleanup manual pendiente con `rm -rf` por ask-rule): −3.3pp chains hit@5 vs baseline. Causa: 1 epoch undertraining + señal positiva ruidosa (55 turns positivos × ~4 chunks cada uno, todos label=1.0 aunque solo uno era golden).
- **Run 2 noisy** (`~/.cache/obsidian-rag/reranker-ft-20260422-182127/`, 2.1 GB, gate=0 override, 3 epochs, ver [`docs/finetune-run-2026-04-22.md`](docs/finetune-run-2026-04-22.md)): **mismo −3.3pp chains**. Loss convergió de 0.96 a 0.13 (overfitting claro en epoch 3); val margin +0.455 (pos 0.515 vs neg 0.060). El modelo aprendió muy bien la data ruidosa — por eso regresionó chains. **Gate E2E validado**: detectó la regresión y NO promovió. Conclusión firme: sin `corrective_path` limpios, el fine-tune no supera baseline con esta config — hace falta la señal limpia, no más epochs.
- **Fix aplicado**: [`scripts/finetune_reranker.py`](scripts/finetune_reranker.py) ahora lee `corrective_path` de `rag_feedback.extra_json` y lo usa como único positivo cuando está presente. Fallback a todos los paths cuando no.
- **Gate pre-training**: `RAG_FINETUNE_MIN_CORRECTIVES` (default 20). Aborta con exit 5 si la señal limpia es insuficiente.
- **Cómo generar data**:
  - `rag chat` + thumbs-down en turnos malos — el prompt pide el path correcto (commit `23f2899`). La web UI tiene el mismo picker (commit `33ed3f0`).
  - `rag feedback backfill` — rescata corrective_path de turns ya en `rag_feedback` que no lo tienen (aplica a los 55 positivos del run del 2026-04-22 que no recibieron el prompt — el commit 23f2899 es posterior). Muestra query + top-5 del turn, aceptás [1-5]/texto libre/skip/quit. Update in-place via `json_set()`, nunca duplica rows.
  - `rag feedback harvest` — equivalente CLI del skill `rag-feedback-harvester` de Claude Code. Lista queries recent low-confidence sin thumbs y pide [+N/-/c/s/q]. Tagged `source='harvester'` + `original_query_id` en `extra_json` para trazabilidad.
  - `rag feedback status` — progress hacia los 20 del gate + breakdown por bucket (pos_no_cp / neg_no_cp) + comando exacto para re-disparar el fine-tune cuando el gate está open.
  - `rag feedback infer-implicit [--window-seconds N --dry-run --json]` — derivador batch: para cada `rating=-1` que no tenga `corrective_path` todavía, prueba dos ramas (en orden): (1) **opens-based** — busca un `open` en `rag_behavior` dentro del window, mismo `session_id`, path distinto al top-1 que el ranker eligió → ESE es el corrective; `corrective_source = "implicit_behavior_inference"`. (2) **paráfrasis fallback** (post-2026-04-29) — si no hubo opens, busca en `rag_queries` una follow-up query en la misma session que sea paráfrasis del original (`requery_detection.is_paraphrase`) Y cuyo `top_score >= 0.5` Y top-1 distinto del original → ESE top-1 es el corrective; `corrective_source = "implicit_paraphrase_inference"`. **Window default 600s (10 min)** — pre-2026-04-29 era 60s, lo subimos porque con 60s cerrábamos solo 1 corrective_path en 6 días (el user lee la nota abierta antes de actuar, no abre otra al toque). Idempotente: skipea feedbacks que ya tienen `corrective_path`. **Valores posibles de `corrective_source`**: `implicit_behavior_inference` (rama 1, señal fuerte), `implicit_paraphrase_inference` (rama 2, señal más débil pero útil para destrabar el gate de 20 que dispara el LoRA fine-tune del reranker). Tests: [`tests/test_implicit_learning_corrective.py`](tests/test_implicit_learning_corrective.py) (rama opens) + [`tests/test_corrective_paraphrase_fallback.py`](tests/test_corrective_paraphrase_fallback.py) (rama paráfrasis + backwards-compat).
- **Miner JSONL como data alternativa** (commit `5f33d44`): [`scripts/export_training_pairs.py`](scripts/export_training_pairs.py) complementa `rag_feedback` directo con signal implícita de `rag_behavior` (`copy`/`open`/`save`/`kept`/`positive_implicit`) + hard-negs mined de `impression` events reales del historial (no re-retrieve). Análisis del JSONL actual: [`docs/training-pairs-miner-analysis-2026-04-22.md`](docs/training-pairs-miner-analysis-2026-04-22.md) — 176 pairs, ratio neg:pos 6.1:1 (vs 2.4:1 del run previo), 74% con ≥5 hard-negs. Calidad superior al run noisy → próximo intento con estos pairs + `--epochs 2` tiene chances reales de pasar el eval gate. Integración al finetune pendiente (zona de abp2vvvw actual).
- **Monitoreo**: `sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db "SELECT COUNT(*) FROM rag_feedback WHERE json_extract(extra_json, '\$.corrective_path') IS NOT NULL AND json_extract(extra_json, '\$.corrective_path') <> ''"` — conteo directo de corrective_paths disponibles. (Post split 2026-04-21 `rag_feedback` vive en `telemetry.db`, no `ragvec.db`.)
- **Re-trigger**: `python scripts/finetune_reranker.py --epochs 2` una vez que el gate lo permita (2 epochs, no 3 — la loss convergió a 0.22 en epoch 2 en el run noisy; epoch 3 es overfit puro). El gate de `rag eval` decide promoción via symlink `~/.cache/obsidian-rag/reranker-ft-current`.

## Key subsystems — contracts only

Subsystems have autodescriptive docstrings in `rag.py` and dedicated test files. Only contracts/invariants here.

**Sessions**: JSON per session in `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validated `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp passes `wa:<jid>`.

**Episodic memory** (`web/conversation_writer.py`, silent write): after every `/api/chat` `done` event, `web/server.py` spawns a daemon thread via `_spawn_conversation_writer` that appends the turn to `04-Archive/99-obsidian-system/99-AI/conversations/YYYY-MM-DD-HHMM-<slug>.md` (pre-2026-04-25: `00-Inbox/conversations/` — el user pidió que las "carpetas de sistema" vivan bajo `99-obsidian-system/99-AI/` para que su PARA quede limpio). One note per `session_id`, multi-turn. Hand-rolled YAML frontmatter (`session_id`, `created`, `updated`, `turns`, `confidence_avg`, `sources`, `tags`). The session_id → relative_path index lives in `rag_conversations_index` (SQL, upsert). Atomic .md write via `os.replace`; concurrent writes for the same session are not a production scenario (one /api/chat per session at a time) so the pre-T10 whole-body fcntl lock is gone — SQL upsert inside `BEGIN IMMEDIATE` handles index serialisation. Errors land on `LOG_PATH` as `conversation_turn_error` — never raised, never SSE-emitted. Raw conversations are **excluded from the search index** (`is_excluded`: prefix general `04-Archive/99-obsidian-system/` + el legacy `00-Inbox/conversations/` por compat + `04-Archive/conversations/` para los archivados post-consolidación) — leak LLM hallucinations back as ground truth if indexed (T6 regression). Curation happens via `rag consolidate` (Phase 2, below), not by manual editing.

**Conversation writer shutdown drain** (`_CONV_WRITERS` + `@app.on_event("shutdown")`): every in-flight writer registers in `_CONV_WRITERS` and removes itself when `_persist_conversation_turn` returns (success or exception). On server stop the `_drain_conversation_writers` hook joins each pending thread with a combined 5s budget. Anything still running falls through the normal exception path, lands in `_CONV_PENDING_PATH` (`conversation_turn_pending.jsonl`), and gets re-applied at next startup by `_retry_pending_conversation_turns`. Threads stay `daemon=True` on purpose — a wedged SQL/disk write must not block process exit. Stragglers past the cap are logged once as `conversation_writer_shutdown_timeout` to `LOG_PATH`. Tests: `tests/test_web_conv_shutdown.py` (6 cases covering self-remove, empty drain no-op, quick-writer wait, 5s cap with wedged writer, spawn tracking, exception-path release).

**Episodic memory — Phase 2 consolidation** (`scripts/consolidate_conversations.py`, `rag consolidate`, weekly launchd): scans `04-Archive/99-obsidian-system/99-AI/conversations/` in a rolling window (default 14d), embeds each as `first_question + first_answer` via bge-m3, groups by connected components on cosine ≥ 0.75, promotes clusters ≥ 3 to PARA. Target folder picked by regex over cluster bodies: ≥2 matches against `_PROJECT_PATTERNS` (ES+EN action verbs / future-tense / dates) → `01-Projects/`, else `03-Resources/` (conservative default). Synthesis via `resolve_chat_model()` + `CHAT_OPTIONS` — one non-streaming call per cluster (~6s). Consolidated note gets frontmatter `type: consolidated-conversation`, wikilink section to originals (now under `04-Archive/conversations/YYYY-MM/`), and wikilinks to every source note union'd across turns. Originals move via `shutil.move`; archive folder is also excluded from the index so archived raws don't compete with the curated synthesis. Errors per cluster are swallowed (cluster entry gets `error` key; other clusters proceed). Log schema at `~/.local/share/obsidian-rag/consolidation.log` (JSONL: `{run_at, window_days, n_conversations, n_clusters, n_promoted, n_archived, duration_s, dry_run, clusters: [...]}`). CLI flags: `--window-days`, `--threshold`, `--min-cluster`, `--dry-run`, `--json`. Launchd: `com.fer.obsidian-rag-consolidate` (Mondays 06:00 local), registered in `_services_spec()`, installable via `rag setup`.

**Web chat tool-calling** (`web/tools.py`, 9 tools): `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent, implementations live in `rag.py` — `web/tools.py` re-exports). `/api/chat` runs a 2-phase tool loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated by `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá un evento", ...) is detected by `_detect_propose_intent` (defined in `rag.py`, shared between web + CLI) which FORCES the LLM decide round ON for that query — propose tools need LLM arg extraction, can't run from pre-router. Create tools auto-create the reminder/event if the datetime is unambiguous (SSE `created` event → inline `╌ ✓ agregado...` chip, reminders get an inline `deshacer` link backed by `DELETE /api/reminders/{id}`, events don't since Calendar.app AppleScript delete is unreliable) or fall back to a `proposal` card with ✓ Crear / ✗ Descartar when the parser flagged `needs_clarification`. Low-level helpers `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback, `_preprocess_rioplatense_datetime` for `18hs`/`al mediodía`/`X que viene`), `_parse_natural_recurrence` (regex over ES/EN patterns), `_create_reminder` (supports `due_dt`, `priority`, `notes`, `recurrence`), `_create_calendar_event` (via Calendar.app AppleScript — iCloud writable, unlike the JXA read path), `_has_explicit_time` (auto all-day detection), `_delete_reminder`, `_delete_calendar_event` all in `rag.py`. Recurrence on Reminders is best-effort (inner try/on error) since the property is macOS-version-dependent; on Calendar it's stable.

**CLI chat create-intent** (`rag chat`): same `_detect_propose_intent` + same propose tools, but ported to terminal via `_handle_chat_create_intent` at the top of every turn's input. Single-round ollama tool-decide with `_CHAT_CREATE_OVERRIDE` prompt + `tools=[propose_reminder, propose_calendar_event]` only; on tool_call → dispatches + renders a Rich chip `╌ ✓ agregado...` in the same `sáb 25-04 (todo el día)` / `lun 20-04 22:27` shape as the web UI (hard-coded `es-AR` weekdays because `%a` is locale-dependent). command-r's `{parameters: {...}}` arg wrapping is unwrapped the same way as `rag do`. Returns `(handled, created_info)` where `created_info` carries `{kind, reminder_id, title}` on a successful reminder create (None for events — Calendar.app AppleScript delete is unreliable, matches web UX which shows no undo for events). The chat loop stashes `created_info` in `last_created` (session-local, not persisted) and the `/undo` slash command dispatches `_delete_reminder(last_created["reminder_id"])` to reverse the most recent create; `last_created` clears on success so a second `/undo` returns "nothing to undo". Tests: `tests/test_chat_create_handler.py` (8 cases) + `tests/test_chat_undo.py` (5 cases) — all monkeypatched, no live ollama.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`, runs before `dateparser` inside `_parse_natural_datetime`): dateparser 1.4 handles maybe 30% of AR-idiom inputs correctly and silently echoes the anchor time for another 30% (e.g. "a las 10 de la mañana" → anchor time). We hand-roll regex rewrites that normalize to forms dateparser CAN parse — mostly English equivalents with `PREFER_DATES_FROM=future`. Covers: `18hs` → `18:00`; `al mediodía` → `12:00`; `X que viene` → bare weekday/`next week`/`next month`; `el|este|próximo <weekday>` → bare English weekday (because dateparser 1.4 rejects `next <weekday>` silently but accepts bare `thursday` with future-prefer); `pasado mañana` → `day after tomorrow`; `a las N de la mañana|tarde|noche` → `N:00 am`/`(N+12):00`; `a la mañana|tarde|noche|tardecita` → default hour (09/16/20/17); `tipo N` / `a eso de las N` → `N:00` (rioplatense approximations); diminutives (`horitas` → `horas`); `el finde` → `saturday`. Anchor-echo guard after dateparser: if the input carries a time marker but dateparser returned exactly the anchor time, discard and fall through to LLM. LLM fallback prompt (qwen2.5:3b, `HELPER_OPTIONS` deterministic) explicitly flags rioplatense, passes both raw text and normalized hint, and instructs rollforward for bare weekdays + 09:00 default for missing times.

**Ambient agent**: hook in `_index_single_file` on saves within `allowed_folders` (default `["00-Inbox"]`). Config: `~/.local/share/obsidian-rag/ambient.json` (`{jid, enabled, allowed_folders?}`). Skip rules: outside allowed_folders, no config, frontmatter `ambient: skip`, `type: morning-brief|weekly-digest|prep`, dedup 5min (upsert on `rag_ambient_state.path`). Sends via `whatsapp-bridge` POST (`http://localhost:8080/api/send`). Bridge down = message lost but analysis persists in `rag_ambient`.

**Contradiction radar**: Phase 1 (query-time `--counter`), Phase 2 (index-time frontmatter `contradicts:` + `rag_contradictions`), Phase 3 (`rag digest` weekly). Skipped on `--reset` (O(n²)) and `note_body < 200 chars`.

**Contradicciones → ranker penalty (loop cerrado, default OFF)**: `_load_contradiction_priors()` lee `rag_contradictions` (window 90d, cap 5000 paths) y devuelve `{subject_path: log1p(count_distinct_ts)}` — penaliza más fuerte las notas con detecciones SEPARADAS en el tiempo (señal robusta de contradicción persistente, no falso positivo único). Consumido por (a) `retrieve()` como debuff `final -= weights.contradiction_penalty * priors.get(path, 0.0)` después del bloque behavior priors, (b) `collect_ranker_features()` como feature `contradiction_count` (14ta del LightGBM ranker, última posición). Default `weights.contradiction_penalty = 0.0` → loop OFF: hasta que `rag tune` no lo levante (range `(0.0, 0.30)` en `_TUNE_SPACE`), el comportamiento es bit-idéntico al pre-feature. Silent-fail en el read SQL → `{}` (retrieve sigue funcionando sin priors). Schema `rag_contradictions` ya existía (populado por `_log_contradictions` cuando `rag contradictions <path>` corre); este es el primer consumer que lo realimenta al ranker. Tests: `tests/test_contradiction_priors.py`, `tests/test_retrieve_contradiction_penalty.py`, `tests/test_ranker_lgbm_contradiction_feature.py`.

**URL sub-index**: `obsidian_urls_v1` collection embeds **prose context** (±240 chars) not URL strings. `PER_FILE_CAP=2`. Auto-backfill on first `find_urls()` if collection empty.

**Wikilinks**: regex scan against `title_to_paths`. Skips: frontmatter, code, existing links, ambiguous titles, short titles (min-len 4), self-links. First occurrence only. Apply iterates high→low offset.

**Archive**: reuses `find_dead_notes`, maps to `04-Archive/<original-path>` (PARA mirror), stamps frontmatter `archived_at/archived_from/archived_reason`. Opt-out: `archive: never` or `type: moc|index|permanent`. Gate: >20 candidates without `--force` → dry-run. Batch log in `filing_batches/archive-*.jsonl`.

**Morning**: collects 36h window (modified notes, inbox, todos, contradictions, low-conf queries, Apple Reminders, calendar, weather, screentime). Weather hint only if rain ≥70%. Dedup vault-todos vs reminders (Jaccard ≥0.6). System-activity + Screen Time sections are deterministic (no LLM).

**Screen Time**: `_collect_screentime(start, end)` reads `~/Library/Application Support/Knowledge/knowledgeC.db` (`/app/usage` stream, read-only via `immutable=1` URI). Sessions <5s filtered. Bundle→label map + category rollup (code/notas/comms/browser/media/otros). Renders only if ≥5min of activity. Section omitted silently if db missing. Dashboard `/api/dashboard` exposes 7d aggregate + daily series (capped at 7 — CoreDuet aggregates older data away).

**Today**: `[00:00, now)` window, 4 fixed sections, writes `-evening.md` suffix. Feeds next morning organically.

**Followup**: extracts loops (frontmatter todo/due, unchecked `- [ ]`, imperative regex), classifies via qwen2.5:3b judge (temp=0, seed=42, conservative). One embed + one LLM call per loop.

**Read**: fetch URL → readability strip → gate (< 500 chars = error) → command-r summary → two-pass related lookup → tags from existing vocab (never invents) → `00-Inbox/`. Dry-run default, `--save` to write.

**Ranker-vivo (closed-loop ranker)**: implicit feedback from daily use re-tunes `ranker.json` nightly without manual intervention. Four signal sources insert into `rag_behavior`: (1) CLI `rag open` wrapper (opt-in via `RAG_TRACK_OPENS=1` + user-registered `x-rag-open://` handler); (2) WhatsApp listener classifying follow-up turns (`/save`, quoted reply → positive; "no"/"la otra"/rephrase → negative; 120s silence → weak positive); (3) web `/api/behavior` POST from home dashboard `sendBeacon` clicks; (4) morning/today brief diff (`_diff_brief_signal` compares yesterday's written brief vs current on-disk — wikilinks that survived = `kept`, missing = `deleted`, dedup via `rag_brief_state`). Nightly `com.fer.obsidian-rag-online-tune` at 03:30 runs `rag tune --online --days 14 --apply --yes`, which calls `_behavior_augmented_cases` (weight=0.5, drops conflicts), backs up current `ranker.json` → `ranker.{ts}.json` (keeps 3 newest), re-tunes, runs the bootstrap-CI gate (`_run_eval_gate`: scrubs `RAG_EXPLORE`, subprocess `rag eval`, 10min cap, regex parses hit@5). If singles < `GATE_SINGLES_HIT5_MIN` (default 0.60, override via `RAG_EVAL_GATE_SINGLES_MIN`) OR chains < `GATE_CHAINS_HIT5_MIN` (default 0.73, override via `RAG_EVAL_GATE_CHAINS_MIN`) → auto-rollback + exit 1 + log to `rag_tune`. `rag tune --rollback` restores the most recent backup manually. **Floor recalibrados 2026-04-23** desde los originales 0.7619 / 0.6364: con la expansión de `queries.yaml` (42→60 singles post-2026-04-21, +cross-source/synthesis/comparison goldens deliberadamente más duros), el baseline estable cayó a 71.67% / 86.67% y los floors fueron rebajados a los nuevos CI lower bounds (mismo criterio metodológico: "95% confianza de que corridas bajo el floor son regresión real, no noise"). Ver el bloque de comentarios sobre `GATE_SINGLES_HIT5_MIN` en `rag.py` (~línea 23121) para la timeline completa.

## Eval baselines

**Floor (2026-04-27, post-golden-remap vault reorg, commit 6f8994f)** — vault reorg eliminó paths que ya no existen; golden remap redujo el set de n=60→54 singles / n=12→9 chains. Dos corridas reproducibles (bit-idénticas en hit@5 + chain_success; MRR chains dentro de CI):
- Singles: `hit@5 53.70% [40.74, 66.67] · MRR 0.528 [0.407, 0.657] · n=54`
- Chains: `hit@5 72.00% [52.00, 88.00] · MRR 0.633–0.653 [0.460, 0.820] · chain_success 33.33% [11.11, 66.67] · turns=25 chains=9`
- Lower-CI-bound gate (nightly online-tune auto-rollback): singles < 40.74% OR chains < 52.00%
- Nota: la caída vs el floor previo (singles 71.67% → 53.70%, chains 86.67% → 72.00%) NO es regresión del pipeline — es reducción del n y remoción de goldens fáciles que ya no existen en el vault post-reorg. Las queries removidas pertenecían mayoritariamente a `01-Projects/RAG-Local/*` (notas movidas al .trash/ por el user). Los floors nuevos codifican el mismo criterio: "95% confianza de que una corrida bajo el floor es regresión real, no noise".

**Floor (2026-04-17, post-golden-expansion + bootstrap CI)** — queries.yaml doubled (21→42 singles, 9→12 chains; +15 singles in under-represented folders 03-Resources/Agile+Tech, 02-Areas/Personal, 01-Projects/obsidian-rag, 04-Archive memory). `rag eval` now reports percentile bootstrap 95% CI (1000 resamples, seed=42) alongside each metric + `rag eval --latency` reports P50/P95/P99 of retrieve() per bucket and accepts `--max-p95-ms` as a CI gate.
- Singles: `hit@5 88.10% [76.19, 97.62] · MRR 0.772 [0.651, 0.873] · n=42`
- Chains: `hit@5 78.79% [63.64, 90.91] · MRR 0.629 [0.490, 0.768] · chain_success 50.00% [25.00, 75.00] · turns=33 chains=12`
- Latency: singles p95 2447ms · chains p95 3003ms

Every post-expansion metric sits inside the prior floor's CI on the smaller set — expansion surfaced the noise band (~21pp singles hit, ~50pp chain_success) that previously masqueraded as drift.

**Post prompt-per-intent + citation-repair (2026-04-19):** Singles `hit@5 88.10% [76.19, 97.62] · MRR 0.767 [0.643, 0.869]` — identical hit@5, MRR within CI. Chains `hit@5 81.82% [66.67, 93.94] · MRR 0.636 [0.505, 0.773] · chain_success 58.33% [33.33, 83.33]` — +3pp hit@5, +8pp chain_success, both inside prior CI so treat as noise until replicated. Floor unchanged for auto-rollback gate (still 76.19% / 63.64%).

**Post golden-set re-mapping (2026-04-20):** vault reorg (PARA moves: many notes `02-Areas/Coaching/*` → `03-Resources/Coaching/*`, `03-Resources/{Agile,Tech}/*` → `04-Archive/*`, etc.) left 33 of 65 `expected` paths in `queries.yaml` pointing at dead files, artificially cratering eval to singles hit@5 26% / chains 33%. Golden rebuilt by auto-mapping 31 unique paths via filename-stem lookup to the closest surviving note (prefer non-archive, bias `01→02→03→04` for tie-breaks) and dropping one chain whose source notes (`reference_{claude,ollama}_telegram_bot.md`) no longer exist. Post-rebuild eval: Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810]`; Chains `hit@5 75.76% [60.61, 90.91] · MRR 0.641 [0.510, 0.788]`. Both CIs overlap the 2026-04-19 run — within noise band. Floor unchanged (76.19% / 63.64%); current singles 78.57% and chains 75.76% pass the auto-rollback gate.

**Post-T10 (2026-04-20, after JSONL-fallback strip, commit `81e32b4`):** Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810] · recall@5 76.19% · n=42`; Chains `hit@5 86.67% [73.33, 96.67] · MRR 0.728 [0.594, 0.850] · chain_success 63.64% [36.36, 90.91] · turns=30 chains=11`. Singles **bit-identical** vs pre-T10 (expected — T10 is pure storage refactor, retrieval pipeline untouched); chains drifted +11pp inside prior CI (same noise band). Latency: singles p95 2797ms, chains p95 3406ms — slight uptick vs pre-T10 (2447/3003ms) attributable to SQL being the only write path (no JSONL-queue offload anymore). Still ×5 below any action threshold. Floor gate passed at the exact chain_success boundary (63.64%) — fine this run but worth re-measuring next tune cycle.

**Post cross-source corpus (2026-04-21, n=55 singles / 11 chains):** Primer eval con el corpus mixto activo — 20 chunks gmail + 36 chunks reminders + 4071 chunks whatsapp + vault (ingesters Phase 1.a-1.d corridos por primera vez en prod). `queries.yaml` extendido con 6 queries synthesis/comparison (Fase 2) + 7 queries cross-source (Fase 1.f — 4 reminders + 3 gmail). Singles `hit@5 80.00% [69.09, 90.91] · MRR 0.714 [0.609, 0.818]`; Chains `hit@5 83.33% [70.00, 96.67] · MRR 0.706 [0.567, 0.833] · chain_success 54.55% [27.27, 81.82]`. **Todos los metrics overlapean el CI del baseline anterior** — singles +1.4pp noise, chains −3.3pp noise, chain_success −9pp dentro del ±CI. Auto-rollback gate pasa por doble margen (floor 76.19% / 63.64% vs actual 80.00% / 83.33%). 6 de las 7 queries cross-source hitearon — la que falla ("resumen bancario BICA enero 2026") no es un issue de threshold sino retrieval específico del corpus (el thread Gmail está más oculto de lo que el query esperaba). **Decisión Phase 1.f tuning**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` queda todo en el global 0.015 — no hay regresión medible que justifique bajarlo per-source. Si aparecen false-refuse cross-source en logs de producción, re-evaluar.

**Post Calendar API enable (2026-04-21 evening, n=60 singles / 11 chains):** Calendar API del proyecto GCP `701505738025` activada por el user → `rag index --source calendar --reset` ingestó 368 eventos (2y history + 180d future del calendar `fernandoferrari@gmail.com`). Corpus: 8231 chunks (calendar 4.5% · WA 49.5% · vault 45.3% · reminders 0.4% · gmail 0.2%). `queries.yaml` +5 queries calendar (turno psicólogo / workshop AI Engineer / despedida de jardín de astor / reunión con Max ops / turno erica franzen). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.735 [0.639, 0.831] · recall@5 76.94% · n=60`; Chains sin cambio (no se agregaron chains calendar): `hit@5 83.33% · MRR 0.706 · chain_success 54.55%`. **5/5 queries calendar hit con MRR promedio 0.87** — confirmación directa de que el pipeline existente resuelve events.list()-ingested calendar sin tocar scoring/threshold. Smoke test previo sobre query real log: "qué hago el viernes 20hs" pasó de score 0.04 (refuse) → 0.46 (respuesta útil) post-ingesta, validando el hallazgo de producción. **Bug arquitectónico residual detectado**: `classify_intent` devuelve `recent` para "qué tengo esta semana" → itera notas del vault en vez de filtrar `source=calendar` con ventana temporal. Antes era inerte porque calendar estaba vacío; ahora merece ticket como Phase 1.g (intent-source routing) — **addressed later in the same session, see the next entry**. Auto-rollback gate sigue pasando holgado. **Calendar path format drift fix**: `tests/test_eval_bootstrap.py::test_golden_cross_source_paths_have_native_id_format` tenía regex `calendar://(event:)?<id>` basado en design doc §2.7, pero la implementación real del ingester (`scripts/ingest_calendar.py::_event_file_key`) usa `calendar://<calendar_id>/<event_id>` (two-segment, paralelo a WhatsApp). Test updated para seguir implementación como ground truth.

**Post session close (2026-04-21 evening, same n=60 singles / 11 chains):** Final eval tras el `agenda` intent (v1 + v2 window filter), cold-embed non-blocking fix (`_local_embedder_ready` Event), + feedback harvest de 8 negativos adicionales al `rag_feedback` SQL (ratio pos:neg pasó de 55:2 → 55:10). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.779 [0.683, 0.875] · recall@5 76.94%`; Chains `hit@5 80.00% [66.67, 93.33] · MRR 0.733 [0.583, 0.867] · chain_success 54.55% [27.27, 81.82]`. **Singles MRR +0.044 vs el baseline de la mañana** (dentro del CI anterior pero consistent uplift — probablemente absorción de los 8 negatives al golden cache). Chains MRR +0.027 idem. Hit@5 bit-identical en singles, −3.3pp en chains (dentro del CI, noise). Floor gate pasa por doble margen. Calendar queries siguen 5/5 hit (idéntico al baseline anterior — el fix del intent router no las rompió, solo redirigió queries browsing que antes caían en `recent`). Auto-rollback gate holgado.

**Fase D closeout — 3 mejoras validadas (2026-04-21, n=60 singles / 11 chains):** Dos runs de `rag eval` con flags todas-OFF y todas-ON (RAG_ADAPTIVE_ROUTING=1 RAG_ENTITY_LOOKUP=1 RAG_NLI_GROUNDING=1). Resultados **bit-idénticos** en ambas corridas — las 3 mejoras no introducen regresión retrieval medible. **Flags-OFF**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.807 [0.680, 0.923] · chain_success 72.73% [45.45%, 100.00%]`. **Flags-ON**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.790 [0.657, 0.917] · chain_success 72.73% [45.45%, 100.00%]`.

**Post perf-audit close (2026-04-22 evening, n=60 singles / 11 chains, 8311 chunks):** Sesión extensa de audit + fixes (deep_retrieve 30s guard, semantic_cache background write + refusal gating, fast-path num_ctx 2048→4096, WhatsApp corpus reindexado 0→4134 chunks, gliner instalado en tool install, backfill_entities sobre todo el corpus cross-source +356 entidades / +5046 mentions, test_tag isolation, 3462 legacy rows source-backfilled a vault). Corpus post-reindex ahora con whatsapp 49.7% / vault 45.8% / calendar 4.4% / gmail+reminders <1%. **Transformers downgrade 5.6→5.1** forzado por el constraint de gliner. Eval final: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.678 [0.561, 0.794] · recall@5 69.17%`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.740 [0.603, 0.867] · recall@5 77.22% · chain_success 72.73% [45.45%, 100.00%]`. **Chain success +9pp vs baseline de la mañana pre-sesión** (63.64% → 72.73%); singles bit-idéntico. Latencia retrieve(): singles p50 1302ms / p95 1661ms; chains p50 1564ms / **p95 3307ms** (+49% vs pre-sesión — explicado por el corpus 2x post-WA-reindex: más candidatos al BM25 + semantic → más pares al cross-encoder). Floor gate chains pasa por doble margen; singles en banda de noise LLM pre-existing que ranker-vivo ya intentó mover y auto-rollback rechazó. **Transformers 5.1 no introdujo regresión medible**. `semantic_cache_store_failed` se congeló en 314 (0 nuevos desde el fix `background=True` + `_is_refusal()` gating), `gliner_import_failed` se congeló en 49 (0 nuevos desde el `uv tool install --with gliner`). El system_memory_metrics watchdog no disparó escalación durante la sesión (peak wired+active+compressed ~60%). **Nota de variabilidad**: singles hit@5 bajó de 81.67% (baseline previo) a 71.67% en esta sesión — CIs solapan [71.67%, 83.33%] — causa probable: LLM non-determinism (qwen2.5:7b stochastic) + posible drift de paths en queries.yaml. La caída es pre-existente, no causada por los flags (flags-OFF muestra igual caída). **Decisión de flags**: (a) `RAG_ADAPTIVE_ROUTING` → **stays OFF por default** — criterios de floor (lower CI ≥ 76.19%) no se cumplen en esta sesión eval (lower CI 60.00% < 76.19%); aunque la caída es pre-existente y no causada por el flag, las instrucciones son conservadoras. Flipear a ON cuando `rag eval` vuelva a superar el floor de forma estable. (b) `RAG_ENTITY_LOOKUP` → **flipped a ON tras backfill** (same-day evening commit) — `scripts/backfill_entities.py` corrió sobre el corpus (2022 entities / 6520 mentions / 71% chunk coverage), smoke-test directo validó `rag query "todo lo que tengo de Astor"` retornando síntesis cross-source agregada. Ver §Env vars para detalle. (c) `RAG_NLI_GROUNDING` → **stays OFF** — requiere validación de latencia P95 post-50 queries reales.

**Rerank fp16 A/B 2026-04-22 — NO PROMOTED:** probado si degradar el cross-encoder `bge-reranker-v2-m3` de fp32 a fp16 en MPS ahorra latencia sin romper calidad. Baseline fp32 vs candidato fp16 (via `CrossEncoder(model_kwargs={"torch_dtype": torch.float16})` — la alternativa `model.half()` post-load crashea en predict con `AttributeError: 'ne'` en transformers 5.6 / ST 5.4.1, y de hecho en una A/B previa 2026-04-13 colapsaba todos los scores a ~0.001).

| métrica                    | fp32                         | fp16 (model_kwargs)          | delta                 |
|---------------------------|------------------------------|------------------------------|-----------------------|
| singles hit@5 (n=60)      | 71.67% [60.00%, 83.33%]      | 71.67% [60.00%, 83.33%]      | **0 pp**              |
| singles MRR               | 0.669 [0.553, 0.783]         | 0.678 [0.561, 0.794]         | +0.009 (dentro CI)    |
| singles recall@5          | 68.33%                       | 69.17%                       | +0.84 pp              |
| singles wall-time 60 Q    | **63 s**                     | **121 s**                    | **+58 s (~2× slower)**|
| chains                    | n/a (ollama ReadTimeout)     | n/a (ollama ReadTimeout)     | n/a                   |
| retrieve P50/P95          | no reportado (crash)         | no reportado (crash)         | n/a                   |

Ollama helper (qwen2.5:3b para `reformulate_query`) colapsó en la etapa de chains en ambas runs incluso tras el retry — bug infra ortogonal al A/B, documentado como ruido. Suficiente signal en singles: calidad **igual** (hit@5 bit-idéntico, MRR dentro del CI), latencia **peor 2×** (~+970ms/query en wall-clock del stage). Explicación: MPS no tiene kernels fp16 optimizados para la arquitectura xlm-roberta de `bge-reranker-v2-m3`; el overhead de casting dtype supera cualquier win teórico de throughput. Criterio "retrieve P95 baja >200ms" **violado** (fue al revés). Decisión: **NO PROMOTE**, revertido el patch de `RAG_RERANKER_FP16`, código queda en fp32. Nueva NOTE en `get_reranker()` documenta ambos A/B (2026-04-13 y 2026-04-22) para evitar re-intentar.

**Prior floor (2026-04-17, post-title-in-rerank, n=21 singles / 9 chains):** Singles `hit@5 90.48% · MRR 0.821`; Chains `hit@5 80.00% · MRR 0.627 · chain_success 55.56%`. Kept for historical trend, but do not compare new numbers against it without overlapping CIs.

**Even-earlier floor (2026-04-16, post-quick-wins, n=21/9):** Singles `hit@5 90.48% · MRR 0.786`; Chains `hit@5 76.00% · MRR 0.580 · chain_success 55.56%`.

The 2026-04-15 floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains, see `docs/eval-tune-2026-04-15.md`) pre-dates both the expansion and the CI tooling — treat as a qualitative reference only.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

**`seen_titles` as post-rerank penalty** (2026-04-20, `SEEN_TITLE_PENALTY = 0.1` in `retrieve()`). The LLM-instruction path regressed chains (chains hit@5 −16pp, chain_success −33pp — helper treats the list as "avoid these" and drifts off-topic); the post-rerank soft penalty is the shipped replacement. Candidates whose `meta.note` (case-insensitive) matches any `seen_titles` entry get their final score docked by 0.1 — a diversity nudge, not a filter (strong rerank leads still win). In `reformulate_query` the kwarg remains on the signature but is intentionally unused in the prompt (dead per design). Tests in `tests/test_seen_titles_penalty.py`. Empirical lift on queries.yaml chains hit@5 83.33% → 90.00% (both inside CI — re-measure on next tune cycle before claiming stable gain).

## Cross-source corpus (Phase 1, in progress — 2026-04-20 decisions)

The corpus is no longer vault-only. Per `docs/design-cross-source-corpus.md` + §10 user decisions, `retrieve()` is now source-aware and the sqlite-vec collection holds chunks from multiple sources via a `source` metadata discriminator. Collection stays at `obsidian_notes_v11` (no rename / no re-embed) — legacy vault rows without `source` are read as `"vault"` via `normalize_source()`.

**Constants** (`rag.py:~1288`): `VALID_SOURCES` (frozenset of 6), `SOURCE_WEIGHTS` (vault 1.00 → WA 0.75), `SOURCE_RECENCY_HALFLIFE_DAYS` (None for vault/calendar, 30d for WA/messages, 90d for reminders, 180d for gmail), `SOURCE_RETENTION_DAYS` (None for vault/calendar/reminders, 180 for WA/messages, 365 for gmail).

**Helpers**: `normalize_source(v, default="vault")` → safe legacy-row read; `source_weight(src)` → lookup + 0.50 fallback; `source_recency_multiplier(src, created_ts, now)` → exponential decay `2**-(age/halflife)` in [0,1], accepts epoch float or ISO-8601 string (Zulu Z), clamps future-ts at 1.0, None-halflife short-circuits to 1.0.

**Scoring** (inside `retrieve()` post-rerank loop + in `apply_weighted_scores()` for eval parity): after the existing scoring formula produces `final`, multiply by `source_weight(src) * source_recency_multiplier(src, created_ts)`. Vault default → `1.0 * 1.0` = no-op. Old vault data completely untouched.

**Filter** (retrieve/deep_retrieve/multi_retrieve `source` kwarg + `rag query --source S[,S2]`): string or iterable of strings; restricts candidate pool post-rerank. Unknown sources from the CLI are rejected upfront with a helpful error. Legacy vault path: `source=None` or `source="vault"` → identical to pre-Phase-1 behavior.

**Conversational dedup** (`_conv_dedup_window`, applied post-scoring pre top-k slice): collapses WhatsApp/messages chunks from the same `chat_jid` within a ±30min window — keeps only the highest-scored. Non-WA sources pass through unchanged. Intentionally simple O(n²) — pool is capped at `RERANK_POOL_MAX`, constant factor negligible.

### WhatsApp ingester — Phase 1.a (`scripts/ingest_whatsapp.py`, `rag index --source whatsapp`)

Reads from `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db` in read-only immutable mode. Filters empty content, `status@broadcast` pseudo-chat, **`WHATSAPP_BOT_JID` (RagNet group)**, **mensajes con prefix U+200B (output del bot)**, y anything older than 180d. Timestamps (Go RFC3339 with nanoseconds / Z suffix / numeric) parsed defensively. Conversational chunking (§2.6 option A): groups same-sender contiguous messages within 5min windows; splits on speaker change OR >=5min gap OR >800 chars; merges undersized groups (<150 chars) into temporally-nearest neighbor in the same chat. Parent window ±10 messages, 1200 char cap. Embed prefix `[source=whatsapp | chat=X | from=Y] {body}`; display text stays raw. doc_ids are `whatsapp://{chat_jid}/{first_msg_id}::{idx}` — stable across bridge DB compactions. Idempotent upsert (delete prior by `file` key + add). Incremental cursor in `rag_whatsapp_state(chat_jid, last_ts, last_msg_id)`; `--reset` wipes, `--since ISO` overrides uniformly. CLI flags: `--bridge-db`, `--since`, `--reset`, `--max-chats`, `--max-messages`, `--dry-run`, `--json`.

**Anti-feedback-loop guards** (closed 2026-04-28): RagNet (`120363426178035051@g.us`) es la UI del bot — recibe morning briefs, archive pushes, reminder pushes, anticipatory prompts, draft cards via `RAG_DRAFT_VIA_RAGNET`, slash commands del user (`/help`, `/note`, `/cap`), y respuestas del bot a esos comandos. **Nada de eso es contenido conversacional**. Pre-fix el indexer chunkeaba todo eso → el corpus se llenaba de output del propio bot → retrieve devolvía briefs viejos como "evidencia" → el siguiente brief incluía self-references. Los fetchers usados por el brief mismo (`_fetch_whatsapp_unread`, `_fetch_whatsapp_today`, `_fetch_whatsapp_window` en [`rag/integrations/whatsapp.py`](rag/integrations/whatsapp.py)) ya filtraban RagNet a nivel SQL desde Phase 1.a; el indexer era el último path abierto. **Fix 2026-04-28**: `HARDCODED_EXCLUDE_JIDS` agrega `WHATSAPP_BOT_JID` + content-level filter `content.startswith('\u200B')` (defense in depth — cualquier mensaje en cualquier chat con prefix U+200B es output del bot por contrato del listener, no se indexa). Aplica también a [`_read_recent_image_messages`](scripts/ingest_whatsapp.py) (mismo `exclude_jids` default). **Implicación operativa**: si el corpus tenía chunks de RagNet de runs anteriores, siguen ahí — `rag index --source whatsapp --reset` los limpia. Tests: [`tests/test_ingest_whatsapp.py`](tests/test_ingest_whatsapp.py) (3 casos nuevos: RagNet exclusion, U+200B content filter, frozenset sanity).

### Calendar ingester — Phase 1.b (`scripts/ingest_calendar.py`, `rag index --source calendar`)

Google Calendar via OAuth (§10.6 user override — rompe local-first). Creds under `~/.calendar-mcp/{gcp-oauth.keys.json, credentials.json}`. Window `[now − 2y, now + 180d]` on bootstrap, `syncToken` for incremental. `singleEvents=True` (expands RRULE instances). Chunk-per-event, `parent=body`, body cap 800 chars. Cancelled events → delete. State in `rag_calendar_state(calendar_id, sync_token, last_updated, updated_at)`. Hardcoded exclude list filters `addressbook#contacts` + `en.usa#holiday` noise calendars.

### Gmail ingester — Phase 1.c (`scripts/ingest_gmail.py`, `rag index --source gmail`)

Gmail via OAuth (same cred dir convention, `~/.gmail-mcp/`). Thread-level chunking (§2.6 — one chunk per thread, not per message — empirically matches user "cuándo hablamos de X" granularity better than message-level). Quoted replies + signatures stripped via regex before chunking. `parent = subject + first 1200 chars of thread`. Incremental via Gmail's `historyId` cursor in `rag_gmail_state(history_id, updated_at)`. Bootstrap uses `q=newer_than:365d` per §10.2 retention. Deleted threads removed from index on incremental pass.

### Reminders ingester — Phase 1.d (`scripts/ingest_reminders.py`, `rag index --source reminders`)

Apple Reminders via AppleScript (local, same trust boundary as the morning brief's `_fetch_reminders_due`). Pulls every reminder (pending + completed) with id, list, due/completion/creation/modification dates, name, body, priority, flagged state. Chunk-per-reminder, body cap 800 chars. `created_ts` anchor preference: creation → due → modified → completion. Content-hash diffing in `rag_reminders_state(reminder_id, content_hash, last_seen_ts, updated_at)` — on each run, re-fetch the full catalogue, upsert changed/new, delete stale (id no longer present). No cursor / modification-date polling — Reminders.app's `modification date` is unreliable via AppleScript. Field separator in the AS → Python pipe is chr(31) (Unit Separator) to avoid collisions with body content. CLI flags: `--reset`, `--dry-run`, `--only-pending` (default indexes both). Retention None (§10.2); source weight 0.90, recency halflife 90d (§10.3).

### Contacts ingester — Phase 1.e (`scripts/ingest_contacts.py`, `rag index --source contacts`)

Apple Contacts via direct SQLite read on `~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb` (one DB per account: local, iCloud, Google, etc.). Zero pyobjc, zero AppleScript — the .abcddb is plain SQLite + standard Apple Core Data schema (`ZABCDRECORD` + `ZABCDPHONENUMBER` + `ZABCDEMAILADDRESS` + `ZABCDNOTE`). Chunk-per-contact (atomic, <800 chars), body cap 800. Merges first+last+org+phones+emails+note+birthday. Timestamps converted from Cocoa epoch (2001-01-01) to Unix via `_cocoa_to_unix()`. Phone `ZLABEL`s like `_$!<Mobile>!$_` → clean `mobile`. Empty records (no name + no phone + no email + no org) filtered out as Core Data internals / groups.

**Dual role**: besides corpus ingestion, the module exposes `resolve_phone(raw_number) -> Contact | None` used by `scripts/ingest_calls.py` (and future iMessage/WhatsApp enrichment) to map phone digits back to a human name. Two-pass phone index: Pass 1 dedupes `(digits → UID)` canonically (iCloud linked cards across sources share the same phone but have DIFFERENT `ZUNIQUEID`s → pick the contact with the longer `display_name`, tiebreak by lexicographic UID for determinism), Pass 2 fans out to suffix keys (full, last-10, last-8, last-7) dropping genuine cross-UID collisions (an AR mobile and a US landline sharing the last-7 digits → ambiguous, return None rather than mis-attribute). Measured on the dev host: naive-UID-conflict detection dropped 90 of 580 keys and resolved only 3% of call numbers; the two-pass fix raised it to 39% (which is the plausible upper bound since ~60% of incoming calls are spam/telemarketers not in contacts).

State: `rag_contacts_state(contact_uid, content_hash, last_seen_ts, updated_at)`. Content hash excludes `modified_ts` (iCloud bumps it on idle sync). Stale deletion on each run (UIDs in state DB but missing from live AddressBook → delete from corpus). `invalidate_phone_index()` is called at the end of every successful `run()` so downstream in-process callers (same process indexing calls after contacts) see fresh data. doc_ids are `contacts://<ZUNIQUEID>::0`. Retention None (§10.2 — contacts don't age), source weight 0.95, recency halflife None (no decay). CLI flags: `--reset`, `--dry-run`, `--root` (override AddressBook root, useful for tests), `--json`. Tests: [`tests/test_ingest_contacts.py`](tests/test_ingest_contacts.py) (40 cases).

### Calls ingester — Phase 1.f (`scripts/ingest_calls.py`, `rag index --source calls`)

macOS/iOS CallHistory via direct SQLite read on `~/Library/Application Support/CallHistoryDB/CallHistory.storedata` (Core Data). Pulls every call (phone, FaceTime audio/video, incoming / outgoing / missed / unanswered) from `ZCALLRECORD` within the retention window. Chunk-per-call (atomic, <800 chars). Timestamps converted from Cocoa epoch (`ZDATE + 978307200`). Direction matrix from `(ZORIGINATED, ZANSWERED)`: outgoing+answered → "saliente · atendida", outgoing+unanswered → "saliente · sin respuesta", incoming+answered → "entrante · atendida", incoming+unanswered → "perdida" (missed — `is_missed=True`). Service provider mapping: `com.apple.Telephony` → "Teléfono", `com.apple.FaceTime` → "FaceTime", unknown passed through raw.

**Enrichment via Contacts**: the chunk body headline uses `ingest_contacts.resolve_phone(ZADDRESS)` to show human names instead of raw digits. Fallback chain: resolved Contact → `ZNAME` cached by Apple at call time → raw address → "(desconocido)". Headline phrasing is BM25-friendly: "Llamada perdida de Juli" / "Llamada saliente a Astor" so queries like "llamadas perdidas de Juli" hit without relying on embeddings alone.

State: `rag_calls_state(call_uid, content_hash, last_seen_ts, updated_at)`. Calls are effectively immutable once logged, so the hash mostly guards against Apple's rare retroactive edits. Stale deletion when a `ZUNIQUE_ID` rolls off Apple's retention window (macOS keeps a few months). doc_ids are `calls://<ZUNIQUE_ID>::0`. Retention 180d (matches WhatsApp/messages — equally ephemeral), source weight 0.80 (between gmail 0.85 and whatsapp 0.75 — log entries are factual but semantically thin), recency halflife 30d (who-called-yesterday is critical, who-called-six-months-ago is trivia). CLI flags: `--reset`, `--dry-run`, `--since ISO` (hard floor, intersects with retention), `--retention-days N` (override default 180; 0 disables), `--db` (override path), `--json`. Tests: [`tests/test_ingest_calls.py`](tests/test_ingest_calls.py) (34 cases).

### Safari ingester — Phase 2 (`scripts/ingest_safari.py`, `rag index --source safari`)

Dos fuentes, un solo ingester: `~/Library/Safari/History.db` (SQLite) + `~/Library/Safari/Bookmarks.plist` (binary plist que incluye Reading List como subtree `com.apple.ReadingList`). Complementa el inline Chrome ETL existente (`rag.py:_sync_chrome_history` escribe `.md` al vault) — Safari va por la arquitectura source-prefixed (chunks en la DB vectorial directo, sin polución al vault).

**Historia**: SQL JOIN `history_items ← history_visits`. Filtra `load_successful=1` (drop 404s + red fails) + `redirect_source IS NULL` (drop 301/302 intermediate hops), agrega por URL (no por visita — 7407 visitas aplastadas a ~3800 URLs únicas), agarra el título más reciente non-empty como display. Retention 180d sobre `last_visit`. Cap `max_urls=5000` por run (configurable) para evitar runs multi-minute en historial largo. doc_ids: `safari://history/<history_item_id>::0`.

**Bookmarks**: recursive walk del plist tree, skippeando folders internos (`BookmarksBar`, `BookmarksMenu`) y nodos Proxy (Historial placeholder, Reading List shortcut). URIDictionary.title gana sobre Title directo para leaves. Reading List entries cargan `ReadingList.PreviewText` que se concatena al título con ` — ` (rico para BM25 en "artículo que guardé sobre X"). doc_ids: `safari://bm/<UUID>::0` para bookmarks, `safari://rl/<UUID>::0` para Reading List — prefix distinto porque un UUID podría moverse entre bookmarks ↔ RL y necesitamos delete surgical por el otro prefix al migrar.

**State**: dos tablas separadas — `rag_safari_history_state(history_item_id INTEGER PK, content_hash, last_seen_ts, updated_at)` + `rag_safari_bookmark_state(bookmark_uuid TEXT PK, content_hash, ...)`. Diff + stale delete por cada una. Content hash del history excluye `first_visit_ts` (estable por definición) pero incluye `last_visit_ts` + `visit_count` para detectar nuevas visitas. Bookmark hash incluye `is_reading_list` flag para detectar el movimiento RL↔bookmarks.

Source weight 0.80 (mismo banda que calls — signal factual rico pero no curado por el usuario como Contacts). Halflife 90d (browsing context ages mid-term; no es conversacional como WA pero tampoco permanente como Calendar). Retention 180d. CLI flags: `--reset`, `--dry-run`, `--since ISO`, `--retention-days N`, `--max-urls N` (default 5000), `--skip-bookmarks`, `--history-db`, `--bookmarks-plist`, `--json`. Tests: [`tests/test_ingest_safari.py`](tests/test_ingest_safari.py) (37 cases).

**Note on SQLite contention**: cuando `rag serve` / `web/server.py` está corriendo, el primer `rag index --source safari` puede pegar `database is locked` en el bookmarks bulk-add (1000+ rows en una transacción + GLiNER entity extraction concurrente). Reintentá — el state de history ya se commiteó en la primera tanda, así que el retry solo procesa bookmarks. Long-term fix: serializar con `vault_write_lock` el branch de safari (TODO, no-blocker por ahora).

### Remaining (Phase 1.g, 1.h + 2)

- **Phase 1.g — apagar workaround** (gated on 1.c stable in prod ≥1 week): deprecar `/note` + `/ob` del whatsapp-listener ahora que el corpus captura WA por barrido. ~100 LOC, mostly external repo. (Renamed from 1.e; letter freed for Contacts ingester.)
- **Phase 2 — OCR pipeline para adjuntos** (deferred 2026-04-21, no shipped): el design doc §8 flagea "no indexa adjuntos binarios (imágenes WA, PDFs en Gmail). Eso es fase 2". Evaluado en la tanda 2026-04-21 y **skipped** porque el sistema actual tiene 16 imágenes (todas en `04-Archive/99-obsidian-system/99-Attachments/`, screenshots archived) + **0 PDFs** en el vault, y el corpus cross-source (donde estarían los adjuntos reales de Gmail/WA) tiene 0 chunks — los ingesters Phase 1.a-1.d nunca corrieron en prod. Sin data activa, implementar OCR sería scaffolding + agregar dep (`pyobjc-framework-Vision` ~20 MB, o tesseract via brew) sin beneficio medible actual. **Trigger de activación**: cuando los ingesters hayan corrido ≥1 semana y haya ≥20 adjuntos referenciados en el corpus cross-source, implementar usando Apple Vision (`VNRecognizeTextRequest`, local) para imágenes + `pdftotext` (poppler) para PDFs con fallback Vision para scans. Chunk OCR como prose con metadata `attachment_of: <parent doc_id>` + `media_type: "ocr"`.
- **Phase 1.h — re-calibración eval** *(infra shipped 2026-04-21, tuning pending real data)*:  (Renamed from 1.f; letter freed for Calls ingester.)
  - **Infra shipped**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` dict en `rag.py` (scaffolding — todos los valores = baseline 0.015 hoy) + helper `confidence_threshold_for_source(source)` con fallback al global. Invocado en `query()` y `rag serve` sobre `source` del top-result meta. Tests: [`tests/test_confidence_threshold_per_source.py`](tests/test_confidence_threshold_per_source.py) (9 casos). El test `tests/test_eval_bootstrap.py::test_queries_yaml_all_paths_exist_or_placeholder` ahora acepta paths con prefijos `gmail://` / `whatsapp://` / `calendar://` / `reminders://` / `messages://` como placeholders válidos; sanity-test aparte `test_queries_yaml_cross_source_prefixes_cover_all_valid_sources` detecta drift contra `VALID_SOURCES`. Template de queries cross-source está comentado en [`queries.yaml`](queries.yaml) listo para un-commentar cuando los ingesters populen el corpus.
  - **Tuning pending**: re-correr `rag eval` + bajar per-source thresholds empíricamente (expected: WA 0.008-0.010, Calendar 0.012, Gmail 0.010-0.012, Reminders 0.012) una vez que los ingesters hayan corrido ≥1 semana + haya feedback data. Validar los `SOURCE_WEIGHTS` hardcoded (vault 1.00 / calendar 0.95 / reminders 0.90 / gmail 0.85 / WA 0.75 / messages 0.75) contra queries reales. Deferred per §10.8.

## On-disk state (`~/.local/share/obsidian-rag/`)

### Telemetry — SQL tables (post-T10 2026-04-19, post-split 2026-04-21)

Telemetry + learning state lives en **dos** databases bajo `~/.local/share/obsidian-rag/ragvec/`:

- **`ragvec.db`** (~104M) — sqlite-vec `meta_*`/`vec_*` tables del corpus + **10 state tables**: `rag_whatsapp_state`, `rag_calendar_state`, `rag_gmail_state`, `rag_reminders_state`, `rag_contacts_state`, `rag_calls_state`, `rag_safari_history_state`, `rag_safari_bookmark_state`, `rag_wa_media_state`, `rag_schema_version`. Sólo cursors + dedup keys de ingesters.
- **`telemetry.db`** (~36M) — **45+ tablas** operativas: `rag_queries`, `rag_behavior`, `rag_feedback`, `rag_feedback_golden*`, `rag_tune`, `rag_contradictions`, `rag_ambient*`, `rag_brief_*`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`, `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics`, `rag_conversations_index`, `rag_response_cache`, `rag_entities`, `rag_entity_mentions`, `rag_ocr_cache`, `rag_vlm_captions`, `rag_audio_transcripts`, `rag_learned_paraphrases`, `rag_cita_detections`, `rag_score_calibration`, `rag_schema_version`.

**Split rationale** (`scripts/migrate_ragvec_split.py`, 2026-04-21): cada DB comparte un único WAL entre todos sus writers. Mezclar chunks + telemetría en un WAL único causaba bursts de lock contention — el indexer escribiendo 100 chunks interfería con el write sync de cada query log. Separar en 2 DBs permite que cada WAL tenga su propio pattern de writes (indexer bulk vs telemetry append) sin bloquearse entre sí. `_ragvec_state_conn()` resuelve a `telemetry.db` (post-split); los ingesters siguen abriendo directamente `ragvec.db` para su state cursor (ver `rag.DB_PATH / "ragvec.db"` en `scripts/ingest_*.py`).

**Reset total**: `rm ragvec/ragvec.db ragvec/telemetry.db` + `rag index --reset`. Para reset solo telemetría preservando el corpus: `rm ragvec/telemetry.db` (se recrea vacía en el próximo open).

SQL es el único storage path — T10 (2026-04-19) stripped JSONL writers + readers. `RAG_STATE_SQL=1` sigue en todos los plists de launchd por trail / faster rollback, pero es un no-op toggle hoy (ni writers ni readers lo consultan).

Log-style tables (`id INTEGER PK AUTOINCREMENT`, `ts TEXT` ISO-8601, indexed):
- `rag_queries` — query log (q, variants_json, paths_json, scores_json, top_score, t_retrieve, t_gen, cmd, session, mode, citation_repaired, critique_fired/changed, extra_json). Retention 90d via `rag maintenance`.
- `rag_behavior` — ranker-vivo events (source: cli/whatsapp/web/brief × event: open/kept/deleted/positive_implicit/negative_implicit/save). Retention 90d.
- `rag_feedback` — explicit +1/-1 + optional corrective_path (UNIQUE(turn_id,rating,ts)). Keep all.
- `rag_tune` — offline + online tune history (cmd, baseline/best_json, delta, eval_hit5_*, rolled_back). Keep all.
- `rag_contradictions` — radar Phase 2 (UNIQUE(ts,subject_path)). Keep all.
- `rag_ambient` / `rag_ambient_state` — ambient agent log (retention 60d) + dedup state (upsert by path).
- `rag_brief_written` — morning/today brief citation manifest (retention 60d).
- `rag_brief_state` — kept/deleted dedup (upsert by pair_key = hash(brief_type, kind, path)).
- `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log` — 60d retention.
- `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics` — per-minute samplers, 30d retention.
- `rag_draft_decisions` (2026-04-29) — decisiones del user sobre drafts del bot WA (`approved_si | approved_editar | rejected | expired`) + bot_draft + sent_text + original_msgs. **Keep all forever** — el dataset histórico es gold humano para fine-tunear el modelo de drafts (ver "Bot WA draft loop" más arriba). Populated via `POST /api/draft/decision` desde el listener TS.
- `rag_brief_feedback` (2026-04-29) — reactions del user sobre los briefs (morning / evening / digest) pusheados por el daemon (`positive | negative | mute`) + `dedup_key=vault_relpath`. **Keep all forever** — input de tuning para horario / cadencia / contenido de los briefs (ver "Brief feedback loop" más arriba). Populated via `POST /api/brief/feedback` desde el listener TS.

State-style tables:
- `rag_conversations_index` — episodic session_id → relative_path (web/conversation_writer.py upsert; replaces the old conversations_index.json + fcntl dance).
- `rag_feedback_golden` (pk=path,rating, `embedding BLOB` float32 little-endian, `source_ts`) + `rag_feedback_golden_meta` (k/v) — cache rebuilt when `rag_feedback.max(ts) > meta.last_built_source_ts`. `record_feedback` clears both tables synchronously so the next `load_feedback_golden()` call always rebuilds (sidesteps a same-second MAX(ts) collision that could leave a stale cache).
- `rag_response_cache` — semantic response cache (GC#1, 2026-04-22, durability + wiring fixes 2026-04-23). Key shape: `(id, ts, question, q_embedding BLOB, dim, corpus_hash, intent, ttl_seconds, response, paths_json, scores_json, top_score, hit_count, last_hit_ts, extra_json)`. Lookup: cosine ≥ `_SEMANTIC_CACHE_COSINE` (default 0.93 vía `RAG_CACHE_COSINE`) contra las últimas `_SEMANTIC_CACHE_MAX_ROWS` entradas del mismo `corpus_hash` dentro de `ttl_seconds`; hits bump `hit_count` + `last_hit_ts`. **Gates de store** (aplican igual en sync y background): (a) cache disabled, (b) corpus_hash vacío, (c) response vacío, (d) `top_score < 0.015` (refuse por gate de confianza), (e) `_is_refusal(response)` matchea (refuse conceptual del LLM — patrón añadido 2026-04-22 tras observar cache poisoning: una query con top_score 1.18 cacheó "No tengo esa información" y envenenó queries similares permanentemente). Helpers: `semantic_cache_lookup()` / `semantic_cache_store()` / `semantic_cache_clear(corpus_hash?)` / `semantic_cache_stats()`. **2026-04-23 audit + fix** (el cache tenía 0 hits reales con 2,335 queries y 14 queries repeated ≥10×): tres changes concurrentes.
    1. **`corpus_hash` simplificado a count-only** (era count + top-10 mtimes). Cada edit a una nota individual no invalida más el cache global. Solo add/remove de notas (chunk-count delta) dispara invalidación coarse. **2026-04-24 follow-up** (commit `09f00bd`): el count exacto seguía cambiando con cada chunk que un ingester agregaba/removía — audit del web.log mostró 30 SEMANTIC PUTs con 24 corpus_hashes DISTINTOS → 0 hits ever porque cada query mintea hash nuevo. Ahora bucketea por `_CORPUS_HASH_BUCKET = 100`: `_compute_corpus_hash(col)` = `sha256(f"count_bucket:{col.count() // 100}")[:16]`. Hash sólo cambia cuando el count cruza un múltiplo de 100 — sobrevive la rotación normal de WhatsApp/Calendar/Gmail incrementales (typically <50 chunks/run). Tradeoff: bulk +/-100 chunks netos invalidan el cache (correcto: corpus cambió suficiente como para esperar retrieval diferente). Per-entry staleness check (paso 2) ya cubre edits a paths citadas dentro del bucket.
    2. **Per-entry freshness check** (`_cached_entry_is_stale(paths, cached_ts)`) en `semantic_cache_lookup`: si cualquiera de las paths cacheadas tiene `mtime > cached_ts`, la fila se skippea con `probe.reason="stale_source"` sin tumbar el resto del cache. File missing / vault-path unresolvable se tratan como fresh (no blow-up por infra issues; la invalidación global ya atrapa deletes).
    3. **Durabilidad del store en `query()` CLI**: `background=True → background=False`. El store del `rag query` one-shot se perdía en el atexit drain (2s cap) porque el worker daemon se estaba todavía reintentando bajo contention (telemetry.db recibe 2k+ writes/hour entre queries/behavior/cpu/memory metrics). Synchronous store bloquea el return por ≤1.3s pero el user ya vio la respuesta — no hay regresión de latencia percibida, y el cache queda realmente persistente. Background mode sigue disponible para long-running processes (web server, serve.chat) que no tienen el problema del atexit.
    4. **Wiring extendido a `run_chat_turn()`** (helper compartido — cubre `chat()` CLI + futuros callers del unified pipeline). Eligibility: single-vault, no history, no source/folder/tag/date_range filter, no critique/counter/precise. Nuevo field en `ChatTurnRequest`: `cache_lookup: bool = True`, `cache_store: bool = True`, `cache_background: bool = True` (opt-out per-caller). Hit path sintetiza un `RetrieveResult` mínimo desde los paths cacheados para que `to_log_event` loguee normalmente.
    4.b **Wiring en `/api/chat` (web server, 2026-04-23)** — cubre el caller más grande (856/2,335 queries = 37% del tráfico). El web ya tenía un LRU exact-string (`_CHAT_CACHE`, 100 entries × TTL 5min, in-memory) en `web/server.py`. El semantic cache SQL se agrega como *segundo layer* POST-LRU-miss: exact → semantic → pipeline, con el hit del semantic hidratando el LRU así la próxima query exact-string pega instantánea. Gates: no history, single-vault, no propose_intent. Sintetiza `sources_items` desde paths+scores del hit (minimal meta: file/note/folder/score/bar). `done` event trae `cache_layer="semantic"` para que el UI lo distinga del LRU (UI-key ya existente — mismo stage=`cached`). Store post-pipeline con `background=True` (el web server es long-running, no sufre el atexit drop del CLI).
    4.c **Wiring en `rag serve` /query (WhatsApp listener + bots, 2026-04-23)** — cierra el último caller. El serve ya tenía su LRU propio (`_serve_cache`, 64 entries × TTL 5min) keyed en `(sid|folder|tag|loose|question)`. Mismo patrón: lookup semantic post LRU miss (después del weather + tasks short-circuits para no cachear time-sensitive), store semantic pegado al `_cache_put(cache_key, payload)` dentro del mismo `if not force and not qfolder and not qtag:` guard. Hit path sintetiza sources en el shape del listener (`{note, path, score}`, no `{file, note, folder, score, bar}` como el web). Log event: `cmd="serve.cached_semantic"` — bucket propio para distinguirlo de `serve`/`serve.tasks`/`serve.chat` en analytics. Tests: [`tests/test_serve_semantic_cache.py`](tests/test_serve_semantic_cache.py) (12 casos — source-grep contract por consistencia con test_serve_fast_path_consumption.py + test_serve_short_circuits.py existentes).
    5. **`cache_probe` instrumentation** en `rag_queries.extra_json`: `{result: hit|miss|skipped|disabled|error, reason: match|below_threshold|ttl_expired|stale_source|corpus_mismatch|flags_skip|cache_disabled|no_corpus_hash|db_error, top_cosine: float|null, candidates: int, skipped_stale: int, skipped_ttl: int}`. `semantic_cache_lookup(..., return_probe=True)` devuelve tupla `(hit_or_None, probe_dict)` — backward-compat preservada (default `return_probe=False` devuelve solo hit/None).
    6. **`rag cache stats --days N`** extendido: hit rate real del período leyendo `extra_json.cache_probe` + distribución de miss reasons + ahorro estimado (avg `t_gen_ms` de misses × hits) + top 10 queries cacheadas por `hit_count`. Nuevo helper `_cache_telemetry_stats(days=7)` cross-referencea `rag_queries` con `rag_response_cache`.
    Tests: [`tests/test_semantic_cache.py`](tests/test_semantic_cache.py) (22 casos base), [`tests/test_semantic_cache_probe.py`](tests/test_semantic_cache_probe.py) (8 casos — probe shape por cada `reason`), [`tests/test_semantic_cache_freshness.py`](tests/test_semantic_cache_freshness.py) (8 casos — `_cached_entry_is_stale` unit + integration lookup skip), [`tests/test_semantic_cache_run_chat_turn.py`](tests/test_semantic_cache_run_chat_turn.py) (9 casos — hit short-circuits LLM, miss corre pipeline, skip por history/source/critique/multi-vault, cache_lookup=False opt-out, `to_log_event` emite cache fields), [`tests/test_cache_stats_telemetry.py`](tests/test_cache_stats_telemetry.py) (6 casos — eligible/hits/reasons/top_queries/CLI smoke), [`tests/test_web_chat_semantic_cache.py`](tests/test_web_chat_semantic_cache.py) (9 casos — SSE replay shape, ollama.chat no llamado en hit, sources sintetizados, LRU hit beats semantic, store post-pipeline, gates history/propose/multi-vault, lookup-exception fallback).

**Nota sobre cobertura de policy**: Las ~20 tablas restantes (rag_status_samples, rag_home_compute_metrics, rag_synthetic_negatives, rag_synthetic_queries, rag_audio_corrections, rag_anticipate_candidates, rag_behavior_priors_wa, rag_error_queue, rag_learned_paraphrases, rag_llm_captions, rag_negotiation_*, rag_promises, rag_reminder_wa_pushed, rag_routing_decisions, rag_routing_rules, rag_score_calibration, rag_style_fingerprints, rag_whatsapp_scheduled, rag_cita_detections, etc.) tienen policies implícitas en sus ingesters/writers pero no documentadas aquí — consultar el código fuente en `rag.py` (`_TELEMETRY_DDL` y callers de `_sql_append_event`) para determinar retention/schema actuales.

Primitives in `rag.py` (`# ── SQL state store (T1: foundation) ──` section):
- `_ensure_telemetry_tables(conn)` — idempotent DDL, **ensure-once por (proceso, db_path)** desde commit `09f00bd` (2026-04-24). Set keyed `_TELEMETRY_DDL_ENSURED_PATHS` skip-ea las ~32 CREATE TABLE IF NOT EXISTS + ALTER tras la primera invocación contra un path. Cuts ~17K DDL stmts/hr × schema-lock contention (medido: avg conn-open 1.5ms first → 0.64ms next, ~5-8x speedup). **Si agregás una entry nueva a `_TELEMETRY_DDL` y querés que aparezca en procesos already-running, hay que reiniciarlos** (launchctl bootout/bootstrap los daemons `com.fer.obsidian-rag-*`); no hay hot-reload. Tests con tmp DB siguen funcionando porque el set es path-keyed, no proceso-global.
- `_ragvec_state_conn()` — short-lived WAL conn with `synchronous=NORMAL` + `busy_timeout=10000`
- `_sql_append_event(conn, table, row)`, `_sql_upsert(conn, table, row, pk_cols)`, `_sql_query_window(conn, table, since_ts, ...)`, `_sql_max_ts(conn, table)`

Writer contract (post-T10): single-row BEGIN/COMMIT into SQL. On exception, log the error to `sql_state_errors.jsonl` and **silently drop the event** — no JSONL fallback. Callers never see a raised exception. Reader contract: SQL-only. Readers return empty snapshots (behavior priors, feedback golden, behavior-augmented cases, contradictions) or False/None (brief_state, ambient_state lookups) on SQL error; retrieval pipeline stays functional without priors until the DB is readable again.

#### Invariantes del telemetry stack (audit 2026-04-24 + extensión 2026-04-25)

Cuatro reglas que el código tiene que respetar — violar cualquiera deja bugs latentes. Las tres primeras salieron del audit del 2026-04-24 tras 6 días de degradación silenciosa; la cuarta del audit 2026-04-25 tras encontrar tests escribiendo a la prod telemetry.db en 3 clases distintas.

1. **Todo silent-error sink llama `_bump_silent_log_counter()`**. Cualquier función nueva tipo `_log_X_error(...)` que escribe a un `.jsonl` y devuelve sin raisear DEBE invocar el helper en `rag.py` post-write. Sin esto, el alerting a stderr (threshold `RAG_SILENT_LOG_ALERT_THRESHOLD=20/h`) queda parcial — es exactamente cómo 1756 errores SQL en 6 días no dispararon un solo alert. Pre-fix `_silent_log` lo bumpeba pero `_log_sql_state_error` no. Tests: `tests/test_silent_log_alerting.py`.

2. **Async writer = paquete completo de 4 cambios**. Cuando un writer pasa a usar `_enqueue_background_sql`: (a) helper de gate per-writer (`_log_X_event_background_default()`), (b) caller con branch sync/async, (c) autouse fixture en conftest que setea `RAG_LOG_X_ASYNC=0`, (d) doc del env var en este CLAUDE.md. Tocar solo (a)+(b) deja tests rotos en producción y la próxima persona descubre el override por accidente. Tests: `tests/test_sql_async_writers.py`.

3. **Readers SQL: retry + stale-cache fallback, nunca empty default que sobrescriba memo**. `_load_behavior_priors` y `load_feedback_golden` son los modelos. En error path, devolver el cache previo SIN tocar `_X_memo`. El bug clásico que esto previene: el `default=("error", {empty}, None)` del retry era asignado al memo, envenenando el cache hasta que `source_ts` cambiara. Tests: `tests/test_sql_reader_retry.py`.

4. **Tests con TestClient o writers SQL aíslan `DB_PATH` per-file**. Conftest autouse `_isolate_vault_path` cubre `VAULT_PATH` global pero NO hay equivalente para `DB_PATH` — intentos conftest-wide reverteados (sesión 2026-04-25, dos veces) porque disparan warning falso de `_stabilize_rag_state` cuando un test sub-fixture redirige a un sub-tmp. Cualquier test que use `fastapi.testclient.TestClient(app)`, llame `log_query_event` / `log_behavior_event` / `semantic_cache_*` / `record_feedback` directamente, o ejercite endpoints `/api/chat` / `/api/behavior`, DEBE redirigir `rag.DB_PATH` con autouse fixture **snap+restore manual** (no `monkeypatch.setattr`):

   ```python
   @pytest.fixture(autouse=True)
   def _isolate_db_path(tmp_path):
       import rag as _rag
       snap = _rag.DB_PATH
       _rag.DB_PATH = tmp_path / "ragvec"
       try: yield
       finally: _rag.DB_PATH = snap
   ```

   Razón del manual snap+restore: `monkeypatch.setattr` revierte en su propio teardown que corre DESPUÉS del teardown de `_stabilize_rag_state` → la stabilizer ve el tmp todavía aplicado y warning. Mismo patrón que `tests/test_rag_log_sql_read.py::sql_env`. Tests con isolation aplicada (al 2026-04-25): `test_degenerate_query`, `test_semantic_cache*` (5 archivos), `test_rag_log_sql_read`, `test_post_t10_sql_readers`, `test_followup`, `test_read`. **Pendiente** (gap conocido): `test_web_{cors,pwa,chat_low_conf_bypass,sessions_sidebar,static_cache,chat_tools,propose_endpoints,chat_mode}`, `test_propose_mail_send`, `test_drive_search_tool`. Pollution medida 2026-04-25: 161 entries `event=test_tag` en `sql_state_errors.jsonl`, 5 rows `question='test'` en `rag_response_cache`, 57 rows `cmd='web.chat.degenerate'` con `q='?¡@#'` en `rag_queries`. Memoria: [feedback_test_db_path_isolation.md](.claude/projects/-Users-fer-repositories-obsidian-rag/memory/feedback_test_db_path_isolation.md) si existe el symlink en tu workspace.

Diagnóstico data-first: correr `python scripts/audit_telemetry_health.py --days 7` antes de cualquier "auditá el sistema" — agrega los 5 queries que reprodujeron el audit 2026-04-24 en 1 segundo (errores SQL/silent, latency outliers, cache probe distribution, DB sizes). Primer comando del workflow.

rag implicit-feedback [--days 14 --json]  # recolecta feedback implícito de interacciones
rag routing-rules [--reset --debug --json]  # descriptor de rutas + patterns detectados
rag whisper-vocab [--refresh --show --source X --limit N]  # manejo de vocabulario de transcripción WhatsApp
rag vault-cleanup [--dry-run --apply --force]  # limpia carpetas transitorias del vault
rag ingest-drive [--reset --dry-run --json]  # Google Drive ingester — busca DAO + documentos compartidos


**Drift fixes (2026-04-21 evening)** — four CLI readers still tail-read JSONL files that post-T10 either no longer receive the expected events or got repurposed for another log stream. All migrated to SQL:

| Reader | Pre-fix behaviour | Fix |
|---|---|---|
| `rag log` | Rendered all columns empty — `queries.jsonl` now receives `conversation_turn_written` observability events with a different schema | `_read_queries_for_log(n, *, low_confidence)` + `_read_feedback_map_for_log()` in `rag.py` — SELECT from `rag_queries` + `rag_feedback`, hoist `turn_id` from `extra_json`, filter admin rows with empty `q`. Renderer null-safety on `t_retrieve` / `t_gen` / `ts` / `mode` when SQL rows return `None` (metachat / create-intent turns). Tests: `tests/test_rag_log_sql_read.py` (13 cases). |
| `rag emergent` + `rag dashboard` | `_scan_queries_log(days)` read `LOG_PATH` → empty events list → "sin queries en ventana", dashboard showed `n=0` | Same `_scan_queries_log(days)` signature, SELECT from `rag_queries WHERE ts >= ?` (chronological ASC), hoists **all** `extra_json` keys to top-level (`q_reformulated`, `answered`, `gated_low_confidence`, `turn_id`) so callers don't re-parse JSON. Admin rows (`q=""`) excluded. Tests: `tests/test_post_t10_sql_readers.py` (5 cases for this reader). |
| `feedback_counts()` / `rag insights` | `FEEDBACK_PATH.is_file()` → False (only `.bak` remains post-cutover) → returned `(0, 0)` silently; `rag insights` showed vacío | Single SQL `SELECT SUM(CASE WHEN rating > 0 ...)` aggregate. Tests: 3 cases. |
| `_feedback_augmented_cases()` / `rag tune` | `FEEDBACK_PATH` empty → returned `[]` → **rag tune silently lost every corrective-path signal** the user ever gave. This was the highest-impact drift — tune couldn't learn from "this path was actually correct" feedback. | SQL with `json_extract(extra_json, '$.corrective_path')` since `corrective_path` isn't a first-class column. Filter `scope != 'session'` + `len(q) >= min_len` + dedup by normalised query. Tests: 5 cases. |
| `rag patterns` | Same shape as feedback_counts — read empty `feedback.jsonl`, always "sin feedback log todavía" | Inline SQL query with `json_extract(extra_json, '$.reason')` for the `reason` field (also inside `extra_json`). |

Also: deleted dead code `_iter_behavior_jsonl()` (defined post-T10 but never called).

Pattern for new SQL readers: use `_ragvec_state_conn()` context manager; wrap the SELECT in `try/except` + `_silent_log` + return empty value matching the old JSONL-reader shape. Never raise from a reader — the calling CLI command keeps working with degraded signal.

Migration one-shot: `scripts/migrate_state_to_sqlite.py --source-dir ~/.local/share/obsidian-rag [--dry-run] [--round-trip-check] [--reverse] [--summary]`. Refuses to run while `com.fer.obsidian-rag-*` services are up (preflight `pgrep`; `--force` to override). Renames each source → `<name>.bak.<unix_ts>` on successful commit. Cutover of 2026-04-19 imported 7,946 records across 19 sources; 43 malformed pre-existing records dropped (missing NOT NULL fields).

Rollback procedure (post-T10): **the escape hatch now requires a code revert, not just a CLI invocation.** `rag maintenance --rollback-state-migration [--force]` still restores the newest `.bak.<ts>` per source and drops the 20 `rag_*` tables + VACUUM — but the in-code readers/writers only know the SQL path after T10. To fully revert:

1. `git revert <T10-commit-sha>` (or `git reset --hard <pre-T10-sha>` if the T10 commits are the tip). This brings back the JSONL fallback code.
2. Restart launchd services so the reverted `rag.py` is loaded in-process.
3. Run `rag maintenance --rollback-state-migration` — this restores the JSONL .bak files that the reverted code now reads.

The `.bak.<ts>` files under `~/.local/share/obsidian-rag/` are still there (kept for the 30-day window) so data-loss is bounded, but without the code revert the restored files are ignored. `rag maintenance` continues to prune `.bak.*` older than 30d.

### Other state (unchanged; still on disk)

- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults.
- `ranker.{unix_ts}.json` — 3 most recent backups, written on every `rag tune --apply`. Consumed by `rag tune --rollback` + auto-rollback CI gate.
- `sessions/*.json` + `last_session` — multi-turn state (TTL 30d, cap 50 turns).
- `ambient.json` — ambient agent config (jid, enabled, allowed_folders).
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` for archiver).
- `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json` — app state + caches.
- `online-tune.{log,error.log}`, `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink for SQL-path write/read failures. Post-T10 this is the only visible signal when SQL errors happen, since the JSONL fallback is gone and the event is dropped after logging here.

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden; DELETE FROM rag_feedback_golden_meta;` inside **`telemetry.db`** (post-split 2026-04-21 esas tablas se movieron de `ragvec.db` → `telemetry.db`). Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

**Persistent memories del MCP [`mem-vault`](https://github.com/jagoff/mem-vault)** viven en `04-Archive/99-obsidian-system/99-AI/memory/` (folder real, no symlink — el comentario antiguo sobre Claude Code era obsoleto). Configurado via env vars del web server plist:
- `MEM_VAULT_PATH=Notes/`
- `MEM_VAULT_MEMORY_SUBDIR=04-Archive/99-obsidian-system/99-AI/memory`

A diferencia del resto de `99-obsidian-system/`, este folder **NO está excluido por `is_excluded()`** (junto con `99-Mentions/`) — `rag index` lo scanea y los `.md` de memorias entran al index como notas más del vault `home`. Eso permite que `rag query "..."` recupere bug patterns, decisiones y convenciones acumuladas entre sesiones (66 memorias / 398 chunks al 2026-04-29). El MCP `mem-vault` sigue teniendo su propio Qdrant local con las mismas memorias — los dos sistemas coexisten: el MCP es el writer canónico, `rag` es un reader adicional via el embedding pipeline normal.

## Features que dependen de launchd: dejá el daemon ACTIVO al cerrar el commit

**Regla universal del repo**: cuando una feature nueva se completa con un plist `com.fer.obsidian-rag-*`, el daemon tiene que estar **cargado y verificado** al cierre del turno. NO dejar como TODO "corré `rag setup` cuando puedas". Una feature con cron-dependent behavior **no está completa** hasta que se demuestra que ejecuta sola.

Aprendido el 2026-04-25 con `com.fer.obsidian-rag-wa-scheduled-send` (worker de mensajes WhatsApp programados): el código + plist factory se shippearon en el commit `9740fa1`, pero el archivo nunca se copió a `~/Library/LaunchAgents/`. El user programó un mensaje, esperó la hora, y nada — el worker no existía como proceso. La feature parecía rota cuando en realidad solo faltaba el último paso operativo.

### Checklist al agregar un plist nuevo

1. **Código**: factory `_<nombre>_plist(rag_bin: str)` + tuple en la lista de [`rag/__init__.py:39190+`](rag/__init__.py) que `rag setup` consume.
2. **Click subcommand**: el comando que el plist ejecuta (`@cli.command("...")`).
3. **Smoke del comando manual**: `rag <subcomando> --dry-run` corre sin error y reporta.
4. **Generar el plist y copiarlo**:
   ```bash
   .venv/bin/python -c "import rag; print(rag._<nombre>_plist('/Users/fer/.local/bin/rag'))" \
     > ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   ```
5. **Cargar con launchctl**:
   ```bash
   launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   ```
6. **Verificar que está vivo**:
   ```bash
   launchctl list | grep obsidian-rag-<nombre>          # debe aparecer con state 0 o running
   launchctl print gui/$UID/com.fer.obsidian-rag-<nombre>  # ver run interval, last exit, runs
   ```
7. **Esperar al menos un tick** (o `launchctl kickstart -k gui/$UID/com.fer.obsidian-rag-<nombre>` para forzar el primer run) y verificar que el log se generó:
   ```bash
   tail -20 ~/.local/share/obsidian-rag/<nombre>.log
   ```
8. **Solo después** del paso 7, marcar la feature como completa.

### Daemons activos del proyecto (referencia)

Lista de plists registrados (cualquier `obsidian-rag-*` que `launchctl list` muestre):

| Plist | Cadencia | Comando | Propósito |
|---|---|---|---|
| `com.fer.obsidian-rag-watch` | filesystem watcher | `rag watch` | Auto-reindex del vault |
| `com.fer.obsidian-rag-serve` | KeepAlive | `rag serve` | MCP server |
| `com.fer.obsidian-rag-web` | KeepAlive | `web/server.py` | Web UI + chat |
| `com.fer.obsidian-rag-digest` | semanal | `rag digest` | Brief semanal |
| `com.fer.obsidian-rag-morning` | calendar 7am L-V | `rag morning` | Brief matinal |
| `com.fer.obsidian-rag-today` | calendar 22hs L-V | `rag today` | Brief vespertino |
| `com.fer.obsidian-rag-wake-up` | calendar | `rag wake-up` | Setup post-sleep |
| `com.fer.obsidian-rag-emergent` | viernes 10am | `rag emergent` | Detector de temas emergentes |
| `com.fer.obsidian-rag-patterns` | domingo 20:00 | `rag patterns` | Alertas de feedback |
| `com.fer.obsidian-rag-archive` | weekly | `rag archive` | Auto-archivo de notas muertas |
| `com.fer.obsidian-rag-wa-tasks` | 30min | `rag wa-tasks` | Extracción de tareas WhatsApp |
| `com.fer.obsidian-rag-reminder-wa-push` | 5min | `rag remind-wa` | Push de Reminders al WA |
| `com.fer.obsidian-rag-wa-scheduled-send` | 5min | `rag wa-scheduled-send` | **(nuevo 2026-04-25)** Worker de mensajes WA programados |
| `com.fer.obsidian-rag-anticipate` | 10min | `rag anticipate` | Anticipatory agent |
| `com.fer.obsidian-rag-auto-harvest` | weekly | `rag auto-harvest` | Auto-tune feedback |
| `com.fer.obsidian-rag-online-tune` | nightly 03:30 | `rag tune --apply` | Ranker-vivo nightly |
| `com.fer.obsidian-rag-calibrate` | nightly | `rag calibrate` | Score calibration |
| `com.fer.obsidian-rag-maintenance` | weekly | `rag maintenance` | Vacuum + WAL checkpoint + log rotation |
| `com.fer.obsidian-rag-consolidate` | nightly | `rag consolidate` | Memory consolidation |
| `com.fer.obsidian-rag-ingest-whatsapp` | horaria | `rag index --source whatsapp` | WhatsApp ingester |
| `com.fer.obsidian-rag-ingest-gmail` | horaria | `rag index --source gmail` | Gmail ingester |
| `com.fer.obsidian-rag-ingest-calendar` | cada 6h | `rag index --source calendar` | Google Calendar ingester |
| `com.fer.obsidian-rag-ingest-reminders` | cada 6h | `rag index --source reminders` | Apple Reminders ingester |
| `com.fer.obsidian-rag-ingest-drive` | horaria | `rag index --source drive` | Google Drive ingester |
| `com.fer.obsidian-rag-implicit-feedback` | 15min | `rag implicit-feedback` | Auto-harvest de feedback implícito |
| `com.fer.obsidian-rag-routing-rules` | 5min | `rag routing-rules` | Detector de patrones de ruteo |
| `com.fer.obsidian-rag-cloudflare-tunnel` | KeepAlive | `cloudflared tunnel` | HTTPS público via Cloudflare Quick Tunnel |
| `com.fer.obsidian-rag-cloudflare-tunnel-watcher` | daemon | `scripts/cloudflared_watcher.sh` | Monitor de túnel + notificación de URL |
| `com.fer.obsidian-rag-serve-watchdog` | daemon | Monitor de `com.fer.obsidian-rag-serve` | Watchdog para reiniciar serve si cae |
| `com.fer.obsidian-rag-vault-cleanup` | nightly | `rag vault-cleanup` | Limpieza de carpetas transitorias |
| `com.fer.obsidian-rag-whisper-vocab` | 03:15 | `rag whisper-vocab refresh` | Extracción nightly de vocab WhatsApp |
| `com.fer.obsidian-rag-brief-auto-tune` | Domingo 03:00 | `rag brief schedule auto-tune --apply` | **(nuevo 2026-04-29)** Auto-tune del horario de los briefs (morning/today/digest) basado en `rag_brief_feedback` |

Si el listado anterior queda desactualizado, el source de verdad es la lista de tuplas en [`rag/__init__.py:39190+`](rag/__init__.py) (función `_plists_to_install` o similar — `grep "_plists" rag/__init__.py`).

### Bypass: `rag setup` también funciona

Si la feature shippea junto con cambios al `rag setup` (o si el user prefiere reinstalar todo en bloque), `rag setup` instala/recarga TODOS los plists de la tabla anterior. Es más invasivo (puede recargar daemons que ya estaban corriendo bien) pero menos manual. Como compromiso: para plists nuevos individuales → recipe del checklist. Para refactors masivos → `rag setup`.

### Cuándo NO instalar el plist en el commit

Excepción legítima: si la feature requiere config previo del user (ej. OAuth de Gmail, ambient.json, etc.) y el plist crashea sin eso. En ese caso, el commit msg debe decir explícitamente "el plist NO se instala automáticamente porque requiere `<X>` primero" — no "corré `rag setup` cuando puedas" sin más contexto. El [`com.fer.obsidian-rag-ingest-calendar.plist`](rag/__init__.py) es ejemplo: `rag setup` lo skipea si `~/.calendar-mcp/credentials.json` no existe.

## Wave-8 gotchas — pipeline de filtros + carry-over (2026-04-28)

Tres patrones que se mordieron durante la wave-8 de la eval Playwright. Documentados acá porque cualquiera de los tres puede repetirse silenciosamente y costar una sesión entera de debug.

### Filtros definidos pero no cableados

**Síntoma**: el codebase tiene una clase `_XxxFilter` o función `_strip_*`/`_redact_*`/`_normalize_*` con regex completo + docstring + comentario explicando el bug que arregla, pero **ningún call site la invoca**. La intención es real, alguien la dejó "lista para conectar" y nunca la conectó. El bug que se suponía fixeada sigue ahí.

**Caso real**: `_strip_foreign_scripts` (`web/server.py:1504-1531`) existía con docstring "Remove characters from non-allowed scripts (CJK, Cyrillic, Hebrew, Arabic…)". Nunca se llamaba. CJK leak en respuestas de weather siguió en producción hasta wave-8.

**Cómo evitarlo en el futuro**:

1. Cuando agregues un nuevo filtro, también editá el `_emit()` helper dentro de `gen()` (línea ~11631 de `web/server.py`) Y la pipeline de cache replay (línea ~9887 — `_redact_pii(_sem_text)`).
2. Antes de "ya está, queda para wirear después", ya escribí el call site. Si lo dejás para "después" no llega.
3. Hay un test de regresión [`tests/test_filter_wiring.py`](tests/test_filter_wiring.py) que falla si una clase `_*Filter` o función `_strip_*`/`_redact_*` está definida sin call site. Si alguna vez te marca un false-positive (filter intencionalmente no usado), agregalo a la allowlist del test, no lo borres.

### Carry-over del pre-router silenciosamente sobrescrito por el fast-path

**Síntoma**: agregaste lógica al inicio de `gen()` que computa `_forced_tool_pairs` (lo que el pre-router decidió disparar). El log dice que se computó. Pero en la respuesta el tool nunca corre. La causa es que **otro branch downstream** dentro de la misma `gen()` está re-llamando `_detect_tool_intent(question)` y descartando tu `_forced_tool_pairs`.

**Caso real**: wave-8 carry-over anafórico. Pre-router setear `_forced_tool_pairs = [('weather', {'location': 'Barcelona'})]`. Línea 10996 hacía `_forced_tools = [] if _propose_intent else _detect_tool_intent(question)` que retornaba `[]` (sin la query "y en Barcelona?" no matchea ningún keyword). El tool nunca corría aunque el log decía que se había decidido. Fix: esa línea ahora hace `_forced_tools = list(_forced_tool_pairs)`.

**Cómo evitarlo en el futuro**:

```bash
# Antes de cerrar un fix que toque _forced_tool_pairs, grep por re-detección:
grep -n '_detect_tool_intent\|_forced_tools\s*=' web/server.py
```

Si aparece más de una asignación a `_forced_tools` o más de una llamada a `_detect_tool_intent`, **leé el contexto de cada una**. La regla es: el pre-router corre UNA vez al inicio de `gen()`, todo el resto del flow debe LEER de `_forced_tool_pairs`, no recomputar.

### Bumpeo de `_FILTER_VERSION` es parte del fix, no un extra

**Síntoma**: arreglaste un filtro / system prompt / regex que cambia el output user-facing. Validás vía Playwright. El test reporta que el bug sigue. Te volvés loco buscando el bug en tu código. La causa es que **el semantic cache sigue sirviendo respuestas pre-fix** porque la cache key no incluye nada que tu fix haya cambiado.

**Mecanismo**: `_FILTER_VERSION` (`rag/__init__.py:4656`) está horneado dentro de `_hash_chunk_count` (línea 4659+) y usado como parte del corpus_hash que entra en la cache key del semantic cache. Bumpear la string invalida TODAS las entries del cache pre-fix de un saque.

**Cuándo bumpear**:

- Cambia un regex que afecta tools_fired (PII redact, raw tool stripper, iberian leaks, foreign scripts, lo que sea).
- Cambia el `_WEB_SYSTEM_PROMPT` o cualquier de las REGLA N.
- Cambia la traducción de descriptions (weather, etc.) inyectada al CONTEXTO.
- Cualquier cambio que un user con cache hit verá como "no se aplicó tu fix".

**Cuándo NO bumpear**:

- Cambios en performance / refactors sin output change.
- Cambios en features off-by-default (gated por env var).
- Cambios en herramientas administrativas (CLI flags, scripts).

**Convención de naming**: `wave<N>-<YYYY-MM-DD>` ej. `wave8-2026-04-28`. Greppable + cronológico.
