# Playwright multi-turn eval — issues log

**Fecha:** 2026-04-28 noche
**Método:** Playwright UI real contra `http://127.0.0.1:8765/chat`, conversaciones multi-turn (3-5 turns cada una). Cada conv prueba un eje distinto. Se documentan TODOS los issues encontrados sin fixearlos en el momento — la idea es triagear después.

**Contexto previo (waves 1-6 ya en master):** pt leak masivo, raw tool call leak, propose intent guard weather, canonical filename typos, cache stale por tool intent, weather regex relaxation, multi-event helper, Ollama auto-detect deployment, follow-up regex sync (rag/__init__ y web/server), metachat tightening "cómo va a estar".

## Cómo se categorizan los issues

- **CRITICAL** — UX completamente roto / privacy leak / internal error visible al user
- **HIGH** — funciona pero con bug claro (off-topic, hallucination, wrong tool fired, false confirmation)
- **MEDIUM** — UX sub-óptima pero la respuesta es aceptable
- **LOW** — nit/cosmético

## Conversaciones testeadas

| # | Tema | Turns | OK | Issues |
|---|---|---|---|---|
| 1 | Vault deep dive (Coaching) | 5 | 5/5 ✅ | ninguno — context preservation perfect |
| 2 | Cross-source (mail/cal/reminder) | 4 | 4/4 ✅ | nits cosméticos |
| 3 | Reminder create + edit + cancel | 4 | 1/4 ❌ | 3 bugs HIGH |
| 4 | Time travel (hoy/mañana/semana) | 4 | 3/4 ⚠️ | 1 bug HIGH |
| 5 | Personal info | 4 | 4/4 funciona pero **CRITICAL privacy leak** |
| 6 | Técnico (RAG arquitectura) | 4 | 3/4 ❌ | 1 bug CRITICAL |
| 7 | Edge cases (typos/multilang) | 4 | 4/4 ⚠️ | nits |
| 8 | Drift recovery | 4 | 2/4 ❌ | 1 bug HIGH |

**Total: 33 turns, ~25 OK, ~8 con bug, 2 CRITICAL, 5 HIGH, varios MEDIUM/LOW.**

---

## Issues abiertos para triagear

### 🚨 CRITICAL #1 — Password/credenciales del vault salen en chat responses

**Repro:**
```
Conv 5, T2:
user: quién es mi mamá
bot: Tu mamá se llama María y tiene los siguientes datos:
     Correo electrónico: gadeamaria662@gmail.com
     Contraseña: t5#61VSzDhM9       ← PASSWORD LEAKED
     Teléfono: +54 9 342 430 3891
     DNI: 30.824.421
     ...
```

**Impacto:** privacy / security leak severísimo. El vault del user tiene una nota de contacto de su mamá con la contraseña en texto plano (probablemente importada de algún password manager). El LLM no tiene filtro y la lee + emite.

**Fix sugerido:**
1. **Inmediato:** post-process filter en el streaming pipeline que detecte patrones tipo `Contraseña: \S+` / `Password: \S+` / `DNI: [\d.]+` y los reemplace con `[REDACTED]`. Similar al `_replace_iberian_leaks` y `_normalize_canonical_filenames`. Greppable después por `[chat-pii-redact]`.
2. **Mediano plazo:** detectar entidades sensibles en build time del corpus (cuando se ingiere la nota) y marcarlas — el retriever puede saltearlas o devolver un placeholder.
3. **Largo plazo:** opt-in classifier de "datos sensibles" por nota — el user marca `private: true` en el frontmatter y esas notas no se ingieren para chat responses (sí para search por path explícito).

**Prioridad:** P0 — fixear ANTES de cualquier otro issue.

---

### 🚨 CRITICAL #2 — Internal error message expuesto al user

**Repro:**
```
Conv 6, T1:
user: cómo funciona el sistema RAG
bot: retrieve falló: bm25_search llamado en paralelo — es GIL-serialised
     por diseño (CLAUDE.md línea 126, medido 3× slower paralelo en M3 Max).
     Revisá el stack si vino de un ThreadPoolExecutor.
```

**Impacto:** dev-facing assertion message visible al end-user. Confunde + filtra detalles internos del sistema (CLAUDE.md, ThreadPoolExecutor, M3 Max).

**Causa probable:** el assertion check `assert not in_parallel, "..."` en `bm25_search` se ejecuta cuando algún path llama el search desde un ThreadPoolExecutor. La excepción se levanta con el mensaje completo y termina como `LLM falló: ...` o `retrieve falló: ...` en el SSE error event.

**Fix sugerido:**
1. Sanitizar el SSE error event: si el message contiene "CLAUDE.md", "ThreadPoolExecutor", "GIL-serialised", reemplazar por mensaje user-friendly: "Hubo un error procesando tu consulta — probá de nuevo o reformulá".
2. Loggear el error original en `web.log` (full stack trace + assertion text) para debug.
3. Long-term: arreglar el bm25_search-en-paralelo bug — buscar el call site que viola el contract.

**Prioridad:** P0 — fixear junto al PII filter (mismo pipeline).

---

### ⚠️ HIGH #3 — "y mañana?" alucina propose_reminder

**Repro:**
```
Conv 4, T1: "qué hago hoy" → calendar_ahead etc fired ✅
Conv 4, T2: "y mañana?" → emite raw `propose_reminder(...)` syntax,
            stripper lo cataptura → respuesta = clarificación de
            "Necesito un poco más para crearte el recordatorio..."
```

**Impacto:** el follow-up "y mañana?" es naturalmente "qué tengo MAÑANA" (continuación de "qué hago hoy"). El LLM lo confunde con propose-intent.

**Causa probable:** el LLM asocia "mañana" con propose_reminder de turnos previos. La history kept incluye un `propose_reminder` de la conv anterior + el "mañana" del current dispara el modelo a re-emitir el tool. El RawToolCallStripper hace su trabajo (no leak el syntax) pero la respuesta es inútil para el user.

**Fix sugerido:**
1. Cuando el turno previo fue `reminders_due`/`calendar_ahead` (read-intent), el follow-up "y mañana?" / "y la semana?" / "y el viernes?" debería rutear ESPECÍFICAMENTE al MISMO tool con un parámetro de fecha shift, NO a propose_*.
2. Concretamente: pre-router que mire el tool de la respuesta anterior y si es read-intent + temporal, vuelva a fire el mismo tool con date offset.
3. Alternativa: cuando el RawToolCallStripper detecta propose_reminder pero el query es "y X?" (anaphoric), sub-categorizar el clarification: "Querés un recordatorio o querés saber qué tenés para X?"

**Prioridad:** P1.

---

### ⚠️ HIGH #4 — "sumále también X" no actualiza el recordatorio anterior, pero el bot dice que sí

**Repro:**
```
Conv 3, T1: recordáme llamar al médico el lunes a las 10am
            → ✓ creado correctamente
Conv 3, T2: sumále también llevar la receta
            → response: "Se ha programado un recordatorio para que llevás
              la receta al médico el lunes a las 10am"
            → tools=[] empty, createdText="" empty
            → NADA SE GUARDÓ pero el bot DIJO que sí.
```

**Impacto:** false confirmation grave. El user piensa que el recordatorio tiene "llevar la receta" y "llamar" combinados. En realidad solo está "llamar al médico".

**Causa probable:** no hay tool de "edit_reminder" / "update_reminder". El LLM trata de inferir y compone una respuesta natural pero no llama ninguna tool. El stripper de raw-tool-call no se gatilla porque el LLM no emite syntax — solo prosa que afirma el efecto.

**Fix sugerido:**
1. **Mediano plazo:** agregar tool `propose_reminder_edit(reminder_id, new_title=None, new_when=None)`.
2. **Inmediato:** detectar el patrón "sumále/agregale/cambiale + X" cuando el turno previo fue `propose_reminder`, y responder explícitamente: "No puedo editar el recordatorio anterior. Si querés, lo cancelo y creo uno nuevo con todo: 'recordame llamar al médico y llevar la receta el lunes a las 10am'."
3. Refinar REGLA 1.b para PROHIBIR confirmar acciones que no se ejecutaron (NUNCA "se ha programado/creado/etc" si tools_fired=[]).

**Prioridad:** P1.

---

### ⚠️ HIGH #5 — "qué recordatorios tengo" no ve recordatorios recién creados

**Repro:**
```
Conv 3, T1: recordáme llamar al médico el lunes a las 10am
            → ✓ created (event chip visible)
Conv 3, T3: qué recordatorios tengo
            → "No tenés recordatorios pendientes." (reminders_due fired)
```

**Impacto:** UX confusa. El user acaba de crear un recordatorio Y EL BOT DICE QUE NO LO TIENE. Pierde confianza.

**Causa probable:**
- El recordatorio se creó pero EventKit no lo ha sincronizado todavía (cache lag)
- O `reminders_due` filtra a "due hoy/esta semana" y el lunes 04-05 está fuera del rango default
- O el reminder está en estado "proposal pending accept" no "actual reminder"

**Fix sugerido:**
1. Post-create, poll EventKit con un timeout corto (~2s) hasta que el reminder aparezca, antes de devolver "✓ creado".
2. Si `reminders_due` no encuentra nada Y hubo un `propose_reminder` reciente en la session, agregar nota: "(El que acabás de crear puede tardar unos segundos en aparecer)".
3. Aumentar el horizonte default de `reminders_due` de "esta semana" a "30 días" (o respetar el `days_ahead` del schema).

**Prioridad:** P2.

---

### ⚠️ HIGH #6 — "cancelar el del médico" rutea a propose_whatsapp_cancel_scheduled

**Repro:**
```
Conv 3, T4: cancelar el del médico
            → fires propose_whatsapp_cancel_scheduled (WRONG TOOL)
            → response: "No tenés ningún mensaje programado para tu mamá..."
```

**Impacto:** "el del médico" claramente refiere al recordatorio del médico (T1), no a un WhatsApp. El LLM picked the wrong cancel tool del catálogo.

**Causa probable:** no existe `propose_reminder_cancel`. El LLM, con instrucción "el user pide cancelar X", busca un cancel-tool y solo encuentra el de WhatsApp scheduled. Lo dispara con args malos.

**Fix sugerido:**
1. **Mediano plazo:** agregar `propose_reminder_cancel(reminder_id_or_title)` y `propose_calendar_cancel(event_id_or_title)`.
2. **Inmediato:** REGLA en el system prompt: "PROHIBIDO usar `propose_whatsapp_cancel_scheduled` cuando el contexto NO es WhatsApp. Si el user pide cancelar algo y no hay tool disponible, decí 'No puedo cancelar [X] desde acá — abrí Apple Reminders/Calendar y eliminá manualmente'".

**Prioridad:** P2.

---

### ⚠️ HIGH #7 — Weather follow-up sin location carry

**Repro (conv anterior, no en este eval pero confirmado en wave-5):**
```
T1: qué clima en Madrid hoy → weather Madrid ✅
T2: cuánta lluvia se espera? → weather con location=Santa Fe (WRONG)
T3: y en Barcelona? → no fired weather, usó template "no tengo data fresca"
```

**Impacto:** location no se preserva entre turns. El user tiene que repetir "en Madrid" en cada follow-up.

**Causa probable:** `_detect_weather_explicit_location_intent` requiere matching del patrón "clima en X". Los follow-ups "y en X?" / "y mañana?" no tienen "clima" + prep + ciudad.

**Fix sugerido:**
1. Cuando el turno previo fue `weather`, registrar `last_location` en la session.
2. Si current query matchea `^y\s+en\s+(\w+)\??$` → weather con location=match[1].
3. Si current query es anaphoric weather follow-up (e.g. "y mañana?", "cuánta lluvia") → weather con location=last_location.

**Prioridad:** P2.

---

### ⚠️ MEDIUM #8 — Topic dilution después de 3+ follow-ups (limit conocido)

**Repro:**
```
T1: qué dice mi nota Ikigai → Ikigai resources
T2: profundizá en el primer recurso → ✅ stays on Ikigai
T3: qué otros materiales me recomendarías? → ✅ stays on Ikigai (post wave-6 fix)
T4: cuál sería el principal mensaje? → ❌ goes to chats/Maria/Moni (off-topic)
```

**Impacto:** después de 3-4 follow-ups, el `last_user_q` que se concatena para retrieval ya es un follow-up también, perdiendo el anchor del topic original.

**Causa probable:** estrategia CONCAT de last_user_q + current. Funciona 1-2 turns pero se diluye a partir del 3ro.

**Fix sugerido:**
1. **Mejor:** track de "topic anchor" — la primera query del topic actual queda como anchor hasta que detecte topic shift real. Concatenar anchor + current.
2. Alternativa más simple: tomar las últimas 2-3 user queries y concatenar todas (no solo la inmediata anterior).

**Prioridad:** P3 — limit conocido, no rompe UX gravemente.

---

### ⚠️ MEDIUM #9 — Mails en respuesta exponen commit SHA / metadata interna

**Repro:**
```
Conv 2, T1: qué mails pendientes tengo
bot: [jagoff/rag-obsidian] Run failed: CI - master (4a03565) (de Fer F)
     [jagoff/rag-obsidian] Run failed: CI - master (c334008) (de Fer F)
     ...
```

**Impacto:** cosmético. El user pidió "mails pendientes", la respuesta lista subjects con SHAs y `[jagoff/rag-obsidian]` brackets. Información cruda — el user real probablemente quiere "tenés 4 mails de CI fallando + 1 de OSDE + 1 personal" como summary.

**Fix sugerido:**
1. Resumen post-procesamiento de mails: agrupar por sender o sujeto-pattern (e.g. "X mails de CI failed: ...; Y mails de OSDE: ...").
2. Si los mails listados tienen patrón de notification (CI runs, security alerts, marketing), separarlos visualmente del resto.

**Prioridad:** P3.

---

### ⚠️ MEDIUM #10 — "qué eventos puedo postergar" no usa context del turn previo

**Repro:**
```
Conv 2, T2: qué eventos tengo agendados esta semana
            → list de eventos (Pedro, Juan, etc)
Conv 2, T3: y de eso qué puedo postergar?
            → "Según tus notas, no hay ningún evento o compromiso..." +
              menciona WhatsApp con Monica (off-topic)
```

**Impacto:** "y de eso" claramente refiere a los eventos listados en T2. El LLM no traduce "eso" a los eventos específicos y va al vault search genérico.

**Causa probable:** el follow-up resolution funciona para CONCAT del turno previo (que dispara retrieval), pero el retrieval con "qué eventos tengo... y de eso qué puedo postergar?" no encuentra documentos relevantes — los eventos son data live, no vault.

**Fix sugerido:**
1. Cuando el turno previo emitió tool output (calendar_ahead resultado con eventos), pasar ese OUTPUT como contexto al turno siguiente, no solo la pregunta.
2. Concretamente: el system prompt del turno actual debería incluir "EVENTOS RECIENTES (del turno anterior): ...".

**Prioridad:** P3.

---

### ⚠️ MEDIUM #11 — Cache stale serve respuestas con bugs ya fixeados

**Repro:**
```
Conv 1, T1: qué dice mi nota Ikigai
            → 4.5s, cached_layer=semantic
            → response es la misma de antes (con pt leak parcial)
```

**Impacto:** users ven respuestas pre-fix aún post-deploy. Confunde la validación de fixes.

**Causa probable:** el semantic cache key NO incluye una versión del prompt o de los filtros. Cuando cambian los filtros (iberian leaks, raw_tool_call_stripper, prompt rules), las respuestas viejas siguen sirviéndose.

**Fix sugerido:**
1. Incluir `prompt_version` y `filter_version` en el semantic cache key. Bumpear cuando cambia algún regex/system prompt relevante. Las entradas viejas se ignoran automáticamente.
2. Alternativa: TTL más corto (24h → 1h) durante períodos de iteración rápida.

**Prioridad:** P3 — workaround: el user puede agregar 1 char a la query para evitar cache.

---

### ✅ MEDIUM #12 — Server crashes mid-conversation ocasionalmente — FIXED 2026-04-28

**Repro:**
```
Conv 8, T2: qué decías sobre LangChain? → "error: network error"
Conv 8, T3: a ver si tengo otra referencia → "error: Failed to fetch"
Conv 8, T4: olvidemos esto, qué hago hoy? → ✅ recovered
```

**Impacto:** el user pierde 2 turns de la conversación. El recovery tarda ~30s.

**Causa real (post-investigación 2026-04-28):** la hipótesis original ("ollama wedge → auto-recovery") era parcial. El verdadero culprit es **`launchctl print` mostraba `runs = 46` en 2h** con un mix de:
- `SIGKILL by launchd[1]` — jetsam macOS killing el web server bajo memory pressure (qwen2.5:7b + reranker + bge-m3 saturando los 36 GB unified).
- `SIGTERM-then-SIGKILL` escalado — graceful shutdown cortado a los 5s (default `ExitTimeOut`) cuando había una SSE en vuelo, escalado a SIGKILL con la respuesta a medio drenar.

Cuando eso pasa, el browser ve `Failed to fetch` (intentando POST durante el respawn de ~30s de launchd) o `network error` (body interrumpido mid-stream).

**Fix shippeado:**
1. **Plist hardening** (`rag/__init__.py:_web_plist` + plist instalado):
   - `RAG_MEMORY_PRESSURE_THRESHOLD=80` (era 85 default) → desaloja chat model + reranker proactivamente, antes que jetsam pegue.
   - `ExitTimeOut=20` (era 5) → uvicorn tiene 20s para drenar SSE en vuelo antes que escale a SIGKILL.
   - `ProcessType=Interactive` → hint a launchd que es servicio foreground, jetsam lo trata con prioridad menor (último en morir bajo presión).
2. **Client-side auto-retry** (`web/static/app.js`):
   - `_isNetworkError(err)` detecta los strings de fetch errors browser (Failed to fetch, NetworkError, network connection lost, Load failed, net::, connection appears to be offline).
   - `_friendlyChatErrorMessage(err)` reemplaza el "error: Failed to fetch" inglés por "conexión interrumpida — el servidor se reinició mid-respuesta" en español.
   - Auto-retry una vez tras 8s con countdown visible ("↻ reintentando en 8s…") + botón "cancelar" si el user prefiere abortar.
   - Si el segundo intento también falla por red → muestra mensaje terminal "esperá unos segundos más y volvé a tipear la pregunta" sin loop infinito.
3. **Cancelación correcta del timer** en navegación (/new, loadSession, triggerNewChat) vía `abortSideFetches() → cancelPendingAutoRetry()`.

**Verificación:** `tests/test_plist_web_serve.py` (19 passed), `node -e` smoke test del regex de network errors (12/12 pass), `launchctl print` confirmó `exit timeout = 20` + `RAG_MEMORY_PRESSURE_THRESHOLD => 80` post-bootstrap.

**Prioridad:** P2 — fixed.

---

### 💡 LOW #13 — "do you speak english?" interpretado como pregunta sobre el user

**Repro:**
```
Conv 7, T4: do you speak english?
bot: Sí, hablas inglés. Tenés varias notas relacionadas con el inglés...
```

**Impacto:** "you" en español ambiguo (vos / yo). El bot tomó "you" como referencia al user, no a sí mismo. Aceptable pero raro.

**Fix sugerido:** detectar si la pregunta empieza con verbo + you (English), responder explícitamente "Sí, te puedo responder en inglés también — pero por defecto uso español".

**Prioridad:** P4 — nit.

---

### 💡 LOW #14 — "tengo plata ahorrada?" responde con gastos, no ahorros

**Repro:**
```
Conv 7, T3: tengo plata ahorrada?
bot: Tarjeta $549.439 + U$S98.93 + MOZE $5.119.709
```

**Impacto:** el user pregunta por savings, el bot responde con consumed (gastos en tarjeta + balance MOZE). MOZE es app de gastos, no de ahorros.

**Fix sugerido:** mejorar el detector de "savings vs spending". Si el user dice "ahorrado" / "savings" / "tengo guardado", devolver explicitamente "No tengo info sobre ahorros — Money app trackea gastos. Si guardás en una app distinta, conectála."

**Prioridad:** P4.

---

## Resumen de fixes sugeridos por prioridad

### P0 (CRITICAL — fixear ya)
1. **PII redaction post-stream** (passwords, DNI, phone — Conv 5 leak)
2. **Sanitize SSE errors** (no exponer assertion text al user — Conv 6)

### P1 (HIGH — fixear esta sesión o la próxima)
3. **Follow-up shift desde read-intent** ("y mañana?" tras "qué hago hoy" debe re-fire calendar_ahead, no propose_reminder)
4. **No false confirmations** ("sumále" sin tool fired NO debe decir "se ha programado")

### P2 (HIGH — backlog)
5. **EventKit cache lag** post-create
6. **Reminder/calendar cancel tools** (no existen, LLM rutea al de WhatsApp por confusión)
7. **Weather location carry** (`y en Barcelona?` tras Madrid)
8. **Auto-recovery no debe matar web server**

### P3 (MEDIUM)
9. **Topic anchor multi-turn** (limit conocido del CONCAT)
10. **Mails summary** (agrupar CI/notifications)
11. **Tool output context al siguiente turn** ("y de eso qué puedo postergar?")
12. **Cache versioning** (prompt_version + filter_version en key)

### P4 (LOW — nits)
13. **"you" ambiguity** en preguntas multilang
14. **Savings vs spending** distinción

## Telemetría útil para tracking

Tags greppables en `web.log` que ya existen y ayudan al triagear:
- `[chat-stream-error]` — todas las excepciones del stream
- `[chat-raw-tool-call]` — cuando el stripper se gatilla (model leak)
- `[ollama-auto-recovery]` — cuando se reinicia el daemon
- `[chat-fast-path-downgrade]` — cuando el routing cambia de modelo
- `[chat-llm-fallback]` — cuando el pre-router missed y va a LLM safety net

Tag a agregar:
- `[chat-pii-redact]` — cada vez que el filter de PII redacta algo
- `[chat-error-sanitized]` — cuando el SSE error se reescribe

## Próximos pasos

1. **Fixear P0 inmediatamente** (PII + error sanitization).
2. **P1 en la próxima sesión** de eval/iteración.
3. **Triagear P2-P3 con el user** — algunos requieren product decisions (ej. agregar tools de cancel).
