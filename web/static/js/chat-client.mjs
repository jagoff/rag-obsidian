/**
 * chat-client.mjs — Phase W4-phase-2 (2026-05-09)
 *
 * Cliente SSE del chat: envío de mensajes, stream de tokens, eventos SSE,
 * abort controller, retry automático en errores de red, side fetches.
 *
 * Estado original en app.js (líneas ~4227-4866):
 *   - pending, currentController
 *   - inflightSideFetches Set, abortSideFetches()
 *   - pendingRetryTimer, pendingRetryCountdown, cancelPendingAutoRetry()
 *   - _NETWORK_ERROR_RE, _isNetworkError(err), _friendlyChatErrorMessage(err)
 *   - send(question, opts) — función principal con handleEvent() interna
 *   - handleSlashCommand(raw) — comandos /new /save /redo etc.
 *   - SLASH_COMMANDS array
 *   - pushSystemMessage(kind, text)
 *
 * Estrategia de extracción Phase W4:
 *   send() es la función más entrelazada del sistema — usa ~30 variables de
 *   scope del archivo (messagesEl, input, sessionId, vaultScope, getChatMode,
 *   ttsEnabled, lastTurnId, lastUserQuestion, etc.). Moverla requiere un
 *   context object explícito o un store compartido.
 *
 *   Durante la transición, este módulo re-exporta desde window.* para que
 *   callers futuros puedan importar de acá en vez de depender del scope global.
 *   La implementación completa queda en app.js hasta una fase posterior.
 *
 * Globals del window consumidos:
 *   window.send(question, opts?)
 *   window.handleSlashCommand(raw)
 *   window.pushSystemMessage(kind, text)
 *   window.abortSideFetches()
 *   window.cancelPendingAutoRetry()
 */

// ── Constantes ────────────────────────────────────────────────────────────

/**
 * Regex para detectar errores de red transitorios (distintos de HTTP 4xx/5xx).
 * Chrome/Edge: "Failed to fetch"
 * Firefox: "NetworkError when attempting to fetch resource."
 * Safari: "The network connection was lost.", "Load failed"
 */
export const NETWORK_ERROR_RE = /failed to fetch|networkerror|network error|network connection|connection lost|load failed|net::|err_internet|connection appears to be offline/i;

/**
 * Lista canónica de slash commands — refleja app.js SLASH_COMMANDS.
 * Exportada para que slash-popover pueda consumirla sin depender del global.
 */
export const SLASH_COMMANDS = [
  { cmd: "/help",    desc: "atajos y comandos" },
  { cmd: "/cls",     desc: "limpiar vista (sesión intacta)" },
  { cmd: "/new",     desc: "nueva sesión (olvida historial)" },
  { cmd: "/session", desc: "info de la sesión actual" },
  { cmd: "/model",   desc: "modelo de chat en uso" },
  { cmd: "/save",    desc: "guardar conversación en 00-Inbox", arg: "[título]" },
  { cmd: "/reindex", desc: "reindex incremental en background" },
  { cmd: "/redo",    desc: "regenerar última respuesta (opcional: pista)", arg: "[pista]" },
  { cmd: "/tts",     desc: "alternar voz (Mónica)" },
  { cmd: "/wzp",     desc: "enviar WhatsApp a un contacto", arg: "[contacto]: [mensaje]" },
  { cmd: "/mail",    desc: "enviar email (Gmail)", arg: "[email]: [asunto] — [cuerpo]" },
  { cmd: "/rem",     desc: "crear recordatorio", arg: "[texto] [cuándo]" },
  { cmd: "/evt",     desc: "agendar evento", arg: "[título] [cuándo]" },
];

// ── Helpers de clasificación de errores ──────────────────────────────────

/**
 * Devuelve true si el error es un error de red transitorio (retry worth it).
 * @param {Error|null} err
 * @returns {boolean}
 */
export function isNetworkError(err) {
  if (!err) return false;
  return NETWORK_ERROR_RE.test((err.message || "").toString());
}

/**
 * Mensaje friendly en español para el error de red.
 * @param {Error} err
 * @returns {string}
 */
export function friendlyChatErrorMessage(err) {
  if (isNetworkError(err)) return "  conexión interrumpida — el servidor se reinició mid-respuesta";
  return `  error: ${err.message}`;
}

// ── Re-exports desde window (transición progresiva) ───────────────────────

/**
 * Envía una pregunta al backend via SSE.
 * @param {string} question
 * @param {{redo_turn_id?: string, hint?: string, _isAutoRetry?: boolean}} opts
 */
export async function send(question, opts = {}) {
  if (typeof window.send === "function") return window.send(question, opts);
}

/**
 * Interpreta y ejecuta un slash command (/new, /save, /redo, etc.).
 * @param {string} raw — texto raw del input (empieza con /)
 * @returns {Promise<boolean>} — true si fue manejado, false si se debe enviar al LLM
 */
export async function handleSlashCommand(raw) {
  if (typeof window.handleSlashCommand === "function") return window.handleSlashCommand(raw);
  return false;
}

/**
 * Muestra un mensaje de sistema en el chat (meta / error).
 * @param {"meta"|"err"} kind
 * @param {string} text
 */
export function pushSystemMessage(kind, text) {
  if (typeof window.pushSystemMessage === "function") return window.pushSystemMessage(kind, text);
}

/**
 * Aborta todos los side fetches in-flight (related/followups/contacts).
 * Llamar al navegar de sesión o iniciar /new.
 */
export function abortSideFetches() {
  if (typeof window.abortSideFetches === "function") return window.abortSideFetches();
}

/**
 * Cancela el timer de auto-retry pendiente (si el user reinicia una sesión
 * o tipea otra pregunta antes de que el contador llegue a 0).
 */
export function cancelPendingAutoRetry() {
  if (typeof window.cancelPendingAutoRetry === "function") return window.cancelPendingAutoRetry();
}

/**
 * Obtiene si hay un stream en curso.
 * @returns {boolean}
 */
export function isPending() {
  return typeof window.pending === "boolean" ? window.pending : false;
}

/**
 * Aborta el stream actual si hay uno in-flight.
 */
export function abortCurrentStream() {
  const ctrl = window.currentController;
  if (ctrl && typeof ctrl.abort === "function") {
    try { ctrl.abort(); } catch (_) {}
  }
}
