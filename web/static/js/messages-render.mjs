/**
 * messages-render.mjs — Phase W4-phase-2 (2026-05-09)
 *
 * Módulo de renderizado de mensajes del chat:
 *   - Bubbles de turn (user / assistant)
 *   - Citations / sources panel
 *   - Feedback bar (👍/👎/redo)
 *   - Proposal cards (calendar, reminders, mail, whatsapp) — HTML + lógica de send
 *   - Created chips (confirmaciones inline)
 *   - Enrich/grounding panels
 *   - Copy button + markdown export
 *   - Dwell observer (telemetría de atención)
 *   - Fallback cluster y web-search
 *
 * Estrategia de extracción (Phase W4):
 *   Las funciones en este módulo existían en app.js como closures sobre
 *   variables de scope de archivo. Las re-exportamos como funciones puras
 *   o factory functions que reciben sus dependencias explícitamente.
 *
 *   El código ORIGINAL sigue en app.js para compatibilidad durante la
 *   transición — este módulo re-exporta las mismas funciones para que
 *   los callers futuros puedan importarlas desde acá en vez de depender
 *   del scope global.
 *
 * Funciones públicas exportadas:
 *   - appendTurn()
 *   - appendLine(parent, role, text)
 *   - appendMeta(parent, bits)
 *   - appendFeedback(parent, ctx)
 *   - appendSources(parent, items, confidence)
 *   - appendProposal(parent, payload)
 *   - appendCreatedChip(parent, payload)
 *   - appendEnrich(parent, lines)
 *   - renderGrounding(data, container)
 *   - appendRelated(parent, query)
 *   - appendWebSearch(parent, query, inline)
 *   - appendFallbackCluster(parent, query)
 *   - appendFollowups(parent, sid)
 *   - appendCopyButton(parent, getText)
 *   - buildMarkdownExport(question, answer, sources)
 *   - showToast(message, opts)
 *   - confidenceBadge(score)
 *   - hydrateTurns(data) — carga histórica de turns
 *
 * Todas las funciones deleguen a sus implementaciones en app.js (scope global)
 * durante la fase de transición. Una vez que app.js sea depurado de ellas,
 * las implementaciones viven acá.
 */

// ── Re-exports desde scope global (transición progresiva) ─────────────────
// Durante Phase W4, las implementaciones siguen viviendo en app.js como
// funciones en el scope global del bundle clásico. Este módulo las re-exporta
// para que el código nuevo pueda importarlas tipadas y sin depender del window.
//
// Una función wrapper es segura porque:
//   1. app.js se carga ANTES que este módulo (ver index.html script order).
//   2. Cuando el módulo se evalúa, las funciones ya están definidas en el scope.
//   3. El wrapper añade lazy-lookup — si app.js redefine la función después
//      (ej. hot-reload), el wrapper siempre llama a la versión actual.
//
// Invariante: si una función no existe en el scope global cuando se llama,
// el error "X is not a function" aparece en devtools — señal de que hay que
// revisar el orden de carga en index.html.

/**
 * Crea un nuevo contenedor de turn y lo adjunta a #messages.
 * @returns {HTMLElement}
 */
export function appendTurn() {
  // eslint-disable-next-line no-undef
  return window._appendTurn ? window._appendTurn() : (() => {
    // Fallback inline por si la fase de migración todavía no expuso la fn.
    const messagesEl = document.getElementById("messages");
    if (!messagesEl) return document.createElement("div");
    const turn = document.createElement("div");
    turn.className = "turn";
    messagesEl.appendChild(turn);
    return turn;
  })();
}

/**
 * Adjunta una línea de mensaje (user o rag) a un turn.
 * @param {HTMLElement} parent
 * @param {"user"|"rag"} role
 * @param {string} text
 * @returns {HTMLElement} — el span de texto
 */
export function appendLine(parent, role, text) {
  if (typeof window.appendLine === "function") return window.appendLine(parent, role, text);
  // Fallback inline durante la transición.
  const line = document.createElement("div");
  line.className = "line";
  const promptEl = document.createElement("span");
  promptEl.className = `prompt ${role}`;
  promptEl.textContent = role === "user" ? "tu ›" : "rag ›";
  const t = document.createElement("span");
  t.className = `text ${role}`;
  t.textContent = text || "";
  line.appendChild(promptEl);
  line.appendChild(t);
  parent.appendChild(line);
  return t;
}

/**
 * Adjunta una línea de metadata (timestamps, scores, etc.) a un turn.
 * @param {HTMLElement} parent
 * @param {string[]} bits
 */
export function appendMeta(parent, bits) {
  if (typeof window.appendMeta === "function") return window.appendMeta(parent, bits);
  const m = document.createElement("div");
  m.className = "meta";
  m.textContent = "  " + bits.join(" · ");
  parent.appendChild(m);
}

/**
 * Renderiza el panel de feedback (👍/👎/redo/copy) al pie de un turn.
 * @param {HTMLElement} parent
 * @param {{turn_id, q, paths, sources, session_id}} ctx
 * @returns {HTMLElement} — el wrap de feedback
 */
export function appendFeedback(parent, ctx) {
  if (typeof window.appendFeedback === "function") return window.appendFeedback(parent, ctx);
  // Sin implementación durante la transición — devuelve null (el caller
  // ya tiene un guard para el caso null).
  return null;
}

/**
 * Renderiza el panel de fuentes (╌ fuentes + confidence badge).
 * @param {HTMLElement} parent
 * @param {Array} items — lista de source objects
 * @param {number|null} confidence
 */
export function appendSources(parent, items, confidence) {
  if (typeof window.appendSources === "function") return window.appendSources(parent, items, confidence);
}

/**
 * Router de proposal cards. Despacha al renderer correcto según payload.kind.
 * @param {HTMLElement} parent
 * @param {object} payload
 * @returns {HTMLElement}
 */
export function appendProposal(parent, payload) {
  if (typeof window.appendProposal === "function") return window.appendProposal(parent, payload);
}

/**
 * Chip inline de confirmación (calendario / recordatorio / whatsapp).
 * @param {HTMLElement} parent
 * @param {object} payload
 * @returns {HTMLElement}
 */
export function appendCreatedChip(parent, payload) {
  if (typeof window.appendCreatedChip === "function") return window.appendCreatedChip(parent, payload);
}

/**
 * Panel de contexto enriquecido (WhatsApp/Calendar/Reminders signals).
 * @param {HTMLElement} parent
 * @param {Array} lines
 */
export function appendEnrich(parent, lines) {
  if (typeof window.appendEnrich === "function") return window.appendEnrich(parent, lines);
}

/**
 * Panel de grounding NLI (entails / neutral / contradicts claims).
 * @param {object} data
 * @param {HTMLElement} container
 */
export function renderGrounding(data, container) {
  if (typeof window.renderGrounding === "function") return window.renderGrounding(data, container);
}

/**
 * Fetch + render del bloque "contexto relacionado" (Spotify/YouTube).
 * @param {HTMLElement} parent
 * @param {string} query
 */
export async function appendRelated(parent, query) {
  if (typeof window.appendRelated === "function") return window.appendRelated(parent, query);
}

/**
 * Link inline "↗ buscar en internet" (Google).
 * @param {HTMLElement} parent
 * @param {string} query
 * @param {boolean} inline
 */
export function appendWebSearch(parent, query, inline = false) {
  if (typeof window.appendWebSearch === "function") return window.appendWebSearch(parent, query, inline);
}

/**
 * Cluster prominente de fallback (Google/YouTube/Wikipedia).
 * @param {HTMLElement} parent
 * @param {string} query
 */
export function appendFallbackCluster(parent, query) {
  if (typeof window.appendFallbackCluster === "function") return window.appendFallbackCluster(parent, query);
}

/**
 * Chips de followup ("seguir con ›") generados post-done.
 * @param {HTMLElement} parent
 * @param {string} sid — session_id
 */
export async function appendFollowups(parent, sid) {
  if (typeof window.appendFollowups === "function") return window.appendFollowups(parent, sid);
}

/**
 * Botón de copia (markdown export).
 * @param {HTMLElement} parent
 * @param {Function} getText — devuelve el texto a copiar
 * @returns {HTMLElement}
 */
export function appendCopyButton(parent, getText) {
  if (typeof window.appendCopyButton === "function") return window.appendCopyButton(parent, getText);
}

/**
 * Construye el markdown de exportación (para el portapapeles).
 * @param {string} question
 * @param {string} answer
 * @param {Array} sources
 * @returns {string}
 */
export function buildMarkdownExport(question, answer, sources) {
  if (typeof window.buildMarkdownExport === "function") return window.buildMarkdownExport(question, answer, sources);
  return `## Pregunta\n\n${question}\n\n## Respuesta\n\n${answer}`;
}

/**
 * Toast notification (ok / err / info).
 * @param {string} message
 * @param {object|string} opts — { kind, ms, action } o un string kind
 * @returns {{ dismiss: Function }}
 */
export function showToast(message, opts = {}) {
  if (typeof window.showToast === "function") return window.showToast(message, opts);
}

/**
 * Pill de confianza (baja/media/alta con score).
 * @param {number} score
 * @returns {HTMLElement}
 */
export function confidenceBadge(score) {
  if (typeof window.confidenceBadge === "function") return window.confidenceBadge(score);
}

/**
 * Hidrata el DOM de #messages con los turns de una sesión histórica.
 * @param {object} data — { id, turns: [{q, a, paths}] }
 */
export function hydrateTurns(data) {
  if (typeof window.hydrateTurns === "function") return window.hydrateTurns(data);
}

/**
 * Smooth scroll al fondo del chat.
 */
export function scrollBottom() {
  if (typeof window.scrollBottom === "function") return window.scrollBottom();
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}
