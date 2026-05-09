/**
 * sidebar.mjs — Phase W4-phase-2 (2026-05-09)
 *
 * Sidebar claude.ai-style con sessions list, búsqueda, new-chat button
 * y collapse/expand. Incluye el mobile drawer (open/close, focus trap).
 *
 * Estado original en app.js (líneas ~5274-5692):
 *   - SIDEBAR_COLLAPSED_KEY constante
 *   - sidebar, sidebarOpenBtn, sidebarCloseBtn, sidebarCollapseBtn refs
 *   - newChatBtn, mobileNewBtn refs
 *   - sessionsList, sessionsSearch, sessionsRefreshBtn refs
 *   - sessionsCache []
 *   - initSidebarCollapse(), initSidebarMobile(), initNewChatButtons()
 *   - initSessionsSearch(), initSidebarShortcut()
 *   - refreshSessions(), renderSessions(filter), loadSession(sid)
 *   - hydrateTurns(data), formatSessionMeta(s)
 *   - openSidebarMobile(), closeSidebarMobile(), triggerNewChat()
 *   - applySidebarCollapsed(collapsed), getFilterText()
 *
 * Estrategia de extracción Phase W4:
 *   Las funciones originales en app.js usan closure sobre variables de scope
 *   de archivo (messagesEl, input, sessionId, etc.). Durante la transición
 *   este módulo re-exporta las implementaciones actuales de window.* (que
 *   app.js expone después de ejecutarse). Una futura fase puede mover las
 *   implementaciones acá e inyectar las dependencias explícitamente.
 *
 * Globals expuestos por app.js que este módulo consume:
 *   window.refreshSessions()
 *   window.loadSession(sid)
 *   window.triggerNewChat()
 *   window.openSidebarMobile()
 *   window.closeSidebarMobile()
 */

// ── Constantes ────────────────────────────────────────────────────────────
export const SIDEBAR_COLLAPSED_KEY = "obsidian-rag:sidebar-collapsed";

// ── Helpers de acceso al DOM (lazy para que funcionen post-DOMContentLoaded) ──

function _sidebar()           { return document.getElementById("sidebar"); }
function _sidebarOpenBtn()    { return document.getElementById("sidebar-open-btn"); }
function _sidebarCollapseBtn(){ return document.getElementById("sidebar-collapse-btn"); }
function _sessionsList()      { return document.getElementById("sessions-list"); }
function _sessionsSearch()    { return document.getElementById("sessions-search"); }

// ── Re-exports desde window (transición progresiva) ───────────────────────
// app.js ejecuta el código de inicialización de sidebar al evaluarse y
// expone estas funciones en window.* para que HTML onclick y otros módulos
// puedan llamarlas. Este módulo las re-exporta con tipado.

/**
 * Carga y renderiza la lista de sesiones desde /api/sessions.
 */
export async function refreshSessions() {
  if (typeof window.refreshSessions === "function") return window.refreshSessions();
}

/**
 * Carga los turns de una sesión y los renderiza en #messages.
 * @param {string} sid
 */
export async function loadSession(sid) {
  if (typeof window.loadSession === "function") return window.loadSession(sid);
}

/**
 * Inicia una sesión nueva (abort in-flight + limpiar DOM + refresh sidebar).
 */
export function triggerNewChat() {
  if (typeof window.triggerNewChat === "function") return window.triggerNewChat();
}

/**
 * Abre el drawer mobile del sidebar.
 */
export function openSidebarMobile() {
  if (typeof window.openSidebarMobile === "function") return window.openSidebarMobile();
}

/**
 * Cierra el drawer mobile del sidebar.
 */
export function closeSidebarMobile() {
  if (typeof window.closeSidebarMobile === "function") return window.closeSidebarMobile();
}

/**
 * Aplica el estado collapsed/expanded al sidebar.
 * @param {boolean} collapsed
 */
export function applySidebarCollapsed(collapsed) {
  if (typeof window.applySidebarCollapsed === "function") return window.applySidebarCollapsed(collapsed);
  // Fallback inline: toggle data-state.
  const sidebar = _sidebar();
  if (!sidebar) return;
  sidebar.setAttribute("data-state", collapsed ? "collapsed" : "expanded");
  const btn = _sidebarCollapseBtn();
  if (btn) {
    btn.setAttribute("aria-pressed", collapsed ? "true" : "false");
    btn.setAttribute("aria-label", collapsed ? "Expandir sidebar" : "Colapsar sidebar");
  }
}

/**
 * Lee el valor del input de búsqueda de sesiones.
 * @returns {string}
 */
export function getFilterText() {
  if (typeof window.getFilterText === "function") return window.getFilterText();
  const el = _sessionsSearch();
  return el ? (el.value || "").trim().toLowerCase() : "";
}

/**
 * Renderiza la lista de sesiones filtrada en el DOM.
 * Delega a la implementación de app.js.
 * @param {string} filter
 */
export function renderSessions(filter) {
  if (typeof window.renderSessions === "function") return window.renderSessions(filter);
}

/**
 * Formatea la metadata de una sesión para el item de la lista.
 * @param {{turns: number, updated_at: string}} s
 * @returns {string}
 */
export function formatSessionMeta(s) {
  if (typeof window.formatSessionMeta === "function") return window.formatSessionMeta(s);
  // Fallback: "N turns · YYYY-MM-DD HH:MM"
  const bits = [];
  if (Number.isFinite(s.turns)) bits.push(`${s.turns} turn${s.turns === 1 ? "" : "s"}`);
  if (s.updated_at) bits.push(String(s.updated_at).slice(0, 16).replace("T", " "));
  return bits.join(" · ");
}

// ── Init functions (wrappers sobre los init de app.js) ────────────────────

/**
 * Inicializa el collapse/expand del sidebar (desktop).
 * Llamado automáticamente por app.js al boot si sidebar existe.
 */
export function initSidebarCollapse() {
  if (typeof window.initSidebarCollapse === "function") return window.initSidebarCollapse();
}

/**
 * Inicializa el drawer mobile (open/close/backdrop/Esc).
 */
export function initSidebarMobile() {
  if (typeof window.initSidebarMobile === "function") return window.initSidebarMobile();
}

/**
 * Wiring de los botones "new chat" (sidebar + mobile header).
 */
export function initNewChatButtons() {
  if (typeof window.initNewChatButtons === "function") return window.initNewChatButtons();
}

/**
 * Inicializa el input de búsqueda + botón refresh.
 */
export function initSessionsSearch() {
  if (typeof window.initSessionsSearch === "function") return window.initSessionsSearch();
}

/**
 * Keyboard shortcut ⌘\ / Ctrl+\ para toggle sidebar.
 */
export function initSidebarShortcut() {
  if (typeof window.initSidebarShortcut === "function") return window.initSidebarShortcut();
}
