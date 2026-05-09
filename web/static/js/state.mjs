/**
 * state.mjs — Phase W4-phase-3 (2026-05-09)
 *
 * Store compartido de estado del chat. Centraliza todas las variables
 * top-level que antes vivían en el scope de app.js y eran accedidas
 * por cierre implícito desde decenas de funciones.
 *
 * Filosofía:
 *   - El objeto `state` es mutable directamente desde cualquier módulo
 *     que lo importe. No hay getters/setters reactivos — el DOM se actualiza
 *     en los call sites, no en el store.
 *   - Los accessors con side-effects (`setSessionId`, `setTtsEnabled`, etc.)
 *     existen ÚNICAMENTE cuando la mutación requiere persistencia o una
 *     notificación sincrónica a otro subsistema.
 *   - Las variables de DOM (messagesEl, input, etc.) se exponen en `els`
 *     como lazy-resolvers para que el módulo pueda importarse antes de
 *     DOMContentLoaded sin crashear.
 *
 * Uso:
 *   import { state, els, setSessionId } from "./state.mjs";
 *   state.pending = true;
 *   els.input().value = "";
 */

// ── Claves localStorage / sessionStorage ──────────────────────────────────
export const SESSION_KEY = "obsidian-rag:session";
export const VAULT_KEY   = "obsidian-rag:vault";
export const TTS_KEY     = "obsidian-rag:tts";
export const HISTORY_KEY = "obsidian-rag:history";
export const HISTORY_CAP = 100;
export const CHAT_MODE_KEY = "rag-chat-mode";
export const SIDEBAR_COLLAPSED_KEY = "obsidian-rag:sidebar-collapsed";
export const SCOPE_KEY = "obsidian-rag:scope";
export const VALID_MODES = new Set(["auto", "fast", "deep"]);

// ── Bootstrap helpers (se usan al inicializar el store) ───────────────────

function _loadHistory() {
  try {
    const arr = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(arr) ? arr.filter((s) => typeof s === "string") : [];
  } catch { return []; }
}

function _loadSessionId() {
  // session_id vive en sessionStorage (por-tab), no localStorage.
  // Si había uno guardado en localStorage (versiones viejas) lo borramos.
  try { localStorage.removeItem(SESSION_KEY); } catch {}
  return sessionStorage.getItem(SESSION_KEY) || null;
}

// ── Store principal ────────────────────────────────────────────────────────
export const state = {
  // Sesión + vault
  sessionId:         _loadSessionId(),
  vaultScope:        localStorage.getItem(VAULT_KEY) || "",
  lastTurnId:        null,

  // TTS + audio
  ttsEnabled:        localStorage.getItem(TTS_KEY) === "1",
  currentAudio:      null,

  // Stream control
  pending:           false,
  currentController: null,   // AbortController para /api/chat en curso

  // Side fetches (related, followups, contacts)
  inflightSideFetches: new Set(),

  // Auto-retry
  pendingRetryTimer:     null,
  pendingRetryCountdown: null,

  // Historial de queries (para ↑/↓)
  history:     _loadHistory(),
  historyIdx:  -1,     // -1 = sin historial restaurado; se fija post-load
  historyDraft: "",

  // Historia de la última pregunta del user (para ⌘↑ edit-last)
  lastUserQuestion: "",
};

// Ajustar historyIdx al valor correcto después de cargar
state.historyIdx = state.history.length;

// ── Accessors con side-effects ─────────────────────────────────────────────

/**
 * Persiste sessionId en sessionStorage y actualiza el store.
 * @param {string|null} id
 */
export function setSessionId(id) {
  state.sessionId = id;
  if (id) {
    sessionStorage.setItem(SESSION_KEY, id);
  } else {
    sessionStorage.removeItem(SESSION_KEY);
    try { localStorage.removeItem(SESSION_KEY); } catch {}
  }
}

/**
 * Persiste vaultScope en localStorage y actualiza el store.
 * @param {string} v — "" = vault activo, "all" = todos, nombre = vault específico
 */
export function setVaultScope(v) {
  state.vaultScope = v;
  if (v) {
    localStorage.setItem(VAULT_KEY, v);
  } else {
    localStorage.removeItem(VAULT_KEY);
  }
}

/**
 * Persiste ttsEnabled en localStorage y actualiza el store.
 * @param {boolean} enabled
 */
export function setTtsEnabled(enabled) {
  state.ttsEnabled = enabled;
  localStorage.setItem(TTS_KEY, enabled ? "1" : "0");
  // Si se desactiva y hay audio en vuelo, lo pausamos.
  if (!enabled && state.currentAudio) {
    try { state.currentAudio.pause(); } catch (_) {}
    state.currentAudio = null;
  }
}

/**
 * Pushea una query al historial y persiste en localStorage.
 * @param {string} q
 */
export function pushHistory(q) {
  q = q.trim();
  if (!q) return;
  if (state.history[state.history.length - 1] !== q) {
    state.history.push(q);
    if (state.history.length > HISTORY_CAP) {
      state.history.splice(0, state.history.length - HISTORY_CAP);
    }
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history)); } catch {}
  }
  state.historyIdx  = state.history.length;
  state.historyDraft = "";
}

// ── DOM element accessors (lazy — safe antes de DOMContentLoaded) ──────────

/**
 * Accessors lazy para los elementos DOM principales del chat.
 * Se usan con `els.input()` etc. para garantizar que no rompan
 * si el módulo se importa antes del parseo del HTML.
 */
export const els = {
  messages:       () => document.getElementById("messages"),
  form:           () => document.getElementById("composer"),
  input:          () => document.getElementById("input"),
  vaultPicker:    () => document.getElementById("vault-picker"),
  chatModeToggle: () => document.getElementById("chat-mode-toggle"),
  ttsToggle:      () => document.getElementById("tts-toggle"),
  helpBtn:        () => document.getElementById("help-btn"),
  helpModal:      () => document.getElementById("help-modal"),
  stopBtn:        () => document.getElementById("stop-btn"),
  sendBtn:        () => document.getElementById("send-btn"),
  menuBtn:        () => document.getElementById("menu-btn"),
  menuSheet:      () => document.getElementById("menu-sheet"),
  sheetVaultPicker: () => document.getElementById("sheet-vault-picker"),
  sheetTtsToggle: () => document.getElementById("sheet-tts-toggle"),
  sidebar:        () => document.getElementById("sidebar"),
  sessionsList:   () => document.getElementById("sessions-list"),
  sessionsSearch: () => document.getElementById("sessions-search"),
};
