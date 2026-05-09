// session.mjs — Gestión de sesión de chat: ID, historial de queries,
// persistencia en sessionStorage/localStorage.

export const SESSION_KEY = "obsidian-rag:session";
export const HISTORY_KEY = "obsidian-rag:history";
export const HISTORY_CAP = 100;

// sessionId vive en sessionStorage (per-tab). localStorage legado se limpia
// al boot para evitar que una tab nueva herede stale context.
let sessionId = sessionStorage.getItem(SESSION_KEY) || null;
try { localStorage.removeItem(SESSION_KEY); } catch {}

export function getSessionId() { return sessionId; }

export function setSessionId(id) {
  sessionId = id;
  if (id) {
    sessionStorage.setItem(SESSION_KEY, id);
  } else {
    sessionStorage.removeItem(SESSION_KEY);
    try { localStorage.removeItem(SESSION_KEY); } catch {}
  }
}

// ── Historial de queries (flecha arriba) ───────────────────────────────

export function loadHistory() {
  try {
    const arr = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(arr) ? arr.filter((s) => typeof s === "string") : [];
  } catch { return []; }
}

let history = loadHistory();
let historyIdx = history.length;
let historyDraft = "";

export function getHistory() { return history; }
export function getHistoryIdx() { return historyIdx; }
export function setHistoryIdx(i) { historyIdx = i; }
export function getHistoryDraft() { return historyDraft; }
export function setHistoryDraft(d) { historyDraft = d; }

export function pushHistory(q) {
  q = q.trim();
  if (!q) return;
  if (history[history.length - 1] !== q) {
    history.push(q);
    if (history.length > HISTORY_CAP) history.splice(0, history.length - HISTORY_CAP);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
  }
  historyIdx = history.length;
  historyDraft = "";
}

/** Carga historial desde /api/history (server-authoritative). */
export async function loadServerHistory() {
  try {
    const res = await fetch("/api/history?limit=200");
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data.history) || !data.history.length) return;
    history = data.history;
    historyIdx = history.length;
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
  } catch {}
}
