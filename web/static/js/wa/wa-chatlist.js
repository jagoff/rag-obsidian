// Render del sidebar — lista de chats con avatar/nombre/preview/unread.
// Click → notifica al entry point para abrir el thread correspondiente.

import { fetchChats } from "./wa-api.js";
import { colorFor, renderInto as renderAvatar } from "./wa-avatars.js";
import * as search from "./wa-search.js";

const els = {
  list: null,
  loading: null,
  search: null,
};

let allChats = [];
let activeJID = null;
let onSelectCallback = null;
let searchTimer = null;
let lastSearchQuery = "";

export function init({ listEl, loadingEl, searchEl, onSelect }) {
  els.list = listEl;
  els.loading = loadingEl;
  els.search = searchEl;
  onSelectCallback = onSelect;

  if (els.search) {
    els.search.addEventListener("input", onSearchInput);
  }

  search.init({
    inputEl: searchEl,
    listEl,
    loadingEl,
    onSelect: (jid /* , messageId */) => {
      // Volvemos a modo chats + abrimos el thread.
      search.exitSearchMode();
      if (els.search) els.search.value = "";
      load();
      if (onSelectCallback) onSelectCallback(jid);
    },
  });
}

export async function load() {
  if (els.loading) els.loading.classList.remove("hidden");
  try {
    const data = await fetchChats({ limit: 80, q: lastSearchQuery || null });
    allChats = data.chats || [];
    render();
    updateStats();
  } catch (e) {
    console.error("[wa-chatlist] load failed", e);
    if (els.list) els.list.innerHTML = `<li class="wa-empty-state">Error cargando chats: ${e.message}</li>`;
  } finally {
    if (els.loading) els.loading.classList.add("hidden");
  }
}

function updateStats() {
  const total = allChats.length;
  const unread = allChats.reduce((acc, c) => acc + (c.unread_count || 0), 0);
  const chatsEl = document.getElementById("wa-stat-chats");
  const unreadEl = document.getElementById("wa-stat-unread");
  if (chatsEl) chatsEl.textContent = total > 999 ? "999+" : String(total);
  if (unreadEl) unreadEl.textContent = unread > 999 ? "999+" : String(unread);
}

function onSearchInput() {
  const q = (els.search.value || "").trim();
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    // Si la query tiene >=3 chars, search.js intercepta y muestra hits
    // del FTS5; sino, filter local sobre chats.
    if (search.maybeHandleSearch(q)) return;
    lastSearchQuery = q;
    load();
  }, 200);
}

function timeLabel(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
  }
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return "ayer";
  return d.toLocaleDateString("es-AR", { day: "2-digit", month: "2-digit" });
}

function render() {
  if (!els.list) return;
  els.list.innerHTML = "";
  for (const c of allChats) {
    const li = document.createElement("li");
    li.className = "wa-chat-item";
    li.dataset.jid = c.jid;
    if (c.jid === activeJID) li.classList.add("active");
    li.addEventListener("click", () => selectChat(c.jid));

    li.innerHTML = `
      <div class="wa-chat-avatar" data-jid="${escapeHtml(c.jid)}"></div>
      <div class="wa-chat-body">
        <div class="wa-chat-name-row">
          <div class="wa-chat-name">${escapeHtml(c.label)}</div>
          <div class="wa-chat-time">${escapeHtml(timeLabel(c.last_ts))}</div>
        </div>
        <div class="wa-chat-preview-row">
          <div class="wa-chat-preview">${c.last_from_me ? "Yo: " : ""}${escapeHtml(c.last_preview)}</div>
          <div class="wa-chat-unread ${c.unread_count > 0 ? "" : "zero"}">${c.unread_count > 99 ? "99+" : c.unread_count}</div>
        </div>
      </div>
    `;
    const avatarEl = li.querySelector(".wa-chat-avatar");
    renderAvatar(avatarEl, c.jid, c.avatar_initials, c.label);
    els.list.appendChild(li);
  }
}

export function setActive(jid) {
  activeJID = jid;
  if (!els.list) return;
  for (const li of els.list.querySelectorAll(".wa-chat-item")) {
    li.classList.toggle("active", li.dataset.jid === jid);
  }
}

/** SSE callback. Reorder chat al top + update preview + unread badge. */
export function applyChatUpdate(payload) {
  if (!payload || !payload.jid) return;
  const idx = allChats.findIndex((c) => c.jid === payload.jid);
  if (idx === -1) {
    // Chat nuevo (no estaba en la lista cargada) → reload completo.
    load();
    return;
  }
  const c = allChats[idx];
  c.last_ts = payload.last_ts || c.last_ts;
  c.last_preview = payload.last_preview ?? c.last_preview;
  c.last_from_me = !!payload.last_from_me;
  if (payload.jid !== activeJID && payload.unread_delta) {
    c.unread_count = (c.unread_count || 0) + payload.unread_delta;
  }
  // Mover al top.
  allChats.splice(idx, 1);
  allChats.unshift(c);
  render();
  updateStats();
}

function selectChat(jid) {
  setActive(jid);
  if (onSelectCallback) onSelectCallback(jid);
  // Clear unread del item activo de manera optimista — el backend ya
  // se actualizó via `markRead` en el thread loader.
  const item = els.list.querySelector(`.wa-chat-item[data-jid="${jid}"] .wa-chat-unread`);
  if (item) {
    item.textContent = "0";
    item.classList.add("zero");
  }
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
