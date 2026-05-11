// Render del sidebar — lista de chats con avatar/nombre/preview/unread.
// Click → notifica al entry point para abrir el thread correspondiente.

import { fetchChats } from "./wa-api.js";
import { colorFor } from "./wa-avatars.js";

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
}

export async function load() {
  if (els.loading) els.loading.classList.remove("hidden");
  try {
    const data = await fetchChats({ limit: 80, q: lastSearchQuery || null });
    allChats = data.chats || [];
    render();
  } catch (e) {
    console.error("[wa-chatlist] load failed", e);
    if (els.list) els.list.innerHTML = `<li class="wa-empty-state">Error cargando chats: ${e.message}</li>`;
  } finally {
    if (els.loading) els.loading.classList.add("hidden");
  }
}

function onSearchInput() {
  const q = (els.search.value || "").trim();
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
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

    const avatarColor = colorFor(c.jid);
    li.innerHTML = `
      <div class="wa-chat-avatar" style="background:${avatarColor}">
        <span>${escapeHtml(c.avatar_initials || "?")}</span>
      </div>
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
