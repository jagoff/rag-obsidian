// Render del sidebar — lista de chats con avatar/nombre/preview/unread.
// Click → notifica al entry point para abrir el thread correspondiente.

import { fetchChats, pinChat, unpinChat } from "./wa-api.js";
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
    li.className = "wa-chat-item" + (c.pinned ? " pinned" : "");
    li.dataset.jid = c.jid;
    if (c.jid === activeJID) li.classList.add("active");
    li.addEventListener("click", (ev) => {
      // No abrir el chat si el click fue sobre el pin button.
      if (ev.target.closest(".wa-chat-pin")) return;
      selectChat(c.jid);
    });

    // Pin aplica a cualquier chat (contacto individual o grupo).
    const canPin = true;
    const pinTitle = c.pinned ? "Despinear" : "Pinear al tope";
    const pinIcon = c.pinned
      // Pinned: tachuela rellena (SVG filled).
      ? `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M16 3l5 5-4 1-3 7-2-2-4 4-1-1 4-4-2-2 7-3z"/></svg>`
      // Unpinned: outline.
      : `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 3l5 5-4 1-3 7-2-2-4 4-1-1 4-4-2-2 7-3z"/></svg>`;
    const pinBtnHTML = canPin
      ? `<button class="wa-chat-pin" type="button" title="${pinTitle}" aria-label="${pinTitle}">${pinIcon}</button>`
      : "";

    li.innerHTML = `
      ${pinBtnHTML}
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
    const pinBtn = li.querySelector(".wa-chat-pin");
    if (pinBtn) {
      pinBtn.addEventListener("click", async (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        // Spring pop feedback inmediato (no espera al server).
        pinBtn.classList.remove("popping");
        // force reflow para reiniciar la animación si se hace click rápido
        void pinBtn.offsetWidth;
        pinBtn.classList.add("popping");
        try {
          const next = !c.pinned;
          if (next) {
            await pinChat(c.jid);
          } else {
            await unpinChat(c.jid);
          }
          c.pinned = next;
          c.pinned_ts = next ? new Date().toISOString() : "";
          // FLIP reorder: snapshot positions → render → animate diff.
          renderWithFlip();
        } catch (e) {
          console.error("[wa-chatlist] pin toggle failed", e);
        }
      });
    }
    els.list.appendChild(li);
  }
}

function resortChats() {
  allChats.sort((a, b) => {
    if (a.pinned && !b.pinned) return -1;
    if (!a.pinned && b.pinned) return 1;
    const tsA = a.pinned ? (a.pinned_ts || "") : (a.last_ts || "");
    const tsB = b.pinned ? (b.pinned_ts || "") : (b.last_ts || "");
    return tsB.localeCompare(tsA);
  });
}

// FLIP animation: First → Last → Invert → Play.
// 1. Snapshot del rect de cada chat-item antes de re-render.
// 2. Re-sort + re-render (positions cambian).
// 3. Para cada item con jid conocido, computamos delta old→new y
//    aplicamos `transform: translateY(dy)` con transition: none.
// 4. requestAnimationFrame → transition + transform: '' → animan a la
//    posición real.
// Esto da el efecto de "los chats se deslizan" cuando pineás/despineás
// o cuando un msg nuevo levanta un chat al tope.
function renderWithFlip() {
  if (!els.list) {
    resortChats();
    render();
    return;
  }
  // FIRST: capturar rects pre-render por JID.
  const firstRects = new Map();
  for (const li of els.list.querySelectorAll(".wa-chat-item")) {
    const jid = li.dataset.jid;
    if (jid) firstRects.set(jid, li.getBoundingClientRect());
  }
  // LAST: re-sort + re-render.
  resortChats();
  render();
  // INVERT + PLAY: aplicar transform delta + transition.
  requestAnimationFrame(() => {
    for (const li of els.list.querySelectorAll(".wa-chat-item")) {
      const jid = li.dataset.jid;
      if (!jid) continue;
      const first = firstRects.get(jid);
      if (!first) continue; // chat nuevo
      const last = li.getBoundingClientRect();
      const dy = first.top - last.top;
      if (Math.abs(dy) < 1) continue;
      li.style.transform = `translateY(${dy}px)`;
      li.style.transition = "none";
    }
    // Next frame: clear transform con transition habilitada.
    requestAnimationFrame(() => {
      for (const li of els.list.querySelectorAll(".wa-chat-item")) {
        if (!li.style.transform) continue;
        li.classList.add("wa-flipping");
        li.style.transition = "";
        li.style.transform = "";
        const cleanup = () => {
          li.classList.remove("wa-flipping");
          li.removeEventListener("transitionend", cleanup);
        };
        li.addEventListener("transitionend", cleanup);
      }
    });
  });
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
  // Reordenar respetando pinned: los pinned siempre quedan arriba
  // ordenados por pinned_ts; los demás por last_ts. Sin esto, un msg
  // nuevo a un chat unpinned lo movía arriba de los pinned.
  allChats.splice(idx, 1);
  allChats.push(c);
  renderWithFlip();
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
