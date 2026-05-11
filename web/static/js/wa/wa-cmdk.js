// Command palette `Cmd+K` (or `Ctrl+K`) — overlay full-screen con
// quick search global + quick actions. Compone con:
//  - /api/wa/chats?q=  (matchea contactos por label)
//  - /api/wa/search?q=&mode=semantic (semántico sobre msgs)
//  - Acciones locales (pin/unpin, navegar, mark all read).
//
// Diseño: overlay con backdrop blur, input grande arriba, lista de
// resultados/acciones abajo. Navegación con ↑/↓ + Enter para confirmar.
// Esc o click fuera cierra.

import { pinChat, unpinChat } from "./wa-api.js";

let overlayEl = null;
let inputEl = null;
let listEl = null;
let timer = null;
let activeIdx = 0;
let items = [];
let openCallback = null;

const ACTIONS = [
  { id: "go-mirror", label: "Ir a Mirror", hint: "/mirror", action: () => location.href = "/mirror" },
  { id: "go-chat", label: "Ir a Chat (rag)", hint: "/chat", action: () => location.href = "/chat" },
  { id: "go-dashboard", label: "Ir a Dashboard", hint: "/dashboard", action: () => location.href = "/dashboard" },
  { id: "go-finance", label: "Ir a Finance", hint: "/finance", action: () => location.href = "/finance" },
  { id: "go-atlas", label: "Ir a Atlas", hint: "/atlas", action: () => location.href = "/atlas" },
  { id: "toggle-theme", label: "Cambiar tema (oscuro / claro)", hint: "T", action: () => {
    const btn = document.getElementById("wa-theme-toggle");
    if (btn) btn.click();
  }},
];

export function init({ onChatSelect }) {
  openCallback = onChatSelect;
  document.addEventListener("keydown", onGlobalKeydown);
}

function onGlobalKeydown(ev) {
  // Cmd+K (Mac) / Ctrl+K (Linux/Win) → toggle palette.
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
    ev.preventDefault();
    open();
    return;
  }
  if (!overlayEl) return;
  if (ev.key === "Escape") {
    ev.preventDefault();
    close();
    return;
  }
  if (ev.key === "ArrowDown") {
    ev.preventDefault();
    setActive(Math.min(activeIdx + 1, items.length - 1));
    return;
  }
  if (ev.key === "ArrowUp") {
    ev.preventDefault();
    setActive(Math.max(activeIdx - 1, 0));
    return;
  }
  if (ev.key === "Enter") {
    ev.preventDefault();
    const it = items[activeIdx];
    if (it) executeItem(it);
    return;
  }
}

function open() {
  if (overlayEl) {
    inputEl.focus();
    inputEl.select();
    return;
  }
  overlayEl = document.createElement("div");
  overlayEl.className = "wa-cmdk-overlay";
  overlayEl.innerHTML = `
    <div class="wa-cmdk-modal" role="dialog" aria-label="Command palette">
      <div class="wa-cmdk-input-row">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="wa-cmdk-icon" aria-hidden="true">
          <circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <input type="text" id="wa-cmdk-input" placeholder="Buscar chat, mensaje, acción…" autocomplete="off" autocorrect="off" spellcheck="false">
        <kbd class="wa-cmdk-esc">esc</kbd>
      </div>
      <ul class="wa-cmdk-list" id="wa-cmdk-list" aria-live="polite"></ul>
      <div class="wa-cmdk-hint">
        <span><kbd>↑↓</kbd> navegar</span>
        <span><kbd>↵</kbd> abrir</span>
        <span><kbd>esc</kbd> cerrar</span>
      </div>
    </div>
  `;
  document.body.appendChild(overlayEl);
  inputEl = overlayEl.querySelector("#wa-cmdk-input");
  listEl = overlayEl.querySelector("#wa-cmdk-list");
  overlayEl.addEventListener("click", (ev) => {
    if (ev.target === overlayEl) close();
  });
  inputEl.addEventListener("input", onInput);
  inputEl.focus();
  // Estado inicial: muestra acciones (sin query).
  renderResults("");
}

function close() {
  if (!overlayEl) return;
  overlayEl.remove();
  overlayEl = null;
  inputEl = null;
  listEl = null;
  items = [];
  activeIdx = 0;
}

function onInput() {
  const q = inputEl.value.trim();
  clearTimeout(timer);
  timer = setTimeout(() => renderResults(q), 140);
}

async function renderResults(q) {
  if (!listEl) return;
  if (q.length === 0) {
    items = ACTIONS.map((a) => ({ ...a, kind: "action" }));
    activeIdx = 0;
    paint();
    return;
  }
  if (q.length < 2) {
    items = [];
    activeIdx = 0;
    paint();
    return;
  }
  // 3 fetchs en paralelo: chats + msgs FTS + actions filtradas.
  let chats = [];
  let msgs = [];
  try {
    const [r1, r2] = await Promise.all([
      fetch(`/api/wa/chats?q=${encodeURIComponent(q)}&limit=10`, { credentials: "same-origin" }),
      fetch(`/api/wa/search?q=${encodeURIComponent(q)}&limit=20&mode=fts`, { credentials: "same-origin" }),
    ]);
    if (r1.ok) {
      const d = await r1.json();
      chats = d.chats || [];
    }
    if (r2.ok) {
      const d = await r2.json();
      msgs = d.hits || [];
    }
  } catch (e) {
    console.warn("[wa-cmdk] fetch failed", e);
  }
  const filteredActions = ACTIONS.filter(
    (a) => a.label.toLowerCase().includes(q.toLowerCase()),
  );
  items = [
    ...filteredActions.map((a) => ({ ...a, kind: "action" })),
    ...chats.map((c) => ({ kind: "chat", id: c.jid, label: c.label, hint: c.last_preview || "", chat: c })),
    ...msgs.map((m) => ({
      kind: "msg",
      id: m.id,
      label: stripMarks(m.snippet || ""),
      hint: m.chat_name || m.chat_jid,
      msg: m,
    })),
  ];
  activeIdx = 0;
  paint();
}

function paint() {
  if (!listEl) return;
  listEl.innerHTML = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "wa-cmdk-empty";
    li.textContent = "sin resultados";
    listEl.appendChild(li);
    return;
  }
  // Headers entre secciones (action/chat/msg).
  let lastKind = null;
  items.forEach((it, i) => {
    if (it.kind !== lastKind) {
      const hdr = document.createElement("li");
      hdr.className = "wa-cmdk-section";
      hdr.textContent = (
        it.kind === "action" ? "Acciones"
        : it.kind === "chat" ? "Contactos"
        : "Mensajes"
      );
      listEl.appendChild(hdr);
      lastKind = it.kind;
    }
    const li = document.createElement("li");
    li.className = "wa-cmdk-item" + (i === activeIdx ? " active" : "");
    li.dataset.idx = String(i);
    li.innerHTML = `
      <div class="wa-cmdk-item-label">${escapeHtml(it.label || "")}</div>
      ${it.hint ? `<div class="wa-cmdk-item-hint">${escapeHtml(it.hint)}</div>` : ""}
    `;
    li.addEventListener("mouseenter", () => setActive(i));
    li.addEventListener("click", () => executeItem(it));
    listEl.appendChild(li);
  });
  // Scroll al active.
  const active = listEl.querySelector(".wa-cmdk-item.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function setActive(idx) {
  activeIdx = idx;
  // Re-paint sin re-fetch.
  paint();
}

function executeItem(it) {
  if (!it) return;
  if (it.kind === "action") {
    close();
    it.action();
    return;
  }
  if (it.kind === "chat") {
    close();
    if (openCallback) openCallback(it.id, null);
    return;
  }
  if (it.kind === "msg") {
    close();
    if (openCallback) openCallback(it.msg.chat_jid, it.msg.id);
    return;
  }
}

function stripMarks(s) {
  return String(s || "").replace(/<\/?mark>/g, "");
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export { open, close };
