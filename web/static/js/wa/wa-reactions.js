// Long-press picker + revoke menu para mensajes en el thread.
//
// Comportamiento:
// - mousedown / touchstart sobre `.wa-msg` arranca timer 450ms.
// - Si el user suelta antes → no abre menu.
// - Si excede el timer → abre picker flotante con 8 emojis comunes +
//   "más…" (input nativo) y, si la burbuja es own, opción "Eliminar".
// - Click fuera del picker lo cierra.

import { react, revoke } from "./wa-api.js";

const PRESET = ["❤️", "👍", "😂", "😮", "😢", "🙏", "🔥", "👏"];
const LONGPRESS_MS = 450;

let pressTimer = null;
let menuEl = null;
let activeMsgEl = null;
let suppressClickUntil = 0;

let currentJID = null;

function getAdminToken() {
  // El repo ya guarda admin-token en localStorage["admin_token"] por
  // patrón existente de otros endpoints destructive.
  return localStorage.getItem("admin_token") || "";
}

export function setActiveJID(jid) {
  currentJID = jid;
  closeMenu();
}

export function attach(bodyEl) {
  if (!bodyEl) return;

  const startPress = (ev) => {
    const target = ev.target.closest && ev.target.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    activeMsgEl = target;
    clearTimeout(pressTimer);
    pressTimer = setTimeout(() => {
      openMenu(target, ev);
    }, LONGPRESS_MS);
  };
  const cancelPress = () => {
    clearTimeout(pressTimer);
  };
  bodyEl.addEventListener("mousedown", startPress);
  bodyEl.addEventListener("touchstart", startPress, { passive: true });
  bodyEl.addEventListener("mouseup", cancelPress);
  bodyEl.addEventListener("mouseleave", cancelPress);
  bodyEl.addEventListener("touchend", cancelPress);
  bodyEl.addEventListener("touchcancel", cancelPress);

  // Suprimir el siguiente click si el user soltó después de abrir el menu.
  bodyEl.addEventListener("click", (ev) => {
    if (Date.now() < suppressClickUntil) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  }, true);

  document.addEventListener("click", (ev) => {
    if (menuEl && !menuEl.contains(ev.target)) {
      closeMenu();
    }
  });
}

function openMenu(msgEl, originEv) {
  closeMenu();
  if (!currentJID) return;
  activeMsgEl = msgEl;
  suppressClickUntil = Date.now() + 300;

  menuEl = document.createElement("div");
  menuEl.className = "wa-reaction-menu";
  const presetEls = PRESET.map((emoji) => {
    const b = document.createElement("button");
    b.className = "wa-reaction-emoji";
    b.textContent = emoji;
    b.title = `Reaccionar con ${emoji}`;
    b.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      sendReaction(emoji);
    });
    return b;
  });
  const more = document.createElement("button");
  more.className = "wa-reaction-more";
  more.textContent = "…";
  more.title = "Otro emoji";
  more.addEventListener("click", (e) => {
    e.preventDefault();
    const custom = (window.prompt("Emoji a usar:") || "").trim();
    if (custom) sendReaction(custom);
  });

  for (const el of presetEls) menuEl.appendChild(el);
  menuEl.appendChild(more);

  const isOwn = msgEl.classList.contains("own");
  if (isOwn) {
    const sep = document.createElement("div");
    sep.className = "wa-reaction-sep";
    menuEl.appendChild(sep);
    const del = document.createElement("button");
    del.className = "wa-reaction-delete";
    del.textContent = "Eliminar para todos";
    del.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      doRevoke();
    });
    menuEl.appendChild(del);
  }

  // Posicionar arriba de la burbuja (o abajo si estamos cerca del top).
  document.body.appendChild(menuEl);
  const r = msgEl.getBoundingClientRect();
  const mr = menuEl.getBoundingClientRect();
  let top = r.top - mr.height - 6;
  if (top < 8) top = r.bottom + 6;
  let left = r.left;
  if (left + mr.width > window.innerWidth - 8) {
    left = window.innerWidth - mr.width - 8;
  }
  menuEl.style.top = `${Math.max(8, top)}px`;
  menuEl.style.left = `${Math.max(8, left)}px`;
}

function closeMenu() {
  if (menuEl) {
    menuEl.remove();
    menuEl = null;
  }
  activeMsgEl = null;
}

async function sendReaction(emoji) {
  if (!activeMsgEl || !currentJID) return closeMenu();
  const messageId = activeMsgEl.dataset.id || "";
  const isOwn = activeMsgEl.classList.contains("own");
  const sender = activeMsgEl.dataset.sender || "";
  closeMenu();
  if (!messageId) return;
  // Optimistic UI — agregamos el reaction inmediatamente.
  let rEl = activeMsgEl.querySelector(".wa-msg-reactions");
  if (!rEl) {
    rEl = document.createElement("div");
    rEl.className = "wa-msg-reactions";
    activeMsgEl.appendChild(rEl);
  }
  rEl.textContent = emoji;
  try {
    await react(currentJID, messageId, sender, isOwn, emoji);
  } catch (e) {
    console.error("[wa-reactions] react failed", e);
    rEl.textContent = "⚠";
  }
}

async function doRevoke() {
  if (!activeMsgEl || !currentJID) return closeMenu();
  const messageId = activeMsgEl.dataset.id || "";
  closeMenu();
  if (!messageId) return;
  const tok = getAdminToken();
  if (!tok) {
    window.alert("Necesitás el admin token guardado en localStorage[\"admin_token\"] para eliminar mensajes. Mirá ~/.config/obsidian-rag/admin_token.txt.");
    return;
  }
  try {
    await revoke(currentJID, messageId, tok);
    // El SSE va a marcar revoked en breve; pero para feedback instant
    // marcamos optimistic.
    const msgEl = document.querySelector(`.wa-msg[data-id="${cssEscape(messageId)}"]`);
    if (msgEl) {
      msgEl.classList.add("revoked");
      msgEl.innerHTML = "🚫 Este mensaje fue eliminado";
    }
  } catch (e) {
    console.error("[wa-reactions] revoke failed", e);
    window.alert(`No se pudo eliminar: ${e.message}`);
  }
}

function cssEscape(s) {
  return String(s).replace(/"/g, '\\"');
}
