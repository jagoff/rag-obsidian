// Long-press picker + revoke menu para mensajes en el thread.
//
// Comportamiento:
// - mousedown / touchstart sobre `.wa-msg` arranca timer 450ms (mobile).
// - Si el user suelta antes → no abre menu.
// - Si excede el timer → abre picker flotante con 8 emojis comunes +
//   "más…" (input nativo) y, si la burbuja es own, opción "Eliminar".
// - Right-click (contextmenu) sobre cualquier `.wa-msg` también abre
//   el menu (atajo desktop, no requiere mantener apretado).
// - Botón kebab visible al hover sobre own bubbles abre el menu (más
//   descubrible que el long-press para usuarios desktop).
// - Click fuera del picker lo cierra.

import { react, revoke } from "./wa-api.js";

const PRESET = ["❤️", "👍", "😂", "😮", "😢", "🙏", "🔥", "👏"];
const LONGPRESS_MS = 450;

let pressTimer = null;
let menuEl = null;
let activeMsgEl = null;
let suppressClickUntil = 0;

let currentJID = null;

export function setActiveJID(jid) {
  currentJID = jid;
  closeMenu();
}

export function attach(bodyEl) {
  if (!bodyEl) return;

  const startPress = (ev) => {
    const target = ev.target.closest && ev.target.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    // El click sobre el kebab tiene su propio handler; no arrancamos
    // long-press desde ahí.
    if (ev.target.closest && ev.target.closest(".wa-msg-kebab")) return;
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

  // Right-click (contextmenu) — atajo desktop al mismo menu, evita
  // tener que mantener apretado 450ms.
  bodyEl.addEventListener("contextmenu", (ev) => {
    const target = ev.target.closest && ev.target.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    ev.preventDefault();
    activeMsgEl = target;
    openMenu(target, ev);
  });

  // Click sobre el kebab `⋮` (visible al hover en burbujas own) →
  // abre el mismo menu. Delegated para que sirva con messages
  // renderizados después del attach.
  bodyEl.addEventListener("click", (ev) => {
    const kebab = ev.target.closest && ev.target.closest(".wa-msg-kebab");
    if (!kebab) return;
    ev.preventDefault();
    ev.stopPropagation();
    const target = kebab.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    activeMsgEl = target;
    openMenu(target, ev);
  }, true);

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

// Inserta el kebab `⋮` en burbujas propias. El renderer (`wa-thread.js`)
// lo llama después de armar el `.wa-msg.own`. Se separa para que la
// lógica de "qué affordances tiene cada msg" viva acá y no se duplique
// en el renderer.
export function attachOwnMenuAffordance(msgEl) {
  if (!msgEl || !msgEl.classList || !msgEl.classList.contains("own")) return;
  if (msgEl.classList.contains("revoked")) return;
  if (msgEl.querySelector(".wa-msg-kebab")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "wa-msg-kebab";
  btn.setAttribute("aria-label", "Acciones del mensaje");
  btn.title = "Acciones (reaccionar, eliminar)";
  btn.textContent = "⋮";
  msgEl.appendChild(btn);
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
  // Confirmación destructiva — el delete-for-everyone es irreversible
  // (el bridge persiste un revoke event y los peers reciben el tomb).
  if (!window.confirm("Eliminar este mensaje para todos? No se puede deshacer.")) {
    return;
  }
  try {
    // admin-auth.js inyecta el Bearer del admin_token automáticamente
    // (loopback-only). Si el browser está en LAN/tunnel, el server
    // responde 401 — el catch muestra el detail.
    await revoke(currentJID, messageId);
    // El SSE va a marcar revoked en breve; pero para feedback instant
    // marcamos optimistic.
    const msgEl = document.querySelector(`.wa-msg[data-id="${cssEscape(messageId)}"]`);
    if (msgEl) {
      msgEl.classList.add("revoked");
      msgEl.innerHTML = "🚫 Este mensaje fue eliminado";
    }
  } catch (e) {
    console.error("[wa-reactions] revoke failed", e);
    const msg = /401/.test(e.message)
      ? "No se pudo eliminar: este device no tiene admin token (solo desde localhost / ra.ai)."
      : `No se pudo eliminar: ${e.message}`;
    window.alert(msg);
  }
}

function cssEscape(s) {
  return String(s).replace(/"/g, '\\"');
}
