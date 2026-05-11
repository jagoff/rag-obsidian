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
  // Switch de chat cancela el modo selección (evitar revoke crosss-chat
  // por error si el user cambió de pestaña sin terminar la operación).
  if (_selecting) exitSelectionMode();
}

export function attach(bodyEl) {
  if (!bodyEl) return;

  const startPress = (ev) => {
    if (_selecting) return;
    const target = ev.target.closest && ev.target.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    // El click sobre el trash tiene su propio handler; no arrancamos
    // long-press desde ahí.
    if (ev.target.closest && ev.target.closest(".wa-msg-trash")) return;
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
    if (_selecting) return;
    const target = ev.target.closest && ev.target.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    ev.preventDefault();
    activeMsgEl = target;
    openMenu(target, ev);
  });

  // Click sobre el ícono trash `🗑` (visible al hover en burbujas own)
  // → directo al delete confirm. NO abre el menu para no superponerse
  // con otros UI elements (feedback user: el menu se cruzaba con
  // day-dividers + reactions, era confuso). Para reacciones o
  // selección múltiple usar long-press / contextmenu / right-click.
  bodyEl.addEventListener("click", (ev) => {
    if (_selecting) return;
    const trash = ev.target.closest && ev.target.closest(".wa-msg-trash");
    if (!trash) return;
    ev.preventDefault();
    ev.stopPropagation();
    const target = trash.closest(".wa-msg");
    if (!target || target.classList.contains("revoked")) return;
    activeMsgEl = target;
    doRevoke();
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

// Inserta el ícono trash 🗑 en burbujas propias. El renderer
// (`wa-thread.js`) lo llama después de armar el `.wa-msg.own`. Click
// va directo al delete confirm — para reacciones / multi-select usar
// long-press / right-click. Visible solo al hover (default opacity 0).
export function attachOwnMenuAffordance(msgEl) {
  if (!msgEl || !msgEl.classList || !msgEl.classList.contains("own")) return;
  if (msgEl.classList.contains("revoked")) return;
  if (msgEl.querySelector(".wa-msg-trash")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "wa-msg-trash";
  btn.setAttribute("aria-label", "Eliminar mensaje");
  btn.title = "Eliminar mensaje";
  // SVG inline para no depender de fuentes con glyph 🗑️ (en macOS el
  // emoji renderiza colorido y desentona con el resto del UI).
  btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
    + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    + 'stroke-linejoin="round" aria-hidden="true">'
    + '<polyline points="3 6 5 6 21 6"></polyline>'
    + '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>'
    + '<path d="M10 11v6"></path><path d="M14 11v6"></path>'
    + '<path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>'
    + '</svg>';
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
    const multi = document.createElement("button");
    multi.className = "wa-reaction-delete";
    multi.textContent = "Seleccionar varios…";
    multi.title = "Marcar mensajes propios y eliminarlos en bulk";
    multi.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      enterSelectionMode(msgEl);
    });
    menuEl.appendChild(multi);
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


// ── Selection mode + bulk delete ─────────────────────────────────────
// Modo "marcar varios y borrar de una". Se activa desde el menu kebab
// con "Seleccionar varios…". Mientras está activo:
//   - `.wa-thread.selecting` agrega visual feedback (checkmark a la
//     izquierda de cada burbuja own).
//   - Click sobre una burbuja `.wa-msg.own` toggle `.selected`.
//   - El long-press / contextmenu / kebab quedan desactivados (el
//     event handler chequea `_selecting`).
//   - Un action bar fijo abajo del thread muestra "N seleccionados ·
//     Eliminar (N) · Cancelar".
// El delete itera `revoke()` (admin-auth.js inyecta el Bearer); en
// error sigue con los siguientes y reporta al final.
let _selecting = false;
let _selectBar = null;
const _selectedIds = new Set();

export function isSelecting() {
  return _selecting;
}

function enterSelectionMode(initialMsgEl) {
  if (_selecting) return;
  _selecting = true;
  _selectedIds.clear();
  closeMenu();
  const thread = document.querySelector(".wa-thread");
  if (thread) thread.classList.add("selecting");
  // Pre-seleccionar la burbuja que abrió el menu.
  if (initialMsgEl && initialMsgEl.classList.contains("own")) {
    const id = initialMsgEl.dataset.id || "";
    if (id) {
      _selectedIds.add(id);
      initialMsgEl.classList.add("selected");
    }
  }
  mountSelectBar();
  updateSelectBar();
  // Listener para click sobre own bubbles. Lo bindeamos al body del
  // thread para sobrevivir re-renders del lazy load.
  const body = document.getElementById("wa-thread-body");
  if (body && !body.__waSelectClickBound) {
    body.addEventListener("click", onSelectClick, true);
    body.__waSelectClickBound = true;
  }
}

function exitSelectionMode() {
  _selecting = false;
  const thread = document.querySelector(".wa-thread");
  if (thread) thread.classList.remove("selecting");
  for (const el of document.querySelectorAll(".wa-msg.selected")) {
    el.classList.remove("selected");
  }
  _selectedIds.clear();
  if (_selectBar) _selectBar.hidden = true;
}

function onSelectClick(ev) {
  if (!_selecting) return;
  const target = ev.target.closest && ev.target.closest(".wa-msg.own");
  if (!target || target.classList.contains("revoked")) return;
  // Evitamos toggles cuando el user clickea sobre links / media / el
  // botón delete del action bar (que vive fuera del thread body).
  if (ev.target.closest(".wa-msg-trash")) return;
  ev.preventDefault();
  ev.stopPropagation();
  const id = target.dataset.id || "";
  if (!id) return;
  if (_selectedIds.has(id)) {
    _selectedIds.delete(id);
    target.classList.remove("selected");
  } else {
    _selectedIds.add(id);
    target.classList.add("selected");
  }
  updateSelectBar();
}

function mountSelectBar() {
  if (_selectBar) {
    _selectBar.hidden = false;
    return;
  }
  const thread = document.querySelector(".wa-thread");
  if (!thread) return;
  const bar = document.createElement("div");
  bar.className = "wa-select-bar";
  bar.innerHTML = `
    <span class="wa-select-count">0 seleccionados</span>
    <button type="button" class="wa-select-cancel">Cancelar</button>
    <button type="button" class="wa-select-delete" disabled>Eliminar (0)</button>
  `;
  // Insertarlo antes del composer para que quede sobre éste.
  const composer = thread.querySelector(".wa-composer");
  if (composer) thread.insertBefore(bar, composer);
  else thread.appendChild(bar);
  bar.querySelector(".wa-select-cancel").addEventListener("click", () => {
    exitSelectionMode();
  });
  bar.querySelector(".wa-select-delete").addEventListener("click", () => {
    bulkRevoke();
  });
  _selectBar = bar;
}

function updateSelectBar() {
  if (!_selectBar) return;
  const n = _selectedIds.size;
  _selectBar.querySelector(".wa-select-count").textContent =
    `${n} seleccionado${n === 1 ? "" : "s"}`;
  const del = _selectBar.querySelector(".wa-select-delete");
  del.textContent = `Eliminar (${n})`;
  del.disabled = n === 0;
}

async function bulkRevoke() {
  if (!currentJID || _selectedIds.size === 0) return;
  const ids = [..._selectedIds];
  if (!window.confirm(
    `Eliminar ${ids.length} mensaje${ids.length === 1 ? "" : "s"} para todos? No se puede deshacer.`,
  )) return;
  const delBtn = _selectBar?.querySelector(".wa-select-delete");
  if (delBtn) {
    delBtn.disabled = true;
    delBtn.textContent = `Eliminando…`;
  }
  let ok = 0;
  const errors = [];
  for (const id of ids) {
    try {
      await revoke(currentJID, id);
      ok += 1;
      const el = document.querySelector(`.wa-msg[data-id="${cssEscape(id)}"]`);
      if (el) {
        el.classList.remove("selected");
        el.classList.add("revoked");
        el.innerHTML = "🚫 Este mensaje fue eliminado";
      }
    } catch (e) {
      errors.push(`${id}: ${e.message}`);
    }
  }
  exitSelectionMode();
  if (errors.length) {
    const detail = errors.slice(0, 3).join("\n");
    const extra = errors.length > 3 ? `\n…y ${errors.length - 3} más` : "";
    const auth = errors.some((e) => /401/.test(e))
      ? "\n(¿este device sin admin token? probá desde localhost / ra.ai)"
      : "";
    window.alert(
      `Eliminados ${ok}/${ids.length}. Fallaron ${errors.length}:\n${detail}${extra}${auth}`,
    );
  }
}
