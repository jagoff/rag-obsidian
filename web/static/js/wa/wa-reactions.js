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

import { react, revoke, hide, translate } from "./wa-api.js";

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
    if (ev.target.closest && (ev.target.closest(".wa-msg-trash") || ev.target.closest(".wa-msg-select"))) return;
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
  // → directo al delete (sin confirm). El círculo `◯` justo abajo
  // entra al modo selección. Ambos visibles solo al hover (default
  // opacity 0) y posicionados fuera de la burbuja para no chocar con
  // contenido / day-dividers / reactions.
  bodyEl.addEventListener("click", (ev) => {
    if (_selecting) return;
    const trash = ev.target.closest && ev.target.closest(".wa-msg-trash");
    if (trash) {
      ev.preventDefault();
      ev.stopPropagation();
      const target = trash.closest(".wa-msg");
      if (!target || target.classList.contains("revoked")) return;
      activeMsgEl = target;
      doRevoke();
      return;
    }
    const select = ev.target.closest && ev.target.closest(".wa-msg-select");
    if (select) {
      ev.preventDefault();
      ev.stopPropagation();
      const target = select.closest(".wa-msg");
      if (!target || target.classList.contains("revoked")) return;
      enterSelectionMode(target);
    }
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

// Inserta dos affordances hover-only en CUALQUIER burbuja (own o
// other):
//   1) `.wa-msg-trash` arriba — click directo = delete.
//        own  → revoke real (delete-for-everyone, notifica al peer).
//        other → hide local (delete-for-me, solo en este device).
//   2) `.wa-msg-select` justo abajo del trash — entra al modo
//      selección múltiple (acepta own + other, mezcla está OK porque
//      el bulk delete rutea cada uno al endpoint correcto).
// Para reacciones queda long-press / right-click.
export function attachOwnMenuAffordance(msgEl) {
  if (!msgEl || !msgEl.classList) return;
  if (msgEl.classList.contains("revoked")) return;
  if (msgEl.querySelector(".wa-msg-trash")) return;
  const trash = document.createElement("button");
  trash.type = "button";
  trash.className = "wa-msg-trash";
  trash.setAttribute("aria-label", "Eliminar mensaje");
  trash.title = "Eliminar mensaje";
  // SVG inline para no depender de fuentes con glyph 🗑️ (en macOS el
  // emoji renderiza colorido y desentona con el resto del UI).
  trash.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
    + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    + 'stroke-linejoin="round" aria-hidden="true">'
    + '<polyline points="3 6 5 6 21 6"></polyline>'
    + '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>'
    + '<path d="M10 11v6"></path><path d="M14 11v6"></path>'
    + '<path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>'
    + '</svg>';
  msgEl.appendChild(trash);

  const select = document.createElement("button");
  select.type = "button";
  select.className = "wa-msg-select";
  select.setAttribute("aria-label", "Seleccionar varios mensajes");
  select.title = "Seleccionar varios";
  // Círculo vacío (igual al checkmark de selección, pero sin tick).
  select.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
    + 'stroke="currentColor" stroke-width="2" aria-hidden="true">'
    + '<circle cx="12" cy="12" r="9"></circle>'
    + '</svg>';
  msgEl.appendChild(select);
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

  // "Traducir" — disponible para CUALQUIER msg con contenido textual.
  // El backend tiene su propia heurística (skipea si ya está en
  // español rioplatense) y cachea por msg_id.
  if (msgEl.dataset.id && !msgEl.classList.contains("revoked")) {
    const sepT = document.createElement("div");
    sepT.className = "wa-reaction-sep";
    menuEl.appendChild(sepT);
    const tr = document.createElement("button");
    tr.className = "wa-reaction-delete";
    tr.textContent = "Traducir";
    tr.title = "Traducir al español rioplatense (qwen2.5:3b)";
    tr.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      doTranslate(msgEl);
    });
    menuEl.appendChild(tr);
  }

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
  const isOwn = activeMsgEl.classList.contains("own");
  closeMenu();
  if (!messageId) return;
  try {
    // own → revoke (delete-for-everyone, notifica al peer).
    // other → hide (delete-for-me local, no notifica).
    if (isOwn) {
      await revoke(currentJID, messageId);
    } else {
      await hide(currentJID, messageId);
    }
    // Feedback instant. Para own: removemos la burbuja entera (pedido
    // user: "si elimino los de mi lado, no hace falta mostrar ese
    // mensaje"). Para other: lo dejamos como tomb porque la regla
    // distinta sería si el peer borra el suyo (eso sí se muestra,
    // pero esos vienen vía SSE con revoked_by=peer, no por este path).
    const msgEl = document.querySelector(`.wa-msg[data-id="${cssEscape(messageId)}"]`);
    if (msgEl) msgEl.remove();
  } catch (e) {
    console.error("[wa-reactions] delete failed", e);
    const msg = /401/.test(e.message)
      ? "No se pudo eliminar: este device no tiene admin token (solo desde localhost / ra.ai)."
      : `No se pudo eliminar: ${e.message}`;
    window.alert(msg);
  }
}

function cssEscape(s) {
  return String(s).replace(/"/g, '\\"');
}

async function doTranslate(msgEl) {
  closeMenu();
  if (!msgEl) return;
  const msgId = msgEl.dataset.id || "";
  // Extraer el texto plano del msg — el body es un <span> con
  // textContent (sin formatting hostil). Excluimos pre-existentes
  // de gap-pin, quoted, time, sender, etc. que son sibling divs.
  const body = msgEl.querySelector(":scope > span:not(.wa-msg-time):not(.wa-msg-sender):not(.wa-msg-quoted):not(.wa-msg-reactions)");
  const content = (body?.textContent || "").trim();
  if (!content) return;

  // Si ya existe una traducción para este msg, toggle (mostrar/ocultar).
  const existing = msgEl.querySelector(".wa-msg-translation");
  if (existing) {
    existing.remove();
    return;
  }

  // Placeholder inmediato — "traduciendo…" mientras el LLM trabaja.
  const tr = document.createElement("div");
  tr.className = "wa-msg-translation loading";
  tr.textContent = "traduciendo…";
  // Insertar antes del time/reactions para que quede pegado al texto.
  const timeEl = msgEl.querySelector(".wa-msg-time");
  if (timeEl) msgEl.insertBefore(tr, timeEl);
  else msgEl.appendChild(tr);

  try {
    const data = await translate(msgId, content);
    tr.classList.remove("loading");
    if (data.skipped) {
      tr.textContent = "ya está en español";
      tr.classList.add("skipped");
      // Auto-fade después de 2.5s.
      setTimeout(() => tr.remove(), 2500);
      return;
    }
    tr.innerHTML = "";
    const tag = document.createElement("span");
    tag.className = "wa-msg-translation-lang";
    tag.textContent = (data.source_lang || "?") + " → es-AR";
    const body = document.createElement("span");
    body.className = "wa-msg-translation-body";
    body.textContent = data.translated || "";
    tr.appendChild(tag);
    tr.appendChild(body);
  } catch (e) {
    console.error("[wa-translate] failed", e);
    tr.classList.remove("loading");
    tr.classList.add("error");
    tr.textContent = "no pude traducir: " + e.message;
    setTimeout(() => tr.remove(), 3000);
  }
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
  // Sin confirm (pedido user 2026-05-11) — el modo selección ya
  // requiere acción explícita (entrar al modo + tickear + click
  // Eliminar), así que la confirmación extra es ruido.
  const delBtn = _selectBar?.querySelector(".wa-select-delete");
  if (delBtn) {
    delBtn.disabled = true;
    delBtn.textContent = `Eliminando…`;
  }
  let ok = 0;
  const errors = [];
  for (const id of ids) {
    const el = document.querySelector(`.wa-msg[data-id="${cssEscape(id)}"]`);
    const isOwn = el?.classList.contains("own");
    try {
      if (isOwn) {
        await revoke(currentJID, id);
      } else {
        await hide(currentJID, id);
      }
      ok += 1;
      // Removemos la burbuja entera (sin tomb) en ambos casos.
      if (el) el.remove();
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
