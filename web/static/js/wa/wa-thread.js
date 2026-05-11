// Render del thread (hilo de un chat). Reverse-pagination on scroll-top.
// Day-dividers entre mensajes de días distintos.

import { fetchThread, markRead } from "./wa-api.js";
import { renderInto as renderAvatar } from "./wa-avatars.js";
import * as composer from "./wa-composer.js";
import * as reactions from "./wa-reactions.js";

const els = {
  body: null,
  empty: null,
  header: { name: null, avatar: null, presence: null },
  composer: null,
};

let currentJID = null;
let currentCtx = { is_group: false };
let oldestTs = null;     // ts del mensaje más viejo cargado (para reverse-pagination)
let loadingOlder = false;
let scrollPositions = new Map(); // jid → scrollTop persistido entre switches
let pendingByContent = new Map(); // `${jid}|${content}` → tempId (para dedup contra SSE)

export function init({ bodyEl, emptyEl, nameEl, avatarEl, presenceEl, composerEl }) {
  els.body = bodyEl;
  els.empty = emptyEl;
  els.header.name = nameEl;
  els.header.avatar = avatarEl;
  els.header.presence = presenceEl;
  els.composer = composerEl;

  if (els.body) {
    els.body.addEventListener("scroll", onScroll, { passive: true });
    // Doble-click sobre una burbuja → empieza reply. Fase 6 sumará
    // long-press picker; por ahora dblclick es atajo barato.
    els.body.addEventListener("dblclick", (ev) => {
      const msgEl = ev.target.closest && ev.target.closest(".wa-msg");
      if (msgEl) startReplyTo(msgEl);
    });
  }
  composer.init({
    rootEl: composerEl,
    onOptimisticInsert: (jid, message) => {
      pendingByContent.set(`${jid}|${message.content}`, message.id);
      appendMessageIfActive(jid, message);
    },
  });
  reactions.attach(els.body);
}

export async function open(jid) {
  if (currentJID === jid) return;
  // Persistir scroll del thread saliente.
  if (currentJID && els.body) {
    scrollPositions.set(currentJID, els.body.scrollTop);
  }
  currentJID = jid;
  oldestTs = null;
  if (els.empty) els.empty.style.display = "none";
  if (els.body) els.body.innerHTML = '<div class="wa-empty-state">Cargando…</div>';

  let data;
  try {
    data = await fetchThread(jid, { limit: 50 });
  } catch (e) {
    console.error("[wa-thread] open failed", e);
    if (els.body) els.body.innerHTML = `<div class="wa-empty-state">Error: ${e.message}</div>`;
    return;
  }

  // Header
  if (els.header.name) els.header.name.textContent = data.label || jid;
  if (els.header.avatar) {
    renderAvatar(els.header.avatar, jid, initialsFromLabel(data.label || ""), data.label);
  }
  if (els.header.presence) els.header.presence.textContent = "";

  // Render messages
  if (els.body) {
    els.body.innerHTML = "";
    renderMessages(data.messages || [], data, /*prepend=*/ false);
    oldestTs = data.next_before_ts;
    // Scroll restore: si veníamos de este chat, restaurar; sino al fondo.
    const restored = scrollPositions.get(jid);
    if (typeof restored === "number") {
      els.body.scrollTop = restored;
    } else {
      els.body.scrollTop = els.body.scrollHeight;
    }
  }

  // Marcar leído contra el último ts visible
  if (data.messages && data.messages.length > 0) {
    const lastTs = data.messages[data.messages.length - 1].ts;
    markRead(jid, lastTs).catch(() => {});
  }

  currentCtx = { is_group: !!data.is_group };
  composer.setActiveChat(jid);
  reactions.setActiveJID(jid);
}

async function onScroll() {
  if (!els.body || loadingOlder || !oldestTs || !currentJID) return;
  if (els.body.scrollTop > 60) return;
  loadingOlder = true;
  const prevHeight = els.body.scrollHeight;
  try {
    const data = await fetchThread(currentJID, { limit: 50, beforeTs: oldestTs });
    if (data.messages && data.messages.length > 0) {
      renderMessages(data.messages, data, /*prepend=*/ true);
      oldestTs = data.next_before_ts;
      // Mantener viewport en el mismo mensaje (delta scroll-height).
      const newHeight = els.body.scrollHeight;
      els.body.scrollTop = newHeight - prevHeight;
    } else {
      oldestTs = null; // fin de historial
    }
  } catch (e) {
    console.error("[wa-thread] paginate failed", e);
  } finally {
    loadingOlder = false;
  }
}

function renderMessages(messages, threadCtx, prepend) {
  const frag = document.createDocumentFragment();
  let lastDay = prepend ? null : findLastDay();
  for (const m of messages) {
    const day = dayKey(m.ts);
    if (day && day !== lastDay) {
      frag.appendChild(renderDayDivider(day, m.ts));
      lastDay = day;
    }
    frag.appendChild(renderMsg(m, threadCtx));
  }
  if (prepend) {
    els.body.insertBefore(frag, els.body.firstChild);
  } else {
    els.body.appendChild(frag);
  }
}

function renderMsg(m, ctx) {
  const div = document.createElement("div");
  div.className = "wa-msg " + (m.is_from_me ? "own" : "other");
  if (m.revoked) div.classList.add("revoked");
  div.dataset.id = m.id || "";
  div.dataset.sender = m.sender || "";

  if (m.revoked) {
    div.textContent = "🚫 Este mensaje fue eliminado";
    return div;
  }

  // Sender (solo en grupos + no propios)
  if (ctx.is_group && !m.is_from_me && m.sender_label) {
    const sender = document.createElement("div");
    sender.className = "wa-msg-sender";
    sender.textContent = m.sender_label;
    div.appendChild(sender);
  }

  // Quoted reply
  if (m.quoted && (m.quoted.text || m.quoted.id)) {
    const q = document.createElement("div");
    q.className = "wa-msg-quoted";
    q.textContent = m.quoted.text || `↩ msg ${m.quoted.id.slice(0, 8)}…`;
    div.appendChild(q);
  }

  // Media hint (Fase 7 lo va a renderear visualmente)
  if (m.media_type) {
    const hint = document.createElement("div");
    hint.className = "wa-msg-media-hint";
    hint.textContent = `[${m.media_type}${m.filename ? `: ${m.filename}` : ""}]`;
    div.appendChild(hint);
  }

  // Content
  if (m.content) {
    const body = document.createElement("span");
    body.textContent = m.content;
    div.appendChild(body);
  }

  // Time
  const t = document.createElement("span");
  t.className = "wa-msg-time";
  t.textContent = formatTime(m.ts);
  div.appendChild(t);

  // Reactions
  if (m.reactions && m.reactions.length > 0) {
    const r = document.createElement("div");
    r.className = "wa-msg-reactions";
    const grouped = {};
    for (const rx of m.reactions) {
      grouped[rx.emoji] = (grouped[rx.emoji] || 0) + 1;
    }
    r.textContent = Object.entries(grouped)
      .map(([e, n]) => (n > 1 ? `${e}${n}` : e))
      .join(" ");
    div.appendChild(r);
  }

  return div;
}

function dayKey(iso) {
  if (!iso) return null;
  return iso.slice(0, 10);
}

function findLastDay() {
  if (!els.body) return null;
  const dividers = els.body.querySelectorAll(".wa-day-divider");
  if (dividers.length === 0) return null;
  return dividers[dividers.length - 1].dataset.day || null;
}

function renderDayDivider(day, sampleTs) {
  const d = document.createElement("div");
  d.className = "wa-day-divider";
  d.dataset.day = day;
  d.textContent = humanDay(sampleTs);
  return d;
}

function humanDay(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
  const now = new Date();
  const today = now.toDateString();
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === today) return "HOY";
  if (d.toDateString() === y.toDateString()) return "AYER";
  return d.toLocaleDateString("es-AR", { weekday: "long", day: "numeric", month: "long" });
}

function formatTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
}

function initialsFromLabel(label) {
  const parts = (label || "").trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

// ── SSE handlers públicos ──────────────────────────────────────────

/** Append un mensaje nuevo al thread si el chat está activo.
 * Si no es el chat activo, ignoramos (la sidebar lo refleja via chat_update).
 *
 * Dedup contra optimistic insert: si ya existe un msg pending con el
 * mismo content+jid, lo "promovemos" reemplazando su id+pending state
 * por el real del bridge (cap dedup: 30s para que tickets viejos no
 * absorban mensajes idénticos posteriores).
 */
export function appendMessageIfActive(jid, message) {
  if (!els.body || jid !== currentJID) return;

  // Dedup optimistic
  const dedupKey = `${jid}|${message.content || ""}`;
  const tempId = pendingByContent.get(dedupKey);
  if (tempId && !message.pending) {
    const existing = els.body.querySelector(`.wa-msg[data-id="${cssEscape(tempId)}"]`);
    if (existing) {
      existing.dataset.id = message.id || tempId;
      existing.classList.remove("pending");
      pendingByContent.delete(dedupKey);
      return;
    }
    pendingByContent.delete(dedupKey);
  }

  const wasAtBottom = els.body.scrollHeight - els.body.scrollTop - els.body.clientHeight < 80;

  const day = dayKey(message.ts);
  const lastDay = findLastDay();
  if (day && day !== lastDay) {
    els.body.appendChild(renderDayDivider(day, message.ts));
  }
  const msgEl = renderMsg(message, currentCtx);
  if (message.pending) msgEl.classList.add("pending");
  els.body.appendChild(msgEl);
  if (wasAtBottom) {
    els.body.scrollTop = els.body.scrollHeight;
  }
  // Marca leído contra este ts (el user lo está viendo).
  if (!message.pending) {
    markRead(jid, message.ts).catch(() => {});
  }
}

/** Trigger del long-press / click derecho sobre una burbuja para responder.
 * Fase 6 va a sumar long-press para reactions; por ahora exponemos el
 * helper para que la UI lo invoque desde un menú contextual o doble-click.
 */
export function startReplyTo(messageEl) {
  if (!messageEl) return;
  const id = messageEl.dataset.id || "";
  const isOwn = messageEl.classList.contains("own");
  // El sender_label puede no estar en el DOM (own no lo renderea); leemos
  // del content visible solamente para el preview.
  const contentEl = messageEl.querySelector("span:not(.wa-msg-time)");
  const text = contentEl ? contentEl.textContent : "";
  composer.setReply({
    message_id: id,
    original_text: text,
    sender_jid: isOwn ? "" : "",
  });
}

/** Aplica un reaction_changed al mensaje existente en el DOM (si está). */
export function applyReactionChange(payload) {
  if (!els.body || !payload || !payload.message_id) return;
  const msgEl = els.body.querySelector(`.wa-msg[data-id="${cssEscape(payload.message_id)}"]`);
  if (!msgEl) return;
  let rEl = msgEl.querySelector(".wa-msg-reactions");
  if (payload.removed) {
    if (rEl) rEl.remove();
    return;
  }
  if (!rEl) {
    rEl = document.createElement("div");
    rEl.className = "wa-msg-reactions";
    msgEl.appendChild(rEl);
  }
  rEl.textContent = payload.emoji || "";
}

/** Marca un mensaje como revocado en el DOM. */
export function applyRevoke(payload) {
  if (!els.body || !payload || !payload.message_id) return;
  const msgEl = els.body.querySelector(`.wa-msg[data-id="${cssEscape(payload.message_id)}"]`);
  if (!msgEl) return;
  msgEl.classList.add("revoked");
  msgEl.innerHTML = "🚫 Este mensaje fue eliminado";
}

/** Muestra typing indicator si la presence pertenece al chat activo. */
let presenceTimer = null;
export function applyPresence(payload) {
  if (!els.header.presence || !payload || payload.chat_jid !== currentJID) return;
  const state = payload.state || "";
  const media = payload.media || "";
  let text = "";
  if (state === "composing") text = media === "audio" ? "grabando audio…" : "escribiendo…";
  els.header.presence.textContent = text;
  if (presenceTimer) clearTimeout(presenceTimer);
  if (text) {
    presenceTimer = setTimeout(() => {
      if (els.header.presence) els.header.presence.textContent = "";
    }, 6000);
  }
}

function cssEscape(s) {
  // Minimal escape para selectores — WhatsApp message IDs son hex,
  // pero por las dudas escapamos comillas dobles.
  return String(s).replace(/"/g, '\\"');
}
