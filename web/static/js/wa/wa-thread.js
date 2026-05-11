// Render del thread (hilo de un chat). Reverse-pagination on scroll-top.
// Day-dividers entre mensajes de días distintos.

import { fetchThread, markRead } from "./wa-api.js";
import { renderInto as renderAvatar } from "./wa-avatars.js";

const els = {
  body: null,
  empty: null,
  header: { name: null, avatar: null, presence: null },
  composer: null,
};

let currentJID = null;
let oldestTs = null;     // ts del mensaje más viejo cargado (para reverse-pagination)
let loadingOlder = false;
let scrollPositions = new Map(); // jid → scrollTop persistido entre switches

export function init({ bodyEl, emptyEl, nameEl, avatarEl, presenceEl, composerEl }) {
  els.body = bodyEl;
  els.empty = emptyEl;
  els.header.name = nameEl;
  els.header.avatar = avatarEl;
  els.header.presence = presenceEl;
  els.composer = composerEl;

  if (els.body) {
    els.body.addEventListener("scroll", onScroll, { passive: true });
  }
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
    renderAvatar(els.header.avatar, jid, initialsFromLabel(data.label || ""));
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

  // Habilitar composer (sin send real todavía — Fase 5).
  if (els.composer) {
    els.composer.hidden = false;
    const input = els.composer.querySelector("textarea");
    const btn = els.composer.querySelector(".wa-send-btn");
    if (input) {
      input.placeholder = "Escribí un mensaje… (envío en Fase 5)";
      input.disabled = true;
    }
    if (btn) btn.disabled = true;
  }
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
