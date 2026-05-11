// Render del thread (hilo de un chat). Reverse-pagination on scroll-top.
// Day-dividers entre mensajes de días distintos.

import { fetchThread, markRead } from "./wa-api.js";
import { renderInto as renderAvatar } from "./wa-avatars.js";
import * as composer from "./wa-composer.js";
import * as reactions from "./wa-reactions.js";
import * as media from "./wa-media.js";

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

export function getActiveJID() {
  return currentJID;
}

export async function reload() {
  // Re-fetch del thread activo desde el bridge. Útil cuando volvemos
  // del background y queremos garantizar coherencia post-gap SSE.
  if (!currentJID) return;
  const jid = currentJID;
  currentJID = null;  // forzar el bypass del early-return en open()
  await open(jid);
}

export async function open(jid) {
  if (currentJID === jid) return;
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
    // Saltamos directo al ÚLTIMO mensaje (no a `scrollTop = scrollHeight`,
    // que depende de heights todavía sin medir cuando hay imágenes/audios
    // con loading="lazy" / preload="metadata"). `scrollIntoView` sobre el
    // último child anchora visualmente ahí; cuando media de mensajes
    // anteriores carga después, re-llamamos el mismo método y volvemos
    // a quedar pegados al último.
    snapToLastMessage();
    wireMediaScrollKeepers(els.body);
  }

  // Marcar leído contra el último ts visible
  if (data.messages && data.messages.length > 0) {
    const lastTs = data.messages[data.messages.length - 1].ts;
    markRead(jid, lastTs).catch(() => {});
  }

  currentCtx = { is_group: !!data.is_group };
  composer.setActiveChat(jid);
  reactions.setActiveJID(jid);

  // Auto-resume del gap (≥24h sin leer, ≥5 msgs inbound nuevos).
  // Llamada async POST-render — el thread ya está pintado, el resume
  // aparece pinned arriba cuando el LLM termina (2-5s). Si no hay gap
  // o falla el LLM, no hace nada.
  loadGapSummary(jid).catch((e) => console.warn("[wa-thread] gap-summary fail", e));
}

async function loadGapSummary(jid) {
  // Skip si el thread cambió mientras esperábamos.
  const captureJID = jid;
  let resp;
  try {
    resp = await fetch(`/api/wa/thread/${encodeURIComponent(jid)}/gap-summary`, {
      credentials: "same-origin",
    });
  } catch (_) {
    return;
  }
  if (!resp.ok) return;
  const data = await resp.json().catch(() => null);
  if (!data || !data.summary) return;
  if (currentJID !== captureJID) return; // user cambió de chat
  renderGapSummary(data.summary);
}

function renderGapSummary(s) {
  if (!els.body) return;
  // Si ya hay un pin previo (caso re-open mismo chat), reemplazar.
  const old = document.getElementById("wa-gap-pin");
  if (old) old.remove();
  const pin = document.createElement("div");
  pin.id = "wa-gap-pin";
  pin.className = "wa-gap-pin" + (s.urgent ? " urgent" : "");
  const head = document.createElement("div");
  head.className = "wa-gap-pin-head";
  head.innerHTML = `<span class="wa-gap-pin-icon">${s.urgent ? "🔴" : "📌"}</span>`
    + `<span class="wa-gap-pin-label">Te perdiste ${s.hours_ago}h · ${s.msgs_count} msgs nuevos</span>`
    + `<button class="wa-gap-pin-close" type="button" title="ocultar">×</button>`;
  pin.appendChild(head);
  for (const line of s.summary) {
    const ln = document.createElement("div");
    ln.className = "wa-gap-pin-line";
    ln.textContent = line;
    pin.appendChild(ln);
  }
  head.querySelector(".wa-gap-pin-close").addEventListener("click", () => {
    pin.remove();
  });
  // Insertar al principio del body (antes del primer day-divider / msg).
  els.body.insertBefore(pin, els.body.firstChild);
}

async function onScroll() {
  // Dynamic title collapse — iOS Settings.app pattern. Cuando el user
  // empieza a scrollear hacia abajo (>40px), el thread-name se condensa
  // de 18px → 15.5px via class `.condensed`. Al volver al top, expand.
  if (els.header?.name) {
    const collapsed = (els.body.scrollTop || 0) > 40;
    els.header.name.classList.toggle("condensed", collapsed);
  }

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
  // Unread divider: lo insertamos UNA vez, antes del primer inbound
  // posterior a `last_seen_ts`. Solo en el render inicial (prepend=false);
  // los mensajes paginados hacia arriba son todos read por definición.
  let unreadInserted = !!prepend;
  const lastSeen = (threadCtx && threadCtx.last_seen_ts) || "";
  for (const m of messages) {
    const day = dayKey(m.ts);
    if (day && day !== lastDay) {
      frag.appendChild(renderDayDivider(day, m.ts));
      lastDay = day;
    }
    if (!unreadInserted && lastSeen && !m.is_from_me && m.ts > lastSeen) {
      frag.appendChild(renderUnreadDivider());
      unreadInserted = true;
    }
    frag.appendChild(renderMsg(m, threadCtx));
  }
  if (prepend) {
    els.body.insertBefore(frag, els.body.firstChild);
  } else {
    els.body.appendChild(frag);
  }
}

function snapToLastMessage() {
  if (!els.body) return;
  const last = els.body.lastElementChild;
  if (last && typeof last.scrollIntoView === "function") {
    // `block: 'end'` deja el bottom del último mensaje pegado al bottom
    // del viewport. `behavior: 'instant'` mata cualquier smooth heredado
    // del CSS o del polyfill.
    last.scrollIntoView({ block: "end", behavior: "instant" });
  } else {
    els.body.scrollTop = els.body.scrollHeight;
  }
}

// Mantiene el viewport pegado al último mensaje mientras las
// imágenes/audios cargan async. Se autocancela apenas el user toca
// wheel/touchpad o scrollea > 80px hacia arriba — para no pelearle
// la intención.
function wireMediaScrollKeepers(body) {
  let active = true;
  const cancel = () => { active = false; };
  body.addEventListener("wheel", cancel, { passive: true, once: true });
  body.addEventListener("touchstart", cancel, { passive: true, once: true });
  body.addEventListener("scroll", () => {
    if (els.body.scrollHeight - els.body.scrollTop - els.body.clientHeight > 80) {
      active = false;
    }
  }, { passive: true });

  const media = body.querySelectorAll("img, video, audio");
  for (const el of media) {
    const ev = el.tagName === "IMG" ? "load" : "loadedmetadata";
    el.addEventListener(ev, () => { if (active) snapToLastMessage(); }, { once: true });
    el.addEventListener("error", () => { if (active) snapToLastMessage(); }, { once: true });
  }
}

function renderCallBubble(m) {
  const wrap = document.createElement("div");
  wrap.className = "wa-call-bubble";
  wrap.dataset.id = m.id || "";
  wrap.dataset.status = m.call_status || "offered";
  wrap.dataset.video = m.call_is_video ? "1" : "0";
  const icon = document.createElement("span");
  icon.className = "wa-call-icon";
  if (m.call_status === "missed") icon.textContent = "📵";
  else if (m.call_status === "rejected") icon.textContent = "❌";
  else if (m.call_is_video) icon.textContent = "📹";
  else icon.textContent = "📞";
  const text = document.createElement("span");
  text.className = "wa-call-text";
  text.textContent = m.content || "Llamada";
  const time = document.createElement("span");
  time.className = "wa-call-time";
  try {
    const d = new Date(m.ts);
    if (!isNaN(d.getTime())) {
      time.textContent = d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
    }
  } catch {}
  wrap.appendChild(icon);
  wrap.appendChild(text);
  if (time.textContent) wrap.appendChild(time);
  return wrap;
}

// SSE handler para wa_call: inserta o updatea el bubble cuando el call
// está en el thread activo. Si el call ya existe (transition de status:
// offered → accepted → terminated), se reemplaza in-place.
export function applyCallEvent(payload) {
  if (!payload || !payload.call_id || payload.jid !== currentJID) return;
  if (!els.body) return;
  const synthetic = {
    id: `call:${payload.call_id}`,
    ts: payload.ts || new Date().toISOString(),
    sender: payload.from_jid || "",
    sender_label: (payload.from_jid || "").split("@")[0] || "",
    content: _callContentFromPayload(payload),
    is_from_me: false,
    media_type: "call",
    call_status: payload.status,
    call_is_video: payload.is_video,
    call_duration_s: payload.duration_s,
  };
  const existing = els.body.querySelector(`.wa-call-bubble[data-id="call:${cssEscape(payload.call_id)}"]`);
  const fresh = renderCallBubble(synthetic);
  if (existing) {
    existing.replaceWith(fresh);
  } else {
    const wasAtBottom = els.body.scrollHeight - els.body.scrollTop - els.body.clientHeight < 80;
    els.body.appendChild(fresh);
    if (wasAtBottom) snapToLastMessage();
  }
}

function _callContentFromPayload(p) {
  const verb = p.is_video ? "Videollamada" : "Llamada";
  const ds = p.duration_s || 0;
  const mm = Math.floor(ds / 60);
  const ss = ds % 60;
  switch (p.status) {
    case "missed": return `📵 ${verb} perdida`;
    case "rejected": return `❌ ${verb} rechazada`;
    case "terminated": return `📞 ${verb} · ${mm}:${String(ss).padStart(2, "0")}`;
    case "accepted": return `📞 ${verb} en curso`;
    default: return `📞 ${verb} entrante`;
  }
}

function renderUnreadDivider() {
  const div = document.createElement("div");
  div.className = "wa-unread-divider";
  div.textContent = "no leído";
  return div;
}

function renderMsg(m, ctx) {
  // Calls: render como bubble centrada, no como msg normal.
  if (m.media_type === "call") {
    return renderCallBubble(m);
  }
  const div = document.createElement("div");
  div.className = "wa-msg " + (m.is_from_me ? "own" : "other");
  if (m.revoked) div.classList.add("revoked");
  div.dataset.id = m.id || "";
  div.dataset.sender = m.sender || "";

  if (m.revoked) {
    div.textContent = "🚫 Este mensaje fue eliminado";
    return div;
  }

  // Sender label arriba de cada mensaje (inbound + outbound). Pedido
  // user 2026-05-11: en propios va "Yo" como label uniforme; en
  // inbound el nombre resuelto via push_name → Apple Contacts → vault.
  if (m.sender_label) {
    const sender = document.createElement("div");
    sender.className = "wa-msg-sender";
    sender.textContent = m.is_from_me ? "Yo" : m.sender_label;
    div.appendChild(sender);
  }

  // Quoted reply
  if (m.quoted && (m.quoted.text || m.quoted.id)) {
    const q = document.createElement("div");
    q.className = "wa-msg-quoted";
    q.textContent = m.quoted.text || `↩ msg ${m.quoted.id.slice(0, 8)}…`;
    div.appendChild(q);
  }

  // Media render (Fase 7) — solo si tenemos `id` real (no pending optimistic).
  if (m.media_type && m.id && !String(m.id).startsWith("tmp-") && !m.pending) {
    const mediaMsg = { ...m, jid: currentJID, chat_jid: currentJID };
    media.renderInto(div, mediaMsg);
  } else if (m.media_type) {
    // Fallback hint mientras está pending (sin id real).
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

  // Reactions: un chip por emoji con su count. Se wrappean a múltiples
  // líneas si hay muchas (antes salían pegoteadas como "❤️ 👍 😂 …" en
  // una sola línea, reportado 2026-05-11 "se ven horribles").
  if (m.reactions && m.reactions.length > 0) {
    const r = document.createElement("div");
    r.className = "wa-msg-reactions";
    const grouped = {};
    for (const rx of m.reactions) {
      grouped[rx.emoji] = (grouped[rx.emoji] || 0) + 1;
    }
    for (const [emoji, n] of Object.entries(grouped)) {
      const chip = document.createElement("span");
      chip.className = "wa-reaction-chip";
      const em = document.createElement("span");
      em.className = "wa-reaction-emoji";
      em.textContent = emoji;
      chip.appendChild(em);
      if (n > 1) {
        const c = document.createElement("span");
        c.className = "wa-reaction-count";
        c.textContent = String(n);
        chip.appendChild(c);
      }
      r.appendChild(chip);
    }
    div.appendChild(r);
  }

  // Affordances hover (trash + select) en CUALQUIER burbuja (own +
  // other). Para own → revoke real. Para other → hide local
  // (delete-for-me, sin notificar peer). Long-press + contextmenu
  // siguen para el menu de reacciones.
  reactions.attachOwnMenuAffordance(div);

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
  // Microinteraction: msgs nuevos via SSE entran con fade+slide. La
  // class `.fresh` dispara `@keyframes wa-msg-in` y se auto-elimina al
  // final para que el msg no quede con animation residual.
  msgEl.classList.add("fresh");
  msgEl.addEventListener("animationend", () => msgEl.classList.remove("fresh"), { once: true });
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
