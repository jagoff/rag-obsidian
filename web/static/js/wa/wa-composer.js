// Composer del thread — textarea autosize + Enter-to-send + reply preview.
// Optimistic UI: insertamos burbuja "pending" inmediatamente; cuando
// el SSE trae el `new_message` real con el mismo content+jid+~5s, el
// thread la reemplaza por la versión persistida del bridge.

import { sendText, typing as sendTyping, toneShift } from "./wa-api.js";
import { uploadMedia } from "./wa-media.js";
import * as voice from "./wa-voice.js";

const els = {
  root: null,
  input: null,
  btn: null,
  replyBar: null,
  toneBar: null,
};

let currentJID = null;
let pendingReply = null; // {message_id, original_text, sender_jid}
let onOptimisticInsertCb = null;

// Tone shifter: stack de undos para back/forward entre tonos. Cada entry
// es el texto del input ANTES del shift. La undo button vuelve al
// previous, no necesariamente al original (si aplicaste 2 shifts encadenados).
let toneUndoStack = [];

export function init({ rootEl, onOptimisticInsert }) {
  els.root = rootEl;
  if (!rootEl) return;
  els.input = rootEl.querySelector("textarea");
  els.btn = rootEl.querySelector(".wa-send-btn");
  onOptimisticInsertCb = onOptimisticInsert;

  // Reply bar se crea lazy en setReply(); cuando no hay reply
  // activo no queda DOM ruido.

  // Tone shifter chips — se montan lazy debajo del textarea. Hidden
  // hasta que el textarea tenga contenido.
  mountToneBar();

  els.input.addEventListener("input", () => {
    autosize();
    pingTyping();
    scheduleDraftCheck();
    updateToneBarVisibility();
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  els.input.addEventListener("blur", () => stopTyping());
  els.btn.addEventListener("click", submit);

  // Paste image desde clipboard.
  els.input.addEventListener("paste", (e) => {
    if (!currentJID || !e.clipboardData) return;
    const items = Array.from(e.clipboardData.items || []);
    for (const it of items) {
      if (it.kind === "file" && it.type && it.type.startsWith("image/")) {
        const f = it.getAsFile();
        if (f) {
          e.preventDefault();
          sendMediaFile(f);
          return;
        }
      }
    }
  });

  // Drag & drop sobre todo el composer.
  const root = els.root;
  const onDragOver = (e) => {
    if (!currentJID) return;
    e.preventDefault();
    root.classList.add("dragging");
  };
  const onDragLeave = () => root.classList.remove("dragging");
  const onDrop = (e) => {
    e.preventDefault();
    root.classList.remove("dragging");
    if (!currentJID || !e.dataTransfer) return;
    const files = Array.from(e.dataTransfer.files || []);
    for (const f of files) sendMediaFile(f);
  };
  root.addEventListener("dragover", onDragOver);
  root.addEventListener("dragleave", onDragLeave);
  root.addEventListener("drop", onDrop);

  // Voice notes init
  voice.init({
    btnEl: document.getElementById("wa-mic-btn"),
    recordBarEl: document.getElementById("wa-record-bar"),
    recordTimerEl: document.getElementById("wa-record-timer"),
    onSend: ({ jid, blob, transcript }) => {
      // Optimistic insert al thread.
      const tempId = `tmp-${Date.now()}-voice-${Math.random().toString(36).slice(2, 8)}`;
      if (onOptimisticInsertCb) {
        onOptimisticInsertCb(jid, {
          id: tempId,
          ts: new Date().toISOString(),
          sender: "yo",
          sender_label: "yo",
          content: transcript || "[voice note]",
          is_from_me: true,
          media_type: "audio",
          filename: "voice.ogg",
          quoted: null,
          reactions: [],
          revoked: false,
          pending: true,
        });
      }
    },
  });
}

async function sendMediaFile(file) {
  if (!currentJID || !file) return;
  const replyTo = pendingReply;
  const tempId = `tmp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const isImage = (file.type || "").startsWith("image/");
  const isVideo = (file.type || "").startsWith("video/");
  const isAudio = (file.type || "").startsWith("audio/");
  const mediaType = isImage ? "image" : isVideo ? "video" : isAudio ? "audio" : "document";

  if (onOptimisticInsertCb) {
    onOptimisticInsertCb(currentJID, {
      id: tempId,
      ts: new Date().toISOString(),
      sender: "yo",
      sender_label: "yo",
      content: `[subiendo ${file.name}…]`,
      is_from_me: true,
      media_type: mediaType,
      filename: file.name,
      quoted: null,
      reactions: [],
      revoked: false,
      pending: true,
    });
  }
  setReply(null);

  try {
    await uploadMedia(currentJID, file, { replyToId: replyTo ? replyTo.message_id : "" });
    // Optimistic queda como pending hasta que SSE traiga el real.
  } catch (e) {
    console.error("[wa-composer] sendMediaFile failed", e);
    const el = document.querySelector(`.wa-msg[data-id="${tempId}"]`);
    if (el) {
      el.classList.remove("pending");
      el.classList.add("failed");
      const t = el.querySelector(".wa-msg-time");
      if (t) t.textContent = `falló: ${e.message || "upload"}`;
    }
  }
}

// Typing emit: composing cada 5s mientras hay focus + texto. Auto-paused
// si pasan 4s sin tipear o si el textarea pierde focus.
let typingActive = false;
let typingInterval = null;
let typingIdleTimer = null;

function pingTyping() {
  if (!currentJID || !els.input) return;
  const has = (els.input.value || "").trim().length > 0;
  if (!has) {
    stopTyping();
    return;
  }
  if (!typingActive) {
    typingActive = true;
    sendTyping(currentJID, "composing");
    typingInterval = setInterval(() => {
      if (typingActive && currentJID) sendTyping(currentJID, "composing");
    }, 5000);
  }
  // Reset idle: si no tipea por 4s, pausamos.
  clearTimeout(typingIdleTimer);
  typingIdleTimer = setTimeout(stopTyping, 4000);
}

function stopTyping() {
  if (!typingActive) return;
  typingActive = false;
  clearInterval(typingInterval);
  typingInterval = null;
  clearTimeout(typingIdleTimer);
  typingIdleTimer = null;
  if (currentJID) sendTyping(currentJID, "paused");
}

export function setActiveChat(jid) {
  stopTyping();
  voice.setActiveJID(jid);
  currentJID = jid;
  pendingReply = null;
  // Switch de chat resetea el undo stack del tone shifter — el texto
  // que estaba en el composer pertenecía al chat anterior.
  toneUndoStack = [];
  refreshUndoBtn();
  updateToneBarVisibility();
  hideReplyBar();
  clearDraftBadge();
  // El composer usa la class `.idle` (no `disabled` HTML) para que la
  // UI no quede atascada si el JS no carga (ver wa.html — el composer
  // arranca con clase `.idle`). El submit() también guardea sobre
  // `currentJID` por las dudas.
  if (els.root) {
    els.root.classList.toggle("idle", !jid);
  }
  if (els.input) {
    els.input.value = "";
    els.input.placeholder = jid ? "escribí un mensaje…" : "elegí un chat para escribir";
    autosize();
  }
}

export function setReply(replyTo) {
  pendingReply = replyTo;
  if (!replyTo) {
    hideReplyBar();
    return;
  }
  if (!els.replyBar) {
    if (!els.root || !els.input) return;
    els.replyBar = document.createElement("div");
    els.replyBar.className = "wa-reply-bar";
    els.root.insertBefore(els.replyBar, els.input);
  }
  els.replyBar.innerHTML = `
    <div class="wa-reply-bar-text">
      <strong>Respondiendo a:</strong>
      <div class="wa-reply-bar-preview"></div>
    </div>
    <button class="wa-reply-bar-close" aria-label="Cancelar respuesta">×</button>
  `;
  const preview = els.replyBar.querySelector(".wa-reply-bar-preview");
  if (preview) preview.textContent = (replyTo.original_text || "").slice(0, 200);
  const close = els.replyBar.querySelector(".wa-reply-bar-close");
  if (close) close.addEventListener("click", () => setReply(null));
  if (els.input) els.input.focus();
}

function hideReplyBar() {
  if (els.replyBar) {
    els.replyBar.remove();
    els.replyBar = null;
  }
}

function autosize() {
  if (!els.input) return;
  els.input.style.height = "auto";
  const next = Math.min(els.input.scrollHeight, 120);
  els.input.style.height = `${next}px`;
}

// ── Contradiction radar (debounced draft_check) ─────────────────────
// Cada keystroke reseteo un timer de 800ms. Cuando para de tipear,
// disparo POST /api/wa/draft_check {text, jid}. Si hay conflicts,
// renderizo un badge sutil arriba del composer con el detalle expandible.

let draftCheckTimer = null;
let draftCheckCtrl = null;
let draftBadgeEl = null;

function scheduleDraftCheck() {
  if (draftCheckTimer) clearTimeout(draftCheckTimer);
  const text = (els.input && els.input.value || "").trim();
  if (text.length < 30 || !currentJID) {
    clearDraftBadge();
    return;
  }
  draftCheckTimer = setTimeout(() => runDraftCheck(text, currentJID), 800);
}

async function runDraftCheck(text, jid) {
  // Cancela request en vuelo (el user siguió escribiendo).
  if (draftCheckCtrl) {
    try { draftCheckCtrl.abort(); } catch {}
  }
  draftCheckCtrl = new AbortController();
  try {
    const resp = await fetch("/api/wa/draft_check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, jid, window_hours: 72 }),
      signal: draftCheckCtrl.signal,
    });
    const data = await resp.json();
    if (data && data.ok && (data.conflicts || []).length > 0) {
      renderDraftBadge(data.conflicts);
    } else {
      clearDraftBadge();
    }
  } catch (e) {
    if (e.name !== "AbortError") clearDraftBadge();
  }
}

function clearDraftBadge() {
  if (draftBadgeEl) {
    draftBadgeEl.remove();
    draftBadgeEl = null;
  }
}

function renderDraftBadge(conflicts) {
  clearDraftBadge();
  if (!els.root || !els.input) return;
  const wrap = document.createElement("div");
  wrap.className = "wa-draft-warn";
  const summary = document.createElement("button");
  summary.type = "button";
  summary.className = "wa-draft-warn-summary";
  summary.innerHTML = `⚠ <strong>${conflicts.length}</strong> ` +
    `posible contradicción con tu historial reciente — click para ver`;
  const list = document.createElement("div");
  list.className = "wa-draft-warn-list";
  list.hidden = true;
  for (const c of conflicts) {
    const row = document.createElement("div");
    row.className = "wa-draft-warn-row";
    const meta = document.createElement("div");
    meta.className = "wa-draft-warn-meta";
    meta.textContent = `${c.path || "(unknown)"} · score ${c.score}`;
    const reason = document.createElement("div");
    reason.className = "wa-draft-warn-reason";
    reason.textContent = c.reason || "(sin reason)";
    const snippet = document.createElement("div");
    snippet.className = "wa-draft-warn-snippet";
    snippet.textContent = c.snippet || "";
    row.appendChild(reason);
    row.appendChild(snippet);
    row.appendChild(meta);
    list.appendChild(row);
  }
  summary.addEventListener("click", () => {
    list.hidden = !list.hidden;
  });
  const close = document.createElement("button");
  close.type = "button";
  close.className = "wa-draft-warn-close";
  close.setAttribute("aria-label", "descartar");
  close.textContent = "×";
  close.addEventListener("click", clearDraftBadge);
  wrap.appendChild(summary);
  wrap.appendChild(close);
  wrap.appendChild(list);
  els.root.insertBefore(wrap, els.input);
  draftBadgeEl = wrap;
}

async function submit() {
  if (!els.input || !els.btn) return;
  if (!currentJID) {
    window.alert("elegí un chat antes de mandar mensajes.");
    return;
  }
  const text = (els.input.value || "").trim();
  if (!text) return;

  const replyTo = pendingReply;
  const tempId = `tmp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Optimistic insert
  if (onOptimisticInsertCb) {
    onOptimisticInsertCb(currentJID, {
      id: tempId,
      ts: new Date().toISOString(),
      sender: "yo",
      sender_label: "yo",
      content: text,
      is_from_me: true,
      media_type: null,
      filename: null,
      quoted: replyTo
        ? { id: replyTo.message_id, text: replyTo.original_text || "" }
        : null,
      reactions: [],
      revoked: false,
      pending: true,
    });
  }

  els.input.value = "";
  autosize();
  setReply(null);
  clearDraftBadge();
  stopTyping();
  // Send → reset tone undo stack (msg ya fue).
  toneUndoStack = [];
  updateToneBarVisibility();
  if (els.btn) els.btn.classList.add("sending");

  try {
    const r = await sendText(currentJID, text, replyTo);
    if (!r.ok) {
      markFailed(tempId, r.error_kind || "fail");
    } else {
      // El SSE va a traer el mensaje real del bridge — el thread va
      // a fusionar contra `tempId` por content+chat+ts cerca.
      markSent(tempId);
    }
  } catch (e) {
    console.error("[wa-composer] send failed", e);
    markFailed(tempId, e.message || "error");
  } finally {
    if (els.btn) els.btn.classList.remove("sending");
    if (els.input) els.input.focus();
  }
}

function markSent(tempId) {
  const el = document.querySelector(`.wa-msg[data-id="${tempId}"]`);
  if (el) el.classList.remove("pending");
}

function markFailed(tempId, reason) {
  const el = document.querySelector(`.wa-msg[data-id="${tempId}"]`);
  if (!el) return;
  el.classList.remove("pending");
  el.classList.add("failed");
  const t = el.querySelector(".wa-msg-time");
  if (t) t.textContent = `falló: ${reason}`;
}


// ─────────────────────────────────────────────────────────────
// Tone shifter (chips debajo del composer)
// ─────────────────────────────────────────────────────────────
//
// 4 chips: +formal · +casual · +corto · +cariñoso.
// - Hidden cuando el textarea está vacío.
// - Click → fetch /api/wa/tone-shift → reemplaza input + push undo.
// - Botón "↺" aparece después del primer shift; pop del stack.
// - Mientras espera el LLM, chip muestra spinner inline y disables todos.
//
// Innovador porque ningún cliente reescribe contextual el draft. El
// LLM ya está warm (qwen2.5:3b), call cuesta 1-2s warm, cached
// instant para back/forward.

const _TONES = [
  { id: "formal", label: "+ formal", title: "Más formal y prolijo" },
  { id: "casual", label: "+ casual", title: "Más relajado e informal" },
  { id: "short", label: "+ corto", title: "Más corto y directo" },
  { id: "warm", label: "+ cariñoso", title: "Más cálido" },
];

function mountToneBar() {
  if (!els.root || els.toneBar) return;
  const bar = document.createElement("div");
  bar.className = "wa-tone-bar";
  bar.hidden = true;
  for (const t of _TONES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "wa-tone-chip";
    btn.dataset.tone = t.id;
    btn.title = t.title;
    btn.textContent = t.label;
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      applyTone(t.id);
    });
    bar.appendChild(btn);
  }
  // Undo button — hidden hasta que haya algo en el stack.
  const undo = document.createElement("button");
  undo.type = "button";
  undo.className = "wa-tone-undo";
  undo.title = "Revertir al texto anterior";
  undo.textContent = "↺";
  undo.hidden = true;
  undo.addEventListener("click", (ev) => {
    ev.preventDefault();
    undoTone();
  });
  bar.appendChild(undo);
  // Insertar después del textarea (antes del kbd hint si existe).
  const inputEl = els.input;
  if (inputEl && inputEl.parentNode) {
    inputEl.parentNode.insertBefore(bar, inputEl.nextSibling);
  } else {
    els.root.appendChild(bar);
  }
  els.toneBar = bar;
}

function updateToneBarVisibility() {
  if (!els.toneBar || !els.input) return;
  const has = (els.input.value || "").trim().length > 0;
  els.toneBar.hidden = !has;
  // Reset undo stack si el user borró todo a mano.
  if (!has && toneUndoStack.length) toneUndoStack = [];
  refreshUndoBtn();
}

function refreshUndoBtn() {
  if (!els.toneBar) return;
  const undo = els.toneBar.querySelector(".wa-tone-undo");
  if (!undo) return;
  undo.hidden = toneUndoStack.length === 0;
}

async function applyTone(tone) {
  if (!els.input || !els.toneBar) return;
  const text = (els.input.value || "").trim();
  if (!text) return;
  const chips = els.toneBar.querySelectorAll(".wa-tone-chip");
  const target = els.toneBar.querySelector(`[data-tone="${tone}"]`);
  // UI: disable mientras corre + highlight el target con shimmer.
  chips.forEach((c) => { c.disabled = true; });
  if (target) target.classList.add("loading");
  try {
    const data = await toneShift(text, tone);
    if (!data || !data.shifted) throw new Error("respuesta vacía");
    // Push undo del texto previo.
    toneUndoStack.push(text);
    if (toneUndoStack.length > 5) toneUndoStack.shift();
    els.input.value = data.shifted;
    autosize();
    refreshUndoBtn();
    // Si el LLM devolvió el mismo texto (noop), feedback visual.
    if (data.noop && target) {
      target.classList.add("noop");
      setTimeout(() => target.classList.remove("noop"), 1200);
    }
  } catch (e) {
    console.error("[wa-composer] tone-shift failed", e);
    if (target) target.classList.add("error");
    setTimeout(() => target?.classList.remove("error"), 1500);
  } finally {
    chips.forEach((c) => { c.disabled = false; });
    if (target) target.classList.remove("loading");
    els.input.focus();
  }
}

function undoTone() {
  if (!els.input || toneUndoStack.length === 0) return;
  const prev = toneUndoStack.pop();
  els.input.value = prev;
  autosize();
  refreshUndoBtn();
  els.input.focus();
}
