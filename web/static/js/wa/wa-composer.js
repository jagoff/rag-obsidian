// Composer del thread — textarea autosize + Enter-to-send + reply preview.
// Optimistic UI: insertamos burbuja "pending" inmediatamente; cuando
// el SSE trae el `new_message` real con el mismo content+jid+~5s, el
// thread la reemplaza por la versión persistida del bridge.

import { sendText } from "./wa-api.js";

const els = {
  root: null,
  input: null,
  btn: null,
  replyBar: null,
};

let currentJID = null;
let pendingReply = null; // {message_id, original_text, sender_jid}
let onOptimisticInsertCb = null;

export function init({ rootEl, onOptimisticInsert }) {
  els.root = rootEl;
  if (!rootEl) return;
  els.input = rootEl.querySelector("textarea");
  els.btn = rootEl.querySelector(".wa-send-btn");
  onOptimisticInsertCb = onOptimisticInsert;

  // Inserto bar de reply preview justo arriba del textarea.
  els.replyBar = document.createElement("div");
  els.replyBar.className = "wa-reply-bar";
  els.replyBar.hidden = true;
  rootEl.insertBefore(els.replyBar, els.input);

  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  els.btn.addEventListener("click", submit);
}

export function setActiveChat(jid) {
  currentJID = jid;
  pendingReply = null;
  hideReplyBar();
  if (els.input) {
    els.input.disabled = !jid;
    els.input.value = "";
    els.input.placeholder = jid ? "Escribí un mensaje…" : "Elegí un chat para escribir";
    autosize();
  }
  if (els.btn) els.btn.disabled = !jid;
  if (els.root) els.root.hidden = !jid;
}

export function setReply(replyTo) {
  pendingReply = replyTo;
  if (!els.replyBar) return;
  if (!replyTo) {
    hideReplyBar();
    return;
  }
  els.replyBar.hidden = false;
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
    els.replyBar.hidden = true;
    els.replyBar.innerHTML = "";
  }
}

function autosize() {
  if (!els.input) return;
  els.input.style.height = "auto";
  const next = Math.min(els.input.scrollHeight, 120);
  els.input.style.height = `${next}px`;
}

async function submit() {
  if (!currentJID || !els.input || !els.btn) return;
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
  els.btn.disabled = true;

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
    els.btn.disabled = false;
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
