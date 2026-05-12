// wzp · Promise Tracker — chip "🪨 N" en thread header.
//
// Click → drawer con lista de promesas pending del JID activo, con
// acciones [✓ cumplida] / [✗ cancelar]. Detección runtime: cualquier
// outbound /api/wa/send que matchee `_has_promise_hint` rioplatense
// inserta row en `rag_promises`. Drawer las lee de
// `GET /api/wa/promises?jid=<jid>&status=pending`.

import * as api from "./wa-api.js";

let _root = null;
let _open = false;
let _activeJid = null;
let _activeName = null;
let _promises = [];

export function init() {
  mountDrawer();
  document.addEventListener("keydown", onGlobalKeydown);
  document.addEventListener("click", onDocumentClick, true);
}

export function mountTrigger(jid, name) {
  _activeJid = jid;
  _activeName = name || "";
  const metaEl = document.querySelector(".wa-thread-meta");
  if (!metaEl) return;
  let btn = document.getElementById("wa-promises-trigger");
  if (!btn) {
    btn = document.createElement("button");
    btn.id = "wa-promises-trigger";
    btn.className = "wa-promises-trigger";
    btn.type = "button";
    btn.setAttribute("aria-label", "promesas con este contacto");
    btn.title = "promesas · seguimiento de compromisos";
    btn.innerHTML = `<span aria-hidden="true">🪨</span><span class="wa-promises-count" id="wa-promises-count" hidden>0</span>`;
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      toggle();
    });
    metaEl.appendChild(btn);
  }
  // Refresh count para el nuevo JID
  refreshCount();
}

function mountDrawer() {
  const panel = document.createElement("aside");
  panel.id = "wa-promises-drawer";
  panel.className = "wa-promises-drawer";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Promesas");
  panel.hidden = true;
  panel.innerHTML = `
    <header class="wa-promises-header">
      <h2 id="wa-promises-title">🪨 Promesas</h2>
      <button class="wa-promises-close" type="button" aria-label="cerrar">✕</button>
    </header>
    <div class="wa-promises-body" id="wa-promises-body" aria-live="polite">
      <div class="wa-promises-loading">◜ revisando…</div>
    </div>
    <footer class="wa-promises-footer">
      <span class="wa-promises-hint">detección por regex · <kbd>esc</kbd> cierra</span>
    </footer>
  `;
  document.body.appendChild(panel);
  _root = panel;
  panel.querySelector(".wa-promises-close").addEventListener("click", close);
}

async function toggle() {
  if (_open) close();
  else await open();
}

async function open() {
  if (_open || !_activeJid) return;
  _open = true;
  _root.hidden = false;
  requestAnimationFrame(() => _root.classList.add("is-open"));
  await refresh();
}

function close() {
  if (!_open) return;
  _open = false;
  _root.classList.remove("is-open");
  setTimeout(() => { if (!_open) _root.hidden = true; }, 220);
}

function onGlobalKeydown(ev) {
  if (ev.key === "Escape" && _open) {
    ev.preventDefault();
    close();
  }
}

function onDocumentClick(ev) {
  if (!_open || !_root) return;
  if (_root.contains(ev.target)) return;
  const trigger = document.getElementById("wa-promises-trigger");
  if (trigger && trigger.contains(ev.target)) return;
  close();
}

async function refresh() {
  const body = document.getElementById("wa-promises-body");
  const title = document.getElementById("wa-promises-title");
  if (!body || !_activeJid) return;
  if (title) {
    title.textContent = _activeName
      ? `🪨 Promesas con ${_activeName}`
      : "🪨 Promesas";
  }
  body.innerHTML = `<div class="wa-promises-loading">◜ revisando…</div>`;
  try {
    const data = await api.fetchWaPromises({ jid: _activeJid, status: "pending", limit: 20 });
    _promises = data.promises || [];
    render();
    updateCount(_promises.length);
  } catch (e) {
    console.warn("[wa-promises] fetch failed", e);
    body.innerHTML = `<div class="wa-promises-error">no se pudo revisar</div>`;
  }
}

async function refreshCount() {
  if (!_activeJid) return;
  try {
    const data = await api.fetchWaPromises({ jid: _activeJid, status: "pending", limit: 20 });
    _promises = data.promises || [];
    updateCount(_promises.length);
  } catch (e) {
    // silent
  }
}

function updateCount(n) {
  const badge = document.getElementById("wa-promises-count");
  const trigger = document.getElementById("wa-promises-trigger");
  if (!badge || !trigger) return;
  if (!n) {
    badge.hidden = true;
    trigger.classList.remove("has-promises");
  } else {
    badge.textContent = String(n);
    badge.hidden = false;
    trigger.classList.add("has-promises");
  }
}

function render() {
  const body = document.getElementById("wa-promises-body");
  if (!body) return;
  if (!_promises.length) {
    body.innerHTML = `<div class="wa-promises-empty">🌤 sin promesas pending</div>`;
    return;
  }
  body.innerHTML = _promises.map(renderCard).join("");
  wireCardActions(body);
}

function renderCard(p) {
  const text = escapeHtml(p.promise_text || "");
  const when = formatRelative(p.ts);
  const dir = p.direction === "outbound" ? "⬆ vos" : "⬇ él/ella";
  const dueStatus = computeDueStatus(p.due_ts);
  const dueBadge = dueStatus
    ? `<span class="wa-promises-due ${dueStatus.cls}" title="${escapeHtml(dueStatus.fullText)}">${escapeHtml(dueStatus.shortText)}</span>`
    : "";
  return `
    <article class="wa-promises-card ${p.direction} ${dueStatus ? dueStatus.cls : ""}" data-id="${p.id}">
      <header>
        <span class="wa-promises-dir">${dir}</span>
        <span class="wa-promises-when">${when}</span>
      </header>
      <p class="wa-promises-text">"${text}"</p>
      ${dueBadge ? `<div class="wa-promises-meta">${dueBadge}</div>` : ""}
      <div class="wa-promises-actions">
        <button class="wa-promises-action primary" data-action="resolve">✓ cumplida</button>
        <button class="wa-promises-action subtle" data-action="cancel">✗ cancelar</button>
      </div>
    </article>
  `;
}

function computeDueStatus(dueTs) {
  if (!dueTs) return null;
  let due;
  try {
    due = new Date(dueTs).getTime();
    if (Number.isNaN(due)) return null;
  } catch {
    return null;
  }
  const now = Date.now();
  const delta = due - now;
  const absHours = Math.abs(delta) / 3600000;
  const fullText = new Date(due).toLocaleString("es-AR", {
    weekday: "short", day: "numeric", month: "short",
    hour: "2-digit", minute: "2-digit",
  });
  if (delta < 0) {
    // overdue
    const txt = absHours < 24
      ? `⏰ vencida hace ${Math.round(absHours)}h`
      : `⏰ vencida hace ${Math.round(absHours / 24)}d`;
    return { cls: "due-overdue", shortText: txt, fullText };
  }
  if (absHours < 24) {
    return {
      cls: "due-soon",
      shortText: `⏳ vence en ${Math.round(absHours)}h`,
      fullText,
    };
  }
  if (absHours < 168) {
    return {
      cls: "due-future",
      shortText: `📅 ${Math.round(absHours / 24)}d`,
      fullText,
    };
  }
  return { cls: "due-future", shortText: `📅 ${fullText}`, fullText };
}

function wireCardActions(root) {
  root.querySelectorAll(".wa-promises-action").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const action = btn.dataset.action;
      const card = btn.closest(".wa-promises-card");
      const id = parseInt(card?.dataset.id || "0", 10);
      if (!id) return;
      card.classList.add("dismissed");
      try {
        if (action === "resolve") {
          await api.resolveWaPromise(id);
        } else if (action === "cancel") {
          await api.cancelWaPromise(id);
        }
        setTimeout(() => {
          _promises = _promises.filter((p) => p.id !== id);
          render();
          updateCount(_promises.length);
        }, 220);
      } catch (e) {
        console.warn("[wa-promises] action failed", action, e);
        card.classList.remove("dismissed");
      }
    });
  });
}

function formatRelative(ts) {
  if (!ts) return "";
  let d;
  try {
    d = new Date(ts).getTime() / 1000;
  } catch {
    return ts;
  }
  const delta = Math.max(0, Math.floor(Date.now() / 1000) - d);
  if (delta < 60) return "ahora";
  if (delta < 3600) return `hace ${Math.floor(delta / 60)} min`;
  if (delta < 86400) return `hace ${Math.floor(delta / 3600)} h`;
  return `hace ${Math.floor(delta / 86400)} d`;
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
