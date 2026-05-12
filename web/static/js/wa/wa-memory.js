// wzp · Memoria Universal — drawer "🧠 Recordar" anclado al thread header.
//
// Dado el JID del chat activo, surface lo que sabe Ra:
//   · Notas del vault que mencionan al contacto (top-N de multi_retrieve).
//   · Últimos N mensajes WA del propio thread (refresca memoria reciente).
//   · Summary armado server-side con counts + última actividad.
//
// Endpoint: GET /api/wa/memory/<jid>?max_notes=5&max_wa=5 (cache 300s).
//
// El botón se monta dinámicamente al abrir cada chat (thread.open() llama
// `mountTrigger(jid, name)`), porque el header se re-renderiza con cada
// chat seleccionado.

import * as api from "./wa-api.js";

let _root = null;
let _open = false;
let _activeJid = null;
let _activeName = null;

export function init() {
  mountDrawer();
  document.addEventListener("keydown", onGlobalKeydown);
  document.addEventListener("click", onDocumentClick, true);
}

// Llamada cada vez que `thread.open(jid)` resuelve. Inserta el botón 🧠
// en `.wa-thread-meta` (junto al nombre del chat) si todavía no existe.
export function mountTrigger(jid, name) {
  _activeJid = jid;
  _activeName = name || "";
  const metaEl = document.querySelector(".wa-thread-meta");
  if (!metaEl) return;
  let btn = document.getElementById("wa-memory-trigger");
  if (!btn) {
    btn = document.createElement("button");
    btn.id = "wa-memory-trigger";
    btn.className = "wa-memory-trigger";
    btn.type = "button";
    btn.setAttribute("aria-label", "abrir memoria");
    btn.title = "memoria · lo que sabe Ra sobre este contacto";
    btn.innerHTML = `<span aria-hidden="true">🧠</span><span class="wa-memory-trigger-label">recordar</span>`;
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      toggle();
    });
    metaEl.appendChild(btn);
  }
}

function mountDrawer() {
  const panel = document.createElement("aside");
  panel.id = "wa-memory-drawer";
  panel.className = "wa-memory-drawer";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Memoria universal");
  panel.hidden = true;
  panel.innerHTML = `
    <header class="wa-memory-header">
      <h2 id="wa-memory-title">🧠 Recordar</h2>
      <button class="wa-memory-close" type="button" aria-label="cerrar">✕</button>
    </header>
    <div class="wa-memory-body" id="wa-memory-body" aria-live="polite">
      <div class="wa-memory-loading">◜ buscando…</div>
    </div>
    <footer class="wa-memory-footer">
      <span class="wa-memory-hint">cache 5 min · <kbd>esc</kbd> cierra</span>
    </footer>
  `;
  document.body.appendChild(panel);
  _root = panel;
  panel.querySelector(".wa-memory-close").addEventListener("click", close);
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
  const trigger = document.getElementById("wa-memory-trigger");
  if (trigger && trigger.contains(ev.target)) return;
  close();
}

async function refresh() {
  const body = document.getElementById("wa-memory-body");
  const title = document.getElementById("wa-memory-title");
  if (!body || !_activeJid) return;
  if (title) {
    title.textContent = _activeName
      ? `🧠 sobre ${_activeName}`
      : "🧠 Recordar";
  }
  body.innerHTML = `<div class="wa-memory-loading">◜ buscando…</div>`;
  try {
    const data = await api.fetchWaMemory(_activeJid, { maxNotes: 5, maxWa: 5 });
    render(data);
  } catch (e) {
    console.warn("[wa-memory] fetch failed", e);
    body.innerHTML = `<div class="wa-memory-error">no se pudo buscar — reintentá en un rato</div>`;
  }
}

function render(data) {
  const body = document.getElementById("wa-memory-body");
  if (!body) return;
  const name = escapeHtml(data.name || _activeName || "este contacto");
  const summary = escapeHtml(data.summary || "");
  const notes = (data.notes || []).map(renderNote).join("");
  const wa = (data.wa_recent || []).map(renderWaMsg).join("");

  body.innerHTML = `
    <section class="wa-memory-section wa-memory-summary">
      <p class="wa-memory-blurb">${summary}</p>
    </section>

    <section class="wa-memory-section">
      <h3 class="wa-memory-section-title">notas del vault</h3>
      ${notes || `<p class="wa-memory-empty">sin notas que mencionen a ${name}</p>`}
    </section>

    <section class="wa-memory-section">
      <h3 class="wa-memory-section-title">últimos mensajes</h3>
      ${wa || `<p class="wa-memory-empty">sin mensajes recientes</p>`}
    </section>
  `;
}

function renderNote(n) {
  const title = escapeHtml(n.title || n.path || "(sin título)");
  const snippet = escapeHtml(n.snippet || "");
  const score = Math.round((n.score || 0) * 100);
  const path = escapeAttr(n.path || "");
  const when = formatRelative(n.mtime || 0);
  return `
    <article class="wa-memory-note">
      <header>
        <span class="wa-memory-note-title" title="${path}">${title}</span>
        <span class="wa-memory-note-score" title="relevancia">${score}%</span>
      </header>
      <p class="wa-memory-note-snippet">${snippet}</p>
      ${when ? `<span class="wa-memory-note-when">${when}</span>` : ""}
    </article>
  `;
}

function renderWaMsg(m) {
  const content = escapeHtml(m.content || "");
  const when = formatRelative(m.ts || 0);
  const author = m.from_me ? "vos" : "él/ella";
  return `
    <article class="wa-memory-wa ${m.from_me ? "from-me" : "from-them"}">
      <header>
        <span class="wa-memory-wa-author">${author}</span>
        <span class="wa-memory-wa-when">${when}</span>
      </header>
      <p>${content}</p>
    </article>
  `;
}

function formatRelative(ts) {
  if (!ts) return "";
  const delta = Math.max(0, Math.floor(Date.now() / 1000) - Number(ts));
  if (delta < 60) return "ahora";
  if (delta < 3600) return `hace ${Math.floor(delta / 60)} min`;
  if (delta < 86400) return `hace ${Math.floor(delta / 3600)} h`;
  const days = Math.floor(delta / 86400);
  if (days < 30) return `hace ${days} d`;
  return new Date(Number(ts) * 1000).toLocaleDateString("es-AR", {
    day: "numeric", month: "short", year: "numeric",
  });
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttr(s) {
  return escapeHtml(s);
}
