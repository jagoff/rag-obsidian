// wzp · Anticipador — drawer "✨ hoy" anclado al sidebar header.
//
// Surface el top-N del daemon `com.fer.obsidian-rag-anticipate` adentro del
// cliente para que el user vea + actúe sobre pendientes sin abrir RagNet ni
// la CLI. Backend: GET /api/wa/anticipate/today (cache 120s in-process).
//
// Acciones por candidate:
//   · borrador  → abre el chat del `target_jid` + pre-llena composer con
//                 `draft` (LLM-composed con dossier + estilo histórico).
//   · snooze    → POST /api/anticipate/feedback {rating:"mute"} +
//                 desaparece del drawer.
//   · ignorar   → POST /api/anticipate/feedback {rating:"negative"} +
//                 desaparece.
//
// Si el candidate no trae draft pero sí target_jid → botón [abrir chat].
// Si no trae target_jid (ej. anticipate-echo) → sólo snooze/ignore.

import * as api from "./wa-api.js";

const $ = (id) => document.getElementById(id);

let _root = null;
let _trigger = null;
let _open = false;
let _candidates = [];
let _onChatSelect = null;
let _bgTimer = null;

const ICON_BY_KIND = {
  "anticipate-calendar": "⏰",
  "anticipate-echo": "📓",
  "anticipate-commitment": "🪨",
};

const FRIENDLY_KIND = {
  "anticipate-calendar": "agenda",
  "anticipate-echo": "resonancia",
  "anticipate-commitment": "pendiente",
};

export function init({ onChatSelect } = {}) {
  _onChatSelect = onChatSelect || (() => {});
  mountTrigger();
  mountDrawer();
  document.addEventListener("keydown", onGlobalKeydown);
  document.addEventListener("click", onDocumentClick, true);
  // Refresh count en background — primera vez ASAP, después cada 5min
  refreshCount();
  _bgTimer = setInterval(refreshCount, 5 * 60 * 1000);
}

function mountTrigger() {
  const headerRight = document.querySelector(".wa-header-right");
  if (!headerRight) return;
  const btn = document.createElement("button");
  btn.className = "wa-anticipate-trigger";
  btn.id = "wa-anticipate-trigger";
  btn.type = "button";
  btn.title = "anticipador · pendientes inteligentes";
  btn.setAttribute("aria-label", "abrir anticipador");
  btn.innerHTML = `
    <span class="wa-anticipate-sparkle" aria-hidden="true">✨</span>
    <span class="wa-anticipate-count" id="wa-anticipate-count" hidden>0</span>
  `;
  // Insertar como primer hijo (queda a la izquierda del theme toggle).
  headerRight.insertBefore(btn, headerRight.firstChild);
  _trigger = btn;
  btn.addEventListener("click", toggle);
}

function mountDrawer() {
  const panel = document.createElement("section");
  panel.className = "wa-anticipate-drawer";
  panel.id = "wa-anticipate-drawer";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Pendientes del día");
  panel.hidden = true;
  panel.innerHTML = `
    <header class="wa-anticipate-header">
      <h2>✨ Hoy</h2>
      <button class="wa-anticipate-close" type="button" aria-label="cerrar">✕</button>
    </header>
    <div class="wa-anticipate-body" id="wa-anticipate-body" aria-live="polite">
      <div class="wa-anticipate-loading">◜ revisando…</div>
    </div>
    <footer class="wa-anticipate-footer">
      <span class="wa-anticipate-hint">se actualiza cada 10 min · <kbd>esc</kbd> cierra</span>
    </footer>
  `;
  document.body.appendChild(panel);
  _root = panel;
  panel.querySelector(".wa-anticipate-close").addEventListener("click", close);
}

async function toggle() {
  if (_open) close();
  else await open();
}

async function open() {
  if (_open) return;
  _open = true;
  _root.hidden = false;
  // Frame para que la transition arranque desde el estado hidden
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
  if (!_open) return;
  if (!_root) return;
  if (_root.contains(ev.target)) return;
  if (_trigger && _trigger.contains(ev.target)) return;
  close();
}

async function refresh() {
  const body = $("wa-anticipate-body");
  if (!body) return;
  body.innerHTML = `<div class="wa-anticipate-loading">◜ revisando…</div>`;
  try {
    const data = await api.fetchAnticipateToday({ limit: 3, minScore: 0.30 });
    _candidates = data.candidates || [];
    updateCount(_candidates.length, !!data.disabled);
    render(data);
  } catch (e) {
    console.warn("[wa-anticipate] fetch failed", e);
    body.innerHTML = `<div class="wa-anticipate-error">no se pudo revisar — reintentá en un rato</div>`;
  }
}

async function refreshCount() {
  // Background — sólo el badge del trigger. Mismo fetch + cache server-side.
  try {
    const data = await api.fetchAnticipateToday({ limit: 3, minScore: 0.30 });
    _candidates = data.candidates || [];
    updateCount(_candidates.length, !!data.disabled);
  } catch (e) {
    // silent — el trigger queda sin badge
  }
}

function updateCount(n, disabled) {
  const badge = $("wa-anticipate-count");
  if (!badge) return;
  if (disabled || !n) {
    badge.hidden = true;
    _trigger?.classList.remove("has-candidates");
  } else {
    badge.textContent = String(n);
    badge.hidden = false;
    _trigger?.classList.add("has-candidates");
  }
}

function render(data) {
  const body = $("wa-anticipate-body");
  if (!body) return;
  if (data.disabled) {
    body.innerHTML = `<div class="wa-anticipate-empty">anticipador desactivado · <code>RAG_ANTICIPATE_DISABLED=1</code></div>`;
    return;
  }
  if (!_candidates.length) {
    body.innerHTML = `<div class="wa-anticipate-empty">nada pendiente por ahora</div>`;
    return;
  }
  body.innerHTML = _candidates.map(renderCard).join("");
  wireCardActions(body);
}

function renderCard(c) {
  const icon = ICON_BY_KIND[c.kind] || "✨";
  const score = Math.round((c.score || 0) * 100);
  const safeMsg = escapeHtml(c.message || "").slice(0, 360);
  const safeReason = escapeHtml(c.reason || "");
  const target = c.target_name ? ` · ${escapeHtml(c.target_name)}` : "";
  const friendly = escapeHtml(FRIENDLY_KIND[c.kind] || c.kind || "señal");
  const hasDraft = !!(c.draft && c.target_jid);
  const hasJid = !!c.target_jid;
  const sourceNote = c.source_note || "";
  let primaryBtn = "";
  if (hasDraft) {
    primaryBtn = `<button class="wa-anticipate-action primary" data-action="draft">✍ borrador</button>`;
  } else if (hasJid) {
    primaryBtn = `<button class="wa-anticipate-action primary" data-action="open">→ abrir chat</button>`;
  }
  const sourceLink = sourceNote
    ? `<a href="obsidian://open?file=${encodeURIComponent(sourceNote)}" class="wa-anticipate-source" target="_blank" title="Ver fuente en vault">📄 fuente</a>`
    : "";
  return `
    <article class="wa-anticipate-card" data-dedup="${escapeAttr(c.dedup_key)}"
             data-jid="${escapeAttr(c.target_jid || "")}"
             data-kind="${escapeAttr(c.kind)}">
      <header class="wa-anticipate-card-head">
        <span class="wa-anticipate-icon" aria-hidden="true">${icon}</span>
        <span class="wa-anticipate-kind">${friendly}${target}</span>
        <span class="wa-anticipate-score" title="score">${score}%</span>
      </header>
      <p class="wa-anticipate-msg">${safeMsg}</p>
      ${safeReason ? `<p class="wa-anticipate-reason">${safeReason}</p>` : ""}
      ${sourceLink ? `<p class="wa-anticipate-source-row">${sourceLink}</p>` : ""}
      <div class="wa-anticipate-actions">
        ${primaryBtn}
        <button class="wa-anticipate-action" data-action="snooze">snooze 24h</button>
        <button class="wa-anticipate-action subtle" data-action="ignore">ignorar siempre</button>
      </div>
    </article>
  `;
}

function wireCardActions(root) {
  root.querySelectorAll(".wa-anticipate-action").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const action = btn.dataset.action;
      const card = btn.closest(".wa-anticipate-card");
      if (!card) return;
      const dedup = card.dataset.dedup;
      const jid = card.dataset.jid || "";
      const c = _candidates.find((x) => x.dedup_key === dedup);

      // Optimistic dismiss para snooze/ignore. Para draft/open salimos del
      // drawer así que no hace falta animar dismiss.
      if (action === "snooze" || action === "ignore") {
        card.classList.add("dismissed");
      }

      try {
        if (action === "draft") {
          await openWithDraft(jid, c?.draft || "");
          api.submitAnticipateFeedback(dedup, "positive", "draft_opened").catch(() => {});
          close();
        } else if (action === "open") {
          await openWithDraft(jid, "");
          api.submitAnticipateFeedback(dedup, "positive", "chat_opened").catch(() => {});
          close();
        } else if (action === "snooze") {
          await api.submitAnticipateFeedback(dedup, "mute", "snooze_24h");
          dismissCard(card);
        } else if (action === "ignore") {
          await api.submitAnticipateFeedback(dedup, "negative", "ignore_forever");
          dismissCard(card);
        }
      } catch (e) {
        console.warn("[wa-anticipate] action failed", action, e);
        card.classList.remove("dismissed");
      }
    });
  });
}

function dismissCard(card) {
  setTimeout(() => {
    card.remove();
    const remaining = document.querySelectorAll(".wa-anticipate-card").length;
    if (!remaining) {
      const body = $("wa-anticipate-body");
      if (body) body.innerHTML = `<div class="wa-anticipate-empty">nada pendiente por ahora</div>`;
    }
    updateCount(remaining, false);
  }, 220);
}

async function openWithDraft(jid, draft) {
  if (!jid) return;
  _onChatSelect(jid);
  if (!draft) return;
  // Pequeño delay para que thread.open() resuelva y composer.setActiveChat
  // termine de resetear el input — entonces escribimos el draft.
  setTimeout(() => {
    const input = document.getElementById("wa-composer-input");
    if (!input) return;
    input.value = draft;
    input.focus();
    // Mover caret al final
    const len = draft.length;
    try { input.setSelectionRange(len, len); } catch (_) {}
    // Disparar input event para que autosize() del composer corra
    input.dispatchEvent(new Event("input", { bubbles: true }));
    // Marcador visual sutil
    input.classList.add("wa-prefilled-from-anticipator");
    setTimeout(() => input.classList.remove("wa-prefilled-from-anticipator"), 2500);
  }, 180);
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
