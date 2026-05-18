/**
 * messages-render.mjs — Phase W4-phase-3 (2026-05-09)
 *
 * Implementaciones reales de todas las funciones de renderizado de mensajes
 * del chat. En esta fase las funciones viven aquí con código real y ya no
 * delegan a window.*. app.mjs (que corre después) hace override de las
 * versiones en app.js con estas implementaciones.
 *
 * Dependencias:
 *   - state.mjs  → state.inflightSideFetches, state.pending
 *   - No depende de send() — appendFeedback recibe sendFn como parámetro
 */

import { state } from "./state.mjs";

// ── Helpers DOM y utilitarios internos ────────────────────────────────────

/** Crea un elemento DOM con clase y texto opcionales. */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Escapa HTML para uso seguro en innerHTML. */
export function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** URL obsidian:// para abrir una nota del vault. */
export function obsidianUrl(filePath) {
  return "obsidian://open?file=" + encodeURIComponent(filePath);
}

/**
 * Convierte un URI whatsapp://<jid> a un wa.me link universal.
 * Devuelve "" para grupos o inputs inválidos.
 */
export function waHref(uri) {
  if (!uri || typeof uri !== "string") return "";
  let jid = uri;
  if (jid.indexOf("whatsapp://") === 0) {
    jid = jid.slice("whatsapp://".length);
    const slash = jid.indexOf("/");
    if (slash >= 0) jid = jid.slice(0, slash);
  }
  if (jid.indexOf("@g.us") >= 0) return "";
  const phone = jid.split("@")[0];
  if (/^\d{6,}$/.test(phone)) return "https://wa.me/" + phone;
  return "";
}

// ── Comportamiento a11y: prefers-reduced-motion ───────────────────────────

/** Respeta prefers-reduced-motion para scroll suave. */
export function smoothBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto" : "smooth";
}

// ── Turn y línea básica ───────────────────────────────────────────────────

/**
 * Crea un contenedor de turn y lo adjunta a #messages.
 * @returns {HTMLElement}
 */
export function appendTurn() {
  const messagesEl = document.getElementById("messages");
  if (!messagesEl) return document.createElement("div");
  const turn = el("div", "turn");
  messagesEl.appendChild(turn);
  return turn;
}

/**
 * Adjunta una línea de mensaje (user o rag) a un turn.
 * @param {HTMLElement} parent
 * @param {"user"|"rag"} role
 * @param {string} text
 * @returns {HTMLElement} — el span de texto
 */
export function appendLine(parent, role, text) {
  const line = el("div", "line");
  line.appendChild(el("span", `prompt ${role}`, role === "user" ? "tu ›" : "rag ›"));
  const t = el("span", `text ${role}`);
  t.textContent = text || "";
  line.appendChild(t);
  parent.appendChild(line);
  return t;
}

/**
 * Adjunta una línea de metadata (timestamps, scores, etc.) a un turn.
 * @param {HTMLElement} parent
 * @param {string[]} bits
 */
export function appendMeta(parent, bits) {
  const m = el("div", "meta", "  " + bits.join(" · "));
  parent.appendChild(m);
}

// ── Scroll ────────────────────────────────────────────────────────────────

/** Scroll suave al fondo del chat. */
export function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: smoothBehavior() });
}

// ── Confidence badge ──────────────────────────────────────────────────────

/**
 * Pill de confianza calibrado contra la distribución real de top_score.
 * Thresholds: alta ≥ 0.50, media ≥ 0.10, baja < 0.10.
 * @param {number} score
 * @returns {HTMLElement}
 */
export function confidenceBadge(score) {
  const s = Number.isFinite(score) ? score : 0;
  let level = "low";
  let label = "baja";
  if (s >= 0.50) { level = "high"; label = "alta"; }
  else if (s >= 0.10) { level = "mid"; label = "media"; }
  const span = el("span", `conf-pill conf-${level}`);
  span.title = `score top rerank: ${s.toFixed(2)}`;
  span.textContent = `confianza ${label} · ${s.toFixed(2)}`;
  return span;
}

// ── Toast ─────────────────────────────────────────────────────────────────

/**
 * Toast notification (ok / err / info). Auto-dismiss tras `ms` ms.
 * @param {string} message
 * @param {object|string} opts — { kind, ms, action } o string kind
 * @returns {{ dismiss: Function }}
 */
export function showToast(message, opts = {}) {
  if (typeof opts === "string") opts = { kind: opts };
  const { kind = "ok", action = null } = opts;
  const ms = opts.ms ?? (action ? 10000 : 4000);

  let container = document.getElementById("toast-container");
  if (!container) {
    container = el("div", "toast-container");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const toast = el("div", `toast toast-${kind}`);
  const msgEl = el("span", "toast-msg", message);
  toast.appendChild(msgEl);

  let dismissTimer = null;
  const dismiss = () => {
    if (!toast.isConnected) return;
    if (dismissTimer) clearTimeout(dismissTimer);
    toast.classList.remove("toast-visible");
    toast.addEventListener("transitionend", () => toast.remove(), { once: true });
    setTimeout(() => toast.remove(), 500);
  };

  if (action) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "toast-action";
    btn.textContent = action.label;
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      btn.disabled = true;
      try { await action.onClick(); } finally { dismiss(); }
    });
    toast.appendChild(btn);
  }

  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("toast-visible"));
  dismissTimer = setTimeout(dismiss, ms);
  return { dismiss };
}

// ── Copy helpers ──────────────────────────────────────────────────────────

/**
 * Copia texto al portapapeles. Clipboard API con fallback a execCommand.
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export async function copyTextToClipboard(text) {
  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try { await navigator.clipboard.writeText(text); return true; } catch (_) {}
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:-1000px;left:-1000px;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, text.length);
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_) { return false; }
}

/**
 * Construye el markdown de exportación para el portapapeles.
 * @param {string} question
 * @param {string} answer
 * @param {Array} sources
 * @returns {string}
 */
export function buildMarkdownExport(question, answer, sources) {
  const parts = [];
  if (question && question.trim()) parts.push(`## Pregunta\n\n${question.trim()}`);
  if (answer && answer.trim()) parts.push(`## Respuesta\n\n${answer.trim()}`);
  if (Array.isArray(sources) && sources.length) {
    const seen = new Set();
    const lines = ["## Fuentes", ""];
    for (const s of sources) {
      if (!s || !s.file || seen.has(s.file)) continue;
      seen.add(s.file);
      const note = s.note || s.file.replace(/\.md$/, "").split("/").pop();
      const score = Number.isFinite(s.score) ? ` · ${(s.score >= 0 ? "+" : "") + s.score.toFixed(1)}` : "";
      const isExternalSrc = /^https?:\/\//i.test(s.file);
      const isModelSrc = (s.source_kind || s.sourceKind) === "model" || /^model:\/\//i.test(s.file);
      if (isModelSrc) {
        lines.push(`- Modelo — ${s.folder || "conocimiento general"}${score}`);
      } else if (isExternalSrc) {
        const label = note || s.folder || "link";
        lines.push(`- [${label}](${s.file})${score}`);
      } else {
        lines.push(`- [[${note}]] — \`${s.file}\`${score}`);
      }
    }
    parts.push(lines.join("\n"));
  }
  parts.push(`_via rag · ${new Date().toISOString().slice(0, 19).replace("T", " ")}_`);
  return parts.join("\n\n");
}

/**
 * Botón de copia (markdown export).
 * @param {HTMLElement} parent
 * @param {Function} getText — devuelve el texto a copiar
 * @returns {HTMLElement}
 */
export function appendCopyButton(parent, getText) {
  const COPY_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="13" height="13" rx="2"/>
    <path d="M5 15V5a2 2 0 0 1 2-2h10"/>
  </svg>`;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg-action copy-btn";
  btn.setAttribute("aria-label", "copiar respuesta");
  btn.title = "copiar markdown";
  btn.innerHTML = `${COPY_SVG}<span class="msg-action-label" aria-live="polite" aria-atomic="true">copiar</span>`;
  btn.addEventListener("click", async () => {
    const text = typeof getText === "function" ? getText() : "";
    const label = btn.querySelector(".msg-action-label");
    if (!text || !text.trim()) {
      btn.classList.add("err");
      if (label) label.textContent = "sin texto";
      setTimeout(() => { btn.classList.remove("err"); if (label) label.textContent = "copiar"; }, 1400);
      return;
    }
    const ok = await copyTextToClipboard(text);
    if (ok) {
      btn.classList.add("done");
      if (label) label.textContent = "copiado";
      setTimeout(() => { btn.classList.remove("done"); if (label) label.textContent = "copiar"; }, 1200);
    } else {
      btn.classList.add("err");
      if (label) label.textContent = "falló — ⌘C manual";
      setTimeout(() => { btn.classList.remove("err"); if (label) label.textContent = "copiar"; }, 2400);
    }
  });
  parent.appendChild(btn);
  return btn;
}

// ── Feedback ──────────────────────────────────────────────────────────────

const THUMB_UP_SVG = `<svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M7 10v11"/>
  <path d="M7 10c2.2-2.5 3.4-4.6 3.6-6.4.1-.9.8-1.6 1.7-1.6.9 0 1.7.8 1.7 1.7 0 1.3-.3 2.5-.9 3.6-.2.3 0 .7.4.7h4.4a2 2 0 0 1 2 2.3l-1.2 7.6a2 2 0 0 1-2 1.7H7"/>
  <path d="M3 10h4v11H3z"/>
</svg>`;
const THUMB_DOWN_SVG = `<svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M17 14V3"/>
  <path d="M17 14c-2.2 2.5-3.4 4.6-3.6 6.4-.1.9-.8 1.6-1.7 1.6-.9 0-1.7-.8-1.7-1.7 0-1.3.3-2.5.9-3.6.2-.3 0-.7-.4-.7H6.1a2 2 0 0 1-2-2.3L5.3 6.1a2 2 0 0 1 2-1.7H17"/>
  <path d="M21 14h-4V3h4z"/>
</svg>`;
const REDO_SVG = `<svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M3 12a9 9 0 1 0 3-6.7"/>
  <polyline points="3 4 3 10 9 10"/>
</svg>`;

/**
 * Renderiza el panel de feedback (👍/👎/redo/copy) al pie de un turn.
 * @param {HTMLElement} parent
 * @param {{turn_id, q, paths, sources, session_id}} ctx
 * @param {Function} [sendFn] — función send(question, opts) para el botón redo
 * @returns {HTMLElement}
 */
export function appendFeedback(parent, ctx, sendFn) {
  const wrap = el("div", "feedback");
  const prompt = el("span", "feedback-prompt", "¿útil?");
  const up = document.createElement("button");
  up.type = "button";
  up.className = "fb-btn fb-up";
  up.setAttribute("aria-label", "útil");
  up.innerHTML = `${THUMB_UP_SVG}<span class="fb-label">útil</span>`;
  const down = document.createElement("button");
  down.type = "button";
  down.className = "fb-btn fb-down";
  down.setAttribute("aria-label", "no ayudó");
  down.innerHTML = `${THUMB_DOWN_SVG}<span class="fb-label">no ayudó</span>`;
  const status = el("span", "feedback-status", "");

  async function submit(rating, reason, correctivePath) {
    if (wrap.dataset.sent) return;
    wrap.dataset.sent = "1";
    up.disabled = true;
    down.disabled = true;
    (rating > 0 ? up : down).classList.add("picked");
    (rating > 0 ? down : up).classList.add("dimmed");
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turn_id: ctx.turn_id,
          rating,
          q: ctx.q,
          paths: ctx.paths,
          session_id: ctx.session_id,
          reason: reason || null,
          corrective_path: correctivePath || null,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let label = rating > 0 ? "  gracias — anotado" : "  anotado — seguimos afinando";
      if (correctivePath) label = "  anotado + corrección — ranker aprende";
      status.textContent = label;
      status.classList.add(rating > 0 ? "ok" : "warn");
    } catch (err) {
      status.textContent = "  no pude registrar";
      status.classList.add("err");
      delete wrap.dataset.sent;
      up.disabled = false;
      down.disabled = false;
      up.classList.remove("picked", "dimmed");
      down.classList.remove("picked", "dimmed");
    }
  }

  function openNegativeFeedback() {
    if (wrap.dataset.reasonOpen || wrap.dataset.sent) return;
    wrap.dataset.reasonOpen = "1";
    down.classList.add("picked");
    up.classList.add("dimmed");

    const row = el("div", "feedback-corrective");
    const candidates = [];
    const seen = new Set();
    for (const src of (ctx.sources || [])) {
      const p = src && src.file;
      if (!p || p.indexOf("://") !== -1 || seen.has(p)) continue;
      seen.add(p);
      candidates.push(src);
      if (candidates.length >= 5) break;
    }

    const hasCandidates = candidates.length > 0;
    row.appendChild(el("div", "fb-corr-header",
      hasCandidates
        ? "¿cuál era el path correcto? elegí uno o escribí el tuyo — opcional"
        : "¿qué faltó? (opcional)"
    ));

    let selectedPath = null;
    const cardNodes = [];

    if (hasCandidates) {
      const grid = el("div", "fb-corr-grid");
      candidates.forEach((src, idx) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "fb-corr-card";
        card.setAttribute("aria-pressed", "false");
        const score = Number.isFinite(src.score) ? src.score : null;
        const title = (src.title || src.file || `fuente ${idx + 1}`).toString();
        const path = src.file || "";
        card.innerHTML = `
          <span class="fb-corr-num">${idx + 1}</span>
          <span class="fb-corr-body">
            <span class="fb-corr-title">${escapeHtml(title)}</span>
            <span class="fb-corr-path">${escapeHtml(path)}</span>
          </span>
          ${score !== null ? `<span class="fb-corr-score">${score.toFixed(2)}</span>` : ""}
        `;
        card.addEventListener("click", () => {
          if (selectedPath === path) {
            selectedPath = null;
            card.classList.remove("picked");
            card.setAttribute("aria-pressed", "false");
          } else {
            selectedPath = path;
            cardNodes.forEach((c) => { c.classList.remove("picked"); c.setAttribute("aria-pressed", "false"); });
            noneBtn.classList.remove("picked");
            noneBtn.setAttribute("aria-pressed", "false");
            card.classList.add("picked");
            card.setAttribute("aria-pressed", "true");
          }
        });
        cardNodes.push(card);
        grid.appendChild(card);
      });

      const noneBtn = document.createElement("button");
      noneBtn.type = "button";
      noneBtn.className = "fb-corr-card fb-corr-none";
      noneBtn.setAttribute("aria-pressed", "false");
      noneBtn.innerHTML = `<span class="fb-corr-body"><span class="fb-corr-title">ninguna de estas</span><span class="fb-corr-path">el correcto no apareció entre las fuentes</span></span>`;
      noneBtn.addEventListener("click", () => {
        if (selectedPath === "__none__") {
          selectedPath = null;
          noneBtn.classList.remove("picked");
          noneBtn.setAttribute("aria-pressed", "false");
        } else {
          selectedPath = "__none__";
          cardNodes.forEach((c) => { c.classList.remove("picked"); c.setAttribute("aria-pressed", "false"); });
          noneBtn.classList.add("picked");
          noneBtn.setAttribute("aria-pressed", "true");
        }
      });
      grid.appendChild(noneBtn);
      row.appendChild(grid);
    }

    const field = document.createElement("input");
    field.type = "text";
    field.className = "fb-reason-input";
    field.placeholder = hasCandidates
      ? "…o pegá el path real (ej: 02-Areas/Salud/postura.md) — opcional"
      : "¿qué faltó? ej: falta la nota X, muy genérico";
    field.maxLength = 200;
    field.setAttribute("aria-label", "motivo (opcional)");

    const actions = el("div", "fb-corr-actions");
    const sendBtn = document.createElement("button");
    sendBtn.type = "button";
    sendBtn.className = "fb-text-btn";
    sendBtn.textContent = "enviar";

    const skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "fb-text-btn fb-text-muted";
    skipBtn.textContent = "omitir";

    async function commit() {
      const freeText = field.value.trim();
      let corrective = null;
      let reason = null;
      if (freeText && (freeText.indexOf("/") !== -1 || /\.md$/i.test(freeText))) {
        corrective = freeText;
      } else if (selectedPath && selectedPath !== "__none__") {
        corrective = selectedPath;
        if (freeText) reason = freeText;
      } else if (freeText) {
        reason = freeText;
      }
      row.remove();
      await submit(-1, reason, corrective);
    }

    function cancel() {
      row.remove();
      delete wrap.dataset.reasonOpen;
      down.classList.remove("picked");
      up.classList.remove("dimmed");
    }

    field.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); commit(); }
      else if (ev.key === "Escape") { cancel(); }
    });
    sendBtn.addEventListener("click", commit);
    skipBtn.addEventListener("click", commit);

    actions.appendChild(field);
    actions.appendChild(sendBtn);
    actions.appendChild(skipBtn);
    row.appendChild(actions);
    wrap.appendChild(row);
    (hasCandidates ? cardNodes[0] || field : field).focus();
  }

  // Redo button — usa sendFn (callback inyectado) para evitar circular import
  const redo = document.createElement("button");
  redo.type = "button";
  redo.className = "fb-btn fb-redo";
  redo.setAttribute("aria-label", "regenerar respuesta");
  redo.title = "regenerar respuesta (/redo para pista)";
  redo.innerHTML = `${REDO_SVG}<span class="fb-label">regenerar</span>`;
  redo.addEventListener("click", () => {
    if (wrap.dataset.sent) return;
    if (!ctx.turn_id) return;
    redo.disabled = true;
    redo.classList.add("picked");
    // Usar sendFn inyectado o fallback a window.send (compat con app.js)
    const fn = typeof sendFn === "function" ? sendFn : window.send;
    if (typeof fn === "function") fn("(redo)", { redo_turn_id: ctx.turn_id });
  });

  up.addEventListener("click", () => submit(1));
  down.addEventListener("click", openNegativeFeedback);

  wrap.appendChild(prompt);
  wrap.appendChild(up);
  wrap.appendChild(down);
  wrap.appendChild(redo);
  wrap.appendChild(status);
  parent.appendChild(wrap);
  return wrap;
}

// ── Sources panel ─────────────────────────────────────────────────────────

// Dwell observer — telemetría de atención sobre source rows.
const _DWELL_MIN_MS = 1500;
const _DWELL_MAX_MS = 5 * 60 * 1000;
const _dwellStart = new WeakMap();
const _dwellReported = new WeakSet();

const _dwellObserver = (typeof IntersectionObserver !== "undefined")
  ? new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const row = entry.target;
        if (entry.isIntersecting) {
          if (!_dwellStart.has(row)) _dwellStart.set(row, Date.now());
        } else {
          const start = _dwellStart.get(row);
          _dwellStart.delete(row);
          if (start == null || _dwellReported.has(row)) continue;
          const elapsed = Date.now() - start;
          if (elapsed < _DWELL_MIN_MS || elapsed > _DWELL_MAX_MS) continue;
          _dwellReported.add(row);
          _emitDwell(row, elapsed);
        }
      }
    }, { threshold: [0, 0.5] })
  : null;

function _emitDwell(row, dwellMs) {
  const payload = {
    source: "web", event: "open",
    path: row.dataset.path || null,
    rank: row.dataset.rank ? Number(row.dataset.rank) : null,
    query: row.dataset.q || null,
    dwell_ms: Math.floor(dwellMs),
    session: row.dataset.session || null,
  };
  if (!payload.path) return;
  try {
    fetch("/api/behavior", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {});
  } catch {}
}

function _observeDwell(rows) {
  if (!_dwellObserver) return;
  for (const r of rows) _dwellObserver.observe(r);
}

// Flush al backgroundear la tab.
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "hidden") return;
    const rows = document.querySelectorAll(".source-row[data-path]");
    for (const row of rows) {
      if (_dwellReported.has(row)) continue;
      const start = _dwellStart.get(row);
      if (start == null) continue;
      const elapsed = Date.now() - start;
      if (elapsed < _DWELL_MIN_MS || elapsed > _DWELL_MAX_MS) continue;
      _dwellReported.add(row);
      _emitDwell(row, elapsed);
    }
  });
}

/**
 * Renderiza el panel de fuentes (╌ fuentes + confidence badge).
 * @param {HTMLElement} parent
 * @param {Array} items — lista de source objects
 * @param {number|null} confidence
 */
export function appendSources(parent, items, confidence) {
  const wrap = el("div", "sources");
  const head = el("div", "sources-rule");
  head.textContent = "╌ fuentes ";
  if (Number.isFinite(confidence)) head.appendChild(confidenceBadge(confidence));
  wrap.appendChild(head);
  const seen = new Set();
  const rows = [];
  const parentTurn = parent.closest ? parent.closest(".turn") : null;
  let rank = 0;
  for (const s of items) {
    const sourceKind = s.source_kind || s.sourceKind || "";
    const dedupeKey = s.file || `${sourceKind}:${s.note || rank}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    rank += 1;
    const row = el("div", "source-row");
    const filled = ((s.bar || "").match(/■/g) || []).length;
    const tone = filled >= 3 ? "good" : filled >= 1 ? "mid" : "low";
    const bar = el("span", `bar bar-${tone}`);
    bar.textContent = s.bar || "■■■■■";
    row.appendChild(bar);

    const isExternal = typeof s.file === "string" && /^https?:\/\//i.test(s.file);
    const isModel = sourceKind === "model" || (typeof s.file === "string" && s.file.indexOf("model://") === 0);
    const isWA = typeof s.file === "string" && s.file.indexOf("whatsapp://") === 0;
    const waUrl = isWA ? waHref(s.file) : "";
    const wantsBlank = isExternal || (isWA && waUrl);
    const linkable = !isModel && (isExternal || waUrl || !isWA);

    let noteEl;
    if (linkable) {
      noteEl = el("a", "note", s.note || s.file);
      noteEl.href = isExternal ? s.file : (waUrl || obsidianUrl(s.file));
      if (wantsBlank) { noteEl.target = "_blank"; noteEl.rel = "noopener noreferrer"; }
    } else {
      noteEl = el("span", "note", s.note || s.file);
    }
    noteEl.title = s.file;
    row.appendChild(noteEl);

    let pathLabel;
    if (isModel) pathLabel = s.folder || "conocimiento general";
    else if (isExternal) pathLabel = s.folder || s.domain || "internet";
    else if (isWA) pathLabel = s.folder || "WhatsApp";
    else pathLabel = s.file;

    let pathEl;
    if (linkable) {
      pathEl = el("a", "path", pathLabel);
      pathEl.href = isExternal ? s.file : (waUrl || obsidianUrl(s.file));
      if (wantsBlank) { pathEl.target = "_blank"; pathEl.rel = "noopener noreferrer"; }
    } else {
      pathEl = el("span", "path", pathLabel);
    }
    pathEl.title = s.file;
    row.appendChild(pathEl);

    if (!isModel && s.file && s.file.indexOf("://") === -1) {
      row.dataset.path = s.file;
      row.dataset.rank = String(rank);
      if (parentTurn && parentTurn.dataset.q) row.dataset.q = parentTurn.dataset.q;
      if (parentTurn && parentTurn.dataset.session) row.dataset.session = parentTurn.dataset.session;
    }

    if (!isModel && s.file && s.file.indexOf("://") === -1 && parentTurn?.dataset?.turnId) {
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = "source-thumb";
      thumb.setAttribute("aria-label", "marcar como fuente correcta");
      thumb.title = "marcar como fuente correcta — entrena el ranker";
      thumb.textContent = "👍";
      const _q = parentTurn.dataset.q || "";
      const _turn_id = parentTurn.dataset.turnId;
      const _session_id = parentTurn.dataset.session || null;
      const _path = s.file;
      thumb.addEventListener("click", async (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (thumb.disabled) return;
        thumb.disabled = true;
        thumb.classList.add("picked");
        try {
          const res = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              turn_id: _turn_id, rating: 1, q: _q,
              paths: items.map((x) => x && x.file).filter(Boolean),
              corrective_path: _path, session_id: _session_id,
            }),
          });
          if (!res.ok) throw new Error("HTTP " + res.status);
          thumb.textContent = "✓";
          thumb.title = "anotado — el ranker aprende";
        } catch (_) {
          thumb.disabled = false;
          thumb.classList.remove("picked");
          thumb.title = "error — reintentá";
        }
      });
      row.appendChild(thumb);
    }

    wrap.appendChild(row);
    rows.push(row);
  }
  parent.appendChild(wrap);
  const trackable = rows.filter((r) => r.dataset && r.dataset.path);
  if (trackable.length) _observeDwell(trackable);
}

// ── Enrich, Grounding, Related, WebSearch, FallbackCluster ───────────────

/**
 * Panel de contexto enriquecido (WhatsApp/Calendar/Reminders signals).
 * @param {HTMLElement} parent
 * @param {Array} lines
 */
export function appendEnrich(parent, lines) {
  const wrap = el("div", "enrich-block");
  wrap.appendChild(el("div", "enrich-head", "📎 contexto relacionado"));
  for (const ln of lines) {
    const row = el("div", "enrich-line");
    row.appendChild(el("span", "enrich-icon", ln.icon || "·"));
    const textCol = el("span", "enrich-text-col");
    textCol.appendChild(el("span", "enrich-text", ln.text || ""));
    if (ln.snippet) textCol.appendChild(el("span", "enrich-snippet", ` — ${ln.snippet}`));
    row.appendChild(textCol);
    if (ln.relative) row.appendChild(el("span", "enrich-time", ln.relative));
    wrap.appendChild(row);
  }
  parent.appendChild(wrap);
}

/**
 * Panel de grounding NLI (entails / neutral / contradicts claims).
 * @param {object} data
 * @param {HTMLElement} container
 */
export function renderGrounding(data, container) {
  if (!data || data.total === 0) return;
  const details = el("details", "grounding-panel");
  const parts = [];
  if (data.supported)    parts.push(`✓ ${data.supported}`);
  if (data.contradicted) parts.push(`✗ ${data.contradicted}`);
  if (data.neutral)      parts.push(`· ${data.neutral}`);
  const summary = document.createElement("summary");
  summary.className = "grounding-summary";
  summary.textContent = `${parts.join(" / ")} claims`;
  details.appendChild(summary);
  const ul = document.createElement("ul");
  ul.className = "grounding-list";
  for (const claim of (data.claims || [])) {
    const li = document.createElement("li");
    li.className = `grounding-claim grounding-claim-${claim.verdict}`;
    const icon = claim.verdict === "entails" ? "✓" : claim.verdict === "contradicts" ? "✗" : "·";
    li.appendChild(document.createTextNode(`${icon} ${claim.text}`));
    if (claim.evidence_note) li.appendChild(el("small", "grounding-note", ` (${claim.evidence_note})`));
    ul.appendChild(li);
  }
  details.appendChild(ul);
  container.appendChild(details);
}

/**
 * Fetch + render del bloque "contexto relacionado" (Spotify/YouTube).
 * Usa state.inflightSideFetches para cancelar si el user navega.
 * @param {HTMLElement} parent
 * @param {string} query
 */
export async function appendRelated(parent, query) {
  if (!query) return;
  let items = [];
  const ac = new AbortController();
  state.inflightSideFetches.add(ac);
  try {
    const res = await fetch("/api/related", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: ac.signal,
    });
    if (!res.ok) return;
    const data = await res.json();
    items = Array.isArray(data.items) ? data.items : [];
  } catch { return; }
  finally { state.inflightSideFetches.delete(ac); }
  if (!items.length) return;
  const wrap = el("div", "related");
  wrap.appendChild(el("div", "related-head", "📎 contexto relacionado"));
  for (const it of items) {
    const row = document.createElement("a");
    row.className = `related-item related-${it.source}`;
    row.href = it.url;
    row.target = "_blank";
    row.rel = "noopener noreferrer";
    const badge = el("span", "related-badge", it.source);
    const title = el("span", "related-title", it.title);
    row.appendChild(badge);
    row.appendChild(title);
    if (it.subtitle) row.appendChild(el("span", "related-sub", it.subtitle));
    wrap.appendChild(row);
  }
  parent.appendChild(wrap);
}

/**
 * Link inline "↗ buscar en internet" (DuckDuckGo).
 * @param {HTMLElement} parent
 * @param {string} query
 * @param {boolean} inline
 */
export function appendWebSearch(parent, query, inline = false) {
  const link = document.createElement("a");
  link.className = "web-search-link" + (inline ? " inline" : "");
  link.href = `https://duckduckgo.com/?q=${encodeURIComponent(query)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "↗ buscar en internet";
  link.title = `DuckDuckGo: ${query}`;
  if (inline) {
    parent.appendChild(document.createTextNode(" "));
    parent.appendChild(link);
  } else {
    const wrap = el("div", "web-search");
    wrap.appendChild(link);
    parent.appendChild(wrap);
  }
}

/**
 * Cluster prominente de fallback (DuckDuckGo/YouTube/Wikipedia).
 * @param {HTMLElement} parent
 * @param {string} query
 */
export function appendFallbackCluster(parent, query) {
  if (!query) return;
  const wrap = el("div", "fallback-cluster");
  wrap.appendChild(el("div", "fallback-head", "no encontré eso en tus notas — ¿querés que busque en...?"));
  const buttons = el("div", "fallback-buttons");
  const q = encodeURIComponent(query);
  const specs = [
    { cls: "fallback-duckduckgo", label: "🔍 DuckDuckGo", url: `https://duckduckgo.com/?q=${q}` },
    { cls: "fallback-youtube", label: "▶ YouTube",    url: `https://www.youtube.com/results?search_query=${q}` },
    { cls: "fallback-wiki",    label: "📖 Wikipedia", url: `https://es.wikipedia.org/wiki/Special:Search?search=${q}` },
  ];
  for (const s of specs) {
    const a = document.createElement("a");
    a.className = `fallback-btn ${s.cls}`;
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = s.label;
    a.title = `${s.label.replace(/^[^\s]+\s*/, "")}: ${query}`;
    buttons.appendChild(a);
  }
  wrap.appendChild(buttons);
  parent.appendChild(wrap);
}

/**
 * Chips de followup ("seguir con ›") generados post-done.
 * Usa state.inflightSideFetches + state.pending para abortar si el user navega.
 * @param {HTMLElement} parent
 * @param {string} sid — session_id
 */
export async function appendFollowups(parent, sid) {
  try {
    if (parent && parent.querySelector && parent.querySelector(".proposal")) return;
  } catch (_) {}
  const ac = new AbortController();
  state.inflightSideFetches.add(ac);
  let arr = [];
  try {
    const res = await fetch("/api/followups", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
      signal: ac.signal,
    });
    if (!res.ok) return;
    const data = await res.json();
    arr = (data.followups || []).filter((x) => typeof x === "string");
  } catch (_) { return; }
  finally { state.inflightSideFetches.delete(ac); }
  try {
    if (!arr.length) return;
    const inputEl = document.getElementById("input");
    const formEl = document.getElementById("composer");
    const wrap = el("div", "followups");
    wrap.appendChild(el("span", "followups-label", "seguir con ›"));
    for (const q of arr) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "follow-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => {
        if (state.pending) return;
        if (inputEl) inputEl.value = q;
        // autoGrow() está en app.js — llamar via window por ahora
        if (typeof window.autoGrow === "function") window.autoGrow();
        if (formEl) formEl.requestSubmit();
      });
      wrap.appendChild(chip);
    }
    parent.appendChild(wrap);
  } catch {}
}

// ── Proposal: router y delegación ────────────────────────────────────────

/**
 * Router de proposal cards. Despacha al renderer correcto según payload.kind.
 * Para los tipos complejos (whatsapp, mail, calendar, reminders) delega a
 * la implementación de app.js via window mientras se completa la migración.
 * @param {HTMLElement} parent
 * @param {object} payload
 * @returns {HTMLElement}
 */
export function appendProposal(parent, payload) {
  // Delegar a la implementación completa de app.js durante la transición.
  // Las proposals son el bloque más complejo (~1500 LOC) y requieren su propia
  // fase de extracción. window.appendProposal apunta a la versión de app.js.
  if (typeof window._appAppendProposal === "function") {
    return window._appAppendProposal(parent, payload);
  }
  // Fallback mínimo para no crashear.
  const div = el("div", "proposal proposal-unknown");
  div.textContent = `[propuesta: ${payload.kind || "?"}]`;
  parent.appendChild(div);
  return div;
}

/**
 * Chip inline de confirmación.
 * @param {HTMLElement} parent
 * @param {object} payload
 * @returns {HTMLElement}
 */
export function appendCreatedChip(parent, payload) {
  if (typeof window._appAppendCreatedChip === "function") {
    return window._appAppendCreatedChip(parent, payload);
  }
  const div = el("div", "created-chip");
  div.textContent = `✓ ${payload.kind || "creado"}`;
  parent.appendChild(div);
  return div;
}

/**
 * Hidrata el DOM de #messages con los turns de una sesión histórica.
 * @param {object} data — { id, turns: [{q, a, paths}] }
 */
export function hydrateTurns(data) {
  // Delegar a la implementación de app.js durante la transición — usa renderMarkdown.
  if (typeof window._appHydrateTurns === "function") {
    return window._appHydrateTurns(data);
  }
  // Fallback mínimo.
  const messagesEl = document.getElementById("messages");
  if (!messagesEl) return;
  messagesEl.innerHTML = "";
  const turns = Array.isArray(data && data.turns) ? data.turns : [];
  if (!turns.length) {
    const turn = appendTurn();
    appendMeta(turn, ["sesión vacía"]);
    return;
  }
  for (const t of turns) {
    const turn = appendTurn();
    if (t.q) appendLine(turn, "user", t.q);
    if (t.a) appendLine(turn, "rag", t.a);
  }
  scrollBottom();
}
