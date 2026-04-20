const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const vaultPicker = document.getElementById("vault-picker");
const modelPicker = document.getElementById("model-picker");
const ttsToggle = document.getElementById("tts-toggle");
const helpBtn = document.getElementById("help-btn");
const helpModal = document.getElementById("help-modal");
const stopBtn = document.getElementById("stop-btn");

const SESSION_KEY = "obsidian-rag:session";
const VAULT_KEY = "obsidian-rag:vault";
const TTS_KEY = "obsidian-rag:tts";
// Session id lives in sessionStorage (per-tab), not localStorage. The DOM
// doesn't restore prior turns on reload, but the server-side history was
// being silently rehydrated — so a fresh-looking /chat tab inherited stale
// FinOps/etc. context and biased every answer. Per-tab keeps multi-turn
// across reloads while a new tab gets a clean slate.
let sessionId = sessionStorage.getItem(SESSION_KEY) || null;
// Legacy: prior versions stored session_id in localStorage, which leaked
// stale history into fresh tabs. Drop any leftover entry on first load.
try { localStorage.removeItem(SESSION_KEY); } catch {}
let vaultScope = localStorage.getItem(VAULT_KEY) || "";
let ttsEnabled = localStorage.getItem(TTS_KEY) === "1";
let pending = false;
let currentController = null;      // AbortController for in-flight /api/chat
let currentAudio = null;           // In-flight <audio> playback
let lastUserQuestion = "";         // For ⌘↑ edit-last

// Terminal-style query history. Persisted across sessions via localStorage.
const HISTORY_KEY = "obsidian-rag:history";
const HISTORY_CAP = 100;
function loadHistory() {
  try {
    const arr = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(arr) ? arr.filter((s) => typeof s === "string") : [];
  } catch { return []; }
}
let history = loadHistory();
let historyIdx = history.length;   // length = "no entry restored, draft is current"
let historyDraft = "";             // in-progress text saved when entering history mode
function pushHistory(q) {
  q = q.trim();
  if (!q) return;
  if (history[history.length - 1] !== q) {
    history.push(q);
    if (history.length > HISTORY_CAP) history.splice(0, history.length - HISTORY_CAP);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
  }
  historyIdx = history.length;
  historyDraft = "";
}
// History popover — mirrors the slash popover UX so up-arrow opens a
// browsable list of past queries instead of swapping the input value
// silently. Newest entry is row 1 (terminal-style).
const historyPopover = document.getElementById("history-popover");
let historyPopoverIdx = 0;
let historyPopoverItems = [];
function renderHistoryPopover() {
  historyPopover.innerHTML = "";
  if (!historyPopoverItems.length) {
    historyPopover.appendChild(el("div", "slash-popover-empty", "sin historial todavía"));
    return;
  }
  historyPopoverItems.forEach((q, i) => {
    const row = el("div", "history-item" + (i === historyPopoverIdx ? " active" : ""));
    row.setAttribute("role", "option");
    row.appendChild(el("span", "history-idx", String(i + 1)));
    row.appendChild(el("span", "history-q", q));
    row.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      pickHistoryEntry(q);
    });
    row.addEventListener("mouseenter", () => {
      historyPopoverIdx = i;
      [...historyPopover.children].forEach((c, j) => c.classList.toggle("active", j === historyPopoverIdx));
    });
    historyPopover.appendChild(row);
  });
  const active = historyPopover.children[historyPopoverIdx];
  if (active) active.scrollIntoView({ block: "nearest" });
}
function openHistoryPopover() {
  if (!history.length) return false;
  if (historyIdx === history.length) historyDraft = input.value;
  // Cap visible entries — popover stays compact, scroll handles the rest.
  // Newest first matches terminal arrow-up convention.
  historyPopoverItems = history.slice(-30).reverse();
  historyPopoverIdx = 0;
  historyPopover.hidden = false;
  renderHistoryPopover();
  return true;
}
function hideHistoryPopover() {
  historyPopover.hidden = true;
  historyPopoverItems = [];
}
function pickHistoryEntry(q) {
  input.value = q;
  autoGrow();
  input.setSelectionRange(input.value.length, input.value.length);
  hideHistoryPopover();
  input.focus();
}

// Hydrate from server-side queries.jsonl so up-arrow surfaces real past
// queries even on a fresh tab/browser. localStorage cache makes the first
// arrow press instant; the fetch refines it with the authoritative log.
(async function loadServerHistory() {
  try {
    const res = await fetch("/api/history?limit=200");
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data.history) || !data.history.length) return;
    history = data.history;
    historyIdx = history.length;
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
  } catch {}
})();

// Vault picker ----------------------------------------------------
async function loadVaults() {
  try {
    const res = await fetch("/api/vaults");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    vaultPicker.innerHTML = "";

    const activeOpt = document.createElement("option");
    activeOpt.value = "";
    activeOpt.textContent = data.active ? `${data.active} (activo)` : "activo";
    vaultPicker.appendChild(activeOpt);

    const others = (data.registered || []).filter((n) => n !== data.active);
    for (const name of others) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      vaultPicker.appendChild(opt);
    }

    if (data.registered && data.registered.length > 1) {
      const allOpt = document.createElement("option");
      allOpt.value = "all";
      allOpt.textContent = "todos";
      vaultPicker.appendChild(allOpt);
    }

    const options = Array.from(vaultPicker.options).map((o) => o.value);
    if (vaultScope && options.includes(vaultScope)) {
      vaultPicker.value = vaultScope;
    } else {
      vaultPicker.value = "";
      vaultScope = "";
      localStorage.removeItem(VAULT_KEY);
    }
  } catch (err) {
    vaultPicker.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "activo";
    vaultPicker.appendChild(opt);
  }
}

vaultPicker.addEventListener("change", () => {
  vaultScope = vaultPicker.value;
  if (vaultScope) localStorage.setItem(VAULT_KEY, vaultScope);
  else localStorage.removeItem(VAULT_KEY);
});

loadVaults();

// Chat model picker ------------------------------------------------
// Runtime switch between installed chat models. The backend persists
// the choice in ~/.local/share/obsidian-rag/chat-model.json so the
// selection survives server restarts. No reload needed — the next
// /api/chat call picks up the override automatically.
async function loadChatModels() {
  if (!modelPicker) return;
  try {
    const res = await fetch("/api/chat/model");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    modelPicker.innerHTML = "";
    // Sort so the current one lands first; that's what's active right now.
    const available = data.available || [];
    const current = data.current;
    const rest = available.filter((m) => m !== current);
    const ordered = current ? [current, ...rest] : rest;
    for (const name of ordered) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name === current ? `${name} (activo)` : name;
      modelPicker.appendChild(opt);
    }
    // If the current model isn't in `available` (env override pointing to
    // an external registry), still show it as selected.
    if (current && !ordered.includes(current)) {
      const opt = document.createElement("option");
      opt.value = current;
      opt.textContent = `${current} (activo)`;
      modelPicker.insertBefore(opt, modelPicker.firstChild);
    }
    modelPicker.value = current || "";
    modelPicker.dataset.current = current || "";
  } catch (err) {
    modelPicker.innerHTML = '<option value="">n/a</option>';
  }
}

modelPicker?.addEventListener("change", async () => {
  const selected = modelPicker.value;
  const previous = modelPicker.dataset.current || "";
  if (!selected || selected === previous) return;
  // Optimistic UI: mark busy; revert on error.
  modelPicker.disabled = true;
  try {
    const res = await fetch("/api/chat/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selected }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    modelPicker.dataset.current = data.current;
    // Re-render the "(activo)" marker on the new option.
    await loadChatModels();
  } catch (err) {
    console.error("[chat-model] switch failed:", err);
    modelPicker.value = previous;
    // Non-blocking toast via the textarea placeholder would be ideal,
    // but alert() is the simplest way to surface a validation error
    // (e.g. model not installed) without more UI plumbing.
    alert(`No se pudo cambiar el modelo: ${err.message}`);
  } finally {
    modelPicker.disabled = false;
  }
});

loadChatModels();

// TTS toggle ------------------------------------------------------
// SVGs defined up-front so renderTtsToggle() can use them on initial
// render (called below before the rest of the icons block runs). Style
// matches the thumbs/copy icons: lucide-style 1.6px stroke, currentColor.
const SPEAKER_ON_SVG = `
  <svg class="tts-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M11 5 6 9H3v6h3l5 4z"/>
    <path d="M15.5 8.5a5 5 0 0 1 0 7"/>
    <path d="M18.5 5.5a9 9 0 0 1 0 13"/>
  </svg>`;
const SPEAKER_OFF_SVG = `
  <svg class="tts-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M11 5 6 9H3v6h3l5 4z"/>
    <line x1="22" y1="9" x2="16" y2="15"/>
    <line x1="16" y1="9" x2="22" y2="15"/>
  </svg>`;

function renderTtsToggle() {
  ttsToggle.setAttribute("aria-pressed", ttsEnabled ? "true" : "false");
  ttsToggle.querySelector(".tts-icon").innerHTML = ttsEnabled ? SPEAKER_ON_SVG : SPEAKER_OFF_SVG;
  ttsToggle.classList.toggle("active", ttsEnabled);
}
ttsToggle.addEventListener("click", () => {
  ttsEnabled = !ttsEnabled;
  localStorage.setItem(TTS_KEY, ttsEnabled ? "1" : "0");
  renderTtsToggle();
  if (!ttsEnabled && currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
});
renderTtsToggle();

// Help modal ------------------------------------------------------
function openHelp() { helpModal.hidden = false; }
function closeHelp() { helpModal.hidden = true; }
helpBtn.addEventListener("click", openHelp);
helpModal.addEventListener("click", (e) => {
  if (e.target instanceof Element && e.target.dataset.close !== undefined) closeHelp();
});

// Slash command registry — single source of truth shared between the
// popover (autocomplete) and the handler (execution). `arg` surfaces as
// muted text after the cmd name in the list so the user sees e.g.
// `/save [título]`. Handler lookup uses `cmd` as the literal prefix.
const SLASH_COMMANDS = [
  { cmd: "/help",    desc: "atajos y comandos" },
  { cmd: "/cls",     desc: "limpiar vista (sesión intacta)" },
  { cmd: "/new",     desc: "nueva sesión (olvida historial)" },
  { cmd: "/session", desc: "info de la sesión actual" },
  { cmd: "/model",   desc: "modelo de chat en uso" },
  { cmd: "/save",    desc: "guardar conversación en 00-Inbox", arg: "[título]" },
  { cmd: "/reindex", desc: "reindex incremental en background" },
  { cmd: "/tts",     desc: "alternar voz (Mónica)" },
];

const slashPopover = document.getElementById("slash-popover");
let slashIndex = 0;
let slashVisibleItems = [];

function slashMatchingCommands(value) {
  if (!value.startsWith("/")) return [];
  const spaceIdx = value.indexOf(" ");
  // Once the user types a space after a complete command, stop showing
  // the popover — they're in arg-typing mode (e.g. `/save my title`).
  if (spaceIdx >= 0) return [];
  const prefix = value.toLowerCase();
  return SLASH_COMMANDS.filter((c) => c.cmd.startsWith(prefix));
}

function renderSlashPopover(items) {
  slashPopover.innerHTML = "";
  if (!items.length) {
    const empty = el("div", "slash-popover-empty", "sin comandos — Enter envía al vault");
    slashPopover.appendChild(empty);
    return;
  }
  items.forEach((c, i) => {
    const row = el("div", "slash-item" + (i === slashIndex ? " active" : ""));
    row.setAttribute("role", "option");
    row.dataset.cmd = c.cmd;
    const name = el("span", "slash-cmd", c.cmd);
    if (c.arg) {
      const arg = el("span", "slash-arg", c.arg);
      name.appendChild(arg);
    }
    row.appendChild(name);
    row.appendChild(el("span", "slash-desc", c.desc));
    row.addEventListener("mousedown", (ev) => {
      // mousedown, not click — click fires after blur, which hides the
      // popover and loses the selection.
      ev.preventDefault();
      completeSlash(c);
    });
    row.addEventListener("mouseenter", () => {
      slashIndex = i;
      renderSlashPopover(slashVisibleItems);
    });
    slashPopover.appendChild(row);
  });
}

function updateSlashPopover() {
  const items = slashMatchingCommands(input.value);
  slashVisibleItems = items;
  if (!input.value.startsWith("/")) {
    slashPopover.hidden = true;
    return;
  }
  if (slashIndex >= items.length) slashIndex = 0;
  renderSlashPopover(items);
  slashPopover.hidden = false;
}

function hideSlashPopover() {
  slashPopover.hidden = true;
  slashVisibleItems = [];
  slashIndex = 0;
}

function completeSlash(c) {
  // `/save` takes an arg, so leave a trailing space and keep the popover
  // closed — the user is about to type. Arg-less commands get completed
  // fully so a second Enter submits directly.
  input.value = c.arg ? `${c.cmd} ` : c.cmd;
  autoGrow();
  input.focus();
  hideSlashPopover();
}

// Input autogrow + enter-to-send --------------------------------
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}
input.addEventListener("input", () => {
  autoGrow();
  updateSlashPopover();
  if (!historyPopover.hidden) hideHistoryPopover();
});

input.addEventListener("keydown", (e) => {
  // Slash popover navigation takes priority over the normal Enter/Esc
  // handling so ↑/↓ + Tab/Enter behave like in Claude Code.
  if (!slashPopover.hidden && slashVisibleItems.length) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      slashIndex = (slashIndex + 1) % slashVisibleItems.length;
      renderSlashPopover(slashVisibleItems);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      slashIndex = (slashIndex - 1 + slashVisibleItems.length) % slashVisibleItems.length;
      renderSlashPopover(slashVisibleItems);
      return;
    }
    if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
      e.preventDefault();
      const pick = slashVisibleItems[slashIndex];
      if (pick) completeSlash(pick);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideSlashPopover();
      return;
    }
  }
  if (!historyPopover.hidden) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      historyPopoverIdx = (historyPopoverIdx + 1) % historyPopoverItems.length;
      renderHistoryPopover();
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      historyPopoverIdx = (historyPopoverIdx - 1 + historyPopoverItems.length) % historyPopoverItems.length;
      renderHistoryPopover();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const pick = historyPopoverItems[historyPopoverIdx];
      if (pick) pickHistoryEntry(pick);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideHistoryPopover();
      input.value = historyDraft;
      autoGrow();
      return;
    }
  }
  if ((e.key === "ArrowUp" || e.key === "ArrowDown") &&
      !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey &&
      history.length > 0) {
    const v = input.value;
    const caret = input.selectionStart;
    const edgeOk = e.key === "ArrowUp"
      ? !v.slice(0, caret).includes("\n")
      : !v.slice(caret).includes("\n");
    if (edgeOk) {
      e.preventDefault();
      openHistoryPopover();
      return;
    }
  }
  if (e.key === "w" && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
    e.preventDefault();
    const v = input.value;
    const end = input.selectionEnd;
    let start = input.selectionStart;
    if (start === end) {
      while (start > 0 && /\s/.test(v[start - 1])) start--;
      while (start > 0 && !/\s/.test(v[start - 1])) start--;
    }
    if (start === end) return;
    input.value = v.slice(0, start) + v.slice(end);
    input.setSelectionRange(start, start);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    return;
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("blur", () => {
  // Small delay so a mousedown on a slash-item still registers before the
  // popover disappears.
  setTimeout(() => { hideSlashPopover(); hideHistoryPopover(); }, 120);
});
input.addEventListener("focus", updateSlashPopover);

// Global keyboard shortcuts -------------------------------------
document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (e.key === "Escape") {
    if (!helpModal.hidden) { closeHelp(); return; }
    if (pending && currentController) { currentController.abort(); return; }
    if (currentAudio) { currentAudio.pause(); currentAudio = null; return; }
  }
  if (mod && e.key === "/") {
    e.preventDefault();
    if (helpModal.hidden) openHelp(); else closeHelp();
    return;
  }
  if (mod && e.key.toLowerCase() === "k") {
    e.preventDefault();
    input.focus();
    input.select();
    return;
  }
  if (mod && e.key === "ArrowUp") {
    if (document.activeElement !== input) return;
    if (!lastUserQuestion) return;
    if (input.value.trim()) return;
    e.preventDefault();
    input.value = lastUserQuestion;
    autoGrow();
    input.setSelectionRange(input.value.length, input.value.length);
  }
});

stopBtn.addEventListener("click", () => {
  if (currentController) currentController.abort();
});

// Tool-call chip labels — backend emits raw tool names (search_vault,
// read_note, …); UI shows a Spanish-friendly short form. Unknown names
// fall back to the raw identifier so new tools still render.
const TOOL_LABELS = {
  search_vault: "búsqueda vault",
  read_note: "leyendo nota",
  reminders_due: "reminders",
  gmail_recent: "gmail",
  finance_summary: "finanzas",
  calendar_ahead: "calendario",
  weather: "clima",
};
function toolLabel(name) {
  if (!name) return "herramienta";
  return TOOL_LABELS[name] || name;
}
function formatMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 10000) return `${Math.round(n)}ms`;
  return `${(n / 1000).toFixed(1)}s`;
}

// Rendering helpers --------------------------------------------
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendTurn() {
  const turn = el("div", "turn");
  messagesEl.appendChild(turn);
  return turn;
}

function appendLine(parent, role, text) {
  const line = el("div", "line");
  line.appendChild(el("span", `prompt ${role}`, role === "user" ? "tu ›" : "rag ›"));
  const t = el("span", `text ${role}`);
  t.textContent = text || "";
  line.appendChild(t);
  parent.appendChild(line);
  return t;
}

function appendMeta(parent, bits) {
  const m = el("div", "meta", "  " + bits.join(" · "));
  parent.appendChild(m);
}

// Confidence badge — mirrors server's _confidence_badge (score ≥3 verde,
// ≥0 amarillo, <0 rojo). Renderizado como pill en el header de fuentes.
function confidenceBadge(score) {
  const s = Number.isFinite(score) ? score : 0;
  let level = "mid";
  let label = "media";
  if (s >= 3.0) { level = "high"; label = "alta"; }
  else if (s < 0) { level = "low"; label = "baja"; }
  const span = el("span", `conf-pill conf-${level}`);
  span.title = `score top rerank: ${s.toFixed(2)}`;
  span.textContent = `confianza ${label} · ${s.toFixed(1)}`;
  return span;
}

// Outlined thumbs icons, aesthetic-matched to the terminal UI.
const THUMB_UP_SVG = `
  <svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M7 10v11"/>
    <path d="M7 10c2.2-2.5 3.4-4.6 3.6-6.4.1-.9.8-1.6 1.7-1.6.9 0 1.7.8 1.7 1.7 0 1.3-.3 2.5-.9 3.6-.2.3 0 .7.4.7h4.4a2 2 0 0 1 2 2.3l-1.2 7.6a2 2 0 0 1-2 1.7H7"/>
    <path d="M3 10h4v11H3z"/>
  </svg>`;
const THUMB_DOWN_SVG = `
  <svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M17 14V3"/>
    <path d="M17 14c-2.2 2.5-3.4 4.6-3.6 6.4-.1.9-.8 1.6-1.7 1.6-.9 0-1.7-.8-1.7-1.7 0-1.3.3-2.5.9-3.6.2-.3 0-.7-.4-.7H6.1a2 2 0 0 1-2-2.3L5.3 6.1a2 2 0 0 1 2-1.7H17"/>
    <path d="M21 14h-4V3h4z"/>
  </svg>`;
const COPY_SVG = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="13" height="13" rx="2"/>
    <path d="M5 15V5a2 2 0 0 1 2-2h10"/>
  </svg>`;
function appendFeedback(parent, ctx) {
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

  async function submit(rating, reason) {
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
          rating: rating,
          q: ctx.q,
          paths: ctx.paths,
          session_id: ctx.session_id,
          reason: reason || null,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      status.textContent = rating > 0 ? "  gracias — anotado" : "  anotado — seguimos afinando";
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

  function openReasonInput() {
    if (wrap.dataset.reasonOpen || wrap.dataset.sent) return;
    wrap.dataset.reasonOpen = "1";
    down.classList.add("picked");
    up.classList.add("dimmed");

    const row = el("div", "feedback-reason");
    const field = document.createElement("input");
    field.type = "text";
    field.className = "fb-reason-input";
    field.placeholder = "¿qué faltó? (opcional) — ej: falta la nota X, muy genérico";
    field.maxLength = 200;
    field.setAttribute("aria-label", "motivo (opcional)");

    const send = document.createElement("button");
    send.type = "button";
    send.className = "fb-text-btn";
    send.textContent = "enviar";

    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "fb-text-btn fb-text-muted";
    skip.textContent = "omitir";

    async function commit() {
      const reason = field.value.trim();
      row.remove();
      await submit(-1, reason);
    }

    field.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); commit(); }
      else if (ev.key === "Escape") { row.remove(); delete wrap.dataset.reasonOpen;
        down.classList.remove("picked"); up.classList.remove("dimmed"); }
    });
    send.addEventListener("click", commit);
    skip.addEventListener("click", commit);

    row.appendChild(field);
    row.appendChild(send);
    row.appendChild(skip);
    wrap.appendChild(row);
    field.focus();
  }

  up.addEventListener("click", () => submit(1));
  down.addEventListener("click", openReasonInput);

  wrap.appendChild(prompt);
  wrap.appendChild(up);
  wrap.appendChild(down);
  wrap.appendChild(status);
  parent.appendChild(wrap);
  return wrap;
}

// Web-search escape hatch — surfaces when the vault has weak/no answer
// (sin sources, o confianza baja). One click → Google búsqueda en pestaña
// nueva. El usuario decide si vale la pena salir del vault.
function appendWebSearch(parent, query) {
  const wrap = el("div", "web-search");
  const link = document.createElement("a");
  link.className = "web-search-link";
  link.href = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "↗  buscar en internet";
  link.title = `Google: ${query}`;
  wrap.appendChild(link);
  parent.appendChild(wrap);
}

// ── Proposal cards ─────────────────────────────────────────────────────────
//
// The server fires an SSE `proposal` event whenever the LLM invokes
// `propose_reminder` or `propose_calendar_event`. The payload shape is
// identical to what those tools return — {kind, proposal_id, fields,
// needs_clarification?}. We render an inline card with editable-looking
// display of the parsed fields and a ✓/✗ pair. Clicking ✓ POSTs to the
// corresponding create endpoint; ✗ silently dismisses. Nothing lands in
// Calendar/Reminders until the user confirms.

// Toast utility — prominent system-level confirmation when reminders/
// events are created. Not just the inline chip next to the button: the
// chip is easy to miss if the chat has scrolled. Toast lives top-right,
// auto-dismisses after 4s, stacks if multiple fire.
function showToast(message, kind = "ok") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = el("div", "toast-container");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const toast = el("div", `toast toast-${kind}`, message);
  container.appendChild(toast);
  // Animate in on next frame so the browser picks up the transition.
  requestAnimationFrame(() => toast.classList.add("toast-visible"));
  setTimeout(() => {
    toast.classList.remove("toast-visible");
    toast.addEventListener(
      "transitionend",
      () => toast.remove(),
      { once: true },
    );
    // Safety net in case transitionend never fires (e.g. `prefers-reduced-motion`).
    setTimeout(() => toast.remove(), 500);
  }, 4000);
}

function formatIsoDatetime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    // es-AR short: "jue, 23/04 16:00"
    return d.toLocaleString("es-AR", {
      weekday: "short", day: "2-digit", month: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function appendProposal(parent, payload) {
  const kind = payload.kind;                    // "reminder" | "event"
  const fields = payload.fields || {};
  const needsClarif = payload.needs_clarification === true;

  const card = el("div", `proposal proposal-${kind}`);

  const head = el("div", "proposal-head");
  head.appendChild(el("span", "proposal-icon", kind === "event" ? "📅" : "✓"));
  head.appendChild(el(
    "span", "proposal-kind",
    kind === "event" ? "Nuevo evento" : "Nuevo recordatorio",
  ));
  card.appendChild(head);

  const title = el("div", "proposal-title", fields.title || "(sin título)");
  card.appendChild(title);

  const meta = el("div", "proposal-meta");
  const addMeta = (label, value) => {
    if (value === null || value === undefined || value === "") return;
    const row = el("div", "proposal-meta-row");
    row.appendChild(el("span", "proposal-meta-label", label));
    row.appendChild(el("span", "proposal-meta-val", value));
    meta.appendChild(row);
  };

  if (kind === "reminder") {
    const dueStr = fields.due_iso
      ? formatIsoDatetime(fields.due_iso)
      : (fields.due_text ? `(no parseada) “${fields.due_text}”` : "sin fecha");
    addMeta("cuándo", dueStr);
    if (fields.list) addMeta("lista", fields.list);
    const prioMap = { 1: "alta", 5: "media", 9: "baja" };
    if (fields.priority) addMeta("prioridad", prioMap[fields.priority] || String(fields.priority));
    if (fields.notes) addMeta("nota", fields.notes);
  } else {
    const startStr = fields.start_iso
      ? formatIsoDatetime(fields.start_iso)
      : `(no parseado) “${fields.start_text || ""}”`;
    addMeta("inicio", startStr);
    if (fields.end_iso) addMeta("fin", formatIsoDatetime(fields.end_iso));
    if (fields.calendar) addMeta("calendario", fields.calendar);
    if (fields.location) addMeta("lugar", fields.location);
    if (fields.all_day) addMeta("todo el día", "sí");
    if (fields.notes) addMeta("nota", fields.notes);
  }
  if (fields.recurrence) {
    const rec = fields.recurrence;
    const human = fields.recurrence_text
      || `${rec.freq.toLowerCase()}${rec.interval > 1 ? ` · cada ${rec.interval}` : ""}`;
    addMeta("repite", human);
  }
  card.appendChild(meta);

  if (needsClarif) {
    card.appendChild(el(
      "div", "proposal-warn",
      "⚠ Falta fecha/hora clara — pedí al agente que aclare o descartá.",
    ));
  }

  const actions = el("div", "proposal-actions");
  const createBtn = document.createElement("button");
  createBtn.type = "button";
  createBtn.className = "proposal-btn proposal-btn-create";
  createBtn.textContent = "✓ Crear";
  createBtn.disabled = needsClarif;
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "proposal-btn proposal-btn-cancel";
  cancelBtn.textContent = "✗ Descartar";
  const status = el("span", "proposal-status", "");
  actions.appendChild(createBtn);
  actions.appendChild(cancelBtn);
  actions.appendChild(status);
  card.appendChild(actions);

  cancelBtn.addEventListener("click", () => {
    if (card.dataset.resolved) return;
    card.dataset.resolved = "cancelled";
    createBtn.disabled = true;
    cancelBtn.disabled = true;
    status.textContent = "  descartada";
    status.classList.add("cancelled");
    card.classList.add("dimmed");
  });

  createBtn.addEventListener("click", async () => {
    if (card.dataset.resolved) return;
    card.dataset.resolved = "creating";
    createBtn.disabled = true;
    cancelBtn.disabled = true;
    status.textContent = "  creando…";
    status.classList.remove("ok", "err");

    const url = kind === "event" ? "/api/calendar/create" : "/api/reminders/create";
    const body = kind === "event"
      ? {
          title: fields.title,
          start_iso: fields.start_iso,
          end_iso: fields.end_iso,
          calendar: fields.calendar,
          location: fields.location,
          notes: fields.notes,
          all_day: fields.all_day,
          recurrence: fields.recurrence,
        }
      : {
          text: fields.title,
          due_iso: fields.due_iso,
          list: fields.list,
          priority: fields.priority,
          notes: fields.notes,
          recurrence: fields.recurrence,
        };

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      card.dataset.resolved = "created";
      status.textContent = kind === "event" ? "  ✓ evento creado" : "  ✓ recordatorio creado";
      status.classList.add("ok");
      card.classList.add("created");
      // Prominent confirmation. Visible regardless of chat scroll state.
      showToast(
        kind === "event"
          ? "✓ Agregado a tu Calendario"
          : "✓ Agregado a tus Recordatorios",
        "ok",
      );
    } catch (err) {
      delete card.dataset.resolved;
      createBtn.disabled = false;
      cancelBtn.disabled = false;
      status.textContent = `  error: ${err.message}`;
      status.classList.add("err");
      showToast(
        kind === "event"
          ? `✗ No se pudo agregar al Calendario: ${err.message}`
          : `✗ No se pudo agregar a Recordatorios: ${err.message}`,
        "err",
      );
    }
  });

  parent.appendChild(card);
  return card;
}

function appendSources(parent, items, confidence) {
  const wrap = el("div", "sources");
  const head = el("div", "sources-rule");
  head.textContent = "╌ fuentes ";
  if (Number.isFinite(confidence)) head.appendChild(confidenceBadge(confidence));
  wrap.appendChild(head);
  const seen = new Set();
  for (const s of items) {
    if (seen.has(s.file)) continue;
    seen.add(s.file);
    const row = el("div", "source-row");
    const filled = (s.bar.match(/■/g) || []).length;
    const tone = filled >= 4 ? "good" : filled >= 2 ? "mid" : "low";
    const bar = el("span", `bar bar-${tone}`);
    bar.textContent = s.bar;
    row.appendChild(bar);
    const note = el("a", "note", s.note || s.file);
    note.href = obsidianUrl(s.file);
    note.title = s.file;
    row.appendChild(note);
    const path = el("a", "path", s.file);
    path.href = obsidianUrl(s.file);
    path.title = s.file;
    row.appendChild(path);
    wrap.appendChild(row);
  }
  parent.appendChild(wrap);
}

// Build the markdown exported by the copy button. Includes the question,
// the assistant answer (already markdown), and a `## Fuentes` list with
// `[[Nota]]` wikilinks so pasting into Obsidian yields clickable links.
// Trailing ISO timestamp so the clip is self-dating when pasted into a
// note or message.
function buildMarkdownExport(question, answer, sources) {
  const parts = [];
  if (question && question.trim()) {
    parts.push(`## Pregunta\n\n${question.trim()}`);
  }
  if (answer && answer.trim()) {
    parts.push(`## Respuesta\n\n${answer.trim()}`);
  }
  if (Array.isArray(sources) && sources.length) {
    const seen = new Set();
    const lines = ["## Fuentes", ""];
    for (const s of sources) {
      if (!s || !s.file || seen.has(s.file)) continue;
      seen.add(s.file);
      const note = s.note || s.file.replace(/\.md$/, "").split("/").pop();
      const score = Number.isFinite(s.score) ? ` · ${(s.score >= 0 ? "+" : "") + s.score.toFixed(1)}` : "";
      lines.push(`- [[${note}]] — \`${s.file}\`${score}`);
    }
    parts.push(lines.join("\n"));
  }
  parts.push(`_via rag · ${new Date().toISOString().slice(0, 19).replace("T", " ")}_`);
  return parts.join("\n\n");
}

// Copy button — sits on the .line holding the rag response; copies raw
// markdown (fullText), not rendered HTML, so it pastes cleanly into
// Obsidian, notes, or another chat.
function appendCopyButton(parent, getText) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg-action copy-btn";
  btn.setAttribute("aria-label", "copiar respuesta");
  btn.title = "copiar markdown";
  btn.innerHTML = `${COPY_SVG}<span class="msg-action-label">copiar</span>`;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getText());
      btn.classList.add("done");
      btn.querySelector(".msg-action-label").textContent = "copiado";
      setTimeout(() => {
        btn.classList.remove("done");
        btn.querySelector(".msg-action-label").textContent = "copiar";
      }, 1200);
    } catch {
      btn.classList.add("err");
    }
  });
  parent.appendChild(btn);
  return btn;
}

// TTS playback — called when toggle is on and stream finishes. Aborts any
// previously in-flight audio so rapid turns don't stack overlapping voices.
async function speak(text) {
  if (!ttsEnabled || !text || !text.trim()) return;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.slice(0, 1500) }),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url));
    currentAudio = audio;
    audio.play().catch(() => {});
  } catch {}
}

// Follow-up chips — generated post-done from the last turn's context.
// Clicking a chip re-submits that question as a new turn.
async function appendFollowups(parent, sid) {
  try {
    const res = await fetch("/api/followups", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const arr = (data.followups || []).filter((x) => typeof x === "string");
    if (!arr.length) return;
    const wrap = el("div", "followups");
    wrap.appendChild(el("span", "followups-label", "seguir con ›"));
    for (const q of arr) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "follow-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => {
        if (pending) return;
        input.value = q;
        autoGrow();
        form.requestSubmit();
      });
      wrap.appendChild(chip);
    }
    parent.appendChild(wrap);
  } catch {}
}

function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function obsidianUrl(filePath) {
  return "obsidian://open?file=" + encodeURIComponent(filePath);
}

// Markdown via marked. Dos transformaciones se aplican ANTES de marked:
//   • <<ext>>…<</ext>>  → <span class="ext">⚠ …</span>
//   • [[Wikilinks]]      → [Wikilinks](obsidian://…)
marked.use({
  breaks: true,
  gfm: true,
  renderer: {
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens);
      const isNote = href && href.endsWith(".md") && !href.startsWith("http");
      const target = isNote ? obsidianUrl(href) : href;
      const titleAttr = title ? ` title="${title}"` : "";
      const ext = !isNote && /^https?:\/\//.test(href) ? ` target="_blank" rel="noopener noreferrer"` : "";
      return `<a href="${target}"${titleAttr}${ext}>${text}</a>`;
    },
    // Drop raw HTML blocks/inline emitted by the LLM. marked passes them
    // through by default which is an XSS surface when the model halluci-
    // nates <script>/<iframe>/<img onerror> tags or a vault note contains
    // pasted HTML. Escaping to text preserves the visible content but
    // removes executable markup.
    html({ text }) {
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    },
  },
});

// Post-render sanitiser: parses the marked output into a detached DOM,
// strips disallowed elements + dangerous attributes, returns cleaned HTML.
// Belt-and-suspenders with the renderer.html override above — catches
// anything that slipped through (mostly on-event attrs inside crafted
// link titles). No external dep.
const _SAFE_TAGS = new Set([
  "a", "abbr", "b", "blockquote", "br", "code", "del", "div", "em",
  "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li", "ol",
  "p", "pre", "s", "span", "strong", "sub", "sup", "table", "tbody",
  "td", "th", "thead", "tr", "ul",
]);
const _SAFE_ATTRS = new Set([
  "href", "title", "alt", "src", "target", "rel", "class", "id",
  "colspan", "rowspan", "start", "type",
]);

function _sanitizeNode(node) {
  // Walk in reverse so we can remove children without indexing surprises.
  const children = Array.from(node.childNodes);
  for (const child of children) {
    if (child.nodeType === Node.ELEMENT_NODE) {
      const tag = child.tagName.toLowerCase();
      if (!_SAFE_TAGS.has(tag)) {
        child.remove();
        continue;
      }
      // Kill all on* handlers + anything not in the safe-attr list.
      for (const attr of Array.from(child.attributes)) {
        const name = attr.name.toLowerCase();
        if (name.startsWith("on") || !_SAFE_ATTRS.has(name)) {
          child.removeAttribute(attr.name);
          continue;
        }
        // href/src must not be javascript: / data: (except data:image/…)
        if (name === "href" || name === "src") {
          const val = attr.value.trim().toLowerCase();
          const isJs = val.startsWith("javascript:");
          const isDataNonImg = val.startsWith("data:") && !val.startsWith("data:image/");
          if (isJs || isDataNonImg) {
            child.removeAttribute(attr.name);
          }
        }
      }
      _sanitizeNode(child);
    }
  }
}

function _sanitizeHtml(html) {
  const doc = new DOMParser().parseFromString(
    `<div id="__root">${html}</div>`, "text/html"
  );
  const root = doc.getElementById("__root");
  if (!root) return "";
  _sanitizeNode(root);
  return root.innerHTML;
}

function preprocess(text) {
  // Accept malformed closings (LLM frequently drops one `<` or `>`):
  //   <<ext>>…<</ext>>   canonical
  //   <<ext>>…</ext>>    one `<` dropped  ← caso real observado
  //   <<ext>>…<</ext>    one `>` dropped
  //   <<ext>>…</ext>     both dropped
  let out = text.replace(/<<ext>>([\s\S]*?)<{1,2}\/ext>{1,2}/g, (_, body) => {
    return `\u0000EXT_OPEN\u0000${body}\u0000EXT_CLOSE\u0000`;
  });
  // Strip any stray opener/closer that survived (no partner found).
  out = out.replace(/<{1,2}\/?ext>{1,2}/g, "");
  out = out.replace(/\[\[([^\]]+)\]\]/g, (_, name) => {
    return `[${name}](${name}.md)`;
  });
  return out;
}

function postprocess(html) {
  return html
    .replace(/\u0000EXT_OPEN\u0000/g, '<span class="ext">⚠ ')
    .replace(/\u0000EXT_CLOSE\u0000/g, "</span>");
}

function renderMarkdown(text) {
  return _sanitizeHtml(postprocess(marked.parse(preprocess(text))));
}

// Send --------------------------------------------------------
async function send(question) {
  if (pending) return;
  pending = true;
  lastUserQuestion = question;
  pushHistory(question);
  input.disabled = true;
  stopBtn.hidden = false;
  currentController = new AbortController();

  const turn = appendTurn();
  appendLine(turn, "user", question);
  const thinking = el("div", "thinking");
  thinking.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  // Seed the stage label immediately so the user sees "pensando…" on click.
  // Heartbeat/status events replace this with "buscando…" / "generando…"
  // once they start flowing (~500ms–3s in). Rendered as real DOM (not a
  // ::after pseudo) so the seconds portion can be coloured independently
  // as a semáforo (verde → amarillo → rojo) once the ticker kicks in.
  const stageLabelEl = el("span", "stage-label", "pensando…");
  const stageSecsEl = el("span", "stage-secs");
  thinking.appendChild(stageLabelEl);
  thinking.appendChild(stageSecsEl);
  turn.appendChild(thinking);
  scrollBottom();

  let ragText = null;
  let ragLine = null;
  let fullText = "";
  let sources = null;
  let confidence = null;
  let hadProposal = false;
  let metaShown = false;
  let aborted = false;
  // Live tickers for the two long waits (retrieve + generate). Server fires
  // one `status` event per phase and then silence for 2-10s while the
  // reranker + LLM chew. Without a running counter the dots look frozen
  // after the first second. One ticker at a time — start a new phase and
  // the previous stops. 200ms refresh = one decimal fidelity without
  // shaking the layout (tabular-nums in CSS locks the glyph width).
  let stageTimer = null;
  let stageStart = 0;
  function stopStageTicker() {
    if (stageTimer) {
      clearInterval(stageTimer);
      stageTimer = null;
    }
  }
  function formatSecs(ms) {
    // 1 decimal up to 9.9s, integer after — avoids visual noise past the
    // threshold where sub-second fidelity stops being useful.
    const s = ms / 1000;
    return s < 10 ? s.toFixed(1) : Math.floor(s).toString();
  }
  function stageTier(ms) {
    // Semáforo agnóstico a la fase: los mismos umbrales que disparan el
    // cambio de copy en retrieveLabel. Verde < 3s (snappy), amarillo < 8s
    // (trabajando), rojo ≥ 8s (percibido como lento). Mantenerlos
    // alineados con la copy evita que el color y el texto discrepen.
    const s = ms / 1000;
    if (s < 3) return "green";
    if (s < 8) return "yellow";
    return "red";
  }
  function retrieveLabel(ms) {
    const s = ms / 1000;
    // "búsqueda profunda" matches the auto-deep-retrieval branch in the
    // backend (confidence < 0.10).
    if (s < 3) return "buscando";
    if (s < 8) return "revisando notas";
    return "búsqueda profunda";
  }
  function generateLabel(ms) {
    const s = ms / 1000;
    if (s < 8)  return "generando";
    if (s < 15) return "casi listo";
    return "todavía trabajando";
  }
  function startStageTicker(phase) {
    stopStageTicker();
    stageStart = performance.now();
    const labelFn = phase === "generating" ? generateLabel : retrieveLabel;
    // El contador y el semáforo sólo tienen sentido durante `generating`:
    // el usuario quiere medir "cuánto lleva generándome la respuesta", no
    // "cuánto lleva el pipeline entero". Durante `retrieving` actualizamos
    // sólo la label (buscando → revisando notas → búsqueda profunda) y
    // mantenemos limpio el span de segundos para que el número arranque
    // desde 0 recién cuando empieza la generación.
    const showSecs = phase === "generating";
    const tick = () => {
      if (!thinking.isConnected) { stopStageTicker(); return; }
      const elapsed = performance.now() - stageStart;
      stageLabelEl.textContent = showSecs ? `${labelFn(elapsed)} · ` : labelFn(elapsed);
      if (showSecs) {
        stageSecsEl.textContent = `${formatSecs(elapsed)}s`;
        stageSecsEl.setAttribute("data-tier", stageTier(elapsed));
      } else {
        stageSecsEl.textContent = "";
        stageSecsEl.removeAttribute("data-tier");
      }
    };
    tick();
    stageTimer = setInterval(tick, 200);
  }
  // Legacy names kept for minimal diff at call sites below.
  const stopGeneratingTicker = stopStageTicker;
  const startGeneratingTicker = () => startStageTicker("generating");
  // Tool-call progress chips — persist across the thinking→token boundary
  // so `status {stage:"tool"}` events keep rendering after the dots
  // disappear. Chips are appended in fire order; duplicates across rounds
  // coexist (each status event = one chip).
  let toolsBar = null;
  const toolChips = [];
  function ensureToolsBar() {
    if (toolsBar) return toolsBar;
    toolsBar = el("div", "tools-bar");
    // Insert right after the user line so chips sit above thinking/reply.
    if (thinking.isConnected) turn.insertBefore(toolsBar, thinking);
    else turn.appendChild(toolsBar);
    return toolsBar;
  }
  function clearToolChips() {
    if (toolsBar && toolsBar.isConnected) toolsBar.remove();
    toolsBar = null;
    toolChips.length = 0;
  }
  function stopAllChipPulses() {
    for (const c of toolChips) c.classList.remove("pending");
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId, vault_scope: vaultScope || null }),
      signal: currentController.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleEvent(raw);
      }
    }
  } catch (err) {
    stopGeneratingTicker();
    if (err && err.name === "AbortError") {
      aborted = true;
      if (thinking.isConnected) thinking.remove();
      if (ragText) ragText.classList.remove("pending");
      turn.appendChild(el("div", "meta", "  ⏹ detenido"));
    } else {
      thinking.remove();
      turn.appendChild(el("div", "error", `  error: ${err.message}`));
    }
  } finally {
    stopGeneratingTicker();
    pending = false;
    input.disabled = false;
    stopBtn.hidden = true;
    currentController = null;
    input.value = "";
    autoGrow();
    input.focus();
  }

  function handleEvent(raw) {
    const lines = raw.split("\n");
    let event = "message";
    let data = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!data) return;
    let parsed;
    try { parsed = JSON.parse(data); } catch { return; }

    if (event === "session") {
      sessionId = parsed.id;
      sessionStorage.setItem(SESSION_KEY, sessionId);
    } else if (event === "meta") {
      if (!metaShown) {
        appendMeta(turn, parsed.bits);
        metaShown = true;
      }
    } else if (event === "sources") {
      sources = parsed.items;
      if (Number.isFinite(parsed.confidence)) confidence = parsed.confidence;
      // Server marks propose-intent turns so we know to skip the vault
      // sources panel + web-search button (irrelevant when the user is
      // CREATING something, not asking about existing notes).
      if (parsed.propose_intent) hadProposal = true;
    } else if (event === "proposal") {
      // The server emits this after a propose_reminder /
      // propose_calendar_event tool resolves. We render the card
      // immediately so the user sees it while the LLM's narrative
      // keeps streaming. The thinking spinner (if still showing) is
      // torn down on the next token event.
      hadProposal = true;
      appendProposal(turn, parsed);
      scrollBottom();
    } else if (event === "token") {
      if (!ragLine) {
        stopGeneratingTicker();
        thinking.remove();
        ragLine = document.createElement("div");
        ragLine.className = "line";
        const prompt = el("span", "prompt rag", "rag ›");
        ragText = el("span", "text rag pending");
        ragLine.appendChild(prompt);
        ragLine.appendChild(ragText);
        turn.appendChild(ragLine);
      }
      fullText += parsed.delta;
      ragText.innerHTML = renderMarkdown(fullText);
      scrollBottom();
    } else if (event === "done") {
      if (ragText) {
        ragText.classList.remove("pending");
        ragText.innerHTML = renderMarkdown(fullText);
      }
      const conf = Number.isFinite(confidence) ? confidence : parsed.top_score;
      // On propose-intent turns skip the sources panel AND the
      // web-search fallback — vault retrieval / googling are irrelevant
      // when the user asked the system to CREATE a reminder/event.
      if (!hadProposal && sources && sources.length) {
        appendSources(turn, sources, conf);
      }
      if (!hadProposal && question && (!sources || !sources.length || (Number.isFinite(conf) && conf < 1.0))) {
        appendWebSearch(turn, question);
      }
      const feedbackBar = parsed.turn_id
        ? appendFeedback(turn, {
            turn_id: parsed.turn_id,
            q: question,
            paths: (sources || []).map((s) => s.file).filter(Boolean),
            session_id: sessionId,
          })
        : null;
      if (fullText.trim()) {
        if (feedbackBar) {
          appendCopyButton(feedbackBar, () => buildMarkdownExport(question, fullText, sources));
        } else {
          const actions = el("div", "msg-actions");
          appendCopyButton(actions, () => buildMarkdownExport(question, fullText, sources));
          turn.appendChild(actions);
        }
      }
      if (sessionId && !aborted) appendFollowups(turn, sessionId);
      if (ttsEnabled && !aborted && fullText.trim()) speak(fullText);
      scrollBottom();
    } else if (event === "empty") {
      stopGeneratingTicker();
      thinking.remove();
      clearToolChips();
      turn.appendChild(el("div", "empty", `  ${parsed.message || "Sin resultados relevantes."}`));
      if (question) appendWebSearch(turn, question);
    } else if (event === "error") {
      stopGeneratingTicker();
      thinking.remove();
      clearToolChips();
      turn.appendChild(el("div", "error", `  ${parsed.message || "Error"}`));
    } else if (event === "heartbeat") {
      // Server-side heartbeat is a liveness signal. Only act on it as a
      // fallback if no client-side ticker is running for this phase (e.g.
      // the `status` event that would start the ticker never arrived).
      // Otherwise the local 200ms ticker produces smoother copy and the
      // heartbeat would flicker it back to the server's 1s cadence.
      if (!stageTimer) {
        if (parsed.stage === "generating") startStageTicker("generating");
        else startStageTicker("retrieving");
      }
    } else if (event === "status") {
      if (parsed.stage === "tool") {
        const bar = ensureToolsBar();
        const chip = el("span", "tool-chip pending");
        chip.dataset.tool = parsed.name || "";
        chip.textContent = `Consultando ${toolLabel(parsed.name)}…`;
        bar.appendChild(chip);
        toolChips.push(chip);
        scrollBottom();
      } else if (parsed.stage === "tool_done") {
        // Resolve the oldest still-pending chip for this tool name — works
        // for both sequential rounds and parallel fan-outs where chips
        // share a name. Falls back to no-op if nothing matches.
        const name = parsed.name || "";
        const pending = toolChips.find(
          (c) => c.dataset.tool === name && c.classList.contains("pending"),
        );
        if (pending) {
          pending.classList.remove("pending");
          pending.textContent = `${toolLabel(name)} (${formatMs(parsed.ms)})`;
        }
      } else {
        if (parsed.stage === "generating") {
          stopAllChipPulses();
          startGeneratingTicker();
        } else if (parsed.stage === "retrieving") {
          // Server emits `status {stage:"retrieving"}` at the very start
          // of retrieve(); run the counter from there so the user sees
          // exactly the time the pipeline has been working.
          startStageTicker("retrieving");
        } else if (parsed.stage === "cached") {
          // Cache replay is sub-100ms end-to-end — no point showing a
          // running counter. Give a quick visual cue and let the normal
          // sources/token/done events tear down `thinking`.
          stopStageTicker();
          stageLabelEl.textContent = "desde caché";
          stageSecsEl.textContent = "";
          stageSecsEl.removeAttribute("data-tier");
        } else {
          // Unknown/unlabelled status (e.g. a future stage name) — fall
          // back to the retrieve ticker. Guarantees the counter never
          // stalls at a static label again.
          if (!stageTimer) startStageTicker("retrieving");
        }
      }
    }
  }
}

// Slash commands ------------------------------------------------
// Note: most commands are local (instant). The ones that touch disk
// (/save, /reindex) hit dedicated POST endpoints so they don't go
// through the retrieve pipeline. /cls wipes DOM only, /new mints a
// fresh session id (server-side session is kept on disk, TTL-reaped).
function pushSystemMessage(kind, text) {
  const turn = appendTurn();
  const cls = kind === "err" ? "error" : "meta";
  turn.appendChild(el("div", cls, `  ${text}`));
  scrollBottom();
}

async function handleSlashCommand(raw) {
  const trimmed = raw.trim();
  const lower = trimmed.toLowerCase();
  const spaceIdx = trimmed.indexOf(" ");
  const cmd = (spaceIdx >= 0 ? trimmed.slice(0, spaceIdx) : trimmed).toLowerCase();
  const arg = spaceIdx >= 0 ? trimmed.slice(spaceIdx + 1).trim() : "";

  if (cmd === "/cls" || cmd === "/clear") {
    messagesEl.innerHTML = "";
    input.value = "";
    autoGrow();
    input.focus();
    return true;
  }
  if (cmd === "/help" || cmd === "/?") {
    openHelp();
    input.value = "";
    autoGrow();
    return true;
  }
  if (cmd === "/new") {
    sessionId = null;
    sessionStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(SESSION_KEY); // legacy cleanup
    messagesEl.innerHTML = "";
    input.value = "";
    autoGrow();
    pushSystemMessage("meta", "nueva sesión — historial en blanco");
    return true;
  }
  if (cmd === "/tts") {
    ttsEnabled = !ttsEnabled;
    localStorage.setItem(TTS_KEY, ttsEnabled ? "1" : "0");
    renderTtsToggle();
    pushSystemMessage("meta", `voz ${ttsEnabled ? "activada" : "silenciada"}`);
    input.value = "";
    autoGrow();
    return true;
  }
  if (cmd === "/session") {
    input.value = "";
    autoGrow();
    if (!sessionId) {
      pushSystemMessage("meta", "sin sesión activa — escribí algo para crear una");
      return true;
    }
    try {
      const res = await fetch(`/api/session/${encodeURIComponent(sessionId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const s = await res.json();
      const bits = [
        `id: ${s.id}`,
        `turns: ${s.turns}`,
        s.first_q ? `primera: "${s.first_q}"` : "",
        s.updated_at ? `actualizada: ${s.updated_at.slice(0, 19).replace("T", " ")}` : "",
      ].filter(Boolean);
      pushSystemMessage("meta", bits.join(" · "));
    } catch (err) {
      pushSystemMessage("err", `session: ${err.message}`);
    }
    return true;
  }
  if (cmd === "/model") {
    input.value = "";
    autoGrow();
    try {
      const res = await fetch("/api/model");
      const data = await res.json();
      pushSystemMessage("meta", `modelo de chat: ${data.model}`);
    } catch (err) {
      pushSystemMessage("err", `model: ${err.message}`);
    }
    return true;
  }
  if (cmd === "/save") {
    input.value = "";
    autoGrow();
    if (!sessionId) {
      pushSystemMessage("err", "no hay sesión para guardar");
      return true;
    }
    try {
      const res = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, title: arg || null }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      pushSystemMessage("meta", `guardado en: ${data.path}`);
    } catch (err) {
      pushSystemMessage("err", `save: ${err.message}`);
    }
    return true;
  }
  if (cmd === "/reindex") {
    input.value = "";
    autoGrow();
    try {
      const res = await fetch("/api/reindex", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      pushSystemMessage("meta", "reindex lanzado en background (incremental)");
    } catch (err) {
      pushSystemMessage("err", `reindex: ${err.message}`);
    }
    return true;
  }
  // Unknown /foo — let send() handle it so the LLM can answer instead of
  // silently swallowing a mistyped command.
  return false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  if (q.startsWith("/")) {
    const handled = await handleSlashCommand(q);
    if (handled) return;
  }
  send(q);
});

// Click the "rag" title in the topbar → clear the visible conversation.
// Mirrors `/cls`: wipes the DOM + aborts in-flight, but keeps the session
// id and the up-arrow query history intact. Server-side turns and arrow
// history survive so the user can keep the thread going after clicking.
const brand = document.querySelector(".topbar-title");
if (brand) {
  brand.style.cursor = "pointer";
  brand.setAttribute("role", "button");
  brand.setAttribute("tabindex", "0");
  brand.setAttribute("title", "Click para limpiar la vista");
  const clearView = () => {
    if (currentController) currentController.abort();
    messagesEl.innerHTML = "";
    input.value = "";
    autoGrow();
    input.focus();
  };
  brand.addEventListener("click", clearView);
  brand.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      clearView();
    }
  });
}

// Auto-submit when arriving from a deep-link like /chat?q=foo
(() => {
  const params = new URLSearchParams(window.location.search);
  const seed = params.get("q");
  if (!seed) return;
  history.replaceState({}, "", window.location.pathname);
  input.value = seed;
  setTimeout(() => send(seed), 50);
})();
