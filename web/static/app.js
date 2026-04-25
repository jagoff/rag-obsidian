const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const vaultPicker = document.getElementById("vault-picker");
const chatModeToggle = document.getElementById("chat-mode-toggle");
const ttsToggle = document.getElementById("tts-toggle");
const helpBtn = document.getElementById("help-btn");
const helpModal = document.getElementById("help-modal");
const stopBtn = document.getElementById("stop-btn");
// Mobile Tier 1 controls — pueden ser null en páginas legacy que todavía
// no renderizan el HTML nuevo (deployment window). Todos los accesos van
// gated con optional chaining para que la ausencia no crashee el boot.
const sendBtn = document.getElementById("send-btn");
const menuBtn = document.getElementById("menu-btn");
const menuSheet = document.getElementById("menu-sheet");
const sheetVaultPicker = document.getElementById("sheet-vault-picker");

const sheetTtsToggle = document.getElementById("sheet-tts-toggle");

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
// Side-effect fetches (related, followups, contacts popover, etc) que
// arrancan tras el done event y no están atadas al ciclo principal de
// /api/chat. Si el user navega a otra sesión / `/new` antes de que la
// respuesta llegue, hay que abortarlos — sino el render se aplica al
// turn equivocado o al DOM ya limpio. abortSideFetches() corre en
// /new, loadSession(), y triggerNewChat. 2026-04-25 a11y audit lote 2.
const inflightSideFetches = new Set();
function abortSideFetches() {
  for (const ac of inflightSideFetches) {
    try { ac.abort(); } catch (_) {}
  }
  inflightSideFetches.clear();
}
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
    // No active descendant when the popover is empty — clear the attr so
    // screen readers don't announce a stale id.
    historyPopover.removeAttribute("aria-activedescendant");
    historyPopover.removeAttribute("aria-owns");
    input.removeAttribute("aria-owns");
    return;
  }
  historyPopoverItems.forEach((q, i) => {
    const row = el("div", "history-item" + (i === historyPopoverIdx ? " active" : ""));
    // Stable per-row id so aria-activedescendant can point at it. Unique
    // per open of the popover; regenerated on every render (simpler than
    // caching across re-renders).
    row.id = `history-option-${i}`;
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", i === historyPopoverIdx ? "true" : "false");
    row.appendChild(el("span", "history-idx", String(i + 1)));
    row.appendChild(el("span", "history-q", q));
    row.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      pickHistoryEntry(q);
    });
    row.addEventListener("mouseenter", () => {
      historyPopoverIdx = i;
      [...historyPopover.children].forEach((c, j) => {
        c.classList.toggle("active", j === historyPopoverIdx);
        // Keep aria-selected in sync with the visual state so screen
        // readers announce the same item the user is hovering.
        c.setAttribute("aria-selected", j === historyPopoverIdx ? "true" : "false");
      });
      historyPopover.setAttribute("aria-activedescendant", `history-option-${historyPopoverIdx}`);
    });
    historyPopover.appendChild(row);
  });
  // Point aria-activedescendant at whichever row is currently highlighted.
  // Browsers use this (on a role="listbox") to tell assistive tech which
  // option inside the listbox has focus, without the listbox itself losing
  // keyboard focus. Required for ↑/↓ navigation to be announced.
  historyPopover.setAttribute("aria-activedescendant", `history-option-${historyPopoverIdx}`);
  // a11y: aria-owns sobre el input (= elemento focuseado) declara la
  // relación con los items del popover en el AX tree, ya que el DOM
  // no los contiene como descendientes. Sin esto VoiceOver / NVDA no
  // saben que el aria-activedescendant referenciado existe en el
  // árbol de contenido. Espejamos también en el popover (harmless
  // redundancy: el listbox enumera sus opciones por aria-owns).
  const ownIds = historyPopoverItems.map((_, i) => `history-option-${i}`).join(" ");
  historyPopover.setAttribute("aria-owns", ownIds);
  input.setAttribute("aria-owns", ownIds);
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
  historyPopover.removeAttribute("aria-activedescendant");
  historyPopover.removeAttribute("aria-owns");
  // Limpiamos aria-owns del input solo si era de history. El input
  // puede tener slash/contacts ownership al mismo tiempo no: los 3
  // popovers son mutuamente exclusivos, así que clear sin chequeo es
  // seguro.
  input.removeAttribute("aria-owns");
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

// Chat mode toggle (auto/fast/deep) ---------------------------------
// Reemplaza el antiguo model-picker. El mode le dice al backend cómo
// despachar la generación:
//   · auto:   let adaptive routing decide (default, recomendado)
//   · fast:   fuerza qwen2.5:3b (respuestas literales, ~2s)
//   · deep:   fuerza qwen2.5:7b (razonamiento, ~8s)
// Persiste en localStorage; se manda como `mode` en el POST /api/chat.
// El endpoint legacy POST /api/chat/model sigue existiendo como escape
// hatch para devs (env var OBSIDIAN_RAG_WEB_CHAT_MODEL o override file).
const CHAT_MODE_KEY = "rag-chat-mode";
const VALID_MODES = new Set(["auto", "fast", "deep"]);

function getChatMode() {
  const raw = localStorage.getItem(CHAT_MODE_KEY);
  return VALID_MODES.has(raw) ? raw : "auto";
}

function setChatMode(mode) {
  if (!VALID_MODES.has(mode)) mode = "auto";
  localStorage.setItem(CHAT_MODE_KEY, mode);
  if (!chatModeToggle) return;
  for (const btn of chatModeToggle.querySelectorAll(".chat-mode-btn")) {
    btn.setAttribute(
      "aria-checked",
      btn.dataset.mode === mode ? "true" : "false",
    );
  }
}

if (chatModeToggle) {
  chatModeToggle.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".chat-mode-btn");
    if (btn) setChatMode(btn.dataset.mode);
  });
  // Keyboard navigation: arrow keys cycle between the 3 radios (ARIA
  // radiogroup contract). Left/Up → anterior; Right/Down → siguiente.
  chatModeToggle.addEventListener("keydown", (ev) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) {
      return;
    }
    ev.preventDefault();
    const buttons = Array.from(
      chatModeToggle.querySelectorAll(".chat-mode-btn"),
    );
    const current = buttons.findIndex(
      (b) => b.getAttribute("aria-checked") === "true",
    );
    const dir =
      ev.key === "ArrowRight" || ev.key === "ArrowDown" ? 1 : -1;
    const next = (current + dir + buttons.length) % buttons.length;
    setChatMode(buttons[next].dataset.mode);
    buttons[next].focus();
  });
  setChatMode(getChatMode()); // init from localStorage
}

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
// Focus trap WCAG 2.4.3 / 2.4.7: cuando el modal se abre, Tab cicla
// dentro; Shift+Tab al primer elemento revuelve al último (y vice-
// versa). Al cerrar, restaura el focus al elemento que abrió el modal
// (defaulteable a `document.activeElement` previo). Esc cierra el
// modal y está manejado en el `keydown` global más abajo.
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex=\"-1\"])",
].join(", ");
let helpLastFocus = null;
function _helpTrap(e) {
  if (e.key !== "Tab") return;
  const f = [...helpModal.querySelectorAll(FOCUSABLE_SELECTOR)]
    .filter((el) => !el.hasAttribute("hidden") && el.offsetParent !== null);
  if (!f.length) return;
  const first = f[0];
  const last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}
function openHelp() {
  helpLastFocus = document.activeElement;
  helpModal.hidden = false;
  // Focus al primer elemento focusable del modal — típicamente el
  // botón "×" del header. preventScroll evita que el page jump al
  // entrar.
  const first = helpModal.querySelector(FOCUSABLE_SELECTOR);
  if (first) first.focus({ preventScroll: true });
  helpModal.addEventListener("keydown", _helpTrap);
}
function closeHelp() {
  helpModal.hidden = true;
  helpModal.removeEventListener("keydown", _helpTrap);
  // Restaurar focus al trigger que abrió el modal (helpBtn,
  // composer-plus-btn, o ⌘/ shortcut → activeElement previo). Si el
  // elemento se removió del DOM mientras el modal estaba abierto,
  // fallback al body.
  if (helpLastFocus && typeof helpLastFocus.focus === "function" &&
      document.contains(helpLastFocus)) {
    try { helpLastFocus.focus({ preventScroll: true }); } catch (_) {}
  }
  helpLastFocus = null;
}
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
  { cmd: "/redo",    desc: "regenerar última respuesta (opcional: pista)", arg: "[pista]" },
  { cmd: "/tts",     desc: "alternar voz (Mónica)" },
  // Shortcuts para acciones de creación — reescriben a lenguaje natural
  // y van al pipeline LLM normal (tool-calling con propose_*). La UI
  // rendereiza la tarjeta de confirmación como cualquier otra propuesta
  // — el slash solo es azúcar sintáctica para evitarle al user tipear
  // "mandale un mensaje a X que diga ...". `/wzp` y `/mail` muestran
  // además un popover fuzzy con los contactos cacheados (ver
  // `contactsPopover` más abajo).
  { cmd: "/wzp",     desc: "enviar WhatsApp a un contacto", arg: "[contacto]: [mensaje]" },
  { cmd: "/mail",    desc: "enviar email (Gmail)", arg: "[email]: [asunto] — [cuerpo]" },
  { cmd: "/rem",     desc: "crear recordatorio", arg: "[texto] [cuándo]" },
  { cmd: "/evt",     desc: "agendar evento", arg: "[título] [cuándo]" },
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
    slashPopover.removeAttribute("aria-activedescendant");
    slashPopover.removeAttribute("aria-owns");
    input.removeAttribute("aria-owns");
    return;
  }
  items.forEach((c, i) => {
    const row = el("div", "slash-item" + (i === slashIndex ? " active" : ""));
    // Per-item id so the listbox-level aria-activedescendant on the
    // container can point at the currently-highlighted row without the
    // row itself needing keyboard focus.
    row.id = `slash-option-${i}`;
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", i === slashIndex ? "true" : "false");
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
  slashPopover.setAttribute("aria-activedescendant", `slash-option-${slashIndex}`);
  // a11y: aria-owns sobre input + popover (ver comentario detallado en
  // renderHistoryPopover). El input "posee" los items en el AX tree.
  const ownIds = items.map((_, i) => `slash-option-${i}`).join(" ");
  slashPopover.setAttribute("aria-owns", ownIds);
  input.setAttribute("aria-owns", ownIds);
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
  slashPopover.removeAttribute("aria-activedescendant");
  slashPopover.removeAttribute("aria-owns");
  input.removeAttribute("aria-owns");
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

// ── Popover de contactos para /wzp + /mail ────────────────────────────
// Activa cuando el input matchea `/wzp <texto>` o `/mail <texto>`. Muestra
// hasta 20 contactos del cache de Apple Contacts filtrados por substring
// normalizado (sin acentos) — fetch a /api/contacts. Reusa la
// infraestructura visual del slash-popover (mismo CSS, mismo aria-listbox).
//
// Decisiones de diseño:
//   - Debounce 80ms entre fetches: balance entre "sentir vivo" mientras
//     tipeás y no martillar el endpoint con cada keystroke. El backend
//     filtra ~350 contactos en sub-ms; el bottleneck es el round-trip.
//   - El popover convive con el slash-popover sin colisionar: cuando el
//     user todavía no metió espacio (estado `/wzp`), muestra el slash
//     popover; cuando metió espacio (`/wzp `), cierra el slash y abre
//     el contacts popover.
//   - "Letra-por-letra" como pidió Fer (2026-04-24): cada keystroke
//     dispara updateContactsPopover. No esperamos hasta que termine de
//     tipear el contacto entero — eso permite descubrir contactos que
//     no recuerda con qué letra arrancan ("Fer" matchea "Alfredo").
const contactsPopover = document.getElementById("contacts-popover");
let contactsIndex = 0;
let contactsItems = [];          // [{name, phones, emails, score}]
let contactsFetchAbort = null;   // AbortController del fetch en curso
let contactsDebounceTimer = null;
let contactsLastQuery = null;    // string|null — último query ya fetchedo

function _contactsCommandSpec(value) {
  // Devuelve {cmd, kind, query} si el input está en estado `/wzp <X>`
  // o `/mail <X>` (con al menos un espacio post-comando). Null en otro
  // caso. `query` es el texto tras el espacio — vacío al recién tipear
  // el espacio inicial.
  if (!value || !value.startsWith("/")) return null;
  const spaceIdx = value.indexOf(" ");
  if (spaceIdx < 0) return null;
  const cmd = value.slice(0, spaceIdx).toLowerCase();
  if (cmd !== "/wzp" && cmd !== "/mail") return null;
  // Solo el primer "argumento" — si el user ya escribió `: <mensaje>`,
  // el contacto ya quedó atrás y no queremos seguir mostrando el picker.
  // Detectamos eso buscando un `:` en el resto.
  const rest = value.slice(spaceIdx + 1);
  const colonIdx = rest.indexOf(":");
  if (colonIdx >= 0) return null;
  // El "query" es lo que el user tipeó como nombre de contacto, hasta el
  // próximo espacio (single-token) — para soportar nombres compuestos
  // ("Maria Jose") tomamos todo el rest sin trim para no perder el space.
  return {
    cmd,
    kind: cmd === "/wzp" ? "phone" : "email",
    query: rest,
  };
}

function renderContactsPopover(items) {
  contactsPopover.innerHTML = "";
  if (!items.length) {
    contactsPopover.appendChild(el(
      "div", "slash-popover-empty",
      "sin contactos — Enter manda al chat igual",
    ));
    contactsPopover.removeAttribute("aria-activedescendant");
    contactsPopover.removeAttribute("aria-owns");
    input.removeAttribute("aria-owns");
    return;
  }
  items.forEach((c, i) => {
    const row = el("div", "slash-item" + (i === contactsIndex ? " active" : ""));
    row.id = `contacts-option-${i}`;
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", i === contactsIndex ? "true" : "false");
    row.dataset.name = c.name;
    // Estructura visual: nombre del contacto a la izquierda, hint
    // (primer phone o email truncado) a la derecha. Reusa las mismas
    // clases que el slash popover para consistencia.
    const nameSpan = el("span", "slash-cmd", c.name);
    row.appendChild(nameSpan);
    let hint = "";
    if (c.phones && c.phones.length) hint = c.phones[0];
    else if (c.emails && c.emails.length) hint = c.emails[0];
    if (hint) row.appendChild(el("span", "slash-desc", hint));
    row.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      pickContact(c);
    });
    row.addEventListener("mouseenter", () => {
      contactsIndex = i;
      renderContactsPopover(contactsItems);
    });
    contactsPopover.appendChild(row);
  });
  contactsPopover.setAttribute("aria-activedescendant", `contacts-option-${contactsIndex}`);
  // a11y: aria-owns sobre input + popover (ver comentario en
  // renderHistoryPopover). Sin esto el aria-activedescendant queda
  // huérfano para los screen readers que validan ownership.
  const ownIds = items.map((_, i) => `contacts-option-${i}`).join(" ");
  contactsPopover.setAttribute("aria-owns", ownIds);
  input.setAttribute("aria-owns", ownIds);
}

function hideContactsPopover() {
  contactsPopover.hidden = true;
  contactsItems = [];
  contactsIndex = 0;
  contactsLastQuery = null;
  contactsPopover.removeAttribute("aria-activedescendant");
  contactsPopover.removeAttribute("aria-owns");
  input.removeAttribute("aria-owns");
  if (contactsFetchAbort) {
    try { contactsFetchAbort.abort(); } catch (_) {}
    contactsFetchAbort = null;
  }
  if (contactsDebounceTimer) {
    clearTimeout(contactsDebounceTimer);
    contactsDebounceTimer = null;
  }
}

async function _doContactsFetch(spec) {
  // Cancelamos el fetch anterior si quedó en vuelo — el último query
  // gana. Sin esto, una respuesta lenta podría sobreescribir la UI
  // después de que el user ya tipeó más letras.
  if (contactsFetchAbort) {
    try { contactsFetchAbort.abort(); } catch (_) {}
    inflightSideFetches.delete(contactsFetchAbort);
  }
  const ac = new AbortController();
  contactsFetchAbort = ac;
  // También registramos en el set global de side-fetches para que
  // /new / loadSession / triggerNewChat aborten el contact lookup
  // cuando cambia el contexto de chat. Borrado en finally abajo.
  inflightSideFetches.add(ac);
  try {
    const url = `/api/contacts?q=${encodeURIComponent(spec.query.trim())}` +
                `&kind=${spec.kind}&limit=20`;
    const res = await fetch(url, { signal: ac.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    // Por si el spec cambió entre el fetch y la respuesta — comparamos
    // contra el estado actual del input. Si el user salió del estado
    // contacts-popover (ej. tipeó `:`), descartamos.
    const liveSpec = _contactsCommandSpec(input.value);
    if (!liveSpec || liveSpec.cmd !== spec.cmd || liveSpec.kind !== spec.kind) {
      return;
    }
    contactsItems = data.contacts || [];
    contactsIndex = 0;
    contactsPopover.hidden = false;
    renderContactsPopover(contactsItems);
  } catch (err) {
    if (err.name === "AbortError") return;
    // Silent-fail: el popover no debe gritar al usuario. Cerramos y
    // dejamos que tipee el nombre a mano.
    hideContactsPopover();
  } finally {
    inflightSideFetches.delete(ac);
  }
}

function updateContactsPopover() {
  const spec = _contactsCommandSpec(input.value);
  if (!spec) {
    if (!contactsPopover.hidden) hideContactsPopover();
    return;
  }
  // Cerrar el slash popover si está abierto — el contacts popover toma
  // su lugar visual una vez que el user pasó el espacio post-comando.
  if (!slashPopover.hidden) hideSlashPopover();
  // Misma query que la última fetched → no re-pedimos.
  const queryKey = `${spec.kind}:${spec.query}`;
  if (queryKey === contactsLastQuery) return;
  contactsLastQuery = queryKey;
  // Debounce 80ms — si el user sigue tipeando, se cancela.
  if (contactsDebounceTimer) clearTimeout(contactsDebounceTimer);
  contactsDebounceTimer = setTimeout(() => {
    contactsDebounceTimer = null;
    _doContactsFetch(spec);
  }, 80);
}

function pickContact(c) {
  // Reemplaza el "query" actual (texto post-comando) por el nombre
  // elegido + ": " para que el user pase a tipear el mensaje. El cmd
  // (`/wzp` o `/mail`) se preserva; el resto del input también si el
  // user había escrito algo después (defensive — normalmente no hay
  // nada).
  const spec = _contactsCommandSpec(input.value);
  if (!spec) {
    hideContactsPopover();
    return;
  }
  // Para `/mail` queremos el email, no el nombre — el LLM necesita un
  // address válido. Para `/wzp` el nombre alcanza (propose_whatsapp_send
  // resuelve el JID).
  let pickedToken = c.name;
  if (spec.kind === "email") {
    pickedToken = (c.emails && c.emails[0]) || c.name;
  }
  input.value = `${spec.cmd} ${pickedToken}: `;
  autoGrow();
  input.focus();
  // Caret al final para que el user empiece a tipear el mensaje.
  const len = input.value.length;
  input.setSelectionRange(len, len);
  hideContactsPopover();
}

// Input autogrow + enter-to-send --------------------------------
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

// Send button: enabled ↔ disabled según si hay texto en el input.
// Mobile-only UI (desktop usa Enter), pero el update es barato y corre
// en ambos contextos. Evita mandar un submit vacío si el user tapea
// el botón con input en blanco.
function updateSendBtnState() {
  if (!sendBtn) return;
  const hasText = input.value.trim().length > 0;
  // Si estamos en medio de un stream el send está oculto (stop-btn
  // visible); el disabled no importa en ese caso, pero lo seteamos
  // por prolijidad del estado interno.
  sendBtn.disabled = !hasText || pending;
}

input.addEventListener("input", () => {
  autoGrow();
  updateSendBtnState();
  updateSlashPopover();
  // El contacts popover se actualiza también en cada keystroke. Si el
  // input no matchea `/wzp <X>` ni `/mail <X>`, la función se cierra
  // sola — barato (un check de prefix + indexOf).
  updateContactsPopover();
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
  // Contacts popover navigation — misma UX que el slash popover. Cuando
  // está visible (tipo /wzp <q> o /mail <q>) las flechas mueven el
  // highlight, Tab/Enter pickean, Escape cierra y deja al user seguir
  // tipeando libre. Si está vacío (no hubo matches) no interceptamos
  // teclas — Enter manda el chat con el texto literal.
  if (!contactsPopover.hidden && contactsItems.length) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      contactsIndex = (contactsIndex + 1) % contactsItems.length;
      renderContactsPopover(contactsItems);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      contactsIndex = (contactsIndex - 1 + contactsItems.length) % contactsItems.length;
      renderContactsPopover(contactsItems);
      return;
    }
    if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
      e.preventDefault();
      const pick = contactsItems[contactsIndex];
      if (pick) pickContact(pick);
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      hideContactsPopover();
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
    // Solo abrir el popover de history cuando la flecha sería un no-op
    // en el texto — o sea, cursor al inicio exacto (ArrowUp) / al
    // final exacto (ArrowDown). Antes la regla era "la primera / última
    // línea visual", lo que rompía la UX en mensajes multi-línea:
    // moverse por el texto con las flechas disparaba el popover y
    // molestaba (Fer F. 2026-04-24). Con caret === 0 / caret === length
    // el user recupera el comportamiento estándar de textarea — ArrowUp
    // en medio del texto mueve el cursor una línea arriba, sin popover.
    // Selección activa (start !== end) también bloquea — no queremos
    // "pisar" una selección del user.
    const v = input.value;
    const caretStart = input.selectionStart;
    const caretEnd = input.selectionEnd;
    const hasSelection = caretStart !== caretEnd;
    const edgeOk = !hasSelection && (e.key === "ArrowUp"
      ? caretStart === 0
      : caretStart === v.length);
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
  setTimeout(() => {
    hideSlashPopover();
    hideHistoryPopover();
    hideContactsPopover();
  }, 120);
});
input.addEventListener("focus", () => {
  updateSlashPopover();
  updateContactsPopover();
});

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

// ── Copy-as-implicit-positive (2026-04-22) ──────────────────────────────
// When the user copies text from a RAG response, that's a stronger
// implicit positive than a 👍 — they're actually using the content.
// We attribute the copy to (turn_id, query, top_source_path) via the
// data-* attributes the `done` handler pins on the turn wrap, and POST
// it to /api/behavior with event="copy".
//
// Gates, in order:
//   1. Selection text length ≥ 20 chars — below that it's usually a
//      label fragment / path piece, not real content.
//   2. Selection must be inside a `.turn[data-turn-id]` — skips copies
//      from system messages (/help, /stats), composer, topbar, etc.
//   3. data-top-path must be present — means the turn had a vault-
//      relative top source (cross-source turns don't get it).
//   4. Fire-and-forget — no retry, no await on the outer event.
//
// Observable in `rag_behavior`:
//   event='copy' source='web' path=<top> query=<q> rank=1
// Feeds the ranker-vivo via _compute_behavior_priors_from_rows() the
// same way clicks/opens do — higher-quality signal because the user
// took action.
document.addEventListener("copy", () => {
  try {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;
    const text = (sel.toString() || "").trim();
    if (text.length < 20) return;
    // The ancestor `.turn` that owns this selection — use the anchor
    // node, since that's where the selection started.
    let node = sel.anchorNode;
    if (node && node.nodeType === Node.TEXT_NODE) node = node.parentNode;
    if (!node) return;
    const turn = node.closest && node.closest(".turn[data-turn-id]");
    if (!turn) return;
    const payload = {
      source: "web",
      event: "copy",
      query: turn.dataset.q || null,
      path: turn.dataset.topPath || null,
      rank: turn.dataset.topPath ? 1 : null,
      session: turn.dataset.session || null,
    };
    // No path → no signal worth logging (ranker-vivo keys on path).
    if (!payload.path) return;
    fetch("/api/behavior", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,  // survive tab close / nav mid-flight
    }).catch(() => { /* silent — copy shouldn't nag the user */ });
  } catch {
    /* Any DOM / selection API hiccup: swallow. The user's copy still
       worked natively; we just lost the telemetry. */
  }
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

// Confidence badge — mirrors server's `confidence_badge()` in rag.py.
// Calibrated 2026-04-21 against real `rag_queries.top_score` distribution
// (n=904): p50=0.14, p75=0.48, p95=0.97. Pre-calibración usaba thresholds
// para score range [-5, 10] que nunca se triggereaban — todo se renderizaba
// "media · amarillo" independientemente del quality real. Constantes
// sincronizadas con rag.py `SCORE_BADGE_LOW_HIGH` / `SCORE_BADGE_MID_HIGH`.
function confidenceBadge(score) {
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
// Redo icon — circular arrow. Click on the feedback bar regenerates
// the last response (same server path as `/redo`). Matches the stroke
// weight of THUMB_*_SVG / COPY_SVG so the row stays visually uniform.
const REDO_SVG = `
  <svg class="fb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M3 12a9 9 0 1 0 3-6.7"/>
    <polyline points="3 4 3 10 9 10"/>
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
          rating: rating,
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

  // 👎 flow: pre-2026-04-22 was a plain text input ("¿qué faltó?"). Now
  // mirrors the CLI corrective_path prompt (rag.py:~18997): show the top-5
  // source cards as selectable, plus "ninguna" option + optional free-text.
  // Selecting a card inserts a clean (query, positive) pair into rag_feedback
  // that `rag tune` consumes as augmentation — see _feedback_augmented_cases().
  function openNegativeFeedback() {
    if (wrap.dataset.reasonOpen || wrap.dataset.sent) return;
    wrap.dataset.reasonOpen = "1";
    down.classList.add("picked");
    up.classList.add("dimmed");

    const row = el("div", "feedback-corrective");

    // Filter candidates: vault-relative only (skip calendar://, whatsapp://
    // etc — same invariant the server enforces on the way in). De-dupe,
    // cap at 5, match the CLI's candidate-build logic.
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
    const header = el("div", "fb-corr-header",
      hasCandidates
        ? "¿cuál era el path correcto? elegí uno o escribí el tuyo — opcional"
        : "¿qué faltó? (opcional)"
    );
    row.appendChild(header);

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
            // Toggle off.
            selectedPath = null;
            card.classList.remove("picked");
            card.setAttribute("aria-pressed", "false");
          } else {
            selectedPath = path;
            cardNodes.forEach((c) => {
              c.classList.remove("picked");
              c.setAttribute("aria-pressed", "false");
            });
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
          cardNodes.forEach((c) => {
            c.classList.remove("picked");
            c.setAttribute("aria-pressed", "false");
          });
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
    const send = document.createElement("button");
    send.type = "button";
    send.className = "fb-text-btn";
    send.textContent = "enviar";

    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "fb-text-btn fb-text-muted";
    skip.textContent = "omitir";

    async function commit() {
      const freeText = field.value.trim();
      let corrective = null;
      let reason = null;
      // Precedence: free-text path input > card selection > "ninguna".
      // "ninguna" leaves corrective null but preserves the rating signal +
      // any free-text reason. A free-text entry that looks like a path
      // (contains "/" or ends with .md) is treated as corrective_path;
      // otherwise as a reason note.
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
    send.addEventListener("click", commit);
    skip.addEventListener("click", commit);

    actions.appendChild(field);
    actions.appendChild(send);
    actions.appendChild(skip);
    row.appendChild(actions);
    wrap.appendChild(row);
    (hasCandidates ? cardNodes[0] || field : field).focus();
  }

  // Redo button — regenerate without hint. Separate from `/redo <hint>`
  // (keyboard path); this is the one-click "give me another take".
  // Disabled when the user already submitted a rating (rare: you don't
  // typically redo *after* giving feedback — that pushes new telemetry
  // that muddies the signal for the original turn).
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
    send("(redo)", { redo_turn_id: ctx.turn_id });
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

// Small HTML escape — enough for titles and paths (no markup expected),
// keeps us off innerHTML-injection for user-controlled frontmatter titles.
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// External enrichment — Spotify + YouTube, fetched when the vault answer
// is weak (low confidence, empty retrieval). One block, only renders if
// the backend returned at least one item.
async function appendRelated(parent, query) {
  if (!query) return;
  let items = [];
  // AbortController registrado en el set global para que loadSession /
  // /new lo cancelen si el user navega antes de que el fetch resuelva.
  const ac = new AbortController();
  inflightSideFetches.add(ac);
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
  } catch (err) {
    // AbortError es esperado cuando el user navega — silent. Cualquier
    // otro error es de red / parsing y también lo silenciamos (el
    // bloque "related" es decorativo, no crítico).
    return;
  } finally {
    inflightSideFetches.delete(ac);
  }
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
    const sub = el("span", "related-sub", it.subtitle || "");
    row.appendChild(badge);
    row.appendChild(title);
    if (it.subtitle) row.appendChild(sub);
    wrap.appendChild(row);
  }
  parent.appendChild(wrap);
}

// Web-search escape hatch — surfaces when the vault has weak/no answer
// (sin sources, o confianza baja). One click → Google búsqueda en pestaña
// nueva. El usuario decide si vale la pena salir del vault.
function appendWebSearch(parent, query, inline = false) {
  const link = document.createElement("a");
  link.className = "web-search-link" + (inline ? " inline" : "");
  link.href = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "↗ buscar en internet";
  link.title = `Google: ${query}`;
  if (inline) {
    parent.appendChild(document.createTextNode(" "));
    parent.appendChild(link);
  } else {
    const wrap = el("div", "web-search");
    wrap.appendChild(link);
    parent.appendChild(wrap);
  }
}

// Fallback cluster — cuando el backend emite `done` con
// `low_conf_bypass=true`, el vault no tenía info útil y el sistema
// devolvió un template fijo (no LLM call). En vez de ofrecer solo el
// link sutil "↗ buscar en internet", renderamos un cluster prominente
// de 3 botones (Google / YouTube / Wikipedia) para dar escape fácil.
// Todos los botones son links directos (no API calls, no trackeo).
// Sigue apareciendo el bloque `📎 contexto relacionado` (YouTube videos
// de `/api/related`) cuando aplique, DEBAJO del cluster.
function appendFallbackCluster(parent, query) {
  if (!query) return;
  const wrap = el("div", "fallback-cluster");
  wrap.appendChild(el(
    "div", "fallback-head",
    "no encontré eso en tus notas — ¿querés que busque en...?",
  ));
  const buttons = el("div", "fallback-buttons");
  const q = encodeURIComponent(query);
  const specs = [
    { cls: "fallback-google",   label: "🔍 Google",    url: `https://www.google.com/search?q=${q}` },
    { cls: "fallback-youtube",  label: "▶ YouTube",    url: `https://www.youtube.com/results?search_query=${q}` },
    { cls: "fallback-wiki",     label: "📖 Wikipedia", url: `https://es.wikipedia.org/wiki/Special:Search?search=${q}` },
  ];
  for (const s of specs) {
    const a = document.createElement("a");
    a.className = `fallback-btn ${s.cls}`;
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = s.label;
    a.title = `${s.label.replace(/^[^\s]+\s*/, '')}: ${query}`;
    buttons.appendChild(a);
  }
  wrap.appendChild(buttons);
  parent.appendChild(wrap);
}

// Cross-source enrichment footer: WhatsApp/Calendar/Reminders signals
// surfaced after the answer streams. Each line: icon + text + optional
// snippet (italic) + relative time (right-aligned).
function appendEnrich(parent, lines) {
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

// NLI grounding panel — claim-level verdicts (entails / neutral / contradicts).
// Emitted after `done` when RAG_NLI_GROUNDING=1. Collapsed by default so the
// response stays clean; power users can expand to inspect the evidence.
function renderGrounding(data, container) {
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
    const icon = claim.verdict === "entails"     ? "✓"
               : claim.verdict === "contradicts" ? "✗" : "·";
    li.appendChild(document.createTextNode(`${icon} ${claim.text}`));
    if (claim.evidence_note) {
      li.appendChild(el("small", "grounding-note", ` (${claim.evidence_note})`));
    }
    ul.appendChild(li);
  }
  details.appendChild(ul);
  container.appendChild(details);
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
// auto-dismisses after 4s (or 10s if it has an action button), stacks
// if multiple fire.
//
// Options:
//   kind: "ok" | "err" | "info"            (left-edge accent color)
//   ms:   milliseconds before auto-dismiss (default 4000, action 10000)
//   action: {label, onClick}                optional inline button
function showToast(message, opts = {}) {
  // Back-compat: accept second arg as a string kind.
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
    toast.addEventListener(
      "transitionend",
      () => toast.remove(),
      { once: true },
    );
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
      try {
        await action.onClick();
      } finally {
        dismiss();
      }
    });
    toast.appendChild(btn);
  }

  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("toast-visible"));
  dismissTimer = setTimeout(dismiss, ms);
  return { dismiss };
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

// ── Created chips (Calendar event / Reminder auto-create confirmations) ───
//
// Rendered inline below the LLM's response text, same family as the
// `╌ fuentes` panel. Scrolls naturally with the conversation; no floating
// toasts that cover the top bar. Reminders get an inline `deshacer`
// text-button (Reminders.app's AppleScript delete-by-id is reliable).
// Calendar events are info-only (delete is unworkable, see the earlier
// debug trail in git history).

function formatDateOnly(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleDateString("es-AR", {
      weekday: "short", day: "2-digit", month: "2-digit",
    });
  } catch {
    return iso;
  }
}

// Friendly relative formatter for the WhatsApp scheduling chip.
//   - Same-calendar-day → "hoy 14:30"
//   - Next/previous calendar day → "mañana 9:00" / "ayer 18:00"
//   - Otherwise → es-AR short with weekday + day + month + time
//     (e.g. "vie 26 abr 14:30")
//
// We use *calendar-day* delta (not 24h-window) so 23:59 today and 00:01
// today both show as "hoy ..." and don't flip arbitrarily based on the
// elapsed milliseconds. The browser's local timezone is used for the
// comparison — this matches what the user expects ("mañana" = mañana
// donde está, no donde está el server).
function formatFriendlyDate(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    const now = new Date();
    const dayMs = 24 * 60 * 60 * 1000;
    const startOfDay = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const dayDiff = Math.round((startOfDay(d) - startOfDay(now)) / dayMs);
    const time = d.toLocaleTimeString("es-AR", {
      hour: "2-digit", minute: "2-digit",
    });
    if (dayDiff === 0) return `hoy ${time}`;
    if (dayDiff === 1) return `mañana ${time}`;
    if (dayDiff === -1) return `ayer ${time}`;
    return d.toLocaleString("es-AR", {
      weekday: "short", day: "numeric", month: "short",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return isoStr;
  }
}

// Convert a `<input type="datetime-local">` value (e.g. "2026-04-26T09:00")
// into an ISO8601 string with explicit Argentina offset (-03:00). The
// backend accepts naive ISO too, but explicit-better-than-implicit — if
// the user's browser is somehow in a non-AR timezone, this still ships a
// timestamp the backend interprets as Argentina wall-clock time.
function toIsoArgentina(localDateTime) {
  if (!localDateTime) return "";
  let v = String(localDateTime).trim();
  if (!v) return "";
  // datetime-local usually emits "YYYY-MM-DDTHH:MM"; some browsers add
  // ":SS" when the field is configured with seconds precision. Normalize
  // so the output is always "YYYY-MM-DDTHH:MM:SS-03:00".
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(v)) {
    v = v + ":00";
  }
  return v + "-03:00";
}

// Relative-time formatting for "last contact" line in the WhatsApp card
// header. Buckets:
//   < 60min   → "hace 12 min"
//   < 24h     → "hoy 14:30"
//   1 día     → "ayer 18:30"
//   2 días    → "antes de ayer 09:00"
//   3-6 días  → "hace 3 días"
//   1-3 sem   → "hace 2 semanas"
//   < 1 año   → "hace 4 meses"
//   ≥ 1 año   → "hace 2 años"
// Localiza con TZ del browser usando Date.parse del ISO con offset.
function formatRelativeContact(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return "";
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.round(diffMs / 60000);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (diffMin < 0) {
      return formatFriendlyDate(isoStr);
    }
    if (diffMin < 60) {
      return diffMin <= 1 ? "hace un instante" : `hace ${diffMin} min`;
    }
    const dDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    const nowDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const dayDiff = Math.round((nowDay - dDay) / (1000 * 60 * 60 * 24));
    if (dayDiff === 0) return `hoy ${hh}:${mm}`;
    if (dayDiff === 1) return `ayer ${hh}:${mm}`;
    if (dayDiff === 2) return `antes de ayer ${hh}:${mm}`;
    if (dayDiff < 7) return `hace ${dayDiff} días`;
    if (dayDiff < 30) {
      const weeks = Math.round(dayDiff / 7);
      return weeks === 1 ? "hace 1 semana" : `hace ${weeks} semanas`;
    }
    if (dayDiff < 365) {
      const months = Math.round(dayDiff / 30);
      return months === 1 ? "hace 1 mes" : `hace ${months} meses`;
    }
    const years = Math.round(dayDiff / 365);
    return years === 1 ? "hace 1 año" : `hace ${years} años`;
  } catch (_) {
    return "";
  }
}

// Detector heurístico de fechas en lenguaje natural rioplatense.
// Usado en `appendWhatsAppProposal` cuando el LLM NO populó
// `fields.scheduled_for` pero el `message_text` tiene un patrón de
// fecha (caso típico: el LLM mete "mañana 9hs" en el cuerpo en vez de
// extraerla a `scheduled_for`). Sin esto, el user tendría que abrir
// el popover ⏰ y re-tipear la fecha que ya escribió en el chat —
// doble entrada que el user pidió evitar (2026-04-25).
//
// Soporta los patterns más comunes:
//   "mañana 9hs" / "mañana a las 9" / "mañana 9:30" / "mañana a las 14:30"
//   "hoy 14hs" / "hoy a las 14"
//   "pasado mañana 9hs"
//   "el lunes 14:30" / "el viernes a las 9"
//   "en 2 horas" / "en 30 minutos" / "en una hora"
//
// NO soporta (queda para v2 si hace falta):
//   "el 5 de mayo a las 18hs" (fechas absolutas)
//   "al mediodía" / "a la noche" (dayparts sin hora)
//
// Devuelve `{iso, phrase}` o null. La fecha resuelta SIEMPRE es
// futura (descarta matches que caen en el pasado por > 1min).
function detectDateInMessage(text) {
  if (!text || typeof text !== "string") return null;
  const lower = text.toLowerCase();
  const now = new Date();
  const buildIso = (d) => {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${y}-${m}-${day}T${hh}:${mm}:00-03:00`;
  };
  const isFuture = (d) => d.getTime() > now.getTime() + 60_000;

  const wordToNum = { una: 1, "1": 1, dos: 2, tres: 3, cuatro: 4, cinco: 5,
    seis: 6, siete: 7, ocho: 8, nueve: 9, diez: 10 };

  // 1) "en N horas" / "en N minutos" / "en una hora"
  const rxRel = /\ben\s+(una|un|\d+|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+(horas?|minutos?|min\b|hs?\b)/;
  const mRel = lower.match(rxRel);
  if (mRel) {
    const raw = mRel[1];
    const n = wordToNum[raw] || parseInt(raw, 10) || 1;
    const unit = mRel[2];
    const ms = (unit.startsWith("min") ? n * 60_000 : n * 3_600_000);
    const d = new Date(now.getTime() + ms);
    if (isFuture(d)) return { iso: buildIso(d), phrase: mRel[0] };
  }

  // 2) "pasado mañana ..."
  const rxPasado = /\bpasado\s+ma[nñ]ana(?:\s+(?:a\s+las\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?)?/;
  const mPasado = lower.match(rxPasado);
  if (mPasado) {
    const d = new Date(now);
    d.setDate(d.getDate() + 2);
    const h = mPasado[1] != null ? parseInt(mPasado[1], 10) : 9;
    const min = parseInt(mPasado[2] || "0", 10);
    d.setHours(h, min, 0, 0);
    if (isFuture(d)) return { iso: buildIso(d), phrase: mPasado[0] };
  }

  // 3) "mañana 9hs" / "mañana a las 9" / "mañana 9:30"
  const rxManana = /\bma[nñ]ana(?:\s+(?:a\s+las\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?)?/;
  const mManana = lower.match(rxManana);
  if (mManana) {
    const d = new Date(now);
    d.setDate(d.getDate() + 1);
    // Si dijo solo "mañana" sin hora, asumimos 9am como default razonable.
    const h = mManana[1] != null ? parseInt(mManana[1], 10) : 9;
    const min = parseInt(mManana[2] || "0", 10);
    d.setHours(h, min, 0, 0);
    if (isFuture(d)) return { iso: buildIso(d), phrase: mManana[0] };
  }

  // 4) "hoy 14hs" / "hoy a las 14:30"
  const rxHoy = /\bhoy\s+(?:a\s+las\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?/;
  const mHoy = lower.match(rxHoy);
  if (mHoy) {
    const d = new Date(now);
    const h = parseInt(mHoy[1], 10);
    const min = parseInt(mHoy[2] || "0", 10);
    d.setHours(h, min, 0, 0);
    if (isFuture(d)) return { iso: buildIso(d), phrase: mHoy[0] };
  }

  // 5) "el lunes 14:30" / "el viernes a las 9"
  const rxDow = /\bel\s+(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)(?:\s+(?:a\s+las\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?)?/;
  const mDow = lower.match(rxDow);
  if (mDow) {
    const dowMap = {
      lunes: 1, martes: 2,
      "miércoles": 3, miercoles: 3,
      jueves: 4, viernes: 5,
      "sábado": 6, sabado: 6,
      domingo: 0,
    };
    const target = dowMap[mDow[1]];
    if (target != null) {
      const d = new Date(now);
      let delta = target - d.getDay();
      if (delta <= 0) delta += 7; // siempre el próximo
      d.setDate(d.getDate() + delta);
      const h = mDow[2] != null ? parseInt(mDow[2], 10) : 9;
      const min = parseInt(mDow[3] || "0", 10);
      d.setHours(h, min, 0, 0);
      if (isFuture(d)) return { iso: buildIso(d), phrase: mDow[0] };
    }
  }

  return null;
}

// Limpia la frase de fecha del message_text una vez detectada para
// que no quede "mañana 9hs ya llegué" como cuerpo del mensaje (porque
// si se programa para mañana 9hs, decirlo en el cuerpo es redundante
// y suena raro). Devuelve el texto limpio + un flag.
function stripDatePhraseFromMessage(text, phrase) {
  if (!text || !phrase) return text;
  // Buscar y quitar el match exacto (case-insensitive). Acepta espacios
  // adyacentes y posible coma/punto al lado.
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rx = new RegExp(`\\s*[,.;]?\\s*${escaped}\\s*[,.;]?\\s*`, "i");
  return text.replace(rx, " ").replace(/\s+/g, " ").trim();
}

// Fetch + render del bloque de contexto de WhatsApp dentro del card.
// Llamado desde `appendWhatsAppProposal` después del recipientLine.
// Best-effort:
//   - Si el bridge devuelve count=0 → muestra "Sin mensajes anteriores"
//     en gris suave (no genera ruido cuando es un primer contacto).
//   - Si el fetch falla → silent (mejor que ensuciar con un error de
//     algo opcional).
//   - El thread va dentro de un <details> default-cerrado, con el
//     summary mostrando "<relative> · N mensajes". Click expande.
async function appendWhatsAppContext(card, jid, recipientLabel) {
  let data = null;
  try {
    const res = await fetch(
      `/api/whatsapp/context?jid=${encodeURIComponent(jid)}&limit=5`,
      { method: "GET" },
    );
    if (!res.ok) return;
    data = await res.json();
  } catch (_) {
    return;
  }
  if (!data || typeof data !== "object") return;
  const count = Number(data.messages_count || 0);
  const wrap = el("div", "proposal-wa-context");

  if (count === 0) {
    wrap.classList.add("empty");
    wrap.appendChild(el("span", "proposal-wa-context-meta",
      "Sin mensajes anteriores con este contacto"));
    const anchor = card.querySelector(
      ".proposal-wa-text, .proposal-wa-schedule-chip, .proposal-wa-quote, .proposal-warn",
    );
    if (anchor) card.insertBefore(wrap, anchor);
    else card.appendChild(wrap);
    return;
  }

  const lastIso = data.last_contact_at || "";
  const relativeStr = formatRelativeContact(lastIso) || "";
  const summary = document.createElement("summary");
  summary.className = "proposal-wa-context-summary";
  const metaSpan = el("span", "proposal-wa-context-meta",
    relativeStr ? `Último contacto: ${relativeStr}` : "Mensajes anteriores");
  summary.appendChild(metaSpan);
  summary.appendChild(el("span", "proposal-wa-context-count",
    ` · ${count} mensaje${count === 1 ? "" : "s"}`));

  const details = document.createElement("details");
  details.className = "proposal-wa-context-details";
  details.appendChild(summary);

  const thread = el("div", "proposal-wa-context-thread");
  for (const msg of (data.messages || [])) {
    const bubble = el(
      "div",
      msg.is_from_me
        ? "proposal-wa-context-msg me"
        : "proposal-wa-context-msg them",
    );
    const head = el("div", "proposal-wa-context-msg-head");
    head.appendChild(el("span", "proposal-wa-context-msg-who",
      msg.is_from_me ? "yo" : (msg.who || recipientLabel || "")));
    if (msg.ts) {
      const friendly = formatFriendlyDate(msg.ts);
      if (friendly) {
        head.appendChild(el("span", "proposal-wa-context-msg-ts", ` · ${friendly}`));
      }
    }
    bubble.appendChild(head);
    bubble.appendChild(el("div", "proposal-wa-context-msg-text", msg.text || ""));
    thread.appendChild(bubble);
  }
  details.appendChild(thread);
  wrap.appendChild(details);

  const anchor = card.querySelector(
    ".proposal-wa-text, .proposal-wa-schedule-chip, .proposal-wa-quote, .proposal-warn",
  );
  if (anchor) card.insertBefore(wrap, anchor);
  else card.appendChild(wrap);
}

function appendCreatedChip(parent, payload) {
  const kind = payload.kind;                    // "reminder" | "event"
  const fields = payload.fields || {};
  const chip = el("div", `created-chip created-chip-${kind}`);

  // Left block: divider + ✓ + kind label + title + when.
  const left = el("span", "created-chip-left");
  left.appendChild(el("span", "created-chip-rule", "╌ "));
  left.appendChild(el("span", "created-chip-icon", "✓ "));
  left.appendChild(el(
    "span", "created-chip-kind",
    kind === "event" ? "agregado al calendario" : "agregado a recordatorios",
  ));
  if (fields.title) {
    left.appendChild(el("span", "created-chip-title", ` · ${fields.title}`));
  }

  // When: dates/times formatted es-AR. All-day events show "(todo el día)";
  // timed events show full datetime. Reminders show due_iso or sin-fecha.
  let whenText = "";
  if (kind === "event") {
    if (fields.start_iso) {
      whenText = fields.all_day
        ? ` · ${formatDateOnly(fields.start_iso)} (todo el día)`
        : ` · ${formatIsoDatetime(fields.start_iso)}`;
    }
  } else {
    whenText = fields.due_iso
      ? ` · ${formatIsoDatetime(fields.due_iso)}`
      : " · sin fecha";
  }
  if (whenText) left.appendChild(el("span", "created-chip-when", whenText));
  chip.appendChild(left);

  // Right block: `deshacer` text-button for reminders only (Calendar
  // events don't get it — no working programmatic delete).
  if (kind === "reminder" && payload.reminder_id) {
    const undo = document.createElement("button");
    undo.type = "button";
    undo.className = "created-chip-undo";
    undo.textContent = "deshacer";
    const status = el("span", "created-chip-status", "");
    undo.addEventListener("click", async () => {
      if (chip.dataset.resolved) return;
      chip.dataset.resolved = "pending";
      undo.disabled = true;
      status.textContent = " · deshaciendo…";
      try {
        const res = await fetch("/api/reminders/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reminder_id: payload.reminder_id }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `HTTP ${res.status}`);
        }
        chip.dataset.resolved = "undone";
        chip.classList.add("dimmed");
        undo.remove();
        status.textContent = " · deshecho";
        status.classList.add("status-ok");
      } catch (err) {
        delete chip.dataset.resolved;
        undo.disabled = false;
        status.textContent = ` · error: ${err.message}`;
        status.classList.add("status-err");
      }
    });
    chip.appendChild(undo);
    chip.appendChild(status);
  }

  parent.appendChild(chip);
  return chip;
}


// Whatsapp-specific proposal card. Distinct from the reminder/event
// variant because:
//   1. Fields are different (contact_name + message_text + jid vs
//      title + due/start).
//   2. [Enviar] posts to /api/whatsapp/send (not /api/reminders|calendar/
//      create), and the confirmation is stricter — a mis-send to a
//      third party is NOT undoable (WhatsApp has no "delete sent").
//   3. We show the message body as an editable textarea so the user
//      can tweak before firing. Clicking [Editar] just focuses the
//      textarea and switches [Enviar] from primary→confirm.
//   4. If the contact couldn't be resolved (fields.error ==
//      "not_found" | "no_phone" | "empty_query"), we surface the error
//      inline and disable [Enviar] until the user clarifies.
function appendWhatsAppProposal(parent, payload) {
  const fields = payload.fields || {};
  const proposalId = payload.proposal_id || "";
  const err = fields.error || null;
  // `whatsapp_reply` is the reply variant — adds a quote preview above the
  // textarea and ships `reply_to` in the POST body. `whatsapp_message`
  // (existing send) keeps working unchanged. Distinguishing on `kind`
  // (not just on the presence of `reply_to`) so a missing-target reply
  // still renders as a reply card with the warning visible.
  const isReply = payload.kind === "whatsapp_reply";
  const replyTo = fields.reply_to || null;
  const replyHint = fields.reply_to_hint || "";
  const replyWarn = fields.reply_to_warning || "";

  // Scheduling state — mutable so the user can flip from "send now" to
  // "scheduled" via the ⏰ popover (or change a pre-existing schedule).
  // Starts with whatever the LLM proposed (may be null/undefined). Format:
  // ISO8601 with offset, e.g. "2026-04-26T09:00:00-03:00".
  let scheduledFor = fields.scheduled_for || null;

  // Heurística de fallback: si el LLM no detectó la fecha pero el
  // user la escribió en el cuerpo del mensaje ("mañana 9hs ya llegué"),
  // la extraemos client-side y la auto-aplicamos. Sin esto, el user
  // tendría que abrir el ⏰ y re-tipear lo que ya escribió en el chat
  // — doble entrada que pidió evitar (2026-04-25).
  //
  // `autoDetectedPhrase`: la sub-cadena de fecha que sacamos del
  // body — usado para (a) limpiar el textarea cuando se cree, (b)
  // permitir undo si fue falso positivo, (c) marcar el chip con
  // "(del mensaje)" para que se vea de dónde salió la fecha.
  // `originalMessageText`: snapshot del cuerpo ANTES de limpiar, por
  // si el user clickea "↩ no, mandalo ya".
  let autoDetectedPhrase = null;
  let originalMessageText = fields.message_text || "";
  if (!scheduledFor && fields.message_text) {
    const detected = detectDateInMessage(fields.message_text);
    if (detected) {
      scheduledFor = detected.iso;
      autoDetectedPhrase = detected.phrase;
    }
  }

  const card = el("div", `proposal proposal-whatsapp${isReply ? " proposal-whatsapp-reply" : ""}`);
  // a11y: ver comentario en appendProposal — region landmark con
  // aria-labelledby al heading del card.
  card.setAttribute("role", "region");

  const head = el("div", "proposal-head");
  head.appendChild(el("span", "proposal-icon", isReply ? "↩" : "💬"));
  const waHeadId = `proposal-head-${Math.random().toString(36).slice(2, 9)}`;
  const waHeadLabel = el("span", "proposal-kind", isReply ? "Responder en WhatsApp" : "Mensaje de WhatsApp");
  waHeadLabel.id = waHeadId;
  head.appendChild(waHeadLabel);
  card.setAttribute("aria-labelledby", waHeadId);
  card.appendChild(head);

  const recipientLabel = fields.full_name || fields.contact_name || "(sin destinatario)";
  const recipientHref = waHref(fields.jid || "");
  const recipientLine = el("div", "proposal-title");
  recipientLine.appendChild(document.createTextNode("Para: "));
  if (recipientHref) {
    // Click → abre el chat en WhatsApp app vía wa.me universal link.
    // No reemplaza la acción [Enviar]: solo permite "verificar antes
    // de mandar" abriendo la conversación existente en otro tab.
    const a = el("a", "proposal-wa-link", recipientLabel);
    a.href = recipientHref;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = "Abrir chat con " + recipientLabel + " en WhatsApp";
    recipientLine.appendChild(a);
  } else {
    recipientLine.appendChild(document.createTextNode(recipientLabel));
  }
  card.appendChild(recipientLine);

  // Conversation context — last messages with this contact + last
  // contact date. Replaces the old "seguir con ›" chips below the
  // card (those preguntaban sobre el chat RAG, ruido cuando ya hay
  // un proposal). Fetched async after render so el card aparece
  // instantáneamente y el contexto se llena en ~50-200ms (lectura
  // local de SQLite del bridge). Best-effort: si el bridge está
  // caído o no hay JID, simplemente no se muestra (silent fail).
  if (fields.jid) {
    appendWhatsAppContext(card, fields.jid, recipientLabel);
  }

  // Schedule chip — rendered between the recipient line and the textarea
  // when a scheduled_for is present. Variants:
  //   .pending → grey (default while card is unsent)
  //   .sent    → green (after a successful schedule POST)
  //   .late / .failed reserved for future "scheduled-but-not-fired-yet"
  //   surfacing — see scheduled_messages table in the backend.
  let scheduleChip = null;
  function renderScheduleChip() {
    if (!scheduledFor) {
      if (scheduleChip) {
        scheduleChip.remove();
        scheduleChip = null;
      }
      return;
    }
    if (!scheduleChip) {
      scheduleChip = el("div", "proposal-wa-schedule-chip pending");
      recipientLine.insertAdjacentElement("afterend", scheduleChip);
    }
    // Limpiar children — re-render desde cero (puede tener undo button
    // de un render anterior que ya no aplica).
    scheduleChip.innerHTML = "";
    scheduleChip.appendChild(document.createTextNode(
      `📅 Programado para ${formatFriendlyDate(scheduledFor)}`,
    ));
    if (autoDetectedPhrase) {
      // Indicator visual: el user ve que la fecha se "leyó del mensaje"
      // y no fue invento del agente. Transparencia = trust.
      const hint = el("span", "proposal-wa-schedule-detected",
        ' (detectado del mensaje)');
      scheduleChip.appendChild(hint);
      // Undo button — falso positivo del detector → vuelve al envío
      // inmediato y restaura el body original. Único click reverso.
      const undoBtn = document.createElement("button");
      undoBtn.type = "button";
      undoBtn.className = "proposal-wa-schedule-undo";
      undoBtn.textContent = "↩ no, mandalo ya";
      undoBtn.title = "Cancelar la programación detectada y mandar el mensaje ahora";
      undoBtn.addEventListener("click", () => {
        if (card.dataset.resolved) return;
        scheduledFor = null;
        autoDetectedPhrase = null;
        textarea.value = originalMessageText;
        autoGrow();
        renderScheduleChip();
        if (typeof refreshActionButtons === "function") refreshActionButtons();
      });
      scheduleChip.appendChild(undoBtn);
    }
  }
  renderScheduleChip();

  // Reply variant: render the original message as a styled blockquote
  // above the textarea, mimicking WhatsApp's reply UI (left border in
  // the contact's accent color, muted bg, smaller text).
  if (isReply) {
    if (replyTo) {
      const quote = el("div", "proposal-wa-quote");
      const senderName = recipientLabel;
      const head2 = el("div", "proposal-wa-quote-sender", senderName);
      const text = el("div", "proposal-wa-quote-text",
        (replyTo.original_text || "").trim() || "(mensaje sin texto)");
      const ts = replyTo.original_ts
        ? el("div", "proposal-wa-quote-ts", replyTo.original_ts)
        : null;
      quote.appendChild(head2);
      quote.appendChild(text);
      if (ts) quote.appendChild(ts);
      card.appendChild(quote);
    } else if (replyWarn) {
      // Reply target couldn't be resolved — surface the warning + offer to
      // send without quote anyway. The hint helps the user remember what
      // they originally asked for.
      const hintLabel = replyHint ? ` ("${replyHint}")` : "";
      const warn = el("div", "proposal-warn proposal-wa-warn",
        `⚠ ${replyWarn}${hintLabel ? " " + hintLabel : ""}`);
      card.appendChild(warn);
    }
  }

  // Editable textarea for the message body.
  const textarea = document.createElement("textarea");
  textarea.className = "proposal-wa-text";
  textarea.rows = 3;
  // Si auto-detectamos la fecha en el body del mensaje, limpiamos esa
  // frase del textarea — es absurdo decirle a Grecia "mañana 9hs ya
  // llegué" si el envío ya está programado para mañana 9hs. Si la
  // limpieza dejaría el textarea vacío (ej. user solo dijo "mañana 9hs"
  // sin más texto), mantenemos el original así el card no aparece sin
  // body visible. El undo button restaura todo si fue falso positivo.
  if (autoDetectedPhrase) {
    const cleaned = stripDatePhraseFromMessage(originalMessageText, autoDetectedPhrase);
    textarea.value = cleaned || originalMessageText;
  } else {
    textarea.value = originalMessageText;
  }
  textarea.placeholder = "Texto del mensaje";
  card.appendChild(textarea);

  // Contact-lookup error surfaced inline.
  if (err) {
    const errMap = {
      "not_found":   `No encontré a "${fields.contact_name}" en tus Contactos. Probá con el nombre completo.`,
      "no_phone":    `El contacto "${fields.contact_name}" no tiene un número cargado en Contactos.`,
      "empty_query": `El agente no detectó a quién mandarlo.`,
    };
    const msg = errMap[err] || `No se pudo resolver el destinatario: ${err}`;
    card.appendChild(el("div", "proposal-warn", `⚠ ${msg}`));
  }

  const actions = el("div", "proposal-actions");
  const sendBtn = document.createElement("button");
  sendBtn.type = "button";
  sendBtn.className = "proposal-btn proposal-btn-create";
  sendBtn.disabled = !!err || !fields.jid;

  // Two secondary scheduling buttons that swap based on schedule state:
  //   - clockBtn (⏰ icon-only) → visible when no schedule, opens popover
  //   - changeTimeBtn (⏰ Cambiar hora) → visible when schedule exists,
  //     opens popover pre-filled with current schedule
  // Only one is in the DOM at a time; refreshActionButtons() handles the swap.
  const clockBtn = document.createElement("button");
  clockBtn.type = "button";
  clockBtn.className = "proposal-btn proposal-wa-clock-btn";
  clockBtn.setAttribute("aria-label", "Programar para más tarde");
  clockBtn.title = "Programar para más tarde";
  clockBtn.textContent = "⏰";

  const changeTimeBtn = document.createElement("button");
  changeTimeBtn.type = "button";
  changeTimeBtn.className = "proposal-btn proposal-wa-change-time-btn";
  changeTimeBtn.textContent = "⏰ Cambiar hora";

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "proposal-btn proposal-btn-cancel";
  cancelBtn.textContent = "✗ Descartar";

  const status = el("span", "proposal-status", "");
  actions.appendChild(sendBtn);
  // clockBtn / changeTimeBtn slot is filled by refreshActionButtons()
  actions.appendChild(cancelBtn);
  actions.appendChild(status);
  card.appendChild(actions);

  // Sync the primary button label + which secondary scheduling button is
  // shown, based on whether we have a schedule. Idempotent — safe to call
  // any number of times after scheduledFor changes.
  function refreshActionButtons() {
    if (scheduledFor) {
      sendBtn.textContent = "📅 Programar";
      if (clockBtn.parentNode) clockBtn.remove();
      if (!changeTimeBtn.parentNode) {
        sendBtn.insertAdjacentElement("afterend", changeTimeBtn);
      }
    } else {
      sendBtn.textContent = "✈ Enviar";
      if (changeTimeBtn.parentNode) changeTimeBtn.remove();
      if (!clockBtn.parentNode) {
        sendBtn.insertAdjacentElement("afterend", clockBtn);
      }
    }
  }
  refreshActionButtons();

  // Inline popover for the date/time picker. Toggled by clockBtn / changeTimeBtn.
  // Sits at the bottom of the card (NOT a modal) so the rest of the chat
  // stays scrollable. Single popover instance — reopening reuses or
  // toggles closed depending on current state.
  let popover = null;
  function openPopover() {
    if (card.dataset.resolved) return;
    if (popover) {
      // Toggle: clicking the trigger again closes the popover.
      popover.remove();
      popover = null;
      return;
    }
    popover = el("div", "proposal-wa-schedule-popover");
    popover.setAttribute("role", "dialog");
    popover.setAttribute("aria-label", "Programar mensaje");

    const pad = (n) => String(n).padStart(2, "0");
    const toLocalInputValue = (d) =>
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;

    // Quick-time chips first (above the input) so the most common case —
    // pick a preset, hit Programar — is one tap. The datetime-local input
    // is the fallback for arbitrary times.
    const quick = el("div", "proposal-wa-quick-times");
    const quickBtns = [
      { label: "+15min", offsetMin: 15 },
      { label: "+1h", offsetMin: 60 },
      { label: "+3h", offsetMin: 180 },
      { label: "Mañana 9hs", custom: () => {
        const t = new Date();
        t.setDate(t.getDate() + 1);
        t.setHours(9, 0, 0, 0);
        return t;
      } },
      { label: "Mañana 18hs", custom: () => {
        const t = new Date();
        t.setDate(t.getDate() + 1);
        t.setHours(18, 0, 0, 0);
        return t;
      } },
    ];
    popover.appendChild(quick);

    const inputWrap = el("div", "proposal-wa-schedule-row");
    const inputLabel = el("label", "proposal-wa-schedule-label", "Fecha y hora:");
    const input = document.createElement("input");
    input.type = "datetime-local";
    input.className = "proposal-wa-schedule-input";
    // Pre-fill: existing schedule if any, otherwise now+1h as a sensible default.
    const seedDate = scheduledFor ? new Date(scheduledFor) : new Date(Date.now() + 60 * 60 * 1000);
    if (!isNaN(seedDate.getTime())) {
      input.value = toLocalInputValue(seedDate);
    }
    inputLabel.appendChild(input);
    inputWrap.appendChild(inputLabel);
    popover.appendChild(inputWrap);

    const popoverErr = el("div", "proposal-wa-schedule-err", "");
    popover.appendChild(popoverErr);

    for (const q of quickBtns) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "proposal-wa-quick-time";
      b.textContent = q.label;
      b.addEventListener("click", () => {
        const target = q.custom ? q.custom() : new Date(Date.now() + q.offsetMin * 60 * 1000);
        input.value = toLocalInputValue(target);
        popoverErr.textContent = "";
      });
      quick.appendChild(b);
    }

    const popoverActions = el("div", "proposal-wa-schedule-actions");
    const acceptBtn = document.createElement("button");
    acceptBtn.type = "button";
    acceptBtn.className = "proposal-btn proposal-btn-create";
    acceptBtn.textContent = "Programar";
    const cancelPopBtn = document.createElement("button");
    cancelPopBtn.type = "button";
    cancelPopBtn.className = "proposal-btn proposal-btn-cancel";
    cancelPopBtn.textContent = "Cancelar";
    popoverActions.appendChild(acceptBtn);
    popoverActions.appendChild(cancelPopBtn);
    popover.appendChild(popoverActions);

    cancelPopBtn.addEventListener("click", () => {
      popover.remove();
      popover = null;
    });

    acceptBtn.addEventListener("click", () => {
      popoverErr.textContent = "";
      const v = input.value;
      if (!v) {
        popoverErr.textContent = "Elegí fecha y hora";
        return;
      }
      const target = new Date(v);
      if (isNaN(target.getTime())) {
        popoverErr.textContent = "Fecha inválida";
        return;
      }
      if (target.getTime() <= Date.now()) {
        popoverErr.textContent = "La fecha tiene que ser futura";
        return;
      }
      // Lock in the new schedule and re-render chip + button state.
      // The user still has to hit the primary "📅 Programar" button to POST.
      scheduledFor = toIsoArgentina(v);
      renderScheduleChip();
      refreshActionButtons();
      popover.remove();
      popover = null;
    });

    card.appendChild(popover);
    // Focus the datetime-local input for keyboard-first usage.
    setTimeout(() => { try { input.focus(); } catch {} }, 0);
  }

  clockBtn.addEventListener("click", openPopover);
  changeTimeBtn.addEventListener("click", openPopover);

  cancelBtn.addEventListener("click", () => {
    if (card.dataset.resolved) return;
    card.dataset.resolved = "cancelled";
    sendBtn.disabled = true;
    cancelBtn.disabled = true;
    clockBtn.disabled = true;
    changeTimeBtn.disabled = true;
    textarea.disabled = true;
    if (popover) {
      popover.remove();
      popover = null;
    }
    status.textContent = "  descartada";
    status.classList.add("cancelled");
    card.classList.add("dimmed");
  });

  sendBtn.addEventListener("click", async () => {
    if (card.dataset.resolved) return;
    const body = textarea.value.trim();
    if (!body) {
      status.textContent = "  mensaje vacío";
      status.classList.add("err");
      return;
    }
    if (!fields.jid) {
      status.textContent = "  destinatario no resuelto";
      status.classList.add("err");
      return;
    }
    // Snapshot the current schedule — `scheduledFor` is mutable but this
    // closure should commit to the value at click time so a concurrent
    // popover edit during the in-flight POST doesn't desync the UI.
    const scheduleAtClick = scheduledFor;
    card.dataset.resolved = "sending";
    sendBtn.disabled = true;
    cancelBtn.disabled = true;
    clockBtn.disabled = true;
    changeTimeBtn.disabled = true;
    textarea.disabled = true;
    status.textContent = scheduleAtClick ? "  programando…" : "  enviando…";
    status.classList.remove("ok", "err");

    try {
      const sendBody = {
        jid: fields.jid,
        message_text: body,
        proposal_id: proposalId,
      };
      // Schedule path: ship `scheduled_for` so the backend persists a
      // pending row instead of firing the bridge. Backend response shape:
      //   { ok: true, scheduled: true, id, scheduled_for_utc, status: "pending" }
      if (scheduleAtClick) {
        sendBody.scheduled_for = scheduleAtClick;
      }
      // Forward the reply_to context when this card is a reply with a
      // resolved target. The bridge currently ignores it (no native
      // quote support — see rag._whatsapp_send_to_jid docstring) but
      // the audit log records the message_id so we can correlate.
      if (isReply && replyTo && replyTo.message_id) {
        sendBody.reply_to = {
          message_id: replyTo.message_id,
          original_text: (replyTo.original_text || "").slice(0, 1024),
          sender_jid: replyTo.sender_jid || "",
        };
      }
      const res = await fetch("/api/whatsapp/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sendBody),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data && data.scheduled) {
        // Scheduled path — chip flips to .sent (green) and status shows
        // the friendly date echoed back from the server (`scheduled_for_utc`
        // is the canonical persisted value; falling back to the local
        // schedule if the server didn't include it).
        const when = data.scheduled_for_utc || scheduleAtClick;
        const friendly = formatFriendlyDate(when);
        card.dataset.resolved = "scheduled";
        status.textContent = `  ✓ programado para ${friendly}`;
        status.classList.add("ok");
        card.classList.add("created");
        if (scheduleChip) {
          scheduleChip.classList.remove("pending");
          scheduleChip.classList.add("sent");
          scheduleChip.textContent = `📅 Programado para ${friendly}`;
        }
        showToast(`✓ Programado para ${friendly}`, "ok");
      } else {
        card.dataset.resolved = "sent";
        status.textContent = `  ✓ enviado a ${recipientLabel}`;
        status.classList.add("ok");
        card.classList.add("created");
        showToast(`✓ Mensaje enviado a ${recipientLabel}`, "ok");
      }
    } catch (err) {
      delete card.dataset.resolved;
      sendBtn.disabled = false;
      cancelBtn.disabled = false;
      clockBtn.disabled = false;
      changeTimeBtn.disabled = false;
      textarea.disabled = false;
      status.textContent = `  error: ${err.message}`;
      status.classList.add("err");
      const verb = scheduleAtClick ? "No se pudo programar" : "No se pudo enviar";
      showToast(`✗ ${verb}: ${err.message}`, "err");
    }
  });

  parent.appendChild(card);
  return card;
}


// Mail-specific proposal card. Mismo modelo que WhatsApp — el send es
// destructivo a un tercero, el user confirma con click. Distinto en:
//   1. Tres campos editables: To / Subject / Body (vs un solo textarea).
//   2. POST a /api/mail/send (no /api/whatsapp/send).
//   3. El error de validación ("invalid_email") se surface a la
//      izquierda del campo To en vez de bloquear el botón — el user
//      puede arreglar el destinatario sin descartar el draft.
function appendMailProposal(parent, payload) {
  const fields = payload.fields || {};
  const proposalId = payload.proposal_id || "";
  const err = fields.error || null;

  const card = el("div", "proposal proposal-mail");
  // a11y: ver comentario en appendProposal — region landmark con
  // aria-labelledby al heading del card.
  card.setAttribute("role", "region");

  const head = el("div", "proposal-head");
  head.appendChild(el("span", "proposal-icon", "✉"));
  const mailHeadId = `proposal-head-${Math.random().toString(36).slice(2, 9)}`;
  const mailHeadLabel = el("span", "proposal-kind", "Email");
  mailHeadLabel.id = mailHeadId;
  head.appendChild(mailHeadLabel);
  card.setAttribute("aria-labelledby", mailHeadId);
  card.appendChild(head);

  // Tres inputs/textareas editables. Reusa la clase `.proposal-wa-text`
  // del WhatsApp draft para que el styling sea consistente — son las
  // mismas premisas (text area dentro de la card, focus visible).
  const toInput = document.createElement("input");
  toInput.type = "email";
  toInput.className = "proposal-wa-text";
  toInput.value = fields.to || "";
  toInput.placeholder = "destinatario@ejemplo.com";
  const toRow = el("div", "proposal-meta-row");
  toRow.appendChild(el("span", "proposal-meta-label", "para"));
  toRow.appendChild(toInput);
  card.appendChild(toRow);

  const subjInput = document.createElement("input");
  subjInput.type = "text";
  subjInput.className = "proposal-wa-text";
  subjInput.value = fields.subject || "";
  subjInput.placeholder = "Asunto";
  const subjRow = el("div", "proposal-meta-row");
  subjRow.appendChild(el("span", "proposal-meta-label", "asunto"));
  subjRow.appendChild(subjInput);
  card.appendChild(subjRow);

  const bodyArea = document.createElement("textarea");
  bodyArea.className = "proposal-wa-text";
  bodyArea.rows = 5;
  bodyArea.value = fields.body || "";
  bodyArea.placeholder = "Cuerpo del mail";
  card.appendChild(bodyArea);

  if (err) {
    const errMap = {
      "empty_to": "Falta el destinatario.",
      "invalid_email": `"${fields.to}" no parece un email válido (debe tener @).`,
    };
    const msg = errMap[err] || `Error: ${err}`;
    card.appendChild(el("div", "proposal-warn", `⚠ ${msg}`));
  }

  const actions = el("div", "proposal-actions");
  const sendBtn = document.createElement("button");
  sendBtn.type = "button";
  sendBtn.className = "proposal-btn proposal-btn-create";
  sendBtn.textContent = "✈ Enviar";
  // No deshabilitamos por error — el user puede arreglar el `to` y reintentar.
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "proposal-btn proposal-btn-cancel";
  cancelBtn.textContent = "✗ Descartar";
  const status = el("span", "proposal-status", "");
  actions.appendChild(sendBtn);
  actions.appendChild(cancelBtn);
  actions.appendChild(status);
  card.appendChild(actions);

  cancelBtn.addEventListener("click", () => {
    if (card.dataset.resolved) return;
    card.dataset.resolved = "cancelled";
    sendBtn.disabled = true;
    cancelBtn.disabled = true;
    toInput.disabled = true;
    subjInput.disabled = true;
    bodyArea.disabled = true;
    status.textContent = "  descartado";
    status.classList.add("cancelled");
    card.classList.add("dimmed");
  });

  sendBtn.addEventListener("click", async () => {
    if (card.dataset.resolved) return;
    const toVal = toInput.value.trim();
    const subjVal = subjInput.value.trim();
    const bodyVal = bodyArea.value.trim();
    if (!toVal || !toVal.includes("@")) {
      status.textContent = "  destinatario inválido";
      status.classList.add("err");
      return;
    }
    if (!bodyVal) {
      status.textContent = "  cuerpo vacío";
      status.classList.add("err");
      return;
    }
    card.dataset.resolved = "sending";
    sendBtn.disabled = true;
    cancelBtn.disabled = true;
    toInput.disabled = true;
    subjInput.disabled = true;
    bodyArea.disabled = true;
    status.textContent = "  enviando…";
    status.classList.remove("ok", "err");

    try {
      const res = await fetch("/api/mail/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          to: toVal,
          subject: subjVal || "(sin asunto)",
          body: bodyVal,
          proposal_id: proposalId,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      card.dataset.resolved = "sent";
      status.textContent = `  ✓ enviado a ${toVal}`;
      status.classList.add("ok");
      card.classList.add("created");
      showToast(`✓ Mail enviado a ${toVal}`, "ok");
    } catch (err) {
      delete card.dataset.resolved;
      sendBtn.disabled = false;
      cancelBtn.disabled = false;
      toInput.disabled = false;
      subjInput.disabled = false;
      bodyArea.disabled = false;
      status.textContent = `  error: ${err.message}`;
      status.classList.add("err");
      showToast(`✗ No se pudo enviar mail: ${err.message}`, "err");
    }
  });

  parent.appendChild(card);
  return card;
}


function appendProposal(parent, payload) {
  const kind = payload.kind;                    // "reminder" | "event" | "whatsapp_message" | "whatsapp_reply" | "mail"
  const fields = payload.fields || {};
  const needsClarif = payload.needs_clarification === true;

  // whatsapp_message / whatsapp_reply use their own renderer — different
  // fields (jid, contact_name, message_text + optional reply_to),
  // different actions (Enviar / Editar / Cancelar), different endpoint
  // (/api/whatsapp/send for both — reply ships an extra reply_to field).
  if (kind === "whatsapp_message" || kind === "whatsapp_reply") {
    return appendWhatsAppProposal(parent, payload);
  }
  // mail: tres campos editables (to / subject / body) + POST a /api/mail/send.
  if (kind === "mail") {
    return appendMailProposal(parent, payload);
  }

  const card = el("div", `proposal proposal-${kind}`);
  // a11y: cada propuesta es una región interactiva con form-like
  // semantics (campos + 2 botones). role="region" + aria-labelledby
  // la convierten en un landmark navegable por screen reader.
  card.setAttribute("role", "region");

  const head = el("div", "proposal-head");
  head.appendChild(el("span", "proposal-icon", kind === "event" ? "📅" : "✓"));
  // ID único en el head para que aria-labelledby del card lo
  // referencie. Math.random sirve — son cards efímeras (no se
  // serializan ni se buscan), no necesitamos un counter global.
  const headLabelId = `proposal-head-${Math.random().toString(36).slice(2, 9)}`;
  const headLabel = el(
    "span", "proposal-kind",
    kind === "event" ? "Nuevo evento" : "Nuevo recordatorio",
  );
  headLabel.id = headLabelId;
  head.appendChild(headLabel);
  card.setAttribute("aria-labelledby", headLabelId);
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

// ── Dwell-per-chunk observer (2026-04-22) ──────────────────────────────
// When a source-row scrolls into view we start a timer; when it scrolls
// out we compute the sustained viewport time and — if ≥ the threshold —
// emit a behavior `open` event with dwell_ms attached. This is the
// passive counterpart to the active `copy` event (db2a169) and feeds
// the ranker-vivo with attention signal that a plain "did the user
// click" boolean can't capture.
//
// Design choices:
//   - Single module-level IntersectionObserver (IO) shared across all
//     turns. Per-row observers would multiply listener cost with every
//     new turn and never get garbage-collected.
//   - threshold: [0, 0.5] — we fire at the boundary *and* when half the
//     row is visible. The former marks "user scrolled past"; the latter
//     marks "user actually paused on it".
//   - Minimum dwell: 1500ms (tests can tune it down). Below that the
//     user basically skimmed; above it they read something.
//   - We re-use `event='open'` instead of inventing `dwell` so the
//     existing CTR aggregator in _compute_behavior_priors_from_rows
//     counts it as a click without any schema/aggregator change. The
//     dwell_ms value still rides along for anyone who wants richer
//     analytics later.
//   - Fire-and-forget POST — the user's session never notices our
//     telemetry calls, especially not when they're leaving the page.
const _DWELL_MIN_MS = 1500;
const _DWELL_MAX_MS = 5 * 60 * 1000;  // 5min cap — sanity upper bound
const _dwellStart = new WeakMap();  // row element → Date.now() when entered
const _dwellReported = new WeakSet();  // row elements that already emitted

const _dwellObserver = (typeof IntersectionObserver !== "undefined")
  ? new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const row = entry.target;
        if (entry.isIntersecting) {
          // Start the timer if we haven't already.
          if (!_dwellStart.has(row)) {
            _dwellStart.set(row, Date.now());
          }
        } else {
          // Row left the viewport → compute dwell + maybe emit.
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
    source: "web",
    event: "open",  // reuse the existing positive CTR signal (see commit db2a169)
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
      keepalive: true,  // survive nav/tab close
    }).catch(() => { /* silent — dwell telemetry never blocks UX */ });
  } catch { /* same */ }
}

function _observeDwell(rows) {
  if (!_dwellObserver) return;
  for (const r of rows) _dwellObserver.observe(r);
}

// On page hide (tab backgrounded, nav) flush whatever is still in-flight.
// Without this, a user who reads a source for 10s and then switches tabs
// before scrolling away never emits the event — the IO callback only
// fires on intersection transitions.
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "hidden") return;
    // For every row currently being timed, if the elapsed is within the
    // valid range, flush it now (keepalive=true). Subsequent visibility
    // changes won't double-emit because we add to _dwellReported.
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

function appendSources(parent, items, confidence) {
  const wrap = el("div", "sources");
  const head = el("div", "sources-rule");
  head.textContent = "╌ fuentes ";
  if (Number.isFinite(confidence)) head.appendChild(confidenceBadge(confidence));
  wrap.appendChild(head);
  const seen = new Set();
  // Collect rows for the dwell observer below — we instrument each
  // source-row with data-* attrs so the IntersectionObserver callback
  // can attribute a dwell event to (path, rank, query, session) without
  // a side-channel.
  const rows = [];
  const parentTurn = parent.closest ? parent.closest(".turn") : null;
  let rank = 0;
  for (const s of items) {
    if (seen.has(s.file)) continue;
    seen.add(s.file);
    rank += 1;
    const row = el("div", "source-row");
    // Source-row tone — post-2026-04-21 recalibration del score_bar (mapping
    // [0, 1.0] → 5 cells), con estos thresholds de cells matcheamos los del
    // badge: alta ≥ 0.50 ≈ 3+ cells · media ≥ 0.10 ≈ 1+ cell · baja < 1 cell.
    // Pre-calibración usaba filled >= 4/2/0 bajo un mapping [-5, 10] que
    // concentraba todas las respuestas en el bucket "mid" (2 cells).
    const filled = (s.bar.match(/■/g) || []).length;
    const tone = filled >= 3 ? "good" : filled >= 1 ? "mid" : "low";
    const bar = el("span", `bar bar-${tone}`);
    bar.textContent = s.bar;
    row.appendChild(bar);
    // Tres tipos de fuente con tratamiento distinto:
    //  - External URL (Drive, web docs, etc.): href directo a `target="_blank"`.
    //  - WhatsApp 1:1 (`whatsapp://<phone>@s.whatsapp.net/<msg>`): wa.me/<phone>
    //    universal link que abre el chat en la app de WhatsApp. Grupos
    //    (`@g.us`) no tienen deep-link público — se renderean como texto.
    //  - Vault-local (.md path): obsidian:// para que doble-click abra la nota.
    const isExternal = typeof s.file === "string" && /^https?:\/\//i.test(s.file);
    const isWA = typeof s.file === "string" && s.file.indexOf("whatsapp://") === 0;
    const waUrl = isWA ? waHref(s.file) : "";
    // `wantsBlank` = true para external + WA-with-link → target=_blank.
    const wantsBlank = isExternal || (isWA && waUrl);
    // `linkable` = true cuando podemos generar un href útil. Si es WA
    // group (sin waUrl) renderemos el row sin <a>.
    const linkable = isExternal || waUrl || !isWA;

    let noteEl;
    if (linkable) {
      noteEl = el("a", "note", s.note || s.file);
      noteEl.href = isExternal ? s.file : (waUrl || obsidianUrl(s.file));
      if (wantsBlank) {
        noteEl.target = "_blank";
        noteEl.rel = "noopener noreferrer";
      }
    } else {
      // WA group sin deep-link: span plano para no engañar al user con
      // un link que no hace nada cuando lo toca.
      noteEl = el("span", "note", s.note || s.file);
    }
    noteEl.title = s.file;
    row.appendChild(noteEl);

    // Path/folder label: en externals mostramos el folder ("Google
    // Drive"), en WA mostramos "WhatsApp" en lugar del JID feo, en
    // vault el path .md original.
    let pathLabel;
    if (isExternal) pathLabel = s.folder || "externo";
    else if (isWA) pathLabel = s.folder || "WhatsApp";
    else pathLabel = s.file;

    let pathEl;
    if (linkable) {
      pathEl = el("a", "path", pathLabel);
      pathEl.href = isExternal ? s.file : (waUrl || obsidianUrl(s.file));
      if (wantsBlank) {
        pathEl.target = "_blank";
        pathEl.rel = "noopener noreferrer";
      }
    } else {
      pathEl = el("span", "path", pathLabel);
    }
    pathEl.title = s.file;
    row.appendChild(pathEl);
    // Dwell tracking metadata — vault-relative paths only (the server
    // rejects ones with :// since commit db2a169). Cross-source ids
    // get the attrs dropped so the observer skips them entirely.
    if (s.file && s.file.indexOf("://") === -1) {
      row.dataset.path = s.file;
      row.dataset.rank = String(rank);
      if (parentTurn && parentTurn.dataset.q) row.dataset.q = parentTurn.dataset.q;
      if (parentTurn && parentTurn.dataset.session) row.dataset.session = parentTurn.dataset.session;
    }
    wrap.appendChild(row);
    rows.push(row);
  }
  parent.appendChild(wrap);
  // Attach dwell observer if any rows qualify (skip cross-source-only
  // turns where no row has data-path).
  const trackable = rows.filter((r) => r.dataset && r.dataset.path);
  if (trackable.length) _observeDwell(trackable);
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
      // Externals (Drive, web) get a markdown link instead of an
      // Obsidian wikilink — un `[[Wikilink]]` a un URL externo se
      // rompe cuando pasa por Obsidian o por un pipeline markdown.
      const isExternalSrc = /^https?:\/\//i.test(s.file);
      if (isExternalSrc) {
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

// Copy helper — wraps navigator.clipboard con fallback a execCommand por
// si el browser no está en secure context o el permission fue denegado.
// Devuelve true si la copia funcionó, false si ambos paths fallaron.
async function copyTextToClipboard(text) {
  // Path preferido: Clipboard API moderna. Requiere secure context
  // (HTTPS o localhost). Si el browser no la expone o tira permission
  // error, caemos al fallback.
  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      /* fall through */
    }
  }
  // Fallback: textarea invisible + document.execCommand("copy"). Funciona
  // en HTTP non-localhost y browsers viejos. Deprecated pero sigue vivo
  // en Chrome/Safari/Firefox actuales (2026). El textarea se monta fuera
  // de viewport para no causar scroll/flash.
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.left = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, text.length);
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_) {
    return false;
  }
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
  // a11y: aria-live="polite" sobre el label que muta de
  // "copiar" → "copiado" / "sin texto" / "falló — ⌘C manual" para que
  // VoiceOver / NVDA anuncien el cambio. Sin esto el feedback visual
  // se pierde para usuarios de screen reader. "polite" no interrumpe
  // (no es un error crítico).
  btn.innerHTML = `${COPY_SVG}<span class="msg-action-label" aria-live="polite" aria-atomic="true">copiar</span>`;
  btn.addEventListener("click", async () => {
    const text = typeof getText === "function" ? getText() : "";
    const label = btn.querySelector(".msg-action-label");
    if (!text || !text.trim()) {
      btn.classList.add("err");
      if (label) label.textContent = "sin texto";
      setTimeout(() => {
        btn.classList.remove("err");
        if (label) label.textContent = "copiar";
      }, 1400);
      return;
    }
    const ok = await copyTextToClipboard(text);
    if (ok) {
      btn.classList.add("done");
      if (label) label.textContent = "copiado";
      setTimeout(() => {
        btn.classList.remove("done");
        if (label) label.textContent = "copiar";
      }, 1200);
    } else {
      // Error visible — el user tocó "copiar" y nada pasaba. Ahora al
      // menos se entera y puede seleccionar con ⌘A / ⌘C manualmente.
      btn.classList.add("err");
      if (label) label.textContent = "falló — ⌘C manual";
      setTimeout(() => {
        btn.classList.remove("err");
        if (label) label.textContent = "copiar";
      }, 2400);
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
  // Suprimir followups cuando el turn renderizó un proposal estructurado
  // (whatsapp / mail / calendar / etc). En esos casos el user ya tiene
  // una acción concreta para confirmar y los chips "seguir con ›" sobre
  // temas del RAG son ruido — distraen del proposal mismo. La feature
  // se decidió con el user 2026-04-25.
  try {
    if (parent && parent.querySelector && parent.querySelector(".proposal")) {
      return;
    }
  } catch (_) {}
  // AbortController registrado en el set global para cancelar si el
  // user navega de sesión antes de que el LLM termine de generar
  // followups (puede tardar 1-3s). Sin esto el fetch resuelve y
  // appendea chips a un turn que ya no es visible.
  const ac = new AbortController();
  inflightSideFetches.add(ac);
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
  } catch (_) {
    // AbortError o cualquier otro error → silent (chips son opcionales).
    return;
  } finally {
    inflightSideFetches.delete(ac);
  }
  try {
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

// a11y: Respeta `prefers-reduced-motion` para usuarios con sensibilidad
// vestibular / motion sickness — el smooth-scroll programático puede
// causar náuseas. El media query lo evalúa en cada call (en vez de
// cachearlo) por si el user lo cambia en Settings sin reload.
// WCAG 2.3.3 (Animation from Interactions, AAA).
function smoothBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto" : "smooth";
}

function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: smoothBehavior() });
}

function obsidianUrl(filePath) {
  return "obsidian://open?file=" + encodeURIComponent(filePath);
}

// Convierte un `whatsapp://<jid>/<msg_id>` o un JID crudo a un universal
// link `https://wa.me/<phone>` que abre la app de WhatsApp en el chat
// correspondiente. iOS + Android lo manejan vía universal links; en
// desktop redirige a web.whatsapp.com.
//
// Returns "" para:
//  - Inputs vacíos / inválidos
//  - Group JIDs (`@g.us`): WhatsApp NO expone deep-link a grupos, así
//    que mostrar el link sería engañoso. El caller renderea el row
//    como texto plano en ese caso.
//
// Notar que WhatsApp NO permite deep-link a un mensaje específico
// (sólo al chat), así que el `msg_id` del URI se descarta — alcanza
// con abrir la conversación.
function waHref(uri) {
  if (!uri || typeof uri !== "string") return "";
  let jid = uri;
  if (jid.indexOf("whatsapp://") === 0) {
    jid = jid.slice("whatsapp://".length);
    const slash = jid.indexOf("/");
    if (slash >= 0) jid = jid.slice(0, slash);
  }
  // Group JIDs (`@g.us`) no tienen deep-link público.
  if (jid.indexOf("@g.us") >= 0) return "";
  // 1:1 chat: extraer dígitos antes del @s.whatsapp.net.
  const phone = jid.split("@")[0];
  if (/^\d{6,}$/.test(phone)) return "https://wa.me/" + phone;
  return "";
}

// Markdown via marked. Dos transformaciones se aplican ANTES de marked:
//   • <<ext>>…<</ext>>  → <span class="ext">⚠ …</span>
//   • [[Wikilinks]]      → [Wikilinks](obsidian://…)
marked.use({
  breaks: true,
  gfm: true,
  renderer: {
    // GFM strikethrough (~text~) destruye paths como `iCloud~md~obsidian`.
    // Reconstruimos los tildes literales en lugar de emitir <del>.
    del({ tokens }) {
      return "~" + this.parser.parseInline(tokens) + "~";
    },
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

// Path-link placeholders: detected paths are replaced with PATH_<n>
// sentinels before marked.parse so neither GFM tokenizers nor markdown
// link parsing (which can't handle spaces in href) interfere. Postprocess
// swaps the sentinels for real <a> tags.
let _pathLinkBuffer = [];

function preprocess(text) {
  _pathLinkBuffer = [];
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
  // Auto-linkify bare paths. Order matters: longest/most-specific patterns
  // first, and paths inside existing markdown links (`](...)`) are skipped
  // by negative lookbehind. Each match is stashed; the visible string is
  // replaced with a non-markdown sentinel so marked won't re-tokenize it.
  const toFileHref = (p) => {
    const abs = p.startsWith("~") ? p.replace(/^~/, "/Users/fer") : p;
    return "file://" + abs.split("/").map(encodeURIComponent).join("/");
  };
  const stash = (label, href) => {
    const idx = _pathLinkBuffer.length;
    _pathLinkBuffer.push({ label, href });
    return `\u0000PATH${idx}\u0000`;
  };
  // 1. Vault root (with optional subpath). Subpath segments use a
  //    controlled alphabet (letters, digits, dash, space, dot) so the
  //    match doesn't bleed into surrounding prose. Notes with spaces
  //    in names (e.g. "Info - Foo.md") still match within a segment.
  out = out.replace(
    /(?<!\]\()(?<![\w/[])~\/Library\/Mobile Documents\/iCloud~md~obsidian\/Documents\/Notes(?:\/[\w\- ]+(?:\.[A-Za-z0-9]+)?)*(?:\.md)?/g,
    (m) => stash(m, toFileHref(m)),
  );
  // 2. PARA-relative paths. Three shapes:
  //    a) Note file:   00-Inbox/foo.md, 03-Resources/Info/Info - Foo.md
  //    b) Directory:   02-Areas/Personal/Guitar/  (trailing slash)
  //    c) Folder leaf: 02-Areas/Personal/Guitar   (no trailing slash)
  //    Files → obsidian://open. Directories → file:// vault root + path.
  const vaultRoot = "/Users/fer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes";
  out = out.replace(
    /(?<!\]\()(?<![\w/[])(0[0-5]-[A-Za-z]+(?:\/[\w\-]+(?: [\w\-]+)*)*(?:\.md|\/)?)/g,
    (m) => {
      if (m.endsWith(".md")) {
        return stash(m, m); // obsidian:// (handled in postprocess)
      }
      const abs = vaultRoot + "/" + m.replace(/\/$/, "");
      return stash(m, "file://" + abs.split("/").map(encodeURIComponent).join("/"));
    },
  );
  // 3. Other home-relative paths (no spaces — safe heuristic).
  out = out.replace(
    /(?<!\]\()(?<![\w/[])~\/[^\s)`'"]+[^\s)`'".,;:]/g,
    (m) => stash(m, toFileHref(m)),
  );
  // 4. Absolute paths under /Users/, /tmp/, /opt/, etc. (no spaces).
  out = out.replace(
    /(?<!\]\()(?<![\w/[])(\/(?:Users|tmp|opt|var|etc|usr|private)\/[^\s)`'"]+[^\s)`'".,;:])/g,
    (m) => stash(m, toFileHref(m)),
  );
  return out;
}

function postprocess(html) {
  let out = html
    .replace(/\u0000EXT_OPEN\u0000/g, "— ")
    .replace(/\u0000EXT_CLOSE\u0000/g, " —");
  out = out.replace(/\u0000PATH(\d+)\u0000/g, (_, idx) => {
    const entry = _pathLinkBuffer[Number(idx)];
    if (!entry) return "";
    const isNote = entry.href.endsWith(".md") && !entry.href.startsWith("file:") && !entry.href.startsWith("http");
    const target = isNote ? obsidianUrl(entry.href) : entry.href;
    const escapedLabel = entry.label.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `<a href="${target}" title="${escapedLabel.replace(/"/g, "&quot;")}">${escapedLabel}</a>`;
  });
  return out;
}

function renderMarkdown(text) {
  return _sanitizeHtml(postprocess(marked.parse(preprocess(text))));
}

// Last turn_id from the most recent RAG response — captured in the
// SSE `done` event below. Used by `/redo` and the ↻ button in the
// feedback bar to regenerate without requiring the client to remember
// the original question (the server resolves it from rag_queries SQL
// by this turn_id; see _resolve_redo_question in web/server.py).
let lastTurnId = null;

// Send --------------------------------------------------------
// `opts.redo_turn_id` + `opts.hint` are optional — when present the
// server resolves the original question from SQL and regenerates with
// an optional soft-steer. `question` is still required by the Pydantic
// validator (non-empty); pass "(redo)" or similar as placeholder when
// redo_turn_id is set.
async function send(question, opts = {}) {
  if (pending) return;
  // a11y / race-condition: deshabilitar el botón en el DOM ANTES de
  // tocar `pending` evita el doble-tap rápido (touch users en mobile,
  // mouse double-click en desktop). El guard JS `if (pending) return`
  // previene la segunda invocación lógica, pero un click adicional
  // dispara igual el handler — `disabled` lo bloquea a nivel browser.
  // En el finally se restablece visibilidad (sendBtn.hidden=false) +
  // updateSendBtnState() recalcula `disabled` según contenido del
  // textarea. 2026-04-25 a11y audit lote 2.
  if (sendBtn) sendBtn.disabled = true;
  pending = true;
  lastUserQuestion = question;
  pushHistory(question);
  input.disabled = true;
  stopBtn.hidden = false;
  // Mobile Tier 1: mientras el LLM stream está activo, el send-btn se
  // oculta (el stop lo reemplaza en el mismo slot). El user tapea stop
  // para abortar — no necesita recordar Esc ni keyboard shortcut.
  if (sendBtn) sendBtn.hidden = true;
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
  // Metachat flag: el backend marca `metachat: true` en sources + done
  // cuando el turno fue servido por el short-circuit canned reply ("Hola",
  // "gracias", "qué podés hacer"). Sin sources + confidence=null la lógica
  // de "weakAnswer" abajo dispararía `↗ buscar en internet` + `appendRelated`
  // (YouTube) al lado del saludo — una UX absurda. Este flag bloquea esos
  // fallbacks para canned replies.
  let isMetachat = false;
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
  function startStageTicker(phase, staticLabel) {
    stopStageTicker();
    stageStart = performance.now();
    // Second-arg `staticLabel` (2026-04-22): when the server ships an
    // intent-aware hint along with the `retrieving` status (see
    // `_build_retrieve_hint` in web/server.py), use that literal as the
    // label and freeze the dynamic retrieveLabel semáforo copy. Keeps
    // the "search feels intentional" UX pointed at WHAT is being done
    // instead of escalating through generic "buscando → revisando notas
    // → búsqueda profunda" that says nothing about the actual query.
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
      // When a staticLabel is provided (e.g. intent-aware hint on the
      // retrieving phase), it wins over the dynamic retrieveLabel — the
      // hint carries more information than the generic semáforo copy.
      const lbl = staticLabel || labelFn(elapsed);
      stageLabelEl.textContent = showSecs ? `${lbl} · ` : lbl;
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
    const reqBody = {
      question,
      session_id: sessionId,
      vault_scope: vaultScope || null,
      mode: getChatMode(),
    };
    if (opts.redo_turn_id) reqBody.redo_turn_id = opts.redo_turn_id;
    if (opts.hint) reqBody.hint = opts.hint;
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
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
      // a11y: role="alert" sobre el error de red/transport — el user
      // está esperando una respuesta y necesita enterarse que falló.
      const errNode = el("div", "error", `  error: ${err.message}`);
      errNode.setAttribute("role", "alert");
      turn.appendChild(errNode);
    }
  } finally {
    stopGeneratingTicker();
    pending = false;
    input.disabled = false;
    stopBtn.hidden = true;
    // Mobile Tier 1: restaurar visibilidad del send-btn + recalcular
    // disabled (el input quedó vacío post-clear, entonces va a quedar
    // disabled hasta que el user tipee de nuevo — UX correcto).
    if (sendBtn) sendBtn.hidden = false;
    currentController = null;
    input.value = "";
    autoGrow();
    updateSendBtnState();
    // En mobile NO refocuseamos el input — eso reabre el keyboard y
    // tapa la respuesta recién streameada. En desktop sí mantenemos el
    // focus para que el user siga preguntando sin clickear.
    // Heurística: matchMedia (max-width: 640px) = mobile breakpoint
    // que empareja con el CSS.
    const isMobileViewport = window.matchMedia("(max-width: 640px)").matches;
    if (!isMobileViewport) {
      input.focus();
    }
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
      // Metachat flag: cuando el short-circuit sirvió un canned reply
      // ("Hola", "gracias"), el backend marca `metachat: true` acá. La
      // bandera apaga todos los CTAs de fallback (buscar en internet,
      // YouTube related, fallback cluster) para que el saludo no termine
      // con un link a Google "Hola" al costado.
      if (parsed.metachat) isMetachat = true;
    } else if (event === "proposal") {
      // The server emits this when the tool returned `needs_clarification`
      // (ambiguous datetime) OR when the auto-create failed with an
      // error. We render the card so the user can confirm/edit/retry.
      // The thinking spinner (if still showing) is torn down on the next
      // token event.
      hadProposal = true;
      appendProposal(turn, parsed);
      scrollBottom();
    } else if (event === "created") {
      // Auto-create happy path: a propose_* tool created a reminder/event.
      // Inline chip below the LLM response (same aesthetic as ╌ fuentes)
      // instead of a floating toast that covers the top bar. Chip stays in
      // the conversation scroll naturally; Reminders get a `deshacer`
      // inline link-button (Calendar events don't — AppleScript delete is
      // unworkable on macOS 14+ with >1k events, as documented earlier).
      hadProposal = true;
      appendCreatedChip(turn, parsed);
      scrollBottom();
    } else if (event === "token") {
      if (!ragLine) {
        stopGeneratingTicker();
        thinking.remove();
        ragLine = document.createElement("div");
        ragLine.className = "line";
        const prompt = el("span", "prompt rag", "rag ›");
        ragText = el("span", "text rag pending");
        // a11y: aria-live="polite" sobre el contenedor que muta con
        // cada chunk SSE → screen readers anuncian la respuesta a
        // medida que llega. aria-atomic="false" porque cada token es
        // un append (no queremos que VoiceOver re-lea todo el
        // párrafo en cada chunk). role="region" + aria-label dan un
        // landmark navegable. 2026-04-25 a11y audit lote 2.
        ragText.setAttribute("role", "region");
        ragText.setAttribute("aria-live", "polite");
        ragText.setAttribute("aria-atomic", "false");
        ragText.setAttribute("aria-label", "Respuesta del asistente");
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
      // Belt-and-suspenders: el backend marca `metachat: true` también
      // en el done event. Si el sources event llegó sin la flag por
      // algún motivo (SSE parcial, versión desincronizada), todavía
      // podemos detectar acá y apagar los fallbacks.
      if (parsed.metachat) isMetachat = true;
      const conf = Number.isFinite(confidence) ? confidence : parsed.top_score;
      // Two thresholds — conflating them polluted the UI with YouTube videos
      // on queries where the vault answered correctly.
      //
      // `weakAnswer` (inline "↗ buscar en internet" link, cheap): conf < 1.0
      // es laxo a propósito — un link hint sutil sirve incluso cuando el
      // vault respondió OK pero el usuario podría querer cross-reference.
      //
      // `vaultReallyFailed` (YouTube API call, costoso + ruidoso): solo
      // cuando NO hay sources O conf < 0.10 (mismo threshold del backend
      // CONFIDENCE_DEEP_THRESHOLD — "abajo de esto el vault no respondió").
      // Pre-fix (2026-04-21): "en que ciclo estamos?" con conf=0.3 y
      // `dev cycles.md` como top source disparaba YouTube con videos
      // genéricos de PHP — el signal más ruidoso del sistema.
      const weakAnswer = !sources || !sources.length || (Number.isFinite(conf) && conf < 1.0);
      const vaultReallyFailed = !sources || !sources.length || (Number.isFinite(conf) && conf < 0.10);
      // A mention hit (sentinel score 5.0 from the Mentions folder) means
      // the question is about an entity in the user's life, not a generic
      // topic. Skip external enrichment / web search — those would surface
      // tourism/wikipedia noise instead of the personal context.
      // On propose-intent turns skip the sources panel AND the web-search
      // fallback — vault retrieval / googling are irrelevant when the user
      // asked the system to CREATE a reminder/event.
      const mentionMatched = (sources || []).some((s) => s.score >= 5.0);
      // Low-confidence bypass (backend skipped the LLM call entirely):
      // mostrar el cluster prominente de "¿querés que busque en...?" con
      // 3 botones (Google/YouTube/Wikipedia) en vez del link sutil
      // inline. Razón: si el sistema directamente saltó al template
      // fijo, el usuario necesita un escape visible, no un hint tímido.
      // Cuando hay weakAnswer pero NO bypass (el LLM intentó y no
      // encontró), mantenemos el link inline — el user puede insistir
      // sin que el layout explote en CTAs.
      const lowConfBypass = parsed.low_conf_bypass === true;
      // `isSourceSpecific` (2026-04-24, Fer F. user report iter 4): el
      // backend marca `source_specific: true` en el done event cuando
      // el pre-router disparó un tool de fuente concreta del user
      // (gmail_recent / calendar_ahead / reminders_due). Google no tiene
      // acceso a mis mails pendientes ni a mis recordatorios de Apple —
      // el CTA "↗ buscar en internet" y el cluster YouTube son ruido en
      // esas queries. El gate apaga ambos.
      const isSourceSpecific = parsed.source_specific === true;
      // `isMetachat` bloquea los 3 fallbacks abajo. Un canned reply de
      // metachat ("Hola") viene sin sources y con confidence=null — sin
      // el gate, la lógica de `weakAnswer` dispara `↗ buscar en internet`
      // (link a Google) + `appendRelated` (YouTube). Absurdo para un
      // saludo; el backend ya dio una respuesta conversacional canned.
      if (!hadProposal && !isMetachat && !isSourceSpecific && question && weakAnswer && !mentionMatched) {
        if (lowConfBypass) {
          appendFallbackCluster(turn, question);
        } else if (ragText) {
          const target = ragText.lastElementChild || ragText;
          appendWebSearch(target, question, true);
        }
      }
      if (!hadProposal && !isMetachat && sources && sources.length) {
        appendSources(turn, sources, conf);
      }
      // appendRelated() renderea YouTube videos específicos. Usamos el
      // threshold ESTRICTO (vaultReallyFailed): solo si NO hay sources o
      // conf < 0.10. Queries con respuesta correcta + confidence 0.1-1.0
      // no deben disparar YouTube — el vault ya respondió bien y videos
      // genéricos son ruido. Ver comentario arriba de los thresholds.
      // `isSourceSpecific` también bloquea YouTube — preguntar por
      // mails/calendar/reminders nunca debería terminar en una playlist.
      if (!hadProposal && !isMetachat && !isSourceSpecific && question && vaultReallyFailed && !mentionMatched) {
        appendRelated(turn, question);
      }
      const feedbackBar = parsed.turn_id
        ? appendFeedback(turn, {
            turn_id: parsed.turn_id,
            q: question,
            paths: (sources || []).map((s) => s.file).filter(Boolean),
            // Pass the full source objects so the 👎 corrective flow can
            // render selectable cards with title + score, not just bare paths.
            sources: sources || [],
            session_id: sessionId,
          })
        : null;
      // Sidebar list se actualiza tras cada turn confirmado — la sesión
      // nueva aparece en "recientes" apenas el primer turn cierra, o
      // sube al tope si es un turn de continuación. Silent-fail para
      // tests / páginas legacy sin sidebar. 2026-04-24.
      try { if (typeof refreshSessions === "function") refreshSessions(); } catch {}
      // Capture the turn_id so `/redo` and ↻ can regenerate without the
      // client needing to remember the original question (server resolves
      // it from rag_queries SQL). Also used by the redo button below.
      if (parsed.turn_id) lastTurnId = parsed.turn_id;
      // Pin enough metadata on the turn DOM element so the global `copy`
      // listener below can attribute a selection to a (turn, query, top
      // source) tuple without a side-channel. Truncated query keeps the
      // attribute value short (DOM attrs pay a memory price per node;
      // CSS selectors on them stay cheap for <500 chars).
      if (parsed.turn_id) {
        turn.dataset.turnId = parsed.turn_id;
        if (question) {
          turn.dataset.q = question.length > 300
            ? question.slice(0, 300) : question;
        }
        const topPath = (sources && sources[0] && sources[0].file) || "";
        if (topPath && topPath.indexOf("://") === -1) {
          // Skip cross-source ids (calendar://, whatsapp://) — those
          // are not vault-relative paths and /api/behavior will 400
          // on them (VAULT_PATH.resolve() relative_to check).
          turn.dataset.topPath = topPath;
        }
        if (sessionId) turn.dataset.session = sessionId;
      }
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
    } else if (event === "enrich") {
      if (Array.isArray(parsed.lines) && parsed.lines.length) {
        appendEnrich(turn, parsed.lines);
      }
    } else if (event === "grounding") {
      if (parsed.total > 0) {
        renderGrounding(parsed, turn);
      }
    } else if (event === "empty") {
      stopGeneratingTicker();
      thinking.remove();
      clearToolChips();
      // a11y: role="alert" (= aria-live="assertive" + aria-atomic="true")
      // — los estados terminales sin respuesta sí necesitan
      // interrumpir al screen reader (a diferencia del stream "polite"
      // arriba) porque el usuario está esperando la respuesta y este
      // mensaje le dice "no hubo nada / falló".
      const emptyNode = el("div", "empty", `  ${parsed.message || "Sin resultados relevantes."}`);
      emptyNode.setAttribute("role", "alert");
      turn.appendChild(emptyNode);
      if (question) {
        appendWebSearch(turn, question);
        appendRelated(turn, question);
      }
    } else if (event === "error") {
      stopGeneratingTicker();
      thinking.remove();
      clearToolChips();
      // a11y: role="alert" — ver comentario del bloque "empty" arriba.
      const errNode = el("div", "error", `  ${parsed.message || "Error"}`);
      errNode.setAttribute("role", "alert");
      turn.appendChild(errNode);
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
          //
          // `parsed.hint` (2026-04-22): server ships an intent-aware
          // label like "Contando notas…" / "Buscando por persona…"
          // so the ticker shows WHAT is being searched, not just
          // the generic semáforo copy. Falls back to the legacy
          // retrieveLabel state machine when no hint is provided
          // (e.g. semantic intent → no incremental info).
          startStageTicker("retrieving", parsed.hint || null);
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
    // Abort in-flight antes de limpiar — sin esto el stream viejo
    // seguiría apendeando al DOM nuevo. 2026-04-24.
    if (currentController) {
      try { currentController.abort(); } catch (_) {}
      currentController = null;
    }
    // También cancelar related/followups/contacts pending del turn
    // anterior — sino aparecen chips de followup de la sesión vieja
    // pegoteados encima del "nueva sesión — historial en blanco".
    abortSideFetches();
    sessionId = null;
    sessionStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(SESSION_KEY); // legacy cleanup
    messagesEl.innerHTML = "";
    input.value = "";
    autoGrow();
    pushSystemMessage("meta", "nueva sesión — historial en blanco");
    // Refresca la sidebar para que la sesión viejita aparezca en
    // recientes (o desaparezca si quedó vacía). Silent-fail: la
    // sidebar puede no estar montada (tests, legacy pages).
    try { if (typeof refreshSessions === "function") refreshSessions(); } catch {}
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
  if (cmd === "/redo") {
    // Regenerate the last turn, optionally with a soft-steer hint.
    // Falls back to a user-visible hint when there's no lastTurnId
    // yet (fresh session with zero turns).
    input.value = "";
    autoGrow();
    if (!lastTurnId) {
      pushSystemMessage("err", "no hay respuesta previa para regenerar");
      return true;
    }
    const hint = (arg || "").trim();
    // Placeholder question — server ignores it when redo_turn_id is set
    // (resolves the real q from rag_queries SQL), but the Pydantic
    // ChatRequest validator requires non-empty question.
    await send(hint ? `(redo: ${hint})` : "(redo)",
               { redo_turn_id: lastTurnId, hint: hint || null });
    return true;
  }
  // ── Shortcuts para acciones de creación ────────────────────────────
  // Estos 4 reescriben el slash a una frase natural que matchea el
  // routing del `_WEB_TOOL_ADDENDUM` y va al pipeline LLM normal —
  // termina llamando a propose_whatsapp_send / propose_mail_send /
  // propose_reminder / propose_calendar_event y mostrando la tarjeta
  // de confirmación. La razón de no esquivar al LLM es flexibilidad:
  // "mandale a Grecia: que llego en 10" puede tener fechas relativas,
  // citas, o variantes de saludo que el LLM puede normalizar.
  if (cmd === "/wzp") {
    if (!arg) {
      pushSystemMessage("err", "uso: /wzp <contacto>: <mensaje>");
      input.value = "/wzp ";
      autoGrow();
      input.focus();
      return true;
    }
    // Si el user escribió "Grecia: hola" lo dejamos parseable; si solo
    // tipeó "Grecia hola" igual el LLM lo mete a propose_whatsapp_send.
    // Usamos "mandale" porque está en el routing y es rioplatense
    // (más natural que "envíale").
    const rewritten = `mandale un mensaje por whatsapp a ${arg}`;
    input.value = "";
    autoGrow();
    await send(rewritten);
    return true;
  }
  if (cmd === "/mail") {
    if (!arg) {
      pushSystemMessage("err", "uso: /mail <email>: <asunto> — <cuerpo>");
      input.value = "/mail ";
      autoGrow();
      input.focus();
      return true;
    }
    const rewritten = `mandale un mail a ${arg}`;
    input.value = "";
    autoGrow();
    await send(rewritten);
    return true;
  }
  if (cmd === "/rem") {
    if (!arg) {
      pushSystemMessage("err", "uso: /rem <texto> [cuándo]");
      input.value = "/rem ";
      autoGrow();
      input.focus();
      return true;
    }
    const rewritten = `recordame ${arg}`;
    input.value = "";
    autoGrow();
    await send(rewritten);
    return true;
  }
  if (cmd === "/evt") {
    if (!arg) {
      pushSystemMessage("err", "uso: /evt <título> [cuándo]");
      input.value = "/evt ";
      autoGrow();
      input.focus();
      return true;
    }
    const rewritten = `agendá un evento: ${arg}`;
    input.value = "";
    autoGrow();
    await send(rewritten);
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

// Click on the "rag" brand → clear the visible conversation.
// Mirrors `/cls`: wipes the DOM + aborts in-flight, but keeps the session
// id and the up-arrow query history intact. Server-side turns and arrow
// history survive so the user can keep the thread going after clicking.
//
// Post-sidebar (2026-04-24): hay DOS instancias de `.topbar-title` — el
// brand del #mobile-header (sólo visible en mobile) y el de la sidebar
// (visible en desktop). Bindeamos a ambos con querySelectorAll para
// que funcione en los dos breakpoints.
const clearView = () => {
  if (currentController) currentController.abort();
  messagesEl.innerHTML = "";
  input.value = "";
  autoGrow();
  input.focus();
};
document.querySelectorAll(".topbar-title").forEach((brand) => {
  brand.style.cursor = "pointer";
  brand.setAttribute("role", "button");
  brand.setAttribute("tabindex", "0");
  brand.setAttribute("title", "Click para limpiar la vista");
  brand.addEventListener("click", clearView);
  brand.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      clearView();
    }
  });
});

// Auto-submit when arriving from a deep-link like /chat?q=foo
(() => {
  const params = new URLSearchParams(window.location.search);
  const seed = params.get("q");
  if (!seed) return;
  history.replaceState({}, "", window.location.pathname);
  input.value = seed;
  setTimeout(() => send(seed), 50);
})();

// ═══════════════════════════════════════════════════════════════════════
// Mobile Tier 1 wiring (2026-04-23)
// ═══════════════════════════════════════════════════════════════════════
// 3 piezas:
//   a) En mobile blur el input al boot — el autofocus HTML hace que iOS
//      abra el keyboard al entrar, tapando medio viewport. Esperamos a
//      que el user tapee el input (o el send-btn) para prender el KB.
//   b) Sheet-menu handlers: abrir con ⋯, cerrar con backdrop / X / Esc.
//      Al abrir, sincronizamos options y value desde los selects
//      originales del topbar — single-source-of-truth sigue siendo
//      vault-picker / model-picker / tts-toggle en el DOM.
//   c) Change handlers del sheet: propagar value al original + disparar
//      un `change` event para que los listeners existentes corran sin
//      tocar (persistencia localStorage, fetch a /api/..., etc.).
// ═══════════════════════════════════════════════════════════════════════

(function initMobileTier1() {
  const mqMobile = window.matchMedia("(max-width: 640px)");
  // pointer:coarse cubre tablets / iPad además de phones (matchMedia
  // 640px es solo viewport-width). Evitar autofocus en cualquier
  // touch device — todos abren keyboard on focus.
  const isTouch = window.matchMedia("(pointer: coarse)").matches;

  // ── (a) Focus inicial: solo desktop, nunca touch ──────────────────
  // El HTML autofocus se quitó (a11y audit lote 2, 2026-04-25): en iOS
  // Safari abría el teclado al cargar la página y tapaba el empty-hero.
  // Ahora el focus inicial lo decide JS y solo corre en desktop. Para
  // deep-links (/chat?q=...) el submit es inmediato, no tocamos focus
  // — el flujo de send() ya gestiona el blur.
  const seed = new URLSearchParams(window.location.search).get("q");
  if (!seed && !isTouch && !mqMobile.matches) {
    // requestAnimationFrame espera al primer paint para que el focus
    // corra después de que el layout esté estable (el composer puede
    // estar centrado en empty-state, focusear antes del layout deja
    // el caret en posición rara).
    requestAnimationFrame(() => {
      try { input.focus({ preventScroll: true }); } catch (_) {}
    });
  }

  // ── (b) Sheet open/close ──────────────────────────────────────────
  if (menuBtn && menuSheet) {
    menuBtn.addEventListener("click", () => openSheet());
    // Close: backdrop + botón X (ambos llevan data-close-sheet).
    menuSheet.addEventListener("click", (e) => {
      const t = e.target;
      if (t && (t.hasAttribute("data-close-sheet") ||
                t.closest("[data-close-sheet]"))) {
        closeSheet();
      }
    });
    // Esc key cierra. Se integra con el handler global de Esc más
    // arriba sin conflictos porque ese chequea `helpModal.hidden` y
    // `currentController` — agregar una pre-verificación del sheet.
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !menuSheet.hidden) {
        closeSheet();
      }
    });
  }

  // ── (c) Sync bidireccional sheet ↔ originales ─────────────────────
  if (sheetVaultPicker && vaultPicker) {
    sheetVaultPicker.addEventListener("change", () => {
      vaultPicker.value = sheetVaultPicker.value;
      vaultPicker.dispatchEvent(new Event("change"));
    });
  }
  if (sheetTtsToggle && ttsToggle) {
    // TTS toggle es un button (no select). Tap en el sheet = click en
    // el original → el handler original toggle state + persiste.
    sheetTtsToggle.addEventListener("click", () => {
      ttsToggle.click();
      // Re-sync el visual state del sheet (aria-pressed, label text)
      // inmediatamente después del click.
      syncSheetFromOriginals();
    });
  }

  // Initial state del send button (antes del primer input event).
  updateSendBtnState();
})();

function openSheet() {
  if (!menuSheet) return;
  // Antes de mostrar, copiar estado actual de los originales para que
  // el sheet arranque con values correctos aunque el user haya cambiado
  // vault/modelo en otra tab desde la última apertura.
  syncSheetFromOriginals();
  menuSheet.hidden = false;
  if (menuBtn) menuBtn.setAttribute("aria-expanded", "true");
  // Focus el primer control interactivo — importante para VoiceOver y
  // para users de keyboard físico (bluetooth en iPhone).
  const firstFocusable = menuSheet.querySelector("select, button, a");
  if (firstFocusable) firstFocusable.focus();
}

function closeSheet() {
  if (!menuSheet) return;
  menuSheet.hidden = true;
  if (menuBtn) {
    menuBtn.setAttribute("aria-expanded", "false");
    // Devolver focus al botón que abrió el sheet (a11y best practice).
    menuBtn.focus();
  }
}

/**
 * Copia options + value + state desde los pickers/toggle originales
 * del topbar al sheet. Se corre al abrir y después de interactions
 * que puedan cambiar el estado (como el TTS toggle click).
 */
function syncSheetFromOriginals() {
  // Vault picker: copiar el innerHTML completo (options + selected).
  if (sheetVaultPicker && vaultPicker) {
    sheetVaultPicker.innerHTML = vaultPicker.innerHTML;
    sheetVaultPicker.value = vaultPicker.value;
  }
  // TTS toggle: reflejar aria-pressed + actualizar el label textual.
  if (sheetTtsToggle && ttsToggle) {
    const pressed = ttsToggle.getAttribute("aria-pressed") === "true";
    sheetTtsToggle.setAttribute("aria-pressed", pressed ? "true" : "false");
    const stateLabel = sheetTtsToggle.querySelector(".sheet-toggle-state");
    if (stateLabel) stateLabel.textContent = pressed ? "on" : "off";
  }
}


// ═══════════════════════════════════════════════════════════════════════
// Sidebar (2026-04-24) — claude.ai-style
// ═══════════════════════════════════════════════════════════════════════
// Responsabilidades:
//   1) Collapse toggle desktop → icon-only + persist en localStorage.
//   2) Mobile drawer: hamburger abre, X/backdrop/Esc cierran.
//   3) Fetch + render de /api/sessions con click-to-hydrate.
//   4) New-chat button wiring (misma lógica que `/new` + UI niceties).
//   5) Search filter client-side sobre la lista de sesiones.
//
// Single source of truth sigue siendo #vault-picker / #model-picker /
// #tts-toggle — la sidebar los contiene directamente ahora, así que
// NADA cambia en la lógica de persistencia/selección. La sidebar sólo
// agrega sessions + collapse + mobile drawer.
// ═══════════════════════════════════════════════════════════════════════

const SIDEBAR_COLLAPSED_KEY = "obsidian-rag:sidebar-collapsed";

const sidebar = document.getElementById("sidebar");
const sidebarOpenBtn = document.getElementById("sidebar-open-btn");
const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
const sidebarCollapseBtn = document.getElementById("sidebar-collapse-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const mobileNewBtn = document.getElementById("mobile-new-btn");
const sessionsList = document.getElementById("sessions-list");
const sessionsSearch = document.getElementById("sessions-search");
const sessionsRefreshBtn = document.getElementById("sessions-refresh-btn");

// Cache de la última respuesta de /api/sessions — el search filter
// trabaja sobre esta memoria en vez de re-pegarle al server a cada
// tecleada. Se refresca en los hooks de refreshSessions().
let sessionsCache = [];

// ── Collapse state (desktop) ─────────────────────────────────────────
function applySidebarCollapsed(collapsed) {
  if (!sidebar) return;
  sidebar.setAttribute("data-state", collapsed ? "collapsed" : "expanded");
  if (sidebarCollapseBtn) {
    sidebarCollapseBtn.setAttribute("aria-pressed", collapsed ? "true" : "false");
    sidebarCollapseBtn.setAttribute(
      "aria-label",
      collapsed ? "Expandir sidebar" : "Colapsar sidebar"
    );
  }
}

function initSidebarCollapse() {
  if (!sidebar || !sidebarCollapseBtn) return;
  // En mobile el collapsed se ignora (CSS media query sobreescribe al
  // estado `open`); igual leemos el flag persistido para respetarlo en
  // desktop. Default: expanded.
  const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  applySidebarCollapsed(saved);
  sidebarCollapseBtn.addEventListener("click", () => {
    const isCollapsed = sidebar.getAttribute("data-state") === "collapsed";
    const next = !isCollapsed;
    applySidebarCollapsed(next);
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0"); } catch {}
  });
}

// ── Mobile drawer open/close ─────────────────────────────────────────
// Focus trap WCAG 2.4.3: cuando el drawer está abierto en mobile, Tab
// cicla dentro del sidebar (sino el focus salta al composer detrás
// del backdrop). Mismo patrón que el help modal arriba — reusa
// FOCUSABLE_SELECTOR. El trap solo aplica cuando data-state=="open"
// (mobile drawer); en desktop el sidebar es siempre visible y el
// focus puede salir libremente.
let _sidebarLastFocus = null;
function _sidebarTrap(e) {
  if (e.key !== "Tab") return;
  if (!sidebar || sidebar.getAttribute("data-state") !== "open") return;
  const f = [...sidebar.querySelectorAll(FOCUSABLE_SELECTOR)]
    .filter((el) => !el.hasAttribute("hidden") && el.offsetParent !== null);
  if (!f.length) return;
  const first = f[0];
  const last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

function openSidebarMobile() {
  if (!sidebar) return;
  _sidebarLastFocus = document.activeElement;
  sidebar.setAttribute("data-state", "open");
  if (sidebarOpenBtn) sidebarOpenBtn.setAttribute("aria-expanded", "true");
  // Focus el primer control interactivo — a11y + keyboard users.
  const firstFocusable = sidebar.querySelector(FOCUSABLE_SELECTOR);
  if (firstFocusable) firstFocusable.focus({ preventScroll: true });
  sidebar.addEventListener("keydown", _sidebarTrap);
}

function closeSidebarMobile() {
  if (!sidebar) return;
  sidebar.removeEventListener("keydown", _sidebarTrap);
  // Restaurar el estado desktop persistido (expanded / collapsed)
  // cuando cerramos el drawer mobile. Así el desktop no se rompe si
  // el user redimensiona la ventana mientras tenía el drawer abierto.
  const savedCollapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  sidebar.setAttribute("data-state", savedCollapsed ? "collapsed" : "expanded");
  if (sidebarOpenBtn) {
    sidebarOpenBtn.setAttribute("aria-expanded", "false");
  }
  // Restaurar focus al trigger que abrió el drawer si todavía existe
  // en el DOM (típicamente el sidebar-open-btn). Si el user navegó al
  // input mientras el drawer estaba abierto, no querríamos saltar
  // sobre su intención — pero en este flow el trap previene eso, así
  // que el lastFocus es siempre el botón de apertura.
  if (_sidebarLastFocus && typeof _sidebarLastFocus.focus === "function" &&
      document.contains(_sidebarLastFocus)) {
    try { _sidebarLastFocus.focus({ preventScroll: true }); } catch (_) {}
  } else if (sidebarOpenBtn) {
    sidebarOpenBtn.focus({ preventScroll: true });
  }
  _sidebarLastFocus = null;
}

function initSidebarMobile() {
  if (!sidebar) return;
  if (sidebarOpenBtn) {
    sidebarOpenBtn.addEventListener("click", openSidebarMobile);
  }
  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", closeSidebarMobile);
  }
  // Backdrop sibling: listener directo — el click en el backdrop
  // (sólo visible en mobile via CSS) cierra el drawer.
  const backdrop = document.getElementById("sidebar-backdrop");
  if (backdrop) backdrop.addEventListener("click", closeSidebarMobile);
  // Cualquier otro [data-sidebar-close] (el X dentro del sidebar, etc.)
  // también cierra. Delegado a nivel document para no duplicar.
  document.addEventListener("click", (e) => {
    const t = e.target;
    if (!t) return;
    if (t.hasAttribute && t.hasAttribute("data-sidebar-close")) {
      closeSidebarMobile();
      return;
    }
    const ancestor = t.closest && t.closest("[data-sidebar-close]");
    if (ancestor) closeSidebarMobile();
  });
  // Esc cierra si el drawer está abierto (sólo relevante en mobile
  // — en desktop el drawer no tiene estado "open"). Se integra con
  // el handler Esc global porque sólo reaccionamos cuando data-state
  // es exactamente "open".
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (sidebar.getAttribute("data-state") === "open") {
      closeSidebarMobile();
    }
  });
}

// ── New-chat buttons ─────────────────────────────────────────────────
function triggerNewChat() {
  // Reusa la lógica del slash command /new (abort + clear + refresh).
  // No podemos llamar handleSlashCommand("/new") directamente sin meter
  // "/new" como texto visible en el input — duplicamos el bloque acá.
  if (currentController) {
    try { currentController.abort(); } catch (_) {}
    currentController = null;
  }
  // También cancelar fetches secundarios (related/followups) del turn
  // que estaba en curso. Mismo motivo que en /new arriba.
  abortSideFetches();
  sessionId = null;
  sessionStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(SESSION_KEY);
  messagesEl.innerHTML = "";
  input.value = "";
  autoGrow();
  pushSystemMessage("meta", "nueva sesión — historial en blanco");
  refreshSessions();
  // En mobile cerramos el drawer tras el click — el user quiere escribir.
  closeSidebarMobile();
  const isMobileViewport = window.matchMedia("(max-width: 767px)").matches;
  if (!isMobileViewport) input.focus();
}

function initNewChatButtons() {
  if (newChatBtn) newChatBtn.addEventListener("click", triggerNewChat);
  if (mobileNewBtn) mobileNewBtn.addEventListener("click", triggerNewChat);
}

// ── Sessions list: fetch + render + click to hydrate ─────────────────
async function refreshSessions() {
  if (!sessionsList) return;
  try {
    const res = await fetch("/api/sessions?limit=40", {
      headers: { "Accept": "application/json" },
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    sessionsCache = Array.isArray(data.sessions) ? data.sessions : [];
    renderSessions(getFilterText());
  } catch (err) {
    sessionsList.innerHTML = "";
    const li = document.createElement("li");
    li.className = "sessions-empty sessions-error";
    li.textContent = "no se pudo cargar el historial";
    sessionsList.appendChild(li);
  }
}

function getFilterText() {
  if (!sessionsSearch) return "";
  return (sessionsSearch.value || "").trim().toLowerCase();
}

function renderSessions(filter) {
  if (!sessionsList) return;
  sessionsList.innerHTML = "";
  const q = (filter || "").trim().toLowerCase();
  const items = q
    ? sessionsCache.filter((s) => (s.title || "").toLowerCase().includes(q))
    : sessionsCache;
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "sessions-empty";
    li.textContent = q
      ? "sin coincidencias"
      : "sin conversaciones aún — escribí algo en el chat";
    sessionsList.appendChild(li);
    return;
  }
  for (const s of items) {
    const li = document.createElement("li");
    li.className = "session-item";
    li.setAttribute("role", "button");
    li.setAttribute("tabindex", "0");
    li.setAttribute("data-session-id", s.id);
    li.setAttribute("title", s.title || "sin título");
    if (s.id === sessionId) {
      li.setAttribute("aria-current", "true");
    }
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title || "sin título";
    li.appendChild(title);
    const meta = document.createElement("span");
    meta.className = "session-meta";
    meta.textContent = formatSessionMeta(s);
    li.appendChild(meta);
    li.addEventListener("click", () => loadSession(s.id));
    li.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        loadSession(s.id);
      }
    });
    sessionsList.appendChild(li);
  }
}

function formatSessionMeta(s) {
  // Compacto: "N turns · YYYY-MM-DD HH:MM"
  const bits = [];
  if (Number.isFinite(s.turns)) bits.push(`${s.turns} turn${s.turns === 1 ? "" : "s"}`);
  if (s.updated_at) {
    const t = String(s.updated_at).slice(0, 16).replace("T", " ");
    bits.push(t);
  }
  return bits.join(" · ");
}

async function loadSession(sid) {
  if (!sid) return;
  if (sid === sessionId && messagesEl.childElementCount > 0) {
    // Ya estamos en esta sesión y tiene contenido — solo cerramos el
    // drawer mobile y focuseamos el input.
    closeSidebarMobile();
    return;
  }
  // Abort in-flight antes de cambiar de sesión.
  if (currentController) {
    try { currentController.abort(); } catch (_) {}
    currentController = null;
  }
  // Cancelar related/followups/contacts pending del turn de la
  // sesión vieja — sino aparecen chips o YouTube relacionados al
  // turn previo metidos en la lista hidratada de la sesión nueva.
  abortSideFetches();
  try {
    const res = await fetch(`/api/session/${encodeURIComponent(sid)}/turns`, {
      headers: { "Accept": "application/json" },
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    hydrateTurns(data);
    sessionId = data.id || sid;
    sessionStorage.setItem(SESSION_KEY, sessionId);
    // Marcar la sesión como current en la UI.
    sessionsList.querySelectorAll(".session-item").forEach((el) => {
      if (el.getAttribute("data-session-id") === sessionId) {
        el.setAttribute("aria-current", "true");
      } else {
        el.removeAttribute("aria-current");
      }
    });
    closeSidebarMobile();
    const isMobileViewport = window.matchMedia("(max-width: 767px)").matches;
    if (!isMobileViewport) input.focus();
  } catch (err) {
    pushSystemMessage("err", `no se pudo cargar la sesión: ${err.message}`);
  }
}

function hydrateTurns(data) {
  messagesEl.innerHTML = "";
  const turns = Array.isArray(data && data.turns) ? data.turns : [];
  if (!turns.length) {
    pushSystemMessage("meta", "sesión vacía");
    return;
  }
  for (const t of turns) {
    const turn = appendTurn();
    if (t.q) appendLine(turn, "user", t.q);
    if (t.a) {
      // Use the same rendering path as live streams: append "rag" line,
      // then parse as Markdown so code blocks / lists look right. We
      // render as a one-shot (not streaming) since the turn is historical.
      const line = document.createElement("div");
      line.className = "line";
      const prompt = document.createElement("span");
      prompt.className = "prompt rag";
      prompt.textContent = "rag ›";
      const text = document.createElement("span");
      text.className = "text rag md-output";
      try {
        text.innerHTML = renderMarkdown(t.a);
      } catch {
        text.textContent = t.a;
      }
      line.appendChild(prompt);
      line.appendChild(text);
      turn.appendChild(line);
    }
    // Copy button por turn — sin esto el historial era read-only. Usamos
    // buildMarkdownExport con los paths guardados en la sesión como
    // pseudo-sources (no tenemos score, pero sí el file name para el
    // wikilink [[Nota]]). 2026-04-24.
    if (t.a && t.a.trim()) {
      const actions = el("div", "msg-actions");
      const pseudoSources = (t.paths || []).map((p) => ({ file: p }));
      appendCopyButton(
        actions,
        () => buildMarkdownExport(t.q || "", t.a || "", pseudoSources),
      );
      turn.appendChild(actions);
    }
  }
  scrollBottom();
  // Meta inline: cuántas turns se cargaron.
  pushSystemMessage(
    "meta",
    `sesión cargada · ${turns.length} turn${turns.length === 1 ? "" : "s"}`
  );
}

// ── Search filter (client-side over sessionsCache) ───────────────────
function initSessionsSearch() {
  if (!sessionsSearch) return;
  let debounceTimer = null;
  sessionsSearch.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => renderSessions(getFilterText()), 80);
  });
  sessionsSearch.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      sessionsSearch.value = "";
      renderSessions("");
    }
  });
  if (sessionsRefreshBtn) {
    sessionsRefreshBtn.addEventListener("click", refreshSessions);
  }
}

// ── Keyboard shortcut: ⌘\ / Ctrl+\ toggles collapse ──────────────────
function initSidebarShortcut() {
  document.addEventListener("keydown", (e) => {
    const cmd = e.metaKey || e.ctrlKey;
    if (cmd && e.key === "\\") {
      e.preventDefault();
      if (!sidebar || !sidebarCollapseBtn) return;
      // En mobile el shortcut abre/cierra el drawer; en desktop colapsa.
      const isMobile = window.matchMedia("(max-width: 767px)").matches;
      if (isMobile) {
        if (sidebar.getAttribute("data-state") === "open") {
          closeSidebarMobile();
        } else {
          openSidebarMobile();
        }
      } else {
        sidebarCollapseBtn.click();
      }
    }
  });
}

// ── Boot ─────────────────────────────────────────────────────────────
if (sidebar) {
  initSidebarCollapse();
  initSidebarMobile();
  initNewChatButtons();
  initSessionsSearch();
  initSidebarShortcut();
  // Primera carga de sesiones — no bloquea boot; si falla, el user ve
  // "no se pudo cargar el historial" y puede reintentar con el refresh
  // button.
  refreshSessions();
}


// ═══════════════════════════════════════════════════════════════════════
// Empty state toggle (2026-04-24) — claude.ai landing pattern
// ═══════════════════════════════════════════════════════════════════════
// `body.chat-empty` se activa cuando #messages no tiene ningún .line
// (o sea: no hubo todavía un turn real de user/rag). En empty state la
// CSS centra el composer verticalmente y muestra el #empty-hero.
//
// Usamos MutationObserver en vez de llamar updateEmptyState() en cada
// appendTurn / clearView / hydrateTurns / pushSystemMessage para no tener
// que recordar en 8 lugares distintos. El observer cubre cualquier mutación
// del subtree de #messages automáticamente.
//
// `.line` como criterio (no cualquier .turn) para que las meta-only
// notifications como "nueva sesión — historial en blanco" NO saquen al
// chat del estado empty — el hero sigue visible hasta que el user
// efectivamente pregunta algo o se hidrata una sesión con turns reales.
// ═══════════════════════════════════════════════════════════════════════

function updateChatEmptyState() {
  const hasRealTurn = messagesEl.querySelector(".line") !== null;
  document.body.classList.toggle("chat-empty", !hasRealTurn);
}

// Observa childList + subtree: cualquier mutación de #messages (append de
// turn, clear por /cls, hydrate de session) dispara el check.
const __messagesObserver = new MutationObserver(updateChatEmptyState);
__messagesObserver.observe(messagesEl, { childList: true, subtree: true });

// Run once on boot antes de que el observer registre algún evento — el
// default HTML tiene #messages vacío, entonces arrancamos en chat-empty.
updateChatEmptyState();


// ═══════════════════════════════════════════════════════════════════════
// Composer toolbar wiring (2026-04-24) — claude.ai-style controls
// ═══════════════════════════════════════════════════════════════════════
// Dos piezas activas en la bottom-row del composer:
//   · #composer-plus-btn     → abre el help modal (MVP; más adelante
//                               puede ser un menú de attach/voice/etc)
//   · #composer-mic-btn      → stub "próximamente" — whisper-cli STT
//                               ya existe en el proyecto pero wiring
//                               al browser es otro feature
// + Quick-chips: al click rellenan #input con data-query y submitean.
// (El badge de modelo se removió junto con el model-picker cuando
// pasamos al toggle de modo auto/rápido/profundo en la sidebar.)
// ═══════════════════════════════════════════════════════════════════════

const composerPlusBtn = document.getElementById("composer-plus-btn");
const composerMicBtn = document.getElementById("composer-mic-btn");

// ── Plus button: abre el help modal (atajos + slash commands) ──
if (composerPlusBtn && helpBtn) {
  composerPlusBtn.addEventListener("click", () => helpBtn.click());
}

// ── Mic button: stub — muestra un system message explicando que viene. ──
// Dedup intencional: si el user clickea el mic varias veces seguidas, NO
// apilamos el mismo meta-mensaje (Fer F. 2026-04-24 lo reportó con 3
// líneas idénticas en pantalla). Si la última línea del chat ya es este
// mismo texto, swallow el click. El mensaje sigue siendo visible para que
// el user lo vea — solo evitamos la duplicación visual ruidosa.
const _MIC_STUB_MSG = "dictado por voz en preparación — por ahora usá /tts para escuchar respuestas";

function _isLastMessageSameMicStub() {
  const last = messages.lastElementChild;
  if (!last) return false;
  const metas = last.querySelectorAll(".meta");
  if (!metas.length) return false;
  const tail = metas[metas.length - 1];
  return tail.textContent.trim() === _MIC_STUB_MSG.trim();
}

if (composerMicBtn) {
  composerMicBtn.addEventListener("click", () => {
    if (_isLastMessageSameMicStub()) return;
    pushSystemMessage("meta", _MIC_STUB_MSG);
  });
}

// ── Quick-chips: click rellena input + submit directo ──
document.querySelectorAll(".quick-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.getAttribute("data-query") || "";
    if (!q) return;
    input.value = q;
    autoGrow();
    updateSendBtnState();
    // Si el chip trae espacio al final (ej "buscar en el vault "), el
    // user probablemente quiere completarlo con más texto — no
    // submiteamos, solo focuseamos con el cursor al final.
    if (q.endsWith(" ")) {
      input.focus();
      input.setSelectionRange(q.length, q.length);
      return;
    }
    // Submit directo — mismo path que Enter.
    form.requestSubmit();
  });
});
