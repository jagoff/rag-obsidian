const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const vaultPicker = document.getElementById("vault-picker");
const ttsToggle = document.getElementById("tts-toggle");
const helpBtn = document.getElementById("help-btn");
const helpModal = document.getElementById("help-modal");
const stopBtn = document.getElementById("stop-btn");

const SESSION_KEY = "obsidian-rag:session";
const VAULT_KEY = "obsidian-rag:vault";
const TTS_KEY = "obsidian-rag:tts";
let sessionId = localStorage.getItem(SESSION_KEY) || null;
let vaultScope = localStorage.getItem(VAULT_KEY) || "";
let ttsEnabled = localStorage.getItem(TTS_KEY) === "1";
let pending = false;
let currentController = null;      // AbortController for in-flight /api/chat
let currentAudio = null;           // In-flight <audio> playback
let lastUserQuestion = "";         // For ⌘↑ edit-last

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
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("blur", () => {
  // Small delay so a mousedown on a slash-item still registers before the
  // popover disappears.
  setTimeout(hideSlashPopover, 120);
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
    const scoreStr = (s.score >= 0 ? "+" : "") + s.score.toFixed(1);
    row.appendChild(el("span", "bar", `${s.bar}  ${scoreStr}`));
    const note = el("a", "note", s.note || s.file);
    note.href = obsidianUrl(s.file);
    note.title = s.file;
    row.appendChild(note);
    row.appendChild(el("span", "path", s.file));
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
  },
});

function preprocess(text) {
  let out = text.replace(/<<ext>>([\s\S]*?)<<\/ext>>/g, (_, body) => {
    return `\u0000EXT_OPEN\u0000${body}\u0000EXT_CLOSE\u0000`;
  });
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
  return postprocess(marked.parse(preprocess(text)));
}

// Send --------------------------------------------------------
async function send(question) {
  if (pending) return;
  pending = true;
  lastUserQuestion = question;
  input.disabled = true;
  stopBtn.hidden = false;
  currentController = new AbortController();

  const turn = appendTurn();
  appendLine(turn, "user", question);
  const thinking = el("div", "thinking");
  thinking.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  turn.appendChild(thinking);
  scrollBottom();

  let ragText = null;
  let ragLine = null;
  let fullText = "";
  let sources = null;
  let confidence = null;
  let metaShown = false;
  let aborted = false;

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
      localStorage.setItem(SESSION_KEY, sessionId);
    } else if (event === "meta") {
      if (!metaShown) {
        appendMeta(turn, parsed.bits);
        metaShown = true;
      }
    } else if (event === "sources") {
      sources = parsed.items;
      if (Number.isFinite(parsed.confidence)) confidence = parsed.confidence;
    } else if (event === "token") {
      if (!ragLine) {
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
      if (sources && sources.length) {
        const conf = Number.isFinite(confidence) ? confidence : parsed.top_score;
        appendSources(turn, sources, conf);
      }
      if (fullText.trim()) {
        const actions = el("div", "msg-actions");
        appendCopyButton(actions, () => buildMarkdownExport(question, fullText, sources));
        turn.appendChild(actions);
      }
      if (parsed.turn_id) {
        appendFeedback(turn, {
          turn_id: parsed.turn_id,
          q: question,
          paths: (sources || []).map((s) => s.file).filter(Boolean),
          session_id: sessionId,
        });
      }
      if (sessionId && !aborted) appendFollowups(turn, sessionId);
      if (ttsEnabled && !aborted && fullText.trim()) speak(fullText);
      scrollBottom();
    } else if (event === "empty") {
      thinking.remove();
      turn.appendChild(el("div", "empty", `  ${parsed.message || "Sin resultados relevantes."}`));
    } else if (event === "error") {
      thinking.remove();
      turn.appendChild(el("div", "error", `  ${parsed.message || "Error"}`));
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
    localStorage.removeItem(SESSION_KEY);
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

// Auto-submit when arriving from a deep-link like /chat?q=foo
(() => {
  const params = new URLSearchParams(window.location.search);
  const seed = params.get("q");
  if (!seed) return;
  history.replaceState({}, "", window.location.pathname);
  input.value = seed;
  setTimeout(() => send(seed), 50);
})();
