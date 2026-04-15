const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const vaultPicker = document.getElementById("vault-picker");

const SESSION_KEY = "obsidian-rag:session";
const VAULT_KEY = "obsidian-rag:vault";
let sessionId = localStorage.getItem(SESSION_KEY) || null;
let vaultScope = localStorage.getItem(VAULT_KEY) || "";
let pending = false;

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

// Input autogrow + enter-to-send --------------------------------
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}
input.addEventListener("input", autoGrow);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
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

function appendSources(parent, items) {
  const wrap = el("div", "sources");
  wrap.appendChild(el("div", "sources-rule", "╌ fuentes ".padEnd(64, "╌")));
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

function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function obsidianUrl(filePath) {
  return "obsidian://open?file=" + encodeURIComponent(filePath);
}

// Markdown via marked. Dos transformaciones se aplican ANTES de marked:
//   • <<ext>>…<</ext>>  → <span class="ext">⚠ …</span>      (marker propio del prompt)
//   • [[Wikilinks]]      → [Wikilinks](obsidian://…)           (links de Obsidian)
// Y en el renderer custom reescribimos links .md a `obsidian://open?file=…`
// para que clickear abra la nota en Obsidian, no navegue a un 404.
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
    // Placeholder token neutral al parser; lo re-reemplazamos post-render.
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
  input.disabled = true;

  const turn = appendTurn();
  appendLine(turn, "user", question);
  const thinking = el("div", "thinking", "pensando…");
  turn.appendChild(thinking);
  scrollBottom();

  let ragText = null;
  let ragLine = null;
  let fullText = "";
  let sources = null;
  let metaShown = false;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId, vault_scope: vaultScope || null }),
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
    thinking.remove();
    turn.appendChild(el("div", "error", `  error: ${err.message}`));
  } finally {
    pending = false;
    input.disabled = false;
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
      if (sources && sources.length) appendSources(turn, sources);
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

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  send(q);
});
