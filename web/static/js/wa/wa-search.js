// Search FTS5 cross-chat. Refuerza el input ya existente del sidebar:
// cuando hay 3+ chars en el query, en vez de filtrar la chat list
// localmente (por nombre), busca matches en TODOS los mensajes y
// muestra hits con snippet en lugar de chats.

import { fetchChats } from "./wa-api.js";  // re-usado para volver a chats

const els = {
  input: null,
  list: null,
  loading: null,
  modeBar: null,
};

let activeMode = "chats"; // "chats" | "search"
let searchBackend = "fts"; // "fts" | "semantic"
let onSelectCallback = null;
let timer = null;

export function init({ inputEl, listEl, loadingEl, onSelect }) {
  els.input = inputEl;
  els.list = listEl;
  els.loading = loadingEl;
  els.modeBar = document.getElementById("wa-search-mode");
  onSelectCallback = onSelect;

  // Toggle FTS / Semantic. Hidden por default — solo aparece cuando
  // hay search activa para no consumir real estate cuando estás
  // navegando los chats.
  if (els.modeBar) {
    for (const btn of els.modeBar.querySelectorAll(".wa-search-mode-btn")) {
      btn.addEventListener("click", () => {
        searchBackend = btn.dataset.mode || "fts";
        for (const b of els.modeBar.querySelectorAll(".wa-search-mode-btn")) {
          b.classList.toggle("active", b === btn);
        }
        const q = (els.input?.value || "").trim();
        if (q.length >= 3) runSearch(q);
      });
    }
  }
}

/** Toma el control del input cuando la query es >=3 chars, devuelve
 * `true` si está manejando la búsqueda; el caller de chatlist debe
 * skipear su filter local cuando esto retorna true.
 */
export function maybeHandleSearch(query) {
  const q = (query || "").trim();
  if (q.length < 3) {
    if (activeMode === "search") activeMode = "chats";
    if (els.modeBar) els.modeBar.hidden = true;
    return false;
  }
  activeMode = "search";
  if (els.modeBar) els.modeBar.hidden = false;
  clearTimeout(timer);
  timer = setTimeout(() => runSearch(q), 200);
  return true;
}

export function isInSearchMode() {
  return activeMode === "search";
}

export function exitSearchMode() {
  activeMode = "chats";
  if (els.modeBar) els.modeBar.hidden = true;
}

async function runSearch(q) {
  if (!els.list) return;
  if (els.loading) els.loading.classList.remove("hidden");
  try {
    const params = new URLSearchParams({
      q,
      limit: "80",
      mode: searchBackend,
    });
    const r = await fetch("/api/wa/search?" + params, { credentials: "same-origin" });
    if (!r.ok) throw new Error(`search ${r.status}`);
    const data = await r.json();
    renderHits(data.hits || [], q, data.mode || searchBackend);
  } catch (e) {
    console.error("[wa-search] failed", e);
    els.list.innerHTML = `<li class="wa-empty-state">error de search: ${e.message}</li>`;
  } finally {
    if (els.loading) els.loading.classList.add("hidden");
  }
}

function renderHits(hits, q, mode) {
  if (!els.list) return;
  els.list.innerHTML = "";
  if (hits.length === 0) {
    els.list.innerHTML = `<li class="wa-search-empty">sin resultados para "${escapeHtml(q)}"</li>`;
    return;
  }
  for (const h of hits) {
    const li = document.createElement("li");
    li.className = "wa-search-hit";
    li.dataset.jid = h.chat_jid;
    li.dataset.messageId = h.id;
    li.addEventListener("click", () => {
      if (onSelectCallback) onSelectCallback(h.chat_jid, h.id);
    });
    const ts = h.ts ? new Date(h.ts) : null;
    const tsLabel = ts && !Number.isNaN(ts.getTime())
      ? ts.toLocaleDateString("es-AR", { day: "2-digit", month: "2-digit", year: "2-digit" })
      : "";
    // FTS devuelve `snippet` con `<mark>` highlights; semantic devuelve
    // `content` raw + `distance`. Renderizamos distinto pero con la
    // misma class (`.wa-search-snippet`).
    let bodyHTML;
    let scoreTag = "";
    if (mode === "semantic") {
      const dist = typeof h.distance === "number" ? h.distance.toFixed(2) : "";
      bodyHTML = escapeHtml(h.content || "");
      scoreTag = dist ? `<span class="wa-search-score" title="cosine distance — 0 = idéntico">d=${dist}</span>` : "";
    } else {
      bodyHTML = sanitizeSnippetHTML(h.snippet || "");
    }
    li.innerHTML = `
      <div class="wa-search-meta">
        <span class="wa-search-chat">${escapeHtml(h.chat_name || h.chat_jid)}</span>
        <span class="wa-search-ts">${tsLabel}${scoreTag}</span>
      </div>
      <div class="wa-search-snippet">${bodyHTML}</div>
    `;
    els.list.appendChild(li);
  }
}

// Permitir SOLO `<mark>` y `</mark>` en el snippet — el resto se escapa.
function sanitizeSnippetHTML(s) {
  if (!s) return "";
  const escaped = String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  return escaped
    .replace(/&lt;mark&gt;/g, "<mark>")
    .replace(/&lt;\/mark&gt;/g, "</mark>");
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
