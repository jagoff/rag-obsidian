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
    // Dos fetchs en paralelo:
    //   - /api/wa/chats?q=... → contactos cuyo label matchea.
    //   - /api/wa/search?... → mensajes que matchean (FTS o semantic).
    // Renderizamos los contactos arriba, después los mensajes
    // (pedido user: "primero aparecen contactos que matcheen con la
    // busqueda, despues chats").
    const msgParams = new URLSearchParams({
      q, limit: "80", mode: searchBackend,
    });
    const chatParams = new URLSearchParams({ q, limit: "20" });
    const [chatResp, msgResp] = await Promise.all([
      fetch("/api/wa/chats?" + chatParams, { credentials: "same-origin" }),
      fetch("/api/wa/search?" + msgParams, { credentials: "same-origin" }),
    ]);
    if (!chatResp.ok) throw new Error(`chats ${chatResp.status}`);
    if (!msgResp.ok) throw new Error(`search ${msgResp.status}`);
    const chatData = await chatResp.json();
    const msgData = await msgResp.json();
    renderCombined(
      chatData.chats || [],
      msgData.hits || [],
      q,
      msgData.mode || searchBackend,
    );
  } catch (e) {
    console.error("[wa-search] failed", e);
    els.list.innerHTML = `<li class="wa-empty-state">error de search: ${e.message}</li>`;
  } finally {
    if (els.loading) els.loading.classList.add("hidden");
  }
}

function renderCombined(chats, hits, q, mode) {
  if (!els.list) return;
  els.list.innerHTML = "";
  if (chats.length === 0 && hits.length === 0) {
    els.list.innerHTML = `<li class="wa-search-empty">sin resultados para "${escapeHtml(q)}"</li>`;
    return;
  }
  // Sección 1: Contactos.
  if (chats.length > 0) {
    appendSection("Contactos", chats.length);
    for (const c of chats) {
      const li = document.createElement("li");
      li.className = "wa-search-contact";
      li.dataset.jid = c.jid;
      li.addEventListener("click", () => {
        if (onSelectCallback) onSelectCallback(c.jid, null);
      });
      const preview = (c.last_preview || "").trim();
      li.innerHTML = `
        <div class="wa-search-contact-name">${escapeHtml(c.label || c.jid)}</div>
        ${preview ? `<div class="wa-search-contact-preview">${escapeHtml(preview)}</div>` : ""}
      `;
      els.list.appendChild(li);
    }
  }
  // Sección 2: Mensajes.
  if (hits.length > 0) {
    appendSection("Mensajes", hits.length);
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
}

function appendSection(title, count) {
  const li = document.createElement("li");
  li.className = "wa-search-section";
  li.innerHTML = `<span class="wa-search-section-title">${escapeHtml(title)}</span>`
    + `<span class="wa-search-section-count">${count}</span>`;
  els.list.appendChild(li);
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
