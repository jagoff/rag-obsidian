const state = {
  config: {},
  kinds: [],
  path: "",
};

const listsEl = document.querySelector("#lists");
const statusEl = document.querySelector("#status");
const pathEl = document.querySelector("#config-path");

const fallbackKinds = [
  ["chats", "Grupos / chats"],
  ["people", "Personas"],
  ["topics", "Temas"],
  ["words", "Palabras exactas"],
  ["fuzzy_words", "Palabras parecidas"],
  ["paths", "Paths exactos"],
  ["path_prefixes", "Prefijos de path"],
  ["path_globs", "Globs de path"],
].map(([key, label]) => ({ key, label }));

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`.trim();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function valuesFor(kind) {
  const values = state.config[kind] || [];
  return Array.from(new Set(values)).sort((a, b) => String(a).localeCompare(String(b)));
}

function render() {
  const kinds = state.kinds.length ? state.kinds : fallbackKinds;
  pathEl.textContent = state.path || "-";
  listsEl.innerHTML = kinds.map((kind) => {
    const values = valuesFor(kind.key);
    const rows = values.length
      ? values.map((value) => `
          <div class="item-row">
            <code title="${escapeHtml(value)}">${escapeHtml(value)}</code>
            <button class="delete" data-kind="${escapeHtml(kind.key)}" data-value="${escapeHtml(value)}" title="Eliminar">×</button>
          </div>
        `).join("")
      : `<div class="empty">vacío</div>`;
    return `
      <section class="blacklist-section" data-kind="${escapeHtml(kind.key)}">
        <div class="section-head">
          <h2>${escapeHtml(kind.label || kind.key)}</h2>
          <span class="count">${values.length}</span>
        </div>
        <form class="add-form" data-kind="${escapeHtml(kind.key)}">
          <input name="value" autocomplete="off" spellcheck="false" />
          <button class="add" type="submit" title="Agregar">+</button>
        </form>
        <div class="items">${rows}</div>
      </section>
    `;
  }).join("");
}

async function requestJson(url, options = {}) {
  const resp = await fetch(url, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!resp.ok) {
    let detail = `${resp.status}`;
    try {
      const data = await resp.json();
      detail = data.detail || detail;
    } catch (_err) {
      detail = await resp.text();
    }
    throw new Error(detail);
  }
  return resp.json();
}

async function load() {
  setStatus("cargando");
  const data = await requestJson("/api/blacklist");
  state.config = data.config || {};
  state.kinds = data.kinds || [];
  state.path = data.path || "";
  render();
  setStatus("listo", "ok");
}

async function addItem(kind, value) {
  setStatus("guardando");
  const data = await requestJson("/api/blacklist", {
    method: "POST",
    body: JSON.stringify({ kind, value }),
  });
  state.config = data.config || {};
  state.path = data.path || state.path;
  render();
  setStatus(data.changed ? "agregado" : "sin cambios", "ok");
}

async function deleteItem(kind, value) {
  setStatus("guardando");
  const data = await requestJson("/api/blacklist/delete", {
    method: "POST",
    body: JSON.stringify({ kind, value }),
  });
  state.config = data.config || {};
  state.path = data.path || state.path;
  render();
  setStatus(data.changed ? "eliminado" : "sin cambios", "ok");
}

listsEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target.closest(".add-form");
  if (!form) return;
  const input = form.elements.value;
  const value = String(input.value || "").trim();
  if (!value) return;
  try {
    await addItem(form.dataset.kind, value);
    input.value = "";
    input.focus();
  } catch (err) {
    setStatus(err.message || "error", "error");
  }
});

listsEl.addEventListener("click", async (event) => {
  const button = event.target.closest("button.delete");
  if (!button) return;
  try {
    await deleteItem(button.dataset.kind, button.dataset.value);
  } catch (err) {
    setStatus(err.message || "error", "error");
  }
});

load().catch((err) => {
  render();
  setStatus(err.message || "error", "error");
});
