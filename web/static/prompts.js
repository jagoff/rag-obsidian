const state = {
  prompts: [],
  inline: [],
  selected: null,
  selectedContent: "",
  filter: "all",
  query: "",
  dirty: false,
};

const listEl = document.querySelector("#prompt-list");
const detailEl = document.querySelector("#detail");
const editorEl = document.querySelector("#editor");
const searchEl = document.querySelector("#search");
const saveBtn = document.querySelector("#save-btn");
const reloadBtn = document.querySelector("#reload-btn");
const saveStateEl = document.querySelector("#save-state");
const rootLabelEl = document.querySelector("#root-label");

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function authHeaders() {
  const token = await window.__ragAdminAuth?.loadToken?.();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function requestJson(url, options = {}) {
  const headers = {
    ...(await authHeaders()),
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {}),
  };
  const resp = await fetch(url, {
    cache: "no-store",
    ...options,
    headers,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      detail = data.detail || detail;
    } catch (_err) {
      const text = await resp.text();
      if (text) detail = text.slice(0, 300);
    }
    throw new Error(detail);
  }
  return resp.json();
}

function setSaveState(text, kind = "") {
  saveStateEl.textContent = text;
  saveStateEl.className = `save-state ${kind}`.trim();
}

function setDirty(dirty) {
  state.dirty = dirty;
  saveBtn.disabled = !dirty || !state.selected?.editable;
  if (dirty) setSaveState("sin guardar", "warn");
  else if (state.selected?.editable) setSaveState("listo", "ok");
}

function promptLabel(item) {
  const version = item.version ? `.${item.version}` : "";
  return item.name ? `${item.name}${version}` : item.symbol || item.path;
}

function badges(item) {
  const values = [];
  if (item.importance) {
    values.push({
      cls: `importance importance-${item.importance}`,
      label: item.importance_label || item.importance,
    });
  }
  if (item.status) values.push({ cls: item.status, label: item.status });
  if (item.group) values.push({ cls: "group", label: item.group });
  if (!item.editable) values.push({ cls: "readonly", label: "source" });
  return values.map(({ cls, label }) => {
    const classes = ["badge", cls].join(" ");
    return `<span class="${escapeHtml(classes)}">${escapeHtml(label)}</span>`;
  }).join("");
}

function importanceClass(item) {
  const importance = item.importance || "medium";
  return ["high", "medium", "low"].includes(importance) ? ` importance-${importance}` : "";
}

function detailBlock(label, value) {
  if (!value) return "";
  return `
    <section class="detail-section">
      <div class="detail-label">${escapeHtml(label)}</div>
      <p>${escapeHtml(value)}</p>
    </section>
  `;
}

function renderTextLines(lines) {
  return lines.filter(Boolean).join("\n");
}

function metadataText(item) {
  return renderTextLines([
    `# ${promptLabel(item)}`,
    "",
    `Path: ${item.path}`,
    item.symbol ? `Símbolo: ${item.symbol}` : "",
    item.importance_label ? `Importancia: ${item.importance_label}` : "",
    item.impact ? `Impacto: ${item.impact}` : "",
    "",
    item.purpose || "",
    item.effective ? `\nQué hace efectivamente:\n${item.effective}` : "",
  ]);
}

function renderInlineContent(item, content) {
  return renderTextLines([
    metadataText(item),
    "",
    "Solo lectura: este prompt vive inline en código. La edición directa segura está habilitada para los prompts Markdown bajo rag/prompts.",
    "",
    "# Contenido",
    "",
    content || "",
  ]);
}

function renderPromptContent(item, content) {
  if (!content) return "";
  return content;
}

function rowText(item) {
  const effective = item.effective && item.effective !== item.purpose ? item.effective : "";
  return effective || item.impact || "";
}

function rowTitle(item) {
  const label = promptLabel(item);
  const importance = item.importance_label ? ` · ${item.importance_label}` : "";
  return `${label}${importance}`;
}

function rowClasses(item, active) {
  return [
    "prompt-row",
    active ? "is-active" : "",
    item.editable ? "" : "inline",
    importanceClass(item).trim(),
  ].filter(Boolean).join(" ");
}

function renderBadge(label, cls = "") {
  return `<span class="${escapeHtml(["badge", cls].filter(Boolean).join(" "))}">${escapeHtml(label)}</span>`;
}

function renderImportanceMeta(item) {
  if (!item.importance) return "";
  return renderBadge(item.importance_label || item.importance, `importance importance-${item.importance}`);
}

function renderStatusMeta(item) {
  return [
    item.kind ? renderBadge(item.kind) : "",
    item.status ? renderBadge(item.status, item.status) : "",
    item.superseded_by ? renderBadge(`superseded by ${item.superseded_by}`) : "",
  ].filter(Boolean).join("");
}

function renderIncludeBadges(item) {
  return (item.includes || []).map((inc) => renderBadge(inc)).join("");
}

function renderPath(item) {
  return `<code>${escapeHtml(item.path)}</code>`;
}

function renderMeta(item) {
  return [
    renderPath(item),
    renderImportanceMeta(item),
    renderStatusMeta(item),
    renderIncludeBadges(item),
  ].filter(Boolean).join("");
}

function renderPurpose(item) {
  return `
    <p class="detail-purpose">${escapeHtml(item.purpose || "")}</p>
    ${detailBlock("Qué hace efectivamente", item.effective)}
    ${detailBlock("Impacto", item.impact || item.importance_reason)}
  `;
}

function rowMarkup(item) {
  const active = state.selected?.id === item.id;
  const extra = rowText(item);
  return `
    <button class="${escapeHtml(rowClasses(item, active))}" type="button" data-id="${escapeHtml(item.id)}" title="${escapeHtml(rowTitle(item))}">
      <div class="row-top">
        <span class="row-title">${escapeHtml(promptLabel(item))}</span>
        <span class="badges">${badges(item)}</span>
      </div>
      <div class="row-purpose">${escapeHtml(item.purpose)}</div>
      ${extra ? `<div class="row-effective">${escapeHtml(extra)}</div>` : ""}
      <div class="row-path">${escapeHtml(item.path)}</div>
    </button>
  `;
}

function updateSelectedMetadata(prompt, content) {
  state.selected = prompt;
  state.selectedContent = content || "";
  editorEl.value = renderPromptContent(prompt, state.selectedContent);
}

function setEditableEditor(enabled) {
  editorEl.disabled = !enabled;
}

function latestEditablePrompt() {
  return state.prompts.find((p) => p.latest && p.importance === "high")
    || state.prompts.find((p) => p.latest)
    || state.prompts[0];
}

function searchableText(item) {
  return [
    promptLabel(item),
    item.path,
    item.purpose,
    item.effective,
    item.impact,
    item.importance,
    item.importance_label,
    item.importance_reason,
    item.group,
    item.kind,
    item.status,
    item.symbol,
  ].join(" ").toLowerCase();
}

function matchesQuery(item, q) {
  return !q || searchableText(item).includes(q);
}

function matchesFilter(item) {
  if (state.filter === "editable" && !item.editable) return false;
  if (state.filter === "inline" && item.editable) return false;
  if (state.filter === "high" && item.importance !== "high") return false;
  if (state.filter === "medium" && item.importance !== "medium") return false;
  if (state.filter === "low" && item.importance !== "low") return false;
  return true;
}

function allItems() {
  return [
    ...state.prompts,
    ...state.inline.map((item) => ({ ...item, inline: true })),
  ];
}

function filteredItems() {
  const q = state.query.trim().toLowerCase();
  return allItems().filter((item) => matchesFilter(item) && matchesQuery(item, q));
}

function renderList() {
  const items = filteredItems();
  if (!items.length) {
    listEl.innerHTML = `<div class="list-empty">sin resultados</div>`;
    return;
  }
  listEl.innerHTML = items.map(rowMarkup).join("");
}

function renderDetail(item) {
  if (!item) {
    detailEl.className = "detail empty";
    detailEl.innerHTML = `<h2>Seleccioná un prompt</h2><p>—</p>`;
    editorEl.value = "";
    editorEl.disabled = true;
    return;
  }
  detailEl.className = "detail";
  detailEl.innerHTML = `
    <h2>${escapeHtml(promptLabel(item))}</h2>
    ${renderPurpose(item)}
    <div class="detail-grid">${renderMeta(item)}</div>
  `;
}

async function selectInline(item) {
  if (state.dirty && !window.confirm("Hay cambios sin guardar. ¿Descartarlos?")) return;
  setSaveState("cargando");
  state.selected = item;
  state.selectedContent = "";
  const data = await requestJson(`/api/prompts/inline?id=${encodeURIComponent(item.id)}`);
  state.selected = data.prompt || item;
  renderList();
  renderDetail(state.selected);
  editorEl.value = renderInlineContent(state.selected, data.content || "");
  setEditableEditor(false);
  setDirty(false);
  setSaveState("solo lectura", "warn");
}

async function selectPrompt(item) {
  if (!item.editable) {
    await selectInline(item);
    return;
  }
  if (state.dirty && !window.confirm("Hay cambios sin guardar. ¿Descartarlos?")) return;
  setSaveState("cargando");
  const data = await requestJson(`/api/prompts/file?path=${encodeURIComponent(item.path)}`);
  updateSelectedMetadata(data.prompt, data.content || "");
  setEditableEditor(true);
  renderList();
  renderDetail(state.selected);
  setDirty(false);
}

async function loadList(keepSelection = true) {
  setSaveState("cargando");
  const data = await requestJson("/api/prompts");
  state.prompts = data.prompts || [];
  state.inline = data.inline || [];
  rootLabelEl.textContent = `${data.root || "rag/prompts"} · ${state.prompts.length} editables · ${state.inline.length} inline`;
  renderList();
  if (keepSelection && state.selected) {
    const found = allItems().find((item) => item.id === state.selected.id);
    if (found && found.editable) await selectPrompt(found);
    else if (found) await selectInline(found);
    else {
      state.selected = null;
      renderDetail(null);
    }
  } else if (!state.selected && state.prompts.length) {
    await selectPrompt(latestEditablePrompt());
  }
  setSaveState(state.selected?.editable ? "listo" : "solo lectura", state.selected?.editable ? "ok" : "warn");
}

async function saveCurrent() {
  if (!state.selected?.editable || !state.dirty) return;
  setSaveState("guardando");
  saveBtn.disabled = true;
  const data = await requestJson("/api/prompts/file", {
    method: "POST",
    body: JSON.stringify({
      path: state.selected.path,
      content: editorEl.value,
    }),
  });
  state.selected = data.prompt;
  state.selectedContent = editorEl.value.endsWith("\n") ? editorEl.value : `${editorEl.value}\n`;
  const idx = state.prompts.findIndex((item) => item.id === state.selected.id);
  if (idx >= 0) state.prompts[idx] = state.selected;
  renderList();
  renderDetail(state.selected);
  setDirty(false);
  setSaveState("guardado", "ok");
}

listEl.addEventListener("click", async (event) => {
  const row = event.target.closest(".prompt-row");
  if (!row) return;
  const item = allItems().find((candidate) => candidate.id === row.dataset.id);
  if (!item) return;
  try {
    await selectPrompt(item);
  } catch (err) {
    setSaveState(err.message || "error", "error");
  }
});

searchEl.addEventListener("input", () => {
  state.query = searchEl.value || "";
  renderList();
});

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((el) => el.classList.remove("is-active"));
    button.classList.add("is-active");
    state.filter = button.dataset.filter || "all";
    renderList();
  });
});

editorEl.addEventListener("input", () => {
  if (!state.selected?.editable) return;
  setDirty(editorEl.value !== state.selectedContent);
});

saveBtn.addEventListener("click", () => {
  saveCurrent().catch((err) => {
    setSaveState(err.message || "error", "error");
    saveBtn.disabled = false;
  });
});

reloadBtn.addEventListener("click", () => {
  if (state.dirty && !window.confirm("Hay cambios sin guardar. ¿Descartarlos?")) return;
  loadList(true).catch((err) => {
    listEl.innerHTML = `<div class="list-error">${escapeHtml(err.message || "error")}</div>`;
    setSaveState(err.message || "error", "error");
  });
});

window.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveCurrent().catch((err) => setSaveState(err.message || "error", "error"));
  }
});

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

loadList(false).catch((err) => {
  listEl.innerHTML = `<div class="list-error">${escapeHtml(err.message || "error")}</div>`;
  setSaveState(err.message || "error", "error");
});
