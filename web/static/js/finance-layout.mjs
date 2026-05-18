// finance-layout.mjs — drag & drop de secciones, collapse y resize.
// Estado persistido en localStorage (solo este navegador).

import {
  applySizeDataset,
  isValidSizeOverride,
  readJSON,
  readObject,
  readSizeOverrides,
  removeKeys,
  writeJSON,
  writeObjectOrRemove,
  writeSizeOverrides,
} from "./layout-persistence.mjs";

// ── Constantes de localStorage ─────────────────────────────────────────────────

const LS_SECTIONS_ORDER = "finance.sections.order.v1";
const LS_SECTIONS_COLLAPSED = "finance.sections.collapsed.v1";
const LS_SECTION_SIZES = "finance.sections.sizes.v1";

const SECTION_HEIGHTS = ["normal", "tall", "xl"];
const SECTION_SIZE_OPTIONS = { widths: ["full"], heights: SECTION_HEIGHTS };
const SECTION_HEIGHT_LABEL = {
  normal: "normal",
  tall: "alto",
  xl: "extra alto",
};

// ── Orden persistido ───────────────────────────────────────────────────────────

function readSavedOrder() {
  const parsed = readJSON(LS_SECTIONS_ORDER, null);
  return Array.isArray(parsed) ? parsed : null;
}

function saveCurrentOrder() {
  const order = Array.from(document.querySelectorAll(".learn-section"))
    .map((s) => s.id)
    .filter(Boolean);
  if (!writeJSON(LS_SECTIONS_ORDER, order)) {
    console.warn("[finance] no pude persistir el orden de secciones");
  }
  updateResetButtonVisibility();
}

export function applySavedOrder() {
  const saved = readSavedOrder();
  if (!saved) return;
  const container = document.querySelector("main");
  if (!container) return;
  for (const id of saved) {
    const section = document.getElementById(id);
    if (!section) continue;
    container.appendChild(section);
  }
}

function clearSavedLayout() {
  removeKeys([LS_SECTIONS_ORDER, LS_SECTIONS_COLLAPSED, LS_SECTION_SIZES]);
  window.location.reload();
}

// ── Collapse por sección ─────────────────────────────────────────────────────

function readCollapsedMap() {
  return readObject(LS_SECTIONS_COLLAPSED, {});
}

function writeCollapsedMap(map) {
  const trimmed = {};
  if (map && typeof map === "object") {
    for (const [key, value] of Object.entries(map)) {
      if (key && typeof value === "boolean") trimmed[key] = value;
    }
  }
  if (!writeObjectOrRemove(LS_SECTIONS_COLLAPSED, trimmed)) {
    console.warn("[finance] no pude persistir collapse de secciones");
  }
}

function initSectionDefaultCollapse(section) {
  if (!section || section.dataset.defaultCollapsed) return;
  const body = section.querySelector(".learn-section-body");
  const btn = section.querySelector(".collapse-btn");
  const collapsed = Boolean(
    (body && (body.hidden || body.classList.contains("collapsed"))) ||
    (btn && btn.getAttribute("aria-expanded") === "false")
  );
  section.dataset.defaultCollapsed = collapsed ? "true" : "false";
}

function defaultCollapsed(section) {
  initSectionDefaultCollapse(section);
  return section.dataset.defaultCollapsed === "true";
}

function setSectionCollapsed(section, collapsed) {
  const body = section.querySelector(".learn-section-body");
  const btn = section.querySelector(".collapse-btn");
  if (body) {
    body.hidden = collapsed;
    body.classList.toggle("collapsed", collapsed);
  }
  if (btn) {
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.textContent = collapsed ? "+" : "−";
  }
}

function persistSectionCollapse(section, collapsed) {
  if (!section || !section.id) return;
  const map = readCollapsedMap();
  if (collapsed === defaultCollapsed(section)) delete map[section.id];
  else map[section.id] = collapsed;
  writeCollapsedMap(map);
  updateResetButtonVisibility();
}

export function applySavedCollapse() {
  const map = readCollapsedMap();
  document.querySelectorAll(".learn-section").forEach((section) => {
    initSectionDefaultCollapse(section);
    const collapsed = Object.prototype.hasOwnProperty.call(map, section.id)
      ? !!map[section.id]
      : defaultCollapsed(section);
    setSectionCollapsed(section, collapsed);
  });
}

function wireSectionCollapse(section) {
  if (!section || section.dataset.collapsePersistInit === "1") return;
  const btn = section.querySelector(".collapse-btn");
  if (!btn) return;
  section.dataset.collapsePersistInit = "1";
  btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
  btn.addEventListener("dragstart", (ev) => ev.preventDefault());
  btn.addEventListener("click", () => {
    // finance.js toggles the CSS class; persist the final state after it runs.
    window.setTimeout(() => {
      const body = section.querySelector(".learn-section-body");
      const collapsed = body
        ? body.classList.contains("collapsed")
        : btn.getAttribute("aria-expanded") === "false";
      setSectionCollapsed(section, collapsed);
      persistSectionCollapse(section, collapsed);
    }, 0);
  });
}

// ── Tamaño persistido por sección ─────────────────────────────────────────────

function normalizeSectionHeight(value) {
  return SECTION_HEIGHTS.includes(value) ? value : "normal";
}

function currentSectionHeight(section) {
  return normalizeSectionHeight(section?.dataset?.h);
}

function readSectionSizes() {
  return readSizeOverrides(LS_SECTION_SIZES, SECTION_SIZE_OPTIONS);
}

function resizeChartsSoon() {
  if (typeof window === "undefined") return;
  const notify = () => {
    window.dispatchEvent(new Event("resize"));
  };
  if (window.requestAnimationFrame) window.requestAnimationFrame(notify);
  else window.setTimeout(notify, 0);
}

function updateSectionSizeButton(section) {
  const btn = section.querySelector(".section-size-btn");
  if (!btn) return;
  const cur = currentSectionHeight(section);
  const next = nextSectionHeight(cur);
  btn.title = `Tamaño: ${SECTION_HEIGHT_LABEL[cur]}. Click: ${SECTION_HEIGHT_LABEL[next]}`;
  btn.setAttribute("aria-label", `Cambiar tamaño de sección. Actual: ${SECTION_HEIGHT_LABEL[cur]}`);
}

function setSectionSize(section, height, opts = {}) {
  if (!section || !section.id) return;
  const persist = opts.persist !== false;
  const h = normalizeSectionHeight(height);
  applySizeDataset(section, { w: "full", h });
  updateSectionSizeButton(section);
  resizeChartsSoon();
  if (!persist) return;

  const overrides = readSectionSizes();
  if (h === "normal") delete overrides[section.id];
  else overrides[section.id] = { w: "full", h };
  writeSizeOverrides(LS_SECTION_SIZES, overrides, SECTION_SIZE_OPTIONS);
  updateResetButtonVisibility();
}

export function applySavedSectionSize(section) {
  if (!section || !section.id) return false;
  const overrides = readSectionSizes();
  const override = overrides[section.id];
  const h = isValidSizeOverride(override, SECTION_SIZE_OPTIONS) ? override.h : "normal";
  applySizeDataset(section, { w: "full", h });
  updateSectionSizeButton(section);
  return h !== "normal";
}

function hasSectionSizeOverrides() {
  return Object.keys(readSectionSizes()).length > 0;
}

function nextSectionHeight(current) {
  const idx = SECTION_HEIGHTS.indexOf(normalizeSectionHeight(current));
  return SECTION_HEIGHTS[(idx + 1) % SECTION_HEIGHTS.length];
}

function cycleSectionHeight(section) {
  setSectionSize(section, nextSectionHeight(currentSectionHeight(section)));
}

export function injectSectionSizeControls(section) {
  if (!section || section.dataset.sizeControlsInit === "1") return;
  const header = section.querySelector(".learn-section-hdr");
  if (!header) return;
  section.dataset.sizeControlsInit = "1";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "section-size-btn";
  btn.textContent = "↕";
  btn.setAttribute("draggable", "false");
  btn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    cycleSectionHeight(section);
  });
  btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
  btn.addEventListener("dragstart", (ev) => ev.preventDefault());

  const collapse = header.querySelector(".collapse-btn");
  header.insertBefore(btn, collapse || null);
  updateSectionSizeButton(section);
}

function heightFromDrag(startHeight, deltaY) {
  const startIdx = SECTION_HEIGHTS.indexOf(normalizeSectionHeight(startHeight));
  let shift = 0;
  if (deltaY > 220) shift = 2;
  else if (deltaY > 64) shift = 1;
  else if (deltaY < -220) shift = -2;
  else if (deltaY < -64) shift = -1;
  const nextIdx = Math.max(0, Math.min(SECTION_HEIGHTS.length - 1, startIdx + shift));
  return SECTION_HEIGHTS[nextIdx];
}

function onSectionResizePointerDown(ev) {
  if (ev.button != null && ev.button !== 0) return;
  const handle = ev.currentTarget;
  const section = handle.closest(".learn-section");
  if (!section) return;

  ev.preventDefault();
  ev.stopPropagation();
  const startY = ev.clientY;
  const startHeight = currentSectionHeight(section);
  let nextHeight = startHeight;
  section.classList.add("is-resizing");
  try { handle.setPointerCapture(ev.pointerId); } catch {}

  const onMove = (moveEv) => {
    const deltaY = moveEv.clientY - startY;
    nextHeight = heightFromDrag(startHeight, deltaY);
    setSectionSize(section, nextHeight, { persist: false });
  };
  const onDone = () => {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onDone);
    window.removeEventListener("pointercancel", onDone);
    section.classList.remove("is-resizing");
    setSectionSize(section, nextHeight, { persist: true });
  };

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onDone, { once: true });
  window.addEventListener("pointercancel", onDone, { once: true });
}

export function injectSectionResizeHandle(section) {
  if (!section || section.dataset.resizeHandleInit === "1") return;
  section.dataset.resizeHandleInit = "1";
  const handle = document.createElement("div");
  handle.className = "finance-resize-handle";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "horizontal");
  handle.setAttribute("aria-label", "Redimensionar sección");
  handle.title = "Arrastrá para cambiar el alto";
  handle.setAttribute("draggable", "false");
  handle.addEventListener("pointerdown", onSectionResizePointerDown);
  section.appendChild(handle);
}

// ── Drag & drop de secciones ─────────────────────────────────────────────────

let _draggingSection = null;

function isInteractiveDragTarget(target) {
  return Boolean(target && target.closest && target.closest(
    "button, a, input, select, textarea, [contenteditable], .finance-resize-handle, .tx-scroll, canvas"
  ));
}

function onSectionDragStart(ev) {
  const section = ev.currentTarget;
  const target = ev.target;
  const fromHeader = Boolean(target && target.closest && target.closest(".learn-section-hdr"));
  if (!fromHeader || isInteractiveDragTarget(target)) {
    ev.preventDefault();
    return;
  }
  _draggingSection = section;
  section.classList.add("is-dragging");
  try {
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/plain", section.id);
  } catch {}
}

function onSectionDragEnd(ev) {
  const section = ev.currentTarget;
  section.classList.remove("is-dragging");
  _draggingSection = null;
  document.querySelectorAll(".learn-section.drop-before, .learn-section.drop-after")
    .forEach((s) => s.classList.remove("drop-before", "drop-after"));
}

function onSectionDragOver(ev) {
  if (!_draggingSection) return;
  const section = ev.currentTarget;
  if (section === _draggingSection) return;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  const rect = section.getBoundingClientRect();
  const before = (ev.clientY - rect.top) < rect.height / 2;
  section.classList.toggle("drop-before", before);
  section.classList.toggle("drop-after", !before);
}

function onSectionDragLeave(ev) {
  const section = ev.currentTarget;
  if (section.contains(ev.relatedTarget)) return;
  section.classList.remove("drop-before", "drop-after");
}

function onSectionDrop(ev) {
  ev.preventDefault();
  if (!_draggingSection) return;
  const target = ev.currentTarget;
  if (target === _draggingSection) return;
  const before = target.classList.contains("drop-before");
  target.classList.remove("drop-before", "drop-after");
  if (before) {
    target.parentNode.insertBefore(_draggingSection, target);
  } else {
    target.parentNode.insertBefore(_draggingSection, target.nextSibling);
  }
  saveCurrentOrder();
}

export function makeSectionDraggable(section) {
  if (section.dataset.draggableInit === "1") return;
  section.dataset.draggableInit = "1";
  section.setAttribute("draggable", "true");
  const header = section.querySelector(".learn-section-hdr");
  if (header && !header.querySelector(".drag-grip")) {
    const grip = document.createElement("span");
    grip.className = "drag-grip";
    grip.setAttribute("aria-hidden", "true");
    grip.title = "arrastrá para reordenar";
    grip.textContent = "⋮⋮";
    header.insertBefore(grip, header.firstChild);
  }
  section.addEventListener("dragstart", onSectionDragStart);
  section.addEventListener("dragend", onSectionDragEnd);
  section.addEventListener("dragover", onSectionDragOver);
  section.addEventListener("dragleave", onSectionDragLeave);
  section.addEventListener("drop", onSectionDrop);
}

// ── Botón reset + visibilidad ─────────────────────────────────────────────────

function hasCustomLayout() {
  if (readSavedOrder()) return true;
  const map = readCollapsedMap();
  if (map && Object.keys(map).length > 0) return true;
  if (hasSectionSizeOverrides()) return true;
  return false;
}

export function updateResetButtonVisibility() {
  const btn = document.getElementById("reset-layout-btn");
  if (!btn) return;
  btn.hidden = !hasCustomLayout();
}

export function injectResetButton() {
  const header = document.querySelector("header");
  if (!header || document.getElementById("reset-layout-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "reset-layout-btn";
  btn.className = "reset-layout-btn";
  btn.title = "Resetear layout (orden + tamaños + secciones colapsadas)";
  btn.setAttribute("aria-label", "Resetear layout de las secciones");
  btn.textContent = "↺ layout";
  btn.hidden = !hasCustomLayout();
  btn.addEventListener("click", () => {
    if (confirm("¿Resetear el layout de las secciones al default? (orden + tamaños + collapse)")) {
      clearSavedLayout();
    }
  });
  header.appendChild(btn);
}

// ── Init principal del layout ─────────────────────────────────────────────────

export function initFinanceLayout() {
  // Aplicar orden persistido antes de que los renderers escriban.
  applySavedOrder();

  document.querySelectorAll(".learn-section").forEach((section) => {
    initSectionDefaultCollapse(section);
    applySavedSectionSize(section);
    makeSectionDraggable(section);
    injectSectionSizeControls(section);
    injectSectionResizeHandle(section);
    wireSectionCollapse(section);
  });

  applySavedCollapse();
  injectResetButton();
}
