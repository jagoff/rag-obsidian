// layout.mjs — drag & drop de paneles, collapse por panel/sección,
// botón "↺ layout" reset. Estado persistido en localStorage.

import {
  LS_HERO_COLLAPSED,
  LS_HERO_ORDER,
  LS_HERO_SUB_COLLAPSED,
} from "./panel-today.mjs?v=103";
import {
  LS_PANEL_SIZES,
  applySavedPanelSize,
  clearPanelSizeOverrides,
  hasPanelSizeOverrides,
  observeAllPanels,
  observePanel,
  setPanelSize,
} from "./autosizer.mjs?v=103";
import {
  clearServerLayout,
  compactBooleanMap,
  hydrateServerLayout,
  readObject,
  readString,
  removeKeys,
  writeJSON,
  writeObjectOrRemove,
} from "../layout-persistence.mjs?v=103";

// ── Constantes de localStorage ─────────────────────────────────────────────────

const LS_PANELS_ORDER = "home.v2.panels.order.v1";
const LS_PANELS_COLLAPSED = "home.v2.panels.collapsed.v1";
const LS_SECTIONS_COLLAPSED = "home.v2.sections.collapsed.v1";
const HERO_BODY_ID = "today-hero-body";
const SECTION_BODY_IDS = ["sec-acc-body", "sec-mon-body", "sec-amb-body"];
const ORDER_CONTAINER_IDS = [HERO_BODY_ID, "home-cmdbar", ...SECTION_BODY_IDS];
const ORDER_ITEM_SELECTOR = ":scope > .panel, :scope > .kpi, :scope > .hero-section";
const SERVER_LAYOUT_PAGE = "home.v2";
const SERVER_LAYOUT_KEYS = [
  LS_PANELS_ORDER,
  LS_PANELS_COLLAPSED,
  LS_SECTIONS_COLLAPSED,
  LS_PANEL_SIZES,
  LS_HERO_COLLAPSED,
  LS_HERO_ORDER,
  LS_HERO_SUB_COLLAPSED,
];

// ── Orden persistido ───────────────────────────────────────────────────────────

function readSavedOrder() {
  return readObject(LS_PANELS_ORDER, null);
}

function saveCurrentOrder() {
  const order = {};
  for (const secId of ORDER_CONTAINER_IDS) {
    const sec = document.getElementById(secId);
    if (!sec) continue;
    order[secId] = Array.from(sec.querySelectorAll(ORDER_ITEM_SELECTOR))
      .map((p) => p.id)
      .filter(Boolean);
  }
  if (!writeJSON(LS_PANELS_ORDER, order)) {
    console.warn("[home.v2] no pude persistir el orden de paneles");
  }
  updateResetButtonVisibility();
}

function layoutContainers() {
  return ORDER_CONTAINER_IDS
    .map((id) => document.getElementById(id))
    .filter(Boolean);
}

function queryLayoutItems(selector) {
  return layoutContainers().flatMap((container) => (
    Array.from(container.querySelectorAll(`:scope > ${selector}`))
  ));
}

function isLayoutItem(node) {
  return !!(
    node
    && node.nodeType === 1
    && (
      node.classList?.contains("panel")
      || node.classList?.contains("kpi")
      || node.classList?.contains("hero-section")
    )
  );
}

export function applySavedOrder() {
  const saved = readSavedOrder();
  if (!saved) return;
  for (const secId of ORDER_CONTAINER_IDS) {
    const sec = document.getElementById(secId);
    if (!sec) continue;
    const ids = Array.isArray(saved[secId]) ? saved[secId] : [];
    for (const pid of ids) {
      const panel = document.getElementById(pid);
      if (!panel) continue;
      sec.appendChild(panel);
    }
  }
}

function clearSavedOrder() {
  clearServerLayout(SERVER_LAYOUT_PAGE);
  removeKeys([
    LS_PANELS_ORDER,
    LS_PANELS_COLLAPSED,
    LS_HERO_COLLAPSED,
    LS_HERO_ORDER,
    LS_HERO_SUB_COLLAPSED,
    LS_SECTIONS_COLLAPSED,
  ]);
  clearPanelSizeOverrides();
  // Más simple recargar que reconstruir el orden hard-coded del HTML.
  window.setTimeout(() => window.location.reload(), 150);
}

// ── Collapse por panel ─────────────────────────────────────────────────────────

function readCollapsedMap() {
  return readObject(LS_PANELS_COLLAPSED, {});
}

function saveCollapsedMap(map) {
  const trimmed = compactBooleanMap(map);
  if (!writeObjectOrRemove(LS_PANELS_COLLAPSED, trimmed)) {
    console.warn("[home.v2] no pude persistir collapse de paneles");
  }
}

export function applySavedCollapse() {
  const map = readCollapsedMap();
  for (const pid of Object.keys(map)) {
    const panel = document.getElementById(pid);
    if (!panel) continue;
    panel.setAttribute("data-collapsed", "true");
    const btn = panel.querySelector(".panel-collapse-btn");
    if (btn) {
      btn.setAttribute("aria-expanded", "false");
      const icon = btn.querySelector(".toggle-icon");
      if (icon) icon.textContent = "▶";
    }
  }
}

function togglePanelCollapse(panel) {
  const collapsed = panel.getAttribute("data-collapsed") === "true";
  const next = !collapsed;
  panel.setAttribute("data-collapsed", next ? "true" : "false");
  const btn = panel.querySelector(".panel-collapse-btn");
  if (btn) {
    btn.setAttribute("aria-expanded", next ? "false" : "true");
    const icon = btn.querySelector(".toggle-icon");
    if (icon) icon.textContent = next ? "▶" : "▼";
  }
  const map = readCollapsedMap();
  if (next) map[panel.id] = true;
  else delete map[panel.id];
  saveCollapsedMap(map);
  updateResetButtonVisibility();
}

export function injectCollapseButton(panel) {
  const head = panel.querySelector(".panel-head");
  if (!head || head.querySelector(".panel-collapse-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "panel-collapse-btn";
  btn.setAttribute("aria-label", "Colapsar/expandir panel");
  btn.setAttribute("aria-expanded", "true");
  btn.title = "Colapsar/expandir";
  btn.innerHTML = '<span class="toggle-icon" aria-hidden="true">▼</span>';
  btn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    togglePanelCollapse(panel);
  });
  // Evitar que el click inicie un drag (el panel padre tiene draggable=true)
  btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
  head.appendChild(btn);
}

function injectKpiControls(kpi) {
  if (kpi.querySelector(":scope > .kpi-layout-controls")) {
    kpi.dataset.kpiControlsInit = "1";
    return;
  }
  kpi.dataset.kpiControlsInit = "1";
  const controls = document.createElement("div");
  controls.className = "kpi-layout-controls";

  const grip = document.createElement("span");
  grip.className = "drag-grip kpi-drag-grip";
  grip.setAttribute("aria-hidden", "true");
  grip.title = "arrastrá para reordenar";
  grip.textContent = "⋮⋮";
  controls.appendChild(grip);

  for (const dim of ["w", "h"]) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `kpi-size-btn kpi-size-${dim}`;
    btn.setAttribute("aria-label", dim === "w" ? "Toggle ancho" : "Toggle alto");
    btn.title = dim === "w" ? "ancho: compacto ↔ ancho completo" : "alto: corto ↕ alto";
    btn.textContent = dim === "w" ? "↔" : "↕";
    btn.setAttribute("draggable", "false");
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      togglePanelDim(kpi, dim);
    });
    btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
    controls.appendChild(btn);
  }

  const collapse = document.createElement("button");
  collapse.type = "button";
  collapse.className = "panel-collapse-btn kpi-collapse-btn";
  collapse.setAttribute("aria-label", "Colapsar/expandir indicador");
  collapse.setAttribute("aria-expanded", "true");
  collapse.title = "Minimizar/expandir";
  collapse.innerHTML = '<span class="toggle-icon" aria-hidden="true">▼</span>';
  collapse.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    togglePanelCollapse(kpi);
  });
  collapse.addEventListener("mousedown", (ev) => ev.stopPropagation());
  controls.appendChild(collapse);

  kpi.appendChild(controls);
}

// ── Chips de tamaño manual (override del autosizer) ───────────────────

function togglePanelDim(panel, dim) {
  if (dim === "w") {
    // ancho: half ↔ full (2 estados, sin cambios).
    const cur = panel.dataset.w || "half";
    const next = cur === "half" ? "full" : "half";
    setPanelSize(panel.id, next, panel.dataset.h || "half");
  } else {
    // alto: half → full → xl → half (3 estados, regla 2026-05-13 ask
    // user "3 medidas de largo"). xl = grid-row span 3.
    const cur = panel.dataset.h || "half";
    const next = cur === "half" ? "full" : cur === "full" ? "xl" : "half";
    setPanelSize(panel.id, panel.dataset.w || "half", next);
  }
  try { updateResetButtonVisibility(); } catch {}
}

function chipLabel(dim, panel) {
  // Etiqueta dinámica que refleja lo que el click va a hacer next.
  // Ancho: half → "↔ wide" → full → "↔ small" → half.
  // Alto:  half → "↕ tall" → full → "↕ taller" → xl → "↕ short" → half.
  if (dim === "w") {
    const cur = panel.dataset.w || "half";
    return cur === "half" ? "↔ wide" : "↔ small";
  }
  const cur = panel.dataset.h || "half";
  if (cur === "half") return "↕ tall";
  if (cur === "full") return "↕ taller";
  return "↕ short";
}

function refreshChipLabels(panel) {
  for (const dim of ["w", "h"]) {
    const chip = panel.querySelector(`.panel-size-chip-${dim}`);
    if (chip) chip.textContent = chipLabel(dim, panel);
  }
}

export function injectSizeChips(panel) {
  const head = panel.querySelector(".panel-head") || panel.querySelector(":scope > h3");
  if (!head || head.querySelector(".panel-size-chip")) return;
  // Insertar ANTES del botón collapse (que queda al final).
  const collapseBtn = head.querySelector(".panel-collapse-btn, .hero-collapse-btn");
  const ref = collapseBtn || null;
  for (const dim of ["w", "h"]) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `panel-size-chip panel-size-chip-${dim}`;
    chip.dataset.dim = dim;
    chip.setAttribute("aria-label", dim === "w" ? "Toggle ancho" : "Toggle alto");
    chip.title = dim === "w" ? "ancho: half ↔ full" : "alto: half ↕ full";
    chip.textContent = chipLabel(dim, panel);
    // Prevenir interferencia del drag handler del panel padre. Sin esto
    // el browser inicia drag sobre el botón antes de disparar click.
    chip.setAttribute("draggable", "false");
    chip.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      togglePanelDim(panel, dim);
      refreshChipLabels(panel);
    });
    chip.addEventListener("mousedown", (ev) => ev.stopPropagation());
    chip.addEventListener("dragstart", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
    });
    if (ref) head.insertBefore(chip, ref);
    else head.appendChild(chip);
  }
  // Observar cambios de data-w / data-h (autosizer puede cambiarlos) para
  // mantener las etiquetas sincronizadas.
  const mo = new MutationObserver(() => refreshChipLabels(panel));
  mo.observe(panel, { attributes: true, attributeFilter: ["data-w", "data-h"] });
}

// ── Resize handles (drag bordes → snap a half/full) ───────────────────

const RESIZE_SNAP_PX = 100; // delta mínimo del drag para cambiar snap.

function startResize(ev, panel, axis) {
  ev.preventDefault();
  ev.stopPropagation();
  const startX = ev.clientX;
  const startY = ev.clientY;
  const startW = panel.dataset.w || "half";
  const startH = panel.dataset.h || "half";
  panel.classList.add("is-resizing");
  document.body.style.cursor = axis === "x" ? "ew-resize"
    : axis === "y" ? "ns-resize" : "nwse-resize";

  function onMove(e) {
    // Ghost overlay opcional — por ahora solo mostramos el cursor.
    // Preview en vivo: actualizamos data-w/data-h cuando el delta cruza
    // el umbral, así el user ve el panel resize antes de soltar.
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (axis === "x" || axis === "xy") {
      const w = dx > RESIZE_SNAP_PX ? "full"
        : dx < -RESIZE_SNAP_PX ? "half" : startW;
      if (panel.dataset.w !== w) panel.dataset.w = w;
    }
    if (axis === "y" || axis === "xy") {
      // 3 medidas de largo (regla 2026-05-13): half / full / xl.
      // dy > 2*SNAP → xl (span 3), SNAP < dy ≤ 2*SNAP → full (span 2),
      // dy < -SNAP → half (span 1). Mismas distancias para shrink.
      const h = dy > RESIZE_SNAP_PX * 2 ? "xl"
        : dy > RESIZE_SNAP_PX ? "full"
        : dy < -RESIZE_SNAP_PX ? "half"
        : startH;
      if (panel.dataset.h !== h) panel.dataset.h = h;
    }
  }

  function onUp() {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    panel.classList.remove("is-resizing");
    document.body.style.cursor = "";
    // Persistir el final state como override manual.
    setPanelSize(panel.id, panel.dataset.w || "half", panel.dataset.h || "half");
    try { updateResetButtonVisibility(); } catch {}
  }

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

function setupPanel(panel) {
  applySavedPanelSize(panel);
  makePanelDraggable(panel);
  injectSizeChips(panel);
  injectResizeHandles(panel);
  injectCollapseButton(panel);
  observePanel(panel);
}

function allPanels() {
  return queryLayoutItems(".panel");
}

export function injectResizeHandles(panel) {
  if (panel.dataset.resizeInit === "1") return;
  panel.dataset.resizeInit = "1";
  for (const axis of ["x", "y", "xy"]) {
    const h = document.createElement("div");
    h.className = `panel-resize-handle panel-resize-${axis}`;
    h.dataset.axis = axis;
    h.title = axis === "x" ? "arrastrá horizontal: half ↔ full"
      : axis === "y" ? "arrastrá vertical: half ↕ full"
      : "arrastrá: half/full ancho + alto";
    h.addEventListener("mousedown", (ev) => startResize(ev, panel, axis));
    // Evitar que el handle inicie HTML5 drag del panel padre.
    h.addEventListener("dragstart", (ev) => ev.preventDefault());
    panel.appendChild(h);
  }
}

// ── Drag & drop de paneles ─────────────────────────────────────────────────────

let _draggingPanel = null;

function disableNativeChildDrag(root) {
  root.querySelectorAll("a, img").forEach((el) => {
    el.setAttribute("draggable", "false");
  });
}

function _dropPlacement(ev, el) {
  const rect = el.getBoundingClientRect();
  const x = rect.width ? (ev.clientX - rect.left) / rect.width : 0.5;
  const y = rect.height ? (ev.clientY - rect.top) / rect.height : 0.5;
  if (y < 0.33) return { before: true, axis: "y" };
  if (y > 0.67) return { before: false, axis: "y" };
  return { before: x < 0.5, axis: "x" };
}

function _setDropMark(el, placement) {
  const before = !!placement?.before;
  const axis = placement?.axis === "y" ? "y" : "x";
  el.classList.toggle("drop-before", before);
  el.classList.toggle("drop-after", !before);
  el.classList.toggle("drop-axis-x", axis === "x");
  el.classList.toggle("drop-axis-y", axis === "y");
}

function _clearDropMark(el) {
  el.classList.remove("drop-before", "drop-after", "drop-axis-x", "drop-axis-y");
}

function _allDropMarked() {
  return document.querySelectorAll(
    ".panel.drop-before, .panel.drop-after, .kpi.drop-before, .kpi.drop-after, .hero-section.drop-before, .hero-section.drop-after",
  );
}

function _orderItems(container) {
  return Array.from(container?.querySelectorAll(ORDER_ITEM_SELECTOR) || [])
    .filter((el) => el !== _draggingPanel);
}

function _containerInsertionFromPoint(container, ev) {
  const items = _orderItems(container)
    .map((el) => ({ el, rect: el.getBoundingClientRect() }))
    .sort((a, b) => {
      const dy = a.rect.top - b.rect.top;
      if (Math.abs(dy) > 8) return dy;
      return a.rect.left - b.rect.left;
    });
  if (!items.length) return { target: null, before: false, axis: "y" };
  const rows = [];
  for (const item of items) {
    let row = rows.find((r) => Math.abs(r.top - item.rect.top) <= 12);
    if (!row) {
      row = { top: item.rect.top, bottom: item.rect.bottom, items: [] };
      rows.push(row);
    }
    row.top = Math.min(row.top, item.rect.top);
    row.bottom = Math.max(row.bottom, item.rect.bottom);
    row.items.push(item);
  }
  for (const row of rows) {
    row.items.sort((a, b) => a.rect.left - b.rect.left);
    if (ev.clientY < row.top) {
      return { target: row.items[0].el, before: true, axis: "y" };
    }
    if (ev.clientY <= row.bottom) {
      for (const item of row.items) {
        const midX = item.rect.left + item.rect.width / 2;
        if (ev.clientX < midX) return { target: item.el, before: true, axis: "x" };
      }
      return { target: row.items[row.items.length - 1].el, before: false, axis: "x" };
    }
  }
  return { target: items[items.length - 1].el, before: false, axis: "y" };
}

function _insertDraggingAt(container, placement) {
  if (!_draggingPanel || !container) return false;
  if (!placement?.target) {
    container.appendChild(_draggingPanel);
    return true;
  }
  if (placement.target === _draggingPanel) return false;
  if (placement.before) {
    container.insertBefore(_draggingPanel, placement.target);
  } else {
    container.insertBefore(_draggingPanel, placement.target.nextSibling);
  }
  return true;
}

function onPanelDragStart(ev) {
  // Si el drag se origina en un control interactivo del header (chips
  // de resize, botón collapse), cancelar — el browser propaga dragstart
  // desde el child al panel padre cuando el child no es draggable.
  if (ev.target && ev.target.closest(
    ".panel-size-chip, .panel-collapse-btn, .hero-collapse-btn, .kpi-size-btn, .panel-resize-handle",
  )) {
    ev.preventDefault();
    return;
  }
  const panel = ev.currentTarget;
  _draggingPanel = panel;
  panel.classList.add("is-dragging");
  try {
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/plain", panel.id);
  } catch {}
}

function onPanelDragEnd(ev) {
  const panel = ev.currentTarget;
  panel.classList.remove("is-dragging");
  _draggingPanel = null;
  _allDropMarked().forEach(_clearDropMark);
  document.querySelectorAll(".section-body.drop-zone, .cmdbar.drop-zone, .today-hero-body.drop-zone")
    .forEach((s) => s.classList.remove("drop-zone"));
}

function onPanelDragOver(ev) {
  if (!_draggingPanel) return;
  const panel = ev.currentTarget;
  if (panel === _draggingPanel) return;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  _setDropMark(panel, _dropPlacement(ev, panel));
}

function onPanelDragLeave(ev) {
  const panel = ev.currentTarget;
  if (panel.contains(ev.relatedTarget)) return;
  _clearDropMark(panel);
}

function onPanelDrop(ev) {
  ev.preventDefault();
  if (!_draggingPanel) return;
  const target = ev.currentTarget;
  if (target === _draggingPanel) return;
  const placement = _dropPlacement(ev, target);
  const before = placement.before;
  _clearDropMark(target);
  if (before) {
    target.parentNode.insertBefore(_draggingPanel, target);
  } else {
    target.parentNode.insertBefore(_draggingPanel, target.nextSibling);
  }
  saveCurrentOrder();
}

export function makePanelDraggable(panel) {
  disableNativeChildDrag(panel);
  if (panel.dataset.draggableInit === "1") return;
  panel.dataset.draggableInit = "1";
  panel.setAttribute("draggable", "true");
  const head = panel.querySelector(".panel-head") || panel.querySelector(":scope > h3");
  if (head && !head.querySelector(".drag-grip, .hero-drag-grip")) {
    const grip = document.createElement("span");
    grip.className = "drag-grip";
    grip.setAttribute("aria-hidden", "true");
    grip.title = "arrastrá para reordenar";
    grip.textContent = "⋮⋮";
    head.insertBefore(grip, head.firstChild);
  }
  panel.addEventListener("dragstart", onPanelDragStart);
  panel.addEventListener("dragend", onPanelDragEnd);
  panel.addEventListener("dragover", onPanelDragOver);
  panel.addEventListener("dragleave", onPanelDragLeave);
  panel.addEventListener("drop", onPanelDrop);
}

function setupKpi(kpi) {
  applySavedPanelSize(kpi);
  disableNativeChildDrag(kpi);
  injectKpiControls(kpi);
  injectResizeHandles(kpi);
  makePanelDraggable(kpi);
}

function allKpis() {
  return queryLayoutItems(".kpi");
}

function allHeroSections() {
  return queryLayoutItems(".hero-section");
}

function setupHeroSection(sec) {
  applySavedPanelSize(sec);
  makePanelDraggable(sec);
  injectSizeChips(sec);
  injectResizeHandles(sec);
}

function setupLayoutItem(item) {
  if (!isLayoutItem(item)) return;
  if (item.classList.contains("panel")) {
    setupPanel(item);
  } else if (item.classList.contains("kpi")) {
    setupKpi(item);
  } else if (item.classList.contains("hero-section")) {
    setupHeroSection(item);
  }
}

export function refreshLayoutControls() {
  applySavedOrder();
  allPanels().forEach(setupPanel);
  allKpis().forEach(setupKpi);
  allHeroSections().forEach(setupHeroSection);
  observeAllPanels();
}

export function makeSectionDroppable(secId) {
  const sec = document.getElementById(secId);
  if (!sec) return;
  sec.addEventListener("dragover", (ev) => {
    if (!_draggingPanel) return;
    if (ev.target !== sec) return;
    ev.preventDefault();
    try { ev.dataTransfer.dropEffect = "move"; } catch {}
    _allDropMarked().forEach(_clearDropMark);
    const placement = _containerInsertionFromPoint(sec, ev);
    if (placement.target) _setDropMark(placement.target, placement);
    sec.classList.add("drop-zone");
  });
  sec.addEventListener("dragleave", (ev) => {
    if (ev.target !== sec) return;
    sec.classList.remove("drop-zone");
  });
  sec.addEventListener("drop", (ev) => {
    if (!_draggingPanel) return;
    if (ev.target !== sec) return;
    ev.preventDefault();
    sec.classList.remove("drop-zone");
    const placement = _containerInsertionFromPoint(sec, ev);
    _allDropMarked().forEach(_clearDropMark);
    if (_insertDraggingAt(sec, placement)) saveCurrentOrder();
  });
}

function observeLayoutContainer(sec) {
  if (!sec || sec.dataset.layoutObserverInit === "1") return;
  sec.dataset.layoutObserverInit = "1";
  const observeItem = (item) => {
    if (!isLayoutItem(item)) return;
    setupLayoutItem(item);
    mo.observe(item, { attributes: true, attributeFilter: ["hidden"] });
  };
  const mo = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "childList") {
        for (const n of m.addedNodes) observeItem(n);
        continue;
      }
      if (m.type !== "attributes" || m.attributeName !== "hidden") continue;
      const item = m.target;
      if (isLayoutItem(item) && !item.hidden) setupLayoutItem(item);
    }
  });
  Array.from(sec.children).forEach(observeItem);
  mo.observe(sec, { childList: true });
}

// ── Botón reset + visibilidad ──────────────────────────────────────────────────

function hasCustomLayout() {
  if (readSavedOrder()) return true;
  const map = readCollapsedMap();
  if (map && Object.keys(map).length > 0) return true;
  if (readString(LS_HERO_COLLAPSED) === "1") return true;
  if (readString(LS_HERO_ORDER)) return true;
  if (readString(LS_HERO_SUB_COLLAPSED)) return true;
  if (readString(LS_SECTIONS_COLLAPSED)) return true;
  if (hasPanelSizeOverrides()) return true;
  return false;
}

export function updateResetButtonVisibility() {
  const btn = document.getElementById("reset-order-btn");
  if (!btn) return;
  btn.hidden = !hasCustomLayout();
}

export function injectResetButton() {
  const meta = document.getElementById("topbar-meta");
  if (!meta || document.getElementById("reset-order-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "reset-order-btn";
  btn.className = "reset-order-btn";
  btn.title = "Resetear layout (orden + tamaños + paneles colapsados)";
  btn.setAttribute("aria-label", "Resetear layout de los paneles");
  btn.textContent = "↺ layout";
  btn.hidden = !hasCustomLayout();
  btn.addEventListener("click", () => {
    if (confirm("¿Resetear el layout de los paneles al default? (orden + tamaños + collapse)")) {
      clearSavedOrder();
    }
  });
  meta.appendChild(btn);
}

// ── Collapse por sección ───────────────────────────────────────────────────────

function readSectionsCollapsed() {
  return readObject(LS_SECTIONS_COLLAPSED, {});
}

function compactSectionCollapseMap(map) {
  const trimmed = {};
  if (!map || typeof map !== "object") return trimmed;
  for (const [key, value] of Object.entries(map)) {
    if (key && typeof value === "boolean") trimmed[key] = value;
  }
  return trimmed;
}

function writeSectionsCollapsed(map) {
  writeObjectOrRemove(LS_SECTIONS_COLLAPSED, compactSectionCollapseMap(map));
}

function setSectionCollapsed(section, btn, collapsed) {
  section.setAttribute("data-collapsed", collapsed ? "true" : "false");
  btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  const icon = btn.querySelector(".toggle-icon");
  if (icon) icon.textContent = collapsed ? "▶" : "▼";
}

export function initCollapsibleSections() {
  const isMobile = window.matchMedia("(max-width: 720px)").matches;
  const saved = readSectionsCollapsed();
  document.querySelectorAll(".section-toggle").forEach((btn) => {
    const section = btn.closest(".section");
    if (!section) return;
    const key = Array.from(section.classList).find(
      (c) => c.startsWith("section-") && c !== "section",
    );
    if (!key) return;
    let shouldCollapse;
    if (Object.prototype.hasOwnProperty.call(saved, key)) {
      shouldCollapse = !!saved[key];
    } else {
      shouldCollapse = isMobile && (
        key === "section-monitoring" || key === "section-ambient"
      );
    }
    setSectionCollapsed(section, btn, shouldCollapse);
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const collapsed = section.getAttribute("data-collapsed") === "true";
      const next = !collapsed;
      setSectionCollapsed(section, btn, next);
      const map = readSectionsCollapsed();
      map[key] = next;
      writeSectionsCollapsed(map);
      try { updateResetButtonVisibility(); } catch {}
    });
  });
}

// ── Init principal del layout ──────────────────────────────────────────────────

export async function initLayout() {
  // Exponer updateResetButtonVisibility globalmente para que panel-today.mjs la llame.
  window._updateResetButtonVisibility = updateResetButtonVisibility;
  // 0. Hidratar desde SQLite del servidor antes de leer localStorage.
  await hydrateServerLayout(SERVER_LAYOUT_PAGE, SERVER_LAYOUT_KEYS);
  // 1. Aplicar orden persistido ANTES de que los renderers escriban.
  applySavedOrder();
  // 2. Hacer cada panel draggable + insertar grip + chips + handles + collapse
  allPanels().forEach((panel) => {
    setupPanel(panel);
  });
  allKpis().forEach((kpi) => {
    setupKpi(kpi);
  });
  // 2b. Mismo tratamiento para las .hero-section (LO QUE PASÓ / SIN PROCESAR
  //     / PREGUNTAS / AGENDA). El hero usa el mismo drag/resize/persistencia
  //     que paneles y KPIs.
  const heroBody = document.getElementById("today-hero-body");
  if (heroBody) {
    heroBody.querySelectorAll(":scope > .hero-section").forEach(setupHeroSection);
    // El hero-body se re-renderiza cuando llega el brief — watch nuevas
    // hero-sections para aplicar el mismo setup.
    const heroMo = new MutationObserver((muts) => {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n.nodeType === 1 && n.classList?.contains("hero-section")) {
            setupHeroSection(n);
          }
        }
      }
    });
    heroMo.observe(heroBody, { childList: true });
  }
  // 3. Aplicar estado de collapse persistido
  applySavedCollapse();
  initCollapsibleSections();
  // 4. Hacer cada section-body/cmdbar un drop zone para "soltar al final"
  ORDER_CONTAINER_IDS.forEach(makeSectionDroppable);
  // 5. Inyectar botón reset en la topbar
  injectResetButton();
  // 6. Auto-sizing por content (4-col grid + 2 alturas).
  observeAllPanels();
  // 6b. Watcher para cualquier caja que aparezca tarde o quede movida por
  //     layout persistido: panel, KPI y hero-section comparten wiring.
  ORDER_CONTAINER_IDS.forEach((secId) => {
    observeLayoutContainer(document.getElementById(secId));
  });
}
