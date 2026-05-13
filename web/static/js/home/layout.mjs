// layout.mjs — drag & drop de paneles, collapse por panel/sección,
// botón "↺ layout" reset. Estado persistido en localStorage.

import {
  LS_HERO_COLLAPSED,
  LS_HERO_ORDER,
  LS_HERO_SUB_COLLAPSED,
} from "./panel-today.mjs";
import { observeAllPanels, observePanel, clearPanelSizeOverrides, setPanelSize } from "./autosizer.mjs";

// ── Constantes de localStorage ─────────────────────────────────────────────────

const LS_PANELS_ORDER = "home.v2.panels.order.v1";
const LS_PANELS_COLLAPSED = "home.v2.panels.collapsed.v1";
const LS_SECTIONS_COLLAPSED = "home.v2.sections.collapsed.v1";
const SECTION_BODY_IDS = ["sec-acc-body", "sec-mon-body", "sec-amb-body"];

// ── Orden persistido ───────────────────────────────────────────────────────────

function readSavedOrder() {
  try {
    const raw = localStorage.getItem(LS_PANELS_ORDER);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch { return null; }
}

function saveCurrentOrder() {
  const order = {};
  for (const secId of SECTION_BODY_IDS) {
    const sec = document.getElementById(secId);
    if (!sec) continue;
    order[secId] = Array.from(sec.querySelectorAll(":scope > .panel"))
      .map((p) => p.id)
      .filter(Boolean);
  }
  try {
    localStorage.setItem(LS_PANELS_ORDER, JSON.stringify(order));
  } catch (e) {
    console.warn("[home.v2] no pude persistir el orden de paneles:", e);
  }
  updateResetButtonVisibility();
}

export function applySavedOrder() {
  const saved = readSavedOrder();
  if (!saved) return;
  for (const secId of SECTION_BODY_IDS) {
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
  try { localStorage.removeItem(LS_PANELS_ORDER); } catch {}
  try { localStorage.removeItem(LS_PANELS_COLLAPSED); } catch {}
  try { localStorage.removeItem(LS_HERO_COLLAPSED); } catch {}
  try { localStorage.removeItem(LS_HERO_ORDER); } catch {}
  try { localStorage.removeItem(LS_HERO_SUB_COLLAPSED); } catch {}
  try { localStorage.removeItem(LS_SECTIONS_COLLAPSED); } catch {}
  clearPanelSizeOverrides();
  // Más simple recargar que reconstruir el orden hard-coded del HTML.
  window.location.reload();
}

// ── Collapse por panel ─────────────────────────────────────────────────────────

function readCollapsedMap() {
  try {
    const raw = localStorage.getItem(LS_PANELS_COLLAPSED);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : {};
  } catch { return {}; }
}

function saveCollapsedMap(map) {
  try {
    const trimmed = {};
    for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
    if (Object.keys(trimmed).length === 0) {
      localStorage.removeItem(LS_PANELS_COLLAPSED);
    } else {
      localStorage.setItem(LS_PANELS_COLLAPSED, JSON.stringify(trimmed));
    }
  } catch (e) {
    console.warn("[home.v2] no pude persistir collapse de paneles:", e);
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

// ── Chips de tamaño manual (override del autosizer) ───────────────────

function togglePanelDim(panel, dim) {
  const cur = panel.dataset[dim] || (dim === "w" ? "half" : "half");
  const next = cur === "half" ? "full" : "half";
  const w = dim === "w" ? next : (panel.dataset.w || "half");
  const h = dim === "h" ? next : (panel.dataset.h || "half");
  setPanelSize(panel.id, w, h);
  try { updateResetButtonVisibility(); } catch {}
}

function chipLabel(dim, panel) {
  // Etiqueta dinámica que refleja el estado actual + lo que el click va
  // a hacer. Ej: panel está en data-w=half → chip dice "↔ wide" (click
  // lo expande). data-w=full → "↔ small". Mismo para alto.
  const cur = panel.dataset[dim] || "half";
  if (dim === "w") return cur === "half" ? "↔ wide" : "↔ small";
  return cur === "half" ? "↕ tall" : "↕ short";
}

function refreshChipLabels(panel) {
  for (const dim of ["w", "h"]) {
    const chip = panel.querySelector(`.panel-size-chip-${dim}`);
    if (chip) chip.textContent = chipLabel(dim, panel);
  }
}

export function injectSizeChips(panel) {
  const head = panel.querySelector(".panel-head");
  if (!head || head.querySelector(".panel-size-chip")) return;
  // Insertar ANTES del botón collapse (que queda al final).
  const collapseBtn = head.querySelector(".panel-collapse-btn");
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

// ── Drag & drop de paneles ─────────────────────────────────────────────────────

let _draggingPanel = null;

function onPanelDragStart(ev) {
  // Si el drag se origina en un control interactivo del header (chips
  // de resize, botón collapse), cancelar — el browser propaga dragstart
  // desde el child al panel padre cuando el child no es draggable.
  if (ev.target && ev.target.closest(".panel-size-chip, .panel-collapse-btn")) {
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
  document.querySelectorAll(".panel.drop-before, .panel.drop-after")
    .forEach((p) => p.classList.remove("drop-before", "drop-after"));
  document.querySelectorAll(".section-body.drop-zone")
    .forEach((s) => s.classList.remove("drop-zone"));
}

function onPanelDragOver(ev) {
  if (!_draggingPanel) return;
  const panel = ev.currentTarget;
  if (panel === _draggingPanel) return;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  const rect = panel.getBoundingClientRect();
  const useX = rect.width > rect.height * 1.2;
  const before = useX
    ? (ev.clientX - rect.left) < rect.width / 2
    : (ev.clientY - rect.top) < rect.height / 2;
  panel.classList.toggle("drop-before", before);
  panel.classList.toggle("drop-after", !before);
}

function onPanelDragLeave(ev) {
  const panel = ev.currentTarget;
  if (panel.contains(ev.relatedTarget)) return;
  panel.classList.remove("drop-before", "drop-after");
}

function onPanelDrop(ev) {
  ev.preventDefault();
  if (!_draggingPanel) return;
  const target = ev.currentTarget;
  if (target === _draggingPanel) return;
  const before = target.classList.contains("drop-before");
  target.classList.remove("drop-before", "drop-after");
  if (before) {
    target.parentNode.insertBefore(_draggingPanel, target);
  } else {
    target.parentNode.insertBefore(_draggingPanel, target.nextSibling);
  }
  saveCurrentOrder();
}

export function makePanelDraggable(panel) {
  if (panel.dataset.draggableInit === "1") return;
  panel.dataset.draggableInit = "1";
  panel.setAttribute("draggable", "true");
  const head = panel.querySelector(".panel-head");
  if (head && !head.querySelector(".drag-grip")) {
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

export function makeSectionDroppable(secId) {
  const sec = document.getElementById(secId);
  if (!sec) return;
  sec.addEventListener("dragover", (ev) => {
    if (!_draggingPanel) return;
    if (ev.target !== sec) return;
    ev.preventDefault();
    try { ev.dataTransfer.dropEffect = "move"; } catch {}
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
    sec.appendChild(_draggingPanel);
    saveCurrentOrder();
  });
}

// ── Botón reset + visibilidad ──────────────────────────────────────────────────

function hasCustomLayout() {
  if (readSavedOrder()) return true;
  const map = readCollapsedMap();
  if (map && Object.keys(map).length > 0) return true;
  try {
    if (localStorage.getItem(LS_HERO_COLLAPSED) === "1") return true;
    if (localStorage.getItem(LS_HERO_ORDER)) return true;
    if (localStorage.getItem(LS_HERO_SUB_COLLAPSED)) return true;
    if (localStorage.getItem(LS_SECTIONS_COLLAPSED)) return true;
    if (localStorage.getItem("home.v2.panel-sizes.v1")) return true;
  } catch {}
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
  btn.title = "Resetear layout (orden + paneles colapsados)";
  btn.setAttribute("aria-label", "Resetear layout de los paneles");
  btn.textContent = "↺ layout";
  btn.hidden = !hasCustomLayout();
  btn.addEventListener("click", () => {
    if (confirm("¿Resetear el layout de los paneles al default? (orden + collapse)")) {
      clearSavedOrder();
    }
  });
  meta.appendChild(btn);
}

// ── Collapse por sección ───────────────────────────────────────────────────────

function readSectionsCollapsed() {
  try {
    const raw = localStorage.getItem(LS_SECTIONS_COLLAPSED);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : {};
  } catch { return {}; }
}

function writeSectionsCollapsed(map) {
  try {
    const trimmed = {};
    for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
    if (Object.keys(trimmed).length === 0) {
      localStorage.removeItem(LS_SECTIONS_COLLAPSED);
    } else {
      localStorage.setItem(LS_SECTIONS_COLLAPSED, JSON.stringify(trimmed));
    }
  } catch {}
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
    if (shouldCollapse) {
      section.setAttribute("data-collapsed", "true");
      btn.setAttribute("aria-expanded", "false");
    }
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const collapsed = section.getAttribute("data-collapsed") === "true";
      const next = !collapsed;
      section.setAttribute("data-collapsed", next ? "true" : "false");
      btn.setAttribute("aria-expanded", next ? "false" : "true");
      const map = readSectionsCollapsed();
      map[key] = next;
      writeSectionsCollapsed(map);
      try { updateResetButtonVisibility(); } catch {}
    });
  });
}

// ── Init principal del layout ──────────────────────────────────────────────────

export function initLayout() {
  // Exponer updateResetButtonVisibility globalmente para que panel-today.mjs la llame.
  window._updateResetButtonVisibility = updateResetButtonVisibility;
  // 1. Aplicar orden persistido ANTES de que los renderers escriban.
  applySavedOrder();
  // 2. Hacer cada panel draggable + insertar grip + chips de tamaño + botón collapse
  document.querySelectorAll(".section-body > .panel").forEach((panel) => {
    makePanelDraggable(panel);
    injectSizeChips(panel);
    injectCollapseButton(panel);
  });
  // 3. Aplicar estado de collapse persistido
  applySavedCollapse();
  // 4. Hacer cada section-body un drop zone para "soltar al final"
  SECTION_BODY_IDS.forEach(makeSectionDroppable);
  // 5. Inyectar botón reset en la topbar
  injectResetButton();
  // 6. Auto-sizing por content (4-col grid + 2 alturas).
  observeAllPanels();
  // 6b. Watcher para paneles hidden que aparecen tarde (p-mood, p-sleep,
  //     p-spotify, p-correlations, p-patterns) — se enganchan cuando el
  //     renderer correspondiente los unhide.
  SECTION_BODY_IDS.forEach((secId) => {
    const sec = document.getElementById(secId);
    if (!sec) return;
    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type !== "attributes" || m.attributeName !== "hidden") continue;
        const panel = m.target;
        if (panel?.classList?.contains("panel") && !panel.hidden) {
          injectSizeChips(panel);
          observePanel(panel);
        }
      }
    });
    sec.querySelectorAll(":scope > .panel").forEach((p) => {
      mo.observe(p, { attributes: true, attributeFilter: ["hidden"] });
    });
  });
}
