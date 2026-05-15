// finance-layout.mjs — drag & drop de secciones, collapse, resize
// Adaptado de home/layout.mjs para el dashboard de finanzas.

// ── Constantes de localStorage ─────────────────────────────────────────────────

const LS_SECTIONS_ORDER = "finance.sections.order.v1";
const LS_SECTIONS_COLLAPSED = "finance.sections.collapsed.v1";

// ── Orden persistido ───────────────────────────────────────────────────────────

function readSavedOrder() {
  try {
    const raw = localStorage.getItem(LS_SECTIONS_ORDER);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch { return null; }
}

function saveCurrentOrder() {
  const order = Array.from(document.querySelectorAll(".learn-section"))
    .map((s) => s.id)
    .filter(Boolean);
  try {
    localStorage.setItem(LS_SECTIONS_ORDER, JSON.stringify(order));
  } catch (e) {
    console.warn("[finance] no pude persistir el orden de secciones:", e);
  }
  updateResetButtonVisibility();
}

export function applySavedOrder() {
  const saved = readSavedOrder();
  if (!saved) return;
  const container = document.querySelector("main");
  if (!container) return;
  const ids = Array.isArray(saved) ? saved : [];
  for (const id of ids) {
    const section = document.getElementById(id);
    if (!section) continue;
    container.appendChild(section);
  }
}

function clearSavedOrder() {
  try { localStorage.removeItem(LS_SECTIONS_ORDER); } catch {}
  try { localStorage.removeItem(LS_SECTIONS_COLLAPSED); } catch {}
  window.location.reload();
}

// ── Collapse por sección ─────────────────────────────────────────────────────

function readCollapsedMap() {
  try {
    const raw = localStorage.getItem(LS_SECTIONS_COLLAPSED);
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
      localStorage.removeItem(LS_SECTIONS_COLLAPSED);
    } else {
      localStorage.setItem(LS_SECTIONS_COLLAPSED, JSON.stringify(trimmed));
    }
  } catch (e) {
    console.warn("[finance] no pude persistir collapse de secciones:", e);
  }
}

export function applySavedCollapse() {
  const map = readCollapsedMap();
  for (const sid of Object.keys(map)) {
    const section = document.getElementById(sid);
    if (!section) continue;
    const body = section.querySelector(".learn-section-body");
    const btn = section.querySelector(".collapse-btn");
    if (body) body.hidden = true;
    if (btn) {
      btn.setAttribute("aria-expanded", "false");
      btn.textContent = "+";
    }
  }
}

function toggleSectionCollapse(section) {
  const body = section.querySelector(".learn-section-body");
  const btn = section.querySelector(".collapse-btn");
  if (!body || !btn) return;

  const isCollapsed = body.hidden;
  body.hidden = !isCollapsed;
  btn.setAttribute("aria-expanded", isCollapsed ? "true" : "false");
  btn.textContent = isCollapsed ? "−" : "+";

  const map = readCollapsedMap();
  if (!isCollapsed) map[section.id] = true;
  else delete map[section.id];
  saveCollapsedMap(map);
  updateResetButtonVisibility();
}

// ── Drag & drop de secciones ─────────────────────────────────────────────────────

let _draggingSection = null;

function onSectionDragStart(ev) {
  const section = ev.currentTarget;
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
  const useX = rect.width > rect.height * 1.2;
  const before = useX
    ? (ev.clientX - rect.left) < rect.width / 2
    : (ev.clientY - rect.top) < rect.height / 2;
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

// ── Botón reset + visibilidad ──────────────────────────────────────────────────

function hasCustomLayout() {
  if (readSavedOrder()) return true;
  const map = readCollapsedMap();
  if (map && Object.keys(map).length > 0) return true;
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
  btn.title = "Resetear layout (orden + secciones colapsadas)";
  btn.setAttribute("aria-label", "Resetear layout de las secciones");
  btn.textContent = "↺ layout";
  btn.hidden = !hasCustomLayout();
  btn.addEventListener("click", () => {
    if (confirm("¿Resetear el layout de las secciones al default? (orden + collapse)")) {
      clearSavedOrder();
    }
  });
  header.appendChild(btn);
}

// ── Init principal del layout ──────────────────────────────────────────────────

export function initFinanceLayout() {
  // 1. Aplicar orden persistido ANTES de que los renderers escriban.
  applySavedOrder();
  // 2. Hacer cada sección draggable + insertar grip
  document.querySelectorAll(".learn-section").forEach((section) => {
    makeSectionDraggable(section);
  });
  // 3. Aplicar estado de collapse persistido
  applySavedCollapse();
  // 4. Inyectar botón reset en el header
  injectResetButton();
}
