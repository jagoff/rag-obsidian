// layout.mjs — layout libre con Gridstack.js. Cada panel ocupa un slot
// del grid 12-col que el user puede arrastrar y redimensionar libremente.
// Estado (x, y, w, h por panel ID) persistido en localStorage. Mobile usa
// 1 columna auto-stack (Gridstack lo hace solo).
//
// Sustituye al sistema anterior basado en HTML5 drag-drop. Mantiene los
// nombres exportados (initLayout, updateResetButtonVisibility) por
// compatibilidad con app.mjs y panel-today.mjs.

import {
  LS_HERO_COLLAPSED,
  LS_HERO_ORDER,
  LS_HERO_SUB_COLLAPSED,
} from "./panel-today.mjs";

// ── localStorage keys ─────────────────────────────────────────────────────────

const LS_GRID_LAYOUT = "home.v2.gridstack.v1";
const LS_GRID_EDIT_MODE = "home.v2.gridstack.edit.v1";
const LS_GRID_HINT_SEEN = "home.v2.gridstack.hint.v1";
const LS_PANELS_COLLAPSED = "home.v2.panels.collapsed.v1";
const LS_SECTIONS_COLLAPSED = "home.v2.sections.collapsed.v1";

// ── Default layout (12-col grid) ──────────────────────────────────────────────
// Si querés cambiar el orden inicial al primer load, editá este map.
// Cuando el user arrastra, el override se persiste en LS_GRID_LAYOUT.

const DEFAULT_LAYOUT = {
  // Hero header (title + ↻ refresh + progress bar). El brief se divide en
  // 4 sub-cajas (hero-narrative / hero-inbox / hero-questions / hero-agenda)
  // que son grid-items independientes y movibles.
  "today-hero":       { x: 0,  y: 0,  w: 12, h: 2 },
  "hero-narrative":   { x: 0,  y: 2,  w: 6,  h: 6 },
  "hero-inbox":       { x: 6,  y: 2,  w: 3,  h: 6 },
  "hero-questions":   { x: 9,  y: 2,  w: 3,  h: 6 },
  "hero-agenda":      { x: 0,  y: 8,  w: 12, h: 4 },
  // KPI bar — full-width strip
  "cmdbar":           { x: 0,  y: 12, w: 12, h: 2 },
  // Accionable — tier 1, paneles más anchos (4-col)
  "p-patterns":       { x: 0,  y: 14, w: 12, h: 3 },
  "p-inbox":          { x: 0,  y: 17, w: 4,  h: 4 },
  "p-questions":      { x: 4,  y: 17, w: 4,  h: 4 },
  "p-tomorrow":       { x: 8,  y: 17, w: 4,  h: 4 },
  "p-wa-unreplied":   { x: 0,  y: 21, w: 4,  h: 4 },
  "p-loops-urgent":   { x: 4,  y: 21, w: 4,  h: 4 },
  "p-contradictions": { x: 8,  y: 21, w: 4,  h: 4 },
  // Monitoring — tier 2
  "p-finance":        { x: 0,  y: 25, w: 4,  h: 4 },
  "p-sleep":          { x: 4,  y: 25, w: 4,  h: 4 },
  "p-mood":           { x: 8,  y: 25, w: 4,  h: 4 },
  "p-correlations":   { x: 0,  y: 29, w: 4,  h: 4 },
  "p-cards":          { x: 4,  y: 29, w: 4,  h: 4 },
  "p-retrieval":      { x: 8,  y: 29, w: 4,  h: 4 },
  "p-loops-aging":    { x: 0,  y: 33, w: 4,  h: 4 },
  "p-authority":      { x: 4,  y: 33, w: 4,  h: 4 },
  "p-eval-trend":     { x: 8,  y: 33, w: 4,  h: 4 },
  // Ambient — tier 3, más chicos (3-col)
  "p-weather":        { x: 0,  y: 37, w: 3,  h: 3 },
  "p-vault-activity": { x: 3,  y: 37, w: 3,  h: 3 },
  "p-captured":       { x: 6,  y: 37, w: 3,  h: 3 },
  "p-web":            { x: 9,  y: 37, w: 3,  h: 3 },
  "p-bookmarks":      { x: 0,  y: 40, w: 3,  h: 3 },
  "p-youtube":        { x: 3,  y: 40, w: 3,  h: 3 },
  "p-drive":          { x: 6,  y: 40, w: 3,  h: 3 },
  "p-spotify":        { x: 9,  y: 40, w: 3,  h: 3 },
};

// Map de hero-section CSS class → grid-stack-item gs-id.
const HERO_SUB_MAP = {
  "s-narrative": "hero-narrative",
  "s-inbox":     "hero-inbox",
  "s-questions": "hero-questions",
  "s-agenda":    "hero-agenda",
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function readSavedLayout() {
  try {
    const raw = localStorage.getItem(LS_GRID_LAYOUT);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const map = {};
    for (const item of parsed) {
      if (item && typeof item.id === "string") {
        map[item.id] = {
          x: Number(item.x) || 0,
          y: Number(item.y) || 0,
          w: Number(item.w) || 4,
          h: Number(item.h) || 3,
        };
      }
    }
    return map;
  } catch {
    return null;
  }
}

function layoutFor(panelId, savedMap) {
  if (savedMap && savedMap[panelId]) return savedMap[panelId];
  if (DEFAULT_LAYOUT[panelId]) return DEFAULT_LAYOUT[panelId];
  return { w: 4, h: 3, autoPosition: true };
}

// ── Wrap panels en grid-stack-item + insertar en grid container ──────────────

function wrapElementAsGridItem(grid, el, gsId, savedMap, opts = {}) {
  if (!el) return;
  // Si ya está envuelto, skip (idempotente).
  if (el.parentElement?.classList.contains("grid-stack-item-content")) return;
  const layout = layoutFor(gsId, savedMap);

  const wrap = document.createElement("div");
  wrap.className = "grid-stack-item";
  wrap.setAttribute("gs-id", gsId);
  if (layout.x !== undefined) wrap.setAttribute("gs-x", String(layout.x));
  if (layout.y !== undefined) wrap.setAttribute("gs-y", String(layout.y));
  wrap.setAttribute("gs-w", String(layout.w || 4));
  wrap.setAttribute("gs-h", String(layout.h || 3));
  wrap.setAttribute("gs-min-w", String(opts.minW ?? 2));
  wrap.setAttribute("gs-min-h", String(opts.minH ?? 2));

  const content = document.createElement("div");
  content.className = "grid-stack-item-content";
  // Insertar wrap en el lugar original del elemento, después mover el
  // elemento dentro de content. Así preservamos el orden para casos donde
  // grid no es el container directo.
  el.parentNode?.insertBefore(wrap, el);
  content.appendChild(el);
  wrap.appendChild(content);

  // Si el elemento está hidden, marcamos el wrapper como pending.
  if (el.hasAttribute && el.hasAttribute("hidden")) {
    wrap.classList.add("ra-pending");
    wrap.style.display = "none";
  }

  // Mover al final del grid container (Gridstack lo posiciona via gs-x/y).
  grid.appendChild(wrap);
}

function flattenPanelsIntoGrid() {
  const main = document.querySelector("main.shell");
  if (!main) return null;

  // Crear container si no existe — al principio del main, antes que todo.
  let grid = document.getElementById("ra-grid");
  if (!grid) {
    grid = document.createElement("div");
    grid.id = "ra-grid";
    grid.className = "grid-stack";
    main.insertBefore(grid, main.firstChild);
  }

  const savedMap = readSavedLayout();

  // 1. Today hero — full-width arriba.
  const hero = document.querySelector("section.today-hero");
  if (hero) wrapElementAsGridItem(grid, hero, "today-hero", savedMap, { minW: 4, minH: 3 });

  // 2. Command bar (KPIs) — strip horizontal.
  const cmdbar = document.querySelector("section.cmdbar");
  if (cmdbar) wrapElementAsGridItem(grid, cmdbar, "cmdbar", savedMap, { minW: 3, minH: 1 });

  // 3. Todos los paneles.
  const panels = Array.from(document.querySelectorAll(".section-body > .panel"));
  for (const panel of panels) {
    wrapElementAsGridItem(grid, panel, panel.id, savedMap);
  }

  // Ocultar sections wrappers ahora vacíos (sólo los `.section` regulares;
  // today-hero y cmdbar quedaron envueltos en grid-stack-item).
  document.querySelectorAll(".section").forEach((sec) => {
    const body = sec.querySelector(".section-body");
    if (body && body.children.length === 0) sec.hidden = true;
  });

  return grid;
}

// ── Gridstack init + persistencia ─────────────────────────────────────────────

let _grid = null;

function initGridstack(gridEl) {
  if (!window.GridStack) {
    console.warn("[home.v2] GridStack no cargó — layout libre desactivado");
    return null;
  }

  const editMode = readEditMode();

  _grid = window.GridStack.init({
    column: 12,
    cellHeight: 80,
    margin: 8,
    float: false,
    animate: true,
    disableDrag: !editMode,
    disableResize: !editMode,
    handle: ".panel-head",
    resizable: { handles: "e, se, s, sw, w" },
    minRow: 1,
    acceptWidgets: true,
    columnOpts: {
      breakpointForWindow: true,
      breakpoints: [
        { w: 768,  c: 1  },
        { w: 1100, c: 6  },
        { w: 1600, c: 12 },
      ],
    },
  }, gridEl);

  _grid.on("change", () => {
    saveLayout();
  });

  return _grid;
}

function saveLayout() {
  if (!_grid) return;
  try {
    const layout = _grid.save(false);  // [{id, x, y, w, h}]
    localStorage.setItem(LS_GRID_LAYOUT, JSON.stringify(layout));
  } catch (e) {
    console.warn("[home.v2] no pude persistir layout:", e);
  }
  updateResetButtonVisibility();
}

// ── Hidden panels: observer + dynamic add ─────────────────────────────────────

function installHiddenObserver() {
  const wrappers = document.querySelectorAll(".grid-stack-item.ra-pending");
  wrappers.forEach((wrap) => {
    const panel = wrap.querySelector(".panel");
    if (!panel) return;
    const obs = new MutationObserver(() => {
      if (!panel.hasAttribute("hidden")) {
        // Mostrar el wrapper + registrar widget en Gridstack.
        wrap.style.display = "";
        wrap.classList.remove("ra-pending");
        if (_grid) {
          try { _grid.makeWidget(wrap); } catch {}
        }
        obs.disconnect();
      }
    });
    obs.observe(panel, { attributes: true, attributeFilter: ["hidden"] });
  });
}

// ── Edit mode (lock/unlock) ───────────────────────────────────────────────────

function readEditMode() {
  try {
    const val = localStorage.getItem(LS_GRID_EDIT_MODE);
    // Si nunca se seteó (primera visita), arranca ON para que el user
    // vea inmediatamente que puede mover/redimensionar.
    if (val === null) return true;
    return val === "1";
  } catch { return true; }
}

function writeEditMode(on) {
  try {
    if (on) localStorage.setItem(LS_GRID_EDIT_MODE, "1");
    else localStorage.removeItem(LS_GRID_EDIT_MODE);
  } catch {}
}

function setEditMode(on) {
  if (!_grid) return;
  writeEditMode(on);
  document.body.classList.toggle("ra-layout-editing", on);
  if (on) {
    _grid.enable();
  } else {
    _grid.disable();
  }
  const btn = document.getElementById("layout-edit-btn");
  if (btn) {
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.textContent = on ? "🔓 layout" : "🔒 layout";
    btn.title = on
      ? "Bloquear layout (estás editando)"
      : "Desbloquear layout para mover/redimensionar paneles";
  }
}

function injectEditButton() {
  const meta = document.getElementById("topbar-meta");
  if (!meta || document.getElementById("layout-edit-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "layout-edit-btn";
  btn.className = "layout-edit-btn";
  const initialOn = readEditMode();
  btn.textContent = initialOn ? "🔓 layout" : "🔒 layout";
  btn.title = initialOn
    ? "Bloquear layout (estás editando)"
    : "Desbloquear layout para mover/redimensionar paneles";
  btn.setAttribute("aria-pressed", initialOn ? "true" : "false");
  btn.addEventListener("click", () => {
    const next = !readEditMode();
    setEditMode(next);
  });
  meta.appendChild(btn);
}

// ── Reset button ──────────────────────────────────────────────────────────────

function hasCustomLayout() {
  try {
    if (localStorage.getItem(LS_GRID_LAYOUT)) return true;
    if (localStorage.getItem(LS_PANELS_COLLAPSED)) return true;
    if (localStorage.getItem(LS_HERO_COLLAPSED) === "1") return true;
    if (localStorage.getItem(LS_HERO_ORDER)) return true;
    if (localStorage.getItem(LS_HERO_SUB_COLLAPSED)) return true;
    if (localStorage.getItem(LS_SECTIONS_COLLAPSED)) return true;
  } catch {}
  return false;
}

export function updateResetButtonVisibility() {
  const btn = document.getElementById("reset-order-btn");
  if (!btn) return;
  btn.hidden = !hasCustomLayout();
}

function clearAllLayout() {
  const keys = [
    LS_GRID_LAYOUT,
    LS_GRID_EDIT_MODE,
    LS_PANELS_COLLAPSED,
    LS_HERO_COLLAPSED,
    LS_HERO_ORDER,
    LS_HERO_SUB_COLLAPSED,
    LS_SECTIONS_COLLAPSED,
  ];
  for (const k of keys) {
    try { localStorage.removeItem(k); } catch {}
  }
  window.location.reload();
}

function injectResetButton() {
  const meta = document.getElementById("topbar-meta");
  if (!meta || document.getElementById("reset-order-btn")) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "reset-order-btn";
  btn.className = "reset-order-btn";
  btn.title = "Resetear layout (posición/tamaño + collapse) al default";
  btn.setAttribute("aria-label", "Resetear layout");
  btn.textContent = "↺ layout";
  btn.hidden = !hasCustomLayout();
  btn.addEventListener("click", () => {
    if (confirm("¿Resetear el layout al default? (posición, tamaño y collapse)")) {
      clearAllLayout();
    }
  });
  meta.appendChild(btn);
}

// ── Collapse por panel (preservado del sistema anterior) ──────────────────────

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

function applySavedCollapse() {
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

function injectCollapseButton(panel) {
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
  // Evitar que el click inicie un drag de Gridstack.
  btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
  btn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
  head.appendChild(btn);
}

// ── Init principal ────────────────────────────────────────────────────────────

export function initLayout() {
  // Exponer para que panel-today.mjs pueda llamar al toggle del reset btn.
  window._updateResetButtonVisibility = updateResetButtonVisibility;

  // 1. Aplanar todos los paneles en un único grid-stack container.
  const gridEl = flattenPanelsIntoGrid();
  if (!gridEl) return;

  // 2. Inyectar botón collapse por panel.
  document.querySelectorAll(".grid-stack-item .panel").forEach((panel) => {
    injectCollapseButton(panel);
  });

  // 3. Aplicar estado collapse persistido.
  applySavedCollapse();

  // 4. Wait for Gridstack script. Si todavía no cargó, esperar.
  if (window.GridStack) {
    bootGrid(gridEl);
  } else {
    // Polling corto hasta que el defer-loaded script aparezca.
    let waited = 0;
    const iv = setInterval(() => {
      waited += 50;
      if (window.GridStack) {
        clearInterval(iv);
        bootGrid(gridEl);
      } else if (waited > 4000) {
        clearInterval(iv);
        console.warn("[home.v2] GridStack no apareció en 4s — sin grid libre");
      }
    }, 50);
  }
}

function bootGrid(gridEl) {
  initGridstack(gridEl);
  installHiddenObserver();
  installHeroSplitter();
  injectEditButton();
  injectResetButton();
  // Reflejar edit mode persistido (Gridstack arranca disabled por opts,
  // pero el botón debe mostrar el estado correcto).
  if (readEditMode()) {
    document.body.classList.add("ra-layout-editing");
  }
  // Mostrar hint banner si es la primera vez que el user ve el layout libre.
  maybeShowEditHint();
}

// ── Hero split: las sub-cajas del brief (LO QUE PASÓ / SIN PROCESAR /
//    PREGUNTAS / AGENDA) salen del today-hero-body y se convierten en
//    grid-items independientes. El today-hero queda como header chico
//    (título + botón refresh). Cada vez que el brief se re-genera,
//    panel-today.mjs reescribe el innerHTML de today-hero-body — el
//    observer detecta los nuevos .hero-section y los redistribuye al grid.
let _heroSplitInProgress = false;
function splitHeroSections() {
  if (_heroSplitInProgress) return;
  if (!_grid) return;
  const heroBody = document.getElementById("today-hero-body");
  if (!heroBody) return;
  _heroSplitInProgress = true;
  try {
    const savedMap = readSavedLayout();
    const sections = Array.from(heroBody.querySelectorAll(":scope > .hero-section"));
    for (const sec of sections) {
      const subKey = Array.from(sec.classList).find((c) => c.startsWith("s-"));
      if (!subKey) continue;
      const gsId = HERO_SUB_MAP[subKey];
      if (!gsId) continue;

      const existing = document.querySelector(`#ra-grid > [gs-id="${gsId}"]`);
      if (existing) {
        // Re-render: swap content into existing wrapper, preserve layout.
        const content = existing.querySelector(".grid-stack-item-content");
        if (content) {
          content.innerHTML = "";
          content.appendChild(sec);
          existing.classList.remove("ra-pending");
          existing.style.display = "";
        }
      } else {
        // First time: crear wrapper + agregar widget a Gridstack.
        const layout = savedMap?.[gsId] || DEFAULT_LAYOUT[gsId] || { w: 4, h: 4 };
        const wrap = document.createElement("div");
        wrap.className = "grid-stack-item";
        wrap.setAttribute("gs-id", gsId);
        wrap.setAttribute("gs-x", String(layout.x ?? 0));
        wrap.setAttribute("gs-y", String(layout.y ?? 0));
        wrap.setAttribute("gs-w", String(layout.w ?? 4));
        wrap.setAttribute("gs-h", String(layout.h ?? 4));
        wrap.setAttribute("gs-min-w", "2");
        wrap.setAttribute("gs-min-h", "2");
        const content = document.createElement("div");
        content.className = "grid-stack-item-content";
        content.appendChild(sec);
        wrap.appendChild(content);
        document.getElementById("ra-grid").appendChild(wrap);
        try { _grid.makeWidget(wrap); } catch (e) {
          console.warn("[home.v2] no pude agregar hero-section al grid:", e);
        }
      }
    }
    // Si el brief no tiene sections (todavía cargando), no tocamos nada.
  } finally {
    _heroSplitInProgress = false;
  }
}

function installHeroSplitter() {
  const heroBody = document.getElementById("today-hero-body");
  if (!heroBody) return;
  // Split inicial si el brief ya está renderizado (raro pero posible).
  splitHeroSections();
  // Observer para futuras re-renderizaciones.
  const obs = new MutationObserver((muts) => {
    // Sólo nos interesa cuando aparecen nuevos .hero-section.
    let hasNewSection = false;
    for (const m of muts) {
      for (const node of m.addedNodes) {
        if (node.nodeType === 1 && node.classList?.contains("hero-section")) {
          hasNewSection = true;
          break;
        }
      }
      if (hasNewSection) break;
    }
    if (hasNewSection) splitHeroSections();
  });
  obs.observe(heroBody, { childList: true });
}

function maybeShowEditHint() {
  try {
    if (localStorage.getItem(LS_GRID_HINT_SEEN) === "1") return;
  } catch {}
  if (!readEditMode()) return;
  const hint = document.createElement("div");
  hint.className = "ra-edit-hint";
  hint.setAttribute("role", "status");
  hint.innerHTML = (
    'Layout libre activo · arrastrá desde <code>el header</code> · ' +
    'redimensioná desde los <code>bordes</code> · ' +
    '<code>🔒 layout</code> bloquea · <code>↺ layout</code> resetea'
  );
  document.body.appendChild(hint);
  setTimeout(() => {
    hint.style.transition = "opacity .4s ease-out";
    hint.style.opacity = "0";
    setTimeout(() => hint.remove(), 500);
  }, 8000);
  try { localStorage.setItem(LS_GRID_HINT_SEEN, "1"); } catch {}
}
