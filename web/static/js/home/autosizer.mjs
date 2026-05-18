// autosizer.mjs — clasifica cada panel del home en 1 de 4 tamaños
// discretos (half/full × half/full) según el content.
//
// Regla:
//   - content vacío o muy chico (<150px scrollHeight) → data-w=half data-h=half
//   - content overflow (scrollHeight > clientHeight) o grande (>450px) → data-w=full data-h=full
//   - default intermedio → data-w=half data-h=full
//
// Override manual via localStorage["home.v2.panel-sizes.v1"] — formato
// { "p-inbox": { "w": "full", "h": "full" }, ... }. Si existe, el
// autosizer respeta el override y no toca data-w/data-h.

import {
  applySizeDataset,
  hasSizeOverrides,
  isValidSizeOverride,
  readSizeOverrides,
  removeKey,
  writeSizeOverrides,
} from "../layout-persistence.mjs?v=103";

export const LS_PANEL_SIZES = "home.v2.panel-sizes.v1";
const PANEL_SIZE_OPTIONS = { widths: ["half", "full"], heights: ["half", "full", "xl"] };
const RESIZE_DEBOUNCE_MS = 200;

// Altura del .panel-body disponible en un cell de data-h=half:
//   cell = 280px (1 row del grid-auto-rows)
//   .panel-head ~50px + .panel-foot ~30px (cuando hay) + padding ~16px
//   → body real disponible ~184px. Margen de 6px para anti-jitter.
// Si scrollHeight del body excede esto, promote a data-h=full (~528px body).
const HEIGHT_HALF_FITS_PX = 190;

const _pending = new Map(); // panelId -> timeoutId

function readOverrides() {
  return readSizeOverrides(LS_PANEL_SIZES, PANEL_SIZE_OPTIONS);
}

function measureBodyContent(panel) {
  // OJO: `.panel-body` tiene `flex: 1` → scrollHeight devuelve el alto de
  // la celda del grid (~528 en full, ~190 en half), NO el content real.
  // Para medir content natural sumamos offsetHeight de cada child del
  // body. offsetHeight de un block-level child refleja el alto que
  // ocuparía sin restricción del padre flex.
  const body = panel.querySelector(".panel-body");
  if (!body) return 0;
  let total = 0;
  for (const child of body.children) {
    // marginTop/bottom se computan via getComputedStyle si interesa, pero
    // typicamente los children son <ul>/<table>/<div> sin margenes
    // verticales fuertes (los items tienen gap interno).
    total += child.offsetHeight || 0;
  }
  return total;
}

function classifyByContent(panel) {
  const bodyH = measureBodyContent(panel);
  // Threshold conservador: solo promote a full si el content body
  // realmente excede lo que cabe en half (~190px disponibles + buffer).
  // Esto deja la mayoría de panels en half×half (compacto) y solo
  // panels con tablas largas o listas grandes saltan a half×full.
  const h = bodyH > HEIGHT_HALF_FITS_PX ? "full" : "half";
  return { w: panel.dataset.w || "half", h };
}

function applySize(panel, w, h) {
  applySizeDataset(panel, { w, h });
}

export function applySavedPanelSize(panel) {
  if (!panel || !panel.id) return false;
  const overrides = readOverrides();
  const ov = overrides[panel.id];
  if (!isValidSizeOverride(ov, PANEL_SIZE_OPTIONS)) return false;
  applySize(panel, ov.w, ov.h);
  return true;
}

export function hasPanelSizeOverrides() {
  return hasSizeOverrides(LS_PANEL_SIZES, PANEL_SIZE_OPTIONS);
}

function resizePanel(panel) {
  if (!panel || !panel.id) return;
  if (applySavedPanelSize(panel)) return;
  // Si está hidden, no clasificar (scrollHeight devolvería 0 o stale).
  if (panel.hidden || panel.offsetParent === null) return;
  // Si está collapsed, mantener height=half (el cuerpo está oculto).
  if (panel.getAttribute("data-collapsed") === "true") {
    applySize(panel, panel.dataset.w || "half", "half");
    return;
  }
  const { w, h } = classifyByContent(panel);
  applySize(panel, w, h);
}

function scheduleResize(panel) {
  if (!panel || !panel.id) return;
  const prev = _pending.get(panel.id);
  if (prev) clearTimeout(prev);
  const tid = setTimeout(() => {
    _pending.delete(panel.id);
    resizePanel(panel);
  }, RESIZE_DEBOUNCE_MS);
  _pending.set(panel.id, tid);
}

export function observePanel(panel) {
  if (!panel || panel.dataset.autosizerInit === "1") return;
  panel.dataset.autosizerInit = "1";
  // Primer pass inmediato (sin debounce) para evitar flash del default.
  resizePanel(panel);
  const body = panel.querySelector(".panel-body");
  if (!body) return;
  // MutationObserver: cualquier cambio de DOM dentro del cuerpo
  // (re-render de panel-today.mjs, fetch que repobla, etc) re-clasifica.
  const mo = new MutationObserver(() => scheduleResize(panel));
  mo.observe(body, { childList: true, subtree: true, characterData: true });
  // ResizeObserver: cuando cambia el ancho del viewport (responsive),
  // el threshold tiny/large cambia de significado.
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => scheduleResize(panel));
    ro.observe(body);
  }
}

export function observeAllPanels(root = document) {
  root
    .querySelectorAll("#today-hero-body > .panel, #home-cmdbar > .panel, .section-body > .panel")
    .forEach(observePanel);
}

// Manual override API — desde DevTools o un botón futuro:
// window.ragSetPanelSize("p-inbox", "full", "full")
export function setPanelSize(panelId, w, h) {
  if (!panelId) return;
  const overrides = readOverrides();
  if (!w && !h) {
    delete overrides[panelId];
  } else if (isValidSizeOverride({ w, h }, PANEL_SIZE_OPTIONS)) {
    overrides[panelId] = { w, h };
  } else {
    delete overrides[panelId];
  }
  writeSizeOverrides(LS_PANEL_SIZES, overrides, PANEL_SIZE_OPTIONS);
  const panel = document.getElementById(panelId);
  if (panel) {
    if (!applySavedPanelSize(panel)) resizePanel(panel);
  }
}

export function clearPanelSizeOverrides() {
  removeKey(LS_PANEL_SIZES);
  document
    .querySelectorAll("#today-hero-body > .panel, #home-cmdbar > .panel, .section-body > .panel")
    .forEach(resizePanel);
}

// Expose para debugging desde la console.
if (typeof window !== "undefined") {
  window.ragSetPanelSize = setPanelSize;
  window.ragClearPanelSizes = clearPanelSizeOverrides;
}
