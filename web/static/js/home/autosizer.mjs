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

const LS_PANEL_SIZES = "home.v2.panel-sizes.v1";
const RESIZE_DEBOUNCE_MS = 200;

// Altura disponible en cada celda (descontando .panel-head ~52px + gap):
//   data-h=half (1 row, 280px) → body ~228px disponibles
//   data-h=full (2 rows, 580px) → body ~528px disponibles
// Si scrollHeight (content real, sin clip) supera el disponible de "half",
// promote a "full". Si excede full también, mantiene full + el overflow
// se scrollea dentro del .panel-body.
const HEIGHT_HALF_FITS_PX = 228;

const _pending = new Map(); // panelId -> timeoutId

function readOverrides() {
  try {
    const raw = localStorage.getItem(LS_PANEL_SIZES);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : {};
  } catch { return {}; }
}

function measureNaturalHeight(panel) {
  // scrollHeight de un elemento con overflow:hidden y altura limitada por
  // el grid SIGUE devolviendo el content height real (no el visible). Es
  // exactamente lo que queremos: cuántos px ocuparía el content si no
  // hubiera clip. Sumamos las alturas de head + body + foot.
  let total = 0;
  for (const sel of [".panel-head", ".panel-body", ".panel-foot"]) {
    const el = panel.querySelector(sel);
    if (!el) continue;
    total += el.scrollHeight;
  }
  return total;
}

function classifyByContent(panel) {
  const natural = measureNaturalHeight(panel);
  // Solo decidimos height por content. Width queda half default
  // (override manual via chip ↔). Esto evita que panels con poco content
  // pero texto wide se promuevan a full ancho innecesariamente.
  const h = natural > HEIGHT_HALF_FITS_PX + 40 ? "full" : "half";
  return { w: panel.dataset.w || "half", h };
}

function applySize(panel, w, h) {
  if (panel.dataset.w !== w) panel.dataset.w = w;
  if (panel.dataset.h !== h) panel.dataset.h = h;
}

function resizePanel(panel) {
  if (!panel || !panel.id) return;
  const overrides = readOverrides();
  const ov = overrides[panel.id];
  if (ov && (ov.w === "half" || ov.w === "full") && (ov.h === "half" || ov.h === "full")) {
    applySize(panel, ov.w, ov.h);
    return;
  }
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
  root.querySelectorAll(".section-body > .panel").forEach(observePanel);
}

// Manual override API — desde DevTools o un botón futuro:
// window.ragSetPanelSize("p-inbox", "full", "full")
export function setPanelSize(panelId, w, h) {
  const overrides = readOverrides();
  if (!w && !h) {
    delete overrides[panelId];
  } else {
    overrides[panelId] = { w, h };
  }
  try {
    if (Object.keys(overrides).length === 0) {
      localStorage.removeItem(LS_PANEL_SIZES);
    } else {
      localStorage.setItem(LS_PANEL_SIZES, JSON.stringify(overrides));
    }
  } catch (e) {
    console.warn("[home.v2] no pude persistir override de tamaño:", e);
  }
  const panel = document.getElementById(panelId);
  if (panel) resizePanel(panel);
}

export function clearPanelSizeOverrides() {
  try { localStorage.removeItem(LS_PANEL_SIZES); } catch {}
  document.querySelectorAll(".section-body > .panel").forEach(resizePanel);
}

// Expose para debugging desde la console.
if (typeof window !== "undefined") {
  window.ragSetPanelSize = setPanelSize;
  window.ragClearPanelSizes = clearPanelSizeOverrides;
}
