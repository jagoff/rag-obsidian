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
const THRESHOLD_TINY_PX = 150;
const THRESHOLD_LARGE_PX = 450;

const _pending = new Map(); // panelId -> timeoutId

function readOverrides() {
  try {
    const raw = localStorage.getItem(LS_PANEL_SIZES);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : {};
  } catch { return {}; }
}

function classifyByContent(panel) {
  const body = panel.querySelector(".panel-body");
  if (!body) return { w: "half", h: "full" };
  // scrollHeight = full content height (incluye overflow).
  // clientHeight = visible viewport (clipped por overflow:hidden).
  const sh = body.scrollHeight;
  const ch = body.clientHeight;
  const overflowing = sh > ch + 8; // tolerancia anti-1px jitter
  if (overflowing || sh > THRESHOLD_LARGE_PX) {
    return { w: "full", h: "full" };
  }
  if (sh < THRESHOLD_TINY_PX) {
    return { w: "half", h: "half" };
  }
  return { w: "half", h: "full" };
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
