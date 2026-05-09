// app.mjs — Entry point del chat de obsidian-rag.
//
// Phase W4-phase-3 (2026-05-09): extracción REAL de app.js.
//
// Orden de carga:
//   1. marked.min.js  (síncrono, scope global)
//   2. app.js         (síncrono, script clásico — define globals)
//   3. app.mjs        (defer module — este archivo)
//
// En esta fase:
//   - state.mjs exporta el store compartido de estado
//   - messages-render.mjs tiene implementaciones reales de todas las
//     funciones de renderizado (appendTurn, appendSources, appendFeedback,
//     appendRelated, appendFollowups, showToast, etc.)
//   - Este módulo hace override de las versiones de app.js en window.*
//     con las implementaciones de los módulos ES para que el código nuevo
//     que importe desde acá use las versiones modulares.
//
// Globals preservados en window.* (para compat con app.js call sites):
//   window.activeScope, window.setActiveScope, window.clearActiveScope,
//   window.getActiveScopePayload — Feature H (definidos en app.js)
//
// Exposición de internos de app.js en window.* para los módulos:
//   window._appAppendProposal    — appendProposal() de app.js
//   window._appAppendCreatedChip — appendCreatedChip() de app.js
//   window._appHydrateTurns      — hydrateTurns() de app.js (usa renderMarkdown)
//   window.autoGrow              — autoGrow() de app.js (usado en appendFollowups)

// ── Phase 1 ────────────────────────────────────────────────────────────────
import "./utils.mjs";
import "./markdown.mjs";
import "./settings.mjs";
import "./voice.mjs";
import "./session.mjs";

// ── State store ────────────────────────────────────────────────────────────
// Importar para que esté disponible en los módulos que lo necesitan.
// El estado real sigue siendo las variables let/const de app.js por ahora —
// state.mjs es el destino futuro. La sincronización entre ambos ocurre
// gradualmente a medida que los módulos migran a importar state.
import { state } from "./state.mjs";

// ── Phase 3 — implementaciones reales ─────────────────────────────────────
import {
  el,
  escapeHtml,
  obsidianUrl,
  waHref,
  smoothBehavior,
  appendTurn,
  appendLine,
  appendMeta,
  scrollBottom,
  confidenceBadge,
  showToast,
  copyTextToClipboard,
  buildMarkdownExport,
  appendCopyButton,
  appendFeedback,
  appendSources,
  appendEnrich,
  renderGrounding,
  appendRelated,
  appendWebSearch,
  appendFallbackCluster,
  appendFollowups,
  appendProposal,
  appendCreatedChip,
  hydrateTurns,
} from "./messages-render.mjs";

// ── Sidebar (wrappers hasta phase 4) ──────────────────────────────────────
import "./sidebar.mjs";

// ── Mobile UI (wrappers hasta phase 4) ────────────────────────────────────
import "./mobile-ui.mjs";

// ── Chat client (wrappers hasta phase 4) ──────────────────────────────────
import "./chat-client.mjs";

// ── Override window.* con implementaciones reales de módulos ──────────────
// Las versiones de app.js usaban closure sobre variables locales.
// Las versiones de los módulos importan state.mjs y son las canónicas.
// Override seguro: hacemos asignación incondicional porque los módulos
// son la versión "final" y app.js es el shim provisional.

// messages-render — funciones de render puro (sin dependencias circulares)
window._appendTurn          = appendTurn;
window.appendLine           = appendLine;
window.appendMeta           = appendMeta;
window.scrollBottom         = scrollBottom;
window.confidenceBadge      = confidenceBadge;
window.showToast            = showToast;
window.buildMarkdownExport  = buildMarkdownExport;
window.appendCopyButton     = appendCopyButton;
window.appendSources        = appendSources;
window.appendEnrich         = appendEnrich;
window.renderGrounding      = renderGrounding;
window.appendRelated        = appendRelated;
window.appendWebSearch      = appendWebSearch;
window.appendFallbackCluster = appendFallbackCluster;
window.appendFollowups      = appendFollowups;

// appendFeedback necesita sendFn — inyectar window.send como callback.
// Cuando app.js llama appendFeedback(parent, ctx), el window.* ya apunta
// a esta versión que cierra sobre window.send (que a su vez es send() de app.js).
window.appendFeedback = (parent, ctx) => appendFeedback(parent, ctx, window.send);

// appendProposal y appendCreatedChip: las implementaciones complejas
// de app.js se exponen en window._app* para que messages-render.mjs las use.
// Primero asignamos los _app* aliases apuntando a las versiones ORIGINALES
// de app.js (que ahora están en window.* sin el prefijo _app).
if (typeof window.appendProposal === "function" && !window._appAppendProposal) {
  window._appAppendProposal = window.appendProposal;
}
if (typeof window.appendCreatedChip === "function" && !window._appAppendCreatedChip) {
  window._appAppendCreatedChip = window.appendCreatedChip;
}
if (typeof window.hydrateTurns === "function" && !window._appHydrateTurns) {
  window._appHydrateTurns = window.hydrateTurns;
}

// Exponer autoGrow para que appendFollowups pueda llamarla.
// autoGrow está definida en app.js como función local sin exposición explícita.
// La buscamos en el scope de app.js que ya corrió.
if (typeof autoGrow === "function" && !window.autoGrow) {
  window.autoGrow = autoGrow; // eslint-disable-line no-undef
}

// ── Sincronizar state.mjs con el estado de app.js ─────────────────────────
// El store state.mjs inicializa desde localStorage/sessionStorage igual que
// app.js — son idénticos al boot. Para las mutaciones runtime (sessionId
// cambia cuando llega el evento "session" del SSE, etc.), los call sites de
// app.js actualizan sus propias variables locales Y el state del store.
// Esta sincronización es gradual: los nuevos módulos leen de state, app.js
// sigue usando sus variables locales, y al final app.js queda como stub.
//
// Por ahora: exponer una función para que send() de app.js notifique al store.
window.__syncState = function(patch) {
  Object.assign(state, patch);
};

// ── Log de carga ───────────────────────────────────────────────────────────
// Útil en DevTools para verificar que los módulos cargaron y overridearon
// correctamente. En producción estas líneas son no-ops (console.debug
// no aparece a menos que el user tenga DevTools abierto con nivel debug).
if (typeof console !== "undefined" && console.debug) {
  console.debug("[app.mjs] Phase W4-phase-3 cargado — módulos ES activos");
}
