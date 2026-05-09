/**
 * mobile-ui.mjs — Phase W4-phase-2 (2026-05-09)
 *
 * Interacciones touch/mobile del chat:
 *   - Bottom-sheet menu (⋯ button → abre el sheet con vault-picker, TTS toggle)
 *   - Sync bidireccional sheet ↔ pickers originales del topbar
 *   - Focus inicial (desktop-only, evita abrir keyboard en iOS al cargar)
 *   - Send button state (disabled/enabled según contenido del input)
 *   - Contact match overlay + badge (highlight del nombre en el textarea)
 *   - Quick-chips (chips en el empty-hero que prefilian el input)
 *
 * Estado original en app.js (líneas ~5146-5272):
 *   - menuBtn, menuSheet, sheetVaultPicker, sheetTtsToggle refs
 *   - openSheet(), closeSheet(), syncSheetFromOriginals()
 *   - initMobileTier1() — IIFE
 *
 * Estrategia de extracción Phase W4:
 *   Igual que sidebar.mjs — durante la transición re-exportamos desde window.
 *   El IIFE initMobileTier1() en app.js corre al evaluar el bundle, así que
 *   al momento de importar este módulo el código ya está ejecutado.
 *
 * Globals del window consumidos:
 *   window.openSheet()
 *   window.closeSheet()
 *   window.syncSheetFromOriginals()
 */

// ── Re-exports desde window ───────────────────────────────────────────────

/**
 * Abre el sheet de opciones mobile.
 */
export function openSheet() {
  if (typeof window.openSheet === "function") return window.openSheet();
}

/**
 * Cierra el sheet de opciones mobile.
 */
export function closeSheet() {
  if (typeof window.closeSheet === "function") return window.closeSheet();
}

/**
 * Copia opciones + estado desde los pickers/toggle del topbar al sheet.
 * Llamar después de interacciones que pueden cambiar el estado (ej. TTS click).
 */
export function syncSheetFromOriginals() {
  if (typeof window.syncSheetFromOriginals === "function") return window.syncSheetFromOriginals();
}

/**
 * Inicializa el Contact Match feature:
 *   - overlay encima del textarea con highlight del nombre detectado
 *   - badge con nombre canónico + teléfono debajo del textarea
 *   - debounce 300ms → POST /api/whatsapp/contacts/match
 *
 * app.js inicializa esto al boot si #input-overlay está en el DOM.
 * Este módulo expone las funciones internas por si algún caller futuro
 * necesita forzar un re-render o limpiar el estado.
 */
export function clearContactMatchOverlay() {
  const overlay = document.getElementById("input-overlay");
  const badge = document.getElementById("contact-match-badge");
  if (overlay) overlay.textContent = "";
  if (badge) { badge.hidden = true; badge.textContent = ""; }
}

/**
 * Actualiza el estado del send button según si hay texto en el input.
 * Idempotente — seguro llamar en cualquier momento.
 */
export function updateSendBtnState() {
  if (typeof window.updateSendBtnState === "function") return window.updateSendBtnState();
  // Fallback inline.
  const sendBtn = document.getElementById("send-btn");
  const input   = document.getElementById("input");
  if (!sendBtn || !input) return;
  const pending = typeof window.pending === "boolean" ? window.pending : false;
  sendBtn.disabled = !input.value.trim().length || pending;
}

/**
 * Información de detección del user agent para iOS.
 * @returns {boolean}
 */
export function isIOSVersionBelow16() {
  if (typeof window.isIOSVersionBelow16 === "function") return window.isIOSVersionBelow16();
  const ua = navigator.userAgent || "";
  const m  = ua.match(/iPhone OS (\d+)_/);
  return m ? parseInt(m[1], 10) < 16 : false;
}

/**
 * Smooth scroll behaviour que respeta prefers-reduced-motion.
 * @returns {"smooth"|"auto"}
 */
export function smoothBehavior() {
  if (typeof window.smoothBehavior === "function") return window.smoothBehavior();
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}
