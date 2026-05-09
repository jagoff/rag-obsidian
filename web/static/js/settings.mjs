// settings.mjs — Vault picker, chat-mode toggle, TTS toggle.
// Persiste en localStorage. Single-source-of-truth: los selects/botones del
// topbar (sidebar en desktop), sincronizados al sheet de mobile por
// mobile-ui.mjs.

import { el } from "./utils.mjs";

// ── Claves localStorage ────────────────────────────────────────────────
export const VAULT_KEY   = "obsidian-rag:vault";
export const TTS_KEY     = "obsidian-rag:tts";
export const CHAT_MODE_KEY = "rag-chat-mode";
export const VALID_MODES = new Set(["auto", "fast", "deep"]);

// ── Estado mutable exportado ───────────────────────────────────────────
export let vaultScope  = localStorage.getItem(VAULT_KEY) || "";
export let ttsEnabled  = localStorage.getItem(TTS_KEY) === "1";

// ── Vault picker ───────────────────────────────────────────────────────

/** Carga la lista de vaults desde /api/vaults y pobla el <select>. */
export async function loadVaults() {
  const vaultPicker = document.getElementById("vault-picker");
  if (!vaultPicker) return;
  try {
    const res = await fetch("/api/vaults");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    vaultPicker.innerHTML = "";

    const activeOpt = document.createElement("option");
    activeOpt.value = "";
    activeOpt.textContent = data.active ? `${data.active} (activo)` : "activo";
    vaultPicker.appendChild(activeOpt);

    const others = (data.registered || []).filter((n) => n !== data.active);
    for (const name of others) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      vaultPicker.appendChild(opt);
    }

    if (data.registered && data.registered.length > 1) {
      const allOpt = document.createElement("option");
      allOpt.value = "all";
      allOpt.textContent = "todos";
      vaultPicker.appendChild(allOpt);
    }

    const options = Array.from(vaultPicker.options).map((o) => o.value);
    const storedScope = localStorage.getItem(VAULT_KEY);
    if (storedScope !== null && options.includes(storedScope)) {
      vaultPicker.value = storedScope;
      vaultScope = storedScope;
    } else if (storedScope === null && options.includes("all")) {
      vaultPicker.value = "all";
      vaultScope = "all";
      localStorage.setItem(VAULT_KEY, "all");
    } else {
      vaultPicker.value = "";
      vaultScope = "";
      localStorage.removeItem(VAULT_KEY);
    }
  } catch (_) {
    vaultPicker.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "activo";
    vaultPicker.appendChild(opt);
  }
}

/** Registra el listener de cambio del vault picker. */
export function initVaultPicker() {
  const vaultPicker = document.getElementById("vault-picker");
  if (!vaultPicker) return;
  vaultPicker.addEventListener("change", () => {
    vaultScope = vaultPicker.value;
    if (vaultScope) localStorage.setItem(VAULT_KEY, vaultScope);
    else localStorage.removeItem(VAULT_KEY);
  });
}

// ── Chat mode toggle (auto / fast / deep) ─────────────────────────────

export function getChatMode() {
  const stored = localStorage.getItem(CHAT_MODE_KEY);
  return stored && VALID_MODES.has(stored) ? stored : "auto";
}

export function setChatMode(mode) {
  if (!VALID_MODES.has(mode)) mode = "auto";
  const chatModeToggle = document.getElementById("chat-mode-toggle");
  localStorage.setItem(CHAT_MODE_KEY, mode);
  if (!chatModeToggle) return;
  chatModeToggle.dataset.mode = mode;
  chatModeToggle.setAttribute("aria-label", `Modo: ${mode}`);
  const label = chatModeToggle.querySelector(".mode-label");
  if (label) label.textContent = mode;
}

export function initChatModeToggle() {
  const chatModeToggle = document.getElementById("chat-mode-toggle");
  if (!chatModeToggle) return;
  // Aplicar estado inicial al DOM.
  setChatMode(getChatMode());
  chatModeToggle.addEventListener("click", (ev) => {
    const modes = [...VALID_MODES];
    const cur = getChatMode();
    const next = modes[(modes.indexOf(cur) + 1) % modes.length];
    setChatMode(next);
    ev.currentTarget.blur();
  });
  chatModeToggle.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      chatModeToggle.click();
    }
  });
}

// ── TTS toggle ─────────────────────────────────────────────────────────

const SPEAKER_ON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg>`;
const SPEAKER_OFF_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>`;

export function renderTtsToggle() {
  const ttsToggle = document.getElementById("tts-toggle");
  if (!ttsToggle) return;
  ttsToggle.innerHTML = ttsEnabled ? SPEAKER_ON_SVG : SPEAKER_OFF_SVG;
  ttsToggle.setAttribute("aria-pressed", ttsEnabled ? "true" : "false");
  ttsToggle.setAttribute("title", ttsEnabled ? "Desactivar voz" : "Activar voz");
}

export function initTtsToggle() {
  const ttsToggle = document.getElementById("tts-toggle");
  if (!ttsToggle) return;
  renderTtsToggle();
  ttsToggle.addEventListener("click", () => {
    ttsEnabled = !ttsEnabled;
    localStorage.setItem(TTS_KEY, ttsEnabled ? "1" : "0");
    renderTtsToggle();
  });
}

// ── Setters externos (usados por session.mjs vía /tts slash command) ──

/** Cambia el estado de TTS desde fuera (ej. slash /tts). */
export function setTtsEnabled(value) {
  ttsEnabled = !!value;
  localStorage.setItem(TTS_KEY, ttsEnabled ? "1" : "0");
  renderTtsToggle();
}
