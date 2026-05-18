// app.mjs — entry point del dashboard /v2 (mission control).
// Fetches /api/home, despacha todos los renderers, orquesta el auto-refresh.

import { $, setCurrentPayload, startAutoRefresh } from "./core.mjs";
import { renderCmdBar } from "./command-bar.mjs?v=103";
import {
  renderTodayHero, initHeroCollapse, initReminderButtonHandler,
} from "./panel-today.mjs?v=105";
import {
  renderInbox, renderQuestions, renderTomorrow,
  renderWAUnreplied, renderLoopsUrgent, renderContradictions,
  renderPatterns, renderAuthority,
} from "./panel-signals.mjs?v=103";
import { renderFinance, renderCards } from "./panel-finance.mjs";
import {
  renderRetrievalHealth, renderLoopsAging, renderEvalTrend,
  renderVaultActivity, renderCaptured,
} from "./panel-monitoring.mjs?v=103";
import {
  renderWeather, renderWeb, renderBookmarks,
  renderYouTube, renderDrive, renderSpotify, renderHealth, renderPeekaboo,
} from "./panel-ambient.mjs?v=105";
import { renderSleep } from "./panel-sleep.mjs";
import { renderMood, renderCorrelations } from "./panel-mood.mjs";
import { initLayout, refreshLayoutControls, updateResetButtonVisibility } from "./layout.mjs?v=103";
import { initRefreshButton } from "./refresh.mjs";

// ── Topbar ─────────────────────────────────────────────────────────────────────

function updateTopbar(payload) {
  const dateEl = $("#today-date");
  const lastEl = $("#last-update");
  if (payload.date && dateEl) dateEl.textContent = payload.date;
  if (payload.generated_at && lastEl) {
    const ms = Date.now() - new Date(payload.generated_at).getTime();
    const min = Math.floor(ms / 60_000);
    const ago = min < 1 ? "ahora" : min < 60 ? `${min}m` : `${Math.floor(min / 60)}h`;
    lastEl.textContent = `actualizado ${ago}`;
  }
  const dot = $("#serve-dot");
  if (dot) dot.classList.remove("warn", "crit");
}

// ── Render completo ────────────────────────────────────────────────────────────

function render(payload) {
  setCurrentPayload(payload);
  updateTopbar(payload);
  renderTodayHero(payload);
  renderCmdBar(payload);
  renderPatterns(payload);
  renderInbox(payload);
  renderQuestions(payload);
  renderTomorrow(payload);
  renderWAUnreplied(payload);
  renderLoopsUrgent(payload);
  renderContradictions(payload);
  renderFinance(payload);
  renderCards(payload);
  renderRetrievalHealth(payload);
  renderLoopsAging(payload);
  renderAuthority(payload);
  renderEvalTrend(payload);
  renderWeather(payload);
  renderVaultActivity(payload);
  renderCaptured(payload);
  renderWeb(payload);
  renderBookmarks(payload);
  renderYouTube(payload);
  renderDrive(payload);
  renderSpotify(payload);
  renderSleep(payload);
  renderMood(payload);
  renderHealth(payload);
  renderPeekaboo(payload);
  // renderCorrelations: panel `p-correlations` (Pearson cross-source entre métricas).
  // Distinto de renderPatterns que rendea `p-patterns` (entidades cross-source).
  renderCorrelations(payload);
  refreshLayoutControls();
}

// ── Fetch + load ───────────────────────────────────────────────────────────────

async function load(opts = {}) {
  try {
    const url = opts.regenerate ? "/api/home?regenerate=true" : "/api/home";
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const payload = await r.json();
    render(payload);
  } catch (err) {
    console.error("[home.v2] load failed:", err);
    const dot = $("#serve-dot");
    const txt = $("#serve-text");
    if (dot) dot.classList.add("crit");
    if (txt) txt.textContent = "serve · down";
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  // Layout (drag/drop, collapse, reset) — ANTES del primer load para que
  // el orden persistido se aplique antes de que los renderers escriban.
  try {
    await initLayout();
  } catch (err) {
    console.warn("[home.v2] layout init failed:", err);
  }
  // Hero collapse toggle
  initHeroCollapse();
  // Handler para botones inline "crear reminder"
  initReminderButtonHandler();
  // Refresh manual con SSE progress bar
  initRefreshButton(
    (payload) => {
      // callback "done": recibe el payload desde el SSE y re-renderiza.
      setCurrentPayload(payload);
      render(payload);
    },
    load,  // fallback al endpoint normal si el SSE timeout
  );
  // Primer fetch
  load();
  // Auto-refresh cada 5 min
  startAutoRefresh(load, 5 * 60 * 1000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}

// ── Browser globals requeridos desde HTML inline ───────────────────────────────
// No hay onclick= en el HTML de home.v2.html según el código actual,
// pero si en el futuro se agregan, exportarlos acá:
// window.homeV2Reload = () => load({ regenerate: true });
