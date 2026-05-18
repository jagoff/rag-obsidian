/* obsidian-rag finance dashboard — Chart.js 4 + vanilla JS.
 *
 * Visual upgrade (v2):
 *   - Sparklines en cada KPI hero (mini line chart con últimos 6 meses).
 *   - Credit card con look de tarjeta física (gradient + chip + brand).
 *   - Line charts con gradient fill + smooth curves + dots solo en hover.
 *   - Donut con label central (total + label).
 *   - Top stores con bars rounded + datalabels custom.
 *   - Avatares circulares con iniciales por destinatario, color por hash.
 *   - Empty states con SVG simple.
 *   - Animaciones de entrada controladas por CSS (respeta prefers-reduced-motion).
 *
 * Patrón core idéntico a learning.js: readTokens, applyChartDefaults,
 * polling con backoff cuando hidden, theme toggle re-aplica colors,
 * selector de ventana destruye charts y re-fetcha.
 */

"use strict";

// ── Color tokens ─────────────────────────────────────────────────────────
const C = {
  cyan: "", green: "", yellow: "", red: "", purple: "", orange: "", pink: "",
  dim: "", border: "", card: "", text: "", textDim: "", grid: "", bg: "",
};

function readTokens() {
  const s = getComputedStyle(document.documentElement);
  const r = (n) => s.getPropertyValue(n).trim();
  C.cyan    = r("--cyan");
  C.green   = r("--green");
  C.yellow  = r("--yellow");
  C.red     = r("--red");
  C.purple  = r("--purple");
  C.orange  = r("--orange");
  C.pink    = r("--pink");
  C.dim     = r("--text-faint");
  C.border  = r("--border");
  C.card    = r("--bg-card");
  C.text    = r("--text");
  C.textDim = r("--text-dim");
  C.grid    = r("--grid");
  C.bg      = r("--bg");
}
readTokens();

function paletteSeries() {
  return [C.cyan, C.green, C.yellow, C.red, C.purple, C.orange, C.pink];
}
const PALETTE_CLASSES = ["", "green", "yellow", "red", "purple", "orange", "pink"];

function applyChartDefaults() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.color = C.textDim;
  Chart.defaults.borderColor = C.border;
  Chart.defaults.font.family = "'SF Mono','Menlo','Monaco','JetBrains Mono',ui-monospace,monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.padding = 12;
  Chart.defaults.plugins.tooltip.backgroundColor = C.card;
  Chart.defaults.plugins.tooltip.titleColor = C.text;
  Chart.defaults.plugins.tooltip.bodyColor = C.text;
  Chart.defaults.plugins.tooltip.borderColor = C.border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleFont = { size: 11 };
  Chart.defaults.plugins.tooltip.bodyFont = { size: 11 };
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.tooltip.displayColors = false;
  // CRITICAL: NO usar `Chart.defaults.scale.grid = {...}` — eso REEMPLAZA el
  // objeto entero y pierde props como `display`, `drawOnChartArea`, `tickWidth`,
  // `lineWidth`, `tickLength`, `offset`, `drawTicks`, `tickColor`. Ese reemplazo
  // hace que las barras (BarController) NO se rendereen — el canvas queda solo
  // con los ejes. Bug confirmado el 2026-04-29 sobre Chart.js v4.4.7. Hay que
  // mutar la prop puntual:
  Chart.defaults.scale.grid.color = C.grid;
  Chart.defaults.animation.duration = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 600;
  Chart.defaults.animation.easing = "easeOutQuart";
}

// ── State ────────────────────────────────────────────────────────────────
const POLL_MS = 60_000;
const POLL_MAX_MS = 300_000;
const POLL_HIDDEN_GRACE_MS = 300_000;

const state = {
  windowDays: 30,
  paused: false,
  data: null,
  charts: {},
  poll: null,
};

let _hiddenSince = null;

function pollNextDelay(baseMs) {
  if (!document.hidden || _hiddenSince == null) return baseMs;
  const hiddenFor = Date.now() - _hiddenSince;
  if (hiddenFor < POLL_HIDDEN_GRACE_MS) return baseMs;
  const doublings = Math.floor((hiddenFor - POLL_HIDDEN_GRACE_MS) / POLL_HIDDEN_GRACE_MS) + 1;
  return Math.min(baseMs * Math.pow(2, doublings), POLL_MAX_MS);
}

// ── DOM refs ─────────────────────────────────────────────────────────────
const el = {
  metaPeriod: document.getElementById("meta-period"),
  metaUpdated: document.getElementById("meta-updated"),
  liveToggle: document.getElementById("live-toggle"),
  liveLabel: document.getElementById("live-label"),
  themeToggle: document.getElementById("theme-toggle"),
  themeIcon: document.getElementById("theme-icon"),
  segButtons: Array.from(document.querySelectorAll(".seg-btn")),
  collapseButtons: Array.from(document.querySelectorAll(".collapse-btn")),
};

const SUN_SVG  = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>';
const MOON_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

// ── Theme ────────────────────────────────────────────────────────────────
function currentTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit) return explicit;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function setTheme(next) {
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("rag-theme", next); } catch (_) {}
  if (el.themeIcon) el.themeIcon.innerHTML = next === "light" ? MOON_SVG : SUN_SVG;
  if (el.themeToggle) el.themeToggle.setAttribute("aria-label", next === "light" ? "Cambiar a tema oscuro" : "Cambiar a tema claro");
  readTokens();
  applyChartDefaults();
  if (state.data) renderAll(state.data);
}

if (el.themeIcon) el.themeIcon.innerHTML = currentTheme() === "light" ? MOON_SVG : SUN_SVG;
if (el.themeToggle) {
  el.themeToggle.addEventListener("click", () => {
    setTheme(currentTheme() === "light" ? "dark" : "light");
  });
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
  try { if (localStorage.getItem("rag-theme")) return; } catch (_) {}
  setTheme(matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
});

// ── Window selector ──────────────────────────────────────────────────────
el.segButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const days = +btn.dataset.days;
    if (days === state.windowDays) return;
    state.windowDays = days;
    el.segButtons.forEach((b) => {
      const active = +b.dataset.days === days;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    destroyAllCharts();
    fetchSnapshot();
  });
});

// ── Live toggle ──────────────────────────────────────────────────────────
function setLiveState(s, label) {
  if (!el.liveToggle) return;
  el.liveToggle.dataset.state = s;
  if (el.liveLabel) el.liveLabel.textContent = label;
}

if (el.liveToggle) {
  el.liveToggle.addEventListener("click", () => {
    state.paused = !state.paused;
    el.liveToggle.setAttribute("aria-pressed", String(state.paused));
    if (state.paused) {
      setLiveState("paused", "pausado");
      if (state.poll) { clearTimeout(state.poll); state.poll = null; }
    } else {
      setLiveState("off", "reconectando…");
      fetchSnapshot();
      startPolling();
    }
  });
}

// ── Collapsible sections ─────────────────────────────────────────────────
el.collapseButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const id = btn.dataset.collapse;
    if (!id) return;
    const body = document.getElementById(id);
    if (!body) return;
    const collapsed = body.classList.toggle("collapsed");
    btn.setAttribute("aria-expanded", String(!collapsed));
    btn.textContent = collapsed ? "+" : "−";
  });
});

// ── Visibility / cleanup ─────────────────────────────────────────────────
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    _hiddenSince = Date.now();
  } else if (!state.paused) {
    _hiddenSince = null;
    fetchSnapshot();
    if (!state.poll) startPolling();
  }
});

window.addEventListener("beforeunload", () => {
  try { if (state.poll) clearTimeout(state.poll); } catch (_) {}
});

// ── Helpers ──────────────────────────────────────────────────────────────
function announceStatus(msg) {
  const node = document.getElementById("finance-status");
  if (node) node.textContent = msg;
}

function nowHM() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function fmtMoneyARS(v) {
  if (v == null || isNaN(v)) return "—";
  return "$" + Math.abs(Number(v)).toLocaleString("es-AR", { maximumFractionDigits: 0 });
}

function fmtMoneyARSCompact(v) {
  if (v == null || isNaN(v)) return "—";
  const abs = Math.abs(Number(v));
  if (abs >= 1_000_000) return "$" + (abs / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (abs >= 1_000) return "$" + (abs / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
  return "$" + abs.toFixed(0);
}

function fmtMoneyARSSigned(v) {
  if (v == null || isNaN(v)) return "—";
  const abs = Math.abs(Number(v)).toLocaleString("es-AR", { maximumFractionDigits: 0 });
  if (v < 0) return "−$" + abs;
  return "$" + abs;
}

function fmtMoneyUSD(v) {
  if (v == null || isNaN(v)) return "—";
  return "U$S" + Math.abs(Number(v)).toLocaleString("es-AR", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function fmtPctSigned(v) {
  if (v == null || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + (v * 100).toFixed(1) + "%";
}

function fmtInt(v) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toLocaleString("es-AR");
}

function fmtMonth(ym) {
  if (!ym || typeof ym !== "string" || ym.length < 7) return ym || "";
  const months = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
  const m = parseInt(ym.slice(5, 7), 10);
  const y = ym.slice(2, 4);
  return `${months[m - 1] || "?"} '${y}`;
}

function fmtDateShort(iso) {
  if (!iso) return "—";
  const months = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
  const parts = iso.split("-");
  if (parts.length !== 3) return iso;
  const m = parseInt(parts[1], 10);
  return `${parts[2]} ${months[m - 1] || "?"}`;
}

function alpha(color, a) {
  if (!color) return `rgba(127,127,127,${a})`;
  const c = color.trim();
  if (c.startsWith("#")) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split("").map((x) => x + x).join("");
    if (hex.length !== 6) return c;
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const parts = m[1].split(",").map((x) => x.trim());
    return `rgba(${parts[0]},${parts[1]},${parts[2]},${a})`;
  }
  return c;
}

// Vertical gradient (top color → bottom transparent) — used for area fills.
function verticalGradient(ctx, color, height) {
  const g = ctx.createLinearGradient(0, 0, 0, height || 240);
  g.addColorStop(0, alpha(color, 0.42));
  g.addColorStop(0.5, alpha(color, 0.16));
  g.addColorStop(1, alpha(color, 0.0));
  return g;
}

// Horizontal gradient (left light → right strong) — used for bar fills.
function horizontalGradient(ctx, colorA, colorB, width) {
  const g = ctx.createLinearGradient(0, 0, width || 600, 0);
  g.addColorStop(0, alpha(colorA, 0.55));
  g.addColorStop(1, alpha(colorB, 0.95));
  return g;
}

function destroyChart(id) {
  const ch = state.charts[id];
  if (ch) {
    try { ch.destroy(); } catch (_) {}
    delete state.charts[id];
  }
}

function destroyAllCharts() {
  Object.keys(state.charts).forEach(destroyChart);
}

function createChart(canvasId, config) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  try {
    const ch = new Chart(canvas.getContext("2d"), config);
    state.charts[canvasId] = ch;
    return ch;
  } catch (e) {
    console.warn(`[finance] failed to render ${canvasId}:`, e);
    return null;
  }
}

// ── Sparklines ───────────────────────────────────────────────────────────
//
// Cada KPI hero tiene un canvas inyectado con la serie de los últimos
// 6 meses. Sin axes, sin grid, sin dots — solo la curva con gradient.

function ensureSparkCanvas(kpiNode, id) {
  let wrap = kpiNode.querySelector(".kpi-spark");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "kpi-spark";
    const cv = document.createElement("canvas");
    cv.id = id;
    cv.setAttribute("aria-hidden", "true");
    wrap.appendChild(cv);
    kpiNode.appendChild(wrap);
    kpiNode.classList.add("has-spark");
  }
  return wrap.querySelector("canvas");
}

function renderSparkline(canvasId, series, colorMode = "neutral") {
  if (!series || !series.length) {
    destroyChart(canvasId);
    return;
  }
  // colorMode: "neutral" (cyan), "expense" (red), "income" (green),
  // "balance" (red if last<0 else green)
  let color = C.cyan;
  if (colorMode === "expense") color = C.red;
  else if (colorMode === "income") color = C.green;
  else if (colorMode === "balance") color = (series[series.length - 1] < 0) ? C.red : C.green;
  else if (colorMode === "neutral") color = C.cyan;

  const labels = series.map((_, i) => String(i));
  createChart(canvasId, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: series,
        borderColor: color,
        backgroundColor: (ctx) => verticalGradient(ctx.chart.ctx, color, ctx.chart.height || 36),
        fill: true,
        borderWidth: 1.6,
        pointRadius: 0,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { display: false, beginAtZero: false },
      },
      elements: { line: { capBezierPoints: true } },
    },
  });
}

// ── KPI rendering ────────────────────────────────────────────────────────
const KPI_MAP = [
  { id: "kpi-expenses-ars",  key: "expenses_ars",  fmt: "ars",  higherIsBetter: false, sparkColor: "expense" },
  { id: "kpi-expenses-usd",  key: "expenses_usd",  fmt: "usd",  higherIsBetter: false, sparkColor: "expense" },
  { id: "kpi-income-ars",    key: "income_ars",    fmt: "ars",  higherIsBetter: true,  sparkColor: "income"  },
  { id: "kpi-balance-ars",   key: "balance_ars",   fmt: "ars_signed", higherIsBetter: true, sparkColor: "balance" },
  { id: "kpi-txs-count",     key: "txs_count",     fmt: "int",  higherIsBetter: null,  sparkColor: "neutral" },
  { id: "kpi-top-category",  key: "top_category",  fmt: "category_name", higherIsBetter: null, sparkColor: null },
];

function renderKPIs(kpis) {
  if (!kpis) return;
  KPI_MAP.forEach((cfg) => {
    const node = document.getElementById(cfg.id);
    if (!node) return;
    const data = kpis[cfg.key] || {};
    const valEl = node.querySelector(".kpi-value");
    const dEl = node.querySelector(".kpi-delta");

    let valTxt = "—";
    let valSubtitle = null;
    if (data.value != null && !isNaN(data.value)) {
      if (cfg.fmt === "ars")        valTxt = fmtMoneyARS(data.value);
      else if (cfg.fmt === "ars_signed") valTxt = fmtMoneyARSSigned(data.value);
      else if (cfg.fmt === "usd")   valTxt = fmtMoneyUSD(data.value);
      else if (cfg.fmt === "int")   valTxt = fmtInt(data.value);
      else if (cfg.fmt === "category_name") {
        valTxt = data.name || "—";
        if (data.value > 0) valSubtitle = fmtMoneyARS(data.value);
      }
    }
    if (valEl) {
      valEl.textContent = valTxt;
      if (cfg.fmt === "category_name") valEl.style.fontSize = "16px";
    }

    if (data.insufficient === true) {
      node.classList.add("insufficient");
      if (dEl) {
        dEl.textContent = valSubtitle || "—";
        dEl.className = "kpi-delta neutral";
      }
    } else {
      node.classList.remove("insufficient");
      if (dEl) {
        if (cfg.fmt === "category_name") {
          dEl.textContent = valSubtitle || "";
          dEl.className = "kpi-delta neutral";
        } else if (data.fallback_to_latest && data.period_label) {
          dEl.textContent = `último: ${fmtMonth(data.period_label)}`;
          dEl.className = "kpi-delta neutral";
        } else if (data.delta_pct == null) {
          dEl.textContent = "";
          dEl.className = "kpi-delta neutral";
        } else {
          dEl.textContent = fmtPctSigned(data.delta_pct);
          let cls = "neutral";
          if (cfg.higherIsBetter === true)  cls = data.delta_pct >= 0 ? "positive" : "negative";
          if (cfg.higherIsBetter === false) cls = data.delta_pct >= 0 ? "negative" : "positive";
          dEl.className = `kpi-delta ${cls}`;
        }
      }
    }

    // Sparkline (solo si hay serie y colorMode definido)
    if (cfg.sparkColor && Array.isArray(data.spark) && data.spark.length > 1) {
      const sparkId = `spark-${cfg.id}`;
      ensureSparkCanvas(node, sparkId);
      renderSparkline(sparkId, data.spark, cfg.sparkColor);
    }
  });
}

// ── Render: ingresos vs gastos por mes (line con gradient fill) ───────
function renderByMonth(byMonth) {
  if (!byMonth || !byMonth.labels || !byMonth.labels.length) {
    destroyChart("chart-by-month-ars");
    destroyChart("chart-by-month-usd");
    return;
  }
  const labels = byMonth.labels.map(fmtMonth);

  // Si toda la serie de un currency es 0, mostrar empty state en su card.
  const renderEmptyOrChart = (canvasId, incomeData, expenseData, currency) => {
    const sum = (incomeData || []).reduce((s, x) => s + Math.abs(x), 0)
              + (expenseData || []).reduce((s, x) => s + Math.abs(x), 0);
    const wrap = document.getElementById(canvasId)?.closest(".chart-wrap");
    if (sum === 0) {
      destroyChart(canvasId);
      if (wrap && !wrap.querySelector(".empty-state")) {
        wrap.classList.add("is-empty");
        wrap.innerHTML = emptyStateSVG(`Sin movimientos en ${currency} en los últimos 12 meses.`);
      }
      return;
    }
    if (wrap && wrap.classList.contains("is-empty")) {
      wrap.classList.remove("is-empty");
      wrap.innerHTML = `<canvas id="${canvasId}" aria-label="Líneas: ingresos y gastos en ${currency} por mes"></canvas>`;
    }
    createChart(canvasId, mkConfig(incomeData, expenseData, currency));
  };

  const mkConfig = (incomeData, expenseData, currency) => ({
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Ingresos ${currency}`,
          data: incomeData,
          borderColor: C.green,
          backgroundColor: (ctx) => verticalGradient(ctx.chart.ctx, C.green, ctx.chart.height),
          tension: 0.4,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: C.green,
          pointHoverBorderColor: C.bg,
          pointHoverBorderWidth: 2,
          borderWidth: 2,
        },
        {
          label: `Gastos ${currency}`,
          data: expenseData,
          borderColor: C.red,
          backgroundColor: (ctx) => verticalGradient(ctx.chart.ctx, C.red, ctx.chart.height),
          tension: 0.4,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: C.red,
          pointHoverBorderColor: C.bg,
          pointHoverBorderWidth: 2,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { grid: { display: false } },
        y: {
          beginAtZero: true,
          border: { display: false },
          ticks: { callback: (v) => currency === "ARS" ? fmtMoneyARSCompact(v) : fmtMoneyUSD(v) },
        },
      },
      plugins: {
        legend: { position: "top", align: "end", labels: { boxHeight: 8, usePointStyle: true, pointStyle: "circle" } },
        tooltip: {
          displayColors: true,
          callbacks: {
            title: (ctx) => ctx[0].label,
            label: (ctx) => {
              const v = ctx.parsed.y;
              return ` ${ctx.dataset.label}: ${currency === "ARS" ? fmtMoneyARS(v) : fmtMoneyUSD(v)}`;
            },
          },
        },
      },
    },
  });
  renderEmptyOrChart("chart-by-month-ars", byMonth.income_ars, byMonth.expenses_ars, "ARS");
  renderEmptyOrChart("chart-by-month-usd", byMonth.income_usd, byMonth.expenses_usd, "USD");
}

// ── Render: categorías (donut con label central + lista con barras) ─────
function emptyStateSVG(text) {
  return `<div class="empty-state">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="9"/>
      <path d="M12 7v5"/><path d="M12 16h.01"/>
    </svg>
    <h3>Sin gastos en la ventana</h3>
    <p>${text}</p>
  </div>`;
}

function renderCategoryDonut(canvasId, listId, payload, currency) {
  const list = document.getElementById(listId);
  const wrap = document.getElementById(canvasId)?.closest(".chart-wrap");
  if (!payload || !payload.items || !payload.items.length) {
    destroyChart(canvasId);
    if (wrap) {
      wrap.classList.add("is-empty");
      // Replace canvas with empty state once
      if (!wrap.querySelector(".empty-state")) {
        wrap.innerHTML = emptyStateSVG(`No hay gastos en ${currency} dentro de la ventana seleccionada.`);
      }
    }
    if (list) list.innerHTML = "";
    return;
  }

  // Canvas debe existir (re-recrearlo si el empty state lo borró antes).
  if (wrap && wrap.classList.contains("is-empty")) {
    wrap.classList.remove("is-empty");
    wrap.innerHTML = `<canvas id="${canvasId}" aria-label="Donut: gastos por categoría"></canvas>`;
  }

  const items = payload.items.slice(0, 8);
  const rest = payload.items.slice(8);
  const restSum = rest.reduce((s, x) => s + x.amount, 0);
  if (restSum > 0) items.push({ name: `Otros (${rest.length})`, amount: restSum });
  const palette = paletteSeries();
  const colors = items.map((_, i) => palette[i % palette.length]);
  const total = items.reduce((s, x) => s + x.amount, 0) || 1;

  // Wrap para center label.
  if (wrap) wrap.classList.add("donut-wrap");
  let center = wrap?.querySelector(".donut-center");
  if (wrap && !center) {
    center = document.createElement("div");
    center.className = "donut-center";
    wrap.appendChild(center);
  }
  if (center) {
    const fmt = currency === "ARS" ? fmtMoneyARSCompact : fmtMoneyUSD;
    center.innerHTML = `
      <div class="donut-center-label">Total ${currency}</div>
      <div class="donut-center-value">${fmt(total)}</div>
      <div class="donut-center-sub">${items.length - (restSum > 0 ? 1 : 0)} categor${(items.length - (restSum > 0 ? 1 : 0)) === 1 ? "ía" : "ías"}</div>
    `;
  }

  createChart(canvasId, {
    type: "doughnut",
    data: {
      labels: items.map((it) => it.name),
      datasets: [{
        data: items.map((it) => it.amount),
        backgroundColor: colors,
        borderColor: C.bg,
        borderWidth: 3,
        hoverOffset: 10,
        hoverBorderColor: C.bg,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      layout: { padding: 6 },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: true,
          callbacks: {
            label: (ctx) => {
              const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;
              const pct = ((ctx.parsed / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${fmt(ctx.parsed)} · ${pct}%`;
            },
          },
        },
      },
    },
  });

  // Lista con barras + swatch del color.
  if (list) {
    list.innerHTML = "";
    const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;
    items.forEach((it, i) => {
      const li = document.createElement("li");
      li.className = "cat-row";
      const pct = (it.amount / total) * 100;
      const colorIdx = i % palette.length;
      const barCls = PALETTE_CLASSES[colorIdx];
      li.innerHTML = `
        <span class="cat-row-swatch" style="background: ${colors[i]}"></span>
        <span class="cat-row-name" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</span>
        <span class="cat-row-amount">${fmt(it.amount)} · ${pct.toFixed(1)}%</span>
        <span class="cat-row-bar"><span class="cat-row-bar-fill ${barCls}" style="width: ${pct}%"></span></span>
      `;
      list.appendChild(li);
    });
  }
}

// ── Render: top stores (bar horizontal con datalabels) ───────────────────
function renderTopStores(canvasId, payload, currency) {
  const wrap = document.getElementById(canvasId)?.closest(".chart-wrap");
  if (!payload || !payload.items || !payload.items.length) {
    destroyChart(canvasId);
    if (wrap && !wrap.querySelector(".empty-state")) {
      wrap.classList.add("is-empty");
      wrap.innerHTML = emptyStateSVG(`No hay gastos en ${currency} dentro de la ventana seleccionada.`);
    }
    return;
  }
  if (wrap && wrap.classList.contains("is-empty")) {
    wrap.classList.remove("is-empty");
    wrap.innerHTML = `<canvas id="${canvasId}" aria-label="Barras: top comercios"></canvas>`;
  }

  const items = payload.items.slice().reverse();
  const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;

  // Plugin custom para pintar el monto al final de cada barra.
  const dataLabelsPlugin = {
    id: "barEndLabel",
    afterDatasetDraw(chart, args) {
      const { ctx, scales: { y, x }, data } = chart;
      const ds = data.datasets[args.index];
      ctx.save();
      ctx.fillStyle = C.text;
      ctx.font = "11px 'SF Mono', Menlo, monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ds.data.forEach((v, i) => {
        const yC = y.getPixelForValue(i);
        const xC = x.getPixelForValue(v);
        const txt = currency === "ARS" ? fmtMoneyARSCompact(v) : fmtMoneyUSD(v);
        const w = ctx.measureText(txt).width;
        // Si la barra es muy corta, dibujar el texto afuera (a la derecha);
        // sino, adentro (a la izquierda del extremo, en blanco).
        if (xC + w + 8 < x.getPixelForValue(x.max)) {
          ctx.fillStyle = C.text;
          ctx.fillText(txt, xC + 6, yC);
        } else {
          ctx.fillStyle = "#fff";
          ctx.fillText(txt, xC - w - 6, yC);
        }
      });
      ctx.restore();
    },
  };

  createChart(canvasId, {
    type: "bar",
    data: {
      labels: items.map((it) => it.name),
      datasets: [{
        label: `Gasto ${currency}`,
        data: items.map((it) => it.amount),
        backgroundColor: (ctx) => {
          const chart = ctx.chart;
          if (!chart.chartArea) return C.cyan;
          return horizontalGradient(chart.ctx, C.cyan, C.purple, chart.width);
        },
        borderColor: alpha(C.cyan, 0.7),
        borderWidth: 0,
        borderRadius: 6,
        borderSkipped: false,
        barThickness: 16,
        maxBarThickness: 18,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 60 } },
      scales: {
        x: {
          beginAtZero: true,
          border: { display: false },
          grid: { color: C.grid },
          ticks: { callback: (v) => fmt(v), display: true },
        },
        y: {
          border: { display: false },
          grid: { display: false },
          ticks: {
            autoSkip: false,
            callback: function (val) {
              const lab = this.getLabelForValue(val);
              return lab && lab.length > 26 ? lab.slice(0, 25) + "…" : lab;
            },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: false,
          callbacks: {
            title: (ctx) => ctx[0].label,
            label: (ctx) => {
              const it = items[ctx.dataIndex];
              return `${fmt(it.amount)} · ${it.count} ${it.count === 1 ? "compra" : "compras"}`;
            },
          },
        },
      },
    },
    plugins: [dataLabelsPlugin],
  });
}

// ── Render: por cuenta (tabla) ───────────────────────────────────────────
function renderByAccount(payload) {
  const tbody = document.querySelector("#by-account-table tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  const items = (payload && payload.items) || [];
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="muted" style="text-align:center;padding:16px;">Sin movimientos</td></tr>`;
    return;
  }
  items.forEach((it) => {
    const tr = document.createElement("tr");
    const netCls = it.net >= 0 ? "income" : "expense";
    tr.innerHTML = `
      <td>${escapeHtml(it.account || "—")}</td>
      <td class="tx-amount expense">${fmtMoneyARS(it.expenses)}</td>
      <td class="tx-amount income">${fmtMoneyARS(it.income)}</td>
      <td class="tx-amount ${netCls}">${fmtMoneyARSSigned(it.net)}</td>
      <td class="tx-amount">${fmtInt(it.count)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: tarjetas de crédito (visual real) ────────────────────────────
// ── Card breakdown helpers ──────────────────────────────────────────────
//
// Categorización heurística por description del consumo. Match orden-
// dependiente (primer match gana) — definí los más específicos primero.
// El objetivo no es ML, es darle al user un mapa mental rápido del ciclo:
// "¿cuánto se va en streaming vs servicios vs movilidad?".
const _CC_CATEGORIES = [
  { id: "streaming",   label: "Streaming / SaaS",    color: "purple",
    pat: /(netflix|spotify|disney|hbo|claude|openai|chatgpt|google\s*\*?(?:google|youtube)|apple\.com|apple\s+bill|amazon\s+prime|notion|figma|github|cursor|midjourney|anthropic)/i },
  { id: "servicios",   label: "Servicios / facturas", color: "yellow",
    pat: /(epe|edenor|edesur|metrogas|aguas|aysa|claro|movistar|personal|telecom|telecentro|teleplu|fibertel|cablevision|directv|expensas|abl|adventistas)/i },
  { id: "movilidad",   label: "Movilidad",            color: "cyan",
    pat: /(uber|cabify|didi|sube|ypf|shell|axion|esso|peaje|tag|estacion)/i },
  { id: "compras",     label: "Compras / Marketplace", color: "green",
    pat: /(merpago|mercadopago|mercpago|mercado\s*libre|mercadolibre|tiendanube|amazon|aliexpress|nike|adidas|carrefour|coto|disco|jumbo|vea|dia\s|farmacity)/i },
  { id: "comida",      label: "Comida / delivery",    color: "orange",
    pat: /(rappi|pedidos\s*ya|pedidosya|mcdonalds|burger|starbucks|havanna)/i },
  { id: "salud",       label: "Salud",                color: "pink",
    pat: /(farmacit|farmacia|farmacy|swiss\s*medical|osde|galeno|medicus|prepaga|odontolog)/i },
];

function _inferCategory(desc) {
  const s = String(desc || "");
  for (const c of _CC_CATEGORIES) {
    if (c.pat.test(s)) return c;
  }
  return { id: "otros", label: "Otros", color: "" };
}

// Categorización heurística de "otros conceptos" (cargos extra del banco).
// Sellos / IIBB / IVA / Débitos AFIP / IVA RG / etc.
const _CC_EXTRA_CATEGORIES = [
  { id: "sellos",   label: "Impuesto de sellos",    pat: /sellos/i },
  { id: "iibb",     label: "Ingresos brutos",       pat: /iibb|ing(\.|resos)?\s*brutos/i },
  { id: "iva",      label: "IVA",                   pat: /^iva\b|rg\s*4240/i },
  { id: "debitos",  label: "Débitos AFIP / RG",     pat: /db\.?rg|rg\s*5617|d[eé]bito\s*(autom|afip)/i },
  { id: "interes",  label: "Intereses",             pat: /inter[ée]s|financiaci[oó]n/i },
  { id: "comision", label: "Comisiones",            pat: /comisi[oó]n|cargo\s*por|mantenimiento/i },
  { id: "iva_perc", label: "Percepciones IVA",      pat: /percep.*iva|iva.*percep/i },
];

function _inferExtraCategory(desc) {
  const s = String(desc || "");
  for (const c of _CC_EXTRA_CATEGORIES) {
    if (c.pat.test(s)) return c;
  }
  return { id: "otros", label: "Otros cargos" };
}

// Top comercios — normaliza "PAYU*AR*UBER", "Payu*ar*uber" → "Uber".
// Conserva el shortest stable token después de strip de prefijos comunes.
function _normalizeMerchant(desc) {
  if (!desc) return "—";
  let s = String(desc).trim();
  // Strip prefijos típicos: merpago*, payu*ar*, ar*, etc.
  s = s.replace(/^(merpago\*|mercpago\*|mercpag\*|payu\*ar\*|payu\*|ar\*|sp\*|pp\*|mp\*|google\s*\*)/i, "");
  // Cortar al primer separador si hay payment id pegado.
  s = s.split(/\s+\d{8,}/)[0].trim();
  // Capitalizar palabra inicial — el resto queda como viene (el PDF ya está raro).
  if (s.length > 0) s = s[0].toUpperCase() + s.slice(1);
  return s || "—";
}

// Días entre hoy y una fecha ISO (puede ser negativo si la fecha pasó).
function _daysUntil(iso) {
  if (!iso) return null;
  const target = new Date(iso + "T00:00:00");
  if (isNaN(target.getTime())) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const ms = target.getTime() - today.getTime();
  return Math.round(ms / 86_400_000);
}

function _dueBadge(days) {
  if (days == null) return "";
  let cls = "neutral", txt;
  if (days < 0) { cls = "down"; txt = `venció hace ${Math.abs(days)}d`; }
  else if (days === 0) { cls = "down"; txt = "vence hoy"; }
  else if (days <= 3) { cls = "down"; txt = `en ${days}d`; }
  else if (days <= 7) { cls = "neutral"; txt = `en ${days}d`; }
  else { cls = "up"; txt = `en ${days}d`; }
  return `<span class="kpi-delta ${cls}" style="font-size: 11px;">${txt}</span>`;
}

// Bar chart vertical de gasto diario dentro del ciclo. La idea es que el
// user vea cuándo se concentra el gasto (¿fin de mes? ¿días puntuales?).
// Las barras del día 0 hasta last day del ciclo, con $0 para días sin
// compras. Eje X solo muestra día (1-31) — el mes ya está implícito en
// el header "ciclo 2026-04-30 → 2026-05-28".
function _renderCardDailyBars(canvasId, purchases, color) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx || !purchases || !purchases.length) return;

  // Bucketize: { "YYYY-MM-DD": sum }.
  const byDay = new Map();
  for (const p of purchases) {
    if (!p.date) continue;
    byDay.set(p.date, (byDay.get(p.date) || 0) + (p.amount || 0));
  }
  if (!byDay.size) return;

  // Build day range from min(date) to max(date) inclusive, $0 for gaps.
  const dates = [...byDay.keys()].sort();
  const start = new Date(dates[0] + "T00:00:00");
  const end = new Date(dates[dates.length - 1] + "T00:00:00");
  const labels = [];
  const values = [];
  const fullDates = [];
  for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
    const iso = d.toISOString().slice(0, 10);
    fullDates.push(iso);
    labels.push(String(d.getDate()).padStart(2, "0"));
    values.push(byDay.get(iso) || 0);
  }

  state.charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: color || C.cyan,
        borderColor: color || C.cyan,
        borderRadius: 3, borderWidth: 0,
        maxBarThickness: 18,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex;
              return fmtDateShort(fullDates[idx]);
            },
            label: (ctx) => " " + fmtMoneyARS(ctx.parsed.y),
          },
        },
      },
      scales: {
        x: {
          ticks: { color: C.dim, font: { size: 9 }, maxRotation: 0, autoSkip: true, autoSkipPadding: 12 },
          grid: { display: false },
          border: { color: C.border },
        },
        y: {
          ticks: { color: C.dim, font: { size: 9 }, callback: (v) => fmtMoneyARSCompact(v) },
          grid: { color: C.grid },
          border: { display: false },
          beginAtZero: true,
        },
      },
    },
  });
}

// Render donut simple usando Chart.js — devuelve canvas id para mount.
function _renderCardDonut(canvasId, labels, values, colors, centerLabel, centerValue) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  state.charts[canvasId] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: C.card, borderWidth: 2, hoverOffset: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "62%",
      plugins: {
        legend: { display: true, position: "right", labels: { color: C.dim, font: { size: 11 }, boxWidth: 10, boxHeight: 10 } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed;
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total > 0 ? ((v / total) * 100).toFixed(1) : "0";
              return ` ${ctx.label}: ${fmtMoneyARS(v)} (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

function renderCards(cards) {
  const container = document.getElementById("cards-container");
  if (!container) return;
  container.innerHTML = "";
  if (!cards || !cards.length) {
    container.innerHTML = `<div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="2" y="6" width="20" height="14" rx="2"/>
        <path d="M2 11h20"/>
      </svg>
      <h3>Sin resúmenes de tarjeta</h3>
      <p>Cuando aparezca un resumen de tarjeta <code>.xlsx</code> o <code>.pdf</code> en <code>Finances/VISA/</code>, se mostrará acá con todos los consumos del ciclo.</p>
    </div>`;
    return;
  }

  const palette = paletteSeries();

  cards.forEach((card, cardIdx) => {
    const cardKey = `cc-${cardIdx}`;
    const wrap = document.createElement("div");
    wrap.className = "credit-card-wrap";

    // ── Visual de la tarjeta (gradient + chip + brand). Sin cambios.
    const visual = document.createElement("div");
    visual.className = "credit-card-visual";
    const last4 = card.last4 || "0000";
    const brand = (card.brand || "Card").toUpperCase();
    const holder = (card.holder || "Titular").toUpperCase();
    visual.innerHTML = `
      <div class="cc-top">
        <div>
          <div class="cc-bank">obsidian-rag</div>
          <div class="cc-bank" style="opacity: 0.55; font-size: 10px;">resumen del ciclo</div>
        </div>
        <div class="cc-chip" aria-hidden="true"></div>
      </div>
      <div class="cc-number" aria-label="Número de tarjeta enmascarado">
        ••••&nbsp;&nbsp;••••&nbsp;&nbsp;••••&nbsp;&nbsp;${escapeHtml(last4)}
      </div>
      <div class="cc-bottom">
        <div class="cc-holder">
          <small>Titular</small>
          ${escapeHtml(holder)}
        </div>
        <div class="cc-brand">${escapeHtml(brand)}</div>
      </div>
    `;
    wrap.appendChild(visual);

    // ── Detalle al lado (totales, fechas, mínimos, días).
    const detail = document.createElement("div");
    detail.className = "cc-detail";

    const dueDays = _daysUntil(card.due_date);
    const nextDueDays = _daysUntil(card.next_due_date);

    // Card data
    const purchasesAR  = (card.all_purchases_ars || []);
    const purchasesUSD = (card.all_purchases_usd || []);
    const totalConsumosARS = purchasesAR.reduce((s, p) => s + (p.amount || 0), 0);
    const totalConsumosUSD = purchasesUSD.reduce((s, p) => s + (p.amount || 0), 0);
    const otherARS = card.other_charges_total_ars || 0;

    // Header stats — totales + mínimos + due-badge.
    const headerHtml = `
      <div class="cc-totals" style="grid-template-columns: 1fr 1fr;">
        <div class="cc-total">
          <span class="cc-total-label">Total a pagar · ARS</span>
          <span class="cc-total-value">${fmtMoneyARS(card.total_ars)}</span>
          ${card.minimum_ars != null && card.minimum_ars !== card.total_ars ? `
            <span style="font-size: 10px; color: var(--text-faint);">mínimo ${fmtMoneyARS(card.minimum_ars)}</span>` : ""}
        </div>
        <div class="cc-total">
          <span class="cc-total-label">Total a pagar · USD</span>
          <span class="cc-total-value">${fmtMoneyUSD(card.total_usd)}</span>
          ${card.minimum_usd != null && card.minimum_usd !== card.total_usd ? `
            <span style="font-size: 10px; color: var(--text-faint);">mínimo ${fmtMoneyUSD(card.minimum_usd)}</span>` : ""}
        </div>
      </div>
      <div class="cc-meta" style="grid-template-columns: repeat(2, 1fr);">
        <div class="cc-meta-row">
          <span class="cc-meta-label">Vencimiento</span>
          <span class="cc-meta-value">
            ${escapeHtml(fmtDateShort(card.due_date))}
            ${_dueBadge(dueDays)}
          </span>
        </div>
        <div class="cc-meta-row">
          <span class="cc-meta-label">Cierre</span>
          <span class="cc-meta-value">${escapeHtml(fmtDateShort(card.closing_date))}</span>
        </div>
        ${card.next_due_date ? `
        <div class="cc-meta-row">
          <span class="cc-meta-label">Próx. vencimiento</span>
          <span class="cc-meta-value">
            ${escapeHtml(fmtDateShort(card.next_due_date))}
            ${_dueBadge(nextDueDays)}
          </span>
        </div>` : ""}
        ${card.next_closing_date ? `
        <div class="cc-meta-row">
          <span class="cc-meta-label">Próx. cierre</span>
          <span class="cc-meta-value">${escapeHtml(fmtDateShort(card.next_closing_date))}</span>
        </div>` : ""}
      </div>
    `;

    // ── Recurrentes (necesario antes de forecast — fixed monthly los usa).
    const merchantCount = new Map();
    [...purchasesAR, ...purchasesUSD].forEach((p) => {
      const m = _normalizeMerchant(p.description);
      const cur = p.currency || "ARS";
      const key = `${m}|${cur}`;
      const ex = merchantCount.get(key) || { name: m, currency: cur, n: 0, total: 0, items: [] };
      ex.n += 1;
      ex.total += p.amount || 0;
      ex.items.push(p);
      merchantCount.set(key, ex);
    });
    const recurring = [...merchantCount.values()].filter((m) => m.n >= 2)
      .sort((a, b) => b.total - a.total);

    // ── Proyección próximo mes ─────────────────────────────────────────
    //
    // Estimación derivada del ciclo actual + heurística:
    //
    //   FIJO MENSUAL = servicios + streaming + recurrentes detectados
    //                  (lo que se repite todos los meses sí o sí)
    //   VARIABLE    = compras + movilidad + comida + salud + otros
    //                  (lo que cambia según el mes — Uber, marketplace, etc.)
    //
    // Proyección próximo mes ≈ FIJO + asumir VARIABLE similar.
    //
    // El user puede pisar el FIJO con un input manual (persistido en
    // localStorage por last4) cuando sabe que cierta cuota termina o
    // entra una nueva. NO tenemos data de cuotas en el payload del
    // banco — fix sería en `web/finance_dashboard.py` parseando el
    // xlsx con detalle. Por ahora override manual.
    //
    // Categorías "fijas" vs "variables" definidas en sets abajo —
    // categoría id viene de _inferCategory. "Otros" cae en variable
    // por default (asunción conservadora).
    const FIXED_CAT_IDS = new Set(["servicios", "streaming", "salud"]);
    let fixedARS = 0, variableARS = 0;
    purchasesAR.forEach((p) => {
      const cat = _inferCategory(p.description);
      if (FIXED_CAT_IDS.has(cat.id)) fixedARS += (p.amount || 0);
      else variableARS += (p.amount || 0);
    });
    // Para los "Otros" sin categoría, miramos si están en recurrentes
    // detectados (≥2 cobros) → si sí, contar como fijo (ej. Adventistas
    // que no matchea ninguna regex pero claramente es fijo si aparece
    // mensualmente).
    const recurringARSMerchants = new Set(
      recurring.filter((r) => r.currency === "ARS").map((r) => r.name.toLowerCase())
    );
    let movedToFixed = 0;
    purchasesAR.forEach((p) => {
      const cat = _inferCategory(p.description);
      if (cat.id === "otros") {
        const m = _normalizeMerchant(p.description).toLowerCase();
        if (recurringARSMerchants.has(m)) {
          movedToFixed += (p.amount || 0);
        }
      }
    });
    fixedARS += movedToFixed;
    variableARS -= movedToFixed;

    // Override del fijo (localStorage por tarjeta last4).
    const STORAGE_KEY = `cc-fixed-override-${card.last4 || "x"}`;
    let fixedOverride = null;
    try {
      const v = localStorage.getItem(STORAGE_KEY);
      if (v != null && v !== "") {
        const n = parseFloat(v);
        if (!isNaN(n)) fixedOverride = n;
      }
    } catch (_) {}
    const fixedEffective = fixedOverride != null ? fixedOverride : fixedARS;
    const projectedNextARS = fixedEffective + variableARS;

    // Variación vs total este mes (% delta).
    const deltaPct = card.total_ars > 0
      ? ((projectedNextARS - card.total_ars) / card.total_ars) * 100
      : null;
    const deltaCls = deltaPct == null ? "neutral"
      : deltaPct > 5 ? "down"
      : deltaPct < -5 ? "up" : "neutral";
    const deltaTxt = deltaPct == null ? "—"
      : (deltaPct > 0 ? "+" : "") + deltaPct.toFixed(1) + "%";

    const forecastInputId = `${cardKey}-fixed-input`;
    const forecastSection = (purchasesAR.length > 0) ? `
      <div class="cc-section cc-forecast">
        <div class="cc-section-title">Resumen del ciclo + proyección</div>
        <div class="cc-forecast-grid">
          <div class="cc-fcast-cell">
            <span class="cc-fcast-label">Total este mes · ARS</span>
            <span class="cc-fcast-value">${fmtMoneyARS(card.total_ars)}</span>
          </div>
          <div class="cc-fcast-cell">
            <span class="cc-fcast-label">
              Gasto fijo mensual
              <small style="color: var(--text-faint); font-size: 9px;">
                (servicios + streaming + recurrentes)
              </small>
            </span>
            <span class="cc-fcast-value" style="color: var(--yellow);">${fmtMoneyARS(fixedEffective)}</span>
            ${fixedOverride != null ? `
              <small style="font-size: 9px; color: var(--text-faint);">
                override · detectado ${fmtMoneyARS(fixedARS)}
              </small>` : ""}
          </div>
          <div class="cc-fcast-cell">
            <span class="cc-fcast-label">Gasto variable</span>
            <span class="cc-fcast-value" style="color: var(--cyan);">${fmtMoneyARS(variableARS)}</span>
          </div>
          <div class="cc-fcast-cell cc-fcast-projection">
            <span class="cc-fcast-label">Proyección próximo mes</span>
            <span class="cc-fcast-value" style="color: var(--green);">${fmtMoneyARS(projectedNextARS)}</span>
            <small class="kpi-delta ${deltaCls}" style="font-size: 11px;">${deltaTxt} vs este mes</small>
          </div>
        </div>

        <div class="cc-forecast-editor">
          <label for="${forecastInputId}" class="cc-forecast-label">
            Ajustar gasto fijo mensual
            <small style="color: var(--text-faint); font-size: 10px; display: block;">
              Si termina una cuota o agregás un servicio nuevo, ajustá acá.
              Persiste por tarjeta (last4 = ${escapeHtml(card.last4 || "?")}).
            </small>
          </label>
          <div class="cc-forecast-input-row">
            <span class="cc-forecast-prefix">$</span>
            <input type="number" inputmode="numeric" step="1000" id="${forecastInputId}"
                   class="cc-forecast-input"
                   placeholder="${fixedARS.toFixed(0)}"
                   value="${fixedOverride != null ? fixedOverride.toFixed(0) : ""}"
                   aria-label="Gasto fijo mensual override en ARS">
            <button type="button" class="cc-forecast-reset" data-cc-reset="${cardKey}"
                    title="Volver al valor detectado">↺</button>
          </div>
        </div>

        <div class="cc-forecast-note">
          Cuotas en curso no se detectan automáticamente. Si el banco
          incluye una cuota que termina próximamente, restala manualmente
          del fijo. Suba la cifra cuando agregues un servicio nuevo.
        </div>
      </div>
    ` : "";

    // ── Suscripciones (sección dedicada) ────────────────────────────
    //
    // Pulled OUT of "Por categoría" — el user las quiere marcadas aparte
    // porque son los gastos más críticos para forecast (no son discretas,
    // son contractuales). Agrupa por merchant normalizado, suma totales
    // por moneda, muestra #cobros del ciclo.
    //
    // Heurística: cualquier compra que matchea `_CC_CATEGORIES[streaming]`.
    // Si el user agrega un servicio nuevo, basta con sumar el regex en
    // `_CC_CATEGORIES` (top del archivo).
    const subsRaw = [...purchasesAR, ...purchasesUSD]
      .filter((p) => _inferCategory(p.description).id === "streaming");
    const subsByMerchant = new Map();
    subsRaw.forEach((p) => {
      const m = _normalizeMerchant(p.description);
      const cur = p.currency || "ARS";
      const key = `${m}|${cur}`;
      const ex = subsByMerchant.get(key) || { name: m, currency: cur, n: 0, total: 0, items: [] };
      ex.n += 1;
      ex.total += p.amount || 0;
      ex.items.push(p);
      subsByMerchant.set(key, ex);
    });
    const subs = [...subsByMerchant.values()]
      .sort((a, b) => {
        // USD primero porque suelen ser las pesadas (Claude, Netflix);
        // luego por total descendente.
        if (a.currency !== b.currency) return a.currency === "USD" ? -1 : 1;
        return b.total - a.total;
      });
    const subsTotalARS = subs.filter((s) => s.currency === "ARS").reduce((a, b) => a + b.total, 0);
    const subsTotalUSD = subs.filter((s) => s.currency === "USD").reduce((a, b) => a + b.total, 0);
    const subsSection = subs.length ? `
      <div class="cc-section cc-subs">
        <div class="cc-section-title" style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 14px;">▶</span>
          Suscripciones · ${subs.length}
          <span style="margin-left: auto; font-size: 11px; color: var(--text-dim); text-transform: none; letter-spacing: 0; font-weight: 400;">
            ${subsTotalARS > 0 ? fmtMoneyARS(subsTotalARS) : ""}${subsTotalARS > 0 && subsTotalUSD > 0 ? " · " : ""}${subsTotalUSD > 0 ? fmtMoneyUSD(subsTotalUSD) : ""}
          </span>
        </div>
        <ul class="cat-list cc-subs-list">
          ${subs.map((s) => {
            const amount = s.currency === "USD" ? fmtMoneyUSD(s.total) : fmtMoneyARS(s.total);
            const monthly = s.n > 1 ? ` · ${s.currency === "USD" ? fmtMoneyUSD(s.total / s.n) : fmtMoneyARS(s.total / s.n)}/cobro` : "";
            return `
              <li class="cat-row cc-sub-row">
                <span class="cat-row-swatch cc-sub-dot ${s.currency === "USD" ? "usd" : "ars"}"></span>
                <span class="cat-row-name" title="${escapeHtml(s.items.map((i) => i.description).join(' · '))}">
                  ${escapeHtml(s.name)}
                  <small class="cc-sub-meta">${s.n} cobro${s.n === 1 ? "" : "s"} · ${s.currency}${monthly}</small>
                </span>
                <span class="cat-row-amount cc-sub-amount">${amount}</span>
              </li>`;
          }).join("")}
        </ul>
        <div class="cc-subs-footer">
          Estos cargos se asumen recurrentes y forman parte del gasto fijo
          de la proyección. Si cancelaste alguno, restalo del fijo arriba.
        </div>
      </div>
    ` : "";

    // ── Daily bars: gasto ARS por día del ciclo.
    const dailyCanvasId = `${cardKey}-daily`;
    const dailySection = purchasesAR.length >= 3 ? `
      <div class="cc-section">
        <div class="cc-section-title">Gasto diario del ciclo · ARS</div>
        <div class="cc-daily-wrap">
          <canvas id="${dailyCanvasId}" aria-label="Barras: gasto ARS por día del ciclo"></canvas>
        </div>
      </div>
    ` : "";

    // ── Composición ARS: consumos vs cargos extra (donut).
    const compoCanvasId = `${cardKey}-compo`;
    const compoSection = (totalConsumosARS + otherARS) > 0 ? `
      <div class="cc-section">
        <div class="cc-section-title">Composición del resumen · ARS</div>
        <div class="cc-section-grid-2">
          <div class="cc-mini-chart-wrap">
            <canvas id="${compoCanvasId}" aria-label="Donut: composición del resumen ARS"></canvas>
          </div>
          <ul class="cat-list" style="padding-top: 4px;">
            <li class="cat-row">
              <span class="cat-row-swatch" style="background: var(--cyan);"></span>
              <span class="cat-row-name">Consumos del ciclo</span>
              <span class="cat-row-amount">${fmtMoneyARS(totalConsumosARS)}</span>
              <div class="cat-row-bar"><div class="cat-row-bar-fill" style="width: ${((totalConsumosARS / (totalConsumosARS + otherARS)) * 100).toFixed(1)}%;"></div></div>
            </li>
            <li class="cat-row">
              <span class="cat-row-swatch" style="background: var(--yellow);"></span>
              <span class="cat-row-name">Impuestos / cargos extra</span>
              <span class="cat-row-amount">${fmtMoneyARS(otherARS)}</span>
              <div class="cat-row-bar"><div class="cat-row-bar-fill yellow" style="width: ${((otherARS / (totalConsumosARS + otherARS)) * 100).toFixed(1)}%;"></div></div>
            </li>
          </ul>
        </div>
      </div>
    ` : "";

    // ── Por categoría inferida (heuristic).
    // Streaming/SaaS se excluye acá — vive en su propia sección "Suscripciones"
    // arriba para que el user las vea como cargos fijos contractuales aparte
    // del resto del consumo.
    const catAggARS = new Map();
    const catColorMap = new Map();
    purchasesAR.forEach((p) => {
      const c = _inferCategory(p.description);
      if (c.id === "streaming") return;  // ya está en subsSection
      catAggARS.set(c.label, (catAggARS.get(c.label) || 0) + (p.amount || 0));
      catColorMap.set(c.label, c.color || "");
    });
    const catSorted = [...catAggARS.entries()].sort((a, b) => b[1] - a[1]);
    const catSection = catSorted.length ? `
      <div class="cc-section">
        <div class="cc-section-title">Por categoría · ARS</div>
        <ul class="cat-list">
          ${catSorted.map(([name, amount]) => {
            const pct = totalConsumosARS > 0 ? (amount / totalConsumosARS) * 100 : 0;
            const color = catColorMap.get(name) || "";
            return `
              <li class="cat-row">
                <span class="cat-row-swatch" style="background: var(--${color || "cyan"});"></span>
                <span class="cat-row-name">${escapeHtml(name)}</span>
                <span class="cat-row-amount">${fmtMoneyARS(amount)} · ${pct.toFixed(0)}%</span>
                <div class="cat-row-bar"><div class="cat-row-bar-fill ${color}" style="width: ${pct.toFixed(1)}%;"></div></div>
              </li>`;
          }).join("")}
        </ul>
      </div>
    ` : "";

    // ── Recurrentes detectados (mismo merchant ≥2 veces, ARS o USD).
    // `recurring` ya está computado arriba (lo usa el forecast block).
    const recurringSection = recurring.length ? `
      <div class="cc-section">
        <div class="cc-section-title">Recurrentes detectados · ${recurring.length}</div>
        <ul class="cat-list">
          ${recurring.map((m) => `
            <li class="cat-row">
              <span class="cat-row-swatch" style="background: var(--purple);"></span>
              <span class="cat-row-name" title="${escapeHtml(m.items.map((i) => i.description).join(' · '))}">
                ${escapeHtml(m.name)}
                <small style="color: var(--text-faint); margin-left: 6px;">${m.n} cobros</small>
              </span>
              <span class="cat-row-amount">${m.currency === "USD" ? fmtMoneyUSD(m.total) : fmtMoneyARS(m.total)}</span>
            </li>
          `).join("")}
        </ul>
      </div>
    ` : "";

    // ── Cargos extra agrupados.
    const extraAgg = new Map();
    (card.other_charges || []).forEach((c) => {
      const ec = _inferExtraCategory(c.description);
      const ex = extraAgg.get(ec.label) || { label: ec.label, total: 0, items: [] };
      ex.total += c.amount || 0;
      ex.items.push(c);
      extraAgg.set(ec.label, ex);
    });
    const extraSorted = [...extraAgg.values()].sort((a, b) => b.total - a.total);
    const extraSection = extraSorted.length ? `
      <div class="cc-section">
        <div class="cc-section-title">Cargos extra · ${fmtMoneyARS(otherARS)} (${(card.other_charges || []).length} ítems)</div>
        <ul class="cat-list">
          ${extraSorted.map((e) => {
            const pct = otherARS > 0 ? (e.total / otherARS) * 100 : 0;
            return `
              <li class="cat-row">
                <span class="cat-row-swatch" style="background: var(--yellow);"></span>
                <span class="cat-row-name" title="${escapeHtml(e.items.map((i) => i.description).join(' · '))}">
                  ${escapeHtml(e.label)}
                  <small style="color: var(--text-faint); margin-left: 6px;">${e.items.length} ítem${e.items.length === 1 ? "" : "s"}</small>
                </span>
                <span class="cat-row-amount">${fmtMoneyARS(e.total)} · ${pct.toFixed(0)}%</span>
                <div class="cat-row-bar"><div class="cat-row-bar-fill yellow" style="width: ${pct.toFixed(1)}%;"></div></div>
              </li>`;
          }).join("")}
        </ul>
        <details style="margin-top: 8px;">
          <summary class="muted" style="cursor: pointer; font-size: 11px;">Ver desglose línea por línea</summary>
          <ul style="margin-top: 6px; padding-left: 16px; font-size: 11px; color: var(--text-dim);">
            ${(card.other_charges || []).map((c) => `<li>${escapeHtml(c.description || "—")} · ${fmtMoneyARS(c.amount)}</li>`).join("")}
          </ul>
        </details>
      </div>
    ` : "";

    // ── Consumos: tabs ARS / USD.
    const tabARSId = `${cardKey}-tab-ars`;
    const tabUSDId = `${cardKey}-tab-usd`;
    const consumosSection = (purchasesAR.length + purchasesUSD.length) ? `
      <div class="cc-section">
        <div class="cc-section-title">Consumos del ciclo</div>
        <div class="cc-tabs" role="tablist">
          <button class="cc-tab active" type="button" role="tab" data-tab="${tabARSId}"
                  aria-selected="true">ARS · ${purchasesAR.length}</button>
          <button class="cc-tab" type="button" role="tab" data-tab="${tabUSDId}"
                  aria-selected="false">USD · ${purchasesUSD.length}</button>
        </div>
        <div class="cc-tab-panel" id="${tabARSId}">
          ${purchasesAR.length ? `
          <div class="tx-scroll" style="max-height: 280px;">
            <table class="tx-table">
              <thead><tr><th>Fecha</th><th>Comercio</th><th>Categoría</th><th class="tx-amount">Monto</th></tr></thead>
              <tbody>
                ${purchasesAR.slice().sort((a, b) => (b.date || "").localeCompare(a.date || ""))
                  .map((p) => {
                    const cat = _inferCategory(p.description);
                    return `
                    <tr>
                      <td>${escapeHtml(fmtDateShort(p.date))}</td>
                      <td>${escapeHtml(p.description || "—")}</td>
                      <td><span class="tx-cat-pill">${escapeHtml(cat.label)}</span></td>
                      <td class="tx-amount expense">${fmtMoneyARS(p.amount)}</td>
                    </tr>`;
                  }).join("")}
              </tbody>
            </table>
          </div>` : `<div class="muted" style="padding: 12px; font-size: 12px;">Sin consumos en ARS este ciclo.</div>`}
        </div>
        <div class="cc-tab-panel" id="${tabUSDId}" style="display: none;">
          ${purchasesUSD.length ? `
          <div class="tx-scroll" style="max-height: 280px;">
            <table class="tx-table">
              <thead><tr><th>Fecha</th><th>Comercio</th><th>Categoría</th><th class="tx-amount">Monto</th></tr></thead>
              <tbody>
                ${purchasesUSD.slice().sort((a, b) => (b.date || "").localeCompare(a.date || ""))
                  .map((p) => {
                    const cat = _inferCategory(p.description);
                    return `
                    <tr>
                      <td>${escapeHtml(fmtDateShort(p.date))}</td>
                      <td>${escapeHtml(p.description || "—")}</td>
                      <td><span class="tx-cat-pill">${escapeHtml(cat.label)}</span></td>
                      <td class="tx-amount expense">${fmtMoneyUSD(p.amount)}</td>
                    </tr>`;
                  }).join("")}
              </tbody>
            </table>
          </div>` : `<div class="muted" style="padding: 12px; font-size: 12px;">Sin consumos en USD este ciclo.</div>`}
        </div>
      </div>
    ` : "";

    const analysisSections = [
      dailySection,
      compoSection,
      catSection,
      recurringSection,
      extraSection,
    ].filter(Boolean).join("");
    const analysisGroup = analysisSections ? `
      <details class="cc-detail-group">
        <summary>Analítica del resumen</summary>
        <div class="cc-detail-group-body">${analysisSections}</div>
      </details>
    ` : "";
    const consumosGroup = consumosSection ? `
      <details class="cc-detail-group">
        <summary>Consumos del ciclo</summary>
        <div class="cc-detail-group-body">${consumosSection}</div>
      </details>
    ` : "";

    detail.innerHTML = `
      ${headerHtml}
      ${forecastSection}
      ${subsSection}
      ${analysisGroup}
      ${consumosGroup}
      <div class="muted" style="font-size: 11px;">
        Origen: <code>${escapeHtml(card.source_file || "—")}</code>
      </div>
    `;
    wrap.appendChild(detail);
    container.appendChild(wrap);
    detail.querySelectorAll(".cc-detail-group").forEach((group) => {
      group.addEventListener("toggle", () => {
        requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
      });
    });

    // ── Post-mount: charts + handlers.
    if ((totalConsumosARS + otherARS) > 0) {
      _renderCardDonut(
        compoCanvasId,
        ["Consumos", "Cargos extra"],
        [totalConsumosARS, otherARS],
        [palette[0], palette[2]],
      );
    }
    if (purchasesAR.length >= 3) {
      _renderCardDailyBars(dailyCanvasId, purchasesAR, C.cyan);
    }

    // Tab switcher (consumos ARS / USD).
    wrap.querySelectorAll(".cc-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.dataset.tab;
        wrap.querySelectorAll(".cc-tab").forEach((b) => {
          b.classList.toggle("active", b === btn);
          b.setAttribute("aria-selected", b === btn ? "true" : "false");
        });
        wrap.querySelectorAll(".cc-tab-panel").forEach((p) => {
          p.style.display = p.id === targetId ? "" : "none";
        });
      });
    });

    // Forecast override: input + reset. Persiste por last4. Re-render
    // toda la sección cards cuando cambia (más simple que recomputar in-place).
    const input = wrap.querySelector(`#${forecastInputId}`);
    if (input) {
      const commit = () => {
        const v = input.value.trim();
        try {
          if (v === "") localStorage.removeItem(STORAGE_KEY);
          else localStorage.setItem(STORAGE_KEY, String(parseFloat(v)));
        } catch (_) {}
        renderCards(cards);
      };
      input.addEventListener("change", commit);
      input.addEventListener("blur", commit);
    }
    const resetBtn = wrap.querySelector(`[data-cc-reset="${cardKey}"]`);
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
        renderCards(cards);
      });
    }
  });
}

// ── Render: transferencias (avatar + nombre + monto) ─────────────────────
function avatarColor(name) {
  // Hash simple → índice paleta.
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  const palette = paletteSeries();
  return palette[Math.abs(h) % palette.length];
}

function avatarInitials(name) {
  if (!name) return "?";
  const parts = name.trim().split(/[\s,]+/).filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

function renderTransfers(payload, recent) {
  const summary = payload || {};
  const byMonth = summary.by_month || { labels: [], amounts: [] };

  const wrapBM = document.getElementById("chart-transfers-by-month")?.closest(".chart-wrap");
  if (!byMonth.labels.length) {
    destroyChart("chart-transfers-by-month");
    if (wrapBM && !wrapBM.querySelector(".empty-state")) {
      wrapBM.classList.add("is-empty");
      wrapBM.innerHTML = emptyStateSVG("No hay PDFs con transferencias en la carpeta.");
    }
  } else {
    if (wrapBM && wrapBM.classList.contains("is-empty")) {
      wrapBM.classList.remove("is-empty");
      wrapBM.innerHTML = `<canvas id="chart-transfers-by-month"></canvas>`;
    }
    // Line con gradient fill. Para serie temporal con muchos zeros (los
    // meses sin PDF), una línea resalta mejor el trend cuando aparece una
    // transferencia que un mar de bars vacíos. Bar charts con muchos
    // valores en 0 también disparaban un bug de `base: NaN` en Chart.js 4.4.7.
    createChart("chart-transfers-by-month", {
      type: "line",
      data: {
        labels: byMonth.labels.map(fmtMonth),
        datasets: [{
          label: "Total transferido (ARS)",
          data: byMonth.amounts,
          borderColor: C.purple,
          backgroundColor: (ctx) => verticalGradient(ctx.chart.ctx, C.purple, ctx.chart.height),
          borderWidth: 2,
          tension: 0.35,
          fill: true,
          pointRadius: (ctx) => ctx.parsed.y > 0 ? 4 : 0,
          pointBackgroundColor: C.purple,
          pointBorderColor: C.bg,
          pointBorderWidth: 2,
          pointHoverRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, border: { display: false } },
          y: {
            beginAtZero: true,
            border: { display: false },
            ticks: { callback: (v) => fmtMoneyARSCompact(v) },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            callbacks: { label: (ctx) => fmtMoneyARS(ctx.parsed.y) },
          },
        },
      },
    });
  }

  // Top destinatarios.
  const list = document.getElementById("transfers-by-recipient");
  if (list) {
    list.innerHTML = "";
    const items = (summary.by_recipient || []).slice(0, 12);
    if (!items.length) {
      list.innerHTML = `<li class="cat-row"><span class="cat-row-name muted">Sin transferencias detectadas</span></li>`;
    } else {
      const total = items.reduce((s, x) => s + x.amount, 0) || 1;
      const palette = paletteSeries();
      items.forEach((it, i) => {
        const li = document.createElement("li");
        li.className = "cat-row";
        const pct = (it.amount / total) * 100;
        const barCls = PALETTE_CLASSES[i % palette.length];
        const swatchColor = palette[i % palette.length];
        li.innerHTML = `
          <span class="cat-row-swatch" style="background: ${swatchColor}"></span>
          <span class="cat-row-name" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</span>
          <span class="cat-row-amount">${fmtMoneyARS(it.amount)} · ${it.count}×</span>
          <span class="cat-row-bar"><span class="cat-row-bar-fill ${barCls}" style="width: ${pct}%"></span></span>
        `;
        list.appendChild(li);
      });
    }
  }

  // Feed de transferencias recientes con avatares.
  const recentEl = document.getElementById("transfers-recent");
  if (recentEl) {
    recentEl.innerHTML = "";
    const items = (recent || []).slice(0, 30);
    if (!items.length) {
      recentEl.innerHTML = `<div class="muted" style="text-align:center;padding:24px;font-size: 12px;">Sin transferencias detectadas</div>`;
    } else {
      items.forEach((t) => {
        const row = document.createElement("div");
        row.className = "transfer-row";
        const init = avatarInitials(t.recipient);
        const color = avatarColor(t.recipient);
        row.innerHTML = `
          <span class="transfer-avatar" style="background: ${color}; box-shadow: 0 0 0 2px ${alpha(color, 0.18)} inset, 0 1px 0 rgba(255,255,255,0.06) inset;">${escapeHtml(init)}</span>
          <span class="transfer-recipient" title="${escapeHtml(t.recipient)}">${escapeHtml(t.recipient)}</span>
          <span class="transfer-meta">${escapeHtml(fmtDateShort(t.date))}</span>
          <span class="transfer-amount">−${fmtMoneyARS(t.amount)}</span>
        `;
        recentEl.appendChild(row);
      });
    }
  }
}

// ── Render: movimientos recientes (MOZE) ─────────────────────────────────
//
// Asignamos un color stable a cada categoría (hash → índice paleta) para
// el pill — así "House" siempre tiene el mismo color en toda la tabla.
function categoryColorIndex(name) {
  let h = 0;
  for (let i = 0; i < (name || "").length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 7;
}

function renderRecent(items) {
  const tbody = document.querySelector("#recent-table tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (!items || !items.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted" style="text-align:center;padding:16px;">Sin movimientos</td></tr>`;
    return;
  }
  items.forEach((t) => {
    const tr = document.createElement("tr");
    const isExpense = t.type === "expense";
    const cls = isExpense ? "expense" : "income";
    const fmt = (t.currency === "ARS") ? fmtMoneyARS : fmtMoneyUSD;
    const amountText = fmt(t.amount) + (t.currency && t.currency !== "ARS" ? ` ${t.currency}` : "");
    const catColor = categoryColorIndex(t.category || "—");
    tr.innerHTML = `
      <td>${escapeHtml(fmtDateShort(t.date))}</td>
      <td><span class="tx-cat-pill c-${catColor}">${escapeHtml(t.category || "—")}</span></td>
      <td title="${t.note || ""}">${escapeHtml(t.name || "—")}</td>
      <td class="tx-store">${escapeHtml(t.store || "—")}</td>
      <td class="tx-store">${escapeHtml(t.account || "—")}</td>
      <td class="tx-amount ${cls}">${amountText}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: fuentes detectadas ───────────────────────────────────────────
function renderSources(meta) {
  const sumEl = document.getElementById("sources-meta");
  const listEl = document.getElementById("sources-list");
  if (!sumEl || !listEl) return;

  const moze = meta.moze_sources || [];
  const pdfs = meta.pdf_sources || [];
  const cards = meta.card_files || [];

  sumEl.innerHTML = `
    Carpeta: <code>${escapeHtml(meta.finance_dir || "—")}</code><br>
    ${fmtInt(meta.n_transactions)} transacciones MOZE · ${fmtInt(meta.n_transfers)} transferencias PDF · ${fmtInt(meta.n_cards)} resúmenes de tarjeta · ${fmtInt(meta.n_card_transactions || 0)} consumos USD de tarjeta.
  `;

  const sections = [];
  if (moze.length) {
    sections.push(`<div><strong>MOZE CSV</strong> (${moze.length} archivos)</div>`);
    moze.forEach((s) => {
      sections.push(`<div class="muted" style="font-size: 12px; margin-left: 16px;">
        <code>${escapeHtml(s.path)}</code> · ${fmtInt(s.rows_total)} filas · ${fmtInt(s.rows_kept)} kept · ${fmtInt(s.rows_dup)} duplicadas · mtime ${escapeHtml(s.mtime)}
      </div>`);
    });
  }
  if (pdfs.length) {
    sections.push(`<div><strong>PDF</strong> (${pdfs.length} archivos)</div>`);
    pdfs.forEach((s) => {
      sections.push(`<div class="muted" style="font-size: 12px; margin-left: 16px;">
        <code>${escapeHtml(s.path)}</code> · ${fmtInt(s.rows_kept)} transferencias · mtime ${escapeHtml(s.mtime)}
      </div>`);
    });
  }
  if (cards.length) {
    sections.push(`<div><strong>Resúmenes de tarjeta</strong> (${cards.length} archivos)</div>`);
    cards.forEach((path) => {
      sections.push(`<div class="muted" style="font-size: 12px; margin-left: 16px;"><code>${escapeHtml(path)}</code></div>`);
    });
  }
  listEl.innerHTML = sections.join("\n");
}

// ── Util ─────────────────────────────────────────────────────────────────
function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// ── renderAll ────────────────────────────────────────────────────────────
function renderAll(data) {
  if (!data) return;
  renderKPIs(data.kpis);
  renderByMonth(data.by_month);
  renderCategoryDonut("chart-cat-donut-ars", "cat-list-ars", data.by_category_ars, "ARS");
  renderCategoryDonut("chart-cat-donut-usd", "cat-list-usd", data.by_category_usd, "USD");
  renderTopStores("chart-top-stores-ars", data.top_stores_ars, "ARS");
  renderTopStores("chart-top-stores-usd", data.top_stores_usd, "USD");
  renderByAccount(data.by_account);
  renderCards(data.cards);
  renderTransfers(data.transfers, data.transfers_recent);
  renderRecent(data.recent);
  renderSources(data.meta || {});
  if (el.metaPeriod && data.meta) {
    const wd = data.meta.window_days || state.windowDays;
    const usdWd = data.meta.usd_window_days || wd;
    el.metaPeriod.textContent = data.meta.usd_window_fallback
      ? `${wd} días ARS · ${usdWd} días USD · ${data.meta.months || 12}m`
      : `${wd} días · ${data.meta.months || 12}m`;
  }
  if (el.metaUpdated) {
    el.metaUpdated.textContent = `actualizado ${nowHM()}`;
  }
}

// ── Fetch + polling ──────────────────────────────────────────────────────
async function fetchSnapshot() {
  setLiveState("on", "en vivo");
  try {
    const url = `/api/finance?months=12&window_days=${state.windowDays}`;
    const res = await fetch(url, { headers: { "Accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.data = data;
    renderAll(data);
    announceStatus(`Datos de finanzas actualizados a las ${nowHM()}`);
  } catch (e) {
    console.warn("[finance] fetch failed:", e);
    setLiveState("off", "error de conexión");
    announceStatus(`Error al cargar datos: ${e.message || e}`);
  }
}

function startPolling() {
  if (state.poll) clearTimeout(state.poll);
  if (state.paused) return;
  const next = pollNextDelay(POLL_MS);
  state.poll = setTimeout(() => {
    fetchSnapshot().finally(() => startPolling());
  }, next);
}

// ── Bootstrap ────────────────────────────────────────────────────────────
applyChartDefaults();
fetchSnapshot();
startPolling();
