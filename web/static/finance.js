/* obsidian-rag finance dashboard — Chart.js 4 + vanilla JS.
 *
 * Hermano de dashboard.js / learning.js pero specialized en finanzas
 * personales. Lee `/api/finance?months=12&window_days=30` y pinta:
 *
 *   - 6 KPIs hero (gastos ARS / USD, ingresos, balance, # transacciones,
 *     top categoría)
 *   - Sección "Ingresos vs gastos por mes" — 2 line charts (ARS, USD)
 *   - Sección "Categorías" — 2 donuts (ARS, USD) + listas con barra
 *   - Sección "Top comercios" — 2 bar charts horizontales
 *   - Sección "Por cuenta" — tabla
 *   - Sección "Tarjeta de crédito" — card con detalle del último resumen
 *   - Sección "Transferencias bancarias" — bars + lista de top destinatarios
 *   - Sección "Movimientos recientes" — tabla scrolleable (50 últimas)
 *
 * Patrón calcado de learning.js: readTokens() / applyChartDefaults() para
 * que Chart.js respete las CSS variables, polling cada 60s con backoff
 * cuando hidden, theme toggle re-aplica defaults + redraws, selector de
 * ventana destruye charts + re-fetcha.
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

// Paleta indexable para datasets sin color asignado.
function paletteSeries() {
  return [C.cyan, C.green, C.yellow, C.red, C.purple, C.orange, C.pink];
}

// Nombres de la paleta para clases CSS (.cat-row-bar-fill.green, etc.).
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
  Chart.defaults.scale.grid = { color: C.grid };
  Chart.defaults.animation.duration = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 350;
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

// SVG icons sun/moon — mismo viewBox que dashboard.js / learning.js.
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
  // Re-render todo: las paletas de los donuts dependen de tokens.
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

// ── KPI rendering ────────────────────────────────────────────────────────
//
// Cada KPI tiene 3 spans en el HTML: .kpi-value, .kpi-delta, .kpi-badge.
// Payload: {value, delta_pct, n_samples, insufficient}. Para "Top categoría"
// hay un campo extra `name` que reemplaza el value formato moneda con el
// nombre de la categoría.
const KPI_MAP = [
  { id: "kpi-expenses-ars",  key: "expenses_ars",  fmt: "ars",  higherIsBetter: false },
  { id: "kpi-expenses-usd",  key: "expenses_usd",  fmt: "usd",  higherIsBetter: false },
  { id: "kpi-income-ars",    key: "income_ars",    fmt: "ars",  higherIsBetter: true  },
  { id: "kpi-balance-ars",   key: "balance_ars",   fmt: "ars_signed", higherIsBetter: true },
  { id: "kpi-txs-count",     key: "txs_count",     fmt: "int",  higherIsBetter: null  },
  { id: "kpi-top-category",  key: "top_category",  fmt: "category_name", higherIsBetter: null },
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
      // Para "Top categoría", el subtítulo va con el monto debajo del nombre.
      if (cfg.fmt === "category_name") {
        valEl.style.fontSize = "16px";
      }
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
  });
}

// ── Render: ingresos vs gastos por mes ───────────────────────────────────
function renderByMonth(byMonth) {
  if (!byMonth || !byMonth.labels || !byMonth.labels.length) {
    destroyChart("chart-by-month-ars");
    destroyChart("chart-by-month-usd");
    return;
  }
  const labels = byMonth.labels.map(fmtMonth);
  const mkConfig = (incomeData, expenseData, currency) => ({
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Ingresos ${currency}`,
          data: incomeData,
          borderColor: C.green,
          backgroundColor: alpha(C.green, 0.18),
          tension: 0.3,
          fill: true,
          pointRadius: 3,
          pointBackgroundColor: C.green,
        },
        {
          label: `Gastos ${currency}`,
          data: expenseData,
          borderColor: C.red,
          backgroundColor: alpha(C.red, 0.15),
          tension: 0.3,
          fill: true,
          pointRadius: 3,
          pointBackgroundColor: C.red,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (v) => currency === "ARS" ? fmtMoneyARS(v) : fmtMoneyUSD(v),
          },
        },
      },
      plugins: {
        legend: { position: "top", align: "end" },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              return `${ctx.dataset.label}: ${currency === "ARS" ? fmtMoneyARS(v) : fmtMoneyUSD(v)}`;
            },
          },
        },
      },
    },
  });
  createChart("chart-by-month-ars", mkConfig(byMonth.income_ars, byMonth.expenses_ars, "ARS"));
  createChart("chart-by-month-usd", mkConfig(byMonth.income_usd, byMonth.expenses_usd, "USD"));
}

// ── Render: categorías (donut + lista con barra) ─────────────────────────
function renderCategoryDonut(canvasId, listId, payload, currency) {
  if (!payload || !payload.items || !payload.items.length) {
    destroyChart(canvasId);
    const list = document.getElementById(listId);
    if (list) list.innerHTML = `<li class="cat-row"><span class="cat-row-name muted">Sin gastos en la ventana</span></li>`;
    return;
  }
  const items = payload.items.slice(0, 8);  // top 8 + agrupados
  const rest = payload.items.slice(8);
  const restSum = rest.reduce((s, x) => s + x.amount, 0);
  if (restSum > 0) items.push({ name: `Otros (${rest.length})`, amount: restSum });
  const palette = paletteSeries();
  const colors = items.map((_, i) => palette[i % palette.length]);
  const total = items.reduce((s, x) => s + x.amount, 0) || 1;

  createChart(canvasId, {
    type: "doughnut",
    data: {
      labels: items.map((it) => it.name),
      datasets: [
        {
          data: items.map((it) => it.amount),
          backgroundColor: colors,
          borderColor: C.bg,
          borderWidth: 2,
          hoverOffset: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "60%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;
              const pct = ((ctx.parsed / total) * 100).toFixed(1);
              return `${ctx.label}: ${fmt(ctx.parsed)} · ${pct}%`;
            },
          },
        },
      },
    },
  });

  // Lista con barras.
  const list = document.getElementById(listId);
  if (list) {
    list.innerHTML = "";
    const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;
    items.forEach((it, i) => {
      const li = document.createElement("li");
      li.className = "cat-row";
      const pct = (it.amount / total) * 100;
      const barCls = PALETTE_CLASSES[i % palette.length];
      li.innerHTML = `
        <span class="cat-row-name" title="${it.name}">${escapeHtml(it.name)}</span>
        <span class="cat-row-amount">${fmt(it.amount)} · ${pct.toFixed(1)}%</span>
        <span class="cat-row-bar"><span class="cat-row-bar-fill ${barCls}" style="width: ${pct}%"></span></span>
      `;
      list.appendChild(li);
    });
  }
}

// ── Render: top stores (bar horizontal) ──────────────────────────────────
function renderTopStores(canvasId, payload, currency) {
  if (!payload || !payload.items || !payload.items.length) {
    destroyChart(canvasId);
    return;
  }
  const items = payload.items.slice().reverse();  // chart.js bar horizontal: orden invertido
  const fmt = currency === "ARS" ? fmtMoneyARS : fmtMoneyUSD;
  createChart(canvasId, {
    type: "bar",
    data: {
      labels: items.map((it) => it.name),
      datasets: [
        {
          label: `Gasto ${currency}`,
          data: items.map((it) => it.amount),
          backgroundColor: alpha(C.cyan, 0.7),
          borderColor: C.cyan,
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: (v) => fmt(v) },
        },
        y: {
          ticks: {
            autoSkip: false,
            callback: function (val) {
              // Truncamos labels largos para que no rompan el layout.
              const lab = this.getLabelForValue(val);
              return lab && lab.length > 28 ? lab.slice(0, 27) + "…" : lab;
            },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
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

// ── Render: tarjetas de crédito ──────────────────────────────────────────
function renderCards(cards) {
  const container = document.getElementById("cards-container");
  if (!container) return;
  container.innerHTML = "";
  if (!cards || !cards.length) {
    container.innerHTML = `<div class="empty-state">
      <h3>Sin resúmenes de tarjeta</h3>
      <p>Cuando aparezca un <code>Último resumen - &lt;Marca&gt; &lt;NNNN&gt;.xlsx</code> en la carpeta <code>Finances/</code>, se mostrará acá.</p>
    </div>`;
    return;
  }
  cards.forEach((card) => {
    const el = document.createElement("div");
    el.className = "credit-card";
    const purchasesAR = (card.all_purchases_ars || []);
    const purchasesUSD = (card.all_purchases_usd || []);
    const totalRowsAR = purchasesAR.length;
    const totalRowsUSD = purchasesUSD.length;
    el.innerHTML = `
      <div class="credit-card-hdr">
        <div class="credit-card-brand">${escapeHtml(card.brand || "Tarjeta")}<span class="last4">·· ${escapeHtml(card.last4 || "")}</span></div>
        <div class="credit-card-totals">
          <div class="credit-card-total">
            <span class="credit-card-total-label">Total ARS</span>
            <span class="credit-card-total-value">${fmtMoneyARS(card.total_ars)}</span>
          </div>
          <div class="credit-card-total">
            <span class="credit-card-total-label">Total USD</span>
            <span class="credit-card-total-value">${fmtMoneyUSD(card.total_usd)}</span>
          </div>
        </div>
      </div>
      <div class="credit-card-dates">
        <span class="credit-card-date">Cierre: <strong>${escapeHtml(card.closing_date || "—")}</strong></span>
        <span class="credit-card-date">Vencimiento: <strong>${escapeHtml(card.due_date || "—")}</strong></span>
        ${card.next_closing_date ? `<span class="credit-card-date">Próx cierre: <strong>${escapeHtml(card.next_closing_date)}</strong></span>` : ""}
        ${card.next_due_date ? `<span class="credit-card-date">Próx venc.: <strong>${escapeHtml(card.next_due_date)}</strong></span>` : ""}
        ${card.holder ? `<span class="credit-card-date">Titular: <strong>${escapeHtml(card.holder)}</strong></span>` : ""}
      </div>
      ${(totalRowsAR + totalRowsUSD) ? `
      <div class="tx-scroll" style="max-height: 320px;">
        <table class="tx-table">
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Descripción</th>
              <th>Cuotas</th>
              <th class="tx-amount">Monto</th>
            </tr>
          </thead>
          <tbody>
            ${[...purchasesAR, ...purchasesUSD]
              .sort((a, b) => (b.date || "").localeCompare(a.date || ""))
              .map((p) => `
                <tr>
                  <td>${escapeHtml(p.date || "—")}</td>
                  <td>${escapeHtml(p.description || "—")}</td>
                  <td>${escapeHtml(p.installments || "")}</td>
                  <td class="tx-amount expense">${p.currency === "USD" ? fmtMoneyUSD(p.amount) : fmtMoneyARS(p.amount)}</td>
                </tr>
              `).join("")}
          </tbody>
        </table>
      </div>` : ""}
      ${(card.other_charges && card.other_charges.length) ? `
        <details>
          <summary class="muted" style="cursor: pointer; font-size: 12px;">
            Otros conceptos · ${fmtMoneyARS(card.other_charges_total_ars || 0)} (${card.other_charges.length} ítems)
          </summary>
          <ul style="margin-top: 8px; padding-left: 16px; font-size: 12px; color: var(--text-dim);">
            ${card.other_charges.map((c) => `<li>${escapeHtml(c.description || "—")} · ${fmtMoneyARS(c.amount)}</li>`).join("")}
          </ul>
        </details>
      ` : ""}
      <div class="muted" style="font-size: 11px;">
        Origen: <code>${escapeHtml(card.source_file || "—")}</code>
      </div>
    `;
    container.appendChild(el);
  });
}

// ── Render: transferencias bancarias (PDF) ───────────────────────────────
function renderTransfers(payload, recent) {
  const summary = payload || {};
  const byMonth = summary.by_month || { labels: [], amounts: [] };

  // Bar chart por mes.
  if (!byMonth.labels.length) {
    destroyChart("chart-transfers-by-month");
  } else {
    createChart("chart-transfers-by-month", {
      type: "bar",
      data: {
        labels: byMonth.labels.map(fmtMonth),
        datasets: [
          {
            label: "Total transferido (ARS)",
            data: byMonth.amounts,
            backgroundColor: alpha(C.purple, 0.7),
            borderColor: C.purple,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, ticks: { callback: (v) => fmtMoneyARS(v) } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => fmtMoneyARS(ctx.parsed.y) } },
        },
      },
    });
  }

  // Lista de top destinatarios.
  const list = document.getElementById("transfers-by-recipient");
  if (list) {
    list.innerHTML = "";
    const items = (summary.by_recipient || []).slice(0, 12);
    if (!items.length) {
      list.innerHTML = `<li class="cat-row"><span class="cat-row-name muted">Sin transferencias detectadas</span></li>`;
    } else {
      const total = items.reduce((s, x) => s + x.amount, 0) || 1;
      items.forEach((it, i) => {
        const li = document.createElement("li");
        li.className = "cat-row";
        const pct = (it.amount / total) * 100;
        const barCls = PALETTE_CLASSES[i % 7];
        li.innerHTML = `
          <span class="cat-row-name" title="${it.name}">${escapeHtml(it.name)}</span>
          <span class="cat-row-amount">${fmtMoneyARS(it.amount)} · ${it.count}×</span>
          <span class="cat-row-bar"><span class="cat-row-bar-fill ${barCls}" style="width: ${pct}%"></span></span>
        `;
        list.appendChild(li);
      });
    }
  }

  // Lista de movimientos recientes.
  const recentEl = document.getElementById("transfers-recent");
  if (recentEl) {
    recentEl.innerHTML = "";
    const items = (recent || []).slice(0, 30);
    if (!items.length) {
      recentEl.innerHTML = `<div class="muted" style="text-align:center;padding:16px;">Sin transferencias detectadas</div>`;
    } else {
      items.forEach((t) => {
        const row = document.createElement("div");
        row.className = "transfer-row";
        row.innerHTML = `
          <span class="transfer-date">${escapeHtml(t.date)}</span>
          <span class="transfer-recipient">${escapeHtml(t.recipient)}</span>
          <span class="transfer-amount">${fmtMoneyARSSigned(-t.amount)}</span>
        `;
        recentEl.appendChild(row);
      });
    }
  }
}

// ── Render: movimientos recientes (MOZE) ─────────────────────────────────
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
    tr.innerHTML = `
      <td>${escapeHtml(t.date)}</td>
      <td class="tx-cat">${escapeHtml(t.category || "—")} · ${escapeHtml(t.subcategory || "—")}</td>
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
    ${fmtInt(meta.n_transactions)} transacciones MOZE · ${fmtInt(meta.n_transfers)} transferencias PDF · ${fmtInt(meta.n_cards)} resúmenes de tarjeta.
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
  // meta-period label.
  if (el.metaPeriod && data.meta) {
    const wd = data.meta.window_days || state.windowDays;
    el.metaPeriod.textContent = `${wd} días · ${data.meta.months || 12}m`;
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
