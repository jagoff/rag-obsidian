/* obsidian-rag learning dashboard — Chart.js 4 + vanilla JS + SSE.
 *
 * Hermano gemelo de dashboard.js pero specialized en señales de aprendizaje.
 * Patrón calcado:
 *   - readTokens() / applyChartDefaults() para que Chart.js respete las
 *     CSS variables (cyan/green/yellow/red/purple/orange/pink/grid).
 *   - Polling con setTimeout recursivo + backoff exponencial cuando la
 *     pestaña está hidden (mismo POLL_HIDDEN_GRACE_MS = 5min).
 *   - SSE (`/api/dashboard/learning/stream`) con auto-reconnect en error.
 *   - Theme toggle re-aplica defaults + redraws.
 *   - Selector de ventana (7d/30d/90d/365d) destruye charts y re-fetcha.
 *
 * Sectiones renderizadas (11), ~40 charts en total. Cada renderXxx()
 * acepta el sub-payload y lo redibuja desde cero (destroy + create).
 * Cuando `series.insufficient === true`, en lugar de crear un Chart.js
 * inyecta un <div class="insufficient">Datos insuficientes — N muestras
 * </div> en el chart-card.
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

// Paleta indexable para series sin color asignado (vault types, ranker
// weights, etc.) — orden estable para que el mismo dataset siempre tome
// el mismo color a través de re-renders.
function paletteSeries() {
  return [C.cyan, C.green, C.yellow, C.red, C.purple, C.orange, C.pink];
}

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
const POLL_MS = 60_000;            // cada 60s (visible)
const POLL_MAX_MS = 300_000;       // cap 5 min
const POLL_HIDDEN_GRACE_MS = 300_000;  // 5 min de gracia hidden

const state = {
  days: 30,
  paused: false,
  data: null,
  charts: {},        // chartId → Chart instance
  poll: null,        // setTimeout handle
  evtSrc: null,      // EventSource
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

// SVG icons sun/moon (mismo viewBox que dashboard.js).
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
  // Repintar todos los charts en lugar de destruir + recrear: más
  // barato que re-fetchar y re-procesar todo el payload.
  Object.values(state.charts).forEach((ch) => {
    if (!ch) return;
    try {
      // Recolorear datasets aplicando paleta (los que tienen marker
      // _paletteIdx en el dataset toman re-color desde la paleta nueva).
      ch.data.datasets.forEach((ds) => {
        if (typeof ds._paletteIdx === "number") {
          const col = paletteSeries()[ds._paletteIdx % 7];
          ds.borderColor = col;
          ds.backgroundColor = ds._areaFill ? alpha(col, 0.18) : col;
          if (ds.pointBackgroundColor) ds.pointBackgroundColor = col;
        }
      });
      ch.update("none");
    } catch (_) {}
  });
}

// Inicializa icon + listener.
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

// ── Window selector (segmented) ──────────────────────────────────────────
el.segButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const days = +btn.dataset.days;
    if (days === state.days) return;
    state.days = days;
    el.segButtons.forEach((b) => {
      const active = +b.dataset.days === days;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    // Window cambió → invalidamos charts + re-fetch.
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
      if (state.evtSrc) { state.evtSrc.close(); state.evtSrc = null; }
      if (state.poll) { clearTimeout(state.poll); state.poll = null; }
    } else {
      setLiveState("off", "reconectando…");
      fetchSnapshot();
      startPolling();
      startStream();
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
    if (state.evtSrc) { state.evtSrc.close(); state.evtSrc = null; }
  } else if (!state.paused) {
    _hiddenSince = null;
    fetchSnapshot();
    if (!state.poll) startPolling();
    startStream();
  }
});

window.addEventListener("beforeunload", () => {
  try { if (state.evtSrc) state.evtSrc.close(); } catch (_) {}
  try { if (state.poll) clearTimeout(state.poll); } catch (_) {}
});

// ── Helpers ──────────────────────────────────────────────────────────────
function announceStatus(msg) {
  const node = document.getElementById("learning-status");
  if (node) node.textContent = msg;
}

function nowHM() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function shortDate(d) {
  if (!d) return "";
  if (d instanceof Date) {
    return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  }
  // ISO YYYY-MM-DD or YYYY-MM-DDTHH...
  const s = String(d).split("T")[0];
  const parts = s.split("-");
  if (parts.length === 3) return `${parts[1]}-${parts[2]}`;
  return s;
}

function fmtPct(v) {
  if (v == null || isNaN(v)) return "—";
  return (v * 100).toFixed(1) + "%";
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

function fmtIntSigned(v) {
  if (v == null || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + fmtInt(v);
}

function fmtDecimal(v, digits = 1) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function fmtDecimalSigned(v, digits = 1) {
  if (v == null || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + Number(v).toFixed(digits);
}

// Convierte hex/rgb a rgba con alpha. Acepta "#rrggbb", "#rgb", o
// "rgb(r,g,b)" / "rgba(r,g,b,a)".
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

// Destruye un Chart.js por id si existe.
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

// Pinta o saca el placeholder "datos insuficientes — N muestras" en el
// chart-card al que pertenece el canvas con id `canvasId`. Devuelve true
// si se pintó el placeholder (= no crear chart) y false si no.
function maybeInsufficient(canvasId, payload) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return false;
  const card = canvas.closest(".chart-card") || canvas.parentElement;
  // Limpiar placeholder previo si lo hubo.
  const prev = card && card.querySelector(".insufficient[data-placeholder]");
  if (prev) prev.remove();

  if (!payload || payload.insufficient !== true) return false;

  destroyChart(canvasId);
  // Ocultamos el chart-wrap; pintamos placeholder como hermano.
  const wrap = canvas.closest(".chart-wrap");
  if (wrap) wrap.style.display = "none";
  const div = document.createElement("div");
  div.className = "insufficient";
  div.setAttribute("data-placeholder", "1");
  const n = payload.n_samples != null ? `${payload.n_samples} muestras` : "sin muestras";
  div.innerHTML = `<span>Datos insuficientes para tendencia</span><span class="ins-n">${n}</span>`;
  if (card) card.appendChild(div);
  return true;
}

// Resetea visibilidad del canvas si previamente se ocultó por insufficient.
function resetCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const wrap = canvas.closest(".chart-wrap");
  if (wrap) wrap.style.display = "";
  return canvas;
}

// Wrapper estandar: creamos un Chart con id = canvasId, registrando
// la instancia en state.charts.
function createChart(canvasId, config) {
  destroyChart(canvasId);
  const canvas = resetCanvas(canvasId);
  if (!canvas) return null;
  try {
    const ch = new Chart(canvas.getContext("2d"), config);
    state.charts[canvasId] = ch;
    return ch;
  } catch (e) {
    console.warn(`[learning] failed to render ${canvasId}:`, e);
    return null;
  }
}

// ── KPI rendering ────────────────────────────────────────────────────────
//
// Cada KPI tiene un id en el HTML (kpi-eval-singles, etc.) con tres spans
// dentro: .kpi-value, .kpi-delta, .kpi-badge. Cada entry del payload es
// {value, delta_30d, n_samples, insufficient}.
const KPI_MAP = [
  { id: "kpi-eval-singles",       key: "eval_hit5_singles",         fmt: "pct",   higherIsBetter: true  },
  { id: "kpi-eval-chains",        key: "eval_hit5_chains",          fmt: "pct",   higherIsBetter: true  },
  { id: "kpi-feedback-total",     key: "feedback_total",            fmt: "int",   higherIsBetter: true  },
  { id: "kpi-behavior-per-query", key: "behavior_per_query",        fmt: "decimal", higherIsBetter: true },
  { id: "kpi-cache-hit",          key: "cache_hit_rate",            fmt: "pct",   higherIsBetter: true  },
  { id: "kpi-paraphrases",        key: "paraphrases_count",         fmt: "int",   higherIsBetter: true  },
  { id: "kpi-entities",           key: "entities_count",            fmt: "int",   higherIsBetter: true  },
  { id: "kpi-contradictions",     key: "contradictions_resolved_pct", fmt: "pct", higherIsBetter: true  },
];

// ── Veredicto (estado de los 12 sistemas de aprendizaje) ────────────────
//
// Backend: `verdict()` en web/learning_queries.py devuelve:
//   { summary: {alive, stale, dormant, total},
//     systems: [{id, name, status, last_active_human, metric, note}, …] }
// Origen: el diagnóstico manual del 2026-04-26 que detectó loop roto en
// anticipatory + 3 sistemas dormidos. La idea es que ese diagnóstico ya
// no haya que correrlo a mano — vive permanente en el dashboard. Si algo
// se rompe se ve en rojo arriba de todo.

function renderVerdict(v) {
  if (!v) return;
  const summary = v.summary || {};
  const systems = Array.isArray(v.systems) ? v.systems : [];

  // Counts visibles (verde/amarillo/rojo).
  const alive = summary.alive || 0;
  const stale = summary.stale || 0;
  const dormant = summary.dormant || 0;
  const total = summary.total || systems.length || 0;

  const setText = (id, txt) => {
    const node = document.getElementById(id);
    if (node) node.textContent = txt;
  };
  setText("verdict-count-alive", String(alive));
  setText("verdict-count-stale", String(stale));
  setText("verdict-count-dormant", String(dormant));

  // Headline narrativa: el sistema "está aprendiendo" si la mayoría de los
  // tracks están vivos. Si hay ≥1 dormant que NO sea por "data esperada
  // ausente" (paraphrases, routing rules, audio corrections suelen ser 0 sin
  // ser un bug), se mantiene la afirmación pero se enfatiza el rojo.
  let headline;
  if (alive >= total * 0.6) {
    headline = `El sistema está aprendiendo · ${alive}/${total} tracks activos`;
  } else if (alive >= total * 0.3) {
    headline = `Aprendizaje parcial · ${alive}/${total} tracks activos`;
  } else {
    headline = `Aprendizaje detenido · solo ${alive}/${total} tracks activos`;
  }
  setText("verdict-headline", headline);

  // Grid de tarjetas, una por sistema.
  const grid = document.getElementById("verdict-grid");
  if (!grid) return;
  // Limpiar y rebuildear (cantidad fija de 12, así que no es costoso).
  grid.innerHTML = "";
  systems.forEach((s) => {
    if (!s || !s.id) return;
    const card = document.createElement("div");
    card.className = `vsys vsys-${s.status || "dormant"}`;
    card.setAttribute("data-system", s.id);

    const row1 = document.createElement("div");
    row1.className = "vsys-row1";
    const name = document.createElement("span");
    name.className = "vsys-name";
    name.textContent = s.name || s.id;
    name.title = s.name || s.id;  // tooltip si se trunca
    const when = document.createElement("span");
    when.className = "vsys-when";
    when.textContent = s.last_active_human || "—";
    if (s.last_active_ts) when.title = s.last_active_ts;
    row1.appendChild(name);
    row1.appendChild(when);
    card.appendChild(row1);

    if (s.metric) {
      const metric = document.createElement("span");
      metric.className = "vsys-metric";
      metric.textContent = s.metric;
      card.appendChild(metric);
    }
    if (s.note) {
      const note = document.createElement("span");
      note.className = "vsys-note";
      note.textContent = s.note;
      card.appendChild(note);
    }
    grid.appendChild(card);
  });
}

function renderKPIs(kpis) {
  if (!kpis) return;
  KPI_MAP.forEach((cfg) => {
    const node = document.getElementById(cfg.id);
    if (!node) return;
    const data = kpis[cfg.key] || {};
    const valEl = node.querySelector(".kpi-value");
    const dEl = node.querySelector(".kpi-delta");

    // Valor formateado.
    let valTxt = "—";
    if (data.value != null && !isNaN(data.value)) {
      if (cfg.fmt === "pct")     valTxt = fmtPct(data.value);
      else if (cfg.fmt === "int")     valTxt = fmtInt(data.value);
      else                            valTxt = fmtDecimal(data.value, 1);
    }
    if (valEl) valEl.textContent = valTxt;

    // Insufficient badge.
    if (data.insufficient === true) {
      node.classList.add("insufficient");
      if (dEl) {
        dEl.textContent = "—";
        dEl.className = "kpi-delta neutral";
      }
      return;
    }
    node.classList.remove("insufficient");

    // Delta + arrow.
    const delta = data.delta_30d;
    if (dEl) {
      if (delta == null || isNaN(delta)) {
        dEl.textContent = "—";
        dEl.className = "kpi-delta neutral";
      } else {
        let arrow = "—";
        let cls = "neutral";
        if (delta > 0) { arrow = "▲"; cls = cfg.higherIsBetter ? "up" : "down"; }
        else if (delta < 0) { arrow = "▼"; cls = cfg.higherIsBetter ? "down" : "up"; }
        let deltaTxt;
        if (cfg.fmt === "pct")          deltaTxt = fmtPctSigned(delta);
        else if (cfg.fmt === "int")     deltaTxt = fmtIntSigned(delta);
        else                            deltaTxt = fmtDecimalSigned(delta, 1);
        dEl.textContent = `${arrow} ${deltaTxt} · ${state.days}d`;
        dEl.className = `kpi-delta ${cls}`;
      }
    }
  });
}

// ── Section: Retrieval Quality ───────────────────────────────────────────
function renderRetrievalQuality(payload) {
  if (!payload) return;

  // 1. Eval over time — line chart 4 series.
  const eot = payload.eval_over_time || {};
  if (!maybeInsufficient("chart-eval-over-time", eot)) {
    const series = eot.series || [];
    const labels = series.map((p) => shortDate(p.ts));
    createChart("chart-eval-over-time", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "hit@5 singles", data: series.map((p) => p.hit5_singles), borderColor: C.cyan,   backgroundColor: alpha(C.cyan, 0.15),   tension: 0.3, pointRadius: 2, _paletteIdx: 0 },
          { label: "hit@5 chains",  data: series.map((p) => p.hit5_chains),  borderColor: C.green,  backgroundColor: alpha(C.green, 0.15),  tension: 0.3, pointRadius: 2, _paletteIdx: 1 },
          { label: "MRR singles",   data: series.map((p) => p.mrr_singles),  borderColor: C.yellow, backgroundColor: alpha(C.yellow, 0.10), tension: 0.3, pointRadius: 2, borderDash: [4, 3], _paletteIdx: 2 },
          { label: "MRR chains",    data: series.map((p) => p.mrr_chains),   borderColor: C.purple, backgroundColor: alpha(C.purple, 0.10), tension: 0.3, pointRadius: 2, borderDash: [4, 3], _paletteIdx: 4 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Tune deltas — bar chart, color rojo si rolled_back.
  const td = payload.tune_deltas || {};
  if (!maybeInsufficient("chart-tune-deltas", td)) {
    const series = td.series || [];
    const labels = series.map((p) => shortDate(p.ts));
    const data   = series.map((p) => p.delta_pct);
    const colors = series.map((p) => (p.rolled_back ? C.red : C.green));
    createChart("chart-tune-deltas", {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Δ% vs baseline",
          data,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { ticks: { callback: (v) => v + "%" } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const p = series[ctx.dataIndex];
            const ncases = p && p.n_cases != null ? ` · n=${p.n_cases}` : "";
            const rolled = p && p.rolled_back ? " · rolled back" : "";
            return `${ctx.parsed.y}%${ncases}${rolled}`;
          } } },
        },
      },
    });
  }

  // 3. Latencia vs top score — scatter.
  const lvs = payload.latency_vs_score || {};
  if (!maybeInsufficient("chart-latency-vs-score", lvs)) {
    const points = (lvs.points || []).map((p) => ({ x: p.t_retrieve, y: p.top_score }));
    createChart("chart-latency-vs-score", {
      type: "scatter",
      data: {
        datasets: [{
          label: "queries",
          data: points,
          backgroundColor: alpha(C.cyan, 0.4),
          borderColor: C.cyan,
          pointRadius: 2,
          _paletteIdx: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "t_retrieve (s)" }, beginAtZero: true },
          y: { title: { display: true, text: "top score" }, beginAtZero: true, max: 1.0 },
        },
        plugins: { legend: { display: false } },
      },
    });
  }
}

// ── Section: Ranker Weights ──────────────────────────────────────────────
//
// Small-multiples: 12 mini-canvases en grid 4×3, una mini-línea por weight
// + línea horizontal punteada en `baseline`. Si insufficient, placeholder.
function renderRankerWeights(payload) {
  const grid = document.getElementById("ranker-weights-grid");
  if (!grid) return;

  // Limpiar mini-canvases existentes (destruimos antes para no leakear).
  Array.from(grid.querySelectorAll("canvas")).forEach((c) => destroyChart(c.id));
  grid.innerHTML = "";

  // Si insufficient, placeholder único.
  if (!payload || payload.insufficient === true) {
    const div = document.createElement("div");
    div.className = "insufficient";
    div.style.gridColumn = "1 / -1";
    div.setAttribute("data-placeholder", "1");
    const n = payload && payload.n_samples != null ? `${payload.n_samples} muestras` : "sin muestras";
    div.innerHTML = `<span>Datos insuficientes para tendencia</span><span class="ins-n">${n}</span>`;
    grid.appendChild(div);
    return;
  }

  const keys = payload.weight_keys || [];
  const series = payload.series || [];
  const baseline = payload.baseline || [];
  const current = payload.current || [];
  const labels = series.map((s) => shortDate(s.ts));

  keys.forEach((key, i) => {
    const cell = document.createElement("div");
    cell.className = "sm-cell";
    const cur = current[i];
    const base = baseline[i];
    cell.innerHTML = `
      <h4 title="${key}">${key}</h4>
      <div class="sm-wrap"><canvas id="rw-${i}"></canvas></div>
      <div class="sm-meta"><span>cur ${cur != null ? cur.toFixed(2) : "—"}</span><span>base ${base != null ? base.toFixed(2) : "—"}</span></div>
    `;
    grid.appendChild(cell);

    const data = series.map((s) => (s.values && s.values[i] != null) ? s.values[i] : null);
    const baseVal = base;
    const baseLine = labels.map(() => baseVal);

    const palette = paletteSeries();
    const col = palette[i % palette.length];

    createChart(`rw-${i}`, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "weight", data, borderColor: col, backgroundColor: alpha(col, 0.15), tension: 0.3, pointRadius: 0, borderWidth: 1.5, _paletteIdx: i },
          { label: "baseline", data: baseLine, borderColor: C.dim, borderDash: [3, 3], pointRadius: 0, borderWidth: 1, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { display: false },
          y: { suggestedMin: -2, suggestedMax: 2, ticks: { font: { size: 9 }, maxTicksLimit: 4 } },
        },
        plugins: { legend: { display: false }, tooltip: { enabled: true } },
      },
    });
  });
}

// ── Section: Score Calibration ───────────────────────────────────────────
function renderScoreCalibration(payload) {
  const grid = document.getElementById("score-calibration-grid");
  if (!grid) return;

  // Destruir charts previos.
  Array.from(grid.querySelectorAll("canvas")).forEach((c) => destroyChart(c.id));
  grid.innerHTML = "";

  const curves = (payload && payload.curves) || [];
  if (curves.length === 0) {
    const div = document.createElement("div");
    div.className = "insufficient";
    div.style.gridColumn = "1 / -1";
    div.setAttribute("data-placeholder", "1");
    div.innerHTML = `<span>Sin curvas calibradas todavía</span><span class="ins-n">0 sources</span>`;
    grid.appendChild(div);
    return;
  }

  curves.forEach((curve, idx) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const trained = curve.trained_at ? shortDate(curve.trained_at) : "—";
    card.innerHTML = `
      <h3>Calibración · ${curve.source || "?"}</h3>
      <div class="chart-wrap"><canvas id="sc-${idx}"></canvas></div>
      <div class="sm-meta" style="font-size:10px;margin-top:6px;">
        <span>trained ${trained}</span>
        <span>n+ ${curve.n_pos != null ? curve.n_pos : "?"} · n− ${curve.n_neg != null ? curve.n_neg : "?"}</span>
      </div>
    `;
    grid.appendChild(card);

    if (curve.insufficient === true) {
      // Placeholder en lugar del canvas.
      const wrap = card.querySelector(".chart-wrap");
      if (wrap) wrap.style.display = "none";
      const div = document.createElement("div");
      div.className = "insufficient";
      div.setAttribute("data-placeholder", "1");
      const n = curve.n_pos != null ? `n+=${curve.n_pos}` : "sin muestras";
      div.innerHTML = `<span>Datos insuficientes para tendencia</span><span class="ins-n">${n}</span>`;
      card.appendChild(div);
      return;
    }

    const raw = curve.raw_knots || [];
    const cal = curve.cal_knots || [];
    const palette = paletteSeries();
    const col = palette[idx % palette.length];

    createChart(`sc-${idx}`, {
      type: "line",
      data: {
        labels: raw.map((v) => v.toFixed(2)),
        datasets: [
          { label: "raw → cal", data: cal, borderColor: col, backgroundColor: alpha(col, 0.15), tension: 0.2, pointRadius: 3, _paletteIdx: idx },
          { label: "y = x (identidad)", data: raw, borderColor: C.dim, borderDash: [4, 3], pointRadius: 0, borderWidth: 1, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "raw score" } },
          y: { title: { display: true, text: "calibrated" }, beginAtZero: true, max: 1.0 },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  });
}

// ── Section: Feedback Explícito ──────────────────────────────────────────
function renderFeedbackExplicit(payload) {
  if (!payload) return;

  // 1. Thumbs over time — stacked area.
  const tot = payload.thumbs_over_time || {};
  if (!maybeInsufficient("chart-thumbs-over-time", tot)) {
    const series = tot.series || [];
    const labels = series.map((p) => shortDate(p.date));
    createChart("chart-thumbs-over-time", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "positivos",   data: series.map((p) => p.positive),   borderColor: C.green,  backgroundColor: alpha(C.green, 0.30),  fill: true, tension: 0.3, pointRadius: 1, _paletteIdx: 1, _areaFill: true },
          { label: "negativos",   data: series.map((p) => p.negative),   borderColor: C.red,    backgroundColor: alpha(C.red, 0.30),    fill: true, tension: 0.3, pointRadius: 1, _paletteIdx: 3, _areaFill: true },
          { label: "correctivos", data: series.map((p) => p.corrective), borderColor: C.yellow, backgroundColor: alpha(C.yellow, 0.30), fill: true, tension: 0.3, pointRadius: 1, _paletteIdx: 2, _areaFill: true },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Corrective cumulative — line.
  const cc = payload.corrective_paths_cumulative || {};
  if (!maybeInsufficient("chart-corrective-cumulative", cc)) {
    const series = cc.series || [];
    createChart("chart-corrective-cumulative", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "acumulado",
          data: series.map((p) => p.cumulative),
          borderColor: C.cyan,
          backgroundColor: alpha(C.cyan, 0.20),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 0,
          _areaFill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 3. By scope — donut.
  const bs = payload.by_scope || {};
  // No "insufficient" en este — siempre rendereamos aunque todos sean 0.
  const labels = ["answer", "retrieval", "both", "unknown"];
  const data = labels.map((k) => bs[k] || 0);
  const total = data.reduce((a, b) => a + b, 0);
  if (total === 0) {
    maybeInsufficient("chart-feedback-by-scope", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-feedback-by-scope");
    // Limpiar placeholder si lo había.
    const card = document.getElementById("chart-feedback-by-scope")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    createChart("chart-feedback-by-scope", {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: [C.cyan, C.green, C.purple, C.dim],
          borderColor: C.card,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "right" } },
      },
    });
  }
}

// ── Section: Feedback Implícito ──────────────────────────────────────────
function renderFeedbackImplicit(payload) {
  if (!payload) return;

  // 1. By source — donut.
  const bs = payload.by_source || {};
  const keys = Object.keys(bs);
  const data = keys.map((k) => bs[k] || 0);
  const total = data.reduce((a, b) => a + b, 0);
  if (total === 0) {
    maybeInsufficient("chart-implicit-by-source", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-implicit-by-source");
    const card = document.getElementById("chart-implicit-by-source")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    const palette = paletteSeries();
    createChart("chart-implicit-by-source", {
      type: "doughnut",
      data: {
        labels: keys,
        datasets: [{
          data,
          backgroundColor: keys.map((_, i) => palette[i % palette.length]),
          borderColor: C.card,
          borderWidth: 2,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
    });
  }

  // 2. Implicit signal rate over time — line.
  const isr = payload.implicit_signal_rate_over_time || {};
  if (!maybeInsufficient("chart-implicit-rate", isr)) {
    const series = isr.series || [];
    createChart("chart-implicit-rate", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "% implícito",
          data: series.map((p) => p.implicit_pct),
          borderColor: C.purple,
          backgroundColor: alpha(C.purple, 0.18),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 4,
          _areaFill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } } },
        plugins: { legend: { display: false } },
      },
    });
  }
}

// ── Section: Behavior ────────────────────────────────────────────────────
function renderBehavior(payload) {
  if (!payload) return;

  // 1. CTR by source over time — multi-line.
  const ctr = payload.ctr_by_source_over_time || {};
  if (!maybeInsufficient("chart-ctr-by-source", ctr)) {
    const sources = ctr.sources || [];
    const series = ctr.series || [];
    const palette = paletteSeries();
    const datasets = sources.map((src, i) => ({
      label: src,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : null),
      borderColor: palette[i % palette.length],
      backgroundColor: alpha(palette[i % palette.length], 0.15),
      tension: 0.3,
      pointRadius: 1,
      _paletteIdx: i,
    }));
    createChart("chart-ctr-by-source", {
      type: "line",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Dwell distribution — bar histogram.
  const dd = payload.dwell_distribution || {};
  if (!maybeInsufficient("chart-dwell-distribution", dd)) {
    const buckets = dd.buckets || [];
    createChart("chart-dwell-distribution", {
      type: "bar",
      data: {
        labels: buckets.map((b) => b.label),
        datasets: [{
          label: "frecuencia",
          data: buckets.map((b) => b.count),
          backgroundColor: alpha(C.cyan, 0.6),
          borderColor: C.cyan,
          borderWidth: 1,
          _paletteIdx: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 3. Top paths — horizontal bar (top 20).
  const tp = payload.top_paths || [];
  if (!Array.isArray(tp) || tp.length === 0) {
    maybeInsufficient("chart-top-paths", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-top-paths");
    const card = document.getElementById("chart-top-paths")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();

    const top = tp.slice(0, 20);
    createChart("chart-top-paths", {
      type: "bar",
      data: {
        labels: top.map((r) => r.path && r.path.length > 50 ? "…" + r.path.slice(-49) : (r.path || "?")),
        datasets: [{
          label: "clicks",
          data: top.map((r) => r.clicks),
          backgroundColor: alpha(C.green, 0.6),
          borderColor: C.green,
          borderWidth: 1,
          _paletteIdx: 1,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const r = top[ctx.dataIndex];
            const dwell = r && r.avg_dwell_s != null ? ` · ${fmtDecimal(r.avg_dwell_s, 1)}s avg` : "";
            return `${ctx.parsed.x} clicks${dwell}`;
          } } },
        },
      },
    });
  }

  // 4. Heatmap DOW × hour — CSS grid (no Chart.js).
  renderHeatmapDowHour(payload.heatmap_dow_hour);
}

function renderHeatmapDowHour(matrix) {
  const grid = document.getElementById("heatmap-dow-hour");
  const axis = document.getElementById("heatmap-hours-axis");
  if (!grid || !axis) return;

  grid.innerHTML = "";
  axis.innerHTML = "";

  const dows = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"];
  // Si no hay matriz o tamaño incorrecto, placeholder dentro del card.
  const valid = Array.isArray(matrix) && matrix.length === 7
                && matrix.every((row) => Array.isArray(row) && row.length === 24);

  if (!valid) {
    grid.innerHTML = '<div class="insufficient" data-placeholder="1" style="grid-column: 1 / -1;"><span>Datos insuficientes para tendencia</span><span class="ins-n">sin muestras</span></div>';
    return;
  }

  // Calcular max global para escalar intensidades 0-4.
  let max = 0;
  matrix.forEach((row) => row.forEach((v) => { if (v > max) max = v; }));

  const intensity = (v) => {
    if (max === 0) return 0;
    const r = v / max;
    if (r === 0) return 0;
    if (r < 0.25) return 1;
    if (r < 0.5) return 2;
    if (r < 0.75) return 3;
    return 4;
  };

  matrix.forEach((row, dow) => {
    const label = document.createElement("div");
    label.className = "heatmap-label";
    label.textContent = dows[dow];
    grid.appendChild(label);
    row.forEach((v, h) => {
      const cell = document.createElement("div");
      cell.className = "heatmap-cell";
      cell.dataset.intensity = String(intensity(v));
      cell.title = `${dows[dow]} ${String(h).padStart(2, "0")}:00 · ${v}`;
      grid.appendChild(cell);
    });
  });

  // Eje horario debajo.
  const spacer = document.createElement("span");
  spacer.className = "heatmap-spacer";
  axis.appendChild(spacer);
  for (let h = 0; h < 24; h++) {
    const t = document.createElement("span");
    t.textContent = (h % 3 === 0) ? String(h) : "";
    axis.appendChild(t);
  }
}

// ── Section: Query Learning ──────────────────────────────────────────────
function renderQueryLearning(payload) {
  if (!payload) return;

  // 1. Paraphrases count over time — line.
  const pc = payload.paraphrases_count_over_time || {};
  if (!maybeInsufficient("chart-paraphrases-count", pc)) {
    const series = pc.series || [];
    createChart("chart-paraphrases-count", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "acumulado",
          data: series.map((p) => p.cumulative),
          borderColor: C.purple,
          backgroundColor: alpha(C.purple, 0.20),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 4,
          _areaFill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 2. Cache hit rate over time — line.
  const chr = payload.cache_hit_rate_over_time || {};
  if (!maybeInsufficient("chart-cache-hit-rate-time", chr)) {
    const series = chr.series || [];
    createChart("chart-cache-hit-rate-time", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "hit rate",
          data: series.map((p) => p.hit_rate),
          borderColor: C.green,
          backgroundColor: alpha(C.green, 0.20),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 1,
          _areaFill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const p = series[ctx.dataIndex];
            const n = p && p.n_queries != null ? ` · ${p.n_queries} queries` : "";
            return `${(ctx.parsed.y * 100).toFixed(1)}%${n}`;
          } } },
        },
      },
    });
  }

  // 3. Top paraphrases — horizontal bar.
  const tp = payload.top_paraphrases || [];
  if (!Array.isArray(tp) || tp.length === 0) {
    maybeInsufficient("chart-top-paraphrases", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-top-paraphrases");
    const card = document.getElementById("chart-top-paraphrases")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    const top = tp.slice(0, 20);
    createChart("chart-top-paraphrases", {
      type: "bar",
      data: {
        labels: top.map((r) => {
          const text = r.paraphrase || r.q_normalized || "?";
          return text.length > 60 ? text.slice(0, 59) + "…" : text;
        }),
        datasets: [{
          label: "hits",
          data: top.map((r) => r.hit_count),
          backgroundColor: alpha(C.cyan, 0.6),
          borderColor: C.cyan,
          borderWidth: 1,
          _paletteIdx: 0,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const r = top[ctx.dataIndex];
            const norm = r && r.q_normalized ? `\norig: ${r.q_normalized}` : "";
            return `${ctx.parsed.x} hits${norm}`;
          } } },
        },
      },
    });
  }
}

// ── Section: Anticipatory ────────────────────────────────────────────────
function renderAnticipatory(payload) {
  if (!payload) return;

  // 1. Candidates by kind over time — stacked bar.
  const ck = payload.candidates_by_kind_over_time || {};
  if (!maybeInsufficient("chart-anti-candidates", ck)) {
    const kinds = ck.kinds || [];
    const series = ck.series || [];
    const palette = paletteSeries();
    const datasets = kinds.map((kind, i) => ({
      label: kind,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : 0),
      backgroundColor: palette[i % palette.length],
      borderColor: palette[i % palette.length],
      _paletteIdx: i,
    }));
    createChart("chart-anti-candidates", {
      type: "bar",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Selection / send rate — multi-line.
  const ssr = payload.selection_send_rate || {};
  if (!maybeInsufficient("chart-anti-selection-send", ssr)) {
    const series = ssr.series || [];
    createChart("chart-anti-selection-send", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [
          { label: "selection",  data: series.map((p) => p.selection_rate), borderColor: C.cyan,  backgroundColor: alpha(C.cyan, 0.15),  tension: 0.3, pointRadius: 1, _paletteIdx: 0 },
          { label: "send",       data: series.map((p) => p.send_rate),      borderColor: C.green, backgroundColor: alpha(C.green, 0.15), tension: 0.3, pointRadius: 1, _paletteIdx: 1 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1.0, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 3. User reactions — donut.
  const ur = payload.user_reactions || {};
  const labels = ["positive", "negative", "mute", "unknown"];
  const data = labels.map((k) => ur[k] || 0);
  const total = data.reduce((a, b) => a + b, 0);
  if (total === 0) {
    maybeInsufficient("chart-anti-reactions", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-anti-reactions");
    const card = document.getElementById("chart-anti-reactions")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    createChart("chart-anti-reactions", {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: [C.green, C.red, C.yellow, C.dim],
          borderColor: C.card,
          borderWidth: 2,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
    });
  }

  // 4. Weights current — horizontal bar.
  const wc = payload.weights_current || {};
  const wkeys = Object.keys(wc);
  if (wkeys.length === 0) {
    maybeInsufficient("chart-anti-weights", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-anti-weights");
    const card = document.getElementById("chart-anti-weights")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    createChart("chart-anti-weights", {
      type: "bar",
      data: {
        labels: wkeys,
        datasets: [{
          label: "weight",
          data: wkeys.map((k) => wc[k]),
          backgroundColor: alpha(C.purple, 0.6),
          borderColor: C.purple,
          borderWidth: 1,
          _paletteIdx: 4,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }
}

// ── Section: Routing Learning ────────────────────────────────────────────
function renderRoutingLearning(payload) {
  if (!payload) return;

  // 1. Decisions by bucket — stacked bar.
  const db = payload.decisions_by_bucket_over_time || {};
  if (!maybeInsufficient("chart-routing-decisions", db)) {
    const buckets = db.buckets || [];
    const series = db.series || [];
    const palette = paletteSeries();
    const datasets = buckets.map((bucket, i) => ({
      label: bucket,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : 0),
      backgroundColor: palette[i % palette.length],
      borderColor: palette[i % palette.length],
      _paletteIdx: i,
    }));
    createChart("chart-routing-decisions", {
      type: "bar",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Active rules count — line.
  const arc = payload.active_rules_count_over_time || {};
  if (!maybeInsufficient("chart-routing-rules", arc)) {
    const series = arc.series || [];
    createChart("chart-routing-rules", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "reglas activas",
          data: series.map((p) => p.count),
          borderColor: C.orange,
          backgroundColor: alpha(C.orange, 0.20),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 5,
          _areaFill: true,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } },
    });
  }

  // 3. Evidence ratio distribution — bar histogram.
  const erd = payload.evidence_ratio_distribution || {};
  if (!maybeInsufficient("chart-routing-evidence", erd)) {
    const buckets = erd.buckets || [];
    createChart("chart-routing-evidence", {
      type: "bar",
      data: {
        labels: buckets.map((b) => b.range),
        datasets: [{
          label: "frecuencia",
          data: buckets.map((b) => b.count),
          backgroundColor: alpha(C.yellow, 0.6),
          borderColor: C.yellow,
          borderWidth: 1,
          _paletteIdx: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }
}

// ── Section: Whisper Learning ────────────────────────────────────────────
function renderWhisperLearning(payload) {
  if (!payload) return;

  // 1. Vocab size by source — donut.
  const vbs = payload.vocab_size_by_source || {};
  const map = vbs.by_source || {};
  const keys = Object.keys(map);
  const data = keys.map((k) => map[k] || 0);
  const total = data.reduce((a, b) => a + b, 0);
  if (total === 0) {
    maybeInsufficient("chart-whisper-vocab-source", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-whisper-vocab-source");
    const card = document.getElementById("chart-whisper-vocab-source")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    const palette = paletteSeries();
    createChart("chart-whisper-vocab-source", {
      type: "doughnut",
      data: {
        labels: keys,
        datasets: [{
          data,
          backgroundColor: keys.map((_, i) => palette[i % palette.length]),
          borderColor: C.card,
          borderWidth: 2,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "right" } } },
    });
  }

  // 2. Corrections by source over time — stacked bar.
  const cor = payload.corrections_by_source_over_time || {};
  if (!maybeInsufficient("chart-whisper-corrections", cor)) {
    const sources = cor.sources || [];
    const series = cor.series || [];
    const palette = paletteSeries();
    const datasets = sources.map((src, i) => ({
      label: src,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : 0),
      backgroundColor: palette[i % palette.length],
      borderColor: palette[i % palette.length],
      _paletteIdx: i,
    }));
    createChart("chart-whisper-corrections", {
      type: "bar",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 3. Avg logprob over time — line (puede ser negativo).
  const alp = payload.avg_logprob_over_time || {};
  if (!maybeInsufficient("chart-whisper-logprob", alp)) {
    const series = alp.series || [];
    createChart("chart-whisper-logprob", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "avg logprob",
          data: series.map((p) => p.avg_logprob),
          borderColor: C.cyan,
          backgroundColor: alpha(C.cyan, 0.15),
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const p = series[ctx.dataIndex];
            const n = p && p.n_transcripts != null ? ` · ${p.n_transcripts} trans` : "";
            return `${ctx.parsed.y.toFixed(3)}${n}`;
          } } },
        },
      },
    });
  }

  // 4. Top vocab — horizontal bar.
  const tv = payload.top_vocab || [];
  if (!Array.isArray(tv) || tv.length === 0) {
    maybeInsufficient("chart-whisper-top-vocab", { insufficient: true, n_samples: 0 });
  } else {
    resetCanvas("chart-whisper-top-vocab");
    const card = document.getElementById("chart-whisper-top-vocab")?.closest(".chart-card");
    const prev = card?.querySelector(".insufficient[data-placeholder]");
    if (prev) prev.remove();
    const top = tv.slice(0, 20);
    createChart("chart-whisper-top-vocab", {
      type: "bar",
      data: {
        labels: top.map((r) => r.term || "?"),
        datasets: [{
          label: "weight",
          data: top.map((r) => r.weight),
          backgroundColor: alpha(C.green, 0.6),
          borderColor: C.green,
          borderWidth: 1,
          _paletteIdx: 1,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => {
            const r = top[ctx.dataIndex];
            const src = r && r.source ? ` · ${r.source}` : "";
            return `${fmtDecimal(ctx.parsed.x, 2)}${src}`;
          } } },
        },
      },
    });
  }
}

// ── Section: Vault Intelligence ──────────────────────────────────────────
function renderVaultIntelligence(payload) {
  if (!payload) return;

  // 1. Entities by type over time — multi-line.
  const ebt = payload.entities_by_type_over_time || {};
  if (!maybeInsufficient("chart-vault-entities-type", ebt)) {
    const types = ebt.types || [];
    const series = ebt.series || [];
    const palette = paletteSeries();
    const datasets = types.map((t, i) => ({
      label: t,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : null),
      borderColor: palette[i % palette.length],
      backgroundColor: alpha(palette[i % palette.length], 0.15),
      tension: 0.3,
      pointRadius: 1,
      _paletteIdx: i,
    }));
    createChart("chart-vault-entities-type", {
      type: "line",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 2. Mentions per day — line.
  const mpd = payload.mentions_per_day || {};
  if (!maybeInsufficient("chart-vault-mentions", mpd)) {
    const series = mpd.series || [];
    createChart("chart-vault-mentions", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [{
          label: "mentions",
          data: series.map((p) => p.count),
          borderColor: C.cyan,
          backgroundColor: alpha(C.cyan, 0.20),
          fill: true,
          tension: 0.3,
          pointRadius: 1,
          _paletteIdx: 0,
          _areaFill: true,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } },
    });
  }

  // 3. Contradictions per week — bar (detected vs resolved).
  const cpw = payload.contradictions_per_week || {};
  if (!maybeInsufficient("chart-vault-contradictions", cpw)) {
    const series = cpw.series || [];
    createChart("chart-vault-contradictions", {
      type: "bar",
      data: {
        labels: series.map((p) => shortDate(p.week_start)),
        datasets: [
          { label: "detectadas", data: series.map((p) => p.detected), backgroundColor: alpha(C.red, 0.6),   borderColor: C.red,   borderWidth: 1, _paletteIdx: 3 },
          { label: "resueltas",  data: series.map((p) => p.resolved), backgroundColor: alpha(C.green, 0.6), borderColor: C.green, borderWidth: 1, _paletteIdx: 1 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 4. Filing by folder — multi-line stacked area.
  const fbf = payload.filing_by_folder_over_time || {};
  if (!maybeInsufficient("chart-vault-filing", fbf)) {
    const folders = fbf.folders || [];
    const series = fbf.series || [];
    const palette = paletteSeries();
    const datasets = folders.map((f, i) => ({
      label: f,
      data: series.map((p) => (p.values && p.values[i] != null) ? p.values[i] : 0),
      borderColor: palette[i % palette.length],
      backgroundColor: alpha(palette[i % palette.length], 0.30),
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      _paletteIdx: i,
      _areaFill: true,
    }));
    createChart("chart-vault-filing", {
      type: "line",
      data: { labels: series.map((p) => shortDate(p.date)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // 5. Surface vs Archive — multi-line.
  const sva = payload.surface_archive_over_time || {};
  if (!maybeInsufficient("chart-vault-surface-archive", sva)) {
    const series = sva.series || [];
    createChart("chart-vault-surface-archive", {
      type: "line",
      data: {
        labels: series.map((p) => shortDate(p.date)),
        datasets: [
          { label: "surface", data: series.map((p) => p.surface), borderColor: C.cyan,  backgroundColor: alpha(C.cyan, 0.15),  tension: 0.3, pointRadius: 1, _paletteIdx: 0 },
          { label: "archive", data: series.map((p) => p.archive), borderColor: C.dim,   backgroundColor: alpha(C.dim, 0.10),   tension: 0.3, pointRadius: 1 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }
}

// ── Render all ───────────────────────────────────────────────────────────
function renderAll(payload) {
  if (!payload) return;
  state.data = payload;
  const meta = payload.meta || {};
  const sections = payload.sections || {};

  // Meta header.
  if (el.metaPeriod) {
    const days = meta.window_days != null ? meta.window_days : state.days;
    el.metaPeriod.textContent = `· ${days}d`;
  }
  if (el.metaUpdated) el.metaUpdated.textContent = `actualizado ${nowHM()}`;

  // Veredicto: estado de los 12 sistemas (alive/stale/dormant).
  // Se renderiza ANTES de las KPIs porque conceptualmente es el "headline".
  renderVerdict(payload.verdict);

  // KPIs + secciones.
  renderKPIs(payload.kpis);
  renderRetrievalQuality(sections.retrieval_quality);
  renderRankerWeights(sections.ranker_weights);
  renderScoreCalibration(sections.score_calibration);
  renderFeedbackExplicit(sections.feedback_explicit);
  renderFeedbackImplicit(sections.feedback_implicit);
  renderBehavior(sections.behavior);
  renderQueryLearning(sections.query_learning);
  renderAnticipatory(sections.anticipatory);
  renderRoutingLearning(sections.routing_learning);
  renderWhisperLearning(sections.whisper_learning);
  renderVaultIntelligence(sections.vault_intelligence);

  announceStatus(`Datos del dashboard de aprendizaje actualizados (ventana ${state.days}d)`);
}

// ── Fetch + polling ──────────────────────────────────────────────────────
async function fetchSnapshot() {
  try {
    const res = await fetch(`/api/dashboard/learning?days=${state.days}`, {
      headers: { "Accept": "application/json" },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    renderAll(payload);
  } catch (err) {
    console.warn("[learning] fetch error:", err);
    if (el.metaUpdated) el.metaUpdated.textContent = `error ${err.message}`;
    announceStatus(`Error al cargar datos del dashboard de aprendizaje: ${err.message}`);
  }
}

function startPolling() {
  if (state.poll) { clearTimeout(state.poll); state.poll = null; }
  const tick = () => {
    if (state.paused) { state.poll = null; return; }
    fetchSnapshot();
    state.poll = setTimeout(tick, pollNextDelay(POLL_MS));
  };
  state.poll = setTimeout(tick, pollNextDelay(POLL_MS));
}

// ── SSE stream ───────────────────────────────────────────────────────────
function startStream() {
  if (state.evtSrc) state.evtSrc.close();
  if (state.paused) return;
  setLiveState("off", "conectando…");

  let src;
  try {
    src = new EventSource("/api/dashboard/learning/stream");
  } catch (e) {
    console.warn("[learning] EventSource unavailable:", e);
    setLiveState("off", "sin stream");
    return;
  }
  state.evtSrc = src;

  src.addEventListener("hello", () => setLiveState("live", "en vivo"));
  src.addEventListener("heartbeat", () => setLiveState("live", "en vivo"));

  src.addEventListener("snapshot", (e) => {
    try {
      const payload = JSON.parse(e.data);
      renderAll(payload);
      setLiveState("live", "en vivo");
    } catch (err) {
      console.warn("[learning] bad snapshot event:", err);
    }
  });

  src.onerror = () => {
    setLiveState("off", "desconectado");
    try { src.close(); } catch (_) {}
    state.evtSrc = null;
    if (!state.paused) setTimeout(startStream, 4000);
  };
}

// ── Boot ─────────────────────────────────────────────────────────────────
applyChartDefaults();
fetchSnapshot();
startPolling();
startStream();
