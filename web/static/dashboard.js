/* obsidian-rag dashboard — Chart.js 4 + vanilla JS, real-time via SSE + polling */

const C = {
  cyan: "", green: "", yellow: "", red: "", purple: "", orange: "", pink: "",
  dim: "", border: "", card: "", text: "", textDim: "", grid: "",
};

function readTokens() {
  const s = getComputedStyle(document.documentElement);
  const read = (name) => s.getPropertyValue(name).trim();
  C.cyan    = read("--cyan");
  C.green   = read("--green");
  C.yellow  = read("--yellow");
  C.red     = read("--red");
  C.purple  = read("--purple");
  C.orange  = read("--orange");
  C.pink    = read("--pink");
  C.dim     = read("--text-faint");
  C.border  = read("--border");
  C.card    = read("--bg-card");
  C.text    = read("--text");
  C.textDim = read("--text-dim");
  C.grid    = read("--grid");
}
readTokens();

function applyChartDefaults() {
  Chart.defaults.color = C.textDim;
  Chart.defaults.borderColor = C.border;
  Chart.defaults.font.family = "'SF Mono','Menlo','Monaco','JetBrains Mono',ui-monospace,monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.padding = 14;
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
applyChartDefaults();

// ── State ─────────────────────────────────────────────────────────────────
const POLL_MS = 10_000;        // refresh aggregations every 10s
const TICKER_MAX = 12;         // live events kept on screen

const state = {
  days: 30,
  charts: {},                  // name → Chart instance
  built: false,                // first render done?
  data: null,                  // last full payload
  poll: null,                  // setInterval handle
  evtSrc: null,                // EventSource
  paused: false,
  ticker: [],                  // recent stream events
  liveQueriesToday: 0,         // live increments before next poll
};

const el = {
  content: document.getElementById("content"),
  metaPeriod: document.getElementById("meta-period"),
  metaUpdated: document.getElementById("meta-updated"),
  daysPicker: document.getElementById("days-picker"),
  liveToggle: document.getElementById("live-toggle"),
  liveLabel: document.getElementById("live-label"),
  themeToggle: document.getElementById("theme-toggle"),
  themeIcon: document.getElementById("theme-icon"),
};

const SUN_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>';
const MOON_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

function currentTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit) return explicit;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function setTheme(next) {
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("rag-theme", next); } catch (e) {}
  el.themeIcon.innerHTML = next === "light" ? MOON_SVG : SUN_SVG;
  el.themeToggle.setAttribute("aria-label", next === "light" ? "Cambiar a tema oscuro" : "Cambiar a tema claro");
  readTokens();
  applyChartDefaults();
  // Rebuild charts only after first build; otherwise they don't exist yet.
  if (state.built) {
    Object.values(state.charts).forEach(c => c.destroy());
    state.charts = {};
    state.built = false;
    load(false);
  }
}

// Initial icon
el.themeIcon.innerHTML = currentTheme() === "light" ? MOON_SVG : SUN_SVG;
el.themeToggle.addEventListener("click", () => {
  setTheme(currentTheme() === "light" ? "dark" : "light");
});
// React to OS changes when user hasn't set an explicit preference.
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
  try {
    if (localStorage.getItem("rag-theme")) return;
  } catch (e) {}
  setTheme(matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
});

el.daysPicker.addEventListener("change", () => {
  state.days = +el.daysPicker.value;
  // Keep chart instances alive — just refresh data in place.
  load(false);
});

el.liveToggle.addEventListener("click", () => {
  state.paused = !state.paused;
  el.liveToggle.setAttribute("aria-pressed", String(state.paused));
  if (state.paused) {
    setLiveState("paused", "pausado");
    if (state.evtSrc) { state.evtSrc.close(); state.evtSrc = null; }
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
  } else {
    setLiveState("off", "reconectando…");
    load(false);
    startPolling();
    startStream();
  }
});

document.addEventListener("visibilitychange", () => {
  // Pause polling/stream when tab hidden to avoid burning CPU.
  if (document.hidden) {
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
    if (state.evtSrc) { state.evtSrc.close(); state.evtSrc = null; }
  } else if (!state.paused) {
    load(false);
    startPolling();
    startStream();
  }
});

// Initial boot
load(true);
startPolling();
startStream();

// ── Loading + polling ─────────────────────────────────────────────────────

async function load(showSkeleton) {
  if (showSkeleton && state.built) {
    // Only blow away charts if we explicitly want a hard reset (e.g. error recovery).
    Object.values(state.charts).forEach(c => c.destroy());
    state.charts = {};
    state.built = false;
  }
  try {
    const res = await fetch(`/api/dashboard?days=${state.days}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    state.data = d;
    state.liveQueriesToday = 0;
    el.metaPeriod.textContent = `${d.kpis.total_queries} queries · ${state.days}d`;
    el.metaUpdated.textContent = `actualizado ${nowHM()}`;
    if (!state.built) {
      buildLayout(d);
      state.built = true;
    }
    refresh(d);
  } catch (err) {
    if (!state.built) {
      el.content.innerHTML = `<div class="loading" style="color:var(--red)">error: ${err.message}</div>`;
    } else {
      el.metaUpdated.textContent = `error ${err.message}`;
    }
  }
}

function startPolling() {
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(() => { if (!state.paused) load(false); }, POLL_MS);
}

// ── SSE stream ────────────────────────────────────────────────────────────

function startStream() {
  if (state.evtSrc) state.evtSrc.close();
  if (state.paused) return;
  setLiveState("off", "conectando…");
  const src = new EventSource("/api/dashboard/stream");
  state.evtSrc = src;

  src.addEventListener("hello", () => setLiveState("live", "en vivo"));
  src.addEventListener("heartbeat", () => setLiveState("live", "en vivo"));

  src.addEventListener("query", (e) => onStreamEvent("query", JSON.parse(e.data)));
  src.addEventListener("feedback", (e) => onStreamEvent("feedback", JSON.parse(e.data)));
  src.addEventListener("ambient", (e) => onStreamEvent("ambient", JSON.parse(e.data)));
  src.addEventListener("contradiction", (e) => onStreamEvent("contradiction", JSON.parse(e.data)));

  src.onerror = () => {
    setLiveState("off", "desconectado");
    src.close();
    state.evtSrc = null;
    // backoff + reconnect
    if (!state.paused) setTimeout(startStream, 4000);
  };
}

function onStreamEvent(kind, ev) {
  // Push to ticker
  pushTicker(kind, ev);

  // Bump live counters only on terminal phases — in_flight events are
  // pre-completion notices for the SAME logical query and would double-count.
  const kpiQueries = document.querySelector('[data-kpi="queries"]');
  if (kind === "query" && ev.phase !== "in_flight" && kpiQueries && state.data) {
    state.liveQueriesToday += 1;
    const total = (state.data.kpis.total_queries || 0) + state.liveQueriesToday;
    kpiQueries.textContent = total;
    el.metaPeriod.textContent = `${total} queries · ${state.days}d (live)`;

    // Append to today's bar in queries-per-day chart
    const ch = state.charts.queriesDay;
    if (ch && ev.ts) {
      const today = ev.ts.split("T")[0];
      const label = shortDate(today);
      const labels = ch.data.labels;
      const data = ch.data.datasets[0].data;
      if (labels[labels.length - 1] === label) {
        data[data.length - 1] += 1;
      } else {
        labels.push(label);
        data.push(1);
      }
      ch.update("none");
    }
  }

  if (kind === "feedback") {
    const kpiFb = document.querySelector('[data-kpi="feedback"]');
    if (kpiFb && state.data) {
      const k = state.data.kpis;
      if (ev.rating === 1) k.feedback_positive += 1;
      else if (ev.rating === -1) k.feedback_negative += 1;
      kpiFb.textContent = `${k.feedback_positive}/${k.feedback_negative}`;
    }
    // Force a poll soon to refresh the actionable feedback panel.
    setTimeout(() => { if (!state.paused) load(false); }, 800);
  }

  if (kind === "ambient") {
    const kpiAmb = document.querySelector('[data-kpi="ambient"]');
    if (kpiAmb && state.data) {
      state.data.kpis.ambient_hooks += 1;
      kpiAmb.textContent = state.data.kpis.ambient_hooks;
    }
  }
}

function setLiveState(s, label) {
  el.liveToggle.dataset.state = s;
  el.liveLabel.textContent = label;
}

// ── Layout (built once) ───────────────────────────────────────────────────

function buildLayout(d) {
  el.content.innerHTML = `
    <div class="ticker">
      <div class="ticker-head">
        <h2>Eventos en vivo</h2>
        <span style="font-size:11px;color:var(--text-faint)" id="ticker-meta">esperando…</span>
      </div>
      <div class="ticker-list" id="ticker-list" role="log" aria-live="polite" aria-relevant="additions" aria-label="Eventos en vivo">
        <div class="ticker-empty">sin eventos todavía. cuando llegue una query, aparecerá acá.</div>
      </div>
    </div>

    <div class="kpis" id="kpis"></div>
    <div class="health" id="health"></div>

    <div class="charts">
      <div class="chart-card wide">
        <h2>Queries por dia</h2>
        <div class="chart-wrap"><canvas id="ch-queries-day"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>

      <div class="chart-card wide">
        <h2>Feedback accionable — señales para mejorar el RAG</h2>
        <div id="feedback-panel"></div>
      </div>

      <div class="chart-card">
        <h2>Latencia total (p50 diario)</h2>
        <div class="chart-wrap"><canvas id="ch-latency"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Distribucion de scores</h2>
        <div class="chart-wrap"><canvas id="ch-scores"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Actividad por hora</h2>
        <div class="chart-wrap"><canvas id="ch-hours"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Origen de queries</h2>
        <div class="chart-wrap"><canvas id="ch-sources"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Comandos</h2>
        <div class="chart-wrap"><canvas id="ch-cmds"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Hot topics</h2>
        <div class="chart-wrap"><canvas id="ch-topics"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Score trend (promedio diario)</h2>
        <div class="chart-wrap"><canvas id="ch-score-trend"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Feedback por día (positivo vs negativo)</h2>
        <div class="chart-wrap"><canvas id="ch-feedback-trend"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Ambient hooks por dia</h2>
        <div class="chart-wrap"><canvas id="ch-ambient"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card">
        <h2>Contradicciones por dia</h2>
        <div class="chart-wrap"><canvas id="ch-contra"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </div>
      <div class="chart-card" id="card-index">
        <h2>Index</h2>
        <div id="index-content"></div>
      </div>
      <div class="chart-card" id="card-latency-stats">
        <h2>Latencia (percentiles)</h2>
        <div id="latency-stats-content"></div>
      </div>
      <div class="chart-card" id="card-tune">
        <h2>Historial de tune</h2>
        <div id="tune-content"></div>
      </div>
    </div>
  `;

  // Build chart instances ONCE — refresh() will mutate their data.
  state.charts.queriesDay = new Chart(document.getElementById("ch-queries-day"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.cyan, 0.5), borderColor: C.cyan, borderWidth: 1, borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 45 } }, y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.latency = new Chart(document.getElementById("ch-latency"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "p50", data: [], borderColor: C.cyan, backgroundColor: hexAlpha(C.cyan, 0.1), tension: 0.3, fill: true, pointRadius: 2 },
      { label: "p95", data: [], borderColor: C.yellow, backgroundColor: "transparent", borderDash: [4, 4], tension: 0.3, pointRadius: 2 },
    ] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: "seg" } }, x: { ticks: { maxRotation: 45 } } } },
  });

  state.charts.scores = new Chart(document.getElementById("ch-scores"), {
    type: "bar",
    data: { labels: Array.from({ length: 10 }, (_, i) => `${(i/10).toFixed(1)}`), datasets: [{ data: new Array(10).fill(0), backgroundColor: [], borderColor: [], borderWidth: 1, borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "score" } }, y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.hours = new Chart(document.getElementById("ch-hours"), {
    type: "bar",
    data: { labels: Array.from({ length: 24 }, (_, i) => `${i}h`), datasets: [{ data: new Array(24).fill(0), backgroundColor: [], borderColor: C.cyan, borderWidth: 1, borderRadius: 2 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.sources = new Chart(document.getElementById("ch-sources"), {
    type: "doughnut",
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderColor: C.card, borderWidth: 2 }] },
    options: { responsive: true, cutout: "55%", plugins: { legend: { position: "right" } } },
  });

  state.charts.cmds = new Chart(document.getElementById("ch-cmds"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderColor: [], borderWidth: 1, borderRadius: 3 }] },
    options: { indexAxis: "y", responsive: true, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.topics = new Chart(document.getElementById("ch-topics"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.purple, 0.5), borderColor: C.purple, borderWidth: 1, borderRadius: 3 }] },
    options: { indexAxis: "y", responsive: true, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.scoreTrend = new Chart(document.getElementById("ch-score-trend"), {
    type: "line",
    data: { labels: [], datasets: [{ label: "score promedio", data: [], borderColor: C.green, backgroundColor: hexAlpha(C.green, 0.08), tension: 0.3, fill: true, pointRadius: 3, pointBackgroundColor: [] }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 1.0, title: { display: true, text: "score" } }, x: { ticks: { maxRotation: 45 } } } },
  });

  state.charts.feedbackTrend = new Chart(document.getElementById("ch-feedback-trend"), {
    type: "bar",
    data: { labels: [], datasets: [
      { label: "positivos", data: [], backgroundColor: hexAlpha(C.green, 0.6), borderColor: C.green, borderWidth: 1, borderRadius: 3, stack: "fb" },
      { label: "negativos", data: [], backgroundColor: hexAlpha(C.red, 0.6), borderColor: C.red, borderWidth: 1, borderRadius: 3, stack: "fb" },
    ] },
    options: { responsive: true, scales: { x: { stacked: true, ticks: { maxRotation: 45 } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.ambient = new Chart(document.getElementById("ch-ambient"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.orange, 0.5), borderColor: C.orange, borderWidth: 1, borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });

  state.charts.contra = new Chart(document.getElementById("ch-contra"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.red, 0.5), borderColor: C.red, borderWidth: 1, borderRadius: 3 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });
}

// ── Refresh (mutates existing charts/DOM in place) ────────────────────────

function refresh(d) {
  const k = d.kpis;
  const idx = d.index || {};

  // KPIs
  const kpis = [
    { key: "queries", label: "Queries", value: k.total_queries, cls: "cyan", sub: `${k.total_queries_all_time} all-time` },
    { key: "notes", label: "Notas indexadas", value: idx.notes || "—", cls: "green", sub: `${idx.chunks || 0} chunks` },
    { key: "latency", label: "Latencia promedio", value: k.avg_latency ? `${k.avg_latency}s` : "—", cls: "yellow", sub: `ret ${k.avg_retrieve || "—"}s · gen ${k.avg_generate || "—"}s` },
    { key: "feedback", label: "Feedback +/-", value: `${k.feedback_positive}/${k.feedback_negative}`, cls: "green", sub: k.feedback_positive + k.feedback_negative > 0 ? `${Math.round(k.feedback_positive / (k.feedback_positive + k.feedback_negative) * 100)}% positivo` : "sin feedback" },
    { key: "sessions", label: "Sesiones", value: k.sessions, cls: "purple", sub: `${k.wa_sessions} WhatsApp` },
    { key: "ambient", label: "Ambient hooks", value: k.ambient_hooks, cls: "orange", sub: `${k.ambient_wikilinks} wikilinks aplicados` },
    { key: "contra", label: "Contradicciones", value: k.contradictions_found, cls: "red", sub: "detectadas en indexación" },
    { key: "surface", label: "Surface pairs", value: k.surface_pairs, cls: "cyan", sub: `${d.surface_runs} runs` },
    { key: "tags", label: "Tags", value: idx.tags || "—", cls: "", sub: `${idx.folders || 0} carpetas` },
    { key: "filings", label: "Filing proposals", value: k.filings, cls: "yellow", sub: "en el período" },
  ];

  const kpiContainer = document.getElementById("kpis");
  // Mutate existing DOM when the set of KPIs hasn't changed (preserves focus/hover/tooltips on poll).
  const existingKeys = Array.from(kpiContainer.querySelectorAll(".kpi")).map(n => n.dataset.key).join(",");
  const nextKeys = kpis.map(k => k.key).join(",");
  if (existingKeys === nextKeys && existingKeys !== "") {
    for (const kpi of kpis) {
      const card = kpiContainer.querySelector(`.kpi[data-key="${kpi.key}"]`);
      if (!card) continue;
      const valEl = card.querySelector(".kpi-value");
      const subEl = card.querySelector(".kpi-sub");
      if (valEl) {
        if (valEl.textContent !== String(kpi.value)) valEl.textContent = kpi.value;
        valEl.className = `kpi-value ${kpi.cls}`;
        valEl.dataset.kpi = kpi.key;
      }
      if (subEl && subEl.textContent !== kpi.sub) subEl.textContent = kpi.sub;
    }
  } else {
    kpiContainer.innerHTML = kpis.map(kpi => `
      <div class="kpi" data-key="${kpi.key}">
        <span class="kpi-label">${kpi.label}</span>
        <span class="kpi-value ${kpi.cls}" data-kpi="${kpi.key}">${kpi.value}</span>
        <span class="kpi-sub">${kpi.sub}</span>
      </div>
    `).join("");
  }

  renderHealth(d);
  renderFeedbackPanel(d.feedback || {});

  // Queries per day
  const qDays = Object.keys(d.queries_per_day);
  updateChart("queriesDay", qDays.map(shortDate), [Object.values(d.queries_per_day)]);

  // Latency
  const lDays = Object.keys(d.latency_per_day);
  const lP50 = lDays.map(day => d.latency_per_day[day].p50);
  const lP95 = lDays.map(day => d.latency_per_day[day].p95);
  updateChart("latency", lDays.map(shortDate), [lP50, lP95]);

  // Score distribution
  const scoreColors = d.score_distribution.map((_, i) => i >= 5 ? C.green : i >= 2 ? C.yellow : C.red);
  state.charts.scores.data.datasets[0].data = d.score_distribution;
  state.charts.scores.data.datasets[0].backgroundColor = scoreColors.map(c => hexAlpha(c, 0.6));
  state.charts.scores.data.datasets[0].borderColor = scoreColors;
  state.charts.scores.update("none");
  setEmpty(state.charts.scores, !d.score_distribution.some(v => v > 0));

  // Hours
  const hourVals = Array.from({ length: 24 }, (_, i) => d.hours[String(i)] || 0);
  const maxHour = Math.max(...hourVals, 1);
  state.charts.hours.data.datasets[0].data = hourVals;
  state.charts.hours.data.datasets[0].backgroundColor = hourVals.map(v => hexAlpha(C.cyan, 0.2 + (v / maxHour) * 0.8));
  state.charts.hours.update("none");
  setEmpty(state.charts.hours, !hourVals.some(v => v > 0));

  // Sources
  const srcLabels = Object.keys(d.sources);
  const srcVals = Object.values(d.sources);
  const srcPalette = [C.green, C.cyan, C.purple, C.yellow, C.orange, C.pink];
  state.charts.sources.data.labels = srcLabels;
  state.charts.sources.data.datasets[0].data = srcVals;
  state.charts.sources.data.datasets[0].backgroundColor = srcPalette.slice(0, srcLabels.length).map(c => hexAlpha(c, 0.7));
  state.charts.sources.update("none");
  setEmpty(state.charts.sources, !srcLabels.length);

  // Commands
  const cmdLabels = Object.keys(d.cmds);
  const cmdVals = Object.values(d.cmds);
  const cmdPalette = [C.cyan, C.green, C.yellow, C.purple, C.orange, C.pink, C.red];
  state.charts.cmds.data.labels = cmdLabels;
  state.charts.cmds.data.datasets[0].data = cmdVals;
  state.charts.cmds.data.datasets[0].backgroundColor = cmdLabels.map((_, i) => hexAlpha(cmdPalette[i % 7], 0.6));
  state.charts.cmds.data.datasets[0].borderColor = cmdLabels.map((_, i) => cmdPalette[i % 7]);
  state.charts.cmds.update("none");
  setEmpty(state.charts.cmds, !cmdLabels.length);

  // Topics
  const topicData = (d.hot_topics || []).filter(t => t.count >= 2).slice(0, 12);
  state.charts.topics.data.labels = topicData.map(t => t.topic);
  state.charts.topics.data.datasets[0].data = topicData.map(t => t.count);
  state.charts.topics.update("none");
  setEmpty(state.charts.topics, !topicData.length);

  // Score trend
  const sTrend = (d.health && d.health.score_trend) || {};
  const stDays = Object.keys(sTrend);
  const stVals = Object.values(sTrend);
  state.charts.scoreTrend.data.labels = stDays.map(shortDate);
  state.charts.scoreTrend.data.datasets[0].data = stVals;
  state.charts.scoreTrend.data.datasets[0].pointBackgroundColor = stVals.map(v => v >= 0.2 ? C.green : v >= 0.05 ? C.yellow : C.red);
  state.charts.scoreTrend.update("none");
  setEmpty(state.charts.scoreTrend, !stDays.length);

  // Feedback trend (merge pos+neg day keys)
  const fb = d.feedback || {};
  const fbDays = Array.from(new Set([...Object.keys(fb.per_day_pos || {}), ...Object.keys(fb.per_day_neg || {})])).sort();
  state.charts.feedbackTrend.data.labels = fbDays.map(shortDate);
  state.charts.feedbackTrend.data.datasets[0].data = fbDays.map(d2 => (fb.per_day_pos || {})[d2] || 0);
  state.charts.feedbackTrend.data.datasets[1].data = fbDays.map(d2 => (fb.per_day_neg || {})[d2] || 0);
  state.charts.feedbackTrend.update("none");
  setEmpty(state.charts.feedbackTrend, !fbDays.length);

  // Ambient
  const aDays = Object.keys(d.ambient_per_day);
  updateChart("ambient", aDays.map(shortDate), [Object.values(d.ambient_per_day)]);

  // Contradictions
  const cDays = Object.keys(d.contradictions_per_day);
  updateChart("contra", cDays.map(shortDate), [Object.values(d.contradictions_per_day)]);

  // Index stats
  const idxEl = document.getElementById("index-content");
  if (idx.chunks) {
    let rows = `
      <tr><td>Chunks</td><td>${idx.chunks.toLocaleString()}</td></tr>
      <tr><td>Notas</td><td>${idx.notes.toLocaleString()}</td></tr>
      <tr><td>Tags</td><td>${idx.tags}</td></tr>
      <tr><td>Carpetas</td><td>${idx.folders}</td></tr>
    `;
    if (idx.top_pagerank && idx.top_pagerank.length) {
      rows += '<tr><td colspan="2" style="color:var(--text-faint);padding-top:10px">Top PageRank</td></tr>';
      for (const pr of idx.top_pagerank) {
        const name = pr.path.split("/").pop().replace(".md", "");
        rows += `<tr><td style="color:var(--cyan)">${escapeHtml(name)}</td><td>${pr.score}</td></tr>`;
      }
    }
    idxEl.innerHTML = `<table class="stats-table">${rows}</table>`;
  } else {
    idxEl.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:20px">index no disponible</div>';
  }

  // Latency stats
  const ls = d.latency_stats;
  document.getElementById("latency-stats-content").innerHTML = `
    <table class="stats-table">
      <tr><th></th><th>p50</th><th>p95</th></tr>
      <tr><td>Retrieve</td><td>${ls.retrieve_p50}s</td><td>${ls.retrieve_p95}s</td></tr>
      <tr><td>Generate</td><td>${ls.generate_p50}s</td><td>${ls.generate_p95}s</td></tr>
      <tr><td style="font-weight:700">Total</td><td style="font-weight:700">${ls.total_p50}s</td><td style="font-weight:700">${ls.total_p95}s</td></tr>
    </table>
    <div style="margin-top:12px;font-size:11px;color:var(--text-faint)">
      Score p50: ${d.score_stats.p50} · p95: ${d.score_stats.p95} · rango: ${d.score_stats.min}–${d.score_stats.max}
    </div>
  `;

  // Tune history
  const tuneEl = document.getElementById("tune-content");
  if (d.tune_history.length) {
    tuneEl.innerHTML = d.tune_history.map(t => `
      <div class="tune-row">
        <span class="tune-date">${shortDate(t.ts)}</span>
        <span class="tune-metric">hit: ${fmt(t.baseline_hit)} → ${fmt(t.best_hit)}</span>
        <span class="tune-metric">MRR: ${fmt(t.baseline_mrr)} → ${fmt(t.best_mrr)}</span>
        <span class="tune-delta">+${fmt(t.delta)}</span>
      </div>
    `).join("");
  } else {
    tuneEl.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:20px">sin tune runs</div>';
  }
}

function updateChart(name, labels, datasets) {
  const ch = state.charts[name];
  if (!ch) return;
  ch.data.labels = labels;
  datasets.forEach((data, i) => { ch.data.datasets[i].data = data; });
  ch.update("none");
  setEmpty(ch, !labels.length || !datasets.some(ds => ds.some(v => Number(v) > 0)));
}

function setEmpty(ch, isEmpty) {
  if (!ch || !ch.canvas) return;
  const wrap = ch.canvas.parentElement;
  if (wrap && wrap.classList.contains("chart-wrap")) {
    wrap.dataset.empty = isEmpty ? "true" : "false";
  }
}

// ── Health section ────────────────────────────────────────────────────────

function renderHealth(d) {
  const h = d.health;
  if (!h) return;

  const issues = [];
  if (h.score_low_pct > 40) issues.push("retrieval débil en muchas queries");
  if (h.bad_citation_rate > 10) issues.push("LLM inventando citations");
  if (h.gate_rate > 20) issues.push("gate rate alto");
  if (d.latency_stats.total_p50 > 60) issues.push("latencia alta");

  let verdictClass, verdictHeadline, verdictIconName;
  if (issues.length === 0) { verdictClass = "ok"; verdictHeadline = "Sistema saludable"; verdictIconName = "check"; }
  else if (issues.length <= 2) { verdictClass = "mid"; verdictHeadline = "Requiere atención"; verdictIconName = "half"; }
  else { verdictClass = "bad"; verdictHeadline = "Problemas"; verdictIconName = "x"; }

  document.getElementById("health").innerHTML = `
    <div class="health-card">
      <h3>Calidad del retrieval</h3>
      <div class="health-row"><span class="health-label">Score alto (≥0.3)</span><span class="health-value good">${h.score_high} (${h.score_high_pct}%)</span></div>
      <div class="health-row"><span class="health-label">Score medio (0.05–0.3)</span><span class="health-value warn">${h.score_mid} (${h.score_mid_pct}%)</span></div>
      <div class="health-row"><span class="health-label">Score bajo (&lt;0.05)</span><span class="health-value ${h.score_low_pct > 30 ? 'bad' : 'warn'}">${h.score_low} (${h.score_low_pct}%)</span></div>
      <div class="health-bar">
        <div style="flex:${h.score_high};background:var(--green)"></div>
        <div style="flex:${h.score_mid};background:var(--yellow)"></div>
        <div style="flex:${h.score_low};background:var(--red)"></div>
      </div>
    </div>

    <div class="health-card">
      <h3>Indicadores operativos</h3>
      <div class="health-row"><span class="health-label">Gate rate (rechazadas)</span><span class="health-value ${h.gate_rate > 20 ? 'bad' : h.gate_rate > 10 ? 'warn' : 'good'}">${h.gate_rate}% (${h.gated_count})</span></div>
      <div class="health-row"><span class="health-label">Bad citations (paths inventados)</span><span class="health-value ${h.bad_citation_rate > 10 ? 'bad' : h.bad_citation_rate > 5 ? 'warn' : 'good'}">${h.bad_citation_rate}% (${h.bad_citation_total} total)</span></div>
      <div class="health-row"><span class="health-label">Respuesta promedio</span><span class="health-value">${h.avg_answer_len} chars</span></div>
      <div class="health-row"><span class="health-label">Tiempo en retrieve</span><span class="health-value">${h.retrieve_pct}%</span></div>
      <div class="health-row"><span class="health-label">Tiempo en generate</span><span class="health-value">${h.generate_pct}%</span></div>
      <div class="health-bar" style="margin-top:8px">
        <div style="flex:${h.retrieve_pct};background:var(--cyan)"></div>
        <div style="flex:${h.generate_pct};background:var(--purple)"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-faint);margin-top:4px">
        <span>retrieve ${h.retrieve_pct}%</span><span>generate ${h.generate_pct}%</span>
      </div>
    </div>

    <div class="health-card" style="display:flex;flex-direction:column;justify-content:space-between">
      <div>
        <h3>Veredicto</h3>
        <div class="health-verdict ${verdictClass}">
          ${icon(verdictIconName, { size: 16 })}
          <span>${verdictHeadline}</span>
        </div>
        ${issues.length ? `<ul class="verdict-issues">${issues.map(i => `<li><span class="tag-icon" style="color:var(--${verdictClass === "bad" ? "red" : "yellow"})">${icon("alert", { size: 11 })}</span>${escapeHtml(i)}</li>`).join("")}</ul>` : ""}
      </div>
      <div class="legend">
        <div><span class="legend-dot" style="background:var(--green)"></span>score alto = doc correcto en top-5</div>
        <div><span class="legend-dot" style="background:var(--yellow)"></span>score medio = relevante pero no ideal</div>
        <div><span class="legend-dot" style="background:var(--red)"></span>score bajo = retrieval no encontró nada útil</div>
        <div style="margin-top:6px"><span class="legend-dot" style="background:var(--cyan)"></span>gate = query rechazada por baja confianza</div>
        <div><span class="legend-dot" style="background:var(--red)"></span>bad citation = LLM inventó un path de nota</div>
      </div>
    </div>
  `;
}

// ── Feedback panel — actionable signals to improve the RAG ────────────────

function renderFeedbackPanel(fb) {
  const el = document.getElementById("feedback-panel");
  if (!el) return;

  const negPaths = fb.top_negative_paths || [];
  const corrective = fb.corrective_misses || [];
  const reasons = fb.negative_reasons || [];

  const sat = fb.net_satisfaction;
  const satClass = sat == null ? "" : sat >= 50 ? "good" : sat >= 0 ? "warn" : "bad";

  const upIcon = icon("up", { size: 11, color: "var(--green)" });
  const downIcon = icon("down", { size: 11, color: "var(--red)" });
  const negPathsHtml = negPaths.length
    ? negPaths.map(p => `
        <div class="fb-row neg">
          <div>
            <div class="fb-path">${escapeHtml(shortenPath(p.path))}</div>
            <div class="fb-meta">${p.pos_count > 0 ? `también ${p.pos_count} ${upIcon}` : "sólo señales negativas"}</div>
          </div>
          <div class="fb-count">${p.count} ${downIcon}</div>
        </div>
      `).join("")
    : `<div class="fb-empty"><span class="tag-icon" style="color:var(--green)">${icon("check", { size: 12 })}</span> sin notas con feedback negativo recurrente</div>`;

  const correctiveHtml = corrective.length
    ? corrective.map(c => `
        <div class="fb-row miss" title="usuario indicó que la respuesta correcta estaba en otra nota">
          <div>
            <div class="fb-q">${escapeHtml(c.q)}</div>
            <div class="fb-meta">debió retornar → <span style="color:var(--orange)">${escapeHtml(shortenPath(c.missing_path))}</span></div>
            <div class="fb-meta" style="opacity:0.7">en lugar de: ${(c.retrieved || []).map(p => escapeHtml(shortenPath(p))).join(" · ") || "(nada)"}</div>
          </div>
          <div style="font-size:10px;color:var(--text-faint);align-self:center">${shortDate(c.ts)}</div>
        </div>
      `).join("")
    : `<div class="fb-empty">sin retrieval misses corregidos por el usuario</div>`;

  const reasonsHtml = reasons.length
    ? reasons.map(r => `
        <div class="fb-row neg">
          <div>
            <div class="fb-q">${escapeHtml(r.q)}</div>
            <div class="fb-meta" style="color:var(--text-dim);margin-top:4px">"${escapeHtml(r.reason)}"</div>
          </div>
          <div style="font-size:10px;color:var(--text-faint);align-self:center">${shortDate(r.ts)}</div>
        </div>
      `).join("")
    : `<div class="fb-empty">sin razones negativas registradas</div>`;

  el.innerHTML = `
    <div class="fb-cal">
      <div class="fb-cal-item"><span class="fb-cal-label">Satisfacción neta</span><span class="fb-cal-value ${satClass}">${sat == null ? "—" : sat + "%"}</span></div>
      <div class="fb-cal-item"><span class="fb-cal-label"><span class="tag-icon" style="color:var(--green)">${icon("up", { size: 10 })}</span>positivos (período)</span><span class="fb-cal-value good">${fb.recent_pos || 0}</span></div>
      <div class="fb-cal-item"><span class="fb-cal-label"><span class="tag-icon" style="color:var(--red)">${icon("down", { size: 10 })}</span>negativos (período)</span><span class="fb-cal-value bad">${fb.recent_neg || 0}</span></div>
      <div class="fb-cal-item" title="Score alto pero feedback negativo → reranker o LLM mienten con confianza">
        <span class="fb-cal-label">Falsos positivos (gate)</span>
        <span class="fb-cal-value ${fb.false_confident > 3 ? 'bad' : fb.false_confident > 0 ? 'warn' : 'good'}">${fb.false_confident || 0}</span>
      </div>
      <div class="fb-cal-item" title="Score bajo pero el usuario marcó que sí servía → gate demasiado estricto">
        <span class="fb-cal-label">Falsos negativos (gate)</span>
        <span class="fb-cal-value ${fb.false_gated > 3 ? 'bad' : fb.false_gated > 0 ? 'warn' : 'good'}">${fb.false_gated || 0}</span>
      </div>
      <div class="fb-cal-item" title="Queries donde el usuario aportó la nota correcta — gold para queries.yaml">
        <span class="fb-cal-label">Retrieval misses corregidas</span>
        <span class="fb-cal-value ${fb.n_corrective_misses > 5 ? 'bad' : fb.n_corrective_misses > 0 ? 'warn' : 'good'}">${fb.n_corrective_misses || 0}</span>
      </div>
    </div>

    <div class="feedback-grid" style="margin-top:14px">
      <div>
        <h3 style="font-size:11px;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px">Notas problemáticas (revisar / dividir / re-tag)</h3>
        <div class="fb-list">${negPathsHtml}</div>
      </div>
      <div>
        <h3 style="font-size:11px;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px">Retrieval misses (candidatas para queries.yaml)</h3>
        <div class="fb-list">${correctiveHtml}</div>
      </div>
    </div>

    <div style="margin-top:14px">
      <h3 style="font-size:11px;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px">Razones de fallo recientes</h3>
      <div class="fb-list">${reasonsHtml}</div>
    </div>
  `;
}

// ── Icons (Lucide-style, 16x16) ───────────────────────────────────────────

const ICONS = {
  refresh:  '<path d="M21 12a9 9 0 0 0-15-6.7L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/><path d="M21 21v-5h-5"/>',
  warning:  '<path d="M10.3 3.3L1.7 18a2 2 0 0 0 1.7 3h17.2a2 2 0 0 0 1.7-3L13.7 3.3a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  ban:      '<circle cx="12" cy="12" r="10"/><path d="M4.9 4.9l14.2 14.2"/>',
  inbox:    '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.5 5.5l-3 6.5V18a2 2 0 0 0 2 2h15a2 2 0 0 0 2-2v-6l-3-6.5a2 2 0 0 0-1.8-1.1H7.3a2 2 0 0 0-1.8 1.1z"/>',
  link:     '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
  up:       '<path d="M7 10v12"/><path d="M15 5.9L14 9h5.5a2 2 0 0 1 2 2.3l-1.4 8a2 2 0 0 1-2 1.7H7V10l5-8c1.7.2 3 1.5 3 3.2z"/>',
  down:     '<path d="M17 14V2"/><path d="M9 18.1L10 15H4.5a2 2 0 0 1-2-2.3l1.4-8a2 2 0 0 1 2-1.7H17v12l-5 8c-1.7-.2-3-1.5-3-3.2z"/>',
  check:    '<circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/>',
  half:     '<circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 0 0 20z"/>',
  x:        '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6"/><path d="M9 9l6 6"/>',
  dot:      '<circle cx="12" cy="12" r="4"/>',
  alert:    '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
};

function icon(name, { size = 14, color = "currentColor", cls = "" } = {}) {
  const body = ICONS[name];
  if (!body) return "";
  return `<svg class="icon ${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

// ── Live ticker ───────────────────────────────────────────────────────────

function pushTicker(kind, ev) {
  state.ticker.unshift({ kind, ev, t: Date.now() });
  state.ticker = state.ticker.slice(0, TICKER_MAX);
  renderTicker();
}

function renderTicker() {
  const list = document.getElementById("ticker-list");
  const meta = document.getElementById("ticker-meta");
  if (!list) return;
  if (!state.ticker.length) {
    list.innerHTML = '<div class="ticker-empty">sin eventos todavía. cuando llegue una query, aparecerá acá.</div>';
    return;
  }
  meta.textContent = `últimos ${state.ticker.length} eventos`;
  list.innerHTML = state.ticker.map(({ kind, ev }) => {
    if (kind === "query") {
      const score = typeof ev.score === "number" ? ev.score : null;
      const scoreCls = score == null ? "" : score >= 0.3 ? "high" : score >= 0.05 ? "mid" : "low";
      let tag = "";
      let scoreCell = score != null ? score.toFixed(2) : "—";
      let latencyCell = ev.latency != null ? ev.latency + "s" : "—";
      if (ev.phase === "in_flight") {
        tag = `<span class="tag-icon" style="color:var(--cyan)">${icon("refresh", { size: 12, cls: "spin" })}</span>`;
        scoreCell = '<span style="color:var(--cyan)">…</span>';
        latencyCell = '<span style="color:var(--cyan)">en curso</span>';
      } else if (ev.phase === "error") {
        tag = `<span class="tag-icon" style="color:var(--red)">${icon("warning", { size: 12 })}</span>`;
        scoreCell = '<span style="color:var(--red)">err</span>';
        latencyCell = `<span style="color:var(--red)" title="${escapeHtml(ev.error || "")}">${escapeHtml((ev.error || "fallo").slice(0, 40))}</span>`;
      } else if (ev.gated) {
        tag = `<span class="tag-icon" style="color:var(--text-faint)">${icon("ban", { size: 12 })}</span>`;
      }
      return `
        <div class="ticker-item">
          <span class="t-time">${timeOf(ev.ts)}</span>
          <span class="t-source ${ev.source || "cli"}">${ev.source || "cli"}</span>
          <span class="t-q" title="${escapeHtml(ev.q || "")}">${tag}${escapeHtml(ev.q || "(sin texto)")}</span>
          <span class="t-score ${scoreCls}">${scoreCell}</span>
          <span class="t-latency">${latencyCell}</span>
        </div>
      `;
    }
    if (kind === "feedback") {
      const isPos = ev.rating === 1;
      const isNeg = ev.rating === -1;
      const ic = isPos ? icon("up", { size: 12, color: "var(--green)" })
                 : isNeg ? icon("down", { size: 12, color: "var(--red)" })
                 : icon("dot", { size: 12, color: "var(--text-faint)" });
      return `
        <div class="ticker-item">
          <span class="t-time">${timeOf(ev.ts)}</span>
          <span class="t-source feedback">feedback</span>
          <span class="t-q"><span class="tag-icon">${ic}</span>${escapeHtml(ev.q || ev.reason || "(sin texto)")}</span>
          <span class="t-score">${ic}</span>
          <span class="t-latency"></span>
        </div>
      `;
    }
    if (kind === "ambient") {
      return `
        <div class="ticker-item">
          <span class="t-time">${timeOf(ev.ts)}</span>
          <span class="t-source" style="background:rgba(255,166,87,0.12);color:var(--orange)">ambient</span>
          <span class="t-q" title="${escapeHtml(ev.path || "")}"><span class="tag-icon" style="color:var(--orange)">${icon("inbox", { size: 12 })}</span>${escapeHtml(shortenPath(ev.path || ""))}</span>
          <span class="t-score" style="color:var(--orange)">+${ev.wikilinks_applied || 0}${icon("link", { size: 10, color: "var(--orange)" })}</span>
          <span class="t-latency"></span>
        </div>
      `;
    }
    if (kind === "contradiction") {
      return `
        <div class="ticker-item">
          <span class="t-time">${timeOf(ev.ts)}</span>
          <span class="t-source" style="background:rgba(255,123,114,0.12);color:var(--red)">contradicción</span>
          <span class="t-q"><span class="tag-icon" style="color:var(--red)">${icon("warning", { size: 12 })}</span>${escapeHtml(shortenPath(ev.path || ""))}</span>
          <span class="t-score"></span>
          <span class="t-latency"></span>
        </div>
      `;
    }
    return "";
  }).join("");
}

// ── Helpers ───────────────────────────────────────────────────────────────

function shortDate(iso) {
  if (!iso) return "?";
  const parts = iso.split("T")[0].split("-");
  return `${parts[1]}/${parts[2]}`;
}

function timeOf(iso) {
  if (!iso) return "?";
  const t = iso.split("T")[1] || "";
  return t.slice(0, 5);
}

function nowHM() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return typeof v === "number" ? v.toFixed(3) : String(v);
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function shortenPath(p) {
  if (!p) return "";
  const parts = p.split("/");
  if (parts.length <= 2) return p;
  return `…/${parts.slice(-2).join("/")}`;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
