/* obsidian-rag dashboard — Chart.js 4 + vanilla JS */

const C = {
  cyan:   "#79c0ff",
  green:  "#7ee787",
  yellow: "#e3c27a",
  red:    "#ff7b72",
  purple: "#d2a8ff",
  orange: "#ffa657",
  pink:   "#f778ba",
  dim:    "#5a5a60",
  border: "#33333a",
  card:   "#26262c",
  text:   "#ececed",
  textDim:"#8a8a90",
};

// Chart.js global defaults
Chart.defaults.color = C.textDim;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'SF Mono','Menlo','Monaco','JetBrains Mono',ui-monospace,monospace";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.plugins.tooltip.backgroundColor = "#2a2a30";
Chart.defaults.plugins.tooltip.borderColor = C.border;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleFont = { size: 11 };
Chart.defaults.plugins.tooltip.bodyFont = { size: 11 };
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.scale.grid = { color: "rgba(255,255,255,0.04)" };

const daysPicker = document.getElementById("days-picker");
const content = document.getElementById("content");
const metaPeriod = document.getElementById("meta-period");

let charts = [];

daysPicker.addEventListener("change", () => load(+daysPicker.value));

load(30);

async function load(days) {
  content.innerHTML = '<div class="loading">cargando datos </div>';
  charts.forEach(c => c.destroy());
  charts = [];

  try {
    const res = await fetch(`/api/dashboard?days=${days}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    metaPeriod.textContent = `${d.kpis.total_queries} queries · ${days}d`;
    render(d);
  } catch (err) {
    content.innerHTML = `<div class="loading" style="color:var(--red)">error: ${err.message}</div>`;
  }
}

function render(d) {
  const k = d.kpis;
  const idx = d.index || {};

  content.innerHTML = `
    <div class="kpis" id="kpis"></div>
    <div class="health" id="health"></div>
    <div class="charts">
      <div class="chart-card wide">
        <h2>Queries por dia</h2>
        <div class="chart-wrap"><canvas id="ch-queries-day"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Latencia total (p50 diario)</h2>
        <div class="chart-wrap"><canvas id="ch-latency"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Distribucion de scores</h2>
        <div class="chart-wrap"><canvas id="ch-scores"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Actividad por hora</h2>
        <div class="chart-wrap"><canvas id="ch-hours"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Origen de queries</h2>
        <div class="chart-wrap"><canvas id="ch-sources"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Comandos</h2>
        <div class="chart-wrap"><canvas id="ch-cmds"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Hot topics</h2>
        <div class="chart-wrap"><canvas id="ch-topics"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Score trend (promedio diario)</h2>
        <div class="chart-wrap"><canvas id="ch-score-trend"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Ambient hooks por dia</h2>
        <div class="chart-wrap"><canvas id="ch-ambient"></canvas></div>
      </div>
      <div class="chart-card">
        <h2>Contradicciones por dia</h2>
        <div class="chart-wrap"><canvas id="ch-contra"></canvas></div>
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

  // ── KPI cards ──
  const kpis = [
    { label: "Queries", value: k.total_queries, cls: "cyan", sub: `${k.total_queries_all_time} all-time` },
    { label: "Notas indexadas", value: idx.notes || "—", cls: "green", sub: `${idx.chunks || 0} chunks` },
    { label: "Latencia promedio", value: k.avg_latency ? `${k.avg_latency}s` : "—", cls: "yellow", sub: `ret ${k.avg_retrieve || "—"}s · gen ${k.avg_generate || "—"}s` },
    { label: "Feedback +/-", value: `${k.feedback_positive}/${k.feedback_negative}`, cls: "green", sub: k.feedback_positive + k.feedback_negative > 0 ? `${Math.round(k.feedback_positive / (k.feedback_positive + k.feedback_negative) * 100)}% positivo` : "sin feedback" },
    { label: "Sesiones", value: k.sessions, cls: "purple", sub: `${k.wa_sessions} WhatsApp` },
    { label: "Ambient hooks", value: k.ambient_hooks, cls: "orange", sub: `${k.ambient_wikilinks} wikilinks aplicados` },
    { label: "Contradicciones", value: k.contradictions_found, cls: "red", sub: "detectadas en indexación" },
    { label: "Surface pairs", value: k.surface_pairs, cls: "cyan", sub: `${d.surface_runs} runs` },
    { label: "Tags", value: idx.tags || "—", cls: "", sub: `${idx.folders || 0} carpetas` },
    { label: "Filing proposals", value: k.filings, cls: "yellow", sub: "en el período" },
  ];

  const kpiContainer = document.getElementById("kpis");
  for (const kpi of kpis) {
    const el = document.createElement("div");
    el.className = "kpi";
    el.innerHTML = `
      <span class="kpi-label">${kpi.label}</span>
      <span class="kpi-value ${kpi.cls}">${kpi.value}</span>
      <span class="kpi-sub">${kpi.sub}</span>
    `;
    kpiContainer.appendChild(el);
  }

  // ── Health section ──
  renderHealth(d);

  // ── Queries per day (line) ──
  const qDays = Object.keys(d.queries_per_day);
  const qVals = Object.values(d.queries_per_day);
  charts.push(new Chart(document.getElementById("ch-queries-day"), {
    type: "bar",
    data: {
      labels: qDays.map(shortDate),
      datasets: [{
        data: qVals,
        backgroundColor: hexAlpha(C.cyan, 0.5),
        borderColor: C.cyan,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45 } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  }));

  // ── Latency per day (line) ──
  const lDays = Object.keys(d.latency_per_day);
  const lP50 = lDays.map(day => d.latency_per_day[day].p50);
  const lP95 = lDays.map(day => d.latency_per_day[day].p95);
  charts.push(new Chart(document.getElementById("ch-latency"), {
    type: "line",
    data: {
      labels: lDays.map(shortDate),
      datasets: [
        { label: "p50", data: lP50, borderColor: C.cyan, backgroundColor: hexAlpha(C.cyan, 0.1), tension: 0.3, fill: true, pointRadius: 2 },
        { label: "p95", data: lP95, borderColor: C.yellow, backgroundColor: "transparent", borderDash: [4, 4], tension: 0.3, pointRadius: 2 },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "seg" } },
        x: { ticks: { maxRotation: 45 } },
      },
    },
  }));

  // ── Score distribution (bar) ──
  const scoreLabels = Array.from({ length: 10 }, (_, i) => `${(i/10).toFixed(1)}`);
  const scoreColors = d.score_distribution.map((_, i) =>
    i >= 5 ? C.green : i >= 2 ? C.yellow : C.red
  );
  charts.push(new Chart(document.getElementById("ch-scores"), {
    type: "bar",
    data: {
      labels: scoreLabels,
      datasets: [{
        data: d.score_distribution,
        backgroundColor: scoreColors.map(c => hexAlpha(c, 0.6)),
        borderColor: scoreColors,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "score" } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  }));

  // ── Hours heatmap (bar) ──
  const hourLabels = Array.from({ length: 24 }, (_, i) => `${i}h`);
  const hourVals = hourLabels.map((_, i) => d.hours[String(i)] || 0);
  const maxHour = Math.max(...hourVals, 1);
  const hourColors = hourVals.map(v => hexAlpha(C.cyan, 0.2 + (v / maxHour) * 0.8));
  charts.push(new Chart(document.getElementById("ch-hours"), {
    type: "bar",
    data: {
      labels: hourLabels,
      datasets: [{
        data: hourVals,
        backgroundColor: hourColors,
        borderColor: C.cyan,
        borderWidth: 1,
        borderRadius: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  }));

  // ── Sources (doughnut) ──
  const srcLabels = Object.keys(d.sources);
  const srcVals = Object.values(d.sources);
  const srcColors = [C.green, C.cyan, C.purple, C.yellow, C.orange, C.pink];
  charts.push(new Chart(document.getElementById("ch-sources"), {
    type: "doughnut",
    data: {
      labels: srcLabels,
      datasets: [{
        data: srcVals,
        backgroundColor: srcColors.slice(0, srcLabels.length).map(c => hexAlpha(c, 0.7)),
        borderColor: C.card,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      cutout: "55%",
      plugins: {
        legend: { position: "right" },
      },
    },
  }));

  // ── Commands (horizontal bar) ──
  const cmdLabels = Object.keys(d.cmds);
  const cmdVals = Object.values(d.cmds);
  const cmdColors = cmdLabels.map((_, i) => [C.cyan, C.green, C.yellow, C.purple, C.orange, C.pink, C.red][i % 7]);
  charts.push(new Chart(document.getElementById("ch-cmds"), {
    type: "bar",
    data: {
      labels: cmdLabels,
      datasets: [{
        data: cmdVals,
        backgroundColor: cmdColors.map(c => hexAlpha(c, 0.6)),
        borderColor: cmdColors,
        borderWidth: 1,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  }));

  // ── Hot topics (horizontal bar) ──
  const topicData = d.hot_topics.filter(t => t.count >= 2).slice(0, 12);
  if (topicData.length) {
    charts.push(new Chart(document.getElementById("ch-topics"), {
      type: "bar",
      data: {
        labels: topicData.map(t => t.topic),
        datasets: [{
          data: topicData.map(t => t.count),
          backgroundColor: hexAlpha(C.purple, 0.5),
          borderColor: C.purple,
          borderWidth: 1,
          borderRadius: 3,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, ticks: { precision: 0 } } },
      },
    }));
  } else {
    document.getElementById("ch-topics").parentElement.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:40px">sin datos suficientes</div>';
  }

  // ── Score trend (line) ──
  const sTrend = d.health.score_trend || {};
  const stDays = Object.keys(sTrend);
  const stVals = Object.values(sTrend);
  if (stDays.length) {
    charts.push(new Chart(document.getElementById("ch-score-trend"), {
      type: "line",
      data: {
        labels: stDays.map(shortDate),
        datasets: [{
          label: "score promedio",
          data: stVals,
          borderColor: C.green,
          backgroundColor: hexAlpha(C.green, 0.08),
          tension: 0.3,
          fill: true,
          pointRadius: 3,
          pointBackgroundColor: stVals.map(v => v >= 0.2 ? C.green : v >= 0.05 ? C.yellow : C.red),
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 1.0, title: { display: true, text: "score" } },
          x: { ticks: { maxRotation: 45 } },
        },
      },
    }));
  } else {
    document.getElementById("ch-score-trend").parentElement.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:40px">sin datos</div>';
  }

  // ── Ambient per day (bar) ──
  const aDays = Object.keys(d.ambient_per_day);
  const aVals = Object.values(d.ambient_per_day);
  if (aDays.length) {
    charts.push(new Chart(document.getElementById("ch-ambient"), {
      type: "bar",
      data: {
        labels: aDays.map(shortDate),
        datasets: [{
          data: aVals,
          backgroundColor: hexAlpha(C.orange, 0.5),
          borderColor: C.orange,
          borderWidth: 1,
          borderRadius: 3,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      },
    }));
  } else {
    document.getElementById("ch-ambient").parentElement.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:40px">sin datos</div>';
  }

  // ── Contradictions per day (bar) ──
  const cDays = Object.keys(d.contradictions_per_day);
  const cVals = Object.values(d.contradictions_per_day);
  if (cDays.length) {
    charts.push(new Chart(document.getElementById("ch-contra"), {
      type: "bar",
      data: {
        labels: cDays.map(shortDate),
        datasets: [{
          data: cVals,
          backgroundColor: hexAlpha(C.red, 0.5),
          borderColor: C.red,
          borderWidth: 1,
          borderRadius: 3,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      },
    }));
  } else {
    document.getElementById("ch-contra").parentElement.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:40px">sin datos</div>';
  }

  // ── Index stats table ──
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
        rows += `<tr><td style="color:var(--cyan)">${name}</td><td>${pr.score}</td></tr>`;
      }
    }
    idxEl.innerHTML = `<table class="stats-table">${rows}</table>`;
  } else {
    idxEl.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:20px">index no disponible</div>';
  }

  // ── Latency stats table ──
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

  // ── Tune history ──
  const tuneEl = document.getElementById("tune-content");
  if (d.tune_history.length) {
    let html = "";
    for (const t of d.tune_history) {
      html += `<div class="tune-row">
        <span class="tune-date">${shortDate(t.ts)}</span>
        <span class="tune-metric">hit: ${fmt(t.baseline_hit)} → ${fmt(t.best_hit)}</span>
        <span class="tune-metric">MRR: ${fmt(t.baseline_mrr)} → ${fmt(t.best_mrr)}</span>
        <span class="tune-delta">+${fmt(t.delta)}</span>
      </div>`;
    }
    tuneEl.innerHTML = html;
  } else {
    tuneEl.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:20px">sin tune runs</div>';
  }
}

// ── Helpers ──

function shortDate(iso) {
  if (!iso) return "?";
  const parts = iso.split("T")[0].split("-");
  return `${parts[1]}/${parts[2]}`;
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

// ── Health section ──

function renderHealth(d) {
  const h = d.health;
  if (!h) return;

  const el = document.getElementById("health");

  // Compute overall verdict
  const issues = [];
  if (h.score_low_pct > 40) issues.push("retrieval débil en muchas queries");
  if (h.bad_citation_rate > 10) issues.push("LLM inventando citations");
  if (h.gate_rate > 20) issues.push("gate rate alto");
  if (d.latency_stats.total_p50 > 60) issues.push("latencia alta");

  let verdictClass, verdictText;
  if (issues.length === 0) {
    verdictClass = "ok";
    verdictText = "Sistema saludable";
  } else if (issues.length <= 2) {
    verdictClass = "mid";
    verdictText = "Atención: " + issues.join(", ");
  } else {
    verdictClass = "bad";
    verdictText = "Problemas: " + issues.join(", ");
  }

  const verdictIcon = verdictClass === "ok" ? "●" : verdictClass === "mid" ? "◐" : "○";

  el.innerHTML = `
    <div class="health-card">
      <h3>Calidad del retrieval</h3>
      <div class="health-row">
        <span class="health-label">Score alto (≥0.3)</span>
        <span class="health-value good">${h.score_high} (${h.score_high_pct}%)</span>
      </div>
      <div class="health-row">
        <span class="health-label">Score medio (0.05–0.3)</span>
        <span class="health-value warn">${h.score_mid} (${h.score_mid_pct}%)</span>
      </div>
      <div class="health-row">
        <span class="health-label">Score bajo (&lt;0.05)</span>
        <span class="health-value ${h.score_low_pct > 30 ? 'bad' : 'warn'}">${h.score_low} (${h.score_low_pct}%)</span>
      </div>
      <div class="health-bar">
        <div style="flex:${h.score_high};background:var(--green)"></div>
        <div style="flex:${h.score_mid};background:var(--yellow)"></div>
        <div style="flex:${h.score_low};background:var(--red)"></div>
      </div>
    </div>

    <div class="health-card">
      <h3>Indicadores operativos</h3>
      <div class="health-row">
        <span class="health-label">Gate rate (rechazadas)</span>
        <span class="health-value ${h.gate_rate > 20 ? 'bad' : h.gate_rate > 10 ? 'warn' : 'good'}">${h.gate_rate}% (${h.gated_count})</span>
      </div>
      <div class="health-row">
        <span class="health-label">Bad citations (paths inventados)</span>
        <span class="health-value ${h.bad_citation_rate > 10 ? 'bad' : h.bad_citation_rate > 5 ? 'warn' : 'good'}">${h.bad_citation_rate}% (${h.bad_citation_total} total)</span>
      </div>
      <div class="health-row">
        <span class="health-label">Respuesta promedio</span>
        <span class="health-value">${h.avg_answer_len} chars</span>
      </div>
      <div class="health-row">
        <span class="health-label">Tiempo en retrieve</span>
        <span class="health-value">${h.retrieve_pct}%</span>
      </div>
      <div class="health-row">
        <span class="health-label">Tiempo en generate</span>
        <span class="health-value">${h.generate_pct}%</span>
      </div>
      <div class="health-bar" style="margin-top:8px">
        <div style="flex:${h.retrieve_pct};background:var(--cyan)" title="retrieve"></div>
        <div style="flex:${h.generate_pct};background:var(--purple)" title="generate"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-faint);margin-top:4px">
        <span>retrieve ${h.retrieve_pct}%</span><span>generate ${h.generate_pct}%</span>
      </div>
    </div>

    <div class="health-card" style="display:flex;flex-direction:column;justify-content:space-between">
      <div>
        <h3>Veredicto</h3>
        <div class="health-verdict ${verdictClass}">${verdictIcon} ${verdictText}</div>
      </div>
      <div style="margin-top:16px;font-size:11px;color:var(--text-faint);line-height:1.6">
        <div><span style="color:var(--green)">●</span> score alto = doc correcto en top-5</div>
        <div><span style="color:var(--yellow)">●</span> score medio = relevante pero no ideal</div>
        <div><span style="color:var(--red)">●</span> score bajo = retrieval no encontró nada útil</div>
        <div style="margin-top:6px"><span style="color:var(--cyan)">●</span> gate = query rechazada por baja confianza</div>
        <div><span style="color:var(--red)">●</span> bad citation = LLM inventó un path de nota</div>
      </div>
    </div>
  `;
}
