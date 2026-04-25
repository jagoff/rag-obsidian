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
const POLL_MS = 60_000;        // refresh aggregations every 60s; SSE pushes live deltas in between
const TICKER_MAX = 5;          // live events kept on screen

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
    if (state.waPoll) { clearInterval(state.waPoll); state.waPoll = null; }
  } else {
    setLiveState("off", "reconectando…");
    load(false);
    startPolling();
    startStream();
    if (state.waInitDone) startWaPolling();
  }
});

document.addEventListener("visibilitychange", () => {
  // Pause polling/stream when tab hidden to avoid burning CPU.
  if (document.hidden) {
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
    if (state.evtSrc) { state.evtSrc.close(); state.evtSrc = null; }
    if (state.waPoll) { clearInterval(state.waPoll); state.waPoll = null; }
  } else if (!state.paused) {
    load(false);
    startPolling();
    startStream();
    if (state.waInitDone) {
      // Pull a fresh snapshot first (we likely missed N seconds of
      // changes while hidden), then resume the 30s loop. Init handles
      // the first-run case earlier in load().
      refreshWaScheduled();
      startWaPolling();
    }
  }
});

// Hard cleanup on tab close / navigation: visibilitychange handles
// "tab hidden" but does NOT reliably fire on close, and the MEM/CPU
// EventSources are independent of state.evtSrc — without this hook
// they leak as zombie connections from the browser's PoV until GC.
window.addEventListener("beforeunload", () => {
  try { if (state.evtSrc) state.evtSrc.close(); } catch (_) {}
  try { if (state.poll) clearInterval(state.poll); } catch (_) {}
  try { if (state.waPoll) clearInterval(state.waPoll); } catch (_) {}
  try { if (MEM && MEM.es) MEM.es.close(); } catch (_) {}
  try { if (CPU && CPU.es) CPU.es.close(); } catch (_) {}
});

// Initial boot
load(true);
startPolling();
startStream();

// ── Loading + polling ─────────────────────────────────────────────────────

// SR live-region: dashboard.html has `<span id="dashboard-status"
// role="status" aria-live="polite">` sitting OUTSIDE <main>. It
// survives buildLayout() rebuilds (which clobber #content.innerHTML)
// so we can re-text it across the whole load lifecycle. Politely
// (not assertively) — we don't want to interrupt SR users every poll.
function announceStatus(msg) {
  const node = document.getElementById("dashboard-status");
  if (node) node.textContent = msg;
}

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
      memInit();
      cpuInit();
      waInit();
    }
    refresh(d);
    announceStatus(`Datos del dashboard actualizados (${d.kpis.total_queries} queries en ${state.days} días)`);
  } catch (err) {
    if (!state.built) {
      // role=alert + aria-live=assertive so SR users get the error even
      // if they were on another part of the page when the fetch failed.
      el.content.innerHTML = `<div class="loading" style="color:var(--red)" role="alert" aria-live="assertive">error: ${err.message}</div>`;
    } else {
      el.metaUpdated.textContent = `error ${err.message}`;
    }
    announceStatus(`Error al cargar datos del dashboard: ${err.message}`);
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
  // Each major block is a `<section aria-labelledby>` wrapping its h2.
  // Why: the layout is built dynamically from a flat string of <div>s,
  // which means screen readers couldn't find landmarks via the rotor.
  // Wrapping in `<section>` (a sectioning content element) + giving the
  // visible h2 an id and pointing aria-labelledby at it = each region
  // shows up named in the rotor without changing visible markup.
  // Ticker keeps role="log" but live="off" — the global `live-toggle`
  // (header) pauses the upstream stream, so we don't also force SR to
  // announce every sample as it arrives (P2 #9, see audit notes).
  el.content.innerHTML = `
    <section class="ticker" aria-labelledby="sec-ticker">
      <div class="ticker-head">
        <h2 id="sec-ticker">Eventos en vivo</h2>
        <span style="font-size:11px;color:var(--text-faint)" id="ticker-meta">esperando…</span>
      </div>
      <div class="ticker-list" id="ticker-list" role="log" aria-live="off" aria-relevant="additions" aria-label="Eventos en vivo (usar el botón en vivo del header para pausar)">
        <div class="ticker-empty">sin eventos todavía. cuando llegue una query, aparecerá acá.</div>
      </div>
    </section>

    <section class="kpis" id="kpis" aria-label="Indicadores clave"></section>
    <section class="signals-panel" id="signals-panel" aria-labelledby="signals-title">
      <h3 id="signals-title">Señales al ranker-vivo <span class="signals-window" id="signals-window">—</span></h3>
      <div class="signals-grid" id="signals-grid">
        <div class="signals-empty">esperando eventos…</div>
      </div>
    </section>
    <section class="health" id="health" aria-label="Salud del sistema"></section>

    <div class="charts">
      <section class="chart-card wide" id="card-memory" aria-labelledby="sec-memory">
        <h2 id="sec-memory">Memoria del rag <span style="font-size:11px;font-weight:400;color:var(--text-faint);margin-left:6px;">rag + ollama + sqlite-vec + whatsapp</span> <span id="mem-live-dot" style="font-size:10px;font-weight:400;color:var(--green);margin-left:8px;">● live</span></h2>
        <div class="memcard-grid">
          <div class="memcard-main">
            <div class="memcard-head">
              <div>
                <span class="memcard-total" id="mem-total">—<span class="unit">GB</span></span>
                <span class="memcard-delta flat" id="mem-delta"></span>
              </div>
              <div class="memcard-window" id="mem-window">
                <button data-min="5">5m</button>
                <button data-min="60" class="active">1h</button>
                <button data-min="360">6h</button>
                <button data-min="1440">24h</button>
              </div>
            </div>
            <div class="memcard-chart-wrap"><canvas id="ch-memory" aria-label="Gráfico de uso de memoria por categoría (rag, ollama, sqlite-vec, whatsapp) en el tiempo"></canvas></div>
          </div>
          <div class="memcard-top">
            <h3>Top procesos</h3>
            <ul id="mem-top-list"><li><span class="name">—</span></li></ul>
          </div>
        </div>
      </section>

      <section class="chart-card wide" id="card-cpu" aria-labelledby="sec-cpu">
        <h2 id="sec-cpu">CPU del rag <span style="font-size:11px;font-weight:400;color:var(--text-faint);margin-left:6px;">rag + ollama + sqlite-vec + whatsapp · % de 1 core</span> <span id="cpu-live-dot" style="font-size:10px;font-weight:400;color:var(--green);margin-left:8px;">● live</span></h2>
        <div class="memcard-grid">
          <div class="memcard-main">
            <div class="memcard-head">
              <div>
                <span class="memcard-total" id="cpu-total">—<span class="unit">%</span></span>
                <span class="memcard-delta flat" id="cpu-sub"></span>
              </div>
              <div class="memcard-window" id="cpu-window">
                <button data-min="5">5m</button>
                <button data-min="60" class="active">1h</button>
                <button data-min="360">6h</button>
                <button data-min="1440">24h</button>
              </div>
            </div>
            <div class="memcard-chart-wrap"><canvas id="ch-cpu" aria-label="Gráfico de uso de CPU por categoría (rag, ollama, sqlite-vec, whatsapp) en el tiempo"></canvas></div>
          </div>
          <div class="memcard-top">
            <h3>Top procesos</h3>
            <ul id="cpu-top-list"><li><span class="name">—</span></li></ul>
          </div>
        </div>
      </section>

      <section class="chart-card" aria-labelledby="sec-queries-day">
        <h2 id="sec-queries-day">Queries por dia</h2>
        <div class="chart-wrap"><canvas id="ch-queries-day" aria-label="Gráfico de barras: queries por día en el período seleccionado"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>

      <section class="chart-card wide" aria-labelledby="sec-feedback">
        <h2 id="sec-feedback">Feedback accionable — señales para mejorar el RAG</h2>
        <div id="feedback-panel"></div>
      </section>

      <section class="chart-card wide" id="sec-wa-scheduled-card" aria-labelledby="sec-wa-scheduled">
        <h2 id="sec-wa-scheduled">Mensajes programados de WhatsApp <span id="wa-pending-badge" class="wa-pending-badge">(0 pending)</span></h2>
        <div class="wa-tabs" role="tablist" aria-label="Filtrar mensajes programados por estado">
          <button type="button" role="tab" id="wa-tab-pending" aria-selected="true" aria-controls="wa-scheduled-body" data-wa-tab="pending" class="wa-tab active">Pendientes <span class="wa-tab-count">0</span></button>
          <button type="button" role="tab" id="wa-tab-sent" aria-selected="false" aria-controls="wa-scheduled-body" data-wa-tab="sent" class="wa-tab">Enviados <span class="wa-tab-count">0</span></button>
          <button type="button" role="tab" id="wa-tab-failed" aria-selected="false" aria-controls="wa-scheduled-body" data-wa-tab="failed" class="wa-tab">Fallados <span class="wa-tab-count">0</span></button>
          <button type="button" role="tab" id="wa-tab-all" aria-selected="false" aria-controls="wa-scheduled-body" data-wa-tab="all" class="wa-tab">Todos <span class="wa-tab-count">0</span></button>
        </div>
        <div id="wa-scheduled-body" class="wa-scheduled-body" role="tabpanel" aria-labelledby="wa-tab-pending">
          <div class="fb-empty">cargando…</div>
        </div>
      </section>

      <section class="chart-card" aria-labelledby="sec-latency">
        <h2 id="sec-latency">Latencia total (p50 diario)</h2>
        <div class="chart-wrap"><canvas id="ch-latency" aria-label="Gráfico de líneas: latencia total p50 y p95 por día (segundos)"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-scores">
        <h2 id="sec-scores">Distribucion de scores</h2>
        <div class="chart-wrap"><canvas id="ch-scores" aria-label="Histograma de scores de retrieval por bucket (0.0 a 1.0)"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-hours">
        <h2 id="sec-hours">Actividad por hora</h2>
        <div class="chart-wrap"><canvas id="ch-hours" aria-label="Gráfico de barras: cantidad de queries por hora del día (0h a 23h)"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-sources">
        <h2 id="sec-sources">Origen de queries</h2>
        <div class="chart-wrap"><canvas id="ch-sources" aria-label="Gráfico de dona: origen de las queries (whatsapp, web, cli, etc.)"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-cmds">
        <h2 id="sec-cmds">Comandos</h2>
        <div class="chart-wrap"><canvas id="ch-cmds" aria-label="Gráfico de barras horizontales: comandos rag más usados"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-topics">
        <h2 id="sec-topics">Hot topics</h2>
        <div class="chart-wrap"><canvas id="ch-topics" aria-label="Gráfico de barras horizontales: temas calientes en el período"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" id="card-keywords" aria-labelledby="sec-keywords">
        <h2 id="sec-keywords">Keywords del chat <span id="kw-total" style="font-size:11px;font-weight:400;color:var(--text-faint)"></span></h2>
        <div id="kw-cloud" class="kw-cloud" aria-label="Nube de palabras más usadas en chat"></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-score-trend">
        <h2 id="sec-score-trend">Score trend (promedio diario)</h2>
        <div class="chart-wrap"><canvas id="ch-score-trend" aria-label="Gráfico de líneas: score promedio diario en el período"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-feedback-trend">
        <h2 id="sec-feedback-trend">Feedback por día (positivo vs negativo)</h2>
        <div class="chart-wrap"><canvas id="ch-feedback-trend" aria-label="Gráfico de barras apiladas: feedback positivo y negativo por día"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-ambient">
        <h2 id="sec-ambient">Ambient hooks por dia</h2>
        <div class="chart-wrap"><canvas id="ch-ambient" aria-label="Gráfico de barras: ambient hooks ejecutados por día"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" aria-labelledby="sec-contra">
        <h2 id="sec-contra">Contradicciones por dia</h2>
        <div class="chart-wrap"><canvas id="ch-contra" aria-label="Gráfico de barras: contradicciones detectadas en indexación por día"></canvas><div class="chart-empty">sin datos en el período</div></div>
      </section>
      <section class="chart-card" id="card-index" aria-labelledby="sec-index">
        <h2 id="sec-index">Index</h2>
        <div id="index-content"></div>
      </section>
      <section class="chart-card" id="card-latency-stats" aria-labelledby="sec-latency-stats">
        <h2 id="sec-latency-stats">Latencia (percentiles)</h2>
        <div id="latency-stats-content"></div>
      </section>
      <section class="chart-card" id="card-tune" aria-labelledby="sec-tune">
        <h2 id="sec-tune">Historial de tune</h2>
        <div id="tune-content"></div>
      </section>
      <section class="chart-card" id="card-screentime-apps" aria-labelledby="sec-screentime-apps">
        <h2 id="sec-screentime-apps">Pantalla · top apps</h2>
        <div class="chart-wrap"><canvas id="ch-screentime-apps" aria-label="Gráfico de barras horizontales: top apps usadas según Screen Time"></canvas><div class="chart-empty">sin datos (knowledgeC.db)</div></div>
      </section>
      <section class="chart-card" id="card-screentime-daily" aria-labelledby="sec-screentime-daily">
        <h2 id="sec-screentime-daily">Pantalla · diario <span id="screentime-total" style="font-size:11px;font-weight:400;color:var(--text-faint)"></span></h2>
        <div class="chart-wrap"><canvas id="ch-screentime-daily" aria-label="Gráfico de barras: tiempo de pantalla diario según Screen Time"></canvas><div class="chart-empty">sin datos (knowledgeC.db)</div></div>
      </section>
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

  state.charts.screentimeApps = new Chart(document.getElementById("ch-screentime-apps"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.cyan, 0.5), borderColor: C.cyan, borderWidth: 1, borderRadius: 3 }] },
    options: {
      indexAxis: "y", responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => fmtHm(ctx.parsed.x) } },
      },
      scales: { x: { beginAtZero: true, ticks: { callback: (v) => fmtHm(v) } } },
    },
  });

  state.charts.screentimeDaily = new Chart(document.getElementById("ch-screentime-daily"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: hexAlpha(C.purple, 0.5), borderColor: C.purple, borderWidth: 1, borderRadius: 3 }] },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => fmtHm(ctx.parsed.y) } },
      },
      scales: {
        x: { ticks: { maxRotation: 45 } },
        y: { beginAtZero: true, ticks: { callback: (v) => fmtHm(v) } },
      },
    },
  });
}

function fmtHm(s) {
  s = Math.round(Number(s) || 0);
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return m ? `${h}h ${String(m).padStart(2, "0")}m` : `${h}h`;
  }
  if (s >= 60) return `${Math.floor(s / 60)}m`;
  return `${s}s`;
}

// ── Refresh (mutates existing charts/DOM in place) ────────────────────────

function refresh(d) {
  const k = d.kpis;
  const idx = d.index || {};

  // KPIs
  const kpis = [
    { key: "queries", label: "Queries", value: k.total_queries, cls: "cyan", sub: `${k.total_queries_all_time} all-time` },
    { key: "notes", label: "Notas indexadas", value: idx.notes_files ?? idx.notes ?? "—", cls: "green", sub: `${idx.chunks || 0} chunks · ${idx.notes_titles ?? "—"} títulos únicos` },
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
  renderSignalsPanel(d.signals || {});

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
  const topicData = (d.hot_topics || []).filter(t => t.count >= 2).slice(0, 5);
  state.charts.topics.data.labels = topicData.map(t => t.topic);
  state.charts.topics.data.datasets[0].data = topicData.map(t => t.count);
  state.charts.topics.update("none");
  setEmpty(state.charts.topics, !topicData.length);

  // Chat keyword cloud — font-size tier by log-scaled frequency.
  // Tiers collapse cleanly when few words dominate (e.g. 2 words w/ counts
  // 8 and 2 still render as t5 vs t2 instead of all clumping at the max).
  renderKeywordCloud(d.chat_keywords || []);

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
    const nFiles = idx.notes_files ?? idx.notes ?? 0;
    const nTitles = idx.notes_titles ?? idx.notes ?? 0;
    let rows = `
      <tr><td>Chunks</td><td>${idx.chunks.toLocaleString()}</td></tr>
      <tr><td>Notas (archivos)</td><td>${Number(nFiles).toLocaleString()}</td></tr>
      <tr><td>Títulos únicos</td><td>${Number(nTitles).toLocaleString()}</td></tr>
      <tr><td>Tags</td><td>${idx.tags}</td></tr>
      <tr><td>Carpetas</td><td>${idx.folders}</td></tr>
    `;
    if (idx.top_pagerank && idx.top_pagerank.length) {
      rows += '<tr><td colspan="2" style="color:var(--text-faint);padding-top:10px">Top PageRank</td></tr>';
      for (const pr of idx.top_pagerank) {
        const name = pr.path.split("/").pop().replace(".md", "");
        const cell = pathLink(pr.path, escapeHtml(name), { title: pr.path });
        rows += `<tr><td style="color:var(--cyan)">${cell}</td><td>${pr.score}</td></tr>`;
      }
    }
    idxEl.innerHTML = `<table class="stats-table"><caption class="sr-only">Estadísticas del index del vault: chunks, notas, títulos únicos, tags, carpetas y top notas por PageRank</caption>${rows}</table>`;
  } else {
    idxEl.innerHTML = '<div style="color:var(--text-faint);text-align:center;padding:20px">index no disponible</div>';
  }

  // Latency stats
  const ls = d.latency_stats;
  document.getElementById("latency-stats-content").innerHTML = `
    <table class="stats-table">
      <caption class="sr-only">Latencia por etapa (retrieve, generate, total) en segundos para los percentiles 50 y 95</caption>
      <tr><th scope="col"><span class="sr-only">Etapa</span></th><th scope="col">p50</th><th scope="col">p95</th></tr>
      <tr><td>Retrieve</td><td>${ls.retrieve_p50}s</td><td>${ls.retrieve_p95}s</td></tr>
      <tr><td>Generate</td><td>${ls.generate_p50}s</td><td>${ls.generate_p95}s</td></tr>
      <tr><td style="font-weight:700">Total</td><td style="font-weight:700">${ls.total_p50}s</td><td style="font-weight:700">${ls.total_p95}s</td></tr>
    </table>
    <div style="margin-top:12px;font-size:11px;color:var(--text-faint)">
      Score p50: ${d.score_stats.p50} · p95: ${d.score_stats.p95} · rango: ${d.score_stats.min}–${d.score_stats.max}
    </div>
  `;

  // Screen Time
  const st = d.screentime || {};
  const stAppsCh = state.charts.screentimeApps;
  const stDailyCh = state.charts.screentimeDaily;
  const stTotalEl = document.getElementById("screentime-total");
  if (st.available && (st.top_apps?.length || st.daily?.length)) {
    const apps = (st.top_apps || []).slice(0, 5);
    stAppsCh.data.labels = apps.map(a => a.label || a.bundle);
    stAppsCh.data.datasets[0].data = apps.map(a => a.secs);
    stAppsCh.update("none");
    setEmpty(stAppsCh, !apps.length);

    const daily = st.daily || [];
    stDailyCh.data.labels = daily.map(e => e.day.slice(5));  // MM-DD
    stDailyCh.data.datasets[0].data = daily.map(e => e.secs);
    stDailyCh.update("none");
    setEmpty(stDailyCh, !daily.length);

    if (stTotalEl) {
      stTotalEl.textContent = st.total_label
        ? `· ${st.total_label} · ${st.window_days}d`
        : "";
    }
  } else {
    stAppsCh.data.labels = [];
    stAppsCh.data.datasets[0].data = [];
    stAppsCh.update("none");
    setEmpty(stAppsCh, true);
    stDailyCh.data.labels = [];
    stDailyCh.data.datasets[0].data = [];
    stDailyCh.update("none");
    setEmpty(stDailyCh, true);
    if (stTotalEl) stTotalEl.textContent = "";
  }

  // Tune history
  const tuneEl = document.getElementById("tune-content");
  if (d.tune_history.length) {
    tuneEl.innerHTML = d.tune_history.slice(-5).reverse().map(t => `
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

function renderKeywordCloud(items) {
  const el = document.getElementById("kw-cloud");
  const total = document.getElementById("kw-total");
  if (!el) return;
  if (!items.length) {
    el.innerHTML = '<div class="kw-cloud-empty">sin keywords en el período</div>';
    if (total) total.textContent = "";
    return;
  }
  // Log-scale counts to 5 tiers. log damps the long tail so a term used
  // 40× doesn't reduce everything else to illegible 11px noise.
  const counts = items.map(i => i.count);
  const maxLog = Math.log(Math.max(...counts) + 1);
  const minLog = Math.log(Math.min(...counts) + 1);
  const span = Math.max(maxLog - minLog, 0.0001);
  const sumAll = counts.reduce((a, b) => a + b, 0);
  const frag = document.createDocumentFragment();
  // Shuffle so the cloud feels organic, not sorted by frequency.
  const shuffled = items.slice(0, 5).sort(() => Math.random() - 0.5);
  for (const { word, count } of shuffled) {
    const lv = Math.log(count + 1);
    const tier = Math.min(5, Math.max(1, Math.ceil(((lv - minLog) / span) * 5) || 1));
    const span_ = document.createElement("span");
    span_.className = `kw t${tier}`;
    span_.textContent = word;
    span_.title = `${count} menciones`;
    frag.appendChild(span_);
  }
  el.innerHTML = "";
  el.appendChild(frag);
  if (total) {
    total.textContent = `· ${items.length} términos · ${sumAll} menciones`;
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

// ── Signals panel (2026-04-22) ─────────────────────────────────────────
// Renderiza las señales implícitas al ranker-vivo en los últimos N días:
// open / copy / save / kept / positive_implicit (CTR numerators) vs
// impression (denominator). Fuente: rag_behavior agregado por el
// backend en /api/dashboard.signals.
//
// Motivación: pre-fix no había forma de saber si las modificaciones del
// UX (corrective-path picker, copy events, etc) estaban efectivamente
// alimentando al ranker. Ahora el user ve en una fila si la columna
// CTR-numerator está siendo alimentada.

// Presentation order — positivos (CTR numerator) primero, impression
// al final como denominador. Emojis pegados al label así la fila se
// lee como "📋 copy · 42" visible de un vistazo.
const SIGNAL_LABELS = [
  { key: "copy",               label: "copy",       cls: "cyan",   desc: "user copió contenido del RAG" },
  { key: "open",               label: "open",       cls: "green",  desc: "abrió la nota desde una cita" },
  { key: "save",               label: "save",       cls: "green",  desc: "guardó la respuesta como nota" },
  { key: "kept",               label: "kept",       cls: "green",  desc: "mantuvo una propuesta del ambient agent" },
  { key: "positive_implicit",  label: "positive",   cls: "green",  desc: "señal positiva implícita" },
  { key: "negative_implicit",  label: "negative",   cls: "red",    desc: "señal negativa implícita" },
  { key: "deleted",            label: "deleted",    cls: "red",    desc: "borró una propuesta del ambient agent" },
  { key: "impression",         label: "impression", cls: "yellow", desc: "chunk mostrado al user (denominator)" },
];

function renderSignalsPanel(signals) {
  const grid = document.getElementById("signals-grid");
  const win = document.getElementById("signals-window");
  if (!grid) return;
  const counts = signals.counts || {};
  const bySource = signals.by_source || {};
  const days = signals.window_days || state.days || 30;
  if (win) win.textContent = `· últimos ${days}d`;

  // Include event types that have any count; skip zero-count rows so
  // the panel doesn't get noisy with 7+ always-empty cells on a fresh
  // install. The empty-state message below kicks in when everything
  // is zero.
  const visible = SIGNAL_LABELS.filter(s => (counts[s.key] || 0) > 0);
  if (!visible.length) {
    grid.innerHTML = `<div class="signals-empty">aún no hay señales en los últimos ${days}d — copiá / guardá / rateá una respuesta para empezar a alimentar al ranker-vivo</div>`;
    return;
  }
  grid.innerHTML = visible.map(s => {
    const n = counts[s.key] || 0;
    const src = bySource[s.key] || {};
    const srcBits = Object.entries(src)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k} ${v}`)
      .join(" · ");
    return `
      <div class="signal-cell" title="${s.desc}${srcBits ? '\n\n' + srcBits : ''}">
        <span class="signal-label">${s.label}</span>
        <span class="signal-value ${s.cls}">${n}</span>
        ${srcBits ? `<span class="signal-sub">${srcBits}</span>` : ""}
      </div>
    `;
  }).join("");
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
            <div class="fb-path">${pathLink(p.path, escapeHtml(shortenPath(p.path)))}</div>
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
            <div class="fb-q"><a href="${escapeHtml(chatQueryHref(c.q))}" style="color:inherit;text-decoration:none;border-bottom:1px dotted currentColor;" title="reintentar en chat">${escapeHtml(c.q)}</a></div>
            <div class="fb-meta">debió retornar → <span style="color:var(--orange)">${pathLink(c.missing_path, escapeHtml(shortenPath(c.missing_path)))}</span></div>
            <div class="fb-meta" style="opacity:0.7">en lugar de: ${(c.retrieved || []).map(p => pathLink(p, escapeHtml(shortenPath(p)))).join(" · ") || "(nada)"}</div>
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
      // Each phase tag gets an sr-only label so SR users hear "en curso",
      // "error" or "rechazada" instead of just the colored icon — the
      // SVG itself stays aria-hidden because the surrounding cells
      // (score "…"/err/gate) repeat the info visually but with text
      // that screen readers DO pick up.
      if (ev.phase === "in_flight") {
        tag = `<span class="tag-icon" style="color:var(--cyan)"><span class="sr-only">en curso</span>${icon("refresh", { size: 12, cls: "spin" })}</span>`;
        scoreCell = '<span style="color:var(--cyan)">…</span>';
        latencyCell = '<span style="color:var(--cyan)">en curso</span>';
      } else if (ev.phase === "error") {
        tag = `<span class="tag-icon" style="color:var(--red)"><span class="sr-only">error</span>${icon("warning", { size: 12 })}</span>`;
        scoreCell = '<span style="color:var(--red)">err</span>';
        latencyCell = `<span style="color:var(--red)" title="${escapeHtml(ev.error || "")}">${escapeHtml((ev.error || "fallo").slice(0, 40))}</span>`;
      } else if (ev.gated) {
        tag = `<span class="tag-icon" style="color:var(--text-faint)"><span class="sr-only">rechazada por gate</span>${icon("ban", { size: 12 })}</span>`;
      }
      return `
        <div class="ticker-item">
          <span class="t-time">${timeOf(ev.ts)}</span>
          <span class="t-source ${ev.source || "cli"}">${ev.source || "cli"}</span>
          <span class="t-q" title="${escapeHtml(ev.q || "")}">${tag}<a href="${escapeHtml(chatQueryHref(ev.q || ""))}" style="color:inherit;text-decoration:none;border-bottom:1px dotted currentColor;" title="reintentar en chat" aria-label="Reintentar en chat: ${escapeHtml(ev.q || "(sin texto)")}">${escapeHtml(ev.q || "(sin texto)")}</a></span>
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
          <span class="t-q" title="${escapeHtml(ev.path || "")}"><span class="tag-icon" style="color:var(--orange)">${icon("inbox", { size: 12 })}</span>${pathLink(ev.path, escapeHtml(shortenPath(ev.path || "")))}</span>
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
          <span class="t-q"><span class="tag-icon" style="color:var(--red)">${icon("warning", { size: 12 })}</span>${pathLink(ev.path, escapeHtml(shortenPath(ev.path || "")))}</span>
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

// obsidian:// deep-link wrapper. Returns linked HTML if path present.
function obsidianHref(path) {
  if (!path) return "";
  return `obsidian://open?vault=Notes&file=${encodeURIComponent(path)}`;
}
function pathLink(path, displayHtml, opts = {}) {
  const href = obsidianHref(path);
  if (!href) return displayHtml;
  const title = opts.title || path;
  return `<a href="${escapeHtml(href)}" title="${escapeHtml(title)}" style="color:inherit;text-decoration:none;border-bottom:1px dotted currentColor;">${displayHtml}</a>`;
}
// Re-run a query in the chat UI.
function chatQueryHref(q) {
  return `/chat?q=${encodeURIComponent(q || "")}`;
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

// ── Per-process severity thresholds ──────────────────────────────────
// Maps a process label (as produced by the server) to [warn, hot] cutoffs.
// MEM units: MB. CPU units: % of one core (can exceed 100 on multi-threaded procs).
//
// Labels not listed here stay neutral — deliberately, to avoid false
// positives on:
//   - ollama runners (each model has a different RSS/CPU footprint, and
//     generation legitimately pegs several cores at 400%+);
//   - ad-hoc rag subcommands (`rag morning`, `rag eval`, etc.) that are
//     expected to run hot while they do their job.
//
// Tune these if a legit process keeps tripping red.
const MEM_THRESHOLDS_MB = {
  "obsidian-rag-mcp":        [400, 800],
  "rag web":                 [600, 1200],
  "rag watch":               [200, 500],
  "rag (python)":            [800, 1500],
  "rag (resource_tracker)":  [40, 100],
  "ollama serve":            [400, 800],
  "sqlite-vec-gui":          [400, 800],
  "whatsapp-bridge":         [150, 300],
  "whatsapp-listener":       [150, 300],
  "whatsapp-mcp":            [150, 300],
  "whatsapp-vault-sync":     [150, 300],
};

const CPU_THRESHOLDS_PCT = {
  "rag web":                 [60, 150],
  "rag watch":               [80, 200],
  "ollama serve":            [30, 80],
  "sqlite-vec-gui":          [30, 80],
  "obsidian-rag-mcp":        [60, 150],
  "whatsapp-bridge":         [15, 40],
  "whatsapp-listener":       [15, 40],
  "whatsapp-mcp":            [15, 40],
  "whatsapp-vault-sync":     [15, 40],
};

function severityFor(thresholds, label, value) {
  const cuts = thresholds[label];
  if (!cuts || value == null) return "ok";
  if (value >= cuts[1]) return "hot";
  if (value >= cuts[0]) return "warn";
  return "ok";
}

// ── RAG-stack memory · live stacked area (2s SSE ticks + 60s backfill) ──
// Buckets only our stack: rag python, ollama, sqlite-vec, whatsapp-*.
// System processes outside the rag stack are intentionally excluded.
const MEM = {
  cats: ["rag", "ollama", "sqlite-vec", "whatsapp"],
  labels: { "rag": "rag", "ollama": "ollama",
            "sqlite-vec": "sqlite-vec", "whatsapp": "whatsapp" },
  colors: { "rag": C.orange, "ollama": C.purple,
            "sqlite-vec": C.cyan, "whatsapp": C.green },
  windowMin: 60,
  maxPoints: 1200,
  samples: [],
  current: null,
  chart: null,
  es: null,
  backfillInFlight: null,
};

function memFmtGB(mb) {
  if (mb == null) return "—";
  return (mb / 1024).toFixed(mb < 1024 ? 2 : 1);
}

function memBuildChart() {
  const canvas = document.getElementById("ch-memory");
  if (!canvas) return;
  const datasets = MEM.cats.map((c) => ({
    label: MEM.labels[c],
    data: [],
    borderColor: MEM.colors[c],
    backgroundColor: hexAlpha(MEM.colors[c], 0.45),
    borderWidth: 1,
    pointRadius: 0,
    pointHitRadius: 6,
    tension: 0.25,
    fill: true,
    stack: "mem",
  }));
  MEM.chart = new Chart(canvas, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, padding: 10 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${memFmtGB(ctx.parsed.y)} GB`,
            footer: (items) => {
              const total = items.reduce((a, it) => a + (it.parsed.y || 0), 0);
              return `total: ${memFmtGB(total)} GB`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
        },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { callback: (v) => `${(v / 1024).toFixed(1)} GB` },
        },
      },
    },
  });
}

function memSyncChart() {
  if (!MEM.chart) return;
  const showSeconds = MEM.windowMin <= 5;
  const labels = MEM.samples.map((s) => {
    const d = new Date(s.ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (!showSeconds) return `${hh}:${mm}`;
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  });
  MEM.chart.data.labels = labels;
  MEM.cats.forEach((c, i) => {
    MEM.chart.data.datasets[i].data = MEM.samples.map((s) => (s.by_category && s.by_category[c]) || 0);
  });
  MEM.chart.update("none");
}

function memSyncHeader() {
  const totalEl = document.getElementById("mem-total");
  const deltaEl = document.getElementById("mem-delta");
  if (totalEl && MEM.current) {
    totalEl.innerHTML = `${memFmtGB(MEM.current.total_mb)}<span class="unit">GB</span>`;
  }
  if (deltaEl) {
    const now = (MEM.current && MEM.current.total_mb) || 0;
    const then = MEM.samples.length ? (MEM.samples[0].total_mb || 0) : now;
    const d = now - then;
    const sign = d > 20 ? "up" : d < -20 ? "down" : "flat";
    const arrow = sign === "up" ? "▲" : sign === "down" ? "▼" : "●";
    deltaEl.className = `memcard-delta ${sign}`;
    deltaEl.textContent = `${arrow} ${d >= 0 ? "+" : ""}${memFmtGB(Math.abs(d))} GB`;
  }
  const topEl = document.getElementById("mem-top-list");
  if (topEl) {
    const top = (MEM.current && MEM.current.top) || [];
    topEl.innerHTML = top.length
      ? top.slice(0, 5).map((p) => {
          const sev = severityFor(MEM_THRESHOLDS_MB, p.name, p.mb);
          const tip = sev === "ok" ? p.name
            : `${p.name} · ${sev === "hot" ? "consumo alto" : "consumo elevado"}`;
          return `<li class="${sev}"><span class="name" title="${escapeHtml(tip)}">${escapeHtml(p.name)}</span><span class="val">${memFmtGB(p.mb)} GB</span></li>`;
        }).join("")
      : `<li><span class="name">—</span></li>`;
  }
}

function memTrim() {
  const cutoff = Date.now() - MEM.windowMin * 60_000;
  let samples = MEM.samples.filter((s) => {
    const t = new Date(s.ts).getTime();
    return Number.isFinite(t) && t >= cutoff;
  });
  if (samples.length > MEM.maxPoints) {
    const stride = Math.ceil(samples.length / MEM.maxPoints);
    samples = samples.filter((_, i) => i % stride === 0 || i === samples.length - 1);
  }
  MEM.samples = samples;
}

async function memBackfill() {
  if (MEM.backfillInFlight) return MEM.backfillInFlight;
  MEM.backfillInFlight = (async () => {
    try {
      const resp = await fetch(`/api/system-memory?minutes=${MEM.windowMin}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      MEM.cats = data.categories || MEM.cats;
      const history = data.samples || [];
      const seen = new Set(MEM.samples.map((s) => s.ts));
      for (const s of history) {
        if (!seen.has(s.ts)) MEM.samples.push(s);
      }
      MEM.samples.sort((a, b) => new Date(a.ts) - new Date(b.ts));
      if (data.current) MEM.current = data.current;
      memTrim();
    } catch (_) {
      // silent — live stream will recover
    } finally {
      MEM.backfillInFlight = null;
    }
  })();
  return MEM.backfillInFlight;
}

function memOpenStream() {
  try { if (MEM.es) MEM.es.close(); } catch (_) {}
  const es = new EventSource("/api/system-memory/stream");
  MEM.es = es;
  const dot = document.getElementById("mem-live-dot");
  es.addEventListener("sample", (ev) => {
    let sample; try { sample = JSON.parse(ev.data); } catch (_) { return; }
    MEM.current = sample;
    const last = MEM.samples[MEM.samples.length - 1];
    if (!last || new Date(sample.ts) >= new Date(last.ts)) {
      MEM.samples.push(sample);
    }
    memTrim();
    memSyncChart();
    memSyncHeader();
    if (dot) { dot.style.color = "var(--green)"; dot.textContent = "● live"; }
  });
  es.onerror = () => {
    if (dot) { dot.style.color = "var(--yellow)"; dot.textContent = "● reconectando…"; }
    if (es.readyState === EventSource.CLOSED) setTimeout(memOpenStream, 5000);
  };
}

async function memInit() {
  const winEl = document.getElementById("mem-window");
  if (winEl) {
    winEl.addEventListener("click", async (ev) => {
      const btn = ev.target.closest("button[data-min]");
      if (!btn) return;
      MEM.windowMin = Number(btn.dataset.min);
      for (const b of winEl.querySelectorAll("button")) {
        b.classList.toggle("active", Number(b.dataset.min) === MEM.windowMin);
      }
      MEM.samples = [];
      await memBackfill();
      memSyncChart();
      memSyncHeader();
    });
  }
  memBuildChart();
  await memBackfill();
  memSyncChart();
  memSyncHeader();
  memOpenStream();
}

// ── RAG-stack CPU · live stacked area (2s SSE ticks + 60s backfill) ──
// Same scope + category palette as the memory chart. Values are
// "% of one core" per category — summed across multi-threaded procs
// so a single ollama runner pegging 4 cores shows ~400%. `ncores`
// (reported by the server) is used to render a secondary total label.
const CPU = {
  cats: ["rag", "ollama", "sqlite-vec", "whatsapp"],
  labels: { "rag": "rag", "ollama": "ollama",
            "sqlite-vec": "sqlite-vec", "whatsapp": "whatsapp" },
  colors: { "rag": C.orange, "ollama": C.purple,
            "sqlite-vec": C.cyan, "whatsapp": C.green },
  windowMin: 60,
  maxPoints: 1200,
  samples: [],
  current: null,
  chart: null,
  es: null,
  backfillInFlight: null,
  ncores: 1,
};

function cpuBuildChart() {
  const canvas = document.getElementById("ch-cpu");
  if (!canvas) return;
  const datasets = CPU.cats.map((c) => ({
    label: CPU.labels[c],
    data: [],
    borderColor: CPU.colors[c],
    backgroundColor: hexAlpha(CPU.colors[c], 0.45),
    borderWidth: 1,
    pointRadius: 0,
    pointHitRadius: 6,
    tension: 0.25,
    fill: true,
    stack: "cpu",
  }));
  CPU.chart = new Chart(canvas, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, padding: 10 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y || 0).toFixed(1)}%`,
            footer: (items) => {
              const total = items.reduce((a, it) => a + (it.parsed.y || 0), 0);
              const pctOfMachine = CPU.ncores > 0 ? (total / CPU.ncores).toFixed(1) : "—";
              return `total: ${total.toFixed(1)}%  (${pctOfMachine}% de ${CPU.ncores} cores)`;
            },
          },
        },
      },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { callback: (v) => `${v}%` },
        },
      },
    },
  });
}

function cpuSyncChart() {
  if (!CPU.chart) return;
  const showSeconds = CPU.windowMin <= 5;
  const labels = CPU.samples.map((s) => {
    const d = new Date(s.ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (!showSeconds) return `${hh}:${mm}`;
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  });
  CPU.chart.data.labels = labels;
  CPU.cats.forEach((c, i) => {
    CPU.chart.data.datasets[i].data = CPU.samples.map((s) => (s.by_category && s.by_category[c]) || 0);
  });
  CPU.chart.update("none");
}

function cpuSyncHeader() {
  const totalEl = document.getElementById("cpu-total");
  const subEl = document.getElementById("cpu-sub");
  if (totalEl && CPU.current) {
    totalEl.innerHTML = `${(CPU.current.total_pct || 0).toFixed(0)}<span class="unit">%</span>`;
  }
  if (subEl) {
    const cur = (CPU.current && CPU.current.total_pct) || 0;
    const ncores = CPU.ncores || 1;
    const pctOfMachine = (cur / ncores).toFixed(1);
    subEl.className = "memcard-delta flat";
    subEl.textContent = `${pctOfMachine}% de ${ncores} cores`;
  }
  const topEl = document.getElementById("cpu-top-list");
  if (topEl) {
    const top = (CPU.current && CPU.current.top) || [];
    topEl.innerHTML = top.length
      ? top.slice(0, 5).map((p) => {
          const sev = severityFor(CPU_THRESHOLDS_PCT, p.name, p.pct);
          const tip = sev === "ok" ? p.name
            : `${p.name} · ${sev === "hot" ? "consumo alto" : "consumo elevado"}`;
          return `<li class="${sev}"><span class="name" title="${escapeHtml(tip)}">${escapeHtml(p.name)}</span><span class="val">${(p.pct || 0).toFixed(1)}%</span></li>`;
        }).join("")
      : `<li><span class="name">—</span></li>`;
  }
}

function cpuTrim() {
  const cutoff = Date.now() - CPU.windowMin * 60_000;
  let samples = CPU.samples.filter((s) => {
    const t = new Date(s.ts).getTime();
    return Number.isFinite(t) && t >= cutoff;
  });
  if (samples.length > CPU.maxPoints) {
    const stride = Math.ceil(samples.length / CPU.maxPoints);
    samples = samples.filter((_, i) => i % stride === 0 || i === samples.length - 1);
  }
  CPU.samples = samples;
}

async function cpuBackfill() {
  if (CPU.backfillInFlight) return CPU.backfillInFlight;
  CPU.backfillInFlight = (async () => {
    try {
      const resp = await fetch(`/api/system-cpu?minutes=${CPU.windowMin}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      CPU.cats = data.categories || CPU.cats;
      CPU.ncores = data.ncores || CPU.ncores;
      const history = data.samples || [];
      const seen = new Set(CPU.samples.map((s) => s.ts));
      for (const s of history) {
        if (!seen.has(s.ts)) CPU.samples.push(s);
      }
      CPU.samples.sort((a, b) => new Date(a.ts) - new Date(b.ts));
      if (data.current) CPU.current = data.current;
      cpuTrim();
    } catch (_) {
      // silent — live stream will recover
    } finally {
      CPU.backfillInFlight = null;
    }
  })();
  return CPU.backfillInFlight;
}

function cpuOpenStream() {
  try { if (CPU.es) CPU.es.close(); } catch (_) {}
  const es = new EventSource("/api/system-cpu/stream");
  CPU.es = es;
  const dot = document.getElementById("cpu-live-dot");
  es.addEventListener("sample", (ev) => {
    let sample; try { sample = JSON.parse(ev.data); } catch (_) { return; }
    CPU.current = sample;
    if (sample.ncores) CPU.ncores = sample.ncores;
    const last = CPU.samples[CPU.samples.length - 1];
    if (!last || new Date(sample.ts) >= new Date(last.ts)) {
      CPU.samples.push(sample);
    }
    cpuTrim();
    cpuSyncChart();
    cpuSyncHeader();
    if (dot) { dot.style.color = "var(--green)"; dot.textContent = "● live"; }
  });
  es.onerror = () => {
    if (dot) { dot.style.color = "var(--yellow)"; dot.textContent = "● reconectando…"; }
    if (es.readyState === EventSource.CLOSED) setTimeout(cpuOpenStream, 5000);
  };
}

async function cpuInit() {
  const winEl = document.getElementById("cpu-window");
  if (winEl) {
    winEl.addEventListener("click", async (ev) => {
      const btn = ev.target.closest("button[data-min]");
      if (!btn) return;
      CPU.windowMin = Number(btn.dataset.min);
      for (const b of winEl.querySelectorAll("button")) {
        b.classList.toggle("active", Number(b.dataset.min) === CPU.windowMin);
      }
      CPU.samples = [];
      await cpuBackfill();
      cpuSyncChart();
      cpuSyncHeader();
    });
  }
  cpuBuildChart();
  await cpuBackfill();
  cpuSyncChart();
  cpuSyncHeader();
  cpuOpenStream();
}

// ── WhatsApp scheduled messages ──────────────────────────────────────────
//
// Sección que lista los rows de `rag_whatsapp_scheduled` (backend en
// `/api/whatsapp/scheduled`) con tabs para filtrar por estado y acciones
// inline (cancelar, reprogramar, reenviar). El polling es independiente
// del de `/api/dashboard` (KPIs/charts) — un mini-loop de 30s que se
// pausa con state.paused / document.hidden, igual que el resto.
//
// Idempotencia visual: re-renders comparan una signature `id:status` para
// evitar parpadeos en cada poll cuando nada cambió.
//
// Mobile (<=640px): la tabla se colapsa a cards via CSS (data-label).

const WA_POLL_MS = 30_000;
const WA_TABS = [
  { id: "pending", label: "Pendientes" },
  { id: "sent",    label: "Enviados" },
  { id: "failed",  label: "Fallados" },
  { id: "all",     label: "Todos" },
];

function waInit() {
  if (state.waInitDone) return;
  state.waInitDone = true;
  state.waTab = "pending";
  state.waItems = [];
  state.waLastSig = "";
  waAttachListeners();
  refreshWaScheduled();
  startWaPolling();
}

function startWaPolling() {
  if (state.waPoll) clearInterval(state.waPoll);
  state.waPoll = setInterval(() => {
    if (state.paused || document.hidden) return;
    refreshWaScheduled();
  }, WA_POLL_MS);
}

async function refreshWaScheduled() {
  try {
    // No filter on the GET — we use one fetch and split client-side.
    // It also keeps the per-tab counts always-fresh in one round-trip.
    const res = await fetch("/api/whatsapp/scheduled?limit=200");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.waItems = Array.isArray(data.items) ? data.items : [];
    renderWaScheduled();
  } catch (err) {
    const body = document.getElementById("wa-scheduled-body");
    if (body && !state.waItems.length) {
      body.innerHTML = `<div class="fb-empty" style="color:var(--red)">error cargando: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function waItemsForTab(tab) {
  const all = state.waItems || [];
  if (tab === "pending") return all.filter(it => it.status === "pending");
  if (tab === "sent")    return all.filter(it => it.status === "sent" || it.status === "sent_late");
  if (tab === "failed")  return all.filter(it => it.status === "failed");
  return all;
}

function waCountByTab() {
  const c = { pending: 0, sent: 0, failed: 0, all: 0 };
  for (const it of state.waItems || []) {
    c.all += 1;
    if (it.status === "pending") c.pending += 1;
    else if (it.status === "sent" || it.status === "sent_late") c.sent += 1;
    else if (it.status === "failed") c.failed += 1;
  }
  return c;
}

function renderWaScheduled() {
  const root = document.getElementById("sec-wa-scheduled-card");
  if (!root) return;

  const counts = waCountByTab();

  // Pending badge in the section header.
  const badge = document.getElementById("wa-pending-badge");
  if (badge) badge.textContent = `(${counts.pending} pending)`;

  // Tabs: count + selected state. Don't blow away the buttons (they
  // own click listeners + focus state) — just mutate.
  const body = document.getElementById("wa-scheduled-body");
  for (const tab of WA_TABS) {
    const btn = root.querySelector(`button[data-wa-tab="${tab.id}"]`);
    if (!btn) continue;
    const isActive = (tab.id === state.waTab);
    btn.setAttribute("aria-selected", String(isActive));
    btn.classList.toggle("active", isActive);
    const c = btn.querySelector(".wa-tab-count");
    if (c) c.textContent = counts[tab.id];
    if (isActive && body) body.setAttribute("aria-labelledby", `wa-tab-${tab.id}`);
  }

  if (!body) return;

  // Truly empty corpus → minimal centered message.
  if (!state.waItems.length) {
    if (state.waLastSig !== "EMPTY_ALL") {
      body.innerHTML = `<div class="fb-empty">no hay mensajes programados</div>`;
      state.waLastSig = "EMPTY_ALL";
    }
    return;
  }

  const items = waItemsForTab(state.waTab);
  if (!items.length) {
    const sig = `EMPTY:${state.waTab}`;
    if (state.waLastSig !== sig) {
      const lbl = (WA_TABS.find(t => t.id === state.waTab) || { label: "" }).label.toLowerCase();
      body.innerHTML = `<div class="fb-empty">sin mensajes en "${escapeHtml(lbl)}"</div>`;
      state.waLastSig = sig;
    }
    return;
  }

  // Idempotency: if the signature matches the previous render, skip the
  // innerHTML reset. Signature includes tab + (id, status, scheduled_for_utc)
  // per row so reschedule moves and status flips force re-render.
  const sig = state.waTab + "|" + items
    .map(it => `${it.id}:${it.status}:${it.scheduled_for_utc || ""}`)
    .join(",");
  if (sig === state.waLastSig) return;
  state.waLastSig = sig;

  body.innerHTML = `
    <table class="wa-table">
      <caption class="sr-only">Mensajes de WhatsApp programados — para, mensaje, fecha programada, estado y acciones disponibles</caption>
      <thead>
        <tr>
          <th scope="col">Para</th>
          <th scope="col">Mensaje</th>
          <th scope="col">Programado</th>
          <th scope="col">Estado</th>
          <th scope="col">Acción</th>
        </tr>
      </thead>
      <tbody>
        ${items.map(waRowHtml).join("")}
      </tbody>
    </table>
  `;
}

function waRowHtml(it) {
  const display = waContactDisplay(it);
  const whoCell = waContactCell(it, display);
  const msgFull = it.message_text || "";
  const msgShort = msgFull.length > 60 ? msgFull.slice(0, 60) + "…" : msgFull;
  const sched = waFormatScheduled(it.scheduled_for_utc);
  return `
    <tr data-wa-id="${it.id}" data-wa-status="${escapeHtml(it.status)}">
      <td data-label="Para">${whoCell}</td>
      <td data-label="Mensaje" title="${escapeHtml(msgFull)}"><span class="wa-msg">${escapeHtml(msgShort)}</span></td>
      <td data-label="Programado" class="wa-when">${escapeHtml(sched)}</td>
      <td data-label="Estado">${waStatusChip(it)}</td>
      <td data-label="Acción" class="wa-actions">${waActionsHtml(it)}</td>
    </tr>
  `;
}

function waContactDisplay(it) {
  if (it && it.contact_name && String(it.contact_name).trim()) {
    return String(it.contact_name).trim();
  }
  return waJidToHumanPhone((it && it.jid) || "");
}

function waContactCell(it, displayText) {
  const phone = waJidToPhone((it && it.jid) || "");
  const safe = escapeHtml(displayText);
  if (!phone) return `<span class="wa-who">${safe}</span>`;
  const href = `https://wa.me/${phone}`;
  return `<a class="wa-who" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="Abrir chat en WhatsApp">${safe}</a>`;
}

// "5491100000000@s.whatsapp.net" → "5491100000000". Groups (@g.us) and
// status broadcasts return "" because wa.me doesn't deep-link them.
function waJidToPhone(jid) {
  if (!jid) return "";
  if (jid.includes("@g.us")) return "";
  const part = jid.split("@")[0] || "";
  if (/^\d{6,}$/.test(part)) return part;
  return "";
}

// Format AR cell: "+54 9 11 1234 5678". For non-AR or short numbers
// fall back to "+<digits>" so it's still clickable / readable.
function waJidToHumanPhone(jid) {
  const phone = waJidToPhone(jid);
  if (!phone) return jid || "?";
  if (phone.startsWith("549") && phone.length >= 12) {
    const rest = phone.slice(3);
    const local = rest.slice(-8);
    const area = rest.slice(0, rest.length - 8);
    return `+54 9 ${area} ${local.slice(0, 4)} ${local.slice(4)}`;
  }
  if (phone.startsWith("54") && phone.length >= 11) {
    const rest = phone.slice(2);
    const local = rest.slice(-8);
    const area = rest.slice(0, rest.length - 8);
    return `+54 ${area} ${local.slice(0, 4)} ${local.slice(4)}`;
  }
  return `+${phone}`;
}

// "hoy 14:30" / "mañana 9:00" / "ayer 21:00" / "vie 26 abr 14:30"
// (con año si es otro año). Usa Intl es-AR igual que otras partes
// del dashboard. Strip dots ("vie." → "vie") para consistencia.
function waFormatScheduled(iso) {
  if (!iso) return "?";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const now = new Date();
  const same = (a, b) =>
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate();
  const tomorrow = new Date(now); tomorrow.setDate(now.getDate() + 1);
  const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
  const hm = new Intl.DateTimeFormat("es-AR", {
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(d);
  if (same(d, now)) return `hoy ${hm}`;
  if (same(d, tomorrow)) return `mañana ${hm}`;
  if (same(d, yesterday)) return `ayer ${hm}`;
  const sameYear = d.getFullYear() === now.getFullYear();
  const dateLabel = new Intl.DateTimeFormat("es-AR", {
    weekday: "short", day: "numeric", month: "short",
    year: sameYear ? undefined : "numeric",
  }).format(d).replace(/\./g, "").replace(/,/g, "").replace(/\s+/g, " ").trim();
  return `${dateLabel} ${hm}`;
}

function waStatusChip(it) {
  const s = (it && it.status) || "pending";
  let cls = s, label = s, title = "";
  if (s === "pending") {
    cls = "pending"; label = "pending";
  } else if (s === "sent") {
    cls = "sent"; label = "sent";
    if (it.delta_minutes != null) title = `entregado ${it.delta_minutes}min después de lo programado`;
  } else if (s === "sent_late") {
    cls = "sent-late"; label = "sent late";
    if (it.delta_minutes != null) title = `llegó ${it.delta_minutes}min tarde`;
  } else if (s === "failed") {
    cls = "failed"; label = "failed";
    const bits = [];
    if (it.last_error) bits.push(it.last_error);
    if (it.attempt_count) bits.push(`intentos: ${it.attempt_count}`);
    if (bits.length) title = bits.join(" · ");
  } else if (s === "cancelled") {
    cls = "cancelled"; label = "cancelled";
  }
  const t = title ? ` title="${escapeHtml(title)}"` : "";
  return `<span class="wa-chip wa-chip-${cls}"${t}>${escapeHtml(label)}</span>`;
}

function waActionsHtml(it) {
  const s = it && it.status;
  if (s === "pending") {
    const who = escapeHtml(waContactDisplay(it));
    return `
      <button type="button" class="wa-btn wa-btn-cancel" data-wa-action="cancel" aria-label="Cancelar mensaje a ${who}">Cancelar</button>
      <button type="button" class="wa-btn-link" data-wa-action="reschedule" aria-label="Reprogramar mensaje a ${who}">Reprogramar</button>
    `;
  }
  if (s === "failed") {
    const who = escapeHtml(waContactDisplay(it));
    return `<button type="button" class="wa-btn-link" data-wa-action="resend" aria-label="Reenviar mensaje a ${who}">Reenviar</button>`;
  }
  return `<span class="wa-action-empty">—</span>`;
}

function waAttachListeners() {
  const root = document.getElementById("sec-wa-scheduled-card");
  if (!root) return;
  root.addEventListener("click", (ev) => {
    // Tab switch (delegated, idempotent — same tab is no-op).
    const tab = ev.target.closest("button[data-wa-tab]");
    if (tab) {
      ev.preventDefault();
      if (tab.dataset.waTab === state.waTab) return;
      state.waTab = tab.dataset.waTab;
      state.waLastSig = "";  // force re-render on tab change
      renderWaScheduled();
      return;
    }
    const btn = ev.target.closest("[data-wa-action]");
    if (!btn) return;
    const action = btn.dataset.waAction;
    // The reschedule popover lives in its own <tr> appended after the
    // target row — find the target via data-target-id when we're inside it.
    const popupRow = btn.closest("tr.wa-resched-row");
    let id;
    if (popupRow) {
      id = Number(popupRow.dataset.targetId);
    } else {
      const tr = btn.closest("tr[data-wa-id]");
      if (!tr) return;
      id = Number(tr.dataset.waId);
    }
    if (action === "cancel") return waCancel(id);
    if (action === "resend") return waResend(id);
    if (action === "reschedule") return waToggleReschedule(id);
    if (action === "reschedule-confirm") return waConfirmReschedule(id);
    if (action === "reschedule-cancel") return waCloseReschedule(id);
    if (action === "reschedule-chip") return waApplyChip(btn, id);
  });
}

function waCancel(id) {
  const item = (state.waItems || []).find(x => x.id === id);
  if (!item || item.status !== "pending") return;
  if (!confirm(`¿Cancelar el mensaje programado a ${waContactDisplay(item)}?`)) return;

  // Optimistic update: flip status to cancelled in the local list and
  // re-render. Revert on server NACK.
  const prevStatus = item.status;
  item.status = "cancelled";
  state.waLastSig = "";
  renderWaScheduled();

  fetch(`/api/whatsapp/scheduled/${id}/cancel`, { method: "POST" })
    .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then(data => {
      if (data && data.ok) {
        // Pull authoritative state (sent_at, etc.).
        refreshWaScheduled();
      } else {
        item.status = prevStatus;
        state.waLastSig = "";
        renderWaScheduled();
        alert(`No se pudo cancelar: ${(data && data.reason) || "desconocido"}`);
      }
    })
    .catch(err => {
      item.status = prevStatus;
      state.waLastSig = "";
      renderWaScheduled();
      alert(`Error cancelando: ${err.message}`);
    });
}

function waResend(id) {
  const item = (state.waItems || []).find(x => x.id === id);
  if (!item) return;
  if (!confirm(`¿Mandar este mensaje fallido ahora a ${waContactDisplay(item)}?`)) return;

  // Locate the resend button to disable while in flight.
  const tr = document.querySelector(`tr[data-wa-id="${id}"]`);
  const btn = tr && tr.querySelector('[data-wa-action="resend"]');
  if (btn) { btn.disabled = true; btn.textContent = "enviando…"; }

  const payload = {
    jid: item.jid,
    message_text: item.message_text || "",
  };
  if (item.contact_name) payload.contact_name = item.contact_name;

  fetch("/api/whatsapp/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(r => r.json().then(d => ({ ok: r.ok, status: r.status, data: d })))
    .then(({ ok, status, data }) => {
      if (!ok) {
        const reason = (data && (data.detail || data.reason)) || `HTTP ${status}`;
        if (btn) { btn.disabled = false; btn.textContent = "Reenviar"; }
        alert(`Error reenviando: ${reason}`);
        return;
      }
      // /send doesn't mutate the failed scheduled row — refresh anyway
      // so any new event/state is reflected.
      refreshWaScheduled();
    })
    .catch(err => {
      if (btn) { btn.disabled = false; btn.textContent = "Reenviar"; }
      alert(`Error reenviando: ${err.message}`);
    });
}

function waToggleReschedule(id) {
  // Close any other open popover (single-popover invariant).
  document.querySelectorAll("tr.wa-resched-row").forEach(r => {
    if (r.dataset.targetId !== String(id)) r.remove();
  });
  // Toggle this one.
  const existing = document.querySelector(`tr.wa-resched-row[data-target-id="${id}"]`);
  if (existing) { existing.remove(); return; }

  const item = (state.waItems || []).find(x => x.id === id);
  const tr = document.querySelector(`tr[data-wa-id="${id}"]`);
  if (!item || !tr) return;
  const initialLocal = waIsoToLocalInput(item.scheduled_for_utc);

  const popup = document.createElement("tr");
  popup.className = "wa-resched-row";
  popup.dataset.targetId = String(id);
  popup.innerHTML = `
    <td colspan="5">
      <div class="wa-resched">
        <label class="wa-resched-label">
          <span>Nueva fecha/hora:</span>
          <input type="datetime-local" class="wa-resched-input" value="${escapeHtml(initialLocal)}">
        </label>
        <div class="wa-resched-chips" role="group" aria-label="Atajos rápidos">
          <button type="button" class="wa-chip-quick" data-wa-action="reschedule-chip" data-delta="15">+15min</button>
          <button type="button" class="wa-chip-quick" data-wa-action="reschedule-chip" data-delta="60">+1h</button>
          <button type="button" class="wa-chip-quick" data-wa-action="reschedule-chip" data-delta="180">+3h</button>
          <button type="button" class="wa-chip-quick" data-wa-action="reschedule-chip" data-delta="tomorrow-9">Mañana 9hs</button>
        </div>
        <div class="wa-resched-actions">
          <button type="button" class="wa-btn wa-btn-confirm" data-wa-action="reschedule-confirm">Confirmar</button>
          <button type="button" class="wa-btn-link" data-wa-action="reschedule-cancel">Cancelar</button>
        </div>
        <div class="wa-resched-error" role="alert" aria-live="polite" hidden></div>
      </div>
    </td>
  `;
  tr.parentNode.insertBefore(popup, tr.nextSibling);
  // Focus the input for keyboard users.
  const input = popup.querySelector(".wa-resched-input");
  if (input) input.focus();
}

function waCloseReschedule(id) {
  const popup = document.querySelector(`tr.wa-resched-row[data-target-id="${id}"]`);
  if (popup) popup.remove();
}

function waApplyChip(btn, id) {
  const popup = document.querySelector(`tr.wa-resched-row[data-target-id="${id}"]`);
  if (!popup) return;
  const input = popup.querySelector(".wa-resched-input");
  if (!input) return;
  const delta = btn.dataset.delta;
  const now = new Date();
  let target;
  if (delta === "tomorrow-9") {
    target = new Date(now);
    target.setDate(now.getDate() + 1);
    target.setHours(9, 0, 0, 0);
  } else {
    const min = Number(delta);
    target = new Date(now.getTime() + min * 60_000);
  }
  input.value = waDateToLocalInput(target);
}

function waConfirmReschedule(id) {
  const popup = document.querySelector(`tr.wa-resched-row[data-target-id="${id}"]`);
  if (!popup) return;
  const input = popup.querySelector(".wa-resched-input");
  const errEl = popup.querySelector(".wa-resched-error");
  const setErr = (msg) => {
    if (!errEl) return;
    errEl.textContent = msg;
    errEl.hidden = !msg;
  };
  setErr("");
  const v = input && input.value;
  if (!v) { setErr("elegí una fecha/hora"); return; }

  // datetime-local emits "YYYY-MM-DDTHH:MM" without offset; the server
  // assumes -03:00 (AR) when none is present.
  const confirmBtn = popup.querySelector('[data-wa-action="reschedule-confirm"]');
  if (confirmBtn) { confirmBtn.disabled = true; confirmBtn.textContent = "guardando…"; }

  fetch(`/api/whatsapp/scheduled/${id}/reschedule`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scheduled_for: v }),
  })
    .then(r => r.json().then(d => ({ ok: r.ok, status: r.status, data: d })))
    .then(({ ok, status, data }) => {
      if (!ok || !(data && data.ok)) {
        const reason = (data && (data.detail || data.reason)) || `HTTP ${status}`;
        setErr("no se pudo: " + reason);
        if (confirmBtn) { confirmBtn.disabled = false; confirmBtn.textContent = "Confirmar"; }
        return;
      }
      popup.remove();
      refreshWaScheduled();
    })
    .catch(err => {
      setErr("error: " + err.message);
      if (confirmBtn) { confirmBtn.disabled = false; confirmBtn.textContent = "Confirmar"; }
    });
}

function waIsoToLocalInput(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return waDateToLocalInput(d);
}

function waDateToLocalInput(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
