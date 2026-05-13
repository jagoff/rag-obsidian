// Personal Mirror — frontend renderer.
//
// Flow:
//   1. fetch /api/mirror → render 8 blocks sólidos.
//   2. fetch /api/mirror/insights (lazy) → render block "lo que el sistema notó".
//   3. auto-refresh cada 5min.

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;
const SPARK_CHARS = "▁▂▃▄▅▆▇█";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ── Helpers ─────────────────────────────────────────────────────

function fmtRelDays(daysAgo) {
  if (daysAgo == null) return "";
  if (daysAgo === 0) return "hoy";
  if (daysAgo === 1) return "ayer";
  if (daysAgo < 7) return `hace ${daysAgo}d`;
  if (daysAgo < 30) return `hace ${Math.floor(daysAgo / 7)}sem`;
  return `hace ${Math.floor(daysAgo / 30)}mes`;
}

function fmtTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("es-AR", {
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function fmtDateLong(isoDate) {
  if (!isoDate) return "";
  try {
    const d = new Date(isoDate + "T12:00:00");
    return d.toLocaleDateString("es-AR", {
      weekday: "long", day: "numeric", month: "long",
    });
  } catch {
    return isoDate;
  }
}

function makeRow(name, meta, badge) {
  const row = document.createElement("div");
  row.className = "row";

  const nameEl = document.createElement("span");
  nameEl.className = "name";
  nameEl.textContent = name;
  if (badge) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = badge;
    nameEl.appendChild(b);
  }
  row.appendChild(nameEl);

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    row.appendChild(metaEl);
  }
  return row;
}

function emptyState(text) {
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = text || "—";
  return div;
}

function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

// ── Sparkline ─────────────────────────────────────────────────

function sparkline(values, opts = {}) {
  if (!values || !values.length) return "";
  const min = opts.min ?? Math.min(...values, -1);
  const max = opts.max ?? Math.max(...values, 1);
  const range = max - min || 1;
  return values.map((v) => {
    const idx = Math.floor(((v - min) / range) * (SPARK_CHARS.length - 1));
    return SPARK_CHARS[Math.max(0, Math.min(SPARK_CHARS.length - 1, idx))];
  }).join("");
}

// ── Renderers ─────────────────────────────────────────────────

function renderActiveProjects(data) {
  const el = $("#active_projects .content");
  clear(el);
  const items = data?.items || [];
  if (!items.length) {
    el.appendChild(emptyState("sin proyectos activos"));
    return;
  }
  for (const p of items) {
    el.appendChild(makeRow(
      p.name,
      `${p.note_count_30d} notas · ${fmtRelDays(p.days_ago)}`,
    ));
  }
}

function renderTopEntities(data) {
  const el = $("#top_entities .content");
  clear(el);
  const items = data?.items || [];
  if (!items.length) {
    el.appendChild(emptyState("sin entidades extraídas"));
    return;
  }
  for (const e of items) {
    el.appendChild(makeRow(
      e.name,
      `${e.n_mentions_7d} menciones · ${e.n_sources} fuentes`,
      e.kind || null,
    ));
  }
}

function renderMood(data) {
  const el = $("#mood .content");
  clear(el);
  const score = data?.score;
  if (score == null) {
    el.appendChild(emptyState("sin signal de mood hoy · activá con `rag mood enable`"));
    return;
  }
  const lineEl = document.createElement("div");
  lineEl.className = "score-line";
  const valueEl = document.createElement("span");
  valueEl.className = "score-value " + (
    score > 0.1 ? "score-pos" : score < -0.1 ? "score-neg" : "score-zero"
  );
  valueEl.textContent = score >= 0 ? `+${score.toFixed(2)}` : score.toFixed(2);
  lineEl.appendChild(valueEl);

  const lblEl = document.createElement("span");
  lblEl.className = "score-label";
  const lbl = score > 0.5 ? "bien" : score > 0.1 ? "tirando" : score > -0.1 ? "neutro" : score > -0.5 ? "regular" : "bajón";
  lblEl.textContent = lbl;
  lineEl.appendChild(lblEl);
  el.appendChild(lineEl);

  if (data.n_signals) {
    const meta = document.createElement("div");
    meta.className = "sources";
    const sources = (data.sources_used || []).join(", ") || "—";
    meta.textContent = `${data.n_signals} señales · sources: ${sources}`;
    el.appendChild(meta);
  }
}

function renderPendientes(data) {
  const el = $("#pendientes .content");
  clear(el);
  const items = data?.items || [];
  if (!items.length) {
    el.appendChild(emptyState("nada pendiente próximas 12-72h"));
    return;
  }
  for (const p of items) {
    const when = p.when ? fmtTime(p.when) : "";
    el.appendChild(makeRow(p.title, when, p.category));
  }
}

function renderMoodTimeline(data) {
  const el = $("#mood_timeline .content");
  clear(el);
  const days = data?.days || [];
  if (!days.length) {
    el.appendChild(emptyState("sin data histórica · activá `rag mood enable`"));
    return;
  }
  const scores = days.map((d) => d.score);
  const spark = sparkline(scores, { min: -1, max: 1 });
  const sparkEl = document.createElement("div");
  sparkEl.className = "sparkline";
  sparkEl.textContent = spark;
  el.appendChild(sparkEl);

  const meta = document.createElement("div");
  meta.className = "sparkline-meta";
  meta.innerHTML = `<span>${days[0].date}</span><span>${days.length}d</span><span>${days[days.length - 1].date}</span>`;
  el.appendChild(meta);
}

function renderDormantNotes(data) {
  const el = $("#dormant_notes .content");
  clear(el);
  const items = data?.items || [];
  if (!items.length) {
    el.appendChild(emptyState("sin notas dormidas detectadas"));
    return;
  }
  for (const n of items) {
    el.appendChild(makeRow(
      n.title,
      `${fmtRelDays(n.days_ago)} · ${(n.size_bytes / 1024).toFixed(1)}KB`,
    ));
  }
}

function renderSpotifyTop(data) {
  const el = $("#spotify_top .content");
  clear(el);
  const items = data?.items || [];
  if (!items.length) {
    el.appendChild(emptyState("sin escucha registrada"));
    return;
  }
  for (const s of items) {
    el.appendChild(makeRow(
      s.artist,
      `${s.plays} plays · ${s.distinct_tracks} tracks`,
    ));
  }
}

function renderScreenTime(data) {
  const el = $("#screen_time .content");
  clear(el);
  const apps = data?.apps || [];
  if (!apps.length) {
    el.appendChild(emptyState("sin datos de Screen Time"));
    return;
  }
  for (const app of apps.slice(0, 5)) { // Top 5 apps
    const hours = app.total_hours || 0;
    const hoursClass = hours > 4 ? "warn" : hours > 2 ? "caution" : "";
    el.appendChild(makeRow(
      app.app_name || app.bundle_id || "?",
      `${hours.toFixed(1)}h`,
      hoursClass ? "⚠️" : null,
    ));
  }
}

function renderScreenContext(data) {
  const el = $("#screen_context .content");
  clear(el);
  const recent = data?.recent || [];
  if (!recent.length) {
    const today = data?.count_today ?? 0;
    el.appendChild(emptyState(today > 0
      ? `sin captura reciente (hoy: ${today})`
      : "sin captura reciente"));
    return;
  }
  for (const obs of recent) {
    const card = document.createElement("div");
    card.className = "screen-obs";
    const ageMin = obs.age_minutes ?? 0;
    const ageLabel = ageMin === 0 ? "ahora" : `hace ${ageMin}m`;
    const app = obs.app_name || "?";
    const title = (obs.window_title || "").trim();
    const caption = (obs.caption || "").trim();
    card.innerHTML = `
      <div class="screen-obs-head">
        <span class="screen-obs-app">${app}</span>
        <span class="screen-obs-age">${ageLabel}</span>
      </div>
      ${title ? `<div class="screen-obs-title">${title}</div>` : ""}
      ${caption ? `<div class="screen-obs-caption">${caption}</div>` : ""}
    `;
    el.appendChild(card);
  }
  const counts = data?.count_today;
  if (counts && counts > recent.length) {
    const foot = document.createElement("div");
    foot.className = "screen-obs-foot";
    foot.textContent = `+${counts - recent.length} más hoy · ${data?.count_7d ?? counts} esta semana`;
    el.appendChild(foot);
  }
}

function renderObservations(data) {
  const el = $("#observations .content");
  clear(el);
  const stats = [
    { lbl: "queries hoy", num: data?.queries_today ?? 0 },
    { lbl: "anticipate pushes", num: data?.anticipate_pushes_today ?? 0 },
    { lbl: "contradicciones", num: data?.contradictions_open ?? 0 },
    { lbl: "eval runs 7d", num: data?.eval_runs_7d ?? 0 },
  ];
  for (const s of stats) {
    const div = document.createElement("div");
    div.className = "stat";
    div.innerHTML = `<span class="num">${s.num}</span><span class="lbl">${s.lbl}</span>`;
    el.appendChild(div);
  }
}

function renderInsights(data) {
  const el = $("#insights .content");
  clear(el);
  el.classList.remove("insights-loading");
  const items = data?.insights || [];
  if (!items.length) {
    const err = data?.error;
    el.appendChild(emptyState(err ? `LLM falló: ${err.slice(0, 60)}` : "sin insights nuevos"));
    return;
  }
  for (const text of items) {
    const div = document.createElement("div");
    div.className = "insight";
    div.textContent = text;
    el.appendChild(div);
  }
}

// ── Fetch + lifecycle ─────────────────────────────────────────

async function fetchMirror(refresh = false) {
  const url = refresh ? "/api/mirror?refresh=1" : "/api/mirror";
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

async function fetchInsights() {
  const resp = await fetch("/api/mirror/insights");
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function updateHeader(data) {
  $("#header-date").textContent = fmtDateLong(data.date);
  const cacheEl = $("#cache-indicator");
  if (data.cache_hit) {
    cacheEl.textContent = "cached";
    cacheEl.className = "cached";
  } else {
    const wall = (data.wall_s || 0).toFixed(2);
    cacheEl.textContent = `${wall}s`;
    cacheEl.className = "fresh";
  }
}

function updateStatus(text) {
  $("#status-text").textContent = text;
}

async function load(refresh = false) {
  const app = $("#app");
  app.classList.add("loading");
  updateStatus("cargando …");

  try {
    const data = await fetchMirror(refresh);
    const s = data.sources || {};

    renderActiveProjects(s.active_projects);
    renderTopEntities(s.top_entities);
    renderMood(s.mood_today);
    renderPendientes(s.pendientes);
    renderMoodTimeline(s.mood_timeline);
    renderDormantNotes(s.dormant_notes);
    renderSpotifyTop(s.spotify_top);
    renderScreenTime(s.screen_time || {});
    renderScreenContext(s.screen_context || {});
    renderObservations(s.observations || {});
    updateHeader(data);
    updateStatus(`última actualización · ${new Date().toLocaleTimeString("es-AR")}`);

    app.classList.remove("loading");

    // Insights LAZY — corre en background, no bloquea el render principal.
    fetchInsights().then(renderInsights).catch((err) => {
      renderInsights({ insights: [], error: String(err) });
    });
  } catch (err) {
    updateStatus(`error: ${err.message}`);
    app.classList.remove("loading");
  }
}

// ── Init ────────────────────────────────────────────────────────

$("#refresh-btn").addEventListener("click", async () => {
  const btn = $("#refresh-btn");
  btn.classList.add("spinning");
  await load(true);
  setTimeout(() => btn.classList.remove("spinning"), 400);
});

load();
setInterval(() => load(false), REFRESH_INTERVAL_MS);
