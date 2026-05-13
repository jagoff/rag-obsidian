// panel-ambient.mjs — paneles de ambiente: weather, web (Chrome top week),
// bookmarks, youtube, drive, spotify.

import { escapeHTML, fmtTimeAgo, youtubeUrl, renderPanelList } from "./core.mjs";

export function renderWeather(payload) {
  const wf = payload.weather_forecast;
  const panel = document.getElementById("p-weather");
  if (!panel) return;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");

  if (!wf || (!wf.days && !wf.current && !Array.isArray(wf))) {
    body.innerHTML = `<div class="empty">sin datos del clima</div>`;
    count.textContent = "—";
    return;
  }

  const loc = wf.location || "";
  const cur = wf.current || {};
  const headerParts = [];
  if (loc) headerParts.push(escapeHTML(loc.split(",")[0]));
  if (cur.description) headerParts.push(escapeHTML(cur.description));
  if (cur.temp_C != null) headerParts.push(`${cur.temp_C}°C`);
  // Probabilidad de lluvia AHORA — siempre se muestra (regla 2026-05-11
  // user: "Weather tiene que informar la probabilidad de lluvia"). Antes
  // solo aparecía cuando ≥30%, lo que dejaba el panel sin la métrica
  // clave en días secos (que es información útil: "0% = no llueve").
  const curRain = cur.rain_probability_pct;
  if (curRain != null && curRain !== "") {
    headerParts.push(`💧 ${curRain}%`);
  }
  const headerHTML = headerParts.length
    ? `<div class="row-meta" style="margin-bottom: var(--space-3); font-size: 13px; color: var(--text);">
        ${headerParts.join(" · ")}
       </div>`
    : "";

  const days = Array.isArray(wf) ? wf : (wf.days || wf.forecast || []);
  const dayIcon = (desc) => {
    const d = (desc || "").toLowerCase();
    if (/lluvi|chuva|rain/.test(d)) return "🌧";
    if (/tormen|trueno|thunder/.test(d)) return "⛈";
    if (/nub|nublad|cloud/.test(d)) return "☁";
    if (/parcial|partly/.test(d)) return "⛅";
    if (/despej|sole|clear|sun/.test(d)) return "☀";
    if (/niebl|fog|mist/.test(d)) return "🌫";
    return "·";
  };
  const dayLabel = (dateStr) => {
    if (!dateStr) return "";
    // Usar fecha LOCAL para que "hoy"/"mañana" no shifteen por timezone.
    const localDate = (offsetDays = 0) => {
      const d = new Date();
      d.setDate(d.getDate() + offsetDays);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    };
    if (dateStr === localDate(0)) return "hoy";
    if (dateStr === localDate(1)) return "mañana";
    try {
      const d = new Date(dateStr + "T12:00");
      return d.toLocaleDateString("es-AR", { weekday: "short", day: "2-digit" });
    } catch { return dateStr.slice(5); }
  };

  const rows = days.slice(0, 1).map((d) => {
    const icon = dayIcon(d.description);
    const tempRange = (d.minC != null && d.maxC != null)
      ? `${d.minC}°–${d.maxC}°`
      : (d.avgC != null ? `${d.avgC}°` : "");
    // Probabilidad de lluvia SIEMPRE — even 0% es info útil para el user.
    // Threshold ≥30% de antes ocultaba la métrica los días secos.
    const rain = Number(d.chanceofrain) || 0;
    const rainLabel = `💧 ${rain}%`;
    const metaBits = [
      d.description ? escapeHTML(d.description) : "",
      rainLabel,
    ].filter(Boolean);
    return `<div class="row" style="padding: 4px 0;">
      <div class="row-main">
        <div class="row-title" style="display: flex; gap: 8px; align-items: center;">
          <span style="font-size: 16px; min-width: 20px;">${icon}</span>
          <span><strong>${escapeHTML(dayLabel(d.date))}</strong> · ${tempRange}</span>
        </div>
        ${metaBits.length ? `<div class="row-meta" style="margin-left: 28px;">${metaBits.join(" · ")}</div>` : ""}
      </div>
    </div>`;
  });

  body.innerHTML = headerHTML + rows.join("");
  count.textContent = cur.temp_C != null ? `${cur.temp_C}°C` : (days.length ? `${days.length}d` : "—");
}

export function renderWeb(payload) {
  const items = payload.signals?.chrome_top_week || [];
  const rows = items.slice(0, 5).map((it) => ({
    title: it.title || it.url,
    meta: [
      it.url ? new URL(it.url).hostname : null,
      it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
    ].filter(Boolean),
    aside: it.visit_count ? String(it.visit_count) : null,
    href: it.url,
  }));
  renderPanelList("p-web", rows, { emptyText: "sin actividad" });
}

export function renderBookmarks(payload) {
  const items = payload.signals?.chrome_bookmarks || [];
  const rows = items.slice(0, 5).map((it) => ({
    title: it.name || it.url,
    meta: [
      it.folder ? it.folder.split("/").pop() : null,
      it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
    ].filter(Boolean),
    aside: it.visit_count ? String(it.visit_count) : null,
    href: it.url,
  }));
  renderPanelList("p-bookmarks", rows, { emptyText: "sin bookmarks recientes" });
}

export function renderYouTube(payload) {
  const items = payload.signals?.youtube_watched || [];
  const rows = items.slice(0, 5).map((it) => ({
    title: it.title || "",
    meta: [
      it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
    ].filter(Boolean),
    href: youtubeUrl(it.video_id) || it.url,
  }));
  renderPanelList("p-youtube", rows, { emptyText: "sin videos" });
}

export function renderDrive(payload) {
  const items = payload.signals?.drive_recent || [];
  const rows = items.slice(0, 5).map((it) => ({
    title: it.name || "",
    meta: [
      it.modified ? fmtTimeAgo(it.modified) : null,
      it.owner_email ? it.owner_email.split("@")[0] : null,
    ].filter(Boolean),
    href: it.webViewLink || it.web_view_link || it.url || it.link || null,
  }));
  renderPanelList("p-drive", rows, { emptyText: "sin actividad reciente" });
}

export function renderSpotify(payload) {
  const sp = payload.signals?.spotify;
  const panel = document.getElementById("p-spotify");
  if (!panel) return;
  // Caso "empty" (server siempre devuelve payload desde 2026-05-13):
  // tabla `rag_spotify_log` vacía + Spotify cerrado. Mostrar placeholder
  // accionable en vez de hidear (UX previo era confuso — user no sabía
  // si el panel estaba roto o feature off).
  if (!sp || sp.state === "empty" || (!sp.now_playing && !(sp.recent_today || []).length)) {
    panel.hidden = false;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    if (count) count.textContent = "—";
    if (body) {
      const msg = sp?.message || "Sin historial — daemon spotify-poll no corriendo.";
      body.innerHTML = `<div class="empty">${escapeHTML(msg)}</div>`;
    }
    return;
  }
  panel.hidden = false;

  const fmtSecs = (s) => {
    if (!s || s < 1) return "";
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return sec >= 5 ? `${m}m ${Math.round(sec)}s` : `${m}m`;
  };
  const trackHref = (id) => id || null;

  const np = sp.now_playing;
  const recent = sp.recent_today || [];

  // Custom render — hero album-art block for np + compact list for recent.
  // Bypass renderPanelList porque el hero con tapa grande no encaja en el
  // row template estándar; el resto sigue siendo una lista normal.
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");

  let html = "";

  if (np) {
    const isPlaying = np.state === "playing";
    const stateBadge = isPlaying ? "▶ ahora" : "⏸ pausado";
    const stateClass = isPlaying ? "spotify-state-playing" : "spotify-state-paused";
    const art = np.art_url
      ? `<img class="spotify-art" src="${escapeHTML(np.art_url)}" alt="${escapeHTML(np.album || np.name || "")}" loading="lazy">`
      : `<div class="spotify-art spotify-art-placeholder" aria-hidden="true">🎧</div>`;
    const href = trackHref(np.track_id);
    const meta = [
      `<span class="${stateClass}">${escapeHTML(stateBadge)}</span>`,
      escapeHTML(np.artist),
      np.album ? escapeHTML(np.album) : null,
    ].filter(Boolean).join(" · ");
    const heroInner = `
      ${art}
      <div class="spotify-hero-text">
        <div class="row-title">${escapeHTML(np.name)}</div>
        <div class="row-meta">${meta}</div>
      </div>`;
    html += href
      ? `<a class="spotify-hero row--linked" href="${escapeHTML(href)}">${heroInner}</a>`
      : `<div class="spotify-hero">${heroInner}</div>`;
  }

  const npId = np?.track_id;
  const rest = recent.filter((t) => t.track_id !== npId).slice(0, 4);
  if (rest.length) {
    html += `<div class="spotify-recent">`;
    for (const t of rest) {
      const ago = fmtTimeAgo(new Date(t.first_seen * 1000).toISOString());
      const metaParts = [escapeHTML(t.artist)];
      if (ago && ago !== "ahora") metaParts.push(escapeHTML(ago));
      const aside = t.duration_played_s > 30
        ? `<span class="row-aside">${escapeHTML(fmtSecs(t.duration_played_s))}</span>`
        : "";
      const inner = `<div class="row-main">
        <div class="row-title">${escapeHTML(t.name)}</div>
        <div class="row-meta">${metaParts.join(" · ")}</div>
      </div>${aside}`;
      const href = trackHref(t.track_id);
      html += href
        ? `<a class="row row--linked" href="${escapeHTML(href)}">${inner}</a>`
        : `<div class="row">${inner}</div>`;
    }
    html += `</div>`;
  }

  if (!html) {
    html = `<div class="empty">sin actividad hoy</div>`;
    panel.classList.add("is-empty");
  } else {
    panel.classList.remove("is-empty");
  }
  body.innerHTML = html;

  if (count) {
    const totalToday = recent.length + (np && !recent.some((t) => t.track_id === npId) ? 1 : 0);
    count.textContent = totalToday;
    count.classList.remove("has-warning", "has-critical");
    count.classList.add("has-items");
  }
}

// ── Health panel (Apple Health ETL) ──────────────────────────────────────

function _sparklineSVG(values, opts = {}) {
  // ASCII-like SVG sparkline, 100×24px, nulls = gap. Same dimensions as
  // mood sparkline so the panels line up visually when stacked.
  const vals = values || [];
  const numeric = vals.filter((v) => v != null && Number.isFinite(v));
  if (numeric.length < 2) return "";
  const min = Math.min(...numeric);
  const max = Math.max(...numeric);
  const range = max - min || 1;
  const w = 100;
  const h = 24;
  const stepX = vals.length > 1 ? w / (vals.length - 1) : 0;
  const points = vals.map((v, i) => {
    if (v == null || !Number.isFinite(v)) return null;
    const x = i * stepX;
    const y = h - ((v - min) / range) * (h - 2) - 1;
    return { x, y };
  });
  // Build path skipping nulls.
  let path = "";
  let pen = "M";
  for (const p of points) {
    if (p == null) { pen = "M"; continue; }
    path += `${pen}${p.x.toFixed(1)},${p.y.toFixed(1)} `;
    pen = "L";
  }
  const stroke = opts.stroke || "currentColor";
  return `<svg class="health-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true"><path d="${path.trim()}" fill="none" stroke="${stroke}" stroke-width="1.5"/></svg>`;
}

function _ringPct(value, goal) {
  if (!value || !goal) return null;
  return Math.min(100, Math.round((value / goal) * 100));
}

// ── Peekaboo screen observations ──────────────────────────────────────────

function _fmtAgo(tsEpoch) {
  if (!tsEpoch) return "";
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - tsEpoch));
  if (sec < 60) return "ahora";
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
  return `${Math.floor(sec / 86400)}d`;
}

export function renderPeekaboo(payload) {
  const obs = payload.signals?.screen_observations || [];
  const panel = document.getElementById("p-peekaboo");
  if (!panel) return;
  if (!obs.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
  if (count) {
    count.textContent = obs.length;
    count.classList.add("has-items");
  }

  // Layout: hero (latest) + grid de thumbnails compactos.
  const latest = obs[0];
  const rest = obs.slice(1, 6);

  const heroCaption = (latest.caption || "").trim();
  const isNoAnswer = !heroCaption || heroCaption === "<no answer>";
  const captionText = isNoAnswer ? "— sin caption —" : heroCaption;

  const heroHTML = `
    <a class="peekaboo-hero" href="${escapeHTML(latest.thumb_url || "")}" target="_blank" rel="noopener">
      ${latest.thumb_url
        ? `<img class="peekaboo-art" src="${escapeHTML(latest.thumb_url)}" alt="${escapeHTML(latest.app_name || "")}" loading="lazy">`
        : `<div class="peekaboo-art peekaboo-art-placeholder" aria-hidden="true">👁</div>`}
      <div class="peekaboo-text">
        <div class="row-title">${escapeHTML(latest.app_name || "?")} <span class="peekaboo-ago">· ${escapeHTML(_fmtAgo(latest.ts))}</span></div>
        <div class="row-meta peekaboo-caption ${isNoAnswer ? "is-empty" : ""}">${escapeHTML(captionText)}</div>
      </div>
    </a>`;

  const thumbsHTML = rest.length ? `
    <div class="peekaboo-thumbs">
      ${rest.map(o => `
        <a class="peekaboo-thumb" href="${escapeHTML(o.thumb_url || "")}" target="_blank" rel="noopener" title="${escapeHTML((o.app_name || "?") + " · " + (o.caption || "").slice(0, 80))}">
          ${o.thumb_url
            ? `<img src="${escapeHTML(o.thumb_url)}" alt="" loading="lazy">`
            : `<div class="peekaboo-thumb-placeholder">👁</div>`}
          <span class="peekaboo-thumb-meta">${escapeHTML(o.app_name || "?")} · ${escapeHTML(_fmtAgo(o.ts))}</span>
        </a>
      `).join("")}
    </div>
  ` : "";

  body.innerHTML = heroHTML + thumbsHTML;
}


export function renderHealth(payload) {
  const h = payload.signals?.health;
  const panel = document.getElementById("p-health");
  if (!panel) return;
  if (!h) {
    // Sin export: surface placeholder accionable (mismo patrón que
    // Spotify cuando spotify_log está vacío).
    panel.hidden = false;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    if (count) count.textContent = "—";
    if (body) {
      body.innerHTML = `<div class="empty">Sin export Apple Health · correr <code>rag health-import</code> después de Export Health Data desde iPhone</div>`;
    }
    return;
  }
  panel.hidden = false;

  const fmt = (n, d = 0) => n == null ? "—" : Number(n).toLocaleString("es-AR", {maximumFractionDigits: d});
  const steps = h.steps;
  const kcal = h.active_kcal;
  const exMin = h.exercise_min;
  const standH = h.stand_hours;
  const hrRest = h.resting_hr;
  const hrv = h.hrv_sdnn;
  const dist = h.distance_km;
  const flights = h.flights_climbed;
  const nW = h.n_workouts;

  const kcalPct = _ringPct(kcal, h.active_kcal_goal);
  const exPct = _ringPct(exMin, h.exercise_goal);
  const standPct = _ringPct(standH, h.stand_goal);

  const ringHTML = (label, pct, val, goal, unit) => {
    if (pct == null) return "";
    const cls = pct >= 100 ? "ring-done" : pct >= 60 ? "ring-on" : "ring-low";
    return `<div class="health-ring ${cls}" title="${escapeHTML(label)} ${val}/${goal} ${unit}">
      <div class="health-ring-bar"><div class="health-ring-fill" style="width:${pct}%"></div></div>
      <div class="health-ring-meta">${escapeHTML(label)} · ${fmt(val)}/${fmt(goal)} ${unit} · ${pct}%</div>
    </div>`;
  };

  const stalePill = h.stale
    ? `<span class="health-stale" title="último día con data">stale · ${escapeHTML(h.date || "")}</span>`
    : "";

  const sparkSteps = _sparklineSVG(h.spark_steps_14d, {stroke: "var(--cyan)"});
  const sparkKcal  = _sparklineSVG(h.spark_kcal_14d,  {stroke: "var(--amber)"});
  const sparkHrv   = _sparklineSVG(h.spark_hrv_14d,   {stroke: "var(--green, #1db954)"});

  const wTypes = (h.workout_types || "").split(",").filter(Boolean).slice(0, 3).join(", ");
  const workoutLine = nW
    ? `<div class="health-workouts">🏃 ${nW} workout${nW > 1 ? "s" : ""}${wTypes ? " · " + escapeHTML(wTypes) : ""}</div>`
    : "";

  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
  if (count) {
    count.textContent = fmt(steps);
    count.classList.remove("has-warning", "has-critical");
    if (steps && steps > 5000) count.classList.add("has-items");
  }

  body.innerHTML = `
    <div class="health-summary">
      <div class="health-kv"><span class="kv-label">pasos</span><span class="kv-val">${fmt(steps)}</span>${sparkSteps}</div>
      <div class="health-kv"><span class="kv-label">kcal</span><span class="kv-val">${fmt(kcal)}</span>${sparkKcal}</div>
      <div class="health-kv"><span class="kv-label">HRV</span><span class="kv-val">${fmt(hrv, 1)}</span>${sparkHrv}</div>
      <div class="health-kv"><span class="kv-label">HR rest</span><span class="kv-val">${fmt(hrRest)}</span></div>
      <div class="health-kv"><span class="kv-label">distancia</span><span class="kv-val">${fmt(dist, 2)} km</span></div>
      ${flights ? `<div class="health-kv"><span class="kv-label">pisos</span><span class="kv-val">${fmt(flights)}</span></div>` : ""}
    </div>
    ${ringHTML("activity", kcalPct, kcal, h.active_kcal_goal, "kcal")}
    ${ringHTML("ejercicio", exPct, exMin, h.exercise_goal, "min")}
    ${ringHTML("stand", standPct, standH, h.stand_goal, "hr")}
    ${workoutLine}
    ${stalePill}
  `;
}
