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
  const rows = [];

  if (np) {
    const isPlaying = np.state === "playing";
    const stateBadge = isPlaying ? "▶ ahora" : "⏸ pausado";
    const stateClass = isPlaying ? "spotify-state-playing" : "spotify-state-paused";
    const meta = [{ cls: stateClass, text: stateBadge }, np.artist];
    if (np.album) meta.push(np.album);
    rows.push({
      title: np.name,
      meta,
      href: trackHref(np.track_id),
    });
  }

  const npId = np?.track_id;
  const rest = recent.filter((t) => t.track_id !== npId).slice(0, 4);
  for (const t of rest) {
    const meta = [t.artist];
    const ago = fmtTimeAgo(new Date(t.first_seen * 1000).toISOString());
    if (ago && ago !== "ahora") meta.push(ago);
    rows.push({
      title: t.name,
      meta,
      aside: t.duration_played_s > 30 ? fmtSecs(t.duration_played_s) : null,
      href: trackHref(t.track_id),
    });
  }
  renderPanelList("p-spotify", rows, {
    emptyText: "sin actividad hoy",
    showCount: true,
  });
  // Override count para mostrar el total de hoy incluyendo el np.
  const countEl = panel.querySelector("[data-count]");
  if (countEl) {
    const totalToday = recent.length + (np && !recent.some((t) => t.track_id === npId) ? 1 : 0);
    countEl.textContent = totalToday;
  }
}
