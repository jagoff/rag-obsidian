// status.js — renderiza /api/status con auto-refresh cada 10s.
//
// Simplísimo (vanilla, sin frameworks) — espeja el estilo de home.js /
// dashboard.js. El endpoint devuelve un payload agrupado por categoría,
// así que la mayor parte del trabajo es DOM-building straight-forward.
//
// Estados:
//   ok   → verde (servicio OK)
//   warn → amarillo (loaded, pero aún no corrió / info parcial)
//   down → rojo (debería estar corriendo y no está, o exit != 0)
//
// Auto-refresh: 10s default. Pausable con el toggle. La pestaña también
// pausa automáticamente cuando está oculta (document.hidden) para no
// gastar subprocess.run en el server mientras el user está en otra app.
//
// Acciones inline (▶ ejecutar / ■ parar): para servicios launchd-
// controlled (kind=daemon|scheduled), el backend incluye `label` +
// `running` en cada service y la UI muestra un botón que hace
// POST /api/status/action. Util para "trigger digest now" sin abrir
// terminal y acordarse del label exacto.

(function () {
  "use strict";

  const REFRESH_MS = 10000;

  const OVERALL_COPY = {
    ok: {
      title: "Sistema OK",
      sub: "Todos los servicios críticos están respondiendo.",
    },
    degraded: {
      title: "Sistema degradado",
      sub: "Algún servicio no-crítico está caído o pendiente. El chat sigue funcionando.",
    },
    down: {
      title: "Sistema caído",
      sub: "Un servicio CORE no responde. El chat probablemente no funciona.",
    },
  };

  const $content = document.getElementById("content");
  const $updated = document.getElementById("meta-updated");
  const $liveToggle = document.getElementById("live-toggle");
  const $liveLabel = document.getElementById("live-label");
  const $refreshNow = document.getElementById("refresh-now");
  const $themeToggle = document.getElementById("theme-toggle");
  const $themeIcon = document.getElementById("theme-icon");

  let timer = null;
  let live = true;
  let lastPayload = null;

  // ── Fetch + render ──────────────────────────────────────────────────
  async function fetchStatus(forceNoCache) {
    const url = forceNoCache ? "/api/status?nocache=1" : "/api/status";
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  }

  function render(payload) {
    lastPayload = payload;
    const root = document.createElement("div");

    // Overall hero.
    const overall = document.createElement("div");
    overall.className = `overall ${payload.overall}`;
    const copy = OVERALL_COPY[payload.overall] || OVERALL_COPY.degraded;
    overall.innerHTML = `
      <span class="overall-dot" aria-hidden="true"></span>
      <div class="overall-text">
        <div class="overall-title">${escapeHTML(copy.title)}</div>
        <div class="overall-sub">${escapeHTML(copy.sub)}</div>
      </div>
      <div class="overall-counts">
        <div class="overall-count ok"><div class="n">${payload.counts.ok || 0}</div><div class="lbl">ok</div></div>
        <div class="overall-count warn"><div class="n">${payload.counts.warn || 0}</div><div class="lbl">warn</div></div>
        <div class="overall-count down"><div class="n">${payload.counts.down || 0}</div><div class="lbl">down</div></div>
      </div>
    `;
    root.appendChild(overall);

    // Categorías.
    for (const cat of payload.categories || []) {
      const catEl = document.createElement("section");
      catEl.className = "category";
      catEl.setAttribute("data-category", cat.id);

      const counts = { ok: 0, warn: 0, down: 0 };
      for (const s of cat.services || []) counts[s.status] = (counts[s.status] || 0) + 1;

      const head = document.createElement("div");
      head.className = "category-head";
      head.innerHTML = `
        <h2>${escapeHTML(cat.label)}</h2>
        <span class="cat-counts">
          <span class="n-ok">${counts.ok} ok</span> ·
          <span class="n-warn">${counts.warn} warn</span> ·
          <span class="n-down">${counts.down} down</span>
        </span>
      `;
      catEl.appendChild(head);

      const services = document.createElement("div");
      services.className = "services";
      for (const svc of cat.services || []) services.appendChild(renderService(svc));
      catEl.appendChild(services);
      root.appendChild(catEl);
    }

    $content.replaceChildren(root);
    $updated.textContent = `actualizado · ${fmtNow()}`;
  }

  function renderService(svc) {
    const el = document.createElement("div");
    el.className = `service ${svc.status}`;
    el.setAttribute("data-id", svc.id || "");
    el.setAttribute("role", "status");
    el.setAttribute("aria-label", `${svc.name}: ${svc.status}`);
    el.title = `${svc.name} — ${svc.detail}`;

    // Detail: si tiene meta.url (ej. tunnel), hacerlo clickeable.
    let detailHTML = escapeHTML(svc.detail || "");
    if (svc.meta && svc.meta.url && typeof svc.meta.url === "string" && svc.meta.url.startsWith("http")) {
      const url = svc.meta.url;
      // Reemplazar la URL dentro del detail por un <a>. Es seguro porque
      // el detail se construyó server-side con la misma url.
      const safe = escapeHTML(url);
      detailHTML = detailHTML.replace(safe, `<a href="${safe}" target="_blank" rel="noopener">${safe}</a>`);
    }

    // Action button — solo para servicios launchd-controlled (svc.label
    // viene del backend para daemon/scheduled). running=true → stop;
    // running=false → start.
    const hasAction = typeof svc.label === "string" && svc.label.length > 0;
    let actionsHTML = "";
    if (hasAction) {
      const isRunning = svc.running === true;
      const action = isRunning ? "stop" : "start";
      const labelText = isRunning ? "parar" : "ejecutar";
      const icon = isRunning ? "■" : "▶";
      const cls = isRunning ? "service-action stop" : "service-action start";
      const ariaTitle = `${labelText} ${escapeHTML(svc.name || svc.label)}`;
      actionsHTML = `
        <button type="button" class="${cls}"
                data-label="${escapeHTML(svc.label)}"
                data-action="${action}"
                aria-label="${ariaTitle}"
                title="${ariaTitle}">
          <span class="action-icon" aria-hidden="true">${icon}</span>
          <span class="action-label">${labelText}</span>
        </button>
      `;
    }

    el.innerHTML = `
      <span class="service-dot" aria-hidden="true"></span>
      <div class="service-main">
        <div class="service-name">${escapeHTML(svc.name || svc.id || "?")}</div>
        <div class="service-detail">${detailHTML}</div>
      </div>
      ${actionsHTML}
      <span class="service-kind kind-${svc.kind || "probe"}">${escapeHTML(svc.kind || "probe")}</span>
    `;

    // Wire action button (si hay).
    const btn = el.querySelector("button.service-action");
    if (btn) {
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        triggerAction(btn);
      });
    }
    return el;
  }

  // ── Start / stop launchd services ───────────────────────────────────
  // POST /api/status/action con {label, action}. Bloquea el botón con
  // un placeholder "…" mientras la request está in-flight para evitar
  // doble-click. Al terminar, refresca el payload entero (tick) para
  // que el state on-screen sea fresh — el server ya invalida su cache.
  async function triggerAction(btn) {
    const label = btn.getAttribute("data-label");
    const action = btn.getAttribute("data-action");
    if (!label || !action) return;
    if (btn.dataset.busy === "1") return;

    const iconEl = btn.querySelector(".action-icon");
    const labelEl = btn.querySelector(".action-label");
    const prevIcon = iconEl ? iconEl.textContent : "";
    const prevLabel = labelEl ? labelEl.textContent : "";
    btn.dataset.busy = "1";
    btn.disabled = true;
    if (iconEl) iconEl.textContent = "…";
    if (labelEl) labelEl.textContent = action === "start" ? "lanzando" : "parando";

    try {
      const resp = await fetch("/api/status/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, action }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.ok === false) {
        const msg = data.detail || data.stderr || data.stdout || `HTTP ${resp.status}`;
        showActionBanner(`${action} ${label}: ${msg}`, "err");
      } else {
        showActionBanner(`${action === "start" ? "ejecutando" : "parando"} ${label}…`, "ok");
      }
    } catch (e) {
      showActionBanner(`${action} ${label}: ${e.message || e}`, "err");
    } finally {
      btn.dataset.busy = "0";
      btn.disabled = false;
      if (iconEl) iconEl.textContent = prevIcon;
      if (labelEl) labelEl.textContent = prevLabel;
      // Re-fetch para que el botón rote a stop/start si correspondió.
      tick(true);
    }
  }

  function showActionBanner(msg, kind) {
    const banner = document.createElement("div");
    banner.className = kind === "err" ? "error-banner" : "info-banner";
    banner.textContent = msg;
    $content.prepend(banner);
    setTimeout(() => banner.remove(), 3500);
  }

  function escapeHTML(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtNow() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  // ── Refresh loop ────────────────────────────────────────────────────
  async function tick(forceNoCache) {
    try {
      const payload = await fetchStatus(forceNoCache);
      render(payload);
    } catch (e) {
      console.error("[status] fetch failed", e);
      // Si ya hay contenido renderizado, mostrar banner pero no limpiar.
      const banner = document.createElement("div");
      banner.className = "error-banner";
      banner.textContent = `Error consultando /api/status: ${e.message || e}`;
      if (lastPayload) {
        $content.prepend(banner);
        setTimeout(() => banner.remove(), 4000);
      } else {
        $content.replaceChildren(banner);
      }
      $updated.textContent = `error · ${fmtNow()}`;
    }
  }

  function startLoop() {
    if (timer) clearInterval(timer);
    timer = setInterval(() => {
      if (document.hidden) return;  // pausa auto cuando la pestaña no está visible
      tick(false);
    }, REFRESH_MS);
  }

  function stopLoop() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  function setLive(on) {
    live = on;
    $liveToggle.setAttribute("aria-pressed", on ? "true" : "false");
    $liveLabel.textContent = on ? "auto-refresh ON" : "auto-refresh OFF";
    if (on) startLoop(); else stopLoop();
  }

  // ── Wiring ──────────────────────────────────────────────────────────
  $liveToggle.addEventListener("click", () => setLive(!live));
  $refreshNow.addEventListener("click", () => tick(true));

  // Theme toggle — mismo mecanismo que dashboard/home (localStorage
  // "rag-theme" = "light" | "dark"). El inline <script> del <head> ya
  // aplicó el theme antes del paint, así que acá sólo toggleamos.
  $themeToggle.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("rag-theme", next); } catch (e) {}
  });

  // ── Insights: latency sparkline (real) + mock heatmap ──────────────
  // El sparkline de latencia se alimenta de /api/status/latency (ver
  // server.py). El payload trae 25 buckets horarios con p50/p95/p99 y
  // un summary con deltas vs baseline 7d. Las otras cards del insights
  // grid son HTML estático con datos dummy hasta que las shippeemos
  // una por una.

  const $latSparkline = document.getElementById("lat-sparkline");
  const $latCount = document.getElementById("lat-count");
  const $latP95 = document.getElementById("lat-p95-1h");
  const $latDelta = document.getElementById("lat-p95-delta");

  async function fetchLatency() {
    try {
      const resp = await fetch("/api/status/latency", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderLatency(data);
    } catch (e) {
      console.error("[status] latency fetch failed", e);
      // Dejamos el placeholder "—" en el valor pero apagamos el delta
      // para no mentir con 0%.
      if ($latDelta) {
        $latDelta.textContent = "sin datos";
        $latDelta.className = "insight-delta neutral";
      }
    }
  }

  function renderLatency(payload) {
    const series = Array.isArray(payload.series) ? payload.series : [];
    const summary = payload.summary || {};

    // Valor grande: p95 de la última hora con datos.
    if (summary.p95_1h_ms != null) {
      $latP95.textContent = fmtMs(summary.p95_1h_ms);
    } else {
      $latP95.textContent = "—";
    }

    // Delta vs baseline 7d. Regla: >+20% es "bad" (rojo), 0-20% "worse"
    // (amarillo sutil), negativo es "better" (verde).
    if (summary.delta_p95_pct != null) {
      const d = summary.delta_p95_pct;
      const arrow = d > 0 ? "↑" : d < 0 ? "↓" : "→";
      const cls = d > 20 ? "bad" : d > 5 ? "worse" : d < -5 ? "better" : "neutral";
      $latDelta.textContent = `${arrow} ${Math.abs(d).toFixed(1)}% vs 7d`;
      $latDelta.className = `insight-delta ${cls}`;
      $latDelta.title = `p95 baseline 7d: ${fmtMs(summary.p95_baseline_ms)} · última hora: ${fmtMs(summary.p95_1h_ms)}`;
    } else {
      $latDelta.textContent = "sin baseline";
      $latDelta.className = "insight-delta neutral";
    }

    // Meta caption: count de queries en 24h + nº de buckets con data.
    const nBuckets = series.filter((s) => s.count > 0).length;
    $latCount.textContent = `${summary.count_24h || 0} queries · ${nBuckets}/${series.length} horas activas`;

    drawSparkline($latSparkline, series, summary.p95_baseline_ms);
  }

  // ── SVG sparkline draw ──────────────────────────────────────────────
  // Vanilla SVG, cero libs. Dos líneas superpuestas (p50 verde, p95
  // amarillo), un área sutil bajo p95 para dar peso, y una línea
  // punteada con el baseline 7d p95 (si hay). Escala Y arranca en 0 y
  // sube al max observado (con 10% de padding arriba).
  function drawSparkline(svg, series, baselineMs) {
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const vbW = 400;
    const vbH = 64;
    const padL = 2;
    const padR = 2;
    const padT = 6;
    const padB = 6;
    const plotW = vbW - padL - padR;
    const plotH = vbH - padT - padB;

    // Max value (incluye baseline para que la dashed-line no se salga).
    const values = [];
    for (const s of series) {
      if (s.p50_ms != null) values.push(s.p50_ms);
      if (s.p95_ms != null) values.push(s.p95_ms);
    }
    if (baselineMs != null) values.push(baselineMs);
    if (values.length === 0) {
      // Sin datos — placeholder text.
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", vbW / 2);
      t.setAttribute("y", vbH / 2);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("dominant-baseline", "central");
      t.setAttribute("fill", "var(--text-faint)");
      t.setAttribute("font-size", "10");
      t.textContent = "sin datos de latencia en las últimas 24h";
      svg.appendChild(t);
      return;
    }
    const maxV = Math.max(...values) * 1.1;

    const xForIdx = (i) =>
      padL + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW);
    const yForMs = (v) => padT + plotH - (v / maxV) * plotH;

    // Gap hints: ticks verticales sutiles en buckets sin datos.
    for (let i = 0; i < series.length; i++) {
      if (series[i].count === 0) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        const x = xForIdx(i);
        line.setAttribute("x1", x);
        line.setAttribute("y1", padT + plotH - 2);
        line.setAttribute("x2", x);
        line.setAttribute("y2", padT + plotH + 1);
        line.setAttribute("class", "sl-gap-hint");
        svg.appendChild(line);
      }
    }

    // Baseline dashed line (7d p95).
    if (baselineMs != null) {
      const by = yForMs(baselineMs);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", padL);
      line.setAttribute("y1", by);
      line.setAttribute("x2", padL + plotW);
      line.setAttribute("y2", by);
      line.setAttribute("class", "sl-baseline");
      svg.appendChild(line);
    }

    // Build path strings for p50 + p95. Gaps rompen el path (M en vez de L).
    const p50d = buildPath(series, "p50_ms", xForIdx, yForMs);
    const p95d = buildPath(series, "p95_ms", xForIdx, yForMs);

    // Area bajo p95 (sólo tramos contiguos con datos).
    const areaD = buildAreaPath(series, "p95_ms", xForIdx, yForMs, padT + plotH);
    if (areaD) {
      const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
      area.setAttribute("d", areaD);
      area.setAttribute("class", "sl-p95-area");
      svg.appendChild(area);
    }

    if (p95d) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", p95d);
      p.setAttribute("class", "sl-p95");
      svg.appendChild(p);
    }
    if (p50d) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", p50d);
      p.setAttribute("class", "sl-p50");
      svg.appendChild(p);
    }

    // Dot en el último punto con datos (destaca el "ahora").
    const lastIdx = lastIdxWithData(series, "p95_ms");
    if (lastIdx >= 0) {
      const cx = xForIdx(lastIdx);
      const cy95 = yForMs(series[lastIdx].p95_ms);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy95);
      dot.setAttribute("r", "2.5");
      dot.setAttribute("class", "sl-dot");
      dot.setAttribute("stroke", "var(--yellow)");
      svg.appendChild(dot);
    }
  }

  function buildPath(series, key, xForIdx, yForMs) {
    let d = "";
    let started = false;
    for (let i = 0; i < series.length; i++) {
      const v = series[i][key];
      if (v == null) { started = false; continue; }
      const x = xForIdx(i);
      const y = yForMs(v);
      d += (started ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2) + " ";
      started = true;
    }
    return d.trim();
  }

  function buildAreaPath(series, key, xForIdx, yForMs, baseY) {
    // Agrupa en runs contiguos y cierra cada uno contra baseY.
    const runs = [];
    let current = [];
    for (let i = 0; i < series.length; i++) {
      const v = series[i][key];
      if (v == null) {
        if (current.length) { runs.push(current); current = []; }
      } else {
        current.push({ x: xForIdx(i), y: yForMs(v) });
      }
    }
    if (current.length) runs.push(current);

    let d = "";
    for (const run of runs) {
      if (run.length < 2) continue;
      d += "M" + run[0].x.toFixed(2) + " " + baseY.toFixed(2) + " ";
      for (const p of run) d += "L" + p.x.toFixed(2) + " " + p.y.toFixed(2) + " ";
      d += "L" + run[run.length - 1].x.toFixed(2) + " " + baseY.toFixed(2) + " Z ";
    }
    return d.trim();
  }

  function lastIdxWithData(series, key) {
    for (let i = series.length - 1; i >= 0; i--) {
      if (series[i][key] != null) return i;
    }
    return -1;
  }

  function fmtMs(ms) {
    if (ms == null) return "—";
    if (ms >= 1000) return (ms / 1000).toFixed(1) + "s";
    return Math.round(ms) + "ms";
  }

  // ── Heatmap mock (PREVIEW) ──────────────────────────────────────────
  // Generate 7 rows × 24 cols de cuadraditos con un patrón semi-random
  // pero determinístico, para que el diseño se vea con datos verosímiles
  // (no todos verdes, no todos random). Real version leerá de un endpoint
  // que persiste samples del status probe por hora.
  function buildHeatmapMock() {
    const container = document.getElementById("heatmap-mock");
    if (!container) return;
    const days = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"];
    // Patrón verosímil: la mayoría 100%; algunos spikes de 95/low los
    // jueves y domingos a las 3-4am (ingest heavy); 99s aislados.
    function classFor(d, h) {
      if ((d === 3 && h === 3) || (d === 6 && h === 4)) return "uptime-low";  // downtime real
      if ((d === 3 && h === 4) || (d === 6 && h === 5)) return "uptime-95";   // recovery
      if ((h === 3 || h === 4) && d % 2 === 0) return "uptime-99";             // ingest slowdowns
      if (h >= 2 && h <= 5 && d === 0) return "uptime-99";                      // monday morning warmup
      if (d === 6 && h === 23) return "uptime-none";                            // sin data futura
      return "uptime-100";
    }
    const frag = document.createDocumentFragment();
    for (let d = 0; d < 7; d++) {
      const row = document.createElement("div");
      row.className = "heatmap-row";
      const label = document.createElement("span");
      label.className = "heatmap-label";
      label.textContent = days[d];
      row.appendChild(label);
      const grid = document.createElement("div");
      grid.className = "heatmap-grid";
      for (let h = 0; h < 24; h++) {
        const cell = document.createElement("div");
        cell.className = `heatmap-cell ${classFor(d, h)}`;
        cell.title = `${days[d]} ${String(h).padStart(2, "0")}h`;
        grid.appendChild(cell);
      }
      row.appendChild(grid);
      frag.appendChild(row);
    }
    const axis = document.createElement("div");
    axis.className = "heatmap-axis";
    axis.innerHTML = "<span>00</span><span>06</span><span>12</span><span>18</span><span>24</span>";
    frag.appendChild(axis);
    container.appendChild(frag);
  }

  // Kick-off: primer fetch inmediato + loop + insights.
  tick(true);
  fetchLatency();
  buildHeatmapMock();
  startLoop();

  // Ligar refresh manual para también refetchear latency (no dejar el
  // sparkline stale si el user clickeó "refrescar" porque notó algo
  // raro).
  const _origRefresh = $refreshNow.onclick;
  $refreshNow.addEventListener("click", () => fetchLatency());

  // Refetch del latency cada 60s (alineado con el cache TTL server-side).
  setInterval(() => {
    if (!document.hidden) fetchLatency();
  }, 60000);
})();
