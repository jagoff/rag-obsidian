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

    el.innerHTML = `
      <span class="service-dot" aria-hidden="true"></span>
      <div class="service-main">
        <div class="service-name">${escapeHTML(svc.name || svc.id || "?")}</div>
        <div class="service-detail">${detailHTML}</div>
      </div>
      <span class="service-kind kind-${svc.kind || "probe"}">${escapeHTML(svc.kind || "probe")}</span>
    `;
    return el;
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

  // Kick-off: primer fetch inmediato + loop.
  tick(true);
  startLoop();
})();
