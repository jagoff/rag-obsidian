/* /logs — dashboard de logs del sistema.
 *
 * Sidebar: lista de services agrupados por dir, ordenada por status
 * (errores primero) + recencia. Click en un service abre el viewer.
 *
 * Viewer: tail del stdout o stderr del service elegido. Cada línea
 * con su level inferido (ok/warn/error/info) y coloreado. Filtro de
 * substring + toggle "sólo warn/error".
 *
 * Auto-refresh cada 8s (sidebar) + 4s (viewer si hay uno abierto).
 * Pausable con el botón del header. Al cambiar de service, el viewer
 * scrollea al final (más reciente abajo, como `tail -f`).
 */

(function () {
  "use strict";

  // ── State ────────────────────────────────────────────────────────────
  const state = {
    services: [],
    selectedKey: null, // "<dir>::<service>"
    selectedKind: "stdout", // "stdout" | "stderr"
    viewerData: null,
    viewerQuery: "",
    viewerOnlyErrors: false,
    sidebarFilter: "all", // "all" | "error" | "warn"
    sidebarQuery: "",
    live: true,
    sidebarTimer: null,
    viewerTimer: null,
  };

  const SIDEBAR_REFRESH_MS = 8000;
  const VIEWER_REFRESH_MS = 4000;
  const VIEWER_TAIL_DEFAULT = 500;

  // ── Helpers ──────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") e.className = attrs[k];
        else if (k === "dataset") Object.assign(e.dataset, attrs[k]);
        else if (k.startsWith("on") && typeof attrs[k] === "function") {
          e.addEventListener(k.slice(2), attrs[k]);
        } else if (k === "html") e.innerHTML = attrs[k];
        else e.setAttribute(k, attrs[k]);
      }
    }
    for (const k of kids) {
      if (k == null) continue;
      e.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
    }
    return e;
  }

  function fmtAge(secs) {
    if (secs == null) return "—";
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.round(secs / 60)}m`;
    if (secs < 86400) return `${Math.round(secs / 3600)}h`;
    return `${Math.round(secs / 86400)}d`;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function highlightQuery(text, query) {
    if (!query) return escapeHtml(text);
    const safe = escapeHtml(text);
    const safeQ = escapeHtml(query).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (!safeQ) return safe;
    return safe.replace(new RegExp(safeQ, "gi"), (m) => `<mark>${m}</mark>`);
  }

  function setUpdated() {
    const t = new Date();
    $("meta-updated").textContent =
      "actualizado " + t.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  // ── Theme toggle (espeja status.js) ──────────────────────────────────
  (function initTheme() {
    const stored = (() => {
      try { return localStorage.getItem("rag-theme"); } catch { return null; }
    })();
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
    function syncIcon() {
      const cur = document.documentElement.getAttribute("data-theme") ||
        (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      $("theme-icon").textContent = cur === "light" ? "☀" : "◐";
    }
    syncIcon();
    $("theme-toggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") ||
        (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      const next = cur === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("rag-theme", next); } catch {}
      syncIcon();
    });
  })();

  // ── Sidebar render ───────────────────────────────────────────────────
  async function fetchServices() {
    try {
      const resp = await fetch("/api/logs", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.services = data.services || [];
      renderTotals(data.totals || {});
      renderSidebar();
      setUpdated();
      clearError();
    } catch (e) {
      showError(`Error cargando services: ${e.message}`);
    }
  }

  function renderTotals(totals) {
    $("total-services").textContent = totals.services ?? 0;
    $("total-error").textContent = totals.error ?? 0;
    $("total-warn").textContent = totals.warn ?? 0;
    $("total-ok").textContent = totals.ok ?? 0;
  }

  function passesSidebarFilter(svc) {
    if (state.sidebarFilter !== "all" && svc.status !== state.sidebarFilter) {
      return false;
    }
    if (state.sidebarQuery) {
      const q = state.sidebarQuery.toLowerCase();
      if (!svc.service.toLowerCase().includes(q) && !svc.dir.toLowerCase().includes(q)) {
        return false;
      }
    }
    return true;
  }

  function renderSidebar() {
    const host = $("service-list");
    host.replaceChildren();

    // Agrupar por dir primero (todos los obsidian-rag juntos, después
    // todos los whatsapp-listener), y dentro de cada grupo mantener el
    // orden global del backend (status primero, recencia después). Si
    // hay <2 dirs distintos no mostramos labels para no agregar ruido.
    const filtered = state.services.filter(passesSidebarFilter);
    if (filtered.length === 0) {
      host.appendChild(el("div", { class: "loading" }, "sin services para mostrar"));
      return;
    }
    const byDir = new Map();
    for (const s of filtered) {
      if (!byDir.has(s.dir)) byDir.set(s.dir, []);
      byDir.get(s.dir).push(s);
    }
    const showLabels = byDir.size > 1;
    // Orden de dirs: el primer dir que aparezca (obsidian-rag) primero,
    // después whatsapp-listener. Si en el futuro hay más, fallback al
    // orden alfabético.
    const dirOrder = ["obsidian-rag", "whatsapp-listener"];
    const sortedDirs = Array.from(byDir.keys()).sort((a, b) => {
      const ai = dirOrder.indexOf(a);
      const bi = dirOrder.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
    const renderItems = [];
    for (const dir of sortedDirs) {
      if (showLabels) renderItems.push({ kind: "label", dir });
      for (const svc of byDir.get(dir)) renderItems.push({ kind: "svc", svc });
    }

    for (const it of renderItems) {
      if (it.kind === "label") {
        host.appendChild(el("div", { class: "service-group-label" }, it.dir));
        continue;
      }
      const svc = it.svc;
      const key = `${svc.dir}::${svc.service}`;
      const item = el("button", {
        type: "button",
        class: "service-item status-" + svc.status + (svc.all_empty ? " status-empty" : "") +
               (state.selectedKey === key ? " is-selected" : ""),
        dataset: { key: key },
        onclick: () => selectService(svc),
      },
        el("span", { class: "service-dot" }),
        el("span", { class: "service-text" },
          el("span", { class: "service-name" }, svc.service),
          el("span", { class: "service-meta" },
            (svc.preview || "(sin actividad)") + " · " + fmtAge(svc.mtime_age_s)
          )
        ),
        svc.error_count_recent > 0
          ? el("span", { class: "service-badge" + (svc.status === "warn" ? " warn" : "") },
              String(svc.error_count_recent))
          : null
      );
      host.appendChild(item);
    }
  }

  // ── Viewer ───────────────────────────────────────────────────────────
  function selectService(svc) {
    const key = `${svc.dir}::${svc.service}`;
    state.selectedKey = key;
    // Por defecto abrir el stderr si hay errores; sino el stdout.
    state.selectedKind = svc.status === "error" ? "stderr" : "stdout";
    // Si el kind elegido no existe (ej. service sólo stdout), fallback al otro.
    const has = (k) => svc.files.some((f) => f.kind === k);
    if (!has(state.selectedKind)) {
      state.selectedKind = has("stdout") ? "stdout" : "stderr";
    }
    state.viewerQuery = "";
    state.viewerOnlyErrors = false;
    $("viewer-search").value = "";
    $("viewer-only-errors").setAttribute("aria-pressed", "false");
    renderSidebar();
    renderViewerHeader(svc);
    fetchAndRenderViewer(true);
  }

  function findSelectedService() {
    if (!state.selectedKey) return null;
    return state.services.find((s) => `${s.dir}::${s.service}` === state.selectedKey) || null;
  }

  function renderViewerHeader(svc) {
    if (!svc) {
      $("viewer-name").textContent = "— elegí un service —";
      $("viewer-sub").textContent = "";
      $("viewer-tabs").replaceChildren();
      $("viewer-controls").hidden = true;
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" },
          "elegí un service del listado",
          el("div", { class: "small" }, "los services con errores recientes están arriba, marcados en rojo")
        )
      );
      return;
    }

    $("viewer-name").textContent = svc.service;
    $("viewer-sub").textContent = `${svc.dir} · ${svc.files.length} archivo${svc.files.length === 1 ? "" : "s"}`;

    // Tabs: stdout / stderr (sólo los que existen)
    const tabs = $("viewer-tabs");
    tabs.replaceChildren();
    for (const file of svc.files) {
      const isActive = file.kind === state.selectedKind;
      const label = file.kind === "stderr" ? "stderr" : "stdout";
      // Badge con error count si es stderr y no está vacío.
      const isErrFile = file.kind === "stderr" && file.size_bytes > 0;
      const tab = el("button", {
        type: "button",
        class: "viewer-tab" + (isActive ? " is-active" : ""),
        role: "tab",
        "aria-selected": isActive ? "true" : "false",
        onclick: () => {
          state.selectedKind = file.kind;
          renderViewerHeader(svc);
          fetchAndRenderViewer(true);
        },
      },
        label,
        isErrFile ? el("span", { class: "tab-badge" }, file.size_human) : null
      );
      tabs.appendChild(tab);
    }

    $("viewer-controls").hidden = false;
  }

  async function fetchAndRenderViewer(scrollToBottom) {
    const svc = findSelectedService();
    if (!svc) return;
    const file = svc.files.find((f) => f.kind === state.selectedKind);
    if (!file) {
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" }, "este service no tiene un archivo para esta tab")
      );
      return;
    }
    if (file.size_bytes === 0) {
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" }, "(archivo vacío)",
          el("div", { class: "small" }, file.name)
        )
      );
      $("viewer-counts").textContent = "";
      return;
    }
    try {
      const params = new URLSearchParams({
        name: file.ref,
        tail: String(VIEWER_TAIL_DEFAULT),
      });
      if (state.viewerQuery) params.set("q", state.viewerQuery);
      if (state.viewerOnlyErrors) params.set("only_errors", "1");
      const resp = await fetch(`/api/logs/file?${params.toString()}`, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.viewerData = data;
      renderViewerBody(data, scrollToBottom);
      renderViewerCounts(data);
    } catch (e) {
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" },
          "error cargando el log",
          el("div", { class: "small" }, e.message)
        )
      );
    }
  }

  function renderViewerBody(data, scrollToBottom) {
    const body = $("viewer-body");
    body.replaceChildren();
    if (!data.lines || data.lines.length === 0) {
      body.appendChild(el("div", { class: "empty-state" },
        "(sin líneas)",
        el("div", { class: "small" },
          data.filtered_by_query || data.filtered_by_level
            ? "ningún match para los filtros activos"
            : "el archivo está vacío"
        )
      ));
      return;
    }
    // Render eficiente: HTML strings concatenadas. ~500 líneas ⇒ DOM
    // creation directo es ~80ms, con HTML directo es <10ms.
    // Para el ts: si todas las líneas son del mismo día, mostramos sólo
    // `HH:MM:SS` (más legible). Si hay ≥2 días distintos, `MM-DD HH:MM`
    // (compromiso entre contexto + ancho — el día completo no entra en 74px).
    const days = new Set();
    for (const ln of data.lines) {
      if (ln.ts) days.add(ln.ts.slice(0, 10));
    }
    const sameDay = days.size <= 1;

    const parts = [];
    for (const ln of data.lines) {
      const cls = "log-line lvl-" + ln.level;
      const lvlLabel = ln.level === "info" ? "·" : ln.level;
      const txt = highlightQuery(ln.text, state.viewerQuery);
      let tsLabel, tsClass = "ts", tsTitle = "";
      if (ln.ts) {
        const day = ln.ts.slice(0, 10);    // 2026-04-26
        const time = ln.ts.slice(11, 19);  // 19:47:50
        tsLabel = sameDay ? time : (day.slice(5) + " " + time.slice(0, 5));
        if (ln.ts_inferred) tsClass += " inferred";
        tsTitle = ln.ts;
      } else {
        tsLabel = "—";
        tsClass += " empty";
        tsTitle = "(sin timestamp en la línea)";
      }
      parts.push(
        `<div class="${cls}">` +
          `<span class="lnum">${ln.n}</span>` +
          `<span class="${tsClass}" title="${escapeHtml(tsTitle)}">${tsLabel}</span>` +
          `<span class="lvl">${lvlLabel}</span>` +
          `<span class="text">${txt}</span>` +
        `</div>`
      );
    }
    body.innerHTML = parts.join("");

    if (scrollToBottom) {
      // Scrollear al final para sentir tipo `tail -f`. Un raf para que
      // el browser termine el layout antes.
      requestAnimationFrame(() => {
        body.scrollTop = body.scrollHeight;
      });
    }
  }

  function renderViewerCounts(data) {
    const c = data.counts || {};
    const total = data.lines_total || 0;
    const shown = data.lines_returned || 0;
    let html = `${shown}/${total} líneas`;
    if (c.error) html += ` · <span class="c-error">${c.error} err</span>`;
    if (c.warn) html += ` · <span class="c-warn">${c.warn} warn</span>`;
    if (c.ok) html += ` · <span class="c-ok">${c.ok} ok</span>`;
    $("viewer-counts").innerHTML = html;
  }

  // ── Error banner ─────────────────────────────────────────────────────
  function showError(msg) {
    const host = $("error-banner-host");
    host.replaceChildren(el("div", { class: "error-banner" }, msg));
  }
  function clearError() {
    $("error-banner-host").replaceChildren();
  }

  // ── Auto-refresh ─────────────────────────────────────────────────────
  function startTimers() {
    stopTimers();
    if (!state.live) return;
    state.sidebarTimer = setInterval(fetchServices, SIDEBAR_REFRESH_MS);
    state.viewerTimer = setInterval(() => {
      if (state.selectedKey) fetchAndRenderViewer(false);
    }, VIEWER_REFRESH_MS);
  }
  function stopTimers() {
    if (state.sidebarTimer) clearInterval(state.sidebarTimer);
    if (state.viewerTimer) clearInterval(state.viewerTimer);
    state.sidebarTimer = null;
    state.viewerTimer = null;
  }

  // ── Wire-up ──────────────────────────────────────────────────────────
  function wireUp() {
    $("live-toggle").addEventListener("click", () => {
      state.live = !state.live;
      $("live-toggle").setAttribute("aria-pressed", state.live ? "true" : "false");
      $("live-label").textContent = state.live ? "auto ON" : "auto OFF";
      if (state.live) startTimers();
      else stopTimers();
    });

    $("refresh-now").addEventListener("click", async () => {
      await fetchServices();
      if (state.selectedKey) await fetchAndRenderViewer(false);
    });

    $("search-services").addEventListener("input", (e) => {
      state.sidebarQuery = e.target.value.trim();
      renderSidebar();
    });

    document.querySelectorAll(".sidebar-filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".sidebar-filter-btn").forEach((b) => {
          b.classList.toggle("is-active", b === btn);
          b.setAttribute("aria-pressed", b === btn ? "true" : "false");
        });
        state.sidebarFilter = btn.dataset.filter;
        renderSidebar();
      });
    });

    let viewerSearchTimer = null;
    $("viewer-search").addEventListener("input", (e) => {
      const v = e.target.value;
      // Debounce 200ms para no rehacer el fetch en cada keystroke.
      if (viewerSearchTimer) clearTimeout(viewerSearchTimer);
      viewerSearchTimer = setTimeout(() => {
        state.viewerQuery = v.trim();
        if (state.selectedKey) fetchAndRenderViewer(false);
      }, 200);
    });

    $("viewer-only-errors").addEventListener("click", () => {
      state.viewerOnlyErrors = !state.viewerOnlyErrors;
      $("viewer-only-errors").setAttribute("aria-pressed", state.viewerOnlyErrors ? "true" : "false");
      if (state.selectedKey) fetchAndRenderViewer(false);
    });

    // Cuando la pestaña va a background, pausar timers para no quemar
    // requests si el user no está mirando.
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopTimers();
      else if (state.live) startTimers();
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────
  wireUp();
  fetchServices().then(() => startTimers());
})();
