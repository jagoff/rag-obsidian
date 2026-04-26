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
  // El "feed global" es una vista virtual: cuando selectedKey === GLOBAL_KEY,
  // el viewer pide /api/logs/errors en vez de /api/logs/file y muestra
  // líneas de varios services mergeadas. selectedKind toma valores
  // 1h/24h/7d en ese modo (ventana temporal).
  const GLOBAL_KEY = "__global__::errors";
  const state = {
    services: [],
    selectedKey: null, // "<dir>::<service>" | GLOBAL_KEY
    selectedKind: "stdout", // "stdout" | "stderr" | "1h"|"24h"|"7d" si GLOBAL_KEY
    viewerData: null,
    viewerQuery: "",
    viewerOnlyErrors: false,
    sidebarFilter: "all", // "all" | "error" | "warn"
    sidebarQuery: "",
    globalSummary: null,  // { total, lines_total, top_services, counts_by_level }
    live: true,
    sidebarTimer: null,
    viewerTimer: null,
  };

  const GLOBAL_WINDOWS = {
    "1h":  { secs: 3600,    label: "1h" },
    "24h": { secs: 86400,   label: "24h" },
    "7d":  { secs: 604800,  label: "7d" },
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

    // Pseudo-service "feed global" al tope. Siempre visible (no cae bajo
    // los filtros de status/search del sidebar) porque es la entrada
    // principal cuando el user quiere "ver qué falló en todo el sistema"
    // sin saber qué service mirar primero.
    const totalErrors = state.services.reduce((a, s) => a + (s.error_count_recent || 0), 0);
    const globalItem = el("button", {
      type: "button",
      class: "service-item global-item status-" + (totalErrors > 0 ? "error" : "ok") +
             (state.selectedKey === GLOBAL_KEY ? " is-selected" : ""),
      onclick: () => selectGlobalFeed(),
      title: "Feed global de errores de todos los services",
    },
      el("span", { class: "service-dot" }),
      el("span", { class: "service-text" },
        el("span", { class: "service-name" }, "feed global"),
        el("span", { class: "service-meta" },
          totalErrors > 0
            ? `${totalErrors} errores · todos los services`
            : "todo OK"
        )
      ),
      totalErrors > 0
        ? el("span", { class: "service-badge" }, String(totalErrors))
        : null
    );
    host.appendChild(globalItem);

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

  function selectGlobalFeed() {
    state.selectedKey = GLOBAL_KEY;
    if (!GLOBAL_WINDOWS[state.selectedKind]) state.selectedKind = "1h";
    state.viewerQuery = "";
    state.viewerOnlyErrors = false;
    $("viewer-search").value = "";
    $("viewer-only-errors").setAttribute("aria-pressed", "false");
    renderSidebar();
    renderViewerHeaderGlobal();
    fetchAndRenderViewer(false);
  }

  function findSelectedService() {
    if (!state.selectedKey || state.selectedKey === GLOBAL_KEY) return null;
    return state.services.find((s) => `${s.dir}::${s.service}` === state.selectedKey) || null;
  }

  function renderViewerHeader(svc) {
    if (!svc) {
      $("viewer-name").textContent = "— elegí un service —";
      $("viewer-sub").textContent = "";
      $("viewer-tabs").replaceChildren();
      $("viewer-controls").hidden = true;
      $("viewer-charts").hidden = true;
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

  function renderViewerHeaderGlobal() {
    $("viewer-name").textContent = "feed global";
    $("viewer-sub").textContent = "errores agregados de todos los services";

    // Tabs: 1h / 24h / 7d (ventanas temporales) en lugar de stdout/stderr.
    const tabs = $("viewer-tabs");
    tabs.replaceChildren();
    for (const [k, w] of Object.entries(GLOBAL_WINDOWS)) {
      const isActive = state.selectedKind === k;
      const tab = el("button", {
        type: "button",
        class: "viewer-tab" + (isActive ? " is-active" : ""),
        role: "tab",
        "aria-selected": isActive ? "true" : "false",
        onclick: () => {
          state.selectedKind = k;
          renderViewerHeaderGlobal();
          fetchAndRenderViewer(false);
        },
      }, w.label);
      tabs.appendChild(tab);
    }
    $("viewer-controls").hidden = false;
  }

  async function fetchAndRenderViewer(scrollToBottom) {
    if (state.selectedKey === GLOBAL_KEY) {
      await fetchAndRenderGlobalFeed();
      return;
    }
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
      $("viewer-charts").hidden = true;
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
      renderCharts(data);
    } catch (e) {
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" },
          "error cargando el log",
          el("div", { class: "small" }, e.message)
        )
      );
    }
  }

  async function fetchAndRenderGlobalFeed() {
    const win = GLOBAL_WINDOWS[state.selectedKind] || GLOBAL_WINDOWS["1h"];
    try {
      const params = new URLSearchParams({
        since_seconds: String(win.secs),
        level: state.viewerOnlyErrors ? "error" : "warn_error",
      });
      const resp = await fetch(`/api/logs/errors?${params.toString()}`, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.viewerData = data;
      // Filter cliente-side por substring (el endpoint no lo hace para
      // mantenerse cacheable a nivel ventana × level).
      let lines = data.lines || [];
      if (state.viewerQuery) {
        const q = state.viewerQuery.toLowerCase();
        lines = lines.filter((l) => l.text.toLowerCase().includes(q) ||
                                     l.service.toLowerCase().includes(q));
      }
      // Adaptar el shape al renderViewerBody (que espera `n` reverse-index).
      const adapted = {
        lines: lines.map((l, i) => ({
          n: lines.length - i,
          text: l.text,
          level: l.level,
          ts: l.ts,
          ts_inferred: l.ts_inferred,
          ts_synthetic: l.ts_synthetic,
          // Extra para el renderer: marcar el service de origen.
          service: l.service,
          ref: l.ref,
          kind: l.kind,
        })),
        lines_total: data.lines_total,
        lines_returned: lines.length,
        counts: data.counts_by_level,
        filtered_by_query: !!state.viewerQuery,
        filtered_by_level: state.viewerOnlyErrors,
        global: true,
        top_services: data.top_services || [],
      };
      renderViewerBody(adapted, false);
      renderViewerCounts(adapted);
      renderCharts(adapted);
    } catch (e) {
      $("viewer-body").replaceChildren(
        el("div", { class: "empty-state" },
          "error cargando el feed global",
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

    const isGlobal = !!data.global;
    const parts = [];
    for (const ln of data.lines) {
      const cls = "log-line lvl-" + ln.level + (isGlobal ? " is-global" : "");
      const lvlLabel = ln.level === "info" ? "·" : ln.level;
      const txt = highlightQuery(ln.text, state.viewerQuery);
      let tsLabel, tsClass = "ts", tsTitle = "";
      if (ln.ts) {
        const day = ln.ts.slice(0, 10);    // 2026-04-26
        const time = ln.ts.slice(11, 19);  // 19:47:50
        tsLabel = sameDay ? time : (day.slice(5) + " " + time.slice(0, 5));
        if (ln.ts_synthetic) {
          tsClass += " synthetic";
          tsTitle = `${ln.ts} (timestamp aproximado, derivado del mtime del archivo)`;
        } else if (ln.ts_inferred) {
          tsClass += " inferred";
          tsTitle = `${ln.ts} (heredado de la línea anterior)`;
        } else {
          tsTitle = ln.ts;
        }
      } else {
        tsLabel = "—";
        tsClass += " empty";
        tsTitle = "(sin timestamp en la línea)";
      }
      // En modo global, además de las cols normales, agregamos una "service
      // chip" antes del texto. Es un span clickeable que cambia el viewer
      // al log de ese service.
      let svcChip = "";
      if (isGlobal && ln.service) {
        svcChip = `<button class="svc-chip" type="button" ` +
                  `data-svc-ref="${escapeHtml(ln.ref || "")}" ` +
                  `data-svc-name="${escapeHtml(ln.service)}" ` +
                  `title="abrir el log de ${escapeHtml(ln.service)}">${escapeHtml(ln.service)}</button>`;
      }
      parts.push(
        `<div class="${cls}">` +
          `<span class="lnum">${ln.n}</span>` +
          `<span class="${tsClass}" title="${escapeHtml(tsTitle)}">${tsLabel}</span>` +
          `<span class="lvl">${lvlLabel}</span>` +
          `<span class="text">${svcChip}${txt}</span>` +
        `</div>`
      );
    }
    body.innerHTML = parts.join("");

    // Wire-up para los chips clickeables del modo global.
    if (isGlobal) {
      body.querySelectorAll(".svc-chip").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          const svcName = btn.dataset.svcName;
          const svc = state.services.find((s) => s.service === svcName);
          if (svc) selectService(svc);
        });
      });
    }

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

  // ── Charts: donut + timeline ──────────────────────────────────────────
  // Ambos derivan del mismo `data.lines` que ya tenemos en estado, así
  // que no hay request adicional. Si las líneas no traen `ts` (logs sin
  // timestamps detectables), el timeline muestra el placeholder "sin
  // timestamps" y el donut sigue funcionando con los counts.

  /** Donut: stroke-dasharray sobre un círculo r=26 con perímetro 2πr.
   *  Cada segmento ocupa una proporción del perímetro = count/total. */
  function renderDonutChart(counts) {
    const order = ["error", "warn", "ok", "info"];
    const total = order.reduce((a, k) => a + (counts[k] || 0), 0);
    const svg = $("charts-donut");

    // Limpiar segmentos previos (mantenemos el donut-bg).
    svg.querySelectorAll(".donut-seg, .donut-center-num").forEach((n) => n.remove());

    if (total === 0) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "donut-center-num");
      t.setAttribute("x", "32"); t.setAttribute("y", "32");
      t.style.fill = "var(--text-faint)";
      t.style.fontSize = "10px";
      t.textContent = "—";
      svg.appendChild(t);
      return;
    }

    const r = 26;
    const C = 2 * Math.PI * r; // ≈163.36
    let offset = 0;
    for (const k of order) {
      const v = counts[k] || 0;
      if (v === 0) continue;
      const len = (v / total) * C;
      const seg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      seg.setAttribute("class", "donut-seg s-" + k);
      seg.setAttribute("cx", "32"); seg.setAttribute("cy", "32"); seg.setAttribute("r", String(r));
      // Empezar en las 12 (rotación -90°).
      seg.setAttribute("transform", "rotate(-90 32 32)");
      seg.setAttribute("stroke-dasharray", `${len} ${C - len}`);
      seg.setAttribute("stroke-dashoffset", String(-offset));
      svg.appendChild(seg);
      offset += len;
    }

    // Número central: total de líneas.
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("class", "donut-center-num");
    t.setAttribute("x", "32"); t.setAttribute("y", "32");
    t.textContent = total >= 1000 ? `${(total / 1000).toFixed(1)}k` : String(total);
    svg.appendChild(t);
  }

  /** Timeline: barras apiladas. Bins automáticos según rango temporal.
   *  Sólo cuenta líneas con ts NO inferred (las inferred son ruido para
   *  el chart porque heredarían a un único bin). */
  function renderTimelineChart(lines) {
    const svg = $("charts-timeline");
    svg.replaceChildren();
    const tsLines = lines.filter((l) => l.ts && !l.ts_inferred);
    if (tsLines.length === 0) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "tl-empty");
      t.setAttribute("x", "300"); t.setAttribute("y", "30");
      t.textContent = "sin timestamps detectables en este log";
      svg.appendChild(t);
      $("tl-range").textContent = "—";
      return;
    }
    // Rango temporal en ms.
    const tsMs = tsLines.map((l) => Date.parse(l.ts));
    const tMin = Math.min(...tsMs);
    const tMax = Math.max(...tsMs);
    const span = Math.max(1, tMax - tMin);
    // Bins: target 60, mínimo 1. Si todo cabe en 1min, 60 bins de 1s.
    // Si cabe en 1h, 60 bins de 1min. Si cabe en 24h, 60 bins de 24min.
    // Si span es 0 (todas las líneas en el mismo segundo), 1 bin.
    const N_BINS = Math.min(60, Math.max(8, Math.floor(tsLines.length / 4)));
    const binSize = Math.max(1, span / N_BINS);
    const bins = Array.from({ length: N_BINS }, () => ({ error: 0, warn: 0, ok: 0, info: 0, total: 0 }));
    for (const l of tsLines) {
      const t = Date.parse(l.ts);
      let idx = Math.floor((t - tMin) / binSize);
      if (idx >= N_BINS) idx = N_BINS - 1;
      bins[idx][l.level] = (bins[idx][l.level] || 0) + 1;
      bins[idx].total++;
    }
    const maxTotal = Math.max(1, ...bins.map((b) => b.total));

    // Geometría: viewBox 600×56, padding superior 4px, gap 1px entre bars.
    const W = 600, H = 56;
    const PAD = 4;
    const usableH = H - PAD - 2;  // 2px abajo para visual breathing
    const barW = W / N_BINS;
    const gap = barW > 4 ? 1 : 0;
    const order = ["error", "warn", "ok", "info"];

    // Axis baseline (línea horizontal de referencia abajo).
    const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
    axis.setAttribute("class", "tl-axis");
    axis.setAttribute("x1", "0"); axis.setAttribute("x2", String(W));
    axis.setAttribute("y1", String(H - 1)); axis.setAttribute("y2", String(H - 1));
    svg.appendChild(axis);

    for (let i = 0; i < N_BINS; i++) {
      const b = bins[i];
      if (b.total === 0) continue;
      const x = i * barW;
      // Apilar de abajo arriba: info → ok → warn → error.
      let yCursor = H - 1;
      for (const lvl of ["info", "ok", "warn", "error"]) {
        const v = b[lvl] || 0;
        if (v === 0) continue;
        const h = (v / maxTotal) * usableH;
        const y = yCursor - h;
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("class", "tl-bar-" + lvl);
        r.setAttribute("x", String(x + gap / 2));
        r.setAttribute("y", String(y));
        r.setAttribute("width", String(Math.max(0.5, barW - gap)));
        r.setAttribute("height", String(h));
        // Tooltip con bin range + counts.
        const binStart = new Date(tMin + i * binSize);
        const binEnd = new Date(tMin + (i + 1) * binSize);
        const fmt = (d) => d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
        const parts = [];
        for (const k of order) if (b[k]) parts.push(`${k}:${b[k]}`);
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = `${fmt(binStart)}–${fmt(binEnd)} · ${parts.join(" ")}`;
        r.appendChild(t);
        svg.appendChild(r);
        yCursor = y;
      }
    }

    // Etiqueta de rango: "HH:MM:SS → HH:MM:SS" o "MM-DD HH:MM → MM-DD HH:MM"
    // según span.
    const dMin = new Date(tMin);
    const dMax = new Date(tMax);
    const sameDay = dMin.toDateString() === dMax.toDateString();
    const fmtTs = (d) => sameDay
      ? d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
      : d.toLocaleString("es-AR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
    $("tl-range").textContent = `${fmtTs(dMin)} → ${fmtTs(dMax)}`;
  }

  function renderCharts(data) {
    const counts = data.counts || {};
    $("leg-error").textContent = counts.error || 0;
    $("leg-warn").textContent = counts.warn || 0;
    $("leg-ok").textContent = counts.ok || 0;
    $("leg-info").textContent = counts.info || 0;
    renderDonutChart(counts);
    renderTimelineChart(data.lines || []);
    $("viewer-charts").hidden = false;
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
