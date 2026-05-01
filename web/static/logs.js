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
    rankings: null,        // payload completo del último /api/logs/rankings
    rankingsWindow: null,  // segundos seleccionados (1h/24h/7d)
    rankingsTimer: null,
  };

  const GLOBAL_WINDOWS = {
    "1h":  { secs: 3600,    label: "1h" },
    "24h": { secs: 86400,   label: "24h" },
    "7d":  { secs: 604800,  label: "7d" },
  };

  const SIDEBAR_REFRESH_MS = 8000;
  const VIEWER_REFRESH_MS = 4000;
  const VIEWER_TAIL_DEFAULT = 500;
  const RANKINGS_REFRESH_MS = 12000;
  const RANKINGS_TOP_N = 5;
  // Default window: 24h. El select del header lo override-a y persistimos
  // la elección en localStorage para que sobreviva refresh/restart.
  const RANKINGS_WINDOW_DEFAULT = 86400;
  const RANKINGS_WINDOW_KEY = "rag-logs-rankings-window";

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

  // ── Rankings ────────────────────────────────────────────────────────
  // Panel "Rankings" arriba del layout: 5 cards top-N con dimensiones
  // rankeables (services con más errores, más warns, errores que más se
  // repiten, errores más recientes, logs más ruidosos). Click en cualquier
  // item navega al log correspondiente.
  //
  // Polling: 12s (más lento que sidebar 8s porque el endpoint scanea todos
  // los archivos y es ~30-50ms vs 5-10ms del index).

  function loadRankingsWindow() {
    let stored = null;
    try { stored = parseInt(localStorage.getItem(RANKINGS_WINDOW_KEY) || "", 10); } catch {}
    if (stored && [3600, 86400, 604800].includes(stored)) return stored;
    return RANKINGS_WINDOW_DEFAULT;
  }
  function saveRankingsWindow(secs) {
    try { localStorage.setItem(RANKINGS_WINDOW_KEY, String(secs)); } catch {}
  }

  function fmtRelativeTs(ts) {
    if (!ts) return "—";
    let dt;
    try { dt = new Date(ts); } catch { return ts; }
    if (isNaN(dt.getTime())) return ts;
    const diffMs = Date.now() - dt.getTime();
    const sec = Math.max(0, Math.round(diffMs / 1000));
    if (sec < 60) return `hace ${sec}s`;
    if (sec < 3600) return `hace ${Math.round(sec / 60)}m`;
    if (sec < 86400) return `hace ${Math.round(sec / 3600)}h`;
    return `hace ${Math.round(sec / 86400)}d`;
  }

  // Fade-collapse del whitespace múltiple para que el preview en el
  // ranking de errores recientes / patterns no muestre 5 espacios seguidos.
  function collapseWs(s) { return String(s || "").replace(/\s+/g, " ").trim(); }

  // Navegación a un service desde un click en cualquier ranking. Si el
  // service NO está en `state.services` (caso raro: la ventana de
  // rankings es 7d pero el sidebar/index sólo lista servicios con logs
  // recientes), refrescamos el sidebar primero antes de intentar de
  // nuevo.
  async function navigateToService(serviceName, dirHint) {
    const matchByDir = (s) => s.service === serviceName &&
      (!dirHint || s.dir === dirHint);
    let svc = state.services.find(matchByDir);
    if (!svc) {
      // Probable: state.services tiene match por nombre pero otro dir.
      // Soltamos el dirHint y aceptamos el primer match por nombre.
      svc = state.services.find((s) => s.service === serviceName);
    }
    if (!svc) {
      // Refrescar y volver a buscar.
      await fetchServices();
      svc = state.services.find(matchByDir) ||
            state.services.find((s) => s.service === serviceName);
    }
    if (svc) selectService(svc);
  }

  function rankingItem({ rank, label, sub, count, onClick, ariaLabel, title }) {
    const inner = el("div", { class: "ranking-text" });
    inner.appendChild(el("span", { class: "ranking-text-main" }, label));
    if (sub) {
      inner.appendChild(el("span", { class: "ranking-text-meta" }, sub));
    }
    const item = el("li", {
      class: "ranking-item",
      role: "button",
      tabindex: "0",
      onclick: onClick,
      onkeydown: (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick(e);
        }
      },
      "aria-label": ariaLabel || label,
      title: title || label,
    },
      inner,
      el("span", { class: "ranking-count" }, String(count)),
    );
    return item;
  }

  function emptyRankingItem(text) {
    return el("li", { class: "ranking-empty" }, text || "sin datos en esta ventana");
  }

  async function fetchRankings() {
    if (!state.rankingsWindow) state.rankingsWindow = loadRankingsWindow();
    const totalsHost = $("rankings-totals");
    if (totalsHost) totalsHost.classList.add("is-loading");
    const params = new URLSearchParams({
      since_seconds: String(state.rankingsWindow),
      top_n: String(RANKINGS_TOP_N),
    });
    try {
      const resp = await fetch(`/api/logs/rankings?${params.toString()}`, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.rankings = data;
      renderRankings();
    } catch (e) {
      // Silencioso: si el endpoint falla, el panel queda con el último
      // render. El error general del page (sidebar fail) ya se muestra
      // en el banner.
      console.warn("[rankings] fetch failed:", e);
    } finally {
      if (totalsHost) totalsHost.classList.remove("is-loading");
    }
  }

  function renderRankings() {
    const data = state.rankings;
    if (!data) return;
    const r = data.rankings || {};
    const totalsHost = $("rankings-totals");
    if (totalsHost) {
      const t = data.totals || {};
      const errs = t.errors ?? 0;
      const warns = t.warns ?? 0;
      const services = t.services_with_errors ?? 0;
      totalsHost.textContent =
        `${errs} error${errs === 1 ? "" : "es"} · ${warns} warning${warns === 1 ? "" : "s"} · `
        + `${services} service${services === 1 ? "" : "s"} afectado${services === 1 ? "" : "s"}`;
    }

    // Card 1: services con más errores.
    const c1 = $("rk-services-errors");
    c1.replaceChildren();
    const list1 = r.services_by_errors || [];
    if (list1.length === 0) {
      c1.appendChild(emptyRankingItem("sin errores en esta ventana 🎉"));
    } else {
      for (const it of list1) {
        c1.appendChild(rankingItem({
          label: it.service,
          count: it.count,
          onClick: () => navigateToService(it.service),
          ariaLabel: `${it.service}, ${it.count} errores. Abrir log.`,
          title: `${it.count} errores · click para abrir el log`,
        }));
      }
    }

    // Card 2: services con más warnings.
    const c2 = $("rk-services-warns");
    c2.replaceChildren();
    const list2 = r.services_by_warns || [];
    if (list2.length === 0) {
      c2.appendChild(emptyRankingItem("sin warnings en esta ventana"));
    } else {
      for (const it of list2) {
        c2.appendChild(rankingItem({
          label: it.service,
          count: it.count,
          onClick: () => navigateToService(it.service),
          ariaLabel: `${it.service}, ${it.count} warnings. Abrir log.`,
          title: `${it.count} warnings · click para abrir el log`,
        }));
      }
    }

    // Card 3: errores que más se repiten (patrones clusterizados).
    const c3 = $("rk-error-patterns");
    c3.replaceChildren();
    const list3 = r.error_patterns || [];
    if (list3.length === 0) {
      c3.appendChild(emptyRankingItem("sin patrones de error agrupados"));
    } else {
      for (const it of list3) {
        const services = (it.services || []).slice(0, 3).join(", ");
        const more = (it.services || []).length > 3 ? ` +${it.services.length - 3}` : "";
        const sub = `${services}${more} · último ${fmtRelativeTs(it.last_ts)}`;
        const example = collapseWs(it.example || it.signature);
        c3.appendChild(rankingItem({
          label: example,
          sub,
          count: it.count,
          onClick: () => {
            // Abrir el primer service afectado del pattern.
            const target = (it.services && it.services[0]) || null;
            if (target) navigateToService(target);
          },
          ariaLabel: `${example}, ${it.count} ocurrencias en ${(it.services || []).length} service${(it.services || []).length === 1 ? "" : "s"}.`,
          title: example,
        }));
      }
    }

    // Card 4: errores más recientes.
    const c4 = $("rk-recent-errors");
    c4.replaceChildren();
    const list4 = r.recent_errors || [];
    if (list4.length === 0) {
      c4.appendChild(emptyRankingItem("sin errores recientes 🎉"));
    } else {
      for (const it of list4) {
        const text = collapseWs(it.text);
        const sub = `${it.service} · ${fmtRelativeTs(it.ts)}`;
        c4.appendChild(rankingItem({
          label: text,
          sub,
          count: fmtRelativeTs(it.ts).replace("hace ", ""),
          onClick: () => navigateToService(it.service, it.dir),
          ariaLabel: `${it.service}: ${text}`,
          title: text,
        }));
      }
    }

    // Card 5: logs más ruidosos.
    const c5 = $("rk-noisy-logs");
    c5.replaceChildren();
    const list5 = r.noisy_logs || [];
    if (list5.length === 0) {
      c5.appendChild(emptyRankingItem("sin actividad en esta ventana"));
    } else {
      for (const it of list5) {
        c5.appendChild(rankingItem({
          label: it.service,
          count: it.count,
          onClick: () => navigateToService(it.service),
          ariaLabel: `${it.service}, ${it.count} líneas. Abrir log.`,
          title: `${it.count} líneas · click para abrir el log`,
        }));
      }
    }

    // Card 6: latency outliers (p99 por service). El backend reporta
    // p50 + p99 + max + n; mostramos `p99ms` como count visible y `p50 / n` en sub.
    const c6 = $("rk-latency-outliers");
    c6.replaceChildren();
    const list6 = r.latency_outliers || [];
    if (list6.length === 0) {
      c6.appendChild(emptyRankingItem("no hay datos de latency en esta ventana"));
    } else {
      for (const it of list6) {
        const sub = `p50 ${formatMs(it.p50_ms)} · max ${formatMs(it.max_ms)} · n=${it.n}`;
        c6.appendChild(rankingItem({
          label: it.service,
          sub,
          count: formatMs(it.p99_ms),
          onClick: () => navigateToService(it.service),
          ariaLabel: `${it.service}: p99 ${formatMs(it.p99_ms)}, ${it.n} muestras.`,
          title: `${it.service} · p99=${formatMs(it.p99_ms)} · click para abrir el log`,
        }));
      }
    }

    // Card 7: silent services. Items son services con kind=daemon|scheduled
    // que están loaded en launchd pero llevan tiempo sin escribir logs.
    // El "count" visible es el tiempo de silencio formateado.
    const c7 = $("rk-silent-services");
    c7.replaceChildren();
    const list7 = r.silent_services || [];
    if (list7.length === 0) {
      c7.appendChild(emptyRankingItem("ningún service silencioso 🎉"));
    } else {
      for (const it of list7) {
        const isNeverRan = it.kind === "never_ran";
        const silenceLabel = isNeverRan
          ? "nunca corrió"
          : fmtDuration(it.silence_seconds || 0);
        const sub = `${it.kind}${it.category ? " · " + it.category : ""}${
          it.last_activity ? " · última " + fmtRelativeTs(it.last_activity) : ""
        }`;
        c7.appendChild(rankingItem({
          label: it.service,
          sub,
          count: silenceLabel,
          onClick: () => {
            // Intentar mapear el label launchd al service log basename.
            // El backend devuelve `label` con prefijo (com.fer...). El
            // service log tiene basename sin prefijo. Heurística: strip
            // `com.fer.obsidian-rag-` o `com.fer.whatsapp-` y probar.
            const guess = (it.label || "")
              .replace(/^com\.fer\.obsidian-rag-/, "")
              .replace(/^com\.fer\.whatsapp-/, "")
              .replace(/^com\.fer\./, "");
            if (guess) navigateToService(guess);
          },
          ariaLabel: `${it.service}: ${isNeverRan ? "nunca corrió" : `silencio hace ${silenceLabel}`}. Abrir log.`,
          title: `${it.service} · ${silenceLabel} · click para abrir el log`,
        }));
      }
    }

    // Card 8: growth rate (bytes/min). Mostramos bytes/min como count;
    // sub muestra lines/min para contexto.
    const c8 = $("rk-growth-rate");
    c8.replaceChildren();
    const list8 = r.growth_rate || [];
    if (list8.length === 0) {
      c8.appendChild(emptyRankingItem("sin actividad en esta ventana"));
    } else {
      for (const it of list8) {
        const sub = `${it.lines_per_min.toFixed(1)} líneas/min · total ${formatBytes(it.total_bytes_in_window)}`;
        c8.appendChild(rankingItem({
          label: it.service,
          sub,
          count: formatBytes(it.bytes_per_min) + "/m",
          onClick: () => navigateToService(it.service),
          ariaLabel: `${it.service}: ${formatBytes(it.bytes_per_min)}/min. Abrir log.`,
          title: `${it.service} · ${formatBytes(it.bytes_per_min)}/min · click para abrir el log`,
        }));
      }
    }

    // Card 9: nuevos error patterns (regression detector). Solo se popula
    // si la ventana es ≥ 2h. Si no, mostramos hint para cambiar la ventana.
    const c9 = $("rk-new-error-patterns");
    c9.replaceChildren();
    const list9 = r.new_error_patterns || [];
    const hasRegressionWindow = (data.totals && data.totals.has_regression_window) === true;
    if (!hasRegressionWindow) {
      c9.appendChild(emptyRankingItem("elegí ventana ≥ 24h para detectar errores nuevos"));
    } else if (list9.length === 0) {
      c9.appendChild(emptyRankingItem("ningún error nuevo en la última hora 🎉"));
    } else {
      for (const it of list9) {
        const services = (it.services || []).slice(0, 3).join(", ");
        const more = (it.services || []).length > 3 ? ` +${it.services.length - 3}` : "";
        const sub = `${services}${more} · primero ${fmtRelativeTs(it.first_ts)}`;
        const example = collapseWs(it.example || it.signature);
        c9.appendChild(rankingItem({
          label: example,
          sub,
          count: it.count,
          onClick: () => {
            const target = (it.services && it.services[0]) || null;
            if (target) navigateToService(target);
          },
          ariaLabel: `${example}, ${it.count} ocurrencias nuevas en la última hora.`,
          title: example,
        }));
      }
    }
  }

  // Helpers de formato extra para los rankings v2.
  function formatMs(ms) {
    if (ms == null || isNaN(ms)) return "—";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.round(ms / 60000)}m`;
  }
  function formatBytes(b) {
    if (b == null || isNaN(b)) return "0";
    const n = Number(b);
    if (n < 1024) return `${Math.round(n)} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }
  function fmtDuration(secs) {
    if (secs == null) return "—";
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.round(secs / 60)}m`;
    if (secs < 86400) return `${Math.round(secs / 3600)}h`;
    return `${Math.round(secs / 86400)}d`;
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
      // Wrapper row: el button del service ocupa el ancho, el botón
      // diagnose va a la derecha con stopPropagation. Antes era UN solo
      // <button>, pero queremos dos acciones independientes (seleccionar
      // vs. diagnosticar) y un button anidado dentro de otro es HTML
      // inválido. El row es flex; las status classes (status-X +
      // is-selected) siguen yendo al `.service-item` para que los
      // selectores CSS existentes (border-left por status, hover,
      // selección) sigan funcionando byte-idénticos.
      const row = el("div", {
        class: "service-row",
        dataset: { key: key },
      });
      const item = el("button", {
        type: "button",
        class: "service-item status-" + svc.status +
               (svc.all_empty ? " status-empty" : "") +
               (state.selectedKey === key ? " is-selected" : ""),
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
      row.appendChild(item);
      // Botón "▶" diagnose con IA — sólo si el service tiene errores
      // recientes. No mostramos para services en status=ok porque no hay
      // nada que diagnosticar y agrega ruido visual a la lista.
      if (svc.error_count_recent > 0 && (svc.status === "error" || svc.status === "warn")) {
        const diagBtn = el("button", {
          type: "button",
          class: "diag-trigger service-diag-trigger",
          title: `Diagnosticar errores de ${svc.service} con IA`,
          "aria-label": `Diagnosticar errores de ${svc.service} con IA`,
          onclick: (e) => {
            e.stopPropagation();
            e.preventDefault();
            void diagnoseServiceErrors(svc);
          },
        }, "▶");
        row.appendChild(diagBtn);
      }
      host.appendChild(row);
    }
  }

  /**
   * Click handler del botón "▶" en cada service-item del sidebar.
   *
   * Estrategia: el LLM diagnostica MEJOR el patrón general que casos
   * individuales — si tengo 50 instancias del mismo `database is locked`,
   * agrupar es lo correcto. Por eso traemos las últimas N líneas
   * error/warn del service y se las pasamos al modal como contexto, en
   * vez de un único error aislado.
   *
   * Pasos:
   *   1. Encontrar el archivo del service con más probabilidad de
   *      tener errores (stderr primero, fallback stdout).
   *   2. Fetch /api/logs/file?only_errors=1 con tail=200 — el endpoint
   *      ya filtra a level=warn/error.
   *   3. Quedarnos con las últimas N (10 default) líneas.
   *   4. Abrir el modal con error_text = resumen del grupo +
   *      context_lines = los samples completos.
   */
  async function diagnoseServiceErrors(svc) {
    if (!window.DiagnoseModal) {
      alert("Modal de diagnóstico no cargado. Recargá la página.");
      return;
    }
    // Elegir el archivo: stderr primero, fallback stdout. Si ninguno
    // tiene `ref` (raro), usamos el primero del array.
    const stderrFile = svc.files.find((f) => f.kind === "stderr" && f.ref);
    const stdoutFile = svc.files.find((f) => f.kind === "stdout" && f.ref);
    const file = stderrFile || stdoutFile || svc.files[0];
    if (!file || !file.ref) {
      alert(`No hay archivo de log para ${svc.service}.`);
      return;
    }
    let lines = [];
    let mtime = null;
    try {
      const params = new URLSearchParams({
        name: file.ref,
        tail: "200",
        only_errors: "1",
      });
      const resp = await fetch(`/api/logs/file?${params.toString()}`, { cache: "no-store" });
      if (!resp.ok) {
        alert(`Error fetching logs: HTTP ${resp.status}`);
        return;
      }
      const data = await resp.json();
      lines = Array.isArray(data.lines) ? data.lines : [];
      mtime = data.mtime || null;
    } catch (e) {
      alert(`Error de red: ${e.message || e}`);
      return;
    }
    if (lines.length === 0) {
      alert(`No hay errores recientes en ${svc.service}.`);
      return;
    }
    // Ordenamos: las líneas vienen del más reciente al más viejo (n=1 es
    // la última). Para el LLM tiene más sentido el orden cronológico
    // (viejo → nuevo) así sigue el flujo del log. El último error
    // (más reciente) lo ponemos como `error_text` del modal porque es
    // el que disparó el click.
    const sortedAsc = [...lines].sort((a, b) => (a.n || 0) - (b.n || 0));
    const latest = sortedAsc[sortedAsc.length - 1];
    // Tomar las últimas 10 (incluida la latest); el resto van como
    // contexto. Si hay menos de 10, mandamos todas.
    const N = 10;
    const groupSlice = sortedAsc.slice(-N);
    const samples = groupSlice.map((l) => {
      const tsPart = l.ts ? `[${l.ts.slice(11, 19)}] ` : "";
      const lvlPart = `(${l.level}) `;
      return `${tsPart}${lvlPart}${l.text}`;
    });
    // El prompt del backend espera UN error_text + N context_lines.
    // Construimos un error_text "agregado" que le dice al LLM que esto
    // es un PATRÓN, no un caso aislado, y le damos el más reciente como
    // representativo. Los samples van en context_lines.
    const errorText =
      `[Grupo de ${lines.length} errores recientes en service '${svc.service}']\n\n` +
      `Última ocurrencia (${latest.ts || "sin ts"}):\n` +
      `${latest.text}\n\n` +
      `(Diagnosticá el patrón general del grupo, no solo este caso. ` +
      `Las ${groupSlice.length} muestras más recientes están en el contexto.)`;
    window.DiagnoseModal.open({
      service: svc.service,
      file: `${svc.dir}/${svc.service} (${file.kind}, group of ${lines.length} errors)`,
      line_n: latest.n || 0,
      error_text: errorText,
      context_lines: samples,
      timestamp: latest.ts || mtime || null,
    });
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
      // 2026-04-26: removido el botón "🩺 fix" por línea individual.
      // El feedback del user fue claro: "resolver logs así no sirve, se
      // resuelve por grupo". Una línea aislada NO le da al LLM el patrón
      // — si tengo 50 instancias del mismo `database is locked`, hace
      // falta agruparlas para que el diagnóstico tenga sentido.
      // El botón "▶" ahora vive en cada service-row del sidebar (handler
      // `diagnoseServiceErrors` arriba), que envía las últimas 10 líneas
      // error/warn del service como contexto al LLM.
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
    state.rankingsTimer = setInterval(fetchRankings, RANKINGS_REFRESH_MS);
  }
  function stopTimers() {
    if (state.sidebarTimer) clearInterval(state.sidebarTimer);
    if (state.viewerTimer) clearInterval(state.viewerTimer);
    if (state.rankingsTimer) clearInterval(state.rankingsTimer);
    state.sidebarTimer = null;
    state.viewerTimer = null;
    state.rankingsTimer = null;
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
      await fetchRankings();
    });

    // Selector de ventana del panel de rankings. Persiste la elección
    // en localStorage. Cambiarla refetcha inmediatamente.
    const rankingsSelect = $("rankings-window");
    if (rankingsSelect) {
      // Sync el valor del select con el localStorage al boot.
      const initialWindow = loadRankingsWindow();
      state.rankingsWindow = initialWindow;
      rankingsSelect.value = String(initialWindow);
      rankingsSelect.addEventListener("change", () => {
        const v = parseInt(rankingsSelect.value, 10) || RANKINGS_WINDOW_DEFAULT;
        state.rankingsWindow = v;
        saveRankingsWindow(v);
        fetchRankings();
      });
    }

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

    // ── View toggle (logs vs queue) ─────────────────────────────────
    document.querySelectorAll(".view-toggle-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.dataset.view;
        document.querySelectorAll(".view-toggle-btn").forEach((b) => {
          b.setAttribute("aria-pressed", b === btn ? "true" : "false");
        });
        const isQueue = view === "queue";
        document.querySelector(".layout").hidden = isQueue;
        $("queue-panel").hidden = !isQueue;
        document.getElementById("totals").hidden = isQueue;
        const rk = document.getElementById("rankings");
        if (rk) rk.hidden = isQueue;
        if (isQueue) {
          fetchQueueNow();
          startQueueTimer();
        } else {
          stopQueueTimer();
        }
      });
    });

    // Queue controls.
    $("qc-worker-toggle").addEventListener("click", async () => {
      const btn = $("qc-worker-toggle");
      const wasEnabled = btn.getAttribute("aria-pressed") === "true";
      try {
        const resp = await fetch("/api/logs/queue/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !wasEnabled }),
        });
        const d = await resp.json();
        btn.setAttribute("aria-pressed", d.worker_enabled ? "true" : "false");
        $("qc-worker-label").textContent = d.worker_enabled ? "worker ON" : "worker OFF";
        fetchQueueNow();
      } catch (e) {
        alert(`error toggling worker: ${e.message}`);
      }
    });

    $("qc-scan-now").addEventListener("click", async () => {
      const btn = $("qc-scan-now");
      btn.disabled = true;
      btn.textContent = "↻ escaneando…";
      try {
        const resp = await fetch("/api/logs/queue/scan-now", { method: "POST" });
        const d = await resp.json();
        btn.textContent = `+${d.new_entries} nuevos`;
        setTimeout(() => { btn.textContent = "↻ escanear"; btn.disabled = false; }, 2000);
        fetchQueueNow();
      } catch (e) {
        btn.textContent = `✗ ${e.message}`;
        btn.disabled = false;
      }
    });

    $("qc-process-next").addEventListener("click", async () => {
      const btn = $("qc-process-next");
      btn.disabled = true;
      btn.textContent = "▶ procesando (puede tardar minutos)…";
      try {
        const resp = await fetch("/api/logs/queue/process-next", { method: "POST" });
        const d = await resp.json();
        if (d.status === "no-pending") {
          btn.textContent = "no hay pending";
        } else {
          btn.textContent = `✓ ${d.resolution_status || "done"}`;
        }
        setTimeout(() => {
          btn.textContent = "▶ procesar siguiente";
          btn.disabled = false;
        }, 3500);
        fetchQueueNow();
      } catch (e) {
        btn.textContent = `✗ ${e.message}`;
        btn.disabled = false;
      }
    });
  }

  // ── Queue panel ──────────────────────────────────────────────────────
  let queueTimer = null;
  const QUEUE_REFRESH_MS = 5000;

  function startQueueTimer() {
    stopQueueTimer();
    queueTimer = setInterval(fetchQueueNow, QUEUE_REFRESH_MS);
  }
  function stopQueueTimer() {
    if (queueTimer) { clearInterval(queueTimer); queueTimer = null; }
  }

  async function fetchQueueNow() {
    try {
      const resp = await fetch("/api/logs/queue?limit=100", { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderQueue(data);
      updateQueueBadge(data.counts_by_status || {});
    } catch (e) {
      console.warn("queue fetch failed:", e);
    }
  }

  function renderQueue(data) {
    const counts = data.counts_by_status || {};
    $("qc-pending").textContent = counts.pending || 0;
    $("qc-processing").textContent = counts.processing || 0;
    $("qc-resolved").textContent = counts.resolved || 0;
    $("qc-needs-human").textContent = counts["needs-human"] || 0;
    $("qc-failed").textContent = counts.failed || 0;

    // Worker toggle state.
    const wt = $("qc-worker-toggle");
    wt.setAttribute("aria-pressed", data.worker_enabled ? "true" : "false");
    $("qc-worker-label").textContent = data.worker_enabled ? "worker ON" : "worker OFF";

    // Rate limit indicator.
    const rl = data.worker_rate_limit || {};
    const rlEl = $("queue-rate-limit");
    if (!rl.can_invoke_now) {
      rlEl.textContent = `⚠ ${rl.reason}`;
      rlEl.classList.add("qrl-warn");
    } else {
      rlEl.textContent = `rate: ${rl.current_hour_count || 0}/${rl.hourly_cap || 5} invocaciones de Devin en la última hora`;
      rlEl.classList.remove("qrl-warn");
    }

    // Tabla.
    const tbody = $("queue-tbody");
    const entries = data.entries || [];
    if (entries.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="queue-empty">la queue está vacía — clickeá "↻ escanear" para ver si hay errores nuevos</td></tr>`;
      return;
    }
    const rows = entries.map((e) => {
      const lastSeen = e.last_seen_at ? e.last_seen_at.slice(5, 16).replace("T", " ") : "—";
      const resolution = e.resolution_status
        ? `<span class="queue-status ${e.resolution_status}">${e.resolution_status}</span>`
        : "—";
      return `<tr data-id="${e.id}">
        <td style="color:var(--text-faint);font-variant-numeric:tabular-nums">${e.id}</td>
        <td><span class="queue-status ${e.status}">${e.status}</span></td>
        <td>${escapeHtml(e.service)}</td>
        <td class="queue-error-text" title="${escapeHtml(e.error_text)}">${escapeHtml(e.error_text)}</td>
        <td class="queue-occ">${e.occurrence_count}</td>
        <td class="queue-age">${lastSeen}</td>
        <td>${resolution}</td>
      </tr>`;
    }).join("");
    tbody.innerHTML = rows;
    // Click row → abrir detalle.
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => openQueueDetail(parseInt(tr.dataset.id, 10)));
    });
  }

  async function openQueueDetail(id) {
    try {
      const resp = await fetch(`/api/logs/queue/${id}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d = await resp.json();
      const lines = [
        `service: ${d.service}`,
        `file: ${d.file_ref}`,
        `signature: ${d.error_signature}`,
        `status: ${d.status}`,
        `occurrences: ${d.occurrence_count}`,
        `first seen: ${d.first_seen_at}`,
        `last seen: ${d.last_seen_at}`,
        `attempts: ${d.attempts}`,
      ];
      if (d.completed_at) lines.push(`completed: ${d.completed_at} (${d.duration_s}s)`);
      if (d.resolution_status) lines.push(`resolution: ${d.resolution_status} — ${d.resolution_reason || ""}`);
      lines.push("");
      lines.push("--- error text ---");
      lines.push(d.error_text);
      if (d.devin_output) {
        lines.push("");
        lines.push("--- devin output ---");
        lines.push(d.devin_output);
      }
      // Simple dialog — reusamos el window.alert porque el modal serio
      // está comprometido con el diagnose flow.
      alert(lines.join("\n"));
    } catch (e) {
      alert(`error loading detail: ${e.message}`);
    }
  }

  function updateQueueBadge(counts) {
    const badge = $("queue-badge");
    const pending = (counts.pending || 0) + (counts.processing || 0);
    if (pending > 0) {
      badge.hidden = false;
      badge.textContent = String(pending);
    } else {
      badge.hidden = true;
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────
  wireUp();
  // Sidebar primero (el resto de la página depende de tener `state.services`),
  // rankings en paralelo. Si rankings termina antes que el primer sidebar
  // fetch, el render queda visible mientras la sidebar termina de cargar.
  fetchServices().then(() => startTimers());
  fetchRankings();
  // Poll queue in background para el badge incluso si estás en vista logs.
  setInterval(() => {
    if (!document.hidden) {
      fetch("/api/logs/queue?limit=1", { cache: "no-store" })
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) updateQueueBadge(d.counts_by_status || {}); })
        .catch(() => {});
    }
  }, 15000);
})();
