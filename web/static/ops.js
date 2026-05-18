(function () {
  "use strict";

  var AUTO_REFRESH_MS = 30000;
  var endpoints = {
    mission: "/api/health/mission-control",
    missionAction: "/api/health/mission-control/action",
    anticipate: "/api/anticipate/inbox?only_actionable=true&limit=12&days=7",
    memory: "/api/memory/unified?limit=12",
    negotiations: "/api/negotiations?limit=30",
  };

  var actionLabels = {
    open_anticipate_inbox: "Abrir inbox anticipate",
    run_feedback_harvest: "Correr feedback harvest",
    inspect_silent_errors: "Inspeccionar errores silenciosos",
    check_telemetry_db: "Revisar telemetry DB",
  };

  var SUN_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>';
  var MOON_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  var state = {
    live: true,
    timer: null,
    loading: {},
    data: {
      mission: null,
      anticipate: null,
      memory: null,
      negotiations: null,
    },
    errors: {
      mission: null,
      anticipate: null,
      memory: null,
      negotiations: null,
    },
    actionMessages: {},
  };

  var dom = {
    alerts: document.getElementById("ops-alerts"),
    updated: document.getElementById("ops-updated"),
    refresh: document.getElementById("ops-refresh"),
    live: document.getElementById("ops-live"),
    liveLabel: document.getElementById("ops-live-label"),
    theme: document.getElementById("ops-theme"),
    themeIcon: document.getElementById("ops-theme-icon"),
    missionBody: document.getElementById("mission-body"),
    missionMeta: document.getElementById("mission-meta"),
    anticipateBody: document.getElementById("anticipate-body"),
    anticipateMeta: document.getElementById("anticipate-meta"),
    memoryBody: document.getElementById("memory-body"),
    memoryMeta: document.getElementById("memory-meta"),
    negotiationsBody: document.getElementById("negotiations-body"),
    negotiationsMeta: document.getElementById("negotiations-meta"),
  };

  function node(tag, className, text) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = String(text);
    return el;
  }

  function clear(el) {
    if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function text(value, fallback) {
    if (value === null || value === undefined || value === "") return fallback || "--";
    return String(value);
  }

  function num(value) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return n.toLocaleString("es-AR");
  }

  function pct(value) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return Math.round(n * 100) + "%";
  }

  function fmtTs(value) {
    if (!value) return "--";
    var d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString("es-AR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function nowLabel() {
    return new Date().toLocaleTimeString("es-AR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function statusKind(status) {
    var s = String(status || "").toLowerCase();
    if (s === "ok" || s === "healthy" || s === "sent" || s === "active") return "ok";
    if (s === "down" || s === "error" || s === "failed") return "down";
    if (s === "degraded" || s === "warn" || s === "partial" || s === "blocked") return "warn";
    return "idle";
  }

  function badge(label, status) {
    return node("span", "badge " + statusKind(status), label);
  }

  function setTile(id, value, meta, status) {
    var tile = document.getElementById(id);
    if (!tile) return;
    var valueEl = tile.querySelector(".tile-value");
    var metaEl = tile.querySelector(".tile-meta");
    if (valueEl) valueEl.textContent = text(value);
    if (metaEl) metaEl.textContent = text(meta);
    tile.dataset.status = statusKind(status);
  }

  function errorMessage(err) {
    if (!err) return "";
    var msg = err.detail || err.message || String(err);
    if (err.status) return "HTTP " + err.status + ": " + msg;
    return msg;
  }

  async function fetchJson(url, options) {
    var resp = await fetch(url, Object.assign({ cache: "no-store" }, options || {}));
    var contentType = resp.headers.get("content-type") || "";
    var payload = null;
    if (contentType.indexOf("application/json") !== -1) {
      payload = await resp.json().catch(function () { return null; });
    } else {
      var raw = await resp.text().catch(function () { return ""; });
      payload = raw ? { detail: raw } : null;
    }
    if (!resp.ok) {
      var err = new Error((payload && (payload.detail || payload.error)) || resp.statusText || "request failed");
      err.status = resp.status;
      err.detail = payload && (payload.detail || payload.error);
      err.payload = payload;
      throw err;
    }
    return payload || {};
  }

  async function fetchSection(name) {
    state.loading[name] = true;
    render();
    try {
      state.data[name] = await fetchJson(endpoints[name]);
      state.errors[name] = null;
    } catch (err) {
      state.errors[name] = err;
    } finally {
      state.loading[name] = false;
      render();
    }
  }

  async function refreshAll() {
    if (dom.refresh) dom.refresh.disabled = true;
    await Promise.all([
      fetchSection("mission"),
      fetchSection("anticipate"),
      fetchSection("memory"),
      fetchSection("negotiations"),
    ]);
    if (dom.updated) dom.updated.textContent = "actualizado " + nowLabel();
    if (dom.refresh) dom.refresh.disabled = false;
    scheduleNext();
  }

  function scheduleNext() {
    if (state.timer) window.clearTimeout(state.timer);
    state.timer = null;
    if (!state.live) return;
    state.timer = window.setTimeout(refreshAll, AUTO_REFRESH_MS);
  }

  function showAlert(message, kind) {
    if (!dom.alerts) return;
    var item = node("div", "alert " + (kind || "warn"));
    item.appendChild(node("span", "", message));
    var close = node("button", "btn", "cerrar");
    close.type = "button";
    close.addEventListener("click", function () { item.remove(); });
    item.appendChild(close);
    dom.alerts.prepend(item);
    while (dom.alerts.children.length > 4) {
      dom.alerts.removeChild(dom.alerts.lastChild);
    }
  }

  function render() {
    renderSummary();
    renderMission();
    renderAnticipate();
    renderMemory();
    renderNegotiations();
  }

  function renderSummary() {
    var mission = state.data.mission;
    var anticipate = state.data.anticipate;
    var memory = state.data.memory;
    var negotiations = state.data.negotiations;

    setTile(
      "tile-mission",
      state.errors.mission ? "error" : text(mission && mission.overall),
      mission ? "ventana " + text(mission.window_days, "7") + "d" : errorMessage(state.errors.mission) || "--",
      state.errors.mission ? "down" : mission && mission.overall
    );

    var actions = Array.isArray(mission && mission.actions) ? mission.actions.length : null;
    setTile(
      "tile-actions",
      actions == null ? "--" : num(actions),
      actions ? "pendientes" : "sin acciones",
      actions ? "warn" : "ok"
    );

    var antSummary = anticipate && anticipate.summary;
    setTile(
      "tile-anticipate",
      state.errors.anticipate ? "error" : num(antSummary && antSummary.actionable),
      antSummary ? num(antSummary.returned) + " visibles" : errorMessage(state.errors.anticipate) || "--",
      state.errors.anticipate ? "down" : (Number(antSummary && antSummary.actionable) > 0 ? "warn" : "ok")
    );

    var memCounts = memory && memory.summary && memory.summary.counts;
    var memTotal = memCounts ? Number(memCounts.memo || 0) + Number(memCounts.conversations || 0) + Number(memCounts.runbooks || 0) : null;
    setTile(
      "tile-memory",
      state.errors.memory ? "error" : (memTotal == null ? "--" : num(memTotal)),
      memCounts ? "memo " + num(memCounts.memo) + " - conv " + num(memCounts.conversations) : errorMessage(state.errors.memory) || "--",
      state.errors.memory ? "down" : (memory && memory.summary && memory.summary.status)
    );

    var negItems = Array.isArray(negotiations && negotiations.items) ? negotiations.items : null;
    setTile(
      "tile-negotiations",
      state.errors.negotiations ? "error" : (negItems ? num(negItems.length) : "--"),
      negItems ? "ultimas cargadas" : errorMessage(state.errors.negotiations) || "--",
      state.errors.negotiations ? "down" : "idle"
    );
  }

  function renderError(body, err) {
    clear(body);
    body.appendChild(node("div", "error-box", errorMessage(err)));
  }

  function renderLoading(body) {
    clear(body);
    body.appendChild(node("div", "loading", "cargando..."));
  }

  function renderMission() {
    if (state.loading.mission && !state.data.mission) return renderLoading(dom.missionBody);
    if (state.errors.mission) return renderError(dom.missionBody, state.errors.mission);

    var payload = state.data.mission;
    if (!payload) return;
    clear(dom.missionBody);
    if (dom.missionMeta) {
      dom.missionMeta.textContent = "generado " + fmtTs(payload.generated_at) + " - " + text(payload.window_days, "7") + "d";
    }

    var top = node("div", "kpi-line");
    top.appendChild(badge("overall " + text(payload.overall), payload.overall));
    top.appendChild(node("span", "subtle", "subsystems " + num((payload.subsystems || []).length)));
    dom.missionBody.appendChild(top);

    dom.missionBody.appendChild(node("div", "section-label", "acciones"));
    var actions = Array.isArray(payload.actions) ? payload.actions : [];
    if (!actions.length) {
      dom.missionBody.appendChild(node("div", "empty", "sin acciones pendientes"));
    } else {
      actions.forEach(function (actionId) {
        var row = node("div", "action-row");
        var main = node("div");
        main.appendChild(node("div", "row-title", actionLabels[actionId] || actionId));
        var msg = state.actionMessages[actionId];
        main.appendChild(node("div", "row-meta", msg || actionId));
        var btn = node("button", "btn btn-primary", "Ejecutar");
        btn.type = "button";
        btn.dataset.missionAction = actionId;
        row.appendChild(main);
        row.appendChild(btn);
        dom.missionBody.appendChild(row);
      });
    }

    dom.missionBody.appendChild(node("div", "section-label", "subsystems"));
    var list = node("div", "compact-list");
    (payload.subsystems || []).forEach(function (sub) {
      var item = node("div", "compact-item");
      item.appendChild(node("span", "status-dot " + statusKind(sub.status)));
      var content = node("div");
      var title = node("div", "row-title", text(sub.label || sub.id) + " - " + text(sub.status));
      title.classList.add("status-" + statusKind(sub.status));
      content.appendChild(title);
      var issues = Array.isArray(sub.issues) ? sub.issues.filter(Boolean) : [];
      content.appendChild(node("div", "row-meta", issues.length ? issues.slice(0, 2).join(" - ") : "ok"));
      var details = compactDetails(sub.details);
      if (details) content.appendChild(node("div", "json-preview", details));
      item.appendChild(content);
      list.appendChild(item);
    });
    dom.missionBody.appendChild(list);
  }

  function compactDetails(details) {
    if (!details || typeof details !== "object") return "";
    var parts = [];
    if (details.source_status) parts.push("source=" + details.source_status);
    if (details.total) parts.push("total=" + details.total);
    if (details.count) parts.push("count=" + details.count);
    if (details.sent_rate != null) parts.push("sent=" + pct(details.sent_rate));
    if (details.cache_hit_rate != null) parts.push("cache=" + pct(details.cache_hit_rate));
    if (details.error_rollup && details.error_rollup.total_errors != null) {
      parts.push("errors=" + details.error_rollup.total_errors);
    }
    if (details.db_stats && details.db_stats.size_bytes != null) {
      parts.push("db=" + bytes(details.db_stats.size_bytes));
    }
    return parts.slice(0, 5).join(" - ");
  }

  function renderAnticipate() {
    if (state.loading.anticipate && !state.data.anticipate) return renderLoading(dom.anticipateBody);
    if (state.errors.anticipate) return renderError(dom.anticipateBody, state.errors.anticipate);

    var payload = state.data.anticipate;
    if (!payload) return;
    clear(dom.anticipateBody);
    var summary = payload.summary || {};
    if (dom.anticipateMeta) {
      dom.anticipateMeta.textContent = num(summary.actionable) + " actionable - " + num(summary.hidden_by_only_actionable) + " ocultos";
    }
    var items = Array.isArray(payload.items) ? payload.items : [];
    if (!items.length) {
      dom.anticipateBody.appendChild(node("div", "empty", "sin candidatos accionables"));
      return;
    }
    items.forEach(function (item) {
      var row = node("div", "data-row");
      var main = node("div");
      main.appendChild(node("div", "row-title", text(item.kind) + " - score " + text(item.score)));
      main.appendChild(node("div", "row-text", text(item.message_preview || item.reason)));
      main.appendChild(node("div", "row-meta", fmtTs(item.ts) + " - " + text(item.dedup_key)));
      row.appendChild(main);
      row.appendChild(badge(text(item.status), item.status));
      dom.anticipateBody.appendChild(row);
    });
  }

  function renderMemory() {
    if (state.loading.memory && !state.data.memory) return renderLoading(dom.memoryBody);
    if (state.errors.memory) return renderError(dom.memoryBody, state.errors.memory);

    var payload = state.data.memory;
    if (!payload) return;
    clear(dom.memoryBody);
    var summary = payload.summary || {};
    var counts = summary.counts || {};
    if (dom.memoryMeta) {
      dom.memoryMeta.textContent = text(summary.status) + " - generado " + fmtTs(payload.generated_at);
    }
    var kpis = node("div", "kpi-line");
    kpis.appendChild(badge("memo " + num(counts.memo), summary.status));
    kpis.appendChild(badge("conv " + num(counts.conversations), summary.status));
    kpis.appendChild(badge("runbooks " + num(counts.runbooks), summary.status));
    dom.memoryBody.appendChild(kpis);

    var sections = payload.sections || {};
    renderMemorySection("memo", sections.memo);
    renderMemorySection("conversations", sections.conversations);
    renderMemorySection("runbooks", sections.runbooks);
  }

  function renderMemorySection(name, section) {
    dom.memoryBody.appendChild(node("div", "section-label", name));
    if (!section) {
      dom.memoryBody.appendChild(node("div", "empty", "sin datos"));
      return;
    }
    if (section.ok === false) {
      dom.memoryBody.appendChild(node("div", "error-box", section.error || "section unavailable"));
      return;
    }
    var latest = Array.isArray(section.latest) ? section.latest : [];
    if (!latest.length) {
      dom.memoryBody.appendChild(node("div", "empty", "sin items"));
      return;
    }
    var list = node("div", "compact-list");
    latest.slice(0, 6).forEach(function (item) {
      var row = node("div", "compact-item");
      row.appendChild(node("span", "status-dot idle"));
      var content = node("div");
      content.appendChild(node("div", "row-title", itemTitle(item)));
      content.appendChild(node("div", "row-meta", itemMeta(item)));
      list.appendChild(row);
      row.appendChild(content);
    });
    dom.memoryBody.appendChild(list);
  }

  function renderNegotiations() {
    if (state.loading.negotiations && !state.data.negotiations) return renderLoading(dom.negotiationsBody);
    if (state.errors.negotiations) return renderError(dom.negotiationsBody, state.errors.negotiations);

    var payload = state.data.negotiations;
    if (!payload) return;
    clear(dom.negotiationsBody);
    var items = Array.isArray(payload.items) ? payload.items : [];
    if (dom.negotiationsMeta) {
      dom.negotiationsMeta.textContent = num(items.length) + " registros";
    }
    if (!items.length) {
      dom.negotiationsBody.appendChild(node("div", "empty", "sin negociaciones"));
      return;
    }
    var wrap = node("div", "data-table-wrap");
    var table = node("table", "data-table");
    var thead = node("thead");
    var hr = node("tr");
    ["id", "status", "target", "intencion", "updated"].forEach(function (label) {
      hr.appendChild(node("th", "", label));
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = node("tbody");
    items.slice(0, 30).forEach(function (item) {
      var tr = node("tr");
      tr.appendChild(node("td", "mono primary", text(item.id)));
      var statusCell = node("td");
      statusCell.appendChild(badge(text(item.status), item.status));
      tr.appendChild(statusCell);
      tr.appendChild(node("td", "primary", text(item.target_name || item.target_jid)));
      tr.appendChild(node("td", "", text(item.user_intent || item.intent || item.closure_summary)));
      tr.appendChild(node("td", "mono", fmtTs(item.updated_at || item.created_at)));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    dom.negotiationsBody.appendChild(wrap);
  }

  function itemTitle(item) {
    if (!item || typeof item !== "object") return "--";
    return text(
      item.title ||
      item.key ||
      item.name ||
      item.relative_path ||
      item.session_id ||
      item.id ||
      item.path
    );
  }

  function itemMeta(item) {
    if (!item || typeof item !== "object") return "--";
    var bits = [];
    if (item.updated_at || item.ts || item.created_at) bits.push(fmtTs(item.updated_at || item.ts || item.created_at));
    if (item.score != null) bits.push("score " + text(item.score));
    if (item.bytes != null) bits.push(bytes(item.bytes));
    if (item.kind) bits.push(item.kind);
    if (item.text || item.content || item.summary) bits.push(text(item.text || item.content || item.summary).slice(0, 160));
    return bits.length ? bits.join(" - ") : JSON.stringify(item).slice(0, 180);
  }

  function bytes(value) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "--";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  }

  async function postMissionAction(actionId, button) {
    if (!actionId) return;
    if (button) button.disabled = true;
    state.actionMessages[actionId] = "ejecutando...";
    renderMission();
    try {
      var headers = new Headers({ "Content-Type": "application/json" });
      if (window.__ragAdminAuth && typeof window.__ragAdminAuth.loadToken === "function") {
        var token = await window.__ragAdminAuth.loadToken();
        if (token) headers.set("Authorization", "Bearer " + token);
      }
      var payload = await fetchJson(endpoints.missionAction, {
        method: "POST",
        headers: headers,
        body: JSON.stringify({ action: actionId, action_id: actionId }),
      });
      state.actionMessages[actionId] = payload.detail || payload.message || "ok";
      showAlert(actionLabels[actionId] || actionId + ": ok", "ok");
      await fetchSection("mission");
    } catch (err) {
      if (err.status === 404) {
        state.actionMessages[actionId] = "endpoint no disponible";
        showAlert("Mission action no disponible: HTTP 404", "warn");
      } else {
        state.actionMessages[actionId] = errorMessage(err);
        showAlert((actionLabels[actionId] || actionId) + ": " + errorMessage(err), "error");
      }
      renderMission();
    } finally {
      if (button) button.disabled = false;
    }
  }

  function currentTheme() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function setTheme(next) {
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("rag-theme", next); } catch (_) {}
    if (dom.themeIcon) dom.themeIcon.innerHTML = next === "light" ? MOON_ICON : SUN_ICON;
  }

  function wireEvents() {
    if (dom.refresh) {
      dom.refresh.addEventListener("click", refreshAll);
    }
    if (dom.live) {
      dom.live.addEventListener("click", function () {
        state.live = !state.live;
        dom.live.setAttribute("aria-pressed", String(state.live));
        if (dom.liveLabel) dom.liveLabel.textContent = state.live ? "auto" : "pausa";
        scheduleNext();
      });
    }
    if (dom.theme) {
      setTheme(currentTheme());
      dom.theme.addEventListener("click", function () {
        setTheme(currentTheme() === "light" ? "dark" : "light");
      });
    }
    document.addEventListener("click", function (ev) {
      var refreshBtn = ev.target.closest("[data-refresh]");
      if (refreshBtn) {
        fetchSection(refreshBtn.dataset.refresh).then(scheduleNext);
        return;
      }
      var actionBtn = ev.target.closest("[data-mission-action]");
      if (actionBtn) {
        postMissionAction(actionBtn.dataset.missionAction, actionBtn);
      }
    });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        if (state.timer) window.clearTimeout(state.timer);
        state.timer = null;
      } else {
        refreshAll();
      }
    });
  }

  wireEvents();
  refreshAll();
})();
