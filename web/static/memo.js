// /memo — dashboard read-only del MCP `memo` con scoring + dupes + search.
//
// Contesta la pregunta: "¿sirve o no esta memoria?" via:
//   1. Quality score 0-100 con breakdown (actionable/tags/size/fresh/unique).
//   2. Body preview inline para skim sin clickear.
//   3. Dupe-pairs section con candidatos a fusionar/borrar.
//   4. FTS search box en el topbar.
//   5. Filter pills: tipos + "solo dupes".
//   6. Sparkline ASCII de saves últimos 30d.
//
// Endpoints consumidos:
//   GET /api/memo                    snapshot completo
//   GET /api/memo/note?id=…          detalle 1 memoria + vecinos
//   GET /api/memo/search?q=…         FTS5 search

const $ = (sel) => document.querySelector(sel);

const STATE = {
  type: null,
  limit: 80,
  selectedId: null,
  onlyDupes: false,
  searchQuery: "",
  snapshot: null,
  // Time-machine: si tmDate=null, vista "hoy" (comportamiento normal).
  // Si tmDate=YYYY-MM-DD, vista historical: tabla viene del snapshot,
  // ocultamos verdict/dupes/recall/dead (no aplican a corpus pasado).
  tmDate: null,
  tmSnapshot: null,
  tmDiff: null,
};

const TYPE_ORDER = ["bug", "decision", "preference", "fact", "note"];

function escapeHTML(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function scoreClass(score) {
  if (score >= 80) return "score-fill-good";
  if (score >= 60) return "score-fill-mid";
  return "score-fill-bad";
}
function scoreLabel(score) {
  if (score >= 80) return "good";
  if (score >= 60) return "warn";
  return "bad";
}

function formatSize(bytes) {
  if (!bytes) return "0B";
  if (bytes < 1024) return bytes + "B";
  return (bytes / 1024).toFixed(1) + "K";
}

// ── Toast + inline-confirm helpers (reemplazan alert/confirm de chrome) ──

const TOAST_ICONS = { success: "✓", error: "✕", warn: "⚠", info: "ℹ" };

function toast({ title, detail = "", kind = "info", ttl = 5000 }) {
  const stack = $("#toast-stack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[kind] || "ℹ"}</span>
    <div class="toast-body">
      <div class="title">${escapeHTML(title)}</div>
      ${detail ? `<div class="detail">${escapeHTML(detail)}</div>` : ""}
    </div>
    <button class="toast-close" aria-label="cerrar">×</button>
  `;
  stack.appendChild(el);
  const dismiss = () => {
    if (el.parentNode) {
      el.classList.add("removing");
      setTimeout(() => el.remove(), 220);
    }
  };
  el.querySelector(".toast-close").addEventListener("click", dismiss);
  if (ttl > 0) setTimeout(dismiss, ttl);
  return { dismiss };
}

function compactErrors(errors, maxChars = 900) {
  const items = (errors || []).slice(0, 3).map((err) => {
    const id = err.id || err.pair?.a || "";
    const suffix = err.pair?.b ? ` / ${err.pair.b}` : "";
    const label = id ? `${id}${suffix}: ` : "";
    return `${label}${err.error || err.warning || "error desconocido"}`;
  });
  const text = items.join("\n\n");
  return text.length > maxChars ? `${text.slice(0, maxChars)}…` : text;
}

// Confirm inline en lugar del modal del navegador. Reemplaza el botón
// `btn` por un span con "¿Seguro? [sí] [no]". Resuelve true/false.
function inlineConfirm(btn, question) {
  return new Promise((resolve) => {
    const orig = btn.innerHTML;
    const wasDisabled = btn.disabled;
    const wrap = document.createElement("span");
    wrap.className = "inline-confirm";
    wrap.innerHTML = `
      <span>${escapeHTML(question)}</span>
      <button class="yes">sí, hacelo</button>
      <button class="no">cancelar</button>
    `;
    btn.replaceWith(wrap);
    const restore = (val) => {
      const newBtn = btn;
      newBtn.innerHTML = orig;
      newBtn.disabled = wasDisabled;
      wrap.replaceWith(newBtn);
      resolve(val);
    };
    wrap.querySelector(".yes").addEventListener("click", () => restore(true));
    wrap.querySelector(".no").addEventListener("click", () => restore(false));
  });
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Tiny inline SVG sparkline (sin dependencias). 31 puntos = últimos 30d
//    + hoy. Curva linear, área llena, color cyan.
function renderSparkline(data) {
  if (!data || data.length === 0) return "";
  const W = 160;
  const H = 36;
  const counts = data.map((d) => d.count);
  const max = Math.max(...counts, 1);
  const step = W / (counts.length - 1 || 1);
  const points = counts
    .map((c, i) => `${(i * step).toFixed(1)},${(H - (c / max) * (H - 2) - 1).toFixed(1)}`)
    .join(" ");
  const area = `M0,${H} L${points} L${W},${H} Z`;
  return `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%; height:36px;">
      <path d="${area}" fill="rgba(121,192,255,0.18)" />
      <polyline points="${points}" fill="none" stroke="var(--cyan)" stroke-width="1.4" />
    </svg>
  `;
}

function renderVerdict(v) {
  if (!v || !v.criteria || v.criteria.length === 0) {
    $("#verdict-banner").innerHTML = "";
    return;
  }
  const outcome = v.outcome || "unknown";
  const pillLabel = outcome === "yes" ? "Vale la pena" : outcome === "cleanup" ? "Con cleanup" : outcome === "no" ? "Evaluar alternativa" : "—";

  const criteriaHTML = v.criteria
    .map(
      (c) => `
      <div class="verdict-criterion ${escapeHTML(c.status)}">
        <div class="label">${escapeHTML(c.label)}</div>
        <div class="value">${escapeHTML(c.value)}</div>
        <div class="detail">${escapeHTML(c.detail)}</div>
        <div class="threshold">${escapeHTML(c.threshold)}</div>
      </div>`
    )
    .join("");

  // Mostrar botón cuando el summary menciona cleanup (outcomes "yes" + "cleanup")
  const showCleanupBtn = outcome === "cleanup" || outcome === "yes";
  const cleanupBtnHTML = showCleanupBtn ? `
    <div class="verdict-actions">
      <button id="cleanup-btn" class="cleanup-btn">
        <span class="btn-icon">🧹</span>
        <span class="btn-label">Hacer cleanup ahora</span>
      </button>
      <span class="cleanup-hint">Borra dead memorias (score &lt; 40 + sin recall + creadas &gt; 30d)</span>
    </div>` : "";

  $("#verdict-banner").className = `verdict-banner ${outcome}`;
  $("#verdict-banner").innerHTML = `
    <div class="verdict-headline">
      <span class="verdict-pill ${outcome}">${escapeHTML(pillLabel)}</span>
      <span class="verdict-summary">${escapeHTML(v.summary)}</span>
    </div>
    <div class="verdict-criteria">${criteriaHTML}</div>
    ${cleanupBtnHTML}
  `;

  if (showCleanupBtn) {
    $("#cleanup-btn").addEventListener("click", runCleanup);
  }
}

async function runCleanup() {
  const btn = $("#cleanup-btn");
  if (!btn || btn.disabled) return;
  // Inline confirm reemplaza el btn por "¿sí? / cancelar" + lo restaura
  // al elegir. Event listener original sobre btn persiste porque
  // replaceWith mueve el nodo sin clonarlo.
  const ok = await inlineConfirm(btn, "Borrar dead memories — irreversible.");
  if (!ok) {
    toast({ title: "Cleanup cancelado", kind: "info", ttl: 2500 });
    return;
  }
  btn.disabled = true;
  btn.classList.add("running");
  const label = btn.querySelector(".btn-label");
  const icon = btn.querySelector(".btn-icon");
  const t0 = Date.now();
  let dotTick = 0;
  const dotInterval = setInterval(() => {
    dotTick = (dotTick + 1) % 4;
    const dots = ".".repeat(dotTick);
    const secs = Math.floor((Date.now() - t0) / 1000);
    label.textContent = `Limpiando${dots} ${secs}s`;
  }, 400);
  icon.textContent = "⏳";

  try {
    const r = await fetch("/api/memo/cleanup", { method: "POST" });
    const j = await r.json();
    clearInterval(dotInterval);
    if (!j.ok) {
      icon.textContent = "❌";
      label.textContent = j.error || "error";
      btn.classList.remove("running");
      btn.classList.add("error");
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove("error");
        icon.textContent = "🧹";
        label.textContent = "Hacer cleanup ahora";
      }, 4000);
      return;
    }
    const dead = j.results.dead || {};
    const dupes = j.results.dupes || {};
    icon.textContent = "✓";
    label.textContent = `Listo · ${dead.deleted || 0} dead borradas, ${dead.preserved || 0} preservadas`;
    btn.classList.remove("running");
    btn.classList.add("done");
    if (dupes.detected) {
      $("#cleanup-btn").insertAdjacentHTML("afterend",
        `<span class="cleanup-extra"> · ${dupes.detected} near-dupes detectados (revisar manual)</span>`);
    }
    // Auto-refresh after 1.5s
    setTimeout(() => {
      refresh();
      loadV06Features();
    }, 1500);
  } catch (e) {
    clearInterval(dotInterval);
    icon.textContent = "❌";
    label.textContent = `error: ${e.message}`;
    btn.classList.add("error");
  }
}

function renderTopRecalled(usage) {
  const list = usage.top_recalled || [];
  $("#top-recalled-meta").textContent = `${usage.total_recall_events} eventos · ${usage.recalled_count} de ${usage.total} memorias`;
  if (list.length === 0) {
    $("#top-recalled-list").innerHTML = `<div class="loading">Sin recalls. Hook UserPromptSubmit no está corriendo o memo no matchea nada.</div>`;
    return;
  }
  $("#top-recalled-list").innerHTML = list
    .map(
      (m) => `
      <div class="recall-row" data-id="${escapeHTML(m.id)}">
        <span class="count-badge">${m.count}×</span>
        <div style="flex: 1; min-width: 0;">
          <div class="title">${escapeHTML(m.title)}</div>
          <div class="meta-line">
            <span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span>
            · max score ${m.max_score}
            · último recall ${escapeHTML(m.last_recalled_ago)} atrás
          </div>
          <div class="meta-line">${memoPathLinkHTML(m, "inline-path")}</div>
        </div>
      </div>`
    )
    .join("");
  for (const r of document.querySelectorAll("#top-recalled-list .recall-row")) {
    r.addEventListener("click", () => selectMemo(r.dataset.id));
  }
}

const DEAD_SELECTION = new Set();

function renderDeadList(usage) {
  const list = usage.dead_memorias || [];
  $("#dead-meta").textContent = `${usage.dead_count} memorias`;
  if (list.length === 0) {
    $("#dead-list").innerHTML = `<div class="loading">Sin dead memorias. Todo lo que tiene &gt; 2d se está usando. 👍</div>`;
    return;
  }

  // Filter selection to ids still present
  const presentIds = new Set(list.map((m) => m.id));
  for (const id of [...DEAD_SELECTION]) {
    if (!presentIds.has(id)) DEAD_SELECTION.delete(id);
  }

  const header = `
    <div class="dead-toolbar">
      <label class="dead-select-all">
        <input type="checkbox" id="dead-select-all-cb">
        <span>seleccionar todo</span>
      </label>
      <span class="dead-count" id="dead-selected-count">0 seleccionadas</span>
      <span style="flex:1"></span>
      <button class="dead-btn dead-btn-bulk" id="dead-delete-bulk" disabled>
        🗑 Borrar seleccionadas
      </button>
      <button class="dead-btn dead-btn-all" id="dead-delete-all">
        💥 Borrar todas (${list.length})
      </button>
    </div>
  `;
  const rows = list.map((m) => `
    <div class="dead-row" data-id="${escapeHTML(m.id)}">
      <label class="dead-check">
        <input type="checkbox" class="dead-cb" data-id="${escapeHTML(m.id)}" ${DEAD_SELECTION.has(m.id) ? 'checked' : ''}>
      </label>
      <span class="count-badge" style="color: var(--red);">0×</span>
      <div class="dead-body" data-id="${escapeHTML(m.id)}">
        <div class="title">${escapeHTML(m.title)}</div>
        <div class="meta-line">
          <span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span>
          · creado ${escapeHTML(m.age)} atrás · nunca matcheó un prompt
        </div>
        <div class="meta-line">${memoPathLinkHTML(m, "inline-path")}</div>
      </div>
      <button class="dead-btn-row" data-id="${escapeHTML(m.id)}" title="Borrar esta memoria">🗑</button>
    </div>
  `).join("");

  $("#dead-list").innerHTML = header + rows;
  attachDeadHandlers(list);
  updateDeadSelectionUI();
}

function updateDeadSelectionUI() {
  const n = DEAD_SELECTION.size;
  const countEl = $("#dead-selected-count");
  const bulkBtn = $("#dead-delete-bulk");
  const allCb = $("#dead-select-all-cb");
  if (countEl) countEl.textContent = `${n} seleccionada${n === 1 ? '' : 's'}`;
  if (bulkBtn) bulkBtn.disabled = n === 0;
  if (allCb) {
    const totalCbs = document.querySelectorAll(".dead-cb").length;
    allCb.checked = n > 0 && n === totalCbs;
    allCb.indeterminate = n > 0 && n < totalCbs;
  }
}

function attachDeadHandlers(list) {
  // Click body → ver detalle
  for (const el of document.querySelectorAll("#dead-list .dead-body")) {
    el.addEventListener("click", () => selectMemo(el.dataset.id));
  }
  // Checkbox individual
  for (const cb of document.querySelectorAll("#dead-list .dead-cb")) {
    cb.addEventListener("change", () => {
      if (cb.checked) DEAD_SELECTION.add(cb.dataset.id);
      else DEAD_SELECTION.delete(cb.dataset.id);
      updateDeadSelectionUI();
    });
  }
  // Select-all
  const allCb = $("#dead-select-all-cb");
  if (allCb) {
    allCb.addEventListener("change", () => {
      const checked = allCb.checked;
      DEAD_SELECTION.clear();
      for (const cb of document.querySelectorAll("#dead-list .dead-cb")) {
        cb.checked = checked;
        if (checked) DEAD_SELECTION.add(cb.dataset.id);
      }
      updateDeadSelectionUI();
    });
  }
  // Botón row individual
  for (const btn of document.querySelectorAll("#dead-list .dead-btn-row")) {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const row = list.find((m) => m.id === id);
      deleteMemoIds([id], row?.title || id);
    });
  }
  // Bulk
  $("#dead-delete-bulk")?.addEventListener("click", async () => {
    const ids = [...DEAD_SELECTION];
    if (!ids.length) return;
    const btn = $("#dead-delete-bulk");
    const ok = await inlineConfirm(btn, `Borrar ${ids.length} seleccionada${ids.length === 1 ? '' : 's'}.`);
    if (!ok) return;
    deleteMemoIds(ids, `${ids.length} seleccionadas`);
  });
  // All
  $("#dead-delete-all")?.addEventListener("click", async () => {
    const ids = list.map((m) => m.id);
    const btn = $("#dead-delete-all");
    const ok = await inlineConfirm(btn, `Borrar las ${ids.length} dead memorias — irreversible.`);
    if (!ok) return;
    deleteMemoIds(ids, `${ids.length} dead memorias`);
  });
}

async function deleteMemoIds(ids, label) {
  const progress = toast({
    title: `Borrando ${label}...`,
    detail: ids.length > 1 ? `${ids.length} memorias en cola` : undefined,
    kind: "info",
    ttl: 0,
  });
  try {
    const r = await fetch("/api/memo/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ids),
    });
    const result = await r.json();
    progress.dismiss();
    if (!result.ok) {
      toast({ title: "Borrado falló", detail: result.error || "?", kind: "error", ttl: 6000 });
      return;
    }
    const d = result.deleted?.length || 0;
    const e = result.errors?.length || 0;
    const w = result.warnings?.length || 0;
    const kind = e > 0 ? (d > 0 ? "warn" : "error") : "success";
    const summary = `${d} borrada${d === 1 ? '' : 's'}`
      + (e ? ` · ${e} errores` : "")
      + (w ? ` · ${w} warnings` : "");
    const detail = e ? compactErrors(result.errors) : (w ? compactErrors(result.warnings) : undefined);
    toast({ title: summary, detail, kind, ttl: 5000 });
    // Limpiar selección de los efectivamente borrados
    for (const id of (result.deleted || [])) DEAD_SELECTION.delete(id);
    refresh();
    loadV06Features();
  } catch (e) {
    progress.dismiss();
    toast({ title: "Borrado falló", detail: e.message, kind: "error", ttl: 6000 });
  }
}

function renderStats(s) {
  const a = s.activity || {};
  const h = s.health || {};
  const total = s.totals?.all ?? 0;
  const last7 = s.saves_timeline?.slice(-7) || [];
  const avgPerDay = last7.length
    ? Math.round(last7.reduce((acc, d) => acc + d.count, 0) / last7.length)
    : 0;

  const dupePctClass = h.near_dupe_pct >= 15 ? "bad" : h.near_dupe_pct >= 5 ? "warn" : "good";
  const actionPctClass = h.actionable_pct >= 50 ? "good" : h.actionable_pct >= 30 ? "warn" : "bad";
  const tagPctClass = h.tagless_pct >= 20 ? "bad" : h.tagless_pct >= 5 ? "warn" : "good";
  const scoreClassName = scoreLabel(h.avg_score);

  $("#stats-grid").innerHTML = `
    <div class="stat-card">
      <div class="label">Total</div>
      <div class="value">${total}</div>
      <div class="sub">${s.totals.by_type.length} tipos · avg ${formatSize(0)}</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg quality score</div>
      <div class="value ${scoreClassName}">${h.avg_score}</div>
      <div class="sub">0-100 · breakdown al clickear</div>
    </div>
    <div class="stat-card">
      <div class="label">% Accionable</div>
      <div class="value ${actionPctClass}">${h.actionable_pct}%</div>
      <div class="sub">${h.actionable_count} bug/decision/preference</div>
    </div>
    <div class="stat-card">
      <div class="label">% Near-dupes</div>
      <div class="value ${dupePctClass}">${h.near_dupe_pct}%</div>
      <div class="sub">${h.near_dupe_count} memorias en pares</div>
    </div>
    <div class="stat-card">
      <div class="label">% Sin tags</div>
      <div class="value ${tagPctClass}">${h.tagless_pct}%</div>
      <div class="sub">${h.tagless_count} sin clasificar</div>
    </div>
    <div class="stat-card">
      <div class="label">Stale (&gt; 6 meses)</div>
      <div class="value">${h.stale_pct}%</div>
      <div class="sub">${h.stale_count} sin tocar</div>
    </div>
    <div class="stat-card sparkline-card">
      <div class="label">Saves últimos 30d</div>
      <div class="value">${a.saved_30d ?? 0}</div>
      <div class="sub">hoy ${a.saved_today ?? 0} · ${avgPerDay}/d</div>
      <div class="sparkline">${renderSparkline(s.saves_timeline)}</div>
    </div>
    <div class="stat-card">
      <div class="label">Recall events totales</div>
      <div class="value">${s.usage?.total_recall_events ?? 0}</div>
      <div class="sub">veces que memo inyectó en un prompt</div>
    </div>
  `;
}

function clampPct(value) {
  const n = Number(value) || 0;
  return Math.max(0, Math.min(100, n));
}

function utilityClass(pctValue, goodAt, warnAt) {
  if (pctValue >= goodAt) return "good";
  if (pctValue >= warnAt) return "warn";
  return "bad";
}

function renderUtilityCharts(s) {
  const total = s.totals?.all || 0;
  const coverage = s.utility?.coverage || {};
  const buckets = s.utility?.quality_buckets || [];
  if (!total) {
    $("#utility-charts").innerHTML = `<div class="loading">Sin memorias para graficar.</div>`;
    return;
  }

  const recallPct = clampPct(coverage.recall_hit_rate_pct);
  const activePct = clampPct(coverage.active_7d_pct);
  const deadPct = clampPct(coverage.dead_pct);
  const recallClass = utilityClass(recallPct, 30, 15);
  const healthyPct = clampPct(
    buckets
      .filter((b) => b.label === "Alta" || b.label === "Media")
      .reduce((acc, b) => acc + Number(b.pct || 0), 0)
  );
  const healthyClass = utilityClass(healthyPct, 70, 50);

  const funnelRows = [
    {
      label: "Usadas",
      count: coverage.recalled || 0,
      pct: recallPct,
      kind: "recalled",
    },
    {
      label: "Activas 7d",
      count: coverage.active_7d || 0,
      pct: activePct,
      kind: "active",
    },
    {
      label: "Muertas",
      count: coverage.dead || 0,
      pct: deadPct,
      kind: "dead",
    },
  ];
  const funnelHTML = funnelRows.map((row) => `
    <div class="funnel-row">
      <div class="funnel-label">${escapeHTML(row.label)}</div>
      <div class="funnel-track">
        <div class="funnel-fill ${row.kind}" style="width:${row.pct}%;"></div>
      </div>
      <div class="funnel-value">${row.count} · ${row.pct}%</div>
    </div>
  `).join("");

  const safeBuckets = buckets.length
    ? buckets
    : ["Alta", "Media", "Baja", "Ruido"].map((label) => ({ label, count: 0, pct: 0 }));
  const stackHTML = safeBuckets.map((b) => `
    <div
      class="quality-seg ${escapeHTML(b.label)}"
      style="width:${clampPct(b.pct)}%;"
      title="${escapeHTML(b.label)} · ${b.count} (${b.pct}%)"
    ></div>
  `).join("");
  const qualityRows = safeBuckets.map((b) => `
    <div class="quality-row">
      <div class="quality-label">${escapeHTML(b.label)}</div>
      <div class="quality-track">
        <div class="quality-fill ${escapeHTML(b.label)}" style="width:${clampPct(b.pct)}%;"></div>
      </div>
      <div class="quality-value">${b.count} · ${b.pct}%</div>
    </div>
  `).join("");

  $("#utility-charts").innerHTML = `
    <div class="utility-card">
      <div class="utility-head">
        <div class="utility-title">Uso real</div>
        <div class="utility-kpi ${recallClass}">${recallPct}%</div>
      </div>
      ${funnelHTML}
      <div class="utility-caption">
        ${coverage.recalled || 0} de ${total} memorias aparecieron en prompts;
        ${coverage.active_7d || 0} siguen vivas esta semana.
      </div>
    </div>
    <div class="utility-card">
      <div class="utility-head">
        <div class="utility-title">Calidad del corpus</div>
        <div class="utility-kpi ${healthyClass}">${healthyPct.toFixed(1)}%</div>
      </div>
      <div class="quality-stack">${stackHTML}</div>
      <div class="quality-bars">${qualityRows}</div>
      <div class="utility-caption">
        Porcentaje de memorias con score alto o medio. "Ruido" combina bajo score,
        falta de tags, tamaño pobre o duplicación.
      </div>
    </div>
  `;
}

function renderTypes(s) {
  const byType = new Map((s.totals.by_type || []).map((r) => [r.type, r.count]));
  const allCount = s.totals.all || 0;
  const orderedTypes = TYPE_ORDER.filter((t) => byType.has(t));
  for (const r of s.totals.by_type) {
    if (!orderedTypes.includes(r.type)) orderedTypes.push(r.type);
  }

  const pillsHTML = ["all", ...orderedTypes]
    .map((t) => {
      const active = (STATE.type === null && t === "all") || STATE.type === t;
      const count = t === "all" ? allCount : byType.get(t) ?? 0;
      return `<span class="type-pill ${active ? "active" : ""}" data-type="${t}">
        ${escapeHTML(t)} <span class="count">${count}</span>
      </span>`;
    })
    .join("");

  const dupeToggleClass = STATE.onlyDupes ? "active" : "";
  $("#types-row").innerHTML = `
    <span class="label-sm">Filtrar:</span>
    ${pillsHTML}
    <span class="filter-toggle ${dupeToggleClass}" id="only-dupes-toggle">
      ⚠️ solo near-dupes (${s.health.near_dupe_count})
    </span>
  `;

  for (const pill of document.querySelectorAll(".type-pill")) {
    pill.addEventListener("click", () => {
      const t = pill.dataset.type;
      STATE.type = t === "all" ? null : t;
      STATE.searchQuery = "";
      $("#search-input").value = "";
      refresh();
    });
  }
  $("#only-dupes-toggle").addEventListener("click", () => {
    STATE.onlyDupes = !STATE.onlyDupes;
    renderRecent(STATE.snapshot);
    renderTypes(STATE.snapshot);
  });
}

function flagsHTML(m) {
  const flags = [];
  if (m.in_dupe_cluster) flags.push(`<span class="flag-chip flag-dupe">dupe ${m.neighbor_count || ""}</span>`);
  if (!m.tags || m.tags.length === 0) flags.push(`<span class="flag-chip flag-tagless">sin tags</span>`);
  if (m.body_size > 0 && m.body_size < 200) flags.push(`<span class="flag-chip flag-tiny">tiny ${formatSize(m.body_size)}</span>`);
  return flags.join(" ");
}

function memoPathLinkHTML(m, extraClass = "") {
  const label = m.full_path || m.path || "";
  if (!label) return "";
  const href = m.obsidian_url || `obsidian://open?path=${encodeURIComponent(label)}`;
  const className = extraClass ? ` ${extraClass}` : "";
  return `
    <a
      class="memo-path-link${escapeHTML(className)}"
      href="${escapeHTML(href)}"
      title="${escapeHTML(label)}"
      onclick="event.stopPropagation()"
    >${escapeHTML(label)}</a>`;
}

function rowHTML(m) {
  const tags = (m.tags || []).slice(0, 3).map((t) => `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
  const flags = flagsHTML(m);
  const preview = m.body_preview ? escapeHTML(m.body_preview) + "…" : "";
  const pathLink = memoPathLinkHTML(m, "inline-path");
  const recallCount = m.recall_count ?? 0;
  const recallHTML = recallCount > 0
    ? `<span style="color: var(--cyan); font-weight:600;">${recallCount}×</span><div style="font-size:10.5px; color:var(--text-faint);">${escapeHTML(m.last_recalled_ago || "")}</div>`
    : `<span style="color: var(--text-faint);">0×</span>`;
  return `
    <tr data-id="${escapeHTML(m.id)}" class="${m.id === STATE.selectedId ? "selected" : ""}">
      <td><span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span></td>
      <td class="title-cell">
        <div class="title">${escapeHTML(m.title)}</div>
        ${pathLink ? `<div class="path-inline">${pathLink}</div>` : ""}
        <div class="preview">${preview}</div>
        <div class="flags">${tags} ${flags}</div>
      </td>
      <td class="score-cell">
        ${m.score}
        <div class="score-bar"><div class="${scoreClass(m.score)}" style="width:${m.score}%;"></div></div>
      </td>
      <td style="font-variant-numeric: tabular-nums;">${recallHTML}</td>
      <td class="ago" title="${escapeHTML(m.updated || "")}">${escapeHTML(m.ago)}</td>
    </tr>`;
}

function renderRecent(s) {
  if (!s) return;
  let rows = s.recent || [];
  if (STATE.onlyDupes) rows = rows.filter((r) => r.in_dupe_cluster);

  $("#recent-meta").textContent = `${rows.length} mostradas`;
  if (rows.length === 0) {
    $("#memos-tbody").innerHTML = `<tr><td colspan="5" class="loading">Sin resultados.</td></tr>`;
    return;
  }
  $("#memos-tbody").innerHTML = rows.map(rowHTML).join("");
  bindRowClicks();
}

function bindRowClicks() {
  for (const tr of document.querySelectorAll("#memos-tbody tr[data-id]")) {
    tr.addEventListener("click", () => selectMemo(tr.dataset.id));
  }
}

function renderDupePairs(s) {
  const pairs = s.dupe_pairs || [];
  if (pairs.length === 0) {
    $("#dupe-pairs").innerHTML = `<div class="loading">Sin pares cercanos. 👍</div>`;
    $("#dupes-meta").textContent = "0 pares · sin dupes detectados";
    $("#merge-all-btn").style.display = "none";
    return;
  }
  $("#dupes-meta").textContent = `${pairs.length} pares con embedding dist < 0.12`;
  $("#merge-all-btn").style.display = "block";
  $("#dupe-pairs").innerHTML = pairs
    .map(
      (p) => `
      <div class="pair">
        <div class="pair-head">
          <span class="dist">dist ${p.distance}</span>
          <span>cos ≈ ${(1 - p.distance * p.distance / 2).toFixed(3)}</span>
        </div>
        <div class="side" data-id="${escapeHTML(p.a.id)}">
          <span class="type-tag" data-type="${escapeHTML(p.a.type)}">${escapeHTML(p.a.type)}</span>
          <span class="nt">${escapeHTML(p.a.title)}</span>
        </div>
        <div class="side" data-id="${escapeHTML(p.b.id)}">
          <span class="type-tag" data-type="${escapeHTML(p.b.type)}">${escapeHTML(p.b.type)}</span>
          <span class="nt">${escapeHTML(p.b.title)}</span>
        </div>
      </div>`
    )
    .join("");
  for (const side of document.querySelectorAll("#dupe-pairs .side")) {
    side.addEventListener("click", () => selectMemo(side.dataset.id));
  }
}

// 3D rotating tag cloud — Fibonacci sphere lattice + perspective projection.
// Cada tag se posiciona uniforme sobre la esfera; tamaño escala con count;
// rotación auto + drag manual; depth → opacity + scale para sensación 3D.
function renderTags(s) {
  const tags = s.tags_top || [];
  const wrap = $("#tags-cloud");
  if (tags.length === 0) {
    wrap.innerHTML = `<div class="loading" style="padding:14px 16px;">Sin tags.</div>`;
    return;
  }
  if (wrap._tagCloud) {
    wrap._tagCloud.update(tags);
    return;
  }
  wrap.innerHTML = `<div class="stage" id="tag-stage"></div>`;
  const stage = $("#tag-stage");
  const radius = 130;
  let rotX = -0.25, rotY = 0;
  let velX = 0.002, velY = 0.006;
  let dragging = false, lastX = 0, lastY = 0;
  let nodes = [];

  function layout(tagsArr) {
    stage.innerHTML = "";
    const N = Math.min(tagsArr.length, 60);
    const phi = Math.PI * (3 - Math.sqrt(5)); // golden angle
    const max = tagsArr[0]?.count || 1;
    nodes = [];
    for (let i = 0; i < N; i++) {
      const t = tagsArr[i];
      const y = 1 - (i / (N - 1 || 1)) * 2;       // -1..1
      const r = Math.sqrt(1 - y * y);
      const theta = phi * i;
      const x = Math.cos(theta) * r;
      const z = Math.sin(theta) * r;
      const fontSize = 10 + Math.min(18, Math.sqrt(t.count / max) * 16);
      const el = document.createElement("span");
      el.className = "t3d";
      el.style.fontSize = `${fontSize}px`;
      el.innerHTML = `${escapeHTML(t.tag)}<span class="c">${t.count}</span>`;
      el.addEventListener("click", () => {
        // Click un tag → filtra busqueda FTS
        $("#search-input").value = t.tag;
        STATE.searchQuery = t.tag;
        refresh();
      });
      stage.appendChild(el);
      nodes.push({ x, y, z, el });
    }
  }

  function tick() {
    if (!dragging) {
      rotX += velX;
      rotY += velY;
    }
    const cx = Math.cos(rotX), sx = Math.sin(rotX);
    const cy = Math.cos(rotY), sy = Math.sin(rotY);
    for (const n of nodes) {
      // Rotación Y → rotación X
      const x1 = n.x * cy + n.z * sy;
      const z1 = -n.x * sy + n.z * cy;
      const y1 = n.y * cx - z1 * sx;
      const z2 = n.y * sx + z1 * cx;
      const px = x1 * radius;
      const py = y1 * radius;
      const depth = (z2 + 1) / 2; // 0=back, 1=front
      const scale = 0.55 + depth * 0.65;
      const opacity = 0.30 + depth * 0.70;
      n.el.style.transform = `translate3d(${px}px, ${py}px, 0) scale(${scale})`;
      n.el.style.opacity = String(opacity);
      n.el.style.zIndex = String(Math.round(depth * 100));
    }
    requestAnimationFrame(tick);
  }

  wrap.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    rotY += dx * 0.008;
    rotX += -dy * 0.008;
    velX = -dy * 0.0008;
    velY = dx * 0.0008;
  });
  // Touch — mobile/iPad swipe support
  wrap.addEventListener("touchstart", (e) => {
    if (!e.touches[0]) return;
    dragging = true;
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
  }, { passive: true });
  wrap.addEventListener("touchmove", (e) => {
    if (!dragging || !e.touches[0]) return;
    const dx = e.touches[0].clientX - lastX;
    const dy = e.touches[0].clientY - lastY;
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
    rotY += dx * 0.008;
    rotX += -dy * 0.008;
    velX = -dy * 0.0008;
    velY = dx * 0.0008;
  }, { passive: true });
  wrap.addEventListener("touchend", () => { dragging = false; });

  layout(tags);
  requestAnimationFrame(tick);
  wrap._tagCloud = { update: (t) => layout(t) };
  return;
}

function renderScoreBreakdown(breakdown) {
  if (!breakdown) return "";
  const entries = [
    ["actionable", "Tipo accionable", 25],
    ["tags", "Tiene tags", 20],
    ["size", "Tamaño OK", 20],
    ["fresh", "Reciente", 15],
    ["unique", "No es near-dupe", 20],
  ];
  return `
    <div class="score-breakdown">
      ${entries
        .map(([k, label, max]) => {
          const v = breakdown[k] ?? 0;
          const cls = v === 0 ? "row zero" : "row";
          return `<div class="${cls}"><span>${label}</span><span class="v">${v} / ${max}</span></div>`;
        })
        .join("")}
    </div>
  `;
}

async function selectMemo(id) {
  STATE.selectedId = id;
  for (const tr of document.querySelectorAll("#memos-tbody tr")) {
    tr.classList.toggle("selected", tr.dataset.id === id);
  }
  $("#detail").innerHTML = `<div class="loading">Cargando…</div>`;
  $("#detail-id").textContent = id.slice(0, 8);

  try {
    const d = await api(`/api/memo/note?id=${encodeURIComponent(id)}`);
    if (!d.ok) {
      $("#detail").innerHTML = `<div class="empty">Error: ${escapeHTML(d.error || "desconocido")}</div>`;
      return;
    }
    const m = d.memo;
    const tags = (m.tags || []).map((t) => `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
    const flags = flagsHTML({
      in_dupe_cluster: m.in_dupe_cluster,
      tags: m.tags,
      body_size: m.body_size,
    });

    const neighbors = d.neighbors || [];
    const neighborsHTML = neighbors.length
      ? `
        <div style="margin-top: 14px;">
          <div style="font-size: 11px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
            5 vecinos más cercanos (kNN)
          </div>
          <div class="neighbors-list">
            ${neighbors
              .map(
                (n) => `
                <div class="neighbor" data-id="${escapeHTML(n.id)}">
                  <span class="dist ${n.near_dupe ? "dupe" : ""}">${n.distance}</span>
                  <span class="type-tag" data-type="${escapeHTML(n.type)}">${escapeHTML(n.type)}</span>
                  <span class="nt">${escapeHTML(n.title)}</span>
                </div>`
              )
              .join("")}
          </div>
        </div>`
      : "";

    $("#detail").innerHTML = `
      <h2>${escapeHTML(m.title)}</h2>
      <div class="meta-row">
        <span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span>
        <span>·</span>
        <span title="${escapeHTML(m.updated || "")}">${escapeHTML(m.ago)} atrás</span>
        <span>·</span>
        <span>${formatSize(m.body_size)}</span>
        ${flags ? `<span>·</span> ${flags}` : ""}
      </div>
      <div>${tags}</div>
      <div class="path">${memoPathLinkHTML(m)}</div>

      <div style="display:flex; align-items:center; gap:12px; margin-top:14px;">
        <div style="font-size: 24px; font-weight: 600; color: var(--${scoreLabel(m.score) === "good" ? "green" : scoreLabel(m.score) === "warn" ? "yellow" : "red"});">
          ${m.score}
        </div>
        <div style="font-size: 11px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.04em;">
          quality score / 100
        </div>
      </div>
      ${renderScoreBreakdown(m.score_breakdown)}

      ${neighborsHTML}

      <div style="font-size: 11px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 14px; margin-bottom: 4px;">
        Body
      </div>
      <div class="body">${escapeHTML(m.body || "")}</div>
    `;

    for (const n of document.querySelectorAll(".neighbors-list .neighbor")) {
      n.addEventListener("click", () => selectMemo(n.dataset.id));
    }
  } catch (e) {
    $("#detail").innerHTML = `<div class="empty">Error: ${escapeHTML(e.message)}</div>`;
  }
}

function renderError(msg) {
  $("#error-banner").innerHTML = `<div class="banner-error">${escapeHTML(msg)}</div>`;
}

async function refresh() {
  try {
    if (STATE.searchQuery) {
      $("#results-title").innerHTML = `Resultados FTS <span class="search-mode-tag">${escapeHTML(STATE.searchQuery)}</span>`;
      const r = await api(`/api/memo/search?q=${encodeURIComponent(STATE.searchQuery)}&limit=50`);
      if (!r.ok) {
        renderError(r.error || "search failed");
        return;
      }
      $("#error-banner").innerHTML = "";
      $("#recent-meta").textContent = `${r.results.length} matches`;
      if (r.results.length === 0) {
        $("#memos-tbody").innerHTML = `<tr><td colspan="5" class="loading">Sin matches para "${escapeHTML(STATE.searchQuery)}".</td></tr>`;
      } else {
        $("#memos-tbody").innerHTML = r.results.map(rowHTML).join("");
        bindRowClicks();
      }
      return;
    }

    $("#results-title").textContent = "Memorias recientes";
    const params = new URLSearchParams({ limit: STATE.limit });
    if (STATE.type) params.set("type", STATE.type);
    const s = await api(`/api/memo?${params}`);
    STATE.snapshot = s;
    if (!s.ok) {
      renderError(s.error || "snapshot failed");
    } else {
      $("#error-banner").innerHTML = "";
    }
    renderVerdict(s.verdict);
    renderStats(s);
    renderUtilityCharts(s);
    renderTypes(s);
    renderRecent(s);
    renderTopRecalled(s.usage || {});
    renderDeadList(s.usage || {});
    renderDupePairs(s);
    renderTags(s);

    if (STATE.selectedId) {
      const tr = document.querySelector(`#memos-tbody tr[data-id="${CSS.escape(STATE.selectedId)}"]`);
      if (tr) tr.classList.add("selected");
    }
  } catch (e) {
    renderError(`fetch falló: ${e.message}`);
  }
}

// ── Search input con debounce 200ms.
let searchTimer = null;
$("#search-input").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  searchTimer = setTimeout(() => {
    STATE.searchQuery = q;
    refresh();
  }, 200);
});

// ── Merge all dupe pairs (toast feedback, sin alert/confirm chrome) ──
$("#merge-all-btn").addEventListener("click", async () => {
  if (!STATE.snapshot || !STATE.snapshot.dupe_pairs || STATE.snapshot.dupe_pairs.length === 0) {
    toast({ title: "Nada para fusionar", kind: "info", ttl: 2500 });
    return;
  }
  const btn = $("#merge-all-btn");
  const n = STATE.snapshot.dupe_pairs.length;
  const ok = await inlineConfirm(btn, `Fusionar ${n} pares — irreversible.`);
  if (!ok) {
    toast({ title: "Fusión cancelada", kind: "info", ttl: 2500 });
    return;
  }

  btn.disabled = true;
  btn.textContent = "Fusionando...";
  const progress = toast({
    title: `Fusionando ${n} pares...`,
    detail: "esto puede tardar unos segundos",
    kind: "info",
    ttl: 0,
  });

  try {
    const pairs = STATE.snapshot.dupe_pairs.map((p) => ({ a: p.a.id, b: p.b.id }));
    const r = await fetch("/api/memo/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pairs),
    });
    const result = await r.json();
    progress.dismiss();
    if (result.ok) {
      const m = result.merged?.length || 0;
      const e = result.errors?.length || 0;
      const w = result.warnings?.length || 0;
      const sk = result.skipped?.length || 0;
      const kind = e > 0 ? "warn" : (m > 0 ? "success" : "info");
      const summary = `${m} fusionados`
        + (e ? ` · ${e} errores` : "")
        + (w ? ` · ${w} warnings` : "")
        + (sk ? ` · ${sk} skipped` : "");
      let detail = "";
      if (sk) detail += `${sk} pares saltados (overlap con otros ya borrados). `;
      if (w) detail += compactErrors(result.warnings) + " ";
      if (e) {
        detail += compactErrors(result.errors);
      }
      toast({ title: summary, detail: detail.trim() || undefined, kind, ttl: 7000 });
    } else {
      toast({
        title: "Fusión falló",
        detail: result.error || "?",
        kind: "error",
        ttl: 7000,
      });
    }
  } catch (e) {
    progress.dismiss();
    toast({ title: "Fusión falló", detail: e.message, kind: "error", ttl: 7000 });
  } finally {
    btn.disabled = false;
    btn.textContent = "Fusionar todos";
    refresh();
    loadV06Features();
  }
});

// ─────────────────────────────────────────────────────────────────────
// Time-machine (v0.6 differenciador): scrubber temporal + diff panel.
// ─────────────────────────────────────────────────────────────────────

function todayISO() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function daysAgoISO(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

async function fetchTimeMachine(date) {
  const r = await api(`/api/memo/timemachine/snapshot?date=${date}&limit=200`);
  return r;
}

async function fetchDiff(fromDate, toDate) {
  const params = new URLSearchParams({ from_date: fromDate });
  if (toDate) params.set("to_date", toDate);
  return api(`/api/memo/timemachine/diff?${params}`);
}

function renderHistoricalTable(snap) {
  const rows = (snap.rows || []).map((r) => {
    const ago = r.updated ? r.updated.slice(0, 10) : "—";
    const tags = (r.tags || []).slice(0, 4).map((t) =>
      `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
    return `
      <tr data-id="${escapeHTML(r.id_full)}">
        <td><span class="type-tag" data-type="${escapeHTML(r.type)}">${escapeHTML(r.type)}</span></td>
        <td class="title-cell">
          <div class="title">${escapeHTML(r.title)}${r.body_unavailable ? ' <span class="flag-chip flag-tiny">body n/a</span>' : ''}</div>
          <div class="flags">${tags}</div>
        </td>
        <td class="score-cell"><span style="color: var(--text-faint)">—</span></td>
        <td class="score-cell"><span style="color: var(--text-faint)">—</span></td>
        <td class="ago">${escapeHTML(ago)}</td>
      </tr>`;
  }).join("");
  $("#memos-tbody").innerHTML = rows ||
    `<tr><td colspan="5" class="loading">Snapshot vacío en esa fecha.</td></tr>`;
  bindRowClicks();
}

function renderHistoricalStats(snap) {
  const counts = snap.type_counts || {};
  const cards = [
    `<div class="stat-card"><div class="label">Total al ${escapeHTML(snap.as_of_date)}</div><div class="value">${snap.total}</div></div>`,
    ...TYPE_ORDER.filter(t => counts[t]).map(t =>
      `<div class="stat-card"><div class="label">${t}</div><div class="value">${counts[t]}</div></div>`),
  ];
  $("#stats-grid").innerHTML = cards.join("");
}

function renderDiff(d) {
  if (!d || !d.ok) {
    $("#diff-panel").style.display = "none";
    return;
  }
  $("#diff-panel").style.display = "";
  $("#diff-range").textContent = `${d.from_date} → ${d.to_date}`;
  $("#diff-meta").textContent = d.summary;
  $("#diff-stats").innerHTML = `
    <div class="diff-stat added"><span class="n">+${d.added.length}</span><span class="l">added</span></div>
    <div class="diff-stat removed"><span class="n">−${d.removed.length}</span><span class="l">removed</span></div>
    <div class="diff-stat updated"><span class="n">~${d.updated.length}</span><span class="l">updated</span></div>
  `;
  const rowHTML = (sign, cls, r, extra = "") => `
    <div class="diff-row ${cls}" data-id="${escapeHTML(r.id_full)}">
      <span class="sign">${sign}</span>
      <span class="type-tag" data-type="${escapeHTML(r.type || 'note')}">${escapeHTML(r.type || '—')}</span>
      <span class="t">${escapeHTML(r.title || '(untitled)')}</span>
      ${extra}
    </div>`;
  const added = d.added.slice(0, 100).map(r => rowHTML("+", "added", r)).join("");
  const removed = d.removed.slice(0, 100).map(r => rowHTML("−", "removed", r)).join("");
  const updated = d.updated.slice(0, 100).map(u => rowHTML(
    "~", "updated", u,
    `<span class="changed-fields">${escapeHTML(u.changed_fields.join(","))}</span>`,
  )).join("");
  $("#diff-rows").innerHTML = added + removed + updated;
  for (const row of document.querySelectorAll("#diff-rows .diff-row")) {
    row.addEventListener("click", () => selectMemo(row.dataset.id));
  }
}

async function applyTimeMachine() {
  const date = STATE.tmDate;
  const tmBar = $("#tm-bar");
  if (!date || date === todayISO()) {
    STATE.tmDate = null;
    STATE.tmSnapshot = null;
    STATE.tmDiff = null;
    tmBar.classList.remove("historical");
    $("#tm-summary").textContent = "Snapshot del corpus en cualquier fecha pasada (memo v0.6 time-machine).";
    $("#diff-panel").style.display = "none";
    $("#verdict-banner").style.display = "";
    $("#utility-charts").style.display = "";
    refresh();
    return;
  }
  tmBar.classList.add("historical");
  $("#tm-summary").innerHTML =
    `Viendo memoria al <strong>${escapeHTML(date)}</strong> — verdict/dupes/recall son live, ocultos en vista histórica.`;
  $("#verdict-banner").style.display = "none";
  $("#utility-charts").style.display = "none";

  const [snap, diff] = await Promise.all([
    fetchTimeMachine(date),
    fetchDiff(date, todayISO()),
  ]);
  if (!snap.ok) {
    renderError(snap.error || "time-machine snapshot failed");
    return;
  }
  STATE.tmSnapshot = snap;
  STATE.tmDiff = diff;
  $("#results-title").innerHTML =
    `Snapshot al <span class="search-mode-tag">${escapeHTML(snap.as_of_date)}</span>`;
  $("#recent-meta").textContent = `${snap.total} memorias`;
  renderHistoricalStats(snap);
  renderHistoricalTable(snap);
  renderDiff(diff);
}

$("#tm-date").addEventListener("change", (e) => {
  STATE.tmDate = e.target.value || null;
  applyTimeMachine();
});
$("#tm-today").addEventListener("click", () => {
  STATE.tmDate = null;
  $("#tm-date").value = "";
  applyTimeMachine();
});
$("#tm-7d").addEventListener("click", () => {
  const d = daysAgoISO(7);
  STATE.tmDate = d;
  $("#tm-date").value = d;
  applyTimeMachine();
});
$("#tm-30d").addEventListener("click", () => {
  const d = daysAgoISO(30);
  STATE.tmDate = d;
  $("#tm-date").value = d;
  applyTimeMachine();
});
$("#tm-diff").addEventListener("click", async () => {
  const from = STATE.tmDate || daysAgoISO(30);
  const diff = await fetchDiff(from, todayISO());
  STATE.tmDiff = diff;
  renderDiff(diff);
  $("#diff-panel").scrollIntoView({ behavior: "smooth", block: "start" });
});

// ─────────────────────────────────────────────────────────────────────
// Knowledge graph — force-directed con velocity-verlet, vanilla SVG.
// ─────────────────────────────────────────────────────────────────────

const NODE_COLOR_BY_TYPE = {
  person: "#f778ba",
  project: "#79c0ff",
  technology: "#ffa657",
  file: "#7ee787",
  org: "#d2a8ff",
  concept: "#e3c27a",
};

function nodeRadius(count) {
  return 4 + Math.min(20, Math.sqrt(count) * 2.2);
}

async function fetchGraph() {
  return api(`/api/memo/graph?limit_nodes=80&min_count=1`);
}

async function renderGraph(data) {
  const wrap = $("#graph-wrap");
  if (!data || !data.ok) {
    wrap.innerHTML = `<div class="graph-empty-state">graph error: ${escapeHTML((data && data.error) || "?")}</div>`;
    return;
  }
  if (!data.nodes || data.nodes.length === 0) {
    wrap.innerHTML = `
      <div class="graph-empty-state">
        Graph vacío. Corré <code>memo extract-entities --all</code><br>
        para extraer entidades del corpus y poblar el grafo.
      </div>`;
    return;
  }
  $("#graph-meta").textContent =
    `${data.nodes.length} entidades · ${data.edges.length} edges · ${data.stats.entities} total`;

  // ── Esperar hasta 3s a que las libs 3D estén disponibles ─────
  const libsReady = async () => {
    for (let i = 0; i < 30; i++) {
      if (typeof ForceGraph3D === "function" && typeof THREE !== "undefined") return true;
      await new Promise((r) => setTimeout(r, 100));
    }
    return false;
  };
  if (!(await libsReady())) {
    wrap.innerHTML = `<div class="graph-empty-state">No se pudo cargar el motor 3D (three.js / 3d-force-graph).</div>`;
    return;
  }

  const nodes = data.nodes.map((n) => ({ ...n }));
  const links = data.edges.map((e) => ({ source: e.source, target: e.target, weight: e.weight }));

  wrap.innerHTML = `
    <div id="graph-3d-container"></div>
    <div class="graph-legend">
      ${Object.entries(NODE_COLOR_BY_TYPE).map(([k, c]) =>
    `<div class="row"><span class="sw" style="background:${c}"></span>${k}</div>`).join("")}
    </div>
    <div class="graph-zoom-controls">
      <button class="gzoom-btn" id="gzoom-in"  title="Zoom in">+</button>
      <button class="gzoom-btn" id="gzoom-reset" title="Reset">⟳</button>
      <button class="gzoom-btn" id="gzoom-out" title="Zoom out">−</button>
      <button class="gzoom-btn active" id="gzoom-labels" title="Toggle labels" style="font-size:11px;letter-spacing:-0.5px">aA</button>
    </div>
    <div class="graph-hint">drag → rotar · scroll → zoom · click → foco</div>
  `;

  const container = document.getElementById("graph-3d-container");
  container.style.width = "100%";
  container.style.height = "100%";

  const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#14141a";
  let showLabels = true;

  const Graph = ForceGraph3D({ controlType: "trackball" })
    (container)
    .backgroundColor(bg)
    .graphData({ nodes, links })
    .nodeId("id")
    .nodeVal((n) => Math.max(1, n.count || 1))
    .nodeRelSize(3.5)
    .nodeOpacity(1.0)
    .nodeLabel((n) => `
      <div style="background:#1c1c24;border:1px solid #3e3e46;border-radius:6px;padding:6px 10px;font-size:11px;color:#ececed;font-family:system-ui,sans-serif;max-width:220px;">
        <strong style="color:#fff">${escapeHTML(n.name)}</strong>
        <div style="color:#8a8a94;margin-top:3px;">${escapeHTML(n.type)} · ${n.count} mención${n.count === 1 ? "" : "es"}</div>
      </div>`)
    .nodeThreeObject((n) => {
      const radius = Math.max(2.5, Math.min(14, Math.sqrt(n.count || 1) * 1.9));
      const geo = new THREE.SphereGeometry(radius, 14, 14);
      const color = NODE_COLOR_BY_TYPE[n.type] || "#aaaaaa";
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
      const sphere = new THREE.Mesh(geo, mat);
      n._material = mat;
      n._radius = radius;
      if (typeof SpriteText === "function") {
        const sprite = new SpriteText(n.name || "");
        sprite.material.depthWrite = false;
        sprite.color = "#d0d0d8";
        sprite.textHeight = 3;
        sprite.padding = 0.8;
        sprite.position.set(0, radius + 4, 0);
        sprite.visible = showLabels;
        sphere.add(sprite);
        n._labelSprite = sprite;
      }
      return sphere;
    })
    .nodeThreeObjectExtend(false)
    .linkColor(() => "rgba(110,110,140,0.35)")
    .linkWidth((l) => Math.max(0.3, Math.min(2.8, Math.sqrt(l.weight || 1) * 0.55)))
    .linkOpacity(0.5)
    .onNodeClick((node) => {
      const distance = 85;
      const r = Math.hypot(node.x || 0, node.y || 0, node.z || 0);
      const distRatio = r > 0 ? 1 + distance / r : 1;
      Graph.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: ((node.z || 0) * distRatio) || distance },
        node,
        900
      );
    })
    .onNodeHover((node) => { container.style.cursor = node ? "pointer" : ""; });

  Graph.d3Force("charge").strength(-130);
  Graph.d3Force("link").distance(32);
  Graph.cooldownTicks(160);
  Graph.onEngineStop(() => Graph.zoomToFit(400, 50));

  // Zoom buttons
  const camMove = (factor) => {
    const { x, y, z } = Graph.cameraPosition();
    Graph.cameraPosition({ x: x * factor, y: y * factor, z: z * factor }, null, 300);
  };
  document.getElementById("gzoom-in").addEventListener("click", () => camMove(0.7));
  document.getElementById("gzoom-out").addEventListener("click", () => camMove(1.43));
  document.getElementById("gzoom-reset").addEventListener("click", () => Graph.zoomToFit(600, 50));

  // Labels toggle
  document.getElementById("gzoom-labels").addEventListener("click", (ev) => {
    showLabels = !showLabels;
    nodes.forEach((n) => { if (n._labelSprite) n._labelSprite.visible = showLabels; });
    ev.currentTarget.classList.toggle("active", showLabels);
  });
}

// ─────────────────────────────────────────────────────────────────────
// Timeline temporal — saves/updates/deletes per día (SVG stacked bars).
// ─────────────────────────────────────────────────────────────────────

async function fetchTimeline(days = 30) {
  return api(`/api/memo/temporal/timeline?days=${days}`);
}

function renderTimeline(data) {
  if (!data || !data.ok || !data.series || data.series.length === 0) {
    $("#timeline-svg").innerHTML = "";
    $("#timeline-totals").innerHTML = '<span style="color:var(--text-faint)">sin datos</span>';
    return;
  }
  const series = data.series;
  const W = 800, H = 110, pad = 8;
  const bw = (W - pad * 2) / series.length - 1.2;
  const max = Math.max(1, ...series.map((s) => s.saves + s.updates + s.deletes));
  const sy = (H - pad * 2) / max;

  const svg = $("#timeline-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  let html = "";
  series.forEach((s, i) => {
    const x = pad + i * (bw + 1.2);
    let yCursor = H - pad;
    if (s.saves) {
      const h = s.saves * sy;
      html += `<rect x="${x}" y="${yCursor - h}" width="${bw}" height="${h}" fill="var(--green)" opacity="0.85"><title>${s.date} · saves ${s.saves}</title></rect>`;
      yCursor -= h;
    }
    if (s.updates) {
      const h = s.updates * sy;
      html += `<rect x="${x}" y="${yCursor - h}" width="${bw}" height="${h}" fill="var(--yellow)" opacity="0.85"><title>${s.date} · updates ${s.updates}</title></rect>`;
      yCursor -= h;
    }
    if (s.deletes) {
      const h = s.deletes * sy;
      html += `<rect x="${x}" y="${yCursor - h}" width="${bw}" height="${h}" fill="var(--red)" opacity="0.85"><title>${s.date} · deletes ${s.deletes}</title></rect>`;
    }
  });
  const labelIdx = [0, Math.floor(series.length / 2), series.length - 1];
  for (const i of labelIdx) {
    const x = pad + i * (bw + 1.2);
    html += `<text x="${x}" y="${H - 1}" font-size="9" fill="var(--text-faint)">${series[i].date.slice(5)}</text>`;
  }
  svg.innerHTML = html;
  const t = data.total;
  $("#timeline-totals").innerHTML = `
    <span><strong>${t.saves}</strong> saves</span>
    <span><strong>${t.updates}</strong> updates</span>
    <span><strong>${t.deletes}</strong> deletes</span>
    <span style="color:var(--text-faint)">en ${data.days}d</span>
  `;
}

// ─────────────────────────────────────────────────────────────────────
// Stale memorias.
// ─────────────────────────────────────────────────────────────────────

async function fetchStale(days = 90) {
  return api(`/api/memo/temporal/stale?days=${days}&limit=30`);
}

function renderStale(data) {
  if (!data || !data.ok) {
    $("#stale-list").innerHTML = `<div class="loading">sin datos</div>`;
    return;
  }
  $("#stale-meta").textContent = `${data.total_stale} total · sin update > ${data.threshold_days}d`;
  if (!data.rows.length) {
    $("#stale-list").innerHTML = `<div class="loading">🎉 sin memorias stale (corpus joven o todas frescas)</div>`;
    return;
  }
  $("#stale-list").innerHTML = data.rows.map((r) => `
    <div class="stale-row" data-id="${escapeHTML(r.id_full)}">
      <span class="days ${r.days_old > 180 ? 'old' : ''}">${r.days_old}d</span>
      <span class="type-tag" data-type="${escapeHTML(r.type)}">${escapeHTML(r.type)}</span>
      <span class="t">${escapeHTML(r.title)}</span>
    </div>
  `).join("");
  for (const el of document.querySelectorAll("#stale-list .stale-row")) {
    el.addEventListener("click", () => selectMemo(el.dataset.id));
  }
}

async function loadV06Features() {
  try {
    const [g, t, s] = await Promise.all([
      fetchGraph(), fetchTimeline(30), fetchStale(90),
    ]);
    renderGraph(g);
    renderTimeline(t);
    renderStale(s);
  } catch (e) {
    console.warn("memo v0.6 features load failed:", e);
  }
}

refresh();
loadV06Features();
