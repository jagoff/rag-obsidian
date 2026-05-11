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

  $("#verdict-banner").className = `verdict-banner ${outcome}`;
  $("#verdict-banner").innerHTML = `
    <div class="verdict-headline">
      <span class="verdict-pill ${outcome}">${escapeHTML(pillLabel)}</span>
      <span class="verdict-summary">${escapeHTML(v.summary)}</span>
    </div>
    <div class="verdict-criteria">${criteriaHTML}</div>
  `;
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
        </div>
      </div>`
    )
    .join("");
  for (const r of document.querySelectorAll("#top-recalled-list .recall-row")) {
    r.addEventListener("click", () => selectMemo(r.dataset.id));
  }
}

function renderDeadList(usage) {
  const list = usage.dead_memorias || [];
  $("#dead-meta").textContent = `${usage.dead_count} memorias`;
  if (list.length === 0) {
    $("#dead-list").innerHTML = `<div class="loading">Sin dead memorias. Todo lo que tiene &gt; 2d se está usando. 👍</div>`;
    return;
  }
  $("#dead-list").innerHTML = list
    .map(
      (m) => `
      <div class="recall-row" data-id="${escapeHTML(m.id)}">
        <span class="count-badge" style="color: var(--red);">0×</span>
        <div style="flex: 1; min-width: 0;">
          <div class="title">${escapeHTML(m.title)}</div>
          <div class="meta-line">
            <span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span>
            · creado ${escapeHTML(m.age)} atrás · nunca matcheó un prompt
          </div>
        </div>
      </div>`
    )
    .join("");
  for (const r of document.querySelectorAll("#dead-list .recall-row")) {
    r.addEventListener("click", () => selectMemo(r.dataset.id));
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

function rowHTML(m) {
  const tags = (m.tags || []).slice(0, 3).map((t) => `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
  const flags = flagsHTML(m);
  const preview = m.body_preview ? escapeHTML(m.body_preview) + "…" : "";
  const recallCount = m.recall_count ?? 0;
  const recallHTML = recallCount > 0
    ? `<span style="color: var(--cyan); font-weight:600;">${recallCount}×</span><div style="font-size:10.5px; color:var(--text-faint);">${escapeHTML(m.last_recalled_ago || "")}</div>`
    : `<span style="color: var(--text-faint);">0×</span>`;
  return `
    <tr data-id="${escapeHTML(m.id)}" class="${m.id === STATE.selectedId ? "selected" : ""}">
      <td><span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span></td>
      <td class="title-cell">
        <div class="title">${escapeHTML(m.title)}</div>
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
    return;
  }
  $("#dupes-meta").textContent = `${pairs.length} pares con embedding dist < 0.12`;
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

function renderTags(s) {
  const tags = s.tags_top || [];
  if (tags.length === 0) {
    $("#tags-cloud").innerHTML = `<div class="loading">Sin tags.</div>`;
    return;
  }
  $("#tags-cloud").innerHTML = tags
    .map((t) => `<span class="tag-cloud-item">${escapeHTML(t.tag)}<span class="count">${t.count}</span></span>`)
    .join("");
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
      <div class="path">${escapeHTML(m.path || "")}</div>

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

refresh();
