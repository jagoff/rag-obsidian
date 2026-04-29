/* atlas.js — Hidrata /atlas con datos de /api/atlas.
 *
 * Tres bloques principales:
 *   1) KPIs + entidades (Chart.js sparklines)
 *   2) Hot/Stale + co-ocurrencias
 *   3) Force-directed graph estilo Obsidian (D3 v7)
 *
 * El graph usa el mismo patrón que Obsidian: forceSimulation con
 * forceLink + forceManyBody + forceCenter + forceCollide. Drag,
 * zoom y pan via d3-zoom + d3-drag. Click en un nodo highlightea
 * el subgrafo 1-hop (dim del resto).
 */

const SUN_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>';
const MOON_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

const TYPE_COLORS = {
  person: "#79c0ff",        // cyan
  location: "#7ee787",      // green
  organization: "#ffa657",  // orange
  event: "#d2a8ff",         // purple
};

const TYPE_LABELS = {
  person: "persona",
  location: "lugar",
  organization: "organización",
  event: "evento",
};

const state = {
  windowDays: 30,
  payload: null,
  graphSim: null,
  graphSvg: null,
  graphTransform: null,
  showLabels: true,
  selectedNode: null,
  sparkCharts: new Map(),  // entity_id → Chart.js instance
};

// ── Theme toggle (mismo patrón que dashboard/finance) ────────────────────
function applyTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  const icon = document.getElementById("theme-icon");
  if (icon) icon.innerHTML = t === "light" ? MOON_SVG : SUN_SVG;
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("rag-theme", next); } catch (e) {}
  applyTheme();
  // El graph usa colores CSS via getComputedStyle — re-pintamos.
  if (state.graphSvg) repaintGraphColors();
}
applyTheme();

// ── Collapse handlers ───────────────────────────────────────────────────
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".collapse-btn");
  if (!btn) return;
  const targetId = btn.dataset.collapse;
  const body = document.getElementById(targetId);
  if (!body) return;
  const isCollapsed = body.classList.toggle("collapsed");
  btn.setAttribute("aria-expanded", String(!isCollapsed));
  btn.textContent = isCollapsed ? "+" : "−";
});

// ── Window-days segmented control ───────────────────────────────────────
document.querySelectorAll(".seg-btn[data-days]").forEach((b) => {
  b.addEventListener("click", () => {
    const days = parseInt(b.dataset.days, 10);
    if (days === state.windowDays) return;
    document.querySelectorAll(".seg-btn[data-days]").forEach((x) => {
      x.classList.toggle("active", x === b);
      x.setAttribute("aria-selected", x === b ? "true" : "false");
    });
    state.windowDays = days;
    fetchAndRender();
  });
});

document.getElementById("theme-toggle").addEventListener("click", toggleTheme);

// ── Fetch + render ──────────────────────────────────────────────────────
async function fetchAndRender() {
  setStatus("Cargando atlas…");
  try {
    const r = await fetch(`/api/atlas?window_days=${state.windowDays}&top_entities=50&graph_top_notes=250`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const payload = await r.json();
    state.payload = payload;
    renderAll(payload);
    setStatus("Atlas listo.");
  } catch (e) {
    setStatus("Error cargando atlas: " + e.message);
    console.error(e);
  }
}

function setStatus(msg) {
  const s = document.getElementById("atlas-status");
  if (s) s.textContent = msg;
}

function renderAll(p) {
  // Header meta
  const meta = p.meta || {};
  const mp = document.getElementById("meta-period");
  if (mp) mp.textContent = `Ventana ${meta.window_days || state.windowDays}d`;
  const mu = document.getElementById("meta-updated");
  if (mu && meta.generated_at) {
    const d = new Date(meta.generated_at);
    mu.textContent = `actualizado ${d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" })}`;
  }

  renderKPIs(p.kpis || {});
  renderEntities(p.entities_by_type || {});
  renderHotStale(p.hot || [], p.stale || []);
  renderCooc(p.cooccurrence || []);
  renderGraph(p.graph || { nodes: [], links: [] });
}

// ── KPIs ────────────────────────────────────────────────────────────────
function renderKPIs(k) {
  const set = (id, value, sub) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.querySelector(".kpi-value").textContent = formatCount(value);
    if (sub) {
      const ss = el.querySelector(".kpi-sub");
      if (ss) ss.textContent = sub;
    }
  };
  set("kpi-entities", k.n_entities, `personas + lugares + orgs + eventos`);
  set("kpi-mentions", k.n_mentions, `total a través del vault`);
  set("kpi-notes", k.n_notes, `${state.payload?.graph?.truncated ? `mostrando top ${state.payload.graph.nodes.length}` : "todas"}`);
  set("kpi-edges", k.n_edges, `wikilinks 1-hop`);
}

function formatCount(n) {
  if (typeof n !== "number" || !isFinite(n)) return "—";
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

// ── Entities lists con sparklines ───────────────────────────────────────
function renderEntities(byType) {
  // Destruimos sparklines previas para liberar memoria.
  state.sparkCharts.forEach((c) => c.destroy());
  state.sparkCharts.clear();

  Object.keys(TYPE_COLORS).forEach((type) => {
    const list = byType[type] || [];
    const ul = document.getElementById(`list-${type}`);
    const cnt = document.getElementById(`cnt-${type}`);
    if (cnt) cnt.textContent = `${list.length}`;
    if (!ul) return;
    if (!list.length) {
      ul.innerHTML = '<li class="empty">sin datos</li>';
      return;
    }
    ul.innerHTML = "";
    list.slice(0, 15).forEach((e) => {  // top 15 visibles por columna
      const li = document.createElement("li");
      li.className = "ent-row";
      li.dataset.entityId = String(e.id);
      li.dataset.type = type;
      li.title = `${e.name} — ${TYPE_LABELS[type]} · ${e.mention_count} menciones totales · ${e.recent_mentions} en últimos ${state.windowDays}d${e.aliases?.length ? ` · alias: ${e.aliases.join(", ")}` : ""}`;
      li.innerHTML = `
        <span class="ent-name">${escapeHtml(e.name)}</span>
        <canvas class="ent-spark" id="spark-${e.id}"></canvas>
        <span class="ent-count">${formatCount(e.mention_count)}</span>
      `;
      ul.appendChild(li);
      // Sparkline inline.
      const ctx = li.querySelector("canvas");
      if (ctx && e.sparkline?.length) {
        const chart = new Chart(ctx, {
          type: "line",
          data: {
            labels: e.sparkline.map((_, i) => String(i)),
            datasets: [{
              data: e.sparkline,
              borderColor: TYPE_COLORS[type],
              backgroundColor: TYPE_COLORS[type] + "33",
              borderWidth: 1.4,
              fill: true,
              tension: 0.3,
              pointRadius: 0,
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
          },
        });
        state.sparkCharts.set(e.id, chart);
      }
    });
  });
}

// ── Hot vs Stale ────────────────────────────────────────────────────────
function renderHotStale(hot, stale) {
  const ulHot = document.getElementById("list-hot");
  const ulStale = document.getElementById("list-stale");
  if (ulHot) {
    if (!hot.length) ulHot.innerHTML = '<li class="empty">nada destacado en la ventana</li>';
    else {
      ulHot.innerHTML = hot.map((e) => `
        <li class="hs-row" title="${e.recent_mentions} menciones en últimos ${state.windowDays}d (vs ${e.prev_mentions} en los ${state.windowDays}d anteriores)">
          <span class="hs-name">${escapeHtml(e.name)}<span class="type-pill" style="color:${TYPE_COLORS[e.type]}">${TYPE_LABELS[e.type] || e.type}</span></span>
          <span class="hs-meta up">+${e.recent_mentions - e.prev_mentions}</span>
        </li>
      `).join("");
    }
  }
  if (ulStale) {
    if (!stale.length) ulStale.innerHTML = '<li class="empty">nada que extrañes recientemente</li>';
    else {
      ulStale.innerHTML = stale.map((e) => {
        const days = e.days_since_last;
        const daysLabel = days >= 365 ? `${Math.round(days / 365 * 10) / 10}a` : `${days}d`;
        return `
        <li class="hs-row" title="${e.mention_count} menciones totales — última hace ${daysLabel}">
          <span class="hs-name">${escapeHtml(e.name)}<span class="type-pill" style="color:${TYPE_COLORS[e.type]}">${TYPE_LABELS[e.type] || e.type}</span></span>
          <span class="hs-meta down">${daysLabel} sin contacto</span>
        </li>
      `;
      }).join("");
    }
  }
}

// ── Co-ocurrencias ──────────────────────────────────────────────────────
function renderCooc(pairs) {
  const el = document.getElementById("cooc-list");
  if (!el) return;
  if (!pairs.length) {
    el.innerHTML = '<div class="empty">sin co-ocurrencias suficientes en la ventana</div>';
    return;
  }
  const max = Math.max(...pairs.map((p) => p.count));
  el.innerHTML = pairs.slice(0, 30).map((p) => {
    const pct = max > 0 ? (p.count / max) * 100 : 0;
    return `
      <div class="cooc-row" title="${p.count} chunks contienen a ambas">
        <span class="cooc-side left" style="color:${TYPE_COLORS[p.a_type] || "var(--text)"}">${escapeHtml(p.a_name)}</span>
        <span class="cooc-side right" style="color:${TYPE_COLORS[p.b_type] || "var(--text)"}">${escapeHtml(p.b_name)}</span>
        <span class="cooc-count">${p.count}</span>
        <div class="cooc-bar"><span style="width:${pct.toFixed(1)}%"></span></div>
      </div>
    `;
  }).join("");
}

// ── Graph estilo Obsidian (D3 force-directed) ───────────────────────────
function renderGraph(graph) {
  const card = document.getElementById("graph-card");
  const svgEl = document.getElementById("graph-svg");
  const stats = document.getElementById("graph-stats");
  const tip = document.getElementById("graph-tip");
  if (!card || !svgEl) return;

  // Clear previous
  if (state.graphSim) state.graphSim.stop();
  d3.select(svgEl).selectAll("*").remove();

  const nodes = (graph.nodes || []).map((n) => ({ ...n }));  // copia mutable para D3
  const links = (graph.links || []).map((l) => ({ ...l }));

  const totalNotes = graph.total_notes || nodes.length;
  const totalEdges = graph.total_edges || links.length;
  if (stats) {
    stats.textContent = graph.truncated
      ? `${nodes.length} de ${totalNotes} notas · ${links.length} de ${totalEdges} conexiones (top por degree)`
      : `${nodes.length} notas · ${links.length} conexiones`;
  }

  if (!nodes.length) {
    d3.select(svgEl).append("text")
      .attr("x", "50%").attr("y", "50%")
      .attr("text-anchor", "middle")
      .attr("fill", "var(--text-faint)")
      .style("font-size", "13px")
      .text("Sin notas indexadas todavía. Corré `rag index` para poblar el grafo.");
    return;
  }

  const rect = card.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  const svg = d3.select(svgEl)
    .attr("viewBox", [0, 0, width, height])
    .attr("preserveAspectRatio", "xMidYMid meet");

  // ─ Folder color palette: reproducible per-folder hue. Igual que Obsidian
  //   le asigna un color por carpeta/grupo si abrís Group Settings.
  const folderColors = buildFolderColorMap(nodes);

  // ─ Layers ────────────────────────────────────────────────
  const root = svg.append("g").attr("class", "g-root");
  const linkLayer = root.append("g").attr("class", "g-links").attr("stroke-linecap", "round");
  const nodeLayer = root.append("g").attr("class", "g-nodes");
  const labelLayer = root.append("g").attr("class", "g-labels");

  // ─ Zoom + pan ────────────────────────────────────────────
  const zoom = d3.zoom()
    .scaleExtent([0.15, 4])
    .on("zoom", (event) => {
      root.attr("transform", event.transform);
      // Mostrar labels solo en zoom-in para evitar visual overload.
      const scale = event.transform.k;
      const labelClass = scale < 0.7 ? "g-label hidden" : "g-label";
      labelLayer.selectAll("text").attr("class", labelClass);
      state.graphTransform = event.transform;
    });
  svg.call(zoom).on("dblclick.zoom", null);  // disable double-click zoom (interfiere con click)

  // ─ Force simulation ─────────────────────────────────────
  // Parámetros calibrados para 250 nodos. Mismo perfil que Obsidian:
  // repulsión moderada, link distance proporcional a carga, centering
  // gravitacional débil, collide igual al radio + slack.
  const nodeRadius = (n) => Math.max(3, Math.min(14, 3 + Math.sqrt(n.degree || 1) * 1.6));

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links)
      .id((d) => d.id)
      .distance(46)
      .strength(0.35))
    .force("charge", d3.forceManyBody().strength(-130))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius((d) => nodeRadius(d) + 3))
    .force("x", d3.forceX(width / 2).strength(0.04))
    .force("y", d3.forceY(height / 2).strength(0.04));

  state.graphSim = sim;

  // ─ Links ────────────────────────────────────────────────
  const linkSel = linkLayer.selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "g-link")
    .attr("stroke-width", (d) => Math.max(0.6, Math.min(2.5, Math.sqrt(d.weight || 1))));

  // ─ Nodes ────────────────────────────────────────────────
  const nodeSel = nodeLayer.selectAll("circle")
    .data(nodes, (d) => d.id)
    .join("circle")
    .attr("class", "g-node")
    .attr("r", nodeRadius)
    .attr("fill", (d) => folderColors.get(d.folder) || "var(--text-faint)")
    .call(makeDrag(sim))
    .on("mouseenter", (event, d) => onNodeHover(d, event))
    .on("mouseleave", () => onNodeLeave())
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d);
    });

  // ─ Labels ───────────────────────────────────────────────
  const labelSel = labelLayer.selectAll("text")
    .data(nodes, (d) => d.id)
    .join("text")
    .attr("class", "g-label")
    .attr("dx", 8)
    .attr("dy", "0.32em")
    .text((d) => d.label || "(sin título)");

  // ─ Click en background = clear selection ────────────────
  svg.on("click", () => clearSelection());

  // ─ Tick handler ─────────────────────────────────────────
  sim.on("tick", () => {
    linkSel
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    nodeSel
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y);
    labelSel
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y);
  });

  // Stop after enough ticks — force layouts converge fast.
  setTimeout(() => sim.alphaTarget(0).alpha(0).stop(), 8000);

  // ─ Legend ───────────────────────────────────────────────
  renderGraphLegend(folderColors);

  // ─ Controls ─────────────────────────────────────────────
  document.getElementById("graph-zoom-in").onclick = () => svg.transition().duration(220).call(zoom.scaleBy, 1.4);
  document.getElementById("graph-zoom-out").onclick = () => svg.transition().duration(220).call(zoom.scaleBy, 1 / 1.4);
  document.getElementById("graph-reset").onclick = () => svg.transition().duration(280).call(zoom.transform, d3.zoomIdentity);
  document.getElementById("graph-toggle-labels").onclick = () => {
    state.showLabels = !state.showLabels;
    labelLayer.style("display", state.showLabels ? null : "none");
  };

  // Tooltips wiring — atado al SVG container para coords correctas.
  function onNodeHover(d, event) {
    if (!tip) return;
    tip.innerHTML = `
      <div><strong>${escapeHtml(d.label || "(sin título)")}</strong></div>
      <div class="tip-folder">${escapeHtml(d.folder || "—")}</div>
      <div class="tip-folder">${d.degree} conexión${d.degree === 1 ? "" : "es"} · ${d.n_chunks} chunk${d.n_chunks === 1 ? "" : "s"}</div>
    `;
    const rect = card.getBoundingClientRect();
    tip.style.left = `${event.clientX - rect.left + 14}px`;
    tip.style.top = `${event.clientY - rect.top + 14}px`;
    tip.classList.add("show");
  }
  function onNodeLeave() {
    if (tip) tip.classList.remove("show");
  }

  function selectNode(d) {
    state.selectedNode = d;
    const neighborIds = new Set([d.id]);
    links.forEach((l) => {
      const sId = (typeof l.source === "object") ? l.source.id : l.source;
      const tId = (typeof l.target === "object") ? l.target.id : l.target;
      if (sId === d.id) neighborIds.add(tId);
      if (tId === d.id) neighborIds.add(sId);
    });
    nodeSel.classed("dim", (n) => !neighborIds.has(n.id));
    linkSel.classed("hl", (l) => {
      const sId = (typeof l.source === "object") ? l.source.id : l.source;
      const tId = (typeof l.target === "object") ? l.target.id : l.target;
      return sId === d.id || tId === d.id;
    });
    linkSel.classed("dim", (l) => {
      const sId = (typeof l.source === "object") ? l.source.id : l.source;
      const tId = (typeof l.target === "object") ? l.target.id : l.target;
      return sId !== d.id && tId !== d.id;
    });
    labelSel.classed("hidden", (n) => !neighborIds.has(n.id));
    labelSel.classed("hl", (n) => n.id === d.id);
  }
  function clearSelection() {
    state.selectedNode = null;
    nodeSel.classed("dim", false);
    linkSel.classed("dim", false).classed("hl", false);
    labelSel.classed("hidden", false).classed("hl", false);
  }
}

function repaintGraphColors() {
  // Forzar re-render del graph con la nueva paleta del tema.
  if (state.payload) renderGraph(state.payload.graph || { nodes: [], links: [] });
}

function buildFolderColorMap(nodes) {
  // Asignamos un color reproducible por carpeta (top-level segment).
  // Top-level porque el vault tiene PARA (00-Inbox/01-Projects/02-Areas/...)
  // y queremos que carpetas hermanas (ej. 02-Areas/Personal vs
  // 02-Areas/Work) tengan distinto color, pero las sub-sub no fragmenten.
  const palette = [
    "#79c0ff", "#7ee787", "#ffa657", "#d2a8ff", "#f778ba",
    "#e3c27a", "#5a9bd5", "#a5d6a7", "#ffb38a", "#c8a8ff",
    "#ff9ec4", "#f5cf8a",
  ];
  const folderTop = (f) => {
    if (!f) return "(sin carpeta)";
    const parts = f.split("/");
    if (parts.length >= 2) return parts[0] + "/" + parts[1];
    return parts[0];
  };
  const counts = new Map();
  nodes.forEach((n) => {
    const k = folderTop(n.folder);
    counts.set(k, (counts.get(k) || 0) + 1);
  });
  // Ordenamos por cantidad para que los grupos más grandes reciban los
  // colores más legibles primero.
  const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const map = new Map();
  ordered.forEach(([folder, _], i) => {
    map.set(folder, palette[i % palette.length]);
  });
  // Map por carpeta completa (no solo top) → color del top-level que le
  // corresponde. Así se ve la consistencia visual.
  const out = new Map();
  nodes.forEach((n) => {
    const c = map.get(folderTop(n.folder));
    out.set(n.folder, c);
  });
  // Guardamos también el top-level map para la legend.
  out._topMap = map;
  return out;
}

function renderGraphLegend(folderColors) {
  const el = document.getElementById("graph-legend");
  if (!el) return;
  const topMap = folderColors._topMap;
  if (!topMap) { el.innerHTML = ""; return; }
  // Top 8 carpetas para no saturar la legend.
  const items = [...topMap.entries()].slice(0, 8);
  el.innerHTML = items.map(([folder, color]) => `
    <span><span class="graph-legend-dot" style="background:${color}"></span>${escapeHtml(folder)}</span>
  `).join("");
}

function makeDrag(sim) {
  function dragstarted(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) sim.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

// ── helpers ─────────────────────────────────────────────────────────────
function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ── Boot ────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  // Re-fit the graph on resize (debounced).
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (!state.payload) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => repaintGraphColors(), 220);
  });
  fetchAndRender();
});
