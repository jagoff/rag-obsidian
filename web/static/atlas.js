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
  // Filter state — cuando el user hace click en una entidad del top-list,
  // dimmamos los nodos del grafo cuyo path NO contiene esa entidad. Implica
  // un fetch al endpoint /api/atlas/note por nota, pero como ya tenemos las
  // entities por nota en el backend, lo más eficiente es hacer una consulta
  // server-side. Por simplicidad acá lo manejamos con búsqueda fuzzy en el
  // label (proxy razonable cuando el name de la entity matches el name
  // de la nota — ej. "Maria" matches "Info - Maria").
  entityFilter: null,  // { id, name, type } | null
  searchQuery: "",     // string
};

// Detectar mobile para cap del graph_top_notes (perf en iPhone con 250+ nodos
// + force-sim es brutal). 720px es el breakpoint del rest del proyecto.
const IS_MOBILE = window.matchMedia("(max-width: 720px)").matches;
const GRAPH_TOP_NOTES = IS_MOBILE ? 150 : 250;

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
    const r = await fetch(`/api/atlas?window_days=${state.windowDays}&top_entities=50&graph_top_notes=${GRAPH_TOP_NOTES}`);
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
      li.dataset.entityName = e.name;
      li.dataset.type = type;
      li.title = `${e.name} — ${TYPE_LABELS[type]} · ${e.mention_count} menciones totales · ${e.recent_mentions} en últimos ${state.windowDays}d${e.aliases?.length ? ` · alias: ${e.aliases.join(", ")}` : ""}\n\nClick para filtrar el grafo a notas que la mencionan.`;
      li.innerHTML = `
        <span class="ent-name">${escapeHtml(e.name)}</span>
        <canvas class="ent-spark" id="spark-${e.id}"></canvas>
        <span class="ent-count">${formatCount(e.mention_count)}</span>
      `;
      li.addEventListener("click", () => {
        // Toggle: click en la misma entity 2 veces limpia el filtro
        if (state.entityFilter && state.entityFilter.id === e.id) {
          clearEntityFilter();
        } else {
          applyEntityFilter({ id: e.id, name: e.name, type });
        }
      });
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

  // Guardamos refs a las selecciones para que las funciones de filter
  // (search, entity-filter) puedan re-aplicar clases sin re-render.
  state.graphLinks = links;
  state.graphNodes = nodes;
  state.nodeSel = nodeSel;
  state.linkSel = linkSel;
  state.labelSel = labelSel;
  state.zoom = zoom;
  state.svg = svg;

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
    // Side-panel con detalle de la nota.
    openSidePanel(d);
  }
  function clearSelection() {
    state.selectedNode = null;
    nodeSel.classed("dim", false);
    linkSel.classed("dim", false).classed("hl", false);
    labelSel.classed("hidden", false).classed("hl", false);
    closeSidePanel();
    // Re-aplicar entity filter si está activo (clearSelection no debe
    // limpiar el filter, son cosas independientes).
    if (state.entityFilter) applyEntityFilter(state.entityFilter);
  }
}

// ── Side-panel: detalle de una nota seleccionada ────────────────────────
async function openSidePanel(node) {
  const panel = document.getElementById("graph-panel");
  const inner = document.getElementById("gp-inner");
  if (!panel || !inner) return;
  panel.classList.add("open");
  inner.innerHTML = `
    <button class="gp-close" id="gp-close" type="button" aria-label="Cerrar panel">×</button>
    <div class="gp-title">${escapeHtml(node.label || "(sin título)")}</div>
    <div class="gp-folder">${escapeHtml(node.folder || "—")}</div>
    <div class="gp-loading">Cargando detalle…</div>
  `;
  document.getElementById("gp-close").addEventListener("click", () => {
    if (state.nodeSel) state.nodeSel.classed("dim", false);
    if (state.linkSel) state.linkSel.classed("dim", false).classed("hl", false);
    if (state.labelSel) state.labelSel.classed("hidden", false).classed("hl", false);
    state.selectedNode = null;
    closeSidePanel();
  });

  // Fetch detalle.
  let detail;
  try {
    const r = await fetch(`/api/atlas/note?path=${encodeURIComponent(node.id)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    detail = await r.json();
  } catch (e) {
    inner.querySelector(".gp-loading").textContent = "Error al cargar detalle: " + e.message;
    return;
  }

  // Render del panel completo.
  const obsLink = detail.vault_uri
    ? `<a class="gp-action" href="${escapeHtml(detail.vault_uri)}" title="Abrir en Obsidian">📓 abrir en Obsidian</a>`
    : "";

  const previewBlock = detail.preview
    ? `<div class="gp-section">
         <h4>Preview</h4>
         <div class="gp-preview">${escapeHtml(detail.preview)}</div>
       </div>` : "";

  const entitiesBlock = detail.entities?.length
    ? `<div class="gp-section">
         <h4>Entidades mencionadas (${detail.entities.length})</h4>
         <ul class="gp-list">
           ${detail.entities.slice(0, 12).map((e) => `
             <li class="gp-list-row" data-entity-id="${e.id}" data-entity-name="${escapeHtml(e.name)}" data-entity-type="${e.type}" title="${e.mention_count} menciones totales — click para filtrar el grafo">
               <span class="ent-pill" style="background:${TYPE_COLORS[e.type] || "var(--text-faint)"}">${(TYPE_LABELS[e.type] || e.type).slice(0, 3)}</span>
               <span style="flex:1; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(e.name)}</span>
               <span style="color:var(--text-faint); font-size:10px;">${e.chunks_in_note}×</span>
             </li>
           `).join("")}
         </ul>
       </div>` : "";

  const neighborsBlock = detail.neighbors?.length
    ? `<div class="gp-section">
         <h4>Vecinos 1-hop (${detail.neighbors.length})</h4>
         <ul class="gp-list">
           ${detail.neighbors.slice(0, 15).map((n) => `
             <li class="gp-list-row" data-neighbor-path="${escapeHtml(n.path)}" title="${escapeHtml(n.path)}">
               <span class="gp-neighbor-arrow">${n.direction === "out" ? "→" : "←"}</span>
               <span style="flex:1; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(n.label)}</span>
             </li>
           `).join("")}
         </ul>
       </div>` : "";

  inner.innerHTML = `
    <button class="gp-close" id="gp-close" type="button" aria-label="Cerrar panel">×</button>
    <div class="gp-title">${escapeHtml(node.label || "(sin título)")}</div>
    <div class="gp-folder">${escapeHtml(node.folder || "—")} · ${node.degree} conexión${node.degree === 1 ? "" : "es"} · ${node.n_chunks} chunk${node.n_chunks === 1 ? "" : "s"}</div>
    <div class="gp-actions">${obsLink}</div>
    ${previewBlock}
    ${entitiesBlock}
    ${neighborsBlock}
  `;

  document.getElementById("gp-close").addEventListener("click", () => {
    if (state.nodeSel) state.nodeSel.classed("dim", false);
    if (state.linkSel) state.linkSel.classed("dim", false).classed("hl", false);
    if (state.labelSel) state.labelSel.classed("hidden", false).classed("hl", false);
    state.selectedNode = null;
    closeSidePanel();
  });

  // Click en entidad del panel → filter del grafo.
  inner.querySelectorAll(".gp-list-row[data-entity-id]").forEach((row) => {
    row.addEventListener("click", () => {
      const entId = parseInt(row.dataset.entityId, 10);
      const entName = row.dataset.entityName;
      const entType = row.dataset.entityType;
      applyEntityFilter({ id: entId, name: entName, type: entType });
    });
  });

  // Click en vecino → selecciono el vecino (si está en el grafo visible).
  inner.querySelectorAll(".gp-list-row[data-neighbor-path]").forEach((row) => {
    row.addEventListener("click", () => {
      const path = row.dataset.neighborPath;
      if (!state.graphNodes) return;
      const target = state.graphNodes.find((n) => n.id === path);
      if (target && state.nodeSel) {
        // Trigger del mismo handler que un click en el SVG.
        state.nodeSel.filter((n) => n.id === path).dispatch("click");
      }
    });
  });
}

function closeSidePanel() {
  const panel = document.getElementById("graph-panel");
  if (panel) panel.classList.remove("open");
}

// ── Entity filter — click en una entity dimmea las notas que NO la mencionan ──
function applyEntityFilter(entity) {
  state.entityFilter = entity;
  // Heurística sin hit al backend: si el label de la nota INCLUYE el name
  // de la entity (case-insensitive), la consideramos match. Cubre los casos
  // típicos del vault del user (`Info - Maria` matches `Maria`,
  // `Moka - 1 a 1` matches `Moka`, etc.). Para el caso general donde el name
  // de la entity NO está en el label, igual el visual feedback es útil
  // (muestra qué notas TIENEN la entity en el título).
  const needle = entity.name.toLowerCase();
  if (!state.nodeSel || !state.linkSel) return;
  const matchIds = new Set();
  state.graphNodes.forEach((n) => {
    if ((n.label || "").toLowerCase().includes(needle)) matchIds.add(n.id);
  });
  state.nodeSel.classed("dim", (n) => !matchIds.has(n.id));
  state.linkSel.classed("dim", (l) => {
    const sId = (typeof l.source === "object") ? l.source.id : l.source;
    const tId = (typeof l.target === "object") ? l.target.id : l.target;
    return !matchIds.has(sId) && !matchIds.has(tId);
  });
  state.linkSel.classed("hl", false);
  state.labelSel.classed("hidden", (n) => !matchIds.has(n.id));
  // Banner explicativo.
  const banner = document.getElementById("graph-filter-banner");
  if (banner) {
    banner.hidden = false;
    banner.innerHTML = `Filtro: <strong>${escapeHtml(entity.name)}</strong> (${TYPE_LABELS[entity.type] || entity.type}) · ${matchIds.size} nota${matchIds.size === 1 ? "" : "s"} — click la entity de nuevo para limpiar`;
  }
  // Marcar la entity activa visualmente en la lista.
  document.querySelectorAll(".ent-row").forEach((row) => {
    row.classList.toggle("active", parseInt(row.dataset.entityId, 10) === entity.id);
  });
  // Mostrar el botón × del filter.
  const clearBtn = document.getElementById("graph-clear-filter");
  if (clearBtn) clearBtn.hidden = false;
}

function clearEntityFilter() {
  state.entityFilter = null;
  if (state.nodeSel) state.nodeSel.classed("dim", false);
  if (state.linkSel) state.linkSel.classed("dim", false).classed("hl", false);
  if (state.labelSel) state.labelSel.classed("hidden", false);
  document.querySelectorAll(".ent-row").forEach((row) => row.classList.remove("active"));
  const banner = document.getElementById("graph-filter-banner");
  if (banner) banner.hidden = true;
  const clearBtn = document.getElementById("graph-clear-filter");
  if (clearBtn) clearBtn.hidden = true;
  // Re-aplicar selección de nodo si hay una activa.
  if (state.selectedNode && state.nodeSel) {
    state.nodeSel.filter((n) => n.id === state.selectedNode.id).dispatch("click");
  }
}

// ── Buscador del grafo — input en el header del card ────────────────────
function applySearch(query) {
  state.searchQuery = (query || "").trim().toLowerCase();
  if (!state.nodeSel || !state.labelSel) return;
  if (!state.searchQuery) {
    state.nodeSel.classed("search-hit", false);
    return;
  }
  // Match: substring del label (case-insensitive). Sirve también para folders
  // (ej. buscar "01-Projects" highlightea todas las notas en proyectos).
  const matchIds = new Set();
  state.graphNodes.forEach((n) => {
    const hay = ((n.label || "") + " " + (n.folder || "")).toLowerCase();
    if (hay.includes(state.searchQuery)) matchIds.add(n.id);
  });
  state.nodeSel.classed("search-hit", (n) => matchIds.has(n.id));
  // Forzar labels visibles para los hits.
  state.labelSel.classed("hidden", (n) => !matchIds.has(n.id));
  // Centrar el zoom en el primer hit (si hay).
  if (matchIds.size > 0 && matchIds.size <= 5 && state.zoom && state.svg) {
    const firstHit = state.graphNodes.find((n) => matchIds.has(n.id));
    if (firstHit && firstHit.x != null) {
      const rect = document.getElementById("graph-main").getBoundingClientRect();
      const scale = 1.6;
      const tx = rect.width / 2 - firstHit.x * scale;
      const ty = rect.height / 2 - firstHit.y * scale;
      state.svg.transition().duration(420).call(
        state.zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      );
    }
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

  // Buscador del grafo — debounced 120ms para no recalcular mientras escribís.
  const searchInput = document.getElementById("graph-search");
  if (searchInput) {
    let searchTimer = null;
    searchInput.addEventListener("input", (e) => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => applySearch(e.target.value), 120);
    });
  }

  // Botón × para limpiar el entity filter.
  const clearBtn = document.getElementById("graph-clear-filter");
  if (clearBtn) clearBtn.addEventListener("click", () => clearEntityFilter());

  // Esc cierra el side-panel.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (state.searchQuery) {
        if (searchInput) searchInput.value = "";
        applySearch("");
      } else if (state.entityFilter) {
        clearEntityFilter();
      } else if (state.selectedNode) {
        // Trigger clearSelection via background-svg click handler.
        if (state.svg) state.svg.dispatch("click");
      }
    }
  });

  fetchAndRender();
});
