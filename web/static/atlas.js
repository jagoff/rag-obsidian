/* atlas.js — Hidrata /atlas con datos de /api/atlas.
 *
 * Tres bloques principales:
 *   1) KPIs + entidades (Chart.js sparklines)
 *   2) Hot/Stale + co-ocurrencias
 *   3) Force-directed graph 3D (Three.js + 3d-force-graph)
 *
 * El graph usa `3d-force-graph` (Vasco Asturiano) sobre WebGL: nodos como
 * esferas, links como cilindros, drag con orbit-controls (rotar + pan +
 * zoom). Click en un nodo enfoca la cámara y highlightea el subgrafo
 * 1-hop. Search + entity-filter actualizan colores reactivamente vía
 * `Graph.nodeColor(Graph.nodeColor())` (re-trigger del accessor).
 *
 * Las dependencias se cargan UMD en atlas.html como globales:
 *   - THREE        (three@0.158)
 *   - ForceGraph3D (3d-force-graph@1.73)
 *   - SpriteText   (three-spritetext@1.8) — labels que rotan a cámara
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
  // ── 3D Graph state ─────────────────────────────────────────────────────
  graph3d: null,            // ForceGraph3D instance
  graphNodes: [],           // copia del input para filter logic
  graphLinks: [],
  folderColors: null,       // Map<folder, hex> + ._topMap
  showLabels: false,        // toggle aA — default off (visualmente menos cargado en 3D)
  selectedNode: null,       // referencia al objeto nodo seleccionado
  selectedNeighborIds: null,// Set<id> de vecinos 1-hop del seleccionado
  selectedLinks: null,      // Set<link> que tocan al seleccionado
  searchHits: null,         // Set<id> matches del input de búsqueda
  entityMatches: null,      // Set<id> matches del entity filter (heurística por label)
  // Filter state — cuando el user hace click en una entidad del top-list,
  // dimmamos los nodos del grafo cuyo path NO contiene esa entidad. Heurística
  // en el cliente vs hit al backend: si el label de la nota INCLUYE el name
  // de la entity (case-insensitive), match — cubre bien el caso típico del
  // vault del user (`Info - Maria` matches `Maria`).
  entityFilter: null,       // { id, name, type } | null
  searchQuery: "",          // string
  sparkCharts: new Map(),   // entity_id → Chart.js instance
};

// Colores fijos para los estados especiales del graph 3D. Solid (no rgba),
// porque ForceGraph3D no respeta alpha en .nodeColor() — la opacidad sale
// del material three.js, no del color string.
const COLOR_DIM = "#3a3a40";
const COLOR_SELECTED = "#ffd86b";  // amarillo cálido
const COLOR_NEIGHBOR_LINK = "#79c0ff";
// Para los links: tres.js material respeta el color (hex) pero la opacidad
// global la maneja `linkOpacity()`. Si pasamos rgba acá la alpha se multiplica
// y queda invisible — usar hex sólido y dejar que linkOpacity haga su parte.
const COLOR_DEFAULT_LINK = "#a0a0aa";
const COLOR_DIM_LINK = "#3e3e46";

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
  if (state.graph3d) repaintGraphColors();
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

// ── Graph 3D (Three.js + 3d-force-graph) ────────────────────────────────
function renderGraph(graph) {
  const card = document.getElementById("graph-card");
  const containerEl = document.getElementById("graph-3d");
  const stats = document.getElementById("graph-stats");
  if (!card || !containerEl) return;

  // Si la lib UMD no cargó (offline / CDN caída), mostrar fallback.
  // ForceGraph3D bundlea three.js internamente, así que con esto basta.
  if (typeof ForceGraph3D !== "function") {
    containerEl.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-faint); font-size:12px; text-align:center; padding:20px;">
        No se pudo cargar el motor 3D (3d-force-graph).<br>
        Revisá la conexión y recargá la página.
      </div>`;
    return;
  }

  const nodes = (graph.nodes || []).map((n) => ({ ...n }));
  const links = (graph.links || []).map((l) => ({ ...l }));

  // Pre-procesamos vecinos 1-hop para que la lógica de selección sea O(1)
  // por nodo (en vez de scanear `links` cada click).
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  nodes.forEach((n) => { n._neighbors = new Set(); n._linkRefs = []; });
  links.forEach((l) => {
    const sId = (typeof l.source === "object") ? l.source.id : l.source;
    const tId = (typeof l.target === "object") ? l.target.id : l.target;
    const s = nodeById.get(sId);
    const t = nodeById.get(tId);
    if (s) { s._neighbors.add(tId); s._linkRefs.push(l); }
    if (t) { t._neighbors.add(sId); t._linkRefs.push(l); }
  });

  const totalNotes = graph.total_notes || nodes.length;
  const totalEdges = graph.total_edges || links.length;
  if (stats) {
    stats.textContent = graph.truncated
      ? `${nodes.length} de ${totalNotes} notas · ${links.length} de ${totalEdges} conexiones (top por degree)`
      : `${nodes.length} notas · ${links.length} conexiones`;
  }

  if (!nodes.length) {
    containerEl.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-faint); font-size:13px;">
        Sin notas indexadas todavía. Corré <code>rag index</code> para poblar el grafo.
      </div>`;
    return;
  }

  // ─ Folder color palette: reproducible per-folder hue. Igual que Obsidian
  //   le asigna un color por carpeta/grupo si abrís Group Settings.
  const folderColors = buildFolderColorMap(nodes);
  state.folderColors = folderColors;

  // Tear-down previo para evitar leaks si re-renderizamos (theme switch,
  // resize, etc.).
  if (state.graph3d) {
    try { state.graph3d._destructor && state.graph3d._destructor(); } catch (e) {}
    containerEl.innerHTML = "";
  }

  // ─ Tamaño nodo: degree → radio (3..10). Lo usamos como `nodeVal` que
  //   3d-force-graph mapea a sphere-size = sqrt(val) * relSize.
  const nodeVal = (n) => Math.max(0.5, Math.min(20, (n.degree || 1)));

  // ─ Background del canvas según tema (CSS var).
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#1a1a1f";

  const Graph = ForceGraph3D({ controlType: "orbit" })
    (containerEl)
    .backgroundColor(bg)
    .graphData({ nodes, links })
    .nodeId("id")
    .nodeVal(nodeVal)
    .nodeRelSize(7)                // bola más grande para que el color se vea sin lambert washout
    .nodeOpacity(1.0)
    .nodeResolution(16)            // detalle de la esfera; 16 = bola redonda
    .nodeLabel((n) => `
      <div style="background:#222228;border:1px solid #3e3e46;border-radius:6px;padding:6px 10px;font-size:11px;color:#ececed;font-family:'SF Mono',monospace;">
        <div><strong>${escapeHtml(n.label || "(sin título)")}</strong></div>
        <div style="color:#7a7a82;margin-top:2px;">${escapeHtml(n.folder || "—")}</div>
        <div style="color:#7a7a82;">${n.degree} conexión${n.degree === 1 ? "" : "es"} · ${n.n_chunks} chunk${n.n_chunks === 1 ? "" : "s"}</div>
      </div>`)
    .nodeColor(graphNodeColor)
    .linkColor(graphLinkColor)
    .linkWidth((l) => Math.max(0.4, Math.min(2.0, Math.sqrt(l.weight || 1) * 0.6)))
    .linkOpacity(0.4)
    .linkDirectionalParticles(0)   // partículas las prendemos solo en links highlighted
    .linkDirectionalParticleWidth(1.4)
    .linkDirectionalParticleSpeed(0.006)
    .onNodeClick((node) => {
      // Aim camera at clicked node from outside.
      const distance = 90;
      const r = Math.hypot(node.x || 0, node.y || 0, node.z || 0);
      const distRatio = r > 0 ? 1 + distance / r : 1;
      Graph.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: ((node.z || 0) * distRatio) || distance },
        node,
        1200
      );
      selectNode(node);
    })
    .onNodeHover((node) => {
      document.body.style.cursor = node ? "pointer" : "";
    })
    .onBackgroundClick(() => clearSelection());

  // Sprite labels (opcionales) — se prenden con el botón aA.
  applyLabelMode(Graph);

  // Cooling más rápido para no quedar gastando CPU eternamente.
  Graph.d3Force("charge").strength(-110);
  Graph.d3Force("link").distance(36);
  Graph.cooldownTicks(180);

  state.graph3d = Graph;
  state.graphNodes = nodes;
  state.graphLinks = links;

  // ─ Legend ───────────────────────────────────────────────
  renderGraphLegend(folderColors);

  // ─ Controls (zoom in/out/reset/toggle labels) ───────────
  document.getElementById("graph-zoom-in").onclick = () => zoomCamera(0.7);
  document.getElementById("graph-zoom-out").onclick = () => zoomCamera(1.4);
  document.getElementById("graph-reset").onclick = () => Graph.zoomToFit(600, 60);
  document.getElementById("graph-toggle-labels").onclick = () => {
    state.showLabels = !state.showLabels;
    applyLabelMode(state.graph3d);
  };

  // Resize handler propio (3d-force-graph se resizea solo cuando le pasamos
  // width/height, lo hacemos con auto-detect del bounding rect).
  function resize() {
    const r = containerEl.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) {
      Graph.width(r.width).height(r.height);
    }
  }
  resize();
  // ResizeObserver es más confiable que window resize para el caso del card
  // que cambia con el side-panel toggle.
  const ro = new ResizeObserver(resize);
  ro.observe(containerEl);
  state._resizeObs = ro;

  // ZoomToFit cuando la simulación se calmó — encuadra todo el grafo en el
  // primer paint sin que el user tenga que hacerlo a mano.
  setTimeout(() => Graph.zoomToFit(900, 60), 1500);
}

// ─── Color accessors (re-evaluados en cada frame por 3d-force-graph) ────

function graphNodeColor(node) {
  if (!state.folderColors) return "#a0a0a6";
  const baseColor = state.folderColors.get(node.folder) || "#a0a0a6";
  // Selected node — always bright.
  if (state.selectedNode && state.selectedNode.id === node.id) return COLOR_SELECTED;
  // Selected mode: dim non-neighbors.
  if (state.selectedNode && state.selectedNeighborIds && !state.selectedNeighborIds.has(node.id)) {
    return COLOR_DIM;
  }
  // Search-active: yellow on hits, dim on misses.
  if (state.searchHits) {
    return state.searchHits.has(node.id) ? COLOR_SELECTED : COLOR_DIM;
  }
  // Entity-filter: dim non-matches.
  if (state.entityMatches && !state.entityMatches.has(node.id)) return COLOR_DIM;
  return baseColor;
}

function graphLinkColor(link) {
  if (state.selectedLinks && state.selectedLinks.has(link)) return COLOR_NEIGHBOR_LINK;
  if (state.selectedNode || state.searchHits || state.entityMatches) return COLOR_DIM_LINK;
  return COLOR_DEFAULT_LINK;
}

// Forzar re-evaluación de los accessors (los re-set triggea un re-paint).
function refreshGraphVisuals() {
  if (!state.graph3d) return;
  state.graph3d.nodeColor(graphNodeColor).linkColor(graphLinkColor);
  // Highlight: prendemos partículas solo en los links del subgrafo seleccionado.
  if (state.selectedNode) {
    state.graph3d.linkDirectionalParticles((l) => state.selectedLinks?.has(l) ? 3 : 0);
  } else {
    state.graph3d.linkDirectionalParticles(0);
  }
}

function applyLabelMode(Graph) {
  if (!Graph) return;
  if (state.showLabels && typeof SpriteText === "function") {
    Graph.nodeThreeObject((node) => {
      const sprite = new SpriteText(node.label || "");
      sprite.material.depthWrite = false;            // siempre visible aunque se solape
      sprite.color = "#ececed";
      sprite.textHeight = 3.2;
      sprite.padding = 1;
      sprite.position.set(0, nodeRadiusFor(node) + 4, 0);
      return sprite;
    }).nodeThreeObjectExtend(true);
  } else {
    // Quitar labels: forzamos accessor que retorne null + flag de "no extender"
    // para que three-force-graph use el sphere default sin el sprite.
    Graph.nodeThreeObject(() => null).nodeThreeObjectExtend(false);
  }
}

function nodeRadiusFor(node) {
  // Aprox del radio que 3d-force-graph dibuja con .nodeRelSize(4.5) y nodeVal(degree).
  const v = Math.max(0.5, Math.min(20, node.degree || 1));
  return Math.cbrt(v) * 4.5 * 0.5;
}

function zoomCamera(factor) {
  if (!state.graph3d) return;
  const cur = state.graph3d.cameraPosition();
  state.graph3d.cameraPosition(
    { x: cur.x * factor, y: cur.y * factor, z: cur.z * factor },
    undefined,
    300
  );
}

// ─── Selección + clear ────────────────────────────────────────────────

function selectNode(node) {
  if (!state.graph3d) return;
  state.selectedNode = node;
  // Vecinos ya están pre-computados en n._neighbors / n._linkRefs.
  state.selectedNeighborIds = new Set([node.id, ...(node._neighbors || [])]);
  state.selectedLinks = new Set(node._linkRefs || []);
  refreshGraphVisuals();
  openSidePanel(node);
}

function clearSelection() {
  state.selectedNode = null;
  state.selectedNeighborIds = null;
  state.selectedLinks = null;
  closeSidePanel();
  refreshGraphVisuals();
  // Re-aplicar entity filter si está activo (clearSelection no debe
  // limpiar el filter, son cosas independientes).
  if (state.entityFilter) applyEntityFilter(state.entityFilter);
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
  document.getElementById("gp-close").addEventListener("click", () => clearSelection());

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

  document.getElementById("gp-close").addEventListener("click", () => clearSelection());

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
      if (target && state.graph3d) {
        // Mismo flujo que un click en una esfera del 3D.
        const distance = 90;
        const r = Math.hypot(target.x || 0, target.y || 0, target.z || 0);
        const distRatio = r > 0 ? 1 + distance / r : 1;
        state.graph3d.cameraPosition(
          { x: (target.x || 0) * distRatio, y: (target.y || 0) * distRatio, z: ((target.z || 0) * distRatio) || distance },
          target,
          1200
        );
        selectNode(target);
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
  // `Moka - 1 a 1` matches `Moka`, etc.).
  const needle = entity.name.toLowerCase();
  const matchIds = new Set();
  (state.graphNodes || []).forEach((n) => {
    if ((n.label || "").toLowerCase().includes(needle)) matchIds.add(n.id);
  });
  state.entityMatches = matchIds;
  refreshGraphVisuals();
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
  state.entityMatches = null;
  document.querySelectorAll(".ent-row").forEach((row) => row.classList.remove("active"));
  const banner = document.getElementById("graph-filter-banner");
  if (banner) banner.hidden = true;
  const clearBtn = document.getElementById("graph-clear-filter");
  if (clearBtn) clearBtn.hidden = true;
  refreshGraphVisuals();
}

// ── Buscador del grafo — input en el header del card ────────────────────
function applySearch(query) {
  state.searchQuery = (query || "").trim().toLowerCase();
  if (!state.searchQuery) {
    state.searchHits = null;
    refreshGraphVisuals();
    return;
  }
  // Match: substring del label (case-insensitive). Sirve también para folders
  // (ej. buscar "01-Projects" highlightea todas las notas en proyectos).
  const matchIds = new Set();
  (state.graphNodes || []).forEach((n) => {
    const hay = ((n.label || "") + " " + (n.folder || "")).toLowerCase();
    if (hay.includes(state.searchQuery)) matchIds.add(n.id);
  });
  state.searchHits = matchIds;
  refreshGraphVisuals();
  // Centrar la cámara en el primer hit (si hay 1-5 matches — más es ruido).
  if (matchIds.size > 0 && matchIds.size <= 5 && state.graph3d) {
    const firstHit = state.graphNodes.find((n) => matchIds.has(n.id));
    if (firstHit && firstHit.x != null) {
      const distance = 80;
      const r = Math.hypot(firstHit.x || 0, firstHit.y || 0, firstHit.z || 0);
      const distRatio = r > 0 ? 1 + distance / r : 1;
      state.graph3d.cameraPosition(
        { x: (firstHit.x || 0) * distRatio, y: (firstHit.y || 0) * distRatio, z: ((firstHit.z || 0) * distRatio) || distance },
        firstHit,
        900
      );
    }
  }
}

function repaintGraphColors() {
  // En 3D el background del canvas no se actualiza solo cuando cambia el tema
  // — hay que mandárselo de nuevo. Los colores per-nodo salen del accessor
  // así que con un refresh basta.
  if (state.graph3d) {
    const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#1a1a1f";
    state.graph3d.backgroundColor(bg);
    refreshGraphVisuals();
  }
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
        clearSelection();
      }
    }
  });

  fetchAndRender();
});
