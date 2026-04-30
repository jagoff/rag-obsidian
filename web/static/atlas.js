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
  // Toggle aA — labels persisten entre visitas via localStorage. Default
  // off porque con 500 nodos la pantalla queda cargada; el user los prende
  // cuando quiere leer los nombres.
  showLabels: (() => {
    try { return localStorage.getItem("atlas-show-labels") === "1"; }
    catch (e) { return false; }
  })(),
  selectedNode: null,       // referencia al objeto nodo seleccionado
  selectedNeighborIds: null,// Set<id> de vecinos 1-hop del seleccionado
  selectedLinks: null,      // Set<link> que tocan al seleccionado
  searchHits: null,         // Set<id> matches del input de búsqueda
  entityMatches: null,      // Set<id> matches del entity filter (heurística por label)
  // Layout mode — "structural" (default) usa wikilinks vía force-sim;
  // "semantic" pide al backend `/api/atlas/semantic-layout` que devuelve
  // {path: [x,y,z]} de PCA(embeddings) y los aplicamos como fx/fy/fz.
  // Persistimos a localStorage para que el toggle sobreviva reloads.
  layoutMode: (() => {
    try {
      const m = localStorage.getItem("atlas-layout-mode");
      return m === "semantic" ? "semantic" : "structural";
    } catch (e) { return "structural"; }
  })(),
  semanticLayoutCache: null,  // {path: [x,y,z]} cacheado client-side
  semanticLayoutMeta: null,   // {n_notes, explained_variance_ratio, missing}
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

// Detectar mobile para cap del graph_top_notes (perf en iPhone con 500+ nodos
// + force-sim 3D es pesado). 720px es el breakpoint del rest del proyecto.
// Subimos el cap desktop a 500 para que el "preguntale al vault" pueda
// matchear más fuentes contra el grafo visible — three.js maneja 500
// esferas + 1000 links sin sudar en una Mac de los últimos 5 años.
const IS_MOBILE = window.matchMedia("(max-width: 720px)").matches;
const GRAPH_TOP_NOTES = IS_MOBILE ? 200 : 500;

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

// ── Entities lists con barra de magnitud relativa ───────────────────────
//
// Cada fila: [nombre] [barra horizontal de magnitud relativa al max de
// la columna] [count]. Reemplazó las sparklines Chart.js inline (eran
// 56x18px → tan chicas que la mitad se veían como una raya, la otra
// mitad invisibles, terrible UX). La barra es siempre legible y deja
// comparar magnitudes entre entidades del mismo tipo de un vistazo.
//
// Magnitud: `mention_count / max_in_column`. El max sale del primer
// item de la lista (que ya viene ordenada DESC por el backend).
function renderEntities(byType) {
  // Liberamos sparkCharts viejas si todavía existen (compatibilidad
  // con la versión anterior que las usaba).
  state.sparkCharts.forEach((c) => { try { c.destroy(); } catch (e) {} });
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
    // Max de la columna para escalar la barra.
    const maxMentions = list.reduce(
      (m, e) => Math.max(m, e.mention_count || 0),
      1,
    );
    const baseColor = TYPE_COLORS[type];

    ul.innerHTML = "";
    list.slice(0, 15).forEach((e) => {
      const li = document.createElement("li");
      li.className = "ent-row";
      li.dataset.entityId = String(e.id);
      li.dataset.entityName = e.name;
      li.dataset.type = type;
      li.title = `${e.name} — ${TYPE_LABELS[type]} · ${e.mention_count} menciones totales · ${e.recent_mentions} en últimos ${state.windowDays}d${e.aliases?.length ? ` · alias: ${e.aliases.join(", ")}` : ""}\n\nClick para filtrar el grafo a notas que la mencionan.`;

      const pct = Math.max(2, Math.min(100,
        ((e.mention_count || 0) / maxMentions) * 100,
      ));
      // Para la entidad #1 de la columna usamos color sólido. Para el
      // resto bajamos el opacity proporcionalmente al rank — visualmente
      // se nota la jerarquía.
      const fillColor = baseColor;
      const fillOpacity = 0.55 + 0.45 * (pct / 100);

      li.innerHTML = `
        <span class="ent-name">${escapeHtml(e.name)}</span>
        <span class="ent-bar"><span style="width:${pct.toFixed(1)}%;background:${fillColor};opacity:${fillOpacity.toFixed(2)};"></span></span>
        <span class="ent-count">${formatCount(e.mention_count)}</span>
      `;
      li.addEventListener("click", () => {
        if (state.entityFilter && state.entityFilter.id === e.id) {
          clearEntityFilter();
        } else {
          applyEntityFilter({ id: e.id, name: e.name, type });
        }
      });
      ul.appendChild(li);
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
async function renderGraph(graph) {
  const card = document.getElementById("graph-card");
  const containerEl = document.getElementById("graph-3d");
  const stats = document.getElementById("graph-stats");
  if (!card || !containerEl) return;

  // Esperar hasta 3s a que ForceGraph3D / THREE estén disponibles. Defensivo
  // contra: red lenta, CDN tardando en responder, cache stale del SW que
  // sirvió HTML viejo con orden de scripts diferente, etc. Reintenta cada
  // 100ms (30 intentos = 3s total).
  const libsReady = async () => {
    for (let i = 0; i < 30; i++) {
      if (typeof ForceGraph3D === "function" && typeof THREE !== "undefined") return true;
      await new Promise((r) => setTimeout(r, 100));
    }
    return false;
  };
  if (!(await libsReady())) {
    // Fallback con botón para forzar reload sin caché. Diagnóstico fino:
    // distinguimos si falló three.js o 3d-force-graph para que en consola
    // quede claro qué bundle no llegó.
    const missing = [];
    if (typeof THREE === "undefined") missing.push("three.js");
    if (typeof ForceGraph3D !== "function") missing.push("3d-force-graph");
    const what = missing.join(" + ");
    console.error("[atlas-3d] motor 3D no cargó: faltan globals →", what);
    containerEl.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:var(--text-dim); font-size:12px; text-align:center; padding:20px; gap:14px;">
        <div>
          No se pudo cargar el motor 3D (${what}).<br>
          Probablemente el browser sirvió una versión cacheada vieja del PWA.
        </div>
        <button type="button" id="atlas-reload-nocache" class="btn-icon" style="padding:6px 14px; font-size:12px; cursor:pointer;">
          Recargar sin caché
        </button>
        <div style="color:var(--text-faint); font-size:10px; max-width:420px;">
          También podés DevTools → Application → Service Workers → Unregister, o cerrar todas las tabs de localhost y abrir una nueva.
        </div>
      </div>`;
    document.getElementById("atlas-reload-nocache")?.addEventListener("click", async () => {
      // Unregister + clear caches + reload — equivalente al hard-reload.
      try {
        if (navigator.serviceWorker) {
          const regs = await navigator.serviceWorker.getRegistrations();
          for (const r of regs) await r.unregister();
        }
        if (window.caches) {
          const keys = await caches.keys();
          for (const k of keys) await caches.delete(k);
        }
      } catch (e) { console.warn("[atlas-3d] cleanup falló", e); }
      location.reload();
    });
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

  // Trackball controls: drag rota libremente en los 3 ejes (X, Y, Z),
  // a diferencia de "orbit" que solo deja 2 ejes. Más freedom = se siente
  // 3D de verdad cuando manipulás la cámara.
  const Graph = ForceGraph3D({ controlType: "trackball" })
    (containerEl)
    .backgroundColor(bg)
    .graphData({ nodes, links })
    .nodeId("id")
    .nodeVal(nodeVal)
    .nodeRelSize(7)
    .nodeOpacity(1.0)
    .nodeLabel((n) => `
      <div style="background:#222228;border:1px solid #3e3e46;border-radius:6px;padding:6px 10px;font-size:11px;color:#ececed;font-family:'SF Mono',monospace;">
        <div><strong>${escapeHtml(n.label || "(sin título)")}</strong></div>
        <div style="color:#7a7a82;margin-top:2px;">${escapeHtml(n.folder || "—")}</div>
        <div style="color:#7a7a82;">${n.degree} conexión${n.degree === 1 ? "" : "es"} · ${n.n_chunks} chunk${n.n_chunks === 1 ? "" : "s"}</div>
      </div>`)
    // Reemplazamos la mesh default (MeshLambertMaterial) por una mesh
    // BasicMaterial sin lighting — así los colores de carpeta no se
    // lavan por la luz ambient/directional. Los re-coloreamos cheap
    // mutando `node._material.color` desde refreshGraphVisuals().
    .nodeThreeObject(buildNodeObject)
    .nodeThreeObjectExtend(false)
    .linkColor(graphLinkColor)
    .linkWidth((l) => Math.max(0.4, Math.min(2.0, Math.sqrt(l.weight || 1) * 0.6)))
    .linkOpacity(0.4)
    .linkDirectionalParticles(0)   // partículas las prendemos solo en links highlighted
    .linkDirectionalParticleWidth(1.4)
    .linkDirectionalParticleSpeed(0.006)
    .onNodeClick((node, event) => {
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
      // Abrir la nota en Obsidian. Cmd/Ctrl+click NO abre (fallback para
      // cuando el user solo quiere explorar sin que la app se le abra
      // sola). Si el server no nos pasó vault_uri (vault_path no
      // configurado), no hacemos nada.
      const skipOpen = event && (event.metaKey || event.ctrlKey);
      if (!skipOpen && node.vault_uri) {
        openInObsidian(node.vault_uri);
      }
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

  // ZoomToFit + ángulo inicial 3D + auto-rotate. Después del zoomToFit
  // (que apunta de frente al centro, vista plana), inclinamos la cámara
  // hacia arriba y al costado para que el primer paint MUESTRE perspectiva
  // 3D. Cuando termina la transición (~1.8s), arrancamos el auto-rotate
  // — el grafo orbita lento alrededor del eje Y como un planeta.
  setTimeout(() => {
    Graph.zoomToFit(900, 60);
    setTimeout(() => {
      const cam = Graph.cameraPosition();
      const dist = Math.hypot(cam.x || 0, cam.y || 0, cam.z || 0) || 250;
      Graph.cameraPosition(
        { x: dist * 0.55, y: dist * 0.45, z: dist * 0.7 },
        { x: 0, y: 0, z: 0 },
        1800
      );
      // Esperamos a que termine la animación de la cámara antes de
      // empezar la órbita, si no la pelea con el cameraPosition
      // intermedio del flight.
      setTimeout(() => startAutoRotate(), 2000);
    }, 1100);
  }, 1500);

  // Pausar auto-rotate cuando el user interactúa con el canvas; reanudar
  // después de 8s de inactividad. Listeners pasivos para no bloquear scroll.
  const onUserInteraction = () => {
    pauseAutoRotate();
    scheduleResumeAutoRotate(8000);
  };
  containerEl.addEventListener("mousedown", onUserInteraction, { passive: true });
  containerEl.addEventListener("wheel", onUserInteraction, { passive: true });
  containerEl.addEventListener("touchstart", onUserInteraction, { passive: true });
}

// ── Auto-rotate de la cámara: physics-feel sin física real ─────────────
//
// Calculamos un orbit lento alrededor del eje Y manteniendo la altura y
// el radio de la cámara constantes. requestAnimationFrame loop con dt
// real para que la velocidad sea independiente del framerate. Pausa
// cooperativa via state._autoRotatePaused — sale del loop cuando la
// flag se prende, no hace polling.
//
// Por qué no usar `controls.autoRotate` de three.js: TrackballControls no
// tiene esa property (solo OrbitControls). Implementarlo a mano nos
// da control fino sobre la velocidad y la curva, además de poder
// integrarlo con la pausa-en-interacción sin pelearse con los controls.

function startAutoRotate() {
  if (!state.graph3d) return;
  if (state._autoRotateRaf) return;  // ya corriendo

  state._autoRotatePaused = false;
  let lastTs = null;
  // Velocidad: 1 revolución cada ~60s (rotSpeed * 60_000ms ≈ 2π).
  // Lento como un planeta — el ojo nota el 3D sin marearse.
  const rotSpeed = (2 * Math.PI) / 60000;

  const tick = (ts) => {
    if (state._autoRotatePaused || !state.graph3d) {
      state._autoRotateRaf = null;
      return;
    }
    if (lastTs === null) lastTs = ts;
    const dt = ts - lastTs;
    lastTs = ts;

    const cam = state.graph3d.cameraPosition();
    const radius = Math.hypot(cam.x, cam.z);
    const angle = Math.atan2(cam.z, cam.x) + dt * rotSpeed;
    const newX = radius * Math.cos(angle);
    const newZ = radius * Math.sin(angle);
    state.graph3d.cameraPosition(
      { x: newX, y: cam.y, z: newZ },
      undefined,
      0
    );

    state._autoRotateRaf = requestAnimationFrame(tick);
  };
  state._autoRotateRaf = requestAnimationFrame(tick);
}

function pauseAutoRotate() {
  state._autoRotatePaused = true;
  if (state._autoRotateRaf) {
    cancelAnimationFrame(state._autoRotateRaf);
    state._autoRotateRaf = null;
  }
}

function scheduleResumeAutoRotate(delay) {
  if (state._autoRotateResumeTimer) {
    clearTimeout(state._autoRotateResumeTimer);
  }
  state._autoRotateResumeTimer = setTimeout(() => {
    state._autoRotateResumeTimer = null;
    // No reanudar si el user prendió pausa permanente o si hay una
    // respuesta del ask-vault streameando (no queremos marear).
    if (state._autoRotateUserPaused) return;
    if (state._askInFlight) {
      // Re-checkear cuando termine el ask.
      scheduleResumeAutoRotate(2000);
      return;
    }
    startAutoRotate();
  }, delay || 8000);
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

// Construye la mesh 3D de un nodo: SphereGeometry + MeshBasicMaterial
// (sin lighting, color puro). Guardamos referencia al material en
// `node._material` para poder mutarle el color barato sin recrear la
// geometry en cada cambio de estado (highlight, dim, selección, etc.).
//
// El SpriteText con el nombre se crea SIEMPRE como child de la mesh y
// solo se prende/apaga vía `sprite.visible`. Así el toggle aA es
// instantáneo (no recrea geometries) y los labels se mueven naturalmente
// con la esfera durante el force-sim.
function buildNodeObject(node) {
  const radius = nodeRadiusFor(node);
  const geo = new THREE.SphereGeometry(radius, 14, 14);
  const mat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(graphNodeColor(node)),
    transparent: false,
  });
  const sphere = new THREE.Mesh(geo, mat);
  node._material = mat;            // ref para refreshGraphVisuals()
  node._radius = radius;            // ref para position-y del label

  if (typeof SpriteText === "function") {
    const sprite = new SpriteText(node.label || "");
    sprite.material.depthWrite = false;
    sprite.color = "#ececed";
    sprite.textHeight = 3.2;
    sprite.padding = 1;
    sprite.position.set(0, radius + 4, 0);
    sprite.visible = !!state.showLabels;
    sphere.add(sprite);
    node._labelSprite = sprite;     // ref para applyLabelMode() instantáneo
  }
  return sphere;
}

// Forzar re-paint visual: mutamos el color de cada material directamente
// (cheap — no recrea geometries) + re-set del accessor de links + toggle
// de partículas direccionales en el subgrafo seleccionado.
function refreshGraphVisuals() {
  if (!state.graph3d) return;
  // Node colors: mutación in-place del material (NO recreates geometry).
  if (state.graphNodes) {
    for (const n of state.graphNodes) {
      if (n._material) {
        n._material.color.set(graphNodeColor(n));
      }
    }
  }
  // Link colors: re-trigger del accessor (la lib evalúa cada frame).
  state.graph3d.linkColor(graphLinkColor);
  // Highlight: partículas direccionales solo en los links del subgrafo
  // seleccionado para llamar la atención sin saturar el resto.
  if (state.selectedNode) {
    state.graph3d.linkDirectionalParticles((l) => state.selectedLinks?.has(l) ? 3 : 0);
  } else {
    state.graph3d.linkDirectionalParticles(0);
  }
}

// Toggle aA: prende/apaga los SpriteText labels. Como cada nodo guarda
// su sprite en `n._labelSprite`, solo flip de `.visible` — instantáneo,
// no recrea geometries. También persistimos al localStorage para que el
// estado dure entre visitas.
// `Graph` queda como argumento por compatibilidad pero ya no se usa.
function applyLabelMode(_Graph) {
  if (!state.graphNodes) return;
  for (const n of state.graphNodes) {
    if (n._labelSprite) n._labelSprite.visible = !!state.showLabels;
  }
  // Persistir preferencia.
  try {
    localStorage.setItem("atlas-show-labels", state.showLabels ? "1" : "0");
  } catch (e) {}
  // Visual feedback en el botón.
  const btn = document.getElementById("graph-toggle-labels");
  if (btn) {
    btn.setAttribute("aria-pressed", state.showLabels ? "true" : "false");
    btn.classList.toggle("active", state.showLabels);
    btn.title = state.showLabels ? "Ocultar nombres de notas" : "Mostrar nombres de notas";
  }
}

function nodeRadiusFor(node) {
  // Aprox del radio que 3d-force-graph dibuja con .nodeRelSize(7) y
  // nodeVal(degree). cbrt porque 3d-force-graph usa volumen ∝ val.
  const v = Math.max(0.5, Math.min(20, node.degree || 1));
  return Math.cbrt(v) * 7 * 0.5;
}

// ── Click en nodo → abrir nota en Obsidian ──────────────────────────────
//
// `vault_uri` viene en cada nodo del payload de /api/atlas (ej.
// "obsidian://open?vault=Notes&file=01-Projects/X.md"). Lo disparamos
// via `<a>.click()` en vez de `window.open`/`location.href` porque:
//   - Los browsers tratan el click programático en un anchor con href
//     custom-protocol como "user-initiated" → no triggea popup blocker.
//   - location.href disrumpe la página actual si el SO no maneja el
//     protocol (raro, pero seguro defensivo).
function openInObsidian(uri) {
  if (!uri || typeof uri !== "string") return;
  if (!uri.startsWith("obsidian://")) return;
  try {
    const a = document.createElement("a");
    a.href = uri;
    a.rel = "noopener noreferrer";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => a.remove(), 100);
  } catch (e) {
    // Fallback: navegación directa. El browser puede mostrar prompt
    // "esta página intenta abrir Obsidian" la primera vez.
    try { window.location.href = uri; } catch (e2) {}
  }
}

// ── Layout modes: estructural (wikilinks) ↔ semántico (PCA embeddings) ──
//
// Estructural: usa los wikilinks como atracciones del force-sim → notas
// linkeadas se acercan, otras se alejan. Es la vista "cómo vos linkeaste".
//
// Semántico: pide a `/api/atlas/semantic-layout` el resultado del PCA
// 1024d→3D sobre los embeddings bge-m3 mean-pool por nota. Aplica las
// coords como `fx/fy/fz` (posiciones fijas en force-sim → no se mueven).
// El force-sim sigue corriendo solo para colisión + leve perturbación,
// pero la posición principal queda anchored.

async function setLayoutMode(mode) {
  const newMode = mode === "semantic" ? "semantic" : "structural";
  if (state.layoutMode === newMode) {
    // Asegurarnos de que los botones reflejen el estado actual aunque
    // sea un no-op (ej. al boot).
    _refreshModeButtons();
    return;
  }
  state.layoutMode = newMode;
  try { localStorage.setItem("atlas-layout-mode", newMode); } catch (e) {}
  _refreshModeButtons();

  if (newMode === "semantic") {
    await applySemanticLayout();
  } else {
    applyStructuralLayout();
  }
}

function _refreshModeButtons() {
  const sBtn = document.getElementById("graph-mode-structural");
  const mBtn = document.getElementById("graph-mode-semantic");
  if (sBtn) {
    sBtn.classList.toggle("active", state.layoutMode === "structural");
    sBtn.setAttribute("aria-selected", state.layoutMode === "structural" ? "true" : "false");
  }
  if (mBtn) {
    mBtn.classList.toggle("active", state.layoutMode === "semantic");
    mBtn.setAttribute("aria-selected", state.layoutMode === "semantic" ? "true" : "false");
  }
}

async function applySemanticLayout() {
  if (!state.graph3d || !state.graphNodes || !state.graphNodes.length) return;
  const stats = document.getElementById("graph-stats");

  // Cache hit del cliente: si ya pedimos este conjunto de paths, reusamos.
  const paths = state.graphNodes.map((n) => n.id);
  const pathsKey = paths.slice().sort().join("|");
  let layout = null;
  let meta = null;
  if (state.semanticLayoutCache && state._semanticLayoutPathsKey === pathsKey) {
    layout = state.semanticLayoutCache;
    meta = state.semanticLayoutMeta;
  } else {
    if (stats) stats.textContent = "calculando layout semántico…";
    try {
      const r = await fetch("/api/atlas/semantic-layout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      layout = data.layout || {};
      meta = {
        n_notes: data.n_notes || 0,
        explained_variance_ratio: data.explained_variance_ratio || [],
        missing: data.missing || [],
      };
      state.semanticLayoutCache = layout;
      state.semanticLayoutMeta = meta;
      state._semanticLayoutPathsKey = pathsKey;
    } catch (e) {
      console.error("[atlas-semantic] fetch failed", e);
      if (stats) stats.textContent = "no se pudo calcular layout semántico — quedó estructural";
      // Fallback: revert a estructural sin rebote infinito.
      state.layoutMode = "structural";
      try { localStorage.setItem("atlas-layout-mode", "structural"); } catch (e2) {}
      _refreshModeButtons();
      return;
    }
  }

  // Aplicar coords como fx/fy/fz para que el force-sim los respete.
  // Cada nodo que tiene entrada en `layout` queda anchored ahí; los que
  // NO tienen (caso raro: nota nueva sin embedding aún) quedan libres.
  let pinned = 0;
  for (const n of state.graphNodes) {
    const xyz = layout[n.id];
    if (xyz && xyz.length === 3) {
      n.fx = xyz[0];
      n.fy = xyz[1];
      n.fz = xyz[2];
      pinned++;
    } else {
      n.fx = n.fy = n.fz = undefined;
    }
  }

  // Re-heat la simulación para que los nodos se muevan a sus nuevas
  // posiciones con animación suave (no salto duro).
  try { state.graph3d.d3ReheatSimulation(); } catch (e) {}

  // Actualizar stats del grafo. Mostramos el % de varianza explicada por
  // las 3 PCs — > 30% típicamente significa que el layout es coherente.
  if (stats) {
    const evr = (meta?.explained_variance_ratio || []).slice(0, 3);
    const totalVar = evr.reduce((s, v) => s + v, 0);
    const pct = (totalVar * 100).toFixed(0);
    stats.textContent = `🧠 semántico · ${pinned} notas posicionadas por similitud · ${pct}% varianza explicada`;
  }

  // Re-fit de la cámara después de un delay para que la simulación se
  // estabilice antes del zoomToFit.
  setTimeout(() => {
    if (state.graph3d) state.graph3d.zoomToFit(900, 80);
  }, 800);
}

function applyStructuralLayout() {
  if (!state.graph3d || !state.graphNodes) return;
  // Liberar todos los pins → el force-sim vuelve a posicionar por links.
  for (const n of state.graphNodes) {
    n.fx = n.fy = n.fz = undefined;
  }
  try { state.graph3d.d3ReheatSimulation(); } catch (e) {}
  const stats = document.getElementById("graph-stats");
  if (stats && state.payload) {
    // Restaurar el texto original del modo estructural.
    const g = state.payload.graph || {};
    const n = (g.nodes || []).length;
    const l = (g.links || []).length;
    const totalNotes = g.total_notes || n;
    const totalEdges = g.total_edges || l;
    stats.textContent = g.truncated
      ? `${n} de ${totalNotes} notas · ${l} de ${totalEdges} conexiones (top por degree)`
      : `${n} notas · ${l} conexiones`;
  }
  setTimeout(() => {
    if (state.graph3d) state.graph3d.zoomToFit(900, 60);
  }, 800);
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

// ── Ask-vault: chat conversacional con highlight 3D ─────────────────────
//
// Flow:
//   1. user tipea pregunta + Enter / click → askVault(q)
//   2. POST /api/chat (mismo endpoint del chat principal) con SSE
//   3. Cuando llega el evento `sources`, mapeamos los paths a node IDs
//      del grafo y los prendemos en amarillo via state.searchHits.
//   4. Cámara vuela al centroide de los nodos resaltados.
//   5. Tokens del LLM se accumulan en el panel de respuesta arriba.
//
// Limitación: la API de /atlas devuelve los top-N notas (default 500);
// si el chat cita una nota fuera de ese set, no la podemos highlightear
// — se cuenta en el footer pero no brilla.

function _askSessionId() {
  if (!state._askSessionId) {
    state._askSessionId = "atlas-" + Date.now().toString(36) +
      "-" + Math.random().toString(36).slice(2, 8);
  }
  return state._askSessionId;
}

async function askVault(question) {
  const q = (question || "").trim();
  if (!q) return;
  if (state._askInFlight) return;

  const panel = document.getElementById("ask-vault-answer");
  const qEl = document.getElementById("ask-vault-question");
  const tEl = document.getElementById("ask-vault-text");
  const stEl = document.getElementById("ask-vault-status");
  const btn = document.getElementById("ask-vault-btn");
  const btnText = btn?.querySelector(".ask-vault-btn-text");
  const btnSpin = btn?.querySelector(".ask-vault-btn-spinner");

  state._askInFlight = true;
  state._askAnswer = "";
  state._askSourcePaths = [];
  if (qEl) qEl.textContent = q;
  if (tEl) tEl.innerHTML = '<span class="cursor"></span>';
  if (stEl) stEl.textContent = "buscando en el vault…";
  if (panel) panel.hidden = false;
  if (btn) btn.disabled = true;
  if (btnText) btnText.hidden = true;
  if (btnSpin) btnSpin.hidden = false;

  // Pausamos el auto-rotate mientras hay una respuesta en streaming —
  // la cámara va a volar al cluster de fuentes y no queremos que la
  // órbita la pelée.
  pauseAutoRotate();

  // Limpiar cualquier filtro o selección previa para que el highlight
  // del ask sea claro.
  state.selectedNode = null;
  state.selectedNeighborIds = null;
  state.selectedLinks = null;
  state.entityFilter = null;
  state.entityMatches = null;

  let aborted = false;
  try {
    const t0 = performance.now();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        session_id: _askSessionId(),
        vault_scope: null,
        mode: "default",
      }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        _handleAskEvent(raw);
      }
    }
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    const matched = state._askMatchedCount || 0;
    const total = state._askSourcePaths.length;
    const offGraph = total - matched;
    let footer = `${dt}s · ${total} fuente${total === 1 ? "" : "s"}`;
    if (matched > 0) footer += ` · ${matched} en el grafo`;
    if (offGraph > 0) footer += ` · ${offGraph} fuera del top visible`;
    if (stEl) stEl.textContent = footer;
  } catch (e) {
    aborted = true;
    if (tEl) tEl.textContent = "Error: " + (e.message || e);
    if (stEl) stEl.textContent = "falló — revisá la consola";
    console.error("[ask-vault]", e);
  } finally {
    state._askInFlight = false;
    if (btn) btn.disabled = false;
    if (btnText) btnText.hidden = false;
    if (btnSpin) btnSpin.hidden = true;
    if (!aborted && tEl) {
      const cur = tEl.querySelector(".cursor");
      if (cur) cur.remove();
    }
    // Reanudar auto-rotate después de 12s — le da tiempo al user a leer
    // la respuesta antes de que la cámara empiece a moverse de nuevo.
    scheduleResumeAutoRotate(12000);
  }
}

function _handleAskEvent(raw) {
  const lines = raw.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return;
  let parsed;
  try { parsed = JSON.parse(data); } catch { return; }

  if (event === "sources") {
    const items = parsed.items || [];
    const paths = items.map((i) => i && i.file).filter(Boolean);
    state._askSourcePaths = paths;
    _highlightAskNodes(paths);
    const stEl = document.getElementById("ask-vault-status");
    if (stEl) stEl.textContent = `pensando… ${paths.length} fuente${paths.length === 1 ? "" : "s"} encontrada${paths.length === 1 ? "" : "s"}`;
  } else if (event === "token") {
    const delta = parsed.delta || "";
    state._askAnswer += delta;
    const tEl = document.getElementById("ask-vault-text");
    if (tEl) {
      tEl.innerHTML = escapeHtml(state._askAnswer) +
        '<span class="cursor"></span>';
      const panel = document.getElementById("ask-vault-answer");
      if (panel) panel.scrollTop = panel.scrollHeight;
    }
  } else if (event === "status") {
    const stEl = document.getElementById("ask-vault-status");
    if (stEl && parsed.stage) {
      stEl.textContent = `${parsed.stage}…`;
    }
  }
}

function _highlightAskNodes(paths) {
  if (!state.graph3d || !state.graphNodes) return;
  const pathSet = new Set(paths);
  const matches = new Set();
  let cx = 0, cy = 0, cz = 0, count = 0;
  state.graphNodes.forEach((n) => {
    if (pathSet.has(n.id)) {
      matches.add(n.id);
      if (n.x != null && n.y != null && n.z != null) {
        cx += n.x; cy += n.y; cz += n.z; count++;
      }
    }
  });
  state._askMatchedCount = matches.size;
  state.searchHits = matches;       // reuso del visual del search local
  refreshGraphVisuals();

  if (count > 0) {
    cx /= count; cy /= count; cz /= count;
    const distance = matches.size === 1 ? 70 : 140;
    const r = Math.hypot(cx, cy, cz);
    const distRatio = r > 0 ? 1 + distance / r : 1;
    state.graph3d.cameraPosition(
      { x: cx * distRatio, y: cy * distRatio, z: (cz * distRatio) || distance },
      { x: cx, y: cy, z: cz },
      1500
    );
  }
}

function clearAskVault() {
  const panel = document.getElementById("ask-vault-answer");
  const input = document.getElementById("ask-vault-input");
  if (panel) panel.hidden = true;
  if (input) input.value = "";
  state._askAnswer = "";
  state._askSourcePaths = [];
  state._askMatchedCount = 0;
  state.searchHits = null;
  refreshGraphVisuals();
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

  // Toggle estructural ↔ semántico.
  const sBtn = document.getElementById("graph-mode-structural");
  const mBtn = document.getElementById("graph-mode-semantic");
  if (sBtn) sBtn.addEventListener("click", () => setLayoutMode("structural"));
  if (mBtn) mBtn.addEventListener("click", () => setLayoutMode("semantic"));
  // Si la preferencia guardada es semántico, aplicarla apenas el grafo
  // termine de hacer el primer render (el state.graphNodes se llena en
  // renderGraph). Espero 3s para que el initial zoomToFit + animation
  // de cámara terminen antes de re-layout.
  if (state.layoutMode === "semantic") {
    setTimeout(() => {
      if (state.graphNodes && state.graphNodes.length) {
        applySemanticLayout();
      }
      _refreshModeButtons();
    }, 3500);
  } else {
    _refreshModeButtons();
  }

  // ── Ask vault: input + botón + Enter para submit ────────────────────────
  const askInput = document.getElementById("ask-vault-input");
  const askBtn = document.getElementById("ask-vault-btn");
  const askClear = document.getElementById("ask-vault-clear");
  if (askBtn && askInput) {
    askBtn.addEventListener("click", () => askVault(askInput.value));
    askInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askVault(askInput.value);
      }
    });
  }
  if (askClear) askClear.addEventListener("click", () => clearAskVault());

  // Esc cierra: ask-vault → search → entity filter → side-panel.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const askPanel = document.getElementById("ask-vault-answer");
      if (askPanel && !askPanel.hidden) {
        clearAskVault();
        return;
      }
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
