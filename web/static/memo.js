// /memo dashboard — vista read-only de las memorias del MCP `memo`.
//
// Fetch sequence:
//   1. GET /api/memo?limit=50[&type=<t>]  → stats + recent + tags
//   2. Click en una fila → GET /api/memo/note?id=<id>  → body markdown
//
// Sin frameworks. Reusa el look del resto del web UI (mismo color
// scheme + variables CSS).

const $ = (sel) => document.querySelector(sel);

const STATE = {
  type: null,
  limit: 50,
  selectedId: null,
};

const TYPE_ORDER = ["bug", "fact", "decision", "note", "preference"];

function escapeHTML(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function fetchSnapshot() {
  const params = new URLSearchParams({ limit: STATE.limit });
  if (STATE.type) params.set("type", STATE.type);
  const r = await fetch(`/api/memo?${params}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchNote(id) {
  const r = await fetch(`/api/memo/note?id=${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function renderStats(s) {
  const a = s.activity || {};
  const total = s.totals?.all ?? 0;
  $("#stats-grid").innerHTML = `
    <div class="stat-card">
      <div class="label">Total memorias</div>
      <div class="value">${total}</div>
      <div class="sub">${s.totals.by_type.length} tipos</div>
    </div>
    <div class="stat-card">
      <div class="label">Guardadas hoy</div>
      <div class="value">${a.saved_today ?? 0}</div>
      <div class="sub">últimas 24h</div>
    </div>
    <div class="stat-card">
      <div class="label">Últimos 7 días</div>
      <div class="value">${a.saved_7d ?? 0}</div>
      <div class="sub">saves</div>
    </div>
    <div class="stat-card">
      <div class="label">Últimos 30 días</div>
      <div class="value">${a.saved_30d ?? 0}</div>
      <div class="sub">saves</div>
    </div>
    <div class="stat-card">
      <div class="label">Eventos totales</div>
      <div class="value">${a.events_total ?? 0}</div>
      <div class="sub">save/update/delete</div>
    </div>
  `;

  $("#paths-hint").textContent = `${s.memo_dir}`;
}

function renderTypes(s) {
  const byType = new Map((s.totals.by_type || []).map((r) => [r.type, r.count]));
  const allCount = s.totals.all || 0;

  const types = ["all", ...TYPE_ORDER.filter((t) => byType.has(t))];
  // tipos no esperados (raro) van al final
  for (const r of s.totals.by_type) {
    if (!types.includes(r.type)) types.push(r.type);
  }

  const html = types
    .map((t) => {
      const active = (STATE.type === null && t === "all") || STATE.type === t;
      const count = t === "all" ? allCount : byType.get(t) ?? 0;
      return `<span class="type-pill ${active ? "active" : ""}" data-type="${t}">
        ${escapeHTML(t)} <span class="count">${count}</span>
      </span>`;
    })
    .join("");

  $("#types-row").innerHTML = html;

  for (const pill of document.querySelectorAll(".type-pill")) {
    pill.addEventListener("click", () => {
      const t = pill.dataset.type;
      STATE.type = t === "all" ? null : t;
      refresh();
    });
  }
}

function renderRecent(s) {
  const rows = s.recent || [];
  $("#recent-meta").textContent = `${rows.length} mostradas`;

  if (rows.length === 0) {
    $("#memos-tbody").innerHTML = `<tr><td colspan="4" class="loading">Sin resultados.</td></tr>`;
    return;
  }

  $("#memos-tbody").innerHTML = rows
    .map((m) => {
      const tags = (m.tags || []).slice(0, 4).map((t) => `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
      return `
        <tr data-id="${escapeHTML(m.id)}" class="${m.id === STATE.selectedId ? "selected" : ""}">
          <td><span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span></td>
          <td>${escapeHTML(m.title)}</td>
          <td>${tags}</td>
          <td class="ago" title="${escapeHTML(m.updated || "")}">${escapeHTML(m.ago)}</td>
        </tr>`;
    })
    .join("");

  for (const tr of document.querySelectorAll("#memos-tbody tr[data-id]")) {
    tr.addEventListener("click", () => selectMemo(tr.dataset.id));
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

async function selectMemo(id) {
  STATE.selectedId = id;
  for (const tr of document.querySelectorAll("#memos-tbody tr")) {
    tr.classList.toggle("selected", tr.dataset.id === id);
  }
  $("#detail").innerHTML = `<div class="loading">Cargando…</div>`;
  $("#detail-id").textContent = id.slice(0, 8);

  try {
    const d = await fetchNote(id);
    if (!d.ok) {
      $("#detail").innerHTML = `<div class="empty">Error: ${escapeHTML(d.error || "desconocido")}</div>`;
      return;
    }
    const m = d.memo;
    const tags = (m.tags || []).map((t) => `<span class="tag-chip">${escapeHTML(t)}</span>`).join("");
    $("#detail").innerHTML = `
      <h2>${escapeHTML(m.title)}</h2>
      <div class="meta-row">
        <span class="type-tag" data-type="${escapeHTML(m.type)}">${escapeHTML(m.type)}</span>
        <span>·</span>
        <span title="${escapeHTML(m.updated || "")}">${escapeHTML(m.ago)} atrás</span>
        ${tags ? `<span>·</span> ${tags}` : ""}
      </div>
      <div class="path">${escapeHTML(m.path || "")}</div>
      <div class="body">${escapeHTML(m.body || "")}</div>
    `;
  } catch (e) {
    $("#detail").innerHTML = `<div class="empty">Error: ${escapeHTML(e.message)}</div>`;
  }
}

function renderError(msg) {
  $("#error-banner").innerHTML = `<div class="banner-error">${escapeHTML(msg)}</div>`;
}

async function refresh() {
  try {
    const s = await fetchSnapshot();
    if (!s.ok) {
      renderError(s.error || "snapshot failed");
    } else {
      $("#error-banner").innerHTML = "";
    }
    renderStats(s);
    renderTypes(s);
    renderRecent(s);
    renderTags(s);

    // Si había uno seleccionado y sigue en la lista, re-resaltar.
    if (STATE.selectedId) {
      const tr = document.querySelector(`#memos-tbody tr[data-id="${CSS.escape(STATE.selectedId)}"]`);
      if (tr) tr.classList.add("selected");
    }
  } catch (e) {
    renderError(`fetch /api/memo falló: ${e.message}`);
  }
}

refresh();
