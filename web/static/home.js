// Home — information centralizer. Aggregates every channel:
//   today brief · reminders · mail · gmail · whatsapp · calendar · weather
//   · open loops · contradictions · low-confidence queries · inbox activity
// Data source: GET /api/home (silent-fail per channel).
// Auto-refresh: every 60s unless tab is hidden.

const els = {
  date: document.getElementById("date-label"),
  status: document.getElementById("status-line"),
  urgent: document.getElementById("urgent-wrap"),
  narrative: document.getElementById("narrative-wrap"),
  panels: document.getElementById("panels"),
  error: document.getElementById("error-wrap"),
  btnReload: document.getElementById("btn-reload"),
  btnRegen: document.getElementById("btn-regenerate"),
};

const REFRESH_MS = 60_000;
let refreshTimer = null;

function esc(s) {
  return String(s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
  } catch (_) { return ""; }
}

function fmtDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("es-AR", { day: "numeric", month: "short" });
  } catch (_) { return ""; }
}

// [[Title]] wikilinks → plain text (not navigable from the web)
function stripWikilinks(md) {
  return String(md || "").replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, t, a) => a || t);
}

// ── Panel builder ─────────────────────────────────────────────────────────
function panel(id, title, count, bodyHtml, emptyText = "sin actividad") {
  const countStr = typeof count === "number" ? count : "";
  const body = count === 0 ? `<div class="empty">${esc(emptyText)}</div>` : bodyHtml;
  return `
    <section class="panel" id="panel-${id}">
      <div class="head">
        <h3>${esc(title)}</h3>
        <span class="count">${countStr}</span>
      </div>
      ${body}
    </section>`;
}

// ── Renderers per channel ────────────────────────────────────────────────
function renderReminders(items) {
  if (!items || !items.length) return panel("reminders", "reminders", 0, "");
  const lis = items.slice(0, 8).map((r) => {
    const bucket = r.bucket || "undated";
    const due = r.due_display ? `<span class="meta">${esc(r.due_display)}</span>` : "";
    const list = r.list ? `<span class="meta">${esc(r.list)}</span>` : "";
    return `<li>
      <span class="bucket ${esc(bucket)}">${esc(bucket)}</span>
      <b>${esc(r.name || "")}</b>
      ${due}${list}
    </li>`;
  }).join("");
  return panel("reminders", "reminders", items.length, `<ul>${lis}</ul>`);
}

function renderCalendar(today, tomorrow) {
  const t = today || [];
  const mx = tomorrow || [];
  const total = t.length + mx.length;
  if (total === 0) return panel("calendar", "calendar", 0, "");
  let html = "<ul>";
  if (t.length) {
    html += t.slice(0, 6).map((e) => {
      const when = e.start ? `<span class="meta">${esc(e.start)}${e.end ? " – " + esc(e.end) : ""}</span>` : "";
      return `<li><span class="bucket today">today</span><b>${esc(e.title || "")}</b>${when}</li>`;
    }).join("");
  }
  if (mx.length) {
    html += mx.slice(0, 6).map((e) => {
      const when = e.start ? `<span class="meta">${esc(e.date || "")} ${esc(e.start || "")}</span>` : `<span class="meta">${esc(e.date || "")}</span>`;
      return `<li><span class="bucket tomorrow">ahead</span><b>${esc(e.title || "")}</b>${when}</li>`;
    }).join("");
  }
  html += "</ul>";
  return panel("calendar", "calendar", total, html);
}

function renderGmail(g) {
  if (!g) return panel("gmail", "gmail", 0, "");
  const awaiting = g.awaiting_reply || [];
  const starred = g.starred || [];
  const unreadCount = g.unread_count || 0;
  const total = awaiting.length + starred.length;
  if (total === 0 && unreadCount === 0) return panel("gmail", "gmail", 0, "");
  let html = "";
  if (unreadCount) {
    html += `<div class="meta" style="margin-bottom:6px;">${unreadCount} sin leer en INBOX</div>`;
  }
  html += "<ul>";
  if (awaiting.length) {
    html += awaiting.slice(0, 5).map((m) => `<li>
      <span class="bucket overdue">${(m.days_old || 0)}d</span>
      <b>${esc(m.subject || "")}</b>
      <span class="meta">${esc(m.from_name || m.from || "")}</span>
      ${m.snippet ? `<div class="snippet">${esc(m.snippet)}</div>` : ""}
    </li>`).join("");
  }
  if (starred.length) {
    html += starred.slice(0, 3).map((m) => `<li>
      <span class="bucket today">★</span>
      <b>${esc(m.subject || "")}</b>
      <span class="meta">${esc(m.from_name || m.from || "")}</span>
    </li>`).join("");
  }
  html += "</ul>";
  return panel("gmail", "gmail", total || unreadCount, html);
}

function renderMail(items) {
  if (!items || !items.length) return panel("mail", "apple mail", 0, "");
  const lis = items.slice(0, 6).map((m) => {
    const vip = m.is_vip ? `<span class="bucket today">VIP</span>` : "";
    return `<li>
      ${vip}<b>${esc(m.subject || "")}</b>
      <span class="meta">${esc(m.sender || "")}</span>
    </li>`;
  }).join("");
  return panel("mail", "apple mail", items.length, `<ul>${lis}</ul>`);
}

function renderWhatsApp(items) {
  if (!items || !items.length) return panel("whatsapp", "whatsapp", 0, "");
  const lis = items.slice(0, 6).map((c) => `<li>
    <b>${esc(c.name || "")}</b>
    <span class="meta">${c.count || 0} msg</span>
    ${c.last_snippet ? `<div class="snippet">${esc(c.last_snippet)}</div>` : ""}
  </li>`).join("");
  return panel("whatsapp", "whatsapp", items.length, `<ul>${lis}</ul>`);
}

function renderWeather(w) {
  if (!w) return panel("weather", "weather", 0, "sin lluvia esperada");
  let body = `<div class="meta" style="color:var(--text);font-size:13px;margin-bottom:6px;">${esc(w.summary || "")}</div>`;
  if (Array.isArray(w.forecast) && w.forecast.length) {
    body += "<ul>";
    body += w.forecast.slice(0, 5).map((d) => `<li>
      <b>${esc(d.day || "")}</b>
      <span class="meta">${esc(d.summary || "")}</span>
    </li>`).join("");
    body += "</ul>";
  }
  return panel("weather", "weather", 1, body);
}

function renderLoops(activo, stale) {
  const a = activo || [];
  const s = stale || [];
  const total = a.length + s.length;
  if (total === 0) return panel("loops", "open loops", 0, "");
  let html = "<ul>";
  html += s.slice(0, 4).map((l) => `<li>
    <span class="bucket stale">stale</span>
    <b>${esc(l.loop_text || "")}</b>
    <span class="meta">${esc(l.source_note || "")} · ${l.age_days || 0}d</span>
  </li>`).join("");
  html += a.slice(0, 4).map((l) => `<li>
    <span class="bucket activo">activo</span>
    <b>${esc(l.loop_text || "")}</b>
    <span class="meta">${esc(l.source_note || "")} · ${l.age_days || 0}d</span>
  </li>`).join("");
  html += "</ul>";
  return panel("loops", "open loops", total, html);
}

function renderContradictions(items) {
  if (!items || !items.length) return panel("contradictions", "contradicciones", 0, "");
  const lis = items.slice(0, 5).map((c) => {
    const targets = (c.targets || []).slice(0, 2).map((t) => esc(t.path || "")).join(", ");
    const why = (c.targets || [])[0]?.why || "";
    return `<li>
      <b>${esc(c.subject_path || "")}</b>
      <span class="meta">↔ ${targets}</span>
      ${why ? `<div class="snippet">${esc(why)}</div>` : ""}
    </li>`;
  }).join("");
  return panel("contradictions", "contradicciones", items.length, `<ul>${lis}</ul>`);
}

function renderLowConf(items) {
  if (!items || !items.length) return panel("lowconf", "preguntas sin respuesta", 0, "");
  const lis = items.slice(0, 6).map((q) => `<li>
    <b>"${esc(q.q || "")}"</b>
    <span class="meta">score ${Number(q.top_score || 0).toFixed(3)}</span>
  </li>`).join("");
  return panel("lowconf", "preguntas sin respuesta", items.length, `<ul>${lis}</ul>`);
}

function renderInbox(items) {
  if (!items || !items.length) return panel("inbox", "capturado hoy", 0, "");
  const lis = items.slice(0, 6).map((i) => {
    const tagBits = i.tags && i.tags.length
      ? i.tags.map((t) => `#${esc(t)}`).join(" ")
      : `<span class="bucket sin-tags">sin-tags</span>`;
    return `<li>
      <b>${esc(i.title || "")}</b>
      <span class="meta">${tagBits} · ${fmtTime(i.modified)}</span>
      ${i.snippet ? `<div class="snippet">${esc(i.snippet.slice(0, 160))}</div>` : ""}
    </li>`;
  }).join("");
  return panel("inbox", "capturado hoy", items.length, `<ul>${lis}</ul>`);
}

function renderActivity(items) {
  if (!items || !items.length) return panel("activity", "notas tocadas", 0, "");
  const lis = items.slice(0, 6).map((n) => `<li>
    <b>${esc(n.title || "")}</b>
    <span class="meta">${esc(n.path || "")} · ${fmtTime(n.modified)}</span>
    ${n.snippet ? `<div class="snippet">${esc(n.snippet.slice(0, 160))}</div>` : ""}
  </li>`).join("");
  return panel("activity", "notas tocadas", items.length, `<ul>${lis}</ul>`);
}

// ── Urgent banner ───────────────────────────────────────────────────────
function renderUrgent(items) {
  if (!items || !items.length) {
    els.urgent.innerHTML = "";
    return;
  }
  const lis = items.slice(0, 6).map((u) => `<li>${esc(u)}</li>`).join("");
  els.urgent.innerHTML = `
    <div class="urgent">
      <h3>urgente</h3>
      <ul>${lis}</ul>
    </div>`;
}

// ── Narrative (today brief) ─────────────────────────────────────────────
function renderNarrative(text, source, briefPath, totalSignals) {
  if (!text) {
    if (totalSignals > 0) {
      els.narrative.innerHTML = `
        <div class="narrative" style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
          <span style="color:var(--text-dim);">brief del día pendiente · click en regenerar para armarlo (~10-15s)</span>
          <button class="home-btn" onclick="window._homeRegenerate && window._homeRegenerate()">regenerar</button>
        </div>`;
    } else {
      els.narrative.innerHTML = "";
    }
    return;
  }
  const html = window.marked
    ? marked.parse(stripWikilinks(text))
    : `<pre>${esc(stripWikilinks(text))}</pre>`;
  els.narrative.innerHTML = `<div class="narrative">${html}</div>`;
}

// ── Full render ─────────────────────────────────────────────────────────
function render(data) {
  els.error.innerHTML = "";
  els.date.textContent = data.date || "";

  renderUrgent(data.urgent || []);
  renderNarrative(
    data.today?.narrative,
    data.today?.narrative_source,
    data.today?.brief_path,
    data.today?.counts?.total || 0,
  );

  const signals = data.signals || {};
  const today = data.today?.evidence || {};

  const html = [
    renderReminders(signals.reminders),
    renderCalendar(signals.calendar, data.tomorrow_calendar),
    renderGmail(signals.gmail),
    renderWhatsApp(signals.whatsapp),
    renderMail(signals.mail_unread),
    renderActivity(today.recent_notes),
    renderInbox(today.inbox_today),
    renderLoops(signals.loops_activo, signals.loops_stale),
    renderLowConf(today.low_conf_queries),
    renderContradictions(today.new_contradictions),
    renderWeather(data.weather_forecast || signals.weather),
  ].join("");

  els.panels.innerHTML = html;

  const source = data.today?.narrative_source || "none";
  const dot = source === "generated" ? '<span class="dot gen">●</span>'
    : source === "cached" ? '<span class="dot">●</span>'
    : '<span class="dot stale">●</span>';
  const srcText = source === "cached" ? "brief del vault"
    : source === "generated" ? "brief recién generado"
    : (data.today?.counts?.total || 0) === 0 ? "sin actividad hoy"
    : "brief pendiente";
  els.status.innerHTML =
    `${dot} ${srcText} · ${(data.today?.counts?.total || 0)} señales vault · actualizado ${fmtTime(data.generated_at)}`;
}

async function load(regenerate = false) {
  els.btnReload.disabled = true;
  els.btnRegen.disabled = true;
  if (regenerate) {
    els.status.innerHTML = '<span class="dot gen">●</span> generando brief… (10-15s)';
  }
  try {
    const resp = await fetch(`/api/home?regenerate=${regenerate ? "true" : "false"}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    render(data);
  } catch (err) {
    els.error.innerHTML = `<div class="error">error: ${esc(err.message || String(err))}</div>`;
  } finally {
    els.btnReload.disabled = false;
    els.btnRegen.disabled = false;
  }
}

// ── Auto-refresh ────────────────────────────────────────────────────────
function scheduleRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    if (document.visibilityState === "visible") load(false);
  }, REFRESH_MS);
}

els.btnReload.addEventListener("click", () => load(false));
els.btnRegen.addEventListener("click", () => load(true));
window._homeRegenerate = () => load(true);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") load(false);
});

load(false);
scheduleRefresh();
