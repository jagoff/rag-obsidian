// Home — information centralizer. Aggregates every channel:
//   today brief · reminders · mail · gmail · whatsapp · calendar · weather
//   · open loops · contradictions · low-confidence queries · inbox activity
// Data source: GET /api/home (silent-fail per channel).
// Auto-refresh: every 60s unless tab is hidden.

// ── Behavior tracking ────────────────────────────────────────────────────────
// Session ID persisted in sessionStorage so all clicks in a tab share one id.
// Format: "web:<12 hex chars>" — satisfies SESSION_ID_RE in rag.py.
function _ragSessionId() {
  try {
    let sid = sessionStorage.getItem("rag_session");
    if (!sid) {
      const hex = Array.from(crypto.getRandomValues(new Uint8Array(6)))
        .map((b) => b.toString(16).padStart(2, "0")).join("");
      sid = "web:" + hex;
      sessionStorage.setItem("rag_session", sid);
    }
    return sid;
  } catch (_) {
    // Private browsing or storage blocked — generate per-call, fire-and-forget
    const hex = Array.from(crypto.getRandomValues(new Uint8Array(6)))
      .map((b) => b.toString(16).padStart(2, "0")).join("");
    return "web:" + hex;
  }
}

// Fire-and-forget behavior event. Uses sendBeacon for unload safety;
// falls back to fetch with keepalive. Never blocks the click.
function _trackBehavior(payload) {
  const body = JSON.stringify({ source: "web", ...payload, session: _ragSessionId() });
  const url = "/api/behavior";
  if (typeof navigator.sendBeacon === "function") {
    navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
  } else {
    fetch(url, { method: "POST", body, keepalive: true,
                 headers: { "Content-Type": "application/json" } }).catch(() => {});
  }
}

const els = {
  date: document.getElementById("date-label"),
  status: document.getElementById("status-line"),
  urgent: document.getElementById("urgent-wrap"),
  narrative: document.getElementById("narrative-wrap"),
  panels: document.getElementById("panels"),
  error: document.getElementById("error-wrap"),
  progress: document.getElementById("progress"),
};

// Refresh buttons live inside the narrative card (inline when the brief is
// pending, icon-only corner when the brief exists). Both carry class
// `.js-refresh` so loading-state and click delegation don't care which one
// is mounted.
function setRefreshSpinning(on) {
  document.querySelectorAll(".js-refresh").forEach((b) => {
    b.disabled = on;
    b.classList.toggle("spinning", on);
  });
}

const REFRESH_MS = 60_000;
const MIN_RELOAD_GAP_MS = 5_000;  // ignore back-to-back triggers (visibility + interval)
const WARMING_POLL_MS = 3_000;    // fast poll while server cache is cold
let refreshTimer = null;
let firstLoad = true;
let lastLoadStarted = 0;
let inflight = false;
let warmingPoll = null;

// Skeleton panels painted on first load so the page has structure from
// the first paint — avoids a ~4.5s blank gap while /api/home fans out.
// Subsequent refreshes keep the prior data on screen until the new
// payload arrives (no flicker).
const SKELETON_PANELS = [
  "gmail", "reminders", "calendar", "whatsapp · sin responder",
  "notas tocadas", "inbox hoy", "open loops",
  "low-conf", "contradicciones", "weather",
  "autoridad", "retrieval health", "loops aging", "web 7d",
  "drive 48h", "finanzas",
];

function renderSkeletons() {
  const html = SKELETON_PANELS.map((label) => `
    <section class="panel loading" aria-busy="true">
      <div class="head">
        <h3>${label}</h3>
        <span class="count">…</span>
      </div>
      <div class="skel-line w80"></div>
      <div class="skel-line w60"></div>
      <div class="skel-line w40"></div>
    </section>
  `).join("");
  els.panels.innerHTML = html;
}

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
        <button class="collapse-btn" type="button" aria-label="Colapsar/expandir" title="Colapsar / expandir">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="18 15 12 9 6 15"/>
          </svg>
        </button>
      </div>
      ${body}
    </section>`;
}

// ── Panel importance + user collapse state ───────────────────────────────
// Priority = where this channel belongs when it HAS data. Lower = higher
// on page. Auto-empty panels get shoved past 1000 so they sink no matter
// what. User-collapsed panels keep their priority slot (they just fold).
// Top 6 = inbox (actionable) + loops/signals. Rest is activity, then
// external context, then ambient/telemetry.
const PANEL_PRIORITY = {
  "panel-gmail": 10,
  "panel-reminders": 20,
  "panel-calendar": 30,
  "panel-wa-unreplied": 40,
  "panel-loops": 50,
  "panel-contradictions": 60,
  "panel-aging": 70,
  "panel-lowconf": 80,
  "panel-inbox": 90,
  "panel-activity": 100,
  "panel-drive": 110,
  "panel-chrome": 120,
  "panel-youtube": 125,
  "panel-bookmarks": 130,
  "panel-evaltrend": 140,
  "panel-pagerank": 150,
  "panel-weather": 160,
  "panel-finance": 170,
};
function panelPriority(id) {
  if (id in PANEL_PRIORITY) return PANEL_PRIORITY[id];
  if (id.startsWith("panel-vault-")) return 105;  // between inbox and activity
  return 200;
}

const COLLAPSED_KEY = "rag-home-collapsed";
function getCollapsedSet() {
  try {
    const raw = JSON.parse(localStorage.getItem(COLLAPSED_KEY) || "[]");
    return new Set(Array.isArray(raw) ? raw : []);
  } catch (_) { return new Set(); }
}
function saveCollapsedSet(set) {
  try { localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...set])); } catch (_) {}
}

// Run after every render(): assign inline `order` (empties → 1000+prio so
// they tail but still keep their category grouping) and restore the
// user-collapsed class from localStorage.
function applyPanelState() {
  const collapsed = getCollapsedSet();
  document.querySelectorAll("#panels .panel").forEach((p) => {
    const autoEmpty = !!p.querySelector(".empty");
    const pri = panelPriority(p.id);
    p.style.order = autoEmpty ? 1000 + pri : pri;
    if (collapsed.has(p.id)) p.classList.add("user-collapsed");
    else p.classList.remove("user-collapsed");
  });
}

// ── Renderers per channel ────────────────────────────────────────────────
function renderReminders(items) {
  if (!items || !items.length) return panel("reminders", "reminders", 0, "");
  const lis = items.slice(0, 8).map((r, i) => {
    const bucket = r.bucket || "undated";
    // backend emits `due` (ISO) or `due_display` (preformatted); fall back to ISO
    const dueText = r.due_display || (r.due ? fmtDate(r.due) + (r.due.includes("T") ? " " + fmtTime(r.due) : "") : "");
    const due = dueText ? `<span class="meta">${esc(dueText)}</span>` : "";
    const list = r.list ? `<span class="meta">${esc(r.list)}</span>` : "";
    const rid = r.id || "";
    const disabled = rid ? "" : "disabled";
    const cbTitle = rid ? "Marcar como completada" : "Sin id — no se puede completar desde acá";
    return `<li data-reminder-id="${esc(rid)}" data-reminder-idx="${i}">
      <button class="rem-check" type="button" aria-label="Completar" title="${esc(cbTitle)}" ${disabled}>
        <svg class="rem-check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </button>
      <span class="bucket ${esc(bucket)}">${esc(bucket)}</span>
      <b>${esc(r.name || "")}</b>
      ${due}${list}
    </li>`;
  }).join("");
  return panel("reminders", "reminders", items.length, `<ul class="rem-list">${lis}</ul>`);
}

function renderCalendar(today, tomorrow) {
  const t = today || [];
  const mx = tomorrow || [];
  const total = t.length + mx.length;
  if (total === 0) return panel("calendar", "calendar", 0, "");
  let html = "<ul>";
  if (t.length) {
    html += t.slice(0, 5).map((e) => {
      const when = e.start ? `<span class="meta">${esc(e.start)}${e.end ? " – " + esc(e.end) : ""}</span>` : "";
      return `<li><span class="bucket today">today</span><b>${esc(e.title || "")}</b>${when}</li>`;
    }).join("");
  }
  if (mx.length) {
    html += mx.slice(0, 5).map((e) => {
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
    html += `<div class="meta" style="margin-bottom:6px;">`
      + linked("https://mail.google.com/mail/u/0/#inbox",
          `${unreadCount} sin leer en INBOX`,
          { external: true, title: "abrir bandeja" })
      + `</div>`;
  }
  html += "<ul>";
  if (awaiting.length) {
    html += awaiting.slice(0, 5).map((m) => `<li>
      <span class="bucket overdue">${(m.days_old || 0)}d</span>
      ${linked(gmailHref(m), `<b>${esc(m.subject || "")}</b>`,
        { external: true, title: "abrir en gmail" })}
      <span class="meta">${esc(m.from_name || m.from || "")}</span>
      ${m.snippet ? `<div class="snippet">${esc(m.snippet)}</div>` : ""}
    </li>`).join("");
  }
  if (starred.length) {
    html += starred.slice(0, 3).map((m) => `<li>
      <span class="bucket today">★</span>
      ${linked(gmailHref(m), `<b>${esc(m.subject || "")}</b>`,
        { external: true, title: "abrir en gmail" })}
      <span class="meta">${esc(m.from_name || m.from || "")}</span>
    </li>`).join("");
  }
  html += "</ul>";
  return panel("gmail", "gmail", total || unreadCount, html);
}

function renderMail(items) {
  if (!items || !items.length) return panel("mail", "apple mail", 0, "");
  const lis = items.slice(0, 5).map((m) => {
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
    ${linked(waHref(c.jid), `<b>${esc(c.name || "")}</b>`,
      { external: true, title: "abrir chat" })}
    <span class="meta">${c.count || 0} msg</span>
    ${c.last_snippet ? `<div class="snippet">${esc(c.last_snippet)}</div>` : ""}
  </li>`).join("");
  return panel("whatsapp", "whatsapp", items.length, `<ul>${lis}</ul>`);
}

function renderWeather(w) {
  // API shape from /api/home → weather_forecast:
  //   { location, current: { description, temp_C },
  //     days: [{ date, minC, maxC, avgC, description, chanceofrain }] }
  // Legacy shape kept as fallback: { summary, forecast: [{ day, summary }] }.
  if (!w) return panel("weather", "weather", 0, "", "sin datos");
  const cur = w.current || {};
  const days = Array.isArray(w.days) ? w.days
    : Array.isArray(w.forecast) ? w.forecast.map((f) => ({ date: f.day, description: f.summary })) : [];
  const summary = w.summary || cur.description || "";
  if (!summary && !days.length) return panel("weather", "weather", 0, "", "sin datos");
  let body = "";
  if (summary || cur.temp_C) {
    const temp = cur.temp_C ? ` · ${esc(cur.temp_C)}°` : "";
    body += `<div class="meta" style="color:var(--text);font-size:13px;margin-bottom:6px;">${esc(summary)}${temp}</div>`;
  }
  if (days.length) {
    body += "<ul>";
    body += days.slice(0, 5).map((d) => {
      const range = (d.minC && d.maxC) ? `${esc(d.minC)}°–${esc(d.maxC)}°` : "";
      const rain = Number(d.chanceofrain) >= 50 ? ` · ${d.chanceofrain}% lluvia` : "";
      const meta = [esc(d.description || ""), range].filter(Boolean).join(" · ") + rain;
      return `<li><b>${esc(d.date || d.day || "")}</b><span class="meta">${meta}</span></li>`;
    }).join("");
    body += "</ul>";
  }
  return panel("weather", "weather", days.length || 1, body);
}

function renderLoops(activo, stale) {
  const a = activo || [];
  const s = stale || [];
  const total = a.length + s.length;
  if (total === 0) return panel("loops", "open loops", 0, "");
  const allItems = [...s, ...a];
  const row = (l, cls, rank) => {
    const href = obsidianHref(l.source_note);
    return `<li>
      <span class="bucket ${cls}">${cls}</span>
      ${linked(href, `<b>${esc(l.loop_text || "")}</b>`,
        { title: "abrir nota en obsidian", noteData: { path: l.source_note, rank } })}
      <span class="meta">${esc(l.source_note || "")} · ${l.age_days || 0}d</span>
    </li>`;
  };
  let html = "<ul>";
  html += s.slice(0, 4).map((l, i) => row(l, "stale", i + 1)).join("");
  html += a.slice(0, 4).map((l, i) => row(l, "activo", s.slice(0, 4).length + i + 1)).join("");
  html += "</ul>";
  return panel("loops", "open loops", total, html);
}

function renderContradictions(items) {
  if (!items || !items.length) return panel("contradictions", "contradicciones", 0, "");
  const lis = items.slice(0, 5).map((c, ci) => {
    const targets = (c.targets || []).slice(0, 2).map((t, ti) =>
      linked(obsidianHref(t.path), esc(t.path || ""),
        { title: "abrir nota relacionada", noteData: { path: t.path, rank: ti + 1 } })
    ).join(", ");
    const why = (c.targets || [])[0]?.why || "";
    return `<li>
      ${linked(obsidianHref(c.subject_path),
        `<b>${esc(c.subject_path || "")}</b>`,
        { title: "abrir en obsidian", noteData: { path: c.subject_path, rank: ci + 1 } })}
      <span class="meta">↔ ${targets}</span>
      ${why ? `<div class="snippet">${esc(why)}</div>` : ""}
    </li>`;
  }).join("");
  return panel("contradictions", "contradicciones", items.length, `<ul>${lis}</ul>`);
}

function renderLowConf(items) {
  if (!items || !items.length) return panel("lowconf", "preguntas sin respuesta", 0, "");
  const lis = items.slice(0, 5).map((q) => `<li>
    ${linked(chatHref(q.q), `<b>"${esc(q.q || "")}"</b>`,
      { title: "reintentar en chat" })}
    <span class="meta">score ${Number(q.top_score || 0).toFixed(3)}</span>
  </li>`).join("");
  return panel("lowconf", "preguntas sin respuesta", items.length, `<ul>${lis}</ul>`);
}

function renderInbox(items) {
  if (!items || !items.length) return panel("inbox", "capturado hoy", 0, "");
  const lis = items.slice(0, 5).map((i, idx) => {
    const tagBits = i.tags && i.tags.length
      ? i.tags.map((t) => `#${esc(t)}`).join(" ")
      : `<span class="bucket sin-tags">sin-tags</span>`;
    return `<li>
      ${linked(obsidianHref(i.path), `<b>${esc(i.title || "")}</b>`,
        { title: "abrir en obsidian", noteData: { path: i.path, rank: idx + 1 } })}
      <span class="meta">${tagBits} · ${fmtTime(i.modified)}</span>
      ${i.snippet ? `<div class="snippet">${esc(i.snippet.slice(0, 160))}</div>` : ""}
    </li>`;
  }).join("");
  return panel("inbox", "capturado hoy", items.length, `<ul>${lis}</ul>`);
}

function renderActivity(items) {
  if (!items || !items.length) return panel("activity", "notas tocadas", 0, "");
  const lis = items.slice(0, 5).map((n, idx) => `<li>
    ${linked(obsidianHref(n.path), `<b>${esc(n.title || "")}</b>`,
      { title: "abrir en obsidian", noteData: { path: n.path, rank: idx + 1 } })}
    <span class="meta">${esc(n.path || "")} · ${fmtTime(n.modified)}</span>
    ${n.snippet ? `<div class="snippet">${esc(n.snippet.slice(0, 160))}</div>` : ""}
  </li>`).join("");
  return panel("activity", "notas tocadas", items.length, `<ul>${lis}</ul>`);
}

// obsidian:// deep-link — opens the note in the desktop app.
// Vault name is hardcoded since the whole app targets a single vault.
function obsidianHref(path) {
  if (!path) return "";
  return `obsidian://open?vault=Notes&file=${encodeURIComponent(path)}`;
}

// Gmail web search — no thread_id in payload, but from+subject lands on the
// thread reliably. Falls back to inbox if both missing.
function gmailHref(m) {
  if (!m) return "";
  const parts = [];
  const fromAddr = (m.from || "").match(/<([^>]+)>/)?.[1] || m.from || "";
  if (fromAddr) parts.push(`from:${fromAddr}`);
  if (m.subject) parts.push(`subject:"${m.subject.replace(/"/g, "")}"`);
  const q = parts.join(" ");
  return q
    ? `https://mail.google.com/mail/u/0/#search/${encodeURIComponent(q)}`
    : `https://mail.google.com/mail/u/0/#inbox`;
}

// wa.me — only useful for individual chats (jid like 5493...@s.whatsapp.net).
// Group jids (...@g.us) don't open externally; return chat list instead.
function waHref(jid) {
  if (!jid) return "";
  const phone = String(jid).split("@")[0];
  if (/^\d{6,}$/.test(phone)) return `https://wa.me/${phone}`;
  return "https://web.whatsapp.com/";
}

// Re-run a low-confidence query in the chat UI (auto-submits via ?q=).
function chatHref(q) {
  return `/chat?q=${encodeURIComponent(q || "")}`;
}

// Helper: wrap a label in <a> if href is non-empty, else plain.
// opts.noteData = { path, rank, query } adds data-note-* attrs for click tracking.
function linked(href, html, opts = {}) {
  if (!href) return html;
  const target = opts.external ? ` target="_blank" rel="noopener noreferrer"` : "";
  const title = opts.title ? ` title="${esc(opts.title)}"` : "";
  let data = "";
  if (opts.noteData) {
    const nd = opts.noteData;
    if (nd.path) data += ` data-note-path="${esc(nd.path)}"`;
    if (nd.rank != null) data += ` data-note-rank="${Number(nd.rank)}"`;
    if (nd.query) data += ` data-note-query="${esc(nd.query)}"`;
  }
  return `<a href="${esc(href)}"${target}${title}${data}>${html}</a>`;
}

function truncate(s, n) {
  const str = String(s || "");
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

// signals.pagerank_top — wikilink graph authority. Items already sorted
// asc by rank. Render top-5 with a visual bar scaled against the max pr
// in the slice (purely relative, pr is unitless).
function renderPageRank(items) {
  if (!items || !items.length) return panel("pagerank", "autoridad", 0, "");
  const slice = items.slice(0, 5);
  const maxPr = Math.max(...slice.map((x) => Number(x.pr || 0)), 1e-9);
  const lis = slice.map((it, idx) => {
    const pct = Math.max(4, Math.round((Number(it.pr || 0) / maxPr) * 100));
    const href = obsidianHref(it.path);
    const title = esc(it.title || it.path || "");
    const rankLabel = it.rank != null ? `#${it.rank}` : "";
    const trackData = it.path
      ? ` data-note-path="${esc(it.path)}" data-note-rank="${idx + 1}"`
      : "";
    return `<li>
      <span class="rank">${esc(rankLabel)}</span>
      <a href="${esc(href)}"${trackData}><b>${title}</b></a>
      <div class="pr-bar" aria-hidden="true"><span style="width:${pct}%"></span></div>
    </li>`;
  }).join("");
  return panel("pagerank", "autoridad", slice.length, `<ul class="pr-list">${lis}</ul>`);
}

// signals.eval_trend — latest retrieval metrics vs fixed baseline. The
// deltas are what matter (singles drift naturally with vault growth —
// see CLAUDE.md "Eval baselines"); chains/chain_success are the real
// signal of ranker health.
function renderEvalTrend(t) {
  if (!t || !t.latest) return panel("evaltrend", "retrieval health", 0, "");
  const L = t.latest || {};
  const B = t.baseline || {};
  const sL = L.singles || {};
  const cL = L.chains || {};

  const row = (label, cur, base, fmt) => {
    const curN = Number(cur);
    const baseN = Number(base);
    const delta = Number.isFinite(curN) && Number.isFinite(baseN) ? (curN - baseN) : null;
    const curStr = Number.isFinite(curN) ? fmt(curN) : "—";
    const baseStr = Number.isFinite(baseN) ? fmt(baseN) : "—";
    let deltaHtml = "";
    if (delta != null) {
      const cls = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "flat";
      const arrow = cls === "up" ? "▲" : cls === "down" ? "▼" : "·";
      const sign = delta > 0 ? "+" : "";
      deltaHtml = `<span class="delta ${cls}">${arrow} ${sign}${fmt(delta)}</span>`;
    }
    return `<li>
      <b>${esc(label)}</b>
      <span class="meta">${esc(curStr)} <span class="baseline">vs ${esc(baseStr)}</span></span>
      ${deltaHtml}
    </li>`;
  };

  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const num = (x) => x.toFixed(3);

  // Sparkline from history: plot singles hit@5 over time if available.
  // 60×20 SVG polyline. Cheap and legible; no library.
  let spark = "";
  const hist = Array.isArray(t.history) ? t.history.filter((h) => h && Number.isFinite(h.singles_hit5)) : [];
  if (hist.length >= 2) {
    const vals = hist.slice(-20).map((h) => Number(h.singles_hit5));
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = Math.max(max - min, 1e-6);
    const W = 60, H = 20;
    const pts = vals.map((v, i) => {
      const x = (i / (vals.length - 1)) * W;
      const y = H - ((v - min) / range) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    spark = `<svg class="sparkline" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">
      <polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.2"/>
    </svg>`;
  }

  const rows = [
    row("singles hit@5", sL.hit5, B.singles_hit5, pct),
    row("chains hit@5",  cL.hit5, B.chains_hit5, pct),
    row("chain_success", cL.chain_success, B.chain_success, pct),
    row("chains MRR",    cL.mrr, B.chains_mrr, num),
  ].join("");

  const nmeta = sL.n || cL.turns
    ? `<div class="meta" style="margin-bottom:6px;">n=${esc(sL.n || 0)} singles · ${esc(cL.turns || 0)} turns / ${esc(cL.chains || 0)} chains ${spark}</div>`
    : (spark ? `<div class="meta" style="margin-bottom:6px;">${spark}</div>` : "");

  return panel("evaltrend", "retrieval health", 1, `${nmeta}<ul class="eval-list">${rows}</ul>`);
}

// signals.followup_aging — open-loop age distribution. 3 buckets + a
// small sample. Colors escalate with age.
function renderFollowupAging(a) {
  if (!a || !a.total) return panel("aging", "loops aging", 0, "");
  const b = a.buckets || {};
  const fresh  = Number(b["0_7"] || 0);
  const amber  = Number(b["8_30"] || 0);
  const stale  = Number(b["stale_30plus"] || 0);
  const badges = `
    <div class="age-badges">
      <span class="bucket age-fresh">0-7d · ${fresh}</span>
      <span class="bucket age-amber">8-30d · ${amber}</span>
      <span class="bucket age-stale">stale · ${stale}</span>
    </div>`;
  const sample = Array.isArray(a.sample) ? a.sample.slice(0, 3) : [];
  const lis = sample.map((s, idx) => {
    const age = Number(s.age_days || 0);
    const cls = age <= 7 ? "age-fresh" : age <= 30 ? "age-amber" : "age-stale";
    return `<li>
      <span class="bucket ${cls}">${age}d</span>
      ${linked(obsidianHref(s.note),
        `<b>${esc(truncate(s.loop || "", 80))}</b>`,
        { title: "abrir nota en obsidian", noteData: { path: s.note, rank: idx + 1 } })}
      <span class="meta">${esc(s.note || "")}${s.status ? " · " + esc(s.status) : ""}</span>
    </li>`;
  }).join("");
  const body = sample.length ? `${badges}<ul>${lis}</ul>` : badges;
  return panel("aging", "loops aging", a.total, body);
}

// signals.chrome_top_week — top visited URLs (external). Host shown as
// affordance before click; count as a bucket badge.
function renderChromeTopWeek(items) {
  if (!items || !items.length) return panel("chrome", "web 7d", 0, "");
  const lis = items.slice(0, 5).map((u) => {
    let host = "";
    try { host = new URL(u.url).host.replace(/^www\./, ""); } catch (_) { host = ""; }
    const title = truncate(u.title || host || u.url || "", 50);
    const count = Number(u.visit_count || 0);
    return `<li>
      <a href="${esc(u.url || "")}" target="_blank" rel="noopener noreferrer"><b>${esc(title)}</b></a>
      <span class="meta">${esc(host)}${u.last_visit_iso ? " · " + esc(fmtDate(u.last_visit_iso)) : ""}</span>
      <span class="bucket today">${count}</span>
    </li>`;
  }).join("");
  return panel("chrome", "web 7d", items.length, `<ul>${lis}</ul>`);
}

// signals.drive_recent — files modified in Google Drive within the last
// 48h. Item shape from rag._fetch_drive_evidence: `{ name, link,
// mime_label, days_ago, modifier }`. `days_ago` is a float so sub-day
// ages render as hours for finer signal.
function renderDriveRecent(items) {
  if (!items || !items.length) return panel("drive", "drive 48h", 0, "");
  const lis = items.slice(0, 5).map((f) => {
    const name = truncate(f.name || "", 60);
    const href = f.link || "";
    const age = Number(f.days_ago || 0);
    const ageStr = age < 1 ? `${Math.round(age * 24)}h` : `${age.toFixed(1)}d`;
    const modifier = f.modifier ? ` · ${esc(f.modifier)}` : "";
    const linkHtml = href
      ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer"><b>${esc(name)}</b></a>`
      : `<b>${esc(name)}</b>`;
    return `<li>
      ${linkHtml}
      <span class="meta">${esc(f.mime_label || "archivo")} · ${esc(ageStr)}${modifier}</span>
    </li>`;
  }).join("");
  return panel("drive", "drive 48h", items.length, `<ul>${lis}</ul>`);
}

// signals.chrome_bookmarks — bookmarks whose URL was visited in last 48h.
// Distinct from `renderChromeTopWeek` (all visits): narrows to *saved*
// pages the user reaches for. Item shape: `{ name, url, folder,
// visit_count, last_visit_iso }`.
function renderChromeBookmarks(items) {
  // Google homepage / search is high-volume noise — visiting it says
  // nothing about current context. Drop it so the panel stays useful
  // as a "what am I working on" signal. Sibling google domains
  // (docs/drive/calendar/etc.) stay — those ARE work context.
  const filtered = (items || []).filter((b) => {
    let host = "";
    try { host = new URL(b.url).host.replace(/^www\./, ""); } catch (_) { return true; }
    return host !== "google.com";
  });
  if (!filtered.length) return panel("bookmarks", "bookmarks 48h", 0, "");
  const lis = filtered.slice(0, 5).map((b) => {
    let host = "";
    try { host = new URL(b.url).host.replace(/^www\./, ""); } catch (_) { host = ""; }
    const title = truncate(b.name || host || b.url || "", 56);
    const folder = b.folder ? truncate(b.folder, 36) : "";
    const count = Number(b.visit_count || 0);
    return `<li>
      <a href="${esc(b.url || "")}" target="_blank" rel="noopener noreferrer"><b>${esc(title)}</b></a>
      <span class="meta">${esc(host)}${folder ? " · " + esc(folder) : ""}${b.last_visit_iso ? " · " + esc(fmtDate(b.last_visit_iso)) : ""}</span>
      <span class="bucket today">${count}</span>
    </li>`;
  }).join("");
  return panel("bookmarks", "bookmarks 48h", filtered.length, `<ul>${lis}</ul>`);
}

// signals.youtube_watched — recent YouTube /watch pages from Chrome
// history (last 7 days, deduplicated by video_id). Title comes from the
// Chrome <title> cache ("Video Title - YouTube" stripped server-side).
// Thumbnail comes from the canonical i.ytimg.com pattern — no API/key,
// lazy-loaded so cold paint stays fast.
function renderYouTubeWatched(items) {
  if (!items || !items.length) return panel("youtube", "youtube 7d", 0, "");
  const lis = items.slice(0, 5).map((v) => {
    const title = truncate(v.title || v.url || "", 60);
    const when = v.last_visit_iso ? fmtDate(v.last_visit_iso) : "";
    const vid = v.video_id || "";
    const thumb = vid
      ? `<img src="https://i.ytimg.com/vi/${esc(vid)}/mqdefault.jpg" alt="" loading="lazy"
             style="width:72px;height:40px;object-fit:cover;border-radius:3px;flex:0 0 auto;background:var(--border);">`
      : "";
    return `<li style="display:flex;gap:10px;align-items:flex-start;">
      ${thumb}
      <div style="min-width:0;flex:1;">
        <a href="${esc(v.url || "")}" target="_blank" rel="noopener noreferrer"><b>${esc(title)}</b></a>
        <span class="meta">youtube${when ? " · " + esc(when) : ""}</span>
      </div>
    </li>`;
  }).join("");
  return panel("youtube", "youtube 7d", items.length, `<ul>${lis}</ul>`);
}

// signals.vault_activity — per-vault recent notes, never mixed. Shape:
// `{ vault_name: [{ path, title, modified, snippet }] }`. Emits one
// panel per vault so personal/work stay separated. Panel title encodes
// the vault name directly — easy to spot at a glance.
function renderVaultActivity(byVault) {
  if (!byVault || typeof byVault !== "object") return "";
  return Object.keys(byVault).sort().map((name) => {
    const items = byVault[name] || [];
    if (!items.length) return panel(`vault-${name}`, `vault ${name} 48h`, 0, "");
    const lis = items.slice(0, 5).map((n, idx) => {
      const title = truncate(n.title || "", 56);
      const snippet = truncate(n.snippet || "", 120);
      const obsidianUrl = `obsidian://open?vault=${encodeURIComponent(name)}&file=${encodeURIComponent(n.path || "")}`;
      const trackData = n.path
        ? ` data-note-path="${esc(n.path)}" data-note-rank="${idx + 1}"`
        : "";
      return `<li>
        <a href="${esc(obsidianUrl)}"${trackData}><b>${esc(title)}</b></a>
        <span class="meta">${esc(n.modified ? fmtDate(n.modified) : "")}${n.path ? " · " + esc(n.path) : ""}</span>
        ${snippet ? `<div class="snippet">${esc(snippet)}</div>` : ""}
      </li>`;
    }).join("");
    return panel(`vault-${name}`, `vault ${name} 48h`, items.length, `<ul>${lis}</ul>`);
  }).join("");
}

// signals.finance — MOZE (Money app) export parsed locally. Shape:
//   { month_label, days_elapsed, days_in_month,
//     ars: { this_month, prev_month, delta_pct, run_rate_daily,
//            projected, income, top_categories:[{name,amount,share}] },
//     usd: { this_month, prev_month },
//     latest: [{ date, type, category, name, amount, currency }] }
// Expenses are abs() already; Income is positive.
function renderFinance(f) {
  if (!f || !f.ars) return panel("finance", "finanzas", 0, "", "sin export MOZE");
  const ars = f.ars;
  const fmtArs = (n) => {
    const v = Math.round(Number(n || 0));
    return "$" + v.toLocaleString("es-AR");
  };
  const fmtPct = (p) => {
    if (p == null || !isFinite(p)) return "";
    const sign = p > 0 ? "+" : "";
    return `${sign}${p.toFixed(1)}%`;
  };
  const deltaCls = ars.delta_pct == null ? "flat"
    : ars.delta_pct > 0.5 ? "down"   // more spent = red
    : ars.delta_pct < -0.5 ? "up"
    : "flat";
  const arrow = deltaCls === "up" ? "▼" : deltaCls === "down" ? "▲" : "·";

  const header = `
    <li>
      <b>${fmtArs(ars.this_month)} ARS</b>
      <span class="meta">mes ${esc(f.month_label || "")} · día ${esc(f.days_elapsed)}/${esc(f.days_in_month)} · proy ${fmtArs(ars.projected)}</span>
      <span class="delta ${deltaCls}">${arrow} ${esc(fmtPct(ars.delta_pct))} vs mes ant (${fmtArs(ars.prev_month)})</span>
    </li>`;

  const cats = (ars.top_categories || []).slice(0, 5).map((c) => {
    const pct = Math.round((Number(c.share || 0)) * 100);
    return `<li>
      <b>${esc(c.name || "—")}</b>
      <span class="meta">${fmtArs(c.amount)} · ${pct}%</span>
      <div class="pr-bar"><span style="width:${pct}%;"></span></div>
    </li>`;
  }).join("");

  const latest = (f.latest || []).slice(0, 4).map((t) => {
    const sign = t.type === "Income" ? "+" : "-";
    const cur = t.currency || "";
    const amt = Math.round(Math.abs(Number(t.amount || 0))).toLocaleString("es-AR");
    const cls = t.type === "Income" ? "up" : "down";
    return `<li>
      <b>${esc(t.name || t.category || "—")}</b>
      <span class="meta">${esc(t.date)} · ${esc(t.category || "")}${t.store ? " · " + esc(t.store) : ""}</span>
      <span class="delta ${cls}">${sign}$${esc(amt)} ${esc(cur)}</span>
    </li>`;
  }).join("");

  const usdLine = (f.usd && (f.usd.this_month || f.usd.prev_month))
    ? `<li><b>USD mes</b><span class="meta">prev ${esc(Math.round(f.usd.prev_month || 0).toLocaleString("en-US"))}</span><span class="delta flat">$${esc(Math.round(f.usd.this_month || 0).toLocaleString("en-US"))}</span></li>`
    : "";

  const body = `<ul>${header}${usdLine}</ul>`
    + (cats ? `<div class="meta" style="margin:10px 0 4px;">top categorías</div><ul>${cats}</ul>` : "")
    + (latest ? `<div class="meta" style="margin:10px 0 4px;">últimos movimientos</div><ul>${latest}</ul>` : "");

  return panel("finance", "finanzas", null, body);
}

// signals.whatsapp_unreplied — chats whose last message is inbound and
// still awaits a reply. Distinct from `renderWhatsApp` (unread inbound
// in last 24h): this one flags where *you* owe the next move. Item
// shape: `{ name, jid, last_snippet, hours_waiting }`. Age bucket
// colors reuse the followup-aging palette for visual consistency.
function renderWhatsAppUnreplied(items) {
  if (!items || !items.length) return panel("wa-unreplied", "whatsapp · sin responder", 0, "");
  const lis = items.slice(0, 5).map((c) => {
    const h = Number(c.hours_waiting || 0);
    const ageStr = h < 1 ? "<1h" : h < 24 ? `${Math.round(h)}h` : `${Math.round(h / 24)}d`;
    const cls = h >= 24 ? "age-stale" : h >= 8 ? "age-amber" : "age-fresh";
    return `<li>
      <span class="bucket ${cls}">${esc(ageStr)}</span>
      ${linked(waHref(c.jid), `<b>${esc(c.name || "")}</b>`,
        { external: true, title: "abrir chat" })}
      ${c.last_snippet ? `<div class="snippet">${esc(c.last_snippet)}</div>` : ""}
    </li>`;
  }).join("");
  return panel("wa-unreplied", "whatsapp · sin responder", items.length, `<ul>${lis}</ul>`);
}

// ── Urgent banner ───────────────────────────────────────────────────────
function renderUrgent(items) {
  if (!items || !items.length) {
    els.urgent.innerHTML = "";
    return;
  }
  const lis = items.slice(0, 5).map((u) => `<li>${esc(u)}</li>`).join("");
  els.urgent.innerHTML = `
    <div class="urgent">
      <h3>urgente</h3>
      <ul>${lis}</ul>
    </div>`;
}

// ── Narrative (today brief) ─────────────────────────────────────────────
function renderNarrative(text, source, briefPath, totalSignals) {
  const refreshBtnFull = `
    <button type="button" class="home-btn home-btn-icon js-refresh"
            title="Refrescar todo (fuentes + brief)" aria-label="Refrescar">
      <svg class="refresh-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21 12a9 9 0 1 1-3-6.7"/>
        <path d="M21 3v6h-6"/>
      </svg>
      <span>refrescar</span>
    </button>`;
  const refreshBtnIcon = `
    <button type="button" class="home-btn home-btn-icon js-refresh narrative-refresh"
            title="Refrescar todo (fuentes + brief)" aria-label="Refrescar">
      <svg class="refresh-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21 12a9 9 0 1 1-3-6.7"/>
        <path d="M21 3v6h-6"/>
      </svg>
    </button>`;
  if (!text) {
    if (totalSignals > 0) {
      els.narrative.innerHTML = `
        <div class="narrative narrative-pending">
          <span style="color:var(--text-dim);">brief del día pendiente · armarlo toma ~10-15s</span>
          ${refreshBtnFull}
        </div>`;
    } else {
      els.narrative.innerHTML = `
        <div class="narrative narrative-pending">
          <span style="color:var(--text-dim);">sin actividad hoy</span>
          ${refreshBtnFull}
        </div>`;
    }
    return;
  }
  const html = window.marked
    ? marked.parse(stripWikilinks(text))
    : `<pre>${esc(stripWikilinks(text))}</pre>`;
  els.narrative.innerHTML = `
    <div class="narrative narrative-filled">
      ${refreshBtnIcon}
      ${html}
    </div>`;
  injectTomorrowReminderButtons();
}

// Tracks which "Para mañana" items were already converted in this tab.
// Needed because load() re-renders the narrative via innerHTML — the
// inline `.reminder-created` class would otherwise vanish after refresh
// and let the user double-post the same item as a new reminder.
const _createdReminderTexts = new Set();

// Locate the "Para mañana" section in the rendered narrative and annotate
// each <li> with an "enter arrow" button that POSTs to /api/reminders/create
// on click. We snapshot the li's text BEFORE appending the button so the
// icon's accessible-name SVG doesn't leak into the reminder body.
function injectTomorrowReminderButtons() {
  const root = els.narrative.querySelector(".narrative");
  if (!root) return;
  const heads = [...root.querySelectorAll("h2")];
  const target = heads.find((h) => /ma[ñn]ana/i.test(h.textContent || ""));
  if (!target) return;
  let el = target.nextElementSibling;
  while (el && el.tagName !== "H2") {
    if (el.tagName === "UL" || el.tagName === "OL") {
      [...el.querySelectorAll(":scope > li")].forEach((li) => {
        if (li.querySelector(".add-reminder-btn")) return;
        const original = (li.textContent || "").trim();
        if (!original) return;
        li.dataset.reminderText = original;
        const btn = document.createElement("button");
        btn.className = "add-reminder-btn";
        btn.type = "button";
        btn.title = "Crear Apple Reminder (mañana 9:00)";
        btn.setAttribute("aria-label", "Crear Apple Reminder");
        btn.innerHTML =
          '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor"' +
          ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
          '<polyline points="9 10 4 15 9 20"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/></svg>';
        li.appendChild(btn);
        // Persist the created-state across narrative re-renders.
        if (_createdReminderTexts.has(original)) {
          li.classList.add("reminder-created");
          btn.disabled = true;
          btn.title = "Ya agregado a Reminders";
        }
      });
    }
    el = el.nextElementSibling;
  }
}

// Delegated click handler on the narrative container — narrative HTML is
// re-rendered via innerHTML every load(), so per-button listeners would
// leak. Creates the Apple Reminder once per <li>; success paints a green
// checkmark, error stains the icon red with the message in the tooltip.
els.narrative.addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".add-reminder-btn");
  if (!btn) return;
  ev.preventDefault();
  ev.stopPropagation();
  const li = btn.closest("li");
  if (!li || li.classList.contains("reminder-created")) return;
  const text = li.dataset.reminderText || (li.textContent || "").trim();
  if (!text) return;
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    const res = await fetch("/api/reminders/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, due: "tomorrow" }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    _createdReminderTexts.add(text);
    li.classList.add("reminder-created");
    btn.title = "Agregado a Reminders";
    // Server already busted _HOME_STATE cache; fetch fresh so the
    // reminders panel surfaces the new item without waiting for the
    // 60s auto-cycle. bypassDebounce because load() otherwise swallows
    // calls issued within MIN_RELOAD_GAP_MS of the previous one.
    load(false, { bypassDebounce: true });
  } catch (e) {
    btn.classList.add("err");
    btn.title = "Error: " + (e.message || String(e));
    btn.disabled = false;
  } finally {
    btn.classList.remove("loading");
  }
});

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
    renderGmail(signals.gmail),
    renderReminders(signals.reminders),
    renderCalendar(signals.calendar, data.tomorrow_calendar),
    renderWhatsAppUnreplied(signals.whatsapp_unreplied),
    renderActivity(today.recent_notes),
    renderInbox(today.inbox_today),
    renderLoops(signals.loops_activo, signals.loops_stale),
    renderLowConf(today.low_conf_queries),
    renderContradictions(today.new_contradictions),
    renderWeather(data.weather_forecast || signals.weather),
    renderPageRank(signals.pagerank_top),
    renderEvalTrend(signals.eval_trend),
    renderFollowupAging(signals.followup_aging),
    renderChromeTopWeek(signals.chrome_top_week),
    renderChromeBookmarks(signals.chrome_bookmarks),
    renderYouTubeWatched(signals.youtube_watched),
    renderVaultActivity(signals.vault_activity),
    renderDriveRecent(signals.drive_recent),
    renderFinance(signals.finance),
  ].join("");

  els.panels.innerHTML = html;
  applyPanelState();

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

async function load(regenerate = false, { bypassDebounce = false } = {}) {
  // Debounce: collapse rapid back-to-back loads (visibilitychange firing
  // right after the 60s interval would double-fetch). Always honor an
  // explicit regenerate. Warming short-polls pass bypassDebounce — they
  // fire every 3s and must not be swallowed by the 5s window.
  const now = Date.now();
  if (inflight) return;  // never issue two concurrent fetches
  if (!regenerate && !bypassDebounce && now - lastLoadStarted < MIN_RELOAD_GAP_MS) return;
  lastLoadStarted = now;
  inflight = true;

  const spinStart = Date.now();
  setRefreshSpinning(true);
  els.progress.classList.add("active");

  // First paint: show skeleton panels so the page has structure while
  // the fetch fans out (~4.5s cold). On refresh the prior data stays
  // visible — no flicker, just the top progress bar + pulsing dot.
  if (firstLoad) renderSkeletons();

  const dot = regenerate
    ? '<span class="dot gen loading">●</span> generando brief… (10-15s)'
    : '<span class="dot loading">●</span> cargando canales…';
  els.status.innerHTML = dot;

  const slowTimer = setTimeout(() => {
    if (inflight) {
      els.status.innerHTML = '⏱ tardando más de lo normal — Ollama puede estar offline';
      els.status.classList.add("status-slow");
    }
  }, 8000);

  try {
    const resp = await fetch(`/api/home?regenerate=${regenerate ? "true" : "false"}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (data.warming) {
      // Backend pre-warmer is still computing the first payload. Keep
      // skeletons up and short-poll so we catch the real payload as soon
      // as it lands (waiting for the 60s auto-refresh stalls the UI up
      // to 90s on cold starts — pre-warmer finishes in ~30s).
      els.status.innerHTML = '<span class="dot loading">●</span> calentando caché… (primer arranque, ~30s)';
      if (warmingPoll) clearTimeout(warmingPoll);
      warmingPoll = setTimeout(() => {
        warmingPoll = null;
        load(false, { bypassDebounce: true });
      }, WARMING_POLL_MS);
      return;
    }
    if (warmingPoll) { clearTimeout(warmingPoll); warmingPoll = null; }
    render(data);
    // render() rewrites the narrative card's innerHTML, which destroys
    // the spinning button + rebuilds a fresh one without the class. Re-
    // apply so the spin keeps going until MIN_SPIN_MS elapses.
    setRefreshSpinning(true);
    firstLoad = false;
  } catch (err) {
    els.error.innerHTML = `<div class="error">error: ${esc(err.message || String(err))}</div>`;
    els.status.innerHTML = '<span class="dot stale">●</span> error — reintentá';
  } finally {
    clearTimeout(slowTimer);
    els.status.classList.remove("status-slow");
    // Cache hit returns in ~300ms — faster than perception. Hold the
    // spinner + disabled state long enough that the click registers as
    // feedback. Slow regenerate paths already exceed this floor.
    const MIN_SPIN_MS = 800;
    const remain = Math.max(0, MIN_SPIN_MS - (Date.now() - spinStart));
    setTimeout(() => {
      inflight = false;
      setRefreshSpinning(false);
      els.progress.classList.remove("active");
    }, remain);
  }
}

// ── Auto-refresh ────────────────────────────────────────────────────────
function scheduleRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    if (document.visibilityState === "visible") load(false);
  }, REFRESH_MS);
}

// One action — delegated click catches whichever refresh button is
// currently mounted (inline pending / corner when brief exists).
// Fetches fresh evidence + regenerates the LLM brief (~10-15s).
document.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".js-refresh");
  if (!btn || btn.disabled) return;
  load(true);
});

// Manual collapse toggle: click chevron → fold body, persist in
// localStorage so the state survives reloads and re-renders. Re-renders
// call applyPanelState() which restores the class from the stored set.
els.panels.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".collapse-btn");
  if (!btn) return;
  ev.stopPropagation();
  ev.preventDefault();
  const p = btn.closest(".panel");
  if (!p) return;
  const set = getCollapsedSet();
  if (p.classList.toggle("user-collapsed")) set.add(p.id);
  else set.delete(p.id);
  saveCollapsedSet(set);
});

// Event delegation on the panels container: the reminders list is
// re-rendered via innerHTML on every load() so per-element listeners
// would leak. Delegation survives re-renders.
els.panels.addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".rem-check");
  if (!btn || btn.disabled) return;
  const li = btn.closest("li[data-reminder-id]");
  if (!li) return;
  const rid = li.dataset.reminderId;
  if (!rid) return;
  if (li.classList.contains("completed") || li.classList.contains("completing")) return;
  li.classList.add("completing");
  btn.disabled = true;
  try {
    const res = await fetch("/api/reminders/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reminder_id: rid }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    li.classList.remove("completing");
    li.classList.add("completed");
    // Small delay so the strike-through/fade is visible, then soft-remove.
    setTimeout(() => { li.style.display = "none"; }, 900);
  } catch (err) {
    li.classList.remove("completing");
    li.classList.add("complete-err");
    btn.disabled = false;
    btn.title = `Error: ${err.message}`;
  }
});
// Track clicks on vault note links (obsidian:// href). These links are
// rendered with data-note-path and data-note-rank set by the renderers
// (see addNoteTracking()). sendBeacon fires before navigation so the
// event survives even if the page unloads.
els.panels.addEventListener("click", (ev) => {
  const a = ev.target.closest("a[data-note-path]");
  if (!a) return;
  const path = a.dataset.notePath || null;
  const rank = a.dataset.noteRank ? Number(a.dataset.noteRank) : null;
  const query = a.dataset.noteQuery || null;
  if (!path) return;
  _trackBehavior({ event: "open", path, rank, query });
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") load(false);
});

load(false);
scheduleRefresh();
