// core.mjs — selectores, fetch helpers, formatters compartidos y auto-refresh.
// Importado por app.mjs y por cada panel-*.mjs.

// ── Selectores ────────────────────────────────────────────────────────────────

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ── Filtros de remitente ───────────────────────────────────────────────────────

// Detecta emails de bots/notificaciones/el propio user para no contarlos
// como "contactos reales".
export const isBotOrSelf = (sender) => {
  if (!sender) return true;
  const s = String(sender).toLowerCase();
  if (/notifications?@|noreply@|no-?reply@|donot-?reply@|notificaci[oó]n@|automat|bounce@|mailer-?daemon/.test(s)) return true;
  if (/no\s*responder|no-?responder|do\s*not\s*reply|noreply/.test(s)) return true;
  if (/fernandoferrari|fer\.f@/.test(s)) return true;
  return false;
};

// Extrae nombre legible del campo "sender". Cubre:
//   "Monica Ferrari <monica.ferrari@gmail.com>"
//   "\"Fer F.\" <fernandoferrari@gmail.com>"
//   "monica.ferrari@gmail.com"
export const parseSenderName = (sender) => {
  if (!sender) return "";
  const s = String(sender).trim();
  const m = s.match(/^"?([^"<]+?)"?\s*<.+>$/);
  if (m) return m[1].trim();
  if (s.includes("@")) {
    const local = s.split("@")[0];
    return local.split(/[._]/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  }
  return s;
};

// ── Formatters ────────────────────────────────────────────────────────────────

export const escapeHTML = (s) =>
  String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

export const fmtNumber = (n) => {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (typeof n !== "number") return String(n);
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
};

export const fmtCurrencyARS = (n) => {
  if (n === null || n === undefined) return "—";
  return "$" + Math.round(n).toLocaleString("es-AR");
};

export const fmtTimeAgo = (iso) => {
  if (!iso) return "—";
  const ms = Date.now() - new Date(iso).getTime();
  const min = Math.floor(ms / 60_000);
  if (min < 1) return "ahora";
  if (min < 60) return `${min}m`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d`;
};

export const ageBucket = (hours) => {
  if (hours < 24 * 7) return { kind: "ok",   text: hours < 24 ? `${Math.round(hours)}h` : `${Math.round(hours / 24)}d` };
  if (hours < 24 * 30) return { kind: "warn", text: `${Math.round(hours / 24)}d` };
  return { kind: "stale", text: `STALE ${Math.round(hours / 24)}d` };
};

// ── URL builders ──────────────────────────────────────────────────────────────

// Cache del payload actual para que obsidianUrl() funcione sin pasar
// el payload explícitamente desde cada renderer.
let _currentPayload = null;

export function setCurrentPayload(p) {
  _currentPayload = p;
}

export function getCurrentPayload() {
  return _currentPayload;
}

export function obsidianUrl(path, vaultAlias) {
  if (!path) return null;
  const dirNames = _currentPayload?.vault_dir_names || {};
  const dirName = dirNames[vaultAlias || ""]
    || dirNames.home
    || vaultAlias
    || "Notes";
  return `obsidian://open?vault=${encodeURIComponent(dirName)}`
    + `&file=${encodeURIComponent(path)}`;
}

export function gmailThreadUrl(threadId) {
  if (!threadId) return null;
  return `https://mail.google.com/mail/u/0/#inbox/${encodeURIComponent(threadId)}`;
}

export function whatsappUrl(jid) {
  if (!jid) return null;
  const s = String(jid);
  if (!/@(s\.whatsapp\.net|c\.us)$/.test(s)) return null;
  const phone = s.split("@")[0].replace(/\D/g, "");
  if (!phone || phone.length < 7) return null;
  return `https://wa.me/${phone}`;
}

export function youtubeUrl(videoId) {
  if (!videoId) return null;
  return `https://youtube.com/watch?v=${encodeURIComponent(videoId)}`;
}

// ── Panel list renderer ────────────────────────────────────────────────────────

// Rellena un panel id `panelId` con una lista de rows.
// Cada row: { title, meta, aside, href }
export function renderPanelList(panelId, rows, opts = {}) {
  const panel = document.getElementById(panelId);
  if (!panel) return 0;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
  const foot = panel.querySelector("[data-foot]");
  const { emptyText = "sin items", capChip = "info", showCount = true, footText = "" } = opts;

  if (!rows || !rows.length) {
    panel.classList.add("is-empty");
    body.innerHTML = `<div class="empty">${escapeHTML(emptyText)}</div>`;
    if (showCount) count.textContent = "0";
    if (foot) foot.textContent = "";
    return 0;
  }
  panel.classList.remove("is-empty");
  if (showCount) {
    count.textContent = rows.length;
    count.classList.remove("has-items", "has-warning", "has-critical");
    count.classList.add(
      capChip === "critical" ? "has-critical"
      : capChip === "warning" ? "has-warning"
      : "has-items",
    );
  }
  body.innerHTML = rows.map((r) => {
    const aside = r.aside ? `<span class="row-aside">${escapeHTML(r.aside)}</span>` : "";
    const meta = r.meta && r.meta.length
      ? `<div class="row-meta">${r.meta.map((m) =>
          typeof m === "string"
            ? escapeHTML(m)
            : `<span class="${m.cls || ""}">${escapeHTML(m.text)}</span>`,
        ).join(" · ")}</div>`
      : "";
    const titleHTML = escapeHTML(r.title);
    const inner = `<div class="row-main">
        <div class="row-title">${titleHTML}</div>
        ${meta}
      </div>
      ${aside}`;
    if (r.href) {
      const href = String(r.href);
      const isExternal = /^https?:/.test(href);
      const target = isExternal ? ' target="_blank" rel="noopener"' : "";
      return `<a class="row row--linked" href="${escapeHTML(href)}"${target}>${inner}</a>`;
    }
    return `<div class="row">${inner}</div>`;
  }).join("");
  if (foot) foot.innerHTML = footText || "";
  return rows.length;
}

// ── KPI helper ────────────────────────────────────────────────────────────────

export function setKPI(id, { value, label, meta, tone, trend }) {
  const el = document.getElementById(id);
  if (!el) return;
  const numeric = Number(value);
  const isZeroOrEmpty =
    value == null ||
    value === "" ||
    value === "—" ||
    (Number.isFinite(numeric) && numeric === 0);
  el.hidden = isZeroOrEmpty;
  if (isZeroOrEmpty) return;
  el.classList.remove("is-critical", "is-warning", "is-ok");
  if (tone === "critical") el.classList.add("is-critical");
  else if (tone === "warning") el.classList.add("is-warning");
  else if (tone === "ok") el.classList.add("is-ok");
  el.querySelector("[data-value]").textContent = value;
  const metaEl = el.querySelector("[data-meta]");
  if (trend) {
    metaEl.innerHTML = `<span class="kpi-trend ${trend.dir}">${trend.text}</span>`;
  } else {
    metaEl.textContent = meta || "";
  }
}

// ── Auto-refresh ───────────────────────────────────────────────────────────────

// Inicia el ciclo de refresh cada `intervalMs` ms (default 5 min).
// `fetchFn` es la función async que fetchea y re-renderiza.
export function startAutoRefresh(fetchFn, intervalMs = 5 * 60 * 1000) {
  setInterval(fetchFn, intervalMs);
}
