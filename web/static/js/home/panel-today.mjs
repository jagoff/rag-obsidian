// panel-today.mjs — Today hero (bloque grande con 4 sub-secciones del día).
// Incluye: highlights cross-source, narrative LLM, patrones de actividad,
// inbox, preguntas abiertas, agenda de hoy y de mañana.

import {
  escapeHTML, fmtTimeAgo,
  obsidianUrl, gmailThreadUrl, whatsappUrl,
  getCurrentPayload,
  isBotOrSelf,
  isReminderDueToday, isReminderDueTomorrow, reminderTitle,
} from "./core.mjs";
import {
  readJSON,
  readObject,
  readString,
  removeKey,
  writeJSON,
  writeObjectOrRemove,
  writeString,
} from "../layout-persistence.mjs?v=103";

// ── Wikilinks → Markdown links ────────────────────────────────────────────────

export function wikilinksToMarkdown(s) {
  if (!s) return "";
  return s
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, (_, target, label) => {
      const url = obsidianUrl(target.trim());
      return url ? `[${label.trim()}](${url})` : label.trim();
    })
    .replace(/\[\[([^\]]+)\]\]/g, (_, target) => {
      const t = target.trim();
      const url = obsidianUrl(t);
      return url ? `[${t}](${url})` : t;
    });
}

export function stripWikilinks(s) {
  if (!s) return "";
  return s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2")
          .replace(/\[\[([^\]]+)\]\]/g, "$1");
}

// ── Sanitizador mínimo del output de marked ────────────────────────────────────

export function sanitizeHTML(html) {
  return String(html)
    .replace(/<\s*script[\s\S]*?<\/\s*script\s*>/gi, "")
    .replace(/\son\w+\s*=\s*"[^"]*"/gi, "")
    .replace(/\son\w+\s*=\s*'[^']*'/gi, "")
    .replace(/javascript:/gi, "");
}

export function mdToHTML(md) {
  if (!md) return "";
  const withLinks = wikilinksToMarkdown(md);
  if (window.marked) {
    try {
      let html = window.marked.parse(withLinks);
      html = sanitizeHTML(html);
      html = html.replace(
        /<a\s+href="(obsidian:\/\/[^"]+)"/g,
        '<a class="wikilink" href="$1"',
      );
      html = html.replace(
        /<a\s+href="(https?:\/\/[^"]+)"(?![^>]*target=)/g,
        '<a href="$1" target="_blank" rel="noopener"',
      );
      return html;
    } catch (e) {
      console.warn("[home.v2] marked.parse failed:", e);
    }
  }
  return `<pre style="white-space:pre-wrap;font-family:inherit;margin:0;">${escapeHTML(stripWikilinks(md))}</pre>`;
}

// ── Splittear narrative en sub-secciones ──────────────────────────────────────

export function splitNarrative(md) {
  const out = { narrative: "", inbox: "", questions: "", today: "", tomorrow: "" };
  if (!md) return out;
  const chunks = md.split(/^##\s+/m).map((c, i) => i === 0 ? c : "## " + c);
  for (const chunk of chunks) {
    const headerMatch = chunk.match(/^##\s+(.+?)\n([\s\S]*)$/);
    if (!headerMatch) continue;
    const header = headerMatch[1].toLowerCase();
    const body = headerMatch[2].trim();
    // "Para mañana" PRIMERO para evitar que matchee con "hoy"
    if (/ma[ñn]ana|tomorrow/.test(header)) {
      out.tomorrow = body;
    } else if (/para\s*hoy|agenda\s*hoy|🌅.*hoy|hoy\b.*agenda/.test(header)) {
      out.today = body;
    } else if (/lo que pas[oó]|qu[eé] pas[oó]/.test(header)) {
      out.narrative = body;
    } else if (/sin procesar|inbox|por procesar/.test(header)) {
      out.inbox = body;
    } else if (/pregunta|abierta|open|low.?conf/.test(header)) {
      out.questions = body;
    }
  }
  return out;
}

// ── Highlights (chips de actividad cross-source) ──────────────────────────────

export function renderHighlights(h) {
  if (!h || typeof h !== "object") return "";
  const chips = [];
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");
  const usefulTopic = (topic) => {
    const t = String(topic || "").trim().toLowerCase();
    if (!t || t.length < 4) return false;
    return !/^(estoy|esta|este|eso|algo|tema|cosas?|info|notas?|hoy|ayer|mañana|mail|gmail|whatsapp)$/.test(t);
  };

  // Chip 1: top WhatsApp chat del día
  if (h.wa_top_chat && h.wa_top_chat.count > 0) {
    const c = h.wa_top_chat;
    const url = whatsappUrl(c.jid);
    const inner = `
      <span class="hl-icon">💬</span>
      <span class="hl-label">${escapeHTML(c.name || "?")}</span>
      <span class="hl-value">${c.count}</span>
    `;
    chips.push(url
      ? `<a class="hl-chip is-link" href="${escapeHTML(url)}" title="${escapeHTML(c.last_snippet || "")}">${inner}</a>`
      : `<div class="hl-chip" title="${escapeHTML(c.last_snippet || "")}">${inner}</div>`);
  }

  // Chip 2: resumen volumen WA
  if ((h.wa_total_msgs || 0) > 0 && (h.wa_active_chats || 0) > 1) {
    chips.push(`
      <div class="hl-chip" title="suma de inbound en los chats activos del día">
        <span class="hl-icon">📊</span>
        <span class="hl-label">${h.wa_active_chats} chats</span>
        <span class="hl-value">${h.wa_total_msgs}</span>
      </div>`);
  }

  // Chip 3: gmail unread + awaiting reply
  const gmailTotal = (h.gmail_unread || 0) + (h.gmail_awaiting_reply || 0);
  if (gmailTotal > 0) {
    const parts = [];
    if (h.gmail_unread) parts.push(`${h.gmail_unread} sin leer`);
    if (h.gmail_awaiting_reply) parts.push(`${h.gmail_awaiting_reply} esperan resp`);
    chips.push(`
      <a class="hl-chip is-link" href="https://mail.google.com/mail/u/0/#inbox" target="_blank" rel="noopener" title="${escapeHTML(parts.join(" · "))}">
        <span class="hl-icon">📧</span>
        <span class="hl-label">Mails</span>
        <span class="hl-value">${gmailTotal}</span>
      </a>`);
  }

  // Chip 4: reuniones del día
  if ((h.calendar_events || 0) > 0) {
    chips.push(`
      <div class="hl-chip">
        <span class="hl-icon">📅</span>
        <span class="hl-label">Reuniones</span>
        <span class="hl-value">${h.calendar_events}</span>
      </div>`);
  }

  // Chip 5: videos vistos
  if ((h.youtube_videos || 0) > 0) {
    chips.push(`
      <div class="hl-chip">
        <span class="hl-icon">📺</span>
        <span class="hl-label">YouTube</span>
        <span class="hl-value">${h.youtube_videos}</span>
      </div>`);
  }

  // Chip 6: notas del vault tocadas hoy
  if ((h.vault_notes_today || 0) > 0) {
    chips.push(`
      <div class="hl-chip">
        <span class="hl-icon">📝</span>
        <span class="hl-label">Notas</span>
        <span class="hl-value">${h.vault_notes_today}</span>
      </div>`);
  }

  // Chip 7: top persona cross-source
  if (h.top_person && h.top_person.sources_count >= 2) {
    const p = h.top_person;
    const sources = (p.appearances || []).map((a) => a.source).join(" + ");
    const waApp = (p.appearances || []).find((a) => a.source === "whatsapp");
    let waJid = null;
    if (waApp) {
      const waList = getCurrentPayload()?.signals?.whatsapp || [];
      const match = waList.find((w) => w.name === waApp.display_name);
      if (match) waJid = match.jid;
    }
    const url = waJid ? whatsappUrl(waJid) : null;
    const inner = `
      <span class="hl-icon">👥</span>
      <span class="hl-label">${escapeHTML(p.name)}</span>
      <span class="hl-meta">${escapeHTML(sources)}</span>
    `;
    chips.push(url
      ? `<a class="hl-chip hl-chip--persona is-link" href="${escapeHTML(url)}" title="aparece en ${escapeHTML(sources)}">${inner}</a>`
      : `<div class="hl-chip hl-chip--persona" title="aparece en ${escapeHTML(sources)}">${inner}</div>`);
  }

  // Chip 8: top topic cross-source
  if (
    h.top_topic
    && usefulTopic(h.top_topic.topic)
    && (h.top_topic.sources_count || (h.top_topic.sources || []).length) >= 2
  ) {
    const t = h.top_topic;
    const sources = (t.sources || []).join(" + ");
    chips.push(`
      <div class="hl-chip hl-chip--topic" title="tema en ${escapeHTML(sources)}">
        <span class="hl-icon">🎯</span>
        <span class="hl-label">${escapeHTML(t.topic)}</span>
        <span class="hl-meta">${escapeHTML(sources)}</span>
      </div>`);
  }

  if (!chips.length) return "";
  return `<div class="hero-highlights">${chips.join("")}</div>`;
}

// ── Patrones de actividad cross-source ────────────────────────────────────────

export function renderPatternsSub(p) {
  if (!p || typeof p !== "object") return "";
  const cards = [];
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

  // Spikes — chats WA hot del día
  const spikes = (p.spikes || []).filter((s) => Number(s.avg_7d || 0) > 0);
  if (spikes.length) {
    const items = spikes.map((s) => {
      const url = whatsappUrl(s.jid);
      const inner = `
        <strong>${escapeHTML(s.name)}</strong>
        <span class="ptn-num">${s.today}</span>
        <span class="ptn-meta">vs ${s.avg_7d}/d · ×${s.ratio}</span>
      `;
      return url
        ? `<li><a href="${escapeHTML(url)}" title="${escapeHTML(s.last_snippet || "")}">${inner}</a></li>`
        : `<li>${inner}</li>`;
    }).join("");
    cards.push(`
      <div class="ptn-card ptn-card--spike">
        <h4><span>⚡</span> Activity alta</h4>
        <ul>${items}</ul>
      </div>`);
  }

  // Silencios — gente que escribía y hoy no
  if ((p.silences || []).length) {
    const items = p.silences.map((s) => {
      const url = whatsappUrl(s.jid);
      const hours = s.hours_silent;
      const ago = hours == null ? ""
        : hours < 24 ? `${Math.round(hours)}h`
        : `${(hours / 24).toFixed(1)}d`;
      const inner = `
        <strong>${escapeHTML(s.name)}</strong>
        <span class="ptn-meta">${s.msgs_7d}msg/7d · silente ${ago}</span>
      `;
      return url
        ? `<li><a href="${escapeHTML(url)}">${inner}</a></li>`
        : `<li>${inner}</li>`;
    }).join("");
    cards.push(`
      <div class="ptn-card ptn-card--silence">
        <h4><span>🔇</span> Silencios</h4>
        <ul>${items}</ul>
      </div>`);
  }

  // Concentraciones — temas en ≥3 fuentes
  if ((p.concentrations || []).length) {
    const items = p.concentrations.map((c) => `
      <li>
        <strong>${escapeHTML(c.topic)}</strong>
        <span class="ptn-meta">${escapeHTML((c.sources || []).join(" + "))}</span>
      </li>`).join("");
    cards.push(`
      <div class="ptn-card ptn-card--concentration">
        <h4><span>🎯</span> Concentración temática</h4>
        <ul>${items}</ul>
      </div>`);
  }

  // Gaps — gente que te escribió y no tenés slot mañana
  if ((p.gaps || []).length) {
    const items = p.gaps.slice(0, 4).map((g) => {
      const hours = g.hours_waiting || 0;
      const ago = hours < 24 ? `${Math.round(hours)}h`
        : `${(hours / 24).toFixed(1)}d`;
      return `
        <li>
          <strong>${escapeHTML(g.person || "?")}</strong>
          <span class="ptn-meta">hace ${ago} · "${escapeHTML(truncate(g.snippet || "", 60))}"</span>
        </li>`;
    }).join("");
    cards.push(`
      <div class="ptn-card ptn-card--gap">
        <h4><span>❗</span> Sin slot mañana</h4>
        <ul>${items}</ul>
      </div>`);
  }

  if (!cards.length) return "";
  return `<div class="hero-patterns">${cards.join("")}</div>`;
}

function shouldRenderNarrative(payload, md) {
  if (!String(md || "").trim()) return false;
  const source = payload.today?.narrative_source || "";
  if (source === "cached" || source === "stale") return false;
  const text = String(md).toLowerCase();
  return !/mercado\s*libre|meli\+|netflix|pinterest|youtube|prueba gratis|receipt|invoice|factura|comprobante/.test(text);
}

// ── Render del today hero ─────────────────────────────────────────────────────

// Set de textos de reminder creados en esta sesión (evita doble-create en re-renders).
const _createdReminderTexts = new Set();

export function renderTodayHero(payload) {
  const dateEl = document.getElementById("hero-date");
  const countsEl = document.getElementById("hero-counts");
  const bodyEl = document.getElementById("today-hero-body");
  if (!dateEl || !bodyEl) return;

  // Estas cajas pueden haber sido movidas a otros contenedores por el layout
  // persistido. Cada refresh reconstruye el hero desde el payload; removemos
  // las instancias viejas para no dejar IDs duplicados antes de aplicar el
  // orden global desde layout.mjs.
  document
    .querySelectorAll("#home-cmdbar > .hero-section, .section-body > .hero-section")
    .forEach((sec) => sec.remove());
  const preservedHeroBodyItems = document.createDocumentFragment();
  Array.from(bodyEl.children).forEach((el) => {
    if (el.classList?.contains("panel") || el.classList?.contains("kpi")) {
      preservedHeroBodyItems.appendChild(el);
    }
  });

  dateEl.textContent = payload.date || "—";

  const evidence = payload.today?.evidence || {};
  const signals = payload.signals || {};
  const heroSection = bodyEl.closest(".today-hero");
  const inboxItems = evidence.inbox_today || [];
  const lowConfItems = evidence.low_conf_queries || [];
  const tomorrowEvents = Array.isArray(payload.tomorrow_calendar)
    ? payload.tomorrow_calendar
    : [];
  const todayEvents = Array.isArray(payload.today_calendar)
    ? payload.today_calendar
    : [];

  const rawNarrative = payload.today?.narrative || "";
  const md = shouldRenderNarrative(payload, rawNarrative) ? rawNarrative : "";
  const split = splitNarrative(md);

  // Si el cuadro queda sin contenido real, NO lo renderizamos — preferimos
  // ocultar a mostrar un placeholder vacío ("Aún sin brief…", "todo
  // procesado ✓"). Aplica a las hero-sections regulares; agenda tiene su
  // propio gate más abajo porque su body tiene siempre los dos sub-bloques.
  const sectionHTML = (cls, emoji, label, count, contentHTML, _emptyText) => {
    const trimmed = (contentHTML || "").trim();
    if (!trimmed) return "";
    return `
    <div class="hero-section ${cls}" id="hero-${cls}">
      <h3><span>${emoji} ${escapeHTML(label)}</span>${count != null ? `<span class="count">${count}</span>` : ""}</h3>
      <div class="prose">${trimmed}</div>
    </div>
  `;
  };

  const fromName = (s) => (s || "").split("<")[0].trim() || s || "";
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");
  const dateKey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const todayKey = dateKey(new Date());
  const isTodayValue = (value) => {
    if (!value) return false;
    const d = new Date(value);
    return !Number.isNaN(d.getTime()) && dateKey(d) === todayKey;
  };
  const isNoiseMail = (m) => {
    const text = `${m.from || ""} ${m.subject || ""}`.toLowerCase();
    return isBotOrSelf(m.from)
      || /\b(receipt|invoice|factura|comprobante)\b/.test(text)
      || /\b(compraste|prueba gratis|bienvenido al plan)\b/.test(text)
      || /mercado\s*libre|meli\+|mail\.anthropic\.com/.test(text);
  };
  const isUsefulMail = (m) =>
    !isNoiseMail(m) && (!m.internal_date_ms || isTodayValue(Number(m.internal_date_ms)));

  const vaultsInPlay = new Set();
  for (const item of [...(evidence.recent_notes || []), ...inboxItems, ...(evidence.todos || [])]) {
    if (item && item.vault) vaultsInPlay.add(item.vault);
  }
  const showVaultTag = vaultsInPlay.size > 1;
  const vaultTag = (item) => (showVaultTag && item?.vault)
    ? ` <span class="vault-tag">[${escapeHTML(item.vault)}]</span>`
    : "";

  const highlights = { ...(payload.today?.highlights || {}) };
  highlights.youtube_videos = (signals.youtube_watched || [])
    .filter((v) => isTodayValue(v.last_visit_iso))
    .length;
  const patterns = payload.today?.patterns || {};
  let narrativeHTML = "";

  if (split.narrative) {
    narrativeHTML += renderHighlights(highlights);
    narrativeHTML += `<div class="hero-prose">${mdToHTML(split.narrative)}</div>`;
    narrativeHTML += renderPatternsSub(patterns);
  }

  // Sub-section: inbox
  let inboxHTML = "";
  if (split.inbox) {
    inboxHTML = mdToHTML(split.inbox);
  } else {
    const items = [];
    const wrapLink = (url, innerHTML, isObsidian = false) => {
      if (!url) return innerHTML;
      const target = /^https?:/.test(url) ? ' target="_blank" rel="noopener"' : "";
      const cls = isObsidian ? ' class="wikilink"' : "";
      return `<a${cls} href="${escapeHTML(url)}"${target}>${innerHTML}</a>`;
    };
    for (const it of inboxItems.slice(0, 3)) {
      const tags = (it.tags || []).slice(0, 3)
        .map((t) => `<code>#${escapeHTML(t)}</code>`).join(" ");
      const inner = `📝 <strong>${escapeHTML(it.title || it.path)}</strong>${vaultTag(it)}${tags ? " " + tags : ""}`;
      items.push(`<li>${wrapLink(obsidianUrl(it.path, it.vault), inner, true)}</li>`);
    }
    for (const m of (signals.gmail?.recent || []).filter(isUsefulMail).slice(0, 3)) {
      const inner = `📧 <strong>${escapeHTML(fromName(m.from))}</strong>: ${escapeHTML(truncate(m.subject || "", 80))}`;
      items.push(`<li>${wrapLink(gmailThreadUrl(m.thread_id), inner)}</li>`);
    }
    const mailUnread = (signals.mail_unread || []).slice(0, 2);
    if (!signals.gmail?.recent?.length && mailUnread.length) {
      for (const m of mailUnread) {
        items.push(`<li>📬 <strong>${escapeHTML(fromName(m.from || m.sender))}</strong>: ${escapeHTML(truncate(m.subject || "", 80))}</li>`);
      }
    }
    inboxHTML = items.length ? `<ul>${items.join("")}</ul>` : "";
  }

  // Sub-section: preguntas abiertas
  let questionsHTML = "";
  if (split.questions) {
    questionsHTML = mdToHTML(split.questions);
  } else {
    const items = [];
    const wrapLink = (url, innerHTML, isObsidian = false) => {
      if (!url) return innerHTML;
      const target = /^https?:/.test(url) ? ' target="_blank" rel="noopener"' : "";
      const cls = isObsidian ? ' class="wikilink"' : "";
      return `<a${cls} href="${escapeHTML(url)}"${target}>${innerHTML}</a>`;
    };
    for (const q of lowConfItems.slice(0, 3)) {
      const text = q.q || q.question || q.text || "";
      const rawScore = q.score ?? q.top_score;
      const scoreNum = Number(rawScore);
      const score = Number.isFinite(scoreNum)
        ? ` <em>(${scoreNum.toFixed(2)})</em>`
        : "";
      const inner = `❓ ${escapeHTML(text)}${score}`;
      const url = text ? `/chat?q=${encodeURIComponent(text)}` : null;
      items.push(`<li>${wrapLink(url, inner)}</li>`);
    }
    questionsHTML = items.length ? `<ul>${items.join("")}</ul>` : "";
  }

  const renderAgendaItems = (events, reminders, fallbackPrefix) => {
    const items = [];
    for (const e of events.slice(0, 6)) {
      const tr = (e.time_range || "").trim();
      const prefix = tr || fallbackPrefix;
      const title = e.title || e.summary || "(sin título)";
      items.push(`<li><strong>${escapeHTML(prefix)}</strong> — ${escapeHTML(title)}</li>`);
    }
    for (const r of reminders.slice(0, 3)) {
      items.push(`<li>📌 ${escapeHTML(truncate(reminderTitle(r), 80))}</li>`);
    }
    return items.length ? `<ul>${items.join("")}</ul>` : "";
  };

  let todayHTML = "";
  if (split.today) {
    todayHTML = mdToHTML(split.today);
  } else {
    const reminders = (signals.reminders || []).filter(isReminderDueToday);
    todayHTML = renderAgendaItems(todayEvents, reminders, "todo el día");
  }

  const tomorrowBriefIsStale = split.tomorrow
    && tomorrowEvents.length === 0
    && todayEvents.length > 0;
  let tomorrowHTML = "";
  if (split.tomorrow && !tomorrowBriefIsStale) {
    tomorrowHTML = mdToHTML(split.tomorrow);
  } else {
    const reminders = (signals.reminders || []).filter(isReminderDueTomorrow);
    tomorrowHTML = renderAgendaItems(tomorrowEvents, reminders, "todo el día");
  }

  const inboxCount = inboxItems.length
    + ((signals.gmail?.recent || []).filter(isUsefulMail).length);
  const questionsCount = lowConfItems.length;
  const todayCount = todayEvents.length
    + (signals.reminders || []).filter(isReminderDueToday).length;
  const tomorrowCount = tomorrowEvents.length
    + (signals.reminders || []).filter(isReminderDueTomorrow).length;

  // Agenda combina "Para hoy" + "Para mañana" en una sola hero-section
  // (col 4 de la grilla). Se renderea como h3 + dos sub-blocks .hero-sub
  // con su propio h4 + body. El reminder injector matchea `.s-tomorrow
  // .prose` que sigue válido porque cada hero-sub mantiene su class
  // s-today / s-tomorrow y su propio .prose adentro.
  const agendaBody = `
    <div class="hero-sub s-today">
      <h4><span>🌅 Para hoy</span>${todayCount != null ? `<span class="count">${todayCount}</span>` : ""}</h4>
      <div class="prose">${todayHTML || `<div class="empty">agenda libre hoy · día abierto</div>`}</div>
    </div>
    <div class="hero-sub s-tomorrow">
      <h4><span>🌅 Para mañana</span>${tomorrowCount != null ? `<span class="count">${tomorrowCount}</span>` : ""}</h4>
      <div class="prose">${tomorrowHTML || `<div class="empty">agenda libre mañana · día abierto</div>`}</div>
    </div>
  `;
  const agendaCount = (todayCount || 0) + (tomorrowCount || 0);
  // Misma regla "hide if empty" para agenda: cuando ningún sub-bloque tiene
  // items reales (todayHTML/tomorrowHTML vacíos), no la renderizamos.
  const agendaHasContent = !!((todayHTML || "").trim() || (tomorrowHTML || "").trim());
  const agendaHTML = agendaHasContent ? `
    <div class="hero-section s-agenda" id="hero-s-agenda">
      <h3><span>🌅 Agenda</span>${agendaCount ? `<span class="count">${agendaCount}</span>` : ""}</h3>
      ${agendaBody}
    </div>
  ` : "";

  const heroPieces = [
    sectionHTML("s-narrative", "🪞", "Lo que pasó hoy", null, narrativeHTML, ""),
    sectionHTML("s-inbox", "📥", "Sin procesar", inboxCount || null, inboxHTML, ""),
    sectionHTML("s-questions", "🔍", "Preguntas abiertas", questionsCount || null, questionsHTML, ""),
    agendaHTML,
  ].filter((s) => (s || "").trim().length > 0);
  const actionableCount = (split.narrative ? 1 : 0)
    + (inboxCount || 0)
    + (questionsCount || 0)
    + (todayCount || 0)
    + (tomorrowCount || 0);
  countsEl.textContent = heroPieces.length
    ? `${actionableCount} accionables · ${inboxItems.length} inbox · ${questionsCount} preguntas`
    : "";

  const hasPreservedHeroItems = preservedHeroBodyItems.childNodes.length > 0;
  if (heroSection) heroSection.hidden = heroPieces.length === 0 && !hasPreservedHeroItems;
  // Cuando TODOS los cuadros están vacíos, mostrar un solo placeholder para
  // el caso pre-brief (no dejar el hero completamente en blanco sin pista
  // de cómo arrancarlo).
  if (heroPieces.length === 0) {
    bodyEl.innerHTML = "";
  } else {
    bodyEl.innerHTML = heroPieces.join("");
  }
  if (preservedHeroBodyItems.firstChild) {
    bodyEl.appendChild(preservedHeroBodyItems);
  }

  // Inyectar botones inline "crear reminder" en cada <li> de "Para mañana"
  _injectTomorrowReminderButtons(_createdReminderTexts);

  // Aplicar orden persistido (drag) + estado collapse persistido (LS) y
  // re-wirear handlers. El hero re-renderiza innerHTML cada refresh, así
  // que cada render pierde el wiring previo — hay que volver a engancharlo.
  applyHeroOrder();
  wireHeroLayout();
  applyHeroSubCollapse();
}

// ── Hero subs: persistencia drag-order + collapse ─────────────────────────────
//
// El hero-body re-genera innerHTML cada refresh (ver renderTodayHero arriba),
// así que el orden y el estado de collapse de cada hero-section NO sobreviven
// el ciclo render por sí solos. La estrategia es:
//   1. applyHeroOrder()        → reordena los div.hero-section según LS antes
//                                 de exponerlos al user.
//   2. wireHeroLayout()        → inyecta grip de drag + botón collapse,
//                                 attachea handlers HTML5 DnD.
//   3. applyHeroSubCollapse()  → setea data-collapsed="true" en cajas
//                                 colapsadas según LS.
// Estas 3 corren al final de renderTodayHero, por eso son idempotentes y
// chequean estado previo antes de duplicar nodos / handlers.

const LS_HERO_ORDER = "home.v2.hero-subs.order.v1";
// v1 → v2 (2026-05-11): el estado colapsado heredado dejaba a usuarios
// con boxes "vacíos" visualmente (h3 visible + body oculto por CSS
// `[data-collapsed="true"] .prose`) sin pista de cómo expandir. Bumpear
// la key invalida el estado viejo — las cajas empiezan expandidas por
// default; el user puede re-colapsar si quiere.
const LS_HERO_SUB_COLLAPSED = "home.v2.hero-subs.collapsed.v2";

// Clase identificadora estable por hero-section (se persiste en LS).
const HERO_SECTION_KEYS = ["s-narrative", "s-inbox", "s-questions", "s-agenda"];

function _heroSectionKey(el) {
  for (const k of HERO_SECTION_KEYS) {
    if (el.classList.contains(k)) return k;
  }
  return null;
}

function _readHeroOrder() {
  const parsed = readJSON(LS_HERO_ORDER, null);
  return Array.isArray(parsed) ? parsed : null;
}

function _saveHeroOrder() {
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  const order = Array.from(body.querySelectorAll(":scope > .hero-section"))
    .map(_heroSectionKey)
    .filter(Boolean);
  if (!writeJSON(LS_HERO_ORDER, order)) {
    console.warn("[home.v2] no pude persistir hero order");
  }
  try { window._updateResetButtonVisibility?.(); } catch {}
}

export function applyHeroOrder() {
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  const saved = _readHeroOrder();
  if (!saved) return;
  for (const key of saved) {
    const el = body.querySelector(`:scope > .hero-section.${key}`);
    if (el) body.appendChild(el);
  }
}

function _readHeroSubCollapsed() {
  return readObject(LS_HERO_SUB_COLLAPSED, {});
}

function _writeHeroSubCollapsed(map) {
  const trimmed = {};
  for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
  if (!writeObjectOrRemove(LS_HERO_SUB_COLLAPSED, trimmed)) {
    console.warn("[home.v2] no pude persistir hero sub collapse");
  }
  try { window._updateResetButtonVisibility?.(); } catch {}
}

export function applyHeroSubCollapse() {
  const map = _readHeroSubCollapsed();
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  body.querySelectorAll(":scope > .hero-section").forEach((sec) => {
    const key = _heroSectionKey(sec);
    if (!key) return;
    const collapsed = !!map[key];
    sec.setAttribute("data-collapsed", collapsed ? "true" : "false");
    const btn = sec.querySelector(":scope > h3 > .hero-collapse-btn");
    if (btn) {
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      const icon = btn.querySelector(".toggle-icon");
      if (icon) icon.textContent = collapsed ? "▶" : "▼";
    }
  });
}

function _toggleHeroSubCollapse(sec) {
  const key = _heroSectionKey(sec);
  if (!key) return;
  const collapsed = sec.getAttribute("data-collapsed") === "true";
  const next = !collapsed;
  sec.setAttribute("data-collapsed", next ? "true" : "false");
  const btn = sec.querySelector(":scope > h3 > .hero-collapse-btn");
  if (btn) {
    btn.setAttribute("aria-expanded", next ? "false" : "true");
    const icon = btn.querySelector(".toggle-icon");
    if (icon) icon.textContent = next ? "▶" : "▼";
  }
  const map = _readHeroSubCollapsed();
  if (next) map[key] = true;
  else delete map[key];
  _writeHeroSubCollapsed(map);
}

// ── Hero subs: drag & drop ────────────────────────────────────────────────────

let _draggingHero = null;

function _heroDropPlacement(ev, el) {
  const rect = el.getBoundingClientRect();
  const x = rect.width ? (ev.clientX - rect.left) / rect.width : 0.5;
  const y = rect.height ? (ev.clientY - rect.top) / rect.height : 0.5;
  if (y < 0.33) return { before: true, axis: "y" };
  if (y > 0.67) return { before: false, axis: "y" };
  return { before: x < 0.5, axis: "x" };
}

function _setHeroDropMark(el, placement) {
  const before = !!placement?.before;
  const axis = placement?.axis === "y" ? "y" : "x";
  el.classList.toggle("drop-before", before);
  el.classList.toggle("drop-after", !before);
  el.classList.toggle("drop-axis-x", axis === "x");
  el.classList.toggle("drop-axis-y", axis === "y");
}

function _clearHeroDropMark(el) {
  el.classList.remove("drop-before", "drop-after", "drop-axis-x", "drop-axis-y");
}

function _heroSections(body) {
  return Array.from(body?.querySelectorAll(":scope > .hero-section") || [])
    .filter((el) => el !== _draggingHero);
}

function _heroInsertionFromPoint(body, ev) {
  const items = _heroSections(body)
    .map((el) => ({ el, rect: el.getBoundingClientRect() }))
    .sort((a, b) => {
      const dy = a.rect.top - b.rect.top;
      if (Math.abs(dy) > 8) return dy;
      return a.rect.left - b.rect.left;
    });
  const rows = [];
  for (const item of items) {
    let row = rows.find((r) => Math.abs(r.top - item.rect.top) <= 12);
    if (!row) {
      row = { top: item.rect.top, bottom: item.rect.bottom, items: [] };
      rows.push(row);
    }
    row.top = Math.min(row.top, item.rect.top);
    row.bottom = Math.max(row.bottom, item.rect.bottom);
    row.items.push(item);
  }
  for (const row of rows) {
    row.items.sort((a, b) => a.rect.left - b.rect.left);
    if (ev.clientY < row.top) {
      return { target: row.items[0].el, before: true, axis: "y" };
    }
    if (ev.clientY <= row.bottom) {
      for (const item of row.items) {
        const midX = item.rect.left + item.rect.width / 2;
        if (ev.clientX < midX) return { target: item.el, before: true, axis: "x" };
      }
      return { target: row.items[row.items.length - 1].el, before: false, axis: "x" };
    }
  }
  const last = items[items.length - 1]?.el || null;
  return { target: last, before: false, axis: "y" };
}

function _clearAllHeroDropMarks() {
  document.querySelectorAll(".hero-section.drop-before, .hero-section.drop-after")
    .forEach(_clearHeroDropMark);
}

function _moveHeroAtPlacement(body, placement) {
  if (!_draggingHero || !body || !placement?.target) return false;
  if (placement.target === _draggingHero) return false;
  if (placement.before) {
    body.insertBefore(_draggingHero, placement.target);
  } else {
    body.insertBefore(_draggingHero, placement.target.nextSibling);
  }
  _saveHeroOrder();
  return true;
}

function _onHeroDragStart(ev) {
  if (ev.target && ev.target.closest(
    ".hero-collapse-btn, .panel-resize-handle, button, input, select, textarea",
  )) {
    ev.preventDefault();
    return;
  }
  const sec = ev.currentTarget;
  _draggingHero = sec;
  sec.classList.add("is-dragging");
  try {
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/plain", _heroSectionKey(sec) || "");
  } catch {}
}

function _onHeroDragEnd() {
  if (_draggingHero) _draggingHero.classList.remove("is-dragging");
  _draggingHero = null;
  _clearAllHeroDropMarks();
}

function _onHeroDragOver(ev) {
  if (!_draggingHero) return;
  const sec = ev.currentTarget;
  if (sec === _draggingHero) return;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  _setHeroDropMark(sec, _heroDropPlacement(ev, sec));
}

function _onHeroDragLeave(ev) {
  const sec = ev.currentTarget;
  if (sec.contains(ev.relatedTarget)) return;
  _clearHeroDropMark(sec);
}

function _onHeroDrop(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  if (!_draggingHero) return;
  const target = ev.currentTarget;
  if (target === _draggingHero) return;
  const placement = _heroDropPlacement(ev, target);
  const before = placement.before;
  _clearHeroDropMark(target);
  if (before) {
    target.parentNode.insertBefore(_draggingHero, target);
  } else {
    target.parentNode.insertBefore(_draggingHero, target.nextSibling);
  }
  _saveHeroOrder();
}

function _onHeroBodyDragOver(ev) {
  if (!_draggingHero) return;
  const body = ev.currentTarget;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  if (ev.target !== body) return;
  _clearAllHeroDropMarks();
  const placement = _heroInsertionFromPoint(body, ev);
  if (placement.target) _setHeroDropMark(placement.target, placement);
}

function _onHeroBodyDrop(ev) {
  if (!_draggingHero) return;
  const body = ev.currentTarget;
  if (ev.target !== body) return;
  ev.preventDefault();
  _clearAllHeroDropMarks();
  _moveHeroAtPlacement(body, _heroInsertionFromPoint(body, ev));
}

export function wireHeroLayout() {
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  body.querySelectorAll(":scope > .hero-section").forEach((sec) => {
    // Collapse btn dentro del h3 — drag/resize/orden lo maneja layout.mjs
    // con el mismo motor que paneles y KPIs.
    const h3 = sec.querySelector(":scope > h3");
    if (h3 && !h3.querySelector(".hero-collapse-btn")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "hero-collapse-btn";
      btn.setAttribute("aria-label", "Colapsar/expandir caja");
      btn.setAttribute("aria-expanded", "true");
      btn.title = "Colapsar/expandir";
      btn.innerHTML = '<span class="toggle-icon" aria-hidden="true">▼</span>';
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        _toggleHeroSubCollapse(sec);
      });
      btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
      // Append al final del h3 — el count (si existe) tiene margin-left: auto
      // que empuja todo lo demás a la derecha, y el btn queda al final, a la
      // derecha del count.
      h3.appendChild(btn);
    }
    // Anchors/imgs dentro de `.prose` son draggable=true por default en
    // HTML5 — eso secuestra el dragstart de la caja cuando el user agarra
    // sobre el contenido. Forzar draggable=false en los hijos.
    // Re-corre cada render porque innerHTML del body se regenera.
    sec.querySelectorAll("a, img").forEach((el) => {
      el.setAttribute("draggable", "false");
    });
  });
}

// ── Botones inline para crear reminders ───────────────────────────────────────

export function _injectTomorrowReminderButtons(createdSet) {
  const root = document.querySelector(".s-tomorrow .prose");
  if (!root) return;
  const items = root.querySelectorAll("li");
  items.forEach((li) => {
    if (li.querySelector(".add-reminder-btn")) return;
    const original = (li.textContent || "").trim();
    if (!original) return;
    li.dataset.reminderText = original;
    const btn = document.createElement("button");
    btn.className = "add-reminder-btn";
    btn.type = "button";
    btn.title = "Crear Apple Reminder (mañana 9:00)";
    btn.textContent = "+ rec";
    li.appendChild(btn);
    if (createdSet.has(original)) {
      li.classList.add("reminder-created");
      btn.disabled = true;
      btn.title = "Ya agregado a Reminders";
    }
  });
}

// ── Handler click para los botones inline ─────────────────────────────────────

export function initReminderButtonHandler() {
  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".add-reminder-btn");
    if (!btn) return;
    ev.preventDefault();
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
    } catch (e) {
      btn.classList.add("err");
      btn.title = "Error: " + (e.message || String(e));
      btn.disabled = false;
    } finally {
      btn.classList.remove("loading");
    }
  });
}

// ── Hero collapse toggle ───────────────────────────────────────────────────────

const LS_HERO_COLLAPSED = "home.v2.hero.collapsed.v1";

export function initHeroCollapse() {
  const heroToggle = document.getElementById("hero-toggle");
  if (!heroToggle) return;
  const hero = heroToggle.closest(".today-hero");
  if (!hero) return;
  if (readString(LS_HERO_COLLAPSED) === "1") {
    hero.setAttribute("data-collapsed", "true");
    heroToggle.setAttribute("aria-expanded", "false");
  }
  heroToggle.addEventListener("click", () => {
    const collapsed = hero.getAttribute("data-collapsed") === "true";
    const next = !collapsed;
    hero.setAttribute("data-collapsed", next ? "true" : "false");
    heroToggle.setAttribute("aria-expanded", next ? "false" : "true");
    if (next) writeString(LS_HERO_COLLAPSED, "1");
    else removeKey(LS_HERO_COLLAPSED);
    try { window._updateResetButtonVisibility?.(); } catch {}
  });
}

export { LS_HERO_COLLAPSED, LS_HERO_ORDER, LS_HERO_SUB_COLLAPSED };
