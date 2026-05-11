// panel-today.mjs — Today hero (bloque grande con 4 sub-secciones del día).
// Incluye: highlights cross-source, narrative LLM, patrones de actividad,
// inbox, preguntas abiertas, agenda de hoy y de mañana.

import {
  escapeHTML, fmtTimeAgo,
  obsidianUrl, gmailThreadUrl, whatsappUrl, youtubeUrl,
  getCurrentPayload,
} from "./core.mjs";

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
  if (h.top_topic && (h.top_topic.sources_count || (h.top_topic.sources || []).length) >= 2) {
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
  if ((p.spikes || []).length) {
    const items = p.spikes.map((s) => {
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

// ── Render del today hero ─────────────────────────────────────────────────────

// Set de textos de reminder creados en esta sesión (evita doble-create en re-renders).
const _createdReminderTexts = new Set();

export function renderTodayHero(payload) {
  const dateEl = document.getElementById("hero-date");
  const countsEl = document.getElementById("hero-counts");
  const bodyEl = document.getElementById("today-hero-body");
  if (!dateEl || !bodyEl) return;

  dateEl.textContent = payload.date || "—";

  const counts = payload.today?.counts || {};
  const evidence = payload.today?.evidence || {};
  const signals = payload.signals || {};
  const inboxItems = evidence.inbox_today || [];
  const lowConfItems = signals.low_conf || [];
  const tomorrowEvents = Array.isArray(payload.tomorrow_calendar)
    ? payload.tomorrow_calendar
    : [];
  const todayEvents = Array.isArray(payload.today_calendar)
    ? payload.today_calendar
    : [];

  const totalThings = (counts.total) ||
    ((evidence.recent_notes?.length || 0) +
     (inboxItems.length) +
     (lowConfItems.length));
  countsEl.textContent = `${totalThings} señales · ${inboxItems.length} inbox · ${lowConfItems.length} preguntas`;

  const md = payload.today?.narrative || "";
  const split = splitNarrative(md);

  const sectionHTML = (cls, emoji, label, count, contentHTML, emptyText) => `
    <div class="hero-section ${cls}">
      <h3><span>${emoji} ${escapeHTML(label)}</span>${count != null ? `<span class="count">${count}</span>` : ""}</h3>
      <div class="prose">${contentHTML || `<div class="empty">${escapeHTML(emptyText)}</div>`}</div>
    </div>
  `;

  const fromName = (s) => (s || "").split("<")[0].trim() || s || "";
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

  const vaultsInPlay = new Set();
  for (const item of [...(evidence.recent_notes || []), ...inboxItems, ...(evidence.todos || [])]) {
    if (item && item.vault) vaultsInPlay.add(item.vault);
  }
  const showVaultTag = vaultsInPlay.size > 1;
  const vaultTag = (item) => (showVaultTag && item?.vault)
    ? ` <span class="vault-tag">[${escapeHTML(item.vault)}]</span>`
    : "";

  const highlights = payload.today?.highlights || {};
  const patterns = payload.today?.patterns || {};
  let narrativeHTML = renderHighlights(highlights);

  if (split.narrative) {
    narrativeHTML += `<div class="hero-prose">${mdToHTML(split.narrative)}</div>`;
  } else {
    const lines = [];
    const recent = (evidence.recent_notes || []).slice(0, 3);
    if (recent.length) {
      lines.push("<p>Notas tocadas hoy:</p><ul>" + recent.map((n) => {
        const url = obsidianUrl(n.path, n.vault);
        const titleHTML = `<strong>${escapeHTML(n.title || n.path)}</strong>${vaultTag(n)}`;
        return `<li>${url ? `<a class="wikilink" href="${escapeHTML(url)}">${titleHTML}</a>` : titleHTML}</li>`;
      }).join("") + "</ul>");
    }
    const ytWatched = (signals.youtube_watched || []).slice(0, 3);
    if (ytWatched.length) {
      lines.push("<p>Videos que viste:</p><ul>" + ytWatched.map((v) => {
        const url = youtubeUrl(v.video_id);
        const txt = escapeHTML(v.title || v.video_id || "");
        return `<li>${url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener">${txt}</a>` : txt}</li>`;
      }).join("") + "</ul>");
    }
    const gmailRecent = (signals.gmail?.recent || []).slice(0, 3);
    if (gmailRecent.length) {
      lines.push("<p>Mails recibidos:</p><ul>" + gmailRecent.map((m) => {
        const url = gmailThreadUrl(m.thread_id);
        const inner = `<strong>${escapeHTML(fromName(m.from))}</strong>: ${escapeHTML(truncate(m.subject || "", 70))}`;
        return `<li>${url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener">${inner}</a>` : inner}</li>`;
      }).join("") + "</ul>");
    }
    if (lines.length) {
      narrativeHTML += `<div class="hero-prose">${lines.join("")}</div>`;
    }
  }
  narrativeHTML += renderPatternsSub(patterns);

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
    for (const m of (signals.gmail?.recent || []).slice(0, 3)) {
      const inner = `📧 <strong>${escapeHTML(fromName(m.from))}</strong>: ${escapeHTML(truncate(m.subject || "", 80))}`;
      items.push(`<li>${wrapLink(gmailThreadUrl(m.thread_id), inner)}</li>`);
    }
    const wa = [...(signals.whatsapp_unreplied || [])]
      .sort((a, b) => (b.hours_waiting || 0) - (a.hours_waiting || 0))
      .slice(0, 3);
    for (const w of wa) {
      const hrs = w.hours_waiting != null ? ` <em>(${Math.round(w.hours_waiting)}h)</em>` : "";
      const inner = `💬 <strong>${escapeHTML(w.name || "")}</strong>${hrs}: ${escapeHTML(truncate(w.last_snippet || "", 70))}`;
      items.push(`<li>${wrapLink(whatsappUrl(w.jid), inner)}</li>`);
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
      const score = q.score != null
        ? ` <em>(${(typeof q.score === "number" ? q.score : Number(q.score)).toFixed(2)})</em>`
        : "";
      const inner = `❓ ${escapeHTML(text)}${score}`;
      const url = text ? `/chat?q=${encodeURIComponent(text)}` : null;
      items.push(`<li>${wrapLink(url, inner)}</li>`);
    }
    const aging = signals.followup_aging || {};
    const stalePlus = aging.stale_30plus || aging.stale_count || 0;
    if (!items.length && stalePlus > 0) {
      items.push(`<li>⚠️ <strong>${stalePlus}</strong> loops STALE (≥30d sin avance)</li>`);
    }
    const contrad = signals.contradictions || [];
    const newContrad = (evidence.new_contradictions || []).slice(0, 2);
    const allContrad = newContrad.length ? newContrad : (Array.isArray(contrad) ? contrad.slice(0, 2) : []);
    for (const c of allContrad) {
      const text = c.text || c.summary || c.title || c.note_a || "";
      if (!text) continue;
      const inner = `⚠ ${escapeHTML(truncate(text, 100))}`;
      const url = obsidianUrl(c.subject_path || c.path, c.vault);
      items.push(`<li>${wrapLink(url, inner, true)}</li>`);
    }
    if (!items.length) {
      const activo = (signals.loops_activo || []).slice(0, 3);
      for (const l of activo) {
        const txt = l.loop_text || l.text || l.title || "";
        if (!txt) continue;
        const inner = `🔄 ${escapeHTML(truncate(txt, 100))}`;
        const url = obsidianUrl(l.source_note, l.vault);
        items.push(`<li>${wrapLink(url, inner, true)}</li>`);
      }
    }
    questionsHTML = items.length ? `<ul>${items.join("")}</ul>` : "";
  }

  const dueMatchesToday = (r) => {
    const due = (r.due || r.due_at || "").toLowerCase();
    return /\btoday\b|\bhoy\b/.test(due);
  };
  const dueMatchesTomorrow = (r) => {
    const due = (r.due || r.due_at || "").toLowerCase();
    return /\btomorrow\b|ma[ñn]ana/.test(due);
  };

  const renderAgendaItems = (events, reminders, fallbackPrefix) => {
    const items = [];
    for (const e of events.slice(0, 6)) {
      const tr = (e.time_range || "").trim();
      const prefix = tr || fallbackPrefix;
      const title = e.title || e.summary || "(sin título)";
      items.push(`<li><strong>${escapeHTML(prefix)}</strong> — ${escapeHTML(title)}</li>`);
    }
    for (const r of reminders.slice(0, 3)) {
      items.push(`<li>📌 ${escapeHTML(truncate(r.title || r.text || "", 80))}</li>`);
    }
    return items.length ? `<ul>${items.join("")}</ul>` : "";
  };

  let todayHTML = "";
  if (split.today) {
    todayHTML = mdToHTML(split.today);
  } else {
    const reminders = (signals.reminders || []).filter(dueMatchesToday);
    todayHTML = renderAgendaItems(todayEvents, reminders, "todo el día");
  }

  const tomorrowBriefIsStale = split.tomorrow
    && tomorrowEvents.length === 0
    && todayEvents.length > 0;
  let tomorrowHTML = "";
  if (split.tomorrow && !tomorrowBriefIsStale) {
    tomorrowHTML = mdToHTML(split.tomorrow);
  } else {
    const reminders = (signals.reminders || []).filter(dueMatchesTomorrow);
    tomorrowHTML = renderAgendaItems(tomorrowEvents, reminders, "todo el día");
  }

  const inboxCount = inboxItems.length
    + (signals.gmail?.recent?.length || 0)
    + (signals.whatsapp_unreplied?.length || 0);
  const questionsCount = lowConfItems.length
    + ((signals.followup_aging?.stale_30plus || 0) > 0 ? 1 : 0)
    + (evidence.new_contradictions?.length || 0);
  const todayCount = todayEvents.length
    + (signals.reminders || []).filter(dueMatchesToday).length;
  const tomorrowCount = tomorrowEvents.length
    + (signals.reminders || []).filter(dueMatchesTomorrow).length;

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
  const agendaHTML = `
    <div class="hero-section s-agenda">
      <h3><span>🌅 Agenda</span>${agendaCount ? `<span class="count">${agendaCount}</span>` : ""}</h3>
      ${agendaBody}
    </div>
  `;

  bodyEl.innerHTML = [
    sectionHTML("s-narrative", "🪞", "Lo que pasó hoy", null, narrativeHTML, "Aún sin brief — pulsá ↻ arriba para generar"),
    sectionHTML("s-inbox", "📥", "Sin procesar", inboxCount || null, inboxHTML, "todo procesado ✓"),
    sectionHTML("s-questions", "🔍", "Preguntas abiertas", questionsCount || null, questionsHTML, "sin preguntas pendientes"),
    agendaHTML,
  ].join("");

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
const LS_HERO_SUB_COLLAPSED = "home.v2.hero-subs.collapsed.v1";

// Clase identificadora estable por hero-section (se persiste en LS).
const HERO_SECTION_KEYS = ["s-narrative", "s-inbox", "s-questions", "s-agenda"];

function _heroSectionKey(el) {
  for (const k of HERO_SECTION_KEYS) {
    if (el.classList.contains(k)) return k;
  }
  return null;
}

function _readHeroOrder() {
  try {
    const raw = localStorage.getItem(LS_HERO_ORDER);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : null;
  } catch { return null; }
}

function _saveHeroOrder() {
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  const order = Array.from(body.querySelectorAll(":scope > .hero-section"))
    .map(_heroSectionKey)
    .filter(Boolean);
  try {
    localStorage.setItem(LS_HERO_ORDER, JSON.stringify(order));
  } catch (e) {
    console.warn("[home.v2] no pude persistir hero order:", e);
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
  try {
    const raw = localStorage.getItem(LS_HERO_SUB_COLLAPSED);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === "object") ? parsed : {};
  } catch { return {}; }
}

function _writeHeroSubCollapsed(map) {
  try {
    const trimmed = {};
    for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
    if (Object.keys(trimmed).length === 0) {
      localStorage.removeItem(LS_HERO_SUB_COLLAPSED);
    } else {
      localStorage.setItem(LS_HERO_SUB_COLLAPSED, JSON.stringify(trimmed));
    }
  } catch (e) {
    console.warn("[home.v2] no pude persistir hero sub collapse:", e);
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

function _onHeroDragStart(ev) {
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
  document.querySelectorAll(".hero-section.drop-before, .hero-section.drop-after")
    .forEach((el) => el.classList.remove("drop-before", "drop-after"));
}

function _onHeroDragOver(ev) {
  if (!_draggingHero) return;
  const sec = ev.currentTarget;
  if (sec === _draggingHero) return;
  ev.preventDefault();
  try { ev.dataTransfer.dropEffect = "move"; } catch {}
  const rect = sec.getBoundingClientRect();
  const before = (ev.clientX - rect.left) < rect.width / 2;
  sec.classList.toggle("drop-before", before);
  sec.classList.toggle("drop-after", !before);
}

function _onHeroDragLeave(ev) {
  const sec = ev.currentTarget;
  if (sec.contains(ev.relatedTarget)) return;
  sec.classList.remove("drop-before", "drop-after");
}

function _onHeroDrop(ev) {
  ev.preventDefault();
  if (!_draggingHero) return;
  const target = ev.currentTarget;
  if (target === _draggingHero) return;
  const before = target.classList.contains("drop-before");
  target.classList.remove("drop-before", "drop-after");
  if (before) {
    target.parentNode.insertBefore(_draggingHero, target);
  } else {
    target.parentNode.insertBefore(_draggingHero, target.nextSibling);
  }
  _saveHeroOrder();
}

export function wireHeroLayout() {
  const body = document.getElementById("today-hero-body");
  if (!body) return;
  body.querySelectorAll(":scope > .hero-section").forEach((sec) => {
    // Drag grip + collapse btn dentro del h3 — solo si no existen ya.
    const h3 = sec.querySelector(":scope > h3");
    if (h3 && !h3.querySelector(".hero-drag-grip")) {
      const grip = document.createElement("span");
      grip.className = "hero-drag-grip";
      grip.setAttribute("aria-hidden", "true");
      grip.title = "arrastrá para reordenar";
      grip.textContent = "⋮⋮";
      h3.insertBefore(grip, h3.firstChild);
    }
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
    // Drag handlers — idempotente vía data flag.
    if (sec.dataset.heroDragInit !== "1") {
      sec.dataset.heroDragInit = "1";
      sec.setAttribute("draggable", "true");
      sec.addEventListener("dragstart", _onHeroDragStart);
      sec.addEventListener("dragend", _onHeroDragEnd);
      sec.addEventListener("dragover", _onHeroDragOver);
      sec.addEventListener("dragleave", _onHeroDragLeave);
      sec.addEventListener("drop", _onHeroDrop);
    }
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
  try {
    if (localStorage.getItem(LS_HERO_COLLAPSED) === "1") {
      hero.setAttribute("data-collapsed", "true");
      heroToggle.setAttribute("aria-expanded", "false");
    }
  } catch {}
  heroToggle.addEventListener("click", () => {
    const collapsed = hero.getAttribute("data-collapsed") === "true";
    const next = !collapsed;
    hero.setAttribute("data-collapsed", next ? "true" : "false");
    heroToggle.setAttribute("aria-expanded", next ? "false" : "true");
    try {
      if (next) localStorage.setItem(LS_HERO_COLLAPSED, "1");
      else localStorage.removeItem(LS_HERO_COLLAPSED);
    } catch {}
    try { window._updateResetButtonVisibility?.(); } catch {}
  });
}

export { LS_HERO_COLLAPSED, LS_HERO_ORDER, LS_HERO_SUB_COLLAPSED };
