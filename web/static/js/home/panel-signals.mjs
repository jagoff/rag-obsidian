// panel-signals.mjs — paneles de señales: inbox, questions, tomorrow,
// wa-unreplied, loops-urgent, contradictions, patterns (cross-source entities),
// authority (top contactos).

import {
  escapeHTML, fmtTimeAgo,
  isBotOrSelf, parseSenderName,
  obsidianUrl, gmailThreadUrl, whatsappUrl,
  renderPanelList,
} from "./core.mjs";

export function renderInbox(payload) {
  const evidence = payload.today?.evidence || {};
  const signals = payload.signals || {};
  const inboxToday = evidence.inbox_today || [];
  const gmailRecent = signals.gmail?.recent || [];
  const waUnreplied = [...(signals.whatsapp_unreplied || [])]
    .sort((a, b) => (b.hours_waiting || 0) - (a.hours_waiting || 0));
  const mailUnread = signals.mail_unread || [];
  const fromName = (s) => (s || "").split("<")[0].trim() || s || "";
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

  const rows = [];
  for (const it of inboxToday.slice(0, 3)) {
    rows.push({
      title: it.title || it.path,
      meta: [
        it.vault ? `[${it.vault}]` : null,
        ...(it.tags || []).map((t) => `#${t}`),
        fmtTimeAgo(it.modified),
      ].filter(Boolean),
      href: obsidianUrl(it.path, it.vault),
    });
  }
  for (const m of gmailRecent.slice(0, 3)) {
    rows.push({
      title: `📧 ${fromName(m.from)}: ${truncate(m.subject || "", 70)}`,
      meta: [m.internal_date_ms ? fmtTimeAgo(new Date(m.internal_date_ms).toISOString()) : null].filter(Boolean),
      href: gmailThreadUrl(m.thread_id),
    });
  }
  for (const w of waUnreplied.slice(0, 3)) {
    rows.push({
      title: `💬 ${w.name || ""}`,
      meta: [
        w.hours_waiting != null ? `${Math.round(w.hours_waiting)}h esperando` : null,
        truncate(w.last_snippet || "", 60),
      ].filter(Boolean),
      href: whatsappUrl(w.jid),
    });
  }
  if (!rows.length && mailUnread.length) {
    for (const m of mailUnread.slice(0, 3)) {
      rows.push({
        title: `📬 ${fromName(m.from || m.sender)}: ${truncate(m.subject || "", 70)}`,
        meta: [],
        href: m.message_id ? `message:${encodeURIComponent(m.message_id)}` : null,
      });
    }
  }

  const total = inboxToday.length + gmailRecent.length + waUnreplied.length + mailUnread.length;
  renderPanelList("p-inbox", rows, {
    emptyText: "todo el inbox procesado ✓",
    capChip: rows.length > 5 ? "warning" : "info",
    footText: rows.length ? `${rows.length} de ${total}` : "",
  });
}

export function renderQuestions(payload) {
  const signals = payload.signals || {};
  const evidence = payload.today?.evidence || {};
  const lowConf = signals.low_conf || [];
  const newContrad = evidence.new_contradictions || [];
  const contradictions = signals.contradictions || [];
  const loopsActivo = signals.loops_activo || [];
  const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

  const rows = [];
  for (const it of lowConf.slice(0, 4)) {
    const text = it.q || it.question || it.text || "";
    rows.push({
      title: `❓ ${text}`,
      meta: [
        it.score != null ? `score ${(typeof it.score === "number" ? it.score : Number(it.score)).toFixed(2)}` : null,
        it.ts ? fmtTimeAgo(it.ts) : null,
      ].filter(Boolean),
      href: text ? `/chat?q=${encodeURIComponent(text)}` : null,
    });
  }
  const contradList = newContrad.length ? newContrad : (Array.isArray(contradictions) ? contradictions : []);
  for (const c of contradList.slice(0, 3)) {
    const text = c.text || c.summary || c.title || c.note_a || c.subject_path || "";
    if (!text) continue;
    rows.push({
      title: `⚠ ${truncate(text, 90)}`,
      meta: [c.ts ? fmtTimeAgo(c.ts) : null].filter(Boolean),
      href: obsidianUrl(c.subject_path || c.path, c.vault),
    });
  }
  if (!rows.length) {
    for (const l of loopsActivo.slice(0, 4)) {
      const text = l.loop_text || l.text || l.title || "";
      if (!text) continue;
      rows.push({
        title: `🔄 ${truncate(text, 90)}`,
        meta: [
          l.age_days != null ? `${l.age_days}d` : null,
          l.source_note ? l.source_note.split("/").pop().replace(/\.md$/, "") : null,
        ].filter(Boolean),
        href: obsidianUrl(l.source_note, l.vault),
      });
    }
  }
  renderPanelList("p-questions", rows, {
    emptyText: "no hay preguntas sin respuesta",
    capChip: rows.length > 4 ? "warning" : "info",
  });
}

export function renderTomorrow(payload) {
  const events = Array.isArray(payload.tomorrow_calendar)
    ? payload.tomorrow_calendar
    : [];
  const signals = payload.signals || {};
  const reminders = (signals.reminders || []).filter((r) => {
    const due = (r.due || r.due_at || "").toLowerCase();
    return due.includes("tomorrow") || due.includes("mañana");
  });

  const rows = [];
  for (const e of events.slice(0, 8)) {
    const tr = (e.time_range || "").trim();
    rows.push({
      title: e.title || e.summary || "(sin título)",
      meta: [
        tr || "todo el día",
        e.location || null,
      ].filter(Boolean),
    });
  }
  for (const r of reminders.slice(0, 3)) {
    rows.push({
      title: `📌 ${r.title || r.text || ""}`,
      meta: [r.list || null].filter(Boolean),
    });
  }
  renderPanelList("p-tomorrow", rows, {
    emptyText: "agenda libre mañana",
  });
}

export function renderWAUnreplied(payload) {
  const items = payload.signals?.whatsapp_unreplied || [];
  const cleanSnippet = (s) => {
    if (!s) return null;
    const trimmed = s.trim();
    if (/^https?:\/\//.test(trimmed)) {
      try {
        return `↗ ${new URL(trimmed).hostname.replace(/^www\./, "")}`;
      } catch { return "↗ link"; }
    }
    return trimmed.replace(/\s+/g, " ").slice(0, 70);
  };
  const rows = items.slice(0, 6).map((it) => ({
    title: it.name || it.jid,
    meta: [
      cleanSnippet(it.last_snippet),
      it.hours_since != null
        ? (it.hours_since < 1 ? "<1h"
           : it.hours_since < 24 ? `${Math.round(it.hours_since)}h`
           : `${Math.round(it.hours_since / 24)}d`)
        : null,
    ].filter(Boolean),
    href: whatsappUrl(it.jid),
  }));
  renderPanelList("p-wa-unreplied", rows, {
    emptyText: "todo respondido",
    capChip: rows.length > 8 ? "critical" : "warning",
  });
}

export function renderLoopsUrgent(payload) {
  const stale = payload.signals?.loops_stale || [];
  const rows = stale.slice(0, 8).map((it) => ({
    title: it.loop_text || "",
    meta: [
      it.source_note ? it.source_note.split("/").pop().replace(/\.md$/, "") : null,
      it.extracted_at ? fmtTimeAgo(it.extracted_at) : null,
    ].filter(Boolean),
    href: obsidianUrl(it.source_note, it.vault),
  }));
  renderPanelList("p-loops-urgent", rows, {
    emptyText: "ningún loop STALE",
    capChip: rows.length > 0 ? "warning" : "info",
  });
}

export function renderContradictions(payload) {
  const items = payload.signals?.contradictions || [];
  const slugToTitle = (s) => {
    if (!s) return "";
    const stem = s.split("/").pop().replace(/\.md$/, "");
    return stem.replace(/_/g, " ")
      .replace(/^obsidian rag /i, "")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .slice(0, 80);
  };
  const rows = items.slice(0, 6).map((it) => {
    const subj = slugToTitle(it.subject_path);
    const tgt = (it.targets && it.targets[0]) || {};
    const tgtTitle = slugToTitle(tgt.note || tgt.path || "");
    const why = (tgt.why || "").trim();
    return {
      title: `📝 ${subj}`,
      meta: [
        tgtTitle ? `⟷ ${tgtTitle}` : null,
        why ? `motivo: ${why.slice(0, 90)}` : null,
        it.ts ? fmtTimeAgo(it.ts) : null,
      ].filter(Boolean),
      href: obsidianUrl(it.subject_path, it.vault),
    };
  });
  renderPanelList("p-contradictions", rows, {
    emptyText: "sin contradicciones detectadas entre tus notas",
    capChip: rows.length > 0 ? "warning" : "info",
    footText: rows.length ? "notas que se contradicen entre sí" : "",
  });
}

export function renderPatterns(payload) {
  const panel = document.getElementById("p-patterns");
  if (!panel) return;
  const correlations = payload.today?.correlations
    || payload.signals?.correlations
    || {};
  const people = correlations.people || [];
  const topics = correlations.topics || [];
  const overlaps = correlations.time_overlaps || [];
  const gaps = correlations.gaps || [];
  const total = people.length + topics.length + overlaps.length + gaps.length;

  // Fallback: si no hay cross-source explícitos, derivar de gmail + WA
  if (total === 0) {
    const signals = payload.signals || {};
    const counts = new Map();
    for (const m of signals.gmail?.recent || []) {
      if (isBotOrSelf(m.from)) continue;
      const n = parseSenderName(m.from);
      if (!n) continue;
      const e = counts.get(n) || { sources: new Set(), mentions: 0 };
      e.sources.add("📧"); e.mentions++;
      counts.set(n, e);
    }
    for (const w of signals.whatsapp_unreplied || []) {
      const n = (w.name || "").trim();
      if (!n || isBotOrSelf(n)) continue;
      const e = counts.get(n) || { sources: new Set(), mentions: 0 };
      e.sources.add("💬"); e.mentions++;
      counts.set(n, e);
    }
    const tops = Array.from(counts.entries())
      .sort((a, b) => (b[1].sources.size - a[1].sources.size) || (b[1].mentions - a[1].mentions))
      .slice(0, 5);
    if (tops.length === 0) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    count.textContent = String(tops.length);
    body.innerHTML = tops.map(([name, e]) => `
      <div class="pattern-row">
        <div>
          <span class="pattern-name">👤 ${escapeHTML(name)}</span>
          <span class="pattern-sources"> · ${[...e.sources].join(" ")} (${e.mentions} ${e.mentions === 1 ? "mensaje" : "mensajes"})</span>
        </div>
      </div>`).join("");
    return;
  }
  panel.hidden = false;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
  count.textContent = String(total);
  const rows = [];
  for (const p of people.slice(0, 5)) {
    const sources = (p.appearances || [])
      .map((a) => a.source)
      .filter((v, i, arr) => arr.indexOf(v) === i)
      .join(" + ");
    rows.push(`
      <div class="pattern-row">
        <div>
          <span class="pattern-name">👤 ${escapeHTML(p.name)}</span>
          <span class="pattern-sources"> · ${escapeHTML(sources)} (${p.sources_count})</span>
        </div>
      </div>`);
  }
  for (const t of topics.slice(0, 5)) {
    rows.push(`
      <div class="pattern-row">
        <div>
          <span class="pattern-name">💡 ${escapeHTML(t.topic)}</span>
          <span class="pattern-sources"> · ${escapeHTML(t.sources.join(" + "))}</span>
        </div>
      </div>`);
  }
  for (const o of overlaps.slice(0, 3)) {
    const labels = (o.items || [])
      .map((it) => `${it.source}: "${(it.label || "").slice(0, 40)}"`)
      .join(" ↔ ");
    rows.push(`
      <div class="pattern-row">
        <div>
          <span class="pattern-name">⏱ ${escapeHTML(o.time)}</span>
          <span class="pattern-sources"> · ${escapeHTML(labels)}</span>
        </div>
      </div>`);
  }
  for (const g of gaps.slice(0, 5)) {
    const hours = Math.round(g.hours_waiting || 0);
    const person = g.person || "?";
    const snippet = (g.snippet || "").slice(0, 60);
    rows.push(`
      <div class="pattern-row pattern-gap">
        <div>
          <span class="pattern-name">⚠ ${escapeHTML(person)}</span>
          <span class="pattern-sources">
            · ${hours}h sin responder · "${escapeHTML(snippet)}" · sin slot mañana
          </span>
        </div>
      </div>`);
  }
  body.innerHTML = rows.join("");
}

export function renderAuthority(payload) {
  const signals = payload.signals || {};
  const counts = new Map();
  for (const m of signals.gmail?.recent || []) {
    if (isBotOrSelf(m.from)) continue;
    const n = parseSenderName(m.from);
    if (!n) continue;
    const e = counts.get(n) || { sources: new Set(), count: 0, lastTs: 0 };
    e.sources.add("📧"); e.count++;
    e.lastTs = Math.max(e.lastTs, m.internal_date_ms || 0);
    counts.set(n, e);
  }
  for (const w of signals.whatsapp_unreplied || []) {
    const n = (w.name || "").trim();
    if (!n || isBotOrSelf(n)) continue;
    const e = counts.get(n) || { sources: new Set(), count: 0, lastTs: 0 };
    e.sources.add("💬"); e.count++;
    counts.set(n, e);
  }
  for (const m of signals.mail_unread || []) {
    const sender = m.from || m.sender || "";
    if (isBotOrSelf(sender)) continue;
    const n = parseSenderName(sender);
    if (!n) continue;
    const e = counts.get(n) || { sources: new Set(), count: 0, lastTs: 0 };
    e.sources.add("📬"); e.count++;
    counts.set(n, e);
  }
  const tops = Array.from(counts.entries())
    .sort((a, b) =>
      (b[1].sources.size - a[1].sources.size) ||
      (b[1].count - a[1].count) ||
      (b[1].lastTs - a[1].lastTs)
    )
    .slice(0, 5);
  const rows = tops.map(([name, e], i) => ({
    title: `#${i + 1}  ${name}`,
    meta: [
      [...e.sources].join(" "),
      `${e.count} ${e.count === 1 ? "mensaje" : "mensajes"}`,
    ],
  }));
  renderPanelList("p-authority", rows, {
    emptyText: "sin actividad de contactos esta semana",
  });
}
