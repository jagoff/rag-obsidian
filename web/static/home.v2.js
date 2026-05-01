// home.v2.js — orchestrator for the mission-control dashboard.
//
// Responsabilidades:
//   1. Fetch /api/home (con SWR — el endpoint ya hace stale-while-revalidate)
//   2. Renderizar la command bar (4 KPIs prominentes con trend)
//   3. Alimentar cada panel desde signals/today/urgent
//   4. Generar ASCII mini-charts + sparkline SVG inline
//   5. Auto-refresh cada 5 min; SSE opcional via /api/home/stream para
//      progress en vivo (no implementado en v1 del refactor)
//   6. Click handlers para drill-down (collapse / expand de paneles)
//
// Convención: cada panel id `p-<name>` tiene `[data-body]`, `[data-count]`,
// `[data-foot]` que se rellenan via los renderers. Un renderer devuelve
// el conteo (number) que se muestra en el chip del header.

(function () {
  "use strict";

  // ──────────────────────────────────────────────────────────────
  // Utilities
  // ──────────────────────────────────────────────────────────────

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // Detect notif/bot/self emails para no contar como "contactos reales".
  // Trampas conocidas: github noreply usa display name "Fer F" pero
  // viene de notifications@github.com → pattern matching en el email.
  // El propio user (fernandoferrari@gmail.com) también aparece en
  // recent cuando hace forward o sent — filtrarlo.
  const isBotOrSelf = (sender) => {
    if (!sender) return true;
    const s = String(sender).toLowerCase();
    if (/notifications?@|noreply@|no-?reply@|donot-?reply@|notificaci[oó]n@|automat|bounce@|mailer-?daemon/.test(s)) return true;
    if (/no\s*responder|no-?responder|do\s*not\s*reply|noreply/.test(s)) return true;
    if (/fernandoferrari|fer\.f@/.test(s)) return true;     // self
    return false;
  };

  // Extraer nombre legible del sender. Gmail puede venir como:
  //   "Monica Ferrari <monica.ferrari@gmail.com>"
  //   "\"Fer F.\" <fernandoferrari@gmail.com>"
  //   "monica.ferrari@gmail.com"
  // Devuelve "Monica Ferrari" o "" si no se puede parsear bien.
  const parseSenderName = (sender) => {
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

  const escapeHTML = (s) =>
    String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const fmtNumber = (n) => {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (typeof n !== "number") return String(n);
    if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "k";
    return String(n);
  };

  const fmtCurrencyARS = (n) => {
    if (n === null || n === undefined) return "—";
    return "$" + Math.round(n).toLocaleString("es-AR");
  };

  const fmtTimeAgo = (iso) => {
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

  const ageBucket = (hours) => {
    if (hours < 24 * 7) return { kind: "ok",   text: hours < 24 ? `${Math.round(hours)}h` : `${Math.round(hours / 24)}d` };
    if (hours < 24 * 30) return { kind: "warn", text: `${Math.round(hours / 24)}d` };
    return { kind: "stale", text: `STALE ${Math.round(hours / 24)}d` };
  };

  // SVG sparkline. `values` = array of numbers. `tone` ∈ ok|warn|crit|info
  function sparkline(values, tone = "info") {
    if (!Array.isArray(values)) return "";
    // Sanitize: drop nulls/NaN/non-numbers; necesitamos >=2 puntos válidos.
    const clean = values.map((v) => Number(v)).filter((v) => Number.isFinite(v));
    if (clean.length < 2) return "";
    values = clean;
    const w = 100, h = 32, pad = 2;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const stepX = (w - 2 * pad) / (values.length - 1);
    const points = values.map((v, i) => {
      const x = pad + i * stepX;
      const y = h - pad - ((v - min) / range) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const areaPoints = `${pad},${h - pad} ${points} ${(w - pad).toFixed(1)},${h - pad}`;
    const cls = tone === "ok" ? "is-ok"
              : tone === "warn" ? "is-warning"
              : tone === "crit" ? "is-critical"
              : "";
    return `<svg class="sparkline ${cls}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
      <polygon class="area" points="${areaPoints}"></polygon>
      <polyline class="line" points="${points}"></polyline>
    </svg>`;
  }

  // Stacked horizontal bar (3 segments). `parts` = {fresh, aging, stale}.
  function stackedBar(parts) {
    const total = (parts.fresh || 0) + (parts.aging || 0) + (parts.stale || 0);
    if (!total) return "";
    const f = (parts.fresh || 0) / total * 100;
    const a = (parts.aging || 0) / total * 100;
    const s = (parts.stale || 0) / total * 100;
    return `<div class="stacked-bar" role="img" aria-label="distribución por edad">
      ${f > 0 ? `<div class="seg fresh" style="flex-basis: ${f}%"></div>` : ""}
      ${a > 0 ? `<div class="seg aging" style="flex-basis: ${a}%"></div>` : ""}
      ${s > 0 ? `<div class="seg stale" style="flex-basis: ${s}%"></div>` : ""}
    </div>
    <div class="stacked-bar-legend">
      <span><span class="swatch fresh"></span>0-7d · ${parts.fresh || 0}</span>
      <span><span class="swatch aging"></span>8-30d · ${parts.aging || 0}</span>
      <span><span class="swatch stale"></span>STALE · ${parts.stale || 0}</span>
    </div>`;
  }

  // ──────────────────────────────────────────────────────────────
  // Panel helpers
  // ──────────────────────────────────────────────────────────────

  // ──────────────────────────────────────────────────────────────
  // URL builders — construye URLs accionables por tipo de item
  // (nota → obsidian://, mail → gmail thread, WA → wa.me, etc).
  // Cada renderer pasa `r.href` con el resultado para que la row
  // entera sea clickeable.
  // ──────────────────────────────────────────────────────────────

  // Cache del map alias→dir-name del vault. El payload lo trae como
  // `vault_dir_names`. Usamos `_currentPayload` que se setea en cada
  // load() para que urlFor() pueda accederlo desde cualquier renderer.
  let _currentPayload = null;

  function obsidianUrl(path, vaultAlias) {
    if (!path) return null;
    const dirNames = _currentPayload?.vault_dir_names || {};
    // Si el item tiene `vault: "home"` el dir name es "Notes" (basename
    // del path absoluto del vault). El alias es para la UI; Obsidian
    // necesita el dir name. Default: si no tenemos mapeo, usar el alias
    // directamente — Obsidian fallback al primer vault que matche.
    const dirName = dirNames[vaultAlias || ""]
      || dirNames.home
      || vaultAlias
      || "Notes";
    return `obsidian://open?vault=${encodeURIComponent(dirName)}`
      + `&file=${encodeURIComponent(path)}`;
  }

  function gmailThreadUrl(threadId) {
    if (!threadId) return null;
    return `https://mail.google.com/mail/u/0/#inbox/${encodeURIComponent(threadId)}`;
  }

  function whatsappUrl(jid) {
    // WhatsApp jid formats:
    //   "5491155894168@s.whatsapp.net"   → individual con phone (linkable)
    //   "5491155894168@c.us"             → individual legacy (linkable)
    //   "5491155894168-1358009189@g.us"  → group (sin URL pública)
    //   "143065847189596@lid"            → LID anónimo (no es phone)
    //   "1234567890@broadcast"           → broadcast list (no linkable)
    // Solo aceptamos los individuales con phone real.
    if (!jid) return null;
    const s = String(jid);
    if (!/@(s\.whatsapp\.net|c\.us)$/.test(s)) return null;
    const phone = s.split("@")[0].replace(/\D/g, "");
    if (!phone || phone.length < 7) return null;   // sanity check
    return `https://wa.me/${phone}`;
  }

  function youtubeUrl(videoId) {
    if (!videoId) return null;
    return `https://youtube.com/watch?v=${encodeURIComponent(videoId)}`;
  }

  // Set a panel's body to a list of rows. Each row: {title, meta, aside, href}
  function renderPanelList(panelId, rows, opts = {}) {
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
      // Cuando hay href, la ROW ENTERA es clickeable (no solo el title).
      // Usamos <a class="row row--linked"> para que cualquier click en
      // cualquier parte de la row navegue. target="_blank" para mails,
      // urls, youtube; sin target para obsidian:// (la app maneja).
      // Cursor pointer + hover via CSS class .row--linked.
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

  // ──────────────────────────────────────────────────────────────
  // Renderers — uno por panel, cada uno toma el payload completo
  // ──────────────────────────────────────────────────────────────

  function setKPI(id, { value, label, meta, tone, trend }) {
    const el = document.getElementById(id);
    if (!el) return;
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

  function renderCmdBar(payload) {
    const inboxToday = payload.today?.evidence?.inbox_today || [];
    const reminders = payload.signals?.reminders || [];
    const wa = payload.signals?.whatsapp_unreplied || [];
    const loops = payload.signals?.loops_stale || [];

    const inboxCount = inboxToday.length;
    setKPI("kpi-inbox", {
      value: inboxCount,
      tone: inboxCount === 0 ? "ok" : inboxCount > 5 ? "critical" : "warning",
      meta: inboxCount === 0 ? "todo procesado" :
            inboxCount === 1 ? "1 pendiente" :
            `${inboxCount} pendientes`,
    });

    const remindersDue = reminders.filter((r) => {
      // due hoy/overdue
      if (!r.due_date) return true;
      const due = new Date(r.due_date);
      const now = new Date();
      const eod = new Date(now); eod.setHours(23, 59, 59, 999);
      return due <= eod;
    }).length;
    setKPI("kpi-reminders", {
      value: remindersDue,
      tone: remindersDue === 0 ? "ok" : remindersDue > 3 ? "critical" : "warning",
      meta: remindersDue === 0 ? "tranquilo" :
            remindersDue === 1 ? "1 para hoy" :
            `${remindersDue} para hoy`,
    });

    setKPI("kpi-wa", {
      value: wa.length,
      tone: wa.length === 0 ? "ok" : wa.length > 8 ? "critical" : "warning",
      meta: wa.length === 0 ? "todo respondido" :
            wa.length === 1 ? "1 chat espera respuesta" :
            `${wa.length} chats esperan respuesta`,
    });

    setKPI("kpi-loops", {
      value: loops.length,
      tone: loops.length === 0 ? "ok" : loops.length > 5 ? "critical" : "warning",
      meta: loops.length === 0 ? "ningún loop envejeciendo" :
            loops.length === 1 ? "1 loop STALE" :
            `${loops.length} loops STALE`,
    });
  }

  function renderInbox(payload) {
    // Bug fix: el panel sólo miraba `evidence.inbox_today` (capturas
    // del vault). Si el inbox del vault estaba vacío pero había
    // mails VIP / WhatsApp pendientes / mail Apple sin leer, el panel
    // mostraba "todo procesado ✓" mintiéndole al user. Cascada igual
    // a la del hero: vault inbox → gmail.recent → whatsapp_unreplied
    // → mail_unread.
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
          // Apple Mail usa scheme `message://` con message-id, pero no
          // siempre lo tenemos. Fallback a Mail.app via mailto: del sender.
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

  function renderQuestions(payload) {
    // Bug fix: el panel sólo miraba `signals.low_conf`. Si el día no
    // tuvo queries low-confidence (común), ignoraba contradicciones
    // detectadas y loops activos / stale que sí pueden tener items
    // accionables. Cascada como en el hero: low_conf → contradictions
    // → loops_activo → loops_stale.
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
        // low-conf queries: linkear a /chat con la query pre-filled para
        // que el user pueda re-preguntar.
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

  function renderTomorrow(payload) {
    // Bug fix: `tomorrow_calendar` viene como ARRAY directo en el
    // payload, no como objeto con `.events`. El panel usaba `cal.events`
    // y siempre quedaba vacío aunque hubiera 2 events de mañana.
    // Misma raíz que el bug del hero "Para mañana".
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

  function renderWAUnreplied(payload) {
    const items = payload.signals?.whatsapp_unreplied || [];
    // Cleanup del snippet: si es URL pelada, mostrar dominio + "(link)";
    // sino, primer ~70 chars sin saltos.
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

  function renderLoopsUrgent(payload) {
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

  function renderContradictions(payload) {
    // Bug fix UX: el render previo mostraba los slugs raw de los archivos
    // (ej. "obsidian_rag_today_brief_4_instrucciones_contradictorias_..."
    // y "vs obsidian_rag_home_brief_vacio_cuando_today_total_0_...") sin
    // explicar la idea de "contradicción" ni mostrar la razón. El user no
    // entendía el panel. Ahora:
    //   - "Notas que se contradicen" como hint del panel (en el HTML).
    //   - Título: nombre legible derivado del slug (under_score → words).
    //   - Meta línea 1: "⟷ Otra nota" (legible también).
    //   - Meta línea 2: "razón: <why del LLM>" (lo más importante — POR QUÉ).
    const items = payload.signals?.contradictions || [];
    const slugToTitle = (s) => {
      if (!s) return "";
      const stem = s.split("/").pop().replace(/\.md$/, "");
      return stem.replace(/_/g, " ")
        .replace(/^obsidian rag /i, "")     // prefijo común en mem-vault
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

  // ──────────────────────────────────────────────────────────────
  // Today hero — el bloque grande arriba con 4 sub-secciones
  // (Lo que pasó hoy / Sin procesar / Preguntas abiertas / Para mañana).
  // El narrative del LLM es markdown con 4 H2; lo splitemos en 4 trozos
  // y renderemos cada uno en su columna.
  // ──────────────────────────────────────────────────────────────

  // Convertir [[wikilinks]] del LLM a markdown link estándar
  // [Label](obsidian://...) — antes los strippeábamos a texto plano,
  // perdiendo la posibilidad de hacer click. Ahora abren la nota en
  // Obsidian.app directamente. El user pidió: "las notas de obsidian
  // pueden ser clickeables también".
  //
  // Casos:
  //   [[Foo]]            → [Foo](obsidian://open?vault=Notes&file=Foo)
  //   [[Foo|Display]]    → [Display](obsidian://open?vault=Notes&file=Foo)
  //
  // Vault name: el LLM no incluye el vault en el wikilink. Usamos el
  // current vault (home), así Obsidian busca en ese vault primero. Si
  // la nota está en el vault work, Obsidian igual la encuentra porque
  // el `file=<nombre>` se resuelve por nombre, no por path absoluto.
  function wikilinksToMarkdown(s) {
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
  // Backwards-compat alias para callers que solo quieren el plain text
  // (vault tag derivation, count, etc).
  function stripWikilinks(s) {
    if (!s) return "";
    return s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2")
            .replace(/\[\[([^\]]+)\]\]/g, "$1");
  }

  // Sanitizer mínimo para el output de marked. Quita <script>, on*=,
  // javascript: hrefs. Defensa pragmática — el contenido viene de un
  // LLM local que confiamos pero que podría escupir basura por error.
  function sanitizeHTML(html) {
    return String(html)
      .replace(/<\s*script[\s\S]*?<\/\s*script\s*>/gi, "")
      .replace(/\son\w+\s*=\s*"[^"]*"/gi, "")
      .replace(/\son\w+\s*=\s*'[^']*'/gi, "")
      .replace(/javascript:/gi, "");
  }

  function mdToHTML(md) {
    if (!md) return "";
    // Convertir wikilinks A markdown ANTES de pasar a marked, para que
    // [Label](obsidian://...) se renderee como <a> nativo. Después
    // post-process para agregar target=_blank en URLs externas y class
    // "wikilink" en obsidian:// (para CSS distinto si querés).
    const withLinks = wikilinksToMarkdown(md);
    if (window.marked) {
      try {
        let html = window.marked.parse(withLinks);
        html = sanitizeHTML(html);
        // Marcar los links a obsidian:// con clase distintiva. Los http
        // van a tab nueva. Skip los links internos (#anchor).
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
    // Fallback sin marked: strippear wikilinks a texto plano.
    return `<pre style="white-space:pre-wrap;font-family:inherit;margin:0;">${escapeHTML(stripWikilinks(md))}</pre>`;
  }

  // Splittear el narrative en sus 4 sub-secciones por H2.
  // El formato esperado del LLM (system prompt en rag.py):
  //   ## 🪞 Lo que pasó hoy
  //   ...prose
  //   ## 📥 Sin procesar
  //   - item
  //   ## 🔍 Preguntas abiertas
  //   ...
  //   ## 🌅 Para mañana
  //   - item
  //
  // Devolvemos `{narrative, inbox, questions, tomorrow}` con el contenido
  // (sin el header) de cada sub-sección. Si el LLM se zarpó con el orden
  // o agregó/sacó secciones, matcheamos por keyword en el header.
  function splitNarrative(md) {
    const out = { narrative: "", inbox: "", questions: "", tomorrow: "" };
    if (!md) return out;
    // Split por líneas que empiezan con `## ` (H2). El primer chunk
    // es lo que va antes del primer H2 (puede ser vacío).
    const chunks = md.split(/^##\s+/m).map((c, i) => i === 0 ? c : "## " + c);
    for (const chunk of chunks) {
      const headerMatch = chunk.match(/^##\s+(.+?)\n([\s\S]*)$/);
      if (!headerMatch) continue;
      const header = headerMatch[1].toLowerCase();
      const body = headerMatch[2].trim();
      if (/lo que pas[oó]|hoy/.test(header) && /pas/.test(header)) {
        out.narrative = body;
      } else if (/sin procesar|inbox|por procesar/.test(header)) {
        out.inbox = body;
      } else if (/pregunta|abierta|open|low.?conf/.test(header)) {
        out.questions = body;
      } else if (/ma[ñn]ana|tomorrow|para hoy m/.test(header)) {
        out.tomorrow = body;
      }
    }
    return out;
  }

  // ──────────────────────────────────────────────────────────────
  // Highlights — fila de chips arriba del prose en "Lo que pasó hoy"
  // que sintetiza la actividad cross-source del día sin esperar al
  // LLM. Cada chip es un mini-stat clickeable cuando aplica (top WA
  // chat → abre el chat, top persona → abre WA si tiene jid, etc.).
  // El backend (`web/server.py:_home_compute`) garantiza que estos
  // valores estén SIEMPRE disponibles, no solo cuando regenerate=true.
  // Inputs: `payload.today.highlights` (ver schema en server.py).
  // Output: HTML string. Si no hay highlights, retorna "" para que el
  // contenedor no rendee espacio vacío.
  // ──────────────────────────────────────────────────────────────
  function renderHighlights(h) {
    if (!h || typeof h !== "object") return "";
    const chips = [];
    const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

    // Chip 1: Top WhatsApp chat del día — clickeable al chat.
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

    // Chip 2: total mensajes WA del día (resumen volumen).
    if ((h.wa_total_msgs || 0) > 0 && (h.wa_active_chats || 0) > 1) {
      chips.push(`
        <div class="hl-chip" title="suma de inbound en los chats activos del día">
          <span class="hl-icon">📊</span>
          <span class="hl-label">${h.wa_active_chats} chats</span>
          <span class="hl-value">${h.wa_total_msgs}</span>
        </div>`);
    }

    // Chip 3: gmail unread + awaiting reply.
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

    // Chip 4: meetings del día.
    if ((h.calendar_events || 0) > 0) {
      chips.push(`
        <div class="hl-chip">
          <span class="hl-icon">📅</span>
          <span class="hl-label">Reuniones</span>
          <span class="hl-value">${h.calendar_events}</span>
        </div>`);
    }

    // Chip 5: videos vistos.
    if ((h.youtube_videos || 0) > 0) {
      chips.push(`
        <div class="hl-chip">
          <span class="hl-icon">📺</span>
          <span class="hl-label">YouTube</span>
          <span class="hl-value">${h.youtube_videos}</span>
        </div>`);
    }

    // Chip 6: notas del vault tocadas hoy.
    if ((h.vault_notes_today || 0) > 0) {
      chips.push(`
        <div class="hl-chip">
          <span class="hl-icon">📝</span>
          <span class="hl-label">Notas</span>
          <span class="hl-value">${h.vault_notes_today}</span>
        </div>`);
    }

    // Chip 7: top persona cross-source. Construimos URL si la persona
    // aparece en WA (extraemos jid del bucket whatsapp en signals).
    if (h.top_person && h.top_person.sources_count >= 2) {
      const p = h.top_person;
      const sources = (p.appearances || []).map((a) => a.source).join(" + ");
      // Buscar jid en appearances source=whatsapp para hacerlo clickeable
      const waApp = (p.appearances || []).find((a) => a.source === "whatsapp");
      let waJid = null;
      if (waApp) {
        const waList = _currentPayload?.signals?.whatsapp || [];
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

    // Chip 8: top topic cross-source.
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

  // ──────────────────────────────────────────────────────────────
  // Patterns — sub-bloque debajo del prose con detecciones automáticas:
  //   - spikes: chats WA con activity ≥2.5× su baseline 7d
  //   - silences: chats que escribían mucho la semana pasada y hoy 0
  //   - concentrations: temas que aparecen en ≥3 fuentes
  //   - gaps: personas que te escribieron y no aparecen mañana
  // Backend en `rag/today_patterns.py`. Cada lista vacía oculta su
  // sub-card. Si nada hay nada, retornamos "" y no rendea el bloque.
  // ──────────────────────────────────────────────────────────────
  function renderPatterns(p) {
    if (!p || typeof p !== "object") return "";
    const cards = [];
    const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");

    // Spikes — chats WA hot del día.
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

    // Silencios — gente que escribía y hoy no.
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

    // Concentraciones — temas en ≥3 fuentes.
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

    // Gaps — gente que te escribió y no tenés slot mañana.
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

  function renderTodayHero(payload) {
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
    // tomorrow_calendar viene como array directo en el payload (NO como
    // objeto con .events). Bug histórico: el código previo asumía la
    // shape equivocada y nunca caía al fallback aunque hubiera events.
    const tomorrowEvents = Array.isArray(payload.tomorrow_calendar)
      ? payload.tomorrow_calendar
      : [];

    // Counts chip al lado de la fecha
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

    // ── Helpers para los fallbacks ricos ────────────────────────────
    // Cuando el LLM omite una sub-sección (cosa que hace seguido si
    // la regla dice "OMITÍ si vacío"), llenamos con data real de
    // signals. Evita las cajas vacías que el user reportaba.

    const fromName = (s) => (s || "").split("<")[0].trim() || s || "";
    const truncate = (s, n) => (s || "").length > n ? (s || "").slice(0, n) + "…" : (s || "");
    // Detectar si hay >1 vault para mostrar el tag [vault: name] en cada
    // item. Sino, sería ruido visual cuando solo hay un vault.
    const vaultsInPlay = new Set();
    for (const item of [...(evidence.recent_notes || []), ...inboxItems, ...(evidence.todos || [])]) {
      if (item && item.vault) vaultsInPlay.add(item.vault);
    }
    const showVaultTag = vaultsInPlay.size > 1;
    const vaultTag = (item) => (showVaultTag && item?.vault)
      ? ` <span class="vault-tag">[${escapeHTML(item.vault)}]</span>`
      : "";

    // Sub-section: 🪞 "Lo que pasó hoy" — estructura final:
    //   1. Fila de chips de highlights (Top WA · Mails · Meetings · YT · Persona · Tema)
    //   2. Prose narrativo del LLM (o fallback derivado de signals)
    //   3. Sub-bloque "Patrones" (spikes · silencios · concentraciones · gaps)
    // Cada bloque se oculta si no tiene data; el prose siempre intenta
    // rendear algo (LLM cached o fallback).
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
    narrativeHTML += renderPatterns(patterns);

    // Sub-section: 📥 "Sin procesar" — inbox vault + mails VIP +
    // WhatsApp esperando + mails Apple sin leer. El user los necesita
    // ver siempre que existan, no importa si el LLM los listó.
    let inboxHTML = "";
    if (split.inbox) {
      inboxHTML = mdToHTML(split.inbox);
    } else {
      const items = [];
      // Helper local para wrappear el contenido de un <li> en un link
      // si tenemos URL accionable. Usa target=_blank para http(s).
      const wrapLink = (url, innerHTML, isObsidian = false) => {
        if (!url) return innerHTML;
        const target = /^https?:/.test(url) ? ' target="_blank" rel="noopener"' : "";
        const cls = isObsidian ? ' class="wikilink"' : "";
        return `<a${cls} href="${escapeHTML(url)}"${target}>${innerHTML}</a>`;
      };
      // Inbox del vault (puede venir de cualquiera de los vaults
      // registrados — el tag [home]/[work] aclara cuál si hay >1).
      for (const it of inboxItems.slice(0, 3)) {
        const tags = (it.tags || []).slice(0, 3)
          .map((t) => `<code>#${escapeHTML(t)}</code>`).join(" ");
        const inner = `📝 <strong>${escapeHTML(it.title || it.path)}</strong>${vaultTag(it)}${tags ? " " + tags : ""}`;
        items.push(`<li>${wrapLink(obsidianUrl(it.path, it.vault), inner, true)}</li>`);
      }
      // Mails VIP / recientes (gmail.recent suele incluir VIP)
      for (const m of (signals.gmail?.recent || []).slice(0, 3)) {
        const inner = `📧 <strong>${escapeHTML(fromName(m.from))}</strong>: ${escapeHTML(truncate(m.subject || "", 80))}`;
        items.push(`<li>${wrapLink(gmailThreadUrl(m.thread_id), inner)}</li>`);
      }
      // WhatsApp esperando respuesta (los más viejos primero)
      const wa = [...(signals.whatsapp_unreplied || [])]
        .sort((a, b) => (b.hours_waiting || 0) - (a.hours_waiting || 0))
        .slice(0, 3);
      for (const w of wa) {
        const hrs = w.hours_waiting != null ? ` <em>(${Math.round(w.hours_waiting)}h)</em>` : "";
        const inner = `💬 <strong>${escapeHTML(w.name || "")}</strong>${hrs}: ${escapeHTML(truncate(w.last_snippet || "", 70))}`;
        items.push(`<li>${wrapLink(whatsappUrl(w.jid), inner)}</li>`);
      }
      // Apple Mail unread (si hay y no se duplicó con gmail)
      const mailUnread = (signals.mail_unread || []).slice(0, 2);
      if (!signals.gmail?.recent?.length && mailUnread.length) {
        for (const m of mailUnread) {
          items.push(`<li>📬 <strong>${escapeHTML(fromName(m.from || m.sender))}</strong>: ${escapeHTML(truncate(m.subject || "", 80))}</li>`);
        }
      }
      inboxHTML = items.length ? `<ul>${items.join("")}</ul>` : "";
    }

    // Sub-section: 🔍 "Preguntas abiertas" — low-conf queries +
    // followup-aging stale + contradicciones detectadas. Si nada de
    // eso, queda con el placeholder.
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
      // Low-conf queries (queries que el RAG no contestó bien) → /chat?q=
      for (const q of lowConfItems.slice(0, 3)) {
        const text = q.q || q.question || q.text || "";
        const score = q.score != null
          ? ` <em>(${(typeof q.score === "number" ? q.score : Number(q.score)).toFixed(2)})</em>`
          : "";
        const inner = `❓ ${escapeHTML(text)}${score}`;
        const url = text ? `/chat?q=${encodeURIComponent(text)}` : null;
        items.push(`<li>${wrapLink(url, inner)}</li>`);
      }
      // Cabos sueltos viejos (followup-aging stale ≥30d)
      const aging = signals.followup_aging || {};
      const stalePlus = aging.stale_30plus || aging.stale_count || 0;
      if (!items.length && stalePlus > 0) {
        items.push(`<li>⚠️ <strong>${stalePlus}</strong> loops STALE (≥30d sin avance)</li>`);
      }
      // Contradicciones recientes — clickeables a la nota subject
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
      // Loops activos — clickeables a la nota source
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

    // Sub-section: 🌅 "Para mañana" — events del calendar + reminders
    // que vencen mañana. Bug fix: tomorrow_calendar es array directo.
    let tomorrowHTML = "";
    if (split.tomorrow) {
      tomorrowHTML = mdToHTML(split.tomorrow);
    } else {
      const items = [];
      // Events del calendar de mañana
      for (const e of tomorrowEvents.slice(0, 6)) {
        const tr = (e.time_range || "").trim();
        const prefix = tr || "todo el día";
        const title = e.title || e.summary || "(sin título)";
        items.push(`<li><strong>${escapeHTML(prefix)}</strong> — ${escapeHTML(title)}</li>`);
      }
      // Reminders de mañana (si hay y no duplican calendar)
      const reminders = (signals.reminders || []).filter((r) => {
        const due = (r.due || r.due_at || "").toLowerCase();
        return due.includes("tomorrow") || due.includes("mañana");
      }).slice(0, 3);
      for (const r of reminders) {
        items.push(`<li>📌 ${escapeHTML(truncate(r.title || r.text || "", 80))}</li>`);
      }
      tomorrowHTML = items.length ? `<ul>${items.join("")}</ul>` : "";
    }

    // Counts visibles en el chip de cada sección — usar lo que hay
    // realmente, no solo el bucket "oficial".
    const inboxCount = inboxItems.length
      + (signals.gmail?.recent?.length || 0)
      + (signals.whatsapp_unreplied?.length || 0);
    const questionsCount = lowConfItems.length
      + ((signals.followup_aging?.stale_30plus || 0) > 0 ? 1 : 0)
      + (evidence.new_contradictions?.length || 0);
    const tomorrowCount = tomorrowEvents.length;

    bodyEl.innerHTML = [
      sectionHTML("s-narrative", "🪞", "Lo que pasó hoy", null, narrativeHTML, "Aún sin brief — pulsá ↻ arriba para generar"),
      sectionHTML("s-inbox", "📥", "Sin procesar", inboxCount || null, inboxHTML, "todo procesado ✓"),
      sectionHTML("s-questions", "🔍", "Preguntas abiertas", questionsCount || null, questionsHTML, "sin preguntas pendientes"),
      sectionHTML("s-tomorrow", "🌅", "Para mañana", tomorrowCount || null, tomorrowHTML, "agenda libre · día abierto"),
    ].join("");

    // Botones inline "crear reminder" en cada <li> de "Para mañana"
    injectTomorrowReminderButtons();
  }

  // Tracks created reminders per session — re-renders no debe poder hacer
  // doble-create del mismo texto.
  const _createdReminderTexts = new Set();

  function injectTomorrowReminderButtons() {
    const root = document.querySelector(".hero-section.s-tomorrow .prose");
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
      if (_createdReminderTexts.has(original)) {
        li.classList.add("reminder-created");
        btn.disabled = true;
        btn.title = "Ya agregado a Reminders";
      }
    });
  }

  // Click handler para los botones inline
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

  // Refresh manual del brief vía SSE (/api/home/stream). El endpoint
  // emite eventos `stage` por cada uno de los ~14 fetchers + sub-stages
  // de signals, y un `done` final con el payload. La barra se actualiza
  // basado en cuántos stages completaron / total esperado, así refleja
  // progress REAL en lugar de un timer fake. El user reportaba que la
  // barra se clavaba en 95% mientras el compute seguía corriendo —
  // causa: setInterval avanzaba a ritmo fijo (5%/seg) que llegaba a 95%
  // antes que el compute terminara (30-45s).
  document.addEventListener("DOMContentLoaded", () => {
    const refreshBtn = document.getElementById("brief-refresh");
    if (!refreshBtn) return;
    const progressEl = document.getElementById("hero-progress");
    const progressBar = document.getElementById("progress-bar");
    const progressLabel = document.getElementById("progress-label");
    let activeStream = null;
    let trickleTimer = null;

    // El SSE endpoint emite un evento `hello` al inicio con la lista
    // exacta de stages que va a ejecutar. Usamos ese count para %
    // 100% real. Default 26 hasta que llegue el hello (cubre el gap
    // de los primeros milisegundos).
    let estimatedTotalStages = 26;
    // Mapa de etiqueta legible por stage — solo para los principales.
    // Los demás se muestran tal cual ("forecast", "drive", etc).
    const STAGE_LABELS = {
      "today": "leyendo evidencia del día…",
      "signals": "fan-out de 9 señales…",
      "signals.gmail": "consultando Gmail…",
      "signals.whatsapp": "consultando WhatsApp…",
      "signals.calendar": "consultando Calendar…",
      "signals.mail_unread": "consultando Apple Mail…",
      "signals.youtube": "consultando YouTube…",
      "signals.contradictions": "buscando contradicciones…",
      "signals.loops_activo": "rastreando loops…",
      "signals.low_conf": "revisando queries low-conf…",
      "tomorrow": "agenda de mañana…",
      "forecast": "pronóstico del clima…",
      "pagerank": "computando autoridad…",
      "vaults": "actividad del vault…",
      "drive": "Google Drive…",
      "wa_unreplied": "chats WhatsApp pendientes…",
      "bookmarks": "bookmarks de Chrome…",
      "chrome": "top sitios web…",
      "eval": "trend de retrieval…",
      "followup": "loops aging…",
      "finance": "snapshot de finanzas…",
      "cards": "movimientos de tarjeta…",
      "spotify": "Spotify del día…",
      "sleep": "sueño de anoche…",
      "mood": "score de mood…",
      "youtube": "videos vistos…",
      "narrative": "qwen2.5:7b escribiendo…",
      "correlator": "armando patrones cross-source…",
    };

    const totalBlocks = 20;
    function setBar(pct, label) {
      if (!progressBar) return;
      const filled = Math.round((pct / 100) * totalBlocks);
      const bar = "█".repeat(filled) + "░".repeat(totalBlocks - filled);
      progressBar.textContent = `[${bar}] ${Math.round(pct)}%`;
      if (label && progressLabel) progressLabel.textContent = label;
    }

    function startProgressSSE(regenerate) {
      if (!progressEl || !progressBar) return null;
      progressEl.hidden = false;
      let pct = 0;
      let donesSeen = 0;
      let lastStageTs = Date.now();
      let lastStageLabel = "leyendo señales…";
      setBar(0, "iniciando compute…");

      const url = `/api/home/stream${regenerate ? "?regenerate=true" : ""}`;
      const es = new EventSource(url);

      // El server emite `hello` con la lista exacta de stages al
      // arranque. Recalibramos el total para que % refleje progress
      // real al 100% en lugar de un estimate.
      es.addEventListener("hello", (e) => {
        try {
          const data = JSON.parse(e.data);
          const main = (data.stages || []).length;
          const sub = Object.values(data.substages || {})
            .reduce((sum, arr) => sum + (Array.isArray(arr) ? arr.length : 0), 0);
          if (main + sub > 0) {
            estimatedTotalStages = main + sub;
          }
        } catch (err) {
          console.warn("[home.v2] hello event parse failed:", err);
        }
      });

      es.addEventListener("stage", (e) => {
        try {
          const data = JSON.parse(e.data);
          const stage = data.stage || "";
          const status = data.status || "";
          if (status === "done" || status === "timeout" || status === "error") {
            donesSeen++;
            // % real basado en stages que terminaron, capped a 92%
            // (los últimos 8% para el render del payload + final).
            const real = Math.min(92, Math.round((donesSeen / estimatedTotalStages) * 92));
            pct = Math.max(pct, real);
          } else if (status === "start") {
            const label = STAGE_LABELS[stage] || stage.replace(/_/g, " ") + "…";
            lastStageLabel = label;
            lastStageTs = Date.now();
          }
          setBar(pct, lastStageLabel);
        } catch (err) {
          console.warn("[home.v2] stage event parse failed:", err);
        }
      });

      es.addEventListener("done", (e) => {
        try {
          const payload = JSON.parse(e.data);
          _currentPayload = payload;
          setBar(100, "listo!");
          render(payload);
        } catch (err) {
          console.error("[home.v2] done event parse failed:", err);
        } finally {
          stopProgress();
        }
      });

      es.addEventListener("error", (e) => {
        // EventSource auto-reconnects on transient errors. Solo cortamos
        // si el server cerró cleanly (readyState=CLOSED).
        if (es.readyState === EventSource.CLOSED) {
          console.warn("[home.v2] SSE closed unexpectedly");
          stopProgress();
        }
      });

      // Trickle: si el server tarda en mandar `stage` events (LLM corriendo
      // sin sub-eventos), avanzamos un poco para que el bar no se sienta
      // congelado. Max +0.3% por segundo, hasta 90%.
      trickleTimer = setInterval(() => {
        const sinceLast = (Date.now() - lastStageTs) / 1000;
        if (sinceLast > 1 && pct < 90) {
          pct = Math.min(90, pct + 0.3);
          setBar(pct, lastStageLabel);
        }
      }, 1000);

      return es;
    }

    function stopProgress() {
      if (activeStream) {
        try { activeStream.close(); } catch {}
        activeStream = null;
      }
      if (trickleTimer) {
        clearInterval(trickleTimer);
        trickleTimer = null;
      }
      if (progressEl) progressEl.hidden = true;
      refreshBtn.disabled = false;
      refreshBtn.textContent = "↻";
    }

    refreshBtn.addEventListener("click", () => {
      if (activeStream) return;   // ya hay uno corriendo
      refreshBtn.disabled = true;
      refreshBtn.textContent = "↻";
      activeStream = startProgressSSE(true);
      // Safety timeout: si por algún motivo el SSE no termina en 90s,
      // forzamos un fallback al endpoint normal y cerramos el stream.
      setTimeout(() => {
        if (activeStream) {
          console.warn("[home.v2] SSE timeout — falling back to /api/home");
          stopProgress();
          load({ regenerate: false });   // sirve cached
        }
      }, 90_000);
    });
  });

  // Hero collapse toggle (Item A): click on the ▼/▶ button toggles
  // `data-collapsed` on the .today-hero and updates aria-expanded.
  // Estado persistido en localStorage para que sobreviva al reload.
  const LS_HERO_COLLAPSED = "home.v2.hero.collapsed.v1";
  document.addEventListener("DOMContentLoaded", () => {
    const heroToggle = document.getElementById("hero-toggle");
    if (!heroToggle) return;
    const hero = heroToggle.closest(".today-hero");
    if (!hero) return;
    // Restaurar estado previo
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
      // Botón "↺ layout" aparece si hay layout custom
      try { updateResetButtonVisibility(); } catch {}
    });
  });

  // Section collapse toggles (Item F): each .section-toggle button
  // toggles its parent .section's `data-collapsed`. On mobile we
  // start with Monitoring + Ambiente collapsed by default to reduce
  // initial scroll; user expands them with tap. Hero + Accionable
  // stay expanded. Si el user toca el toggle, el estado persiste y le
  // gana al default mobile (si ya guardó "false" en mobile, lo dejamos
  // expandido aunque sea Monitoring/Ambient).
  const LS_SECTIONS_COLLAPSED = "home.v2.sections.collapsed.v1";
  function readSectionsCollapsed() {
    try {
      const raw = localStorage.getItem(LS_SECTIONS_COLLAPSED);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return (parsed && typeof parsed === "object") ? parsed : {};
    } catch { return {}; }
  }
  function writeSectionsCollapsed(map) {
    try {
      // Limpiar entries falsy para JSON chico
      const trimmed = {};
      for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
      if (Object.keys(trimmed).length === 0) {
        localStorage.removeItem(LS_SECTIONS_COLLAPSED);
      } else {
        localStorage.setItem(LS_SECTIONS_COLLAPSED, JSON.stringify(trimmed));
      }
    } catch {}
  }
  function initCollapsibleSections() {
    const isMobile = window.matchMedia("(max-width: 720px)").matches;
    const saved = readSectionsCollapsed();
    document.querySelectorAll(".section-toggle").forEach((btn) => {
      const section = btn.closest(".section");
      if (!section) return;
      // Identificar la sección por la clase específica (section-monitoring,
      // section-ambient, section-accionable, etc) — la usamos como key
      // estable en el localStorage.
      const key = Array.from(section.classList).find(
        (c) => c.startsWith("section-") && c !== "section",
      );
      if (!key) return;
      // Decidir estado inicial: saved > default mobile > expandido.
      // Hay key guardada: respetar lo que guardó el user (true o false).
      // No hay key + mobile + monitoring/ambient: collapse default.
      // Resto: expandido.
      let shouldCollapse;
      if (Object.prototype.hasOwnProperty.call(saved, key)) {
        shouldCollapse = !!saved[key];
      } else {
        shouldCollapse = isMobile && (
          key === "section-monitoring" || key === "section-ambient"
        );
      }
      if (shouldCollapse) {
        section.setAttribute("data-collapsed", "true");
        btn.setAttribute("aria-expanded", "false");
      }
      btn.addEventListener("click", (e) => {
        // Permitir click en el toggle button sin scroll
        e.preventDefault();
        const collapsed = section.getAttribute("data-collapsed") === "true";
        const next = !collapsed;
        section.setAttribute("data-collapsed", next ? "true" : "false");
        btn.setAttribute("aria-expanded", next ? "false" : "true");
        // Persistir el estado nuevo (true/false explícito, no solo
        // collapsed: false sobrevive al default mobile).
        const map = readSectionsCollapsed();
        map[key] = next;
        writeSectionsCollapsed(map);
        // Botón "↺ layout" aparece si hay layout custom
        try { updateResetButtonVisibility(); } catch {}
      });
    });
  }
  document.addEventListener("DOMContentLoaded", initCollapsibleSections);

  function renderFinance(payload) {
    const fin = payload.signals?.finance;
    const panel = document.getElementById("p-finance");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    const foot = panel.querySelector("[data-foot]");
    // Shape real (verificado 2026-04-27): fin.ars es OBJECT con {this_month,
    // prev_month, delta_pct, run_rate_daily, projected, top_categories[]}.
    if (!fin || !fin.ars || typeof fin.ars !== "object") {
      panel.classList.add("is-empty");
      body.innerHTML = `<div class="empty">finanzas no disponibles</div>`;
      count.textContent = "—";
      return;
    }
    panel.classList.remove("is-empty");
    const arsThis = Number(fin.ars.this_month) || 0;
    const arsProj = Number(fin.ars.projected) || null;
    const deltaPct = Number(fin.ars.delta_pct);
    const trendDir = !Number.isFinite(deltaPct) ? "flat"
      : deltaPct > 5 ? "up"
      : deltaPct < -5 ? "down"
      : "flat";
    const trendText = !Number.isFinite(deltaPct) ? ""
      : `${deltaPct > 0 ? "▲ +" : "▼ "}${Math.abs(deltaPct).toFixed(1)}% vs mes ant`;
    // Top categories como bullet
    const top = (fin.ars.top_categories || []).slice(0, 4);
    const topRows = top.map((c) => {
      const sharePct = Math.round((c.share || 0) * 100);
      const barLen = Math.max(1, Math.round(sharePct / 5));
      const bar = "█".repeat(barLen);
      return `<div class="row" style="padding: 3px 0; border: 0;">
        <div class="row-main">
          <div class="row-title" style="display:flex;justify-content:space-between;font-size:12px;">
            <span>${escapeHTML(c.name)}</span>
            <span style="color:var(--text-faint);">${sharePct}%</span>
          </div>
          <div class="ascii-bar" style="margin-top: 2px;">${bar}</div>
        </div>
        <span class="row-aside">${fmtCurrencyARS(c.amount)}</span>
      </div>`;
    }).join("");

    body.innerHTML = `
      <div class="panel-kpi">
        <span class="value">${fmtCurrencyARS(arsThis)}</span>
        <span class="delta ${trendDir}">${escapeHTML(trendText)}</span>
      </div>
      <div class="row-meta" style="margin-top: 6px; margin-bottom: 12px;">
        ${fin.month_label ? `<span>${escapeHTML(fin.month_label)}</span>` : ""}
        ${fin.days_elapsed && fin.days_in_month ? `<span>día ${fin.days_elapsed}/${fin.days_in_month}</span>` : ""}
        ${arsProj ? `<span>proy ${fmtCurrencyARS(arsProj)}</span>` : ""}
      </div>
      ${topRows}
    `;
    count.textContent = fmtCurrencyARS(arsThis);
    if (foot) foot.textContent = fin.source_file ? fin.source_file.split("/").pop() : "";
  }

  function renderCards(payload) {
    // Renombrado de "Tarjetas · ciclo" → "Últimos gastos" a pedido del
    // user. La data de cards.all_purchases_ars / all_purchases_usd ya
    // viene en signals.cards[0]; solo cambiamos qué de eso mostramos.
    // Listamos las top compras por monto descendente del último ciclo
    // (las más relevantes), agrupadas por tarjeta cuando hay >1.
    const cards = payload.signals?.cards || [];
    const panel = document.getElementById("p-cards");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    if (!cards.length) {
      body.innerHTML = `<div class="empty">sin datos de tarjetas</div>`;
      count.textContent = "—";
      return;
    }

    // Recolectar todas las purchases de todas las cards, ordenar por fecha desc
    const allPurchases = [];
    for (const c of cards) {
      const cardLabel = `${c.brand || ""} ····${c.last4 || "????"}`;
      for (const p of c.all_purchases_ars || c.top_purchases_ars || []) {
        allPurchases.push({ ...p, _card: cardLabel, _curr: "ARS" });
      }
      for (const p of c.all_purchases_usd || c.top_purchases_usd || []) {
        allPurchases.push({ ...p, _card: cardLabel, _curr: "USD" });
      }
    }
    if (!allPurchases.length) {
      body.innerHTML = `<div class="empty">sin movimientos en el último ciclo</div>`;
      count.textContent = "—";
      return;
    }

    // Sort: prioritize by date desc; el server ya los ordena pero por las dudas
    allPurchases.sort((a, b) => (b.date || "").localeCompare(a.date || ""));

    // Limpiar descripciones tipo "Merpago*idilicadeco" → "Idilicadeco"
    const cleanDesc = (s) => {
      if (!s) return "";
      return s
        .replace(/^(Merpago|Mercpago|Payu|Dlo|Pago tic)\*+/i, "")
        .replace(/\s+\d{6,}$/, "")
        .replace(/\b\w/g, (c, i, str) => i === 0 || str[i-1] === " " ? c.toUpperCase() : c)
        .slice(0, 50);
    };
    const fmtAmount = (n, curr) => curr === "USD"
      ? `US$ ${Number(n).toFixed(2)}`
      : `$${Math.round(Number(n)).toLocaleString("es-AR")}`;
    const fmtDate = (d) => {
      if (!d) return "";
      try {
        const dt = new Date(d + "T12:00");
        return dt.toLocaleDateString("es-AR", { day: "2-digit", month: "short" });
      } catch { return d; }
    };

    const showCardLabel = cards.length > 1;
    const rows = allPurchases.slice(0, 6).map((p) => ({
      title: cleanDesc(p.description),
      meta: [
        fmtDate(p.date),
        showCardLabel ? p._card : null,
      ].filter(Boolean),
      aside: fmtAmount(p.amount, p._curr),
    }));
    renderPanelList("p-cards", rows, {
      footText: cards.length === 1 ? `${cards[0].brand} ····${cards[0].last4}` : `${cards.length} tarjetas`,
    });
    count.textContent = String(allPurchases.length);
  }

  function renderRetrievalHealth(payload) {
    const trend = payload.signals?.eval_trend;
    const panel = document.getElementById("p-retrieval");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    // Shape real (verificado 2026-04-27): trend.latest = {ts, singles:{hit5,
    // mrr, n, ...}, chains:{...}, latency:{...}}; trend.baseline =
    // {singles_hit5, chains_hit5, ...}; history idem latest.
    if (!trend || !trend.latest) {
      panel.classList.add("is-empty");
      body.innerHTML = `<div class="empty">sin datos de eval reciente</div>`;
      count.textContent = "—";
      return;
    }
    panel.classList.remove("is-empty");
    const singles = trend.latest.singles || {};
    const chains = trend.latest.chains || {};
    const baseline = trend.baseline || {};
    const hit5Singles = Number(singles.hit5) || 0;
    const hit5Chains = Number(chains.hit5) || 0;
    const baseHit5Singles = Number(baseline.singles_hit5) || 0;
    const deltaSingles = baseHit5Singles ? (hit5Singles - baseHit5Singles) * 100 : null;
    const tone = deltaSingles != null && deltaSingles < -5 ? "critical"
      : deltaSingles != null && deltaSingles < 0 ? "warning"
      : "ok";
    const history = (trend.history || []).map((h) => h.singles?.hit5).filter((v) => Number.isFinite(v));
    const sparkSvg = history.length >= 2
      ? sparkline(history, tone === "critical" ? "crit" : tone === "warning" ? "warn" : "ok")
      : "";

    body.innerHTML = `
      <div class="panel-kpi">
        <span class="value">${(hit5Singles * 100).toFixed(1)}%</span>
        ${deltaSingles != null ? `<span class="delta ${deltaSingles < 0 ? "up" : "down"}">${deltaSingles > 0 ? "+" : ""}${deltaSingles.toFixed(1)}pp vs base</span>` : ""}
      </div>
      ${sparkSvg}
      <div class="row-meta" style="margin-top: 6px; flex-direction: column; align-items: stretch; gap: 2px;">
        <span>singles · n=${singles.n || "—"} · MRR ${(Number(singles.mrr) * 100).toFixed(0)}%</span>
        <span>chains · n=${chains.turns || "—"} · hit@5 ${(hit5Chains * 100).toFixed(0)}%</span>
      </div>
    `;
    count.textContent = `${(hit5Singles * 100).toFixed(0)}%`;
    count.classList.remove("has-items", "has-warning", "has-critical");
    count.classList.add(tone === "critical" ? "has-critical" : tone === "warning" ? "has-warning" : "has-items");
  }

  function renderLoopsAging(payload) {
    const f = payload.signals?.followup_aging;
    const panel = document.getElementById("p-loops-aging");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");

    // Bug fix nombres de keys: el server emite `buckets["0_7"]`,
    // `buckets["8_30"]`, `buckets["stale_30plus"]` (con guión bajo),
    // pero el render previo buscaba `buckets["0-7d"]` etc. (con guión y
    // sufijo "d"). Mismatch silencioso → todos quedaban en 0 aunque la
    // data llegara. Fallback compat con keys viejas.
    let fresh = 0, aging = 0, stale = 0;
    let total = 0;
    let sample = [];
    if (f && f.buckets) {
      const b = f.buckets;
      fresh = Number(b["0_7"] ?? b["0-7d"] ?? b.fresh ?? 0);
      aging = Number(b["8_30"] ?? b["8-30d"] ?? b.aging ?? 0);
      stale = Number(b["stale_30plus"] ?? b["stale"] ?? b.STALE ?? 0);
      total = Number(f.total ?? (fresh + aging + stale));
      sample = f.sample || [];
    }

    // Si el cache de followup_aging está cold y el fetcher devolvió
    // null (timeout 5s), fallback derivamos los buckets de
    // signals.loops_stale + loops_activo (que sí vienen sync). Es una
    // aproximación — los buckets de aging real necesitan un LLM-judge
    // per loop — pero al menos el panel deja de mentir "sin datos".
    if (total === 0) {
      const loopsStale = (payload.signals?.loops_stale || []).length;
      const loopsActivo = (payload.signals?.loops_activo || []).length;
      if (loopsStale > 0 || loopsActivo > 0) {
        stale = loopsStale;
        // No tenemos breakdown 0_7 vs 8_30 sin LLM-judge, pero podemos
        // distribuir loops_activo por age_days si existe.
        for (const l of payload.signals?.loops_activo || []) {
          const age = Number(l.age_days || 0);
          if (age >= 8) aging++;
          else fresh++;
        }
        total = fresh + aging + stale;
        // Sample: los más viejos (stale primero, después loops_activo)
        sample = [
          ...(payload.signals?.loops_stale || []).slice(0, 2),
          ...(payload.signals?.loops_activo || []).slice(0, 2),
        ];
      }
    }

    if (total === 0) {
      body.innerHTML = `<div class="empty">sin loops abiertos · todo cerrado</div>`;
      count.textContent = "0";
      return;
    }

    body.innerHTML = `
      <div class="panel-kpi">
        <span class="value">${total}</span>
        ${stale > 0 ? `<span class="delta up">${stale} STALE</span>` : `<span class="delta down">tranquilo</span>`}
      </div>
      ${stackedBar({ fresh, aging, stale })}
      ${sample.slice(0, 3).length ? `
        <div class="row-meta" style="margin-top: 8px; flex-direction: column; align-items: stretch;">
          ${sample.slice(0, 3).map((s) =>
            `<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">· ${escapeHTML((s.loop || s.loop_text || s.text || "").slice(0, 60))}</div>`
          ).join("")}
        </div>
      ` : ""}
    `;
    count.textContent = String(total);
  }

  function renderAuthority(payload) {
    // Renombrado: "Autoridad · top 5" → "Top contactos · 7d" a pedido
    // del user que no entendía qué significaba "autoridad" (era pagerank
    // de notas — métrica técnica). El panel ahora muestra las personas
    // con más interacción cross-source en la última semana, derivado de
    // gmail.recent + whatsapp_unreplied (los chats activos pesan).
    // Mucho más accionable: "estos son los que esperan respuesta".
    //
    // Usa los helpers `isBotOrSelf` y `parseSenderName` definidos al
    // tope del IIFE — filtran github notifs, country club bots, y el
    // propio user (forwards/sent que aparecen como "Fer F.").
    const signals = payload.signals || {};
    const counts = new Map();   // key=name → {sources:Set, count:int, lastTs}
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

  function renderEvalTrend(payload) {
    const trend = payload.signals?.eval_trend;
    const panel = document.getElementById("p-eval-trend");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");
    if (!trend || !trend.history?.length) {
      panel.classList.add("is-empty");
      body.innerHTML = `<div class="empty">sin historial</div>`;
      count.textContent = "—";
      return;
    }
    panel.classList.remove("is-empty");
    const hist = trend.history;
    // Shape: cada history item tiene singles.hit5, singles.mrr, chains.hit5
    const hit5Singles = hist.map((h) => Number(h.singles?.hit5)).filter((v) => Number.isFinite(v));
    const hit5Chains = hist.map((h) => Number(h.chains?.hit5)).filter((v) => Number.isFinite(v));
    const mrrSingles = hist.map((h) => Number(h.singles?.mrr)).filter((v) => Number.isFinite(v));
    body.innerHTML = `
      <div style="font-size:11px;color:var(--text-dim);margin-bottom:2px;">hit@5 singles</div>
      ${sparkline(hit5Singles, "info")}
      <div style="font-size:11px;color:var(--text-dim);margin: 8px 0 2px;">hit@5 chains</div>
      ${sparkline(hit5Chains, "info")}
      <div style="font-size:11px;color:var(--text-dim);margin: 8px 0 2px;">MRR singles</div>
      ${sparkline(mrrSingles, "info")}
    `;
    count.textContent = hist.length;
  }

  function renderWeather(payload) {
    // Shape real (verificado en /api/home, weather_forecast):
    //   { location, current: {description, temp_C}, days: [
    //     {date, minC, maxC, avgC, description, chanceofrain,
    //      chanceofthunder}, ...] }
    // Bug previo: el render usaba `d.summary` / `d.temp_min` / `d.temp_max`
    // que NO existen en este shape. El panel quedaba mostrando solo la
    // fecha sin temperatura ni descripción — por eso el user dijo "no
    // tiene sentido, solo muestra la fecha".
    const wf = payload.weather_forecast;
    const panel = document.getElementById("p-weather");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");

    if (!wf || (!wf.days && !wf.current && !Array.isArray(wf))) {
      body.innerHTML = `<div class="empty">sin datos del clima</div>`;
      count.textContent = "—";
      return;
    }

    // Header: localización + condición actual + temperatura
    const loc = wf.location || "";
    const cur = wf.current || {};
    const headerParts = [];
    if (loc) headerParts.push(escapeHTML(loc.split(",")[0]));
    if (cur.description) headerParts.push(escapeHTML(cur.description));
    if (cur.temp_C != null) headerParts.push(`${cur.temp_C}°C`);
    const headerHTML = headerParts.length
      ? `<div class="row-meta" style="margin-bottom: var(--space-3); font-size: 13px; color: var(--text);">
          ${headerParts.join(" · ")}
         </div>`
      : "";

    // Pronóstico de los próximos días
    const days = Array.isArray(wf) ? wf : (wf.days || wf.forecast || []);
    const dayIcon = (desc) => {
      const d = (desc || "").toLowerCase();
      if (/lluvi|chuva|rain/.test(d)) return "🌧";
      if (/tormen|trueno|thunder/.test(d)) return "⛈";
      if (/nub|nublad|cloud/.test(d)) return "☁";
      if (/parcial|partly/.test(d)) return "⛅";
      if (/despej|sole|clear|sun/.test(d)) return "☀";
      if (/niebl|fog|mist/.test(d)) return "🌫";
      return "·";
    };
    const dayLabel = (dateStr) => {
      if (!dateStr) return "";
      // Usar fecha LOCAL (no UTC) para que "hoy"/"mañana" no shifteen
      // por timezone. Bug previo: `new Date().toISOString()` da UTC,
      // así que después de 21:00 ART (UTC-3) "hoy" se calculaba como
      // el día siguiente y el panel mostraba "jue 30 → hoy → mañana"
      // off-by-one.
      const localDate = (offsetDays = 0) => {
        const d = new Date();
        d.setDate(d.getDate() + offsetDays);
        return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
      };
      if (dateStr === localDate(0)) return "hoy";
      if (dateStr === localDate(1)) return "mañana";
      try {
        const d = new Date(dateStr + "T12:00");  // mediodía evita DST/timezone edge
        return d.toLocaleDateString("es-AR", { weekday: "short", day: "2-digit" });
      } catch { return dateStr.slice(5); }
    };

    const rows = days.slice(0, 4).map((d) => {
      const icon = dayIcon(d.description);
      const tempRange = (d.minC != null && d.maxC != null)
        ? `${d.minC}°–${d.maxC}°`
        : (d.avgC != null ? `${d.avgC}°` : "");
      const rain = Number(d.chanceofrain) || 0;
      const metaBits = [
        d.description ? escapeHTML(d.description) : "",
        rain >= 30 ? `💧 ${rain}%` : "",
      ].filter(Boolean);
      return `<div class="row" style="padding: 4px 0;">
        <div class="row-main">
          <div class="row-title" style="display: flex; gap: 8px; align-items: center;">
            <span style="font-size: 16px; min-width: 20px;">${icon}</span>
            <span><strong>${escapeHTML(dayLabel(d.date))}</strong> · ${tempRange}</span>
          </div>
          ${metaBits.length ? `<div class="row-meta" style="margin-left: 28px;">${metaBits.join(" · ")}</div>` : ""}
        </div>
      </div>`;
    });

    body.innerHTML = headerHTML + rows.join("");
    count.textContent = cur.temp_C != null ? `${cur.temp_C}°C` : (days.length ? `${days.length}d` : "—");
  }

  function renderVaultActivity(payload) {
    // Bug fix multi-vault: el código previo solo leía `act.home` y
    // ignoraba todos los demás vaults registrados (work, etc). Ahora
    // itera todos los vaults del payload, etiqueta cada item con su
    // vault de origen, y los ordena por mtime desc cross-vault.
    const act = payload.signals?.vault_activity || {};
    const merged = [];
    for (const [vaultName, items] of Object.entries(act)) {
      if (!Array.isArray(items)) continue;
      for (const it of items) {
        merged.push({ ...it, _vault: vaultName });
      }
    }
    merged.sort((a, b) => (b.modified || "").localeCompare(a.modified || ""));
    const hasMultipleVaults = Object.keys(act).length > 1;
    const rows = merged.slice(0, 6).map((it) => ({
      title: it.title || it.path,
      meta: [
        // Mostrar el vault solo si hay más de uno (sino es ruido)
        hasMultipleVaults ? `[${it._vault}]` : null,
        it.path ? it.path.split("/").slice(0, -1).join("/") : null,
        it.modified ? fmtTimeAgo(it.modified) : null,
      ].filter(Boolean),
      href: obsidianUrl(it.path, it._vault),
    }));
    renderPanelList("p-vault-activity", rows, {
      emptyText: "sin actividad",
    });
  }

  function renderCaptured(payload) {
    // Bug fix: si no hubo capturas hoy, el panel quedaba vacío con
    // "nada capturado hoy" sin info útil. Fallback: si hoy=0, mostrar
    // las últimas capturas de los últimos 7 días desde vault_activity
    // filtradas por path 00-Inbox/. Cambiamos también el footer para
    // indicar el rango cuando estamos en fallback mode.
    const evidence = payload.today?.evidence || {};
    const inboxToday = evidence.inbox_today || [];
    // Fecha local (no UTC) para que el filtro "hoy" sea consistente con
    // la realidad del usuario en ART.
    const localToday = (() => {
      const d = new Date();
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    })();
    const todayItems = inboxToday.filter((it) =>
      (it.modified || "").startsWith(localToday)
    );

    // Hoy hay capturas → mostrarlas (caso ideal)
    if (todayItems.length > 0) {
      const rows = todayItems.slice(0, 6).map((it) => ({
        title: it.title || it.path,
        meta: [
          it.vault ? `[${it.vault}]` : null,
          ...(it.tags || []).slice(0, 3).map((t) => `#${t}`),
          fmtTimeAgo(it.modified),
        ].filter(Boolean),
        href: obsidianUrl(it.path, it._vault || it.vault),
      }));
      renderPanelList("p-captured", rows, {
        emptyText: "nada capturado hoy",
        footText: "items en 00-Inbox · hoy",
      });
      return;
    }

    // Fallback: usar vault_activity (últimas 48h). Primero intentar
    // filtrar por path de Inbox; si no hay matches (ej. el user no
    // captó nada nuevo en 00-Inbox), mostrar las últimas notas del
    // vault — es lo más cercano a "lo último que tocaste". Mejor que
    // mentir "nada capturado".
    const vaultAct = payload.signals?.vault_activity || {};
    const allItems = [];
    for (const [vaultName, items] of Object.entries(vaultAct)) {
      if (!Array.isArray(items)) continue;
      for (const it of items) {
        allItems.push({ ...it, _vault: vaultName });
      }
    }
    allItems.sort((a, b) => (b.modified || "").localeCompare(a.modified || ""));

    // Primer pase: solo notas en Inbox-like folders
    const inboxOnly = allItems.filter((it) => {
      const p = it.path || "";
      return /(?:^|\/)(00-Inbox|0-Inbox|Inbox|Clippings)\//.test(p)
        || /^(00-Inbox|0-Inbox|Inbox|Clippings)\//.test(p);
    });

    let rows;
    let footHint;
    if (inboxOnly.length > 0) {
      rows = inboxOnly.slice(0, 6);
      footHint = "items en Inbox/Clippings · últimas 48h";
    } else if (allItems.length > 0) {
      // Sin items en Inbox pero hay actividad — mostrar últimas notas
      rows = allItems.slice(0, 6);
      footHint = "últimas notas tocadas · 48h";
    } else {
      renderPanelList("p-captured", [], {
        emptyText: "sin actividad en los últimos 2 días",
      });
      return;
    }

    const hasMultipleVaults = Object.keys(vaultAct).length > 1;
    const formattedRows = rows.map((it) => ({
      title: it.title || it.path,
      meta: [
        hasMultipleVaults ? `[${it._vault}]` : null,
        it.path ? it.path.split("/").slice(0, -1).join("/") : null,
        ...(it.tags || []).slice(0, 2).map((t) => `#${t}`),
        fmtTimeAgo(it.modified),
      ].filter(Boolean),
      href: obsidianUrl(it.path, it._vault || it.vault),
    }));
    renderPanelList("p-captured", formattedRows, {
      emptyText: "sin actividad reciente",
      footText: footHint,
    });
  }

  function renderWeb(payload) {
    const items = payload.signals?.chrome_top_week || [];
    const rows = items.slice(0, 5).map((it) => ({
      title: it.title || it.url,
      meta: [
        it.url ? new URL(it.url).hostname : null,
        it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
      ].filter(Boolean),
      aside: it.visit_count ? String(it.visit_count) : null,
      href: it.url,
    }));
    renderPanelList("p-web", rows, { emptyText: "sin actividad" });
  }

  function renderBookmarks(payload) {
    const items = payload.signals?.chrome_bookmarks || [];
    const rows = items.slice(0, 5).map((it) => ({
      title: it.name || it.url,
      meta: [
        it.folder ? it.folder.split("/").pop() : null,
        it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
      ].filter(Boolean),
      aside: it.visit_count ? String(it.visit_count) : null,
      href: it.url,
    }));
    renderPanelList("p-bookmarks", rows, { emptyText: "sin bookmarks recientes" });
  }

  function renderYouTube(payload) {
    const items = payload.signals?.youtube_watched || [];
    const rows = items.slice(0, 5).map((it) => ({
      title: it.title || "",
      meta: [
        it.last_visit_iso ? fmtTimeAgo(it.last_visit_iso) : null,
      ].filter(Boolean),
      href: youtubeUrl(it.video_id) || it.url,
    }));
    renderPanelList("p-youtube", rows, { emptyText: "sin videos" });
  }

  function renderDrive(payload) {
    const items = payload.signals?.drive_recent || [];
    const rows = items.slice(0, 5).map((it) => ({
      title: it.name || "",
      meta: [
        it.modified ? fmtTimeAgo(it.modified) : null,
        it.owner_email ? it.owner_email.split("@")[0] : null,
      ].filter(Boolean),
      href: it.webViewLink || it.web_view_link || it.url || it.link || null,
    }));
    renderPanelList("p-drive", rows, { emptyText: "sin actividad reciente" });
  }

  function renderSpotify(payload) {
    const sp = payload.signals?.spotify;
    const panel = document.getElementById("p-spotify");
    if (!panel) return;
    // Hide whole panel cuando no hay nada — Spotify cerrado + sin
    // historial del día. Mejor que mostrar "sin actividad" vacío
    // (la mayoría del tiempo va a estar así para el primer día post-
    // install antes de que el poller acumule data).
    if (!sp || (!sp.now_playing && !(sp.recent_today || []).length)) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    // Helper local: segundos → "Nm Ns" o "Ns" para tracks de <1 min.
    // Usado en aside del row para mostrar cuánto sonó cada track.
    const fmtSecs = (s) => {
      if (!s || s < 1) return "";
      if (s < 60) return `${Math.round(s)}s`;
      const m = Math.floor(s / 60);
      const sec = s % 60;
      return sec >= 5 ? `${m}m ${Math.round(sec)}s` : `${m}m`;
    };
    // Spotify desktop URI handler — abre el track en el desktop app.
    // Web URL como fallback para cuando no hay desktop instalado.
    const trackHref = (id) => {
      if (!id) return null;
      // `spotify:track:abc` → URI handler nativa. Browser-friendly también
      // (Chrome lo abre directo en Spotify desktop si está instalado).
      return id;
    };

    const np = sp.now_playing;
    const recent = sp.recent_today || [];
    const rows = [];

    if (np) {
      const isPlaying = np.state === "playing";
      const stateBadge = isPlaying ? "▶ ahora" : "⏸ pausado";
      const stateClass = isPlaying
        ? "spotify-state-playing"
        : "spotify-state-paused";
      const meta = [
        { cls: stateClass, text: stateBadge },
        np.artist,
      ];
      if (np.album) meta.push(np.album);
      rows.push({
        title: np.name,
        meta,
        href: trackHref(np.track_id),
      });
    }

    // Tracks recientes que NO son el currently playing (dedupe por
    // track_id). Si Spotify está cerrado, np es null y simplemente
    // mostramos toda la lista reciente.
    const npId = np?.track_id;
    const rest = recent
      .filter((t) => t.track_id !== npId)
      .slice(0, 4);
    for (const t of rest) {
      const meta = [t.artist];
      const ago = fmtTimeAgo(new Date(t.first_seen * 1000).toISOString());
      if (ago && ago !== "ahora") meta.push(ago);
      rows.push({
        title: t.name,
        meta,
        // Solo mostrar duración si es >30s — los polls de 60s no
        // detectan tracks de <30s con resolución útil.
        aside: t.duration_played_s > 30 ? fmtSecs(t.duration_played_s) : null,
        href: trackHref(t.track_id),
      });
    }
    renderPanelList("p-spotify", rows, {
      emptyText: "sin actividad hoy",
      // Count chip muestra TOTAL de tracks distintos hoy (incluye el
      // currently playing si existe). Da idea del "stream" del día.
      showCount: true,
    });
    // Override count para mostrar el TOTAL de hoy (incluyendo np si
    // estaba), no solo los rows visibles. Más informativo que "5".
    const countEl = panel.querySelector("[data-count]");
    if (countEl) {
      const totalToday = recent.length + (np && !recent.some((t) => t.track_id === npId) ? 1 : 0);
      countEl.textContent = totalToday;
    }
  }

  // ──────────────────────────────────────────────────────────────
  // Sleep panel (Pillow + mood self-report)
  // ──────────────────────────────────────────────────────────────

  // Helper: build inline SVG sparkline. `values` may contain nulls
  // (gaps where the user didn't sleep / track). Renders a polyline
  // skipping nulls + a dot for each non-null. Y-axis is inferred
  // from value range (0..1 default for quality), with a small pad.
  function renderSparkline(values, opts = {}) {
    const W = opts.width || 120;
    const H = opts.height || 24;
    const ymin = opts.ymin ?? 0;
    const ymax = opts.ymax ?? 1;
    const padX = 2;
    const padY = 2;
    const innerW = W - 2 * padX;
    const innerH = H - 2 * padY;
    const N = values.length;
    if (N === 0) return "";

    const xFor = (i) => padX + (i / Math.max(1, N - 1)) * innerW;
    const yFor = (v) => {
      const t = Math.max(0, Math.min(1, (v - ymin) / (ymax - ymin)));
      return padY + (1 - t) * innerH;
    };

    // Polyline path — skip null gaps by starting a new sub-path.
    const segs = [];
    let started = false;
    values.forEach((v, i) => {
      if (v == null) { started = false; return; }
      const cmd = started ? "L" : "M";
      segs.push(`${cmd}${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`);
      started = true;
    });
    const pathD = segs.join(" ");

    // Dots — color the last non-null differently.
    const dots = [];
    let lastIdx = -1;
    for (let i = N - 1; i >= 0; i--) {
      if (values[i] != null) { lastIdx = i; break; }
    }
    values.forEach((v, i) => {
      if (v == null) return;
      const cls = i === lastIdx ? "spark-dot last" : "spark-dot";
      dots.push(`<circle class="${cls}" cx="${xFor(i).toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="1.5"></circle>`);
    });

    return `<svg class="spark-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
      <path class="spark-line" d="${pathD}" />
      ${dots.join("")}
    </svg>`;
  }

  // Map de mood label → emoji + texto corto.
  const MOOD_OPTIONS = [
    { key: "good", emoji: "😀", label: "bien" },
    { key: "meh", emoji: "😐", label: "normal" },
    { key: "bad", emoji: "😞", label: "mal" },
  ];

  // Map del wakeup_mood de Pillow (escala 0-3) a label legible.
  const WAKEUP_MOOD_LABELS = {
    0: "—",
    1: "mal",
    2: "normal",
    3: "bien",
  };

  // Mood score → label corto sin ser paternalista.
  // Cada label dura una sola palabra para no incentivar al usuario a leer
  // demasiado en una etiqueta. El número crudo SIEMPRE va al lado para
  // transparencia — preferimos no escudar el dato.
  function moodLabel(score) {
    if (score == null) return "—";
    if (score <= -0.6) return "muy bajo";
    if (score <= -0.3) return "bajo";
    if (score <= -0.1) return "tibio";
    if (score < 0.1)   return "neutro";
    if (score < 0.3)   return "estable";
    if (score < 0.6)   return "arriba";
    return "alto";
  }

  // Score → CSS class para colorear el badge headline. Usa la misma
  // gama que el sparkline (rojo → ámbar → neutro → verde) para que
  // visualmente sean coherentes.
  function moodScoreClass(score) {
    if (score == null) return "mood-score-na";
    if (score <= -0.4) return "mood-score-low";
    if (score <= -0.1) return "mood-score-tepid";
    if (score < 0.2)   return "mood-score-neutral";
    if (score < 0.5)   return "mood-score-up";
    return "mood-score-high";
  }

  // Trend → glyph + texto. NO verbaliza el dato emocional — solo dirección
  // observada vs el promedio de los últimos 7 días.
  function moodTrendBadge(trend) {
    if (trend === "improving") return `<span class="mood-trend up">↑ subiendo</span>`;
    if (trend === "declining") return `<span class="mood-trend down">↓ bajando</span>`;
    return `<span class="mood-trend stable">· estable</span>`;
  }

  // Map source → emoji para chips compactos. Misma paleta que ya usamos
  // en otros paneles (spotify=verde, journal=cuaderno, etc.).
  const MOOD_SOURCE_EMOJI = {
    spotify: "🎧",
    journal: "📓",
    wa_outbound: "💬",
    queries: "🔍",
    calendar: "📅",
    pillow: "🌙",
    manual: "✋",
  };

  // Map signal_kind → texto humano para evidence chips. NO usamos el
  // valor del signal acá — solo el kind. Si quisiéramos más detalle,
  // hay que abrir `rag mood explain` desde la CLI (link al final del panel).
  const MOOD_KIND_LABEL = {
    artist_mood_lookup: "artistas escuchados",
    compulsive_repeat: "repetición de track",
    late_night_listening: "música tarde",
    keyword_negative: "palabras en notas",
    note_sentiment: "tono journal",
    tone_short: "WhatsApp corto",
    existential_pattern: "queries del RAG",
    density_overload: "agenda saturada",
    back_to_back_meetings: "reuniones seguidas",
    wakeup_mood: "despertar (Pillow)",
    fatigue: "fatiga (Pillow)",
    self_report: "self-report manual",
  };

  // Helper: enriched sparkline para mood. Reusa renderSparkline pero
  // agrega:
  //   - <title> en cada dot con date + score (tooltip nativo del browser)
  //   - placeholder visual cuando hay <3 puntos válidos (chart vacío
  //     se ve roto si hay solo 1 dato)
  //   - linea horizontal en y=0 (zero baseline) cuando hay >=3 puntos
  function renderMoodSparkline(values, dates) {
    const validCount = (values || []).filter(v => v != null).length;
    if (validCount === 0) {
      return `<div class="mood-spark-placeholder">acumulando data…</div>`;
    }
    if (validCount < 3) {
      // 1-2 puntos: mostrar valor numérico en lugar de chart casi vacío.
      const lastIdx = values.length - 1 - [...values].reverse().findIndex(v => v != null);
      const last = values[lastIdx];
      const sign = last > 0 ? "+" : "";
      return `<div class="mood-spark-placeholder">
        acumulando data… (${validCount} día${validCount > 1 ? "s" : ""},
        último <span class="${moodScoreClass(last)}">${sign}${last.toFixed(2)}</span>)
      </div>`;
    }
    // ≥ 3 puntos: renderSparkline normal + zero-line + tooltips.
    const W = 160;
    const H = 28;
    const padY = 2;
    // Inline el SVG para inyectar <title> en cada dot + zero-baseline.
    const padX = 2;
    const innerW = W - 2 * padX;
    const innerH = H - 2 * padY;
    const ymin = -1, ymax = 1;
    const N = values.length;
    const xFor = i => padX + (i / Math.max(1, N - 1)) * innerW;
    const yFor = v => {
      const t = Math.max(0, Math.min(1, (v - ymin) / (ymax - ymin)));
      return padY + (1 - t) * innerH;
    };
    // Path skipping nulls.
    const segs = [];
    let started = false;
    values.forEach((v, i) => {
      if (v == null) { started = false; return; }
      const cmd = started ? "L" : "M";
      segs.push(`${cmd}${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`);
      started = true;
    });
    const pathD = segs.join(" ");
    // Last non-null index for "current" dot styling.
    let lastIdx = -1;
    for (let i = N - 1; i >= 0; i--) {
      if (values[i] != null) { lastIdx = i; break; }
    }
    // Dots con <title> tooltip.
    const dots = values.map((v, i) => {
      if (v == null) return "";
      const cls = i === lastIdx ? "spark-dot last" : "spark-dot";
      const sign = v > 0 ? "+" : "";
      const dateStr = (dates && dates[i]) || "";
      return `<circle class="${cls}" cx="${xFor(i).toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="1.5">
        <title>${escapeHTML(dateStr)}: ${sign}${v.toFixed(2)}</title>
      </circle>`;
    }).join("");
    // Zero baseline horizontal.
    const yZero = yFor(0);
    return `<svg class="spark-svg mood-spark-svg" viewBox="0 0 ${W} ${H}"
      preserveAspectRatio="none" role="img"
      aria-label="evolución del score 14 días">
      <line class="spark-zero" x1="${padX}" y1="${yZero.toFixed(1)}"
            x2="${(W - padX).toFixed(1)}" y2="${yZero.toFixed(1)}"></line>
      <path class="spark-line" d="${pathD}"></path>
      ${dots}
    </svg>`;
  }

  // Map del label de self-report al score que va a la DB.
  // Coherente con MOOD_OPTIONS del sleep widget.
  const MOOD_SELF_REPORT_OPTIONS = [
    { key: "good", emoji: "😀", label: "bien" },
    { key: "meh", emoji: "😐", label: "normal" },
    { key: "bad", emoji: "😞", label: "mal" },
    { key: "sad", emoji: "😔", label: "triste" },
  ];

  function renderMood(payload) {
    const m = payload.signals?.mood;
    const panel = document.getElementById("p-mood");
    if (!panel) return;
    if (!m || m.score == null) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    // Headline: score numérico (transparencia) + label corto + trend.
    const scoreClass = moodScoreClass(m.score);
    const sign = m.score > 0 ? "+" : "";
    const scoreText = `${sign}${m.score.toFixed(2)}`;
    const labelText = moodLabel(m.score);

    // Sparkline 14d enriquecido (tooltips + zero baseline + placeholder
    // cuando <3 días con data).
    const sparkVals = m.spark_score_14d || [];
    const sparkDates = m.spark_dates_14d || [];
    const sparkSVG = renderMoodSparkline(sparkVals, sparkDates);

    // Sources chips.
    const sources = m.sources_used || [];
    const sourcesHTML = sources.length
      ? sources.map((s) => {
          const emoji = MOOD_SOURCE_EMOJI[s] || "·";
          return `<span class="mood-src">${emoji} ${escapeHTML(s)}</span>`;
        }).join("")
      : `<span class="mood-src empty">sin sources</span>`;

    // Drift warning factual (no diagnóstico).
    const drift = m.drift || {};
    const driftHTML = drift.drifting
      ? `<div class="mood-drift" role="status">
          ⚠ ${drift.n_consecutive} días consecutivos abajo del baseline
          <span class="muted">(promedio ${drift.avg_score.toFixed(2)})</span>
        </div>`
      : "";

    // Top evidence colapsable. Cada signal muestra el value crudo + el
    // % de contribución al score del día (mide qué tan fuerte movió la
    // señal el agregado vs los otros). El backend computa pct con
    // |value*weight| / total — consistente con /api/mood/history.
    const topEvidence = m.top_evidence || [];
    const evidenceHTML = topEvidence.length
      ? `<details class="mood-evidence">
          <summary>top ${topEvidence.length} señal(es)</summary>
          <ul>${topEvidence.map((e) => {
            const v = e.value;
            const vSign = v > 0 ? "+" : "";
            const vCls = v <= -0.5 ? "neg-strong"
                       : v <= -0.2 ? "neg"
                       : v >= 0.5 ? "pos-strong"
                       : v >= 0.2 ? "pos"
                       : "flat";
            const kindHuman = MOOD_KIND_LABEL[e.signal_kind] || e.signal_kind;
            const srcEmoji = MOOD_SOURCE_EMOJI[e.source] || "·";
            // % contribución (default 0 si el backend no lo manda — tests
            // viejos pueden no incluirlo).
            const pctTxt = (e.pct != null && isFinite(e.pct))
              ? `<span class="pct muted">${e.pct.toFixed(0)}%</span>`
              : "";
            return `<li>
              <span class="src" aria-hidden="true">${srcEmoji}</span>
              <span class="kind">${escapeHTML(kindHuman)}</span>
              <span class="val ${vCls}">${vSign}${v.toFixed(2)}</span>
              ${pctTxt}
            </li>`;
          }).join("")}</ul>
        </details>`
      : "";

    // Quick self-report buttons. Reusa POST /api/mood que ya existe
    // (originariamente del panel sleep). Cierra el loop: tap → signal
    // → siguiente refresh muestra el dato en sources/top_evidence.
    // No marca "selected" persistente porque el panel no sabe cuál
    // fue el último report del user (no lo guardamos en mood payload
    // para mantenerlo simple); sí da feedback visual transitorio.
    const reportBtnsHTML = MOOD_SELF_REPORT_OPTIONS.map((opt) => `
      <button type="button" class="mood-self-btn"
              data-mood="${opt.key}"
              aria-label="reportar mood: ${opt.label}"
              title="${opt.label}">
        <span aria-hidden="true">${opt.emoji}</span>
      </button>
    `).join("");
    const selfReportHTML = `
      <div class="mood-self-report" data-self-report>
        <span class="mood-self-prompt muted">¿cómo te sentís ahora?</span>
        <div class="mood-self-buttons" role="group"
             aria-label="reportar mood actual">
          ${reportBtnsHTML}
        </div>
      </div>
    `;

    const body = panel.querySelector("[data-body]");
    body.innerHTML = `
      <div class="mood-summary">
        <div class="mood-row mood-headline-row">
          <span class="mood-score ${scoreClass}">${scoreText}</span>
          <span class="mood-label">${labelText}</span>
          ${moodTrendBadge(m.trend)}
        </div>
        <div class="mood-sparkline">
          ${sparkSVG}
          <span class="muted spark-meta">14d · vs ${m.week_avg.toFixed(2)} (7d avg)</span>
        </div>
        ${driftHTML}
        <div class="mood-sources">${sourcesHTML}</div>
        ${evidenceHTML}
        ${selfReportHTML}
      </div>
    `;

    // Wire buttons → POST /api/mood (mismo endpoint que el sleep widget).
    const reportWidget = body.querySelector("[data-self-report]");
    reportWidget?.querySelectorAll(".mood-self-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const mood = btn.dataset.mood;
        if (!mood) return;
        // Optimistic UI: marcar como pulsado, deshabilitar el grupo
        // hasta que la respuesta vuelva.
        const allBtns = reportWidget.querySelectorAll(".mood-self-btn");
        allBtns.forEach((b) => {
          b.classList.remove("selected");
          b.disabled = true;
        });
        btn.classList.add("selected");
        const prompt = reportWidget.querySelector(".mood-self-prompt");
        if (prompt) prompt.textContent = "guardando…";
        try {
          const r = await fetch("/api/mood", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mood }),
          });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          if (prompt) prompt.textContent = "guardado ✓";
          // Re-habilitar para que pueda corregir el report.
          allBtns.forEach((b) => { b.disabled = false; });
        } catch (err) {
          console.error("mood self-report failed", err);
          allBtns.forEach((b) => {
            b.disabled = false;
            b.classList.remove("selected");
          });
          if (prompt) prompt.textContent = "error — reintentá";
        }
      });
    });

    // Count chip = n_signals que dispararon el score (transparencia).
    const countEl = panel.querySelector("[data-count]");
    if (countEl) countEl.textContent = m.n_signals;

    // Footer: button "ver historial" que abre el modal con timeline 30d
    // + histograma por source + breakdown granular. Mantiene también el
    // hint CLI por si el user prefiere drill-down via terminal.
    const foot = panel.querySelector("[data-foot]");
    if (foot) {
      foot.innerHTML = `
        <button type="button" class="mood-history-btn"
                aria-label="ver historial detallado del mood"
                data-mood-history-open>
          ver historial 30d
        </button>
        <span class="muted">o <code>rag mood explain</code></span>
      `;
      const openBtn = foot.querySelector("[data-mood-history-open]");
      openBtn?.addEventListener("click", () => openMoodHistoryModal());
    }

    // Fade-in animation cuando el panel pasa de hidden a visible — solo
    // primera vez por session. Honor prefers-reduced-motion (CSS lo
    // decide via @media). Marcamos con data attribute para que la regla
    // CSS aplique solo el primer render.
    if (!panel.dataset.firstShown) {
      panel.dataset.firstShown = "1";
      panel.classList.add("mood-fade-in");
      // Removemos la clase después de la animación para no acumular
      // estados que afecten re-renders.
      setTimeout(() => panel.classList.remove("mood-fade-in"), 600);
    }
  }

  // ── Mood history modal ────────────────────────────────────────────
  // Lazy-load: fetch /api/mood/history?days=30 solo cuando el user
  // abre el modal (no en cada home refresh). Cache simple por
  // requestId — si el user re-abre dentro de la misma sesión, reusamos.
  let _moodHistoryCache = null;
  let _moodHistoryFetching = false;

  async function openMoodHistoryModal() {
    const dlg = document.getElementById("mood-history-modal");
    if (!dlg) return;
    const body = dlg.querySelector("[data-mood-modal-body]");
    if (!body) return;

    // Mostrar el dialog antes de fetchear (better perceived perf — el
    // user ve algo inmediatamente).
    if (typeof dlg.showModal === "function") {
      dlg.showModal();
    } else {
      // Fallback para browsers sin <dialog> (raros en 2026).
      dlg.setAttribute("open", "");
    }

    // Wire close button + escape (escape ya viene gratis con <dialog>).
    const closeBtn = dlg.querySelector("[data-mood-modal-close]");
    if (closeBtn && !closeBtn.dataset.wired) {
      closeBtn.dataset.wired = "1";
      closeBtn.addEventListener("click", () => dlg.close());
    }
    // Click en backdrop cierra (UX standard de modales).
    if (!dlg.dataset.backdropWired) {
      dlg.dataset.backdropWired = "1";
      dlg.addEventListener("click", (e) => {
        const rect = dlg.getBoundingClientRect();
        const inDialog = e.clientX >= rect.left && e.clientX <= rect.right
                       && e.clientY >= rect.top && e.clientY <= rect.bottom;
        if (!inDialog) dlg.close();
      });
    }

    // Cache hit: render rápido sin re-fetch.
    if (_moodHistoryCache) {
      renderMoodHistory(_moodHistoryCache, body);
      return;
    }
    if (_moodHistoryFetching) return;  // race protection
    _moodHistoryFetching = true;
    body.innerHTML = `<div class="empty">cargando…</div>`;
    try {
      const r = await fetch("/api/mood/history?days=30");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      _moodHistoryCache = data;
      renderMoodHistory(data, body);
    } catch (err) {
      console.error("mood history fetch failed", err);
      body.innerHTML = `<div class="empty">no se pudo cargar el historial — reintentá más tarde</div>`;
    } finally {
      _moodHistoryFetching = false;
    }
  }

  function renderMoodHistory(data, body) {
    const days = data.days || [];
    const histogram = data.histogram || [];
    const totalDays = data.total_days_with_data || 0;
    const range = data.range_days || 30;

    if (totalDays === 0) {
      body.innerHTML = `
        <div class="mood-modal-empty">
          <p>todavía no hay data acumulada de mood.</p>
          <p class="muted">activá el daemon con <code>rag mood enable</code>
          y esperá unas horas para que junte señales.</p>
        </div>
      `;
      return;
    }

    // Histogram horizontal: cada source es una bar con su % de
    // contribución total + cantidad de días activos. La idea es ver
    // de un vistazo qué source domina la señal.
    const histHTML = histogram.length
      ? `<section class="mood-modal-section" aria-labelledby="mood-hist-title">
          <h3 id="mood-hist-title">contribución por source · ${range}d</h3>
          <ul class="mood-histogram">${histogram.map((h) => {
            const sign = h.total_contrib > 0 ? "+" : "";
            const sigCls = h.total_contrib > 0 ? "pos" : h.total_contrib < 0 ? "neg" : "flat";
            const emoji = MOOD_SOURCE_EMOJI[h.source] || "·";
            // Bar width = % del total. Min 2% para que se vea aunque
            // contribuya casi nada.
            const w = Math.max(2, h.pct);
            return `<li class="mood-hist-row">
              <span class="hist-src">
                <span aria-hidden="true">${emoji}</span>
                <span>${escapeHTML(h.source)}</span>
              </span>
              <span class="hist-bar-wrap" aria-hidden="true">
                <span class="hist-bar ${sigCls}" style="width: ${w}%"></span>
              </span>
              <span class="hist-pct">${h.pct.toFixed(0)}%</span>
              <span class="hist-contrib ${sigCls}">${sign}${h.total_contrib.toFixed(2)}</span>
              <span class="hist-meta muted">${h.days_active}d · ${h.n_signals} sig</span>
            </li>`;
          }).join("")}</ul>
        </section>`
      : "";

    // Timeline diario: una row por día con sparkbar + breakdown chips.
    // Días sin data se ven como gap (background apagado).
    const timelineHTML = `
      <section class="mood-modal-section" aria-labelledby="mood-timeline-title">
        <h3 id="mood-timeline-title">timeline ${range}d</h3>
        <ul class="mood-timeline">${days.map((d) => {
          const hasData = d.n_signals > 0;
          const score = d.score || 0;
          const sign = score > 0 ? "+" : "";
          const scoreCls = moodScoreClass(score);
          // Bar height proporcional a |score|. -1 = full down, +1 = full up.
          // Centro = neutral.
          const barH = Math.abs(score) * 100;
          const barDir = score < 0 ? "neg" : "pos";
          const sourcesShort = (d.by_source || [])
            .slice(0, 3)
            .map((s) => `<span class="ts-src" title="${escapeHTML(s.source)}: ${s.contrib > 0 ? '+' : ''}${s.contrib.toFixed(2)} (${s.pct.toFixed(0)}%)">${MOOD_SOURCE_EMOJI[s.source] || "·"}</span>`)
            .join("");
          // Date display: solo MM-DD para no saturar.
          const dateShort = (d.date || "").slice(5);
          return `<li class="mood-tl-row ${hasData ? '' : 'no-data'}">
            <span class="tl-date muted">${dateShort}</span>
            <span class="tl-bar-cell" aria-hidden="true">
              ${hasData ? `<span class="tl-bar ${barDir} ${scoreCls}"
                style="height: ${barH}%"></span>` : ''}
            </span>
            <span class="tl-score ${hasData ? scoreCls : ''}">
              ${hasData ? `${sign}${score.toFixed(2)}` : '—'}
            </span>
            <span class="tl-srcs" aria-hidden="true">${sourcesShort}</span>
            <span class="tl-n muted">${hasData ? `${d.n_signals} sig` : ''}</span>
          </li>`;
        }).join("")}</ul>
      </section>
    `;

    body.innerHTML = `
      <div class="mood-modal-content">
        <p class="mood-modal-summary muted">
          ${totalDays} de ${range} días con data · ${histogram.length} sources activas
        </p>
        ${histHTML}
        ${timelineHTML}
      </div>
    `;
  }

  // ── Cross-source patterns panel ──────────────────────────────────────
  // Lazy-load: el panel hace su propio fetch al primer render, no
  // viene en /api/home. Cache simple por session.
  let _patternsCache = null;
  let _patternsFetching = false;

  function patternsSeverityClass(severity) {
    if (severity === "strong") return "patterns-strong";
    if (severity === "moderate") return "patterns-moderate";
    return "patterns-weak";
  }

  function patternsLagLabel(lag) {
    if (lag === 0) return "mismo día";
    if (lag === 1) return "+1 día";
    if (lag === 7) return "+1 semana";
    return `+${lag}d`;
  }

  // Friendly label para metric_name que espeja `metric_label` del
  // backend. Si el server agrega métricas nuevas, este map queda corto
  // y caemos al raw name (acceptable hasta que actualicemos).
  const PATTERNS_METRIC_LABELS = {
    mood_score: "mood",
    mood_self_report: "mood self-report",
    sleep_quality: "sleep quality",
    sleep_duration_h: "sleep horas",
    sleep_awakenings: "awakenings",
    sleep_deep_pct: "sleep deep%",
    wakeup_mood: "wake-up mood",
    spotify_minutes: "spotify min",
    spotify_distinct_tracks: "spotify tracks",
    queries_total: "queries total",
    queries_existential: "queries existencial",
    wa_outbound_avg_chars: "WA chars/msg",
  };

  function patternsLabel(metricName) {
    return PATTERNS_METRIC_LABELS[metricName] || metricName;
  }

  async function fetchPatterns(force = false) {
    if (_patternsCache && !force) return _patternsCache;
    if (_patternsFetching) {
      // Wait for in-flight request.
      while (_patternsFetching) {
        await new Promise((r) => setTimeout(r, 100));
      }
      return _patternsCache;
    }
    _patternsFetching = true;
    try {
      const r = await fetch("/api/patterns?days=30&top=20");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      _patternsCache = await r.json();
      return _patternsCache;
    } catch (err) {
      console.error("patterns fetch failed", err);
      return null;
    } finally {
      _patternsFetching = false;
    }
  }

  async function renderCorrelations(_payload) {
    // El panel se llama "p-correlations" en el DOM porque el
    // nombre "p-patterns" ya estaba tomado por otro panel
    // pre-existente (cross-source de entidades). La función también
    // se llama renderCorrelations (no renderPatterns) para evitar
    // shadowing con la función pre-existente que renderea
    // `p-patterns` (entidades). Bug histórico: ambas se llamaban
    // renderPatterns, JS hoist la última y mi función nunca corría.
    const panel = document.getElementById("p-correlations");
    if (!panel) return;
    // Show panel immediately con "computando…" mientras el endpoint
    // procesa (cómputo Pearson sobre 200 pairs es <1s pero cold-load
    // del módulo Python puede ser hasta 3s en el primer hit).
    panel.hidden = false;
    const body = panel.querySelector("[data-body]");

    const data = await fetchPatterns();
    if (!data) {
      // Error fetching → mostrar mensaje no-paniconante.
      body.innerHTML = `<div class="empty">no se pudo cargar los patrones</div>`;
      return;
    }

    const findings = data.top || [];
    const bySev = data.by_severity || {};
    const metricsWithData = data.metrics_with_data || [];
    const metricsWithEnoughData = metricsWithData.filter(([_, n]) => n >= 21).length;

    if (findings.length === 0) {
      // Empty state: explicar por qué (no hay enough data probablemente).
      const totalMetrics = metricsWithData.length;
      panel.hidden = false;
      body.innerHTML = `
        <div class="patterns-empty">
          <p class="muted">sin patrones detectados todavía</p>
          <p class="patterns-empty-hint">
            ${metricsWithEnoughData}/${totalMetrics} métricas tienen ≥21 días
            (mínimo para correlación significativa).
          </p>
        </div>
      `;
      const countEl = panel.querySelector("[data-count]");
      if (countEl) countEl.textContent = "0";
      const foot = panel.querySelector("[data-foot]");
      if (foot) {
        foot.innerHTML = `<span class="muted">esperando data · <code>rag patterns metrics</code></span>`;
      }
      return;
    }

    // Top 3 findings inline en el panel + button para ver todos.
    const top3 = findings.slice(0, 3);
    const itemsHTML = top3.map((f) => {
      const [a, b] = f.pair;
      const sevCls = patternsSeverityClass(f.severity);
      const lagLbl = patternsLagLabel(f.lag);
      const rSign = f.r > 0 ? "+" : "";
      return `<li class="patterns-row ${sevCls}">
        <span class="patterns-pair">
          <span class="metric">${escapeHTML(patternsLabel(a))}</span>
          <span class="patterns-sep" aria-hidden="true">×</span>
          <span class="metric">${escapeHTML(patternsLabel(b))}</span>
        </span>
        <span class="patterns-lag muted">${escapeHTML(lagLbl)}</span>
        <span class="patterns-r" title="Pearson r=${f.r}, n=${f.n}, p=${f.p}">
          ${rSign}${f.r.toFixed(2)}
        </span>
      </li>`;
    }).join("");

    body.innerHTML = `
      <div class="patterns-summary">
        <ul class="patterns-list">${itemsHTML}</ul>
      </div>
    `;

    const countEl = panel.querySelector("[data-count]");
    if (countEl) countEl.textContent = String(findings.length);

    const foot = panel.querySelector("[data-foot]");
    if (foot) {
      const sevSummary = [];
      if (bySev.strong) sevSummary.push(`${bySev.strong} strong`);
      if (bySev.moderate) sevSummary.push(`${bySev.moderate} moderate`);
      foot.innerHTML = `
        <button type="button" class="patterns-history-btn"
                aria-label="ver todos los patrones detallados"
                data-patterns-history-open>
          ver todos (${findings.length})
        </button>
        ${sevSummary.length ? `<span class="muted">· ${sevSummary.join(" · ")}</span>` : ""}
      `;
      const openBtn = foot.querySelector("[data-patterns-history-open]");
      openBtn?.addEventListener("click", () => openPatternsModal());
    }

    // Fade-in animation primera vez, mismo patron que p-mood.
    if (!panel.dataset.firstShown) {
      panel.dataset.firstShown = "1";
      panel.classList.add("mood-fade-in");
      setTimeout(() => panel.classList.remove("mood-fade-in"), 600);
    }
  }

  // ── Patterns modal — full detail ────────────────────────────────────
  function openPatternsModal() {
    const dlg = document.getElementById("patterns-modal");
    if (!dlg) return;
    const body = dlg.querySelector("[data-patterns-modal-body]");
    if (!body) return;

    if (typeof dlg.showModal === "function") {
      dlg.showModal();
    } else {
      dlg.setAttribute("open", "");
    }

    // Wire close button + backdrop click (idempotente).
    const closeBtn = dlg.querySelector("[data-patterns-modal-close]");
    if (closeBtn && !closeBtn.dataset.wired) {
      closeBtn.dataset.wired = "1";
      closeBtn.addEventListener("click", () => dlg.close());
    }
    if (!dlg.dataset.backdropWired) {
      dlg.dataset.backdropWired = "1";
      dlg.addEventListener("click", (e) => {
        const rect = dlg.getBoundingClientRect();
        const inDialog = e.clientX >= rect.left && e.clientX <= rect.right
                       && e.clientY >= rect.top && e.clientY <= rect.bottom;
        if (!inDialog) dlg.close();
      });
    }

    body.innerHTML = `<div class="empty">cargando…</div>`;
    fetchPatterns().then((data) => {
      if (!data) {
        body.innerHTML = `<div class="empty">no se pudo cargar los patrones</div>`;
        return;
      }
      renderPatternsModal(data, body);
    });
  }

  function renderPatternsModal(data, body) {
    const findings = data.top || [];
    const bySev = data.by_severity || {};
    const metrics = data.metrics_with_data || [];
    const lagsTested = data.lags_tested || [];
    const range = data.days_range || 30;

    // Sección 1: summary stats.
    const summaryHTML = `
      <p class="mood-modal-summary muted">
        ${data.n_findings} correlaciones · ${range} días · lags
        ${lagsTested.join(", ")} · ${metrics.length} métricas con data
      </p>
    `;

    // Sección 2: lista completa de findings.
    const findingsHTML = findings.length
      ? `<section class="mood-modal-section" aria-labelledby="patterns-findings-title">
          <h3 id="patterns-findings-title">correlaciones detectadas</h3>
          <ul class="patterns-full-list">${findings.map((f) => {
            const [a, b] = f.pair;
            const sevCls = patternsSeverityClass(f.severity);
            const lagLbl = patternsLagLabel(f.lag);
            const rSign = f.r > 0 ? "+" : "";
            return `<li class="patterns-full-row ${sevCls}">
              <span class="patterns-sev-dot" aria-hidden="true"></span>
              <span class="patterns-full-pair">
                <span class="metric">${escapeHTML(patternsLabel(a))}</span>
                <span class="patterns-sep" aria-hidden="true">×</span>
                <span class="metric">${escapeHTML(patternsLabel(b))}</span>
              </span>
              <span class="patterns-full-lag">${escapeHTML(lagLbl)}</span>
              <span class="patterns-full-r">${rSign}${f.r.toFixed(2)}</span>
              <span class="patterns-full-meta muted">
                n=${f.n} · p=${f.p.toFixed(3)}
              </span>
              <span class="patterns-full-desc muted">
                ${escapeHTML(f.description)}
              </span>
            </li>`;
          }).join("")}</ul>
        </section>`
      : `<div class="mood-modal-empty">
          <p>sin correlaciones detectadas</p>
          <p class="muted">necesitás ≥21 días de data en ≥2 métricas + |r|≥0.4 + p<0.05.</p>
        </div>`;

    // Sección 3: cobertura por métrica.
    const coverageHTML = metrics.length
      ? `<section class="mood-modal-section" aria-labelledby="patterns-coverage-title">
          <h3 id="patterns-coverage-title">cobertura por métrica · ${range}d</h3>
          <ul class="patterns-coverage">${metrics.map(([name, n]) => {
            const ready = n >= 21;
            const cls = ready ? "ready" : (n >= 7 ? "partial" : "low");
            const w = Math.min(100, (n / range) * 100);
            return `<li class="patterns-cov-row ${cls}">
              <span class="cov-name">${escapeHTML(patternsLabel(name))}</span>
              <span class="cov-bar-wrap" aria-hidden="true">
                <span class="cov-bar" style="width: ${w}%"></span>
              </span>
              <span class="cov-n">${n}d</span>
            </li>`;
          }).join("")}</ul>
        </section>`
      : "";

    body.innerHTML = `
      <div class="mood-modal-content">
        ${summaryHTML}
        ${findingsHTML}
        ${coverageHTML}
      </div>
    `;
  }

  function renderSleep(payload) {
    const sleep = payload.signals?.sleep;
    const panel = document.getElementById("p-sleep");
    if (!panel) return;
    if (!sleep || !sleep.last_night) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    const ln = sleep.last_night;
    const week = sleep.week || {};
    const delta = sleep.delta || {};
    const moodNow = sleep.mood_now;

    // Headline: duración + quality
    const totalH = ln.sleep_total_h || 0;
    const totalLabel = (() => {
      const mins = Math.round(totalH * 60);
      const h = Math.floor(mins / 60);
      const m = mins % 60;
      return `${h}h${m.toString().padStart(2, "0")}m`;
    })();
    const qLabel = ln.quality != null ? `Q ${ln.quality.toFixed(2)}` : "";

    // Stages: deep / REM / awakenings con warn thresholds
    const deepPct = ln.deep_pct;
    const remPct = ln.rem_pct;
    const awak = ln.awakenings ?? 0;
    const deepWarn = deepPct != null && deepPct < 15;
    const remWarn = remPct != null && remPct < 15;
    const awakWarnCls = awak >= 5 ? "stale" : awak >= 3 ? "warn" : "";

    // Delta vs hist — solo mostramos los más relevantes
    const fmtDelta = (val, suffix, decimals = 2) => {
      if (val == null || !isFinite(val)) return null;
      const sign = val > 0 ? "+" : "";
      const cls = Math.abs(val) < 0.01 ? "delta-flat"
                : val > 0 ? "delta-up"
                : "delta-down";
      const arrow = val > 0 ? "↑" : val < 0 ? "↓" : "·";
      return `<span class="${cls}">${arrow} ${sign}${val.toFixed(decimals)}${suffix}</span>`;
    };
    const deltaParts = [
      fmtDelta(delta.duration_h, "h", 1),
      delta.quality != null ? fmtDelta(delta.quality, "Q") : null,
      delta.deep_pct != null ? fmtDelta(delta.deep_pct, "%", 1) : null,
    ].filter(Boolean);

    // Wake-up mood read-only desde Pillow
    const wakeupMood = ln.wakeup_mood;
    const wakeupLabel = wakeupMood != null
      ? `${WAKEUP_MOOD_LABELS[wakeupMood] || "—"}`
      : null;

    // Mood now selected
    const moodNowKey = moodNow?.label;

    // Build mood buttons
    const moodBtns = MOOD_OPTIONS.map((m) => {
      const cls = m.key === moodNowKey ? "mood-btn selected" : "mood-btn";
      return `<button type="button" class="${cls}" data-mood="${m.key}"
        title="${m.label}" aria-label="estado: ${m.label}">${m.emoji}</button>`;
    }).join("");

    const moodTimeAgo = moodNow?.ts
      ? fmtTimeAgo(new Date(moodNow.ts * 1000).toISOString())
      : null;

    // Sparkline values (quality 7d) — null gaps preserved
    const sparkVals = sleep.spark_quality_7d || [];
    const sparkSVG = renderSparkline(sparkVals, { width: 120, height: 22, ymin: 0, ymax: 1 });

    // Insight (anomalía detectada server-side)
    const insightHTML = sleep.insight
      ? `<div class="sleep-insight" role="status">⚠ ${escapeHTML(sleep.insight)}</div>`
      : "";

    // Patterns (correlations Pearson r ≥ 0.3 sobre todo el histórico).
    // Filtramos los obvios (duration↔quality es trivial: dormir más
    // duerme mejor) para que el panel resalte solo lo no-obvio.
    const TRIVIAL_KINDS = new Set(["duration↔quality"]);
    const patternsTop = (sleep.patterns?.top || [])
      .filter((p) => !TRIVIAL_KINDS.has(p.kind))
      .slice(0, 3);
    const patternsHTML = patternsTop.length
      ? `<details class="sleep-patterns">
          <summary>${patternsTop.length} patrones (n=${sleep.patterns.top[0].n})</summary>
          <ul>${patternsTop.map((p) => {
            const sevCls = `sev-${p.severity}`;
            const rSign = p.r > 0 ? "+" : "";
            return `<li class="${sevCls}">
              <span class="desc">${escapeHTML(p.description)}</span>
              <span class="r">r=${rSign}${p.r.toFixed(2)}</span>
              <span class="sev">${p.severity}</span>
            </li>`;
          }).join("")}</ul>
        </details>`
      : "";

    const body = panel.querySelector("[data-body]");
    body.innerHTML = `
      <div class="sleep-summary">
        <div class="sleep-row">
          <span class="sleep-headline">${totalLabel}<span class="quality">${escapeHTML(qLabel)}</span></span>
          <span class="sleep-clock">${ln.bedtime_local || "—"}<span class="arrow">→</span>${ln.waketime_local || "—"}</span>
        </div>
        <div class="sleep-stages">
          <span class="stage stage-deep ${deepWarn ? "warn" : ""}">
            <span class="stage-label">deep</span><span>${deepPct != null ? deepPct.toFixed(0) + "%" : "—"}</span>
          </span>
          <span class="stage stage-rem ${remWarn ? "warn" : ""}">
            <span class="stage-label">rem</span><span>${remPct != null ? remPct.toFixed(0) + "%" : "—"}</span>
          </span>
          <span class="stage stage-awakenings ${awakWarnCls}">
            <span class="stage-label">awk</span><span>${awak}</span>
          </span>
        </div>
        <div class="sleep-sparkline">
          ${sparkSVG}
          <span class="spark-label">Q · 7d</span>
        </div>
        ${deltaParts.length ? `<div class="sleep-delta">vs hist: ${deltaParts.join(" ")}</div>` : ""}
        ${insightHTML}
        ${patternsHTML}
        <div class="sleep-mood" data-mood-widget>
          <span class="mood-label">ahora:</span>
          ${moodBtns}
          <span class="mood-current">
            ${moodNowKey ? `<span class="mood-saved">✓</span>` : ""}
            ${wakeupLabel ? `<span class="wakeup-mood">despertaste: <span class="label">${escapeHTML(wakeupLabel)}</span></span>` : ""}
            ${moodTimeAgo ? `<span class="when">${escapeHTML(moodTimeAgo)}</span>` : ""}
          </span>
        </div>
      </div>
    `;

    // Wire mood buttons → POST /api/mood
    const widget = body.querySelector("[data-mood-widget]");
    widget?.querySelectorAll(".mood-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const mood = btn.dataset.mood;
        if (!mood) return;
        // Optimistic UI: marca seleccionado inmediatamente
        widget.querySelectorAll(".mood-btn").forEach((b) => b.classList.remove("selected"));
        btn.classList.add("selected");
        try {
          const r = await fetch("/api/mood", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mood }),
          });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          // Refresh el indicador "✓" sin re-fetch entero
          const cur = widget.querySelector(".mood-current");
          if (cur && !cur.querySelector(".mood-saved")) {
            cur.insertAdjacentHTML("afterbegin", `<span class="mood-saved">✓</span> `);
          }
        } catch (err) {
          console.error("mood post failed", err);
          btn.classList.remove("selected");
        }
      });
    });

    // Count chip: muestra duración total como signal rápido al colapsar
    const countEl = panel.querySelector("[data-count]");
    if (countEl) countEl.textContent = totalLabel;

    // Footer: source attribution (1×/día) + insight si aplica
    const foot = panel.querySelector("[data-foot]");
    if (foot) {
      const histN = sleep.hist?.n;
      foot.innerHTML = histN
        ? `<span class="row-meta">${histN} noches en hist · Pillow + Apple Watch</span>`
        : "";
    }
  }

  // ──────────────────────────────────────────────────────────────
  // Topbar status
  // ──────────────────────────────────────────────────────────────

  function updateTopbar(payload) {
    const dateEl = $("#today-date");
    const lastEl = $("#last-update");
    if (payload.date) dateEl.textContent = payload.date;
    if (payload.generated_at) {
      lastEl.textContent = `actualizado ${fmtTimeAgo(payload.generated_at)}`;
    }
    // serve dot — verde por default. Si el fetch falló, otro path lo cambia.
    $("#serve-dot").classList.remove("warn", "crit");
  }

  // ──────────────────────────────────────────────────────────────
  // Main fetch + render
  // ──────────────────────────────────────────────────────────────

  async function load(opts = {}) {
    try {
      const url = opts.regenerate ? "/api/home?regenerate=true" : "/api/home";
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const payload = await r.json();
      // Cache para que urlFor() y los renderers puedan acceder al
      // map vault_dir_names sin tener que pasar payload explícito.
      _currentPayload = payload;
      render(payload);
    } catch (err) {
      console.error("[home.v2] load failed:", err);
      $("#serve-dot").classList.add("crit");
      $("#serve-text").textContent = "serve · down";
    }
  }

  // Item B: Patrones del día — render del panel cross-source dedicado.
  // Lee `payload.today.correlations` (poblado por el correlator del web
  // cuando regenerate=true). Si NO hay people ni topics, oculta el panel
  // entero. Si hay, renderea cada item con su nombre + sources separados.
  function renderPatterns(payload) {
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

    // Fallback: si el correlator no encontró matches cross-source
    // explícitos hoy, derivar "top contactos del día" cruzando los
    // remitentes de gmail.recent + nombres de whatsapp_unreplied. Es
    // útil aunque no sea una correlation formal — al menos el panel
    // muestra a quiénes tenés que prestar atención hoy.
    if (total === 0) {
      const signals = payload.signals || {};
      const counts = new Map();   // name → {sources: Set, mentions: int}
      for (const m of signals.gmail?.recent || []) {
        if (isBotOrSelf(m.from)) continue;   // skip notifications/noreply/self
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
      // Mostrar los top 5 — sin exigir cross-source, ya el correlator
      // del LLM lo intentó arriba. Acá es "con quién tuviste contacto".
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

  function render(payload) {
    updateTopbar(payload);
    renderTodayHero(payload);
    renderCmdBar(payload);
    renderPatterns(payload);
    renderInbox(payload);
    renderQuestions(payload);
    renderTomorrow(payload);
    renderWAUnreplied(payload);
    renderLoopsUrgent(payload);
    renderContradictions(payload);
    renderFinance(payload);
    renderCards(payload);
    renderRetrievalHealth(payload);
    renderLoopsAging(payload);
    renderAuthority(payload);
    renderEvalTrend(payload);
    renderWeather(payload);
    renderVaultActivity(payload);
    renderCaptured(payload);
    renderWeb(payload);
    renderBookmarks(payload);
    renderYouTube(payload);
    renderDrive(payload);
    renderSpotify(payload);
    renderSleep(payload);
    renderMood(payload);
    renderPatterns(payload);
    // renderCorrelations: panel `p-correlations` (Pearson cross-source
    // entre métricas diarias). Distinto del panel `p-patterns`
    // (entidades cross-source) que rendea renderPatterns.
    renderCorrelations(payload);
  }

  // Auto-refresh cada 5 min
  document.addEventListener("DOMContentLoaded", () => {
    load();
    setInterval(load, 5 * 60 * 1000);
  });

  // ──────────────────────────────────────────────────────────────
  // Drag-and-drop: reorder de paneles (cross-section permitido)
  // ──────────────────────────────────────────────────────────────
  //
  // El user puede agarrar cualquier .panel y soltarlo en otra posición,
  // incluso fuera de su sección original (ej. mover YouTube de
  // "Ambiente" al top de "Accionable"). El orden se persiste en
  // localStorage[LS_PANELS_ORDER] como un mapa
  //   { <section-body-id>: [panel-ids...] }
  // y se re-aplica en el próximo boot — antes de que los renderers
  // llenen los paneles — para evitar flicker.
  //
  // HTML5 DnD API (desktop). En touch el browser no dispara los eventos
  // de DnD nativo de manera consistente, así que en mobile el feature
  // queda como no-op. El user accede a la feature desde desktop.

  const LS_PANELS_ORDER = "home.v2.panels.order.v1";
  const LS_PANELS_COLLAPSED = "home.v2.panels.collapsed.v1";
  const SECTION_BODY_IDS = ["sec-acc-body", "sec-mon-body", "sec-amb-body"];

  function readSavedOrder() {
    try {
      const raw = localStorage.getItem(LS_PANELS_ORDER);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return null;
      return parsed;
    } catch { return null; }
  }

  function saveCurrentOrder() {
    const order = {};
    for (const secId of SECTION_BODY_IDS) {
      const sec = document.getElementById(secId);
      if (!sec) continue;
      order[secId] = Array.from(sec.querySelectorAll(":scope > .panel"))
        .map((p) => p.id)
        .filter(Boolean);
    }
    try {
      localStorage.setItem(LS_PANELS_ORDER, JSON.stringify(order));
    } catch (e) {
      console.warn("[home.v2] no pude persistir el orden de paneles:", e);
    }
    updateResetButtonVisibility();
  }

  function applySavedOrder() {
    const saved = readSavedOrder();
    if (!saved) return;
    // Para cada sección, tomamos la lista de IDs guardados y los
    // re-ordenamos via appendChild (mueve el nodo si ya existe).
    // Paneles NO listados (ej. agregados en un deploy futuro) quedan
    // en su posición default — no los tocamos.
    for (const secId of SECTION_BODY_IDS) {
      const sec = document.getElementById(secId);
      if (!sec) continue;
      const ids = Array.isArray(saved[secId]) ? saved[secId] : [];
      for (const pid of ids) {
        const panel = document.getElementById(pid);
        if (!panel) continue;            // panel ya no existe (deprecated)
        sec.appendChild(panel);          // mueve si ya está, agrega si vino de otra sección
      }
    }
  }

  function clearSavedOrder() {
    try { localStorage.removeItem(LS_PANELS_ORDER); } catch {}
    // El bot\u00f3n "\u21ba orden" tambi\u00e9n limpia el estado de collapse \u2014 si el user
    // dijo "resetear", quiere TODO el layout custom borrado: orden + paneles
    // colapsados + hero colapsado + sections colapsadas.
    try { localStorage.removeItem(LS_PANELS_COLLAPSED); } catch {}
    try { localStorage.removeItem(LS_HERO_COLLAPSED); } catch {}
    try { localStorage.removeItem(LS_SECTIONS_COLLAPSED); } catch {}
    // Restore default DOM order — más simple recargar que reconstruir
    // el orden hard-coded del HTML.
    window.location.reload();
  }

  // ── Collapse por panel: cada .panel tiene un botón ▼/▶ en su header
  // que toggle `data-collapsed`. CSS oculta panel-body + panel-foot
  // cuando data-collapsed="true". Estado persistido en localStorage
  // como mapa { <panel-id>: true } — sólo guardamos los collapsed.

  function readCollapsedMap() {
    try {
      const raw = localStorage.getItem(LS_PANELS_COLLAPSED);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return (parsed && typeof parsed === "object") ? parsed : {};
    } catch { return {}; }
  }

  function saveCollapsedMap(map) {
    try {
      // Limpiar entries falsy para que el JSON quede chico
      const trimmed = {};
      for (const [k, v] of Object.entries(map)) if (v) trimmed[k] = true;
      if (Object.keys(trimmed).length === 0) {
        localStorage.removeItem(LS_PANELS_COLLAPSED);
      } else {
        localStorage.setItem(LS_PANELS_COLLAPSED, JSON.stringify(trimmed));
      }
    } catch (e) {
      console.warn("[home.v2] no pude persistir collapse de paneles:", e);
    }
  }

  function applySavedCollapse() {
    const map = readCollapsedMap();
    for (const pid of Object.keys(map)) {
      const panel = document.getElementById(pid);
      if (!panel) continue;
      panel.setAttribute("data-collapsed", "true");
      const btn = panel.querySelector(".panel-collapse-btn");
      if (btn) {
        btn.setAttribute("aria-expanded", "false");
        const icon = btn.querySelector(".toggle-icon");
        if (icon) icon.textContent = "\u25B6";
      }
    }
  }

  function togglePanelCollapse(panel) {
    const collapsed = panel.getAttribute("data-collapsed") === "true";
    const next = !collapsed;
    panel.setAttribute("data-collapsed", next ? "true" : "false");
    const btn = panel.querySelector(".panel-collapse-btn");
    if (btn) {
      btn.setAttribute("aria-expanded", next ? "false" : "true");
      const icon = btn.querySelector(".toggle-icon");
      if (icon) icon.textContent = next ? "\u25B6" : "\u25BC";
    }
    const map = readCollapsedMap();
    if (next) map[panel.id] = true;
    else delete map[panel.id];
    saveCollapsedMap(map);
    updateResetButtonVisibility();
  }

  function injectCollapseButton(panel) {
    const head = panel.querySelector(".panel-head");
    if (!head || head.querySelector(".panel-collapse-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "panel-collapse-btn";
    btn.setAttribute("aria-label", "Colapsar/expandir panel");
    btn.setAttribute("aria-expanded", "true");
    btn.title = "Colapsar/expandir";
    btn.innerHTML = '<span class="toggle-icon" aria-hidden="true">\u25BC</span>';
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      togglePanelCollapse(panel);
    });
    // Evitar que el click en el bot\u00f3n inicie un drag (los eventos de DnD
    // se disparan en el panel padre con draggable=true, pero el click
    // del bot\u00f3n no debe ser interpretado como dragstart).
    btn.addEventListener("mousedown", (ev) => ev.stopPropagation());
    // Lo agregamos como \u00faltimo hijo del head \u2014 a la derecha del .count
    head.appendChild(btn);
  }

  let _draggingPanel = null;

  function onPanelDragStart(ev) {
    const panel = ev.currentTarget;
    _draggingPanel = panel;
    panel.classList.add("is-dragging");
    try {
      ev.dataTransfer.effectAllowed = "move";
      // Firefox necesita data seteada para que dispare drop
      ev.dataTransfer.setData("text/plain", panel.id);
    } catch {}
  }

  function onPanelDragEnd(ev) {
    const panel = ev.currentTarget;
    panel.classList.remove("is-dragging");
    _draggingPanel = null;
    document.querySelectorAll(".panel.drop-before, .panel.drop-after")
      .forEach((p) => p.classList.remove("drop-before", "drop-after"));
    document.querySelectorAll(".section-body.drop-zone")
      .forEach((s) => s.classList.remove("drop-zone"));
  }

  function onPanelDragOver(ev) {
    if (!_draggingPanel) return;
    const panel = ev.currentTarget;
    if (panel === _draggingPanel) return;
    ev.preventDefault();
    try { ev.dataTransfer.dropEffect = "move"; } catch {}
    // Decidir before/after según posición del cursor relativa al panel.
    // Si el panel es claramente más ancho que alto (desktop multi-col),
    // usamos X; en single-col (mobile/tablet angosto) usamos Y.
    const rect = panel.getBoundingClientRect();
    const useX = rect.width > rect.height * 1.2;
    const before = useX
      ? (ev.clientX - rect.left) < rect.width / 2
      : (ev.clientY - rect.top) < rect.height / 2;
    panel.classList.toggle("drop-before", before);
    panel.classList.toggle("drop-after", !before);
  }

  function onPanelDragLeave(ev) {
    const panel = ev.currentTarget;
    // Sólo limpiar si salimos del panel (no si entramos a un hijo)
    if (panel.contains(ev.relatedTarget)) return;
    panel.classList.remove("drop-before", "drop-after");
  }

  function onPanelDrop(ev) {
    ev.preventDefault();
    if (!_draggingPanel) return;
    const target = ev.currentTarget;
    if (target === _draggingPanel) return;
    const before = target.classList.contains("drop-before");
    target.classList.remove("drop-before", "drop-after");
    if (before) {
      target.parentNode.insertBefore(_draggingPanel, target);
    } else {
      target.parentNode.insertBefore(_draggingPanel, target.nextSibling);
    }
    saveCurrentOrder();
  }

  function makePanelDraggable(panel) {
    if (panel.dataset.draggableInit === "1") return;
    panel.dataset.draggableInit = "1";
    panel.setAttribute("draggable", "true");
    // Insertar el grip "⋮⋮" en el .panel-head como affordance visual
    const head = panel.querySelector(".panel-head");
    if (head && !head.querySelector(".drag-grip")) {
      const grip = document.createElement("span");
      grip.className = "drag-grip";
      grip.setAttribute("aria-hidden", "true");
      grip.title = "arrastrá para reordenar";
      grip.textContent = "⋮⋮";
      head.insertBefore(grip, head.firstChild);
    }
    panel.addEventListener("dragstart", onPanelDragStart);
    panel.addEventListener("dragend", onPanelDragEnd);
    panel.addEventListener("dragover", onPanelDragOver);
    panel.addEventListener("dragleave", onPanelDragLeave);
    panel.addEventListener("drop", onPanelDrop);
  }

  // Drop directo sobre la section-body (cuando arrastrás a un espacio
  // libre fuera de cualquier panel — se appendea al final). Esto te
  // permite mover un panel a una sección vacía o al fondo de una
  // sección sin tener que apuntar exacto sobre otro panel.
  function makeSectionDroppable(secId) {
    const sec = document.getElementById(secId);
    if (!sec) return;
    sec.addEventListener("dragover", (ev) => {
      if (!_draggingPanel) return;
      if (ev.target !== sec) return;       // sólo si el cursor está sobre el grid container, no sobre un panel hijo
      ev.preventDefault();
      try { ev.dataTransfer.dropEffect = "move"; } catch {}
      sec.classList.add("drop-zone");
    });
    sec.addEventListener("dragleave", (ev) => {
      if (ev.target !== sec) return;
      sec.classList.remove("drop-zone");
    });
    sec.addEventListener("drop", (ev) => {
      if (!_draggingPanel) return;
      if (ev.target !== sec) return;
      ev.preventDefault();
      sec.classList.remove("drop-zone");
      sec.appendChild(_draggingPanel);
      saveCurrentOrder();
    });
  }

  function hasCustomLayout() {
    if (readSavedOrder()) return true;
    const map = readCollapsedMap();
    if (map && Object.keys(map).length > 0) return true;
    // Hero collapsed o sections con estado guardado también cuentan
    // como "layout custom" — el botón "↺ layout" debe aparecer.
    try {
      if (localStorage.getItem(LS_HERO_COLLAPSED) === "1") return true;
      if (localStorage.getItem(LS_SECTIONS_COLLAPSED)) return true;
    } catch {}
    return false;
  }

  function updateResetButtonVisibility() {
    const btn = document.getElementById("reset-order-btn");
    if (!btn) return;
    btn.hidden = !hasCustomLayout();
  }

  function injectResetButton() {
    const meta = document.getElementById("topbar-meta");
    if (!meta || document.getElementById("reset-order-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "reset-order-btn";
    btn.className = "reset-order-btn";
    btn.title = "Resetear layout (orden + paneles colapsados)";
    btn.setAttribute("aria-label", "Resetear layout de los paneles");
    btn.textContent = "↺ layout";
    btn.hidden = !hasCustomLayout();
    btn.addEventListener("click", () => {
      if (confirm("¿Resetear el layout de los paneles al default? (orden + collapse)")) {
        clearSavedOrder();
      }
    });
    meta.appendChild(btn);
  }

  document.addEventListener("DOMContentLoaded", () => {
    // 1. Aplicar orden persistido ANTES de que los renderers escriban.
    //    (Los renderers usan getElementById por panel-id, así que mover
    //     el nodo en el DOM no rompe el data-binding.)
    applySavedOrder();
    // 2. Hacer cada panel draggable + insertar el grip + botón collapse
    document.querySelectorAll(".section-body > .panel").forEach((panel) => {
      makePanelDraggable(panel);
      injectCollapseButton(panel);
    });
    // 3. Aplicar estado de collapse persistido (ya con los botones inyectados)
    applySavedCollapse();
    // 4. Hacer cada section-body un drop zone para "soltar al final"
    SECTION_BODY_IDS.forEach(makeSectionDroppable);
    // 5. Inyectar botón reset en la topbar
    injectResetButton();
  });
})();
