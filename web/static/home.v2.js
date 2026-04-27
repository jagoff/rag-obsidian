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
      const title = r.href
        ? `<a href="${escapeHTML(r.href)}">${escapeHTML(r.title)}</a>`
        : escapeHTML(r.title);
      const aside = r.aside ? `<span class="row-aside">${escapeHTML(r.aside)}</span>` : "";
      const meta = r.meta && r.meta.length
        ? `<div class="row-meta">${r.meta.map((m) =>
            typeof m === "string"
              ? escapeHTML(m)
              : `<span class="${m.cls || ""}">${escapeHTML(m.text)}</span>`,
          ).join(" · ")}</div>`
        : "";
      return `<div class="row">
        <div class="row-main">
          <div class="row-title">${title}</div>
          ${meta}
        </div>
        ${aside}
      </div>`;
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
    const items = payload.today?.evidence?.inbox_today || [];
    const rows = items.slice(0, 8).map((it) => ({
      title: it.title || it.path,
      meta: [
        ...(it.tags || []).map((t) => `#${t}`),
        fmtTimeAgo(it.modified),
      ],
    }));
    renderPanelList("p-inbox", rows, {
      emptyText: "todo el inbox procesado ✓",
      capChip: rows.length > 5 ? "warning" : "info",
      footText: rows.length ? `top 8 de ${items.length}` : "",
    });
  }

  function renderQuestions(payload) {
    const items = payload.signals?.low_conf || [];
    const rows = items.slice(0, 6).map((it) => ({
      title: it.q || it.question || it.text || "",
      meta: [
        it.score != null ? `score ${it.score.toFixed?.(2) || it.score}` : null,
        it.ts ? fmtTimeAgo(it.ts) : null,
      ].filter(Boolean),
    }));
    renderPanelList("p-questions", rows, {
      emptyText: "no hay preguntas sin respuesta",
      capChip: "warning",
    });
  }

  function renderTomorrow(payload) {
    const cal = payload.tomorrow_calendar || {};
    const items = cal.events || [];
    const rows = items.slice(0, 8).map((e) => ({
      title: e.title || e.summary || "(sin título)",
      meta: [e.start || e.time, e.location].filter(Boolean),
    }));
    renderPanelList("p-tomorrow", rows, {
      emptyText: cal.message || "agenda libre mañana",
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
    }));
    renderPanelList("p-loops-urgent", rows, {
      emptyText: "ningún loop STALE",
      capChip: rows.length > 0 ? "warning" : "info",
    });
  }

  function renderContradictions(payload) {
    const items = payload.signals?.contradictions || [];
    const rows = items.slice(0, 6).map((it) => {
      const subj = (it.subject_path || "").split("/").pop().replace(/\.md$/, "");
      const tgt = (it.targets && it.targets[0]) || {};
      return {
        title: subj,
        meta: [
          tgt.note ? `vs ${tgt.note}` : null,
          tgt.why ? tgt.why.slice(0, 60) : null,
          it.ts ? fmtTimeAgo(it.ts) : null,
        ].filter(Boolean),
      };
    });
    renderPanelList("p-contradictions", rows, {
      emptyText: "sin contradicciones detectadas",
      capChip: rows.length > 0 ? "warning" : "info",
    });
  }

  // ──────────────────────────────────────────────────────────────
  // Today hero — el bloque grande arriba con 4 sub-secciones
  // (Lo que pasó hoy / Sin procesar / Preguntas abiertas / Para mañana).
  // El narrative del LLM es markdown con 4 H2; lo splitemos en 4 trozos
  // y renderemos cada uno en su columna.
  // ──────────────────────────────────────────────────────────────

  // Strip [[wikilinks]] del texto para que marked no los rompa (no son
  // markdown válido). Replico el helper que tiene v1.
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
    const stripped = stripWikilinks(md);
    if (window.marked) {
      try {
        return sanitizeHTML(window.marked.parse(stripped));
      } catch (e) {
        console.warn("[home.v2] marked.parse failed:", e);
      }
    }
    // Fallback regex-based — peor pero suficiente.
    return `<pre style="white-space:pre-wrap;font-family:inherit;margin:0;">${escapeHTML(stripped)}</pre>`;
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

  function renderTodayHero(payload) {
    const dateEl = document.getElementById("hero-date");
    const countsEl = document.getElementById("hero-counts");
    const bodyEl = document.getElementById("today-hero-body");
    if (!dateEl || !bodyEl) return;

    dateEl.textContent = payload.date || "—";

    const counts = payload.today?.counts || {};
    const evidence = payload.today?.evidence || {};
    const inboxItems = evidence.inbox_today || [];
    const lowConfItems = payload.signals?.low_conf || [];
    const tomorrowEvents = payload.tomorrow_calendar?.events || [];

    // Counts chip al lado de la fecha
    const totalThings = (counts.total) ||
      ((evidence.recent_notes?.length || 0) +
       (inboxItems.length) +
       (lowConfItems.length));
    countsEl.textContent = `${totalThings} señales · ${inboxItems.length} inbox · ${lowConfItems.length} preguntas`;

    const md = payload.today?.narrative || "";
    const split = splitNarrative(md);

    // Si no hay narrative (LLM aún no corrió), generamos contenido a
    // partir de evidence directamente — el user no ve "cargando" eterno.
    const hasNarrative = !!md.trim();

    const sectionHTML = (cls, emoji, label, count, contentHTML, emptyText) => `
      <div class="hero-section ${cls}">
        <h3><span>${emoji} ${escapeHTML(label)}</span>${count != null ? `<span class="count">${count}</span>` : ""}</h3>
        <div class="prose">${contentHTML || `<div class="empty">${escapeHTML(emptyText)}</div>`}</div>
      </div>
    `;

    // Sub-section: narrative del LLM ("Lo que pasó hoy")
    const narrativeHTML = split.narrative
      ? mdToHTML(split.narrative)
      : (hasNarrative ? "" : (() => {
          // Fallback derivado de evidence: notas tocadas hoy
          const recent = (evidence.recent_notes || []).slice(0, 5);
          if (!recent.length) return "";
          return "<p>Notas tocadas hoy:</p><ul>" + recent.map((n) =>
            `<li><strong>${escapeHTML(n.title || n.path)}</strong></li>`,
          ).join("") + "</ul>";
        })());

    // Sub-section: Sin procesar (inbox del día)
    let inboxHTML = "";
    if (split.inbox) {
      inboxHTML = mdToHTML(split.inbox);
    } else if (inboxItems.length) {
      inboxHTML = "<ul>" + inboxItems.slice(0, 8).map((it) => {
        const tags = (it.tags || []).slice(0, 3).map((t) => `<code>#${escapeHTML(t)}</code>`).join(" ");
        return `<li><strong>${escapeHTML(it.title || it.path)}</strong>${tags ? ` ${tags}` : ""}</li>`;
      }).join("") + "</ul>";
    }

    // Sub-section: Preguntas abiertas (low-conf queries)
    let questionsHTML = "";
    if (split.questions) {
      questionsHTML = mdToHTML(split.questions);
    } else if (lowConfItems.length) {
      questionsHTML = "<ul>" + lowConfItems.slice(0, 5).map((q) => {
        const text = q.q || q.question || q.text || "";
        const score = q.score != null ? ` <em>(score ${q.score.toFixed?.(2) || q.score})</em>` : "";
        return `<li>${escapeHTML(text)}${score}</li>`;
      }).join("") + "</ul>";
    }

    // Sub-section: Para mañana
    let tomorrowHTML = "";
    if (split.tomorrow) {
      tomorrowHTML = mdToHTML(split.tomorrow);
    } else if (tomorrowEvents.length) {
      tomorrowHTML = "<ul>" + tomorrowEvents.slice(0, 6).map((e) => {
        const when = e.start || e.time || "";
        return `<li><strong>${escapeHTML(when)}</strong> ${escapeHTML(e.title || e.summary || "")}</li>`;
      }).join("") + "</ul>";
    }

    bodyEl.innerHTML = [
      sectionHTML("s-narrative", "🪞", "Lo que pasó hoy", null, narrativeHTML, "Aún sin brief — pulsá ↻ arriba para generar"),
      sectionHTML("s-inbox", "📥", "Sin procesar", inboxItems.length || null, inboxHTML, "todo procesado ✓"),
      sectionHTML("s-questions", "🔍", "Preguntas abiertas", lowConfItems.length || null, questionsHTML, "sin preguntas pendientes"),
      sectionHTML("s-tomorrow", "🌅", "Para mañana", null, tomorrowHTML, "agenda libre · día abierto"),
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

  // Refresh manual del brief (regenerate=true → LLM corre síncrono ~30-45s
  // con qwen2.5:7b warm). Durante el wait mostramos un progress bar ASCII
  // que avanza en steps fijos cada 3s. NO es real — el endpoint no expone
  // SSE de progreso; es una indicación visual de "todavía vivo".
  document.addEventListener("DOMContentLoaded", () => {
    const refreshBtn = document.getElementById("brief-refresh");
    if (!refreshBtn) return;
    const progressEl = document.getElementById("hero-progress");
    const progressBar = document.getElementById("progress-bar");
    const progressLabel = document.getElementById("progress-label");
    let progressTimer = null;

    function startProgress() {
      if (!progressEl || !progressBar) return;
      progressEl.hidden = false;
      const totalBlocks = 20;
      let pct = 0;
      const labels = [
        "leyendo señales del vault…",
        "consultando gmail / wa / calendar…",
        "armando entidades cross-source…",
        "esperando al LLM…",
        "qwen2.5:7b escribiendo…",
        "post-procesando voz…",
      ];
      let labelIdx = 0;
      progressBar.textContent =
        `[${"░".repeat(totalBlocks)}] 0%`;
      progressLabel.textContent = labels[0];
      progressTimer = setInterval(() => {
        pct = Math.min(95, pct + 3);
        const filled = Math.round((pct / 100) * totalBlocks);
        const bar = "█".repeat(filled) + "░".repeat(totalBlocks - filled);
        progressBar.textContent = `[${bar}] ${pct}%`;
        if (pct % 15 === 0) {
          labelIdx = Math.min(labels.length - 1, labelIdx + 1);
          progressLabel.textContent = labels[labelIdx];
        }
      }, 1500);
    }

    function stopProgress() {
      if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
      }
      if (progressEl) progressEl.hidden = true;
    }

    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      refreshBtn.textContent = "↻";
      startProgress();
      try {
        await load({ regenerate: true });
      } finally {
        refreshBtn.disabled = false;
        stopProgress();
      }
    });
  });

  // Hero collapse toggle (Item A): click on the ▼/▶ button toggles
  // `data-collapsed` on the .today-hero and updates aria-expanded.
  document.addEventListener("DOMContentLoaded", () => {
    const heroToggle = document.getElementById("hero-toggle");
    if (!heroToggle) return;
    const hero = heroToggle.closest(".today-hero");
    if (!hero) return;
    heroToggle.addEventListener("click", () => {
      const collapsed = hero.getAttribute("data-collapsed") === "true";
      hero.setAttribute("data-collapsed", collapsed ? "false" : "true");
      heroToggle.setAttribute("aria-expanded", collapsed ? "true" : "false");
    });
  });

  // Section collapse toggles (Item F): each .section-toggle button
  // toggles its parent .section's `data-collapsed`. On mobile we
  // start with Monitoring + Ambiente collapsed by default to reduce
  // initial scroll; user expands them with tap. Hero + Accionable
  // stay expanded.
  function initCollapsibleSections() {
    const isMobile = window.matchMedia("(max-width: 720px)").matches;
    document.querySelectorAll(".section-toggle").forEach((btn) => {
      const section = btn.closest(".section");
      if (!section) return;
      // Default-collapse Monitoring + Ambiente on mobile boot
      if (isMobile && (
        section.classList.contains("section-monitoring") ||
        section.classList.contains("section-ambient")
      )) {
        section.setAttribute("data-collapsed", "true");
        btn.setAttribute("aria-expanded", "false");
      }
      btn.addEventListener("click", (e) => {
        // Permitir click en el toggle button sin scroll
        e.preventDefault();
        const collapsed = section.getAttribute("data-collapsed") === "true";
        section.setAttribute("data-collapsed", collapsed ? "false" : "true");
        btn.setAttribute("aria-expanded", collapsed ? "true" : "false");
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
    const rows = cards.slice(0, 4).map((c) => ({
      title: `${c.brand || ""} ····${c.last4 || "????"}`,
      meta: [
        c.due_date ? `due ${c.due_date}` : null,
        c.next_closing_date ? `next close ${c.next_closing_date}` : null,
      ].filter(Boolean),
      aside: c.balance_ars ? fmtCurrencyARS(c.balance_ars) : null,
    }));
    renderPanelList("p-cards", rows, {});
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
    if (!f || !f.buckets) {
      body.innerHTML = `<div class="empty">sin datos de loops</div>`;
      count.textContent = "—";
      return;
    }
    const buckets = f.buckets || {};
    const fresh = buckets["0-7d"] || buckets.fresh || 0;
    const aging = buckets["8-30d"] || buckets.aging || 0;
    const stale = buckets["stale"] || buckets.STALE || 0;
    body.innerHTML = `
      <div class="panel-kpi">
        <span class="value">${f.total || (fresh + aging + stale)}</span>
        ${stale > 0 ? `<span class="delta up">${stale} STALE</span>` : `<span class="delta down">tranquilo</span>`}
      </div>
      ${stackedBar({ fresh, aging, stale })}
      ${(f.sample || []).slice(0, 3).length ? `
        <div class="row-meta" style="margin-top: 8px; flex-direction: column; align-items: stretch;">
          ${(f.sample || []).slice(0, 3).map((s) =>
            `<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">· ${escapeHTML((s.text || s.loop_text || "").slice(0, 60))}</div>`
          ).join("")}
        </div>
      ` : ""}
    `;
    count.textContent = f.total || (fresh + aging + stale);
  }

  function renderAuthority(payload) {
    const items = payload.signals?.pagerank_top || [];
    const rows = items.slice(0, 5).map((it, i) => ({
      title: `#${i + 1}  ${it.title || ""}`,
      meta: [it.path ? it.path.split("/").slice(0, -1).join("/") : null].filter(Boolean),
      aside: it.pr ? `pr ${it.pr.toFixed?.(2) || it.pr}` : null,
    }));
    renderPanelList("p-authority", rows, {
      emptyText: "sin datos",
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
    const wf = payload.weather_forecast;
    const today = payload.signals?.weather;
    const panel = document.getElementById("p-weather");
    if (!panel) return;
    const body = panel.querySelector("[data-body]");
    const count = panel.querySelector("[data-count]");

    let items = [];
    if (Array.isArray(wf)) items = wf;
    else if (wf && typeof wf === "object") {
      items = wf.forecast || wf.days || Object.values(wf).filter((v) => v && typeof v === "object");
    }
    if (today && (!items.length || !items.some((d) => d.date === today.date))) {
      items = [today, ...items];
    }
    if (!items.length) {
      body.innerHTML = `<div class="empty">sin datos</div>`;
      count.textContent = "—";
      return;
    }
    const rows = items.slice(0, 4).map((d) => ({
      title: d.date || d.day || "",
      meta: [
        d.summary || d.condition || "",
        d.temp_min != null && d.temp_max != null
          ? `${d.temp_min}°–${d.temp_max}°`
          : (d.temp != null ? `${d.temp}°` : null),
      ].filter(Boolean),
    }));
    renderPanelList("p-weather", rows, {});
  }

  function renderVaultActivity(payload) {
    const act = payload.signals?.vault_activity || {};
    const home = act.home || [];
    const rows = home.slice(0, 6).map((it) => ({
      title: it.title || it.path,
      meta: [
        it.path ? it.path.split("/").slice(0, -1).join("/") : null,
        it.modified ? fmtTimeAgo(it.modified) : null,
      ].filter(Boolean),
    }));
    renderPanelList("p-vault-activity", rows, {
      emptyText: "sin actividad",
    });
  }

  function renderCaptured(payload) {
    const items = payload.today?.evidence?.inbox_today || [];
    // captured = inbox items modificados hoy
    const today = new Date().toISOString().slice(0, 10);
    const todayItems = items.filter((it) => (it.modified || "").startsWith(today));
    const rows = todayItems.slice(0, 6).map((it) => ({
      title: it.title || it.path,
      meta: [
        ...(it.tags || []).slice(0, 3).map((t) => `#${t}`),
        fmtTimeAgo(it.modified),
      ],
    }));
    renderPanelList("p-captured", rows, {
      emptyText: "nada capturado hoy",
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
      href: it.url,
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
      href: it.link,
    }));
    renderPanelList("p-drive", rows, { emptyText: "sin actividad reciente" });
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
      || null;
    if (!correlations) {
      panel.hidden = true;
      return;
    }
    const people = correlations.people || [];
    const topics = correlations.topics || [];
    const overlaps = correlations.time_overlaps || [];
    const total = people.length + topics.length + overlaps.length;
    if (total === 0) {
      panel.hidden = true;
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
  }

  // Auto-refresh cada 5 min
  document.addEventListener("DOMContentLoaded", () => {
    load();
    setInterval(load, 5 * 60 * 1000);
  });
})();
