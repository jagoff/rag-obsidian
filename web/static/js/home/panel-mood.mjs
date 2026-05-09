// panel-mood.mjs — panel de mood score + modal historial 30d + correlations (Pearson).

import { escapeHTML } from "./core.mjs";
import { moodScoreClass, renderMoodSparkline } from "./charts.mjs";

// ── Constantes de mood ─────────────────────────────────────────────────────────

export const MOOD_SELF_REPORT_OPTIONS = [
  { key: "good", emoji: "😀", label: "bien" },
  { key: "meh", emoji: "😐", label: "normal" },
  { key: "bad", emoji: "😞", label: "mal" },
  { key: "sad", emoji: "😔", label: "triste" },
];

export const MOOD_SOURCE_EMOJI = {
  spotify: "🎧",
  journal: "📓",
  wa_outbound: "💬",
  queries: "🔍",
  calendar: "📅",
  pillow: "🌙",
  manual: "✋",
};

export const MOOD_KIND_LABEL = {
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

export function moodLabel(score) {
  if (score == null) return "—";
  if (score <= -0.6) return "muy bajo";
  if (score <= -0.3) return "bajo";
  if (score <= -0.1) return "tibio";
  if (score < 0.1)   return "neutro";
  if (score < 0.3)   return "estable";
  if (score < 0.6)   return "arriba";
  return "alto";
}

export function moodTrendBadge(trend) {
  if (trend === "improving") return `<span class="mood-trend up">↑ subiendo</span>`;
  if (trend === "declining") return `<span class="mood-trend down">↓ bajando</span>`;
  return `<span class="mood-trend stable">· estable</span>`;
}

// ── Render del panel de mood ───────────────────────────────────────────────────

export function renderMood(payload) {
  const m = payload.signals?.mood;
  const panel = document.getElementById("p-mood");
  if (!panel) return;
  if (!m || m.score == null) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const scoreClass = moodScoreClass(m.score);
  const sign = m.score > 0 ? "+" : "";
  const scoreText = `${sign}${m.score.toFixed(2)}`;
  const labelText = moodLabel(m.score);

  const sparkVals = m.spark_score_14d || [];
  const sparkDates = m.spark_dates_14d || [];
  const sparkSVG = renderMoodSparkline(sparkVals, sparkDates);

  const sources = m.sources_used || [];
  const sourcesHTML = sources.length
    ? sources.map((s) => {
        const emoji = MOOD_SOURCE_EMOJI[s] || "·";
        return `<span class="mood-src">${emoji} ${escapeHTML(s)}</span>`;
      }).join("")
    : `<span class="mood-src empty">sin sources</span>`;

  const drift = m.drift || {};
  const driftHTML = drift.drifting
    ? `<div class="mood-drift" role="status">
        ⚠ ${drift.n_consecutive} días consecutivos abajo del baseline
        <span class="muted">(promedio ${drift.avg_score.toFixed(2)})</span>
      </div>`
    : "";

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
      <div class="mood-self-buttons" role="group" aria-label="reportar mood actual">
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

  // Wire self-report buttons → POST /api/mood
  const reportWidget = body.querySelector("[data-self-report]");
  reportWidget?.querySelectorAll(".mood-self-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const mood = btn.dataset.mood;
      if (!mood) return;
      const allBtns = reportWidget.querySelectorAll(".mood-self-btn");
      allBtns.forEach((b) => { b.classList.remove("selected"); b.disabled = true; });
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
        allBtns.forEach((b) => { b.disabled = false; });
      } catch (err) {
        console.error("mood self-report failed", err);
        allBtns.forEach((b) => { b.disabled = false; b.classList.remove("selected"); });
        if (prompt) prompt.textContent = "error — reintentá";
      }
    });
  });

  const countEl = panel.querySelector("[data-count]");
  if (countEl) countEl.textContent = m.n_signals;

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

  // Fade-in primera vez
  if (!panel.dataset.firstShown) {
    panel.dataset.firstShown = "1";
    panel.classList.add("mood-fade-in");
    setTimeout(() => panel.classList.remove("mood-fade-in"), 600);
  }
}

// ── Modal historial de mood ────────────────────────────────────────────────────

let _moodHistoryCache = null;
let _moodHistoryFetching = false;

export async function openMoodHistoryModal() {
  const dlg = document.getElementById("mood-history-modal");
  if (!dlg) return;
  const body = dlg.querySelector("[data-mood-modal-body]");
  if (!body) return;

  if (typeof dlg.showModal === "function") {
    dlg.showModal();
  } else {
    dlg.setAttribute("open", "");
  }

  const closeBtn = dlg.querySelector("[data-mood-modal-close]");
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

  if (_moodHistoryCache) {
    renderMoodHistory(_moodHistoryCache, body);
    return;
  }
  if (_moodHistoryFetching) return;
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

export function renderMoodHistory(data, body) {
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

  const histHTML = histogram.length
    ? `<section class="mood-modal-section" aria-labelledby="mood-hist-title">
        <h3 id="mood-hist-title">contribución por source · ${range}d</h3>
        <ul class="mood-histogram">${histogram.map((h) => {
          const sign = h.total_contrib > 0 ? "+" : "";
          const sigCls = h.total_contrib > 0 ? "pos" : h.total_contrib < 0 ? "neg" : "flat";
          const emoji = MOOD_SOURCE_EMOJI[h.source] || "·";
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

  const timelineHTML = `
    <section class="mood-modal-section" aria-labelledby="mood-timeline-title">
      <h3 id="mood-timeline-title">timeline ${range}d</h3>
      <ul class="mood-timeline">${days.map((d) => {
        const hasData = d.n_signals > 0;
        const score = d.score || 0;
        const sign = score > 0 ? "+" : "";
        const scoreCls = moodScoreClass(score);
        const barH = Math.abs(score) * 100;
        const barDir = score < 0 ? "neg" : "pos";
        const sourcesShort = (d.by_source || [])
          .slice(0, 3)
          .map((s) => `<span class="ts-src" title="${escapeHTML(s.source)}: ${s.contrib > 0 ? '+' : ''}${s.contrib.toFixed(2)} (${s.pct.toFixed(0)}%)">${MOOD_SOURCE_EMOJI[s.source] || "·"}</span>`)
          .join("");
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

// ── Panel de correlaciones Pearson cross-source ───────────────────────────────

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

export function patternsLabel(metricName) {
  return PATTERNS_METRIC_LABELS[metricName] || metricName;
}

export function patternsSeverityClass(severity) {
  if (severity === "strong") return "patterns-strong";
  if (severity === "moderate") return "patterns-moderate";
  return "patterns-weak";
}

export function patternsLagLabel(lag) {
  if (lag === 0) return "mismo día";
  if (lag === 1) return "+1 día";
  if (lag === 7) return "+1 semana";
  return `+${lag}d`;
}

let _patternsCache = null;
let _patternsFetching = false;

export async function fetchPatterns(force = false) {
  if (_patternsCache && !force) return _patternsCache;
  if (_patternsFetching) {
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

export async function renderCorrelations(_payload) {
  const panel = document.getElementById("p-correlations");
  if (!panel) return;
  panel.hidden = false;
  const body = panel.querySelector("[data-body]");

  const data = await fetchPatterns();
  if (!data) {
    body.innerHTML = `<div class="empty">no se pudo cargar los patrones</div>`;
    return;
  }

  const findings = data.top || [];
  const bySev = data.by_severity || {};
  const metricsWithData = data.metrics_with_data || [];
  const metricsWithEnoughData = metricsWithData.filter(([_, n]) => n >= 21).length;

  if (findings.length === 0) {
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

  if (!panel.dataset.firstShown) {
    panel.dataset.firstShown = "1";
    panel.classList.add("mood-fade-in");
    setTimeout(() => panel.classList.remove("mood-fade-in"), 600);
  }
}

// ── Modal de patrones completo ─────────────────────────────────────────────────

export function openPatternsModal() {
  const dlg = document.getElementById("patterns-modal");
  if (!dlg) return;
  const body = dlg.querySelector("[data-patterns-modal-body]");
  if (!body) return;

  if (typeof dlg.showModal === "function") {
    dlg.showModal();
  } else {
    dlg.setAttribute("open", "");
  }

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

export function renderPatternsModal(data, body) {
  const findings = data.top || [];
  const bySev = data.by_severity || {};
  const metrics = data.metrics_with_data || [];
  const lagsTested = data.lags_tested || [];
  const range = data.days_range || 30;

  const summaryHTML = `
    <p class="mood-modal-summary muted">
      ${data.n_findings} correlaciones · ${range} días · lags
      ${lagsTested.join(", ")} · ${metrics.length} métricas con data
    </p>
  `;

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
