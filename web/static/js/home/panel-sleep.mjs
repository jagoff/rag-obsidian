// panel-sleep.mjs — panel de sleep (Pillow + mood self-report).

import { escapeHTML, fmtTimeAgo } from "./core.mjs";
import { renderSparkline } from "./charts.mjs";

// Map de mood label → emoji + texto corto.
export const MOOD_OPTIONS = [
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

export function renderSleep(payload) {
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

  // Stages con warn thresholds
  const deepPct = ln.deep_pct;
  const remPct = ln.rem_pct;
  const awak = ln.awakenings ?? 0;
  const deepWarn = deepPct != null && deepPct < 15;
  const remWarn = remPct != null && remPct < 15;
  const awakWarnCls = awak >= 5 ? "stale" : awak >= 3 ? "warn" : "";

  // Delta vs hist
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

  // Wake-up mood de Pillow
  const wakeupMood = ln.wakeup_mood;
  const wakeupLabel = wakeupMood != null
    ? `${WAKEUP_MOOD_LABELS[wakeupMood] || "—"}`
    : null;

  const moodNowKey = moodNow?.label;
  const moodBtns = MOOD_OPTIONS.map((m) => {
    const cls = m.key === moodNowKey ? "mood-btn selected" : "mood-btn";
    return `<button type="button" class="${cls}" data-mood="${m.key}"
      title="${m.label}" aria-label="estado: ${m.label}">${m.emoji}</button>`;
  }).join("");

  const moodTimeAgo = moodNow?.ts
    ? fmtTimeAgo(new Date(moodNow.ts * 1000).toISOString())
    : null;

  const sparkVals = sleep.spark_quality_7d || [];
  const sparkSVG = renderSparkline(sparkVals, { width: 120, height: 22, ymin: 0, ymax: 1 });

  const insightHTML = sleep.insight
    ? `<div class="sleep-insight" role="status">⚠ ${escapeHTML(sleep.insight)}</div>`
    : "";

  // Patrones sleep (Pearson r)
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
      widget.querySelectorAll(".mood-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      try {
        const r = await fetch("/api/mood", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mood }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
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

  const countEl = panel.querySelector("[data-count]");
  if (countEl) countEl.textContent = totalLabel;

  const foot = panel.querySelector("[data-foot]");
  if (foot) {
    const histN = sleep.hist?.n;
    foot.innerHTML = histN
      ? `<span class="row-meta">${histN} noches en hist · Pillow + Apple Watch</span>`
      : "";
  }
}
