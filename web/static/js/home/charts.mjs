// charts.mjs — ASCII mini-charts + sparkline SVG inline compartidos.
// Importado por múltiples paneles (finance, retrieval, sleep, mood, etc.).

import { escapeHTML } from "./core.mjs";

// ── Sparkline SVG básico ──────────────────────────────────────────────────────
// `values` = array de números. `tone` ∈ ok|warn|crit|info
export function sparkline(values, tone = "info") {
  if (!Array.isArray(values)) return "";
  // Sanitizar: eliminar nulls/NaN/no-números; necesitamos >=2 puntos válidos.
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

// ── Barra horizontal apilada (3 segmentos: fresh / aging / stale) ─────────────
export function stackedBar(parts) {
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

// ── Sparkline enriquecido (para sleep/mood) ────────────────────────────────────
// Soporte de gaps (nulls), dots por punto, configuración de dimensiones.
export function renderSparkline(values, opts = {}) {
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

  // Polyline path — new sub-path en cada gap null.
  const segs = [];
  let started = false;
  values.forEach((v, i) => {
    if (v == null) { started = false; return; }
    const cmd = started ? "L" : "M";
    segs.push(`${cmd}${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`);
    started = true;
  });
  const pathD = segs.join(" ");

  // Dots — el último non-null se resalta diferente.
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

// ── Sparkline de mood (con tooltips + zero baseline) ─────────────────────────
export function moodScoreClass(score) {
  if (score == null) return "mood-score-na";
  if (score <= -0.4) return "mood-score-low";
  if (score <= -0.1) return "mood-score-tepid";
  if (score < 0.2)   return "mood-score-neutral";
  if (score < 0.5)   return "mood-score-up";
  return "mood-score-high";
}

export function renderMoodSparkline(values, dates) {
  const validCount = (values || []).filter(v => v != null).length;
  if (validCount === 0) {
    return `<div class="mood-spark-placeholder">acumulando data…</div>`;
  }
  if (validCount < 3) {
    const lastIdx = values.length - 1 - [...values].reverse().findIndex(v => v != null);
    const last = values[lastIdx];
    const sign = last > 0 ? "+" : "";
    return `<div class="mood-spark-placeholder">
      acumulando data… (${validCount} día${validCount > 1 ? "s" : ""},
      último <span class="${moodScoreClass(last)}">${sign}${last.toFixed(2)}</span>)
    </div>`;
  }
  // ≥ 3 puntos: SVG inline con tooltips + zero-baseline.
  const W = 160;
  const H = 28;
  const padY = 2;
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
  const segs = [];
  let started = false;
  values.forEach((v, i) => {
    if (v == null) { started = false; return; }
    const cmd = started ? "L" : "M";
    segs.push(`${cmd}${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`);
    started = true;
  });
  const pathD = segs.join(" ");
  let lastIdx = -1;
  for (let i = N - 1; i >= 0; i--) {
    if (values[i] != null) { lastIdx = i; break; }
  }
  const dots = values.map((v, i) => {
    if (v == null) return "";
    const cls = i === lastIdx ? "spark-dot last" : "spark-dot";
    const sign = v > 0 ? "+" : "";
    const dateStr = (dates && dates[i]) || "";
    return `<circle class="${cls}" cx="${xFor(i).toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="1.5">
      <title>${escapeHTML(dateStr)}: ${sign}${v.toFixed(2)}</title>
    </circle>`;
  }).join("");
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
