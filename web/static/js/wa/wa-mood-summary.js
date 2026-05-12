// wzp · Mood Mirror — weekly summary chip en sidebar footer.
//
// Lee `/api/wa/mood/weekly` y muestra un chip pequeño con el avg de 7d
// + emoji indicator. Hover muestra detalle (low/high days, delta vs
// semana previa). Refresh diario (cada 12h en background).

const SIX_HOURS_MS = 6 * 60 * 60 * 1000;

let _chip = null;
let _icon = null;

export function init() {
  _chip = document.getElementById("wa-stat-mood");
  _icon = document.getElementById("wa-stat-mood-icon");
  if (!_chip || !_icon) return;
  refresh();
  setInterval(refresh, SIX_HOURS_MS);
}

async function refresh() {
  try {
    const r = await fetch("/api/wa/mood/weekly", { credentials: "same-origin" });
    if (!r.ok) return;
    const data = await r.json();
    const s = (data || {}).summary || {};
    if (!s.has_data) {
      _chip.hidden = true;
      return;
    }
    render(s);
  } catch (e) {
    _chip.hidden = true;
  }
}

function render(s) {
  const avg = Number(s.avg_7d || 0);
  const icon = pickIcon(avg);
  _icon.textContent = icon;
  _chip.hidden = false;
  _chip.title = formatTooltip(s);
  // Color class según avg
  _chip.classList.remove("mood-up", "mood-down", "mood-flat");
  if (avg > 0.10) _chip.classList.add("mood-up");
  else if (avg < -0.10) _chip.classList.add("mood-down");
  else _chip.classList.add("mood-flat");
}

function pickIcon(avg) {
  if (avg > 0.15) return "🌞";
  if (avg > 0.05) return "🙂";
  if (avg > -0.05) return "😐";
  if (avg > -0.15) return "🙁";
  return "🌧️";
}

function formatTooltip(s) {
  const avg = Number(s.avg_7d || 0).toFixed(2);
  const lines = [`mood 7d · avg ${avg}`];
  if (s.low_days || s.high_days) {
    const parts = [];
    if (s.low_days) parts.push(`${s.low_days}d bajos`);
    if (s.high_days) parts.push(`${s.high_days}d altos`);
    lines.push(parts.join(" · "));
  }
  if (s.delta_vs_prev_week !== null && s.delta_vs_prev_week !== undefined) {
    const d = Number(s.delta_vs_prev_week);
    const arrow = d > 0.02 ? "↑" : d < -0.02 ? "↓" : "→";
    lines.push(`vs sem. previa: ${arrow} ${d > 0 ? "+" : ""}${d.toFixed(2)}`);
  }
  return lines.join("\n");
}
