// panel-monitoring.mjs — paneles de monitoreo: retrieval health, loops aging,
// eval trend, vault activity, captured.

import { escapeHTML, fmtTimeAgo, obsidianUrl, renderPanelList } from "./core.mjs";
import { sparkline, stackedBar } from "./charts.mjs";

export function renderRetrievalHealth(payload) {
  const trend = payload.signals?.eval_trend;
  const panel = document.getElementById("p-retrieval");
  if (!panel) return;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
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

export function renderLoopsAging(payload) {
  const f = payload.signals?.followup_aging;
  const panel = document.getElementById("p-loops-aging");
  if (!panel) return;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");

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

  // Fallback derivado de loops_stale + loops_activo si el cache de followup_aging está frío.
  if (total === 0) {
    const loopsStale = (payload.signals?.loops_stale || []).length;
    const loopsActivo = (payload.signals?.loops_activo || []).length;
    if (loopsStale > 0 || loopsActivo > 0) {
      stale = loopsStale;
      for (const l of payload.signals?.loops_activo || []) {
        const age = Number(l.age_days || 0);
        if (age >= 8) aging++;
        else fresh++;
      }
      total = fresh + aging + stale;
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

export function renderEvalTrend(payload) {
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

export function renderVaultActivity(payload) {
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

export function renderCaptured(payload) {
  const evidence = payload.today?.evidence || {};
  const inboxToday = evidence.inbox_today || [];
  const localToday = (() => {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  })();
  const todayItems = inboxToday.filter((it) =>
    (it.modified || "").startsWith(localToday)
  );

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

  renderPanelList("p-captured", [], {
    emptyText: "nada capturado hoy",
  });
}
