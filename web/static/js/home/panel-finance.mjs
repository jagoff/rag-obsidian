// panel-finance.mjs — paneles de finanzas: ARS y movimientos de tarjetas.

import { escapeHTML, fmtCurrencyARS, renderPanelList } from "./core.mjs";

export function renderFinance(payload) {
  const fin = payload.signals?.finance;
  const panel = document.getElementById("p-finance");
  if (!panel) return;
  const body = panel.querySelector("[data-body]");
  const count = panel.querySelector("[data-count]");
  const foot = panel.querySelector("[data-foot]");
  // Shape real: fin.ars = { this_month, prev_month, delta_pct, run_rate_daily, projected, top_categories[] }
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

export function renderCards(payload) {
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

  allPurchases.sort((a, b) => (b.date || "").localeCompare(a.date || ""));

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
