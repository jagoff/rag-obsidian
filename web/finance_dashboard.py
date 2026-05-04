"""Backend del dashboard de finanzas (`/finance`).

Lee dos carpetas iCloud (separadas desde el 2026-05-04):

- ``iCloud~amoos~Tally4/Documents`` (env `OBSIDIAN_RAG_MOZE_DIR`):
  CSVs ``MOZE_*.csv``.
- ``CloudDocs/Finances`` (env `OBSIDIAN_RAG_FINANCE_DIR`):
  xlsx de tarjetas + PDFs de transferencias.

Y consolida en un payload con shape estable que el frontend renderiza sin
checks defensivos.

Fuentes soportadas:

1. ``MOZE_*.csv`` — el export del app personal del user. Es la fuente
   PRIMARIA para gastos/ingresos categorizados (Main Category, Subcategory,
   Account, Currency, Store, Date, Time, Price, Fee, Bonus, Note, Tags).
   Si hay varios CSV (re-exports en distintos meses), todos se mergean
   y se deduplica por fingerprint ``(date|time|account|name|store|abs(price))``
   — el user dijo "MOZE puede estar repetida" y este es el contrato.

2. ``Último resumen - <Marca> <Últimos4>.xlsx`` — los resúmenes de tarjeta
   de crédito que emite el banco. Reutiliza ``_parse_credit_card_xlsx``
   (web/server.py). Cada xlsx queda como una "tarjeta" en el dashboard:
   total a pagar, vencimiento, lista de consumos del ciclo. NO se mergea
   con MOZE — el banco emite estos por ciclo y queremos verlos aparte.

3. ``*Santander*.pdf`` (o cualquier PDF con tabla de transferencias) —
   listado de transferencias bancarias. Se parsea con `pdftotext -layout`
   buscando líneas con shape `DD/MM/YYYY  Transferencia  <destinatario>
   <cuenta>  $<monto>`. Se usa para una sección aparte de "transferencias
   bancarias" (no se mergea con MOZE — pueden o no estar; suelen ser
   diferentes movimientos: MOZE = gastos categorizados, PDF = comprobantes
   de transferencia salientes).

El dashboard NO intenta deduplicar entre fuentes (MOZE vs banco) — son
vistas complementarias, no la misma data en formatos distintos. El user
es el que aclaró: "moze es app personal, los otros son del banco".

Funciones públicas:
    snapshot(now=None) → dict con shape estable consumido por
        ``GET /api/finance``. Cache LRU por (paths, mtimes) — re-export
        de cualquier archivo invalida.

Performance budget: cold ≈ 200ms (parsea CSV ~600 filas + xlsx + PDF);
warm < 5ms (cache hit). El endpoint pone TTL de 60s encima.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

# Cache compartido — clave = tupla (path, mtime) ordenada de TODOS los
# archivos relevantes en _MOZE_BACKUP_DIR + _FINANCE_BACKUP_DIR. Re-export
# de cualquier archivo invalida; agregar/quitar archivos también.
_DASHBOARD_CACHE: dict = {"key": None, "payload": None}
_DASHBOARD_CACHE_LOCK = threading.Lock()

# Tipos canónicos. MOZE usa "Expense", "Income", "Balance Adjustment",
# "Receivable" — los normalizamos para que el frontend no haga case-insensitive
# matching.
_TYPE_NORMALIZE = {
    "expense": "expense",
    "income": "income",
    "balance adjustment": "balance_adjustment",
    "receivable": "receivable",
    "transfer": "transfer",
}

# USDB = "USD billete" (cash USD físicos en la lógica de MOZE, distinto de
# USD digital en cuenta). Para los KPIs los unificamos a "USD" pero el
# detalle por cuenta se preserva en accounts/.
_CURRENCY_BUCKET = {
    "ARS": "ARS",
    "USD": "USD",
    "USDB": "USD",
    "USDT": "USD",  # Binance stablecoin → USD a fines del dashboard
}


# ── Helpers de parseo ────────────────────────────────────────────────────


def _moze_pnum(s: str) -> float:
    """Parsea precios de MOZE. Las celdas vienen como ``"3710512"``,
    ``"2026,74"``, ``"-120000"``. El patrón:

    - Sin separador → entero (`3710512`).
    - Con coma → decimal ES (`2026,74` → 2026.74).
    - Con punto → decimal US (`24.99` → 24.99). Raro en MOZE pero defensivo.
    - Múltiples puntos → miles ES sin decimal (`1.234.567` → 1234567).

    Mismo algoritmo que `_parse_ars_or_usd` (web/server.py) pero para celdas
    sin símbolo de moneda. Devuelve 0.0 si no parsea.
    """
    s = (s or "").strip()
    if not s:
        return 0.0
    if "," in s:
        # Formato ES: punto = miles, coma = decimal.
        s2 = s.replace(".", "").replace(",", ".")
    elif s.count(".") > 1:
        # Múltiples puntos → todos miles.
        s2 = s.replace(".", "")
    else:
        # Un solo punto o ninguno: dejar como está.
        s2 = s
    try:
        return float(s2)
    except ValueError:
        return 0.0


def _moze_date(raw: str) -> date | None:
    """MOZE export usa ``MM/DD/YYYY`` (formato US, observado en el CSV
    real). Si en el futuro el export cambia a ISO, soportamos ambos.
    Devuelve `None` si no parsea — el caller debe skipear la fila.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    # Try MM/DD/YYYY (US — formato observado en el CSV de MOZE).
    try:
        return datetime.strptime(raw, "%m/%d/%Y").date()
    except ValueError:
        pass
    # Fallback ISO (defensive).
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass
    return None


def _fingerprint(date_iso: str, time_str: str, account: str, name: str, store: str, amount: float) -> str:
    """Fingerprint para dedup de filas de MOZE. La clave incluye `time`
    porque MOZE permite múltiples gastos del mismo monto/store en el mismo
    día (ej. 2 ubers idénticos a horas distintas). NO incluye Currency
    porque la combinación (account, amount) ya implica moneda — agregarlo
    sería ruido que evita el dedup cuando el user re-exporta y el case del
    currency cambia.
    """
    parts = (date_iso, (time_str or "").strip(), account.strip(), name.strip(),
             store.strip(), f"{abs(amount):.4f}")
    return "|".join(parts)


# ── MOZE CSV ─────────────────────────────────────────────────────────────


def _load_moze_rows(finance_dir: Path) -> tuple[list[dict], list[dict]]:
    """Carga + dedupea TODOS los `MOZE_*.csv` del dir.

    Devuelve `(transactions, sources)`:

    - ``transactions``: lista de dicts normalizados con keys
      ``date, time, type, type_raw, category, subcategory, name, store,
      account, currency, currency_bucket, amount, fee, bonus, project,
      note, tags, source, source_file, fingerprint``. Ordenada por
      ``date`` descendente. ``amount`` mantiene el signo original del CSV
      (MOZE usa positivo siempre; el ``type`` distingue Expense/Income).

    - ``sources``: lista de `{path, mtime, rows_total, rows_kept,
      rows_dup}` por archivo, para mostrar "leído de N CSVs (M dedupeadas)"
      en el footer del dashboard.
    """
    csvs = sorted(
        finance_dir.glob("MOZE_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not csvs:
        return ([], [])

    seen_fingerprints: set[str] = set()
    transactions: list[dict] = []
    sources: list[dict] = []

    for path in csvs:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        rows_total = 0
        rows_kept = 0
        rows_dup = 0
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    rows_total += 1
                    d = _moze_date(r.get("Date", ""))
                    if not d:
                        continue
                    amount = _moze_pnum(r.get("Price", ""))
                    type_raw = (r.get("Type") or "").strip()
                    type_norm = _TYPE_NORMALIZE.get(type_raw.lower(), "other")
                    account = (r.get("Account") or "").strip()
                    name = (r.get("Name") or "").strip()
                    store = (r.get("Store") or "").strip()
                    time_str = (r.get("Time") or "").strip()
                    fp = _fingerprint(d.isoformat(), time_str, account, name, store, amount)
                    if fp in seen_fingerprints:
                        rows_dup += 1
                        continue
                    seen_fingerprints.add(fp)
                    rows_kept += 1
                    currency = (r.get("Currency") or "").strip().upper()
                    transactions.append({
                        "date": d.isoformat(),
                        "time": time_str,
                        "type": type_norm,
                        "type_raw": type_raw,
                        "category": (r.get("Main Category") or "").strip() or "—",
                        "subcategory": (r.get("Subcategory") or "").strip() or "—",
                        "name": name,
                        "store": store,
                        "account": account,
                        "currency": currency,
                        "currency_bucket": _CURRENCY_BUCKET.get(currency, currency or "ARS"),
                        "amount": amount,
                        "fee": _moze_pnum(r.get("Fee", "0")),
                        "bonus": _moze_pnum(r.get("Bonus", "0")),
                        "project": (r.get("Project") or "").strip(),
                        "note": (r.get("Note") or "").strip(),
                        "tags": (r.get("Tags") or "").strip(),
                        "source": "moze",
                        "source_file": path.name,
                        "fingerprint": fp,
                    })
        except (OSError, UnicodeDecodeError, csv.Error):
            # Archivo corrupto / encoding raro → skipear archivo entero
            # silenciosamente y seguir con los demás.
            continue
        sources.append({
            "path": path.name,
            "mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "rows_total": rows_total,
            "rows_kept": rows_kept,
            "rows_dup": rows_dup,
        })

    transactions.sort(key=lambda t: (t["date"], t["time"]), reverse=True)
    return (transactions, sources)


# ── Santander PDF ────────────────────────────────────────────────────────


# Línea de transferencia: "DD/MM/YYYY  Transferencia  <destinatario>  <cuenta>  $<monto>"
# El layout de pdftotext mantiene columnas con espacios. La cuenta es
# `\d{3}-\d{6}/\d` (CBU corto) — la usamos como anchor para separar
# destinatario de monto.
_PDF_TRANSFER_RE = re.compile(
    r"^\s*(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<type>Transferencia|Pago|Débito|Debito|Crédito|Credito)\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<account>\d{3}-\d{6}/\d|--|-)\s+"
    r"\$\s*(?P<amount>[\d\.]+,\d{2})\s*$",
    re.IGNORECASE,
)


def _parse_pdf_amount_ars(s: str) -> float:
    """`60.000,00` → 60000.00. PDF Santander siempre formato ES."""
    return _moze_pnum(s)


def _load_pdf_transfers(finance_dir: Path) -> tuple[list[dict], list[dict]]:
    """Parsea CADA `*.pdf` del dir buscando líneas con shape de transferencia
    bancaria. Si no hay ``pdftotext`` instalado o el PDF no rinde texto,
    devuelve listas vacías sin romper.

    Soporta nombres "Abril - Santander.pdf", "Marzo - Santander.pdf",
    "Comprobantes 2026-03.pdf", etc. — no se asume naming.
    """
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ([], [])

    pdfs = sorted(finance_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    if not pdfs:
        return ([], [])

    transfers: list[dict] = []
    sources: list[dict] = []

    for path in pdfs:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        try:
            res = subprocess.run(
                [pdftotext, "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode != 0:
            continue
        text = res.stdout or ""
        rows_kept = 0
        for line in text.splitlines():
            m = _PDF_TRANSFER_RE.match(line)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group("date"), "%d/%m/%Y").date()
            except ValueError:
                continue
            amount = _parse_pdf_amount_ars(m.group("amount"))
            recipient = re.sub(r"\s+", " ", m.group("desc")).strip()
            ttype = m.group("type").lower()
            transfers.append({
                "date": d.isoformat(),
                "type": "transfer" if "transfer" in ttype else ttype,
                "type_raw": m.group("type"),
                "recipient": recipient,
                "account": m.group("account"),
                "amount": amount,
                "currency": "ARS",
                "currency_bucket": "ARS",
                "source": "bank_pdf",
                "source_file": path.name,
            })
            rows_kept += 1
        sources.append({
            "path": path.name,
            "mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "rows_kept": rows_kept,
        })

    transfers.sort(key=lambda t: t["date"], reverse=True)
    return (transfers, sources)


# ── Tarjetas de crédito (xlsx) — reutiliza el parser existente ──────────


def _load_credit_cards(finance_dir: Path) -> list[dict]:
    """Llama a ``web.server._parse_credit_card_xlsx`` para cada xlsx
    `Último resumen*.xlsx` / `Ultimo resumen*.xlsx` en el dir, y devuelve
    los dicts. Sin cache propio — el endpoint del dashboard cachea TODO
    junto.

    Import lazy de ``web.server`` para evitar dependencia circular en
    import-time (el dashboard lo importa el server después de definir
    sus parsers).
    """
    try:
        from web.server import _parse_credit_card_xlsx  # type: ignore
    except ImportError:
        return []

    seen: set[Path] = set()
    for pattern in ("Último resumen*.xlsx", "Ultimo resumen*.xlsx"):
        for p in finance_dir.glob(pattern):
            seen.add(p)
    if not seen:
        return []
    cards: list[dict] = []
    for p in sorted(seen, key=lambda p: p.name):
        parsed = _parse_credit_card_xlsx(p)
        if parsed:
            cards.append(parsed)
    cards.sort(key=lambda c: (c.get("due_date") is None, c.get("due_date") or ""))
    return cards


# ── Agregaciones para el dashboard ───────────────────────────────────────


def _ym(d: str) -> str:
    """`YYYY-MM-DD` → `YYYY-MM`. El frontend usa esto como bucket mensual."""
    return d[:7] if len(d) >= 7 else d


def _kpi_sparks(txs: list[dict], now: datetime, n_months: int = 6) -> dict:
    """Serie mensual de los últimos N meses por KPI, alimenta el sparkline
    del frontend. Cada serie es de `n_months` puntos (rellenando con 0
    los meses sin data) terminando en el mes actual.

    Devuelve dict con keys ``expenses_ars, expenses_usd, income_ars,
    balance_ars, txs_count``. Cada value es una lista de N floats.
    """
    # Generar labels de los últimos N meses terminando en `now`.
    cur_y, cur_m = now.year, now.month
    labels: list[str] = []
    for _ in range(n_months):
        labels.append(f"{cur_y:04d}-{cur_m:02d}")
        cur_m -= 1
        if cur_m == 0:
            cur_m = 12
            cur_y -= 1
    labels.reverse()
    label_set = set(labels)
    expenses_ars = {lab: 0.0 for lab in labels}
    expenses_usd = {lab: 0.0 for lab in labels}
    income_ars = {lab: 0.0 for lab in labels}
    income_usd = {lab: 0.0 for lab in labels}
    txs_count = {lab: 0 for lab in labels}
    for t in txs:
        ym = _ym(t["date"])
        if ym not in label_set:
            continue
        cb = t["currency_bucket"]
        if t["type"] == "expense":
            txs_count[ym] += 1
            if cb == "ARS":
                expenses_ars[ym] += abs(t["amount"])
            elif cb == "USD":
                expenses_usd[ym] += abs(t["amount"])
        elif t["type"] == "income":
            txs_count[ym] += 1
            if cb == "ARS":
                income_ars[ym] += abs(t["amount"])
            elif cb == "USD":
                income_usd[ym] += abs(t["amount"])
    balance_ars_series = [income_ars[lab] - expenses_ars[lab] for lab in labels]
    return {
        "labels": labels,
        "expenses_ars": [expenses_ars[lab] for lab in labels],
        "expenses_usd": [expenses_usd[lab] for lab in labels],
        "income_ars": [income_ars[lab] for lab in labels],
        "income_usd": [income_usd[lab] for lab in labels],
        "balance_ars": balance_ars_series,
        "txs_count": [float(txs_count[lab]) for lab in labels],
    }


def _kpis(txs: list[dict], now: datetime) -> dict:
    """Calcula los KPI hero del dashboard: gastos del mes actual, gastos
    del mes previo, ingresos del mes, balance neto, # transacciones, top
    categoría. Cada KPI viene con `value`, `delta_pct` (vs mes previo
    cuando aplica), `n_samples`, e `insufficient` para que el frontend
    pinte el badge "datos insuf." sin lógica. Adicional: cada KPI trae
    `spark` (serie de los últimos 6 meses) para que el frontend pinte un
    sparkline mini bajo el valor.
    """
    this_ym = now.strftime("%Y-%m")
    prev_anchor = (now.replace(day=1) - timedelta(days=1))
    prev_ym = prev_anchor.strftime("%Y-%m")

    expenses_this: dict[str, float] = defaultdict(float)
    expenses_prev: dict[str, float] = defaultdict(float)
    income_this: dict[str, float] = defaultdict(float)
    income_prev: dict[str, float] = defaultdict(float)
    by_cat_this: Counter = Counter()
    n_this = 0
    n_prev = 0

    for t in txs:
        ym = _ym(t["date"])
        bucket = t["currency_bucket"]
        if t["type"] == "expense":
            if ym == this_ym:
                expenses_this[bucket] += abs(t["amount"])
                if bucket == "ARS":
                    by_cat_this[t["category"]] += abs(t["amount"])
                n_this += 1
            elif ym == prev_ym:
                expenses_prev[bucket] += abs(t["amount"])
                n_prev += 1
        elif t["type"] == "income":
            if ym == this_ym:
                income_this[bucket] += abs(t["amount"])
            elif ym == prev_ym:
                income_prev[bucket] += abs(t["amount"])

    def _delta(a: float, b: float) -> float | None:
        if not b:
            return None
        return (a - b) / b

    def _kpi(value: float, delta: float | None, n: int, threshold: int = 1) -> dict:
        return {
            "value": value,
            "delta_pct": delta,
            "n_samples": n,
            "insufficient": n < threshold,
        }

    top_cat_name = "—"
    top_cat_amount = 0.0
    if by_cat_this:
        top_cat_name, top_cat_amount = by_cat_this.most_common(1)[0]

    ars_this = expenses_this["ARS"]
    ars_prev = expenses_prev["ARS"]
    usd_this = expenses_this["USD"]
    usd_prev = expenses_prev["USD"]

    sparks = _kpi_sparks(txs, now, n_months=6)

    expenses_ars = _kpi(ars_this, _delta(ars_this, ars_prev), n_this)
    expenses_ars["spark"] = sparks["expenses_ars"]
    expenses_usd = _kpi(usd_this, _delta(usd_this, usd_prev), n_this)
    expenses_usd["spark"] = sparks["expenses_usd"]
    income_ars = _kpi(income_this["ARS"], _delta(income_this["ARS"], income_prev["ARS"]), n_this)
    income_ars["spark"] = sparks["income_ars"]
    balance_ars = _kpi(income_this["ARS"] - ars_this, None, n_this)
    balance_ars["spark"] = sparks["balance_ars"]
    txs_count_kpi = _kpi(float(n_this), _delta(float(n_this), float(n_prev)) if n_prev else None, n_this)
    txs_count_kpi["spark"] = sparks["txs_count"]

    return {
        "expenses_ars": expenses_ars,
        "expenses_usd": expenses_usd,
        "income_ars": income_ars,
        "balance_ars": balance_ars,
        "txs_count": txs_count_kpi,
        "top_category": {
            "value": top_cat_amount,
            "name": top_cat_name,
            "n_samples": n_this,
            "insufficient": n_this == 0,
            "spark": [float(by_cat_this.get(top_cat_name, 0))],  # placeholder, no es time-serie
        },
        "spark_labels": sparks["labels"],
    }


def _by_month(txs: list[dict], months: int = 12) -> dict:
    """Serie temporal: gastos vs ingresos por mes (últimos N meses), por
    bucket de moneda. Devuelve:

        {
            "labels": ["2025-05", "2025-06", ..., "2026-04"],
            "expenses_ars": [...],
            "expenses_usd": [...],
            "income_ars": [...],
            "income_usd": [...],
        }
    """
    if not txs:
        return {
            "labels": [],
            "expenses_ars": [],
            "expenses_usd": [],
            "income_ars": [],
            "income_usd": [],
        }
    # Bucketizar.
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {
        "expenses_ars": 0.0, "expenses_usd": 0.0,
        "income_ars": 0.0, "income_usd": 0.0,
    })
    for t in txs:
        ym = _ym(t["date"])
        cb = t["currency_bucket"]
        if t["type"] == "expense":
            key = "expenses_ars" if cb == "ARS" else ("expenses_usd" if cb == "USD" else None)
        elif t["type"] == "income":
            key = "income_ars" if cb == "ARS" else ("income_usd" if cb == "USD" else None)
        else:
            key = None
        if key:
            buckets[ym][key] += abs(t["amount"])
    if not buckets:
        return {
            "labels": [],
            "expenses_ars": [],
            "expenses_usd": [],
            "income_ars": [],
            "income_usd": [],
        }
    sorted_keys = sorted(buckets.keys())
    end = sorted_keys[-1]
    end_y, end_m = int(end[:4]), int(end[5:7])
    labels: list[str] = []
    cursor_y, cursor_m = end_y, end_m
    for _ in range(months):
        labels.append(f"{cursor_y:04d}-{cursor_m:02d}")
        cursor_m -= 1
        if cursor_m == 0:
            cursor_m = 12
            cursor_y -= 1
    labels.reverse()
    series_ea = [buckets[lab]["expenses_ars"] for lab in labels]
    series_eu = [buckets[lab]["expenses_usd"] for lab in labels]
    series_ia = [buckets[lab]["income_ars"] for lab in labels]
    series_iu = [buckets[lab]["income_usd"] for lab in labels]
    return {
        "labels": labels,
        "expenses_ars": series_ea,
        "expenses_usd": series_eu,
        "income_ars": series_ia,
        "income_usd": series_iu,
    }


def _by_category(txs: list[dict], window_days: int, now: datetime, currency: str = "ARS") -> dict:
    """Donut data: gastos por main category dentro de la ventana."""
    cutoff = (now - timedelta(days=window_days)).date().isoformat()
    by_cat: Counter = Counter()
    by_cat_subs: dict[str, Counter] = defaultdict(Counter)
    for t in txs:
        if t["type"] != "expense":
            continue
        if t["currency_bucket"] != currency:
            continue
        if t["date"] < cutoff:
            continue
        by_cat[t["category"]] += abs(t["amount"])
        by_cat_subs[t["category"]][t["subcategory"]] += abs(t["amount"])
    items = [
        {"name": name, "amount": amount}
        for name, amount in by_cat.most_common()
    ]
    subs = {
        cat: [{"name": s, "amount": a} for s, a in c.most_common()]
        for cat, c in by_cat_subs.items()
    }
    return {"items": items, "subcategories": subs, "currency": currency, "window_days": window_days}


def _top_stores(txs: list[dict], window_days: int, now: datetime, currency: str = "ARS", n: int = 15) -> dict:
    """Top N comercios por gasto en la ventana. ARS-only por default
    (los gastos USD suelen ser muy pocos: subscriptions web)."""
    cutoff = (now - timedelta(days=window_days)).date().isoformat()
    by_store: Counter = Counter()
    counts: Counter = Counter()
    for t in txs:
        if t["type"] != "expense":
            continue
        if t["currency_bucket"] != currency:
            continue
        if t["date"] < cutoff:
            continue
        # Si no hay store, fallback a name (algunos consumos manuales no
        # tienen Store seteado).
        key = t["store"] or t["name"] or "—"
        by_store[key] += abs(t["amount"])
        counts[key] += 1
    items = [
        {"name": name, "amount": amount, "count": counts[name]}
        for name, amount in by_store.most_common(n)
    ]
    return {"items": items, "currency": currency, "window_days": window_days}


def _by_account(txs: list[dict]) -> dict:
    """Resumen por cuenta (ARS Cash, Santander ARS, Santander USD, IOL,
    USD Cash, etc.). Solo gastos + ingresos (skip Balance Adjustment
    porque distorsiona — son ajustes contables, no flow de plata)."""
    by_acc_ex: Counter = Counter()
    by_acc_in: Counter = Counter()
    counts: Counter = Counter()
    for t in txs:
        if t["type"] == "expense":
            by_acc_ex[t["account"]] += abs(t["amount"])
            counts[t["account"]] += 1
        elif t["type"] == "income":
            by_acc_in[t["account"]] += abs(t["amount"])
            counts[t["account"]] += 1
    accounts = sorted(
        set(by_acc_ex.keys()) | set(by_acc_in.keys()),
        key=lambda a: by_acc_ex.get(a, 0.0) + by_acc_in.get(a, 0.0),
        reverse=True,
    )
    items = [
        {
            "account": a,
            "expenses": by_acc_ex.get(a, 0.0),
            "income": by_acc_in.get(a, 0.0),
            "count": counts[a],
            "net": by_acc_in.get(a, 0.0) - by_acc_ex.get(a, 0.0),
        }
        for a in accounts
    ]
    return {"items": items}


def _recent_transactions(txs: list[dict], n: int = 50) -> list[dict]:
    """Las últimas N transacciones (excluyendo balance adjustments).
    Recortamos campos que el frontend no usa en la tabla."""
    out: list[dict] = []
    for t in txs:
        if t["type"] in ("balance_adjustment", "other"):
            continue
        out.append({
            "date": t["date"],
            "time": t["time"],
            "type": t["type"],
            "category": t["category"],
            "subcategory": t["subcategory"],
            "name": t["name"],
            "store": t["store"],
            "account": t["account"],
            "currency": t["currency"],
            "amount": t["amount"],
            "note": t["note"],
        })
        if len(out) >= n:
            break
    return out


def _transfers_summary(transfers: list[dict], now: datetime, months: int = 12) -> dict:
    """Resumen de transferencias bancarias (PDF). Por mes + top destinatarios.

    El `by_month` siempre devuelve los últimos `months` meses (rellenando
    con 0 los que no tienen PDF) — así el bar chart del frontend no se ve
    "vacío" cuando solo hay 1 PDF. Si en el futuro el user agrega PDFs de
    meses anteriores, las barras correspondientes aparecen automáticas.
    """
    if not transfers:
        return {"by_month": {"labels": [], "amounts": []}, "by_recipient": [], "total": 0.0, "count": 0}
    by_month: dict[str, float] = defaultdict(float)
    by_rec: Counter = Counter()
    rec_count: Counter = Counter()
    total = 0.0
    for t in transfers:
        ym = _ym(t["date"])
        by_month[ym] += t["amount"]
        by_rec[t["recipient"]] += t["amount"]
        rec_count[t["recipient"]] += 1
        total += t["amount"]
    # Generar labels secuenciales de los últimos N meses terminando en `now`,
    # rellenando los meses sin PDF con 0.
    cur_y, cur_m = now.year, now.month
    labels: list[str] = []
    for _ in range(months):
        labels.append(f"{cur_y:04d}-{cur_m:02d}")
        cur_m -= 1
        if cur_m == 0:
            cur_m = 12
            cur_y -= 1
    labels.reverse()
    return {
        "by_month": {"labels": labels, "amounts": [by_month.get(k, 0.0) for k in labels]},
        "by_recipient": [
            {"name": name, "amount": amount, "count": rec_count[name]}
            for name, amount in by_rec.most_common(20)
        ],
        "total": total,
        "count": len(transfers),
    }


# ── Snapshot público ─────────────────────────────────────────────────────


def snapshot(
    finance_dir: Path | None = None,
    now: datetime | None = None,
    months: int = 12,
    window_days: int = 30,
    moze_dir: Path | None = None,
) -> dict:
    """Snapshot completo para `/api/finance`. Cache LRU por (paths, mtimes).

    Args:
        finance_dir: Override del dir de tarjetas/PDFs (para tests). Default:
            ``_FINANCE_BACKUP_DIR`` de web.server (env `OBSIDIAN_RAG_FINANCE_DIR`).
        moze_dir: Override del dir de MOZE CSV (para tests). Default:
            ``_MOZE_BACKUP_DIR`` de web.server (env `OBSIDIAN_RAG_MOZE_DIR`).
        now: Override de "now" (para tests). Default: `datetime.now()`.
        months: Cuántos meses de serie temporal devolver (default 12).
        window_days: Ventana para los KPI por categoría / top stores
            (default 30 días).

    Returns:
        Dict con shape estable. Nunca lanza — silent-fail per
        convención web/.
    """
    now = now or datetime.now()
    # Resolución de paths:
    # - Ambos None → defaults de web.server (los reales del usuario).
    # - Solo `finance_dir` pasado → asumir mismo dir para MOZE (back-compat
    #   con tests pre-split y con setups donde ambas fuentes vivían juntas).
    # - Solo `moze_dir` pasado → simétrico (poco común pero defensivo).
    if finance_dir is None and moze_dir is None:
        from web.server import _FINANCE_BACKUP_DIR, _MOZE_BACKUP_DIR  # type: ignore
        finance_dir = _FINANCE_BACKUP_DIR
        moze_dir = _MOZE_BACKUP_DIR
    elif moze_dir is None:
        moze_dir = finance_dir
    elif finance_dir is None:
        finance_dir = moze_dir

    # MOZE dir missing es OK si igual hay tarjetas/PDFs en finance_dir.
    # Solo abortamos si NINGUNO de los dos existe.
    if not finance_dir.exists() and not moze_dir.exists():
        return _empty_payload(now, "finance_dir_missing", str(finance_dir), str(moze_dir))

    # Cache key: tupla con (str(path), mtime) de cada archivo CSV/PDF/XLSX
    # de ambos dirs. Re-export de cualquier archivo invalida; agregar/quitar
    # archivos también.
    try:
        moze_files = list(moze_dir.glob("MOZE_*.csv")) if moze_dir.exists() else []
        finance_files = (
            list(finance_dir.glob("Último resumen*.xlsx"))
            + list(finance_dir.glob("Ultimo resumen*.xlsx"))
            + list(finance_dir.glob("*.pdf"))
        ) if finance_dir.exists() else []
        all_files = moze_files + finance_files
        cache_key = tuple(sorted((str(p), p.stat().st_mtime) for p in all_files))
        # Incluir window_days/months en la key porque distintas vistas
        # del mismo dataset cachean por separado.
        cache_key = (cache_key, months, window_days, now.strftime("%Y-%m"))
    except OSError:
        cache_key = None

    if cache_key is not None:
        with _DASHBOARD_CACHE_LOCK:
            if _DASHBOARD_CACHE.get("key") == cache_key:
                return _DASHBOARD_CACHE["payload"]

    transactions, moze_sources = _load_moze_rows(moze_dir) if moze_dir.exists() else ([], [])
    transfers, pdf_sources = _load_pdf_transfers(finance_dir) if finance_dir.exists() else ([], [])
    cards = _load_credit_cards(finance_dir) if finance_dir.exists() else []

    if not transactions and not transfers and not cards:
        return _empty_payload(now, "no_data", str(finance_dir), str(moze_dir))

    payload = {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "finance_dir": str(finance_dir),
            "moze_dir": str(moze_dir),
            "months": months,
            "window_days": window_days,
            "moze_sources": moze_sources,
            "pdf_sources": pdf_sources,
            "card_files": [c.get("source_file") for c in cards],
            "n_transactions": len(transactions),
            "n_transfers": len(transfers),
            "n_cards": len(cards),
        },
        "kpis": _kpis(transactions, now),
        "by_month": _by_month(transactions, months),
        "by_category_ars": _by_category(transactions, window_days, now, "ARS"),
        "by_category_usd": _by_category(transactions, window_days, now, "USD"),
        "top_stores_ars": _top_stores(transactions, window_days, now, "ARS"),
        "top_stores_usd": _top_stores(transactions, window_days, now, "USD"),
        "by_account": _by_account(transactions),
        "recent": _recent_transactions(transactions, n=50),
        "transfers": _transfers_summary(transfers, now, months=months),
        "transfers_recent": transfers[:50],
        "cards": cards,
    }

    if cache_key is not None:
        with _DASHBOARD_CACHE_LOCK:
            _DASHBOARD_CACHE["key"] = cache_key
            _DASHBOARD_CACHE["payload"] = payload

    return payload


def _empty_payload(now: datetime, reason: str, finance_dir: str, moze_dir: str = "") -> dict:
    """Shape válido pero vacío. El frontend pinta "sin datos" sin crashear."""
    return {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "finance_dir": finance_dir,
            "moze_dir": moze_dir,
            "reason": reason,
            "moze_sources": [],
            "pdf_sources": [],
            "card_files": [],
            "n_transactions": 0,
            "n_transfers": 0,
            "n_cards": 0,
        },
        "kpis": {
            "expenses_ars": {"value": 0, "delta_pct": None, "n_samples": 0, "insufficient": True, "spark": []},
            "expenses_usd": {"value": 0, "delta_pct": None, "n_samples": 0, "insufficient": True, "spark": []},
            "income_ars": {"value": 0, "delta_pct": None, "n_samples": 0, "insufficient": True, "spark": []},
            "balance_ars": {"value": 0, "delta_pct": None, "n_samples": 0, "insufficient": True, "spark": []},
            "txs_count": {"value": 0, "delta_pct": None, "n_samples": 0, "insufficient": True, "spark": []},
            "top_category": {"value": 0, "name": "—", "n_samples": 0, "insufficient": True, "spark": []},
            "spark_labels": [],
        },
        "by_month": {"labels": [], "expenses_ars": [], "expenses_usd": [], "income_ars": [], "income_usd": []},
        "by_category_ars": {"items": [], "subcategories": {}, "currency": "ARS", "window_days": 30},
        "by_category_usd": {"items": [], "subcategories": {}, "currency": "USD", "window_days": 30},
        "top_stores_ars": {"items": [], "currency": "ARS", "window_days": 30},
        "top_stores_usd": {"items": [], "currency": "USD", "window_days": 30},
        "by_account": {"items": []},
        "recent": [],
        "transfers": {"by_month": {"labels": [], "amounts": []}, "by_recipient": [], "total": 0, "count": 0},
        "transfers_recent": [],
        "cards": [],
    }
