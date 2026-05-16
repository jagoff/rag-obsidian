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

2. ``Último resumen - <Marca> <Últimos4>.xlsx`` y
   ``Resumen de tarjeta de crédito <Marca>-*.pdf`` — los resúmenes de
   tarjeta de crédito que emite el banco. Cada resumen queda como una
   "tarjeta" en el dashboard: total a pagar, vencimiento, lista de consumos
   del ciclo. Para evitar doble conteo, los consumos ARS quedan aparte; los
   consumos USD sí se suman a los agregados porque MOZE suele no traerlos.

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
import unicodedata
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


def _load_moze_rows(
    finance_dir: Path,
    extra_dirs: tuple[Path, ...] = (),
) -> tuple[list[dict], list[dict]]:
    """Carga + dedupea TODOS los `MOZE_*.csv` accesibles.

    Pre 2026-05-04 globeaba solo `finance_dir`. Post-Tally4 el caller
    puede pasar `extra_dirs` apuntando al cache regenerado desde el
    backup `.zip` (`~/.local/share/obsidian-rag/moze_cache/`). Tests
    aislados llaman sin `extra_dirs` para no contaminarse con CSV
    reales del usuario.

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
    candidates: list[Path] = []
    if finance_dir.exists():
        candidates.extend(finance_dir.glob("MOZE_*.csv"))
    seen = {p.resolve() for p in candidates}
    for d in extra_dirs:
        if not d or not d.exists() or d.resolve() == finance_dir.resolve():
            continue
        for p in d.glob("MOZE_*.csv"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            candidates.append(p)
    csvs = sorted(candidates, key=lambda p: p.stat().st_mtime)
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
    """Parsea CADA `*.pdf` del subdirectorio `Debito/` buscando líneas con
    shape de transferencia bancaria. Si no hay ``pdftotext`` instalado o el
    PDF no rinde texto, devuelve listas vacías sin romper.

    Soporta nombres "Abril - Santander.pdf", "Marzo - Santander.pdf",
    "Comprobantes 2026-03.pdf", etc. — no se asume naming.
    """
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ([], [])

    pdfs = sorted(finance_dir.glob("Debito/*.pdf"), key=lambda p: p.stat().st_mtime)
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


# ── Ingresos (PDF recibos de sueldo) ─────────────────────────────────────


def _load_income_pdfs(finance_dir: Path) -> tuple[list[dict], list[dict]]:
    """Parsea CADA `*.pdf` del subdirectorio `Ingresos/` buscando líneas con
    shape de recibo de sueldo. Extrae periodo y monto neto. Si no hay
    ``pdftotext`` instalado o el PDF no rinde texto, devuelve listas vacías
    sin romper.
    """
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ([], [])

    pdfs = sorted(finance_dir.glob("Ingresos/*.pdf"), key=lambda p: p.stat().st_mtime)
    if not pdfs:
        return ([], [])

    incomes: list[dict] = []
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
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        if res.returncode != 0:
            continue

        text = res.stdout
        # Buscar "Periodo a Pagar" y "Total Neto"
        # Formato: "Periodo a Pagar" en una línea, "Abril 2026" en la siguiente
        # Formato: "Total Neto" en una línea, monto numérico al final de la línea siguiente
        period_match = re.search(r"Periodo a Pagar.*?\n\s*([A-Za-z]+)\s+(\d{4})", text, re.MULTILINE)
        # El monto está al final de la línea siguiente a "Total Neto"
        # Buscamos el último número en esa línea (el formato ES tiene puntos y comas)
        neto_match = re.search(r"Total Neto.*?\n.*?(\d{1,3}(?:\.\d{3})*,\d{2})", text, re.MULTILINE)

        if period_match and neto_match:
            month_str = period_match.group(1)
            year = int(period_match.group(2))
            # Mapear mes español a número
            month_map = {
                "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
                "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
                "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
            }
            month_lower = month_str.lower()
            month = month_map.get(month_lower, 1)

            # Parsear monto: "7.035.567,00" → 7035567.00
            amount_str = neto_match.group(1).replace(".", "").replace(",", ".")
            try:
                amount = float(amount_str)
            except ValueError:
                continue

            # Fecha del recibo (primer día del mes)
            date = datetime(year, month, 1).isoformat()

            incomes.append({
                "date": date,
                "amount": amount,
                "currency": "ARS",
                "source": "income_pdf",
                "source_file": path.name,
            })
            sources.append({
                "path": path.name,
                "mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "amount": amount,
            })

    incomes.sort(key=lambda i: i["date"], reverse=True)
    return (incomes, sources)


# ── Tarjetas de crédito (xlsx/PDF) ──────────────────────────────────────


_CARD_PDF_AMOUNT_RE = re.compile(
    r"(?<![\d])(?P<amount>\d[\d\.]*,\d{2})(?P<negative>-)?\s*\*?"
)
_CARD_PDF_FULL_DATE_RE = re.compile(
    r"^\s*(?P<year>\d{2,4})\s+"
    r"(?P<month>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\.]+)\s+"
    r"(?P<day>\d{1,2})\s+(?P<rest>.+?)\s*$"
)
_CARD_PDF_DAY_RE = re.compile(r"^\s*(?P<day>\d{1,2})\s+(?P<rest>.+?)\s*$")
_CARD_PDF_DATE_TOKEN = (
    r"(\d{1,2})\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\.]+)\s+"
    r"(\d{2,4})"
)
_CARD_PDF_TAX_KEYWORDS = (
    "IMPUESTO",
    "IIBB",
    "IVA ",
    "DB.RG",
    "PERCEP",
    "SELL",
)
_CARD_PDF_SKIP_KEYWORDS = (
    "SALDO ANTERIOR",
    "SALDO ACTUAL",
    "PAGO MINIMO",
    "PAGO EN PESOS",
    "SU PAGO",
    "TOTAL CONSUMOS",
)
_CARD_PDF_MONTHS = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "setiem": 9, "setiembre": 9,
    "sep": 9, "sept": 9, "septiem": 9, "septiembre": 9,
    "oct": 10, "octub": 10, "octubre": 10,
    "nov": 11, "noviem": 11, "noviembre": 11,
    "dic": 12, "diciem": 12, "diciembre": 12,
}


def _strip_accents(s: str) -> str:
    """Normaliza acentos para matchear meses del PDF Santander."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _pdf_card_month(token: str) -> int | None:
    key = _strip_accents(token or "").lower().strip().strip(".")
    if not key:
        return None
    if key in _CARD_PDF_MONTHS:
        return _CARD_PDF_MONTHS[key]
    for prefix, month in _CARD_PDF_MONTHS.items():
        if key.startswith(prefix):
            return month
    return None


def _pdf_card_year(token: str) -> int | None:
    try:
        y = int(token)
    except (TypeError, ValueError):
        return None
    if y < 100:
        return 2000 + y if y < 80 else 1900 + y
    return y


def _pdf_card_date(day_token: str, month_token: str, year_token: str) -> str | None:
    month = _pdf_card_month(month_token)
    year = _pdf_card_year(year_token)
    if not month or not year:
        return None
    try:
        return date(year, month, int(day_token)).isoformat()
    except (TypeError, ValueError):
        return None


def _parse_pdf_card_statement_dates(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Extrae cierre/vencimiento actual y próximo del header Santander PDF."""
    closing_date = due_date = next_closing_date = next_due_date = None
    cur_re = re.compile(
        rf"CIERRE\s+{_CARD_PDF_DATE_TOKEN}\s+VENCIMIENTO\s+{_CARD_PDF_DATE_TOKEN}",
        re.IGNORECASE,
    )
    m = cur_re.search(text)
    if m:
        groups = m.groups()
        closing_date = _pdf_card_date(groups[0], groups[1], groups[2])
        due_date = _pdf_card_date(groups[3], groups[4], groups[5])

    next_re = re.compile(
        rf"Prox\.?\s*Cierre:\s+{_CARD_PDF_DATE_TOKEN}.*?"
        rf"Prox\.?\s*Vto\.?:\s+{_CARD_PDF_DATE_TOKEN}",
        re.IGNORECASE | re.DOTALL,
    )
    m = next_re.search(text)
    if m:
        groups = m.groups()
        next_closing_date = _pdf_card_date(groups[0], groups[1], groups[2])
        next_due_date = _pdf_card_date(groups[3], groups[4], groups[5])

    return closing_date, due_date, next_closing_date, next_due_date


def _parse_pdf_card_line_prefix(
    line: str,
    current_year: int | None,
    current_month: int | None,
) -> tuple[str | None, str | None, int | None, int | None]:
    """Parsea el prefijo de fecha de una línea de movimientos VISA PDF.

    Santander imprime la fecha completa una vez por mes (`26 Marzo 01`) y
    luego solo el día (`05 006635 ...`). Heredamos año/mes de la última
    línea completa para esas continuaciones.
    """
    m = _CARD_PDF_FULL_DATE_RE.match(line)
    if m:
        year = _pdf_card_year(m.group("year"))
        month = _pdf_card_month(m.group("month"))
        if not year or not month:
            return (None, None, current_year, current_month)
        try:
            d = date(year, month, int(m.group("day"))).isoformat()
        except ValueError:
            return (None, None, year, month)
        return (d, m.group("rest"), year, month)

    m = _CARD_PDF_DAY_RE.match(line)
    if not m or not current_year or not current_month:
        return (None, None, current_year, current_month)
    try:
        d = date(current_year, current_month, int(m.group("day"))).isoformat()
    except ValueError:
        return (None, None, current_year, current_month)
    return (d, m.group("rest"), current_year, current_month)


def _parse_pdf_card_amount_from_rest(rest: str) -> tuple[float | None, int | None]:
    matches = list(_CARD_PDF_AMOUNT_RE.finditer(rest))
    if not matches:
        return (None, None)
    m = matches[-1]
    amount = _moze_pnum(m.group("amount"))
    if m.group("negative"):
        amount = -amount
    return (amount, m.start())


def _clean_pdf_card_description(desc: str) -> str:
    desc = re.sub(r"\s*USD\s*[\d\.]+,\d{2}\s*$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"^\d{6}\s+[*K]?\s*", "", desc)
    desc = re.sub(r"\s+C\.\d{2}/\d{2}\s*$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+", " ", desc)
    return desc.strip(" -")


def _parse_credit_card_pdf_text(
    text: str,
    source_file: str,
    source_mtime: float | None = None,
) -> dict | None:
    """Parsea un resumen VISA Santander renderizado con `pdftotext -layout`.

    Esta variante PDF es la que vive en `Finances/VISA/Resumen de tarjeta
    de crédito VISA-*.pdf`. El banco imprime los importes ARS y USD en dos
    columnas; en líneas USD aparece además el monto original junto a la
    descripción, por eso tomamos siempre el último importe de la línea.
    """
    if not text.strip():
        return None

    brand = "Visa" if "VISA" in (source_file + "\n" + text[:1000]).upper() else None
    closing_date, due_date, next_closing_date, next_due_date = _parse_pdf_card_statement_dates(text)

    total_consumos_ars = None
    total_consumos_usd = None
    holder = None
    last4 = None
    total_re = re.compile(
        r"Tarjeta\s+(?P<last4>\d{4})\s+Total\s+Consumos\s+de\s+"
        r"(?P<holder>.+?)\s+"
        r"(?P<ars>\d[\d\.]*,\d{2})\s*\*"
        r"(?:\s+(?P<usd>\d[\d\.]*,\d{2})\s*\*)?",
        re.IGNORECASE,
    )
    m_total = total_re.search(text)
    if m_total:
        last4 = m_total.group("last4")
        holder = re.sub(r"\s+", " ", m_total.group("holder")).strip() or None
        total_consumos_ars = _moze_pnum(m_total.group("ars"))
        if m_total.group("usd"):
            total_consumos_usd = _moze_pnum(m_total.group("usd"))
    elif m := re.search(r"Tarjeta\s+(\d{4})", text, re.IGNORECASE):
        last4 = m.group(1)

    total_ars = total_consumos_ars
    total_usd = total_consumos_usd
    saldo_re = re.compile(
        r"SALDO\s+ACTUAL.*?\$\s*(?P<ars>\d[\d\.]*,\d{2})"
        r".*?U\$S\s*(?P<usd>\d[\d\.]*,\d{2})",
        re.IGNORECASE,
    )
    for m_saldo in saldo_re.finditer(text):
        total_ars = _moze_pnum(m_saldo.group("ars"))
        total_usd = _moze_pnum(m_saldo.group("usd"))

    minimum_ars = None
    m_min = re.search(
        r"PAGO\s+MINIMO.*?\$\s*(?P<amount>\d[\d\.]*,\d{2})",
        text,
        re.IGNORECASE,
    )
    if m_min:
        minimum_ars = _moze_pnum(m_min.group("amount"))

    purchases: list[dict] = []
    other_charges: list[dict] = []
    in_movements = False
    seen_total = False
    current_year: int | None = None
    current_month: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        upper = _strip_accents(line).upper()
        if "FECHA" in upper and "COMPROBANTE" in upper and "REFERENCIA" in upper:
            if not seen_total:
                in_movements = True
            continue
        if "TARJETA" in upper and "TOTAL CONSUMOS" in upper:
            seen_total = True
            in_movements = False
            continue

        if not in_movements and not seen_total:
            continue

        date_iso, rest, current_year, current_month = _parse_pdf_card_line_prefix(
            line, current_year, current_month
        )
        if not date_iso or not rest:
            continue
        amount, amount_pos = _parse_pdf_card_amount_from_rest(rest)
        if amount is None or amount_pos is None:
            continue

        if any(k in upper for k in _CARD_PDF_SKIP_KEYWORDS):
            continue

        is_tax = any(k in upper for k in _CARD_PDF_TAX_KEYWORDS)
        desc = _clean_pdf_card_description(rest[:amount_pos])
        if not desc:
            continue

        if is_tax:
            if amount > 0:
                other_charges.append({
                    "description": desc,
                    "amount": amount,
                    "currency": "ARS",
                })
            continue

        currency = "USD" if "USD" in upper or "U$S" in upper else "ARS"
        purchases.append({
            "date": date_iso,
            "description": desc,
            "amount": amount,
            "currency": currency,
        })

    if (
        total_ars is None and total_usd is None
        and not closing_date and not due_date
        and not purchases
    ):
        return None

    def _purchase_sort_key(p: dict) -> tuple[bool, float]:
        return (p.get("amount", 0) > 0, abs(float(p.get("amount") or 0)))

    all_ars_purchases = sorted(
        (p for p in purchases if p["currency"] == "ARS"),
        key=_purchase_sort_key,
        reverse=True,
    )
    all_usd_purchases = sorted(
        (p for p in purchases if p["currency"] == "USD"),
        key=_purchase_sort_key,
        reverse=True,
    )

    return {
        "brand": brand,
        "last4": last4,
        "holder": holder,
        "closing_date": closing_date,
        "due_date": due_date,
        "next_closing_date": next_closing_date,
        "next_due_date": next_due_date,
        "total_ars": total_ars,
        "total_usd": total_usd,
        "minimum_ars": minimum_ars,
        "minimum_usd": None,
        "top_purchases_ars": all_ars_purchases[:5],
        "top_purchases_usd": all_usd_purchases[:3],
        "all_purchases_ars": all_ars_purchases,
        "all_purchases_usd": all_usd_purchases,
        "other_charges": other_charges,
        "other_charges_total_ars": (
            sum(c["amount"] for c in other_charges) if other_charges else 0
        ),
        "source_file": source_file,
        "source_mtime": (
            datetime.fromtimestamp(source_mtime).isoformat(timespec="seconds")
            if source_mtime
            else None
        ),
    }


def _parse_credit_card_pdf(path: Path) -> dict | None:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    try:
        res = subprocess.run(
            [pdftotext, "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if res.returncode != 0:
        return None
    return _parse_credit_card_pdf_text(res.stdout or "", path.name, mtime)


def _load_credit_cards(finance_dir: Path) -> list[dict]:
    """Carga resúmenes de tarjeta xlsx y PDF del subdirectorio `VISA/`.

    Los xlsx siguen usando ``web.server._parse_credit_card_xlsx``; los PDF
    Santander se parsean acá porque el dashboard tiene que leer el archivo
    histórico real (`Resumen de tarjeta de crédito VISA-*.pdf`).

    Import lazy de ``web.server`` para evitar dependencia circular en
    import-time (el dashboard lo importa el server después de definir
    sus parsers).
    """
    try:
        from web.server import _parse_credit_card_xlsx  # type: ignore
    except ImportError:
        _parse_credit_card_xlsx = None  # type: ignore

    xlsx_seen: set[Path] = set()
    for pattern in ("VISA/Último resumen*.xlsx", "VISA/Ultimo resumen*.xlsx"):
        for p in finance_dir.glob(pattern):
            xlsx_seen.add(p)
    pdfs = sorted(finance_dir.glob("VISA/*.pdf"), key=lambda p: p.name)

    cards: list[dict] = []
    if _parse_credit_card_xlsx:
        for p in sorted(xlsx_seen, key=lambda p: p.name):
            parsed = _parse_credit_card_xlsx(p)
            if parsed:
                cards.append(parsed)
    for p in pdfs:
        parsed = _parse_credit_card_pdf(p)
        if parsed:
            cards.append(parsed)
    # En `/finance` mostramos histórico de PDFs; lo más reciente primero.
    cards.sort(key=lambda c: c.get("due_date") or c.get("closing_date") or "", reverse=True)
    return cards


# ── Agregaciones para el dashboard ───────────────────────────────────────


def _ym(d: str) -> str:
    """`YYYY-MM-DD` → `YYYY-MM`. El frontend usa esto como bucket mensual."""
    return d[:7] if len(d) >= 7 else d


def _expense_amount(t: dict) -> float:
    """Monto a sumar para un gasto.

    MOZE históricamente trae gastos con signo variable y los agregadores
    usan `abs`. Los PDFs de tarjeta pueden traer créditos/reversos con
    monto negativo; esos movimientos llevan `amount_is_signed=True` para
    que resten del gasto neto en vez de inflarlo.
    """
    amount = float(t.get("amount") or 0.0)
    if t.get("amount_is_signed"):
        return amount
    return abs(amount)


def _kpi_sparks(txs: list[dict], now: datetime, n_months: int = 6, incomes: list[dict] | None = None) -> dict:
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
                expenses_ars[ym] += _expense_amount(t)
            elif cb == "USD":
                expenses_usd[ym] += _expense_amount(t)
        elif t["type"] == "income":
            txs_count[ym] += 1
            if cb == "ARS":
                income_ars[ym] += abs(t["amount"])
            elif cb == "USD":
                income_usd[ym] += abs(t["amount"])
    for i in (incomes or []):
        ym = _ym(i["date"])
        if ym not in label_set:
            continue
        if i["currency"] == "ARS":
            income_ars[ym] += abs(i["amount"])
        elif i["currency"] == "USD":
            income_usd[ym] += abs(i["amount"])
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


def _kpis(txs: list[dict], now: datetime, incomes: list[dict] | None = None) -> dict:
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
    usd_expenses_by_month: dict[str, float] = defaultdict(float)
    usd_counts_by_month: dict[str, int] = defaultdict(int)
    by_cat_this: Counter = Counter()
    n_this = 0
    n_prev = 0

    for t in txs:
        ym = _ym(t["date"])
        bucket = t["currency_bucket"]
        if t["type"] == "expense":
            expense_amount = _expense_amount(t)
            if bucket == "USD":
                usd_expenses_by_month[ym] += expense_amount
                usd_counts_by_month[ym] += 1
            if ym == this_ym:
                expenses_this[bucket] += expense_amount
                if bucket == "ARS":
                    by_cat_this[t["category"]] += expense_amount
                n_this += 1
            elif ym == prev_ym:
                expenses_prev[bucket] += _expense_amount(t)
                n_prev += 1
        elif t["type"] == "income":
            if ym == this_ym:
                income_this[bucket] += abs(t["amount"])
            elif ym == prev_ym:
                income_prev[bucket] += abs(t["amount"])

    for i in (incomes or []):
        ym = _ym(i["date"])
        bucket = i["currency"]
        if ym == this_ym:
            income_this[bucket] += abs(i["amount"])
        elif ym == prev_ym:
            income_prev[bucket] += abs(i["amount"])

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
    usd_period = this_ym
    usd_n_samples = usd_counts_by_month.get(this_ym, 0)
    usd_fallback_to_latest = False
    if not usd_this:
        nonzero_usd_months = [
            ym for ym, amount in usd_expenses_by_month.items()
            if abs(amount) > 0.000001
        ]
        if nonzero_usd_months:
            usd_period = max(nonzero_usd_months)
            usd_this = usd_expenses_by_month[usd_period]
            usd_n_samples = usd_counts_by_month.get(usd_period, 0)
            usd_prev = 0.0
            usd_fallback_to_latest = usd_period != this_ym

    sparks = _kpi_sparks(txs, now, n_months=6, incomes=incomes)

    expenses_ars = _kpi(ars_this, _delta(ars_this, ars_prev), n_this)
    expenses_ars["spark"] = sparks["expenses_ars"]
    expenses_usd = _kpi(usd_this, _delta(usd_this, usd_prev), usd_n_samples)
    expenses_usd["spark"] = sparks["expenses_usd"]
    expenses_usd["period"] = usd_period
    expenses_usd["period_label"] = usd_period
    expenses_usd["fallback_to_latest"] = usd_fallback_to_latest
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


def _by_month(txs: list[dict], months: int = 12, incomes: list[dict] | None = None) -> dict:
    """Serie temporal: gastos vs ingresos por mes (últimos N meses), por
    bucket de moneda. Los ingresos se toman SOLO de los PDF de recibos de
    sueldo (fuente de verdad), NO de MOZE (desactualizado). Devuelve:

        {
            "labels": ["2025-05", "2025-06", ..., "2026-04"],
            "expenses_ars": [...],
            "expenses_usd": [...],
            "income_ars": [...],
            "income_usd": [...],
        }
    """
    incomes = incomes or []
    if not txs and not incomes:
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
    # Solo gastos de MOZE (NO ingresos de MOZE, usamos PDF como fuente de verdad)
    for t in txs:
        ym = _ym(t["date"])
        cb = t["currency_bucket"]
        if t["type"] == "expense":
            key = "expenses_ars" if cb == "ARS" else ("expenses_usd" if cb == "USD" else None)
        elif t["type"] == "income":
            # Ignorar ingresos de MOZE (desactualizados)
            continue
        else:
            key = None
        if key:
            buckets[ym][key] += _expense_amount(t)
    # Agregar ingresos SOLO de PDF de recibos de sueldo (fuente de verdad)
    for i in incomes:
        ym = _ym(i["date"])
        key = "income_ars" if i["currency"] == "ARS" else ("income_usd" if i["currency"] == "USD" else None)
        if key:
            buckets[ym][key] += i["amount"]
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
        amount = _expense_amount(t)
        by_cat[t["category"]] += amount
        by_cat_subs[t["category"]][t["subcategory"]] += amount
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
        by_store[key] += _expense_amount(t)
        counts[key] += 1
    items = [
        {"name": name, "amount": amount, "count": counts[name]}
        for name, amount in by_store.most_common(n)
    ]
    return {"items": items, "currency": currency, "window_days": window_days}


def _has_expenses_in_window(txs: list[dict], window_days: int, now: datetime, currency: str) -> bool:
    cutoff = (now - timedelta(days=window_days)).date().isoformat()
    for t in txs:
        if t.get("type") != "expense":
            continue
        if t.get("currency_bucket") != currency:
            continue
        if t.get("date", "") < cutoff:
            continue
        if abs(_expense_amount(t)) > 0.000001:
            return True
    return False


def _has_expenses(txs: list[dict], currency: str) -> bool:
    for t in txs:
        if t.get("type") != "expense":
            continue
        if t.get("currency_bucket") != currency:
            continue
        if abs(_expense_amount(t)) > 0.000001:
            return True
    return False


def _by_account(txs: list[dict]) -> dict:
    """Resumen por cuenta (ARS Cash, Santander ARS, Santander USD, IOL,
    USD Cash, etc.). Solo gastos + ingresos (skip Balance Adjustment
    porque distorsiona — son ajustes contables, no flow de plata)."""
    by_acc_ex: Counter = Counter()
    by_acc_in: Counter = Counter()
    counts: Counter = Counter()
    for t in txs:
        if t["type"] == "expense":
            by_acc_ex[t["account"]] += _expense_amount(t)
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


def _normalize_transfer_to_transaction(t: dict) -> dict:
    """Normaliza una transferencia de PDF al formato de transaction."""
    return {
        "date": t["date"],
        "time": "",
        "type": t["type"],
        "category": "Transferencia",
        "subcategory": "",
        "name": t["recipient"],
        "store": t["recipient"],
        "account": t["account"],
        "currency": t["currency"],
        "amount": t["amount"],
        "note": f"PDF: {t['source_file']}",
    }


def _normalize_card_purchase_to_transaction(p: dict, card_info: dict) -> dict:
    """Normaliza un consumo de tarjeta de crédito al formato de transaction."""
    currency = (p.get("currency") or "").strip().upper()
    return {
        "date": p["date"],
        "time": "",
        "type": "expense",
        "category": "Tarjeta de crédito",
        "subcategory": f"{card_info.get('brand', '')} {card_info.get('last4', '')}",
        "name": p["description"],
        "store": p["description"],
        "account": f"Tarjeta {card_info.get('brand', '')} {card_info.get('last4', '')}",
        "currency": currency,
        "currency_bucket": _CURRENCY_BUCKET.get(currency, currency or "ARS"),
        "amount": p["amount"],
        "amount_is_signed": True,
        "source": "credit_card",
        "source_file": card_info.get("source_file", ""),
        "note": f"Resumen: {card_info.get('source_file', '')}",
    }


def _card_purchases_for_dashboard(cards: list[dict], currency: str = "USD") -> list[dict]:
    """Movimientos de tarjeta que entran a agregados del dashboard.

    Solo se suman USD para cubrir el hueco real observado en `/finance`:
    los gastos ARS ya suelen estar en MOZE y sumarlos acá duplicaría el
    total. Los ARS de tarjeta siguen disponibles en `cards` y `recent`.
    """
    out: list[dict] = []
    wanted = currency.upper()
    key = "all_purchases_usd" if wanted == "USD" else "all_purchases_ars"
    for card in cards:
        card_info = {
            "brand": card.get("brand", ""),
            "last4": card.get("last4", ""),
            "source_file": card.get("source_file", ""),
        }
        for p in card.get(key, []):
            tx = _normalize_card_purchase_to_transaction(p, card_info)
            if tx["currency_bucket"] == wanted:
                out.append(tx)
    out.sort(key=lambda t: (t["date"] or "", t["time"] or ""), reverse=True)
    return out


def _all_recent_transactions(
    transactions: list[dict],
    transfers: list[dict],
    cards: list[dict],
    n: int = 50,
) -> list[dict]:
    """Combina MOZE transactions, PDF transfers y credit cards en una sola lista
    de movimientos recientes ordenados por fecha descendente."""
    all_movements: list[dict] = []

    # MOZE transactions
    for t in transactions:
        if t["type"] in ("balance_adjustment", "other"):
            continue
        all_movements.append({
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

    # PDF transfers
    for t in transfers:
        all_movements.append(_normalize_transfer_to_transaction(t))

    # Credit card purchases
    for card in cards:
        card_info = {
            "brand": card.get("brand", ""),
            "last4": card.get("last4", ""),
            "source_file": card.get("source_file", ""),
        }
        for p in card.get("all_purchases_ars", []):
            all_movements.append(_normalize_card_purchase_to_transaction(p, card_info))
        for p in card.get("all_purchases_usd", []):
            all_movements.append(_normalize_card_purchase_to_transaction(p, card_info))

    # Ordenar por fecha descendente (date + time)
    all_movements.sort(
        key=lambda x: (x["date"] or "", x["time"] or ""),
        reverse=True,
    )

    return all_movements[:n]


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
        finance_dir: Override del dir de finanzas (para tests). Default:
            ``_FINANCE_BACKUP_DIR`` de web.server (env `OBSIDIAN_RAG_FINANCE_DIR`).
            Se espera la estructura:
            - `VISA/Último resumen*.xlsx` / `VISA/Ultimo resumen*.xlsx` (tarjetas)
            - `Debito/*.pdf` (transferencias bancarias)
            - `Ingresos/` (recibos de sueldo - no usado aún en el dashboard)
        moze_dir: Override del dir de MOZE CSV (para tests). Default:
            ``_MOZE_BACKUP_DIR`` de web.server (env `OBSIDIAN_RAG_MOZE_DIR`).
        now: Override de "now" (para tests). Default: `datetime.now()`.
        months: Cuántos meses de serie temporal devolver (default 12).
        window_days: Ventana para los KPI por categoría / top stores
            (default 30 días).

    Returns:
        Dict con shape estable. Nunca lanza — silent-fail per
        convención web/. Si NINGUNO de los dos dirs tiene datos, fallback a
        payload vacío.
    """
    now = now or datetime.now()
    # `using_defaults` distingue invocaciones de producción (sin args, hay
    # que mirar el cache global de Tally4) de tests que pasan tmpdirs y
    # necesitan aislamiento total.
    using_defaults = finance_dir is None and moze_dir is None

    # Resolución de paths:
    # - Ambos None → defaults de web.server (los reales del usuario).
    # - Solo `finance_dir` pasado → asumir mismo dir para MOZE (back-compat
    #   con tests pre-split y con setups donde ambas fuentes vivían juntas).
    # - Solo `moze_dir` pasado → simétrico (poco común pero defensivo).
    if using_defaults:
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

    # Tally4 backup → CSV cache (silent-fail si node/realm-js no disponible).
    # Solo en producción (using_defaults) — los tests aislados no tocan
    # iCloud ni el cache global del usuario.
    _MOZE_CACHE: Path | None = None
    if using_defaults:
        try:
            from rag.integrations import tally4_realm
            tally4_realm.ensure_moze_csv(moze_dir)
            _MOZE_CACHE = tally4_realm.CACHE_DIR
        except Exception:
            _MOZE_CACHE = None

    # Cache key: tupla con (str(path), mtime) de cada archivo CSV/PDF/XLSX
    # de ambos dirs + el cache de Tally4. Re-export de cualquier archivo
    # invalida; agregar/quitar archivos también.
    try:
        moze_files = list(moze_dir.glob("MOZE_*.csv")) if moze_dir.exists() else []
        if _MOZE_CACHE and _MOZE_CACHE.exists() and _MOZE_CACHE != moze_dir:
            moze_files.extend(_MOZE_CACHE.glob("MOZE_*.csv"))
        finance_files = (
            list(finance_dir.glob("VISA/Último resumen*.xlsx"))
            + list(finance_dir.glob("VISA/Ultimo resumen*.xlsx"))
            + list(finance_dir.glob("VISA/*.pdf"))
            + list(finance_dir.glob("Debito/*.pdf"))
            + list(finance_dir.glob("Ingresos/*.pdf"))
        ) if finance_dir.exists() else []
        all_files = moze_files + finance_files
        cache_key = tuple(sorted((str(p), p.stat().st_mtime) for p in all_files))
        # Incluir window_days/months en la key porque distintas vistas
        # del mismo dataset cachean por separado.
        cache_key = (cache_key, months, window_days, now.strftime("%Y-%m"))
    except OSError:
        cache_key = None

    extra_moze: tuple[Path, ...] = (_MOZE_CACHE,) if _MOZE_CACHE else ()
    transactions, moze_sources = (
        _load_moze_rows(moze_dir, extra_dirs=extra_moze)
        if moze_dir.exists() or (_MOZE_CACHE and _MOZE_CACHE.exists())
        else ([], [])
    )
    transfers, pdf_sources = _load_pdf_transfers(finance_dir) if finance_dir.exists() else ([], [])
    cards = _load_credit_cards(finance_dir) if finance_dir.exists() else []
    incomes, income_sources = _load_income_pdfs(finance_dir) if finance_dir.exists() else ([], [])
    card_dashboard_transactions = _card_purchases_for_dashboard(cards, "USD")
    analytics_transactions = transactions + card_dashboard_transactions
    usd_window_days = window_days
    usd_window_fallback = False
    if (
        analytics_transactions
        and not _has_expenses_in_window(analytics_transactions, window_days, now, "USD")
        and _has_expenses(analytics_transactions, "USD")
    ):
        usd_window_days = max(window_days, months * 31)
        usd_window_fallback = True

    if cache_key is not None:
        with _DASHBOARD_CACHE_LOCK:
            # Si hay archivos de ingresos, siempre regenerar para asegurar frescura
            # (cambio reciente en la estructura)
            if incomes:
                pass  # No usar cache
            elif _DASHBOARD_CACHE.get("key") == cache_key:
                return _DASHBOARD_CACHE["payload"]

    if not transactions and not transfers and not cards and not incomes:
        return _empty_payload(now, "no_data", str(finance_dir), str(moze_dir))

    payload = {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "finance_dir": str(finance_dir),
            "moze_dir": str(moze_dir),
            "months": months,
            "window_days": window_days,
            "usd_window_days": usd_window_days,
            "usd_window_fallback": usd_window_fallback,
            "moze_sources": moze_sources,
            "pdf_sources": pdf_sources,
            "income_sources": income_sources,
            "card_files": [c.get("source_file") for c in cards],
            "n_transactions": len(transactions),
            "n_transfers": len(transfers),
            "n_cards": len(cards),
            "n_card_transactions": len(card_dashboard_transactions),
            "n_incomes": len(incomes),
        },
        "kpis": _kpis(analytics_transactions, now, incomes=incomes),
        "by_month": _by_month(analytics_transactions, months, incomes=incomes),
        "by_category_ars": _by_category(analytics_transactions, window_days, now, "ARS"),
        "by_category_usd": _by_category(analytics_transactions, usd_window_days, now, "USD"),
        "top_stores_ars": _top_stores(analytics_transactions, window_days, now, "ARS"),
        "top_stores_usd": _top_stores(analytics_transactions, usd_window_days, now, "USD"),
        "by_account": _by_account(analytics_transactions),
        "recent": _all_recent_transactions(transactions, transfers, cards, n=50),
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
            "window_days": 30,
            "usd_window_days": 30,
            "usd_window_fallback": False,
            "moze_sources": [],
            "pdf_sources": [],
            "income_sources": [],
            "card_files": [],
            "n_transactions": 0,
            "n_transfers": 0,
            "n_cards": 0,
            "n_card_transactions": 0,
            "n_incomes": 0,
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
