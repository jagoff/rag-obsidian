"""Smoke tests del backend de `/finance` (web/finance_dashboard.py).

Cubren los caminos críticos:

1. ``snapshot()`` con un finance_dir vacío devuelve shape vacío válido.
2. Parser MOZE: dedup por fingerprint funciona si hay 2 CSV con la misma row.
3. Parser PDF: regex de transferencias matchea la línea típica de Santander.
4. KPIs: cálculo de delta_pct y top_category con data de prueba.
5. by_month: alineación con los últimos N meses incluyendo months sin data.

NO toca el dir real del user (`~/Library/Mobile Documents/.../Finances`) —
todos los tests usan tmpdirs aislados.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from web import finance_dashboard as fd


def test_snapshot_empty_dir(tmp_path: Path) -> None:
    """Sin archivos en el dir → shape vacío válido (no crashea)."""
    payload = fd.snapshot(finance_dir=tmp_path, now=datetime(2026, 4, 29))
    assert payload["meta"]["reason"] == "no_data"
    assert payload["meta"]["n_transactions"] == 0
    assert payload["kpis"]["expenses_ars"]["insufficient"] is True
    assert payload["by_month"]["labels"] == []
    assert payload["recent"] == []
    assert payload["cards"] == []


def test_snapshot_missing_dir(tmp_path: Path) -> None:
    """Dir inexistente → shape `finance_dir_missing` sin crashear."""
    fake = tmp_path / "does_not_exist"
    payload = fd.snapshot(finance_dir=fake, now=datetime(2026, 4, 29))
    assert payload["meta"]["reason"] == "finance_dir_missing"
    assert payload["meta"]["n_transactions"] == 0


def _write_moze_csv(path: Path, rows: list[dict]) -> None:
    """Helper: escribe un CSV de MOZE con el header completo de 16 columnas."""
    header = "Account,Currency,Type,Main Category,Subcategory,Price,Fee,Bonus,Name,Store,Date,Time,Project,Note,Tags,Target"
    cols = ["Account", "Currency", "Type", "Main Category", "Subcategory", "Price",
            "Fee", "Bonus", "Name", "Store", "Date", "Time", "Project", "Note",
            "Tags", "Target"]
    lines = [header]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_moze_dedup_intra_csv(tmp_path: Path) -> None:
    """2 CSV con la misma fila → dedup deja 1. La FECHA del export más
    nuevo es la que importa para el orden, pero la dedup es por fingerprint."""
    row = {
        "Account": "Santander ARS",
        "Currency": "ARS",
        "Type": "Expense",
        "Main Category": "Consumibles",
        "Subcategory": "Ice cream",
        "Price": "-15000",
        "Fee": "0",
        "Bonus": "0",
        "Name": "Helado",
        "Store": "Manalu",
        "Date": "04/26/2026",
        "Time": "19:04",
    }
    _write_moze_csv(tmp_path / "MOZE_20260420_120000.csv", [row])
    _write_moze_csv(tmp_path / "MOZE_20260426_191517.csv", [row])
    txs, sources = fd._load_moze_rows(tmp_path)
    assert len(txs) == 1, "dedup por fingerprint debería dejar solo 1"
    assert len(sources) == 2
    # El segundo archivo debería marcar la row como duplicada.
    second = next(s for s in sources if s["path"].endswith("191517.csv"))
    assert second["rows_dup"] == 1


def test_moze_distinct_rows_kept(tmp_path: Path) -> None:
    """Dos rows con `Time` distinto NO son duplicadas — MOZE permite 2
    gastos idénticos en el mismo día a horas distintas (ej. 2 ubers).
    """
    base = {
        "Account": "Santander ARS",
        "Currency": "ARS",
        "Type": "Expense",
        "Main Category": "Vehicles",
        "Subcategory": "Uber",
        "Price": "-2000",
        "Fee": "0",
        "Bonus": "0",
        "Name": "Uber",
        "Store": "Uber",
        "Date": "04/26/2026",
    }
    _write_moze_csv(tmp_path / "MOZE_20260426.csv", [
        {**base, "Time": "10:00"},
        {**base, "Time": "18:00"},
    ])
    txs, _ = fd._load_moze_rows(tmp_path)
    assert len(txs) == 2


def test_moze_pnum_decimal_es() -> None:
    """`2026,74` → 2026.74. Formato ES (decimal coma)."""
    assert fd._moze_pnum("2026,74") == pytest.approx(2026.74)
    assert fd._moze_pnum("1.234.567,89") == pytest.approx(1234567.89)
    assert fd._moze_pnum("3710512") == pytest.approx(3710512.0)
    assert fd._moze_pnum("-15000") == pytest.approx(-15000.0)
    assert fd._moze_pnum("") == 0.0


def test_pdf_transfer_regex_santander_line() -> None:
    """La línea típica de Santander matchea el regex y extrae los 5 fields."""
    line = "   23/04/2026          Transferencia         Maria Elisa Gadea                                156-237184/9                            $ 60.000,00"
    m = fd._PDF_TRANSFER_RE.match(line)
    assert m is not None
    assert m.group("date") == "23/04/2026"
    assert m.group("type").lower() == "transferencia"
    assert "Maria Elisa Gadea" in m.group("desc")
    assert m.group("account") == "156-237184/9"
    assert m.group("amount") == "60.000,00"


def test_kpis_with_synthetic_data() -> None:
    """KPIs computan delta correctamente y marcan top_category."""
    txs = [
        {"date": "2026-04-15", "type": "expense", "category": "House",
         "currency_bucket": "ARS", "amount": -100000.0, "time": "12:00"},
        {"date": "2026-04-20", "type": "expense", "category": "Food",
         "currency_bucket": "ARS", "amount": -50000.0, "time": "14:00"},
        {"date": "2026-03-15", "type": "expense", "category": "House",
         "currency_bucket": "ARS", "amount": -75000.0, "time": "12:00"},
        {"date": "2026-04-10", "type": "income", "category": "Salary",
         "currency_bucket": "ARS", "amount": 1000000.0, "time": "09:00"},
    ]
    kpis = fd._kpis(txs, datetime(2026, 4, 29))
    assert kpis["expenses_ars"]["value"] == pytest.approx(150000.0)
    # Delta vs 75000 = (150000-75000)/75000 = 1.0
    assert kpis["expenses_ars"]["delta_pct"] == pytest.approx(1.0)
    assert kpis["income_ars"]["value"] == pytest.approx(1000000.0)
    assert kpis["balance_ars"]["value"] == pytest.approx(850000.0)
    assert kpis["top_category"]["name"] == "House"
    assert kpis["top_category"]["value"] == pytest.approx(100000.0)


def test_by_month_includes_empty_months() -> None:
    """`_by_month` rellena meses sin data con 0 — el chart no muestra gaps."""
    txs = [
        {"date": "2026-04-15", "type": "expense",
         "currency_bucket": "ARS", "amount": -100000.0},
        {"date": "2026-01-15", "type": "expense",
         "currency_bucket": "ARS", "amount": -50000.0},
    ]
    out = fd._by_month(txs, months=6)
    # La serie debe terminar en abril 2026 con 6 elementos.
    assert len(out["labels"]) == 6
    assert out["labels"][-1] == "2026-04"
    # Labels: [2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04]
    # Index 0 = 2025-11 — sin gasto.
    # Index 2 = 2026-01 (50000 expense).
    # Index 5 = 2026-04 (100000 expense).
    assert out["labels"][2] == "2026-01"
    assert out["expenses_ars"][2] == pytest.approx(50000.0)
    assert out["expenses_ars"][5] == pytest.approx(100000.0)
    # Mes vacío entre las dos transacciones queda en 0.
    assert out["expenses_ars"][3] == 0.0


def test_top_stores_window_filter() -> None:
    """Solo cuenta gastos dentro de la ventana, ARS only por default."""
    txs = [
        {"date": "2026-04-25", "type": "expense", "currency_bucket": "ARS",
         "store": "Manalu", "name": "Helado", "amount": -15000.0},
        {"date": "2026-04-20", "type": "expense", "currency_bucket": "ARS",
         "store": "Manalu", "name": "Helado", "amount": -12000.0},
        {"date": "2026-01-01", "type": "expense", "currency_bucket": "ARS",
         "store": "Manalu", "name": "Helado", "amount": -99999.0},  # fuera
        {"date": "2026-04-25", "type": "expense", "currency_bucket": "USD",
         "store": "Apple", "name": "Sub", "amount": -10.0},  # USD, skip ARS
    ]
    out = fd._top_stores(txs, window_days=30, now=datetime(2026, 4, 29), currency="ARS")
    assert len(out["items"]) == 1
    assert out["items"][0]["name"] == "Manalu"
    assert out["items"][0]["amount"] == pytest.approx(27000.0)
    assert out["items"][0]["count"] == 2


def test_credit_card_pdf_text_extracts_usd_purchases() -> None:
    """Los resúmenes VISA PDF Santander aportan consumos USD reales."""
    text = """
          3004 SANTA FE                                                          CIERRE      31 Dic 25 VENCIMIENTO 09 Ene 26
                                                                                 Prox.Cierre: 29 Ene 26             Prox.Vto.: 06 Feb 26

Fecha            Comprobante Referencia                                                              $                      U$S
  25 Noviem. 27 278050   GOOGLE *Google O P1gunINz USD          9,99                                        9,99
              28 697517 K SUNO INC.        in1SYaHVEUSD         24,00                                       24,00
  25 Diciem. 14 252745 K VERCEL INC.                USD        20,00                                       20,00-
              20 651214 K MERPAGO*SUPERDELCA                                             210.917,40

Tarjeta 1059 Total Consumos de FERNANDO R FERRARI                                            1445.371,58 *              120,84 *
  25 Diciem. 31                 IMPUESTO DE SELLOS        $                                       1.515,18
                                                          SALDO ACTUAL                 $ 1515.123,45             U$S        120,84
                                                          PAGO MINIMO                  $             144.410,00
"""
    card = fd._parse_credit_card_pdf_text(
        text,
        "Resumen de tarjeta de crédito VISA-09-01-2026.pdf",
        source_mtime=0,
    )
    assert card is not None
    assert card["brand"] == "Visa"
    assert card["last4"] == "1059"
    assert card["holder"] == "FERNANDO R FERRARI"
    assert card["closing_date"] == "2025-12-31"
    assert card["due_date"] == "2026-01-09"
    assert card["next_closing_date"] == "2026-01-29"
    assert card["next_due_date"] == "2026-02-06"
    assert card["total_usd"] == pytest.approx(120.84)

    usd_by_desc = {p["description"]: p for p in card["all_purchases_usd"]}
    assert usd_by_desc["GOOGLE *Google O P1gunINz"]["date"] == "2025-11-27"
    assert usd_by_desc["GOOGLE *Google O P1gunINz"]["amount"] == pytest.approx(9.99)
    assert usd_by_desc["SUNO INC. in1SYaHVE"]["amount"] == pytest.approx(24.0)
    assert usd_by_desc["VERCEL INC."]["amount"] == pytest.approx(-20.0)
    assert card["other_charges_total_ars"] == pytest.approx(1515.18)


def test_snapshot_includes_credit_card_usd_in_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Los USD de tarjeta entran en KPIs/series; ARS de tarjeta no duplica MOZE."""
    (tmp_path / "VISA").mkdir()
    fd._DASHBOARD_CACHE["key"] = None
    fd._DASHBOARD_CACHE["payload"] = None

    card = {
        "brand": "Visa",
        "last4": "1059",
        "source_file": "Resumen de tarjeta de crédito VISA-08-04-2026.pdf",
        "all_purchases_ars": [
            {"date": "2026-04-10", "description": "MERPAGO", "amount": 200000.0, "currency": "ARS"},
        ],
        "all_purchases_usd": [
            {"date": "2026-04-10", "description": "APPLE.COM/BILL", "amount": 24.99, "currency": "USD"},
            {"date": "2026-04-14", "description": "VERCEL INC.", "amount": -5.0, "currency": "USD"},
        ],
    }
    monkeypatch.setattr(fd, "_load_moze_rows", lambda *a, **k: ([], []))
    monkeypatch.setattr(fd, "_load_pdf_transfers", lambda *a, **k: ([], []))
    monkeypatch.setattr(fd, "_load_income_pdfs", lambda *a, **k: ([], []))
    monkeypatch.setattr(fd, "_load_credit_cards", lambda *a, **k: [card])

    payload = fd.snapshot(
        finance_dir=tmp_path,
        moze_dir=tmp_path,
        now=datetime(2026, 5, 16),
        months=3,
    )

    assert payload["meta"]["n_transactions"] == 0
    assert payload["meta"]["n_card_transactions"] == 2
    assert payload["meta"]["usd_window_fallback"] is True
    assert payload["meta"]["usd_window_days"] == 93
    assert payload["kpis"]["expenses_usd"]["value"] == pytest.approx(19.99)
    assert payload["kpis"]["expenses_usd"]["period"] == "2026-04"
    assert payload["kpis"]["expenses_usd"]["fallback_to_latest"] is True
    assert payload["kpis"]["expenses_ars"]["value"] == 0
    assert payload["by_month"]["labels"][-1] == "2026-04"
    assert payload["by_month"]["expenses_usd"][-1] == pytest.approx(19.99)
    assert payload["by_month"]["expenses_ars"][-1] == 0
    assert payload["by_category_usd"]["items"][0]["amount"] == pytest.approx(19.99)
    assert payload["top_stores_usd"]["items"][0]["name"] == "APPLE.COM/BILL"
