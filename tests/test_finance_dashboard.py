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
