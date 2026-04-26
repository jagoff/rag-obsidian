"""Tests para `scripts/drift_watcher.py`.

Verifican el contrato del watcher:
- Cold start (0 / 1 row con singles_n>=20) → no alert.
- Delta menor al threshold → no alert.
- Delta mayor al threshold (singles -7pp / chains -10pp) → alert + jsonl.
- Idempotencia: segunda corrida con mismos datos NO duplica alert.

Todo monkeypatchea ``DB_PATH`` / ``ALERTS_PATH`` a ``tmp_path``; el script
nunca toca la DB real ni el JSONL real del sistema. El push a WhatsApp se
desactiva con ``push_whatsapp=False`` para no abrir sockets en CI.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# scripts/ no es un package estándar — pero hay un __init__.py vacío y
# conftest.py inserta el repo root en sys.path. Importamos directo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drift_watcher  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_db(tmp_path: Path) -> Path:
    """Crea una telemetry.db vacía con el schema mínimo de rag_eval_runs."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE rag_eval_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts TEXT NOT NULL,"
        " singles_hit5 REAL,"
        " singles_mrr REAL,"
        " singles_n INTEGER,"
        " chains_hit5 REAL,"
        " chains_mrr REAL,"
        " chains_chain_success REAL,"
        " chains_turns INTEGER,"
        " chains_n INTEGER,"
        " extra_json TEXT"
        ")"
    )
    conn.commit()
    conn.close()
    return db


def _insert_run(
    db: Path, *, ts: str, singles_hit5: float, chains_hit5: float = 0.85,
    singles_n: int = 60, chains_n: int = 30,
) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO rag_eval_runs (ts, singles_hit5, singles_n, chains_hit5, chains_n) "
        "VALUES (?, ?, ?, ?, ?)",
        (ts, singles_hit5, singles_n, chains_hit5, chains_n),
    )
    conn.commit()
    conn.close()


# ── Tests: insufficient data ─────────────────────────────────────────────────


def test_zero_rows_exits_clean(tmp_path: Path, fake_db: Path, capsys):
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "insufficient data" in out
    assert not alerts.exists()


def test_one_row_exits_clean(tmp_path: Path, fake_db: Path, capsys):
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.7167)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "insufficient data" in out
    assert not alerts.exists()


def test_filters_runs_with_low_singles_n(tmp_path: Path, fake_db: Path, capsys):
    """Runs con singles_n < 20 (fixtures sintéticos) se ignoran. Si los
    dos últimos runs reales son fixtures, no hay data útil."""
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.5, singles_n=2)
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.9, singles_n=2)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "insufficient data" in out
    assert not alerts.exists()


# ── Tests: no drift ──────────────────────────────────────────────────────────


def test_small_drop_below_threshold_no_alert(tmp_path: Path, fake_db: Path, capsys):
    """singles -3pp y chains estables → no alert."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6867, chains_hit5=0.8667)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "delta_singles=-3.00pp" in out
    assert not alerts.exists()


def test_improvement_no_alert(tmp_path: Path, fake_db: Path, capsys):
    """Mejora positiva → OK con delta + signo positivo."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8000)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.7500, chains_hit5=0.9000)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "+3.33pp" in out
    assert "+10.00pp" in out
    assert not alerts.exists()


# ── Tests: drift detected ────────────────────────────────────────────────────


def test_singles_drop_7pp_writes_alert(tmp_path: Path, fake_db: Path, capsys):
    """singles -7pp (debajo del threshold de -5pp) → alert en jsonl."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6467, chains_hit5=0.8667)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRIFT detected on singles_hit5" in out
    assert "-7.00pp" in out
    assert alerts.is_file()

    lines = alerts.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "singles"
    assert rec["current_run_ts"] == "2026-04-25T03:30:00"
    assert rec["prev_run_ts"] == "2026-04-24T03:30:00"
    assert rec["delta"] == pytest.approx(-0.07, abs=1e-6)
    assert rec["prev"] == pytest.approx(0.7167, abs=1e-6)
    assert rec["current"] == pytest.approx(0.6467, abs=1e-6)


def test_chains_drop_10pp_writes_alert(tmp_path: Path, fake_db: Path, capsys):
    """chains -10pp (debajo del threshold de -7pp) → alert en jsonl,
    singles estable → no se duplica."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.7167, chains_hit5=0.7667)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRIFT detected on chains_hit5" in out
    assert "DRIFT detected on singles_hit5" not in out

    lines = alerts.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "chains"
    assert rec["delta"] == pytest.approx(-0.10, abs=1e-6)


def test_both_drop_writes_two_alerts(tmp_path: Path, fake_db: Path):
    """Si caen ambos hit5, escribimos un alert por kind."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6000, chains_hit5=0.7000)
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(fake_db, alerts, push_whatsapp=False)
    assert rc == 0

    lines = alerts.read_text().strip().splitlines()
    assert len(lines) == 2
    kinds = {json.loads(line)["kind"] for line in lines}
    assert kinds == {"singles", "chains"}


# ── Tests: idempotencia ──────────────────────────────────────────────────────


def test_idempotent_within_dedup_window(tmp_path: Path, fake_db: Path):
    """Dos runs consecutivas con la misma data → solo UN alert en el jsonl."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6467, chains_hit5=0.8667)
    alerts = tmp_path / "drift_alerts.jsonl"

    now1 = datetime(2026, 4, 25, 9, 0, 0)
    drift_watcher.evaluate(fake_db, alerts, now=now1, push_whatsapp=False)

    # Segunda corrida 6h después — sigue dentro de la ventana de 12h.
    now2 = now1 + timedelta(hours=6)
    drift_watcher.evaluate(fake_db, alerts, now=now2, push_whatsapp=False)

    lines = alerts.read_text().strip().splitlines()
    assert len(lines) == 1


def test_dedup_window_expires_after_12h(tmp_path: Path, fake_db: Path):
    """Pasada la ventana de dedup, un alert con el mismo run_ts se vuelve
    a escribir. (Edge case raro en prod — el cron ve un run_ts viejo
    porque no llegó eval gate nuevo — pero queremos que lo tratemos
    como señal recurrente, no silencio permanente.)"""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6467, chains_hit5=0.8667)
    alerts = tmp_path / "drift_alerts.jsonl"

    now1 = datetime(2026, 4, 25, 9, 0, 0)
    drift_watcher.evaluate(fake_db, alerts, now=now1, push_whatsapp=False)

    # 13h después (fuera de ventana) — debería re-escribir.
    now2 = now1 + timedelta(hours=13)
    drift_watcher.evaluate(fake_db, alerts, now=now2, push_whatsapp=False)

    lines = alerts.read_text().strip().splitlines()
    assert len(lines) == 2


def test_dedup_distinguishes_kinds(tmp_path: Path, fake_db: Path):
    """Un alert previo de `singles` no bloquea uno nuevo de `chains`."""
    _insert_run(fake_db, ts="2026-04-24T03:30:00", singles_hit5=0.7167, chains_hit5=0.8667)
    _insert_run(fake_db, ts="2026-04-25T03:30:00", singles_hit5=0.6467, chains_hit5=0.8667)
    alerts = tmp_path / "drift_alerts.jsonl"

    now1 = datetime(2026, 4, 25, 9, 0, 0)
    drift_watcher.evaluate(fake_db, alerts, now=now1, push_whatsapp=False)

    # Reemplazamos el segundo run con uno que tira chains también.
    conn = sqlite3.connect(fake_db)
    conn.execute(
        "UPDATE rag_eval_runs SET chains_hit5 = ? WHERE ts = ?",
        (0.7000, "2026-04-25T03:30:00"),
    )
    conn.commit()
    conn.close()

    # Misma run_ts pero ahora también drift de chains — debe emitir el
    # alert de chains (kind nuevo) sin duplicar el de singles.
    now2 = now1 + timedelta(hours=1)
    drift_watcher.evaluate(fake_db, alerts, now=now2, push_whatsapp=False)

    lines = alerts.read_text().strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds.count("singles") == 1
    assert kinds.count("chains") == 1


# ── Tests: resilience ────────────────────────────────────────────────────────


def test_missing_db_exits_clean(tmp_path: Path, capsys):
    """DB inexistente → stderr log + exit 0, sin tocar el jsonl."""
    db = tmp_path / "does_not_exist.db"
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(db, alerts, push_whatsapp=False)
    captured = capsys.readouterr()
    assert rc == 0
    assert "db missing" in captured.err
    assert not alerts.exists()


def test_corrupt_db_exits_clean(tmp_path: Path, capsys):
    """DB con bytes basura → sqlite3 levanta, manejamos y salimos limpio."""
    db = tmp_path / "telemetry.db"
    db.write_bytes(b"not a sqlite db, just garbage")
    alerts = tmp_path / "drift_alerts.jsonl"
    rc = drift_watcher.evaluate(db, alerts, push_whatsapp=False)
    captured = capsys.readouterr()
    assert rc == 0
    # El error puede salir como "db open failed" o "db query failed"
    # según cuándo SQLite detecte el corrupt header.
    assert "db" in captured.err and "failed" in captured.err
    assert not alerts.exists()


def test_main_never_raises(tmp_path: Path, monkeypatch, capsys):
    """`main()` debe atrapar todo y devolver 0 incluso si evaluate explota."""

    def boom(*_a, **_kw):
        raise RuntimeError("synthetic catastrophe")

    monkeypatch.setattr(drift_watcher, "evaluate", boom)
    rc = drift_watcher.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "uncaught exception" in captured.err
