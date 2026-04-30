"""Tests para `rag.integrations.pillow_sleep` — ingester del export de
Pillow (iOS sleep tracker) sincronizado por iCloud.

Surfaces cubiertas:
- `parse_pillow_dump(path)` — parser puro del formato Core Data dump.
  El test arma un fixture con 2 ZSLEEPSESSION rows (1 nap + 1 night) +
  alguna entity ruidosa (ZALARM/ZPILLOWUSER) y verifica que solo extraiga
  la night. Cubre: regex de entity headers que tolera dígitos
  (link tables Z_5SLEEPSESSION), parsing de timestamps Apple→unix,
  cálculo de `date` en zona horaria local del session, derivado de
  `sleep_total_s` desde stages.
- `ingest()` con archivo no existente → silent-fail, devuelve
  `{skipped: True, reason: "file_not_found_or_empty"}`.
- `ingest()` con archivo válido → upsert idempotente (correr 2 veces no
  duplica filas) + emisión de mood signals desde wakeup_mood.
- `record_self_report_mood()` — endpoint helper. Valida label allowlist
  + persiste a `rag_mood_signals` con source="manual".
- `last_night()` / `recent_nights()` / `weekly_stats()` — read-side
  helpers que consume el panel home.

Mocking strategy:
- `RAG_PILLOW_DUMP_PATH` env var hace override del path del dump,
  evitando depender del archivo real en iCloud durante tests.
- Telemetry DB usa el path por defecto del rag init pero el conftest
  global lo redirige a tmp_path para no contaminar la real.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag.integrations import pillow_sleep as ps


# Sample minimum dump: 1 night + 1 nap + algunos noise headers.
# Apple epoch (2001-01-01 UTC). Una sesión que arrancó a las 23:00 local
# (Argentina UTC-3) del 2026-04-29 → 02:00 UTC del 2026-04-30.
# 2026-04-30 00:00 UTC = 798336000 segundos desde 2001-01-01 UTC = 798336000.
# Probemos: 2001-01-01 a 2026-04-30 son 25 años + ~120 días aprox ~= 798M.
def _apple_ts(dt_utc: datetime) -> float:
    return dt_utc.timestamp() - datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()


def _build_dump(start_utc: datetime, end_utc: datetime) -> str:
    """Build a minimal Pillow dump fixture with one night + one nap +
    noise entities, in the canonical format (entity header on its own
    line, then one row per entity per line)."""
    s = _apple_ts(start_utc)
    e = _apple_ts(end_utc)
    nap_s = _apple_ts(end_utc.replace(hour=15, minute=0))
    nap_e = _apple_ts(end_utc.replace(hour=15, minute=20))
    return (
        "ZALARM\n"
        "ZPILLOWUSER\n"
        f"Z_PK -> 1Z_ENT -> 2ZAVERAGEBEDTIME -> 0ZUNIQUEIDENTIFIER -> PillowUser\n"
        "ZSLEEPSESSION\n"
        f"Z_PK -> 100Z_ENT -> 6Z_OPT -> 5"
        f"ZALARMENABLED -> 0ZALARMTYPERAW -> 0ZANALYSISALGORITHMRAW -> 1"
        f"ZAUDIORECORDINGENABLED -> 0ZAUTOMATICSESSION -> 1"
        f"ZGROSSMOTIONSINSESSION -> 0ZISEDITED -> 0ZISNAP -> 0"
        f"ZNAPTYPERAW -> -1ZNUMBEROFAWAKENINGS -> 2ZNUMBEROFSNOOZES -> 0"
        f"ZPHYSICALACTIVITYORIGIN -> 0ZPRODUCEDBYAPPLEWATCH -> 0"
        f"ZSLEEPAIDENABLED -> 0ZSYNCEDTORUNKEEPER -> 0ZUSEDAPPLEWATCH -> 1"
        f"ZWAKEUPMOOD -> 3ZDURATION -> 0.0ZENDTIME -> {e}"
        f"ZFATIGUE -> 0.2ZSLEEPQUALITY -> 0.85ZSMARTWAKEUPDURATION -> 0.0"
        f"ZSTARTTIME -> {s}ZTIMEAWAKE -> 600.0ZTIMEAWAKEUNTILSTOPPING -> 0.0"
        f"ZTIMEINDEEPSLEEP -> 5400.0ZTIMEINLIGHTSLEEP -> 14400.0"
        f"ZTIMEINREMSLEEP -> 7200.0ZTIMETOSLEEP -> 0.0"
        f"ZDEVICEUSED -> iPhone17,2ZMORPHEUSVERSIONUSED -> Auto-D_v12"
        f"ZSOURCEID -> 25211383ZTIMEZONEIDENTIFIER -> America/Argentina/Cordoba"
        f"ZUNIQUEIDENTIFIER -> AAAAAAAA-1111-2222-3333-444444444444"
        f"ZSLEEPTRACKINGMETHODRAW -> 0\n"
        f"Z_PK -> 101Z_ENT -> 6Z_OPT -> 1"
        f"ZALARMENABLED -> 0ZALARMTYPERAW -> 0ZANALYSISALGORITHMRAW -> 0"
        f"ZAUDIORECORDINGENABLED -> 0ZAUTOMATICSESSION -> 1"
        f"ZGROSSMOTIONSINSESSION -> 0ZISEDITED -> 0ZISNAP -> 1"
        f"ZNAPTYPERAW -> 0ZNUMBEROFAWAKENINGS -> 0ZNUMBEROFSNOOZES -> 0"
        f"ZPHYSICALACTIVITYORIGIN -> 0ZPRODUCEDBYAPPLEWATCH -> 0"
        f"ZSLEEPAIDENABLED -> 0ZSYNCEDTORUNKEEPER -> 0ZUSEDAPPLEWATCH -> 0"
        f"ZWAKEUPMOOD -> 0ZDURATION -> 0.0ZENDTIME -> {nap_e}"
        f"ZFATIGUE -> 0.0ZSLEEPQUALITY -> 0.6ZSMARTWAKEUPDURATION -> 0.0"
        f"ZSTARTTIME -> {nap_s}ZTIMEAWAKE -> 0.0ZTIMEAWAKEUNTILSTOPPING -> 0.0"
        f"ZTIMEINDEEPSLEEP -> 0.0ZTIMEINLIGHTSLEEP -> 1200.0"
        f"ZTIMEINREMSLEEP -> 0.0ZTIMETOSLEEP -> 0.0"
        f"ZDEVICEUSED -> iPhone17,2ZMORPHEUSVERSIONUSED -> "
        f"ZSOURCEID -> 25211383ZTIMEZONEIDENTIFIER -> America/Argentina/Cordoba"
        f"ZUNIQUEIDENTIFIER -> BBBBBBBB-5555-6666-7777-888888888888"
        f"ZSLEEPTRACKINGMETHODRAW -> 0\n"
        "Z_5SLEEPSESSION\n"
        "Z_5SLEEPNOTE -> 1Z_6SLEEPSESSION -> 100\n"
    )


@pytest.fixture
def dump_path(tmp_path, monkeypatch):
    """Write a minimal pillow dump to tmp + redirect the integration to it."""
    p = tmp_path / "PillowData.txt"
    # Night: 2026-04-29 23:00 ART (UTC-3) = 2026-04-30 02:00 UTC
    # End:   2026-04-30 07:00 ART = 2026-04-30 10:00 UTC
    start = datetime(2026, 4, 30, 2, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 30, 10, 0, tzinfo=timezone.utc)
    p.write_text(_build_dump(start, end), encoding="utf-8")
    monkeypatch.setenv("RAG_PILLOW_DUMP_PATH", str(p))
    return p


def test_parse_pillow_dump_extracts_night_skips_nap(dump_path):
    sessions = ps.parse_pillow_dump()  # uses env override
    # Default: include_naps=False. La nap (uuid BBBB...) debe estar excluida.
    assert len(sessions) == 1
    s = sessions[0]
    assert s.uuid == "AAAAAAAA-1111-2222-3333-444444444444"
    assert s.pk == 100
    assert s.is_nap is False
    assert s.quality == pytest.approx(0.85, abs=1e-3)
    assert s.wakeup_mood == 3
    assert s.fatigue == pytest.approx(0.2, abs=1e-3)
    assert s.awakenings == 2
    assert s.deep_s == 5400.0
    assert s.light_s == 14400.0
    assert s.rem_s == 7200.0
    # date = local-tz YYYY-MM-DD del end. Argentina UTC-3, end = 10:00 UTC →
    # 07:00 ART → 2026-04-30. La sesión la asociamos al día en que se despertó.
    assert s.date == "2026-04-30"
    assert s.tz == "America/Argentina/Cordoba"
    # sleep_total_s = sum stages
    assert s.sleep_total_s == 27000.0


def test_parse_pillow_dump_include_naps(dump_path):
    sessions = ps.parse_pillow_dump(include_naps=True)
    assert len(sessions) == 2
    naps = [s for s in sessions if s.is_nap]
    assert len(naps) == 1
    assert naps[0].uuid == "BBBBBBBB-5555-6666-7777-888888888888"


def test_ingest_silent_fail_when_file_missing(tmp_path, monkeypatch):
    nonexistent = tmp_path / "no-such-pillow-dump.txt"
    monkeypatch.setenv("RAG_PILLOW_DUMP_PATH", str(nonexistent))
    result = ps.ingest()
    assert result["skipped"] is True
    assert result["reason"] == "file_not_found_or_empty"
    assert result["total_parsed"] == 0


def test_ingest_upsert_idempotent_and_emits_mood(dump_path, _isolated_telemetry):
    # First ingest.
    r1 = ps.ingest()
    assert r1["ingested"] == 1
    assert r1["mood_signals"] == 1  # wakeup_mood=3 → genera 1 signal
    # Second ingest — UPSERT no debería duplicar la fila.
    r2 = ps.ingest()
    assert r2["ingested"] == 1
    # En SQLite verificar count = 1
    from rag import _ragvec_state_conn
    with _ragvec_state_conn() as conn:
        n = conn.execute("SELECT count(*) FROM rag_sleep_sessions").fetchone()[0]
        assert n == 1
        # Mood signal idempotente: el ingester limpia antes de re-emitir.
        m = conn.execute(
            "SELECT count(*) FROM rag_mood_signals "
            " WHERE source='pillow' AND signal_kind='wakeup_mood'"
        ).fetchone()[0]
        assert m == 1
        # Fatigue signal también — wakeup_mood=3 + fatigue=0.2 → ambos.
        f = conn.execute(
            "SELECT count(*) FROM rag_mood_signals "
            " WHERE source='pillow' AND signal_kind='fatigue'"
        ).fetchone()[0]
        assert f == 1


def test_record_self_report_mood_validates_label(_isolated_telemetry):
    bad = ps.record_self_report_mood("excellent")
    assert bad["ok"] is False
    assert "good" in bad["error"]


def test_record_self_report_mood_persists(_isolated_telemetry):
    r = ps.record_self_report_mood("good", notes="testing")
    assert r["ok"] is True
    assert r["value"] == 1.0
    assert r["label"] == "good"
    from rag import _ragvec_state_conn
    with _ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT value, evidence FROM rag_mood_signals "
            " WHERE source='manual' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        assert row[0] == 1.0
        import json as _json
        ev = _json.loads(row[1])
        assert ev["label"] == "good"
        assert ev["notes"] == "testing"


def test_last_night_and_recent_nights(dump_path, _isolated_telemetry):
    ps.ingest()
    ln = ps.last_night()
    assert ln is not None
    assert ln["uuid"] == "AAAAAAAA-1111-2222-3333-444444444444"
    assert ln["sleep_total_h"] == pytest.approx(7.5)
    assert ln["deep_pct"] == pytest.approx(20.0, abs=1e-1)
    assert ln["bedtime_local"]  # truthy clock string
    assert ln["waketime_local"]

    nights = ps.recent_nights(limit=10)
    assert len(nights) == 1
    assert nights[0]["uuid"] == ln["uuid"]


def test_weekly_stats_smoke(dump_path, _isolated_telemetry):
    ps.ingest()
    ws = ps.weekly_stats()
    assert ws["week"]["n"] == 1
    assert ws["hist"]["n"] == 1
    # Sparkline 7d: 7 slots, todos None excepto el día del fixture.
    assert len(ws["spark_quality_7d"]) == 7
    not_none = [x for x in ws["spark_quality_7d"] if x is not None]
    assert len(not_none) == 1
    assert not_none[0] == pytest.approx(0.85, abs=1e-2)


def test_pearson_returns_zero_with_few_pairs(_isolated_telemetry):
    r, n = ps._pearson([1.0, 2.0], [1.0, 2.0])
    assert r == 0.0
    assert n == 2


def test_pearson_perfect_positive_correlation(_isolated_telemetry):
    r, n = ps._pearson([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert r == pytest.approx(1.0)
    assert n == 5


def test_pearson_skips_none_pairs(_isolated_telemetry):
    r, n = ps._pearson([1.0, None, 2.0, 3.0], [1.0, 2.0, None, 3.0])
    # Only (1,1) and (3,3) are valid pairs → n=2 → returns (0.0, 2)
    assert n == 2
    assert r == 0.0


def test_detect_patterns_empty_when_too_few_nights(_isolated_telemetry):
    # Fresh DB → no rows → detect_patterns returns []
    findings = ps.detect_patterns()
    assert findings == []


def test_bedtime_normalized_handles_midnight_wrap():
    # 23:30 → 23.5 (no wrap needed, already > 12)
    # 01:30 → 25.5 (wraps to next day)
    # 14:00 → 14 (afternoon, > 12)
    # The TZ doesn't matter for the wrap logic itself.
    from datetime import datetime, timezone
    # Build a timestamp at exactly 23:30 UTC (no tz given → uses system tz,
    # but we test the math directly with utc to isolate from system TZ).
    ts_2330 = datetime(2026, 4, 30, 23, 30, tzinfo=timezone.utc).timestamp()
    ts_0130 = datetime(2026, 4, 30, 1, 30, tzinfo=timezone.utc).timestamp()
    h_2330 = ps._bedtime_normalized(ts_2330, "UTC")
    h_0130 = ps._bedtime_normalized(ts_0130, "UTC")
    assert h_2330 == pytest.approx(23.5)
    assert h_0130 == pytest.approx(25.5)


@pytest.fixture
def _isolated_telemetry(tmp_path, monkeypatch):
    """Each test gets its own telemetry.db so ingest UPSERTs / mood inserts
    don't cross-contaminate. We monkeypatch DB_PATH to tmp_path and let
    `_ensure_telemetry_tables` create the schema fresh on first use."""
    monkeypatch.setattr("rag.DB_PATH", tmp_path)
    # Force re-init de la tabla en el nuevo path.
    monkeypatch.setattr("rag._TELEMETRY_DDL_ENSURED_PATHS", set())
    yield tmp_path
