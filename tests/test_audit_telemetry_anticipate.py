"""Tests for `scripts.audit_telemetry_health.check_anticipate_health`.

Verifica que el health check del Anticipatory Agent detecta las cuatro
patologías que importan en prod:

1. send rate global bajo (<5%) con muestra suficiente → "degraded"
2. última emit muy vieja (>24h) → "stale" (daemon down)
3. signals "silent" (0 emits con ≥1 evaluated) → status per-signal
4. signals "noisy" (>10 emits con avg_score <0.3) → status per-signal

Y que:
- la tabla missing o vacía produce "unknown"
- el output es JSON-serializable end-to-end
- la `issues` list contiene strings descriptivos human-readable

Aislamiento: monkeypatch `rag.DB_PATH` ANTES de cualquier llamada a
`_ragvec_state_conn()` para que las tablas se creen en tmp_path en
lugar de tocar `~/.local/share/obsidian-rag/ragvec/telemetry.db`. Si
la fixture corriera el monkeypatch DESPUÉS, la primera invocación
escribiría a la DB de prod (gap auditado 2026-04-21 en el hardening
pass — ver `_isolate_state` en conftest.py).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag import SqliteVecClient

from scripts.audit_telemetry_health import check_anticipate_health


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db_anticipate(tmp_path, monkeypatch):
    """Aísla telemetry.db en tmp_path y dispara el DDL de
    rag_anticipate_candidates.

    El monkeypatch de `rag.DB_PATH` se aplica ANTES de abrir cualquier
    conn — `_ragvec_state_conn()` resuelve el path en tiempo de
    invocación, así que el primer `with` ya escribe en tmp_path. Sin
    este orden, la primera llamada toca la telemetry.db de prod (mismo
    gap que cubre `_stabilize_rag_state` en conftest.py).
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    SqliteVecClient(path=str(db_path))
    with rag._ragvec_state_conn() as _conn:
        pass  # trigger DDL via _ensure_telemetry_tables
    return db_path


def _open_telemetry(db_root: Path) -> sqlite3.Connection:
    """Abre una conn raw a telemetry.db (autocommit) — la API que
    consume `check_anticipate_health` espera un sqlite3.Connection
    directo, no el context manager de `_ragvec_state_conn`."""
    conn = sqlite3.connect(str(db_root / "telemetry.db"), isolation_level=None)
    return conn


def _insert(
    *,
    ts: datetime,
    kind: str = "anticipate-calendar",
    score: float = 0.7,
    dedup_key: str = "k",
    selected: int = 1,
    sent: int = 1,
    reason: str = "r",
    message_preview: str = "m",
) -> None:
    """Inserta una row directa en rag_anticipate_candidates.

    Bypassea `_anticipate_log_candidate` adrede — queremos controlar
    `ts` y `sent` exactos para reproducir escenarios stale / silent /
    noisy sin depender del wallclock real.
    """
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_anticipate_candidates "
            "(ts, kind, score, dedup_key, selected, sent, reason, message_preview) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts.isoformat(timespec="seconds"),
                kind,
                float(score),
                dedup_key,
                int(selected),
                int(sent),
                reason,
                message_preview,
            ),
        )
        conn.commit()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_empty_table_is_unknown(state_db_anticipate):
    """Tabla con DDL pero 0 rows → status=unknown.

    El daemon nunca corrió (instalación fresca, post-wipe, o el
    LaunchAgent está down desde el deploy). No podemos decir nada útil
    — devolvemos `unknown` para que el orchestrator de alerts no
    grite false-positive.
    """
    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["status"] == "unknown"
    assert out["total_evaluated"] == 0
    assert out["total_sent"] == 0
    assert out["send_rate"] == 0.0
    assert out["last_emit_age_hours"] is None
    assert out["by_signal"] == {}
    assert any("0 rows" in s or "never ran" in s for s in out["issues"]), out["issues"]


def test_recent_emit_is_healthy(state_db_anticipate):
    """1 row sent=1 hace 2h, score alto → status=healthy.

    Caso golden: el daemon disparó hace poco, todo verde.
    """
    now = datetime.now()
    _insert(ts=now - timedelta(hours=2), kind="anticipate-calendar",
            score=0.85, dedup_key="cal:01", sent=1)

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["status"] == "healthy"
    assert out["total_evaluated"] == 1
    assert out["total_sent"] == 1
    assert out["send_rate"] == 1.0
    assert out["last_emit_age_hours"] is not None
    assert 1.0 < out["last_emit_age_hours"] < 4.0
    # por-signal: la signal está healthy también
    assert out["by_signal"]["anticipate-calendar"]["status"] == "healthy"


def test_last_emit_36h_old_is_stale(state_db_anticipate):
    """Última emit hace 36h → status=stale, mensaje "daemon may be down".

    Si pasaron >24h sin que ningún signal disparara, asumimos que el
    LaunchAgent murió o la queue está atorada.
    """
    now = datetime.now()
    _insert(ts=now - timedelta(hours=36), kind="anticipate-calendar",
            score=0.8, dedup_key="cal:old", sent=1)

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["status"] == "stale"
    assert out["last_emit_age_hours"] is not None
    assert 35.0 < out["last_emit_age_hours"] < 38.0
    # issues contiene el string "daemon may be down" (lo levantamos en
    # alerting downstream).
    assert any("daemon may be down" in s for s in out["issues"]), out["issues"]


def test_low_send_rate_is_degraded(state_db_anticipate):
    """50 evaluated, 1 sent (2%) en la ventana → status=degraded.

    Threshold mal calibrado o dedup demasiado agresivo: el agente
    evalúa pero descarta casi todo. Necesitamos ≥30 rows para
    declarar degraded — con muestra chica el ratio es ruido.
    """
    now = datetime.now()
    # 49 evaluated-but-not-sent + 1 sent reciente
    for i in range(49):
        _insert(
            ts=now - timedelta(hours=1, minutes=i),
            kind="anticipate-calendar",
            score=0.4,
            dedup_key=f"d:{i}",
            sent=0,
        )
    _insert(
        ts=now - timedelta(hours=2),
        kind="anticipate-calendar",
        score=0.9,
        dedup_key="d:sent",
        sent=1,
    )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["status"] == "degraded"
    assert out["total_evaluated"] == 50
    assert out["total_sent"] == 1
    assert out["send_rate"] == pytest.approx(0.02, abs=1e-4)
    assert any("low send rate" in s for s in out["issues"]), out["issues"]


def test_low_send_rate_below_30_rows_stays_healthy(state_db_anticipate):
    """<30 rows totales evaluadas → NO se declara degraded aunque
    send_rate sea bajo. Muestra chica = ratio ruidoso.

    El threshold de 30 rows existe adrede para no gritar false-positive
    cuando el agente recién arrancó. La row sent=1 fuerza que el
    status global no sea "stale" (last_emit_age_hours <24h).
    """
    now = datetime.now()
    # row sent=1 reciente para que last_emit_age_hours quede <24h
    _insert(ts=now - timedelta(hours=1), kind="anticipate-calendar",
            score=0.9, dedup_key="seed", sent=1)
    # 20 rows evaluadas pero no sent — total 21 (<30)
    for i in range(20):
        _insert(
            ts=now - timedelta(hours=2, minutes=i),
            kind="anticipate-calendar",
            score=0.4,
            dedup_key=f"d:{i}",
            sent=0,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["total_evaluated"] == 21
    assert out["total_sent"] == 1
    assert out["send_rate"] == pytest.approx(1 / 21, abs=1e-3)
    # send_rate ~4.7% pero <30 rows → no degraded
    assert out["status"] == "healthy", out


def test_per_signal_breakdown(state_db_anticipate):
    """by_signal contiene una entry por kind con métricas calculadas
    correctamente (evaluated, emits, avg_score, send_rate)."""
    now = datetime.now()
    # calendar: 4 evaluated, 2 sent, avg score 0.7
    for i, sent in enumerate((1, 1, 0, 0)):
        _insert(
            ts=now - timedelta(hours=1 + i),
            kind="anticipate-calendar",
            score=0.7,
            dedup_key=f"cal:{i}",
            sent=sent,
        )
    # echo: 2 evaluated, 1 sent, avg score 0.5
    for i, sent in enumerate((1, 0)):
        _insert(
            ts=now - timedelta(hours=1 + i),
            kind="anticipate-echo",
            score=0.5,
            dedup_key=f"echo:{i}",
            sent=sent,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    bs = out["by_signal"]
    assert set(bs.keys()) == {"anticipate-calendar", "anticipate-echo"}

    cal = bs["anticipate-calendar"]
    assert cal["evaluated"] == 4
    assert cal["emits"] == 2
    assert cal["avg_score"] == pytest.approx(0.7, abs=1e-3)
    assert cal["send_rate"] == pytest.approx(0.5, abs=1e-3)
    assert cal["status"] == "healthy"

    echo = bs["anticipate-echo"]
    assert echo["evaluated"] == 2
    assert echo["emits"] == 1
    assert echo["avg_score"] == pytest.approx(0.5, abs=1e-3)
    assert echo["send_rate"] == pytest.approx(0.5, abs=1e-3)


def test_silent_signal_detected(state_db_anticipate):
    """Una signal con ≥1 evaluated y 0 emits → status="silent" + issue.

    Diagnóstico típico: threshold mal puesto, signal rota, o feature
    deshabilitada. La row sent=1 de otra signal se incluye para que
    el status global no sea stale (sin ningún sent ever, last_emit
    es None, y el global cae a healthy con send_rate=0 con muestra
    chica — pero queremos verificar que la silent flag dispara aun así).
    """
    now = datetime.now()
    # Una signal sana para que last_emit_age_hours <24h
    _insert(ts=now - timedelta(hours=1), kind="anticipate-calendar",
            score=0.9, dedup_key="cal:ok", sent=1)
    # echo: 5 evaluated, 0 sent → silent
    for i in range(5):
        _insert(
            ts=now - timedelta(hours=2, minutes=i),
            kind="anticipate-echo",
            score=0.3,
            dedup_key=f"echo:{i}",
            sent=0,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    assert out["by_signal"]["anticipate-echo"]["status"] == "silent"
    assert out["by_signal"]["anticipate-echo"]["emits"] == 0
    assert any(
        "silent signal: anticipate-echo" in s for s in out["issues"]
    ), out["issues"]


def test_noisy_signal_detected(state_db_anticipate):
    """15 emits con avg_score=0.2 (<0.3) → status="noisy" + issue.

    Demasiados disparos con poca confianza — false-positive farm.
    Threshold hay que subirlo o dedup hay que apretarlo.
    """
    now = datetime.now()
    for i in range(15):
        _insert(
            ts=now - timedelta(hours=1, minutes=i),
            kind="anticipate-commitment",
            score=0.2,
            dedup_key=f"cm:{i}",
            sent=1,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    cm = out["by_signal"]["anticipate-commitment"]
    assert cm["status"] == "noisy"
    assert cm["emits"] == 15
    assert cm["avg_score"] == pytest.approx(0.2, abs=1e-3)
    assert any(
        "noisy signal: anticipate-commitment" in s for s in out["issues"]
    ), out["issues"]


def test_issues_list_contains_descriptive_strings(state_db_anticipate):
    """`issues` siempre es list[str] no-vacía cuando hay un problema —
    cada string es human-readable y cita el signal/métrica afectada."""
    now = datetime.now()
    # Mix: noisy commitment + silent echo + degraded global
    for i in range(15):
        _insert(
            ts=now - timedelta(hours=1, minutes=i),
            kind="anticipate-commitment",
            score=0.2,
            dedup_key=f"cm:{i}",
            sent=1,
        )
    for i in range(5):
        _insert(
            ts=now - timedelta(hours=2, minutes=i),
            kind="anticipate-echo",
            score=0.4,
            dedup_key=f"echo:{i}",
            sent=0,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    issues = out["issues"]
    assert isinstance(issues, list)
    assert len(issues) >= 2
    assert all(isinstance(s, str) and s.strip() for s in issues)
    # Contiene refs específicas a las signals problemáticas
    assert any("anticipate-commitment" in s for s in issues), issues
    assert any("anticipate-echo" in s for s in issues), issues


def test_output_is_json_serializable(state_db_anticipate):
    """El dict completo (incluyendo by_signal y issues) round-trippea
    por json.dumps sin custom encoder. Requisito del flag --json del
    audit script."""
    now = datetime.now()
    _insert(ts=now - timedelta(hours=2), kind="anticipate-calendar",
            score=0.8, dedup_key="cal:01", sent=1)
    for i in range(15):
        _insert(
            ts=now - timedelta(hours=1, minutes=i),
            kind="anticipate-commitment",
            score=0.2,
            dedup_key=f"cm:{i}",
            sent=1,
        )

    conn = _open_telemetry(state_db_anticipate)
    try:
        out = check_anticipate_health(conn, days=7)
    finally:
        conn.close()

    blob = json.dumps(out)
    parsed = json.loads(blob)
    assert parsed["status"] == out["status"]
    assert parsed["by_signal"] == out["by_signal"]
    assert parsed["issues"] == out["issues"]
    # Numéricos preservan tipo (round trip int/float)
    assert isinstance(parsed["total_evaluated"], int)
    assert isinstance(parsed["send_rate"], (int, float))
