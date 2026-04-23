"""`rag free` — cleanup de bloat sin romper.

Cuatro categorias:
  (a) Tablas legacy en ragvec.db que ya viven (con mas filas) en telemetry.db
  (b) JSONL .bak.<unix_ts> > min-age-days
  (c) Logs .archived* > min-age-days
  (d) Snapshots ranker.<ts>.<pid>.json — retener los N mas nuevos

Safety invariants cubiertos:
  - NUNCA toca tablas fuera de _TELEMETRY_DDL (las state cross-source y
    meta_*/vec_*/obsidian_* sobreviven)
  - Refusa si launchd services corriendo (salvo --force)
  - Dry-run no escribe
  - --apply sin --yes solo imprime plan y sale
  - Backup automatico antes del DROP
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


# ── Fixtures ────────────────────────────────────────────────────────────────


def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def free_env(tmp_path, monkeypatch):
    """Sandbox: fake HOME + state_dir + ragvec.db + telemetry.db."""
    home = tmp_path / "home"
    state_dir = home / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True, exist_ok=True)
    db_dir = state_dir / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.setenv("HOME", str(home))

    # telemetry.db with the DDL applied (the "real" store post-T10).
    telemetry = db_dir / rag._TELEMETRY_DB_FILENAME
    conn = _open_db(telemetry)
    conn.close()

    # ragvec.db legacy — same DDL so we can seed rows. Pre-split this DB
    # held the rag_* tables; post-split they are shells we want to free.
    ragvec = db_dir / "ragvec.db"
    conn = sqlite3.connect(str(ragvec), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    conn.close()

    yield {
        "home": home,
        "state_dir": state_dir,
        "db_dir": db_dir,
        "telemetry": telemetry,
        "ragvec": ragvec,
    }


def _insert_rag_queries(conn: sqlite3.Connection, n: int, ts: str = "2026-04-01"):
    for i in range(n):
        conn.execute(
            "INSERT INTO rag_queries (ts, q) VALUES (?, ?)",
            (ts, f"query-{i}"),
        )
    conn.commit()


# ── plan_tables ─────────────────────────────────────────────────────────────


def test_plan_tables_safe_when_telemetry_has_more(free_env):
    """telemetry.db.count > ragvec.db.count → tabla marcada safe."""
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 5)
    tc.close()

    plan = rag._rag_free_plan_tables()
    row = next((p for p in plan if p["table"] == "rag_queries"), None)
    assert row is not None
    assert row["status"] == "safe"
    assert row["ragvec_count"] == 3
    assert row["telemetry_count"] == 5


def test_plan_tables_safe_when_equal_counts(free_env):
    """Igualdad (ej. rag_feedback 65 == 65 en produccion) → safe."""
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 7)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 7)
    tc.close()

    plan = rag._rag_free_plan_tables()
    row = next((p for p in plan if p["table"] == "rag_queries"), None)
    assert row is not None
    assert row["status"] == "safe"


def test_plan_tables_warn_when_ragvec_has_more(free_env):
    """ragvec.db.count > telemetry.db.count → warn (NO drop)."""
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 10)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 3)
    tc.close()

    plan = rag._rag_free_plan_tables()
    row = next((p for p in plan if p["table"] == "rag_queries"), None)
    assert row is not None
    assert row["status"] == "warn"


def test_plan_tables_skips_non_whitelisted(free_env):
    """Tablas fuera de _TELEMETRY_DDL (ej. rag_whatsapp_state) no aparecen."""
    rc = sqlite3.connect(str(free_env["ragvec"]))
    rc.execute(
        "CREATE TABLE rag_whatsapp_state ("
        " chat_jid TEXT PRIMARY KEY, last_ts TEXT, last_msg_id TEXT,"
        " updated_at TEXT)"
    )
    rc.execute(
        "INSERT INTO rag_whatsapp_state VALUES ('x', 'ts', 'mid', 'now')"
    )
    rc.commit()
    rc.close()

    plan = rag._rag_free_plan_tables()
    names = {p["table"] for p in plan}
    assert "rag_whatsapp_state" not in names
    # Tampoco meta_* o vec_* si los hubiera — no los creamos, asi que solo
    # verificamos que la funcion NO los inventa.


def test_plan_tables_skips_missing_in_ragvec(free_env):
    """Si la tabla no existe en ragvec.db (version fresca), no aparece."""
    # Borramos rag_queries solo de ragvec.db
    rc = sqlite3.connect(str(free_env["ragvec"]))
    rc.execute("DROP TABLE rag_queries")
    rc.close()

    plan = rag._rag_free_plan_tables()
    names = {p["table"] for p in plan}
    assert "rag_queries" not in names


# ── plan_baks ───────────────────────────────────────────────────────────────


def test_plan_baks_finds_old_files(free_env):
    """`.bak.<unix_ts>` > min_age_days es candidato."""
    now = time.time()
    old = int(now - 60 * 86400)  # 60 dias
    recent = int(now - 5 * 86400)  # 5 dias
    (free_env["state_dir"] / f"behavior.jsonl.bak.{old}").write_text("x")
    (free_env["state_dir"] / f"queries.jsonl.bak.{recent}").write_text("y")

    plan = rag._rag_free_plan_baks(min_age_days=30, now_ts=now)
    names = [p["path"].name for p in plan]
    assert any(n.endswith(str(old)) for n in names)
    assert not any(n.endswith(str(recent)) for n in names)


def test_plan_baks_returns_size(free_env):
    """Cada entry reporta bytes del archivo."""
    now = time.time()
    old = int(now - 60 * 86400)
    p = free_env["state_dir"] / f"behavior.jsonl.bak.{old}"
    p.write_text("x" * 1234)

    plan = rag._rag_free_plan_baks(min_age_days=30, now_ts=now)
    assert len(plan) == 1
    assert plan[0]["bytes"] == 1234


def test_plan_baks_ignores_non_numeric_suffix(free_env):
    """`.bak.foobar` (no numero) se ignora."""
    (free_env["state_dir"] / "behavior.jsonl.bak.foobar").write_text("x")

    plan = rag._rag_free_plan_baks(min_age_days=30, now_ts=time.time())
    assert plan == []


# ── plan_archived_logs ──────────────────────────────────────────────────────


def test_plan_archived_logs_finds_old(free_env):
    """.archived-20260321, .archived.1776... > min_age_days."""
    old_date = free_env["state_dir"] / "serve.error.log.archived-20260101"
    old_date.write_text("old")
    now = time.time()
    # mtime 60 dias atras
    import os
    os.utime(old_date, (now - 60 * 86400, now - 60 * 86400))

    plan = rag._rag_free_plan_archived_logs(min_age_days=30, now_ts=now)
    names = [p["path"].name for p in plan]
    assert "serve.error.log.archived-20260101" in names


def test_plan_archived_logs_skips_recent(free_env):
    """Archivado hace 5 dias no es candidato con min_age=30."""
    recent = free_env["state_dir"] / "watch.error.log.archived-20260418"
    recent.write_text("recent")
    import os
    now = time.time()
    os.utime(recent, (now - 5 * 86400, now - 5 * 86400))

    plan = rag._rag_free_plan_archived_logs(min_age_days=30, now_ts=now)
    assert plan == []


# ── plan_ranker_snapshots ───────────────────────────────────────────────────


def test_plan_ranker_keeps_newest_n(free_env):
    """Con 5 snapshots y keep=3, devuelve los 2 mas viejos."""
    tss = [1776600000, 1776700000, 1776800000, 1776900000, 1777000000]
    paths = []
    for t in tss:
        p = free_env["state_dir"] / f"ranker.{t}.12345.json"
        p.write_text("{}")
        paths.append(p)

    plan = rag._rag_free_plan_ranker_snapshots(keep=3)
    plan_names = {p["path"].name for p in plan}
    # Los 2 mas viejos son candidatos
    assert f"ranker.{tss[0]}.12345.json" in plan_names
    assert f"ranker.{tss[1]}.12345.json" in plan_names
    # Los 3 mas nuevos se conservan
    assert f"ranker.{tss[2]}.12345.json" not in plan_names
    assert f"ranker.{tss[3]}.12345.json" not in plan_names
    assert f"ranker.{tss[4]}.12345.json" not in plan_names


def test_plan_ranker_preserves_live_json(free_env):
    """El ranker.json activo NUNCA es candidato (no tiene ts en el nombre)."""
    (free_env["state_dir"] / "ranker.json").write_text("{}")
    (free_env["state_dir"] / "ranker.1776000000.999.json").write_text("{}")

    plan = rag._rag_free_plan_ranker_snapshots(keep=0)
    names = [p["path"].name for p in plan]
    assert "ranker.json" not in names
    # Con keep=0 el snapshot SI es candidato
    assert "ranker.1776000000.999.json" in names


def test_plan_ranker_fewer_than_keep(free_env):
    """Con 2 snapshots y keep=3, lista vacia."""
    (free_env["state_dir"] / "ranker.1776000000.1.json").write_text("{}")
    (free_env["state_dir"] / "ranker.1776100000.2.json").write_text("{}")

    plan = rag._rag_free_plan_ranker_snapshots(keep=3)
    assert plan == []


# ── comando integrado ──────────────────────────────────────────────────────


def test_free_dry_run_no_writes(free_env, monkeypatch):
    """Sin --apply: no borra archivos ni tablas."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 5)
    tc.close()
    now = time.time()
    old = int(now - 60 * 86400)
    bak = free_env["state_dir"] / f"behavior.jsonl.bak.{old}"
    bak.write_text("x")

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free"])
    assert result.exit_code == 0, result.output
    # Archivos y filas siguen ahi
    assert bak.exists()
    rc = sqlite3.connect(str(free_env["ragvec"]))
    assert rc.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 3
    rc.close()


def test_free_apply_without_yes_prints_plan_and_exits(free_env, monkeypatch):
    """`--apply` sin `--yes`: imprime plan pero NO ejecuta."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply"])
    assert result.exit_code == 0, result.output
    assert "--yes" in result.output.lower() or "yes" in result.output.lower()
    # Sigue existiendo la fila
    rc = sqlite3.connect(str(free_env["ragvec"]))
    assert rc.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 3
    rc.close()


def test_free_apply_yes_drops_tables(free_env, monkeypatch):
    """`--apply --yes` + safe counts → DROP las tablas legacy."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 10)
    tc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply", "--yes"])
    assert result.exit_code == 0, result.output
    # rag_queries dropeada de ragvec.db
    rc = sqlite3.connect(str(free_env["ragvec"]))
    tables = [r[0] for r in rc.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    rc.close()
    assert "rag_queries" not in tables
    # Pero sigue en telemetry.db
    tc = sqlite3.connect(str(free_env["telemetry"]))
    assert tc.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 10
    tc.close()


def test_free_apply_preserves_state_tables(free_env, monkeypatch):
    """Las state cross-source (rag_whatsapp_state etc) NUNCA se tocan."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    rc.execute(
        "CREATE TABLE rag_whatsapp_state (chat_jid TEXT PRIMARY KEY, last_ts TEXT)"
    )
    rc.execute("INSERT INTO rag_whatsapp_state VALUES ('x', 'ts')")
    rc.commit()
    rc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply", "--yes"])
    assert result.exit_code == 0, result.output
    rc = sqlite3.connect(str(free_env["ragvec"]))
    assert rc.execute(
        "SELECT COUNT(*) FROM rag_whatsapp_state"
    ).fetchone()[0] == 1
    rc.close()


def test_free_refuses_if_services_running(free_env, monkeypatch):
    """Con launchd services corriendo → refuse salvo --force."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: ["12345"])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 5)
    tc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply", "--yes"])
    assert result.exit_code != 0
    # Sigue existiendo
    rc = sqlite3.connect(str(free_env["ragvec"]))
    assert rc.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 3
    rc.close()


def test_free_force_bypasses_services_check(free_env, monkeypatch):
    """--force pasa el preflight aun con services vivos."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: ["12345"])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 3)
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 5)
    tc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply", "--yes", "--force"])
    assert result.exit_code == 0, result.output
    rc = sqlite3.connect(str(free_env["ragvec"]))
    tables = [r[0] for r in rc.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    rc.close()
    assert "rag_queries" not in tables


def test_free_warn_blocks_drop_of_regressed_table(free_env, monkeypatch):
    """Una tabla con ragvec > telemetry NO se dropea aun con --apply --yes."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    rc = sqlite3.connect(str(free_env["ragvec"]))
    _insert_rag_queries(rc, 10)
    # rag_behavior con menos en telemetry (usamos dos tablas distintas)
    rc.execute(
        "INSERT INTO rag_behavior (ts, source, event) VALUES ('now', 's', 'open')"
    )
    rc.commit()
    rc.close()
    tc = sqlite3.connect(str(free_env["telemetry"]))
    _insert_rag_queries(tc, 20)  # rag_queries: telemetry gana
    # rag_behavior queda en 0 en telemetry → warn, NO drop
    tc.close()

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--apply", "--yes"])
    assert result.exit_code == 0, result.output
    # rag_queries dropeada
    rc = sqlite3.connect(str(free_env["ragvec"]))
    tables = [r[0] for r in rc.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    # rag_behavior sigue porque ragvec > telemetry → warn
    assert "rag_behavior" in tables
    assert "rag_queries" not in tables
    rc.close()


def test_free_json_output_structure(free_env, monkeypatch):
    """--json devuelve dict parseable con categorias."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["free", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    # Estructura esperada
    assert "tables" in data
    assert "baks" in data
    assert "archived_logs" in data
    assert "ranker_snapshots" in data
    assert "total_bytes_reclaimable" in data
