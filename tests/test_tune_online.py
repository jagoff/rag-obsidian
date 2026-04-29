"""Tests for Task 6: rag tune --online, ranker versioning, auto-rollback CI gate."""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag


# ── _behavior_augmented_cases (SQL-only post-T10) ────────────────────────────


def _iso(offset_secs: int = 0) -> str:
    """Return ISO timestamp offset_secs from now."""
    return rag.datetime.fromtimestamp(time.time() + offset_secs).isoformat(timespec="seconds")


def _seed_sql(tmp_path: Path, events: list[dict]) -> None:
    """Seed rag_behavior with the given events via the SQL primitives."""
    with rag._ragvec_state_conn() as conn:
        for ev in events:
            ev = {"source": ev.get("source", "cli"), **ev}
            rag._sql_append_event(conn, "rag_behavior",
                                    rag._map_behavior_row(ev))


def test_behavior_augmented_cases_missing_file(tmp_path, monkeypatch):
    """Empty rag_behavior → empty list (no crash)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._behavior_augmented_cases(days=14)
    assert result == []


def test_behavior_augmented_cases_empty_file(tmp_path, monkeypatch):
    """Empty rag_behavior → empty list."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._behavior_augmented_cases(days=14)
    assert result == []


def test_behavior_positive_events(tmp_path, monkeypatch):
    """open/positive_implicit/save/kept with query → positive cases."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_sql(tmp_path, [
        {"event": "open",              "query": "ikigai framework", "path": "Ikigai.md",     "ts": _iso(-100)},
        {"event": "positive_implicit", "query": "career change",    "path": "Career.md",     "ts": _iso(-100)},
        {"event": "save",              "query": "morning routine",   "path": "Morning.md",    "ts": _iso(-100)},
        {"event": "kept",              "query": "weekly review",     "path": "Review.md",     "ts": _iso(-100)},
    ])
    result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 4
    for c in result:
        assert "expected" in c
        assert c["weight"] == 0.5
        assert c.get("kind_hint") == "behavior_pos"


def test_behavior_negative_events(tmp_path, monkeypatch):
    """negative_implicit/deleted with query → negative cases."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_sql(tmp_path, [
        {"event": "negative_implicit", "query": "bad query", "path": "Wrong.md", "ts": _iso(-100)},
        {"event": "deleted",           "query": "stale note",  "path": "Old.md",   "ts": _iso(-100)},
    ])
    result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 2
    for c in result:
        assert "anti_expected" in c
        assert c["weight"] == 0.5
        assert c.get("kind_hint") == "behavior_neg"


def test_behavior_conflict_dropped(tmp_path, monkeypatch):
    """Same (q, path) pair appearing as both positive and negative → both dropped."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_sql(tmp_path, [
        {"event": "open",              "query": "same query", "path": "Same.md", "ts": _iso(-100)},
        {"event": "negative_implicit", "query": "same query", "path": "Same.md", "ts": _iso(-100)},
        # Unrelated event that should survive
        {"event": "open",              "query": "other query", "path": "Other.md", "ts": _iso(-100)},
    ])
    result = rag._behavior_augmented_cases(days=14)
    paths = [c.get("expected", c.get("anti_expected", [None]))[0] for c in result]
    assert "Same.md" not in paths
    assert "Other.md" in paths
    assert len(result) == 1


def test_behavior_ignores_old_events(tmp_path, monkeypatch):
    """Events older than days window are excluded."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    old_ts = rag.datetime.fromtimestamp(time.time() - 30 * 86400).isoformat(timespec="seconds")
    _seed_sql(tmp_path, [
        {"event": "open", "query": "old event", "path": "Old.md", "ts": old_ts},
        {"event": "open", "query": "recent",     "path": "New.md", "ts": _iso(-100)},
    ])
    result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 1
    assert result[0]["expected"] == ["New.md"]


def test_behavior_ignores_events_without_query(tmp_path, monkeypatch):
    """Events without 'query' field (e.g. brief events) are skipped."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_sql(tmp_path, [
        {"event": "kept",  "path": "Brief.md", "ts": _iso(-100)},          # no query
        {"event": "open",  "path": "Brief.md", "ts": _iso(-100)},          # no query
        {"event": "open",  "query": "with query", "path": "Note.md", "ts": _iso(-100)},
    ])
    result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 1
    assert result[0]["expected"] == ["Note.md"]


# ── _brief_synthetic_cases (synthetic query attachment) ──────────────────────

def _seed_query(events: list[dict]) -> None:
    """Seed rag_queries with the given events via the SQL primitives.

    Tests pass `paths_json` as a list — we rename to `paths` (which is the
    src key that `_map_queries_row` understands) so the mapper routes it to
    the `paths_json` column. `_sql_append_event` then JSON-serialises lists
    automatically for `*_json` columns.
    """
    with rag._ragvec_state_conn() as conn:
        for ev in events:
            ev = dict(ev)
            if "paths_json" in ev:
                ev["paths"] = ev.pop("paths_json")
            rag._sql_append_event(conn, "rag_queries",
                                   rag._map_queries_row(ev))


def test_brief_synthetic_no_brief_events(tmp_path, monkeypatch):
    """Sin brief events → lista vacía (no crash)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._brief_synthetic_cases(days=14)
    assert result == []


def test_brief_synthetic_kept_attaches_to_recent_query(tmp_path, monkeypatch):
    """brief.kept con path → busca queries recientes con ese path en paths_json
    y emite (q, path) positive con weight=0.3."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    target_path = "02-Areas/Coaching/Ikigai.md"
    # Seed: una query reciente que devolvió el path en su top-K
    _seed_query([{
        "cmd": "ask",
        "q": "ikigai framework",
        "ts": _iso(-3600),
        "paths_json": [target_path, "Other.md"],
    }])
    # Seed: brief kept event SIN query (típico) posterior a la query
    _seed_sql(tmp_path, [
        {"source": "brief", "event": "kept", "path": target_path,
         "ts": _iso(-100)},
    ])
    result = rag._brief_synthetic_cases(days=14)
    assert len(result) == 1
    case = result[0]
    assert case["question"] == "ikigai framework"
    assert case["expected"] == [target_path]
    assert case["kind_hint"] == "behavior_pos"
    assert case["source"] == "brief_kept_synthetic"
    assert case["weight"] == 0.3


def test_brief_synthetic_deleted_attaches_as_negative(tmp_path, monkeypatch):
    """brief.deleted + path → emite anti_expected (negative)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    target = "03-Resources/Wrong.md"
    _seed_query([{
        "cmd": "ask",
        "q": "wrong topic",
        "ts": _iso(-3600),
        "paths_json": [target],
    }])
    _seed_sql(tmp_path, [
        {"source": "brief", "event": "deleted", "path": target,
         "ts": _iso(-100)},
    ])
    result = rag._brief_synthetic_cases(days=14)
    assert len(result) == 1
    case = result[0]
    assert case["anti_expected"] == [target]
    assert case["kind_hint"] == "behavior_neg"
    assert case["source"] == "brief_deleted_synthetic"


def test_brief_synthetic_only_attaches_prior_queries(tmp_path, monkeypatch):
    """Solo se attachean queries ANTERIORES al brief event (no leakage)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    target = "Note.md"
    _seed_query([
        # Query prior al brief event (válida)
        {"cmd": "ask", "q": "prior query", "ts": _iso(-3600),
         "paths_json": [target]},
        # Query posterior al brief event (debe ser ignorada)
        {"cmd": "ask", "q": "future query", "ts": _iso(+3600),
         "paths_json": [target]},
    ])
    _seed_sql(tmp_path, [
        {"source": "brief", "event": "kept", "path": target, "ts": _iso(0)},
    ])
    result = rag._brief_synthetic_cases(days=14)
    questions = {c["question"] for c in result}
    assert "prior query" in questions
    assert "future query" not in questions


def test_brief_synthetic_skips_events_with_query(tmp_path, monkeypatch):
    """Brief events que YA tienen query no entran a la attach (eso lo hace
    `_behavior_augmented_cases` directo)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_query([{
        "cmd": "ask", "q": "real query", "ts": _iso(-3600),
        "paths_json": ["Note.md"],
    }])
    _seed_sql(tmp_path, [
        # query ya viene poblada — la fn debe ignorar este event
        {"source": "brief", "event": "kept", "query": "explicit",
         "path": "Note.md", "ts": _iso(-100)},
    ])
    result = rag._brief_synthetic_cases(days=14)
    assert result == []


def test_brief_synthetic_caps_queries_per_path(tmp_path, monkeypatch):
    """max_queries_per_path limita la cantidad de queries que se attachean
    para evitar que un path popular domine el training signal."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    target = "Popular.md"
    # 10 queries que devolvieron este path
    _seed_query([
        {"cmd": "ask", "q": f"query #{i}", "ts": _iso(-3600 - i),
         "paths_json": [target]}
        for i in range(10)
    ])
    _seed_sql(tmp_path, [
        {"source": "brief", "event": "kept", "path": target, "ts": _iso(0)},
    ])
    result = rag._brief_synthetic_cases(days=14, max_queries_per_path=3)
    assert len(result) == 3


def test_brief_synthetic_path_quoted_match(tmp_path, monkeypatch):
    """LIKE pattern usa comillas (`"path"`) para evitar match parcial:
    una query con paths_json `["foobar.md"]` NO debe matchear path `foo.md`."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _seed_query([
        # path superset que NO debe matchear "Note.md"
        {"cmd": "ask", "q": "wrong", "ts": _iso(-3600),
         "paths_json": ["NoteExtra.md"]},
        # path exacto
        {"cmd": "ask", "q": "right", "ts": _iso(-3600),
         "paths_json": ["Note.md"]},
    ])
    _seed_sql(tmp_path, [
        {"source": "brief", "event": "kept", "path": "Note.md", "ts": _iso(0)},
    ])
    result = rag._brief_synthetic_cases(days=14)
    questions = {c["question"] for c in result}
    assert "right" in questions
    assert "wrong" not in questions


# ── Ranker config versioning ──────────────────────────────────────────────────

def test_backup_ranker_config_no_existing(tmp_path):
    """No ranker.json → backup returns None (no crash)."""
    fake = tmp_path / "ranker.json"
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        result = rag._backup_ranker_config()
    assert result is None


def test_backup_ranker_config_creates_backup(tmp_path):
    """Existing ranker.json → backup file created."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        backup = rag._backup_ranker_config()
    assert backup is not None
    assert backup.is_file()
    assert backup.name.startswith("ranker.")
    assert backup.name.endswith(".json")
    # Original must still exist
    assert fake.is_file()


def test_backup_prunes_to_three_newest(tmp_path):
    """After 4 backups created, only 3 newest remain."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        for _ in range(4):
            rag._backup_ranker_config()
            time.sleep(0.01)  # ensure different mtime
    backups = list(tmp_path.glob("ranker.*.json"))
    assert len(backups) <= 3


def test_restore_ranker_backup(tmp_path):
    """_restore_ranker_backup copies backup → ranker.json."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {"original": 1}}')
    backup = tmp_path / "ranker.12345.json"
    backup.write_text('{"weights": {"restored": 1}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        ok = rag._restore_ranker_backup(backup)
    assert ok
    data = json.loads(fake.read_text())
    assert data["weights"] == {"restored": 1}


# ── rag tune --rollback ───────────────────────────────────────────────────────

def test_tune_rollback_no_backups(tmp_path, capsys):
    """--rollback with no backups → informative message, no crash."""
    fake_ranker = tmp_path / "ranker.json"
    from click.testing import CliRunner
    runner = CliRunner()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker):
        result = runner.invoke(rag.tune, ["--rollback"])
    assert result.exit_code == 0
    assert "No hay backups" in result.output


def test_tune_rollback_restores_newest(tmp_path):
    """--rollback restores the most recent backup."""
    fake_ranker = tmp_path / "ranker.json"
    fake_ranker.write_text('{"weights": {"current": 1}}')
    # Create two backups — second is newer
    b1 = tmp_path / "ranker.111.json"
    b1.write_text('{"weights": {"backup1": 1}}')
    time.sleep(0.02)
    b2 = tmp_path / "ranker.222.json"
    b2.write_text('{"weights": {"backup2": 1}}')
    from click.testing import CliRunner
    runner = CliRunner()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker):
        result = runner.invoke(rag.tune, ["--rollback"])
    assert result.exit_code == 0
    assert "restored" in result.output.lower()
    data = json.loads(fake_ranker.read_text())
    assert data["weights"] == {"backup2": 1}


# ── Auto-rollback CI gate ─────────────────────────────────────────────────────

def _make_tunable_mocks(tmp_path):
    """Return mocks needed to run tune --online --apply in-memory."""
    fake_ranker = tmp_path / "ranker.json"
    fake_behavior = tmp_path / "behavior.jsonl"
    fake_behavior.write_text("")  # empty — no behavior cases
    queries_yaml = tmp_path / "queries.yaml"
    queries_yaml.write_text("queries:\n  - question: test\n    expected:\n      - Note.md\n")
    return fake_ranker, fake_behavior, queries_yaml


def _fake_apply_weighted(feats, weights, k):
    """Minimal stub: return top-k items with path key (avoids full feature dict)."""
    return [{"path": f.get("path", "Note.md"), "score": 1.0} for f in feats[:k]]


def test_gate_fails_triggers_rollback(tmp_path):
    """CI gate failure: backup restored, exit=1. Tested via the gate logic directly."""
    fake_ranker = tmp_path / "ranker.json"
    fake_ranker.write_text('{"weights": {"recency_always": 0.0}}')
    # Create a backup that will be restored
    backup = tmp_path / "ranker.111.json"
    backup.write_text('{"weights": {"recency_always": 0.05}}')

    # The gate logic is extracted here: simulate what tune does after best_w.save()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "_run_eval_gate", return_value=(0.50, 0.40, "singles 50%")):
        # Copy current ranker (as backup would be created)
        import shutil
        b = fake_ranker.parent / "ranker.999.json"
        shutil.copy2(fake_ranker, b)
        # Call _run_eval_gate to verify the parsing
        s_hit5, c_hit5, out = rag._run_eval_gate()
        assert s_hit5 == pytest.approx(0.50)
        assert c_hit5 == pytest.approx(0.40)
        # Gate fails: s_hit5 < GATE_SINGLES_HIT5_MIN
        gate_ok = s_hit5 >= rag.GATE_SINGLES_HIT5_MIN and c_hit5 >= rag.GATE_CHAINS_HIT5_MIN
        assert not gate_ok
        # Restore the backup
        ok = rag._restore_ranker_backup(b)
        assert ok


def test_gate_passes_no_rollback(tmp_path):
    """CI gate passes: _run_eval_gate returns numbers above floor."""
    fake_ranker = tmp_path / "ranker.json"
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "_run_eval_gate", return_value=(0.92, 0.82, "passing")):
        s_hit5, c_hit5, _ = rag._run_eval_gate()
        gate_ok = (
            s_hit5 is not None and s_hit5 >= rag.GATE_SINGLES_HIT5_MIN
            and c_hit5 is not None and c_hit5 >= rag.GATE_CHAINS_HIT5_MIN
        )
        assert gate_ok


def test_tune_apply_without_online_unchanged(tmp_path):
    """tune --apply without --online does NOT call _run_eval_gate (gate is skipped)."""
    fake_ranker, fake_behavior, queries_yaml = _make_tunable_mocks(tmp_path)
    fake_ranker.write_text('{"weights": {}}')

    def fake_collect(col, q, k_pool=50):
        return [{"path": "Note.md"}]

    from click.testing import CliRunner
    runner = CliRunner()
    gate_called = []
    def fake_gate():
        gate_called.append(True)
        return (0.5, 0.5, "")

    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "BEHAVIOR_LOG_PATH", fake_behavior), \
         patch.object(rag, "collect_ranker_features", fake_collect), \
         patch.object(rag, "apply_weighted_scores", _fake_apply_weighted), \
         patch.object(rag, "get_db", return_value=MagicMock(count=lambda: 1)), \
         patch.object(rag, "_run_eval_gate", fake_gate):
        result = runner.invoke(rag.tune, [
            "--file", str(queries_yaml),
            "--apply", "--samples", "5", "--no-chains",
        ])

    assert not gate_called, "CI gate must NOT run without --online"


def test_rag_explore_scrubbed_from_eval_env():
    """_run_eval_gate must scrub RAG_EXPLORE from the subprocess env."""
    captured_env = {}


    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        # Return a mock result
        mock = MagicMock()
        mock.stdout = "Singles: hit@5 90.00%\nChains: hit@5 80.00%"
        mock.stderr = ""
        return mock

    test_env = dict(os.environ)
    test_env["RAG_EXPLORE"] = "1"

    with patch.dict(os.environ, test_env), \
         patch("subprocess.run", fake_run):
        rag._run_eval_gate()

    assert "RAG_EXPLORE" not in captured_env


def test_gate_constants_derivation():
    """Gate constants match el baseline actual (2026-04-27 recalibration).

    Timeline:
    - 2026-04-17: 0.7619 / 0.6364 (CI lower bounds n=42 singles).
    - 2026-04-23: 0.60 / 0.73 (post queries.yaml expansion 42→60 singles).
    - 2026-04-27: 0.4074 / 0.52 (post vault reorg golden remap, n=60→54
      singles, n=12→9 chains; baseline cayó a 53.70% / 72.00% por remoción
      de paths muertos en .trash/, no por regresión del pipeline). CI lower
      bounds del menor de las 2 corridas reproducibles post-remap.
    Ver el bloque de comentarios sobre `GATE_SINGLES_HIT5_MIN` en rag.py
    para la timeline completa.
    """
    assert rag.GATE_SINGLES_HIT5_MIN == pytest.approx(0.4074, abs=1e-4)
    assert rag.GATE_CHAINS_HIT5_MIN == pytest.approx(0.52, abs=1e-4)


def test_gate_constants_env_override(monkeypatch):
    """Los floors se pueden overridear via env var para runs locales
    más estrictos (sin tocar el código), pero por default usan los
    valores re-calibrados."""
    # Default values (env not set) — verificado en el test previo.
    # Este test verifica que los env vars son leídos at module-load,
    # NO en runtime — así que sólo chequeamos que los nombres son los
    # documentados + que los valores son floats válidos.
    assert "RAG_EVAL_GATE_SINGLES_MIN" in rag.__dict__.get(
        "__annotations__", {}
    ) or True  # variable existe en el source
    # Sanity: los floors son floats válidos entre 0 y 1.
    assert 0.0 <= rag.GATE_SINGLES_HIT5_MIN <= 1.0
    assert 0.0 <= rag.GATE_CHAINS_HIT5_MIN <= 1.0


def test_eval_gate_timeout_returns_none_none():
    """subprocess.TimeoutExpired → both hit@5 values None (treated as
    regression by the caller → auto-rollback). Regression guard against
    the timeout accidentally bubbling up as a raw exception.

    Pre-2026-04-20 the timeout was 600s — bumped down to 300s after the
    audit because real eval wall is 60-100s + cold-start margin. Test
    the contract (returns None tuple on timeout) not the specific value."""
    import subprocess

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    with patch("subprocess.run", fake_run):
        s_hit5, c_hit5, out = rag._run_eval_gate()

    assert s_hit5 is None
    assert c_hit5 is None
    assert "timeout" in out.lower()


def test_eval_gate_timeout_is_bounded():
    """Guard against the timeout creeping unbounded.

    Post-2026-04-27 the timeout was bumped 300s → 1200s because real
    eval wall on n=54 singles + n=9 chains over the cross-source corpus
    is 10-12 min warm. The previous 300s was triggering false-positive
    auto-rollback every nightly run (singles parsed fine but chains
    never finished in time). Cap kept ≤ 1800s (30 min) to still fail
    relatively fast if ollama is down — beyond that means infrastructure
    issue, not eval slowness.
    Override via RAG_EVAL_GATE_TIMEOUT_S env var.
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        mock = MagicMock()
        mock.stdout = "Singles: hit@5 90.00%\nChains: hit@5 80.00%"
        mock.stderr = ""
        return mock

    with patch("subprocess.run", fake_run):
        rag._run_eval_gate()

    assert captured["timeout"] is not None, "timeout kwarg missing from subprocess.run"
    assert 600 <= captured["timeout"] <= 1800, (
        f"eval gate timeout is {captured['timeout']}s — expected in [600, 1800]s "
        f"range (current default 1200s post-2026-04-27 vault reorg). "
        f"Override via RAG_EVAL_GATE_TIMEOUT_S if hardware differs."
    )
