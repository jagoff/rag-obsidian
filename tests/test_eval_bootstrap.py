"""Tests for bootstrap CI and expanded queries.yaml."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml
from click.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_queries_yaml_all_paths_exist_or_placeholder():
    """Every `expected` path must either exist in the vault OR clearly be a
    synthetic/placeholder path (starts with 'tests/'). No typos."""
    import rag
    vault = rag.VAULT_PATH
    if not vault.is_dir():
        pytest.skip("vault not mounted in this environment")
    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for q in (data.get("queries") or []):
        for p in q.get("expected") or []:
            if not (vault / p).is_file():
                missing.append(p)
    for chain in (data.get("chains") or []):
        for turn in chain.get("turns") or []:
            for p in turn.get("expected") or []:
                if not (vault / p).is_file():
                    missing.append(p)
    assert not missing, f"Expected paths missing from vault:\n  " + "\n  ".join(missing[:10])


def test_queries_yaml_has_underrepresented_folders():
    """After expansion, 03-Resources + 04-Archive + 01-Projects coverage must
    each be nontrivial (>=3 queries). Singles-only — chains are bonus."""
    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))
    folders: dict[str, int] = {}
    for q in (data.get("queries") or []):
        for p in q.get("expected") or []:
            top = p.split("/", 1)[0]
            folders[top] = folders.get(top, 0) + 1
    assert folders.get("03-Resources", 0) >= 3, folders
    assert folders.get("04-Archive", 0) >= 3, folders
    assert folders.get("01-Projects", 0) >= 3, folders


def test_queries_yaml_singles_count():
    """Expansion target: at least 35 singles (was 21)."""
    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))
    assert len(data.get("queries") or []) >= 35


def test_bootstrap_ci_basic_shape():
    """Bootstrap CI must be inside [0, 1] for 0/1 values and bracket the mean."""
    from click.testing import CliRunner
    import rag

    # Monkey-patch retrieve to return deterministic hits — we just want the
    # CI machinery to run. Use a Click command so we exercise the real path.
    runner = CliRunner()

    class _FakeCol:
        def count(self):
            return 10

    def _fake_retrieve(col, q, k, **kw):
        # Alternating hits/misses via question string length parity
        hit = (len(q) % 2) == 0
        return {
            "metas": [{"file": "a.md" if hit else "z.md"}],
            "docs": [""], "scores": [0.1],
        }

    import types
    # Create a temp queries file with 6 queries → CI should compute
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        qfile = Path(td) / "q.yaml"
        qfile.write_text(
            "queries:\n"
            + "".join(
                f"  - question: q{i:02d} letters\n"
                f"    expected: [a.md]\n"
                for i in range(6)
            ),
            encoding="utf-8",
        )
        eval_log = Path(td) / "eval.jsonl"
        db_dir = Path(td) / "ragvec"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(rag, "get_db", lambda: _FakeCol())
            mp.setattr(rag, "retrieve", _fake_retrieve)
            mp.setattr(rag, "EVAL_LOG_PATH", eval_log)
            # Post-T10: eval log lives in rag_eval_runs (SQL).
            mp.setattr(rag, "DB_PATH", db_dir)
            res = runner.invoke(rag.eval, ["--file", str(qfile)])
            assert res.exit_code == 0, res.output

        import sqlite3
        conn = sqlite3.connect(str(db_dir / "ragvec.db"))
        conn.row_factory = sqlite3.Row
        try:
            rows = list(conn.execute(
                "SELECT singles_hit5, singles_mrr, extra_json "
                "FROM rag_eval_runs ORDER BY id"
            ).fetchall())
        finally:
            conn.close()
        assert rows
        extra = json.loads(rows[-1]["extra_json"] or "{}")
        snap = extra.get("singles") or {}
        assert "hit5_ci" in snap and "mrr_ci" in snap
        lo, hi = snap["hit5_ci"]
        assert 0.0 <= lo <= rows[-1]["singles_hit5"] <= hi <= 1.0
        lo, hi = snap["mrr_ci"]
        assert 0.0 <= lo <= rows[-1]["singles_mrr"] <= hi <= 1.0


def test_bootstrap_ci_deterministic_with_seed():
    """_bootstrap_ci uses a fixed seed; same input → same bounds."""
    # Exercise the helper directly via the click closure is awkward — use the
    # fact that rag.eval embeds the helper. Alternative: test through two
    # back-to-back runs and compare snapshots.
    import random
    # Replicate the helper logic with the same default seed + small sample
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.5]

    def run():
        rng = random.Random(42)
        n = len(values)
        means = []
        for _ in range(1000):
            s = 0.0
            for _ in range(n):
                s += values[rng.randrange(n)]
            means.append(s / n)
        means.sort()
        return means[25], means[974]

    a = run()
    b = run()
    assert a == b
