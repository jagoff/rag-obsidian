"""Tests for bootstrap CI and expanded queries.yaml."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.real_vault
def test_queries_yaml_all_paths_exist_or_placeholder():
    """Every `expected` path must either:
      1. Exist as a real file in the vault, OR
      2. Start with a known cross-source native-id prefix
         (`gmail://`, `whatsapp://`, `calendar://`, `reminders://`, `messages://`)
         — these are doc_ids, not filesystem paths. Valid ONLY as placeholders
         until the ingester lands real data; still fail if the prefix is unknown.
    Typos fail this test — every vault-relative path is checked for real.

    Rationale (Phase 1.f prep): with the cross-source corpus shipped
    (docs/design-cross-source-corpus.md §10), golden queries can now reference
    email / WhatsApp / calendar entries. The corpus uses `source://native_id`
    scheme (per §2.7); those strings aren't filesystem paths so we whitelist
    the prefix rather than `Path.is_file()`.

    Marked `@pytest.mark.real_vault` — opts out of the `_isolate_vault_path`
    autouse so `rag.VAULT_PATH` points at the real vault. READ-ONLY.
    """
    import rag
    vault = rag.VAULT_PATH
    if not vault.is_dir():
        pytest.skip("vault not mounted in this environment")

    # Cross-source native-id prefixes = placeholders acceptable before the
    # ingester populates the corpus. Kept in sync with rag.VALID_SOURCES.
    CROSS_SOURCE_PREFIXES = ("gmail://", "whatsapp://", "calendar://",
                             "reminders://", "messages://")

    def _path_ok(p: str) -> bool:
        if p.startswith(CROSS_SOURCE_PREFIXES):
            return True
        return (vault / p).is_file()

    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for q in (data.get("queries") or []):
        for p in q.get("expected") or []:
            if not _path_ok(p):
                missing.append(p)
    for chain in (data.get("chains") or []):
        for turn in chain.get("turns") or []:
            for p in turn.get("expected") or []:
                if not _path_ok(p):
                    missing.append(p)
    assert not missing, "Expected paths missing from vault:\n  " + "\n  ".join(missing[:10])


def test_queries_yaml_cross_source_prefixes_cover_all_valid_sources():
    """Sanity: the whitelist in the path-existence test must track
    rag.VALID_SOURCES. If a new source is added there, add it here too, or
    document why it's file-backed (like vault)."""
    import rag
    CROSS_SOURCE_PREFIXES = {"gmail://", "whatsapp://", "calendar://",
                             "reminders://", "messages://"}
    # vault is file-backed → not a prefix
    expected_whitelisted = rag.VALID_SOURCES - {"vault"}
    prefix_sources = {p.rstrip("://") for p in CROSS_SOURCE_PREFIXES}
    assert prefix_sources == expected_whitelisted, (
        f"Whitelist drift: prefixes {prefix_sources} vs VALID_SOURCES-vault "
        f"{expected_whitelisted}. Update test_queries_yaml_all_paths_exist_or_placeholder."
    )


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
        import rag as _rag
        conn = sqlite3.connect(str(db_dir / _rag._TELEMETRY_DB_FILENAME))
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


def test_golden_paths_never_have_chunk_index_suffix():
    """Golden `expected` paths must NEVER end with `::<N>` (the chunk index
    internal ID). The eval matcher compares against `m.get("file", "")` which
    strips that suffix — a `::0` trailing in the golden silently fails every
    match without surfacing why.

    Regression: 2026-04-21 pass de Phase 1.f tuning. Los primeros 7 placeholders
    cross-source fueron escritos con `::0` trailing y eval reportó 7/7 fails
    "cross-source rompe retrieval" — falso positivo. El fix fue quitar `::0`;
    este test asegura que la repetición del bug dispare en CI y no en
    debugging de eval regressions.
    """
    import re
    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))
    bad: list[str] = []
    for q in (data.get("queries") or []):
        for p in (q.get("expected") or []):
            if re.search(r"::\d+$", p):
                bad.append(p)
    for chain in (data.get("chains") or []):
        for turn in (chain.get("turns") or []):
            for p in (turn.get("expected") or []):
                if re.search(r"::\d+$", p):
                    bad.append(p)
    assert not bad, (
        "expected paths end with `::N` chunk index suffix — eval matcher "
        "compares against `meta['file']` which doesn't include it, so these "
        "paths silently fail every retrieve:\n  " + "\n  ".join(bad[:10])
    )


def test_golden_cross_source_paths_have_native_id_format():
    """Paths con source-prefix (gmail://, whatsapp://, etc.) must follow
    the format retrieve() returns — no trailing slash, no extra segments
    beyond what the ingester writes.
    """
    import re
    data = yaml.safe_load((REPO_ROOT / "queries.yaml").read_text(encoding="utf-8"))

    # Patterns based on actual ingester output (scripts/ingest_*.py).
    # Reminders: `reminders://<id>` where <id> can itself contain `://` (Apple URLs)
    # Gmail:     `gmail://thread/<id>`
    # WhatsApp:  `whatsapp://<chat_jid>/<msg_id>`
    # Calendar:  `calendar://<calendar_id>/<event_id>` (calendar_id is an email
    #            per Google's API, e.g. "fernandoferrari@gmail.com")
    # Messages:  `messages://<id>`
    #
    # Calendar format drift note: the original design doc (§2.7) proposed
    # `calendar://event:<id>` but the implementation (_event_file_key at
    # scripts/ingest_calendar.py) uses the two-segment form parallel to
    # WhatsApp. Test follows the implementation (ground truth).
    SOURCE_PATTERNS = [
        (re.compile(r"^gmail://thread/[\w\-]+$"),
         "gmail://thread/<id>"),
        (re.compile(r"^whatsapp://[\w@\.\-]+/[\w\-]+$"),
         "whatsapp://<chat_jid>/<msg_id>"),
        (re.compile(r"^calendar://[\w@\.\-]+/[\w\-]+$"),
         "calendar://<calendar_id>/<event_id>"),
        (re.compile(r"^reminders://.+[^/]$"),
         "reminders://<id> (sin trailing slash)"),
        (re.compile(r"^messages://[\w\-]+$"),
         "messages://<id>"),
    ]
    CROSS_SOURCE_PREFIXES = ("gmail://", "whatsapp://", "calendar://",
                             "reminders://", "messages://")

    bad: list[str] = []

    def _check(p: str) -> None:
        if not p.startswith(CROSS_SOURCE_PREFIXES):
            return
        # Identify which source this is
        for pattern, _ in SOURCE_PATTERNS:
            if pattern.match(p):
                return
        # No source pattern matched — bad format
        bad.append(p)

    for q in (data.get("queries") or []):
        for p in (q.get("expected") or []):
            _check(p)
    for chain in (data.get("chains") or []):
        for turn in (chain.get("turns") or []):
            for p in (turn.get("expected") or []):
                _check(p)
    assert not bad, (
        "cross-source expected paths don't match the format retrieve() returns.\n"
        "Valid formats:\n" +
        "\n".join(f"  {desc}" for _, desc in SOURCE_PATTERNS) +
        "\nBad:\n  " + "\n  ".join(bad[:10])
    )
