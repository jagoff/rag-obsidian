"""Regression tests for the launchd plists that drive the **learning
loops** (paraphrases, ranker tune, whisper-vocab, routing-rules,
brief-auto-tune, implicit-feedback, calibrate, etc.).

These plists are easy to break in two recurrent ways:

1. **Wrong CLI command path** — Click groups can be renamed (e.g.
   `whisper-vocab` was promoted to a sub-group `whisper vocab` on
   2026-04-25), but the plist body is just an array of strings. If the
   command no longer exists, launchd swallows it and the daemon fails
   silently every cron tick. Whisper-vocab spent ~6 days in this state
   until detected on 2026-05-01.

2. **Missing `--apply` / `--auto-promote` flags** — many commands have
   a "dry-run by default, write only with --flag" contract for safety
   in interactive use. The cron version of the same command MUST pass
   the flag, otherwise the loop is half-closed: collector OK, trainer
   OK, apply ✗ (the table never gets new rows). Routing-rules spent
   ~1 day in this state until detected on 2026-05-01.

The test strategy is to parse each plist's `ProgramArguments`, then
**actually invoke the command with `--help`** to assert it resolves to
a real Click handler. This is robust to refactors of the CLI tree —
if the command path changes, the test fails with a clear error.

Adding a new learning-loop daemon? Add it to `_LEARNING_PLISTS` below.
"""
from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module

RAG_BIN = "/usr/local/bin/rag"

_HAS_PLUTIL = shutil.which("plutil") is not None
requires_plutil = pytest.mark.skipif(
    not _HAS_PLUTIL, reason="plutil is macOS-only"
)


def _parse(xml: str) -> dict:
    return plistlib.loads(xml.encode())


def _resolve_real_rag_bin() -> str | None:
    """Locate the actual `rag` binary on this machine — used by the
    `cli_invocable` test below. Skips if not found (CI / pip-only)."""
    candidates = [
        Path.home() / ".local" / "bin" / "rag",
        Path("/usr/local/bin/rag"),
        Path("/opt/homebrew/bin/rag"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# ─────────────────────────── _whisper_vocab_plist ──────────────────────────


def test_whisper_vocab_plist_program_args_has_three_levels():
    """The real Click tree is `rag whisper vocab refresh` — three
    command tokens after the binary, not two with a hyphen.

    Pre-fix (2026-04-25 → 2026-05-01): the plist had
    ``['{rag_bin}', 'whisper-vocab', 'refresh']`` which Click parsed as
    "subcommand whisper-vocab not found", failing every nightly run.
    """
    d = _parse(rag_module._whisper_vocab_plist(RAG_BIN))
    args = d["ProgramArguments"]
    # rag_bin + 3 command tokens
    assert args == [RAG_BIN, "whisper", "vocab", "refresh"], (
        f"expected ['{RAG_BIN}', 'whisper', 'vocab', 'refresh'], "
        f"got {args!r}. The CLI tree is `rag whisper vocab refresh` (3 tokens), "
        f"NOT `rag whisper-vocab refresh` (the legacy form)."
    )


def test_whisper_vocab_plist_label():
    d = _parse(rag_module._whisper_vocab_plist(RAG_BIN))
    assert d["Label"] == "com.fer.obsidian-rag-whisper-vocab"


def test_whisper_vocab_plist_runs_at_0315():
    """03:15 is deliberate — sneaks in before the implicit-feedback
    pipeline (03:25) and online-tune (03:30) so the freshly-refreshed
    vocab is available for any nightly LLM-touching steps."""
    d = _parse(rag_module._whisper_vocab_plist(RAG_BIN))
    sched = d["StartCalendarInterval"]
    assert sched["Hour"] == 3
    assert sched["Minute"] == 15


@requires_plutil
def test_whisper_vocab_plist_lint():
    xml = rag_module._whisper_vocab_plist(RAG_BIN)
    r = subprocess.run(["plutil", "-lint", "-"], input=xml.encode(),
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


# ─────────────────────────── _routing_rules_plist ──────────────────────────


def test_routing_rules_plist_has_auto_promote_flag():
    """Without `--auto-promote`, `extract-rules` only **lists**
    candidates — it never writes to `rag_routing_rules`. The cron
    version MUST promote, otherwise the loop is half-closed: 17
    routing decisions accumulated for weeks with 0 promoted rules
    (pre-fix 2026-05-01).
    """
    d = _parse(rag_module._routing_rules_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert "--auto-promote" in args, (
        f"--auto-promote flag missing from routing-rules plist (cron). "
        f"Got args={args!r}. Without it, candidates are listed but never "
        f"promoted to rag_routing_rules → loop is half-closed."
    )


def test_routing_rules_plist_program_args():
    d = _parse(rag_module._routing_rules_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert args[:3] == [RAG_BIN, "routing", "extract-rules"], (
        f"expected first 3 tokens [bin, 'routing', 'extract-rules'], got {args[:3]!r}"
    )


def test_routing_rules_plist_runs_every_5min():
    d = _parse(rag_module._routing_rules_plist(RAG_BIN))
    assert d.get("StartInterval") == 300, (
        f"expected StartInterval=300 (5min), got {d.get('StartInterval')!r}"
    )


@requires_plutil
def test_routing_rules_plist_lint():
    xml = rag_module._routing_rules_plist(RAG_BIN)
    r = subprocess.run(["plutil", "-lint", "-"], input=xml.encode(),
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


# ─────────────────────────── CLI invocability check ────────────────────────
#
# The strongest test we can write for "this plist actually executes" is
# to spawn the same command-line with --help and assert it resolves to
# a Click handler (exit 0 + reasonable stdout). This catches both
# command-rename drift AND silent CLI regressions in a single test per
# daemon. We skip when the rag binary is not available locally
# (CI / pip-only checkouts).

_LEARNING_PLISTS = [
    pytest.param(
        "_whisper_vocab_plist",
        ["whisper", "vocab", "refresh", "--help"],
        id="whisper-vocab",
    ),
    pytest.param(
        "_routing_rules_plist",
        ["routing", "extract-rules", "--help"],
        id="routing-rules",
    ),
    pytest.param(
        "_implicit_feedback_plist",
        # implicit-feedback uses /bin/bash -c '<cmd> && <cmd> && <cmd>'
        # — we test each of the 3 sub-commands separately below.
        None,
        id="implicit-feedback",
        marks=pytest.mark.skip(reason="composed bash; see test_implicit_feedback_subcommands"),
    ),
    pytest.param(
        "_online_tune_plist",
        ["tune", "--help"],
        id="online-tune",
    ),
    pytest.param(
        "_calibration_plist",
        ["calibrate", "--help"],
        id="calibrate",
    ),
    pytest.param(
        "_auto_harvest_plist",
        ["feedback", "auto-harvest", "--help"],
        id="auto-harvest",
    ),
    pytest.param(
        "_active_learning_nudge_plist",
        ["active-learning", "nudge", "--help"],
        id="active-learning-nudge",
    ),
    pytest.param(
        "_brief_auto_tune_plist",
        ["brief", "schedule", "auto-tune", "--help"],
        id="brief-auto-tune",
    ),
    pytest.param(
        "_anticipate_plist",
        ["anticipate", "run", "--help"],
        id="anticipate",
    ),
]


@pytest.mark.parametrize("fn_name,help_args", _LEARNING_PLISTS)
def test_plist_command_resolves_to_real_click_handler(fn_name, help_args):
    """For each learning-loop plist, run the equivalent `<command tokens> --help`
    via the real `rag` binary and assert it succeeds (exit 0).

    This is the single most valuable test in the file — it catches
    command-tree refactors that silently break daemons.
    """
    if help_args is None:
        pytest.skip("composite command, see dedicated test")

    rag_bin = _resolve_real_rag_bin()
    if rag_bin is None:
        pytest.skip("no rag binary on this machine")

    fn = getattr(rag_module, fn_name)
    xml = fn(rag_bin)
    d = _parse(xml)
    plist_args = d["ProgramArguments"]
    # plist_args[0] is the binary; the rest are the command tokens.
    # We don't compare the full plist args to help_args (the plist
    # has --apply/--json/--days etc.); we just verify the *command path*
    # in the plist matches the command path in help_args (everything
    # before --help / first --flag).
    plist_cmd = []
    for tok in plist_args[1:]:
        if tok.startswith("--"):
            break
        plist_cmd.append(tok)
    expected_cmd = []
    for tok in help_args:
        if tok.startswith("--"):
            break
        expected_cmd.append(tok)
    assert plist_cmd == expected_cmd, (
        f"plist command path {plist_cmd!r} != expected {expected_cmd!r}; "
        f"plist args were {plist_args!r}"
    )

    # Now verify the command actually exists in the CLI by running it
    # with --help. Use a 30s timeout — `rag --help` is normally <1s,
    # but cold imports can spike on first run after pip install.
    r = subprocess.run(
        [rag_bin, *help_args],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"`{rag_bin} {' '.join(help_args)}` exited {r.returncode}\n"
        f"stdout: {r.stdout[:500]!r}\nstderr: {r.stderr[:500]!r}"
    )
    # Belt-and-suspenders: --help should print "Usage:" — if it
    # printed "Error: No such command" we'd see that here.
    assert "Usage:" in r.stdout or "Usage:" in r.stderr, (
        f"--help output didn't include 'Usage:'\n"
        f"stdout: {r.stdout[:500]!r}\nstderr: {r.stderr[:500]!r}"
    )


def test_implicit_feedback_subcommands_resolve():
    """The implicit-feedback plist invokes 3 subcommands chained via
    `bash -c`. Verify each one exists in the CLI."""
    rag_bin = _resolve_real_rag_bin()
    if rag_bin is None:
        pytest.skip("no rag binary on this machine")
    for sub in (
        ["feedback", "infer-implicit", "--help"],
        ["feedback", "detect-requery", "--help"],
        ["feedback", "classify-sessions", "--help"],
    ):
        r = subprocess.run([rag_bin, *sub], capture_output=True, text=True,
                           timeout=30)
        assert r.returncode == 0, (
            f"`{rag_bin} {' '.join(sub)}` exited {r.returncode}; "
            f"stderr: {r.stderr[:300]!r}"
        )
