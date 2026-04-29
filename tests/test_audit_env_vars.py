"""Tests for ``scripts/audit_env_vars.py``.

The script is hermetic — it parses Python source via ``ast`` and reads
markdown via regex, never imports project modules. Tests build a synthetic
mini-tree under ``tmp_path`` matching the layout the script expects
(``rag/__init__.py``, ``docs/env-vars-catalog.md``, ``CLAUDE.md``, etc.) so
each scenario is self-contained and doesn't depend on the live repo state.

Cases covered (≥4):

  1. Var present in code but missing from docs → reported in ``undocumented``.
  2. Var present in docs but never referenced in code → reported in ``stale``.
  3. ``_CONFIG_VARS`` default disagrees with the literal in
     ``os.environ.get(VAR, default)`` → reported in ``default_mismatches``.
  4. ``--strict`` returns exit code 1 when drift exists, 0 otherwise.
  5. ``--json`` emits parseable JSON with all expected keys.
  6. Subscript form (``os.environ["VAR"]``) is detected.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

# Path to the script under test, resolved from repo root via this test file.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_env_vars.py"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tree(
    root: Path,
    *,
    rag_init: str = "",
    catalog: str = "",
    claude_md: str = "",
    web_server: str | None = None,
    mcp_server: str | None = None,
    extra_scripts: dict[str, str] | None = None,
    extra_integrations: dict[str, str] | None = None,
) -> None:
    """Build a minimal repo skeleton under ``root``.

    Only the files relevant to the test case are written. The script
    silently skips missing files so partial trees produce well-defined
    outputs.
    """
    (root / "rag").mkdir(parents=True, exist_ok=True)
    (root / "rag" / "integrations").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "web").mkdir(parents=True, exist_ok=True)

    (root / "rag" / "__init__.py").write_text(rag_init, encoding="utf-8")
    (root / "docs" / "env-vars-catalog.md").write_text(catalog, encoding="utf-8")
    (root / "CLAUDE.md").write_text(claude_md, encoding="utf-8")
    if web_server is not None:
        (root / "web" / "server.py").write_text(web_server, encoding="utf-8")
    if mcp_server is not None:
        (root / "mcp_server.py").write_text(mcp_server, encoding="utf-8")
    for name, body in (extra_scripts or {}).items():
        (root / "scripts" / name).write_text(body, encoding="utf-8")
    for name, body in (extra_integrations or {}).items():
        (root / "rag" / "integrations" / name).write_text(body, encoding="utf-8")


def _run_audit(root: Path, *flags: str) -> tuple[int, str]:
    """Invoke the audit script as a subprocess so the CLI surface is tested."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--root", str(root), *flags],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout


def _import_audit_module():
    """Import the script as a module so we can call ``audit()`` directly.

    Done lazily — when only the JSON CLI is exercised we don't need this.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_env_vars", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── Test cases ──────────────────────────────────────────────────────────────


def test_detects_undocumented_var_in_code(tmp_path: Path) -> None:
    """Var used in code but not in any doc → 'undocumented'."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os

            FOO = os.environ.get("RAG_ONLY_IN_CODE", "")
            BAR = os.environ.get("RAG_DOCUMENTED", "1")
            """,
        ).strip(),
        catalog="`RAG_DOCUMENTED` is fine.\n",
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    assert "RAG_ONLY_IN_CODE" in report["undocumented"]
    assert "RAG_DOCUMENTED" not in report["undocumented"]


def test_detects_stale_var_in_docs(tmp_path: Path) -> None:
    """Var mentioned in docs but never referenced in code → 'stale'."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            X = os.environ.get("RAG_REAL", "")
            """,
        ).strip(),
        catalog="`RAG_REAL` is in code.\n`RAG_GHOST_IN_DOCS` is not.\n",
        claude_md="Also see `RAG_ANOTHER_GHOST`.\n",
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    assert "RAG_GHOST_IN_DOCS" in report["stale"]
    assert "RAG_ANOTHER_GHOST" in report["stale"]
    assert "RAG_REAL" not in report["stale"]


def test_detects_default_mismatch(tmp_path: Path) -> None:
    """``_CONFIG_VARS`` default vs ``os.environ.get`` literal disagreement."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os

            _CONFIG_VARS = (
                ("RAG_FAST_PATH", "1", "bool", "fast path on"),
                ("RAG_TIMEOUT", "30", "int", "timeout in seconds"),
            )

            FAST = os.environ.get("RAG_FAST_PATH", "0")  # mismatch with config
            TIMEOUT = os.environ.get("RAG_TIMEOUT", "30")  # match
            """,
        ).strip(),
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    mismatches = {m["var"]: m for m in report["default_mismatches"]}
    assert "RAG_FAST_PATH" in mismatches
    assert mismatches["RAG_FAST_PATH"]["config_default"] == "1"
    assert mismatches["RAG_FAST_PATH"]["code_default"] == "0"
    assert "RAG_TIMEOUT" not in mismatches


def test_strict_exits_one_on_drift(tmp_path: Path) -> None:
    """``--strict`` exits 1 when drift is found, 0 when clean."""
    # Drift case: var in code, no docs.
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            X = os.environ.get("RAG_DRIFT", "")
            """,
        ).strip(),
    )

    rc, _ = _run_audit(tmp_path, "--strict")
    assert rc == 1

    # Without --strict, same drift still exits 0 (informational mode).
    rc, _ = _run_audit(tmp_path)
    assert rc == 0


def test_strict_exits_zero_when_clean(tmp_path: Path) -> None:
    """No drift + ``--strict`` → exit 0."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os

            _CONFIG_VARS = (
                ("RAG_CLEAN", "1", "bool", "clean"),
            )

            X = os.environ.get("RAG_CLEAN", "1")
            """,
        ).strip(),
        catalog="`RAG_CLEAN` documented.\n",
    )

    rc, _ = _run_audit(tmp_path, "--strict")
    assert rc == 0


def test_json_output_is_parseable(tmp_path: Path) -> None:
    """``--json`` emits structured output with all expected keys."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            X = os.environ.get("RAG_FOO", "")
            """,
        ).strip(),
    )

    rc, stdout = _run_audit(tmp_path, "--json")
    assert rc == 0
    payload = json.loads(stdout)
    for key in (
        "undocumented",
        "stale",
        "default_mismatches",
        "code_vars_count",
        "doc_vars_count",
    ):
        assert key in payload
    assert "RAG_FOO" in payload["undocumented"]


def test_subscript_form_is_detected(tmp_path: Path) -> None:
    """``os.environ["VAR"]`` is also picked up (no default to check)."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            X = os.environ["RAG_SUBSCRIPT"]
            """,
        ).strip(),
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    assert "RAG_SUBSCRIPT" in report["undocumented"]


def test_dynamic_keys_are_skipped(tmp_path: Path) -> None:
    """``os.environ.get(name)`` (variable key) is ignored — can't audit it."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            name = "RAG_DYNAMIC"
            X = os.environ.get(name, "")
            Y = os.environ.get("RAG_LITERAL", "")
            """,
        ).strip(),
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    # Only the literal-key reference should be detected.
    assert "RAG_DYNAMIC" not in report["undocumented"]
    assert "RAG_LITERAL" in report["undocumented"]


def test_walks_integrations_and_scripts(tmp_path: Path) -> None:
    """Script must scan ``rag/integrations/*.py`` and ``scripts/*.py`` too."""
    _make_tree(
        tmp_path,
        rag_init="import os\n",
        extra_integrations={
            "whatsapp.py": dedent(
                """
                import os
                X = os.environ.get("RAG_FROM_INTEGRATION", "")
                """,
            ).strip(),
        },
        extra_scripts={
            "ingest_x.py": dedent(
                """
                import os
                Y = os.environ.get("RAG_FROM_SCRIPT", "")
                """,
            ).strip(),
        },
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    assert "RAG_FROM_INTEGRATION" in report["undocumented"]
    assert "RAG_FROM_SCRIPT" in report["undocumented"]


@pytest.mark.parametrize(
    "prefix",
    ["RAG", "OBSIDIAN_RAG", "OBSIDIAN", "OLLAMA"],
)
def test_supported_prefixes(tmp_path: Path, prefix: str) -> None:
    """All four expected prefixes (RAG/OBSIDIAN_RAG/OBSIDIAN/OLLAMA) match."""
    var = f"{prefix}_TEST_VAR"
    _make_tree(
        tmp_path,
        rag_init=dedent(
            f"""
            import os
            X = os.environ.get("{var}", "")
            """,
        ).strip(),
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    assert var in report["undocumented"]


def test_unprefixed_vars_are_ignored(tmp_path: Path) -> None:
    """Vars without the expected prefixes (``HOME``, ``PATH``, etc.) are skipped."""
    _make_tree(
        tmp_path,
        rag_init=dedent(
            """
            import os
            home = os.environ.get("HOME", "")
            path = os.environ.get("PATH", "")
            mine = os.environ.get("RAG_MINE", "")
            """,
        ).strip(),
    )

    mod = _import_audit_module()
    report = mod.audit(tmp_path)

    # HOME / PATH must not pollute the audit.
    assert "HOME" not in report["undocumented"]
    assert "PATH" not in report["undocumented"]
    assert "RAG_MINE" in report["undocumented"]
