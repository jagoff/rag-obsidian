#!/usr/bin/env python3
"""Audit env-var consistency between code and docs.

Walks the Python source tree (``rag/``, ``web/``, ``mcp_server.py``,
``scripts/*.py``) using ``ast`` and collects every ``os.environ.get(...)``,
``os.getenv(...)`` and ``os.environ["..."]`` reference whose key is a string
literal. Cross-checks against:

  - ``docs/env-vars-catalog.md`` (mention-only, regex over backticks)
  - ``_CONFIG_VARS`` in ``rag/__init__.py`` (curated list with default values)
  - ``CLAUDE.md`` (mention-only, regex over backticks)

Drift reported:

  - **Undocumented**: vars referenced in code but not mentioned in any doc.
  - **Stale**: vars listed in docs/_CONFIG_VARS but not used anywhere in code.
  - **Default mismatches**: same var has a different default literal between
    ``_CONFIG_VARS`` and the actual ``os.environ.get(VAR, default)`` call.

Output: human-readable text by default. ``--json`` for CI consumption.
``--strict`` exits 1 if any drift exists (otherwise always exits 0).

Usage::

    python scripts/audit_env_vars.py                # text report on cwd
    python scripts/audit_env_vars.py --json         # JSON to stdout
    python scripts/audit_env_vars.py --strict       # CI gate
    python scripts/audit_env_vars.py --root /path   # audit other tree

The script is intentionally hermetic: it never imports ``rag``, only parses
source via ``ast``. Safe to run in CI without the project venv.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from re import compile as _re_compile

# Var-name shape: matches RAG_*, OBSIDIAN_RAG_*, OBSIDIAN_*, OLLAMA_*.
# Order in alternation matters for clarity (longer prefix first) — Python's
# regex engine tries left-to-right. Both anchored alternatives produce the
# same overall match for var names, but `OBSIDIAN_RAG` first avoids any
# confusion when a future reader reasons about the precedence.
_VAR_NAME_PATTERN = _re_compile(
    r"^(?:OBSIDIAN_RAG|OBSIDIAN|OLLAMA|RAG)_[A-Z][A-Z0-9_]*$",
)

# Backticked var refs in markdown docs.
_DOC_VAR_PATTERN = _re_compile(
    r"`((?:OBSIDIAN_RAG|OBSIDIAN|OLLAMA|RAG)_[A-Z][A-Z0-9_]+)`",
)

# Source files (relative to root) that we walk via AST. Globs are resolved
# with ``Path.glob`` — single-star segments do not recurse, ``**`` does.
_SOURCE_GLOBS: tuple[str, ...] = (
    "rag/__init__.py",
    "rag/integrations/*.py",
    "web/server.py",
    "web/*.py",
    "mcp_server.py",
    "scripts/*.py",
)

# Doc files that we scan for backticked mentions.
_DOC_FILES: tuple[str, ...] = (
    "docs/env-vars-catalog.md",
    "CLAUDE.md",
)


# ── AST helpers ──────────────────────────────────────────────────────────────


def _resolve_attr_chain(node: ast.expr | None) -> str | None:
    """Resolve a dotted attribute chain to a string (``a.b.c``).

    Returns ``None`` for non-attribute / non-name nodes (e.g. subscripts,
    calls). Used to identify ``os.environ.get`` / ``os.getenv`` callers
    without false positives on similarly-named locals.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_attr_chain(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _str_literal(node: ast.expr | None) -> str | None:
    """Return the string value of an ``ast.Constant`` node, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _EnvVarVisitor(ast.NodeVisitor):
    """Collect env-var references inside a parsed source file.

    Each finding is ``(var_name, default_literal_or_None, lineno)``. We only
    record references whose KEY is a literal string matching the project's
    var-name shape; dynamic keys (``os.getenv(name)``) are skipped because
    they cannot be cross-checked against docs anyway.

    Defaults are recorded only when they are also string literals — the
    point is to compare against the strings stored in ``_CONFIG_VARS``,
    which are themselves literals. Numeric / dynamic defaults aren't
    actionable for that check.
    """

    def __init__(self) -> None:
        self.findings: list[tuple[str, str | None, int]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 — ast API
        target = _resolve_attr_chain(node.func)
        if target in {"os.environ.get", "os.getenv"} and node.args:
            key = _str_literal(node.args[0])
            if key and _VAR_NAME_PATTERN.match(key):
                default = (
                    _str_literal(node.args[1]) if len(node.args) >= 2 else None
                )
                self.findings.append((key, default, node.lineno))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if _resolve_attr_chain(node.value) == "os.environ":
            key = _str_literal(node.slice)
            if key and _VAR_NAME_PATTERN.match(key):
                self.findings.append((key, None, node.lineno))
        self.generic_visit(node)


# ── Collectors ───────────────────────────────────────────────────────────────


def collect_code_vars(
    root: Path,
) -> dict[str, list[tuple[str, int, str | None]]]:
    """Scan ``_SOURCE_GLOBS`` under ``root`` and return findings.

    The mapping is ``var_name -> [(rel_path, lineno, default_literal), ...]``.
    Multiple references to the same var (e.g. read in both ``rag.py`` and
    ``web/server.py``) are kept distinct so the default-mismatch check can
    point at the offending file:line.
    """
    out: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)
    seen: set[Path] = set()
    for pattern in _SOURCE_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file() or path.suffix != ".py":
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="ignore"),
                    filename=str(path),
                )
            except (OSError, SyntaxError):
                continue
            visitor = _EnvVarVisitor()
            visitor.visit(tree)
            rel = str(path.relative_to(root))
            for var, default, lineno in visitor.findings:
                out[var].append((rel, lineno, default))
    return dict(out)


def collect_doc_mentions(root: Path) -> dict[str, set[str]]:
    """Map ``var_name -> {doc_path, ...}`` for every backticked mention."""
    out: dict[str, set[str]] = defaultdict(set)
    for rel in _DOC_FILES:
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _DOC_VAR_PATTERN.finditer(text):
            out[match.group(1)].add(rel)
    return dict(out)


def collect_config_vars(root: Path) -> dict[str, str]:
    """Parse ``_CONFIG_VARS`` from ``rag/__init__.py`` via AST.

    Returns ``var_name -> default_literal`` for every entry whose first two
    tuple positions are string constants. Entries with non-literal defaults
    (rare in this codebase) are ignored — the mismatch check only makes
    sense when both sides are strings.
    """
    init_path = root / "rag" / "__init__.py"
    if not init_path.is_file():
        return {}
    try:
        tree = ast.parse(
            init_path.read_text(encoding="utf-8", errors="ignore"),
            filename=str(init_path),
        )
    except (OSError, SyntaxError):
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_CONFIG_VARS":
                    return _extract_config_tuple(node.value)
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "_CONFIG_VARS"
                and node.value is not None
            ):
                return _extract_config_tuple(node.value)
    return {}


def _extract_config_tuple(node: ast.expr) -> dict[str, str]:
    """Walk the ``_CONFIG_VARS`` literal tuple-of-tuples."""
    if not isinstance(node, ast.Tuple):
        return {}
    out: dict[str, str] = {}
    for entry in node.elts:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) < 2:
            continue
        name = _str_literal(entry.elts[0])
        default = _str_literal(entry.elts[1])
        if name is not None and default is not None:
            out[name] = default
    return out


# ── Audit ────────────────────────────────────────────────────────────────────


def audit(root: Path) -> dict:
    """Run all collectors and compute the drift report dict."""
    code_vars = collect_code_vars(root)
    doc_mentions = collect_doc_mentions(root)
    config_vars = collect_config_vars(root)

    code_names = set(code_vars.keys())
    documented_names = set(doc_mentions.keys()) | set(config_vars.keys())

    undocumented = sorted(code_names - documented_names)
    stale = sorted(documented_names - code_names)

    mismatches: list[dict] = []
    for var in sorted(config_vars):
        cfg_default = config_vars[var]
        for rel, lineno, code_default in code_vars.get(var, []):
            if code_default is None:
                continue
            if code_default != cfg_default:
                mismatches.append(
                    {
                        "var": var,
                        "config_default": cfg_default,
                        "code_default": code_default,
                        "code_location": f"{rel}:{lineno}",
                    },
                )

    return {
        "undocumented": undocumented,
        "stale": stale,
        "default_mismatches": mismatches,
        "code_vars_count": len(code_names),
        "doc_vars_count": len(documented_names),
    }


def has_drift(report: dict) -> bool:
    return bool(
        report["undocumented"]
        or report["stale"]
        or report["default_mismatches"],
    )


# ── Rendering ────────────────────────────────────────────────────────────────


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"env-vars audit — code: {report['code_vars_count']} vars, "
        f"docs: {report['doc_vars_count']} vars",
    )
    lines.append("")

    if report["undocumented"]:
        lines.append(
            f"## Undocumented in CLAUDE.md / docs/env-vars-catalog.md "
            f"({len(report['undocumented'])})",
        )
        for var in report["undocumented"]:
            lines.append(f"  - {var}")
        lines.append("")
    else:
        lines.append("## Undocumented: none")
        lines.append("")

    if report["stale"]:
        lines.append(
            f"## Stale (in docs but not used in code) "
            f"({len(report['stale'])})",
        )
        for var in report["stale"]:
            lines.append(f"  - {var}")
        lines.append("")
    else:
        lines.append("## Stale: none")
        lines.append("")

    if report["default_mismatches"]:
        lines.append(
            f"## Default mismatches between _CONFIG_VARS and code "
            f"({len(report['default_mismatches'])})",
        )
        for m in report["default_mismatches"]:
            lines.append(
                f"  - {m['var']}: _CONFIG_VARS={m['config_default']!r} "
                f"vs code={m['code_default']!r} ({m['code_location']})",
            )
        lines.append("")
    else:
        lines.append("## Default mismatches: none")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit env-var consistency between Python source and the "
            "project docs (CLAUDE.md, docs/env-vars-catalog.md, "
            "_CONFIG_VARS in rag/__init__.py)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any drift is detected (CI gate).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to audit (default: cwd).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    report = audit(root)

    if args.as_json:
        sys.stdout.write(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        )
    else:
        sys.stdout.write(render_text(report))

    return 1 if (args.strict and has_drift(report)) else 0


if __name__ == "__main__":
    sys.exit(main())
