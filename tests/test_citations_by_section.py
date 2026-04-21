"""Tests for _wikilinks_by_section + _citations_by_section.

Covers the section-aware wikilink bucketing that morning + today briefs
use to populate rag_brief_written.citations_by_section_json.
"""
from __future__ import annotations

from pathlib import Path

import rag


# ── _wikilinks_by_section ─────────────────────────────────────────────────────

def test_wikilinks_by_section_groups_by_h2():
    text = (
        "# Title\n"
        "[[A]]\n"
        "## Agenda\n"
        "[[B]] and [[C]]\n"
        "## Mails\n"
        "[[D]]\n"
    )
    out = rag._wikilinks_by_section(text)
    assert out == {
        "(preamble)": ["A"],
        "Agenda": ["B", "C"],
        "Mails": ["D"],
    }


def test_wikilinks_by_section_handles_h3_as_nested_bucket():
    # H3 is treated as its own bucket (flat grouping, not hierarchical).
    text = (
        "## Today\n[[X]]\n### Subsection\n[[Y]]\n## Tomorrow\n[[Z]]\n"
    )
    out = rag._wikilinks_by_section(text)
    assert out == {"Today": ["X"], "Subsection": ["Y"], "Tomorrow": ["Z"]}


def test_wikilinks_by_section_ignores_code_fence_content():
    text = (
        "## Real\n"
        "[[Real-A]]\n"
        "```\n"
        "[[Fake]]\n"
        "```\n"
        "[[Real-B]]\n"
    )
    out = rag._wikilinks_by_section(text)
    assert out == {"Real": ["Real-A", "Real-B"]}


def test_wikilinks_by_section_empty_text():
    assert rag._wikilinks_by_section("") == {}


def test_wikilinks_by_section_no_wikilinks():
    assert rag._wikilinks_by_section("## A\nnothing here\n## B\nmore\n") == {}


def test_wikilinks_by_section_wikilink_with_alias():
    # [[Target|alias]] — _WIKILINK_RE captures `Target`
    out = rag._wikilinks_by_section("## S\n[[Target|display text]]\n")
    assert out == {"S": ["Target"]}


def test_wikilinks_by_section_preserves_duplicates_within_section():
    out = rag._wikilinks_by_section("## S\n[[A]] [[A]] [[B]]\n")
    assert out == {"S": ["A", "A", "B"]}


def test_wikilinks_by_section_h1_not_a_bucket():
    # Single-# headings get ignored; content under them goes to (preamble).
    text = "# H1\n[[A]]\n## H2\n[[B]]\n"
    out = rag._wikilinks_by_section(text)
    assert out == {"(preamble)": ["A"], "H2": ["B"]}


# ── _citations_by_section ────────────────────────────────────────────────────

def test_citations_by_section_resolves_and_filters(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Alpha.md").write_text("alpha")
    (vault / "Beta.md").write_text("beta")
    title_to_paths = {"Alpha": {"Alpha.md"}, "Beta": {"Beta.md"}}
    text = (
        "## Notas\n[[Alpha]]\n[[Unknown]]\n## Otros\n[[Beta]]\n## Empty\n"
    )
    out = rag._citations_by_section(text, title_to_paths, vault)
    # Unknown title dropped silently; Empty section has no wikilinks → omitted.
    assert out == {"Notas": ["Alpha.md"], "Otros": ["Beta.md"]}


def test_citations_by_section_dedups_within_section(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "A.md").write_text("a")
    title_to_paths = {"A": {"A.md"}}
    text = "## S\n[[A]] [[A]]\n"
    out = rag._citations_by_section(text, title_to_paths, vault)
    # _resolve_wikilinks_to_paths dedups via its seen-set
    assert out == {"S": ["A.md"]}


def test_citations_by_section_drops_ambiguous(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    title_to_paths = {"Dup": {"01/Dup.md", "02/Dup.md"}}
    text = "## S\n[[Dup]]\n[[Only-Here]]\n"
    # Only-Here doesn't resolve (no mapping) + Dup is ambiguous → section dropped
    out = rag._citations_by_section(text, title_to_paths, vault)
    assert out == {}


def test_citations_by_section_resolves_path_style_wikilinks(tmp_path: Path):
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    (vault / "02-Areas" / "Foo.md").write_text("x")
    # Title lookup misses (Foo not in title_to_paths), but path lookup hits.
    title_to_paths: dict[str, set[str]] = {}
    text = "## S\n[[02-Areas/Foo]]\n"
    out = rag._citations_by_section(text, title_to_paths, vault)
    assert out == {"S": ["02-Areas/Foo.md"]}


def test_citations_by_section_empty_input(tmp_path: Path):
    assert rag._citations_by_section("", {}, tmp_path) == {}
