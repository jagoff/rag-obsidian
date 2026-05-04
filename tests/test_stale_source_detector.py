"""Tests para stale_source_detector (D): cuando una query lexicalmente
similar fue contestada antes con sources que ya no existen, el detector
devuelve un hint apuntando a la conversation o al runbook destilado.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from rag import stale_source_detector as ssd


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    for d in (
        "00-Inbox",
        "03-Resources/runbooks/from-conversations",
        "04-Archive/99-obsidian-system/99-AI/conversations",
    ):
        (vault / d).mkdir(parents=True)
    return vault


def _conv(vault: Path, slug: str, query: str, sources: list[str],
          distilled_to: str | None = None) -> Path:
    conv_dir = vault / "04-Archive/99-obsidian-system/99-AI/conversations"
    p = conv_dir / f"{slug}.md"
    src_block = "\n".join(f"  - {s}" for s in sources)
    distilled_line = f"distilled_to: {distilled_to}\n" if distilled_to else ""
    p.write_text(
        f"---\ncreated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"sources:\n{src_block}\n{distilled_line}---\n\n"
        f"## Turn 1\n\n> {query}\n\nrespuesta\n",
        encoding="utf-8",
    )
    return p


def test_tokenize_drops_stopwords():
    toks = ssd._tokenize("cómo crear una cuenta global en AWS")
    assert "cuenta" in toks
    assert "global" in toks
    assert "aws" in toks
    assert "una" not in toks  # stopword
    assert "en" not in toks   # stopword


def test_jaccard_basic():
    a = {"foo", "bar", "baz"}
    b = {"bar", "baz", "qux"}
    # inter=2, union=4 → 0.5
    assert abs(ssd._jaccard(a, b) - 0.5) < 1e-9
    assert ssd._jaccard(set(), a) == 0.0


def test_finds_match_when_sources_missing(tmp_vault):
    # Past query similar a la que estamos buscando, sources missing.
    _conv(
        tmp_vault, "aws-past",
        "como crear cuenta global aws",
        ["00-Inbox/missing.md"],
    )
    hits = ssd.find_stale_matches(
        "como creo una nueva cuenta global en aws",
        tmp_vault,
        min_overlap=0.3,
    )
    assert len(hits) == 1
    assert hits[0]["overlap"] >= 0.3
    assert "00-Inbox/missing.md" in hits[0]["sources_missing"]


def test_skip_when_sources_present(tmp_vault):
    (tmp_vault / "00-Inbox/exists.md").write_text("# exists\n", encoding="utf-8")
    _conv(
        tmp_vault, "alive",
        "como crear cuenta global aws",
        ["00-Inbox/exists.md"],
    )
    hits = ssd.find_stale_matches("crear cuenta global aws", tmp_vault)
    # Sources present → no se considera stale.
    assert hits == []


def test_skip_when_no_overlap(tmp_vault):
    _conv(
        tmp_vault, "unrelated",
        "como hacer pasta carbonara",
        ["00-Inbox/missing-recipe.md"],
    )
    hits = ssd.find_stale_matches(
        "kubernetes deployment yaml",
        tmp_vault,
        min_overlap=0.3,
    )
    assert hits == []


def test_hint_prefers_distilled_runbook(tmp_vault):
    runbook_path = "03-Resources/runbooks/from-conversations/aws-cuenta.md"
    _conv(
        tmp_vault, "aws-distilled",
        "como crear cuenta global aws",
        ["00-Inbox/gone.md"],
        distilled_to=runbook_path,
    )
    hint = ssd.stale_source_hint(
        "crear cuenta global aws",
        vault=tmp_vault,
    )
    assert hint is not None
    assert runbook_path in hint
    assert "runbook destilado" in hint


def test_hint_returns_none_when_no_match(tmp_vault):
    hint = ssd.stale_source_hint(
        "algo random sin matches",
        vault=tmp_vault,
    )
    assert hint is None


def test_hint_includes_distill_tip(tmp_vault):
    _conv(
        tmp_vault, "stale",
        "como crear cuenta global aws",
        ["00-Inbox/missing.md"],
    )
    hint = ssd.stale_source_hint("crear cuenta global aws", vault=tmp_vault)
    assert hint is not None
    assert "rag distill-conversations" in hint
