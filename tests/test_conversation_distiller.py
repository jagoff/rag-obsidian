"""Tests para conversation_distiller (B): rescatar bot answers cuyas
sources se evaporaron, escribiéndolos como notas runbook indexables.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag import conversation_distiller as cd


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    for d in (
        "00-Inbox",
        "03-Resources",
        "04-Archive/99-obsidian-system/99-AI/conversations",
    ):
        (vault / d).mkdir(parents=True)
    return vault


def _conv(vault: Path, slug: str, sources: list[str], confidence: float = 0.6,
          extra_fm: str = "", body: str | None = None) -> Path:
    conv_dir = vault / "04-Archive/99-obsidian-system/99-AI/conversations"
    p = conv_dir / f"{slug}.md"
    created = datetime.now().isoformat(timespec="seconds")
    src_block = "\n".join(f"  - {s}" for s in sources) if sources else ""
    body_text = body or (
        "## Turn 1 — 14:00\n\n"
        "> como crear cuenta global aws?\n\n"
        "Para crear una cuenta global, hacé esto:\n\n"
        "1. Entrá a https://avature-global.awsapps.com/start\n"
        "2. Usá la cuenta `prod-AWSInfra`\n"
        "3. Corré la lambda `AFT-new-account`\n\n"
        "**Sources**: [[00-Inbox/Crear cuenta]]\n"
    )
    p.write_text(
        f"---\nsession_id: web:abc\ncreated: {created}\n"
        f"turns: 1\nconfidence_avg: {confidence}\n"
        + (f"sources:\n{src_block}\n" if src_block else "")
        + extra_fm
        + "tags:\n  - conversation\n---\n\n" + body_text,
        encoding="utf-8",
    )
    return p


def test_finds_orphaned_with_missing_sources(tmp_vault):
    # Source missing → debería aparecer.
    _conv(tmp_vault, "orphan", ["00-Inbox/no-existe.md"])
    # Source presente → no debería aparecer (con require_missing_source=True).
    (tmp_vault / "00-Inbox/exists.md").write_text("# exists\n", encoding="utf-8")
    _conv(tmp_vault, "alive", ["00-Inbox/exists.md"])
    out = cd.find_orphaned_conversations(tmp_vault, min_confidence=0.5)
    paths = [c["conv_path"] for c in out]
    assert any("orphan" in p for p in paths)
    assert not any("alive" in p for p in paths)


def test_skip_low_confidence(tmp_vault):
    _conv(tmp_vault, "weak", ["00-Inbox/x.md"], confidence=0.3)
    out = cd.find_orphaned_conversations(tmp_vault, min_confidence=0.5)
    assert not any("weak" in c["conv_path"] for c in out)


def test_skip_already_distilled(tmp_vault):
    _conv(
        tmp_vault, "done", ["00-Inbox/x.md"],
        extra_fm="distilled_to: 04-Archive/99-obsidian-system/99-AI/runbooks/from-conversations/done.md\n",
    )
    out = cd.find_orphaned_conversations(tmp_vault)
    assert not any("done" in c["conv_path"] for c in out)


def test_distill_writes_runbook_with_bot_answer(tmp_vault):
    _conv(tmp_vault, "aws", ["00-Inbox/Crear cuenta.md"])
    res = cd.run_distillation(tmp_vault, apply=True, min_confidence=0.5)
    assert len(res["distilled"]) == 1
    runbook_rel = res["distilled"][0]["runbook"]
    assert runbook_rel.startswith("04-Archive/99-obsidian-system/99-AI/runbooks/from-conversations/")
    rb = (tmp_vault / runbook_rel).read_text(encoding="utf-8")
    assert "type: runbook" in rb
    assert "AFT-new-account" in rb  # bot answer preserved
    assert "como crear cuenta global aws?" in rb  # original query as context
    assert "**Sources**" not in rb  # sources block stripped
    assert "00-Inbox/Crear cuenta.md" in rb  # missing source listed in fm


def test_distill_idempotent(tmp_vault):
    _conv(tmp_vault, "twice", ["00-Inbox/missing.md"])
    cd.run_distillation(tmp_vault, apply=True)
    res2 = cd.run_distillation(tmp_vault, apply=True)
    # Segunda corrida no destila de nuevo (frontmatter ya tiene distilled_to).
    assert res2["candidates"] == 0


def test_dry_run_writes_nothing(tmp_vault):
    _conv(tmp_vault, "preview", ["00-Inbox/missing.md"])
    res = cd.run_distillation(tmp_vault, apply=False)
    assert len(res["distilled"]) == 1
    assert res["distilled"][0]["runbook"]
    runbook = tmp_vault / res["distilled"][0]["runbook"]
    assert not runbook.exists()


def test_extract_bot_answer_strips_user_query():
    turn = (
        "## Turn 1\n\n"
        "> esta es la pregunta\n\n"
        "Esta es la respuesta del bot.\n\n"
        "**Sources**: [[foo]]\n"
    )
    user_q, answer = cd._extract_bot_answer(turn)
    assert user_q == "esta es la pregunta"
    assert "Esta es la respuesta" in answer
    assert "**Sources**" not in answer
    assert "esta es la pregunta" not in answer


def test_distill_handles_multiple_turns(tmp_vault):
    body = (
        "## Turn 1\n\n> primera\n\nrespuesta uno\n\n"
        "## Turn 2\n\n> segunda\n\nrespuesta dos con `código`\n"
    )
    _conv(tmp_vault, "multi", ["00-Inbox/missing.md"], body=body)
    res = cd.run_distillation(tmp_vault, apply=True)
    rb = (tmp_vault / res["distilled"][0]["runbook"]).read_text(encoding="utf-8")
    assert "respuesta uno" in rb
    assert "respuesta dos" in rb
