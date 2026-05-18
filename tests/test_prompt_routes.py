from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from web import prompt_routes


def _set_prompt_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    repo = root
    prompts = repo / "rag" / "prompts"
    (prompts / "intents").mkdir(parents=True)
    (prompts / "rules").mkdir(parents=True)
    monkeypatch.setattr(prompt_routes, "_REPO_ROOT", repo)
    monkeypatch.setattr(prompt_routes, "_PROMPTS_ROOT", prompts)
    return prompts


def test_prompts_list_reads_frontmatter_and_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _set_prompt_root(monkeypatch, tmp_path)
    (prompts / "intents" / "demo.v1.md").write_text(
        "---\n"
        "name: demo\n"
        "version: v1\n"
        "includes: [language_es_AR.v1]\n"
        "notes: |\n"
        "  Prompt viejo.\n"
        "---\n"
        "Body viejo\n",
        encoding="utf-8",
    )
    (prompts / "intents" / "demo.v2.md").write_text(
        "---\n"
        "name: demo\n"
        "version: v2\n"
        "notes: |\n"
        "  Prompt actual para demo.\n"
        "---\n"
        "Body actual\n",
        encoding="utf-8",
    )

    payload = prompt_routes.api_prompts_list()

    by_path = {item["path"]: item for item in payload["prompts"]}
    assert by_path["rag/prompts/intents/demo.v1.md"]["status"] == "legacy"
    assert by_path["rag/prompts/intents/demo.v1.md"]["importance"] == "low"
    assert by_path["rag/prompts/intents/demo.v2.md"]["status"] == "latest"
    assert by_path["rag/prompts/intents/demo.v2.md"]["importance"] == "medium"
    assert by_path["rag/prompts/intents/demo.v2.md"]["purpose"] == "Prompt actual para demo."
    assert by_path["rag/prompts/intents/demo.v2.md"]["effective"] == "Prompt actual para demo."


def test_prompt_read_and_write_are_scoped_to_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _set_prompt_root(monkeypatch, tmp_path)
    target = prompts / "rules" / "demo_rule.v1.md"
    target.write_text("---\nname: demo_rule\nversion: v1\nkind: rule\n---\nOld\n", encoding="utf-8")

    read = prompt_routes.api_prompt_read("rag/prompts/rules/demo_rule.v1.md")
    assert "Old" in read["content"]

    prompt_routes.api_prompt_write(
        prompt_routes.PromptWriteRequest(
            path="rag/prompts/rules/demo_rule.v1.md",
            content="---\nname: demo_rule\nversion: v1\nkind: rule\n---\nNew",
        )
    )
    assert target.read_text(encoding="utf-8").endswith("New\n")

    with pytest.raises(HTTPException) as exc:
        prompt_routes.api_prompt_read("pyproject.toml")
    assert exc.value.status_code == 403


def test_inline_prompt_read_extracts_literal_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_prompt_root(monkeypatch, tmp_path)
    source = tmp_path / "demo_prompts.py"
    source.write_text('_PROMPT = "hola\\nchau"\n', encoding="utf-8")
    monkeypatch.setattr(
        prompt_routes,
        "_INLINE_PROMPTS",
        [
            {
                "id": "demo._PROMPT",
                "path": "demo_prompts.py",
                "symbol": "_PROMPT",
                "purpose": "Prompt demo.",
            }
        ],
    )

    payload = prompt_routes.api_inline_prompt_read("demo._PROMPT")

    assert payload["prompt"]["editable"] is False
    assert payload["prompt"]["importance"] == "medium"
    assert payload["prompt"]["importance_label"] == "media"
    assert "hola\nchau" in payload["content"]
