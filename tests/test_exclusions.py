from __future__ import annotations

from pathlib import Path

import rag.exclusions as exclusions


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(exclusions, "_DB_PATH", tmp_path / "blacklist.db")
    monkeypatch.setattr(exclusions, "_CONFIG_PATH", tmp_path / "blacklist.json")
    monkeypatch.setattr(exclusions, "_LEGACY_IGNORED_PATH", tmp_path / "ignored_notes.json")
    monkeypatch.setattr(exclusions, "_CACHE", None)
    monkeypatch.setattr(exclusions, "_LEGACY_CACHE", None)


def test_default_blocks_cloud_services(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    assert exclusions.is_chat_blocked("Cloud Services") is True
    assert exclusions.is_chat_blocked("Avature Cloud Services") is False


def test_blocks_exact_and_fuzzy_words(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    exclusions.save_blacklist({
        "words": ["japon"],
        "fuzzy_words": ["kubernetes"],
    })

    assert exclusions.is_text_blocked("viaje a japon") is True
    assert exclusions.is_text_blocked("viaje a japones") is False
    assert exclusions.is_text_blocked("profundizar kubernetes") is True
    assert exclusions.is_text_blocked("profundizar kuberneteses") is True


def test_fuzzy_words_cover_near_variants(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    exclusions.save_blacklist({"fuzzy_words": ["japon"]})

    assert exclusions.is_text_blocked("tema japones") is True
    assert exclusions.is_text_blocked("tema japonés") is True


def test_blocks_exact_prefix_glob_and_legacy_paths(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    exclusions.save_blacklist({
        "paths": ["02-Areas/privado.md"],
        "path_prefixes": ["04-Archive/private"],
        "path_globs": ["**/secret-*.md"],
    })
    (tmp_path / "ignored_notes.json").write_text(
        '{"paths": ["01-Projects/nope.md"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(exclusions, "_LEGACY_CACHE", None)

    assert exclusions.is_path_blocked("02-Areas/privado.md") is True
    assert exclusions.is_path_blocked("04-Archive/private/note.md") is True
    assert exclusions.is_path_blocked("03-Resources/x/secret-plan.md") is True
    assert exclusions.is_path_blocked("01-Projects/nope.md") is True
    assert exclusions.is_path_blocked("02-Areas/publico.md") is False


def test_kind_aliases(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    assert exclusions.add_blacklist_item("grupo", "Familia") is True
    assert exclusions.add_blacklist_item("palabra_parecida", "japon") is True
    cfg = exclusions.load_blacklist()
    assert "Familia" in cfg["chats"]
    assert "japon" in cfg["fuzzy_words"]
