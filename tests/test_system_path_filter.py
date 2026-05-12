"""Tests for `_is_daemon_generated_path` / `_is_system_path` umbrella.

Regla global del user (2026-05-11): todo bajo `99-obsidian/**` es infra
del sistema y NUNCA debe surface al user como path crudo en home /
brief / today evidence. La función `_is_daemon_generated_path` (alias
`_is_system_path`) lleva ese gate.
"""
from __future__ import annotations

import pytest

from rag import _is_daemon_generated_path, _is_system_path


@pytest.mark.parametrize("rel_path", [
    "99-obsidian/99-AI/memory/2026-05-09-foo.md",
    "99-obsidian/99-AI/runbooks/some-runbook.md",
    "99-obsidian/99-AI/system/whatsapp-share-tools/plan.md",
    "99-obsidian/99-AI/conversations/2026-04-30.md",
    "99-obsidian/99-AI/skills/remember/SKILL.md",
    "99-obsidian/99-AI/external-ingest/Calendar/today.md",
    "99-obsidian/99-AI/external-ingest/Finanzas/MOZE/2026-05.md",
    "99-obsidian/99-AI/Wiki/Categoria.md",
    "00-Inbox/reviews/2026-05-11-evening.md",
    "99-obsidian/99-AI/plans/strategy.md",
    "99-obsidian/99-Templates/daily.md",
    "99-obsidian/99-Daily routine/2026-05-11.md",
    "99-obsidian/whatever-new-system-folder/foo.md",
])
def test_system_paths_are_filtered(rel_path: str) -> None:
    """Cualquier path bajo `99-obsidian/**` cuenta como sistema."""
    assert _is_daemon_generated_path(rel_path) is True
    assert _is_system_path(rel_path) is True


@pytest.mark.parametrize("rel_path", [
    "00-Inbox/Idea Random.md",
    "00-Inbox/2026-05-11.md",
    "01-Projects/RAG-Local/plan.md",
    "02-Areas/Personal/Salud/notas.md",
    "03-Resources/Articulo.md",
    "04-Archive/old-stuff.md",
])
def test_user_paths_are_NOT_filtered(rel_path: str) -> None:
    """Las 4 jerarquías PARA del user + archive quedan intactas."""
    assert _is_daemon_generated_path(rel_path) is False
    assert _is_system_path(rel_path) is False


@pytest.mark.parametrize("rel_path", [
    "03-Resources/Calendar/2026-W18.md",  # legacy daemon path pre-migración
    "03-Resources/Screentime/2026-04-29.md",
    "03-Resources/GitHub/2026-04-29.md",
    "03-Resources/Reminders/active.md",
    "00-Inbox/WA-2026-04-25.md",
])
def test_legacy_daemon_paths_still_filtered(rel_path: str) -> None:
    """Pre-migración (2026-05-08) los ingesters vivían en `03-Resources/`.
    Mantener la lista para compat con vaults sin migrar.
    """
    assert _is_daemon_generated_path(rel_path) is True


def test_defensive_empty_and_none() -> None:
    """No crashear si el input es None o vacío."""
    assert _is_daemon_generated_path("") is False
    assert _is_daemon_generated_path(None) is False  # type: ignore[arg-type]
    assert _is_system_path("") is False
    assert _is_system_path(None) is False  # type: ignore[arg-type]


def test_leading_slash_normalized() -> None:
    """Paths absolutos accidentales con `/` al principio no rompen el match."""
    assert _is_daemon_generated_path("/99-obsidian/99-AI/memory/x.md") is True
    assert _is_daemon_generated_path("/01-Projects/plan.md") is False
