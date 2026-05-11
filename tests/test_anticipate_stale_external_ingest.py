"""Tests para la signal `stale_external_ingest`.

Cubrimos:
- Source con archivos frescos → no emit.
- Source con último archivo > threshold AND baseline reciente → emit.
- Source con baseline > 30d (feature inactiva) → no emit aunque stale.
- Source nunca configurado (carpeta no existe) → no emit.
- Per-source threshold override.
- Disable completo via RAG_STALE_EXT_INGEST_SOURCES="".
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag_anticipate.signals.stale_external_ingest import (
    stale_external_ingest_signal,
)


def _make_md(path: Path, mtime: datetime) -> Path:
    """Crea un .md con mtime forzado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nstub\n---\nbody\n", encoding="utf-8")
    epoch = mtime.timestamp()
    os.utime(path, (epoch, epoch))
    return path


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """Vault temporal con la jerarquía external-ingest."""
    vault = tmp_path / "Notes"
    base = vault / "99-obsidian" / "99-AI" / "external-ingest"
    base.mkdir(parents=True)

    # Monkeypatch vault resolver para que apunte a tmp.
    import rag_anticipate.signals.stale_external_ingest as mod

    monkeypatch.setattr(mod, "_vault_path", lambda: vault)
    yield vault


def test_no_emit_when_source_fresh(tmp_vault):
    """Gmail con archivo de hoy → no emit."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    _make_md(
        tmp_vault / "99-obsidian/99-AI/external-ingest/Gmail/2026-05-11.md",
        now - timedelta(hours=2),
    )
    cands = stale_external_ingest_signal(now)
    assert cands == []


def test_emit_when_stale_with_baseline(tmp_vault):
    """Gmail con último archivo hace 10d pero baseline hace 25d → emit."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest/Gmail"
    _make_md(base / "2026-04-16.md", now - timedelta(days=25))  # baseline
    _make_md(base / "2026-05-01.md", now - timedelta(days=10))  # último (stale)
    cands = stale_external_ingest_signal(now)
    assert len(cands) == 1
    assert "Gmail" in cands[0].message
    assert cands[0].kind == "anticipate-stale_external_ingest"
    assert "10d" in cands[0].message
    # Source-specific hint should be present
    assert "credentials.json" in cands[0].message


def test_no_emit_when_baseline_too_old(tmp_vault):
    """Source con último archivo hace 40d → asumimos inactivo, no spam."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest/Gmail"
    _make_md(base / "2026-03-30.md", now - timedelta(days=42))
    cands = stale_external_ingest_signal(now)
    assert cands == []


def test_no_emit_when_source_dir_missing(tmp_vault):
    """Source nunca configurado (carpeta no existe) → no emit."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    # No creo ninguna carpeta — base existe pero Gmail/ no.
    cands = stale_external_ingest_signal(now)
    assert cands == []


def test_per_source_threshold_override(tmp_vault, monkeypatch):
    """`RAG_STALE_EXT_INGEST_DAYS_CHROME=3` baja threshold a 3d para Chrome."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest/Chrome"
    _make_md(base / "2026-04-20.md", now - timedelta(days=21))
    _make_md(base / "2026-05-06.md", now - timedelta(days=5))

    # Default (7d) → 5d no stale → no emit
    monkeypatch.delenv("RAG_STALE_EXT_INGEST_DAYS_CHROME", raising=False)
    monkeypatch.delenv("RAG_STALE_EXT_INGEST_DAYS", raising=False)
    assert stale_external_ingest_signal(now) == []

    # Override 3d → 5d IS stale → emit
    monkeypatch.setenv("RAG_STALE_EXT_INGEST_DAYS_CHROME", "3")
    cands = stale_external_ingest_signal(now)
    assert len(cands) == 1
    assert "Chrome" in cands[0].message
    # Chrome-specific hint
    assert "SQLite" in cands[0].message or "Chrome" in cands[0].message


def test_disable_via_empty_sources(tmp_vault, monkeypatch):
    """`RAG_STALE_EXT_INGEST_SOURCES=""` desactiva la signal."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest/Gmail"
    _make_md(base / "2026-04-16.md", now - timedelta(days=25))
    _make_md(base / "2026-05-01.md", now - timedelta(days=10))

    monkeypatch.setenv("RAG_STALE_EXT_INGEST_SOURCES", "")
    cands = stale_external_ingest_signal(now)
    assert cands == []


def test_cap_max_emit_two(tmp_vault, monkeypatch):
    """Si 3 sources stale, solo se emiten los 2 más viejos."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest"
    # 3 sources stale con baseline reciente
    for source, days_stale_last in (("Gmail", 10), ("Chrome", 8), ("Calendar", 15)):
        sdir = base / source
        _make_md(sdir / "2026-04-16.md", now - timedelta(days=25))  # baseline
        _make_md(
            sdir / "2026-05-recent.md",
            now - timedelta(days=days_stale_last),
        )
    cands = stale_external_ingest_signal(now)
    assert len(cands) == 2
    # El más viejo (Calendar 15d) debería ir primero.
    assert "Calendar" in cands[0].message


def test_dedup_key_includes_source_and_day(tmp_vault):
    """dedup_key debe incluir source + fecha para que la signal re-emita
    al día siguiente si el problema persiste."""
    now = datetime(2026, 5, 11, 12, 0, 0)
    base = tmp_vault / "99-obsidian/99-AI/external-ingest/Gmail"
    _make_md(base / "2026-04-16.md", now - timedelta(days=25))
    _make_md(base / "2026-05-01.md", now - timedelta(days=10))
    cands = stale_external_ingest_signal(now)
    assert cands
    assert cands[0].dedup_key == "stale_external_ingest:Gmail:2026-05-11"
