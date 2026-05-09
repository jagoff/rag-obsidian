"""Tests for the 'stale_project' Anticipatory Agent signal.

Cubre:
- Vault sin 01-Projects → []
- Proyecto fresh (mtime hoy) → []
- Proyecto con <3 notas (stub) → []
- Proyecto stale 7d → emit score 0.4
- Proyecto stale 14d → emit score 0.6
- Proyecto stale 30d → emit score 0.8
- Proyecto stale 60d → emit score 0.9
- Proyecto stale 90d+ → emit score 1.0
- Múltiples stales → emit el más viejo
- Hidden / underscore folders → ignored
- Subcarpetas internas .git/ → no contribuyen al count
- dedup_key bucketed por staleness
- Signal registrado en el registry global
- Determinismo con `now` distinto

La signal es 100% filesystem. Aislamos con `monkeypatch.setattr(rag,
"_resolve_vault_path", ...)` apuntando al `tmp_path` de cada test.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.stale_project import (
    _score_for_staleness,
    _staleness_bucket,
    stale_project_signal,
)


_REF_NOW = datetime(2026, 5, 9, 12, 0, 0)


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "01-Projects").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


@pytest.fixture
def empty_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _make_project(
    vault: Path, name: str, num_notes: int, days_ago: int,
) -> Path:
    """Crea proyecto `01-Projects/<name>/` con N notas mtime = now - days_ago."""
    proj = vault / "01-Projects" / name
    proj.mkdir(parents=True)
    target_ts = (_REF_NOW - timedelta(days=days_ago)).timestamp()
    for i in range(num_notes):
        p = proj / f"nota-{i:02d}.md"
        p.write_text(f"# Nota {i}\n\nbody.\n", encoding="utf-8")
        os.utime(p, (target_ts, target_ts))
    return proj


# ── Helper unit tests ─────────────────────────────────────────────────────────


def test_staleness_bucket_mapping():
    assert _staleness_bucket(7) == "7d"
    assert _staleness_bucket(13) == "7d"
    assert _staleness_bucket(14) == "14d"
    assert _staleness_bucket(29) == "14d"
    assert _staleness_bucket(30) == "30d"
    assert _staleness_bucket(59) == "30d"
    assert _staleness_bucket(60) == "60d"
    assert _staleness_bucket(89) == "60d"
    assert _staleness_bucket(90) == "90d+"
    assert _staleness_bucket(365) == "90d+"
    assert _staleness_bucket(0) == "fresh"
    assert _staleness_bucket(6) == "fresh"


def test_score_for_staleness():
    assert _score_for_staleness(7) == 0.4
    assert _score_for_staleness(13) == 0.4
    assert _score_for_staleness(14) == 0.6
    assert _score_for_staleness(29) == 0.6
    assert _score_for_staleness(30) == 0.8
    assert _score_for_staleness(59) == 0.8
    assert _score_for_staleness(60) == 0.9
    assert _score_for_staleness(89) == 0.9
    assert _score_for_staleness(90) == 1.0
    assert _score_for_staleness(365) == 1.0


# ── Vault structure ───────────────────────────────────────────────────────────


def test_no_projects_dir_returns_empty(empty_vault):
    assert stale_project_signal(_REF_NOW) == []


def test_empty_projects_dir(mock_vault):
    assert stale_project_signal(_REF_NOW) == []


def test_fresh_project_returns_empty(mock_vault):
    """Proyecto con mtime <7d → no emit."""
    _make_project(mock_vault, "RAG-Local", num_notes=5, days_ago=3)
    assert stale_project_signal(_REF_NOW) == []


def test_project_with_too_few_notes(mock_vault):
    """Stub con 2 notas — aunque sea stale, no emit (no es un 'proyecto')."""
    _make_project(mock_vault, "Stub", num_notes=2, days_ago=30)
    assert stale_project_signal(_REF_NOW) == []


# ── Staleness scoring ─────────────────────────────────────────────────────────


def test_stale_7d_emits_score_0_4(mock_vault):
    _make_project(mock_vault, "Album-Muros", num_notes=5, days_ago=8)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-stale_project"
    assert c.score == 0.4
    assert "Album-Muros" in c.message
    assert "8d" in c.message


def test_stale_14d_emits_score_0_6(mock_vault):
    _make_project(mock_vault, "X", num_notes=5, days_ago=14)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.6


def test_stale_30d_emits_score_0_8(mock_vault):
    _make_project(mock_vault, "X", num_notes=5, days_ago=35)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.8


def test_stale_60d_emits_score_0_9(mock_vault):
    _make_project(mock_vault, "X", num_notes=5, days_ago=65)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.9


def test_stale_90d_plus_emits_score_1_0(mock_vault):
    _make_project(mock_vault, "X", num_notes=5, days_ago=120)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 1.0


# ── Multi-project selection ───────────────────────────────────────────────────


def test_multiple_stales_emits_oldest(mock_vault):
    """Con 3 proyectos stale, emit el de mayor staleness (top-1)."""
    _make_project(mock_vault, "Old", num_notes=5, days_ago=100)
    _make_project(mock_vault, "Medium", num_notes=5, days_ago=20)
    _make_project(mock_vault, "Recent", num_notes=5, days_ago=8)
    result = stale_project_signal(_REF_NOW)
    assert len(result) == 1
    assert "Old" in result[0].message
    assert result[0].score == 1.0


def test_dedup_key_bucket_format(mock_vault):
    _make_project(mock_vault, "MyProj", num_notes=5, days_ago=35)
    result = stale_project_signal(_REF_NOW)
    assert result[0].dedup_key == "stale_project:MyProj:30d"


def test_dedup_key_changes_across_buckets(mock_vault):
    """Mismo proyecto, distinta staleness → distinto bucket → re-pusheable."""
    proj = _make_project(mock_vault, "P", num_notes=5, days_ago=10)
    r1 = stale_project_signal(_REF_NOW)
    assert r1[0].dedup_key == "stale_project:P:7d"

    # Avanzar el reloj 5 días — staleness sube a 15 → bucket 14d
    later = _REF_NOW + timedelta(days=5)
    r2 = stale_project_signal(later)
    assert r2[0].dedup_key == "stale_project:P:14d"


# ── Hidden / internal folder handling ─────────────────────────────────────────


def test_hidden_folders_ignored(mock_vault):
    """`.git/`, `_drafts/` no cuentan como proyectos."""
    _make_project(mock_vault, ".git", num_notes=5, days_ago=30)
    _make_project(mock_vault, "_drafts", num_notes=5, days_ago=30)
    assert stale_project_signal(_REF_NOW) == []


def test_internal_subdirs_ignored_in_count(mock_vault):
    """Notas dentro de `.git/` o `_internal/` no cuentan al num_notes."""
    proj = mock_vault / "01-Projects" / "MyProj"
    proj.mkdir(parents=True)
    target_ts = (_REF_NOW - timedelta(days=20)).timestamp()
    # 2 notas legítimas
    for i in range(2):
        p = proj / f"nota-{i}.md"
        p.write_text("body", encoding="utf-8")
        os.utime(p, (target_ts, target_ts))
    # 5 notas en .git/ — NO cuentan
    git = proj / ".git"
    git.mkdir()
    for i in range(5):
        p = git / f"junk-{i}.md"
        p.write_text("garbage", encoding="utf-8")
        os.utime(p, (target_ts, target_ts))
    # Total legítimas: 2 < 3 _STALE_MIN_NOTES → no emit
    assert stale_project_signal(_REF_NOW) == []


# ── Recursive subdir activity ─────────────────────────────────────────────────


def test_activity_in_subdir_counts(mock_vault):
    """Si la última actividad fue en un subdir (`Letras/foo.md`), eso cuenta
    como actividad del proyecto entero."""
    proj = mock_vault / "01-Projects" / "Album"
    sub = proj / "Letras"
    sub.mkdir(parents=True)
    # 3 notas viejas en raíz
    old_ts = (_REF_NOW - timedelta(days=100)).timestamp()
    for i in range(3):
        p = proj / f"nota-{i}.md"
        p.write_text("body", encoding="utf-8")
        os.utime(p, (old_ts, old_ts))
    # 1 nota fresca en subdir — activa el proyecto
    fresh_ts = (_REF_NOW - timedelta(days=2)).timestamp()
    p = sub / "letra-fresh.md"
    p.write_text("body", encoding="utf-8")
    os.utime(p, (fresh_ts, fresh_ts))
    # mtime newest = fresh_ts → no stale
    assert stale_project_signal(_REF_NOW) == []


# ── Registry ──────────────────────────────────────────────────────────────────


def test_signal_registered_in_global_registry():
    import rag_anticipate.signals  # noqa: F401  trigger autodiscovery
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "stale_project" in names


# ── Determinism ───────────────────────────────────────────────────────────────


def test_now_param_changes_staleness(mock_vault):
    """Mismo vault, distinto `now` → distinto gap days → distinto score."""
    _make_project(mock_vault, "P", num_notes=5, days_ago=8)

    # Reloj 1 día más tarde
    later = _REF_NOW + timedelta(days=10)
    result = stale_project_signal(later)
    # Ahora staleness es ~18 días → bucket 14d → score 0.6
    assert result[0].score == 0.6
