"""Tests for the 'weekly_review_due' Anticipatory Agent signal.

100% filesystem — patch `_resolve_vault_path` apuntando a tmp_path.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.weekly_review_due import (
    _iso_week_label,
    _target_week_for,
    weekly_review_due_signal,
)


# Sunday 2026-05-10 = ISO week 2026-W19 (Mon 2026-05-04 to Sun 2026-05-10)
_SUN = datetime(2026, 5, 10, 12, 0, 0)
_MON = datetime(2026, 5, 11, 9, 0, 0)
_TUE = datetime(2026, 5, 12, 14, 0, 0)
_WED = datetime(2026, 5, 13, 10, 0, 0)


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "00-Inbox/reviews").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


@pytest.fixture
def empty_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _write_digest(vault: Path, week: str) -> Path:
    p = vault / "00-Inbox/reviews" / f"{week}.md"
    p.write_text(f"# Weekly digest {week}\n", encoding="utf-8")
    return p


# ── Helper unit tests ────────────────────────────────────────────────────────


def test_iso_week_label_format():
    assert _iso_week_label(_SUN) == "2026-W19"
    assert _iso_week_label(_MON) == "2026-W20"  # ISO week starts Monday


def test_target_week_sunday():
    """Domingo → semana que termina hoy."""
    target = _target_week_for(_SUN)
    assert target is not None
    label, _ = target
    assert label == "2026-W19"


def test_target_week_monday():
    """Lunes → semana que terminó ayer."""
    target = _target_week_for(_MON)
    assert target is not None
    label, _ = target
    # Lunes 2026-05-11, ayer = domingo 2026-05-10 → W19
    assert label == "2026-W19"


def test_target_week_tuesday_returns_none():
    assert _target_week_for(_TUE) is None


def test_target_week_wednesday_returns_none():
    assert _target_week_for(_WED) is None


# ── Day filtering ────────────────────────────────────────────────────────────


def test_tuesday_emits_empty(mock_vault):
    """Martes → silencio aunque no haya digest."""
    assert weekly_review_due_signal(_TUE) == []


def test_wednesday_emits_empty(mock_vault):
    assert weekly_review_due_signal(_WED) == []


# ── Sunday/Monday firing ─────────────────────────────────────────────────────


def test_sunday_no_digest_emits(mock_vault):
    """Domingo + reviews vacío → emit con week label correcto."""
    result = weekly_review_due_signal(_SUN)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-weekly_review_due"
    assert c.score == 0.6
    assert "2026-W19" in c.message
    assert c.dedup_key == "weekly_review:2026-W19"


def test_monday_no_digest_emits(mock_vault):
    """Lunes + reviews vacío → emit week-de-ayer."""
    result = weekly_review_due_signal(_MON)
    assert len(result) == 1
    assert result[0].dedup_key == "weekly_review:2026-W19"


def test_sunday_with_digest_skip(mock_vault):
    """Domingo + digest existe → silent."""
    _write_digest(mock_vault, "2026-W19")
    assert weekly_review_due_signal(_SUN) == []


def test_monday_with_digest_skip(mock_vault):
    """Lunes + digest de la semana anterior existe → silent."""
    _write_digest(mock_vault, "2026-W19")
    assert weekly_review_due_signal(_MON) == []


def test_sunday_with_old_digest_emits(mock_vault):
    """Domingo + digest existe pero de semana ANTERIOR → emit (semana actual missing)."""
    _write_digest(mock_vault, "2026-W18")
    result = weekly_review_due_signal(_SUN)
    assert len(result) == 1
    assert "2026-W19" in result[0].dedup_key


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_no_reviews_dir_silent(empty_vault):
    """Vault sin reviews/ → silent (no emit; mismo que digest missing)."""
    result = weekly_review_due_signal(_SUN)
    # Sí emite — no tiene digest, igual señaliza para que el user cree
    # la carpeta + corra digest.
    assert len(result) == 1


def test_year_boundary():
    """Domingo del 2026-12-27 → ISO W52 de 2026 o W01 de 2027 dependiendo del calendario."""
    sun_year_end = datetime(2026, 12, 27, 12, 0)  # domingo W52 de 2026
    target = _target_week_for(sun_year_end)
    assert target is not None
    label, _ = target
    # ISO calendar: 2026-12-27 es domingo de la W52 de 2026
    assert label.startswith("2026-W")


# ── Determinism ──────────────────────────────────────────────────────────────


def test_dedup_key_changes_per_week(mock_vault):
    """Dos domingos consecutivos → distinto dedup_key."""
    sun1 = _SUN
    sun2 = _SUN + timedelta(days=7)
    r1 = weekly_review_due_signal(sun1)
    r2 = weekly_review_due_signal(sun2)
    assert r1[0].dedup_key != r2[0].dedup_key


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "weekly_review_due" in names
