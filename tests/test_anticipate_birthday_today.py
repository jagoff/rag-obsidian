"""Tests for the 'birthday_today' Anticipatory Agent signal.

100% filesystem — patch `_resolve_vault_path` apuntando a tmp_path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.birthday_today import (
    _matches_today,
    _parse_birthday,
    birthday_today_signal,
)


# Reference: 2026-05-10 (sunday, non-leap year)
_NOW = datetime(2026, 5, 10, 12, 0, 0)
_FEB28_NON_LEAP = datetime(2026, 2, 28, 12, 0, 0)
_FEB29_LEAP = datetime(2024, 2, 29, 12, 0, 0)


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "99-obsidian/99-Contacts").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


@pytest.fixture
def empty_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _write_contact(
    vault: Path, name: str, body: str, frontmatter: str | None = None,
) -> Path:
    p = vault / "99-obsidian/99-Contacts" / f"{name}.md"
    content = ""
    if frontmatter is not None:
        content += f"---\n{frontmatter}\n---\n\n"
    content += body
    p.write_text(content, encoding="utf-8")
    return p


# ── _parse_birthday ─────────────────────────────────────────────────────────


def test_parse_basic_dd_mm_yyyy():
    assert _parse_birthday("- **Cumpleaños**: 10/05/1981") == (10, 5, 1981)


def test_parse_with_dash_separator():
    assert _parse_birthday("Cumpleaños: 10-05-1981") == (10, 5, 1981)


def test_parse_no_year():
    assert _parse_birthday("Cumpleaños: 10/05") == (10, 5, None)


def test_parse_short_year_30s_to_99():
    """Year 81 → 1981 (>=30 → 1900s)."""
    assert _parse_birthday("Cumpleaños: 10/05/81") == (10, 5, 1981)


def test_parse_short_year_under_30():
    """Year 05 → 2005 (<30 → 2000s)."""
    assert _parse_birthday("Cumpleaños: 10/05/05") == (10, 5, 2005)


def test_parse_no_tilde():
    """`Cumpleanos` (sin tilde) también matchea."""
    assert _parse_birthday("Cumpleanos: 10/05/1981") == (10, 5, 1981)


def test_parse_invalid_month():
    assert _parse_birthday("Cumpleaños: 10/13/1981") is None


def test_parse_invalid_day():
    assert _parse_birthday("Cumpleaños: 32/05/1981") is None


def test_parse_no_match():
    assert _parse_birthday("texto sin cumpleaños") is None


def test_parse_year_in_future_dropped():
    """Cumpleaños con año futuro → year = None pero day/month válidos."""
    result = _parse_birthday("Cumpleaños: 10/05/2099")
    assert result is not None
    assert result[0] == 10 and result[1] == 5
    assert result[2] is None


def test_parse_year_pre_1900_dropped():
    result = _parse_birthday("Cumpleaños: 10/05/1850")
    assert result is not None
    assert result[2] is None


# ── _matches_today ──────────────────────────────────────────────────────────


def test_matches_today_exact():
    assert _matches_today(10, 5, _NOW) is True


def test_matches_today_diff_day():
    assert _matches_today(11, 5, _NOW) is False


def test_matches_today_diff_month():
    assert _matches_today(10, 6, _NOW) is False


def test_matches_today_29feb_non_leap():
    """29-feb cumple emite el 28-feb cuando el año NO es bisiesto."""
    assert _matches_today(29, 2, _FEB28_NON_LEAP) is True


def test_matches_today_29feb_leap():
    """29-feb cumple en año bisiesto → solo matchea el 29-feb, no el 28."""
    assert _matches_today(29, 2, _FEB29_LEAP) is True
    feb28_leap = datetime(2024, 2, 28, 12, 0)
    assert _matches_today(29, 2, feb28_leap) is False


# ── Empty vault / dirs ───────────────────────────────────────────────────────


def test_no_contacts_dir_returns_empty(empty_vault):
    assert birthday_today_signal(_NOW) == []


def test_empty_contacts_dir(mock_vault):
    assert birthday_today_signal(_NOW) == []


def test_contact_no_birthday_field(mock_vault):
    _write_contact(mock_vault, "Juan", "# Juan\n\nNo birthday here.\n")
    assert birthday_today_signal(_NOW) == []


# ── Happy path ───────────────────────────────────────────────────────────────


def test_birthday_today_emits(mock_vault):
    _write_contact(mock_vault, "Maria",
                   "- **Cumpleaños**: 10/05/1991\n")
    result = birthday_today_signal(_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-birthday_today"
    assert c.score == 1.0
    assert "Maria" in c.message
    assert "35 años" in c.message  # 2026 - 1991


def test_birthday_today_no_year_no_age(mock_vault):
    _write_contact(mock_vault, "Pedro", "Cumpleaños: 10/05\n")
    result = birthday_today_signal(_NOW)
    assert len(result) == 1
    assert "Pedro" in result[0].message
    assert "años" not in result[0].message  # sin edad


def test_birthday_other_day_skipped(mock_vault):
    _write_contact(mock_vault, "Juan", "Cumpleaños: 11/05/1980\n")
    assert birthday_today_signal(_NOW) == []


def test_29feb_non_leap_pushed_on_feb28(mock_vault):
    _write_contact(mock_vault, "Leap", "Cumpleaños: 29/02/2000\n")
    result = birthday_today_signal(_FEB28_NON_LEAP)
    assert len(result) == 1
    assert "Leap" in result[0].message
    assert "29-feb" in result[0].message  # hint


# ── Skip self ────────────────────────────────────────────────────────────────


def test_skip_self_by_filename(mock_vault):
    """Filename `Yo.md` → skip aunque tenga cumple HOY."""
    _write_contact(mock_vault, "Yo", "Cumpleaños: 10/05/1981\n")
    assert birthday_today_signal(_NOW) == []


def test_skip_self_by_frontmatter_type(mock_vault):
    """Frontmatter `type: self` → skip."""
    _write_contact(mock_vault, "Random",
                   "Cumpleaños: 10/05/1981\n",
                   frontmatter="type: self\nname: Fernando")
    assert birthday_today_signal(_NOW) == []


def test_skip_self_by_relacion_yo(mock_vault):
    _write_contact(mock_vault, "Random",
                   "Cumpleaños: 10/05/1981\n",
                   frontmatter="relacion: yo")
    assert birthday_today_signal(_NOW) == []


# ── Skip _template ───────────────────────────────────────────────────────────


def test_skip_template_filename(mock_vault):
    _write_contact(mock_vault, "_template", "Cumpleaños: 10/05/1981\n")
    assert birthday_today_signal(_NOW) == []


# ── Multiple birthdays ───────────────────────────────────────────────────────


def test_multiple_birthdays_same_day_emit_each(mock_vault):
    _write_contact(mock_vault, "A", "Cumpleaños: 10/05/1980\n")
    _write_contact(mock_vault, "B", "Cumpleaños: 10/05/1985\n")
    _write_contact(mock_vault, "C", "Cumpleaños: 10/05/1990\n")
    result = birthday_today_signal(_NOW)
    assert len(result) == 3
    names = [c.message for c in result]
    assert any("A" in m for m in names)
    assert any("B" in m for m in names)
    assert any("C" in m for m in names)


def test_max_5_cap(mock_vault):
    """6 cumples → emit solo 5."""
    for i in range(6):
        _write_contact(mock_vault, f"Person{i}",
                       f"Cumpleaños: 10/05/{1980 + i}\n")
    result = birthday_today_signal(_NOW)
    assert len(result) == 5


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_contains_year(mock_vault):
    """dedup_key cambia con el año → re-pusheable año siguiente."""
    _write_contact(mock_vault, "Juan", "Cumpleaños: 10/05/1980\n")
    result = birthday_today_signal(_NOW)
    assert result[0].dedup_key == "birthday:Juan:2026"


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "birthday_today" in names
