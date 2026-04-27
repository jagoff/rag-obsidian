"""Tests for the 'streak_break' Anticipatory Agent signal.

Cubre:
- Sin 04-Archive/99-obsidian-system/99-AI/reviews dir → []
- Brief de hoy existe (gap=0) → []
- Brief de ayer + hoy no (gap=1) → []
- Último brief hace 2 días → emit score 0.4
- Último brief hace 5 días → emit score 1.0
- Último brief hace 10 días (fuera ventana 7d) → [] (pausa voluntaria)
- Archivos `-evening.md` NO cuentan como morning brief
- dedup_key estable por día
- Determinismo con `now` param distinto
- Signal registrado en el registry global

La signal es 100% filesystem — no DB, no retrieve(), no network. Aislamos
con `monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)` al
`tmp_path` que cada test construye.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.streak_break import (
    _count_gap_days,
    _find_last_morning_brief,
    streak_break_signal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Reference `now` compartido por todos los tests. Mediodía fijo para evitar
# problemas de borde en la conversión datetime→date.
_REF_NOW = datetime(2025, 6, 15, 12, 0, 0)


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Vault vacío + 04-Archive/99-obsidian-system/99-AI/reviews creado + `_resolve_vault_path` apuntando acá."""
    vault = tmp_path / "vault"
    (vault / "04-Archive/99-obsidian-system/99-AI/reviews").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


@pytest.fixture
def empty_vault(tmp_path, monkeypatch):
    """Vault SIN 04-Archive/99-obsidian-system/99-AI/reviews (para simular setup inicial)."""
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _write_brief(vault: Path, brief_date, body: str = "# morning brief\n\nstub.\n") -> Path:
    """Crea `04-Archive/99-obsidian-system/99-AI/reviews/{brief_date.isoformat()}.md` con body mínimo."""
    path = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / f"{brief_date.isoformat()}.md"
    path.write_text(body, encoding="utf-8")
    return path


def _write_evening(vault: Path, brief_date, body: str = "# evening brief\n") -> Path:
    """Crea `04-Archive/99-obsidian-system/99-AI/reviews/{brief_date.isoformat()}-evening.md` (NO debe contar)."""
    path = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / f"{brief_date.isoformat()}-evening.md"
    path.write_text(body, encoding="utf-8")
    return path


# ── Tests: estructura del vault ──────────────────────────────────────────────

def test_no_reviews_dir_returns_empty(empty_vault):
    """Sin folder 04-Archive/99-obsidian-system/99-AI/reviews → [] silenciosamente."""
    result = streak_break_signal(_REF_NOW)
    assert result == []


def test_empty_reviews_dir_returns_empty(mock_vault):
    """Con 04-Archive/99-obsidian-system/99-AI/reviews pero sin briefs → [] (pausa voluntaria)."""
    result = streak_break_signal(_REF_NOW)
    assert result == []


# ── Tests: gap normal (0-1 días) ─────────────────────────────────────────────

def test_brief_today_no_emit(mock_vault):
    """Brief escrito HOY (gap=0) → streak OK, no emit."""
    _write_brief(mock_vault, _REF_NOW.date())
    result = streak_break_signal(_REF_NOW)
    assert result == []


def test_brief_yesterday_no_emit(mock_vault):
    """Brief de AYER, ninguno hoy (gap=1) → descanso normal, no emit."""
    yesterday = _REF_NOW.date() - timedelta(days=1)
    _write_brief(mock_vault, yesterday)
    result = streak_break_signal(_REF_NOW)
    assert result == []


# ── Tests: gap ≥2 días (emit) ────────────────────────────────────────────────

def test_last_brief_2_days_ago_emits_score_04(mock_vault):
    """Último brief hace 2 días → emit con score 0.4 (2/5)."""
    two_days_ago = _REF_NOW.date() - timedelta(days=2)
    _write_brief(mock_vault, two_days_ago)
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-streak_break"
    assert c.score == pytest.approx(0.4, abs=0.01)
    assert c.snooze_hours == 24
    assert "2 días" in c.message
    assert "rag morning" in c.message


def test_last_brief_3_days_ago_emits_score_06(mock_vault):
    """Último brief hace 3 días → score 0.6 (3/5)."""
    three_days_ago = _REF_NOW.date() - timedelta(days=3)
    _write_brief(mock_vault, three_days_ago)
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.6, abs=0.01)
    assert "3 días" in result[0].message


def test_last_brief_5_days_ago_emits_score_10(mock_vault):
    """Último brief hace 5 días → score saturado a 1.0."""
    five_days_ago = _REF_NOW.date() - timedelta(days=5)
    _write_brief(mock_vault, five_days_ago)
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == pytest.approx(1.0, abs=0.01)
    assert "5 días" in result[0].message


def test_last_brief_7_days_ago_saturates_to_10(mock_vault):
    """Último brief hace 7 días (borde de la ventana) → aún en ventana, score 1.0."""
    seven_days_ago = _REF_NOW.date() - timedelta(days=7)
    _write_brief(mock_vault, seven_days_ago)
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    # min(1.0, 7/5) = 1.0
    assert result[0].score == pytest.approx(1.0, abs=0.01)


# ── Tests: fuera de ventana (pausa voluntaria) ───────────────────────────────

def test_last_brief_10_days_ago_no_emit(mock_vault):
    """Último brief hace 10 días (fuera ventana 7d) → [] (user pausó)."""
    ten_days_ago = _REF_NOW.date() - timedelta(days=10)
    _write_brief(mock_vault, ten_days_ago)
    result = streak_break_signal(_REF_NOW)
    assert result == []


def test_last_brief_30_days_ago_no_emit(mock_vault):
    """Brief muy viejo (30d atrás) → []. No spam post-vacaciones."""
    far_back = _REF_NOW.date() - timedelta(days=30)
    _write_brief(mock_vault, far_back)
    result = streak_break_signal(_REF_NOW)
    assert result == []


# ── Tests: filtros de filename ───────────────────────────────────────────────

def test_evening_briefs_do_not_count_as_morning(mock_vault):
    """Archivos `YYYY-MM-DD-evening.md` NO deben contar como morning brief.

    Si hay solo evening briefs en los últimos días, el signal debe actuar
    como si no hubiera morning briefs → [] (pausa voluntaria dado que no
    hay morning en ventana).
    """
    # Evening briefs en los últimos 5 días (no cuentan).
    for i in range(5):
        _write_evening(mock_vault, _REF_NOW.date() - timedelta(days=i))
    result = streak_break_signal(_REF_NOW)
    # Sin ningún morning brief en ventana → pausa voluntaria → [].
    assert result == []


def test_evening_ignored_gap_measured_from_morning_only(mock_vault):
    """Con morning de hace 3d y evening de hace 1d, el gap se mide contra morning."""
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=3))
    _write_evening(mock_vault, _REF_NOW.date() - timedelta(days=1))
    result = streak_break_signal(_REF_NOW)
    # Gap=3 (desde morning), no gap=1 (desde evening).
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.6, abs=0.01)
    assert "3 días" in result[0].message


def test_non_iso_md_files_ignored(mock_vault):
    """Archivos con nombres no-ISO (`notes.md`, `2025-W12.md`) no cuentan."""
    (mock_vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "notes.md").write_text("random", encoding="utf-8")
    (mock_vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2025-W12.md").write_text("weekly", encoding="utf-8")
    (mock_vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "README.md").write_text("readme", encoding="utf-8")
    result = streak_break_signal(_REF_NOW)
    # Sin morning briefs válidos → []
    assert result == []


def test_invalid_date_in_filename_ignored(mock_vault):
    """Filenames con fecha imposible (`2025-13-45.md`) se ignoran silenciosamente."""
    (mock_vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2025-13-45.md").write_text("bad", encoding="utf-8")
    result = streak_break_signal(_REF_NOW)
    assert result == []


# ── Tests: picking último brief con múltiples archivos ───────────────────────

def test_picks_most_recent_brief_in_window(mock_vault):
    """Con varios briefs en ventana, usa el más reciente para calcular el gap."""
    # Brief hace 6 días + brief hace 3 días. El gap debería ser 3 (no 6).
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=6))
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=3))
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.6, abs=0.01)  # 3/5
    assert "3 días" in result[0].message


# ── Tests: dedup_key estable por día ─────────────────────────────────────────

def test_dedup_key_stable_same_day(mock_vault):
    """Dos calls en el mismo día producen el mismo dedup_key."""
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=3))
    r1 = streak_break_signal(_REF_NOW)
    r2 = streak_break_signal(_REF_NOW + timedelta(hours=3))  # mismo día, otra hora
    assert len(r1) == 1
    assert len(r2) == 1
    assert r1[0].dedup_key == r2[0].dedup_key
    assert r1[0].dedup_key == "streak_break:2025-06-15"


def test_dedup_key_changes_next_day(mock_vault):
    """Día siguiente → dedup_key diferente (nuevo push permitido)."""
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=3))
    r_today = streak_break_signal(_REF_NOW)
    r_tomorrow = streak_break_signal(_REF_NOW + timedelta(days=1))
    assert len(r_today) == 1
    assert len(r_tomorrow) == 1
    assert r_today[0].dedup_key != r_tomorrow[0].dedup_key
    assert r_today[0].dedup_key == "streak_break:2025-06-15"
    assert r_tomorrow[0].dedup_key == "streak_break:2025-06-16"


def test_dedup_key_has_correct_prefix_and_iso_date(mock_vault):
    """dedup_key formato: `streak_break:YYYY-MM-DD`."""
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=2))
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].dedup_key.startswith("streak_break:")
    _, iso = result[0].dedup_key.split(":", 1)
    # Debe parsear como fecha ISO.
    assert len(iso) == 10
    assert iso[4] == "-" and iso[7] == "-"


# ── Tests: determinismo con `now` param ──────────────────────────────────────

def test_deterministic_with_different_now(mock_vault):
    """El `now` param controla la decisión: mismo brief, distinto `now` → distintos gaps."""
    brief_date = _REF_NOW.date() - timedelta(days=3)
    _write_brief(mock_vault, brief_date)

    # now = brief+3d → gap=3 → score 0.6
    now_a = datetime.combine(brief_date + timedelta(days=3), datetime.min.time().replace(hour=12))
    r_a = streak_break_signal(now_a)
    assert len(r_a) == 1
    assert r_a[0].score == pytest.approx(0.6, abs=0.01)

    # now = brief+1d → gap=1 → no emit
    now_b = datetime.combine(brief_date + timedelta(days=1), datetime.min.time().replace(hour=12))
    r_b = streak_break_signal(now_b)
    assert r_b == []

    # now = brief+5d → gap=5 → score 1.0
    now_c = datetime.combine(brief_date + timedelta(days=5), datetime.min.time().replace(hour=12))
    r_c = streak_break_signal(now_c)
    assert len(r_c) == 1
    assert r_c[0].score == pytest.approx(1.0, abs=0.01)


# ── Tests: silent-fail ───────────────────────────────────────────────────────

def test_silent_fail_on_bad_vault(monkeypatch):
    """Si `_resolve_vault_path` explota, la signal devuelve [] sin propagar."""
    def _boom():
        raise RuntimeError("vault config roto")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    result = streak_break_signal(_REF_NOW)
    assert result == []


def test_silent_fail_on_nonexistent_vault(tmp_path, monkeypatch):
    """Si el vault apunta a un dir inexistente, devuelve []."""
    nonexistent = tmp_path / "does-not-exist"
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: nonexistent)
    result = streak_break_signal(_REF_NOW)
    assert result == []


# ── Tests: helpers internos (unit) ───────────────────────────────────────────

def test_find_last_morning_brief_empty_vault(mock_vault):
    """Helper retorna None con 04-Archive/99-obsidian-system/99-AI/reviews vacío."""
    assert _find_last_morning_brief(mock_vault, _REF_NOW) is None


def test_find_last_morning_brief_picks_most_recent(mock_vault):
    """Helper devuelve la fecha más reciente en la ventana."""
    d1 = _REF_NOW.date() - timedelta(days=5)
    d2 = _REF_NOW.date() - timedelta(days=2)
    d3 = _REF_NOW.date() - timedelta(days=6)
    _write_brief(mock_vault, d1)
    _write_brief(mock_vault, d2)
    _write_brief(mock_vault, d3)
    result = _find_last_morning_brief(mock_vault, _REF_NOW)
    assert result == d2


def test_find_last_morning_brief_skips_out_of_window(mock_vault):
    """Helper ignora archivos con filename-date fuera de within_days."""
    old = _REF_NOW.date() - timedelta(days=20)
    _write_brief(mock_vault, old)
    assert _find_last_morning_brief(mock_vault, _REF_NOW, within_days=7) is None


def test_count_gap_days_basic():
    """Helper count_gap_days: today-last_brief con clamp a 0."""
    now = datetime(2025, 6, 15, 12, 0, 0)
    assert _count_gap_days(now.date(), now) == 0
    assert _count_gap_days(now.date() - timedelta(days=1), now) == 1
    assert _count_gap_days(now.date() - timedelta(days=5), now) == 5
    # Clock skew (brief "futuro") → clamp a 0.
    assert _count_gap_days(now.date() + timedelta(days=1), now) == 0


# ── Registry / integración ───────────────────────────────────────────────────

def test_signal_is_registered():
    """El decorator registró la signal en el SIGNALS global."""
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "streak_break" in names


def test_signal_visible_in_rag_anticipate_signals():
    """La signal aparece en `rag._ANTICIPATE_SIGNALS` (el tuple final del orchestrator)."""
    names = [label for (label, _fn) in rag._ANTICIPATE_SIGNALS]
    assert "streak_break" in names


def test_message_has_expected_shape(mock_vault):
    """Message contiene emoji 🔥, el gap en días, y la sugerencia de comando."""
    _write_brief(mock_vault, _REF_NOW.date() - timedelta(days=4))
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    msg = result[0].message
    assert "🔥" in msg
    assert "Racha rota" in msg
    assert "4 días" in msg
    assert "`rag morning`" in msg


def test_reason_contains_debug_info(mock_vault):
    """El field `reason` expone last_brief + gap_days para debug via --explain."""
    last = _REF_NOW.date() - timedelta(days=3)
    _write_brief(mock_vault, last)
    result = streak_break_signal(_REF_NOW)
    assert len(result) == 1
    r = result[0].reason
    assert last.isoformat() in r
    assert "gap_days=3" in r
