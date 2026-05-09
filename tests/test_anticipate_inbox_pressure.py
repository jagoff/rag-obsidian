"""Tests for the 'inbox_pressure' Anticipatory Agent signal.

Cubre:
- No inbox dir → []
- < THRESHOLD notas stale → []
- Exactamente THRESHOLD notas stale → emit con score base
- Score saturation con count alto
- Notas más frescas que MIN_AGE_HOURS no cuentan
- 00-Inbox/conversations/ con 50 archivos NO cuenta → []
- dedup_key incluye fecha de hoy
- Message incluye el count correcto
- Silent-fail si vault explota
- Registry check

El signal es filesystem-only (no retrieve, no DB). Aislamos el vault con
`monkeypatch.setattr(rag, "_resolve_vault_path", ...)` a un tmp_path que
construye cada test. Los thresholds (`_INBOX_EMIT_THRESHOLD`, `_INBOX_MIN_AGE_HOURS`,
`_INBOX_SCORE_RAMP`) se importan del módulo para que el test no se rompa
cuando se tunean los valores (ver bump 2026-05-09 que bajó threshold 15→10).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals import inbox_pressure as _inbox_mod
from rag_anticipate.signals.inbox_pressure import (
    _count_stale_inbox,
    inbox_pressure_signal,
)

_THRESHOLD = _inbox_mod._INBOX_EMIT_THRESHOLD
_MIN_AGE_HOURS = _inbox_mod._INBOX_MIN_AGE_HOURS
_SCORE_BASE = _inbox_mod._INBOX_SCORE_BASE
_SCORE_RAMP = _inbox_mod._INBOX_SCORE_RAMP
# Edad "fresca" garantizada por debajo del threshold (mitad).
_FRESH_HOURS = max(1.0, _MIN_AGE_HOURS / 2.0)
# Edad "stale" garantizada por encima del threshold (4× para evitar boundary).
_STALE_HOURS = max(_MIN_AGE_HOURS * 2.0, 48.0)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _age_note(path: Path, hours: float) -> None:
    """Setea mtime de `path` a hace `hours` horas vs `datetime.now()`.

    Usa `os.utime(atime, mtime)` — ambos al mismo ts para que `stat`
    refleje la edad en ambos campos.
    """
    ts = (datetime.now() - timedelta(hours=hours)).timestamp()
    os.utime(path, (ts, ts))


def _populate_inbox(inbox: Path, count: int, age_hours: float = 48.0) -> list[Path]:
    """Crea `count` notas `.md` en `inbox/` con edad uniforme `age_hours`.

    Retorna la lista de paths para que el caller pueda re-agingar notas
    individuales (ej. para el test del umbral de 24h).
    """
    paths: list[Path] = []
    for i in range(count):
        p = inbox / f"note-{i:03d}.md"
        p.write_text(f"# Note {i}\nContenido.", encoding="utf-8")
        _age_note(p, age_hours)
        paths.append(p)
    return paths


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Construye un vault tmp con `00-Inbox/` vacío y lo registra como activo.

    Retorna `(vault_root, inbox_dir)` para que el test pueble el inbox
    como necesite.
    """
    vault = tmp_path / "vault"
    inbox = vault / "00-Inbox"
    inbox.mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault, inbox


# ── Tests ────────────────────────────────────────────────────────────────────

def test_no_inbox_dir_returns_empty(tmp_path, monkeypatch):
    """Vault existe pero no tiene `00-Inbox/` → signal devuelve []."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_inbox_with_few_stale_notes_returns_empty(mock_vault):
    """Inbox con <THRESHOLD notas stale → no emit."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=max(1, _THRESHOLD - 1), age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_inbox_empty_returns_empty(mock_vault):
    """Inbox existe pero está vacío → []."""
    _vault, _inbox = mock_vault
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_inbox_with_exactly_threshold_emits_score_base(mock_vault):
    """Inbox con exactamente THRESHOLD notas stale → emit con score base."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=_THRESHOLD, age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-inbox_pressure"
    # (THRESHOLD - THRESHOLD) / RAMP + BASE = BASE
    assert c.score == pytest.approx(_SCORE_BASE, abs=0.01)
    assert c.snooze_hours == 48


def test_inbox_with_full_ramp_emits_score_1_0(mock_vault):
    """Inbox con THRESHOLD + RAMP - X notas → score saturado a 1.0."""
    _vault, inbox = mock_vault
    # Score = (count - THRESHOLD) / RAMP + BASE = 1.0 cuando count = THRESHOLD + RAMP * (1 - BASE)
    full_count = int(_THRESHOLD + _SCORE_RAMP * (1.0 - _SCORE_BASE)) + 5
    _populate_inbox(inbox, count=full_count, age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    assert result[0].score == pytest.approx(1.0, abs=0.01)


def test_inbox_score_clamped_to_1(mock_vault):
    """Más allá del ramp completo el score sigue clampeado a 1.0."""
    _vault, inbox = mock_vault
    full_count = int(_THRESHOLD + _SCORE_RAMP * 3)
    _populate_inbox(inbox, count=full_count, age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    assert result[0].score == pytest.approx(1.0, abs=0.01)


def test_inbox_with_only_one_stale_no_emit(mock_vault):
    """Mayoría de notas frescas (<MIN_AGE_HOURS) + 1 stale → solo 1 stale → no emit."""
    _vault, inbox = mock_vault
    total = max(_THRESHOLD + 5, 20)
    paths = _populate_inbox(inbox, count=total, age_hours=_STALE_HOURS)
    # Re-agingar todas menos la última a edad fresca.
    for p in paths[:-1]:
        _age_note(p, _FRESH_HOURS)
    # Solo la última queda stale. 1 < THRESHOLD → no emit.
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_conversations_subfolder_not_counted(mock_vault):
    """00-Inbox/conversations/ con 50 archivos NO cuenta (no recursivo).

    La signal solo mira top-level del inbox. La subfolder `conversations/`
    (episodic memory auto-generada por el web server) NO representa trabajo
    pendiente del user → no debe disparar presión.
    """
    vault, inbox = mock_vault
    # 50 archivos en conversations/ (episodic memory, NO triage).
    conv = inbox / "conversations"
    conv.mkdir()
    for i in range(50):
        p = conv / f"conv-{i:03d}.md"
        p.write_text(f"# Conv {i}\n", encoding="utf-8")
        _age_note(p, 48.0)
    # Solo 5 notas reales en top-level (menos del threshold).
    _populate_inbox(inbox, count=5, age_hours=48.0)
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_conversations_subfolder_ignored_even_when_threshold_met_elsewhere(mock_vault):
    """Conversations subfolder + inbox top-level ≥THRESHOLD → emit con count
    sólo del top-level, NO suma de las 50 conversations + top-level.

    Verifica que el count del message es solo el top-level."""
    _vault, inbox = mock_vault
    conv = inbox / "conversations"
    conv.mkdir()
    for i in range(50):
        p = conv / f"conv-{i:03d}.md"
        p.write_text(f"# Conv {i}\n", encoding="utf-8")
        _age_note(p, _STALE_HOURS)
    _populate_inbox(inbox, count=_THRESHOLD, age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    # Count debe ser THRESHOLD, no THRESHOLD+50.
    assert f"{_THRESHOLD} notas" in result[0].message


def test_dedup_key_includes_today_date(mock_vault):
    """dedup_key incluye la fecha de hoy (cambia por día, 1×/día max)."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=20, age_hours=48.0)
    now = datetime(2026, 5, 14, 9, 0, 0)
    result = inbox_pressure_signal(now)
    assert len(result) == 1
    assert result[0].dedup_key == "inbox_pressure:2026-05-14"


def test_dedup_key_differs_per_day(mock_vault):
    """Dos llamadas en días distintos producen dedup_keys distintos."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=20, age_hours=48.0)
    now1 = datetime(2026, 5, 14, 9, 0, 0)
    now2 = datetime(2026, 5, 15, 9, 0, 0)
    r1 = inbox_pressure_signal(now1)
    r2 = inbox_pressure_signal(now2)
    assert r1[0].dedup_key != r2[0].dedup_key
    assert "2026-05-14" in r1[0].dedup_key
    assert "2026-05-15" in r2[0].dedup_key


def test_message_includes_count(mock_vault):
    """El message contiene el count exacto de notas stale."""
    _vault, inbox = mock_vault
    count_test = _THRESHOLD + 12
    _populate_inbox(inbox, count=count_test, age_hours=_STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    msg = result[0].message
    assert f"{count_test} notas" in msg
    assert f">{_MIN_AGE_HOURS}h" in msg
    assert "📥" in msg
    assert "rag inbox --apply" in msg


def test_non_md_files_not_counted(mock_vault):
    """Archivos no-.md en el inbox no cuentan."""
    _vault, inbox = mock_vault
    # Bajo threshold de archivos .md stale.
    _populate_inbox(inbox, count=max(1, _THRESHOLD - 1), age_hours=_STALE_HOURS)
    # 20 archivos .txt / .pdf stale (shouldn't count).
    for i in range(20):
        p = inbox / f"attachment-{i:03d}.pdf"
        p.write_text("binario", encoding="utf-8")
        _age_note(p, _STALE_HOURS)
    result = inbox_pressure_signal(datetime.now())
    # Solo .md stale (bajo threshold) → no emit.
    assert result == []


def test_count_stale_inbox_helper_direct(mock_vault):
    """El helper `_count_stale_inbox` devuelve el count esperado."""
    vault, inbox = mock_vault
    total = 20
    paths = _populate_inbox(inbox, count=total, age_hours=_STALE_HOURS)
    # Todas stale.
    assert _count_stale_inbox(vault) == total
    # Fresh-ear 5 → quedan total-5 stale.
    for p in paths[:5]:
        _age_note(p, _FRESH_HOURS)
    assert _count_stale_inbox(vault) == total - 5


def test_count_stale_inbox_respects_min_age_hours(mock_vault):
    """El helper acepta un `min_age_hours` custom."""
    vault, inbox = mock_vault
    _populate_inbox(inbox, count=10, age_hours=48.0)
    # A 24h todas cuentan (48 > 24).
    assert _count_stale_inbox(vault, min_age_hours=24) == 10
    # A 72h ninguna cuenta (48 < 72).
    assert _count_stale_inbox(vault, min_age_hours=72) == 0


def test_silent_fail_on_bad_vault(monkeypatch):
    """Si `_resolve_vault_path` explota, la signal devuelve [] silenciosamente."""
    def _boom():
        raise RuntimeError("vault config roto")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_silent_fail_on_nonexistent_vault(monkeypatch, tmp_path):
    """Si el vault apunta a un dir que no existe → []."""
    nonexistent = tmp_path / "does-not-exist"
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: nonexistent)
    result = inbox_pressure_signal(datetime.now())
    assert result == []


def test_signal_is_registered():
    """Sanity check: el decorator registró la signal en el registry global."""
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "inbox_pressure" in names


def test_signal_in_rag_anticipate_signals_tuple():
    """La signal aparece en el tuple global `rag._ANTICIPATE_SIGNALS`."""
    import rag as _rag
    names = [n for (n, _fn) in _rag._ANTICIPATE_SIGNALS]
    assert "inbox_pressure" in names


def test_max_one_candidate(mock_vault):
    """Siempre máximo 1 candidate, nunca más."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=100, age_hours=48.0)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1


def test_returns_anticipatory_candidate_shape(mock_vault):
    """El candidate retornado tiene los campos esperados del dataclass."""
    _vault, inbox = mock_vault
    _populate_inbox(inbox, count=20, age_hours=48.0)
    result = inbox_pressure_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    # Todos los campos del dataclass presentes y con tipos correctos.
    assert isinstance(c.kind, str)
    assert isinstance(c.score, float)
    assert 0.0 <= c.score <= 1.0
    assert isinstance(c.message, str) and c.message
    assert isinstance(c.dedup_key, str) and c.dedup_key
    assert isinstance(c.snooze_hours, int)
    assert isinstance(c.reason, str)
