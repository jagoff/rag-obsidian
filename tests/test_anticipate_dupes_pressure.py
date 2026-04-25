"""Tests for the 'dupes_pressure' Anticipatory Agent signal.

Cubre:
- Empty vault → []
- 3 notas distintas → []
- 3 pares similares (abajo del threshold) → []
- ≥5 pares similares → emit
- Score escala con count
- dedup_key estable por semana ISO (y distinto entre semanas)
- Pares en folders excluidos (.obsidian/, 00-Inbox/conversations/) → skip
- Silent-fail si vault explota
- Registry checks

La signal es filesystem-only (no retrieve, no embed, no DB). Aislamos
el vault con `monkeypatch.setattr(rag, "_resolve_vault_path", ...)` a un
tmp_path que construye cada test.

Nota sobre macOS case-insensitive FS: los tests NO dependen de que
"ikigai.md" y "Ikigai.md" coexistan como archivos distintos (en HFS+/APFS
default se resuelven al mismo archivo). Usamos variantes con separadores
distintos (`-`, `_`, ` `, `.`, `--`, `__`) que SON bytewise-distintos en
el filesystem y todos normalizan al mismo stem canónico.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.dupes_pressure import (
    _find_title_similar_pairs,
    _normalize_stem,
    dupes_pressure_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_notes(folder: Path, names: list[str]) -> list[Path]:
    """Crea archivos `.md` con los nombres dados en `folder`. Retorna paths."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = folder / n
        p.write_text(f"# {n}\nBody.", encoding="utf-8")
        out.append(p)
    return out


# 7 variantes del stem "coachingnotes" que SON bytewise-distintas en
# cualquier filesystem (se diferencian en separadores, no en case) y
# todas normalizan al mismo stem canónico. C(7,2) = 21 pares.
_SEVEN_COACHING_VARIANTS = [
    "coaching-notes.md",
    "coaching_notes.md",
    "coaching notes.md",
    "coaching.notes.md",
    "coachingnotes.md",
    "coaching--notes.md",
    "coaching__notes.md",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Vault tmp con `02-Areas/` vacío, registrado como activo."""
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


# ── Tests ────────────────────────────────────────────────────────────────────


def test_normalize_stem_collapses_separators():
    """Separadores heterogéneos colapsan al mismo stem canónico."""
    assert _normalize_stem("coaching-notes") == "coachingnotes"
    assert _normalize_stem("coaching_notes") == "coachingnotes"
    assert _normalize_stem("coaching notes") == "coachingnotes"
    assert _normalize_stem("coaching.notes") == "coachingnotes"
    assert _normalize_stem("Coaching-Notes") == "coachingnotes"


def test_empty_vault_returns_empty(mock_vault):
    """Vault sin archivos `.md` → signal devuelve []."""
    result = dupes_pressure_signal(datetime.now())
    assert result == []


def test_three_distinct_notes_no_emit(mock_vault):
    """3 notas con títulos completamente distintos → 0 pares → []."""
    _write_notes(mock_vault / "02-Areas", [
        "ikigai.md",
        "macroeconomía.md",
        "routine.md",
    ])
    result = dupes_pressure_signal(datetime.now())
    assert result == []


def test_three_similar_pairs_below_threshold_no_emit(mock_vault):
    """3 archivos que normalizan al mismo stem → C(3,2)=3 pares < 5 → []."""
    _write_notes(mock_vault / "02-Areas", [
        "coaching-notes.md",
        "coaching_notes.md",
        "coaching notes.md",
    ])
    pairs = _find_title_similar_pairs(mock_vault)
    assert len(pairs) == 3  # sanity: helper ve 3 pares
    result = dupes_pressure_signal(datetime.now())
    assert result == []


def test_six_pairs_from_four_variants_emits(mock_vault):
    """4 archivos mismo stem canónico → C(4,2)=6 pares ≥ 5 → emit."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    pairs = _find_title_similar_pairs(mock_vault)
    assert len(pairs) == 6
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-dupes_pressure"
    # score = (6 - 5) / 15 + 0.5 ≈ 0.567
    assert c.score == pytest.approx(0.5 + 1 / 15.0, abs=0.01)
    assert c.snooze_hours == 336
    assert "6 pares" in c.message
    assert "rag dupes --threshold 0.85" in c.message
    assert "👥" in c.message


def test_twenty_one_pairs_saturates_score_to_1(mock_vault):
    """7 archivos mismo stem → C(7,2)=21 pares → score saturado a 1.0."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS)
    pairs = _find_title_similar_pairs(mock_vault)
    assert len(pairs) == 21
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1
    # (21-5)/15 + 0.5 = 1.567 → clamp → 1.0
    assert result[0].score == pytest.approx(1.0, abs=0.001)


def test_score_scales_monotonically(mock_vault, monkeypatch):
    """El score crece con el count: 6 < 12 pares → score(6) < score(12)."""
    # 6 pares con 4 variantes.
    folder = mock_vault / "02-Areas"
    _write_notes(folder, _SEVEN_COACHING_VARIANTS[:4])
    r_low = dupes_pressure_signal(datetime.now())
    assert len(r_low) == 1
    low_score = r_low[0].score

    # Agregar 1 archivo más → 5 variantes → C(5,2)=10 pares.
    _write_notes(folder, [_SEVEN_COACHING_VARIANTS[4]])
    r_mid = dupes_pressure_signal(datetime.now())
    assert len(r_mid) == 1
    mid_score = r_mid[0].score

    # Otro más → 6 variantes → C(6,2)=15 pares.
    _write_notes(folder, [_SEVEN_COACHING_VARIANTS[5]])
    r_high = dupes_pressure_signal(datetime.now())
    assert len(r_high) == 1
    high_score = r_high[0].score

    assert low_score < mid_score <= high_score
    assert high_score == pytest.approx(1.0, abs=0.001)  # 15→1.0


def test_dedup_key_includes_iso_week(mock_vault):
    """dedup_key contiene el año ISO y el número de semana."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    # 2026-05-14 es jueves de la semana ISO 20 del 2026.
    now = datetime(2026, 5, 14, 9, 0, 0)
    result = dupes_pressure_signal(now)
    assert len(result) == 1
    assert result[0].dedup_key == "dupes_pressure:2026-W20"


def test_dedup_key_stable_within_same_week(mock_vault):
    """Dos llamadas en distintos días de la MISMA semana ISO → misma dedup_key."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    # Lunes 2026-05-11 y viernes 2026-05-15 son ambos de la semana ISO 20.
    monday = datetime(2026, 5, 11, 9, 0, 0)
    friday = datetime(2026, 5, 15, 9, 0, 0)
    r1 = dupes_pressure_signal(monday)
    r2 = dupes_pressure_signal(friday)
    assert r1[0].dedup_key == r2[0].dedup_key
    assert r1[0].dedup_key == "dupes_pressure:2026-W20"


def test_dedup_key_differs_across_weeks(mock_vault):
    """Dos llamadas en semanas ISO distintas → dedup_keys distintas."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    wk20 = datetime(2026, 5, 14, 9, 0, 0)  # W20
    wk21 = datetime(2026, 5, 21, 9, 0, 0)  # W21
    r1 = dupes_pressure_signal(wk20)
    r2 = dupes_pressure_signal(wk21)
    assert r1[0].dedup_key != r2[0].dedup_key
    assert "W20" in r1[0].dedup_key
    assert "W21" in r2[0].dedup_key


def test_excluded_folders_are_skipped(mock_vault):
    """Pares en `.obsidian/` y `00-Inbox/conversations/` NO cuentan.

    Aunque pongamos 7 variantes del mismo stem en cada carpeta excluida
    (21 pares cada una, 42 pares totales), el signal debe devolver []
    porque `is_excluded(rel)` filtra ambos prefixes del scan.
    """
    # 7 variantes en `.obsidian/` (excluido por dotfolder rule).
    _write_notes(mock_vault / ".obsidian", _SEVEN_COACHING_VARIANTS)
    # 7 variantes en `00-Inbox/conversations/` (excluido explícitamente).
    _write_notes(
        mock_vault / "00-Inbox" / "conversations",
        _SEVEN_COACHING_VARIANTS,
    )
    # Un único archivo legítimo en 02-Areas (no genera pares por sí solo).
    _write_notes(mock_vault / "02-Areas", ["routine.md"])

    pairs = _find_title_similar_pairs(mock_vault)
    assert pairs == []
    result = dupes_pressure_signal(datetime.now())
    assert result == []


def test_excluded_folder_pairs_do_not_inflate_count(mock_vault):
    """Mix: 4 variantes en 02-Areas (6 pares, emite) + 7 en `.obsidian/`
    (deberían ser ignorados). El count reportado en el message es 6, no 27.
    """
    _write_notes(mock_vault / ".obsidian", _SEVEN_COACHING_VARIANTS)
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1
    assert "6 pares" in result[0].message


def test_similar_suffix_pairs_match(mock_vault):
    """Pares que difieren por sufijo ("-v1" vs "-v2") matchean vía
    SequenceMatcher ratio ≥0.85."""
    _write_notes(mock_vault / "02-Areas", [
        "proyecto-alfa-v1.md",
        "proyecto-alfa-v2.md",
        "proyecto-alfa-v3.md",
        "proyecto-alfa-v4.md",
    ])
    pairs = _find_title_similar_pairs(mock_vault)
    # C(4,2)=6 pares si todos matchean entre sí.
    assert len(pairs) == 6
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1


def test_silent_fail_on_bad_vault(monkeypatch):
    """`_resolve_vault_path` rompe → signal devuelve [] silenciosamente."""
    def _boom():
        raise RuntimeError("vault config roto")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    assert dupes_pressure_signal(datetime.now()) == []


def test_silent_fail_on_nonexistent_vault(tmp_path, monkeypatch):
    """Vault inexistente → []."""
    monkeypatch.setattr(
        rag, "_resolve_vault_path", lambda: tmp_path / "does-not-exist"
    )
    assert dupes_pressure_signal(datetime.now()) == []


def test_max_one_candidate(mock_vault):
    """Siempre máximo 1 candidate, nunca más."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS)
    # Agregar otro grupo de pares (proyectos alfa).
    _write_notes(mock_vault / "01-Projects", [
        "proyecto-alfa-v1.md",
        "proyecto-alfa-v2.md",
        "proyecto-alfa-v3.md",
        "proyecto-alfa-v4.md",
    ])
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1


def test_returns_anticipatory_candidate_shape(mock_vault):
    """Shape del dataclass: kind, score ∈ [0,1], message, dedup_key,
    snooze_hours=336, reason."""
    _write_notes(mock_vault / "02-Areas", _SEVEN_COACHING_VARIANTS[:4])
    result = dupes_pressure_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-dupes_pressure"
    assert isinstance(c.score, float) and 0.0 <= c.score <= 1.0
    assert isinstance(c.message, str) and c.message
    assert isinstance(c.dedup_key, str) and c.dedup_key.startswith("dupes_pressure:")
    assert c.snooze_hours == 336
    assert isinstance(c.reason, str) and "pairs=" in c.reason


def test_non_md_files_not_counted(mock_vault):
    """Archivos no-.md con nombres similares NO cuentan."""
    folder = mock_vault / "02-Areas"
    folder.mkdir(parents=True, exist_ok=True)
    for v in _SEVEN_COACHING_VARIANTS:
        # Escribir como .txt en vez de .md.
        (folder / v.replace(".md", ".txt")).write_text("x")
    assert _find_title_similar_pairs(mock_vault) == []
    assert dupes_pressure_signal(datetime.now()) == []


def test_signal_is_registered():
    """Sanity: el decorator registró la signal en el registry global."""
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "dupes_pressure" in names


def test_signal_in_rag_anticipate_signals_tuple():
    """La signal aparece en el tuple global `rag._ANTICIPATE_SIGNALS`."""
    import rag as _rag
    names = [n for (n, _fn) in _rag._ANTICIPATE_SIGNALS]
    assert "dupes_pressure" in names
