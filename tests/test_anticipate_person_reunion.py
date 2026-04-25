"""Tests for the `person_reunion` anticipatory signal.

Cubre:
1. Empty vault → []
2. Nota de hoy sin wikilinks → []
3. [[Juan]] mencionado hoy pero también en otra nota reciente (<30d) → []
4. [[Juan]] mencionado hoy y la única nota previa fue hace 60d → candidate
5. Score escala con gap (180d vs 60d)
6. Wikilinks no-persona (minúscula, path, dígitos) → ignored
7. dedup_key estable cross-runs
8. Múltiples personas → máximo 2, las de mayor gap ganan

Mocks: monkeypatch `rag._resolve_vault_path` + `rag.VAULT_PATH`, `os.utime`
para fijar mtimes de los archivos del fixture.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pytest

import rag
from rag_anticipate.signals.person_reunion import (
    _extract_capitalized_wikilinks,
    _find_last_mention_before,
    person_reunion_signal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Vault temporal aislado. El signal lee `_resolve_vault_path()` y
    `VAULT_PATH` — mockeamos ambos para apuntar al tmp_path."""
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def _touch(path, *, days_ago: float = 0.0, hours_ago: float = 0.0):
    """Helper — fija mtime al pasado relativo a now."""
    ts = time.time() - days_ago * 86400.0 - hours_ago * 3600.0
    os.utime(path, (ts, ts))
    return ts


def _write(vault, rel, body, *, days_ago=0.0, hours_ago=0.0):
    """Crea una nota con body + mtime en el pasado."""
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    _touch(p, days_ago=days_ago, hours_ago=hours_ago)
    return p


def _pad(body, min_chars=400):
    """Padding — el signal exige ≥300 chars; agregamos filler neutro."""
    filler = "\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. " * 20
    return body + filler[: max(0, min_chars - len(body))]


# ── Extractor helpers ────────────────────────────────────────────────────────


def test_extract_capitalized_wikilinks_happy_path():
    text = "Hoy vi a [[Juan Perez]] y a [[Maria]] en la reunión."
    assert _extract_capitalized_wikilinks(text) == ["Juan Perez", "Maria"]


def test_extract_capitalized_wikilinks_filters_non_persons():
    text = (
        "Referencias: [[juan]] [[some-tag]] [[2024-05-01]] "
        "[[folder/note]] [[A Very Long Title With Way Too Many Words Here]] "
        "[[Juan]]"
    )
    # minúscula, slug, fecha, path, >4 palabras → filtrados
    # solo queda [[Juan]]
    assert _extract_capitalized_wikilinks(text) == ["Juan"]


def test_extract_dedup_preserves_order():
    text = "[[Ana]] habla con [[Bruno]] y después [[Ana]] otra vez."
    assert _extract_capitalized_wikilinks(text) == ["Ana", "Bruno"]


# ── Signal tests ─────────────────────────────────────────────────────────────


def test_empty_vault_returns_empty(mock_vault):
    """1. Vault sin notas → []."""
    out = person_reunion_signal(datetime.now())
    assert out == []


def test_today_note_without_wikilinks(mock_vault):
    """2. Nota reciente sin wikilinks → []."""
    _write(
        mock_vault,
        "today.md",
        _pad("Hoy fue un día tranquilo. Sin menciones."),
        hours_ago=1,
    )
    out = person_reunion_signal(datetime.now())
    assert out == []


def test_person_mentioned_recently_is_skipped(mock_vault):
    """3. [[Juan]] en today + en otra nota de hace 20d + en nota de 60d.
    La mención más reciente previa es 20d < 30d threshold → []."""
    _write(
        mock_vault,
        "02-Areas/today.md",
        _pad("Reunión con [[Juan]] sobre el roadmap."),
        hours_ago=2,
    )
    _write(
        mock_vault,
        "02-Areas/recent-juan.md",
        _pad("Nota previa: hablé con [[Juan]] sobre otro tema."),
        days_ago=20,
    )
    _write(
        mock_vault,
        "02-Areas/old-juan.md",
        _pad("Hace mucho: mencioné a [[Juan]]."),
        days_ago=60,
    )
    out = person_reunion_signal(datetime.now())
    assert out == []


def test_person_with_long_gap_emits_candidate(mock_vault):
    """4. [[Juan]] hoy + última mención hace 60d → candidate emitido."""
    _write(
        mock_vault,
        "02-Areas/today.md",
        _pad("Re-conecté con [[Juan]] después de mucho."),
        hours_ago=1,
    )
    _write(
        mock_vault,
        "02-Areas/old-juan.md",
        _pad("Hace tiempo hablé con [[Juan]]."),
        days_ago=60,
    )
    out = person_reunion_signal(datetime.now())
    assert len(out) == 1
    c = out[0]
    assert c.kind == "anticipate-person_reunion"
    assert "Juan" in c.message
    assert "today" in c.message
    assert c.dedup_key == "reunion:Juan:02-Areas/today.md"
    assert c.snooze_hours == 72
    # 60d / 180d = 0.333...
    assert 0.3 < c.score < 0.4


def test_score_scales_with_gap(mock_vault, tmp_path):
    """5. Gap de 180d debería dar score 1.0; gap de 60d ~0.33."""
    # Escenario A: gap 60d
    _write(
        mock_vault,
        "today.md",
        _pad("Hoy: [[Carla]] vuelve."),
        hours_ago=1,
    )
    _write(
        mock_vault,
        "old-carla-60.md",
        _pad("Mención vieja de [[Carla]]."),
        days_ago=60,
    )
    out60 = person_reunion_signal(datetime.now())
    assert len(out60) == 1
    score60 = out60[0].score

    # Escenario B: gap 180d (reemplazamos la vieja)
    (mock_vault / "old-carla-60.md").unlink()
    _write(
        mock_vault,
        "old-carla-180.md",
        _pad("Mención MUY vieja de [[Carla]]."),
        days_ago=180,
    )
    out180 = person_reunion_signal(datetime.now())
    assert len(out180) == 1
    score180 = out180[0].score

    # El de 180d debe ser mayor, y ~1.0
    assert score180 > score60
    assert score180 == pytest.approx(1.0, abs=0.01)
    assert score60 == pytest.approx(60.0 / 180.0, abs=0.02)


def test_non_person_wikilinks_are_ignored(mock_vault):
    """6. Wikilinks que no son nombres propios (minúsculas, paths,
    fechas) no disparan reuniones aunque haya notas viejas con ellos."""
    _write(
        mock_vault,
        "today.md",
        _pad(
            "Hoy reviso [[proyecto-x]], [[2024-05-01]] y "
            "[[04-Archive/stuff]]."
        ),
        hours_ago=1,
    )
    _write(
        mock_vault,
        "old-stuff.md",
        _pad("Vieja nota con [[proyecto-x]] y [[2024-05-01]]."),
        days_ago=120,
    )
    out = person_reunion_signal(datetime.now())
    assert out == []


def test_dedup_key_is_stable_across_runs(mock_vault):
    """7. Dos runs con mismo estado → misma dedup_key."""
    _write(
        mock_vault,
        "02-Areas/today.md",
        _pad("Volvió [[Diego]]."),
        hours_ago=1,
    )
    _write(
        mock_vault,
        "02-Areas/old-diego.md",
        _pad("Hace rato: [[Diego]]."),
        days_ago=90,
    )
    out1 = person_reunion_signal(datetime.now())
    out2 = person_reunion_signal(datetime.now())
    assert len(out1) == 1 and len(out2) == 1
    assert out1[0].dedup_key == out2[0].dedup_key
    assert out1[0].dedup_key == "reunion:Diego:02-Areas/today.md"


def test_multiple_persons_caps_at_two_by_gap(mock_vault):
    """8. Tres personas con gaps 40d / 90d / 150d → devuelve las 2 de
    mayor gap (150d y 90d), ignora la de 40d."""
    _write(
        mock_vault,
        "today.md",
        _pad("Hoy: [[Ana]], [[Bruno]] y [[Carla]] reaparecieron."),
        hours_ago=1,
    )
    _write(mock_vault, "old-ana.md", _pad("[[Ana]] antigua."), days_ago=40)
    _write(mock_vault, "old-bruno.md", _pad("[[Bruno]] antiguo."), days_ago=90)
    _write(mock_vault, "old-carla.md", _pad("[[Carla]] vieja."), days_ago=150)

    out = person_reunion_signal(datetime.now())
    assert len(out) == 2
    persons = {c.dedup_key.split(":")[1] for c in out}
    assert persons == {"Bruno", "Carla"}
    # Ordenadas por gap descendente: Carla (150d) primero, Bruno (90d) después
    assert out[0].dedup_key.split(":")[1] == "Carla"
    assert out[1].dedup_key.split(":")[1] == "Bruno"
    assert out[0].score > out[1].score


# ── Helper-level tests (smoke) ───────────────────────────────────────────────


def test_find_last_mention_before_returns_none_when_no_match(mock_vault):
    """Smoke de `_find_last_mention_before` — vault vacío o sin mención."""
    prev = _find_last_mention_before(mock_vault, "Nobody", time.time())
    assert prev is None


def test_find_last_mention_before_finds_latest(mock_vault):
    """Dos notas viejas con [[Eva]] — devuelve la de mayor mtime < cutoff."""
    _write(mock_vault, "old-eva-1.md", _pad("[[Eva]]."), days_ago=100)
    _write(mock_vault, "old-eva-2.md", _pad("[[Eva]]."), days_ago=50)
    cutoff = time.time() - 1  # cualquier archivo antes de ahora
    prev = _find_last_mention_before(mock_vault, "Eva", cutoff)
    assert prev is not None
    rel, mtime = prev
    assert rel == "old-eva-2.md"
    # ~50 días atrás
    assert (time.time() - mtime) / 86400.0 == pytest.approx(50.0, abs=1.0)
