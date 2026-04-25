"""Tests for the 'orphan_surface' Anticipatory Agent signal.

Cubre:
1. Empty vault → []
2. Nota reciente CON wikilinks → [] (no es orphan)
3. Nota reciente SIN wikilinks + ≥200 chars + en 02-Areas → emit
4. Nota muy vieja (>24h) sin links → [] (fuera de ventana)
5. Nota muy nueva (<2h) sin links → [] (grace period)
6. Nota en 00-Inbox sin links → [] (bucket excluido)
7. Score escala con size (500 vs 3000 chars)
8. Multiple orphans → máximo 2, ordenados por tamaño desc
9. dedup_key estable cross-calls
+ message format, silent-fail y registration sanity check.

El signal camina el vault filesystem-only — no retrieve(), no embeddings.
Aislamos con `monkeypatch.setattr(rag, "_resolve_vault_path", ...)` a un
`tmp_path` construido por fixture.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.orphan_surface import (
    _count_outgoing_wikilinks,
    orphan_surface_signal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Reference `now` compartido — evita drift entre el mtime seteado por
# `os.utime` y la llamada a la signal. La signal recibe `now` como
# parámetro explícito → usamos este valor literal en todos los tests.
_REF_NOW = datetime(2025, 6, 15, 12, 0, 0)


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Construye un vault tmp con los 3 buckets PARA + 00-Inbox y lo
    registra como el vault activo.

    Monkeypatchea tanto `_resolve_vault_path` como `VAULT_PATH` para
    cubrir las dos rutas que podrían usar las helpers internas de rag.py.
    """
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    (vault / "01-Projects").mkdir(parents=True)
    (vault / "03-Resources").mkdir(parents=True)
    (vault / "00-Inbox").mkdir(parents=True)

    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def _make_note(vault: Path, rel: str, body: str, hours_old: float) -> Path:
    """Crea una nota con mtime = `_REF_NOW - hours_old`.

    Usa `os.utime(path, (ts, ts))` donde `ts = (now - timedelta(hours=N)).timestamp()`.
    Crea dirs intermedios si hace falta.
    """
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    ts = (_REF_NOW - timedelta(hours=hours_old)).timestamp()
    os.utime(path, (ts, ts))
    return path


def _body(n_chars: int, with_links: bool = False) -> str:
    """Genera un body de ~n_chars. Opcional: incluir un wikilink."""
    base = "Contenido sustantivo de la nota. " * ((n_chars // 32) + 1)
    body = base[:n_chars]
    if with_links:
        body = "Ver [[otra-nota]] para contexto.\n\n" + body
    return body


# ── Helper unit tests ────────────────────────────────────────────────────────

def test_count_outgoing_wikilinks_zero():
    assert _count_outgoing_wikilinks("sin links acá") == 0


def test_count_outgoing_wikilinks_multiple():
    # Nota: el helper de rag (`_extract_wikilinks_from_markdown`) excluye `#`
    # del target del wikilink (no matchea `[[foo#section]]`). Probamos los
    # dos casos que sí soporta: simple + con alias `|`.
    text = "Ver [[nota-uno]] y también [[nota-dos|alias]]."
    assert _count_outgoing_wikilinks(text) == 2


# ── Signal tests ─────────────────────────────────────────────────────────────

def test_empty_vault_returns_empty(mock_vault):
    """Vault sin notas → []."""
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_recent_note_with_links_is_not_orphan(mock_vault):
    """Nota en la ventana CON wikilinks → no es orphan, no emit."""
    _make_note(
        mock_vault,
        "02-Areas/con-links.md",
        _body(500, with_links=True),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_recent_orphan_in_areas_emits(mock_vault):
    """Nota 5h de antigüedad, sin links, ≥200 chars en 02-Areas → emit."""
    _make_note(
        mock_vault,
        "02-Areas/huerfana.md",
        _body(600),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-orphan_surface"
    assert c.snooze_hours == 24
    assert c.dedup_key == "orphan:02-Areas/huerfana.md"
    assert "[[huerfana]]" in c.message
    # ≥500 chars → score 0.6.
    assert c.score == pytest.approx(0.6, abs=0.01)


def test_old_note_out_of_window_no_emit(mock_vault):
    """Nota con mtime >24h atrás → fuera de ventana superior, no emit."""
    _make_note(
        mock_vault,
        "02-Areas/vieja.md",
        _body(600),
        hours_old=48.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_very_recent_note_in_grace_period_no_emit(mock_vault):
    """Nota con mtime <2h → dentro del grace period, no emit."""
    _make_note(
        mock_vault,
        "02-Areas/recien-guardada.md",
        _body(600),
        hours_old=0.5,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_note_in_inbox_is_skipped(mock_vault):
    """Nota orphan en 00-Inbox → skip (bucket excluido)."""
    _make_note(
        mock_vault,
        "00-Inbox/quick-note.md",
        _body(600),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_note_below_min_chars_is_skipped(mock_vault):
    """Nota orphan pero <200 chars → skip (fragment trivial)."""
    _make_note(
        mock_vault,
        "02-Areas/fragmento.md",
        "Muy corta.",  # ~10 chars
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_score_scales_with_size(mock_vault):
    """Score escala por bands: 500 → 0.6, 1500 → 0.8, 3000 → 0.9."""
    # 600 chars → banda ≥500 → 0.6
    _make_note(mock_vault, "02-Areas/medium.md", _body(600), hours_old=5.0)
    r = orphan_surface_signal(_REF_NOW)
    assert len(r) == 1 and r[0].score == pytest.approx(0.6, abs=0.01)

    # Ahora bumpear a 3500 chars → banda ≥3000 → 0.9
    _make_note(mock_vault, "02-Areas/medium.md", _body(3500), hours_old=5.0)
    r = orphan_surface_signal(_REF_NOW)
    assert len(r) == 1 and r[0].score == pytest.approx(0.9, abs=0.01)


def test_score_band_1500(mock_vault):
    """1500 chars cae en la banda intermedia → 0.8."""
    _make_note(mock_vault, "02-Areas/big.md", _body(1800), hours_old=5.0)
    r = orphan_surface_signal(_REF_NOW)
    assert len(r) == 1
    assert r[0].score == pytest.approx(0.8, abs=0.01)


def test_score_band_below_500(mock_vault):
    """Nota entre 200 y 499 chars → pasa el filtro pero score bajo (0.4)."""
    _make_note(mock_vault, "02-Areas/justita.md", _body(300), hours_old=5.0)
    r = orphan_surface_signal(_REF_NOW)
    assert len(r) == 1
    assert r[0].score == pytest.approx(0.4, abs=0.01)


def test_multiple_orphans_max_two_ordered_by_size(mock_vault):
    """Con ≥3 orphans en ventana → devuelve MÁXIMO 2, las más grandes primero."""
    _make_note(mock_vault, "02-Areas/chica.md",  _body(300),  hours_old=5.0)
    _make_note(mock_vault, "02-Areas/media.md",  _body(1800), hours_old=6.0)
    _make_note(mock_vault, "01-Projects/enorme.md", _body(4000), hours_old=7.0)
    _make_note(mock_vault, "03-Resources/grande.md", _body(2500), hours_old=8.0)

    result = orphan_surface_signal(_REF_NOW)
    assert len(result) == 2
    # Primero la más grande (4000 chars), después la segunda (2500 chars).
    assert "[[enorme]]" in result[0].message
    assert "[[grande]]" in result[1].message
    # Scores consistentes con las bands.
    assert result[0].score == pytest.approx(0.9, abs=0.01)
    assert result[1].score == pytest.approx(0.8, abs=0.01)


def test_dedup_key_stable_cross_calls(mock_vault):
    """Dos llamadas con el mismo estado producen el mismo dedup_key."""
    _make_note(
        mock_vault,
        "03-Resources/stable.md",
        _body(600),
        hours_old=5.0,
    )
    r1 = orphan_surface_signal(_REF_NOW)
    r2 = orphan_surface_signal(_REF_NOW)
    assert len(r1) == 1 and len(r2) == 1
    assert r1[0].dedup_key == r2[0].dedup_key
    assert r1[0].dedup_key == "orphan:03-Resources/stable.md"


def test_message_format(mock_vault):
    """El message tiene el shape esperado — emoji, wikilink, tamaño, sugerencia."""
    _make_note(
        mock_vault,
        "02-Areas/formato.md",
        _body(800),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert len(result) == 1
    msg = result[0].message
    assert "🔗 Nota nueva sin links:" in msg
    assert "[[formato]]" in msg
    assert "0 wikilinks outgoing" in msg
    assert "rag wikilinks suggest --path 02-Areas/formato.md" in msg


def test_projects_bucket_also_surfaces(mock_vault):
    """01-Projects también está en la allowlist — orphan ahí emite."""
    _make_note(
        mock_vault,
        "01-Projects/p.md",
        _body(700),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert len(result) == 1
    assert "01-Projects/p.md" in result[0].dedup_key


def test_resources_bucket_also_surfaces(mock_vault):
    """03-Resources también en la allowlist."""
    _make_note(
        mock_vault,
        "03-Resources/r.md",
        _body(700),
        hours_old=5.0,
    )
    result = orphan_surface_signal(_REF_NOW)
    assert len(result) == 1
    assert "03-Resources/r.md" in result[0].dedup_key


def test_silent_fail_on_bad_vault(monkeypatch):
    """Si `_resolve_vault_path` explota, devuelve [] sin propagar."""
    def _boom():
        raise RuntimeError("vault config roto")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_silent_fail_on_nonexistent_vault(monkeypatch, tmp_path):
    """Vault apunta a un dir inexistente → []."""
    nonexistent = tmp_path / "does-not-exist"
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: nonexistent)
    result = orphan_surface_signal(_REF_NOW)
    assert result == []


def test_signal_is_registered():
    """Sanity check: el decorator registró la signal en el registry global."""
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "orphan_surface" in names


def test_registered_in_rag_anticipate_signals():
    """`rag._ANTICIPATE_SIGNALS` debe incluir la signal una vez que rag.py
    lea el registry del package."""
    # El tuple final lo construye rag.py — verificamos indirectamente vía
    # el registry del package (que es lo que rag.py suma al tuple).
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "orphan_surface" in names
