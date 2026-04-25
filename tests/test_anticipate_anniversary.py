"""Tests for the 'anniversary' Anticipatory Agent signal.

Cubre:
- Empty vault → []
- Nota 365d exactos → emit con score 1.0
- Nota 370d → score 0.5 (extremo de la ventana)
- Nota 400d → fuera de ventana, no emit
- Nota 365d en 00-Inbox → no emit (bucket excluido)
- Nota 365d pero body <500 chars → no emit
- dedup_key estable cross-calls
- Frontmatter `created:` preferido sobre mtime
- Máximo 1 candidate cuando hay varias en ventana

El signal camina el vault filesystem-only — no retrieve(), no embeddings.
Aislamos con `monkeypatch.setattr(rag, "_resolve_vault_path", ...)` a un
`tmp_path` que construye cada test.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag import SqliteVecClient as _TestVecClient
from rag_anticipate.signals.anniversary import anniversary_signal


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB + expone un tmp_path para construir vault.

    Copia literal del fixture usado por tests/test_anticipate_agent.py para
    evitar polución entre runs.
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    client.get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )
    with rag._ragvec_state_conn() as _conn:
        pass
    return tmp_path


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Construye un vault tmp vacío y lo registra como el vault activo.

    Crea los 3 buckets PARA (02-Areas, 01-Projects, 03-Resources) + el
    00-Inbox para tener los paths disponibles en tests. Monkeypatchea
    `rag._resolve_vault_path` a este tmp.
    """
    vroot = tmp_path / "vault"
    (vroot / "02-Areas").mkdir(parents=True)
    (vroot / "01-Projects").mkdir(parents=True)
    (vroot / "03-Resources").mkdir(parents=True)
    (vroot / "00-Inbox").mkdir(parents=True)

    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vroot)
    return vroot


# Reference `now` compartido por todos los tests para evitar drift entre
# la creación del mtime y la llamada a la signal. La signal recibe `now`
# como parámetro explícito → pasamos este mismo valor al llamarla.
_REF_NOW = datetime(2025, 6, 15, 12, 0, 0)


def _make_note(vault: Path, rel: str, body: str, days_old: float) -> Path:
    """Crea una nota en `vault/<rel>` con `body` y mtime = _REF_NOW - days_old.

    Usa `os.utime` para mockear el mtime (atime=mtime para que stat lo
    refleje). Crea dirs intermedios si hace falta. El mtime se calcula
    contra `_REF_NOW` fijo, así los tests pasan un `now=_REF_NOW` a la
    signal y evitamos drift sub-segundo en las comparaciones de ventana.
    """
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    ts = (_REF_NOW - timedelta(days=days_old)).timestamp()
    os.utime(path, (ts, ts))
    return path


def _long_body(prefix: str = "Contenido sustantivo ", min_chars: int = 600) -> str:
    """Genera un body ≥min_chars para que la nota pase el filtro de 500 chars."""
    base = prefix + "bla " * 200  # ~800 chars
    assert len(base) >= min_chars
    return base


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_vault_returns_empty(state_db, vault):
    """Vault sin notas → signal devuelve []."""
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_note_exactly_365d_emits_score_1(state_db, vault):
    """Nota con mtime exactamente hace 365 días → emit con score == 1.0."""
    _make_note(
        vault,
        "02-Areas/Personal/nota-aniversario.md",
        _long_body("Recuerdo del año pasado "),
        days_old=365.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-anniversary"
    # 365d → score 1.0 (distance = 0).
    assert c.score == pytest.approx(1.0, abs=0.02)
    assert "[[nota-aniversario]]" in c.message
    assert c.snooze_hours == 720
    # dedup_key con el file_rel.
    assert "02-Areas/Personal/nota-aniversario.md" in c.dedup_key
    assert c.dedup_key.startswith("anniv:")


def test_note_370d_emits_lower_score(state_db, vault):
    """Nota con mtime hace 370d → score ~0.5 (distancia máxima en la ventana)."""
    _make_note(
        vault,
        "01-Projects/proyecto-viejo.md",
        _long_body("Proyecto antiguo "),
        days_old=370.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    # Distance = 5d → score = 1 - 5/10 = 0.5
    assert c.score == pytest.approx(0.5, abs=0.05)
    assert c.score < 1.0


def test_note_400d_out_of_window_no_emit(state_db, vault):
    """Nota con mtime hace 400d → fuera de ventana, no emit."""
    _make_note(
        vault,
        "02-Areas/vieja.md",
        _long_body("Demasiado vieja "),
        days_old=400.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_note_200d_too_recent_no_emit(state_db, vault):
    """Nota reciente (200d) → no emit (aún no es aniversario)."""
    _make_note(
        vault,
        "02-Areas/reciente.md",
        _long_body("Demasiado nueva "),
        days_old=200.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_note_in_inbox_is_skipped(state_db, vault):
    """Nota con mtime 365d pero bajo 00-Inbox → skip (bucket no allowed)."""
    _make_note(
        vault,
        "00-Inbox/quick-note.md",
        _long_body("Nota inbox "),
        days_old=365.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_note_too_small_is_skipped(state_db, vault):
    """Nota 365d pero body <500 chars → skip."""
    _make_note(
        vault,
        "02-Areas/chica.md",
        "Muy corta.",  # <500 chars
        days_old=365.0,
    )
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_dedup_key_stable_cross_calls(state_db, vault):
    """Dos llamadas con el mismo estado producen el mismo dedup_key."""
    _make_note(
        vault,
        "03-Resources/libro-leido.md",
        _long_body("Reseña del libro "),
        days_old=365.0,
    )
    r1 = anniversary_signal(_REF_NOW)
    r2 = anniversary_signal(_REF_NOW)
    assert len(r1) == 1
    assert len(r2) == 1
    assert r1[0].dedup_key == r2[0].dedup_key
    # Shape check.
    assert r1[0].dedup_key.startswith("anniv:03-Resources/libro-leido.md:")


def test_score_in_valid_range(state_db, vault):
    """Score siempre en [0, 1] para cualquier punto de la ventana."""
    _make_note(
        vault,
        "02-Areas/medio.md",
        _long_body("Medio aniv "),
        days_old=362.5,  # distance 2.5 → score 0.75
    )
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    assert 0.0 <= result[0].score <= 1.0
    assert result[0].score == pytest.approx(0.75, abs=0.05)


def test_multiple_anniv_notes_picks_closest_to_365(state_db, vault):
    """Con varias notas en la ventana, devuelve MÁXIMO 1 — la más cercana a 365d."""
    _make_note(
        vault,
        "02-Areas/exacta.md",
        _long_body("La más cercana "),
        days_old=365.0,  # score 1.0
    )
    _make_note(
        vault,
        "01-Projects/lejana.md",
        _long_body("La más lejos "),
        days_old=369.0,  # score 0.6
    )
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    # La que gana es la de 365d exactos → contiene "[[exacta]]".
    assert "[[exacta]]" in result[0].message
    assert result[0].score == pytest.approx(1.0, abs=0.02)


def test_message_format(state_db, vault):
    """El message tiene el formato esperado (emoji 🎂, wikilink, pregunta final)."""
    body = (
        "---\n"
        "tags: [reflection]\n"
        "---\n\n"
        "Primera línea del body que debería aparecer en el preview.\n\n"
        + "padding " * 100
    )
    _make_note(vault, "02-Areas/formato.md", body, days_old=365.0)
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    msg = result[0].message
    assert "🎂 Hace 1 año escribiste:" in msg
    assert "[[formato]]" in msg
    assert "¿Releer, actualizar o archivar?" in msg
    # Preview debe tener la primera línea del body (no el frontmatter).
    assert "Primera línea del body" in msg


def test_frontmatter_created_preferred_over_mtime(state_db, vault):
    """Si `created:` está en el frontmatter, se usa eso en lugar del mtime.

    Construimos una nota con mtime de 100d (recién tocada) pero frontmatter
    `created:` de hace 365 días → debe emitir igual.
    """
    # created exactamente hace 365 días.
    created_date = (_REF_NOW - timedelta(days=365)).strftime("%Y-%m-%d")
    body = (
        f"---\ncreated: {created_date}\n---\n\n"
        + "Contenido sustantivo de la nota. " * 30
    )
    _make_note(
        vault,
        "02-Areas/con-fm-created.md",
        body,
        days_old=100.0,  # mtime reciente → si usara mtime, no emitiría
    )
    result = anniversary_signal(_REF_NOW)
    assert len(result) == 1
    assert "[[con-fm-created]]" in result[0].message
    # Score cerca de 1.0 porque el frontmatter apunta a 365d exactos.
    assert result[0].score >= 0.9


def test_silent_fail_on_bad_vault(state_db, monkeypatch):
    """Si `_resolve_vault_path` explota, la signal devuelve [] silenciosamente."""
    def _boom():
        raise RuntimeError("vault config roto")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_silent_fail_on_nonexistent_vault(state_db, monkeypatch, tmp_path):
    """Si el vault apunta a un dir que no existe, devuelve []."""
    nonexistent = tmp_path / "does-not-exist"
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: nonexistent)
    result = anniversary_signal(_REF_NOW)
    assert result == []


def test_signal_is_registered():
    """Sanity check: el decorator registró la signal en el registry global."""
    # Forzar import del package para activar autodiscovery.
    import rag_anticipate
    names = [n for (n, _fn) in rag_anticipate.SIGNALS]
    assert "anniversary" in names
