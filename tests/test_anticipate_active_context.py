"""Tests para `rag_anticipate.signals.active_context` (Peekaboo Fase 2f).

Cubre:
- Feature OFF (RAG_SCREEN_OBSERVE != 1) → [] inmediato sin tocar DB.
- Tabla rag_screen_observations vacía → [].
- Última observación >30min → []
- Caption sin tokens distintivos → [].
- Sin proyectos dormant → [].
- Match keyword caption ↔ slug proyecto → emite candidate.
- Project demasiado fresco (<5d) → no match.
- Project muy viejo → score capeado a 0.9.
- Solo 1 candidate por pasada (primera coincidencia).
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

import rag
from rag_anticipate.signals.active_context import (
    _DORMANT_MIN_DAYS,
    _clear_dormant_cache,
    _tokenize,
    active_context_signal,
)


@pytest.fixture(autouse=True)
def _reset_dormant_cache():
    """Cada test arranca con cache limpio (cache_key incluye vault_path
    pero tmp_path varía por test, así no debería haber overlap — esto
    es belt-and-suspenders por si el cache se ensucia entre runs)."""
    _clear_dormant_cache()
    yield
    _clear_dormant_cache()

_REF_NOW = datetime(2026, 5, 13, 14, 0, 0)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """tmp DB + tmp vault, redirige rag.DB_PATH + rag.VAULT_PATH."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    vault = tmp_path / "vault"
    (vault / "01-Projects").mkdir(parents=True)
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    # ensure DB schema
    con = sqlite3.connect(str(db_dir / "telemetry.db"))
    rag._ensure_telemetry_tables(con)
    con.commit()
    con.close()
    return tmp_path


def _seed_obs(db_dir: Path, app: str, title: str, caption: str, age_seconds: int = 60) -> None:
    ts = int(time.time()) - age_seconds
    con = sqlite3.connect(str(db_dir / "telemetry.db"))
    con.execute(
        "INSERT INTO rag_screen_observations "
        "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, app, title, caption, 0, 100, "frontmost"),
    )
    con.commit()
    con.close()


def _make_project(vault: Path, name: str, days_dormant: float) -> Path:
    proj = vault / "01-Projects" / name
    proj.mkdir(parents=True)
    note = proj / "plan.md"
    note.write_text("contenido", encoding="utf-8")
    ts = time.time() - days_dormant * 86400
    os.utime(proj, (ts, ts))
    os.utime(note, (ts, ts))
    return proj


# ── tokenize ────────────────────────────────────────────────────────────────


def test_tokenize_skips_short_and_stopwords():
    out = _tokenize("para esta este sustantivo entonces project FastAPI")
    # "para","esta","este","entonces" → stopwords; ≥4 chars: sustantivo + project + fastapi.
    assert "sustantivo" in out
    assert "project" in out
    assert "fastapi" in out
    assert "para" not in out
    assert "esta" not in out


def test_tokenize_handles_accents():
    out = _tokenize("retroceso categoría síntesis ñoño")
    assert "categoría" in out
    assert "síntesis" in out
    assert "ñoño" in out  # 4 chars y no stopword


# ── gate ────────────────────────────────────────────────────────────────────


def test_observer_off_returns_empty(isolated, monkeypatch):
    monkeypatch.delenv("RAG_SCREEN_OBSERVE", raising=False)
    out = active_context_signal(_REF_NOW)
    assert out == []


def test_db_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "nonexistent")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path / "vault")
    assert active_context_signal(_REF_NOW) == []


# ── observation paths ──────────────────────────────────────────────────────


def test_empty_observations_returns_empty(isolated):
    out = active_context_signal(_REF_NOW)
    assert out == []


def test_stale_observation_returns_empty(isolated):
    # 45min atrás — fuera de la ventana 30min.
    _seed_obs(
        isolated / "ragvec", "Code", "rag-local repo",
        "editando archivo de python en proyecto FastAPI",
        age_seconds=45 * 60,
    )
    _make_project(isolated / "vault", "rag-local", days_dormant=10)
    assert active_context_signal(_REF_NOW) == []


def test_caption_without_distinctive_tokens(isolated):
    """Caption con solo stopwords + tokens <4 chars → no match."""
    _seed_obs(
        isolated / "ragvec", "App", "win",
        "para este este este",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "rag-local", days_dormant=10)
    assert active_context_signal(_REF_NOW) == []


# ── match paths ────────────────────────────────────────────────────────────


def test_match_emits_candidate(isolated):
    _seed_obs(
        isolated / "ragvec", "Code", "fastapi-server.py",
        "Visual Studio Code mostrando fastapi server.py con código de retrieval",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "fastapi-experiment", days_dormant=10)
    out = active_context_signal(_REF_NOW)
    assert len(out) == 1
    cand = out[0]
    assert cand.kind == "anticipate-active-context"
    assert "fastapi-experiment" in cand.message
    assert cand.dedup_key == "active-context:fastapi-experiment"
    assert cand.snooze_hours == 24
    # 10d dormant → score 0.5 + 0.05*(10-5) = 0.75
    assert abs(cand.score - 0.75) < 0.001
    assert cand.source_note is not None
    assert "fastapi-experiment" in cand.source_note


def test_fresh_project_does_not_match(isolated):
    """Proyecto <5d → no es dormant, skip."""
    _seed_obs(
        isolated / "ragvec", "Code", "main.py",
        "editando archivo de proyecto fastapi",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "fastapi", days_dormant=2)  # <5d
    assert active_context_signal(_REF_NOW) == []


def test_very_old_project_caps_at_0_9(isolated):
    _seed_obs(
        isolated / "ragvec", "Code", "main.py",
        "código de proyecto fastapi experimental",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "fastapi-experimental", days_dormant=60)
    out = active_context_signal(_REF_NOW)
    assert len(out) == 1
    assert out[0].score == 0.9  # cap exacto


def test_first_match_wins(isolated):
    """Si hay 2 proyectos dormant que matchean, el más recientemente
    activo (top de la lista ordenada) gana."""
    _seed_obs(
        isolated / "ragvec", "Code", "shared.py",
        "fastapi shared module mostrando código",
        age_seconds=60,
    )
    # Crear 2 proyectos con "fastapi" en el slug.
    proj_recent = _make_project(isolated / "vault", "fastapi-recent", days_dormant=6)
    proj_old = _make_project(isolated / "vault", "fastapi-old", days_dormant=30)
    # Ajustar mtime para asegurar orden.
    os.utime(proj_recent, (time.time() - 6 * 86400, time.time() - 6 * 86400))
    os.utime(proj_old, (time.time() - 30 * 86400, time.time() - 30 * 86400))

    out = active_context_signal(_REF_NOW)
    assert len(out) == 1
    # El más recientemente activo gana (fastapi-recent, dormant 6d).
    assert "fastapi-recent" in out[0].source_note


def test_no_projects_returns_empty(isolated):
    _seed_obs(
        isolated / "ragvec", "Code", "main.py",
        "código fastapi en pantalla",
        age_seconds=60,
    )
    # No projects creados — solo el dir 01-Projects vacío.
    assert active_context_signal(_REF_NOW) == []


def test_dormant_cache_avoids_repeat_rglob(isolated, monkeypatch):
    """Audit perf #3: _dormant_projects cachea TTL 30min para evitar
    walks repetidos. Verificá que el rglob NO se ejecuta dos veces si
    la signal corre dos veces dentro del TTL."""
    _seed_obs(
        isolated / "ragvec", "Code", "fastapi.py",
        "código fastapi en pantalla",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "fastapi-cached", days_dormant=10)

    # Spy: contar invocaciones de rglob.
    from pathlib import Path as _P
    original_rglob = _P.rglob
    rglob_count = {"n": 0}

    def counting_rglob(self, pattern):
        rglob_count["n"] += 1
        return original_rglob(self, pattern)

    monkeypatch.setattr(_P, "rglob", counting_rglob)

    out1 = active_context_signal(_REF_NOW)
    first_count = rglob_count["n"]
    assert len(out1) == 1, "primera pasada debe emitir candidate"
    assert first_count >= 1, "rglob debe correr en la primera pasada"

    out2 = active_context_signal(_REF_NOW)
    second_count = rglob_count["n"]
    assert len(out2) == 1, "segunda pasada también emite (mismo proyecto)"
    assert second_count == first_count, (
        f"rglob no debe re-correr dentro del TTL "
        f"(esperado {first_count}, got {second_count})"
    )


def test_just_under_min_days_not_dormant(isolated):
    """Proyecto a 4.5 días NO es dormant todavía (umbral 5d)."""
    _seed_obs(
        isolated / "ragvec", "Code", "main.py",
        "código del proyecto target en pantalla",
        age_seconds=60,
    )
    _make_project(isolated / "vault", "target-proj", days_dormant=_DORMANT_MIN_DAYS - 0.5)
    assert active_context_signal(_REF_NOW) == []
