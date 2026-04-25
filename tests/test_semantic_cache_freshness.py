"""Tests para el check de freshness per-entry — post 2026-04-23.

Con el `_compute_corpus_hash` reducido a count-only, las invalidaciones
globales solo disparan en adds/removes. Los edits a notas individuales
se detectan ahora en `semantic_cache_lookup` via `_cached_entry_is_stale`:
si cualquiera de las `paths[]` cacheadas tiene mtime > cached_ts, la
fila se skippea (reason='stale_source') sin tumbar el resto del cache.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

import rag


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_COSINE", 0.9)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_DEFAULT_TTL", 86400)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_MAX_ROWS", 100)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    yield tmp_path


def _emb(*floats: float, dim: int = 64):
    base = np.zeros(dim, dtype="float32")
    for i, v in enumerate(floats):
        base[i] = v
    if np.linalg.norm(base) == 0:
        base[0] = 1.0
    return base / np.linalg.norm(base)


# ── `_cached_entry_is_stale` unit tests ──────────────────────────────────────


def test_freshness_empty_paths_never_stale(clean_cache_env):
    """Entries sin paths (ej. respuestas meta) nunca se consideran stale."""
    assert rag._cached_entry_is_stale([], cached_ts=time.time() - 3600) is False


def test_freshness_fresh_file(clean_cache_env):
    """File existe + mtime ≤ cached_ts → fresh (no stale)."""
    vault = clean_cache_env
    f = vault / "a.md"
    f.write_text("# a")
    # Archivo escrito ahora; cached_ts en el futuro → mtime ≤ cached_ts → fresh.
    future_ts = time.time() + 3600
    assert rag._cached_entry_is_stale(["a.md"], cached_ts=future_ts) is False


def test_freshness_edited_after_cached_ts(clean_cache_env):
    """File existe + mtime > cached_ts → stale (editado post-cache)."""
    vault = clean_cache_env
    f = vault / "a.md"
    f.write_text("# a")
    # cached_ts en el pasado; archivo editado después → stale.
    past_ts = time.time() - 3600
    # Asegurar mtime > past_ts (que en este caso ya se cumple — stat actual).
    assert rag._cached_entry_is_stale(["a.md"], cached_ts=past_ts) is True


def test_freshness_missing_file_assumes_fresh(clean_cache_env):
    """File no existe → tratar como fresh (el corpus_hash global ya invalida
    en removes, no queremos doble-punish)."""
    # Nunca creamos el archivo.
    assert rag._cached_entry_is_stale(
        ["nope/doesnotexist.md"], cached_ts=time.time(),
    ) is False


def test_freshness_multi_path_any_stale_wins(clean_cache_env):
    """Con N paths, UNA sola stale hace toda la entry stale."""
    vault = clean_cache_env
    (vault / "a.md").write_text("a")
    (vault / "b.md").write_text("b")
    # cached_ts en el pasado → ambos files son stale (mtime actual > cached_ts).
    past_ts = time.time() - 3600
    assert rag._cached_entry_is_stale(["a.md", "b.md"], cached_ts=past_ts) is True


def test_freshness_vault_path_error_assumes_fresh(monkeypatch, clean_cache_env):
    """Si `_resolve_vault_path` explota, assume fresh (no invalidar el cache
    por un problema de infra)."""
    def boom():
        raise RuntimeError("vault unresolvable")
    monkeypatch.setattr(rag, "_resolve_vault_path", boom)
    # Incluso con cached_ts muy antiguo, no stale (no podemos verificar).
    assert rag._cached_entry_is_stale(
        ["a.md"], cached_ts=time.time() - 99999,
    ) is False


# ── Integration: lookup applies freshness ────────────────────────────────────


@pytest.mark.slow
def test_lookup_skips_stale_entry(clean_cache_env):
    """Integration: almacenar una entry con path, editar el archivo,
    lookup debe reportar stale_source en el probe + devolver None."""
    vault = clean_cache_env
    note = vault / "ikigai.md"
    note.write_text("# ikigai (initial)")

    emb = _emb(1.0)
    # Storear — la entry quedó marcada con mtime actual.
    rag.semantic_cache_store(
        emb, question="q", response="cached answer",
        paths=["ikigai.md"], scores=[0.9], top_score=0.9,
        intent="semantic", corpus_hash="H",
    )
    # Sanity: lookup inmediato hits.
    hit, probe = rag.semantic_cache_lookup(emb, "H", return_probe=True)
    assert hit is not None
    assert probe["result"] == "hit"

    # Esperar 1.1s para que la granularidad de mtime del FS nos dé un
    # mtime estrictamente > cached_ts. APFS tiene resolución nanosegundos
    # pero el mtime que lee el store del cache es ISO-8601 a microsegs —
    # aun así con 1.1s se garantiza el delta.
    time.sleep(1.1)
    # Re-write → bump mtime.
    note.write_text("# ikigai (edited)")

    hit2, probe2 = rag.semantic_cache_lookup(emb, "H", return_probe=True)
    assert hit2 is None, "entry should be skipped as stale after edit"
    assert probe2["result"] == "miss"
    assert probe2["reason"] == "stale_source"
    assert probe2["skipped_stale"] >= 1


@pytest.mark.slow
def test_lookup_stale_doesnt_blow_up_fresh_entries(clean_cache_env):
    """Tener UNA entry stale + UNA entry fresh → la fresh sigue hitting."""
    vault = clean_cache_env
    stale_note = vault / "old.md"
    stale_note.write_text("old")
    fresh_note = vault / "new.md"
    fresh_note.write_text("new")

    emb_old = _emb(1.0, 0.0)
    emb_new = _emb(0.0, 1.0)
    rag.semantic_cache_store(
        emb_old, question="q1", response="old resp",
        paths=["old.md"], scores=[0.9], top_score=0.9,
        intent="semantic", corpus_hash="H",
    )
    rag.semantic_cache_store(
        emb_new, question="q2", response="new resp",
        paths=["new.md"], scores=[0.9], top_score=0.9,
        intent="semantic", corpus_hash="H",
    )
    time.sleep(1.1)
    # Edit solo la "old" — debe quedar stale, la "new" sigue fresh.
    stale_note.write_text("old (edited)")

    hit_old = rag.semantic_cache_lookup(emb_old, "H")
    assert hit_old is None, "stale entry should miss"

    hit_new = rag.semantic_cache_lookup(emb_new, "H")
    assert hit_new is not None, "fresh entry should still hit"
    assert hit_new["response"] == "new resp"
