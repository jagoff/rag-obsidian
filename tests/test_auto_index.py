"""Tests del auto-index: detectar y absorber cambios sin que el usuario
corra `rag index` explícito.

Cubre:
  - state load/save/key (tolerante a archivo ausente/corrupto).
  - _with_vault context manager (swap + restore + invalida cache).
  - auto_index_vault: first-time, no_changes, incremental, orphans.
"""
import time
from pathlib import Path

import pytest

import rag


@pytest.fixture
def tmp_setup(tmp_path, monkeypatch):
    """Aisla DB + state path + ambient/filing logs en tmp."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "AUTO_INDEX_STATE_PATH", tmp_path / "auto_index_state.json")
    # No queremos hooks ambient ni embeds reales en estos tests.
    monkeypatch.setattr(rag, "AMBIENT_CONFIG_PATH", tmp_path / "ambient.json")

    def fake_embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", fake_embed)
    return tmp_path


def _mk_vault(root: Path, files: dict[str, str]) -> Path:
    """Crea un vault con los .md indicados {rel_path: content}."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


# ── State ─────────────────────────────────────────────────────────────────────


def test_state_load_empty_when_missing(tmp_setup):
    assert rag._auto_index_state_load() == {}


def test_state_load_recovers_from_corrupt(tmp_setup):
    rag.AUTO_INDEX_STATE_PATH.write_text("not json")
    assert rag._auto_index_state_load() == {}


def test_state_save_load_roundtrip(tmp_setup):
    rag._auto_index_state_save({"abc": 123.45})
    assert rag._auto_index_state_load() == {"abc": 123.45}


def test_vault_state_key_is_stable(tmp_setup, tmp_path):
    v = tmp_path / "v"; v.mkdir()
    k1 = rag._vault_state_key(v)
    k2 = rag._vault_state_key(v)
    assert k1 == k2
    assert len(k1) == 16


def test_vault_state_key_differs_per_path(tmp_setup, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    assert rag._vault_state_key(v1) != rag._vault_state_key(v2)


# ── _with_vault ───────────────────────────────────────────────────────────────


def test_with_vault_swaps_and_restores(tmp_setup, tmp_path):
    saved_vp = rag.VAULT_PATH
    saved_cn = rag.COLLECTION_NAME
    other = tmp_path / "other"; other.mkdir()
    with rag._with_vault(other):
        assert rag.VAULT_PATH == other
        assert rag.COLLECTION_NAME != saved_cn   # hash sufijado
    assert rag.VAULT_PATH == saved_vp
    assert rag.COLLECTION_NAME == saved_cn


def test_with_vault_restores_on_exception(tmp_setup, tmp_path):
    saved_vp = rag.VAULT_PATH
    other = tmp_path / "other"; other.mkdir()
    try:
        with rag._with_vault(other):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert rag.VAULT_PATH == saved_vp


# ── auto_index_vault ──────────────────────────────────────────────────────────


def test_first_time_indexes_everything(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {
        "02-Areas/a.md": "# A\n\nbody a",
        "02-Areas/b.md": "# B\n\nbody b",
        "03-Resources/c.md": "# C\n\nbody c",
    })
    stats = rag.auto_index_vault(v)
    assert stats["kind"] == "first_time"
    assert stats["indexed"] == 3
    assert stats["scanned"] == 3
    # State persisted.
    assert rag._vault_state_key(v) in rag._auto_index_state_load()


def test_no_changes_after_first_index_is_fast_path(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {"a.md": "content"})
    rag.auto_index_vault(v)   # first time
    # Sin tocar nada, segunda corrida = no_changes.
    stats = rag.auto_index_vault(v)
    assert stats["kind"] == "no_changes"
    assert stats["indexed"] == 0
    assert stats["removed"] == 0


@pytest.mark.slow
def test_incremental_picks_up_modified_file(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {"a.md": "original"})
    rag.auto_index_vault(v)
    # Asegurar que mtime cambia (algunas filesystems tienen resolución 1s).
    time.sleep(0.05)
    (v / "a.md").write_text("modified", encoding="utf-8")
    # Bump mtime explícito para no depender de la precisión del FS.
    import os as _os
    _os.utime(v / "a.md", (time.time() + 1, time.time() + 1))

    stats = rag.auto_index_vault(v)
    assert stats["kind"] == "incremental"
    assert stats["indexed"] == 1


@pytest.mark.slow
def test_incremental_picks_up_new_file(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {"a.md": "a"})
    rag.auto_index_vault(v)
    time.sleep(0.05)
    (v / "b.md").write_text("nuevo", encoding="utf-8")

    stats = rag.auto_index_vault(v)
    assert stats["indexed"] == 1
    assert stats["kind"] == "incremental"


def test_incremental_removes_orphans(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {"a.md": "a", "b.md": "b"})
    rag.auto_index_vault(v)
    # Borrar b.md del disk.
    (v / "b.md").unlink()

    stats = rag.auto_index_vault(v)
    assert stats["removed"] == 1
    assert stats["kind"] == "incremental"


def test_unchanged_file_with_bumped_mtime_is_skipped_by_hash_gate(tmp_setup, tmp_path):
    """Si tocás mtime sin cambiar el contenido, _index_single_file detecta
    via hash que nada cambió y devuelve 'skipped' — auto_index no cuenta
    ese caso como reindex."""
    v = _mk_vault(tmp_path / "vault", {"a.md": "exact same content"})
    rag.auto_index_vault(v)
    import os as _os
    future = time.time() + 1000
    _os.utime(v / "a.md", (future, future))

    stats = rag.auto_index_vault(v)
    # mtime > last_check → entra al loop, pero hash gate skipea → indexed=0.
    assert stats["indexed"] == 0


def test_excluded_paths_are_skipped(tmp_setup, tmp_path):
    v = _mk_vault(tmp_path / "vault", {
        "02-Areas/keep.md": "yes",
        ".trash/skip.md": "no",
        ".obsidian/config.md": "no",
    })
    stats = rag.auto_index_vault(v)
    assert stats["scanned"] == 1   # solo 02-Areas/keep.md


def test_empty_vault_no_files_returns_no_changes(tmp_setup, tmp_path):
    v = tmp_path / "vault"; v.mkdir()
    stats = rag.auto_index_vault(v)
    assert stats["kind"] == "no_changes"
    assert stats["indexed"] == 0
    assert stats["scanned"] == 0
