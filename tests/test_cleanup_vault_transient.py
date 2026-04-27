"""Tests para `scripts/cleanup_vault_transient.py`.

Verifican el contrato del cleanup:

- Folders con TTL: solo archivos más viejos que el TTL se mueven (mtime).
- Wiki/: wipe completo cada corrida (ignora TTL).
- memory/ + skills/: PROTECTED, nunca se tocan.
- .DS_Store: siempre se borra (sin trash).
- Carpetas desconocidas: se reportan como `unknown_skipped`, no se tocan.
- Vault inexistente: `ok=False` con error claro.
- Dry-run: summary correcto pero ningún archivo movido.

Todos los tests usan `tmp_path` para construir un vault sintético — no
tocan iCloud ni el `.trash/` real.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cleanup_vault_transient import (  # noqa: E402
    CLEANUP_POLICIES,
    PROTECTED,
    run,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _set_mtime(path: Path, days_ago: float) -> None:
    """Override mtime de un archivo a `now - days_ago` días."""
    target = time.time() - days_ago * 86400
    os.utime(path, (target, target))


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Crea la estructura mínima del vault con `99-AI/` y todas las carpetas."""
    base = tmp_path / "04-Archive" / "99-obsidian-system" / "99-AI"
    for folder in [
        "tmp", "conversations", "sessions", "Wiki",
        "plans", "system", "reviews",
        "memory", "skills",
    ]:
        (base / folder).mkdir(parents=True)
    return tmp_path


# ── Happy path ──────────────────────────────────────────────────────────────


def test_dry_run_moves_nothing(vault: Path):
    """Dry-run reporta lo correcto pero NO modifica el FS."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    old = base / "tmp" / "ancient.txt"
    old.write_text("antiguo")
    _set_mtime(old, 30)  # 30 días, > TTL de tmp/ (7d)

    summary = run(dry_run=True, vault=vault)
    assert summary["ok"], summary
    assert summary["dry_run"] is True
    assert summary["n_moved"] == 1, summary["folders"]
    # FS intacto
    assert old.exists()
    assert not (vault / ".trash").exists()


def test_apply_moves_to_trash(vault: Path):
    """Apply real mueve archivos viejos a `<vault>/.trash/`."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    old = base / "tmp" / "old-commit.txt"
    old.write_text("legacy")
    _set_mtime(old, 30)

    summary = run(dry_run=False, vault=vault)
    assert summary["ok"]
    assert summary["n_moved"] == 1
    # Original ya no existe
    assert not old.exists()
    # Pero está en .trash con nombre flat (`tmp-old-commit.txt`)
    trashed = vault / ".trash" / "tmp-old-commit.txt"
    assert trashed.exists()
    assert trashed.read_text() == "legacy"


def test_ttl_respects_age(vault: Path):
    """Archivo dentro del TTL se queda; archivo fuera se mueve."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    fresh = base / "conversations" / "ayer.md"
    fresh.write_text("reciente")
    _set_mtime(fresh, 1)  # 1 día, dentro del TTL de conversations (30d)

    stale = base / "conversations" / "anteayer.md"
    stale.write_text("viejo")
    _set_mtime(stale, 60)  # 60 días, fuera del TTL

    summary = run(dry_run=False, vault=vault)
    conv = next(f for f in summary["folders"] if f["name"] == "conversations")
    assert conv["n_moved"] == 1
    assert conv["n_kept"] == 1
    assert fresh.exists()
    assert not stale.exists()


def test_wiki_wipe_ignores_ttl(vault: Path):
    """Wiki/ es wipe completo — incluso archivos recientes se van."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    brand_new = base / "Wiki" / "index.md"
    brand_new.write_text("auto-generated")
    _set_mtime(brand_new, 0.001)  # creado hace segundos

    summary = run(dry_run=False, vault=vault)
    wiki = next(f for f in summary["folders"] if f["name"] == "Wiki")
    assert wiki["policy"] == "wipe"
    assert wiki["n_moved"] == 1
    assert wiki["n_kept"] == 0
    assert not brand_new.exists()


# ── Protected folders ───────────────────────────────────────────────────────


def test_protected_folders_untouched(vault: Path):
    """memory/ y skills/ nunca se tocan, ni archivos viejos."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    for protected in PROTECTED:
        ancient = base / protected / "anciano.md"
        ancient.write_text("debe quedarse")
        _set_mtime(ancient, 9999)  # casi 30 años

    summary = run(dry_run=False, vault=vault)
    for protected in PROTECTED:
        entry = next(f for f in summary["folders"] if f["name"] == protected)
        assert entry["policy"] == "protected"
        assert entry.get("n_moved", 0) == 0
        assert (base / protected / "anciano.md").exists(), \
            f"{protected}/anciano.md no debe haberse movido"


def test_unknown_folder_skipped(vault: Path):
    """Carpeta nueva que aparece bajo 99-AI/ se reporta y NO se toca."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    (base / "novedad").mkdir()
    file_in_unknown = base / "novedad" / "data.txt"
    file_in_unknown.write_text("misterio")
    _set_mtime(file_in_unknown, 99999)

    summary = run(dry_run=False, vault=vault)
    novedad = next(f for f in summary["folders"] if f["name"] == "novedad")
    assert novedad["policy"] == "unknown_skipped"
    assert file_in_unknown.exists()


# ── DS_Store handling ───────────────────────────────────────────────────────


def test_ds_store_purged_everywhere(vault: Path):
    """`.DS_Store` se borra incluso dentro de carpetas protegidas."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    (base / ".DS_Store").write_text("garbage")
    (base / "memory" / ".DS_Store").write_text("garbage")
    (base / "skills" / ".DS_Store").write_text("garbage")

    summary = run(dry_run=False, vault=vault)
    assert summary["n_dsstore_purged"] == 3
    assert not (base / ".DS_Store").exists()
    assert not (base / "memory" / ".DS_Store").exists()
    assert not (base / "skills" / ".DS_Store").exists()


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_missing_vault_returns_ok_false(tmp_path: Path):
    """Si el vault no existe (CI sin iCloud), retorna ok=False con error."""
    summary = run(dry_run=False, vault=tmp_path / "nope")
    assert summary["ok"] is False
    assert "vault not found" in summary["error"]
    assert summary["n_moved"] == 0


def test_missing_99ai_returns_ok_false(tmp_path: Path):
    """Si el vault existe pero 99-AI/ no, retorna ok=False (fresh setup)."""
    summary = run(dry_run=False, vault=tmp_path)
    assert summary["ok"] is False
    assert "99-AI base not found" in summary["error"]


def test_trash_collision_suffixes(vault: Path):
    """Si ya existe el destino en .trash/, sufija con .1, .2, ..."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    trash = vault / ".trash"
    trash.mkdir()
    # Pre-existente: simulamos que un cleanup anterior dejó el mismo nombre
    (trash / "tmp-x.txt").write_text("vieja-versión-1")
    (trash / "tmp-x.txt.1").write_text("vieja-versión-2")

    file = base / "tmp" / "x.txt"
    file.write_text("nueva")
    _set_mtime(file, 30)

    run(dry_run=False, vault=vault)
    # Pre-existentes intactos
    assert (trash / "tmp-x.txt").read_text() == "vieja-versión-1"
    assert (trash / "tmp-x.txt.1").read_text() == "vieja-versión-2"
    # La nueva fue al siguiente slot (.2)
    assert (trash / "tmp-x.txt.2").read_text() == "nueva"


def test_subfolders_in_system_respected(vault: Path):
    """`system/<slug>/file.md` con TTL 180d aplica al archivo, no al dir."""
    base = vault / "04-Archive/99-obsidian-system/99-AI"
    slug = base / "system" / "mi-slug"
    slug.mkdir()
    old = slug / "ancient.md"
    old.write_text("vieja spec")
    _set_mtime(old, 365)  # > 180d

    fresh = slug / "current.md"
    fresh.write_text("plan activo")
    _set_mtime(fresh, 30)  # < 180d

    summary = run(dry_run=False, vault=vault)
    sys_entry = next(f for f in summary["folders"] if f["name"] == "system")
    assert sys_entry["n_moved"] == 1
    assert sys_entry["n_kept"] == 1
    assert not old.exists()
    assert fresh.exists()
    # El subdir queda (no eliminamos dirs vacíos)
    assert slug.exists()


# ── Sanity checks on policy table ───────────────────────────────────────────


def test_policy_table_consistent():
    """Aserciones estructurales sobre las políticas — guardia contra
    typos en futuras ediciones de CLEANUP_POLICIES."""
    for name, policy in CLEANUP_POLICIES.items():
        assert "mode" in policy and policy["mode"] in {"by_age", "wipe"}
        assert "ttl_days" in policy
        assert isinstance(policy["ttl_days"], int)
        assert policy["ttl_days"] >= 0
    # Whitelist no se solapa con políticas
    overlap = PROTECTED & set(CLEANUP_POLICIES.keys())
    assert not overlap, f"folders in both PROTECTED and CLEANUP_POLICIES: {overlap}"
