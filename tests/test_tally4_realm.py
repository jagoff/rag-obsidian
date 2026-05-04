"""Smoke tests para `rag.integrations.tally4_realm`.

NO ejecuta el extractor Node.js real — eso requiere `npm install realm`
(~150MB de binaries) y nos llevaría 1-2min por test. Cubrimos los caminos
defensivos: dir vacío, sin zip, npm/node ausente.

El happy-path (zip → realm → CSV) se valida manualmente con el backup
real del user. Si llega regresión, abrir un .realm fixture mínimo y
agregar test e2e gateado por env var.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_ensure_moze_csv_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dir inexistente → ``None`` sin crashear."""
    from rag.integrations import tally4_realm

    fake = tmp_path / "does_not_exist"
    assert tally4_realm.ensure_moze_csv(fake) is None


def test_ensure_moze_csv_no_zip(tmp_path: Path) -> None:
    """Dir existe pero sin zip → ``None`` (no toca el cache)."""
    from rag.integrations import tally4_realm

    assert tally4_realm.ensure_moze_csv(tmp_path) is None


def test_latest_zip_picks_newest(tmp_path: Path) -> None:
    """Múltiples zips → devuelve el de mtime más nuevo."""
    from rag.integrations import tally4_realm
    import time

    older = tmp_path / "MOZE_4.0_2026-04-01_10:00:00.zip"
    newer = tmp_path / "MOZE_4.0_2026-05-04_12:00:00.zip"
    older.write_bytes(b"old")
    time.sleep(0.01)
    newer.write_bytes(b"new")

    picked = tally4_realm._latest_zip(tmp_path)
    assert picked == newer


def test_ensure_realm_npm_no_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si `node`/`npm` no están en PATH, ``_ensure_realm_npm`` devuelve False
    sin lanzar. Usamos `shutil.which` patcheado para simular ausencia."""
    from rag.integrations import tally4_realm

    # Re-route extractor dir para no contaminar el real del user.
    fake_extractor = tmp_path / "realm-extractor"
    monkeypatch.setattr(tally4_realm, "EXTRACTOR_DIR", fake_extractor)

    monkeypatch.setattr(tally4_realm.shutil, "which", lambda _: None)
    assert tally4_realm._ensure_realm_npm() is False


def test_extract_zip_to_csv_bad_zip(tmp_path: Path) -> None:
    """Un zip corrupto se reporta como RuntimeError (silent-fail en
    el caller, pero la función interna sí lanza)."""
    from rag.integrations import tally4_realm

    bad_zip = tmp_path / "MOZE_corrupt.zip"
    bad_zip.write_bytes(b"not a real zip")

    out = tmp_path / "out.csv"
    with pytest.raises(RuntimeError, match=r"zip inv|missing|node"):
        tally4_realm._extract_zip_to_csv(bad_zip, out)


def test_prune_old_caches_keeps_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_prune_old_caches(keep=N)` deja los N más nuevos por mtime."""
    from rag.integrations import tally4_realm
    import time

    monkeypatch.setattr(tally4_realm, "CACHE_DIR", tmp_path)

    files = []
    for i in range(5):
        p = tmp_path / f"MOZE_{i}.csv"
        p.write_text(f"row {i}\n")
        time.sleep(0.005)  # garantiza mtime estrictamente creciente
        files.append(p)

    tally4_realm._prune_old_caches(keep=2)

    # Quedan los 2 más nuevos (índices 3, 4).
    survivors = sorted(tmp_path.glob("MOZE_*.csv"))
    assert {p.name for p in survivors} == {"MOZE_3.csv", "MOZE_4.csv"}


def test_cache_dir_gitignored() -> None:
    """El módulo no debe escribir al repo — CACHE_DIR es bajo home."""
    from rag.integrations import tally4_realm

    assert str(tally4_realm.CACHE_DIR).startswith(str(Path.home()))
    assert ".local/share/obsidian-rag" in str(tally4_realm.CACHE_DIR)
