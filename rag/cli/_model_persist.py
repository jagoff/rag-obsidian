"""Persistencia de model tiers en plists de daemons.

Cuando hacés `rag model set <tier> <model> --persist`, escribimos la
env var `RAG_<TIER>_MODEL` en el bloque `EnvironmentVariables` de TODOS
los plists `com.fer.obsidian-rag-*.plist` instalados. Después
disparamos `launchctl kickstart -k` para que cada daemon levante con
el modelo nuevo. Sin `--persist`, el cambio dura sólo lo que dura el
proceso CLI actual (env var del proceso).

Usamos `plistlib` (stdlib) para parsear/escribir — más robusto que regex
sobre el XML. Si el plist no tiene el bloque `EnvironmentVariables`, lo
agregamos.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from rag import models as _models

_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
_PLIST_GLOB = "com.fer.obsidian-rag-*.plist"


def _list_plists() -> list[Path]:
    """Plists obsidian-rag instalados (ignora `.bak.*`)."""
    if not _LAUNCH_AGENTS_DIR.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(_LAUNCH_AGENTS_DIR.glob(_PLIST_GLOB)):
        # Ignorar backups (`.plist.bak.YYYYMMDD-HHMMSS`)
        if ".bak." in p.name:
            continue
        if not p.name.endswith(".plist"):
            continue
        out.append(p)
    return out


def _backup(plist: Path) -> Path:
    """Backup `<plist>.bak.YYYYMMDD-HHMMSS` antes de mutar."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = plist.with_suffix(f"{plist.suffix}.bak.{stamp}")
    shutil.copy2(plist, bak)
    return bak


def _patch_plist(plist: Path, env_var: str, value: str | None) -> bool:
    """Setear / borrar `env_var` en `EnvironmentVariables` de `plist`.

    `value=None` → borra la entry. Devuelve True si el plist cambió.
    """
    try:
        with plist.open("rb") as fh:
            data = plistlib.load(fh)
    except Exception:
        return False
    env = data.get("EnvironmentVariables")
    if not isinstance(env, dict):
        env = {}
    prev = env.get(env_var)
    if value is None:
        if env_var not in env:
            return False
        env.pop(env_var, None)
    else:
        if prev == value:
            return False
        env[env_var] = value
    data["EnvironmentVariables"] = env
    _backup(plist)
    with plist.open("wb") as fh:
        plistlib.dump(data, fh, sort_keys=False)
    return True


def _label_from_plist(plist: Path) -> str | None:
    """Extrae `<key>Label</key>` para el bootstrap/kickstart."""
    try:
        with plist.open("rb") as fh:
            data = plistlib.load(fh)
        label = data.get("Label")
        if isinstance(label, str) and label:
            return label
    except Exception:
        return None
    return None


def _kickstart(label: str) -> None:
    """`launchctl kickstart -k gui/<uid>/<label>` — no levanta excepción."""
    try:
        uid_proc = subprocess.run(
            ["id", "-u"], check=True, capture_output=True, text=True, timeout=2,
        )
        uid = uid_proc.stdout.strip()
        target = f"gui/{uid}/{label}"
        subprocess.run(
            ["launchctl", "kickstart", "-k", target],
            check=False, capture_output=True, timeout=10,
        )
    except Exception:
        pass


def persist_tier_to_plists(tier: str, model: str) -> list[str]:
    """Escribir `RAG_<TIER>_MODEL=<model>` a todos los plists + kickstart.

    Devuelve los nombres (sin path) de los plists efectivamente modificados.
    """
    if tier not in _models.TIERS:
        raise ValueError(f"Tier desconocido {tier!r}")
    env_var = _models.ENV_VARS[tier]
    touched: list[str] = []
    for plist in _list_plists():
        if _patch_plist(plist, env_var, model):
            touched.append(plist.name)
            label = _label_from_plist(plist)
            if label:
                _kickstart(label)
    return touched


def unset_tier_in_plists(tier: str) -> list[str]:
    """Borrar `RAG_<TIER>_MODEL` de todos los plists + kickstart.

    Útil para `rag model reset <tier> --persist`.
    """
    if tier not in _models.TIERS:
        raise ValueError(f"Tier desconocido {tier!r}")
    env_var = _models.ENV_VARS[tier]
    touched: list[str] = []
    for plist in _list_plists():
        if _patch_plist(plist, env_var, None):
            touched.append(plist.name)
            label = _label_from_plist(plist)
            if label:
                _kickstart(label)
    return touched
