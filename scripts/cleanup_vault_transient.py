"""Cleanup transient folders bajo `04-Archive/99-obsidian-system/99-AI/`.

Background (2026-04-27): el vault Obsidian acumula carpetas de "sistema"
(outputs de scripts, transcripts efímeros, drafts de commits, exports
auto-regenerados) bajo `99-AI/`. Esos archivos NO contaminan el RAG
porque `is_excluded()` ya filtra todo `04-Archive/99-obsidian-system/`
del indexado, pero sí ocupan espacio en disco e iCloud, y ensucian el
graph view de Obsidian + las búsquedas full-text del propio Obsidian
(no del RAG). Este script los purga periódicamente moviéndolos al
`.trash/` nativo del vault (reversible: Obsidian los conserva por X
días configurable antes del purge final).

Política por carpeta:

| Carpeta              | TTL     | Por qué                                  |
|----------------------|---------|------------------------------------------|
| `tmp/`               | 7 días  | Drafts de commit msg, scratch, ephemeral |
| `conversations/`     | 30 días | Chat transcripts del web/api/chat        |
| `sessions/`          | 30 días | Resúmenes de sesión de Devin/Claude      |
| `Wiki/`              | wipe    | Auto-regenerada por exporter MOZE        |
| `plans/`             | 180 días| Planes viejos ya implementados/abandonados |
| `system/<slug>/`     | 180 días| Design docs, post-mortems añejos         |
| `reviews/`           | 365 días| Daily/evening reviews >1 año             |

Carpetas EXPLÍCITAMENTE intactas (whitelist en `PROTECTED`):

- `memory/` — sistema de memoria persistente del agente
- `skills/` — definiciones de skills (symlink chain ~/.claude/skills →
  ~/.agents/skills → vault). Borrar acá rompería el linker

Cualquier OTRA carpeta nueva que aparezca bajo `99-AI/` se loggea como
`unknown_skipped` (no se toca) — agregala a `CLEANUP_POLICIES` o
`PROTECTED` la próxima vez que la veas en el log.

Modo de borrado: archivo viejo → `<vault>/.trash/<flat-path>`.
Mantenemos la convención de Obsidian (separador `-`, paths flat) para
que cuando el user vacíe el `.trash` desde la UI, los archivos de este
script se vayan junto con el resto. Si el destino ya existe, se sufija
con `.1`, `.2`, ... para no pisar.

Invocación:

  $ python -m scripts.cleanup_vault_transient            # apply real
  $ python -m scripts.cleanup_vault_transient --dry-run  # simular
  $ python -m scripts.cleanup_vault_transient --json     # summary JSON
  $ rag vault-cleanup                                    # CLI wrapper

Daemon: `com.fer.obsidian-rag-vault-cleanup` corre diario a 02:00 vía
launchd (instalado por `rag setup`). Logs en
`~/.local/share/obsidian-rag/cleanup-vault.log`.

Rollback: si necesitás recuperar algo, `<vault>/.trash/` lo conserva
hasta que vacíes la papelera de Obsidian (Settings → Files & Links →
Trash) o hasta el cleanup TTL del propio Obsidian (default sin TTL —
manual). Para apagar este script: `launchctl bootout gui/$(id -u)
~/Library/LaunchAgents/com.fer.obsidian-rag-vault-cleanup.plist` y/o
`rag setup --remove` para destruir todos los plists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------- Configuration ----------------------------------------------------


VAULT_DEFAULT = Path(
    "/Users/fer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
)


# Cleanup policy per top-level folder under 99-AI/.
#
# `mode`:
#   - "by_age": mover archivos con mtime < now - ttl_days a .trash/
#   - "wipe":   mover TODOS los archivos cada corrida (ignora ttl_days)
#
# `ttl_days`:
#   - número positivo: edad mínima en días para que un archivo se mueva
#   - 0: no se usa (modo wipe)
CLEANUP_POLICIES: dict[str, dict[str, Any]] = {
    "tmp":           {"mode": "by_age", "ttl_days": 7},
    "conversations": {"mode": "by_age", "ttl_days": 30},
    "sessions":      {"mode": "by_age", "ttl_days": 30},
    "Wiki":          {"mode": "wipe",   "ttl_days": 0},
    "plans":         {"mode": "by_age", "ttl_days": 180},
    "system":        {"mode": "by_age", "ttl_days": 180},
    "reviews":       {"mode": "by_age", "ttl_days": 365},
}


# Carpetas explícitamente intactas. Si una carpeta está en PROTECTED, el
# script NO la toca aunque alguien por error la agregue a CLEANUP_POLICIES.
# Belt-and-suspenders.
PROTECTED: set[str] = {"memory", "skills"}


# Files que siempre se borran (no merece la pena moverlos al trash).
ALWAYS_PURGE = {".DS_Store"}


# ---------- Vault resolution -------------------------------------------------


def _resolve_vault() -> Path:
    """Resolve the Obsidian vault root.

    Mismo orden de fallback que `rag/__init__.py`:
    1. `OBSIDIAN_RAG_VAULT` env var (override explícito)
    2. `VAULT_DEFAULT` constante (path real del user, hardcoded)
    """
    env = os.environ.get("OBSIDIAN_RAG_VAULT", "").strip()
    if env:
        return Path(env).expanduser()
    return VAULT_DEFAULT


# ---------- Trash move -------------------------------------------------------


def _flatten_path(rel: Path) -> str:
    """Convert a relative path to Obsidian's flat trash naming scheme.

    `tmp/commit_msg.txt` → `tmp-commit_msg.txt`. Mismo separador que usa
    Obsidian para sus propios archivos en `.trash/` (observado 2026-04-27).
    """
    return "-".join(rel.parts)


def _move_to_trash(file: Path, vault: Path, base: Path, dry_run: bool) -> Path:
    """Move `file` to `<vault>/.trash/<flat-name>`. Returns the dest path.

    El nombre flat se computa relativo al `base` (99-AI/), no al `vault`,
    para que el resultado sea conciso (`tmp-commit_msg.txt` y no
    `04-Archive-99-obsidian-system-99-AI-tmp-commit_msg.txt`).

    Si el destino existe, sufija con `.1`, `.2`, ... para no pisar.
    """
    trash = vault / ".trash"
    rel = file.relative_to(base)
    dest = trash / _flatten_path(rel)
    if dest.exists():
        i = 1
        while True:
            candidate = dest.with_name(f"{dest.name}.{i}")
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    if not dry_run:
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(dest))
    return dest


# ---------- Core cleanup -----------------------------------------------------


def _count_files(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.is_file())


def _purge_dsstore(folder: Path, dry_run: bool) -> int:
    """`.DS_Store` files don't deserve a trash slot — just rm them."""
    n = 0
    for f in folder.rglob(".DS_Store"):
        if not f.is_file():
            continue
        if not dry_run:
            try:
                f.unlink()
            except OSError:
                continue
        n += 1
    return n


def _process_folder(
    folder: Path,
    vault: Path,
    base: Path,
    policy: dict[str, Any],
    now: float,
    dry_run: bool,
) -> dict[str, int]:
    """Apply `policy` to `folder`. Returns counts."""
    n_moved = 0
    n_kept = 0
    cutoff = now - policy["ttl_days"] * 86400

    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        if f.name in ALWAYS_PURGE:
            # already handled by _purge_dsstore but keep the guard
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        should_move = (
            policy["mode"] == "wipe"
            or mtime < cutoff
        )
        if should_move:
            _move_to_trash(f, vault, base, dry_run)
            n_moved += 1
        else:
            n_kept += 1

    return {"n_moved": n_moved, "n_kept": n_kept}


def run(
    *,
    dry_run: bool = False,
    vault: Path | None = None,
) -> dict[str, Any]:
    """Run cleanup on all configured folders. Returns a summary dict.

    Skips silently (returns ok=False with reason) if:
    - The vault root doesn't exist (running on CI without iCloud sync)
    - The 99-AI/ base folder doesn't exist (fresh setup, nothing to clean)
    """
    vault = vault if vault is not None else _resolve_vault()
    base = vault / "04-Archive" / "99-obsidian-system" / "99-AI"

    summary: dict[str, Any] = {
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "vault": str(vault),
        "base": str(base),
        "dry_run": dry_run,
        "folders": [],
        "n_moved": 0,
        "n_kept": 0,
        "n_protected": 0,
        "n_unknown": 0,
        "n_dsstore_purged": 0,
    }

    if not vault.exists():
        summary["ok"] = False
        summary["error"] = f"vault not found: {vault}"
        return summary
    if not base.exists():
        summary["ok"] = False
        summary["error"] = f"99-AI base not found: {base}"
        return summary

    now = time.time()

    # First pass: nuke .DS_Store everywhere under 99-AI (incl. protected dirs)
    summary["n_dsstore_purged"] = _purge_dsstore(base, dry_run)

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name

        if name in PROTECTED:
            n_files = _count_files(folder)
            summary["folders"].append({
                "name": name,
                "policy": "protected",
                "n_files": n_files,
                "n_moved": 0,
            })
            summary["n_protected"] += n_files
            continue

        policy = CLEANUP_POLICIES.get(name)
        if policy is None:
            n_files = _count_files(folder)
            summary["folders"].append({
                "name": name,
                "policy": "unknown_skipped",
                "n_files": n_files,
                "n_moved": 0,
            })
            summary["n_unknown"] += n_files
            continue

        counts = _process_folder(folder, vault, base, policy, now, dry_run)
        policy_label = (
            f"ttl_{policy['ttl_days']}d"
            if policy["mode"] == "by_age"
            else "wipe"
        )
        summary["folders"].append({
            "name": name,
            "policy": policy_label,
            **counts,
        })
        summary["n_moved"] += counts["n_moved"]
        summary["n_kept"] += counts["n_kept"]

    return summary


# ---------- CLI entry point --------------------------------------------------


def _print_human(summary: dict[str, Any], stream=sys.stdout) -> None:
    prefix = "[dry-run] " if summary["dry_run"] else ""
    if not summary["ok"]:
        print(
            f"{prefix}vault-cleanup error: {summary.get('error')}",
            file=sys.stderr,
        )
        return
    print(
        f"{prefix}vault-cleanup OK · {summary['n_moved']} moved → .trash/ · "
        f"{summary['n_kept']} kept · {summary['n_protected']} protected · "
        f"{summary['n_unknown']} unknown · "
        f"{summary['n_dsstore_purged']} .DS_Store purged",
        file=stream,
    )
    for f in summary["folders"]:
        line = f"  · {f['name']:<16s} ({f['policy']})"
        if "n_moved" in f and "n_kept" in f:
            line += f": {f['n_moved']} moved · {f['n_kept']} kept"
        elif "n_files" in f:
            line += f": {f['n_files']} files (untouched)"
        print(line, file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanup_vault_transient",
        description="Cleanup transient vault folders under 99-AI/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No mover archivos, solo simular y reportar",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Imprimir summary como JSON (para servicios/scripts)",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Override vault path (default: $OBSIDIAN_RAG_VAULT or hardcoded)",
    )
    args = parser.parse_args(argv)

    summary = run(dry_run=args.dry_run, vault=args.vault)

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_human(summary)

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
