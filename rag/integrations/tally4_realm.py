"""Tally4 (MOZE 4.x) `.realm` → MOZE-style CSV bridge.

## Contexto

Hasta abril 2026 el user exportaba CSV `MOZE_YYYYMMDD_HHMMSS.csv` desde la
app MOZE legacy y los dejaba caer en `CloudDocs/Finances`. El resto del
pipeline (`_fetch_finance`, `_sync_moze_notes`, dashboard) consume esos CSV.

A partir del 2026-05-04 el user migró a la app **Tally4** (MOZE 4.0), que
ya no exporta CSV. En su lugar guarda un backup .zip en su propio
container iCloud — `iCloud~amoos~Tally4/Documents/MOZE_4.0_<ts>.zip` —
con un `moze.realm` adentro (Realm Object Database, formato propietario,
v22). El esquema relevante es `AHRecord` (1 row por transacción).

Este módulo es el **puente**: detecta el zip más nuevo, extrae el realm,
lo convierte a un CSV con el mismo header que las exports legacy y lo
deja en el cache local `~/.local/share/obsidian-rag/moze_cache/`. El
resto del código sigue globeando `MOZE_*.csv` sin enterarse del cambio.

## Surfaces públicas

- ``ensure_moze_csv(moze_dir)`` — entry point. Idempotente. Si el cache
  CSV está al día respecto del zip más nuevo, no hace nada y devuelve
  el path. Si no, extrae + convierte y devuelve el path nuevo.
- ``CACHE_DIR`` — donde caen los CSV generados.

## Decisiones operativas

- **Realm-js npm**: única opción 2026 que abre `.realm` v22 en JS sin
  reverse engineering. Lo instalamos lazy a `EXTRACTOR_DIR` la primera
  vez que se necesita (silent-fail si no hay `node`/`npm` en PATH).
- **Copia del realm**: el script JS abre WRITABLE (realm-js 12+ requiere
  open writable para auto-upgrade el formato). Trabajamos sobre una
  copia en tmpdir — el original en iCloud queda intocable.
- **Cache key**: nombre del CSV = `MOZE_<int_zip_mtime>.csv`. Si la
  mtime del zip más nuevo coincide con un CSV existente, skip. Esto
  hace que `ensure_moze_csv()` sea cheap (~1ms) en el caso común.
- **Silent-fail**: si node falta, el realm no se puede abrir, o
  cualquier paso falla, devolvemos `None` y el caller cae a buscar
  CSVs históricos manuales (back-compat).

## Aprendido el 2026-05-04

Tally4 4.0 usa Realm format v22 — `realm-js` requiere open writable
(no readonly) para auto-upgradeear schemas internos. La opción
`{readOnly: true}` falla con "file format version (22) requires upgrade".
Por eso copiamos el realm a tmpdir antes de abrir.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────

#: Directorio donde vive el `node_modules/realm` (instalado lazy).
EXTRACTOR_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "realm-extractor"

#: Directorio cache donde caen los CSV generados a partir de los zip Tally4.
CACHE_DIR = Path.home() / ".local" / "share" / "obsidian-rag" / "moze_cache"

#: Path al script Node.js (vive en el repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_SCRIPT = _REPO_ROOT / "scripts" / "tally4_realm_to_csv.js"

#: Pattern de los zips Tally4. Tally4 4.x usa `MOZE_4.0_YYYY-MM-DD_HH:MM:SS.zip`.
_ZIP_GLOBS = ("MOZE_*.zip",)


# ── Public API ───────────────────────────────────────────────────────────


def ensure_moze_csv(moze_dir: Path) -> Path | None:
    """Si hay un zip Tally4 más nuevo que el cache CSV, extrae y convierte.
    Devuelve el path al CSV cacheado (o ``None`` si no se pudo generar).

    No lanza — silent-fail per convención del resto de los integrators.
    """
    zip_path = _latest_zip(moze_dir)
    if zip_path is None:
        return None

    try:
        zip_mtime = int(zip_path.stat().st_mtime)
    except OSError:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = CACHE_DIR / f"MOZE_{zip_mtime}.csv"
    if target_csv.exists():
        return target_csv

    if not _ensure_realm_npm():
        logger.info("tally4_realm: realm-js no disponible — skip extracción")
        return None

    try:
        _extract_zip_to_csv(zip_path, target_csv)
    except Exception as exc:
        logger.warning("tally4_realm: extracción falló: %s", exc)
        # Limpiar CSV parcial si quedó algo escrito.
        with _suppress_oserror():
            target_csv.unlink()
        return None

    if not target_csv.exists() or target_csv.stat().st_size == 0:
        return None

    # Prune CSVs viejos del cache: dejamos solo los 3 más recientes para
    # poder rollback si hace falta sin que crezca infinito.
    _prune_old_caches(keep=3)

    return target_csv


# ── Internals ────────────────────────────────────────────────────────────


def _latest_zip(moze_dir: Path) -> Path | None:
    if not moze_dir.exists():
        return None
    candidates: list[Path] = []
    for pat in _ZIP_GLOBS:
        candidates.extend(moze_dir.glob(pat))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def _ensure_realm_npm() -> bool:
    """Idempotent: instala `realm` en EXTRACTOR_DIR si falta. Devuelve
    True si el módulo está disponible (o se acaba de instalar OK)."""
    realm_pkg = EXTRACTOR_DIR / "node_modules" / "realm" / "package.json"
    if realm_pkg.exists():
        return True

    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        logger.info(
            "tally4_realm: node/npm no en PATH — omitiendo extracción "
            "(installá con `brew install node` o `nvm`)."
        )
        return False

    EXTRACTOR_DIR.mkdir(parents=True, exist_ok=True)
    pkg_json = EXTRACTOR_DIR / "package.json"
    if not pkg_json.exists():
        pkg_json.write_text('{"name":"obsidian-rag-realm-extractor","private":true}\n')

    try:
        proc = subprocess.run(
            [npm, "install", "--silent", "--no-audit", "--no-fund", "realm"],
            cwd=str(EXTRACTOR_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("tally4_realm: npm install falló: %s", exc)
        return False
    if proc.returncode != 0:
        logger.warning(
            "tally4_realm: npm install devolvió %s — stderr=%s",
            proc.returncode, (proc.stderr or "").strip()[:500],
        )
        return False
    return realm_pkg.exists()


def _extract_zip_to_csv(zip_path: Path, target_csv: Path) -> None:
    """Unzip `moze.realm` to a tmpdir, run the Node extractor, write CSV.
    Lanza ``RuntimeError`` si falla — el caller envuelve en try/except.
    """
    if not EXTRACTOR_SCRIPT.exists():
        raise RuntimeError(f"extractor script missing: {EXTRACTOR_SCRIPT}")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not in PATH")

    with tempfile.TemporaryDirectory(prefix="tally4-realm-") as td:
        tmp_realm = Path(td) / "moze.realm"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # En Tally4 4.x el archivo se llama exacto `moze.realm`
                # en la raíz. Defensive: aceptar variantes futuras.
                names = zf.namelist()
                realm_name = next(
                    (n for n in names if n.endswith("moze.realm") or n.endswith(".realm")),
                    None,
                )
                if realm_name is None:
                    raise RuntimeError(f"no .realm dentro del zip — names={names[:5]}")
                with zf.open(realm_name) as src, tmp_realm.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"zip inválido: {exc}") from exc

        env = os.environ.copy()
        env["NODE_PATH"] = str(EXTRACTOR_DIR / "node_modules")
        proc = subprocess.run(
            [node, str(EXTRACTOR_SCRIPT), "--realm", str(tmp_realm), "--out", str(target_csv)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()[:500]
            raise RuntimeError(f"node extractor exit={proc.returncode} stderr={err}")


def _prune_old_caches(*, keep: int) -> None:
    if not CACHE_DIR.exists():
        return
    csvs = sorted(CACHE_DIR.glob("MOZE_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in csvs[keep:]:
        with _suppress_oserror():
            stale.unlink()


class _suppress_oserror:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is None or issubclass(exc_type, OSError)
