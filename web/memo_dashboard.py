"""Backend del dashboard `/memo` — visor de las memorias que persiste
el MCP [`memo`](https://github.com/jagoff/memo).

`memo` es el sucesor del antiguo `mem-vault`: misma idea (memoria persistente
de Claude/agentes sobre bugs, decisiones, preferencias, facts del user)
pero stack local-first puro — sqlite-vec + MLX, sin Qdrant ni Ollama.

Storage layout (memo ≥ 0.3.x):
- `~/.local/share/memo/memvec.db` — `meta` table + `vec0` sqlite-vec index.
  El reader entra acá; los embeddings los ignoramos (no es esta UI quien
  retrieva).
- `~/.local/share/memo/history.db` — append-only events log (save / update /
  delete). Usado para los conteos de "saved hoy / 7d".
- `<vault>/99-obsidian/99-AI/memory/*.md` — source of truth. El body lo
  leemos directo del .md cuando el user clickea una memoria.

Path overrides: `MEMO_DATA_DIR` env apunta al directorio de los .db; si no
está, se asume el default `~/.local/share/memo/`. El path del vault se
toma de `rag.VAULT_PATH` (ya resuelto con la precedencia de obsidian-rag
— env override > vaults.json > iCloud default).

Endpoints (definidos en web/server.py, no acá):
- `GET /memo`  → `memo.html`
- `GET /api/memo` → `snapshot()` (TTL 30s)
- `GET /api/memo/note?id=…` → `note_detail()`

Performance budget: cold ~50-100ms para 450 memorias; warm < 5ms gracias
al cache del endpoint.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_RECENT_DEFAULT = 30
_RECENT_MAX = 200


def _memo_dir() -> Path:
    """Resolver dónde viven los .db de memo.

    Precedencia:
    1. `MEMO_DATA_DIR` env (override explícito del user).
    2. `~/.local/share/memo/` (default que setea `memo init`).
    """
    override = os.environ.get("MEMO_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".local/share/memo"


def _memvec_db() -> Path:
    return _memo_dir() / "memvec.db"


def _history_db() -> Path:
    return _memo_dir() / "history.db"


def _vault_path() -> Path:
    """Vault root para resolver el `.md` de cada memoria. Reusa la
    resolución oficial de `rag` (env override > vaults.json > iCloud).
    """
    try:
        from rag import VAULT_PATH  # type: ignore

        return VAULT_PATH
    except Exception:
        return Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"


def _open_ro(db: Path) -> sqlite3.Connection | None:
    """Abrir `db` en modo read-only. Devuelve None si no existe — el caller
    decide qué hacer (silent-fail con counts en 0).
    """
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_tags(raw: Any) -> list[str]:
    """Tags en memo viven como JSON list serializado dentro de TEXT.
    Devolver lista vacía ante cualquier error en vez de fallar el snapshot.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw][:12]
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [str(t) for t in loaded][:12]
    except Exception:
        pass
    return []


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _humanize_ago(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    if secs < 86400 * 30:
        return f"{secs // 86400}d"
    if secs < 86400 * 365:
        return f"{secs // (86400 * 30)}mo"
    return f"{secs // (86400 * 365)}y"


def snapshot(limit: int = _RECENT_DEFAULT, type_filter: str | None = None) -> dict:
    """Snapshot completo para hidratar `/memo`.

    Forma:
        {
            "ok": True,
            "memo_dir": "/Users/.../memo",
            "vault_path": "/Users/.../Notes",
            "totals": {"all": 452, "by_type": [{"type": "note", "count": 140}, ...]},
            "activity": {
                "saved_today": 12,
                "saved_7d": 47,
                "saved_30d": 180,
                "events_total": 449,
            },
            "tags_top": [{"tag": "project", "count": 38}, ...],
            "recent": [{"id": "...", "title": "...", "type": "...", "tags": [...],
                        "updated": "...", "ago": "2h", "path": "..."}, ...],
        }
    """
    out: dict[str, Any] = {
        "ok": True,
        "memo_dir": str(_memo_dir()),
        "vault_path": str(_vault_path()),
        "totals": {"all": 0, "by_type": []},
        "activity": {"saved_today": 0, "saved_7d": 0, "saved_30d": 0, "events_total": 0},
        "tags_top": [],
        "recent": [],
    }

    limit = max(1, min(int(limit or _RECENT_DEFAULT), _RECENT_MAX))
    type_filter = (type_filter or "").strip().lower() or None

    mv = _open_ro(_memvec_db())
    if mv is None:
        out["ok"] = False
        out["error"] = (
            f"memvec.db no encontrado en {_memvec_db()}. "
            "Corré `memo init` o seteá MEMO_DATA_DIR."
        )
        return out

    try:
        out["totals"]["all"] = mv.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
        out["totals"]["by_type"] = [
            {"type": r["type"], "count": r["c"]}
            for r in mv.execute(
                "SELECT type, COUNT(*) AS c FROM meta GROUP BY type ORDER BY c DESC"
            )
        ]

        if type_filter:
            recent_rows = mv.execute(
                "SELECT id, title, type, tags, updated, created, path "
                "FROM meta WHERE type = ? ORDER BY updated DESC LIMIT ?",
                (type_filter, limit),
            ).fetchall()
        else:
            recent_rows = mv.execute(
                "SELECT id, title, type, tags, updated, created, path "
                "FROM meta ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()

        recent: list[dict[str, Any]] = []
        tag_counter: Counter[str] = Counter()
        for r in recent_rows:
            tags = _parse_tags(r["tags"])
            updated_dt = _parse_iso(r["updated"])
            recent.append(
                {
                    "id": r["id"],
                    "title": r["title"] or "(sin título)",
                    "type": r["type"] or "note",
                    "tags": tags,
                    "updated": r["updated"],
                    "created": r["created"],
                    "ago": _humanize_ago(updated_dt),
                    "path": r["path"],
                }
            )
        out["recent"] = recent

        for r in mv.execute("SELECT tags FROM meta"):
            for t in _parse_tags(r["tags"]):
                tag_counter[t] += 1
        out["tags_top"] = [
            {"tag": t, "count": c} for t, c in tag_counter.most_common(20)
        ]
    finally:
        mv.close()

    h = _open_ro(_history_db())
    if h is not None:
        try:
            now = datetime.now(timezone.utc)
            iso_today = (now - timedelta(days=1)).isoformat()
            iso_7d = (now - timedelta(days=7)).isoformat()
            iso_30d = (now - timedelta(days=30)).isoformat()
            row = h.execute(
                "SELECT "
                "  SUM(CASE WHEN ts >= ? AND op = 'save' THEN 1 ELSE 0 END) AS today, "
                "  SUM(CASE WHEN ts >= ? AND op = 'save' THEN 1 ELSE 0 END) AS d7, "
                "  SUM(CASE WHEN ts >= ? AND op = 'save' THEN 1 ELSE 0 END) AS d30, "
                "  COUNT(*) AS total "
                "FROM events",
                (iso_today, iso_7d, iso_30d),
            ).fetchone()
            out["activity"] = {
                "saved_today": int(row["today"] or 0),
                "saved_7d": int(row["d7"] or 0),
                "saved_30d": int(row["d30"] or 0),
                "events_total": int(row["total"] or 0),
            }
        finally:
            h.close()

    return out


_PATH_TRAVERSAL_RE = re.compile(r"(?:^|/)\.\.(?:/|$)")


def _safe_resolve(rel_path: str) -> Path | None:
    """Validar y resolver `rel_path` dentro del vault. Rechaza absolute,
    `..` traversal, y cualquier resolved path que escape del vault root.
    """
    if not rel_path or len(rel_path) > 500:
        return None
    if _PATH_TRAVERSAL_RE.search(rel_path):
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return None
    vault = _vault_path()
    try:
        full = (vault / rel_path).resolve()
        if not str(full).startswith(str(vault.resolve())):
            return None
        return full
    except Exception:
        return None


def note_detail(memo_id: str | None = None, path: str | None = None) -> dict:
    """Detalle de UNA memoria.

    Acepta `memo_id` (full id o prefijo ≥4 chars) o `path` (rel al vault).
    Devuelve metadata + body completo (sin caps — las memorias son chicas,
    < 4KB típicas).
    """
    out: dict[str, Any] = {"ok": False, "error": "", "memo": None}

    if not memo_id and not path:
        out["error"] = "missing memo_id or path"
        return out

    mv = _open_ro(_memvec_db())
    if mv is None:
        out["error"] = "memvec.db no encontrado"
        return out

    row: sqlite3.Row | None = None
    try:
        if memo_id:
            memo_id = memo_id.strip()
            if len(memo_id) < 4:
                out["error"] = "id prefix too short (min 4)"
                return out
            row = mv.execute(
                "SELECT id, title, type, tags, created, updated, body_hash, path "
                "FROM meta WHERE id = ? OR id LIKE ? LIMIT 1",
                (memo_id, memo_id + "%"),
            ).fetchone()
        else:
            row = mv.execute(
                "SELECT id, title, type, tags, created, updated, body_hash, path "
                "FROM meta WHERE path = ? LIMIT 1",
                (path,),
            ).fetchone()
    finally:
        mv.close()

    if not row:
        out["error"] = "not found"
        return out

    full = _safe_resolve(row["path"])
    body = ""
    if full and full.exists():
        try:
            raw = full.read_text(encoding="utf-8", errors="replace")
            body = _strip_frontmatter(raw)
        except Exception as exc:  # pragma: no cover
            body = f"(error leyendo .md: {exc})"
    else:
        body = "(archivo .md no encontrado — corré `memo doctor`)"

    updated_dt = _parse_iso(row["updated"])
    out["ok"] = True
    out["memo"] = {
        "id": row["id"],
        "title": row["title"] or "(sin título)",
        "type": row["type"] or "note",
        "tags": _parse_tags(row["tags"]),
        "created": row["created"],
        "updated": row["updated"],
        "ago": _humanize_ago(updated_dt),
        "path": row["path"],
        "body": body,
    }
    return out


_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _strip_frontmatter(raw: str) -> str:
    """Sacar el bloque YAML `---...---` del principio. Si no hay, devolver
    el raw intacto.
    """
    m = _FRONTMATTER_RE.match(raw)
    if m:
        return raw[m.end():].lstrip()
    return raw
