"""Backend del dashboard `/memo` — visor + scoring de las memorias del MCP
[`memo`](https://github.com/jagoff/memo) (sucesor de `mem-vault`, 2026-05-10).

Objetivo de este módulo: NO solo listar memorias, sino contestar la
pregunta del user *"¿sirve o no esta memoria?"* con señales concretas:

1. **Quality score por memoria** (0-100). Combina:
   - `type_actionable` (bug/decision/preference son accionables; note/fact
     son referenciales).
   - `has_tags` (≥1 tag = mejor clasificada).
   - `good_size` (200B ≤ body ≤ 4000B; muy chico = trivial, muy grande =
     bloat).
   - `fresh` (updated en últimos 30d).
   - `unique` (sin near-dupes a distancia < 0.20).
   El breakdown se devuelve junto al score así el user ve QUÉ pesa.

2. **Near-dupe clusters** via kNN sobre los 2560-dim embeddings ya
   guardados por memo (`vec` table de sqlite-vec). Distance < 0.20 →
   par candidato a fusionar. Clusters armados con union-find.

3. **Health del set completo**: cuántas tagless, cuántas stale, cuántas
   en clusters de dupes, ratio accionables.

4. **FTS search** sobre la `fts` table que memo mantiene actualizada al
   guardar.

5. **Saves timeline** (últimos 30d) — picos de capture muestran sesiones
   intensas, valles muestran períodos sin trabajo cognitivo capturado.

Storage layout (read-only sobre los 3 sqlite que memo maneja):
- `~/.local/share/memo/memvec.db` — `meta` + `vec` (sqlite-vec 2560-dim) + `fts`.
- `~/.local/share/memo/history.db` — events log (save/update/delete; en
  la práctica el 100% de los eventos son `save` porque memo no expone
  un flujo de "update").
- `<vault>/99-obsidian/99-AI/memory/*.md` — source of truth para el body.

Performance:
- Cold snapshot ≈ 1.5-2s (dominated by kNN sobre 463 vectores; 3ms × 463
  = 1.4s + filesystem reads para body previews).
- Warm < 5ms (TTL 30s en el endpoint).
- FTS search ≈ 5-20ms.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import sqlite_vec  # type: ignore
    _HAVE_SQLITE_VEC = True
except ImportError:  # pragma: no cover
    _HAVE_SQLITE_VEC = False


_RECENT_DEFAULT = 50
_RECENT_MAX = 500

# Quality scoring weights (deben sumar 100 para que el score sea %).
_W_ACTIONABLE = 25
_W_TAGS = 20
_W_SIZE = 20
_W_FRESH = 15
_W_UNIQUE = 20

# Type bucketing.
_TYPE_ACTIONABLE = {"bug", "decision", "preference"}
_TYPE_INFORMATIONAL = {"note", "fact"}

# Body-size sweet spot (bytes).
_SIZE_MIN_OK = 200
_SIZE_MAX_OK = 4000

# Freshness window.
_FRESH_DAYS = 30
_STALE_DAYS = 180

# Near-dupe threshold. sqlite-vec por default devuelve L2; con embeddings
# L2-normalised, dist² ≈ 2 - 2·cos. Memo normaliza, así que:
#   dist 0.10 ≈ cos 0.995 — casi idénticas
#   dist 0.15 ≈ cos 0.989
#   dist 0.20 ≈ cos 0.980 — demasiado laxo (transitively conecta todo el corpus)
# Pruebas con el corpus actual: 0.12 captura ~30 pares accionables;
# 0.20 captura todo (442 nodos conectados en una sola componente).
_DUPE_DIST_THRESHOLD = 0.12

_BODY_PREVIEW_CHARS = 180

# Donde Claude Code persiste el log de cada turno por sesión. Cada vez que
# el hook `UserPromptSubmit` de memo (`memo recall-hook`) inyecta el bloque
# "Relevant memories from your past (memo)", queda registrado acá como un
# attachment de tipo `hook_additional_context` con timestamp + memoria_id
# (prefijo 8 hex) + score. Ese log es la fuente de verdad para medir si
# memo *se está usando* — la única fuente, porque memo en sí NO loguea
# retrievals.
_CLAUDE_PROJECTS_DIR = Path.home() / ".claude/projects"

# Memoria "viva" si fue recalled en los últimos N días.
_ACTIVE_DAYS = 7

# Memoria "muerta" si nunca fue recalled Y tiene > N días de creada.
# (Una memoria de hoy NO es muerta — todavía no tuvo oportunidad de
# matchear un prompt.) 2d es razonable: el user trabaja todos los días
# en obsidian-rag, si en 48h no matcheó ningún prompt, problema.
_DEAD_MIN_AGE_DAYS = 2


def _memo_dir() -> Path:
    override = os.environ.get("MEMO_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    data_override = os.environ.get("MEMO_DATA_DIR")
    if data_override:
        data_dir = Path(data_override).expanduser().resolve()
        # Test/dev installs often keep state DBs beside MEMO_DATA_DIR. Real
        # memo >=0.7 separates markdown data_dir from sqlite state_dir.
        if (data_dir / "memvec.db").exists() or (data_dir / "history.db").exists():
            return data_dir
    cfg = _memo_storage_config()
    if cfg.get("state_dir"):
        return Path(cfg["state_dir"]).expanduser().resolve()
    return Path.home() / ".local/share/memo"


def _memo_storage_config() -> dict[str, str]:
    cfg_path = Path(os.environ.get("MEMO_CONFIG", "~/.config/memo/config.toml")).expanduser()
    if not cfg_path.exists():
        return {}
    out: dict[str, str] = {}
    in_storage = False
    try:
        for raw in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                in_storage = line == "[storage]"
                continue
            if not in_storage or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in {"data_dir", "state_dir", "vault_path", "memory_subdir"}:
                out[key] = val
    except OSError:
        return {}
    return out


def _memory_dir() -> Path:
    override = os.environ.get("MEMO_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    cfg = _memo_storage_config()
    if cfg.get("data_dir"):
        return Path(cfg["data_dir"]).expanduser().resolve()
    vault = os.environ.get("MEMO_VAULT_PATH") or cfg.get("vault_path")
    subdir = os.environ.get("MEMO_MEMORY_SUBDIR") or cfg.get("memory_subdir")
    if vault and subdir:
        return (Path(vault).expanduser() / subdir).resolve()
    return Path.home() / "Documents/memo"


def _memvec_db() -> Path:
    return _memo_dir() / "memvec.db"


def _history_db() -> Path:
    return _memo_dir() / "history.db"


def _vault_path() -> Path:
    return _memory_dir()


def _open_ro(db: Path, *, with_vec: bool = False) -> sqlite3.Connection | None:
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    if with_vec and _HAVE_SQLITE_VEC:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pass
    return conn


def _parse_tags(raw: Any) -> list[str]:
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


_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_PATH_TRAVERSAL_RE = re.compile(r"(?:^|/)\.\.(?:/|$)")


def _strip_frontmatter(raw: str) -> str:
    m = _FRONTMATTER_RE.match(raw)
    if m:
        return raw[m.end():].lstrip()
    return raw


def _safe_resolve(rel_path: str) -> Path | None:
    if not rel_path or len(rel_path) > 500:
        return None
    if _PATH_TRAVERSAL_RE.search(rel_path):
        return None
    p = Path(rel_path)
    vault = _memory_dir().resolve()
    try:
        full = p.expanduser().resolve() if p.is_absolute() else (vault / rel_path).resolve()
        full.relative_to(vault)
        return full
    except Exception:
        return None


def _path_fields(rel_path: str) -> dict[str, str]:
    full = _safe_resolve(rel_path)
    full_path = str(full) if full else ""
    return {
        "path": rel_path,
        "full_path": full_path,
        "obsidian_url": f"obsidian://open?path={quote(full_path, safe='')}" if full_path else "",
    }


def _invalidate_caches() -> None:
    with _DUPE_LOCK:
        _DUPE_CACHE["key"] = None
        _DUPE_CACHE["payload"] = None
    with _RECALL_LOCK:
        _RECALL_CACHE["ts"] = 0.0
        _RECALL_CACHE["payload"] = None


def _resolve_meta_row(conn: sqlite3.Connection, memo_id: str) -> sqlite3.Row | None:
    memo_id = (memo_id or "").strip()
    if len(memo_id) < 4:
        return None
    row = conn.execute(
        "SELECT id, title, type, tags, created, updated, body_hash, path, extra_json "
        "FROM meta WHERE id = ?",
        (memo_id,),
    ).fetchone()
    if row is not None:
        return row
    rows = conn.execute(
        "SELECT id, title, type, tags, created, updated, body_hash, path, extra_json "
        "FROM meta WHERE id LIKE ? ORDER BY updated DESC LIMIT 2",
        (memo_id + "%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]
    return None


def _body_for_path(rel_path: str) -> str:
    full = _safe_resolve(rel_path)
    if not full or not full.exists():
        return ""
    try:
        return _strip_frontmatter(full.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""


def _yaml_scalar(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _frontmatter(row: dict[str, Any], tags: list[str], updated: str) -> str:
    tag_lines = "\n".join(f"- {_yaml_scalar(t)}" for t in tags)
    if not tag_lines:
        tag_lines = "[]"
    return (
        "---\n"
        f"created: {_yaml_scalar(row.get('created', ''))}\n"
        f"id: {_yaml_scalar(row.get('id', ''))}\n"
        "tags:\n"
        f"{tag_lines}\n"
        f"title: {_yaml_scalar(row.get('title', ''))}\n"
        f"type: {_yaml_scalar(row.get('type', 'note'))}\n"
        f"updated: {_yaml_scalar(updated)}\n"
        "---\n\n"
    )


def _rewrite_frontmatter(row: dict[str, Any], tags: list[str], updated: str) -> None:
    full = _safe_resolve(str(row.get("path") or ""))
    if not full:
        raise OSError("path invalido")
    body = ""
    if full.exists():
        body = _strip_frontmatter(full.read_text(encoding="utf-8", errors="replace"))
    full.write_text(_frontmatter(row, tags, updated) + body.lstrip(), encoding="utf-8")


def _read_body_meta(rel_path: str) -> tuple[int, str]:
    """Devuelve (body_size_bytes, body_preview_180chars). Silent-fail con
    (0, '') si el .md no existe.
    """
    full = _safe_resolve(rel_path)
    if not full or not full.exists():
        return (0, "")
    try:
        raw = full.read_text(encoding="utf-8", errors="replace")
        body = _strip_frontmatter(raw)
        size = full.stat().st_size
        preview = " ".join(body.split())[:_BODY_PREVIEW_CHARS]
        return (size, preview)
    except Exception:
        return (0, "")


# ── Dupe-cluster cache (key: mtime+size de memvec.db; TTL implícito por
# mtime). El cómputo dura ~1.5s, así que cachear paga.
_DUPE_CACHE: dict[str, Any] = {"key": None, "payload": None}
_DUPE_LOCK = threading.Lock()

# ── Recall-log cache. Scanning 1099 .jsonl files toma ~700ms. Cacheamos
# por TTL en lugar de mtime (rastreo de mtime de 1099 archivos es caro).
_RECALL_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_RECALL_LOCK = threading.Lock()
_RECALL_TTL_SEC = 60.0  # invalidado cada minuto

_RECALL_RE = re.compile(r"\*\*\[([a-f0-9]+)\][^*]+\*\*\s*\(score\s+([0-9.]+)\)")


def _scan_recall_log() -> dict[str, list[tuple[str, float]]]:
    """Walk `~/.claude/projects/**/*.jsonl` y extrae cada evento de
    recall del hook `UserPromptSubmit` de memo.

    Returns:
        {memo_id_prefix_8hex: [(timestamp_iso, score), ...]}

    El prefix es de 8 chars (lo que memo imprime en su markdown output).
    El caller resuelve cada prefix al full id via `meta` table.

    Performance: ~700ms cold para 1099 archivos × ~500KB. Cacheado 60s.
    """
    now = time.time()
    with _RECALL_LOCK:
        if (
            _RECALL_CACHE.get("payload") is not None
            and now - _RECALL_CACHE.get("ts", 0) < _RECALL_TTL_SEC
        ):
            return _RECALL_CACHE["payload"]

    events: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not _CLAUDE_PROJECTS_DIR.exists():
        with _RECALL_LOCK:
            _RECALL_CACHE["ts"] = now
            _RECALL_CACHE["payload"] = {}
        return {}

    for jsonl in _CLAUDE_PROJECTS_DIR.rglob("*.jsonl"):
        try:
            with open(jsonl, errors="replace") as f:
                for line in f:
                    # Cheap pre-filter: salta líneas sin el marker.
                    if "Relevant memories" not in line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") != "attachment":
                        continue
                    att = obj.get("attachment", {})
                    if att.get("hookName") != "UserPromptSubmit":
                        continue
                    content = att.get("content")
                    if isinstance(content, list):
                        content = "\n".join(c if isinstance(c, str) else "" for c in content)
                    if not content:
                        continue
                    ts = obj.get("timestamp", "")
                    for m in _RECALL_RE.finditer(content):
                        events[m.group(1)].append((ts, float(m.group(2))))
        except Exception:
            continue  # silent-fail: 1 archivo roto no rompe el snapshot

    payload = dict(events)
    with _RECALL_LOCK:
        _RECALL_CACHE["ts"] = now
        _RECALL_CACHE["payload"] = payload
    return payload


def _build_recall_index(memo_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Asocia cada full memo_id con sus recalls.

    Memo guarda ids hex de 32 chars y los imprime como prefijo de 8.
    Buscamos cada prefix entre los memo_ids vivos. Si match único →
    asignar. Si match ambiguo (raro: 2 memorias con mismos primeros 8)
    → asignar a todas (over-count tolerable; muy raro estadísticamente).

    Returns:
        {memo_id: {
            "count": int,
            "scores": [float, ...],
            "last_recalled": str (ISO),
            "first_recalled": str (ISO),
            "max_score": float,
        }}
    """
    recall_events = _scan_recall_log()
    if not recall_events:
        return {}

    # Index ids por prefijo 8.
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for mid in memo_ids:
        by_prefix[mid[:8]].append(mid)

    out: dict[str, dict[str, Any]] = {}
    for pfx, evts in recall_events.items():
        for full in by_prefix.get(pfx, []):
            scores = [s for _, s in evts]
            timestamps = [t for t, _ in evts if t]
            out[full] = {
                "count": len(evts),
                "scores": scores,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "first_recalled": min(timestamps) if timestamps else "",
                "last_recalled": max(timestamps) if timestamps else "",
            }
    return out


def _compute_verdict(health: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Sintetiza 4 criterios traffic-light + un veredicto en una línea:
    "Vale la pena memo: SÍ / NO / CON CLEANUP".

    Criterios:
    1. Recall hit rate ≥ 30% → bien usado (sino: poco efectivo).
    2. Avg quality score ≥ 70 → bien capturado.
    3. Near-dupe pct ≤ 10% → señal limpia.
    4. Active 7d ≥ 5 memorias → sistema vivo.

    El user con esto decide en 2 segundos.
    """
    criteria = []

    hit = usage.get("recall_hit_rate_pct", 0)
    criteria.append({
        "label": "Recall hit rate",
        "value": f"{hit}%",
        "status": "good" if hit >= 30 else ("warn" if hit >= 15 else "bad"),
        "detail": f"{usage.get('recalled_count', 0)} de {usage.get('total', 0)} memorias se usaron al menos 1 vez",
        "threshold": "≥ 30% bien · 15-30 ok · < 15 mal",
    })

    avg = health.get("avg_score", 0)
    criteria.append({
        "label": "Calidad capture",
        "value": f"{avg}/100",
        "status": "good" if avg >= 70 else ("warn" if avg >= 50 else "bad"),
        "detail": f"avg quality score sobre las {usage.get('total', 0)} memorias",
        "threshold": "≥ 70 bien · 50-70 ok · < 50 mal",
    })

    dupe = health.get("near_dupe_pct", 0)
    criteria.append({
        "label": "Señal vs ruido",
        "value": f"{dupe}%",
        "status": "good" if dupe <= 10 else ("warn" if dupe <= 25 else "bad"),
        "detail": f"{health.get('near_dupe_count', 0)} memorias en pares near-dupe (dist < 0.12)",
        "threshold": "≤ 10% limpio · 10-25 ok · > 25 ruido alto",
    })

    active = usage.get("active_7d_count", 0)
    criteria.append({
        "label": "Sistema vivo",
        "value": str(active),
        "status": "good" if active >= 5 else ("warn" if active >= 1 else "bad"),
        "detail": f"memorias recalled en los últimos {_ACTIVE_DAYS} días",
        "threshold": "≥ 5 vivo · 1-4 tibio · 0 muerto",
    })

    good = sum(1 for c in criteria if c["status"] == "good")
    bad = sum(1 for c in criteria if c["status"] == "bad")

    if bad >= 2:
        outcome = "no"
        text = "Más ruido que señal: evaluar alternativa (mem0 / letta / zep) o tightening del capture."
    elif good >= 3:
        outcome = "yes"
        text = "Memo está pagando su costo. Hacer cleanup periódico de near-dupes + dead memorias."
    else:
        outcome = "cleanup"
        text = (
            f"Vale la pena con cleanup: borrar {usage.get('dead_count', 0)} dead + "
            f"fusionar {health.get('near_dupe_count', 0)//2} near-dupes."
        )

    return {
        "outcome": outcome,
        "summary": text,
        "criteria": criteria,
    }


def _dupe_cache_key() -> str:
    p = _memvec_db()
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except FileNotFoundError:
        return "missing"


def _compute_dupe_map() -> dict[str, Any]:
    """Para cada memoria, encontrar sus 5 vecinos más cercanos. Identificar
    pares cuya distancia esté bajo `_DUPE_DIST_THRESHOLD` — esos son los
    candidatos honestos a fusionar.

    NO usamos transitive union-find: con embeddings ricos como Qwen3-4B,
    cualquier corpus técnico tiene "vecindarios densos" donde dist < 0.20
    no implica duplicación semántica real (e.g. dos memorias sobre "RAG +
    cache" pueden quedar a 0.18 y NO son dupes). Pairs directos son la
    señal accionable; transitive cluster es ruido.

    Returns:
        {
          "neighbors": {id: [(nbr_id, distance), ...]},  # top 5 SIEMPRE
          "near_dupe_ids": set(id),  # memorias con ≥1 par bajo threshold
          "pairs": [(id1, id2, dist), ...],  # ordenados ascendente, dedupe a-b/b-a
        }
    """
    if not _HAVE_SQLITE_VEC:
        return {"neighbors": {}, "near_dupe_ids": set(), "pairs": []}

    key = _dupe_cache_key()
    with _DUPE_LOCK:
        if _DUPE_CACHE.get("key") == key and _DUPE_CACHE.get("payload"):
            return _DUPE_CACHE["payload"]

    conn = _open_ro(_memvec_db(), with_vec=True)
    if conn is None:
        return {"neighbors": {}, "near_dupe_ids": set(), "pairs": []}

    neighbors: dict[str, list[tuple[str, float]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    near_dupe_ids: set[str] = set()

    try:
        ids = [r["id"] for r in conn.execute("SELECT id FROM meta")]
        for mid in ids:
            row = conn.execute("SELECT embedding FROM vec WHERE id = ?", (mid,)).fetchone()
            if not row:
                continue
            knn_rows = conn.execute(
                "SELECT id, distance FROM vec WHERE embedding MATCH ? AND k = 6",
                (row["embedding"],),
            ).fetchall()
            top: list[tuple[str, float]] = []
            for r in knn_rows:
                if r["id"] == mid:
                    continue
                top.append((r["id"], float(r["distance"])))
            neighbors[mid] = top[:5]
            for nid, dist in top:
                if dist < _DUPE_DIST_THRESHOLD:
                    a, b = sorted([mid, nid])
                    if (a, b) not in seen_pairs:
                        seen_pairs.add((a, b))
                        pairs.append((a, b, dist))
                        near_dupe_ids.add(a)
                        near_dupe_ids.add(b)
    finally:
        conn.close()

    pairs.sort(key=lambda x: x[2])

    payload = {
        "neighbors": neighbors,
        "near_dupe_ids": near_dupe_ids,
        "pairs": pairs,
    }
    with _DUPE_LOCK:
        _DUPE_CACHE["key"] = key
        _DUPE_CACHE["payload"] = payload
    return payload


def _score_memo(
    *,
    type_: str,
    tags: list[str],
    body_size: int,
    updated_dt: datetime | None,
    in_dupe_cluster: bool,
) -> dict[str, Any]:
    """Calcula el quality score 0-100 + breakdown explícito de qué pesa.

    Devolver el breakdown habilita explicar la nota al user — no es
    una caja negra.
    """
    breakdown: dict[str, int] = {}

    if type_ in _TYPE_ACTIONABLE:
        breakdown["actionable"] = _W_ACTIONABLE
    elif type_ in _TYPE_INFORMATIONAL:
        breakdown["actionable"] = _W_ACTIONABLE // 2
    else:
        breakdown["actionable"] = 0

    if len(tags) >= 2:
        breakdown["tags"] = _W_TAGS
    elif len(tags) == 1:
        breakdown["tags"] = _W_TAGS // 2
    else:
        breakdown["tags"] = 0

    if _SIZE_MIN_OK <= body_size <= _SIZE_MAX_OK:
        breakdown["size"] = _W_SIZE
    elif body_size > 0:
        breakdown["size"] = _W_SIZE // 2
    else:
        breakdown["size"] = 0

    if updated_dt is not None:
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - updated_dt).days
        if age_days <= _FRESH_DAYS:
            breakdown["fresh"] = _W_FRESH
        elif age_days <= _STALE_DAYS:
            breakdown["fresh"] = _W_FRESH // 2
        else:
            breakdown["fresh"] = 0
    else:
        breakdown["fresh"] = 0

    breakdown["unique"] = 0 if in_dupe_cluster else _W_UNIQUE

    score = sum(breakdown.values())
    return {"score": score, "breakdown": breakdown}


def snapshot(limit: int = _RECENT_DEFAULT, type_filter: str | None = None) -> dict:
    """Snapshot completo para `/memo`. Devuelve estructura tipada lista
    para hidratar la UI sin más procesamiento del lado del cliente.
    """
    out: dict[str, Any] = {
        "ok": True,
        "memo_dir": str(_memo_dir()),
        "memory_dir": str(_memory_dir()),
        "vault_path": str(_vault_path()),
        "totals": {"all": 0, "by_type": []},
        "activity": {
            "saved_today": 0,
            "saved_7d": 0,
            "saved_30d": 0,
            "events_total": 0,
        },
        "health": {
            "actionable_count": 0,
            "actionable_pct": 0,
            "tagless_count": 0,
            "tagless_pct": 0,
            "tiny_count": 0,
            "huge_count": 0,
            "stale_count": 0,
            "stale_pct": 0,
            "near_dupe_count": 0,
            "near_dupe_pct": 0,
            "avg_score": 0,
        },
        "saves_timeline": [],
        "dupe_pairs": [],
        "usage": {
            "total": 0,
            "recalled_count": 0,
            "recall_hit_rate_pct": 0,
            "dead_count": 0,
            "active_7d_count": 0,
            "total_recall_events": 0,
            "top_recalled": [],
            "dead_memorias": [],
        },
        "utility": {
            "coverage": {},
            "quality_buckets": [],
            "type_usefulness": [],
        },
        "verdict": {"outcome": "unknown", "summary": "", "criteria": []},
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

    dupe_map = _compute_dupe_map()
    near_dupe_ids = dupe_map["near_dupe_ids"]
    pairs = dupe_map["pairs"]
    neighbors_global = dupe_map["neighbors"]

    try:
        out["totals"]["all"] = mv.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
        out["totals"]["by_type"] = [
            {"type": r["type"], "count": r["c"]}
            for r in mv.execute(
                "SELECT type, COUNT(*) AS c FROM meta GROUP BY type ORDER BY c DESC"
            )
        ]

        # ── Health pass: scan TODAS las memorias (no solo recent) para
        # tener counts globales fiables. Body sizes vienen del filesystem
        # — cachear con el snapshot.
        all_rows = mv.execute(
            "SELECT id, title, type, tags, updated, created, path FROM meta"
        ).fetchall()

        # ── Recall index: cuántas veces / cuándo se usó cada memoria.
        recall_index = _build_recall_index([r["id"] for r in all_rows])

        tagless = tiny = huge = stale = actionable = 0
        scores_sum = 0
        quality_counts: Counter[str] = Counter()
        type_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "recalled": 0,
                "active_7d": 0,
                "recall_events": 0,
                "score_sum": 0,
            }
        )
        tag_counter: Counter[str] = Counter()
        body_meta_cache: dict[str, tuple[int, str]] = {}
        now_utc = datetime.now(timezone.utc)
        cutoff_7d_for_types = now_utc - timedelta(days=_ACTIVE_DAYS)

        for r in all_rows:
            tags = _parse_tags(r["tags"])
            type_ = (r["type"] or "").lower()
            updated_dt = _parse_iso(r["updated"])
            size, preview = _read_body_meta(r["path"])
            body_meta_cache[r["id"]] = (size, preview)

            if not tags:
                tagless += 1
            else:
                for t in tags:
                    tag_counter[t] += 1

            if 0 < size < _SIZE_MIN_OK:
                tiny += 1
            if size > _SIZE_MAX_OK:
                huge += 1
            if updated_dt:
                age = (datetime.now(timezone.utc) - (
                    updated_dt if updated_dt.tzinfo else updated_dt.replace(tzinfo=timezone.utc)
                )).days
                if age > _STALE_DAYS:
                    stale += 1
            if type_ in _TYPE_ACTIONABLE:
                actionable += 1

            scored = _score_memo(
                type_=type_,
                tags=tags,
                body_size=size,
                updated_dt=updated_dt,
                in_dupe_cluster=r["id"] in near_dupe_ids,
            )
            score = int(scored["score"])
            scores_sum += score
            if score >= 80:
                quality_counts["Alta"] += 1
            elif score >= 60:
                quality_counts["Media"] += 1
            elif score >= 40:
                quality_counts["Baja"] += 1
            else:
                quality_counts["Ruido"] += 1

            rinfo = recall_index.get(r["id"], {})
            cnt = int(rinfo.get("count", 0))
            tstat = type_stats[type_ or "unknown"]
            tstat["count"] += 1
            tstat["score_sum"] += score
            tstat["recall_events"] += cnt
            if cnt > 0:
                tstat["recalled"] += 1
            last_recalled = rinfo.get("last_recalled", "")
            if last_recalled:
                last_dt = _parse_iso(last_recalled)
                if last_dt:
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if last_dt >= cutoff_7d_for_types:
                        tstat["active_7d"] += 1

        total = out["totals"]["all"] or 1  # avoid /0
        out["health"] = {
            "actionable_count": actionable,
            "actionable_pct": round(actionable * 100 / total, 1),
            "tagless_count": tagless,
            "tagless_pct": round(tagless * 100 / total, 1),
            "tiny_count": tiny,
            "huge_count": huge,
            "stale_count": stale,
            "stale_pct": round(stale * 100 / total, 1),
            "near_dupe_count": len(near_dupe_ids),
            "near_dupe_pct": round(len(near_dupe_ids) * 100 / total, 1),
            "avg_score": round(scores_sum / total, 1),
        }
        out["tags_top"] = [
            {"tag": t, "count": c} for t, c in tag_counter.most_common(20)
        ]
        out["utility"]["quality_buckets"] = [
            {
                "label": label,
                "count": int(quality_counts.get(label, 0)),
                "pct": round(quality_counts.get(label, 0) * 100 / total, 1),
            }
            for label in ("Alta", "Media", "Baja", "Ruido")
        ]
        out["utility"]["type_usefulness"] = [
            {
                "type": type_,
                "count": int(stats["count"]),
                "recalled": int(stats["recalled"]),
                "active_7d": int(stats["active_7d"]),
                "recall_events": int(stats["recall_events"]),
                "recall_rate_pct": round(stats["recalled"] * 100 / (stats["count"] or 1), 1),
                "avg_score": round(stats["score_sum"] / (stats["count"] or 1), 1),
            }
            for type_, stats in sorted(
                type_stats.items(),
                key=lambda item: (-item[1]["recall_events"], -item[1]["recalled"], item[0]),
            )
        ]

        # ── Recent list con scoring + previews.
        if type_filter:
            recent_rows = [r for r in all_rows if (r["type"] or "").lower() == type_filter]
        else:
            recent_rows = list(all_rows)
        recent_rows.sort(key=lambda r: r["updated"] or "", reverse=True)
        recent_rows = recent_rows[:limit]

        recent: list[dict[str, Any]] = []
        for r in recent_rows:
            tags = _parse_tags(r["tags"])
            type_ = (r["type"] or "").lower()
            updated_dt = _parse_iso(r["updated"])
            size, preview = body_meta_cache.get(r["id"], (0, ""))
            scored = _score_memo(
                type_=type_,
                tags=tags,
                body_size=size,
                updated_dt=updated_dt,
                in_dupe_cluster=r["id"] in near_dupe_ids,
            )
            rinfo = recall_index.get(r["id"], {})
            last_recalled = rinfo.get("last_recalled", "")
            last_recall_dt = _parse_iso(last_recalled) if last_recalled else None
            recent.append({
                "id": r["id"],
                "title": r["title"] or "(sin título)",
                "type": type_,
                "tags": tags,
                "updated": r["updated"],
                "created": r["created"],
                "ago": _humanize_ago(updated_dt),
                **_path_fields(r["path"]),
                "body_size": size,
                "body_preview": preview,
                "score": scored["score"],
                "score_breakdown": scored["breakdown"],
                "in_dupe_cluster": r["id"] in near_dupe_ids,
                "neighbor_count": sum(
                    1 for _, d in neighbors_global.get(r["id"], [])
                    if d < _DUPE_DIST_THRESHOLD
                ),
                "recall_count": int(rinfo.get("count", 0)),
                "max_recall_score": round(float(rinfo.get("max_score", 0.0)), 2),
                "last_recalled_ago": _humanize_ago(last_recall_dt),
            })
        out["recent"] = recent

        # ── Top dupe pairs: pares directos bajo threshold, sorted por
        # distancia (más cercanos primero). El user los puede revisar uno
        # por uno como candidato a fusionar/borrar.
        title_by_id = {r["id"]: r["title"] for r in all_rows}
        type_by_id = {r["id"]: (r["type"] or "") for r in all_rows}
        out["dupe_pairs"] = [
            {
                "a": {
                    "id": a,
                    "title": title_by_id.get(a, "(?)"),
                    "type": type_by_id.get(a, ""),
                },
                "b": {
                    "id": b,
                    "title": title_by_id.get(b, "(?)"),
                    "type": type_by_id.get(b, ""),
                },
                "distance": round(d, 4),
            }
            for a, b, d in pairs[:30]
        ]
    finally:
        mv.close()

    # ── Activity + timeline desde history.db.
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

            # ── Saves timeline (últimos 30 días, bucketed por día local).
            tl_rows = h.execute(
                "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS c "
                "FROM events WHERE ts >= ? AND op = 'save' "
                "GROUP BY day ORDER BY day",
                (iso_30d,),
            ).fetchall()
            tl_map = {r["day"]: int(r["c"]) for r in tl_rows}
            timeline: list[dict[str, Any]] = []
            today = datetime.now().date()
            for offset in range(30, -1, -1):
                d = today - timedelta(days=offset)
                key = d.isoformat()
                timeline.append({"day": key, "count": tl_map.get(key, 0)})
            out["saves_timeline"] = timeline
        finally:
            h.close()

    # ── Usage block: la pregunta real "¿se usan?". Computado acá porque
    # necesita `recall_index` + `all_rows` + creación-age para distinguir
    # "dead" de "recién nacido y todavía sin chance de matchear".
    total = out["totals"]["all"] or 1
    now_utc = datetime.now(timezone.utc)
    cutoff_7d = now_utc - timedelta(days=_ACTIVE_DAYS)
    cutoff_dead = now_utc - timedelta(days=_DEAD_MIN_AGE_DAYS)
    total_recall_events = 0
    active_7d_count = 0
    dead_memorias: list[dict[str, Any]] = []
    top_recalled_raw: list[dict[str, Any]] = []
    for r in all_rows:
        rinfo = recall_index.get(r["id"], {})
        cnt = int(rinfo.get("count", 0))
        total_recall_events += cnt
        last_recalled = rinfo.get("last_recalled", "")
        if last_recalled:
            last_dt = _parse_iso(last_recalled)
            if last_dt:
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if last_dt >= cutoff_7d:
                    active_7d_count += 1
        if cnt > 0:
            top_recalled_raw.append({
                "id": r["id"],
                "title": r["title"] or "(sin título)",
                "type": (r["type"] or "").lower(),
                **_path_fields(r["path"]),
                "count": cnt,
                "max_score": round(float(rinfo.get("max_score", 0.0)), 2),
                "last_recalled_ago": _humanize_ago(_parse_iso(last_recalled)),
            })
        else:
            created_dt = _parse_iso(r["created"])
            if created_dt:
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if created_dt <= cutoff_dead:
                    dead_memorias.append({
                        "id": r["id"],
                        "title": r["title"] or "(sin título)",
                        "type": (r["type"] or "").lower(),
                        **_path_fields(r["path"]),
                        "created": r["created"],
                        "age": _humanize_ago(created_dt),
                    })

    top_recalled_raw.sort(key=lambda x: -x["count"])
    dead_memorias.sort(key=lambda x: x["created"] or "")

    recalled_count = len(recall_index)
    out["usage"] = {
        "total": int(out["totals"]["all"]),
        "recalled_count": recalled_count,
        "recall_hit_rate_pct": round(recalled_count * 100 / total, 1),
        "dead_count": len(dead_memorias),
        "active_7d_count": active_7d_count,
        "total_recall_events": total_recall_events,
        "top_recalled": top_recalled_raw[:15],
        "dead_memorias": dead_memorias[:30],
    }
    out["utility"]["coverage"] = {
        "total": int(out["totals"]["all"]),
        "recalled": recalled_count,
        "active_7d": active_7d_count,
        "never_recalled": max(0, int(out["totals"]["all"]) - recalled_count),
        "dead": len(dead_memorias),
        "recall_hit_rate_pct": round(recalled_count * 100 / total, 1),
        "active_7d_pct": round(active_7d_count * 100 / total, 1),
        "dead_pct": round(len(dead_memorias) * 100 / total, 1),
    }
    out["verdict"] = _compute_verdict(out["health"], out["usage"])

    return out


def note_detail(memo_id: str | None = None, path: str | None = None) -> dict:
    """Detalle de UNA memoria — metadata + body + score breakdown + vecinos."""
    out: dict[str, Any] = {"ok": False, "error": "", "memo": None, "neighbors": []}

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
        except Exception as exc:
            body = f"(error leyendo .md: {exc})"
    else:
        body = "(archivo .md no encontrado — corré `memo doctor`)"

    body_size = full.stat().st_size if full and full.exists() else 0
    tags = _parse_tags(row["tags"])
    updated_dt = _parse_iso(row["updated"])
    dupe_map = _compute_dupe_map()
    in_dupe_cluster = row["id"] in dupe_map["near_dupe_ids"]
    scored = _score_memo(
        type_=(row["type"] or "").lower(),
        tags=tags,
        body_size=body_size,
        updated_dt=updated_dt,
        in_dupe_cluster=in_dupe_cluster,
    )

    # ── Neighbors enriched: id + title + type + distance.
    raw_neighbors = dupe_map["neighbors"].get(row["id"], [])
    neighbor_ids = [nid for nid, _ in raw_neighbors]
    titles: dict[str, sqlite3.Row] = {}
    if neighbor_ids:
        mv2 = _open_ro(_memvec_db())
        if mv2 is not None:
            try:
                placeholders = ",".join("?" * len(neighbor_ids))
                for r in mv2.execute(
                    f"SELECT id, title, type FROM meta WHERE id IN ({placeholders})",
                    neighbor_ids,
                ):
                    titles[r["id"]] = r
            finally:
                mv2.close()
    neighbors_list = [
        {
            "id": nid,
            "distance": round(dist, 4),
            "near_dupe": dist < _DUPE_DIST_THRESHOLD,
            "title": titles.get(nid, {"title": "(?)"})["title"] if nid in titles else "(?)",
            "type": titles.get(nid, {"type": ""})["type"] if nid in titles else "",
        }
        for nid, dist in raw_neighbors
    ]

    out["ok"] = True
    out["memo"] = {
        "id": row["id"],
        "title": row["title"] or "(sin título)",
        "type": (row["type"] or "note").lower(),
        "tags": tags,
        "created": row["created"],
        "updated": row["updated"],
        "ago": _humanize_ago(updated_dt),
        **_path_fields(row["path"]),
        "body": body,
        "body_size": body_size,
        "score": scored["score"],
        "score_breakdown": scored["breakdown"],
        "in_dupe_cluster": in_dupe_cluster,
    }
    out["neighbors"] = neighbors_list
    return out


def search(query: str, limit: int = 20) -> dict:
    """FTS5 search sobre title + tags + body. Devuelve la misma shape de
    `recent` (id, title, type, tags, score, etc.) para que el frontend
    pueda renderizar con el mismo template.

    `query` se sanitiza: caracteres `"` reemplazados; espacios → AND
    implícito de FTS5. Si vacío, devolver lista vacía.
    """
    out: dict[str, Any] = {"ok": True, "results": [], "query": query}
    query = (query or "").strip()
    if not query:
        return out

    safe = query.replace('"', " ").replace("'", " ")
    # Wrap each word as prefix-search for usabilidad: "cad" matchea "Caddy".
    parts = [f'{w}*' for w in safe.split() if w]
    if not parts:
        return out
    fts_query = " AND ".join(parts)

    limit = max(1, min(int(limit or 20), 100))

    mv = _open_ro(_memvec_db())
    if mv is None:
        out["ok"] = False
        out["error"] = "memvec.db no encontrado"
        return out

    dupe_map = _compute_dupe_map()
    near_dupe_ids = dupe_map["near_dupe_ids"]

    try:
        rows = mv.execute(
            "SELECT m.id, m.title, m.type, m.tags, m.updated, m.created, m.path, "
            "       rank "
            "FROM fts JOIN meta m ON m.id = fts.id "
            "WHERE fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        out["ok"] = False
        out["error"] = f"FTS query inválida: {e}"
        mv.close()
        return out

    try:
        results = []
        for r in rows:
            tags = _parse_tags(r["tags"])
            updated_dt = _parse_iso(r["updated"])
            size, preview = _read_body_meta(r["path"])
            scored = _score_memo(
                type_=(r["type"] or "").lower(),
                tags=tags,
                body_size=size,
                updated_dt=updated_dt,
                in_dupe_cluster=r["id"] in near_dupe_ids,
            )
            results.append({
                "id": r["id"],
                "title": r["title"] or "(sin título)",
                "type": (r["type"] or "note").lower(),
                "tags": tags,
                "updated": r["updated"],
                "ago": _humanize_ago(updated_dt),
                **_path_fields(r["path"]),
                "body_size": size,
                "body_preview": preview,
                "score": scored["score"],
                "in_dupe_cluster": r["id"] in near_dupe_ids,
                "fts_rank": float(r["rank"]) if r["rank"] is not None else 0.0,
            })
        out["results"] = results
    finally:
        mv.close()

    return out
