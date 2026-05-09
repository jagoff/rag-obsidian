"""Sessions — multi-turn conversation persistence.

Phase 5 de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el sub-sistema de sesiones (storage + lifecycle + history
window + summary compression) desde `rag/__init__.py`.

## Contexto

Persiste multi-turn conversation para que follow-ups ("profundizá",
pronouns sin antecedente) puedan referenciar prior turns. Compartido
entre `rag chat`, `rag query --continue`, MCP `rag_query(session_id=...)`,
y el listener WhatsApp (que pasa `wa:<jid>` como id).

Storage: un JSON file por session bajo `SESSIONS_DIR`.
`LAST_SESSION_FILE` guarda el id más reciente para que `--continue` /
`--resume` lo defaulteen. Caller-supplied ids permitidos (necesario
para JIDs de WhatsApp), validados contra `SESSION_ID_RE`.

## Lazy imports

`_silent_log` vive en `rag/__init__.py`. `_compress_turns` también.
Lazy adentro de cada función para evitar circular import.

## Re-export

`rag/__init__.py` hace `from rag._sessions import *  # noqa`.
Preserva 100% compat con call sites históricos.
"""

from __future__ import annotations

import fcntl
import json
import re
import secrets
import time
from datetime import datetime
from pathlib import Path

__all__ = [
    "SESSIONS_DIR",
    "LAST_SESSION_FILE",
    "SESSION_TTL_DAYS",
    "SESSION_MAX_TURNS",
    "SESSION_HISTORY_WINDOW",
    "SESSION_COMPRESSION_THRESHOLD",
    "SESSION_SUMMARY_VERSION",
    "SESSION_ID_RE",
    "new_session_id",
    "_valid_session_id",
    "session_path",
    "load_session",
    "save_session",
    "ensure_session",
    "append_turn",
    "session_history",
    "session_summary",
    "last_session_id",
    "_set_last_session",
    "list_sessions",
    "cleanup_sessions",
]


SESSIONS_DIR = Path.home() / ".local/share/obsidian-rag/sessions"
LAST_SESSION_FILE = Path.home() / ".local/share/obsidian-rag/last_session"
SESSION_TTL_DAYS = 30
SESSION_MAX_TURNS = 50           # cap per file — keeps JSON small, bounds retrieval context
# 2026-04-28 P3 (continuidad multi-turn): subido 6→10 (5 turnos user+assistant
# vs 3 antes). Caso típico que tapaba la ventana de 6: turn 1 (long answer
# del LLM) + turn 2 (clarification corta del user) + turn 3 (segunda
# respuesta) ya consumía los 6 mensajes — el turn 4 perdía contexto del
# turn 1 que era la pregunta original. Cap absoluto sigue en
# SESSION_MAX_TURNS=50; este es el cap del WINDOW que se manda al LLM.
SESSION_HISTORY_WINDOW = 10      # last N messages fed to reformulate_query / LLM
SESSION_COMPRESSION_THRESHOLD = 7  # turn count at which compressed_history kicks in
SESSION_SUMMARY_VERSION = 1      # bump if compressor prompt/format changes (invalidates cache)

# `@` is required for WhatsApp jids (e.g. `wa:120363...@g.us:vault`); without
# it the regex rejected real WA ids and ensure_session minted a fresh UUID,
# breaking session continuity AND analytics source-attribution.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,80}$")


def new_session_id() -> str:
    """Short opaque id: hex timestamp + 6 random hex chars (sortable + unique)."""
    return f"{int(time.time()):x}-{secrets.token_hex(3)}"


def _valid_session_id(sid: str) -> bool:
    return bool(SESSION_ID_RE.match(sid))


def session_path(sid: str) -> Path:
    if not _valid_session_id(sid):
        raise ValueError(f"invalid session id: {sid!r}")
    return SESSIONS_DIR / f"{sid}.json"


def load_session(sid: str) -> dict | None:
    """Read a session file. Returns None if missing, invalid id, or unreadable.

    If the file exists but is corrupt (truncated mid-write before the flock
    hardening of 2026-04-24, or hit by a crash), quarantine it to
    `<path>.corrupt-<ts>` and return None. Otherwise the same corrupted
    "last session" file would log JSONDecodeError every time the chat
    boots — which is exactly what `session_load_json` (60 errors / day in
    the audit) was tracking.
    """
    from rag import _silent_log  # noqa: PLC0415

    try:
        p = session_path(sid)
    except ValueError:
        return None
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _silent_log("session_load_json", exc)
        # Quarantine corrupt file so the next read doesn't log the same
        # error indefinitely. Best-effort: if rename fails (permission,
        # cross-device), just leave it be — the silent log already fired.
        try:
            backup = p.with_suffix(f".json.corrupt-{int(time.time())}")
            p.rename(backup)
        except Exception:  # pragma: no cover - rescue path
            pass
        return None


def save_session(sess: dict) -> None:
    """Atomic write via tmp file + replace + per-session flock.

    `tmp.replace(p)` es atómico a nivel filesystem (APFS/ext4/ntfs
    garantizan que nadie ve un archivo a medio escribir). PERO si 2
    writers con copias in-memory divergentes hacen save_session
    concurrente, el último en llamar `replace()` gana y la otra
    escritura se pierde silenciosamente (ABA race). Escenarios donde
    esto puede pasar:
      - User dispara 2 chats rápidos para la misma sesión (Enter doble,
        redo mientras original está corriendo).
      - Web handler + episodic writer thread escriben la misma sesión
        al mismo tiempo.
      - Dos web processes en distintos puertos comparten SESSIONS_DIR.

    Mitigación (2026-04-24 audit hardening): flock exclusivo sobre un
    `.lock` sidecar file por sesión. Los writers concurrentes se
    serializan — siguen siendo "last writer wins" (no hay merge
    semántico) pero al menos no escriben encimados. El lock file se
    crea on-demand y permanece (no hay cleanup explícito) porque son
    byte-free sidecars. Si alguien borra SESSIONS_DIR completo, se
    regenera transparentemente.

    Esta fix NO resuelve el problema de merge (turnos perdidos si 2
    writers tienen in-memory state distinto); sólo previene
    corrupción del archivo en caso de que 2 threads escriban
    bit-concurrent. Para el merge propio haría falta un pattern
    read-modify-write bajo el mismo lock, que es un refactor mayor
    documentado en el issue tracker.
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sess["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p = session_path(sess["id"])
    tmp = p.with_suffix(".json.tmp")
    lock_path = p.with_suffix(".json.lock")

    # Abrir el lock file (se crea si no existe). `a+` para garantizar que
    # otro proceso nunca lo vea como "empty" durante el open initial —
    # APFS nunca muestra tamaño intermediate para files creados así.
    lock_fh = lock_path.open("a+")
    try:
        # Blocking lock — writers concurrentes se serializan. Típicamente
        # 1-5ms de wait bajo carga normal, mucho menos que el `write_text`
        # siguiente. Si el lock está stuck (proceso muerto tenía el lock),
        # fcntl lo libera cuando el FD se cierra al morir el proceso.
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        tmp.write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    finally:
        # Unlock + close. El close() per-se libera el flock (behavior
        # de fcntl en POSIX), así que en caso de que el flock UN falle
        # por cualquier razón exótica, el close() es el net de seguridad.
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fh.close()

    _set_last_session(sess["id"])


def ensure_session(sid: str | None, *, mode: str) -> dict:
    """Return an existing session or a fresh one. Caller-supplied ids are kept
    verbatim if valid; otherwise a new random id is minted.
    """
    if sid:
        existing = load_session(sid)
        if existing:
            return existing
        new_id = sid if _valid_session_id(sid) else new_session_id()
    else:
        new_id = new_session_id()
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": new_id,
        "created_at": now,
        "updated_at": now,
        "mode": mode,
        "turns": [],
    }


def append_turn(sess: dict, turn: dict) -> None:
    """Append one turn to the session, capping total stored turns."""
    sess.setdefault("turns", []).append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        **turn,
    })
    if len(sess["turns"]) > SESSION_MAX_TURNS:
        sess["turns"] = sess["turns"][-SESSION_MAX_TURNS:]


def session_history(sess: dict, window: int = SESSION_HISTORY_WINDOW) -> list[dict]:
    """Flatten session turns into `[{role, content}]` for reformulate_query /
    the chat LLM. Returns the last `window` messages.
    """
    msgs: list[dict] = []
    for turn in sess.get("turns", []):
        q = turn.get("q")
        a = turn.get("a")
        if q:
            msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs[-window:]


def session_summary(
    sess: dict,
    *,
    window: int = SESSION_HISTORY_WINDOW,
    threshold: int = SESSION_COMPRESSION_THRESHOLD,
) -> str | None:
    """Lazily compute / cache a compressed summary of turns aged out of the
    raw history window. Returns the summary string, or None when the session
    is short enough to feed raw turns directly.

    Closes the empirical chains-vs-singles gap in `rag eval`: by turn 10+ the
    helper reformulator was being fed 6 raw messages with the topic-anchoring
    first turns already dropped, losing context. The summary covers `turns[:n
    - window]` and is prepended to `reformulate_query` as a labelled section.

    Mutates `sess["compressed_history"]` when (re)computing — caller is
    responsible for calling `save_session()` to persist.
    """
    from rag import _compress_turns  # noqa: PLC0415

    turns = sess.get("turns") or []
    n = len(turns)
    if n < threshold:
        return None
    need_until = n - window  # exclusive idx of turns to summarize
    if need_until <= 0:
        return None
    cached = sess.get("compressed_history") or {}
    if (
        cached.get("version") == SESSION_SUMMARY_VERSION
        and cached.get("covers_until_idx", 0) >= need_until
    ):
        return cached.get("summary") or None
    summary_text = _compress_turns(turns[:need_until])
    if not summary_text:
        return cached.get("summary") if cached else None
    sess["compressed_history"] = {
        "version": SESSION_SUMMARY_VERSION,
        "covers_until_idx": need_until,
        "summary": summary_text,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    return summary_text


def last_session_id() -> str | None:
    try:
        sid = LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return sid or None


def _set_last_session(sid: str) -> None:
    # Atomic write: tmp + replace. Un crash entre el `write_text` y el
    # próximo read deja `LAST_SESSION_FILE` vacío → `last_session_id()`
    # devuelve None → `rag chat --continue` mint-ea una sesión nueva
    # y pierde la continuity UX. Barato de mitigar con el mismo patrón
    # que `save_session` / `_save_vaults_config`.
    from rag import _silent_log  # noqa: PLC0415

    try:
        LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = LAST_SESSION_FILE.with_suffix(LAST_SESSION_FILE.suffix + ".tmp")
        tmp.write_text(sid, encoding="utf-8")
        tmp.replace(LAST_SESSION_FILE)
    except Exception as exc:
        _silent_log("last_session_write", exc)


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent session summaries (newest first) — id, turn count, first question."""
    from rag import _silent_log  # noqa: PLC0415

    if not SESSIONS_DIR.is_dir():
        return []
    out: list[dict] = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            _silent_log("session_list_parse", exc)
            continue
        turns = s.get("turns", [])
        first_q = next((t.get("q", "") for t in turns if t.get("q")), "")
        out.append({
            "id": s.get("id", p.stem),
            "updated_at": s.get("updated_at", ""),
            "created_at": s.get("created_at", ""),
            "turns": len(turns),
            "first_q": first_q[:80],
            "mode": s.get("mode", ""),
        })
    out.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return out[:limit]


def cleanup_sessions(ttl_days: int = SESSION_TTL_DAYS) -> int:
    """Remove session files older than `ttl_days` by mtime. Returns count removed."""
    if not SESSIONS_DIR.is_dir():
        return 0
    cutoff = time.time() - ttl_days * 86400
    removed = 0
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            pass
    return removed
