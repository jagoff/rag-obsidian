"""System-wide blacklist policy for RAG sources.

The durable store is SQLite so web workers, CLI commands, and background
daemons share the same policy atomically:

    ~/.local/share/obsidian-rag/blacklist.db

The older JSON config is read once as a migration source:

    ~/.config/obsidian-rag/blacklist.json
"""
from __future__ import annotations

from difflib import SequenceMatcher
import fnmatch
import json
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

DEFAULT_BLOCKED_CHATS = frozenset({"Cloud Services"})

_CONFIG_PATH = Path(
    os.environ.get("RAG_BLACKLIST_PATH", "~/.config/obsidian-rag/blacklist.json")
).expanduser()
_DB_PATH = Path(
    os.environ.get(
        "RAG_BLACKLIST_DB_PATH",
        str(
            Path(os.environ.get("OBSIDIAN_RAG_STATE_DIR", "~/.local/share/obsidian-rag"))
            .expanduser()
            / "blacklist.db"
        ),
    )
).expanduser()
_LEGACY_IGNORED_PATH = Path(
    os.environ.get(
        "RAG_IGNORED_NOTES_PATH",
        "~/.local/share/obsidian-rag/ignored_notes.json",
    )
).expanduser()

_VALID_KIND_KEYS = frozenset({
    "chats",
    "people",
    "topics",
    "words",
    "fuzzy_words",
    "paths",
    "path_prefixes",
    "path_globs",
})

_CACHE: tuple[object, dict[str, list[str]]] | None = None
_LEGACY_CACHE: tuple[float | None, set[str]] | None = None


def normalize_blacklist_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def _clean_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lstrip("/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _coerce_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _empty_config() -> dict[str, list[str]]:
    return {
        "chats": sorted(DEFAULT_BLOCKED_CHATS),
        "people": [],
        "topics": [],
        "words": [],
        "fuzzy_words": [],
        "paths": [],
        "path_prefixes": [],
        "path_globs": [],
    }


def _normalize_config(raw: Any) -> dict[str, list[str]]:
    cfg = _empty_config()
    if not isinstance(raw, dict):
        return cfg

    chats = (
        _coerce_list(raw.get("chats"))
        + _coerce_list(raw.get("groups"))
        + _coerce_list(raw.get("chat_names"))
    )
    people = _coerce_list(raw.get("people")) + _coerce_list(raw.get("persons"))
    topics = (
        _coerce_list(raw.get("topics"))
        + _coerce_list(raw.get("terms"))
        + _coerce_list(raw.get("keywords"))
    )
    words = (
        _coerce_list(raw.get("words"))
        + _coerce_list(raw.get("exact_words"))
        + _coerce_list(raw.get("palabras"))
        + _coerce_list(raw.get("palabras_exactas"))
    )
    fuzzy_words = (
        _coerce_list(raw.get("fuzzy_words"))
        + _coerce_list(raw.get("similar_words"))
        + _coerce_list(raw.get("palabras_parecidas"))
        + _coerce_list(raw.get("near_words"))
    )
    paths = _coerce_list(raw.get("paths")) + _coerce_list(raw.get("routes"))
    prefixes = _coerce_list(raw.get("path_prefixes")) + _coerce_list(raw.get("prefixes"))
    globs = _coerce_list(raw.get("path_globs")) + _coerce_list(raw.get("globs"))

    cfg["chats"] = sorted(set(cfg["chats"]) | set(chats))
    cfg["people"] = sorted(set(people))
    cfg["topics"] = sorted(set(topics))
    cfg["words"] = sorted(set(words))
    cfg["fuzzy_words"] = sorted(set(fuzzy_words))
    cfg["paths"] = sorted({_clean_path(p) for p in paths if _clean_path(p)})
    cfg["path_prefixes"] = sorted(
        p if p.endswith("/") else f"{p}/"
        for p in {_clean_path(p) for p in prefixes if _clean_path(p)}
    )
    cfg["path_globs"] = sorted({_clean_path(p) for p in globs if _clean_path(p)})
    return cfg


def _item_identity(kind_key: str, value: Any) -> str:
    if kind_key in {"paths", "path_prefixes", "path_globs"}:
        return _clean_path(value)
    return normalize_blacklist_text(value)


def _display_value(kind_key: str, value: Any) -> str:
    if kind_key in {"paths", "path_prefixes", "path_globs"}:
        return _clean_path(value)
    return str(value or "").strip()


def _kind_key(kind: str) -> str:
    k = normalize_blacklist_text(kind).replace("-", "_")
    aliases = {
        "chat": "chats",
        "group": "chats",
        "grupo": "chats",
        "person": "people",
        "persona": "people",
        "topic": "topics",
        "tema": "topics",
        "term": "topics",
        "keyword": "topics",
        "word": "words",
        "palabra": "words",
        "exact_word": "words",
        "word_exact": "words",
        "palabra_exacta": "words",
        "fuzzy_word": "fuzzy_words",
        "similar_word": "fuzzy_words",
        "near_word": "fuzzy_words",
        "palabra_parecida": "fuzzy_words",
        "path": "paths",
        "ruta": "paths",
        "path_prefix": "path_prefixes",
        "prefix": "path_prefixes",
        "path_glob": "path_globs",
        "glob": "path_globs",
    }
    key = aliases.get(k, k)
    if key not in _VALID_KIND_KEYS:
        raise ValueError(f"blacklist kind inválido: {kind}")
    return key


def blacklist_path() -> Path:
    return _DB_PATH


def blacklist_db_path() -> Path:
    return _DB_PATH


def legacy_blacklist_json_path() -> Path:
    return _CONFIG_PATH


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH, timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS blacklist_items (
          kind TEXT NOT NULL,
          value TEXT NOT NULL,
          normalized TEXT NOT NULL,
          created_at REAL NOT NULL,
          PRIMARY KEY (kind, normalized)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS blacklist_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )


def _upsert_item(con: sqlite3.Connection, kind_key: str, value: Any) -> None:
    if kind_key not in _VALID_KIND_KEYS:
        return
    display = _display_value(kind_key, value)
    normalized = _item_identity(kind_key, display)
    if not normalized:
        return
    con.execute(
        """
        INSERT OR IGNORE INTO blacklist_items(kind, value, normalized, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (kind_key, display, normalized, time.time()),
    )


def _load_legacy_json_blacklist() -> dict[str, list[str]]:
    if not _CONFIG_PATH.is_file():
        return _empty_config()
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        raw = {}
    return _normalize_config(raw)


def _ensure_db() -> None:
    with _connect() as con:
        _ensure_schema(con)
        for chat in DEFAULT_BLOCKED_CHATS:
            _upsert_item(con, "chats", chat)

        migrated = con.execute(
            "SELECT value FROM blacklist_meta WHERE key = 'legacy_json_migrated_at'"
        ).fetchone()
        if migrated is None:
            legacy = _load_legacy_json_blacklist()
            for kind_key in sorted(_VALID_KIND_KEYS):
                for value in legacy.get(kind_key, []):
                    _upsert_item(con, kind_key, value)
            con.execute(
                """
                INSERT OR REPLACE INTO blacklist_meta(key, value)
                VALUES ('legacy_json_migrated_at', ?)
                """,
                (str(time.time()),),
            )
        con.commit()


def _db_cache_key() -> tuple[int, int] | None:
    try:
        st = _DB_PATH.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _read_db_config() -> dict[str, list[str]]:
    raw: dict[str, list[str]] = {key: [] for key in _VALID_KIND_KEYS}
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT kind, value FROM blacklist_items ORDER BY kind, value"
        ).fetchall()
    for kind, value in rows:
        if kind in _VALID_KIND_KEYS:
            raw.setdefault(kind, []).append(value)
    return _normalize_config(raw)


def _write_db_config(cfg: dict[str, list[str]]) -> dict[str, list[str]]:
    data = _normalize_config(cfg)
    with _connect() as con:
        _ensure_schema(con)
        con.execute("DELETE FROM blacklist_items")
        for kind_key in sorted(_VALID_KIND_KEYS):
            for value in data.get(kind_key, []):
                _upsert_item(con, kind_key, value)
        con.execute(
            """
            INSERT OR REPLACE INTO blacklist_meta(key, value)
            VALUES ('saved_at', ?)
            """,
            (str(time.time()),),
        )
        con.commit()
    return data


def load_blacklist() -> dict[str, list[str]]:
    global _CACHE
    try:
        _ensure_db()
        cache_key = _db_cache_key()
        if _CACHE is not None and _CACHE[0] == cache_key:
            return _CACHE[1]
        cfg = _read_db_config()
        _CACHE = (cache_key, cfg)
        return cfg
    except sqlite3.Error:
        try:
            json_mtime = _CONFIG_PATH.stat().st_mtime if _CONFIG_PATH.is_file() else None
        except OSError:
            json_mtime = None
        cache_key = ("json-fallback", json_mtime)
        if _CACHE is not None and _CACHE[0] == cache_key:
            return _CACHE[1]
        cfg = _load_legacy_json_blacklist()
        _CACHE = (cache_key, cfg)
        return cfg


def save_blacklist(cfg: dict[str, list[str]]) -> None:
    global _CACHE
    data = _write_db_config(cfg)
    _CACHE = (_db_cache_key(), data)


def migrate_legacy_json_to_db() -> dict[str, list[str]]:
    current = load_blacklist()
    legacy = _load_legacy_json_blacklist()
    merged = _normalize_config({
        key: [*current.get(key, []), *legacy.get(key, [])]
        for key in _VALID_KIND_KEYS
    })
    save_blacklist(merged)
    return merged


def add_blacklist_item(kind: str, value: str) -> bool:
    cfg = load_blacklist()
    key = _kind_key(kind)
    existing = list(cfg.get(key) or [])
    value_id = _item_identity(key, value)
    if not value_id or value_id in {_item_identity(key, x) for x in existing}:
        return False
    existing.append(value)
    cfg[key] = sorted(set(existing))
    save_blacklist(cfg)
    return True


def remove_blacklist_item(kind: str, value: str) -> bool:
    cfg = load_blacklist()
    key = _kind_key(kind)
    value_id = _item_identity(key, value)
    if key == "chats" and value_id in {
        normalize_blacklist_text(x) for x in DEFAULT_BLOCKED_CHATS
    }:
        return False
    existing = list(cfg.get(key) or [])
    filtered = [x for x in existing if _item_identity(key, x) != value_id]
    if len(filtered) == len(existing):
        return False
    cfg[key] = filtered
    save_blacklist(cfg)
    return True


def _legacy_ignored_paths() -> set[str]:
    global _LEGACY_CACHE
    try:
        mtime = _LEGACY_IGNORED_PATH.stat().st_mtime if _LEGACY_IGNORED_PATH.is_file() else None
    except OSError:
        mtime = None
    if _LEGACY_CACHE is not None and _LEGACY_CACHE[0] == mtime:
        return _LEGACY_CACHE[1]
    paths: set[str] = set()
    if mtime is not None:
        try:
            raw = json.loads(_LEGACY_IGNORED_PATH.read_text(encoding="utf-8"))
            paths = {_clean_path(p) for p in _coerce_list(raw.get("paths"))}
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            paths = set()
    _LEGACY_CACHE = (mtime, paths)
    return paths


def is_chat_blocked(name: Any) -> bool:
    norm = normalize_blacklist_text(name)
    if not norm:
        return False
    cfg = load_blacklist()
    blocked = {
        normalize_blacklist_text(x)
        for x in [*cfg.get("chats", []), *cfg.get("people", [])]
    }
    return norm in blocked


def is_person_blocked(name: Any) -> bool:
    norm = normalize_blacklist_text(name)
    if not norm:
        return False
    blocked = {normalize_blacklist_text(x) for x in load_blacklist().get("people", [])}
    return norm in blocked


def is_path_blocked(path: Any, *, include_legacy_ignore: bool = True) -> bool:
    rel = _clean_path(path)
    if not rel:
        return False
    cfg = load_blacklist()
    exact = {_clean_path(x) for x in cfg.get("paths", [])}
    if include_legacy_ignore:
        exact |= _legacy_ignored_paths()
    if rel in exact:
        return True
    for prefix in cfg.get("path_prefixes", []):
        prefix = _clean_path(prefix)
        if prefix and (rel == prefix.rstrip("/") or rel.startswith(prefix)):
            return True
    return any(fnmatch.fnmatch(rel, pat) for pat in cfg.get("path_globs", []))


def _phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    idx = text.find(phrase)
    while idx >= 0:
        before = text[idx - 1] if idx > 0 else " "
        after_i = idx + len(phrase)
        after = text[after_i] if after_i < len(text) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        idx = text.find(phrase, idx + 1)
    return False


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]{4,}", text)


def _fuzzy_word_matches(term: str, norm_text: str) -> bool:
    norm_term = normalize_blacklist_text(term)
    if len(norm_term) < 4 or " " in norm_term:
        return False
    for token in _word_tokens(norm_text):
        if token == norm_term:
            return True
        if min(len(token), len(norm_term)) >= 4:
            length_delta = abs(len(token) - len(norm_term))
            if length_delta <= 4 and (token.startswith(norm_term) or norm_term.startswith(token)):
                return True
        if SequenceMatcher(None, token, norm_term).ratio() >= 0.86:
            return True
    return False


def blocked_text_matches(text: Any) -> list[str]:
    norm_text = normalize_blacklist_text(text)
    if not norm_text:
        return []
    cfg = load_blacklist()
    terms = [
        *cfg.get("topics", []),
        *cfg.get("people", []),
        *cfg.get("chats", []),
        *cfg.get("words", []),
    ]
    hits: list[str] = []
    for term in terms:
        norm_term = normalize_blacklist_text(term)
        if len(norm_term) < 3:
            continue
        if _phrase_in_text(norm_term, norm_text):
            hits.append(term)
    for term in cfg.get("fuzzy_words", []):
        if _fuzzy_word_matches(term, norm_text):
            hits.append(term)
    return hits


def is_text_blocked(text: Any) -> bool:
    return bool(blocked_text_matches(text))


def should_exclude_record(
    *,
    path: Any = None,
    chat_name: Any = None,
    person: Any = None,
    text: Any = None,
) -> bool:
    return (
        is_path_blocked(path) if path else False
    ) or (
        is_chat_blocked(chat_name) if chat_name else False
    ) or (
        is_person_blocked(person) if person else False
    ) or (
        is_text_blocked(text) if text else False
    )
