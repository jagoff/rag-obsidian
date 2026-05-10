"""Journal mood scorer — extracted from rag/mood.py 2026-05-09.

Two signals from journal notes (modified in last N hours):
  - ``keyword_negative``: regex match against ``_NEG_KEYWORDS_RE``.
    Magnitude scales with unique-keyword count.
  - ``note_sentiment``: LLM-judged sentiment via qwen2.5:3b helper. Cached
    per ``(path, mtime)``. Skips notes already flagged by the keyword
    signal (no double-counting).

Folders scanned: ``00-Inbox``, ``02-Areas/journal``, ``01-Projects/journal``
(see ``_JOURNAL_FOLDERS`` in ``rag/mood.py``). Vault auto-resolves from
``rag.VAULT_PATH`` when the caller doesn't pass one.

Re-exported from ``rag.mood`` so call sites keep working without change.

Constants + shared helpers (``_emit``, ``_silent_log_safe``, ``_now_ts``,
the LLM cache ``_SENTIMENT_LLM_CACHE``, regex ``_NEG_KEYWORDS_RE``,
folder list ``_JOURNAL_FOLDERS``) stay in ``rag/mood.py`` because:
  1. They're used by other scorers (Spotify, WA, queries, calendar).
  2. The cache is module-level state that callers (tests) clear via
     ``rag.mood._SENTIMENT_LLM_CACHE.clear()``.
This module imports them lazy from ``rag.mood`` to keep monkeypatches
propagating.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

__all__ = [
    "_journal_recent_notes",
    "_read_note_excerpt",
    "_journal_keyword_signal",
    "_journal_sentiment_llm",
    "_journal_sentiment_signal",
    "score_journal_recent",
]


def _journal_recent_notes(
    vault: Path, within_h: int, now_ts: float, max_files: int = 30,
) -> list[Path]:
    """Devuelve hasta `max_files` paths .md en `_JOURNAL_FOLDERS` con
    `mtime ≥ now - within_h*3600`, ordenadas por mtime DESC. Si una
    folder no existe, se saltea silenciosamente."""
    from rag.mood import _JOURNAL_FOLDERS

    cutoff = now_ts - (within_h * 3600.0)
    candidates: list[tuple[Path, float]] = []
    for folder_rel in _JOURNAL_FOLDERS:
        folder = vault / folder_rel
        if not folder.exists() or not folder.is_dir():
            continue
        with contextlib.suppress(OSError):
            for p in folder.rglob("*.md"):
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime >= cutoff:
                    candidates.append((p, mtime))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in candidates[:max_files]]


def _read_note_excerpt(path: Path, max_chars: int | None = None) -> str:
    """Lee primeros `max_chars` chars de la nota. Strippea YAML frontmatter
    si está al inicio (entre `---\\n` y `\\n---\\n`). Silent-fail → ""."""
    if max_chars is None:
        from rag.mood import _JOURNAL_MAX_CHARS_FOR_LLM
        max_chars = _JOURNAL_MAX_CHARS_FOR_LLM
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    # Strip frontmatter
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]
    return text[:max_chars]


def _journal_keyword_signal(
    notes: list[Path],
    *,
    out: list[dict[str, Any]],
    persist: bool,
    ts: float,
) -> set[str]:
    """Recorre `notes` y matchea el regex de keywords negativas. Por cada
    nota con ≥1 match emite una señal `keyword_negative`. Devuelve el
    set de paths que matchearon (para que el sentiment scorer los
    saltee — no llamar LLM si ya tenemos signal directa).

    Magnitud por nota: -0.5 base, escala con cantidad de matches únicos
    (cap -1.0 con 4+ keywords distintas)."""
    from rag.mood import _NEG_KEYWORDS_RE, _emit

    matched_paths: set[str] = set()
    for path in notes:
        excerpt = _read_note_excerpt(path)
        if not excerpt:
            continue
        matches = list(_NEG_KEYWORDS_RE.finditer(excerpt))
        if not matches:
            continue
        unique_kw = {m.group(0).lower() for m in matches}
        intensity = min(1.0, 0.5 + 0.15 * (len(unique_kw) - 1))
        value = -intensity
        try:
            rel_path = str(path.relative_to(path.parents[len(path.parents) - 2]))
        except (ValueError, IndexError):
            rel_path = path.name
        evidence = {
            "note_path": rel_path,
            "keywords": sorted(unique_kw)[:5],
            "n_matches": len(matches),
        }
        _emit(out, "journal", "keyword_negative", value, weight=1.0,
              evidence=evidence, persist=persist, ts=ts)
        matched_paths.add(str(path))
    return matched_paths


def _journal_sentiment_llm(text: str, *, model: str = "qwen2.5:3b") -> float | None:
    """LLM call. Pide un float -1..+1 sobre el sentimiento general del
    texto. Devuelve None si el LLM no devuelve algo parseable o si
    timea — el caller debe tratar None como "no señal".

    Cache hit por (path, mtime) lo maneja el caller (`_journal_sentiment_signal`).
    Acá solo hace la llamada raw."""
    from rag.mood import _silent_log_safe

    try:
        from rag import _helper_client  # noqa: PLC0415
    except Exception:
        return None
    prompt = (
        "Devolvé SOLO un número entre -1.0 y 1.0 que represente el "
        "sentimiento general del siguiente texto. -1 = muy negativo, "
        "0 = neutro, +1 = muy positivo. Respondé sólo el número, sin "
        "explicación.\n\nTexto:\n"
        f"{text.strip()}\n\nNúmero:"
    )
    try:
        resp = _helper_client().chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 8},
        )
        raw = (resp.get("message") or {}).get("content", "").strip()
    except Exception as exc:
        _silent_log_safe("mood_journal_sentiment_llm_failed", exc)
        return None

    # Parse: extraer primer float-looking token.
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    # Clamp.
    return max(-1.0, min(1.0, val))


def _journal_sentiment_signal(
    notes: list[Path],
    skip_paths: set[str],
    *,
    out: list[dict[str, Any]],
    persist: bool,
    ts: float,
    use_llm: bool,
) -> None:
    """Para cada nota NO incluida en `skip_paths` (i.e. sin keyword match)
    y con length ≥ MIN_CHARS_FOR_LLM, llama al LLM y emite señal
    `note_sentiment` si el LLM devuelve algo parseable.

    Cache LLM por (path_str, mtime) para evitar re-llamar al mismo file
    si dos invocaciones consecutivas (ej. CLI dry-run + daemon real)
    procesan el mismo archivo.

    Si `use_llm=False`, no llama (escenario test)."""
    from rag.mood import _JOURNAL_MIN_CHARS_FOR_LLM, _SENTIMENT_LLM_CACHE, _emit

    if not use_llm:
        return
    for path in notes:
        if str(path) in skip_paths:
            continue
        excerpt = _read_note_excerpt(path)
        if len(excerpt) < _JOURNAL_MIN_CHARS_FOR_LLM:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cache_key = (str(path), mtime)
        if cache_key in _SENTIMENT_LLM_CACHE:
            value = _SENTIMENT_LLM_CACHE[cache_key]
        else:
            value = _journal_sentiment_llm(excerpt)
            if value is None:
                continue
            _SENTIMENT_LLM_CACHE[cache_key] = value
        # Solo emitimos si el sentimiento es notable (|value| ≥ 0.3).
        # Notas neutras no aportan al agregador.
        if abs(value) < 0.3:
            continue
        try:
            rel_path = str(path.relative_to(path.parents[len(path.parents) - 2]))
        except (ValueError, IndexError):
            rel_path = path.name
        evidence = {
            "note_path": rel_path,
            "chars_analyzed": len(excerpt),
            "model": "qwen2.5:3b",
        }
        _emit(out, "journal", "note_sentiment", value, weight=0.6,
              evidence=evidence, persist=persist, ts=ts)


def score_journal_recent(
    *,
    within_h: int | None = None,
    vault: Path | None = None,
    now: float | None = None,
    persist: bool = True,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """Calcula 2 señales de journal (keyword_negative + note_sentiment)
    sobre las notas modificadas en las últimas `within_h` horas dentro
    de `_JOURNAL_FOLDERS`. Devuelve lista de dicts.

    Si `vault=None`, importa `VAULT_PATH` de rag. Si el vault no existe
    o las folders no existen, devuelve `[]` sin tirar.

    `use_llm=False` desactiva la rama LLM (útil para tests o dry-runs
    que no quieren pegarle a Ollama)."""
    from rag.mood import _JOURNAL_DEFAULT_WINDOW_H, _now_ts, _silent_log_safe

    if within_h is None:
        within_h = _JOURNAL_DEFAULT_WINDOW_H
    out: list[dict[str, Any]] = []
    ts = now if now is not None else _now_ts()

    if vault is None:
        try:
            from rag import VAULT_PATH  # noqa: PLC0415
            vault = VAULT_PATH
        except Exception as exc:
            _silent_log_safe("mood_journal_vault_unresolved", exc)
            return out

    if not vault.exists():
        return out

    notes = _journal_recent_notes(vault, within_h, ts)
    if not notes:
        return out

    skip = _journal_keyword_signal(notes, out=out, persist=persist, ts=ts)
    _journal_sentiment_signal(notes, skip, out=out, persist=persist, ts=ts, use_llm=use_llm)
    return out
