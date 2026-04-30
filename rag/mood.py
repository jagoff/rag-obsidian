"""rag/mood.py — Mood signal scoring (Phase A foundation).

Captura señales de estado anímico desde varias sources, las normaliza a un
valor `-1.0..+1.0` (-1 = bajón, +1 = arriba), y persiste en
`rag_mood_signals`. El agregador `compute_daily_score()` (commit 4) lee
estas señales + escribe el score diario en `rag_mood_score_daily`.

**Scope de este módulo (commit 2)**: scorers de Spotify (3 señales) +
journal local (2 señales). Los scorers de WA outbound + queries +
calendar density vienen en el commit 3. El agregador diario + CLI vienen
en el commit 4. Daemon poller en commit 5. Consumers (today brief,
anticipatory) en commits 6-8.

**Behind flag** `RAG_MOOD_ENABLED=1`. Default off — los writers
exit-early si está apagado, así que el feature está dormant hasta que
el user lo prenda explícitamente.

Signal kinds emitidos:

| source | signal_kind | rango | trigger |
|---|---|---|---|
| `spotify` | `artist_mood_lookup` | -1..+1 | weighted-avg de mood por artista en window |
| `spotify` | `compulsive_repeat` | -1..0 | mismo track ≥ N plays en window |
| `spotify` | `late_night_listening` | -1..0 | actividad ≥ 02:00 con tracks de mood ≤ 0 |
| `journal` | `keyword_negative` | -1..0 | regex curado matchea en notas mtime nuevas |
| `journal` | `note_sentiment` | -1..+1 | LLM (qwen2.5:3b) sobre notas largas sin keyword match |

Las señales se persisten con `weight` para que el agregador pueda hacer
weighted-avg. Pesos default acá; tunables después en `compute_daily_score()`
sin tocar este módulo.

**No-go zones**:

- No verbalizar nunca el score al usuario desde acá. Eso es decisión de
  los consumers (today_correlator + anticipatory). Este módulo solo
  captura y persiste.
- No llamar al LLM si el flag está off (cero costo en feature off).
- No fallar el caller si una sub-source rompe — silent-fail con
  `_silent_log` y devolver lista vacía. Los daemons que consumen este
  módulo no deben crashear porque Spotify esté cerrado o el vault esté
  unmounted.

Aprendido el 2026-04-30 — mood drift como señal blanda en lugar de
pop-up paternalista.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Constants

_RAG_MOOD_FLAG = "RAG_MOOD_ENABLED"

# Spotify scorers
_SPOTIFY_DEFAULT_WINDOW_H = 6
_SPOTIFY_REPEAT_THRESHOLD = 5  # plays únicos del mismo track_id que disparan compulsive_repeat
_SPOTIFY_LATE_NIGHT_HOUR = 2   # ≥ 02:00 local count as late-night

# Journal scorers
_JOURNAL_DEFAULT_WINDOW_H = 24
_JOURNAL_MIN_CHARS_FOR_LLM = 80   # < 80 chars → solo regex, no LLM
_JOURNAL_MAX_CHARS_FOR_LLM = 800  # truncate si la nota es muy larga (cost control)
_JOURNAL_FOLDERS = ("00-Inbox", "02-Areas/journal", "01-Projects/journal")

# WA outbound scorers
_WA_DEFAULT_WINDOW_H = 24
_WA_BASELINE_DAYS = 14
_WA_MIN_MSGS_FOR_SIGNAL = 5  # < 5 mensajes en window = no datos suficientes

# Queries existential scorers
_QUERIES_DEFAULT_WINDOW_H = 24
# Regex curado de queries que indican estado existencial / rumiación.
# Match si la query del usuario al RAG tiene tono retrospectivo /
# auto-evaluativo bajo. Igual que el de journal: rioplatense + neutro.
_QUERIES_EXISTENTIAL_RE = re.compile(
    r"\b(?:"
    r"qu[eé]\s+hice|"
    r"qu[eé]\s+pas[oó]\s+con|"
    r"perd[ií]\s+(?:el\s+)?(?:control|tiempo|rumbo)|"
    r"siempre\s+lo\s+mismo|"
    r"no\s+avanzo|"
    r"estoy\s+estancado|"
    r"estoy\s+perdido|"
    r"no\s+(?:s[eé]|s[eé]\s+qu[eé])\s+(?:hago|hacer)|"
    r"sin\s+rumbo|"
    r"d[oó]nde\s+estoy|"
    r"qu[eé]\s+(?:carajo|mierda|onda)|"
    r"todo\s+(?:mal|para\s+(?:atr[aá]s|el\s+orto))|"
    r"para\s+qu[eé]\s+(?:sigo|hago)|"
    r"me\s+siento\s+(?:mal|de\s+culo|cagado|para\s+atr[aá]s)|"
    r"no\s+puedo\s+(?:m[aá]s|seguir)"
    r")\b",
    re.IGNORECASE,
)

# Calendar density scorer
_CALENDAR_DENSITY_THRESHOLD = 6   # ≥ N eventos hoy → signal
_CALENDAR_BACKTOBACK_GAP_MIN = 15  # gap < N min entre eventos = back-to-back

# Regex del scorer keyword_negative — palabras/expresiones argentinas y
# neutras que indican estado bajo. NO matcheamos cosas tipo "estoy bien"
# o "todo ok" porque generan ruido (false-positive en negaciones tipo
# "no estoy bien" requiere parser real, fuera de scope acá).
_NEG_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"baj[oó]n|"
    r"bajoneado|bajoneada|"
    r"ansiedad|"
    r"angustia|angustiado|angustiada|"
    r"deprimid[oa]|"
    r"no\s+doy\s+m[aá]s|"
    r"no\s+puedo\s+m[aá]s|"
    r"agotad[oa]|"
    r"cagado|"
    r"al\s+horno|"
    r"hart[oa]|"
    r"sin\s+ganas|"
    r"todo\s+mal|"
    r"para\s+el\s+orto|"
    r"al\s+cuete|"
    r"vac[ií]o|"
    r"solo|sola|"
    r"triste|"
    r"miedo|"
    r"abrumad[oa]"
    r")\b",
    re.IGNORECASE,
)

# Cache de la lookup table de artistas — se carga una vez por proceso.
_ARTIST_MOOD_CACHE: dict[str, float] | None = None
_ARTIST_MOOD_PATH = Path(__file__).parent / "data" / "artist_mood.json"

# Cache LLM sentiment por (path, mtime). Evita re-llamar al LLM por la
# misma nota sin cambios.
_SENTIMENT_LLM_CACHE: dict[tuple[str, float], float] = {}


# ──────────────────────────────────────────────────────────────────────────
# Helpers compartidos


def _is_mood_enabled() -> bool:
    """True si `RAG_MOOD_ENABLED` está prendido. Default False."""
    val = os.environ.get(_RAG_MOOD_FLAG, "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def _now_ts() -> float:
    return time.time()


def _today_local(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts if ts is not None else _now_ts()))


def _silent_log_safe(event: str, exc: BaseException) -> None:
    """Best-effort silent_log. Si rag.__init__ no está disponible (tests
    importando mood.py standalone) cae a noop."""
    try:
        from rag import _silent_log  # noqa: PLC0415
        _silent_log(event, exc)
    except Exception:  # pragma: no cover
        pass


def _persist_signal(
    source: str,
    signal_kind: str,
    value: float,
    weight: float = 1.0,
    evidence: dict[str, Any] | None = None,
    ts: float | None = None,
) -> None:
    """INSERT en `rag_mood_signals`. Idempotencia NO garantizada — el
    caller tiene que evitar emitir señales duplicadas (típicamente: si
    el daemon corre cada 30 min, cada scorer debería disparar 1 señal
    por window, no por track).

    Silent-fail si la DB no está accesible o el flag está off."""
    if not _is_mood_enabled():
        return
    ts = ts if ts is not None else _now_ts()
    date = _today_local(ts)
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT INTO rag_mood_signals "
                "(ts, date, source, signal_kind, value, weight, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts, date, source, signal_kind,
                    float(value), float(weight),
                    json.dumps(evidence or {}, ensure_ascii=False),
                ),
            )
    except Exception as exc:
        _silent_log_safe(f"mood_persist_signal_failed:{source}:{signal_kind}", exc)


def _emit(
    out: list[dict[str, Any]],
    source: str,
    signal_kind: str,
    value: float,
    weight: float,
    evidence: dict[str, Any],
    *,
    persist: bool,
    ts: float | None = None,
) -> None:
    """Append signal a `out` y opcionalmente persist en DB."""
    out.append({
        "source": source,
        "signal_kind": signal_kind,
        "value": value,
        "weight": weight,
        "evidence": evidence,
    })
    if persist:
        _persist_signal(source, signal_kind, value, weight, evidence, ts=ts)


# ──────────────────────────────────────────────────────────────────────────
# Artist mood lookup table


def _load_artist_mood_table() -> dict[str, float]:
    """Carga `rag/data/artist_mood.json` (cached). Devuelve dict
    `{artist_lowercase: float}`. Si el JSON no existe o está mal formado,
    devuelve dict vacío y silent-loguea — el feature degrada a "ningún
    artista contribuye" en vez de crashear."""
    global _ARTIST_MOOD_CACHE
    if _ARTIST_MOOD_CACHE is not None:
        return _ARTIST_MOOD_CACHE
    try:
        with _ARTIST_MOOD_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        table = {
            str(k).strip().lower(): float(v)
            for k, v in (data.get("table") or {}).items()
        }
        _ARTIST_MOOD_CACHE = table
        return table
    except Exception as exc:
        _silent_log_safe("mood_artist_table_load_failed", exc)
        _ARTIST_MOOD_CACHE = {}
        return _ARTIST_MOOD_CACHE


def _lookup_artist_mood(artist: str) -> float | None:
    """`None` si el artista no está en la tabla (no contribuye al
    weighted-avg). Float entre -1 y +1 si está."""
    if not artist:
        return None
    return _load_artist_mood_table().get(artist.strip().lower())


# ──────────────────────────────────────────────────────────────────────────
# Spotify scorers


def _spotify_fetch_window(window_h: int, now_ts: float) -> list[tuple[str, str, float, float]]:
    """Lee `rag_spotify_log` filas con `last_seen ≥ now - window_h*3600`.
    Devuelve `[(track_id, artist, first_seen, last_seen), ...]` ordenado
    por first_seen ascendente. Silent-fail → lista vacía."""
    cutoff = now_ts - (window_h * 3600.0)
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT track_id, artist, first_seen, last_seen "
                "FROM rag_spotify_log WHERE last_seen >= ? "
                "ORDER BY first_seen ASC",
                (cutoff,),
            ).fetchall()
        return [(r[0], r[1] or "", float(r[2]), float(r[3])) for r in rows]
    except Exception as exc:
        _silent_log_safe("mood_spotify_fetch_failed", exc)
        return []


def _spotify_artist_signal(
    rows: list[tuple[str, str, float, float]],
    *,
    out: list[dict[str, Any]],
    persist: bool,
    ts: float,
) -> None:
    """Weighted-avg del mood por artista. Cada track contribuye
    proporcionalmente a su `duration_played_s = last_seen - first_seen`
    (más tiempo escuchado = más peso). Artistas no listados en el JSON
    no contribuyen.

    Si después de filtrar no queda ningún artista mappeado, no emite
    señal (no contaminar el agregador con "0 = neutro" cuando en
    realidad es "no data")."""
    weighted_sum = 0.0
    weight_total = 0.0
    matched: list[tuple[str, float, float]] = []  # (artist, mood, duration_s)
    for _track_id, artist, first_seen, last_seen in rows:
        mood = _lookup_artist_mood(artist)
        if mood is None:
            continue
        duration = max(60.0, last_seen - first_seen)  # mínimo 60s para evitar peso 0
        weighted_sum += mood * duration
        weight_total += duration
        matched.append((artist, mood, duration))

    if weight_total == 0.0:
        return

    avg_mood = weighted_sum / weight_total
    # Top 3 artistas por duración para evidence.
    top3 = sorted(matched, key=lambda x: x[2], reverse=True)[:3]
    evidence = {
        "n_tracks_total": len(rows),
        "n_tracks_matched": len(matched),
        "top_artists": [{"artist": a, "mood": m, "duration_s": int(d)} for a, m, d in top3],
    }
    _emit(out, "spotify", "artist_mood_lookup", avg_mood, weight=1.0,
          evidence=evidence, persist=persist, ts=ts)


def _spotify_repeat_signal(
    rows: list[tuple[str, str, float, float]],
    *,
    out: list[dict[str, Any]],
    persist: bool,
    ts: float,
    threshold: int = _SPOTIFY_REPEAT_THRESHOLD,
) -> None:
    """Si hay un track con ≥ N plays en la window, señal negativa.
    Compulsive-repeat = repetición obsesiva, indicador de rumiación.
    Magnitud escala con el ratio plays / threshold (cap a -1)."""
    if not rows:
        return
    counts = Counter(track_id for track_id, _, _, _ in rows)
    if not counts:
        return
    top_track_id, top_count = counts.most_common(1)[0]
    if top_count < threshold:
        return

    # Ratio: 5 plays = -0.5, 10 plays = -1.0, capped.
    intensity = min(1.0, top_count / (2.0 * threshold))
    value = -intensity

    # Evidence: artist + name del track más repetido (el primer match).
    artist_for_evidence = ""
    for tid, art, _, _ in rows:
        if tid == top_track_id:
            artist_for_evidence = art
            break

    evidence = {
        "track_id": top_track_id,
        "artist": artist_for_evidence,
        "plays": top_count,
        "threshold": threshold,
    }
    _emit(out, "spotify", "compulsive_repeat", value, weight=0.8,
          evidence=evidence, persist=persist, ts=ts)


def _spotify_late_night_signal(
    rows: list[tuple[str, str, float, float]],
    *,
    out: list[dict[str, Any]],
    persist: bool,
    ts: float,
    hour_threshold: int = _SPOTIFY_LATE_NIGHT_HOUR,
) -> None:
    """Tracks con `first_seen.hour ≥ 02:00` y mood ≤ 0 → señal negativa.
    Escuchar música a la madrugada Y de mood bajo es un signal más fuerte
    que cualquiera por separado. Si hay tracks late-night pero todos de
    mood ≥ 0 (ej. fiesta), no señal. Si hay tracks late-night sin mood
    mappeado, no señal (no asumir)."""
    bad_count = 0
    bad_evidence: list[dict[str, Any]] = []
    for _tid, artist, first_seen, _last_seen in rows:
        hour = time.localtime(first_seen).tm_hour
        # 02:00 a 05:59 — fines de fiesta + insomnio territory
        if not (hour_threshold <= hour < 6):
            continue
        mood = _lookup_artist_mood(artist)
        if mood is None or mood > 0:
            continue
        bad_count += 1
        if len(bad_evidence) < 3:
            bad_evidence.append({"artist": artist, "hour": hour, "mood": mood})

    if bad_count == 0:
        return

    # Magnitud: 1-2 tracks late-night-sad → -0.3, 3-5 → -0.6, 6+ → -1.0
    intensity = min(1.0, bad_count / 6.0)
    value = -max(0.3, intensity)
    evidence = {
        "n_late_night_sad_tracks": bad_count,
        "examples": bad_evidence,
        "hour_threshold": hour_threshold,
    }
    _emit(out, "spotify", "late_night_listening", value, weight=0.9,
          evidence=evidence, persist=persist, ts=ts)


def score_spotify_window(
    *,
    now: float | None = None,
    window_h: int = _SPOTIFY_DEFAULT_WINDOW_H,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Calcula las 3 señales de Spotify en una window de `window_h` horas
    hacia atrás desde `now` (default = ahora). Devuelve lista de dicts
    con shape `{source, signal_kind, value, weight, evidence}` para que
    el caller pueda inspeccionar (tests, dry-run de CLI). Si `persist`
    True, también escribe en `rag_mood_signals`.

    Silent-fail: si Spotify está cerrado, `rag_spotify_log` está vacío,
    o la DB es inaccesible, devuelve `[]` sin tirar."""
    out: list[dict[str, Any]] = []
    ts = now if now is not None else _now_ts()
    rows = _spotify_fetch_window(window_h, ts)
    if not rows:
        return out
    _spotify_artist_signal(rows, out=out, persist=persist, ts=ts)
    _spotify_repeat_signal(rows, out=out, persist=persist, ts=ts)
    _spotify_late_night_signal(rows, out=out, persist=persist, ts=ts)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Journal scorers


def _journal_recent_notes(
    vault: Path, within_h: int, now_ts: float, max_files: int = 30,
) -> list[Path]:
    """Devuelve hasta `max_files` paths .md en `_JOURNAL_FOLDERS` con
    `mtime ≥ now - within_h*3600`, ordenadas por mtime DESC. Si una
    folder no existe, se saltea silenciosamente."""
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


def _read_note_excerpt(path: Path, max_chars: int = _JOURNAL_MAX_CHARS_FOR_LLM) -> str:
    """Lee primeros `max_chars` chars de la nota. Strippea YAML frontmatter
    si está al inicio (entre `---\\n` y `\\n---\\n`). Silent-fail → ""."""
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
    within_h: int = _JOURNAL_DEFAULT_WINDOW_H,
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


# ──────────────────────────────────────────────────────────────────────────
# WhatsApp outbound scorer


def _wa_bridge_db_path() -> Path | None:
    """Path al SQLite del WA bridge. None si la integration no está
    disponible o el path no existe."""
    try:
        from rag.integrations.whatsapp import WHATSAPP_BRIDGE_DB_PATH  # noqa: PLC0415
    except Exception:
        return None
    p = Path(WHATSAPP_BRIDGE_DB_PATH)
    return p if p.exists() else None


def _wa_fetch_outbound_chars(
    db_path: Path, since_ts: float, until_ts: float,
) -> list[int]:
    """Lee la tabla `messages` del bridge y devuelve la lista de
    `len(content)` de mensajes con `is_from_me=1` entre los dos
    timestamps. Silent-fail → []."""
    import sqlite3  # noqa: PLC0415
    since_iso = datetime.fromtimestamp(since_ts).isoformat()
    until_iso = datetime.fromtimestamp(until_ts).isoformat()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except Exception as exc:
        _silent_log_safe("mood_wa_open_db_failed", exc)
        return []
    try:
        rows = conn.execute(
            "SELECT content FROM messages "
            "WHERE is_from_me=1 AND timestamp >= ? AND timestamp < ?",
            (since_iso, until_iso),
        ).fetchall()
    except Exception as exc:
        _silent_log_safe("mood_wa_query_failed", exc)
        return []
    finally:
        with contextlib.suppress(Exception):
            conn.close()
    return [len(r[0]) for r in rows if r[0]]


def score_wa_outbound_window(
    *,
    now: float | None = None,
    window_h: int = _WA_DEFAULT_WINDOW_H,
    baseline_days: int = _WA_BASELINE_DAYS,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Compara avg `chars/msg` outbound del último `window_h` con el
    baseline de los últimos `baseline_days` días. Si el avg actual es
    ≤ 50% del baseline (escribís más corto que normal a contactos
    cercanos), señal -0.4 a -0.7 según el ratio.

    Si hay < `_WA_MIN_MSGS_FOR_SIGNAL` mensajes outbound en la window,
    no hay datos suficientes → no signal. Si el bridge no está
    instalado, no signal.

    Importante: NO se mira el contenido de los mensajes — solo agregados
    estadísticos (avg chars). El feature respeta privacidad."""
    out: list[dict[str, Any]] = []
    ts = now if now is not None else _now_ts()
    db = _wa_bridge_db_path()
    if db is None:
        return out

    window_chars = _wa_fetch_outbound_chars(db, ts - window_h * 3600.0, ts)
    if len(window_chars) < _WA_MIN_MSGS_FOR_SIGNAL:
        return out

    baseline_until = ts - window_h * 3600.0
    baseline_since = ts - baseline_days * 86400.0
    baseline_chars = _wa_fetch_outbound_chars(db, baseline_since, baseline_until)
    if len(baseline_chars) < _WA_MIN_MSGS_FOR_SIGNAL * 3:
        return out

    avg_window = sum(window_chars) / len(window_chars)
    avg_baseline = sum(baseline_chars) / len(baseline_chars)
    if avg_baseline == 0:
        return out

    ratio = avg_window / avg_baseline
    if ratio >= 0.6:
        return out  # no signal — escribís normal

    # ratio 0.5 → -0.4, ratio 0.3 → -0.6, ratio 0.1 → -0.8
    intensity = min(0.8, (0.6 - ratio) * 1.5)
    value = -intensity
    evidence = {
        "msgs_window": len(window_chars),
        "msgs_baseline": len(baseline_chars),
        "avg_chars_window": round(avg_window, 1),
        "avg_chars_baseline": round(avg_baseline, 1),
        "ratio": round(ratio, 2),
    }
    _emit(out, "wa_outbound", "tone_short", value, weight=0.5,
          evidence=evidence, persist=persist, ts=ts)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Queries existential scorer


def _queries_fetch_recent(window_h: int, now_ts: float) -> list[str]:
    """Devuelve la lista de `q` (literal user query) de `rag_queries`
    en la window. Filtramos a `cmd IN ('query', 'chat', 'ask')` para
    quedarnos con queries reales del user (no internal multi-query
    expansions)."""
    cutoff_dt = datetime.fromtimestamp(now_ts) - timedelta(hours=window_h)
    cutoff_iso = cutoff_dt.isoformat()
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT q FROM rag_queries "
                "WHERE ts >= ? AND COALESCE(cmd,'') IN ('', 'query', 'chat', 'ask')",
                (cutoff_iso,),
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        _silent_log_safe("mood_queries_fetch_failed", exc)
        return []


def score_queries_existential(
    *,
    now: float | None = None,
    window_h: int = _QUERIES_DEFAULT_WINDOW_H,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Cuenta queries del user al RAG con patterns existenciales.
    1 match → -0.3, 2 matches → -0.5, 3+ → -0.7.

    No es un signal fuerte por sí solo (gente busca "qué hice ayer"
    sin estar mal); el agregador lo combina con otros."""
    out: list[dict[str, Any]] = []
    ts = now if now is not None else _now_ts()
    queries = _queries_fetch_recent(window_h, ts)
    if not queries:
        return out

    matched: list[str] = []
    for q in queries:
        if _QUERIES_EXISTENTIAL_RE.search(q):
            matched.append(q[:120])

    if not matched:
        return out

    n = len(matched)
    if n >= 3:
        value = -0.7
    elif n == 2:
        value = -0.5
    else:
        value = -0.3
    evidence = {
        "n_matched": n,
        "n_total_queries": len(queries),
        "examples": matched[:3],
    }
    _emit(out, "queries", "existential_pattern", value, weight=0.4,
          evidence=evidence, persist=persist, ts=ts)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Calendar density scorer


def _parse_event_time(s: str) -> tuple[int, int] | None:
    """Parsea string tipo '09:30' o '9:30 AM' → (hour, minute) o None."""
    s = s.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*([AaPp][Mm])?$", s)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    suffix = (m.group(3) or "").lower()
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return hour, minute


def score_calendar_density(
    *,
    now: float | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Si el calendar de hoy tiene ≥ N eventos, señal -0.3. Si además
    hay ≥ 3 events back-to-back (gap < `_CALENDAR_BACKTOBACK_GAP_MIN`),
    señal adicional -0.4. Stress, no bajón — el agregador lo combina
    con los demás. NO mira contenido de los eventos (solo cantidad y
    horarios)."""
    out: list[dict[str, Any]] = []
    ts = now if now is not None else _now_ts()
    try:
        from rag.integrations.calendar import _fetch_calendar_today  # noqa: PLC0415
        events = _fetch_calendar_today(max_events=30)
    except Exception as exc:
        _silent_log_safe("mood_calendar_fetch_failed", exc)
        return out

    if not events or len(events) < _CALENDAR_DENSITY_THRESHOLD:
        return out

    # Density base signal.
    density_value = -min(0.6, 0.3 + 0.05 * (len(events) - _CALENDAR_DENSITY_THRESHOLD))
    _emit(out, "calendar", "density_overload", density_value, weight=0.4,
          evidence={"n_events": len(events)}, persist=persist, ts=ts)

    # Back-to-back: ordenar por start, contar gaps cortos.
    parsed: list[tuple[int, int, int, int]] = []  # (start_h, start_m, end_h, end_m)
    for ev in events:
        start = _parse_event_time(ev.get("start", ""))
        end = _parse_event_time(ev.get("end", ""))
        if start and end:
            parsed.append((*start, *end))
    parsed.sort()

    btb_count = 0
    for i in range(len(parsed) - 1):
        prev_end_min = parsed[i][2] * 60 + parsed[i][3]
        next_start_min = parsed[i + 1][0] * 60 + parsed[i + 1][1]
        gap = next_start_min - prev_end_min
        if 0 <= gap < _CALENDAR_BACKTOBACK_GAP_MIN:
            btb_count += 1

    if btb_count >= 3:
        btb_value = -min(0.6, 0.3 + 0.1 * (btb_count - 3))
        _emit(out, "calendar", "back_to_back_meetings", btb_value, weight=0.5,
              evidence={"n_back_to_back": btb_count, "n_events": len(events)},
              persist=persist, ts=ts)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Daily aggregator + drift detection


def _read_signals_for_date(date: str) -> list[dict[str, Any]]:
    """Lee todas las señales de un día desde `rag_mood_signals`.
    Devuelve `[{source, signal_kind, value, weight, evidence}, ...]`.
    `evidence` ya parseado a dict (era JSON). Silent-fail → []."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, signal_kind, value, weight, evidence "
                "FROM rag_mood_signals WHERE date = ? ORDER BY ts ASC",
                (date,),
            ).fetchall()
    except Exception as exc:
        _silent_log_safe("mood_aggregator_read_failed", exc)
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            evidence = json.loads(r[4]) if r[4] else {}
        except Exception:
            evidence = {}
        out.append({
            "source": r[0],
            "signal_kind": r[1],
            "value": float(r[2]),
            "weight": float(r[3]),
            "evidence": evidence,
        })
    return out


def compute_daily_score(date: str | None = None) -> dict[str, Any]:
    """Lee `rag_mood_signals` para `date` (YYYY-MM-DD, default = hoy local),
    computa weighted-avg, y UPSERT en `rag_mood_score_daily`.

    Algoritmo:
      score = sum(value_i * weight_i) / sum(weight_i)
    si total_weight = 0 → score 0 con n_signals=0 (caso "sin datos
    todavía hoy"). El UPSERT escribe igual para que el consumer pueda
    distinguir "no corrió aún" (no row) de "corrió y no había datos"
    (row con n_signals=0).

    Devuelve dict `{date, score, n_signals, sources_used, top_evidence}`
    con el resultado. Top_evidence = top 3 señales por |value*weight|
    (las que más contribuyen al score), serializado como list de dicts
    `{source, signal_kind, value, weight, evidence}`."""
    if not _is_mood_enabled():
        return {"date": date or _today_local(), "score": 0.0, "n_signals": 0,
                "sources_used": [], "top_evidence": []}

    date = date or _today_local()
    signals = _read_signals_for_date(date)

    if signals:
        total_weight = sum(s["weight"] for s in signals)
        if total_weight > 0:
            score = sum(s["value"] * s["weight"] for s in signals) / total_weight
        else:
            score = 0.0
        sources_used = sorted({s["source"] for s in signals})
        # Top evidence por |contribution| = |value * weight|.
        ranked = sorted(
            signals, key=lambda s: abs(s["value"] * s["weight"]), reverse=True,
        )[:3]
        top_evidence = [
            {
                "source": s["source"],
                "signal_kind": s["signal_kind"],
                "value": s["value"],
                "weight": s["weight"],
                "evidence": s["evidence"],
            }
            for s in ranked
        ]
    else:
        score = 0.0
        sources_used = []
        top_evidence = []

    n_signals = len(signals)
    updated_at = _now_ts()
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT INTO rag_mood_score_daily "
                "(date, score, n_signals, sources_used, top_evidence, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET "
                "  score=excluded.score, n_signals=excluded.n_signals, "
                "  sources_used=excluded.sources_used, "
                "  top_evidence=excluded.top_evidence, "
                "  updated_at=excluded.updated_at",
                (
                    date, float(score), int(n_signals),
                    json.dumps(sources_used, ensure_ascii=False),
                    json.dumps(top_evidence, ensure_ascii=False),
                    updated_at,
                ),
            )
    except Exception as exc:
        _silent_log_safe("mood_aggregator_upsert_failed", exc)

    return {
        "date": date, "score": score, "n_signals": n_signals,
        "sources_used": sources_used, "top_evidence": top_evidence,
    }


def get_score_for_date(date: str) -> dict[str, Any] | None:
    """Lee row de `rag_mood_score_daily` para una fecha. None si no hay
    row (no se computó todavía o el feature está off)."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT date, score, n_signals, sources_used, top_evidence, updated_at "
                "FROM rag_mood_score_daily WHERE date = ?",
                (date,),
            ).fetchone()
    except Exception as exc:
        _silent_log_safe("mood_get_score_failed", exc)
        return None
    if not row:
        return None
    try:
        sources = json.loads(row[3]) if row[3] else []
    except Exception:
        sources = []
    try:
        evidence = json.loads(row[4]) if row[4] else []
    except Exception:
        evidence = []
    return {
        "date": row[0],
        "score": float(row[1]),
        "n_signals": int(row[2]),
        "sources_used": sources,
        "top_evidence": evidence,
        "updated_at": float(row[5]),
    }


def get_recent_scores(days: int = 14) -> list[dict[str, Any]]:
    """Devuelve hasta `days` filas de `rag_mood_score_daily` ordenadas
    por date DESC (más reciente primero). Útil para sparkline en CLI."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT date, score, n_signals FROM rag_mood_score_daily "
                "ORDER BY date DESC LIMIT ?",
                (int(days),),
            ).fetchall()
    except Exception as exc:
        _silent_log_safe("mood_get_recent_failed", exc)
        return []
    return [
        {"date": r[0], "score": float(r[1]), "n_signals": int(r[2])}
        for r in rows
    ]


def recent_drift(
    *,
    days: int = 7,
    threshold: float = -0.4,
    min_consecutive: int = 3,
) -> dict[str, Any]:
    """Detecta racha sostenida de score ≤ `threshold` en los últimos
    `days`. Una racha es ≥ `min_consecutive` días consecutivos
    (terminando hoy o ayer — toleramos un día sin data).

    Devuelve `{drifting: bool, n_consecutive: int, dates: list[str],
    avg_score: float, reason: str|None}`. `drifting=True` solo si la
    racha ≥ min_consecutive Y termina hoy o ayer (no detectamos bajones
    históricos viejos).

    Threshold conservador (-0.4) intencional — evita falsos positivos
    en días con un solo signal moderado."""
    rows = get_recent_scores(days=days + 1)
    if not rows:
        return {"drifting": False, "n_consecutive": 0, "dates": [],
                "avg_score": 0.0, "reason": "no_data"}

    today = _today_local()
    yesterday = _today_local(_now_ts() - 86400)

    # Ordenar ascendente por date para iterar cronológicamente.
    rows = sorted(rows, key=lambda r: r["date"])

    # Buscar la racha más reciente que termine en today o yesterday.
    best_run: list[dict[str, Any]] = []
    current_run: list[dict[str, Any]] = []
    for r in rows:
        if r["score"] <= threshold and r["n_signals"] > 0:
            current_run.append(r)
        else:
            if len(current_run) >= len(best_run):
                best_run = list(current_run)
            current_run = []
    # Capturar racha final si terminó en el último row.
    if len(current_run) >= len(best_run):
        best_run = list(current_run)

    if not best_run:
        return {"drifting": False, "n_consecutive": 0, "dates": [],
                "avg_score": 0.0, "reason": "no_streak"}

    last_date = best_run[-1]["date"]
    if last_date not in (today, yesterday):
        return {"drifting": False, "n_consecutive": len(best_run),
                "dates": [r["date"] for r in best_run],
                "avg_score": sum(r["score"] for r in best_run) / len(best_run),
                "reason": f"stale_streak_ended_{last_date}"}

    n_consec = len(best_run)
    avg = sum(r["score"] for r in best_run) / n_consec
    drifting = n_consec >= min_consecutive
    return {
        "drifting": drifting,
        "n_consecutive": n_consec,
        "dates": [r["date"] for r in best_run],
        "avg_score": avg,
        "reason": None if drifting else f"only_{n_consec}_days_under_threshold",
    }


__all__ = [
    "score_spotify_window",
    "score_journal_recent",
    "score_wa_outbound_window",
    "score_queries_existential",
    "score_calendar_density",
    "compute_daily_score",
    "get_score_for_date",
    "get_recent_scores",
    "recent_drift",
    "_is_mood_enabled",
    "_persist_signal",
    "_load_artist_mood_table",
]
