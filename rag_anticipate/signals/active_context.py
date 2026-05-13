"""Active context signal (Peekaboo Fase 2f) — nudge cuando la pantalla
actual menciona un proyecto dormant que el user dejó hace ≥5 días.

Idea: si el observer pasivo capturó algo reciente (≤30min) y la caption
contiene tokens distintivos del slug de un proyecto en `01-Projects/`
con mtime ≥5d, emite un candidate "estás mirando X, ¿retomamos el
proyecto Y que pausaste hace Z días?".

## Heurística

1. **Última observación reciente**: SELECT la row más reciente de
   `rag_screen_observations` con ts ≥ now-30min. Si no hay → return [].
   Si hay pero está vacía (caption empty) → return [].
2. **Dormant projects**: subdirs de `01-Projects/` cuyo mtime más
   reciente del contenido sea ≥5 días. Cap 20 candidates (los más
   recientemente activos primero — queremos resumir cosas que SÍ
   estaban activas, no cosas archivadas hace meses).
3. **Match**: tokeniza el slug del proyecto (split por `-_ `, lowercase).
   Tokeniza la caption igual. Skip tokens en stopwords. Si hay ≥1 token
   distintivo en común (length ≥4 para evitar matches espurios "el", "de") →
   match.
4. **Candidate** por la *primera* coincidencia (más recientemente
   activa). El orchestrator se encarga del dedup por kind+key.

## Score

    score = 0.5 + min(0.4, 0.05 * (days_dormant - 5))
    cap a 0.9. 5 días → 0.5; 13+ días → 0.9.

Threshold default del agent es 0.35, así que prácticamente todos los
candidates pasan ese filtro. La diferenciación viene por dedup/snooze.

## dedup_key

    f"active-context:{project_slug}"

Snooze 24h: si el user ya recibió "retomá X" hoy y mañana sigue
mirando X sin actuar, no hace falta volver a empujar.

## Silent-fail

Cualquier excepción → return []. La signal NO toca el orchestrator.

## Privacy

Skip silently si:
- `RAG_SCREEN_OBSERVE != "1"` (feature off).
- Tabla `rag_screen_observations` no existe (DB vieja).
- Última row tiene caption empty o app_name vacío.

NO hace LLM calls. NO toca el VLM. Solo SQL + filesystem.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]{4,}", re.IGNORECASE)

# Stopwords ES + EN — palabras muy comunes que generan false-positives.
_STOPWORDS: frozenset[str] = frozenset({
    "para", "como", "este", "esta", "esto", "esas", "esos", "porque", "donde",
    "cuando", "tambien", "ademas", "entonces", "siempre", "nunca", "tener",
    "hacer", "estar", "puedo", "tengo", "hace", "días", "días", "ayer",
    "antes", "ahora", "luego", "aquí", "alli", "allí",
    "este", "from", "with", "your", "this", "that", "have", "will",
    "would", "could", "should", "into", "than", "what", "when",
    "where", "their", "there",
})

_ACTIVE_WINDOW_SECS = 30 * 60            # captura reciente <= 30min
_DORMANT_MIN_DAYS = 5                    # proyecto dormant cuando >5d sin tocar
_PROJECTS_CAP = 20                       # cap para evitar walks largos


def _tokenize(text: str) -> set[str]:
    """Tokens ≥4 chars, lowercase, sin stopwords."""
    return {
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    }


def _last_observation(conn: sqlite3.Connection, now_ts: int) -> dict | None:
    """Última row de rag_screen_observations ≤30min. None si no hay o tabla
    no existe."""
    try:
        row = conn.execute(
            "SELECT ts, app_name, window_title, caption "
            "FROM rag_screen_observations "
            "WHERE ts >= ? ORDER BY ts DESC LIMIT 1",
            (now_ts - _ACTIVE_WINDOW_SECS,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    ts, app, title, caption = row
    if not (caption or "").strip():
        return None
    return {
        "ts": int(ts),
        "app_name": (app or "").strip(),
        "window_title": (title or "").strip(),
        "caption": caption.strip(),
    }


def _dormant_projects(vault_path: Path, now_ts: float, min_days: int = _DORMANT_MIN_DAYS) -> list[tuple[Path, int]]:
    """Subdirs de `01-Projects/` con mtime más reciente >= min_days días.

    Retorna [(path, days_dormant), ...] ordenado desc por recencia
    (más recientemente activo primero). Cap PROJECTS_CAP.
    """
    projects_dir = vault_path / "01-Projects"
    if not projects_dir.is_dir():
        return []
    cutoff = now_ts - min_days * 86400
    results: list[tuple[Path, int, float]] = []
    try:
        for sub in projects_dir.iterdir():
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            # mtime más reciente del contenido
            latest = sub.stat().st_mtime
            try:
                for p in sub.rglob("*.md"):
                    try:
                        m = p.stat().st_mtime
                        if m > latest:
                            latest = m
                    except OSError:
                        continue
            except OSError:
                continue
            if latest >= cutoff:
                continue  # demasiado fresco — no es dormant
            days = int((now_ts - latest) / 86400)
            results.append((sub, days, latest))
    except OSError:
        return []
    # Más recientemente activo (mtime mayor) primero.
    results.sort(key=lambda r: r[2], reverse=True)
    return [(p, d) for (p, d, _m) in results[:_PROJECTS_CAP]]


@register_signal(name="active_context", snooze_hours=24)
def active_context_signal(now: datetime) -> list:
    """Ver docstring del módulo."""
    if os.environ.get("RAG_SCREEN_OBSERVE", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return []

    try:
        from rag import AnticipatoryCandidate, DB_PATH, VAULT_PATH  # noqa: PLC0415

        db = DB_PATH / "telemetry.db"
        if not db.exists():
            return []

        now_ts = int(time.time())
        con = sqlite3.connect(str(db), timeout=5.0)
        try:
            obs = _last_observation(con, now_ts)
        finally:
            con.close()
        if obs is None:
            return []

        caption_tokens = _tokenize(obs["caption"])
        if not caption_tokens:
            return []

        candidates_proj = _dormant_projects(VAULT_PATH, time.time())
        if not candidates_proj:
            return []

        for proj_path, days_dormant in candidates_proj:
            slug_tokens = _tokenize(proj_path.name.replace("-", " ").replace("_", " "))
            overlap = caption_tokens & slug_tokens
            if not overlap:
                continue

            score = min(0.9, 0.5 + 0.05 * (days_dormant - _DORMANT_MIN_DAYS))
            dedup_key = f"active-context:{proj_path.name.lower()}"
            preview_match = sorted(overlap)[0]  # determinístico
            app_label = obs["app_name"] or "tu pantalla"
            message = (
                f"🖥️ Estás mirando algo en {app_label} sobre `{preview_match}`. "
                f"Tenías el proyecto [[{proj_path.name}]] pausado hace "
                f"{days_dormant}d — ¿retomamos?"
            )
            reason = (
                f"caption_match={preview_match!r} project={proj_path.name!r} "
                f"days_dormant={days_dormant} app={obs['app_name']!r}"
            )
            return [AnticipatoryCandidate(
                kind="anticipate-active-context",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=reason,
                source_note=str(proj_path),
            )]
        return []
    except Exception:
        return []
