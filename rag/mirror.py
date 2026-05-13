"""Personal Mirror — vista del sistema mostrándote a vos.

Aggregator que junta data de múltiples sources (vault, telemetry SQL,
supervisor IPC, integraciones cross-source) y las renderiza como un
"espejo" del estado del user en un punto en el tiempo.

Game changer 2026-05-09: NO existe en RAG comercial — combina retrieval +
mood + entidades + signals + pendientes + spotify + screen time + memoria
en una sola vista coherente.

Layout (9 sections):

1. ``active_projects`` — proyectos en ``01-Projects/`` con mtime
   últimos 30d, count de notas, last touch.
2. ``top_entities`` — entidades más mencionadas últimos 7d (notas + WA).
3. ``mood_today`` — score actual + n_signals + sources.
4. ``mood_timeline`` — sparkline ASCII últimos 30d.
5. ``pendientes`` — calendar próximos + reminders + commitments stale.
6. ``dormant_notes`` — notas con mtime > 30d que no fueron abiertas
   pero son citadas o tienen alta importancia.
7. ``spotify_top`` — top 5 artistas/tracks últimos 7d.
8. ``screen_time`` — top 5 apps por uso últimos 7d (macOS Screen Time).
9. ``observations`` — heurísticas LLM-ready (drift, contradicciones,
   anticipatory feedback).

Cache: 30min TTL in-process. Invalidate por eventos:
- ``mood.signal.inserted``
- ``vault.note.changed``
- ``wa.message.inbound``

Performance:
- 9 sources en paralelo via ``ThreadPoolExecutor`` con timeout 3s
  cada una.
- Source que timea retorna empty + flag ``error``.
- Total wall < 4s sin LLM (insights se calculan separadamente).

LLM insights: NO en este módulo. Ver ``generate_insights()`` que se
llama lazy desde el endpoint ``/api/mirror/insights``.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "assemble_mirror",
    "cache_invalidate",
    "generate_insights",
]


_TELEMETRY_DB = (
    Path(os.environ.get("OBSIDIAN_RAG_DB_PATH")
         or str(Path.home() / ".local/share/obsidian-rag/ragvec"))
    / "telemetry.db"
)
_PER_SOURCE_TIMEOUT_S = 3.0
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 1800  # 30 min
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


# ── Cache helpers ──────────────────────────────────────────────────────────


def cache_invalidate() -> None:
    """Clear todo el cache. Llamado desde event handlers (mood update,
    nota nueva, mensaje WA inbound)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _cache_get(key: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > _CACHE_TTL_S:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


# ── DB helper ──────────────────────────────────────────────────────────────


def _open_telemetry_ro() -> sqlite3.Connection | None:
    if not _TELEMETRY_DB.exists():
        return None
    try:
        uri = f"file:{_TELEMETRY_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.execute("PRAGMA busy_timeout=2000")
        return conn
    except Exception as exc:
        logger.warning("mirror: telemetry conn failed: %s", exc)
        return None


# ── Sources (cada una thread-safe, timeout-bounded, fallback empty) ─────────


def _source_active_projects(date: str) -> dict[str, Any]:
    """Top 5 proyectos en 01-Projects/ con mtime últimos 30d."""
    try:
        from rag import _resolve_vault_path  # noqa: PLC0415
        vault = _resolve_vault_path()
    except Exception:
        return {"items": [], "error": "vault no disponible"}

    projects_dir = Path(vault) / "01-Projects"
    if not projects_dir.is_dir():
        return {"items": []}

    cutoff = time.time() - 30 * 86400
    items = []
    try:
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue
            note_count = 0
            most_recent_mtime = 0.0
            for note in project_dir.rglob("*.md"):
                try:
                    mt = note.stat().st_mtime
                except OSError:
                    continue
                if mt > cutoff:
                    note_count += 1
                if mt > most_recent_mtime:
                    most_recent_mtime = mt
            if note_count == 0:
                continue
            items.append({
                "name": project_dir.name,
                "note_count_30d": note_count,
                "last_touch_ts": most_recent_mtime,
                "last_touch_iso": (
                    datetime.fromtimestamp(most_recent_mtime, tz=timezone.utc)
                    .isoformat(timespec="seconds")
                ),
                "days_ago": int((time.time() - most_recent_mtime) / 86400),
            })
    except OSError as exc:
        logger.warning("mirror: scan projects failed: %s", exc)
        return {"items": [], "error": str(exc)}

    items.sort(key=lambda x: x["last_touch_ts"], reverse=True)
    return {"items": items[:5]}


def _source_top_entities(date: str) -> dict[str, Any]:
    """Top 8 entidades más mencionadas últimos 7d."""
    conn = _open_telemetry_ro()
    if conn is None:
        return {"items": []}

    try:
        cutoff = time.time() - 7 * 86400
        cur = conn.execute(
            """
            SELECT e.canonical_name, e.entity_type, COUNT(*) AS n_mentions,
                   MAX(em.ts) AS last_seen_ts,
                   COUNT(DISTINCT em.source) AS n_sources
            FROM rag_entity_mentions em
            JOIN rag_entities e ON em.entity_id = e.id
            WHERE em.ts > ?
            GROUP BY em.entity_id
            ORDER BY n_mentions DESC
            LIMIT 8
            """,
            (cutoff,),
        )
        items = []
        for row in cur.fetchall():
            name, kind, n, last_seen, n_sources = row
            items.append({
                "name": name,
                "kind": kind,
                "n_mentions_7d": int(n),
                "last_seen_ts": float(last_seen) if last_seen else None,
                "last_seen_iso": (
                    datetime.fromtimestamp(last_seen, tz=timezone.utc)
                    .isoformat(timespec="seconds")
                    if last_seen else None
                ),
                "n_sources": int(n_sources),
            })
        return {"items": items}
    except sqlite3.Error as exc:
        logger.warning("mirror: top_entities sql failed: %s", exc)
        return {"items": [], "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_mood_today(date: str) -> dict[str, Any]:
    """Daily score de hoy via SQL directo (más rápido que IPC al supervisor)."""
    conn = _open_telemetry_ro()
    if conn is None:
        return {"score": None, "n_signals": 0, "sources_used": []}

    try:
        cur = conn.execute(
            "SELECT score, n_signals, sources_used FROM rag_mood_score_daily "
            "WHERE date = ? ORDER BY updated_at DESC LIMIT 1",
            (date,),
        )
        row = cur.fetchone()
        if row is None:
            return {"score": None, "n_signals": 0, "sources_used": []}
        score, n_signals, sources_used_json = row
        import json  # noqa: PLC0415
        try:
            sources = json.loads(sources_used_json) if sources_used_json else []
        except json.JSONDecodeError:
            sources = []
        return {
            "score": float(score) if score is not None else None,
            "n_signals": int(n_signals),
            "sources_used": sources,
            "date": date,
        }
    except sqlite3.Error as exc:
        return {"score": None, "n_signals": 0, "sources_used": [], "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_mood_timeline(date: str) -> dict[str, Any]:
    """Últimos 30d de mood score para sparkline."""
    conn = _open_telemetry_ro()
    if conn is None:
        return {"days": []}
    try:
        cutoff_date = (
            datetime.now(tz=timezone.utc) - timedelta(days=30)
        ).date().isoformat()
        cur = conn.execute(
            "SELECT date, score FROM rag_mood_score_daily "
            "WHERE date >= ? ORDER BY date ASC",
            (cutoff_date,),
        )
        days = [
            {"date": d, "score": float(s) if s is not None else 0.0}
            for d, s in cur.fetchall()
        ]
        return {"days": days, "n": len(days)}
    except sqlite3.Error as exc:
        return {"days": [], "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_pendientes(date: str) -> dict[str, Any]:
    """Pendientes — calendar events próximos + Apple Reminders due hoy.

    El helper ``_pendientes_collect`` requiere argumentos no triviales
    (DB collection, now datetime, lookback days); evitamos importarlo
    para mantener este source liviano. Hacemos query directa al
    integration de Apple Reminders + Calendar.
    """
    items: list[dict[str, Any]] = []

    # Apple Reminders due hoy/mañana.
    try:
        from rag.integrations.reminders import _fetch_reminders  # noqa: PLC0415
        reminders = _fetch_reminders() or []
        now = datetime.now(tz=timezone.utc)
        cutoff_72h = now + timedelta(hours=72)
        for r in reminders[:30]:
            if not isinstance(r, dict):
                continue
            due = r.get("due_date")
            if not due:
                continue
            try:
                due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if due_dt > cutoff_72h:
                continue
            items.append({
                "category": "reminder",
                "title": str(r.get("title", ""))[:120],
                "when": due_dt.isoformat(timespec="minutes"),
            })
    except Exception as exc:
        logger.debug("mirror: reminders skip: %s", exc)

    # Calendar próximas 12h.
    try:
        from rag.integrations.calendar import _fetch_calendar_events  # noqa: PLC0415
        events = _fetch_calendar_events() or []
        now = datetime.now(tz=timezone.utc)
        cutoff_12h = now + timedelta(hours=12)
        for ev in events[:30]:
            if not isinstance(ev, dict):
                continue
            start = ev.get("start") or ev.get("when")
            if not start:
                continue
            try:
                start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if start_dt < now or start_dt > cutoff_12h:
                continue
            items.append({
                "category": "calendar",
                "title": str(ev.get("title") or ev.get("summary", ""))[:120],
                "when": start_dt.isoformat(timespec="minutes"),
            })
    except Exception as exc:
        logger.debug("mirror: calendar skip: %s", exc)

    items.sort(key=lambda x: x.get("when", ""))
    return {"items": items[:10]}


def _source_dormant_notes(date: str) -> dict[str, Any]:
    """Notas con mtime ≥30d + alta densidad de wikilinks (importantes)
    pero NO abiertas recientemente."""
    try:
        from rag import _resolve_vault_path  # noqa: PLC0415
        vault = Path(_resolve_vault_path())
    except Exception:
        return {"items": []}

    cutoff_age = time.time() - 30 * 86400

    candidates = []
    try:
        for folder in ("01-Projects", "02-Areas", "03-Resources"):
            d = vault / folder
            if not d.is_dir():
                continue
            for note in d.rglob("*.md"):
                try:
                    st = note.stat()
                except OSError:
                    continue
                if st.st_mtime > cutoff_age:
                    continue
                if st.st_size > 30000 or st.st_size < 100:
                    continue  # skip too short / too long
                candidates.append((note, st.st_mtime, st.st_size))
    except OSError:
        return {"items": []}

    # Ordenar por size descendente (heurística: notas más grandes son
    # más importantes / contienen más conexiones).
    candidates.sort(key=lambda x: x[2], reverse=True)
    items = []
    for note, mt, sz in candidates[:5]:
        try:
            rel = note.relative_to(vault)
        except ValueError:
            rel = note
        items.append({
            "path": str(rel),
            "title": note.stem,
            "size_bytes": sz,
            "last_touch_ts": mt,
            "days_ago": int((time.time() - mt) / 86400),
        })
    return {"items": items}


def _source_spotify_top(date: str) -> dict[str, Any]:
    """Top 5 artistas últimos 7d."""
    conn = _open_telemetry_ro()
    if conn is None:
        return {"items": []}
    try:
        cutoff = time.time() - 7 * 86400
        cur = conn.execute(
            "SELECT artist, COUNT(*) AS n, COUNT(DISTINCT track_id) AS tracks "
            "FROM rag_spotify_log WHERE last_seen > ? AND state = 'playing' "
            "GROUP BY artist ORDER BY n DESC LIMIT 5",
            (cutoff,),
        )
        items = [
            {"artist": a, "plays": int(n), "distinct_tracks": int(t)}
            for a, n, t in cur.fetchall()
        ]
        return {"items": items}
    except sqlite3.Error as exc:
        return {"items": [], "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_screen_time(date: str) -> dict[str, Any]:
    """Top 5 apps por uso en los últimos 7d (Screen Time de macOS)."""
    db_path = Path.home() / "Library/Application Support/ScreenTime/MTDatabase.db"
    if not db_path.exists():
        return {"apps": []}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query para obtener uso por app en los últimos 7 días
        query = """
        SELECT ZBUNDLEID, ZTOTALTIMEINSECONDS
        FROM ZUSAGE
        WHERE ZDAY >= date('now', '-7 days')
        ORDER BY ZDAY DESC, ZTOTALTIMEINSECONDS DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        # Agregar por bundle ID y convertir a horas
        apps: dict[str, dict] = {}
        for row in rows:
            bundle_id = row["ZBUNDLEID"] or ""
            seconds = row["ZTOTALTIMEINSECONDS"] or 0
            if bundle_id not in apps:
                apps[bundle_id] = {"bundle_id": bundle_id, "total_seconds": 0}
            apps[bundle_id]["total_seconds"] += seconds

        # Convertir a lista y ordenar por uso total
        apps_list = []
        for bundle_id, data in apps.items():
            total_hours = data["total_seconds"] / 3600
            app_name = bundle_id.split(".")[-1] if "." in bundle_id else bundle_id
            apps_list.append({
                "bundle_id": bundle_id,
                "app_name": app_name,
                "total_hours": round(total_hours, 2),
                "total_seconds": data["total_seconds"],
            })

        apps_list.sort(key=lambda x: x["total_hours"], reverse=True)

        return {"apps": apps_list[:5]}  # Top 5 apps
    except Exception as exc:
        logger.warning("mirror: screen_time failed: %s", exc)
        return {"apps": [], "error": str(exc)}


def _source_screen_context(date: str) -> dict[str, Any]:
    """Últimas 3 observaciones de pantalla del Peekaboo observer (Fase 2).

    Fuente: `rag_screen_observations` poblada por `screen_observer_job`
    cada 15min cuando `RAG_SCREEN_OBSERVE=1`. Da contexto **live** de qué
    está mirando el user — complementa screen_time (cuánto tiempo en cada
    app) con contenido (qué pantalla concretamente).

    Retorna `{recent: [{ts, app_name, window_title, caption, age_minutes}, ...],
    count_today, count_7d}`. Empty `recent` si:
    - El feature está OFF (tabla vacía).
    - Última observación >4h atrás (probablemente sleep/AFK — no contexto live).

    Cap recent=3 para mantener el response liviano. El caption se trunca a
    140 chars (suficiente para preview, full body queda en la tabla).
    """
    conn = _open_telemetry_ro()
    if conn is None:
        return {"recent": [], "count_today": 0, "count_7d": 0}
    try:
        # Verificá que la tabla existe — feature recién agregado, DB viejas
        # pueden no tenerla todavía.
        try:
            conn.execute("SELECT 1 FROM rag_screen_observations LIMIT 0")
        except sqlite3.Error:
            return {"recent": [], "count_today": 0, "count_7d": 0}

        now_ts = int(time.time())
        cutoff_live = now_ts - 4 * 3600  # ventana "live": últimas 4h
        cutoff_today = now_ts - 24 * 3600
        cutoff_7d = now_ts - 7 * 86400

        # Fase 3 (2026-05-13): incluir `image_url` por row para que el mirror
        # frontend renderice thumbnail. `recent` = últimas 3 en últimas 4h
        # (live ticker, comportamiento original). `today` = todas las
        # observations con image_path NOT NULL del día calendar local (gallery
        # browsable). Cap today=200 para no inflar el payload.
        cur = conn.execute(
            "SELECT id, ts, app_name, window_title, caption, image_path "
            "FROM rag_screen_observations "
            "WHERE ts >= ? "
            "ORDER BY ts DESC LIMIT 3",
            (cutoff_live,),
        )
        recent: list[dict[str, Any]] = []
        for row in cur.fetchall():
            obs_id, ts, app, title, caption, img_path = row
            cap = (caption or "").strip()
            if len(cap) > 140:
                cap = cap[:137] + "…"
            recent.append({
                "id": int(obs_id),
                "ts": int(ts),
                "app_name": app or "",
                "window_title": title or "",
                "caption": cap,
                "age_minutes": max(0, (now_ts - int(ts)) // 60),
                "image_url": (f"/api/screen-capture/{int(obs_id)}" if img_path else None),
            })

        # Today gallery — todas las obs del día calendar local con imagen.
        import datetime as _dt  # noqa: PLC0415
        today_start = int(_dt.datetime.combine(
            _dt.date.today(), _dt.time.min,
        ).timestamp())
        try:
            today_cur = conn.execute(
                "SELECT id, ts, app_name, window_title, caption, image_path "
                "FROM rag_screen_observations "
                "WHERE ts >= ? AND image_path IS NOT NULL "
                "ORDER BY ts DESC LIMIT 200",
                (today_start,),
            )
            today_gallery: list[dict[str, Any]] = []
            for row in today_cur.fetchall():
                obs_id, ts, app, title, caption, img_path = row
                cap = (caption or "").strip()
                if len(cap) > 140:
                    cap = cap[:137] + "…"
                today_gallery.append({
                    "id": int(obs_id),
                    "ts": int(ts),
                    "app_name": app or "",
                    "window_title": title or "",
                    "caption": cap,
                    "age_minutes": max(0, (now_ts - int(ts)) // 60),
                    "image_url": f"/api/screen-capture/{int(obs_id)}",
                })
        except sqlite3.Error as exc:
            logger.warning("mirror: today gallery query failed: %s", exc)
            today_gallery = []

        try:
            count_today = int(conn.execute(
                "SELECT COUNT(*) FROM rag_screen_observations WHERE ts >= ?",
                (cutoff_today,),
            ).fetchone()[0])
        except sqlite3.Error:
            count_today = 0
        try:
            count_7d = int(conn.execute(
                "SELECT COUNT(*) FROM rag_screen_observations WHERE ts >= ?",
                (cutoff_7d,),
            ).fetchone()[0])
        except sqlite3.Error:
            count_7d = 0

        return {
            "recent": recent,
            "today": today_gallery,
            "count_today": count_today,
            "count_7d": count_7d,
        }
    except Exception as exc:
        logger.warning("mirror: screen_context failed: %s", exc)
        return {"recent": [], "today": [], "count_today": 0, "count_7d": 0, "error": str(exc)[:200]}
    finally:
        conn.close()


def _source_observations(date: str) -> dict[str, Any]:
    """Observaciones rápidas: drift alerts recientes, contradicciones
    abiertas, anticipatory pushes hoy."""
    conn = _open_telemetry_ro()
    if conn is None:
        return {}
    obs: dict[str, Any] = {}
    try:
        # Drift alerts últimos 7d.
        cutoff = time.time() - 7 * 86400
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM rag_eval_runs WHERE updated_at > ?",
                (cutoff,),
            )
            obs["eval_runs_7d"] = int(cur.fetchone()[0])
        except sqlite3.Error:
            obs["eval_runs_7d"] = 0

        # Contradicciones unresolved.
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM rag_contradictions WHERE resolved IS NULL "
                "OR resolved = 0"
            )
            obs["contradictions_open"] = int(cur.fetchone()[0])
        except sqlite3.Error:
            obs["contradictions_open"] = 0

        # Anticipatory pushes hoy.
        try:
            today_iso = date
            cur = conn.execute(
                "SELECT COUNT(*) FROM rag_anticipate_candidates "
                "WHERE date(ts, 'unixepoch', 'localtime') = ? AND sent = 1",
                (today_iso,),
            )
            obs["anticipate_pushes_today"] = int(cur.fetchone()[0])
        except sqlite3.Error:
            obs["anticipate_pushes_today"] = 0

        # Queries totales hoy.
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM rag_queries "
                "WHERE date(ts) = ?",
                (today_iso,),
            )
            obs["queries_today"] = int(cur.fetchone()[0])
        except sqlite3.Error:
            obs["queries_today"] = 0

        return obs
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Aggregator ─────────────────────────────────────────────────────────────


_SOURCES: dict[str, Callable[[str], dict[str, Any]]] = {
    "active_projects": _source_active_projects,
    "top_entities": _source_top_entities,
    "mood_today": _source_mood_today,
    "mood_timeline": _source_mood_timeline,
    "pendientes": _source_pendientes,
    "dormant_notes": _source_dormant_notes,
    "spotify_top": _source_spotify_top,
    "screen_time": _source_screen_time,
    "screen_context": _source_screen_context,
    "observations": _source_observations,
}


def assemble_mirror(
    date: str | None = None,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run las 8 sources en paralelo + retorna dict con todos los blocks.

    Args:
        date: ISO ``YYYY-MM-DD``. None → hoy.
        use_cache: ``True`` (default) usa cache 30min. ``False`` recompute.

    Retorna:
        ``{
            "date": "2026-05-09",
            "computed_at": <unix_ts>,
            "wall_s": <float>,
            "cache_hit": <bool>,
            "sources": {
                "active_projects": {...},
                "top_entities": {...},
                ...
            }
        }``
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).date().isoformat()

    cache_key = f"mirror:{date}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            response = dict(cached)
            response["cache_hit"] = True
            return response

    t0 = time.time()
    sources_data: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=len(_SOURCES), thread_name_prefix="mirror") as ex:
        future_to_name = {
            ex.submit(fn, date): name
            for name, fn in _SOURCES.items()
        }
        for fut in as_completed(future_to_name, timeout=_PER_SOURCE_TIMEOUT_S * 3):
            name = future_to_name[fut]
            try:
                sources_data[name] = fut.result(timeout=_PER_SOURCE_TIMEOUT_S)
            except Exception as exc:
                logger.warning("mirror: source %s failed: %s", name, exc)
                sources_data[name] = {"error": str(exc)[:200]}

    response = {
        "date": date,
        "computed_at": time.time(),
        "wall_s": time.time() - t0,
        "cache_hit": False,
        "sources": sources_data,
    }
    _cache_set(cache_key, response)
    return response


# ── LLM insights ──────────────────────────────────────────────────────────


_INSIGHTS_PROMPT = """Sos un sistema de Personal Mirror que observa el comportamiento del user.

Te paso un snapshot del estado actual:

{summary}

Generá 3-5 insights en español rioplatense (voseo). Cada insight debe ser:
- Específico (mencionar names, numbers, dates).
- Personal (vos pensás como si CONOCIERAS al user).
- Accionable o reflexivo (no obvio).

Formato JSON estricto: ``{{"insights": ["...", "...", "..."]}}``. Sin markdown.

Ejemplos válidos:
- "Notaste que estás escuchando 80% Charly García esta semana? El finde pasado solo escuchabas Spinetta."
- "Hace 3 semanas mencionaste 'llamar al dentista' — sigue sin agenda."
- "Tu mood promedio bajó de +0.8 a +0.3 en los últimos 7d. Coincide con el aumento de queries 'ansiedad'."

Generá SOLO el JSON, nada más:"""


def generate_insights(mirror_data: dict[str, Any]) -> dict[str, Any]:
    """Genera 3-5 insights vía LLM basado en el mirror snapshot.

    Sync con timeout 8s. Si falla → empty.
    """
    summary = _summarize_for_llm(mirror_data)
    try:
        from rag.llm_backend import get_backend  # noqa: PLC0415
    except ImportError:
        return {"insights": [], "error": "llm_backend no disponible"}

    try:
        backend = get_backend()
        prompt = _INSIGHTS_PROMPT.format(summary=summary)
        # qwen2.5:3b helper, deterministic
        from rag import HELPER_OPTIONS, resolve_chat_model  # noqa: PLC0415
        try:
            helper_model = resolve_chat_model("helper")
        except Exception:
            helper_model = "qwen2.5:3b"
        resp = backend.chat(
            model=helper_model,
            messages=[{"role": "user", "content": prompt}],
            options=HELPER_OPTIONS,
            stream=False,
        )
        content = ""
        if hasattr(resp, "message") and resp.message is not None:
            content = resp.message.content or ""
        elif isinstance(resp, dict):
            content = (resp.get("message") or {}).get("content", "")

        # Parse JSON.
        import json  # noqa: PLC0415
        # Try strict JSON parse first.
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"insights": [], "error": "llm respondió non-JSON"}
        insights = parsed.get("insights", [])
        if not isinstance(insights, list):
            insights = []
        return {
            "insights": [str(s)[:500] for s in insights[:5]],
            "model": helper_model,
        }
    except Exception as exc:
        logger.warning("mirror insights LLM failed: %s", exc)
        return {"insights": [], "error": str(exc)[:200]}


def _summarize_for_llm(mirror: dict[str, Any]) -> str:
    """Compacta el mirror data en prosa breve para el prompt LLM."""
    s = mirror.get("sources", {})
    lines = [f"Fecha: {mirror.get('date')}", ""]

    proj = s.get("active_projects", {}).get("items", [])
    if proj:
        lines.append("Proyectos activos:")
        for p in proj[:5]:
            lines.append(
                f"- {p['name']} ({p['note_count_30d']} notas 30d, "
                f"hace {p['days_ago']}d)"
            )

    ent = s.get("top_entities", {}).get("items", [])
    if ent:
        lines.append("\nEntidades top 7d:")
        for e in ent[:6]:
            lines.append(f"- {e['name']} ({e['kind']}): {e['n_mentions_7d']} menciones")

    mood = s.get("mood_today", {})
    if mood.get("score") is not None:
        lines.append(
            f"\nMood hoy: {mood['score']:+.2f} ({mood['n_signals']} señales · "
            f"sources: {', '.join(mood.get('sources_used', []))})"
        )

    timeline = s.get("mood_timeline", {}).get("days", [])
    if timeline:
        scores_recent = [d["score"] for d in timeline[-7:]]
        scores_old = [d["score"] for d in timeline[-14:-7]] if len(timeline) >= 14 else []
        if scores_old and scores_recent:
            avg_recent = sum(scores_recent) / len(scores_recent)
            avg_old = sum(scores_old) / len(scores_old)
            lines.append(
                f"Mood trend 14d: {avg_old:+.2f} → {avg_recent:+.2f} "
                f"(Δ {avg_recent - avg_old:+.2f})"
            )

    pend = s.get("pendientes", {}).get("items", [])
    if pend:
        lines.append("\nPendientes:")
        for p in pend[:5]:
            lines.append(f"- {p['title']} [{p['category']}]")

    spot = s.get("spotify_top", {}).get("items", [])
    if spot:
        lines.append("\nSpotify top 7d:")
        for sp in spot[:3]:
            lines.append(f"- {sp['artist']} ({sp['plays']} plays)")

    obs = s.get("observations", {})
    if obs:
        bits = []
        if obs.get("queries_today"):
            bits.append(f"queries hoy: {obs['queries_today']}")
        if obs.get("anticipate_pushes_today"):
            bits.append(f"anticipate pushes: {obs['anticipate_pushes_today']}")
        if obs.get("contradictions_open"):
            bits.append(f"contradicciones abiertas: {obs['contradictions_open']}")
        if bits:
            lines.append("\nMétricas: " + " · ".join(bits))

    dormant = s.get("dormant_notes", {}).get("items", [])
    if dormant:
        lines.append("\nNotas dormidas (>30d):")
        for d in dormant[:3]:
            lines.append(f"- {d['title']} (hace {d['days_ago']}d)")

    return "\n".join(lines)
