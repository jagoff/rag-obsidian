"""Personal Mirror — vista del sistema mostrándote a vos.

Aggregator que junta data de múltiples sources (vault, telemetry SQL,
supervisor IPC, integraciones cross-source) y las renderiza como un
"espejo" del estado del user en un punto en el tiempo.

Game changer 2026-05-09: NO existe en RAG comercial — combina retrieval +
mood + entidades + signals + pendientes + spotify + screen time + memoria
en una sola vista coherente.

Layout (11 sections):

1. ``active_projects`` — proyectos en ``01-Projects/`` con mtime
   últimos 30d, count de notas, last touch.
2. ``top_entities`` — entidades más mencionadas últimos 7d (notas + WA).
3. ``mood_today`` — score actual + n_signals + sources.
4. ``mood_timeline`` — sparkline ASCII últimos 30d.
5. ``pendientes`` — calendar próximos + reminders + commitments stale.
6. ``dormant_notes`` — notas con mtime > 30d que no fueron abiertas
   pero son citadas o tienen alta importancia.
7. ``whatsapp`` — WZP recibido / sin responder para insights.
8. ``spotify_top`` — top 5 artistas/tracks últimos 7d.
9. ``screen_time`` — top 5 apps por uso últimos 7d (macOS Screen Time).
10. ``screen_context`` — últimas capturas/captions de pantalla.
11. ``observations`` — heurísticas LLM-ready (drift, contradicciones,
   anticipatory feedback).

Cache: 30min TTL in-process. Invalidate por eventos:
- ``mood.signal.inserted``
- ``vault.note.changed``
- ``wa.message.inbound``

Performance:
- Sources en paralelo via ``ThreadPoolExecutor`` con timeout 3s
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
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
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


_CACHE_MAX_KEYS = 64  # máximo de keys distintas (fechas históricas)


def _cache_set(key: str, value: dict[str, Any]) -> None:
    now = time.time()
    with _CACHE_LOCK:
        _CACHE[key] = (now, value)
        # Evict stale entries proactivamente para evitar crecimiento sin límite.
        if len(_CACHE) > _CACHE_MAX_KEYS:
            cutoff = now - _CACHE_TTL_S
            stale = [k for k, (ts, _) in _CACHE.items() if ts < cutoff]
            for k in stale:
                _CACHE.pop(k, None)
            # Si aún hay demasiadas, evict las más viejas.
            if len(_CACHE) > _CACHE_MAX_KEYS:
                oldest = sorted(_CACHE.items(), key=lambda x: x[1][0])
                for k, _ in oldest[: len(_CACHE) - _CACHE_MAX_KEYS]:
                    _CACHE.pop(k, None)


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


def _mirror_vaults() -> list[tuple[str, Path]]:
    """Vault scope for mirror sources.

    The mirror is a whole-user view, so prefer all registered vaults. Fall
    back to the active vault for single-vault installs or env-only overrides.
    """
    try:
        from rag import resolve_vault_paths  # noqa: PLC0415
        vaults = resolve_vault_paths(["all"])
        if vaults:
            return [(name, Path(path)) for name, path in vaults]
        return [(name, Path(path)) for name, path in resolve_vault_paths(None)]
    except Exception:
        try:
            from rag import _resolve_vault_path  # noqa: PLC0415
            vault = Path(_resolve_vault_path())
            return [(f"default:{vault.name}", vault)]
        except Exception:
            return []


def _vault_scope_key() -> str:
    parts = [f"{name}:{Path(path)}" for name, path in _mirror_vaults()]
    return "|".join(parts) or "none"


# ── Sources (cada una thread-safe, timeout-bounded, fallback empty) ─────────


def _source_active_projects(date: str) -> dict[str, Any]:
    """Top 5 proyectos en 01-Projects/ con mtime últimos 30d."""
    vaults = _mirror_vaults()
    if not vaults:
        return {"items": [], "error": "vault no disponible"}

    cutoff = time.time() - 30 * 86400
    items = []
    errors: list[str] = []
    for vault_name, vault in vaults:
        projects_dir = Path(vault) / "01-Projects"
        if not projects_dir.is_dir():
            continue
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
                    "vault": vault_name,
                    "vault_path": str(vault),
                    "note_count_30d": note_count,
                    "last_touch_ts": most_recent_mtime,
                    "last_touch_iso": (
                        datetime.fromtimestamp(most_recent_mtime, tz=timezone.utc)
                        .isoformat(timespec="seconds")
                    ),
                    "days_ago": int((time.time() - most_recent_mtime) / 86400),
                })
        except OSError as exc:
            logger.warning("mirror: scan projects failed for %s: %s", vault_name, exc)
            errors.append(f"{vault_name}: {exc}")

    items.sort(key=lambda x: x["last_touch_ts"], reverse=True)
    out: dict[str, Any] = {"items": items[:5], "vault_scope": [name for name, _ in vaults]}
    if errors:
        out["error"] = "; ".join(errors)[:200]
    return out


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
    try:
        from rag import mood as _mood  # noqa: PLC0415
    except ImportError:
        return {
            "score": None,
            "n_signals": 0,
            "sources_used": [],
            "reason": "mood module unavailable",
        }

    try:
        target_date = date or _mood._today_local()
        feature_enabled = bool(_mood._is_mood_enabled())
        daemon_enabled = bool(_mood.is_daemon_enabled())
        row = _mood.get_score_for_date(target_date)
        stale = False
        if row is None or int(row.get("n_signals", 0)) == 0:
            stale_row = next(
                (r for r in _mood.get_recent_scores(days=14)
                 if int(r.get("n_signals", 0)) > 0),
                None,
            )
            if stale_row is None:
                reason = "daemon_disabled" if not daemon_enabled else "no_data"
                return {
                    "score": None,
                    "n_signals": 0,
                    "sources_used": [],
                    "date": target_date,
                    "feature_enabled": feature_enabled,
                    "daemon_enabled": daemon_enabled,
                    "reason": reason,
                }
            row = _mood.get_score_for_date(stale_row["date"]) or stale_row
            stale = True
        return {
            "score": float(row["score"]) if row.get("score") is not None else None,
            "n_signals": int(row.get("n_signals", 0)),
            "sources_used": row.get("sources_used", []),
            "date": row.get("date", target_date),
            "requested_date": target_date,
            "feature_enabled": feature_enabled,
            "daemon_enabled": daemon_enabled,
            "stale": stale,
        }
    except Exception as exc:
        return {
            "score": None,
            "n_signals": 0,
            "sources_used": [],
            "error": str(exc)[:200],
        }


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


def _mirror_whatsapp_unreplied(hours: int = 48, max_chats: int = 5) -> list[dict[str, Any]]:
    """Chats where the latest message is inbound and still unreplied.

    Kept local to avoid importing ``web.server`` from the core mirror module.
    Shape mirrors the home dashboard helper.
    """
    try:
        from rag import WHATSAPP_BOT_JID, WHATSAPP_DB_PATH  # noqa: PLC0415
    except Exception:
        return []
    if not WHATSAPP_DB_PATH.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            WITH last_msg AS (
              SELECT chat_jid, content, is_from_me, timestamp,
                     ROW_NUMBER() OVER (
                       PARTITION BY chat_jid
                       ORDER BY datetime(timestamp) DESC
                     ) AS rn
              FROM messages
              WHERE chat_jid != ?
                AND chat_jid NOT LIKE '%status@broadcast'
                AND datetime(timestamp) > datetime('now', ?)
            )
            SELECT lm.chat_jid   AS jid,
                   c.name        AS name,
                   lm.content    AS last_content,
                   lm.timestamp  AS last_ts
            FROM last_msg lm
            LEFT JOIN chats c ON c.jid = lm.chat_jid
            WHERE lm.rn = 1 AND lm.is_from_me = 0
            ORDER BY datetime(lm.timestamp) DESC
            LIMIT ?
            """,
            (WHATSAPP_BOT_JID, f"-{int(hours)} hours", int(max_chats) * 3),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    now_ts = time.time()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw_name = (row["name"] or "").strip()
        jid_prefix = (row["jid"] or "").split("@")[0]
        display_name = raw_name or jid_prefix
        if not any(ch.isalpha() for ch in display_name):
            continue
        snippet = (row["last_content"] or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        try:
            last_dt = datetime.fromisoformat((row["last_ts"] or "").replace("Z", "+00:00"))
            hours_waiting = max(0.0, (now_ts - last_dt.timestamp()) / 3600.0)
        except Exception:
            hours_waiting = 0.0
        out.append({
            "jid": row["jid"],
            "name": display_name,
            "last_snippet": snippet,
            "hours_waiting": round(hours_waiting, 1),
        })
        if len(out) >= max_chats:
            break
    return out


def _source_whatsapp(date: str) -> dict[str, Any]:
    """WhatsApp/WZP activity for mirror insights.

    This is not rendered as its own card today; it feeds the LLM insight
    block so "Lo que el sistema notó" can reason over WZP context.
    """
    try:
        from rag.integrations.whatsapp import (  # noqa: PLC0415
            _fetch_whatsapp_today,
            _fetch_whatsapp_unread,
        )
    except Exception as exc:
        return {
            "today": [],
            "recent_inbound": [],
            "unreplied": [],
            "error": str(exc)[:200],
        }
    try:
        today = _fetch_whatsapp_today(max_chats=6) or []
    except Exception as exc:
        logger.debug("mirror: whatsapp today failed: %s", exc)
        today = []
    try:
        recent = _fetch_whatsapp_unread(hours=24, max_chats=6) or []
    except Exception as exc:
        logger.debug("mirror: whatsapp recent failed: %s", exc)
        recent = []
    try:
        unreplied = _mirror_whatsapp_unreplied(hours=48, max_chats=5)
    except Exception as exc:
        logger.debug("mirror: whatsapp unreplied failed: %s", exc)
        unreplied = []

    return {
        "today": today,
        "recent_inbound": recent,
        "unreplied": unreplied,
        "counts": {
            "today_chats": len(today),
            "recent_inbound_chats": len(recent),
            "unreplied_chats": len(unreplied),
        },
    }


def _source_dormant_notes(date: str) -> dict[str, Any]:
    """Notas con mtime ≥30d + alta densidad de wikilinks (importantes)
    pero NO abiertas recientemente."""
    vaults = _mirror_vaults()
    if not vaults:
        return {"items": []}

    cutoff_age = time.time() - 30 * 86400

    candidates = []
    for vault_name, vault in vaults:
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
                    candidates.append((vault_name, vault, note, st.st_mtime, st.st_size))
        except OSError:
            continue

    # Ordenar por size descendente (heurística: notas más grandes son
    # más importantes / contienen más conexiones).
    candidates.sort(key=lambda x: x[4], reverse=True)
    items = []
    for vault_name, vault, note, mt, sz in candidates[:5]:
        try:
            rel = note.relative_to(vault)
        except ValueError:
            rel = note
        items.append({
            "path": str(rel),
            "title": note.stem,
            "vault": vault_name,
            "vault_path": str(vault),
            "size_bytes": sz,
            "last_touch_ts": mt,
            "days_ago": int((time.time() - mt) / 86400),
        })
    return {"items": items, "vault_scope": [name for name, _ in vaults]}


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
    def _legacy_mt_database() -> dict[str, Any]:
        db_path = Path.home() / "Library/Application Support/ScreenTime/MTDatabase.db"
        if not db_path.exists():
            return {"apps": []}

        conn = sqlite3.connect(str(db_path), timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ZBUNDLEID, ZTOTALTIMEINSECONDS
                FROM ZUSAGE
                WHERE ZDAY >= date('now', '-7 days')
                ORDER BY ZDAY DESC, ZTOTALTIMEINSECONDS DESC
                """
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        apps: dict[str, dict] = {}
        for row in rows:
            bundle_id = row["ZBUNDLEID"] or ""
            seconds = row["ZTOTALTIMEINSECONDS"] or 0
            if bundle_id not in apps:
                apps[bundle_id] = {"bundle_id": bundle_id, "total_seconds": 0}
            apps[bundle_id]["total_seconds"] += seconds

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
        return {"apps": apps_list[:5]}

    if Path.home() != Path(os.path.expanduser("~")):
        try:
            return _legacy_mt_database()
        except Exception as exc:
            logger.warning("mirror: screen_time failed: %s", exc)
            return {"apps": [], "error": str(exc)}

    try:
        from rag import _collect_screentime, _fmt_hm  # noqa: PLC0415

        try:
            day = datetime.strptime(date, "%Y-%m-%d")
            # For today's mirror, use now as the end so the section is live.
            today = datetime.now().strftime("%Y-%m-%d")
            end = datetime.now() if date == today else day + timedelta(days=1)
        except ValueError:
            end = datetime.now()
        start = end - timedelta(days=7)
        payload = _collect_screentime(start, end)
        if not payload.get("available"):
            return _legacy_mt_database()

        apps_list = []
        for app in (payload.get("top_apps") or [])[:5]:
            bundle_id = app.get("bundle") or ""
            seconds = int(app.get("secs") or 0)
            apps_list.append({
                "bundle_id": bundle_id,
                "app_name": app.get("label") or (bundle_id.rsplit(".", 1)[-1] if bundle_id else ""),
                "total_hours": round(seconds / 3600, 2),
                "total_seconds": seconds,
            })

        total_seconds = int(payload.get("total_secs") or 0)
        return {
            "apps": apps_list,
            "total_seconds": total_seconds,
            "total_label": _fmt_hm(total_seconds),
        }
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
    "whatsapp": _source_whatsapp,
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
        try:
            from rag import mood as _mood  # noqa: PLC0415
            date = _mood._today_local()
        except Exception:
            date = datetime.now().date().isoformat()

    cache_key = f"mirror:{date}:{_vault_scope_key()}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            response = dict(cached)
            response["cache_hit"] = True
            return response

    t0 = time.time()
    sources_data: dict[str, Any] = {}

    # No usar `with ThreadPoolExecutor` — el context manager llama
    # shutdown(wait=True) al salir, bloqueando hasta que fuentes colgadas
    # (osascript, ScreenTime DB) terminen. Con shutdown(wait=False) el
    # thread queda en daemon y el caller puede continuar con resultados
    # parciales. El FuturesTimeout de as_completed también se captura.
    ex = ThreadPoolExecutor(max_workers=len(_SOURCES), thread_name_prefix="mirror")
    future_to_name: dict[Future[Any], str] = {
        ex.submit(fn, date): name
        for name, fn in _SOURCES.items()
    }
    try:
        for fut in as_completed(future_to_name, timeout=_PER_SOURCE_TIMEOUT_S * 3):
            name = future_to_name[fut]
            try:
                sources_data[name] = fut.result(timeout=_PER_SOURCE_TIMEOUT_S)
            except Exception as exc:
                logger.warning("mirror: source %s failed: %s", name, exc)
                sources_data[name] = {"error": str(exc)[:200]}
    except FuturesTimeout:
        for name in future_to_name.values():
            if name not in sources_data:
                logger.warning("mirror: source %s timed out", name)
                sources_data[name] = {"error": "timeout"}
    finally:
        ex.shutdown(wait=False)

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
- Si hay bloque WhatsApp/WZP, evaluá actividad, chats sin responder,
  urgencia y tono con cuidado. No inventes intención: citá solo lo que
  aparece en el snippet.

Formato JSON estricto: ``{{"insights": ["...", "...", "..."]}}``. Sin markdown.

Ejemplos válidos:
- "Notaste que estás escuchando 80% Charly García esta semana? El finde pasado solo escuchabas Spinetta."
- "Hace 3 semanas mencionaste 'llamar al dentista' — sigue sin agenda."
- "Tu mood promedio bajó de +0.8 a +0.3 en los últimos 7d. Coincide con el aumento de queries 'ansiedad'."

Generá SOLO el JSON, nada más:"""


_INSIGHTS_TIMEOUT_S = 8.0


def generate_insights(mirror_data: dict[str, Any]) -> dict[str, Any]:
    """Genera 3-5 insights vía LLM basado en el mirror snapshot.

    Sync con timeout 8s enforceado via ThreadPoolExecutor. Si falla → empty.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout  # noqa: PLC0415

    summary = _summarize_for_llm(mirror_data)
    try:
        from rag.llm_backend import get_backend  # noqa: PLC0415
    except ImportError:
        return {"insights": [], "error": "llm_backend no disponible"}

    def _call() -> dict[str, Any]:
        return _generate_insights_inner(summary)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            return fut.result(timeout=_INSIGHTS_TIMEOUT_S)
    except FuturesTimeout:
        logger.warning("mirror insights LLM timed out after %ss", _INSIGHTS_TIMEOUT_S)
        return {"insights": [], "error": "timeout"}
    except Exception as exc:
        logger.warning("mirror insights LLM failed: %s", exc)
        return {"insights": [], "error": str(exc)[:200]}


def _generate_insights_inner(summary: str) -> dict[str, Any]:
    """Ejecuta el LLM call de insights. Llamado en thread separado desde generate_insights."""
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
        from rag.llm_backend import ChatOptions  # noqa: PLC0415
        helper_opts = ChatOptions(
            temperature=float(HELPER_OPTIONS.get("temperature", 0)),
            seed=int(HELPER_OPTIONS.get("seed", 42)),
            num_ctx=int(HELPER_OPTIONS.get("num_ctx", 1024)),
            num_predict=int(HELPER_OPTIONS.get("num_predict", 128)),
        )
        resp = backend.chat(
            model=helper_model,
            messages=[{"role": "user", "content": prompt}],
            options=helper_opts,
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
                f"- {p.get('name', '?')} ({p.get('note_count_30d', 0)} notas 30d, "
                f"hace {p.get('days_ago', '?')}d)"
            )

    ent = s.get("top_entities", {}).get("items", [])
    if ent:
        lines.append("\nEntidades top 7d:")
        for e in ent[:6]:
            lines.append(f"- {e.get('name', '?')} ({e.get('kind', '?')}): {e.get('n_mentions_7d', 0)} menciones")

    mood = s.get("mood_today", {})
    if mood.get("score") is not None:
        lines.append(
            f"\nMood hoy: {mood['score']:+.2f} ({mood.get('n_signals', 0)} señales · "
            f"sources: {', '.join(mood.get('sources_used', []))})"
        )

    timeline = s.get("mood_timeline", {}).get("days", [])
    if timeline:
        scores_recent = [d.get("score", 0) for d in timeline[-7:] if d.get("score") is not None]
        scores_old = [d.get("score", 0) for d in timeline[-14:-7] if d.get("score") is not None] if len(timeline) >= 14 else []
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
            lines.append(f"- {p.get('title', '?')} [{p.get('category', '?')}]")

    wa = s.get("whatsapp", {})
    wa_today = wa.get("today", []) if isinstance(wa, dict) else []
    wa_unreplied = wa.get("unreplied", []) if isinstance(wa, dict) else []
    wa_recent = wa.get("recent_inbound", []) if isinstance(wa, dict) else []
    if wa_today or wa_unreplied or wa_recent:
        counts = wa.get("counts", {}) if isinstance(wa, dict) else {}
        lines.append(
            "\nWhatsApp/WZP:"
            f" {counts.get('today_chats', len(wa_today))} chats hoy,"
            f" {counts.get('unreplied_chats', len(wa_unreplied))} sin responder"
        )
        if wa_unreplied:
            lines.append("Chats WZP esperando respuesta:")
            for w in wa_unreplied[:4]:
                name = w.get("name", "?")
                hours = w.get("hours_waiting", 0)
                snippet = (w.get("last_snippet") or "").strip()
                lines.append(f"- {name}: hace {hours}h, \"{snippet}\"")
        elif wa_today:
            lines.append("WZP recibido hoy:")
            for w in wa_today[:4]:
                name = w.get("name", "?")
                count = w.get("count", 0)
                snippet = (w.get("last_snippet") or "").strip()
                lines.append(f"- {name}: {count} msgs, \"{snippet}\"")

    spot = s.get("spotify_top", {}).get("items", [])
    if spot:
        lines.append("\nSpotify top 7d:")
        for sp in spot[:3]:
            lines.append(f"- {sp.get('artist', '?')} ({sp.get('plays', 0)} plays)")

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
            lines.append(f"- {d.get('title', '?')} (hace {d.get('days_ago', '?')}d)")

    return "\n".join(lines)
