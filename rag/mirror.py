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
import re
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
_LIVE_SOURCE_NAMES = ("whatsapp",)


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


def _is_live_mirror_date(date: str) -> bool:
    try:
        from rag import mood as _mood  # noqa: PLC0415
        today = _mood._today_local()
    except Exception:
        today = datetime.now().date().isoformat()
    return date == today


def _refresh_live_sources(response: dict[str, Any], date: str) -> dict[str, Any]:
    """Overlay minute-level sources on top of the slower mirror cache."""
    if not _is_live_mirror_date(date):
        return response
    sources = dict(response.get("sources") or {})
    refreshed: list[str] = []
    for name in _LIVE_SOURCE_NAMES:
        fn = _SOURCES.get(name)
        if fn is None:
            continue
        try:
            sources[name] = fn(date)
        except Exception as exc:
            logger.warning("mirror: live source %s failed: %s", name, exc)
            sources[name] = {"error": str(exc)[:200]}
        refreshed.append(name)
    if refreshed:
        response["sources"] = sources
        response["live_refreshed"] = refreshed
        response["live_computed_at"] = time.time()
    return response


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
                rel_project = _blacklist_relpath(project_dir, Path(vault))
                if _blacklisted_record(path=rel_project, text=project_dir.name):
                    continue
                note_count = 0
                most_recent_mtime = 0.0
                for note in project_dir.rglob("*.md"):
                    if _blacklisted_record(path=_blacklist_relpath(note, Path(vault))):
                        continue
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
        return _source_top_entities_fallback()

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
            if _blacklisted_record(
                person=name if str(kind or "").lower() in {"person", "people"} else None,
                text=name,
            ):
                continue
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
        if items:
            return {"items": items}
        return _source_top_entities_fallback(reason="no_entities_indexed")
    except sqlite3.Error as exc:
        logger.warning("mirror: top_entities sql failed: %s", exc)
        fallback = _source_top_entities_fallback(reason="entity_sql_failed")
        fallback["error"] = str(exc)
        return fallback
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_top_entities_fallback(reason: str = "telemetry_unavailable") -> dict[str, Any]:
    """Fallback for the "Quién está en tu cabeza" card when NER has no rows.

    Entity extraction is optional/backfilled, but WhatsApp activity is often
    the clearest live signal for "who is top of mind". Keep the card useful
    instead of rendering an empty block.
    """
    try:
        from rag.integrations.whatsapp import _fetch_whatsapp_today  # noqa: PLC0415

        chats = _fetch_whatsapp_today(max_chats=8) or []
    except Exception as exc:
        return {"items": [], "reason": reason, "error": str(exc)[:200]}

    items: list[dict[str, Any]] = []
    for chat in chats[:8]:
        if not isinstance(chat, dict):
            continue
        name = str(chat.get("name") or "").strip()
        if not name:
            continue
        if _wa_chat_name_excluded(name) or _blacklisted_record(chat_name=name, text=name):
            continue
        count = int(chat.get("count") or 0)
        items.append({
            "name": name,
            "kind": "chat",
            "n_mentions_7d": count,
            "n_sources": 1,
            "meta": f"{count} msgs hoy" if count else "chat activo hoy",
            "fallback": "whatsapp_today",
        })
    return {"items": items, "reason": reason, "fallback": "whatsapp_today"}


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
                return {
                    "score": None,
                    "n_signals": 0,
                    "sources_used": [],
                    "date": target_date,
                    "feature_enabled": True,
                    "daemon_enabled": daemon_enabled,
                    "reason": "no_data",
                }
            row = _mood.get_score_for_date(stale_row["date"]) or stale_row
            stale = True
        return {
            "score": float(row["score"]) if row.get("score") is not None else None,
            "n_signals": int(row.get("n_signals", 0)),
            "sources_used": row.get("sources_used", []),
            "date": row.get("date", target_date),
            "requested_date": target_date,
            "feature_enabled": True,
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
        return {"days": [], "reason": "telemetry_unavailable"}
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
        if days:
            return {"days": days, "n": len(days)}
        return {"days": [], "n": 0, "reason": "no_data"}
    except sqlite3.Error as exc:
        return {"days": [], "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_pendientes(date: str) -> dict[str, Any]:
    """Pendientes — compact, timeout-safe view for Mirror.

    `/api/pendientes` can spend ~8-10s collecting Mail/Gmail/WhatsApp/Apple
    services. Mirror has a 3s per-source budget, so this card uses the cheap
    Reminder/Calendar fallback plus fast vault-loop extraction across the
    default home + work scope. The full evidence endpoint remains richer; the
    Mirror card must never disappear because one service is slow.
    """
    payload = _source_pendientes_light(date)
    try:
        import rag as _rag  # noqa: PLC0415

        try:
            vaults = _rag.resolve_vault_paths(None) or []
        except Exception:
            vaults = []

        loops: list[dict[str, Any]] = []
        for vault_name, vault_path in vaults:
            try:
                found = _rag._pendientes_extract_loops_fast(
                    Path(vault_path), days=14, max_items=12,
                )
            except Exception:
                continue
            for loop in found or []:
                loop["_vault"] = vault_name
                loop["_vault_path"] = str(vault_path)
                loops.append(loop)
        loops.sort(key=lambda x: x.get("age_days", 0), reverse=True)

        for loop in loops[:5]:
            title = _clip(loop.get("loop_text"), 120)
            if not title:
                continue
            source_note = Path(str(loop.get("source_note") or ""))
            source_path = str(source_note)
            try:
                source_path = _blacklist_relpath(source_note, Path(loop.get("_vault_path") or ""))
            except Exception:
                pass
            if _blacklisted_record(path=source_path, text=title):
                continue
            vault = f"[{loop.get('_vault')}] " if loop.get("_vault") else ""
            src = source_note.stem
            age = f"{loop.get('age_days', 0)}d"
            payload.setdefault("items", []).append({
                "category": "vault loop",
                "title": title,
                "meta": f"{vault}{age} · {src}".strip(),
                "when": "",
            })

        counts = payload.setdefault("counts", {})
        counts["loops"] = len(loops)
        if loops:
            services = payload.setdefault("services_consulted", [])
            if "Vault loops" not in services:
                services.append("Vault loops")
        payload["items"] = (payload.get("items") or [])[:10]
        payload["vault_scope"] = [name for name, _ in vaults]
        payload["reason"] = "mirror_fast_collector"
        return payload
    except Exception as exc:
        logger.warning("mirror: pendientes collector failed: %s", exc)
        payload["error"] = str(exc)[:200]
        return payload


def _source_pendientes_light(date: str) -> dict[str, Any]:
    """Cheap fallback used only if the full pendientes collector fails."""
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
            title = str(r.get("title", ""))[:120]
            if _blacklisted_record(text=title):
                continue
            items.append({
                "category": "reminder",
                "title": title,
                "when": due_dt.isoformat(timespec="minutes"),
                "meta": "due",
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
            title = str(ev.get("title") or ev.get("summary", ""))[:120]
            if _blacklisted_record(text=title):
                continue
            items.append({
                "category": "calendar",
                "title": title,
                "when": start_dt.isoformat(timespec="minutes"),
                "meta": "calendar",
            })
    except Exception as exc:
        logger.debug("mirror: calendar skip: %s", exc)

    items.sort(key=lambda x: x.get("when", ""))
    counts = {
        "calendar": sum(1 for i in items if i.get("category") == "calendar"),
        "reminders": sum(1 for i in items if i.get("category") == "reminder"),
    }
    return {
        "items": items[:10],
        "counts": counts,
        "services_consulted": [
            label for key, label in (("calendar", "Calendar"), ("reminders", "Reminders"))
            if counts.get(key)
        ],
        "reason": "collector_failed_fallback",
    }


def _clip(value: Any, limit: int = 120) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _pendientes_services_from_evidence(ev: dict[str, Any]) -> list[str]:
    services: list[str] = []
    if (ev.get("gmail") or {}).get("awaiting_reply"):
        services.append("Gmail")
    if ev.get("mail_unread"):
        services.append("Apple Mail")
    if ev.get("whatsapp"):
        services.append("WhatsApp")
    if ev.get("reminders"):
        services.append("Reminders")
    if ev.get("calendar_range") or ev.get("calendar"):
        services.append("Calendar")
    if ev.get("loops_stale") or ev.get("loops_activo"):
        services.append("Vault loops")
    return services


def _shape_pendientes_payload(
    ev: dict[str, Any],
    urgent: list[str],
    services: list[str],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def add(category: str, title: Any, *, meta: Any = "", when: Any = "") -> None:
        clean_title = _clip(title)
        if not clean_title:
            return
        if _blacklisted_record(text=f"{clean_title} {meta}".strip()):
            return
        items.append({
            "category": category,
            "title": clean_title,
            "meta": _clip(meta, 90),
            "when": str(when or ""),
        })

    for line in urgent[:3]:
        add("urgente", line)

    calendar = ev.get("calendar_range") or ev.get("calendar") or []
    for event in calendar[:3]:
        when_bits = [event.get("date_label"), event.get("time_range")]
        when = " · ".join(str(x) for x in when_bits if x) or event.get("start") or ""
        add(
            "calendar",
            event.get("title") or event.get("summary"),
            meta=when,
            when=event.get("start") or "",
        )

    reminders = ev.get("reminders") or []
    bucket_order = {"overdue": 0, "today": 1, "upcoming": 2, "undated": 3}
    reminders_sorted = sorted(reminders, key=lambda r: bucket_order.get(r.get("bucket"), 9))
    for reminder in reminders_sorted[:4]:
        bucket = reminder.get("bucket") or "reminder"
        bits = [bucket]
        if reminder.get("list"):
            bits.append(reminder["list"])
        if reminder.get("due"):
            bits.append(reminder["due"])
        add("reminder", reminder.get("name") or reminder.get("title"), meta=" · ".join(bits))

    gmail = (ev.get("gmail") or {}).get("awaiting_reply") or []
    for mail in gmail[:2]:
        who = mail.get("from") or ""
        age = f"{mail.get('days_old', 0)}d" if mail.get("days_old") is not None else ""
        add("gmail", mail.get("subject"), meta=" · ".join(x for x in (age, who) if x))

    unread = ev.get("mail_unread") or []
    for mail in unread[:2]:
        prefix = "VIP · " if mail.get("is_vip") else ""
        add("mail", mail.get("subject"), meta=f"{prefix}{mail.get('sender', '')}")

    loops = (ev.get("loops_activo") or [])[:3] + (ev.get("loops_stale") or [])[:2]
    for loop in loops[:4]:
        vault = f"[{loop.get('_vault')}] " if loop.get("_vault") else ""
        src = Path(str(loop.get("source_note") or "")).stem
        age = f"{loop.get('age_days', 0)}d"
        add("vault loop", loop.get("loop_text"), meta=f"{vault}{age} · {src}".strip())

    whatsapp = ev.get("whatsapp") or []
    for chat in whatsapp[:3]:
        count = int(chat.get("count") or 0)
        snippet = _clip(chat.get("last_snippet"), 70)
        meta = f"{count} msgs · {snippet}" if snippet else f"{count} msgs"
        add("whatsapp", chat.get("name"), meta=meta)

    counts = {
        "urgent": len(urgent),
        "calendar": len(calendar),
        "reminders": len(reminders),
        "gmail_awaiting": len(gmail),
        "mail_unread": len(unread),
        "whatsapp": len(whatsapp),
        "loops": len(ev.get("loops_activo") or []) + len(ev.get("loops_stale") or []),
        "contradictions": len(ev.get("contradictions") or []),
    }
    nonzero_counts = {k: v for k, v in counts.items() if v}
    return {
        "items": items[:10],
        "urgent": urgent[:5],
        "counts": counts,
        "services_consulted": services,
        "summary": " · ".join(f"{k}: {v}" for k, v in nonzero_counts.items()),
    }


_WA_MEDIA_LABELS = {
    "image": "imagen",
    "video": "video",
    "audio": "audio",
    "document": "archivo",
    "sticker": "sticker",
}


def _wa_message_snippet(
    content: Any,
    media_type: Any = None,
    filename: Any = None,
    *,
    limit: int = 120,
) -> str:
    text = str(content or "").strip().replace("\n", " ")
    if not text:
        media = str(media_type or "").strip().lower()
        label = _WA_MEDIA_LABELS.get(media, media or "")
        if label:
            text = f"[{label}]"
            if filename:
                text += f" {Path(str(filename)).name}"
    return _clip(text, limit)


def _wa_chat_name_excluded(name: Any) -> bool:
    try:
        from rag.integrations.whatsapp import whatsapp_chat_name_excluded  # noqa: PLC0415

        return bool(whatsapp_chat_name_excluded(str(name or "")))
    except Exception:
        return False


def _blacklisted_record(
    *,
    path: Any = None,
    chat_name: Any = None,
    person: Any = None,
    text: Any = None,
) -> bool:
    try:
        from rag.exclusions import should_exclude_record  # noqa: PLC0415

        return bool(
            should_exclude_record(
                path=path,
                chat_name=chat_name,
                person=person,
                text=text,
            )
        )
    except Exception:
        return False


def _blacklist_relpath(path: Path, vault: Path) -> str:
    try:
        return str(path.relative_to(vault))
    except ValueError:
        return str(path)


def _wa_item_excluded(item: dict[str, Any]) -> bool:
    snippets: list[str] = [
        str(item.get("last_snippet") or ""),
        str(item.get("topic_hint") or ""),
    ]
    for msg in item.get("recent_context") or []:
        if isinstance(msg, dict):
            snippets.append(str(msg.get("snippet") or ""))
    return _blacklisted_record(
        chat_name=item.get("name"),
        text=" ".join(s for s in snippets if s),
    )


def _filter_whatsapp_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in items
        if not _wa_chat_name_excluded(item.get("name")) and not _wa_item_excluded(item)
    ]


def _whatsapp_recent_context(
    con: sqlite3.Connection,
    jid: str,
    *,
    hours: int = 12,
    limit: int = 8,
) -> list[dict[str, Any]]:
    try:
        rows = con.execute(
            """
            SELECT timestamp, is_from_me, content, media_type, filename
            FROM messages
            WHERE chat_jid = ?
              AND datetime(timestamp) > datetime('now', ?)
            ORDER BY datetime(timestamp) DESC
            LIMIT ?
            """,
            (jid, f"-{int(hours)} hours", int(limit)),
        ).fetchall()
    except sqlite3.Error:
        return []

    items: list[dict[str, Any]] = []
    for row in reversed(rows):
        snippet = _wa_message_snippet(
            row["content"],
            row["media_type"],
            row["filename"],
            limit=140,
        )
        if not snippet:
            continue
        items.append({
            "ts": row["timestamp"],
            "who": "yo" if int(row["is_from_me"] or 0) else "ellos",
            "snippet": snippet,
            "media_type": row["media_type"] or "",
        })
    return items


def _whatsapp_topic_hint(context: list[dict[str, Any]]) -> str:
    joined = " ".join(str(item.get("snippet") or "") for item in context).lower()
    has_media = any(item.get("media_type") for item in context)
    bits: list[str] = []
    if any(k in joined for k in ("cumple", "cumpleaños", "astor", "regalo")):
        bits.append("cumpleaños/regalos de Astor")
    if any(k in joined for k in ("invitación", "invitaciones", "dibujo", "logo", "foto de referencia")):
        bits.append("invitaciones y referencia visual")
    if any(k in joined for k in ("trato", "seca", "normal", "indiferente", "traumático")):
        bits.append("tono de la conversación")
    if any(k in joined for k in ("post update", "tareas", "update", "confirmo")):
        bits.append("coordinación operativa")
    if has_media:
        bits.append("intercambio de medios")
    if bits:
        return "; ".join(dict.fromkeys(bits))
    for item in reversed(context):
        snippet = str(item.get("snippet") or "")
        if snippet and not snippet.startswith("["):
            return _clip(snippet, 90)
    return ""


def _enrich_whatsapp_items(
    today: list[dict[str, Any]],
    recent: list[dict[str, Any]],
    unreplied: list[dict[str, Any]],
) -> None:
    try:
        from rag import WHATSAPP_DB_PATH  # noqa: PLC0415
    except Exception:
        return
    if not WHATSAPP_DB_PATH.is_file():
        return

    by_jid: dict[str, list[dict[str, Any]]] = {}
    for collection in (unreplied, recent, today):
        for item in collection:
            jid = str(item.get("jid") or "")
            if jid:
                by_jid.setdefault(jid, []).append(item)
    if not by_jid:
        return

    try:
        con = sqlite3.connect(f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0)
        con.row_factory = sqlite3.Row
    except sqlite3.Error:
        return
    try:
        for jid, items in list(by_jid.items())[:8]:
            context = _whatsapp_recent_context(con, jid)
            if not context:
                continue
            topic = _whatsapp_topic_hint(context)
            for item in items:
                item["recent_context"] = context
                if topic:
                    item["topic_hint"] = topic
    finally:
        con.close()


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
              SELECT chat_jid, content, media_type, filename, is_from_me, timestamp,
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
                   lm.media_type AS media_type,
                   lm.filename   AS filename,
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
        if _wa_chat_name_excluded(display_name):
            continue
        snippet = _wa_message_snippet(row["last_content"], row["media_type"], row["filename"])
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
    try:
        _enrich_whatsapp_items(today, recent, unreplied)
    except Exception as exc:
        logger.debug("mirror: whatsapp enrich failed: %s", exc)
    today = _filter_whatsapp_items(today)
    recent = _filter_whatsapp_items(recent)
    unreplied = _filter_whatsapp_items(unreplied)

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
                    rel_note = _blacklist_relpath(note, Path(vault))
                    if _blacklisted_record(path=rel_note, text=note.stem):
                        continue
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
        if _blacklisted_record(path=str(rel), text=note.stem):
            continue
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
        return _source_spotify_fallback(reason="telemetry_unavailable")
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
        if items:
            return {"items": items, "mode": "top_artists"}
        return _source_spotify_fallback(reason="no_spotify_log")
    except sqlite3.Error as exc:
        fallback = _source_spotify_fallback(reason="spotify_sql_failed")
        fallback["error"] = str(exc)
        return fallback
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _source_spotify_fallback(reason: str) -> dict[str, Any]:
    """Use local Spotify state/history when the aggregate log is empty."""
    try:
        from rag.integrations.spotify_local import (  # noqa: PLC0415
            now_playing,
            recent_tracks_lookback,
        )

        recent = recent_tracks_lookback(days=14, limit=5) or []
        if recent:
            return {
                "items": [
                    {
                        "artist": t.get("artist") or "",
                        "track": t.get("name") or "",
                        "album": t.get("album") or "",
                        "plays": 1,
                        "distinct_tracks": 1,
                        "last_seen": t.get("last_seen"),
                    }
                    for t in recent
                ],
                "mode": "recent_tracks",
                "reason": reason,
            }
        current = now_playing(timeout=1.0)
        if current:
            return {
                "items": [{
                    "artist": current.get("artist") or "",
                    "track": current.get("name") or "",
                    "album": current.get("album") or "",
                    "plays": 1,
                    "distinct_tracks": 1,
                    "state": current.get("state") or "",
                }],
                "mode": "now_playing",
                "reason": reason,
            }
    except Exception as exc:
        snapshot = _source_spotify_snapshot_fallback(reason=reason)
        if snapshot.get("items"):
            snapshot["apple_script_error"] = str(exc)[:200]
            return snapshot
        return {"items": [], "reason": reason, "error": str(exc)[:200]}

    snapshot = _source_spotify_snapshot_fallback(reason=reason)
    if snapshot.get("items"):
        return snapshot
    return {"items": [], "reason": reason}


_SPOTIFY_TRACK_RE = re.compile(
    r"^-\s+(?:`(?P<played>[^`]+)`\s+)?"
    r"\[(?P<track>[^\]]+)\]\((?P<url>[^)]+)\)\s+—\s+"
    r"(?P<artist>[^·\n]+)"
)
_SPOTIFY_ARTIST_RE = re.compile(r"^-\s+\[(?P<artist>[^\]]+)\]\((?P<url>[^)]+)\)")


def _source_spotify_snapshot_fallback(reason: str) -> dict[str, Any]:
    """Read the latest Spotify ingest notes from the vault.

    The live desktop log can be empty when Spotify is closed or the listener
    has not observed a track yet. The external-ingest notes still contain the
    last synced top/recent snapshot, which is better signal than an empty card.
    """
    for _vault_name, vault in _mirror_vaults():
        spotify_dir = Path(vault) / "99-obsidian/99-AI/external-ingest/Spotify"
        top_path = spotify_dir / "_top.md"
        top = _parse_spotify_top_note(top_path)
        if top.get("items"):
            top["reason"] = reason
            return top

        recent_files = sorted(
            (p for p in spotify_dir.glob("*.md") if p.name != "_top.md"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for path in recent_files[:3]:
            recent = _parse_spotify_recent_note(path)
            if recent.get("items"):
                recent["reason"] = reason
                return recent
    return {"items": [], "reason": reason}


def _frontmatter_value(raw: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", raw, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_spotify_top_note(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"items": []}
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"items": []}

    items: list[dict[str, Any]] = []
    in_tracks = False
    for line in raw.splitlines():
        if line.startswith("## Top tracks"):
            in_tracks = True
            continue
        if line.startswith("## Top artists"):
            break
        if not in_tracks:
            continue
        m = _SPOTIFY_TRACK_RE.match(line.strip())
        if not m:
            continue
        items.append({
            "track": m.group("track").strip(),
            "artist": m.group("artist").strip(),
            "url": m.group("url").strip(),
            "plays": 1,
            "distinct_tracks": 1,
        })
        if len(items) >= 5:
            break

    if not items:
        in_artists = False
        for line in raw.splitlines():
            if line.startswith("## Top artists"):
                in_artists = True
                continue
            if not in_artists:
                continue
            m = _SPOTIFY_ARTIST_RE.match(line.strip())
            if not m:
                continue
            items.append({
                "artist": m.group("artist").strip(),
                "url": m.group("url").strip(),
                "plays": 1,
                "distinct_tracks": 1,
            })
            if len(items) >= 5:
                break

    return {
        "items": items,
        "mode": "top_snapshot",
        "snapshot_date": _frontmatter_value(raw, "refreshed_date"),
    }


def _parse_spotify_recent_note(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"items": []}
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"items": []}

    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        m = _SPOTIFY_TRACK_RE.match(line.strip())
        if not m:
            continue
        items.append({
            "track": m.group("track").strip(),
            "artist": m.group("artist").strip(),
            "url": m.group("url").strip(),
            "played_at": (m.group("played") or "").strip(),
            "plays": 1,
            "distinct_tracks": 1,
        })
        if len(items) >= 5:
            break

    return {
        "items": items,
        "mode": "recent_snapshot",
        "snapshot_date": _frontmatter_value(raw, "snapshot_date") or path.stem,
    }


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


def _screen_observer_enabled() -> bool:
    """Return whether the passive screen observer is configured for this user.

    The Mirror web process does not necessarily inherit the supervisor plist
    env, so the state file is the source of truth for user intent. Keep the
    env fallback for manual shells and tests that run the observer directly.
    """
    env = os.environ.get("RAG_SCREEN_OBSERVE", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    try:
        from rag.integrations.peekaboo import _observe_state_enabled  # noqa: PLC0415
        return bool(_observe_state_enabled())
    except Exception:
        return False


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
    observer_enabled = _screen_observer_enabled()

    def _empty(reason: str) -> dict[str, Any]:
        return {
            "recent": [],
            "today": [],
            "count_today": 0,
            "count_7d": 0,
            "observer_enabled": observer_enabled,
            "reason": reason,
        }

    conn = _open_telemetry_ro()
    if conn is None:
        return _empty("telemetry_unavailable")
    try:
        # Verificá que la tabla existe — feature recién agregado, DB viejas
        # pueden no tenerla todavía.
        try:
            conn.execute("SELECT 1 FROM rag_screen_observations LIMIT 0")
        except sqlite3.Error:
            return _empty("table_missing")

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

        reason = None
        if not recent and not today_gallery and count_today == 0 and count_7d == 0:
            reason = "no_observations" if observer_enabled else "observer_disabled"

        out = {
            "recent": recent,
            "today": today_gallery,
            "count_today": count_today,
            "count_7d": count_7d,
            "observer_enabled": observer_enabled,
        }
        if reason:
            out["reason"] = reason
        return out
    except Exception as exc:
        logger.warning("mirror: screen_context failed: %s", exc)
        out = _empty("source_failed")
        out["error"] = str(exc)[:200]
        return out
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
            response["sources"] = dict(cached.get("sources") or {})
            response["cache_hit"] = True
            return _refresh_live_sources(response, date)

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

Generá 3-5 insights en español rioplatense (voseo). Cada insight debe tener
2 frases y 180-450 caracteres: una observación concreta y una lectura o acción
prudente.

Reglas de grounding:
- Usá SOLO datos presentes en el snapshot. No inventes fechas, emociones,
  intenciones, queries, correlaciones ni actividad que no esté ahí.
- No escribas "mencionaste", "dijiste", "prometiste" ni "te comprometiste"
  salvo que el snapshot traiga una cita literal de una fuente conversacional.
- Los ítems `vault loop` son pendientes detectados por regex en notas: no son
  frases dichas hoy. Presentalos como "aparece como pendiente detectado" o
  omitilos si parecen ejemplos, plantillas o texto citado.
- No digas "sigue sin dedicar tiempo" o "no respondiste" salvo que el snapshot
  incluya evidencia explícita para sostenerlo.
- Si hay bloque WhatsApp/WZP, evaluá actividad, chats sin responder, urgencia y
  tono con cuidado. Citá solo snippets presentes.
- Si no hay evidencia suficiente para un insight, omitilo.

Formato JSON estricto: ``{{"insights": ["...", "...", "..."]}}``. Sin markdown.

Generá SOLO el JSON, nada más:"""


_INSIGHTS_TIMEOUT_S = 8.0
_INSIGHT_MAX_CHARS = 800
_UNGROUNDED_CLAIM_RE = re.compile(
    r"\b(?:hoy\s+)?(?:mencionaste|dijiste|contaste|prometiste)\b"
    r"|\bte comprometiste\b"
    r"|\bsigue sin dedicar tiempo\b",
    re.IGNORECASE,
)
_QUOTED_TEXT_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{3,120})[\"'“”‘’]")


def _insight_grounded(summary: str, insight: str) -> bool:
    """Conservative post-filter for common helper hallucinations."""
    if len(insight.strip(" .…")) < 24:
        return False
    if _blacklisted_record(text=insight):
        return False
    summary_l = summary.lower()
    insight_l = insight.lower()
    if _UNGROUNDED_CLAIM_RE.search(insight_l):
        return False
    for quoted in _QUOTED_TEXT_RE.findall(insight):
        quote = quoted.strip().lower()
        if quote and quote not in summary_l:
            return False
    return True


def _format_wait(hours: Any) -> str:
    try:
        value = float(hours or 0)
    except (TypeError, ValueError):
        value = 0.0
    if value < 0.05:
        return "recién"
    if value < 1:
        return f"{max(1, round(value * 60))}m"
    if value < 24:
        return f"{value:.1f}h"
    return f"{int(value // 24)}d"


def _format_days_ago(days: Any) -> str:
    try:
        value = int(days)
    except (TypeError, ValueError):
        return f"hace {days}d"
    if value <= 0:
        return "hoy"
    if value == 1:
        return "ayer"
    return f"hace {value}d"


def _merge_insights(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*primary, *secondary]:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen or _blacklisted_record(text=text):
            continue
        seen.add(key)
        merged.append(text[:_INSIGHT_MAX_CHARS])
        if len(merged) >= 5:
            break
    return merged


def _whatsapp_chat_brief(chat: dict[str, Any]) -> str:
    name = str(chat.get("name") or "chat")
    topic = _clip(chat.get("topic_hint"), 120)
    context = chat.get("recent_context") or []
    bits: list[str] = []
    for msg in context[-4:]:
        snippet = _clip(msg.get("snippet"), 70)
        if snippet:
            bits.append(f"{msg.get('who', '?')}: \"{snippet}\"")
    detail = " / ".join(bits[-3:])
    if topic and detail:
        return f"{name}: {topic}; {detail}"
    if topic:
        return f"{name}: {topic}"
    if detail:
        return f"{name}: {detail}"
    snippet = _clip(chat.get("last_snippet"), 90)
    return f"{name}: \"{snippet}\"" if snippet else name


def _deterministic_insights(mirror_data: dict[str, Any]) -> list[str]:
    """Grounded insights from the structured snapshot, no generation."""
    s = mirror_data.get("sources", {})
    out: list[str] = []

    wa = s.get("whatsapp", {}) if isinstance(s.get("whatsapp"), dict) else {}
    wa_counts = wa.get("counts") or {}
    unreplied = wa.get("unreplied") or []
    if unreplied:
        first = unreplied[0]
        top_today = wa.get("today") or []
        names = ", ".join(
            str(x.get("name") or "chat") for x in unreplied[:3]
        )
        snippet = _clip(first.get("last_snippet"), 90)
        detail = f"El más reciente es {first.get('name', 'un chat')} ({_format_wait(first.get('hours_waiting'))})"
        if snippet:
            detail += f": \"{snippet}\"."
        else:
            detail += "."
        active_bits: list[str] = []
        for chat in top_today[:2]:
            topic = _clip(chat.get("topic_hint"), 130)
            if not topic:
                topic = _clip(_whatsapp_chat_brief(chat), 130)
            if topic:
                active_bits.append(
                    f"{chat.get('name', 'chat')} ({chat.get('count', 0)}) gira alrededor de {topic}"
                )
        active_text = ""
        if active_bits:
            active_text = " Los chats más activos no son genéricos: " + "; ".join(active_bits) + "."
        out.append(
            f"WZP está movido: {wa_counts.get('today_chats', len(wa.get('today') or []))} chats hoy y "
            f"{wa_counts.get('unreplied_chats', len(unreplied))} con respuesta tuya pendiente. "
            f"{detail}{active_text} Si tenés poco margen, priorizá {names} según el bloqueo real de cada charla."
        )

    pend = s.get("pendientes", {}) if isinstance(s.get("pendientes"), dict) else {}
    pend_items = pend.get("items") or []
    if pend_items:
        cats: dict[str, int] = {}
        for item in pend_items:
            cat = str(item.get("category") or "otro")
            cats[cat] = cats.get(cat, 0) + 1
        cat_text = ", ".join(f"{v} señales de {k}" for k, v in sorted(cats.items()))
        first = pend_items[0]
        meta = f" ({_clip(first.get('meta'), 80)})" if first.get("meta") else ""
        out.append(
            f"Tus pendientes visibles mezclan {cat_text}. El primero que aparece es "
            f"\"{_clip(first.get('title'), 120)}\"{meta}; conviene tratar los `vault loop` "
            "como señales para revisar la fuente, no como compromisos textuales."
        )

    mood = s.get("mood_today", {}) if isinstance(s.get("mood_today"), dict) else {}
    timeline = (
        s.get("mood_timeline", {}).get("days", [])
        if isinstance(s.get("mood_timeline"), dict) else []
    )
    if mood.get("score") is not None:
        text = (
            f"El mood de hoy aparece en {float(mood.get('score')):+.2f} con "
            f"{int(mood.get('n_signals') or 0)} señales."
        )
        scores_recent = [d.get("score", 0) for d in timeline[-7:] if d.get("score") is not None]
        scores_old = [d.get("score", 0) for d in timeline[-14:-7] if d.get("score") is not None]
        if scores_recent and scores_old:
            avg_recent = sum(scores_recent) / len(scores_recent)
            avg_old = sum(scores_old) / len(scores_old)
            text += (
                f" En la ventana de 14 días la tendencia va de {avg_old:+.2f} a "
                f"{avg_recent:+.2f}; miralo como contexto, no como diagnóstico."
            )
        else:
            text += " Todavía no hay suficiente serie reciente para comparar tendencia con confianza."
        out.append(text)

    projects = (
        s.get("active_projects", {}).get("items", [])
        if isinstance(s.get("active_projects"), dict) else []
    )
    if projects:
        top = projects[0]
        others = ", ".join(str(p.get("name") or "?") for p in projects[1:3])
        tail = f" También aparecen {others}." if others else ""
        out.append(
            f"Tu actividad de notas está concentrada en {top.get('name', '?')}: "
            f"{top.get('note_count_30d', 0)} notas tocadas en 30 días y último movimiento "
            f"{_format_days_ago(top.get('days_ago'))}.{tail} Es una buena señal de dónde está el foco real."
        )

    dormant = (
        s.get("dormant_notes", {}).get("items", [])
        if isinstance(s.get("dormant_notes"), dict) else []
    )
    if dormant:
        names = ", ".join(str(d.get("title") or "?") for d in dormant[:3])
        out.append(
            f"Hay memoria útil quedando atrás: {names}. Están dormidas hace más de 30 días, "
            "así que pueden ser buen material para rescatar si conectan con lo que estás moviendo ahora."
        )

    return [
        x[:_INSIGHT_MAX_CHARS]
        for x in out
        if not _blacklisted_record(text=x)
    ][:5]


def generate_insights(mirror_data: dict[str, Any]) -> dict[str, Any]:
    """Genera 3-5 insights vía LLM basado en el mirror snapshot.

    Sync con timeout 8s enforceado via ThreadPoolExecutor. Si falla → empty.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout  # noqa: PLC0415

    summary = _summarize_for_llm(mirror_data)
    grounded = _deterministic_insights(mirror_data)
    if len(grounded) >= 3:
        return {"insights": grounded[:5], "model": "grounded-rules"}
    try:
        from rag.llm_backend import get_backend  # noqa: PLC0415
    except ImportError:
        if grounded:
            return {"insights": grounded, "model": "grounded-rules"}
        return {"insights": [], "error": "llm_backend no disponible"}

    def _call() -> dict[str, Any]:
        return _generate_insights_inner(summary)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            result = fut.result(timeout=_INSIGHTS_TIMEOUT_S)
            if grounded:
                result["insights"] = _merge_insights(
                    grounded, result.get("insights") or [],
                )
                result["model"] = f"grounded-rules+{result.get('model', 'llm')}"
            return result
    except FuturesTimeout:
        logger.warning("mirror insights LLM timed out after %ss", _INSIGHTS_TIMEOUT_S)
        if grounded:
            return {"insights": grounded, "model": "grounded-rules", "warning": "llm timeout"}
        return {"insights": [], "error": "timeout"}
    except Exception as exc:
        logger.warning("mirror insights LLM failed: %s", exc)
        if grounded:
            return {"insights": grounded, "model": "grounded-rules", "warning": str(exc)[:200]}
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

        def _insights_option_int(
            env_name: str, helper_key: str, default: int, minimum: int,
        ) -> int:
            env_raw = os.environ.get(env_name)
            raw = env_raw if env_raw is not None else HELPER_OPTIONS.get(helper_key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = default
            if env_raw is not None:
                return max(1, value)
            return max(minimum, value)

        helper_opts = ChatOptions(
            temperature=float(HELPER_OPTIONS.get("temperature", 0)),
            seed=int(HELPER_OPTIONS.get("seed", 42)),
            num_ctx=_insights_option_int(
                "RAG_MIRROR_INSIGHTS_NUM_CTX", "num_ctx", 2048, 2048,
            ),
            num_predict=_insights_option_int(
                "RAG_MIRROR_INSIGHTS_NUM_PREDICT", "num_predict", 384, 384,
            ),
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
        grounded = [
            text[:_INSIGHT_MAX_CHARS]
            for s in insights
            if (text := str(s).strip()) and _insight_grounded(summary, text)
        ]
        return {"insights": grounded[:5], "model": helper_model}
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
        lines.append("\nPendientes detectados (no asumir que fueron mencionados hoy):")
        for p in pend[:5]:
            bits = [str(p.get("category") or "?")]
            if p.get("meta"):
                bits.append(str(p["meta"]))
            if p.get("when"):
                bits.append(f"when={p['when']}")
            lines.append(f"- {p.get('title', '?')} [{'; '.join(bits)}]")

    wa = s.get("whatsapp", {})
    wa_today = wa.get("today", []) if isinstance(wa, dict) else []
    wa_unreplied = wa.get("unreplied", []) if isinstance(wa, dict) else []
    wa_recent = wa.get("recent_inbound", []) if isinstance(wa, dict) else []
    if wa_today or wa_unreplied or wa_recent:
        counts = wa.get("counts", {}) if isinstance(wa, dict) else {}
        lines.append(
            "\nWhatsApp/WZP:"
            f" {counts.get('today_chats', len(wa_today))} chats hoy,"
            f" {counts.get('unreplied_chats', len(wa_unreplied))} con respuesta tuya pendiente"
        )
        if wa_unreplied:
            lines.append("Chats WZP esperando respuesta tuya (último mensaje inbound):")
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
