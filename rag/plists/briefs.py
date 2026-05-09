"""Brief outputs user-facing — morning / today / digest.

Schedules dinámicos: leen `rag_brief_schedule_prefs` vía
`rag.brief_schedule.get_brief_schedule_pref()` desde `_spec.py` antes de
generar el XML. Si hay override, usa esa hora/minute; si no, defaults
históricos. Auto-tune del schedule (`brief-auto-tune`) MUTA esos prefs
en background — vive en `proactive.py` por ser un *driver* de briefs,
no un brief en sí.
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist

__all__ = ["_digest_plist", "_morning_plist", "_today_plist"]


def _digest_plist(rag_bin: str, hour: int = 22, minute: int = 0) -> str:
    """Weekly digest plist — Sunday `hour:minute` (default 22:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29).
    Defaults preserve the historical schedule when no pref exists; the
    auto-tune writer enforces the safe band before persisting, so a
    pref-driven override never lands outside `[21:00, 23:30]`.
    """
    out, err = _logs("digest")
    return _render_plist({
        "label": "com.fer.obsidian-rag-digest",
        "program_arguments": [rag_bin, "digest"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar": {
                "Weekday": 0,
                "Hour": int(hour),
                "Minute": int(minute),
            },
        },
        "stdout_path": out,
        "stderr_path": err,
    })


def _morning_plist(rag_bin: str, hour: int = 7, minute: int = 0) -> str:
    """Morning brief plist — Mon-Fri `hour:minute` (default 07:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29).
    Defaults preserve the historical schedule when no pref exists; the
    auto-tune writer enforces the safe band before persisting, so a
    pref-driven override never lands outside `[06:30, 09:00]`.

    Voice brief opt-in (Anticipatory Phase 2.C, 2026-04-29):
    ``RAG_MORNING_VOICE`` se inyecta vacío por default — text-only
    behavior. Para activar el audio matinal: el user edita el plist
    seteando el value a ``1`` (o ``true``/``yes``) y hace
    ``launchctl bootout`` + ``bootstrap`` (o re-instala con
    ``rag setup``). Voz de la lectura: env var ``TTS_VOICE`` (default
    ``Mónica``). Cap de audio: 5MB; si excede, fallback a text-only
    silencioso. Audios viejos (>30d) limpiados por ``rag maintenance``.
    """
    h = int(hour)
    m = int(minute)
    out, err = _logs("morning")
    return _render_plist({
        "label": "com.fer.obsidian-rag-morning",
        "program_arguments": [rag_bin, "morning"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EXPLORE": "1",
            "RAG_MORNING_VOICE": "",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar_list": [
                {"Weekday": wd, "Hour": h, "Minute": m}
                for wd in (1, 2, 3, 4, 5)
            ],
        },
        "stdout_path": out,
        "stderr_path": err,
    })


def _today_plist(rag_bin: str, hour: int = 22, minute: int = 0) -> str:
    """Evening "today" brief plist — Mon-Fri `hour:minute` (default 22:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29). The
    safe band for `today` is `[18:00, 21:00]`; the historical default
    (22:00) lives outside that band by design — it's the user's
    explicit baseline, only auto-tune-driven shifts have to stay inside.
    """
    h = int(hour)
    m = int(minute)
    out, err = _logs("today")
    return _render_plist({
        "label": "com.fer.obsidian-rag-today",
        "program_arguments": [rag_bin, "today"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EXPLORE": "1",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar_list": [
                {"Weekday": wd, "Hour": h, "Minute": m}
                for wd in (1, 2, 3, 4, 5)
            ],
        },
        "stdout_path": out,
        "stderr_path": err,
    })
