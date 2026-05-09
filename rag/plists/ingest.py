"""Cross-source ETL factories: ingest-whatsapp + ingest-cross-source.

WhatsApp queda aparte (cadencia 15min, hot path); el resto wrappea
gmail/calendar/reminders/calls/safari/drive/pillow en un solo worker
1h con TTL per-source + gates por credencial.

Migrado de rag/plists/_legacy.py en Phase 3 commit 3 (2026-05-09).
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist

__all__ = [
    "_ingest_cross_source_plist",
    "_ingest_whatsapp_plist",
]


def _ingest_whatsapp_plist(rag_bin: str) -> str:
    """Cross-source: WhatsApp ingester, cada 15min.

    Incremental por design — lee `messages` con `timestamp > cursor` desde la
    bridge SQLite, chunka, upsertea. En steady state con 0 mensajes nuevos el
    run termina en <1s (solo overhead de abrir la DB). Primer run full scan
    tarda ~1min por 4000 chunks (medido 2026-04-21: 12984 msgs / 65s / 4070
    chunks). Interval 900s es lo suficientemente freq para que queries tipo
    "último mensaje de X" no se sientan stale, y lo suficientemente spaced
    para no competir con watch/serve en CPU.

    `RAG_INDEX_LOCAL_EMBED=1` (Ola 6, 2026-05-06 cero-Ollama): embedder
    in-process siempre activo. Reemplaza el bulk-embed path via Ollama HTTP
    que requería `LLM_KEEP_ALIVE=-1` para pin bge-m3 en VRAM.

    `RunAtLoad=true` (2026-04-22): garantiza run inmediato al instalar o
    post-reboot; sin esto el primer refresh se demoraba hasta 15min tras
    cargar el servicio — suficiente para que "último mensaje de X" post-
    arranque del Mac devuelva data stale. El incremental cost es chico (<1s
    cuando no hay nuevos).
    """
    out, err = _logs("ingest-whatsapp")
    return _render_plist({
        "label": "com.fer.obsidian-rag-ingest-whatsapp",
        "program_arguments": [rag_bin, "index", "--source", "whatsapp"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 900},
        "run_at_load": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _ingest_cross_source_plist(rag_bin: str) -> str:
    """Worker unificado cross-source — every 1h. Consolidación 2026-05-04.

    Reemplaza 7 plists individuales (ingest-{gmail,calendar,reminders,
    calls,safari,drive,pillow}). El comando `rag ingest-cross-source`
    itera cada sub-ingester y lo dispara si la edad desde el último run
    supera su TTL (`_INGEST_TTL_SECONDS`):

      - gmail / calendar / reminders : 1h (hourly)
      - calls / safari / drive       : 6h
      - pillow                       : 24h (+ gate 9am)

    Cursors per-source en `~/.local/share/obsidian-rag/ingest_cursors.json`
    (JSON flat, atomic write). Gates por creds se aplican acá (no en
    el install) — skip silencioso + cursor update si el token no existe.

    WhatsApp NO está acá (cadencia 15 min, plist propio).

    RunAtLoad=true: corre inmediato al instalar / post-reboot para que
    un user nuevo vea data cross-source en la primera hora (en vez de
    esperar la próxima hora redonda).
    """
    out, err = _logs("ingest-cross-source")
    return _render_plist({
        "label": "com.fer.obsidian-rag-ingest-cross-source",
        "program_arguments": [rag_bin, "ingest-cross-source"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 3600},
        "run_at_load": True,
        "stdout_path": out,
        "stderr_path": err,
    })
