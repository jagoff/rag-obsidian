"""Hot-path frecuentes migrados al supervisor (F3.1 shadow mode).

Re-implementación in-supervisor de los 8 daemons launchd con cadencia
≤1h:

| Cadence  | Job                | Plist viejo                            |
|----------|--------------------|----------------------------------------|
| 15min    | anticipate         | com.fer.obsidian-rag-anticipate        |
| 5min     | routing_rules      | com.fer.obsidian-rag-routing-rules     |
| 5min     | wa_fast            | com.fer.obsidian-rag-wa-fast           |
| 15min    | ingest_whatsapp    | com.fer.obsidian-rag-ingest-whatsapp   |
| 1h       | ingest_cross_source| com.fer.obsidian-rag-ingest-cross-source|
| 30min    | mood_poll          | com.fer.obsidian-rag-mood-poll         |
| 5min     | spotify_poll       | com.fer.obsidian-rag-spotify-poll      |
| 30min    | wa_tasks           | com.fer.obsidian-rag-wa-tasks          |

SHADOW MODE igual que ``nightly.py`` (F2): subprocess wrappers con
paridad estricta de env vars + args. mood_poll y spotify_poll usan
``.venv/bin/python scripts/<name>.py`` directo (no van por el binario
``rag``).

mood_poll respeta su gate opt-in propio: el script bailea silently si
``~/.local/share/obsidian-rag/mood_enabled`` no existe — el supervisor
NO duplica el gate, simplemente confía en el script.

Telemetría: igual que ``nightly.py`` — exit_code/stdout_lines/
stderr_lines/last_stderr en ``rag_supervisor_jobs.signals``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.runtime.jobs.nightly import _RAG_BIN, _run_subprocess
from rag.runtime.scheduler import interval

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VENV_PY = _REPO_ROOT / ".venv" / "bin" / "python"
_UV_TOOL_PY = Path.home() / ".local/share/uv/tools/obsidian-rag/bin/python3"


# ── VLM image captioner (Game-Changer G5, 2026-05-11) ──────────────────────


@interval(
    hours=2,
    label="vault_image_captioner",
    description="Genera captions VLM (granite-vision MLX) para imágenes del vault + bridge media. Sidecars .caption.md indexables por rag watch.",
)
def vault_image_captioner_job() -> dict[str, Any]:
    """Corre `scripts/vault_image_captioner.py --limit 10` cada 2h.

    Cap pequeño porque cada caption tarda 5-10s en MPS. En 1 día procesa
    ~120 imágenes — suficiente para empezar a indexar el backlog +
    seguir el ritmo de imágenes nuevas en WA.
    """
    return _run_subprocess(
        [str(_VENV_PY), str(_REPO_ROOT / "scripts" / "vault_image_captioner.py"),
         "--limit", "10"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONPATH": str(_REPO_ROOT),
            "RAG_VLM_CAPTION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=1800,  # 30min — 10 imgs × 10s típico = 100s, margen amplio
    )


# ── WA voice backfill (Game-Changer G1, 2026-05-11) ────────────────────────


@interval(
    minutes=15,
    label="wa_voice_backfill",
    description="Transcribe en batch los voice notes inbound de WA que el user no haya clickeado en /wa. Pobla rag_wa_voice_transcripts + escribe notas al vault.",
)
def wa_voice_backfill_job() -> dict[str, Any]:
    """Corre `scripts/wa_voice_backfill.py --days 7 --limit 20` cada 15min.

    Es el complemento async del endpoint on-demand
    /api/wa/voice/transcript/{msg_id}. Sin este job, solo los audios
    que el user clickea se transcriben — pierde ~80% del contenido WA.
    """
    return _run_subprocess(
        [str(_VENV_PY), str(_REPO_ROOT / "scripts" / "wa_voice_backfill.py"),
         "--days", "7", "--limit", "20"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONPATH": str(_REPO_ROOT),
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,  # 15min — 20 audios × ~5s típico = 100s, margen amplio
    )


# ── Anticipate ──────────────────────────────────────────────────────────────


@interval(
    minutes=15,
    label="anticipate",
    description="Anticipatory agent — push proactivo cada 15min (game-changer).",
)
def anticipate_job() -> dict[str, Any]:
    """Equivalente a ``rag anticipate run`` del plist viejo. Schedule
    cada 15min. daily_cap=3 limita pushes; cadencia más alta no compra
    coverage adicional."""
    return _run_subprocess(
        [_RAG_BIN, "anticipate", "run"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=600,  # 10 min — anticipate típicamente <30s, margen amplio
    )


# ── Routing rules detector ──────────────────────────────────────────────────


@interval(
    minutes=5,
    label="routing_rules",
    description="Routing pattern detector — extrae rutas + auto-promote.",
)
def routing_rules_job() -> dict[str, Any]:
    """Equivalente a ``rag routing extract-rules --auto-promote`` del
    plist viejo. F4.1 lo va a reemplazar con sqlite trigger reactivo;
    por ahora cron 5min."""
    return _run_subprocess(
        [_RAG_BIN, "routing", "extract-rules", "--auto-promote"],
        extra_env={"NO_COLOR": "1", "TERM": "dumb"},
        timeout=300,
    )


# ── WhatsApp fast worker ────────────────────────────────────────────────────


@interval(
    minutes=5,
    label="wa_fast",
    description="WA worker time-sensitive: remind-wa + wa-scheduled-send.",
)
def wa_fast_job() -> dict[str, Any]:
    """Equivalente a ``rag wa-fast`` del plist viejo. Worker unificado
    que corre remind-wa + wa-scheduled-send en serie. Idempotente."""
    return _run_subprocess(
        [_RAG_BIN, "wa-fast"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=300,
    )


# ── WhatsApp ingester ───────────────────────────────────────────────────────


@interval(
    minutes=15,
    label="ingest_whatsapp",
    description="WhatsApp ingester — bridge SQLite → corpus chunks.",
)
def ingest_whatsapp_job() -> dict[str, Any]:
    """Equivalente a ``rag index --source whatsapp``. F4.3 lo va a
    reemplazar con file watcher reactivo sobre bridge SQLite."""
    return _run_subprocess(
        [_RAG_BIN, "index", "--source", "whatsapp"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,  # 15min — primer run full scan tarda ~1min, margen
    )


# ── Cross-source ingester ───────────────────────────────────────────────────


@interval(
    hours=1,
    label="ingest_cross_source",
    description="Cross-source ETL: gmail/calendar/reminders/calls/safari/drive/pillow.",
)
def ingest_cross_source_job() -> dict[str, Any]:
    """Equivalente a ``rag ingest-cross-source`` (consolidación 2026-05-04
    de 7 ingesters). Cada source tiene su TTL adentro; supervisor solo
    dispara el outer loop cada hora."""
    return _run_subprocess(
        [_RAG_BIN, "ingest-cross-source"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=1800,  # 30min — todos los sources con TTL puede tomar tiempo
    )


# ── Mood poll (opt-in via state file) ───────────────────────────────────────


@interval(
    minutes=30,
    label="mood_poll",
    description="Mood signal poller — Spotify+journal+WA+queries+calendar (opt-in).",
)
def mood_poll_job() -> dict[str, Any]:
    """Equivalente a ``python scripts/mood_poll.py`` del plist viejo.

    Gate opt-in: el script verifica ``~/.local/share/obsidian-rag/
    mood_enabled`` y bailea silently si no existe. Supervisor NO
    duplica el gate.

    F4.4 lo va a reemplazar con on-demand TTL cache via IPC desde el
    web UI; por ahora cron 30min."""
    py = _UV_TOOL_PY if _UV_TOOL_PY.exists() else _VENV_PY
    script = _REPO_ROOT / "scripts" / "mood_poll.py"
    return _run_subprocess(
        [str(py), str(script)],
        extra_env={
            "RAG_MOOD_ENABLED": "1",
            "RAG_STATE_SQL": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        timeout=300,
    )


# ── Spotify poll ────────────────────────────────────────────────────────────


@interval(
    minutes=5,
    label="spotify_poll",
    description="Spotify Now Playing → rag_spotify_log.",
)
def spotify_poll_job() -> dict[str, Any]:
    """Equivalente a ``python scripts/spotify_poll.py`` del plist viejo.

    F4.5 lo va a reemplazar con macOS NSDistributedNotificationCenter
    listener para ``com.spotify.client.PlaybackStateChanged``; por ahora
    cron 5min (down de 60s, audit 2026-05-09)."""
    py = _UV_TOOL_PY if _UV_TOOL_PY.exists() else _VENV_PY
    script = _REPO_ROOT / "scripts" / "spotify_poll.py"
    return _run_subprocess(
        [str(py), str(script)],
        extra_env={
            "RAG_STATE_SQL": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        timeout=120,
    )


# ── WhatsApp tasks extractor ────────────────────────────────────────────────


@interval(
    minutes=30,
    label="wa_tasks",
    description="WA action-item extractor — chats → 00-Inbox/WA-YYYY-MM-DD.md.",
)
def wa_tasks_job() -> dict[str, Any]:
    """Equivalente a ``rag wa-tasks`` del plist viejo. LLM-heavy:
    1 qwen2.5:3b call per chat con mensajes nuevos (cap 12 chats).
    F4.3 puede reemplazarlo con sqlite trigger del bridge SQLite."""
    return _run_subprocess(
        [_RAG_BIN, "wa-tasks"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,  # 15min — 12 chats × ~30s LLM call worst case
    )
