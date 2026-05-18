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
import os
from pathlib import Path
import signal
import shlex
import subprocess
import threading
import time
from typing import Any

from rag.runtime.jobs.nightly import _RAG_BIN, _run_subprocess
from rag.runtime.scheduler import interval

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VENV_PY = _REPO_ROOT / ".venv" / "bin" / "python"
_UV_TOOL_PY = Path.home() / ".local/share/uv/tools/obsidian-rag/bin/python3"
_ANTICIPATE_PROCESS_NEEDLE = "rag anticipate run"
_INDEX_LOCK_BUSY_MARKER = "Otro `rag index` ya está activo"
_WA_TASKS_JOB_LOCK = threading.Lock()
_TRUTHY = {"1", "true", "yes", "on"}


def _poller_python() -> Path:
    """Python for lightweight poller scripts.

    Prefer the repo venv. The uv-tool interpreter can exist on disk while
    being unusable after a Python upgrade (`No module named encodings`), which
    silently breaks mood/spotify subprocess jobs under the supervisor.
    """
    return _VENV_PY if _VENV_PY.exists() else _UV_TOOL_PY


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _swap_pressure_active(
    memory_pct: float | None,
    swap_gb: float | None,
    max_swap_gb: float,
    *,
    floor_env: str = "RAG_ANTICIPATE_SWAP_MEMORY_FLOOR_PCT",
    floor_default: float = 70.0,
) -> bool:
    """Treat swap as pressure only when current memory pressure agrees.

    macOS can keep swap files allocated long after pressure cleared. The
    supervisor should not suppress anticipate forever on stale swap alone.
    """
    if swap_gb is None or max_swap_gb <= 0 or swap_gb < max_swap_gb:
        return False
    if memory_pct is None:
        return True
    min_memory_pct = _env_float(floor_env, floor_default)
    return memory_pct >= min_memory_pct


def _skip_result(reason: str, **signals: Any) -> dict[str, Any]:
    return {
        "exit_code": 0,
        "stdout_lines": 0,
        "stderr_lines": 0,
        "last_stderr": None,
        "skipped": True,
        "skip_reason": reason,
        **signals,
    }


def _runtime_pressure_snapshot() -> tuple[float | None, float | None]:
    try:
        from rag._memory_pressure_watchdog import (  # noqa: PLC0415
            _system_memory_used_pct,
            _system_swap_used_gb,
        )
        return _system_memory_used_pct(), _system_swap_used_gb()
    except Exception as exc:  # noqa: BLE001 — telemetry guard must be silent
        logger.warning("anticipate pressure check failed: %s", exc)
        return None, None


def _command_is_anticipate_run(cmd: str) -> bool:
    """Match only real ``rag anticipate run`` processes.

    ``ps`` returns a flattened command string, so a broad substring check can
    match unrelated workers whose prompt/log text mentions the command.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if len(parts) < 3:
        return False

    def _name(idx: int) -> str:
        return Path(parts[idx]).name

    if _name(0) == "rag":
        return parts[1:3] == ["anticipate", "run"]
    if _name(0).startswith("python") and len(parts) >= 4 and _name(1) == "rag":
        return parts[2:4] == ["anticipate", "run"]
    if _name(0).startswith("python") and len(parts) >= 5 and parts[1:3] == ["-m", "rag"]:
        return parts[3:5] == ["anticipate", "run"]
    return False


def _command_matches_needle(cmd: str, needle: str) -> bool:
    if needle == _ANTICIPATE_PROCESS_NEEDLE:
        return _command_is_anticipate_run(cmd)
    return needle in cmd


def _running_process_count(needle: str) -> int:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("process scan failed for %s: %s", needle, exc)
        return 0
    if result.returncode != 0:
        return 0
    current_pid = os.getpid()
    count = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if pid == current_pid:
            continue
        if " rg " in f" {cmd} " or " grep " in f" {cmd} ":
            continue
        if _command_matches_needle(cmd, needle):
            count += 1
    return count


def _anticipate_preflight_skip() -> dict[str, Any] | None:
    if _env_flag("RAG_ANTICIPATE_DISABLED"):
        return _skip_result("disabled")

    running = _running_process_count(_ANTICIPATE_PROCESS_NEEDLE)
    if running > 0:
        logger.warning("skip anticipate: %d existing run(s) still active", running)
        return _skip_result("already_running", running_instances=running)

    if os.environ.get("RAG_ANTICIPATE_PRESSURE_GUARD", "1").strip() == "0":
        return None

    memory_pct, swap_gb = _runtime_pressure_snapshot()
    max_memory_pct = _env_float("RAG_ANTICIPATE_MAX_MEMORY_PCT", 70.0)
    max_swap_gb = _env_float("RAG_ANTICIPATE_MAX_SWAP_GB", 1.5)

    if memory_pct is not None and memory_pct >= max_memory_pct:
        logger.warning(
            "skip anticipate: memory pressure %.1f%% >= %.1f%%",
            memory_pct,
            max_memory_pct,
        )
        return _skip_result(
            "memory_pressure",
            memory_pct=memory_pct,
            max_memory_pct=max_memory_pct,
            swap_gb=swap_gb,
        )
    if _swap_pressure_active(memory_pct, swap_gb, max_swap_gb):
        min_memory_pct = _env_float("RAG_ANTICIPATE_SWAP_MEMORY_FLOOR_PCT", 70.0)
        logger.debug(
            "skip anticipate: swap pressure %.2fGB >= %.2fGB "
            "(memory %.1f%% >= %.1f%%)",
            swap_gb,
            max_swap_gb,
            memory_pct if memory_pct is not None else -1.0,
            min_memory_pct,
        )
        return _skip_result(
            "swap_pressure",
            memory_pct=memory_pct,
            swap_gb=swap_gb,
            max_swap_gb=max_swap_gb,
            min_memory_pct_for_swap=min_memory_pct,
        )
    return None


def _process_tree_pids(root_pid: int) -> list[int]:
    seen: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
        except Exception:
            continue
        for raw in result.stdout.split():
            try:
                child_pid = int(raw)
            except ValueError:
                continue
            if child_pid not in seen:
                pending.append(child_pid)
    return sorted(seen)


def _process_tree_rss_gb(root_pid: int) -> float | None:
    pids = _process_tree_pids(root_pid)
    if not pids:
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    total_kb = 0
    for raw in result.stdout.split():
        try:
            total_kb += int(raw)
        except ValueError:
            continue
    if total_kb <= 0:
        return None
    return total_kb / 1024.0 / 1024.0


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to terminate process group %s: %s", proc.pid, exc)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to kill process group %s: %s", proc.pid, exc)


def _run_guarded_subprocess(
    args: list[str],
    *,
    timeout: int,
    extra_env: dict[str, str] | None = None,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    started = time.monotonic()
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    abort_reason: str | None = None
    abort_kind: str | None = None
    exit_code: int | None = None
    stdout = ""
    stderr = ""

    while True:
        elapsed = time.monotonic() - started
        remaining = max(0.1, timeout - elapsed)
        try:
            stdout, stderr = proc.communicate(timeout=min(poll_interval, remaining))
            exit_code = proc.returncode
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            rss_gb = _process_tree_rss_gb(proc.pid)
            memory_pct, swap_gb = _runtime_pressure_snapshot()
            max_rss_gb = _env_float("RAG_ANTICIPATE_MAX_RSS_GB", 10.0)
            abort_memory_pct = _env_float("RAG_ANTICIPATE_ABORT_MEMORY_PCT", 82.0)
            abort_swap_gb = _env_float("RAG_ANTICIPATE_ABORT_SWAP_GB", 2.0)
            if elapsed >= timeout:
                abort_reason = f"timeout after {timeout}s"
                abort_kind = "timeout"
                exit_code = -1
            elif rss_gb is not None and rss_gb >= max_rss_gb:
                abort_reason = f"rss {rss_gb:.2f}GB >= {max_rss_gb:.2f}GB"
                abort_kind = "rss_pressure"
                exit_code = -9
            elif _swap_pressure_active(
                memory_pct,
                swap_gb,
                abort_swap_gb,
                floor_env="RAG_ANTICIPATE_ABORT_SWAP_MEMORY_FLOOR_PCT",
                floor_default=70.0,
            ):
                abort_reason = f"swap {swap_gb:.2f}GB >= {abort_swap_gb:.2f}GB"
                abort_kind = "swap_pressure"
                exit_code = -9
            elif memory_pct is not None and memory_pct >= abort_memory_pct:
                abort_reason = (
                    f"memory {memory_pct:.1f}% >= {abort_memory_pct:.1f}%"
                )
                abort_kind = "memory_pressure"
                exit_code = -9

            if abort_reason is None:
                continue

            pressure_skip = abort_kind in {"memory_pressure", "swap_pressure"}
            if pressure_skip:
                logger.info("skip guarded job %s: %s", " ".join(args), abort_reason)
            else:
                logger.warning("abort guarded job %s: %s", " ".join(args), abort_reason)
            _terminate_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                stdout, stderr = proc.communicate()
            break

    stdout_lines = stdout.count("\n") if stdout else 0
    stderr_lines = stderr.count("\n") if stderr else 0
    pressure_skip = abort_kind in {"memory_pressure", "swap_pressure"}
    reported_exit_code = 0 if pressure_skip else int(exit_code or 0)
    last_err = (stderr or "")[-200:] if reported_exit_code != 0 else None
    if abort_reason and not pressure_skip:
        last_err = f"{abort_reason}; stderr={last_err or ''}"[:200]
    if reported_exit_code != 0:
        logger.warning(
            "guarded job exit=%d: %s — stderr tail: %s",
            reported_exit_code,
            " ".join(args),
            last_err,
        )
    payload = {
        "exit_code": reported_exit_code,
        "stdout_lines": stdout_lines,
        "stderr_lines": stderr_lines,
        "last_stderr": last_err,
        "guarded": True,
        "killed_reason": abort_reason,
    }
    if pressure_skip:
        payload.update({
            "skipped": True,
            "skip_reason": abort_kind,
            "raw_exit_code": int(exit_code or 0),
            "pressure_reason": abort_reason,
        })
    return payload


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
    skipped = _anticipate_preflight_skip()
    if skipped is not None:
        return skipped

    timeout_s = int(_env_float("RAG_ANTICIPATE_TIMEOUT_S", 180.0))
    return _run_guarded_subprocess(
        [_RAG_BIN, "anticipate", "run"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "RAG_ANTICIPATE_PRESSURE_GUARD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=timeout_s,  # anticipate típicamente <30s; cap duro por OOM
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
            "RAG_INDEX_LOCK_WAIT_SECONDS": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        benign_failure_markers=(_INDEX_LOCK_BUSY_MARKER,),
        benign_skip_reason="index_lock_busy",
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
            "RAG_INDEX_LOCK_WAIT_SECONDS": "0",
            "RAG_SAFARI_BOOKMARK_LOCK_BUDGET_S": "60",
            "RAG_SAFARI_BOOKMARK_MAX_WRITE": "250",
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
    py = _poller_python()
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
    py = _poller_python()
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
    if not _WA_TASKS_JOB_LOCK.acquire(blocking=False):
        logger.info("skip wa-tasks: previous extractor still running")
        return _skip_result("already_running")
    try:
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
    finally:
        _WA_TASKS_JOB_LOCK.release()


# ── Peekaboo screen observer (Fase 2c, 2026-05-13) ─────────────────────────


@interval(
    minutes=15,
    label="screen_observer",
    description="Peekaboo screen capture + caption granite → rag_screen_observations (opt-in via RAG_SCREEN_OBSERVE).",
)
def screen_observer_job() -> dict[str, Any]:
    """Tick in-process del observer Peekaboo cada 15min.

    **Doble opt-in** (ambos en env del supervisor o shell del user):
        - ``RAG_PEEKABOO_ENABLE=1`` — binario activado.
        - ``RAG_SCREEN_OBSERVE=1`` — daemon activado.

    Si cualquiera falta, observe_once retorna `skipped_reason:
    observe_disabled` o `peekaboo_disabled` y el job termina en
    <10ms (sin tocar Peekaboo CLI ni granite).

    ## In-process (Fase 2g optimization, 2026-05-13)

    Llamada directa a `observe_once()` — NO subprocess. Razón:
    cada subprocess fresh re-importaba `rag` + re-cargaba granite
    MLX-VLM (~14s cold load per tick). In-process reusa el granite
    YA warm en el supervisor (cargado por el primer tick + idle TTL
    > 15min mantiene el modelo residente).

    Trade-off:
        - Win: -14s wall-time por tick (5s warm vs 19s cold).
        - Risk: si granite crashea (Metal cmd buffer hang, OOM),
          mata el supervisor entero. Mitigación: launchd
          `KeepAlive=true` respawna; `try/except Exception` envuelve
          la call para que cualquier excepción del VLM se reporte
          como `exit_code != 0` sin propagar.

    ## Mode: screen (Fase 2g, 2026-05-13)

    Cambiamos default de `frontmost` → `screen` (display completo).
    Razón: `--mode frontmost` con Ghostty (GPU-rendered terminal)
    devuelve PNG 500x500 mayormente blanca — el bounding box de la
    ventana no captura su pixel content via CGWindowList APIs.
    `--mode screen` captura el display real (~2048x858) con todas
    las ventanas visibles. Costo: PNG más grande (~400KB vs 7KB),
    granite caption ~30s vs ~5s warm. Tradeoff aceptable porque
    sin ese fix las captures son inútiles para granite.

    ## Retention

    7d via `run_maintenance` (housekeeping daily a las 04:00).
    """
    import time  # noqa: PLC0415
    started = time.time()
    try:
        from rag.integrations.peekaboo import observe_once  # noqa: PLC0415
        out = observe_once(mode="screen")
        wall_ms = int((time.time() - started) * 1000)
        ok = bool(out.get("ok"))
        reason = out.get("skipped_reason")
        err = out.get("error")
        # Mapeo al shape esperado por `rag_supervisor_jobs.signals`:
        # - skipped_reason ≠ None → exit_code=0 (skip es operación normal).
        # - error ≠ None → exit_code=1 + last_stderr.
        # - ok=True → exit_code=0.
        if err and not ok:
            return {
                "exit_code": 1,
                "stdout_lines": 1,
                "stderr_lines": 1,
                "last_stderr": str(err)[:200],
                "duration_ms": wall_ms,
                "observation_id": out.get("observation_id"),
            }
        return {
            "exit_code": 0,
            "stdout_lines": 1,
            "stderr_lines": 0,
            "last_stderr": None,
            "duration_ms": wall_ms,
            "observation_id": out.get("observation_id"),
            "skipped_reason": reason,
        }
    except Exception as exc:  # noqa: BLE001 — observer error NO debe matar supervisor
        logger.exception("screen_observer_job crashed: %s", exc)
        return {
            "exit_code": 1,
            "stdout_lines": 0,
            "stderr_lines": 1,
            "last_stderr": f"{type(exc).__name__}: {exc}"[:200],
            "duration_ms": int((time.time() - started) * 1000),
        }
