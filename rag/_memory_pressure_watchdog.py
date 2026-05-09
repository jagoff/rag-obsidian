"""Memory-pressure watchdog — daemon thread que descarga modelos bajo presión.

Phase 2b de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el memory-pressure watchdog desde `rag/__init__.py` (sibling del
recién extraído `rag/_wal_watchdog.py`).

## Motivación (2026-04-21 post-commit `3ef6645` + `982f9cb`)

`chat_keep_alive()` ya protege del caso "modelo grande pineado forever"
(auto-clamp a "20m" si el chat model es command-r / qwen3:30b-a3b). Pero
queda un edge case: usuario corriendo simultáneamente `rag do` + web +
otras apps pesadas satura los 36 GB unified igualmente, porque:

  (b) Helper model + embed model + chat model + reranker ocupan ~8 GB
      pinned forever; si además el usuario abre Claude Code + Chrome con
      50 tabs, wired + compressed + active puede empujar a >90% usage y
      triggerear el beachball incluso con qwen2.5:7b como default.

## Solución

Thread daemon que muestrea `vm_stat` + `sysctl hw.memsize` cada
`RAG_MEMORY_PRESSURE_INTERVAL` segundos (default 60s). Si el pct used ≥
`RAG_MEMORY_PRESSURE_THRESHOLD` (default 85%), escala:

  1. MLX chat model → MLXBackend.unload(). Libera ~4.7 GB (qwen2.5:7b).
  2. Re-chequea; si sigue alto, `maybe_unload_reranker(force=True)`.
     Libera ~300 MB de MPS.
  3. Re-chequea; si sigue alto, `maybe_unload_nli_model(force=True)`.
     Libera ~400 MB (mDeBERTa).

CLI one-shot (rag query / chat / do que terminan en <60s) NO dispara el
watchdog — no hay contención real y el overhead del thread no aporta.
Solo long-running daemons (`rag serve`, web server) lo arrancan.

## Env vars

- `RAG_MEMORY_PRESSURE_DISABLE=1`   → no arrancar watchdog (testing / CI)
- `RAG_MEMORY_PRESSURE_THRESHOLD=N` → umbral % (default 85)
- `RAG_MEMORY_PRESSURE_INTERVAL=N`  → sampling cada N segundos (default 60)
- `RAG_MEMORY_PRESSURE_SWAP_GB=N`   → trigger por swap activo (default 4.0 GB)
- `RAG_MEMORY_PRESSURE_COOLDOWN_S=N` → cooldown post-acción (default 300s)
- `RAG_MPS_CACHE_DROP_INTERVAL=N`   → loop periódico empty_cache (default 60s,
                                       auto-skip cuando backend full-MLX)
- `RAG_FORCE_MPS_EMPTY_CACHE=1`     → override force empty_cache aún bajo MLX

## Lazy imports

Este módulo depende de `_silent_log`, `_daemon_shutdown_event`,
`_ragvec_state_conn`, `_sql_append_event`, `resolve_chat_model`,
`maybe_unload_reranker` y `maybe_unload_nli_model` — todos definidos
en `rag/__init__.py`. Lazy imports adentro de las funciones evitan
circular import.

## Re-export

`rag/__init__.py` hace `from rag._memory_pressure_watchdog import *  # noqa`.
Preserva 100% compat con call sites históricos
(`rag.start_memory_pressure_watchdog()`).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

__all__ = [
    "_memory_watchdog_started",
    "_memory_watchdog_lock",
    "_system_memory_used_pct_kernel",
    "_system_memory_used_pct",
    "_system_swap_used_gb",
    "_handle_memory_pressure",
    "_memory_pressure_watchdog_loop",
    "_periodic_mps_cache_drop_loop",
    "_torch_mps_empty_cache",
    "start_memory_pressure_watchdog",
]

_memory_watchdog_started = False
_memory_watchdog_lock = threading.Lock()


def _system_memory_used_pct_kernel() -> float | None:
    """Primary probe: parsea ``memory_pressure -Q`` (kernel truth).

    Output format (estable desde macOS 11+):
        ``System-wide memory free percentage: N%``

    Returns 100 - N. None si el binario no existe / output no parsea.
    Es la signal canónica que usan beachballs / OOM killer; refleja qué
    cree el kernel sobre cuánta memoria queda reclaimable, no la fórmula
    Linux-style de páginas comprometidas (que en macOS infla por contar
    ``active`` como usado cuando es reclaimable bajo presión).
    """
    if sys.platform != "darwin":
        return None
    try:
        for mp_bin in ("/usr/bin/memory_pressure", "/sbin/memory_pressure", "memory_pressure"):
            try:
                out = subprocess.run(
                    [mp_bin, "-Q"],
                    capture_output=True, text=True, timeout=2, check=False,
                ).stdout
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            if not out:
                continue
            m = re.search(r"memory free percentage:\s*(\d+)\s*%", out)
            if m:
                free_pct = int(m.group(1))
                used_pct = 100.0 - free_pct
                return max(0.0, min(100.0, used_pct))
        return None
    except Exception:
        return None


def _system_memory_used_pct() -> float | None:
    """Return macOS system memory usage as % used, or None si no se puede medir.

    Primary: ``memory_pressure -Q`` (kernel truth — refleja la señal canónica
    que dispara beachballs / OOM). Fallback: fórmula
    ``(wired + active + compressed) / total_bytes`` via vm_stat + sysctl.

    El fallback es necesario en sandboxes / containers donde el binario
    ``memory_pressure`` no está accesible. Sobre el watchdog crónico:
    cuando ambos paths están disponibles, el primary suele dar 10-15pp
    menos que el fallback en idle (porque ``active`` cuenta como usado
    pero es reclaimable). Audit 2026-05-06: el formula path falsamente
    triggereaba unloads → MLX reload → GPU Hang en loop.

    Zero-dep: shell-outs a binarios macOS. psutil no está en
    pyproject.toml a propósito.

    Linux / otros: None — el watchdog se desactiva silencioso.
    """
    if sys.platform != "darwin":
        return None
    primary = _system_memory_used_pct_kernel()
    if primary is not None:
        return primary
    from rag import _silent_log  # noqa: PLC0415
    try:
        vm_out = ""
        total_out = ""
        for vm_bin in ("/usr/bin/vm_stat", "/bin/vm_stat", "vm_stat"):
            try:
                vm_out = subprocess.run(
                    [vm_bin], capture_output=True, text=True, timeout=2, check=False,
                ).stdout
                if vm_out:
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        for sct_bin in ("/usr/sbin/sysctl", "/sbin/sysctl", "sysctl"):
            try:
                total_out = subprocess.run(
                    [sct_bin, "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=2, check=False,
                ).stdout
                if total_out:
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        if not vm_out or not total_out:
            _silent_log(
                "system_memory_used_pct_no_output",
                Exception(f"vm_stat={bool(vm_out)} sysctl={bool(total_out)}"),
            )
            return None
        total_bytes = int(total_out.strip())
        if total_bytes <= 0:
            return None
        page_size = 4096
        stats: dict[str, float] = {}
        for line in vm_out.splitlines():
            if "page size of" in line:
                m = re.search(r"page size of (\d+)", line)
                if m:
                    page_size = int(m.group(1))
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip().rstrip(".")
            try:
                stats[key.strip()] = float(val)
            except ValueError:
                continue
        wired = stats.get("Pages wired down", 0) * page_size
        active = stats.get("Pages active", 0) * page_size
        compressed = stats.get("Pages occupied by compressor", 0) * page_size
        used = wired + active + compressed
        return (used / total_bytes) * 100.0
    except Exception:
        return None


def _system_swap_used_gb() -> float | None:
    """Return macOS swap usage in GB, or None si no se puede medir.

    Sample `sysctl vm.swapusage`:
        vm.swapusage: total = 4096.00M  used = 2907.06M  free = 1188.94M

    Early-warning del thrashing antes de que el pct comprometido llegue
    al threshold: si swap > 1-2 GB el kernel ya está paginando activo
    y cualquier cold-load de modelo (MPS) toma >>30s.

    Path absoluto a `/usr/sbin/sysctl` por si el daemon launchd tiene
    un PATH minimalista que no incluye sbin.
    """
    if sys.platform != "darwin":
        return None
    from rag import _silent_log  # noqa: PLC0415
    sysctl_paths = ("/usr/sbin/sysctl", "/sbin/sysctl", "sysctl")
    out = ""
    for p in sysctl_paths:
        try:
            out = subprocess.run(
                [p, "-n", "vm.swapusage"],
                capture_output=True, text=True, timeout=2, check=False,
            ).stdout.strip()
            if out:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception as exc:
            _silent_log("system_swap_used_gb_subprocess", exc)
            continue
    if not out:
        return None
    try:
        m = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", out, re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2).upper()
        mult = {"K": 1.0/1024/1024, "M": 1.0/1024, "G": 1.0, "T": 1024.0}.get(unit, 0.0)
        return val * mult
    except Exception as exc:
        _silent_log("system_swap_used_gb_parse", exc)
        return None


def _handle_memory_pressure(pct_before: float, threshold: float) -> dict:
    """Respuesta escalonada a un evento de memory pressure.

    Paso 1: unload del chat model vía MLX backend keep_alive=0.
    Paso 2: re-medir; si sigue ≥ threshold, force-unload del reranker.
    Paso 3: re-medir; si sigue ≥ threshold, force-unload del NLI model.

    Todos los pasos son best-effort — cualquier excepción se loggea y el
    watchdog sigue corriendo. Retorna un dict con lo que se hizo, útil
    para tests y métricas futuras.
    """
    from rag import (  # noqa: PLC0415
        _silent_log,
        maybe_unload_nli_model,
        maybe_unload_reranker,
        resolve_chat_model,
    )
    actions: dict = {
        "pct_before": round(pct_before, 2),
        "threshold": threshold,
        "chat_unloaded": False,
        "reranker_unloaded": False,
        "chat_model": None,
        "pct_after_chat": None,
        "pct_after_reranker": None,
    }

    # Paso 1: chat model
    try:
        chat_model = resolve_chat_model()
    except RuntimeError:
        chat_model = None
    actions["chat_model"] = chat_model
    if chat_model:
        # Post-Ola 7: MLX-only path. El modelo vive in-process en MPS;
        # `MLXBackend.unload()` pop-ea de `_loaded` + `mx.clear_cache()`
        # para liberar VRAM/RAM. Si el unload falla, marcamos timeout y
        # seguimos al paso 2 (reranker). El watchdog NO mata runners
        # propios (eso era responsabilidad del legado
        # manejo del backend hung).
        from rag.llm_backend import get_backend
        try:
            _backend = get_backend()
            if _backend.unload(chat_model):
                actions["chat_unloaded"] = True
            else:
                # Modelo no estaba en _loaded — no es error
                actions["chat_unload_skipped"] = "not_resident"
        except Exception as exc:
            actions["chat_unload_timeout"] = True
            _silent_log("memory_watchdog_unload_chat_mlx", exc)

    # Paso 2: re-medir y decidir reranker
    pct_after = _system_memory_used_pct()
    actions["pct_after_chat"] = round(pct_after, 2) if pct_after is not None else None
    if pct_after is not None and pct_after >= threshold:
        # Audit 2026-04-26: respetar RAG_RERANKER_NEVER_UNLOAD=1 también
        # bajo memory pressure (pre-fix, el watchdog hacía bypass total
        # del flag → contradecía la promesa documentada del operator).
        # Si está set, loguea warning + skip. El operator que confió en
        # NEVER_UNLOAD asumió que iba a tener reranker pinned siempre;
        # si el sistema realmente colapsa, el OS hará swap-or-OOM —
        # pero eso es responsabilidad del operator, no nuestra. Para
        # forzar el unload aunque NEVER_UNLOAD esté set, el operator
        # debe sacar el flag del plist.
        _never_unload = os.environ.get("RAG_RERANKER_NEVER_UNLOAD", "").strip().lower() in ("1", "true", "yes")
        if _never_unload:
            actions["reranker_unload_skipped_never_unload"] = True
            # Audit 2026-04-26 (BUG #36): wrap en Exception, no str.
            _silent_log("memory_watchdog_reranker_pinned",
                        Exception(
                            f"pct_after_chat={pct_after} >= threshold={threshold} "
                            f"but RAG_RERANKER_NEVER_UNLOAD=1 — skipping"))
        else:
            try:
                if maybe_unload_reranker(force=True):
                    actions["reranker_unloaded"] = True
                pct_after2 = _system_memory_used_pct()
                actions["pct_after_reranker"] = round(pct_after2, 2) if pct_after2 is not None else None
            except Exception as exc:
                _silent_log("memory_watchdog_unload_reranker", exc)

    # Audit 2026-04-26 (BUG #9): Paso 3 — si después de chat + reranker
    # la presión sigue arriba, descargar NLI model (mDeBERTa ~400 MB).
    # Pre-fix nunca se descargaba bajo presión → NLI quedaba pinned
    # consumiendo memoria sin servir tráfico activo.
    pct_check = _system_memory_used_pct()
    if pct_check is not None and pct_check >= threshold:
        try:
            if maybe_unload_nli_model(force=True):
                actions["nli_unloaded"] = True
                pct_after3 = _system_memory_used_pct()
                actions["pct_after_nli"] = round(pct_after3, 2) if pct_after3 is not None else None
        except Exception as exc:
            _silent_log("memory_watchdog_unload_nli", exc)
    return actions


def _memory_pressure_watchdog_loop(threshold: float, interval: int) -> None:
    """Loop del thread daemon. Corre hasta que el proceso termina.

    Trigger condicional dual:
      - `pct >= threshold`: memory comprometida (wired+active+compressed) cruzó
        el umbral porcentual.
      - `swap_gb >= RAG_MEMORY_PRESSURE_SWAP_GB` (default 1.5 GB): el kernel ya
        está paginando activo. Esta condición se chequea ANTES de que el pct
        cruce el threshold porque el thrashing arranca en cuanto hay swap I/O
        sostenido — la latencia de cualquier cold-load se dispara, y los
        timeouts del backend (60s helper, 90s chat) golpean en cadena.
    """
    from rag import _daemon_shutdown_event, _ragvec_state_conn, _silent_log, _sql_append_event  # noqa: PLC0415
    try:
        swap_gb_threshold = float(os.environ.get("RAG_MEMORY_PRESSURE_SWAP_GB", "4.0"))
    except ValueError:
        swap_gb_threshold = 4.0
    try:
        cooldown_s = float(os.environ.get("RAG_MEMORY_PRESSURE_COOLDOWN_S", "300"))
    except ValueError:
        cooldown_s = 300.0
    # Bug Hunt 2026-05-08 H5: los `print()` de este watchdog viajaban a
    # stdout sin gate por env var, contaminando CLI flows (`rag chat`)
    # y rich/click output. Gate detrás de `RAG_DEBUG=1` para que solo se
    # vea cuando el operador está debuggeando. Los disparos REALES de
    # pressure se siguen logueando (línea ~13391) sin gate — ese log
    # SÍ es accionable y poco frecuente.
    _watchdog_verbose = (
        os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes")
    )
    if _watchdog_verbose:
        print(
            f"[memory-watchdog] loop start swap_threshold={swap_gb_threshold}GB "
            f"cooldown={cooldown_s}s",
            flush=True,
        )
    _tick_count = 0
    _last_action_ts = 0.0
    while True:
        try:
            if _daemon_shutdown_event.wait(timeout=interval):
                return
            pct = _system_memory_used_pct()
            swap_gb = _system_swap_used_gb()
            _tick_count += 1
            if _watchdog_verbose and _tick_count % 4 == 1:
                # Heartbeat cada ~minuto para confirmar que el thread tickea
                print(
                    f"[memory-watchdog] tick={_tick_count} pct={pct} "
                    f"swap_gb={swap_gb}",
                    flush=True,
                )
            if pct is None and swap_gb is None:
                continue
            trigger_pct = pct is not None and pct >= threshold
            trigger_swap = (
                swap_gb is not None
                and swap_gb_threshold > 0
                and swap_gb >= swap_gb_threshold
            )
            # Cooldown: tras una acción, no volver a actuar hasta que pase
            # cooldown_s. Evita el thrash loop unload→reload→unload→...
            # cada `interval` segundos cuando el sistema está bajo presión
            # crónica (el unload libera memoria pero el siguiente request
            # vuelve a cargar el modelo, sube la pressure otra vez).
            now = time.time()
            in_cooldown = (now - _last_action_ts) < cooldown_s
            if (trigger_pct or trigger_swap) and in_cooldown:
                continue
            if trigger_pct or trigger_swap:
                _last_action_ts = now
                actions = _handle_memory_pressure(pct if pct is not None else 0.0, threshold)
                if swap_gb is not None:
                    actions["swap_gb"] = round(swap_gb, 2)
                actions["trigger_pct"] = trigger_pct
                actions["trigger_swap"] = trigger_swap
                print(
                    f"[memory-watchdog] pressure={actions['pct_before']}% "
                    f"threshold={threshold}% swap={actions.get('swap_gb')}GB "
                    f"trigger_pct={trigger_pct} trigger_swap={trigger_swap} "
                    f"chat_unloaded={actions['chat_unloaded']} "
                    f"reranker_unloaded={actions['reranker_unloaded']} "
                    f"pct_after={actions.get('pct_after_chat')}%",
                    flush=True,
                )
                # Persistir el evento de presión en system_memory_metrics para
                # forensics post-incident. Bug #3 audit 2026-04-27: la tabla
                # existía en el DDL pero el watchdog nunca escribía a ella —
                # 0 filas incluso con el watchdog activo.
                try:
                    with _ragvec_state_conn() as _conn:
                        _sql_append_event(_conn, "system_memory_metrics", {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "extra_json": actions,
                        })
                except Exception as _sql_exc:
                    try:
                        _silent_log("memory_sql_write_failed", _sql_exc)
                    except Exception:
                        pass
        except Exception as exc:
            _silent_log("memory_watchdog_loop", exc)


def _periodic_mps_cache_drop_loop(interval: int) -> None:
    """Loop daemon que periódicamente libera memoria fragmentada del MPS
    allocator de PyTorch — SIN descargar modelos.

    Diagnóstico empírico 2026-05-02 (vmmap del web/server.py PID vivo):
        owned unmapped (graphics): 8.0 GB en 12 regiones

    Esa categoría son allocations del Metal Performance Shaders backend.
    Cada `predict()` del reranker (BAAI/bge-reranker-v2-m3, 600 MB),
    cada `predict()` del NLI model (mDeBERTa, 600 MB) y cada `encode()`
    del bge-m3 local embedder (~500 MB) reservan tensors temporales en
    MPS. PyTorch los recicla pero el MPS allocator es agresivo y NO
    devuelve la memoria al sistema operativo hasta `torch.mps.empty_cache()`.

    En el codepath actual `empty_cache()` SOLO se invoca dentro de
    `maybe_unload_reranker()` y `maybe_unload_nli_model()`. Si el operador
    setea `RAG_RERANKER_NEVER_UNLOAD=1` (default en el web server según
    el plist), el reranker se queda pinned, los modelos siguen vivos pero
    el MPS allocator acumula memoria fragmentada con cada query → 8 GB
    en pocos minutos → unified memory presure → swap activo → backend no
    consigue VRAM para cargar modelos → drafts del listener fallan en
    cascada con "❌ no pude generar draft (LLM no respondió o timeout)".

    `torch.mps.empty_cache()` libera la memoria pre-allocada NO usada por
    los tensors live. Es safe llamarlo en cualquier momento — los modelos
    cargados (weights, KV cache, buffers de inferencia activa) NO se
    afectan. Solo se devuelve la memoria que ya estaba "free internamente"
    pero retenida por el allocator.

    Cadencia: 60s default. Más frecuente no aporta (la fragmentación
    crece linealmente con el tráfico, no exponencial), menos frecuente
    deja crecer mucho el bloat entre flushes.

    Override: `RAG_MPS_CACHE_DROP_INTERVAL=0` apaga el daemon. Cualquier
    valor > 0 lo activa con ese intervalo en segundos.
    """
    from rag import _daemon_shutdown_event, _silent_log  # noqa: PLC0415
    while True:
        try:
            if _daemon_shutdown_event.wait(timeout=interval):
                return
            _torch_mps_empty_cache()
            # gc.collect() ayuda a liberar tensors orphan (referencias
            # circulares en buffers de Python) que MPS no tocaría hasta
            # que Python los recolecte.
            import gc as _gc
            _gc.collect()
        except Exception as exc:
            _silent_log("mps_cache_drop_loop", exc)


def _torch_mps_empty_cache() -> None:
    """Best-effort MPS cache drop. No-op si torch/MPS no está disponible
    (Intel mac o ambiente sin MPS).

    MLX-aware (2026-05-08): cuando AMBOS backends son MLX (post-Ola 9 default),
    `torch.mps.empty_cache()` invalida los Metal command buffers de MLX en
    ejecución → GPU Hang Error reproducible determinísticamente. En ese caso
    es no-op — MLX maneja su propia cache (`mx.clear_cache()`).

    Override manual: `RAG_FORCE_MPS_EMPTY_CACHE=1` (rollback al comportamiento
    histórico). Solo necesario si el operador todavía tiene un consumidor
    PyTorch/MPS pinned (ej. reranker bge en MPS) y necesita drop explícito.
    """
    embed_backend = os.environ.get("RAG_EMBED_BACKEND", "mlx").strip().lower()
    llm_backend = os.environ.get("RAG_LLM_BACKEND", "mlx").strip().lower()
    force = os.environ.get("RAG_FORCE_MPS_EMPTY_CACHE", "").strip().lower() in ("1", "true", "yes")
    if (embed_backend == "mlx" and llm_backend == "mlx") and not force:
        return
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def start_memory_pressure_watchdog() -> bool:
    """Arrancar el watchdog como daemon thread. Idempotente.

    Retorna True si se arrancó (o ya estaba arrancado), False si se
    skippeó por `RAG_MEMORY_PRESSURE_DISABLE=1` o por plataforma no-darwin.

    Llamar desde el startup de long-running processes: `rag serve`, web
    server FastAPI startup, ambient agent daemon. El CLI one-shot NO
    necesita el watchdog (el proceso termina antes del primer tick).

    Además del watchdog de presión, arranca un loop periódico de
    `torch.mps.empty_cache()` (cada 60s default) que libera la memoria
    fragmentada del MPS allocator sin descargar modelos. Ver doc-block
    en `_periodic_mps_cache_drop_loop` para el rationale completo —
    confirmamos empíricamente con vmmap que el web/server.py llegaba
    a 8 GB de "owned unmapped (graphics)" porque ningún codepath
    invocaba empty_cache() salvo al unload, y el reranker está pinned.
    """
    global _memory_watchdog_started
    with _memory_watchdog_lock:
        if _memory_watchdog_started:
            return True
        if os.environ.get("RAG_MEMORY_PRESSURE_DISABLE") == "1":
            return False
        if sys.platform != "darwin":
            # `_system_memory_used_pct` devuelve None en no-darwin — sin
            # métrica, el watchdog es un no-op que gasta un thread.
            return False
        try:
            threshold = float(os.environ.get("RAG_MEMORY_PRESSURE_THRESHOLD", "85"))
        except ValueError:
            threshold = 85.0
        try:
            interval = int(os.environ.get("RAG_MEMORY_PRESSURE_INTERVAL", "60"))
        except ValueError:
            interval = 60
        t = threading.Thread(
            target=_memory_pressure_watchdog_loop,
            args=(threshold, interval),
            name="rag-memory-watchdog",
            daemon=True,
        )
        t.start()
        # Lanzar también el daemon periódico de MPS cache drop. Independiente
        # del threshold de memory pressure — corre siempre, libera memoria
        # fragmentada cada N segundos. Si el operador desactiva el watchdog
        # de pressure (RAG_MEMORY_PRESSURE_DISABLE=1) el cache drop también
        # se desactiva — son la misma defensa de memoria.
        #
        # MLX-aware (2026-05-08): cuando el embedder corre en MLX (post-Ola 9
        # default), `torch.mps.empty_cache()` corriendo en thread paralelo
        # invalida los Metal command buffers de MLX → GPU Hang Error en
        # `rag index` (`kIOGPUCommandBufferCallbackErrorHang/InnocentVictim`).
        # Reproducido determinísticamente 2026-05-08 con 6 retries idénticos.
        # Cuando ambos backends son MLX, skipeamos el daemon — MLX maneja su
        # propia cache (`mx.clear_cache()`). Solo activamos cuando todavía
        # hay un consumidor PyTorch/MPS pinned (ej. reranker bge en MPS).
        embed_backend = os.environ.get("RAG_EMBED_BACKEND", "mlx").strip().lower()
        llm_backend = os.environ.get("RAG_LLM_BACKEND", "mlx").strip().lower()
        mlx_only = (embed_backend == "mlx" and llm_backend == "mlx")
        try:
            mps_interval = int(os.environ.get("RAG_MPS_CACHE_DROP_INTERVAL", "60"))
        except ValueError:
            mps_interval = 60
        if mlx_only and "RAG_MPS_CACHE_DROP_INTERVAL" not in os.environ:
            # Auto-skip cuando backend full-MLX y el operador no forzó override.
            mps_interval = 0
            print(
                "[mps-cache-drop] skipped (RAG_{EMBED,LLM}_BACKEND=mlx) — "
                "evita conflict con Metal command buffers de MLX",
                flush=True,
            )
        if mps_interval > 0:
            t_mps = threading.Thread(
                target=_periodic_mps_cache_drop_loop,
                args=(mps_interval,),
                name="rag-mps-cache-drop",
                daemon=True,
            )
            t_mps.start()
            print(
                f"[mps-cache-drop] started interval={mps_interval}s",
                flush=True,
            )
        _memory_watchdog_started = True
        print(
            f"[memory-watchdog] started threshold={threshold}% interval={interval}s",
            flush=True,
        )
        return True
