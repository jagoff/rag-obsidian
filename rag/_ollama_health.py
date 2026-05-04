"""Latency degradation watchdog para ollama daemon.

Bug detectado en eval 2026-04-28: el daemon ollama se atasca después
de 5-7 queries tool-heavy consecutivas (KV cache fragmentation o
internal wedge). El restart manual (`brew services restart ollama`)
recupera la performance. Este módulo automatiza la detección + restart.

API pública (importable desde rag.__init__ via lazy import):
- `start_latency_degradation_watchdog(...)` — arranca el thread daemon.
- `_latency_degradation_check()` — single-shot check (testeable).
- `_percentile(vals, p)` — utility.

ENV VARS para tunear:
- RAG_LATENCY_WATCHDOG_DISABLE=1 → no arrancar el watchdog
- RAG_LATENCY_WATCHDOG_INTERVAL=30 (segundos entre checks)
- RAG_LATENCY_WATCHDOG_WINDOW_MIN=10 (minutos de ventana reciente)
- RAG_LATENCY_WATCHDOG_THRESHOLD=3.0 (ratio recent/baseline para trigger;
  3.0 reemplazó 1.8 el 2026-05-01 — ver Trade-off abajo)
- RAG_LATENCY_WATCHDOG_MIN_RECENT=5 (cantidad mínima de queries en ventana para evaluar — evita false positives con poca data)
- RAG_LATENCY_WATCHDOG_COOLDOWN=1800 (segundos entre restarts permitidos — evita
  thrashing; 1800s=30min reemplazó 300s el 2026-05-01 — ver Trade-off abajo)
- RAG_LATENCY_WATCHDOG_HEALTH_TIMEOUT=5 (segundos para el health-check
  directo a `/api/generate` durante la escalación — ver Fix #1 abajo)
- RAG_LATENCY_WATCHDOG_HEALTH_MODEL=qwen2.5:7b (modelo usado para el
  health-check; debe estar pulled localmente)

Fix #1 escalación con bypass de cooldown (2026-05-01):
  Antes: cuando el watchdog detectaba degradación sostenida pero quedaba
  bloqueado por cooldown_seconds (default 30min), la única escalación era
  `_kill_stuck_rag_index_clients()`. Si NO había `rag index` corriendo
  (caso típico hoy — el indexer es event-driven, no daemon), la
  escalación era no-op (`killed=0 pids=[]`) y el sistema quedaba
  degradado los 30 minutos completos. El watchdog externo
  `com.fer.obsidian-rag-ollama-watchdog` (generate-health-based,
  separado) eventualmente recuperaba pero tardaba ~5 min adicionales.

  Síntoma observado 2026-05-01 ~18:46: user mandó audio a RagNet, el
  listener TS quedó bloqueado en `/query` (que necesita embed bge-m3 de
  Ollama), Ollama estaba wedged, watchdog interno spam-loggeaba
  `restart_skipped_cooldown ratio=5.22` y `escalated_killed_rag_index
  killed=0 pids=[]` durante 12 minutos hasta que el watchdog externo
  finalmente bouncea los runners.

  Fix: durante la escalación, el watchdog ahora pega un health-check
  directo a `POST /api/generate` con `num_predict=1` + timeout 5s. Si
  el endpoint NO responde (HTTP 000 / timeout / 5xx), Ollama está
  efectivamente roto y el cooldown está protegiendo un restart anterior
  que NO funcionó. En ese caso, la escalación BYPASEA el cooldown y
  fuerza `_restart_ollama_daemon()` directo. Si el health-check sí
  responde, la escalación cae al comportamiento legacy (kill rag_index
  zombies si los hay) — la asunción es "Ollama responde lento pero
  responde, no es wedge real, es saturación por consumer pegado".

  Tunable: `RAG_LATENCY_WATCHDOG_HEALTH_TIMEOUT` (default 5s) y
  `RAG_LATENCY_WATCHDOG_HEALTH_MODEL` (default qwen2.5:7b). Si el modelo
  default no está pulled, el health-check loggea el detalle pero no
  rompe — se cae al comportamiento legacy.

Trade-off threshold + cooldown (2026-05-01):
  El watchdog usa `pkill -9 -f ollama` que mata el daemon + sus runners,
  cerrando cualquier conexión TCP en vuelo. Si el web server está
  streamando una respuesta /api/chat al user en ese instante, el cliente
  ve "Server disconnected without sending a response" y la query muere.

  El loop autodestructivo observado 2026-05-01:
    1. Watchdog detecta p95_recent / p95_baseline >= 1.8x → restart.
    2. pkill mata ollama. Web server tiene 1+ requests en vuelo → mueren.
    3. Cold-load del modelo qwen2.5:7b post-restart toma 8-15s. La query
       que el user reintenta paga ese cold-load → total_ms ~100s.
    4. Esa query queda en `rag_queries.extra_json.total_ms`. Sube p95.
    5. 5 minutos después (cooldown=300s), watchdog corre check. p95 sigue
       arriba (queries lentas post-restart en window). Ratio >= 1.8 →
       restart OTRA VEZ.
    6. Goto 2.

  Confirmado en logs:
    [ollama-health-watchdog] restart_attempted: p95_recent=106503ms
      p95_baseline=52134ms ratio=2.04 reason=restart_ok=True
    [ollama-health-watchdog] restart_skipped_cooldown: ...
    [ollama-health-watchdog] restart_attempted: p95_recent=...

  Defaults nuevos (3.0x + 1800s):
    - 3.0x es threshold honesto para "ollama realmente roto". 1.8x es
      ruido de cold-load + cualquier query con tool-decide o reformulator
      que paga 30-60s extra.
    - 30 min de cooldown da tiempo suficiente para que las queries
      lentas post-restart salgan del window de 10 min y el p95 se
      estabilice antes del próximo check.

  Si el restart legítimamente NO funcionó (degradación real persistente
  como saturación de unified memory), el watchdog va a re-trigger 30 min
  después. Eso es correcto — un retry rápido (5 min) no agrega valor
  porque el cold-load apenas terminó y el sistema ya está warm de nuevo.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


_started = False
_start_lock = threading.Lock()
_last_restart_ts: float = 0.0
_last_restart_lock = threading.Lock()

# Contador de checks consecutivos donde el watchdog detectó degradación
# (ratio >= threshold) pero NO pudo restartear porque los gates bloquearon
# (in_flight chats, cooldown). Se resetea cuando un check vuelve al estado
# `ok`. Cuando llega a `ESCALATE_KILL_AFTER_SKIPS` (default 10 → ~5 min),
# el watchdog escala a un kill heurístico de `rag index` clients que estén
# ESTABLISHED contra :11434 — la asunción es que un `rag index` colgado en
# `sock_recv` por >5min mientras el chat está degradado SOSTENIDO es la
# causa raíz, no un index legítimamente progresando.
#
# Por qué este escalado existe: 2026-05-01, una sesión devin paralela
# dejó un `rag index --no-contradict` en bash retry-loop que se quedó en
# `__recvfrom` 1.5h. El gate `in_flight_guard` (correctamente diseñado
# para no cortar streams /api/chat) impidió cualquier restart de ollama,
# y el chat del user quedó pegado eternamente. Con el Fix #1 (timeout 120s
# en `embed()`) ese caso ya no debería existir, pero este escalado es
# defensa de último recurso por si aparece otro consumer de Ollama no
# protegido.
_consecutive_degraded_skips = 0
_consecutive_skips_lock = threading.Lock()
ESCALATE_KILL_AFTER_SKIPS = int(
    os.environ.get("RAG_LATENCY_WATCHDOG_ESCALATE_AFTER", "10")
)

# Persistencia de `_last_restart_ts` en disco (2026-05-01).
#
# Bug observado: cuando el web server reinicia (manual, plist reload, crash),
# el módulo se reimporta y `_last_restart_ts = 0.0`. El primer check del
# watchdog post-boot ve `now - 0 = enorme >> cooldown` → restart inmediato
# de ollama. Si las queries lentas que dispararon p95 alto siguen en la
# ventana de 10min (típico hasta 10min después del último outlier), el
# watchdog restartea ollama JUSTO mientras el user está mandando una query
# nueva post-boot → "synthesis falló: Server disconnected".
#
# Loop reproducido en el log del 2026-05-01 ~15:14 (user: "podes recomendarme
# series respecto a las que yo tengo rankeadas en mis notas?"):
#   1. Web server arranca a las T0.
#   2. p95_recent (10min window) seguía con la query de 287s de hace 5min.
#   3. Watchdog ve ratio=5.51 ≥ 3.0, cooldown_ts=0 → 1er restart a T0+30s.
#   4. User manda chat a T0+45s → ollama está cold-loading post-restart.
#   5. Server cierra conexión a los ~7-8s mientras prefilea → user ve error.
#
# Fix: persistir `_last_restart_ts` en JSON simple. Cargar al `start_*()`
# para que el primer check post-boot conozca el timestamp real del último
# restart. Escribir cada vez que se restartea ollama. Silent-fail en I/O —
# no es crítico, solo perf de cold-start.
_STATE_FILE = Path.home() / ".local/share/obsidian-rag" / "ollama_health_state.json"


def _load_persisted_restart_ts() -> float:
    """Lee `_last_restart_ts` del archivo de estado. Retorna 0.0 si no
    existe o falla — es seguro defaultear porque eso replica el
    comportamiento pre-fix."""
    try:
        with _STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("last_restart_ts")
        if isinstance(ts, (int, float)) and ts > 0:
            return float(ts)
    except Exception:
        pass
    return 0.0


def _persist_restart_ts(ts: float) -> None:
    """Escribe `_last_restart_ts` al archivo de estado. Silent-fail —
    no es crítico para la operación, solo para que el state sobreviva
    restarts del proceso."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via temp + rename — evita JSON corrupto si
        # el proceso muere mid-write.
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"last_restart_ts": float(ts)}, f)
        tmp.replace(_STATE_FILE)
    except Exception:
        pass

# In-flight chat counter — el web server llama begin_chat()/end_chat()
# alrededor de cada /api/chat handler. El watchdog NO restartea ollama
# mientras haya >=1 request activa, porque pkill -9 cierra la conexión
# TCP del runner y el cliente ve "Server disconnected without sending a
# response" — exactamente el bug que el watchdog dice arreglar.
# Observado 2026-05-01: query del user "podes recomendarme series"
# disparó retrieve OK con conf=0.385 sobre `Best Series all times.md`,
# pero el watchdog mató ollama mid-synthesis (`phase=synthesis
# exc=Server disconnected ttft_ms=8371`) → user vio "synthesis falló".
_in_flight_count = 0
_in_flight_lock = threading.Lock()


def begin_chat() -> None:
    """Marca que una request /api/chat empezó. Idempotent reentry-safe.

    Llamar al inicio del handler, en pareja con `end_chat()` en finally.
    El watchdog usa este counter para evitar restartar ollama mid-stream.
    """
    global _in_flight_count
    with _in_flight_lock:
        _in_flight_count += 1


def end_chat() -> None:
    """Marca que una request /api/chat terminó (éxito o error)."""
    global _in_flight_count
    with _in_flight_lock:
        _in_flight_count = max(0, _in_flight_count - 1)


def in_flight_chats() -> int:
    """Cantidad de chat handlers actualmente activos (testeable)."""
    with _in_flight_lock:
        return _in_flight_count


def _percentile(vals: list[float], p: float) -> float | None:
    """Calcula percentile p (0-100) via nearest-rank en vals ordenados."""
    if not vals:
        return None
    sorted_vals = sorted(float(v) for v in vals if v is not None)
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100.0) - 1))
    return float(sorted_vals[idx])


def _read_recent_query_latencies(window_minutes: int) -> tuple[list[float], list[float]]:
    """Read p95-relevant query latencies from `rag_queries` SQLite.

    Returns `(recent_vals, baseline_vals)` donde recent es la ventana
    de los últimos `window_minutes` y baseline es 7 días.

    Si la tabla no existe o está vacía, retorna `([], [])`.
    """
    try:
        # Lazy import — no querer ciclo con rag/__init__.py
        from rag import _ragvec_state_conn  # type: ignore
    except Exception:
        return [], []
    try:
        with _ragvec_state_conn() as conn:
            recent_rows = conn.execute(
                """
                SELECT CAST(json_extract(extra_json, '$.total_ms') AS INTEGER)
                FROM rag_queries
                WHERE cmd LIKE 'web%'
                  AND ts >= datetime('now', ?)
                  AND json_extract(extra_json, '$.total_ms') IS NOT NULL
                """,
                (f"-{int(window_minutes)} minutes",),
            ).fetchall()
            baseline_rows = conn.execute(
                """
                SELECT CAST(json_extract(extra_json, '$.total_ms') AS INTEGER)
                FROM rag_queries
                WHERE cmd LIKE 'web%'
                  AND ts >= datetime('now', '-7 days')
                  AND json_extract(extra_json, '$.total_ms') IS NOT NULL
                """
            ).fetchall()
    except Exception:
        return [], []
    recent = [r[0] for r in recent_rows if r[0] is not None]
    baseline = [r[0] for r in baseline_rows if r[0] is not None]
    return recent, baseline


def _list_ollama_clients() -> list[tuple[int, str]]:
    """Best-effort: lista (PID, command) de procesos con conexión TCP
    abierta a `localhost:11434`, EXCLUYENDO el daemon ollama.

    Implementación: `lsof -i :11434 -P -F pcn` parseado a mano (no hay
    libs estándar para esto sin dependencias extra). Si lsof falla por
    cualquier razón (no instalado, sandbox, permisos), devuelve `[]` —
    el caller trata `[]` como "no hay zombies que matar".

    Returns: lista de tuplas `(pid, full_command)`. PIDs únicos. El
    comando es el output de `ps -p <pid> -o command=` (sin truncar).
    """
    try:
        result = subprocess.run(
            ["lsof", "-i", ":11434", "-P", "-F", "pc"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    pids: set[int] = set()
    cmd_by_pid: dict[int, str] = {}
    current_pid: int | None = None
    for line in result.stdout.splitlines():
        if not line:
            continue
        tag, rest = line[0], line[1:]
        if tag == "p":
            try:
                current_pid = int(rest)
            except ValueError:
                current_pid = None
        elif tag == "c" and current_pid is not None:
            cmd = rest.strip()
            # Skip the ollama daemon + ollama runners themselves.
            if "ollama" in cmd.lower():
                continue
            pids.add(current_pid)
            cmd_by_pid[current_pid] = cmd
    # Get full command for each PID (lsof's `c` field truncates at 9 chars
    # which is useless for "rag index --foo" detection).
    out: list[tuple[int, str]] = []
    for pid in pids:
        try:
            ps = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except Exception:
            continue
        if ps.returncode != 0:
            continue
        full_cmd = ps.stdout.strip()
        if full_cmd:
            out.append((pid, full_cmd))
    return out


def _kill_stuck_rag_index_clients() -> tuple[int, list[int]]:
    """Best-effort: matar `rag index` clients que estén con conexión
    abierta a Ollama mientras el watchdog detectó degradación sostenida.

    Heurística: si después de >5min de degradación (`ESCALATE_KILL_AFTER_SKIPS`
    checks consecutivos en estado degraded sin poder restartear), hay un
    proceso cuyo full command incluye `rag index` Y tiene conexión a
    `:11434`, asumimos que es un zombie en `sock_recv` (Fix #1 hace que
    los `rag index` legítimos terminen en <120s por request, así que un
    rag index activo >5min hablando con Ollama es altamente sospechoso).

    Procedure:
      1. SIGTERM a todos los matches.
      2. Esperar 5s para que terminen graceful.
      3. SIGKILL a los survivors.

    Returns: `(count_targeted, [pids])` — count incluye cualquier PID al
    que se le mandó SIGTERM, vivan o no después.

    NEVER kills the web server (`uvicorn`/`python -m web.server` etc.) ni
    ningún proceso cuyo command no matchee `rag index` exactamente.
    """
    clients = _list_ollama_clients()
    targets: list[int] = []
    for pid, cmd in clients:
        # Match conservador: el ÚNICO comando que matamos es `rag index`.
        # `rag query` interactivo, `rag chat`, `python -m web.server` y
        # cualquier otro consumer de Ollama queda intacto.
        if "rag index" in cmd or "rag.py index" in cmd:
            targets.append(pid)
    if not targets:
        return 0, []
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(5)
    survivors: list[int] = []
    for pid in targets:
        try:
            os.kill(pid, 0)  # signal 0 = "still alive?"
            survivors.append(pid)
        except OSError:
            pass  # already dead
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(targets), targets


def _quick_generate_health_check(
    *,
    timeout_s: float = 5.0,
    model: str = "qwen2.5:7b",
    base_url: str = "http://127.0.0.1:11434",
) -> tuple[bool, str]:
    """Pega un POST /api/generate con num_predict=1 y timeout corto.

    Devuelve `(healthy, detail)`. `healthy=True` solo si Ollama respondió
    HTTP 2xx con un body parseable y `done=True` (o sea, el round-trip
    completo) dentro del timeout. Si falla la conexión, hay timeout, o
    el body no es JSON válido, devuelve `(False, "...")` con el detail.

    Diseño: usa `urllib.request` (stdlib) para no agregar deps. El body
    pide `num_predict=1` para minimizar el costo del check — un token
    de output basta para confirmar que el runner está vivo y sirviendo.

    Si el modelo no existe localmente, Ollama responde HTTP 404 con
    `{"error":"model 'X' not found"}`. Eso lo contamos como UNHEALTHY
    (el watchdog asume `qwen2.5:7b` por default; si fue removido, el
    user debería override via `RAG_LATENCY_WATCHDOG_HEALTH_MODEL`). Lo
    importante es no devolver `True` falsamente — preferimos cooldown
    legacy a un bypass injustificado.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/api/generate"
    payload = _json.dumps({
        "model": model,
        "prompt": "health",
        "stream": False,
        "options": {"num_predict": 1},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        return False, f"http_error_{exc.code}"
    except urllib.error.URLError as exc:
        # Timeout / connection refused / DNS — todos URLError.
        reason = getattr(exc, "reason", exc)
        return False, f"url_error_{type(reason).__name__}"
    except Exception as exc:
        return False, f"exception_{type(exc).__name__}"
    if status >= 500:
        return False, f"http_5xx_{status}"
    try:
        data = _json.loads(body.decode("utf-8"))
    except Exception:
        return False, "non_json_response"
    # Algunos errores vienen con HTTP 2xx + {"error":"..."} en el body.
    if isinstance(data, dict) and data.get("error"):
        return False, f"server_error: {str(data.get('error'))[:80]}"
    if isinstance(data, dict) and data.get("done") is True:
        return True, "ok"
    return False, "missing_done_flag"


def _restart_ollama_daemon() -> tuple[bool, str]:
    """Reinicia el daemon ollama. Retorna (ok, detail).

    Detecta el deployment activo:
      1. homebrew.mxcl.ollama loaded → kickstart launchctl
      2. Ollama.app running (sin homebrew) → kill + open -a Ollama
      3. Neither → fallback: kill todos + open -a Ollama (best effort)

    Pre-2026-04-28 wave-4 hardcoded la ruta homebrew. Cuando el user
    deshabilitó el plist (porque el daemon duplicado causaba hangs), el
    watchdog quedó inútil. Auto-detección lo arregla.
    """
    try:
        uid = os.getuid()
    except Exception:
        return False, "no_uid"

    # 1) Detect homebrew deployment.
    try:
        check = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        homebrew_loaded = (
            check.returncode == 0
            and "homebrew.mxcl.ollama" in check.stdout
        )
    except Exception:
        homebrew_loaded = False

    if homebrew_loaded:
        try:
            result = subprocess.run(
                ["launchctl", "kickstart", "-k",
                 f"gui/{uid}/homebrew.mxcl.ollama"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if result.returncode == 0:
                return True, "kickstarted_homebrew"
        except Exception:
            pass  # fallthrough to .app path

    # 2) Ollama.app path: kill all ollama procs then reopen.
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "ollama"],
            capture_output=True, timeout=5, check=False,
        )
        time.sleep(2)  # let processes exit cleanly
        subprocess.run(
            ["open", "-a", "Ollama"],
            capture_output=True, timeout=10, check=False,
        )
        return True, "restarted_app"
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _latency_degradation_check(
    *,
    window_minutes: int = 10,
    threshold: float = 3.0,
    min_recent: int = 5,
    cooldown_seconds: int = 1800,
) -> dict:
    """Single-shot check + maybe restart. Retorna dict con resultado.

    Estructura del retorno:
      {
        "action": "skip" | "ok" | "restart_attempted" | "restart_skipped_cooldown",
        "reason": str,
        "p95_recent_ms": int|None,
        "p95_baseline_ms": int|None,
        "ratio": float|None,
        "n_recent": int,
        "n_baseline": int,
      }

    Esta es la función testeable. El loop la llama cada N segundos.
    """
    global _consecutive_degraded_skips, _last_restart_ts
    out: dict = {
        "action": "skip",
        "reason": "",
        "p95_recent_ms": None,
        "p95_baseline_ms": None,
        "ratio": None,
        "n_recent": 0,
        "n_baseline": 0,
    }
    recent, baseline = _read_recent_query_latencies(window_minutes)
    out["n_recent"] = len(recent)
    out["n_baseline"] = len(baseline)
    if len(recent) < min_recent:
        out["reason"] = f"insufficient_recent (n={len(recent)} < {min_recent})"
        return out
    p95_recent = _percentile(recent, 95)
    p95_baseline = _percentile(baseline, 95) if baseline else None
    out["p95_recent_ms"] = int(p95_recent) if p95_recent else None
    out["p95_baseline_ms"] = int(p95_baseline) if p95_baseline else None
    if not p95_baseline or p95_baseline <= 0:
        out["reason"] = "no_baseline"
        return out
    if not p95_recent:
        out["reason"] = "no_p95_recent"
        return out
    ratio = p95_recent / p95_baseline
    out["ratio"] = round(ratio, 2)
    if ratio < threshold:
        # Ollama está sano — resetear contador de skips sostenidos.
        with _consecutive_skips_lock:
            _consecutive_degraded_skips = 0
        out["action"] = "ok"
        out["reason"] = f"ratio={ratio:.2f} < threshold={threshold}"
        return out
    # Need to restart — gate (a): NO matar ollama si hay /api/chat en
    # vuelo. pkill -9 cierra el socket del runner; si una respuesta SSE
    # está streaming a un cliente, el cliente ve "Server disconnected"
    # y la query muere. Skipear acá deja al request actual completar;
    # el próximo check (30s) re-evalúa con el counter en cero.
    n_in_flight = in_flight_chats()
    skipped_for_in_flight = n_in_flight > 0
    if skipped_for_in_flight:
        out["action"] = "restart_skipped_in_flight"
        out["reason"] = f"in_flight={n_in_flight}"
    else:
        # Gate (b): cooldown.
        with _last_restart_lock:
            now_t = time.time()
            if now_t - _last_restart_ts < cooldown_seconds:
                out["action"] = "restart_skipped_cooldown"
                out["reason"] = f"last_restart_was_{int(now_t - _last_restart_ts)}s_ago"
            else:
                _last_restart_ts = now_t
                # Persistir EN CUANTO seteamos el ts en memoria — antes de
                # bouncear el daemon — para que aunque el restart cause
                # efectos colaterales (web server crashea, plist reload,
                # etc.) el cooldown ya esté escrito.
                _persist_restart_ts(now_t)
                ok, detail = _restart_ollama_daemon()
                out["action"] = "restart_attempted"
                out["reason"] = f"restart_ok={ok} detail={detail}"
                # Restart actually fired — reset el contador de degraded
                # skips porque el sistema está post-restart fresh.
                with _consecutive_skips_lock:
                    _consecutive_degraded_skips = 0
                return out
    # Estamos en un skip (in_flight o cooldown) AND degradados. Incrementar
    # contador y, si pasamos el threshold, escalar a kill de zombies.
    with _consecutive_skips_lock:
        _consecutive_degraded_skips += 1
        skips = _consecutive_degraded_skips
        should_escalate = skips >= ESCALATE_KILL_AFTER_SKIPS
        if should_escalate:
            _consecutive_degraded_skips = 0  # reset post-escalation
    if should_escalate:
        prev_action = out["action"]  # captura antes de reasignar
        # Fix #1 (2026-05-01): antes de caer al kill legacy de `rag index`
        # (que es no-op si no hay un index zombie), pegar un health-check
        # directo a /api/generate. Si Ollama NO responde, el cooldown
        # está protegiendo un restart anterior que claramente NO funcionó
        # → bypass + force restart. Si Ollama SÍ responde, comportamiento
        # legacy. Ver doc del módulo "Fix #1 escalación con bypass".
        try:
            health_timeout = float(
                os.environ.get("RAG_LATENCY_WATCHDOG_HEALTH_TIMEOUT", "5")
            )
        except ValueError:
            health_timeout = 5.0
        health_model = os.environ.get(
            "RAG_LATENCY_WATCHDOG_HEALTH_MODEL", "qwen2.5:7b"
        )
        healthy, health_detail = _quick_generate_health_check(
            timeout_s=health_timeout, model=health_model,
        )
        if not healthy and prev_action == "restart_skipped_cooldown":
            # Bypass cooldown — el daemon está roto, los 30min de cooldown
            # están bloqueando exactamente el restart que necesitamos.
            with _last_restart_lock:
                _last_restart_ts = time.time()
                _persist_restart_ts(_last_restart_ts)
            ok, detail = _restart_ollama_daemon()
            out["action"] = "escalated_force_restart_unhealthy"
            out["reason"] = (
                f"degraded_skips={skips} prev_action={prev_action} "
                f"health={health_detail} restart_ok={ok} detail={detail}"
            )
        else:
            n_killed, pids = _kill_stuck_rag_index_clients()
            out["action"] = "escalated_killed_rag_index"
            out["reason"] = (
                f"degraded_skips={skips} prev_action={prev_action} "
                f"health={'ok' if healthy else health_detail} "
                f"killed={n_killed} pids={pids}"
            )
    else:
        # Stamp el counter en la reason para visibilidad en logs sin
        # cambiar el action (queda como `restart_skipped_*`).
        out["reason"] = f"{out['reason']} degraded_skips={skips}/{ESCALATE_KILL_AFTER_SKIPS}"
    return out


def _watchdog_loop(
    interval: int,
    window_minutes: int,
    threshold: float,
    min_recent: int,
    cooldown_seconds: int,
) -> None:
    while True:
        try:
            time.sleep(interval)
            result = _latency_degradation_check(
                window_minutes=window_minutes,
                threshold=threshold,
                min_recent=min_recent,
                cooldown_seconds=cooldown_seconds,
            )
            if result["action"] in (
                "restart_attempted",
                "restart_skipped_cooldown",
                "restart_skipped_in_flight",
                "escalated_killed_rag_index",
                "escalated_force_restart_unhealthy",
            ):
                print(
                    f"[ollama-health-watchdog] {result['action']}: "
                    f"p95_recent={result['p95_recent_ms']}ms "
                    f"p95_baseline={result['p95_baseline_ms']}ms "
                    f"ratio={result['ratio']} reason={result['reason']}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ollama-health-watchdog] error: {exc!r}", flush=True)


def start_latency_degradation_watchdog() -> bool:
    """Idempotent. Returns True if started, False if skipped/disabled."""
    global _started, _last_restart_ts
    with _start_lock:
        if _started:
            return True
        if os.environ.get("RAG_LATENCY_WATCHDOG_DISABLE") == "1":
            return False
        try:
            interval = int(os.environ.get("RAG_LATENCY_WATCHDOG_INTERVAL", "30"))
            window = int(os.environ.get("RAG_LATENCY_WATCHDOG_WINDOW_MIN", "10"))
            threshold = float(os.environ.get("RAG_LATENCY_WATCHDOG_THRESHOLD", "3.0"))
            min_recent = int(os.environ.get("RAG_LATENCY_WATCHDOG_MIN_RECENT", "5"))
            cooldown = int(os.environ.get("RAG_LATENCY_WATCHDOG_COOLDOWN", "1800"))
        except ValueError:
            interval, window, threshold, min_recent, cooldown = 30, 10, 3.0, 5, 1800
        # Hidratar `_last_restart_ts` desde disco para que el cooldown
        # sobreviva restarts del web server. Pre-fix esto era 0.0 en
        # cada boot → el primer check post-arranque siempre pasaba el
        # cooldown gate y restarteaba ollama si p95 estaba alto.
        # Ver doc del módulo y comentario adyacente a `_STATE_FILE`.
        with _last_restart_lock:
            persisted = _load_persisted_restart_ts()
            if persisted > 0:
                _last_restart_ts = persisted
        t = threading.Thread(
            target=_watchdog_loop,
            args=(interval, window, threshold, min_recent, cooldown),
            name="ollama-health-watchdog",
            daemon=True,
        )
        t.start()
        _started = True
        with _last_restart_lock:
            persisted_age = (time.time() - _last_restart_ts) if _last_restart_ts > 0 else None
        persisted_msg = (
            f" persisted_last_restart={int(persisted_age)}s_ago"
            if persisted_age is not None else " persisted_last_restart=none"
        )
        print(
            f"[ollama-health-watchdog] started "
            f"interval={interval}s window={window}min threshold={threshold}x "
            f"min_recent={min_recent} cooldown={cooldown}s{persisted_msg}",
            flush=True,
        )
        return True
