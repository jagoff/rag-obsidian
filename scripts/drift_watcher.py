#!/usr/bin/env python3
"""Drift watcher para `rag eval` — alerta cuando hit@5 cae entre runs.

Pensado para correr como cron (LaunchAgent ``com.fer.obsidian-rag-drift-watcher``,
cada 6h). Reduce el lag de detección frente al gate nightly de
``com.fer.obsidian-rag-online-tune`` (03:30) — sin esto, una regresión a las
14:00 se detecta recién 13h después.

Telemetría leída:
- Tabla ``rag_eval_runs`` en ``~/.local/share/obsidian-rag/ragvec/telemetry.db``.
- Solo runs con ``singles_n >= 20`` (para filtrar fixtures sintéticos n=2 que
  rompen baseline).

Reglas de alerta:
- ``delta_singles < -0.05`` (-5pp) — singles tiene n grande (60) y baseline 71.67%,
  así que un −5pp es señal real (no ruido).
- ``delta_chains < -0.07`` (-7pp) — chains baseline alto (86.67%) y delta menor
  cae dentro del ruido de muestreo, por eso threshold más relajado.

Outputs en orden:
1. stdout — summary multi-line (ts, prev/current, delta).
2. JSONL en ``~/.local/share/obsidian-rag/drift_alerts.jsonl`` con
   ``{ts, kind, prev, current, delta, current_run_ts}``.
3. Push a WhatsApp (best-effort, fire-and-forget) — POST directo al bridge
   local en ``http://localhost:8080/api/send`` con el JID de ``ambient.json``.
   No depende de importar ``rag`` (peer puede tener edits in-flight).

Idempotencia:
- Antes de escribir un alert nuevo, leemos las últimas 5 líneas del JSONL.
  Si hay un alert con el mismo ``current_run_ts`` y ``kind`` y ts < 12h,
  skip silencioso. Evita spam si el cron de 6h ve el mismo run dos veces.

Resilience:
- Try/except amplio en lectura de DB (no existe, locked, schema-mismatch) →
  log a stderr, exit 0. Nunca fallamos el script porque eso rompe el plist.
- WA push failure NO escala — el JSONL queda como source of truth offline.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path.home() / ".local/share/obsidian-rag"
DB_PATH = DATA_DIR / "ragvec/telemetry.db"
ALERTS_PATH = DATA_DIR / "drift_alerts.jsonl"
AMBIENT_CONFIG_PATH = DATA_DIR / "ambient.json"
WHATSAPP_BRIDGE_URL = "http://localhost:8080/api/send"
ANTILOOP_MARKER = "\u200b"  # zero-width space; bot listener ignora msgs con este prefix

# Mínimo de queries singles por run para considerar el snapshot real (filtra
# fixtures sintéticos con n=2 que disparan falsos positivos).
MIN_SINGLES_N = 20

# Thresholds: deltas en escala 0-1 (no porcentaje). -0.05 = -5pp.
SINGLES_DROP_THRESHOLD = -0.05
CHAINS_DROP_THRESHOLD = -0.07

# Ventana de dedup para idempotencia (alerts del mismo run+kind dentro de
# esta ventana se descartan).
DEDUP_WINDOW_HOURS = 12

# Cuántas líneas del jsonl miramos para chequear duplicados. 5 alcanza
# porque el cron corre cada 6h (≤2 alerts por kind por día como mucho).
DEDUP_LOOKBACK_LINES = 5


# ── Reads ────────────────────────────────────────────────────────────────────


def _fetch_recent_runs(db_path: Path) -> list[dict] | None:
    """Lee las últimas runs de ``rag_eval_runs`` con ``singles_n >= MIN_SINGLES_N``.

    Retorna list[dict] con ``ts, singles_hit5, chains_hit5, singles_n``,
    o ``None`` si la DB no existe / está locked / falla el query.
    """
    if not db_path.is_file():
        print(f"[drift-watcher] db missing: {db_path}", file=sys.stderr)
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        print(f"[drift-watcher] db open failed: {exc!r}", file=sys.stderr)
        return None
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ts, singles_hit5, chains_hit5, singles_n "
            "FROM rag_eval_runs "
            "WHERE singles_n >= ? "
            "ORDER BY ts DESC "
            "LIMIT 5",
            (MIN_SINGLES_N,),
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"[drift-watcher] db query failed: {exc!r}", file=sys.stderr)
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return [dict(r) for r in rows]


def _load_recent_alerts(path: Path, lookback: int = DEDUP_LOOKBACK_LINES) -> list[dict]:
    """Lee las últimas ``lookback`` líneas del jsonl. Silent-fail → []."""
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"[drift-watcher] alerts read failed: {exc!r}", file=sys.stderr)
        return []
    out: list[dict] = []
    for line in lines[-lookback:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ── Idempotency ──────────────────────────────────────────────────────────────


def _is_duplicate(
    recent: list[dict], current_run_ts: str, kind: str, *, now: datetime,
    window_hours: int = DEDUP_WINDOW_HOURS,
) -> bool:
    """True si hay un alert previo con mismo ``current_run_ts`` + ``kind``
    y emitido dentro de la ventana de dedup."""
    cutoff = now - timedelta(hours=window_hours)
    for rec in recent:
        if rec.get("kind") != kind:
            continue
        if rec.get("current_run_ts") != current_run_ts:
            continue
        ts_str = rec.get("ts")
        if not isinstance(ts_str, str):
            continue
        try:
            rec_ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        # Normalizar tz: si viene naive, asumimos hora local — comparamos
        # en el mismo espacio que `now`.
        if rec_ts.tzinfo is None and now.tzinfo is not None:
            rec_ts = rec_ts.replace(tzinfo=now.tzinfo)
        elif rec_ts.tzinfo is not None and now.tzinfo is None:
            rec_ts = rec_ts.replace(tzinfo=None)
        if rec_ts >= cutoff:
            return True
    return False


# ── Writes ───────────────────────────────────────────────────────────────────


def _append_alert(path: Path, payload: dict) -> bool:
    """Append una línea JSONL al alerts log. Crea parent dir si falta.
    Retorna True en éxito."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        print(f"[drift-watcher] alert write failed: {exc!r}", file=sys.stderr)
        return False


def _push_whatsapp(text: str, *, config_path: Path = AMBIENT_CONFIG_PATH,
                   bridge_url: str = WHATSAPP_BRIDGE_URL) -> bool:
    """Best-effort push al bridge local de WhatsApp.

    Lee el JID del ``ambient.json`` directamente (no depende de importar
    ``rag``, que puede estar en estado inconsistente si un peer está
    editándolo). Prefixa U+200B para que el listener del bot RAG no
    procese el mensaje como query entrante.

    Retorna True en 2xx del bridge, False en cualquier otra cosa
    (config missing, JID missing, bridge unreachable, timeout). NO
    raisea — el caller asume best-effort.
    """
    if not config_path.is_file():
        return False
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    jid = cfg.get("jid") if isinstance(cfg, dict) else None
    enabled = cfg.get("enabled", True) if isinstance(cfg, dict) else False
    if not jid or not enabled:
        return False
    payload_text = text if text.startswith(ANTILOOP_MARKER) else (ANTILOOP_MARKER + text)
    body = json.dumps({"recipient": jid, "message": payload_text}).encode("utf-8")
    req = urllib.request.Request(
        bridge_url, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# ── Core logic ───────────────────────────────────────────────────────────────


def _format_summary(kind: str, prev: dict, current: dict, delta: float) -> str:
    """Render multi-line para stdout + WhatsApp. Delta en pp (puntos
    porcentuales)."""
    metric = f"{kind}_hit5"
    delta_pp = delta * 100
    return (
        f"[drift-watcher] DRIFT detected on {metric}\n"
        f"  prev:    {prev.get('ts')} → {(prev.get(metric) or 0) * 100:.2f}%\n"
        f"  current: {current.get('ts')} → {(current.get(metric) or 0) * 100:.2f}%\n"
        f"  delta:   {delta_pp:+.2f}pp"
    )


def evaluate(
    db_path: Path = DB_PATH,
    alerts_path: Path = ALERTS_PATH,
    *,
    now: datetime | None = None,
    push_whatsapp: bool = True,
) -> int:
    """Lógica del watcher, parametrizada para tests.

    Returns un exit code (0 = OK, siempre — nunca fallamos el plist).
    """
    if now is None:
        # Hora local — alineada con cómo pytest fixtures + jsonl logs
        # ya escriben ts (datetime.now().isoformat). Mantenemos la
        # convención existente del proyecto.
        now = datetime.now()

    rows = _fetch_recent_runs(db_path)
    if rows is None:
        # SQL error ya logueado a stderr — exit clean.
        return 0
    if len(rows) < 2:
        print("[drift-watcher] insufficient data")
        return 0

    current, prev = rows[0], rows[1]

    # Si alguno de los hit5 es None (run viejo sin métrica), tratamos
    # como ausencia de señal — no alertamos.
    cur_singles = current.get("singles_hit5")
    prev_singles = prev.get("singles_hit5")
    cur_chains = current.get("chains_hit5")
    prev_chains = prev.get("chains_hit5")

    delta_singles = (
        cur_singles - prev_singles
        if cur_singles is not None and prev_singles is not None
        else 0.0
    )
    delta_chains = (
        cur_chains - prev_chains
        if cur_chains is not None and prev_chains is not None
        else 0.0
    )

    drift_kinds: list[tuple[str, float]] = []
    if delta_singles < SINGLES_DROP_THRESHOLD:
        drift_kinds.append(("singles", delta_singles))
    if delta_chains < CHAINS_DROP_THRESHOLD:
        drift_kinds.append(("chains", delta_chains))

    if not drift_kinds:
        print(
            f"[drift-watcher] OK delta_singles={delta_singles * 100:+.2f}pp "
            f"delta_chains={delta_chains * 100:+.2f}pp"
        )
        return 0

    recent_alerts = _load_recent_alerts(alerts_path)
    current_run_ts = current.get("ts") or ""

    for kind, delta in drift_kinds:
        if _is_duplicate(recent_alerts, current_run_ts, kind, now=now):
            print(f"[drift-watcher] dedup skip kind={kind} run_ts={current_run_ts}")
            continue

        prev_val = prev_singles if kind == "singles" else prev_chains
        cur_val = cur_singles if kind == "singles" else cur_chains

        summary = _format_summary(kind, prev, current, delta)
        print(summary)

        payload = {
            "ts": now.isoformat(timespec="seconds"),
            "kind": kind,
            "prev": prev_val,
            "current": cur_val,
            "delta": delta,
            "current_run_ts": current_run_ts,
            "prev_run_ts": prev.get("ts"),
        }
        _append_alert(alerts_path, payload)

        if push_whatsapp:
            ok = _push_whatsapp(summary)
            if not ok:
                # No es failure del watcher — solo telemetría a stderr.
                print(
                    f"[drift-watcher] wa push skipped/failed kind={kind}",
                    file=sys.stderr,
                )

    return 0


def main() -> int:
    try:
        return evaluate()
    except Exception as exc:  # pragma: no cover — defensa final
        # No queremos NUNCA romper el plist. Cualquier excepción no
        # capturada arriba sale acá con stderr + exit 0.
        print(f"[drift-watcher] uncaught exception: {exc!r}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
