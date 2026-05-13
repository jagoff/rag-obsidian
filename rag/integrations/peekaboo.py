"""Peekaboo screen-capture integration — Fase 1 (on-demand, pull only).

Thin wrapper sobre el CLI [`peekaboo`](https://github.com/openclaw/Peekaboo).
Captura ventana/pantalla activa a PNG temporal, captiona con granite MLX-VLM
(reusa [`rag.ocr._vlm_describe`](../ocr.py)), retorna `{caption, image_path,
mode, took_ms}`. **Sin persistencia** — la PNG va a `/tmp` con 0600, queda en
disco hasta que el caller (o tmpreaper) la borre.

## Por qué subprocess + no MCP/Node

Peekaboo CLI es Swift nativo (~40MB, instalable via brew). Su servidor MCP
es Node 22+ y agrega un bridge process extra. Fase 1 sólo necesita capture
on-demand desde el orquestador Python — subprocess al CLI es el path más
corto, sin daemon nuevo, sin bridge.

## Gate

`RAG_PEEKABOO_ENABLE=1` requerido. Default OFF. Si OFF, `capture_and_caption`
retorna `{"error": "peekaboo_disabled"}` sin tocar el CLI.

## TCC (permisos macOS)

`peekaboo image` necesita **Screen Recording** otorgado al **responsible
process** del shell que lo invoca (típicamente la app terminal: Ghostty,
Terminal, iTerm). Si TCC no está concedido, el CLI sale con exit 1 y stderr
`Screen recording permission is required`. Ese error se propaga como
`{"error": "tcc_denied", ...}` para que el caller pueda mostrar instrucciones.

## Fase 2 (futuro, no implementado acá)

Signal pasivo `active_window_change` en `rag_anticipate/signals/` con poll
periódico + dedup por title hash. Esta Fase 1 deja la base reutilizable.
"""

from __future__ import annotations

import hashlib
import json as _json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


_PEEKABOO_BIN_ENV = "RAG_PEEKABOO_BIN"
_PEEKABOO_TIMEOUT_ENV = "RAG_PEEKABOO_TIMEOUT_SECS"
_PEEKABOO_ENABLE_ENV = "RAG_PEEKABOO_ENABLE"

# Fase 2 — signal pasivo (default OFF, opt-in).
_SCREEN_OBSERVE_ENV = "RAG_SCREEN_OBSERVE"
_SCREEN_QUIET_HOURS_ENV = "RAG_SCREEN_QUIET_HOURS"  # ej. "22:00-07:00"
_SCREEN_APP_DENY_ENV = "RAG_SCREEN_APP_DENY"        # CSV: "1Password,Banking"
_SCREEN_DEDUP_WINDOW_SECS = 60

_VALID_MODES = {"frontmost", "window", "screen", "multi", "menubar"}


def _is_enabled() -> bool:
    """True si el feature está activado por env."""
    return os.environ.get(_PEEKABOO_ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _resolve_binary() -> str | None:
    """Resolve absoluto al ejecutable peekaboo. None si no está instalado."""
    override = os.environ.get(_PEEKABOO_BIN_ENV, "").strip()
    if override:
        return override if Path(override).exists() else None
    return shutil.which("peekaboo")


def _resolve_timeout() -> float:
    raw = os.environ.get(_PEEKABOO_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 10.0


def _capture_png(
    mode: str = "frontmost",
    app: str | None = None,
    retina: bool = False,
) -> tuple[Path | None, str | None]:
    """Llama `peekaboo image` y guarda PNG en /tmp. Returns (path, error).

    Modes:
        frontmost — ventana en foreground del sistema (default).
        window    — primera ventana del app (requiere app=).
        screen    — display completo.
        multi     — todas las pantallas, una PNG c/u (returna la primera).
        menubar   — barra de menú.

    `retina=True` agrega `--retina` (2x density).
    """
    if mode not in _VALID_MODES:
        return None, f"invalid_mode: {mode!r} not in {sorted(_VALID_MODES)}"

    bin_path = _resolve_binary()
    if bin_path is None:
        return None, "peekaboo_not_installed"

    fd, tmp_path_str = tempfile.mkstemp(prefix="rag-peek-", suffix=".png", dir="/tmp")
    os.close(fd)
    os.chmod(tmp_path_str, 0o600)
    tmp_path = Path(tmp_path_str)

    cmd: list[str] = [bin_path, "image", "--mode", mode, "--path", tmp_path_str]
    if app:
        cmd.extend(["--app", app])
    if retina:
        cmd.append("--retina")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_resolve_timeout(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return None, "peekaboo_timeout"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return None, f"peekaboo_exec_error: {exc}"

    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        stderr = (proc.stderr or "").strip()
        if "permission" in stderr.lower() or "screen recording" in stderr.lower():
            return None, f"tcc_denied: {stderr}"
        return None, f"peekaboo_failed (exit {proc.returncode}): {stderr[:300]}"

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return None, "peekaboo_empty_output"

    return tmp_path, None


def _caption(image_path: Path, prompt: str | None = None) -> tuple[str, str | None]:
    """Wrapper sobre [`rag.ocr._vlm_describe`](../ocr.py). Returns (caption, error).

    Silent-fail del VLM se convierte en `error="vlm_empty"` para que el caller
    distinga capture exitosa pero caption vacío vs capture rota.
    """
    try:
        from rag.ocr import _vlm_describe  # noqa: PLC0415
    except Exception as exc:
        return "", f"vlm_import_error: {exc}"

    actual_prompt = prompt or ""
    try:
        text = _vlm_describe(image_path, actual_prompt) or ""
    except Exception as exc:
        try:
            from rag import _silent_log  # noqa: PLC0415
            _silent_log("peekaboo_caption", exc)
        except Exception:
            pass
        return "", f"vlm_error: {exc}"

    text = text.strip()
    if not text:
        return "", "vlm_empty"
    return text, None


def capture_and_caption(
    mode: str = "frontmost",
    app: str | None = None,
    prompt: str | None = None,
    retina: bool = False,
    keep_image: bool = False,
) -> dict:
    """Orquesta capture + caption. Returns dict serializable.

    Args:
        mode/app/retina — pasados a `_capture_png`.
        prompt — override del caption prompt (default: `_VLM_CAPTION_PROMPT`).
        keep_image — si False (default), borra la PNG después de captionar.
            Si True, devuelve `image_path` para que el caller lea/persista.

    Returns:
        Dict con keys:
            ok (bool) — true sólo si capture+caption ambos andan.
            caption (str) — texto del VLM (vacío en error).
            mode/app/retina — echo de inputs.
            image_path (str | None) — sólo si keep_image=True y capture ok.
            took_ms (int) — wall-time total.
            error (str | None) — código de error si ok=False.
    """
    started = time.time()
    base_result: dict = {
        "ok": False,
        "caption": "",
        "mode": mode,
        "app": app,
        "retina": retina,
        "image_path": None,
        "took_ms": 0,
        "error": None,
    }

    if not _is_enabled():
        base_result["error"] = "peekaboo_disabled"
        base_result["took_ms"] = int((time.time() - started) * 1000)
        return base_result

    png_path, capture_err = _capture_png(mode=mode, app=app, retina=retina)
    if capture_err or png_path is None:
        base_result["error"] = capture_err or "peekaboo_unknown_error"
        base_result["took_ms"] = int((time.time() - started) * 1000)
        return base_result

    caption_text, caption_err = _caption(png_path, prompt=prompt)
    if caption_err and not caption_text:
        if not keep_image:
            png_path.unlink(missing_ok=True)
        base_result["error"] = caption_err
        base_result["took_ms"] = int((time.time() - started) * 1000)
        return base_result

    base_result["ok"] = True
    base_result["caption"] = caption_text
    if keep_image:
        base_result["image_path"] = str(png_path)
    else:
        png_path.unlink(missing_ok=True)
    base_result["took_ms"] = int((time.time() - started) * 1000)
    return base_result


# ── Fase 2 — passive observer ────────────────────────────────────────────────


def _is_observe_enabled() -> bool:
    """Gate del daemon de captura pasiva. Separado de `RAG_PEEKABOO_ENABLE`
    porque hay 2 niveles de opt-in: el binario activado para uso on-demand
    (`rag screen`, MCP tool) y el daemon activado para capturar background."""
    return os.environ.get(_SCREEN_OBSERVE_ENV, "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _parse_quiet_hours(spec: str | None) -> tuple[int, int] | None:
    """`"22:00-07:00"` → (1320, 420) en minutos-desde-medianoche.

    None / vacío / spec malformado → None (sin quiet hours).
    """
    if not spec or "-" not in spec:
        return None
    try:
        start_s, end_s = spec.split("-", 1)
        sh, sm = start_s.strip().split(":")
        eh, em = end_s.strip().split(":")
        start_min = int(sh) * 60 + int(sm)
        end_min = int(eh) * 60 + int(em)
        if not (0 <= start_min < 1440 and 0 <= end_min < 1440):
            return None
        return start_min, end_min
    except (ValueError, AttributeError):
        return None


def _in_quiet_hours(now: datetime, spec: str | None = None) -> bool:
    """True si `now` cae en la ventana quiet hours. Soporta wrap medianoche
    (start > end → window cruza 00:00, ej. 22:00-07:00)."""
    parsed = _parse_quiet_hours(
        spec if spec is not None else os.environ.get(_SCREEN_QUIET_HOURS_ENV, ""),
    )
    if parsed is None:
        return False
    start_min, end_min = parsed
    now_min = now.hour * 60 + now.minute
    if start_min <= end_min:
        return start_min <= now_min < end_min
    return now_min >= start_min or now_min < end_min


def _app_denylist() -> frozenset[str]:
    """Apps cuya pantalla NUNCA se observa. CSV en env, case-insensitive."""
    raw = os.environ.get(_SCREEN_APP_DENY_ENV, "")
    return frozenset(s.strip().lower() for s in raw.split(",") if s.strip())


def _simhash64(text: str) -> int:
    """Fingerprint 64-bit del caption (signed int64 para SQLite INTEGER).

    Usa sha1[:16] hex → int. Determinístico, sin dependencias extra.
    No es un simhash "real" (sin token-bucketing), pero alcanza para
    dedup exacto post-fact. Para semantic dedup, hace falta embedder.
    """
    h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    val = int(h[:16], 16)
    # Convertir a signed int64 para que entre en SQLite INTEGER sin overflow.
    if val >= 2**63:
        val -= 2**64
    return val


def _capture_with_meta(
    mode: str = "frontmost",
    app: str | None = None,
    retina: bool = False,
) -> tuple[Path | None, dict, str | None]:
    """Variante de `_capture_png` que también parsea `--json` stdout para
    extraer metadata de ventana (app_name, window_title). Returns (path,
    meta_dict, error).

    Meta keys conocidos (defensivo — Peekaboo puede no popularlos siempre):
        app_name, window_title.
    """
    meta: dict = {}
    if mode not in _VALID_MODES:
        return None, meta, f"invalid_mode: {mode!r} not in {sorted(_VALID_MODES)}"

    bin_path = _resolve_binary()
    if bin_path is None:
        return None, meta, "peekaboo_not_installed"

    fd, tmp_path_str = tempfile.mkstemp(prefix="rag-peek-", suffix=".png", dir="/tmp")
    os.close(fd)
    os.chmod(tmp_path_str, 0o600)
    tmp_path = Path(tmp_path_str)

    cmd: list[str] = [bin_path, "image", "--mode", mode, "--path", tmp_path_str, "--json"]
    if app:
        cmd.extend(["--app", app])
    if retina:
        cmd.append("--retina")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_resolve_timeout(), check=False,
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return None, meta, "peekaboo_timeout"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return None, meta, f"peekaboo_exec_error: {exc}"

    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        # En modo `--json`, Peekaboo emite errores en stdout como
        # `{"success": false, "error": "..."}` y deja stderr vacío. Hay que
        # mirar ambos para detectar TCC.
        combined = (stderr + " " + stdout).lower()
        if "permission" in combined or "screen recording" in combined:
            detail = stderr or stdout
            return None, meta, f"tcc_denied: {detail[:300]}"
        detail = stderr or stdout
        return None, meta, f"peekaboo_failed (exit {proc.returncode}): {detail[:300]}"

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return None, meta, "peekaboo_empty_output"

    # Best-effort parse del JSON stdout. No crítico — si falla, meta queda
    # vacío y observe_once igual escribe la observación sin app/title.
    try:
        parsed = _json.loads((proc.stdout or "").strip())
        # El shape exacto depende de la versión de Peekaboo; navegamos
        # defensivamente sobre las keys más comunes.
        if isinstance(parsed, dict):
            data = parsed.get("data", parsed)  # algunos shapes envuelven en `data`
            if isinstance(data, dict):
                app_name = (
                    data.get("app_name")
                    or data.get("application")
                    or data.get("app")
                )
                window_title = (
                    data.get("window_title")
                    or data.get("title")
                    or data.get("window")
                )
                if isinstance(app_name, str):
                    meta["app_name"] = app_name.strip()
                if isinstance(window_title, str):
                    meta["window_title"] = window_title.strip()
    except (_json.JSONDecodeError, AttributeError, TypeError):
        pass

    return tmp_path, meta, None


def _query_last_observation(
    con,
    app_name: str | None,
    within_seconds: int = _SCREEN_DEDUP_WINDOW_SECS,
) -> dict | None:
    """Devuelve el last row de `rag_screen_observations` para `app_name`
    dentro de los últimos `within_seconds`. Para dedup titular.

    `con` es una sqlite3.Connection abierta a telemetry.db. None si no hay
    match o si la tabla no existe (caller debe ensure-once antes)."""
    if not app_name:
        return None
    cutoff = int(time.time()) - max(1, within_seconds)
    try:
        row = con.execute(
            "SELECT id, ts, app_name, window_title, caption_simhash "
            "FROM rag_screen_observations "
            "WHERE app_name = ? AND ts >= ? "
            "ORDER BY ts DESC LIMIT 1",
            (app_name, cutoff),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {
        "id": row[0],
        "ts": row[1],
        "app_name": row[2],
        "window_title": row[3],
        "caption_simhash": row[4],
    }


def observe_once(
    now: datetime | None = None,
    *,
    mode: str = "frontmost",
    dedup_seconds: int = _SCREEN_DEDUP_WINDOW_SECS,
) -> dict:
    """Single tick del observer: capture + caption + insert con dedup.

    Returns dict con keys:
        ok (bool), observation_id (int|None), app_name, window_title,
        caption, took_ms, skipped_reason (str|None), error (str|None).

    Skip conditions (no escritura a DB, no VLM call cuando es posible):
        - `RAG_SCREEN_OBSERVE` OFF → `observe_disabled`.
        - Capture gate `RAG_PEEKABOO_ENABLE` OFF → `peekaboo_disabled`
          (porque sin captura no hay observación).
        - Quiet hours match → `quiet_hours`.
        - App en denylist → `app_denied` (post-capture, requiere subprocess
          para conocer el app_name; PNG borrada y caption skipeado).
        - Titular dedup hit → `dedup_title` (post-capture, pre-VLM).
        - VLM empty caption → `vlm_empty` (capture ok pero no escribimos
          row con caption vacío — sería ruido).
    """
    started = time.time()
    result: dict = {
        "ok": False,
        "observation_id": None,
        "app_name": None,
        "window_title": None,
        "caption": "",
        "took_ms": 0,
        "skipped_reason": None,
        "error": None,
    }

    def _finalize(**kw) -> dict:
        result.update(kw)
        result["took_ms"] = int((time.time() - started) * 1000)
        return result

    if not _is_observe_enabled():
        return _finalize(skipped_reason="observe_disabled")
    if not _is_enabled():
        return _finalize(skipped_reason="peekaboo_disabled")

    now_local = now or datetime.now()
    if _in_quiet_hours(now_local):
        return _finalize(skipped_reason="quiet_hours")

    png_path, meta, capture_err = _capture_with_meta(mode=mode)
    if capture_err or png_path is None:
        return _finalize(error=capture_err or "peekaboo_unknown_error")

    app_name = meta.get("app_name")
    window_title = meta.get("window_title")
    result["app_name"] = app_name
    result["window_title"] = window_title

    deny = _app_denylist()
    if app_name and app_name.lower() in deny:
        png_path.unlink(missing_ok=True)
        return _finalize(skipped_reason="app_denied")

    # Titular dedup ANTES de invocar el VLM (que es la parte cara).
    # Necesita ensure de la tabla — caller responsable de pasar el conn
    # ya ensure-eado. observe_once abre + cierra su propio conn vía rag.
    try:
        import rag as _rag  # noqa: PLC0415
        import sqlite3  # noqa: PLC0415
        db = _rag.DB_PATH / "telemetry.db"
        con = sqlite3.connect(str(db), timeout=5.0)
        try:
            _rag._ensure_telemetry_tables(con)
            last = _query_last_observation(con, app_name, within_seconds=dedup_seconds)
            if last and last.get("window_title") == window_title and window_title:
                con.close()
                png_path.unlink(missing_ok=True)
                return _finalize(skipped_reason="dedup_title")
        except Exception as exc:
            con.close()
            png_path.unlink(missing_ok=True)
            return _finalize(error=f"db_pre_check_error: {exc}")
    except Exception as exc:
        png_path.unlink(missing_ok=True)
        return _finalize(error=f"rag_import_error: {exc}")

    caption_text, caption_err = _caption(png_path)
    png_path.unlink(missing_ok=True)  # PNG efímera — ya tenemos caption.

    # vlm_empty no es error — el VLM corrió pero no devolvió texto útil
    # (imagen oscura, blank, etc.). Skipear sin escribir row evita ruido.
    # Otros vlm_err (excepción del modelo) sí son error real.
    if caption_err == "vlm_empty" or (not caption_text and not caption_err):
        con.close()
        return _finalize(skipped_reason="vlm_empty")
    if caption_err and not caption_text:
        con.close()
        return _finalize(error=caption_err)

    ts_epoch = int(now_local.timestamp())
    simhash = _simhash64(caption_text)
    try:
        cur = con.execute(
            "INSERT INTO rag_screen_observations "
            "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ts_epoch, app_name, window_title, caption_text,
                simhash, int((time.time() - started) * 1000), mode,
            ),
        )
        observation_id = cur.lastrowid
        con.commit()
    except Exception as exc:
        con.close()
        return _finalize(error=f"db_insert_error: {exc}")
    finally:
        try:
            con.close()
        except Exception:
            pass

    return _finalize(
        ok=True,
        observation_id=observation_id,
        caption=caption_text,
    )


__all__ = [
    "_is_enabled",
    "_resolve_binary",
    "_capture_png",
    "_caption",
    "capture_and_caption",
    # Fase 2:
    "_is_observe_enabled",
    "_parse_quiet_hours",
    "_in_quiet_hours",
    "_app_denylist",
    "_simhash64",
    "_capture_with_meta",
    "_query_last_observation",
    "observe_once",
]
