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

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


_PEEKABOO_BIN_ENV = "RAG_PEEKABOO_BIN"
_PEEKABOO_TIMEOUT_ENV = "RAG_PEEKABOO_TIMEOUT_SECS"
_PEEKABOO_ENABLE_ENV = "RAG_PEEKABOO_ENABLE"

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


__all__ = [
    "_is_enabled",
    "_resolve_binary",
    "_capture_png",
    "_caption",
    "capture_and_caption",
]
