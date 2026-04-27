"""Whisper STT + learning loop — extracted from `rag/__init__.py` (Phase 2 of monolith split, 2026-04-25).

Two responsibilities, one module:

1. **STT primitives** — load `faster-whisper`, transcribe an audio file,
   cache the result by (abs_path, mtime) in `rag_audio_transcripts`.
2. **Learning loop CLI** — admin commands for the vocab + corrections
   pipeline that lives in the sibling package `rag_whisper_learning`
   (vocab refresh, pattern detection, doctor, export/import, stats).

## Surfaces (re-exported on `rag.<name>` via the shim at the bottom of `rag/__init__.py`)

Helpers (callable from Python):
- `_WHISPER_MODEL_DEFAULT` — string, model name fallback.
- `_whisper_model_cache` — dict, memoised `WhisperModel` instances by name.
- `_load_whisper_model(name)` — lazy + memoised loader.
- `_audio_transcript_cache_get(abs_path, mtime)` — SQL cache read.
- `_audio_transcript_cache_put(abs_path, mtime, text, ...)` — SQL cache write.
- `transcribe_audio(path, *, model, language, use_cache)` — main entrypoint.

Click commands (registered on the global `cli` group at import time):
- `transcribe` — `rag transcribe <audio>`, standalone command.
- `whisper` — `rag whisper`, group with these subcommands:
    - `vocab refresh` / `vocab show` — manage `rag_whisper_vocab`.
    - `whisper patterns` — detect repeated single-word swaps.
    - `whisper doctor` — health check end-to-end.
    - `whisper export <-o FILE>` / `whisper import FILE` — backup roundtrip.
    - `whisper stats` — aggregated counters.

## Invariants
- **`faster-whisper` is optional** (`stt` extras group). `_load_whisper_model`
  raises `RuntimeError` with a concrete `uv tool install ...[stt]` hint if
  it's missing — never silent, because a missing dep here means the user
  asked for transcription and we couldn't do it.
- **Cache misses don't fail.** `_audio_transcript_cache_get` returning
  `None` just means we re-transcribe; SQL errors are silenced via
  `_silent_log`.
- **Cache writes are best-effort.** `_audio_transcript_cache_put` swallows
  errors so the caller still gets the transcript object back.
- **CPU-only `compute_type="int8"`** by default — 4-8× faster than float32
  on M-series, imperceptible quality drop for small/base. Switch to
  `int8_float16` when Metal is wired up (out of MVP scope).

## Why deferred imports
`rag/whisper.py` is loaded at the bottom of `rag/__init__.py` (after `cli`,
`console`, and all helpers like `_silent_log`, `_ragvec_state_conn`,
`log_query_event` are defined). The two module-level imports below
(`from rag import cli, console`) work because at that point the parent
package is fully loaded. Helpers used inside function bodies are imported
lazily via `from rag import X` so that:
1. Tests that `monkeypatch.setattr(rag, "_X", ...)` see the patched
   value — `from rag import _X` re-resolves `rag._X` on each call.
2. Future refactors that move helpers around the parent package don't
   require touching this file.

## Why is `rag.whisper` (attribute on `rag`) the Click Group, not the module?
The re-export shim at the bottom of `rag/__init__.py` does
`from rag.whisper import whisper, vocab, ...`. The Python import machinery
clobbers the auto-set submodule attribute `rag.whisper` with the local
name `whisper` (the Click Group). This is intentional — tests like
`assert "doctor" in rag.whisper.commands` rely on `rag.whisper` being
the Click Group. The submodule remains accessible via
`sys.modules["rag.whisper"]` if needed.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from rag import cli, console


# ── STT primitives ──────────────────────────────────────────────────────────
_WHISPER_MODEL_DEFAULT = "small"
_whisper_model_cache: dict[str, object] = {}


def _load_whisper_model(name: str):
    """Lazy-load + memoise a WhisperModel. Raises RuntimeError with a
    concrete install command when faster-whisper isn't available — never
    silent, because a missing dep here means the user asked for
    transcription and we couldn't do it.

    Thread-safe: the dict.setdefault pattern below is atomic under the
    GIL. Loading the same model in two threads at once might duplicate
    work once, but won't corrupt state.
    """
    if name in _whisper_model_cache:
        return _whisper_model_cache[name]
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper no está instalado. "
            "`uv tool install --reinstall --editable '.[stt]'` "
            "o `uv pip install 'obsidian-rag[stt]'`"
        ) from exc
    # compute_type="int8" for CPU — 4-8× faster than float32, imperceptible
    # quality drop for small/base. Switch to "int8_float16" when Metal
    # acceleration is wired up.
    model = WhisperModel(name, device="cpu", compute_type="int8")
    _whisper_model_cache.setdefault(name, model)
    return _whisper_model_cache[name]


def _audio_transcript_cache_get(abs_path: str, mtime: float) -> dict | None:
    """Look up the cache by (path, mtime). Returns None on miss or DB
    unavailable — a miss just means we re-transcribe, not fail."""
    from rag import _ragvec_state_conn, _silent_log
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT text, language, duration_s, model, transcribed_at "
                "FROM rag_audio_transcripts WHERE audio_path = ? AND mtime = ?",
                (abs_path, mtime),
            ).fetchone()
    except Exception as exc:
        _silent_log("audio_transcript_cache_read_failed", exc)
        return None
    if not row:
        return None
    return {
        "text": row[0], "language": row[1],
        "duration_s": row[2], "model": row[3],
        "transcribed_at": row[4],
        "cached": True,
    }


def _audio_transcript_cache_put(
    abs_path: str, mtime: float, text: str,
    language: str | None, duration_s: float | None, model: str,
) -> None:
    """Upsert a transcript into the cache. Best-effort — if the DB write
    fails we return the transcript anyway so the user doesn't lose it."""
    from rag import _ragvec_state_conn, _silent_log
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_audio_transcripts "
                "(audio_path, mtime, text, language, duration_s, model, transcribed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (abs_path, mtime, text, language, duration_s, model, time.time()),
            )
    except Exception as exc:
        _silent_log("audio_transcript_cache_write_failed", exc)


def transcribe_audio(
    path: str | Path,
    *,
    model: str = _WHISPER_MODEL_DEFAULT,
    language: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Transcribe an audio file with faster-whisper + SQL cache.

    Args:
        path: Path to the audio file. Any format ffmpeg reads (mp3, m4a,
            wav, opus, mp4, ogg, flac, aac, …).
        model: faster-whisper model name. Options: tiny / base / small /
            medium / large-v3 / …. Default "small" (~480MB, balanced).
        language: Force a language code (e.g. "es", "en"). None = auto-
            detect.
        use_cache: When True (default), skip the whisper run if we already
            have a transcript for this (abs_path, mtime) in
            rag_audio_transcripts. Set False to force re-transcription
            (useful when testing new models).

    Returns:
        {"text": str, "language": str, "duration_s": float, "model": str,
         "cached": bool, "transcribed_at": float}

    Raises:
        FileNotFoundError: path doesn't exist.
        RuntimeError: faster-whisper not installed (see _load_whisper_model).
    """
    # Re-resolve through `rag` so `monkeypatch.setattr(rag, "_load_whisper_model", ...)`
    # in tests propagates here — see module docstring §"Why deferred imports".
    from rag import (
        _audio_transcript_cache_get as _cache_get,
        _audio_transcript_cache_put as _cache_put,
        _load_whisper_model as _load_model,
    )
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    abs_path = str(p)
    mtime = p.stat().st_mtime

    if use_cache:
        hit = _cache_get(abs_path, mtime)
        if hit is not None:
            return hit

    wm = _load_model(model)
    segments_iter, info = wm.transcribe(
        abs_path, language=language, beam_size=1, vad_filter=True,
    )
    # faster-whisper returns a generator — materialise eagerly so the
    # caller can consume `text` without worrying about iterator
    # exhaustion. On long audios this is O(N) memory but N is the
    # transcript, not the audio samples.
    parts: list[str] = []
    for seg in segments_iter:
        # Each segment has .text with leading/trailing spaces; strip +
        # collapse intra-sentence double-spaces.
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    full_text = " ".join(parts).strip()
    result = {
        "text": full_text,
        "language": getattr(info, "language", None) or language,
        "duration_s": float(getattr(info, "duration", 0.0)) or None,
        "model": model,
        "cached": False,
        "transcribed_at": time.time(),
    }
    if use_cache and full_text:
        _cache_put(
            abs_path, mtime, full_text,
            result["language"], result["duration_s"], model,
        )
    return result


# ── `rag transcribe <audio>` standalone command ─────────────────────────────
@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default=_WHISPER_MODEL_DEFAULT, show_default=True,
              help="Modelo de faster-whisper: tiny/base/small/medium/large-v3")
@click.option("--lang", "language", default=None,
              help="Forzar idioma (ej: es, en). None = auto-detect")
@click.option("--no-cache", "no_cache", is_flag=True,
              help="Forzar re-transcripción, ignorando cache")
@click.option("--json", "as_json", is_flag=True,
              help="Salida JSON con metadata (text, language, duration_s, cached)")
def transcribe(audio_path: str, model: str, language: str | None,
               no_cache: bool, as_json: bool):
    """Transcribir un archivo de audio a texto con faster-whisper.

    Cache automático en SQL (rag_audio_transcripts) keyeado por path +
    mtime — re-correr sobre un audio sin cambios es instantáneo.

    Uso:
      rag transcribe audio.m4a                 # modelo small, auto-lang
      rag transcribe voice.opus --lang es      # forzar español
      rag transcribe talk.wav --model base     # modelo más chico (+rápido)
      rag transcribe note.mp3 --json           # JSON con metadata

    Requiere `faster-whisper` (dep opcional `stt`):
      uv tool install --reinstall --editable '.[stt]'
    """
    from rag import log_query_event, transcribe_audio as _transcribe

    t0 = time.perf_counter()
    try:
        result = _transcribe(
            audio_path, model=model, language=language,
            use_cache=not no_cache,
        )
    except RuntimeError as exc:
        # Dependencia faltante — mensaje claro + exit con código distinto
        # al genérico para que scripts puedan distinguir.
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(6)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2)

    elapsed = time.perf_counter() - t0
    log_query_event({
        "cmd": "transcribe",
        "q": f"[audio] {Path(audio_path).name}",
        "t_retrieve": round(elapsed, 3),
        "extra_json": {
            "audio_path": str(Path(audio_path).resolve()),
            "model": model,
            "language": result.get("language"),
            "duration_s": result.get("duration_s"),
            "cached": bool(result.get("cached")),
            "text_len": len(result.get("text") or ""),
        },
    })

    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))
        return

    text = result.get("text") or ""
    if not text:
        console.print("[yellow]Sin texto transcripto (audio vacío o silencio).[/yellow]")
        return

    # Human output: texto principal + footer con metadata.
    click.echo(text)
    cached_tag = " [dim](caché)[/dim]" if result.get("cached") else ""
    lang = result.get("language") or "?"
    dur = result.get("duration_s")
    dur_str = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "?"
    console.print(
        f"\n[dim]── {model} · {lang} · {dur_str} audio · "
        f"{elapsed:.1f}s tiempo · {len(text)} chars{cached_tag}[/dim]"
    )


# ── `rag whisper` group + subcommands (learning loop admin) ─────────────────
# Phase 2 del plan whatsapp-whisper-learning. Comandos para administrar el
# vocab aprendido + correcciones manuales. Doc completo:
# `04-Archive/99-obsidian-system/99-AI/system/whatsapp-whisper-learning/plan.md`
# en el vault (visible desde Obsidian).

@cli.group(invoke_without_command=True)
@click.pass_context
def whisper(ctx: click.Context):
    """Whisper learning loop — vocab + correcciones del transcript de audios.

    Subcomandos:
      rag whisper vocab refresh   # refresh `rag_whisper_vocab` from corpus
      rag whisper vocab show      # imprimir top vocab terms (debug)
      rag whisper stats           # resumen rápido (transcripciones, correcciones)
    """
    if ctx.invoked_subcommand is None:
        # Default: print stats
        ctx.invoke(whisper_stats)


@whisper.group(invoke_without_command=True)
@click.pass_context
def vocab(ctx: click.Context):
    """Vocab aprendido del corpus que se inyecta al --prompt de whisper.

    Sources: corrections (gold), contacts (Apple Contacts), notes (vault),
    chats (WhatsApp últimos 30d). Refresh nightly via launchd o manualmente.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(vocab_show)


@vocab.command("refresh")
@click.option("--quiet", is_flag=True, help="Solo imprime stats finales, no rows individuales.")
def vocab_refresh(quiet: bool):
    """Re-compute `rag_whisper_vocab` desde el corpus + correcciones.

    Cost típico: ~5-10s para un vault de 5K notas + 14K chats. Reemplaza la
    tabla completa (DELETE + INSERT, no incremental).
    """
    from rag_whisper_learning import refresh_vocab as _refresh
    with console.status("[dim]Computando vocab…[/dim]", spinner="dots"):
        stats = _refresh()
    console.print(
        f"[green]✓[/green] vocab refreshed: "
        f"{stats['total_inserted']} terms en {stats['ms_elapsed']}ms"
    )
    for source, count in stats["sources"].items():
        console.print(f"  [dim]{source:14s}: {count} candidates[/dim]")


@vocab.command("show")
@click.option("--source", default=None, help="Filtrar por source (corrections/contacts/notes/chats).")
@click.option("--limit", default=30, type=int, help="Top-N a mostrar.")
def vocab_show(source: str | None, limit: int):
    """Imprimir top vocab terms ordenados por weight."""
    from rag_whisper_learning import get_top_vocab_terms
    terms = get_top_vocab_terms(limit=limit, source=source)
    if not terms:
        console.print("[yellow]vocab vacío — corré `rag whisper vocab refresh`[/yellow]")
        return
    console.print(f"[bold]Top {len(terms)} vocab terms" + (f" (source={source})" if source else "") + ":[/bold]")
    for t in terms:
        console.print(f"  [dim]{t['weight']:6.2f}[/dim]  [cyan]{t['source']:12s}[/cyan]  {t['term']}")


@whisper.command("patterns")
@click.option("--min-count", default=2, type=int,
              help="Mínimo de repeticiones para mostrar un pattern (default 2).")
def whisper_patterns(min_count: int):
    """Detectar patrones repetidos en correcciones — ej. samando → fernando 3 veces.

    Útil para identificar errores sistemáticos que el modelo whisper hace
    en palabras específicas. El pattern repetido es signal MUY fuerte de
    que la palabra debería estar en el `--prompt` con prioridad alta — y
    de hecho el job de vocab refresh ya las trata como gold signal.

    Algoritmo: single-word swaps (1 palabra removida, 1 agregada) entre
    `original` y `corrected`. Multi-word changes se descartan por noise.
    """
    from rag_whisper_learning import find_correction_patterns
    patterns = find_correction_patterns(min_count=min_count)
    if not patterns:
        console.print(
            f"[yellow]Sin patrones repetidos (≥{min_count} veces)[/yellow] · "
            f"[dim]el sistema necesita más correcciones acumuladas para encontrar señales fuertes[/dim]"
        )
        return
    console.print(f"[bold]{len(patterns)} pattern(s) repetido(s) en correcciones:[/bold]")
    for p in patterns:
        src_parts = []
        if p.sources.get("explicit", 0) > 0:
            src_parts.append(f"[green]{p.sources['explicit']} /fix[/green]")
        if p.sources.get("llm", 0) > 0:
            src_parts.append(f"[cyan]{p.sources['llm']} llm[/cyan]")
        if p.sources.get("vault_diff", 0) > 0:
            src_parts.append(f"[yellow]{p.sources['vault_diff']} vault[/yellow]")
        src_label = " · ".join(src_parts) if src_parts else "?"
        console.print(
            f"  [dim]×{p.count}[/dim]  "
            f"[red]{p.original}[/red] → [green]{p.corrected}[/green]  "
            f"[dim]({src_label})[/dim]"
        )


@whisper.command("doctor")
def whisper_doctor():
    """Health check del whisper learning loop end-to-end.

    Verifica todos los componentes que el sistema necesita para funcionar:
    modelos bajados, server activo, schema SQL completo, vocab fresh,
    LLM disponible, daemons launchd corriendo. Output con semaphore visual
    (✓ / ⚠ / ✗).

    Útil después de un deploy nuevo o cuando algo "dejó de andar"
    inexplicablemente. Sale con exit code 0 si todo OK, 1 si hay errors,
    2 si solo warnings.

    Reusa el patrón de `rag stats` pero específico al learning loop.
    """
    from rag import _ragvec_state_conn
    import shutil as _shutil
    import subprocess as _subprocess
    home = Path.home()
    checks: list[tuple[str, str, str, str]] = []  # (status, label, value, hint)

    # 1. Modelos whisper
    turbo = home / "whisper-models/ggml-large-v3-turbo.bin"
    small = home / "whisper-models/ggml-small.bin"
    vad = home / "whisper-models/ggml-silero-v5.1.2.bin"
    if turbo.is_file():
        size_mb = turbo.stat().st_size / 1024 / 1024
        checks.append(("ok", "modelo turbo", f"{size_mb:.0f}MB", ""))
    elif small.is_file():
        checks.append(("warn", "modelo turbo", "no bajado",
                       "fallback a small.bin (calidad menor); bajar con `curl -L -O ...`"))
    else:
        checks.append(("err", "modelo whisper", "ninguno encontrado",
                       "bajar al menos ggml-small.bin a ~/whisper-models/"))
    if vad.is_file():
        checks.append(("ok", "VAD silero v5", f"{vad.stat().st_size // 1024}KB", ""))
    else:
        checks.append(("warn", "VAD silero v5", "no bajado",
                       "VAD reduce alucinaciones — bajar de ggml-org/whisper-vad"))

    # 2. Whisper server vivo
    try:
        import urllib.request as _urlreq
        with _urlreq.urlopen("http://127.0.0.1:9199/", timeout=2) as resp:
            checks.append(("ok", "whisper-server", f"http 9199 ({resp.status})", ""))
    except Exception:
        checks.append(("warn", "whisper-server", "no responde en :9199",
                       "el listener lo arranca al startup; reload con launchctl"))

    # 3. Telemetry DB + schema
    db_path = home / ".local/share/obsidian-rag/ragvec/telemetry.db"
    if not db_path.is_file():
        checks.append(("err", "telemetry.db", "no existe",
                       f"esperaba en {db_path}"))
    else:
        try:
            import sqlite3 as _sql
            con = _sql.connect(str(db_path))
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            con.close()
            required = {"rag_audio_transcripts", "rag_audio_corrections", "rag_whisper_vocab"}
            missing = required - tables
            if missing:
                checks.append(("err", "telemetry schema",
                               f"falta: {', '.join(sorted(missing))}",
                               "correr cualquier `rag` command para disparar migrations"))
            else:
                checks.append(("ok", "telemetry schema", "3 tablas Phase 2", ""))
        except Exception as exc:
            checks.append(("warn", "telemetry schema", f"error: {exc}", ""))

    # 4. Vocab fresh
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), MAX(refreshed_at) FROM rag_whisper_vocab"
            ).fetchone()
        n_vocab, last_refresh = row
        if n_vocab == 0:
            checks.append(("warn", "vocab", "0 terms",
                           "correr `rag whisper vocab refresh`"))
        elif last_refresh is None:
            checks.append(("warn", "vocab", f"{n_vocab} terms (sin timestamp)", ""))
        else:
            ago_h = (time.time() - last_refresh) / 3600
            if ago_h > 48:
                checks.append(("warn", "vocab",
                               f"{n_vocab} terms (refresh hace {ago_h:.0f}h)",
                               "el plist nightly debería correr 03:15; chequear launchctl"))
            else:
                checks.append(("ok", "vocab",
                               f"{n_vocab} terms (refresh hace {ago_h:.0f}h)", ""))
    except Exception as exc:
        checks.append(("warn", "vocab", f"error: {exc}", ""))

    # 5. Ollama + qwen2.5:7b
    try:
        ollama_bin = _shutil.which("ollama") or "/opt/homebrew/bin/ollama"
        proc = _subprocess.run(
            [ollama_bin, "list"], capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0 and "qwen2.5:7b" in proc.stdout:
            checks.append(("ok", "qwen2.5:7b", "instalado en Ollama", ""))
        elif proc.returncode == 0:
            checks.append(("warn", "qwen2.5:7b", "no instalado",
                           "LLM auto-correct va a fallar; `ollama pull qwen2.5:7b`"))
        else:
            checks.append(("warn", "ollama", "no responde",
                           "service Homebrew launchd; chequear localhost:11434"))
    except Exception as exc:
        checks.append(("warn", "ollama", f"error: {exc}", ""))

    # 6. launchd daemons
    try:
        proc = _subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5
        )
        listener_running = "com.fer.whatsapp-listener" in proc.stdout
        web_running = "com.fer.obsidian-rag-web" in proc.stdout
        vocab_loaded = "com.fer.obsidian-rag-whisper-vocab" in proc.stdout
        if listener_running:
            checks.append(("ok", "whatsapp-listener", "loaded", ""))
        else:
            checks.append(("err", "whatsapp-listener", "no loaded",
                           "`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.whatsapp-listener.plist`"))
        if web_running:
            checks.append(("ok", "obsidian-rag-web", "loaded (`/transcripts` ok)", ""))
        else:
            checks.append(("err", "obsidian-rag-web", "no loaded",
                           "el dashboard no va a estar disponible"))
        if vocab_loaded:
            checks.append(("ok", "vocab nightly job", "plist loaded", ""))
        else:
            checks.append(("warn", "vocab nightly job", "no loaded",
                           "vocab no se va a refreshear automático"))
    except Exception as exc:
        checks.append(("warn", "launchd daemons", f"error: {exc}", ""))

    # Render
    n_ok = sum(1 for c in checks if c[0] == "ok")
    n_warn = sum(1 for c in checks if c[0] == "warn")
    n_err = sum(1 for c in checks if c[0] == "err")
    console.print("[bold]Whisper learning loop — health check[/bold]")
    for status, label, value, hint in checks:
        if status == "ok":
            mark = "[green]✓[/green]"
        elif status == "warn":
            mark = "[yellow]⚠[/yellow]"
        else:
            mark = "[red]✗[/red]"
        line = f"  {mark} [bold]{label:22s}[/bold]  {value}"
        console.print(line)
        if hint:
            console.print(f"    [dim]→ {hint}[/dim]")
    console.print(
        f"\n[bold]Resumen:[/bold] "
        f"[green]{n_ok} ok[/green] · "
        f"[yellow]{n_warn} warn[/yellow] · "
        f"[red]{n_err} err[/red]"
    )
    if n_err > 0:
        sys.exit(1)
    if n_warn > 0:
        sys.exit(2)


@whisper.command("export")
@click.option("--output", "-o", type=click.Path(dir_okay=False, writable=True),
              default=None,
              help="Path al archivo JSON de salida. Default: stdout.")
@click.option("--source", default=None,
              help="Filtrar por source ('explicit'/'llm'/'vault_diff'). Default: todas.")
def whisper_export(output: str | None, source: str | None):
    """Exportar correcciones a JSON para backup o migración entre máquinas.

    Cada row de `rag_audio_corrections` se serializa con todas sus columnas.
    El JSON resultante se puede importar en otra máquina con `rag whisper import`
    o procesar offline.

    Ejemplo:

        rag whisper export -o ~/Backups/corrections-2026-04-25.json
        rag whisper export --source explicit  # solo correcciones manuales /fix
    """
    from rag import _ragvec_state_conn
    sql = (
        "SELECT id, audio_hash, original, corrected, source, ts, chat_id, context "
        "FROM rag_audio_corrections"
    )
    params: tuple = ()
    if source:
        if source not in ("explicit", "llm", "vault_diff"):
            console.print(f"[red]source inválido: {source}[/red] · esperaba: explicit/llm/vault_diff")
            return
        sql += " WHERE source = ?"
        params = (source,)
    sql += " ORDER BY ts ASC"
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        console.print(f"[red]error reading corrections: {exc}[/red]")
        return
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(rows),
        "filter_source": source,
        "schema_version": 1,
        "corrections": [
            {
                "id": r[0],
                "audio_hash": r[1],
                "original": r[2],
                "corrected": r[3],
                "source": r[4],
                "ts": r[5],
                "chat_id": r[6],
                "context": r[7],
            }
            for r in rows
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(blob, encoding="utf-8")
        console.print(
            f"[green]✓[/green] {len(rows)} corrections exported to "
            f"[cyan]{output}[/cyan]"
        )
    else:
        # stdout — útil para piping (`rag whisper export | jq ...`)
        click.echo(blob)


@whisper.command("import")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--dry-run", is_flag=True,
              help="Mostrar qué se importaría sin escribir a la DB.")
def whisper_import(input_file: str, dry_run: bool):
    """Importar correcciones desde un JSON exportado por `rag whisper export`.

    Idempotente: si una correction con el mismo (audio_hash, original,
    corrected, ts) ya existe, skip — esto permite re-importar el mismo
    backup múltiples veces sin duplicados.

    Use cases:
    - Migrar a una máquina nueva: export + scp + import.
    - Restaurar después de un VACUUM accidental que borró rows.
    - Compartir correcciones gold entre 2 setups (raro pero posible).

    Ejemplo:

        rag whisper import ~/Backups/corrections-2026-04-25.json
        rag whisper import backup.json --dry-run
    """
    from rag import _ragvec_state_conn
    try:
        payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]error reading {input_file}: {exc}[/red]")
        return
    rows = payload.get("corrections", [])
    if not isinstance(rows, list):
        console.print("[red]formato inválido: 'corrections' debe ser una lista[/red]")
        return
    schema_v = payload.get("schema_version", 1)
    if schema_v != 1:
        console.print(f"[yellow]warning: schema_version={schema_v} (esperaba 1)[/yellow]")
    if dry_run:
        console.print(
            f"[bold]DRY-RUN[/bold]: importaría {len(rows)} correction(s) de "
            f"[cyan]{input_file}[/cyan]"
        )
        sources = {}
        for r in rows:
            src = r.get("source", "?")
            sources[src] = sources.get(src, 0) + 1
        for src, n in sorted(sources.items()):
            console.print(f"  [dim]{src:12s}: {n}[/dim]")
        return
    inserted = 0
    skipped = 0
    failed = 0
    try:
        with _ragvec_state_conn() as conn:
            for r in rows:
                try:
                    # Idempotencia: skip si (audio_hash, original, corrected, ts)
                    # ya existe. Esto NO es UNIQUE constraint en el schema,
                    # solo guard manual aquí.
                    existing = conn.execute(
                        "SELECT 1 FROM rag_audio_corrections "
                        "WHERE audio_hash = ? AND original = ? "
                        "AND corrected = ? AND ts = ? LIMIT 1",
                        (r.get("audio_hash", ""), r["original"],
                         r["corrected"], r["ts"]),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue
                    conn.execute(
                        "INSERT INTO rag_audio_corrections "
                        "(audio_hash, original, corrected, source, ts, "
                        " chat_id, context) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            r.get("audio_hash", ""),
                            r["original"],
                            r["corrected"],
                            r.get("source", "explicit"),
                            r["ts"],
                            r.get("chat_id"),
                            r.get("context"),
                        ),
                    )
                    inserted += 1
                except (KeyError, TypeError) as exc:
                    failed += 1
                    console.print(f"[yellow]skip row malformada: {exc}[/yellow]")
    except Exception as exc:
        console.print(f"[red]error importing: {exc}[/red]")
        return
    console.print(
        f"[green]✓[/green] imported "
        f"[cyan]{inserted}[/cyan] new · "
        f"[dim]{skipped} skipped (already exist) · "
        f"{failed} failed[/dim]"
    )


@whisper.command("stats")
def whisper_stats():
    """Resumen del estado del learning loop: transcripciones logueadas,
    correcciones acumuladas, top sources, vocab en uso."""
    from rag import _ragvec_state_conn
    try:
        with _ragvec_state_conn() as conn:
            n_transcripts = conn.execute(
                "SELECT COUNT(*) FROM rag_audio_transcripts"
            ).fetchone()[0]
            n_corrections = conn.execute(
                "SELECT COUNT(*) FROM rag_audio_corrections"
            ).fetchone()[0]
            corrections_by_source = dict(conn.execute(
                "SELECT source, COUNT(*) FROM rag_audio_corrections GROUP BY source"
            ).fetchall())
            n_vocab = conn.execute(
                "SELECT COUNT(*) FROM rag_whisper_vocab"
            ).fetchone()[0]
            vocab_by_source = dict(conn.execute(
                "SELECT source, COUNT(*) FROM rag_whisper_vocab GROUP BY source"
            ).fetchall())
            last_refresh = conn.execute(
                "SELECT MAX(refreshed_at) FROM rag_whisper_vocab"
            ).fetchone()[0]
            avg_logprob = conn.execute(
                "SELECT AVG(avg_logprob) FROM rag_audio_transcripts WHERE avg_logprob IS NOT NULL"
            ).fetchone()[0]
    except Exception as exc:
        console.print(f"[red]error reading state: {exc}[/red]")
        return
    console.print("[bold]Whisper learning loop — estado:[/bold]")
    console.print(f"  transcripciones logueadas: [cyan]{n_transcripts}[/cyan]")
    if avg_logprob is not None:
        console.print(f"  avg_logprob promedio:      [dim]{avg_logprob:.3f}[/dim] (-0=conf alta, -1=baja)")
    console.print(f"  correcciones acumuladas:   [cyan]{n_corrections}[/cyan]")
    for src, n in sorted(corrections_by_source.items()):
        console.print(f"    [dim]{src:12s}: {n}[/dim]")
    console.print(f"  vocab terms:               [cyan]{n_vocab}[/cyan]")
    for src, n in sorted(vocab_by_source.items()):
        console.print(f"    [dim]{src:12s}: {n}[/dim]")
    if last_refresh:
        ago = time.time() - last_refresh
        ago_h = int(ago / 3600)
        ago_m = int((ago % 3600) / 60)
        last_str = datetime.fromtimestamp(last_refresh).strftime("%Y-%m-%d %H:%M")
        console.print(f"  último vocab refresh:      [dim]{last_str} ({ago_h}h {ago_m}m ago)[/dim]")
