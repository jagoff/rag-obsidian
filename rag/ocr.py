"""OCR + VLM caption + intent detector — extracted from `rag/__init__.py` (Phase 6 of monolith split, 2026-04-25).

Three concerns bundled into one module because they form a single pipeline:

1. **OCR primitives** (Apple Vision via `ocrmac`): walk a markdown body
   for embedded images, run Apple's on-device OCR, cache results in
   `rag_ocr_cache`. Used by the indexer to make image-heavy notes
   (link-hubs, screenshots, whiteboards) searchable.
2. **VLM caption fallback** (granite-vision-3.2-2b via mlx-vlm): when OCR
   returns < `_VLM_FALLBACK_MIN_OCR` chars, run a vision-language model that
   produces a grep-friendly caption. Backend: mlx-community/granite-vision-3.2-2b-4bit
   via `mlx_vlm.generate` (Apple Silicon, MPS). Cache in `rag_vlm_captions`.
   Per-run budget cap so a fresh index doesn't burn time on captions.
3. **OCR → intent detector**: the OCR/VLM text gets classified by
   qwen2.5:3b into `event` / `reminder` / `note`. Events trigger
   `propose_calendar_event`, reminders trigger `propose_reminder`, notes
   are no-op. SHA256-keyed dedup via `rag_cita_detections` so the same
   image isn't re-classified across indexer runs.

## Why bundled

Each concern individually is meaningful (~200-500 lines) but they share
state (the OCR text feeds VLM-fallback feeds intent detector) and tests
exercise them together. Splitting into 3 separate modules would force
3-way deferred imports for symbols that are tightly coupled.

## Surfaces (re-exported on `rag.<X>` via shim at the bottom of `rag/__init__.py`)

OCR:
- `_ocrmac_module`, `_ocrmac_import_attempted` — global state.
- `_load_ocrmac_module()` — lazy + cached importer.
- `_IMAGE_EXTENSIONS`, `_OCR_MIN_CONFIDENCE`.
- `_EMBED_WIKILINK_RE`, `_EMBED_MARKDOWN_RE` — markdown image regexes.
- `_extract_embedded_images(body, note_path, vault_root)`.
- `_ocr_image(image_path)`.

VLM:
- `VLM_MODEL`, `_VLM_FALLBACK_MIN_OCR`, `_VLM_CAPTION_MAX_CHARS`,
  `_VLM_CAPTION_MAX_PER_RUN`, `_VLM_CAPTION_PROMPT`.
- `_vlm_caption_enabled()`, `_vlm_load()`, `_vlm_idle_unload()`,
  `_vlm_describe(image_path, prompt)`,
  `_vlm_caption_budget_reset/available/consume()`.
- `_caption_image(image_path)`.
- `_image_text_or_caption(image_path)` — OCR-or-VLM dispatcher.
- `_enrich_body_with_ocr(body, note_path, vault_root)`.
- `_file_hash_with_images(raw, note_path, vault_root)`.

Intent detector:
- `_CITA_MIN_CONFIDENCE`, `_CITA_MIN_CHARS`, `_CITA_VALID_KINDS`,
  `_CITA_PROMPT_SYSTEM`, `_CITA_PROMPT_USER_TEMPLATE`.
- `_cita_detect_enabled()`, `_normalize_ocr_for_hash`, `_ocr_hash_key`.
- `_detect_cita_from_ocr(ocr_text)`.
- `_maybe_create_cita_from_ocr(...)`.
- `_cita_result(...)`, `_persist_cita_detection(...)`.

## Tests-friendly: monkey-patch propagation via `rag.<X>`

Tests heavily patch:
- `rag._ocrmac_module`, `rag._ocr_image`, `rag._vlm_describe`,
  `rag._VLM_CAPTION_MAX_PER_RUN`, `rag._detect_cita_from_ocr`.

For each direct internal caller of these names (e.g. `_load_ocrmac_module`
reading `_ocrmac_module`, `_image_text_or_caption` calling `_ocr_image`,
`_maybe_create_cita_from_ocr` calling `_detect_cita_from_ocr`), we
resolve via `_rag.<X>` (where `_rag` is `import rag as _rag` inside
the function body) so monkey-patches propagate.

The convention: helpers in `rag/__init__.py` (`_silent_log`,
`_ragvec_state_conn`, `_helper_client`, `propose_calendar_event`,
`propose_reminder`, `file_hash`, `_now_utc_iso_z`, `dateparser`-based
helpers) are imported via `from rag import X` inside function bodies —
each call re-resolves `rag.X`, so patches still propagate.

## Deferred imports

`rag/ocr.py` is loaded by the re-export shim at the bottom of
`rag/__init__.py`, after every helper it depends on is defined. The
heavy `ocrmac` import is still lazy (inside `_load_ocrmac_module`);
mlx-vlm (`_vlm_load`) is also lazy — module load stays fast.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

# NOTE: helpers from `rag.__init__` are imported INSIDE each function body
# (deferred) — module-level `from rag import _helper_client, propose_*, ...`
# would capture the binding at module-load time and miss subsequent test
# monkey-patches (`monkeypatch.setattr(rag, "_helper_client", stub)`).
# `from rag import X` inside a function re-resolves `rag.X` on every call.
#
# Same for monkey-patched names (`_ocrmac_module`, `_ocr_image`,
# `_vlm_describe`, `_VLM_CAPTION_MAX_PER_RUN`, `_detect_cita_from_ocr`):
# resolved through `import rag as _rag` + `_rag.<X>` inside function bodies.


# Sentinel for the lazy-loaded ocrmac module attribute that tests patch.

# ── OCR on embedded images (Apple Vision via ocrmac) ─────────────────────────
# Indexer hook: el body de una nota se enriquece con el texto OCR de sus
# imágenes embebidas ANTES del chunking. Sin esto, notas tipo link-hub
# (filename descriptivo + body = un link + un `![[screenshot.png]]`) eran
# invisibles al retrieval — el reranker solo ve el body textual y no tiene
# forma de leer la captura. Con OCR, el texto de la imagen (ej. tabla de
# dev cycles en una screenshot) se concatena al body y aterriza en un
# chunk indexable.
#
# Cache: SQL `rag_ocr_cache`, key = image abs path, invalidado por mtime.
# Sin TTL — el texto no envejece, solo cambia cuando la imagen cambia.
# El hash del chunk `_index_single_file` se computa sobre raw + mtimes de
# imágenes (`_file_hash_with_images`) así que una imagen actualizada
# fuerza reindex aunque el markdown no cambie.
#
# Soft deps: `ocrmac` + pyobjc son macOS-only. Import fallido → silent
# skip. Env `RAG_OCR=0` desactiva explícitamente.

# Lazy-load. `ocrmac` arrastra pyobjc → Vision + Quartz + CoreML + AppKit,
# ~130ms de import time en macOS. El CLI arranca para muchos subcomandos
# que no hacen OCR (query, chat, session list, vault list, etc.), así que
# importar al cargar el módulo era pagar esos 130ms cada vez (37% del
# cold-start de `rag --help`). Con este pattern la import se dispara
# solo cuando `_ocr_image()` corre por primera vez.
#
# `_ocrmac_module` sigue existiendo como attribute del módulo — los tests
# lo patchean via `monkeypatch.setattr(rag, "_ocrmac_module", fake)` y
# ese contrato se preserva: si el attribute ya es no-None al llamar
# `_load_ocrmac_module()` (o sea, alguien lo seteó), lo devolvemos tal
# cual sin re-importar.
#
# `_ocrmac_import_attempted` guardia contra re-intentos de import fallidos
# — si pyobjc rompe una vez, rompe siempre hasta reboot, no sentido en
# pagar el ImportError cada llamada.
_ocrmac_module = None  # type: ignore[assignment]
_ocrmac_import_attempted = False


def _load_ocrmac_module():
    """Devuelve el módulo `ocrmac.ocrmac` o None si el import falla.

    Lazy + cached: la primera llamada paga el import (~130ms macOS), las
    subsiguientes retornan el singleton. Idempotente ante fallo — una
    vez que `_ocrmac_import_attempted` queda en True, ni re-intenta ni
    re-loggea. Preserva el contract de tests que hacen monkeypatch de
    `rag._ocrmac_module`: si ya está seteado (por test o por carga
    previa), no lo sobrescribe.

    State lives on the `rag` package namespace (not on `rag.ocr`) so
    `monkeypatch.setattr(rag, "_ocrmac_module", fake)` in tests is the
    source of truth — `_rag._ocrmac_module` reads the patched value, and
    the loader writes back via the same attribute when it succeeds.
    """
    import rag as _rag
    if _rag._ocrmac_module is not None:
        return _rag._ocrmac_module
    if _rag._ocrmac_import_attempted:
        return None
    _rag._ocrmac_import_attempted = True
    try:
        from ocrmac import ocrmac as _mod
        _rag._ocrmac_module = _mod
    except Exception:  # noqa: BLE001 — pyobjc init puede fallar, no solo ImportError
        _rag._ocrmac_module = None  # type: ignore[assignment]
    return _rag._ocrmac_module


# Extensiones de imagen soportadas (Apple Vision handlea todas via Core Image).
_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".gif", ".bmp", ".tiff", ".tif",
})

# Min confidence para considerar un texto OCR. ocrmac devuelve [0,1] por
# annotation. Umbral bajo (0.3) porque para tablas / screenshots la
# confianza real suele ser 0.5+ pero algunas palabras de una o dos letras
# bajan; preferimos falsos-positivos leves (ruido) a pérdida de señal
# (tabla de números ej. "10.54" es crítica y puede scorear 0.5-0.7).
_OCR_MIN_CONFIDENCE = 0.3

# Regex para extraer embeds de imagen. Dos formatos:
#   - Wikilink Obsidian:  ![[folder/image.png]]  o  ![[image.png|alias]]
#   - Markdown estándar:  ![alt text](path/to/image.png)
_EMBED_WIKILINK_RE = re.compile(r"!\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]")
_EMBED_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def _extract_embedded_images(body: str, note_path: Path, vault_root: Path) -> list[Path]:
    """Parse `body` para encontrar imágenes embebidas y devolver paths
    absolutos (solo las que existen en disco). Formatos soportados:
    wikilink Obsidian (`![[img.png]]`) y markdown estándar (`![alt](p.png)`).

    Resolución de paths (en orden):
      1. URL externa (http/https/data:) → skip.
      2. Path absoluto → se usa tal cual si existe.
      3. Path relativo a `note_path.parent` → se prueba ahí primero.
      4. Solo nombre de archivo (ej. `![[captura.png]]`) → scan del vault
         entero por filename match (Obsidian default behavior cuando no
         hay path).

    Ignora:
      - Extensiones no-imagen (`.md`, `.pdf`, `.canvas`, ...) — la función
        es específica para OCR, no para todos los embeds.
      - Imágenes referenciadas pero no encontradas en disco (broken
        links) — silent skip.
      - URLs externas (no fetcheamos; la política es 100% local).
    """
    if not body:
        return []
    out: list[Path] = []
    seen: set[Path] = set()

    def _resolve_and_add(candidate: str) -> None:
        # Skip externals.
        low = candidate.strip().lower()
        if not low or low.startswith(("http://", "https://", "data:", "ftp://")):
            return
        # Skip non-image extensions (embeds de notas/pdfs/canvas no aplican).
        ext = Path(candidate).suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            return
        # Absolute path.
        p = Path(candidate)
        if p.is_absolute():
            if p.is_file() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
            return
        # Relative to note parent.
        rel = (note_path.parent / candidate).resolve()
        if rel.is_file() and rel not in seen:
            seen.add(rel)
            out.append(rel)
            return
        # Filename-only wikilink: scan vault.
        if "/" not in candidate and "\\" not in candidate:
            try:
                matches = list(vault_root.rglob(candidate))
            except OSError:
                matches = []
            for m in matches:
                if m.is_file() and m.resolve() not in seen:
                    seen.add(m.resolve())
                    out.append(m)
                    return

    for m in _EMBED_WIKILINK_RE.finditer(body):
        _resolve_and_add(m.group(1))
    for m in _EMBED_MARKDOWN_RE.finditer(body):
        _resolve_and_add(m.group(1))
    return out


def _ocr_image(image_path: Path) -> str:
    """OCR `image_path` usando Apple Vision (ocrmac). Resultado cacheado
    en `rag_ocr_cache` SQL table, key = abs path, invalidación por mtime.

    Returns `""` (no crash) en cualquiera de estos casos:
      - env `RAG_OCR=0` — usuario desactivó OCR explícitamente.
      - `_ocrmac_module is None` — import falló (non-macOS / pyobjc roto).
      - imagen no existe o stat falla.
      - ocrmac lanza — imagen corrupta, formato raro, memory issue.

    Filtra annotations por confianza ≥ `_OCR_MIN_CONFIDENCE` (0.3). El
    texto resultante es la concatenación space-separated de todas las
    annotations que pasaron el umbral — suficiente para que el chunker
    downstream (que tokeniza por whitespace) encuentre matches.
    """
    from rag import _silent_log, _ragvec_state_conn
    if os.environ.get("RAG_OCR", "").strip() == "0":
        return ""
    ocrmac_mod = _load_ocrmac_module()
    if ocrmac_mod is None:
        return ""
    try:
        mtime = image_path.stat().st_mtime
    except OSError:
        # Audit 2026-04-26 (MEDIUM): si el archivo desapareció, dropear
        # el cache row stale para que el chunk no embeba OCR text
        # fantasma. Pre-fix el cache quedaba forever.
        try:
            abs_gone = str(image_path.resolve()) if image_path else ""
            if abs_gone:
                with _ragvec_state_conn() as conn:
                    conn.execute(
                        "DELETE FROM rag_ocr_cache WHERE image_path = ?",
                        (abs_gone,),
                    )
        except Exception:
            pass
        return ""
    abs_key = str(image_path.resolve())

    # Cache read.
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT mtime, text FROM rag_ocr_cache WHERE image_path = ?",
                (abs_key,),
            ).fetchone()
            if row is not None and abs(row[0] - mtime) < 1e-6:
                return row[1] or ""
    except Exception as exc:
        _silent_log("ocr_cache_read", exc)
        # Continuamos al OCR real si la cache falla — no bloqueamos el
        # indexer por problemas de estado.

    # OCR real con timeout 30s (audit 2026-04-26 MEDIUM: pre-fix sin
    # timeout, una imagen corrupta o un Apple Vision hang bloqueaba al
    # indexer thread indefinidamente — observado hasta 30s en
    # screenshots complejos, teóricamente unbounded).
    try:
        import concurrent.futures as _cf  # noqa: PLC0415
        def _do_ocr() -> list:
            return ocrmac_mod.OCR(
                abs_key, language_preference=["es-ES", "en-US"],
            ).recognize()
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(_do_ocr)
            try:
                annotations = _fut.result(timeout=30.0)
            except _cf.TimeoutError:
                _silent_log(f"ocr_timeout:{abs_key}", "Vision hang >30s")
                return ""
    except Exception as exc:
        _silent_log(f"ocr_recognize:{abs_key}", exc)
        return ""

    # annotations = [(text, confidence, bbox), ...]. Filtramos por umbral
    # y concatenamos. Separador ` ` (no `\n`) porque el chunker split
    # por whitespace lo mismo, y keep el tamaño compacto.
    texts = [
        t for (t, c, _bbox) in annotations
        if isinstance(t, str) and t and c >= _OCR_MIN_CONFIDENCE
    ]
    out = " ".join(texts).strip()

    # Cache write (best-effort; no rompe el indexer si falla).
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_ocr_cache "
                "(image_path, mtime, text, ocr_at) VALUES (?, ?, ?, ?)",
                (abs_key, float(mtime), out, time.time()),
            )
    except Exception as exc:
        _silent_log(f"ocr_cache_write:{abs_key}", exc)

    return out


# ── VLM caption fallback ────────────────────────────────────────────────────
#
# Hay imágenes en el vault que NO tienen texto (o tienen muy poquito) pero sí
# información visual relevante — fotos familiares, diagramas de arquitectura,
# whiteboards dibujados, gráficos de torta, screenshots puramente gráficos,
# paisajes, flyers de eventos en imagen. Con OCR solo, esas imágenes son
# invisibles al retrieval aunque la nota contenedora las describa parcialmente
# en texto.
#
# La solución: cuando OCR devuelve vacío (o menos de `_VLM_FALLBACK_MIN_OCR`
# chars), corremos un modelo vision-language local (granite-vision-3.2-2b vía
# mlx-vlm) con un prompt que pide descripción grep-friendly + transcripción de
# texto visible. El caption resultante se concatena al body igual que el OCR,
# pero con un marker distinto (`<!-- VLM-caption: -->`) para que sea grepable.
#
# Cache: tabla `rag_vlm_captions` keyed por `(abs_path, mtime)` — misma
# invariante que `rag_ocr_cache`. El hash del chunk (`_file_hash_with_images`)
# suma mtimes de imágenes, así que una imagen nueva fuerza re-chunking aunque
# el .md no haya cambiado — y por mtime invalidation, también fuerza
# re-caption.
#
# Silent-fail total:
#   - `RAG_VLM_CAPTION=0` → feature off, wrapper devuelve lo que dé OCR.
#   - mlx-vlm falla al cargar / generar → return "" (el OCR text gana si había).
#   - Budget per-run excedido → return "" sin llamar al modelo (safety net
#     contra loops infinitos o primer indexing de vaults gigantes).
#
# Costo real a considerar: granite-vision-3.2-2b ocupa ~3 GB en MPS VRAM,
# ~1-3s por imagen warm. Primera corrida de `rag index --reset` sobre un
# vault con 500 imágenes sin OCR = ~20 min. Por eso
# `RAG_VLM_CAPTION_MAX_PER_RUN=500` default — el cap previene que un bug
# o un corpus inesperadamente grande se coma el día. Override con la var env.

VLM_MODEL = os.environ.get("RAG_VLM_MODEL", "").strip() or "mlx-community/granite-vision-3.2-2b-4bit"

# Chars mínimos de OCR para NO hacer fallback al VLM. 20 = threshold sano
# empíricamente: menos que esto suele ser ruido de OCR ("OK", "x", "•"), más
# es probable que sea texto real valioso que no necesita caption encima.
_VLM_FALLBACK_MIN_OCR = 20

# Longitud máxima del caption post-procesado. 500 chars = 1-2 oraciones
# descriptivas sin inflar el chunk. El prompt pide <80 palabras pero algunos
# modelos ignoran el cap — truncamos por las dudas.
_VLM_CAPTION_MAX_CHARS = 500

# Budget per-process. Protege contra runaway loops y contra primer indexing
# de vault gigante. El counter es module-global (single-process assumption
# del indexer). Override con `RAG_VLM_CAPTION_MAX_PER_RUN`.
_VLM_CAPTION_MAX_PER_RUN = int(os.environ.get("RAG_VLM_CAPTION_MAX_PER_RUN", "500"))
_vlm_caption_calls_used: int = 0

# Singleton mlx-vlm model + processor. Protected by _VLM_LOCK.
_VLM_MODEL_OBJ: object | None = None
_VLM_PROCESSOR: object | None = None
_VLM_LOCK = threading.Lock()
_VLM_LAST_USED: float = 0.0


def _vlm_caption_enabled() -> bool:
    """True salvo `RAG_VLM_CAPTION=0/false/no` explícito. Default ON."""
    val = os.environ.get("RAG_VLM_CAPTION", "").strip().lower()
    return val not in ("0", "false", "no")


def _vlm_load() -> tuple[object, object]:
    """Lazy-load granite-vision via mlx-vlm. Singleton guardado en módulo."""
    global _VLM_MODEL_OBJ, _VLM_PROCESSOR, _VLM_LAST_USED
    import rag as _rag
    hf_id = _rag.VLM_MODEL
    with _VLM_LOCK:
        if _VLM_MODEL_OBJ is None:
            from mlx_vlm import load as _mlx_load  # noqa: PLC0415
            _VLM_MODEL_OBJ, _VLM_PROCESSOR = _mlx_load(hf_id)
        _VLM_LAST_USED = time.time()
        return _VLM_MODEL_OBJ, _VLM_PROCESSOR


def _vlm_idle_unload(idle_seconds: float = 600) -> bool:
    """Evicta el modelo si lleva más de `idle_seconds` sin usarse. Libera ~3 GB MPS."""
    global _VLM_MODEL_OBJ, _VLM_PROCESSOR, _VLM_LAST_USED
    if _VLM_MODEL_OBJ is None:
        return False
    if time.time() - _VLM_LAST_USED < idle_seconds:
        return False
    with _VLM_LOCK:
        _VLM_MODEL_OBJ = None
        _VLM_PROCESSOR = None
        try:
            import mlx.core as mx  # noqa: PLC0415
            mx.clear_cache()
        except Exception:
            pass
    return True


def _vlm_describe(image_path: "str | Path", prompt: str = "") -> str:
    """Caption via mlx-vlm (granite-vision). Silent-fail → "" en cualquier error.

    NOTA: `mlx_vlm.prompt_utils.apply_chat_template` NO soporta `granite_vision`
    (lanza `ValueError: Unsupported model: granite_vision` en mlx-vlm 0.4.4).
    Usamos `processor.tokenizer.apply_chat_template` con la estructura HF
    multi-content `[{"role": "user", "content": [{"type": "image"}, {"type": "text", ...}]}]`,
    que el chat template de granite (en `tokenizer_config.json`) renderea
    como `<|system|>\\n...\\n<|user|>\\n<image>\\n{prompt}\\n<|assistant|>\\n`.
    """
    from mlx_vlm import generate as _mlx_generate  # noqa: PLC0415
    # Serializar el forward MLX con el resto del proceso (embedder, chat,
    # helper) vía `_MLX_FORWARD_LOCK`. Sin esto, un caption disparado durante
    # `_flush_batch()` del indexer colisiona Metal command buffers con el
    # forward del embedder Qwen3 → `kIOGPUCommandBufferCallbackErrorHang` /
    # `InnocentVictim` reproducible. Mismo patrón que `MLXEmbedder._encode_batch`
    # (rag/mlx_embed.py:187) y `MLXBackend.chat` (rag/llm_backend.py).
    from rag.llm_backend import _MLX_FORWARD_LOCK  # noqa: PLC0415
    actual_prompt = prompt or _VLM_CAPTION_PROMPT
    try:
        model, processor = _vlm_load()
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": actual_prompt},
            ]},
        ]
        formatted = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        with _MLX_FORWARD_LOCK:
            out = _mlx_generate(
                model, processor, formatted,
                image=[str(image_path)],
                verbose=False,
                max_tokens=256,
            )
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        return (text or "").strip()
    except Exception as exc:
        if os.environ.get("RAG_DEBUG"):
            import sys as _sys
            _sys.stderr.write(f"[ocr-vlm] caption failed: {exc}\n")
        return ""


def _vlm_caption_budget_reset() -> None:
    """Reinicia el contador de llamadas. Llamado por el CLI al arrancar
    comandos de indexing (`rag index`, `rag watch`, `rag scan-citas`) para
    que cada run tenga su propio budget limpio. Tests también resetean
    entre casos para aislar."""
    global _vlm_caption_calls_used
    _vlm_caption_calls_used = 0


def _vlm_caption_budget_available() -> bool:
    # Resolve via `rag` so tests can monkeypatch.setattr(rag, "_VLM_CAPTION_MAX_PER_RUN", N).
    import rag as _rag
    return _vlm_caption_calls_used < _rag._VLM_CAPTION_MAX_PER_RUN


def _vlm_caption_budget_consume() -> None:
    global _vlm_caption_calls_used
    _vlm_caption_calls_used += 1


# Prompt del VLM. Español rioplatense-neutral, optimizado para retrieval:
# pide (a) transcripción de texto visible si lo hay, (b) descripción de
# escena grep-friendly, (c) nombres propios y fechas si aparecen. El
# formato es prosa libre sin markdown porque el chunker normaliza eso.
_VLM_CAPTION_PROMPT = (
    "Describí qué hay en esta imagen en 1-2 oraciones (≤80 palabras). "
    "Si hay texto visible en la imagen (letreros, títulos, nombres, "
    "fechas, números), transcribilo literal al principio. Si es una "
    "foto, mencioná objetos/personas/escena; si es un diagrama, qué "
    "representa; si es una captura de UI, qué app/pantalla. "
    "Sé específico — evitá 'una imagen' o 'una foto'. "
    "Sin markdown, sin comillas, sin preámbulos. "
    "Respondé en español."
)


# Prompt especializado para recibos / facturas / tickets (VLM #10 task).
# El VLM emite JSON estructurado parseable. Formato fijo para que el caller
# extraiga total/merchant/date programáticamente sin LLM judge encima.
# El prompt acepta cualquier formato (papel, PDF screenshot, app móvil) y
# devuelve "null" en campos que no encuentre — evita hallucination por
# campo obligatorio. El parser tolera JSON con prefijos/sufijos no-JSON
# (markdown fences, prosa de "acá tenés:" delante) — recortar por la 1ra `{`.
_VLM_RECEIPT_PROMPT = (
    "Extraé los datos de este recibo / factura / ticket como JSON. "
    "Campos obligatorios:\n"
    '  - merchant: nombre del comercio (string)\n'
    '  - date: fecha en formato YYYY-MM-DD (string)\n'
    '  - total: monto total como número (number, sin símbolo)\n'
    '  - currency: moneda ISO 4217 ("ARS", "USD", "EUR", etc.) (string)\n'
    '  - items: lista de items con {description, quantity, price} (array)\n'
    '  - category: una de "food", "transport", "shopping", "services", '
    '"entertainment", "health", "other" (string)\n\n'
    "Devolvé SOLO el JSON, sin markdown fences, sin prosa antes ni después. "
    "Si un campo no se ve claramente, usá null. NO inventes datos. "
    "Si la imagen NO es un recibo, devolvé exactamente: "
    '{"error": "not_a_receipt"}'
)


# Prompt especializado para gráficos / charts / dashboards. Caption-style
# (no JSON) pero chart-aware: identifica tipo (bar, line, pie, scatter),
# ejes, rango de valores, y la métrica principal con su valor. Útil para
# screenshots de Grafana/dashboards/papers/blogs que el caption genérico
# describe muy alto-nivel. Output: prosa española corta optimizada para
# grep ("max 25K en marzo", "tendencia descendente Q4 2026").
_VLM_CHART_PROMPT = (
    "Esta imagen es un gráfico, chart o dashboard. Describilo en 2-3 "
    "oraciones (≤120 palabras) cubriendo:\n"
    "  - Tipo: bar / line / pie / scatter / area / heatmap / dashboard.\n"
    "  - Qué representa (título o métrica principal si está visible).\n"
    "  - Eje X y eje Y (qué dimensiones / unidades / rango).\n"
    "  - Valores clave: máximo, mínimo, tendencia, el data point más "
    "    notable. Transcribí los números literales que veas.\n"
    "  - Si hay múltiples series, nombrarlas.\n\n"
    "Sé específico con números y nombres. Si NO es un gráfico, devolvé: "
    "'no es un gráfico'. Sin markdown, sin preámbulos. Español."
)


def _caption_image(image_path: Path) -> str:
    """VLM caption de `image_path` — fallback cuando el OCR devolvió poco.

    Returns caption string (máx `_VLM_CAPTION_MAX_CHARS` chars) o "" en
    cualquiera de estos casos (silent-fail):
      - `RAG_VLM_CAPTION=0` — feature off.
      - stat falla (imagen no existe o sin permisos).
      - budget per-run excedido.
      - mlx-vlm falla al cargar o generar.
      - response vacío o mal-formed.

    Cache: `rag_vlm_captions` con key `(abs_path, mtime)`. Re-correr sobre
    la misma imagen es O(1) SQL lookup — NO vuelve a llamar al modelo.
    """
    from rag import _silent_log, _ragvec_state_conn
    import rag as _rag
    if not _vlm_caption_enabled():
        return ""
    try:
        mtime = image_path.stat().st_mtime
    except OSError:
        return ""
    abs_key = str(image_path.resolve())

    # Cache read.
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT mtime, caption FROM rag_vlm_captions WHERE image_path = ?",
                (abs_key,),
            ).fetchone()
            if row is not None and abs(row[0] - mtime) < 1e-6:
                return row[1] or ""
    except Exception as exc:
        _silent_log("vlm_caption_cache_read", exc)

    # Budget gate — DESPUÉS del cache read (cache hits no cuentan contra
    # el budget, solo las invocaciones reales al modelo).
    if not _vlm_caption_budget_available():
        return ""

    # VLM call via mlx-vlm. Resolve via `rag` so tests can
    # monkeypatch.setattr(rag, "_vlm_describe", stub).
    raw = _rag._vlm_describe(abs_key)

    # Budget se consume SOLO cuando efectivamente llamamos al modelo.
    _vlm_caption_budget_consume()

    # Normalización del output.
    caption = (raw or "").strip()
    caption = caption.strip("`\"' \n\r\t").replace("\n\n", " ").replace("\n", " ")
    caption = caption[:_VLM_CAPTION_MAX_CHARS]

    model_id = _rag.VLM_MODEL
    # Cache write (también cacheamos captions vacíos — así no re-intentamos
    # en cada index run una imagen que el VLM no supo captionar).
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_vlm_captions "
                "(image_path, mtime, caption, model, captioned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (abs_key, float(mtime), caption, model_id, time.time()),
            )
    except Exception as exc:
        _silent_log(f"vlm_caption_cache_write:{abs_key}", exc)

    return caption


# ── VLM #10: Receipt parser + Chart describer ────────────────────────────────


def _extract_json_object(text: str) -> str | None:
    """Recorta un objeto JSON balanceado del primer `{` al `}` que cierra.
    Tolera markdown fences (```json ... ```), prosa antes/después, comillas
    escapadas dentro de strings. Devuelve None si no hay JSON parseable.
    """
    if not text:
        return None
    s = text.strip()
    # Strip markdown code fences si están
    if s.startswith("```"):
        # ```json\n...\n``` o ```\n...\n```
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3].rstrip()
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _vlm_parse_receipt(image_path: "str | Path") -> dict | None:
    """Parse recibo / factura / ticket via VLM → dict estructurado.

    Devuelve dict con shape `{merchant, date, total, currency, items, category}`
    o `None` si:
      - VLM falla / vacío.
      - JSON no parseable.
      - VLM detectó que NO es recibo (`{"error": "not_a_receipt"}` se
        normaliza a `None` con flag interno).
      - cualquier excepción.

    Silent-fail siempre. NO eleva. Para debugging: `RAG_DEBUG=1`.

    Cache: `rag_vlm_receipts` con key `(abs_path, mtime)`. Re-correr sobre
    la misma imagen es O(1) SQL lookup. Cache invalidates cuando mtime
    cambia (re-foto del recibo, edit, rotate).
    """
    import json as _json
    from rag import _silent_log, _ragvec_state_conn  # noqa: PLC0415
    import rag as _rag  # noqa: PLC0415

    if not _vlm_caption_enabled():
        return None
    p = Path(str(image_path))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    abs_key = str(p.resolve())

    # Cache read.
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT mtime, parsed_json FROM rag_vlm_receipts WHERE image_path = ?",
                (abs_key,),
            ).fetchone()
            if row is not None and abs(row[0] - mtime) < 1e-6:
                cached = row[1] or ""
                if not cached:
                    return None
                try:
                    obj = _json.loads(cached)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    except Exception as exc:
        _silent_log("vlm_receipt_cache_read", exc)

    if not _vlm_caption_budget_available():
        return None

    raw = _rag._vlm_describe(abs_key, prompt=_VLM_RECEIPT_PROMPT)
    _vlm_caption_budget_consume()

    parsed: dict | None = None
    json_blob = _extract_json_object(raw or "")
    if json_blob:
        try:
            obj = _json.loads(json_blob)
            if isinstance(obj, dict):
                if obj.get("error") == "not_a_receipt":
                    parsed = None
                else:
                    parsed = obj
        except Exception as exc:
            _silent_log(f"vlm_receipt_json_parse:{abs_key}", exc)

    # Cache write — siempre (incluso None) para no re-llamar al modelo.
    cached_value = _json.dumps(parsed, ensure_ascii=False) if parsed else ""
    model_id = _rag.VLM_MODEL
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_vlm_receipts "
                "(image_path, mtime, parsed_json, model, parsed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (abs_key, float(mtime), cached_value, model_id, time.time()),
            )
    except Exception as exc:
        _silent_log(f"vlm_receipt_cache_write:{abs_key}", exc)

    return parsed


def _vlm_describe_chart(image_path: "str | Path") -> str:
    """Caption chart-aware via VLM. Prompt especializado pide tipo + ejes +
    valores clave. Uso: screenshots de Grafana, dashboards, gráficos en
    papers/blogs. Devuelve string normalizado o "" si VLM falla / detectó
    que no es chart.

    NO usa cache propio — reusa `rag_vlm_captions` con un marker en el
    caption mismo (`[chart]` prefix) para diferenciar de captions genéricos
    si el caller reusa la misma imagen para ambos. Idempotente.

    Silent-fail siempre.
    """
    from rag import _silent_log  # noqa: PLC0415
    import rag as _rag  # noqa: PLC0415
    if not _vlm_caption_enabled():
        return ""
    if not _vlm_caption_budget_available():
        return ""
    try:
        raw = _rag._vlm_describe(str(image_path), prompt=_VLM_CHART_PROMPT)
    except Exception as exc:
        _silent_log(f"vlm_chart:{image_path}", exc)
        return ""
    _vlm_caption_budget_consume()
    cap = (raw or "").strip()
    cap = cap.strip("`\"' \n\r\t").replace("\n\n", " ").replace("\n", " ")
    if cap.lower() in ("no es un gráfico", "no es un grafico", "no es chart"):
        return ""
    return cap[:_VLM_CAPTION_MAX_CHARS]


def _image_text_or_caption(image_path: Path) -> tuple[str, str]:
    """Wrapper unificado: devuelve `(text, source)` donde source es uno de:
      - `"ocr"`: el texto viene de Apple Vision (había texto reconocible).
      - `"vlm"`: OCR fue vacío/poco → fallback al VLM para captionear.
      - `""`: ni OCR ni VLM aportaron nada (imagen sin señal).

    Orden: OCR primero (barato, determinístico, local). Si el OCR devuelve
    menos de `_VLM_FALLBACK_MIN_OCR` chars, se intenta VLM. Si VLM
    también vacío, devolvemos lo que dé OCR (que puede ser "" o un
    fragmento corto) preferentemente como `"ocr"` — el OCR corto todavía
    tiene más chance de ser verdadero texto de la imagen que un caption
    inventado.

    Es el punto de entrada que deben usar los callers (`_enrich_body_with_ocr`,
    `rag scan-citas`, `rag capture --image`, WA ingester). NUNCA raise.

    Both `_ocr_image` and `_caption_image` are resolved through `rag` so test
    monkey-patches (`monkeypatch.setattr(rag, "_ocr_image", ...)` and
    `monkeypatch.setattr(rag, "_caption_image", ...)`) propagate here.
    """
    from rag import _silent_log
    import rag as _rag
    try:
        ocr_text = _rag._ocr_image(image_path)
    except Exception as exc:
        _silent_log(f"image_text_ocr:{image_path}", exc)
        ocr_text = ""

    if ocr_text and len(ocr_text.strip()) >= _VLM_FALLBACK_MIN_OCR:
        return ocr_text, "ocr"

    # Fallback VLM — solo si está habilitado.
    try:
        caption = _rag._caption_image(image_path)
    except Exception as exc:
        _silent_log(f"image_text_vlm:{image_path}", exc)
        caption = ""

    if caption:
        return caption, "vlm"
    if ocr_text:
        # OCR corto pero no vacío — preferible a nada.
        return ocr_text, "ocr"
    return "", ""


def _enrich_body_with_ocr(
    body: str, note_path: Path, vault_root: Path,
    images_source: str | None = None,
) -> str:
    """Retorna `body` extendido con el texto OCR de sus imágenes embebidas.

    Para cada imagen encontrada por `_extract_embedded_images`, corre
    `_ocr_image` y concatena el texto extraído al final del body con un
    marker HTML-comment que identifica la imagen (user-grep-friendly,
    no visible en render de Markdown).

    `images_source`: opcional, el texto desde el cual extraer embeds. Default
    = `body`. Usado por el indexer para extraer del `raw` (pre-`clean_md`)
    pero apendear al `text` (post-`clean_md`) — `clean_md` convierte los
    `![[img.png]]` a `!img.png` (matchea la misma regex de wikilinks) y
    eso pierde el embed. Pasando el `raw` como `images_source`, el parser
    sigue viendo los `![[...]]` intactos.

    Imágenes cuyo OCR devuelve "" (cache miss + OCR empty, ocrmac
    desactivado, imagen vacía) se skippean sin agregar marker — evita
    contaminar el body con placeholders vacíos.

    Retorna `body` intacto si no hay imágenes embebidas.
    """
    from rag import _silent_log
    src = images_source if images_source is not None else body
    images = _extract_embedded_images(src, note_path, vault_root)
    if not images:
        return body
    parts: list[str] = [body]
    for img in images:
        # Wrapper unificado: OCR primero, fallback a VLM si OCR vacío/corto.
        # El `source` ∈ {"ocr", "vlm", ""} gobierna el marker que elegimos
        # abajo — importante para grep/debug y para distinguir texto
        # literal (OCR) de descripción inferida (VLM).
        # Resolve via `rag` so test monkey-patches on
        # `_image_text_or_caption` and `_maybe_create_cita_from_ocr` propagate.
        import rag as _rag
        text, source = _rag._image_text_or_caption(img)
        if not text:
            continue
        # Marker: path relativo al vault si posible (más legible), fallback
        # al filename. HTML-comment style porque markdown lo oculta en
        # renders de Obsidian — el user no lo ve a menos que mire el source.
        try:
            rel_marker = str(img.resolve().relative_to(vault_root.resolve()))
        except ValueError:
            rel_marker = img.name
        marker = "VLM-caption" if source == "vlm" else "OCR"
        parts.append(f"\n\n<!-- {marker}: {rel_marker} -->\n{text}")
        # Cita-from-image detector hook. Corre DESPUÉS del append para
        # que el body enrichment siempre gane — si el detector crashea o
        # el helper está caído, el OCR/caption sigue llegando al chunker
        # tal cual. Silent-fail triple: la función ya es silent, pero
        # envolvemos en try/except por las dudas. El detector es
        # source-agnóstico: un caption VLM que dice "flyer cumple Flor 26
        # de mayo" clasifica igual de bien que el OCR literal del mismo
        # flyer.
        try:
            _rag._maybe_create_cita_from_ocr(text, img, source="index")
        except Exception as exc:
            _silent_log(f"cita_detect_enrich:{img}", exc)
    return "".join(parts)


def _file_hash_with_images(raw: str, note_path: Path, vault_root: Path) -> str:
    """Hash del archivo + firmas mtime de sus imágenes embebidas.

    Sin esto, el indexer skippea reindex cuando el .md no cambió pero
    una imagen embebida sí (caso típico: tomé una screenshot nueva, la
    guardé con el mismo nombre, pero la nota contenedora no cambió). El
    OCR cache está invalidado por mtime pero el indexer nunca llegaría
    a re-chunkear. Combinando los mtimes de las imágenes al hash base,
    un cambio en cualquier imagen fuerza reindex de la nota.
    """
    from rag import file_hash
    base = file_hash(raw)
    images = _extract_embedded_images(raw, note_path, vault_root)
    if not images:
        return base
    sig_parts: list[str] = [base]
    for p in sorted(images, key=lambda x: str(x)):
        try:
            sig_parts.append(f"{p}:{p.stat().st_mtime}")
        except OSError:
            sig_parts.append(f"{p}:missing")
    return file_hash("\n".join(sig_parts))



# ── OCR → cita intent detector (re-export) ───────────────────────────────────
# Movido a `rag/ocr_cita_detector.py` (2026-05-09). Re-exportado para
# preservar `from rag.ocr import _maybe_create_cita_from_ocr` /
# `_detect_cita_from_ocr` etc. y los monkeypatches que usan
# `rag._detect_cita_from_ocr` (re-resueltos via sys.modules a call-time).
from rag.ocr_cita_detector import (  # noqa: F401, E402
    _CITA_MIN_CHARS,
    _CITA_MIN_CONFIDENCE,
    _CITA_PROMPT_SYSTEM,
    _CITA_PROMPT_USER_TEMPLATE,
    _CITA_VALID_KINDS,
    _DETECTOR_TIMEOUT,
    _cita_detect_enabled,
    _cita_result,
    _detect_cita_from_ocr,
    _maybe_create_cita_from_ocr,
    _normalize_ocr_for_hash,
    _ocr_hash_key,
    _persist_cita_detection,
)
