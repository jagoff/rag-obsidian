"""OCR + VLM caption + intent detector — extracted from `rag/__init__.py` (Phase 6 of monolith split, 2026-04-25).

Three concerns bundled into one module because they form a single pipeline:

1. **OCR primitives** (Apple Vision via `ocrmac`): walk a markdown body
   for embedded images, run Apple's on-device OCR, cache results in
   `rag_ocr_cache`. Used by the indexer to make image-heavy notes
   (link-hubs, screenshots, whiteboards) searchable.
2. **VLM caption fallback** (qwen2.5vl:3b via ollama): when OCR returns
   < `_VLM_FALLBACK_MIN_OCR` chars, run a vision-language model that
   produces a grep-friendly caption. Cache in `rag_vlm_captions`. Per-run
   budget cap so a fresh index doesn't burn 5+ min on captions.
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
- `_vlm_caption_enabled()`, `_vlm_client()`,
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
- `rag._ocrmac_module`, `rag._ocr_image`, `rag._vlm_client`,
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
heavy `ocrmac` / `ollama` imports are still lazy (inside `_load_ocrmac_module`
and `_vlm_client` respectively) — module load stays fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import ollama

# NOTE: helpers from `rag.__init__` are imported INSIDE each function body
# (deferred) — module-level `from rag import _helper_client, propose_*, ...`
# would capture the binding at module-load time and miss subsequent test
# monkey-patches (`monkeypatch.setattr(rag, "_helper_client", stub)`).
# `from rag import X` inside a function re-resolves `rag.X` on every call.
#
# Same for monkey-patched names (`_ocrmac_module`, `_ocr_image`,
# `_vlm_client`, `_VLM_CAPTION_MAX_PER_RUN`, `_detect_cita_from_ocr`):
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
# chars), corremos un modelo vision-language local (qwen2.5vl:3b vía ollama)
# con un prompt que pide descripción grep-friendly + transcripción de texto
# visible. El caption resultante se concatena al body igual que el OCR, pero
# con un marker distinto (`<!-- VLM-caption: -->`) para que sea grepable.
#
# Cache: tabla `rag_vlm_captions` keyed por `(abs_path, mtime)` — misma
# invariante que `rag_ocr_cache`. El hash del chunk (`_file_hash_with_images`)
# suma mtimes de imágenes, así que una imagen nueva fuerza re-chunking aunque
# el .md no haya cambiado — y por mtime invalidation, también fuerza
# re-caption.
#
# Silent-fail total:
#   - `RAG_VLM_CAPTION=0` → feature off, wrapper devuelve lo que dé OCR.
#   - ollama no responde / timeout → return "" (el OCR text gana si había).
#   - Modelo no existe (usuario no corrió `ollama pull qwen2.5vl:3b`) →
#     hint en stderr UNA vez por proceso + return "".
#   - Budget per-run excedido → return "" sin llamar a ollama (safety net
#     contra loops infinitos o primer indexing de vaults gigantes).
#
# Costo real a considerar: qwen2.5vl:3b ocupa ~4 GB en RAM, ~2-4s por imagen
# en MPS. Primera corrida de `rag index --reset` sobre un vault con 500
# imágenes sin OCR = ~25 min. Por eso `RAG_VLM_CAPTION_MAX_PER_RUN=500`
# default — el cap previene que un bug o un corpus inesperadamente grande
# se coma el día. Override con la var env.

VLM_MODEL = os.environ.get("RAG_VLM_MODEL", "").strip() or "qwen2.5vl:3b"

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

# Warned-once set por nombre de modelo. Evita spam en stderr cuando el
# usuario no tiene el VLM pulled y cada imagen falla con "model not found".
_vlm_model_missing_warned: set[str] = set()

# Cliente ollama dedicado para VLM. Timeout más alto (60s) que helper text
# porque qwen2.5vl en MPS tarda 2-4s warm / hasta 10s en cold-load.
_VLM_CLIENT: "ollama.Client | None" = None


def _vlm_caption_enabled() -> bool:
    """True salvo `RAG_VLM_CAPTION=0/false/no` explícito. Default ON."""
    val = os.environ.get("RAG_VLM_CAPTION", "").strip().lower()
    return val not in ("0", "false", "no")


def _vlm_client() -> "ollama.Client":
    """Lazy-init singleton. 60s timeout cubre cold-load del modelo (~10s en
    MPS primera vez) + inference típica (~2-4s)."""
    global _VLM_CLIENT
    if _VLM_CLIENT is None:
        _VLM_CLIENT = ollama.Client(timeout=60.0)
    return _VLM_CLIENT


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


def _caption_image(image_path: Path) -> str:
    """VLM caption de `image_path` — fallback cuando el OCR devolvió poco.

    Returns caption string (máx `_VLM_CAPTION_MAX_CHARS` chars) o "" en
    cualquiera de estos casos (silent-fail):
      - `RAG_VLM_CAPTION=0` — feature off.
      - stat falla (imagen no existe o sin permisos).
      - budget per-run excedido.
      - ollama timeout / unreachable.
      - modelo no pulled — imprime hint en stderr UNA vez por proceso.
      - response vacío o mal-formed.

    Cache: `rag_vlm_captions` con key `(abs_path, mtime)`. Re-correr sobre
    la misma imagen es O(1) SQL lookup — NO vuelve a llamar al modelo.
    """
    from rag import _silent_log, _ragvec_state_conn, OLLAMA_KEEP_ALIVE
    # Resolve via `rag` so tests can monkeypatch.setattr(rag, "_vlm_client", stub).
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
        # Seguimos — cache roto no bloquea VLM call.

    # Budget gate — DESPUÉS del cache read (cache hits no cuentan contra
    # el budget, solo las invocaciones reales al modelo).
    if not _vlm_caption_budget_available():
        return ""

    # VLM call.
    try:
        resp = _rag._vlm_client().chat(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": _VLM_CAPTION_PROMPT,
                "images": [abs_key],
            }],
            options={
                "temperature": 0,
                "seed": 42,
                "num_predict": 120,
                # num_ctx dejamos que el servidor elija — imagen + prompt
                # +  caption caben cómodo en el default de qwen2.5vl.
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except Exception as exc:
        # Detectamos "model not found" → hint one-shot. Otros errores
        # (timeout, network) silent-fail normal.
        msg = str(exc).lower()
        if ("not found" in msg or "pull" in msg) and VLM_MODEL not in _vlm_model_missing_warned:
            _vlm_model_missing_warned.add(VLM_MODEL)
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"\n[obsidian-rag] VLM caption skipped: modelo '{VLM_MODEL}' "
                    f"no está disponible en ollama. Corré:\n"
                    f"    ollama pull {VLM_MODEL}\n"
                    f"Para desactivar el caption fallback: "
                    f"export RAG_VLM_CAPTION=0\n\n"
                )
            except Exception:
                pass
        _silent_log(f"vlm_caption:{abs_key}", exc)
        return ""

    # Budget se consume SOLO cuando efectivamente llamamos al modelo (éxito
    # o respuesta vacía — no cuando falló el pull). Evita que un modelo
    # no-pulled queme budget en el primer intento y deje el resto sin
    # siquiera intentar.
    _vlm_caption_budget_consume()

    # Normalización del output.
    try:
        raw = resp.message.content or ""
    except Exception:
        raw = ""
    caption = raw.strip()
    # Strip markdown residual + comillas extra si el modelo ignoró el prompt.
    caption = caption.strip("`\"' \n\r\t").replace("\n\n", " ").replace("\n", " ")
    caption = caption[:_VLM_CAPTION_MAX_CHARS]

    # Cache write (también cacheamos captions vacíos — así no re-intentamos
    # en cada index run una imagen que el VLM no supo captionar).
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_vlm_captions "
                "(image_path, mtime, caption, model, captioned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (abs_key, float(mtime), caption, VLM_MODEL, time.time()),
            )
    except Exception as exc:
        _silent_log(f"vlm_caption_cache_write:{abs_key}", exc)

    return caption


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


# ── OCR → intent detector (event / reminder / note) ────────────────────────
#
# El OCR extrae texto de imágenes embebidas (Apple Vision, ver `_ocr_image`).
# Esa misma rama se extiende acá: el helper qwen2.5:3b clasifica el texto
# en tres kinds y extrae datos estructurados:
#
#   - `event`: cita, turno, reunión, cumple, vuelo — cosa con fecha y/o
#     hora específica que va al calendario. Dispara `propose_calendar_event`.
#   - `reminder`: tarea, to-do, factura a pagar, lista de compras, llamar
#     a X — acción a hacer, con o sin deadline. Dispara `propose_reminder`
#     con el path de la imagen en el campo `notes` (Apple Reminders no
#     soporta attachments vía AppleScript; el path en `notes` es lo más
#     cercano — queda como texto grep-friendly en la app).
#   - `note`: info sin acción — receta médica sin fecha, foto de código,
#     meme, captura de UI, texto de referencia. No-op: la nota OCR ya se
#     guardó en el vault (si el trigger fue `rag capture --image` o el
#     indexer), nada más para hacer.
#
# Triple salvaguarda contra dupes (preservada del diseño original):
#   1. Sidecar `rag_cita_detections` keyed por `sha256(normalized_ocr)[:16]` —
#      impide dos llamadas al helper para el mismo texto, aunque llegue por
#      rutas distintas (indexer, `rag capture --image`, `rag scan-citas`,
#      WhatsApp hook).
#   2. `_find_duplicate_calendar_event` adentro de `propose_calendar_event`
#      (segunda capa a nivel calendar real).
#   3. Confidence floor `_CITA_MIN_CONFIDENCE` descarta casos borderline.
#
# Silent-fail en todos los niveles: fallo de OCR, helper timeout, JSON
# malformado, sqlite lock, osascript error — ninguno propaga. El indexer y
# los ingesters siguen funcionando sin citas nuevas.
#
# Rollback: `export RAG_CITA_DETECT=0`. Con eso `_detect_cita_from_ocr`
# devuelve None de una (sin helper call) y `_maybe_create_cita_from_ocr`
# se vuelve no-op. OCR body enrichment sigue igual — es feature ortogonal.

# Umbral de confianza auto-create. qwen2.5:3b con temp=0 + seed=42 devuelve
# scores consistentes; 0.70 filtra ambigüedades pero pasa casos reales
# ("turno dentista miércoles 15hs consultorio Palermo" → 0.9+). Override
# por `rag scan-citas --min-confidence 0.5` para barridos agresivos.
_CITA_MIN_CONFIDENCE = 0.70

# OCR más corto que esto no tiene suficiente señal (ej. screenshot de un
# botón con "OK"). Skipeamos sin gastar helper call.
_CITA_MIN_CHARS = 20

# Kinds válidos del detector. `note` = no-op (solo se loggea). El resto
# dispara acción.
_CITA_VALID_KINDS = frozenset({"event", "reminder", "note"})

# Audit 2026-04-25 R2-OCR #1: timeout explícito del detector LLM (qwen2.5:3b).
# Sin esto, una llamada colgada bloquea el endpoint /api/chat/upload-image
# por minutos. El detector va a través de `rag._helper_client()`, que ya
# fija este mismo valor en `_TimedOllamaProxy(timeout=30.0)` —
# `_DETECTOR_TIMEOUT` documenta el contrato del lado consumidor (ocr.py)
# y permite testearlo sin importar implementación.
#
# 30s cubre cold-load de qwen2.5:3b (~10s en MPS) + inference típica
# (1-3s) con margen para retry interno del cliente. Si esto cambia,
# actualizar también `rag.__init__._helper_client()` para mantenerlos
# en sync (el test `test_detector_client_honors_timeout_contract` lo
# verifica).
_DETECTOR_TIMEOUT: float = 30.0


def _cita_detect_enabled() -> bool:
    """True salvo `RAG_CITA_DETECT=0/false/no` explícito. Default ON."""
    val = os.environ.get("RAG_CITA_DETECT", "").strip().lower()
    return val not in ("0", "false", "no")


def _normalize_ocr_for_hash(ocr_text: str) -> str:
    """Lowercase + whitespace-collapsed para que dos OCR passes sobre la
    misma imagen colisionen en el hash aunque ocrmac produzca orden de
    palabras distinto entre runs (raro, pero posible con tablas).
    """
    return " ".join((ocr_text or "").lower().split())


def _ocr_hash_key(ocr_text: str) -> str:
    """SHA256 (primeros 16 hex chars) del texto normalizado. PRIMARY KEY de
    `rag_cita_detections`. 16 chars = 64 bits, colisión accidental ~irreal
    para la cardinalidad esperada (≤ miles de imágenes por user).
    """
    norm = _normalize_ocr_for_hash(ocr_text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


_CITA_PROMPT_SYSTEM = (
    "Sos un clasificador de tareas agendables a partir de texto OCR "
    "crudo. El texto viene de una imagen — puede tener errores de "
    "reconocimiento (acentos perdidos, palabras cortadas, líneas "
    "mezcladas, chrome de app). Tu trabajo es decidir si el texto "
    "describe un EVENTO de calendario, un RECORDATORIO/tarea, o "
    "simplemente INFORMACIÓN sin acción agendable. Y extraer los datos."
)

_CITA_PROMPT_USER_TEMPLATE = (
    "Texto OCR de la imagen (puede estar ruidoso):\n"
    "<OCR>\n{ocr}\n</OCR>\n\n"
    "Devolvé JSON estricto, sin preámbulos, con estas llaves:\n"
    "  kind (str): 'event' | 'reminder' | 'note'\n"
    "    - 'event': cita, turno, reunión, cumple, vuelo, clase, "
    "entrevista — cosas con fecha Y/O hora específica que van al "
    "calendario (ej. 'Turno dentista martes 15hs', 'Cumple Flor 26 "
    "de mayo', 'Vuelo AR1234 15/06 08:20').\n"
    "    - 'reminder': tarea, to-do, factura a pagar, lista de compras, "
    "llamar/contactar a alguien, devolver algo, renovar documento — "
    "algo a HACER, con o sin deadline (ej. 'Pagar luz antes del 15', "
    "'Comprar huevos tomates pan', 'Llamar al plomero').\n"
    "    - 'note': información pura sin acción — receta médica sin "
    "fecha, foto de código, meme, captura de UI, texto de referencia, "
    "imagen decorativa (ej. 'Ibuprofeno 400mg cada 8hs', 'Error 500 "
    "stack trace', una foto familiar cualquiera).\n"
    "  title (str): título corto y descriptivo (<80 chars). Para event "
    "nombrá persona/servicio si aparece. Para reminder usá verbo + "
    "objeto ('Pagar luz', 'Comprar huevos'). Para note describí qué es.\n"
    "  when (str): fecha/hora/día en NL tal como aparece en el texto. "
    "'' si no hay. Usá lo que el dateparser entiende: 'martes 15hs', "
    "'15/05 10:00', 'mañana 14hs', 'el viernes', '26 de mayo', 'antes "
    "del 15'. NO inventes horario si no está en el texto.\n"
    "  location (str): dirección, consultorio, link de Zoom, negocio — "
    "relevante para event y a veces para reminder. '' si no aplica.\n"
    "  confidence (float 0.0–1.0): qué tan seguro estás. Alto (≥0.8) si "
    "los marcadores del kind son claros. Medio (0.5–0.8) si hay "
    "ambigüedad. Bajo (<0.5) si casi no hay pistas.\n\n"
    "Si NO está claro qué es, usá kind='note' con confidence baja.\n"
    "Si el texto tiene varios items, agarrá el más importante / "
    "próximo en el tiempo."
)


def _detect_cita_from_ocr(ocr_text: str) -> dict | None:
    """Helper call qwen2.5:3b con format=json: clasifica el texto OCR como
    event, reminder o note, y extrae {title, when, location}.

    Returns:
      - dict con shape `{kind, title, when, location, confidence}` —
        normalizado y validado. NUNCA raise.
      - None si: `RAG_CITA_DETECT=0`, ocr_text vacío o muy corto, helper
        timeout / unreachable, JSON malformado, shape inválida.

    Callers deben chequear `None` + `kind in {event, reminder}` +
    `confidence >= threshold` antes de crear algo — el helper a veces
    devuelve `kind='event'` con `confidence=0.3`, lo cual es una
    clasificación dudosa que no queremos auto-agendar.

    Backward-compat con el schema viejo `{is_cita, start}`: si el modelo
    (o un test monkeypatched) devuelve las keys viejas, las mapeamos:
    `is_cita=True` → `kind='event'`, `is_cita=False` → `kind='note'`,
    `start` → `when`. Eso hace que el upgrade sea transparente para
    callers externos.
    """
    from rag import _helper_client, _silent_log, HELPER_MODEL, HELPER_OPTIONS, OLLAMA_KEEP_ALIVE
    if not _cita_detect_enabled():
        return None
    text = (ocr_text or "").strip()
    if len(text) < _CITA_MIN_CHARS:
        return None
    # Cap para controlar el prompt size — 1500 chars cubre el 99% de
    # screenshots reales. Cortar el final es OK: el título / fecha suele
    # estar al principio del OCR (Apple Vision scanea top→bottom).
    capped = text[:1500]
    prompt = _CITA_PROMPT_USER_TEMPLATE.format(ocr=capped)
    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[
                {"role": "system", "content": _CITA_PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={**HELPER_OPTIONS, "num_predict": 180, "num_ctx": 2048},
            keep_alive=OLLAMA_KEEP_ALIVE,
            format="json",
        )
        raw = resp.message.content.strip()
        data = json.loads(raw)
    except Exception as exc:
        _silent_log("cita_detect_helper", exc)
        return None
    if not isinstance(data, dict):
        return None
    # Normalización defensiva — el helper a veces devuelve tipos raros.
    try:
        # Backward-compat: schema viejo usaba `is_cita` + `start`.
        kind_raw = data.get("kind")
        if kind_raw is None and "is_cita" in data:
            kind_raw = "event" if data.get("is_cita") else "note"
        kind = str(kind_raw or "note").strip().lower()
        if kind not in _CITA_VALID_KINDS:
            kind = "note"

        title = str(data.get("title") or "").strip()[:120]
        when_raw = data.get("when")
        if when_raw is None and "start" in data:
            when_raw = data.get("start")
        when = str(when_raw or "").strip()[:200]
        location = str(data.get("location") or "").strip()[:200]

        conf_raw = data.get("confidence")
        if isinstance(conf_raw, str):
            try:
                conf_raw = float(conf_raw)
            except ValueError:
                conf_raw = 0.0
        confidence = float(conf_raw or 0.0)
        confidence = max(0.0, min(1.0, confidence))
    except Exception as exc:
        _silent_log("cita_detect_normalize", exc)
        return None
    return {
        "kind": kind,
        "title": title,
        "when": when,
        "location": location,
        "confidence": confidence,
    }


def _maybe_create_cita_from_ocr(
    ocr_text: str,
    image_path: Path,
    source: str,
    *,
    min_confidence: float | None = None,
) -> dict | None:
    """Pipeline OCR → classifier → action (event / reminder / note) con
    sidecar dedup. Silent-fail en cada paso.

    Flujo por kind:
      - `kind="event"` + when parseable → `propose_calendar_event`. Persist
        decision="cita" con `event_uid` (o "duplicate" si ya estaba, o
        "error", o "ambiguous" si el parser de fecha no lo resolvió).
      - `kind="event"` + when="" → persist "ambiguous" sin crear.
      - `kind="reminder"` → `propose_reminder` (con o sin fecha). El path
        de la imagen se incluye en `notes` para que quede referenciado
        en Apple Reminders (no hay attachment API). Persist "reminder" +
        `reminder_id`.
      - `kind="note"` → no-op. Persist "note" para que re-runs skippeen.

    `source` arg se propaga al sidecar para auditoría ("index" / "capture"
    / "scan-citas" / "whatsapp"). `min_confidence` override local cuando
    el caller es `rag scan-citas --min-confidence 0.5`.

    Returns dict con shape unificada:
      `{cached: bool, decision: str, kind: str, title, when, location,
        confidence, event_uid, reminder_id}`.

    NUNCA raise — cada paso tiene try/except silent-log.
    """
    from rag import _ragvec_state_conn, _silent_log, propose_calendar_event, propose_reminder
    if not _cita_detect_enabled():
        return None
    text = (ocr_text or "").strip()
    if len(text) < _CITA_MIN_CHARS:
        return None
    threshold = (
        float(min_confidence) if min_confidence is not None else _CITA_MIN_CONFIDENCE
    )
    key = _ocr_hash_key(text)
    img_str = str(image_path) if image_path else ""

    # Step 1: dedup lookup. Si ya lo procesamos antes, short-circuit.
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT decision, kind, title, start_text, location, "
                "confidence, event_uid, reminder_id, created_at "
                "FROM rag_cita_detections WHERE ocr_hash = ?",
                (key,),
            ).fetchone()
    except Exception as exc:
        _silent_log("cita_sidecar_read", exc)
        row = None
    if row is not None:
        return {
            "cached": True,
            "decision": row[0],
            "kind": row[1] or "",
            "title": row[2],
            "when": row[3],
            # Backward-compat alias para callers que esperan `start`.
            "start": row[3],
            "location": row[4],
            "confidence": row[5],
            "event_uid": row[6],
            "reminder_id": row[7],
            "created_at": row[8],
        }

    # Step 2: run detector. Resolve via `rag` so tests can
    # monkeypatch.setattr(rag, "_detect_cita_from_ocr", fake) and have it
    # propagate here.
    import rag as _rag
    detected = _rag._detect_cita_from_ocr(text)
    if detected is None:
        # Helper unavailable / malformed — NO persist. Reintentamos next run.
        return None

    now_ts = time.time()
    kind = detected.get("kind") or "note"
    title = detected.get("title") or ""
    when = detected.get("when") or ""
    location = detected.get("location") or ""
    confidence = float(detected.get("confidence") or 0.0)

    # Step 3: below-threshold — persist so dedup wins next time.
    if confidence < threshold:
        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision="low_confidence", kind=kind,
            title=title, start_text=when, location=location,
            confidence=confidence, event_uid=None, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision="low_confidence", kind=kind,
            title=title, when=when, location=location,
            confidence=confidence,
        )

    # Step 4: route by kind.
    if kind == "note":
        # Nothing to schedule. Persist so we don't re-invoke the helper.
        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision="note", kind="note",
            title=title, start_text=when, location=location,
            confidence=confidence, event_uid=None, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision="note", kind="note",
            title=title, when=when, location=location,
            confidence=confidence,
        )

    if kind == "event":
        if not when:
            # Event classification without a parseable date — persist as
            # ambiguous (user can re-evaluate manually, and dedup won't
            # re-call helper for the same OCR text).
            _persist_cita_detection(
                ocr_hash=key, image_path=img_str, source=source,
                decision="ambiguous", kind="event",
                title=title, start_text=when, location=location,
                confidence=confidence, event_uid=None, reminder_id=None,
                created_at=now_ts,
            )
            return _cita_result(
                cached=False, decision="ambiguous", kind="event",
                title=title, when=when, location=location,
                confidence=confidence,
            )
        event_title = title or "Cita"
        notes_blob = f"Auto-detectado de OCR ({source}): {image_path}\n\n{text[:500]}"
        try:
            result_json = propose_calendar_event(
                title=event_title, start=when,
                location=(location or None), notes=notes_blob,
            )
            result = json.loads(result_json)
        except Exception as exc:
            _silent_log("cita_propose_event", exc)
            result = {"created": False, "error": str(exc)}

        event_uid = None
        decision = "error"
        if isinstance(result, dict):
            if result.get("duplicate"):
                decision = "duplicate"
                existing = result.get("existing") or {}
                event_uid = existing.get("uid") or existing.get("event_uid")
            elif result.get("created"):
                decision = "cita"
                event_uid = result.get("event_uid")
            elif result.get("needs_clarification"):
                decision = "ambiguous"
            else:
                decision = "error"

        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision=decision, kind="event",
            title=event_title, start_text=when, location=location,
            confidence=confidence, event_uid=event_uid, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision=decision, kind="event",
            title=event_title, when=when, location=location,
            confidence=confidence, event_uid=event_uid,
        )

    # kind == "reminder"
    reminder_title = title or "Tarea"
    # Apple Reminders NO soporta attachments vía AppleScript; el path de
    # la imagen queda en el body del reminder para referencia — Reminders.
    # app lo muestra como texto, grep-friendly desde CLI también.
    notes_blob = (
        f"Imagen: {image_path}\n"
        f"Origen: {source}\n\n"
        f"{text[:500]}"
    )
    try:
        result_json = propose_reminder(
            title=reminder_title,
            when=when,  # puede ser "" — `propose_reminder` lo tolera
            notes=notes_blob,
        )
        result = json.loads(result_json)
    except Exception as exc:
        _silent_log("cita_propose_reminder", exc)
        result = {"created": False, "error": str(exc)}

    reminder_id = None
    decision = "error"
    if isinstance(result, dict):
        if result.get("created"):
            decision = "reminder"
            reminder_id = result.get("reminder_id")
        elif result.get("needs_clarification"):
            decision = "ambiguous"
        else:
            decision = "error"

    _persist_cita_detection(
        ocr_hash=key, image_path=img_str, source=source,
        decision=decision, kind="reminder",
        title=reminder_title, start_text=when, location=location,
        confidence=confidence, event_uid=None, reminder_id=reminder_id,
        created_at=now_ts,
    )
    return _cita_result(
        cached=False, decision=decision, kind="reminder",
        title=reminder_title, when=when, location=location,
        confidence=confidence, reminder_id=reminder_id,
    )


def _cita_result(
    *, cached: bool, decision: str, kind: str, title: str, when: str,
    location: str, confidence: float,
    event_uid: str | None = None, reminder_id: str | None = None,
) -> dict:
    """Shape uniforme de retorno para `_maybe_create_cita_from_ocr`.
    Incluye `start` como alias de `when` para backward-compat con callers
    que esperan el schema viejo (tests pre-2026-04-23 tarde, renders del
    CLI anteriores al routing por kind).
    """
    return {
        "cached": bool(cached),
        "decision": decision,
        "kind": kind,
        "title": title,
        "when": when,
        "start": when,  # alias
        "location": location,
        "confidence": float(confidence),
        "event_uid": event_uid,
        "reminder_id": reminder_id,
    }


def _persist_cita_detection(
    *, ocr_hash: str, image_path: str, source: str, decision: str,
    kind: str | None, title: str, start_text: str, location: str,
    confidence: float,
    event_uid: str | None, reminder_id: str | None, created_at: float,
) -> None:
    """INSERT OR IGNORE en `rag_cita_detections`.

    Si otro caller ganó la carrera (mismo `ocr_hash` persistido primero),
    respetamos su fila — por eso OR IGNORE y no OR REPLACE. Silent-fail:
    excepciones log-only, no bloquean el caller.

    Incluye columnas `kind` + `reminder_id` post-2026-04-23. Instalaciones
    viejas sin la migration lazy (`_migrate_cita_detections_add_kind`) van
    a fallar el INSERT por columnas inexistentes; el except genérico lo
    captura y persistimos con el subset mínimo (best-effort degradation).
    """
    from rag import _ragvec_state_conn, _silent_log
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO rag_cita_detections "
                "(ocr_hash, image_path, source, decision, kind, title, "
                "start_text, location, confidence, event_uid, reminder_id, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ocr_hash, image_path, source, decision, kind, title,
                    start_text, location, float(confidence), event_uid,
                    reminder_id, float(created_at),
                ),
            )
    except Exception as exc:
        _silent_log(f"cita_sidecar_write:{ocr_hash}", exc)
        # Fallback: retry con subset pre-kind/reminder_id (pre-migration
        # schema). Protege operadores que corrieron el feature original
        # (commit 1d55b27) y NO bajaron `_migrate_cita_detections_add_kind`
        # todavía (ej. tests que instancian un DB bare sin
        # `_ensure_telemetry_tables`).
        try:
            with _ragvec_state_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO rag_cita_detections "
                    "(ocr_hash, image_path, source, decision, title, "
                    "start_text, location, confidence, event_uid, "
                    "created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ocr_hash, image_path, source, decision, title,
                        start_text, location, float(confidence),
                        event_uid, float(created_at),
                    ),
                )
        except Exception as exc2:
            _silent_log(f"cita_sidecar_write_fallback:{ocr_hash}", exc2)
