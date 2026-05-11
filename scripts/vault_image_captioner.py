#!/usr/bin/env python3
"""VLM captioner — describe imágenes del vault + bridge media en batch.

Game-Changer G5 (2026-05-11). Las imágenes del vault y los media
inbound de WhatsApp hoy son **invisibles al retrieval** salvo por
filename. Este script:

1. Scans paths configurados (vault sub-folders + bridge media dir)
   por imágenes (.png/.jpg/.jpeg/.heic/.webp) SIN un sidecar
   `<image-path>.caption.md`.
2. Para cada una, corre `mlx-vlm` (granite-vision-3.2-2b) con prompt
   "describí en 2 frases en castellano qué muestra esta imagen + si
   hay texto legible, transcribilo después de '--TEXTO:--'".
3. Escribe sidecar `<image>.caption.md` con frontmatter
   `{type: image-caption, original: [[<image-relative>]], ts}` + el
   caption. `rag watch` lo indexa al vault → corpus retrieval ahora
   encuentra la imagen via su contenido visual.

Cap: `--limit 10` por run para no quemar GPU. VLM tarda 5-10s por
imagen en M2 Pro.

Uso:
  scripts/vault_image_captioner.py                  # default 10 imgs
  scripts/vault_image_captioner.py --limit 50 --vault-only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".webp"}

_CAPTION_PROMPT = (
    "Describí en 2-3 frases en castellano qué muestra esta imagen. "
    "Si hay texto legible (carteles, screenshots, recibos, captures de UI), "
    "transcribilo literalmente después del marcador '--TEXTO:--'. "
    "Sé concreto: nombres de marcas, números, fechas si aparecen. "
    "Si no entendés la imagen, decílo brevemente."
)


def _vault_image_dirs() -> list[Path]:
    """Directorios del vault donde buscar imágenes. Solo PARA principal
    + 99-AI external-ingest; evitamos `.obsidian/`, `.trash/`, etc.
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return []
    candidates = [
        VAULT_PATH / "00-Inbox",
        VAULT_PATH / "01-Projects",
        VAULT_PATH / "02-Areas",
        VAULT_PATH / "03-Resources",
        VAULT_PATH / "99-obsidian" / "99-AI",
    ]
    return [c for c in candidates if c.is_dir()]


def _bridge_media_dir() -> Path | None:
    try:
        import rag as _rag  # noqa: PLC0415
        # bridge stores media en `store/<chat_jid>/<filename>` dentro del
        # whatsapp-bridge dir. Scan recursivo desde su store/.
        bridge_db = _rag.WHATSAPP_DB_PATH
        store = bridge_db.parent  # store/
        if store.is_dir():
            return store
    except Exception:
        pass
    return None


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMAGE_EXTS


def _sidecar_path(image_path: Path) -> Path:
    """Sidecar dentro del vault. Para imágenes del bridge (fuera del vault)
    redirigimos el sidecar a `VAULT/99-AI/external-ingest/image-captions/`.
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return image_path.with_suffix(image_path.suffix + ".caption.md")
    try:
        rel = image_path.relative_to(VAULT_PATH)
        return image_path.with_suffix(image_path.suffix + ".caption.md")
    except ValueError:
        # Imagen fuera del vault (ej. bridge media) — sidecar va al vault.
        slug = image_path.name.replace("/", "_")
        target_dir = VAULT_PATH / "99-obsidian" / "99-AI" / "external-ingest" / "image-captions"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{slug}.caption.md"


def _collect_pending(limit: int, vault_only: bool) -> list[Path]:
    pending: list[Path] = []
    dirs = _vault_image_dirs()
    if not vault_only:
        bm = _bridge_media_dir()
        if bm:
            dirs.append(bm)
    for base in dirs:
        for p in base.rglob("*"):
            if not p.is_file() or not _is_image(p):
                continue
            sidecar = _sidecar_path(p)
            if sidecar.is_file():
                continue
            pending.append(p)
            if len(pending) >= limit:
                return pending
    return pending


def _caption_one(image_path: Path) -> tuple[bool, str]:
    """Llama mlx-vlm y devuelve (ok, text_or_error)."""
    from rag.ocr import _vlm_caption_enabled, _vlm_describe  # noqa: PLC0415

    if not _vlm_caption_enabled():
        return False, "vlm_disabled"
    try:
        caption = _vlm_describe(image_path, prompt=_CAPTION_PROMPT)
    except Exception as exc:  # noqa: BLE001
        return False, f"vlm_error: {exc}"[:200]
    if not caption or not caption.strip():
        return False, "empty_caption"
    return True, caption.strip()


def _write_sidecar(image_path: Path, caption: str) -> Path | None:
    sidecar = _sidecar_path(image_path)
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
        rel_str = ""
        try:
            rel_str = str(image_path.relative_to(VAULT_PATH))
        except ValueError:
            rel_str = str(image_path)
    except Exception:
        rel_str = str(image_path)

    body = (
        "---\n"
        "type: image-caption\n"
        f"original: \"{rel_str}\"\n"
        f"captioned_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "tags: [image-caption, vlm]\n"
        "---\n\n"
        f"# Caption de {image_path.name}\n\n"
        f"{caption}\n"
    )
    try:
        sidecar.write_text(body, encoding="utf-8")
        return sidecar
    except OSError as e:
        print(f"[captioner] sidecar write failed for {image_path}: {e}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10,
                    help="Cap imágenes por run (default 10)")
    ap.add_argument("--vault-only", action="store_true",
                    help="Solo imágenes del vault, skip bridge media")
    ap.add_argument("--dry-run", action="store_true",
                    help="Lista pending sin captionar")
    args = ap.parse_args()

    print(
        f"[captioner] start limit={args.limit} vault_only={args.vault_only} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    pending = _collect_pending(limit=args.limit, vault_only=args.vault_only)
    print(f"[captioner] pending images: {len(pending)}", flush=True)
    if args.dry_run:
        for p in pending[:20]:
            print(f"  would caption: {p}", flush=True)
        return 0

    ok_count = 0
    err_count = 0
    t0 = time.perf_counter()
    for p in pending:
        ok, info = _caption_one(p)
        if ok:
            sidecar = _write_sidecar(p, info)
            outcome = f"sidecar={sidecar.name}" if sidecar else "sidecar_write_failed"
            print(f"[captioner] ok {p.name[:50]} → {outcome}", flush=True)
            ok_count += 1
        else:
            print(f"[captioner] err {p.name[:50]} → {info}", flush=True)
            err_count += 1
            # Si VLM está disabled, abortamos el run completo — no
            # sirve seguir intentando.
            if info == "vlm_disabled":
                print("[captioner] VLM not enabled — set RAG_VLM_CAPTION=1 + ensure mlx-vlm installed", flush=True)
                break

    dt = time.perf_counter() - t0
    print(
        f"[captioner] done · ok={ok_count} err={err_count} elapsed={dt:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
