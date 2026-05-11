"""Latent vault filer — clasifica notas de `00-Inbox/` y propone destino PARA.

Game-Changer G3 (2026-05-11). El vault sigue PARA (`00-Inbox` →
`01-Projects/02-Areas/03-Resources/04-Archive`) pero las notas tienden
a quedarse en inbox indefinidamente. Este módulo automatiza la
clasificación, leaving al user un `rag filer plan` (dry-run) que dice
qué movería + un `rag filer apply` separado que ejecuta los movimientos.

NUNCA mueve sin `--apply` explícito + (opcional) `--yes` para skip
confirmación interactiva. Cada movimiento es reversible mientras la
nota tenga su `created` en frontmatter (recreación trivial).

Pipeline:

1. **Scan** `VAULT/00-Inbox/*.md` (no recursivo — el inbox es flat).
2. **Classify** cada nota: LLM judge qwen2.5:3b lee frontmatter + body
   y devuelve JSON:
       {destination: "01-Projects/<topic>" | "02-Areas/<area>" |
                     "03-Resources/<category>" | "04-Archive/" |
                     "99-obsidian/99-AI/<subfolder>" | "stay",
        confidence: 0..1,
        reason: "<≤80 chars>"}
3. **Plan**: tabla `[note, destination, confidence, reason]`.
4. **Apply** (opt-in): por cada confidence ≥ THRESHOLD, mueve el archivo
   con `shutil.move`. Si destination subfolder no existe, lo crea.
   `stay` y `confidence < THRESHOLD` quedan en inbox.

Wikilinks rewriting: NO se hace en este shipping. Si la nota está
referenciada por wikilink `[[X]]`, Obsidian re-resuelve el link
automáticamente al nuevo path (búsqueda fuzzy por filename). Caso edge
donde rompe: links con path explícito `[[01-Projects/X]]` — esos
requieren rewrite manual del user después del move. Acceptable tradeoff
para el shipping inicial.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_APPLY_CONFIDENCE_THRESHOLD = 0.65


def _inbox_dir() -> Path | None:
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return None
    p = VAULT_PATH / "00-Inbox"
    return p if p.is_dir() else None


def _scan_inbox() -> list[Path]:
    d = _inbox_dir()
    if not d:
        return []
    notes = sorted(p for p in d.glob("*.md") if p.is_file())
    return notes


def _read_note(p: Path, max_chars: int = 3000) -> str:
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    return text[:max_chars]


_CLASSIFY_PROMPT = """Sos un archivero del vault Obsidian del user. Convención PARA:

- `01-Projects/<topic>/` — proyectos activos con deadline o entregable (ej. RAG-Local, Album-Muros-Fractales). Sub-carpeta por proyecto.
- `02-Areas/<area>/` — responsabilidades permanentes sin fin (Finanzas, Salud, Hogar, Grecia, Astor). Sub-carpeta por área.
- `03-Resources/<category>/` — referencia técnica reutilizable (Receta, Articulo, Snippet, Tutorial). Sub-carpeta por categoría.
- `04-Archive/` — material inactivo pero conservado.
- `99-obsidian/99-AI/` — artefactos del sistema (planes agente, conversations, system docs). Si la nota fue claramente generada por un agente o describe infra del sistema.

Dada la nota a continuación, devolvé JSON estricto sin texto extra:

{"destination": "<path>", "confidence": <0..1>, "reason": "<≤80 chars en castellano>"}

Reglas:
- `destination` puede ser "stay" si la nota es captura ambigua sin tema claro — se queda en `00-Inbox/`.
- `confidence ≥ 0.7` solo si tenés evidencia clara en el body (proyecto mencionado, dominio temático obvio, recursividad explícita).
- Si el body está vacío o solo tiene un title + 1 línea, confidence ≤ 0.4 y destination=`stay`.
- Naming del subfolder: lowercase con guiones, ej. `01-Projects/rag-local`, `02-Areas/finanzas-personales`.

NOTA:
"""


def classify_note(p: Path) -> dict:
    """LLM judge → destination + confidence + reason.

    Failsafe: si LLM falla, retorna {destination: 'stay', confidence: 0}.
    """
    body = _read_note(p)
    if not body.strip():
        return {"destination": "stay", "confidence": 0.0,
                "reason": "nota vacía", "path": str(p)}

    prompt = f"{_CLASSIFY_PROMPT}{body}\n\nRespondé SOLO JSON:"
    try:
        from rag import _helper_client, HELPER_MODEL, HELPER_OPTIONS  # noqa: PLC0415
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 200, "num_ctx": 4096},
            format="json",
        )
    except Exception as e:  # noqa: BLE001
        return {"destination": "stay", "confidence": 0.0,
                "reason": f"llm_error: {e!r}"[:80], "path": str(p)}

    raw = ""
    if isinstance(resp, dict):
        raw = ((resp.get("message") or {}).get("content") or "").strip()
    else:
        msg = getattr(resp, "message", None)
        raw = (getattr(msg, "content", None) or "").strip()

    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {"destination": "stay", "confidence": 0.0,
                "reason": "llm_unparseable", "path": str(p)}

    dest = (data.get("destination") or "stay").strip()
    conf = float(data.get("confidence") or 0.0)
    reason = (data.get("reason") or "")[:200]
    # Guardrails: destination tiene que arrancar con un prefix válido.
    if dest != "stay" and not re.match(
        r"^(01-Projects|02-Areas|03-Resources|04-Archive|99-obsidian/99-AI)/",
        dest,
    ):
        return {"destination": "stay", "confidence": 0.0,
                "reason": f"invalid_dest: {dest!r}"[:80], "path": str(p)}
    return {"destination": dest, "confidence": conf, "reason": reason, "path": str(p)}


def plan(limit: int | None = None) -> list[dict]:
    """Scan inbox + clasifica cada nota. NO mueve nada — solo devuelve plan.

    Útil para `rag filer plan` o como dry-run antes de `--apply`.
    """
    notes = _scan_inbox()
    if limit:
        notes = notes[:limit]
    out: list[dict] = []
    for p in notes:
        out.append(classify_note(p))
    return out


def apply_plan(items: list[dict], *, threshold: float = _APPLY_CONFIDENCE_THRESHOLD) -> dict:
    """Ejecuta los movimientos del plan con confidence >= threshold.

    Returns: stats dict con counts moved/skipped/errors.
    """
    try:
        from rag import VAULT_PATH  # noqa: PLC0415
    except Exception:
        return {"error": "vault path resolver failed"}

    stats = {"moved": 0, "skipped_low_conf": 0,
             "skipped_stay": 0, "errors": 0, "details": []}
    for item in items:
        path = Path(item.get("path") or "")
        if not path.is_file():
            stats["errors"] += 1
            stats["details"].append({"path": str(path), "outcome": "missing"})
            continue
        dest = item.get("destination", "stay")
        conf = float(item.get("confidence") or 0.0)
        if dest == "stay":
            stats["skipped_stay"] += 1
            continue
        if conf < threshold:
            stats["skipped_low_conf"] += 1
            stats["details"].append({
                "path": str(path), "outcome": "low_conf",
                "conf": conf, "dest": dest,
            })
            continue
        target_dir = VAULT_PATH / dest
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name
            # Si ya existe colisión, suffixar con timestamp.
            if target.exists():
                ts = datetime.now().strftime("%H%M%S")
                target = target_dir / f"{path.stem}-{ts}{path.suffix}"
            shutil.move(str(path), str(target))
            stats["moved"] += 1
            stats["details"].append({
                "from": str(path), "to": str(target),
                "conf": conf, "outcome": "moved",
            })
        except OSError as e:
            stats["errors"] += 1
            stats["details"].append({
                "path": str(path), "outcome": "error", "error": str(e),
            })
    return stats


__all__ = ["plan", "apply_plan", "classify_note"]
