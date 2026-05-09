"""Filing — asistente de inbox que propone destino + upward-link + apply/undo.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el sub-sistema de filing (proposal builder + personalización
fase 3 + apply/undo + vault_write_lock) desde `rag/__init__.py`.

## Contexto

Tres etapas:
- **Fase 1**: dry-run puro — calcula propuestas + loguea a
  `rag_filing_log`. No mueve archivos.
- **Fase 2**: confirmación interactiva (CLI consume el log para
  alimentar un ranker).
- **Fase 3** (personalización): k-NN sobre decisiones pasadas
  para sesgar la propuesta hacia folders que el user ya validó.

## Apply + undo

- `_open_filing_batch` + `_append_filing_batch` graban entry por
  entry para que un Ctrl-C deje un récord parcial revertible.
- `_apply_filing_move` usa `os.link` + unlink atómico para evitar
  TOCTOU race con el vault watcher (cae a `shutil.move` cross-device
  cuando el FS no soporta hardlinks).
- `_rollback_filing_batch` revierte en orden inverso (remueve
  upward-link + move back).

## vault_write_lock

Advisory lock global para mutadores del vault + índice. Cubre
`rag index`, `rag file --apply`, `rag inbox --apply`, `rag archive`.
Vivía históricamente acá porque filing fue uno de los primeros
mutadores; ahora es shared infra. Re-export shim mantiene
`rag.vault_write_lock` accesible.

## Lazy imports

Deps en `VAULT_PATH`, `DB_PATH`, `_silent_log`, `embed`, `clean_md`,
`_is_moc_note`, `_suggest_folder_for_note`, `_note_centroids`,
`_ragvec_state_conn`, `_sql_append_event`, `_sql_write_with_retry`,
`_log_sql_state_error`, `_index_single_file` — todos en
`rag/__init__.py`. Lazy adentro de funciones evita circular import.

`_map_filing_row` SÍ se importa top-level desde `rag._row_mappers`
(módulo hoja sin deps al parent).

## Re-export

`rag/__init__.py` hace `from rag._filing import *  # noqa`.
Preserva 100% compat con `rag.vault_write_lock(...)`,
`rag.build_filing_proposal(...)`, `rag.FILING_LOG_PATH`, etc.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# OK importar acá: `_row_mappers` solo depende de `datetime`, sin circular.
from rag._row_mappers import _map_filing_row

if TYPE_CHECKING:
    from rag import SqliteVecCollection

__all__ = [
    "FILING_LOG_PATH",
    "FILING_CONFIDENCE_FIRM",
    "FILING_CONFIDENCE_TENTATIVE",
    "FILING_PERSONALIZE_MIN_HISTORY",
    "FILING_PERSONALIZE_TOP_K",
    "FILING_PERSONALIZE_MIN_SIM",
    "FILING_AGREE_BOOST",
    "FILING_BATCHES_DIR",
    "VAULT_WRITE_LOCK_PATH",
    "FILING_UPWARD_MARKER",
    "_top_k_neighbors",
    "_infer_upward_link",
    "build_filing_proposal",
    "_filing_log_proposal",
    "_load_filing_decisions",
    "_embed_note_body",
    "_personalized_folder_vote",
    "vault_write_lock",
    "_append_upward_link",
    "_remove_upward_link",
    "_apply_filing_move",
    "_open_filing_batch",
    "_append_filing_batch",
    "_write_filing_batch",
    "_last_filing_batch",
    "_rollback_filing_batch",
]


FILING_LOG_PATH = Path.home() / ".local/share/obsidian-rag/filing.jsonl"
# Umbrales de confianza para la etiqueta visual (la decisión final siempre es
# del usuario; el sistema solo colorea):
#   ≥ 0.55  → firm      (≥55% de los K vecinos vota la misma carpeta)
#   0.35-0.55 → tentative
#   < 0.35  → low       (señal insuficiente, sugerir revisión manual)
FILING_CONFIDENCE_FIRM = 0.55
FILING_CONFIDENCE_TENTATIVE = 0.35

# Personalización fase 3
FILING_PERSONALIZE_MIN_HISTORY = 5      # below this → fallback a baseline puro
FILING_PERSONALIZE_TOP_K = 5            # vecinos en el espacio de decisiones pasadas
FILING_PERSONALIZE_MIN_SIM = 0.30       # piso de cosine para que una decisión "cuente"
FILING_AGREE_BOOST = 0.15               # bump de confidence cuando baseline+personalized coinciden

# Apply + undo
FILING_BATCHES_DIR = Path.home() / ".local/share/obsidian-rag/filing_batches"
# Token que marca el upward-link appendeado al pie de la nota — permite al
# undo detectarlo y removerlo limpio. Cualquier edición manual del usuario
# queda por encima del token intacta.
FILING_UPWARD_MARKER = "<!-- rag-file:upward -->"


def _vault_write_lock_path() -> Path:
    """Resolve VAULT_WRITE_LOCK_PATH at call time.

    Honra monkeypatches sobre `rag.VAULT_WRITE_LOCK_PATH` (los tests lo
    patchean para apuntar a un tmp). Fallback: `rag.DB_PATH / ".write.lock"`
    (compat con call sites que solo patchean DB_PATH).
    """
    import rag as _rag  # noqa: PLC0415
    explicit = getattr(_rag, "VAULT_WRITE_LOCK_PATH", None)
    if isinstance(explicit, Path) and not str(explicit).endswith(
        "_uninitialized.lock"
    ):
        return explicit
    db_path = getattr(_rag, "DB_PATH", None)
    if isinstance(db_path, Path):
        return db_path / ".write.lock"
    return Path("/tmp/_rag_vault_write_lock_uninitialized.lock")


# Compat alias — pre-modularización el path se accedía como
# `rag.VAULT_WRITE_LOCK_PATH`. Para preservar accesos directos al símbolo
# (no via función), late-bind via __getattr__ del módulo `rag`. Acá
# exportamos un sentinel que `vault_write_lock` resuelve internamente.
VAULT_WRITE_LOCK_PATH: Path = Path("/tmp/_rag_vault_write_lock_uninitialized.lock")
# Nota: el valor real lo resuelve `_vault_write_lock_path()` al primer uso.
# Tests + call sites que toquen el const directamente lo verán como path
# placeholder hasta que el módulo `rag` cargue + asigne el valor real.
# Para compat 100%, el shim en `rag/__init__.py` hace
# `_filing.VAULT_WRITE_LOCK_PATH = DB_PATH / ".write.lock"` after-load.


def _top_k_neighbors(
    col: "SqliteVecCollection",
    note_path: str,
    k: int = 8,
    skip_folder_prefix: str = "00-",
) -> list[tuple[dict, float]]:
    """Top-k semantic neighbors (meta, similarity) dedupeado por file, excluye
    la nota misma y cualquier path en Inbox-style folders (00-*).

    sqlite-vec cosine-space: distance = 1 - cos_sim → similarity = max(0, 1-dist).
    """
    from rag import VAULT_PATH, clean_md, embed  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return []
    col_total = int(col.count())
    if not full.is_file() or col_total == 0:
        return []
    text = clean_md(full.read_text(encoding="utf-8", errors="ignore"))[:3000].strip()
    if not text:
        return []
    try:
        q_embed = embed([text])[0]
    except Exception:
        return []
    n = min(k * 4, col_total)
    res = col.query(
        query_embeddings=[q_embed], n_results=n,
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out: list[tuple[dict, float]] = []
    seen_files: set[str] = set()
    for m, d in zip(metas, dists):
        f = m.get("file", "")
        if not f or f == note_path or f in seen_files:
            continue
        if f.startswith(skip_folder_prefix):
            continue
        seen_files.add(f)
        sim = max(0.0, 1.0 - float(d))
        out.append((m, sim))
        if len(out) >= k:
            break
    return out


def _infer_upward_link(
    neighbors: list[tuple[dict, float]],
) -> tuple[str, str]:
    """De los vecinos top-k, elegir target para el upward-link.

    Preferencia: primer MOC detectado (via _is_moc_note, que ya maneja
    title/tag/folder-index). Si no hay MOC entre los vecinos, devolver el
    top-1 como link horizontal (menos ideal pero no deja la nota huérfana).
    Retorna (title, kind) con kind ∈ {"moc", "neighbor", ""}.
    """
    from rag import _is_moc_note  # noqa: PLC0415

    for m, _ in neighbors:
        if _is_moc_note(m):
            return (m.get("note", ""), "moc")
    if neighbors:
        return (neighbors[0][0].get("note", ""), "neighbor")
    return ("", "")


def build_filing_proposal(
    col: "SqliteVecCollection",
    note_path: str,
    k: int = 8,
    history: list[dict] | None = None,
) -> dict:
    """Compone folder sugerido + upward-link + neighbors para una nota del
    Inbox. Retorna dict plano que CLI + logger consumen igual.

    Compone tres señales:
      - baseline: `_suggest_folder_for_note` (mode-voting sobre vecinos del
        corpus, fuera de Inbox). Lo que ya teníamos en fase 1/2.
      - upward-link: `_infer_upward_link` (MOC entre neighbors o top-1).
      - personalización (fase 3): si hay ≥FILING_PERSONALIZE_MIN_HISTORY
        decisiones pasadas, votamos por k-NN contra ellas y comparamos con
        baseline. Tres estados:
          * "agreed"           — ambos coinciden → boost de confidence.
          * "personalized"     — disagree y la señal personal es más fuerte
                                  → la propuesta cambia.
          * "baseline+history" — disagree pero baseline gana → no cambia
                                  pero marcamos que hay señal histórica.
        Cold-start (history < umbral) → simplemente "baseline".

    `history` opcional: el caller puede precargarlo (ej. el CLI lo hace
    una vez antes del loop) para evitar releer filing.jsonl por nota.
    """
    from rag import VAULT_PATH, _suggest_folder_for_note  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    if not full.is_file():
        return {"path": note_path, "error": "not_found"}

    neighbors = _top_k_neighbors(col, note_path, k=k)
    base_folder, base_confidence = _suggest_folder_for_note(col, note_path, k=k)
    upward_title, upward_kind = _infer_upward_link(neighbors)

    folder = base_folder
    confidence = base_confidence
    source = "baseline"
    evidence: list[dict] = []

    if history is None:
        history = _load_filing_decisions()
    if len(history) >= FILING_PERSONALIZE_MIN_HISTORY:
        q_embed = _embed_note_body(note_path)
        votes, evidence = _personalized_folder_vote(col, q_embed, history)
        if votes:
            personal_folder = max(votes, key=votes.get)
            total = sum(votes.values())
            personal_conf = (votes[personal_folder] / total) if total else 0.0
            if personal_folder == base_folder and base_folder:
                source = "agreed"
                confidence = min(0.99, base_confidence + FILING_AGREE_BOOST)
            elif personal_conf > base_confidence:
                folder = personal_folder
                confidence = round(personal_conf, 3)
                source = "personalized"
            else:
                source = "baseline+history"

    return {
        "path": note_path,
        "note": full.stem,
        "folder": folder,
        "confidence": confidence,
        "source": source,
        "evidence": evidence,
        "upward_title": upward_title,
        "upward_kind": upward_kind,
        "neighbors": [
            {
                "path": m.get("file", ""),
                "note": m.get("note", ""),
                "sim": round(s, 3),
            }
            for m, s in neighbors[:5]
        ],
    }


def _filing_log_proposal(proposal: dict, decision: str | None = None) -> None:
    """Append-only log. Una línea por propuesta into rag_filing_log.
    `decision` es opcional — fase 1 no lo setea (dry-run), fase 2 lo setea
    con accept/reject/edit/skip para alimentar el ranker. SQL-only since T10.
    """
    from rag import (  # noqa: PLC0415
        _ragvec_state_conn,
        _sql_append_event,
        _sql_write_with_retry,
    )

    ts = datetime.now().isoformat(timespec="seconds")
    event = {"ts": ts, "cmd": "filing_proposal", **proposal}
    if decision is not None:
        event["decision"] = decision

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_filing_log", _map_filing_row(event))
    _sql_write_with_retry(_do, "filing_sql_write_failed")


def _load_filing_decisions(limit: int = 500) -> list[dict]:
    """Lee las últimas N decisiones positivas (accept/edit) de rag_filing_log.

    Solo incluye las que tienen `applied_to`. reject/skip/error/quit se
    descartan — la fase 3 personaliza desde lo que validaste, no desde lo
    que rechazaste. SQL-only since T10. `applied_to` + `decision` viven en
    extra_json (no son columnas nativas), así que se decodifican aquí.
    """
    from rag import _log_sql_state_error, _ragvec_state_conn  # noqa: PLC0415

    try:
        with _ragvec_state_conn() as conn:
            import sqlite3 as _sqlite3  # noqa: PLC0415
            prev_factory = conn.row_factory
            try:
                conn.row_factory = _sqlite3.Row
                rows = list(conn.execute(
                    "SELECT ts, extra_json FROM rag_filing_log "
                    "ORDER BY ts DESC"
                ).fetchall())
            finally:
                conn.row_factory = prev_factory
    except Exception as exc:
        _log_sql_state_error("filing_decisions_sql_read_failed", err=repr(exc))
        return []
    out: list[dict] = []
    for r in rows:
        if len(out) >= limit:
            break
        try:
            extra = json.loads(r["extra_json"] or "{}")
        except Exception:
            continue
        decision = extra.get("decision")
        if decision not in ("accept", "edit"):
            continue
        applied = extra.get("applied_to")
        if not applied:
            continue
        out.append({
            "applied_to": applied,
            "target_folder": str(Path(applied).parent),
            "decision": decision,
            "ts": r["ts"] or "",
        })
    return out


def _embed_note_body(note_path: str) -> list[float] | None:
    """Embed del cuerpo de la nota para lookups de similitud. None si no
    se puede leer o está vacía. Espejo de lo que hace `_top_k_neighbors`
    internamente — extraído acá para que personalización lo reuse.
    """
    from rag import VAULT_PATH, clean_md, embed  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return None
    if not full.is_file():
        return None
    text = clean_md(full.read_text(encoding="utf-8", errors="ignore"))[:3000].strip()
    if not text:
        return None
    try:
        return embed([text])[0]
    except Exception:
        return None


def _personalized_folder_vote(
    col: "SqliteVecCollection",
    q_embed: list[float] | None,
    history: list[dict],
    top_k: int = FILING_PERSONALIZE_TOP_K,
) -> tuple[dict[str, float], list[dict]]:
    """Voto k-NN sobre decisiones pasadas. Para la nota query (q_embed),
    encontrar las top_k decisiones más similares y sumar similitud por
    target_folder.

    Retorna (votes, evidence):
      - votes: {folder: similarity_sum}
      - evidence: lista [{applied_to, target_folder, sim}] de los vecinos
        que efectivamente contribuyeron (para mostrar en UI: "3 similares
        en este mismo folder").

    Skip silencioso si no hay history, no hay query embedding, o ningún
    `applied_to` está en los centroides (ej. notas borradas después).
    """
    from rag import _note_centroids  # noqa: PLC0415

    if not history or q_embed is None:
        return {}, []
    import numpy as np  # noqa: PLC0415
    files, _metas, arr = _note_centroids(col)
    file_to_idx = {f: i for i, f in enumerate(files)}

    embs: list = []
    info: list[dict] = []
    for d in history:
        idx = file_to_idx.get(d["applied_to"])
        if idx is None:
            continue
        embs.append(arr[idx])
        info.append(d)

    if not embs:
        return {}, []

    decision_arr = np.asarray(embs, dtype=np.float32)
    q = np.asarray(q_embed, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return {}, []
    q = q / n
    # decision_arr ya viene L2-normalizado de _note_centroids → dot = cosine.
    sims = decision_arr @ q

    order = np.argsort(sims)[::-1]
    votes: dict[str, float] = {}
    evidence: list[dict] = []
    for i in order[:top_k]:
        sim = float(sims[i])
        if sim < FILING_PERSONALIZE_MIN_SIM:
            break
        folder = info[i]["target_folder"]
        votes[folder] = votes.get(folder, 0.0) + sim
        evidence.append({
            "applied_to": info[i]["applied_to"],
            "target_folder": folder,
            "sim": round(sim, 3),
        })
    return votes, evidence


@contextlib.contextmanager
def vault_write_lock(op: str, *, blocking: bool = True, timeout: float = 300.0):
    """Advisory lock para serializar mutadores del vault + índice.

    Cubre `rag index` (full + reset), `rag file --apply`, `rag inbox --apply`,
    y cualquier flujo que mueva notas o escriba chunks a sqlite-vec en batch.
    NO se usa en paths read-only (query, chat, stats) ni en re-index de una
    sola nota vía watch/ambient (esos serializan a través de los locks
    internos de sqlite-vec/SQLite; tomar el lock global ahí bloquearía un
    mutator batch por horas).

    `op` es el nombre de la operación (para el error message si otro
    proceso tiene el lock). Con `blocking=False` raise RuntimeError en vez
    de esperar. Con `blocking=True` espera hasta `timeout` y raise si no
    se obtiene.
    """
    lock_path = _vault_write_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        if blocking:
            deadline = time.time() + timeout
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise RuntimeError(
                            f"{op}: otro mutator tiene el lock "
                            f"({lock_path}). Esperé {timeout:.0f}s."
                        )
                    time.sleep(0.5)
        else:
            try:
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError:
                raise RuntimeError(
                    f"{op}: otro mutator tiene el lock ({lock_path})."
                )
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{op} · pid={os.getpid()} · {datetime.now().isoformat()}\n")
        lock_file.flush()
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def _append_upward_link(note_full_path: Path, upward_title: str) -> bool:
    """Agrega `\\n\\n---\\n{marker}\\n↑ [[Title]]\\n` al final de la nota.
    Idempotente: si ya existe un bloque con el marker, lo reemplaza en vez
    de duplicar. Retorna True si se escribió.
    """
    if not note_full_path.is_file() or not upward_title:
        return False
    raw = note_full_path.read_text(encoding="utf-8", errors="ignore")
    block = f"\n\n---\n{FILING_UPWARD_MARKER}\n↑ [[{upward_title}]]\n"
    if FILING_UPWARD_MARKER in raw:
        # Reemplazar el bloque viejo (desde el --- previo al marker hasta EOF).
        idx = raw.find(FILING_UPWARD_MARKER)
        # Buscar el --- que precede al marker (ancla del bloque).
        sep = raw.rfind("\n---\n", 0, idx)
        if sep >= 0:
            raw = raw[:sep] + block
        else:
            raw = raw.rstrip() + block
    else:
        raw = raw.rstrip() + block
    note_full_path.write_text(raw, encoding="utf-8")
    return True


def _remove_upward_link(note_full_path: Path) -> bool:
    """Remueve el bloque agregado por _append_upward_link. Usado por undo.
    Retorna True si había algo que remover.
    """
    if not note_full_path.is_file():
        return False
    raw = note_full_path.read_text(encoding="utf-8", errors="ignore")
    idx = raw.find(FILING_UPWARD_MARKER)
    if idx < 0:
        return False
    sep = raw.rfind("\n---\n", 0, idx)
    if sep < 0:
        return False
    note_full_path.write_text(raw[:sep].rstrip() + "\n", encoding="utf-8")
    return True


def _apply_filing_move(
    col: "SqliteVecCollection",
    src_rel: str,
    target_folder: str,
    upward_title: str,
) -> dict:
    """Ejecuta el move + upward-link + reindex para UNA nota.

    Retorna entry del batch log: {src, dst, upward_title, upward_written}.
    Levanta si target folder queda fuera del vault (seguridad path traversal).
    """
    from rag import VAULT_PATH, _index_single_file, _silent_log  # noqa: PLC0415

    src = (VAULT_PATH / src_rel).resolve()
    src.relative_to(VAULT_PATH.resolve())   # ValueError si escapa
    if not src.is_file():
        raise FileNotFoundError(src_rel)
    target_dir = (VAULT_PATH / target_folder).resolve()
    target_dir.relative_to(VAULT_PATH.resolve())
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if dst.exists():
        raise FileExistsError(str(dst.relative_to(VAULT_PATH)))
    # Mover con shutil para manejar cross-device edge cases (iCloud puede
    # montarse distinto que el home en ciertos contextos).
    #
    # 2026-04-24 audit: TOCTOU race. El `if dst.exists()` chequeo arriba
    # ocurre ANTES del move. Si otro proceso (vault watcher, otra
    # invocación de `rag file --apply` en paralelo) crea `dst` entre
    # el check y el shutil.move, `shutil.move` SOBRESCRIBE silently
    # — perdiendo los datos del competidor. El window es sub-ms en
    # operación normal pero con iCloud sync puede extenderse.
    #
    # Mitigación: usar `os.link` (atomic "create if not exists" en
    # POSIX) como primer intento y caer a shutil.move solo si link
    # rehúsa (cross-device, no soporta hardlinks). Post-link borrar
    # src para simular el move. Si el hardlink falla con FileExistsError,
    # raise — garantizamos que nunca sobrescribimos silently. El
    # shutil.move(str, str) solo se usa si link no está disponible (e.g.
    # iCloud FUSE mount que prohíbe hardlinks entre sub-árboles). En ese
    # caso confiamos en el check anterior + documentamos el riesgo
    # residual.
    try:
        os.link(str(src), str(dst))
    except FileExistsError:
        # Otro proceso creó `dst` entre nuestro check y el link — NO
        # sobrescribir, raise con mensaje claro para que el caller
        # sepa que fue race, no bug del usuario.
        raise FileExistsError(
            f"race condition: {dst.relative_to(VAULT_PATH)} fue creado "
            f"por otro proceso durante el apply. Re-intentá el move "
            f"— el catálogo PARA va a tener ambos archivos hasta entonces."
        )
    except OSError:
        # Cross-device / FS no soporta hardlinks (iCloud en ciertos modos).
        # Fallback al shutil.move original — reintroduce la TOCTOU window
        # pero es necesario para compat. El check `if dst.exists()`
        # anterior sigue protegiendo el 99.9% de casos.
        shutil.move(str(src), str(dst))
    else:
        # Hardlink exitoso → borrar src para completar el "move".
        try:
            src.unlink()
        except OSError as exc:
            _silent_log("inbox.apply.unlink_src_after_link", exc)
    written = _append_upward_link(dst, upward_title) if upward_title else False
    # Reindex: el hook del ambient agent se dispara sobre saves del Inbox,
    # pero acá venimos del Inbox hacia afuera — skip_contradict para no
    # gatillar un check O(n²) costoso en un apply-batch.
    try:
        _index_single_file(col, dst, skip_contradict=True)
    except Exception as exc:
        _silent_log("inbox.apply.index_single_file", exc)
    return {
        "src": src_rel,
        "dst": str(dst.relative_to(VAULT_PATH)),
        "upward_title": upward_title,
        "upward_written": written,
    }


def _open_filing_batch() -> Path:
    """Crea un batch file vacío con timestamp y devuelve su path.

    Diseño: el batch se abre ANTES del loop de moves y se appendea por cada
    move exitoso (ver `_append_filing_batch`). Si el loop se interrumpe
    (Ctrl-C, error, OOM), `rag file --undo` puede revertir los moves ya
    aplicados — antes se escribía todo al final del loop y una interrupción
    dejaba los moves sin récord.
    """
    # Resolver FILING_BATCHES_DIR via el namespace `rag` para que
    # `monkeypatch.setattr(rag, "FILING_BATCHES_DIR", tmp_path)` en los tests
    # propague — la const local del módulo no se ve afectada por monkeypatches
    # sobre `rag.X`.
    import rag as _rag  # noqa: PLC0415
    batches_dir = getattr(_rag, "FILING_BATCHES_DIR", FILING_BATCHES_DIR)
    batches_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = batches_dir / f"{ts}.jsonl"
    path.touch()
    return path


def _append_filing_batch(batch_path: Path, entry: dict) -> None:
    """Appendea una entry al batch. Flush + fsync para que sobreviva crash."""
    with batch_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_filing_batch(entries: list[dict]) -> Path | None:
    """Compat wrapper — bulk write. Uso legacy desde tests y rutas pre-refactor.
    Nueva feature debe usar `_open_filing_batch` + `_append_filing_batch`.
    """
    if not entries:
        return None
    batch_path = _open_filing_batch()
    for e in entries:
        _append_filing_batch(batch_path, e)
    return batch_path


def _last_filing_batch() -> Path | None:
    """Devuelve el batch más reciente (por mtime). None si no hay ninguno."""
    # Resolver via `rag` para honrar monkeypatches en tests (ver
    # `_open_filing_batch` para detalle del patrón).
    import rag as _rag  # noqa: PLC0415
    batches_dir = getattr(_rag, "FILING_BATCHES_DIR", FILING_BATCHES_DIR)
    if not batches_dir.is_dir():
        return None
    batches = sorted(
        batches_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return batches[0] if batches else None


def _rollback_filing_batch(col: "SqliteVecCollection", batch_path: Path) -> list[dict]:
    """Revierte cada entry: remueve upward-link, mueve back del dst al src.
    Retorna lista de resultados por entry: {src, dst, ok, error?}.

    No borra el batch log hasta confirmar: el caller decide qué hacer con él
    (el CLI lo rename a .undone para trazabilidad).
    """
    from rag import VAULT_PATH, _index_single_file, _silent_log  # noqa: PLC0415

    results: list[dict] = []
    with batch_path.open("r", encoding="utf-8") as f:
        entries = [json.loads(ln) for ln in f if ln.strip()]
    # Revertir en orden inverso por seguridad (si hubo dependencias de path).
    for e in reversed(entries):
        src_rel = e["src"]
        dst_rel = e["dst"]
        r = {"src": src_rel, "dst": dst_rel, "ok": False}
        try:
            dst_full = (VAULT_PATH / dst_rel).resolve()
            dst_full.relative_to(VAULT_PATH.resolve())
            if not dst_full.is_file():
                r["error"] = "dst no existe"
                results.append(r)
                continue
            if e.get("upward_written"):
                _remove_upward_link(dst_full)
            src_full = (VAULT_PATH / src_rel).resolve()
            src_full.relative_to(VAULT_PATH.resolve())
            src_full.parent.mkdir(parents=True, exist_ok=True)
            if src_full.exists():
                r["error"] = "src ocupado — no overwrite"
                results.append(r)
                continue
            shutil.move(str(dst_full), str(src_full))
            try:
                _index_single_file(col, src_full, skip_contradict=True)
            except Exception as exc:
                _silent_log("inbox.undo.index_single_file", exc)
            r["ok"] = True
        except Exception as ex:
            r["error"] = str(ex)
        results.append(r)
    return results
