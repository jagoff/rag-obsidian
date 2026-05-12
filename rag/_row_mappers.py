"""Pure row-mapper functions for SQL state tables in `telemetry.db`.

Each `_map_*_row(ev: dict) -> dict` takes a live event payload and returns
a dict whose keys match the target table's columns. Unknown fields land in
`extra_json` / `payload_json` (see T1 DDL). Mappers are deliberately thin
and have ZERO side effects — sin SQL, sin locks, sin filesystem. Solo
transformación dict→dict.

## Por qué un módulo separado

Audit perf 2026-05-08 / Phase 1a de modularización (ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer ~22 funciones pure dict→dict del `rag/__init__.py` (que tenía
64.5k LOC) reduce el archivo y aísla la sub-API más fácil de testear
por separado. Los readers / recorders / migrations del SQL state stack
quedan en `rag/__init__.py` por ahora — su extracción (Phase 1b/1c)
necesita import careful de `_ragvec_state_conn` + `_sql_*` primitives
+ `_silent_log`.

## Re-export

Los símbolos quedan accesibles desde `rag` directamente via
`from rag._row_mappers import *  # noqa: F401, F403` en `rag/__init__.py`.
Esto preserva 100% compat con call sites históricos que hacen
`from rag import _map_queries_row` o `rag._map_queries_row`.

## Tests

Los tests SQL state writers (`tests/test_rag_writers_sql.py`,
`tests/test_sql_state_primitives.py`) ejercen estos mappers
indirectamente via `_sql_append_event`. Tests directos al mapper
(monkeypatch del `datetime.now()` para el default de `ts`) ya viven
en el path historic — esta extracción NO los modifica.
"""

from __future__ import annotations

from datetime import datetime

__all__ = [
    "_QUERIES_COLS",
    "_QUERIES_JSON",
    "_map_queries_row",
    "_map_behavior_row",
    "_map_feedback_row",
    "_map_tune_row",
    "_map_contradiction_row",
    "_map_ambient_row",
    "_map_ambient_state_row",
    "_map_brief_written_row",
    "_map_brief_state_row",
    "_map_wa_tasks_row",
    "_map_archive_row",
    "_map_filing_row",
    "_map_eval_row",
    "_map_surface_row",
    "_map_proactive_row",
    "_map_anticipate_row",
    "_map_draft_decision_row",
    "_map_brief_feedback_row",
    "_map_cpu_row",
    "_map_memory_row",
]


_QUERIES_COLS = ("ts", "trace_id", "cmd", "q", "session", "mode", "top_score",
                 "t_retrieve", "t_gen", "answer_len", "citation_repaired",
                 "critique_fired", "critique_changed")
_QUERIES_JSON = (("variants", "variants_json"), ("paths", "paths_json"),
                 ("scores", "scores_json"), ("filters", "filters_json"),
                 ("bad_citations", "bad_citations_json"))


def _map_queries_row(ev: dict) -> dict:
    out: dict = {}
    for k in _QUERIES_COLS:
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    for src_key, dst_col in _QUERIES_JSON:
        if src_key in ev and ev[src_key] is not None:
            out[dst_col] = ev[src_key]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    # Coerce bool→int for SQLite-friendly ints
    for bk in ("citation_repaired", "critique_fired", "critique_changed"):
        if bk in out and isinstance(out[bk], bool):
            out[bk] = int(out[bk])
    known = set(_QUERIES_COLS) | {k for k, _ in _QUERIES_JSON}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    # q is NOT NULL — supply empty string as fallback rather than let INSERT fail
    out.setdefault("q", "")
    return out


def _map_behavior_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "trace_id", "source", "event", "path", "query", "rank", "dwell_s"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    out.setdefault("source", "unknown")
    out.setdefault("event", "unknown")
    known = {"ts", "trace_id", "source", "event", "path", "query", "rank", "dwell_s"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_feedback_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "turn_id", "rating", "q", "scope"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "paths" in ev and ev["paths"] is not None:
        out["paths_json"] = ev["paths"]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    out.setdefault("rating", 0)
    known = {"ts", "turn_id", "rating", "q", "scope", "paths"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_tune_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "cmd", "samples", "seed", "n_cases", "delta",
              "eval_hit5_singles", "eval_hit5_chains", "rolled_back"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    for src_key, dst_col in (("baseline", "baseline_json"),
                              ("best", "best_json")):
        if src_key in ev and ev[src_key] is not None:
            out[dst_col] = ev[src_key]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    if "rolled_back" in out and isinstance(out["rolled_back"], bool):
        out["rolled_back"] = int(out["rolled_back"])
    known = {"ts", "cmd", "samples", "seed", "n_cases", "delta",
             "eval_hit5_singles", "eval_hit5_chains", "rolled_back",
             "baseline", "best"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_contradiction_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "subject_path", "helper_raw", "skipped"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "contradicts" in ev:
        out["contradicts_json"] = ev["contradicts"] or []
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    out.setdefault("subject_path", "")
    return out


def _map_ambient_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "trace_id", "cmd", "path", "hash"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    known = {"ts", "trace_id", "cmd", "path", "hash"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["payload_json"] = extra
    return out


def _map_ambient_state_row(path: str, h: str, analyzed_at: float,
                            payload: dict) -> dict:
    out: dict = {"path": path, "hash": h, "analyzed_at": analyzed_at}
    if payload:
        out["payload_json"] = payload
    return out


def _map_brief_written_row(brief_type: str, brief_path: str,
                            paths_cited: list, citations_by_section: dict) -> dict:
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "brief_type": brief_type,
        "brief_path": brief_path,
        "paths_cited_json": list(paths_cited or []),
        "citations_by_section_json": citations_by_section or {},
    }


def _map_brief_state_row(brief_path: str, cited_path: str) -> dict:
    """brief_state is an upsert keyed by (brief_path, cited_path). First write
    sets first_ts = last_ts; subsequent writes to the same pair move last_ts
    forward. The writer is called only after `_brief_state_seen` returns False
    so in practice we never update — but the schema supports it."""
    ts = datetime.now().isoformat(timespec="seconds")
    pair_key = f"{brief_path}\x00{cited_path}"
    bt = "today" if brief_path.endswith("-evening.md") else (
        "morning" if (
            "/00-Inbox/reviews/" in brief_path or brief_path.startswith("00-Inbox/reviews/")
            # backward-compat: rows historicos escritos cuando reviews vivia en 99-AI/
            or "/99-obsidian/99-AI/reviews/" in brief_path or brief_path.startswith("99-obsidian/99-AI/reviews/")
        ) else "unknown"
    )
    return {"pair_key": pair_key, "brief_type": bt, "kind": "cited",
            "path": cited_path, "first_ts": ts, "last_ts": ts}


def _map_wa_tasks_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "since", "chats", "items", "path"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    known = {"ts", "since", "chats", "items", "path"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_archive_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "cmd", "min_age_days", "query_window_days", "folder",
              "dry_run", "force", "gate", "n_candidates", "n_plan",
              "n_applied", "n_skipped", "gated", "batch_path"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    for bk in ("dry_run", "force", "gated"):
        if bk in out and isinstance(out[bk], bool):
            out[bk] = int(out[bk])
    known = {"ts", "cmd", "min_age_days", "query_window_days", "folder",
             "dry_run", "force", "gate", "n_candidates", "n_plan",
             "n_applied", "n_skipped", "gated", "batch_path"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_filing_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "cmd", "path", "note", "folder", "confidence",
              "upward_title", "upward_kind"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "neighbors" in ev and ev["neighbors"] is not None:
        out["neighbors_json"] = ev["neighbors"]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    known = {"ts", "cmd", "path", "note", "folder", "confidence",
             "upward_title", "upward_kind", "neighbors"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_eval_row(entry: dict) -> dict:
    out: dict = {"ts": entry.get("ts") or datetime.now().isoformat(timespec="seconds")}
    sg = entry.get("singles") or {}
    ch = entry.get("chains") or {}
    for src_key, dst in (("hit5", "singles_hit5"), ("mrr", "singles_mrr"),
                          ("n", "singles_n")):
        if src_key in sg:
            out[dst] = sg[src_key]
    for src_key, dst in (("hit5", "chains_hit5"), ("mrr", "chains_mrr"),
                          ("chain_success", "chains_chain_success"),
                          ("turns", "chains_turns"), ("chains", "chains_n")):
        if src_key in ch:
            out[dst] = ch[src_key]
    # Preserve the full nested singles/chains dicts in extra_json so richer
    # fields (bootstrap CI bounds, p50/p95 latency) aren't lost to the flat
    # column projection.
    extra: dict = {}
    if sg:
        extra["singles"] = sg
    if ch:
        extra["chains"] = ch
    for k, v in entry.items():
        if k in {"ts", "singles", "chains"}:
            continue
        extra[k] = v
    if extra:
        out["extra_json"] = extra
    return out


def _map_surface_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "cmd", "n_pairs", "sim_threshold", "min_hops", "top",
              "skip_young_days", "llm", "duration_ms"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    if "llm" in out and isinstance(out["llm"], bool):
        out["llm"] = int(out["llm"])
    known = {"ts", "cmd", "n_pairs", "sim_threshold", "min_hops", "top",
             "skip_young_days", "llm", "duration_ms"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_proactive_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "kind", "sent", "reason"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    if "sent" in out and isinstance(out["sent"], bool):
        out["sent"] = int(out["sent"])
    known = {"ts", "kind", "sent", "reason"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_anticipate_row(ev: dict) -> dict:
    """Mapper para rag_anticipate_candidates. Los flags 0/1 se normalizan
    desde bool si vienen así."""
    out: dict = {}
    for k in ("ts", "kind", "score", "dedup_key", "reason", "message_preview"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    for flag in ("selected", "sent"):
        v = ev.get(flag, 0)
        out[flag] = int(bool(v))
    return out


def _map_draft_decision_row(ev: dict) -> dict:
    """Mapper para rag_draft_decisions. Las cols `*_json` aceptan dict/list
    y se serializan automáticamente en `_sql_append_event`."""
    out: dict = {}
    for k in ("ts", "draft_id", "contact_jid", "contact_name",
              "bot_draft", "decision", "sent_text"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    if "original_msgs" in ev and ev["original_msgs"] is not None:
        out["original_msgs_json"] = ev["original_msgs"]
    if "extra" in ev and ev["extra"] is not None:
        out["extra_json"] = ev["extra"]
    return out


def _map_brief_feedback_row(ev: dict) -> dict:
    """Mapper para rag_brief_feedback. Shape simple: ts/dedup_key/rating/
    reason/source. NO hay extra_json — si en el futuro queremos meter
    más metadata por feedback (chat origin, latency entre push y
    reply, etc.), agregamos columna nueva via migration."""
    out: dict = {}
    for k in ("ts", "dedup_key", "rating", "reason", "source"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    return out


def _map_cpu_row(ev: dict) -> dict:
    out: dict = {}
    for k in ("ts", "total_pct", "ncores", "interval_s"):
        if k in ev and ev[k] is not None:
            out[k] = ev[k]
    if "by_category" in ev:
        out["by_category_json"] = ev["by_category"]
    if "top" in ev:
        out["top_json"] = ev["top"]
    if "ts" not in out:
        out["ts"] = datetime.now().isoformat(timespec="seconds")
    known = {"ts", "total_pct", "ncores", "interval_s", "by_category", "top"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out


def _map_memory_row(ev: dict) -> dict:
    out: dict = {"ts": ev.get("ts") or datetime.now().isoformat(timespec="seconds")}
    if "total_mb" in ev and ev["total_mb"] is not None:
        out["total_mb"] = ev["total_mb"]
    if "by_category" in ev:
        out["by_category_json"] = ev["by_category"]
    if "top" in ev:
        out["top_json"] = ev["top"]
    if "vm" in ev:
        out["vm_json"] = ev["vm"]
    known = {"ts", "total_mb", "by_category", "top", "vm"}
    extra = {k: v for k, v in ev.items() if k not in known}
    if extra:
        out["extra_json"] = extra
    return out
