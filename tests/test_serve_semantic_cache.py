"""Tests del wiring del semantic cache en `rag serve` /query (post 2026-04-23).

Cierra el último caller del semantic cache: serve endpoint
(WhatsApp listener + external bots). El LRU in-memory del serve
(`_serve_cache`, 64 entries × 5min TTL) ya existía; el SQL semantic cache
se agrega como segundo layer post LRU miss — mismo patrón que /api/chat
(web server) + rag query (CLI).

`_handle_query` es una closure dentro de `serve()`, así que los tests
grep el source + validan los contratos de shape + flow, igual que los
tests pre-existentes de serve (ej. test_serve_fast_path_consumption.py,
test_serve_short_circuits.py).
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402


_SRC = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")


def _serve_handler_body() -> str:
    """Slice del source que corresponde al cuerpo de `_handle_query` del serve.

    El closure es grande (~55kB) porque incluye weather + tasks + retrieve +
    LLM + post-process + cache logic. Cortamos hasta el siguiente `def `
    (siguiente closure hermana dentro del serve).
    """
    idx = _SRC.find("def _handle_query(body: dict)")
    assert idx >= 0, "serve _handle_query not found"
    # Siguiente closure-def dentro del mismo serve() es `_handle_chat`.
    end_idx = _SRC.find("def _handle_chat(body: dict)", idx)
    if end_idx < 0:
        end_idx = idx + 60000  # fallback seguro
    return _SRC[idx:end_idx]


# ── Wiring presence ─────────────────────────────────────────────────────────


def test_serve_has_semantic_cache_lookup():
    body = _serve_handler_body()
    assert "semantic_cache_lookup" in body, (
        "serve _handle_query missing `semantic_cache_lookup` call — "
        "the semantic cache wiring from the 2026-04-23 push didn't land"
    )
    assert "return_probe=True" in body, (
        "serve semantic lookup must request probe metadata for telemetry"
    )


def test_serve_has_semantic_cache_store():
    body = _serve_handler_body()
    assert "semantic_cache_store" in body, (
        "serve _handle_query missing `semantic_cache_store` call — "
        "post-pipeline persistence not wired"
    )
    # Must use background=True because serve es long-running daemon —
    # el atexit drop que afectaba al CLI one-shot NO aplica acá.
    # El store call específico del serve está cerca del `_cache_put(cache_key, payload)`.
    payload_put_idx = body.find("_cache_put(cache_key, payload)")
    assert payload_put_idx >= 0
    nearby = body[payload_put_idx : payload_put_idx + 1500]
    assert "background=True" in nearby, (
        "serve semantic_cache_store must use background=True (long-running daemon)"
    )


# ── Eligibility gates ───────────────────────────────────────────────────────


def test_serve_semantic_gate_excludes_history_and_filters():
    """Gate debe cerrar en: history present, force=True, qfolder, qtag."""
    body = _serve_handler_body()
    # Anchor: our eligibility var name.
    idx = body.find("_serve_sem_eligible = (")
    assert idx >= 0, (
        "missing `_serve_sem_eligible = (...)` assignment — "
        "eligibility logic must be explicit for downstream reviewers"
    )
    block = body[idx : idx + 400]
    assert "not history" in block
    assert "not force" in block
    assert "not qfolder" in block
    assert "not qtag" in block


def test_serve_semantic_lookup_only_when_eligible():
    """El lookup se gate-a en `if _serve_sem_eligible:` así queries
    con force/history/filter saltean embed + lookup completo."""
    body = _serve_handler_body()
    # Search for the guard right before the lookup call.
    assert "if _serve_sem_eligible:" in body


def test_serve_semantic_store_skipped_on_force_or_filters():
    """El store se gate-a en el mismo `if not force and not qfolder and
    not qtag:` que el LRU — consistencia entre las dos layers."""
    body = _serve_handler_body()
    # El cuerpo tiene MULTIPLES occurrences de ese guard (LRU get arriba,
    # LRU+semantic put al final). Buscamos desde el `_cache_put(cache_key,
    # payload)` hacia atrás para encontrar el guard que rodea al STORE
    # final.
    put_idx = body.find("_cache_put(cache_key, payload)")
    assert put_idx >= 0, "LRU put (payload) anchor missing"
    # Look for the guard in the 2000 chars before the put.
    preceding = body[max(0, put_idx - 2000):put_idx]
    assert "if not force and not qfolder and not qtag:" in preceding, (
        "LRU+semantic store guard missing right before _cache_put(cache_key, payload)"
    )
    # Buscar el semantic_cache_store en los 2500 chars después del put.
    following = body[put_idx : put_idx + 2500]
    assert "semantic_cache_store" in following, (
        "semantic_cache_store must live right after _cache_put(cache_key, "
        "payload), under the same `if not force and not qfolder and "
        "not qtag:` guard — otherwise force/filter queries leak into "
        "the cache"
    )


# ── Flow ordering ──────────────────────────────────────────────────────────


def test_serve_semantic_lookup_before_retrieve():
    """Lookup debe correr ANTES del `result = retrieve(...)` — si no
    short-circuitea antes de pagar el pipeline completo."""
    body = _serve_handler_body()
    lookup_idx = body.find("semantic_cache_lookup")
    retrieve_idx = body.find("result = retrieve(")
    assert lookup_idx >= 0
    assert retrieve_idx >= 0
    assert lookup_idx < retrieve_idx, (
        "semantic_cache_lookup must precede retrieve() — otherwise hits "
        "never short-circuit"
    )


def test_serve_semantic_hit_returns_payload():
    """Hit path devuelve un payload con `cache_layer="semantic"` +
    `cached=True` — le permite al cliente distinguirlo del LRU hit."""
    body = _serve_handler_body()
    assert '"cache_layer": "semantic"' in body, (
        "serve semantic hit payload missing cache_layer marker"
    )
    # Hit path debe hacer `return _sem_payload`.
    assert "return _sem_payload" in body


def test_serve_semantic_hit_logs_query_event():
    """Hit emite un log_query_event con cmd='serve.cached_semantic'."""
    body = _serve_handler_body()
    assert '"cmd": "serve.cached_semantic"' in body, (
        "semantic hit must log its own cmd bucket for analytics — "
        "otherwise serve hit rate is invisible in rag_queries"
    )


# ── Probe + source field consistency ───────────────────────────────────────


def test_serve_semantic_hit_log_includes_cache_probe():
    body = _serve_handler_body()
    # Find the semantic-hit log event
    log_idx = body.find('"cmd": "serve.cached_semantic"')
    assert log_idx >= 0
    nearby = body[log_idx : log_idx + 1500]
    assert '"cache_probe"' in nearby
    assert '"cache_cosine"' in nearby
    assert '"cache_age_seconds"' in nearby
    assert '"cache_hit"' in nearby


def test_serve_semantic_hit_synthesizes_sources_shape():
    """Sources sintetizados deben tener shape {note, path, score} —
    el listener WA parsea estos campos."""
    body = _serve_handler_body()
    # Anchor: the comprehension that builds _sem_sources.
    anchor = body.find("_sem_sources = [")
    assert anchor >= 0, "_sem_sources list comprehension not found"
    nearby = body[anchor : anchor + 500]
    assert '"note"' in nearby
    assert '"path"' in nearby
    assert '"score"' in nearby


# ── Hydration: semantic hit repopulates the LRU ────────────────────────────


def test_serve_semantic_hit_hydrates_lru():
    """Hit del semantic debe llamar _cache_put así el próximo
    exact-repeat pega O(1) sin re-embed + re-lookup."""
    body = _serve_handler_body()
    # The hydration lives inside the hit branch.
    hit_section_idx = body.find("_serve_sem_hit is not None:")
    assert hit_section_idx >= 0
    # Hit section corre hasta el return _sem_payload — ~2500 chars tope.
    hit_section = body[hit_section_idx : hit_section_idx + 3500]
    assert "_cache_put(cache_key" in hit_section, (
        "semantic hit must hydrate the LRU so next exact-match is O(1)"
    )


# ── Smoke test del semantic cache helper (orthogonal a serve wiring) ───────


def test_semantic_cache_lookup_still_available():
    """Sanity check — los helpers importables que serve usa siguen existiendo."""
    assert hasattr(rag, "semantic_cache_lookup")
    assert hasattr(rag, "semantic_cache_store")
    assert hasattr(rag, "_corpus_hash_cached")
    assert hasattr(rag, "_semantic_cache_enabled")
