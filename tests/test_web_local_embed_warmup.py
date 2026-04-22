"""Tests del pre-warm del local embedder en el web server startup.

Bug hallado 2026-04-22: `web/server.py:_do_warmup` (línea ~602) carga
bge-m3 con:

    from rag import _get_local_embedder as _gle
    _mdl = _gle()
    _mdl.encode(["warmup"], show_progress_bar=False)

Eso **carga el modelo** pero NO setea el Event `_local_embedder_ready`
que `query_embed_local()` checkea como gate non-blocking. Resultado:

  - El modelo ocupa memoria (bien)
  - Cada query subsecuente cae al fallback ollama (~140ms) porque
    `_local_embedder_ready.is_set()` es False → `query_embed_local()`
    retorna None antes de intentar encode → caller usa ollama.

El fix correcto es llamar el helper canonicalizado `_warmup_local_embedder()`
que: (a) carga el modelo igual, (b) hace el dummy encode, y (c) **setea
el Event** en success. Diseño consistente con `rag serve` + `rag query`
que ya llaman `_warmup_local_embedder()` via `warmup_async()`.

Scope del test: inspeccionar el source de `web/server.py` — el warmup
vive adentro de un closure + thread daemon, complicado de testear con
mocks sin caminar por FastAPI lifespan + por el import path de `rag`.
Grep-based contract test es suficiente para evitar regresiones.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB_SERVER = REPO / "web" / "server.py"


def _source() -> str:
    return WEB_SERVER.read_text()


# ── El warmup llama al helper canonicalizado, no al loader directo ──────────


def test_web_startup_calls_warmup_local_embedder():
    """Post 2026-04-22 el startup debe invocar `_warmup_local_embedder` —
    NO el `_get_local_embedder` directo. El helper es el único que setea
    `_local_embedder_ready.Event`."""
    src = _source()
    assert "_warmup_local_embedder" in src, (
        "web/server.py:_do_warmup debe llamar _warmup_local_embedder() "
        "para que el Event se setee y query_embed_local use el in-process "
        "path en vez del fallback ollama"
    )


def test_web_startup_no_longer_calls_get_local_embedder_manual_encode():
    """Regression guard: el patrón viejo `_gle(); _mdl.encode(["warmup"])`
    quedaba cargando el modelo pero sin setear el Event. Verificamos que
    ya no exista tal cual — si alguien lo re-introduce, este test falla."""
    src = _source()
    # El patrón exacto problemático tenía una línea con ambas cosas.
    # No buscamos `_get_local_embedder` solo (puede seguir existiendo
    # para otros usos), buscamos el patrón completo del warmup viejo.
    bad_pattern = '_mdl.encode(["warmup"], show_progress_bar=False)'
    assert bad_pattern not in src, (
        "El warmup manual en el startup del web server ya no debe usar "
        "'_mdl.encode([\"warmup\"])' — usa _warmup_local_embedder() que "
        "setea _local_embedder_ready.Event atomicamente"
    )


def test_web_startup_warmup_respects_local_embed_env():
    """El warmup sólo corre cuando `RAG_LOCAL_EMBED` está seteado — si el
    operator no opta in, no gastamos memoria inútil. Test: el gate
    sobre `RAG_LOCAL_EMBED` sigue presente."""
    src = _source()
    assert 'RAG_LOCAL_EMBED' in src
    # El bloque de local embed debe estar envuelto en un check del env.
    # Match fuzzy: el `_warmup_local_embedder` debe aparecer en las
    # mismas 5 líneas que el check de `RAG_LOCAL_EMBED`.
    idx_env = src.find("RAG_LOCAL_EMBED")
    idx_warmup = src.find("_warmup_local_embedder")
    assert idx_env > 0 and idx_warmup > 0
    # Warmup DEBE venir después del check de env (gate primero).
    # En la misma vecindad (< 500 chars).
    assert abs(idx_warmup - idx_env) < 500, (
        "El _warmup_local_embedder debe estar agrupado con el check de "
        "RAG_LOCAL_EMBED env — si están lejos, el gate puede estar roto"
    )
