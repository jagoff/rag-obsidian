"""Lifecycle del web server — warmup + idle sweeper.

Extracto de [`web/server.py`](web/server.py) (Phase W3c, refactor modular
2026-05-09). Vive acá:

- `_IDLE_SWEEPER_THREAD`, `_IDLE_SWEEPER_STOP` — estado del daemon.
- `_start_idle_sweeper()` — daemon thread que evicta el reranker tras
  inactividad (`RAG_RERANKER_NEVER_UNLOAD=1` lo deshabilita).
- `_stop_idle_sweeper()` — registrada como `@_on_shutdown` desde server.py.
- `_warmup()` — registrada como `@_on_startup` desde server.py.

Re-exportado desde [`web/server.py`](web/server.py) con
`from web._lifecycle import (...)`. Los decoradores `@_on_startup` /
`@_on_shutdown` se aplican EN server.py (no acá) porque populan listas
del propio módulo (`_startup_callbacks` / `_shutdown_callbacks`) y
porque hay 9+ otros call sites con decorador en server.py.

Cross-module dependencies: `_warmup` referencia 15+ helpers que viven
en `web.server` (`_load_home_cache`, `_ensure_*_prewarmer`,
`_resolve_web_chat_model`, `_FOLLOWUP_AGING_CACHE`, etc.). Para evitar
circular import, esas referencias se resuelven via lazy lookup
`from web import server as _ws` adentro del cuerpo de `_warmup`.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime


# Idle sweeper — evict reranker from MPS tras inactividad. Antes vivía
# inline dentro de `_warmup()`, lo extraje a top-level con stop event +
# singleton check (mismo patrón que samplers) para que el lifespan
# shutdown corte el daemon thread y los tests no acumulen zombies que
# segfaultean al teardown del módulo.
_IDLE_SWEEPER_THREAD: "threading.Thread | None" = None
_IDLE_SWEEPER_STOP = threading.Event()


def _start_idle_sweeper() -> None:
    """Arranca el daemon thread del idle sweeper. Idempotente."""
    global _IDLE_SWEEPER_THREAD
    if _IDLE_SWEEPER_THREAD is not None and _IDLE_SWEEPER_THREAD.is_alive():
        return
    _IDLE_SWEEPER_STOP.clear()

    def _idle_sweeper() -> None:
        """Evict the reranker from MPS after `_RERANKER_IDLE_TTL` of no
        activity. Keeps the Mac responsive when the user walks away from
        chat — 2-3 GB of unified memory freed on idle.

        By default eviction leaves the reranker unloaded until the next
        request. Operators that prefer latency over idle memory can opt
        back into the old deferred re-warm with
        RAG_RERANKER_REWARM_AFTER_IDLE=1.

        When `RAG_RERANKER_NEVER_UNLOAD=1` (env var) the sweeper loop
        still runs but skips `maybe_unload_reranker()` entirely.
        """
        from rag import maybe_unload_local_embedder, maybe_unload_reranker
        _never_unload = os.environ.get(
            "RAG_RERANKER_NEVER_UNLOAD", "",
        ).strip() not in ("", "0", "false", "no")
        _rewarm_after_idle = os.environ.get(
            "RAG_RERANKER_REWARM_AFTER_IDLE", "",
        ).strip().lower() in ("1", "true", "yes")
        if _never_unload:
            print(
                "[idle-sweep] RAG_RERANKER_NEVER_UNLOAD=1 — reranker pinned, "
                "sweeper disabled",
                flush=True,
            )
        while not _IDLE_SWEEPER_STOP.is_set():
            try:
                # `wait()` retorna True si el stop event se setea →
                # exit limpio (vs `time.sleep(120)` que ataba el thread
                # 2 minutos al shutdown).
                if _IDLE_SWEEPER_STOP.wait(timeout=120):
                    return
                if _never_unload:
                    continue
                if maybe_unload_reranker():
                    print("[idle-sweep] reranker unloaded from MPS", flush=True)
                    if _rewarm_after_idle:
                        def _deferred_rewarm() -> None:
                            # Mismo trato: wait con stop event, exit limpio
                            # si el server cae antes de los 60s.
                            if _IDLE_SWEEPER_STOP.wait(timeout=60):
                                return
                            try:
                                from rag import get_reranker
                                get_reranker()
                                print(
                                    "[idle-sweep] reranker pre-warmed (deferred)",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(f"[reranker-rewarm] failed: {exc}", flush=True)

                        threading.Thread(
                            target=_deferred_rewarm,
                            name="reranker-rewarm",
                            daemon=True,
                        ).start()
                if maybe_unload_local_embedder():
                    print("[idle-sweep] local embedder unloaded from MLX", flush=True)
            except Exception as exc:
                print(f"[idle-sweep] error: {exc}", flush=True)

    _IDLE_SWEEPER_THREAD = threading.Thread(
        target=_idle_sweeper, name="idle-sweeper", daemon=True,
    )
    _IDLE_SWEEPER_THREAD.start()


def _stop_idle_sweeper() -> None:
    """Señaliza al idle sweeper + join breve."""
    _IDLE_SWEEPER_STOP.set()
    if _IDLE_SWEEPER_THREAD is not None:
        _IDLE_SWEEPER_THREAD.join(timeout=2.0)


def _warmup() -> None:
    """Hydrate the home cache and kick the bg prewarmer; DO NOT pin the
    chat model or reranker on boot.

    The previous version called `ollama.chat(model=command-r, ...)` on
    startup which pinned ~19 GB of unified memory even when the user was
    only browsing /dashboard or /. On a 36 GB Mac that's enough wired
    memory to starve the kernel and cause host freezes. Warmup of the
    expensive bits (reranker MPS init, chat model load) now happens
    on-demand from the first /api/chat request — 2-3s extra cost on one
    query in exchange for not holding the machine hostage when idle.

    Corpus/BM25/PageRank are cheap (RAM only, no VRAM), so we still
    preload them so the first retrieve is fast.

    REFACTOR W3c (2026-05-09): vive en `_lifecycle.py` pero llama a 15+
    helpers que viven en `web.server`. Para evitar circular imports al
    import-time, las referencias se resuelven via lazy lookup adentro
    del cuerpo. Este patrón también respeta el monkeypatch que hacen
    los tests (`monkeypatch.setattr(web_server, "_warmup", ...)` —
    aunque acá es defensivo: el TestClient sin context manager no
    invoca el lifespan).
    """
    # Lazy import — preserva back-compat con tests que monkeypatchan
    # símbolos sobre `web.server` (`_resolve_web_chat_model`, helpers
    # de followup_aging, etc.).
    from web import server as _ws  # noqa: PLC0415

    _ws._load_home_cache()
    _ws._ensure_home_prewarmer()
    _ws._ensure_chat_model_prewarmer()
    _ws._ensure_reranker_prewarmer()
    _ws._ensure_corpus_prewarmer()

    # Record this daemon startup in rag_ambient so restart count is queryable
    # via SQL instead of grepping web.log.
    try:
        import importlib.metadata as _imeta
        _rag_ver: "str | None" = _imeta.version("obsidian-rag")
    except Exception:
        _rag_ver = None
    try:
        _startup_payload: dict = {
            "pid": os.getpid(),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        if _rag_ver:
            _startup_payload["version"] = _rag_ver
        with _ws._ragvec_state_conn() as _sc:
            _ws._sql_append_event(_sc, "rag_ambient", {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "cmd": "serve.startup",
                "payload_json": _startup_payload,
            })
    except Exception as _exc:
        print(f"[warmup] startup event skipped: {_exc}", flush=True)

    # Memory-pressure watchdog — evita beachballs si el server + otras apps
    # saturan los 36 GB unified memory. Fires keep_alive=0 sobre el chat
    # model a los >85%, force-unload del reranker si sigue alto. Ver doc
    # extensa en rag.py alrededor de `_system_memory_used_pct()`.
    try:
        from rag import start_memory_pressure_watchdog
        start_memory_pressure_watchdog()
    except Exception as _exc:
        print(f"[warmup] memory-pressure watchdog skipped: {_exc}", flush=True)

    # WAL checkpointer — libera páginas del WAL cada 30s para que los
    # writers concurrentes (queries, behavior, cache) no peguen contra el
    # busy_timeout bajo carga sostenida. Audit 2026-04-24.
    try:
        from rag import start_wal_checkpointer
        start_wal_checkpointer()
    except Exception as _exc:
        print(f"[warmup] wal-checkpointer skipped: {_exc}", flush=True)

    def _do_warmup() -> None:
        try:
            from rag import get_db_for
            for _name, path in _ws.resolve_vault_paths(None):
                try:
                    col = get_db_for(path)
                    if col.count():
                        _ws._load_corpus(col)
                        _ws.get_pagerank(col)
                except Exception:
                    pass
            # End-to-end warmup: load the expensive singletons that the
            # first /api/chat would otherwise pay for (reranker on MPS +
            # bge-m3 SentenceTransformer + one dummy embed pass). Only
            # fires when the operator has opted into pinning them via the
            # same flags that keep them resident.
            if os.environ.get("RAG_RERANKER_NEVER_UNLOAD", "").strip() not in ("", "0", "false", "no"):
                try:
                    from rag import get_reranker as _get_rr
                    _rr = _get_rr()
                    _rr_device = getattr(getattr(_rr, "model", None), "device", "?")
                    print(f"[warmup] reranker loaded on {_rr_device}", flush=True)
                except Exception as _exc:
                    print(f"[warmup] reranker skipped: {_exc}", flush=True)
            # MLX chat-model prewarm (post-Ola-3 cutover, 2026-05-05):
            # opt-in only. The idle baseline must not pin ~5GB of Metal
            # memory just because the dashboard is open. Operators that
            # prefer hot chat startup can set RAG_MLX_PREWARM=1; the legacy
            # negative override RAG_MLX_NO_PREWARM=1 still wins.
            _mlx_prewarm = os.environ.get(
                "RAG_MLX_PREWARM", "",
            ).strip().lower() in ("1", "true", "yes")
            _mlx_no_prewarm = os.environ.get(
                "RAG_MLX_NO_PREWARM", "",
            ).strip().lower() in ("1", "true", "yes")
            if (
                os.environ.get("RAG_LLM_BACKEND", "mlx").strip().lower() == "mlx"
                and _mlx_prewarm
                and not _mlx_no_prewarm
            ):
                try:
                    from rag.llm_backend import get_backend as _get_bk
                    _bk = _get_bk()
                    if _bk.name == "mlx":
                        _chat_model = _ws._resolve_web_chat_model()
                        _t0 = time.time()
                        _bk._load(_chat_model)
                        print(
                            f"[warmup] mlx chat model {_chat_model} loaded in "
                            f"{time.time()-_t0:.1f}s",
                            flush=True,
                        )
                except Exception as _exc:
                    print(f"[warmup] mlx chat prewarm skipped: {_exc}", flush=True)
            _local_embed_prewarm = os.environ.get(
                "RAG_WEB_LOCAL_EMBED_PREWARM", "1",
            ).strip().lower() not in ("0", "false", "no")
            if (
                os.environ.get("RAG_LOCAL_EMBED", "").strip()
                not in ("", "0", "false", "no")
                and _local_embed_prewarm
            ):
                try:
                    # `_warmup_local_embedder()` es el único helper que, además
                    # de cargar el modelo y hacer un dummy encode, setea el
                    # `_local_embedder_ready.Event` que `query_embed_local()`
                    # checkea como gate non-blocking. Antes del 2026-04-22
                    # este startup llamaba `_get_local_embedder()` + `.encode`
                    # directo, lo que cargaba el modelo pero NO seteaba el
                    # Event → cada `/api/chat` caía al fallback ollama
                    # (~140ms vs ~10-30ms local). Ver
                    # test_web_local_embed_warmup.py para el contrato.
                    from rag import _warmup_local_embedder
                    if _warmup_local_embedder():
                        print("[warmup] bge-m3 local embedder ready (event set)", flush=True)
                    else:
                        print("[warmup] bge-m3 local embedder skipped (load/encode failed)", flush=True)
                except Exception as _exc:
                    print(f"[warmup] local embed skipped: {_exc}", flush=True)
            elif os.environ.get("RAG_LOCAL_EMBED", "").strip() not in ("", "0", "false", "no"):
                print(
                    "[warmup] local embedder prewarm skipped "
                    "(RAG_WEB_LOCAL_EMBED_PREWARM=0)",
                    flush=True,
                )
            # Drain any conversation turns that failed to persist on previous
            # runs (transient SQL busy / disk full / etc). Best-effort;
            # survivors stay in the file for the next startup.
            try:
                _retried = _ws._retry_pending_conversation_turns()
                if _retried:
                    print(f"[warmup] retried {_retried} pending conversation turn(s)",
                          flush=True)
            except Exception as _exc:
                print(f"[warmup] conversation-turn retry failed: {_exc}", flush=True)
            # Hydrate followup_aging cache (2026-04-30 + 2026-05-01).
            # Cold path real medido: ~9 min (find_followup_loops + LLM-judge
            # por cada uno de ~95 open loops) y carga el chat model en
            # MLX/Metal. Pagar ESO en cada restart del web server es
            # inaceptable para el baseline idle.
            #
            # Strategy en 2 fases:
            #   1. Si hay un payload fresh (<24h) en disk, hidratá in-memory
            #      desde ahí — instant. Cubre el caso "kickstart después de
            #      sesión normal" donde el cache anterior sigue siendo válido.
            #   2. Si no hay payload fresh en disk, no computar por default.
            #      `RAG_WEB_FOLLOWUP_AGING_COMPUTE=1` restaura el cold compute
            #      para instalaciones que prefieren pagar memoria/latencia a
            #      cambio de tener ese panel siempre fresco.
            try:
                hydrated = _ws._followup_aging_hydrate_from_disk_if_needed()
                if hydrated:
                    print(
                        f"[warmup] followup_aging hydrated from disk "
                        f"(total={(_ws._FOLLOWUP_AGING_CACHE['payload'] or {}).get('total', 0)}, "
                        f"age={int(time.time() - _ws._FOLLOWUP_AGING_CACHE['ts'])}s)",
                        flush=True,
                    )
                elif _ws._followup_aging_web_compute_enabled():
                    t0_fw = time.time()
                    _compute_followup_aging_result = _ws._compute_followup_aging()
                    if _compute_followup_aging_result is not None:
                        now = time.time()
                        _ws._FOLLOWUP_AGING_CACHE["ts"] = now
                        _ws._FOLLOWUP_AGING_CACHE["payload"] = _compute_followup_aging_result
                        _ws._followup_aging_persist(
                            now, _compute_followup_aging_result,
                            elapsed_s=now - t0_fw,
                        )
                        print(
                            f"[warmup] followup_aging cache pre-warmed in "
                            f"{now - t0_fw:.1f}s "
                            f"(total={_compute_followup_aging_result.get('total', 0)}) "
                            f"+ persisted to disk",
                            flush=True,
                        )
                else:
                    print(
                        "[warmup] followup_aging cold compute skipped "
                        "(RAG_WEB_FOLLOWUP_AGING_COMPUTE=0)",
                        flush=True,
                    )
            except Exception as _exc:
                print(f"[warmup] followup_aging pre-warm failed: {_exc}", flush=True)
        except Exception:
            pass

    threading.Thread(target=_do_warmup, daemon=True).start()

    # 2026-05-01 (afternoon): bloquear el startup hook hasta que
    # `_local_embedder_ready` Event esté set (o un cap de 20s
    # elapse) cuando RAG_LOCAL_EMBED=1. Pre-fix, los primeros 5-6
    # /api/chat post-restart pagaban embed_ms = 4-7s porque el
    # warmup async todavía no había seteado el Event y el path
    # de retrieve esperaba 6s antes de fallback-ear a ollama
    # (que ALSO arrancaba cold en algunos casos). El user veía
    # las primeras búsquedas post-restart muy lentas.
    #
    # Trade-off: el daemon tarda 5-15s más en arrancar (el
    # `_warmup_local_embedder()` carga bge-m3 SentenceTransformer
    # + 1 dummy encode = ~5s en M3 Max warm disk). Acceptable —
    # el daemon ya tiene `ThrottleInterval=30s` en el plist así
    # que no rebota tan rápido. Skip via env si hace falta:
    # `RAG_WEB_BLOCK_ON_EMBED_WARMUP=0` restaura el legacy
    # non-blocking startup.
    _block_embed_str = os.environ.get(
        "RAG_WEB_BLOCK_ON_EMBED_WARMUP", "1"
    ).strip().lower()
    if (
        _block_embed_str not in ("0", "false", "no")
        and os.environ.get("RAG_LOCAL_EMBED", "").strip()
            not in ("", "0", "false", "no")
        and os.environ.get("RAG_WEB_LOCAL_EMBED_PREWARM", "1").strip().lower()
            not in ("0", "false", "no")
    ):
        try:
            from rag import _local_embedder_ready
            _t0_embed_block = time.perf_counter()
            _ready = _local_embedder_ready.wait(timeout=20.0)
            _block_ms = int((time.perf_counter() - _t0_embed_block) * 1000)
            if _ready:
                print(
                    f"[warmup] embedder ready in {_block_ms}ms — "
                    f"first query will skip cold-load",
                    flush=True,
                )
            else:
                print(
                    f"[warmup] embedder NOT ready after {_block_ms}ms — "
                    f"continuing anyway, first queries may pay cold-load tax",
                    flush=True,
                )
        except Exception as _exc:
            print(
                f"[warmup] embedder block-on-warmup skipped: "
                f"{type(_exc).__name__}: {_exc}",
                flush=True,
            )

    _start_idle_sweeper()


__all__ = [
    "_IDLE_SWEEPER_THREAD",
    "_IDLE_SWEEPER_STOP",
    "_start_idle_sweeper",
    "_stop_idle_sweeper",
    "_warmup",
]
