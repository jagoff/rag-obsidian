"""Tests for the web launchd plist generator.

Gates the 3 env vars whose absence produced 23 MPS Command-Buffer OOMs + 64
`[local-embed] unavailable (couldn't connect to huggingface.co)` entries in
`web.error.log` (2026-04-22 audit):

  1. `HF_HUB_OFFLINE=1` — `sentence_transformers` / `huggingface_hub` hit the
     hub for HEAD requests on load even when the snapshot is cached. Without
     this, a flaky network → fallback to ollama embed (~140ms vs ~10-30ms
     local) AND a 400 MB bge-m3 download retry.
  2. `TRANSFORMERS_OFFLINE=1` — same rationale, gates the `transformers`-
     level probe that `sentence_transformers` inherits.
  3. `RAG_MEMORY_PRESSURE_INTERVAL=20` — the watchdog samples `vm_stat`
     every N seconds; the default 60s missed the MPS OOM window. 20s gives
     the watchdog 3 chances per minute to catch memory pressure before
     Metal returns `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

The env gets applied at module-init of `rag.py:21` via `os.environ.setdefault`,
but that runs AFTER launchd invokes the entry point. If the entry point (web
daemon invokes `python web/server.py`) imports any module transitively touching
huggingface_hub before `rag.py`'s setdefault fires, the offline flags miss
their window. Setting them explicitly in the plist dict closes that race.

Histórico (Fase 2a, 2026-05-09): los tests de `_serve_plist` se removieron
junto con la función — el daemon estaba deprecado desde 2026-05-01 y FastAPI
web cubre los endpoints reales.
"""
from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module

RAG_BIN = "/usr/local/bin/rag"

_HAS_PLUTIL = shutil.which("plutil") is not None
requires_plutil = pytest.mark.skipif(
    not _HAS_PLUTIL, reason="plutil is macOS-only"
)


def _parse(xml: str) -> dict:
    return plistlib.loads(xml.encode())


# ─────────────────────────── _web_plist ────────────────────────────────────


def test_web_plist_label():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["Label"] == "com.fer.obsidian-rag-web"


def test_web_plist_program_is_web_server_py():
    d = _parse(rag_module._web_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert args[-1].endswith("web/server.py"), args


def test_web_plist_paths_exist_on_disk():
    """Regression: 2026-04-26 the generator computed venv/server paths with a
    single ``Path(__file__).resolve().parent`` (one level up from
    ``rag/__init__.py``), which yields ``…/obsidian-rag/rag/.venv/...`` and
    ``…/obsidian-rag/rag/web/server.py`` — both non-existent. launchd
    therefore failed each spawn with exit 78 and the daemon stayed down
    until the installed plist was hand-patched. Repo root is two levels up.

    These assertions only make sense when run from the actual checkout, so
    skip when the venv is missing (e.g. pip-installed CI environments).

    Heurística vieja (`'/rag/.venv/' not in args[0]`) era frágil: cuando el
    repo vive en `/Users/fer/repos/rag/` el path correcto contiene
    `/rag/.venv/` y la heurística disparaba false-positive. Reemplazada
    por replicación exacta del cómputo de ``rag/plists.py:_web_plist``
    (mismo `Path(__file__).resolve().parent.parent` — `__file__` es
    `rag/plists.py` ahí, `rag/__init__.py` acá; ambos colapsan al
    repo_root) + asserción de equality contra `args[0]` / `args[-1]`."""
    # Replica el cómputo de _web_plist: parent.parent desde el módulo `rag`
    # llega al repo_root igual que parent.parent desde `rag/plists.py`.
    repo_root = Path(rag_module.__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    web_server = repo_root / "web" / "server.py"
    if not venv_python.exists():
        pytest.skip("no local .venv — likely a pip-installed CI checkout")
    d = _parse(rag_module._web_plist(RAG_BIN))
    args = d["ProgramArguments"]
    # Exact-equality check: el plist tiene que apuntar al MISMO interpreter
    # y MISMO server.py que computa _web_plist desde su propio __file__.
    assert args[0] == str(venv_python), (
        f"plist python interpreter mismatch:\n"
        f"  expected: {venv_python}\n  got:      {args[0]}"
    )
    assert args[-1] == str(web_server), (
        f"plist web server.py mismatch:\n"
        f"  expected: {web_server}\n  got:      {args[-1]}"
    )
    # Y existencia real on-disk (cubre el bug 2026-04-26 directamente).
    assert venv_python.is_file(), \
        f"plist python interpreter does not exist on disk: {venv_python!r}"
    assert web_server.is_file(), \
        f"plist web server.py does not exist on disk: {web_server!r}"
    wd = Path(d["WorkingDirectory"])
    assert wd.is_dir(), f"plist WorkingDirectory does not exist: {wd!r}"
    assert wd == repo_root, (
        f"plist WorkingDirectory mismatch:\n"
        f"  expected: {repo_root}\n  got:      {wd}"
    )


def test_web_plist_has_hf_hub_offline():
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("HF_HUB_OFFLINE") == "1", \
        "Without HF_HUB_OFFLINE=1 the daemon makes HEAD requests to huggingface.co on load"


def test_web_plist_has_transformers_offline():
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("TRANSFORMERS_OFFLINE") == "1", \
        "sentence_transformers inherits this flag; without it the HEAD probe still fires"


def test_web_plist_has_fastembed_cache_path():
    """Pin fastembed's model cache to ~/.cache/fastembed.

    Default upstream cache is `tempfile.gettempdir()/fastembed_cache` which on
    macOS resolves to `/var/folders/.../T/fastembed_cache`, a path the OS GCs.
    Combined with `HF_HUB_OFFLINE=1` (above) a cache miss after the GC means
    `fastembed.SparseTextEmbedding('Qdrant/bm25')` (loaded by mem0's qdrant
    vector store for hybrid keyword search) can't redownload and the encoder
    fails. See web.error.log 2026-04-29:
        Could not find the model tar.gz file at /var/folders/.../T/
        fastembed_cache/bm25 and local_files_only=True
    """
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    cache = env.get("FASTEMBED_CACHE_PATH", "")
    home = str(Path.home())
    assert cache.startswith(home) and cache.endswith(".cache/fastembed"), \
        f"FASTEMBED_CACHE_PATH should be HOME/.cache/fastembed (persistent), got {cache!r}"


def test_web_plist_has_rag_memory_pressure_interval_20():
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_MEMORY_PRESSURE_INTERVAL") == "20", \
        "Default 60s misses the MPS-OOM window measured in web.error.log"


def test_web_plist_has_rag_local_embed():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["EnvironmentVariables"].get("RAG_LOCAL_EMBED") == "1"


def test_web_plist_lazy_loads_local_embedder():
    """Web should not pin the query embedder before the first search."""
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_WEB_LOCAL_EMBED_PREWARM") == "0"
    assert env.get("RAG_WEB_BLOCK_ON_EMBED_WARMUP") == "0"
    assert env.get("RAG_LOCAL_EMBED_WAIT_MS") == "0"
    assert env.get("RAG_LOCAL_EMBED_IDLE_TTL") == "300"


def test_web_plist_disables_mlx_boot_prewarm():
    """Idle web startup must not pin the 7B MLX chat model in Metal memory."""
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_MLX_NO_PREWARM") == "1"


def test_web_plist_has_short_mlx_idle_ttl():
    """Resident MLX models should be evicted promptly after chat idle."""
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_MLX_IDLE_TTL") == "180"


def test_web_plist_disables_reranker_eager_rewarm():
    """Reranker should not reload itself after idle-unload by default."""
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_WEB_RERANKER_PREWARM") == "0"
    assert env.get("RAG_RERANKER_REWARM_AFTER_IDLE") == "0"


def test_web_plist_disables_followup_aging_compute():
    """Home dashboard should not pin the chat model via followup-aging by default."""
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_WEB_FOLLOWUP_AGING_COMPUTE") == "0"


def test_web_plist_has_rag_reranker_never_unload():
    # 2026-05-10: RAG_RERANKER_NEVER_UNLOAD=1 fue removido del web plist porque
    # el memory watchdog maneja el unload dinámicamente (ya no se pinea siempre).
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert "RAG_RERANKER_NEVER_UNLOAD" not in d["EnvironmentVariables"], (
        "RAG_RERANKER_NEVER_UNLOAD fue removido del web plist (2026-05-10) "
        "porque el memory watchdog maneja el unload dinámicamente"
    )


def test_web_plist_no_ollama_keep_alive():
    """Ola 6 cero-Ollama (2026-05-06): modelos chat Ollama purgados del disco.
    El web plist NO debe setear LLM_KEEP_ALIVE ni OLLAMA_MAX_LOADED_MODELS
    — no hay daemon Ollama que keep-alivear.
    """
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert "LLM_KEEP_ALIVE" not in env, \
        "Ola 6: modelos Ollama purgados — LLM_KEEP_ALIVE no debe estar en web plist"
    assert "OLLAMA_MAX_LOADED_MODELS" not in env, \
        "Ola 6: OLLAMA_MAX_LOADED_MODELS no debe estar en web plist"


def test_indexers_have_local_embed_and_no_ollama_keepalive():
    """Ola 6 cero-Ollama: todos los plists indexers tienen RAG_INDEX_LOCAL_EMBED=1
    y NO tienen LLM_KEEP_ALIVE (embed path es ahora in-process).
    """
    indexer_plists = [
        ("watch", rag_module._watch_plist(RAG_BIN)),
        ("ingest_whatsapp", rag_module._ingest_whatsapp_plist(RAG_BIN)),
        ("ingest_cross_source", rag_module._ingest_cross_source_plist(RAG_BIN)),
    ]
    for name, xml in indexer_plists:
        d = _parse(xml)
        env = d["EnvironmentVariables"]
        assert env.get("RAG_INDEX_LOCAL_EMBED") == "1", \
            f"{name}: falta RAG_INDEX_LOCAL_EMBED=1 (Ola 6 cero-Ollama)"
        assert "LLM_KEEP_ALIVE" not in env, \
            f"{name}: LLM_KEEP_ALIVE no debe estar (Ola 6 cero-Ollama)"


def test_web_plist_keepalive_runatload():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d.get("KeepAlive") is True
    assert d.get("RunAtLoad") is True


def test_web_plist_has_throttle_interval():
    d = _parse(rag_module._web_plist(RAG_BIN))
    # Prevent crash-loop burning CPU; matches the value already installed manually.
    assert d.get("ThrottleInterval") == 30


def test_web_plist_stdout_stderr_paths():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["StandardOutPath"].endswith("web.log")
    assert d["StandardErrorPath"].endswith("web.error.log")


@requires_plutil
def test_web_plist_plutil_lint():
    xml = rag_module._web_plist(RAG_BIN)
    result = subprocess.run(
        ["plutil", "-lint", "-"], input=xml.encode(), capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


# ─────────────────────────── _services_spec ────────────────────────────────


def test_services_spec_includes_web():
    """`rag setup` must install the web plist alongside the others — it was
    missing historically (the plist was created manually outside setup) and
    therefore had no single source of truth for its env vars."""
    labels = [lbl for lbl, _, _ in rag_module._services_spec(RAG_BIN)]
    assert "com.fer.obsidian-rag-web" in labels


def test_services_spec_web_plist_xml_lintable():
    for label, _, content in rag_module._services_spec(RAG_BIN):
        if label == "com.fer.obsidian-rag-web":
            plistlib.loads(content.encode())  # raises on invalid
            return
    pytest.fail("web plist not in _services_spec")
