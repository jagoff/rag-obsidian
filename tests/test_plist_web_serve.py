"""Tests for the web + serve launchd plist generators.

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
daemon invokes `python web/server.py`, serve invokes `rag serve`) imports any
module transitively touching huggingface_hub before `rag.py`'s setdefault
fires, the offline flags miss their window. Setting them explicitly in the
plist dict closes that race.
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


def test_web_plist_has_rag_memory_pressure_interval_20():
    d = _parse(rag_module._web_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_MEMORY_PRESSURE_INTERVAL") == "20", \
        "Default 60s misses the MPS-OOM window measured in web.error.log"


def test_web_plist_has_rag_local_embed():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["EnvironmentVariables"].get("RAG_LOCAL_EMBED") == "1"


def test_web_plist_has_rag_reranker_never_unload():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["EnvironmentVariables"].get("RAG_RERANKER_NEVER_UNLOAD") == "1"


def test_web_plist_has_ollama_keep_alive():
    d = _parse(rag_module._web_plist(RAG_BIN))
    assert d["EnvironmentVariables"].get("OLLAMA_KEEP_ALIVE") == "-1"


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


# ─────────────────────────── _serve_plist ──────────────────────────────────


def test_serve_plist_has_hf_hub_offline():
    d = _parse(rag_module._serve_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("HF_HUB_OFFLINE") == "1"


def test_serve_plist_has_transformers_offline():
    d = _parse(rag_module._serve_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("TRANSFORMERS_OFFLINE") == "1"


def test_serve_plist_has_rag_memory_pressure_interval_20():
    d = _parse(rag_module._serve_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("RAG_MEMORY_PRESSURE_INTERVAL") == "20"


@requires_plutil
def test_serve_plist_plutil_lint():
    xml = rag_module._serve_plist(RAG_BIN)
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
