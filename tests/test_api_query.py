"""Tests del endpoint `POST /api/query` (sync JSON wrapper para WA listener).

Reemplazo del `/query` legacy del `rag serve --port 7832` (BaseHTTPServer
deprecated 2026-05-01). El endpoint sirve dos use-cases del listener:

  1. `loadVaultContextForDraft` (drafts WA cuando otros le escriben al user) —
     manda `{retrieve_only: true}` y consume `{sources: [...]}`.

  2. `ragQuery` (query directa del bot al user) — manda `{question, session_id}`
     y consume `{answer, sources, paths, ...}`.

Tests cubren:
  - Wire-format compatible con el legacy `/query` del rag serve.
  - Alias `/query` además del path canónico `/api/query`.
  - retrieve_only=true skipea LLM, devuelve solo sources.
  - Pregunta vacía / sin vaults → error JSON, no 500.
  - Sources mantienen el shape `{path, score, excerpt, source}`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import rag
from web import server as server_mod
from web.server import app


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        rag.DB_PATH = snap


@pytest.fixture(autouse=True)
def _reset_buckets():
    server_mod._CHAT_BUCKETS.clear()
    yield


def _canned_retrieve_result() -> dict:
    """Result mock con 2 notas y excerpt para verificar el shape."""
    return {
        "docs": [
            "Best Series all times — Breaking Bad, The Office, Black Mirror",
            "Peliculas y Series — links a sitios de streaming",
        ],
        "metas": [
            {"file": "04-Archive/Personal/Best Series all times.md",
             "note": "Best Series all times", "source": "vault"},
            {"file": "04-Archive/Personal/Peliculas y Series.md",
             "note": "Peliculas y Series", "source": "vault"},
        ],
        "scores": [0.531, 0.296],
        "confidence": 0.531,
        "search_query": "series",
        "query_variants": ["series"],
        "filters_applied": {},
    }


def _patch_retrieve(monkeypatch, result_factory=_canned_retrieve_result):
    """Patchea multi_retrieve + resolve_vault_paths para no pegarle al
    vault real. multi_retrieve se importa en server.py via `from rag
    import multi_retrieve`, así que el monkeypatch tiene que apuntar al
    binding del módulo server, no a rag."""
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: result_factory(),
    )
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )


# ── retrieve_only=True (drafts WA — 99% del tráfico) ────────────────────────


def test_api_query_retrieve_only_returns_sources_no_answer(monkeypatch):
    """Con `retrieve_only=true` el endpoint NO llama al LLM.

    El listener WhatsApp arma su propio prompt para drafts, así que pagar
    los 10-30s de un LLM call que va a descartar es overhead puro.
    Output debe tener `sources` pero NO `answer`.
    """
    _patch_retrieve(monkeypatch)

    client = TestClient(app)
    resp = client.post("/api/query", json={
        "question": "series y peliculas",
        "retrieve_only": True,
    })
    assert resp.status_code == 200
    data = resp.json()

    # No answer (LLM skip).
    assert "answer" not in data, f"retrieve_only=True NO debería incluir answer; got {data}"

    # Sources con shape compatible con loadVaultContextForDraft (listener.ts:12224+).
    assert "sources" in data
    assert len(data["sources"]) == 2
    for s in data["sources"]:
        assert "path" in s
        assert "score" in s
        assert "excerpt" in s
        assert "source" in s

    # Top-1 source corresponde al canned (`Best Series all times.md`, score=0.531).
    assert data["sources"][0]["path"] == "04-Archive/Personal/Best Series all times.md"
    assert data["sources"][0]["score"] == 0.531
    assert "Breaking Bad" in data["sources"][0]["excerpt"]
    assert data["sources"][0]["source"] == "vault"


def test_api_query_retrieve_only_excerpt_capped_to_500_chars(monkeypatch):
    """Excerpts en sources deben truncarse a 500 chars (mismo cap que
    el listener aplica downstream — duplicarlo acá ahorra bytes en el
    JSON respuesta y mantiene paridad con el wire-format del legacy)."""
    long_doc = "X" * 1500
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: {
            "docs": [long_doc],
            "metas": [{"file": "test.md", "note": "test", "source": "vault"}],
            "scores": [0.8],
            "confidence": 0.8,
            "search_query": "x", "query_variants": ["x"],
            "filters_applied": {},
        },
    )
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )
    client = TestClient(app)
    resp = client.post("/api/query", json={
        "question": "test",
        "retrieve_only": True,
    })
    assert resp.status_code == 200
    excerpt = resp.json()["sources"][0]["excerpt"]
    assert len(excerpt) == 500


def test_api_query_retrieve_only_paths_dedup(monkeypatch):
    """`paths` debe ser dedup-eado de las sources (el listener consume
    paths para tracking de cuáles notas están en el contexto del draft;
    duplicados serían ruido)."""
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: {
            # Misma nota aparece 2 veces (chunks distintos del mismo file).
            "docs": ["chunk 1 of A", "chunk 2 of A", "chunk 1 of B"],
            "metas": [
                {"file": "a.md", "note": "A", "source": "vault"},
                {"file": "a.md", "note": "A", "source": "vault"},
                {"file": "b.md", "note": "B", "source": "vault"},
            ],
            "scores": [0.9, 0.85, 0.7],
            "confidence": 0.9,
            "search_query": "x", "query_variants": ["x"],
            "filters_applied": {},
        },
    )
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )
    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "x", "retrieve_only": True})
    assert resp.status_code == 200
    data = resp.json()
    # 3 sources (todos los chunks).
    assert len(data["sources"]) == 3
    # Pero solo 2 paths únicos.
    assert data["paths"] == ["a.md", "b.md"]


# ── retrieve_only=False (query directa del bot) ──────────────────────────────


def test_api_query_full_path_calls_llm(monkeypatch):
    """Sin retrieve_only el endpoint corre retrieve + LLM call sync.

    Devuelve `answer`, `sources`, `paths`, `confidence`, `t_retrieve`,
    `t_gen`, `turn_id` — wire-format compatible con `ragQuery` del
    listener (listener.ts:1628-1650).
    """
    _patch_retrieve(monkeypatch)

    # Mock ollama.chat para no pegarle al daemon real.
    class _FakeMsg:
        def __init__(self, content): self.content = content

    class _FakeResp:
        def __init__(self, content): self.message = _FakeMsg(content)

    monkeypatch.setattr(
        server_mod.ollama, "chat",
        lambda **kw: _FakeResp("Tus series rankeadas: Breaking Bad, The Office, Black Mirror."),
    )

    client = TestClient(app)
    resp = client.post("/api/query", json={
        "question": "qué series tengo rankeadas",
        "session_id": "test-sess",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "Breaking Bad" in data["answer"]
    assert len(data["sources"]) == 2
    assert data["paths"] == [
        "04-Archive/Personal/Best Series all times.md",
        "04-Archive/Personal/Peliculas y Series.md",
    ]
    assert "turn_id" in data
    assert data["confidence"] == 0.531


def test_api_query_full_path_empty_corpus_returns_canned(monkeypatch):
    """Cuando el retrieve no devuelve nada, el endpoint NO llama al LLM
    (no hay context). Devuelve un answer canned + mode=no_results."""
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: {
            "docs": [], "metas": [], "scores": [],
            "confidence": float("-inf"),
            "search_query": "x", "query_variants": ["x"],
            "filters_applied": {},
        },
    )
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )

    # ollama.chat NO debe llamarse cuando no hay docs.
    called = {"n": 0}
    def _fake_chat(**kw):
        called["n"] += 1
        raise AssertionError("LLM call no debería dispararse con corpus vacío")
    monkeypatch.setattr(server_mod.ollama, "chat", _fake_chat)

    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "nada"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "no_results"
    assert "no encontré" in data["answer"].lower()
    assert data["sources"] == []
    assert called["n"] == 0


# ── Errores y validación ────────────────────────────────────────────────────


def test_api_query_no_vaults_returns_error(monkeypatch):
    """Sin vaults registrados → error JSON, no 500."""
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [],
    )
    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "test"})
    assert resp.status_code == 200
    assert resp.json() == {"error": "no vault registered"}


def test_api_query_empty_question_returns_422(monkeypatch):
    """Pregunta vacía no pasa el validator de Pydantic (min_length=1)."""
    client = TestClient(app)
    resp = client.post("/api/query", json={"question": ""})
    assert resp.status_code == 422


def test_api_query_retrieve_exception_returns_error_no_500(monkeypatch):
    """Si multi_retrieve tira excepción, devolvemos JSON con `error`
    en lugar de un 500 — el listener tiene fallback subprocess que se
    activa al ver `error` en el response."""
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test boom")),
    )
    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "x"})
    # 200 con error en JSON, NO 500 — el listener espera JSON, no HTTP
    # error. El fallback subprocess se activa con `data.error` no con
    # `resp.status_code != 200`.
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_api_query_dev_error_signal_sanitized(monkeypatch):
    """Cuando la excepción tiene un dev-error signal (ThreadPoolExecutor,
    GIL-serialised, traceback, etc.) — `_sanitize_error_for_user` la
    reemplaza por un mensaje user-friendly. Mensajes "simples" pasan
    directo (los considera user-safe)."""
    monkeypatch.setattr(
        server_mod, "resolve_vault_paths",
        lambda *a, **kw: [("home", Path("/tmp/fake-vault"))],
    )
    # Mensaje con `ThreadPoolExecutor` matchea `_DEV_ERROR_SIGNALS`.
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("bm25 fail in ThreadPoolExecutor wrapper")
        ),
    )
    client = TestClient(app)
    resp = client.post("/api/query", json={"question": "x"})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    # El mensaje ThreadPoolExecutor (interno) NO debe filtrarse.
    assert "ThreadPoolExecutor" not in data["error"]
    assert "bm25" not in data["error"]


# ── Alias /query (compat con clientes apuntando al wire del rag serve) ──────


def test_legacy_query_alias_same_behavior(monkeypatch):
    """`/query` debe ser un alias de `/api/query` (mismo handler).

    El WhatsApp listener apunta a `${RAG_SERVE_URL}/query` con el
    rag serve legacy en :7832. Cambiar `RAG_SERVE_URL` a
    `http://127.0.0.1:8765` solo cambia el puerto — el path `/query`
    sigue resolviendo gracias al alias.
    """
    _patch_retrieve(monkeypatch)
    client = TestClient(app)
    resp = client.post("/query", json={
        "question": "series",
        "retrieve_only": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert len(data["sources"]) == 2
