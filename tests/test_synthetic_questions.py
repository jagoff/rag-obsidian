"""Synthetic question expansion for chunk embeddings.

Notas se escriben como afirmaciones; queries llegan como preguntas. Gap
semántico lo cierra el reranker, pero con costo. Generar preguntas sintéticas
al indexar y bakearlas en el prefix del chunk biases el embedding hacia match
directo con queries tipo-pregunta.

Tests cubren:
  - generator parsea JSON estricto del helper
  - cache hash-keyed (no regenera)
  - skip notas cortas (<300 chars)
  - env kill-switch bypasea generación
  - malformed JSON → [] sin crashear
  - semantic_chunks prepende línea "Preguntas:" cuando hay preguntas
  - integración con context_summary (ambos pueden coexistir en el prefix)
  - dedup de preguntas equivalentes
  - cap total y cap por-pregunta
"""
from __future__ import annotations

import json
from collections import OrderedDict

import pytest

import rag


# ── FIXTURES ─────────────────────────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeClient:
    """Stand-in para _SUMMARY_CLIENT — `.chat()` devuelve lo que se le ponga."""

    def __init__(self):
        self.next_response = ""
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.next_response)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(rag, "_SUMMARY_CLIENT", client)
    # Reset cache + env so tests are hermetic.
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", False)
    monkeypatch.delenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", raising=False)
    return client


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Redirect the on-disk cache to a tmp file so tests don't pollute real state."""
    tmp_cache = tmp_path / "synthetic_questions.json"
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", False)
    return tmp_cache


# ── GENERATOR ────────────────────────────────────────────────────────────────


def test_generator_parses_json_object(fake_client):
    fake_client.next_response = json.dumps({
        "preguntas": [
            "¿qué reranker uso?",
            "¿qué modelo embebe?",
            "¿cómo corre ollama?",
        ]
    })
    qs = rag._generate_synthetic_questions(
        "body " * 100, "setup", "03-Resources"
    )
    assert qs == [
        "¿qué reranker uso?",
        "¿qué modelo embebe?",
        "¿cómo corre ollama?",
    ]


def test_generator_accepts_english_key(fake_client):
    # Tolerante a "questions" también, no sólo "preguntas".
    fake_client.next_response = json.dumps({
        "questions": ["what model?", "which reranker?"]
    })
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs == ["what model?", "which reranker?"]


def test_generator_returns_none_on_malformed_json(fake_client):
    # Transient failure: return None so get_synthetic_questions won't cache.
    fake_client.next_response = "not json at all { broken"
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs is None


def test_generator_returns_none_on_wrong_shape(fake_client):
    # data es un list en vez de dict → no se puede extraer "preguntas"
    # Treated as transient so the next index pass retries.
    fake_client.next_response = json.dumps(["¿qué?", "¿cómo?"])
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs is None


def test_generator_returns_none_on_ollama_failure(fake_client):
    def boom(**kwargs):
        raise RuntimeError("ollama down")
    fake_client.chat = boom  # type: ignore[assignment]
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs is None


def test_get_synthetic_questions_does_not_cache_transient_failure(fake_client, tmp_path, monkeypatch):
    """Regression: previously a transient LLM failure cached [] by hash,
    preventing retries until the note's content changed. New contract:
    None from the generator is NOT persisted, so the next pass retries.
    """
    cache_path = tmp_path / "synth.json"
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", False)

    # First call: LLM fails → generator returns None → caller sees [], NOT cached.
    original_chat = type(fake_client).chat
    should_fail = {"flag": True}
    def gated_chat(self, **kwargs):
        if should_fail["flag"]:
            raise RuntimeError("ollama down")
        return original_chat(self, **kwargs)
    monkeypatch.setattr(type(fake_client), "chat", gated_chat)

    first = rag.get_synthetic_questions("x" * 500, "hash1", "setup", "03-Resources")
    assert first == []
    assert "hash1" not in rag._load_synthetic_q_cache()

    # Second call on same hash: LLM recovers → retry succeeds → cached.
    should_fail["flag"] = False
    fake_client.next_response = json.dumps({"preguntas": ["q1?", "q2?"]})
    second = rag.get_synthetic_questions("x" * 500, "hash1", "setup", "03-Resources")
    assert second == ["q1?", "q2?"]
    assert rag._load_synthetic_q_cache()["hash1"] == ["q1?", "q2?"]


def test_generator_respects_kill_switch(fake_client, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", "1")
    fake_client.next_response = json.dumps({"preguntas": ["¿x?"]})
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs == []
    # También verificamos que no se llamó el helper (short-circuit).
    assert fake_client.calls == []


def test_generator_caps_count_and_length(fake_client):
    # Modelo devuelve 6 preguntas muy largas — capear a _SYNTHETIC_Q_CAP=4
    # y cada una a _SYNTHETIC_Q_MAX_CHARS=120.
    long_q = "¿" + ("a" * 200) + "?"
    fake_client.next_response = json.dumps({
        "preguntas": [long_q] * 6
    })
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    # Cap = 4 pero dedup por contenido → queda sólo 1 pregunta única (todas iguales).
    assert len(qs) == 1
    assert len(qs[0]) <= rag._SYNTHETIC_Q_MAX_CHARS


def test_generator_dedupes_equivalent_questions(fake_client):
    fake_client.next_response = json.dumps({
        "preguntas": [
            "¿Qué reranker uso?",
            "qué reranker uso",     # mismo contenido, sin signos
            "¿qué reranker uso?",   # mismo contenido, case diferente
            "¿qué modelo embebe?",
        ]
    })
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    # Deberían quedar 2 únicas.
    assert len(qs) == 2


def test_generator_strips_bullet_prefixes(fake_client):
    # Algunos modelos emiten "- pregunta" o "* pregunta" dentro del string.
    fake_client.next_response = json.dumps({
        "preguntas": ["- ¿qué modelo uso?", "• ¿cómo está configurado?"]
    })
    qs = rag._generate_synthetic_questions("x" * 500, "setup", "03-Resources")
    assert qs[0].startswith("¿")
    assert qs[1].startswith("¿")


# ── CACHE ────────────────────────────────────────────────────────────────────


def test_get_caches_by_hash(fake_client, isolated_cache):
    fake_client.next_response = json.dumps({"preguntas": ["¿x?", "¿y?"]})
    text = "body " * 100
    qs1 = rag.get_synthetic_questions(text, "hash-abc", "nota", "03-Resources")
    qs2 = rag.get_synthetic_questions(text, "hash-abc", "nota", "03-Resources")
    assert qs1 == qs2
    assert len(fake_client.calls) == 1  # segundo llamado pegó cache


def test_get_skips_short_notes(fake_client, isolated_cache):
    # Notas <300 chars no generan preguntas (mismo umbral que context summary).
    fake_client.next_response = json.dumps({"preguntas": ["¿x?"]})
    qs = rag.get_synthetic_questions("corta", "hash-short", "nota", "03-Resources")
    assert qs == []
    assert fake_client.calls == []  # ni se llamó al modelo


def test_cache_persists_to_disk(fake_client, isolated_cache):
    fake_client.next_response = json.dumps({"preguntas": ["¿x?"]})
    rag.get_synthetic_questions("body " * 100, "hash-persist", "nota", "03-Resources")
    rag._save_synthetic_q_cache()
    assert isolated_cache.is_file()
    data = json.loads(isolated_cache.read_text())
    assert data["hash-persist"] == ["¿x?"]


def test_cache_loads_from_disk(monkeypatch, tmp_path):
    # Pre-populate tmp cache → reload debería leerlo.
    tmp_cache = tmp_path / "synthetic_questions.json"
    tmp_cache.write_text(json.dumps({"hash-preloaded": ["¿q preloaded?"]}))
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    cache = rag._load_synthetic_q_cache()
    assert cache["hash-preloaded"] == ["¿q preloaded?"]


def test_cache_handles_corrupted_file(monkeypatch, tmp_path):
    tmp_cache = tmp_path / "synthetic_questions.json"
    tmp_cache.write_text("{{not json")
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    cache = rag._load_synthetic_q_cache()
    assert cache == {}


def test_cache_corrupted_file_is_quarantined(monkeypatch, tmp_path):
    """A corrupted cache file must be renamed to `.corrupt-<ts>` so the
    next read doesn't loop the same JSONDecodeError. Without this, every
    process startup would log a fresh `synthetic_q_cache_load` error."""
    tmp_cache = tmp_path / "synthetic_questions.json"
    tmp_cache.write_text("{{not json")
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)

    rag._load_synthetic_q_cache()

    # Original file should be gone (renamed to backup).
    assert not tmp_cache.is_file()
    # Backup with `.corrupt-<digits>` suffix must exist.
    backups = list(tmp_path.glob("synthetic_questions.json.corrupt-*"))
    assert len(backups) == 1
    # Content of backup must match original corrupt payload.
    assert backups[0].read_text() == "{{not json"


def test_cache_save_uses_atomic_tmp_rename(monkeypatch, tmp_path):
    """Writes must go through tmp+rename so concurrent writers can never
    truncate each other into a half-written JSON. Side effect we observe:
    after save, no `.tmp` files leak in the cache dir."""
    tmp_cache = tmp_path / "synthetic_questions.json"
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", OrderedDict({"h": ["¿q?"]}))
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", True)

    rag._save_synthetic_q_cache()

    # Final file must be valid JSON.
    assert tmp_cache.is_file()
    assert json.loads(tmp_cache.read_text()) == {"h": ["¿q?"]}
    # No .tmp residue should have leaked.
    leftovers = list(tmp_path.glob("*.tmp.*"))
    assert leftovers == []


# ── INTEGRACIÓN CON semantic_chunks ──────────────────────────────────────────


def test_semantic_chunks_prepends_preguntas_line():
    text = "Este es el cuerpo de la nota.\n\nCon dos párrafos distintos."
    chunks = rag.semantic_chunks(
        text, "mi-nota", "03-Resources", [], {},
        synthetic_questions=["¿qué hace esta nota?", "¿de qué trata?"],
    )
    assert chunks
    embed_text = chunks[0][0]
    assert "Preguntas: ¿qué hace esta nota? ¿de qué trata?" in embed_text


def test_semantic_chunks_no_preguntas_when_list_empty():
    text = "Cuerpo simple de la nota."
    chunks = rag.semantic_chunks(
        text, "mi-nota", "03-Resources", [], {},
        synthetic_questions=[],
    )
    assert chunks
    assert "Preguntas:" not in chunks[0][0]


def test_semantic_chunks_default_keeps_legacy_behavior():
    # Sin pasar synthetic_questions (compat retroactiva): no debe romper.
    text = "Nota default sin preguntas sintéticas."
    chunks = rag.semantic_chunks(text, "nota", "03-Resources", [], {})
    assert chunks
    assert "Preguntas:" not in chunks[0][0]


def test_semantic_chunks_both_preguntas_and_contexto():
    text = "Cuerpo de nota con suficiente contenido para un chunk."
    chunks = rag.semantic_chunks(
        text, "nota", "03-Resources", [], {},
        context_summary="Trata sobre X e Y.",
        synthetic_questions=["¿qué es X?"],
    )
    embed_text = chunks[0][0]
    # Ambos están, orden: preguntas primero (arriba del prefix), contexto después.
    assert "Preguntas: ¿qué es X?" in embed_text
    assert "Contexto: Trata sobre X e Y." in embed_text
    pr_idx = embed_text.find("Preguntas:")
    ctx_idx = embed_text.find("Contexto:")
    assert 0 <= pr_idx < ctx_idx


def test_semantic_chunks_adds_question_mark_if_missing():
    # Si el modelo devuelve sin "?" final, se normaliza.
    chunks = rag.semantic_chunks(
        "cuerpo suficientemente largo para al menos un chunk.",
        "nota", "03-Resources", [], {},
        synthetic_questions=["qué es esto", "cómo funciona?"],
    )
    embed_text = chunks[0][0]
    assert "qué es esto?" in embed_text
    assert "cómo funciona?" in embed_text


def test_semantic_chunks_preguntas_caps_to_limit():
    qs = [f"¿pregunta {i}?" for i in range(10)]
    chunks = rag.semantic_chunks(
        "cuerpo suficientemente largo para un chunk.",
        "nota", "03-Resources", [], {},
        synthetic_questions=qs,
    )
    embed_text = chunks[0][0]
    # Sólo las primeras _SYNTHETIC_Q_CAP aparecen.
    for i in range(rag._SYNTHETIC_Q_CAP):
        assert f"¿pregunta {i}?" in embed_text
    assert f"¿pregunta {rag._SYNTHETIC_Q_CAP}?" not in embed_text


# ── PRUNE ────────────────────────────────────────────────────────────────────


def test_prune_removes_stale_hashes(monkeypatch, tmp_path):
    # Cache con 3 hashes; sólo 1 aparece en metadatas → 2 se purgan.
    tmp_cache = tmp_path / "synthetic_questions.json"
    tmp_cache.write_text(json.dumps({
        "hash-live": ["¿x?"],
        "hash-stale1": ["¿y?"],
        "hash-stale2": ["¿z?"],
    }))
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_cache)
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", False)

    class FakeCol:
        def get(self, include):
            return {"metadatas": [{"hash": "hash-live"}]}

    pruned = rag._prune_synthetic_q_cache(FakeCol())
    assert pruned == 2
    remaining = json.loads(tmp_cache.read_text())
    assert set(remaining.keys()) == {"hash-live"}
