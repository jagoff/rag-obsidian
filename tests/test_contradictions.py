import json

from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False, **_):
        # Descending constant scores — ranking follows insertion order of `pairs`.
        return [1.0 - i * 0.01 for i in range(len(pairs))]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def fake_reranker(monkeypatch):
    monkeypatch.setattr(rag, "get_reranker", lambda: FakeReranker())


@pytest.fixture
def scripted_ollama(monkeypatch):
    """Factory: call the returned setter with a string to stub the next ollama.chat."""
    state = {"content": '{"contradictions": []}', "calls": 0}

    def _chat(model, messages, options=None, keep_alive=None):
        state["calls"] += 1
        return _FakeResponse(state["content"])

    monkeypatch.setattr(rag.ollama, "chat", _chat)

    def _set(content):
        state["content"] = content
    _set.state = state
    return _set


@pytest.fixture
def col(tmp_path, fake_embed, fake_reranker):
    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    c = client.get_or_create_collection(
        name="contradict_test", metadata={"hnsw:space": "cosine"}
    )
    return c


def _add(col, id_, text, file, note):
    col.add(
        ids=[id_],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=[text],
        metadatas=[{"file": file, "note": note, "parent": text}],
    )


def test_short_answer_returns_empty_without_llm_call(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    out = rag.find_contradictions(col, "color de X", "X es azul.", set())
    assert out == []
    assert scripted_ollama.state["calls"] == 0


def test_empty_vault_returns_empty(tmp_path, fake_embed, fake_reranker, scripted_ollama):
    client = _TestVecClient(path=str(tmp_path / "empty"))
    empty = client.get_or_create_collection(
        name="empty_test", metadata={"hnsw:space": "cosine"}
    )
    out = rag.find_contradictions(empty, "q", "a" * 60, set())
    assert out == []
    assert scripted_ollama.state["calls"] == 0


@pytest.mark.requires_ollama
def test_contradiction_detected_and_parsed(col, scripted_ollama):
    _add(col, "c1", "X es rojo según mis notas antiguas del archivo", "rojo.md", "rojo")
    _add(col, "c2", "X es un polígono con cuatro lados regulares", "cuad.md", "cuad")
    scripted_ollama('{"contradictions": [{"index": 1, "why": "color opuesto"}]}')

    out = rag.find_contradictions(
        col, "de qué color es X",
        "X es completamente azul en todas mis notas recientes.", set(),
    )

    assert len(out) == 1
    assert out[0]["path"] == "rojo.md"
    assert out[0]["note"] == "rojo"
    assert out[0]["why"] == "color opuesto"
    assert "rojo" in out[0]["snippet"]


def test_complementary_chunks_return_empty(col, scripted_ollama):
    _add(col, "c1", "X es un polígono con cuatro lados iguales", "poli.md", "poli")
    _add(col, "c2", "X es considerado un cuadrilátero en geometría", "cuad.md", "cuad")
    scripted_ollama('{"contradictions": []}')

    out = rag.find_contradictions(
        col, "qué es X",
        "X es un cuadrado de cuatro lados iguales y cuatro ángulos rectos.", set(),
    )
    assert out == []


_LONG_ANSWER = "X es definitivamente azul en todas mis notas nuevas del último mes."


@pytest.mark.requires_ollama
def test_json_embedded_in_prose_is_parsed(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama(
        'Claro, aquí está mi análisis: {"contradictions": [{"index": 1, "why": "tensión clara"}]} — fin.'
    )

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert len(out) == 1
    assert out[0]["why"] == "tensión clara"


def test_garbage_response_returns_empty(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama("completamente no-json sin ninguna estructura parseable aquí")

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert out == []


def test_malformed_json_returns_empty(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama('{"contradictions": [{"index": 1 "why": "falta coma"}]}')

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert out == []


def test_non_list_contradictions_returns_empty(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama('{"contradictions": "this is a string not a list"}')

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert out == []


@pytest.mark.requires_ollama
def test_exclude_paths_drops_cited_notes(col, scripted_ollama):
    _add(col, "c1", "X es rojo según las notas del año pasado", "rojo.md", "rojo")
    _add(col, "c2", "X era verde oscuro en mis borradores iniciales", "verde.md", "verde")
    # Helper points at the only surviving candidate after exclude.
    scripted_ollama('{"contradictions": [{"index": 1, "why": "diferente"}]}')

    out = rag.find_contradictions(
        col, "color X", "X es azul ahora en mis notas actualizadas.",
        exclude_paths={"rojo.md"},
    )
    assert len(out) == 1
    assert out[0]["path"] == "verde.md"


@pytest.mark.requires_ollama
def test_out_of_range_index_is_dropped(col, scripted_ollama):
    _add(col, "c1", "X es rojo según mis notas", "rojo.md", "rojo")
    scripted_ollama(
        '{"contradictions": [{"index": 99, "why": "hallucinated"}, '
        '{"index": 1, "why": "real mismatch"}]}'
    )

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert len(out) == 1
    assert out[0]["why"] == "real mismatch"


def test_dedup_by_path_across_chunks_of_same_note(col, scripted_ollama):
    _add(col, "c1", "X es rojo, parte A del archivo", "rojo.md", "rojo")
    _add(col, "c2", "X sigue siendo rojo, parte B del archivo", "rojo.md", "rojo")
    _add(col, "c3", "X es violeta según otra nota aparte", "violeta.md", "violeta")
    scripted_ollama(
        '{"contradictions": [{"index": 1, "why": "a"}, {"index": 2, "why": "b"}]}'
    )

    out = rag.find_contradictions(col, "color X", "X es azul en todas mis notas nuevas.", set())
    paths = [o["path"] for o in out]
    assert len(paths) == len(set(paths))  # dedup enforced before handing to LLM


def test_ollama_exception_returns_empty(col, fake_reranker, fake_embed, monkeypatch):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")

    def _raise(**kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag.ollama, "chat", _raise)

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert out == []


@pytest.mark.requires_ollama
def test_why_is_truncated_to_200_chars(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    long_why = "tensión muy larga " * 30  # >> 200 chars
    scripted_ollama(json.dumps(
        {"contradictions": [{"index": 1, "why": long_why}]}
    ))

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert len(out) == 1
    assert len(out[0]["why"]) <= 200


def test_embed_failure_returns_empty(col, fake_reranker, scripted_ollama, monkeypatch):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")

    def _boom(texts):
        raise RuntimeError("embed model offline")

    monkeypatch.setattr(rag, "embed", _boom)

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert out == []
    assert scripted_ollama.state["calls"] == 0


@pytest.mark.requires_ollama
def test_non_dict_item_in_list_is_skipped(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama(
        '{"contradictions": ["not a dict", {"index": 1, "why": "ok"}, 42]}'
    )

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert len(out) == 1
    assert out[0]["why"] == "ok"


@pytest.mark.requires_ollama
def test_missing_why_is_empty_string(col, scripted_ollama):
    _add(col, "c1", "X es rojo", "rojo.md", "rojo")
    scripted_ollama('{"contradictions": [{"index": 1}]}')

    out = rag.find_contradictions(col, "q", _LONG_ANSWER, set())
    assert len(out) == 1
    assert out[0]["why"] == ""
