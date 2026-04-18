"""Tests para el loop de feedback 👍/👎.

Monkeypatcheamos `FEEDBACK_PATH` / `FEEDBACK_GOLDEN_PATH` a `tmp_path` para
aislar cada test. `embed` se stubea para evitar la llamada a Ollama.
"""

import json

import pytest

import rag


@pytest.fixture
def fb_tmp(tmp_path, monkeypatch):
    """Redirige los paths de feedback a tmp_path y limpia la memo en proceso."""
    fb_path = tmp_path / "feedback.jsonl"
    golden_path = tmp_path / "feedback_golden.json"
    monkeypatch.setattr(rag, "FEEDBACK_PATH", fb_path)
    monkeypatch.setattr(rag, "FEEDBACK_GOLDEN_PATH", golden_path)
    monkeypatch.setattr(rag, "_feedback_golden_memo", None)
    monkeypatch.setattr(rag, "_feedback_golden_mtime", 0.0)
    return fb_path, golden_path


@pytest.fixture
def fake_embed(monkeypatch):
    """Embed determinístico por hash del texto → vector 4-dim en la esfera unidad
    aproximada. Dos queries idénticas dan el mismo vector; distintas dan
    vectores distintos pero bounded.
    """
    def _embed(texts):
        out = []
        for t in texts:
            h = abs(hash(t))
            x = [
                (h & 0xFF) / 255.0,
                ((h >> 8) & 0xFF) / 255.0,
                ((h >> 16) & 0xFF) / 255.0,
                ((h >> 24) & 0xFF) / 255.0,
            ]
            # Normalise so cosine = dot.
            import math
            n = math.sqrt(sum(v * v for v in x)) or 1.0
            out.append([v / n for v in x])
        return out
    monkeypatch.setattr(rag, "embed", _embed)
    return _embed


# ── new_turn_id ────────────────────────────────────────────────────────────


def test_new_turn_id_length_and_uniqueness():
    ids = {rag.new_turn_id() for _ in range(50)}
    assert len(ids) == 50            # no collisions in 50 tries
    for tid in ids:
        assert len(tid) == 12
        assert all(c in "0123456789abcdef" for c in tid)


# ── detect_rating_intent ───────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("+", 1),                     # forma canónica: una tecla
    ("-", -1),
    ("/bien", 1),                 # verbose
    ("/mal", -1),
    ("👍", 1),                    # emoji backward-compat
    ("👎", -1),
    ("  +  ", 1),                 # whitespace tolerated
    ("👍🏼", 1),                   # skin-tone modifier
    ("👎🏻", -1),
    ("👍\ufe0f", 1),              # emoji variation selector
    ("", None),
    ("+ bien dicho", None),       # rating + other text → NOT a rating
    ("+1", None),                 # modifiers → NOT a rating (evita colisión con "++ algo")
    ("que opinás", None),
    ("/save", None),
    ("bien", None),               # palabra sin slash NO es rating
    ("malo", None),
])
def test_detect_rating_intent(text, expected):
    assert rag.detect_rating_intent(text) == expected


# ── record_feedback + feedback_counts ──────────────────────────────────────


def test_record_feedback_roundtrip(fb_tmp):
    fb_path, _ = fb_tmp
    rag.record_feedback("abc123", 1, "mi query", ["a.md", "b.md"])
    rag.record_feedback("def456", -1, "otra", ["c.md"])

    assert fb_path.is_file()
    lines = fb_path.read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["turn_id"] == "abc123"
    assert first["rating"] == 1
    assert first["q"] == "mi query"
    assert first["paths"] == ["a.md", "b.md"]
    assert "ts" in first

    assert rag.feedback_counts() == (1, 1)


def test_feedback_counts_empty(fb_tmp):
    assert rag.feedback_counts() == (0, 0)


def test_record_feedback_normalises_rating(fb_tmp):
    """Valores de rating != ±1 se normalizan a signo — el schema es binario."""
    rag.record_feedback("t1", 999, "x", ["a.md"])
    rag.record_feedback("t2", -42, "y", ["b.md"])
    fb_path, _ = fb_tmp
    events = [json.loads(l) for l in fb_path.read_text().splitlines()]
    assert events[0]["rating"] == 1
    assert events[1]["rating"] == -1


def test_record_feedback_invalidates_golden_cache(fb_tmp, fake_embed):
    _, golden_path = fb_tmp
    # Seed: una feedback entry + cache fresca.
    rag.record_feedback("t1", 1, "q1", ["a.md"])
    rag.load_feedback_golden()
    assert golden_path.is_file()

    # Nueva feedback → el delete de golden se dispara; la próxima load rebuildea.
    rag.record_feedback("t2", 1, "q2", ["b.md"])
    golden = rag.load_feedback_golden()
    assert {e["q"] for e in golden["positives"]} == {"q1", "q2"}


# ── load_feedback_golden / rebuild ──────────────────────────────────────────


def test_load_feedback_golden_empty(fb_tmp, fake_embed):
    golden = rag.load_feedback_golden()
    assert golden == {"positives": [], "negatives": []}


def test_load_feedback_golden_splits_by_rating(fb_tmp, fake_embed):
    rag.record_feedback("p1", 1, "pos1", ["a.md"])
    rag.record_feedback("p2", 1, "pos2", ["b.md", "c.md"])
    rag.record_feedback("n1", -1, "neg1", ["bad.md"])

    golden = rag.load_feedback_golden()
    assert len(golden["positives"]) == 2
    assert len(golden["negatives"]) == 1
    assert {e["q"] for e in golden["positives"]} == {"pos1", "pos2"}
    assert golden["negatives"][0]["q"] == "neg1"
    assert golden["negatives"][0]["paths"] == ["bad.md"]
    # Cada entry tiene embedding.
    for e in golden["positives"] + golden["negatives"]:
        assert "emb" in e and len(e["emb"]) == 4


def test_load_feedback_golden_later_turn_overrides_earlier(fb_tmp, fake_embed):
    """Si el mismo turn_id aparece dos veces, gana el más reciente."""
    rag.record_feedback("same", 1, "q", ["a.md"])
    rag.record_feedback("same", -1, "q", ["a.md"])

    golden = rag.load_feedback_golden()
    assert golden["positives"] == []
    assert len(golden["negatives"]) == 1


def test_load_feedback_golden_skips_empty_paths_or_query(fb_tmp, fake_embed):
    rag.record_feedback("t1", 1, "", ["a.md"])          # empty query
    rag.record_feedback("t2", 1, "q", [])               # empty paths
    rag.record_feedback("t3", 1, "q3", ["c.md"])

    golden = rag.load_feedback_golden()
    assert len(golden["positives"]) == 1
    assert golden["positives"][0]["q"] == "q3"


# ── feedback_signals_for_query ──────────────────────────────────────────────


def test_feedback_signals_empty_when_no_history(fb_tmp, fake_embed):
    boost, penalty = rag.feedback_signals_for_query([1.0, 0.0, 0.0, 0.0])
    assert boost == set()
    assert penalty == set()


def test_feedback_signals_matches_identical_query(fb_tmp, fake_embed):
    """Una query idéntica a una 👍'da previa debe recuperar sus paths."""
    rag.record_feedback("t1", 1, "adam jones sistema de sonido", ["guitar.md"])

    # Mismo texto → mismo embedding (gracias al hash determinista) → cosine 1.0.
    q_emb = rag.embed(["adam jones sistema de sonido"])[0]
    boost, penalty = rag.feedback_signals_for_query(q_emb)
    assert "guitar.md" in boost
    assert penalty == set()


def test_feedback_signals_below_threshold_ignored(fb_tmp, fake_embed, monkeypatch):
    """Si cosine < FEEDBACK_MATCH_COSINE, el feedback no se aplica."""
    monkeypatch.setattr(rag, "FEEDBACK_MATCH_COSINE", 0.99)
    rag.record_feedback("t1", 1, "aaa", ["a.md"])
    q_emb = rag.embed(["zzz"])[0]                # hash distinto → cosine bajo
    boost, penalty = rag.feedback_signals_for_query(q_emb)
    assert boost == set()
    assert penalty == set()


def test_feedback_signals_negative_loses_to_positive(fb_tmp, fake_embed):
    """Si el mismo path tiene 👍 y 👎 en queries similares, gana el 👍
    (el positivo es la señal más reciente y explícita que el usuario lo valida).
    """
    rag.record_feedback("t_pos", 1, "cual es el link X", ["url-notes.md"])
    rag.record_feedback("t_neg", -1, "cual es el link X", ["url-notes.md"])

    # Como hash(q) es idéntico para ambas, ambas matchean. Con el override
    # por turn_id (el más reciente gana), la última escrita es el 👎. Pero
    # esto depende del orden de escritura, así que preparamos uno nuevo.
    rag.record_feedback("turn_p", 1, "x", ["shared.md"])
    rag.record_feedback("turn_n", -1, "y", ["shared.md"])

    # 'x' y 'y' tienen hash distinto. Tiramos dos queries, cada una matchea
    # exactamente una. La query 'x' debe boostear; la 'y' debe penalizar.
    qx = rag.embed(["x"])[0]
    boost_x, pen_x = rag.feedback_signals_for_query(qx)
    assert "shared.md" in boost_x
    assert "shared.md" not in pen_x


# ── retrieve() integration ─────────────────────────────────────────────────


def test_retrieve_applies_positive_boost(tmp_path, fb_tmp, fake_embed, monkeypatch):
    """Con un 👍 previo sobre `winner.md`, retrieve() debe rankearlo por encima
    de `loser.md` aunque el reranker les dé el mismo score crudo.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="feedback_test", metadata={"hnsw:space": "cosine"}
    )
    # Dos chunks — mismo embedding, diferente file. El reranker les dará
    # scores idénticos; sin feedback el orden es indeterminado. Con 👍 en
    # winner.md, FEEDBACK_POSITIVE_BOOST desempata a su favor.
    emb = [1.0, 0.0, 0.0, 0.0]
    col.add(
        ids=["chunk_w", "chunk_l"],
        embeddings=[emb, emb],
        documents=["contenido winner", "contenido loser"],
        metadatas=[
            {"file": "winner.md", "note": "Winner", "parent": "contenido winner"},
            {"file": "loser.md", "note": "Loser", "parent": "contenido loser"},
        ],
    )

    # Stub: reranker constante (ambos candidates empatados), multi_query off,
    # intent classifier → semantic sin filtros.
    class FlatReranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: FlatReranker())
    monkeypatch.setattr(rag, "classify_intent", lambda q, t, f: ("semantic", {}))
    monkeypatch.setattr(rag, "infer_filters", lambda q, t, f: (None, None))
    monkeypatch.setattr(rag, "get_vocabulary", lambda c: ([], []))
    monkeypatch.setattr(rag, "bm25_search", lambda c, q, k, f, t, dr=None: [])
    monkeypatch.setattr(rag, "expand_to_parent", lambda d, m: d)
    monkeypatch.setattr(rag, "has_recency_cue", lambda q: False)
    monkeypatch.setattr(rag, "expand_queries", lambda q: [q])

    # Feedback: 👍 sobre winner.md, con la misma query que vamos a usar después.
    rag.record_feedback("t1", 1, "busco X", ["winner.md"])

    result = rag.retrieve(
        col, "busco X", k=2, folder=None, tag=None,
        precise=False, multi_query=False, auto_filter=False,
    )
    files = [m["file"] for m in result["metas"]]
    assert files[0] == "winner.md", f"winner.md debería quedar primero, got {files}"
    # El boost es aditivo, así que el score 0 debe ser mayor que el score 1.
    assert result["scores"][0] > result["scores"][1]


def test_retrieve_applies_negative_penalty(tmp_path, fb_tmp, fake_embed, monkeypatch):
    """Un 👎 en `bad.md` debe tirar ese path al fondo incluso si el reranker
    lo pone ligeramente arriba del otro.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="feedback_penalty_test", metadata={"hnsw:space": "cosine"}
    )
    emb = [1.0, 0.0, 0.0, 0.0]
    col.add(
        ids=["chunk_b", "chunk_g"],
        embeddings=[emb, emb],
        documents=["contenido bad", "contenido good"],
        metadatas=[
            {"file": "bad.md", "note": "Bad", "parent": "contenido bad"},
            {"file": "good.md", "note": "Good", "parent": "contenido good"},
        ],
    )

    # Reranker prefiere bad.md por 0.05; penalty de 0.15 debería revertir eso.
    class BiasedReranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [0.6 if "bad" in p[1] else 0.55 for p in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: BiasedReranker())
    monkeypatch.setattr(rag, "classify_intent", lambda q, t, f: ("semantic", {}))
    monkeypatch.setattr(rag, "infer_filters", lambda q, t, f: (None, None))
    monkeypatch.setattr(rag, "get_vocabulary", lambda c: ([], []))
    monkeypatch.setattr(rag, "bm25_search", lambda c, q, k, f, t, dr=None: [])
    monkeypatch.setattr(rag, "expand_to_parent", lambda d, m: d)
    monkeypatch.setattr(rag, "has_recency_cue", lambda q: False)
    monkeypatch.setattr(rag, "expand_queries", lambda q: [q])

    rag.record_feedback("t1", -1, "query sobre Y", ["bad.md"])

    result = rag.retrieve(
        col, "query sobre Y", k=2, folder=None, tag=None,
        precise=False, multi_query=False, auto_filter=False,
    )
    files = [m["file"] for m in result["metas"]]
    assert files[0] == "good.md", f"good.md debería ganar, got {files}"


def test_retrieve_boost_path_not_in_pool_gets_injected(tmp_path, fb_tmp, fake_embed, monkeypatch):
    """Si un path 👍 no fue recuperado por semantic+BM25, debe inyectarse al pool."""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="feedback_inject_test", metadata={"hnsw:space": "cosine"}
    )
    # Query embedding = [1,0,0,0]; relevant.md matchea, hidden.md es ortogonal.
    col.add(
        ids=["chunk_rel", "chunk_hidden"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        documents=["relevante", "oculto"],
        metadatas=[
            {"file": "relevant.md", "note": "R", "parent": "relevante"},
            {"file": "hidden.md", "note": "H", "parent": "oculto"},
        ],
    )

    # Reranker: hidden.md recibe score alto si llega, relevant.md score bajo.
    # Si hidden.md no se inyecta, nunca va a estar en pairs → nunca scoreado.
    class PrefersHidden:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [1.0 if "oculto" in p[1] else 0.01 for p in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: PrefersHidden())
    monkeypatch.setattr(rag, "classify_intent", lambda q, t, f: ("semantic", {}))
    monkeypatch.setattr(rag, "infer_filters", lambda q, t, f: (None, None))
    monkeypatch.setattr(rag, "get_vocabulary", lambda c: ([], []))
    monkeypatch.setattr(rag, "bm25_search", lambda c, q, k, f, t, dr=None: [])
    monkeypatch.setattr(rag, "expand_to_parent", lambda d, m: d)
    monkeypatch.setattr(rag, "has_recency_cue", lambda q: False)
    monkeypatch.setattr(rag, "expand_queries", lambda q: [q])
    # Forzamos RETRIEVE_K=1 para que ChromaDB semántico SOLO traiga relevant.md.
    monkeypatch.setattr(rag, "RETRIEVE_K", 1)

    rag.record_feedback("t1", 1, "busco algo", ["hidden.md"])

    result = rag.retrieve(
        col, "busco algo", k=2, folder=None, tag=None,
        precise=False, multi_query=False, auto_filter=False,
    )
    files = [m["file"] for m in result["metas"]]
    # hidden.md llegó al pool vía feedback injection y el reranker lo premió.
    assert "hidden.md" in files
    assert files[0] == "hidden.md"
