from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


@pytest.fixture
def vault_with_dupes(tmp_path, monkeypatch):
    """Vault + collection populated with three "notes" whose chunk
    embeddings make A and B near-identical (cosine ≈ 1) and C orthogonal.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("primer cuerpo de la nota A.")
    (vault / "b.md").write_text("primer cuerpo de la nota B.")
    (vault / "c.md").write_text("contenido completamente distinto.")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="dupes_test", metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=["a.md::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=["chunk A"],
        metadatas=[{"file": "a.md", "note": "A", "folder": ""}],
    )
    col.add(
        ids=["b.md::0"],
        embeddings=[[1.0, 0.001, 0.0, 0.0]],
        documents=["chunk B"],
        metadatas=[{"file": "b.md", "note": "B", "folder": ""}],
    )
    col.add(
        ids=["c.md::0"],
        embeddings=[[0.0, 0.0, 1.0, 0.0]],
        documents=["chunk C"],
        metadatas=[{"file": "c.md", "note": "C", "folder": ""}],
    )
    return vault, col


def test_finds_high_similarity_pair(vault_with_dupes):
    _, col = vault_with_dupes
    pairs = rag.find_duplicate_notes(col, threshold=0.9)
    paths = {(p["a_path"], p["b_path"]) for p in pairs}
    assert ("a.md", "b.md") in paths or ("b.md", "a.md") in paths


def test_skips_dissimilar_pair(vault_with_dupes):
    _, col = vault_with_dupes
    pairs = rag.find_duplicate_notes(col, threshold=0.9)
    for p in pairs:
        assert "c.md" not in (p["a_path"], p["b_path"]), \
            "C is orthogonal — should not pair with anything at threshold 0.9"


def test_threshold_zero_returns_all_pairs(vault_with_dupes):
    _, col = vault_with_dupes
    pairs = rag.find_duplicate_notes(col, threshold=-1.0)
    # 3 notes → 3 pairs (a-b, a-c, b-c)
    assert len(pairs) == 3


def test_high_threshold_returns_empty(vault_with_dupes):
    _, col = vault_with_dupes
    # Above any achievable cosine — fixture's most-similar pair is ≈0.9999995.
    pairs = rag.find_duplicate_notes(col, threshold=1.0001)
    assert pairs == []


def test_pairs_sorted_descending_by_similarity(vault_with_dupes):
    _, col = vault_with_dupes
    pairs = rag.find_duplicate_notes(col, threshold=-1.0)
    sims = [p["similarity"] for p in pairs]
    assert sims == sorted(sims, reverse=True)


def test_limit_caps_pairs(vault_with_dupes):
    _, col = vault_with_dupes
    pairs = rag.find_duplicate_notes(col, threshold=-1.0, limit=2)
    assert len(pairs) == 2


def test_folder_filter_excludes_outsiders(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "in").mkdir(parents=True)
    (vault / "out").mkdir()
    (vault / "in" / "a.md").write_text("a")
    (vault / "in" / "b.md").write_text("b")
    (vault / "out" / "c.md").write_text("c")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "c"))
    col = client.get_or_create_collection(
        name="folder_test", metadata={"hnsw:space": "cosine"}
    )
    for i, (file_, note) in enumerate([
        ("in/a.md", "A"), ("in/b.md", "B"), ("out/c.md", "C"),
    ]):
        col.add(
            ids=[f"{file_}::0"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            documents=[note],
            metadatas=[{"file": file_, "note": note, "folder": file_.rsplit("/", 1)[0]}],
        )
    pairs = rag.find_duplicate_notes(col, threshold=0.5, folder="in")
    paths_seen = set()
    for p in pairs:
        paths_seen.add(p["a_path"])
        paths_seen.add(p["b_path"])
    assert all(p.startswith("in/") for p in paths_seen)


def test_centroid_collapses_multiple_chunks(tmp_path, monkeypatch):
    """A note with 3 chunks should produce a centroid that's the mean."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "multi.md").write_text("multi chunk")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "c"))
    col = client.get_or_create_collection(
        name="centroid_test", metadata={"hnsw:space": "cosine"}
    )
    for i, emb in enumerate([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]):
        col.add(
            ids=[f"multi.md::{i}"],
            embeddings=[emb],
            documents=[f"chunk {i}"],
            metadatas=[{"file": "multi.md", "note": "multi", "folder": ""}],
        )
    files, _, arr = rag._note_centroids(col)
    assert files == ["multi.md"]
    # Centroid is mean = (1/3, 1/3, 1/3, 0), normalized → magnitude 1
    import math
    assert pytest.approx(math.sqrt(sum(x * x for x in arr[0])), abs=1e-5) == 1.0


def test_find_near_duplicates_for_excludes_self(vault_with_dupes):
    _, col = vault_with_dupes
    out = rag.find_near_duplicates_for(col, "a.md", threshold=0.5)
    paths = {r["path"] for r in out}
    assert "a.md" not in paths
    assert "b.md" in paths


def test_find_near_duplicates_for_unknown_path_returns_empty(vault_with_dupes):
    _, col = vault_with_dupes
    assert rag.find_near_duplicates_for(col, "nope.md") == []


def test_empty_collection_returns_empty(tmp_path):
    client = _TestVecClient(path=str(tmp_path / "c"))
    col = client.get_or_create_collection(
        name="empty_test", metadata={"hnsw:space": "cosine"}
    )
    assert rag.find_duplicate_notes(col) == []
