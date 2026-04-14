"""Folder filter widens to include 00-Inbox.

Inbox es staging, no tópico — una captura sin filar puede ser de cualquier
tema, así que no debería ser excluida cuando hay filtro de folder.

Bug que motivó esto (2026-04-14): "como activo claude peers?" →
infer_filters auto-aplicó `folder=03-Resources/Claude`, y la nota
`00-Inbox/Claude-peers.md` quedó invisible. RAG respondió "No tengo esa
información" sobre algo que sí estaba en el vault.
"""
import pytest
import rag


def test_folder_matches_literal():
    assert rag._folder_matches("02-Areas/foo.md", "02-Areas") is True
    assert rag._folder_matches("02-Areas/sub/bar.md", "02-Areas/sub") is True


def test_folder_matches_rejects_unrelated():
    assert rag._folder_matches("03-Resources/x.md", "02-Areas") is False


def test_folder_matches_widens_to_inbox():
    # Filtro pide "03-Resources/Claude" pero la nota vive en Inbox →
    # matchea igual porque inbox es transient staging.
    assert rag._folder_matches(
        "00-Inbox/Claude-peers.md", "03-Resources/Claude"
    ) is True


def test_folder_matches_inbox_is_always_in():
    # Cualquier folder como filtro + file en Inbox → match.
    for folder in ["02-Areas", "03-Resources/Tech", "05-Reviews", "04-Archive/foo"]:
        assert rag._folder_matches("00-Inbox/capture.md", folder) is True


def test_folder_matches_does_not_widen_to_other_folders():
    # No queremos que filtrar por Areas también traiga Archive o Resources.
    # Solo Inbox es meta-folder.
    assert rag._folder_matches(
        "04-Archive/old.md", "02-Areas"
    ) is False
    assert rag._folder_matches(
        "03-Resources/doc.md", "02-Areas"
    ) is False


def test_build_where_folder_includes_inbox_or_clause():
    where = rag.build_where(folder="03-Resources/Claude", tag=None)
    # Un único filtro → no $and, es el $or directo.
    assert "$or" in where
    or_paths = [c["file"]["$contains"] for c in where["$or"]]
    assert "03-Resources/Claude" in or_paths
    assert "00-Inbox/" in or_paths


def test_build_where_folder_plus_tag_combines_correctly():
    where = rag.build_where(folder="02-Areas", tag="foco")
    assert "$and" in where
    # Primer clause: folder con $or inbox. Segundo: tag directo.
    conds = where["$and"]
    assert any("$or" in c for c in conds)
    assert any(c.get("tags", {}).get("$contains") == "foco" for c in conds)


def test_build_where_tag_only_does_not_widen():
    where = rag.build_where(folder=None, tag="coaching")
    assert where == {"tags": {"$contains": "coaching"}}


def test_build_where_no_filters_returns_none():
    assert rag.build_where(folder=None, tag=None) is None


def test_filter_files_folder_widens_to_inbox():
    metas = [
        {"file": "03-Resources/Claude/note1.md", "tags": ""},
        {"file": "00-Inbox/Claude-peers.md", "tags": ""},
        {"file": "02-Areas/unrelated.md", "tags": ""},
    ]
    filtered = rag._filter_files(
        metas, tag=None, folder="03-Resources/Claude"
    )
    files = sorted(m["file"] for m in filtered)
    # La nota de inbox debe aparecer aunque el filtro pida Resources/Claude.
    assert "00-Inbox/Claude-peers.md" in files
    assert "03-Resources/Claude/note1.md" in files
    # La nota en Areas sí queda fuera.
    assert "02-Areas/unrelated.md" not in files
