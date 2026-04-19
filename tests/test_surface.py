"""Tests para `rag surface`: bridge builder proactivo.

Cubre el algoritmo puro (selección de pares, filtros de grafo/MOC/edad/tags)
sin tocar ni Ollama ni la red. La generación LLM y el logging se testean con
monkeypatches de sus helpers.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """Vault vacío + colección fresca. No monkeypatchea embed (cada test inserta
    sus propias embeddings en la colección, así controla la similitud)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "SURFACE_LOG_PATH", tmp_path / "surface.jsonl")
    client = _TestVecClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="surface_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()
    return vault, col


def _add_note(
    col, vault: Path, rel: str, title: str,
    embedding, body: str = "", outlinks: str = "",
    tags: str = "", created: str | None = None,
):
    """Inserta una nota en el vault + un chunk con embedding fijo."""
    full = vault / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body or f"# {title}\n\nbody for {title}", encoding="utf-8")
    meta = {
        "file": rel, "note": title, "folder": str(Path(rel).parent),
        "tags": tags, "outlinks": outlinks, "hash": "x",
    }
    if created:
        meta["created"] = created
    col.add(
        ids=[f"{rel}::0"],
        embeddings=[list(embedding)],
        documents=[body or f"chunk for {title}"],
        metadatas=[meta],
    )
    rag._invalidate_corpus_cache()


def _old(days: int = 60) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")


# Vectores unit-norm preparados para dar cosenos específicos:
#   A=[1,0,0,0], B=[0.8,0.6,0,0] → cos 0.80
#   C=[0.4,0,0.9165,0]          → cos(A,C)=0.40
HIGH_SIM = [0.8, 0.6, 0.0, 0.0]
LOW_SIM = [0.4, 0.0, 0.9165, 0.0]


# ── Happy path ────────────────────────────────────────────────────────────────


def test_finds_bridge_between_distant_similar_notes(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert len(pairs) == 1
    assert pairs[0]["similarity"] >= 0.79
    assert {pairs[0]["a_path"], pairs[0]["b_path"]} == {"02-Areas/a.md", "02-Areas/b.md"}


def test_skips_pairs_below_sim_threshold(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/c.md", "C", LOW_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


# ── Grafo ─────────────────────────────────────────────────────────────────────


def test_skips_pairs_that_are_direct_neighbors(tmp_vault):
    vault, col = tmp_vault
    # A linkea a B — graph_distance = 1, min_hops=3 ⇒ filtro las descarta.
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              outlinks="B", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_skips_pairs_that_are_two_hops_away(tmp_vault):
    vault, col = tmp_vault
    # A → M → B, así A y B están a 2 hops. min_hops=3 ⇒ filtrado.
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              outlinks="M", created=_old())
    _add_note(col, vault, "02-Areas/m.md", "M", [0, 0, 0, 1],
              outlinks="B", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_backlinks_also_count_as_graph_edges(tmp_vault):
    vault, col = tmp_vault
    # B linkea a A (edge no dirigido para surface). Debería filtrarse igual.
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM,
              outlinks="A", created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_min_hops_1_accepts_direct_neighbors(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              outlinks="B", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    # min_hops=1 ⇒ solo filtra "la misma nota", o sea todo par pasa el filtro.
    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=1)

    assert len(pairs) == 1


# ── Filtros de carpeta / MOC / tags / edad ────────────────────────────────────


@pytest.mark.parametrize("skip_folder", ["00-Inbox", "04-Archive", "05-Reviews"])
def test_skips_notes_in_excluded_folders(tmp_vault, skip_folder):
    vault, col = tmp_vault
    _add_note(col, vault, f"{skip_folder}/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_skips_moc_by_title(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "MOC Salud", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_skips_moc_by_tag(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              tags="moc,ideas", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_skips_folder_index_pattern(tmp_vault):
    vault, col = tmp_vault
    # Convención Obsidian: 02-Areas/Salud/Salud.md es la nota-índice de la carpeta.
    _add_note(col, vault, "02-Areas/Salud/Salud.md", "Salud",
              [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/otros/b.md", "B", HIGH_SIM, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_skips_pairs_sharing_two_or_more_tags(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              tags="foco,salud", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM,
              tags="foco,salud,extra", created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert pairs == []


def test_pairs_sharing_one_tag_still_surface(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              tags="foco,x", created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM,
              tags="foco,y", created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert len(pairs) == 1
    assert pairs[0]["shared_tags"] == ["foco"]


def test_skips_notes_younger_than_threshold(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old(2))
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old(60))

    pairs = rag.find_surface_bridges(
        col, sim_threshold=0.7, min_hops=3, skip_young_days=7
    )

    assert pairs == []


def test_notes_without_created_timestamp_pass_age_filter(tmp_vault):
    vault, col = tmp_vault
    # Sin `created`, _note_age_days devuelve None → no bloquea.
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0])
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM)

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert len(pairs) == 1


# ── Ranking + top + snippets ──────────────────────────────────────────────────


def test_top_cap_limits_results(tmp_vault):
    vault, col = tmp_vault
    # 4 notas de alta similitud entre sí → C(4,2)=6 pares. top=2 ⇒ devolver 2.
    for name, emb in [("a", [1, 0, 0, 0]), ("b", [0.99, 0.14, 0, 0]),
                      ("c", [0.98, 0.2, 0, 0]), ("d", [0.97, 0.24, 0, 0])]:
        _add_note(col, vault, f"02-Areas/{name}.md", name.upper(),
                  emb, created=_old())

    pairs = rag.find_surface_bridges(
        col, sim_threshold=0.7, min_hops=3, top=2
    )

    assert len(pairs) == 2
    # Debe estar ordenado descendente por similitud.
    assert pairs[0]["similarity"] >= pairs[1]["similarity"]


def test_pairs_include_snippets_from_disk(tmp_vault):
    vault, col = tmp_vault
    body_a = "---\ntags:\n- x\n---\n\n# A\n\nCuerpo con contenido real de A."
    body_b = "# B\n\nCuerpo con contenido real de B."
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0],
              body=body_a, created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM,
              body=body_b, created=_old())

    pairs = rag.find_surface_bridges(col, sim_threshold=0.7, min_hops=3)

    assert len(pairs) == 1
    # Frontmatter de A debe haberse stripeado antes del snippet.
    assert "tags" not in pairs[0]["a_snippet"]
    assert "contenido real de A" in pairs[0]["a_snippet"]
    assert "contenido real de B" in pairs[0]["b_snippet"]


# ── Helpers aislados ──────────────────────────────────────────────────────────


def test_build_graph_adj_is_undirected():
    corpus = {
        "outlinks": {"a.md": ["B"], "c.md": ["B"]},
        "title_to_paths": {"B": {"b.md"}},
    }
    adj = rag._build_graph_adj(corpus)
    assert adj["a.md"] == {"b.md"}
    assert adj["b.md"] == {"a.md", "c.md"}   # backedge implícito


def test_hop_set_bfs():
    adj = {
        "a.md": {"b.md"},
        "b.md": {"a.md", "c.md"},
        "c.md": {"b.md", "d.md"},
        "d.md": {"c.md"},
    }
    assert rag._hop_set(adj, "a.md", 0) == {"a.md"}
    assert rag._hop_set(adj, "a.md", 1) == {"a.md", "b.md"}
    assert rag._hop_set(adj, "a.md", 2) == {"a.md", "b.md", "c.md"}
    assert rag._hop_set(adj, "a.md", 3) == {"a.md", "b.md", "c.md", "d.md"}


@pytest.mark.parametrize("meta,expected", [
    ({"note": "MOC Salud", "tags": "", "file": "02-Areas/salud.md"}, True),
    ({"note": "Index de libros", "tags": "", "file": "03-Resources/x.md"}, True),
    ({"note": "Map mental", "tags": "", "file": "02-Areas/x.md"}, True),
    ({"note": "Salud", "tags": "moc", "file": "02-Areas/x.md"}, True),
    ({"note": "Salud", "tags": "foco,moc", "file": "02-Areas/x.md"}, True),
    ({"note": "Salud", "tags": "", "file": "02-Areas/Salud/Salud.md"}, True),
    ({"note": "Nota normal", "tags": "foco", "file": "02-Areas/x.md"}, False),
    ({"note": "Mocoso", "tags": "", "file": "02-Areas/x.md"}, False),  # no confundir con MOC
])
def test_is_moc_heuristics(meta, expected):
    assert rag._is_moc_note(meta) is expected


# ── Logging + CLI integration ─────────────────────────────────────────────────


def test_log_run_writes_summary_and_pairs(tmp_vault):
    vault, col = tmp_vault
    pair = {"a_path": "a.md", "b_path": "b.md", "similarity": 0.85}
    rag._surface_log_run({"n_pairs": 1, "sim_threshold": 0.78}, [pair])

    lines = rag.SURFACE_LOG_PATH.read_text().splitlines()
    assert len(lines) == 2
    summary = json.loads(lines[0])
    detail = json.loads(lines[1])
    assert summary["cmd"] == "surface_run"
    assert summary["n_pairs"] == 1
    assert detail["cmd"] == "surface_pair"
    assert detail["a_path"] == "a.md"
    # Mismo timestamp en run + pair(s) — permite agrupar por ts al leer.
    assert summary["ts"] == detail["ts"]


def test_cli_surface_dry_run_end_to_end(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    # Stub el LLM para no depender de Ollama en CI.
    monkeypatch.setattr(
        rag, "_surface_generate_reason",
        lambda pair: f"Ambas hablan de {pair['a_note']} y {pair['b_note']}.",
    )
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.surface, ["--plain", "--sim-threshold", "0.7"])

    assert result.exit_code == 0, result.output
    assert "02-Areas/a.md" in result.output
    assert "02-Areas/b.md" in result.output
    # La línea con la razón generada por el LLM stub debe aparecer.
    assert "Ambas hablan de" in result.output

    # Log debe tener una línea run + una pair.
    log_lines = rag.SURFACE_LOG_PATH.read_text().splitlines()
    assert any(json.loads(l).get("cmd") == "surface_run" for l in log_lines)
    assert any(json.loads(l).get("cmd") == "surface_pair" for l in log_lines)


def test_cli_surface_no_llm_skips_reason(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH_SIM, created=_old())

    sentinel = {"called": False}

    def boom(pair):
        sentinel["called"] = True
        return "NEVER"

    monkeypatch.setattr(rag, "_surface_generate_reason", boom)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(
        rag.surface, ["--plain", "--sim-threshold", "0.7", "--no-llm"]
    )

    assert result.exit_code == 0, result.output
    assert sentinel["called"] is False
    assert "NEVER" not in result.output


def test_cli_surface_no_pairs_still_logs(tmp_vault):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", [1, 0, 0, 0], created=_old())
    _add_note(col, vault, "02-Areas/c.md", "C", LOW_SIM, created=_old())

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.surface, ["--plain", "--sim-threshold", "0.7"])

    assert result.exit_code == 0, result.output
    log_lines = rag.SURFACE_LOG_PATH.read_text().splitlines()
    run = json.loads(log_lines[-1])
    assert run["cmd"] == "surface_run"
    assert run["n_pairs"] == 0
