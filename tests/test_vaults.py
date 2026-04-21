"""Tests del registry multi-vault.

Cubre la precedencia env > registry > default + el ciclo CRUD del registry
(add / list / use / current / remove). Las commands del CLI se ejecutan
con CliRunner contra un VAULTS_CONFIG_PATH redirigido al tmp_path.
"""
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Aisla el registry en tmp_path. Limpia OBSIDIAN_RAG_VAULT del env."""
    monkeypatch.setattr(rag, "VAULTS_CONFIG_PATH", tmp_path / "vaults.json")
    monkeypatch.delenv("OBSIDIAN_RAG_VAULT", raising=False)
    return tmp_path


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_load_returns_empty_when_missing(tmp_registry):
    cfg = rag._load_vaults_config()
    assert cfg == {"vaults": {}, "current": None}


def test_load_recovers_from_corrupt_file(tmp_registry):
    rag.VAULTS_CONFIG_PATH.write_text("not json{{{")
    assert rag._load_vaults_config() == {"vaults": {}, "current": None}


def test_save_load_roundtrip(tmp_registry):
    rag._save_vaults_config({"vaults": {"home": "/x"}, "current": "home"})
    cfg = rag._load_vaults_config()
    assert cfg["vaults"] == {"home": "/x"}
    assert cfg["current"] == "home"


# ── Precedencia ──────────────────────────────────────────────────────────────


def test_resolve_falls_back_to_default(tmp_registry):
    assert rag._resolve_vault_path() == rag._DEFAULT_VAULT


def test_resolve_uses_registry_current(tmp_registry, tmp_path):
    target = tmp_path / "my-vault"
    target.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(target)},
        "current": "work",
    })
    assert rag._resolve_vault_path() == target


def test_resolve_env_overrides_registry(tmp_registry, tmp_path, monkeypatch):
    target = tmp_path / "my-vault"
    target.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(target)},
        "current": "work",
    })
    other = tmp_path / "env-vault"
    other.mkdir()
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", str(other))
    assert rag._resolve_vault_path() == other


def test_resolve_ignores_current_when_name_not_in_vaults(tmp_registry):
    rag._save_vaults_config({"vaults": {}, "current": "ghost"})
    assert rag._resolve_vault_path() == rag._DEFAULT_VAULT


# ── CLI: add ─────────────────────────────────────────────────────────────────


def test_add_first_vault_becomes_current(tmp_registry, tmp_path):
    v = tmp_path / "v1"
    v.mkdir()
    result = CliRunner().invoke(rag.vault, ["add", "v1", str(v)])
    assert result.exit_code == 0, result.output
    assert "(activo)" in result.output
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v1"
    assert cfg["vaults"]["v1"] == str(v)


def test_add_second_vault_does_not_change_current(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v1"
    assert set(cfg["vaults"].keys()) == {"v1", "v2"}


def test_add_overwrites_existing_name(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "x", str(v1)])
    result = runner.invoke(rag.vault, ["add", "x", str(v2)])
    assert "Sobreescribiendo" in result.output
    cfg = rag._load_vaults_config()
    assert cfg["vaults"]["x"] == str(v2)


def test_add_rejects_nonexistent_path(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["add", "ghost", "/no/such/path"])
    assert result.exit_code != 0


# ── CLI: list / current ──────────────────────────────────────────────────────


def test_list_empty_shows_default(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["list"])
    assert result.exit_code == 0
    assert "Sin vaults registrados" in result.output


def test_list_shows_active_marker(tmp_registry, tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    result = runner.invoke(rag.vault, ["list"])
    assert "v" in result.output
    assert "→" in result.output


def test_list_warns_when_env_overrides(tmp_registry, tmp_path, monkeypatch):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = runner.invoke(rag.vault, ["list"])
    assert "OBSIDIAN_RAG_VAULT" in result.output


def test_current_default(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["current"])
    assert "default" in result.output


def test_current_registry(tmp_registry, tmp_path):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    result = runner.invoke(rag.vault, ["current"])
    assert "registry" in result.output
    assert "v" in result.output


def test_current_env(tmp_registry, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = CliRunner().invoke(rag.vault, ["current"])
    assert "env" in result.output
    assert "/some/env/path" in result.output


# ── CLI: use ─────────────────────────────────────────────────────────────────


def test_use_switches_current(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    runner.invoke(rag.vault, ["use", "v2"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v2"


def test_use_unknown_name_errors(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["use", "ghost"])
    assert "no registrado" in result.output


def test_use_warns_when_env_set(tmp_registry, tmp_path, monkeypatch):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = runner.invoke(rag.vault, ["use", "v"])
    assert "OBSIDIAN_RAG_VAULT" in result.output


# ── CLI: remove ──────────────────────────────────────────────────────────────


def test_remove_keeps_other_vaults(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    runner.invoke(rag.vault, ["remove", "v1"])
    cfg = rag._load_vaults_config()
    assert "v1" not in cfg["vaults"]
    assert "v2" in cfg["vaults"]


def test_remove_current_falls_to_next_or_none(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    # current = v1 (primero registrado)
    runner.invoke(rag.vault, ["remove", "v1"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v2"
    runner.invoke(rag.vault, ["remove", "v2"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] is None


def test_remove_unknown_errors(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["remove", "ghost"])
    assert "no registrado" in result.output


# ── resolve_vault_paths ──────────────────────────────────────────────────────


def test_resolve_vault_paths_none_returns_active_single(tmp_registry, tmp_path):
    v = tmp_path / "v"; v.mkdir()
    rag._save_vaults_config({"vaults": {"home": str(v)}, "current": "home"})
    result = rag.resolve_vault_paths(None)
    assert len(result) == 1
    assert result[0][0] == "home"
    assert result[0][1] == v


def test_resolve_vault_paths_all_expands_registry(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v1), "work": str(v2)},
        "current": "home",
    })
    result = rag.resolve_vault_paths(["all"])
    names = sorted(n for n, _ in result)
    assert names == ["home", "work"]


def test_resolve_vault_paths_filters_unknown(tmp_registry, tmp_path):
    v = tmp_path / "v"; v.mkdir()
    rag._save_vaults_config({"vaults": {"home": str(v)}, "current": "home"})
    result = rag.resolve_vault_paths(["home", "ghost"])
    names = [n for n, _ in result]
    assert names == ["home"]   # "ghost" silent drop


def test_resolve_vault_paths_with_env_overrides(tmp_registry, tmp_path, monkeypatch):
    v = tmp_path / "envvault"; v.mkdir()
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", str(v))
    result = rag.resolve_vault_paths(None)
    assert len(result) == 1
    name, path = result[0]
    assert path == v
    assert name.startswith("env:")


# ── _collection_name_for_vault + get_db_for ──────────────────────────────────


def test_collection_name_uses_base_for_default():
    # _DEFAULT_VAULT → nombre base sin sufijo.
    name = rag._collection_name_for_vault(rag._DEFAULT_VAULT)
    assert name == rag._COLLECTION_BASE


def test_collection_name_adds_hash_for_custom_path(tmp_path):
    v = tmp_path / "custom"; v.mkdir()
    name = rag._collection_name_for_vault(v)
    assert name.startswith(rag._COLLECTION_BASE + "_")
    # Hash estable — mismo path → mismo nombre.
    assert rag._collection_name_for_vault(v) == name


def test_get_db_for_returns_distinct_collections_per_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    c1 = rag.get_db_for(v1)
    c2 = rag.get_db_for(v2)
    assert c1.name != c2.name
    # Insertar algo en c1 no aparece en c2 (aislamiento).
    c1.add(
        ids=["a::0"], embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=["doc a"], metadatas=[{"file": "a.md", "note": "a"}],
    )
    assert c1.count() == 1
    assert c2.count() == 0


# ── multi_retrieve ────────────────────────────────────────────────────────────


@pytest.fixture
def two_vaults(tmp_path, monkeypatch):
    """Dos tmp vaults con contenidos distintos, cada uno con su colección."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "embed", lambda ts: [[1.0, 0.0, 0.0, 0.0] for _ in ts])
    # Forzamos nombres de colección reproducibles.
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    c1 = rag.get_db_for(v1)
    c2 = rag.get_db_for(v2)

    def add(col, vault_dir, rel, body):
        full = vault_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(body, encoding="utf-8")
        col.add(
            ids=[f"{rel}::0"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            documents=[body],
            metadatas=[{
                "file": rel, "note": Path(rel).stem,
                "folder": str(Path(rel).parent),
                "tags": "", "outlinks": "", "hash": "x",
            }],
        )
    add(c1, v1, "02-Areas/nota-home.md", "contenido exclusivo de home")
    add(c2, v2, "02-Areas/nota-work.md", "contenido exclusivo de work")
    rag._invalidate_corpus_cache()
    return [("home", v1), ("work", v2)]


def test_multi_retrieve_single_vault_fastpath(two_vaults, monkeypatch):
    # Con un solo vault el wrapper no debería merge-rankear — pasa derecho.
    monkeypatch.setattr(rag, "get_reranker", lambda: _fake_reranker())
    r = rag.multi_retrieve([two_vaults[0]], "contenido", k=3, folder=None)
    assert len(r["docs"]) >= 1
    assert r["vault_scope"] == ["home"]
    # Single vault NO anota _vault en las metas (para no romper código legacy).
    assert all("_vault" not in m for m in r["metas"])


def test_multi_retrieve_merges_both_vaults_and_annotates(two_vaults, monkeypatch):
    monkeypatch.setattr(rag, "get_reranker", lambda: _fake_reranker())
    r = rag.multi_retrieve(two_vaults, "contenido", k=5, folder=None)
    vault_names = {m.get("_vault") for m in r["metas"]}
    assert "home" in vault_names
    assert "work" in vault_names
    # Todas las metas del modo multi deben tener _vault + _vault_path.
    for m in r["metas"]:
        assert m.get("_vault") in ("home", "work")
        assert m.get("_vault_path")
    assert sorted(r["vault_scope"]) == ["home", "work"]


def test_multi_retrieve_skips_empty_vaults(two_vaults, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "get_reranker", lambda: _fake_reranker())
    # Agregamos un tercero vacío (nunca se indexó).
    v3 = tmp_path / "v3"; v3.mkdir()
    rag.get_db_for(v3)   # crea colección vacía
    scope = two_vaults + [("empty", v3)]
    r = rag.multi_retrieve(scope, "contenido", k=5, folder=None)
    vaults_found = {m.get("_vault") for m in r["metas"]}
    assert "empty" not in vaults_found
    assert r["docs"]   # no queda en blanco solo por el vacío


def test_multi_retrieve_empty_scope_returns_empty():
    r = rag.multi_retrieve([], "cualquiera", k=5, folder=None)
    assert r["docs"] == []
    assert r["vault_scope"] == []
    assert r["confidence"] == float("-inf")


# ── Helper: fake reranker ─────────────────────────────────────────────────────


def _fake_reranker():
    """CrossEncoder mock que devuelve score=1.0 para todos los pairs — los
    tests de multi_retrieve no dependen del ranking semántico real, solo
    del merge y anotación."""
    class _R:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [1.0] * len(pairs)
    return _R()


# ── CLI: rag query --vault ────────────────────────────────────────────────────
# Sanidad: el query command acepta override per-invocación del vault activo
# igual que el env var OBSIDIAN_RAG_VAULT / `rag vault use`. Evita tener que
# switchear el current del registry para una consulta única contra un vault
# distinto (ej. `rag query --vault work "dev cycles"`). Single-vault only
# por ahora — cross-vault sigue siendo territorio de `rag chat --vault a,b`.


class _FakeCountedCollection:
    """Colección mínima: count() > 0 para pasar el guard de `query()` y no
    disparar el `Índice vacío.` early-return."""
    def count(self):
        return 1


def _stub_query_pipeline_past_retrieve(monkeypatch):
    """Monkeypatches comunes para que `rag query` llegue hasta el retrieve()
    sin tocar ollama, embeddings reales ni el reranker. Los tests que corren
    después del retrieve con este stub deben pasar `retrieve` vía monkeypatch
    a lo que quieran ejercer. Acá short-circuiteamos devolviendo un result
    vacío — el comando imprime 'Sin resultados.' y sale limpio.
    """
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_vocabulary", lambda col: ([], []))
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("semantic", {}))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: {
        "docs": [], "metas": [], "scores": [], "confidence": 0.0,
        "filters_applied": {}, "query_variants": [], "timing": {},
    })
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)


def test_query_vault_flag_routes_to_named_vault(tmp_registry, tmp_path, monkeypatch):
    """`--vault work` debe resolver `work` contra el registry y abrir su
    colección vía `get_db_for`, sin tocar el `get_db()` del vault activo."""
    v_home = tmp_path / "home"; v_home.mkdir()
    v_work = tmp_path / "work"; v_work.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home), "work": str(v_work)},
        "current": "home",
    })

    captured_paths: list[Path] = []

    def fake_get_db_for(path):
        captured_paths.append(Path(path))
        return _FakeCountedCollection()

    def boom_get_db():
        raise AssertionError("get_db() no debe llamarse cuando --vault está seteado")

    monkeypatch.setattr(rag, "get_db_for", fake_get_db_for)
    monkeypatch.setattr(rag, "get_db", boom_get_db)
    _stub_query_pipeline_past_retrieve(monkeypatch)

    result = CliRunner().invoke(
        rag.cli, ["query", "--plain", "--no-multi", "--vault", "work", "test"],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_paths) == 1, captured_paths
    assert captured_paths[0].resolve() == v_work.resolve()


def test_query_without_vault_uses_default_get_db(tmp_registry, tmp_path, monkeypatch):
    """Sin `--vault` el flujo debe seguir usando `get_db()` (vault activo),
    conservando el comportamiento pre-flag."""
    v_home = tmp_path / "home"; v_home.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home)},
        "current": "home",
    })

    get_db_called = {"n": 0}

    def fake_get_db():
        get_db_called["n"] += 1
        return _FakeCountedCollection()

    def boom_get_db_for(path):
        raise AssertionError(f"get_db_for({path}) no debe llamarse sin --vault")

    monkeypatch.setattr(rag, "get_db", fake_get_db)
    monkeypatch.setattr(rag, "get_db_for", boom_get_db_for)
    _stub_query_pipeline_past_retrieve(monkeypatch)

    result = CliRunner().invoke(rag.cli, ["query", "--plain", "--no-multi", "test"])

    assert result.exit_code == 0, result.output
    assert get_db_called["n"] == 1


def test_query_vault_flag_unknown_name_errors_without_touching_db(
    tmp_registry, tmp_path, monkeypatch,
):
    """`--vault ghost` sin registry match debe fallar claro, sin abrir ninguna
    colección ni llamar retrieve/LLM."""
    v_home = tmp_path / "home"; v_home.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home)},
        "current": "home",
    })

    def boom(*args, **kwargs):
        raise AssertionError(f"no debería llamarse: args={args}")

    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", boom)
    monkeypatch.setattr(rag, "get_db_for", boom)
    monkeypatch.setattr(rag, "retrieve", boom)

    result = CliRunner().invoke(
        rag.cli, ["query", "--plain", "--vault", "ghost", "test"],
    )

    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "ghost" in out or "no resolvió" in out, result.output


@pytest.mark.parametrize("scope", ["home,work", "all"])
def test_query_vault_flag_rejects_multi_vault_scope(
    tmp_registry, tmp_path, monkeypatch, scope,
):
    """`rag query` no soporta cross-vault retrieval en esta primera iteración
    (la ruta multi_retrieve + intent shortcuts + render_related tiene suficiente
    divergencia como para merecer un cambio separado). Para cross-vault queda
    `rag chat --vault a,b`. Debe rechazarse con mensaje claro, sin tocar DB."""
    v_home = tmp_path / "home"; v_home.mkdir()
    v_work = tmp_path / "work"; v_work.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home), "work": str(v_work)},
        "current": "home",
    })

    def boom(*args, **kwargs):
        raise AssertionError(f"no debería llamarse: args={args}")

    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", boom)
    monkeypatch.setattr(rag, "get_db_for", boom)
    monkeypatch.setattr(rag, "retrieve", boom)

    result = CliRunner().invoke(
        rag.cli, ["query", "--plain", "--vault", scope, "test"],
    )

    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "rag chat" in out or "un vault" in out or "solo acepta" in out, result.output
