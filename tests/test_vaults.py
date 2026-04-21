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


# ── `_is_cross_source_target` + pre-sync guard en `rag index` ────────────────
# Motivación: antes del fix, `rag index` corría 12 ETLs cross-source (MOZE,
# WhatsApp, Gmail, Calendar, Reminders, Drive, GitHub, Chrome, Claude,
# YT, Spotify) que escriben `.md` files al `VAULT_PATH` activo. Con
# `OBSIDIAN_RAG_VAULT` o `rag index --vault X` apuntando a un vault
# distinto del home, los ETLs contaminaban el vault equivocado (medido
# en terreno: 19 archivos MOZE copiados a `obsidian-work/02-Areas/Personal/
# Finanzas/MOZE/` en una corrida que intentaba indexar el vault `work`).
# El guard `_is_cross_source_target` decide si el vault target recibe los
# ETLs (default: solo `_DEFAULT_VAULT`; opt-in explícito via registry).


def test_is_cross_source_target_default_vault_is_home(tmp_registry, monkeypatch):
    """El `_DEFAULT_VAULT` (iCloud Notes, back-compat single-vault) siempre
    es target canónico sin configuración."""
    # Sin registry ni env var → default vault.
    assert rag._is_cross_source_target(rag._DEFAULT_VAULT) is True


def test_is_cross_source_target_custom_vault_rejected_by_default(
    tmp_registry, tmp_path,
):
    """Un vault custom (no el _DEFAULT_VAULT) sin opt-in explícito no recibe
    cross-source syncs. Este es EL fix — antes del guard, los ETLs
    escribían acá sin chequear y contaminaban."""
    work = tmp_path / "work-vault"; work.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(work)},
        "current": "work",
    })
    assert rag._is_cross_source_target(work) is False


def test_is_cross_source_target_explicit_opt_in_via_config(
    tmp_registry, tmp_path,
):
    """Si `cross_source_target` está seteado en `vaults.json`, SOLO ese
    vault recibe los ETLs — el resto se rechaza (incluido el default)."""
    home = tmp_path / "home-vault"; home.mkdir()
    work = tmp_path / "work-vault"; work.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(home), "work": str(work)},
        "current": "home",
        "cross_source_target": "work",
    })
    assert rag._is_cross_source_target(work) is True
    assert rag._is_cross_source_target(home) is False
    # Incluso _DEFAULT_VAULT pierde el privilegio cuando hay override.
    assert rag._is_cross_source_target(rag._DEFAULT_VAULT) is False


def test_is_cross_source_target_config_with_unknown_target_falls_back(
    tmp_registry, tmp_path,
):
    """Si `cross_source_target` apunta a un nombre que no existe en
    `vaults`, el override se ignora y vuelve al back-compat (solo
    `_DEFAULT_VAULT`)."""
    work = tmp_path / "work-vault"; work.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(work)},
        "current": "work",
        "cross_source_target": "ghost",
    })
    assert rag._is_cross_source_target(work) is False
    assert rag._is_cross_source_target(rag._DEFAULT_VAULT) is True


def test_run_cross_source_etls_skips_when_vault_not_target(
    tmp_registry, tmp_path, monkeypatch,
):
    """`_run_cross_source_etls(vault_path)` debe skipear TODOS los sync
    helpers cuando el vault no es target canónico. Cero llamadas a
    `_sync_moze_notes`, `_sync_whatsapp_notes`, ni al loop de ETLs."""
    work = tmp_path / "work-vault"; work.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(work)},
        "current": "work",
    })

    def boom(*args, **kwargs):
        raise AssertionError(f"sync helper NO debería llamarse: args={args}")

    for sync_fn in (
        "_sync_moze_notes", "_sync_whatsapp_notes", "_sync_reminders_notes",
        "_sync_apple_calendar_notes", "_sync_gmail_notes",
    ):
        if hasattr(rag, sync_fn):
            monkeypatch.setattr(rag, sync_fn, boom)

    # No debe lanzar — el guard corta antes de tocar los syncs.
    rag._run_cross_source_etls(work)


def test_run_cross_source_etls_runs_sync_helpers_for_home_vault(
    tmp_registry, tmp_path, monkeypatch,
):
    """Back-compat: cuando el vault ES el target (default o explícito),
    los sync helpers corren como siempre. Verifica que MOZE + WhatsApp
    se invocan con el vault correcto como argumento."""
    called: dict[str, list[Path]] = {}

    def _spy(name):
        def _fn(vault_root, *a, **kw):
            called.setdefault(name, []).append(Path(vault_root))
            return {"ok": True}   # minimal stat shape; no months_written
        return _fn

    # Patch moze + wa + los ETLs del loop para que no escriban disco real.
    monkeypatch.setattr(rag, "_sync_moze_notes", _spy("moze"))
    monkeypatch.setattr(rag, "_sync_whatsapp_notes", _spy("whatsapp"))
    for fn_name in (
        "_sync_reminders_notes", "_sync_apple_calendar_notes",
        "_sync_chrome_history", "_sync_gmail_notes", "_sync_gdrive_notes",
        "_sync_github_activity", "_sync_claude_code_transcripts",
        "_sync_youtube_transcripts", "_sync_spotify_notes",
    ):
        if hasattr(rag, fn_name):
            monkeypatch.setattr(rag, fn_name, _spy(fn_name))

    # Target = _DEFAULT_VAULT (back-compat classic).
    rag._run_cross_source_etls(rag._DEFAULT_VAULT)

    # MOZE + WhatsApp son los dos syncs siempre invocados primero. El resto
    # pasa por el loop de tuples — puede haber shape-specific guards, así
    # que nos bastan los dos principales.
    assert "moze" in called
    assert "whatsapp" in called
    assert called["moze"][0].resolve() == rag._DEFAULT_VAULT.resolve()
    assert called["whatsapp"][0].resolve() == rag._DEFAULT_VAULT.resolve()


# ── CLI: `rag index --vault NAME` ────────────────────────────────────────────
# Simétrico al `--vault` de query/chat. Redirige `_run_index` al vault
# nombrado sin cambiar el `current` del registry. Combina con el guard de
# pre-syncs (arriba): si el vault no es target canónico, los ETLs se
# skippean y solo se indexan las notas reales del vault.


def test_index_vault_flag_routes_to_named_vault(tmp_registry, tmp_path, monkeypatch):
    """`rag index --vault work` debe correr `_run_index` con
    `VAULT_PATH` swappeado al vault `work`. Verifica via context manager."""
    v_home = tmp_path / "home"; v_home.mkdir()
    v_work = tmp_path / "work"; v_work.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home), "work": str(v_work)},
        "current": "home",
    })

    captured: dict = {}

    def fake_run_index(reset, no_contradict):
        # Capturamos el VAULT_PATH visto desde adentro de _run_index —
        # debe ser `v_work` (swappeado por _with_vault).
        captured["vault_path"] = rag.VAULT_PATH
        return {}

    monkeypatch.setattr(rag, "_run_index", fake_run_index)
    # vault_write_lock abre un fichero de lock — lo pasamos a un no-op.
    import contextlib
    monkeypatch.setattr(rag, "vault_write_lock",
                        lambda *a, **kw: contextlib.nullcontext())

    result = CliRunner().invoke(
        rag.cli, ["index", "--vault", "work"],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("vault_path") is not None, result.output
    assert captured["vault_path"].resolve() == v_work.resolve()


def test_index_vault_flag_unknown_name_errors_without_running(
    tmp_registry, tmp_path, monkeypatch,
):
    """`rag index --vault ghost` debe fallar claro, sin correr `_run_index`."""
    v_home = tmp_path / "home"; v_home.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home)},
        "current": "home",
    })

    def boom(*args, **kwargs):
        raise AssertionError("_run_index no debería llamarse")

    monkeypatch.setattr(rag, "_run_index", boom)

    result = CliRunner().invoke(rag.cli, ["index", "--vault", "ghost"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "ghost" in out or "no resolvió" in out, result.output


@pytest.mark.parametrize("scope", ["home,work", "all"])
def test_index_vault_flag_rejects_multi_vault_scope(
    tmp_registry, tmp_path, monkeypatch, scope,
):
    """`rag index` es single-vault por invocación. Scope 'home,work' o
    'all' se rechazan con mensaje claro."""
    v_home = tmp_path / "home"; v_home.mkdir()
    v_work = tmp_path / "work"; v_work.mkdir()
    rag._save_vaults_config({
        "vaults": {"home": str(v_home), "work": str(v_work)},
        "current": "home",
    })

    def boom(*args, **kwargs):
        raise AssertionError("_run_index no debería llamarse")

    monkeypatch.setattr(rag, "_run_index", boom)

    result = CliRunner().invoke(rag.cli, ["index", "--vault", scope])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "un vault" in out or "solo acepta" in out, result.output


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
