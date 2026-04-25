"""Tests para el LLM Wiki layer — compiled-knowledge pages generadas durante
`auto_index_vault` y escritas bajo `<vault>/Wiki/`.

Cubre (los 8 casos del dispatch plan del PM + un extra del kill-switch):

  1. Una página wiki se crea cuando el indexer procesa una nota nueva.
  2. El frontmatter tiene las 5 claves esperadas (source, source_hash,
     generated_by, generated_at, wiki_version).
  3. Re-correr el indexer sin cambiar la fuente NO rewritea la página
     (mtime intacto — el cache + `_atomic_write_if_changed` colaboran).
  4. `Wiki/index.md` se crea + lista las páginas generadas.
  5. `is_excluded("Wiki/...")` devuelve True para que el layer no se
     re-ingeste a sí mismo (infinite loop).
  6. Si el helper model falla (timeout / excepción), el indexing de la
     nota original igual termina OK y la página wiki no se escribe
     (silent-fail).
  7. Si el usuario ya tiene una nota escrita a mano en <vault>/Wiki/
     (sin el frontmatter `generated_by: rag-wiki`), el sistema NO la
     pisa cuando le toca generar la página.
  8. `_atomic_write_if_changed` no toca mtime si el diff es cero
     (invariante del contrato — testea el utility del que depende todo
     el resto).
  9. Kill-switch: `OBSIDIAN_RAG_WIKI_ENABLED=0` desactiva todo el
     layer (no se crea Wiki/, no se llama al helper).

Las pruebas nunca tocan ollama real — el `_summary_client` se
monkeypatch para devolver un JSON canned. La única excepción es el
test de `_atomic_write_if_changed`, que trabaja sólo con el FS.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def wiki_env(tmp_path, monkeypatch):
    """Aísla la DB, el state de auto-index y el cache del wiki en `tmp_path`.

    - `WIKI_CACHE_PATH` → tmp (no toca ~/.local/share).
    - `DB_PATH` + `AUTO_INDEX_STATE_PATH` → tmp (mismo pattern que test_auto_index.py).
    - `embed` → stub determinista (no inicializa sentence-transformers).
    - Wiki cache in-memory se resetea a None/False al entrar al fixture
      para que tests consecutivos no compartan estado accidentalmente.
    - `OBSIDIAN_RAG_WIKI_ENABLED=1` explícito por default — el kill-switch
      test lo sobreescribe.
    """
    monkeypatch.setattr(rag, "WIKI_CACHE_PATH", tmp_path / "wiki_pages.json")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "AUTO_INDEX_STATE_PATH", tmp_path / "auto_index_state.json")
    monkeypatch.setattr(rag, "AMBIENT_CONFIG_PATH", tmp_path / "ambient.json")

    monkeypatch.setattr(rag, "_wiki_cache", None)
    monkeypatch.setattr(rag, "_wiki_cache_dirty", False)

    def fake_embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", fake_embed)

    monkeypatch.setenv("OBSIDIAN_RAG_WIKI_ENABLED", "1")
    return tmp_path


def _mk_vault(root: Path, files: dict[str, str]) -> Path:
    """Helper para armar un vault in-memory con los .md indicados."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _stub_helper_model(monkeypatch, payload: dict) -> MagicMock:
    """Reemplaza `_summary_client()` con un mock que devuelve un JSON canned.

    Devuelve el mock client para que el test pueda inspeccionar calls / resetear.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.message.content = json.dumps(payload)
    mock_client.chat.return_value = mock_resp
    monkeypatch.setattr(rag, "_summary_client", lambda: mock_client)
    return mock_client


# ── 1. Página wiki se crea al indexar una nota nueva ────────────────────────


def test_wiki_page_created_on_index(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "02-Areas/test-note.md": "# Test Note\n\n" + ("Some content body " * 30),
    })
    _stub_helper_model(monkeypatch, {
        "summary": ["A test note about testing patterns."],
        "key_claims": ["Testing is good."],
        "mentions": ["Tests"],
    })

    result = rag.auto_index_vault(vault)
    assert result["indexed"] >= 1

    wiki_page = vault / "Wiki" / "02-Areas--test-note.md"
    assert wiki_page.is_file(), (
        "Post-index hook debió crear la página wiki en Wiki/02-Areas--test-note.md. "
        "Files presentes: " + str(list((vault / "Wiki").glob("*")) if (vault / "Wiki").exists() else [])
    )
    body = wiki_page.read_text(encoding="utf-8")
    assert "# Test Note" in body
    assert "A test note about testing patterns." in body
    assert "[[Tests]]" in body, "mentions debería renderizar como wikilinks"


# ── 2. Frontmatter tiene las 5 claves esperadas ─────────────────────────────


def test_wiki_page_frontmatter_correct(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "01-Projects/feature.md": "# Feature\n\n" + ("Body " * 200),
    })
    _stub_helper_model(monkeypatch, {
        "summary": ["A feature."],
        "key_claims": ["It works."],
        "mentions": ["Feature"],
    })

    rag.auto_index_vault(vault)

    wiki_page = vault / "Wiki" / "01-Projects--feature.md"
    body = wiki_page.read_text(encoding="utf-8")
    assert body.startswith("---\n"), "Frontmatter debe estar al principio del file"
    assert 'source: "01-Projects/feature.md"' in body
    assert "source_hash:" in body
    assert f'generated_by: "{rag._WIKI_GENERATED_BY}"' in body
    assert "generated_at:" in body
    assert f"wiki_version: {rag._WIKI_PAGE_VERSION}" in body


# ── 3. No regeneration si el source_hash matchea (mtime intacto) ─────────────


def test_wiki_page_no_regen_on_hash_match(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "note.md": "# Note\n\n" + ("Body " * 200),
    })
    mock_client = _stub_helper_model(monkeypatch, {
        "summary": ["A note."],
        "key_claims": ["Claim."],
        "mentions": ["Note"],
    })

    rag.auto_index_vault(vault)
    wiki_page = vault / "Wiki" / "note.md"
    assert wiki_page.is_file()
    mtime_first = wiki_page.stat().st_mtime
    calls_after_first = mock_client.chat.call_count
    assert calls_after_first >= 1

    # Aseguro que el tiempo avanza lo suficiente para que el FS registre
    # un mtime distinto si hubiera rewrite.
    time.sleep(0.05)

    # Re-ejecutar el hook directamente (auto_index_vault con no_changes
    # no itera paths, así que vamos al hook a mano para simular "volvió
    # a pasar por el mismo file con el mismo hash").
    rag._wiki_ingest_indexed_files(vault, [vault / "note.md"])

    mtime_second = wiki_page.stat().st_mtime
    assert mtime_second == mtime_first, (
        f"Same source_hash → _atomic_write_if_changed debe NO tocar mtime. "
        f"first={mtime_first} second={mtime_second}"
    )
    # Cache hit: el helper NO se llamó de nuevo.
    assert mock_client.chat.call_count == calls_after_first, (
        "Con source_hash idéntico el LLM no debería correr (cache hit)."
    )


# ── 4. Wiki/index.md se crea + lista las páginas ────────────────────────────


def test_wiki_index_md_created(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "a.md": "# Alpha\n\n" + ("Body " * 100),
        "b.md": "# Beta\n\n" + ("Body " * 100),
    })
    _stub_helper_model(monkeypatch, {
        "summary": ["s"], "key_claims": ["c"], "mentions": ["m"],
    })

    rag.auto_index_vault(vault)

    idx_path = vault / "Wiki" / "index.md"
    assert idx_path.is_file()
    idx_body = idx_path.read_text(encoding="utf-8")
    assert "# Wiki Index" in idx_body
    # Links correctos: [[slug|title]]
    assert "[[a|Alpha]]" in idx_body
    assert "[[b|Beta]]" in idx_body
    # Frontmatter del index
    assert 'generated_by: "rag-wiki"' in idx_body
    assert "wiki_role: index" in idx_body


# ── 5. is_excluded skippea Wiki/ (evita infinite loop) ──────────────────────


def test_is_excluded_skips_wiki():
    assert rag.is_excluded("Wiki/note.md") is True
    assert rag.is_excluded("Wiki/02-Areas--foo.md") is True
    assert rag.is_excluded("Wiki/index.md") is True
    assert rag.is_excluded("Wiki") is True
    # Sanity: un subdir que casualmente se llame Wiki dentro de otra
    # carpeta NO debe excluirse — sólo el Wiki/ top-level.
    assert rag.is_excluded("02-Areas/Wiki/note.md") is False


# ── 6. Silent-fail si el helper se cuelga ───────────────────────────────────


def test_wiki_silent_fail_on_ollama_timeout(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "note.md": "# Note\n\n" + ("Body " * 200),
    })
    mock_client = MagicMock()
    mock_client.chat.side_effect = TimeoutError("ollama se colgó")
    monkeypatch.setattr(rag, "_summary_client", lambda: mock_client)

    result = rag.auto_index_vault(vault)
    assert result["indexed"] >= 1, "El indexing del source en sí debe seguir adelante"
    # La página wiki NO debería existir (el helper nunca devolvió nada).
    assert not (vault / "Wiki" / "note.md").exists()


# ── 7. Respeta páginas escritas a mano por el usuario ───────────────────────


def test_wiki_respects_existing_user_notes(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {
        "note.md": "# Note\n\n" + ("Body " * 200),
    })
    # El user YA tiene su propia nota en Wiki/note.md, sin frontmatter
    # `generated_by: rag-wiki`. Es sagrada — no pisarla.
    user_page = vault / "Wiki" / "note.md"
    user_page.parent.mkdir(parents=True, exist_ok=True)
    user_content = "# Mis propias notas\n\nContenido escrito a mano. No pisar.\n"
    user_page.write_text(user_content, encoding="utf-8")

    _stub_helper_model(monkeypatch, {
        "summary": ["s"], "key_claims": ["c"], "mentions": ["m"],
    })

    rag.auto_index_vault(vault)

    assert user_page.read_text(encoding="utf-8") == user_content, (
        "Página user-authored (sin marker) NO debe ser sobreescrita."
    )


# ── 8. _atomic_write_if_changed no toca mtime cuando el diff es cero ────────


def test_atomic_write_if_changed_no_touch_mtime(tmp_path):
    target = tmp_path / "test.md"
    target.write_text("same content\n", encoding="utf-8")
    mtime_before = target.stat().st_mtime

    time.sleep(0.05)

    result = rag._atomic_write_if_changed(target, "same content\n")
    mtime_after = target.stat().st_mtime

    assert result is False, "Cuando no hay cambios debe devolver False (no escribió)"
    assert mtime_before == mtime_after, (
        "mtime NO debe moverse si el contenido es bit-idéntico — el "
        "indexer hash-cache + el wiki idempotent-scan dependen de eso."
    )


# ── 9. Kill-switch desactiva todo el layer ──────────────────────────────────


def test_wiki_kill_switch_disables(wiki_env, monkeypatch):
    vault = wiki_env / "vault"
    _mk_vault(vault, {"note.md": "# Note\n\n" + ("Body " * 200)})
    monkeypatch.setenv("OBSIDIAN_RAG_WIKI_ENABLED", "0")
    # Stub defensive: el _summary_client es COMPARTIDO con
    # _generate_context_summary / _generate_synthetic_questions del
    # indexer normal, así que no podemos asumir call_count == 0 sobre el
    # mock. Lo que sí podemos afirmar es el efecto observable: Wiki/ no
    # se crea y el hook devuelve disabled=True.
    _stub_helper_model(monkeypatch, {
        "summary": ["s"], "key_claims": ["c"], "mentions": ["m"],
    })

    rag.auto_index_vault(vault)

    assert not (vault / "Wiki").exists(), (
        "OBSIDIAN_RAG_WIKI_ENABLED=0 debe impedir que se cree Wiki/ por completo"
    )

    # Además, llamar al hook directamente con la kill-switch activa
    # devuelve disabled=True sin procesar nada.
    result = rag._wiki_ingest_indexed_files(vault, [vault / "note.md"])
    assert result == {
        "pages_written": 0,
        "pages_skipped": 0,
        "errors": 0,
        "disabled": True,
    }
    # El Wiki/ sigue sin existir incluso después de invocar el hook.
    assert not (vault / "Wiki").exists()


# ── Extra: helpers unit-tested en aislamiento ───────────────────────────────


def test_wiki_page_path_flat_slug(tmp_path):
    """Paths anidados colapsan a un único nivel bajo Wiki/."""
    v = tmp_path / "vault"
    got = rag._wiki_page_path(v, "02-Areas/Finanzas/ahorro.md")
    assert got == v / "Wiki" / "02-Areas--Finanzas--ahorro.md"
    # Un file top-level:
    got = rag._wiki_page_path(v, "readme.md")
    assert got == v / "Wiki" / "readme.md"


def test_is_rag_wiki_page_detects_marker(tmp_path):
    """Sólo páginas con `generated_by: rag-wiki` en el frontmatter son
    tratadas como auto-generadas; user-authored notes (sin frontmatter o
    con otros valores) quedan protegidas."""
    auto = tmp_path / "auto.md"
    auto.write_text(
        '---\n'
        'source: "x.md"\n'
        'generated_by: "rag-wiki"\n'
        '---\n'
        '# Auto\n',
        encoding="utf-8",
    )
    assert rag._is_rag_wiki_page(auto) is True

    user = tmp_path / "user.md"
    user.write_text("# Mis notas\n\nNo hay frontmatter.\n", encoding="utf-8")
    assert rag._is_rag_wiki_page(user) is False

    other = tmp_path / "other.md"
    other.write_text(
        '---\n'
        'generated_by: "something-else"\n'
        '---\n'
        '# Otro\n',
        encoding="utf-8",
    )
    assert rag._is_rag_wiki_page(other) is False


def test_wiki_cache_hit_skips_llm(wiki_env, monkeypatch):
    """Si ya hay entrada en el cache para un source_hash dado, el LLM
    no debe correr y se devuelve la entrada cacheada."""
    # Pre-poblar el cache en-memoria con un payload válido.
    cached_payload = {
        "summary": ["from cache"],
        "key_claims": ["cache claim"],
        "mentions": ["CacheEntity"],
    }
    source_hash = "deadbeefcafebabe"

    # Forzar inicialización del cache.
    rag._load_wiki_cache()
    rag._wiki_cache[source_hash] = json.dumps(cached_payload)

    mock_client = MagicMock()
    monkeypatch.setattr(rag, "_summary_client", lambda: mock_client)

    got = rag._get_or_generate_wiki_page_data(
        text="Long enough body " * 50,
        source_hash=source_hash,
        title="Cached",
        folder="",
    )
    assert got == cached_payload
    assert mock_client.chat.call_count == 0, "cache hit ⇒ no LLM call"


def test_wiki_enabled_default_is_on(monkeypatch):
    """Sin env var seteada, `_wiki_enabled()` devuelve True (default on)."""
    monkeypatch.delenv("OBSIDIAN_RAG_WIKI_ENABLED", raising=False)
    assert rag._wiki_enabled() is True


def test_wiki_enabled_respects_falsy(monkeypatch):
    """Variantes explícitas de off: '0', 'false', 'no' (case-insensitive)."""
    for falsy in ("0", "false", "no", "FALSE", "No"):
        monkeypatch.setenv("OBSIDIAN_RAG_WIKI_ENABLED", falsy)
        assert rag._wiki_enabled() is False, \
            f"OBSIDIAN_RAG_WIKI_ENABLED={falsy!r} debe ser off"
