"""Tests para OCR de imágenes embebidas durante el indexing.

Motivación (2026-04-21): nota `dev cycles.md` con su contenido informativo
en una PNG embebida (tabla de cycles) era invisible al retrieval porque el
body textual era solo link + embed. Agregando OCR via Apple Vision, el
texto de la imagen se concatena al body antes de chunkear, haciendo la
data buscable.

Tres piezas bajo test:
  1. `_extract_embedded_images(body, note_path, vault_root) -> list[Path]`
     — encuentra `![[...]]` y `![alt](...)` con extensiones de imagen,
     resuelve paths relativos a la nota y al vault. Ignora non-image
     embeds (notas, PDFs) y URLs externas.
  2. `_ocr_image(image_path) -> str` — corre Apple Vision, cachea por
     (abs_path, mtime). Silent-fails si ocrmac no está importable o la
     imagen no se puede procesar.
  3. `_enrich_body_with_ocr(body, note_path, vault_root) -> str` —
     orquesta los dos helpers anteriores: para cada imagen embebida
     obtiene el texto OCR, concatena al body con un marker que el user
     puede ver (pero no corrompe el display original del markdown).
"""
import sqlite3
from pathlib import Path

import pytest

import rag


# ── _extract_embedded_images: regex-based ─────────────────────────────────────


def test_extract_wikilink_embed_same_folder(tmp_path):
    """`![[imagen.png]]` con la imagen en la misma carpeta que la nota."""
    vault = tmp_path / "vault"
    vault.mkdir()
    folder = vault / "03-Resources"
    folder.mkdir()
    img = folder / "captura.png"
    img.write_bytes(b"fake png")
    note = folder / "dev cycles.md"
    note.write_text("# Dev Cycles\n\n![[captura.png]]\n", encoding="utf-8")

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert len(found) == 1
    assert found[0].resolve() == img.resolve()


def test_extract_wikilink_embed_elsewhere_in_vault(tmp_path):
    """`![[Attachments/img.png]]` resuelve buscando en el vault global
    (Obsidian default: attachments folder configurable pero los wikilinks
    se resuelven por filename global si no son path absoluto)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    attach = vault / "04-Archive/99-Attachments"
    attach.mkdir(parents=True)
    img = attach / "captura-2025.png"
    img.write_bytes(b"fake png")
    note = vault / "03-Resources/dev cycles.md"
    note.parent.mkdir(parents=True)
    note.write_text("![[captura-2025.png]]", encoding="utf-8")

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert len(found) == 1, found
    assert found[0].resolve() == img.resolve()


def test_extract_markdown_embed(tmp_path):
    """`![alt](path/to/img.png)` con path relativo a la nota."""
    vault = tmp_path / "v"
    vault.mkdir()
    img = vault / "folder/img.jpg"
    img.parent.mkdir()
    img.write_bytes(b"fake jpg")
    note = vault / "folder/note.md"
    note.write_text("![cap](img.jpg)", encoding="utf-8")

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert len(found) == 1
    assert found[0].resolve() == img.resolve()


def test_extract_ignores_non_image_extensions(tmp_path):
    """`![[nota.md]]` o `![[archivo.pdf]]` no son imágenes — skip."""
    vault = tmp_path / "v"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "![[otra-nota.md]]\n![[doc.pdf]]\n![[archivo.canvas]]",
        encoding="utf-8",
    )

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert found == []


def test_extract_ignores_external_urls(tmp_path):
    """`![alt](https://...)` es URL externa — skip, no queremos fetch."""
    vault = tmp_path / "v"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text(
        "![remote](https://example.com/foo.png)\n"
        "![also](http://example.com/bar.jpg)",
        encoding="utf-8",
    )

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert found == []


def test_extract_ignores_missing_images(tmp_path):
    """Imagen referenciada pero no existe en disco — skip silently."""
    vault = tmp_path / "v"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("![[does-not-exist.png]]", encoding="utf-8")

    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert found == []


def test_extract_handles_multiple_images(tmp_path):
    """Una nota con varios embeds debe devolver todos."""
    vault = tmp_path / "v"
    vault.mkdir()
    for name in ("a.png", "b.jpg", "c.jpeg"):
        (vault / name).write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text(
        "![[a.png]]\n\n![[b.jpg]]\n\n![alt](c.jpeg)",
        encoding="utf-8",
    )
    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert len(found) == 3
    names = {p.name for p in found}
    assert names == {"a.png", "b.jpg", "c.jpeg"}


@pytest.mark.parametrize("ext", ["png", "jpg", "jpeg", "PNG", "JPG", "heic", "webp"])
def test_extract_supports_common_image_extensions(tmp_path, ext):
    """Extensiones soportadas: png, jpg/jpeg (case-insensitive), heic, webp."""
    vault = tmp_path / "v"
    vault.mkdir()
    img = vault / f"img.{ext}"
    img.write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text(f"![[img.{ext}]]", encoding="utf-8")
    found = rag._extract_embedded_images(
        note.read_text(), note_path=note, vault_root=vault,
    )
    assert len(found) == 1


def test_extract_empty_body_returns_empty():
    """Body vacío → []."""
    from pathlib import Path
    assert rag._extract_embedded_images("", Path("x.md"), Path(".")) == []


# ── _ocr_image: mocks de ocrmac + cache ──────────────────────────────────────


@pytest.fixture
def _clean_ocr_cache(monkeypatch, tmp_path):
    """Cada test arranca con una DB temporal de caché OCR limpia para evitar
    cross-pollination via la cache persistente."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    # Invalidar cualquier singleton de conexión ya abierto.
    if hasattr(rag, "_ocr_cache_invalidate"):
        rag._ocr_cache_invalidate()
    yield
    if hasattr(rag, "_ocr_cache_invalidate"):
        rag._ocr_cache_invalidate()


class _FakeOCRResult:
    """Mock del retorno de ocrmac.OCR(path).recognize() — lista de tuples
    (text, confidence, bbox)."""
    def __init__(self, texts: list[tuple[str, float]]):
        self._texts = texts

    def recognize(self):
        return [(t, c, (0, 0, 0, 0)) for t, c in self._texts]


def _install_fake_ocrmac(monkeypatch, texts: list[tuple[str, float]]):
    """Instala un mock de ocrmac.OCR en el módulo rag para que _ocr_image
    lo encuentre via el import interno. Reemplaza rag._ocrmac_module."""
    class _FakeOCR:
        def __init__(self, path, language_preference=None):
            self._path = path

        def recognize(self):
            return [(t, c, (0, 0, 0, 0)) for t, c in texts]

    class _FakeModule:
        OCR = _FakeOCR

    monkeypatch.setattr(rag, "_ocrmac_module", _FakeModule)


def test_ocr_image_returns_concatenated_text(_clean_ocr_cache, tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    _install_fake_ocrmac(monkeypatch, [
        ("Start Date", 1.0), ("03-Jul-2025", 1.0), ("10.54", 1.0),
    ])
    out = rag._ocr_image(img)
    assert "Start Date" in out
    assert "03-Jul-2025" in out
    assert "10.54" in out


def test_ocr_image_cache_hits_on_second_call(_clean_ocr_cache, tmp_path, monkeypatch):
    """Segunda llamada debe hit la cache sin re-invocar ocrmac (verificado
    via call counter)."""
    img = tmp_path / "cached.png"
    img.write_bytes(b"fake")
    call_count = {"n": 0}

    class _FakeOCR:
        def __init__(self, path, language_preference=None):
            pass
        def recognize(self_inner):
            call_count["n"] += 1
            return [("cached text", 1.0, (0, 0, 0, 0))]

    class _FakeModule:
        OCR = _FakeOCR
    monkeypatch.setattr(rag, "_ocrmac_module", _FakeModule)

    t1 = rag._ocr_image(img)
    t2 = rag._ocr_image(img)
    assert t1 == t2
    assert call_count["n"] == 1, (
        f"Expected 1 OCR call (cache hit on 2nd); got {call_count['n']}"
    )


def test_ocr_image_cache_invalidates_on_mtime_change(
    _clean_ocr_cache, tmp_path, monkeypatch,
):
    """Si la imagen se actualiza (nuevo mtime), la cache debe re-OCRear."""
    img = tmp_path / "img.png"
    img.write_bytes(b"v1")

    call_count = {"n": 0}
    texts_per_call = [("first", 1.0)], [("second", 1.0)]

    class _FakeOCR:
        def __init__(self, path, language_preference=None):
            pass
        def recognize(self_inner):
            n = call_count["n"]
            call_count["n"] += 1
            out = texts_per_call[min(n, 1)]
            return [(t, c, (0, 0, 0, 0)) for t, c in out]

    class _FakeModule:
        OCR = _FakeOCR
    monkeypatch.setattr(rag, "_ocrmac_module", _FakeModule)

    t1 = rag._ocr_image(img)
    assert "first" in t1

    # Modificamos la imagen — mtime cambia.
    import os, time
    time.sleep(0.05)
    img.write_bytes(b"v2 newer")
    os.utime(img, None)   # bump mtime explícitamente

    t2 = rag._ocr_image(img)
    assert "second" in t2
    assert call_count["n"] == 2


def test_ocr_image_returns_empty_when_module_missing(_clean_ocr_cache, tmp_path, monkeypatch):
    """Sin `ocrmac` instalado (Linux / no-ocr setup), retorna "" sin crash."""
    monkeypatch.setattr(rag, "_ocrmac_module", None)
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    out = rag._ocr_image(img)
    assert out == ""


def test_ocr_image_returns_empty_when_ocr_raises(_clean_ocr_cache, tmp_path, monkeypatch):
    """Si ocrmac crashea (imagen corrupta, formato raro), silent-fail → "". """
    class _BadOCR:
        def __init__(self, path, language_preference=None):
            pass
        def recognize(self_inner):
            raise RuntimeError("mock ocr failure")

    class _FakeModule:
        OCR = _BadOCR
    monkeypatch.setattr(rag, "_ocrmac_module", _FakeModule)
    img = tmp_path / "bad.png"
    img.write_bytes(b"fake")
    out = rag._ocr_image(img)
    assert out == ""


def test_ocr_image_disabled_by_env_var(_clean_ocr_cache, tmp_path, monkeypatch):
    """`RAG_OCR=0` desactiva completamente — retorna "" sin tocar ocrmac."""
    class _ShouldNotBeCalled:
        def __init__(self, *a, **kw):
            raise AssertionError("ocrmac no debería instanciarse con RAG_OCR=0")

    class _FakeModule:
        OCR = _ShouldNotBeCalled
    monkeypatch.setattr(rag, "_ocrmac_module", _FakeModule)
    monkeypatch.setenv("RAG_OCR", "0")

    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    out = rag._ocr_image(img)
    assert out == ""


# ── _enrich_body_with_ocr: orchestration ─────────────────────────────────────


def test_enrich_body_no_images_returns_body_unchanged(tmp_path, monkeypatch):
    """Body sin embeds → body intacto, cero llamadas a _ocr_image."""
    vault = tmp_path / "v"; vault.mkdir()
    note = vault / "plain.md"
    note.write_text("solo texto sin imágenes", encoding="utf-8")

    call_count = {"n": 0}
    def spy(*a, **kw):
        call_count["n"] += 1
        return ""
    monkeypatch.setattr(rag, "_ocr_image", spy)

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert out == note.read_text()
    assert call_count["n"] == 0


def test_enrich_body_with_one_image_appends_ocr_text(tmp_path, monkeypatch):
    """Body con 1 embed → body + OCR marker + texto OCR."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "captura.png"
    img.write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text("# Dev Cycles\n\n![[captura.png]]", encoding="utf-8")

    monkeypatch.setattr(
        rag, "_ocr_image",
        lambda p: "Start Date 03-Jul-2025 End Date 16-Jul-2025 Cycle 10.54",
    )
    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)

    # Original content preservado.
    assert "# Dev Cycles" in out
    assert "![[captura.png]]" in out
    # OCR text concatenado.
    assert "Start Date" in out
    assert "10.54" in out
    # Marker identificando la imagen (user-grep-friendly).
    assert "captura.png" in out


def test_enrich_body_empty_ocr_does_not_pollute(tmp_path, monkeypatch):
    """Si OCR devuelve "" para una imagen, NO se agrega marker vacío."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "blank.png"
    img.write_bytes(b"fake")
    note = vault / "note.md"
    original = "# Algo\n\n![[blank.png]]"
    note.write_text(original, encoding="utf-8")

    monkeypatch.setattr(rag, "_ocr_image", lambda p: "")
    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    # Body unchanged — no marker ni placeholder.
    assert out == original


def test_enrich_body_multiple_images(tmp_path, monkeypatch):
    """Múltiples imágenes → cada una con su propio marker + texto."""
    vault = tmp_path / "v"; vault.mkdir()
    (vault / "a.png").write_bytes(b"fake")
    (vault / "b.png").write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text("![[a.png]]\n\n![[b.png]]", encoding="utf-8")

    texts = iter(["texto de A", "texto de B"])
    monkeypatch.setattr(rag, "_ocr_image", lambda p: next(texts))

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert "a.png" in out and "texto de A" in out
    assert "b.png" in out and "texto de B" in out


# ── Integration: _index_single_file llama _enrich_body_with_ocr ──────────────


def test_index_single_file_includes_ocr_text_in_chunks(tmp_path, monkeypatch):
    """End-to-end: `_index_single_file` con una nota que embed una imagen
    debe producir chunks cuyo display_text incluye el texto OCR."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "embed", lambda ts: [[1.0, 0.0, 0.0, 0.0] for _ in ts])
    # Skippear el cross-encoder + contextual summary para que el test sea rápido.
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])

    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "captura.png"
    img.write_bytes(b"fake")
    note = vault / "dev cycles.md"
    note.write_text("# Dev Cycles\n\n![[captura.png]]", encoding="utf-8")

    ocr_text = "Start Date 03-Jul-2025 Cycle 10.54 2025 Q3"
    monkeypatch.setattr(rag, "_ocr_image", lambda p: ocr_text)

    col = rag.get_db_for(vault)
    status = rag._index_single_file(col, note, skip_contradict=True, vault_path=vault)
    assert status == "indexed"

    # Recuperar todos los chunks de esta nota y verificar que al menos uno
    # contiene el texto OCR.
    got = col.get(where={"file": "dev cycles.md"}, include=["documents"])
    all_docs = " ".join(got["documents"])
    assert "10.54" in all_docs, (
        f"OCR text no fue incluido en los chunks indexados. "
        f"Docs: {got['documents']!r}"
    )
    assert "03-Jul-2025" in all_docs
