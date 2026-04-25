"""Regression tests para los fixes del audit 2026-04-24 en `rag.py`:

- `_load_corpus` TOCTOU con `_invalidate_corpus_cache`: pre-fix el check
  `cached["count"] == n` ocurría FUERA del lock, así que otro thread
  podía invalidar el cache entre el release del lock y el check, y
  terminábamos devolviendo data stale como "fresca".
- `parse_frontmatter` silent YAML error: pre-fix un `---\ntags: [bad\n---`
  causaba que el indexer ignorara TODO el frontmatter sin avisar. Ahora
  logea via `_silent_log` para observabilidad.

Los tests de `save_session` + flock viven en `tests/test_sessions.py`.
Los del validator de web/server.py viven en `tests/test_web_edge_cases.py`.
"""
from __future__ import annotations

import threading


import rag


# ── `_load_corpus` TOCTOU fix ───────────────────────────────────────────────


class _FakeCol:
    """Fake chroma collection suficiente para `_load_corpus`."""
    def __init__(self, docs: list[str], metas: list[dict], ids: list[str],
                 col_id: str = "col-1"):
        self._docs = docs
        self._metas = metas
        self._ids = ids
        self.id = col_id

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None):
        return {
            "documents": list(self._docs),
            "metadatas": list(self._metas),
            "ids": list(self._ids),
        }


def test_load_corpus_returns_cached_on_cache_hit(monkeypatch):
    """Happy path: segundo llamado con la misma collection devuelve el
    cached sin re-querying."""
    # Reset global state.
    monkeypatch.setattr(rag, "_corpus_cache", None)

    col = _FakeCol(
        docs=["a", "b"],
        metas=[{"file": "A.md", "note": "A", "folder": "01"},
               {"file": "B.md", "note": "B", "folder": "02"}],
        ids=["id-a", "id-b"],
    )
    first = rag._load_corpus(col)
    second = rag._load_corpus(col)
    assert first is second, "segundo call debería devolver la misma ref (cache hit)"


def test_load_corpus_no_toctou_under_concurrent_invalidation(monkeypatch):
    """Iter 2026-04-24: el check `cached["count"] == n` debe estar DENTRO
    del lock. Pre-fix, un thread invalidador podía correr entre el release
    del lock y el check, resultando en el retorno de un dict "cached" que
    otro thread ya había marcado como stale.

    Este test no valida el fix de forma determinística (la race es sub-ms
    y difícil de forzar) pero ejerce el path concurrente para garantizar
    que no crashea bajo alta contención.
    """
    monkeypatch.setattr(rag, "_corpus_cache", None)

    col = _FakeCol(
        docs=["a"], metas=[{"file": "A.md"}], ids=["id-a"]
    )
    stop = threading.Event()
    errors: list[Exception] = []

    def _reader():
        while not stop.is_set():
            try:
                rag._load_corpus(col)
            except Exception as exc:
                errors.append(exc)
                return

    def _invalidator():
        while not stop.is_set():
            try:
                rag._invalidate_corpus_cache()
            except Exception as exc:
                errors.append(exc)
                return

    readers = [threading.Thread(target=_reader) for _ in range(4)]
    invalidators = [threading.Thread(target=_invalidator) for _ in range(2)]
    all_threads = readers + invalidators
    for t in all_threads:
        t.start()
    # Ejerce la race por ~200ms — suficiente para miles de iteraciones.
    import time
    time.sleep(0.2)
    stop.set()
    for t in all_threads:
        t.join(timeout=1.0)

    assert not errors, (
        f"threads concurrentes (readers + invalidators) lanzaron errores "
        f"a pesar del lock: {errors[:3]}"
    )


def test_load_corpus_respects_invalidation(monkeypatch):
    """Después de `_invalidate_corpus_cache()`, el próximo `_load_corpus`
    debería re-queryear (NO devolver el dict viejo que estaba cacheado)."""
    monkeypatch.setattr(rag, "_corpus_cache", None)

    col = _FakeCol(docs=["a"], metas=[{"file": "A.md"}], ids=["id-a"])
    first = rag._load_corpus(col)
    rag._invalidate_corpus_cache()
    second = rag._load_corpus(col)
    # Son dicts distintos porque se recomputó (aunque el contenido sea similar).
    assert first is not second, (
        "post-invalidate, `_load_corpus` debería re-queryear y devolver "
        "un dict nuevo, no el cached de antes"
    )


# ── `parse_frontmatter` log silent YAML ────────────────────────────────────


def test_parse_frontmatter_logs_malformed_yaml(monkeypatch):
    """Pre-fix (2026-04-24 audit): un frontmatter con YAML inválido se
    swallowed silenciosamente (`return {}` sin log). El indexer indexaba
    la nota sin tags/aliases y nadie se enteraba. Post-fix: se logea
    via `_silent_log` para que quede trace en silent_errors.jsonl.
    """
    logged: list[tuple[str, Exception]] = []
    monkeypatch.setattr(
        rag, "_silent_log",
        lambda tag, exc: logged.append((tag, exc)),
    )

    malformed = "---\ntags: [unclosed\nbody\n---\nBody content\n"
    result = rag.parse_frontmatter(malformed)

    # Contrato existente: devuelve dict vacío, no raise.
    assert result == {}
    # Nuevo: queda trace del error con el tag esperado.
    assert any(tag == "parse_frontmatter_yaml" for tag, _ in logged), (
        f"esperaba log con tag 'parse_frontmatter_yaml', got: {logged}"
    )


def test_parse_frontmatter_valid_yaml_no_log(monkeypatch):
    """No-regresión: YAML válido NO debe loggear (el `_silent_log` es
    solo para errores, no para parse exitoso)."""
    logged: list[tuple[str, Exception]] = []
    monkeypatch.setattr(
        rag, "_silent_log",
        lambda tag, exc: logged.append((tag, exc)),
    )

    valid = "---\ntags:\n  - a\n  - b\narea: project\n---\nBody\n"
    result = rag.parse_frontmatter(valid)

    assert result == {"tags": ["a", "b"], "area": "project"}
    assert logged == [], (
        f"YAML válido NO debería loggear errores, got: {logged}"
    )


def test_parse_frontmatter_no_frontmatter_no_log(monkeypatch):
    """No-regresión: body sin `---` al inicio NO debe loggear (es válido
    no tener frontmatter, no es un error)."""
    logged: list[tuple[str, Exception]] = []
    monkeypatch.setattr(
        rag, "_silent_log",
        lambda tag, exc: logged.append((tag, exc)),
    )

    body_only = "Solo body, sin frontmatter.\n"
    result = rag.parse_frontmatter(body_only)

    assert result == {}
    assert logged == []
