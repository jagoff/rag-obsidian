"""Tests del RetrieveResult dataclass con retrocompat dict (2026-04-22).

Contexto: pre-fix, `retrieve()` y `multi_retrieve()` devolvían un `dict`
genérico con ~13 claves opcionales. Cada uno de los ~196 call sites en
rag.py + web/server.py accedía al dict con un mix de `result["x"]`
(asume presencia, crashea si falta) y `result.get("x", default)`
(defensivo, oculta bugs).

Evidencia del problema:
  - Hoy agregamos `intent` al dict de retorno — tuvimos que tocar 4
    returns distintos + el empty-early-return. Sin tipo, no hay check
    automático de que estén todos.
  - `result.get("intent")` con default None porque antes crasheaba.
  - `multi_retrieve` devolvía un shape diferente (agrega `vault_scope`,
    faltan `graph_docs` en empty path) — el web se enteraba a los golpes.

Fix estructural: dataclass con todos los campos + métodos `__getitem__`
y `get()` para retrocompat 100%. Los call sites legacy siguen
funcionando palabra por palabra. Nuevos call sites pueden usar
`result.intent` / `result.timing` con type-safety.

Contrato:

  @dataclass
  class RetrieveResult:
      docs: list[str]
      metas: list[dict]
      scores: list[float]
      confidence: float
      search_query: str = ""
      filters_applied: dict = field(default_factory=dict)
      query_variants: list[str] = field(default_factory=list)
      timing: dict[str, float] = field(default_factory=dict)
      fast_path: bool = False
      graph_docs: list[str] = field(default_factory=list)
      graph_metas: list[dict] = field(default_factory=list)
      extras: dict = field(default_factory=dict)
      intent: str | None = None
      vault_scope: list[str] = field(default_factory=list)

      # Retrocompat: legacy dict access
      def __getitem__(self, key): ...
      def get(self, key, default=None): ...
      def __contains__(self, key): ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import rag


# ── Existencia + shape ──────────────────────────────────────────────────────


def test_retrieve_result_class_exists():
    assert hasattr(rag, "RetrieveResult")


def test_retrieve_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(rag.RetrieveResult)


def test_retrieve_result_has_required_fields():
    """Los 4 campos obligatorios (sin default) son los que SIEMPRE
    vienen: docs, metas, scores, confidence."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(rag.RetrieveResult)}
    required = {"docs", "metas", "scores", "confidence"}
    assert required.issubset(fields)


def test_retrieve_result_has_optional_fields():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(rag.RetrieveResult)}
    optional = {
        "search_query", "filters_applied", "query_variants", "timing",
        "fast_path", "graph_docs", "graph_metas", "extras", "intent",
        "vault_scope",
    }
    assert optional.issubset(fields), f"missing optional fields: {optional - fields}"


# ── Construcción + atributos ────────────────────────────────────────────────


def test_construct_minimal():
    """Mínimo viable: los 4 obligatorios, el resto con defaults."""
    r = rag.RetrieveResult(
        docs=["doc1"], metas=[{"file": "a.md"}],
        scores=[0.8], confidence=0.8,
    )
    assert r.docs == ["doc1"]
    assert r.metas == [{"file": "a.md"}]
    assert r.scores == [0.8]
    assert r.confidence == 0.8
    # Defaults
    assert r.intent is None
    assert r.fast_path is False
    assert r.timing == {}
    assert r.graph_docs == []
    assert r.extras == {}


def test_construct_with_all_fields():
    r = rag.RetrieveResult(
        docs=["d"], metas=[{}], scores=[0.9], confidence=0.9,
        search_query="test", intent="synthesis",
        timing={"total_ms": 150.5},
        vault_scope=["home", "work"],
        fast_path=True,
    )
    assert r.intent == "synthesis"
    assert r.timing["total_ms"] == 150.5
    assert r.vault_scope == ["home", "work"]
    assert r.fast_path is True


# ── Retrocompat: dict access ────────────────────────────────────────────────


def test_getitem_works_for_all_fields():
    """Legacy call sites que hacen `result["docs"]` deben seguir
    funcionando sin tocar."""
    r = rag.RetrieveResult(
        docs=["x"], metas=[{"file": "a.md"}], scores=[0.7], confidence=0.7,
        intent="list",
    )
    assert r["docs"] == ["x"]
    assert r["metas"] == [{"file": "a.md"}]
    assert r["scores"] == [0.7]
    assert r["confidence"] == 0.7
    assert r["intent"] == "list"


def test_getitem_raises_keyerror_on_unknown():
    """`result["nonexistent"]` debe raisear KeyError como un dict real."""
    r = rag.RetrieveResult(docs=[], metas=[], scores=[], confidence=-1.0)
    with pytest.raises(KeyError):
        _ = r["nonexistent_field"]


def test_get_with_default():
    """`result.get("x", default)` funciona para campos presentes y ausentes."""
    r = rag.RetrieveResult(
        docs=["x"], metas=[{}], scores=[0.5], confidence=0.5,
    )
    assert r.get("docs") == ["x"]
    assert r.get("intent") is None  # field default
    assert r.get("nonexistent") is None  # nonexistent → None
    assert r.get("nonexistent", "fallback") == "fallback"


def test_contains_works():
    r = rag.RetrieveResult(docs=[], metas=[], scores=[], confidence=-1.0)
    assert "docs" in r
    assert "intent" in r  # tiene default → es atributo
    assert "nonexistent_field" not in r


# ── retrieve() devuelve RetrieveResult ──────────────────────────────────────


def test_retrieve_returns_retrieve_result_on_empty_corpus():
    """Corpus vacío (col.count()==0) ahora devuelve RetrieveResult,
    no dict. Legacy callers siguen funcionando via __getitem__ / get()."""
    class _EmptyCol:
        def count(self): return 0
    result = rag.retrieve(_EmptyCol(), "test", k=3, folder=None, intent="count")
    assert isinstance(result, rag.RetrieveResult)
    # Retrocompat
    assert result["docs"] == []
    assert result["metas"] == []
    assert result.get("intent") == "count"
    # Nuevo atributo access
    assert result.intent == "count"
    assert result.fast_path is False


# ── multi_retrieve() devuelve RetrieveResult ────────────────────────────────


def test_multi_retrieve_empty_vaults_returns_retrieve_result():
    result = rag.multi_retrieve([], "q", k=3, folder=None, intent="synthesis")
    assert isinstance(result, rag.RetrieveResult)
    assert result.intent == "synthesis"
    assert result["docs"] == []
    assert result.vault_scope == []


def test_multi_retrieve_single_vault_returns_retrieve_result(tmp_path, monkeypatch):
    class _EmptyCol:
        def count(self): return 0
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _EmptyCol())
    result = rag.multi_retrieve(
        [("home", tmp_path)], "q", k=3, folder=None, intent="comparison",
    )
    assert isinstance(result, rag.RetrieveResult)
    assert result.intent == "comparison"
    assert "home" in result.vault_scope


# ── Iteración / keys → detecta si algo quedó sin migrar ─────────────────────


def test_keys_returns_all_field_names():
    """Para debugging: poder hacer `for k in result.keys()` como un dict."""
    r = rag.RetrieveResult(docs=[], metas=[], scores=[], confidence=-1.0)
    keys = set(r.keys())
    # Al menos los requireds + opcionales más usados
    assert "docs" in keys
    assert "metas" in keys
    assert "scores" in keys
    assert "confidence" in keys
    assert "intent" in keys
    assert "timing" in keys


# ── equality / repr son bien-definidos ──────────────────────────────────────


def test_equality_via_dataclass_default():
    r1 = rag.RetrieveResult(docs=["x"], metas=[{}], scores=[0.5], confidence=0.5)
    r2 = rag.RetrieveResult(docs=["x"], metas=[{}], scores=[0.5], confidence=0.5)
    assert r1 == r2
    r3 = rag.RetrieveResult(docs=["y"], metas=[{}], scores=[0.5], confidence=0.5)
    assert r1 != r3


def test_repr_is_readable():
    r = rag.RetrieveResult(
        docs=["x"], metas=[{}], scores=[0.5], confidence=0.5, intent="semantic",
    )
    s = repr(r)
    assert "RetrieveResult" in s
    assert "intent='semantic'" in s or "intent=\"semantic\"" in s


# ── Round-trip: construir desde dict (migración gradual) ────────────────────


def test_from_dict_for_migration():
    """Helper `RetrieveResult.from_dict(d)` para migración gradual de
    call sites que hoy construyen dicts a mano (tests legacy, fixtures)."""
    d = {
        "docs": ["doc1"], "metas": [{"file": "a.md"}],
        "scores": [0.8], "confidence": 0.8,
        "intent": "list", "fast_path": True,
    }
    r = rag.RetrieveResult.from_dict(d)
    assert r.docs == ["doc1"]
    assert r.intent == "list"
    assert r.fast_path is True


def test_from_dict_ignores_unknown_keys():
    """Tolerancia a dicts legacy con campos que ya no existen."""
    d = {
        "docs": [], "metas": [], "scores": [], "confidence": -1.0,
        "legacy_removed_field": "ignored",
    }
    r = rag.RetrieveResult.from_dict(d)
    assert r.docs == []
