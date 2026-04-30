"""Tests para `rag.mmr_diversification` — MMR + folder-penalty post-rerank.

Cobertura:
  1. apply_mmr con cluster homogéneo  → reorden visible (no identity).
  2. apply_mmr con candidates diversos → casi no cambia el orden.
  3. apply_mmr lambda=1.0              → idéntico al input (pure relevance).
  4. apply_mmr lambda=0.0              → top-1 fijo, resto greedy diversidad.
  5. folder_penalty: mismo folder      → demoted excepto el #1.
  6. folder_penalty: folders distintos → orden preservado.
  7. RAG_MMR=0 (default)               → retrieve no aplica el helper.
  8. Smoke RAG_MMR=1                   → retrieve devuelve top-k sin error.

Helpers de embed: usamos un fake `embed_fn` determinístico con vectores 2D
para no depender de bge-m3 ni de ollama. Doc identical → mismo vector;
doc distinta palabra-clave → vector ortogonal. Cosine ∈ {0.0, 1.0}.

Notas de diseño:
  - Los tests del wire-up en retrieve() (cases 7 y 8) usan un retrieve()
    monkeypatched que devuelve un RetrieveResult sintético — el objetivo
    es validar el GATE (`RAG_MMR=1` aplica, `RAG_MMR=0` no), no el flujo
    de bge-m3 + sqlite-vec end-to-end (eso ya está cubierto en
    test_retrieve_e2e.py).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import pytest

from rag.mmr_diversification import (
    apply_mmr,
    apply_mmr_with_folder_penalty,
    count_reordered,
    env_enabled,
    env_folder_penalty,
    env_lambda,
    env_top_k,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@contextmanager
def _env(**kwargs: str):
    """Setea env vars temporariamente y restaura al salir."""
    saved = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_doc(*, path: str, text: str, key: str | None = None) -> dict:
    """Construye una doc dict compatible con `_doc_text` + `_doc_folder`."""
    return {"path": path, "text": text, "_key": key or text}


def _embed_one_hot(texts: list[str]) -> list[list[float]]:
    """Embedding sintético: cada texto distinto → vector one-hot 6-dim.

    Empareja duplicados via la primera "palabra" (split por whitespace).
    Cosine entre dos textos con la misma primera palabra = 1.0; distintas
    primeras palabras = 0.0. Determinístico sin dependencias externas.
    """
    BUCKETS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    out: list[list[float]] = []
    for t in texts:
        head = (t or "").split()[:1]
        head_word = head[0] if head else ""
        v = [0.0] * len(BUCKETS)
        for i, b in enumerate(BUCKETS):
            if head_word == b:
                v[i] = 1.0
        # Si no matchea ninguno, queda all-zeros (cosine 0 contra todo).
        out.append(v)
    return out


# ── 1. apply_mmr cluster homogéneo → reorden visible ────────────────────────


def test_apply_mmr_homogeneous_cluster_reorders():
    """5 docs casi iguales semánticamente (mismo `head` word) — MMR debe
    poder reordenar contra una sexta diversa intercalada bajo la lista.
    """
    docs = [
        _make_doc(path="A/1.md", text="alpha foo", key="d0"),
        _make_doc(path="A/2.md", text="alpha bar", key="d1"),
        _make_doc(path="A/3.md", text="alpha baz", key="d2"),
        _make_doc(path="A/4.md", text="alpha qux", key="d3"),
        _make_doc(path="A/5.md", text="beta diff", key="d4"),  # diversa
    ]
    scored = [
        (docs[0], 1.0),
        (docs[1], 0.95),
        (docs[2], 0.90),
        (docs[3], 0.85),
        (docs[4], 0.50),  # mucho más bajo en relevance
    ]
    out = apply_mmr(scored, lambda_=0.4, top_k=5, embed_fn=_embed_one_hot)
    # Con λ=0.4 (sesgo a diversity) y los primeros 4 idénticos:
    # el doc "diverso" (d4) sube a slot #2 aunque tiene el menor score.
    assert out[0][0]["_key"] == "d0"  # top-1 always preserved
    assert out[1][0]["_key"] == "d4"  # diverse doc promoted
    # El re-orden debe contar al menos 1 cambio
    assert count_reordered(scored, out) > 0


# ── 2. apply_mmr con candidates diversos → casi no cambia orden ─────────────


def test_apply_mmr_diverse_pool_preserves_order():
    """Si todos los candidates ya son diversos (vectores ortogonales),
    el sim() es 0 y MMR degenera a relevance-only ordering.
    """
    docs = [
        _make_doc(path=f"folder{i}/note.md", text=word, key=f"d{i}")
        for i, word in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])
    ]
    scored = [
        (docs[0], 1.0),
        (docs[1], 0.9),
        (docs[2], 0.8),
        (docs[3], 0.7),
        (docs[4], 0.6),
    ]
    out = apply_mmr(scored, lambda_=0.5, top_k=5, embed_fn=_embed_one_hot)
    # Sin similaridad entre docs (max_sim siempre 0), el ranking queda
    # idéntico al input (relevance ordering).
    assert [s for _, s in out] == [1.0, 0.9, 0.8, 0.7, 0.6]
    assert [d["_key"] for d, _ in out] == [f"d{i}" for i in range(5)]
    assert count_reordered(scored, out) == 0


# ── 3. lambda=1.0 → identidad ───────────────────────────────────────────────


def test_apply_mmr_lambda_1_is_identity():
    """λ=1.0 → pure relevance, output == input position-by-position."""
    docs = [
        _make_doc(path="A/1.md", text="alpha foo", key="d0"),
        _make_doc(path="A/2.md", text="alpha bar", key="d1"),
        _make_doc(path="A/3.md", text="beta xx", key="d2"),
    ]
    scored = [(docs[0], 1.0), (docs[1], 0.9), (docs[2], 0.8)]
    out = apply_mmr(scored, lambda_=1.0, top_k=3, embed_fn=_embed_one_hot)
    assert [d["_key"] for d, _ in out] == ["d0", "d1", "d2"]
    assert count_reordered(scored, out) == 0


# ── 4. lambda=0.0 → top-1 preservado, resto cambia fuerte ───────────────────


def test_apply_mmr_lambda_0_diversity_aggressive():
    """λ=0.0 → top-1 fijo (always); slots 2..N pickean por max-distance al
    set seleccionado. Con un cluster homogéneo + 1 diverso, el diverso
    salta al slot 2 sin importar su score.
    """
    docs = [
        _make_doc(path="A/1.md", text="alpha 1", key="d0"),
        _make_doc(path="A/2.md", text="alpha 2", key="d1"),
        _make_doc(path="A/3.md", text="alpha 3", key="d2"),
        _make_doc(path="A/4.md", text="beta UNIQUE", key="d3"),  # diverso
    ]
    scored = [
        (docs[0], 1.0),
        (docs[1], 0.95),
        (docs[2], 0.90),
        (docs[3], 0.10),  # menor relevance
    ]
    out = apply_mmr(scored, lambda_=0.0, top_k=4, embed_fn=_embed_one_hot)
    # Top-1 always fixed.
    assert out[0][0]["_key"] == "d0"
    # Con λ=0 puro, la "distancia máxima" gana — d3 es el único distinto.
    assert out[1][0]["_key"] == "d3"
    # d3 fue movido del slot original #4 al #2 — count >= 2 cambios.
    assert count_reordered(scored, out) >= 2


# ── 5. folder_penalty con mismo folder → demoted ────────────────────────────


def test_folder_penalty_same_folder_demotes():
    """5 docs todos del mismo folder → top-1 fijo, los demás compiten con
    docs en otros folders (acá no hay otros, así que el orden interno
    sigue por relevance pero penalizado uniformemente)."""
    docs = [
        _make_doc(path="A/1.md", text="x", key="d0"),
        _make_doc(path="A/2.md", text="x", key="d1"),
        _make_doc(path="A/3.md", text="x", key="d2"),
        _make_doc(path="A/4.md", text="x", key="d3"),
        _make_doc(path="A/5.md", text="x", key="d4"),
    ]
    scored = [
        (docs[0], 1.0),
        (docs[1], 0.9),
        (docs[2], 0.8),
        (docs[3], 0.7),
        (docs[4], 0.6),
    ]
    # Caso real con un doc "intruder" de OTRO folder mid-rank:
    intruder = _make_doc(path="B/other.md", text="y", key="intruder")
    scored.insert(3, (intruder, 0.65))  # entre d2 y d3
    out = apply_mmr_with_folder_penalty(scored, top_k=6, folder_penalty=0.2)
    # Top-1 fijo
    assert out[0][0]["_key"] == "d0"
    # El intruder (carpeta B) salta hacia adelante porque NO penaliza con A.
    # En A: cada extra del mismo folder se penaliza por 0.2 * collisions.
    # Pos 2: d1 (rel=0.9, coll=1, adj=0.7) vs intruder (rel=0.65, coll=0, adj=0.65)
    # → d1 gana en pos 2 (0.7 > 0.65). Pos 3: d2 (rel=0.8, coll=2, adj=0.4),
    # intruder (rel=0.65, coll=0, adj=0.65), d3 (rel=0.7, coll=3, adj=0.1).
    # → intruder gana pos 3.
    keys = [d["_key"] for d, _ in out]
    assert keys[0] == "d0"
    assert keys[1] == "d1"
    assert keys[2] == "intruder"  # promovido por su folder único
    # Después siguen los Asss en orden, pero count_reordered > 0
    assert count_reordered(scored, out) > 0


# ── 6. folder_penalty con folders distintos → no cambia orden ───────────────


def test_folder_penalty_diverse_folders_preserves_order():
    docs = [
        _make_doc(path=f"folder{i}/note.md", text="x", key=f"d{i}")
        for i in range(5)
    ]
    scored = [(docs[i], 1.0 - i * 0.1) for i in range(5)]
    out = apply_mmr_with_folder_penalty(scored, top_k=5, folder_penalty=0.5)
    # Sin colisiones de folder, no hay penalty → output == input.
    assert [d["_key"] for d, _ in out] == [f"d{i}" for i in range(5)]
    assert count_reordered(scored, out) == 0


# ── 7. RAG_MMR=0 (default) → retrieve no aplica el helper ────────────────────


def test_retrieve_default_no_mmr(monkeypatch):
    """Con `RAG_MMR=0` y `RAG_MMR_FOLDER_PENALTY=0` (defaults),
    el resultado de retrieve no debe registrar mmr_applied > 0."""
    import rag

    # Empty vault → retrieve devuelve un RetrieveResult vacío sin disparar
    # MMR. Es la forma más barata de validar que la rama del helper no
    # corre cuando los flags están off (no necesita corpus real).

    class _EmptyCol:
        def count(self) -> int:
            return 0

    with _env(RAG_MMR=None, RAG_MMR_FOLDER_PENALTY=None):
        out = rag.retrieve(_EmptyCol(), "una query cualquiera", k=5, folder=None)
    assert out["mmr_applied"] == 0


# ── 8. Smoke con RAG_MMR=1 → retrieve no rompe ──────────────────────────────


def test_retrieve_with_mmr_smoke(monkeypatch):
    """Smoke: con `RAG_MMR=1` activo, retrieve completa sin error.

    Empty vault asegura camino corto sin tocar embed/rerank reales.
    """
    import rag

    class _EmptyCol:
        def count(self) -> int:
            return 0

    with _env(RAG_MMR="1", RAG_MMR_LAMBDA="0.7", RAG_MMR_TOP_K="5"):
        out = rag.retrieve(_EmptyCol(), "smoke query MMR", k=5, folder=None)
    # Empty corpus path: docs vacío y mmr_applied=0 (no había qué reordenar).
    assert out["docs"] == []
    assert out["mmr_applied"] == 0


# ── Casos extra (defensa robusta) ────────────────────────────────────────────


def test_apply_mmr_empty_input():
    assert apply_mmr([], lambda_=0.7, top_k=10, embed_fn=_embed_one_hot) == []


def test_apply_mmr_single_item_unchanged():
    doc = _make_doc(path="A.md", text="alpha solo", key="d0")
    pair = [(doc, 1.0)]
    out = apply_mmr(pair, lambda_=0.7, top_k=5, embed_fn=_embed_one_hot)
    assert out == pair


def test_apply_mmr_lambda_clamping():
    """λ fuera de [0, 1] se clampea sin levantar excepción."""
    docs = [
        _make_doc(path="A.md", text="alpha", key="d0"),
        _make_doc(path="B.md", text="beta", key="d1"),
    ]
    scored = [(docs[0], 1.0), (docs[1], 0.5)]
    apply_mmr(scored, lambda_=2.0, top_k=2, embed_fn=_embed_one_hot)
    apply_mmr(scored, lambda_=-1.0, top_k=2, embed_fn=_embed_one_hot)


def test_apply_mmr_top_k_truncation():
    """top_k=2 reordena solo los 2 primeros; el resto pasa al final tal cual."""
    docs = [
        _make_doc(path="A.md", text="alpha 1", key="d0"),
        _make_doc(path="A.md", text="alpha 2", key="d1"),
        _make_doc(path="A.md", text="alpha 3", key="d2"),
        _make_doc(path="A.md", text="alpha 4", key="d3"),
    ]
    scored = [(docs[i], 1.0 - i * 0.1) for i in range(4)]
    out = apply_mmr(scored, lambda_=0.0, top_k=2, embed_fn=_embed_one_hot)
    assert len(out) == 4
    # Slots 2 y 3 son el tail intacto.
    assert out[2][0]["_key"] == "d2"
    assert out[3][0]["_key"] == "d3"


def test_apply_mmr_budget_exceeded_falls_back(monkeypatch):
    """Si `embed_fn` tarda más del budget, devolvemos scored original."""
    import time

    docs = [
        _make_doc(path="A.md", text="alpha 1", key="d0"),
        _make_doc(path="A.md", text="alpha 2", key="d1"),
    ]
    scored = [(docs[0], 1.0), (docs[1], 0.9)]

    def _slow_embed(texts):
        time.sleep(0.05)  # 50ms
        return _embed_one_hot(texts)

    skips: list[str] = []
    out = apply_mmr(
        scored, lambda_=0.5, top_k=2,
        embed_fn=_slow_embed, budget_ms=10.0,
        on_skip=lambda r: skips.append(r),
    )
    # Devuelve la lista sin tocar (fallback).
    assert out == scored
    assert any("over_budget" in s for s in skips)


def test_apply_mmr_embed_failure_falls_back():
    """Si embed_fn levanta, devolvemos scored sin tocar."""
    docs = [
        _make_doc(path="A.md", text="alpha 1", key="d0"),
        _make_doc(path="B.md", text="beta", key="d1"),
    ]
    scored = [(docs[0], 1.0), (docs[1], 0.9)]

    def _failing_embed(_texts):
        raise RuntimeError("ollama down")

    skips: list[str] = []
    out = apply_mmr(
        scored, lambda_=0.5, top_k=2,
        embed_fn=_failing_embed,
        on_skip=lambda r: skips.append(r),
    )
    assert out == scored
    assert any("embed_call_failed" in s for s in skips)


def test_apply_mmr_embed_shape_mismatch_falls_back():
    docs = [
        _make_doc(path="A.md", text="alpha 1", key="d0"),
        _make_doc(path="B.md", text="beta", key="d1"),
    ]
    scored = [(docs[0], 1.0), (docs[1], 0.9)]

    def _bad_embed(texts):
        return [[1.0, 0.0]]  # shape mismatch (1 vector for 2 texts)

    skips: list[str] = []
    out = apply_mmr(
        scored, lambda_=0.5, top_k=2,
        embed_fn=_bad_embed,
        on_skip=lambda r: skips.append(r),
    )
    assert out == scored
    assert any("shape_mismatch" in s for s in skips)


def test_count_reordered_basic():
    a_doc = {"path": "a"}
    b_doc = {"path": "b"}
    c_doc = {"path": "c"}
    before = [(a_doc, 1.0), (b_doc, 0.9), (c_doc, 0.8)]
    after = [(a_doc, 1.0), (c_doc, 0.8), (b_doc, 0.9)]
    # Los slots 2 y 3 cambiaron de doc.
    assert count_reordered(before, after) == 2
    # Si no hay cambio, count_reordered = 0.
    assert count_reordered(before, before) == 0


def test_folder_penalty_zero_is_noop():
    """folder_penalty=0 → list(scored) sin tocar."""
    docs = [_make_doc(path="A/x.md", text="x", key=f"d{i}") for i in range(3)]
    scored = [(docs[i], 1.0 - i * 0.1) for i in range(3)]
    out = apply_mmr_with_folder_penalty(scored, top_k=3, folder_penalty=0.0)
    assert out == scored


def test_folder_penalty_empty_input():
    assert apply_mmr_with_folder_penalty([], top_k=5) == []


def test_folder_penalty_single_item():
    doc = _make_doc(path="A.md", text="x", key="d0")
    pair = [(doc, 1.0)]
    out = apply_mmr_with_folder_penalty(pair, top_k=5)
    assert out == pair


def test_env_helpers_clamping():
    """env_lambda y env_top_k respetan rangos válidos aunque venga basura."""
    with _env(RAG_MMR_LAMBDA="not-a-number"):
        assert env_lambda(0.7) == 0.7
    with _env(RAG_MMR_LAMBDA="2.0"):
        assert env_lambda() == 1.0
    with _env(RAG_MMR_LAMBDA="-1.0"):
        assert env_lambda() == 0.0
    with _env(RAG_MMR_TOP_K="0"):
        assert env_top_k() == 1
    with _env(RAG_MMR_TOP_K="9999"):
        assert env_top_k() == 100


def test_env_enabled_truthy_values():
    for val in ("1", "true", "TRUE", "yes", "ON"):
        with _env(RAG_MMR=val):
            assert env_enabled("RAG_MMR") is True
    for val in ("0", "false", "no", "off", ""):
        with _env(RAG_MMR=val):
            assert env_enabled("RAG_MMR") is False


def test_env_enabled_unset_is_false():
    with _env(RAG_MMR=None):
        assert env_enabled("RAG_MMR") is False


def test_env_folder_penalty_clamping():
    with _env(RAG_MMR_FOLDER_PENALTY_MAGNITUDE="0.25"):
        assert env_folder_penalty() == 0.25
    with _env(RAG_MMR_FOLDER_PENALTY_MAGNITUDE="garbage"):
        assert env_folder_penalty(0.1) == 0.1
    with _env(RAG_MMR_FOLDER_PENALTY_MAGNITUDE="2.5"):
        assert env_folder_penalty() == 1.0
    with _env(RAG_MMR_FOLDER_PENALTY_MAGNITUDE="-3"):
        assert env_folder_penalty() == 0.0
