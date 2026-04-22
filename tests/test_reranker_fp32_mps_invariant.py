"""Protege la invariante del reranker documentada en CLAUDE.md:

    > Reranker: BAAI/bge-reranker-v2-m3
    > `device="mps"` + `float32` forced — do NOT switch to fp16 on MPS
    > (score collapse to ~0.001, verified 2026-04-13); CPU fallback = 3× slower.

La tentación obvia: "fp16 reduce memoria + es más rápido en MPS". Falso
para este modelo en particular — se midió el 2026-04-13 que los scores
colapsan a ~0.001 (orden de magnitud bajo el threshold
CONFIDENCE_RERANK_MIN=0.015, lo que haría que casi toda query sea
rechazada por el confidence gate sin que nadie se dé cuenta hasta que los
usuarios reporten "el RAG dejó de contestar").

Estos tests verifican, cada vez que la suite corre en una máquina con
MPS, que:
  (a) el modelo está efectivamente en `device="mps"`;
  (b) todos los parámetros siguen en `torch.float32` (no hay un cast a
      fp16 sneakeado en get_reranker() o en un init helper);
  (c) al menos un score real sobre un par (query, doc) razonable está
      "vivo" (> 0.01) — descarta el score-collapse observacionalmente, no
      solo por dtype.

Skips completos cuando no hay MPS (Linux/CI). El test slow corre el
modelo real (~5-10s cold load + 1 pair predict) y se marca con
`@pytest.mark.slow` para poder correrlo con `pytest -m "not slow"` en
feedback loops rápidos.
"""
from __future__ import annotations

import os

import pytest

# Igual que los otros invariant tests — mantener el watchdog silent.
os.environ.setdefault("RAG_MEMORY_PRESSURE_DISABLE", "1")


# Gate hard: importar torch sólo adentro del skipif para no pagar el import
# cuando no hay MPS (Linux CI).
def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mps_available(),
    reason="MPS-only invariant (Apple Silicon). Linux/CI/Intel → skip.",
)


def test_reranker_device_and_dtype_are_mps_fp32():
    """Carga el reranker real y verifica device + dtype sobre todos los
    parámetros. No corre forward — solo toca el struct. ~5s cold, ~50ms
    warm.

    NOTA: los `get_reranker()` tests en otros archivos monkeypatchean el
    objeto para no tocar MPS; este test explícitamente NO monkeypatchea —
    se quiere observar el objeto real para proteger la invariante.
    """
    import torch
    import rag

    r = rag.get_reranker()
    # sentence-transformers.CrossEncoder expone `.model` (HF module) y
    # también es iterable con `.parameters()`. Ser defensivo contra las
    # dos formas.
    model = getattr(r, "model", None) or r
    params = list(model.parameters())
    assert params, "reranker has no parameters — wrong object shape?"

    dtypes = {p.dtype for p in params}
    devices = {p.device.type for p in params}

    assert dtypes == {torch.float32}, (
        f"Reranker params not uniformly fp32: {dtypes}. "
        "CLAUDE.md line: 'do NOT switch to fp16 on MPS (score collapse to "
        "~0.001, verified 2026-04-13)'. Revisar get_reranker()."
    )
    assert devices == {"mps"}, (
        f"Reranker params not on MPS: {devices}. Auto-detect de "
        "sentence-transformers a veces cae en CPU (3× slower) en venvs "
        "sin MPS habilitado. get_reranker() fuerza device='mps' cuando "
        "está disponible — si fallar acá en Apple Silicon, algo roto."
    )


@pytest.mark.slow
def test_reranker_scores_not_collapsed_real_pair():
    """Corre `.predict()` real sobre un par (query, doc) razonable y
    confirma que el score NO colapsó al ~0.001 característico del modo
    fp16-roto. Umbral conservador: > 0.01 (el doc es literalmente la
    respuesta perfecta al query, el score sano ronda 0.9+).

    Marcado slow porque implica el cold-load de bge-reranker-v2-m3 + un
    forward pass en MPS (~5-10s). Correr con:
      .venv/bin/python -m pytest tests/test_reranker_fp32_mps_invariant.py \\
          -m slow -q
    """
    import rag

    r = rag.get_reranker()
    pairs = [("qué lenguaje es seguro y rápido", "Rust es un lenguaje seguro y rápido")]
    scores = r.predict(pairs, batch_size=1, show_progress_bar=False)
    # CrossEncoder.predict returns a numpy array shape (1,).
    score = float(scores[0])
    assert score > 0.01, (
        f"Reranker score collapsed to {score:.6f} — este es el patrón "
        "documentado del bug fp16-en-MPS. Verificar que get_reranker() "
        "sigue forzando fp32 (no hay .half() ni torch_dtype=float16 en "
        "el path)."
    )
