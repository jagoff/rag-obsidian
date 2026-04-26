"""Tests for the OCR cita-detector LLM client timeout contract.

Audit 2026-04-25 R2-OCR #1: el detector LLM (qwen2.5:3b vía
`_helper_client`) debe tener timeout explícito. Sin esto, una llamada
colgada de ollama bloquea el endpoint `/api/chat/upload-image` por
minutos — el upload UI queda spinneando, el user piensa que rompió y
re-postea, multiplicando el bloqueo.

El contrato vive en `rag.ocr._DETECTOR_TIMEOUT` (consumer side) y la
implementación en `rag._helper_client()` (provider side, devuelve un
`_TimedOllamaProxy` con `_timeout` configurado). Estos tests aseguran
que ambos lados están en sync y que el valor está dentro del rango
razonable.
"""

from __future__ import annotations


def test_detector_timeout_constant_defined_and_sane():
    """`_DETECTOR_TIMEOUT` existe, es positivo y ≤60s.

    Si alguien cambia el constant a 0 (deshabilita timeout), o lo sube
    a algo absurdo (300s, 600s), el endpoint vuelve a ser bloqueable
    por una sola call lenta — eso es exactamente lo que el audit
    R2-OCR #1 quiere prevenir.
    """
    from rag.ocr import _DETECTOR_TIMEOUT

    assert isinstance(_DETECTOR_TIMEOUT, (int, float)), (
        "_DETECTOR_TIMEOUT debe ser numérico"
    )
    assert _DETECTOR_TIMEOUT > 0, "timeout debe ser positivo (no deshabilitado)"
    assert _DETECTOR_TIMEOUT <= 60.0, (
        "timeout demasiado alto — una call colgada bloquearía el endpoint "
        "demasiado tiempo. Si necesitás más, revisá el audit primero."
    )


def test_detector_client_honors_timeout_contract():
    """`rag._helper_client()` devuelve un proxy cuyo `_timeout` matchea
    `_DETECTOR_TIMEOUT`. Esto detecta drift entre el contract documentado
    en `rag/ocr.py` y la implementación real en `rag/__init__.py`.

    Si el día de mañana alguien sube el timeout en `_helper_client` sin
    actualizar `_DETECTOR_TIMEOUT` (o viceversa), este test falla y
    obliga a tomar una decisión consciente.
    """
    from rag import _helper_client
    from rag.ocr import _DETECTOR_TIMEOUT

    proxy = _helper_client()
    # `_TimedOllamaProxy` expone `_timeout` (ver `rag/__init__.py:2741`).
    # Si en el futuro la API cambia, este assert falla con un mensaje
    # claro y el dev decide cómo re-cablear.
    assert hasattr(proxy, "_timeout"), (
        "_helper_client() debe devolver un objeto con atributo `_timeout` "
        "(esperado: _TimedOllamaProxy). Revisar rag.__init__._helper_client."
    )
    assert proxy._timeout == _DETECTOR_TIMEOUT, (
        f"timeout drift: rag.ocr._DETECTOR_TIMEOUT={_DETECTOR_TIMEOUT}s vs "
        f"rag._helper_client()._timeout={proxy._timeout}s. "
        "Sincronizar ambos lados (audit R2-OCR #1)."
    )
