"""Tests para Quick Win #4 (2026-05-04) — LLM typo correction via qwen2.5:3b.

Valida:
1. Query con typo → corregida vía LLM (casos reales: "asor", "psicologa", "biba").
2. Query ya correcta → devuelta sin modificar.
3. Sanity check: si el LLM devuelve > 1.5× len(original) → rechazar, devolver original.
4. Cache LRU: 2 llamadas a la misma query → 1 sola llamada LLM.
5. Silent fail: ollama timeout / excepción → devuelve original sin crashear.
6. Feature gate: RAG_TYPO_CORRECTION=0 → _correct_typos_llm devuelve original sin LLM call.
7. Integration: expand_queries("psicologa") usa la query corregida downstream.
8. Wire-up: el ContextVar `_LLM_TYPO_LAST` queda seteado cuando hay corrección.
9. Wire-up: el ContextVar `_LLM_TYPO_LAST` es None cuando no hay corrección.

Refactor 2026-05-04 evening: las globals legacy
`_expand_last_llm_typo_*[0]` fueron eliminadas porque tenían data race en
el web multi-threaded. Reemplazo: `_LLM_TYPO_LAST: ContextVar` populado
por `_correct_typos_llm` y cosechado por `_apply_typo_telemetry(rr)`
desde `retrieve()` / `multi_retrieve()`.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

import rag


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_ollama_response(text: str) -> MagicMock:
    """Simula una respuesta de ollama.chat() con el texto dado."""
    resp = MagicMock()
    resp.message.content = text
    return resp


def _clear_llm_typo_cache():
    """Limpia el cache LRU de _correct_typos_llm entre tests."""
    with rag._llm_typo_cache_lock:
        rag._llm_typo_cache.clear()


# ── Tests: _correct_typos_llm ─────────────────────────────────────────────────

class TestCorrectTyposLlm:
    """Unit tests de _correct_typos_llm — mockea ollama.chat."""

    def setup_method(self):
        _clear_llm_typo_cache()
        # Asegurarse que la feature está ON para todos los tests de esta clase
        self._orig = rag._TYPO_CORRECTION_ENABLED
        rag._TYPO_CORRECTION_ENABLED = True

    def teardown_method(self):
        _clear_llm_typo_cache()
        rag._TYPO_CORRECTION_ENABLED = self._orig

    def test_typo_asor_corrigido_a_astor(self):
        """'asor' → 'Astor' cuando el LLM lo corrige."""
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response("Astor")):
            result = rag._correct_typos_llm("asor")
        assert result == "Astor"

    def test_typo_psicologa_corrigida(self):
        """'psicologa' → 'psicóloga'."""
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response("psicóloga")):
            result = rag._correct_typos_llm("psicologa")
        assert result == "psicóloga"

    def test_typo_biba_corrigida(self):
        """'biba' → 'BICA' (nombre propio de institución financiera)."""
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response("BICA")):
            result = rag._correct_typos_llm("biba")
        assert result == "BICA"

    def test_query_correcta_no_se_modifica(self):
        """Cuando la query ya está bien, el LLM la devuelve igual → no cambia."""
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response("qué hago mañana")):
            result = rag._correct_typos_llm("qué hago mañana")
        assert result == "qué hago mañana"

    def test_sanity_check_output_demasiado_largo(self):
        """Si el LLM devuelve >1.5× la longitud original → rechazar, devolver original."""
        query = "asor"
        # El LLM devuelve algo 3× más largo
        llm_output = "Astor Piazzolla, el gran maestro del tango argentino"
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response(llm_output)):
            result = rag._correct_typos_llm(query)
        assert result == query, (
            f"El sanity check debería haber rechazado la corrección larga: {result!r}"
        )

    def test_cache_lru_segunda_llamada_no_invoca_llm(self):
        """Dos llamadas con la misma query → el LLM se invoca solo una vez."""
        call_count = 0

        def _fake_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_ollama_response("psicóloga")

        with patch.object(rag._helper_client(), "chat", side_effect=_fake_chat):
            r1 = rag._correct_typos_llm("psicologa")
            r2 = rag._correct_typos_llm("psicologa")

        assert call_count == 1, "LLM se invocó más de una vez para la misma query"
        assert r1 == r2 == "psicóloga"

    def test_silent_fail_excepcion_devuelve_original(self):
        """ollama lanza excepción (e.g. timeout) → devuelve la query original sin crashear."""
        query = "hatercito"
        with patch.object(rag._helper_client(), "chat",
                          side_effect=Exception("Connection timeout")):
            result = rag._correct_typos_llm(query)
        assert result == query

    def test_feature_gate_off_no_invoca_llm(self):
        """Con RAG_TYPO_CORRECTION=0 → _correct_typos_llm retorna original sin LLM."""
        rag._TYPO_CORRECTION_ENABLED = False
        call_count = 0

        def _fake_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_ollama_response("Astor")

        with patch.object(rag._helper_client(), "chat", side_effect=_fake_chat):
            result = rag._correct_typos_llm("asor")

        assert result == "asor", "Feature gate OFF debería devolver original"
        assert call_count == 0, "LLM no debería invocarse cuando gate está OFF"

    def test_empty_query_devuelve_original(self):
        """Query vacía → devuelta tal cual sin invocar LLM."""
        with patch.object(rag._helper_client(), "chat") as mock_chat:
            result = rag._correct_typos_llm("")
        mock_chat.assert_not_called()
        assert result == ""

    def test_llm_devuelve_vacio_fallback_a_original(self):
        """Si el LLM devuelve string vacío → fallback a original."""
        query = "psicologa"
        with patch.object(rag._helper_client(), "chat",
                          return_value=_mock_ollama_response("")):
            result = rag._correct_typos_llm(query)
        assert result == query


# ── Tests: expand_queries integration ─────────────────────────────────────────

class TestExpandQueriesTypoIntegration:
    """Verifica que expand_queries() usa la query corregida downstream.

    Post-refactor 2026-05-04: el wire-up se valida via el ContextVar
    `_LLM_TYPO_LAST` (thread-safe) en lugar de las globals legacy.
    """

    def setup_method(self):
        _clear_llm_typo_cache()
        self._orig_enabled = rag._TYPO_CORRECTION_ENABLED
        rag._TYPO_CORRECTION_ENABLED = True
        # Resetear el ContextVar antes de cada test (igual que `retrieve()`
        # hace al inicio para garantizar state limpio).
        rag._typo_telemetry_reset()

    def teardown_method(self):
        _clear_llm_typo_cache()
        rag._TYPO_CORRECTION_ENABLED = self._orig_enabled
        rag._typo_telemetry_reset()

    def test_expand_queries_usa_query_corregida(self):
        """expand_queries("psicologa") → la lista resultante arranca con "psicóloga"."""
        def _fake_chat(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0]["content"] if messages else ""
            # Primera llamada: typo correction
            if "Corregí los errores" in content:
                return _mock_ollama_response("psicóloga")
            # Segunda llamada: paraphrases
            return _mock_ollama_response("turno con psicóloga\ncita con terapeuta")

        with patch.object(rag._helper_client(), "chat", side_effect=_fake_chat):
            result = rag.expand_queries("psicologa")

        assert result[0] == "psicóloga", (
            f"El primer elemento debería ser la query corregida, got {result[0]!r}"
        )

    def test_expand_queries_sets_typo_ctx_when_corrected(self):
        """Cuando hay corrección, el ContextVar `_LLM_TYPO_LAST` queda seteado
        con el delta {original, corrected, was_corrected: True}."""
        def _fake_chat(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0]["content"] if messages else ""
            if "Corregí los errores" in content:
                return _mock_ollama_response("Astor")
            return _mock_ollama_response("info sobre Astor\ndatos de Astor")

        with patch.object(rag._helper_client(), "chat", side_effect=_fake_chat):
            rag.expand_queries("asor")

        delta = rag._typo_telemetry_get()
        assert delta is not None, "El ContextVar debería tener el delta de la corrección"
        assert delta["original"] == "asor"
        assert delta["corrected"] == "Astor"
        assert delta["was_corrected"] is True

    def test_expand_queries_typo_ctx_none_when_no_correction(self):
        """Cuando no hay corrección, el ContextVar `_LLM_TYPO_LAST` es None."""
        def _fake_chat(**kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0]["content"] if messages else ""
            if "Corregí los errores" in content:
                return _mock_ollama_response("qué hago mañana")
            return _mock_ollama_response("qué tengo que hacer hoy\nplanes para mañana")

        with patch.object(rag._helper_client(), "chat", side_effect=_fake_chat):
            rag.expand_queries("qué hago mañana")

        assert rag._typo_telemetry_get() is None, (
            "No debería haber delta para una query ya correcta"
        )

    def test_apply_typo_telemetry_writes_to_retrieve_result(self):
        """`_apply_typo_telemetry(rr)` cosecha el ContextVar y escribe los 3
        fields (`llm_typo_corrected`, `llm_typo_original`,
        `llm_typo_corrected_text`) al `RetrieveResult`."""
        # Setear el CV manualmente como lo haría `_correct_typos_llm`
        rag._LLM_TYPO_LAST.set({
            "original": "asor",
            "corrected": "Astor",
            "was_corrected": True,
        })
        rr = rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=0.0,
            search_query="Astor",
        )
        assert rr.llm_typo_corrected is False  # default
        rag._apply_typo_telemetry(rr)
        assert rr.llm_typo_corrected is True
        assert rr.llm_typo_original == "asor"
        assert rr.llm_typo_corrected_text == "Astor"

    def test_apply_typo_telemetry_noop_when_ctx_empty(self):
        """`_apply_typo_telemetry(rr)` no toca el RetrieveResult cuando el
        ContextVar está en su default (None)."""
        rag._typo_telemetry_reset()
        rr = rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=0.0,
            search_query="qué hago mañana",
        )
        rag._apply_typo_telemetry(rr)
        assert rr.llm_typo_corrected is False
        assert rr.llm_typo_original is None
        assert rr.llm_typo_corrected_text is None

    def test_thread_safe_isolation_via_contextvar(self):
        """Dos threads concurrentes setean el ContextVar con valores
        distintos — cada uno lee SU valor sin interferencia (data race
        que el refactor 2026-05-04 evening eliminó)."""
        import threading
        import contextvars

        results: dict[str, dict | None] = {}
        barrier = threading.Barrier(2)

        def _worker(name: str, original: str, corrected: str):
            ctx = contextvars.copy_context()
            def _set_and_read():
                rag._LLM_TYPO_LAST.set({
                    "original": original,
                    "corrected": corrected,
                    "was_corrected": True,
                })
                barrier.wait()  # asegurar interleaving
                results[name] = rag._typo_telemetry_get()
            ctx.run(_set_and_read)

        t1 = threading.Thread(target=_worker, args=("A", "asor", "Astor"))
        t2 = threading.Thread(target=_worker, args=("B", "biba", "BICA"))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results["A"] == {
            "original": "asor", "corrected": "Astor", "was_corrected": True,
        }
        assert results["B"] == {
            "original": "biba", "corrected": "BICA", "was_corrected": True,
        }
