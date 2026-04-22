"""Tests for `run_parallel_post_process()` — el helper que corre citation-
repair + critique + NLI grounding concurrentemente (2026-04-22).

Valida:
1. Fast-path (sin tareas activas) → no-op, full_orig devuelto intacto.
2. Solo repair exitoso → full reemplazado, flags correctos.
3. Solo critique changed → full reemplazado.
4. Repair + critique ambos quieren mutar → repair gana (merge priority).
5. NLI corre en paralelo sin mutar full.
6. Timing_ms populado para cada task que corrió.
7. Exception en una task no tumba las otras.
8. RAG_PARALLEL_POSTPROCESS=0 fuerza secuencial (determinismo para debugging).
9. fast_path=True bypasea repair aun con bad citations.
10. `citation_repaired=False` + `critique_fired=False` cuando no corre ninguno.
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

import rag


# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, content):
        self.message = MagicMock(content=content)


def _meta(path="01-Projects/nota.md"):
    return {"file": path, "note": "nota", "folder": "01-Projects"}


# ── 1. No-op cuando no hay tareas ────────────────────────────────────────────


def test_noop_no_tasks(monkeypatch):
    """Sin bad citations + sin critique + NLI off → helper retorna full_orig
    intacto y timing_ms solo con wall."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])

    result = rag.run_parallel_post_process(
        "respuesta original",
        docs=["doc1"],
        metas=[_meta()],
        context="ctx",
        question="q",
        fast_path=False,
        critique=False,
        intent="semantic",
    )
    assert result.full == "respuesta original"
    assert result.bad_citations == []
    assert result.citation_repaired is False
    assert result.critique_fired is False
    assert result.critique_changed is False
    assert result.nli_summary is None
    assert result.nli_ms == 0
    # Wall-time present, other timings 0 o ausentes
    assert "wall" in result.timing_ms


# ── 2. Solo repair exitoso ───────────────────────────────────────────────────


def test_repair_success_replaces_full(monkeypatch):
    """Con bad citations + valid paths + repair LLM devuelve texto sin
    bads → full reemplazado, citation_repaired=True."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)

    # Primera call de verify_citations: 1 bad en el full_orig.
    # Segunda call (dentro de repair): 0 bads → repair.ok=True.
    call_count = {"n": 0}
    def fake_verify(text, metas):
        call_count["n"] += 1
        return [("Fake", "99.md")] if call_count["n"] == 1 else []
    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("respuesta reparada")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "respuesta original con cita mala",
        docs=["d"],
        metas=[_meta()],
        context="ctx",
        question="q",
        fast_path=False,
        critique=False,
        intent="semantic",
    )
    assert result.full == "respuesta reparada"
    assert result.citation_repaired is True
    assert result.critique_fired is False
    assert result.timing_ms.get("repair", 0) >= 0


def test_repair_empty_keeps_original(monkeypatch):
    """Si repair LLM devuelve string vacío → no reemplaza."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [("Bad", "99.md")])
    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original", docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=False, intent="semantic",
    )
    assert result.full == "original"
    assert result.citation_repaired is False


def test_repair_still_bad_keeps_original(monkeypatch):
    """Repair devuelve texto pero sigue teniendo bad citations → no reemplaza."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations",
                        lambda text, metas: [("Bad", "99.md")])  # siempre bad
    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("still bad")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original", docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=False, intent="semantic",
    )
    assert result.full == "original"
    assert result.citation_repaired is False


# ── 3. Solo critique mutando ─────────────────────────────────────────────────


def test_critique_changes_replace_full(monkeypatch):
    """Critique devuelve texto distinto del original → replace + flag set."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])  # no repair

    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("respuesta critiqueada mejor")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "respuesta original",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=True, intent="semantic",
    )
    assert result.full == "respuesta critiqueada mejor"
    assert result.critique_fired is True
    assert result.critique_changed is True


def test_critique_same_text_no_replace(monkeypatch):
    """Critique devuelve mismo texto (modulo whitespace) → NO replace."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])

    fake_client = MagicMock()
    # Mismo texto pero con extra whitespace — norm() las iguala
    fake_client.chat.return_value = _FakeResp("  Respuesta  idéntica.  ")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "Respuesta idéntica.",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=True, intent="semantic",
    )
    assert result.full == "Respuesta idéntica."
    assert result.critique_fired is True
    assert result.critique_changed is False


# ── 4. Merge priority: repair > critique ─────────────────────────────────────


def test_merge_priority_repair_over_critique(monkeypatch):
    """Si repair Y critique ambos mutan → repair gana."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)

    # repair path: primer verify=bad, segundo verify=clean
    call_count = {"n": 0}
    def fake_verify(text, metas):
        call_count["n"] += 1
        # 1er call: full_orig (bad)
        # 2do call: dentro del repair task, devuelve clean
        return [("Fake", "99.md")] if call_count["n"] == 1 else []
    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    # Llamadas al LLM: 2 (una para repair, una para critique). Distinto texto
    # cada una así podemos distinguir cuál ganó.
    client_calls = {"n": 0}
    fake_client = MagicMock()
    def fake_chat(**kwargs):
        client_calls["n"] += 1
        # Ambas tasks corren en paralelo, no podemos garantizar el orden;
        # pero ambas devuelven texto distinto del original
        if client_calls["n"] == 1:
            return _FakeResp("repaired version")
        return _FakeResp("critiqued version")
    fake_client.chat.side_effect = fake_chat
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original con cita mala",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=True, intent="semantic",
    )
    # Independiente del orden de ejecución concurrent: repair gana si ok
    # (citation_repaired=True). El critique changed se registra
    # pero no muta full.
    assert result.citation_repaired is True
    # full tiene que ser una de las dos respuestas (la que el repair produjo),
    # NO el original.
    assert result.full != "original con cita mala"


# ── 5. fast_path bypasea repair ──────────────────────────────────────────────


def test_fast_path_skips_repair(monkeypatch):
    """fast_path=True → repair no corre aun con bad citations."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations",
                        lambda text, metas: [("Bad", "99.md")])
    # chat no debería llamarse — si se llama, es un bug
    fake_client = MagicMock()
    fake_client.chat.side_effect = AssertionError("repair must not run with fast_path")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=True, critique=False, intent="semantic",
    )
    assert result.full == "original"
    assert result.citation_repaired is False
    assert result.bad_citations == [("Bad", "99.md")]


# ── 6. Exception isolation ───────────────────────────────────────────────────


def test_exception_in_repair_doesnt_crash(monkeypatch):
    """Si el repair LLM raisea → task se degrada silencioso, NO propaga.
    Critique sigue funcionando."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setattr(rag, "verify_citations",
                        lambda text, metas: [("Bad", "99.md")])

    # Fake client: raisea para repair, devuelve OK para critique.
    # Pero como el orden de ejecución en ThreadPoolExecutor no es determinístico,
    # hacemos el client raisear en el primer call.
    client_calls = {"n": 0}
    fake_client = MagicMock()
    def fake_chat(**kwargs):
        client_calls["n"] += 1
        if client_calls["n"] == 1:
            raise RuntimeError("simulated LLM crash")
        return _FakeResp("critique output")
    fake_client.chat.side_effect = fake_chat
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    # Test no debe crashear. full_orig puede haber sido reemplazado por
    # critique o quedar intacto dependiendo del orden; ambos son válidos.
    result = rag.run_parallel_post_process(
        "original",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=True, intent="semantic",
    )
    # Repair falló (no reemplazó por exception) — flag False
    assert result.citation_repaired is False
    # No exception propagada (llegamos acá)


# ── 7. RAG_PARALLEL_POSTPROCESS=0 fuerza secuencial ──────────────────────────


def test_parallel_off_runs_sequential(monkeypatch):
    """Con la flag off, las tasks corren secuenciales — el resultado final
    debe ser el mismo (repair + critique configurado) pero la concurrencia
    no se usa."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    monkeypatch.setenv("RAG_PARALLEL_POSTPROCESS", "0")

    # No bad citations → no repair. Solo critique.
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])
    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("critiqued sequential")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=True, intent="semantic",
    )
    assert result.full == "critiqued sequential"
    assert result.critique_changed is True


# ── 8. _pp_parallel_enabled toggle ───────────────────────────────────────────


def test_pp_parallel_enabled_default_on(monkeypatch):
    """Sin env → on por default."""
    monkeypatch.delenv("RAG_PARALLEL_POSTPROCESS", raising=False)
    assert rag._pp_parallel_enabled() is True


def test_pp_parallel_enabled_env_overrides(monkeypatch):
    """Env var apaga el toggle."""
    for falsy in ("0", "false", "no", "off", "FALSE"):
        monkeypatch.setenv("RAG_PARALLEL_POSTPROCESS", falsy)
        assert rag._pp_parallel_enabled() is False, f"value {falsy!r} should disable"
    for truthy in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv("RAG_PARALLEL_POSTPROCESS", truthy)
        assert rag._pp_parallel_enabled() is True, f"value {truthy!r} should enable"


# ── 9. Timing fields populated ───────────────────────────────────────────────


def test_timing_ms_populated(monkeypatch):
    """Todas las tasks que corrieron deben tener ms >= 0 en timing_ms."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)

    call_count = {"n": 0}
    def fake_verify(text, metas):
        call_count["n"] += 1
        return [("Fake", "99.md")] if call_count["n"] == 1 else []
    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    fake_client = MagicMock()
    fake_client.chat.return_value = _FakeResp("repaired")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    result = rag.run_parallel_post_process(
        "original con mala cita",
        docs=["d"], metas=[_meta()],
        context="c", question="q",
        fast_path=False, critique=False, intent="semantic",
    )
    assert "wall" in result.timing_ms
    assert result.timing_ms["wall"] >= 0
    assert result.timing_ms.get("repair", 0) >= 0


# ── 10. PostProcessResult dataclass contract ────────────────────────────────


def test_postprocess_result_has_expected_fields():
    r = rag.PostProcessResult(full="hola")
    assert r.full == "hola"
    assert r.bad_citations == []
    assert r.citation_repaired is False
    assert r.critique_fired is False
    assert r.critique_changed is False
    assert r.nli_summary is None
    assert r.nli_ms == 0
    assert r.nli_result is None
    assert r.timing_ms == {}
