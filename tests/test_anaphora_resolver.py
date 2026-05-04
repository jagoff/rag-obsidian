"""Tests for the anaphora resolver (Quick Win #1).

Covers:
  - `_is_anaphoric_query` detector (regex + token-count rules).
  - `_resolve_anaphora` happy path (mocked LLM, history fed to prompt).
  - LRU cache: same `(history, query)` no llama LLM 2 veces.
  - Silent-fail: ollama timeout → return original query.
  - Integration en `multi_retrieve`: history del turn previo + query
    "y en Madrid?" → query reescrita pasada a `retrieve()`.
  - Gate `RAG_ANAPHORA_RESOLVER=0` lo apaga.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ── Detector ─────────────────────────────────────────────────────────────


def test_detector_no_history_returns_false():
    import rag
    assert rag._is_anaphoric_query("y en Madrid?", []) is False
    assert rag._is_anaphoric_query("y en Madrid?", None) is False


def test_detector_connector_y_en():
    import rag
    hist = [{"role": "user", "content": "qué tengo sobre Buenos Aires"}]
    assert rag._is_anaphoric_query("y en Madrid?", hist) is True


def test_detector_connector_y_para():
    import rag
    hist = [{"role": "user", "content": "qué tengo para hoy"}]
    assert rag._is_anaphoric_query("y para mañana?", hist) is True


def test_detector_connector_pero():
    import rag
    hist = [{"role": "user", "content": "tengo notas de Astor"}]
    assert rag._is_anaphoric_query("pero las del jardín?", hist) is True


def test_detector_connector_tambien_with_accent():
    import rag
    hist = [{"role": "user", "content": "notas sobre coaching"}]
    assert rag._is_anaphoric_query("también de Córdoba?", hist) is True
    assert rag._is_anaphoric_query("tambien de Córdoba?", hist) is True


def test_detector_connector_ahora():
    import rag
    hist = [{"role": "user", "content": "qué hay en mi inbox"}]
    assert rag._is_anaphoric_query("ahora con tag urgente", hist) is True


def test_detector_short_query_below_threshold():
    """Una query <8 tokens sin conector explícito sigue contando como
    anafórica si hay history (chats casi siempre tienen follow-ups
    contextuales <8 tokens)."""
    import rag
    hist = [{"role": "user", "content": "qué dijiste sobre el clima"}]
    assert rag._is_anaphoric_query("y eso?", hist) is True
    assert rag._is_anaphoric_query("dale más detalles", hist) is True


def test_detector_long_query_self_contained():
    """Queries >=8 tokens sin conector inicial → False (autónoma)."""
    import rag
    hist = [{"role": "user", "content": "algo previo"}]
    long_q = "quiero saber sobre la teoría general de la relatividad einstein"
    assert rag._is_anaphoric_query(long_q, hist) is False


def test_detector_empty_query():
    import rag
    hist = [{"role": "user", "content": "x"}]
    assert rag._is_anaphoric_query("", hist) is False
    assert rag._is_anaphoric_query("   ", hist) is False


# ── Resolver: happy path ─────────────────────────────────────────────────


def _make_fake_resp(content: str):
    class _Msg:
        def __init__(self, c): self.content = c
    class _Resp:
        def __init__(self, c): self.message = _Msg(c)
    return _Resp(content)


def test_resolver_calls_helper_with_history_and_query(monkeypatch):
    """Verifica que `_resolve_anaphora` invoca al helper LLM con history
    + query en el prompt, y devuelve el rewrite."""
    import rag

    # Limpiar el LRU cache para que este test no se vea contaminado por
    # otros del archivo (mismas keys cacheadas vivirían entre tests).
    rag._cached_anaphora_resolution.cache_clear()

    captured = {"prompt": None, "calls": 0}

    def _fake_chat(model, messages, **kwargs):
        captured["calls"] += 1
        captured["prompt"] = messages[0]["content"]
        return _make_fake_resp("clima en Madrid")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    history = [
        {"role": "user", "content": "qué clima hace en Buenos Aires"},
        {"role": "assistant", "content": "Hoy hay 22°C y sol en Buenos Aires."},
    ]
    out = rag._resolve_anaphora("y en Madrid?", history)
    assert out == "clima en Madrid"
    assert captured["calls"] == 1
    # El prompt debe contener tanto la query nueva como la entidad del
    # turno previo ("Buenos Aires").
    assert "y en Madrid?" in captured["prompt"]
    assert "Buenos Aires" in captured["prompt"]


def test_resolver_caches_repeat_calls(monkeypatch):
    """Misma `(history_hash, query)` → segunda llamada hit cache, 1 sólo
    LLM call."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    calls = {"n": 0}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        return _make_fake_resp("rewrite cacheada")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    history = [{"role": "user", "content": "qué tengo de astor"}]
    out1 = rag._resolve_anaphora("y de Juli?", history)
    out2 = rag._resolve_anaphora("y de Juli?", history)
    assert out1 == out2 == "rewrite cacheada"
    assert calls["n"] == 1, "segunda invocación debe pegar cache, no LLM"


def test_resolver_silent_fail_returns_original(monkeypatch):
    """Helper LLM tira excepción → devolver query original sin crashear."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()

    def _fake_chat(model, messages, **kwargs):
        raise Exception("timed out")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_silent_log", lambda *a, **k: None)
    history = [{"role": "user", "content": "contexto"}]
    out = rag._resolve_anaphora("y eso?", history)
    assert out == "y eso?"


def test_resolver_no_history_returns_query_immediate(monkeypatch):
    """Sin historial → fast-path sin LLM call."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    calls = {"n": 0}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        return _make_fake_resp("never reached")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    out = rag._resolve_anaphora("y en Madrid?", [])
    assert out == "y en Madrid?"
    assert calls["n"] == 0


def test_resolver_clamps_runaway_output(monkeypatch):
    """Si el helper devuelve un output >3× la longitud del input, lo
    descartamos (es señal de que el modelo se fue por la tangente con
    una explicación)."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    big_response = "a" * 1000

    def _fake_chat(model, messages, **kwargs):
        return _make_fake_resp(big_response)

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    history = [{"role": "user", "content": "x"}]
    out = rag._resolve_anaphora("y eso?", history)
    assert out == "y eso?"  # fallback a la original


# ── Gate ─────────────────────────────────────────────────────────────────


def test_gate_default_on(monkeypatch):
    import rag
    monkeypatch.delenv("RAG_ANAPHORA_RESOLVER", raising=False)
    assert rag._anaphora_resolver_enabled() is True


def test_gate_off_with_zero(monkeypatch):
    import rag
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "0")
    assert rag._anaphora_resolver_enabled() is False


def test_gate_off_with_false(monkeypatch):
    import rag
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "false")
    assert rag._anaphora_resolver_enabled() is False


# ── Integration en multi_retrieve ────────────────────────────────────────


def test_multi_retrieve_resolves_anaphora_before_retrieve(monkeypatch, tmp_path):
    """End-to-end: history previo + query "y en Madrid?" → la query
    rewritten ("clima en Madrid") debe ser la que llega a `retrieve()`,
    NO la original. Telemetría debe quedar en `extras`.
    """
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "1")

    # Mock helper LLM para que devuelva la rewrite determinística.
    monkeypatch.setattr(rag.ollama, "chat",
                        lambda *a, **k: _make_fake_resp("clima en Madrid hoy"))

    captured = {"q_to_retrieve": None}

    # Mock `retrieve()` para capturar qué query llega y devolver un
    # RetrieveResult mínimo. Evita tener que indexar un vault real.
    def _fake_retrieve(col, q, k, *args, **kwargs):
        captured["q_to_retrieve"] = q
        return rag.RetrieveResult(
            docs=["doc"], metas=[{"file": "a.md"}], scores=[0.5],
            confidence=0.5, search_query=q, query_variants=[q],
        )

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "deep_retrieve", _fake_retrieve)

    # Mock `get_db_for` para que no toque sqlite-vec real.
    class _FakeCol:
        def count(self): return 100
    monkeypatch.setattr(rag, "get_db_for", lambda p: _FakeCol())

    vaults = [("home", tmp_path)]
    history = [
        {"role": "user", "content": "qué clima hace en Buenos Aires"},
        {"role": "assistant", "content": "22°C y sol."},
    ]
    result = rag.multi_retrieve(
        vaults, "y en Madrid?", k=5, folder=None, history=history,
    )

    # La query reescrita ("clima en Madrid hoy") debe ser la pasada
    # a retrieve(), no la original ("y en Madrid?").
    assert captured["q_to_retrieve"] == "clima en Madrid hoy"
    # Telemetría debe estar en extras del result.
    extras = result.get("extras") or {}
    assert extras.get("anaphora_resolved") is True
    assert extras.get("anaphora_original") == "y en Madrid?"
    assert extras.get("anaphora_rewritten") == "clima en Madrid hoy"


def test_multi_retrieve_skips_when_query_self_contained(monkeypatch, tmp_path):
    """Query autónoma >=8 tokens → no se invoca el resolver, query
    intacta llega a retrieve()."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "1")
    llm_calls = {"n": 0}

    def _fake_chat(*a, **k):
        llm_calls["n"] += 1
        return _make_fake_resp("never reached")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)

    captured = {"q": None}

    def _fake_retrieve(col, q, k, *args, **kwargs):
        captured["q"] = q
        return rag.RetrieveResult(
            docs=[], metas=[], scores=[],
            confidence=float("-inf"), search_query=q, query_variants=[q],
        )

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "deep_retrieve", _fake_retrieve)

    class _FakeCol:
        def count(self): return 100
    monkeypatch.setattr(rag, "get_db_for", lambda p: _FakeCol())

    long_q = "quiero saber sobre la teoría general de la relatividad einstein"
    history = [{"role": "user", "content": "algo previo"}]
    result = rag.multi_retrieve(
        [("home", tmp_path)], long_q, k=5, folder=None, history=history,
    )
    assert captured["q"] == long_q
    assert llm_calls["n"] == 0
    extras = result.get("extras") or {}
    assert extras.get("anaphora_resolved") is False


def test_multi_retrieve_disabled_by_env(monkeypatch, tmp_path):
    """`RAG_ANAPHORA_RESOLVER=0` → resolver no corre aunque la query sea
    anafórica."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "0")

    llm_calls = {"n": 0}

    def _fake_chat(*a, **k):
        llm_calls["n"] += 1
        return _make_fake_resp("not used")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)

    captured = {"q": None}

    def _fake_retrieve(col, q, k, *args, **kwargs):
        captured["q"] = q
        return rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=float("-inf"),
            search_query=q, query_variants=[q],
        )

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "deep_retrieve", _fake_retrieve)

    class _FakeCol:
        def count(self): return 100
    monkeypatch.setattr(rag, "get_db_for", lambda p: _FakeCol())

    history = [{"role": "user", "content": "qué dijiste"}]
    rag.multi_retrieve(
        [("home", tmp_path)], "y en Madrid?", k=5, folder=None,
        history=history,
    )
    # El gate apagado deja la query original.
    assert captured["q"] == "y en Madrid?"
    assert llm_calls["n"] == 0


def test_multi_retrieve_unchanged_rewrite_marks_not_resolved(
    monkeypatch, tmp_path,
):
    """Si el helper devuelve la query igual a la original → marcamos
    `anaphora_resolved=False` (no inflamos telemetría con falsos
    positivos)."""
    import rag

    rag._cached_anaphora_resolution.cache_clear()
    monkeypatch.setenv("RAG_ANAPHORA_RESOLVER", "1")

    monkeypatch.setattr(rag.ollama, "chat",
                        lambda *a, **k: _make_fake_resp("y en Madrid?"))

    def _fake_retrieve(col, q, k, *args, **kwargs):
        return rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=float("-inf"),
            search_query=q, query_variants=[q],
        )

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "deep_retrieve", _fake_retrieve)

    class _FakeCol:
        def count(self): return 100
    monkeypatch.setattr(rag, "get_db_for", lambda p: _FakeCol())

    history = [{"role": "user", "content": "contexto"}]
    result = rag.multi_retrieve(
        [("home", tmp_path)], "y en Madrid?", k=5, folder=None,
        history=history,
    )
    extras = result.get("extras") or {}
    assert extras.get("anaphora_resolved") is False


# ── to_log_event surfaces telemetry ──────────────────────────────────────


def test_to_log_event_surfaces_anaphora_fields():
    """ChatTurnResult.to_log_event() debe levantar los 3 fields anaphora
    desde RetrieveResult.extras al payload del log."""
    import rag

    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=0.5,
        search_query="clima madrid", query_variants=["clima madrid"],
        extras={
            "anaphora_resolved": True,
            "anaphora_original": "y en Madrid?",
            "anaphora_rewritten": "clima en Madrid",
        },
    )
    ctr = rag.ChatTurnResult(
        answer="ok", retrieve_result=rr, question="y en Madrid?",
    )
    ev = ctr.to_log_event(cmd="web", session_id="sess-1")
    assert ev["anaphora_resolved"] is True
    assert ev["anaphora_original"] == "y en Madrid?"
    assert ev["anaphora_rewritten"] == "clima en Madrid"


def test_to_log_event_anaphora_default_when_absent():
    """Sin extras → defaults silenciosos (False, "", "")."""
    import rag

    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=0.5,
        search_query="hola", query_variants=["hola"],
    )
    ctr = rag.ChatTurnResult(
        answer="ok", retrieve_result=rr, question="hola",
    )
    ev = ctr.to_log_event(cmd="cli", session_id="s")
    assert ev["anaphora_resolved"] is False
    assert ev["anaphora_original"] == ""
    assert ev["anaphora_rewritten"] == ""
