"""Tests for `reformulate_query(seen_titles=...)` and `_titles_from_paths`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_titles_from_paths_basic():
    import rag
    out = rag._titles_from_paths([
        "02-Areas/Coaching/Ikigai.md",
        "03-Resources/CNV.md",
    ])
    assert out == ["Ikigai", "CNV"]


def test_titles_from_paths_dedupes_and_preserves_order():
    import rag
    out = rag._titles_from_paths([
        "02-Areas/Coaching/Ikigai.md",
        "04-Archive/Ikigai.md",  # same stem → dedup
        "03-Resources/CNV.md",
    ])
    assert out == ["Ikigai", "CNV"]


def test_titles_from_paths_handles_empty_and_none():
    import rag
    assert rag._titles_from_paths(None) == []
    assert rag._titles_from_paths([]) == []
    assert rag._titles_from_paths(["", ""]) == []


def test_titles_from_paths_respects_limit():
    import rag
    paths = [f"a/{i}.md" for i in range(20)]
    out = rag._titles_from_paths(paths, limit=3)
    assert len(out) == 3
    assert out == ["0", "1", "2"]


def test_reformulate_no_history_no_summary_returns_input(monkeypatch):
    import rag
    # Should short-circuit without calling the LLM
    called = {"n": 0}

    def _fake_chat(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    out = rag.reformulate_query("qué es X", [], summary=None)
    assert out == "qué es X"
    assert called["n"] == 0


def test_reformulate_accepts_seen_titles_kwarg(monkeypatch):
    """Regression: seen_titles kwarg must be accepted without error.

    Current behavior: kwarg is scaffolding only — NOT injected into the
    prompt (the 2026-04-17 attempt regressed chains by -16pp hit@5 and
    -33pp chain_success; reverted but kwarg kept for future iteration).
    """
    import rag

    class _Msg:
        def __init__(self, c): self.content = c

    class _Resp:
        def __init__(self, c): self.message = _Msg(c)

    captured = {"prompt": None}

    def _fake_chat(model, messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return _Resp("out")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    history = [{"role": "user", "content": "x"}]
    # Must not raise even with seen_titles provided
    rag.reformulate_query("y?", history, seen_titles=["Ikigai", "CNV"])
    # Current contract: seen_titles are NOT injected in the prompt
    assert "Ikigai" not in captured["prompt"]
    assert "CNV" not in captured["prompt"]


# ── Retry / timeout tests (2026-04-30) ───────────────────────────────────────


def _make_fake_resp(content: str):
    class _Msg:
        def __init__(self, c): self.content = c
    class _Resp:
        def __init__(self, c): self.message = _Msg(c)
    return _Resp(content)


def test_reformulate_retries_once_on_timeout(monkeypatch):
    """Primer timeout → 1 retry con 3s sleep → éxito en el segundo intento.

    Fix 2026-04-30: 67 timeouts de cold-load de qwen2.5:3b en 48h.
    El retry da tiempo al cold-load y convierte la mayoría en éxitos.
    """
    import rag

    calls = {"n": 0}
    slept = {"total": 0.0}

    def _fake_sleep(s):
        slept["total"] += s

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("timed out")
        return _make_fake_resp("pregunta reformulada")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    import rag as _rag_mod
    # Patch time.sleep inside the rag module namespace
    import unittest.mock as mock
    with mock.patch("time.sleep", side_effect=_fake_sleep):
        history = [{"role": "user", "content": "pregunta anterior"}]
        result = rag.reformulate_query("y eso?", history)

    assert result == "pregunta reformulada"
    assert calls["n"] == 2, "debe haber exactamente 2 intentos"
    assert slept["total"] >= 3.0, "debe dormir >= 3s entre intentos"


def test_reformulate_retries_once_on_503(monkeypatch):
    """HTTP 503 'server busy' → 1 retry → éxito.

    Fix 2026-04-30: 32 errores 503 en 48h por OLLAMA_MAX_LOADED_MODELS=2.
    """
    import rag

    calls = {"n": 0}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("server busy, please try again. maximum pending requests exceeded (status code: 503)")
        return _make_fake_resp("reformulada ok")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        history = [{"role": "user", "content": "algo"}]
        result = rag.reformulate_query("y?", history)

    assert result == "reformulada ok"
    assert calls["n"] == 2


def test_reformulate_degrades_after_all_failures(monkeypatch):
    """Todos los intentos fallan → devuelve la pregunta original sin crashear.

    2026-05-01: subimos de 2 a 4 intentos (1 + 3 retries con backoff
    exponencial 3s/6s/12s) tras un burst de 32×503 cuando online-tune
    catchup disparó el eval saturando Ollama. Total worst-case wait
    21s, dentro del cap de 60s del retrieve() caller.
    """
    import rag

    calls = {"n": 0}
    logged = {"exc": None}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        raise Exception("timed out")

    def _fake_silent_log(where, exc, **kw):
        logged["exc"] = exc

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_silent_log", _fake_silent_log)
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        history = [{"role": "user", "content": "contexto"}]
        result = rag.reformulate_query("pregunta original", history)

    assert result == "pregunta original"
    # 1 intento inicial + 3 retries con backoff exponencial = 4 calls
    assert calls["n"] == 4, "exactamente 4 intentos (no loop infinito)"
    assert logged["exc"] is not None, "debe loguear la excepción del último intento"


def test_reformulate_succeeds_on_third_retry(monkeypatch):
    """Si el 3er retry pega → la query reformulada se devuelve.

    Cubre el caso donde Ollama tarda más en liberar slot — los primeros
    3 intentos pegan 503 pero el 4º ya tiene slot libre. El backoff
    exponencial (3+6+12=21s acumulado) suele alcanzar.
    """
    import rag

    calls = {"n": 0}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] < 4:
            raise Exception("server busy, please try again. maximum pending requests exceeded (status code: 503)")
        return _make_fake_resp("reformulada ok tras 3 retries")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        history = [{"role": "user", "content": "algo"}]
        result = rag.reformulate_query("y?", history)

    assert result == "reformulada ok tras 3 retries"
    assert calls["n"] == 4


def test_reformulate_no_retry_without_history(monkeypatch):
    """Sin historial → short-circuit, ningún LLM call, sin retry."""
    import rag

    calls = {"n": 0}

    def _fake_chat(**kwargs):
        calls["n"] += 1

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    result = rag.reformulate_query("standalone query", [])
    assert result == "standalone query"
    assert calls["n"] == 0
