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


def test_reformulate_no_retry_on_timeout(monkeypatch):
    """Timeout → fail-fast, sin retry, degrada a la pregunta original.

    Fix 2026-05-01 (afternoon): el blanket-retry de cualquier Exception
    multiplicaba el wait en ReadTimeout (60s × 4 attempts = 240s+ observado
    en `reform=261773ms` en chat-timing log). Para ReadTimeout retra-
    nundo NO ayuda — el helper está hung, paga otros 60s por nada.
    Sólo retra-mos en 503 ("server busy") que resuelve en <1s.
    """
    import rag

    calls = {"n": 0}
    slept = {"total": 0.0}

    def _fake_sleep(s):
        slept["total"] += s

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        raise Exception("timed out")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    import unittest.mock as mock
    with mock.patch("time.sleep", side_effect=_fake_sleep):
        history = [{"role": "user", "content": "pregunta anterior"}]
        result = rag.reformulate_query("y eso?", history)

    # Devuelve la pregunta original (degradación), 1 call sólo (no retry).
    assert result == "y eso?"
    assert calls["n"] == 1, "timeout NO debe disparar retry — fail fast"
    assert slept["total"] == 0.0, "no sleep cuando no hay retry"


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


def test_reformulate_degrades_after_503_failures(monkeypatch):
    """Todos los 503 fallan → devuelve la pregunta original sin crashear.

    2026-05-01 (afternoon): el retry policy se splitteó por tipo de error.
    Para 503 ("server busy") seguimos haciendo 4 attempts con backoff
    exponencial (1s/2s/4s, total worst-case 7s acumulado, mucho más
    barato que el viejo 21s pero suficiente — un slot de Ollama se
    libera en <1s al terminar el call que lo tenía). Para no-503
    (ReadTimeout, RemoteProtocolError, etc.) NO retra-mos, se rompe
    el loop tras la primera Exception y se degrada.
    """
    import rag

    calls = {"n": 0}
    logged = {"exc": None}

    def _fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        raise Exception("server busy, please try again. maximum pending requests exceeded (status code: 503)")

    def _fake_silent_log(where, exc, **kw):
        logged["exc"] = exc

    # _silent_log se invoca como _silent_log(where, exc) — aceptar **kw
    # defensivo. Real signature: _silent_log(where: str, exc: Exception | None).
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_silent_log", _fake_silent_log)
    # Hack: el monkeypatch de _silent_log con _fake_silent_log no respeta
    # la signature real (sólo 2 args). El módulo lo invoca con 2 args
    # positional, así que el lambda con **kw está fine.
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        history = [{"role": "user", "content": "contexto"}]
        result = rag.reformulate_query("pregunta original", history)

    assert result == "pregunta original"
    # 503 retries: 1 intento inicial + 3 retries = 4 calls totales.
    assert calls["n"] == 4, "exactamente 4 intentos para 503 (no loop infinito)"
    assert logged["exc"] is not None, "debe loguear la excepción del último intento"


def test_reformulate_no_retry_on_non_503(monkeypatch):
    """Cualquier Exception que NO sea 503 → 1 call sin retry, degrada.

    Cubre el cambio de semántica del 2026-05-01 (afternoon): el blanket-
    retry causaba reform=261s (4× 60s ReadTimeout). Ahora ReadTimeout +
    RemoteProtocolError + cualquier otra Exception salen del loop tras
    el 1er intento y devuelven la pregunta original.
    """
    import rag

    for exc_msg in ("timed out", "Server disconnected without sending a response.",
                    "Connection refused", "some other unexpected error"):
        calls = {"n": 0}

        def _fake_chat(model, messages, **kwargs):
            calls["n"] += 1
            raise Exception(exc_msg)

        monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
        monkeypatch.setattr(rag, "_silent_log", lambda *a, **k: None)
        import unittest.mock as mock
        with mock.patch("time.sleep"):
            history = [{"role": "user", "content": "contexto"}]
            result = rag.reformulate_query("pregunta original", history)

        assert result == "pregunta original", f"degrada en exc={exc_msg!r}"
        assert calls["n"] == 1, f"sin retry para exc={exc_msg!r} (calls={calls['n']})"


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
