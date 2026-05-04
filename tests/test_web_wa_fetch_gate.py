"""
Source-level contract for the WhatsApp fetch gate in /api/chat (web/server.py).

Pre-2026-04-22 the endpoint unconditionally submitted _fetch_whatsapp_unread
to a ThreadPoolExecutor at the top of every request — a 25-180ms SQLite
round-trip to the WhatsApp bridge's messages.db. Profiling showed ~70% of
web queries didn't mention WhatsApp at all and the result was discarded.

The gate is a plain regex on `question` (sub-microsecond) computed once at
the start of the handler. Submit + wait branches both respect the flag, so
a non-WA query now skips the I/O entirely.

These are source-level assertions — the web endpoint is a StreamingResponse
driven by many async layers, hard to end-to-end test without real ollama /
retrieve. The invariant we actually care about is "the submit is behind an
`if _wa_in_query`" and "the wait branch tolerates _wa_future is None".
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_PY = (ROOT / "web" / "server.py").read_text(encoding="utf-8")


def test_wa_fetch_has_gate_variable():
    """`_wa_in_query` must be a single regex computed once per request."""
    assert "_wa_in_query = bool(" in SERVER_PY, (
        "expected `_wa_in_query = bool(re.search(...))` — the gate"
        " variable that controls whether the WA SQLite fetch dispatches"
    )


def test_wa_fetch_submit_is_conditional():
    """The executor submit must be guarded by `if _wa_in_query`."""
    idx = SERVER_PY.find("_wa_in_query = bool(")
    assert idx >= 0
    # Look 2500 chars forward for the guarded submit
    block = SERVER_PY[idx : idx + 2500]
    assert "if _wa_in_query:" in block, (
        "the ThreadPoolExecutor submit for _fetch_whatsapp_unread must"
        " be guarded by `if _wa_in_query:`"
    )
    assert "_wa_executor.submit(_fetch_whatsapp_unread" in block


def test_wa_wait_tolerates_none_future():
    """The wait branch must treat `_wa_future is None` as empty list —
    not crash with AttributeError."""
    # Find the wait block
    idx = SERVER_PY.find("_t_wa_wait_start = time.perf_counter()")
    assert idx >= 0, "wait instrumentation missing"
    block = SERVER_PY[idx : idx + 1500]
    assert "if _wa_future is None:" in block, (
        "expected `if _wa_future is None:` guard in the wait branch"
    )


def test_wa_injection_still_gated_by_query_match():
    """Even if WA data came back by accident (e.g. cached), the injection
    block must still require `_wa_in_query` — no phantom WhatsApp context
    bleeding into a query about something else."""
    idx = SERVER_PY.find("if wa_recent and _wa_in_query:")
    assert idx >= 0, (
        "expected `if wa_recent and _wa_in_query:` gate on the WA"
        " context injection"
    )


def test_gate_regex_matches_expected_patterns():
    """The regex itself must match the common triggers — sanity check
    that we didn't accidentally narrow it to a single word.

    Moved from inline re.search(...) to the module-level constant
    _WA_INTENT_RE (2026-05-04 perf fix) — we import the compiled object
    directly instead of parsing source to extract the pattern string.
    """
    import importlib
    import sys

    # Import server module to get the compiled constant
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    server = importlib.import_module("web.server")
    compiled = server._WA_INTENT_RE

    # Positive matches: all of these mention WhatsApp meaningfully
    for positive in [
        "qué dijo María por WhatsApp",
        "último mensaje de Juan",
        "WA pendiente",
        "chat de Astor",
        "últimos chat del grupo",
    ]:
        assert compiled.search(positive), f"regex missed: {positive!r}"

    # Negative matches: none of these should fire the WA fetch
    for negative in [
        "qué clima hace hoy",
        "qué tengo esta semana",
        "cómo configuro ollama",
        "resumime la nota de finanzas",
    ]:
        assert not compiled.search(negative), f"regex false positive: {negative!r}"


def test_regex_import_is_present():
    """`re` must be imported at module level (not in the function, to avoid
    re-importing on every request)."""
    assert "\nimport re\n" in SERVER_PY or "\nimport re,\n" in SERVER_PY or "\nfrom re import " in SERVER_PY, (
        "`re` module should be imported at top of web/server.py"
    )
