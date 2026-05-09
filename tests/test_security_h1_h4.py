"""Tests for security bugs H-1 (path traversal) y H-4 (AppleScript injection).

H-1: `_default_note_opener` aceptaba paths absolutos (`/etc/passwd`) o con
`..` que escapaban del VAULT_PATH. macOS `open` los abre igual → primitivo
de lectura/exec arbitraria. Reachable desde tool calls del LLM y `/open`.

H-4: `_create_reminder` y `_create_calendar_event` solo escapaban backslash
y comilla doble. Un `name` con `\\nend tell\\ndo shell script "rm -rf ~"`
inyectaba sentencias AppleScript. Fix: validar con allowlist
`_APPLESCRIPT_SAFE_RE` via `_sanitize_applescript_string` (mismo patrón ya
usado en el path de búsqueda de contactos).
"""

from __future__ import annotations

import rag


# ── H-1 — path traversal en _default_note_opener ─────────────────────────


def test_h1_relative_traversal_blocks_open(monkeypatch):
    """`../../etc/passwd` debe abortar antes de invocar `open`."""
    calls: list[list[str]] = []
    logged: list[tuple[str, dict]] = []

    class _FakeSubprocess:
        @staticmethod
        def run(cmd, check=False):  # noqa: ARG004
            calls.append(cmd)

    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", _FakeSubprocess.run)
    monkeypatch.setattr(
        rag,
        "_silent_log",
        lambda where, payload, **_: logged.append((where, payload)),
    )

    rag._default_note_opener("../../etc/passwd")

    assert calls == [], "open() no debería haberse invocado para path con .."
    assert logged, "_silent_log debería haber registrado el intento"
    assert logged[0][0] == "note_opener_traversal"
    assert logged[0][1]["path"] == "../../etc/passwd"


def test_h1_absolute_path_blocks_open(monkeypatch):
    """Path absoluto `/etc/passwd` debe abortar antes de invocar `open`."""
    calls: list[list[str]] = []
    logged: list[tuple[str, dict]] = []

    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", lambda cmd, check=False: calls.append(cmd))  # noqa: ARG005
    monkeypatch.setattr(
        rag,
        "_silent_log",
        lambda where, payload, **_: logged.append((where, payload)),
    )

    rag._default_note_opener("/etc/passwd")

    assert calls == [], "open() no debería haberse invocado para path absoluto"
    assert logged and logged[0][0] == "note_opener_traversal"
    assert logged[0][1]["path"] == "/etc/passwd"


def test_h1_legit_relative_path_passes(monkeypatch, tmp_path):
    """Un path legítimo dentro del vault sí dispara `open`."""
    calls: list[list[str]] = []
    logged: list[tuple[str, dict]] = []

    # Vault en tmp_path con un archivo válido.
    note = tmp_path / "00-Inbox" / "test.md"
    note.parent.mkdir(parents=True)
    note.write_text("# test", encoding="utf-8")

    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", lambda cmd, check=False: calls.append(cmd))  # noqa: ARG005
    monkeypatch.setattr(
        rag,
        "_silent_log",
        lambda where, payload, **_: logged.append((where, payload)),
    )

    rag._default_note_opener("00-Inbox/test.md")

    assert len(calls) == 1, "open() debería haber sido invocado"
    assert calls[0][0] == "open"
    assert str(note) in calls[0][1]
    assert logged == [], "no debería loggear traversal en path válido"


# ── H-4 — AppleScript newline injection ──────────────────────────────────


def test_h4_sanitize_blocks_newline_in_name():
    """`_sanitize_applescript_string` rechaza nombres con `\\n` embebido."""
    payload = (
        'foo\nend tell\ndo shell script "rm -rf ~"\n'
        'tell application "Reminders"'
    )
    assert rag._sanitize_applescript_string(payload) is None


def test_h4_sanitize_blocks_carriage_return():
    """También bloquea CR (`\\r`) — algunos parsers AS lo tratan como newline."""
    assert rag._sanitize_applescript_string("foo\rbar") is None


def test_h4_sanitize_blocks_backtick():
    """Backtick (shell-out via `do shell script`) no está en allowlist."""
    assert rag._sanitize_applescript_string("foo`whoami`") is None


def test_h4_sanitize_accepts_legit_name():
    """Nombres normales con tildes y apóstrofes pasan."""
    assert rag._sanitize_applescript_string("Comprar pan") == "Comprar pan"
    assert rag._sanitize_applescript_string("María's birthday") == "María's birthday"


def test_h4_create_reminder_rejects_injection(monkeypatch):
    """`_create_reminder` aborta sin invocar osascript ante un name malicioso."""
    osascript_calls: list[str] = []
    logged: list[tuple[str, dict]] = []

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda script, timeout=None: osascript_calls.append(script) or "")  # noqa: ARG005
    monkeypatch.setattr(
        rag,
        "_silent_log",
        lambda where, payload, **_: logged.append((where, payload)),
    )

    malicious = (
        'foo"\nend tell\ndo shell script "rm -rf ~"\n'
        'tell application "Reminders"\n'
    )
    ok, err = rag._create_reminder(malicious)

    assert ok is False
    assert "no permitidos" in err
    assert osascript_calls == [], "osascript NO debería haberse invocado"
    assert any(w == "applescript_injection_blocked" for w, _ in logged)


def test_h4_create_calendar_event_rejects_injection(monkeypatch):
    """`_create_calendar_event` aborta ante un title con newline."""
    from datetime import datetime, timedelta

    osascript_calls: list[str] = []
    logged: list[tuple[str, dict]] = []

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda script, timeout=None: osascript_calls.append(script) or "")  # noqa: ARG005
    monkeypatch.setattr(
        rag,
        "_silent_log",
        lambda where, payload, **_: logged.append((where, payload)),
    )
    # Forzar que el path EventKit (objc) no se use — simulamos ImportError.
    import builtins
    real_import = builtins.__import__

    def _no_objc(name, *args, **kwargs):
        if name == "objc" or name.startswith("Foundation"):
            raise ImportError("simulated: forcing AppleScript fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_objc)

    start = datetime.now()
    end = start + timedelta(hours=1)
    malicious = 'evt"\nend tell\ndo shell script "echo pwn"\ntell application "Calendar"\n'
    ok, err = rag._create_calendar_event(malicious, start, end)

    assert ok is False
    assert "no permitidos" in err
    assert osascript_calls == [], "osascript NO debería haberse invocado"
    assert any(w == "applescript_injection_blocked" for w, _ in logged)
