"""Tests para `rag.integrations.apple_mail` — leaf ETL de Apple Mail.

Surfaces cubiertas:
- `_fetch_mail_unread(max_items)` — mensajes no leídos en últimas 36h
  via `osascript`. Parsea pipe-delimited output, marca VIPs, ordena
  VIPs primero, trunca a `max_items`.
- Invariants: silent-fail en disabled / osascript timeout / output
  vacío.

Mocking strategy:
- Las helpers del módulo `rag` (`_apple_enabled`, `_osascript`,
  `_load_mail_vips`, `_strip_html_to_preview`, `_is_vip_sender`) se
  importan dentro del cuerpo de la función para evitar circular import.
  Por eso testeamos el comportamiento monkeypatcheando en `rag.<func>`
  (no en el módulo `apple_mail`) — ahí es donde el `from rag import ...`
  resuelve.
- `OBSIDIAN_RAG_NO_APPLE=1` por default desde el conftest autouse;
  los tests que necesitan el path "real" lo desetean explícitamente.

NO toca el filesystem ni Mail.app real — todo via mocks.
"""
from __future__ import annotations

import pytest

from rag.integrations import apple_mail as mail_mod


# ── Disabled mode ───────────────────────────────────────────────────────────


def test_fetch_mail_unread_returns_empty_when_apple_disabled(monkeypatch):
    """Cuando `OBSIDIAN_RAG_NO_APPLE=1` (autouse en conftest), el helper
    short-circuitea SIN llamar osascript. Verificamos con un sentinel
    boomy que NO se invoque."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    monkeypatch.setattr(
        rag, "_osascript",
        lambda *_a, **_kw: pytest.fail("osascript no debió llamarse"),
    )
    out = mail_mod._fetch_mail_unread(max_items=10)
    assert out == []


# ── osascript empty / timeout ────────────────────────────────────────────────


def test_fetch_mail_unread_returns_empty_on_osascript_timeout(monkeypatch):
    """Cuando `_osascript` timeoutea / falla, devuelve "" → output vacío
    y la función devuelve []."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: "")
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: set())
    monkeypatch.setattr(rag, "_strip_html_to_preview",
                        lambda s, cap=200: s[:cap])
    monkeypatch.setattr(rag, "_is_vip_sender", lambda s, vips: False)

    out = mail_mod._fetch_mail_unread(max_items=10)
    assert out == []


def test_fetch_mail_unread_returns_empty_on_malformed_lines(monkeypatch):
    """Si todas las lines del osascript output son sub-2-pipes (ej.
    osascript devolvió texto random), la función devuelve []."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(
        rag, "_osascript",
        lambda *_a, **_kw: "esto no tiene pipes\notro line\n",
    )
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: set())
    monkeypatch.setattr(rag, "_strip_html_to_preview", lambda s, cap=200: s)
    monkeypatch.setattr(rag, "_is_vip_sender", lambda s, vips: False)

    out = mail_mod._fetch_mail_unread(max_items=10)
    assert out == []


# ── Happy path ──────────────────────────────────────────────────────────────


def test_fetch_mail_unread_parses_pipe_delimited_output(monkeypatch):
    """El osascript script emite `subject|sender|received|body\\n` por
    mensaje. La función parsea cada line en un dict + strip HTML del
    body."""
    import rag
    fake_output = (
        "Hola Fer|Juan Perez <juan@ejemplo.com>|sábado 26 abril 2026 10:00:00|"
        "<p>Mensaje con HTML</p>\n"
        "Reunión|Maria Lopez <maria@trabajo.com>|sábado 26 abril 2026 11:30:00|"
        "Cuerpo plano sin HTML\n"
    )

    # Capturá el original ANTES de monkeypatchear — sino la lambda
    # hace recursion infinita llamándose a sí misma.
    real_strip = rag._strip_html_to_preview

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: fake_output)
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: set())
    monkeypatch.setattr(
        rag, "_strip_html_to_preview",
        lambda s, cap=200: real_strip(s, cap=cap),
    )
    monkeypatch.setattr(rag, "_is_vip_sender", lambda s, vips: False)

    out = mail_mod._fetch_mail_unread(max_items=10)
    assert len(out) == 2

    msg1 = out[0]
    assert msg1["subject"] == "Hola Fer"
    assert "juan@ejemplo.com" in msg1["sender"]
    assert msg1["received"].startswith("sábado")
    # HTML stripped por la helper real.
    assert "<p>" not in msg1["body_preview"]
    assert "Mensaje con HTML" in msg1["body_preview"]
    assert msg1["is_vip"] is False

    msg2 = out[1]
    assert msg2["subject"] == "Reunión"
    assert "trabajo.com" in msg2["sender"]


# ── VIP ordering ────────────────────────────────────────────────────────────


def test_fetch_mail_unread_sorts_vips_to_top_before_truncation(monkeypatch):
    """VIPs van arriba aunque hayan llegado más tarde. La truncation
    a `max_items` ocurre DESPUÉS del sort, así que VIPs sobreviven el
    cap."""
    import rag

    fake_output = (
        "Spam 1|spam@x.com|domingo|cuerpo\n"
        "Spam 2|spam@y.com|domingo|cuerpo\n"
        "Importante|jefe@empresa.com|domingo|cuerpo VIP\n"
        "Spam 3|spam@z.com|domingo|cuerpo\n"
    )

    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: fake_output)
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: {"jefe"})
    monkeypatch.setattr(rag, "_strip_html_to_preview", lambda s, cap=200: s.strip())
    monkeypatch.setattr(
        rag, "_is_vip_sender",
        lambda sender, vips: any(v.lower() in sender.lower() for v in vips),
    )

    # max_items=2 → de los 4 mensajes, mantenemos VIP + 1 spam.
    out = mail_mod._fetch_mail_unread(max_items=2)
    assert len(out) == 2
    # VIP en primer lugar.
    assert out[0]["subject"] == "Importante"
    assert out[0]["is_vip"] is True
    # Segundo es algún spam (no-VIP).
    assert out[1]["is_vip"] is False


# ── max_items truncation ────────────────────────────────────────────────────


def test_fetch_mail_unread_caps_to_max_items(monkeypatch):
    """Más mensajes que `max_items` → truncación al final."""
    import rag
    lines = "\n".join([
        f"Subject {i}|user{i}@x.com|domingo|body {i}"
        for i in range(20)
    ]) + "\n"
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: lines)
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: set())
    monkeypatch.setattr(rag, "_strip_html_to_preview", lambda s, cap=200: s)
    monkeypatch.setattr(rag, "_is_vip_sender", lambda s, vips: False)

    out = mail_mod._fetch_mail_unread(max_items=5)
    assert len(out) == 5


# ── Skip subject vacío ──────────────────────────────────────────────────────


def test_fetch_mail_unread_skips_empty_subjects(monkeypatch):
    """Lines con subject vacío (raro pero ocurre con drafts / bounces)
    se filtran ANTES de incluirse en el output."""
    import rag
    fake_output = (
        "|sender@x.com|domingo|sin subject\n"
        "OK|sender2@x.com|domingo|cuerpo\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: fake_output)
    monkeypatch.setattr(rag, "_load_mail_vips", lambda: set())
    monkeypatch.setattr(rag, "_strip_html_to_preview", lambda s, cap=200: s)
    monkeypatch.setattr(rag, "_is_vip_sender", lambda s, vips: False)

    out = mail_mod._fetch_mail_unread(max_items=10)
    assert len(out) == 1
    assert out[0]["subject"] == "OK"


# ── _MAIL_SCRIPT contract ───────────────────────────────────────────────────


def test_mail_script_uses_unified_inbox_and_36h_window():
    """Sanity check del AppleScript embedded: un `repeat with` sobre el
    `inbox` unificado, lookback de 36 hours, body cap 600. Si alguien
    cambia esto sin actualizar el comentario del módulo, el test
    falla."""
    script = mail_mod._MAIL_SCRIPT
    assert "messages of inbox" in script
    assert "(36 * hours)" in script
    assert "text 1 thru 600" in script
    assert "read status is false" in script
