"""Tests del clasificador de device por User-Agent + logueo en telemetría.

Contexto (2026-04-22): el sistema tiene 3 puntos de entrada:
  - Mac terminal (`rag chat` CLI → cmd=chat)
  - otra Mac (web via Safari / Chrome desktop → cmd=web + UA "Macintosh")
  - iPhone (web via Safari mobile → cmd=web + UA "iPhone")

Pre-fix: `cmd=web` para los tres últimos — imposible distinguir desktop
de móvil ni rendear distinto. Este cambio agrega:

  _classify_device(user_agent: str) -> str
    → "iphone" | "ipad" | "mac" | "linux" | "windows" | "android" | "other"

Política:
  - Allowlist-based de tokens discriminativos (no full-parse).
  - Orden de precedencia: iphone > ipad > android > mac > linux/windows >
    other. Porque iPadOS envía UA con "Macintosh" desde 2019 (iPad en
    modo desktop), y queremos detectar iPad correctamente aunque diga
    "Mac".
  - Unknown/empty → "other" (default conservative, no contamina métricas).

Escalado: el campo `device` llega a `rag_queries.extra_json` via
`log_query_event` y habilita queries como:

  SELECT json_extract(extra_json,'$.device'), AVG(t_retrieve)
  FROM rag_queries WHERE cmd='web' AND ts > datetime('now','-7 days')
  GROUP BY 1;

Desde ahí se puede decidir si el iPhone necesita un pipeline distinto
(menos chunks, respuesta más corta, etc.) con evidencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import rag


# ── iPhone ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ua", [
    # Safari iOS 17 iPhone
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    # Chrome iOS
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/118.0.5993.69 Mobile/15E148 Safari/604.1",
    # Firefox iOS
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/120.0 Mobile/15E148 Safari/605.1.15",
])
def test_classify_iphone(ua):
    assert rag._classify_device(ua) == "iphone"


# ── iPad ──────────────────────────────────────────────────────────────────


def test_classify_ipad_with_ipad_in_ua():
    """iPad clásico — UA contiene 'iPad'."""
    ua = ("Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
    assert rag._classify_device(ua) == "ipad"


def test_classify_ipad_desktop_mode_is_mac():
    """iPadOS 13+ en modo 'Request Desktop Site' envía UA de Macintosh
    sin el token 'iPad'. Indistinguible de un Mac real sin más info —
    política: lo clasificamos como mac (lo que el iPad pide). Si el
    user quiere mobile layout, puede desactivar el modo desktop."""
    ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15")
    assert rag._classify_device(ua) == "mac"


# ── Mac desktop ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ua", [
    # Safari macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) "
    "Gecko/20100101 Firefox/120.0",
])
def test_classify_mac_desktop(ua):
    assert rag._classify_device(ua) == "mac"


# ── Android ─────────────────────────────────────────────────────────────────


def test_classify_android_phone():
    ua = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36")
    assert rag._classify_device(ua) == "android"


def test_classify_android_not_linux():
    """Android UA contiene 'Linux' pero la precedencia dice android — no
    queremos clasificar un teléfono Android como desktop Linux."""
    ua = ("Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36")
    assert rag._classify_device(ua) == "android"


# ── Linux / Windows desktop ─────────────────────────────────────────────────


def test_classify_linux_desktop():
    ua = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    assert rag._classify_device(ua) == "linux"


def test_classify_windows():
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    assert rag._classify_device(ua) == "windows"


# ── Unknown / edge cases ────────────────────────────────────────────────────


def test_classify_empty_ua_is_other():
    assert rag._classify_device("") == "other"


def test_classify_none_ua_is_other():
    assert rag._classify_device(None) == "other"


def test_classify_curl_is_other():
    """Requests sin UA real (curl, python-requests, telemetry bots) caen
    a `other` — no son web chat sessions."""
    assert rag._classify_device("curl/8.1.2") == "other"
    assert rag._classify_device("python-requests/2.31.0") == "other"


def test_classify_gibberish_is_other():
    """UA inyectado/trash no clasifica — no inventa un device plausible."""
    assert rag._classify_device("xxx") == "other"


# ── Caso problemático: iPad clasica como ipad no como mac ───────────────────


def test_ipad_takes_precedence_over_mac_token():
    """El UA del iPad clásico (sin modo desktop) contiene tanto 'iPad'
    como 'Mac OS X' (CPU-model). iPad gana."""
    ua = ("Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1")
    assert rag._classify_device(ua) == "ipad"


def test_iphone_takes_precedence_over_mac_token():
    """Mismo caso con iPhone — el UA dice 'like Mac OS X' pero el primer
    token identificativo es 'iPhone'."""
    ua = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
          "AppleWebKit/605.1.15 Version/17.5 Mobile/15E148 Safari/604.1")
    assert rag._classify_device(ua) == "iphone"


# ── Cross-check: set de valores devueltos está documentado ──────────────────


def test_classifier_domain_is_closed_set():
    """Función debe devolver sólo valores del allowlist — si alguien
    agrega una branch que devuelve `"macos"` (vs `"mac"`) o cualquier
    sinónimo, los tests downstream que filtran por device rompen."""
    valid = {"iphone", "ipad", "mac", "linux", "windows", "android", "other"}
    test_uas = [
        "",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (Linux; Android 14)",
        "Mozilla/5.0 (Windows NT 10.0)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)",
        "curl/8.1.2",
    ]
    for ua in test_uas:
        assert rag._classify_device(ua) in valid, \
            f"_classify_device({ua!r}) retornó valor fuera del dominio"
