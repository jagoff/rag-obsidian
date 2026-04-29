"""Tests for the dedup_key footer in `_brief_push_to_whatsapp` (2026-04-29).

Cuando el daemon de briefs invoca `_brief_push_to_whatsapp(title, vault_relpath, narrative)`,
el body se sufija con `\\n\\n_brief:<vault_relpath>_` (markdown italic, WA lo
renderiza como cursiva pequeña) ANTES de mandarlo al bridge. El listener TS
lo lee al detectar un reply 👍/👎/🔇 dentro de los 30min siguientes y lo
postea a `/api/brief/feedback` con el `vault_relpath` como `dedup_key` —
cierra el loop de feedback de los briefs.

A diferencia del footer de anticipate, este NO es opcional: TODO brief
push lleva el footer (el `vault_relpath` siempre está disponible en el
caller). Mimetiza el shape de `tests/test_proactive_push_dedup_key_footer.py`.
"""
from __future__ import annotations

import pytest

import rag


@pytest.fixture
def brief_env(monkeypatch):
    """Mock ambient config + capture de sends. NO tocamos SQL — el
    push solo loguea via _ambient_log_event que escribe JSONL al disk
    fuera del scope de telemetría que importa para este test."""
    monkeypatch.setattr(
        rag, "_ambient_config",
        lambda: {"jid": "test@s.whatsapp.net"},
    )
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent.append((jid, text)) or True,
    )
    # _ambient_log_event escribe a un JSONL con path absoluto que en
    # un test puede ser RO o no-existir; lo neutralizamos.
    monkeypatch.setattr(rag, "_ambient_log_event", lambda payload: None)
    return sent


# ── Footer presente con vault_relpath del brief ─────────────────────────────

def test_morning_brief_appends_footer(brief_env):
    """Morning brief con vault_relpath canónico → footer
    `_brief:02-Areas/Briefs/...md_` al final del body."""
    ok = rag._brief_push_to_whatsapp(
        "Morning 2026-04-29",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "buen día. tenés 3 reuniones hoy y un deadline.",
    )
    assert ok is True
    assert len(brief_env) == 1
    jid, body = brief_env[0]
    assert jid == "test@s.whatsapp.net"
    # Header + body + footer, en ese orden.
    assert body.startswith("📓 *Morning 2026-04-29*")
    assert body.endswith("_brief:02-Areas/Briefs/2026-04-29-morning.md_")
    # Separación por doble newline (renderiza como párrafo aparte).
    assert "\n\n_brief:02-Areas/Briefs/2026-04-29-morning.md_" in body


def test_evening_brief_appends_footer(brief_env):
    """Evening brief con su propio vault_relpath."""
    rag._brief_push_to_whatsapp(
        "Evening 2026-04-29",
        "02-Areas/Briefs/2026-04-29-evening.md",
        "lo que hiciste hoy, lo que queda pendiente.",
    )
    _, body = brief_env[0]
    assert "_brief:02-Areas/Briefs/2026-04-29-evening.md_" in body


def test_digest_brief_appends_footer(brief_env):
    """Weekly digest con path con guiones + dígitos."""
    rag._brief_push_to_whatsapp(
        "Digest week-17",
        "02-Areas/Briefs/2026-04-week-17-digest.md",
        "esta semana cerraste 5 deadlines y abriste 3 proyectos.",
    )
    _, body = brief_env[0]
    assert body.endswith("_brief:02-Areas/Briefs/2026-04-week-17-digest.md_")


# ── Footer position + separation ─────────────────────────────────────────────

def test_footer_is_last_line(brief_env):
    """El footer queda en la ÚLTIMA línea no vacía del body. El listener
    TS asume que el footer está en su propia línea al final — si lo
    rompemos acá, el regex `extractBriefDedupKey` falla."""
    rag._brief_push_to_whatsapp(
        "Morning",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "primer línea\n\nsegunda línea\n\ntercera línea",
    )
    _, body = brief_env[0]
    # Tomamos la última línea no-vacía del body.
    last = next(
        line.strip()
        for line in reversed(body.split("\n"))
        if line.strip()
    )
    assert last == "_brief:02-Areas/Briefs/2026-04-29-morning.md_"


def test_footer_does_not_break_when_narrative_has_obsidian_links(brief_env):
    """El narrative pasa por `convert_obsidian_links()` antes de armar
    el msg — confirmamos que el footer NO sufre conversión (no contiene
    wikilinks ni paths de obsidian)."""
    rag._brief_push_to_whatsapp(
        "Morning",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "ver [[02-Areas/Productividad/Foco]] para contexto",
    )
    _, body = brief_env[0]
    # El footer debe estar al final, intacto.
    assert body.endswith("_brief:02-Areas/Briefs/2026-04-29-morning.md_")


def test_vault_relpath_with_special_chars(brief_env):
    """Paths con guiones, dígitos y números → footer pasa raw (no
    escaping). Es file-system path simple así que no esperamos `\\n`,
    `_` ni espacios — pero confirmamos."""
    rag._brief_push_to_whatsapp(
        "Brief X",
        "02-Areas/Briefs/special-2026.04.29-X.md",
        "body",
    )
    _, body = brief_env[0]
    assert "_brief:02-Areas/Briefs/special-2026.04.29-X.md_" in body


# ── Side-channel: header + body intactos ─────────────────────────────────────

def test_header_format_unchanged(brief_env):
    """El header `📓 *<title>* — \\``<vault_relpath>\\``` sigue apareciendo
    al inicio. El footer es ADITIVO, no reemplaza nada."""
    rag._brief_push_to_whatsapp(
        "Morning 2026-04-29",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "body content",
    )
    _, body = brief_env[0]
    assert body.startswith("📓 *Morning 2026-04-29* — `02-Areas/Briefs/2026-04-29-morning.md`")
    assert "body content" in body


def test_no_config_no_send_no_footer(monkeypatch):
    """Sin ambient config → la function devuelve False sin llamar al
    send. No hay nada que verificar del footer porque no hay msg."""
    monkeypatch.setattr(rag, "_ambient_config", lambda: None)
    sent = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent.append((jid, text)) or True,
    )
    ok = rag._brief_push_to_whatsapp(
        "Morning", "02-Areas/Briefs/2026-04-29-morning.md", "body",
    )
    assert ok is False
    assert len(sent) == 0
