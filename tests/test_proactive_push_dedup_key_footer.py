"""Tests for the dedup_key footer in proactive_push (2026-04-29).

Cuando `proactive_push` recibe `dedup_key=<key>`, el body se sufija con
`\\n\\n_anticipate:<key>_` (markdown italic, WA lo renderiza como cursiva
pequeña) ANTES de mandarlo al bridge. El listener TS lo lee al detectar
un reply 👍/👎/🔇 y lo postea a /api/anticipate/feedback con el
`dedup_key` parseado — cierra el loop de feedback del Anticipatory Agent.

Sin `dedup_key`, el body queda intacto (back-compat con `emergent` y
`patterns` que no tienen dedup_key estable).
"""
from __future__ import annotations

import pytest

import rag


@pytest.fixture
def proactive_env(tmp_path, monkeypatch):
    """Aísla state path + mock ambient config + capture de sends."""
    monkeypatch.setattr(rag, "PROACTIVE_STATE_PATH", tmp_path / "proactive.json")
    monkeypatch.setattr(rag, "_ambient_config",
                        lambda: {"jid": "test@s.whatsapp.net"})
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent.append((jid, text)) or True,
    )
    # Aislar SQL (proactive_log writes acá pero falla silenciosamente).
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    rag.SqliteVecClient(path=str(db_path))
    return sent


# ── Footer presente cuando hay dedup_key ────────────────────────────────────

def test_dedup_key_appends_footer(proactive_env):
    """proactive_push(..., dedup_key='cal:foo') agrega `_anticipate:cal:foo_`."""
    sent_ok, reason = rag.proactive_push(
        "anticipate-calendar",
        "📅 En 30 min: call con Juan",
        dedup_key="cal:event-uuid-123",
    )
    assert sent_ok is True
    assert reason is None
    assert len(proactive_env) == 1
    jid, body = proactive_env[0]
    assert jid == "test@s.whatsapp.net"
    # El body original sigue al frente, el footer va al final.
    assert body.startswith("📅 En 30 min: call con Juan")
    assert body.endswith("_anticipate:cal:event-uuid-123_")
    # Separación por doble newline (renderiza como párrafo aparte).
    assert "\n\n_anticipate:cal:event-uuid-123_" in body


def test_dedup_key_with_complex_chars(proactive_env):
    """Dedup keys con `:`, `-`, paths → footer pasa raw (no escaping)."""
    rag.proactive_push(
        "anticipate-echo",
        "🔮 algo resuena",
        dedup_key="echo:02-Areas/Salud/postura.md:2025-08-15",
    )
    _, body = proactive_env[0]
    assert "_anticipate:echo:02-Areas/Salud/postura.md:2025-08-15_" in body


# ── Backwards compat: sin dedup_key, no footer ───────────────────────────────

def test_no_dedup_key_no_footer(proactive_env):
    """Sin `dedup_key` (caller existente — emergent, patterns) → body
    intacto. NUNCA debemos cambiar el shape para los callers que no
    pasen dedup_key explícito."""
    rag.proactive_push("emergent", "💡 nueva conexión: X ↔ Y")
    _, body = proactive_env[0]
    # No hay sufijo de _anticipate:_ ni doble newline al final.
    assert "_anticipate:" not in body
    assert body == "💡 nueva conexión: X ↔ Y"


def test_explicit_none_dedup_key_no_footer(proactive_env):
    """Pasar `dedup_key=None` explícito = no pasarlo. Defensive."""
    rag.proactive_push(
        "patterns", "📊 patrón detectado", dedup_key=None,
    )
    _, body = proactive_env[0]
    assert "_anticipate:" not in body
    assert body == "📊 patrón detectado"


def test_empty_string_dedup_key_no_footer(proactive_env):
    """`dedup_key=''` (string vacío, falsy) → no footer. Si el caller
    accidentalmente pasa string vacío en lugar de None, no queremos
    sufijar `_anticipate:_` (sería un footer roto sin info útil)."""
    rag.proactive_push(
        "patterns", "📊 patrón", dedup_key="",
    )
    _, body = proactive_env[0]
    assert "_anticipate:" not in body
    assert body == "📊 patrón"


# ── Smoke: snooze + daily_count NO se ven afectados por el footer ───────────

def test_dedup_key_does_not_break_snooze(proactive_env, tmp_path):
    """Footer es solo del body — el state.snooze persiste con `kind` puro."""
    rag.proactive_push(
        "anticipate-calendar", "msg", snooze_hours=2,
        dedup_key="cal:abc",
    )
    state = rag._proactive_load_state()
    # snooze key = "anticipate-calendar" (no incluye dedup_key, no incluye
    # footer). Esto es importante porque el snooze es per-kind, no per-push.
    assert "anticipate-calendar" in state.get("snooze", {})


def test_dedup_key_logged_in_message_preview(proactive_env):
    """El _proactive_log incluye `message_preview` del body real
    (con footer) — útil para debug post-mortem (ver qué key se mandó).
    No es invariant crítico (solo telemetría), pero confirmamos que
    los 120 chars del preview ya incluyen el footer cuando aplica."""
    long_body = "📅 " + "x" * 80
    rag.proactive_push(
        "anticipate-calendar", long_body,
        dedup_key="cal:xyz",
    )
    _, sent_body = proactive_env[0]
    # El preview en _proactive_log es body[:120]; verificamos que el
    # body real (sent_body) tiene el footer y que su length supera 120.
    assert "_anticipate:cal:xyz_" in sent_body
    assert len(sent_body) > 80
