"""Tests for `propose_whatsapp_send_contact_card` chat tool (2026-04-25).

Cubre:

1. `_render_contact_card_for_whatsapp` — name + multi-phone + email +
   address render en formato `📇 / 📞 / ✉️ / 📍` con label opcional.
2. `_resolve_contact_target_vault` — frontmatter scan, ambiguous (>1 match),
   target_not_found, lista vs scalar, skip private folder/frontmatter.
3. `_resolve_contact_target` — Apple Contacts gana si hay match útil,
   fallback a vault si no hay phones/emails Apple, strip leading `@`.
4. `propose_whatsapp_send_contact_card` — happy path Apple, fallback
   vault, ambiguous candidates, fields filter, recipient inexistente,
   sin datos resolubles, render shape JSON.
5. Tool registry — CHAT_TOOLS + PROPOSAL_TOOL_NAMES, NO PARALLEL_SAFE.
"""
from __future__ import annotations

import json

import rag
from web import tools as _tools


# ── 1. _render_contact_card_for_whatsapp ───────────────────────────────────


def test_render_card_full():
    out = rag._render_contact_card_for_whatsapp(
        name="Veterinaria Centro",
        phones=[
            {"label": "Casa", "value": "+54 9 342 1234567"},
            {"label": "", "value": "+54 9 342 9876543"},
        ],
        emails=["info@centro.com"],
        addresses=["Av. San Martín 2345"],
        fields_filter=None,
    )
    assert "📇 *Veterinaria Centro*" in out
    assert "📞 Casa: +54 9 342 1234567" in out
    assert "📞 +54 9 342 9876543" in out
    assert "✉️ info@centro.com" in out
    assert "📍 Av. San Martín 2345" in out


def test_render_card_filters_to_phone_only():
    out = rag._render_contact_card_for_whatsapp(
        name="Plomero", phones=[{"label": "", "value": "+54 9 11 1111-1111"}],
        emails=["x@y.com"], addresses=["Calle Falsa 123"],
        fields_filter=["phone"],
    )
    assert "📞" in out
    assert "✉️" not in out
    assert "📍" not in out


def test_render_card_skips_empty_values():
    out = rag._render_contact_card_for_whatsapp(
        name="X",
        phones=[{"label": "", "value": ""}, {"label": "", "value": "+5491155"}],
        emails=["", "foo@bar"],
        addresses=["", "valid"],
        fields_filter=None,
    )
    assert "📞 +5491155" in out
    assert "📞 \n" not in out and "📞 " * 2 not in out
    assert "✉️ foo@bar" in out
    assert "📍 valid" in out


def test_render_card_no_data_returns_empty():
    """Sin phones/emails/addresses, devolvemos "" (no name-only) para que el
    caller dispare `no_data_after_filter` en la card."""
    out = rag._render_contact_card_for_whatsapp(
        name="Solo Nombre", phones=[], emails=[], addresses=[],
        fields_filter=None,
    )
    assert out == ""


# ── 2. _resolve_contact_target_vault ───────────────────────────────────────


def test_vault_target_single_match(monkeypatch, tmp_path):
    note = tmp_path / "Plomero Juan.md"
    note.write_text(
        "---\nphone: '+54 9 342 1234567'\nemail: juan@plomeria.com\n---\n# Plomero",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("plomero")
    assert result["error"] is None
    assert result["source"] == "vault"
    assert result["name"] == "Plomero Juan"
    assert len(result["phones"]) == 1
    assert result["phones"][0]["value"] == "+54 9 342 1234567"
    assert result["emails"] == ["juan@plomeria.com"]


def test_vault_target_ambiguous(monkeypatch, tmp_path):
    (tmp_path / "Plomero A.md").write_text(
        "---\nphone: '+5491111'\n---\nbody", encoding="utf-8")
    (tmp_path / "Plomero B.md").write_text(
        "---\nphone: '+5492222'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("plomero")
    assert result["error"] == "ambiguous"
    assert len(result["candidates"]) == 2
    names = {c["name"] for c in result["candidates"]}
    assert names == {"Plomero A", "Plomero B"}


def test_vault_target_not_found(monkeypatch, tmp_path):
    (tmp_path / "Otra cosa.md").write_text(
        "---\nphone: '+5491111'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("inexistente")
    assert result["error"] == "target_not_found"
    assert result["candidates"] == []


def test_vault_target_list_phones(monkeypatch, tmp_path):
    (tmp_path / "Multi.md").write_text(
        "---\nphones:\n  - '+5491111'\n  - '+5492222'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("Multi")
    assert result["error"] is None
    assert len(result["phones"]) == 2


def test_vault_target_skips_private_folder(monkeypatch, tmp_path):
    priv = tmp_path / "99-Private"
    priv.mkdir()
    (priv / "Secreto.md").write_text(
        "---\nphone: '+5491111'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("Secreto")
    assert result["error"] == "target_not_found"


def test_vault_target_skips_private_frontmatter(monkeypatch, tmp_path):
    (tmp_path / "Diario.md").write_text(
        "---\nprivate: true\nphone: '+5491111'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target_vault("Diario")
    assert result["error"] == "target_not_found"


def test_vault_target_empty_query():
    result = rag._resolve_contact_target_vault("")
    assert result["error"] == "empty_query"


# ── 3. _resolve_contact_target (Apple → vault chain) ───────────────────────


def test_apple_wins_over_vault(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Plomero (Apple)",
        "phones": ["+5491155555555"],
        "emails": [],
        "birthday": "",
    })
    # Vault has a different match — should NOT be reached
    (tmp_path / "Plomero Vault.md").write_text(
        "---\nphone: '+5499999999'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target("plomero")
    assert result["source"] == "apple"
    assert result["name"] == "Plomero (Apple)"
    assert result["phones"][0]["value"] == "+5491155555555"


def test_vault_fallback_when_apple_empty(monkeypatch, tmp_path):
    # Apple returns nothing
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    (tmp_path / "Plomero.md").write_text(
        "---\nphone: '+5499999999'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target("plomero")
    assert result["source"] == "vault"
    assert result["phones"][0]["value"] == "+5499999999"


def test_vault_fallback_when_apple_has_no_phone_or_email(monkeypatch, tmp_path):
    """Apple devuelve match pero sin phones/emails utiles → fallback al vault."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "X", "phones": [], "emails": [], "birthday": "",
    })
    (tmp_path / "X.md").write_text(
        "---\nphone: '+5491100'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target("X")
    assert result["source"] == "vault"


def test_resolve_strips_leading_at_sign(monkeypatch, tmp_path):
    """`@Plomero` (LLM hábito wikilink) debe limpiarse antes del lookup."""
    captured = {}
    def _fake_fetch(stem, email=None, canonical=None):
        captured["canonical"] = canonical
        return {
            "full_name": "Plomero", "phones": ["+5491155"], "emails": [],
            "birthday": "",
        }
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    result = rag._resolve_contact_target("@Plomero")
    assert result["error"] is None
    assert "@" not in (captured.get("canonical") or "")


def test_resolve_empty_query():
    result = rag._resolve_contact_target("")
    assert result["error"] == "empty_query"
    result2 = rag._resolve_contact_target("@@")
    assert result2["error"] == "empty_query"


# ── 4. propose_whatsapp_send_contact_card ──────────────────────────────────


def _stub_recipient_ok(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Sole",
        "phones": ["+5491155555555"],
        "emails": [],
        "birthday": "",
    })


def test_propose_card_apple_target_happy(monkeypatch, tmp_path):
    """Recipient Y target ambos resueltos vía Apple Contacts."""
    # Both calls go through _fetch_contact; we use a switch
    calls = {"n": 0}
    def _fake_fetch(stem, email=None, canonical=None):
        calls["n"] += 1
        if "sole" in (canonical or stem or "").lower():
            return {"full_name": "Sole", "phones": ["+5491155555555"],
                    "emails": [], "birthday": ""}
        return {"full_name": "Plomero Carlos",
                "phones": ["+54 9 342 1234567"],
                "emails": ["carlos@plomeria.com"],
                "birthday": ""}
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card("Sole", "Plomero")
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_message"
    assert payload["needs_clarification"] is True
    fields = payload["fields"]
    assert fields["error"] is None
    assert fields["target_source"] == "apple"
    assert fields["target_name"] == "Plomero Carlos"
    msg = fields["message_text"]
    assert "📇 *Plomero Carlos*" in msg
    assert "+54 9 342 1234567" in msg
    assert "carlos@plomeria.com" in msg


def test_propose_card_vault_fallback(monkeypatch, tmp_path):
    """Apple no encuentra → resuelve por frontmatter en una nota."""
    def _fake_fetch(stem, email=None, canonical=None):
        if "sole" in (canonical or stem or "").lower():
            return {"full_name": "Sole", "phones": ["+5491155"],
                    "emails": [], "birthday": ""}
        return None  # plomero no está en Apple
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    (tmp_path / "Plomero.md").write_text(
        "---\nphone: '+54 9 342 99999'\n---\n# Plomero", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card("Sole", "plomero")
    fields = json.loads(raw)["fields"]
    assert fields["error"] is None
    assert fields["target_source"] == "vault"
    assert "+54 9 342 99999" in fields["message_text"]


def test_propose_card_ambiguous_target_returns_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: (
        {"full_name": "Sole", "phones": ["+5491155"], "emails": [], "birthday": ""}
        if "sole" in (kw.get("canonical") or "").lower() else None
    ))
    (tmp_path / "Plomero A.md").write_text(
        "---\nphone: '+5491111'\n---\nbody", encoding="utf-8")
    (tmp_path / "Plomero B.md").write_text(
        "---\nphone: '+5492222'\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card("Sole", "plomero")
    fields = json.loads(raw)["fields"]
    assert fields["error"] == "ambiguous"
    cands = fields.get("candidates") or []
    assert len(cands) == 2
    assert fields["message_text"] == ""


def test_propose_card_recipient_not_found_still_renders(monkeypatch, tmp_path):
    """Recipient no resuelve, pero el target sí → card aparece con error."""
    calls = {"n": 0}
    def _fake_fetch(stem, email=None, canonical=None):
        calls["n"] += 1
        if "inexistente" in (canonical or stem or "").lower():
            return None  # recipient no encontrado
        return {"full_name": "Plomero", "phones": ["+5491100"],
                "emails": [], "birthday": ""}
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card("Inexistente", "Plomero")
    fields = json.loads(raw)["fields"]
    assert fields["error"] == "not_found"  # recipient lookup error
    assert fields["jid"] is None
    # Pero el mensaje ya está armado con los datos del target
    assert "+5491100" in fields["message_text"]


def test_propose_card_fields_filter_applied(monkeypatch, tmp_path):
    _stub_recipient_ok(monkeypatch)
    def _fake_fetch(stem, email=None, canonical=None):
        if "sole" in (canonical or stem or "").lower():
            return {"full_name": "Sole", "phones": ["+5491155"],
                    "emails": [], "birthday": ""}
        return {"full_name": "Vet", "phones": ["+54 9 342 1111"],
                "emails": ["vet@centro.com"], "birthday": ""}
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card(
        "Sole", "Vet", fields=["phone"])
    msg = json.loads(raw)["fields"]["message_text"]
    assert "📞" in msg
    assert "✉️" not in msg
    assert "vet@centro.com" not in msg


def test_propose_card_no_data_after_filter(monkeypatch, tmp_path):
    _stub_recipient_ok(monkeypatch)
    def _fake_fetch(stem, email=None, canonical=None):
        if "sole" in (canonical or stem or "").lower():
            return {"full_name": "Sole", "phones": ["+5491155"],
                    "emails": [], "birthday": ""}
        # Target con solo email, pidieron solo phone
        return {"full_name": "Vet", "phones": [],
                "emails": ["vet@centro.com"], "birthday": ""}
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card(
        "Sole", "Vet", fields=["phone"])
    fields = json.loads(raw)["fields"]
    # Apple devolvió emails pero sin phones → resolver entrega el target
    # con source=apple + emails únicos. Aplicamos `fields=["phone"]` →
    # render filtra emails → cadena vacía → upstream marca
    # `no_data_after_filter`. La card aparece con el error visible.
    assert fields["error"] == "no_data_after_filter"
    assert fields["message_text"] == ""


def test_propose_card_target_empty_query(monkeypatch, tmp_path):
    _stub_recipient_ok(monkeypatch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_contact_card("Sole", "")
    fields = json.loads(raw)["fields"]
    assert fields["error"] == "empty_query"
    assert fields["message_text"] == ""


# ── 5. Tool registry invariants ────────────────────────────────────────────


def test_tool_registered_in_chat_tools():
    assert rag.propose_whatsapp_send_contact_card in _tools.CHAT_TOOLS
    assert "propose_whatsapp_send_contact_card" in _tools.TOOL_FNS


def test_tool_in_proposal_tool_names():
    assert "propose_whatsapp_send_contact_card" in _tools.PROPOSAL_TOOL_NAMES


def test_tool_NOT_in_parallel_safe():
    assert "propose_whatsapp_send_contact_card" not in _tools.PARALLEL_SAFE
