"""Tests for `propose_whatsapp_send_note` chat tool + renderer (2026-04-25).

Cubre:

1. `_render_note_for_whatsapp` — frontmatter strip, headings → bold,
   wikilinks → bare title/alias, embeds → placeholders, links, callouts,
   tasks, bullets, idempotencia aproximada.
2. `_extract_note_section` — heading match case-insensitive, fallback
   cuando no existe, respeto a niveles.
3. `_is_private_note_path` / `_is_private_note_text` — folder + suffix +
   frontmatter `private: true`.
4. `propose_whatsapp_send_note` — resolución por path explícito, por
   query semántico (top-1 / ambiguo), section extract, contacto
   inexistente, nota privada bloqueada, truncation.
5. Tool registry — registrado en CHAT_TOOLS, PROPOSAL_TOOL_NAMES, NO en
   PARALLEL_SAFE (consistente con `propose_whatsapp_send`).
"""
from __future__ import annotations

import json

import pytest

import rag
from web import tools as _tools


# ── 1. _render_note_for_whatsapp ───────────────────────────────────────────


def test_render_strips_frontmatter():
    src = "---\ntitle: foo\ntags: [a, b]\n---\nbody real"
    out = rag._render_note_for_whatsapp(src)
    assert "title: foo" not in out
    assert out.strip() == "body real"


def test_render_headings_to_bold():
    src = "# Receta\n## Ingredientes\n### Pasos"
    out = rag._render_note_for_whatsapp(src)
    assert "*Receta*" in out
    assert "*Ingredientes*" in out
    assert "*Pasos*" in out
    assert "#" not in out  # all headings converted


def test_render_wikilinks_to_bare_title_or_alias():
    src = "Ver [[Astor]] y [[Plomero|Carlos]] sobre [[Tema#Sub]]."
    out = rag._render_note_for_whatsapp(src)
    assert "[[" not in out and "]]" not in out
    assert "Astor" in out
    assert "Carlos" in out  # alias wins
    assert "Plomero" not in out  # alias displaces title
    assert "Tema" in out


def test_render_image_embed_to_placeholder():
    src = "Mirá ![[receta-pasta.png]] arriba.\nY también ![[note.md]]."
    out = rag._render_note_for_whatsapp(src)
    assert "[imagen: receta-pasta.png]" in out
    # `note.md` no es imagen → embed placeholder
    assert "[embed: note.md]" in out
    assert "![[" not in out


def test_render_markdown_image_to_placeholder():
    src = "![Logo del proyecto](./logo.png)\n![](./sin-alt.jpg)"
    out = rag._render_note_for_whatsapp(src)
    assert "[imagen: Logo del proyecto]" in out
    assert "[imagen: sin-titulo]" in out
    assert "![" not in out


def test_render_links_keep_url_and_strip_relative():
    src = "Ver [docs](https://example.com/x) y [otra](./local.md)."
    out = rag._render_note_for_whatsapp(src)
    assert "docs: https://example.com/x" in out
    assert "otra" in out
    assert "./local.md" not in out


def test_render_callout_to_emoji():
    src = "> [!note] Importante\n> el hilo va por acá\n\n> [!warning]\n> sin titulo"
    out = rag._render_note_for_whatsapp(src)
    assert "📝 *Importante*" in out
    assert "el hilo va por acá" in out
    # Callout sin titulo → emoji solo
    assert "📝" in out


def test_render_tasks_and_bullets():
    src = "- [ ] comprar\n- [x] hecho\n- item\n* bullet alt"
    out = rag._render_note_for_whatsapp(src)
    assert "☐ comprar" in out
    assert "☑ hecho" in out
    assert "• item" in out
    assert "• bullet alt" in out


def test_render_bold_strike_normalized():
    src = "**bold** y __also bold__ y ~~tachado~~ y ***both***"
    out = rag._render_note_for_whatsapp(src)
    assert "*bold*" in out
    assert "*also bold*" in out
    assert "~tachado~" in out
    assert "*both*" in out
    assert "**" not in out
    assert "~~" not in out


def test_render_strips_ocr_comments():
    src = "Caption.\n<!-- OCR: img/foo.png -->\nMore body."
    out = rag._render_note_for_whatsapp(src)
    assert "OCR:" not in out
    assert "Caption." in out
    assert "More body." in out


def test_render_idempotent_on_already_wa_text():
    """Rendering text that already looks like WA output should not mangle it.

    Idempotencia exacta no es alcanzable (el segundo pase no hace nada
    porque no hay markdown ya), pero la propiedad relevante es "no
    degrada": pasarlo dos veces da el mismo resultado que pasarlo una.
    """
    src = "*Title*\n\n• item 1\n• item 2\n\n☐ tarea\n☑ hecha"
    once = rag._render_note_for_whatsapp(src)
    twice = rag._render_note_for_whatsapp(once)
    assert once == twice


def test_render_empty_string():
    assert rag._render_note_for_whatsapp("") == ""
    assert rag._render_note_for_whatsapp("   \n  \n") == ""


# ── 2. _extract_note_section ───────────────────────────────────────────────


def test_extract_section_happy():
    src = "# Receta\n## Ingredientes\n- harina\n- agua\n## Pasos\n1. mezclar"
    out = rag._extract_note_section(src, "Ingredientes")
    assert out is not None
    assert "harina" in out
    assert "agua" in out
    # Stops at next ## of same level
    assert "Pasos" not in out
    assert "mezclar" not in out


def test_extract_section_case_insensitive_substring():
    src = "## INGREDIENTES principales\nfoo\n## Otra\nbar"
    out = rag._extract_note_section(src, "ingredientes")
    assert out is not None
    assert "foo" in out
    assert "bar" not in out


def test_extract_section_not_found():
    src = "## A\nfoo\n## B\nbar"
    assert rag._extract_note_section(src, "Inexistente") is None


# ── 3. Privacy guards ──────────────────────────────────────────────────────


def test_is_private_note_path_matches_folder():
    assert rag._is_private_note_path("99-Private/foo.md") is True
    assert rag._is_private_note_path("02-Areas/99-Private/x.md") is True
    assert rag._is_private_note_path("secrets/api-keys.md") is True
    assert rag._is_private_note_path("02-Areas/foo.md") is False


def test_is_private_note_path_matches_suffix():
    assert rag._is_private_note_path("foo.private.md") is True
    assert rag._is_private_note_path("02-Areas/x.private.md") is True
    assert rag._is_private_note_path("02-Areas/x.md") is False


def test_is_private_note_text_frontmatter_true():
    assert rag._is_private_note_text("---\nprivate: true\n---\nbody") is True
    assert rag._is_private_note_text("---\nprivate: yes\n---\nbody") is True
    assert rag._is_private_note_text("---\nprivate: false\n---\nbody") is False
    assert rag._is_private_note_text("---\ntitle: x\n---\nbody") is False
    assert rag._is_private_note_text("body sin fm") is False


# ── 4. propose_whatsapp_send_note (integration) ────────────────────────────


def _stub_contact_ok(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Sole",
        "phones": ["+5491155555555"],
        "emails": [],
        "birthday": "",
    })


def test_propose_send_note_explicit_path(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Receta.md"
    note.write_text("# Receta\nharina + agua", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "Receta.md")
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_message"
    assert payload["needs_clarification"] is True
    fields = payload["fields"]
    assert fields["error"] is None
    assert fields["source_path"] == "Receta.md"
    assert "*Receta*" in fields["message_text"]
    assert "harina + agua" in fields["message_text"]
    assert fields["jid"] == "5491155555555@s.whatsapp.net"


def test_propose_send_note_semantic_top1(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Panqueques.md"
    note.write_text("# Panqueques\n- huevo\n- leche", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "_agent_tool_search", lambda q, k=5: json.dumps([
        {"path": "Panqueques.md", "note": "Panqueques", "score": 0.9, "content": "..."},
        {"path": "Otra.md", "note": "Otra", "score": 0.3, "content": "..."},
    ]))

    raw = rag.propose_whatsapp_send_note("Sole", "panqueques receta")
    payload = json.loads(raw)
    assert payload["fields"]["error"] is None
    assert payload["fields"]["source_path"] == "Panqueques.md"
    assert "Panqueques" in payload["fields"]["message_text"]


def test_propose_send_note_semantic_ambiguous(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "_agent_tool_search", lambda q, k=5: json.dumps([
        {"path": "A.md", "note": "A", "score": 0.3, "content": "..."},
        {"path": "B.md", "note": "B", "score": 0.25, "content": "..."},
        {"path": "C.md", "note": "C", "score": 0.2, "content": "..."},
    ]))

    raw = rag.propose_whatsapp_send_note("Sole", "tema vago")
    payload = json.loads(raw)
    assert payload["fields"]["error"] == "ambiguous"
    assert payload["fields"]["source_path"] is None
    assert payload["fields"]["message_text"] == ""
    cands = payload["fields"].get("candidates") or []
    assert len(cands) == 3
    assert {c["path"] for c in cands} == {"A.md", "B.md", "C.md"}


def test_propose_send_note_section_extract(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Receta.md"
    note.write_text(
        "# Receta\n## Ingredientes\nharina\n## Pasos\nmezclar",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "Receta.md", section="Ingredientes")
    payload = json.loads(raw)
    assert payload["fields"]["error"] is None
    msg = payload["fields"]["message_text"]
    assert "harina" in msg
    assert "mezclar" not in msg
    assert payload["fields"]["section"] == "Ingredientes"


def test_propose_send_note_section_not_found_warning(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Receta.md"
    note.write_text("# Receta\ncuerpo", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "Receta.md", section="Nada")
    payload = json.loads(raw)
    fields = payload["fields"]
    assert fields["error"] is None  # contacto OK + nota OK
    assert fields.get("warning") == "section_not_found"
    # Fallback al contenido completo
    assert "cuerpo" in fields["message_text"]


def test_propose_send_note_contact_not_found_still_renders(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    note = tmp_path / "Receta.md"
    note.write_text("body", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Inexistente", "Receta.md")
    payload = json.loads(raw)
    fields = payload["fields"]
    assert fields["error"] == "not_found"
    assert fields["jid"] is None
    # Mensaje renderizado igual — el user puede editar antes de enviar
    assert "body" in fields["message_text"]
    assert payload["needs_clarification"] is True


def test_propose_send_note_blocks_private_folder(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    priv_dir = tmp_path / "99-Private"
    priv_dir.mkdir()
    note = priv_dir / "secrets.md"
    note.write_text("# Secret\napi key xyz", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "99-Private/secrets.md")
    payload = json.loads(raw)
    fields = payload["fields"]
    assert fields["error"] == "note_is_private"
    assert fields["source_path"] is None
    assert fields["message_text"] == ""
    # La api key NO debe filtrarse al payload
    assert "xyz" not in raw


def test_propose_send_note_blocks_private_frontmatter(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Diario.md"
    note.write_text("---\nprivate: true\n---\npassword: hunter2", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "Diario.md")
    payload = json.loads(raw)
    assert payload["fields"]["error"] == "note_is_private"
    assert "hunter2" not in raw


def test_propose_send_note_truncates_oversized(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    note = tmp_path / "Big.md"
    body = ("x" * 100 + "\n") * 200  # ~20k chars
    note.write_text(body, encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "Big.md")
    payload = json.loads(raw)
    fields = payload["fields"]
    assert fields["was_truncated"] is True
    assert len(fields["message_text"]) <= rag.WHATSAPP_NOTE_MAX_CHARS
    assert fields["message_text"].endswith("(truncado)")


def test_propose_send_note_path_traversal_blocked(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "../../../etc/passwd.md")
    payload = json.loads(raw)
    assert payload["fields"]["error"] == "note_not_found"


def test_propose_send_note_empty_query(monkeypatch, tmp_path):
    _stub_contact_ok(monkeypatch)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    raw = rag.propose_whatsapp_send_note("Sole", "")
    payload = json.loads(raw)
    assert payload["fields"]["error"] == "empty_query"


# ── 5. Tool registry invariants ────────────────────────────────────────────


def test_tool_registered_in_chat_tools():
    assert rag.propose_whatsapp_send_note in _tools.CHAT_TOOLS
    assert "propose_whatsapp_send_note" in _tools.TOOL_FNS


def test_tool_in_proposal_tool_names():
    assert "propose_whatsapp_send_note" in _tools.PROPOSAL_TOOL_NAMES


def test_tool_NOT_in_parallel_safe():
    # Lookup osascript de Apple Contacts es side-effectful y queremos
    # un draft por turno — paralelo complica debugging del bridge hang.
    assert "propose_whatsapp_send_note" not in _tools.PARALLEL_SAFE
