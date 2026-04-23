"""Tests para la nueva tool `_agent_tool_append_to_note` (Fase A.2 del
scribe agent, 2026-04-23).

Caso de uso que resuelve: "agregá a mi nota de Astor que hoy caminó en el
jardín". Pre-fix, la única opción del agente era `propose_write` (crear
entera) — o agregaba un archivo duplicado con sufijo HHMMSS, o
sobrescribía toda la nota perdiendo el contexto previo. Esta tool permite:

- section=None: append al final del body (después del último contenido).
- section="## Heading exacto": insertar justo después de esa línea.

Semantics:
- Path debe existir. Si no, error (no crea implícitamente — para crear,
  el agente usa `propose_write`).
- Si el heading no existe (section pasado pero no matcheó), error
  explícito (el agente puede crear el heading con propose_write overwrite
  si quiere).
- Frontmatter YAML intacto (el append no toca ni el frontmatter ni el
  último `---`).
- Wikilinks del body originales no se tocan (el apply garantiza esto
  porque solo concatena/inserta sobre el body crudo).

Casos (11):
1. path no existe → error
2. section=None → append al final del body
3. section="## Pendientes" → insert después del heading
4. heading no existe → error
5. dotfolder rechazado
6. preserves frontmatter
7. path fuera del vault → error
8. registers entry con kind="append" + section guardado
9. non-md rechazado
10. multiple headings: matchea el primero exacto, no substrings
11. apply crea backup .bak.<ts>
"""
from __future__ import annotations

import time

import rag


def _reset_pending():
    rag._AGENT_PENDING_WRITES.clear()


# ── Caso 1: path no existe → error ─────────────────────────────────────


def test_append_path_must_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_append_to_note(
        path="00-Inbox/no-existe.md",
        content="algo",
    )

    assert result.lower().startswith("error")
    assert "no encontrada" in result.lower() or "no existe" in result.lower()
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 2: section=None → registra append al final ────────────────────


def test_append_end_of_file_registers(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "02-Areas" / "astor.md"
    nota.parent.mkdir(parents=True)
    nota.write_text("# Astor\n\nPrimera línea.\n", encoding="utf-8")

    result = rag._agent_tool_append_to_note(
        path="02-Areas/astor.md",
        content="Hoy caminó en el jardín.",
        rationale="evento del día",
    )

    assert "error" not in result.lower()
    assert len(rag._AGENT_PENDING_WRITES) == 1
    entry = rag._AGENT_PENDING_WRITES[0]
    assert entry["kind"] == "append"
    assert entry["section"] is None
    assert "caminó" in entry["content"]


# ── Caso 3: section="## Pendientes" → registra con section guardado ────


def test_append_under_heading_registers(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "01-Projects" / "rag.md"
    nota.parent.mkdir(parents=True)
    nota.write_text(
        "# Proyecto RAG\n\n## Pendientes\n\n- Item uno\n\n## Notas\n\nBlah.\n",
        encoding="utf-8",
    )

    result = rag._agent_tool_append_to_note(
        path="01-Projects/rag.md",
        content="- Item dos",
        section="## Pendientes",
    )

    assert "error" not in result.lower()
    assert len(rag._AGENT_PENDING_WRITES) == 1
    entry = rag._AGENT_PENDING_WRITES[0]
    assert entry["kind"] == "append"
    assert entry["section"] == "## Pendientes"


# ── Caso 4: heading no existe → error ──────────────────────────────────


def test_append_heading_not_found_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "02-Areas" / "proy.md"
    nota.parent.mkdir(parents=True)
    nota.write_text("# Proy\n\n## Solo este heading\n\nHola.\n", encoding="utf-8")

    result = rag._agent_tool_append_to_note(
        path="02-Areas/proy.md",
        content="algo",
        section="## No existe este heading",
    )

    assert result.lower().startswith("error")
    assert "heading" in result.lower() or "no existe" in result.lower()
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 5: dotfolder rechazado ────────────────────────────────────────


def test_append_dotfolder_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_append_to_note(
        path=".obsidian/plugins.md",
        content="x",
    )

    assert result.lower().startswith("error")
    assert "dotfolder" in result.lower()


# ── Caso 6: preserva frontmatter al append ────────────────────────────


def test_append_preserves_frontmatter_on_apply(tmp_path, monkeypatch):
    """Integration: aplicamos un append sobre una nota con frontmatter.
    El frontmatter debe estar intacto post-write."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "02-Areas" / "con-fm.md"
    nota.parent.mkdir(parents=True)
    original = (
        "---\n"
        "tags: [coaching, astor]\n"
        "created: 2026-04-20\n"
        "---\n"
        "\n"
        "# Astor\n"
        "\n"
        "Crónica de sus días.\n"
    )
    nota.write_text(original, encoding="utf-8")

    rag._agent_tool_append_to_note(
        path="02-Areas/con-fm.md",
        content="Hoy jugó con Lumi.",
    )
    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    result = nota.read_text(encoding="utf-8")
    # Frontmatter intacto al inicio
    assert result.startswith("---\ntags: [coaching, astor]\ncreated: 2026-04-20\n---\n")
    # El body original + el append
    assert "Crónica de sus días." in result
    assert "Hoy jugó con Lumi." in result


# ── Caso 7: path fuera del vault → error ──────────────────────────────


def test_append_path_outside_vault_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_append_to_note(
        path="../escape.md",
        content="x",
    )

    assert result.lower().startswith("error")


# ── Caso 8: entry shape ───────────────────────────────────────────────


def test_append_entry_shape(tmp_path, monkeypatch):
    """El entry en _AGENT_PENDING_WRITES tiene los 5 campos que el
    apply loop espera: kind, path, content, section, rationale."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "a.md"
    nota.write_text("x\n", encoding="utf-8")

    rag._agent_tool_append_to_note(
        path="a.md",
        content="y",
        section=None,
        rationale="test shape",
    )

    assert len(rag._AGENT_PENDING_WRITES) == 1
    entry = rag._AGENT_PENDING_WRITES[0]
    assert set(entry.keys()) >= {"kind", "path", "content", "section", "rationale"}
    assert entry["kind"] == "append"
    assert entry["rationale"] == "test shape"


# ── Caso 9: non-md rechazado ──────────────────────────────────────────


def test_append_non_md_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_append_to_note(
        path="00-Inbox/foo.txt",
        content="x",
    )

    assert result.lower().startswith("error")
    assert ".md" in result


# ── Caso 10: múltiples headings — matchea exacto, no substring ────────


def test_append_under_heading_exact_match(tmp_path, monkeypatch):
    """Si hay `## Notas` y `## Notas personales`, section="## Notas"
    debe matchear SOLO el primero, no el segundo. Lo garantizamos con
    regex de línea completa ($)."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "m.md"
    nota.write_text(
        "# T\n\n## Notas\n\nContent A.\n\n## Notas personales\n\nContent B.\n",
        encoding="utf-8",
    )

    rag._agent_tool_append_to_note(
        path="m.md",
        content="NUEVO",
        section="## Notas",
    )
    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    body = nota.read_text(encoding="utf-8")
    # "NUEVO" debe aparecer entre "## Notas" y "Content A", NO después
    # de "Content A" ni después de "## Notas personales".
    idx_heading = body.index("## Notas\n")
    idx_content_a = body.index("Content A.")
    idx_nuevo = body.index("NUEVO")
    idx_notas_pers = body.index("## Notas personales")
    assert idx_heading < idx_nuevo < idx_content_a < idx_notas_pers


# ── Caso 11: apply crea backup .bak.<unix_ts> ─────────────────────────


def test_append_apply_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "n.md"
    original = "# N\n\nLínea.\n"
    nota.write_text(original, encoding="utf-8")

    rag._agent_tool_append_to_note(path="n.md", content="Extra.")
    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    baks = list(nota.parent.glob("n.md.bak.*"))
    assert len(baks) == 1
    assert baks[0].read_text(encoding="utf-8") == original
    # El sufijo es unix timestamp
    suffix = baks[0].name.rsplit(".bak.", 1)[1]
    assert suffix.isdigit()
    assert abs(int(suffix) - int(time.time())) < 5


# ── Caso 12: append real produce el body esperado (end-of-file) ───────


def test_append_end_of_file_produces_expected_body(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    nota = tmp_path / "log.md"
    nota.write_text("# Log\n\nViejo\n", encoding="utf-8")

    rag._agent_tool_append_to_note(path="log.md", content="Nuevo")
    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    result = nota.read_text(encoding="utf-8")
    # El append garantiza: blank line separadora + content + \n final
    assert "Viejo" in result
    assert "Nuevo" in result
    assert result.endswith("Nuevo\n")
    # Viejo viene antes que Nuevo
    assert result.index("Viejo") < result.index("Nuevo")
