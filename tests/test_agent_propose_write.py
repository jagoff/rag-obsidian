"""Tests para el fix del bug de `_agent_tool_propose_write` (Fase A.1 del
scribe agent, 2026-04-23).

Historia: pre-fix, el apply loop en `rag do` hacía:

    if full.exists():
        full = full.with_name(f"{full.stem} ({datetime.now():%H%M%S}).md")

→ sea lo que sea que el LLM proponga, si el path ya existe se creaba un
archivo nuevo con sufijo silencioso `(HHMMSS).md`. El usuario pedía
"agregá X a mi nota Y" y terminaba con una nota-duplicada `Y (183022).md`
sin entender por qué.

Post-fix:
1. `_agent_tool_propose_write` acepta un kwarg nuevo `overwrite: bool = False`.
2. Si el path ya existe + overwrite=False → devuelve error explícito
   sugiriendo `append_to_note` o `overwrite=True`.
3. Si path existe + overwrite=True → registra entry con `kind="overwrite"`.
4. Si path no existe → registra entry con `kind="create"` (param overwrite
   ignorado — no hay nada que sobrescribir).
5. El apply loop hace backup `<path>.bak.<unix_ts>` antes de overwrite.

Estos 8 casos cubren: happy path create / existing + overwrite=False error
/ existing + overwrite=True ok / dotfolder rechazado / non-md rechazado /
path fuera del vault rechazado / kind registrado correctamente / backup
on overwrite apply.
"""
from __future__ import annotations

import time

import rag


# ── Helper: setup VAULT_PATH via conftest autouse + clear pending ──────


def _reset_pending():
    rag._AGENT_PENDING_WRITES.clear()


# ── Caso 1: path nuevo → registra kind="create" ────────────────────────


def test_propose_write_new_path_registers_create(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_propose_write(
        path="00-Inbox/nueva.md",
        content="# Nueva nota\n\nHola.",
        rationale="test",
    )

    assert "error" not in result.lower()
    assert len(rag._AGENT_PENDING_WRITES) == 1
    entry = rag._AGENT_PENDING_WRITES[0]
    assert entry["kind"] == "create"
    assert entry["path"] == "00-Inbox/nueva.md"
    assert "Nueva nota" in entry["content"]


# ── Caso 2: path existente + overwrite=False (default) → error ─────────


def test_propose_write_existing_path_default_errors(tmp_path, monkeypatch):
    """Este ES el bug principal. Pre-fix, el apply loop creaba
    `nota (HHMMSS).md` silenciosamente. Post-fix, la tool devuelve error
    explícito ANTES del apply loop, así el LLM sabe que tiene que elegir
    entre overwrite=True o append_to_note."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    existing = tmp_path / "00-Inbox" / "existe.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# Existe\n\nContenido original.\n", encoding="utf-8")

    result = rag._agent_tool_propose_write(
        path="00-Inbox/existe.md",
        content="# Nuevo body\n\nAlgo distinto.",
        rationale="test",
    )

    assert result.lower().startswith("error")
    # Debe sugerir las dos alternativas para que el LLM sepa qué hacer
    assert "overwrite" in result.lower()
    assert "append_to_note" in result.lower()
    # Nada debe haberse registrado en la cola
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 3: path existente + overwrite=True → registra kind="overwrite" ─


def test_propose_write_existing_with_overwrite_registers(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    existing = tmp_path / "02-Areas" / "proyecto.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# Viejo\n", encoding="utf-8")

    result = rag._agent_tool_propose_write(
        path="02-Areas/proyecto.md",
        content="# Nuevo reemplazo\n",
        rationale="limpieza",
        overwrite=True,
    )

    assert "error" not in result.lower()
    assert len(rag._AGENT_PENDING_WRITES) == 1
    entry = rag._AGENT_PENDING_WRITES[0]
    assert entry["kind"] == "overwrite"
    assert entry["path"] == "02-Areas/proyecto.md"


# ── Caso 4: dotfolder rechazado (preserva invariante pre-fix) ──────────


def test_propose_write_dotfolder_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_propose_write(
        path=".obsidian/config.md",
        content="x",
    )

    assert result.lower().startswith("error")
    assert "dotfolder" in result.lower()
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 5: path sin .md rechazado ────────────────────────────────────


def test_propose_write_non_md_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    result = rag._agent_tool_propose_write(
        path="00-Inbox/foo.txt",
        content="x",
    )

    assert result.lower().startswith("error")
    assert ".md" in result
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 6: path fuera del vault rechazado (symlink-escape / ..) ───────


def test_propose_write_path_outside_vault_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    # Intento clásico de escape con `..` — tiene que validar contra
    # resolved path dentro del vault.
    result = rag._agent_tool_propose_write(
        path="../escape.md",
        content="x",
    )

    assert result.lower().startswith("error")
    assert rag._AGENT_PENDING_WRITES == []


# ── Caso 7: apply de kind="overwrite" → crea backup .bak.<ts> ──────────


def test_propose_write_apply_overwrite_creates_backup(tmp_path, monkeypatch):
    """Integration: cuando el user confirma un overwrite, el apply
    helper debe dejar un backup .bak.<unix_ts> del archivo original
    antes de escribir el content nuevo. Esto garantiza rollback manual
    con `cp`."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    target = tmp_path / "02-Areas" / "nota.md"
    target.parent.mkdir(parents=True)
    original_body = "# Original\n\nLine 1\nLine 2\n"
    target.write_text(original_body, encoding="utf-8")

    # Propose overwrite
    rag._agent_tool_propose_write(
        path="02-Areas/nota.md",
        content="# Nuevo body\n\nSolo una line.\n",
        overwrite=True,
    )
    assert len(rag._AGENT_PENDING_WRITES) == 1

    # Apply directamente via helper (bypasseamos el loop del CLI para
    # testear la lógica de apply de forma aislada). `_scribe_apply_entry`
    # es el helper que vamos a extraer del loop.
    #
    # Mockeamos _index_single_file para no tocar la DB real.
    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    # El content nuevo está escrito
    assert target.read_text(encoding="utf-8") == "# Nuevo body\n\nSolo una line.\n"

    # Debe existir un backup con prefijo .bak. y el contenido original
    baks = list(target.parent.glob("nota.md.bak.*"))
    assert len(baks) == 1, f"esperaba 1 backup, vi: {baks}"
    bak = baks[0]
    assert bak.read_text(encoding="utf-8") == original_body

    # El sufijo del backup debe ser un unix timestamp razonable (no HHMMSS)
    suffix = bak.name.rsplit(".bak.", 1)[1]
    assert suffix.isdigit()
    # Razonable: dentro de ~5s del presente
    assert abs(int(suffix) - int(time.time())) < 5


# ── Caso 8: apply de kind="create" NO crea backup (no hay qué respaldar) ─


def test_propose_write_apply_create_no_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _reset_pending()

    rag._agent_tool_propose_write(
        path="00-Inbox/fresh.md",
        content="# Fresh\n",
    )
    assert len(rag._AGENT_PENDING_WRITES) == 1
    assert rag._AGENT_PENDING_WRITES[0]["kind"] == "create"

    monkeypatch.setattr(rag, "_index_single_file", lambda col, path: None)
    rag._scribe_apply_entry(rag._AGENT_PENDING_WRITES[0], col=None)

    target = tmp_path / "00-Inbox" / "fresh.md"
    assert target.read_text(encoding="utf-8") == "# Fresh\n"
    # No debe haber .bak.* en el directorio
    assert list(target.parent.glob("*.bak.*")) == []
