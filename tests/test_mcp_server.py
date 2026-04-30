"""Tests adicionales para `mcp_server.py` — wrapper MCP-over-stdio que
expone obsidian-rag a Claude Code / Devin / Cursor.

Este archivo es **complementario** a `test_mcp_tools.py` (que ya cubre
muchos paths con un fake_rag fixture). Nos enfocamos en:

- Cobertura por-tool del happy path + 1 error case (cumpliendo el
  brief "≥12 casos, cada tool con happy + error").
- Lazy-load del módulo `rag` (mockeado entirely vía monkeypatch).
- Validación del regex de `session_id` declarado en docstring de
  `rag_query` (audit 2026-04-25 R2-2 #4).
- Path-traversal en `rag_read_note`.
- Cumple los entregables del audit en español rioplatense.

Nota: NO toca el filesystem ni la red — el módulo `rag` se mockea
completo via `monkeypatch.setattr(mcp_server, "_load_rag", lambda: ...)`.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import mcp_server


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_lazy_state(monkeypatch):
    """Cada test arranca con `_rag = None` y un lock fresco — así no
    arrastramos estado entre tests (importante para los tests del
    lazy-load + thread-safety)."""
    monkeypatch.setattr(mcp_server, "_rag", None)
    monkeypatch.setattr(mcp_server, "_rag_lock", threading.Lock())


@pytest.fixture
def mock_rag(monkeypatch, tmp_path):
    """Reemplaza `_load_rag` con una función que devuelve un MagicMock
    con la API de `rag` que usa `mcp_server`.

    El MagicMock tiene VAULT_PATH, COLLECTION_NAME, EMBED_MODEL,
    RERANKER_MODEL, get_db, _load_corpus, retrieve, find_urls, etc.
    Tests específicos overridean return values cuando hace falta.
    """
    vault = tmp_path / "vault"
    (vault / "01-Notes").mkdir(parents=True)
    (vault / "01-Notes" / "ejemplo.md").write_text(
        "# Ejemplo\n\nNota de ejemplo para tests.\n",
        encoding="utf-8",
    )

    rag_mock = MagicMock()
    rag_mock.VAULT_PATH = vault
    rag_mock.COLLECTION_NAME = "test_collection_v1"
    rag_mock.EMBED_MODEL = "bge-m3"
    rag_mock.RERANKER_MODEL = "bge-reranker-v2-m3"

    col = MagicMock()
    col.count.return_value = 100
    rag_mock.get_db.return_value = col

    rag_mock._load_corpus.return_value = {
        "metas": [
            {"file": "01-Notes/ejemplo.md", "note": "Ejemplo",
             "folder": "01-Notes", "tags": "ejemplo,test"},
        ],
    }

    monkeypatch.setattr(mcp_server, "_load_rag", lambda: rag_mock)
    return rag_mock


# ── rag_query: happy + error ─────────────────────────────────────────────────


def test_rag_query_happy_path_shape_and_session_persistence(mock_rag):
    """Happy path: la query devuelve la lista de chunks con shape
    estable, y cuando hay session_id se persiste el turn."""
    mock_rag.retrieve.return_value = {
        "docs": ["contenido relevante"],
        "metas": [{
            "file": "01-Notes/ejemplo.md", "note": "Ejemplo",
            "folder": "01-Notes", "tags": "ejemplo",
        }],
        "scores": [0.92],
        "confidence": 0.85,
    }
    mock_rag.ensure_session.return_value = object()
    mock_rag.session_history.return_value = None  # primer turn

    out = mock_rag, mcp_server.rag_query(
        "qué es ejemplo", k=3, session_id="mcp-test-001",
    )
    rag, result = out
    assert len(result) == 1
    assert result[0] == {
        "note": "Ejemplo",
        "path": "01-Notes/ejemplo.md",
        "folder": "01-Notes",
        "tags": "ejemplo",
        "score": 0.92,
        "content": "contenido relevante",
    }
    # Session: ensure_session llamada + append_turn + save_session.
    rag.ensure_session.assert_called_once_with("mcp-test-001", mode="mcp")
    rag.append_turn.assert_called_once()
    rag.save_session.assert_called_once()


def test_rag_query_error_invalid_session_id_returns_empty(mock_rag):
    """Audit R2-2 #4: session_ids con caracteres prohibidos rebotan con
    `[]` ANTES de tocar filesystem o invocar el módulo rag."""
    out = mcp_server.rag_query("hola", session_id="../../etc/passwd")
    assert out == []
    # ensure_session NO debe haberse llamado — la guard cortó antes.
    mock_rag.ensure_session.assert_not_called()


# ── rag_read_note: happy + error ─────────────────────────────────────────────


def test_rag_read_note_happy_path_returns_content(mock_rag):
    out = mcp_server.rag_read_note("01-Notes/ejemplo.md")
    assert "Nota de ejemplo para tests" in out
    assert not out.startswith("Error:")


def test_rag_read_note_error_path_traversal_rejected(mock_rag):
    """Path traversal clásico: `../../etc/passwd.md` tiene .md pero
    resuelve fuera del vault. Debe devolver Error: con razón legible."""
    # Probamos ambos: un payload con .md y uno sin.
    out_md = mcp_server.rag_read_note("../../../etc/shadow.md")
    assert out_md.startswith("Error:")
    assert "escapes" in out_md.lower() or "not found" in out_md.lower()

    out_no_md = mcp_server.rag_read_note("../../../etc/shadow")
    assert out_no_md.startswith("Error: path must end in .md")


# ── rag_list_notes: happy + error ────────────────────────────────────────────


def test_rag_list_notes_happy_path_returns_dedup_shape(mock_rag):
    """El corpus puede tener varias chunks por nota — list_notes dedup-ea
    por path."""
    mock_rag._load_corpus.return_value = {
        "metas": [
            {"file": "01-Notes/a.md", "note": "A", "folder": "01-Notes",
             "tags": "x,y"},
            # duplicado del mismo file (chunk 2 de la misma nota) — debe
            # dedup-ear.
            {"file": "01-Notes/a.md", "note": "A", "folder": "01-Notes",
             "tags": "x,y"},
            {"file": "01-Notes/b.md", "note": "B", "folder": "01-Notes",
             "tags": "z"},
        ],
    }
    out = mcp_server.rag_list_notes(folder="01-Notes", limit=100)
    paths = [n["path"] for n in out]
    assert paths == ["01-Notes/a.md", "01-Notes/b.md"]


def test_rag_list_notes_error_filter_excludes_all(mock_rag):
    """Filtrando por una folder inexistente → lista vacía (no raisea)."""
    out = mcp_server.rag_list_notes(folder="99-NoExiste", limit=100)
    assert out == []


# ── rag_links: happy + error ─────────────────────────────────────────────────


def test_rag_links_happy_path_returns_normalised_shape(mock_rag):
    mock_rag.find_urls.return_value = [
        {
            "url": "https://example.com/x",
            "anchor": "ver acá",
            "path": "03-Resources/Bookmarks.md",
            "note": "Bookmarks",
            "line": 42,
            "context": "ver acá los detalles del setup",
            "score": 0.81,
        },
    ]
    out = mcp_server.rag_links("setup detalles", k=5)
    assert len(out) == 1
    assert out[0]["url"] == "https://example.com/x"
    assert out[0]["anchor"] == "ver acá"
    assert out[0]["score"] == 0.81


def test_rag_links_error_clamps_k_to_max_30(mock_rag):
    """k > 30 se clampea a 30 (audit: protege contra DoS de queries
    enormes)."""
    mock_rag.find_urls.return_value = []
    mcp_server.rag_links("algo", k=10_000)
    # El kwarg k que llega a find_urls debe ser 30, no 10_000.
    assert mock_rag.find_urls.call_args.kwargs["k"] == 30


# ── rag_stats: happy + error ─────────────────────────────────────────────────


def test_rag_stats_happy_path_returns_metadata(mock_rag):
    out = mcp_server.rag_stats()
    assert out["chunks"] == 100
    assert out["collection"] == "test_collection_v1"
    assert out["embed_model"] == "bge-m3"
    assert out["reranker"] == "bge-reranker-v2-m3"
    assert "vault" in out["vault_path"] or out["vault_path"].endswith("vault")


def test_rag_stats_error_zero_chunks_still_returns_dict(mock_rag):
    """Vault vacío (col.count() == 0) → stats igualmente devuelven la
    metadata, solo chunks=0. NO debe raisear."""
    mock_rag.get_db.return_value.count.return_value = 0
    out = mcp_server.rag_stats()
    assert out["chunks"] == 0
    assert out["collection"] == "test_collection_v1"


# ── rag_capture: happy + error ───────────────────────────────────────────────


def test_rag_capture_happy_path_writes_and_indexes(mock_rag):
    """capture_note recibe el texto + tags + source; el path retornado
    es vault-relative; auto-index dispara después."""
    written = mock_rag.VAULT_PATH / "00-Inbox" / "2026-04-29-1200-test.md"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text("dummy", encoding="utf-8")
    mock_rag.capture_note.return_value = written

    out = mcp_server.rag_capture(
        "una idea importante",
        tags=["urgent", "mcp-server"],
        source="claude-code",
    )
    assert out["created"] is True
    assert out["path"] == "00-Inbox/2026-04-29-1200-test.md"
    mock_rag.capture_note.assert_called_once()
    mock_rag._index_single_file.assert_called_once()


def test_rag_capture_error_empty_text_does_not_call_capture_note(mock_rag):
    out = mcp_server.rag_capture("")
    assert out["created"] is False
    assert "empty" in out["error"].lower()
    mock_rag.capture_note.assert_not_called()


# ── rag_save_note: happy + error ─────────────────────────────────────────────


def test_rag_save_note_happy_path_writes_to_explicit_folder(mock_rag):
    """Happy path: usa el _slug real para escribir un archivo en
    02-Areas/Test/<slug>.md con frontmatter."""
    import rag as real_rag
    mock_rag._slug = real_rag._slug

    out = mcp_server.rag_save_note(
        "Cuerpo del contenido", "Mi Nota Test",
        folder="02-Areas/Test", tags=["t1", "t2"],
    )
    assert out["created"] is True
    assert out["path"].startswith("02-Areas/Test/")
    assert out["path"].endswith(".md")
    written = mock_rag.VAULT_PATH / out["path"]
    assert written.is_file()
    body = written.read_text(encoding="utf-8")
    assert "# Mi Nota Test" in body
    assert "Cuerpo del contenido" in body
    assert "  - t1" in body


def test_rag_save_note_error_folder_traversal_rejected(mock_rag):
    """Folder con `..` rebota con error explícito antes de tocar el FS."""
    out = mcp_server.rag_save_note(
        "body", "title", folder="../../escape",
    )
    assert out["created"] is False
    assert "invalid folder" in out["error"] or "escapes" in out["error"]


# ── rag_create_reminder: happy + error ───────────────────────────────────────


def test_rag_create_reminder_happy_path_forwards_args_and_parses_json(mock_rag):
    mock_rag.propose_reminder.return_value = (
        '{"kind":"reminder","created":true,"reminder_id":"R-001",'
        '"fields":{"title":"comprar pan","due_iso":"2026-04-30T08:00:00"}}'
    )
    out = mcp_server.rag_create_reminder(
        title="comprar pan", when="mañana 8am",
        reminder_list="Casa", priority=5, notes="integral",
        recurrence="todos los lunes",
    )
    assert out["created"] is True
    assert out["reminder_id"] == "R-001"

    kwargs = mock_rag.propose_reminder.call_args.kwargs
    assert kwargs["title"] == "comprar pan"
    assert kwargs["when"] == "mañana 8am"
    assert kwargs["list"] == "Casa"  # MCP `reminder_list` → propose `list`
    assert kwargs["priority"] == 5
    assert kwargs["notes"] == "integral"
    assert kwargs["recurrence_text"] == "todos los lunes"


def test_rag_create_reminder_error_malformed_json_returns_dict_with_raw(mock_rag):
    mock_rag.propose_reminder.return_value = "not json {{"
    out = mcp_server.rag_create_reminder(title="X", when="hoy")
    assert out["created"] is False
    assert "json" in out["error"].lower()
    assert out["raw"] == "not json {{"
    assert out["kind"] == "reminder"


# ── rag_create_event: happy + error ──────────────────────────────────────────


def test_rag_create_event_happy_path_forwards_args(mock_rag):
    mock_rag.propose_calendar_event.return_value = (
        '{"kind":"event","created":true,"event_uid":"E-42",'
        '"fields":{"title":"reunión","start_iso":"2026-05-01T10:00:00"}}'
    )
    out = mcp_server.rag_create_event(
        title="reunión", start="jueves 10am",
        end="jueves 11am", calendar="Trabajo",
        location="Zoom", notes="prep agenda",
        all_day=False, recurrence=None,
    )
    assert out["created"] is True
    assert out["event_uid"] == "E-42"
    kwargs = mock_rag.propose_calendar_event.call_args.kwargs
    assert kwargs["title"] == "reunión"
    assert kwargs["start"] == "jueves 10am"
    assert kwargs["end"] == "jueves 11am"
    assert kwargs["calendar"] == "Trabajo"
    assert kwargs["all_day"] is False


def test_rag_create_event_error_needs_clarification_passthrough(mock_rag):
    """Cuando la fecha es ambigua, propose_calendar_event devuelve
    `needs_clarification:true` — el wrapper pasa el dict tal cual."""
    mock_rag.propose_calendar_event.return_value = (
        '{"kind":"event","needs_clarification":true,'
        '"proposal_id":"prop-999","fields":{"title":"X","start_iso":null}}'
    )
    out = mcp_server.rag_create_event(title="X", start="algún día de la semana")
    assert out.get("needs_clarification") is True
    assert out["proposal_id"] == "prop-999"
    assert "created" not in out or out["created"] is False


# ── rag_followup: happy + error ──────────────────────────────────────────────


def test_rag_followup_happy_path_with_status_filter(mock_rag):
    mock_rag.find_followup_loops.return_value = [
        {"status": "stale", "age_days": 60, "kind": "todo",
         "source_note": "A.md", "loop_text": "llamar a X"},
        {"status": "activo", "age_days": 5, "kind": "checkbox",
         "source_note": "B.md", "loop_text": "[ ] revisar Y"},
    ]
    activos = mcp_server.rag_followup(days=30, status="activo")
    assert len(activos) == 1
    assert activos[0]["status"] == "activo"


def test_rag_followup_error_negative_limit_clamped_to_one(mock_rag):
    """`max(1, int(limit))` clampea limits ≤ 0 a 1 (defensa contra
    inputs raros del LLM)."""
    mock_rag.find_followup_loops.return_value = [
        {"status": "stale", "age_days": 1, "kind": "x",
         "source_note": "a.md", "loop_text": "..."},
        {"status": "activo", "age_days": 2, "kind": "x",
         "source_note": "b.md", "loop_text": "..."},
    ]
    out = mcp_server.rag_followup(days=30, limit=0)
    # Como limit pasa por max(1, ...), el resultado es 1 elemento.
    assert len(out) == 1


# ── Validación session_id (audit 2026-04-25 R2-2 #4) ─────────────────────────


def test_session_id_regex_accepts_format_declared_in_docstring(mock_rag):
    """El docstring de rag_query declara `[A-Za-z0-9_.:-]{1,64}`. La
    regex compilada (`_SESSION_ID_RE`) tiene que matchear EXACTAMENTE
    eso — auditá que ningún cambio futuro la afloje."""
    valid = [
        "tg:123",                  # telegram bot prefix
        "mcp-claude",              # MCP wrapper
        "web:user_42",             # web UI
        "session.with.dots",       # dots permitidos
        "X" * 64,                  # límite superior
        "a",                       # límite inferior (1 char)
    ]
    for s in valid:
        assert mcp_server._SESSION_ID_RE.match(s), f"válido pero falló: {s!r}"

    invalid = [
        "",                        # vacío (mín = 1)
        "X" * 65,                  # excede máx
        "../../etc",               # path traversal
        "session/with/slash",      # slashes
        "a b",                     # espacios
        "id\nwith\nnewlines",      # control chars
        "<script>alert(1)</script>",  # XSS
        "drop;table",              # punctuation no permitida (;)
    ]
    for s in invalid:
        assert not mcp_server._SESSION_ID_RE.match(s), f"inválido pero pasó: {s!r}"


# ── Lazy-load del módulo `rag` ───────────────────────────────────────────────


def test_load_rag_caches_module_after_first_call():
    """`_load_rag` carga `rag` una sola vez — la 2da llamada hits cache."""
    import sys
    fake = MagicMock()
    fake.__name__ = "rag"
    original = sys.modules.get("rag")
    sys.modules["rag"] = fake
    try:
        # Reset estado.
        mcp_server._rag = None
        r1 = mcp_server._load_rag()
        r2 = mcp_server._load_rag()
        assert r1 is r2 is fake
    finally:
        if original is not None:
            sys.modules["rag"] = original
        mcp_server._rag = None


# ── _touch + idle thresholds ─────────────────────────────────────────────────


def test_touch_advances_last_call_timestamp(monkeypatch):
    """`_touch` actualiza `_last_call` — el idle-killer lo lee para
    decidir si seppukar el proceso."""
    monkeypatch.setattr(mcp_server, "_last_call", 0.0)
    mcp_server._touch()
    assert mcp_server._last_call > 0


def test_idle_constants_within_sane_bounds():
    """Guard contra dropouts accidentales: si alguien baja
    `_IDLE_HOT_SECONDS` < 15min se podría churnear respawns. Si sube
    `_IDLE_COLD_SECONDS` < 30min, hot-evict pierde sentido."""
    assert mcp_server._IDLE_HOT_SECONDS >= 15 * 60
    assert mcp_server._IDLE_COLD_SECONDS > mcp_server._IDLE_HOT_SECONDS
    # Sanity: < 24h (más de eso es inútil — el daemon estará idle un día).
    assert mcp_server._IDLE_COLD_SECONDS < 24 * 3600
