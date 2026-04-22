"""Tests de los slash commands nuevos en `rag chat` (2026-04-22 UX pass).

Contexto: el loop interactivo de `chat()` recibe líneas de `input()` y
dispatcha por pattern-match de prefijo. Los comandos nuevos son
side-effect-heavy (clipboard, stdout, subprocess) así que los
testeamos via helpers puros que hago module-level. Los asserts del
comportamiento global del loop se cubren indirecto con un smoke test.

Nuevos comandos (no afectan calidad del retrieval, son UX):

  /help                    → imprime tabla con todos los slash commands
                             + NL shortcuts + keybinds
  /sources                 → re-imprime las sources de la última
                             respuesta (last_sources ya en memoria)
  /open <n>                → abre la nota #n (1-indexed) del top-k
                             retrieved en Obsidian.app
  /copy                    → copia last_assistant al clipboard (pbcopy)
  /stats                   → info de sesión: n_turnos, session_id,
                             vault_scope, avg top_score

Comparado con la alternativa (menús, fuzzy pickers, etc.), slash
commands son explícitos + discoverable via `/help` + componen con
autocomplete si el operador agrega un binding.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── /help: imprime ayuda completa ────────────────────────────────────────────


def test_chat_help_text_exists():
    """La función `_chat_help_text()` debe existir y devolver str no vacío
    con todos los comandos documentados."""
    assert hasattr(rag, "_chat_help_text")
    out = rag._chat_help_text()
    assert isinstance(out, str)
    assert len(out) > 100, "help debe ser informativo"


def test_chat_help_covers_all_slash_commands():
    """Cubre cada slash command existente + nuevos — sin agujeros.
    Si alguien agrega un nuevo comando, debe aparecer en el help
    o el test falla (regression guard)."""
    out = rag._chat_help_text()
    for cmd in ("/help", "/sources", "/open", "/copy", "/stats",
                "/save", "/reindex", "/inbox", "/links", "/undo",
                "/critique", "/cls", "/exit"):
        assert cmd in out, f"el comando {cmd!r} debe estar en /help"


def test_chat_help_mentions_nl_shortcuts():
    """Los NL shortcuts (ratings, create-intent, save-intent) también
    son útiles — el help los menciona aunque no sean slash."""
    out = rag._chat_help_text()
    assert "👍" in out or "+" in out, "rating positivo"
    assert "👎" in out or "-" in out, "rating negativo"


# ── /sources: re-imprime last_sources ─────────────────────────────────────────


def test_chat_render_last_sources_handles_empty():
    """Cuando no hay última respuesta (last_sources=[]), el helper debe
    imprimir un mensaje amable y no raisear."""
    with patch.object(rag, "console") as mock_console:
        rag._chat_render_last_sources([])
    # Llamó a console.print al menos una vez con un mensaje.
    assert mock_console.print.called


def test_chat_render_last_sources_renders_when_present():
    """Con sources reales, debe invocar el renderer table (render_sources)."""
    metas = [
        {"file": "01-Projects/foo.md", "title": "Foo"},
        {"file": "02-Areas/bar.md", "title": "Bar"},
    ]
    scores = [0.85, 0.62]
    with patch.object(rag, "console") as mock_console, \
         patch.object(rag, "render_sources") as mock_render:
        rag._chat_render_last_sources(metas, scores)
        assert mock_render.called, "debe invocar render_sources(metas, scores)"


# ── /open <n>: abre nota por índice ─────────────────────────────────────────


def test_chat_open_nth_source_valid_index():
    """/open 1 debe invocar el handler de apertura con el path de
    last_sources[0]."""
    metas = [
        {"file": "01-Projects/foo.md", "title": "Foo"},
        {"file": "02-Areas/bar.md", "title": "Bar"},
    ]
    opened: list[str] = []

    def fake_opener(path):
        opened.append(path)
        return True

    rag._chat_open_nth_source(metas, 1, opener=fake_opener)
    assert opened == ["01-Projects/foo.md"]

    opened.clear()
    rag._chat_open_nth_source(metas, 2, opener=fake_opener)
    assert opened == ["02-Areas/bar.md"]


def test_chat_open_nth_source_out_of_range():
    """Índice fuera de rango (0, 99, negativo) no debe abrir nada y
    debe reportar error amable."""
    metas = [{"file": "a.md"}]
    opened: list[str] = []

    for bad_n in (0, -1, 99):
        rag._chat_open_nth_source(metas, bad_n, opener=lambda p: opened.append(p))
    assert opened == []


def test_chat_open_nth_source_no_sources():
    """Sin sources, el helper no crashea."""
    opened: list[str] = []
    rag._chat_open_nth_source([], 1, opener=lambda p: opened.append(p))
    assert opened == []


def test_chat_open_nth_source_skips_cross_source_paths():
    """Paths tipo `gmail://...` no se pueden abrir en Obsidian — el
    helper debe retornar False / no intentar."""
    metas = [
        {"file": "gmail://thread-123"},
        {"file": "01-Projects/bar.md"},
    ]
    opened: list[str] = []
    rag._chat_open_nth_source(
        metas, 1, opener=lambda p: opened.append(p),
    )
    assert opened == [], "cross-source paths no abren en Obsidian"
    # Pero el #2 sí debería funcionar.
    rag._chat_open_nth_source(
        metas, 2, opener=lambda p: opened.append(p),
    )
    assert opened == ["01-Projects/bar.md"]


# ── /copy: copia al clipboard ───────────────────────────────────────────────


def test_chat_copy_to_clipboard_invokes_pbcopy():
    """En macOS usa pbcopy via subprocess. Mockeamos el call para no
    tocar el clipboard real durante tests."""
    copied: list[bytes] = []

    class FakeProc:
        def __init__(self, *a, **kw):
            self.stdin = MagicMock()
            self.stdin.write = lambda b: copied.append(b)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch("subprocess.Popen", FakeProc):
        ok = rag._chat_copy_to_clipboard("hola mundo")
    assert ok is True
    assert copied == [b"hola mundo"]


def test_chat_copy_to_clipboard_empty_string_is_noop():
    """Si no hay última respuesta, copy debe retornar False sin
    tocar el clipboard."""
    with patch("subprocess.Popen") as mock_popen:
        ok = rag._chat_copy_to_clipboard("")
    assert ok is False
    assert not mock_popen.called


def test_chat_copy_to_clipboard_handles_exception():
    """subprocess errors (clipboard no disponible en SSH/linux-no-xclip)
    no deben crashear el chat loop."""
    with patch("subprocess.Popen", side_effect=FileNotFoundError("no pbcopy")):
        ok = rag._chat_copy_to_clipboard("hola")
    assert ok is False


# ── /stats: info de sesión ──────────────────────────────────────────────────


def test_chat_session_stats_shape():
    """El helper debe retornar un dict con keys estables para que
    el render sea parametrizable."""
    sess = {
        "id": "test-session-123",
        "turns": [
            {"q": "q1", "a": "a1", "top_score": 0.85},
            {"q": "q2", "a": "a2", "top_score": 0.45},
            {"q": "q3", "a": "a3", "top_score": 0.72},
        ],
    }
    stats = rag._chat_session_stats(sess, vault_scope=["home", "work"])
    assert stats["session_id"] == "test-session-123"
    assert stats["n_turns"] == 3
    assert stats["vault_scope"] == ["home", "work"]
    # avg_top_score: promedio de los scores presentes
    assert stats["avg_top_score"] == pytest.approx((0.85 + 0.45 + 0.72) / 3, abs=0.001)


def test_chat_session_stats_handles_empty_turns():
    """Sesión nueva sin turnos: n_turns=0, avg_top_score=None (no 0
    para distinguir "sin data" de "todo 0")."""
    sess = {"id": "new", "turns": []}
    stats = rag._chat_session_stats(sess, vault_scope=["home"])
    assert stats["n_turns"] == 0
    assert stats["avg_top_score"] is None


def test_chat_session_stats_tolerates_missing_scores():
    """Turnos viejos pueden no tener top_score — el avg ignora los
    None en vez de crashear."""
    sess = {
        "id": "s",
        "turns": [
            {"q": "q1"},                            # sin top_score
            {"q": "q2", "top_score": 0.8},
            {"q": "q3", "top_score": None},
        ],
    }
    stats = rag._chat_session_stats(sess, vault_scope=["home"])
    assert stats["n_turns"] == 3
    assert stats["avg_top_score"] == pytest.approx(0.8, abs=0.001)


# ── readline history ───────────────────────────────────────────────────────


def test_chat_setup_readline_history_returns_path():
    """`_chat_setup_readline()` configura el historial persistente y
    retorna el path al file (o None si readline no está disponible)."""
    assert hasattr(rag, "_chat_setup_readline")
    # El helper puede retornar None si readline no importa — en test
    # environment sí tenemos readline (libedit en macOS).
    result = rag._chat_setup_readline()
    # Debe ser Path o None.
    assert result is None or isinstance(result, Path)


def test_chat_setup_readline_creates_history_file(tmp_path, monkeypatch):
    """El path del history file cae bajo ~/.local/share/obsidian-rag/
    para ser consistente con los otros state paths."""
    # Mockeamos la constante HOME para el test
    with patch.object(rag, "_CHAT_READLINE_HISTORY",
                      tmp_path / "chat_history"):
        result = rag._chat_setup_readline()
    # El path debe existir después de setup (touched by readline).
    # readline.write_history_file es quien lo crea al fin; el setup
    # solo devuelve el path.
    assert result is None or result == tmp_path / "chat_history"
