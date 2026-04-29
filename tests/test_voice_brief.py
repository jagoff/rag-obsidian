"""Tests para `rag.voice_brief` — Anticipatory Phase 2.C voice brief.

Coverage:
  1. ``synthesize_brief_audio`` con texto simple → genera archivo en
     el path esperado (``<output_dir>/<YYYY-MM-DD>-morning.ogg``).
  2. Idempotencia: dos llamadas con mismo texto → mismo path, no
     regenera (verificá mtime).
  3. Trim a 4000 chars si el texto es más largo.
  4. CLI ``rag voice-brief generate --date 2026-04-29 --text "test"``
     corre y genera output esperado.
  5. CLI smoke ``rag morning --voice --dry-run`` no rompe (mock TTS).
  6. Fallback: si ``say`` no está disponible y no hay TTS alternativo,
     ``synthesize_brief_audio`` devuelve ``None`` silent.
  7. ``strip_markdown_for_speech`` sanea wikilinks/code/footers/headings
     correctamente para que ``say`` no lea el markup.
  8. ``send_audio_to_whatsapp`` POST con ``media_path`` y silent-fail.
  9. ``cleanup_old_voice_briefs`` borra archivos > TTL.
 10. ``_brief_push_to_whatsapp`` con ``audio_path`` válido + send-ok →
     audio sale primero, texto incluye marker, footer intacto al final.

Estos tests NO ejecutan ``say`` ni ``ffmpeg`` reales; mockean el
subprocess para correr en CI sin macOS.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

import rag
from rag import voice_brief


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def vb_dir(tmp_path, monkeypatch):
    """Directorio aislado para voice briefs en cada test."""
    d = tmp_path / "voice_briefs"
    monkeypatch.setenv("RAG_VOICE_BRIEF_DIR", str(d))
    return d


@pytest.fixture
def fake_say(monkeypatch):
    """Mockea ``shutil.which("say")`` + el ``subprocess.run`` que llama say.

    Devuelve un dict ``{"ran": [...]}`` para inspeccionar las llamadas.
    """
    state = {"ran": []}

    def fake_which(cmd):
        if cmd in ("say", "ffmpeg"):
            return f"/usr/bin/{cmd}"
        return None

    def fake_run(cmd, *args, **kwargs):
        state["ran"].append(cmd)
        # Crear archivo dummy en el out path para simular éxito.
        if cmd[0] == "say":
            # cmd = ["say", "-v", voice, "--file-format=AIFF", "-o", out, text]
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"AIFF\x00\x00fakedata" * 100)
        elif cmd[0] == "ffmpeg":
            # cmd = ["ffmpeg", "-y", "-i", aiff, ..., out]
            Path(cmd[-1]).write_bytes(b"OggS\x00\x02fakeoggdata" * 50)
        return mock.Mock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(voice_brief.shutil, "which", fake_which)
    monkeypatch.setattr(voice_brief.subprocess, "run", fake_run)
    return state


@pytest.fixture
def no_say(monkeypatch):
    """No hay ``say`` ni ``ffmpeg`` en el PATH."""
    monkeypatch.setattr(voice_brief.shutil, "which", lambda cmd: None)


# ── 1. Generación básica ────────────────────────────────────────────────────

def test_synthesize_brief_audio_basic(vb_dir, fake_say):
    """Texto simple → archivo en el path esperado, contenido no vacío."""
    text = "Hola, este es un brief simple."
    path = voice_brief.synthesize_brief_audio(
        text, kind="morning", date_str="2026-04-29",
    )
    assert path is not None
    assert path == vb_dir / "2026-04-29-morning.ogg"
    assert path.is_file()
    assert path.stat().st_size > 0


# ── 2. Idempotencia ─────────────────────────────────────────────────────────

def test_synthesize_brief_audio_idempotent(vb_dir, fake_say):
    """Dos llamadas con mismo texto + date → mismo path, archivo no
    regenerado (mtime estable)."""
    text = "brief idempotencia."
    p1 = voice_brief.synthesize_brief_audio(
        text, kind="morning", date_str="2026-04-29",
    )
    assert p1 is not None
    mtime1 = p1.stat().st_mtime
    # Forzar gap mínimo y backdatear el archivo para detectar regeneración.
    backdated = mtime1 - 10
    os.utime(p1, (backdated, backdated))
    # Segunda llamada — debería cortocircuitar y devolver el mismo path
    # SIN regenerar (mtime queda en el backdated).
    p2 = voice_brief.synthesize_brief_audio(
        text, kind="morning", date_str="2026-04-29",
    )
    assert p2 == p1
    assert p2.stat().st_mtime == pytest.approx(backdated, abs=1.0)


# ── 3. Trim a 4000 chars ────────────────────────────────────────────────────

def test_synthesize_brief_audio_trims_long_text(vb_dir, fake_say, monkeypatch):
    """Texto > 4000 chars → trim al primer 4000 + ``...``. Inspeccionamos
    el comando de ``say`` para confirmar que recibió texto cortado."""
    captured_text = []

    def capture_run(cmd, *args, **kwargs):
        if cmd[0] == "say":
            captured_text.append(cmd[-1])
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"AIFF\x00data" * 100)
        elif cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"OggS data" * 50)
        return mock.Mock(returncode=0)

    monkeypatch.setattr(voice_brief.subprocess, "run", capture_run)

    long_text = "a" * 5000
    path = voice_brief.synthesize_brief_audio(
        long_text, kind="morning", date_str="2026-04-30",
    )
    assert path is not None
    assert len(captured_text) == 1
    spoken = captured_text[0]
    assert len(spoken) <= 4003  # 4000 + "..."
    assert spoken.endswith("...")


# ── 4. CLI voice-brief generate ─────────────────────────────────────────────

def test_cli_voice_brief_generate_with_text(vb_dir, fake_say):
    """``rag voice-brief generate --date 2026-04-29 --text "test"`` corre,
    crea el archivo, exit 0."""
    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["voice-brief", "generate", "--date", "2026-04-29",
         "--text", "test brief en español rioplatense"],
    )
    # Debug si falla.
    if result.exit_code != 0:
        print("STDOUT:", result.output)
        print("EXC:", result.exception)
    assert result.exit_code == 0
    expected = vb_dir / "2026-04-29-morning.ogg"
    assert expected.is_file()


# ── 5. CLI morning --voice --dry-run ───────────────────────────────────────

def test_cli_morning_voice_dry_run_does_not_break(monkeypatch, vb_dir, fake_say):
    """``rag morning --voice --dry-run`` no crashea con TTS mockeado.

    En dry-run el morning prints el body y sale antes del push, así
    que el audio no se genera — pero queremos asegurar que la flag se
    parsea OK (no AttributeError, no missing arg en la signature).
    """
    runner = CliRunner()
    # Mock las funciones costosas del morning para que no toque vault/LLM:
    # devolvemos evidence vacía para que el comando salga temprano vía
    # el guard "Mañana en blanco". Esto ejercita el parsing de --voice
    # sin requerir vault o ollama.
    monkeypatch.setattr(rag, "_collect_morning_evidence",
                        lambda *a, **k: {
                            "recent_notes": [], "inbox_pending": [],
                            "todos": [], "new_contradictions": [],
                            "low_conf_queries": [],
                        })
    monkeypatch.setattr(rag, "_diff_brief_signal", lambda: None)

    result = runner.invoke(rag.cli, ["morning", "--voice", "--dry-run"])
    if result.exit_code != 0:
        print("STDOUT:", result.output)
        print("EXC:", result.exception)
    assert result.exit_code == 0


# ── 6. Fallback: sin TTS → None silent ──────────────────────────────────────

def test_synthesize_brief_audio_no_say_returns_none(vb_dir, no_say):
    """Sin ``say`` en el PATH → ``synthesize_brief_audio`` devuelve None
    sin crash, sin escribir nada al disk."""
    path = voice_brief.synthesize_brief_audio(
        "hola", kind="morning", date_str="2026-04-29",
    )
    assert path is None
    # Tampoco debe haber creado el dir vacío... bueno, sí lo creamos en
    # ``_audio_path_for`` para los casos donde sí funciona. Lo acepto;
    # lo importante es que NO hay archivo dentro.
    if vb_dir.is_dir():
        assert list(vb_dir.iterdir()) == []


# ── 7. strip_markdown_for_speech sanea correctamente ───────────────────────

def test_strip_markdown_for_speech_cleans_wikilinks_code_footer():
    """Wikilinks → label, code blocks fuera, footer brief fuera, headings
    sin ``#``."""
    raw = (
        "---\n"
        "type: morning-brief\n"
        "date: '2026-04-29'\n"
        "---\n\n"
        "# Morning brief — 2026-04-29\n\n"
        "Hoy enfocate en [[02-Areas/Foco|Foco profundo]]. "
        "Mirá `make test` antes de commitear.\n\n"
        "```python\nprint('hello')\n```\n\n"
        "- item 1\n"
        "- item 2\n\n"
        "**Importante:** revisá [Notas](02-Areas/Notas.md).\n\n"
        "_brief:02-Areas/Briefs/2026-04-29-morning.md_"
    )
    clean = voice_brief.strip_markdown_for_speech(raw)
    # Frontmatter fuera.
    assert "type: morning-brief" not in clean
    # Heading sin markup.
    assert "Morning brief — 2026-04-29" in clean
    assert "# " not in clean
    # Wikilink → alias.
    assert "Foco profundo" in clean
    assert "[[" not in clean
    # Code block fuera.
    assert "print('hello')" not in clean
    # Inline code fuera.
    assert "make test" not in clean
    # Bullets sin ``-``.
    assert "item 1" in clean
    assert "- item" not in clean
    # MD link → label.
    assert "Notas" in clean
    assert "02-Areas/Notas.md" not in clean
    # Footer brief fuera.
    assert "_brief:" not in clean


# ── 8. send_audio_to_whatsapp ──────────────────────────────────────────────

def test_send_audio_to_whatsapp_silent_fail_when_file_missing(tmp_path):
    """Sin archivo → False sin POST."""
    missing = tmp_path / "nope.ogg"
    ok = voice_brief.send_audio_to_whatsapp("test@s.whatsapp.net", missing)
    assert ok is False


def test_send_audio_to_whatsapp_posts_media_path(tmp_path, monkeypatch):
    """Archivo válido → POST con ``{recipient, media_path}``. Mockeamos
    ``urllib.request.urlopen`` para verificar el body sin red real."""
    audio = tmp_path / "ok.ogg"
    audio.write_bytes(b"OggS\x00\x02fakeogg")
    captured = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok = voice_brief.send_audio_to_whatsapp("test@s.whatsapp.net", audio)
    assert ok is True
    assert "/api/send" in captured["url"]
    body = captured["body"].decode("utf-8")
    assert '"recipient": "test@s.whatsapp.net"' in body
    assert '"media_path"' in body
    assert str(audio) in body


# ── 9. cleanup_old_voice_briefs ────────────────────────────────────────────

def test_cleanup_old_voice_briefs_removes_old_only(vb_dir):
    """Archivos viejos (>30d) borrados; recientes preservados."""
    vb_dir.mkdir(parents=True, exist_ok=True)
    old = vb_dir / "old.ogg"
    new = vb_dir / "new.ogg"
    old.write_bytes(b"old" * 100)
    new.write_bytes(b"new" * 100)
    # Backdate "old" a 60 días.
    sixty_days_ago = time.time() - 60 * 86400
    os.utime(old, (sixty_days_ago, sixty_days_ago))
    res = voice_brief.cleanup_old_voice_briefs(ttl_days=30)
    assert res["deleted"] == 1
    assert res["bytes_freed"] > 0
    assert res["errors"] == []
    assert not old.exists()
    assert new.exists()


# ── 10. _brief_push_to_whatsapp con audio_path ──────────────────────────────

def test_brief_push_with_audio_sends_audio_then_text_with_marker(
    tmp_path, monkeypatch,
):
    """Cuando audio_path está set + send OK → audio sale primero
    (vía send_audio_to_whatsapp) + texto incluye "(audio arriba ↑)"
    intercalado entre body y footer. Footer queda en última línea."""
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"OggS data")
    monkeypatch.setattr(
        rag, "_ambient_config", lambda: {"jid": "test@s.whatsapp.net"},
    )
    sent_text: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent_text.append((jid, text)) or True,
    )
    monkeypatch.setattr(rag, "_ambient_log_event", lambda payload: None)

    sent_audio: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "rag.voice_brief.send_audio_to_whatsapp",
        lambda jid, path: sent_audio.append((jid, path)) or True,
    )

    ok = rag._brief_push_to_whatsapp(
        "Morning 2026-04-29",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "tres reuniones hoy y un deadline.",
        audio_path=audio,
    )
    assert ok is True
    # Audio salió primero.
    assert len(sent_audio) == 1
    assert sent_audio[0] == ("test@s.whatsapp.net", audio)
    # Texto salió segundo, incluye marker, footer intacto al final.
    assert len(sent_text) == 1
    body = sent_text[0][1]
    assert "(audio arriba" in body
    assert body.endswith("_brief:02-Areas/Briefs/2026-04-29-morning.md_")
    # Última línea no-vacía == footer.
    last = next(
        line.strip() for line in reversed(body.split("\n")) if line.strip()
    )
    assert last == "_brief:02-Areas/Briefs/2026-04-29-morning.md_"


def test_brief_push_audio_failure_falls_back_to_text_only(
    tmp_path, monkeypatch,
):
    """Audio send falla → marker NO se incluye, texto sale igual con
    footer al final (fallback transparente)."""
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"OggS data")
    monkeypatch.setattr(
        rag, "_ambient_config", lambda: {"jid": "test@s.whatsapp.net"},
    )
    sent_text: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent_text.append((jid, text)) or True,
    )
    monkeypatch.setattr(rag, "_ambient_log_event", lambda payload: None)
    monkeypatch.setattr(
        "rag.voice_brief.send_audio_to_whatsapp",
        lambda jid, path: False,
    )

    rag._brief_push_to_whatsapp(
        "Morning",
        "02-Areas/Briefs/2026-04-29-morning.md",
        "body content",
        audio_path=audio,
    )
    assert len(sent_text) == 1
    body = sent_text[0][1]
    assert "(audio arriba" not in body
    assert body.endswith("_brief:02-Areas/Briefs/2026-04-29-morning.md_")
