"""TTL cleanup para `chat-uploads/` (audit 2026-04-25 R2-Security #6).

El endpoint ``/api/chat/upload-image`` guarda imágenes en
``~/.local/share/obsidian-rag/chat-uploads/<sha256>.<ext>``. Sin un TTL,
el dir crece sin bound. ``rag._cleanup_chat_uploads`` corre desde
``run_maintenance`` cada noche (plist ``com.fer.obsidian-rag-maintenance``)
y borra los que tienen mtime > ``ttl_days`` (default 30, override via
``RAG_CHAT_UPLOADS_TTL_DAYS``).

Los tests redirigen ``Path.home()`` a ``tmp_path`` con monkeypatch para
que la función opere sobre un dir aislado, sin tocar el cache real del
user.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import rag


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_uploads_dir(tmp_path: Path) -> Path:
    """Crea ~/.local/share/obsidian-rag/chat-uploads/ bajo tmp_path."""
    upload_dir = tmp_path / ".local" / "share" / "obsidian-rag" / "chat-uploads"
    upload_dir.mkdir(parents=True)
    return upload_dir


def _set_mtime_days_ago(path: Path, days: float) -> None:
    """Settear mtime/atime de path a `days` días atrás desde ahora."""
    ts = time.time() - (days * 86400)
    os.utime(path, (ts, ts))


# ── Tests ───────────────────────────────────────────────────────────────────

def test_deletes_old_files(tmp_path, monkeypatch):
    """Default 30 días: borra los con mtime > 30d, conserva los frescos."""
    upload_dir = _make_uploads_dir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RAG_CHAT_UPLOADS_TTL_DAYS", raising=False)

    old_file = upload_dir / "old.png"
    old_file.write_bytes(b"old data " * 100)  # 900 bytes
    new_file = upload_dir / "new.png"
    new_file.write_bytes(b"new data")

    _set_mtime_days_ago(old_file, 35)  # > 30d → debe morir

    result = rag._cleanup_chat_uploads()

    assert result["deleted"] == 1
    assert result["bytes_freed"] == 900
    assert result["errors"] == []
    assert not old_file.exists()
    assert new_file.exists()


def test_dir_does_not_exist_returns_empty(tmp_path, monkeypatch):
    """Si chat-uploads/ no existe (instalación fresh), no crashea."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RAG_CHAT_UPLOADS_TTL_DAYS", raising=False)

    # No creamos el dir a propósito.
    result = rag._cleanup_chat_uploads()

    assert result == {"deleted": 0, "bytes_freed": 0, "errors": []}


def test_respects_ttl_days_arg(tmp_path, monkeypatch):
    """``ttl_days=7`` borra archivos con mtime > 7 días, no los más nuevos."""
    upload_dir = _make_uploads_dir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RAG_CHAT_UPLOADS_TTL_DAYS", raising=False)

    fresh = upload_dir / "fresh.png"
    fresh.write_bytes(b"fresh")
    medium = upload_dir / "medium.png"
    medium.write_bytes(b"medium")
    ancient = upload_dir / "ancient.png"
    ancient.write_bytes(b"ancient")

    _set_mtime_days_ago(fresh, 1)      # menor a 7d → keep
    _set_mtime_days_ago(medium, 8)     # > 7d → debe morir con ttl=7
    _set_mtime_days_ago(ancient, 100)  # > 7d → debe morir con ttl=7

    result = rag._cleanup_chat_uploads(ttl_days=7)

    assert result["deleted"] == 2
    assert fresh.exists()
    assert not medium.exists()
    assert not ancient.exists()


def test_env_var_overrides_default(tmp_path, monkeypatch):
    """``RAG_CHAT_UPLOADS_TTL_DAYS`` aplica cuando no se pasa ttl_days."""
    upload_dir = _make_uploads_dir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("RAG_CHAT_UPLOADS_TTL_DAYS", "5")

    f = upload_dir / "img.png"
    f.write_bytes(b"x")
    _set_mtime_days_ago(f, 10)  # > 5d → debe morir con env=5

    result = rag._cleanup_chat_uploads()

    assert result["deleted"] == 1
    assert not f.exists()


def test_skips_subdirectories(tmp_path, monkeypatch):
    """Si alguien deja un subdir adentro (no debería pero por las dudas), no crashea."""
    upload_dir = _make_uploads_dir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RAG_CHAT_UPLOADS_TTL_DAYS", raising=False)

    sub = upload_dir / "weird-subdir"
    sub.mkdir()
    (sub / "deep.png").write_bytes(b"deep")

    f = upload_dir / "old.png"
    f.write_bytes(b"old")
    _set_mtime_days_ago(f, 35)

    result = rag._cleanup_chat_uploads()

    assert result["deleted"] == 1
    assert not f.exists()
    assert sub.is_dir()  # subdir preservado, no es file


def test_per_file_errors_do_not_abort(tmp_path, monkeypatch):
    """Si un archivo falla al borrar, los otros igual se procesan."""
    upload_dir = _make_uploads_dir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RAG_CHAT_UPLOADS_TTL_DAYS", raising=False)

    f1 = upload_dir / "ok.png"
    f1.write_bytes(b"ok")
    _set_mtime_days_ago(f1, 35)

    f2 = upload_dir / "broken.png"
    f2.write_bytes(b"broken")
    _set_mtime_days_ago(f2, 35)

    # Monkeypatchear unlink para que tire en `broken.png` solamente.
    real_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self.name == "broken.png":
            raise PermissionError("simulated lock")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    result = rag._cleanup_chat_uploads()

    assert result["deleted"] == 1
    assert len(result["errors"]) == 1
    assert "broken.png" in result["errors"][0]
    assert not f1.exists()
    assert f2.exists()  # falló el unlink → sigue ahí
