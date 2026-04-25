"""Tests for the `reading_backlog` anticipatory signal.

Cubre:
1. Empty vault → []
2. 5 notas to-read viejas (count <10) → []
3. 10 notas to-read viejas → emit score 0.5
4. 40 notas to-read viejas → emit score 1.0
5. 15 notas to-read pero todas <7d → []
6. Notas con `status: read` → no cuentan (control negativo)
7. Notas con tag `#to-read` (inline en body) vs frontmatter → ambas cuentan
8. dedup_key varies con week ISO (cross-week)
9. Signal registrada en `_ANTICIPATE_SIGNALS`

Mocks: monkeypatch `rag._resolve_vault_path` para apuntar al tmp_path. Cada
nota se escribe con `_write_note_with_age()` que setea `mtime` en el pasado
via `os.utime`.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

import rag
from rag_anticipate.signals.reading_backlog import (
    _count_reading_backlog,
    _in_reading_folder,
    _is_to_read,
    reading_backlog_signal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Vault temporal aislado. El signal lee `_resolve_vault_path()` —
    monkeypatch apunta al tmp_path para que la rglob no escanee el vault real."""
    vault = tmp_path / "vault"
    (vault / "03-Resources").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _write_note_with_age(vault, rel, body, age_days):
    """Crea una nota con body + mtime hace `age_days` días."""
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    ts = (datetime.now() - timedelta(days=age_days)).timestamp()
    os.utime(path, (ts, ts))
    return path


def _to_read_fm(extra: str = "") -> str:
    """Body con frontmatter `status: to-read`."""
    return f"---\nstatus: to-read\n{extra}---\n\nContenido de la nota.\n"


def _unread_fm_tag() -> str:
    """Body con frontmatter `tags: [unread]`."""
    return "---\ntags: [unread, article]\n---\n\nContenido.\n"


# ── Helper-level tests ───────────────────────────────────────────────────────


def test_is_to_read_status_frontmatter():
    """`status: to-read` en el frontmatter → True."""
    assert _is_to_read("---\nstatus: to-read\n---\n\nbody") is True
    assert _is_to_read("---\nstatus: unread\n---\n\nbody") is True
    # Case-insensitive
    assert _is_to_read("---\nstatus: To-Read\n---\n\nbody") is True


def test_is_to_read_status_read_does_not_count():
    """`status: read` (ya leído) → False."""
    assert _is_to_read("---\nstatus: read\n---\n\nbody") is False
    assert _is_to_read("---\nstatus: done\n---\n\nbody") is False


def test_is_to_read_tag_in_frontmatter_list():
    """`tags: [to-read]` o `tags: [unread]` → True."""
    assert _is_to_read("---\ntags: [to-read]\n---\n\nbody") is True
    assert _is_to_read("---\ntags: [unread, article]\n---\n\nbody") is True
    assert _is_to_read("---\ntags: [foo, bar]\n---\n\nbody") is False


def test_is_to_read_tag_scalar_string():
    """`tags: to-read, article` (formato YAML scalar) → True."""
    assert _is_to_read("---\ntags: to-read, article\n---\n\nbody") is True


def test_is_to_read_inline_hashtag():
    """`#to-read` o `#unread` inline en body → True."""
    assert _is_to_read("---\n---\n\nMarcado como #to-read para leer luego.") is True
    assert _is_to_read("Sin frontmatter, solo #unread inline.") is True
    # Sufijo de palabra no debería matchear (boundary)
    assert _is_to_read("Algo como #to-readness no cuenta.") is False


def test_is_to_read_accepts_dict_fm():
    """También acepta un dict (frontmatter ya parseado)."""
    assert _is_to_read({"status": "to-read"}) is True
    assert _is_to_read({"tags": ["unread"]}) is True
    assert _is_to_read({"status": "read"}) is False
    assert _is_to_read({}) is False


def test_in_reading_folder():
    """Match prefix case-insensitive contra `03-Resources/Reading/` y `Reading/`."""
    assert _in_reading_folder("03-Resources/Reading/article.md") is True
    assert _in_reading_folder("03-resources/reading/foo.md") is True  # lowercase
    assert _in_reading_folder("Reading/foo.md") is True
    assert _in_reading_folder("00-Inbox/foo.md") is False
    assert _in_reading_folder("02-Areas/something.md") is False


# ── Signal tests ─────────────────────────────────────────────────────────────


def test_empty_vault_returns_empty(mock_vault):
    """1. Vault sin notas → []."""
    out = reading_backlog_signal(datetime.now())
    assert out == []


def test_below_threshold_returns_empty(mock_vault):
    """2. 5 notas to-read viejas (count <10) → []."""
    for i in range(5):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=10)
    out = reading_backlog_signal(datetime.now())
    assert out == []


def test_exactly_threshold_emits_score_05(mock_vault):
    """3. 10 notas to-read viejas → emit con score 0.5."""
    for i in range(10):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=10)
    out = reading_backlog_signal(datetime.now())
    assert len(out) == 1
    c = out[0]
    assert c.kind == "anticipate-reading_backlog"
    assert c.score == pytest.approx(0.5, abs=0.01)
    assert "10 notas" in c.message
    assert "backlog de lectura" in c.message
    assert c.snooze_hours == 168
    assert c.dedup_key.startswith("reading_backlog:")


def test_well_above_threshold_saturates_at_1(mock_vault):
    """4. 40 notas to-read viejas → score saturado a 1.0."""
    for i in range(40):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=10)
    out = reading_backlog_signal(datetime.now())
    assert len(out) == 1
    c = out[0]
    assert c.score == pytest.approx(1.0, abs=0.001)
    assert "40 notas" in c.message


def test_recent_notes_do_not_count(mock_vault):
    """5. 15 notas to-read pero todas <7d → []. No molestar con capturas
    recientes — el user todavía está en el ciclo de procesar lo nuevo."""
    for i in range(15):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=2)
    out = reading_backlog_signal(datetime.now())
    assert out == []


def test_status_read_does_not_count(mock_vault):
    """6. Notas con `status: read` (ya leídas) no cuentan, aunque sean viejas.
    Mezclamos con suficientes to-read para verificar que solo se cuentan esas."""
    # 8 notas ya leídas (no deben contar)
    for i in range(8):
        _write_note_with_age(
            mock_vault, f"read-{i}.md",
            "---\nstatus: read\n---\n\nya leído",
            age_days=20,
        )
    # 9 notas to-read (también debajo del threshold)
    for i in range(9):
        _write_note_with_age(mock_vault, f"to-read-{i}.md", _to_read_fm(), age_days=20)
    # Total to-read = 9 < 10 → no emit aunque haya 8 "read" extras
    out = reading_backlog_signal(datetime.now())
    assert out == []

    # Verificamos vía counter directo que los 8 "read" no se contaron
    n = _count_reading_backlog(mock_vault, min_age_days=7)
    assert n == 9


def test_inline_hashtag_and_frontmatter_both_count(mock_vault):
    """7. Mix de notas: con `status: to-read` (FM), con `tags: [unread]` (FM
    list) y con `#to-read` inline en body — todas cuentan. Necesitamos ≥10
    para emitir, así que mezclamos 4+3+3 = 10."""
    # 4 con status: to-read
    for i in range(4):
        _write_note_with_age(mock_vault, f"fm-status-{i}.md", _to_read_fm(), age_days=15)
    # 3 con tags: [unread]
    for i in range(3):
        _write_note_with_age(mock_vault, f"fm-tag-{i}.md", _unread_fm_tag(), age_days=15)
    # 3 con inline #to-read en body (sin frontmatter)
    for i in range(3):
        body = "Artículo capturado. Marcado #to-read para revisar."
        _write_note_with_age(mock_vault, f"inline-{i}.md", body, age_days=15)

    n = _count_reading_backlog(mock_vault, min_age_days=7)
    assert n == 10

    out = reading_backlog_signal(datetime.now())
    assert len(out) == 1
    assert "10 notas" in out[0].message


def test_reading_folder_notes_count_without_explicit_status(mock_vault):
    """Bonus: notas en `03-Resources/Reading/` cuentan aunque no tengan
    frontmatter explícito (folder convention)."""
    for i in range(11):
        _write_note_with_age(
            mock_vault, f"03-Resources/Reading/article-{i}.md",
            "Solo el contenido del artículo, sin frontmatter.",
            age_days=10,
        )
    out = reading_backlog_signal(datetime.now())
    assert len(out) == 1
    assert "11 notas" in out[0].message


def test_dedup_key_varies_with_week_iso(mock_vault):
    """8. dedup_key incluye la semana ISO → dos `now` en semanas distintas
    producen dedup_keys distintos."""
    # Llenamos el vault con 12 notas viejas para garantizar emisión
    for i in range(12):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=14)

    # Dos `now` separados por ~14 días (distintas semanas ISO)
    now_a = datetime(2024, 5, 6, 9, 0, 0)   # ISO 2024-W19
    now_b = datetime(2024, 5, 20, 9, 0, 0)  # ISO 2024-W21

    out_a = reading_backlog_signal(now_a)
    out_b = reading_backlog_signal(now_b)
    assert len(out_a) == 1 and len(out_b) == 1

    key_a = out_a[0].dedup_key
    key_b = out_b[0].dedup_key
    assert key_a != key_b
    assert key_a.startswith("reading_backlog:")
    assert key_b.startswith("reading_backlog:")
    # Formato: `reading_backlog:YYYY-Www`
    assert "2024-W19" in key_a
    assert "2024-W21" in key_b


def test_dedup_key_stable_within_same_week(mock_vault):
    """Dos runs el mismo día → mismo dedup_key (idempotencia semanal)."""
    for i in range(11):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=10)
    now = datetime(2024, 5, 6, 9, 0, 0)
    out_1 = reading_backlog_signal(now)
    out_2 = reading_backlog_signal(now)
    assert len(out_1) == 1 and len(out_2) == 1
    assert out_1[0].dedup_key == out_2[0].dedup_key


def test_signal_is_registered():
    """9. La signal está registrada en `rag._ANTICIPATE_SIGNALS`."""
    names = {name for name, _fn in rag._ANTICIPATE_SIGNALS}
    assert "reading_backlog" in names

    # También en SIGNALS del package (registry interno)
    from rag_anticipate.signals.base import SIGNALS
    package_names = {name for name, _fn in SIGNALS}
    assert "reading_backlog" in package_names


def test_returns_max_one_candidate(mock_vault):
    """Contrato: el signal devuelve a lo sumo 1 candidate por run."""
    for i in range(50):
        _write_note_with_age(mock_vault, f"note-{i}.md", _to_read_fm(), age_days=10)
    out = reading_backlog_signal(datetime.now())
    assert len(out) == 1
