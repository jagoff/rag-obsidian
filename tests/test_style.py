"""Style fingerprint — tests del extractor + filter + persistence."""
from __future__ import annotations

import sqlite3

import pytest

from rag import style as style_mod
from rag.style import (
    _filter_user_messages,
    extract_features,
    render_markdown,
)


# ── Filter ───────────────────────────────────────────────────────────────


def test_filter_drops_bot_replies():
    raw = [
        "​Recordatorio creado",  # zero-width-space prefix
        "che dale escribime",
        "​otra reply del bot",
    ]
    assert _filter_user_messages(raw) == ["che dale escribime"]


def test_filter_drops_commands():
    raw = ["/enable_ambient", "/si", "fuaa quedó re bien"]
    assert _filter_user_messages(raw) == ["fuaa quedó re bien"]


def test_filter_drops_too_short_or_too_long():
    raw = ["hi", "ok", "che cómo va", "x" * 5000]
    assert _filter_user_messages(raw) == ["che cómo va"]


def test_filter_drops_pure_url():
    raw = [
        "https://example.com",
        "mirá esto https://example.com está bueno",
    ]
    assert _filter_user_messages(raw) == [
        "mirá esto https://example.com está bueno",
    ]


def test_filter_strips_whitespace():
    assert _filter_user_messages(["  ", "\n\n", "che dale"]) == ["che dale"]


# ── Extractor ────────────────────────────────────────────────────────────


def test_extract_empty_returns_insufficient():
    f = extract_features([])
    assert f["n_messages"] == 0
    assert f["insufficient_data"] is True


def test_extract_basic_counts_and_lens():
    msgs = ["dale", "joya", "fuaa quedó re bien", "che cómo va"]
    f = extract_features(msgs)

    assert f["n_messages"] == 4
    assert f["avg_chars"] > 0
    assert f["p50_chars"] >= 4
    assert not f["insufficient_data"]


def test_extract_openers_lowercased_and_punct_stripped():
    msgs = ["Dale,", "dale!", "che?", "che..."]
    f = extract_features(msgs)
    openers = dict(f["openers_top"])

    assert openers.get("dale") == 2
    assert openers.get("che") == 2


def test_extract_voseo_dominance():
    msgs = [
        "vos podés hacer lo que quieras",
        "fijate que tenés que escribir",
        "mirá esto",
    ]
    f = extract_features(msgs)

    assert f["voseo_hits"] >= 5
    assert f["tuteo_hits"] == 0
    assert f["voseo_dominance"] == 1.0


def test_extract_tuteo_detected():
    msgs = ["tú puedes hacerlo", "fíjate tú", "mira eso"]
    f = extract_features(msgs)

    assert f["tuteo_hits"] >= 3


def test_extract_slang_argentino():
    msgs = ["che dale joya", "fuaa quedó copado", "tranqui pibe"]
    f = extract_features(msgs)

    # che, dale, joya, fuaa, copado, tranqui, pibe = 7
    assert f["slang_hits"] >= 6


def test_extract_re_prefix_pattern():
    msgs = ["re bien", "está re copado", "re mal todo"]
    f = extract_features(msgs)
    assert f["re_prefix_hits"] >= 3


def test_extract_emoji_rate():
    msgs = ["che dale", "joya 🔥", "fuaa 😂", "todo bien"]
    f = extract_features(msgs)
    assert f["emoji_rate"] == 0.5


def test_extract_lowercase_only_rate():
    msgs = ["che dale", "Hola", "todo bien", "fuaa"]
    f = extract_features(msgs)
    # 3 lowercase / 4 total
    assert f["lowercase_only_rate"] == 0.75


def test_extract_question_open_rate():
    msgs = ["¿cómo va?", "che", "¿qué onda?"]
    f = extract_features(msgs)
    # 2/3 abren con ¿
    assert f["question_open_rate"] == round(2 / 3, 3)


def test_extract_laugh_typical_length():
    msgs = ["jaja", "jajaja", "jajaja sí", "jajajaja"]
    f = extract_features(msgs)
    # most_common length: jajaja = 3 ja's, aparece 2x
    assert f["laugh_typical_jas"] == 3


def test_extract_abbreviations():
    msgs = ["tmb voy", "xq no?", "tb estoy en casa"]
    f = extract_features(msgs)
    assert f["abbrev_hits"] >= 3


# ── Markdown render ──────────────────────────────────────────────────────


def test_render_markdown_includes_n_messages():
    snap = {
        "computed_at_iso": "2026-05-09T22:00:00",
        "window_days": 90,
        "n_messages": 1234,
        "features": extract_features([
            "che dale joya", "fuaa quedó re bien", "todo bien"
        ]),
    }
    md = render_markdown(snap)
    assert "1234" in md
    assert "Cómo escribís" in md
    assert "Openers favoritos" in md
    assert "voseo" in md.lower()


def test_render_markdown_handles_empty_features():
    snap = {"features": {"insufficient_data": True}}
    md = render_markdown(snap)
    assert "Sin datos suficientes" in md


# ── Persistence (con telemetry.db override) ─────────────────────────────


@pytest.fixture
def isolated_telemetry(tmp_path, monkeypatch):
    """Apunta DB_PATH a tmp_path para que el test no toque la real."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    yield tmp_path


def test_ensure_table_idempotent(isolated_telemetry):
    style_mod._ensure_table()
    style_mod._ensure_table()  # segundo call no debe romper

    db = isolated_telemetry / "telemetry.db"
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='rag_style_fingerprint'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1


def test_persist_and_load_latest_roundtrip(isolated_telemetry):
    msgs = ["che dale", "fuaa joya", "todo bien"]
    f = extract_features(msgs)
    rowid = style_mod._persist(f, window_days=30, content_hash="abc123")

    assert rowid > 0

    snap = style_mod.load_latest()
    assert snap is not None
    assert snap["n_messages"] == 3
    assert snap["window_days"] == 30
    assert snap["content_hash"] == "abc123"
    assert snap["features"]["n_messages"] == 3


def test_load_latest_returns_none_on_empty_table(isolated_telemetry):
    # Table no creada todavía → load_latest crea tabla y retorna None.
    snap = style_mod.load_latest()
    assert snap is None


def test_persist_keeps_history(isolated_telemetry):
    """Refresh múltiple no sobrescribe — guarda historial."""
    f1 = extract_features(["primero"])
    f2 = extract_features(["segundo", "tercero"])
    style_mod._persist(f1, window_days=30, content_hash="h1")
    style_mod._persist(f2, window_days=30, content_hash="h2")

    snap = style_mod.load_latest()
    # load_latest devuelve el más reciente.
    assert snap["content_hash"] == "h2"
    assert snap["n_messages"] == 2
