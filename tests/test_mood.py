"""Tests para `rag/mood.py` — scorers Spotify + journal (commit 2 del
pipeline mood).

Cubre:

  Spotify:
  1. Window vacío → no señales emitidas.
  2. Artist mood lookup: solo artistas conocidos contribuyen.
  3. Compulsive-repeat: ≥ N plays mismo track dispara señal negativa.
  4. Late-night: tracks ≥ 02:00 con mood ≤ 0 disparan señal.
  5. Combined: 6h con mix de bajón + repeat + late-night → 3 señales.

  Journal:
  6. Vault sin notas recientes → vacío.
  7. Nota con keyword negativa → señal directa, NO LLM call.
  8. Nota larga sin keyword → LLM call (mockeado).
  9. Nota larga sin keyword + LLM devuelve neutro (|val|<0.3) → no signal.
  10. Cache LLM por (path, mtime): segunda llamada no re-llama al LLM.

  Flag:
  11. RAG_MOOD_ENABLED=0 → _persist_signal no escribe en DB.
  12. RAG_MOOD_ENABLED=1 → _persist_signal escribe.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

import rag
from rag import mood


_SPOTIFY_DDL = """
CREATE TABLE rag_spotify_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    name TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    state TEXT,
    duration_ms INTEGER,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    date TEXT NOT NULL
)
"""

_MOOD_SIGNALS_DDL = """
CREATE TABLE rag_mood_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    signal_kind TEXT NOT NULL,
    value REAL NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    evidence TEXT
)
"""


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Temp DB con `rag_spotify_log` y `rag_mood_signals`. Patchea
    `rag._ragvec_state_conn` para que apunte acá."""
    db_path = tmp_path / "telemetry.db"
    seed = sqlite3.connect(str(db_path))
    seed.execute(_SPOTIFY_DDL)
    seed.execute(_MOOD_SIGNALS_DDL)
    seed.commit()
    seed.close()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    yield db_path


@pytest.fixture
def mood_enabled(monkeypatch):
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    """Reset caches del módulo mood antes de cada test (artist table
    + sentiment LLM)."""
    monkeypatch.setattr(mood, "_ARTIST_MOOD_CACHE", None)
    monkeypatch.setattr(mood, "_SENTIMENT_LLM_CACHE", {})


def _insert_track(
    db_path: Path, *, track_id: str, artist: str, name: str = "track",
    first_seen: float, last_seen: float | None = None,
) -> None:
    last_seen = last_seen if last_seen is not None else first_seen + 180.0
    date = time.strftime("%Y-%m-%d", time.localtime(first_seen))
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO rag_spotify_log "
        "(track_id, name, artist, album, state, duration_ms, "
        " first_seen, last_seen, date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (track_id, name, artist, "album", "playing", 200_000,
         first_seen, last_seen, date),
    )
    conn.commit()
    conn.close()


# ─── Spotify ───────────────────────────────────────────────────────────────


def test_spotify_empty_window_no_signals(temp_db, mood_enabled):
    signals = mood.score_spotify_window(now=time.time(), window_h=6, persist=False)
    assert signals == []


def test_spotify_artist_lookup_emits_weighted_signal(temp_db, mood_enabled):
    """Spinetta (mood -0.6) escuchado 10min + Bizarrap (mood +0.6) 2min →
    weighted-avg negativo (Spinetta pesa 5x más por duration)."""
    now = time.time()
    _insert_track(temp_db, track_id="t1", artist="Spinetta",
                  first_seen=now - 1800, last_seen=now - 1200)  # 600s
    _insert_track(temp_db, track_id="t2", artist="Bizarrap",
                  first_seen=now - 600, last_seen=now - 480)    # 120s
    signals = mood.score_spotify_window(now=now, window_h=6, persist=False)
    artist_signals = [s for s in signals if s["signal_kind"] == "artist_mood_lookup"]
    assert len(artist_signals) == 1
    val = artist_signals[0]["value"]
    # Weighted: (-0.6 * 600 + 0.6 * 120) / 720 = -0.4
    assert -0.5 < val < -0.3
    ev = artist_signals[0]["evidence"]
    assert ev["n_tracks_total"] == 2
    assert ev["n_tracks_matched"] == 2


def test_spotify_artist_lookup_no_signal_if_no_match(temp_db, mood_enabled):
    """Si ningún artista de la window está en la tabla, no emitir señal
    (NO emitir 'neutro 0' que confunde al agregador)."""
    now = time.time()
    _insert_track(temp_db, track_id="t1", artist="UnknownArtist123",
                  first_seen=now - 1800)
    signals = mood.score_spotify_window(now=now, window_h=6, persist=False)
    assert [s for s in signals if s["signal_kind"] == "artist_mood_lookup"] == []


def test_spotify_compulsive_repeat_emits_signal(temp_db, mood_enabled):
    """Mismo track_id 5 veces en window → señal compulsive_repeat
    negativa con magnitud ≥ 0.5."""
    now = time.time()
    for i in range(5):
        _insert_track(temp_db, track_id="repeat-track", artist="UnknownArtist",
                      first_seen=now - 3600 + i * 200,
                      last_seen=now - 3600 + i * 200 + 180)
    signals = mood.score_spotify_window(now=now, window_h=6, persist=False)
    repeat = [s for s in signals if s["signal_kind"] == "compulsive_repeat"]
    assert len(repeat) == 1
    assert repeat[0]["value"] <= -0.5
    assert repeat[0]["evidence"]["plays"] == 5


def test_spotify_compulsive_repeat_below_threshold_no_signal(temp_db, mood_enabled):
    """4 plays del mismo track (threshold = 5) → no señal."""
    now = time.time()
    for i in range(4):
        _insert_track(temp_db, track_id="repeat-track", artist="UnknownArtist",
                      first_seen=now - 3600 + i * 200,
                      last_seen=now - 3600 + i * 200 + 180)
    signals = mood.score_spotify_window(now=now, window_h=6, persist=False)
    assert [s for s in signals if s["signal_kind"] == "compulsive_repeat"] == []


def test_spotify_late_night_only_with_sad_mood(temp_db, mood_enabled):
    """Tracks a las 03:00 — uno de mood -0.6 (Spinetta) y uno de
    mood +0.5 (Duki). Solo el Spinetta dispara late-night signal."""
    now = time.time()
    # Construir un timestamp con hour=3.
    base = time.localtime(now)
    late_night_ts = time.mktime(time.struct_time(
        (base.tm_year, base.tm_mon, base.tm_mday, 3, 30, 0,
         base.tm_wday, base.tm_yday, base.tm_isdst),
    ))
    _insert_track(temp_db, track_id="t1", artist="Spinetta",
                  first_seen=late_night_ts,
                  last_seen=late_night_ts + 300)
    _insert_track(temp_db, track_id="t2", artist="Duki",
                  first_seen=late_night_ts + 600,
                  last_seen=late_night_ts + 900)
    signals = mood.score_spotify_window(
        now=late_night_ts + 1000, window_h=6, persist=False,
    )
    late = [s for s in signals if s["signal_kind"] == "late_night_listening"]
    assert len(late) == 1
    assert late[0]["value"] <= -0.3
    assert late[0]["evidence"]["n_late_night_sad_tracks"] == 1


def test_spotify_persist_writes_to_db(temp_db, mood_enabled):
    """Con persist=True (default), las señales se escriben en
    `rag_mood_signals`."""
    now = time.time()
    _insert_track(temp_db, track_id="t1", artist="Spinetta",
                  first_seen=now - 1800)
    mood.score_spotify_window(now=now, window_h=6, persist=True)
    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute(
        "SELECT source, signal_kind FROM rag_mood_signals"
    ).fetchall()
    conn.close()
    sources = [r[0] for r in rows]
    assert "spotify" in sources


def test_spotify_persist_off_does_not_write(temp_db, mood_enabled):
    """persist=False → solo devuelve dicts, NO escribe en DB."""
    now = time.time()
    _insert_track(temp_db, track_id="t1", artist="Spinetta",
                  first_seen=now - 1800)
    signals = mood.score_spotify_window(now=now, window_h=6, persist=False)
    assert len(signals) >= 1
    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute("SELECT COUNT(*) FROM rag_mood_signals").fetchone()
    conn.close()
    assert rows[0] == 0


# ─── Journal ───────────────────────────────────────────────────────────────


@pytest.fixture
def temp_vault(tmp_path):
    """Vault tmp con folders 00-Inbox y 02-Areas/journal."""
    (tmp_path / "00-Inbox").mkdir()
    (tmp_path / "02-Areas" / "journal").mkdir(parents=True)
    return tmp_path


def test_journal_empty_no_signals(temp_db, temp_vault, mood_enabled):
    signals = mood.score_journal_recent(
        within_h=24, vault=temp_vault, persist=False, use_llm=False,
    )
    assert signals == []


def test_journal_keyword_negative_emits_direct_signal(temp_db, temp_vault, mood_enabled):
    """Nota con 'bajón' → señal keyword_negative directa, NO necesita LLM."""
    note = temp_vault / "00-Inbox" / "today.md"
    note.write_text("Re bajón hoy, no doy más con la semana.", encoding="utf-8")
    signals = mood.score_journal_recent(
        within_h=24, vault=temp_vault, persist=False, use_llm=False,
    )
    kw = [s for s in signals if s["signal_kind"] == "keyword_negative"]
    assert len(kw) == 1
    assert kw[0]["value"] < 0
    # Múltiples keywords = magnitud mayor
    assert "bajón" in kw[0]["evidence"]["keywords"]


def test_journal_no_keyword_uses_llm(temp_db, temp_vault, mood_enabled, monkeypatch):
    """Nota larga sin keyword → llamada al LLM. Mock devuelve -0.6."""
    note = temp_vault / "02-Areas" / "journal" / "n1.md"
    # Texto suficientemente largo (>80 chars) y sin matches al regex.
    note.write_text(
        "Hoy tuve una semana muy difícil en términos emocionales y profesionales. "
        "Las cosas no salen como esperaba y siento mucha presión para entregar resultados.",
        encoding="utf-8",
    )

    class _FakeChat:
        def chat(self, **kwargs):
            return {"message": {"content": "-0.6"}}

    monkeypatch.setattr(rag, "_helper_client", lambda: _FakeChat())
    signals = mood.score_journal_recent(
        within_h=24, vault=temp_vault, persist=False, use_llm=True,
    )
    sent = [s for s in signals if s["signal_kind"] == "note_sentiment"]
    assert len(sent) == 1
    assert sent[0]["value"] == pytest.approx(-0.6)


def test_journal_llm_neutral_no_signal(temp_db, temp_vault, mood_enabled, monkeypatch):
    """Si el LLM devuelve algo en (-0.3, 0.3), no emitir señal (filtro
    de neutralidad para no contaminar el agregador con ruido)."""
    note = temp_vault / "00-Inbox" / "neutral.md"
    note.write_text(
        "Hoy compré pan y leche en el chino de la esquina. Después fui al banco "
        "para pagar la luz, todo normal sin contratiempos.",
        encoding="utf-8",
    )

    class _FakeChat:
        def chat(self, **kwargs):
            return {"message": {"content": "0.05"}}

    monkeypatch.setattr(rag, "_helper_client", lambda: _FakeChat())
    signals = mood.score_journal_recent(
        within_h=24, vault=temp_vault, persist=False, use_llm=True,
    )
    assert [s for s in signals if s["signal_kind"] == "note_sentiment"] == []


def test_journal_llm_cache_hits_on_same_path_mtime(temp_db, temp_vault, mood_enabled, monkeypatch):
    """Segunda llamada al mismo file (sin tocar mtime) usa cache, NO
    llama al LLM una segunda vez."""
    note = temp_vault / "00-Inbox" / "n2.md"
    note.write_text(
        "Una jornada complicada que me dejó pensando mucho en cosas pendientes "
        "y obligaciones que no termino de cerrar nunca a tiempo en general.",
        encoding="utf-8",
    )

    call_count = {"n": 0}

    class _CountingChat:
        def chat(self, **kwargs):
            call_count["n"] += 1
            return {"message": {"content": "-0.5"}}

    monkeypatch.setattr(rag, "_helper_client", lambda: _CountingChat())
    mood.score_journal_recent(within_h=24, vault=temp_vault,
                              persist=False, use_llm=True)
    mood.score_journal_recent(within_h=24, vault=temp_vault,
                              persist=False, use_llm=True)
    assert call_count["n"] == 1


def test_journal_short_note_skips_llm(temp_db, temp_vault, mood_enabled, monkeypatch):
    """Nota < MIN_CHARS_FOR_LLM y sin keyword → no signal (no malgastar
    LLM en notas tipo 'comprar pan')."""
    note = temp_vault / "00-Inbox" / "short.md"
    note.write_text("comprar pan", encoding="utf-8")

    class _NeverCalled:
        def chat(self, **kwargs):
            raise RuntimeError("LLM no debería llamarse para notas cortas")

    monkeypatch.setattr(rag, "_helper_client", lambda: _NeverCalled())
    signals = mood.score_journal_recent(
        within_h=24, vault=temp_vault, persist=False, use_llm=True,
    )
    assert signals == []


# ─── Flag ──────────────────────────────────────────────────────────────────


def test_flag_off_persist_signal_is_noop(temp_db, monkeypatch):
    """Sin RAG_MOOD_ENABLED, _persist_signal no escribe."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    mood._persist_signal("spotify", "test", -0.5, evidence={"x": 1})
    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute("SELECT COUNT(*) FROM rag_mood_signals").fetchone()
    conn.close()
    assert rows[0] == 0


def test_flag_on_persist_signal_writes(temp_db, mood_enabled):
    """Con RAG_MOOD_ENABLED=1, _persist_signal escribe."""
    mood._persist_signal("spotify", "test", -0.5, evidence={"x": 1})
    conn = sqlite3.connect(str(temp_db))
    rows = conn.execute(
        "SELECT source, signal_kind, value, evidence FROM rag_mood_signals"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "spotify"
    assert rows[0][1] == "test"
    assert rows[0][2] == pytest.approx(-0.5)
    assert json.loads(rows[0][3]) == {"x": 1}


def test_artist_table_loads_and_lowercases():
    """La tabla se carga con keys lowercase para que el lookup funcione
    aunque rag_spotify_log guarde el artista con caps mixtas."""
    table = mood._load_artist_mood_table()
    assert "spinetta" in table
    # Algunos sample assertions sobre el sign del mood.
    assert table["spinetta"] < 0
    assert table["bizarrap"] > 0
    # Lookup case-insensitive.
    assert mood._lookup_artist_mood("SPINETTA") == table["spinetta"]
    assert mood._lookup_artist_mood("Bizarrap") == table["bizarrap"]
    assert mood._lookup_artist_mood("Random Unknown 9000") is None
