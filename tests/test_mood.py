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
from datetime import datetime
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


# ─── WhatsApp outbound ─────────────────────────────────────────────────────


_WA_MESSAGES_DDL = """
CREATE TABLE messages (
    id TEXT,
    chat_jid TEXT,
    sender TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    is_from_me BOOLEAN,
    media_type TEXT,
    filename TEXT,
    url TEXT
)
"""


@pytest.fixture
def temp_wa_bridge(tmp_path, monkeypatch):
    """Bridge SQLite tmp con la tabla messages; patch _wa_bridge_db_path."""
    db = tmp_path / "messages.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_WA_MESSAGES_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(mood, "_wa_bridge_db_path", lambda: db)
    return db


def _insert_wa_msg(db: Path, content: str, ts: float, is_from_me: bool = True) -> None:
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO messages(id, chat_jid, sender, content, timestamp, is_from_me) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"id-{ts}", "jid", "me", content, iso, 1 if is_from_me else 0),
    )
    conn.commit()
    conn.close()


def test_wa_no_bridge_no_signal(temp_db, monkeypatch, mood_enabled):
    """Si el bridge DB no existe, score devuelve [] sin tirar."""
    monkeypatch.setattr(mood, "_wa_bridge_db_path", lambda: None)
    assert mood.score_wa_outbound_window(persist=False) == []


def test_wa_too_few_msgs_no_signal(temp_db, temp_wa_bridge, mood_enabled):
    """Con < MIN_MSGS_FOR_SIGNAL en window: no signal."""
    now = time.time()
    for i in range(3):
        _insert_wa_msg(temp_wa_bridge, "hola que hace", now - 3600 + i)
    assert mood.score_wa_outbound_window(now=now, persist=False) == []


def test_wa_short_messages_emits_signal(temp_db, temp_wa_bridge, mood_enabled):
    """Avg chars en window < 60% del baseline → signal negative."""
    now = time.time()
    # Baseline (>24h hasta 14d atrás): 30 mensajes promedio largos.
    long_msg = "Te cuento que estuve pensando en lo que hablamos el otro día y la verdad"
    for i in range(30):
        _insert_wa_msg(temp_wa_bridge, long_msg, now - 86400 * 5 + i)
    # Window (últimas 24h): 6 mensajes muy cortos.
    for i in range(6):
        _insert_wa_msg(temp_wa_bridge, "ok", now - 1800 + i)
    signals = mood.score_wa_outbound_window(now=now, persist=False)
    short = [s for s in signals if s["signal_kind"] == "tone_short"]
    assert len(short) == 1
    assert short[0]["value"] < 0
    assert short[0]["evidence"]["msgs_window"] == 6


def test_wa_normal_length_no_signal(temp_db, temp_wa_bridge, mood_enabled):
    """Window con avg chars similar al baseline → no signal."""
    now = time.time()
    msg = "esto es un mensaje de longitud normal cualquiera del usuario"
    # Baseline.
    for i in range(20):
        _insert_wa_msg(temp_wa_bridge, msg, now - 86400 * 5 + i)
    # Window: misma longitud.
    for i in range(7):
        _insert_wa_msg(temp_wa_bridge, msg, now - 1800 + i)
    signals = mood.score_wa_outbound_window(now=now, persist=False)
    assert [s for s in signals if s["signal_kind"] == "tone_short"] == []


# ─── Queries existential ───────────────────────────────────────────────────


_QUERIES_DDL = """
CREATE TABLE rag_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    trace_id TEXT,
    cmd TEXT,
    q TEXT NOT NULL,
    session TEXT
)
"""


@pytest.fixture
def temp_queries_db(tmp_path, monkeypatch):
    """Temp telemetry.db con rag_queries + rag_mood_signals + rag_spotify_log."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_SPOTIFY_DDL)
    conn.execute(_MOOD_SIGNALS_DDL)
    conn.execute(_QUERIES_DDL)
    conn.commit()
    conn.close()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    yield db


def _insert_query(db: Path, q: str, ts: float, cmd: str = "query") -> None:
    iso = datetime.fromtimestamp(ts).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q) VALUES (?, ?, ?)",
        (iso, cmd, q),
    )
    conn.commit()
    conn.close()


def test_queries_no_existential_no_signal(temp_queries_db, mood_enabled):
    now = time.time()
    _insert_query(temp_queries_db, "qué dije sobre ML el mes pasado", now - 1800)
    _insert_query(temp_queries_db, "ayer qué reuniones tuve", now - 1200)
    signals = mood.score_queries_existential(now=now, persist=False)
    assert signals == []


def test_queries_existential_emits_signal(temp_queries_db, mood_enabled):
    now = time.time()
    _insert_query(temp_queries_db, "qué hice este mes, no avanzo", now - 1800)
    _insert_query(temp_queries_db, "siempre lo mismo, cómo salgo", now - 1200)
    signals = mood.score_queries_existential(now=now, persist=False)
    exi = [s for s in signals if s["signal_kind"] == "existential_pattern"]
    assert len(exi) == 1
    assert exi[0]["value"] <= -0.5  # 2 matches → -0.5
    assert exi[0]["evidence"]["n_matched"] == 2


def test_queries_existential_3plus_stronger_signal(temp_queries_db, mood_enabled):
    now = time.time()
    _insert_query(temp_queries_db, "qué hice todo este tiempo", now - 1800)
    _insert_query(temp_queries_db, "estoy estancado", now - 1500)
    _insert_query(temp_queries_db, "no puedo más con esto", now - 1200)
    _insert_query(temp_queries_db, "para qué sigo intentando", now - 800)
    signals = mood.score_queries_existential(now=now, persist=False)
    exi = [s for s in signals if s["signal_kind"] == "existential_pattern"]
    assert len(exi) == 1
    assert exi[0]["value"] == -0.7
    assert exi[0]["evidence"]["n_matched"] == 4


# ─── Calendar density ──────────────────────────────────────────────────────


def test_calendar_few_events_no_signal(temp_db, monkeypatch, mood_enabled):
    """≤ density_threshold-1 events: no signal."""
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: [
            {"title": "A", "start": "09:00", "end": "10:00"},
            {"title": "B", "start": "11:00", "end": "12:00"},
        ],
    )
    assert mood.score_calendar_density(persist=False) == []


def test_calendar_density_overload_emits_signal(temp_db, monkeypatch, mood_enabled):
    """≥ 6 events → density_overload signal."""
    events = [
        {"title": f"Event {i}", "start": f"{8+i:02d}:00", "end": f"{8+i:02d}:30"}
        for i in range(6)
    ]
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: events,
    )
    signals = mood.score_calendar_density(persist=False)
    density = [s for s in signals if s["signal_kind"] == "density_overload"]
    assert len(density) == 1
    assert density[0]["value"] < 0


def test_calendar_back_to_back_emits_signal(temp_db, monkeypatch, mood_enabled):
    """6 events back-to-back (sin gaps) → density_overload + back_to_back."""
    events = [
        {"title": f"E{i}", "start": f"{9+i:02d}:00", "end": f"{9+i:02d}:55"}
        for i in range(6)
    ]
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: events,
    )
    signals = mood.score_calendar_density(persist=False)
    btb = [s for s in signals if s["signal_kind"] == "back_to_back_meetings"]
    assert len(btb) == 1
    assert btb[0]["evidence"]["n_back_to_back"] >= 3


def test_calendar_no_back_to_back_when_gaps(temp_db, monkeypatch, mood_enabled):
    """6 events con gaps amplios: density_overload sí, back_to_back no."""
    # 9-10, 11-12, 13-14, 15-16, 17-18, 19-20 (1h gap entre cada uno)
    events = [
        {"title": f"E{i}", "start": f"{9+2*i:02d}:00", "end": f"{10+2*i:02d}:00"}
        for i in range(6)
    ]
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: events,
    )
    signals = mood.score_calendar_density(persist=False)
    kinds = [s["signal_kind"] for s in signals]
    assert "density_overload" in kinds
    assert "back_to_back_meetings" not in kinds


def test_parse_event_time_handles_formats():
    assert mood._parse_event_time("09:30") == (9, 30)
    assert mood._parse_event_time("9:30") == (9, 30)
    assert mood._parse_event_time("9:30 AM") == (9, 30)
    assert mood._parse_event_time("12:00 PM") == (12, 0)
    assert mood._parse_event_time("12:00 AM") == (0, 0)
    assert mood._parse_event_time("3:00 PM") == (15, 0)
    assert mood._parse_event_time("garbage") is None
    assert mood._parse_event_time("25:00") is None


# ─── Aggregator + drift ────────────────────────────────────────────────────


_SCORE_DAILY_DDL = """
CREATE TABLE rag_mood_score_daily (
    date TEXT PRIMARY KEY,
    score REAL NOT NULL,
    n_signals INTEGER NOT NULL,
    sources_used TEXT,
    top_evidence TEXT,
    updated_at REAL NOT NULL
)
"""


@pytest.fixture
def temp_full_db(tmp_path, monkeypatch):
    """Temp DB con todas las tablas que usan el agregador + scorers."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_SPOTIFY_DDL)
    conn.execute(_MOOD_SIGNALS_DDL)
    conn.execute(_QUERIES_DDL)
    conn.execute(_SCORE_DAILY_DDL)
    conn.commit()
    conn.close()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    yield db


def _insert_signal(
    db: Path, *, date: str, source: str, kind: str,
    value: float, weight: float = 1.0, evidence: dict | None = None,
    ts: float | None = None,
) -> None:
    ts = ts if ts is not None else time.time()
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO rag_mood_signals(ts, date, source, signal_kind, "
        "value, weight, evidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, date, source, kind, value, weight,
         json.dumps(evidence or {})),
    )
    conn.commit()
    conn.close()


def test_compute_daily_score_no_signals_writes_empty_row(temp_full_db, mood_enabled):
    """Sin señales: UPSERT escribe row con n_signals=0 score=0."""
    today = mood._today_local()
    res = mood.compute_daily_score(today)
    assert res["score"] == 0.0
    assert res["n_signals"] == 0
    row = mood.get_score_for_date(today)
    assert row is not None
    assert row["score"] == 0.0
    assert row["n_signals"] == 0


def test_compute_daily_score_weighted_avg(temp_full_db, mood_enabled):
    """3 señales con distintos weights: score = sum(v*w) / sum(w)."""
    today = mood._today_local()
    _insert_signal(temp_full_db, date=today, source="spotify",
                   kind="artist_mood_lookup", value=-0.6, weight=1.0)
    _insert_signal(temp_full_db, date=today, source="journal",
                   kind="keyword_negative", value=-0.5, weight=1.0)
    _insert_signal(temp_full_db, date=today, source="calendar",
                   kind="density_overload", value=-0.3, weight=0.4)
    res = mood.compute_daily_score(today)
    # (-0.6*1 + -0.5*1 + -0.3*0.4) / 2.4 = -1.22/2.4 ≈ -0.508
    assert res["score"] == pytest.approx(-0.508, abs=0.01)
    assert res["n_signals"] == 3
    assert "spotify" in res["sources_used"]
    assert "journal" in res["sources_used"]
    assert "calendar" in res["sources_used"]


def test_compute_daily_score_top_evidence_ranked_by_contribution(temp_full_db, mood_enabled):
    """Top evidence devuelve las 3 señales por |value*weight| descendente."""
    today = mood._today_local()
    _insert_signal(temp_full_db, date=today, source="spotify",
                   kind="artist_mood_lookup", value=-0.4, weight=1.0)  # |contrib|=0.4
    _insert_signal(temp_full_db, date=today, source="journal",
                   kind="keyword_negative", value=-0.9, weight=1.0)    # |contrib|=0.9
    _insert_signal(temp_full_db, date=today, source="calendar",
                   kind="density_overload", value=-0.3, weight=0.4)    # |contrib|=0.12
    _insert_signal(temp_full_db, date=today, source="queries",
                   kind="existential_pattern", value=-0.7, weight=0.4) # |contrib|=0.28
    res = mood.compute_daily_score(today)
    top = res["top_evidence"]
    assert len(top) == 3
    # Top 1 = journal (0.9), top 2 = spotify (0.4), top 3 = queries (0.28).
    assert top[0]["source"] == "journal"
    assert top[1]["source"] == "spotify"
    assert top[2]["source"] == "queries"


def test_compute_daily_score_idempotent_upsert(temp_full_db, mood_enabled):
    """Llamar compute_daily_score 2 veces para la misma fecha no
    duplica filas en rag_mood_score_daily (UPSERT por date PK)."""
    today = mood._today_local()
    _insert_signal(temp_full_db, date=today, source="spotify",
                   kind="artist_mood_lookup", value=-0.6, weight=1.0)
    mood.compute_daily_score(today)
    # Insertar otra señal y recompute → score actualizado, 1 sola row.
    _insert_signal(temp_full_db, date=today, source="journal",
                   kind="keyword_negative", value=-0.4, weight=1.0)
    mood.compute_daily_score(today)
    conn = sqlite3.connect(str(temp_full_db))
    n = conn.execute("SELECT COUNT(*) FROM rag_mood_score_daily WHERE date=?",
                     (today,)).fetchone()[0]
    conn.close()
    assert n == 1
    row = mood.get_score_for_date(today)
    assert row["n_signals"] == 2


def test_compute_daily_score_flag_off_returns_zero(temp_full_db, monkeypatch):
    """Con RAG_MOOD_ENABLED off, no escribe en DB y devuelve score 0."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    today = mood._today_local()
    res = mood.compute_daily_score(today)
    assert res["score"] == 0.0
    assert res["n_signals"] == 0
    # Nada escrito.
    assert mood.get_score_for_date(today) is None


def _insert_score_row(
    db: Path, *, date: str, score: float, n_signals: int,
    sources: list | None = None, top_evidence: list | None = None,
) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO rag_mood_score_daily(date, score, n_signals, "
        "sources_used, top_evidence, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (date, score, n_signals,
         json.dumps(sources or []),
         json.dumps(top_evidence or []),
         time.time()),
    )
    conn.commit()
    conn.close()


def test_recent_drift_no_data(temp_full_db, mood_enabled):
    res = mood.recent_drift()
    assert res["drifting"] is False
    assert res["reason"] == "no_data"


def test_recent_drift_3_consecutive_under_threshold(temp_full_db, mood_enabled):
    """3 días consecutivos terminando hoy con score ≤ -0.4 → drift."""
    today_ts = time.time()
    for offset in (2, 1, 0):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=-0.5, n_signals=3)
    res = mood.recent_drift(min_consecutive=3)
    assert res["drifting"] is True
    assert res["n_consecutive"] == 3
    assert res["avg_score"] == pytest.approx(-0.5)


def test_recent_drift_streak_broken_in_middle(temp_full_db, mood_enabled):
    """Si hay 2 días bajo, 1 día por encima, 1 día bajo → no drift
    (racha actual = 1)."""
    today_ts = time.time()
    scores = [-0.5, -0.5, +0.1, -0.5]  # offset 3, 2, 1, 0 (today)
    for i, sc in enumerate(scores):
        offset = len(scores) - 1 - i
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=sc, n_signals=2)
    res = mood.recent_drift(min_consecutive=3)
    assert res["drifting"] is False
    # La racha actual termina en today con 1 día — n_consecutive=1
    # (la racha "más larga" son 2 al inicio, pero esa terminó hace mucho).
    assert res["n_consecutive"] in (1, 2)


def test_recent_drift_stale_streak_does_not_trigger(temp_full_db, mood_enabled):
    """Una racha larga que terminó hace 3 días NO es drift (no avisamos
    bajones viejos)."""
    today_ts = time.time()
    # offset 5,4,3 = bajo. offset 2,1,0 = alto.
    for offset in (5, 4, 3):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=-0.6, n_signals=3)
    for offset in (2, 1, 0):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=+0.2, n_signals=3)
    res = mood.recent_drift(days=10, min_consecutive=3)
    assert res["drifting"] is False


def test_recent_drift_ignores_zero_signal_days(temp_full_db, mood_enabled):
    """Días con n_signals=0 NO cuentan para la racha (aunque el score
    grabado sea 0 por default — no hay evidence real)."""
    today_ts = time.time()
    for offset in (3, 2, 1, 0):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        # Todos los 4 días tienen score=-0.5 PERO n_signals=0 → ignorar.
        _insert_score_row(temp_full_db, date=date, score=-0.5, n_signals=0)
    res = mood.recent_drift(min_consecutive=3)
    assert res["drifting"] is False


def test_get_recent_scores_returns_descending(temp_full_db, mood_enabled):
    today_ts = time.time()
    for offset in (5, 3, 1, 0):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=-0.1 * offset, n_signals=2)
    rows = mood.get_recent_scores(days=14)
    assert len(rows) == 4
    # Ordenado date DESC: el primero es el de hoy (offset=0).
    assert rows[0]["date"] >= rows[-1]["date"]


def test_score_glyph_mapping():
    """Sanity check del helper para sparkline."""
    assert rag._mood_score_bar(-1.0) == "▁"
    assert rag._mood_score_bar(1.0) == "█"
    assert rag._mood_score_bar(0.0) == "─"
    assert rag._mood_score_bar(-0.5) in "▁▂▃"
    assert rag._mood_score_bar(0.5) in "▆▇█"


# ─── CLI smoke ─────────────────────────────────────────────────────────────


def test_cli_mood_show_with_data(temp_full_db, mood_enabled):
    today_ts = time.time()
    for offset in (2, 1, 0):
        date = time.strftime("%Y-%m-%d", time.localtime(today_ts - offset * 86400))
        _insert_score_row(temp_full_db, date=date, score=-0.5, n_signals=3,
                          sources=["spotify"])

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "show", "--days", "5", "--plain"])
    assert result.exit_code == 0, result.output
    # Contiene fechas + score + drift line.
    assert "drift" in result.output
    today_str = time.strftime("%Y-%m-%d", time.localtime(today_ts))
    assert today_str in result.output


def test_cli_mood_show_flag_off(temp_full_db, monkeypatch):
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "show", "--plain"])
    assert result.exit_code == 0
    assert "feature off" in result.output


def test_cli_mood_compute(temp_full_db, mood_enabled):
    today = mood._today_local()
    _insert_signal(temp_full_db, date=today, source="spotify",
                   kind="artist_mood_lookup", value=-0.5, weight=1.0)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "compute", "--plain"])
    assert result.exit_code == 0, result.output
    assert "score=" in result.output
    assert "spotify" in result.output


def test_cli_mood_explain(temp_full_db, mood_enabled):
    today = mood._today_local()
    _insert_signal(temp_full_db, date=today, source="journal",
                   kind="keyword_negative", value=-0.6, weight=1.0,
                   evidence={"keywords": ["bajón"], "n_matches": 1})
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "explain", "--plain"])
    assert result.exit_code == 0, result.output
    assert "journal" in result.output
    assert "keyword_negative" in result.output


# ─── Daemon state file + run_poll_cycle ────────────────────────────────────


@pytest.fixture
def daemon_state_dir(tmp_path, monkeypatch):
    """Patch _daemon_state_file → tmp_path/mood_enabled. Default: file
    NO existe (daemon disabled)."""
    state = tmp_path / "mood_enabled"
    monkeypatch.setattr(mood, "_daemon_state_file", lambda: state)
    return state


def test_daemon_disabled_by_default(daemon_state_dir):
    assert mood.is_daemon_enabled() is False


def test_daemon_enable_creates_state_file(daemon_state_dir):
    mood.enable_daemon()
    assert daemon_state_dir.exists()
    assert mood.is_daemon_enabled() is True


def test_daemon_enable_idempotent(daemon_state_dir):
    mood.enable_daemon()
    mood.enable_daemon()
    assert daemon_state_dir.exists()
    assert mood.is_daemon_enabled() is True


def test_daemon_disable_removes_state_file(daemon_state_dir):
    mood.enable_daemon()
    assert mood.is_daemon_enabled() is True
    mood.disable_daemon()
    assert daemon_state_dir.exists() is False
    assert mood.is_daemon_enabled() is False


def test_daemon_disable_idempotent(daemon_state_dir):
    """Llamar disable cuando ya estaba off no tira."""
    mood.disable_daemon()
    mood.disable_daemon()
    assert mood.is_daemon_enabled() is False


def test_run_poll_cycle_feature_off_returns_early(temp_full_db, daemon_state_dir, monkeypatch):
    """Sin RAG_MOOD_ENABLED, run_poll_cycle no llama scorers."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    result = mood.run_poll_cycle(persist=False, use_llm=False)
    assert result["reason"] == "feature_off"
    assert result["n_signals_emitted"] == 0
    assert result["scorers"] == {}


def test_run_poll_cycle_daemon_disabled_returns_early(
    temp_full_db, daemon_state_dir, mood_enabled,
):
    """Con env var ON pero state file ausente, scorers no corren."""
    assert not daemon_state_dir.exists()
    result = mood.run_poll_cycle(persist=False, use_llm=False)
    assert result["reason"] == "daemon_disabled"
    assert result["n_signals_emitted"] == 0


def test_run_poll_cycle_runs_all_scorers_when_enabled(
    temp_full_db, daemon_state_dir, mood_enabled, monkeypatch,
):
    """Con feature on + daemon enabled, los 5 scorers se llaman.
    Sin data real, todos devuelven []. El cycle igual recompute el
    aggregate (UPSERT empty row con n=0)."""
    mood.enable_daemon()

    # Mockear scorers que dependen de WA bridge / icalBuddy / Spotify
    # AppleScript, devuelven [] sin tocar IO real.
    monkeypatch.setattr(mood, "_wa_bridge_db_path", lambda: None)
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: [],
    )

    result = mood.run_poll_cycle(persist=True, use_llm=False)
    assert result.get("reason") is None
    assert result["enabled"] is True
    assert result["daemon_enabled"] is True
    assert set(result["scorers"].keys()) == {
        "spotify", "journal", "wa_outbound", "queries", "calendar",
    }
    # Cada scorer devuelve int (0 si nada) o "error" (si tira).
    for n in result["scorers"].values():
        assert isinstance(n, int) or n == "error"
    # El aggregate corrió (aunque sea con 0 signals).
    assert isinstance(result["score"], dict)
    assert result["score"]["n_signals"] == 0


def test_run_poll_cycle_isolates_failing_scorer(
    temp_full_db, daemon_state_dir, mood_enabled, monkeypatch,
):
    """Si un scorer tira excepción, los demás siguen y el aggregate
    corre. El scorer fallante queda como 'error' en el summary."""
    mood.enable_daemon()

    def _broken(*args, **kwargs):
        raise RuntimeError("simulated calendar failure")

    monkeypatch.setattr(mood, "score_calendar_density", _broken)
    monkeypatch.setattr(mood, "_wa_bridge_db_path", lambda: None)

    result = mood.run_poll_cycle(persist=False, use_llm=False)
    assert result["scorers"]["calendar"] == "error"
    # Los demás scorers se llamaron sin tirar.
    assert isinstance(result["scorers"]["spotify"], int)
    assert isinstance(result["scorers"]["queries"], int)


# ─── CLI enable/disable/status/poll ────────────────────────────────────────


def test_cli_mood_status_off_by_default(daemon_state_dir, monkeypatch):
    """Sin env var ni state file: status muestra ambos off."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "status", "--plain"])
    assert result.exit_code == 0, result.output
    assert "env_flag=off" in result.output
    assert "daemon=disabled" in result.output


def test_cli_mood_enable_then_status(daemon_state_dir, mood_enabled):
    from click.testing import CliRunner
    runner = CliRunner()
    runner.invoke(rag.cli, ["mood", "enable", "--plain"])
    result = runner.invoke(rag.cli, ["mood", "status", "--plain"])
    assert "daemon=enabled" in result.output
    assert daemon_state_dir.exists()


def test_cli_mood_enable_idempotent(daemon_state_dir, mood_enabled):
    from click.testing import CliRunner
    runner = CliRunner()
    r1 = runner.invoke(rag.cli, ["mood", "enable", "--plain"])
    assert "was_already_enabled=False" in r1.output
    r2 = runner.invoke(rag.cli, ["mood", "enable", "--plain"])
    assert "was_already_enabled=True" in r2.output
    assert daemon_state_dir.exists()


def test_cli_mood_disable_after_enable(daemon_state_dir, mood_enabled):
    from click.testing import CliRunner
    runner = CliRunner()
    runner.invoke(rag.cli, ["mood", "enable", "--plain"])
    result = runner.invoke(rag.cli, ["mood", "disable", "--plain"])
    assert result.exit_code == 0
    assert "was_enabled=True" in result.output
    assert not daemon_state_dir.exists()


def test_cli_mood_poll_skipped_when_disabled(temp_full_db, daemon_state_dir, mood_enabled):
    """`rag mood poll` con daemon disabled: sale como 'skipped'."""
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "poll", "--plain", "--dry-run"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload["reason"] == "daemon_disabled"


def test_cli_mood_poll_runs_when_enabled(
    temp_full_db, daemon_state_dir, mood_enabled, monkeypatch,
):
    """Con daemon enabled, poll corre los scorers."""
    mood.enable_daemon()
    monkeypatch.setattr(mood, "_wa_bridge_db_path", lambda: None)
    monkeypatch.setattr(
        "rag.integrations.calendar._fetch_calendar_today",
        lambda max_events=15: [],
    )
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "poll", "--plain", "--dry-run", "--no-llm"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload.get("reason") is None
    assert "scorers" in payload
    assert set(payload["scorers"].keys()) == {
        "spotify", "journal", "wa_outbound", "queries", "calendar",
    }


# ─── _services_spec includes mood-poll plist ──────────────────────────────


def test_services_spec_includes_mood_poll():
    """El plist mood-poll está registrado en _services_spec — `rag setup`
    lo va a generar + cargar."""
    spec = rag._services_spec("/fake/path/to/rag")
    labels = {label for label, _, _ in spec}
    assert "com.fer.obsidian-rag-mood-poll" in labels


def test_mood_poll_plist_has_required_fields():
    """El XML del plist contiene Label, ProgramArguments, RAG_MOOD_ENABLED=1,
    StartInterval=1800."""
    xml = rag._mood_poll_plist("/fake/rag")
    assert "<string>com.fer.obsidian-rag-mood-poll</string>" in xml
    assert "<key>RAG_MOOD_ENABLED</key><string>1</string>" in xml
    assert "<key>StartInterval</key><integer>1800</integer>" in xml
    assert "mood_poll.py" in xml
    assert "<key>RunAtLoad</key><true/>" in xml
