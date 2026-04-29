"""Tests for scripts/ingest_whatsapp.py + `rag index --source whatsapp`.

No live bridge DB / ollama — every test builds a minimal sqlite fixture
and monkeypatches `rag.embed`. Covers:
  - Timestamp parsing (RFC3339 w/ nanoseconds, Z suffix, numeric)
  - read_messages filtering (since_ts, exclude_jids, empty content)
  - Conversational chunking (speaker change, time gap, max_chars, merge)
  - Parent-window expansion
  - upsert_chunks writes source=whatsapp + all expected meta fields
  - Cursor advances incrementally across runs
  - CLI `rag index --source whatsapp` routes + surfaces summary
  - Retention floor (180d) filters old messages
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from scripts import ingest_whatsapp as iw


# ── Fixtures ─────────────────────────────────────────────────────────────

def _mk_bridge_db(path: Path, messages: list[tuple]) -> None:
    """Build a minimal whatsapp-mcp-style bridge DB with given (id, chat_jid,
    chat_name, sender, content, timestamp, is_from_me, media_type) rows."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT, last_message_time TEXT)"
    )
    conn.execute(
        "CREATE TABLE messages ("
        " id TEXT, chat_jid TEXT, sender TEXT, content TEXT, timestamp TEXT,"
        " is_from_me INTEGER, media_type TEXT, filename TEXT, url TEXT,"
        " media_key BLOB, file_sha256 BLOB, file_enc_sha256 BLOB,"
        " file_length INTEGER, PRIMARY KEY (id, chat_jid),"
        " FOREIGN KEY (chat_jid) REFERENCES chats(jid))"
    )
    seen_chats: set[str] = set()
    for (mid, jid, cname, sender, content, ts, from_me, media) in messages:
        if jid not in seen_chats:
            conn.execute(
                "INSERT INTO chats(jid, name, last_message_time) VALUES (?, ?, ?)",
                (jid, cname, ts),
            )
            seen_chats.add(jid)
        conn.execute(
            "INSERT INTO messages(id, chat_jid, sender, content, timestamp, "
            " is_from_me, media_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, jid, sender, content, ts, int(from_me), media),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def fake_bridge(tmp_path):
    db = tmp_path / "messages.db"
    now = datetime.now()

    def iso(offset_s: int) -> str:
        return (now + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S")

    # Chat "ana": alternating speakers, short msgs
    # Chat "grupo": same sender continuous, should merge
    # Chat "status@broadcast": excluded JID
    rows = [
        # (id, jid, chat_name, sender, content, ts, from_me, media)
        ("m1", "ana@s.whatsapp.net", "Ana", "ana", "hola cómo estás", iso(0), 0, None),
        ("m2", "ana@s.whatsapp.net", "Ana", "yo",  "todo bien vos?", iso(10), 1, None),
        ("m3", "ana@s.whatsapp.net", "Ana", "ana", "bien también",    iso(20), 0, None),

        ("g1", "grupo@g.us", "Grupo", "juan", "arranca el proyecto",  iso(100), 0, None),
        ("g2", "grupo@g.us", "Grupo", "juan", "lunes 10am reunión",   iso(110), 0, None),
        ("g3", "grupo@g.us", "Grupo", "juan", "todos confirman",       iso(120), 0, None),

        ("s1", "status@broadcast", "", "somebody", "status msg", iso(50), 0, None),

        # Gap ≥5min → new group
        ("m4", "ana@s.whatsapp.net", "Ana", "ana", "che mañana charlamos", iso(10000), 0, None),
    ]
    _mk_bridge_db(db, rows)
    return db


@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    """Minimal sqlite-vec collection for upsert tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="wa_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    # Deterministic embed — vector dim = 8, keyword-lookup style.
    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    return col


# ── Timestamp parsing ────────────────────────────────────────────────────

def test_parse_bridge_ts_rfc3339_z():
    ts = iw._parse_bridge_ts("2026-04-20T23:11:05Z")
    assert ts is not None
    # Sanity: should be a recent epoch
    assert ts > time.time() - 365 * 86400


def test_parse_bridge_ts_with_nanoseconds():
    # Go writes nanosecond precision; Python only goes to microsec.
    ts = iw._parse_bridge_ts("2026-04-20T23:11:05.123456789-03:00")
    assert ts is not None


def test_parse_bridge_ts_numeric():
    assert iw._parse_bridge_ts(1_700_000_000.5) == 1_700_000_000.5


def test_parse_bridge_ts_empty_returns_none():
    assert iw._parse_bridge_ts(None) is None
    assert iw._parse_bridge_ts("") is None


def test_parse_bridge_ts_unparseable_returns_none():
    assert iw._parse_bridge_ts("not-a-date") is None


# ── Reader ─────────────────────────────────────────────────────────────

def test_read_messages_excludes_status_broadcast(fake_bridge):
    msgs = iw.read_messages(fake_bridge, since_ts=0)
    assert msgs, "should return some messages"
    assert all(m.chat_jid != "status@broadcast" for m in msgs)


def test_read_messages_excludes_ragnet_bot_group(tmp_path):
    """RagNet (`WHATSAPP_BOT_JID`) es la UI del bot — recibe briefs,
    archive pushes, slash commands del user (`/help`, `/note`), y las
    respuestas del bot a esos comandos. Nada de esto es contenido
    conversacional del corpus. El indexer DEBE skipearlo aunque el user
    haya optado por indexar WhatsApp, sino se forma un feedback loop:
    bot pushea brief → indexer chunkea el brief → next retrieve devuelve
    el brief de ayer → el siguiente brief incluye self-references.

    Cerrado 2026-04-28 junto con `RAG_DRAFT_VIA_RAGNET` (que amplificaba
    el leak — testing 1-2 días de ambient sends redirigidos podía
    agregar cientos de chunks de output del bot al corpus).
    """
    db = tmp_path / "messages.db"
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows = [
        # Brief que el bot le mandó a RagNet — tiene el U+200B prefix
        # porque sale via _ambient_whatsapp_send con anti_loop=True.
        ("b1", rag.WHATSAPP_BOT_JID, "RagNet", "bot",
         "\u200B📓 Morning brief\n\nHoy tenés 3 reuniones...",
         now_iso, 1, None),
        # User le tipea `/help` a RagNet — sin U+200B, pero el filter
        # por JID igual lo skipea (es noise de UI, no corpus content).
        ("u1", rag.WHATSAPP_BOT_JID, "RagNet", "fer",
         "/help", now_iso, 0, None),
        # Respuesta del bot al /help — con U+200B (anti-loop) Y en RagNet,
        # doble guardia.
        ("b2", rag.WHATSAPP_BOT_JID, "RagNet", "bot",
         "\u200BComandos disponibles: /note, /cap, ...",
         now_iso, 1, None),
        # Mensaje legítimo de un contacto real — debe pasar.
        ("ok1", "grecia@s.whatsapp.net", "Grecia", "grecia",
         "hola che cómo estás", now_iso, 0, None),
    ]
    _mk_bridge_db(db, rows)
    msgs = iw.read_messages(db, since_ts=0)

    # RagNet entera (4 mensajes en la fixture: 2 bot + 1 user + 1 bot)
    # → 0 mensajes en out.
    assert all(m.chat_jid != rag.WHATSAPP_BOT_JID for m in msgs), (
        f"esperado: 0 chunks de RagNet, got: {[m.chat_jid for m in msgs]}"
    )
    # El mensaje de Grecia sí pasa.
    assert any(m.chat_jid == "grecia@s.whatsapp.net" for m in msgs)
    assert any("Grecia" in m.chat_name for m in msgs)


def test_read_messages_drops_u200b_anti_loop_in_other_chats(tmp_path):
    """Defense in depth: si por alguna razón un mensaje con prefix U+200B
    aparece en un chat que NO es RagNet (ej. el bot manda algo a un
    self-chat futuro o el bridge re-escribe destinos), el content-level
    filter lo descarta. Hoy el JID guard cubre todos los casos prod, pero
    este filter mantiene el invariante 'U+200B = bot output, never index'
    sin depender de actualizar la JID list cada vez que se agrega un
    nuevo chat de bot."""
    db = tmp_path / "messages.db"
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows = [
        # Mensaje del bot con U+200B prefix en un chat 1:1 inventado.
        ("b1", "fer-self@s.whatsapp.net", "Fer Self", "bot",
         "\u200BTest output del bot", now_iso, 1, None),
        # Mismo chat, pero un mensaje real del user (sin U+200B).
        ("u1", "fer-self@s.whatsapp.net", "Fer Self", "fer",
         "una nota mía importante", now_iso, 1, None),
        # Edge case: U+200B en mitad del texto, NO al inicio. Debe pasar
        # — solo el prefix nos dice "bot output", no la presencia.
        ("u2", "ana@s.whatsapp.net", "Ana", "ana",
         "hola\u200Bche", now_iso, 0, None),
    ]
    _mk_bridge_db(db, rows)
    msgs = iw.read_messages(db, since_ts=0)
    contents = [m.content for m in msgs]

    assert "Test output del bot" not in [c.lstrip("\u200B") for c in contents]
    assert "una nota mía importante" in contents
    assert any("\u200B" in c for c in contents), (
        "U+200B en mitad del texto NO debe filtrar el mensaje"
    )


def test_hardcoded_exclude_jids_includes_bot_group():
    """Sanity: la frozenset hardcoded contiene RagNet. Si alguien la
    edita y saca el JID del bot por error, este test rompe."""
    assert "status@broadcast" in iw.HARDCODED_EXCLUDE_JIDS
    assert rag.WHATSAPP_BOT_JID in iw.HARDCODED_EXCLUDE_JIDS


def test_read_messages_respects_since_ts(fake_bridge):
    # All messages have ts >= now → high since_ts drops them all
    future = time.time() + 10000
    assert iw.read_messages(fake_bridge, since_ts=future) == []


def test_read_messages_filters_empty_content(tmp_path):
    """SQL query already has `content IS NOT NULL AND content != ''` — this
    confirms we don't surface media-only rows."""
    db = tmp_path / "m.db"
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows = [
        ("m1", "x@s.whatsapp.net", "X", "x", "",            now_iso, 0, None),  # empty
        ("m2", "x@s.whatsapp.net", "X", "x", "hola",        now_iso, 0, None),
        ("m3", "x@s.whatsapp.net", "X", "x", None,          now_iso, 0, None),  # null (SQL filters)
    ]
    _mk_bridge_db(db, rows)
    msgs = iw.read_messages(db, since_ts=0)
    contents = [m.content for m in msgs]
    assert contents == ["hola"]


def test_read_messages_chat_jid_filter(fake_bridge):
    msgs = iw.read_messages(fake_bridge, since_ts=0, chat_jid="ana@s.whatsapp.net")
    assert all(m.chat_jid == "ana@s.whatsapp.net" for m in msgs)


# ── Chunker ────────────────────────────────────────────────────────────

def _mk_msg(mid: str, jid: str, sender: str, content: str, ts_offset_s: float, from_me: bool = False):
    return iw.WAMessage(
        id=mid, chat_jid=jid, chat_name="chat",
        sender=sender, content=content,
        timestamp=1_700_000_000.0 + ts_offset_s,
        is_from_me=from_me, media_type=None,
    )


def test_chunk_speaker_change_creates_new_group():
    msgs = [
        _mk_msg("1", "j", "ana", "hola", 0),
        _mk_msg("2", "j", "yo",  "bien vos?", 10, from_me=True),
        _mk_msg("3", "j", "ana", "bien", 20),
    ]
    groups = iw.chunk_conversation(msgs)
    # 3 distinct speaker runs → but short ones merge. Verify count ≤ 3.
    assert 1 <= len(groups) <= 3
    # Every message must appear exactly once across groups.
    seen = [m.id for g in groups for m in g]
    assert set(seen) == {"1", "2", "3"}


def test_chunk_gap_forces_new_group():
    """Messages 5+ minutes apart should NOT merge even from same sender."""
    msgs = [
        _mk_msg("1", "j", "ana", "msg1 largo " * 10, 0),
        _mk_msg("2", "j", "ana", "msg2 largo " * 10, 400),  # +6.6 min
    ]
    groups = iw.chunk_conversation(msgs)
    assert len(groups) == 2


def test_chunk_same_sender_merges_within_window():
    """Contiguous same-sender within 5min → single group."""
    msgs = [
        _mk_msg("1", "j", "juan", "primer mensaje medio largo aquí", 0),
        _mk_msg("2", "j", "juan", "segundo mensaje contiguo", 30),
        _mk_msg("3", "j", "juan", "tercer mensaje también", 60),
    ]
    groups = iw.chunk_conversation(msgs)
    assert len(groups) == 1
    assert [m.id for m in groups[0]] == ["1", "2", "3"]


def test_chunk_respects_max_chars():
    """Group splits when projected body exceeds CHUNK_MAX_CHARS."""
    # 10 messages × 100 chars = 1000 chars > 800 max
    msgs = [
        _mk_msg(f"m{i}", "j", "juan", "x" * 100, i * 10)
        for i in range(10)
    ]
    groups = iw.chunk_conversation(msgs)
    assert len(groups) >= 2
    for g in groups:
        total = sum(len(m.content) + 10 for m in g)
        assert total <= iw.CHUNK_MAX_CHARS + 50  # +10 overhead buffer


def test_chunk_intra_chat_only():
    """Messages from different chats never merge into one group."""
    msgs = [
        _mk_msg("1", "chatA", "x", "content A", 0),
        _mk_msg("2", "chatB", "x", "content B", 10),
    ]
    groups = iw.chunk_conversation(msgs)
    assert len(groups) == 2


def test_build_chunks_populates_parent_window():
    # 5 messages, chunk covers msg 2-3 — parent should include all 5.
    msgs = [
        _mk_msg("m1", "j", "a", "texto uno largo aquí", 0),
        _mk_msg("m2", "j", "b", "texto dos también largo", 10),
        _mk_msg("m3", "j", "b", "texto tres dentro del grupo", 20),
        _mk_msg("m4", "j", "a", "texto cuatro aquí", 30),
        _mk_msg("m5", "j", "a", "texto cinco final", 40),
    ]
    chunks = iw.build_chunks(msgs, parent_window=5)
    assert chunks
    # Parent should mention messages from both speakers.
    for c in chunks:
        assert c.parent
        # Should include multiple speakers in the window
        assert ("a:" in c.parent or "b:" in c.parent)


# ── Upsert writer ──────────────────────────────────────────────────────

def test_upsert_chunks_writes_source_whatsapp(tmp_vault_col):
    msgs = [
        _mk_msg("m1", "ana@jid", "ana", "hola cómo estás todo bien", 0),
        _mk_msg("m2", "ana@jid", "yo",  "genial vos", 10, from_me=True),
    ]
    # Force single-chunk by using tight window (both messages fit a group boundary).
    chunks = iw.build_chunks(msgs, same_speaker_window_s=0.5)
    assert chunks, "need at least one chunk for this test"

    n = iw.upsert_chunks(tmp_vault_col, chunks)
    assert n == len(chunks)

    got = tmp_vault_col.get(where={"source": "whatsapp"}, include=["metadatas"])
    assert got["ids"], "no WA rows indexed"
    for meta in got["metadatas"]:
        assert meta["source"] == "whatsapp"
        assert meta["chat_jid"] == "ana@jid"
        assert meta["file"].startswith("whatsapp://ana@jid/")
        assert meta["created_ts"] > 0
        assert "parent" in meta


def test_upsert_is_idempotent(tmp_vault_col):
    msgs = [_mk_msg("m1", "x@jid", "x", "un mensaje cualquiera acá", 0)]
    chunks = iw.build_chunks(msgs)
    iw.upsert_chunks(tmp_vault_col, chunks)
    before = tmp_vault_col.get(where={"source": "whatsapp"}, include=[])
    n_before = len(before["ids"])
    # Re-run with same chunks → count should be the same (delete + re-add path).
    iw.upsert_chunks(tmp_vault_col, chunks)
    after = tmp_vault_col.get(where={"source": "whatsapp"}, include=[])
    assert len(after["ids"]) == n_before


# ── Orchestration / cursor ─────────────────────────────────────────────

def test_run_advances_cursor_per_chat(fake_bridge, tmp_vault_col, monkeypatch):
    summary1 = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col)
    assert summary1["chunks_written"] > 0
    wrote_first = summary1["chunks_written"]

    # Second run with no new messages → cursor filters everything.
    summary2 = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col)
    assert summary2["messages_after_retention"] == 0
    assert summary2["chunks_written"] == 0


def test_run_reset_wipes_cursor(fake_bridge, tmp_vault_col):
    iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col)
    # With reset=True should re-scan and re-upsert.
    summary = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col, reset=True)
    assert summary["messages_after_retention"] > 0


def test_run_desync_guard_blocks_cursor_advance(
    fake_bridge, tmp_vault_col, monkeypatch,
):
    """Si upsert_chunks reporta N > 0 pero la DB no tiene los chunks,
    el cursor NO debe avanzar — así la próxima corrida reintenta.

    Regresión 2026-04-24: el ingester WA del user reportó
    `messages_processed=14394` durante varias corridas pero la collection
    tenía 0 chunks con source=whatsapp. El cursor igual avanzaba y el
    desync se volvía invisible. Fix: probe post-upsert que valida al
    menos 1 chunk en la DB antes de persistir el cursor.
    """
    # Monkeypatch upsert_chunks para retornar N > 0 pero NO escribir.
    # Usamos `monkeypatch.context()` (en vez de `monkeypatch.undo()`
    # global) porque la fixture `tmp_vault_col` también monkeypatchea
    # `rag.embed` con un fake determinístico (necesario en CI sin
    # Ollama). Un `monkeypatch.undo()` plano revertiría AMBOS — el
    # upsert_chunks local Y el rag.embed de la fixture — y la segunda
    # corrida abajo intentaría llamar Ollama real, fallando con
    # `ConnectionError: Failed to connect to Ollama`.
    # `monkeypatch.context()` aísla solo lo que se setteó adentro.
    with monkeypatch.context() as mp:
        def _fake_upsert(col, chunks):
            return len(chunks)  # lies — says it wrote but didn't
        mp.setattr(iw, "upsert_chunks", _fake_upsert)

        summary = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col)
        # Summary tiene el error explicativo.
        assert "error" in summary
        assert "desync" in summary["error"].lower()
        assert "cursor" in summary["error"].lower()
    # Fuera del context: `upsert_chunks` restaurado, `rag.embed` (de la
    # fixture) sigue siendo el fake. La 2da corrida DEBE reprocesar
    # desde 0 porque el cursor nunca avanzó.
    summary2 = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col)
    assert summary2["messages_after_retention"] > 0
    assert summary2["chunks_written"] > 0


def test_run_dry_run_writes_nothing(fake_bridge, tmp_vault_col):
    summary = iw.run(bridge_db=fake_bridge, vault_col=tmp_vault_col, dry_run=True)
    assert summary["chunks_built"] > 0
    assert summary["chunks_written"] == 0
    # No rows in the collection.
    got = tmp_vault_col.get(where={"source": "whatsapp"}, include=[])
    assert got["ids"] == []


def test_run_missing_bridge_reports_error(tmp_vault_col, tmp_path):
    summary = iw.run(
        bridge_db=tmp_path / "does-not-exist.db", vault_col=tmp_vault_col,
    )
    assert "error" in summary
    assert summary["chunks_written"] == 0


def test_retention_floor_drops_very_old_messages(tmp_path, tmp_vault_col):
    db = tmp_path / "m.db"
    very_old = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    fresh = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows = [
        ("old", "x@jid", "X", "x", "msg antiguo", very_old, 0, None),
        ("new", "x@jid", "X", "x", "msg fresco",  fresh,    0, None),
    ]
    _mk_bridge_db(db, rows)
    summary = iw.run(bridge_db=db, vault_col=tmp_vault_col)
    # Only the fresh one should survive the retention filter (180d for WA).
    assert summary["messages_after_retention"] == 1


# ── CLI: rag index --source whatsapp ──────────────────────────────────

def test_cli_index_source_whatsapp_routes(fake_bridge, tmp_vault_col, monkeypatch):
    """Verify the CLI wiring calls the WA ingester (not the vault path)."""
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "messages_read": 10, "messages_after_retention": 5,
            "chunks_built": 2, "chunks_written": 2, "chats_touched": 1,
            "duration_s": 0.1,
        }
    from scripts import ingest_whatsapp as iw_mod
    monkeypatch.setattr(iw_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "whatsapp"])
    assert result.exit_code == 0, result.output
    assert "whatsapp" in result.output
    # Minimal format: `whatsapp: {total} · +{indexed} · {time}s` — the
    # indexed count is `chunks_written` (=2 here) rendered as `+2`.
    assert "+2" in result.output
    assert called.get("reset") is False
    assert called.get("dry_run") is False


def test_cli_index_source_invalid_rejects():
    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "facebook"])
    assert "Fuente inválida" in result.output


def test_cli_index_source_whatsapp_dry_run(monkeypatch):
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "messages_read": 3, "messages_after_retention": 3,
            "chunks_built": 1, "chunks_written": 0, "chats_touched": 1,
            "duration_s": 0.0,
        }
    from scripts import ingest_whatsapp as iw_mod
    monkeypatch.setattr(iw_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "whatsapp", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["dry_run"] is True
    assert "dry · " in result.output


# ── _speaker_label: resolver wiring con phone index ──────────────────────

def _mk_speaker_msg(*, sender: str = "", chat_name: str = "", is_from_me: bool = False,
                    content: str = "hola",
                    chat_jid: str = "5491112345678@s.whatsapp.net") -> iw.WAMessage:
    """Minimal WAMessage for _speaker_label tests.

    Kept separate from the positional `_mk_msg` higher up in the file
    (which uses `(mid, jid, sender, content, ts, from_me=...)` shape); this
    keyword-only variant keeps the speaker-label tests self-documenting
    without breaking the chunker test suite.
    """
    return iw.WAMessage(
        id="msg1", chat_jid=chat_jid, chat_name=chat_name,
        sender=sender, content=content, timestamp=1700000000.0,
        is_from_me=is_from_me, media_type=None,
    )


def test_speaker_label_from_me_always_yo() -> None:
    """is_from_me dominates — sender/chat_name ignored."""
    m = _mk_speaker_msg(is_from_me=True, sender="whatever@s.whatsapp.net",
                        chat_name="Family")
    assert iw._speaker_label(m) == "yo"


def test_speaker_label_resolves_dossier_name(tmp_path, monkeypatch) -> None:
    """JID con phone registrado en 99-Mentions/X.md → nombre del dossier."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Maria.md").write_text(
        "- **Teléfono**: +54 9 342 430 3891\n", encoding="utf-8",
    )
    rag._phone_index_cache = None
    rag._mentions_cache = None
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    m = _mk_speaker_msg(sender="5493424303891@s.whatsapp.net", chat_name="Maria chat")
    assert iw._speaker_label(m) == "Maria"

    rag._phone_index_cache = None
    rag._mentions_cache = None


def test_speaker_label_masks_unmapped_phone(tmp_path, monkeypatch) -> None:
    """JID sin dossier match + sin Contacts match → last-4 masked ('…3891').
    Mockea Apple Contacts al vacío para que el test no dependa del address
    book del host (que sí tiene a Maria en el entorno del autor)."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)  # empty folder, no dossiers
    rag._phone_index_cache = None
    rag._mentions_cache = None
    rag._contacts_phone_index = None
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    # No dossier AND no Contacts match → mask fallback fires.
    monkeypatch.setattr(rag, "_load_contacts_phone_index", lambda ttl_s=86400: {})

    m = _mk_speaker_msg(sender="5493424303891@s.whatsapp.net", chat_name="desconocido")
    label = iw._speaker_label(m)
    assert label.startswith("…"), f"expected masked, got {label!r}"
    assert "3891" in label
    # Must NOT leak the full phone number
    assert "5493424303891" not in label

    rag._phone_index_cache = None
    rag._mentions_cache = None
    rag._contacts_phone_index = None


def test_speaker_label_empty_sender_falls_back_to_chat_name(tmp_path, monkeypatch) -> None:
    """sender='' + chat_name='Juan' → 'Juan' (1-on-1 chat sin JID sender)."""
    (tmp_path / "04-Archive/99-obsidian-system/99-Mentions").mkdir(parents=True)
    rag._phone_index_cache = None
    rag._mentions_cache = None
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    m = _mk_speaker_msg(sender="", chat_name="Juan")
    assert iw._speaker_label(m) == "Juan"

    rag._phone_index_cache = None
    rag._mentions_cache = None


def test_speaker_label_invariant_in_chunk_body(tmp_path, monkeypatch) -> None:
    """Chunks generados via chunk_conversation usan el speaker resolved.
    Full flow: dossier con phone → chunk body empieza con 'Maria:' en lugar
    del JID numérico."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Maria.md").write_text(
        "- **Teléfono**: +54 9 342 430 3891\n", encoding="utf-8",
    )
    rag._phone_index_cache = None
    rag._mentions_cache = None
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    msgs = [
        _mk_speaker_msg(sender="5493424303891@s.whatsapp.net", chat_name="Maria",
                content="hola"),
        _mk_speaker_msg(sender="5493424303891@s.whatsapp.net", chat_name="Maria",
                content="como estas"),
    ]
    # Same sender, back-to-back → one group.
    groups = iw.chunk_conversation(msgs, same_speaker_window_s=300.0,
                                    min_merge_chars=0, max_chars=1000)
    assert len(groups) == 1
    rendered = iw._render_window(groups[0])
    assert "Maria:" in rendered
    assert "5493424303891" not in rendered

    rag._phone_index_cache = None
    rag._mentions_cache = None


# ── Audit 2026-04-24 hardening: _load_cursor NULL guard + atomic saves ───


def test_load_cursor_handles_null_last_ts(tmp_path):
    """Pre-audit: si `rag_whatsapp_state.last_ts` era NULL (corrupción
    manual o migración parcial), `float(row[0])` lanzaba TypeError que
    propagaba hasta arriba — re-procesaba TODOS los mensajes desde
    epoch en esta y todas las próximas corridas (no había recover).
    Post-fix: NULL se trata como missing, cursor vuelve a 0.0 sin
    crash."""
    state_db = tmp_path / "state.db"
    conn = sqlite3.connect(str(state_db))
    conn.execute(
        "CREATE TABLE rag_whatsapp_state ("
        " chat_jid TEXT PRIMARY KEY, last_ts REAL, last_msg_id TEXT,"
        " updated_at TEXT NOT NULL)"
    )
    # Row existe pero last_ts es NULL.
    conn.execute(
        "INSERT INTO rag_whatsapp_state (chat_jid, last_ts, last_msg_id, updated_at) "
        "VALUES (?, NULL, 'x', '2026-04-24')",
        ("broken-jid",),
    )
    conn.commit()

    # No debería lanzar TypeError; debería devolver 0.0 gracefully.
    result = iw._load_cursor(conn, "broken-jid")
    assert result == 0.0
    # Chat no existe → 0.0 también.
    assert iw._load_cursor(conn, "missing-jid") == 0.0
    conn.close()


def test_load_cursor_handles_corrupt_value_gracefully(tmp_path):
    """Si alguien mete un string no-numérico en last_ts, no crash."""
    state_db = tmp_path / "state.db"
    conn = sqlite3.connect(str(state_db))
    conn.execute(
        "CREATE TABLE rag_whatsapp_state ("
        " chat_jid TEXT PRIMARY KEY, last_ts TEXT, last_msg_id TEXT,"
        " updated_at TEXT NOT NULL)"
    )
    # Valor no parseable como float.
    conn.execute(
        "INSERT INTO rag_whatsapp_state (chat_jid, last_ts, last_msg_id, updated_at) "
        "VALUES (?, 'not-a-number', 'x', '2026-04-24')",
        ("corrupt-jid",),
    )
    conn.commit()

    # Debería fallback a 0.0 via el except ValueError.
    result = iw._load_cursor(conn, "corrupt-jid")
    assert result == 0.0
    conn.close()


def test_load_cursor_valid_row_returns_float(tmp_path):
    """No-regresión: happy path con last_ts legítimo devuelve float."""
    state_db = tmp_path / "state.db"
    conn = sqlite3.connect(str(state_db))
    conn.execute(
        "CREATE TABLE rag_whatsapp_state ("
        " chat_jid TEXT PRIMARY KEY, last_ts REAL, last_msg_id TEXT,"
        " updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO rag_whatsapp_state (chat_jid, last_ts, last_msg_id, updated_at) "
        "VALUES (?, 1713400000.5, 'msg-abc', '2026-04-24')",
        ("ana@s.whatsapp.net",),
    )
    conn.commit()

    result = iw._load_cursor(conn, "ana@s.whatsapp.net")
    assert result == 1713400000.5
    assert isinstance(result, float)
    conn.close()
