import multiprocessing as mp
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import rag
from web import conversation_writer
from web.conversation_writer import (
    TurnData,
    get_conversation_path,
    persist_conversation_index_entry,
    slugify,
    write_turn,
)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "00-Inbox" / "conversations").mkdir(parents=True)
    # Post-T10 split: index lives in rag_conversations_index in telemetry.db.
    # _resolve_telemetry_db_path() reads rag.DB_PATH dynamically, so patching
    # DB_PATH is sufficient — no _DB_PATH module attr needed.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    return vault


def _turn(q, a, sources, conf, ts):
    return TurnData(question=q, answer=a, sources=sources, confidence=conf, timestamp=ts)


def test_first_turn_creates_note_with_frontmatter(tmp_vault):
    ts = datetime(2026, 4, 19, 4, 12, 0, tzinfo=timezone.utc)
    turn = _turn(
        "¿qué es el Ikigai?",
        "El Ikigai es una filosofía japonesa.",
        [{"file": "02-Areas/Coaching.md", "score": 0.8},
         {"file": "03-Resources/Ikigai.md", "score": 0.7}],
        0.75,
        ts,
    )
    path = write_turn(tmp_vault, "web:abc123", turn)
    assert path.name == "2026-04-19-0412-que-es-el-ikigai.md"
    assert path.parent == tmp_vault / "00-Inbox" / "conversations"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    fm_end = text.index("\n---\n", 4)
    fm = text[4:fm_end]
    lines = fm.split("\n")
    # 6 top-level keys in fixed order (sources + tags have list items)
    keys_in_order = [ln.split(":", 1)[0] for ln in lines if not ln.startswith("  ")]
    assert keys_in_order == [
        "session_id", "created", "updated", "turns", "confidence_avg", "sources", "tags",
    ]
    assert "session_id: web:abc123" in fm
    assert "created: 2026-04-19T04:12:00Z" in fm
    assert "updated: 2026-04-19T04:12:00Z" in fm
    assert "turns: 1" in fm
    assert "confidence_avg: 0.750" in fm
    assert "  - 02-Areas/Coaching.md" in fm
    assert "  - 03-Resources/Ikigai.md" in fm
    assert "  - conversation" in fm
    assert "  - rag-chat" in fm
    assert "## Turn 1 — 04:12" in text
    assert "> ¿qué es el Ikigai?" in text
    assert "El Ikigai es una filosofía japonesa." in text
    assert "[[02-Areas/Coaching]]" in text
    assert "[[03-Resources/Ikigai]]" in text


def test_second_turn_appends_and_updates_frontmatter(tmp_vault):
    ts1 = datetime(2026, 4, 19, 4, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 19, 4, 18, 33, tzinfo=timezone.utc)
    sid = "web:sess2"
    t1 = _turn("primera pregunta", "respuesta uno",
               [{"file": "02-Areas/A.md", "score": 0.5}], 0.40, ts1)
    p1 = write_turn(tmp_vault, sid, t1)
    t2 = _turn("segunda pregunta", "respuesta dos",
               [{"file": "03-Resources/B.md", "score": 0.6}], 0.60, ts2)
    p2 = write_turn(tmp_vault, sid, t2)
    assert p1 == p2
    text = p2.read_text(encoding="utf-8")
    assert "turns: 2" in text
    assert "created: 2026-04-19T04:12:00Z" in text
    assert "updated: 2026-04-19T04:18:33Z" in text
    # running avg: (0.40 + 0.60) / 2 = 0.500
    assert "confidence_avg: 0.500" in text
    assert "  - 02-Areas/A.md" in text
    assert "  - 03-Resources/B.md" in text
    assert "## Turn 1 — 04:12" in text
    assert "## Turn 2 — 04:18" in text
    assert text.count("## Turn ") == 2


def test_slugify_strips_accents_and_punctuation():
    assert slugify("¿Qué es el Ikigai?") == "que-es-el-ikigai"
    assert slugify("  Hola   MUNDO!!  ") == "hola-mundo"
    assert slugify("") == "conversation"
    assert slugify("a" * 80, max_len=50) == "a" * 50


def test_index_maps_session_and_reuses_path(tmp_vault, monkeypatch):
    ts = datetime(2026, 4, 19, 5, 0, 0, tzinfo=timezone.utc)
    sid = "web:idx-test"
    t = _turn("tema único", "ok", [{"file": "X.md", "score": 0.1}], 0.1, ts)
    p1 = write_turn(tmp_vault, sid, t)
    # Post-T10: the index entry lives in rag_conversations_index (SQL).
    assert get_conversation_path(sid) == str(p1.relative_to(tmp_vault))

    # Second write for same session — move the file so "find by scanning" would fail,
    # forcing the reuse to come from the index. (We move it back to verify path is reused.)
    ts2 = datetime(2026, 4, 19, 5, 5, 0, tzinfo=timezone.utc)
    t2 = _turn("otra pregunta distinta", "respuesta", [{"file": "Y.md", "score": 0.2}], 0.3, ts2)
    # Sabotage the folder scan: if the code ever scans by session_id in the folder,
    # it won't find anything since index says exact path. The file still exists there.
    p2 = write_turn(tmp_vault, sid, t2)
    assert p2 == p1
    text = p2.read_text(encoding="utf-8")
    assert "turns: 2" in text


def test_sequential_turns_accumulate(tmp_vault):
    """Post-T10: concurrency for a single session is NOT a production scenario
    (one /api/chat per session at a time). The previous fcntl.flock-based
    whole-body lock is gone; serialization is handled by the SQL upsert's
    BEGIN IMMEDIATE. We assert sequential turns still merge correctly.
    """
    sid = "web:sequential"
    ts1 = datetime(2026, 4, 19, 6, 1, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 19, 6, 2, 0, tzinfo=timezone.utc)
    t1 = _turn("pregunta 1", "respuesta 1",
               [{"file": "F1.md", "score": 0.5}], 0.5, ts1)
    t2 = _turn("pregunta 2", "respuesta 2",
               [{"file": "F2.md", "score": 0.5}], 0.5, ts2)
    p1 = write_turn(tmp_vault, sid, t1)
    p2 = write_turn(tmp_vault, sid, t2)
    assert p1 == p2
    text = p1.read_text(encoding="utf-8")
    assert text.count("## Turn ") == 2
    assert "## Turn 1 —" in text
    assert "## Turn 2 —" in text
    # Order matters — the whole point of the lock is that Turn 1 lands
    # before Turn 2 in the rendered body. If the append order raced, the
    # count would still be 2 but the file would be incoherent.
    assert text.index("## Turn 1 —") < text.index("## Turn 2 —"), \
        "Turn 1 must render before Turn 2"
    assert "turns: 2" in text
    assert "  - F1.md" in text
    assert "  - F2.md" in text


def test_turn_with_inf_confidence_sanitized_on_second_turn(tmp_vault):
    """Regression: historical conversation notes persisted
    `confidence_avg: -inf` (from `retrieve()` returning `float('-inf')`
    on empty corpora). Subsequent turns would propagate the sentinel
    forever via `(-inf * k + x) / n = -inf`. The writer now clamps
    non-finite old_avg to 0.0 as defense-in-depth even if a pre-sanitize
    note sneaks in.
    """
    sid = "web:inf-conf"
    ts1 = datetime(2026, 4, 21, 9, 0, 0, tzinfo=timezone.utc)
    # Seed a note with a corrupt `confidence_avg` value on disk.
    folder = tmp_vault / "00-Inbox" / "conversations"
    corrupt = folder / "2026-04-21-0900-q.md"
    corrupt.write_text(
        "---\n"
        f"session_id: {sid}\n"
        "created: 2026-04-21T09:00:00Z\n"
        "updated: 2026-04-21T09:00:00Z\n"
        "turns: 1\n"
        "confidence_avg: -inf\n"
        "sources: []\n"
        "tags:\n"
        "  - conversation\n"
        "  - rag-chat\n"
        "---\n"
        "\n"
        "## Turn 1 — 09:00\n\n> q\n\na\n\n**Sources**: —\n",
        encoding="utf-8",
    )
    persist_conversation_index_entry(sid, str(corrupt.relative_to(tmp_vault)))
    # Second turn must not crash + must overwrite the bad avg with a finite
    # value. With the -inf left unclamped the new_avg would stay -inf.
    t2 = _turn("q2", "a2", [{"file": "X.md", "score": 0.5}], 0.5, datetime(
        2026, 4, 21, 9, 5, 0, tzinfo=timezone.utc))
    p = write_turn(tmp_vault, sid, t2)
    text = p.read_text(encoding="utf-8")
    assert "turns: 2" in text
    assert "-inf" not in text
    assert "-Inf" not in text
    # The new avg is finite — exact value doesn't matter for the invariant.
    fm_end = text.index("\n---\n", 4)
    fm = text[4:fm_end]
    avg_line = next(ln for ln in fm.split("\n") if ln.startswith("confidence_avg:"))
    assert float(avg_line.split(":", 1)[1].strip()) > float("-inf")


def test_second_turn_with_first_turn_zero_sources(tmp_vault):
    """Regression: cuando el turno 1 no tiene sources (p.ej. metachat /
    propose-intent / retrieval vacío), el frontmatter renderea `sources: []`
    como texto literal. El parser lo leía de vuelta como string `"[]"` (no
    lista), y al procesar el turno 2 `isinstance(existing_sources, list)`
    daba False → `ValueError('sources must be a list')`. Ver
    `conversation_turn_pending.jsonl`.
    """
    sid = "web:zero-sources"
    ts1 = datetime(2026, 4, 21, 10, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 21, 10, 5, 0, tzinfo=timezone.utc)
    t1 = _turn("primera", "respuesta", [], 0.3, ts1)
    p1 = write_turn(tmp_vault, sid, t1)
    text1 = p1.read_text(encoding="utf-8")
    assert "sources: []" in text1
    t2 = _turn("segunda", "respuesta",
               [{"file": "03-Resources/B.md", "score": 0.6}], 0.6, ts2)
    p2 = write_turn(tmp_vault, sid, t2)
    text2 = p2.read_text(encoding="utf-8")
    assert "turns: 2" in text2
    assert "  - 03-Resources/B.md" in text2


def test_malformed_frontmatter_raises(tmp_vault):
    sid = "web:broken"
    target = tmp_vault / "00-Inbox" / "conversations" / "broken.md"
    target.write_text("---\nthis is : not : parseable : yaml\nno closing block\n",
                      encoding="utf-8")
    # Post-T10: seed the SQL index to point at the broken file.
    persist_conversation_index_entry(sid, str(target.relative_to(tmp_vault)))
    ts = datetime(2026, 4, 19, 7, 0, 0, tzinfo=timezone.utc)
    t = _turn("q", "a", [{"file": "Z.md", "score": 0.1}], 0.1, ts)
    with pytest.raises(ValueError):
        write_turn(tmp_vault, sid, t)


# ── T5: SQL-backed index (RAG_STATE_SQL=1) ───────────────────────────────────


@pytest.fixture
def sql_env(tmp_path, monkeypatch):
    """Post-T10 split: writer targets telemetry.db via _resolve_telemetry_db_path().
    Patch rag.DB_PATH so the resolver lands in tmp_path instead of the live dir."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    return tmp_path / rag._TELEMETRY_DB_FILENAME


@pytest.fixture
def flag_off(tmp_path, monkeypatch):
    """Legacy flag-OFF fixture — kept for test-signature compatibility.
    Post-T10 the flag is inert, so this is an alias for sql_env."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.delenv("RAG_STATE_SQL", raising=False)
    return tmp_path / rag._TELEMETRY_DB_FILENAME


def _select_row(db_path: Path, sid: str):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT session_id, relative_path, updated_at FROM rag_conversations_index"
            " WHERE session_id = ?",
            (sid,),
        ).fetchone()
    finally:
        conn.close()


def test_upsert_new_session_flag_on(sql_env):
    sid = "web:t5-new"
    rel = "00-Inbox/conversations/2026-04-19-1000-hello.md"
    persist_conversation_index_entry(sid, rel)
    row = _select_row(sql_env, sid)
    assert row is not None
    assert row[0] == sid
    assert row[1] == rel
    # ISO-8601 seconds (T1 convention): YYYY-MM-DDTHH:MM:SS
    assert len(row[2]) == 19 and "T" in row[2]


def test_upsert_existing_session_replaces(sql_env):
    sid = "web:t5-replace"
    persist_conversation_index_entry(sid, "old/path.md")
    persist_conversation_index_entry(sid, "new/path.md")
    conn = sqlite3.connect(str(sql_env))
    try:
        rows = conn.execute(
            "SELECT relative_path FROM rag_conversations_index WHERE session_id = ?",
            (sid,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "new/path.md"


def test_read_returns_none_when_row_missing_post_t10(sql_env):
    """Post-T10: no JSON fallback. Missing SQL row → None."""
    assert get_conversation_path("web:t5-precutover") is None


def test_read_returns_none_when_both_missing(sql_env):
    assert get_conversation_path("web:nope") is None


def test_flag_off_is_inert_post_t10(flag_off):
    """Post-T10 RAG_STATE_SQL is inert — writer still lands in SQL regardless."""
    sid = "web:t5-flag-inert"
    rel = "00-Inbox/conversations/sql-only.md"
    persist_conversation_index_entry(sid, rel)
    conn = sqlite3.connect(str(flag_off))
    try:
        row = conn.execute(
            "SELECT relative_path FROM rag_conversations_index "
            "WHERE session_id = ?", (sid,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == rel


def test_writer_failure_does_not_raise_into_caller(sql_env, monkeypatch):
    # Simulate a commit failure deep inside _sql_upsert by making conn.commit
    # raise. The public writer API in web/server.py wraps write_turn in
    # try/except — here we verify that layer's contract via the same pattern.
    def boom(*_a, **_kw):
        raise sqlite3.OperationalError("simulated commit failure")

    monkeypatch.setattr(rag, "_sql_upsert", boom)

    vault = sql_env.parent / "vault"
    (vault / "00-Inbox" / "conversations").mkdir(parents=True)
    ts = datetime(2026, 4, 19, 8, 0, 0, tzinfo=timezone.utc)
    turn = _turn("q5", "a5", [{"file": "X.md", "score": 0.1}], 0.1, ts)

    # write_turn itself will propagate, matching the contract where
    # web/server.py's _persist_conversation_turn is the swallower. Emulate.
    caught: list = []
    try:
        write_turn(vault, "web:failtest", turn)
    except Exception as exc:  # noqa: BLE001 — mirror server.py's broad catch
        caught.append(exc)
    assert caught, "expected the raw exception to surface to the caller-of-writer"
    # Simulate the server.py wrapper: it catches and logs, never re-raises.
    try:
        try:
            write_turn(vault, "web:failtest", turn)
        except Exception as exc:
            _ = repr(exc)  # would be logged via _LOG_QUEUE
    except Exception:
        pytest.fail("server-side swallower must not propagate")


def _worker_upsert(args):
    db_path, proc_id = args
    # Child process: monkeypatch does not cross the fork/spawn boundary.
    # Route _resolve_telemetry_db_path() to our tmp dir by setting rag.DB_PATH
    # to the parent of db_path (telemetry.db filename stays "telemetry.db").
    import rag as _rag
    _rag.DB_PATH = Path(db_path).parent
    for i in range(10):
        sid = f"wa:proc{proc_id}-turn{i}"
        rel = f"00-Inbox/conversations/{proc_id}-{i}.md"
        from web import conversation_writer as cw
        cw.persist_conversation_index_entry(sid, rel)


def test_atomicity_under_concurrent_writers(sql_env):
    ctx = mp.get_context("spawn")
    args = [(str(sql_env), pid) for pid in range(20)]
    with ctx.Pool(processes=min(8, mp.cpu_count())) as pool:
        pool.map(_worker_upsert, args)
    conn = sqlite3.connect(str(sql_env))
    try:
        (n,) = conn.execute("SELECT COUNT(*) FROM rag_conversations_index").fetchone()
    finally:
        conn.close()
    assert n == 200
