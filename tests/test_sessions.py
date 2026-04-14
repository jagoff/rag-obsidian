import json
import os
import time

import pytest

import rag


@pytest.fixture
def sessions_tmp(tmp_path, monkeypatch):
    sdir = tmp_path / "sessions"
    lastf = tmp_path / "last_session"
    monkeypatch.setattr(rag, "SESSIONS_DIR", sdir)
    monkeypatch.setattr(rag, "LAST_SESSION_FILE", lastf)
    return sdir, lastf


def test_new_session_id_matches_regex():
    for _ in range(10):
        sid = rag.new_session_id()
        assert rag._valid_session_id(sid)


def test_roundtrip(sessions_tmp):
    sdir, _ = sessions_tmp
    sess = rag.ensure_session(None, mode="chat")
    rag.append_turn(sess, {"q": "hola", "a": "hi"})
    rag.save_session(sess)

    assert (sdir / f"{sess['id']}.json").is_file()
    loaded = rag.load_session(sess["id"])
    assert loaded is not None
    assert loaded["id"] == sess["id"]
    assert loaded["mode"] == "chat"
    assert len(loaded["turns"]) == 1
    assert loaded["turns"][0]["q"] == "hola"
    assert loaded["turns"][0]["a"] == "hi"
    assert "ts" in loaded["turns"][0]


def test_ensure_session_keeps_valid_custom_id(sessions_tmp):
    sess = rag.ensure_session("tg:12345", mode="telegram")
    assert sess["id"] == "tg:12345"
    assert sess["mode"] == "telegram"
    assert sess["turns"] == []


def test_ensure_session_mints_fresh_id_for_invalid_input(sessions_tmp):
    bad = "foo bar/baz"  # contains space and slash — fails SESSION_ID_RE
    sess = rag.ensure_session(bad, mode="chat")
    assert sess["id"] != bad
    assert rag._valid_session_id(sess["id"])


def test_ensure_session_returns_existing_when_id_matches(sessions_tmp):
    s1 = rag.ensure_session("tg:777", mode="telegram")
    rag.append_turn(s1, {"q": "uno", "a": "a1"})
    rag.save_session(s1)

    s2 = rag.ensure_session("tg:777", mode="telegram")
    assert s2["id"] == "tg:777"
    assert len(s2["turns"]) == 1
    assert s2["turns"][0]["q"] == "uno"


def test_append_turn_caps_at_session_max_turns(sessions_tmp, monkeypatch):
    monkeypatch.setattr(rag, "SESSION_MAX_TURNS", 3)
    sess = rag.ensure_session(None, mode="chat")
    for i in range(5):
        rag.append_turn(sess, {"q": f"q{i}", "a": f"a{i}"})
    assert len(sess["turns"]) == 3
    # oldest turns dropped, newest kept in order
    assert [t["q"] for t in sess["turns"]] == ["q2", "q3", "q4"]


def test_session_history_interleaves_roles_and_windows(sessions_tmp):
    sess = rag.ensure_session(None, mode="chat")
    for i in range(5):
        rag.append_turn(sess, {"q": f"q{i}", "a": f"a{i}"})

    hist = rag.session_history(sess, window=4)
    assert len(hist) == 4
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    # last 4 of [q0, a0, q1, a1, q2, a2, q3, a3, q4, a4] == [q3, a3, q4, a4]
    assert [m["content"] for m in hist] == ["q3", "a3", "q4", "a4"]


def test_session_history_skips_missing_assistant(sessions_tmp):
    sess = rag.ensure_session(None, mode="chat")
    rag.append_turn(sess, {"q": "solo_user"})
    rag.append_turn(sess, {"q": "pregunta", "a": "respuesta"})

    hist = rag.session_history(sess)
    assert hist == [
        {"role": "user", "content": "solo_user"},
        {"role": "user", "content": "pregunta"},
        {"role": "assistant", "content": "respuesta"},
    ]


def test_session_history_defaults_to_module_window(sessions_tmp):
    sess = rag.ensure_session(None, mode="chat")
    for i in range(rag.SESSION_HISTORY_WINDOW + 5):
        rag.append_turn(sess, {"q": f"q{i}", "a": f"a{i}"})
    hist = rag.session_history(sess)
    assert len(hist) == rag.SESSION_HISTORY_WINDOW


def test_list_sessions_orders_by_updated_at_desc(sessions_tmp):
    a = rag.ensure_session("aaa", mode="chat")
    rag.append_turn(a, {"q": "primera"})
    rag.save_session(a)

    # save_session stamps updated_at at second resolution — wait enough for a
    # distinct ISO timestamp so the sort is deterministic.
    time.sleep(1.1)

    b = rag.ensure_session("bbb", mode="chat")
    rag.append_turn(b, {"q": "segunda"})
    rag.save_session(b)

    rows = rag.list_sessions()
    ids = [r["id"] for r in rows]
    assert ids[:2] == ["bbb", "aaa"]
    assert rows[0]["turns"] == 1
    assert rows[0]["first_q"] == "segunda"
    assert rows[0]["mode"] == "chat"


def test_list_sessions_respects_limit(sessions_tmp):
    for i in range(5):
        s = rag.ensure_session(f"id{i}", mode="chat")
        rag.append_turn(s, {"q": f"q{i}"})
        rag.save_session(s)
    assert len(rag.list_sessions(limit=3)) == 3


def test_list_sessions_empty_when_dir_missing(sessions_tmp):
    assert rag.list_sessions() == []


def test_cleanup_sessions_removes_only_expired(sessions_tmp):
    sdir, _ = sessions_tmp

    old = rag.ensure_session("vieja", mode="chat")
    rag.append_turn(old, {"q": "old"})
    rag.save_session(old)
    old_path = sdir / "vieja.json"
    expired = time.time() - 31 * 86400
    os.utime(old_path, (expired, expired))

    new = rag.ensure_session("nueva", mode="chat")
    rag.append_turn(new, {"q": "new"})
    rag.save_session(new)

    removed = rag.cleanup_sessions(ttl_days=30)
    assert removed == 1
    assert not old_path.exists()
    assert (sdir / "nueva.json").exists()


def test_cleanup_sessions_no_dir_returns_zero(sessions_tmp):
    assert rag.cleanup_sessions() == 0


def test_last_session_id_none_before_any_save(sessions_tmp):
    assert rag.last_session_id() is None


def test_last_session_id_tracks_most_recent_save(sessions_tmp):
    a = rag.ensure_session("first", mode="chat")
    rag.append_turn(a, {"q": "x"})
    rag.save_session(a)
    assert rag.last_session_id() == "first"

    b = rag.ensure_session("second", mode="chat")
    rag.append_turn(b, {"q": "y"})
    rag.save_session(b)
    assert rag.last_session_id() == "second"


def test_load_session_returns_none_for_invalid_id(sessions_tmp):
    assert rag.load_session("bad id with spaces") is None
    assert rag.load_session("") is None


def test_load_session_returns_none_when_file_missing(sessions_tmp):
    assert rag.load_session("never_saved") is None


def test_load_session_returns_none_on_corrupt_json(sessions_tmp):
    sdir, _ = sessions_tmp
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "broken.json").write_text("{ not valid json", encoding="utf-8")
    assert rag.load_session("broken") is None


def test_save_session_is_atomic_no_tmp_leftover(sessions_tmp):
    sdir, _ = sessions_tmp
    sess = rag.ensure_session("atomic", mode="chat")
    rag.append_turn(sess, {"q": "hola"})
    rag.save_session(sess)
    leftovers = list(sdir.glob("*.tmp"))
    assert leftovers == []


def test_save_session_updates_updated_at(sessions_tmp):
    sess = rag.ensure_session("stamped", mode="chat")
    original = sess["updated_at"]
    time.sleep(1.1)
    rag.append_turn(sess, {"q": "nueva"})
    rag.save_session(sess)
    assert sess["updated_at"] != original


def test_session_path_rejects_invalid_id(sessions_tmp):
    with pytest.raises(ValueError):
        rag.session_path("invalid id!")
