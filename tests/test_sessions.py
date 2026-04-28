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
    sess = rag.ensure_session("tg:12345", mode="legacy")
    assert sess["id"] == "tg:12345"
    assert sess["mode"] == "legacy"
    assert sess["turns"] == []


def test_ensure_session_mints_fresh_id_for_invalid_input(sessions_tmp):
    bad = "foo bar/baz"  # contains space and slash — fails SESSION_ID_RE
    sess = rag.ensure_session(bad, mode="chat")
    assert sess["id"] != bad
    assert rag._valid_session_id(sess["id"])


def test_ensure_session_returns_existing_when_id_matches(sessions_tmp):
    s1 = rag.ensure_session("tg:777", mode="legacy")
    rag.append_turn(s1, {"q": "uno", "a": "a1"})
    rag.save_session(s1)

    s2 = rag.ensure_session("tg:777", mode="legacy")
    assert s2["id"] == "tg:777"
    assert len(s2["turns"]) == 1
    assert s2["turns"][0]["q"] == "uno"


def test_chat_and_query_share_history_when_session_id_matches(sessions_tmp):
    """P1 contract (2026-04-28 listener fix): si el listener manda el mismo
    `session_id` a `/chat` (mode='chat') y `/query` (mode='serve'), los turns
    se acumulan en el mismo file y `session_history` los devuelve juntos.

    Antes el listener mandaba sids distintos:
      - /chat  → wa:${replyTo}            (sin suffix)
      - /query → wa:${BOT_CHAT_JID}:${vault.name}  (suffix de vault, hardcoded)

    Resultado: 5 sessions distintas en disco para el mismo bot, history
    partida cada vez que el routing decidía entre conversational ("hola")
    y factual ("qué sabes de X"). Después del fix los dos endpoints usan
    `wa:${replyTo}` literal.

    El invariante que P1 explota es: `ensure_session` carga la session
    EXISTENTE por id (sin filtrar por mode), así que el `mode` del segundo
    caller es metadata pura. `session_history` recorre TODOS los turns sin
    filtrar — el mode del fundador no afecta el lookup posterior.
    """
    sid = "wa:120363426178035051@g.us"  # formato real RagNet (@g.us)

    # Turn 1 vía /chat (mode='chat')
    s_chat = rag.ensure_session(sid, mode="chat")
    rag.append_turn(s_chat, {"q": "hola", "a": "qué onda?"})
    rag.save_session(s_chat)

    # Turn 2 vía /query (mode='serve') — mismo sid, distinto mode
    s_query = rag.ensure_session(sid, mode="serve")
    assert s_query["id"] == sid
    # ensure_session NO crea una sesión nueva: levanta la del disco con
    # los turns de /chat ya adentro.
    assert len(s_query["turns"]) == 1
    assert s_query["turns"][0]["q"] == "hola"

    rag.append_turn(s_query, {"q": "qué sabes de Maria?", "a": "no tengo info"})
    rag.save_session(s_query)

    # Turn 3 vía /chat de nuevo — debe ver los 2 turns previos.
    s_chat2 = rag.ensure_session(sid, mode="chat")
    hist = rag.session_history(s_chat2)
    contents = [m["content"] for m in hist]
    assert "hola" in contents
    assert "qué onda?" in contents
    assert "qué sabes de Maria?" in contents
    assert "no tengo info" in contents


@pytest.mark.parametrize(
    "jid",
    [
        # JIDs reales que el listener manda como parte de `wa:${replyTo}`.
        "wa:120363426178035051@g.us",                # RagNet (grupo)
        "wa:242738851246099@lid",                    # self-chat (LID)
        "wa:74191281901822@lid",                     # contacto 1:1 (LID)
        "wa:5493425299606@s.whatsapp.net",           # contacto 1:1 (legacy)
        "wa:5491153280296@s.whatsapp.net",           # números argentinos
    ],
)
def test_listener_session_id_format_is_valid(jid):
    """Cada `wa:${replyTo}` literal del listener debe matchear SESSION_ID_RE
    sin transformación. El listener NO sanitiza el `@`; el server lo acepta
    directo (regex `[A-Za-z0-9_.@:-]{1,80}`).

    Si esta aserción rompe, el listener manda un sid inválido → server
    monta un sid fresh aleatorio → la continuidad multi-turn se pierde
    silenciosamente (mismo bug que P1, distinta causa raíz).
    """
    assert rag._valid_session_id(jid), f"sid del listener inválido: {jid!r}"


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


def test_load_session_quarantines_corrupt_file(sessions_tmp):
    """A corrupt session file must be renamed to `.corrupt-<ts>` so the
    next load doesn't loop the same JSONDecodeError. Pre-fix, every chat
    boot would re-log `session_load_json` on the same broken file (~60
    errors/day in the production telemetry as of the 2026-04-25 audit)."""
    sdir, _ = sessions_tmp
    sdir.mkdir(parents=True, exist_ok=True)
    sess_file = sdir / "ghost.json"
    sess_file.write_text("{ corrupt", encoding="utf-8")

    assert rag.load_session("ghost") is None

    # Original gone, corrupt-prefixed sibling must exist.
    assert not sess_file.is_file()
    backups = list(sdir.glob("ghost.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{ corrupt"

    # Subsequent load on the same id is now a clean None (no file at all).
    assert rag.load_session("ghost") is None


def test_save_session_is_atomic_no_tmp_leftover(sessions_tmp):
    sdir, _ = sessions_tmp
    sess = rag.ensure_session("atomic", mode="chat")
    rag.append_turn(sess, {"q": "hola"})
    rag.save_session(sess)
    leftovers = list(sdir.glob("*.tmp"))
    assert leftovers == []


def test_save_session_uses_per_session_flock(sessions_tmp):
    """2026-04-24 audit hardening: save_session debe usar flock sobre un
    sidecar `.lock` file para serializar writers concurrentes. El lock
    se crea on-demand y permanece (byte-free sidecar).

    Este test verifica que:
      1. El lock file se crea al primer save_session.
      2. El lock file persiste después de save_session (no cleanup).
      3. Writers concurrentes NO se pisan (simulado con threads).
    """
    sdir, _ = sessions_tmp
    sess = rag.ensure_session("locked", mode="chat")
    rag.append_turn(sess, {"q": "hola"})
    rag.save_session(sess)

    lock_path = sdir / "locked.json.lock"
    assert lock_path.is_file(), (
        f"lock sidecar file debería existir post-save: {list(sdir.iterdir())}"
    )
    # El lock file queda — no hay cleanup explícito, son sidecars byte-free.
    sess["turns"].append({"q": "segundo", "ts": "2026-04-24T12:00:00"})
    rag.save_session(sess)
    assert lock_path.is_file()


def test_save_session_serializes_concurrent_writes(sessions_tmp):
    """Dos threads escribiendo la misma sesión en paralelo no deben
    corromper el archivo ni perder el file-level atomicity. La
    fcntl.LOCK_EX serializa las escrituras — el último en entrar al
    lock gana (last-writer-wins), pero el archivo final siempre es
    JSON válido parseable.
    """
    import json
    import threading

    sdir, _ = sessions_tmp
    sess_template = rag.ensure_session("concurrent", mode="chat")
    rag.save_session(sess_template)

    errors: list[Exception] = []

    def _writer(turn_id: int) -> None:
        try:
            s = rag.load_session("concurrent") or sess_template
            rag.append_turn(s, {"q": f"turn-{turn_id}", "a": "resp"})
            rag.save_session(s)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writers concurrentes lanzaron errores: {errors}"
    # El archivo final siempre debe ser JSON válido (no half-written).
    final = json.loads((sdir / "concurrent.json").read_text())
    assert final["id"] == "concurrent"
    # Al menos 1 turn se persistió — no podemos asegurar cuántos por la
    # ABA race documentada en el docstring (last-writer-wins), pero el
    # archivo nunca debe estar corrupto o vacío.
    assert isinstance(final.get("turns"), list)


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


# ── session_summary / _compress_turns / reformulate_query summary plumbing ──


def _sess_with_turns(n: int) -> dict:
    """Hand-built session with `n` synthetic q/a turns. Bypasses append_turn so
    we don't rely on system clock or save_session."""
    return {"turns": [{"q": f"q{i}", "a": f"a{i}"} for i in range(n)]}


def test_session_summary_returns_none_below_threshold():
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD - 1)
    assert rag.session_summary(sess) is None
    assert "compressed_history" not in sess


def test_session_summary_returns_none_when_window_covers_all():
    # threshold met, but n - window <= 0 means no aged-out turns to summarize.
    n = rag.SESSION_COMPRESSION_THRESHOLD
    sess = _sess_with_turns(n)
    # bump window so it ≥ n; still triggers threshold path but no aged turns.
    assert rag.session_summary(sess, window=n) is None
    assert "compressed_history" not in sess


# 2026-04-28 P3: tras subir SESSION_HISTORY_WINDOW de 6 → 10 estos tests
# pasan window=6 explícito para mantener su intent (testar la lógica
# "compress turns aged out of window") sin acoplarse al valor del default.
# El número 6 sigue siendo el que producen los `n=10, window=6` de los
# comentarios — el comportamiento testeado es el mismo que antes del
# cambio del default.
_LEGACY_WINDOW = 6


def test_session_summary_computes_and_caches(monkeypatch):
    calls: list[list[dict]] = []

    def fake_compress(turns):
        calls.append(turns)
        return f"resumen de {len(turns)} turnos"

    monkeypatch.setattr(rag, "_compress_turns", fake_compress)
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)  # n=10, window=6
    out = rag.session_summary(sess, window=_LEGACY_WINDOW)
    assert out == "resumen de 4 turnos"  # turns[:n - window] = turns[:4]
    cache = sess["compressed_history"]
    assert cache["version"] == rag.SESSION_SUMMARY_VERSION
    assert cache["covers_until_idx"] == 4
    assert cache["summary"] == "resumen de 4 turnos"
    assert "ts" in cache
    assert len(calls) == 1


def test_session_summary_uses_cache_when_idx_unchanged(monkeypatch):
    call_count = {"n": 0}

    def fake_compress(turns):
        call_count["n"] += 1
        return "first"

    monkeypatch.setattr(rag, "_compress_turns", fake_compress)
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)
    rag.session_summary(sess, window=_LEGACY_WINDOW)
    rag.session_summary(sess, window=_LEGACY_WINDOW)
    rag.session_summary(sess, window=_LEGACY_WINDOW)
    assert call_count["n"] == 1


def test_session_summary_recomputes_when_window_advances(monkeypatch):
    payloads: list[str] = ["v1", "v2"]

    def fake_compress(turns):
        return payloads.pop(0)

    monkeypatch.setattr(rag, "_compress_turns", fake_compress)
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)  # n=10, until=4
    assert rag.session_summary(sess, window=_LEGACY_WINDOW) == "v1"
    sess["turns"].extend([{"q": f"q{i}", "a": f"a{i}"} for i in range(10, 13)])  # n=13, until=7
    assert rag.session_summary(sess, window=_LEGACY_WINDOW) == "v2"
    assert sess["compressed_history"]["covers_until_idx"] == 7


def test_session_summary_invalidates_cache_on_version_bump(monkeypatch):
    call_count = {"n": 0}

    def fake_compress(turns):
        call_count["n"] += 1
        return f"call{call_count['n']}"

    monkeypatch.setattr(rag, "_compress_turns", fake_compress)
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)
    rag.session_summary(sess, window=_LEGACY_WINDOW)
    monkeypatch.setattr(rag, "SESSION_SUMMARY_VERSION", rag.SESSION_SUMMARY_VERSION + 1)
    out = rag.session_summary(sess, window=_LEGACY_WINDOW)
    assert call_count["n"] == 2
    assert out == "call2"


def test_session_summary_falls_back_to_cached_on_empty_compress(monkeypatch):
    """If a recompute yields empty (LLM hiccup), keep serving the previous
    cached summary rather than dropping context entirely."""
    payloads = iter(["good", ""])

    def fake_compress(turns):
        return next(payloads)

    monkeypatch.setattr(rag, "_compress_turns", fake_compress)
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)
    assert rag.session_summary(sess, window=_LEGACY_WINDOW) == "good"
    sess["turns"].extend([{"q": f"q{i}", "a": f"a{i}"} for i in range(10, 14)])
    out = rag.session_summary(sess, window=_LEGACY_WINDOW)
    assert out == "good"  # falls back to cached, doesn't return empty
    # Cache should still reflect the original (idx=4), not the failed advance.
    assert sess["compressed_history"]["covers_until_idx"] == 4


def test_session_summary_returns_none_when_compress_empty_no_cache(monkeypatch):
    monkeypatch.setattr(rag, "_compress_turns", lambda turns: "")
    sess = _sess_with_turns(rag.SESSION_COMPRESSION_THRESHOLD + 3)
    assert rag.session_summary(sess, window=_LEGACY_WINDOW) is None
    assert "compressed_history" not in sess


def test_save_load_roundtrip_persists_compressed_history(sessions_tmp, monkeypatch):
    monkeypatch.setattr(rag, "_compress_turns", lambda turns: "persistido")
    sess = rag.ensure_session("compress-rt", mode="chat")
    for i in range(rag.SESSION_COMPRESSION_THRESHOLD + 3):
        rag.append_turn(sess, {"q": f"q{i}", "a": f"a{i}"})
    rag.session_summary(sess, window=_LEGACY_WINDOW)
    rag.save_session(sess)

    loaded = rag.load_session("compress-rt")
    assert loaded is not None
    assert loaded["compressed_history"]["summary"] == "persistido"
    assert loaded["compressed_history"]["version"] == rag.SESSION_SUMMARY_VERSION
    assert loaded["compressed_history"]["covers_until_idx"] == 4


def test_reformulate_query_includes_summary_section(monkeypatch):
    captured = {}

    class FakeMsg:
        def __init__(self, content): self.content = content

    class FakeResp:
        def __init__(self, content): self.message = FakeMsg(content)

    def fake_chat(*, model, messages, options, keep_alive):
        captured["prompt"] = messages[0]["content"]
        return FakeResp("reformulado")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    history = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "que tal"},
    ]
    out = rag.reformulate_query("y eso?", history, summary="conversación previa sobre X")
    assert out == "reformulado"
    assert "Resumen de turnos previos:" in captured["prompt"]
    assert "conversación previa sobre X" in captured["prompt"]
    # Original recent history still rendered.
    assert "Usuario: hola" in captured["prompt"]


def test_reformulate_query_no_summary_baseline(monkeypatch):
    captured = {}

    class FakeMsg:
        def __init__(self, content): self.content = content

    class FakeResp:
        def __init__(self, content): self.message = FakeMsg(content)

    def fake_chat(*, model, messages, options, keep_alive):
        captured["prompt"] = messages[0]["content"]
        return FakeResp("reformulado")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    rag.reformulate_query(
        "y eso?",
        [{"role": "user", "content": "hola"}],
    )
    assert "Resumen de turnos previos:" not in captured["prompt"]


def test_reformulate_query_short_circuits_when_neither_history_nor_summary():
    # No LLM call expected — function returns the question verbatim.
    assert rag.reformulate_query("¿qué es X?", []) == "¿qué es X?"


def test_reformulate_query_works_with_summary_only(monkeypatch):
    """Edge case: summary present but raw history empty (e.g., very long
    session whose recent window has just been pruned). Should still fire."""
    captured = {}

    class FakeMsg:
        def __init__(self, content): self.content = content

    class FakeResp:
        def __init__(self, content): self.message = FakeMsg(content)

    def fake_chat(*, model, messages, options, keep_alive):
        captured["prompt"] = messages[0]["content"]
        return FakeResp("ok")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    out = rag.reformulate_query("y eso?", [], summary="contexto previo")
    assert out == "ok"
    assert "contexto previo" in captured["prompt"]
