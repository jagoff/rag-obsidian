"""Tests directos del LRU de embed() y expand_queries().

Los caches viven in-process: chat multi-turn y bots persistentes repiten
paraphrases idénticas (qwen2.5:3b con seed=42 → determinístico) y embeddings
de las mismas variantes — cachear ahorra llamadas a ollama.

El conftest.py autouse limpia ambos caches entre tests, así que cada caso
arranca con dict vacío.
"""
import json
import threading

import pytest
import rag


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeEmbedResp:
    def __init__(self, vectors):
        self.embeddings = vectors


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChatResp:
    def __init__(self, content):
        self.message = _FakeMessage(content)


def _vec_for(text: str) -> list[float]:
    """Vector determinístico-por-string (cabe en el cap del cache sin colisión)."""
    return [float(len(text)), float(sum(ord(c) for c in text) % 1000)]


# ── embed(): cache hits, misses, eviction ─────────────────────────────────────


@pytest.fixture
def fake_embed(monkeypatch):
    """Mock ollama.embed contando llamadas y los inputs solicitados."""
    state = {"calls": 0, "inputs": []}

    def fake(**kwargs):
        state["calls"] += 1
        inputs = list(kwargs.get("input", []))
        state["inputs"].append(inputs)
        return _FakeEmbedResp([_vec_for(t) for t in inputs])

    monkeypatch.setattr(rag.ollama, "embed", fake)
    return state


def test_empty_input_short_circuits(fake_embed):
    assert rag.embed([]) == []
    assert fake_embed["calls"] == 0


def test_second_call_same_input_does_not_invoke_ollama(fake_embed):
    out1 = rag.embed(["hola mundo"])
    out2 = rag.embed(["hola mundo"])
    assert out1 == out2
    assert fake_embed["calls"] == 1


def test_partial_hit_only_sends_missing(fake_embed):
    rag.embed(["a", "b"])
    assert fake_embed["calls"] == 1
    assert fake_embed["inputs"][-1] == ["a", "b"]

    out = rag.embed(["a", "c", "b", "d"])
    # Segunda llamada pide solo c y d (a, b ya cacheados).
    assert fake_embed["calls"] == 2
    assert fake_embed["inputs"][-1] == ["c", "d"]
    # El orden devuelto debe coincidir con el orden pedido.
    assert out == [_vec_for("a"), _vec_for("c"), _vec_for("b"), _vec_for("d")]


def test_repeated_string_in_one_call_only_sent_once_to_ollama(fake_embed):
    # Si el caller pasa el mismo texto dos veces, el dict de cache no
    # explota y ambas posiciones reciben el mismo vector.
    out = rag.embed(["x", "x"])
    assert out == [_vec_for("x"), _vec_for("x")]
    assert fake_embed["calls"] == 1


def test_eviction_drops_oldest_at_cap(fake_embed, monkeypatch):
    monkeypatch.setattr(rag, "_EMBED_CACHE_MAX", 4)
    # Llenar el cache hasta el cap (4 entries).
    for s in ["a", "b", "c", "d"]:
        rag.embed([s])
    assert len(rag._embed_cache) == 4
    assert fake_embed["calls"] == 4

    # Quinto string evicta "a" (oldest por orden de inserción).
    rag.embed(["e"])
    assert len(rag._embed_cache) == 4
    assert "a" not in rag._embed_cache
    assert "e" in rag._embed_cache

    # Re-pedir "a" cuesta una llamada nueva (ya no está cacheado).
    rag.embed(["a"])
    assert fake_embed["calls"] == 6
    # Ahora "b" es el más viejo y queda evicto.
    assert "b" not in rag._embed_cache


def test_concurrent_embed_same_text_is_thread_safe(fake_embed):
    # 8 threads pidiendo el mismo string. Al final el cache tiene una entrada
    # y todos reciben el mismo vector. (No exigimos call==1 — sin
    # request-coalescing puede haber 1..N llamadas; lo importante es que
    # ningún thread crashee y que todos vean el mismo resultado.)
    results = []
    errors = []

    def worker():
        try:
            results.append(rag.embed(["concurrent-text"])[0])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    assert all(r == _vec_for("concurrent-text") for r in results)
    assert "concurrent-text" in rag._embed_cache


def test_concurrent_embed_distinct_texts_no_loss(fake_embed):
    # Cada thread embed-ea un texto único. Verificamos que el cache final
    # contiene todos los textos sin perder ninguno por race en la eviction.
    texts = [f"distinct-{i}" for i in range(16)]
    errors = []

    def worker(t):
        try:
            rag.embed([t])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in texts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for t in texts:
        assert t in rag._embed_cache


# ── expand_queries(): cache hits, miss-on-failure, eviction ──────────────────


@pytest.fixture
def fake_chat(monkeypatch):
    """Mock ollama.chat contando llamadas; respuesta controlada por test."""
    state = {"calls": 0, "next_response": "paraphrase one\nparaphrase two"}

    def fake(**kwargs):
        state["calls"] += 1
        return _FakeChatResp(state["next_response"])

    monkeypatch.setattr(rag.ollama, "chat", fake)
    return state


def test_expand_second_call_hits_cache(fake_chat):
    # ≥6 tokens para pasar el perf gate (_EXPAND_MIN_TOKENS, bumped a 6 el
    # 2026-04-22 post-audit) y hit del cache.
    q = "qué hora es ahora mismo hoy exactamente"
    out1 = rag.expand_queries(q)
    out2 = rag.expand_queries(q)
    assert out1 == out2
    assert fake_chat["calls"] == 1


def test_expand_returns_copy_not_shared_reference(fake_chat):
    out1 = rag.expand_queries("una pregunta cualquiera")
    out1.append("mutated")
    out2 = rag.expand_queries("una pregunta cualquiera")
    # Si devolviera el mismo list, "mutated" leakaría al segundo caller.
    assert "mutated" not in out2


def test_expand_failure_does_not_pollute_cache(monkeypatch):
    # Si ollama explota, expand devuelve [question] pero NO escribe esa
    # respuesta degradada al cache (sino futuros calls quedarían stuck con
    # un único variant aunque ollama vuelva).
    def boom(**kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag.ollama, "chat", boom)
    out = rag.expand_queries("query rota")
    assert out == ["query rota"]
    assert "query rota" not in rag._expand_cache


def test_expand_eviction_drops_oldest_at_cap(fake_chat, monkeypatch):
    monkeypatch.setattr(rag, "_EXPAND_CACHE_MAX", 3)
    fake_chat["next_response"] = "alt one\nalt two"

    # Queries con ≥6 tokens — las más cortas se saltean el LLM (perf gate
    # bumped a 6 el 2026-04-22) y no pasan por el cache, así que no sirven
    # para validar eviction.
    for q in [
        "pregunta uno alfa que viene hoy",
        "pregunta dos beta que viene hoy",
        "pregunta tres gamma que viene hoy",
    ]:
        rag.expand_queries(q)
    assert len(rag._expand_cache) == 3

    rag.expand_queries("pregunta cuatro delta que viene hoy")
    assert len(rag._expand_cache) == 3
    assert "pregunta uno alfa que viene hoy" not in rag._expand_cache
    assert "pregunta cuatro delta que viene hoy" in rag._expand_cache


def test_concurrent_expand_same_query_is_thread_safe(fake_chat):
    # Mismo query desde 8 threads. Tolerante al número exacto de calls
    # (sin coalescing puede haber 1..N), pero todos los threads ven la
    # misma respuesta y el cache queda con esa entrada.
    results = []
    errors = []

    # ≥6 tokens para pasar el perf gate (bumped 4→6 el 2026-04-22) y
    # llegar al cache.
    q = "pregunta concurrente varios threads hoy en simultaneo"

    def worker():
        try:
            results.append(tuple(rag.expand_queries(q)))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(set(results)) == 1
    assert q in rag._expand_cache


# ── Persistencia disco-backed del expand cache ──────────────────────────────


def test_expand_cache_load_from_disk_populates_in_memory(monkeypatch, tmp_path):
    """Disk-persisted entries must be merged into `_expand_cache` on the
    first `_load_expand_cache()` call. Validates the cold-start fix:
    `rag query` arranca proceso fresh y aprovecha paraphrases acumuladas
    de calls anteriores."""
    cache_path = tmp_path / "expand_cache.json"
    cache_path.write_text(json.dumps({
        "qué tema musical de tool tiene mejor solo": [
            "qué tema musical de tool tiene mejor solo",
            "cuál es el mejor solo de tool",
            "tool: tema con el mejor solo de guitarra",
        ],
    }))
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache_disk_loaded", False)
    rag._expand_cache.clear()

    cache = rag._load_expand_cache()

    assert "qué tema musical de tool tiene mejor solo" in cache
    assert len(cache["qué tema musical de tool tiene mejor solo"]) == 3


def test_expand_cache_load_is_lazy_and_idempotent(monkeypatch, tmp_path):
    """Subsequent `_load_expand_cache()` calls must be a no-op: don't
    re-read the file. We test by deleting the file after the first load —
    if the second call re-read from disk, it would now see "no file" and
    skip merging, which would still leave the cache populated; so the
    real signal here is the `_expand_cache_disk_loaded` flag toggling on
    the first call."""
    cache_path = tmp_path / "expand_cache.json"
    cache_path.write_text(json.dumps({"q one alfa beta gamma delta": ["q one alfa beta gamma delta", "p1", "p2"]}))
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache_disk_loaded", False)
    rag._expand_cache.clear()

    assert rag._expand_cache_disk_loaded is False
    rag._load_expand_cache()
    assert rag._expand_cache_disk_loaded is True

    # Even if the disk file is deleted, the second call doesn't re-read.
    cache_path.unlink()

    second_call = rag._load_expand_cache()
    # Cache content should still be there (didn't reset)
    assert "q one alfa beta gamma delta" in second_call


def test_expand_cache_save_writes_atomically(monkeypatch, tmp_path):
    """Save must use tmp+rename so concurrent writers can't corrupt the
    cache. Verifies that the final file is valid JSON and no `.tmp.*`
    residue leaks."""
    from collections import OrderedDict as _OD

    cache_path = tmp_path / "expand_cache.json"
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache", _OD({"q one alfa": ["q one alfa", "p1", "p2"]}))
    monkeypatch.setattr(rag, "_expand_cache_dirty", True)

    rag._save_expand_cache()

    assert cache_path.is_file()
    assert json.loads(cache_path.read_text()) == {"q one alfa": ["q one alfa", "p1", "p2"]}
    leftovers = list(tmp_path.glob("*.tmp.*"))
    assert leftovers == []


def test_expand_cache_save_skips_when_not_dirty(monkeypatch, tmp_path):
    """If nothing changed since last save, don't touch the disk."""
    from collections import OrderedDict as _OD

    cache_path = tmp_path / "expand_cache.json"
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache", _OD({"x": ["x", "p1"]}))
    monkeypatch.setattr(rag, "_expand_cache_dirty", False)

    rag._save_expand_cache()

    assert not cache_path.is_file()


def test_expand_cache_corrupt_file_is_quarantined(monkeypatch, tmp_path):
    """Corrupt cache → quarantine to `.corrupt-<ts>` and return empty.
    Same hardening pattern as `_load_synthetic_q_cache` and `load_session`.
    """
    cache_path = tmp_path / "expand_cache.json"
    cache_path.write_text("{not json}")
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache_disk_loaded", False)
    rag._expand_cache.clear()

    rag._load_expand_cache()

    assert not cache_path.is_file()
    backups = list(tmp_path.glob("expand_cache.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{not json}"


def test_expand_queries_marks_dirty_on_llm_path(monkeypatch, tmp_path):
    """When `expand_queries()` writes a fresh paraphrase, it must flip
    `_expand_cache_dirty` so the next `atexit` save persists it."""
    cache_path = tmp_path / "expand_cache.json"
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)
    monkeypatch.setattr(rag, "_expand_cache_disk_loaded", True)  # skip disk load
    monkeypatch.setattr(rag, "_expand_cache_dirty", False)
    rag._expand_cache.clear()

    # Make _lookup_learned_paraphrases return nothing so we go LLM path.
    monkeypatch.setattr(rag, "_lookup_learned_paraphrases", lambda *a, **k: [])

    captured = []

    class _Resp:
        class message:
            content = "p1 alfa beta gamma\np2 zeta eta theta"

    class _Client:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return _Resp()

    monkeypatch.setattr(rag, "_helper_client", lambda: _Client())

    out = rag.expand_queries("pregunta original alfa beta gamma delta epsilon")

    assert len(out) >= 1
    assert rag._expand_cache_dirty is True


def test_expand_cache_save_then_load_roundtrip(monkeypatch, tmp_path):
    """Full cycle: write to disk, reset state, load back. No data loss."""
    from collections import OrderedDict as _OD

    cache_path = tmp_path / "expand_cache.json"
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)

    # Save phase
    monkeypatch.setattr(rag, "_expand_cache", _OD({
        "query alfa beta gamma delta epsilon": ["query alfa beta gamma delta epsilon", "p1", "p2"],
        "otra query con varias palabras hoy": ["otra query con varias palabras hoy", "x1", "x2"],
    }))
    monkeypatch.setattr(rag, "_expand_cache_dirty", True)
    rag._save_expand_cache()

    # Load phase: simulate a new process
    rag._expand_cache.clear()
    monkeypatch.setattr(rag, "_expand_cache_disk_loaded", False)

    cache = rag._load_expand_cache()
    assert "query alfa beta gamma delta epsilon" in cache
    assert "otra query con varias palabras hoy" in cache
    assert cache["query alfa beta gamma delta epsilon"] == ["query alfa beta gamma delta epsilon", "p1", "p2"]


def test_expand_cache_concurrent_saves_dont_corrupt(monkeypatch, tmp_path):
    """Stress: 8 threads concurrente llaman _save_expand_cache. El
    atomic write (tmp + os.replace) debe garantizar que el archivo
    final SIEMPRE sea JSON válido — nunca truncado, nunca media-escrito.

    Pre-fix la implementación usaba `write_text()` directo lo que
    permitía interleaving entre threads (1 thread podía leer un
    estado parcial de otro). El fix con tmp+rename hace cada save
    atómico a nivel filesystem (APFS/ext4 garantizan rename atomic).
    """
    from collections import OrderedDict as _OD
    import threading

    cache_path = tmp_path / "expand_cache.json"
    monkeypatch.setattr(rag, "EXPAND_CACHE_PATH", cache_path)

    errors: list[Exception] = []

    def writer(thread_id: int):
        try:
            for i in range(20):
                # Cada thread tiene su propio dict que muta y guarda.
                local = _OD({
                    f"q-thread-{thread_id}-{i}": [
                        f"q-thread-{thread_id}-{i}",
                        "p1",
                        "p2",
                    ],
                })
                # Hack: monkeypatch en thread no aplica — escribimos
                # directo al global con un lock.
                with rag._expand_cache_lock:
                    rag._expand_cache.update(local)
                    rag._expand_cache_dirty = True
                rag._save_expand_cache()
        except Exception as e:
            errors.append(e)

    rag._expand_cache.clear()
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"writer threads raisearon: {errors!r}"

    # Final read MUST succeed — el archivo siempre fue JSON válido.
    assert cache_path.is_file()
    final = json.loads(cache_path.read_text())
    assert isinstance(final, dict)
    # Y no debe haber `.tmp.*` files leakeados (cada thread limpia su tmp).
    leftovers = list(tmp_path.glob("*.tmp.*"))
    assert leftovers == [], f"tmp files leakearon: {leftovers!r}"
