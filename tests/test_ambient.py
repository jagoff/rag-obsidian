"""Tests para el Ambient Agent: reactivo a saves en 00-Inbox/.

Mockea la red (Telegram send) y las primitivas caras (find_related) donde
hace falta; los checks determinísticos (skip rules, dedup window, config)
corren con fixtures de tmp_path.
"""
import json
import time
from pathlib import Path

import chromadb
import pytest

import rag


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(
        rag, "AMBIENT_CONFIG_PATH",
        tmp_path / "ambient.json",
    )
    monkeypatch.setattr(
        rag, "AMBIENT_STATE_PATH",
        tmp_path / "ambient_state.jsonl",
    )
    monkeypatch.setattr(
        rag, "AMBIENT_LOG_PATH",
        tmp_path / "ambient.jsonl",
    )
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="amb_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()
    return vault, col


@pytest.fixture
def captured_sends(monkeypatch):
    """Collect Telegram payloads instead of hitting the network."""
    calls: list[dict] = []

    def fake_send(chat_id, bot_token, text):
        calls.append({"chat_id": chat_id, "bot_token": bot_token, "text": text})
        return True

    monkeypatch.setattr(rag, "_ambient_telegram_send", fake_send)
    return calls


def _write_config(path: Path, enabled: bool = True):
    path.write_text(json.dumps({
        "chat_id": "123", "bot_token": "fake-token", "enabled": enabled,
    }))


def _add_chunk(col, rel, note, outlinks="", extra_meta=None):
    meta = {
        "file": rel, "note": note, "folder": str(Path(rel).parent),
        "tags": "", "outlinks": outlinks, "hash": "x",
    }
    if extra_meta:
        meta.update(extra_meta)
    col.add(
        ids=[f"{rel}::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=[f"chunk for {note}"],
        metadatas=[meta],
    )


# ── Config gate ──────────────────────────────────────────────────────────────


def test_hook_noops_when_config_missing(tmp_vault, captured_sends):
    vault, col = tmp_vault
    p = vault / "00-Inbox" / "n.md"
    p.write_text("body")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    assert captured_sends == []


def test_hook_noops_when_enabled_false(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH, enabled=False)
    p = vault / "00-Inbox" / "n.md"
    p.write_text("body")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    assert captured_sends == []


def test_hook_noops_outside_inbox(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    target = vault / "02-Areas"
    target.mkdir(parents=True, exist_ok=True)
    p = target / "n.md"
    p.write_text("body")
    rag._ambient_hook(col, p, "02-Areas/n.md", "h1")
    assert captured_sends == []


# ── Frontmatter opt-out ──────────────────────────────────────────────────────


def test_hook_skips_on_frontmatter_skip(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    p = vault / "00-Inbox" / "n.md"
    p.write_text("---\nambient: skip\n---\nbody")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    assert captured_sends == []


def test_hook_skips_system_notes(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    p = vault / "00-Inbox" / "n.md"
    p.write_text("---\ntype: morning-brief\n---\nbody")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    assert captured_sends == []


# ── Dedup window ─────────────────────────────────────────────────────────────


def test_hook_dedups_same_hash_within_window(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    p = vault / "00-Inbox" / "n.md"
    p.write_text("capturé una idea")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    n1 = len(captured_sends)
    # Same path + hash → must skip
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")
    assert len(captured_sends) == n1


def test_hook_reanalyzes_on_different_hash(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    # Seed a target note so related/wikilinks have something to find.
    (vault / "02-Areas").mkdir(parents=True, exist_ok=True)
    (vault / "02-Areas" / "Claude.md").write_text("about Claude")
    _add_chunk(col, "02-Areas/Claude.md", "Claude")
    rag._invalidate_corpus_cache()

    p = vault / "00-Inbox" / "n.md"
    p.write_text("escribí algo sobre Claude")
    _add_chunk(col, "00-Inbox/n.md", "n", extra_meta={"tags": ""})
    rag._invalidate_corpus_cache()

    rag._ambient_hook(col, p, "00-Inbox/n.md", "hash1")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "hash2")
    # Both fire — different hashes
    assert len(captured_sends) >= 1  # hook may produce quiet events
    # State file should have 2 records regardless
    lines = rag.AMBIENT_STATE_PATH.read_text().splitlines()
    assert len(lines) == 2


# ── Wikilink auto-apply ──────────────────────────────────────────────────────


def test_hook_applies_wikilinks_in_inbox_note(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    # Seed a vault title to link against.
    target = vault / "02-Areas" / "Ikigai.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Ikigai es un concepto japonés.")
    _add_chunk(col, "02-Areas/Ikigai.md", "Ikigai")
    rag._invalidate_corpus_cache()

    p = vault / "00-Inbox" / "captura.md"
    p.write_text("Pensé sobre Ikigai hoy en una caminata.")
    rag._ambient_hook(col, p, "00-Inbox/captura.md", "h1")

    assert "[[Ikigai]]" in p.read_text()


def test_hook_sends_telegram_when_findings(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    # Target note → auto-wikilink + related
    target = vault / "02-Areas" / "Claude.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Claude Code stuff")
    _add_chunk(col, "02-Areas/Claude.md", "Claude")
    rag._invalidate_corpus_cache()

    p = vault / "00-Inbox" / "n.md"
    p.write_text("Trabajando con Claude hoy")
    rag._ambient_hook(col, p, "00-Inbox/n.md", "h1")

    # At least one send with info
    assert captured_sends, "expected telegram send"
    msg = captured_sends[0]["text"]
    assert "Ambient" in msg
    assert "[[n]]" in msg


def test_hook_stays_quiet_without_findings(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    p = vault / "00-Inbox" / "aa.md"     # very short title, below min_title_len
    p.write_text("contenido sin referencias ni duplicados ni nada.")
    rag._ambient_hook(col, p, "00-Inbox/aa.md", "h1")
    # Nothing interesting → no telegram noise
    assert captured_sends == []


# ── Config helpers ───────────────────────────────────────────────────────────


def test_ambient_config_reads_valid_json(tmp_vault):
    _write_config(rag.AMBIENT_CONFIG_PATH)
    c = rag._ambient_config()
    assert c is not None
    assert c["chat_id"] == "123"
    assert c["bot_token"] == "fake-token"


def test_ambient_config_rejects_missing_fields(tmp_vault):
    rag.AMBIENT_CONFIG_PATH.write_text(json.dumps({"chat_id": "123"}))
    assert rag._ambient_config() is None


def test_ambient_config_rejects_enabled_false(tmp_vault):
    _write_config(rag.AMBIENT_CONFIG_PATH, enabled=False)
    assert rag._ambient_config() is None


def test_ambient_config_rejects_corrupt_json(tmp_vault):
    rag.AMBIENT_CONFIG_PATH.write_text("not json at all")
    assert rag._ambient_config() is None


# ── State file ───────────────────────────────────────────────────────────────


def test_ambient_state_records_analysis(tmp_vault):
    rag._ambient_state_record("00-Inbox/n.md", "h1", {"wikilinks_applied": 2})
    assert rag.AMBIENT_STATE_PATH.is_file()
    line = rag.AMBIENT_STATE_PATH.read_text().strip()
    e = json.loads(line)
    assert e["path"] == "00-Inbox/n.md"
    assert e["hash"] == "h1"
    assert e["wikilinks_applied"] == 2
    assert "analyzed_at" in e


def test_ambient_should_skip_false_without_state(tmp_vault):
    assert rag._ambient_should_skip("00-Inbox/n.md", "h1") is False


def test_ambient_should_skip_true_within_window(tmp_vault):
    rag._ambient_state_record("00-Inbox/n.md", "h1", {})
    assert rag._ambient_should_skip("00-Inbox/n.md", "h1") is True


def test_ambient_should_skip_false_for_different_hash(tmp_vault):
    rag._ambient_state_record("00-Inbox/n.md", "h1", {})
    assert rag._ambient_should_skip("00-Inbox/n.md", "h2") is False
