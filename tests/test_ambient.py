"""Tests para el Ambient Agent: reactivo a saves en 00-Inbox/.

Mockea la red (WhatsApp bridge send) y las primitivas caras (find_related)
donde hace falta; los checks determinísticos (skip rules, dedup window,
config) corren con fixtures de tmp_path.
"""
import json
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
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
    client = _TestVecClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="amb_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()
    return vault, col


@pytest.fixture
def captured_sends(monkeypatch):
    """Collect WhatsApp bridge payloads instead of hitting the network."""
    calls: list[dict] = []

    def fake_send(jid, text):
        calls.append({"jid": jid, "text": text})
        return True

    monkeypatch.setattr(rag, "_ambient_whatsapp_send", fake_send)
    return calls


def _write_config(path: Path, enabled: bool = True, jid: str = "120363426178035051@g.us"):
    path.write_text(json.dumps({
        "jid": jid, "enabled": enabled,
    }))


def _write_legacy_telegram_config(path: Path):
    """Schema viejo — se usa para verificar bwcompat (debe ser rechazado)."""
    path.write_text(json.dumps({
        "chat_id": "123", "bot_token": "fake-token", "enabled": True,
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


def test_hook_sends_whatsapp_when_findings(tmp_vault, captured_sends):
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
    assert captured_sends, "expected whatsapp send"
    msg = captured_sends[0]["text"]
    assert "Ambient" in msg
    assert "[[n]]" in msg
    assert captured_sends[0]["jid"] == "120363426178035051@g.us"


def test_hook_stays_quiet_without_findings(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)
    p = vault / "00-Inbox" / "aa.md"     # very short title, below min_title_len
    p.write_text("contenido sin referencias ni duplicados ni nada.")
    rag._ambient_hook(col, p, "00-Inbox/aa.md", "h1")
    # Nothing interesting → no whatsapp noise
    assert captured_sends == []


# ── Config helpers ───────────────────────────────────────────────────────────


def test_ambient_config_reads_valid_json(tmp_vault):
    _write_config(rag.AMBIENT_CONFIG_PATH)
    c = rag._ambient_config()
    assert c is not None
    assert c["jid"] == "120363426178035051@g.us"
    assert c["enabled"] is True


def test_ambient_config_rejects_missing_jid(tmp_vault):
    rag.AMBIENT_CONFIG_PATH.write_text(json.dumps({"enabled": True}))
    assert rag._ambient_config() is None


def test_ambient_config_rejects_enabled_false(tmp_vault):
    _write_config(rag.AMBIENT_CONFIG_PATH, enabled=False)
    assert rag._ambient_config() is None


def test_ambient_config_rejects_corrupt_json(tmp_vault):
    rag.AMBIENT_CONFIG_PATH.write_text("not json at all")
    assert rag._ambient_config() is None


def test_ambient_config_rejects_legacy_telegram_schema(tmp_vault):
    """Bwcompat: schema viejo (chat_id/bot_token) debe devolver None para que
    el usuario re-habilite contra el bot de WhatsApp.
    """
    _write_legacy_telegram_config(rag.AMBIENT_CONFIG_PATH)
    assert rag._ambient_config() is None


# ── WhatsApp send ────────────────────────────────────────────────────────────


def test_whatsapp_send_prefixes_antiloop_marker(monkeypatch):
    """El outgoing debe arrancar con U+200B para que el listener del bot lo
    ignore (si no, el listener re-procesaría el ping como query entrante).
    """
    captured: dict = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok = rag._ambient_whatsapp_send("120@g.us", "hola mundo")
    assert ok is True
    assert captured["url"] == rag.AMBIENT_WHATSAPP_BRIDGE_URL
    assert captured["body"]["recipient"] == "120@g.us"
    assert captured["body"]["message"].startswith("\u200b")
    assert captured["body"]["message"].endswith("hola mundo")


def test_whatsapp_send_does_not_double_prefix(monkeypatch):
    """Si el texto ya arranca con U+200B, no re-prefixar."""
    captured: dict = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rag._ambient_whatsapp_send("120@g.us", "\u200bprefixed")
    assert captured["body"]["message"] == "\u200bprefixed"


def test_whatsapp_send_returns_false_on_network_error(monkeypatch):
    def boom(req, timeout):
        raise OSError("bridge down")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert rag._ambient_whatsapp_send("120@g.us", "x") is False


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


# ── allowed_folders ──────────────────────────────────────────────────────────


def _write_config_with_folders(path: Path, folders):
    path.write_text(json.dumps({
        "jid": "120363426178035051@g.us",
        "enabled": True,
        "allowed_folders": folders,
    }))


def test_config_default_when_no_allowed_folders(tmp_vault):
    """Sin `allowed_folders` → `_ambient_config` retorna None en ese campo;
    el hook cae al default [00-Inbox] al leer.
    """
    _write_config(rag.AMBIENT_CONFIG_PATH)
    c = rag._ambient_config()
    assert c is not None
    assert c.get("allowed_folders") is None


def test_config_normalizes_trailing_slashes(tmp_vault):
    _write_config_with_folders(
        rag.AMBIENT_CONFIG_PATH, ["01-Projects/", "02-Areas"]
    )
    c = rag._ambient_config()
    assert c is not None
    assert c["allowed_folders"] == ["01-Projects", "02-Areas"]


def test_config_drops_empty_entries(tmp_vault):
    _write_config_with_folders(
        rag.AMBIENT_CONFIG_PATH, ["", "  ", "01-Projects"]
    )
    c = rag._ambient_config()
    assert c is not None
    assert c["allowed_folders"] == ["01-Projects"]


def test_config_empty_list_falls_back_to_default_field_none(tmp_vault):
    # `allowed_folders: []` → normalized to None so consumer uses default.
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, [])
    c = rag._ambient_config()
    assert c is not None
    assert c.get("allowed_folders") is None


def test_hook_default_only_fires_inside_inbox(tmp_vault, captured_sends):
    """Config sin allowed_folders → sólo 00-Inbox (comportamiento legacy)."""
    vault, col = tmp_vault
    _write_config(rag.AMBIENT_CONFIG_PATH)

    # Outside inbox → no send.
    (vault / "01-Projects").mkdir(parents=True, exist_ok=True)
    p_proj = vault / "01-Projects" / "n.md"
    p_proj.write_text("body")
    rag._ambient_hook(col, p_proj, "01-Projects/n.md", "h1")
    assert captured_sends == []


def test_hook_fires_on_configured_folder(tmp_vault, captured_sends):
    """Config con `01-Projects` → hook corre ahí y NO corre en 00-Inbox."""
    vault, col = tmp_vault
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, ["01-Projects"])

    # Seed a target note with a distinctive title for auto-wikilink.
    (vault / "02-Areas").mkdir(parents=True, exist_ok=True)
    (vault / "02-Areas" / "Ikigai.md").write_text("Ikigai concepto japonés.")
    _add_chunk(col, "02-Areas/Ikigai.md", "Ikigai")
    rag._invalidate_corpus_cache()

    # Note in 01-Projects → should be analyzed.
    (vault / "01-Projects").mkdir(parents=True, exist_ok=True)
    p_proj = vault / "01-Projects" / "nota.md"
    p_proj.write_text("Pensé sobre Ikigai hoy.")
    rag._ambient_hook(col, p_proj, "01-Projects/nota.md", "hproj")
    assert "[[Ikigai]]" in p_proj.read_text()

    # Note in 00-Inbox → NOT analyzed (inbox not in allowed_folders).
    p_in = vault / "00-Inbox" / "captura.md"
    p_in.write_text("Otra sobre Ikigai pero en inbox.")
    rag._ambient_hook(col, p_in, "00-Inbox/captura.md", "hinbox")
    assert "[[Ikigai]]" not in p_in.read_text()


def test_hook_fires_on_multiple_configured_folders(tmp_vault, captured_sends):
    vault, col = tmp_vault
    _write_config_with_folders(
        rag.AMBIENT_CONFIG_PATH, ["00-Inbox", "01-Projects"]
    )
    (vault / "02-Areas").mkdir(parents=True, exist_ok=True)
    (vault / "02-Areas" / "Ikigai.md").write_text("Ikigai")
    _add_chunk(col, "02-Areas/Ikigai.md", "Ikigai")
    rag._invalidate_corpus_cache()

    # Both folders should analyze.
    (vault / "01-Projects").mkdir(parents=True, exist_ok=True)
    p1 = vault / "01-Projects" / "a.md"
    p1.write_text("idea sobre Ikigai")
    rag._ambient_hook(col, p1, "01-Projects/a.md", "h1")
    assert "[[Ikigai]]" in p1.read_text()

    p2 = vault / "00-Inbox" / "b.md"
    p2.write_text("otra sobre Ikigai")
    rag._ambient_hook(col, p2, "00-Inbox/b.md", "h2")
    assert "[[Ikigai]]" in p2.read_text()


def test_hook_does_not_match_prefix_substring(tmp_vault, captured_sends):
    """`01-Projects` no debe matchear `01-ProjectsOld/`. Verifica que el
    check incluya el trailing slash.
    """
    vault, col = tmp_vault
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, ["01-Projects"])
    (vault / "01-ProjectsOld").mkdir(parents=True, exist_ok=True)
    p = vault / "01-ProjectsOld" / "n.md"
    p.write_text("body")
    rag._ambient_hook(col, p, "01-ProjectsOld/n.md", "h1")
    assert captured_sends == []


# ── rag ambient folders {add,remove,list} ────────────────────────────────────


def test_ambient_folders_list_default(tmp_vault):
    from click.testing import CliRunner
    _write_config(rag.AMBIENT_CONFIG_PATH)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "list"])
    assert result.exit_code == 0
    assert "00-Inbox" in result.output


def test_ambient_folders_add_persists(tmp_vault):
    from click.testing import CliRunner
    vault, _ = tmp_vault
    (vault / "01-Projects").mkdir(parents=True, exist_ok=True)
    _write_config(rag.AMBIENT_CONFIG_PATH)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "add", "01-Projects"])
    assert result.exit_code == 0
    assert "Agregado" in result.output or "Ya estaba" in result.output
    raw = json.loads(rag.AMBIENT_CONFIG_PATH.read_text())
    assert "01-Projects" in (raw.get("allowed_folders") or [])
    # Preserves jid + enabled
    assert raw.get("jid")
    assert raw.get("enabled") is True


def test_ambient_folders_add_dedup(tmp_vault):
    from click.testing import CliRunner
    vault, _ = tmp_vault
    (vault / "01-Projects").mkdir(parents=True, exist_ok=True)
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, ["01-Projects"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "add", "01-Projects"])
    assert result.exit_code == 0
    assert "Ya estaba" in result.output
    raw = json.loads(rag.AMBIENT_CONFIG_PATH.read_text())
    assert raw["allowed_folders"].count("01-Projects") == 1


def test_ambient_folders_add_validates_folder_exists(tmp_vault):
    from click.testing import CliRunner
    _write_config(rag.AMBIENT_CONFIG_PATH)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "add", "99-DoesNotExist"])
    assert result.exit_code == 0
    assert "No existe" in result.output
    raw = json.loads(rag.AMBIENT_CONFIG_PATH.read_text())
    assert "99-DoesNotExist" not in (raw.get("allowed_folders") or [])


def test_ambient_folders_remove_restores_default(tmp_vault):
    from click.testing import CliRunner
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, ["01-Projects"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "remove", "01-Projects"])
    assert result.exit_code == 0
    assert "Default restaurado" in result.output
    raw = json.loads(rag.AMBIENT_CONFIG_PATH.read_text())
    # Field wiped so reader falls back to default.
    assert "allowed_folders" not in raw or not raw.get("allowed_folders")


def test_ambient_folders_remove_keeps_others(tmp_vault):
    from click.testing import CliRunner
    _write_config_with_folders(
        rag.AMBIENT_CONFIG_PATH, ["00-Inbox", "01-Projects"]
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "remove", "01-Projects"])
    assert result.exit_code == 0
    raw = json.loads(rag.AMBIENT_CONFIG_PATH.read_text())
    assert raw["allowed_folders"] == ["00-Inbox"]


def test_ambient_folders_remove_missing_folder(tmp_vault):
    from click.testing import CliRunner
    _write_config_with_folders(rag.AMBIENT_CONFIG_PATH, ["00-Inbox"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ambient", "folders", "remove", "01-Projects"])
    assert result.exit_code == 0
    assert "No estaba" in result.output
