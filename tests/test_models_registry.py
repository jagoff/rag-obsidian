"""Tests del registry `rag/models.py` y CLI `rag model`.

Cobertura:
- `get` / `all_active` / `set_env` / `reset_env` con env vars.
- `validate` para cada tier (whitelist embed, alias chat/helper, etc.).
- `swap` ciclo completo con reload hooks → muta `rag.HELPER_MODEL` /
  `rag.EMBED_MODEL` / `rag.RERANKER_MODEL`.
- CLI `rag model list/set/current/reset` smoke tests.
- Persistencia plist (mock filesystem).
"""

from __future__ import annotations

import plistlib

import pytest
from click.testing import CliRunner


# ── Registry primitives ────────────────────────────────────────────────────


def test_tiers_complete_with_defaults():
    from rag import models
    for tier in models.TIERS:
        assert tier in models.DEFAULTS
        assert tier in models.ENV_VARS
        assert models.ENV_VARS[tier] == f"RAG_{tier.upper()}_MODEL"
        assert models.DEFAULTS[tier]


def test_get_returns_default_when_env_unset(monkeypatch):
    from rag import models
    monkeypatch.delenv("RAG_HELPER_MODEL", raising=False)
    assert models.get("helper") == models.DEFAULTS["helper"]


def test_get_honors_env_override(monkeypatch):
    from rag import models
    monkeypatch.setenv("RAG_CHAT_MODEL", "qwen2.5:14b")
    assert models.get("chat") == "qwen2.5:14b"


def test_get_unknown_tier_raises():
    from rag import models
    with pytest.raises(ValueError, match="Tier desconocido"):
        models.get("bogus")


def test_all_active_snapshot(monkeypatch):
    from rag import models
    for t in models.TIERS:
        monkeypatch.delenv(models.ENV_VARS[t], raising=False)
    snap = models.all_active()
    assert snap == models.DEFAULTS


def test_set_env_returns_previous(monkeypatch):
    from rag import models
    monkeypatch.delenv("RAG_HELPER_MODEL", raising=False)
    prev = models.set_env("helper", "qwen3:4b")
    assert prev == models.DEFAULTS["helper"]
    assert models.get("helper") == "qwen3:4b"
    monkeypatch.delenv("RAG_HELPER_MODEL", raising=False)


# ── Validation per tier ────────────────────────────────────────────────────


def test_validate_embed_whitelist_blocks_unknown():
    from rag import models
    err = models.validate("embed", "some-other-embedder")
    assert err is not None
    assert "whitelist" in err.lower() or "dim" in err.lower()


def test_validate_embed_accepts_alias_and_hf_id():
    from rag import models
    assert models.validate("embed", "qwen3-embedding:0.6b") is None
    assert models.validate("embed", "mlx-community/Qwen3-Embedding-0.6B-8bit") is None


def test_validate_stt_alias_unknown():
    from rag import models
    err = models.validate("stt", "ultra-large-v99")
    assert err is not None
    assert "Whisper" in err


def test_validate_stt_alias_known():
    from rag import models
    assert models.validate("stt", "small") is None
    assert models.validate("stt", "large-v3") is None


def test_validate_stt_full_hf_repo_passes():
    from rag import models
    # cualquier "<org>/<name>" pasa (escape hatch)
    assert models.validate("stt", "mlx-community/whisper-base-mlx") is None


def test_validate_chat_alias():
    from rag import models
    assert models.validate("chat", "qwen2.5:7b") is None
    err = models.validate("chat", "fictional-llm")
    assert err is not None
    assert "MLX_MODEL_ALIAS" in err


def test_validate_empty_model_rejected():
    from rag import models
    err = models.validate("helper", "")
    assert err is not None
    assert "vacío" in err


# ── Swap + reload hooks ────────────────────────────────────────────────────


def test_swap_helper_mutates_constants(monkeypatch):
    """`rag.HELPER_MODEL` y `rag._LOOKUP_MODEL` se actualizan post-swap."""
    import rag
    from rag import models
    monkeypatch.delenv("RAG_HELPER_MODEL", raising=False)
    monkeypatch.delenv("RAG_LOOKUP_MODEL", raising=False)

    # baseline
    assert rag.HELPER_MODEL == models.DEFAULTS["helper"]

    prev = models.swap("helper", "qwen3:4b")
    try:
        assert prev == models.DEFAULTS["helper"]
        assert rag.HELPER_MODEL == "qwen3:4b"
        assert rag._LOOKUP_MODEL == "qwen3:4b"
    finally:
        models.reset_env("helper")
        # post-reset hook re-aplica
        assert rag.HELPER_MODEL == models.DEFAULTS["helper"]


def test_swap_embed_mutates_constant(monkeypatch):
    import rag
    from rag import models
    monkeypatch.delenv("RAG_EMBED_MODEL", raising=False)
    prev = models.swap("embed", "mlx-community/Qwen3-Embedding-0.6B-8bit")
    try:
        assert rag.EMBED_MODEL == "mlx-community/Qwen3-Embedding-0.6B-8bit"
    finally:
        models.reset_env("embed")
        assert rag.EMBED_MODEL == models.DEFAULTS["embed"]


def test_swap_chat_invalidates_resolve_cache(monkeypatch):
    """`resolve_chat_model()` debe leer el override env post-swap."""
    import rag
    from rag import models
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    rag._CHAT_MODEL_RESOLVED = "qwen2.5:7b"  # simular cache previa

    models.swap("chat", "qwen2.5:14b")
    try:
        assert rag.resolve_chat_model() == "qwen2.5:14b"
    finally:
        models.reset_env("chat")


def test_swap_invalid_raises_unless_unsafe(monkeypatch):
    from rag import models
    monkeypatch.delenv("RAG_EMBED_MODEL", raising=False)
    with pytest.raises(ValueError):
        models.swap("embed", "totally-different-embedder")
    # Con unsafe pasa
    prev = models.swap("embed", "totally-different-embedder", unsafe=True)
    try:
        assert prev == models.DEFAULTS["embed"]
        assert models.get("embed") == "totally-different-embedder"
    finally:
        models.reset_env("embed")


def test_swap_runs_registered_hook():
    from rag import models
    received: list[tuple[str, str]] = []

    def hook(old: str, new: str) -> None:
        received.append((old, new))

    models.register_reload_hook("helper", hook)
    try:
        models.swap("helper", "qwen3:4b")
        models.reset_env("helper")
    finally:
        models._RELOAD_HOOKS["helper"].remove(hook)

    assert len(received) == 2
    assert received[0][1] == "qwen3:4b"  # forward swap
    assert received[1][0] == "qwen3:4b"  # reset (back to default)


# ── list_available ─────────────────────────────────────────────────────────


def test_list_available_chat_categorizes():
    from rag import models
    catalog = models.list_available("chat")
    assert "cached" in catalog
    assert "known" in catalog
    # Known list son aliases del subset chat
    chat_aliases = {"qwen2.5:7b", "qwen2.5:14b", "qwen3:30b", "qwen3:30b-a3b",
                    "command-r", "command-r:latest"}
    union = set(catalog["cached"]) | set(catalog["known"])
    assert union == chat_aliases


def test_list_available_stt_uses_whisper_map():
    from rag import models
    from rag.whisper import _WHISPER_NAME_TO_HF
    catalog = models.list_available("stt")
    union = set(catalog["cached"]) | set(catalog["known"])
    assert union == set(_WHISPER_NAME_TO_HF)


# ── CLI commands ───────────────────────────────────────────────────────────


def test_cli_model_list_no_args():
    from rag import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "list"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "helper" in result.output
    assert "embed" in result.output
    assert "rerank" in result.output
    assert "stt" in result.output
    assert "vlm" in result.output


def test_cli_model_list_tier_detail():
    from rag import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "list", "chat"])
    assert result.exit_code == 0
    assert "Activo:" in result.output
    assert "Default:" in result.output
    assert "RAG_CHAT_MODEL" in result.output


def test_cli_model_current_script_friendly(monkeypatch):
    from rag import cli
    monkeypatch.delenv("RAG_HELPER_MODEL", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "current", "helper"])
    assert result.exit_code == 0
    assert result.output.strip() == "qwen2.5:3b"


def test_cli_model_set_ephemeral_then_reset(monkeypatch):
    from rag import cli
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    runner = CliRunner()

    # set
    r = runner.invoke(cli, ["model", "set", "chat", "qwen2.5:14b"])
    assert r.exit_code == 0
    assert "qwen2.5:14b" in r.output

    # current refleja el cambio
    r = runner.invoke(cli, ["model", "current", "chat"])
    assert r.output.strip() == "qwen2.5:14b"

    # reset
    r = runner.invoke(cli, ["model", "reset", "chat"])
    assert r.exit_code == 0


def test_cli_model_set_invalid_aborts(monkeypatch):
    from rag import cli
    monkeypatch.delenv("RAG_EMBED_MODEL", raising=False)
    runner = CliRunner()
    r = runner.invoke(cli, ["model", "set", "embed", "broken-model"])
    assert r.exit_code != 0
    assert "Error" in r.output or "whitelist" in r.output


def test_cli_model_set_unsafe_passes(monkeypatch):
    from rag import cli, models
    monkeypatch.delenv("RAG_EMBED_MODEL", raising=False)
    runner = CliRunner()
    r = runner.invoke(cli, ["model", "set", "embed", "broken-model", "--unsafe"])
    try:
        assert r.exit_code == 0
        assert models.get("embed") == "broken-model"
    finally:
        models.reset_env("embed")


# ── Plist persistence ──────────────────────────────────────────────────────


def test_persist_writes_env_to_plist(tmp_path, monkeypatch):
    """Mock LaunchAgents dir y verificar que el plist se patcha correctamente."""
    from rag.cli import _model_persist as mp

    # Mock del LaunchAgents dir
    fake_la = tmp_path / "LaunchAgents"
    fake_la.mkdir()
    monkeypatch.setattr(mp, "_LAUNCH_AGENTS_DIR", fake_la)

    # Plist mínimo con env vars
    plist_path = fake_la / "com.fer.obsidian-rag-web.plist"
    initial = {
        "Label": "com.fer.obsidian-rag-web",
        "EnvironmentVariables": {"PATH": "/usr/bin"},
    }
    with plist_path.open("wb") as fh:
        plistlib.dump(initial, fh)

    # No-op para kickstart (no querer disparar launchctl real)
    monkeypatch.setattr(mp, "_kickstart", lambda label: None)

    touched = mp.persist_tier_to_plists("chat", "qwen2.5:14b")
    assert "com.fer.obsidian-rag-web.plist" in touched

    # Verificar contenido
    with plist_path.open("rb") as fh:
        data = plistlib.load(fh)
    assert data["EnvironmentVariables"]["RAG_CHAT_MODEL"] == "qwen2.5:14b"
    assert data["EnvironmentVariables"]["PATH"] == "/usr/bin"  # preservado

    # Backup creado
    backups = list(fake_la.glob("*.bak.*"))
    assert len(backups) == 1


def test_unset_removes_env_from_plist(tmp_path, monkeypatch):
    from rag.cli import _model_persist as mp

    fake_la = tmp_path / "LaunchAgents"
    fake_la.mkdir()
    monkeypatch.setattr(mp, "_LAUNCH_AGENTS_DIR", fake_la)
    monkeypatch.setattr(mp, "_kickstart", lambda label: None)

    plist_path = fake_la / "com.fer.obsidian-rag-web.plist"
    initial = {
        "Label": "com.fer.obsidian-rag-web",
        "EnvironmentVariables": {
            "PATH": "/usr/bin",
            "RAG_CHAT_MODEL": "qwen2.5:14b",
        },
    }
    with plist_path.open("wb") as fh:
        plistlib.dump(initial, fh)

    touched = mp.unset_tier_in_plists("chat")
    assert "com.fer.obsidian-rag-web.plist" in touched

    with plist_path.open("rb") as fh:
        data = plistlib.load(fh)
    assert "RAG_CHAT_MODEL" not in data["EnvironmentVariables"]
    assert data["EnvironmentVariables"]["PATH"] == "/usr/bin"


def test_persist_ignores_bak_files(tmp_path, monkeypatch):
    from rag.cli import _model_persist as mp
    fake_la = tmp_path / "LaunchAgents"
    fake_la.mkdir()
    monkeypatch.setattr(mp, "_LAUNCH_AGENTS_DIR", fake_la)
    monkeypatch.setattr(mp, "_kickstart", lambda label: None)

    # Solo un .bak file — no plist real
    bak = fake_la / "com.fer.obsidian-rag-web.plist.bak.20260101-000000"
    with bak.open("wb") as fh:
        plistlib.dump({"Label": "x", "EnvironmentVariables": {}}, fh)

    touched = mp.persist_tier_to_plists("chat", "qwen2.5:14b")
    assert touched == []
