"""Tests del registry multi-vault.

Cubre la precedencia env > registry > default + el ciclo CRUD del registry
(add / list / use / current / remove). Las commands del CLI se ejecutan
con CliRunner contra un VAULTS_CONFIG_PATH redirigido al tmp_path.
"""
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Aisla el registry en tmp_path. Limpia OBSIDIAN_RAG_VAULT del env."""
    monkeypatch.setattr(rag, "VAULTS_CONFIG_PATH", tmp_path / "vaults.json")
    monkeypatch.delenv("OBSIDIAN_RAG_VAULT", raising=False)
    return tmp_path


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_load_returns_empty_when_missing(tmp_registry):
    cfg = rag._load_vaults_config()
    assert cfg == {"vaults": {}, "current": None}


def test_load_recovers_from_corrupt_file(tmp_registry):
    rag.VAULTS_CONFIG_PATH.write_text("not json{{{")
    assert rag._load_vaults_config() == {"vaults": {}, "current": None}


def test_save_load_roundtrip(tmp_registry):
    rag._save_vaults_config({"vaults": {"home": "/x"}, "current": "home"})
    cfg = rag._load_vaults_config()
    assert cfg["vaults"] == {"home": "/x"}
    assert cfg["current"] == "home"


# ── Precedencia ──────────────────────────────────────────────────────────────


def test_resolve_falls_back_to_default(tmp_registry):
    assert rag._resolve_vault_path() == rag._DEFAULT_VAULT


def test_resolve_uses_registry_current(tmp_registry, tmp_path):
    target = tmp_path / "my-vault"
    target.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(target)},
        "current": "work",
    })
    assert rag._resolve_vault_path() == target


def test_resolve_env_overrides_registry(tmp_registry, tmp_path, monkeypatch):
    target = tmp_path / "my-vault"
    target.mkdir()
    rag._save_vaults_config({
        "vaults": {"work": str(target)},
        "current": "work",
    })
    other = tmp_path / "env-vault"
    other.mkdir()
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", str(other))
    assert rag._resolve_vault_path() == other


def test_resolve_ignores_current_when_name_not_in_vaults(tmp_registry):
    rag._save_vaults_config({"vaults": {}, "current": "ghost"})
    assert rag._resolve_vault_path() == rag._DEFAULT_VAULT


# ── CLI: add ─────────────────────────────────────────────────────────────────


def test_add_first_vault_becomes_current(tmp_registry, tmp_path):
    v = tmp_path / "v1"
    v.mkdir()
    result = CliRunner().invoke(rag.vault, ["add", "v1", str(v)])
    assert result.exit_code == 0, result.output
    assert "(activo)" in result.output
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v1"
    assert cfg["vaults"]["v1"] == str(v)


def test_add_second_vault_does_not_change_current(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v1"
    assert set(cfg["vaults"].keys()) == {"v1", "v2"}


def test_add_overwrites_existing_name(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "x", str(v1)])
    result = runner.invoke(rag.vault, ["add", "x", str(v2)])
    assert "Sobreescribiendo" in result.output
    cfg = rag._load_vaults_config()
    assert cfg["vaults"]["x"] == str(v2)


def test_add_rejects_nonexistent_path(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["add", "ghost", "/no/such/path"])
    assert result.exit_code != 0


# ── CLI: list / current ──────────────────────────────────────────────────────


def test_list_empty_shows_default(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["list"])
    assert result.exit_code == 0
    assert "Sin vaults registrados" in result.output


def test_list_shows_active_marker(tmp_registry, tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    result = runner.invoke(rag.vault, ["list"])
    assert "v" in result.output
    assert "→" in result.output


def test_list_warns_when_env_overrides(tmp_registry, tmp_path, monkeypatch):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = runner.invoke(rag.vault, ["list"])
    assert "OBSIDIAN_RAG_VAULT" in result.output


def test_current_default(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["current"])
    assert "default" in result.output


def test_current_registry(tmp_registry, tmp_path):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    result = runner.invoke(rag.vault, ["current"])
    assert "registry" in result.output
    assert "v" in result.output


def test_current_env(tmp_registry, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = CliRunner().invoke(rag.vault, ["current"])
    assert "env" in result.output
    assert "/some/env/path" in result.output


# ── CLI: use ─────────────────────────────────────────────────────────────────


def test_use_switches_current(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    runner.invoke(rag.vault, ["use", "v2"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v2"


def test_use_unknown_name_errors(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["use", "ghost"])
    assert "no registrado" in result.output


def test_use_warns_when_env_set(tmp_registry, tmp_path, monkeypatch):
    v = tmp_path / "v"; v.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v", str(v)])
    monkeypatch.setenv("OBSIDIAN_RAG_VAULT", "/some/env/path")
    result = runner.invoke(rag.vault, ["use", "v"])
    assert "OBSIDIAN_RAG_VAULT" in result.output


# ── CLI: remove ──────────────────────────────────────────────────────────────


def test_remove_keeps_other_vaults(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    runner.invoke(rag.vault, ["remove", "v1"])
    cfg = rag._load_vaults_config()
    assert "v1" not in cfg["vaults"]
    assert "v2" in cfg["vaults"]


def test_remove_current_falls_to_next_or_none(tmp_registry, tmp_path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v2 = tmp_path / "v2"; v2.mkdir()
    runner = CliRunner()
    runner.invoke(rag.vault, ["add", "v1", str(v1)])
    runner.invoke(rag.vault, ["add", "v2", str(v2)])
    # current = v1 (primero registrado)
    runner.invoke(rag.vault, ["remove", "v1"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] == "v2"
    runner.invoke(rag.vault, ["remove", "v2"])
    cfg = rag._load_vaults_config()
    assert cfg["current"] is None


def test_remove_unknown_errors(tmp_registry):
    result = CliRunner().invoke(rag.vault, ["remove", "ghost"])
    assert "no registrado" in result.output
