from __future__ import annotations

import os
import plistlib
from pathlib import Path


def test_run_catch_up_index_forces_safe_index_env(monkeypatch, tmp_path):
    import rag
    from rag.cli import setup as setup_cli

    monkeypatch.setattr(setup_cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(setup_cli, "_start_memory_guard", lambda console, where: True)
    monkeypatch.setenv("RAG_INDEX_SAFE", "0")
    monkeypatch.delenv("RAG_START_SAFE", raising=False)

    captured: dict[str, str | None] = {}

    class DummyCtx:
        def invoke(self, command, **kwargs):
            del command
            captured["no_contradict"] = str(kwargs["no_contradict"])
            captured["RAG_INDEX_SAFE"] = os.environ.get("RAG_INDEX_SAFE")
            captured["RAG_INDEX_EMBED_SLICE_SIZE"] = os.environ.get(
                "RAG_INDEX_EMBED_SLICE_SIZE"
            )
            captured["RAG_INDEX_ABORT_ON_MEMORY_PRESSURE"] = os.environ.get(
                "RAG_INDEX_ABORT_ON_MEMORY_PRESSURE"
            )
            captured["OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY"] = os.environ.get(
                "OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY"
            )
            captured["OBSIDIAN_RAG_SKIP_SYNTHETIC_Q"] = os.environ.get(
                "OBSIDIAN_RAG_SKIP_SYNTHETIC_Q"
            )
            captured["RAG_CONTEXTUAL_RETRIEVAL"] = os.environ.get(
                "RAG_CONTEXTUAL_RETRIEVAL"
            )
            captured["RAG_EXTRACT_ENTITIES"] = os.environ.get("RAG_EXTRACT_ENTITIES")

    setup_cli._run_catch_up_index(DummyCtx())

    assert captured == {
        "no_contradict": "True",
        "RAG_INDEX_SAFE": "1",
        "RAG_INDEX_EMBED_SLICE_SIZE": "16",
        "RAG_INDEX_ABORT_ON_MEMORY_PRESSURE": "1",
        "OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY": "1",
        "OBSIDIAN_RAG_SKIP_SYNTHETIC_Q": "1",
        "RAG_CONTEXTUAL_RETRIEVAL": "0",
        "RAG_EXTRACT_ENTITIES": "0",
    }
    assert rag.os.environ.get("RAG_INDEX_SAFE") == "0"
    assert rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE") is None
    assert rag.os.environ.get("RAG_EXTRACT_ENTITIES") is None


def test_run_catch_up_index_allows_explicit_start_safe_opt_out(monkeypatch, tmp_path):
    from rag.cli import setup as setup_cli

    monkeypatch.setattr(setup_cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(setup_cli, "_start_memory_guard", lambda console, where: True)
    monkeypatch.setenv("RAG_START_SAFE", "0")
    monkeypatch.setenv("RAG_INDEX_SAFE", "0")

    captured: dict[str, str | None] = {}

    class DummyCtx:
        def invoke(self, command, **kwargs):
            del command
            captured["no_contradict"] = str(kwargs["no_contradict"])
            captured["RAG_INDEX_SAFE"] = os.environ.get("RAG_INDEX_SAFE")
            captured["RAG_INDEX_EMBED_SLICE_SIZE"] = os.environ.get(
                "RAG_INDEX_EMBED_SLICE_SIZE"
            )
            captured["RAG_EXTRACT_ENTITIES"] = os.environ.get("RAG_EXTRACT_ENTITIES")

    setup_cli._run_catch_up_index(DummyCtx())

    assert captured == {
        "no_contradict": "True",
        "RAG_INDEX_SAFE": "0",
        "RAG_INDEX_EMBED_SLICE_SIZE": None,
        "RAG_EXTRACT_ENTITIES": None,
    }


def test_start_memory_guard_skips_when_pressure_persists(monkeypatch):
    from rag.cli import setup as setup_cli

    messages: list[str] = []

    class DummyConsole:
        def print(self, message):
            messages.append(str(message))

    monkeypatch.delenv("RAG_START_SAFE", raising=False)
    monkeypatch.setenv("RAG_START_MEMORY_PRESSURE_SLEEP_S", "0")
    monkeypatch.setattr(setup_cli, "_start_memory_snapshot", lambda: (95.0, 0.0))

    assert setup_cli._start_memory_guard(DummyConsole(), "daemon test") is False
    assert any("skip daemon test" in message for message in messages)


def test_start_memory_guard_can_be_disabled(monkeypatch):
    from rag.cli import setup as setup_cli

    class DummyConsole:
        def print(self, message):
            raise AssertionError(message)

    monkeypatch.setenv("RAG_START_SAFE", "0")
    monkeypatch.setattr(setup_cli, "_start_memory_snapshot", lambda: (99.0, 8.0))

    assert setup_cli._start_memory_guard(DummyConsole(), "daemon test") is True


def test_parse_swap_used_gb():
    from rag.cli import setup as setup_cli

    assert setup_cli._parse_swap_used_gb(
        "vm.swapusage: total = 1024.00M  used = 512.00M  free = 512.00M"
    ) == 0.5
    assert setup_cli._parse_swap_used_gb(
        "vm.swapusage: total = 4.00G  used = 2.50G  free = 1.50G"
    ) == 2.5


def _write_test_plist(path: Path, args: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        plistlib.dump(
            {
                "Label": path.name.removesuffix(".plist"),
                "ProgramArguments": args,
                "StandardOutPath": str(path.parent / "out.log"),
                "StandardErrorPath": str(path.parent / "err.log"),
            },
            fh,
        )


def test_restore_archived_external_plist_skips_stale_paths(monkeypatch, tmp_path):
    import rag
    from rag.cli import setup as setup_cli

    monkeypatch.setattr(rag, "_LAUNCH_AGENTS_DIR", tmp_path)
    label = "com.fer.whatsapp-bridge"
    older = tmp_path / ".archive-rag-stop-20260513-000000" / f"{label}.plist"
    newer = tmp_path / ".archive-rag-stop-20260515-000000" / f"{label}.plist"
    _write_test_plist(older, ["/bin/echo"])
    _write_test_plist(newer, ["/definitely/missing/whatsapp-bridge"])

    restored = setup_cli._restore_archived_external_plist(label)

    assert restored == tmp_path / f"{label}.plist"
    data = plistlib.loads(restored.read_bytes())
    assert data["ProgramArguments"] == ["/bin/echo"]


def test_rag_net_labels_drop_legacy_whatsapp_vault_sync():
    import rag
    from rag.cli import setup as setup_cli

    assert "com.fer.whatsapp-vault-sync" not in setup_cli._RAG_NET_LABELS
    assert "com.fer.whatsapp-vault-sync" in rag._DEPRECATED_LABELS
    assert "com.fer.whatsapp-listener-mlx-whisper" in setup_cli._RAG_NET_LABELS


def test_ensure_rag_net_plist_generates_bridge_when_no_archive(monkeypatch, tmp_path):
    import rag
    from rag.cli import setup as setup_cli

    launch_agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(rag, "_LAUNCH_AGENTS_DIR", launch_agents)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    binary = tmp_path / "repos/whatsapp-mcp/whatsapp-bridge/whatsapp-bridge"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    plist_path, state = setup_cli._ensure_rag_net_plist("com.fer.whatsapp-bridge")

    assert state == "generated"
    assert plist_path == launch_agents / "com.fer.whatsapp-bridge.plist"
    data = plistlib.loads(plist_path.read_bytes())
    assert data["ProgramArguments"] == [str(binary)]
    assert data["EnvironmentVariables"]["WHATSAPP_BRIDGE_PORT"] == "8088"
    assert Path(data["StandardOutPath"]).parent.is_dir()
