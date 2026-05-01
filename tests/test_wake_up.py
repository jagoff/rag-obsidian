"""Tests del orquestador `rag wake-up` + el plist que lo dispara a las 04:00.

Regresiones que atrapan:
  - El comando se registra en Click y aparece en `rag --help`.
  - `--dry-run` no dispara ningún step real (smoke test).
  - Cada `--skip-*` flag realmente saltea su step.
  - Los pasos dependen del orden declarado: index → bookmarks → wa-tasks →
    maintenance → radars → morning → warmup. Si alguien reordena y rompe la
    secuencia, el test detecta el cambio.
  - Si un step tira exception, los siguientes igual corren (independencia).
  - Exit code != 0 cuando al menos un step falla (para que launchd lo
    marque rojo en /status).
  - El plist tiene shape correcta: label, schedule 04:00, program args,
    env vars (OLLAMA_KEEP_ALIVE=-1 crítico para que el warmup persista).
  - El plist está registrado en `_services_spec` (sino `rag setup` no lo
    instala).
  - /api/status incluye la entry `com.fer.obsidian-rag-wake-up` en la
    categoría briefs.

No testeamos:
  - La ejecución real de cada sub-step (cada uno tiene sus propios tests).
  - Performance end-to-end (~15-20min — no corre en CI).
"""
from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module  # noqa: E402

RAG_BIN = "/usr/local/bin/rag"


# ── CLI registration ────────────────────────────────────────────────

def test_wake_up_command_registered():
    """`rag wake-up` aparece en el CLI."""
    assert "wake-up" in rag_module.cli.commands


def test_wake_up_help_has_all_skip_flags():
    """El `--help` lista los 7 skips + dry-run (contract para launchd/cron)."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, ["wake-up", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--dry-run", "--skip-index", "--skip-bookmarks", "--skip-wa-tasks",
        "--skip-maintenance", "--skip-radars", "--skip-brief", "--skip-warmup",
    ):
        assert flag in result.output, f"falta {flag} en --help"


# ── Dry-run semantics ───────────────────────────────────────────────

def test_wake_up_dry_run_skips_everything():
    """Con --dry-run ningún sub-command se llama."""
    runner = CliRunner()
    called = []

    with patch.object(rag_module, "index", side_effect=lambda *a, **kw: called.append("index")), \
         patch.object(rag_module, "maintenance", side_effect=lambda *a, **kw: called.append("maintenance")), \
         patch.object(rag_module, "patterns", side_effect=lambda *a, **kw: called.append("patterns")), \
         patch.object(rag_module, "emergent", side_effect=lambda *a, **kw: called.append("emergent")), \
         patch.object(rag_module, "morning", side_effect=lambda *a, **kw: called.append("morning")), \
         patch.object(rag_module.ollama, "chat", side_effect=lambda *a, **kw: called.append("ollama")):
        result = runner.invoke(rag_module.cli, ["wake-up", "--dry-run"])

    assert result.exit_code == 0
    assert called == []  # nada se llamó
    # Output muestra los 8 steps (index, bookmarks, wa-tasks, maintenance,
    # patterns, emergent, morning, warmup).
    assert "8 pasos" in result.output
    assert "dry-run: skippeado" in result.output


def test_wake_up_skip_all_still_valid():
    """Skippear todo → 0 pasos, exit 0 (no crash)."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run",
        "--skip-index", "--skip-bookmarks", "--skip-wa-tasks",
        "--skip-maintenance", "--skip-radars", "--skip-brief", "--skip-warmup",
    ])
    assert result.exit_code == 0
    assert "0 pasos" in result.output


def test_wake_up_skip_warmup_reduces_step_count():
    """--skip-warmup → 7 pasos en vez de 8."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run", "--skip-warmup",
    ])
    assert result.exit_code == 0
    assert "7 pasos" in result.output


def test_wake_up_skip_index_removes_only_index():
    """--skip-index → 7 pasos (bookmarks + wa-tasks + maintenance + radars +
    morning + warmup quedan), index no aparece."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run", "--skip-index",
    ])
    assert result.exit_code == 0
    assert "7 pasos" in result.output
    assert "rag index" not in result.output
    assert "rag bookmarks sync" in result.output
    assert "rag wa-tasks" in result.output
    assert "rag maintenance" in result.output


def test_wake_up_skip_bookmarks_removes_only_bookmarks():
    """--skip-bookmarks → 7 pasos, bookmarks sync no aparece pero los
    otros sí (incluyendo wa-tasks). `rag index` igual sincroniza
    bookmarks como pre-sync, pero el log de wake-up no muestra paso
    explícito."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run", "--skip-bookmarks",
    ])
    assert result.exit_code == 0
    assert "7 pasos" in result.output
    assert "rag bookmarks sync" not in result.output
    assert "rag wa-tasks" in result.output
    assert "rag index" in result.output


def test_wake_up_skip_wa_tasks_removes_only_wa_tasks():
    """--skip-wa-tasks → 7 pasos, wa-tasks no aparece. Útil cuando el
    cron de 30min `com.fer.obsidian-rag-wa-tasks` ya cubre el extractor
    y querés evitar la duplicación en el wake-up."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run", "--skip-wa-tasks",
    ])
    assert result.exit_code == 0
    assert "7 pasos" in result.output
    assert "rag wa-tasks" not in result.output
    assert "rag bookmarks sync" in result.output


def test_wake_up_skip_radars_removes_both_radars():
    """--skip-radars → 6 pasos (patterns + emergent juntos)."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, [
        "wake-up", "--dry-run", "--skip-radars",
    ])
    assert result.exit_code == 0
    assert "6 pasos" in result.output
    assert "rag patterns" not in result.output
    assert "rag emergent" not in result.output
    assert "rag morning" in result.output


# ── Execution order & independence ──────────────────────────────────

def test_wake_up_runs_steps_in_declared_order():
    """Sequence de llamados: index → bookmarks → wa-tasks → maintenance →
    patterns → emergent → morning → (warmup). Si alguien reordena, el
    test falla."""
    runner = CliRunner()
    call_order: list[str] = []

    def _stub(name: str):
        def _fn(*a, **kw):
            call_order.append(name)
        return _fn

    # `wake_up` hace `from rag.wa_tasks import wa_tasks` en su cuerpo, por
    # lo que parchamos el atributo `wa_tasks` del submódulo. Cuidado:
    # `rag.wa_tasks` está shadowed por la función Click `wa_tasks`
    # re-exportada al final de `rag/__init__.py` — `import rag.wa_tasks`
    # devuelve la función, no el módulo. Usamos `sys.modules` para
    # acceder al submódulo real.
    rag_wa_tasks_mod = sys.modules["rag.wa_tasks"]

    with patch.object(rag_module, "index", new=_stub("index")), \
         patch.object(rag_module, "bookmarks_sync", new=_stub("bookmarks")), \
         patch.object(rag_wa_tasks_mod, "wa_tasks", new=_stub("wa_tasks")), \
         patch.object(rag_module, "maintenance", new=_stub("maintenance")), \
         patch.object(rag_module, "patterns", new=_stub("patterns")), \
         patch.object(rag_module, "emergent", new=_stub("emergent")), \
         patch.object(rag_module, "morning", new=_stub("morning")), \
         patch.object(rag_module.ollama, "chat",
                      side_effect=lambda *a, **kw: call_order.append("ollama")), \
         patch.object(rag_module, "resolve_chat_model",
                      return_value="qwen2.5:7b"):
        result = runner.invoke(rag_module.cli, ["wake-up"])

    assert result.exit_code == 0, result.output
    assert call_order == ["index", "bookmarks", "wa_tasks", "maintenance",
                          "patterns", "emergent", "morning", "ollama"]


def test_wake_up_continues_after_step_failure():
    """Si un step tira exception, los siguientes igual corren."""
    runner = CliRunner()
    call_order: list[str] = []

    def _stub(name: str):
        def _fn(*a, **kw):
            call_order.append(name)
        return _fn

    def _explode(*a, **kw):
        call_order.append("maintenance-FAIL")
        raise RuntimeError("simulated failure")

    rag_wa_tasks_mod = sys.modules["rag.wa_tasks"]

    with patch.object(rag_module, "index", new=_stub("index")), \
         patch.object(rag_module, "bookmarks_sync", new=_stub("bookmarks")), \
         patch.object(rag_wa_tasks_mod, "wa_tasks", new=_stub("wa_tasks")), \
         patch.object(rag_module, "maintenance", new=_explode), \
         patch.object(rag_module, "patterns", new=_stub("patterns")), \
         patch.object(rag_module, "emergent", new=_stub("emergent")), \
         patch.object(rag_module, "morning", new=_stub("morning")), \
         patch.object(rag_module.ollama, "chat",
                      side_effect=lambda *a, **kw: call_order.append("ollama")), \
         patch.object(rag_module, "resolve_chat_model",
                      return_value="qwen2.5:7b"):
        result = runner.invoke(rag_module.cli, ["wake-up"])

    # Exit != 0 porque un step falló (launchd lo marca rojo).
    assert result.exit_code == 1
    # Todos los pasos siguientes a maintenance igual corrieron.
    assert call_order == ["index", "bookmarks", "wa_tasks", "maintenance-FAIL",
                          "patterns", "emergent", "morning", "ollama"]
    assert "Fallaron" in result.output
    assert "simulated failure" in result.output


def test_wake_up_ollama_warmup_uses_keep_alive_minus_one():
    """Warmup debe pasar keep_alive=-1 para que el modelo persista."""
    runner = CliRunner()
    seen_kwargs: dict = {}

    def _capture(*args, **kwargs):
        seen_kwargs.update(kwargs)

    rag_wa_tasks_mod = sys.modules["rag.wa_tasks"]

    with patch.object(rag_module, "index", new=lambda *a, **kw: None), \
         patch.object(rag_module, "bookmarks_sync", new=lambda *a, **kw: None), \
         patch.object(rag_wa_tasks_mod, "wa_tasks", new=lambda *a, **kw: None), \
         patch.object(rag_module, "maintenance", new=lambda *a, **kw: None), \
         patch.object(rag_module, "patterns", new=lambda *a, **kw: None), \
         patch.object(rag_module, "emergent", new=lambda *a, **kw: None), \
         patch.object(rag_module, "morning", new=lambda *a, **kw: None), \
         patch.object(rag_module.ollama, "chat", side_effect=_capture), \
         patch.object(rag_module, "resolve_chat_model",
                      return_value="qwen2.5:7b"):
        result = runner.invoke(rag_module.cli, ["wake-up"])

    assert result.exit_code == 0
    assert seen_kwargs.get("keep_alive") == -1
    assert seen_kwargs.get("model") == "qwen2.5:7b"


# ── Plist generator ─────────────────────────────────────────────────

def _parse(xml: str) -> dict:
    return plistlib.loads(xml.encode())


def test_wake_up_plist_label():
    d = _parse(rag_module._wake_up_plist(RAG_BIN))
    assert d["Label"] == "com.fer.obsidian-rag-wake-up"


def test_wake_up_plist_calls_rag_wake_up():
    d = _parse(rag_module._wake_up_plist(RAG_BIN))
    assert d["ProgramArguments"] == [RAG_BIN, "wake-up"]


def test_wake_up_plist_runs_at_0400_daily():
    """StartCalendarInterval dict (no array) = todos los días a la misma
    hora. Si alguien lo cambia a array per-weekday, se excluirían fines
    de semana y el user se pierde el brief sábado/domingo."""
    d = _parse(rag_module._wake_up_plist(RAG_BIN))
    sched = d["StartCalendarInterval"]
    assert isinstance(sched, dict), \
        f"esperaba dict (daily), obtuve {type(sched).__name__}"
    assert sched == {"Hour": 4, "Minute": 0}


def test_wake_up_plist_has_ollama_keep_alive_env():
    """OLLAMA_KEEP_ALIVE=20m mantiene el chat model warm entre el
    wake-up y el primer chat del user.

    Nota 2026-04-30: rolleamos `-1 → 20m` (commit `4f7e41f`). Con `-1`
    + `OLLAMA_MAX_LOADED_MODELS=3` los 3 modelos pinned forever
    saturaban unified memory de 36 GB → MPS swappeaba → tokens/s
    caían 10×. Con `20m` el secundario se descarga si idle, el
    principal (qwen2.5:7b chat) se mantiene warm porque siempre se
    usa. Mejora 4-10× end-to-end.
    """
    d = _parse(rag_module._wake_up_plist(RAG_BIN))
    env = d["EnvironmentVariables"]
    assert env.get("OLLAMA_KEEP_ALIVE") == "20m"


def test_wake_up_plist_logs_to_rag_log_dir():
    d = _parse(rag_module._wake_up_plist(RAG_BIN))
    assert d["StandardOutPath"].endswith("/wake-up.log")
    assert d["StandardErrorPath"].endswith("/wake-up.error.log")


# ── _services_spec wiring ───────────────────────────────────────────

def test_wake_up_in_services_spec():
    """`rag setup` instala wake-up via _services_spec — sino el user
    hace `rag setup` y nunca se crea el plist."""
    spec = rag_module._services_spec(RAG_BIN)
    labels = [label for label, _, _ in spec]
    assert "com.fer.obsidian-rag-wake-up" in labels


def test_wake_up_services_spec_entry_shape():
    """La entry en _services_spec tiene (label, filename, xml)."""
    spec = rag_module._services_spec(RAG_BIN)
    for label, fname, content in spec:
        if label == "com.fer.obsidian-rag-wake-up":
            assert fname == "com.fer.obsidian-rag-wake-up.plist"
            assert "<key>Label</key>" in content
            assert "wake-up" in content
            return
    pytest.fail("no encontré wake-up en _services_spec")


# ── /api/status integration ─────────────────────────────────────────

def test_wake_up_appears_in_status_catalog():
    """/api/status debe incluir la entry de wake-up en briefs."""
    from fastapi.testclient import TestClient  # noqa: PLC0415
    import web.server as server_mod  # noqa: PLC0415

    client = TestClient(server_mod.app)
    resp = client.get("/api/status?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    briefs = next((c for c in d["categories"] if c["id"] == "briefs"), None)
    assert briefs is not None
    ids = [s["id"] for s in briefs["services"]]
    assert "com.fer.obsidian-rag-wake-up" in ids
