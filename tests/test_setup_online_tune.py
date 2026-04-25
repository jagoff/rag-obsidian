"""Tests for the nightly online-tune launchd service and RAG_EXPLORE plists."""
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module

RAG_BIN = "/usr/local/bin/rag"

# `plutil` is macOS-only — tests that lint plist XML via `plutil -lint` skip
# on Linux CI (ubuntu-latest). Structural parsing via `plistlib` still runs
# cross-platform on the other plist tests, so the essential invariants
# (label, schedule, env vars) stay covered everywhere.
_HAS_PLUTIL = shutil.which("plutil") is not None
requires_plutil = pytest.mark.skipif(
    not _HAS_PLUTIL, reason="plutil is macOS-only; plist XML lint skipped on other platforms"
)


def _parse_plist(xml: str) -> dict:
    return plistlib.loads(xml.encode())


@requires_plutil
def test_online_tune_plist_valid_plist():
    xml = rag_module._online_tune_plist(RAG_BIN)
    result = subprocess.run(
        ["plutil", "-lint", "-"],
        input=xml.encode(),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_online_tune_plist_label():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    assert d["Label"] == "com.fer.obsidian-rag-online-tune"


def test_online_tune_plist_schedule_0330():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    cal = d["StartCalendarInterval"]
    assert cal["Hour"] == 3
    assert cal["Minute"] == 30
    assert "Weekday" not in cal  # runs every day, not day-restricted


def test_online_tune_plist_program_arguments():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert args[0] == RAG_BIN
    assert "tune" in args
    assert "--online" in args
    assert "--days" in args
    assert "14" in args
    assert "--apply" in args
    assert "--yes" in args


def test_online_tune_plist_no_rag_explore():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_online_tune_plist_one_shot():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    assert d.get("KeepAlive") is False
    assert d.get("RunAtLoad") is False


def test_morning_plist_has_rag_explore():
    d = _parse_plist(rag_module._morning_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert env.get("RAG_EXPLORE") == "1"


def test_today_plist_has_rag_explore():
    d = _parse_plist(rag_module._today_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert env.get("RAG_EXPLORE") == "1"


def test_watch_plist_no_rag_explore():
    d = _parse_plist(rag_module._watch_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_watch_plist_uses_all_vaults():
    """Regression (2026-04-22): pre-fix the plist invoked `rag watch` bare,
    which defaults to the active vault only — the non-active vault (`work`
    in the user's current 2-vault setup) silently went un-watched. The plist
    MUST pass `--all-vaults` so launchd covers every registered vault in a
    single process."""
    d = _parse_plist(rag_module._watch_plist(RAG_BIN))
    args = d.get("ProgramArguments", [])
    assert "watch" in args, f"expected `watch` subcommand, got {args}"
    assert "--all-vaults" in args, (
        f"watch plist must include --all-vaults to cover every registered "
        f"vault; got args={args}"
    )
    # Order invariant: --all-vaults must come AFTER the subcommand.
    assert args.index("--all-vaults") > args.index("watch")


@pytest.mark.parametrize("plist_fn,label,expected_interval", [
    (rag_module._ingest_whatsapp_plist, "whatsapp", 900),
    (rag_module._ingest_gmail_plist,    "gmail",    3600),
    (rag_module._ingest_calendar_plist, "calendar", 3600),
    (rag_module._ingest_reminders_plist, "reminders", 3600),
])
def test_ingester_plists_run_at_load_and_interval(plist_fn, label, expected_interval):
    """Regression (2026-04-22): reminders estaba en interval 6h (21600s) y
    los 4 ingesters tenían RunAtLoad=false. Post-reboot ninguno refresheaba
    inmediatamente, y reminders quedaba >17h stale en uso normal.

    Fix:
      - `RunAtLoad=true` en los 4 → corrida inmediata al cargar el service.
      - Reminders interval 21600 → 3600 (alineado con gmail/calendar).

    Este test guardspora los intervals exactos y la bandera RunAtLoad.
    """
    d = _parse_plist(plist_fn(RAG_BIN))
    assert d.get("RunAtLoad") is True, (
        f"ingest-{label} plist must have RunAtLoad=true so launchd "
        "runs it immediately at install/reboot. Otherwise the first "
        "refresh waits up to the full StartInterval which makes data "
        "stale in the window after boot."
    )
    assert d.get("StartInterval") == expected_interval, (
        f"ingest-{label} StartInterval drifted from {expected_interval}s; "
        "validate the trade-off before changing (reminders in particular "
        "is local-only so interval shrinks are cheap; gmail/calendar hit "
        "OAuth quotas so 1h is the floor)."
    )


def test_digest_plist_no_rag_explore():
    d = _parse_plist(rag_module._digest_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_emergent_plist_no_rag_explore():
    d = _parse_plist(rag_module._emergent_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_patterns_plist_no_rag_explore():
    d = _parse_plist(rag_module._patterns_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_archive_plist_no_rag_explore():
    d = _parse_plist(rag_module._archive_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_wa_tasks_plist_no_rag_explore():
    d = _parse_plist(rag_module._wa_tasks_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_services_spec_includes_online_tune():
    specs = rag_module._services_spec(RAG_BIN)
    labels = [s[0] for s in specs]
    assert "com.fer.obsidian-rag-online-tune" in labels


def test_services_spec_includes_serve():
    """rag serve is the hot path for WhatsApp — must ship with `rag setup`.

    Regression guard: the plist used to be hand-installed and got out of
    sync (corrupted, unregistered). Registering it in _services_spec() is
    the fix; this test prevents it from being accidentally removed again.
    """
    specs = rag_module._services_spec(RAG_BIN)
    labels = [s[0] for s in specs]
    assert "com.fer.obsidian-rag-serve" in labels


@requires_plutil
def test_serve_plist_valid_plist():
    xml = rag_module._serve_plist(RAG_BIN)
    result = subprocess.run(
        ["plutil", "-lint", "-"],
        input=xml.encode(),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_serve_plist_port_7832_and_keepalive():
    d = _parse_plist(rag_module._serve_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert args[0] == RAG_BIN
    assert args[1] == "serve"
    assert "7832" in args  # listener.ts hardcodes this port
    assert d["KeepAlive"] is True
    assert d["RunAtLoad"] is True


def test_serve_plist_warm_model_env():
    """Serve exists to keep models warm — without these env vars the
    whole point of the service evaporates (reranker unloads after 15min
    idle, bge-m3 pays HTTP round-trip, ollama drops the chat model)."""
    d = _parse_plist(rag_module._serve_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert env.get("OLLAMA_KEEP_ALIVE") == "-1"
    assert env.get("RAG_RERANKER_NEVER_UNLOAD") == "1"
    assert env.get("RAG_LOCAL_EMBED") == "1"


def test_services_spec_total_count():
    specs = rag_module._services_spec(RAG_BIN)
    # 18 base servicios + 4 ingesters cross-source
    # (WhatsApp/Gmail/Reminders/Calendar). Calendar se skipea al install
    # si ~/.calendar-mcp/credentials.json no existe (gate en `setup()`),
    # pero el plist siempre está en el spec.
    # Base roster: watch, serve, web (agregado 2026-04-22 — pre-fix estaba
    # instalado manualmente fuera de setup), digest, morning, today,
    # wake-up (agregado 2026-04-24 — orquestador nocturno 04:00),
    # emergent, patterns, archive, wa-tasks, reminder-wa-push (cron 5min
    # para disparar Apple Reminders via WhatsApp bridge antes de la due
    # — ver docstring en _services_spec), auto-harvest (2026-04-23 —
    # LLM-as-judge nocturno que labelea queries low-conf sin feedback),
    # online-tune, calibrate (2026-04-23 — per-source isotonic regression
    # re-entrenada con feedback), maintenance (2026-04-21 hardening),
    # consolidate, anticipate (2026-04-24 — game-changer push proactivo
    # cada 10 min: calendar proximity + temporal echo + stale commitment).
    assert len(specs) == 22


def test_services_spec_includes_maintenance():
    """Added 2026-04-21: daily `rag maintenance` at 04:00 compacts WAL +
    rotates telemetry logs + conditional VACUUM. Without it the WAL grew
    to 126 MB against a 206 MB main DB in production."""
    specs = rag_module._services_spec(RAG_BIN)
    labels = {s[0] for s in specs}
    assert "com.fer.obsidian-rag-maintenance" in labels
    # Plist content sanity: matches the 04:00 schedule + RunAtLoad=false.
    plist = rag_module._maintenance_plist(RAG_BIN)
    assert "<key>Hour</key><integer>4</integer>" in plist
    assert "<key>RunAtLoad</key><false/>" in plist
    assert "<string>maintenance</string>" in plist


def test_services_spec_includes_anticipate():
    """Added 2026-04-24 — game-changer: el RAG empuja info timely sin que
    preguntes. Cada 10 min evalúa 3 señales (calendar proximity / temporal
    echo / stale commitment) y empuja top-1 a WhatsApp vía proactive_push."""
    specs = rag_module._services_spec(RAG_BIN)
    labels = {s[0] for s in specs}
    assert "com.fer.obsidian-rag-anticipate" in labels
    plist = rag_module._anticipate_plist(RAG_BIN)
    assert "<key>StartInterval</key><integer>600</integer>" in plist  # 10 min
    assert "<key>RunAtLoad</key><false/>" in plist
    assert "<string>anticipate</string>" in plist
    assert "<string>run</string>" in plist


@requires_plutil
def test_anticipate_plist_valid_xml():
    xml = rag_module._anticipate_plist(RAG_BIN)
    result = subprocess.run(
        ["plutil", "-lint", "-"], input=xml.encode(), capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


# ── Cross-source ingesters (2026-04-21) ────────────────────────────────────

def test_services_spec_includes_ingesters():
    specs = rag_module._services_spec(RAG_BIN)
    labels = {s[0] for s in specs}
    assert "com.fer.obsidian-rag-ingest-whatsapp" in labels
    assert "com.fer.obsidian-rag-ingest-gmail" in labels
    assert "com.fer.obsidian-rag-ingest-reminders" in labels
    assert "com.fer.obsidian-rag-ingest-calendar" in labels


@pytest.mark.parametrize("fn_name,expected_source,expected_interval", [
    ("_ingest_whatsapp_plist", "whatsapp", 900),    # 15 min
    ("_ingest_gmail_plist", "gmail", 3600),         # 1 hora
    ("_ingest_reminders_plist", "reminders", 3600), # 1 hora (bajado desde 6h el 2026-04-22)
    ("_ingest_calendar_plist", "calendar", 3600),   # 1 hora
])
@requires_plutil
def test_ingester_plist_valid_plist(fn_name, expected_source, expected_interval):
    """Cada plist de ingester debe ser XML válido + parseable + apuntar al
    comando correcto + tener el interval esperado."""
    xml = getattr(rag_module, fn_name)(RAG_BIN)
    # 1. plutil lint — XML bien formado
    result = subprocess.run(
        ["plutil", "-lint", "-"], input=xml.encode(), capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()

    d = _parse_plist(xml)
    # 2. Label prefix correcto
    assert d["Label"] == f"com.fer.obsidian-rag-ingest-{expected_source}"

    # 3. ProgramArguments: <rag_bin> index --source <source>
    args = d["ProgramArguments"]
    assert args[0] == RAG_BIN
    assert "index" in args
    assert "--source" in args
    assert expected_source in args

    # 4. StartInterval esperado
    assert d["StartInterval"] == expected_interval

    # 5. RunAtLoad=True (2026-04-22 flip): los runs steady-state cuestan <30s
    #    (gmail/calendar api incremental via cursor, WA <1s con 0 nuevos,
    #    reminders 7s). El bootstrap full-scan (que motivó el false original)
    #    solo corre la primera vez que un ingester toca un corpus vacío — ya
    #    pasó en todos. Sin RunAtLoad, el primer refresh post-reboot / post-
    #    install del user demora hasta StartInterval (hasta 1h), dejando
    #    "qué tengo esta semana" con data stale.
    assert d["RunAtLoad"] is True

    # 6. OLLAMA_KEEP_ALIVE=-1 en env — el ingester emite un batch de embeds
    #    con bge-m3; sin esto, cada corrida paga cold-load.
    env = d.get("EnvironmentVariables", {})
    assert env.get("OLLAMA_KEEP_ALIVE") == "-1"


def test_ingester_plists_log_paths_distinct():
    """Cada ingester escribe a su propio stdout/stderr log para que el user
    pueda leer `ingest-whatsapp.log` sin mezclar con gmail/reminders."""
    paths = set()
    for fn_name in ["_ingest_whatsapp_plist", "_ingest_gmail_plist",
                    "_ingest_reminders_plist"]:
        d = _parse_plist(getattr(rag_module, fn_name)(RAG_BIN))
        paths.add(d["StandardOutPath"])
        paths.add(d["StandardErrorPath"])
    # 3 ingesters × 2 paths (out/err) = 6 distinct paths
    assert len(paths) == 6


def test_ingester_plists_no_rag_explore():
    """Los ingesters NO deben tener `RAG_EXPLORE=1` — ese flag pertenece a
    retrieval paths (morning/today), no al indexing. Si se cuela, el
    ingester podría sesgar qué chunks escribe basándose en randomness."""
    for fn_name in ["_ingest_whatsapp_plist", "_ingest_gmail_plist",
                    "_ingest_reminders_plist"]:
        d = _parse_plist(getattr(rag_module, fn_name)(RAG_BIN))
        env = d.get("EnvironmentVariables", {})
        assert "RAG_EXPLORE" not in env, (
            f"{fn_name} leaks RAG_EXPLORE — retrieval flag on an ingester"
        )


# ── Setup: Apple Contacts warmup (2026-04-21) ────────────────────────────────

def test_setup_warmup_runs_when_cache_missing(tmp_path, monkeypatch):
    """Setup (no --remove) trigger a pre-build del Contacts index cuando el
    disk cache no existe. Mockea el index loader para verificar la call sin
    hablar con osascript real."""
    import subprocess
    from click.testing import CliRunner

    # Redirect everything to tmp so we don't touch the user's real launchd / cache.
    monkeypatch.setattr(rag_module, "_LAUNCH_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(rag_module, "_RAG_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(rag_module, "_CONTACTS_PHONE_INDEX_PATH", tmp_path / "no-cache.json")
    monkeypatch.setattr(rag_module, "_rag_binary", lambda: RAG_BIN)
    # Fake rag binary that exists
    rag_bin_path = tmp_path / "rag-bin"
    rag_bin_path.write_text("#!/bin/sh\nexit 0\n")
    rag_bin_path.chmod(0o755)
    monkeypatch.setattr(rag_module, "_rag_binary", lambda: str(rag_bin_path))
    # Avoid real launchctl calls
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("P", (), {"returncode": 0, "stdout": b"", "stderr": b""})(),
    )

    called = {"n": 0}
    def _fake_loader(ttl_s=86400):
        called["n"] += 1
        return {"5491112345678": "Juan Pérez"}  # non-empty → warmup succeeded
    monkeypatch.setattr(rag_module, "_load_contacts_phone_index", _fake_loader)

    result = CliRunner().invoke(rag_module.setup, [])
    assert result.exit_code == 0, result.output
    assert called["n"] == 1, "warmup must be called exactly once"
    assert "contacts cache" in result.output
    assert "phones indexados" in result.output or "1 phones" in result.output


def test_setup_warmup_skipped_when_cache_fresh(tmp_path, monkeypatch):
    """Si el disk cache existe y está dentro del TTL, NO re-build (evita el
    dump de 85s en re-run diarios de `rag setup`)."""
    import subprocess
    from click.testing import CliRunner

    cache_path = tmp_path / "contacts.json"
    cache_path.write_text('{"ts": 0, "index": {"1234567890": "X"}}', encoding="utf-8")

    monkeypatch.setattr(rag_module, "_LAUNCH_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(rag_module, "_RAG_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(rag_module, "_CONTACTS_PHONE_INDEX_PATH", cache_path)
    rag_bin_path = tmp_path / "rag-bin"
    rag_bin_path.write_text("#!/bin/sh\nexit 0\n")
    rag_bin_path.chmod(0o755)
    monkeypatch.setattr(rag_module, "_rag_binary", lambda: str(rag_bin_path))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("P", (), {"returncode": 0, "stdout": b"", "stderr": b""})(),
    )

    called = {"n": 0}
    def _fake_loader(ttl_s=86400):
        called["n"] += 1
        return {}
    monkeypatch.setattr(rag_module, "_load_contacts_phone_index", _fake_loader)

    result = CliRunner().invoke(rag_module.setup, [])
    assert result.exit_code == 0, result.output
    assert called["n"] == 0, "fresh cache → no warmup needed"
    assert "phones indexados" not in result.output


def test_setup_warmup_silent_on_empty_contacts(tmp_path, monkeypatch):
    """Si el dump devuelve índice vacío (Contacts.app restringido o sin
    teléfonos), setup imprime un warning pero no aborta."""
    import subprocess
    from click.testing import CliRunner

    monkeypatch.setattr(rag_module, "_LAUNCH_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(rag_module, "_RAG_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(rag_module, "_CONTACTS_PHONE_INDEX_PATH", tmp_path / "no-cache.json")
    rag_bin_path = tmp_path / "rag-bin"
    rag_bin_path.write_text("#!/bin/sh\nexit 0\n")
    rag_bin_path.chmod(0o755)
    monkeypatch.setattr(rag_module, "_rag_binary", lambda: str(rag_bin_path))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("P", (), {"returncode": 0, "stdout": b"", "stderr": b""})(),
    )

    monkeypatch.setattr(rag_module, "_load_contacts_phone_index", lambda ttl_s=86400: {})

    result = CliRunner().invoke(rag_module.setup, [])
    assert result.exit_code == 0, result.output
    assert "contacts cache vacío" in result.output
    assert "mask fallback" in result.output


def test_setup_remove_skips_warmup(tmp_path, monkeypatch):
    """`rag setup --remove` NO debe warmear el cache (es destructivo)."""
    import subprocess
    from click.testing import CliRunner

    monkeypatch.setattr(rag_module, "_LAUNCH_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(rag_module, "_RAG_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(rag_module, "_CONTACTS_PHONE_INDEX_PATH", tmp_path / "no-cache.json")
    rag_bin_path = tmp_path / "rag-bin"
    rag_bin_path.write_text("#!/bin/sh\nexit 0\n")
    rag_bin_path.chmod(0o755)
    monkeypatch.setattr(rag_module, "_rag_binary", lambda: str(rag_bin_path))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("P", (), {"returncode": 0, "stdout": b"", "stderr": b""})(),
    )

    called = {"n": 0}
    monkeypatch.setattr(
        rag_module, "_load_contacts_phone_index",
        lambda ttl_s=86400: called.update({"n": called["n"] + 1}) or {},
    )

    result = CliRunner().invoke(rag_module.setup, ["--remove"])
    assert result.exit_code == 0, result.output
    assert called["n"] == 0, "--remove must not trigger warmup"
