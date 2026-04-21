"""Tests for the nightly online-tune launchd service and RAG_EXPLORE plists."""
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module

RAG_BIN = "/usr/local/bin/rag"


def _parse_plist(xml: str) -> dict:
    return plistlib.loads(xml.encode())


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
    # 11 base servicios + 3 ingesters cross-source (WhatsApp/Gmail/Reminders).
    # Calendar no entra aún porque depende de OAuth config manual.
    assert len(specs) == 14


# ── Cross-source ingesters (2026-04-21) ────────────────────────────────────

def test_services_spec_includes_ingesters():
    specs = rag_module._services_spec(RAG_BIN)
    labels = {s[0] for s in specs}
    assert "com.fer.obsidian-rag-ingest-whatsapp" in labels
    assert "com.fer.obsidian-rag-ingest-gmail" in labels
    assert "com.fer.obsidian-rag-ingest-reminders" in labels


@pytest.mark.parametrize("fn_name,expected_source,expected_interval", [
    ("_ingest_whatsapp_plist", "whatsapp", 900),     # 15 min
    ("_ingest_gmail_plist", "gmail", 3600),          # 1 hora
    ("_ingest_reminders_plist", "reminders", 21600), # 6 horas
])
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

    # 5. RunAtLoad=False — no queremos que el primer install arranque un
    #    full-scan agresivo (el WA bootstrap tarda ~1min, Reminders 100s).
    #    Primer run espera a StartInterval.
    assert d["RunAtLoad"] is False

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
