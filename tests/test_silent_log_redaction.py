"""Tests para `_silent_log` token redaction (L-1 fix, 2026-05-08).

Las APIs de Google (`google-auth`, `googleapiclient`) embeben fragmentos
de token / refresh_token / client_secret en los messages de excepciones
como `RefreshError("invalid_grant: refresh_token=abc123def456")`. Sin
redact, esos tokens crudos terminan persistidos en
`~/.local/share/obsidian-rag/silent_errors.jsonl` aunque el archivo
tenga chmod 0o600.

`_silent_log` corre `_TOKEN_REDACT_RE.sub(...)` sobre la línea
serializada cuando el `where` empieza con `gmail_` / `drive_` /
`calendar_` (gate por prefix para no pagar el regex en hot paths).

Estos tests verifican:
1. Tokens largos en exc message → redactados.
2. Otros prefixes (no-google) → NO se redacta (perf gate).
3. Strings inofensivos tipo `"token missing"` → no matchean.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import rag


@pytest.fixture(autouse=True)
def _redirect_silent_log(tmp_path, monkeypatch):
    """Cada test escribe el silent_errors.jsonl en tmp_path para no
    contaminar el path real del user."""
    fake_path = tmp_path / "silent_errors.jsonl"
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", fake_path)
    yield fake_path


def _drain_log(path: Path, timeout_s: float = 2.0) -> list[dict]:
    """Espera a que el writer thread escriba la línea y devuelve los
    records parseados."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            break
        time.sleep(0.05)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_redacts_refresh_token_for_gmail_prefix(_redirect_silent_log):
    """L-1: refresh_token en exc string → redactado para where=gmail_*."""
    exc = RuntimeError("invalid_grant: refresh_token=abc123def456ghi")
    rag._silent_log("gmail_oauth_refresh", exc)

    records = _drain_log(_redirect_silent_log)
    assert len(records) >= 1, "esperaba al menos 1 record en jsonl"
    rec = records[-1]
    assert rec["where"] == "gmail_oauth_refresh"
    # El frag crudo no debe aparecer.
    assert "abc123def456ghi" not in json.dumps(rec)
    # El placeholder REDACTED sí.
    assert "REDACTED" in rec["exc"]


def test_redacts_access_token_for_drive_prefix(_redirect_silent_log):
    """Drive API: access_token=<value> (sin quotes alrededor)."""
    exc = RuntimeError("failed: access_token=xyz789secretvalueLong")
    rag._silent_log("drive_list_files", exc)

    records = _drain_log(_redirect_silent_log)
    assert len(records) >= 1
    rec = records[-1]
    assert "xyz789secretvalueLong" not in json.dumps(rec)
    assert "REDACTED" in rec["exc"]


def test_redacts_client_secret_for_calendar_prefix(_redirect_silent_log):
    """Calendar API: client_secret=<value> (sin quotes alrededor)."""
    exc = RuntimeError("config error: client_secret=GOCSPX-abc123longSecret")
    rag._silent_log("calendar_init", exc)

    records = _drain_log(_redirect_silent_log)
    assert len(records) >= 1
    rec = records[-1]
    assert "GOCSPX-abc123longSecret" not in json.dumps(rec)
    assert "REDACTED" in rec["exc"]


def test_no_redact_for_non_google_prefix(_redirect_silent_log):
    """Performance gate: where='ranker_config_load' NO corre regex."""
    exc = RuntimeError("invalid_grant: refresh_token=abc123def456ghi")
    rag._silent_log("ranker_config_load", exc)

    records = _drain_log(_redirect_silent_log)
    assert len(records) >= 1
    rec = records[-1]
    # Si NO empezás con gmail/drive/calendar, el token queda crudo.
    assert "abc123def456ghi" in rec["exc"]


def test_short_token_value_not_matched(_redirect_silent_log):
    """`token: x` (single char) NO matchea (>=6 chars requerido)."""
    exc = RuntimeError("token: x")
    rag._silent_log("gmail_oauth_refresh", exc)

    records = _drain_log(_redirect_silent_log)
    assert len(records) >= 1
    rec = records[-1]
    # El value es 1 char → no matchea regex → queda como estaba.
    assert "REDACTED" not in rec["exc"]
    assert "token: x" in rec["exc"]
