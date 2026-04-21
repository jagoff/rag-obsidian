"""Regression test for `_collect_scoped_tasks_evidence_multi`.

Bug caught in production (WhatsApp → rag query "qué tengo esta semana"):

    rag query falló — servicios fallaron: name '_sup' is not defined

Two callsites inside the per-vault loop used `_sup(Exception)` as exception
suppression shorthand, but `_sup` was never defined anywhere in the module —
the intended call was `contextlib.suppress(Exception)`. Since the per-vault
branch runs for every multi-vault tasks-mode query, the entire code path was
dead: any user asking WhatsApp/web chat for their pending items across both
personal + work vaults got back a generic "servicios fallaron" error instead
of the aggregated agenda.

This test exercises the branch with monkeypatched per-vault helpers that
raise — if `_sup` ever gets reintroduced, this catches it before shipping.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import rag


def _raiser(*_args, **_kwargs):
    raise RuntimeError("simulated vault-helper failure")


def test_multi_vault_tasks_evidence_handles_helper_failures() -> None:
    """Per-vault helpers raising must be swallowed silently — the function
    returns partial evidence, never a NameError or unhandled exception."""
    with patch.object(rag, "_fetch_mail_unread", return_value=None), \
         patch.object(rag, "_fetch_reminders_due", return_value=None), \
         patch.object(rag, "_fetch_calendar_ahead", return_value=None), \
         patch.object(rag, "_fetch_whatsapp_unread", return_value=None), \
         patch.object(rag, "_fetch_weather_rain", return_value=None), \
         patch.object(rag, "_fetch_gmail_evidence", return_value=None), \
         patch.object(rag, "_pendientes_extract_loops_fast", side_effect=_raiser), \
         patch.object(rag, "_pendientes_recent_contradictions", side_effect=_raiser):
        ev = rag._collect_scoped_tasks_evidence_multi(
            [("home", Path("/tmp/__does_not_exist__"))],
            datetime.now(),
            {"reminders_horizon": 7, "calendar_ahead": 7},
        )

    # Structural contract: keys for the per-vault aggregates exist and are
    # empty lists (suppressed exception → nothing appended).
    assert ev["loops_stale"] == []
    assert ev["loops_activo"] == []
    assert ev["contradictions"] == []


def test_multi_vault_tasks_evidence_no_vaults() -> None:
    """Empty vault list = no per-vault iterations. System-service fetchers
    still run (and are all mocked to None here). Must not raise."""
    with patch.object(rag, "_fetch_mail_unread", return_value=None), \
         patch.object(rag, "_fetch_reminders_due", return_value=None), \
         patch.object(rag, "_fetch_calendar_ahead", return_value=None), \
         patch.object(rag, "_fetch_whatsapp_unread", return_value=None), \
         patch.object(rag, "_fetch_weather_rain", return_value=None), \
         patch.object(rag, "_fetch_gmail_evidence", return_value=None):
        ev = rag._collect_scoped_tasks_evidence_multi(
            [],
            datetime.now(),
            {"reminders_horizon": 7, "calendar_ahead": 7},
        )

    assert ev["loops_stale"] == []
    assert ev["loops_activo"] == []
    assert ev["contradictions"] == []
