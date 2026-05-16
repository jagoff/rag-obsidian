from __future__ import annotations

import time

import pytest

from web import server as server_mod


@pytest.fixture(autouse=True)
def _reset_followup_aging_state(monkeypatch):
    old = dict(server_mod._FOLLOWUP_AGING_CACHE)
    old_refreshing = server_mod._FOLLOWUP_AGING_REFRESHING
    monkeypatch.delenv("RAG_WEB_FOLLOWUP_AGING_COMPUTE", raising=False)
    server_mod._FOLLOWUP_AGING_CACHE["ts"] = 0.0
    server_mod._FOLLOWUP_AGING_CACHE["payload"] = None
    server_mod._FOLLOWUP_AGING_REFRESHING = False
    yield
    server_mod._FOLLOWUP_AGING_CACHE["ts"] = old["ts"]
    server_mod._FOLLOWUP_AGING_CACHE["payload"] = old["payload"]
    server_mod._FOLLOWUP_AGING_REFRESHING = old_refreshing


def test_followup_aging_web_compute_disabled_by_default():
    assert server_mod._followup_aging_web_compute_enabled() is False


def test_followup_aging_web_compute_opt_in(monkeypatch):
    monkeypatch.setenv("RAG_WEB_FOLLOWUP_AGING_COMPUTE", "1")
    assert server_mod._followup_aging_web_compute_enabled() is True


def test_fetch_followup_aging_cold_path_does_not_compute_by_default(monkeypatch):
    monkeypatch.setattr(
        server_mod,
        "_followup_aging_hydrate_from_disk_if_needed",
        lambda: False,
    )

    def _boom():
        raise AssertionError("followup aging compute should be opt-in")

    monkeypatch.setattr(server_mod, "_compute_followup_aging", _boom)

    assert server_mod._fetch_followup_aging() is None


def test_fetch_followup_aging_stale_cache_does_not_refresh_by_default(monkeypatch):
    payload = {"total": 1, "buckets": {"0_7": 1}, "sample": []}
    server_mod._FOLLOWUP_AGING_CACHE["ts"] = (
        time.time() - server_mod._FOLLOWUP_AGING_SOFT_TTL - 1
    )
    server_mod._FOLLOWUP_AGING_CACHE["payload"] = payload
    threads: list[dict] = []

    class _Thread:
        def __init__(self, *args, **kwargs):
            threads.append({"args": args, "kwargs": kwargs})

        def start(self):
            raise AssertionError("followup aging refresh should be opt-in")

    monkeypatch.setattr(server_mod.threading, "Thread", _Thread)

    assert server_mod._fetch_followup_aging() is payload
    assert threads == []


def test_fetch_followup_aging_cold_path_can_compute_when_opted_in(monkeypatch):
    payload = {"total": 2, "buckets": {"0_7": 2}, "sample": []}
    monkeypatch.setenv("RAG_WEB_FOLLOWUP_AGING_COMPUTE", "1")
    monkeypatch.setattr(
        server_mod,
        "_followup_aging_hydrate_from_disk_if_needed",
        lambda: False,
    )
    monkeypatch.setattr(server_mod, "_compute_followup_aging", lambda: payload)
    monkeypatch.setattr(
        server_mod,
        "_followup_aging_persist",
        lambda *args, **kwargs: None,
    )

    assert server_mod._fetch_followup_aging() == payload
    assert server_mod._FOLLOWUP_AGING_CACHE["payload"] == payload
