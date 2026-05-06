"""Test memory pressure gate en anticipate-echo signal (2026-05-05 fix).

Cubre:
- _anticipate_signal_echo skips cuando memory > 70%
- _anticipate_signal_echo continúa cuando memory <= 70%
- Memory check failures no crashean el signal (silent-fail)
- Silent logging de memory_pressure y memory_check errors
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import rag
from rag import SqliteVecClient as _TestVecClient


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    col = client.get_or_create_collection(
        name="test_col", metadata={"hnsw:space": "cosine"},
    )
    with rag._ragvec_state_conn() as _conn:
        pass
    return col


@pytest.fixture
def vault_with_recent_notes(tmp_path):
    """Crea un vault con una nota reciente modificada hace <6h."""
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    note = vault / "recent.md"
    note.write_text("x" * 500)  # >500 chars para pasar min_chars filter
    return vault


def test_echo_signal_returns_empty_when_memory_over_70pct(
    state_db, vault_with_recent_notes, monkeypatch
):
    """Si memory > 70%, _anticipate_signal_echo retorna [] sin retrieve."""
    import rag.anticipatory as ant_mod

    # Mock memory a 75%
    monkeypatch.setattr(rag, "_system_memory_used_pct", Mock(return_value=75.0))

    # Mock vault path
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    # Mock retrieve NO debe ser llamado si memory > 70%
    retrieve_mock = Mock()
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # Debe retornar lista vacía
    assert result == []

    # retrieve debe NO ser llamado
    assert not retrieve_mock.called


def test_echo_signal_continues_when_memory_at_70pct_boundary(
    state_db, vault_with_recent_notes, monkeypatch
):
    """Si memory == 70%, debería continuar (threshold es >70%, no >=)."""
    import rag.anticipatory as ant_mod

    # Mock memory a exactamente 70%
    monkeypatch.setattr(rag, "_system_memory_used_pct", Mock(return_value=70.0))
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    # Mock retrieve para que retorne algo
    retrieve_mock = Mock(
        return_value={
            "scores": [0.5],
            "metas": [{"file": "old.md", "note": "old"}],
        }
    )
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # retrieve debería ser llamado (boundary es > no >=)
    assert retrieve_mock.called


def test_echo_signal_continues_when_memory_under_70pct(
    state_db, vault_with_recent_notes, monkeypatch
):
    """Si memory < 70%, _anticipate_signal_echo continúa normalmente."""
    import rag.anticipatory as ant_mod

    # Mock memory a 65%
    monkeypatch.setattr(rag, "_system_memory_used_pct", Mock(return_value=65.0))
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    # Mock retrieve para que retorne algo
    retrieve_mock = Mock(
        return_value={
            "scores": [0.5],
            "metas": [{"file": "old.md", "note": "old"}],
        }
    )
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # retrieve debería ser llamado
    assert retrieve_mock.called


def test_echo_signal_silent_fails_on_memory_check_exception(
    state_db, vault_with_recent_notes, monkeypatch
):
    """Si _system_memory_used_pct() falla, continuamos (silent-fail)."""
    import rag.anticipatory as ant_mod

    # Mock memory check a que lance excepción
    monkeypatch.setattr(
        rag,
        "_system_memory_used_pct",
        Mock(side_effect=Exception("vm_stat not found")),
    )
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    # Mock retrieve para que retorne algo
    retrieve_mock = Mock(
        return_value={
            "scores": [0.5],
            "metas": [{"file": "old.md", "note": "old"}],
        }
    )
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # retrieve debería ser llamado a pesar del error de memory check
    assert retrieve_mock.called


def test_echo_signal_returns_none_memory_on_non_darwin(
    state_db, vault_with_recent_notes, monkeypatch
):
    """_system_memory_used_pct retorna None en no-darwin, echo signal continúa."""
    import rag.anticipatory as ant_mod

    # Mock memory a None (como retorna en Linux)
    monkeypatch.setattr(rag, "_system_memory_used_pct", Mock(return_value=None))
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    # Mock retrieve
    retrieve_mock = Mock(
        return_value={
            "scores": [0.5],
            "metas": [{"file": "old.md", "note": "old"}],
        }
    )
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # retrieve debería ser llamado (None memory significa que no podemos medir, así que continuamos)
    assert retrieve_mock.called


def test_echo_signal_logs_memory_pressure_silent_error(
    state_db, vault_with_recent_notes, monkeypatch
):
    """Cuando memory > 70%, se loguea como silent_log."""
    import rag.anticipatory as ant_mod

    monkeypatch.setattr(rag, "_system_memory_used_pct", Mock(return_value=80.0))
    monkeypatch.setattr(rag, "_resolve_vault_path", Mock(return_value=vault_with_recent_notes))

    silent_log_mock = Mock()
    monkeypatch.setattr(rag, "_silent_log", silent_log_mock)

    now = datetime.now()
    result = ant_mod._anticipate_signal_echo(now)

    # _silent_log debe ser llamado con "anticipate_echo_memory_pressure"
    assert silent_log_mock.called
    call_args = [call[0] for call in silent_log_mock.call_args_list]
    assert any("anticipate_echo_memory_pressure" in str(arg) for arg in call_args)
