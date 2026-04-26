"""Tests para cross-source privacy opt-out (audit 2026-04-25 R2-Cross-source #1).

Sin filtros, queries pueden devolver mails sensibles (banking, 2FA),
chats privados, etc. Este módulo implementa el filtro propuesto en
`docs/design-cross-source-corpus.md §5.3` que estaba sin implementar.

Schema esperado del YAML en `~/.local/share/obsidian-rag/cross-source.yaml`:

    gmail:
      exclude_labels: [banking, 2fa]
      exclude_senders: ["*@bank.com"]
    whatsapp:
      exclude_chats: ["+5491112345678@s.whatsapp.net"]
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


@pytest.fixture(autouse=True)
def _reset_filter_cache(monkeypatch, tmp_path):
    """Cada test arranca con DB_PATH apuntando a tmp_path y el cache
    de filters limpio para no contaminar entre tests."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_CROSS_SOURCE_FILTERS_CACHE", None)
    monkeypatch.setattr(rag, "_CROSS_SOURCE_FILTERS_MTIME", 0.0)
    yield


def _write_filters(tmp_path: Path, content: str) -> None:
    (tmp_path / "cross-source.yaml").write_text(content, encoding="utf-8")


def _mk_pair(meta: dict, score: float = 0.9):
    """Construye un pair con la metadata pedida (sin texto)."""
    cand = ("test content", meta, "test-id")
    return (cand, "test content", score)


# ── _load_cross_source_filters ──────────────────────────────────────────


def test_load_returns_empty_when_file_missing(tmp_path):
    """Sin archivo, devuelve {} → no filter."""
    assert rag._load_cross_source_filters() == {}


def test_load_returns_parsed_yaml(tmp_path):
    _write_filters(tmp_path, """
gmail:
  exclude_labels: [banking]
""")
    out = rag._load_cross_source_filters()
    assert "gmail" in out
    assert out["gmail"]["exclude_labels"] == ["banking"]


def test_load_returns_empty_on_malformed_yaml(tmp_path):
    """YAML roto → {} en vez de crashear (privacidad es nice-to-have,
    no debe romper retrieval)."""
    _write_filters(tmp_path, "this: is: malformed: ::")
    out = rag._load_cross_source_filters()
    assert out == {}


def test_load_caches_by_mtime(tmp_path, monkeypatch):
    """Reloads YAML solo si cambia mtime — no relee el archivo en cada
    query."""
    _write_filters(tmp_path, "gmail:\n  exclude_labels: [v1]")
    out1 = rag._load_cross_source_filters()
    assert out1["gmail"]["exclude_labels"] == ["v1"]

    # Sobreescribimos sin cambiar mtime → cache devuelve la versión vieja
    # No podemos forzar mtime sin file ops, así que verificamos que
    # 2 llamadas seguidas devuelven el MISMO objeto (cache hit).
    out2 = rag._load_cross_source_filters()
    assert out1 is out2


# ── _should_exclude_chunk ──────────────────────────────────────────────


def test_excludes_gmail_by_label():
    """Mail con label 'banking' es excluido si está en exclude_labels."""
    filters = {"gmail": {"exclude_labels": ["banking", "2fa"]}}
    meta = {"source": "gmail", "labels": ["banking", "important"]}
    assert rag._should_exclude_chunk(meta, filters) is True


def test_excludes_gmail_by_label_case_insensitive():
    filters = {"gmail": {"exclude_labels": ["BANKING"]}}
    meta = {"source": "gmail", "labels": ["banking"]}
    assert rag._should_exclude_chunk(meta, filters) is True


def test_keeps_gmail_without_excluded_labels():
    filters = {"gmail": {"exclude_labels": ["banking"]}}
    meta = {"source": "gmail", "labels": ["work", "personal"]}
    assert rag._should_exclude_chunk(meta, filters) is False


def test_excludes_gmail_by_sender_glob():
    filters = {"gmail": {"exclude_senders": ["*@bank.com"]}}
    meta = {"source": "gmail", "from": "noreply@bank.com"}
    assert rag._should_exclude_chunk(meta, filters) is True


def test_keeps_gmail_with_non_matching_sender():
    filters = {"gmail": {"exclude_senders": ["*@bank.com"]}}
    meta = {"source": "gmail", "from": "amigo@gmail.com"}
    assert rag._should_exclude_chunk(meta, filters) is False


def test_excludes_whatsapp_chat_by_jid():
    filters = {"whatsapp": {"exclude_chats": ["+54911@s.whatsapp.net"]}}
    meta = {"source": "whatsapp", "chat_jid": "+54911@s.whatsapp.net"}
    assert rag._should_exclude_chunk(meta, filters) is True


def test_excludes_calendar_by_name():
    filters = {"calendar": {"exclude_calendars": ["Privado"]}}
    meta = {"source": "calendar", "calendar": "Privado"}
    assert rag._should_exclude_chunk(meta, filters) is True


def test_keeps_chunks_when_no_filters_for_source():
    """Vault chunks no tienen filtros configurados → no se excluyen
    aunque otros sources sí los tengan."""
    filters = {"gmail": {"exclude_labels": ["banking"]}}
    meta = {"source": "vault", "file": "test.md"}
    assert rag._should_exclude_chunk(meta, filters) is False


def test_keeps_when_no_filters_at_all():
    """Sin yaml → todo pasa."""
    meta = {"source": "gmail", "labels": ["banking"]}
    assert rag._should_exclude_chunk(meta, {}) is False


# ── _filter_excluded_chunks ────────────────────────────────────────────


def test_filter_excluded_chunks_drops_banking_mail(tmp_path):
    """Integration: yaml con banking → mail con ese label se filtra."""
    _write_filters(tmp_path, """
gmail:
  exclude_labels: [banking]
""")
    pairs = [
        _mk_pair({"source": "gmail", "labels": ["banking"], "from": "x@bank.com"}, 0.95),
        _mk_pair({"source": "gmail", "labels": ["work"], "from": "amigo@gmail.com"}, 0.85),
        _mk_pair({"source": "vault", "file": "nota.md"}, 0.70),
    ]
    out = rag._filter_excluded_chunks(pairs)
    assert len(out) == 2
    sources = [p[0][1].get("labels") or p[0][1].get("file") for p in out]
    assert ["work"] in [out[0][0][1].get("labels"), out[1][0][1].get("labels")]


def test_filter_passthrough_when_no_yaml(tmp_path):
    """Sin yaml → todos los pairs pasan sin tocarse."""
    pairs = [
        _mk_pair({"source": "gmail", "labels": ["banking"]}, 0.9),
    ]
    out = rag._filter_excluded_chunks(pairs)
    assert len(out) == 1
