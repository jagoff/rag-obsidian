"""Tests del helper `is_user_query()` + filtro en cache_stats/health panel.

Auditoría 2026-04-22 sobre `~/.local/share/obsidian-rag/ragvec/telemetry.db`:

  === Distribución empty queries por cmd ===
  cmd       | n
  ----------|-----
  followup  | 375    ← `rag followup` — reporte de cabos sueltos
  read      | 256    ← `rag read <url>` — ingesta de artículos
                       ────
                       611

Ambos comandos usan `log_query_event()` por conveniencia (reusan el
writer) pero NO son queries del usuario — son jobs/ingesters que
loggean sus counts/métricas en `extra_json` con `q=""`. Eso distorsiona
todas las métricas agregadas: "cache hit rate 0%" era 0/1056 donde 611
eran jobs, así que el denominador real de queries-de-usuario era 445.

Contrato:

  is_user_query(cmd, q) → bool

  True  si el row representa una query interactiva del usuario
        (query, chat, web, serve.chat, serve.received, do, links, prep).
  False si es un job/ingester/report (followup, read, capture, inbox,
        morning, today, digest, archive, index, watch, wa-tasks, …).

Política: **allowlist explícita** (no-deny), porque el set de comandos
crece y cualquier cmd nuevo debe considerarse NO-query por default
hasta que lo agreguemos al allowlist. Eso evita que un nuevo cmd de
job contamine métricas silenciosamente.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import rag


# ── Allowlist: user-facing query cmds ────────────────────────────────────────


@pytest.mark.parametrize("cmd", [
    "query",          # CLI one-shot
    "chat",           # CLI interactive
    "web",            # web /api/chat
    "web.chat.low_conf_bypass",
    "web.chat.metachat",
    "serve.chat",     # rag serve HTTP
    "serve.received", # idem
    "serve.tasks",    # /query short-circuit → tasks/agenda
    "serve.error",    # /query|/chat raised exception
    "serve.metachat", # /query short-circuit → metachat (greetings)
    "serve.weather_fallback",  # weather fallback al RAG path
    "serve.finance",  # /query short-circuit → finance/cards
    "do",             # rag do <instruccion>
    "links",          # rag links
    "prep",           # rag prep <tema>
])
def test_is_user_query_allowlist(cmd):
    """Todos los cmds user-facing deben retornar True aun con q vacío —
    el q es irrelevante para la decisión, la decisión es por cmd."""
    assert rag.is_user_query(cmd, "que es ikigai") is True
    # q vacío no cambia el resultado para user cmds (algunos como
    # chat con slash commands pueden tener q="").
    assert rag.is_user_query(cmd, "") is True


# ── Job/ingester cmds ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cmd", [
    "followup",
    "read",
    "capture",
    "inbox",
    "morning",
    "today",
    "digest",
    "archive",
    "index",
    "watch",
    "wa-tasks",
    "web.tasks",
    "test",
])
def test_is_user_query_denies_jobs(cmd):
    assert rag.is_user_query(cmd, "") is False
    # Incluso con q poblado (raro pero posible), si el cmd es job, False.
    assert rag.is_user_query(cmd, "something") is False


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_is_user_query_unknown_cmd_defaults_false():
    """Política conservative: cmds desconocidos NO cuentan como user query.
    Así un cmd nuevo no contamina métricas silenciosamente hasta que lo
    agreguemos al allowlist."""
    assert rag.is_user_query("future_unknown_cmd", "q") is False


def test_is_user_query_none_cmd_is_false():
    """`cmd=None` (row corrupto o legacy) → False, defensivo."""
    assert rag.is_user_query(None, "q") is False


def test_is_user_query_accepts_empty_strings():
    """No debe raisear con empty cmd + empty q."""
    assert rag.is_user_query("", "") is False


# ── Consistency: el número de items en cada set ──────────────────────────────


def test_user_query_cmds_set_exists():
    """Hay un set módulo-level con los cmds allowlisted, para facilitar
    auditoría rápida y monkey-patch en tests futuros."""
    assert hasattr(rag, "_USER_QUERY_CMDS")
    assert isinstance(rag._USER_QUERY_CMDS, (frozenset, set))
    # Sanity: al menos los core deben estar.
    for core in ("query", "chat", "web"):
        assert core in rag._USER_QUERY_CMDS
