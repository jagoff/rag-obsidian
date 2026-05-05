"""Defensive guards en `count()`, `id` y graph_expand.neighbor_fetch.

Repro de los bugs del 2026-05-01 (silent_errors.jsonl: 23 hits de
`graph_expand.outer` con tracebacks pegando en `count()`/`id`/`col.get`):

- `count()`: `fetchone()` puede devolver `None` o `()` mid-`delete_collection`
  → `[0]` rompe con TypeError 'NoneType' object is not subscriptable o
  IndexError 'tuple index out of range'.
- `id`: `_db.execute(...)` puede tirar `InterfaceError: bad parameter or
  other API misuse` cuando el shared sqlite3 connection está en estado
  corrupto por concurrent thread access.
- `col.get(where={"file": {"$in": [None, "x"]}})`: bindear None en un
  `IN (?, ?)` también produce InterfaceError.

Fixture sintético: NO depende de DB real, mockea `_db.execute` para
producir cada modo de fallo y verifica que el caller degrade graceful
en vez de propagar.
"""
from __future__ import annotations

import sqlite3

import pytest

import rag


# ── count() defensive ─────────────────────────────────────────────────────────


class _FakeCursorRow:
    """Fake cursor que devuelve un row controlable en `fetchone()`."""

    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row else []


class _FakeConn:
    """Fake sqlite3 connection — execute() devuelve un cursor configurable."""

    def __init__(self, *, count_row=None, raise_on_count=None, id_row=None,
                 raise_on_id=None):
        self._count_row = count_row
        self._raise_on_count = raise_on_count
        self._id_row = id_row
        self._raise_on_id = raise_on_id

    def execute(self, sql, params=()):
        if "SELECT COUNT(*)" in sql:
            if self._raise_on_count is not None:
                raise self._raise_on_count
            return _FakeCursorRow(self._count_row)
        if "rag_schema_version" in sql:
            if self._raise_on_id is not None:
                raise self._raise_on_id
            return _FakeCursorRow(self._id_row)
        return _FakeCursorRow(None)


def _mk_collection(conn) -> rag.SqliteVecCollection:
    """Construye una SqliteVecCollection sin invocar `_ensure_tables` (que
    requiere conn real). Bypasseamos __init__ via __new__."""
    col = rag.SqliteVecCollection.__new__(rag.SqliteVecCollection)
    col._db = conn
    col.name = "test_collection"
    col._dim = 1024
    col._vec = "vec_test_collection"
    col._meta = "meta_test_collection"
    return col


def test_count_fetchone_none_returns_zero():
    """fetchone() devolviendo None (mid drop/recreate) → count() devuelve 0,
    NO rompe con TypeError 'NoneType' object is not subscriptable."""
    col = _mk_collection(_FakeConn(count_row=None))
    assert col.count() == 0


def test_count_fetchone_empty_tuple_returns_zero():
    """fetchone() devolviendo () → count() devuelve 0, NO IndexError."""
    col = _mk_collection(_FakeConn(count_row=()))
    assert col.count() == 0


def test_count_interface_error_returns_zero():
    """InterfaceError 'bad parameter or other API misuse' (shared conn
    corrupta por concurrent thread access) → count() devuelve 0."""
    exc = sqlite3.InterfaceError("bad parameter or other API misuse")
    col = _mk_collection(_FakeConn(raise_on_count=exc))
    assert col.count() == 0


def test_count_operational_error_returns_zero():
    """OperationalError ('no such table') mid delete_collection →
    count() devuelve 0 (degrade graceful)."""
    exc = sqlite3.OperationalError("no such table: meta_test_collection")
    col = _mk_collection(_FakeConn(raise_on_count=exc))
    assert col.count() == 0


def test_count_happy_path_returns_int():
    """Sanity: count() normal devuelve el integer del row."""
    col = _mk_collection(_FakeConn(count_row=(42,)))
    assert col.count() == 42


# ── id property defensive ─────────────────────────────────────────────────────


def test_id_fetchone_none_returns_zero_version():
    """fetchone() devolviendo None → id devuelve `<name>:0`, no rompe."""
    col = _mk_collection(_FakeConn(id_row=None))
    assert col.id == "test_collection:0"


def test_id_interface_error_returns_zero_version():
    """InterfaceError 'bad parameter or other API misuse' → id devuelve
    `<name>:0` (silent_errors.jsonl 2026-05-01 traceback exact match)."""
    exc = sqlite3.InterfaceError("bad parameter or other API misuse")
    col = _mk_collection(_FakeConn(raise_on_id=exc))
    assert col.id == "test_collection:0"


def test_id_operational_error_returns_zero_version():
    """OperationalError (table missing mid drop) → id devuelve `<name>:0`."""
    exc = sqlite3.OperationalError("no such table: rag_schema_version")
    col = _mk_collection(_FakeConn(raise_on_id=exc))
    assert col.id == "test_collection:0"


def test_id_empty_tuple_returns_zero_version():
    """fetchone() devolviendo () → id devuelve `<name>:0`, no IndexError."""
    col = _mk_collection(_FakeConn(id_row=()))
    assert col.id == "test_collection:0"


def test_id_happy_path_returns_versioned():
    """Sanity: id devuelve `<name>:<version>` cuando el row está bien."""
    col = _mk_collection(_FakeConn(id_row=(7,)))
    assert col.id == "test_collection:7"


# ── _build_where: $in con valores válidos ─────────────────────────────────────


def test_build_where_in_strips_to_valid_strings():
    """$in con strings válidos genera un IN (?, ?, ...) clean."""
    sql, params = rag.SqliteVecCollection._build_where(
        {"file": {"$in": ["a.md", "b.md"]}}
    )
    assert " IN (" in sql
    assert params == ["a.md", "b.md"]


def test_build_where_empty_in_short_circuits():
    """$in con lista vacía genera `0 = 1` (nunca matchea), no rompe."""
    sql, params = rag.SqliteVecCollection._build_where(
        {"file": {"$in": []}}
    )
    assert "0 = 1" in sql
    assert params == []
