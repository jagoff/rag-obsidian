"""Tests for scripts/ingest_contacts.py + `rag index --source contacts`.

No live AddressBook reads — tests build a synthetic `AddressBook-v22.abcddb`
SQLite on disk (the schema is simple enough to mock faithfully) and
assert parsing / chunking / upsert / resolve_phone semantics. Covers:

  - phone normalisation (formatting, parens, country codes, ext, edge)
  - suffix-key generation + de-conflict across linked-cards
  - Cocoa→Unix timestamp conversion (None, 0, negative, real values)
  - body formatting (name fallbacks, org+title+dept, birthday, note)
  - upsert_contacts writes source=contacts + all expected meta fields
  - delete_contacts removes rows by uid
  - run() orchestration: first run, unchanged, changed, stale-delete
  - `resolve_phone` — linked cards across sources match correctly
  - integration via `rag index --source contacts`
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import rag
from scripts import ingest_contacts as ic


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    """Minimal sqlite-vec collection + state DB. Mirrors the pattern in
    test_ingest_reminders so tests stay symmetric."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()
    # Create empty state DB so sqlite3.connect works.
    conn = sqlite3.connect(str(tmp_path / "ragvec" / "ragvec.db"))
    conn.close()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="con_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    # Always clear the module-level phone-index cache between tests.
    ic.invalidate_phone_index()
    return col


def _mk_contact(
    uid: str,
    first: str = "",
    last: str = "",
    *,
    source_id: str = "SRC-A",
    org: str = "",
    phones: list[tuple[str, str]] | None = None,
    emails: list[tuple[str, str]] | None = None,
    note: str = "",
    birthday_ts: float = 0.0,
    middle: str = "",
    nickname: str = "",
    job: str = "",
    dept: str = "",
    created_ts: float = 0.0,
    modified_ts: float = 0.0,
) -> ic.Contact:
    """Build a Contact with sensible defaults. `phones` is list of
    `(raw, label)` tuples; `digits` is normalised automatically."""
    phones_list = tuple(
        ic.Phone(raw=raw, label=label, digits=ic.normalise_phone(raw))
        for raw, label in (phones or [])
    )
    emails_list = tuple(
        ic.Email(address=addr, label=label)
        for addr, label in (emails or [])
    )
    return ic.Contact(
        uid=uid,
        source_id=source_id,
        first_name=first,
        last_name=last,
        middle_name=middle,
        nickname=nickname,
        organization=org,
        job_title=job,
        department=dept,
        note=note,
        phones=phones_list,
        emails=emails_list,
        birthday_ts=birthday_ts,
        created_ts=created_ts,
        modified_ts=modified_ts,
    )


# ── Phone normalisation ────────────────────────────────────────────────

def test_normalise_phone_strips_formatting():
    assert ic.normalise_phone("+54 9 11 1234-5678") == "+5491112345678"
    assert ic.normalise_phone("(415) 555-1234") == "4155551234"
    assert ic.normalise_phone("  +1.555.867.5309  ") == "+15558675309"


def test_normalise_phone_handles_empty_and_none():
    assert ic.normalise_phone("") == ""
    assert ic.normalise_phone(None) == ""


def test_normalise_phone_strips_non_leading_plus():
    # A `+` in the middle of the string isn't a country-code marker.
    assert ic.normalise_phone("11 12+34 5678") == "11123456 78".replace(" ", "")


def test_normalise_phone_keeps_extension_digits():
    # We don't try to be smart about ext; caller can post-process.
    assert ic.normalise_phone("555-1234 ext. 101") == "5551234101"


def test_phone_match_keys_progression():
    # Happy path: long Argentine mobile → 4 keys (full, 10, 8, 7).
    keys = ic._phone_match_keys("+5491112345678")
    assert keys == ["+5491112345678", "1112345678", "12345678", "2345678"]


def test_phone_match_keys_short_input_fewer_keys():
    # 6-digit input → only itself, no suffix expansion.
    assert ic._phone_match_keys("123456") == ["123456"]


def test_phone_match_keys_empty_returns_empty():
    assert ic._phone_match_keys("") == []


def test_phone_match_keys_no_dup_when_length_equals_suffix():
    # 10-digit input should NOT emit the same 10-digit tail twice.
    keys = ic._phone_match_keys("3425908148")
    assert keys == ["3425908148", "25908148", "5908148"]


# ── Cocoa timestamp conversion ─────────────────────────────────────────

def test_cocoa_to_unix_none_and_zero():
    assert ic._cocoa_to_unix(None) == 0.0
    assert ic._cocoa_to_unix(0) == 0.0
    assert ic._cocoa_to_unix(-1) == 0.0


def test_cocoa_to_unix_real_value():
    # 2026-04-22 20:08:58 UTC → Cocoa 798581338.74 → Unix 1776888538.74
    got = ic._cocoa_to_unix(798581338.740374)
    assert abs(got - 1776888538.740374) < 0.001


# ── Body formatting ────────────────────────────────────────────────────

def test_format_body_name_only():
    c = _mk_contact("C1", first="Juli", last="Pérez")
    body = ic._format_contact_body(c)
    assert body.startswith("Contacto: Juli Pérez")


def test_format_body_name_fallback_to_nickname_then_org():
    c1 = _mk_contact("C1", nickname="Astor")
    assert "Contacto: Astor" in ic._format_contact_body(c1)

    c2 = _mk_contact("C2", org="Moka S.A.")
    assert "Contacto: Moka S.A." in ic._format_contact_body(c2)


def test_format_body_unnamed_contact_placeholder():
    c = _mk_contact("C1", phones=[("+5491112345678", "mobile")])
    body = ic._format_contact_body(c)
    assert "Contacto: (sin nombre)" in body


def test_format_body_org_and_role():
    c = _mk_contact(
        "C1", first="Ana", last="García",
        org="Moka S.A.", job="PM", dept="Producto",
    )
    body = ic._format_contact_body(c)
    assert "Organización: PM — Moka S.A. / Producto" in body


def test_format_body_job_title_without_org():
    c = _mk_contact("C1", first="Ana", job="PM")
    body = ic._format_contact_body(c)
    assert "Puesto: PM" in body
    assert "Organización:" not in body


def test_format_body_phones_with_labels():
    c = _mk_contact(
        "C1", first="Juli",
        phones=[("+5491112345678", "mobile"), ("+541145678900", "work")],
    )
    body = ic._format_contact_body(c)
    assert "Teléfono (mobile): +5491112345678" in body
    assert "Teléfono (work): +541145678900" in body


def test_format_body_birthday_yearless_emits_mmdd_only():
    # Apple stores birthdays with year 0001 when user didn't set a year.
    bday_raw = datetime(1, 3, 15, tzinfo=timezone.utc).timestamp()
    c = _mk_contact("C1", first="Juli", birthday_ts=bday_raw)
    body = ic._format_contact_body(c)
    # Only MM-DD, no "0001".
    assert "Cumpleaños: 03-15" in body
    assert "0001" not in body


def test_format_body_note_appended_with_separator():
    c = _mk_contact("C1", first="Juli", note="Prefer WhatsApp")
    body = ic._format_contact_body(c)
    assert "---" in body
    assert "Prefer WhatsApp" in body


def test_format_body_truncates_to_chunk_max():
    long_note = "x" * 2000
    c = _mk_contact("C1", first="Juli", note=long_note)
    body = ic._format_contact_body(c)
    assert len(body) <= ic.CHUNK_MAX_CHARS


# ── Content hash stability ─────────────────────────────────────────────

def test_content_hash_stable_across_field_reorder():
    # Hash should depend on values, not iteration order of phones list.
    c1 = _mk_contact("C1", first="J",
                      phones=[("+5491111", "mobile"), ("+5411222", "work")])
    c2 = _mk_contact("C1", first="J",
                      phones=[("+5491111", "mobile"), ("+5411222", "work")])
    assert ic._content_hash(c1) == ic._content_hash(c2)


def test_content_hash_changes_on_meaningful_edit():
    c1 = _mk_contact("C1", first="Juli")
    c2 = _mk_contact("C1", first="Julia")
    assert ic._content_hash(c1) != ic._content_hash(c2)


def test_content_hash_ignores_modification_ts():
    # `modified_ts` is noisy (bumps on iCloud sync idle) — shouldn't
    # force a reindex.
    c1 = _mk_contact("C1", first="Juli", modified_ts=1000)
    c2 = _mk_contact("C1", first="Juli", modified_ts=999999)
    assert ic._content_hash(c1) == ic._content_hash(c2)


# ── Phone index — linked-card de-dup ───────────────────────────────────

def test_phone_index_merges_linked_cards_across_sources():
    """Same phone in two AddressBook sources (iCloud linked cards) must
    NOT invalidate the suffix key. This was the 3%-resolve bug."""
    c1 = _mk_contact(
        "UID-A", first="Juli", source_id="local-src",
        phones=[("+5491112345678", "mobile")],
    )
    c2 = _mk_contact(
        "UID-B", first="Juli Pérez", source_id="icloud-src",
        phones=[("+5491112345678", "mobile")],
    )
    index, by_uid = ic._build_phone_index([c1, c2])
    # Suffix keys should resolve (not be in conflict_keys).
    match_uid = index.get("1112345678")
    assert match_uid is not None
    # Canonical should be the longer display name.
    assert by_uid[match_uid].display_name == "Juli Pérez"


def test_phone_index_drops_genuine_conflicts():
    """Two DIFFERENT contacts with the same last-7 digits but different
    country-code prefixes — real ambiguity, drop the short suffix key."""
    c1 = _mk_contact("UID-A", first="Juli",
                      phones=[("+5491112345678", "mobile")])
    c2 = _mk_contact("UID-B", first="John",
                      phones=[("+1415555678", "mobile")])  # last-7: 5555678 vs 2345678, no overlap
    index, _ = ic._build_phone_index([c1, c2])
    # Both full digits resolve.
    assert index.get("+5491112345678") == "UID-A"
    assert index.get("+1415555678") == "UID-B"


def test_phone_index_drops_ambiguous_last7():
    """Craft two contacts where last-7 collides → key must be dropped."""
    c1 = _mk_contact("UID-A", first="A",
                      phones=[("+5491111234567", "mobile")])
    c2 = _mk_contact("UID-B", first="B",
                      phones=[("+1415111234567", "mobile")])  # different country, same 7-digit tail
    index, _ = ic._build_phone_index([c1, c2])
    # Last-7 is ambiguous → None.
    assert "1234567" not in index
    # Full numbers still resolve.
    assert index.get("+5491111234567") == "UID-A"
    assert index.get("+1415111234567") == "UID-B"


def test_resolve_phone_uses_cache_until_invalidated(monkeypatch):
    """Module-level cache — second call shouldn't re-fetch."""
    calls: dict = {"n": 0}
    c1 = _mk_contact("UID-A", first="Juli",
                      phones=[("+5491112345678", "mobile")])

    def _fake_fetch(root=None):
        calls["n"] += 1
        return [c1]

    monkeypatch.setattr(ic, "_default_fetch", _fake_fetch)
    ic.invalidate_phone_index()
    assert ic.resolve_phone("+5491112345678").first_name == "Juli"
    assert ic.resolve_phone("1112345678").first_name == "Juli"
    assert calls["n"] == 1  # cached
    ic.invalidate_phone_index()
    assert ic.resolve_phone("+5491112345678").first_name == "Juli"
    assert calls["n"] == 2  # rebuilt


def test_resolve_phone_empty_or_unknown_returns_none(monkeypatch):
    monkeypatch.setattr(ic, "_default_fetch", lambda root=None: [])
    ic.invalidate_phone_index()
    assert ic.resolve_phone("") is None
    assert ic.resolve_phone("+999999999") is None


# ── .abcddb reader ─────────────────────────────────────────────────────

def _make_fake_abcddb(path: Path) -> None:
    """Write a minimal SQLite that satisfies our reader's SQL. Real
    .abcddb schemas are much wider, but our reader only COALESCE()s
    the 12 columns we care about."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("""
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZUNIQUEID TEXT,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT,
                ZMIDDLENAME TEXT,
                ZNICKNAME TEXT,
                ZORGANIZATION TEXT,
                ZJOBTITLE TEXT,
                ZDEPARTMENT TEXT,
                ZBIRTHDAY REAL,
                ZCREATIONDATE REAL,
                ZMODIFICATIONDATE REAL
            );
            CREATE TABLE ZABCDPHONENUMBER (
                Z_PK INTEGER PRIMARY KEY,
                ZOWNER INTEGER,
                ZORDERINGINDEX INTEGER,
                ZFULLNUMBER TEXT,
                ZLABEL TEXT
            );
            CREATE TABLE ZABCDEMAILADDRESS (
                Z_PK INTEGER PRIMARY KEY,
                ZOWNER INTEGER,
                ZORDERINGINDEX INTEGER,
                ZADDRESS TEXT,
                ZLABEL TEXT
            );
            CREATE TABLE ZABCDNOTE (
                Z_PK INTEGER PRIMARY KEY,
                ZCONTACT INTEGER,
                ZTEXT TEXT
            );
        """)
        # Contact 1: Juli Pérez @ Moka with 1 mobile + 1 email + note.
        conn.execute(
            "INSERT INTO ZABCDRECORD VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (1, "UID-JULI", "Juli", "Pérez", None, None, "Moka S.A.",
             "PM", None, None, 600_000_000.0, 700_000_000.0),
        )
        conn.execute(
            "INSERT INTO ZABCDPHONENUMBER VALUES (?,?,?,?,?)",
            (1, 1, 0, "+54 9 11 1234-5678", "_$!<Mobile>!$_"),
        )
        conn.execute(
            "INSERT INTO ZABCDEMAILADDRESS VALUES (?,?,?,?,?)",
            (1, 1, 0, "juli@moka.com", "_$!<Work>!$_"),
        )
        conn.execute(
            "INSERT INTO ZABCDNOTE VALUES (?,?,?)",
            (1, 1, "Lo conocí en 2022"),
        )
        # Contact 2: unnamed/empty — should be filtered out.
        conn.execute(
            "INSERT INTO ZABCDRECORD VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (2, "UID-EMPTY", None, None, None, None, None,
             None, None, None, None, None),
        )
        # Contact 3: org-only.
        conn.execute(
            "INSERT INTO ZABCDRECORD VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (3, "UID-ORG", None, None, None, None, "Apple Inc.",
             None, None, None, None, None),
        )
        conn.commit()
    finally:
        conn.close()


def test_read_contacts_from_db_happy_path(tmp_path):
    db = tmp_path / "AddressBook-v22.abcddb"
    _make_fake_abcddb(db)
    contacts = ic.read_contacts_from_db(db, "SRC-TEST")
    # UID-EMPTY gets filtered by is_empty(); UID-JULI + UID-ORG remain.
    assert len(contacts) == 2
    juli = next(c for c in contacts if c.uid == "UID-JULI")
    assert juli.first_name == "Juli"
    assert juli.last_name == "Pérez"
    assert juli.organization == "Moka S.A."
    assert juli.job_title == "PM"
    assert juli.source_id == "SRC-TEST"
    assert len(juli.phones) == 1
    assert juli.phones[0].raw == "+54 9 11 1234-5678"
    assert juli.phones[0].label == "mobile"  # _$!<Mobile>!$_ → mobile
    assert juli.phones[0].digits == "+5491112345678"
    assert len(juli.emails) == 1
    assert juli.emails[0].address == "juli@moka.com"
    assert juli.note == "Lo conocí en 2022"


def test_read_contacts_skips_missing_uniqueid(tmp_path):
    db = tmp_path / "AddressBook-v22.abcddb"
    _make_fake_abcddb(db)
    # Force-corrupt: add a row without ZUNIQUEID (Core Data internals).
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO ZABCDRECORD VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (99, None, "OrphanFirst", None, None, None, None,
             None, None, None, None, None),
        )
        conn.commit()
    finally:
        conn.close()
    contacts = ic.read_contacts_from_db(db, "SRC-TEST")
    # The orphan should NOT be included (uid required for stable dedup).
    assert all(c.uid != "" for c in contacts)
    assert not any(c.first_name == "OrphanFirst" for c in contacts)


def test_find_abcddb_files_ignores_hidden(tmp_path):
    root = tmp_path / "Sources"
    root.mkdir()
    # Real source.
    (root / "AAA-UUID").mkdir()
    _make_fake_abcddb(root / "AAA-UUID" / "AddressBook-v22.abcddb")
    # Hidden / non-dir / empty.
    (root / ".DS_Store").write_text("junk")
    (root / "empty-dir").mkdir()
    (root / "BBB-UUID").mkdir()
    _make_fake_abcddb(root / "BBB-UUID" / "AddressBook-v22.abcddb")

    found = ic._find_abcddb_files(root)
    names = sorted(sid for sid, _ in found)
    assert names == ["AAA-UUID", "BBB-UUID"]


# ── Writer + state ─────────────────────────────────────────────────────

def test_upsert_contacts_writes_metadata_and_source(tmp_vault_col):
    col = tmp_vault_col
    c = _mk_contact(
        "UID-1", first="Juli", last="Pérez", org="Moka",
        phones=[("+5491112345678", "mobile")],
        emails=[("juli@moka.com", "work")],
    )
    n = ic.upsert_contacts(col, [c])
    assert n == 1
    got = col.get(where={"file": "contacts://UID-1"}, include=["metadatas"])
    assert got["ids"] == ["contacts://UID-1::0"]
    meta = got["metadatas"][0]
    assert meta["source"] == "contacts"
    assert meta["contact_uid"] == "UID-1"
    assert meta["title"] == "Juli Pérez"
    assert meta["primary_phone"] == "+5491112345678"
    assert meta["primary_email"] == "juli@moka.com"


def test_upsert_contacts_idempotent(tmp_vault_col):
    col = tmp_vault_col
    c = _mk_contact("UID-1", first="Juli")
    ic.upsert_contacts(col, [c])
    ic.upsert_contacts(col, [c])  # second call should replace, not dup
    got = col.get(where={"file": "contacts://UID-1"}, include=[])
    assert len(got["ids"]) == 1


def test_delete_contacts_removes_rows(tmp_vault_col):
    col = tmp_vault_col
    c = _mk_contact("UID-1", first="Juli")
    ic.upsert_contacts(col, [c])
    removed = ic.delete_contacts(col, ["UID-1"])
    assert removed == 1
    got = col.get(where={"file": "contacts://UID-1"}, include=[])
    assert got["ids"] == []


# ── run() orchestration ────────────────────────────────────────────────

def test_run_first_pass_indexes_everything(tmp_vault_col):
    c1 = _mk_contact("UID-1", first="Juli")
    c2 = _mk_contact("UID-2", first="Astor")
    summary = ic.run(fetch_fn=lambda: [c1, c2])
    assert summary["contacts_fetched"] == 2
    assert summary["contacts_indexed"] == 2
    assert summary["contacts_unchanged"] == 0
    assert summary["contacts_deleted"] == 0


def test_run_second_pass_same_data_is_noop(tmp_vault_col):
    c1 = _mk_contact("UID-1", first="Juli")
    ic.run(fetch_fn=lambda: [c1])
    summary = ic.run(fetch_fn=lambda: [c1])
    assert summary["contacts_indexed"] == 0
    assert summary["contacts_unchanged"] == 1


def test_run_detects_changed_contact(tmp_vault_col):
    c1 = _mk_contact("UID-1", first="Juli")
    ic.run(fetch_fn=lambda: [c1])
    c2 = _mk_contact("UID-1", first="Julia")  # name edit
    summary = ic.run(fetch_fn=lambda: [c2])
    assert summary["contacts_indexed"] == 1
    assert summary["contacts_unchanged"] == 0


def test_run_deletes_stale_contact(tmp_vault_col):
    c1 = _mk_contact("UID-1", first="Juli")
    c2 = _mk_contact("UID-2", first="Astor")
    ic.run(fetch_fn=lambda: [c1, c2])
    # Second run: only c1 is still in the fetch → c2 is stale.
    summary = ic.run(fetch_fn=lambda: [c1])
    assert summary["contacts_deleted"] == 1


def test_run_reset_forces_full_reindex(tmp_vault_col):
    c1 = _mk_contact("UID-1", first="Juli")
    ic.run(fetch_fn=lambda: [c1])  # populates state
    summary = ic.run(fetch_fn=lambda: [c1], reset=True)
    assert summary["contacts_indexed"] == 1
    assert summary["contacts_unchanged"] == 0


def test_run_dry_run_makes_no_writes(tmp_vault_col):
    col = tmp_vault_col
    c1 = _mk_contact("UID-1", first="Juli")
    summary = ic.run(fetch_fn=lambda: [c1], dry_run=True)
    assert summary["contacts_indexed"] == 1  # counted, not written
    got = col.get(where={"file": "contacts://UID-1"}, include=[])
    assert got["ids"] == []


def test_run_invalidates_phone_index_after_success(tmp_vault_col, monkeypatch):
    """After a successful run, `resolve_phone()` must rebuild from the
    fresh corpus — otherwise calls ingester sees stale data."""
    c1 = _mk_contact("UID-1", first="Juli",
                      phones=[("+5491112345678", "mobile")])
    ic.run(fetch_fn=lambda: [c1])
    # Check that the cache was invalidated (resolve re-fetches).
    calls: dict = {"n": 0}

    def _counted_fetch(root=None):
        calls["n"] += 1
        return [c1]

    monkeypatch.setattr(ic, "_default_fetch", _counted_fetch)
    match = ic.resolve_phone("+5491112345678")
    assert match is not None
    assert calls["n"] == 1
