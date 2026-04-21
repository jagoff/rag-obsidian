"""Tests for email + phone enrichment in _load_mentions_index.

The dossier index used to only track filename stems + frontmatter aliases.
After 2026-04-21, body-level `- **Email**: x@y.com` and `- **Teléfono**: +54...`
bullets are also indexed so queries with email or phone numbers resolve to
the right person for build_person_context.

Scope: temp-vault fixtures only. No live Apple Contacts / WA bridge / LLM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


MENTIONS_SUBPATH = rag._MENTIONS_FOLDER  # "04-Archive/99-obsidian-system/99-Mentions"


@pytest.fixture
def vault_with_mentions(tmp_path: Path, monkeypatch) -> Path:
    """Fresh vault with a couple of dossiers that exercise the parser.

    Mandatory: clear the module-level cache before and after so tests don't
    bleed into each other (the cache key is vault root + max-mtime).
    """
    root = tmp_path / "vault"
    mentions = root / MENTIONS_SUBPATH
    mentions.mkdir(parents=True)

    # Mama-like dossier: bullet list with bold labels (real format).
    (mentions / "Mama.md").write_text(
        "[[Mama|@Mama]]\n"
        "- **Relación**: Mamá\n"
        "- **Apellido / nombre completo**: Monica Ferrari\n"
        "- **Cumpleaños**: 09/02/1960\n"
        "- **Teléfono**: +54 9 3425476623\n"
        "- **Email**: monicaferrari@gmail.com\n"
        "- **Notas**: con frontmatter-less formatting (real-world)\n",
        encoding="utf-8",
    )
    # Alt format: non-bold labels, `*` bullets, email as `Correo:`.
    (mentions / "Juan.md").write_text(
        "---\n"
        "aliases: [Juan Pérez, juanp]\n"
        "---\n"
        "* Relación: colega\n"
        "* Correo: juan.perez@ejemplo.com\n"
        "* Cel: (011) 4567-8901\n",
        encoding="utf-8",
    )
    # Template file with `_` prefix — must be ignored.
    (mentions / "_template.md").write_text(
        "- **Email**: template@example.com\n",
        encoding="utf-8",
    )
    # Short-phone dossier: 7 digits → too short, should NOT be indexed.
    (mentions / "Short.md").write_text(
        "- **Teléfono**: 123-4567\n",  # 7 digits
        encoding="utf-8",
    )

    # Reset module cache + point VAULT_PATH at this tmp vault.
    rag._mentions_cache = None
    monkeypatch.setattr(rag, "VAULT_PATH", root)
    yield root
    rag._mentions_cache = None


# ── Index build ─────────────────────────────────────────────────────────────

def test_index_includes_filename_stems(vault_with_mentions: Path) -> None:
    idx = rag._load_mentions_index(vault_with_mentions)
    stems = {v for v in idx.values() if v.endswith((".md",))}
    assert any(v.endswith("Mama.md") for v in stems)
    assert any(v.endswith("Juan.md") for v in stems)


def test_index_includes_body_emails(vault_with_mentions: Path) -> None:
    idx = rag._load_mentions_index(vault_with_mentions)
    assert "monicaferrari@gmail.com" in idx
    assert idx["monicaferrari@gmail.com"].endswith("Mama.md")
    # Alt label: "Correo:" also captured
    assert "juan.perez@ejemplo.com" in idx
    assert idx["juan.perez@ejemplo.com"].endswith("Juan.md")


def test_index_includes_phone_digits(vault_with_mentions: Path) -> None:
    idx = rag._load_mentions_index(vault_with_mentions)
    # Full digit string
    assert "5493425476623" in idx
    # Last-8 suffix (AR number variance)
    assert "25476623" in idx
    assert idx["5493425476623"].endswith("Mama.md")
    assert idx["25476623"].endswith("Mama.md")
    # Juan's phone: (011) 4567-8901 → normalised to 01145678901 (11 digits)
    assert "01145678901" in idx
    assert idx["01145678901"].endswith("Juan.md")


def test_index_skips_underscore_template(vault_with_mentions: Path) -> None:
    idx = rag._load_mentions_index(vault_with_mentions)
    # Template email must NOT be in the index — _-prefixed files are ignored.
    assert "template@example.com" not in idx


def test_index_drops_short_phones(vault_with_mentions: Path) -> None:
    """<8 digit phones are too short — likely false positives. Must drop."""
    idx = rag._load_mentions_index(vault_with_mentions)
    # Short.md dossier has only a 7-digit phone → nothing from that file
    # should land in the index EXCEPT the stem.
    short_entries = [k for k, v in idx.items() if v.endswith("Short.md")]
    assert short_entries == ["short"], \
        f"short-phone should NOT be indexed, got: {short_entries}"


def test_index_is_cached(vault_with_mentions: Path) -> None:
    first = rag._load_mentions_index(vault_with_mentions)
    second = rag._load_mentions_index(vault_with_mentions)
    # Same dict object returned on second call — cache hit.
    assert first is second


# ── Query resolution ────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_stem", [
    # Names still match
    ("hablé con Mama ayer", "Mama.md"),
    ("Juan Pérez me pidió algo", "Juan.md"),
    # Emails resolve
    ("mensajes de monicaferrari@gmail.com", "Mama.md"),
    ("el mail juan.perez@ejemplo.com", "Juan.md"),
    # Phones with various shapes all resolve
    ("llamada al +54 9 3425476623", "Mama.md"),
    ("3425476623 me llamó", "Mama.md"),
    ("último mensaje a +549-3425-476623", "Mama.md"),
    # Last-8 suffix enough
    ("el número 25476623", "Mama.md"),
    # AR (011) format
    ("whatsapp a (011) 4567-8901", "Juan.md"),
])
def test_match_query_resolves_person(
    vault_with_mentions: Path, query: str, expected_stem: str,
) -> None:
    paths = rag._match_mentions_in_query(query, vault_with_mentions)
    assert paths, f"no match for query {query!r}"
    assert any(p.endswith(expected_stem) for p in paths), \
        f"expected {expected_stem} in {paths}"


@pytest.mark.parametrize("query", [
    # Dates — must NOT trip phone match (10+ digits but not indexed)
    "versión del 2026-04-21",
    "reunión el 15/04/2026",
    # Random number runs that don't match any dossier's phone
    "ruta tiene 150 chars",
    "commit d41bfe7",
    # Email that doesn't match any dossier
    "escribí a someone@nowhere.com",
    # Empty-ish
    "",
    "   ",
])
def test_match_query_no_false_positives(
    vault_with_mentions: Path, query: str,
) -> None:
    paths = rag._match_mentions_in_query(query, vault_with_mentions)
    assert paths == [], f"false positive for {query!r}: {paths}"


def test_match_query_respects_cap(
    vault_with_mentions: Path, monkeypatch,
) -> None:
    """_MENTIONS_MAX_PER_QUERY caps returned paths. Default is 2."""
    # Query hits both dossiers at once — should be capped at the default.
    paths = rag._match_mentions_in_query(
        "Mama y Juan coordinamos algo", vault_with_mentions,
    )
    assert len(paths) <= rag._MENTIONS_MAX_PER_QUERY
    assert len(paths) >= 1


# ── _normalise_phone_digits helper ──────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("+54 9 3425476623", "5493425476623"),
    ("(011) 4567-8901", "01145678901"),
    ("  +1 (555) 123-4567  ", "15551234567"),
    ("12345678", "12345678"),  # exactly 8 digits — keep
    ("1234567", ""),  # 7 digits — too short, drop
    ("no digits here", ""),
    ("", ""),
    (None, ""),  # defensive: None input
])
def test_normalise_phone_digits(raw, expected: str) -> None:
    assert rag._normalise_phone_digits(raw) == expected


# ── Centralized parsers (_parse_dossier_phones / _parse_dossier_emails) ────

def test_parse_dossier_phones_all_labels() -> None:
    """Todos los labels sinónimos de "phone" matchean el mismo regex
    ahora — antes, `- Cel: ...` entraba al mentions index pero NO al
    phone index."""
    body = (
        "- **Teléfono**: +54 9 1111 2222\n"
        "- **Tel**: +54 9 3333 4444\n"
        "- **Cel**: +54 9 5555 6666\n"
        "- **Celular**: +54 9 7777 8888\n"
        "- **Phone**: +1 555 999 0000\n"
        "- **Móvil**: +54 9 0000 1111\n"
        "- **Mobile**: +1 000 222 3333\n"
        "- **WhatsApp**: +54 9 2222 3333\n"
        "- **WA**: +54 9 4444 5555\n"
    )
    phones = rag._parse_dossier_phones(body)
    assert len(phones) == 9, f"expected 9 phones, got {len(phones)}: {phones}"
    # Each entry is digit-only
    assert all(p.isdigit() for p in phones)


def test_parse_dossier_phones_dedup() -> None:
    """Same phone listed under two labels returns once."""
    body = (
        "- **Teléfono**: +54 9 1111 2222\n"
        "- **Cel**: +549 1111 2222\n"  # same number, different formatting
    )
    phones = rag._parse_dossier_phones(body)
    assert len(phones) == 1
    assert phones[0] == "5491111222"[:-1] + "2222"[-4:] or phones[0] == "549" + "1111" + "2222"
    # More flexible: same digit-normalized form
    assert rag._normalise_phone_digits("+54 9 1111 2222") == phones[0]


def test_parse_dossier_phones_short_drop() -> None:
    """<8 digits gets filtered — floor for placeholder guards."""
    body = "- **Teléfono**: 123-4567\n"  # 7 digits
    assert rag._parse_dossier_phones(body) == []


def test_parse_dossier_phones_case_insensitive() -> None:
    body = "- **TELEFONO**: +54 9 1111 2222\n"
    assert rag._parse_dossier_phones(body) == [rag._normalise_phone_digits("+54 9 1111 2222")]


def test_parse_dossier_phones_bullet_variants() -> None:
    """`-` y `*` bullets ambos aceptados."""
    body = (
        "- **Teléfono**: +54 9 1111 2222\n"
        "* **Cel**: +54 9 3333 4444\n"
    )
    assert len(rag._parse_dossier_phones(body)) == 2


def test_parse_dossier_phones_bold_optional() -> None:
    """Bold `**` labels son opcionales."""
    body = (
        "- Teléfono: +54 9 1111 2222\n"
        "- **Cel**: +54 9 3333 4444\n"
    )
    assert len(rag._parse_dossier_phones(body)) == 2


def test_parse_dossier_phones_empty_body() -> None:
    assert rag._parse_dossier_phones("") == []
    assert rag._parse_dossier_phones(None) == []  # type: ignore[arg-type]


def test_parse_dossier_emails_all_labels() -> None:
    body = (
        "- **Email**: a@ex.com\n"
        "- **E-Mail**: b@ex.com\n"
        "- **Correo**: c@ex.com\n"
    )
    emails = rag._parse_dossier_emails(body)
    assert emails == ["a@ex.com", "b@ex.com", "c@ex.com"]


def test_parse_dossier_emails_lowercased() -> None:
    body = "- **Email**: MiXeD@EXAMPLE.COM\n"
    assert rag._parse_dossier_emails(body) == ["mixed@example.com"]


def test_parse_dossier_emails_dedup() -> None:
    body = (
        "- **Email**: same@ex.com\n"
        "- **Correo**: SAME@ex.com\n"
    )
    assert rag._parse_dossier_emails(body) == ["same@ex.com"]


def test_parse_dossier_emails_empty_body() -> None:
    assert rag._parse_dossier_emails("") == []
    assert rag._parse_dossier_emails(None) == []  # type: ignore[arg-type]
