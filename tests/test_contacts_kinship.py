"""Tests para `_infer_kinship` + `_infer_short_name` + extensión del parser
de contact note (`_parse_vault_contact`).

Disparador: bug 2026-05-10 donde el draft de WhatsApp respondía a la hija
(Grecia) con tono frío de coworker. Causa raíz: el parser exponía solo
`relation_label` libre y el LLM no consistentemente inferia el registro
correcto. Fix: campo formal `kinship` (enum) + `short_name` (apodo
familiar) en el output del parser → inyectables al system prompt del
listener TS.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rag.integrations.whatsapp.contacts import (
    _infer_kinship,
    _infer_short_name,
    _parse_vault_contact,
)


# ── _infer_kinship ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "label, expected",
    [
        # family-immediate
        ("hija", "family-immediate"),
        ("Hijo", "family-immediate"),
        ("madre", "family-immediate"),
        ("padre", "family-immediate"),
        ("Mamá", "family-immediate"),
        ("mama", "family-immediate"),
        ("papá", "family-immediate"),
        ("hermana", "family-immediate"),
        ("hermano mayor", "family-immediate"),
        ("esposa", "family-immediate"),
        ("marido", "family-immediate"),
        # romantic-partner
        ("pareja", "romantic-partner"),
        ("novia", "romantic-partner"),
        ("compañera", "romantic-partner"),
        # family-extended
        ("tía", "family-extended"),
        ("mi tio Juan", "family-extended"),
        ("primo cercano", "family-extended"),
        ("abuela materna", "family-extended"),
        ("sobrina", "family-extended"),
        ("suegra", "family-extended"),
        # professional-formal
        ("psiquiatra", "professional-formal"),
        ("médica de cabecera", "professional-formal"),
        ("cliente VIP", "professional-formal"),
        ("paciente", "professional-formal"),
        ("abogado", "professional-formal"),
        # professional-close
        ("colega de la oficina", "professional-close"),
        ("ex jefe", "professional-close"),
        ("socia de proyecto", "professional-close"),
        # friend-close
        ("mejor amigo", "friend-close"),
        ("amiga cercana", "friend-close"),
        # friend-known
        ("amigo de la facu", "friend-known"),
        ("amiga del trabajo", "friend-known"),
        ("conocido del barrio", "friend-known"),
        # unknown (fallback)
        ("", "unknown"),
        ("algo random sin keyword", "unknown"),
        ("xyz123", "unknown"),
    ],
)
def test_infer_kinship(label: str, expected: str) -> None:
    assert _infer_kinship(label) == expected


def test_infer_kinship_priority_family_immediate_over_friend() -> None:
    # "hermano amigo" — hermano matchea PRIMERO porque family-immediate va
    # primero en el orden de chequeo.
    assert _infer_kinship("hermano amigo de la infancia") == "family-immediate"


# ── _infer_short_name ─────────────────────────────────────────────────────

def test_infer_short_name_explicit_wins() -> None:
    """El campo `Apodo` del body siempre tiene prioridad sobre todo."""
    assert _infer_short_name("Greci", ["Grecia"], "Grecia Ferrari", "Grecia") == "Greci"


def test_infer_short_name_falls_back_to_alias() -> None:
    """Sin explicit, primer alias no-template y no-redundante con full_name."""
    assert _infer_short_name("", ["Greci"], "Grecia Ferrari", "Grecia") == "Greci"


def test_infer_short_name_filters_template_alias() -> None:
    """Aliases del template ("Apodo", "Nombre completo") se ignoran."""
    out = _infer_short_name("", ["Apodo", "Nombre completo", "Greci"], "Grecia F.", "Grecia")
    assert out == "Greci"


def test_infer_short_name_filters_alias_equal_to_full_name() -> None:
    """Si el alias es idéntico al full_name (sólo capitalización/acento),
    no aporta — saltar al siguiente fallback."""
    out = _infer_short_name("", ["grecia"], "Grecia", "Grecia")
    # Alias == full_name → skip → cae al primer token o stem
    # full_name no tiene espacios → cae a stem
    assert out == "Grecia"


def test_infer_short_name_first_token_of_full_name() -> None:
    """Sin alias, si full_name tiene 2+ tokens, primer token gana."""
    out = _infer_short_name("", [], "Grecia Ferrari", "Grecia")
    assert out == "Grecia"


def test_infer_short_name_falls_back_to_stem() -> None:
    """Sin nada → stem del archivo."""
    out = _infer_short_name("", [], "", "Mama")
    assert out == "Mama"


# ── _parse_vault_contact (integración) ────────────────────────────────────

def test_parse_vault_contact_with_kinship_inferred(tmp_path: Path) -> None:
    """Caso real: nota de Grecia con `Relación: hija` → kinship inferido."""
    note = tmp_path / "Grecia.md"
    note.write_text(
        textwrap.dedent("""\
            [[Grecia|@Grecia]]
            - **Relación**: hija
            - **Apellido / nombre completo**: Grecia Ferrari
            - **Cumpleaños**: 07/06/2010
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    assert result["relation_label"] == "hija"
    assert result["kinship"] == "family-immediate"
    assert result["short_name"]  # algo no vacío


def test_parse_vault_contact_with_explicit_kinship_in_frontmatter(tmp_path: Path) -> None:
    """Frontmatter `kinship:` explícito gana sobre relation_label."""
    note = tmp_path / "Cliente.md"
    note.write_text(
        textwrap.dedent("""\
            ---
            type: mention
            kinship: friend-close
            ---
            - **Relación**: cliente
            - **Apellido / nombre completo**: Juan Pérez
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    # kinship explícito en frontmatter gana sobre infer ('cliente' →
    # professional-formal). User decidió override.
    assert result["kinship"] == "friend-close"


def test_parse_vault_contact_with_explicit_short_name(tmp_path: Path) -> None:
    """Body field `Apodo` gana sobre cualquier inferencia."""
    note = tmp_path / "Mama.md"
    note.write_text(
        textwrap.dedent("""\
            [[Mama|@Mama]]
            - **Relación**: Mamá
            - **Apellido / nombre completo**: Monica Ferrari
            - **Apodo**: Ma
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    assert result["short_name"] == "Ma"
    assert result["kinship"] == "family-immediate"


def test_parse_vault_contact_with_short_name_frontmatter(tmp_path: Path) -> None:
    """Frontmatter `short_name:` también acepta el override."""
    note = tmp_path / "Seba.md"
    note.write_text(
        textwrap.dedent("""\
            ---
            type: mention
            short_name: Seba
            ---
            - **Relación**: amigo cercano
            - **Apellido / nombre completo**: Sebastián López
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    assert result["short_name"] == "Seba"
    assert result["kinship"] == "friend-close"


def test_parse_vault_contact_unknown_kinship_for_empty_relation(tmp_path: Path) -> None:
    """Sin relation_label → kinship='unknown', short_name fallback al stem."""
    note = tmp_path / "Random.md"
    note.write_text(
        textwrap.dedent("""\
            - **Apellido / nombre completo**: Random Person
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    assert result["kinship"] == "unknown"
    assert result["short_name"] == "Random"  # primer token del full_name


def test_parse_vault_contact_preserves_existing_fields(tmp_path: Path) -> None:
    """Backwards compat: los campos previos siguen presentes con el shape
    anterior — los nuevos son additive."""
    note = tmp_path / "Maria.md"
    note.write_text(
        textwrap.dedent("""\
            [[Maria|@Maria]]
            - **Relación**: esposa
            - **Apellido / nombre completo**: Maria Test
            - **Teléfono**: +54 9 1234567
            - **Email**: maria@example.com
        """),
        encoding="utf-8",
    )
    result = _parse_vault_contact(note)
    # Campos previos: full_name, phones, emails, birthday, source, aliases,
    # relation_label.
    assert set(result.keys()) >= {
        "full_name", "phones", "emails", "birthday", "source", "aliases",
        "relation_label", "kinship", "short_name",
    }
    assert result["full_name"] == "Maria Test"
    assert result["phones"] == ["+54 9 1234567"]
    assert result["emails"] == ["maria@example.com"]
    assert result["relation_label"] == "esposa"
    assert result["kinship"] == "family-immediate"
