from pathlib import Path
from unittest.mock import patch

import pytest

import rag


MENTIONS_REL = "04-Archive/99-obsidian-system/99-Mentions"


@pytest.fixture(autouse=True)
def _reset_caches():
    rag._mentions_cache = None
    rag._contacts_cache.clear()
    rag._contacts_permission_warned = False
    yield
    rag._mentions_cache = None
    rag._contacts_cache.clear()


@pytest.fixture
def fake_vault(tmp_path):
    folder = tmp_path / MENTIONS_REL
    folder.mkdir(parents=True)
    (folder / "_template.md").write_text(
        "---\ntype: mention\n---\nignore me, scaffold only.\n",
        encoding="utf-8",
    )
    (folder / "Grecia.md").write_text(
        "---\ntype: mention\naliases: [Greciaña]\n---\n"
        "- **Relación**: hija\n- **Notas**: 15 años\n",
        encoding="utf-8",
    )
    (folder / "Mama.md").write_text(
        "---\ntype: mention\n---\n- **Relación**: madre\n",
        encoding="utf-8",
    )
    (folder / "Yo.md").write_text(
        "---\ntype: mention\n---\nshould be skipped (min-len 3).\n",
        encoding="utf-8",
    )
    (folder / "Multi.md").write_text(
        "---\ntype: mention\naliases:\n  - segundoalias\n  - tercer\n---\nbody.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_index_skips_underscore_and_short_stems(fake_vault):
    idx = rag._load_mentions_index(fake_vault)
    assert "grecia" in idx
    assert "mama" in idx
    assert "yo" not in idx
    assert "_template" not in idx
    assert all(not Path(p).name.startswith("_") for p in idx.values())


def test_index_picks_up_aliases_inline_and_block(fake_vault):
    idx = rag._load_mentions_index(fake_vault)
    assert "greciana" in idx  # accent stripped from "Greciaña"
    assert "segundoalias" in idx
    assert "tercer" in idx


def test_index_is_accent_insensitive_on_match(fake_vault):
    paths = rag._match_mentions_in_query("hablé con Mamá ayer", fake_vault)
    assert paths == [str(Path(MENTIONS_REL) / "Mama.md")]


def test_match_word_boundary_no_substring_false_positives(fake_vault):
    assert rag._match_mentions_in_query("quiero yogur", fake_vault) == []
    assert rag._match_mentions_in_query("mamarracho de cosa", fake_vault) == []


def test_match_caps_at_two_ordered_by_position(fake_vault):
    paths = rag._match_mentions_in_query("Mama y Grecia hablaron de Multi", fake_vault)
    assert len(paths) == 2
    assert paths[0].endswith("Mama.md")
    assert paths[1].endswith("Grecia.md")


def test_match_empty_when_no_mention(fake_vault):
    assert rag._match_mentions_in_query("¿cuál es mi DNI?", fake_vault) == []


def test_build_person_context_returns_none_on_no_match(fake_vault):
    with patch.object(rag, "_fetch_contact", return_value=None):
        assert rag.build_person_context("¿cuál es mi DNI?", fake_vault) is None


def test_build_person_context_includes_body_and_contact(fake_vault):
    fake_contact = {
        "full_name": "Grecia Test",
        "phones": ["+54 9 000"],
        "emails": ["g@example.com"],
        "birthday": "07/06/2010",
    }
    with patch.object(rag, "_fetch_contact", return_value=fake_contact):
        ctx = rag.build_person_context("info sobre Grecia", fake_vault)
    assert ctx is not None
    assert ctx.startswith(rag._PERSON_CONTEXT_HEADER)
    assert "Relación**: hija" in ctx
    assert "Apple Contacts" in ctx
    assert "Grecia Test" in ctx
    assert "+54 9 000" in ctx
    assert "07/06/2010" in ctx


def test_build_person_context_caps_body_at_1500(fake_vault):
    big = "x" * 5000
    (fake_vault / MENTIONS_REL / "Grecia.md").write_text(
        f"---\ntype: mention\n---\n{big}\n", encoding="utf-8"
    )
    rag._mentions_cache = None
    with patch.object(rag, "_fetch_contact", return_value=None):
        ctx = rag.build_person_context("info sobre Grecia", fake_vault)
    assert ctx is not None
    # Body-only x-count: the header has prose and may include x chars, so
    # we assert the distinct body chunk (between the `### Grecia` marker
    # and the blank-line separator) is capped at _MENTIONS_BODY_CAP.
    body_region = ctx.split("### Grecia\n", 1)[1].split("\n\n", 1)[0]
    assert body_region.count("x") <= rag._MENTIONS_BODY_CAP


def test_build_person_context_omits_contact_block_when_none(fake_vault):
    with patch.object(rag, "_fetch_contact", return_value=None):
        ctx = rag.build_person_context("info sobre Grecia", fake_vault)
    assert ctx is not None
    assert "Apple Contacts" not in ctx


def test_contacts_shim_caches_and_handles_missing_osascript():
    with patch("rag.subprocess.run", side_effect=FileNotFoundError):
        assert rag._fetch_contact("Grecia") is None
    cache_key = "Grecia||"
    assert cache_key in rag._contacts_cache
    with patch("rag.subprocess.run") as mocked:
        result = rag._fetch_contact("Grecia")
    assert result is None
    mocked.assert_not_called()


def test_contacts_shim_parses_osascript_output():
    class _Proc:
        returncode = 0
        stdout = "Grecia 🥰::+54 1|+54 2|::a@x.com|b@x.com|::lunes 7"
        stderr = ""

    with patch("rag.subprocess.run", return_value=_Proc()):
        c = rag._fetch_contact("Grecia")
    assert c == {
        "full_name": "Grecia 🥰",
        "phones": ["+54 1", "+54 2"],
        "emails": ["a@x.com", "b@x.com"],
        "birthday": "lunes 7",
    }


def test_contacts_shim_silent_on_permission_denied():
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "not authorised"

    with patch("rag.subprocess.run", return_value=_Proc()):
        assert rag._fetch_contact("Anyone") is None
    assert rag._contacts_permission_warned is True


def test_contacts_shim_handles_timeout():
    import subprocess as _sp
    with patch("rag.subprocess.run", side_effect=_sp.TimeoutExpired(cmd="osascript", timeout=3.0)):
        assert rag._fetch_contact("Slowperson") is None


def test_parse_mention_metadata_extracts_canonical_and_email():
    body = (
        "[[Mama|@Mama]]\n"
        "- **Relación**: Mama\n"
        "- **Apellido / nombre completo**: Monica Ferrari\n"
        "- **Cumpleaños**: 09/02/yyyy\n"
        "- **Teléfono**: +54 9 ...\n"
        "- **Email**: monicaferrari@gmail.com\n"
    )
    meta = rag._parse_mention_metadata(body)
    assert meta["canonical"] == "Monica Ferrari"
    assert meta["email"] == "monicaferrari@gmail.com"


def test_parse_mention_metadata_skips_template_placeholders():
    body = "- Apellido / nombre completo: ...\n- Email: \n"
    meta = rag._parse_mention_metadata(body)
    assert meta["canonical"] is None
    assert meta["email"] is None


def test_fetch_contact_skips_relation_stems_when_no_canonical():
    """Stem 'Mama' alone must not query Contacts — would false-positive."""
    with patch("rag._osascript_contact_search") as mocked:
        result = rag._fetch_contact("Mama")
    assert result is None
    mocked.assert_not_called()


def test_fetch_contact_validates_returned_name_against_canonical():
    """A contact name lacking the canonical's tokens is rejected."""
    bad = {
        "full_name": "Carina (Mama Bianca)",
        "phones": ["+5499"], "emails": [], "birthday": "",
    }
    with patch("rag._osascript_contact_search", return_value=bad):
        result = rag._fetch_contact("Mama", canonical="Monica Ferrari")
    assert result is None


def test_fetch_contact_accepts_canonical_substring_match():
    good = {
        "full_name": "Mónica",
        "phones": ["+54 9 000"], "emails": [], "birthday": "",
    }
    with patch("rag._osascript_contact_search", return_value=good):
        result = rag._fetch_contact("Mama", canonical="Monica Ferrari")
    assert result == good


def test_fetch_contact_tries_email_first_then_name():
    calls: list[tuple[str, str]] = []

    def fake_search(predicate: str, value: str) -> dict | None:
        calls.append((predicate, value))
        if predicate.startswith("value of emails"):
            return None
        return {"full_name": "Grecia Ferrari", "phones": ["+54 1"], "emails": [], "birthday": ""}

    with patch("rag._osascript_contact_search", side_effect=fake_search):
        rag._fetch_contact("Grecia", email="g@x.com", canonical="Grecia Ferrari")
    assert calls[0] == ("value of emails contains", "g@x.com")
    assert any(p.startswith("name contains") for p, _ in calls[1:])
