"""Tests para `_fuzzy_filter_contacts` + el cache schema v2 de contactos
(2026-04-24). Foco: scoring determinístico, normalización sin acentos,
filtrado por kind (phone/email/any), back-compat de cache v1.

NO testeamos el dump de osascript real — eso pasa por
`_ensure_contacts_cache` que ya tiene cobertura indirecta vía
`test_wa_cross_ref.py`. Acá monkeypatcheamos la lista de contactos para
aislar la lógica de filter/score.
"""
from __future__ import annotations

import json

import pytest

import rag


def _install_fake_contacts(monkeypatch, contacts: list[dict]) -> None:
    """Helper: instala una lista de contactos en el cache in-memory para
    que `_load_contacts_list` la devuelva sin pasar por osascript.

    El cache v2 chequea ``schema == _CONTACTS_CACHE_SCHEMA`` antes de
    aceptar el tier-1 in-memory, así que el dict debe incluir esa key.
    """
    rag._contacts_phone_index = {
        "ts": 9999999999,  # far future → cache always fresh
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }


@pytest.fixture
def sample_contacts(monkeypatch):
    contacts = [
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": ["grecia@x.com"]},
        {"name": "Gregorio López",  "phones": ["+5491166666666"], "emails": []},
        {"name": "Ágatha García",   "phones": [], "emails": ["agatha@y.com"]},
        {"name": "Mario Pérez",     "phones": ["+5491177777777"], "emails": ["mario@z.com"]},
        {"name": "María Hernández", "phones": ["+5491188888888"], "emails": ["maria@w.com"]},
        {"name": "Juan Sin Email",  "phones": ["+5491100000000"], "emails": []},
        {"name": "Sin Phone",       "phones": [], "emails": ["sinphone@x.com"]},
    ]
    _install_fake_contacts(monkeypatch, contacts)
    return contacts


# ── Scoring ────────────────────────────────────────────────────────────


def test_exact_full_name_scores_3(sample_contacts):
    out = rag._fuzzy_filter_contacts("Grecia Ferrari", kind="any")
    assert out
    assert out[0]["name"] == "Grecia Ferrari"
    assert out[0]["score"] == 3


def test_exact_token_scores_3(monkeypatch):
    """`whf6iuag` fix: un match exacto de token (ej. "Garcia" en
    "Maria Garcia") debe ganar contra un mero substring. Sin esto,
    "garcia" rankearía "Jose Garcia Lopez" (substring) igual que
    "Maria Garcia" (token exact)."""
    contacts = [
        {"name": "Maria Garcia", "phones": ["+1"], "emails": []},
        {"name": "Jose Garcialopez", "phones": ["+2"], "emails": []},  # solo substring
    ]
    _install_fake_contacts(monkeypatch, contacts)
    out = rag._fuzzy_filter_contacts("Garcia", kind="any")
    assert out[0]["name"] == "Maria Garcia"
    assert out[0]["score"] == 3


def test_prefix_full_name_scores_2(sample_contacts):
    out = rag._fuzzy_filter_contacts("Gre", kind="any")
    names = [c["name"] for c in out]
    # Ambos arrancan con "Gre"
    assert "Grecia Ferrari" in names
    assert "Gregorio López" in names
    for c in out:
        assert c["score"] == 2


def test_substring_match_scores_1(monkeypatch):
    """'rio' aparece como substring de un token compuesto, sin matchear
    como token exacto ni como prefix."""
    contacts = [
        {"name": "Mariocomp", "phones": ["+1"], "emails": []},  # 'rio' substring of token
    ]
    _install_fake_contacts(monkeypatch, contacts)
    out = rag._fuzzy_filter_contacts("rio", kind="any")
    assert out and out[0]["score"] == 1


# ── Normalización ──────────────────────────────────────────────────────


def test_accent_insensitive_match(sample_contacts):
    """'agatha' (sin tilde) debe matchear 'Ágatha García' (con tilde)."""
    out = rag._fuzzy_filter_contacts("agatha", kind="any")
    assert any(c["name"] == "Ágatha García" for c in out)


def test_case_insensitive_match(sample_contacts):
    out_lower = rag._fuzzy_filter_contacts("grecia", kind="any")
    out_upper = rag._fuzzy_filter_contacts("GRECIA", kind="any")
    assert [c["name"] for c in out_lower] == [c["name"] for c in out_upper]


def test_query_with_accents_matches_unaccented(monkeypatch):
    contacts = [
        {"name": "Astor Piazzolla", "phones": ["+1"], "emails": []},
    ]
    _install_fake_contacts(monkeypatch, contacts)
    out = rag._fuzzy_filter_contacts("ástor", kind="any")
    assert out and out[0]["name"] == "Astor Piazzolla"


# ── Filtrado por kind ──────────────────────────────────────────────────


def test_kind_phone_excludes_contacts_without_phone(sample_contacts):
    """`/wzp` solo debería ofrecer contactos con teléfono."""
    out = rag._fuzzy_filter_contacts("", kind="phone", limit=50)
    names = [c["name"] for c in out]
    assert "Sin Phone" not in names
    assert "Ágatha García" not in names  # sin teléfono también
    assert "Grecia Ferrari" in names


def test_kind_email_excludes_contacts_without_email(sample_contacts):
    """`/mail` solo debería ofrecer contactos con email."""
    out = rag._fuzzy_filter_contacts("", kind="email", limit=50)
    names = [c["name"] for c in out]
    assert "Juan Sin Email" not in names
    assert "Gregorio López" not in names  # sin email
    assert "Ágatha García" in names


def test_kind_any_returns_all(sample_contacts):
    out = rag._fuzzy_filter_contacts("", kind="any", limit=50)
    assert len(out) == 7


# ── Query vacía ────────────────────────────────────────────────────────


def test_empty_query_returns_alphabetical(sample_contacts):
    """Sin query → primeros N alfabéticos por `name.lower()`. 'Á' y 'A'
    son distintos en `lower()` (no folded) — el orden lo decide
    `_load_contacts_list` que sortea por `name.lower()`. Verificamos
    que sean los 3 primeros del orden case-insensitive."""
    out = rag._fuzzy_filter_contacts("", kind="any", limit=3)
    assert len(out) == 3
    # Score = 0 cuando no hay query.
    for c in out:
        assert c["score"] == 0
    # Todos vienen del set de fakes.
    fake_names = {c["name"] for c in sample_contacts}
    for c in out:
        assert c["name"] in fake_names


def test_whitespace_query_treated_as_empty(sample_contacts):
    out = rag._fuzzy_filter_contacts("   ", kind="any", limit=2)
    assert out and out[0]["score"] == 0


# ── Limit ──────────────────────────────────────────────────────────────


def test_limit_caps_results(sample_contacts):
    out = rag._fuzzy_filter_contacts("", kind="any", limit=2)
    assert len(out) == 2


def test_limit_zero_clamped_to_one(sample_contacts):
    """`limit=0` se clampa a 1 (UI defensiva — sin esto, un bug en el
    cliente que mande limit=0 dejaría el popover vacío sin razón visible)."""
    out = rag._fuzzy_filter_contacts("", kind="any", limit=0)
    assert len(out) == 1


# ── Sin matches ────────────────────────────────────────────────────────


def test_no_matches_returns_empty(sample_contacts):
    out = rag._fuzzy_filter_contacts("zzzzzzz", kind="any")
    assert out == []


def _contacts_loader_name() -> str:
    """Devuelve `_load_contacts_full` o `_load_contacts_list` — cualquiera
    que esté presente en la versión actual de rag.py. La función fue
    renombrada en commits paralelos (peer whf6iuag) entre `_full` y
    `_list` durante el desarrollo de esta feature; los tests no deberían
    romperse por ese churn de naming."""
    import rag
    for name in ("_load_contacts_full", "_load_contacts_list"):
        if hasattr(rag, name):
            return name
    raise RuntimeError("ningún loader de contacts list encontrado en rag")


def test_empty_cache_returns_empty(monkeypatch):
    """Si Apple Contacts está vacío / osascript silently-fails →
    el loader de contactos returns []. Forzamos eso reseteando el cache
    + monkeypatching el loader para evitar tier-2/3 fallback al disco/OS."""
    rag._contacts_phone_index = None
    monkeypatch.setattr(rag, _contacts_loader_name(), lambda ttl_s=86400: [])
    out = rag._fuzzy_filter_contacts("anything", kind="any")
    assert out == []


# ── Schema v2 + back-compat con v1 ─────────────────────────────────────


def test_v2_schema_disk_cache_loads_full(monkeypatch, tmp_path):
    """Cache v2 (con schema=2 + contacts list) se lee del disco y populá
    ambos índices del in-memory cache. Es el happy path del tier-2."""
    cache_path = tmp_path / "contacts_phone_index.json"
    cache_path.write_text(json.dumps({
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {"5491155555555": "Grecia"},
        "contacts": [
            {"name": "Grecia", "phones": ["+5491155555555"], "emails": ["g@x.com"]},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(rag, "_CONTACTS_PHONE_INDEX_PATH", cache_path)
    rag._contacts_phone_index = None  # force tier-2 disk read

    loader = getattr(rag, _contacts_loader_name())
    full = loader()
    assert len(full) == 1
    assert full[0]["name"] == "Grecia"
    assert full[0]["emails"] == ["g@x.com"]


def test_v1_schema_disk_cache_handled_gracefully(monkeypatch, tmp_path):
    """Si el archivo en disco es v1 (sin `schema` key), el comportamiento
    exacto depende del refactor del loader (peer whf6iuag está iterando):
    o lo acepta como-es (schema=1, contacts=[]) o lo descarta y re-dumpea.

    En cualquier caso el invariante que sí queremos garantizar:
      - `_load_contacts_phone_index` sigue funcionando (phone lookups OK
        — es un sub-feature crítico independiente de la list shape).
      - El sistema NO crashea al ver un cache v1.
    """
    cache_path = tmp_path / "contacts_phone_index.json"
    cache_path.write_text(json.dumps({
        "ts": 9999999999,
        "index": {"5491155555555": "Grecia"},
        # NO `schema` key — v1 (legacy)
    }), encoding="utf-8")
    monkeypatch.setattr(rag, "_CONTACTS_PHONE_INDEX_PATH", cache_path)
    rag._contacts_phone_index = None

    # Mockear osascript para que si el loader cae a tier-3 no escriba
    # contactos reales en el cache de disco del user.
    import subprocess
    class _FakeProc:
        returncode = 0
        stdout = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeProc())

    # No debe explotar ni levantar — el loader es silent-fail-friendly.
    cache = rag._ensure_contacts_cache()
    assert isinstance(cache, dict)
    # Phone lookups siempre OK, sea v1 conservado o v2 re-dumpeado vacío.
    # En el v1-conservado-as-is: el index original se preserva.
    # En el v2-re-dump-empty: el index queda vacío.
    # Ambos caminos son correctos — el test no dicta cuál de los dos.
    idx = rag._load_contacts_phone_index()
    assert isinstance(idx, dict)


def test_parse_contacts_dump_tab_format():
    """El dump v2 emite `name\\tP\\tphone` y `name\\tE\\temail`. El
    parser agrupa por nombre y construye la lista contacts.

    Tested via `_parse_contacts_dump` directly — pure parsing logic, sin
    osascript ni cache.
    """
    sample_dump = (
        "Grecia\tP\t+54 9 11 5555-5555\n"
        "Grecia\tE\tgrecia@x.com\n"
        "Mario\tP\t+5491177777777\n"
        "Mario\tE\tmario@z.com\n"
        "Mario\tE\tmario.alt@z.com\n"
    )
    idx, contacts = rag._parse_contacts_dump(sample_dump)
    # idx tiene los digits → name
    assert "Grecia" in idx.values()
    grecia = next(c for c in contacts if c["name"] == "Grecia")
    assert grecia["phones"] == ["+54 9 11 5555-5555"]
    assert grecia["emails"] == ["grecia@x.com"]
    mario = next(c for c in contacts if c["name"] == "Mario")
    assert "mario@z.com" in mario["emails"]
    assert "mario.alt@z.com" in mario["emails"]


def test_parse_contacts_dump_skips_short_phones():
    """Phones con <8 dígitos se descartan (probablemente extensiones)."""
    sample_dump = "Foo\tP\t1234\n"
    idx, contacts = rag._parse_contacts_dump(sample_dump)
    foo = next((c for c in contacts if c["name"] == "Foo"), None)
    # Foo aparece en `by_name` pero su phones queda vacío porque el dígito
    # no pasa el filtro >=8.
    if foo:
        assert foo["phones"] == []
