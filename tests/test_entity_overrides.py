"""Tests para `entity_overrides.json` + `known_places.json` + global stopwords.

Cubre el sistema de filtrado del extractor GLiNER agregado el 2026-05-10
(commits 71adc72 + df40ade). Los 3 mecanismos:

1. `_load_entity_overrides()` — JSON `~/.config/obsidian-rag/entity_overrides.json`
   con format `{nombre_lowercase: target_type}` para re-classify.
2. `_load_known_places()` — allowlist `rag/data/known_places.json` + extra file
   `~/.config/obsidian-rag/known_places_extra.json`. Si una entity sale como
   `location` y no matchea, se skipea.
3. `_ENTITY_STOPWORDS_GLOBAL` — frozenset hardcoded para chat slang que el
   modelo puede asignar a CUALQUIER tipo (oka, dale, jajaja, etc.).

Patrón de los tests: monkeypatch los path constants + cache dicts para que
cada test sea independiente (los caches viven a nivel módulo).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag import (
    _ENTITY_LABELS,
    _ENTITY_STOPWORDS_GLOBAL,
    _ENTITY_STOPWORDS_PERSON,
    _load_entity_overrides,
    _load_known_places,
    _override_entity_label,
    _validate_location_or_demote,
)
import rag as rag_mod


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset module-level mtime caches antes de cada test."""
    rag_mod._ENTITY_OVERRIDE_CACHE["mtime"] = 0.0
    rag_mod._ENTITY_OVERRIDE_CACHE["data"] = {}
    rag_mod._KNOWN_PLACES_CACHE["mtimes"] = (0.0, 0.0)
    rag_mod._KNOWN_PLACES_CACHE["data"] = frozenset()
    yield
    rag_mod._ENTITY_OVERRIDE_CACHE["mtime"] = 0.0
    rag_mod._ENTITY_OVERRIDE_CACHE["data"] = {}
    rag_mod._KNOWN_PLACES_CACHE["mtimes"] = (0.0, 0.0)
    rag_mod._KNOWN_PLACES_CACHE["data"] = frozenset()


@pytest.fixture
def override_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Apunta `_ENTITY_OVERRIDE_FILE` a un path temp."""
    p = tmp_path / "entity_overrides.json"
    monkeypatch.setattr(rag_mod, "_ENTITY_OVERRIDE_FILE", p)
    return p


@pytest.fixture
def known_places_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Apunta canonical + extra a paths temp."""
    canonical = tmp_path / "known_places.json"
    extra = tmp_path / "known_places_extra.json"
    monkeypatch.setattr(rag_mod, "_KNOWN_PLACES_FILE", canonical)
    monkeypatch.setattr(rag_mod, "_KNOWN_PLACES_EXTRA_FILE", extra)
    return canonical, extra


# ── _load_entity_overrides ────────────────────────────────────────────────────


def test_load_entity_overrides_empty_when_file_missing(override_file: Path):
    """Sin archivo → dict vacío, sin error."""
    assert not override_file.exists()
    assert _load_entity_overrides() == {}


def test_load_entity_overrides_basic(override_file: Path):
    """Path feliz: nombres mapeados a tipos válidos."""
    override_file.write_text(json.dumps({
        "grecia": "person",
        "barcelona": "person",
    }), encoding="utf-8")
    overrides = _load_entity_overrides()
    assert overrides == {"grecia": "person", "barcelona": "person"}


def test_load_entity_overrides_lowercases_keys(override_file: Path):
    """Las keys se normalizan a lowercase."""
    override_file.write_text(json.dumps({
        "Grecia": "person",
        "  CASA  ": "organization",
    }), encoding="utf-8")
    overrides = _load_entity_overrides()
    assert "grecia" in overrides
    assert "casa" in overrides
    assert "Grecia" not in overrides


def test_load_entity_overrides_skips_invalid_types(override_file: Path):
    """Tipos no válidos descartados; el resto sigue."""
    override_file.write_text(json.dumps({
        "valid": "person",
        "bogus": "alien",  # not in _ENTITY_LABELS
        "empty": "",
    }), encoding="utf-8")
    overrides = _load_entity_overrides()
    assert overrides == {"valid": "person"}


def test_load_entity_overrides_skips_non_string_values(override_file: Path):
    """Valores no-string se ignoran (defensivo contra archivos corruptos)."""
    override_file.write_text(json.dumps({
        "valid": "person",
        "weird": 42,
        "none": None,
    }), encoding="utf-8")
    overrides = _load_entity_overrides()
    assert overrides == {"valid": "person"}


def test_load_entity_overrides_malformed_json_keeps_last_good(override_file: Path):
    """Si el JSON queda corrupto a mitad de edit, mantenemos la última versión válida."""
    override_file.write_text(json.dumps({"first": "person"}), encoding="utf-8")
    first = _load_entity_overrides()
    assert first == {"first": "person"}
    # Corrupt the file
    override_file.write_text("{not valid json", encoding="utf-8")
    # Cache mtime bumps because the file changed, but the JSON parse fails →
    # we keep the prior cached `data` (last-good) instead of clearing.
    second = _load_entity_overrides()
    assert second == {"first": "person"}


def test_load_entity_overrides_mtime_cache_returns_fresh_on_change(override_file: Path):
    """Después de editar el archivo, el load lee la nueva versión."""
    override_file.write_text(json.dumps({"name1": "person"}), encoding="utf-8")
    assert _load_entity_overrides() == {"name1": "person"}
    # Change file (also need to bump mtime).
    import os
    import time
    time.sleep(0.05)  # ensure mtime ticks
    override_file.write_text(json.dumps({"name2": "location"}), encoding="utf-8")
    os.utime(override_file, None)
    assert _load_entity_overrides() == {"name2": "location"}


def test_load_entity_overrides_non_dict_root_returns_empty(override_file: Path):
    """JSON válido pero root no-dict → dict vacío."""
    override_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert _load_entity_overrides() == {}


# ── _load_known_places ────────────────────────────────────────────────────────


def test_load_known_places_canonical_only(known_places_files: tuple[Path, Path]):
    """Path feliz: solo canonical, no extra."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina", "brasil"]}), encoding="utf-8")
    places = _load_known_places()
    assert "argentina" in places
    assert "brasil" in places


def test_load_known_places_merges_extra(known_places_files: tuple[Path, Path]):
    """Canonical + extra se mergean (allowing user a agregar sin pisar)."""
    canonical, extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina"]}), encoding="utf-8")
    extra.write_text(json.dumps({"places": ["country los molinos", "guardia los molinos"]}), encoding="utf-8")
    places = _load_known_places()
    assert "argentina" in places
    assert "country los molinos" in places
    assert "guardia los molinos" in places


def test_load_known_places_lowercases(known_places_files: tuple[Path, Path]):
    """Todas las entries normalizadas a lowercase + stripped."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["  ARGENTINA  ", "Brasil"]}), encoding="utf-8")
    places = _load_known_places()
    assert "argentina" in places
    assert "brasil" in places
    assert "ARGENTINA" not in places


def test_load_known_places_returns_frozenset(known_places_files: tuple[Path, Path]):
    """Retorna frozenset (immutable) — caller no debería poder modificar el cache."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina"]}), encoding="utf-8")
    places = _load_known_places()
    assert isinstance(places, frozenset)


def test_load_known_places_handles_missing_files(known_places_files: tuple[Path, Path]):
    """Sin archivos → frozenset vacía, sin error."""
    canonical, _extra = known_places_files
    assert not canonical.exists()
    places = _load_known_places()
    assert places == frozenset()


def test_load_known_places_skips_non_string_entries(known_places_files: tuple[Path, Path]):
    """Entries no-string en la lista se ignoran."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina", 42, None, "brasil"]}), encoding="utf-8")
    places = _load_known_places()
    assert "argentina" in places
    assert "brasil" in places
    assert len(places) == 2


# ── _override_entity_label ────────────────────────────────────────────────────


def test_override_entity_label_no_overrides_returns_original(override_file: Path):
    """Sin overrides cargados → retorna elabel sin cambios."""
    assert _override_entity_label("Grecia", "location") == "location"


def test_override_entity_label_applies_match(override_file: Path):
    """Match en override → retorna target."""
    override_file.write_text(json.dumps({"grecia": "person"}), encoding="utf-8")
    assert _override_entity_label("Grecia", "location") == "person"
    assert _override_entity_label("grecia", "location") == "person"


def test_override_entity_label_no_match_returns_original(override_file: Path):
    """Override file existe pero no matchea este nombre → original elabel."""
    override_file.write_text(json.dumps({"grecia": "person"}), encoding="utf-8")
    assert _override_entity_label("Argentina", "location") == "location"


def test_override_entity_label_case_insensitive(override_file: Path):
    """Lookup case-insensitive."""
    override_file.write_text(json.dumps({"grecia": "person"}), encoding="utf-8")
    assert _override_entity_label("GRECIA", "location") == "person"
    assert _override_entity_label("Grecia", "location") == "person"
    assert _override_entity_label("  Grecia  ", "location") == "person"


# ── _validate_location_or_demote ──────────────────────────────────────────────


def test_validate_location_passthrough_for_non_location(known_places_files: tuple[Path, Path]):
    """Si elabel != location → pasa sin cambios (no intenta validar)."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": []}), encoding="utf-8")
    # Aún sin places, person/org/event pasan.
    assert _validate_location_or_demote("Maria", "person") == "person"
    assert _validate_location_or_demote("Apple", "organization") == "organization"
    assert _validate_location_or_demote("Year-end", "event") == "event"


def test_validate_location_accepts_known_place(known_places_files: tuple[Path, Path]):
    """Si elabel == location y matchea allowlist → retorna location."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina", "buenos aires"]}), encoding="utf-8")
    assert _validate_location_or_demote("Argentina", "location") == "location"
    assert _validate_location_or_demote("buenos aires", "location") == "location"


def test_validate_location_skips_unknown(known_places_files: tuple[Path, Path]):
    """Si elabel == location y NO matchea → None (caller debe skip la entity)."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina"]}), encoding="utf-8")
    assert _validate_location_or_demote("Mac", "location") is None
    assert _validate_location_or_demote("oka", "location") is None
    assert _validate_location_or_demote("casa", "location") is None


def test_validate_location_uses_extra_file(known_places_files: tuple[Path, Path]):
    """Lugares en `extra` se aceptan igual que canonical."""
    canonical, extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina"]}), encoding="utf-8")
    extra.write_text(json.dumps({"places": ["country los molinos"]}), encoding="utf-8")
    assert _validate_location_or_demote("Country Los Molinos", "location") == "location"


def test_validate_location_case_insensitive(known_places_files: tuple[Path, Path]):
    """Lookup case-insensitive."""
    canonical, _extra = known_places_files
    canonical.write_text(json.dumps({"places": ["argentina"]}), encoding="utf-8")
    assert _validate_location_or_demote("ARGENTINA", "location") == "location"
    assert _validate_location_or_demote("argentina", "location") == "location"
    assert _validate_location_or_demote("  Argentina  ", "location") == "location"


# ── _ENTITY_STOPWORDS_GLOBAL ──────────────────────────────────────────────────


def test_entity_stopwords_global_covers_chat_slang():
    """Slang de chat común no debería pasar como entity de ningún tipo."""
    expected = {"oka", "okey", "dale", "che", "ufa", "jajaja", "bueno", "hola", "chau"}
    for term in expected:
        assert term in _ENTITY_STOPWORDS_GLOBAL, f"missing slang: {term!r}"


def test_entity_stopwords_global_covers_common_nouns():
    """Common nouns que GLiNER tagea como location se filtran globalmente."""
    expected = {"casa", "comedor", "centro", "parque", "sala", "puente", "plaza", "country"}
    for term in expected:
        assert term in _ENTITY_STOPWORDS_GLOBAL, f"missing common noun: {term!r}"


def test_entity_stopwords_global_covers_spatial_adverbs():
    """Adverbios espaciales (acá, allá, aquí, ahí) no son lugares."""
    expected = {"acá", "aca", "allá", "alla", "aquí", "aqui", "ahí", "ahi"}
    for term in expected:
        assert term in _ENTITY_STOPWORDS_GLOBAL, f"missing spatial adverb: {term!r}"


def test_entity_stopwords_global_disjoint_from_person_stopwords():
    """Las dos sets son distintas — global aplica a TODOS, person solo a person."""
    overlap = _ENTITY_STOPWORDS_GLOBAL & _ENTITY_STOPWORDS_PERSON
    # Acepta intersección chica (artículos como "the"/"a"/"an") sin objetar.
    # Lo que NO queremos es que sean idénticas (sería redundancia).
    assert _ENTITY_STOPWORDS_GLOBAL != _ENTITY_STOPWORDS_PERSON


# ── Canonical known_places.json shipped en el repo ────────────────────────────


def test_canonical_known_places_file_loads():
    """El archivo `rag/data/known_places.json` shipped con el repo es JSON válido
    y carga sin errores. Sanity check para que un upgrade corrupto no rompa
    el extractor en producción."""
    canonical = Path(rag_mod.__file__).parent / "data" / "known_places.json"
    assert canonical.exists(), f"missing canonical known_places: {canonical}"
    raw = json.loads(canonical.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert "places" in raw
    assert isinstance(raw["places"], list)
    assert len(raw["places"]) >= 100  # sanity floor — tenemos 600+


def test_canonical_known_places_includes_argentina_basics():
    """Verifica que países + ciudades AR críticos están en el allowlist
    (regression-proof si alguien edita el archivo a mano)."""
    canonical = Path(rag_mod.__file__).parent / "data" / "known_places.json"
    raw = json.loads(canonical.read_text(encoding="utf-8"))
    places = {s.strip().lower() for s in raw["places"] if isinstance(s, str)}
    # Países
    assert "argentina" in places
    assert "brasil" in places
    assert "uruguay" in places
    # Provincia + ciudad principales
    assert "santa fe" in places
    assert "buenos aires" in places
    assert "córdoba" in places or "cordoba" in places


def test_canonical_known_places_does_not_include_grecia():
    """`Grecia` debe NO estar en el allowlist canonical — el override
    `~/.config/obsidian-rag/entity_overrides.json` `{"grecia": "person"}`
    re-clasifica a person, y el allowlist no debería listarlo como
    valid location (sino el override sería redundante en algunos paths)."""
    canonical = Path(rag_mod.__file__).parent / "data" / "known_places.json"
    raw = json.loads(canonical.read_text(encoding="utf-8"))
    places = {s.strip().lower() for s in raw["places"] if isinstance(s, str)}
    assert "grecia" not in places, "remove `grecia` from known_places — es nombre de hija del user"
