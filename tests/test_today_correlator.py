"""Tests for `rag.today_correlator` — pre-extraction of cross-source
patterns (people + topics) before the LLM narrative call.
"""

from rag.today_correlator import (
    _canonicalize_name,
    _extract_name_from_email,
    _extract_names_from_title,
    _tokenize,
    correlate_today_signals,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_canonicalize_name_basic():
    assert _canonicalize_name("Marina Pérez") == "marina pérez"
    # Token-sorted: same canonical regardless of order
    assert _canonicalize_name("Marina Pérez") == _canonicalize_name("Pérez Marina")
    # Lowercase + strip
    assert _canonicalize_name("  PABLO Fer  ") == "fer pablo"


def test_canonicalize_name_strips_emojis_and_symbols():
    assert _canonicalize_name("Grecia 🩷") == "grecia"
    assert _canonicalize_name("*Humanidades* Cuarto Año") in (
        "año cuarto humanidades", "cuarto humanidades año",
    )  # token-sorted
    assert _canonicalize_name("👻") == ""  # only emoji → empty
    assert _canonicalize_name("12345") == ""  # only digits → empty
    assert _canonicalize_name("a") == ""  # too short → empty


def test_extract_name_from_email():
    assert _extract_name_from_email("Pablo F <pablo@x.com>") == "Pablo F"
    assert _extract_name_from_email('"Marina P" <marina@y.com>') == "Marina P"
    # Bare email → take local part, titlecased
    assert _extract_name_from_email("pablo.fer@x.com") == "Pablo Fer"
    assert _extract_name_from_email("oscar_lopez@x.com") == "Oscar Lopez"
    # No parse → return as-is
    assert _extract_name_from_email("Algo Raro Sin Email") == "Algo Raro Sin Email"
    assert _extract_name_from_email("") == ""


def test_extract_names_from_calendar_title():
    # Single name
    assert "Pablo" in _extract_names_from_title("Sync con Pablo")
    # Two-word name
    assert any(
        "Pablo Fernández" in n or "Pablo" in n
        for n in _extract_names_from_title("Reunión con Pablo Fernández")
    )
    # Common non-name capitalized words filtered
    assert _extract_names_from_title("Demo cliente") == []
    assert _extract_names_from_title("Standup diario") == []
    # Multiple names
    titles = _extract_names_from_title("Almuerzo Pablo y María")
    assert any("Pablo" in t for t in titles) and any("María" in t for t in titles)


def test_tokenize_filters_stopwords_and_short():
    tokens = _tokenize("la reunión con Pablo es para revisar finanzas")
    # Keeps content words ≥4 chars
    assert "reunión" in tokens
    assert "pablo" in tokens
    assert "revisar" in tokens
    assert "finanzas" in tokens
    # Drops stopwords + short
    assert "la" not in tokens
    assert "con" not in tokens
    assert "es" not in tokens
    assert "para" not in tokens


def test_tokenize_dedup_within_text():
    """Un token aparece N veces en un texto, cuenta una sola vez."""
    tokens = _tokenize("ghostty ghostty ghostty terminal")
    assert tokens == {"ghostty", "terminal"}


# ── correlate_today_signals — people detection ──────────────────────────────


def test_people_detected_when_in_two_sources():
    """Pablo aparece en gmail + calendar → cruzado."""
    today_ev = {"recent_notes": [], "low_conf_queries": []}
    extras = {
        "gmail_today": [
            {"from": "Pablo F <pablo@x.com>", "subject": "Reunión 14hs",
             "snippet": "confirmás?"},
        ],
        "tomorrow_calendar": [
            {"title": "Sync con Pablo F", "time_range": "10:00–11:00"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    people = result["people"]
    assert len(people) >= 1
    pablo = next((p for p in people if "Pablo" in p["name"]), None)
    assert pablo is not None
    assert pablo["sources_count"] == 2
    assert {a["source"] for a in pablo["appearances"]} == {"gmail_today", "calendar"}


def test_people_NOT_detected_in_only_one_source():
    """Si una persona aparece solo en gmail (no en wa ni calendar),
    NO es una correlación cross-source — se filtra.
    """
    today_ev = {"recent_notes": [], "low_conf_queries": []}
    extras = {
        "gmail_today": [
            {"from": "Random Person <r@x.com>", "subject": "...",
             "snippet": "..."},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    assert result["people"] == []


def test_people_dedup_across_name_variants():
    """Pablo Fer (gmail) y Fer Pablo (calendar) son la misma persona
    (canonicalize sort-tokens).
    """
    today_ev = {}
    extras = {
        "gmail_today": [
            {"from": "Pablo Fer <pf@x.com>", "subject": "x", "snippet": ""},
        ],
        "tomorrow_calendar": [
            {"title": "Sync con Fer Pablo", "time_range": "10:00–11:00"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    assert len(result["people"]) == 1
    assert result["people"][0]["sources_count"] == 2


def test_people_three_sources_ranked_first():
    """Persona en 3 fuentes (gmail+wa+calendar) ordena antes que persona
    en 2 fuentes.
    """
    today_ev = {}
    extras = {
        "gmail_today": [
            {"from": "Marina P <m@x.com>", "subject": "x", "snippet": ""},
            {"from": "Pablo F <pf@x.com>", "subject": "y", "snippet": ""},
        ],
        "whatsapp_today": [
            {"name": "Marina P", "count": 5, "last_snippet": "che"},
        ],
        "tomorrow_calendar": [
            {"title": "Sync con Marina P", "time_range": "10:00–11:00"},
            {"title": "Demo Pablo F", "time_range": "16:00–17:00"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    people = result["people"]
    assert len(people) == 2
    # Marina (3 fuentes) debe aparecer antes que Pablo (2 fuentes)
    assert "Marina" in people[0]["name"]
    assert people[0]["sources_count"] == 3
    assert "Pablo" in people[1]["name"]
    assert people[1]["sources_count"] == 2


def test_people_skips_unnamed_whatsapp_chats():
    """Chats de WA con names tipo "12345" o solo símbolos no se cuentan
    (canonicalize devuelve "" → no agrega aparición).
    """
    today_ev = {}
    extras = {
        "gmail_today": [
            {"from": "Pablo F <pf@x.com>", "subject": "x", "snippet": ""},
        ],
        "whatsapp_today": [
            {"name": "12345", "count": 1, "last_snippet": "..."},
            {"name": "🩷🩷🩷", "count": 1, "last_snippet": "..."},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    # Pablo NO debería estar (solo en gmail), pero si los WA names eran
    # válidos podrían haber generado falsos cruces
    assert len(result["people"]) == 0


# ── correlate_today_signals — topic detection ───────────────────────────────


def test_topics_detected_when_in_two_buckets():
    """Token "ghostty" aparece en youtube + notas + screentime → topic real."""
    today_ev = {
        "recent_notes": [
            {"title": "Configurar terminal Ghostty", "path": "p", "snippet": "tema cursor ghostty"},
        ],
        "low_conf_queries": [],
    }
    extras = {
        "youtube_today": [
            {"title": "How to Configure Ghostty Terminal Cursor"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    topics = result["topics"]
    ghostty_topic = next((t for t in topics if t["topic"] == "ghostty"), None)
    assert ghostty_topic is not None
    assert ghostty_topic["sources_count"] >= 2
    assert "youtube" in ghostty_topic["sources"]
    assert "notas" in ghostty_topic["sources"]


def test_topics_NOT_detected_in_only_one_bucket():
    """Token solo en notas, no en otras fuentes → NO es topic cross-source."""
    today_ev = {
        "recent_notes": [
            {"title": "Algo único", "path": "p", "snippet": "tópico singular xyz"},
        ],
    }
    extras = {}
    result = correlate_today_signals(today_ev, extras)
    assert result["topics"] == []


def test_topics_filters_stopwords():
    """Un stopword que aparece en muchas fuentes NO genera topic."""
    today_ev = {
        "recent_notes": [
            {"title": "Lo que pasó hoy", "path": "p", "snippet": "lo de la mañana"},
        ],
    }
    extras = {
        "gmail_today": [{"from": "x", "subject": "Lo del lunes", "snippet": "lo de"}],
        "calendar_today": [{"title": "Lo de Pablo", "time_range": ""}],
    }
    result = correlate_today_signals(today_ev, extras)
    # "lo", "de", "la" son stopwords → no aparecen
    bad_tokens = {"lo", "de", "la", "del", "el"}
    for t in result["topics"]:
        assert t["topic"] not in bad_tokens


def test_topics_sorted_by_sources_count_desc():
    today_ev = {
        "recent_notes": [
            {"title": "Ghostty terminal", "path": "p", "snippet": "ghostty"},
        ],
    }
    extras = {
        "youtube_today": [{"title": "Ghostty tutorial"}],
        "calendar_today": [{"title": "Ghostty config session", "time_range": "10:00"}],
        "gmail_today": [{"from": "x", "subject": "Solo único", "snippet": "único"}],
    }
    result = correlate_today_signals(today_ev, extras)
    topics = result["topics"]
    # ghostty está en 3 fuentes (youtube + calendar + notas) → top
    assert topics[0]["topic"] == "ghostty"
    assert topics[0]["sources_count"] == 3


# ── correlate_today_signals — empty / edge cases ────────────────────────────


def test_empty_inputs():
    result = correlate_today_signals({}, {})
    assert result == {"people": [], "topics": []}


def test_handles_none_inputs():
    result = correlate_today_signals(None, None)
    assert result == {"people": [], "topics": []}
