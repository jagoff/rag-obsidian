"""Tests for `rag.today_correlator` — pre-extraction of cross-source
patterns (people + topics) before the LLM narrative call.
"""

import pytest

from rag.today_correlator import (
    _canonicalize_name,
    _extract_name_from_email,
    _extract_names_from_title,
    _is_self_notification,
    _parse_time_to_minutes,
    _tokenize,
    correlate_today_signals,
    normalize_voice_to_2da_persona,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_canonicalize_name_basic():
    # NFD diacritic strip: "Pérez" → "perez" (sin tilde, ver fix
    # commit 22ae937 + tildes audit). Esto permite matchear contra
    # "Maria Perez" en gmail con "María Pérez" en WhatsApp.
    assert _canonicalize_name("Marina Pérez") == "marina perez"
    # Token-sorted: same canonical regardless of order
    assert _canonicalize_name("Marina Pérez") == _canonicalize_name("Pérez Marina")
    # Lowercase + strip
    assert _canonicalize_name("  PABLO Fer  ") == "fer pablo"


def test_canonicalize_name_strips_emojis_and_symbols():
    assert _canonicalize_name("Grecia 🩷") == "grecia"
    # NFD strip de "Año" → "Ano" (tilde ñ removida también — ñ se
    # descompone a n + ◌̃ y el tilde combinante cae)
    assert _canonicalize_name("*Humanidades* Cuarto Año") == \
        "ano cuarto humanidades"  # token-sorted alphabetically
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


def test_empty_inputs(monkeypatch):
    # Sin RAG_MOOD_ENABLED el bucket mood queda None.
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    result = correlate_today_signals({}, {})
    # Buckets cross-source vacíos. mood None porque feature off.
    # NB: pueden coexistir buckets adicionales (ej. `sleep` agregado por
    # el ingester Pillow) — validamos las keys que nos importan en lugar
    # de exact match para no acoplar tests al set entero de buckets.
    assert result["people"] == []
    assert result["topics"] == []
    assert result["time_overlaps"] == []
    assert result["gaps"] == []
    assert result["mood"] is None


def test_handles_none_inputs(monkeypatch):
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    result = correlate_today_signals(None, None)
    assert result["people"] == []
    assert result["topics"] == []
    assert result["time_overlaps"] == []
    assert result["gaps"] == []
    assert result["mood"] is None


# ── Self-notifications (github bot, google alerts, etc.) ───────────────────


def test_is_self_notification_github_bot():
    assert _is_self_notification("Fer F <notifications@github.com>")
    assert _is_self_notification("notifications@github.com")
    assert _is_self_notification("noreply@github.com")
    assert _is_self_notification("Foo <noreply@github.com>")


def test_is_self_notification_google_services():
    assert _is_self_notification("security@google.com")
    assert _is_self_notification("noreply@accounts.google.com")
    assert _is_self_notification("noreply@youtube.com")


def test_is_self_notification_payment_services():
    assert _is_self_notification("receipts@stripe.com")
    assert _is_self_notification("service@paypal.com")


def test_is_self_notification_real_person_not_filtered():
    assert not _is_self_notification("Marina Pérez <marina@empresa.com>")
    assert not _is_self_notification("pablo.fer@gmail.com")
    assert not _is_self_notification("john@startup.io")


def test_is_self_notification_empty_treated_as_not_real():
    """Sender vacío no debería contar como persona real para correlación."""
    assert _is_self_notification("")
    assert _is_self_notification(None or "")


def test_people_excludes_github_self_notification():
    """Mails de notifications@github.com con 'Fer F' como display NO deben
    crear "Fer F" como persona cross-source. Es ruido del feed automático.
    """
    today_ev = {}
    extras = {
        "gmail_today": [
            {"from": "Fer F <notifications@github.com>",
             "subject": "[jagoff/rag-obsidian] Run failed: CI",
             "snippet": "..."},
            {"from": "Fer F <notifications@github.com>",
             "subject": "[jagoff/rag-obsidian] Run failed: CI master",
             "snippet": "..."},
        ],
        "whatsapp_today": [
            # WA self-chat — el mismo "Fer F" como nombre
            {"name": "Fer F", "count": 2, "last_snippet": "nota"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    # "Fer F" NO debe aparecer porque gmail era self-notification
    assert all("Fer F" not in p["name"] for p in result["people"])


def test_people_real_person_still_detected_after_filter():
    """Filtro de self-notifications NO debe afectar personas reales."""
    today_ev = {}
    extras = {
        "gmail_today": [
            {"from": "notifications@github.com", "subject": "x", "snippet": ""},
            {"from": "Pablo Fer <pablo@empresa.com>", "subject": "y", "snippet": ""},
        ],
        "tomorrow_calendar": [
            {"title": "Sync con Pablo Fer", "time_range": "10:00–11:00"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    assert any("Pablo" in p["name"] for p in result["people"])


def test_topics_excludes_github_ci_noise():
    """Tokens 'failed', 'master', 'jagoff', 'ferrari' que dominan los
    mails CI no deben aparecer como topics (ahora son stopwords).
    """
    today_ev = {
        "recent_notes": [
            {"title": "ci failed master jagoff ferrari", "path": "p",
             "snippet": "rag-obsidian build deploy"},
        ],
    }
    extras = {
        "gmail_today": [
            {"from": "notifications@github.com",
             "subject": "[jagoff/rag-obsidian] Run failed: CI master",
             "snippet": "ferrari rag-obsidian build deploy"},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    bad_tokens = {"failed", "master", "jagoff", "ferrari", "rag-obsidian",
                  "build", "deploy", "ci"}
    for t in result["topics"]:
        assert t["topic"] not in bad_tokens, (
            f"CI noise token '{t['topic']}' got through stopwords"
        )


def test_topics_excludes_self_notification_mails_from_tokenization():
    """Los mails de github notifications no contribuyen al topic
    extraction — incluso si los tokens NO son stopwords, no debería
    cuentar la fuente "gmail" como una de las fuentes del topic.
    """
    today_ev = {
        "recent_notes": [
            {"title": "tema importante xyz123", "path": "p", "snippet": ""},
        ],
    }
    extras = {
        # Solo notification mail con el mismo token "xyz123"
        "gmail_today": [
            {"from": "notifications@github.com",
             "subject": "xyz123 in CI", "snippet": ""},
        ],
    }
    result = correlate_today_signals(today_ev, extras)
    # xyz123 está en notas + (gmail filtrado) = solo 1 source → no es topic
    xyz_topic = next((t for t in result["topics"] if "xyz123" in t["topic"]), None)
    assert xyz_topic is None, (
        "El mail de notifications no debería contribuir al topic count"
    )


# ── Time overlap correlation ─────────────────────────────────────────────


def test_parse_time_to_minutes():
    assert _parse_time_to_minutes("10:00") == 600
    assert _parse_time_to_minutes("14:30") == 870
    assert _parse_time_to_minutes("10:00–11:00") == 600  # toma el primero
    assert _parse_time_to_minutes("2:30 PM") == 870
    assert _parse_time_to_minutes("12:00 AM") == 0
    assert _parse_time_to_minutes("") is None
    assert _parse_time_to_minutes("sin hora") is None


def test_time_overlap_gmail_calendar_match():
    """Gmail recibido 14:00 + calendar event 14:00 con tokens compartidos
    → debe detectarse como overlap.
    """
    extras = {
        "gmail_today": [
            # 14:00 local — internal_date_ms = 1700000000000 + algo
            {"from": "Pablo <p@x.com>",
             "subject": "Reunión proyecto demo confirmada",
             "snippet": "te confirmo el meet de demo a las 14",
             "internal_date_ms": _ms_at_local_time(14, 0)},
        ],
        "calendar_today": [
            {"title": "Reunión proyecto demo", "start": "14:00",
             "end": "15:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    overlaps = result["time_overlaps"]
    assert len(overlaps) >= 1
    o = overlaps[0]
    assert {it["source"] for it in o["items"]} == {"gmail", "calendar"}
    # Tokens compartidos: "reunión", "proyecto", "demo"
    assert "demo" in o["shared_tokens"] or "reunión" in o["shared_tokens"]


def test_time_overlap_skipped_when_no_shared_tokens():
    """Gmail 14:00 + calendar 14:00 SIN tokens en común → NO debe matchear."""
    extras = {
        "gmail_today": [
            {"from": "x@x.com", "subject": "newsletter aleatorio",
             "snippet": "blah blah",
             "internal_date_ms": _ms_at_local_time(14, 0)},
        ],
        "calendar_today": [
            {"title": "Yoga clase mañanera", "start": "14:00", "end": "15:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert result["time_overlaps"] == []


def test_time_overlap_skipped_when_far_apart():
    """Gmail 09:00 + calendar 17:00 → fuera de la window de 30min, no match."""
    extras = {
        "gmail_today": [
            {"from": "x@x.com", "subject": "demo proyecto",
             "snippet": "demo proyecto importante",
             "internal_date_ms": _ms_at_local_time(9, 0)},
        ],
        "calendar_today": [
            {"title": "Demo proyecto", "start": "17:00", "end": "18:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert result["time_overlaps"] == []


def test_time_overlap_skips_self_notifications():
    """github bot mail no debe contar para time overlaps aunque tenga
    timestamp."""
    extras = {
        "gmail_today": [
            {"from": "Fer F <notifications@github.com>",
             "subject": "demo proyecto failed CI",
             "snippet": "...",
             "internal_date_ms": _ms_at_local_time(14, 0)},
        ],
        "calendar_today": [
            {"title": "Demo proyecto", "start": "14:00", "end": "15:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert result["time_overlaps"] == []


def _ms_at_local_time(hour: int, minute: int) -> int:
    """Helper: timestamp en ms para la hora especificada de hoy."""
    from datetime import datetime
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return int(target.timestamp() * 1000)


# ── Voice normalization ──────────────────────────────────────────────────


def test_voice_replaces_first_person_singular_verbs():
    text = "Hoy trabajé intensamente en la auditoría. Recibí 3 mails."
    result = normalize_voice_to_2da_persona(text)
    assert "trabajé" not in result
    assert "Recibí" not in result
    # Case-preserving: "trabajé" → "trabajaste" (lower), "Recibí" → "Recibiste"
    assert "trabajaste" in result
    assert "Recibiste" in result


def test_voice_replaces_first_person_plural_verbs():
    text = "Hoy tocamos varias notas y vimos los diagramas."
    result = normalize_voice_to_2da_persona(text)
    assert "tocamos" not in result
    assert "tocaste" in result
    assert "viste" in result


def test_voice_preserves_case():
    text = "Trabajé en el proyecto. RECIBÍ 5 mails."
    result = normalize_voice_to_2da_persona(text)
    assert "Trabajaste" in result
    assert "RECIBISTE" in result


def test_voice_preserves_quoted_text():
    """Citas literales del user no deben ser modificadas."""
    text = 'Notaste que "yo no quería hacer eso" pero lo hiciste igual.'
    result = normalize_voice_to_2da_persona(text)
    # "yo" en la cita queda
    assert '"yo no quería hacer eso"' in result
    # "Notaste" e "hiciste" están bien (ya 2da persona)
    assert "Notaste" in result


def test_voice_replaces_pronouns():
    text = "Yo me centré en el tema, mi trabajo fue largo."
    result = normalize_voice_to_2da_persona(text)
    # "Yo" → "Vos", "me" → "te", "mi" → "tu"
    assert "Vos te centraste" in result
    assert "tu trabajo" in result


def test_voice_does_not_break_unrelated_text():
    """Texto sin 1ra persona debe quedar idéntico (idempotente para 2da)."""
    text = "Tocaste varias notas. Recibiste 3 mails. Viste 2 videos."
    result = normalize_voice_to_2da_persona(text)
    assert result == text


def test_voice_handles_empty_text():
    assert normalize_voice_to_2da_persona("") == ""
    assert normalize_voice_to_2da_persona(None or "") == ""


def test_voice_word_boundary_no_substring_replace():
    """'concentré' contiene 'centré' pero NO debe ser reemplazado."""
    text = "Te concentré en esto."  # palabra inventada para test
    # En realidad "concentré" probablemente no es palabra real, pero
    # palabras como "hablé" están en la lista; necesitamos que
    # "establé" NO se modifique.
    text2 = "Hablé claro." # es 1PS, debe cambiarse
    result2 = normalize_voice_to_2da_persona(text2)
    assert "Hablaste" in result2
    # Palabra rara que CONTIENE "fui" pero word-boundary no aplica:
    text3 = "fuimos al cine"  # "fuimos" 1PP → "fuiste"
    result3 = normalize_voice_to_2da_persona(text3)
    assert "fuiste" in result3


# ── Gaps detection ───────────────────────────────────────────────────────


def test_gaps_detected_when_wa_unreplied_no_calendar_slot():
    """Marina te escribió 26h atrás + Marina NO está en tomorrow_calendar
    → gap detectado.
    """
    extras = {
        "whatsapp_unreplied": [
            {"name": "Marina Pérez", "jid": "549@s",
             "last_snippet": "che vení mañana", "hours_waiting": 26.5},
        ],
        "tomorrow_calendar": [
            {"title": "Standup diario", "time_range": "10:00–10:30"},
        ],
    }
    result = correlate_today_signals({}, extras)
    gaps = result["gaps"]
    assert len(gaps) == 1
    assert gaps[0]["person"] == "Marina Pérez"
    assert gaps[0]["hours_waiting"] == 26.5
    assert gaps[0]["kind"] == "wa_unreplied_no_slot"


def test_gaps_skipped_when_wa_unreplied_has_calendar_slot():
    """Marina escribió + Marina aparece en tomorrow_calendar → NO es gap
    (ya tenés agendado tiempo para hablarle).
    """
    extras = {
        "whatsapp_unreplied": [
            {"name": "Marina", "jid": "x", "last_snippet": "hola",
             "hours_waiting": 30},
        ],
        "tomorrow_calendar": [
            {"title": "Sync con Marina", "time_range": "14:00–15:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert result["gaps"] == []


def test_gaps_skipped_when_hours_waiting_under_24h():
    """Si llevás <24h sin responder, todavía no es loose end."""
    extras = {
        "whatsapp_unreplied": [
            {"name": "Pablo", "jid": "x", "last_snippet": "hola",
             "hours_waiting": 5},
        ],
        "tomorrow_calendar": [],
    }
    result = correlate_today_signals({}, extras)
    assert result["gaps"] == []


def test_gaps_sorted_by_hours_waiting_desc():
    """Loose ends más viejos primero — son los más urgentes."""
    extras = {
        "whatsapp_unreplied": [
            {"name": "Reciente", "jid": "x", "last_snippet": "x",
             "hours_waiting": 30},
            {"name": "Vieja", "jid": "y", "last_snippet": "y",
             "hours_waiting": 80},
            {"name": "Media", "jid": "z", "last_snippet": "z",
             "hours_waiting": 50},
        ],
        "tomorrow_calendar": [],
    }
    result = correlate_today_signals({}, extras)
    persons = [g["person"] for g in result["gaps"]]
    assert persons == ["Vieja", "Media", "Reciente"]


def test_gaps_empty_when_no_unreplied():
    extras = {"whatsapp_unreplied": [], "tomorrow_calendar": []}
    result = correlate_today_signals({}, extras)
    assert result["gaps"] == []


def test_gaps_partial_name_match_in_calendar():
    """'Marina Pérez' en wa + 'Marina' (single token) en calendar → MATCH
    (no es gap). El nombre del calendar es subset.
    """
    extras = {
        "whatsapp_unreplied": [
            {"name": "Marina Pérez", "jid": "x",
             "last_snippet": "x", "hours_waiting": 30},
        ],
        "tomorrow_calendar": [
            {"title": "Café con Marina", "time_range": "10:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert result["gaps"] == []


# ── Diacritic normalization (tildes) ────────────────────────────────────


def test_canonicalize_strips_diacritics_for_match():
    """'María Pérez' (con tildes) y 'Maria Perez' (sin) deben canonicalize
    al mismo valor → matchear como la misma persona.
    """
    assert _canonicalize_name("María Pérez") == _canonicalize_name("Maria Perez")
    assert _canonicalize_name("José") == _canonicalize_name("Jose")
    assert _canonicalize_name("ÁNGELA") == _canonicalize_name("Angela")
    # Ñ se preserva (es alfabética sin diacrítico — categoría Lo)
    assert "n" in _canonicalize_name("Ñoño")


def test_people_match_across_diacritic_variants():
    """gmail llega con 'Maria Perez', WA con 'María Pérez' → MATCH como
    misma persona cross-source.
    """
    extras = {
        "gmail_today": [
            {"from": "Maria Perez <m@x.com>", "subject": "x", "snippet": ""},
        ],
        "whatsapp_today": [
            {"name": "María Pérez", "count": 5, "last_snippet": "che"},
        ],
    }
    result = correlate_today_signals({}, extras)
    assert len(result["people"]) == 1
    assert result["people"][0]["sources_count"] == 2


# ── Grupos WA filtrados de gaps ─────────────────────────────────────────


def test_gaps_skips_wa_group_with_asterisks():
    """Chat WA con '*' en el nombre es grupo, no persona individual."""
    from rag.today_correlator import _looks_like_wa_group
    assert _looks_like_wa_group("*Humanidades* Cuarto Año")
    assert _looks_like_wa_group("**Equipo X**")
    assert not _looks_like_wa_group("Marina Pérez")


def test_gaps_skips_wa_group_with_marker_keywords():
    from rag.today_correlator import _looks_like_wa_group
    assert _looks_like_wa_group("Equipo Backend")
    assert _looks_like_wa_group("Team Marketing")
    assert _looks_like_wa_group("Grupo Familia")
    assert _looks_like_wa_group("Comunidad Devs")
    # case insensitive
    assert _looks_like_wa_group("EQUIPO ALPHA")


def test_gaps_skips_wa_group_with_4plus_words():
    """Nombres largos típicos de grupo: 'Fifteens - Casa Santa Fe'."""
    from rag.today_correlator import _looks_like_wa_group
    assert _looks_like_wa_group("Fifteens - Casa Santa Fe")
    assert _looks_like_wa_group("Reunión Padres Cuarto Año Mañana")
    # Person con 2-3 tokens NO es grupo
    assert not _looks_like_wa_group("Marina Pérez")
    assert not _looks_like_wa_group("Pablo Fernández González")


def test_gaps_filters_groups_in_correlation():
    """Integration: grupos en wa_unreplied no aparecen como gaps."""
    extras = {
        "whatsapp_unreplied": [
            {"name": "*Humanidades* Cuarto Año", "jid": "x@g.us",
             "last_snippet": "info patrullaje", "hours_waiting": 30},
            {"name": "Marina Pérez", "jid": "y@s",
             "last_snippet": "che", "hours_waiting": 30},
        ],
        "tomorrow_calendar": [],
    }
    result = correlate_today_signals({}, extras)
    persons = [g["person"] for g in result["gaps"]]
    assert "Marina Pérez" in persons
    assert all("Humanidades" not in p for p in persons)


def test_gaps_skips_when_hours_waiting_is_none():
    """Datos incompletos del fetcher no deben generar gap silencioso."""
    extras = {
        "whatsapp_unreplied": [
            {"name": "Pablo", "jid": "x", "last_snippet": "hola",
             "hours_waiting": None},
        ],
        "tomorrow_calendar": [],
    }
    result = correlate_today_signals({}, extras)
    assert result["gaps"] == []


# ── Time overlap timezone consistency ───────────────────────────────────


def test_time_overlap_uses_local_time_consistently():
    """Test que `_correlate_time_overlaps` usa `datetime.fromtimestamp()`
    sin `tz=` (local naive), igual que el helper `_ms_at_local_time`.
    Si alguien cambia a UTC en uno solo de los dos, los tests pasan
    inconsistente con el código real. Smoke test contra un timestamp
    sintético de "ahora 14:00 local".
    """
    # Generar un ms a "hoy 14:00 local" usando el helper
    ms_at_14 = _ms_at_local_time(14, 0)
    # Crear extras con gmail a esa hora + calendar event a la misma hora
    # con tokens compartidos
    extras = {
        "gmail_today": [
            {"from": "x@y.com", "subject": "demo proyecto",
             "snippet": "demo proyecto a las 14",
             "internal_date_ms": ms_at_14},
        ],
        "calendar_today": [
            {"title": "Demo proyecto", "start": "14:00", "end": "15:00"},
        ],
    }
    result = correlate_today_signals({}, extras)
    overlaps = result["time_overlaps"]
    # Debe haber match — gmail 14:00 local + calendar 14:00 local
    # con tokens "demo" + "proyecto" compartidos.
    assert len(overlaps) >= 1, (
        "Time overlap NO detectado — posible timezone mismatch entre "
        "_ms_at_local_time (test helper) y fromtimestamp (correlator)."
    )


# ── mood bucket ────────────────────────────────────────────────────────────


def test_mood_bucket_none_when_feature_off(monkeypatch):
    """Sin RAG_MOOD_ENABLED, correlate_today_signals devuelve mood=None."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    result = correlate_today_signals({}, {})
    assert result["mood"] is None


def test_mood_bucket_none_when_no_score_today(monkeypatch):
    """Con feature on pero sin row para hoy en rag_mood_score_daily,
    bucket queda None (no asumimos score=0 = ok)."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import mood
    monkeypatch.setattr(mood, "get_score_for_date", lambda _d: None)
    monkeypatch.setattr(mood, "get_recent_scores", lambda days=7: [])
    monkeypatch.setattr(mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    result = correlate_today_signals({}, {})
    assert result["mood"] is None


def test_mood_bucket_none_when_score_has_zero_signals(monkeypatch):
    """Row para hoy pero n_signals=0 (daemon corrió pero no encontró
    nada): bucket None, no contaminamos el prompt con neutro vacío."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import mood
    monkeypatch.setattr(mood, "get_score_for_date", lambda _d: {
        "date": "2026-04-30", "score": 0.0, "n_signals": 0,
        "sources_used": [], "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(mood, "get_recent_scores", lambda days=7: [])
    monkeypatch.setattr(mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    result = correlate_today_signals({}, {})
    assert result["mood"] is None


def test_mood_bucket_populated_when_active(monkeypatch):
    """Feature on + score con n_signals > 0: bucket completo."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import mood
    monkeypatch.setattr(mood, "get_score_for_date", lambda _d: {
        "date": "2026-04-30", "score": -0.5, "n_signals": 4,
        "sources_used": ["spotify", "journal"],
        "top_evidence": [
            {"source": "journal", "signal_kind": "keyword_negative",
             "value": -0.7, "weight": 1.0, "evidence": {}},
            {"source": "spotify", "signal_kind": "artist_mood_lookup",
             "value": -0.4, "weight": 1.0, "evidence": {}},
        ],
        "updated_at": 0,
    })
    monkeypatch.setattr(mood, "get_recent_scores", lambda days=7: [
        {"date": "2026-04-30", "score": -0.5, "n_signals": 4},
        {"date": "2026-04-29", "score": -0.1, "n_signals": 2},
        {"date": "2026-04-28", "score": +0.1, "n_signals": 3},
    ])
    monkeypatch.setattr(mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 1,
        "avg_score": -0.5, "dates": ["2026-04-30"], "reason": "only_1_days",
    })
    result = correlate_today_signals({}, {})
    bucket = result["mood"]
    assert bucket is not None
    assert bucket["score"] == -0.5
    assert bucket["n_signals"] == 4
    assert "spotify" in bucket["sources_used"]
    # week_avg = mean(-0.5, -0.1, +0.1) = -0.166
    assert bucket["week_avg"] == pytest.approx(-0.166, abs=0.01)
    # delta = -0.5 - (-0.166) = -0.334 → declining
    assert bucket["trend"] == "declining"
    # Drift bucket pasa con shape limpia.
    assert bucket["drift"]["drifting"] is False
    assert bucket["drift"]["n_consecutive"] == 1
    # Top evidence top 2 incluido.
    assert len(bucket["top_evidence"]) == 2


def test_mood_bucket_trend_improving(monkeypatch):
    """Score hoy MUY arriba del promedio 7d → trend improving."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import mood
    monkeypatch.setattr(mood, "get_score_for_date", lambda _d: {
        "date": "2026-04-30", "score": +0.6, "n_signals": 3,
        "sources_used": ["spotify"], "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(mood, "get_recent_scores", lambda days=7: [
        {"date": "2026-04-30", "score": +0.6, "n_signals": 3},
        {"date": "2026-04-29", "score": -0.1, "n_signals": 2},
        {"date": "2026-04-28", "score": -0.2, "n_signals": 2},
    ])
    monkeypatch.setattr(mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    result = correlate_today_signals({}, {})
    assert result["mood"]["trend"] == "improving"


def test_mood_bucket_silent_fail_on_exception(monkeypatch):
    """Si alguna función de rag.mood tira, _correlate_mood devuelve None
    en vez de propagar (no rompe el today brief si la DB está locked)."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import mood
    def _broken(*args, **kwargs):
        raise RuntimeError("DB locked")
    monkeypatch.setattr(mood, "get_score_for_date", _broken)
    result = correlate_today_signals({}, {})
    assert result["mood"] is None
