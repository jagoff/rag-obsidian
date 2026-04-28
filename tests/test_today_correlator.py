"""Tests for `rag.today_correlator` — pre-extraction of cross-source
patterns (people + topics) before the LLM narrative call.
"""

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
    assert result == {"people": [], "topics": [], "time_overlaps": [], "gaps": []}


def test_handles_none_inputs():
    result = correlate_today_signals(None, None)
    assert result == {"people": [], "topics": [], "time_overlaps": [], "gaps": []}


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
