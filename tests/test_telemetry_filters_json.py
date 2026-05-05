"""Regresión para el bug donde filters_json quedaba siempre NULL/ausente en rag_queries.

Root cause (2026-05-04): dos paths de escritura no incluían el key "filters"
en el dict que pasaban a log_query_event:

  1. to_log_event() de ChatTurnResult (rag/__init__.py) — path de run_chat_turn
     usado por web /api/chat + CLI chat. rr.filters_applied existía en el
     dataclass pero nunca se serializaba al dict de salida.

  2. log_query_event de web/server.py (cmd=web) — la variable _filters se
     computaba correctamente pero el dict donde vivía (línea ~13196) era un
     literal flotante sin asignación. Python lo evaluaba y descartaba. El
     log_query_event final no incluía el key "filters".

El mapper _map_queries_row busca ev["filters"] para llenar filters_json.
Sin ese key, la columna queda NULL → 6/1531 rows con contenido real en 7d.

Post-fix: to_log_event() incluye "filters": rr.filters_applied or {} y el
log_query_event de cmd=web incluye "filters": _filters or {}.
"""
from __future__ import annotations

import json

import rag


# ---------------------------------------------------------------------------
# Tests de _map_queries_row — unidad del mapper
# ---------------------------------------------------------------------------


def test_map_queries_row_persists_filters_json_with_content():
    """Cuando el evento trae filters con contenido, filters_json queda seteado.

    _map_queries_row devuelve el dict Python pre-serialización; la conversión
    a string JSON ocurre en _sql_serialise_row (llamado por _sql_append_event).
    Acá verificamos que el valor mapeado es el dict correcto.
    """
    ev = {
        "cmd": "query",
        "q": "hola",
        "filters": {"source": "whatsapp", "folder": "01-Projects"},
    }
    row = rag._map_queries_row(ev)
    assert "filters_json" in row
    # El mapper copia el dict tal cual; _sql_serialise_row lo stringifica al INSERT
    assert row["filters_json"] == {"source": "whatsapp", "folder": "01-Projects"}


def test_map_queries_row_persists_filters_json_empty_dict():
    """Cuando filters es {} (sin filtros reales), filters_json igual se persiste."""
    ev = {"cmd": "query", "q": "hola", "filters": {}}
    row = rag._map_queries_row(ev)
    # {} no es None, así que el mapper lo incluye como dict vacío
    assert "filters_json" in row
    assert row["filters_json"] == {}


def test_map_queries_row_no_filters_key_leaves_column_absent():
    """Sin el key filters en el evento, filters_json no se incluye (→ NULL en DB)."""
    ev = {"cmd": "query", "q": "hola"}
    row = rag._map_queries_row(ev)
    assert "filters_json" not in row


def test_map_queries_row_none_filters_leaves_column_absent():
    """filters=None explícito tampoco setea la columna (matches contrato del mapper)."""
    ev = {"cmd": "query", "q": "hola", "filters": None}
    row = rag._map_queries_row(ev)
    assert "filters_json" not in row


def test_map_queries_row_filters_with_date_range():
    """filters con since/until (date_range inferido) se persiste correctamente."""
    ev = {
        "cmd": "query",
        "q": "notas de enero",
        "filters": {"since": "2026-01-01", "until": "2026-01-31", "tag": "proyecto"},
    }
    row = rag._map_queries_row(ev)
    assert "filters_json" in row
    # El mapper preserva el dict; la serialización ocurre en _sql_append_event
    assert row["filters_json"]["since"] == "2026-01-01"
    assert row["filters_json"]["until"] == "2026-01-31"
    assert row["filters_json"]["tag"] == "proyecto"


# ---------------------------------------------------------------------------
# Tests de to_log_event — que el path run_chat_turn incluya filters
# ---------------------------------------------------------------------------


def _make_retrieve_result(filters_applied: dict) -> "rag.RetrieveResult":
    """Construye un RetrieveResult mínimo con filters_applied dado."""
    return rag.RetrieveResult(
        docs=["chunk text"],
        metas=[{"file": "01-Projects/nota.md", "note": "nota", "folder": "01-Projects"}],
        scores=[0.8],
        confidence=0.8,
        filters_applied=filters_applied,
    )


def _make_chat_turn_result(filters_applied: dict) -> "rag.ChatTurnResult":
    """Construye un ChatTurnResult mínimo con retrieve_result dado."""
    rr = _make_retrieve_result(filters_applied)
    return rag.ChatTurnResult(
        answer="respuesta de prueba",
        retrieve_result=rr,
        question="test query",
        turn_id="tid-test",
        intent="semantic",
    )


def test_to_log_event_includes_filters_with_content():
    """to_log_event incluye el key 'filters' cuando filters_applied tiene datos."""
    ctr = _make_chat_turn_result({"source": "calendar", "folder": "02-Areas"})
    ev = ctr.to_log_event(cmd="web", session_id="sess-1")
    assert "filters" in ev, "to_log_event debe incluir 'filters' key"
    assert ev["filters"] == {"source": "calendar", "folder": "02-Areas"}


def test_to_log_event_includes_filters_empty_dict():
    """to_log_event incluye 'filters': {} cuando no hay filtros aplicados."""
    ctr = _make_chat_turn_result({})
    ev = ctr.to_log_event(cmd="web", session_id="sess-1")
    assert "filters" in ev
    assert ev["filters"] == {}


def test_to_log_event_filters_maps_to_filters_json_via_mapper():
    """El dict de to_log_event pasado por _map_queries_row produce filters_json."""
    ctr = _make_chat_turn_result({"tag": "coaching", "since": "2026-01-01"})
    ev = ctr.to_log_event(cmd="chat", session_id="sess-2")
    row = rag._map_queries_row(ev)
    assert "filters_json" in row
    # El mapper preserva el dict pre-serialización
    assert row["filters_json"]["tag"] == "coaching"
    assert row["filters_json"]["since"] == "2026-01-01"


def test_to_log_event_filters_none_filters_applied_coerces_to_empty_dict():
    """Si filters_applied es None (edge case defensivo), to_log_event usa {}."""
    ctr = _make_chat_turn_result({})
    # Forzar filters_applied a None como edge case
    ctr.retrieve_result.filters_applied = None  # type: ignore[assignment]
    ev = ctr.to_log_event(cmd="web", session_id="sess-3")
    # Con "filters": rr.filters_applied or {}, None → {}
    assert ev.get("filters") == {}


# ---------------------------------------------------------------------------
# Test de integración E2E: log_query_event persiste filters_json en DB
# ---------------------------------------------------------------------------


def test_log_query_event_persists_filters_json_to_db(tmp_path, monkeypatch):
    """Integración: log_query_event escribe filters_json en rag_queries SQL."""
    import rag as _rag

    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        _rag.DB_PATH.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

        _rag.log_query_event({
            "cmd": "web",
            "q": "notas sobre coaching",
            "filters": {"source": "vault", "folder": "01-Projects"},
            "top_score": 0.75,
        })

        with _rag._ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT filters_json FROM rag_queries"
                " WHERE cmd='web' ORDER BY ts DESC LIMIT 1"
            ).fetchone()

        assert row is not None, "Debe haber al menos 1 row en rag_queries"
        assert row[0] is not None, "filters_json no debe ser NULL"
        parsed = json.loads(row[0])
        assert parsed.get("source") == "vault"
        assert parsed.get("folder") == "01-Projects"
    finally:
        _rag.DB_PATH = snap


def test_log_query_event_empty_filters_persists_braces(tmp_path, monkeypatch):
    """filters={} se persiste como '{}' — no NULL — para distinguir 'sin filtros' de 'no registrado'."""
    import rag as _rag

    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        _rag.DB_PATH.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

        _rag.log_query_event({
            "cmd": "chat",
            "q": "pregunta sin filtros",
            "filters": {},
        })

        with _rag._ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT filters_json FROM rag_queries"
                " WHERE cmd='chat' ORDER BY ts DESC LIMIT 1"
            ).fetchone()

        assert row is not None
        # {} no es None → mapper lo incluye → se serializa como "{}"
        assert row[0] == "{}"
    finally:
        _rag.DB_PATH = snap
