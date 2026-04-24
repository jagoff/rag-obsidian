"""Tests for the Drive on-demand chat tool.

Origin: user report 2026-04-24 (Fer F.) — "busca en mi google drive y
decime cuánto adeuda alexis de la macbook pro" respondió con la única
planilla snapshoteada (`Lista de precios Online`, iPhones) y no buscó
realmente en Drive. El chat sólo tenía acceso al snapshot diario de
`_sync_gdrive_notes` (4 docs más recientes en 48h, body 8000 chars),
sin un camino on-demand para queries sobre archivos viejos o grandes.

Este archivo cubre el fix en 4 capas (bottom-up):

1. `_drive_search_tokens` — filtrado de stopwords ES/EN.
2. `_agent_tool_drive_search` — helper de rag con el Drive API mockeado.
3. `drive_search` en `web.tools` — wrapper silent-fail + registración en
   `CHAT_TOOLS` / `PARALLEL_SAFE` (esos asserts viven en
   `test_web_chat_tools.py`; acá sólo probamos el comportamiento del
   wrapper en si).
4. Pre-router regex + `_format_drive_block` + `_SOURCE_INTENT_META`
   mapping — el glue layer que el usuario toca cuando escribe en el
   chat.

Todos los tests son puros — no tocan TestClient ni la red. El Drive API
se monkeypatchea vía `rag._drive_service` devolviendo un fake con el
mismo shape que usa el helper.
"""
from __future__ import annotations

import json

import pytest

import rag
from web import server as server_mod
from web import tools as tools_mod
from web.server import (
    _SOURCE_INTENT_META,
    _build_source_intent_hint,
    _detect_tool_intent,
    _format_drive_block,
    _format_forced_tool_output,
    _is_empty_tool_output,
)


# ── 1. Token extractor ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected",
    [
        # El caso del user report: todas las filler words deben caer.
        (
            "busca en mi google drive y decime cuando adeuda alexis de la macbook pro",
            ["adeuda", "alexis", "macbook", "pro"],
        ),
        # Solo stopwords → lista vacía.
        ("buscá en mi drive", []),
        # Puntuación + mayúsculas: se normaliza.
        ("¿Cuánto me DEBE Alexis?", ["debe", "alexis"]),
        # Tokens cortos (<2 chars) se dropean.
        ("busca a, b, cd en drive", ["cd"]),
        # Dedup — "alexis" no se repite aunque aparezca dos veces.
        ("alexis alexis venturino", ["alexis", "venturino"]),
        # EN stopwords también filtran.
        ("search the macbook in my drive for alexis", ["search", "macbook", "alexis"]),
    ],
)
def test_drive_search_tokens(query: str, expected: list[str]):
    assert rag._drive_search_tokens(query) == expected


def test_drive_search_tokens_cap():
    """`max_tokens` trunca el resultado y NO revierte el orden — los primeros
    tokens tienen prioridad (el user suele poner lo más relevante al principio
    de una pregunta)."""
    q = "alfa bravo charlie delta echo foxtrot golf hotel india juliet"
    toks = rag._drive_search_tokens(q, max_tokens=4)
    assert toks == ["alfa", "bravo", "charlie", "delta"]


def test_drive_search_tokens_empty():
    assert rag._drive_search_tokens("") == []
    assert rag._drive_search_tokens("   ") == []


# ── 2. `_agent_tool_drive_search` helper ───────────────────────────────────


class _FakeDriveResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeFiles:
    def __init__(self, listing: list[dict], export_bodies: dict[str, bytes | str]):
        self._listing = listing
        self._export = export_bodies
        self.list_calls: list[dict] = []
        self.export_calls: list[tuple[str, str]] = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return _FakeDriveResponse({"files": list(self._listing)})

    def export(self, fileId: str, mimeType: str):
        self.export_calls.append((fileId, mimeType))
        body = self._export.get(fileId, "")
        return _FakeDriveResponse(body)


class _FakeDriveService:
    def __init__(self, listing: list[dict], export_bodies: dict[str, bytes | str]):
        self._files = _FakeFiles(listing, export_bodies)

    def files(self):
        return self._files


def test_drive_search_no_credentials(monkeypatch):
    """Sin creds de Drive, el helper devuelve un JSON con `error` pero
    nunca raisea — el chat loop lo renderea como 'sin resultados' y
    sigue operando."""
    monkeypatch.setattr(rag, "_drive_service", lambda: None)
    out = rag._agent_tool_drive_search("alexis macbook")
    parsed = json.loads(out)
    assert parsed["files"] == []
    assert parsed["error"] == "no_google_credentials"
    assert parsed["tokens"] == ["alexis", "macbook"]


def test_drive_search_empty_query_after_stopwords(monkeypatch):
    """Query que sólo tiene stopwords → error explícito, sin pegarle al API."""
    monkeypatch.setattr(rag, "_drive_service", lambda: pytest.fail("no debió llamar al API"))
    out = rag._agent_tool_drive_search("busca en mi drive por favor")
    parsed = json.loads(out)
    assert parsed["files"] == []
    assert parsed["tokens"] == []
    assert "vacía" in parsed["error"]


def test_drive_search_happy_path(monkeypatch):
    """Con creds válidas + matches en Drive, devuelve file list + bodies."""
    listing = [
        {
            "id": "file_1",
            "name": "Venta Macbook - Alexis.docx",
            "mimeType": "application/vnd.google-apps.document",
            "modifiedTime": "2026-04-15T10:30:00.000Z",
            "webViewLink": "https://docs.google.com/document/d/file_1/edit",
            "lastModifyingUser": {"displayName": "Fer F."},
        },
        {
            "id": "file_2",
            "name": "Planilla deudas 2026.xlsx",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "modifiedTime": "2026-03-01T08:00:00.000Z",
            "webViewLink": "https://docs.google.com/spreadsheets/d/file_2/edit",
            "lastModifyingUser": {"displayName": "Fer F."},
        },
    ]
    bodies = {
        "file_1": b"Alexis Herrera - Macbook Pro M1 - Adeuda USD 500",
        "file_2": "Alexis,USD 500\nOtros,USD 0",
    }
    svc = _FakeDriveService(listing, bodies)
    monkeypatch.setattr(rag, "_drive_service", lambda: svc)

    out = rag._agent_tool_drive_search("busca en drive alexis macbook")
    parsed = json.loads(out)

    assert parsed["tokens"] == ["alexis", "macbook"]
    assert parsed["query_used"] == "alexis macbook"
    assert len(parsed["files"]) == 2

    f1 = parsed["files"][0]
    assert f1["name"] == "Venta Macbook - Alexis.docx"
    assert f1["mime_label"] == "Doc"
    assert "USD 500" in f1["body"]
    assert f1["link"].startswith("https://docs.google.com/document/")
    assert f1["modifier"] == "Fer F."

    f2 = parsed["files"][1]
    assert f2["mime_label"] == "Sheet"
    assert "Alexis" in f2["body"]

    # El query cruda va como `fullText contains '<tokens>'` + `trashed = false`.
    assert svc._files.list_calls, "nunca llamó al API"
    q_arg = svc._files.list_calls[0]["q"]
    assert "fullText contains 'alexis macbook'" in q_arg
    assert "trashed = false" in q_arg


def test_drive_search_name_fallback_for_name_only_match(monkeypatch):
    """Regresión 2026-04-24 iter 2 (Fer F.): cuando el usuario pregunta por
    tokens que VIVEN en el NOMBRE del archivo pero NO en su contenido
    indexado (caso típico: spreadsheet con nombre descriptivo "Alex -
    Cuotas Macbook" pero contenido sólo numérico — cuotas, fechas), el
    strict `fullText contains 'token1 token2'` devolvía 0 matches y el
    chat respondía "no encontré nada en Drive". Fix: agregamos
    `name contains 'tokenX'` OR por cada token ≥5 chars al clause, así
    cualquiera de esos tokens presentes en el nombre dispara match aunque
    el body no los tenga."""
    listing = [{
        "id": "planilla_alex",
        "name": "Alex - Cuotas Macbook",
        "mimeType": "application/vnd.google-apps.spreadsheet",
        "modifiedTime": "2026-04-17T16:46:42.000Z",
        "webViewLink": "https://docs.google.com/spreadsheets/d/planilla_alex/edit",
        "lastModifyingUser": {"displayName": "Fer F."},
    }]
    bodies = {
        "planilla_alex": "Cuotas,Precio,Fecha\nCuota 01,200000,14/11/2025\nFaltan,730000,",
    }
    svc = _FakeDriveService(listing, bodies)
    monkeypatch.setattr(rag, "_drive_service", lambda: svc)

    out = rag._agent_tool_drive_search(
        "busca en mi drive cuánto adeuda alexis de la macbook pro"
    )
    parsed = json.loads(out)

    # El tokenizer produce tokens ≥5 chars (alexis/macbook/adeuda) + "pro" (3).
    assert parsed["tokens"] == ["adeuda", "alexis", "macbook", "pro"]
    assert len(parsed["files"]) == 1
    assert parsed["files"][0]["name"] == "Alex - Cuotas Macbook"
    assert "730000" in parsed["files"][0]["body"]

    q_arg = svc._files.list_calls[0]["q"]
    # El nuevo shape tiene un OR-chain con name contains sólo para tokens
    # ≥5 chars — "pro" queda fuera para no inflar con falsos positivos.
    assert "name contains 'alexis'" in q_arg
    assert "name contains 'macbook'" in q_arg
    assert "name contains 'adeuda'" in q_arg
    # "pro" (3 chars) NO debe estar en el name-OR chain — demasiado
    # genérico, matchea todo tipo de archivos ("Resume pro", "AWS
    # pricing"…).
    assert "name contains 'pro'" not in q_arg
    # fullText AND sigue siendo el first-match más barato.
    assert "fullText contains 'adeuda alexis macbook pro'" in q_arg


def test_drive_search_short_tokens_only_falls_back_to_fulltext(monkeypatch):
    """Si TODOS los tokens son <5 chars (caso raro), no agregamos name-OR
    chain y usamos sólo `fullText contains`. Esto evita queries con cero
    tokens específicos que traerían falsos positivos masivos (p.ej.
    "dame mail pro" → name contains 'pro' matchea cualquier cosa)."""
    listing = []
    svc = _FakeDriveService(listing, {})
    monkeypatch.setattr(rag, "_drive_service", lambda: svc)

    # "app" + "dev" + "ceo" — todos ≤4 chars.
    out = rag._agent_tool_drive_search("buscá app dev ceo")
    _ = json.loads(out)

    q_arg = svc._files.list_calls[0]["q"]
    # Sólo fullText — sin name-OR chain.
    assert "fullText contains 'app dev ceo'" in q_arg
    assert "name contains" not in q_arg


def test_drive_search_api_error_is_silent(monkeypatch):
    class _BoomService:
        def files(self):
            class _F:
                def list(self, **_):
                    raise RuntimeError("429 rate limited")
            return _F()

    monkeypatch.setattr(rag, "_drive_service", lambda: _BoomService())
    out = rag._agent_tool_drive_search("alexis macbook")
    parsed = json.loads(out)
    assert parsed["files"] == []
    assert parsed["error"].startswith("search_failed")


def test_drive_search_body_cap_respected(monkeypatch):
    """Body larger than `body_cap` se trunca — evita inflar el CONTEXTO del
    LLM más allá de lo necesario."""
    listing = [{
        "id": "file_big",
        "name": "Big.doc",
        "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "2026-01-01T00:00:00.000Z",
        "webViewLink": "",
        "lastModifyingUser": {},
    }]
    huge_body = "X" * 50_000
    svc = _FakeDriveService(listing, {"file_big": huge_body})
    monkeypatch.setattr(rag, "_drive_service", lambda: svc)

    out = rag._agent_tool_drive_search("alexis", body_cap=800)
    parsed = json.loads(out)
    assert len(parsed["files"][0]["body"]) == 800


def test_drive_search_skips_export_for_unknown_mime(monkeypatch):
    """mimeTypes fuera de `_GDRIVE_EXPORT_MIME` (ej. PDF) NO se exportan —
    body queda vacío. Evita errores de 'export not supported' spameando el log.
    """
    listing = [{
        "id": "file_pdf",
        "name": "Contrato.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-01T00:00:00.000Z",
        "webViewLink": "",
        "lastModifyingUser": {},
    }]
    svc = _FakeDriveService(listing, {})
    monkeypatch.setattr(rag, "_drive_service", lambda: svc)

    out = rag._agent_tool_drive_search("contrato")
    parsed = json.loads(out)
    assert parsed["files"][0]["body"] == ""
    assert parsed["files"][0]["mime_label"] == "PDF"
    # NO debería haber llamado export() — la PDF no es exportable.
    assert svc._files.export_calls == []


# ── 3. Web-layer wrapper + registration ────────────────────────────────────


def test_web_tools_exposes_drive_search():
    """El wrapper está registrado en `CHAT_TOOLS`, `TOOL_FNS` y
    `PARALLEL_SAFE`."""
    assert "drive_search" in tools_mod.TOOL_FNS
    assert tools_mod.drive_search in tools_mod.CHAT_TOOLS
    assert "drive_search" in tools_mod.PARALLEL_SAFE
    assert tools_mod.drive_search.__doc__ and "Drive" in tools_mod.drive_search.__doc__


def test_web_tool_addendum_mentions_drive():
    """El addendum (system prompt adicional que ve el LLM en el tool-decide
    round) documenta el nuevo tool. Sin esto el LLM no sabe cuándo llamarlo."""
    assert "drive_search" in tools_mod._WEB_TOOL_ADDENDUM
    assert "google drive" in tools_mod._WEB_TOOL_ADDENDUM.lower()


def test_web_wrapper_silent_fail(monkeypatch):
    """`drive_search` wrapper delega en `_agent_tool_drive_search`; si éste
    devuelve un JSON con error, el wrapper passthrough-ea sin modificarlo."""
    monkeypatch.setattr(tools_mod, "_agent_tool_drive_search",
                        lambda q, max_files=5: '{"files": [], "error": "boom"}')
    out = tools_mod.drive_search("alexis")
    assert json.loads(out) == {"files": [], "error": "boom"}


# ── 4. Pre-router regex ────────────────────────────────────────────────────


@pytest.mark.parametrize("query", [
    "busca en mi google drive qué me adeuda alexis",
    "busca en mi drive por la planilla de gastos",
    "tengo algo en google drive sobre esto?",
    "busca en la planilla de marzo",
    "revisá el spreadsheet de alexis",
    "buscame el sheet de ventas",
    "andá a la presentación de Q4",
    "en mi drive debería estar algo de alexis",
])
def test_pre_router_fires_drive_search(query: str):
    matched = _detect_tool_intent(query)
    names = [n for n, _ in matched]
    assert "drive_search" in names, (
        f"Query {query!r} debería haber enganchado drive_search, matched={names}"
    )
    # Cuando matchea, el arg `query` es la pregunta cruda — el helper
    # filtra stopwords internamente.
    for n, args in matched:
        if n == "drive_search":
            assert args == {"query": query}


@pytest.mark.parametrize("query", [
    # "drive" suelto sin contexto de Google Drive NO debería disparar
    # (ej. "drive-through", "hard drive", "drive-by"). Nuestra regex
    # requiere "google drive", "en mi drive", "mi drive", "tu drive" o
    # keywords Drive-específicas (planilla, spreadsheet, sheet,
    # presentación).
    "necesito un hard drive nuevo",
    "iba manejando el drive-thru",
    # "docs" solo NO debería disparar (es demasiado ambiguo — "los docs
    # de la API"). Se eliminó del regex final; el LLM puede disparar
    # drive_search vía tool-decide loop si el contexto lo amerita.
    "los docs de la API",
])
def test_pre_router_drive_guarded_false_positives(query: str):
    matched = [n for n, _ in _detect_tool_intent(query)]
    assert "drive_search" not in matched, (
        f"Query {query!r} NO debería haber enganchado drive_search, matched={matched}"
    )


# ── 5. `_format_drive_block` renderer ───────────────────────────────────────


def test_format_drive_block_happy_path():
    raw = json.dumps({
        "tokens": ["alexis", "macbook"],
        "query_used": "alexis macbook",
        "files": [{
            "name": "Venta Macbook.md",
            "mime_label": "Doc",
            "modified": "2026-04-15T10:30:00.000Z",
            "link": "https://docs.google.com/document/d/abc/edit",
            "body": "Alexis adeuda USD 500 por la Macbook Pro M1",
        }],
    })
    out = _format_drive_block(raw)
    # Header incluye la query usada — signal explícito de qué se buscó.
    assert out.startswith("### Google Drive (búsqueda: alexis macbook)")
    assert "**Venta Macbook.md**" in out
    assert "(Doc)" in out
    assert "modificado 2026-04-15" in out
    assert "[abrir](https://docs.google.com/document/d/abc/edit)" in out
    assert "USD 500" in out


def test_format_drive_block_empty_with_tokens():
    """Sin files pero con tokens válidos: mensaje explícito 'ningún archivo
    matchea'. Importante para que el LLM no invente contenido."""
    raw = json.dumps({
        "tokens": ["alexis", "macbook"],
        "query_used": "alexis macbook",
        "files": [],
    })
    out = _format_drive_block(raw)
    assert "Sin resultados" in out
    assert "'alexis'" in out or "alexis" in out
    assert "macbook" in out


def test_format_drive_block_no_credentials():
    raw = json.dumps({
        "tokens": ["alexis"],
        "query_used": "alexis",
        "files": [],
        "error": "no_google_credentials",
    })
    out = _format_drive_block(raw)
    assert "auth de Drive no configurada" in out


def test_format_drive_block_api_error():
    raw = json.dumps({
        "tokens": ["alexis"],
        "query_used": "alexis",
        "files": [],
        "error": "search_failed: 429 rate limited",
    })
    out = _format_drive_block(raw)
    assert "falló la API de Drive" in out


def test_format_drive_block_wired_into_dispatcher():
    """`_format_forced_tool_output(name='drive_search', ...)` despacha
    al renderer de drive — importante para que el pre-router pueda
    renderear el tool output como markdown en vez de raw JSON."""
    raw = json.dumps({
        "tokens": ["alexis"],
        "query_used": "alexis",
        "files": [{
            "name": "Doc.doc",
            "mime_label": "Doc",
            "modified": "",
            "link": "",
            "body": "contenido X",
        }],
    })
    dispatched = _format_forced_tool_output("drive_search", raw)
    direct = _format_drive_block(raw)
    assert dispatched == direct


# ── 6. Empty-state detection + source-intent hint ──────────────────────────


def test_is_empty_drive_search_no_files():
    raw = json.dumps({"tokens": ["x"], "query_used": "x", "files": []})
    assert _is_empty_tool_output("drive_search", raw) is True


def test_is_empty_drive_search_with_files():
    raw = json.dumps({
        "tokens": ["x"], "query_used": "x",
        "files": [{"name": "a.doc", "mime_label": "Doc", "body": "..."}],
    })
    assert _is_empty_tool_output("drive_search", raw) is False


def test_is_empty_drive_search_malformed_returns_false():
    """JSON roto → conservador (False) igual que los otros tools."""
    assert _is_empty_tool_output("drive_search", "not json") is False


def test_source_intent_meta_has_drive_search():
    meta = _SOURCE_INTENT_META["drive_search"]
    assert meta["label"] == "tu Google Drive"
    assert meta["live_section"] == "### Google Drive"
    assert "Drive" in meta["empty_phrase"]


def test_hint_for_drive_search():
    hint = _build_source_intent_hint(["drive_search"])
    assert hint is not None
    assert "tu Google Drive" in hint
    assert "### Google Drive" in hint
    assert "No encontré nada en tu Google Drive" in hint
