"""Tests para `rag.integrations.drive` — leaf ETL de Google Drive.

Surfaces cubiertas:
- `_drive_search_tokens(query, max_tokens)` — extracción de keywords
  filtrando stopwords ES/EN + Drive-self-reference noise.
- `_drive_service()` — auth lazy con OAuth shared con google-drive MCP.
- `_fetch_drive_evidence(now, days, max_items)` — files modificados en
  los últimos N días para morning briefs.
- `_GDRIVE_MIME_LABEL` — pegamento de mimetype → label legible.

Importa desde `rag.integrations.drive` directo (no via `rag.<func>` re-export)
para que el coverage cuente en el módulo correcto.

Notas de mocking:
- `googleapiclient` y `google.oauth2` se mockean con un fake module en
  `sys.modules` cuando hace falta — la lib real no está siempre instalada
  y el `try/except ImportError` de `_drive_service` la maneja bien.
- Se mockea el filesystem usando `monkeypatch.setattr(rag.integrations.drive,
  "GDRIVE_CREDS_DIR", tmp_path)` para no leer creds reales del user.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag.integrations import drive as drive_mod


# ── _drive_search_tokens ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected",
    [
        # Caso del user (Fer F. 2026-04-24): stopwords ES/EN + drive
        # self-reference deben caer.
        (
            "buscá en mi google drive cuánto adeuda alexis de la macbook pro",
            ["adeuda", "alexis", "macbook", "pro"],
        ),
        # Solo stopwords → vacío.
        ("buscá en mi drive", []),
        # Puntuación + caps → normaliza.
        ("¿Cuánto me DEBE Alexis?", ["debe", "alexis"]),
        # Tokens cortos (<2 chars) descartados; "cd" (2) sobrevive.
        ("buscá a, b, cd en drive", ["cd"]),
        # Dedup preservando orden.
        ("alexis alexis venturino", ["alexis", "venturino"]),
    ],
)
def test_drive_search_tokens_param(query: str, expected: list[str]):
    assert drive_mod._drive_search_tokens(query) == expected


def test_drive_search_tokens_respects_max_cap():
    """`max_tokens=N` trunca DESPUÉS de filtrar stopwords. El orden es
    el original — los primeros tokens del input ganan."""
    q = "alfa bravo charlie delta echo foxtrot golf hotel"
    out = drive_mod._drive_search_tokens(q, max_tokens=3)
    assert out == ["alfa", "bravo", "charlie"]


def test_drive_search_tokens_empty_input():
    assert drive_mod._drive_search_tokens("") == []
    assert drive_mod._drive_search_tokens("   \t\n  ") == []


# ── _drive_service: missing creds + missing deps ────────────────────────────


def test_drive_service_returns_none_without_credentials_files(
    monkeypatch, tmp_path,
):
    """Si los archivos `tokens.json` / `gcp-oauth.keys.json` no existen
    en `GDRIVE_CREDS_DIR`, `_drive_service` devuelve `None` (no raisea)."""
    monkeypatch.setattr(drive_mod, "GDRIVE_CREDS_DIR", tmp_path / "no-existe")
    assert drive_mod._drive_service() is None


def test_drive_service_returns_none_when_googleapiclient_missing(
    monkeypatch, tmp_path,
):
    """Si googleapiclient NO está instalado, el `try/except ImportError`
    devuelve `None` silenciosamente. Forzamos el fallo overrideando
    `sys.modules` con un módulo que raisea en attribute access."""
    monkeypatch.setattr(drive_mod, "GDRIVE_CREDS_DIR", tmp_path)
    (tmp_path / "tokens.json").write_text(
        json.dumps({"access_token": "x", "refresh_token": "y"}),
        encoding="utf-8",
    )
    (tmp_path / "gcp-oauth.keys.json").write_text(
        json.dumps({"installed": {"client_id": "id", "client_secret": "s"}}),
        encoding="utf-8",
    )

    # Bloquear googleapiclient — el import dentro de la función levanta.
    import builtins
    original_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name.startswith("googleapiclient") or name.startswith("google.oauth2") \
           or name.startswith("google.auth"):
            raise ImportError(f"forced: {name} not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    assert drive_mod._drive_service() is None


def test_drive_service_loads_credentials_when_present(monkeypatch, tmp_path):
    """Happy path: con creds presentes y google libs disponibles
    (mockeadas), `_drive_service` arma un cliente y lo devuelve."""
    monkeypatch.setattr(drive_mod, "GDRIVE_CREDS_DIR", tmp_path)
    (tmp_path / "tokens.json").write_text(
        json.dumps({
            "access_token": "tok",
            "refresh_token": "rtok",
        }),
        encoding="utf-8",
    )
    (tmp_path / "gcp-oauth.keys.json").write_text(
        json.dumps({
            "installed": {
                "client_id": "client",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
        }),
        encoding="utf-8",
    )

    fake_creds_obj = MagicMock()
    fake_creds_obj.expired = False  # no refresh path
    fake_creds_obj.refresh_token = "rtok"
    fake_creds_obj.token = "tok"

    fake_creds_class = MagicMock(return_value=fake_creds_obj)
    fake_request_class = MagicMock()
    fake_build = MagicMock(return_value=SimpleNamespace(name="DriveService"))

    # Inyectamos los fakes en sys.modules antes de que la función importe.
    fake_oauth2 = SimpleNamespace(
        credentials=SimpleNamespace(Credentials=fake_creds_class),
    )
    fake_auth_transport = SimpleNamespace(
        requests=SimpleNamespace(Request=fake_request_class),
    )
    fake_apiclient = SimpleNamespace(
        discovery=SimpleNamespace(build=fake_build),
    )

    # Pre-stash en sys.modules.
    monkeypatch.setitem(sys.modules, "google.oauth2", fake_oauth2)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials",
                        fake_oauth2.credentials)
    monkeypatch.setitem(sys.modules, "google.auth", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.auth.transport",
                        SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests",
                        fake_auth_transport.requests)
    monkeypatch.setitem(sys.modules, "googleapiclient", fake_apiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery",
                        fake_apiclient.discovery)

    svc = drive_mod._drive_service()
    assert svc is not None
    fake_creds_class.assert_called_once()
    fake_build.assert_called_once_with(
        "drive", "v3", credentials=fake_creds_obj, cache_discovery=False,
    )


# ── _fetch_drive_evidence ──────────────────────────────────────────────────


def test_fetch_drive_evidence_no_service_returns_empty(monkeypatch):
    """Sin cliente Drive → `{}`. Silent-fail, NUNCA raisea."""
    monkeypatch.setattr(drive_mod, "_drive_service", lambda: None)
    out = drive_mod._fetch_drive_evidence(datetime.now(), days=5, max_items=5)
    assert out == {}


def test_fetch_drive_evidence_happy_path_parses_files(monkeypatch):
    """Happy path: el servicio devuelve 3 archivos con mimes distintos;
    `_fetch_drive_evidence` los traduce a la shape interna del brief."""
    now = datetime(2026, 4, 29, 18, 0, 0, tzinfo=timezone.utc)
    files = [
        {
            "id": "f1",
            "name": "Plan Q2.docx",
            "modifiedTime": "2026-04-28T10:00:00.000Z",
            "webViewLink": "https://docs.google.com/document/d/f1/edit",
            "mimeType": "application/vnd.google-apps.document",
            "lastModifyingUser": {"displayName": "Fer F."},
        },
        {
            "id": "f2",
            "name": "Ventas.xlsx",
            "modifiedTime": "2026-04-27T08:30:00.000Z",
            "webViewLink": "https://docs.google.com/spreadsheets/d/f2/edit",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "lastModifyingUser": {"displayName": "Otro"},
        },
        {
            "id": "f3",
            "name": "raro.bin",
            "modifiedTime": "2026-04-26T08:30:00.000Z",
            "webViewLink": "",
            "mimeType": "application/octet-stream",
            # lastModifyingUser ausente — defensivo
        },
    ]

    fake_resp = SimpleNamespace(execute=lambda: {"files": files})
    fake_files_api = SimpleNamespace(list=lambda **_: fake_resp)
    fake_svc = SimpleNamespace(files=lambda: fake_files_api)
    monkeypatch.setattr(drive_mod, "_drive_service", lambda: fake_svc)

    out = drive_mod._fetch_drive_evidence(now, days=5, max_items=5)
    assert out["window_days"] == 5
    assert len(out["files"]) == 3

    f0 = out["files"][0]
    assert f0["name"] == "Plan Q2.docx"
    assert f0["mime_label"] == "Doc"
    assert f0["link"].startswith("https://docs.google.com/document/")
    assert f0["modifier"] == "Fer F."
    assert isinstance(f0["days_ago"], float)

    # Sheet → Sheet label
    assert out["files"][1]["mime_label"] == "Sheet"
    # Mime no mapeado → fallback "archivo"
    assert out["files"][2]["mime_label"] == "archivo"
    # Default modifier "" cuando lastModifyingUser ausente.
    assert out["files"][2]["modifier"] == ""


def test_fetch_drive_evidence_api_exception_silent(monkeypatch):
    """Si la llamada al API levanta cualquier cosa (rate-limit, network,
    auth expirado), `_fetch_drive_evidence` devuelve `{}` y sigue."""
    class _BoomFiles:
        def list(self, **_):
            raise RuntimeError("403 quota exceeded")

    fake_svc = SimpleNamespace(files=lambda: _BoomFiles())
    monkeypatch.setattr(drive_mod, "_drive_service", lambda: fake_svc)

    out = drive_mod._fetch_drive_evidence(
        datetime.now(timezone.utc), days=3, max_items=5,
    )
    assert out == {}


def test_fetch_drive_evidence_handles_unparseable_modifiedtime(monkeypatch):
    """Si `modifiedTime` viene malformado o ausente, days_ago defaultea
    a 0.0 (no crashea — el brief lo filtra después)."""
    now = datetime(2026, 4, 29, 18, 0, 0, tzinfo=timezone.utc)
    files = [
        {
            "id": "f1",
            "name": "Sin time.docx",
            # modifiedTime missing entirely
            "mimeType": "application/vnd.google-apps.document",
        },
        {
            "id": "f2",
            "name": "Time roto.docx",
            "modifiedTime": "esto-no-es-rfc3339",
            "mimeType": "application/vnd.google-apps.document",
        },
    ]
    fake_resp = SimpleNamespace(execute=lambda: {"files": files})
    fake_files_api = SimpleNamespace(list=lambda **_: fake_resp)
    monkeypatch.setattr(
        drive_mod, "_drive_service",
        lambda: SimpleNamespace(files=lambda: fake_files_api),
    )

    out = drive_mod._fetch_drive_evidence(now, days=5, max_items=5)
    assert len(out["files"]) == 2
    for f in out["files"]:
        assert f["days_ago"] == 0.0


# ── _GDRIVE_MIME_LABEL contract ────────────────────────────────────────────


def test_gdrive_mime_label_known_keys_present():
    """Sanity: los 3 mimes que más ve un usuario típico (Doc/Sheet/Slide)
    tienen entries — si alguien los borra rompe el rendering del brief."""
    assert drive_mod._GDRIVE_MIME_LABEL[
        "application/vnd.google-apps.document"
    ] == "Doc"
    assert drive_mod._GDRIVE_MIME_LABEL[
        "application/vnd.google-apps.spreadsheet"
    ] == "Sheet"
    assert drive_mod._GDRIVE_MIME_LABEL[
        "application/vnd.google-apps.presentation"
    ] == "Slide"


# ── Invariants del módulo ───────────────────────────────────────────────────


def test_gdrive_scopes_is_readonly_only():
    """Phase 1.b cross-source corpus contract: el scope NUNCA debe
    ampliarse de `drive.readonly` sin override explícito del user.
    Este test es un canary que falla si alguien agrega scopes write."""
    assert drive_mod.GDRIVE_SCOPES == [
        "https://www.googleapis.com/auth/drive.readonly",
    ]


def test_search_stopwords_includes_drive_self_reference():
    """`_GDRIVE_SEARCH_STOPWORDS` debe filtrar palabras que el user
    invariablemente usa cuando habla DE Drive (drive, gdrive, planilla,
    etc.) — sino los queries dilurían matches con esas palabras
    omnipresentes."""
    expected = {"drive", "gdrive", "google", "doc", "docs", "planilla",
                "sheet", "presentación", "archivo", "file"}
    for word in expected:
        assert word in drive_mod._GDRIVE_SEARCH_STOPWORDS, (
            f"stopword esperada faltó: {word}"
        )
