"""Tests para `rag.integrations.gmail` — leaf ETL de Gmail.

Surfaces cubiertas:
- `_gmail_service()` — primary path (`_load_google_credentials`) + legacy
  fallback (`~/.gmail-mcp/credentials.json`). Silent-fail cuando ambos
  paths fallan o cuando google libs no están instaladas.
- `_gmail_send_service()` — hermana de `_gmail_service` pero solo via
  legacy path (scope `gmail.modify` para enviar). Silent-fail si no hay
  creds.
- `_gmail_thread_last_meta(svc, thread_id)` — fetch del último mensaje
  de un thread; happy path + None on exception.
- `_fetch_gmail_evidence(now)` — bundle de unread + starred + awaiting +
  recent. Verifica parseo del response simulado del API + dedup contra
  bucket previo.
- `_fetch_gmail_today(now, max_items)` — corte por today_start_ms +
  filtrado client-side cuando `newer_than:1d` mete mails de ayer.

Mocking strategy:
- Igual approach que `tests/test_integration_drive.py`: `monkeypatch.setitem`
  sobre `sys.modules` para `googleapiclient.*` y `google.oauth2.*` cuando
  hace falta forzar el import path. Para tests de happy path mockeamos el
  servicio entero con `SimpleNamespace` y verificamos que las llamadas API
  se hagan con los args correctos.
- `_load_google_credentials` y `_silent_log` se monkeypatchean en `rag`
  porque viven ahí; `_gmail_service` los importa con `from rag import …`
  dentro del cuerpo, así que el patch en `rag.<func>` se respeta a runtime.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag.integrations import gmail as gmail_mod


# ── _gmail_service: missing deps + missing creds ────────────────────────────


def test_gmail_service_returns_none_when_googleapiclient_missing(monkeypatch):
    """Si `googleapiclient` no está instalado, el primer `try/except
    ImportError` short-circuitea y devuelve None sin tocar ningún
    creds path."""
    import builtins
    original_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name.startswith("googleapiclient"):
            raise ImportError(f"forced: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    assert gmail_mod._gmail_service() is None


def test_gmail_service_returns_none_when_no_creds_anywhere(
    monkeypatch, tmp_path,
):
    """Sin token primario (`_load_google_credentials → None`) y sin archivos
    legacy en `~/.gmail-mcp/`, el servicio devuelve None silenciosamente.
    Caller maneja el silent-fail."""
    import rag
    monkeypatch.setattr(rag, "_load_google_credentials", lambda **_kw: None)
    monkeypatch.setattr(gmail_mod, "GMAIL_CREDS_DIR", tmp_path / "no-mcp")
    fake_apiclient = SimpleNamespace(
        discovery=SimpleNamespace(build=lambda *a, **kw: object()),
    )
    monkeypatch.setitem(sys.modules, "googleapiclient", fake_apiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery",
                        fake_apiclient.discovery)
    monkeypatch.setitem(
        sys.modules, "google.oauth2",
        SimpleNamespace(credentials=SimpleNamespace(Credentials=MagicMock())),
    )
    monkeypatch.setitem(
        sys.modules, "google.oauth2.credentials",
        SimpleNamespace(Credentials=MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules, "google.auth", SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules, "google.auth.transport", SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests",
        SimpleNamespace(Request=MagicMock()),
    )
    assert gmail_mod._gmail_service() is None


def test_gmail_service_happy_path_via_primary_token(monkeypatch):
    """Path canónico: `_load_google_credentials()` devuelve creds válidos
    → armamos cliente Gmail y devolvemos. NO se toca el legacy path."""
    import rag
    fake_creds = SimpleNamespace(valid=True)
    monkeypatch.setattr(
        rag, "_load_google_credentials", lambda **_kw: fake_creds,
    )

    fake_build = MagicMock(return_value=SimpleNamespace(name="GmailService"))
    fake_apiclient = SimpleNamespace(
        discovery=SimpleNamespace(build=fake_build),
    )
    monkeypatch.setitem(sys.modules, "googleapiclient", fake_apiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery",
                        fake_apiclient.discovery)

    svc = gmail_mod._gmail_service()
    assert svc is not None
    fake_build.assert_called_once_with(
        "gmail", "v1", credentials=fake_creds, cache_discovery=False,
    )


def test_gmail_service_falls_back_to_legacy_with_refresh(
    monkeypatch, tmp_path,
):
    """Primary path falla (`_load_google_credentials → None` o creds.valid
    False); el módulo cae al legacy path en `~/.gmail-mcp/credentials.json`,
    refresca el access_token + persiste de vuelta al disco, y arma cliente."""
    import rag
    monkeypatch.setattr(rag, "_load_google_credentials", lambda **_kw: None)

    monkeypatch.setattr(gmail_mod, "GMAIL_CREDS_DIR", tmp_path)
    creds_path = tmp_path / "credentials.json"
    oauth_path = tmp_path / "gcp-oauth.keys.json"
    creds_path.write_text(
        json.dumps({"access_token": "stale", "refresh_token": "rt"}),
        encoding="utf-8",
    )
    oauth_path.write_text(
        json.dumps({"installed": {
            "client_id": "cid", "client_secret": "csec",
            "token_uri": "https://oauth2.googleapis.com/token",
        }}),
        encoding="utf-8",
    )

    refreshed_creds = MagicMock()
    refreshed_creds.expired = True
    refreshed_creds.refresh_token = "rt"
    refreshed_creds.token = "fresh"

    def _refresh_side_effect(_request):
        refreshed_creds.token = "fresh"

    refreshed_creds.refresh.side_effect = _refresh_side_effect

    fake_creds_class = MagicMock(return_value=refreshed_creds)
    fake_request_class = MagicMock()
    fake_build = MagicMock(return_value=SimpleNamespace(name="LegacyGmail"))

    monkeypatch.setitem(
        sys.modules, "google.oauth2",
        SimpleNamespace(credentials=SimpleNamespace(Credentials=fake_creds_class)),
    )
    monkeypatch.setitem(
        sys.modules, "google.oauth2.credentials",
        SimpleNamespace(Credentials=fake_creds_class),
    )
    monkeypatch.setitem(sys.modules, "google.auth", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.auth.transport", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests",
        SimpleNamespace(Request=fake_request_class),
    )
    fake_apiclient = SimpleNamespace(
        discovery=SimpleNamespace(build=fake_build),
    )
    monkeypatch.setitem(sys.modules, "googleapiclient", fake_apiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery",
                        fake_apiclient.discovery)

    svc = gmail_mod._gmail_service()
    assert svc is not None
    refreshed_creds.refresh.assert_called_once()
    persisted = json.loads(creds_path.read_text(encoding="utf-8"))
    assert persisted["access_token"] == "fresh"


# ── _gmail_send_service ─────────────────────────────────────────────────────


def test_gmail_send_service_returns_none_without_legacy_files(
    monkeypatch, tmp_path,
):
    """`_gmail_send_service` solo usa el legacy path (scope `gmail.modify`).
    Si los archivos no existen → None silenciosamente."""
    monkeypatch.setattr(gmail_mod, "GMAIL_CREDS_DIR", tmp_path / "no-existe")
    monkeypatch.setitem(
        sys.modules, "googleapiclient",
        SimpleNamespace(discovery=SimpleNamespace(build=MagicMock())),
    )
    monkeypatch.setitem(
        sys.modules, "googleapiclient.discovery",
        SimpleNamespace(build=MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules, "google.oauth2.credentials",
        SimpleNamespace(Credentials=MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests",
        SimpleNamespace(Request=MagicMock()),
    )
    assert gmail_mod._gmail_send_service() is None


def test_gmail_send_service_happy_path(monkeypatch, tmp_path):
    """Happy path con creds que NO necesitan refresh (creds.expired=False).
    Verificamos que NO se llama refresh y que el cliente se construye con
    el scope correcto."""
    monkeypatch.setattr(gmail_mod, "GMAIL_CREDS_DIR", tmp_path)
    (tmp_path / "credentials.json").write_text(
        json.dumps({"access_token": "ok", "refresh_token": "rt"}),
        encoding="utf-8",
    )
    (tmp_path / "gcp-oauth.keys.json").write_text(
        json.dumps({"installed": {"client_id": "x", "client_secret": "y"}}),
        encoding="utf-8",
    )

    fake_creds = MagicMock()
    fake_creds.expired = False
    fake_creds.refresh_token = "rt"
    fake_creds_class = MagicMock(return_value=fake_creds)

    fake_build = MagicMock(return_value=SimpleNamespace(name="SendSvc"))
    monkeypatch.setitem(
        sys.modules, "googleapiclient",
        SimpleNamespace(discovery=SimpleNamespace(build=fake_build)),
    )
    monkeypatch.setitem(
        sys.modules, "googleapiclient.discovery",
        SimpleNamespace(build=fake_build),
    )
    monkeypatch.setitem(
        sys.modules, "google.oauth2.credentials",
        SimpleNamespace(Credentials=fake_creds_class),
    )
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests",
        SimpleNamespace(Request=MagicMock()),
    )

    svc = gmail_mod._gmail_send_service()
    assert svc is not None
    fake_creds.refresh.assert_not_called()
    fake_build.assert_called_once_with(
        "gmail", "v1", credentials=fake_creds, cache_discovery=False,
    )


# ── _gmail_thread_last_meta ─────────────────────────────────────────────────


def test_gmail_thread_last_meta_happy_path():
    """Happy path: el API devuelve el thread con headers + snippet, y
    mapeamos a la shape interna `{subject, from, snippet, internal_date_ms}`."""
    fake_thread = {
        "messages": [
            {"id": "m1", "internalDate": "1000"},
            {
                "id": "m2",
                "internalDate": "2500",
                "snippet": "Hola, gracias por la propuesta!",
                "payload": {"headers": [
                    {"name": "Subject", "value": "Re: Propuesta"},
                    {"name": "From", "value": "Juan <juan@x.com>"},
                    {"name": "Date", "value": "Mon, 01 Apr 2026 10:00:00 +0000"},
                ]},
            },
        ],
    }
    fake_get = SimpleNamespace(execute=lambda: fake_thread)
    fake_threads_api = SimpleNamespace(
        get=lambda **_kw: fake_get,
    )
    fake_users = SimpleNamespace(threads=lambda: fake_threads_api)
    svc = SimpleNamespace(users=lambda: fake_users)

    out = gmail_mod._gmail_thread_last_meta(svc, "thr1")
    assert out is not None
    assert out["subject"] == "Re: Propuesta"
    assert out["from"] == "Juan <juan@x.com>"
    assert out["snippet"].startswith("Hola, gracias")
    assert out["internal_date_ms"] == 2500


def test_gmail_thread_last_meta_returns_none_on_api_exception():
    """Si la API levanta cualquier cosa → None (caller silent-fails)."""
    class _Boom:
        def get(self, **_kw):
            class _R:
                def execute(self_):
                    raise RuntimeError("403 quota")
            return _R()
    fake_users = SimpleNamespace(threads=lambda: _Boom())
    svc = SimpleNamespace(users=lambda: fake_users)
    assert gmail_mod._gmail_thread_last_meta(svc, "thr1") is None


def test_gmail_thread_last_meta_empty_messages_returns_none():
    """Edge: thread existe pero `messages` vacío → None."""
    fake_get = SimpleNamespace(execute=lambda: {"messages": []})
    fake_threads_api = SimpleNamespace(get=lambda **_kw: fake_get)
    svc = SimpleNamespace(users=lambda: SimpleNamespace(
        threads=lambda: fake_threads_api,
    ))
    assert gmail_mod._gmail_thread_last_meta(svc, "x") is None


def test_gmail_thread_last_meta_handles_malformed_internal_date():
    """`internalDate` que no parsea a int → 0 sin raisear."""
    fake_thread = {
        "messages": [
            {
                "id": "m1",
                "internalDate": "no-es-int",
                "snippet": "x",
                "payload": {"headers": [{"name": "Subject", "value": "S"}]},
            },
        ],
    }
    fake_get = SimpleNamespace(execute=lambda: fake_thread)
    fake_threads_api = SimpleNamespace(get=lambda **_kw: fake_get)
    svc = SimpleNamespace(users=lambda: SimpleNamespace(
        threads=lambda: fake_threads_api,
    ))
    out = gmail_mod._gmail_thread_last_meta(svc, "thr")
    assert out is not None
    assert out["internal_date_ms"] == 0


# ── _fetch_gmail_evidence ───────────────────────────────────────────────────


def test_fetch_gmail_evidence_returns_empty_when_no_service(monkeypatch):
    """Sin cliente Gmail → `{}`. Silent-fail, nunca raisea."""
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: None)
    out = gmail_mod._fetch_gmail_evidence(datetime.now(timezone.utc))
    assert out == {}


def test_fetch_gmail_evidence_returns_empty_when_get_profile_fails(
    monkeypatch,
):
    """Si `getProfile` levanta (creds revocados, network) → `{}`."""
    class _BoomProfile:
        def getProfile(self, **_kw):
            class _R:
                def execute(self_):
                    raise RuntimeError("auth error")
            return _R()
    fake_svc = SimpleNamespace(users=lambda: _BoomProfile())
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)
    out = gmail_mod._fetch_gmail_evidence(datetime.now(timezone.utc))
    assert out == {}


def test_fetch_gmail_evidence_parses_buckets_with_dedup(monkeypatch):
    """Happy path: el servicio devuelve unread_count + threads para
    starred / awaiting / recent, y `_fetch_gmail_evidence` parsea cada
    bucket con dedup contra los previos (un thread starred no aparece
    también en recent)."""
    now = datetime(2026, 4, 29, 18, 0, 0, tzinfo=timezone.utc)

    label_resp = SimpleNamespace(execute=lambda: {"threadsUnread": 7})
    labels_api = SimpleNamespace(get=lambda **_kw: label_resp)

    thread_responses: dict[str, list[dict]] = {
        "is:starred in:inbox newer_than:7d": [{"id": "T_STAR"}],
        "in:inbox newer_than:14d older_than:3d "
        "-category:promotions -category:social "
        "-category:updates -category:forums": [
            {"id": "T_AWAIT_OK"},
            {"id": "T_AWAIT_ME"},
        ],
        "in:inbox newer_than:30d": [
            {"id": "T_STAR"},
            {"id": "T_RECENT"},
        ],
    }

    def _list_factory(**kwargs):
        q = kwargs.get("q", "")
        threads = thread_responses.get(q, [])

        class _Resp:
            def execute(self_):
                return {"threads": threads}
        return _Resp()

    threads_api = SimpleNamespace(list=_list_factory, get=None)

    profile_resp = SimpleNamespace(
        execute=lambda: {"emailAddress": "fer@example.com"},
    )

    thread_metas: dict[str, dict] = {
        "T_STAR": {
            "messages": [{
                "internalDate": "1700000000000",
                "snippet": "starred snippet",
                "payload": {"headers": [
                    {"name": "Subject", "value": "Mail starred"},
                    {"name": "From", "value": "amigo@x.com"},
                ]},
            }],
        },
        "T_AWAIT_OK": {
            "messages": [{
                "internalDate": str(int(
                    (now.timestamp() - 5 * 86400) * 1000,
                )),
                "snippet": "awaiting snippet",
                "payload": {"headers": [
                    {"name": "Subject", "value": "Pending reply"},
                    {"name": "From", "value": "cliente@y.com"},
                ]},
            }],
        },
        "T_AWAIT_ME": {
            "messages": [{
                "internalDate": "1700000000000",
                "snippet": "yo respondi",
                "payload": {"headers": [
                    {"name": "Subject", "value": "Conversation"},
                    {"name": "From", "value": "Fer F. <fer@example.com>"},
                ]},
            }],
        },
        "T_RECENT": {
            "messages": [{
                "internalDate": "1730000000000",
                "snippet": "recent stuff",
                "payload": {"headers": [
                    {"name": "Subject", "value": "Latest"},
                    {"name": "From", "value": "noreply@z.com"},
                ]},
            }],
        },
    }

    def _threads_get(**kwargs):
        tid = kwargs.get("id", "")
        meta = thread_metas.get(tid, {"messages": []})

        class _R:
            def execute(self_):
                return meta
        return _R()

    threads_api.get = _threads_get

    users_api = SimpleNamespace(
        getProfile=lambda **_kw: profile_resp,
        labels=lambda: labels_api,
        threads=lambda: threads_api,
    )
    fake_svc = SimpleNamespace(users=lambda: users_api)
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)

    out = gmail_mod._fetch_gmail_evidence(now)
    assert out["unread_count"] == 7
    assert len(out["starred"]) == 1
    assert out["starred"][0]["thread_id"] == "T_STAR"
    assert len(out["awaiting_reply"]) == 1
    assert out["awaiting_reply"][0]["thread_id"] == "T_AWAIT_OK"
    assert out["awaiting_reply"][0]["days_old"] > 0
    assert len(out["recent"]) == 1
    assert out["recent"][0]["thread_id"] == "T_RECENT"


def test_fetch_gmail_evidence_continues_when_unread_count_fails(monkeypatch):
    """Si `labels.get(INBOX)` falla pero el resto del API anda, devolvemos
    `unread_count=0` + el resto de buckets parseados (cada bucket tiene su
    propio try/except)."""
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)

    class _BoomLabels:
        def get(self, **_kw):
            class _R:
                def execute(self_):
                    raise RuntimeError("INBOX label not found")
            return _R()

    threads_api = SimpleNamespace(
        list=lambda **_kw: SimpleNamespace(execute=lambda: {"threads": []}),
        get=lambda **_kw: SimpleNamespace(execute=lambda: {"messages": []}),
    )
    users_api = SimpleNamespace(
        getProfile=lambda **_kw: SimpleNamespace(
            execute=lambda: {"emailAddress": "x@y.com"},
        ),
        labels=lambda: _BoomLabels(),
        threads=lambda: threads_api,
    )
    fake_svc = SimpleNamespace(users=lambda: users_api)
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)

    out = gmail_mod._fetch_gmail_evidence(now)
    assert out["unread_count"] == 0
    assert out["starred"] == []
    assert out["awaiting_reply"] == []
    assert out["recent"] == []


# ── _fetch_gmail_today ──────────────────────────────────────────────────────


def test_fetch_gmail_today_returns_empty_when_no_service(monkeypatch):
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: None)
    out = gmail_mod._fetch_gmail_today(datetime.now(timezone.utc))
    assert out == []


def test_fetch_gmail_today_filters_yesterday_messages(monkeypatch):
    """`newer_than:1d` puede mezclar mensajes de las 22hs de ayer cuando el
    cron corre a las 22hs. El filter cliente-side recorta exacto al 00:00
    local — verificamos que un mail del día anterior se descarte."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
    today_start_ms = int(now.replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).timestamp() * 1000)

    threads_resp = {
        "threads": [
            {"id": "T_TODAY"},
            {"id": "T_YESTERDAY"},
        ],
    }
    metas: dict[str, dict] = {
        "T_TODAY": {"messages": [{
            "internalDate": str(today_start_ms + 3600_000),
            "snippet": "snip-hoy",
            "payload": {"headers": [
                {"name": "Subject", "value": "Hoy temprano"},
                {"name": "From", "value": "x@x.com"},
            ]},
        }]},
        "T_YESTERDAY": {"messages": [{
            "internalDate": str(today_start_ms - 3600_000),
            "snippet": "snip-ayer",
            "payload": {"headers": [
                {"name": "Subject", "value": "Ayer 23hs"},
                {"name": "From", "value": "x@x.com"},
            ]},
        }]},
    }

    threads_api = SimpleNamespace(
        list=lambda **_kw: SimpleNamespace(execute=lambda: threads_resp),
        get=lambda **kw: SimpleNamespace(
            execute=lambda: metas[kw["id"]],
        ),
    )
    fake_svc = SimpleNamespace(users=lambda: SimpleNamespace(
        threads=lambda: threads_api,
    ))
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)

    out = gmail_mod._fetch_gmail_today(now, max_items=10)
    assert len(out) == 1
    assert out[0]["thread_id"] == "T_TODAY"
    assert out[0]["subject"] == "Hoy temprano"


def test_fetch_gmail_today_caps_to_max_items(monkeypatch):
    """Más threads que `max_items` → la función rompe el loop tras llegar
    al cap (NO procesa los restantes)."""
    now = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
    today_ms = int(now.replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).timestamp() * 1000)

    threads_list = [{"id": f"T{i}"} for i in range(20)]

    def _meta_for(tid: str) -> dict:
        return {"messages": [{
            "internalDate": str(today_ms + 1000 * int(tid[1:])),
            "snippet": f"snip-{tid}",
            "payload": {"headers": [
                {"name": "Subject", "value": f"Subject {tid}"},
                {"name": "From", "value": "x@x.com"},
            ]},
        }]}

    threads_api = SimpleNamespace(
        list=lambda **_kw: SimpleNamespace(
            execute=lambda: {"threads": threads_list},
        ),
        get=lambda **kw: SimpleNamespace(
            execute=lambda: _meta_for(kw["id"]),
        ),
    )
    fake_svc = SimpleNamespace(users=lambda: SimpleNamespace(
        threads=lambda: threads_api,
    ))
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)

    out = gmail_mod._fetch_gmail_today(now, max_items=5)
    assert len(out) == 5


def test_fetch_gmail_today_silent_on_api_exception(monkeypatch):
    """Si `threads.list` levanta → `[]`, sin propagar."""
    class _Boom:
        def list(self, **_kw):
            class _R:
                def execute(self_):
                    raise RuntimeError("rate limit")
            return _R()

    fake_svc = SimpleNamespace(users=lambda: SimpleNamespace(
        threads=lambda: _Boom(),
    ))
    monkeypatch.setattr(gmail_mod, "_gmail_service", lambda: fake_svc)
    out = gmail_mod._fetch_gmail_today(datetime.now(timezone.utc))
    assert out == []


# ── Invariants ──────────────────────────────────────────────────────────────


def test_gmail_scopes_includes_modify_and_settings():
    """Sanity: la lista de scopes legacy NO debe achicarse a sólo readonly
    sin update explícito porque `_gmail_send_service` necesita `gmail.modify`
    para enviar. Si alguien aprieta el scope, este canary falla."""
    assert "https://www.googleapis.com/auth/gmail.modify" in gmail_mod.GMAIL_SCOPES
    assert (
        "https://www.googleapis.com/auth/gmail.settings.basic"
        in gmail_mod.GMAIL_SCOPES
    )


def test_gmail_creds_dir_is_under_home():
    """`GMAIL_CREDS_DIR` apunta al directorio compartido con el gmail-mcp
    NPM package; debe estar bajo `~/.gmail-mcp` para que el setup
    documentado funcione."""
    assert str(gmail_mod.GMAIL_CREDS_DIR).endswith(".gmail-mcp")
