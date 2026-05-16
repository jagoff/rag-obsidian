from __future__ import annotations

import httpx
import pytest

from rag.integrations.whatsapp import bridge_client as bc


class _DummyClient:
    def __init__(self, *args, response=None, error=None, **kwargs):
        self.response = response
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def request(self, method, path, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def test_bridge_client_wraps_transport_error(monkeypatch):
    req = httpx.Request("POST", "http://localhost:8088/api/send")
    err = httpx.ConnectError("connection refused", request=req)
    monkeypatch.setattr(bc.httpx, "Client", lambda *a, **kw: _DummyClient(error=err))

    with pytest.raises(bc.BridgeError) as raised:
        bc.send_text("54911@s.whatsapp.net", "hola")

    assert raised.value.status is None
    assert "bridge unreachable on /api/send" in str(raised.value)


def test_bridge_client_keeps_http_status_on_rejected_response(monkeypatch):
    req = httpx.Request("POST", "http://localhost:8088/api/send")
    resp = httpx.Response(503, text="bridge down", request=req)
    monkeypatch.setattr(bc.httpx, "Client", lambda *a, **kw: _DummyClient(response=resp))

    with pytest.raises(bc.BridgeError) as raised:
        bc.send_text("54911@s.whatsapp.net", "hola")

    assert raised.value.status == 503
    assert "bridge 503 on /api/send" in str(raised.value)
