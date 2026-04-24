"""Tests del /status page + /api/status endpoint (2026-04-24).

Regresiones que atrapan:
  - GET /status devuelve el HTML y GET /api/status devuelve JSON con el
    shape esperado (overall, counts, categories con services anidados).
    Si alguien rompe el payload shape, el frontend se queda en blanco.
  - Los helpers de grading (_status_grade_daemon / _status_grade_scheduled)
    mapean correctamente los outputs de launchctl print a ok/warn/down.
    Testeado con fixtures en memoria — no dependemos de que haya
    launchctl real ni de los plists del usuario.
  - Parser de `_launchctl_print_fields` extrae sólo las keys top-level
    (un tab) y descarta las nested (dos tabs). Un bug acá confunde
    nested `state = active` de endpoints con el state top-level.
  - Los 3 HTML (home, chat/index, dashboard) tienen el link a /status.
    Si alguien edita un HTML y se olvida del link, el user queda
    navegando sin poder llegar a la página.
  - status.html + status.js existen en disco con el wiring PWA mínimo
    (manifest, register-sw, theme init).

No testeamos:
  - El JS del frontend (eso requiere Playwright — verificado a mano).
  - Los subprocess reales de launchctl (el lunchd de CI no los tiene).
  - El caching — `/api/status` con cache hot vs cold. La lógica es
    triviale (timestamp + ttl) y testearla en fake-time complicaría
    el test sin gain real.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web.server as _server


_STATIC_DIR = Path(_server.STATIC_DIR)
_client = TestClient(_server.app)


# ── Endpoint shape ───────────────────────────────────────────────────

def test_status_page_served():
    """GET /status → 200 HTML (el mismo FileResponse pattern que dashboard)."""
    resp = _client.get("/status")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # Sanity checks sobre el contenido del HTML.
    assert "rag" in body and "status" in body
    assert "/static/status.js" in body
    assert "manifest.webmanifest" in body
    assert "register-sw.js" in body


def test_api_status_payload_shape():
    """GET /api/status → 200 JSON con overall + counts + categories."""
    resp = _client.get("/api/status?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert "generated_at" in d
    assert d["overall"] in {"ok", "degraded", "down"}
    assert isinstance(d["counts"], dict)
    for k in ("ok", "warn", "down"):
        assert k in d["counts"]
        assert isinstance(d["counts"][k], int)
    assert isinstance(d["categories"], list)
    assert len(d["categories"]) > 0
    # Al menos una categoría "core" con al menos el web-self probe.
    core = next((c for c in d["categories"] if c["id"] == "core"), None)
    assert core is not None, "falta categoría core"
    assert any(s.get("id") == "web-self" for s in core["services"])


def test_api_status_service_shape():
    """Cada servicio tiene id, name, kind, status, detail, category."""
    resp = _client.get("/api/status?nocache=1")
    d = resp.json()
    for cat in d["categories"]:
        for svc in cat["services"]:
            for field in ("id", "name", "kind", "status", "detail"):
                assert field in svc, f"{svc} falta {field}"
            assert svc["status"] in {"ok", "warn", "down"}, f"status inválido: {svc['status']}"
            assert svc["kind"] in {"daemon", "scheduled", "probe"}, f"kind inválido: {svc['kind']}"


def test_api_status_web_self_always_ok():
    """El probe `web-self` siempre debería estar ok — si no, el server
    no podría haber respondido al request en primer lugar."""
    resp = _client.get("/api/status?nocache=1")
    d = resp.json()
    core = next(c for c in d["categories"] if c["id"] == "core")
    web = next(s for s in core["services"] if s["id"] == "web-self")
    assert web["status"] == "ok"
    # Detail contiene el pid del proceso actual.
    import os
    assert str(os.getpid()) in web["detail"]


def test_api_status_counts_match_services():
    """Los counts top-level suman los status de todas las services."""
    resp = _client.get("/api/status?nocache=1")
    d = resp.json()
    total_counted = sum(d["counts"].values())
    total_services = sum(len(c["services"]) for c in d["categories"])
    assert total_counted == total_services


# ── Grading helpers ──────────────────────────────────────────────────

def test_grade_daemon_running_is_ok():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "running", "pid": "12345"}):
        r = _server._status_grade_daemon("com.fer.fake", "Fake daemon")
    assert r["status"] == "ok"
    assert r["kind"] == "daemon"
    assert "12345" in r["detail"]


def test_grade_daemon_not_running_is_down():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "last exit code": "0",
                                    "runs": "3"}):
        r = _server._status_grade_daemon("com.fer.fake", "Fake daemon")
    assert r["status"] == "down"


def test_grade_daemon_crashed_surfaces_exit_code():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "last exit code": "78",
                                    "runs": "1"}):
        r = _server._status_grade_daemon("com.fer.fake", "Fake daemon")
    assert r["status"] == "down"
    assert "78" in r["detail"]


def test_grade_daemon_not_loaded_is_down():
    with patch.object(_server, "_launchctl_print_fields", return_value=None):
        r = _server._status_grade_daemon("com.fer.missing", "Missing")
    assert r["status"] == "down"
    assert "no cargado" in r["detail"]


def test_grade_scheduled_last_exit_zero_is_ok():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "runs": "5",
                                    "last exit code": "0"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["status"] == "ok"
    assert "runs 5" in r["detail"]


def test_grade_scheduled_never_exited_zero_runs_is_warn():
    """Un scheduled job loaded pero que aún no corrió = warn (no down)."""
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "runs": "0",
                                    "last exit code": "(never exited)"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["status"] == "warn"
    assert "aún no corrió" in r["detail"]


def test_grade_scheduled_nonzero_exit_is_down():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "runs": "3",
                                    "last exit code": "2"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["status"] == "down"
    assert "exit 2" in r["detail"]


def test_grade_scheduled_running_now_is_ok():
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "running", "pid": "999", "runs": "4"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["status"] == "ok"
    assert "999" in r["detail"]


def test_grade_scheduled_not_loaded_is_warn():
    """Un scheduled no cargado = warn (no down): podría ser desactivado
    intencionalmente por el usuario."""
    with patch.object(_server, "_launchctl_print_fields", return_value=None):
        r = _server._status_grade_scheduled("com.fer.missing", "Missing")
    assert r["status"] == "warn"


# ── Parser de launchctl print ────────────────────────────────────────

def test_launchctl_parser_skips_nested_blocks():
    """El parser debe tomar sólo top-level (un tab) y descartar nested."""
    fake_out = "\n".join([
        "gui/501/com.fer.fake = {",
        "\tactive count = 1",
        "\tstate = running",
        "\tpid = 12345",
        "\tendpoints = {",
        "\t\t\"com.apple.fake\" = {",
        "\t\t\tactive instances = 1",
        "\t\t\tstate = active",   # <- este NO debe sobreescribir el top-level
        "\t\t}",
        "\t}",
        "\tlast exit code = 0",
        "}",
    ])

    import subprocess
    class FakeCompleted:
        returncode = 0
        stdout = fake_out
    with patch.object(subprocess, "run", return_value=FakeCompleted()):
        f = _server._launchctl_print_fields("com.fer.fake")

    assert f is not None
    assert f["state"] == "running"   # top-level, no "active" del nested
    assert f["pid"] == "12345"
    assert f["last exit code"] == "0"


def test_launchctl_parser_returns_none_when_service_missing():
    import subprocess
    class FakeCompleted:
        returncode = 113
        stdout = "Could not find service ..."
    with patch.object(subprocess, "run", return_value=FakeCompleted()):
        f = _server._launchctl_print_fields("com.fer.nonexistent")
    assert f is None


def test_launchctl_parser_handles_timeout():
    """Timeout → None (no raise)."""
    import subprocess
    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="launchctl", timeout=3.0)
    with patch.object(subprocess, "run", side_effect=_raise):
        f = _server._launchctl_print_fields("com.fer.slow", timeout=0.01)
    assert f is None


# ── Nav-link wiring ──────────────────────────────────────────────────

def test_home_page_links_to_status():
    html = (_STATIC_DIR / "home.html").read_text(encoding="utf-8")
    assert 'href="/status"' in html


def test_chat_page_links_to_status():
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'href="/status"' in html


def test_dashboard_page_links_to_status():
    html = (_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")
    assert 'href="/status"' in html


# ── Static files on disk ─────────────────────────────────────────────

def test_status_html_exists_with_pwa_wiring():
    html = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert 'rel="manifest"' in html
    assert "register-sw.js" in html
    assert "/static/status.js" in html
    # Theme init inline para evitar flash de tema.
    assert "rag-theme" in html


def test_status_js_exists():
    js = (_STATIC_DIR / "status.js").read_text(encoding="utf-8")
    # Sanity: fetchea /api/status y tiene auto-refresh.
    assert "/api/status" in js
    assert "setInterval" in js or "setTimeout" in js


# ── Action button payload shape ──────────────────────────────────────

def test_grade_daemon_includes_label_and_running_when_running():
    """daemon running → payload tiene label + loaded=True + running=True."""
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "running", "pid": "777"}):
        r = _server._status_grade_daemon("com.fer.fake", "Fake daemon")
    assert r["label"] == "com.fer.fake"
    assert r["loaded"] is True
    assert r["running"] is True


def test_grade_daemon_includes_label_when_not_loaded():
    """daemon no cargado → loaded=False, running=False (UI no debe ofrecer stop)."""
    with patch.object(_server, "_launchctl_print_fields", return_value=None):
        r = _server._status_grade_daemon("com.fer.fake", "Fake daemon")
    assert r["label"] == "com.fer.fake"
    assert r["loaded"] is False
    assert r["running"] is False


def test_grade_scheduled_never_run_includes_label_loaded_and_running_false():
    """scheduled "aún no corrió" → loaded=True, running=False (UI ofrece start)."""
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "not running",
                                    "runs": "0",
                                    "last exit code": "(never exited)"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["label"] == "com.fer.fake"
    assert r["loaded"] is True
    assert r["running"] is False
    # El detalle "aún no corrió" sigue intacto para que la UI lo pinte.
    assert "aún no corrió" in r["detail"]


def test_grade_scheduled_running_now_includes_label_and_running_true():
    """scheduled corriendo ahora → running=True (UI ofrece stop)."""
    with patch.object(_server, "_launchctl_print_fields",
                      return_value={"state": "running", "pid": "999", "runs": "4"}):
        r = _server._status_grade_scheduled("com.fer.fake", "Fake job")
    assert r["label"] == "com.fer.fake"
    assert r["running"] is True
    assert r["loaded"] is True


# ── /api/status/action endpoint ──────────────────────────────────────

def test_status_action_rejects_invalid_action():
    """action != start|stop → 400."""
    r = _client.post("/api/status/action",
                     json={"label": "com.fer.obsidian-rag-digest",
                           "action": "delete"})
    assert r.status_code == 400
    assert "action" in r.json()["detail"].lower()


def test_status_action_rejects_label_not_in_whitelist():
    """label fuera del catálogo → 400 (defensa principal contra abuso)."""
    r = _client.post("/api/status/action",
                     json={"label": "com.fer.MALICIOUS",
                           "action": "start"})
    assert r.status_code == 400
    assert "whitelisted" in r.json()["detail"].lower()


def test_status_action_rejects_empty_label():
    """label vacío → 400."""
    r = _client.post("/api/status/action",
                     json={"label": "", "action": "start"})
    assert r.status_code == 400


def test_status_action_start_invokes_kickstart():
    """action=start → corre `launchctl kickstart gui/<uid>/<label>`."""
    label = "com.fer.obsidian-rag-digest"  # known to be in catalog
    assert label in _server._status_actionable_labels()

    import subprocess
    captured = {}
    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""
    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return FakeCompleted()
    with patch.object(subprocess, "run", side_effect=fake_run):
        r = _client.post("/api/status/action",
                         json={"label": label, "action": "start"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["action"] == "start"
    assert j["label"] == label
    # Verify launchctl was called with kickstart + gui/<uid>/<label>
    assert captured["cmd"][0] == "/bin/launchctl"
    assert captured["cmd"][1] == "kickstart"
    assert captured["cmd"][2].startswith("gui/")
    assert captured["cmd"][2].endswith(f"/{label}")


def test_status_action_stop_invokes_kill_sigterm():
    """action=stop → corre `launchctl kill SIGTERM gui/<uid>/<label>`."""
    label = "com.fer.obsidian-rag-digest"

    import subprocess
    captured = {}
    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""
    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return FakeCompleted()
    with patch.object(subprocess, "run", side_effect=fake_run):
        r = _client.post("/api/status/action",
                         json={"label": label, "action": "stop"})
    assert r.status_code == 200
    assert r.json()["action"] == "stop"
    assert captured["cmd"][0] == "/bin/launchctl"
    assert captured["cmd"][1] == "kill"
    assert captured["cmd"][2] == "SIGTERM"
    assert captured["cmd"][3].endswith(f"/{label}")


def test_status_action_invalidates_cache():
    """Tras un kickstart, la próxima request a /api/status debe regenerar
    el payload (no devolver el cache viejo de antes de la acción).
    Sin esto el user vería 'aún no corrió' por hasta 3s después de
    haber clickeado start, lo que rompe la sensación de feedback."""
    label = "com.fer.obsidian-rag-digest"

    # Warm el cache primero.
    _client.get("/api/status")
    with _server._STATUS_CACHE_LOCK:
        assert _server._STATUS_CACHE["payload"] is not None

    import subprocess
    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""
    with patch.object(subprocess, "run", return_value=FakeCompleted()):
        _client.post("/api/status/action",
                     json={"label": label, "action": "start"})

    # Cache debe estar invalidado.
    with _server._STATUS_CACHE_LOCK:
        assert _server._STATUS_CACHE["payload"] is None
        assert _server._STATUS_CACHE["ts"] == 0.0


def test_status_actionable_labels_covers_known_jobs():
    """El whitelist incluye los servicios que el usuario quiere poder
    disparar manualmente desde la UI. Si alguien remueve un job del
    catálogo, este test los flagea para que sea decisión consciente."""
    labels = _server._status_actionable_labels()
    must_include = {
        "com.fer.obsidian-rag-digest",
        "com.fer.obsidian-rag-morning",
        "com.fer.obsidian-rag-today",
        "com.fer.obsidian-rag-wake-up",
        "com.fer.obsidian-rag-maintenance",
        "com.fer.obsidian-rag-web",  # daemon
        "com.fer.obsidian-rag-watch",  # daemon
    }
    missing = must_include - labels
    assert not missing, f"faltan en el whitelist: {missing}"


def test_status_actionable_labels_excludes_probes():
    """Los probes (web-self, ollama, vault, etc.) NO son actionable
    (son derived, no servicios launchd). Si alguien por error agrega un
    `target` a un probe, este test rompe."""
    labels = _server._status_actionable_labels()
    # Probes que sabemos no tienen `target`:
    forbidden = {"web-self", "ollama", "rag-db", "telemetry-db", "vault",
                 "wa-db", "tunnel-url"}
    assert not (labels & forbidden), f"probes coladas en whitelist: {labels & forbidden}"


def test_status_js_includes_action_button_wiring():
    """Sanity check: status.js sabe sobre /api/status/action + svc.label."""
    js = (_STATIC_DIR / "status.js").read_text(encoding="utf-8")
    assert "/api/status/action" in js
    assert "svc.label" in js
    assert "service-action" in js


def test_status_html_includes_action_button_styles():
    """status.html tiene CSS para .service-action (start/stop pills)."""
    html = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert ".service-action" in html
    assert ".service-action.start" in html
    assert ".service-action.stop" in html
