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


# ── /api/status/latency — sparkline endpoint ────────────────────────
# Feed el card #1 del insights grid. Payload contract testeado por
# shape + 25 buckets + cómputo correcto de percentiles con data sintética.

def test_api_latency_payload_shape():
    """GET /api/status/latency devuelve window_hours + bucket + series + summary."""
    resp = _client.get("/api/status/latency?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert d["window_hours"] == 24
    assert d["bucket"] == "hour"
    assert isinstance(d["series"], list)
    assert isinstance(d["summary"], dict)
    for k in (
        "p50_1h_ms", "p95_1h_ms",
        "p50_baseline_ms", "p95_baseline_ms",
        "delta_p95_pct", "count_24h",
    ):
        assert k in d["summary"], f"summary falta {k!r}"


def test_api_latency_series_has_25_buckets():
    """La serie siempre tiene 25 elementos (24h atrás + hora actual),
    aún cuando no hay data en rag_queries. Sin esto el frontend se
    queda sin eje X estable y el sparkline se comprime raro."""
    resp = _client.get("/api/status/latency?nocache=1")
    d = resp.json()
    assert len(d["series"]) == 25, (
        f"serie debería tener 25 buckets (24h back + current), tiene {len(d['series'])}"
    )
    # Cada bucket tiene el mismo schema.
    for s in d["series"]:
        assert "ts" in s
        assert "count" in s and isinstance(s["count"], int)
        # p50/p95/p99 pueden ser null en horas sin queries.
        for k in ("p50_ms", "p95_ms", "p99_ms"):
            assert k in s
            assert s[k] is None or isinstance(s[k], int)


def test_api_latency_percentiles_with_fake_data(tmp_path, monkeypatch):
    """Insert rows sintéticas en rag_queries (cmd='web' + extra_json.total_ms)
    y verificar que p50 / p95 salen computados via json_extract + window-
    function nearest-rank. Esto prueba end-to-end que el frontend va a
    ver los números correctos cuando haya data.

    Nota: el /api/status/latency cache es sensible a DB_PATH. Seteamos
    DB_PATH al tmp_path ANTES de instanciar el TestClient acá porque el
    _client module-level ya apuntó al DB real; así que llamamos al build
    function directamente (saltea el cache + TestClient) — más
    determinístico para tests.
    """
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    # 10 queries sintéticas, todas en la hora actual, total_ms en pasos
    # de 1000 de 1000..10000. El p50 de [1000..10000] por nearest-rank
    # lower-median es 5000, el p95 es 10000 (último cruzando 0.95).
    for i, ms in enumerate([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]):
        rag.log_query_event({
            "cmd": "web", "q": f"q{i}",
            "t_retrieve": 1.0, "t_gen": 3.0,
            "ttft_ms": 500, "llm_prefill_ms": 500, "llm_decode_ms": 2500,
            "total_ms": ms,
        })

    # Llamar al build directo para saltarse el módulo-level cache y el
    # DB_PATH del TestClient (ambos apuntan al DB real).
    payload = _server._status_latency_build_payload()
    # Contract: 25 buckets + summary completo.
    assert len(payload["series"]) == 25
    # El único bucket con data es el current (último); chequear sus
    # percentiles.
    buckets_with_data = [s for s in payload["series"] if s["count"] > 0]
    assert len(buckets_with_data) == 1, (
        f"solo la hora actual debería tener data, encontré {len(buckets_with_data)}"
    )
    current = buckets_with_data[0]
    assert current["count"] == 10
    # Nearest-rank percentiles sobre [1000..10000]: p50=5000, p95=10000.
    assert current["p50_ms"] == 5000
    assert current["p95_ms"] == 10000
    # Summary: last-hour == current bucket, count_24h == 10.
    assert payload["summary"]["p50_1h_ms"] == 5000
    assert payload["summary"]["p95_1h_ms"] == 10000
    assert payload["summary"]["count_24h"] == 10


def test_api_latency_handles_empty_db_gracefully(tmp_path, monkeypatch):
    """Con rag_queries vacía, el endpoint devuelve 25 buckets vacíos +
    summary con nulls, no 500. Sin esto la /status page rompe en una
    instalación fresh o después de wipe de telemetría."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    payload = _server._status_latency_build_payload()
    assert len(payload["series"]) == 25
    assert all(s["count"] == 0 for s in payload["series"])
    assert payload["summary"]["p50_1h_ms"] is None
    assert payload["summary"]["p95_1h_ms"] is None
    assert payload["summary"]["delta_p95_pct"] is None
    assert payload["summary"]["count_24h"] == 0


def test_status_page_has_insights_section():
    """El HTML tiene la sección insights con las 5 cards + badges de
    vista previa en las cards aún no implementadas.
    Si alguien mueve/borra esto, la validación del layout se pierde."""
    body = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert 'id="insights"' in body
    assert 'id="insight-latency"' in body
    assert 'id="lat-sparkline"' in body
    assert 'id="lat-p95-1h"' in body
    # Card #2 (errors) ya no es preview — tiene su propio id + elementos.
    assert 'id="insight-errors"' in body
    assert 'id="err-total"' in body
    assert 'id="err-donut"' in body
    assert 'id="err-breakdown"' in body
    # 3 preview badges restantes (freshness + log-tail + heatmap uptime).
    # Buscamos la clase aplicada en el markup (<span class="preview-badge">)
    # para no contar la definición CSS inline.
    assert body.count('class="preview-badge"') == 3
    assert 'id="heatmap-mock"' in body


# ── /api/status/errors — error-budget endpoint ──────────────────────
# Usa los jsonl que rag.py ya mantiene (silent_errors + sql_state_errors).
# Tests escriben fixtures temporales y verifican rollup + delta.

def _write_jsonl_errors(path, entries):
    """Helper: escribir `entries` (lista de dict) como jsonl en path."""
    import json as _json
    lines = [_json.dumps(e, ensure_ascii=False) + "\n" for e in entries]
    path.write_text("".join(lines), encoding="utf-8")


def test_api_errors_payload_shape():
    """GET /api/status/errors devuelve los campos del contract."""
    resp = _client.get("/api/status/errors?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    for k in ("window_hours", "total_errors", "by_source", "breakdown",
              "total_errors_prev_24h", "delta_pct"):
        assert k in d, f"falta {k!r}"
    assert d["window_hours"] == 24
    assert isinstance(d["breakdown"], list)
    assert "silent" in d["by_source"]
    assert "sql" in d["by_source"]


def test_api_errors_rollup_with_synthetic_logs(tmp_path, monkeypatch):
    """Escribir entries sintéticas a los 2 jsonl y verificar que el
    rollup agrupa + cuenta + ordena correctamente. Cubre también:
      - Filtro por ventana 24h (entry más vieja se excluye)
      - Top-N (pedimos más de N entries distintas para forzar (other))
      - by_source por cada jsonl
    """
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)

    now = datetime.now()
    recent = now - timedelta(hours=2)
    old = now - timedelta(hours=30)  # fuera de la ventana 24h

    # 3× `reranker_unload` + 2× `google_token_refresh` + 1× `old_bug`
    # (old_bug fuera de ventana, se descarta)
    _write_jsonl_errors(silent_log, [
        {"ts": recent.isoformat(timespec="seconds"), "where": "reranker_unload", "exc_type": "X", "exc": "x"},
        {"ts": recent.isoformat(timespec="seconds"), "where": "reranker_unload", "exc_type": "X", "exc": "x"},
        {"ts": recent.isoformat(timespec="seconds"), "where": "reranker_unload", "exc_type": "X", "exc": "x"},
        {"ts": recent.isoformat(timespec="seconds"), "where": "google_token_refresh", "exc_type": "X", "exc": "x"},
        {"ts": recent.isoformat(timespec="seconds"), "where": "google_token_refresh", "exc_type": "X", "exc": "x"},
        {"ts": old.isoformat(timespec="seconds"), "where": "old_bug", "exc_type": "X", "exc": "x"},
    ])
    # 10× `queries_sql_write_failed` (el top)
    _write_jsonl_errors(sql_log, [
        {"ts": recent.isoformat(timespec="seconds"), "event": "queries_sql_write_failed", "err": "x"}
        for _ in range(10)
    ])

    # Invalidar el cache module-level porque tests anteriores llenaron
    # con el DB real; pedir fresh.
    with _server._ERRORS_CACHE_LOCK:
        _server._ERRORS_CACHE["ts"] = 0.0
        _server._ERRORS_CACHE["payload"] = None

    payload = _server._status_errors_build_payload()
    # 5 silent (3 reranker + 2 google, el old se excluye) + 10 sql = 15.
    assert payload["total_errors"] == 15
    assert payload["by_source"] == {"silent": 5, "sql": 10}
    # Breakdown ordenado por count desc: queries_sql (10) > reranker (3) > google (2).
    assert payload["breakdown"][0]["key"] == "queries_sql_write_failed"
    assert payload["breakdown"][0]["count"] == 10
    assert payload["breakdown"][0]["source"] == "sql"
    assert payload["breakdown"][1]["key"] == "reranker_unload"
    assert payload["breakdown"][1]["count"] == 3
    assert payload["breakdown"][1]["source"] == "silent"
    assert payload["breakdown"][2]["key"] == "google_token_refresh"
    assert payload["breakdown"][2]["count"] == 2


def test_api_errors_delta_vs_prev_window(tmp_path, monkeypatch):
    """Delta vs las 24h previas se computa cuando hay entries en
    [now-48h, now-24h). Sin entries previas → delta_pct=None, nunca
    inf por división por cero."""
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)

    now = datetime.now()
    recent = now - timedelta(hours=2)
    prev = now - timedelta(hours=30)  # en la ventana [24h, 48h)

    # 20 errores hoy, 10 ayer → delta = +100%
    _write_jsonl_errors(silent_log, [
        {"ts": recent.isoformat(timespec="seconds"), "where": "x", "exc_type": "X", "exc": "x"}
        for _ in range(20)
    ] + [
        {"ts": prev.isoformat(timespec="seconds"), "where": "x", "exc_type": "X", "exc": "x"}
        for _ in range(10)
    ])
    sql_log.write_text("", encoding="utf-8")  # empty

    with _server._ERRORS_CACHE_LOCK:
        _server._ERRORS_CACHE["ts"] = 0.0
        _server._ERRORS_CACHE["payload"] = None
    payload = _server._status_errors_build_payload()
    assert payload["total_errors"] == 20
    assert payload["total_errors_prev_24h"] == 10
    assert payload["delta_pct"] == 100.0


def test_api_errors_no_logs_returns_empty_gracefully(tmp_path, monkeypatch):
    """Sin jsonl files (install fresh / logs rotados), el endpoint
    devuelve 0 + breakdown=[] sin 500ear."""
    missing_silent = tmp_path / "does-not-exist-silent.jsonl"
    missing_sql = tmp_path / "does-not-exist-sql.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", missing_silent)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", missing_sql)

    with _server._ERRORS_CACHE_LOCK:
        _server._ERRORS_CACHE["ts"] = 0.0
        _server._ERRORS_CACHE["payload"] = None
    payload = _server._status_errors_build_payload()
    assert payload["total_errors"] == 0
    assert payload["breakdown"] == []
    assert payload["delta_pct"] is None


def test_api_errors_top_n_collapses_rest_to_other(tmp_path, monkeypatch):
    """Cuando hay más de 6 where/event distintos, el 7º en adelante se
    agrega en el bucket `(other)` para que la UI no overflow."""
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)

    now = datetime.now()
    recent = now - timedelta(hours=1)
    # 10 where distintos, todos con count=1 (orden ascendente
    # alfabético tie-break por el sort).
    entries = [
        {"ts": recent.isoformat(timespec="seconds"), "where": f"w{i:02d}", "exc_type": "X", "exc": "x"}
        for i in range(10)
    ]
    _write_jsonl_errors(silent_log, entries)
    sql_log.write_text("", encoding="utf-8")

    with _server._ERRORS_CACHE_LOCK:
        _server._ERRORS_CACHE["ts"] = 0.0
        _server._ERRORS_CACHE["payload"] = None
    payload = _server._status_errors_build_payload()
    # 6 top + 1 (other) = 7 items en breakdown.
    assert len(payload["breakdown"]) == 7
    other = payload["breakdown"][-1]
    assert other["key"] == "(other)"
    # 10 total - 6 top = 4 en other.
    assert other["count"] == 4
    assert other["source"] == "mixed"


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
