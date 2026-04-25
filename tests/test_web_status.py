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
    # Card #4 (actividad reciente / log-tail) tampoco es preview ya — tiene
    # su propio id + filtros + lista que se llena vía /api/status/logs.
    assert 'id="insight-logs"' in body
    assert 'id="logs-list"' in body
    assert 'id="logs-summary"' in body
    # 0 preview badges — todos los 5 cards son reales.
    # Buscamos la clase aplicada en el markup (<span class="preview-badge">)
    # para no contar la definición CSS inline.
    assert body.count('class="preview-badge"') == 0
    assert 'id="heatmap"' in body


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


def test_status_page_freshness_has_real_ids():
    """El HTML tiene los ids del card #3 real (tbody dinámico + total/
    healthy/summary). Si alguien refactorea el markup, este test rompe."""
    body = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert 'id="insight-freshness"' in body
    assert 'id="fresh-tbody"' in body
    assert 'id="fresh-healthy"' in body
    assert 'id="fresh-total"' in body
    assert 'id="fresh-summary"' in body
    # Preview badge ahora es 1 (solo heatmap uptime sigue siendo dummy;
    # log-tail dejó de ser preview cuando se conectó a /api/status/logs).
    assert body.count('class="preview-badge"') == 0


# ── /api/status/freshness — freshness matrix endpoint ───────────────
# Mide last-run de cada ingestor via mtime del stdout log del launchd
# job. Tests se apoyan en la API pública (TestClient) y en el build
# helper directo para no depender de que el FS del CI tenga los logs.

def test_api_freshness_payload_shape():
    """GET /api/status/freshness devuelve los campos del contract."""
    resp = _client.get("/api/status/freshness?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    for k in ("window_hours", "sources", "sources_healthy", "sources_total"):
        assert k in d, f"falta {k!r}"
    assert isinstance(d["sources"], list)
    # Las 6 fuentes estables (vault + 5 ingestores).
    ids = [s["id"] for s in d["sources"]]
    assert ids == ["vault", "whatsapp", "gmail", "calendar", "reminders", "drive"], (
        f"orden / ids inesperados: {ids}"
    )


def test_api_freshness_row_shape():
    """Cada fila tiene el shape esperado por el frontend."""
    resp = _client.get("/api/status/freshness?nocache=1")
    d = resp.json()
    for s in d["sources"]:
        for field in ("id", "label", "kind", "last_run_ts", "age_seconds",
                      "sla_seconds", "drift_ratio", "status", "detail"):
            assert field in s, f"{s['id']} falta {field}"
        assert s["status"] in {"ok", "warn", "stale", "unknown"}
        assert s["kind"] in {"continuous", "scheduled"}


def test_freshness_build_classifies_by_drift(tmp_path, monkeypatch):
    """Con logs sintéticos, verificar que el threshold del drift_ratio
    clasifica ok/warn/stale correctamente (1.0 y 3.0 como corte)."""
    import time as _time
    log_dir = tmp_path / ".local" / "share" / "obsidian-rag"
    log_dir.mkdir(parents=True)
    la_dir = tmp_path / "LaunchAgents"
    la_dir.mkdir()

    # Apuntar el builder al tmp_path reemplazando Path.home() via
    # monkeypatching del módulo.
    from pathlib import Path as _P
    orig_home = _P.home
    monkeypatch.setattr(_P, "home", staticmethod(lambda: tmp_path))

    # Escribir un plist mínimo para que _read_start_interval_s devuelva
    # algo. El fallback del catalog también serviría pero testeamos el
    # parse path explícitamente.
    def _write_plist(label, interval):
        (la_dir / f"{label}.plist").write_text(
            f'<plist><dict><key>StartInterval</key><integer>{interval}</integer></dict></plist>\n',
            encoding="utf-8",
        )
    _write_plist("com.fer.obsidian-rag-watch", 900)
    _write_plist("com.fer.obsidian-rag-ingest-whatsapp", 900)
    _write_plist("com.fer.obsidian-rag-ingest-gmail", 3600)
    _write_plist("com.fer.obsidian-rag-ingest-calendar", 3600)
    _write_plist("com.fer.obsidian-rag-ingest-reminders", 3600)
    _write_plist("com.fer.obsidian-rag-ingest-drive", 3600)

    now = _time.time()
    # vault: fresh (drift < 1), WA: warn (drift 1.5), gmail: stale (drift 4),
    # calendar: falta log (unknown), reminders: fresh, drive: warn.
    (log_dir / "watch.log").write_text("x")
    import os
    os.utime(log_dir / "watch.log", (now - 60, now - 60))  # 1min ago

    (log_dir / "ingest-whatsapp.log").write_text("x")
    os.utime(log_dir / "ingest-whatsapp.log", (now - 1350, now - 1350))  # 22min ago, sla=15m → drift ~1.5

    (log_dir / "ingest-gmail.log").write_text("x")
    os.utime(log_dir / "ingest-gmail.log", (now - 4 * 3600, now - 4 * 3600))  # 4h ago, sla=1h → drift ~4

    # calendar NO existe → unknown

    (log_dir / "ingest-reminders.log").write_text("x")
    os.utime(log_dir / "ingest-reminders.log", (now - 120, now - 120))  # 2min → ok

    (log_dir / "ingest-drive.log").write_text("x")
    os.utime(log_dir / "ingest-drive.log", (now - 2 * 3600, now - 2 * 3600))  # 2h → warn

    # Invalidar cache module-level.
    with _server._FRESHNESS_CACHE_LOCK:
        _server._FRESHNESS_CACHE["ts"] = 0.0
        _server._FRESHNESS_CACHE["payload"] = None

    payload = _server._status_freshness_build_payload()
    by_id = {s["id"]: s for s in payload["sources"]}

    assert by_id["vault"]["status"] == "ok"
    assert by_id["whatsapp"]["status"] == "warn"
    assert by_id["gmail"]["status"] == "stale"
    assert by_id["calendar"]["status"] == "unknown"
    assert by_id["reminders"]["status"] == "ok"
    assert by_id["drive"]["status"] == "warn"

    # sources_healthy = cuenta de ok (vault + reminders = 2).
    assert payload["sources_healthy"] == 2
    assert payload["sources_total"] == 6

    # Los detalles humanos no están vacíos.
    assert by_id["vault"]["detail"].startswith("hace")
    assert by_id["calendar"]["detail"]  # "nunca corrió" o equivalente


def test_read_start_interval_fallback():
    """Plist inexistente devuelve None. Plist sin StartInterval devuelve
    None (e.g. StartCalendarInterval-only jobs)."""
    assert _server._read_start_interval_s("com.nonexistent.label.12345") is None


def test_status_page_uptime_has_real_ids():
    """El HTML del card #5 tiene los ids del componente real (no más
    preview badge)."""
    body = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert 'id="insight-uptime"' in body
    assert 'id="heatmap"' in body
    assert 'id="uptime-headline"' in body
    assert 'id="uptime-meta"' in body
    # Card #5 ya no tiene preview-badge — todos los 5 cards reales.
    # El único `class="preview-badge"` posible sería en CSS o vacío.
    # Buscamos que no haya ningún badge aplicado en markup.
    assert 'class="preview-badge"' not in body
    # `id="heatmap-mock"` ya no existe (renombrado a heatmap).
    assert 'id="heatmap-mock"' not in body


# ── /api/status/uptime — heatmap endpoint ───────────────────────────

def test_api_uptime_payload_shape():
    """GET /api/status/uptime devuelve services con buckets 168-len."""
    resp = _client.get("/api/status/uptime?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert d["window_days"] == 7
    assert d["bucket_hours"] == 1
    assert isinstance(d["services"], list)
    # Hardcoded 5 servicios core.
    assert len(d["services"]) == 5
    for s in d["services"]:
        for k in ("id", "label", "uptime_pct_7d", "samples_7d", "buckets"):
            assert k in s, f"{s.get('id')} falta {k!r}"
        # 168 buckets siempre (7d × 24h), aunque la mayoría sean null.
        assert len(s["buckets"]) == 168
        for b in s["buckets"]:
            assert "ts" in b and "uptime_pct" in b and "samples" in b
            assert b["uptime_pct"] is None or 0.0 <= b["uptime_pct"] <= 100.0


def test_api_uptime_services_match_tracked_set():
    """Los 5 servicios devueltos coinciden con _UPTIME_TRACKED_SERVICES."""
    resp = _client.get("/api/status/uptime?nocache=1")
    d = resp.json()
    ids = [s["id"] for s in d["services"]]
    expected = ["web-self", "ollama", "rag-db", "telemetry-db", "vault"]
    assert ids == expected, f"orden / ids inesperados: {ids}"


def test_uptime_build_with_synthetic_samples(tmp_path, monkeypatch):
    """Insertar samples sintéticos a lo largo de varias horas y verificar
    que los buckets horarios + uptime_pct se computan correctamente.

    Caso: 3 horas distintas, mix ok/down → bucket_pct = ok/total.
    """
    import rag
    from datetime import datetime, timedelta
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    now = datetime.now()
    # Hora -2: 4 samples ok, 1 down → 80%.
    # Hora -1: 5 samples ok → 100%.
    # Hora actual: 2 ok + 2 warn (warn no cuenta como ok) → 50%.
    h2 = now - timedelta(hours=2, minutes=15)
    h1 = now - timedelta(hours=1, minutes=15)
    h0 = now - timedelta(minutes=5)
    samples = []
    for i in range(4):
        samples.append((
            (h2 + timedelta(minutes=i)).isoformat(timespec="seconds"),
            "web-self", "ok",
        ))
    # ts único — `h2 + 4 minutes` para no chocar con i=0 (que usa h2 + 0)
    # bajo el constraint PRIMARY KEY (ts, service_id).
    samples.append(((h2 + timedelta(minutes=4)).isoformat(timespec="seconds"),
                    "web-self", "down"))
    for i in range(5):
        samples.append((
            (h1 + timedelta(minutes=i)).isoformat(timespec="seconds"),
            "web-self", "ok",
        ))
    for i in range(2):
        samples.append((
            (h0 + timedelta(minutes=i)).isoformat(timespec="seconds"),
            "web-self", "ok",
        ))
        samples.append((
            (h0 + timedelta(minutes=i, seconds=30)).isoformat(timespec="seconds"),
            "web-self", "warn",
        ))

    with rag._ragvec_state_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO rag_status_samples (ts, service_id, status) VALUES (?, ?, ?)",
            samples,
        )

    # Invalidar caches de servidor.
    with _server._UPTIME_CACHE_LOCK:
        _server._UPTIME_CACHE["ts"] = 0.0
        _server._UPTIME_CACHE["payload"] = None

    payload = _server._status_uptime_build_payload()
    web_svc = next((s for s in payload["services"] if s["id"] == "web-self"), None)
    assert web_svc is not None
    assert web_svc["samples_7d"] == 14  # 5 + 5 + 4 (h0 = 2 ok + 2 warn)

    # uptime_pct_7d = ok_total / samples_total = (4 + 5 + 2) / 14 = 11/14 ≈ 78.57%
    assert abs(web_svc["uptime_pct_7d"] - (11/14 * 100)) < 0.5

    # Verificar buckets puntuales (los 3 con datos).
    buckets_with_data = [b for b in web_svc["buckets"] if b["samples"] > 0]
    assert len(buckets_with_data) >= 3


def test_periodic_probe_runs_one_tick(tmp_path, monkeypatch):
    """El periodic probe loop, ejecutado una sola iteración, llama a
    `_status_build_payload` + persiste samples + actualiza el cache.

    No spawn del thread real: simulamos una sola iteración del while
    seteando el stop event antes de la 2da. Así no dependemos de timing
    real de 60s en el test.
    """
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    # Reset state global para evitar interferencia con tests previos.
    with _server._STATUS_SAMPLE_LOCK:
        _server._STATUS_SAMPLE_LAST_TS["ts"] = 0.0
    with _server._STATUS_CACHE_LOCK:
        _server._STATUS_CACHE["ts"] = 0.0
        _server._STATUS_CACHE["payload"] = None

    # Mockear _status_build_payload para evitar correr 25 probes reales
    # (ollama, launchctl, etc.) — sólo nos importa que el loop llame a
    # las 3 funciones en orden.
    fake_payload = {
        "categories": [{"id": "core", "services": [
            {"id": "web-self", "status": "ok"},
            {"id": "ollama", "status": "ok"},
            {"id": "vault", "status": "warn"},
        ]}],
    }
    monkeypatch.setattr(_server, "_status_build_payload", lambda: fake_payload)

    # Strategy: dejar que el loop entre a su primera iteración (STOP
    # clear), pero envolver `_persist_status_samples` para que después
    # de correr una vez, setee STOP — así la 2da iteración del while
    # no se ejecuta y el test no se cuelga 60s en el wait().
    real_persist = _server._persist_status_samples
    def _persist_then_stop(payload):
        real_persist(payload)
        _server._STATUS_PROBE_STOP.set()
    monkeypatch.setattr(_server, "_persist_status_samples", _persist_then_stop)

    _server._STATUS_PROBE_STOP.clear()
    _server._status_periodic_probe_loop()

    # Verificar que la persistencia ocurrió (3 rows = 3 servicios
    # rastreados en _UPTIME_TRACKED_IDS, los demás del fake_payload se
    # filtran).
    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT service_id FROM rag_status_samples"
        ).fetchall()
    sids = {r[0] for r in rows}
    # web-self + ollama + vault del fake_payload, todos en _UPTIME_TRACKED_IDS.
    assert "web-self" in sids
    assert "ollama" in sids
    assert "vault" in sids

    # Cache del /api/status también debe estar warm post-tick.
    with _server._STATUS_CACHE_LOCK:
        cached = _server._STATUS_CACHE["payload"]
    assert cached is fake_payload


def test_periodic_probe_thread_can_be_stopped():
    """`_stop_status_probe_thread` señala el Event y hace join. Test
    simula el ciclo de vida completo: start → un tick → stop, sin
    timing real.
    """
    # Reset state.
    _server._STATUS_PROBE_STOP.clear()
    # Si hubo un thread previo en este test session, esperarlo.
    if _server._STATUS_PROBE_THREAD is not None and _server._STATUS_PROBE_THREAD.is_alive():
        _server._STATUS_PROBE_STOP.set()
        _server._STATUS_PROBE_THREAD.join(timeout=2.0)
        _server._STATUS_PROBE_STOP.clear()

    # Start.
    _server._start_status_probe_thread()
    assert _server._STATUS_PROBE_THREAD is not None
    assert _server._STATUS_PROBE_THREAD.is_alive()

    # Stop. El thread debería salir rápido porque el wait() del Event
    # se cancela inmediatamente.
    _server._stop_status_probe_thread()

    # Después del stop, el thread no debería seguir alive (join 2s).
    assert not _server._STATUS_PROBE_THREAD.is_alive(), (
        "el thread no se cerró tras _stop_status_probe_thread"
    )


def test_persist_status_samples_rate_limits(tmp_path, monkeypatch):
    """`_persist_status_samples` debería respetar el rate-limit de 60s
    — múltiples calls back-to-back resultan en un solo INSERT batch."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    # Reset rate-limit clock + cache.
    with _server._STATUS_SAMPLE_LOCK:
        _server._STATUS_SAMPLE_LAST_TS["ts"] = 0.0

    fake_payload = {
        "categories": [
            {"id": "core", "services": [
                {"id": "web-self", "status": "ok"},
                {"id": "ollama", "status": "ok"},
                {"id": "vault", "status": "warn"},
            ]},
        ],
    }

    # Primera llamada → 3 samples insertados.
    _server._persist_status_samples(fake_payload)

    # Segunda llamada inmediata → no debería escribir más (rate-limit).
    _server._persist_status_samples(fake_payload)
    _server._persist_status_samples(fake_payload)

    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT service_id, status FROM rag_status_samples ORDER BY service_id"
        ).fetchall()

    # 3 rows insertadas (1 por servicio rastreado), no 9.
    assert len(rows) == 3
    sids = sorted(r[0] for r in rows)
    assert sids == ["ollama", "vault", "web-self"]
    # vault tiene status "warn", los demás "ok".
    by_id = dict(rows)
    assert by_id["vault"] == "warn"
    assert by_id["web-self"] == "ok"


def test_status_page_logs_has_real_ids():
    """El HTML del card #4 tiene ids + filtros."""
    body = (_STATIC_DIR / "status.html").read_text(encoding="utf-8")
    assert 'id="insight-logs"' in body
    assert 'id="logs-list"' in body
    assert 'id="logs-summary"' in body
    # Filtros: 3 botones de window + 3 de level (10m/1h/24h, all/warn/error).
    assert 'data-window="600"' in body
    assert 'data-window="3600"' in body
    assert 'data-window="86400"' in body
    assert 'data-level="all"' in body
    assert 'data-level="warn"' in body
    assert 'data-level="error"' in body
    # Preview badges ahora sólo 1 (el heatmap uptime).
    assert body.count('class="preview-badge"') == 0


# ── /api/status/logs — log-tail endpoint ────────────────────────────

def test_api_logs_payload_shape():
    """GET /api/status/logs devuelve los campos del contract."""
    resp = _client.get("/api/status/logs?nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    for k in ("window_seconds", "limit", "level_filter", "events",
              "total_in_window", "truncated"):
        assert k in d, f"falta {k!r}"
    assert isinstance(d["events"], list)


def test_api_logs_validates_level():
    """level inválido → 400, no 500."""
    resp = _client.get("/api/status/logs?level=critical&nocache=1")
    assert resp.status_code == 400
    assert "level inválido" in resp.json()["detail"]


def test_api_logs_clamps_limits():
    """since_seconds y limit clamped a sus máximos."""
    resp = _client.get("/api/status/logs?since_seconds=999999&limit=999&nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert d["window_seconds"] <= 86400  # 24h max
    assert d["limit"] <= 200             # max limit


def test_logs_build_filters_by_level(tmp_path, monkeypatch):
    """level=warn devuelve sólo events de silent_errors; level=error sólo
    de sql_state. all merge ambos."""
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)

    now = datetime.now()
    recent = now - timedelta(minutes=5)
    _write_jsonl_errors(silent_log, [
        {"ts": recent.isoformat(timespec="seconds"), "where": "w_a",
         "exc_type": "X", "exc": "msg_a"},
    ])
    _write_jsonl_errors(sql_log, [
        {"ts": recent.isoformat(timespec="seconds"), "event": "e_b",
         "err": "OperationalError('msg_b')"},
    ])

    # level=warn → sólo el silent.
    p_warn = _server._status_logs_build_payload(3600, 50, "warn")
    assert len(p_warn["events"]) == 1
    assert p_warn["events"][0]["level"] == "WARN"
    assert p_warn["events"][0]["where"] == "w_a"
    assert p_warn["events"][0]["source"] == "silent"

    # level=error → sólo el sql.
    p_err = _server._status_logs_build_payload(3600, 50, "error")
    assert len(p_err["events"]) == 1
    assert p_err["events"][0]["level"] == "ERROR"
    assert p_err["events"][0]["where"] == "e_b"
    assert p_err["events"][0]["source"] == "sql"

    # level=all → ambos.
    p_all = _server._status_logs_build_payload(3600, 50, "all")
    assert len(p_all["events"]) == 2


def test_logs_build_extracts_exc_type_from_sql_repr(tmp_path, monkeypatch):
    """Para sql_state_errors el `err` es repr() del exception. Verifica
    que el regex extrae correctamente el exc_type prefix y limpia el
    mensaje (sin las comillas del repr)."""
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)

    silent_log.write_text("", encoding="utf-8")
    now = datetime.now()
    _write_jsonl_errors(sql_log, [
        {"ts": now.isoformat(timespec="seconds"), "event": "x",
         "err": "OperationalError('database is locked')"},
        {"ts": now.isoformat(timespec="seconds"), "event": "y",
         "err": "TimeoutError('budget exhausted')"},
        # err sin formato Tipo(...) → exc_type vacío, msg = err completo.
        {"ts": now.isoformat(timespec="seconds"), "event": "z",
         "err": "weird unstructured error"},
    ])

    payload = _server._status_logs_build_payload(3600, 50, "error")
    by_where = {e["where"]: e for e in payload["events"]}
    assert by_where["x"]["exc_type"] == "OperationalError"
    assert by_where["x"]["message"] == "database is locked"
    assert by_where["y"]["exc_type"] == "TimeoutError"
    assert by_where["y"]["message"] == "budget exhausted"
    # Unstructured err: exc_type vacío, message = el string entero.
    assert by_where["z"]["exc_type"] == ""
    assert by_where["z"]["message"] == "weird unstructured error"


def test_logs_build_sorts_desc_and_caps_limit(tmp_path, monkeypatch):
    """Events ordenados desc por ts; el cap por `limit` deja truncated=
    True y total_in_window con el count completo."""
    from datetime import datetime, timedelta
    silent_log = tmp_path / "silent_errors.jsonl"
    sql_log = tmp_path / "sql_state_errors.jsonl"
    monkeypatch.setattr(_server, "SILENT_ERRORS_LOG_PATH", silent_log)
    monkeypatch.setattr(_server, "_SQL_STATE_ERROR_LOG", sql_log)
    sql_log.write_text("", encoding="utf-8")

    base = datetime.now() - timedelta(minutes=10)
    # 20 events espaciados de a 10s, el más reciente al final.
    entries = [
        {"ts": (base + timedelta(seconds=i * 10)).isoformat(timespec="seconds"),
         "where": f"w_{i:02d}", "exc_type": "X", "exc": "msg"}
        for i in range(20)
    ]
    _write_jsonl_errors(silent_log, entries)

    payload = _server._status_logs_build_payload(3600, 5, "all")
    assert payload["total_in_window"] == 20
    assert payload["truncated"] is True
    assert len(payload["events"]) == 5
    # El primero debe ser el más reciente (w_19), no el más viejo.
    assert payload["events"][0]["where"] == "w_19"
    assert payload["events"][-1]["where"] == "w_15"


def test_fmt_age_spanish_boundaries():
    """El formato 'hace Xs/m/h/d' se elige por boundary correcto."""
    f = _server._fmt_age_spanish
    assert f(-5) == "justo ahora"
    assert f(30) == "hace 30s"
    assert f(59) == "hace 59s"
    assert f(60) == "hace 1m"
    assert f(599) == "hace 9m"
    assert f(3600) == "hace 1h"
    assert f(7200) == "hace 2h"
    assert f(86400) == "hace 1d"
    assert f(172800) == "hace 2d"


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
