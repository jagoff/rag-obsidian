"""Tests del /logs page + /api/logs + /api/logs/file (2026-04-26).

Regresiones que atrapan:
  - GET /logs devuelve el HTML, GET /api/logs devuelve JSON con la lista
    agrupada por service y status agregado, GET /api/logs/file devuelve
    el tail con cada línea clasificada.
  - El clasificador `_classify_log_line` separa correctamente líneas
    error / warn / ok / info, incluyendo edge-cases que ya rompieron en
    desarrollo:
      - `OperationalError` (CamelCase, sin word boundary inicial)
      - `failed=0` / `errors=0` (NO es error — son stats)
      - `INFO: ... .error.log ...` (NO es error — el "error" está en una
        URL del access log)
  - `_resolve_log_path` rechaza path traversal (`../`, paths absolutos,
    nombres fuera del allowlist de _LOG_DIRS).
  - `_read_tail_lines` devuelve las últimas N líneas en orden cronológico
    sin cargar el archivo entero, incluso para archivos grandes (>>
    chunk_size).

No testeamos:
  - La UI del dashboard (verificado con Playwright a mano).
  - Cache TTL del index — la lógica es trivial.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_STATIC_DIR = Path(_server.STATIC_DIR)
_client = TestClient(_server.app)


@pytest.fixture(autouse=True)
def _bypass_admin_token():
    """Bypass `_require_admin_token` (ver test_diagnose_error.py para
    rationale completo). Necesario para los tests de
    `/api/diagnose-error/execute` y `/api/auto-fix*`.
    """
    _server.app.dependency_overrides[_server._require_admin_token] = lambda: None
    yield
    _server.app.dependency_overrides.pop(_server._require_admin_token, None)


# ── Classifier ───────────────────────────────────────────────────────

@pytest.mark.parametrize("line, expected", [
    # Errors via plain "error" keyword.
    ("Traceback (most recent call last):", "error"),
    ("error  2026-04-22.md: database is locked", "error"),
    ("ValueError: invalid literal", "error"),
    # CamelCase suffixes — el bug histórico es que `\bError\b` no matchea
    # cuando viene pegado a "Operational" (no hay word-boundary entre l-E).
    ("sqlite3.OperationalError: no such column: trace_id", "error"),
    ("RuntimeError('foo')", "error"),
    ("psycopg2.errors.DatabaseError: connection failed", "error"),
    # Falsos positivos — stats que dicen `failed=0` no son errores.
    ("live reminder push · pushed=0 skipped=0 failed=0 items=0", "info"),
    ("live wa-scheduled-send · processed=0 sent=0 failed=0", "info"),
    ("errors: 0", "info"),
    # Idem con el cero ANTES de la palabra ("0 failed", "0 errors").
    # Patrón típico de los syncs tipo YouTube transcripts:
    # `(10 fetched · 0 failed · 29 known, 8463ms)`. Bug histórico
    # 2026-04-29: el alerter agarró esto como error porque el regex
    # solo cubría `failed=0`, no `0 failed`.
    ("YT trans. sync: 10 fetched · 0 failed · 29 known, 8463ms", "info"),
    ("0 failed · 29 known, 8463ms)", "info"),
    # Pero `1 failed` / `5 errors` SÍ son errores (cualquier número
    # distinto de cero antes de la palabra debe seguir clasificándose).
    ("scan: 1 failed · 5 known", "error"),
    ("daemon · 5 failed in last hour", "error"),
    # Warnings (incluyendo CamelCase).
    ("warnings.warn(", "warn"),
    ("UserWarning: resource_tracker: leaked semaphore", "warn"),
    ("DeprecationWarning: foo is deprecated", "warn"),
    # OK heartbeat / status normales.
    ("[heartbeat] 2026-04-26T19:07:34 alive=true vaults=2", "ok"),
    ("  ✓ [5] 01-Projects: Punto clave", "ok"),
    # INFO prefix (uvicorn / stdlib logging) — el "error" en una URL del
    # access log NO debe contar como error.
    (
        'INFO:     127.0.0.1 - "GET /api/logs/file?name=foo.error.log HTTP/1.1" 200 OK',
        "info",
    ),
    ("INFO: Application startup complete.", "info"),
    ("DEBUG: foo bar baz", "info"),
    # Líneas neutras → info.
    ("whatsapp: 14732 · 0.07s", "info"),
    ("gmail · +2 · 13.62s", "info"),
    ("Selected: anticipate-commitment (score 0.43)", "info"),
    ("preferred", "info"),  # no debería matchear "error" como substring
    ("referrer", "info"),
    # Empty string.
    ("", "info"),
    ("   ", "info"),
    # Que `failed=2` SÍ se considere error (el lookahead negativo era sólo
    # para `=0`).
    ("daemon failed=2 last_error='timeout'", "error"),
])
def test_classify_log_line(line, expected):
    assert _server._classify_log_line(line) == expected


# ── _read_tail_lines ─────────────────────────────────────────────────

def test_read_tail_lines_small_file(tmp_path: Path):
    p = tmp_path / "small.log"
    p.write_text("a\nb\nc\nd\n")
    assert _server._read_tail_lines(p, 10) == ["a", "b", "c", "d"]
    assert _server._read_tail_lines(p, 2) == ["c", "d"]


def test_read_tail_lines_handles_no_trailing_newline(tmp_path: Path):
    p = tmp_path / "no-newline.log"
    p.write_text("line1\nline2\nline3")
    out = _server._read_tail_lines(p, 10)
    assert out == ["line1", "line2", "line3"]


def test_read_tail_lines_large_file(tmp_path: Path):
    """Genera un archivo > chunk_size (64KB) y verifica que el tail
    mantiene el orden cronológico."""
    p = tmp_path / "big.log"
    lines = [f"line {i}" for i in range(5000)]
    p.write_text("\n".join(lines) + "\n")
    out = _server._read_tail_lines(p, 100)
    assert len(out) == 100
    assert out[-1] == "line 4999"
    assert out[0] == "line 4900"


def test_read_tail_lines_empty_file(tmp_path: Path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert _server._read_tail_lines(p, 10) == []


def test_read_tail_lines_missing_file(tmp_path: Path):
    assert _server._read_tail_lines(tmp_path / "no-existe.log", 10) == []


def test_read_tail_lines_handles_invalid_utf8(tmp_path: Path):
    p = tmp_path / "binary.log"
    p.write_bytes(b"good line\n\xff\xfe partial bytes \xed\nlast\n")
    out = _server._read_tail_lines(p, 10)
    # Las 3 líneas se decodean (con replace para los bytes inválidos)
    # sin reventar el endpoint.
    assert len(out) == 3
    assert out[0] == "good line"
    assert out[2] == "last"


# ── _resolve_log_path security ───────────────────────────────────────

@pytest.mark.parametrize("name", [
    "../etc/passwd",
    "../../../../etc/passwd",
    "/etc/passwd",
    "obsidian-rag/../etc/passwd",
    "obsidian-rag/../../etc/passwd",
    "no-existe-dir/foo.log",
    "",
])
def test_resolve_log_path_rejects_traversal_and_invalid(name):
    assert _server._resolve_log_path(name) is None


def test_resolve_log_path_accepts_real_log_in_obsidian_rag_dir():
    """Tomá un .log que exista en el dir; sino esquipear (ej. en CI)."""
    log_dir = Path.home() / ".local/share/obsidian-rag"
    if not log_dir.is_dir():
        pytest.skip("no obsidian-rag log dir on this machine")
    candidates = [p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".log"]
    if not candidates:
        pytest.skip("no .log files to validate against")
    sample = candidates[0]
    resolved = _server._resolve_log_path(f"obsidian-rag/{sample.name}")
    assert resolved == sample.resolve()


# ── /api/logs index ───────────────────────────────────────────────────

def test_api_logs_index_shape(monkeypatch, tmp_path: Path):
    """`/api/logs` devuelve services agrupados con el shape esperado.

    Stub `_LOG_DIRS` para que el test sea determinístico (no depende de
    los logs reales del usuario).

    Las líneas de error necesitan al menos UN timestamp reciente
    (post audit 2026-04-29: `_has_recent_timestamp` early-out evita
    daemons stale eternos en rojo). Usamos `datetime.now()` para
    inyectar ts fresh en el mock.
    """
    from datetime import datetime
    _now_iso = datetime.now().isoformat(timespec="seconds")
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "watch.log").write_text(f"{_now_iso} [heartbeat] alive=true\n")
    (fake_dir / "watch.error.log").write_text(
        f"{_now_iso} Traceback (most recent call last):\n"
        "  File 'foo'\n"
        f"{_now_iso} OperationalError: no such column: trace_id\n"
    )
    (fake_dir / "anticipate.log").write_text(
        f"{_now_iso} Selected: foo\n{_now_iso} no pusheado: ok\n"
    )
    (fake_dir / "anticipate.error.log").write_text("")

    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    # Nukear el cache para no leer un payload viejo.
    monkeypatch.setattr(_server, "_LOGS_INDEX_CACHE", {"ts": 0.0, "payload": None})

    resp = _client.get("/api/logs?nocache=1")
    assert resp.status_code == 200
    d = resp.json()

    assert "scanned_at" in d
    assert "services" in d and isinstance(d["services"], list)
    totals = d["totals"]
    assert totals["services"] == 2
    # `watch` tiene .error.log con tracebacks → status error.
    # `anticipate` está limpio → status ok.
    assert totals["error"] >= 1
    services_by_name = {s["service"]: s for s in d["services"]}
    assert "watch" in services_by_name
    assert "anticipate" in services_by_name
    watch = services_by_name["watch"]
    assert watch["status"] == "error"
    assert watch["error_count_recent"] >= 1
    assert len(watch["files"]) == 2  # stdout + stderr
    kinds = {f["kind"] for f in watch["files"]}
    assert kinds == {"stdout", "stderr"}
    anticipate = services_by_name["anticipate"]
    assert anticipate["status"] == "ok"


def test_api_logs_index_orders_errors_first(monkeypatch, tmp_path: Path):
    """Services con error van arriba de los ok."""
    from datetime import datetime
    _now_iso = datetime.now().isoformat(timespec="seconds")
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "alpha.log").write_text(f"{_now_iso} ok line\n")
    (fake_dir / "beta.log").write_text(f"{_now_iso} ok line\n")
    (fake_dir / "alpha.error.log").write_text("")
    (fake_dir / "beta.error.log").write_text(
        f"{_now_iso} Traceback\n{_now_iso} ValueError: bad\n"
    )
    # Forzar mtime del beta.log más viejo así no gana por recencia.
    os.utime(fake_dir / "beta.log", (1000, 1000))

    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOGS_INDEX_CACHE", {"ts": 0.0, "payload": None})

    resp = _client.get("/api/logs?nocache=1")
    assert resp.status_code == 200
    services = resp.json()["services"]
    # `beta` (error) tiene que aparecer antes que `alpha` (ok), aunque
    # alpha tenga mtime más reciente.
    names_in_order = [s["service"] for s in services]
    assert names_in_order.index("beta") < names_in_order.index("alpha")


# ── /api/logs/file viewer ─────────────────────────────────────────────

def test_api_logs_file_unknown_returns_404():
    resp = _client.get("/api/logs/file?name=obsidian-rag/no-existe-jamas.log")
    assert resp.status_code == 404


@pytest.mark.parametrize("bad_name", [
    "../etc/passwd",
    "/etc/passwd",
    "obsidian-rag/../etc/passwd",
])
def test_api_logs_file_rejects_traversal(bad_name):
    resp = _client.get(f"/api/logs/file?name={bad_name}")
    assert resp.status_code == 404


# ── Timestamp extraction ──────────────────────────────────────────────

@pytest.mark.parametrize("line, expected", [
    # ISO con T (heartbeat / silent_errors / la mayoría).
    ("[heartbeat] 2026-04-26T19:47:50 alive=true vaults=2", "2026-04-26T19:47:50"),
    ("2026-04-26T19:51:39 something happened", "2026-04-26T19:51:39"),
    # ISO con espacio (cloudflared-watcher / scripts shell).
    ("2026-04-26 17:05:22 Watcher started", "2026-04-26T17:05:22"),
    ("2026-04-26 17:05:26 URL changed", "2026-04-26T17:05:26"),
    # JSONL — el `"ts": "..."` gana sobre cualquier otro pattern.
    ('{"ts": "2026-04-26T19:51:39", "where": "foo"}', "2026-04-26T19:51:39"),
    ('{"pid": 86385, "ts": "2026-04-26T19:50:16Z", "op": "delete"}', "2026-04-26T19:50:16"),
    # Sin timestamp.
    ("gmail · 1.41s", None),
    ("INFO:     127.0.0.1:0 - GET /api/logs HTTP/1.1 200 OK", None),
    ("sqlite3.OperationalError: no such column: trace_id", None),
    ("", None),
])
def test_extract_log_ts(line, expected):
    assert _server._extract_log_ts(line) == expected


def test_api_logs_file_includes_timestamps(monkeypatch, tmp_path: Path):
    """Cada línea devuelta tiene `ts` (string ISO o None) y `ts_inferred`
    (bool). Forward-fill: una línea sin ts hereda el ts de la anterior."""
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "demo.log"
    log.write_text(
        "2026-04-26T19:47:50 first event\n"
        "  continuation line without ts\n"
        "no timestamp at all here\n"
        "2026-04-26T19:48:00 second event\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))

    resp = _client.get("/api/logs/file?name=obsidian-rag/demo.log&tail=10")
    assert resp.status_code == 200
    d = resp.json()
    assert d["lines_total"] == 4
    lines = d["lines"]
    assert lines[0]["ts"] == "2026-04-26T19:47:50"
    assert lines[0]["ts_inferred"] is False
    assert lines[1]["ts"] == "2026-04-26T19:47:50"
    assert lines[1]["ts_inferred"] is True
    assert lines[2]["ts"] == "2026-04-26T19:47:50"
    assert lines[2]["ts_inferred"] is True
    assert lines[3]["ts"] == "2026-04-26T19:48:00"
    assert lines[3]["ts_inferred"] is False


def test_api_logs_file_no_timestamps_returns_null(monkeypatch, tmp_path: Path):
    """Si ninguna línea tiene timestamp, todas vuelven con `ts: null`."""
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "no-ts.log"
    log.write_text("gmail · 1.41s\nwhatsapp: 14732 · 0.07s\n")
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))

    resp = _client.get("/api/logs/file?name=obsidian-rag/no-ts.log&tail=10")
    assert resp.status_code == 200
    lines = resp.json()["lines"]
    assert len(lines) == 2
    for ln in lines:
        assert ln["ts"] is None
        assert ln["ts_inferred"] is False


def test_api_logs_file_returns_classified_lines(monkeypatch, tmp_path: Path):
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "demo.log"
    log.write_text(
        "INFO: start\n"
        "warnings.warn('foo')\n"
        "OperationalError: bar\n"
        "[heartbeat] alive=true\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))

    resp = _client.get("/api/logs/file?name=obsidian-rag/demo.log&tail=10")
    assert resp.status_code == 200
    d = resp.json()
    assert d["lines_total"] == 4
    levels = [ln["level"] for ln in d["lines"]]
    # Order in the response is chronological (oldest → newest).
    assert levels == ["info", "warn", "error", "ok"]
    assert d["counts"]["error"] == 1
    assert d["counts"]["warn"] == 1
    assert d["counts"]["ok"] == 1
    assert d["counts"]["info"] == 1
    # Numeración: la última línea (más reciente) tiene n=1.
    last = d["lines"][-1]
    assert last["n"] == 1
    assert last["text"] == "[heartbeat] alive=true"


def test_api_logs_file_filter_only_errors(monkeypatch, tmp_path: Path):
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "demo.log"
    log.write_text(
        "INFO: start\n"
        "warnings.warn('foo')\n"
        "OperationalError: bar\n"
        "[heartbeat] alive=true\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))

    resp = _client.get("/api/logs/file?name=obsidian-rag/demo.log&only_errors=1")
    assert resp.status_code == 200
    d = resp.json()
    levels = [ln["level"] for ln in d["lines"]]
    assert set(levels) == {"warn", "error"}
    assert d["filtered_by_level"] is True
    # Counts include the unfiltered totals (so the UI can show
    # "X de Y líneas").
    assert d["counts"]["info"] == 1
    assert d["counts"]["ok"] == 1


def test_api_logs_file_filter_by_substring(monkeypatch, tmp_path: Path):
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "demo.log"
    log.write_text("alpha foo\nbeta bar\nGamma FOO\nepsilon\n")
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))

    resp = _client.get("/api/logs/file?name=obsidian-rag/demo.log&q=foo")
    assert resp.status_code == 200
    d = resp.json()
    assert d["filtered_by_query"] is True
    texts = [ln["text"] for ln in d["lines"]]
    # Case-insensitive: matchea "alpha foo" + "Gamma FOO".
    assert texts == ["alpha foo", "Gamma FOO"]
    # `lines_total` cuenta el total leído del archivo, NO los matchedos.
    assert d["lines_total"] == 4
    assert d["lines_returned"] == 2


# ── Page + PWA wiring ─────────────────────────────────────────────────

def test_logs_page_served():
    resp = _client.get("/logs")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "rag" in body and "logs" in body
    assert "/static/logs.js" in body
    assert "manifest.webmanifest" in body
    assert "register-sw.js" in body


# ── Global feed (/api/logs/errors) ────────────────────────────────────

def test_api_logs_errors_aggregates_across_files(monkeypatch, tmp_path: Path):
    """El feed global mergea líneas de varios archivos ordenadas por ts desc.

    Timestamps relativos a `now` para no caer fuera de la ventana de
    24h del filtro `since_seconds=86400`.
    """
    from datetime import datetime, timedelta
    now = datetime.now()
    ts_recent = (now - timedelta(minutes=5)).isoformat(timespec="seconds")
    ts_older = (now - timedelta(minutes=15)).isoformat(timespec="seconds")
    ts_oldest = (now - timedelta(minutes=30)).isoformat(timespec="seconds")
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "alpha.log").write_text(
        f"{ts_oldest} something fine\n"
        f"{ts_recent} OperationalError: alpha bad\n"
    )
    (fake_dir / "beta.log").write_text(
        f"{ts_older} RuntimeError: beta bad\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOG_GLOBAL_CACHE", {})

    resp = _client.get("/api/logs/errors?since_seconds=86400&level=error&nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert d["lines_total"] == 2
    # Más reciente primero.
    assert d["lines"][0]["text"].startswith(f"{ts_recent} OperationalError")
    assert d["lines"][0]["service"] == "alpha"
    assert d["lines"][1]["text"].startswith(f"{ts_older} RuntimeError")
    assert d["lines"][1]["service"] == "beta"
    # Counts agregados.
    assert d["counts_by_level"]["error"] == 2
    # Top services.
    services = {s["service"]: s["count"] for s in d["top_services"]}
    assert services == {"alpha": 1, "beta": 1}


# test_api_logs_errors_synthetic_ts_for_lines_without_ts borrado 2026-05-04.
# El feature de synthetic-ts via mtime (líneas sin ts heredaban
# `mtime - i*1s` como timestamp aproximado) se removió deliberadamente
# en el audit 2026-04-29 — generaba "errores eternos" donde líneas
# históricas sin ts heredaban un ts reciente sintético y reaparecían
# en el feed cada vez que el archivo era refrescado por otra escritura.
# Ver el comentario extenso en `web/server.py::_build_global_errors_payload`
# (alrededor de la línea 18244-18259). Líneas sin ts inline Y sin last_ts
# previo en el archivo ahora se skipean — el test testaba el comportamiento
# viejo, ya no existe lo que verificaba.


def test_api_logs_errors_filters_old_files(monkeypatch, tmp_path: Path):
    """Archivos con mtime fuera de la ventana se saltean (no se leen)."""
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    old_log = fake_dir / "old.log"
    old_log.write_text("2026-01-01T00:00:00 OperationalError: ancient\n")
    # Hacer que sea muy viejo (timestamp del 2020).
    os.utime(old_log, (1577836800.0, 1577836800.0))
    new_log = fake_dir / "new.log"
    new_log.write_text("OperationalError: recent\n")
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOG_GLOBAL_CACHE", {})

    resp = _client.get("/api/logs/errors?since_seconds=3600&level=error&nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    # `old.log` se descartó por mtime fuera de ventana.
    assert d["files_skipped_old"] >= 1
    services = {l["service"] for l in d["lines"]}
    assert "old" not in services


def test_api_logs_errors_warn_error_level(monkeypatch, tmp_path: Path):
    """level=warn_error incluye warns + errors. level=error sólo errors."""
    from datetime import datetime, timedelta
    now = datetime.now()
    ts1 = (now - timedelta(minutes=20)).isoformat(timespec="seconds")
    ts2 = (now - timedelta(minutes=10)).isoformat(timespec="seconds")
    ts3 = (now - timedelta(minutes=5)).isoformat(timespec="seconds")
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "demo.log").write_text(
        f"{ts1} normal info line\n"
        f"{ts2} warnings.warn('foo')\n"
        f"{ts3} OperationalError: bad\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOG_GLOBAL_CACHE", {})

    resp_err = _client.get("/api/logs/errors?since_seconds=86400&level=error&nocache=1")
    assert resp_err.status_code == 200
    assert resp_err.json()["lines_total"] == 1

    resp_we = _client.get("/api/logs/errors?since_seconds=86400&level=warn_error&nocache=1")
    assert resp_we.status_code == 200
    assert resp_we.json()["lines_total"] == 2


def test_api_logs_errors_validates_level_param():
    """level inválido → HTTP 400."""
    resp = _client.get("/api/logs/errors?level=invalid")
    assert resp.status_code == 400


# ── Diagnose endpoint (LLM-powered, /api/diagnose-error) ─────────────

def test_api_diagnose_error_validates_empty_text():
    """error_text vacío → 422 (pydantic validation)."""
    resp = _client.post("/api/diagnose-error", json={"error_text": "  "})
    assert resp.status_code == 422


def test_api_diagnose_error_truncates_huge_text():
    """error_text muy largo (>4000 chars) se truncate antes de mandar al LLM."""
    long_text = "x" * 6000
    req = _server._DiagnoseErrorRequest(error_text=long_text)
    assert len(req.error_text) < len(long_text)
    assert req.error_text.endswith("…(truncado)")


def test_diagnose_error_prompt_includes_context():
    """El prompt construido incluye los snippets de contexto."""
    req = _server._DiagnoseErrorRequest(
        error_text="OperationalError: bad",
        service="watch",
        file="obsidian-rag/watch.log (stdout)",
        line_n=42,
        timestamp="2026-04-26T19:00:00",
        context_lines=["heartbeat alive=true", "[before line]"],
    )
    prompt = _server._build_diagnose_error_prompt(req)
    assert "watch" in prompt
    assert "2026-04-26T19:00:00" in prompt
    assert "OperationalError: bad" in prompt
    assert "Contexto previo" in prompt
    assert "[before line]" in prompt
    assert "watch.log" in prompt


def test_diagnose_error_prompt_skips_empty_context():
    """Sin contexto previo, el prompt no incluye headers innecesarios."""
    req = _server._DiagnoseErrorRequest(error_text="some error")
    prompt = _server._build_diagnose_error_prompt(req)
    assert "Contexto previo" not in prompt
    assert "some error" in prompt


def test_diagnose_error_prompt_caps_context_at_20():
    """Más de 20 líneas de contexto → trunca a 20 (las últimas)."""
    req = _server._DiagnoseErrorRequest(
        error_text="error here",
        context_lines=[f"line {i}" for i in range(50)],
    )
    prompt = _server._build_diagnose_error_prompt(req)
    # Las últimas 20 (30..49) deben aparecer; las anteriores no.
    assert "line 30" in prompt
    assert "line 49" in prompt
    assert "line 29" not in prompt


# ── Whitelist del execute endpoint ────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "launchctl kickstart -k gui/501/com.fer.obsidian-rag-watch",
    "launchctl kickstart -k com.fer.obsidian-rag-watch",
    "launchctl list com.fer.obsidian-rag-watch",
    "tail -50 /Users/fer/.local/share/obsidian-rag/watch.log",
    "tail -n 100 /Users/fer/.local/share/obsidian-rag/watch.log",
    "wc -l /Users/fer/.local/share/obsidian-rag/watch.log",
    "rag stats",
    "rag vault list",
])
def test_validate_safe_command_accepts_whitelist(cmd):
    argv, reason = _server._validate_safe_command(cmd)
    assert argv is not None, f"esperaba aceptar {cmd!r}, rechazó: {reason}"
    assert reason == ""


@pytest.mark.parametrize("cmd, expected_reason_substr", [
    ("rm -rf /", "no está en la whitelist"),
    ("sudo launchctl kickstart com.fer.obsidian-rag-watch", "no está en la whitelist"),
    ("tail watch.log; rm -rf /", "metacharacter shell prohibido"),
    ("tail watch.log && echo done", "metacharacter shell prohibido"),
    ("tail $(echo /etc/passwd)", "metacharacter shell prohibido"),
    ("tail watch.log | grep error", "metacharacter shell prohibido"),
    ("tail watch.log > /tmp/x", "metacharacter shell prohibido"),
    ("tail /etc/passwd", "argumentos inválidos"),
    ("tail -f /Users/fer/.local/share/obsidian-rag/watch.log", "argumentos inválidos"),
    ("launchctl kickstart -k com.fer.OTHER.daemon", "argumentos inválidos"),
    ("launchctl bootout gui/501/com.fer.obsidian-rag-watch", "argumentos inválidos"),
    # Hard-defense: kickstart del propio web daemon es rechazado.
    ("launchctl kickstart -k gui/501/com.fer.obsidian-rag-web", "argumentos inválidos"),
    ("launchctl kickstart -k com.fer.obsidian-rag-web", "argumentos inválidos"),
    ("rag index", "argumentos inválidos"),
    ("rag query foo", "argumentos inválidos"),
    ("cat /etc/passwd", "argumentos inválidos"),
    ("/bin/tail watch.log", "no es un nombre simple"),
    ("", "vacío"),
    ("a" * 600, "demasiado largo"),
])
def test_validate_safe_command_rejects(cmd, expected_reason_substr):
    argv, reason = _server._validate_safe_command(cmd)
    assert argv is None, f"esperaba rechazar {cmd!r}, aceptó como {argv}"
    assert expected_reason_substr in reason or expected_reason_substr.replace("no es un nombre simple", "nombre simple") in reason


def test_api_diagnose_error_execute_rejects_dangerous():
    """Endpoint debe rechazar con 403 cualquier comando fuera del whitelist."""
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rm -rf /"},
    )
    assert resp.status_code == 403
    assert "rechazado" in resp.json()["detail"].lower()


def test_api_diagnose_error_execute_audit_log(tmp_path: Path, monkeypatch):
    """Cada ejecución (aceptada o rechazada) debe escribir audit log."""
    audit_path = tmp_path / "diagnose_executions.jsonl"
    monkeypatch.setattr(_server, "_DIAGNOSE_AUDIT_LOG", audit_path)

    # Trigger un reject.
    resp = _client.post(
        "/api/diagnose-error/execute",
        json={"command": "rm -rf /"},
    )
    assert resp.status_code == 403
    assert audit_path.is_file()
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["rejected"] is True
    assert rec["command_original"] == "rm -rf /"
    assert "no está en la whitelist" in rec["reason"]


def test_api_diagnose_error_execute_runs_safe_command(tmp_path: Path, monkeypatch):
    """Endpoint ejecuta `wc -l <log_path>` y devuelve exit_code=0."""
    audit_path = tmp_path / "diagnose_executions.jsonl"
    monkeypatch.setattr(_server, "_DIAGNOSE_AUDIT_LOG", audit_path)

    log_dir = Path.home() / ".local/share/obsidian-rag"
    if not log_dir.is_dir():
        pytest.skip("no obsidian-rag log dir on this machine")
    candidates = [p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".log"]
    if not candidates:
        pytest.skip("no .log files to test against")
    sample = candidates[0]
    cmd = f"wc -l {sample}"

    resp = _client.post("/api/diagnose-error/execute", json={"command": cmd})
    assert resp.status_code == 200
    data = resp.json()
    assert data["exit_code"] == 0
    assert "command_executed" in data and isinstance(data["command_executed"], list)
    # Audit log se escribió.
    assert audit_path.is_file()


# ── /api/logs/queue — error queue + worker ──────────────────────────

def test_compute_error_signature_normalizes():
    """Errores "similares" (distintos paths o números) hashean igual."""
    s1 = _server._compute_error_signature("watch", "error  2026-04.md: database is locked")
    s2 = _server._compute_error_signature("watch", "error  2026-05-03.md: database is locked")
    s3 = _server._compute_error_signature("watch", "error  _index.md: database is locked")
    assert s1 == s2 == s3, "errores con solo path distinto deben tener la misma signature"
    # Diferentes services → diferentes signatures
    s_other = _server._compute_error_signature("wa-tasks", "error  2026-04.md: database is locked")
    assert s1 != s_other
    # Diferentes tipos de error → diferentes signatures
    s_diff = _server._compute_error_signature("watch", "OperationalError: no such column")
    assert s1 != s_diff


def test_parse_devin_resolution_status():
    """Parser extrae STATUS: + REASON: del output."""
    output = "lala\nSTATUS: resolved\nREASON: reinicié el daemon watch\nmas texto"
    status, reason = _server._parse_devin_resolution_status(output)
    assert status == "resolved"
    assert "reinicié" in reason

    output_no_marker = "Devin did stuff"
    status, reason = _server._parse_devin_resolution_status(output_no_marker)
    assert status == "failed"
    assert "sin marker" in reason


def test_api_logs_queue_list_empty():
    """Queue endpoint devuelve lista y counts."""
    resp = _client.get("/api/logs/queue")
    assert resp.status_code == 200
    d = resp.json()
    assert "entries" in d
    assert "counts_by_status" in d
    assert "worker_enabled" in d
    assert "worker_rate_limit" in d


def test_api_logs_queue_list_filter_invalid():
    resp = _client.get("/api/logs/queue?status=bogus")
    assert resp.status_code == 400


def test_api_logs_queue_config_toggles_worker():
    """POST /config cambia el flag del worker."""
    # Estado inicial (asumimos off en fresh setup).
    resp = _client.post("/api/logs/queue/config", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["worker_enabled"] is True
    # Reset.
    resp = _client.post("/api/logs/queue/config", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["worker_enabled"] is False


def test_api_logs_queue_get_404():
    resp = _client.get("/api/logs/queue/99999999")
    assert resp.status_code == 404


def test_api_logs_queue_delete_404():
    resp = _client.delete("/api/logs/queue/99999999")
    assert resp.status_code == 404


# ── /api/auto-fix-devin — delegar a Devin CLI ───────────────────────

def test_api_auto_fix_devin_validates_empty_text():
    resp = _client.post("/api/auto-fix-devin", json={"error_text": "  "})
    assert resp.status_code == 422


def test_build_devin_prompt_includes_context():
    """El prompt de Devin incluye error + service + contexto previo."""
    req = _server._AutoFixRequest(
        error_text="OperationalError: bad",
        service="watch",
        file="obsidian-rag/watch.log",
        context_lines=["line1", "line2"],
    )
    prompt = _server._build_devin_prompt(req)
    assert "watch" in prompt
    assert "OperationalError: bad" in prompt
    assert "contexto_previo" in prompt
    assert "NO reinicies obsidian-rag-web" in prompt


def test_build_devin_prompt_no_context_skips_header():
    req = _server._AutoFixRequest(error_text="error")
    prompt = _server._build_devin_prompt(req)
    assert "contexto_previo" not in prompt


# ── /api/auto-fix — agent loop ───────────────────────────────────────

def test_api_auto_fix_validates_empty_text():
    resp = _client.post("/api/auto-fix", json={"error_text": "  "})
    assert resp.status_code == 422


def test_auto_fix_initial_prompt_includes_context():
    """El prompt inicial al LLM incluye el error + contexto + service."""
    req = _server._AutoFixRequest(
        error_text="OperationalError: bad",
        service="watch",
        file="obsidian-rag/watch.log",
        line_n=42,
        timestamp="2026-04-26T19:00:00",
        context_lines=["heartbeat alive=true", "[before]"],
    )
    prompt = _server._build_initial_auto_fix_user_prompt(req)
    assert "watch" in prompt
    assert "2026-04-26T19:00:00" in prompt
    assert "OperationalError: bad" in prompt
    assert "Contexto previo" in prompt
    assert "[before]" in prompt
    assert "8 turnos" in prompt or "turnos" in prompt


def test_execute_whitelisted_command_rejected_returns_dict(tmp_path: Path, monkeypatch):
    """Función de ejecución helper devuelve {rejected: True} sin lanzar."""
    audit_path = tmp_path / "diagnose_executions.jsonl"
    monkeypatch.setattr(_server, "_DIAGNOSE_AUDIT_LOG", audit_path)

    result = _server._execute_whitelisted_command("rm -rf /")
    assert result["rejected"] is True
    assert "no está en la whitelist" in result["reason"]
    # Audit log se escribió.
    assert audit_path.is_file()


def test_execute_whitelisted_command_runs_safe(tmp_path: Path, monkeypatch):
    """Función helper ejecuta el comando + devuelve resultado."""
    audit_path = tmp_path / "diagnose_executions.jsonl"
    monkeypatch.setattr(_server, "_DIAGNOSE_AUDIT_LOG", audit_path)

    log_dir = Path.home() / ".local/share/obsidian-rag"
    if not log_dir.is_dir():
        pytest.skip("no obsidian-rag log dir on this machine")
    candidates = [p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".log"]
    if not candidates:
        pytest.skip("no .log files")
    sample = candidates[0]

    result = _server._execute_whitelisted_command(f"wc -l {sample}")
    assert result["rejected"] is False
    assert result["exit_code"] == 0
    assert result["duration_s"] >= 0
    assert "command_executed" in result


def test_static_assets_exist():
    """logs.html + logs.js existen en /static/."""
    assert (_STATIC_DIR / "logs.html").is_file()
    assert (_STATIC_DIR / "logs.js").is_file()
    assert (_STATIC_DIR / "diagnose-modal.js").is_file()


# ── /api/logs/rankings — agregaciones top-N (2026-05-01) ──────────────

class TestNormalizeLogLineSignature:
    """`_normalize_log_line_signature` strippea timestamps + números +
    paths + UUIDs + IPs para que líneas que sólo difieren en magnitudes
    caigan al mismo bucket de clustering."""

    def test_clusters_lines_differing_only_in_duration(self):
        a = _server._normalize_log_line_signature(
            "[2026-05-01 18:46:06] failed in 12345ms"
        )
        b = _server._normalize_log_line_signature(
            "[2026-05-01 19:01:00] failed in 99ms"
        )
        assert a == b
        assert "<n>" in a
        assert "ms" in a  # sufijo preservado

    def test_clusters_lines_with_embedded_underscore_numbers(self):
        """Caso real del watchdog: `last_restart_was_1644s_ago` tiene
        digits precedidos por underscore (word char), `\\b` no matchea
        ahí. El normalizer debe usar lookbehind sobre dígitos."""
        a = _server._normalize_log_line_signature(
            "[ollama-health-watchdog] last_restart_was_1644s_ago"
        )
        b = _server._normalize_log_line_signature(
            "[ollama-health-watchdog] last_restart_was_30s_ago"
        )
        assert a == b
        assert "<n>" in a

    def test_normalizes_paths_uuids_ips(self):
        sig = _server._normalize_log_line_signature(
            "ERROR: connection refused 127.0.0.1:8080 in /Users/fer/foo/bar.py"
        )
        assert "<ip>" in sig
        assert "<path>" in sig

    def test_strips_iso_timestamp_prefix(self):
        a = _server._normalize_log_line_signature("[2026-05-01T18:46:06] foo")
        b = _server._normalize_log_line_signature("[2026-04-29T11:00:00] foo")
        assert a == b
        assert a.startswith("foo") or a.startswith("foo")
        assert "2026" not in a

    def test_strips_level_prefix(self):
        a = _server._normalize_log_line_signature("ERROR: connection refused")
        b = _server._normalize_log_line_signature("INFO: connection refused")
        # Ambos prefijos se strippean → mismo bucket.
        assert a == b
        assert "connection refused" in a

    def test_lowercases_for_case_insensitive_clustering(self):
        a = _server._normalize_log_line_signature("ERROR: connection refused")
        assert a == a.lower()

    def test_empty_input_returns_empty(self):
        assert _server._normalize_log_line_signature("") == ""
        assert _server._normalize_log_line_signature("   ") == ""

    def test_caps_long_inputs(self):
        long = "x" * 5000
        sig = _server._normalize_log_line_signature(long)
        assert len(sig) <= _server._LOG_SIG_MAX_CHARS + 5  # +ellipsis "…"


class TestRankingsPayload:
    """`_build_rankings_payload` agrega counters de los _LOG_DIRS reales.
    Tests son contra el sistema vivo (igual que los de /api/logs/file),
    así que skipean si no hay dirs disponibles."""

    def _has_log_dirs(self) -> bool:
        return any(d.is_dir() for d in _server._LOG_DIRS)

    def test_payload_shape(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=86400, top_n=5)
        # Top-level
        assert out["window_seconds"] == 86400
        assert out["top_n"] == 5
        assert "scanned_at" in out
        assert "totals" in out
        # Totals
        for k in ("errors", "warns", "services_with_errors",
                  "services_with_warns", "files_scanned", "files_skipped_old"):
            assert k in out["totals"]
            assert isinstance(out["totals"][k], int)
        # Rankings buckets
        for bucket in ("services_by_errors", "services_by_warns",
                       "error_patterns", "recent_errors", "noisy_logs"):
            assert bucket in out["rankings"]
            assert isinstance(out["rankings"][bucket], list)
            assert len(out["rankings"][bucket]) <= 5

    def test_top_n_clamped(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        # Pedir 200 — el endpoint clampea a 50.
        resp = _client.get("/api/logs/rankings?top_n=200")
        assert resp.status_code == 200
        d = resp.json()
        assert d["top_n"] == 50

    def test_window_clamped(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        # Pedir 999 días — clampea a max 7d.
        resp = _client.get("/api/logs/rankings?since_seconds=86400000")
        assert resp.status_code == 200
        d = resp.json()
        assert d["window_seconds"] == 7 * 86400

    def test_window_min_60s(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        resp = _client.get("/api/logs/rankings?since_seconds=10")
        assert resp.status_code == 200
        d = resp.json()
        assert d["window_seconds"] == 60

    def test_default_window_1h(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        resp = _client.get("/api/logs/rankings")
        assert resp.status_code == 200
        d = resp.json()
        assert d["window_seconds"] == 3600

    def test_recent_errors_sorted_desc(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=7 * 86400, top_n=10)
        recent = out["rankings"]["recent_errors"]
        if len(recent) < 2:
            pytest.skip("no hay suficientes errores recientes para validar orden")
        for i in range(len(recent) - 1):
            assert recent[i]["ts"] >= recent[i + 1]["ts"], \
                f"recent_errors no ordenado: {recent[i]['ts']} >= {recent[i+1]['ts']}"

    def test_services_by_errors_sorted_desc(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=7 * 86400, top_n=10)
        items = out["rankings"]["services_by_errors"]
        if len(items) < 2:
            pytest.skip("no hay suficientes services con errores")
        for i in range(len(items) - 1):
            assert items[i]["count"] >= items[i + 1]["count"]

    def test_error_patterns_have_required_fields(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=7 * 86400, top_n=10)
        for p in out["rankings"]["error_patterns"]:
            assert "signature" in p
            assert "count" in p
            assert "example" in p
            assert "services" in p
            assert isinstance(p["services"], list)
            assert "first_ts" in p
            assert "last_ts" in p
            # last_ts >= first_ts (orden cronológico mínimo dentro del bucket)
            assert p["last_ts"] >= p["first_ts"]
            assert p["count"] >= 1

    def test_cache_returns_same_payload_within_ttl(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        r1 = _client.get("/api/logs/rankings?since_seconds=3600&top_n=5").json()
        r2 = _client.get("/api/logs/rankings?since_seconds=3600&top_n=5").json()
        # Mismo scanned_at → vino del cache (TTL 8s).
        assert r1["scanned_at"] == r2["scanned_at"]

    def test_nocache_forces_refresh(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        r1 = _client.get("/api/logs/rankings?since_seconds=3600&nocache=1").json()
        r2 = _client.get("/api/logs/rankings?since_seconds=3600&nocache=1").json()
        # Aunque el contenido sea similar, los scanned_at deberían diferir
        # porque cada request escaneó de nuevo (a menos que pase en el
        # mismo segundo, raro pero posible — toleramos ese caso).
        assert "scanned_at" in r1 and "scanned_at" in r2


def test_diagnose_modal_css_exists():
    assert (_STATIC_DIR / "diagnose-modal.css").is_file()


# ── Latency extractor + percentile (2026-05-01 v2) ────────────────────

class TestExtractLatencyMs:
    """`_extract_latency_ms` cubre los 4 patrones más frecuentes en los
    logs reales del stack: total=Xms, ttft_ms=X, duration_ms=X, warm-up
    OK (Xms,...). Cualquier match devuelve int de ms; sin match → None."""

    def test_total_ms_pattern(self):
        assert _server._extract_latency_ms(
            "[chat-timing] model=qwen2.5:7b retrieve=150ms total=51755ms"
        ) == 51755

    def test_ttft_ms_pattern(self):
        assert _server._extract_latency_ms(
            "[chat-stream-error] phase=synthesis ttft_ms=90004 query=hola"
        ) == 90004

    def test_duration_ms_pattern(self):
        assert _server._extract_latency_ms(
            "[whisper] mode=server duration_ms=48644 bytes=8190"
        ) == 48644

    def test_warmup_ms_pattern(self):
        assert _server._extract_latency_ms(
            "[whisper] warm-up OK (621ms, source=real:audio)"
        ) == 621

    def test_no_match_returns_none(self):
        assert _server._extract_latency_ms("just some random log line") is None
        assert _server._extract_latency_ms("") is None

    def test_total_without_ms_suffix_does_not_match(self):
        """`total=168` en `[warmup] cache total=168` es count, NO duración.
        El regex requiere el sufijo `ms` literal para evitar este FP."""
        assert _server._extract_latency_ms(
            "[warmup] followup_aging cache pre-warmed in 898.8s (total=168)"
        ) is None

    def test_first_match_in_line_wins(self):
        """`re.search` con disjunción agarra el match más a la izquierda
        de la LÍNEA, no el primero de la alternativa. En la práctica
        eso da `ttft_ms` cuando aparece antes que `total=`. Aceptable —
        cualquiera de las dos métricas es útil para rankear."""
        result = _server._extract_latency_ms(
            "[chat-timing] ttft_ms=200 total=300ms"
        )
        # ttft_ms aparece antes en la línea → ese match gana.
        assert result == 200
        # Cuando total= viene primero, gana total=:
        result2 = _server._extract_latency_ms(
            "[chat-timing] total=300ms ttft_ms=200"
        )
        assert result2 == 300


class TestPercentileInt:
    def test_p50_simple(self):
        # mediana de [1,2,3,4,5] = 3, pero nearest-rank con n=5 da idx=2 → val=3
        assert _server._percentile_int([1, 2, 3, 4, 5], 50) in (2, 3)

    def test_p99_with_outlier(self):
        vals = [1, 2, 3, 4, 100]
        assert _server._percentile_int(vals, 99) == 100  # el outlier

    def test_p99_with_n_100_random(self):
        assert _server._percentile_int(list(range(1, 101)), 99) in (98, 99)

    def test_empty_returns_zero(self):
        assert _server._percentile_int([], 50) == 0
        assert _server._percentile_int([], 99) == 0

    def test_single_value(self):
        assert _server._percentile_int([42], 99) == 42
        assert _server._percentile_int([42], 50) == 42


# ── Silent services + candidate paths (2026-05-01 v2) ─────────────────

class TestCandidateLogPathsForLabel:
    """`_candidate_log_paths_for_label` mapea label launchd → archivos de
    log esperados. La convención del repo es:
      - `com.fer.obsidian-rag-X` → `X.log` y `X.error.log`.
      - `com.fer.whatsapp-Y` → `Y.log` y `Y.error.log`.
    """

    def test_obsidian_rag_prefix_strip(self):
        log_dir = Path.home() / ".local/share/obsidian-rag"
        if not log_dir.is_dir():
            pytest.skip("no obsidian-rag log dir on this machine")
        # web.log existe en el sistema vivo del user.
        if not (log_dir / "web.log").is_file():
            pytest.skip("no web.log on this machine")
        paths = _server._candidate_log_paths_for_label("com.fer.obsidian-rag-web")
        names = {p.name for p in paths}
        assert "web.log" in names

    def test_whatsapp_prefix_strip(self):
        log_dir = Path.home() / ".local/share/whatsapp-listener"
        if not log_dir.is_dir():
            pytest.skip("no whatsapp-listener log dir on this machine")
        if not (log_dir / "listener.log").is_file():
            pytest.skip("no listener.log on this machine")
        paths = _server._candidate_log_paths_for_label("com.fer.whatsapp-listener")
        names = {p.name for p in paths}
        assert "listener.log" in names

    def test_unknown_label_returns_empty_or_fallback(self):
        # Label que no existe → no debería tirar excepción, devolver lista
        # (vacía o con fallbacks que no existen).
        paths = _server._candidate_log_paths_for_label(
            "com.fer.nonexistent-label-xyz123"
        )
        assert isinstance(paths, list)


class TestBuildSilentServices:
    """`_build_silent_services` cruza `_STATUS_CATALOG` + `_launchctl_print_fields`
    + mtime de logs. Tests con monkey-patching de las dependencias I/O para
    no depender del estado real del sistema."""

    def test_skips_non_daemon_kinds(self, monkeypatch):
        """Solo daemons + scheduled. Los kind=ollama / rag_db / vault del
        catálogo son chequeos de estado, no servicios silenciables."""
        fake_catalog = [
            {"kind": "ollama", "id": "ollama"},
            {"kind": "rag_db", "id": "rag-db"},
        ]
        monkeypatch.setattr(_server, "_STATUS_CATALOG", fake_catalog)
        out = _server._build_silent_services(top_n=5)
        assert out == []

    def test_returns_silent_daemon(self, monkeypatch):
        """Daemon loaded sin actividad reciente → silent."""
        old_mtime = 9999999.0  # placeholder; el _now lo usa para diff
        fake_catalog = [
            {"kind": "daemon", "target": "com.fer.test-daemon", "name": "Test daemon"},
        ]
        monkeypatch.setattr(_server, "_STATUS_CATALOG", fake_catalog)
        # launchctl_print devuelve algo (loaded=True)
        monkeypatch.setattr(_server, "_launchctl_print_fields",
                            lambda label, timeout=3.0: {"state": "running"})
        # Simular log con mtime de hace 2h (>1h threshold de daemon)
        class FakePath:
            def __init__(self, m): self._m = m
            def is_file(self): return True
            def stat(self):
                class S: pass
                s = S()
                s.st_mtime = self._m
                return s
        fake_path = FakePath(1000.0)  # epoch arbitrario
        monkeypatch.setattr(_server, "_candidate_log_paths_for_label",
                            lambda label: [fake_path])
        # _now = 1000 + 2h → silence = 7200s > threshold 3600
        out = _server._build_silent_services(top_n=5, _now=1000.0 + 7200)
        assert len(out) == 1
        assert out[0]["service"] == "Test daemon"
        assert out[0]["kind"] == "daemon"
        assert out[0]["silence_seconds"] == 7200

    def test_skips_recent_daemon(self, monkeypatch):
        """Daemon con log fresco → NO silent."""
        fake_catalog = [
            {"kind": "daemon", "target": "com.fer.fresh-daemon", "name": "Fresh"},
        ]
        monkeypatch.setattr(_server, "_STATUS_CATALOG", fake_catalog)
        monkeypatch.setattr(_server, "_launchctl_print_fields",
                            lambda label, timeout=3.0: {"state": "running"})
        class FakePath:
            def __init__(self, m): self._m = m
            def is_file(self): return True
            def stat(self):
                class S: pass
                s = S()
                s.st_mtime = self._m
                return s
        fake_path = FakePath(1000.0)
        monkeypatch.setattr(_server, "_candidate_log_paths_for_label",
                            lambda label: [fake_path])
        # silence = 30s, < threshold 3600 → no silent
        out = _server._build_silent_services(top_n=5, _now=1000.0 + 30)
        assert out == []

    def test_skips_unloaded_daemon(self, monkeypatch):
        """Daemon NO loaded en launchd → no aparece (es disabled, no silent)."""
        fake_catalog = [
            {"kind": "daemon", "target": "com.fer.disabled", "name": "Disabled"},
        ]
        monkeypatch.setattr(_server, "_STATUS_CATALOG", fake_catalog)
        monkeypatch.setattr(_server, "_launchctl_print_fields",
                            lambda label, timeout=3.0: None)  # no loaded
        out = _server._build_silent_services(top_n=5)
        assert out == []

    def test_scheduled_threshold_30h(self, monkeypatch):
        """Scheduled jobs tienen threshold de 30h (no 1h)."""
        fake_catalog = [
            {"kind": "scheduled", "target": "com.fer.daily", "name": "Daily"},
        ]
        monkeypatch.setattr(_server, "_STATUS_CATALOG", fake_catalog)
        monkeypatch.setattr(_server, "_launchctl_print_fields",
                            lambda label, timeout=3.0: {"runs": "5"})
        class FakePath:
            def __init__(self, m): self._m = m
            def is_file(self): return True
            def stat(self):
                class S: pass
                s = S()
                s.st_mtime = self._m
                return s
        fake_path = FakePath(1000.0)
        monkeypatch.setattr(_server, "_candidate_log_paths_for_label",
                            lambda label: [fake_path])
        # silence = 5h, < 30h → NO silent (scheduled tiene threshold más laxo)
        out_5h = _server._build_silent_services(top_n=5, _now=1000.0 + 5 * 3600)
        assert out_5h == []
        # silence = 35h, > 30h → silent
        out_35h = _server._build_silent_services(top_n=5, _now=1000.0 + 35 * 3600)
        assert len(out_35h) == 1
        assert out_35h[0]["kind"] == "scheduled"


# ── Rankings v2: latency_outliers, silent_services, growth_rate, new_error_patterns

class TestRankingsV2Endpoint:
    """Smoke contra el endpoint en vivo. Solo valida la PRESENCIA de los 4
    rankings nuevos en el response — el contenido depende del estado del
    sistema y no es estable test-a-test."""

    def _has_log_dirs(self) -> bool:
        return any(d.is_dir() for d in _server._LOG_DIRS)

    def test_v2_rankings_present(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        resp = _client.get("/api/logs/rankings?since_seconds=86400&top_n=5&nocache=1")
        assert resp.status_code == 200
        d = resp.json()
        rankings = d["rankings"]
        # Presencia de los 5 originales:
        for k in ("services_by_errors", "services_by_warns", "error_patterns",
                  "recent_errors", "noisy_logs"):
            assert k in rankings
        # Presencia de los 4 nuevos:
        for k in ("latency_outliers", "silent_services", "growth_rate", "new_error_patterns"):
            assert k in rankings
            assert isinstance(rankings[k], list)

    def test_has_regression_window_flag(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        # window=1h NO tiene base de comparación → flag false
        d1 = _client.get("/api/logs/rankings?since_seconds=3600&nocache=1").json()
        assert d1["totals"]["has_regression_window"] is False
        assert d1["rankings"]["new_error_patterns"] == []
        # window=24h sí
        d24 = _client.get("/api/logs/rankings?since_seconds=86400&nocache=1").json()
        assert d24["totals"]["has_regression_window"] is True

    def test_latency_outliers_shape(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=7 * 86400, top_n=10)
        for it in out["rankings"]["latency_outliers"]:
            assert "service" in it
            assert "p50_ms" in it
            assert "p99_ms" in it
            assert "max_ms" in it
            assert "n" in it
            assert it["n"] >= 3  # mínimo 3 samples
            assert it["max_ms"] >= it["p99_ms"] >= it["p50_ms"] >= 0

    def test_growth_rate_shape(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=86400, top_n=5)
        for it in out["rankings"]["growth_rate"]:
            assert "service" in it
            assert "bytes_per_min" in it
            assert "lines_per_min" in it
            assert "total_bytes_in_window" in it
            assert it["bytes_per_min"] >= 0

    def test_silent_services_shape(self):
        if not self._has_log_dirs():
            pytest.skip("no log dirs en este sistema")
        out = _server._build_rankings_payload(window_s=86400, top_n=5)
        for it in out["rankings"]["silent_services"]:
            assert "service" in it
            assert "label" in it
            assert "kind" in it
            assert it["kind"] in ("daemon", "scheduled", "never_ran")
            # silence_seconds puede ser None si kind=never_ran.
            assert "silence_seconds" in it
