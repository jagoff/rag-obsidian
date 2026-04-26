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

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_STATIC_DIR = Path(_server.STATIC_DIR)
_client = TestClient(_server.app)


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
    """
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "watch.log").write_text("[heartbeat] alive=true\n")
    (fake_dir / "watch.error.log").write_text(
        "Traceback (most recent call last):\n"
        "  File 'foo'\n"
        "OperationalError: no such column: trace_id\n"
    )
    (fake_dir / "anticipate.log").write_text("Selected: foo\nno pusheado: ok\n")
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
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "alpha.log").write_text("ok line\n")
    (fake_dir / "beta.log").write_text("ok line\n")
    (fake_dir / "alpha.error.log").write_text("")
    (fake_dir / "beta.error.log").write_text("Traceback\nValueError: bad\n")
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
    """El feed global mergea líneas de varios archivos ordenadas por ts desc."""
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "alpha.log").write_text(
        "2026-04-26T10:00:00 something fine\n"
        "2026-04-26T11:00:00 OperationalError: alpha bad\n"
    )
    (fake_dir / "beta.log").write_text(
        "2026-04-26T10:30:00 RuntimeError: beta bad\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOG_GLOBAL_CACHE", {})

    resp = _client.get("/api/logs/errors?since_seconds=86400&level=error&nocache=1")
    assert resp.status_code == 200
    d = resp.json()
    assert d["lines_total"] == 2
    # Más reciente primero.
    assert d["lines"][0]["text"].startswith("2026-04-26T11:00:00 OperationalError")
    assert d["lines"][0]["service"] == "alpha"
    assert d["lines"][1]["text"].startswith("2026-04-26T10:30:00 RuntimeError")
    assert d["lines"][1]["service"] == "beta"
    # Counts agregados.
    assert d["counts_by_level"]["error"] == 2
    # Top services.
    services = {s["service"]: s["count"] for s in d["top_services"]}
    assert services == {"alpha": 1, "beta": 1}


def test_api_logs_errors_synthetic_ts_for_lines_without_ts(monkeypatch, tmp_path: Path):
    """Cuando una línea no tiene ts, el feed global usa mtime - offset
    como timestamp aproximado y la marca con `ts_synthetic=true`."""
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    log = fake_dir / "no-ts.log"
    log.write_text(
        "OperationalError: bad thing happened\n"
        "Another error line\n"
    )
    monkeypatch.setattr(_server, "_LOG_DIRS", (fake_dir,))
    monkeypatch.setattr(_server, "_LOG_GLOBAL_CACHE", {})

    resp = _client.get("/api/logs/errors?since_seconds=86400&level=error&nocache=1")
    assert resp.status_code == 200
    lines = resp.json()["lines"]
    assert len(lines) == 2
    for ln in lines:
        assert ln["ts_synthetic"] is True
        assert ln["ts"] is not None
        assert ln["service"] == "no-ts"


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
    fake_dir = tmp_path / "obsidian-rag"
    fake_dir.mkdir()
    (fake_dir / "demo.log").write_text(
        "2026-04-26T10:00:00 normal info line\n"
        "2026-04-26T10:01:00 warnings.warn('foo')\n"
        "2026-04-26T10:02:00 OperationalError: bad\n"
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


def test_static_assets_exist():
    """logs.html + logs.js existen en /static/."""
    assert (_STATIC_DIR / "logs.html").is_file()
    assert (_STATIC_DIR / "logs.js").is_file()
