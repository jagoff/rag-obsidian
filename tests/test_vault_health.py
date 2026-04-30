"""Tests para `rag.vault_health` + endpoint `/api/vault/health`.

Cubre:
  1. Vault vacío → score=0 + componentes en 0.
  2. 100 notas todas con tags → tags_pct=100.
  3. Cálculo del weighted score correcto a mano.
  4. Endpoint `/api/vault/health` devuelve JSON parseable.
  5. Cache TTL: 2 calls dentro de 5min hacen 1 sola pasada de queries.
  6. DB locked / inaccesible → endpoint devuelve {score: null, error} HTTP 200.

Cada test crea fixtures locales (ragvec.db + telemetry.db en tmp) y
redirige `rag.DB_PATH` para que el módulo lea desde ahí en lugar del
storage real del usuario. La autouse `_isolate_vault_path` de
`conftest.py` ya redirige `rag.VAULT_PATH` al tmp, pero `DB_PATH` lo
fijamos acá explícitamente porque ese módulo lo lee al import.

Nota sobre la tabla meta_*: el nombre real depende del slug del vault
(`COLLECTION_NAME` de rag/__init__.py). Como en los tests el vault es
un tmp dir distinto del default, `COLLECTION_NAME` resuelve a algo tipo
`obsidian_notes_v11_<8hex>`. Para que el módulo encuentre la tabla en
los fixtures, leemos el nombre vivo desde `rag.COLLECTION_NAME` y
creamos la tabla con ese nombre exacto.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Fixtures comunes ─────────────────────────────────────────────────────


def _create_meta_table(conn: sqlite3.Connection, name: str) -> None:
    """Crea la tabla meta_<name> con el schema real (subset de columnas
    que vault_health necesita: file, note, tags, outlinks, extra_json).
    Resto de columnas opcionales (TEXT NULL) para parity con producción."""
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{name}" ('
        " rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
        " document TEXT, file TEXT, folder TEXT, note TEXT, tags TEXT, "
        " hash TEXT, outlinks TEXT, parent TEXT, title TEXT, area TEXT, "
        " type TEXT, archived_at TEXT, archived_from TEXT, archived_reason TEXT, "
        " contradicts TEXT, ambient TEXT, url TEXT, anchor TEXT, line INTEGER, "
        " source TEXT, profile TEXT, bookmark_folder TEXT, "
        " created_ts REAL, extra_json TEXT)"
    )


def _seed_notes(conn: sqlite3.Connection, table: str, notes: list[dict]) -> None:
    """Inserta una row por nota — vault_health agrupa por `file` así que
    una nota = una row es válida (queries usan MAX/GROUP BY)."""
    for i, n in enumerate(notes):
        outlinks = ",".join(n.get("outlinks", []))
        tags = ",".join(n.get("tags", []))
        extra = {}
        if "modified" in n:
            extra["modified"] = n["modified"]
        if "created" in n:
            extra["created"] = n["created"]
        extra_json = json.dumps(extra) if extra else None
        conn.execute(
            f'INSERT INTO "{table}" '
            "(chunk_id, file, note, tags, outlinks, hash, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"chunk-{i}",
                n["file"],
                n.get("note") or Path(n["file"]).stem,
                tags,
                outlinks,
                n.get("hash", f"hash-{i}"),
                extra_json,
            ),
        )


def _create_telemetry_tables(conn: sqlite3.Connection) -> None:
    """Crea rag_contradictions + rag_queries con schema real."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_contradictions ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
        " subject_path TEXT NOT NULL, contradicts_json TEXT, "
        " helper_raw TEXT, skipped TEXT, UNIQUE(ts, subject_path))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_queries ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
        " cmd TEXT, q TEXT NOT NULL, session TEXT, mode TEXT, "
        " top_score REAL, t_retrieve REAL, t_gen REAL, answer_len INTEGER, "
        " citation_repaired INTEGER, critique_fired INTEGER, "
        " critique_changed INTEGER, variants_json TEXT, paths_json TEXT, "
        " scores_json TEXT, filters_json TEXT, bad_citations_json TEXT, "
        " extra_json TEXT, trace_id TEXT)"
    )


@pytest.fixture
def vh_env(tmp_path, monkeypatch):
    """Entorno aislado: DB_PATH → tmp_path, ragvec.db + telemetry.db creados.

    Devuelve un dict con:
      - "tmp": tmp_path
      - "ragvec_db": Path a ragvec.db
      - "telemetry_db": Path a telemetry.db
      - "meta_table": nombre de la tabla meta_* que vault_health buscará
    """
    import rag
    from rag import vault_health as vh

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Invalidar el cache module-level antes Y después del test para que
    # tests previos no nos contaminen, ni nosotros a los siguientes.
    vh.invalidate_cache()

    ragvec_db = tmp_path / "ragvec.db"
    telemetry_db = tmp_path / rag._TELEMETRY_DB_FILENAME

    # vault_health lee `meta_{COLLECTION_NAME}`. En CI/dev sin vault real,
    # COLLECTION_NAME puede no existir como tabla — vault_health hace
    # fallback a "meta_obsidian_notes_v11" si la primaria no está. Creamos
    # ambas con la misma data (la fallback se usa en tests si la
    # specific-vault no matchea). En la práctica, la tabla con sufijo
    # depende del slug del tmp vault y vault_health la encontraría primero.
    meta_table = f"meta_{rag.COLLECTION_NAME}"

    conn = sqlite3.connect(str(ragvec_db))
    try:
        _create_meta_table(conn, meta_table)
        # Fallback table también — para que el test funcione aunque
        # COLLECTION_NAME haya cambiado entre fixture-create y module-read.
        if meta_table != "meta_obsidian_notes_v11":
            _create_meta_table(conn, "meta_obsidian_notes_v11")
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(str(telemetry_db))
    try:
        _create_telemetry_tables(conn)
        conn.commit()
    finally:
        conn.close()

    yield {
        "tmp": tmp_path,
        "ragvec_db": ragvec_db,
        "telemetry_db": telemetry_db,
        "meta_table": meta_table,
    }
    vh.invalidate_cache()


def _seed_meta(env, notes: list[dict]) -> None:
    conn = sqlite3.connect(str(env["ragvec_db"]))
    try:
        _seed_notes(conn, env["meta_table"], notes)
        # También en la fallback table — vault_health prefiere la sufijada
        # cuando existe pero algunos entornos la dejan vacía.
        if env["meta_table"] != "meta_obsidian_notes_v11":
            _seed_notes(conn, "meta_obsidian_notes_v11", notes)
        conn.commit()
    finally:
        conn.close()


def _seed_contradictions(env, *, total: int, unresolved: int) -> None:
    """Seed rag_contradictions con N totales y M unresolved (skipped IS NULL)."""
    assert unresolved <= total
    conn = sqlite3.connect(str(env["telemetry_db"]))
    try:
        now = datetime.now()
        for i in range(total):
            ts = (now - timedelta(minutes=i)).isoformat(timespec="seconds")
            skipped = None if i < unresolved else "1"
            conn.execute(
                "INSERT INTO rag_contradictions "
                "(ts, subject_path, contradicts_json, helper_raw, skipped) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, f"path-{i}.md", "[]", "x", skipped),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_query_paths(env, paths_lists: list[list[str]], days_ago: int = 1) -> None:
    """Seed rag_queries with each entry's `paths_json` set to the given list."""
    conn = sqlite3.connect(str(env["telemetry_db"]))
    try:
        ts = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
        for paths in paths_lists:
            conn.execute(
                "INSERT INTO rag_queries (ts, q, paths_json) VALUES (?, ?, ?)",
                (ts, "test", json.dumps(paths)),
            )
        conn.commit()
    finally:
        conn.close()


# ── 1. Vault vacío ────────────────────────────────────────────────────────


def test_empty_vault_score_zero(vh_env):
    """Vault sin notas indexadas → score=0 + componentes con N=0.

    Sin notas no hay nada que medir; el score "neutro" es 0 (no 100
    porque "no hay vault" no es señal de salud). Esto matchea el comment
    en `_score_*` cuando total<=0.
    """
    from rag.vault_health import compute_vault_health

    result = compute_vault_health(force=True)

    assert result["score"] == 0
    comps = result["components"]
    assert comps["total_notes"] == 0
    assert comps["tags_pct"] == 0
    assert comps["backlinks_pct"] == 0
    assert comps["orphans"] == 0
    assert comps["dupes"] == 0
    assert comps["dead_notes"] == 0
    # contradictions can be > 0 even sin notas, pero acá no seedeamos.
    assert comps["contradictions"] == 0
    assert result["error"] is None


# ── 2. 100 notas todas con tags → tags_pct=100 ────────────────────────────


def test_all_notes_with_tags_pct_100(vh_env):
    """100 notas todas con ≥1 tag → tags_pct == 100."""
    from rag.vault_health import compute_vault_health

    notes = [
        {"file": f"folder/note-{i}.md", "tags": ["t1"], "outlinks": []}
        for i in range(100)
    ]
    _seed_meta(vh_env, notes)

    result = compute_vault_health(force=True)
    assert result["components"]["total_notes"] == 100
    assert result["components"]["tags_pct"] == 100
    assert result["components"]["sub_scores"]["tags_pct"] == pytest.approx(100.0)


# ── 3. Cálculo del weighted score correcto ────────────────────────────────


def test_weighted_score_matches_manual_computation(vh_env):
    """Construye un escenario con valores exactos y valida que el score
    final sea exactamente la suma ponderada de los sub-scores."""
    from rag.vault_health import (
        WEIGHTS,
        compute_vault_health,
        _score_pct,
        _score_orphans,
        _score_contradictions,
        _score_dupes,
        _score_dead,
    )

    # 4 notas: A linkea a B; B linkea a A; C orfan total; D linkea a un
    # título inexistente (out=1, in=0). Tags: A,B tienen tag; C,D no.
    # → tags_pct = 50; backlinks: A tiene B-as-backlink, B tiene A
    # backlink, C tiene 0 in/out, D tiene 0 in. → with_backlinks: 2/4 = 50%.
    # orphans: C (0 in, 0 out) = 1.
    # No contras, no dupes, no dead (sin extra_json modified).
    notes = [
        {"file": "A.md", "note": "A", "tags": ["t1"], "outlinks": ["B"]},
        {"file": "B.md", "note": "B", "tags": ["t1"], "outlinks": ["A"]},
        {"file": "C.md", "note": "C", "tags": [], "outlinks": []},
        {"file": "D.md", "note": "D", "tags": [], "outlinks": ["nonexistent_title"]},
    ]
    _seed_meta(vh_env, notes)

    result = compute_vault_health(force=True)
    comps = result["components"]
    assert comps["total_notes"] == 4
    assert comps["tags_pct"] == 50  # A, B
    assert comps["backlinks_pct"] == 50  # A (linkeado por B), B (linkeado por A)
    assert comps["orphans"] == 1  # C
    assert comps["dupes"] == 0
    assert comps["contradictions"] == 0
    assert comps["dead_notes"] == 0

    # Sub-scores manuales:
    expected_subs = {
        "tags_pct":       _score_pct(50.0),
        "backlinks_pct":  _score_pct(50.0),
        "orphans":        _score_orphans(1, 4),
        "contradictions": _score_contradictions(0),
        "dupes":          _score_dupes(0, 4),
        "dead_notes":     _score_dead(0, 4),
    }
    expected_score = int(round(sum(WEIGHTS[k] * expected_subs[k] for k in WEIGHTS)))
    assert result["score"] == expected_score

    # Y validamos los sub-scores devueltos también (el front los muestra
    # como-color-thresholds, así que deben coincidir):
    for k, v in expected_subs.items():
        assert result["components"]["sub_scores"][k] == pytest.approx(v, abs=0.01)


# ── 4. Endpoint /api/vault/health devuelve JSON parseable ─────────────────


def test_endpoint_returns_parseable_json(vh_env):
    """TestClient → GET /api/vault/health → 200 + dict con keys esperadas."""
    import web.server as _server

    client = TestClient(_server.app)
    resp = client.get("/api/vault/health")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    # Keys del shape contractual del dashboard.
    for key in ("score", "components", "weights", "last_calculated", "ttl_seconds", "error"):
        assert key in body, f"missing key {key!r} in response: {body}"
    # weights debe sumar 1.0.
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-6


# ── 5. Cache TTL: 2 calls dentro de 5min → 1 sola pasada de queries ───────


def test_cache_avoids_repeated_sql_within_ttl(vh_env, monkeypatch):
    """Llamar `compute_vault_health()` dos veces en <5min hace una sola
    lectura de notas. Lo verificamos contando llamadas a `_read_notes_meta`."""
    from rag import vault_health as vh

    notes = [
        {"file": f"n-{i}.md", "tags": ["t"], "outlinks": []}
        for i in range(5)
    ]
    _seed_meta(vh_env, notes)
    vh.invalidate_cache()

    real_read = vh._read_notes_meta
    counter = {"n": 0}

    def counting_read():
        counter["n"] += 1
        return real_read()

    monkeypatch.setattr(vh, "_read_notes_meta", counting_read)

    # 1ª llamada: cache miss → lee.
    vh.compute_vault_health(force=False)
    assert counter["n"] == 1
    # 2ª llamada inmediata: cache hit → NO lee.
    vh.compute_vault_health(force=False)
    assert counter["n"] == 1, "cache miss dentro del TTL: rompió la regla"

    # 3ª llamada con force=True: bypassa cache → cuenta como 2.
    vh.compute_vault_health(force=True)
    assert counter["n"] == 2

    # 4ª llamada con cache "expirada" via monkeypatching del clock
    # interno del módulo (`_now`). Avanzamos 6 minutos.
    base = vh._now()
    monkeypatch.setattr(vh, "_now", lambda: base + 360.0)
    vh.compute_vault_health(force=False)
    assert counter["n"] == 3, "cache no expira tras 5min de TTL"


# ── 6. DB locked / inaccesible → endpoint devuelve {score: null} HTTP 200 ─


def test_endpoint_handles_db_failure_gracefully(vh_env, monkeypatch):
    """Si compute_vault_health() raisea (no debería, pero defensa de
    último recurso), el endpoint atrapa y devuelve HTTP 200 + score=null."""
    import web.server as _server
    from rag import vault_health as vh

    def boom():
        raise sqlite3.OperationalError("database is locked")

    # Forzar el path de "el módulo raiseó" — patch al import-target del
    # endpoint, no al módulo, para que la captura del except del endpoint
    # se ejercite.
    monkeypatch.setattr(vh, "compute_vault_health", boom)

    client = TestClient(_server.app)
    resp = client.get("/api/vault/health")
    # Contractualmente: HTTP 200 SIEMPRE — la UI no debe ver 5xx.
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] is None
    assert body["error"], "expected error string poblado cuando el módulo raisea"
    assert "locked" in body["error"].lower() or "OperationalError" in body["error"]


# ── 7. Bonus: test del path "DB no existe" (módulo retorna sanos defaults) ─


def test_module_handles_missing_db_files(tmp_path, monkeypatch):
    """Sin ragvec.db ni telemetry.db en DB_PATH → score=0 (vault vacío),
    sin raisear. Todos los componentes count=0."""
    import rag
    from rag import vault_health as vh

    # Apuntar DB_PATH a un dir vacío sin DBs.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", empty)
    vh.invalidate_cache()

    result = vh.compute_vault_health(force=True)
    assert result["error"] is None
    assert result["score"] == 0
    assert result["components"]["total_notes"] == 0


# ── 8. Bonus: env var off-switch ──────────────────────────────────────────


def test_disabled_via_env_returns_null_score(vh_env, monkeypatch):
    from rag import vault_health as vh

    monkeypatch.setenv("OBSIDIAN_RAG_VAULT_HEALTH", "0")
    vh.invalidate_cache()

    result = vh.compute_vault_health(force=True)
    assert result["score"] is None
    assert result["error"] == "disabled"


# ── 9. Bonus: contradictions count = sub-score lineal ─────────────────────


def test_unresolved_contradictions_drop_subscore(vh_env):
    """Seed 30 contradicciones, 25 unresolved → sub-score contradictions = 50."""
    from rag.vault_health import compute_vault_health

    _seed_meta(vh_env, [
        {"file": f"n-{i}.md", "tags": ["t"], "outlinks": []}
        for i in range(10)
    ])
    _seed_contradictions(vh_env, total=30, unresolved=25)

    result = compute_vault_health(force=True)
    assert result["components"]["contradictions"] == 25
    # _score_contradictions(25) == max(0, 100 - 100*25/50) = 50
    assert result["components"]["sub_scores"]["contradictions"] == pytest.approx(50.0)


# ── 10. Bonus: dead notes detection ───────────────────────────────────────


def test_dead_notes_detection(vh_env):
    """Notas viejas (>365d) sin queries en 180d cuentan como dead."""
    from rag.vault_health import compute_vault_health

    old_iso = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    fresh_iso = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    notes = [
        {"file": "old-untouched.md", "tags": ["t"], "outlinks": [], "modified": old_iso},
        {"file": "old-recently-queried.md", "tags": ["t"], "outlinks": [], "modified": old_iso},
        {"file": "fresh.md", "tags": ["t"], "outlinks": [], "modified": fresh_iso},
    ]
    _seed_meta(vh_env, notes)
    # "old-recently-queried" aparece en una query reciente, no es dead.
    _seed_query_paths(vh_env, [["old-recently-queried.md"]], days_ago=10)

    result = compute_vault_health(force=True)
    # Solo "old-untouched.md" cumple ambas condiciones.
    assert result["components"]["dead_notes"] == 1
