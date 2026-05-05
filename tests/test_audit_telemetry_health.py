"""Tests para los health checks del audit_telemetry_health script.

Audit 2026-04-25 R2-Telemetry #5: agregamos retrieval + chat health
checks para detectar degradation silenciosa que `check_anticipate_health`
no cubre (cache invalidation suelta, ranker degradado, latency creep en
LLM, refusal rate alto).

Filosofía de los tests: armamos una telemetry.db mínima en tmp_path con
un schema acotado a las columnas que los health checks consultan.
NO importamos `rag` porque el script `audit_telemetry_health` está
diseñado para correr standalone contra el archivo SQLite — solo recibe
una `sqlite3.Connection` viva y devuelve el dict. Eso baja el setup de
los tests a ~10 líneas de DDL + INSERTs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from scripts.audit_telemetry_health import (
    _audit_feedback_corrective_gap,
    check_chat_health,
    check_retrieval_health,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _seed_telemetry_db(tmp_path):
    """Helper que crea una telemetry.db mínima con las tablas que los
    health checks necesitan.

    Usamos un subset del schema real (`rag/__init__.py:_ensure_telemetry_tables`)
    — solo las columnas que los health checks miran. Si rag.py agrega
    columnas nuevas que estos checks no consumen, este schema NO necesita
    actualizarse — el SELECT solo pide las columnas conocidas.
    """
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            q TEXT NOT NULL,
            top_score REAL,
            t_retrieve REAL,
            t_gen REAL,
            critique_fired INTEGER,
            extra_json TEXT
        )
        """
    )
    conn.commit()
    return conn, db_path


def _insert_query(
    conn: sqlite3.Connection,
    *,
    cmd: str,
    ts: str | None = None,
    top_score: float | None = None,
    t_retrieve: float | None = None,
    t_gen: float | None = None,
    critique_fired: int | None = None,
    extra_json: str | None = None,
) -> None:
    """Insert helper para no repetir el INSERT en cada test.

    `ts` default = ahora, así por defecto las queries caen dentro de la
    ventana de 7 días que usan los checks. Pasá un ts ISO específico
    (ej. 30 días atrás) para testear el lado "fuera de ventana".
    """
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO rag_queries
          (ts, cmd, q, top_score, t_retrieve, t_gen, critique_fired, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, cmd, "test query", top_score, t_retrieve, t_gen, critique_fired, extra_json),
    )
    conn.commit()


# ── Tests retrieval ──────────────────────────────────────────────────────────


def test_retrieval_health_stale_when_no_queries(tmp_path):
    """0 queries en la ventana → status=stale.

    Es el caso "system idle o corrupto": no hay nada que medir, así que
    devolvemos `stale` para que el operador investigue por qué nadie
    está usando el RAG (o por qué el writer dejó de loguear).
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    result = check_retrieval_health(conn, days=7)
    assert result["status"] == "stale"
    assert result["details"]["queries_count"] == 0
    assert any("0 retrieval queries" in i for i in result["issues"])


def test_retrieval_health_degraded_when_top_score_low(tmp_path):
    """Median top_score < 0.4 → degraded (ranker devolviendo basura).

    Sembramos 20 queries con top_score=0.2 para que la mediana caiga
    bajo el threshold. Las latencias y cache las dejamos NULL para
    que solo dispare la regla del ranker.
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    for _ in range(20):
        _insert_query(conn, cmd="query", top_score=0.2)
    result = check_retrieval_health(conn, days=7)
    assert result["status"] == "degraded"
    assert result["details"]["median_top_score"] == 0.2
    assert any("median top_score" in i for i in result["issues"])


def test_retrieval_health_healthy_when_metrics_normal(tmp_path):
    """top_score 0.7, retrieve 800ms, cache hit 65% → healthy.

    Metricas todas dentro de baseline → no debe disparar ninguna regla.
    Sembramos cache mix (13 hits + 7 miss = 65% hit rate sobre 20
    queries elegibles) + 5 skipped (history) que NO entran al rate
    pero que el script reporta separado.

    2026-05-01: schema del cache_probe cambió de `extra_json.cache_hit`
    (bool) a `extra_json.cache_probe.result` (str: 'hit'/'miss'/'skipped')
    para soportar el contador de queries skipped. El audit ahora
    excluye las skipped del denominador (por design — esas queries
    nunca fueron candidatas al cache).
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    # 20 web queries elegibles (13 hits + 7 miss = 65% hit rate)
    for i in range(20):
        result_str = "hit" if i < 13 else "miss"
        _insert_query(
            conn,
            cmd="web",
            top_score=0.7,
            t_retrieve=0.8,
            extra_json=(
                '{"cache_probe": {"result": "' + result_str + '"}}'
            ),
        )
    # 5 web queries `skipped` (NO suman al hit rate — el caller traía
    # history o multi-vault, fuera del cache layer por design).
    for _ in range(5):
        _insert_query(
            conn,
            cmd="web",
            top_score=0.7,
            t_retrieve=0.8,
            extra_json='{"cache_probe": {"result": "skipped"}}',
        )
    # Y 5 queries más (cmd='query') para darle señal al p95 / median
    for _ in range(5):
        _insert_query(conn, cmd="query", top_score=0.7, t_retrieve=0.8)
    result = check_retrieval_health(conn, days=7)
    assert result["status"] == "healthy", f"issues: {result['issues']}"
    assert result["details"]["queries_count"] == 30
    assert result["details"]["cache_hit_rate_pct"] == 65.0
    assert result["details"]["cache_eligible_count"] == 20
    assert result["details"]["cache_skipped_count"] == 5


def test_retrieval_health_degraded_when_p95_high(tmp_path):
    """p95 t_retrieve > 1950ms → degraded (latency creep).

    Sembramos n=20 con los últimos 2 lentos (3s) → la nearest-rank p95
    para n=20 cae en el índice 18 (== 19º elemento ordenado), que es la
    primera de las dos lentas. Top_score lo ponemos sano (0.7) para que
    solo dispare la regla de p95.

    Nota sobre nearest-rank: con n=20, p=0.95, idx=18 (el 19º elemento
    ordenado). Si solo hubiera 1 query lenta, caería en idx 18 → fast,
    y el test no detectaría el outlier — por eso usamos 2 lentas.
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    for _ in range(18):
        _insert_query(conn, cmd="query", top_score=0.7, t_retrieve=0.5)
    for _ in range(2):
        _insert_query(conn, cmd="query", top_score=0.7, t_retrieve=3.0)
    result = check_retrieval_health(conn, days=7)
    assert result["status"] == "degraded"
    assert result["details"]["p95_retrieve_ms"] == 3000.0
    assert any("p95 t_retrieve" in i for i in result["issues"])


# ── Tests chat ───────────────────────────────────────────────────────────────


def test_chat_health_stale_when_no_chats(tmp_path):
    """0 chats en la ventana → stale (endpoint idle o caído)."""
    conn, _ = _seed_telemetry_db(tmp_path)
    result = check_chat_health(conn, days=7)
    assert result["status"] == "stale"
    assert result["details"]["chats_count"] == 0
    assert any("0 chats" in i for i in result["issues"])


def test_chat_health_degraded_when_p95_high(tmp_path):
    """p95 t_gen > 3900ms (3000 baseline × 1.3) → degraded.

    Mismo razonamiento que el test de p95 retrieve: con n=20 nearest-rank
    p95 cae en idx=18 (19º elemento ordenado), así que necesitamos 2
    chats lentos para que el p95 caiga sobre uno.
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    for _ in range(18):
        _insert_query(conn, cmd="chat", t_gen=2.0, critique_fired=0)
    for _ in range(2):
        _insert_query(conn, cmd="chat", t_gen=5.0, critique_fired=0)
    result = check_chat_health(conn, days=7)
    assert result["status"] == "degraded"
    assert result["details"]["p95_gen_ms"] == 5000.0
    assert any("p95 t_gen" in i for i in result["issues"])


def test_chat_health_degraded_when_refusal_high(tmp_path):
    """refusal_rate > 50% → degraded (bot se rinde demasiado).

    15 web.chat.degenerate (refusals) sobre 25 chats totales = 60%.
    """
    conn, _ = _seed_telemetry_db(tmp_path)
    for _ in range(15):
        _insert_query(conn, cmd="web.chat.degenerate")
    for _ in range(10):
        _insert_query(conn, cmd="web", t_gen=2.0)
    result = check_chat_health(conn, days=7)
    assert result["status"] == "degraded"
    assert result["details"]["refusal_rate_pct"] == 60.0
    assert any("refusal rate" in i for i in result["issues"])


def test_chat_health_healthy_when_metrics_normal(tmp_path):
    """p95 2s, refusal ~5%, critique fires 10% → healthy."""
    conn, _ = _seed_telemetry_db(tmp_path)
    # 19 chats sanos + 1 refusal sobre 20 = 5% refusal rate
    for i in range(19):
        # 2 de los 19 con critique_fired=1 → ~10% critique rate, sano
        crit = 1 if i < 2 else 0
        _insert_query(conn, cmd="web", t_gen=2.0, critique_fired=crit)
    _insert_query(conn, cmd="web.chat.degenerate")
    result = check_chat_health(conn, days=7)
    assert result["status"] == "healthy", f"issues: {result['issues']}"
    assert result["details"]["chats_count"] == 20
    assert result["details"]["refusal_rate_pct"] == 5.0


# ── Smoke / contract test ────────────────────────────────────────────────────


def test_health_dicts_are_json_serializable(tmp_path):
    """Los dicts que devuelven los checks tienen que serializar a JSON
    sin custom default — eso es lo que el flag `--json` del script
    espera. Si alguien mete un Counter o un datetime, este test rompe.
    """
    import json

    conn, _ = _seed_telemetry_db(tmp_path)
    _insert_query(conn, cmd="query", top_score=0.7, t_retrieve=0.8)
    _insert_query(conn, cmd="web", t_gen=2.0, critique_fired=0)

    ret = check_retrieval_health(conn, days=7)
    chat = check_chat_health(conn, days=7)

    # default str es el mismo guard que usa main() — si esto rompe,
    # el script con --json va a romper igual en producción.
    json.dumps(ret)
    json.dumps(chat)


# ── Tests feedback_corrective_gap ────────────────────────────────────────────


def _seed_feedback_db(tmp_path) -> sqlite3.Connection:
    """Crea una telemetry.db mínima con solo rag_feedback (las columnas
    que `_audit_feedback_corrective_gap` necesita).
    """
    db_path = tmp_path / "telemetry_fb.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rating INTEGER NOT NULL,
            extra_json TEXT
        )
        """
    )
    conn.commit()
    return conn


def _insert_feedback(
    conn: sqlite3.Connection,
    *,
    rating: int,
    corrective_path: str | None = None,
) -> None:
    extra = None
    if corrective_path is not None:
        import json as _json
        extra = _json.dumps({"corrective_path": corrective_path})
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, extra_json) VALUES (?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), rating, extra),
    )
    conn.commit()


def test_feedback_cp_gap_empty_table(tmp_path):
    """Sin ningún feedback → total_neg=0, gate_open=False, rows_to_close=20."""
    conn = _seed_feedback_db(tmp_path)
    result = _audit_feedback_corrective_gap(conn)
    assert result["total_neg"] == 0
    assert result["has_cp"] == 0
    assert result["missing_cp"] == 0
    assert result["pct_covered"] == 0.0
    assert result["gate_open"] is False
    assert result["rows_to_close_gate"] == 20


def test_feedback_cp_gap_no_cp(tmp_path):
    """562 negativos sin corrective_path → has_cp=0, gate cerrado."""
    conn = _seed_feedback_db(tmp_path)
    for _ in range(562):
        _insert_feedback(conn, rating=-1)
    result = _audit_feedback_corrective_gap(conn)
    assert result["total_neg"] == 562
    assert result["has_cp"] == 0
    assert result["missing_cp"] == 562
    assert result["pct_covered"] == 0.0
    assert result["gate_open"] is False
    assert result["rows_to_close_gate"] == 20


def test_feedback_cp_gap_partial_cp(tmp_path):
    """2 negativos con CP sobre 562 → has_cp=2, gate cerrado, faltan 18."""
    conn = _seed_feedback_db(tmp_path)
    for _ in range(560):
        _insert_feedback(conn, rating=-1)
    _insert_feedback(conn, rating=-1, corrective_path="notas/foo.md")
    _insert_feedback(conn, rating=-1, corrective_path="notas/bar.md")
    result = _audit_feedback_corrective_gap(conn)
    assert result["total_neg"] == 562
    assert result["has_cp"] == 2
    assert result["missing_cp"] == 560
    assert result["pct_covered"] == round(2 / 562 * 100, 2)
    assert result["gate_open"] is False
    assert result["rows_to_close_gate"] == 18


def test_feedback_cp_gap_gate_open(tmp_path):
    """20 negativos con CP → gate_open=True, rows_to_close=0."""
    conn = _seed_feedback_db(tmp_path)
    for i in range(20):
        _insert_feedback(conn, rating=-1, corrective_path=f"notas/nota_{i}.md")
    # Positivos NO deben contar
    for _ in range(50):
        _insert_feedback(conn, rating=1)
    result = _audit_feedback_corrective_gap(conn)
    assert result["total_neg"] == 20
    assert result["has_cp"] == 20
    assert result["gate_open"] is True
    assert result["rows_to_close_gate"] == 0
    assert result["pct_covered"] == 100.0


def test_feedback_cp_gap_empty_string_cp_does_not_count(tmp_path):
    """CP con string vacío ("") no cuenta como valid — debe tratarse como missing."""
    import json as _json
    conn = _seed_feedback_db(tmp_path)
    extra = _json.dumps({"corrective_path": ""})
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, extra_json) VALUES (?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), -1, extra),
    )
    conn.commit()
    result = _audit_feedback_corrective_gap(conn)
    assert result["has_cp"] == 0
    assert result["missing_cp"] == 1


def test_feedback_cp_gap_table_missing(tmp_path):
    """Si la tabla no existe, devuelve un dict con error (sin raise)."""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    result = _audit_feedback_corrective_gap(conn)
    assert "error" in result
    assert "rag_feedback" in result["error"]


def test_feedback_cp_gap_json_serializable(tmp_path):
    """El dict de output debe serializar a JSON sin custom default."""
    import json
    conn = _seed_feedback_db(tmp_path)
    _insert_feedback(conn, rating=-1, corrective_path="notas/a.md")
    result = _audit_feedback_corrective_gap(conn)
    json.dumps(result)  # no debe raise
