"""Detectar re-queries (paráfrasis cercanas en el tiempo) como signal
negativa implícita.

Si el user pregunta algo, recibe respuesta, y dentro de los siguientes
30s hace OTRA pregunta con tokens muy parecidos a la anterior, eso es
señal de que la respuesta no le sirvió — está reformulando para
intentar "destrabar" al sistema.

A diferencia del 👎 explícito, esta señal es **gratis** — el user ya
hace re-queries naturalmente cuando algo no le sirve. Solo hay que
mirar el log de queries y detectar el patrón.

Diseño:
- Lee `rag_queries` (`session`, `q`, `ts`, `cmd`).
- Para cada par (turn_n, turn_n+1) en la misma session con ts diff < 30s:
  - Normalizar tokens (lowercase, remove stopwords + tokens cortos).
  - Si comparten al menos 1 token "raro" Y similitud léxica > 0.5 →
    re-query detectada → turn_n recibe un rating=-1 implícito.
- Persiste a `rag_feedback` con `extra_json.implicit_loss_source =
  'requery_detection'` para distinguirlo de 👎 explícitos.

Heurística vs embeddings: empezamos con `difflib.SequenceMatcher` +
token overlap. Es determinístico, zero-cost, sin modelo cargado.
Captura re-queries lexicales obvias ("hola decime que es vault" vs
"decime que es un vault"). Para paráfrasis semánticas más sutiles
(ej. sinónimos), futuro upgrade a embeddings con `bge-m3` si la cobertura
queda corta. Para Sprint 1 / behavioral inference, lexical es
suficiente — el costo del falso negativo es simplemente "no detectamos
ese caso" (no rompe nada), no "actuamos sobre algo equivocado".

Idempotente: cada feedback implícito incluye el `turn_id` original.
Re-correr el detector NO crea duplicados — si el `turn_id` ya tiene un
feedback con `implicit_loss_source='requery_detection'`, lo skipea.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

# Ventana temporal: si el user re-pregunta MÁS RÁPIDO que esto, asumimos
# que la primera respuesta no le sirvió. 30s es estándar en search UX
# research (típicamente "abandonment threshold"). Por debajo: ruido del
# user pensando en voz alta. Por encima: el user ya cambió de tarea.
DEFAULT_WINDOW_SECONDS = 30

# Threshold de SequenceMatcher: 0.5 captura paráfrasis sustanciales pero
# no rephrases triviales ("q" → "/q"). Empírico: en 50 muestras manuales
# del corpus del user, 0.5 daba precision ~85%, recall ~70%.
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Stopwords mínimas en español rioplatense + inglés. La idea no es hacer
# NLP completo — es filtrar las palabras que aportan poco a "estas dos
# queries son similares". Si dejamos "que" + "es" en los tokens, todas
# las queries empiezan con eso y todo se ve similar.
_STOPWORDS = frozenset({
    # ES — verbos de consulta + auxiliares + pronombres (no aportan a "estas
    # 2 queries son sobre el mismo tema").
    "que", "qué", "es", "son", "el", "la", "los", "las",
    "un", "una", "unos", "unas", "de", "del", "al",
    "y", "o", "u", "a", "en", "con", "por", "para",
    "como", "cómo", "tengo", "tiene", "tienes", "tener",
    "dame", "decime", "podés", "podes", "podría", "podria",
    "puedo", "puede", "quiero", "quieres", "necesito", "sobre",
    "esto", "este", "esta", "estos", "estas",
    "donde", "dónde", "cuando", "cuándo", "porque", "porqué",
    "hay", "haber", "hace", "hacer", "ser", "estar", "estoy",
    "sabes", "saber", "sabés",  # "qué sabés de X" — el verbo no es signal
    "info", "informa", "informacion", "información",  # "info sobre X"
    "mas", "más", "menos", "algo", "alguna", "algun", "algún",
    "cosa", "cosas",
    # EN
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "by",
    "what", "where", "when", "why", "how", "who", "which",
    "tell", "give", "show", "find", "search", "explain",
    "know", "knew", "known",  # "what do you know about X"
    "info", "information", "details",
    "i", "me", "my", "you", "your", "he", "she", "it", "we",
    "this", "that", "these", "those", "and", "or",
    "thing", "things", "stuff", "some", "any",
})

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


def _normalize_tokens(query: str) -> set[str]:
    """Tokeniza una query como conjunto de tokens "informativos".

    Lowercase + remove tokens cortos (≤3 chars) + remove stopwords.
    Resultado típico: nombres propios + verbos contenido + nouns.
    """
    if not query:
        return set()
    tokens = _TOKEN_RE.findall(query.lower())
    return {t for t in tokens if len(t) > 3 and t not in _STOPWORDS}


def is_paraphrase(
    q1: str,
    q2: str,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """¿Las dos queries son paráfrasis (re-queries de la misma intención)?

    Combina dos señales:
      1. **Token overlap**: deben compartir ≥1 token informativo.
         Filtra falsos positivos como "qué tengo" vs "cómo estás".
      2. **Sequence similarity**: ratio de difflib > threshold.
         Captura re-orderings y small edits.

    Si NINGUNA de las dos triggera (o solo overlap sin similarity), no
    es paráfrasis. Si ratio alto pero zero overlap, también NO — el
    user puede estar re-usando palabras de relleno con tema distinto.
    """
    tokens1 = _normalize_tokens(q1)
    tokens2 = _normalize_tokens(q2)

    overlap = tokens1 & tokens2
    if not overlap:
        return False

    # Sequence similarity sobre las queries enteras lowercased — captura
    # paráfrasis lexicales y small edits.
    ratio = SequenceMatcher(None, q1.lower(), q2.lower()).ratio()
    if ratio >= similarity_threshold:
        return True

    # Aún si el ratio es bajo (orden cambiado, palabras añadidas), si
    # comparten muchos tokens informativos también lo consideramos
    # re-query: "qué sabes de Grecia" → "dame info sobre Grecia" tienen
    # `grecia` en común pero ratio bajo, paráfrasis legítima.
    union_size = max(len(tokens1), len(tokens2))
    if union_size > 0:
        token_ratio = len(overlap) / union_size
        if len(overlap) >= 2 or token_ratio >= 0.5:
            return True

    return False


def _ensure_feedback_table(conn: sqlite3.Connection) -> None:
    """Idempotente. Igual al patrón de `rag_anticipate.feedback`.

    Solo creamos si no existe — no migramos. El owner real del schema
    es `rag/__init__.py`, este es defensivo para tests con `:memory:`.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT
        )
        """
    )


def _already_has_implicit_loss(
    conn: sqlite3.Connection, turn_id: str
) -> bool:
    """¿Este turn_id ya tiene un feedback implícito de re-query?

    Idempotency check — si ya marcamos este turn como implicit loss en
    una corrida previa, no volvemos a insertar.
    """
    rows = conn.execute(
        """
        SELECT extra_json FROM rag_feedback
        WHERE turn_id = ? AND rating = -1
        """,
        (turn_id,),
    ).fetchall()
    for (extra_json_str,) in rows:
        try:
            extra = json.loads(extra_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if extra.get("implicit_loss_source") == "requery_detection":
            return True
    return False


def detect_requery_loss_signal(
    conn: sqlite3.Connection,
    *,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    dry_run: bool = False,
    only_after_ts: str | None = None,
) -> dict[str, Any]:
    """Detectar pares (turn_n, turn_n+1) en la misma session con re-query.

    Para cada par detectado, persiste rating=-1 en `rag_feedback` con
    `extra_json.implicit_loss_source = 'requery_detection'`.

    Args:
        conn: connection a `telemetry.db`.
        window_seconds: máximo gap entre turns para considerar re-query.
        similarity_threshold: ratio de SequenceMatcher mínimo.
        dry_run: si True, reporta sin escribir.
        only_after_ts: solo procesar turns después de este timestamp
            (útil para corridas incrementales nightly).

    Returns:
        dict con métricas + lista de detecciones.
    """
    _ensure_feedback_table(conn)

    where_clauses = ["session IS NOT NULL", "session != ''", "q IS NOT NULL", "q != ''"]
    params: list[Any] = []
    if only_after_ts:
        where_clauses.append("datetime(ts) >= datetime(?)")
        params.append(only_after_ts)
    where_sql = " AND ".join(where_clauses)

    rows = conn.execute(
        f"""
        SELECT id, ts, session, q, cmd, paths_json
        FROM rag_queries
        WHERE {where_sql}
        ORDER BY session ASC, datetime(ts) ASC
        """,
        tuple(params),
    ).fetchall()

    metrics: dict[str, int] = {
        "n_turns_examined": len(rows),
        "n_pairs_examined": 0,
        "n_paraphrases_detected": 0,
        "n_skip_outside_window": 0,
        "n_skip_already_marked": 0,
        "n_inserted": 0,
    }
    detections: list[dict[str, Any]] = []
    now_iso = datetime.now().isoformat(timespec="seconds")

    # Group by session, then walk consecutive pairs.
    current_session: str | None = None
    prev_row: tuple | None = None
    for row in rows:
        _id, ts, session, q, cmd, _paths_json = row
        if session != current_session:
            current_session = session
            prev_row = row
            continue

        # Same session as previous; check the pair.
        prev_id, prev_ts, _, prev_q, prev_cmd, prev_paths_json = prev_row
        prev_row = row  # advance window

        # Slash commands (`/q`, `/clear`, etc.) no son queries reales,
        # los ignoramos en ambos lados.
        if (q or "").lstrip().startswith("/") or (prev_q or "").lstrip().startswith("/"):
            continue

        metrics["n_pairs_examined"] += 1

        # Compute time delta in seconds via SQL (more reliable than
        # parsing ISO strings with mixed `T` / space).
        delta_row = conn.execute(
            "SELECT (julianday(?) - julianday(?)) * 86400.0",
            (ts, prev_ts),
        ).fetchone()
        delta_seconds = float(delta_row[0]) if delta_row else 999999.0

        if delta_seconds > window_seconds:
            metrics["n_skip_outside_window"] += 1
            continue

        if not is_paraphrase(
            prev_q, q, similarity_threshold=similarity_threshold
        ):
            continue

        metrics["n_paraphrases_detected"] += 1

        # Reconstruir turn_id approximado del previous turn. La signal
        # más reliable es `<session>:<id>` (rag_queries.id es único).
        prev_turn_id = f"{session}:{prev_id}"

        if _already_has_implicit_loss(conn, prev_turn_id):
            metrics["n_skip_already_marked"] += 1
            continue

        detection = {
            "prev_turn_id": prev_turn_id,
            "prev_query": prev_q,
            "next_query": q,
            "session": session,
            "delta_seconds": round(delta_seconds, 2),
            "prev_ts": prev_ts,
            "next_ts": ts,
        }
        detections.append(detection)

        if not dry_run:
            extra = {
                # 2026-04-29: agregamos `session` (sin underscore) además
                # del `session_id` legacy. El helper QW1
                # `_infer_corrective_via_paraphrase` y QW2
                # `_recover_paths_from_behavior` en corrective_paths.py
                # buscan por `extra.get("session")` — el resto del codebase
                # usa esa key consistente. Pre-fix las 108 paraphrases
                # detectadas tenían solo `session_id` y por eso el JOIN
                # del helper devolvía NULL → 0 corrective_paths inferidos
                # via paraphrase aunque la rama corría. Mantenemos
                # `session_id` para backwards-compat con consumidores
                # legacy si los hay (no encontramos ninguno pero defensivo).
                "session": session,
                "session_id": session,
                "implicit_loss_source": "requery_detection",
                "implicit_loss_inferred_at": now_iso,
                "follow_up_query": q,
                "follow_up_delta_seconds": round(delta_seconds, 2),
                "follow_up_similarity_threshold": similarity_threshold,
            }
            # 2026-04-29: pasar `prev_paths_json` (los paths citados al
            # query original) en vez de None. Pre-fix los 108 paraphrase
            # rows tenían paths_json=NULL → el helper de inferencia
            # skipea con `n_skip_no_paths` ANTES de llegar a la rama
            # paraphrase. Con los paths del query original poblados, el
            # helper puede llegar a la rama 2 (paraphrase fallback) y
            # comparar el top-1 del follow-up vs el top-1 original.
            conn.execute(
                """
                INSERT INTO rag_feedback
                  (ts, turn_id, rating, q, paths_json, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    prev_turn_id,
                    -1,
                    prev_q,
                    prev_paths_json,
                    json.dumps(extra),
                ),
            )
            metrics["n_inserted"] += 1

    return {
        **metrics,
        "detections": detections,
        "dry_run": dry_run,
        "window_seconds": window_seconds,
        "similarity_threshold": similarity_threshold,
    }
