"""Lectores SQL para el panel /fine_tunning. Misma convención que
web/learning_queries.py: una función pública por sección, cada una
envuelve reads con _sql_read_with_retry, devuelve shape estable.
Tasks 3 y 5 implementan funciones específicas."""

import sqlite3
from datetime import datetime, timedelta


def brief_queue(conn: sqlite3.Connection, *, limit: int = 10) -> list[dict]:
    """
    Cola de briefs candidatos a explicar 'qué salió mal' en el panel /fine_tunning.

    Lee rag_brief_feedback (últimos 30d), agrupa por dedup_key, cuenta ratings.
    Filtra a items con (negative + mute) >= 1 AND positive < 1 — briefs donde
    el user expresó descontento sin compensar con un positivo. Excluye items
    ya rateados en rag_ft_panel_ratings (stream='brief') o snoozeados.

    Returns list[dict] con keys:
      item_id (str, dedup_key = vault_relpath del brief)
      stream ('brief')
      label (str, vault_relpath corto — ej. 'Briefs/2026-04-29-morning')
      rating_counts (dict {positive:N, negative:N, mute:N})
      last_ts (ISO str)
    Cap limit a 50. Silent-fail devuelve [].
    """
    limit = max(1, min(int(limit), 50))
    cutoff_iso = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")

    try:
        cur = conn.execute("""
            SELECT bf.dedup_key, MAX(bf.ts) AS last_ts,
                   SUM(CASE WHEN bf.rating='positive' THEN 1 ELSE 0 END) AS pos,
                   SUM(CASE WHEN bf.rating='negative' THEN 1 ELSE 0 END) AS neg,
                   SUM(CASE WHEN bf.rating='mute' THEN 1 ELSE 0 END) AS mute
            FROM rag_brief_feedback bf
            LEFT JOIN rag_ft_panel_ratings ftr
              ON ftr.stream='brief' AND ftr.item_id = bf.dedup_key
            LEFT JOIN rag_ft_active_queue_state fts
              ON fts.stream='brief' AND fts.item_id = bf.dedup_key
            WHERE bf.ts >= ?
              AND ftr.id IS NULL
              AND (fts.snoozed_until_ts IS NULL OR fts.snoozed_until_ts < datetime('now'))
            GROUP BY bf.dedup_key
            HAVING (neg + mute) >= 1 AND pos < 1
            ORDER BY last_ts DESC
            LIMIT ?
        """, (cutoff_iso, limit))
        rows = cur.fetchall()
    except Exception:
        return []

    result = []
    for dedup_key, last_ts, pos, neg, mute in rows:
        # Derive label from dedup_key: if contains '/', use last 2 segments without .md
        if "/" in dedup_key:
            parts = dedup_key.split("/")
            label = "/".join(parts[-2:])
            if label.endswith(".md"):
                label = label[:-3]
        else:
            label = dedup_key
            if label.endswith(".md"):
                label = label[:-3]

        result.append({
            "item_id": dedup_key,
            "stream": "brief",
            "label": label,
            "rating_counts": {
                "positive": int(pos or 0),
                "negative": int(neg or 0),
                "mute": int(mute or 0),
            },
            "last_ts": last_ts or "",
        })

    return result
