"""Gap signal — detecta query clusters recurrentes en los últimos 14 días
que NO tienen cobertura en el vault.

Señal "me preguntaste lo mismo 3+ veces y no hay nota que lo responda".
Es una proyección anticipatoria del comando CLI `rag emergent` (ese es
interactivo y reportado; esta signal corre en el loop del anticipatory
agent y propone UN gap por pasada, sin imprimir ni pushear).

Heurística:

- Ventana fija de 14 días sobre `rag_queries` (vía `_scan_queries_log`).
- Clustering greedy por cosine de embeddings (vía `_cluster_queries`,
  threshold 0.75 = default histórico de `emergent`).
- Solo clusters con ≥3 miembros pasan el filtro (el umbral de `emergent`
  es ≥5 para push WA visible; acá somos más permisivos porque el
  orchestrator después filtra por score y dedupea por 168h).
- Representante del cluster = query más corta (proxy de forma canónica,
  mismo truco que `emergent`).
- Coverage = `retrieve(col, rep, 3, …)`. Si `scores[0] < 0.30` → gap real:
  no hay nota decente cubriendo el tema. Si `scores[0] >= 0.30` → ya
  existe algo en el vault, no es un gap (capaz un refresh, pero eso es
  otra signal).
- Candidate único por pasada: el cluster más grande con gap. Si hay
  empate por tamaño, el primero en aparición (por cómo sale
  `_cluster_queries`).

Score:
    score = min(1.0, len(cluster) / 10.0)

Una racha de 3 queries → 0.30 (justo en el threshold del agent con la
dedup/snooze de 1 semana). 10+ queries del mismo tema sin cobertura →
score saturado a 1.0.

dedup_key: `f"gap:{sha256(rep)[:12]}"` — estable entre runs mientras el
representante del cluster sea el mismo. Snooze 168h = 1 semana: si la
gente sigue preguntando lo mismo pero ya le avisamos, no vale la pena
volver a avisar todos los días.

Silent-fail: cualquier excepción (SQL, embed, retrieve) → `return []`.
El orchestrator tiene su propio try/except por si acaso, pero esta
signal cumple el contrato y no leaks exceptions.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from rag_anticipate.signals.base import register_signal


@register_signal(name="gap", snooze_hours=168)
def gap_signal(now: datetime) -> list:
    """Ver docstring del módulo."""
    from rag import (
        AnticipatoryCandidate,
        _cluster_queries,
        _scan_queries_log,
        get_db,
        retrieve,
    )

    try:
        events = _scan_queries_log(days=14)
        queries = [
            (ev.get("q_reformulated") or ev.get("q") or "").strip()
            for ev in events
        ]
        # Match el filtro de `emergent`: descartar queries muy cortas que
        # tienden a generar clusters falsos ("ok", "no", "?").
        queries = [q for q in queries if q and len(q) >= 6]
        if not queries:
            return []

        clusters = _cluster_queries(queries, threshold=0.75)
        big = [c for c in clusters if len(c) >= 3]
        if not big:
            return []
        # Más grandes primero; empates conservan orden de aparición.
        big.sort(key=len, reverse=True)

        col = get_db()

        # Iteramos los clusters ordenados por tamaño y devolvemos el
        # PRIMER gap real. Un cluster "cubierto" no descalifica a los
        # más chicos que vengan atrás.
        for cluster in big:
            rep = min((queries[i] for i in cluster), key=len)
            try:
                result = retrieve(
                    col, rep, 3,
                    folder=None, tag=None,
                    precise=False, multi_query=False, auto_filter=True,
                )
            except Exception:
                continue
            scores = result.get("scores") or []
            if scores and scores[0] >= 0.30:
                # Cubierto por alguna nota — no es un gap.
                continue

            n = len(cluster)
            score = min(1.0, n / 10.0)
            dedup_key = f"gap:{hashlib.sha256(rep.encode('utf-8')).hexdigest()[:12]}"
            message = (
                f"🧭 {n} veces preguntaste algo como '{rep}' en 14d sin "
                f"nota que lo cubra. ¿Capturar síntesis?"
            )
            reason = (
                f"cluster_size={n} rep={rep!r} "
                f"top_score={scores[0] if scores else 'n/a'}"
            )
            return [AnticipatoryCandidate(
                kind="anticipate-gap",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=168,
                reason=reason,
            )]

        return []
    except Exception:
        return []
