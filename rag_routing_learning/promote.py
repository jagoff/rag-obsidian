"""UPSERT/desactivación/listado de reglas en ``rag_routing_rules``.

Una regla aprendida tiene la forma:

    (pattern, bucket, evidence_count, evidence_ratio, promoted_at, active)

Promover = insertar/actualizar la fila. El listener relee esta tabla cada
hora y construye un bloque "REGLAS APRENDIDAS DE TU HISTORIAL" en el
sysprompt del classifier.

Reglas se desactivan (active=0) en lugar de borrarse — preserva el
historial de qué se intentó y permite auditar "esta regla la tuvimos
activa pero la apagamos".

Funciones:

- ``upsert_rule`` — inserta o actualiza (pattern, bucket) único.
- ``deactivate_rule`` — flip active=0 por id.
- ``list_active_rules`` — todas las activas, para inyectar al sysprompt.
- ``list_candidate_patterns`` — corre extract_pivot_phrases y filtra los
  que NO están ya promovidos (para review).
- ``render_rules_block`` — formatea el bloque de texto que el listener
  inyecta al sysprompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LearnedRule:
    """Una regla activa o histórica del sysprompt."""
    id: int
    pattern: str
    bucket: str
    evidence_count: int
    evidence_ratio: float
    promoted_at: int
    active: int
    notes: str | None = None


# ── Public API ───────────────────────────────────────────────────────────────


def upsert_rule(
    pattern: str,
    bucket: str,
    evidence_count: int,
    evidence_ratio: float,
    notes: str = "",
) -> int | None:
    """Inserta o actualiza una regla. Devuelve id o None si la DB falla.

    Idempotente por (pattern, bucket): si ya existe, actualiza
    evidence_count + evidence_ratio + promoted_at. NO toca ``active`` —
    una regla desactivada manualmente sigue desactivada hasta que el user
    la reactive explícitamente.
    """
    import rag

    now = int(time.time())
    try:
        with rag._ragvec_state_conn() as conn:
            # Buscar existing por (pattern, bucket)
            existing = conn.execute(
                "SELECT id, active FROM rag_routing_rules "
                "WHERE pattern = ? AND bucket = ?",
                (pattern, bucket),
            ).fetchone()
            if existing:
                rid = existing[0]
                conn.execute(
                    "UPDATE rag_routing_rules SET "
                    "evidence_count = ?, evidence_ratio = ?, promoted_at = ? "
                    "WHERE id = ?",
                    (evidence_count, evidence_ratio, now, rid),
                )
                return rid
            cur = conn.execute(
                "INSERT INTO rag_routing_rules "
                "(pattern, bucket, evidence_count, evidence_ratio, promoted_at, active, notes) "
                "VALUES (?, ?, ?, ?, ?, 1, ?)",
                (pattern, bucket, evidence_count, evidence_ratio, now, notes or None),
            )
            return cur.lastrowid
    except Exception:
        return None


def deactivate_rule(rule_id: int) -> bool:
    """Desactiva una regla por id (sin borrarla). True si efectivamente
    cambió un row, False si no existía o la DB falló."""
    import rag

    try:
        with rag._ragvec_state_conn() as conn:
            cur = conn.execute(
                "UPDATE rag_routing_rules SET active = 0 WHERE id = ? AND active = 1",
                (rule_id,),
            )
            return (cur.rowcount or 0) > 0
    except Exception:
        return False


def reactivate_rule(rule_id: int) -> bool:
    """Inverso de deactivate_rule — flips active=1."""
    import rag

    try:
        with rag._ragvec_state_conn() as conn:
            cur = conn.execute(
                "UPDATE rag_routing_rules SET active = 1 WHERE id = ? AND active = 0",
                (rule_id,),
            )
            return (cur.rowcount or 0) > 0
    except Exception:
        return False


def list_active_rules() -> list[LearnedRule]:
    """Retorna todas las reglas activas, ordenadas por evidence_count DESC.

    Esta es la query que el listener hace cada hora para refrescar el
    sysprompt del classifier. Vacía si la DB no está disponible.
    """
    import rag

    try:
        with rag._ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT id, pattern, bucket, evidence_count, evidence_ratio, "
                "       promoted_at, active, notes "
                "FROM rag_routing_rules WHERE active = 1 "
                "ORDER BY evidence_count DESC, evidence_ratio DESC",
            ).fetchall()
    except Exception:
        return []

    return [
        LearnedRule(
            id=r[0],
            pattern=r[1],
            bucket=r[2],
            evidence_count=r[3],
            evidence_ratio=r[4],
            promoted_at=r[5],
            active=r[6],
            notes=r[7],
        )
        for r in rows
    ]


def list_all_rules() -> list[LearnedRule]:
    """Como list_active_rules pero incluye también las desactivadas.
    Útil para el `rag routing list-rules` con un --all flag."""
    import rag

    try:
        with rag._ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT id, pattern, bucket, evidence_count, evidence_ratio, "
                "       promoted_at, active, notes "
                "FROM rag_routing_rules "
                "ORDER BY active DESC, evidence_count DESC",
            ).fetchall()
    except Exception:
        return []

    return [
        LearnedRule(
            id=r[0],
            pattern=r[1],
            bucket=r[2],
            evidence_count=r[3],
            evidence_ratio=r[4],
            promoted_at=r[5],
            active=r[6],
            notes=r[7],
        )
        for r in rows
    ]


def list_candidate_patterns(
    *,
    min_count: int = 5,
    min_ratio: float = 0.90,
    days: int = 60,
) -> list[dict]:
    """Combina patterns.extract_pivot_phrases() + filtra los ya promovidos.

    Devuelve una lista de dicts con info enriquecida — apta para presentar
    a un user en CLI ("¿promover esta regla? sí/no/ya está"):

        {
          "pattern": str,
          "bucket": str,
          "count": int,
          "ratio": float,
          "bucket_breakdown": dict[str, int],
          "examples": list[str],
          "already_promoted": bool,         # ya hay row en rag_routing_rules
          "active": bool,                   # si already_promoted, está activa?
        }

    Esto es "qué hay nuevo desde la última vez que reviste".
    """
    from rag_routing_learning.patterns import extract_pivot_phrases

    candidates = extract_pivot_phrases(
        min_count=min_count,
        min_ratio=min_ratio,
        days=days,
    )
    # Cargar reglas existentes (incluyendo desactivadas) para el flag
    existing = {(r.pattern, r.bucket): r for r in list_all_rules()}

    out: list[dict] = []
    for c in candidates:
        existing_rule = existing.get((c.pattern, c.bucket))
        out.append({
            "pattern": c.pattern,
            "bucket": c.bucket,
            "count": c.count,
            "ratio": c.ratio,
            "bucket_breakdown": dict(c.bucket_breakdown),
            "examples": list(c.examples),
            "already_promoted": existing_rule is not None,
            "active": bool(existing_rule.active) if existing_rule else False,
        })
    return out


def render_rules_block(rules: list[LearnedRule]) -> str:
    """Formatea las reglas activas como bloque de texto para el sysprompt.

    Output (esperado por el listener):

        REGLAS APRENDIDAS DE TU HISTORIAL (basadas en cómo tradicionalmente
        clasificaste audios similares — usalas como sesgo, no como certeza):

        - "tengo que" → casi siempre va a 'reminder' (95% de 23 casos)
        - "turno con" → casi siempre va a 'calendar_timed' (100% de 11 casos)

    Si no hay reglas, devuelve string vacía — el listener detecta eso y
    skip injection.
    """
    if not rules:
        return ""
    header = (
        "REGLAS APRENDIDAS DE TU HISTORIAL (basadas en cómo tradicionalmente "
        "clasificaste audios similares — usalas como sesgo, no como certeza):"
    )
    lines = [header, ""]
    for r in rules:
        pct = round(r.evidence_ratio * 100)
        lines.append(
            f"- \"{r.pattern}\" → casi siempre va a '{r.bucket}' "
            f"({pct}% de {r.evidence_count} casos)"
        )
    return "\n".join(lines)
