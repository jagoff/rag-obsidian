"""Compose proactive WhatsApp drafts from anticipatory signals.

Una signal (ej. commitment_deadline) que tiene target identificable produce
un draft listo para enviar, mimetizando el style fingerprint del user
([`rag/style.py`](rag/style.py)) — features agregados (voseo dominance,
openers/closers, emoji rate, slang argentino), nunca raw text.

Contract: `compose_draft(...)` siempre retorna `dict | None`. Silent-fail
por design — el orchestrator de `anticipate_run_impl` trata `None` como
"no draft, push contextual legacy".

Kill switch: `RAG_PROACTIVE_DRAFTS_DISABLE=1` → todos los `compose_draft`
retornan `None`.

Allowlist (`RAG_PROACTIVE_DRAFTS_MIN_MSGS=10`, ventana 90d sobre bridge):
no compose para jids con menos de N mensajes históricos. Anti-misfire en
contactos nuevos / typos de jid.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Cap chars draft. 280 = 1 mensaje WA típico sin segmentación visual molesta.
_DRAFT_MAX_CHARS_DEFAULT = 280
_ALLOWLIST_WINDOW_DAYS = 90


def _disabled() -> bool:
    return os.environ.get("RAG_PROACTIVE_DRAFTS_DISABLE", "").lower() in {"1", "true", "yes"}


def _allowlist_min_msgs() -> int:
    try:
        return max(0, int(os.environ.get("RAG_PROACTIVE_DRAFTS_MIN_MSGS", "10")))
    except ValueError:
        return 10


def _bridge_db_path() -> Path | None:
    try:
        from rag.integrations.whatsapp import WHATSAPP_BRIDGE_DB_PATH
    except Exception:
        return None
    p = Path(WHATSAPP_BRIDGE_DB_PATH)
    return p if p.exists() else None


def _allowlist_check(jid: str, *, min_msgs: int | None = None) -> bool:
    """True si hay ≥min_msgs históricos con `jid` en los últimos 90d."""
    if not jid:
        return False
    threshold = _allowlist_min_msgs() if min_msgs is None else min_msgs
    if threshold <= 0:
        return True
    db = _bridge_db_path()
    if db is None:
        # Sin bridge no podemos validar — default deny conservador.
        return False
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return False
    try:
        cutoff_iso = (datetime.now() - timedelta(days=_ALLOWLIST_WINDOW_DAYS)).isoformat()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_jid = ? AND timestamp >= ?",
                (jid, cutoff_iso),
            ).fetchone()
        except sqlite3.Error:
            return False
    finally:
        conn.close()
    return (row[0] if row else 0) >= threshold


def _features_summary(features: dict[str, Any]) -> str:
    """Compact projection del style fingerprint para el prompt del helper."""
    if not features or features.get("insufficient_data"):
        return "(sin fingerprint disponible — voseo argentino estándar, tono casual)"
    voseo_dom = features.get("voseo_dominance", 1.0)
    avg_chars = int(features.get("avg_chars_per_msg") or 80)
    emoji_rate = float(features.get("emoji_rate") or 0.0)
    openers = (features.get("top_openers") or [])[:5]
    closers = (features.get("top_closers") or [])[:5]
    slang = features.get("slang_argentino_hits") or {}
    if isinstance(slang, dict):
        top_slang = [s for s, _ in sorted(slang.items(), key=lambda kv: -kv[1])[:5]]
    else:
        top_slang = []
    if emoji_rate > 0.15:
        emoji_hint = "usá emojis libremente"
    elif emoji_rate < 0.05:
        emoji_hint = "usá pocos o ninguno"
    else:
        emoji_hint = "emojis ocasionales"
    parts = [
        f"- Voseo dominance: {voseo_dom:.2f} (1.0=full voseo, 0=tuteo)",
        f"- Avg chars/msg: {avg_chars} (mantente cerca, max {int(avg_chars * 1.5)})",
        f"- Emoji rate: {emoji_rate:.2f} ({emoji_hint})",
    ]
    if openers:
        parts.append(f"- Openers favoritos: {', '.join(openers)}")
    if closers:
        parts.append(f"- Closers favoritos: {', '.join(closers)}")
    if top_slang:
        parts.append(f"- Slang argentino activo: {', '.join(top_slang)}")
    return "\n".join(parts)


def _when_phrase(days_until_due: float) -> str:
    if days_until_due < 0:
        return "ya está vencido"
    if days_until_due < 1:
        return "vence hoy"
    if days_until_due < 2:
        return "vence mañana"
    return f"vence en {int(days_until_due)} días"


def _build_prompt(
    *,
    target_name: str,
    promise_text: str,
    days_until_due: float,
    style_summary: str,
    max_chars: int,
) -> str:
    when = _when_phrase(days_until_due)
    return (
        f"Sos un asistente que redacta UN mensaje de WhatsApp en nombre del user, "
        f"para mandar a {target_name}.\n\n"
        f'PROMESA PENDIENTE: "{promise_text}"\n'
        f"TIMING: {when}.\n\n"
        f"ESTILO DEL USER (mimetizar EXACTO):\n{style_summary}\n\n"
        f"REGLAS:\n"
        f"- 1 solo mensaje, máximo {max_chars} chars.\n"
        f"- Voseo argentino estricto (vos / podés / mirá / fijate / sos). "
        f"NO tuteo (tú / puedes / mira), NO portugués (você / obrigad / essa).\n"
        f"- Tono natural y directo, no formal.\n"
        f"- Mencionar lo prometido sin sonar acusatorio (vos prometiste, no la otra persona).\n"
        f"- NO uses placeholders [foto] [link] [archivo] — escribí como si vas a mandar de verdad.\n"
        f"- NO firmes ni saludes con tu nombre al final.\n\n"
        f"OUTPUT: SOLO un objeto JSON válido, sin markdown ni texto extra:\n"
        f'{{"draft": "<el mensaje>", "confidence": 0.<N>, "reason": "<por qué este draft>"}}'
    )


# Detector portugués leak — bug recurrente del helper bajo MLX (ver CLAUDE.md).
_PORTUGUESE_MARKERS = (
    " você", " obrigad", " essa ", " isso ", " tua ", " falam ", " primeira ",
    " braços", " vistes", " você ", "obrigado", "obrigada",
)


def _portuguese_leak(text: str) -> bool:
    low = " " + text.lower() + " "
    return any(marker in low for marker in _PORTUGUESE_MARKERS)


def _extract_response_text(resp: Any) -> str:
    """Defensive extraction — backend retorna dict o pydantic Message."""
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        if isinstance(msg, dict):
            return str(msg.get("content") or "")
        return str(getattr(msg, "content", "") or "")
    msg = getattr(resp, "message", None)
    if msg is None:
        return ""
    return str(getattr(msg, "content", "") or "")


def _parse_draft_response(raw: str, *, max_chars: int) -> dict[str, Any] | None:
    """Parsea JSON output. Tolerante a wrappers ```json ... ``` y prosa pre/post."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    draft = str(obj.get("draft") or "").strip()
    if not draft:
        return None
    if len(draft) > max_chars:
        draft = draft[: max_chars - 1].rstrip() + "…"
    if _portuguese_leak(draft):
        return None
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = str(obj.get("reason") or "").strip()[:200]
    return {"draft": draft, "confidence": conf, "reason": reason}


def compose_draft(
    *,
    target_jid: str,
    target_name: str,
    promise_text: str,
    days_until_due: float,
    max_chars: int = _DRAFT_MAX_CHARS_DEFAULT,
) -> dict[str, Any] | None:
    """Compose un WA draft para una signal proactive con target identificable.

    Returns ``{"draft", "confidence", "reason", "style_snapshot_hash"}`` en
    success, ``None`` en cualquier fallo:

    - Kill switch (`RAG_PROACTIVE_DRAFTS_DISABLE=1`).
    - Allowlist deny (target con <10 msgs históricos en 90d).
    - LLM call failure (timeout, 503, etc).
    - JSON parse failure / portugués leak / draft vacío.
    """
    if _disabled():
        return None
    if not target_jid or not promise_text or not promise_text.strip():
        return None
    if not _allowlist_check(target_jid):
        return None

    # Imports lazy: módulo se carga early en orchestrator + evita loops.
    try:
        from rag import HELPER_MODEL, HELPER_OPTIONS, _helper_client, _silent_log
    except Exception:
        return None
    try:
        from rag.style import load_latest as _style_load_latest
    except Exception:
        _style_load_latest = lambda: None  # noqa: E731

    snapshot = _style_load_latest()
    features = (snapshot or {}).get("features", {}) or {}
    snapshot_hash = (snapshot or {}).get("content_hash") or "no-snapshot"
    style_summary = _features_summary(features)

    prompt = _build_prompt(
        target_name=(target_name or "esa persona").strip(),
        promise_text=promise_text.strip()[:300],
        days_until_due=days_until_due,
        style_summary=style_summary,
        max_chars=max_chars,
    )

    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_ctx": 2048, "num_predict": 320},
        )
    except Exception as exc:
        try:
            _silent_log("proactive_draft_llm", exc)
        except Exception:
            pass
        return None

    raw = _extract_response_text(resp)
    parsed = _parse_draft_response(raw, max_chars=max_chars)
    if parsed is None:
        return None
    parsed["style_snapshot_hash"] = snapshot_hash
    return parsed


# ── Push to listener (Fase 3) ────────────────────────────────────────────────

# El listener TS expone Bun.serve en este puerto (loopback only). Override via
# env para tests. Mismo threat model que el web del rag — local-first, sin auth.
_LISTENER_PUSH_URL_DEFAULT = "http://127.0.0.1:8766/push-pending-draft"
_LISTENER_PUSH_TIMEOUT_S = 3.0


def _listener_push_url() -> str:
    return os.environ.get("RAG_LISTENER_PUSH_URL", _LISTENER_PUSH_URL_DEFAULT)


def _new_draft_id() -> str:
    """8-char hex random — colisión cosmética; mismo formato que listener bot_drafts."""
    import secrets
    return secrets.token_hex(4)


def push_draft_to_listener(
    *,
    draft_id: str,
    target_jid: str,
    target_name: str,
    draft_text: str,
    signal_kind: str,
) -> bool:
    """POST al listener TS para insertar draft en `pendingDraftsByJid`.

    Returns True si HTTP 2xx. Silent-fail si el listener está caído o
    devuelve error — caller (orchestrator) marca status='skipped' en
    rag_proactive_drafts y cae al push contextual legacy.
    """
    if not target_jid or not draft_text:
        return False
    payload = {
        "draft_id": draft_id,
        "contact_jid": target_jid,
        "contact_name": target_name or "",
        "draft_text": draft_text,
        "source": f"proactive:{signal_kind}",
    }
    try:
        from urllib import request as _urlreq
        from urllib.error import HTTPError, URLError

        data = json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(
            _listener_push_url(),
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with _urlreq.urlopen(req, timeout=_LISTENER_PUSH_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except (HTTPError, URLError, OSError, json.JSONDecodeError):
        try:
            from rag import _silent_log
            _silent_log("proactive_draft_push", Exception("listener push failed"))
        except Exception:
            pass
        return False


def log_proactive_draft(
    *,
    draft_id: str,
    signal_kind: str,
    signal_dedup_key: str,
    target_jid: str,
    target_name: str | None,
    draft_text: str,
    draft_meta: dict | None,
    status: str,
) -> None:
    """Persist row en rag_proactive_drafts. Silent-fail."""
    if status not in ("pushed", "skipped"):
        return
    try:
        from rag import _ragvec_state_conn, _silent_log, _sql_write_with_retry
    except Exception:
        return
    ts = datetime.now().isoformat(timespec="seconds")
    meta_json = json.dumps(draft_meta or {}, ensure_ascii=False, sort_keys=True)
    def _write() -> None:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_proactive_drafts ("
                "id, ts, signal_kind, signal_dedup_key, target_jid, target_name,"
                " draft_text, draft_meta_json, status"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    draft_id, ts, signal_kind, signal_dedup_key, target_jid,
                    target_name, draft_text, meta_json, status,
                ),
            )
            conn.commit()

    try:
        _sql_write_with_retry(_write, "proactive_draft_log_sql")
    except Exception as exc:
        try:
            _silent_log("proactive_draft_log", exc)
        except Exception:
            pass


# ── Telemetry / analytics helper ─────────────────────────────────────────────


def stats(*, days: int = 14) -> dict[str, Any]:
    """Devuelve métricas del proactive draft loop sobre los últimos N días.

    Cruza `rag_proactive_drafts` (lo que el rag PUSH-eó) con
    `rag_draft_decisions` (cómo el user decidió) via `signal_dedup_key` →
    `extra_json.signal_dedup_key` (set por el listener TS al postear la
    decisión, Fase 4). Un draft puede tener:

    - status=pushed + decision=approved_si      → /si
    - status=pushed + decision=approved_editar  → /editar (con sent_text != draft_text)
    - status=pushed + decision=rejected         → /no
    - status=pushed + decision=expired          → TTL 30min sin acción
    - status=pushed + sin decision              → todavía pendiente
    - status=skipped                            → push http falló (listener down)

    Target Fase 1 (commitment_deadline): /si + /editar ≥ 60% sobre los
    pushed que recibieron alguna decision.
    """
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
    out: dict[str, Any] = {
        "days": days,
        "cutoff": cutoff_iso,
        "by_status": {},
        "by_decision": {},
        "total_pushed": 0,
        "total_skipped": 0,
        "pending": 0,
        "useful_rate": 0.0,  # (/si + /editar) / decided
    }
    try:
        from rag import _ragvec_state_conn
    except Exception:
        return out
    try:
        with _ragvec_state_conn() as conn:
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM rag_proactive_drafts"
                " WHERE ts >= ? GROUP BY status",
                (cutoff_iso,),
            ).fetchall():
                out["by_status"][row[0]] = int(row[1])
            out["total_pushed"] = out["by_status"].get("pushed", 0)
            out["total_skipped"] = out["by_status"].get("skipped", 0)

            # Cross-table: decisions del listener para drafts que pusheamos.
            # Match por draft_id (lo que el listener loggea como draft_id en
            # rag_draft_decisions = el mismo que pasamos en push_draft_to_listener).
            decisions = conn.execute(
                "SELECT d.decision, COUNT(*) FROM rag_draft_decisions d"
                " JOIN rag_proactive_drafts p ON p.id = d.draft_id"
                " WHERE p.ts >= ? GROUP BY d.decision",
                (cutoff_iso,),
            ).fetchall()
            for row in decisions:
                out["by_decision"][row[0]] = int(row[1])
    except Exception:
        return out

    decided = sum(out["by_decision"].values())
    useful = out["by_decision"].get("approved_si", 0) + out["by_decision"].get("approved_editar", 0)
    out["pending"] = max(0, out["total_pushed"] - decided)
    out["useful_rate"] = (useful / decided) if decided else 0.0
    return out


__all__ = [
    "compose_draft",
    "push_draft_to_listener",
    "log_proactive_draft",
    "stats",
    # Exposed for tests + Fase 2 signal extraction.
    "_allowlist_check",
    "_features_summary",
    "_parse_draft_response",
    "_portuguese_leak",
    "_new_draft_id",
    "_listener_push_url",
]
