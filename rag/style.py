"""Style fingerprint del user — distila cómo escribe a partir de
mensajes outbound del WhatsApp bridge.

El sistema NUNCA persiste contenido raw — solo features agregados:
openers/closers favoritos, tasas (emoji, caps, ¿), markers de voseo
y slang rioplatense, distribución de longitudes. Eso permite que el
agente compose drafts mimetizando el tono sin tener un dump literal
de los chats del user en disco.

Pipeline (`refresh()`):

1. Lee `messages` del bridge SQLite (`is_from_me=1`, últimos
   ``window_days`` días, no-bot, no-comando).
2. Extrae 12 features agregados (`extract_features()`).
3. Persiste a ``rag_style_fingerprint`` en `telemetry.db` (1 row por
   refresh, conserva historial para ver evolución).
4. Renderea snapshot markdown a
   ``99-AI/style/profile.md`` para que el user pueda revisar lo que
   el sistema "aprendió" sobre su tono.

Privacy: NO se hace embedding de mensajes ni se guarda contenido
plaintext. Los features son contadores + frecuencias agregadas.

Uso desde el draft loop (layer 2 — todavía no implementado):
``inject_style_prompt()`` retorna el system prompt fragment que
explica el tono. Layer 3 (few-shot retrieval per-contact) queda
pendiente.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "extract_features",
    "load_latest",
    "refresh",
    "render_markdown",
    "_filter_user_messages",
]


_BOT_MARKER = "​"  # zero-width space — prefijo del bot en outbound

_MIN_LEN = 4
_MAX_LEN = 2000  # cap defensivo, mensajes más largos suelen ser pegado
_DEFAULT_WINDOW_DAYS = 90

# Markers para detectar slang argentino + voseo en el contenido.
_VOSEO_RE = re.compile(
    r"\b(vos|podés|tenés|querés|fijate|mirá|agarrá|tomá|"
    r"andá|venís|escribís|sabés|hacés|decís|sos|hablás)\b",
    re.IGNORECASE,
)
_TUTEO_RE = re.compile(
    r"\b(tú|tienes|puedes|quieres|fíjate|mira|agarra|toma|"
    r"vienes|escribes|sabes|haces|dices|eres|hablas)\b",
    re.IGNORECASE,
)
_SLANG_RE = re.compile(
    r"\b(che|dale|tranqui|joya|fuaa|boludo|pelotudo|laburar|"
    r"posta|piola|copado|grosso|bardo|quilombo|gauchada|"
    r"bondi|guita|laburo|laburante|chamuyo|pibe|mina)\b",
    re.IGNORECASE,
)
_RE_PREFIX_RE = re.compile(r"\bre\s+\w+", re.IGNORECASE)
_LAUGH_RE = re.compile(r"\b((?:ja){2,})\b", re.IGNORECASE)
_ABBREV_RE = re.compile(
    r"\b(tmb|tb|tbn|xq|q|x|d|pq|porq)\b", re.IGNORECASE,
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # extended pictographs
    "\U00002600-\U000027BF"  # dingbats
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "]"
)


# ── Bridge fetch ─────────────────────────────────────────────────────────


def _bridge_db_path() -> Path | None:
    """Busca el path del bridge sqlite. None si no se encuentra."""
    try:
        from rag.integrations.whatsapp import WHATSAPP_BRIDGE_DB_PATH  # noqa: PLC0415
    except Exception:
        return None
    p = Path(WHATSAPP_BRIDGE_DB_PATH)
    return p if p.exists() else None


def _fetch_outbound_raw(
    db_path: Path, since_iso: str,
) -> list[str]:
    """Lee contenidos `is_from_me=1` desde ``since_iso`` (formato bridge,
    ej. ``2026-02-09 12:00:00-03:00``). Silent-fail → ``[]``."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        logger.warning("style: bridge open failed: %s", exc)
        return []
    try:
        rows = conn.execute(
            "SELECT content FROM messages "
            "WHERE is_from_me=1 AND content IS NOT NULL AND content != '' "
            "AND timestamp >= ?",
            (since_iso,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("style: bridge query failed: %s", exc)
        return []
    finally:
        with contextlib.suppress(Exception):
            conn.close()
    return [r[0] for r in rows if r[0]]


def _filter_user_messages(raw: list[str]) -> list[str]:
    """Quita bot replies (zero-width-space prefix), comandos del bot
    (`/...`), URLs solas, ultra-cortas y ultra-largas."""
    out: list[str] = []
    for msg in raw:
        if not msg:
            continue
        if msg.startswith(_BOT_MARKER):
            continue
        s = msg.strip()
        if not s or len(s) < _MIN_LEN or len(s) > _MAX_LEN:
            continue
        if s.startswith("/"):
            continue
        if s.startswith("http://") or s.startswith("https://"):
            # mensaje que es solo un URL (sin texto extra)
            if " " not in s:
                continue
        out.append(s)
    return out


# ── Feature extraction ───────────────────────────────────────────────────


def _first_token(s: str) -> str:
    s = s.lstrip()
    if not s:
        return ""
    parts = s.split(maxsplit=1)
    return parts[0].lower().rstrip(",.;:!?¡¿") if parts else ""


def _last_token(s: str) -> str:
    s = s.rstrip()
    if not s:
        return ""
    parts = s.rsplit(maxsplit=1)
    return parts[-1].lower().rstrip(",.;:!?¡¿") if parts else ""


def extract_features(messages: list[str]) -> dict[str, Any]:
    """Calcula 12 features agregados sobre la lista de mensajes ya
    filtrada (``_filter_user_messages``).

    Devuelve dict listo para JSON-serialize.
    """
    n = len(messages)
    if n == 0:
        return {"n_messages": 0, "insufficient_data": True}

    lens = [len(m) for m in messages]
    lens_sorted = sorted(lens)
    p50 = lens_sorted[n // 2]
    p95 = lens_sorted[min(n - 1, int(n * 0.95))]

    openers = Counter(_first_token(m) for m in messages if _first_token(m))
    closers = Counter(_last_token(m) for m in messages if _last_token(m))

    n_emoji = sum(1 for m in messages if _EMOJI_RE.search(m))
    n_caps_start = sum(1 for m in messages if m[:1].isupper())
    n_lowercase_only = sum(1 for m in messages if m == m.lower())
    n_question_open = sum(1 for m in messages if m.startswith("¿"))
    n_exclam_open = sum(1 for m in messages if m.startswith("¡"))

    voseo_hits = sum(len(_VOSEO_RE.findall(m)) for m in messages)
    tuteo_hits = sum(len(_TUTEO_RE.findall(m)) for m in messages)
    slang_hits = sum(len(_SLANG_RE.findall(m)) for m in messages)
    re_prefix_hits = sum(len(_RE_PREFIX_RE.findall(m)) for m in messages)
    abbrev_hits = sum(len(_ABBREV_RE.findall(m)) for m in messages)

    laughs = []
    for m in messages:
        for match in _LAUGH_RE.finditer(m):
            laughs.append(len(match.group(1)) // 2)  # cuántas "ja" repetidas
    laugh_dominant = Counter(laughs).most_common(1)
    laugh_typical_len = laugh_dominant[0][0] if laugh_dominant else 0

    return {
        "n_messages": n,
        "avg_chars": round(sum(lens) / n, 1),
        "p50_chars": p50,
        "p95_chars": p95,
        "openers_top": openers.most_common(10),
        "closers_top": closers.most_common(10),
        "emoji_rate": round(n_emoji / n, 3),
        "caps_start_rate": round(n_caps_start / n, 3),
        "lowercase_only_rate": round(n_lowercase_only / n, 3),
        "question_open_rate": round(n_question_open / n, 3),
        "exclam_open_rate": round(n_exclam_open / n, 3),
        "voseo_hits": voseo_hits,
        "tuteo_hits": tuteo_hits,
        "voseo_dominance": (
            round(voseo_hits / max(1, voseo_hits + tuteo_hits), 3)
        ),
        "slang_hits": slang_hits,
        "re_prefix_hits": re_prefix_hits,
        "abbrev_hits": abbrev_hits,
        "laugh_typical_jas": laugh_typical_len,
        "insufficient_data": False,
    }


# ── Persistence ──────────────────────────────────────────────────────────


def _ensure_table() -> None:
    """Idempotent: crea ``rag_style_fingerprint`` si no existe."""
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
    except Exception as exc:
        logger.warning("style: cannot import _ragvec_state_conn: %s", exc)
        return
    with _ragvec_state_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_style_fingerprint (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at REAL NOT NULL,
                window_days INTEGER NOT NULL,
                n_messages INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_style_fp_computed_at "
            "ON rag_style_fingerprint(computed_at DESC)"
        )


def _persist(features: dict[str, Any], window_days: int, content_hash: str) -> int:
    """Inserta una row nueva. Retorna `rowid`."""
    _ensure_table()
    from rag import _ragvec_state_conn  # noqa: PLC0415
    with _ragvec_state_conn() as conn:
        cur = conn.execute(
            "INSERT INTO rag_style_fingerprint "
            "(computed_at, window_days, n_messages, features_json, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                time.time(),
                window_days,
                int(features.get("n_messages", 0)),
                json.dumps(features, ensure_ascii=False, default=str),
                content_hash,
            ),
        )
        return int(cur.lastrowid or 0)


def load_latest() -> dict[str, Any] | None:
    """Lee la fingerprint más reciente. None si no hay ninguna."""
    _ensure_table()
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
    except Exception:
        return None
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT id, computed_at, window_days, n_messages, "
                "features_json, content_hash "
                "FROM rag_style_fingerprint ORDER BY computed_at DESC LIMIT 1"
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("style: load_latest failed: %s", exc)
        return None
    if row is None:
        return None
    fid, computed_at, window_days, n_msgs, features_json, content_hash = row
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError:
        features = {}
    return {
        "id": fid,
        "computed_at": computed_at,
        "computed_at_iso": (
            datetime.fromtimestamp(computed_at).isoformat(timespec="seconds")
        ),
        "window_days": window_days,
        "n_messages": n_msgs,
        "features": features,
        "content_hash": content_hash,
    }


# ── Markdown export ──────────────────────────────────────────────────────


def render_markdown(snapshot: dict[str, Any]) -> str:
    """Render del fingerprint como nota Obsidian para revisión humana."""
    f = snapshot.get("features", {})
    if f.get("insufficient_data"):
        return "# Style fingerprint\n\n_Sin datos suficientes._\n"

    lines = [
        "---",
        f"computed_at: {snapshot.get('computed_at_iso', '?')}",
        f"window_days: {snapshot.get('window_days', '?')}",
        f"n_messages: {snapshot.get('n_messages', 0)}",
        "type: system/style-fingerprint",
        "tags: [style, system, whatsapp, profile]",
        "---",
        "",
        "# Cómo escribís — fingerprint",
        "",
        f"_Calculado sobre {snapshot.get('n_messages', 0)} mensajes outbound de "
        f"WhatsApp en los últimos {snapshot.get('window_days', '?')} días "
        f"(filtrando bot replies + comandos)._",
        "",
        "## Tamaño",
        "",
        f"- avg: **{f.get('avg_chars', '?')}** chars/msg",
        f"- p50: {f.get('p50_chars', '?')} chars",
        f"- p95: {f.get('p95_chars', '?')} chars",
        "",
        "## Openers favoritos",
        "",
    ]
    for tok, n in f.get("openers_top", [])[:10]:
        lines.append(f"- `{tok}` × {n}")
    lines += ["", "## Closers favoritos", ""]
    for tok, n in f.get("closers_top", [])[:10]:
        lines.append(f"- `{tok}` × {n}")
    lines += [
        "",
        "## Tono",
        "",
        f"- emoji rate: {f.get('emoji_rate', 0):.1%}",
        f"- caps al inicio: {f.get('caps_start_rate', 0):.1%}",
        f"- todo minúsculas: {f.get('lowercase_only_rate', 0):.1%}",
        f"- abre con `¿`: {f.get('question_open_rate', 0):.1%}",
        f"- abre con `¡`: {f.get('exclam_open_rate', 0):.1%}",
        "",
        "## Voseo / slang argentino",
        "",
        f"- voseo hits: {f.get('voseo_hits', 0)}",
        f"- tuteo hits: {f.get('tuteo_hits', 0)}",
        f"- voseo dominance: {f.get('voseo_dominance', 0):.1%}",
        f"- slang hits (`che`, `dale`, `joya`, `fuaa`, etc.): "
        f"{f.get('slang_hits', 0)}",
        f"- `re ` prefix (ej. `re bien`): {f.get('re_prefix_hits', 0)}",
        f"- abreviaciones (`tmb`, `xq`, `q `): {f.get('abbrev_hits', 0)}",
        f"- risa típica: `{'ja' * f.get('laugh_typical_jas', 0) or '—'}`",
        "",
        "---",
        "",
        "_Este fingerprint es input al draft loop del bot WA — el "
        "sistema lo usa para mimetizar tu tono cuando arma respuestas en "
        "tu nombre._",
        "",
    ]
    return "\n".join(lines)


def _vault_export_path() -> Path | None:
    """Path al markdown del vault. None si vault no resuelve."""
    try:
        from rag import _resolve_vault_path  # noqa: PLC0415
        vault = Path(_resolve_vault_path())
    except Exception:
        return None
    return vault / "99-obsidian" / "99-AI" / "style" / "profile.md"


# ── Refresh entrypoint ───────────────────────────────────────────────────


def refresh(
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    persist: bool = True,
    export_markdown: bool = True,
) -> dict[str, Any]:
    """Levanta los mensajes outbound del último ``window_days``,
    extrae features, persiste a SQL + markdown opcional.

    Retorna el snapshot completo (mismo shape que ``load_latest()``).
    """
    db = _bridge_db_path()
    if db is None:
        return {
            "ok": False,
            "reason": "WhatsApp bridge no disponible",
            "n_messages": 0,
        }

    since_dt = datetime.now() - timedelta(days=window_days)
    # Bridge timestamps: `YYYY-MM-DD HH:MM:SS-03:00`
    since_iso = since_dt.strftime("%Y-%m-%d %H:%M:%S")

    raw = _fetch_outbound_raw(db, since_iso)
    msgs = _filter_user_messages(raw)
    features = extract_features(msgs)

    content_hash = hashlib.sha256(
        json.dumps(features, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    fid = -1
    if persist:
        try:
            fid = _persist(features, window_days, content_hash)
        except Exception as exc:
            logger.warning("style: persist failed: %s", exc)

    snapshot = {
        "ok": True,
        "id": fid,
        "computed_at": time.time(),
        "computed_at_iso": datetime.now().isoformat(timespec="seconds"),
        "window_days": window_days,
        "n_messages": features.get("n_messages", 0),
        "n_raw": len(raw),
        "features": features,
        "content_hash": content_hash,
    }

    if export_markdown and not features.get("insufficient_data"):
        path = _vault_export_path()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_markdown(snapshot), encoding="utf-8")
                snapshot["exported_to"] = str(path)
            except OSError as exc:
                logger.warning("style: export failed: %s", exc)

    return snapshot
