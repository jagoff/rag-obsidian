"""Identity Layer 2 — style fingerprint del user, injectado al system prompt.

Game-Changer G2 (2026-05-11). Hasta acá los LLMs del sistema producían
respuestas en "español rioplatense genérico" porque el system prompt
solo decía "usá voseo". Pero el user tiene 5 años de outbounds en WA
+ vault + commits — el corpus YA contiene cómo escribe él
específicamente. Este módulo extrae el patrón y lo inyecta a los
prompts del chat / synthesis / draft loop para que el output suene a
él, no a LLM-rioplatense-default.

Pipeline:

1. `compute_fingerprint()` cada 24h via supervisor job:
   - Scan `messages.db` últimos 90d, `is_from_me=1` (outbounds del user).
   - Filtra texto: drop stickers/audios/forwards, len ≥3 palabras.
   - Extrae features:
     * Avg msg length en palabras.
     * Top 30 tokens distinctive (after stopword filter).
     * Frequency de chat slang ("dale", "tranqui", "che", "posta",
       "joya", "barbaro", "buenisimo", "perfecto").
     * Emoji usage rate.
     * Uso de signo de puntuación final (yes/no).
     * Code-switching ES/EN markers.
   - Persistir a `~/.local/share/obsidian-rag/identity_fingerprint.json`
     con `computed_at` ISO timestamp.

2. `summarize_for_prompt()` lee el JSON y genera 5-10 líneas concretas
   en formato directiva, no markdown:
       Tu user (Fer) escribe así:
       - Longitud típica: 14 palabras por msg.
       - Usa "dale" a fin de msg en 38% de respuestas afirmativas.
       - Rara vez pone signo de puntuación al final de msgs cortos.
       - Emoji frequency: 12% de msgs.

3. Caller (system_prompt_for_intent / draft generator / brief) appendea
   el resumen al system prompt cuando el intent es CONVERSACIONAL.
   Skipea para lookup/comparison/synthesis (esas son factuales, el
   estilo no aplica).

Cache TTL: 7 días. Stale > 7d se ignora silenciosamente — caller cae
al prompt sin fingerprint. El recompute es barato (~5s para 90d).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


_FP_PATH = Path.home() / ".local/share/obsidian-rag/identity_fingerprint.json"
_FP_TTL_S = 7 * 86400


# Stopwords español + chat noise — descartadas del top-vocab.
_STOPWORDS = frozenset({
    "que", "de", "la", "el", "y", "a", "en", "no", "es", "con", "por",
    "para", "se", "un", "una", "los", "las", "del", "al", "lo", "su",
    "como", "más", "mas", "me", "te", "le", "si", "ya", "yo", "vos",
    "tu", "mi", "ese", "esa", "esto", "eso", "esta", "este", "pero",
    "ok", "si", "no", "ja", "jaja", "jajaja", "jajajaja",
    "https", "http", "www", "com", "ar", "the", "and", "or", "to",
})

# Slang rioplatense típico — frecuencia tracked individualmente.
_SLANG_MARKERS = (
    "dale", "tranqui", "che", "posta", "joya", "barbaro", "buenisimo",
    "perfecto", "obvio", "claro", "esta", "estas", "boludo", "loco",
    "pibe", "mina", "tipo", "onda", "bien", "mal", "tipo",
)

_TOKEN_RE = re.compile(r"\b[a-záéíóúñü]{2,}\b", re.IGNORECASE)
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U00002600-\U000027BF]")


def _bridge_db_path():
    import rag as _rag  # noqa: PLC0415
    return _rag.WHATSAPP_DB_PATH


def _scan_outbound_texts(days: int = 90) -> list[str]:
    """Devuelve lista de texts outbound del user, filtrados a útiles."""
    bridge = _bridge_db_path()
    if not bridge.is_file():
        return []
    cutoff_dt = datetime.now() - timedelta(days=days)
    from rag.integrations.whatsapp.fetch import _BRIDGE_TZ_OFFSET  # noqa: PLC0415
    cutoff = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S") + _BRIDGE_TZ_OFFSET
    try:
        con = sqlite3.connect(f"file:{bridge}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        return []
    try:
        # `media_type` puede ser NULL o '' para msgs de texto puro —
        # ambos casos son válidos. Solo descartamos los que SÍ tienen
        # un media_type (audio/image/video/document/sticker).
        # Bot_jid se excluye porque es el grupo RagNet propio donde
        # se mandan outputs del sistema, no charlas del user.
        import rag as _rag  # noqa: PLC0415
        bot_jid = _rag.WHATSAPP_BOT_JID
        rows = con.execute(
            """
            SELECT content FROM messages
            WHERE is_from_me = 1
              AND timestamp > ?
              AND content IS NOT NULL AND length(content) > 6
              AND (media_type IS NULL OR media_type = '')
              AND chat_jid NOT LIKE '%status@broadcast'
              AND chat_jid != ?
            LIMIT 30000
            """,
            (cutoff, bot_jid),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows if r[0] and r[0].strip()]


def compute_fingerprint(days: int = 90) -> dict:
    """Computa fingerprint completo + persiste a disk. Retorna el dict.

    Idempotente — corre cuantas veces quieras, gana la última.
    """
    texts = _scan_outbound_texts(days=days)
    if not texts:
        out = {
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "n_samples": 0,
            "stale": True,
            "note": "no outbound samples — bridge db vacía o gate cerrado",
        }
        _persist(out)
        return out

    n_msgs = len(texts)
    # Longitud típica.
    lens = [len(t.split()) for t in texts]
    avg_len = round(sum(lens) / n_msgs, 1)
    median_len = sorted(lens)[n_msgs // 2]
    # Top vocab distinctive.
    counter: Counter = Counter()
    for t in texts:
        for tok in _TOKEN_RE.findall(t.lower()):
            if tok in _STOPWORDS or len(tok) < 3:
                continue
            counter[tok] += 1
    top_tokens = [w for w, _ in counter.most_common(30)]
    # Slang frequency.
    slang_freq = {}
    for marker in _SLANG_MARKERS:
        hits = sum(
            1 for t in texts if re.search(rf"\b{re.escape(marker)}\b", t, re.IGNORECASE)
        )
        if hits >= 5:
            slang_freq[marker] = round(hits / n_msgs * 100, 1)
    # Emoji rate.
    emoji_msgs = sum(1 for t in texts if _EMOJI_RE.search(t))
    emoji_rate = round(emoji_msgs / n_msgs * 100, 1)
    # Punctuation-end rate (¿termina msg corto con signo?).
    short = [t for t in texts if len(t.split()) <= 4]
    if short:
        end_punct = sum(1 for t in short if t.rstrip()[-1] in ".!?,;:")
        short_punct_rate = round(end_punct / len(short) * 100, 1)
    else:
        short_punct_rate = 0.0

    out = {
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": n_msgs,
        "avg_words_per_msg": avg_len,
        "median_words_per_msg": median_len,
        "top_tokens": top_tokens,
        "slang_freq_pct": slang_freq,
        "emoji_rate_pct": emoji_rate,
        "short_msg_punct_rate_pct": short_punct_rate,
        "stale": False,
    }
    _persist(out)
    return out


def _persist(fp: dict) -> None:
    try:
        _FP_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FP_PATH.write_text(json.dumps(fp, ensure_ascii=False, indent=2))
    except OSError as e:
        logger.warning("identity_layer: persist failed: %s", e)


def _load_fresh() -> dict | None:
    """Carga el fingerprint si existe Y está dentro del TTL."""
    if not _FP_PATH.is_file():
        return None
    try:
        data = json.loads(_FP_PATH.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    ca = data.get("computed_at")
    if ca:
        try:
            age_s = (datetime.now() - datetime.fromisoformat(ca)).total_seconds()
            if age_s > _FP_TTL_S:
                return None
        except ValueError:
            pass
    if data.get("stale"):
        return None
    if data.get("n_samples", 0) < 50:
        # Muestreo insuficiente — no inyectar para no contaminar con ruido.
        return None
    return data


def summarize_for_prompt() -> str:
    """Genera 5-8 líneas concretas para appendear al system prompt.

    Devuelve "" si no hay fingerprint fresco o muestreo insuficiente —
    caller cae al prompt base sin distorsión.
    """
    fp = _load_fresh()
    if not fp:
        return ""
    lines = ["", "Cómo escribe el user (extraído de sus últimos msgs):"]
    avg = fp.get("avg_words_per_msg", 0)
    if avg:
        lines.append(f"- Longitud típica: {avg} palabras por mensaje.")
    slang = fp.get("slang_freq_pct", {})
    if slang:
        top_slang = sorted(slang.items(), key=lambda kv: -kv[1])[:5]
        slang_str = ", ".join(f'"{w}" {p}%' for w, p in top_slang)
        lines.append(f"- Slang frecuente: {slang_str}.")
    emoji = fp.get("emoji_rate_pct", 0)
    if emoji is not None:
        lines.append(f"- Emoji rate: {emoji}% de los msgs.")
    punct = fp.get("short_msg_punct_rate_pct", 0)
    if punct is not None:
        if punct < 20:
            lines.append(
                "- Rara vez termina msgs cortos con punto/coma/etc — "
                "evitá puntuación final en respuestas breves."
            )
        elif punct > 70:
            lines.append("- Termina msgs cortos con puntuación. Mantenelo.")
    top_tokens = fp.get("top_tokens", [])[:10]
    if top_tokens:
        lines.append(f"- Vocabulario habitual: {', '.join(top_tokens)}.")
    lines.append(
        "Replicá ese registro cuando generes msgs en su nombre. Sin "
        "calcar tokens uno por uno — capturá el feel."
    )
    return "\n".join(lines)


__all__ = ["compute_fingerprint", "summarize_for_prompt"]
