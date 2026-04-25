"""Vocab refresh — extrae términos raros del corpus + correcciones para
inyectar al `--prompt` de whisper. Job nightly que llena `rag_whisper_vocab`.

## Sources

- **corrections** (priority 1.0) — palabras que el user manualmente corrigió
  via `/fix` command o vault_diff. Gold signal: el usuario explicitamente
  marcó esto como "lo que yo dije" vs "lo que whisper transcribió".
- **contacts** (priority 0.7) — nombres de Contactos.app, ya cacheados por
  el listener en `~/.local/share/whatsapp-listener/contacts-vocab.json`.
- **notes** (priority 0.5) — vocabulario del vault de Obsidian, filtrado a
  "rare but recurrent" (count ∈ [2, 50]). Términos ultra-frecuentes ya están
  en el modelo; términos hapax (count=1) son ruido.
- **chats** (priority 0.4) — vocabulario de mensajes de WhatsApp últimos
  30d. Mismo filtro que notes pero menor priority porque chat noise.

## Algoritmo

1. Tokenize cada source: lowercase, normalize unicode pero **preserva acentos**
   (whisper en español los necesita: "calendarizá" ≠ "calendariza").
2. Filter: stopwords ES+EN, longitud ∈ [3, 25] chars, no-numéricos.
3. Score = log(count + 1) (escala suave) × source_priority.
4. Dedup: si un term está en varios sources, gana el de priority más alta.
5. Top-N (default 500) por score → INSERT en `rag_whisper_vocab`.

## Cost

Para un vault de ~5K notas + 14K mensajes WhatsApp + 200 contactos:
- vault scan: ~3-5s (rglob + read).
- chat scan: ~500ms (SQLite indexed query).
- contacts: <10ms (JSON read).
- corrections: <50ms (SQL).
- TOTAL: ~5-10s nightly. Aceptable para un job background.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

# Source priorities — cuánto más alto, más weight final del término.
# Diseñado para que `corrections` siempre gane sobre `contacts`/`notes`/`chats`.
SOURCE_PRIORITY = {
    "corrections": 1.0,
    "contacts": 0.7,
    "notes": 0.5,
    "chats": 0.4,
}

# Stopwords ES + EN — replicada de rag.py `_TITLE_MATCH_STOPWORDS` para que
# este módulo sea self-contained (no creates circular import). Si crece la
# lista, podemos importar desde rag.py una vez resueltos los imports.
_STOPWORDS = frozenset({
    # Spanish — determinantes, conjunciones, preposiciones, WH, auxiliares.
    "de", "del", "la", "las", "el", "los", "un", "una", "unos", "unas",
    "en", "al", "que", "qué", "como", "cómo", "cuando", "cuándo",
    "donde", "dónde", "quien", "quién", "cual", "cuál", "por", "para",
    "con", "sin", "sobre", "entre", "hasta", "desde", "es", "son",
    "esta", "este", "estos", "estas", "ese", "esa", "esos", "esas",
    "aca", "alla", "aqui", "aquella", "aquel", "aquellos", "aquellas",
    "no", "si", "sí", "mas", "más", "muy", "tambien", "también",
    "se", "le", "les", "me", "te", "nos", "os", "yo", "tu", "vos",
    "mi", "mis", "tus", "su", "sus", "ya",
    "hay", "ser", "ir", "va", "voy", "vas", "fue", "era", "fui",
    "eran", "ha", "he", "has", "han", "habia", "había",
    "lo", "uno", "dos", "tres", "cuatro", "cinco",
    # Spanish — palabras funcionales muy comunes que aparecen ubicuamente
    # en notas y chats pero no son vocabulario distintivo del usuario.
    # Whisper ya las conoce perfecto, no aportan al prompt.
    "todo", "todos", "todas", "toda", "alguna", "alguno", "algunas", "algunos",
    "esto", "eso", "aquello", "ahi", "ahí", "alli", "allí",
    "hoy", "ayer", "mañana", "ahora", "luego", "antes", "despues", "después",
    "siempre", "nunca", "tarde", "temprano", "mucho", "poco", "pocos",
    "varias", "varios", "cada", "otra", "otro", "otras", "otros",
    "hace", "tiene", "tengo", "tener", "puede", "pueden", "puedo",
    "saber", "se", "decir", "dice", "dijo", "dije", "ver", "vez", "veces",
    "ir", "vas", "iba", "iban", "fueron", "vamos", "venir", "vino",
    "estar", "esta", "están", "estaba", "estuvo", "estamos",
    "hacer", "haga", "haz", "hizo", "hace", "haciendo",
    "querer", "quiero", "quiere", "queres", "querés", "queres",
    "deber", "debe", "debo", "debemos", "deberia", "debería",
    "pasa", "pasó", "paso", "pasar", "pasando",
    "decir", "dijo", "decia", "decía", "dicen",
    "parece", "parecio", "pareció", "parecía",
    "porque", "porqué", "pues", "entonces", "ademas", "además",
    "bien", "mal", "mejor", "peor", "menos", "mismo", "misma",
    "casi", "solo", "sólo", "solamente",
    "quizá", "quizás", "tal", "asi", "así",
    "horas", "minutos", "segundos", "dia", "dias", "días", "noche",
    "tiempo", "rato", "momento", "vez",
    "cosa", "cosas", "tema", "caso",
    "favor", "gracias",
    # English — determiners, conjunctions, prepositions, WH, auxiliaries.
    "of", "the", "an", "to", "at", "on", "for", "with", "from", "as",
    "by", "into", "out", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "what", "how", "when", "where",
    "who", "whom", "which", "this", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "it", "and", "or", "but",
    "so", "if", "then", "than", "there", "here",
    "do", "does", "did", "done", "doing",
    # English — palabras funcionales comunes en notas técnicas (pero no
    # vocabulario aprendible).
    "some", "any", "all", "each", "every", "many", "few", "more", "less",
    "most", "such", "very", "much", "also", "only", "just", "still",
    "now", "later", "before", "after", "always", "never", "often",
    "can", "could", "may", "might", "must", "should", "would", "will",
    "make", "made", "making", "get", "got", "getting", "go", "going", "went",
    "say", "said", "saying", "see", "saw", "seen",
    "use", "used", "using",
    "way", "thing", "things", "case", "stuff",
    "good", "great", "ok", "okay", "right", "left", "wrong",
    "new", "old", "next", "last", "first", "second", "third",
    "really", "actually", "basically", "essentially",
    "vez", "veces",  # también stopword en ambos idiomas
})

# Filtros de longitud — términos muy cortos no aportan al prompt y muy largos
# son típicamente artifacts (URLs partidas, hashes, etc.).
_MIN_LEN = 3
_MAX_LEN = 25

# Caps por source — usados en 2 fases:
#   (1) Pre-rank — cuántos términos por source antes del dedup global.
#   (2) Final — cuántos del FINAL (post-dedup) entran a `rag_whisper_vocab`.
#       Caps explícitos por source garantizan diversidad: aunque notes tenga
#       1000 términos válidos, el final solo deja entrar 200 — los slots
#       restantes son para contacts/chats/corrections sin starvation.
_TOP_NOTES = 300         # pre-rank
_TOP_CHATS = 200         # pre-rank
_TOP_CONTACTS = 150      # pre-rank
_TOP_CORRECTIONS = 100   # pre-rank

# Final caps (post-dedup, post-source-priority).
_FINAL_CAP_CORRECTIONS = 100  # gold signal — todo lo que el user corrigió.
_FINAL_CAP_CONTACTS = 100     # nombres de personas que mencionás seguido.
_FINAL_CAP_NOTES = 200        # vocab del vault.
_FINAL_CAP_CHATS = 100        # vocab de WA — menor cap por menor curación.
_TOP_TOTAL = (
    _FINAL_CAP_CORRECTIONS + _FINAL_CAP_CONTACTS
    + _FINAL_CAP_NOTES + _FINAL_CAP_CHATS
)  # = 500

# Filtros para "rare but recurrent" — para vault notes y chats. Términos con
# count=1 son hapax (noise); count <3 todavía es ruido. Sobre 15 es ya común
# y whisper probablemente lo conoce. La sweet spot está en términos que
# aparecen 3-15 veces — vocabulario que el user usa pero no es genérico.
_RARE_MIN_COUNT = 3
_RARE_MAX_COUNT = 15

# Path defaults
_CONTACTS_CACHE_PATH = Path.home() / ".local/share/whatsapp-listener/contacts-vocab.json"
_BRIDGE_DB_PATH = Path.home() / "repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db"


# ── Tokenization ──────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Tokenize preservando acentos — whisper en español los necesita.

    Pipeline:
    1. Lowercase.
    2. Split en non-word chars (preserving \\w para alfanuméricos + acentos
       UTF-8 via re.UNICODE).
    3. Filter por longitud [3, 25] + stopwords + no-numéricos.

    NO strippea acentos (a diferencia de `_tokenize_for_title_match` en rag.py)
    porque "calendarizá" y "calendariza" son palabras distintas para whisper.
    """
    if not text:
        return []
    # Normalizar a NFC (composed) por consistencia — algunos chars compuestos
    # se ven igual pero tienen distinta byte representation, lo cual rompe
    # PRIMARY KEY equality si una source tiene NFD y otra NFC.
    n = unicodedata.normalize("NFC", text.lower())
    tokens = re.split(r"[^\w]+", n, flags=re.UNICODE)
    return [t for t in tokens if _is_useful_term(t)]


def _is_useful_term(t: str) -> bool:
    if len(t) < _MIN_LEN or len(t) > _MAX_LEN:
        return False
    if t in _STOPWORDS:
        return False
    if t.isdigit():
        return False
    return True


# ── Source extractors ─────────────────────────────────────────────────────────


def _extract_vault_terms(vault_path: Path | None) -> Counter[str]:
    """Iterate vault `.md` files, count token frequencies. Skip directories
    típicamente de cross-source (03-Resources/{Chrome,Gmail,WhatsApp}) que
    son ruido (no vocabulario propio del usuario).
    """
    counts: Counter[str] = Counter()
    if not vault_path or not vault_path.is_dir():
        return counts
    skip_dirs = {
        ".trash", ".obsidian", ".git",
        # Cross-source folders — no son escritura "del usuario", son snapshots
        # automáticos. WhatsApp ETL en particular contamina con cantidades
        # masivas de jerga + mensajes ajenos.
        "Chrome", "Gmail", "WhatsApp", "GoogleDrive", "Calendar", "Reminders",
        "GitHub", "Claude", "YouTube", "Spotify",
    }
    for md_path in vault_path.rglob("*.md"):
        # Skip si algún componente del path está en la lista
        if any(part in skip_dirs for part in md_path.parts):
            continue
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for tok in _tokenize(text):
            counts[tok] += 1
    return counts


def _extract_chat_terms_from_bridge(days: int = 30) -> Counter[str]:
    """Read whatsapp-bridge SQLite, count tokens from messages last N days."""
    counts: Counter[str] = Counter()
    if not _BRIDGE_DB_PATH.exists():
        return counts
    cutoff_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - days * 86400)
    )
    try:
        # Read-only mode + immutable=1 para no chocar con el listener
        # ni el bridge que tienen el archivo abierto.
        uri = f"file:{_BRIDGE_DB_PATH}?mode=ro&immutable=1"
        con = sqlite3.connect(uri, uri=True, timeout=5.0)
        for row in con.execute(
            "SELECT content FROM messages "
            "WHERE timestamp > ? AND content IS NOT NULL AND content != ''",
            (cutoff_iso,),
        ):
            text = row[0]
            if not text:
                continue
            for tok in _tokenize(text):
                counts[tok] += 1
        con.close()
    except sqlite3.Error:
        # Schema diferente o DB locked — silent fail, vocab solo usa otros sources.
        return Counter()
    return counts


def _read_contacts_cache() -> list[str]:
    """Lee el cache de Contactos.app generado por el listener. Si no existe
    (primer setup, listener no corre), returns []."""
    if not _CONTACTS_CACHE_PATH.exists():
        return []
    try:
        d = json.loads(_CONTACTS_CACHE_PATH.read_text(encoding="utf-8"))
        return d.get("names", [])
    except Exception:
        return []


def _split_contact_names(names: Iterable[str]) -> Counter[str]:
    """De cada nombre 'Juan Pablo García' → counter({juan, pablo, garcía}).
    Mantiene acentos. Drop tokens duplicados dentro de un mismo nombre
    (raros, pero defensivo)."""
    counts: Counter[str] = Counter()
    for name in names:
        seen_in_name: set[str] = set()
        for tok in _tokenize(name):
            if tok in seen_in_name:
                continue
            seen_in_name.add(tok)
            counts[tok] += 1
    return counts


def _extract_corrections_terms(rag_module) -> Counter[str]:
    """Extrae palabras del `corrected` que NO estén en `original` para cada
    correction explicit/vault_diff. Esas son las palabras que whisper se
    equivocó y el user/sistema corrigió → gold signal del aprendizaje.

    No incluye source='llm' porque las correcciones del LLM son
    auto-generadas y no tienen el alto signal de las explicit.
    """
    counts: Counter[str] = Counter()
    try:
        with rag_module._ragvec_state_conn() as conn:
            for row in conn.execute(
                "SELECT original, corrected FROM rag_audio_corrections "
                "WHERE source IN ('explicit', 'vault_diff') "
                "ORDER BY ts DESC LIMIT 1000"
            ):
                orig_toks = set(_tokenize(row[0] or ""))
                corr_toks = _tokenize(row[1] or "")
                for tok in corr_toks:
                    if tok not in orig_toks:
                        counts[tok] += 1
    except Exception:
        return Counter()
    return counts


# ── Scoring ───────────────────────────────────────────────────────────────────


def _rare_terms_by_freq(
    counts: Counter[str],
    top_n: int,
    min_count: int = _RARE_MIN_COUNT,
    max_count: int = _RARE_MAX_COUNT,
) -> list[tuple[str, float]]:
    """Filter por count ∈ [min_count, max_count] (rare-but-recurrent), score
    log(count+1), top-N desc.

    Notas pueden dar count > max_count para términos muy comunes — los excluimos
    porque whisper ya los conoce. Hapax (count==1) son skipped: ruido.
    """
    if not counts:
        return []
    filtered = [
        (t, c) for t, c in counts.items()
        if min_count <= c <= max_count
    ]
    scored = [(t, math.log(c + 1)) for t, c in filtered]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


# ── Main entry points ─────────────────────────────────────────────────────────


def refresh_vocab(*, vault_path: Path | None = None) -> dict:
    """Full refresh of `rag_whisper_vocab`.

    Args:
        vault_path: si None, usa rag.VAULT_PATH (resolved via vault config).

    Returns:
        dict con stats: sources={name: count}, total_inserted, ms_elapsed.
    """
    import rag  # circular-safe lazy import
    t0 = time.perf_counter()
    if vault_path is None:
        vault_path = rag.VAULT_PATH

    sources_to_terms: dict[str, list[tuple[str, float]]] = {}

    # 1. Vault notes — rare-but-recurrent
    vault_counts = _extract_vault_terms(vault_path)
    sources_to_terms["notes"] = _rare_terms_by_freq(vault_counts, top_n=_TOP_NOTES)

    # 2. WhatsApp chats — rare-but-recurrent
    chat_counts = _extract_chat_terms_from_bridge(days=30)
    sources_to_terms["chats"] = _rare_terms_by_freq(chat_counts, top_n=_TOP_CHATS)

    # 3. Contacts — top-N por frecuencia (los nombres más comunes ganan).
    contact_names = _read_contacts_cache()
    contact_counts = _split_contact_names(contact_names)
    # No filter por count — los contactos son señal directa, todos importan.
    contact_scored = sorted(
        contact_counts.most_common(_TOP_CONTACTS),
        key=lambda x: -x[1],
    )
    sources_to_terms["contacts"] = [
        (t, math.log(c + 1)) for t, c in contact_scored
    ]

    # 4. Corrections — gold signal, todas las palabras corregidas pesan más.
    correction_counts = _extract_corrections_terms(rag)
    correction_scored = sorted(
        correction_counts.most_common(_TOP_CORRECTIONS),
        key=lambda x: -x[1],
    )
    sources_to_terms["corrections"] = [
        # Boost extra al log para que correcciones suban más en el ranking
        # final (priority * log(c+1) * 2 efectivamente).
        (t, 2.0 * math.log(c + 1)) for t, c in correction_scored
    ]

    # Apply source priority + dedup global. Iteramos en orden DESC por priority
    # (corrections > contacts > notes > chats) — el primer source que claim
    # un term gana, los siguientes lo skipean. Esto asegura que un nombre de
    # contacto siempre aparezca como 'contacts' aunque también esté en notas,
    # y una corrección explicit nunca sea pisada por una mention en notes.
    final_caps = {
        "corrections": _FINAL_CAP_CORRECTIONS,
        "contacts": _FINAL_CAP_CONTACTS,
        "notes": _FINAL_CAP_NOTES,
        "chats": _FINAL_CAP_CHATS,
    }
    final_terms: dict[str, tuple[float, str]] = {}
    per_source_count: dict[str, int] = {s: 0 for s in final_caps}
    for source_name in sorted(sources_to_terms.keys(), key=lambda s: -SOURCE_PRIORITY.get(s, 0.0)):
        terms = sources_to_terms[source_name]
        priority = SOURCE_PRIORITY.get(source_name, 0.1)
        cap = final_caps.get(source_name, 100)
        for term, raw_weight in terms:
            if term in final_terms:
                continue  # claimed by higher-priority source
            if per_source_count[source_name] >= cap:
                break  # source filled its quota
            weighted = raw_weight * priority
            final_terms[term] = (weighted, source_name)
            per_source_count[source_name] += 1

    # Order final list por weight DESC (el listener pone los primeros adelante
    # del prompt para máximo impacto).
    sorted_final = sorted(final_terms.items(), key=lambda x: -x[1][0])

    now = time.time()
    with rag._ragvec_state_conn() as conn:
        conn.execute("DELETE FROM rag_whisper_vocab")
        conn.executemany(
            "INSERT INTO rag_whisper_vocab "
            "(term, weight, source, last_seen_ts, refreshed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (term, weight, source, now, now)
                for term, (weight, source) in sorted_final
            ],
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "sources": {k: len(v) for k, v in sources_to_terms.items()},
        "total_inserted": len(sorted_final),
        "ms_elapsed": elapsed_ms,
    }


def get_top_vocab_terms(limit: int = 100, source: str | None = None) -> list[dict]:
    """Read top-N vocab terms by weight. Used by:
    - El listener (Bun) para armar el prompt dinámico (lee directo SQL).
    - El dashboard (Step 3) para mostrar qué está aprendiendo el sistema.

    Args:
        limit: top-N a retornar.
        source: filtrar a un source específico ('notes', 'chats', etc.).
            None = todos los sources.

    Returns: list[dict] con keys term, weight, source, last_seen_ts.
    """
    import rag
    try:
        with rag._ragvec_state_conn() as conn:
            if source:
                rows = conn.execute(
                    "SELECT term, weight, source, last_seen_ts "
                    "FROM rag_whisper_vocab WHERE source = ? "
                    "ORDER BY weight DESC LIMIT ?",
                    (source, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT term, weight, source, last_seen_ts "
                    "FROM rag_whisper_vocab ORDER BY weight DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [
            {"term": r[0], "weight": r[1], "source": r[2], "last_seen_ts": r[3]}
            for r in rows
        ]
    except Exception:
        return []
