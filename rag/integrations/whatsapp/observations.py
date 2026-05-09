"""Contact observation logger — anota observaciones sobre contactos en su
nota del vault (`99-Contacts/<Nombre>.md`).

El user pidió 2026-04-26: "[las notas de 99-Contacts/] son contactos
'vivos', cada información relevante sobre ese contacto debería almacenarse
ahí". Ej: "Seba te llevo un vino" → la nota de Seba debería anotar
"Preferencia de bebidas: Vino".

Doble write (decisión user 2026-04-26 "las dos cosas"):

1. Sección ``## Observaciones`` al final de la nota — bullet con
   timestamp + texto crudo. Auditoría completa, no se borra nada.
2. Si la observación tiene `category` (ej. "Preferencias"), también
   append al bullet ``**<category>**:`` existente o crear uno nuevo.
   Vista consolidada para queries del RAG ("a Seba qué le gusta?").

Idempotencia: hash del observation text — si el mismo texto ya existe en
las últimas N obs, skip. Permite que el LLM dispare el tool múltiples
veces sin spammear la nota.

Pipeline LLM-assisted:
``_infer_observation_category`` usa qwen2.5:3b (HELPER_MODEL) para
inferir qué bullet de la nota corresponde a una observación libre,
mirando los bullets existentes + categorías estándar del template.
Silent-fail: si el LLM rebota → None → caller va solo a `## Observaciones`.
"""

from __future__ import annotations

import hashlib
import re
import threading
from datetime import datetime
from pathlib import Path

from rag.integrations.whatsapp._constants import (
    _CONTACT_OBS_STANDARD_CATEGORIES,
    _OBSERVATIONS_HEADING,
)


# ── Cache + obvious-pattern fast-path ───────────────────────────────────────
# Cache LRU manual: (obs_hash, body_hash) → category. Cap conservador (256
# entries) — el universo de observations × cuerpos de nota distintos es
# pequeño y si se llena, FIFO eviction. Lock para thread-safety (web server
# corre en thread pool de uvicorn; CLI single-thread).
_OBS_CATEGORY_CACHE: dict[tuple[str, str], str | None] = {}
_OBS_CATEGORY_CACHE_LOCK = threading.Lock()
_OBS_CATEGORY_CACHE_MAX = 256

# Patrones que matchean categorías OBVIAS sin necesidad de LLM. Cada entry:
# (compiled_regex, category_name). Lista CONSERVADORA — si el regex no
# matchea, caemos al LLM (no degradamos calidad). False-positive rate aceptado:
# cero — solo agregamos patrones donde "X → categoría Y" es 100% determinista.
_OBVIOUS_OBS_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # Preferencias — "le gusta X", "prefiere X", "le encanta X"
    (re.compile(r"^\s*(le\s+gust\w+|prefiere|le\s+encanta|adora)\b", re.IGNORECASE), "Preferencias"),
    # Cumpleaños — fecha + cumple
    (re.compile(r"\b(cumple\w*|cumplea[ñn]os|nacimiento|nac[ií]o)\b", re.IGNORECASE), "Cumpleaños"),
    # Trabajo — "trabaja en X", "labura en", "se mudó a [trabajo]"
    (re.compile(r"\b(trabaja|labura|se\s+jubil|jubilad[oa]|empez[oó]\s+en|cambi[oó]\s+de\s+trabajo)\b", re.IGNORECASE), "Trabajo / contexto"),
    # Eventos — "viaje a X", "se casa", "tuvo un hijo"
    (re.compile(r"\b(viaje\s+a|se\s+cas\w+|tuvo\s+(un|una)\s+(hij\w+|beb)|se\s+mud\w+\s+a|operaci[oó]n)\b", re.IGNORECASE), "Eventos importantes"),
)


def _hash_short(s: str) -> str:
    """sha1[:16] — cheap, suficiente para cache key (no security-sensitive)."""
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def _match_obvious_category(observation: str) -> str | None:
    """Pre-filter: si la observation matchea un patrón OBVIO, devuelve la
    categoría sin LLM call. Si no matchea, devuelve None (caller cae al LLM).
    """
    for pattern, category in _OBVIOUS_OBS_PATTERNS:
        if pattern.search(observation):
            return category
    return None


def _infer_observation_category(
    observation: str,
    note_body: str = "",
    *,
    max_categories: int = 8,
) -> str | None:
    """Usa el LLM barato (HELPER_MODEL, qwen2.5:3b) para inferir qué bullet
    de la nota del contacto corresponde a una observación libre.

    Args:
        observation: texto ya procesado ("Le gusta el vino", "Trabaja en
            cooperativa", "Anda sensible con el tema herencia").
        note_body: cuerpo markdown de la nota del contacto (sin frontmatter
            necesariamente). Se usa para ver qué bullets `- **<cat>**:
            ...` ya existen y preferirlos sobre crear uno nuevo.
        max_categories: límite de categorías candidatas a presentar al LLM.

    Returns:
        Nombre de la categoría (matchea un bullet existente o crea uno
        nuevo con ese nombre) o `None` si el LLM falla o responde que
        no hay categoría apropiada. `None` = caller va solo a la
        sección `## Observaciones` (auditoría pura).

    Invariantes:
    - Silent-fail: cualquier error del LLM → `None`.
    - Timeout corto (heredado del helper client = 30s).
    - Idempotente: prompt es determinístico (temperature=0, seed=42).
    - No imprime nada — los callers silencian errores.

    Optimizaciones (2026-05-09):
    - Fast-path regex obvio (`_match_obvious_category`) skipea LLM para
      patrones comunes ("le gusta", "trabaja en", "cumpleaños").
    - Cache LRU `(obs_hash, body_hash) → category` evita LLM call repetida
      cuando la misma observation se procesa 2 veces (e.g., el watchdog del
      ambient agent dispara dos veces seguidas).
    """
    if not observation or not observation.strip():
        return None

    obs_clean = observation.strip()

    # Fast-path 1: regex obvios. Matchea ~30-40% de las observations comunes
    # ("le gusta X" → Preferencias) sin gastar LLM call (~200-500ms ahorrados).
    obvious = _match_obvious_category(obs_clean)
    if obvious:
        return obvious

    # Fast-path 2: cache (obs_hash, body_hash) → category. Mismo prompt + mismo
    # body siempre da misma respuesta determinística (HELPER_OPTIONS
    # temperature=0 + seed=42), entonces es seguro cachear.
    cache_key = (_hash_short(obs_clean), _hash_short(note_body or ""))
    with _OBS_CATEGORY_CACHE_LOCK:
        if cache_key in _OBS_CATEGORY_CACHE:
            return _OBS_CATEGORY_CACHE[cache_key]

    # Categorías candidatas = existentes en la nota + estándar del template.
    # Priorizamos las existentes (orden preservado) para que el LLM prefiera
    # matchear una ya presente antes de inventar una nueva.
    existing: list[str] = []
    if note_body:
        # Parseamos bullets `- **<X>**: ...` del cuerpo de la nota. No
        # distingue frontmatter — el regex matchea solo líneas con el
        # patrón exacto del bullet.
        bullet_re = re.compile(
            r"^-\s*\*\*\s*([^*]+?)\s*\*\*\s*:",
            re.MULTILINE,
        )
        for m in bullet_re.finditer(note_body):
            cat = m.group(1).strip()
            # Descartamos estructurales que no aplican a observaciones
            # libres (Teléfono, Email, Dirección, Apellido, Relación).
            if cat.lower() in {
                "teléfono", "telefono", "email", "dirección", "direccion",
                "apellido / nombre completo", "apellido", "relación", "relacion",
            }:
                continue
            if cat not in existing:
                existing.append(cat)

    candidates = existing + [
        c for c in _CONTACT_OBS_STANDARD_CATEGORIES if c not in existing
    ]
    candidates = candidates[:max_categories]

    # Prompt del LLM — corto, determinístico, con ejemplos few-shot.
    # Evitamos dar demasiados ejemplos (cost) pero los suficientes para
    # que el modelo entienda el shape esperado.
    prompt = (
        "Sos un asistente argentino que clasifica observaciones sobre personas "
        "en una de estas categorías de su ficha de contacto. Respondé SIEMPRE "
        "en español rioplatense (voseo). Nunca portugués.\n\n"
        + "\n".join(f"- {c}" for c in candidates)
        + '\n- (ninguna)\n\n'
        "Regla: respondé con EXACTAMENTE una línea, solo el nombre de "
        "la categoría (sin comillas, sin explicación). Si ninguna "
        "aplica, respondé `(ninguna)`.\n\n"
        "Ejemplos:\n"
        "Observación: Le gusta el vino → Preferencias\n"
        "Observación: Trabaja en la cooperativa → Trabajo / contexto\n"
        "Observación: Anda con problemas de salud → Notas\n"
        "Observación: Cumpleaños: 26 de mayo → Cumpleaños\n"
        "Observación: Se mudó a San Pedro → Trabajo / contexto\n"
        "Observación: Le gustan las plantas → Preferencias\n"
        "Observación: Habló del viaje a Europa → Eventos importantes\n"
        f"Observación: {observation.strip()} → "
    )
    try:
        import rag  # deferred: evita ciclo en init
        resp = rag._helper_client().chat(
            model=rag.HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "seed": 42, "num_predict": 32},
            keep_alive=rag.LLM_KEEP_ALIVE,
        )
        raw = (resp.message.content or "").strip()
    except Exception as exc:
        try:
            rag._silent_log("infer_observation_category_failed", exc)
        except Exception:
            pass
        return None

    def _store_cache(value: str | None) -> str | None:
        """Helper: persiste resultado en cache LRU con FIFO eviction."""
        with _OBS_CATEGORY_CACHE_LOCK:
            if len(_OBS_CATEGORY_CACHE) >= _OBS_CATEGORY_CACHE_MAX:
                # FIFO: drop la entrada más vieja (CPython 3.7+ dict preserva insertion order)
                try:
                    oldest = next(iter(_OBS_CATEGORY_CACHE))
                    del _OBS_CATEGORY_CACHE[oldest]
                except StopIteration:
                    pass
            _OBS_CATEGORY_CACHE[cache_key] = value
        return value

    # Parseo tolerante: el modelo a veces devuelve "Categoría: Notas" o
    # "→ Notas" o la palabra con un sufijo ("Notas (default)"). Tomamos
    # la primera línea no vacía y matcheamos contra candidates.
    first_line = next(
        (ln.strip() for ln in raw.splitlines() if ln.strip()),
        "",
    )
    if not first_line:
        return _store_cache(None)

    # Limpieza común (prefijos tipo "→", "-", "*", "Categoría:").
    cleaned = re.sub(
        r"^[\-*→>]+\s*|^\s*categor[íi]a\s*:\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip().strip("`'\"")

    lower = cleaned.lower()
    if lower in {"(ninguna)", "ninguna", "none", "null"}:
        return _store_cache(None)

    # Match exacto (case-insensitive) con cualquier candidate.
    for cand in candidates:
        if cand.lower() == lower:
            return _store_cache(cand)
    # Match por prefijo — cubrir "Notas (default)" → "Notas".
    for cand in candidates:
        if lower.startswith(cand.lower()):
            return _store_cache(cand)
    # Si el LLM devolvió algo libre fuera de la lista, lo aceptamos igual
    # SIEMPRE que sea corto (< 40 chars) y no tenga chars raros. Permite
    # que el LLM invente categorías útiles ("Salud", "Hobbies", etc).
    if 2 <= len(cleaned) <= 40 and re.match(r"^[\w\s/áéíóúñÁÉÍÓÚÑ]+$", cleaned):
        return _store_cache(cleaned)
    return _store_cache(None)


def _append_contact_observation(
    contact_name: str,
    observation: str,
    *,
    category: str | None = None,
    source_kind: str = "manual",   # "manual" | "chat" | "wa" | "audio"
    source_excerpt: str | None = None,
) -> dict:
    """Anotar una observación sobre un contacto en su nota del vault.

    Args:
        contact_name: Nombre tal como el user/LLM lo dice ("Seba", "mi
            Mama", "Sebastian"). Resuelto vía `_lookup_vault_contact`.
        observation: Texto procesado de la observación, listo para
            mostrarse. Ej: "Le gusta el vino", "Preferencia de bebidas:
            Vino", "Trabajo nuevo: cafetería en Costanera".
        category: OPCIONAL — categoría que matchea un bullet existente
            del template (`Trabajo / contexto`, `Notas`, `Preferencias`,
            etc.). Si está dado, appendea al bullet existente o crea uno
            nuevo. Si None, solo va a `## Observaciones`.
        source_kind: De dónde vino la observación. "chat" | "wa" |
            "audio" | "manual". Default "manual" para CLI.
        source_excerpt: OPCIONAL — texto crudo del mensaje original
            ("Seba me llevo un vino"). Se loguea junto a la obs.

    Returns:
        `{ok: bool, file: str, observation_added: bool, category_updated:
        bool, reason?: str}`. `ok=False` con `reason` cuando el contacto
        no existe en el vault o no se puede escribir el archivo.

    Idempotencia: si el `observation` ya está literal en la sección
    `## Observaciones`, no lo agregamos de vuelta. El smart-append a la
    categoría es similar.
    """
    # Re-resolve via package namespace so monkeypatches en
    # `rag.integrations.whatsapp._lookup_vault_contact` propagan al call site.
    from rag.integrations.whatsapp import (
        _load_vault_contacts,
        _lookup_vault_contact,
        _normalize_hint,
        _vault_contacts_dir,
    )
    if not contact_name or not contact_name.strip():
        return {"ok": False, "reason": "empty_contact"}
    if not observation or not observation.strip():
        return {"ok": False, "reason": "empty_observation"}

    vault_match = _lookup_vault_contact(contact_name)
    if not vault_match:
        return {"ok": False, "reason": "contact_not_in_vault",
                "contact_name": contact_name}

    base = _vault_contacts_dir()
    if not base:
        return {"ok": False, "reason": "no_vault"}

    # Re-load para obtener el path real (lookup devuelve solo `parsed`).
    contacts = _load_vault_contacts()
    target_path: Path | None = None
    target_stem = ""
    for c in contacts:
        # Match por full_name o aliases o filename.
        norm_full = _normalize_hint(c["parsed"].get("full_name", ""))
        norm_match_full = _normalize_hint(vault_match.get("full_name", ""))
        if norm_full and norm_full == norm_match_full:
            target_path = c["path"]
            target_stem = c["stem"]
            break
    if not target_path:
        return {"ok": False, "reason": "contact_path_not_found"}

    try:
        text = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "reason": f"read_failed: {str(exc)[:80]}"}

    obs_clean = observation.strip()
    excerpt_clean = (source_excerpt or "").strip()
    today = datetime.now().strftime("%Y-%m-%d")

    # Task #4 (2026-04-26): si el caller no dio `category`, intentamos
    # inferirla con el LLM barato mirando los bullets existentes de la
    # nota + categorías estándar del template. Silent-fail: si el LLM
    # rebota, `category` queda None y la observation va solo a la
    # sección `## Observaciones` (comportamiento pre-fix).
    inferred_category = False
    if category is None:
        try:
            inferred = _infer_observation_category(obs_clean, note_body=text)
        except Exception:
            inferred = None
        if inferred:
            category = inferred
            inferred_category = True

    # Render del bullet para la sección "## Observaciones".
    if excerpt_clean and excerpt_clean.lower() != obs_clean.lower():
        new_bullet = f"- {today} · {obs_clean} _(orig: \"{excerpt_clean}\")_"
    else:
        new_bullet = f"- {today} · {obs_clean}"

    # 1. Sección "## Observaciones" — append idempotente.
    new_text = text
    obs_idx = new_text.find(_OBSERVATIONS_HEADING)
    if obs_idx == -1:
        # No existe la sección — agregarla al final (con linebreak previo
        # si la nota no termina en \n).
        if not new_text.endswith("\n"):
            new_text += "\n"
        if not new_text.endswith("\n\n"):
            new_text += "\n"
        new_text += f"{_OBSERVATIONS_HEADING}\n\n{new_bullet}\n"
        observation_added = True
    else:
        # Idempotencia: skip si la observación literal ya está en la
        # sección. Comparamos el "core" (sin la fecha) para que dos obs
        # del mismo día con texto idéntico no dupliquen.
        existing_section = new_text[obs_idx:]
        # Substring check del observation en el cuerpo de la sección.
        if obs_clean and obs_clean in existing_section:
            observation_added = False
        else:
            # Insertar después del último bullet de la sección. Buscamos
            # el siguiente heading "##" o EOF.
            section_end = new_text.find("\n## ", obs_idx + len(_OBSERVATIONS_HEADING))
            if section_end == -1:
                # Sección hasta EOF.
                if not new_text.endswith("\n"):
                    new_text += "\n"
                new_text += f"{new_bullet}\n"
            else:
                new_text = (
                    new_text[:section_end]
                    + f"\n{new_bullet}"
                    + new_text[section_end:]
                )
            observation_added = True

    # 2. Smart-append al bullet de categoría si está dada.
    category_updated = False
    if category:
        cat_clean = category.strip()
        # Buscar bullet existente `- **<category>**: <value>`.
        pattern = re.compile(
            r"^(-\s*\*\*\s*"
            + re.escape(cat_clean)
            + r"\s*\*\*\s*:\s*)(.*)$",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pattern.search(new_text)
        if m:
            old_value = m.group(2).strip()
            # Idempotencia: si el observation ya está en el value, skip.
            if obs_clean.lower() not in old_value.lower():
                if old_value:
                    new_value = f"{old_value}, {obs_clean}"
                else:
                    new_value = obs_clean
                new_text = (
                    new_text[:m.start()]
                    + m.group(1)
                    + new_value
                    + new_text[m.end():]
                )
                category_updated = True
        else:
            # No existe el bullet — crear uno nuevo. Lo insertamos
            # después del último bullet de propiedades existente
            # (`- **<X>**: ...`), antes de la sección "## Observaciones"
            # o de cualquier otra sección.
            last_bullet_re = re.compile(
                r"^-\s*\*\*[^*]+\*\*\s*:.*$",
                re.MULTILINE,
            )
            last_bullet_m = None
            for last_bullet_m in last_bullet_re.finditer(new_text):
                pass
            new_bullet_str = f"- **{cat_clean}**: {obs_clean}"
            if last_bullet_m:
                insert_at = last_bullet_m.end()
                new_text = (
                    new_text[:insert_at]
                    + f"\n{new_bullet_str}"
                    + new_text[insert_at:]
                )
            else:
                # Sin bullets previos — append al inicio del cuerpo.
                new_text = f"{new_bullet_str}\n{new_text}"
            category_updated = True

    # Si nada cambió (idempotencia full hit), avisamos sin escribir.
    if not observation_added and not category_updated:
        return {
            "ok": True,
            "file": str(target_path.relative_to(base.parent.parent.parent)),
            "observation_added": False,
            "category_updated": False,
            "reason": "duplicate_skipped",
        }

    try:
        target_path.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "reason": f"write_failed: {str(exc)[:80]}"}

    # Invalidar cache del vault para que el próximo lookup vea la nota
    # actualizada (especialmente relevante si el caller es el chat que
    # corre en el mismo proceso). Reseteamos la variable global del módulo
    # `contacts` (donde vive el cache) — es módulo hermano, lo modificamos
    # via attribute access.
    try:
        from rag.integrations.whatsapp import contacts as _contacts_mod
        _contacts_mod._VAULT_CONTACTS_CACHE = None
    except Exception:
        pass

    return {
        "ok": True,
        "file": str(target_path.relative_to(base.parent.parent.parent)),
        "stem": target_stem,
        "observation_added": observation_added,
        "category_updated": category_updated,
        "source_kind": source_kind,
        # Señal para el caller que la categoría fue inferida por el LLM
        # (vs pasada explícitamente). Útil para logging/auditoría — el CLI
        # imprime "categoría inferida por LLM: X" en lugar de solo "X".
        "category": category,
        "category_inferred": inferred_category,
    }


__all__ = [
    "_infer_observation_category",
    "_append_contact_observation",
]
