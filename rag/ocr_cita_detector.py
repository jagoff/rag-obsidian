"""OCR → cita intent detector — extracted from rag/ocr.py 2026-05-09.

Classifies OCR text (from Apple Vision via ``_ocr_image`` or VLM caption
fallback) into one of three kinds via the qwen2.5:3b helper LLM:

  - ``event``: cita, turno, reunión, cumple, vuelo — date/time-anchored
    item that goes to the calendar. Triggers ``propose_calendar_event``.
  - ``reminder``: tarea, to-do, factura a pagar, lista de compras — action
    item with or without deadline. Triggers ``propose_reminder``.
  - ``note``: info without action — receta médica sin fecha, foto de
    código, meme, captura de UI. No-op (already in vault).

Triple dedup safeguard:

  1. Sidecar ``rag_cita_detections`` keyed by SHA256(normalized_ocr)[:16].
  2. ``_find_duplicate_calendar_event`` inside ``propose_calendar_event``.
  3. Confidence floor (``_CITA_MIN_CONFIDENCE``) discards borderline cases.

Silent-fail at every step (helper timeout, JSON malformed, sqlite lock,
osascript error) — the indexer keeps running without new citas.

Rollback: ``RAG_CITA_DETECT=0`` short-circuits everything.

## Deferred imports / monkey-patch propagation

Helpers from ``rag/__init__.py`` (``_helper_client``, ``_silent_log``,
``_ragvec_state_conn``, ``propose_calendar_event``, ``propose_reminder``)
are imported INSIDE each function body so test monkey-patches
(``monkeypatch.setattr(rag, "_helper_client", stub)``) propagate at
call-time. Same for ``_detect_cita_from_ocr`` itself: callers in this
module re-resolve via ``import rag as _rag`` so tests can patch
``_rag._detect_cita_from_ocr`` and have ``_maybe_create_cita_from_ocr``
see the patch.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

__all__ = [
    "_CITA_MIN_CONFIDENCE",
    "_CITA_MIN_CHARS",
    "_CITA_VALID_KINDS",
    "_DETECTOR_TIMEOUT",
    "_CITA_PROMPT_SYSTEM",
    "_CITA_PROMPT_USER_TEMPLATE",
    "_cita_detect_enabled",
    "_normalize_ocr_for_hash",
    "_ocr_hash_key",
    "_detect_cita_from_ocr",
    "_maybe_create_cita_from_ocr",
    "_cita_result",
    "_persist_cita_detection",
]

# Umbral de confianza auto-create. qwen2.5:3b con temp=0 + seed=42 devuelve
# scores consistentes; 0.70 filtra ambigüedades pero pasa casos reales
# ("turno dentista miércoles 15hs consultorio Palermo" → 0.9+). Override
# por `rag scan-citas --min-confidence 0.5` para barridos agresivos.
_CITA_MIN_CONFIDENCE = 0.70

# OCR más corto que esto no tiene suficiente señal (ej. screenshot de un
# botón con "OK"). Skipeamos sin gastar helper call.
_CITA_MIN_CHARS = 20

# Kinds válidos del detector. `note` = no-op (solo se loggea). El resto
# dispara acción.
_CITA_VALID_KINDS = frozenset({"event", "reminder", "note"})

# Audit 2026-04-25 R2-OCR #1: timeout explícito del detector LLM (qwen2.5:3b).
# Sin esto, una llamada colgada bloquea el endpoint /api/chat/upload-image
# por minutos. El detector va a través de `rag._helper_client()`, que ya
# fija este mismo valor en `_TimedOllamaProxy(timeout=60.0)` —
# `_DETECTOR_TIMEOUT` documenta el contrato del lado consumidor (ocr.py)
# y permite testearlo sin importar implementación.
_DETECTOR_TIMEOUT: float = 60.0


def _cita_detect_enabled() -> bool:
    """True salvo `RAG_CITA_DETECT=0/false/no` explícito. Default ON."""
    val = os.environ.get("RAG_CITA_DETECT", "").strip().lower()
    return val not in ("0", "false", "no")


def _normalize_ocr_for_hash(ocr_text: str) -> str:
    """Lowercase + whitespace-collapsed para que dos OCR passes sobre la
    misma imagen colisionen en el hash aunque ocrmac produzca orden de
    palabras distinto entre runs (raro, pero posible con tablas).
    """
    return " ".join((ocr_text or "").lower().split())


def _ocr_hash_key(ocr_text: str) -> str:
    """SHA256 (primeros 16 hex chars) del texto normalizado. PRIMARY KEY de
    `rag_cita_detections`. 16 chars = 64 bits, colisión accidental ~irreal
    para la cardinalidad esperada (≤ miles de imágenes por user).
    """
    norm = _normalize_ocr_for_hash(ocr_text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


_CITA_PROMPT_SYSTEM = (
    "Sos un clasificador de tareas agendables a partir de texto OCR "
    "crudo. El texto viene de una imagen — puede tener errores de "
    "reconocimiento (acentos perdidos, palabras cortadas, líneas "
    "mezcladas, chrome de app). Tu trabajo es decidir si el texto "
    "describe un EVENTO de calendario, un RECORDATORIO/tarea, o "
    "simplemente INFORMACIÓN sin acción agendable. Y extraer los datos."
)

_CITA_PROMPT_USER_TEMPLATE = (
    "TEXTO OCR (puede contener ruido):\n"
    "---\n"
    "{ocr}\n"
    "---\n\n"
    "Devolvé EXACTAMENTE un objeto JSON con estas keys:\n"
    "  - kind: \"event\" | \"reminder\" | \"note\"\n"
    "      * event: tiene fecha/hora específica (cita, turno, vuelo, "
    "reunión, cumple, evento programado).\n"
    "      * reminder: tarea o acción a hacer, con o sin fecha "
    "(comprar X, llamar a Y, pagar factura, leer artículo).\n"
    "      * note: info sin acción agendable (receta médica vieja, "
    "snippet de código, captura de chrome de app, meme, texto "
    "informativo sin call-to-action).\n"
    "  - title: string corto descriptivo (≤ 100 chars). Para event: "
    "qué evento. Para reminder: qué hay que hacer. Para note: tema.\n"
    "  - when: string (ISO 8601 si podés, o lenguaje natural si no — "
    "\"miércoles 15hs\", \"mañana 10am\", \"\" si no hay fecha).\n"
    "  - location: string (lugar físico/virtual, \"\" si no hay).\n"
    "  - confidence: float 0.0-1.0 — qué tan seguro estás de la "
    "clasificación. < 0.5 si dudás, > 0.8 si es claro.\n\n"
    "Si el texto OCR está vacío o es solo símbolos/números sin "
    "contexto, devolvé `kind=\"note\"` con `confidence=0.0`.\n\n"
    "Respondé SOLO el JSON, sin texto antes ni después."
)


def _detect_cita_from_ocr(ocr_text: str) -> dict | None:
    """Helper call qwen2.5:3b con format=json: clasifica el texto OCR como
    event, reminder o note, y extrae {title, when, location}.

    Returns:
      - dict con shape `{kind, title, when, location, confidence}` —
        normalizado y validado. NUNCA raise.
      - None si: `RAG_CITA_DETECT=0`, ocr_text vacío o muy corto, helper
        timeout / unreachable, JSON malformado, shape inválida.

    Callers deben chequear `None` + `kind in {event, reminder}` +
    `confidence >= threshold` antes de crear algo — el helper a veces
    devuelve `kind='event'` con `confidence=0.3`, lo cual es una
    clasificación dudosa que no queremos auto-agendar.

    Backward-compat con el schema viejo `{is_cita, start}`: si el modelo
    (o un test monkeypatched) devuelve las keys viejas, las mapeamos:
    `is_cita=True` → `kind='event'`, `is_cita=False` → `kind='note'`,
    `start` → `when`. Eso hace que el upgrade sea transparente para
    callers externos.
    """
    from rag import _helper_client, _silent_log, HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE
    if not _cita_detect_enabled():
        return None
    text = (ocr_text or "").strip()
    if len(text) < _CITA_MIN_CHARS:
        return None
    # Cap para controlar el prompt size — 1500 chars cubre el 99% de
    # screenshots reales. Cortar el final es OK: el título / fecha suele
    # estar al principio del OCR (Apple Vision scanea top→bottom).
    capped = text[:1500]
    prompt = _CITA_PROMPT_USER_TEMPLATE.format(ocr=capped)
    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[
                {"role": "system", "content": _CITA_PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={**HELPER_OPTIONS, "num_predict": 180, "num_ctx": 2048},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = resp.message.content.strip()
        data = json.loads(raw)
    except Exception as exc:
        _silent_log("cita_detect_helper", exc)
        return None
    if not isinstance(data, dict):
        return None
    # Normalización defensiva — el helper a veces devuelve tipos raros.
    try:
        # Backward-compat: schema viejo usaba `is_cita` + `start`.
        kind_raw = data.get("kind")
        if kind_raw is None and "is_cita" in data:
            kind_raw = "event" if data.get("is_cita") else "note"
        kind = str(kind_raw or "note").strip().lower()
        if kind not in _CITA_VALID_KINDS:
            kind = "note"

        title = str(data.get("title") or "").strip()[:120]
        when_raw = data.get("when")
        if when_raw is None and "start" in data:
            when_raw = data.get("start")
        when = str(when_raw or "").strip()[:200]
        location = str(data.get("location") or "").strip()[:200]

        conf_raw = data.get("confidence")
        if isinstance(conf_raw, str):
            try:
                conf_raw = float(conf_raw)
            except ValueError:
                conf_raw = 0.0
        confidence = float(conf_raw or 0.0)
        confidence = max(0.0, min(1.0, confidence))
    except Exception as exc:
        _silent_log("cita_detect_normalize", exc)
        return None
    return {
        "kind": kind,
        "title": title,
        "when": when,
        "location": location,
        "confidence": confidence,
    }


def _maybe_create_cita_from_ocr(
    ocr_text: str,
    image_path: Path,
    source: str,
    *,
    min_confidence: float | None = None,
) -> dict | None:
    """Pipeline OCR → classifier → action (event / reminder / note) con
    sidecar dedup. Silent-fail en cada paso.

    Flujo por kind:
      - `kind="event"` + when parseable → `propose_calendar_event`. Persist
        decision="cita" con `event_uid` (o "duplicate" si ya estaba, o
        "error", o "ambiguous" si el parser de fecha no lo resolvió).
      - `kind="event"` + when="" → persist "ambiguous" sin crear.
      - `kind="reminder"` → `propose_reminder` (con o sin fecha). El path
        de la imagen se incluye en `notes` para que quede referenciado
        en Apple Reminders (no hay attachment API). Persist "reminder" +
        `reminder_id`.
      - `kind="note"` → no-op. Persist "note" para que re-runs skippeen.

    `source` arg se propaga al sidecar para auditoría ("index" / "capture"
    / "scan-citas" / "whatsapp"). `min_confidence` override local cuando
    el caller es `rag scan-citas --min-confidence 0.5`.

    Returns dict con shape unificada:
      `{cached: bool, decision: str, kind: str, title, when, location,
        confidence, event_uid, reminder_id}`.

    NUNCA raise — cada paso tiene try/except silent-log.
    """
    from rag import _ragvec_state_conn, _silent_log, propose_calendar_event, propose_reminder
    if not _cita_detect_enabled():
        return None
    text = (ocr_text or "").strip()
    if len(text) < _CITA_MIN_CHARS:
        return None
    threshold = (
        float(min_confidence) if min_confidence is not None else _CITA_MIN_CONFIDENCE
    )
    key = _ocr_hash_key(text)
    img_str = str(image_path) if image_path else ""

    # Step 1: dedup lookup. Si ya lo procesamos antes, short-circuit.
    try:
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT decision, kind, title, start_text, location, "
                "confidence, event_uid, reminder_id, created_at "
                "FROM rag_cita_detections WHERE ocr_hash = ?",
                (key,),
            ).fetchone()
    except Exception as exc:
        _silent_log("cita_sidecar_read", exc)
        row = None
    if row is not None:
        return {
            "cached": True,
            "decision": row[0],
            "kind": row[1] or "",
            "title": row[2],
            "when": row[3],
            # Backward-compat alias para callers que esperan `start`.
            "start": row[3],
            "location": row[4],
            "confidence": row[5],
            "event_uid": row[6],
            "reminder_id": row[7],
            "created_at": row[8],
        }

    # Step 2: run detector. Resolve via `rag` so tests can
    # monkeypatch.setattr(rag, "_detect_cita_from_ocr", fake) and have it
    # propagate here.
    import rag as _rag
    detected = _rag._detect_cita_from_ocr(text)
    if detected is None:
        # Helper unavailable / malformed — NO persist. Reintentamos next run.
        return None

    now_ts = time.time()
    kind = detected.get("kind") or "note"
    title = detected.get("title") or ""
    when = detected.get("when") or ""
    location = detected.get("location") or ""
    confidence = float(detected.get("confidence") or 0.0)

    # Step 3: below-threshold — persist so dedup wins next time.
    if confidence < threshold:
        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision="low_confidence", kind=kind,
            title=title, start_text=when, location=location,
            confidence=confidence, event_uid=None, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision="low_confidence", kind=kind,
            title=title, when=when, location=location,
            confidence=confidence,
        )

    # Step 4: route by kind.
    if kind == "note":
        # Nothing to schedule. Persist so we don't re-invoke the helper.
        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision="note", kind="note",
            title=title, start_text=when, location=location,
            confidence=confidence, event_uid=None, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision="note", kind="note",
            title=title, when=when, location=location,
            confidence=confidence,
        )

    if kind == "event":
        if not when:
            # Event classification without a parseable date — persist as
            # ambiguous (user can re-evaluate manually, and dedup won't
            # re-call helper for the same OCR text).
            _persist_cita_detection(
                ocr_hash=key, image_path=img_str, source=source,
                decision="ambiguous", kind="event",
                title=title, start_text=when, location=location,
                confidence=confidence, event_uid=None, reminder_id=None,
                created_at=now_ts,
            )
            return _cita_result(
                cached=False, decision="ambiguous", kind="event",
                title=title, when=when, location=location,
                confidence=confidence,
            )
        event_title = title or "Cita"
        notes_blob = f"Auto-detectado de OCR ({source}): {image_path}\n\n{text[:500]}"
        try:
            result_json = propose_calendar_event(
                title=event_title, start=when,
                location=(location or None), notes=notes_blob,
            )
            result = json.loads(result_json)
        except Exception as exc:
            _silent_log("cita_propose_event", exc)
            result = {"created": False, "error": str(exc)}

        event_uid = None
        decision = "error"
        if isinstance(result, dict):
            if result.get("duplicate"):
                decision = "duplicate"
                existing = result.get("existing") or {}
                event_uid = existing.get("uid") or existing.get("event_uid")
            elif result.get("created"):
                decision = "cita"
                event_uid = result.get("event_uid")
            elif result.get("needs_clarification"):
                decision = "ambiguous"
            else:
                decision = "error"

        _persist_cita_detection(
            ocr_hash=key, image_path=img_str, source=source,
            decision=decision, kind="event",
            title=event_title, start_text=when, location=location,
            confidence=confidence, event_uid=event_uid, reminder_id=None,
            created_at=now_ts,
        )
        return _cita_result(
            cached=False, decision=decision, kind="event",
            title=event_title, when=when, location=location,
            confidence=confidence, event_uid=event_uid,
        )

    # kind == "reminder"
    reminder_title = title or "Tarea"
    # Apple Reminders NO soporta attachments vía AppleScript; el path de
    # la imagen queda en el body del reminder para referencia — Reminders.
    # app lo muestra como texto, grep-friendly desde CLI también.
    notes_blob = (
        f"Imagen: {image_path}\n"
        f"Origen: {source}\n\n"
        f"{text[:500]}"
    )
    try:
        result_json = propose_reminder(
            title=reminder_title,
            when=when,  # puede ser "" — `propose_reminder` lo tolera
            notes=notes_blob,
        )
        result = json.loads(result_json)
    except Exception as exc:
        _silent_log("cita_propose_reminder", exc)
        result = {"created": False, "error": str(exc)}

    reminder_id = None
    decision = "error"
    if isinstance(result, dict):
        if result.get("created"):
            decision = "reminder"
            reminder_id = result.get("reminder_id")
        elif result.get("needs_clarification"):
            decision = "ambiguous"
        else:
            decision = "error"

    _persist_cita_detection(
        ocr_hash=key, image_path=img_str, source=source,
        decision=decision, kind="reminder",
        title=reminder_title, start_text=when, location=location,
        confidence=confidence, event_uid=None, reminder_id=reminder_id,
        created_at=now_ts,
    )
    return _cita_result(
        cached=False, decision=decision, kind="reminder",
        title=reminder_title, when=when, location=location,
        confidence=confidence, reminder_id=reminder_id,
    )


def _cita_result(
    *, cached: bool, decision: str, kind: str, title: str, when: str,
    location: str, confidence: float,
    event_uid: str | None = None, reminder_id: str | None = None,
) -> dict:
    """Shape uniforme de retorno para `_maybe_create_cita_from_ocr`.
    Incluye `start` como alias de `when` para backward-compat con callers
    que esperan el schema viejo (tests pre-2026-04-23 tarde, renders del
    CLI anteriores al routing por kind).
    """
    return {
        "cached": bool(cached),
        "decision": decision,
        "kind": kind,
        "title": title,
        "when": when,
        "start": when,  # alias
        "location": location,
        "confidence": float(confidence),
        "event_uid": event_uid,
        "reminder_id": reminder_id,
    }


def _persist_cita_detection(
    *, ocr_hash: str, image_path: str, source: str, decision: str,
    kind: str | None, title: str, start_text: str, location: str,
    confidence: float,
    event_uid: str | None, reminder_id: str | None, created_at: float,
) -> None:
    """INSERT OR IGNORE en `rag_cita_detections`.

    Si otro caller ganó la carrera (mismo `ocr_hash` persistido primero),
    respetamos su fila — por eso OR IGNORE y no OR REPLACE. Silent-fail:
    excepciones log-only, no bloquean el caller.

    Incluye columnas `kind` + `reminder_id` post-2026-04-23. Instalaciones
    viejas sin la migration lazy (`_migrate_cita_detections_add_kind`) van
    a fallar el INSERT por columnas inexistentes; el except genérico lo
    captura y persistimos con el subset mínimo (best-effort degradation).
    """
    from rag import _ragvec_state_conn, _silent_log
    try:
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO rag_cita_detections "
                "(ocr_hash, image_path, source, decision, kind, title, "
                "start_text, location, confidence, event_uid, reminder_id, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ocr_hash, image_path, source, decision, kind, title,
                    start_text, location, float(confidence), event_uid,
                    reminder_id, float(created_at),
                ),
            )
    except Exception as exc:
        _silent_log(f"cita_sidecar_write:{ocr_hash}", exc)
        # Fallback: retry con subset pre-kind/reminder_id (pre-migration
        # schema). Protege operadores que corrieron el feature original
        # (commit 1d55b27) y NO bajaron `_migrate_cita_detections_add_kind`
        # todavía (ej. tests que instancian un DB bare sin
        # `_ensure_telemetry_tables`).
        try:
            with _ragvec_state_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO rag_cita_detections "
                    "(ocr_hash, image_path, source, decision, title, "
                    "start_text, location, confidence, event_uid, "
                    "created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ocr_hash, image_path, source, decision, title,
                        start_text, location, float(confidence),
                        event_uid, float(created_at),
                    ),
                )
        except Exception as exc2:
            _silent_log(f"cita_sidecar_write_fallback:{ocr_hash}", exc2)
