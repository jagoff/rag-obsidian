#!/usr/bin/env python3
"""Backfill engine: transcribe voice notes inbound de WhatsApp en batch.

Game-changer #G1 (2026-05-11). Sin este daemon, el feature de voice
transcripts (F1, 2026-05-11) solo se activa cuando el user clickea
"📝 ver transcript" en `/wa`. Como la mayoría de los audios nunca se
clickean, el corpus pierde el ~30% del contenido real de los chats.

Este script:
  1. Scans `messages.db.messages` últimos N días, `media_type='audio'`.
  2. Filtra los que NO tienen row en `rag_wa_voice_transcripts` (cache).
  3. Por cada uno, descarga el media via bridge (best-effort), transcribe
     con whisper MLX (`transcribe_audio`, que ya tiene su cache propio),
     persiste en cache + escribe nota al vault bajo
     `99-AI/external-ingest/whatsapp-voice/<contact-slug>/`.
  4. Cap en `--limit` (default 20) por run para evitar runaways y dar
     CPU a otras cosas — corre cada 15min via supervisor.

Uso:
  rag wa-voice-backfill                  # default: 7 días, 20 audios
  scripts/wa_voice_backfill.py --days 30 --limit 100 --dry-run

Telemetría:
  - stdout: "[backfill] processed N, transcribed M, errors E, skipped S"
  - silent_log: cada error individual.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# Permite invocación standalone desde `scripts/` sin PYTHONPATH externo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bridge_db_path() -> Path:
    import rag as _rag  # noqa: PLC0415
    return _rag.WHATSAPP_DB_PATH


def _bot_jid() -> str:
    import rag as _rag  # noqa: PLC0415
    return _rag.WHATSAPP_BOT_JID


def _select_pending(days: int, limit: int) -> list[dict]:
    """Devuelve audios inbound últimos N días sin transcript cacheado.

    Cross-DB query: bridge.messages LEFT JOIN telemetry.rag_wa_voice_transcripts.
    Filtramos `is_from_me=0` (solo inbound — los outbound son propios y
    no aportan al corpus tanto).
    """
    from rag import DB_PATH, _TELEMETRY_DB_FILENAME  # noqa: PLC0415

    bridge = _bridge_db_path()
    if not bridge.is_file():
        print("[backfill] bridge db no existe — skip", flush=True)
        return []
    telemetry = Path(DB_PATH) / _TELEMETRY_DB_FILENAME
    bot = _bot_jid()
    try:
        con = sqlite3.connect(f"file:{telemetry}", uri=True, timeout=10.0)
        con.row_factory = sqlite3.Row
        con.execute(f"ATTACH DATABASE 'file:{bridge}?mode=ro' AS br")
        cutoff = time.time() - days * 86400
        # Bridge stores timestamps como 'YYYY-MM-DD HH:MM:SS-03:00'
        # — convertimos cutoff a string-comparable.
        import datetime as _dt  # noqa: PLC0415
        from rag.integrations.whatsapp.fetch import _BRIDGE_TZ_OFFSET  # noqa: PLC0415
        cutoff_str = _dt.datetime.fromtimestamp(cutoff).strftime(
            "%Y-%m-%d %H:%M:%S"
        ) + _BRIDGE_TZ_OFFSET
        rows = con.execute(
            """
            SELECT m.id, m.chat_jid, m.sender, m.timestamp, m.filename
            FROM br.messages m
            LEFT JOIN main.rag_wa_voice_transcripts t ON t.msg_id = m.id
            WHERE m.media_type = 'audio'
              AND m.is_from_me = 0
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
              AND m.timestamp > ?
              AND m.filename IS NOT NULL AND m.filename != ''
              AND t.msg_id IS NULL
            ORDER BY m.timestamp DESC
            LIMIT ?
            """,
            (bot, cutoff_str, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            con.close()
        except Exception:
            pass


def _transcribe_one(msg: dict) -> tuple[bool, str]:
    """Transcribe un msg específico. Devuelve (ok, info_str).

    Side effects: persiste a `rag_wa_voice_transcripts` + escribe nota
    al vault via `voice_notes.write_voice_note`.
    """
    from rag.integrations.whatsapp import (  # noqa: PLC0415
        _db_local,
        bridge_client as _bc,
        fetch as _wa_fetch,
        voice_notes as _voice_notes,
    )
    from rag.whisper import transcribe_audio  # noqa: PLC0415

    msg_id = msg["id"]
    jid = msg["chat_jid"]
    sender = msg["sender"]
    audio_ts = msg["timestamp"]
    filename = msg["filename"]

    path = _wa_fetch._bridge_media_path(jid, filename)
    if path is None or not path.is_file():
        try:
            _bc.download_media(msg_id, jid)
            path = _wa_fetch._bridge_media_path(jid, filename)
        except _bc.BridgeError as exc:
            err = f"bridge_download_failed: {exc}"
            _db_local.set_voice_transcript(
                msg_id, jid, sender=sender, error=err, audio_ts=audio_ts,
            )
            return False, err
    if path is None or not path.is_file():
        err = "media_not_downloaded"
        _db_local.set_voice_transcript(
            msg_id, jid, sender=sender, error=err, audio_ts=audio_ts,
        )
        return False, err

    try:
        tr = transcribe_audio(path, language="es")
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:200]
        _db_local.set_voice_transcript(
            msg_id, jid, sender=sender, error=err, audio_ts=audio_ts,
        )
        return False, err

    text = (tr or {}).get("text") or ""
    _db_local.set_voice_transcript(
        msg_id, jid,
        sender=sender,
        text=text,
        language=(tr or {}).get("language"),
        duration_s=(tr or {}).get("duration_s"),
        model=(tr or {}).get("model"),
        audio_ts=audio_ts,
    )
    if text.strip():
        try:
            _voice_notes.write_voice_note(
                msg_id=msg_id, jid=jid, sender=sender,
                text=text, audio_ts=audio_ts,
            )
        except Exception as exc:  # noqa: BLE001
            return True, f"transcribed but vault_write_failed: {exc}"
    return True, f"len={len(text)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7,
                    help="Look-back window (default 7).")
    ap.add_argument("--limit", type=int, default=20,
                    help="Cap audios procesados por run (default 20).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo lista lo que haría, sin transcribir.")
    args = ap.parse_args()

    print(
        f"[backfill] start days={args.days} limit={args.limit} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    t0 = time.perf_counter()
    pending = _select_pending(days=args.days, limit=args.limit)
    print(f"[backfill] pending audios: {len(pending)}", flush=True)
    if args.dry_run:
        for m in pending[:10]:
            print(
                f"  would transcribe: {m['id'][:20]}… ts={m['timestamp']} "
                f"jid={m['chat_jid'][:30]}",
                flush=True,
            )
        return 0

    ok_count = 0
    err_count = 0
    for m in pending:
        ok, info = _transcribe_one(m)
        if ok:
            ok_count += 1
        else:
            err_count += 1
        print(
            f"[backfill] {'ok' if ok else 'err'} {m['id'][:20]}… "
            f"jid={m['chat_jid'][:25]} {info}",
            flush=True,
        )

    dt = time.perf_counter() - t0
    print(
        f"[backfill] done · processed={len(pending)} ok={ok_count} "
        f"err={err_count} elapsed={dt:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
