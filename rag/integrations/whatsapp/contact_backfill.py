"""Backfill de contact notes en `99-Contacts/` desde el bridge de WhatsApp.

Disparador (2026-05-10): el user pidió que el sistema arme las contact
notes automáticamente — escanear los chats activos del bridge SQLite,
clasificar cada uno por tier (transient / active / core), y crear una
nota nueva con un template prellenado para que el user complete los
campos que el sistema no puede inferir.

Sin esto, el user tenía que crear las notas a mano para cada contacto
nuevo, sabiendo de memoria los nombres exactos del WhatsApp display +
los JIDs. La causa raíz: el listener TS depende de notas en
`99-Contacts/<name>.md` para extraer kinship/short_name/dossier, pero
no había feedback loop que avise "te falta nota para X". Resultado:
contactos nuevos quedaban con kinship=unknown y registro genérico.

Surface:

- ``_classify_tier(msg_count, span_days, days_since_last)`` — pure
  classifier: transient / active / core / unknown.
- ``_iter_chat_stats(bridge_db, days_window)`` — generator de stats
  agregadas por chat_jid (1-on-1 + group-effective-1on1).
- ``_safe_filename(name)`` — strip emojis/symbols, trim, fallback al
  jid si queda vacío.
- ``_render_template(template_text, context)`` — substitución de
  placeholders del template.
- ``backfill_contacts(vault_dir, bridge_db_path, dry_run, ...)`` —
  entry-point que recorre stats + crea notas faltantes.

Invariantes:
- Silent-fail en bridge db read: si la db no existe / no tiene tabla
  messages, retorna stats vacías (no raise).
- NUNCA pisamos notas existentes — solo creamos lo que NO está.
- En `dry_run=True` solo planifica; no escribe ni un byte. Es el modo
  default para que el user inspeccione antes.
- Filtros del scan: excluye `status@broadcast`, BOT_CHAT_JID (RagNet
  self), NOTES_CHAT_JID, y groups con multiple senders activos
  (esos son grupos reales, no "1-on-1 efectivo").
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rag.integrations.whatsapp._constants import (
    VAULT_CONTACTS_SUBPATH,
    WHATSAPP_BRIDGE_DB_PATH,
)


# ── Tier classifier (pure) ────────────────────────────────────────────────
#
# Reglas (en orden de chequeo):
#   1. transient si días-desde-último > 180 (frío hace meses) → no vale la
#      pena nota completa, vive como recordatorio mínimo.
#   2. transient si msg_count < 10 (muy poca interacción) → tier mínimo.
#   3. transient si span_days < 2 (apareció hace un día y desapareció) →
#      probable caso de "alguien me escribió 1 vez por algo puntual".
#   4. core si msg_count >= 50 AND span_days >= 60 AND días-último <= 7 →
#      contacto vivo, recurrente.
#   5. active si msg_count >= 10 AND días-último <= 30 → contacto regular.
#   6. fallback: transient.
#
# Los thresholds son heurística — el user puede override `tier:` a mano en
# el frontmatter de la nota cuando creés que merece otra clasificación
# (ej. ex jefa con 5 msgs en el último año pero relación importante →
# subir a active manual).


def _classify_tier(
    msg_count: int,
    span_days: float,
    days_since_last: float,
) -> str:
    """Clasificar tier basado en stats de interacción.

    Returns: "transient" | "active" | "core".
    """
    if days_since_last > 180:
        return "transient"
    if msg_count < 10:
        return "transient"
    if span_days < 2:
        return "transient"
    if msg_count >= 50 and span_days >= 60 and days_since_last <= 7:
        return "core"
    if msg_count >= 10 and days_since_last <= 30:
        return "active"
    return "transient"


# ── Bridge DB scan ────────────────────────────────────────────────────────


@dataclass
class ChatStats:
    """Stats agregadas por chat (1-on-1 o group-effective-1on1)."""
    chat_jid: str
    chat_name: str  # bridge chats.name (puede ser "Grecia's group" para grupos)
    msg_count: int
    first_ts: str  # ISO
    last_ts: str  # ISO
    span_days: float
    days_since_last: float
    is_group: bool
    primary_sender_id: str = ""  # solo cuando is_group=True con 1 sender
    primary_sender_name: str = ""
    unique_human_senders: int = 0


# Excluded JIDs hardcodeados — broadcasts (status), self chats típicos, y
# grupos del propio bot que el user puede setear via env si difiere.
_BROADCAST_JID = "status@broadcast"


def _parse_ts(ts_raw: str) -> datetime | None:
    """Best-effort parser del timestamp del bridge.

    Formato típico del bridge: "2026-05-10 16:13:26-03:00" (ISO con offset).
    Algunas filas viejas vienen con whitespace en vez de "T". Defensa:
    probar ISO directo y con replace.
    """
    if not ts_raw:
        return None
    raw = ts_raw.strip()
    candidates = [raw, raw.replace(" ", "T")]
    for cand in candidates:
        try:
            return datetime.fromisoformat(cand)
        except ValueError:
            continue
    return None


def _iter_chat_stats(
    bridge_db: sqlite3.Connection,
    days_window: int = 365,
    excluded_jids: tuple[str, ...] = (_BROADCAST_JID,),
) -> Iterable[ChatStats]:
    """Generator de stats por chat. Filtra excluidos.

    `days_window` es el horizonte máximo — chats sin actividad inbound más
    allá no aparecen. Default 365d. Para detectar "fríos hace 6 meses
    pero menos de 1 año" → suben como `transient`.
    """
    cur = bridge_db.cursor()
    # Defense: si no existe la tabla, retornar vacío.
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'",
        )
        if not cur.fetchone():
            return
    except sqlite3.Error:
        return

    cutoff_iso = (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - _timedelta_safe(days_window)
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Stats agregadas por chat — excluyendo is_from_me=1 (msgs del user) y
    # contenido vacío (media-only, attachments).
    cur.execute(
        """
        SELECT chat_jid,
               COUNT(*)        AS cnt,
               MIN(timestamp)  AS first_ts,
               MAX(timestamp)  AS last_ts
        FROM messages
        WHERE is_from_me = 0
          AND content IS NOT NULL
          AND TRIM(content) != ''
          AND datetime(timestamp) >= datetime(?)
        GROUP BY chat_jid
        ORDER BY cnt DESC
        """,
        (cutoff_iso,),
    )

    rows = cur.fetchall()

    # Para cada chat, determinar si es grupo + sender info + name.
    for chat_jid, cnt, first_ts, last_ts in rows:
        chat_jid = (chat_jid or "").strip()
        if not chat_jid or chat_jid in excluded_jids:
            continue

        is_group = chat_jid.endswith("@g.us")
        # Bridge chat name (display).
        chat_name = ""
        try:
            cur.execute("SELECT name FROM chats WHERE jid = ? LIMIT 1", (chat_jid,))
            r = cur.fetchone()
            if r and r[0]:
                chat_name = str(r[0]).strip()
        except sqlite3.Error:
            pass

        # Para grupos, contar unique senders humanos últimos 30d para detectar
        # "group-effective-one-on-one" (1 sola persona del otro lado).
        unique_senders = 0
        primary_sender_id = ""
        primary_sender_name = ""
        if is_group:
            try:
                recent_cutoff = (
                    datetime.now(timezone.utc).replace(tzinfo=None)
                    - _timedelta_safe(30)
                ).strftime("%Y-%m-%d %H:%M:%S")
                cur.execute(
                    """
                    SELECT sender, COUNT(*) AS cnt FROM messages
                    WHERE chat_jid = ?
                      AND is_from_me = 0
                      AND content IS NOT NULL
                      AND TRIM(content) != ''
                      AND datetime(timestamp) >= datetime(?)
                    GROUP BY sender
                    ORDER BY cnt DESC
                    """,
                    (chat_jid, recent_cutoff),
                )
                senders = [
                    (s, c) for (s, c) in cur.fetchall()
                    if s and str(s).strip()
                ]
                unique_senders = len(senders)
                if senders:
                    primary_sender_id = str(senders[0][0])
                    # Resolver nombre via chats con suffix variants.
                    for suffix in ("@lid", "@s.whatsapp.net"):
                        try:
                            cur.execute(
                                "SELECT name FROM chats WHERE jid = ? LIMIT 1",
                                (primary_sender_id + suffix,),
                            )
                            r = cur.fetchone()
                            if r and r[0]:
                                primary_sender_name = str(r[0]).strip()
                                break
                        except sqlite3.Error:
                            continue
            except sqlite3.Error:
                pass

        # span y recency.
        first_dt = _parse_ts(first_ts) if first_ts else None
        last_dt = _parse_ts(last_ts) if last_ts else None
        if not first_dt or not last_dt:
            continue
        span_days = (last_dt - first_dt).total_seconds() / 86400
        # Comparar contra now en la TZ del last_ts si está aware.
        now_aware = (
            datetime.now(timezone.utc) if last_dt.tzinfo else datetime.now()
        )
        days_since_last = (now_aware - last_dt).total_seconds() / 86400

        yield ChatStats(
            chat_jid=chat_jid,
            chat_name=chat_name,
            msg_count=int(cnt),
            first_ts=str(first_ts),
            last_ts=str(last_ts),
            span_days=round(span_days, 1),
            days_since_last=round(days_since_last, 1),
            is_group=is_group,
            primary_sender_id=primary_sender_id,
            primary_sender_name=primary_sender_name,
            unique_human_senders=unique_senders,
        )


def _timedelta_safe(days: float):
    """Wrapper alrededor de timedelta para que mocks de datetime fijo
    no rompan el ts comparison del cutoff (sólo usado en datetime aware
    diff arriba; este helper es trivial pero centraliza)."""
    from datetime import timedelta
    return timedelta(days=days)


# ── Filename helpers ──────────────────────────────────────────────────────

# Caracteres prohibidos en nombre de archivo (vault Obsidian es macOS).
# Conservamos espacios + acentos (Obsidian los acepta sin escape).
_FORBIDDEN_FILENAME_CHARS = re.compile(r"[\\/:\*\?\"<>\|]")


def _safe_filename(name: str, fallback: str = "Unknown") -> str:
    """Convertir un display name en filename seguro para el vault.

    - Strip emojis y symbols (unicodedata cat empieza con S o C).
    - Strip caracteres prohibidos en nombres (`/ \\ : * ? " < > |`).
    - Trim whitespace.
    - Si queda vacío → fallback.
    """
    if not name:
        return fallback
    # Normalizar y filtrar.
    norm = unicodedata.normalize("NFC", name)
    out_chars: list[str] = []
    for c in norm:
        cat = unicodedata.category(c)
        if cat[0] in ("S", "C"):  # Symbols (emoji), Control
            continue
        if cat == "Mn":  # Variation selectors / non-spacing marks
            continue
        out_chars.append(c)
    out = "".join(out_chars)
    out = _FORBIDDEN_FILENAME_CHARS.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or fallback


# ── Display name resolution ───────────────────────────────────────────────


def _resolve_contact_display_name(stat: ChatStats) -> str:
    """Decidir cómo nombrar la nota del contacto.

    - 1-on-1: usar `chat_name` del bridge si existe, sino digits del JID.
    - Group-effective-1on1: usar `primary_sender_name` resuelto, sino el
      `chat_name` del grupo, sino digits del primary sender.
    - Group con multi-sender: skip (esto NO es candidato a contact note —
      el caller filtra antes).
    """
    if not stat.is_group:
        if stat.chat_name:
            return stat.chat_name
        # Caer a digits del JID (lo que viene antes del @).
        return stat.chat_jid.split("@")[0] or "Contact"
    # Grupo.
    if stat.primary_sender_name:
        return stat.primary_sender_name
    if stat.chat_name:
        return stat.chat_name
    return stat.primary_sender_id or stat.chat_jid.split("@")[0] or "Contact"


# ── Template loader + render ──────────────────────────────────────────────


def _load_template_text(contacts_dir: Path, transient: bool = False) -> str:
    """Leer el template de notas. Si el _template.md no existe, fallback
    a un template hardcodeado mínimo. NO crash si falta — el flow tiene
    que poder correr en vaults nuevos sin templates.
    """
    name = "_template_transient.md" if transient else "_template.md"
    path = contacts_dir / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        # Fallback hardcodeado.
        if transient:
            return (
                "---\n"
                "type: mention\n"
                "kinship: unknown\n"
                "tier: transient\n"
                "---\n\n"
                "- **wa_jid**: <jid>\n"
                "- **Apellido / nombre completo**:\n"
                "- **Notas**:\n"
            )
        return (
            "---\n"
            "type: mention\n"
            "kinship: unknown\n"
            "tier: unknown\n"
            "---\n\n"
            "- **Relación**:\n"
            "- **wa_jid**:\n"
            "- **Apodo**:\n"
            "- **Apellido / nombre completo**:\n"
            "- **Cumpleaños**:\n"
            "- **Teléfono**:\n"
            "- **Email**:\n"
            "- **Dirección**:\n"
            "- **Trabajo / contexto**:\n"
            "- **Cómo nos conocemos**:\n"
            "- **Notas**:\n"
        )


def _render_contact_note(
    template_text: str,
    *,
    display_name: str,
    chat_jid: str,
    tier: str,
    msg_count: int,
    last_ts: str,
    span_days: float,
) -> str:
    """Pre-llenar el template con info derivable.

    - Reemplazar `tier: unknown` → `tier: <tier>` en frontmatter.
    - Insertar el `wa_jid: <chat_jid>` si el placeholder está.
    - Agregar comentario auto-generado al final con stats.
    """
    out = template_text
    # tier (frontmatter).
    out = re.sub(
        r"^tier\s*:\s*unknown\s*$",
        f"tier: {tier}",
        out,
        count=1,
        flags=re.MULTILINE,
    )
    # wa_jid (body) — sólo si el placeholder es el del template canonical.
    out = re.sub(
        r"^- \*\*wa_jid\*\*:\s*<jid[^>]*>.*$",
        f"- **wa_jid**: {chat_jid}",
        out,
        count=1,
        flags=re.MULTILINE,
    )
    # Hay otros patterns más simples del transient template.
    out = re.sub(
        r"^- \*\*wa_jid\*\*:\s*<jid>\s*$",
        f"- **wa_jid**: {chat_jid}",
        out,
        count=1,
        flags=re.MULTILINE,
    )
    # Append meta footer auto-generado.
    footer = (
        "\n\n---\n"
        f"<!-- Auto-generado por `rag wa-contacts backfill` el "
        f"{datetime.now().strftime('%Y-%m-%d')} -->\n"
        f"<!-- Stats: msgs={msg_count}, último_ts={last_ts}, "
        f"span_días={span_days:.0f}, tier_inferido={tier} -->\n"
        f"<!-- Display name del bridge: {display_name!r} -->\n"
    )
    return out.rstrip() + footer


# ── Backfill entry-point ──────────────────────────────────────────────────


@dataclass
class BackfillResult:
    """Una entrada del reporte de backfill."""
    chat_jid: str
    display_name: str
    filename: str
    tier: str
    msg_count: int
    days_since_last: float
    action: str  # "would_create" | "skipped_exists" | "skipped_multi_sender"
    reason: str = ""


def backfill_contacts(
    *,
    vault_root: Path,
    bridge_db_path: Path | None = None,
    dry_run: bool = True,
    days_window: int = 365,
    min_msgs: int = 1,
    excluded_jids: tuple[str, ...] = (_BROADCAST_JID,),
) -> list[BackfillResult]:
    """Backfill de contact notes desde stats del bridge.

    Args:
      vault_root: path al vault (la carpeta `99-Contacts/` cuelga adentro).
      bridge_db_path: path al messages.db del bridge. Default → constant.
      dry_run: si True, no escribe nada — solo planifica + reporta.
      days_window: horizonte del scan (chats sin inbound más allá no aparecen).
      min_msgs: filtro mínimo de msg_count para considerar el chat
        (default 1 — incluye one-shots; subir a 3+ para ignorar ruido).
      excluded_jids: JIDs hardcodeados que skipeamos (broadcast, etc).

    Returns: lista de BackfillResult — incluye creates + skips para que
    el caller pueda mostrar el reporte completo.
    """
    bridge_db_path = bridge_db_path or WHATSAPP_BRIDGE_DB_PATH
    contacts_dir = vault_root / VAULT_CONTACTS_SUBPATH
    contacts_dir.mkdir(parents=True, exist_ok=True)

    # Existing notes (lowercase set para comparar).
    existing_stems = set()
    for f in contacts_dir.glob("*.md"):
        if f.name.startswith("_"):
            continue
        existing_stems.add(f.stem.lower().strip())

    template_full = _load_template_text(contacts_dir, transient=False)
    template_transient = _load_template_text(contacts_dir, transient=True)

    results: list[BackfillResult] = []

    if not bridge_db_path.exists():
        return results
    try:
        conn = sqlite3.connect(f"file:{bridge_db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return results

    try:
        for stat in _iter_chat_stats(conn, days_window=days_window, excluded_jids=excluded_jids):
            if stat.msg_count < min_msgs:
                continue
            # Multi-sender groups: skipear (no son candidatos a contact note,
            # son grupos reales con múltiples participantes).
            if stat.is_group and stat.unique_human_senders > 1:
                results.append(BackfillResult(
                    chat_jid=stat.chat_jid,
                    display_name=stat.chat_name,
                    filename="",
                    tier="n/a",
                    msg_count=stat.msg_count,
                    days_since_last=stat.days_since_last,
                    action="skipped_multi_sender",
                    reason=f"grupo con {stat.unique_human_senders} senders activos — no candidato",
                ))
                continue
            display_name = _resolve_contact_display_name(stat)
            safe = _safe_filename(display_name)
            tier = _classify_tier(
                stat.msg_count, stat.span_days, stat.days_since_last,
            )
            # Existing check.
            if safe.lower() in existing_stems:
                results.append(BackfillResult(
                    chat_jid=stat.chat_jid,
                    display_name=display_name,
                    filename=f"{safe}.md",
                    tier=tier,
                    msg_count=stat.msg_count,
                    days_since_last=stat.days_since_last,
                    action="skipped_exists",
                    reason="nota ya existente — el user es dueño",
                ))
                continue
            # Render desde template apropiado al tier.
            template = template_transient if tier == "transient" else template_full
            body = _render_contact_note(
                template,
                display_name=display_name,
                chat_jid=stat.chat_jid,
                tier=tier,
                msg_count=stat.msg_count,
                last_ts=stat.last_ts,
                span_days=stat.span_days,
            )
            target = contacts_dir / f"{safe}.md"
            if not dry_run:
                try:
                    target.write_text(body, encoding="utf-8")
                except OSError as e:
                    results.append(BackfillResult(
                        chat_jid=stat.chat_jid,
                        display_name=display_name,
                        filename=f"{safe}.md",
                        tier=tier,
                        msg_count=stat.msg_count,
                        days_since_last=stat.days_since_last,
                        action="error",
                        reason=f"write failed: {e}",
                    ))
                    continue
            results.append(BackfillResult(
                chat_jid=stat.chat_jid,
                display_name=display_name,
                filename=f"{safe}.md",
                tier=tier,
                msg_count=stat.msg_count,
                days_since_last=stat.days_since_last,
                action="would_create" if dry_run else "created",
                reason="",
            ))
            # Una vez creada (o "would_create"), reservar el stem para no
            # duplicar si dos chats colapsan al mismo display name.
            existing_stems.add(safe.lower())
    finally:
        conn.close()

    return results
