"""Tests para `rag.integrations.whatsapp.contact_backfill`.

Cubre:
- `_classify_tier` — clasificador puro (transient/active/core).
- `_safe_filename` — strip emojis/symbols/forbidden chars.
- `_resolve_contact_display_name` — nombre del contacto.
- `_render_contact_note` — substitución de placeholders del template.
- `backfill_contacts` end-to-end con bridge sqlite in-memory.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag.integrations.whatsapp.contact_backfill import (
    ChatStats,
    _classify_tier,
    _render_contact_note,
    _resolve_contact_display_name,
    _safe_filename,
    backfill_contacts,
)


# ── _classify_tier ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "msg_count, span_days, days_since_last, expected",
    [
        # transient: poca interacción
        (3, 1.0, 0.5, "transient"),       # msg<10
        (5, 30.0, 5.0, "transient"),      # msg<10
        (50, 100.0, 200.0, "transient"),  # frío >180d
        (100, 1.0, 1.0, "transient"),     # span<2d
        (10, 5.0, 100.0, "transient"),    # last>30d, no llega a active
        # active: contacto regular
        (10, 30.0, 5.0, "active"),
        (20, 30.0, 25.0, "active"),
        (49, 59.0, 7.0, "active"),        # casi core pero msg<50
        # core: contacto frecuente
        (50, 60.0, 7.0, "core"),
        (100, 200.0, 1.0, "core"),
        (200, 365.0, 0.5, "core"),
    ],
)
def test_classify_tier(
    msg_count: int, span_days: float, days_since_last: float, expected: str,
) -> None:
    assert _classify_tier(msg_count, span_days, days_since_last) == expected


# ── _safe_filename ────────────────────────────────────────────────────────


def test_safe_filename_strips_emojis() -> None:
    assert _safe_filename("Maria 🩷") == "Maria"
    assert _safe_filename("Su 🇸🇪") == "Su"


def test_safe_filename_strips_forbidden() -> None:
    assert _safe_filename("foo/bar:baz") == "foobarbaz"
    assert _safe_filename('a"b<c>d|e') == "abcde"


def test_safe_filename_preserves_accents_and_dashes() -> None:
    # Acentos sí, dashes sí, paréntesis sí (Obsidian los acepta).
    assert _safe_filename("María Sol") == "María Sol"
    assert _safe_filename("Pablo Ramirez (Seguro)") == "Pablo Ramirez (Seguro)"


def test_safe_filename_collapses_whitespace() -> None:
    assert _safe_filename("  a   b   c  ") == "a b c"


def test_safe_filename_fallback_when_empty() -> None:
    assert _safe_filename("") == "Unknown"
    # Solo emojis → vacío → fallback.
    assert _safe_filename("🩷🩷🩷") == "Unknown"


# ── _resolve_contact_display_name ─────────────────────────────────────────


def _make_stat(**kwargs) -> ChatStats:
    defaults = dict(
        chat_jid="123@s.whatsapp.net",
        chat_name="Test",
        msg_count=10,
        first_ts="2026-01-01 00:00:00",
        last_ts="2026-05-01 00:00:00",
        span_days=120.0,
        days_since_last=5.0,
        is_group=False,
        primary_sender_id="",
        primary_sender_name="",
        unique_human_senders=0,
    )
    defaults.update(kwargs)
    return ChatStats(**defaults)


def test_resolve_display_name_one_on_one() -> None:
    stat = _make_stat(chat_jid="555@s.whatsapp.net", chat_name="Maria")
    assert _resolve_contact_display_name(stat) == "Maria"


def test_resolve_display_name_one_on_one_no_name() -> None:
    stat = _make_stat(chat_jid="555@s.whatsapp.net", chat_name="")
    assert _resolve_contact_display_name(stat) == "555"


def test_resolve_display_name_group_effective_uses_sender() -> None:
    stat = _make_stat(
        chat_jid="abc@g.us",
        chat_name="Grecia's group",
        is_group=True,
        primary_sender_name="Grecia",
        primary_sender_id="888",
        unique_human_senders=1,
    )
    assert _resolve_contact_display_name(stat) == "Grecia"


def test_resolve_display_name_group_no_sender_resolution() -> None:
    """Si no se pudo resolver el nombre del sender, fallback a chat_name."""
    stat = _make_stat(
        chat_jid="abc@g.us",
        chat_name="Some Group",
        is_group=True,
        primary_sender_name="",
        primary_sender_id="888",
        unique_human_senders=1,
    )
    assert _resolve_contact_display_name(stat) == "Some Group"


# ── _render_contact_note ──────────────────────────────────────────────────


def test_render_replaces_tier_in_frontmatter() -> None:
    template = (
        "---\n"
        "type: mention\n"
        "kinship: unknown\n"
        "tier: unknown\n"
        "---\n\n"
        "- **Notas**:\n"
    )
    out = _render_contact_note(
        template,
        display_name="Test",
        chat_jid="555@s.whatsapp.net",
        tier="active",
        msg_count=20,
        last_ts="2026-05-01 00:00:00",
        span_days=60.0,
    )
    assert "tier: active" in out
    assert "tier: unknown" not in out


def test_render_replaces_wa_jid_full_template() -> None:
    template = (
        "---\n"
        "tier: unknown\n"
        "---\n\n"
        "- **wa_jid**: <jid de WhatsApp si lo conocés, ej. xxx>\n"
    )
    out = _render_contact_note(
        template,
        display_name="Test",
        chat_jid="555@s.whatsapp.net",
        tier="active",
        msg_count=10,
        last_ts="t",
        span_days=10.0,
    )
    assert "- **wa_jid**: 555@s.whatsapp.net" in out


def test_render_replaces_wa_jid_transient_template() -> None:
    template = (
        "---\n"
        "tier: unknown\n"
        "---\n\n"
        "- **wa_jid**: <jid>\n"
    )
    out = _render_contact_note(
        template,
        display_name="X",
        chat_jid="999@g.us",
        tier="transient",
        msg_count=3,
        last_ts="t",
        span_days=1.0,
    )
    assert "- **wa_jid**: 999@g.us" in out


def test_render_appends_footer_with_stats() -> None:
    template = "---\ntier: unknown\n---\n\nbody\n"
    out = _render_contact_note(
        template,
        display_name="X",
        chat_jid="j",
        tier="active",
        msg_count=42,
        last_ts="2026-05-01",
        span_days=60.0,
    )
    assert "Auto-generado por `rag wa-contacts backfill`" in out
    assert "msgs=42" in out
    assert "tier_inferido=active" in out


# ── backfill_contacts (end-to-end con bridge in-memory) ───────────────────


def _make_bridge_db() -> Path:
    """Crear bridge db in-memory persistido a disk para uri=ro mode."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (
            id TEXT, chat_jid TEXT, sender TEXT, content TEXT,
            timestamp TIMESTAMP, is_from_me BOOLEAN,
            media_type TEXT, filename TEXT, url TEXT,
            media_key BLOB, file_sha256 BLOB, file_enc_sha256 BLOB,
            file_length INTEGER,
            PRIMARY KEY (id, chat_jid)
        );
        CREATE TABLE chats (
            jid TEXT PRIMARY KEY,
            name TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    return path


def _insert_msg(
    db_path: Path, chat_jid: str, sender: str, content: str,
    ts: datetime, is_from_me: int = 0,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me, media_type) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        (
            f"id-{ts.isoformat()}-{sender}",
            chat_jid,
            sender,
            content,
            ts.strftime("%Y-%m-%d %H:%M:%S-03:00"),
            is_from_me,
        ),
    )
    conn.commit()
    conn.close()


def _set_chat_name(db_path: Path, jid: str, name: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT OR REPLACE INTO chats (jid, name) VALUES (?, ?)", (jid, name))
    conn.commit()
    conn.close()


def test_backfill_dry_run_creates_no_files(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        # 1 contacto activo.
        now = datetime.now()
        for i in range(15):
            _insert_msg(
                bridge,
                "5491100@s.whatsapp.net",
                "5491100",
                f"hola {i}",
                now - timedelta(days=15) + timedelta(hours=i),
            )
        _set_chat_name(bridge, "5491100@s.whatsapp.net", "Maria")

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        # Cwd template.
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n\n- **wa_jid**: <jid>\n",
            encoding="utf-8",
        )

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=True,
        )
        creates = [r for r in results if r.action == "would_create"]
        assert len(creates) == 1
        assert creates[0].display_name == "Maria"
        # NO file written.
        assert not (contacts_dir / "Maria.md").exists()
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_apply_writes_file(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # 15 msgs spread sobre ~14 días — span > 2, msg_count >= 10, last reciente
        # → tier="active".
        for i in range(15):
            _insert_msg(
                bridge,
                "5491100@s.whatsapp.net",
                "5491100",
                f"hola {i}",
                now - timedelta(days=15) + timedelta(days=i),
            )
        _set_chat_name(bridge, "5491100@s.whatsapp.net", "Maria")

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n\n- **wa_jid**: <jid>\n",
            encoding="utf-8",
        )

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=False,
        )
        creates = [r for r in results if r.action == "created"]
        assert len(creates) == 1
        target = contacts_dir / "Maria.md"
        assert target.exists()
        body = target.read_text(encoding="utf-8")
        assert "tier: active" in body  # 15 msgs en 15d, días-último ≈0
        assert "5491100@s.whatsapp.net" in body
        assert "msgs=15" in body
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_skips_existing_notes(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        for i in range(15):
            _insert_msg(
                bridge,
                "5491100@s.whatsapp.net",
                "5491100",
                f"hola {i}",
                now - timedelta(days=10) + timedelta(hours=i),
            )
        _set_chat_name(bridge, "5491100@s.whatsapp.net", "Maria")

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n", encoding="utf-8",
        )
        # Nota existente con contenido.
        existing = contacts_dir / "Maria.md"
        existing.write_text("MANUAL CONTENT — DON'T OVERWRITE", encoding="utf-8")

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=False,
        )
        skipped = [r for r in results if r.action == "skipped_exists"]
        assert len(skipped) == 1
        # Nota intacta.
        assert existing.read_text(encoding="utf-8") == "MANUAL CONTENT — DON'T OVERWRITE"
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_skips_multi_sender_groups(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # Grupo con 2 senders distintos.
        for i in range(10):
            _insert_msg(
                bridge, "g@g.us", "user1", f"msg {i}",
                now - timedelta(days=5) + timedelta(hours=i),
            )
        for i in range(10):
            _insert_msg(
                bridge, "g@g.us", "user2", f"msg {i}",
                now - timedelta(days=4) + timedelta(hours=i),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n", encoding="utf-8",
        )

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=True,
        )
        multi_sender = [r for r in results if r.action == "skipped_multi_sender"]
        assert len(multi_sender) == 1
        creates = [r for r in results if r.action == "would_create"]
        assert len(creates) == 0
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_excludes_status_broadcast(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        for i in range(20):
            _insert_msg(
                bridge, "status@broadcast", "anyone", f"status {i}",
                now - timedelta(days=10) + timedelta(hours=i),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n", encoding="utf-8",
        )

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=True,
        )
        # status@broadcast no debe aparecer.
        assert all(r.chat_jid != "status@broadcast" for r in results)
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_min_msgs_filter(tmp_path: Path) -> None:
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # Contacto con SOLO 2 msgs (< min_msgs=5).
        for i in range(2):
            _insert_msg(
                bridge, "9999@s.whatsapp.net", "9999", f"hi {i}",
                now - timedelta(days=10) + timedelta(hours=i),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: unknown\n---\n", encoding="utf-8",
        )

        results = backfill_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
            dry_run=True,
            min_msgs=5,
        )
        creates = [r for r in results if r.action == "would_create"]
        assert len(creates) == 0
    finally:
        bridge.unlink(missing_ok=True)


def test_backfill_no_bridge_db_returns_empty(tmp_path: Path) -> None:
    """Defense: si la bridge DB no existe, retornar [] sin raise."""
    contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
    contacts_dir.mkdir(parents=True)
    results = backfill_contacts(
        vault_root=tmp_path,
        bridge_db_path=tmp_path / "nonexistent.db",
        dry_run=True,
    )
    assert results == []


# ── find_promotable_contacts ──────────────────────────────────────────────


def test_promote_check_finds_transient_with_recent_activity(tmp_path: Path) -> None:
    """Caso típico: nota tier=transient, pero el bridge muestra
    actividad que la cualifica como active → candidato a promote."""
    from rag.integrations.whatsapp.contact_backfill import find_promotable_contacts
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # 20 msgs spread sobre 14 días — califica como active.
        for i in range(20):
            _insert_msg(
                bridge,
                "5491100@s.whatsapp.net",
                "5491100",
                f"hola {i}",
                now - timedelta(days=14) + timedelta(days=i * 0.7),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        # Nota existente con tier=transient y wa_jid populado.
        (contacts_dir / "Maria.md").write_text(
            "---\ntype: mention\ntier: transient\n---\n\n"
            "- **wa_jid**: 5491100@s.whatsapp.net\n"
            "- **Apellido / nombre completo**: Maria\n",
            encoding="utf-8",
        )

        candidates = find_promotable_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
        )
        assert len(candidates) == 1
        c = candidates[0]
        assert c.display_name == "Maria"
        assert c.current_tier == "transient"
        assert c.new_tier == "active"
        assert c.msg_count == 20
    finally:
        bridge.unlink(missing_ok=True)


def test_promote_check_skips_aligned_tiers(tmp_path: Path) -> None:
    """Si tier_actual matchea tier_calculado → no candidato."""
    from rag.integrations.whatsapp.contact_backfill import find_promotable_contacts
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # Nota tier=transient + bridge activity también transient (3 msgs).
        for i in range(3):
            _insert_msg(
                bridge, "111@s.whatsapp.net", "111", f"hi {i}",
                now - timedelta(days=10) + timedelta(hours=i),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "Random.md").write_text(
            "---\ntier: transient\n---\n\n"
            "- **wa_jid**: 111@s.whatsapp.net\n",
            encoding="utf-8",
        )

        candidates = find_promotable_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
        )
        assert candidates == []
    finally:
        bridge.unlink(missing_ok=True)


def test_promote_check_unknown_tier_with_active_qualifies(tmp_path: Path) -> None:
    """Notas con tier=unknown (creadas a mano sin frontmatter) también
    suben si la actividad lo justifica."""
    from rag.integrations.whatsapp.contact_backfill import find_promotable_contacts
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        # 60 msgs sobre 65d, últimos hace <1d → cumple core
        # (msg≥50 AND span≥60 AND last≤7).
        for i in range(60):
            _insert_msg(
                bridge, "5491200@s.whatsapp.net", "5491200", f"msg {i}",
                now - timedelta(days=65) + timedelta(days=i * (65 / 60)),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        # Nota sin frontmatter explícito (tier="" → "unknown").
        (contacts_dir / "Cristian.md").write_text(
            "[[Cristian|@Cristian]]\n"
            "- **Relación**: amigo\n"
            "- **wa_jid**: 5491200@s.whatsapp.net\n",
            encoding="utf-8",
        )

        candidates = find_promotable_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
        )
        assert len(candidates) == 1
        assert candidates[0].new_tier == "core"
        assert candidates[0].current_tier == "unknown"
    finally:
        bridge.unlink(missing_ok=True)


def test_promote_check_skips_notes_without_wa_jid(tmp_path: Path) -> None:
    """Notas sin `wa_jid` no se evalúan — no podemos mapear al bridge."""
    from rag.integrations.whatsapp.contact_backfill import find_promotable_contacts
    bridge = _make_bridge_db()
    try:
        now = datetime.now()
        for i in range(20):
            _insert_msg(
                bridge, "999@s.whatsapp.net", "999", f"x {i}",
                now - timedelta(days=10) + timedelta(hours=i * 12),
            )

        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "Mystery.md").write_text(
            "---\ntier: transient\n---\n\n- **Notas**: sin jid\n",
            encoding="utf-8",
        )

        candidates = find_promotable_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
        )
        assert candidates == []
    finally:
        bridge.unlink(missing_ok=True)


def test_promote_check_skips_template_files(tmp_path: Path) -> None:
    """Archivos `_template*.md` se ignoran."""
    from rag.integrations.whatsapp.contact_backfill import find_promotable_contacts
    bridge = _make_bridge_db()
    try:
        contacts_dir = tmp_path / "99-obsidian" / "99-Contacts"
        contacts_dir.mkdir(parents=True)
        (contacts_dir / "_template.md").write_text(
            "---\ntier: transient\n---\n- **wa_jid**: 999@s.whatsapp.net\n",
            encoding="utf-8",
        )
        candidates = find_promotable_contacts(
            vault_root=tmp_path,
            bridge_db_path=bridge,
        )
        assert candidates == []
    finally:
        bridge.unlink(missing_ok=True)
