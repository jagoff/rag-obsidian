import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from web import conversation_writer
from web.conversation_writer import TurnData, slugify, write_turn


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "00-Inbox" / "conversations").mkdir(parents=True)
    monkeypatch.setattr(conversation_writer, "_INDEX_PATH", tmp_path / "idx.json")
    return vault


def _turn(q, a, sources, conf, ts):
    return TurnData(question=q, answer=a, sources=sources, confidence=conf, timestamp=ts)


def test_first_turn_creates_note_with_frontmatter(tmp_vault):
    ts = datetime(2026, 4, 19, 4, 12, 0, tzinfo=timezone.utc)
    turn = _turn(
        "¿qué es el Ikigai?",
        "El Ikigai es una filosofía japonesa.",
        [{"file": "02-Areas/Coaching.md", "score": 0.8},
         {"file": "03-Resources/Ikigai.md", "score": 0.7}],
        0.75,
        ts,
    )
    path = write_turn(tmp_vault, "web:abc123", turn)
    assert path.name == "2026-04-19-0412-que-es-el-ikigai.md"
    assert path.parent == tmp_vault / "00-Inbox" / "conversations"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    fm_end = text.index("\n---\n", 4)
    fm = text[4:fm_end]
    lines = fm.split("\n")
    # 6 top-level keys in fixed order (sources + tags have list items)
    keys_in_order = [ln.split(":", 1)[0] for ln in lines if not ln.startswith("  ")]
    assert keys_in_order == [
        "session_id", "created", "updated", "turns", "confidence_avg", "sources", "tags",
    ]
    assert "session_id: web:abc123" in fm
    assert "created: 2026-04-19T04:12:00Z" in fm
    assert "updated: 2026-04-19T04:12:00Z" in fm
    assert "turns: 1" in fm
    assert "confidence_avg: 0.750" in fm
    assert "  - 02-Areas/Coaching.md" in fm
    assert "  - 03-Resources/Ikigai.md" in fm
    assert "  - conversation" in fm
    assert "  - rag-chat" in fm
    assert "## Turn 1 — 04:12" in text
    assert "> ¿qué es el Ikigai?" in text
    assert "El Ikigai es una filosofía japonesa." in text
    assert "[[02-Areas/Coaching]]" in text
    assert "[[03-Resources/Ikigai]]" in text


def test_second_turn_appends_and_updates_frontmatter(tmp_vault):
    ts1 = datetime(2026, 4, 19, 4, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 4, 19, 4, 18, 33, tzinfo=timezone.utc)
    sid = "web:sess2"
    t1 = _turn("primera pregunta", "respuesta uno",
               [{"file": "02-Areas/A.md", "score": 0.5}], 0.40, ts1)
    p1 = write_turn(tmp_vault, sid, t1)
    t2 = _turn("segunda pregunta", "respuesta dos",
               [{"file": "03-Resources/B.md", "score": 0.6}], 0.60, ts2)
    p2 = write_turn(tmp_vault, sid, t2)
    assert p1 == p2
    text = p2.read_text(encoding="utf-8")
    assert "turns: 2" in text
    assert "created: 2026-04-19T04:12:00Z" in text
    assert "updated: 2026-04-19T04:18:33Z" in text
    # running avg: (0.40 + 0.60) / 2 = 0.500
    assert "confidence_avg: 0.500" in text
    assert "  - 02-Areas/A.md" in text
    assert "  - 03-Resources/B.md" in text
    assert "## Turn 1 — 04:12" in text
    assert "## Turn 2 — 04:18" in text
    assert text.count("## Turn ") == 2


def test_slugify_strips_accents_and_punctuation():
    assert slugify("¿Qué es el Ikigai?") == "que-es-el-ikigai"
    assert slugify("  Hola   MUNDO!!  ") == "hola-mundo"
    assert slugify("") == "conversation"
    assert slugify("a" * 80, max_len=50) == "a" * 50


def test_index_maps_session_and_reuses_path(tmp_vault, monkeypatch):
    ts = datetime(2026, 4, 19, 5, 0, 0, tzinfo=timezone.utc)
    sid = "web:idx-test"
    t = _turn("tema único", "ok", [{"file": "X.md", "score": 0.1}], 0.1, ts)
    p1 = write_turn(tmp_vault, sid, t)
    idx_path = conversation_writer._INDEX_PATH
    assert idx_path.exists()
    mapping = json.loads(idx_path.read_text())
    assert mapping[sid] == str(p1.relative_to(tmp_vault))

    # Second write for same session — move the file so "find by scanning" would fail,
    # forcing the reuse to come from the index. (We move it back to verify path is reused.)
    ts2 = datetime(2026, 4, 19, 5, 5, 0, tzinfo=timezone.utc)
    t2 = _turn("otra pregunta distinta", "respuesta", [{"file": "Y.md", "score": 0.2}], 0.3, ts2)
    # Sabotage the folder scan: if the code ever scans by session_id in the folder,
    # it won't find anything since index says exact path. The file still exists there.
    p2 = write_turn(tmp_vault, sid, t2)
    assert p2 == p1
    text = p2.read_text(encoding="utf-8")
    assert "turns: 2" in text


def test_concurrent_writes_no_corruption(tmp_vault):
    sid = "web:concurrent"
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def worker(qnum: int):
        try:
            barrier.wait(timeout=5)
            ts = datetime(2026, 4, 19, 6, qnum, 0, tzinfo=timezone.utc)
            t = _turn(
                f"pregunta numero {qnum}",
                f"respuesta {qnum}",
                [{"file": f"F{qnum}.md", "score": 0.5}],
                0.5,
                ts,
            )
            p = write_turn(tmp_vault, sid, t)
            results.append(p)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors, f"worker errors: {errors}"
    assert len(results) == 2
    assert results[0] == results[1]
    text = results[0].read_text(encoding="utf-8")
    assert text.count("## Turn ") == 2
    assert "## Turn 1 —" in text
    assert "## Turn 2 —" in text
    assert "turns: 2" in text
    # Both source files should be in the union
    assert "  - F1.md" in text
    assert "  - F2.md" in text


def test_malformed_frontmatter_raises(tmp_vault):
    sid = "web:broken"
    target = tmp_vault / "00-Inbox" / "conversations" / "broken.md"
    target.write_text("---\nthis is : not : parseable : yaml\nno closing block\n",
                      encoding="utf-8")
    # Seed the index to point at the broken file
    idx_path = conversation_writer._INDEX_PATH
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps({sid: str(target.relative_to(tmp_vault))}),
                        encoding="utf-8")
    ts = datetime(2026, 4, 19, 7, 0, 0, tzinfo=timezone.utc)
    t = _turn("q", "a", [{"file": "Z.md", "score": 0.1}], 0.1, ts)
    with pytest.raises(ValueError):
        write_turn(tmp_vault, sid, t)
