"""Tests for the `watch()` debounce/handler machinery.

The previous code-review-followup item #2 flagged the watchdog `Handler`
class + debounce loop as untested — specifically the three edge cases:

  1. rapid file saves during the debounce window (dedup via set)
  2. observer exception recovery (one broken file mustn't abort the batch)
  3. observer.stop() cleanup under load (finally-block guarantees)

The core filter + drain logic is now extracted into `_watch_filter_path`
and `_watch_drain_once`. Tests exercise those directly without spawning
a real watchdog Observer — driving a tree of real inotify events is
flaky in CI and adds 2-3s of observer startup per test.

The CLI body itself (Observer.start() + time.sleep loop) stays
integration-tested through production use; tests here cover the pure
logic it now delegates to.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

import rag


# ── _watch_filter_path ────────────────────────────────────────────────────────


def test_filter_accepts_md_inside_vault(tmp_path):
    """Happy path: .md file under the vault, no exclude match → returns path."""
    vault = tmp_path
    note = vault / "sub" / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text("# hi", encoding="utf-8")

    out = rag._watch_filter_path(str(note), vault.resolve(), ())
    assert out == note


def test_filter_rejects_non_md(tmp_path):
    """Non-.md files — images, attachments — are dropped early."""
    vault = tmp_path
    attachment = vault / "image.png"
    attachment.write_bytes(b"PNG")
    assert rag._watch_filter_path(str(attachment), vault.resolve(), ()) is None


def test_filter_rejects_outside_vault(tmp_path):
    """A .md file whose resolve() lies outside the vault is rejected.
    Mirrors the original handler's `try/except ValueError: return`."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "other" / "note.md"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    assert rag._watch_filter_path(str(outside), vault.resolve(), ()) is None


def test_filter_rejects_excluded_folder(tmp_path):
    """Files under configured exclude folders are dropped. Regression guard
    for the WhatsApp-vault-sync churn storm (default exclude covers
    03-Resources/WhatsApp)."""
    vault = tmp_path
    wa = vault / "03-Resources" / "WhatsApp" / "chat" / "2026-04.md"
    wa.parent.mkdir(parents=True)
    wa.write_text("", encoding="utf-8")

    out = rag._watch_filter_path(
        str(wa), vault.resolve(), ("03-Resources/WhatsApp",)
    )
    assert out is None


def test_filter_exclude_matches_prefix_only(tmp_path):
    """`03-Resources/WhatsAppNotes/x.md` must NOT match exclude `03-Resources/WhatsApp`
    — the separator check prevents prefix-partial matches."""
    vault = tmp_path
    sibling = vault / "03-Resources" / "WhatsAppNotes" / "x.md"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("", encoding="utf-8")

    out = rag._watch_filter_path(
        str(sibling), vault.resolve(), ("03-Resources/WhatsApp",)
    )
    assert out == sibling


def test_filter_exclude_matches_folder_itself(tmp_path):
    """The exclude folder name as an exact relative path is rejected too.
    Handles corner case where watchdog emits a dir-delete event on the
    folder itself (though typically is_directory filters those upstream)."""
    vault = tmp_path
    # Simulate a .md that has the exact exclude path as its relative form.
    # We achieve this by making the exclude folder name end in `.md` —
    # unusual but not invalid, and we want the equality branch covered.
    f = vault / "weird.md"
    f.write_text("", encoding="utf-8")
    # `rel_str == folder` branch.
    assert rag._watch_filter_path(str(f), vault.resolve(), ("weird.md",)) is None


def test_filter_tolerates_nonexistent_path(tmp_path):
    """watchdog on_deleted fires with the now-gone path. resolve() on a
    missing file is fine on POSIX (returns the absolute form), but we
    still need the relative_to check to not raise."""
    vault = tmp_path
    ghost = vault / "deleted.md"
    # File never existed; ensure filter still returns a Path (deletes must
    # be forwarded to _index_single_file so it can drop the stale vectors).
    out = rag._watch_filter_path(str(ghost), vault.resolve(), ())
    assert out == ghost


# ── _watch_drain_once ─────────────────────────────────────────────────────────


@pytest.fixture
def vstate(tmp_path):
    """Minimal vstate dict — no real sqlite-vec collection. The drain function
    passes `col` straight through to `_index_single_file`, which we patch."""
    return {
        "name": "main",
        "path": tmp_path.resolve(),
        "col": object(),  # opaque — _index_single_file is monkeypatched
        "pending": set(),
    }


def test_drain_empty_pending_returns_empty_list(vstate):
    """No pending → no work, no errors, no side effects."""
    out = rag._watch_drain_once(vstate, threading.Lock())
    assert out == []


def test_drain_returns_status_per_file(vstate, tmp_path, monkeypatch):
    """Each queued file produces one tuple. `status` is propagated verbatim
    from `_index_single_file` so the caller can colourise output."""
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for p in (a, b, c):
        p.write_text("", encoding="utf-8")
    vstate["pending"] = {a, b, c}

    returns = {a: "indexed", b: "skipped", c: "removed"}

    def fake_index(col, path, *, vault_path):
        assert col is vstate["col"]
        assert vault_path == vstate["path"]
        return returns[path]

    monkeypatch.setattr(rag, "_index_single_file", fake_index)

    out = rag._watch_drain_once(vstate, threading.Lock())
    assert len(out) == 3
    statuses = {name for _, name, _, _ in out}
    assert statuses == {"indexed", "skipped", "removed"}
    # Pending set is drained.
    assert vstate["pending"] == set()
    # All err slots are None on the success path.
    assert all(err is None for _, _, _, err in out)


def test_drain_continues_after_per_file_error(vstate, tmp_path, monkeypatch):
    """Followup doc #2: the key race test — one file raising must NOT abort
    the drain. Every other file in the batch still gets processed."""
    good1 = tmp_path / "good1.md"
    bad = tmp_path / "bad.md"
    good2 = tmp_path / "good2.md"
    for p in (good1, bad, good2):
        p.write_text("", encoding="utf-8")
    vstate["pending"] = {good1, bad, good2}

    def fake_index(col, path, *, vault_path):
        if path == bad:
            raise RuntimeError("corrupt frontmatter")
        return "indexed"

    monkeypatch.setattr(rag, "_index_single_file", fake_index)

    out = rag._watch_drain_once(vstate, threading.Lock())
    # All three reported.
    assert len(out) == 3
    bad_row = next(row for row in out if row[2].name == "bad.md")
    good_rows = [row for row in out if row[2].name != "bad.md"]
    assert bad_row[1] is None  # status = None when raised
    assert "corrupt frontmatter" in bad_row[3]
    assert all(row[1] == "indexed" and row[3] is None for row in good_rows)
    # Pending still drained.
    assert vstate["pending"] == set()


def test_drain_reports_vault_relative_path(vstate, tmp_path, monkeypatch):
    """The rel-path in the tuple is vault-relative when possible so the
    caller can render short names (`sub/note.md`) instead of absolute
    paths in multi-vault watch output."""
    note = tmp_path / "sub" / "note.md"
    note.parent.mkdir()
    note.write_text("", encoding="utf-8")
    vstate["pending"] = {note}

    monkeypatch.setattr(rag, "_index_single_file", lambda *a, **kw: "indexed")

    [(_, _, rel, _)] = rag._watch_drain_once(vstate, threading.Lock())
    assert rel == Path("sub/note.md")


def test_drain_falls_back_to_absolute_when_not_vault_relative(
    vstate, tmp_path, monkeypatch
):
    """If the queued path somehow isn't under the vault (e.g. a symlinked
    worktree edge case), the drain still emits a tuple — it falls back
    to the original absolute Path rather than crashing."""
    outside = tmp_path.parent / "other-vault" / "note.md"
    outside.parent.mkdir(exist_ok=True)
    outside.write_text("", encoding="utf-8")
    vstate["pending"] = {outside}

    monkeypatch.setattr(rag, "_index_single_file", lambda *a, **kw: "indexed")

    [(_, status, rel, err)] = rag._watch_drain_once(vstate, threading.Lock())
    assert status == "indexed"
    assert err is None
    assert rel == outside


def test_drain_handles_rapid_saves_via_set_dedup(vstate, tmp_path, monkeypatch):
    """Simulate 50 rapid saves of the same 3 files during the debounce
    window — handler dedups via `set.add`, drain sees each file once."""
    files = [tmp_path / f"n{i}.md" for i in range(3)]
    for f in files:
        f.write_text("", encoding="utf-8")

    lock = threading.Lock()
    # Simulate the handler adding each file 20x (as rapid modifications).
    for _ in range(20):
        with lock:
            for f in files:
                vstate["pending"].add(f)

    seen: list[Path] = []

    def fake_index(col, path, *, vault_path):
        seen.append(path)
        return "indexed"

    monkeypatch.setattr(rag, "_index_single_file", fake_index)

    out = rag._watch_drain_once(vstate, lock)
    # Each file indexed exactly once despite 20 queue operations.
    assert len(out) == 3
    assert sorted(p.name for p in seen) == ["n0.md", "n1.md", "n2.md"]


def test_drain_is_thread_safe_against_concurrent_add(
    vstate, tmp_path, monkeypatch
):
    """Handler thread keeps adding to `pending` while the drain thread is
    running — `pending_lock` must serialise the snapshot so the drain
    doesn't miss additions or double-process existing ones.

    We drain twice; everything queued before drain N must land in drain
    N's output or drain N+1's output — never lost, never duplicated.
    """
    lock = threading.Lock()
    indexed: list[Path] = []

    def fake_index(col, path, *, vault_path):
        indexed.append(path)
        return "indexed"

    monkeypatch.setattr(rag, "_index_single_file", fake_index)

    stop = threading.Event()
    total_queued: list[Path] = []

    def producer():
        i = 0
        while not stop.is_set():
            p = tmp_path / f"q{i}.md"
            p.write_text("", encoding="utf-8")
            with lock:
                vstate["pending"].add(p)
            total_queued.append(p)
            i += 1

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    # Let the producer queue a batch, then drain twice with a gap.
    import time as _t
    _t.sleep(0.05)
    first = rag._watch_drain_once(vstate, lock)
    _t.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)
    second = rag._watch_drain_once(vstate, lock)

    total_indexed = [row[2] for row in first + second]
    # Each queued file processed exactly once.
    assert len(set(total_indexed)) == len(total_indexed)
    # No queued file is lost (every producer-emitted path appears exactly
    # once in one of the drain outputs).
    total_indexed_set = {p.resolve() for p in indexed}
    total_queued_set = {p.resolve() for p in total_queued}
    # Queued-after-second-drain-ended would still be in pending; we set
    # `stop` before the second drain so producer had stopped. Check the
    # two sets match.
    assert total_queued_set <= total_indexed_set
    assert vstate["pending"] == set()
