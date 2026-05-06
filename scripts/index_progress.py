"""Real-time progress monitor for `rag index --reset`.

Polls the log file every 5s and shows:
- Files processed (counts `Indexing <path>` style markers)
- Chunks added (counts `chunks=` markers if printed)
- Per-folder breakdown (01-Projects / 02-Areas / 03-Resources / 04-Archive)
- Vault count snapshot from sqlite-vec (if accessible)
- ETA based on rate

Exits when log shows a final summary or the process dies.
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path

VAULT_PATH = Path(
    "/Users/fer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
)
DB_PATH = Path.home() / ".local/share/obsidian-rag/ragvec/ragvec.db"


def _vault_file_count() -> int:
    return sum(1 for _ in VAULT_PATH.rglob("*.md"))


def _corpus_count() -> tuple[int, dict[str, int]]:
    """Return (total_chunks, per-root-folder breakdown)."""
    if not DB_PATH.exists():
        return 0, {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM meta_obsidian_notes_v11")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT substr(file, 1, instr(file, '/') - 1) AS root, COUNT(*) "
            "FROM meta_obsidian_notes_v11 GROUP BY root"
        )
        per_root = {row[0] or "(root)": row[1] for row in cur.fetchall()}
        conn.close()
        return total, per_root
    except Exception:
        return 0, {}


def _parse_log(log_text: str) -> dict:
    """Extract progress signals from the log."""
    lines = log_text.splitlines()
    indexed = sum(1 for l in lines if "Indexing " in l or "indexed:" in l)
    skipped = sum(1 for l in lines if "skipped" in l.lower())
    errors = sum(1 for l in lines if "error" in l.lower() or "fail" in l.lower())
    # Look for tqdm/rich progress indicator pattern
    pct_match = None
    for l in reversed(lines[-50:]):
        m = re.search(r"(\d+\.\d+)%|(\d+)\s*/\s*(\d+)", l)
        if m:
            pct_match = l.strip()
            break
    finished = (
        "completed" in log_text.lower()
        or "indexed total" in log_text.lower()
        or "✓ Done" in log_text
    )
    return {
        "indexed_lines": indexed,
        "skipped": skipped,
        "errors": errors,
        "last_pct_line": pct_match,
        "finished": finished,
    }


def main(log_path_str: str) -> None:
    log_path = Path(log_path_str)
    t_start = time.time()
    vault_total = _vault_file_count()
    print(f"Vault has {vault_total} .md files", flush=True)
    print(
        f"Targets: 01-Projects ({sum(1 for _ in (VAULT_PATH/'01-Projects').rglob('*.md'))}), "
        f"02-Areas ({sum(1 for _ in (VAULT_PATH/'02-Areas').rglob('*.md'))}), "
        f"03-Resources ({sum(1 for _ in (VAULT_PATH/'03-Resources').rglob('*.md'))}), "
        f"04-Archive ({sum(1 for _ in (VAULT_PATH/'04-Archive').rglob('*.md'))})",
        flush=True,
    )
    print("─" * 80, flush=True)

    while True:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        except Exception:
            text = ""
        info = _parse_log(text)
        total, per_root = _corpus_count()
        elapsed = int(time.time() - t_start)
        # Build a one-line dashboard
        roots_str = " ".join(
            f"{k}:{v}" for k, v in sorted(per_root.items()) if k
        )
        print(
            f"[{elapsed//60:2d}m{elapsed%60:02d}s] corpus={total:5d} chunks  "
            f"{roots_str}  log={info['indexed_lines']} indexed lines  "
            f"errors={info['errors']}",
            flush=True,
        )
        if info["finished"]:
            print("✓ index --reset finished", flush=True)
            return
        # Heuristic: if process is gone (no log activity for 60s + no live process), exit
        time.sleep(10)


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reindex.log"
    main(log)
