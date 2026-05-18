from __future__ import annotations


def test_home_spotify_fallback_reads_latest_vault_recent_note(tmp_path, monkeypatch):
    from web import server

    spotify_dir = tmp_path / "99-obsidian/99-AI/external-ingest/Spotify"
    spotify_dir.mkdir(parents=True)
    (spotify_dir / "2026-05-12.md").write_text(
        """---
source: spotify
snapshot_date: 2026-05-12
track_count: 2
---

# Spotify recently played

- `2026-05-11 13:28:42` [Down In A Hole](https://open.spotify.com/track/abc) — Alice In Chains · _Dirt_
- `2026-05-07 21:03:08` [Creep](https://open.spotify.com/track/def) — Stone Temple Pilots · _Core_
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "resolve_vault_paths", lambda _names: [("home", tmp_path)])

    out = server._fetch_spotify_vault_snapshot(limit=2)

    assert out is not None
    assert out["state"] == "history"
    assert out["fallback"] == "vault_recent_snapshot"
    assert out["snapshot_date"] == "2026-05-12"
    assert [item["name"] for item in out["recent_today"]] == ["Down In A Hole", "Creep"]
    assert out["recent_today"][0]["artist"] == "Alice In Chains"
    assert out["recent_today"][0]["album"] == "Dirt"
    assert out["recent_today"][0]["track_id"] == "https://open.spotify.com/track/abc"
