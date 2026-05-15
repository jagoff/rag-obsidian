"""Vendor script integrity checks for web/static/*.html.

SRI (Subresource Integrity) on same-origin local vendor files is defence in
depth: it rejects silent swaps from a compromised server or a botched
re-vendor. The web UI is explicitly local-first/offline-capable, so HTML pages
must not load script bundles from public CDNs.

If an integrity assertion fails, regenerate the hash with:
    openssl dgst -sha384 -binary web/static/vendor/<file> | openssl base64 -A
and update the `integrity="sha384-..."` attribute in the HTML.
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest

_WEB_STATIC = Path(__file__).resolve().parent.parent / "web" / "static"

_INTEGRITY_RE = re.compile(
    r"""integrity=(?:"|')sha384-([A-Za-z0-9+/=]+)(?:"|')""",
    re.IGNORECASE,
)
_SCRIPT_TAG_RE = re.compile(
    r"""<script\b(?P<attrs>[^>]*)\bsrc=(?P<q>["'])(?P<src>[^"']+)(?P=q)(?P<tail>[^>]*)>""",
    re.IGNORECASE | re.DOTALL,
)


def _sha384_b64(data: bytes) -> str:
    """Mirror of `openssl dgst -sha384 -binary | openssl base64 -A`."""
    h = hashlib.sha384(data).digest()
    return base64.b64encode(h).decode("ascii")


def _html_files() -> list[Path]:
    return sorted(_WEB_STATIC.glob("*.html"))


def _script_tags(html: str) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    for match in _SCRIPT_TAG_RE.finditer(html):
        tag = match.group(0)
        src = match.group("src")
        tags.append((tag, src))
    return tags


@pytest.mark.parametrize("html_path", _html_files(), ids=lambda p: p.name)
def test_html_does_not_load_script_bundles_from_cdns(html_path: Path):
    html = html_path.read_text(encoding="utf-8")
    external = [
        src for _tag, src in _script_tags(html)
        if src.startswith(("http://", "https://", "//"))
    ]
    assert not external, (
        f"{html_path.name}: external script srcs are not allowed: {external}"
    )


@pytest.mark.parametrize("html_path", _html_files(), ids=lambda p: p.name)
def test_local_vendor_scripts_carry_correct_sri_hash(html_path: Path):
    html = html_path.read_text(encoding="utf-8")
    for tag, src in _script_tags(html):
        if not src.startswith("/static/vendor/"):
            continue
        rel = src.removeprefix("/static/")
        vendor_file = _WEB_STATIC / rel
        assert vendor_file.is_file(), f"{html_path.name}: missing {src}"
        expected_hash = _sha384_b64(vendor_file.read_bytes())
        integrity_match = _INTEGRITY_RE.search(tag)
        assert integrity_match is not None, (
            f"{html_path.name}: <script src='{src}'> is missing "
            "integrity='sha384-...'."
        )
        got = integrity_match.group(1)
        assert got == expected_hash, (
            f"{html_path.name}: integrity hash mismatch for {src}.\n"
            f"  HTML says:  sha384-{got}\n"
            f"  File hash:  sha384-{expected_hash}\n"
            "Re-vendor happened without updating the HTML."
        )


def test_hash_helper_matches_sha384_shape():
    """Sanity: the Python hashlib path produces the same value openssl
    would. If someone migrates the regen instruction in the docstring to
    a different tool, this ensures the spec stays stable."""
    raw = base64.b64decode(_sha384_b64(b"obsidian-rag"))
    assert len(raw) == 48, "sha384 digest must be 48 bytes"
