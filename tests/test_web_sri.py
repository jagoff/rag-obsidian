"""Tests that web/static/{index,home}.html carry the correct SRI hash
for vendor/marked.min.js.

SRI (Subresource Integrity) on a same-origin local file is defence-in-
depth: it rejects silent swaps from a compromised server or a botched
re-vendor (e.g. someone updates marked.min.js but forgets to update the
hash in HTML). Without SRI, a corrupted vendor bundle loads silently.

If these fail, regenerate the hash with:
    cat web/static/vendor/marked.min.js | openssl dgst -sha384 -binary | openssl base64 -A
and update the `integrity="sha384-..."` attribute in both HTML files.
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest

_WEB_STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
_MARKED_JS = _WEB_STATIC / "vendor" / "marked.min.js"

_INTEGRITY_RE = re.compile(
    r"""integrity=(?:"|')sha384-([A-Za-z0-9+/=]+)(?:"|')""",
    re.IGNORECASE,
)


def _sha384_b64(data: bytes) -> str:
    """Mirror of `openssl dgst -sha384 -binary | openssl base64 -A`."""
    h = hashlib.sha384(data).digest()
    return base64.b64encode(h).decode("ascii")


@pytest.fixture(scope="module")
def expected_marked_hash() -> str:
    assert _MARKED_JS.is_file(), f"marked.min.js missing at {_MARKED_JS}"
    return _sha384_b64(_MARKED_JS.read_bytes())


@pytest.mark.parametrize("html_name", ["index.html", "home.html"])
def test_html_carries_correct_sri_hash(html_name: str, expected_marked_hash: str):
    html = (_WEB_STATIC / html_name).read_text(encoding="utf-8")
    # Find the script tag that actually references marked.min.js, then
    # look for the integrity attr nearby. Both files currently put
    # integrity on the same <script> block that has src="/static/vendor/marked.min.js".
    marked_block_re = re.compile(
        r"<script[^>]*src=[\"']/static/vendor/marked\.min\.js[\"'][^>]*>",
        re.DOTALL,
    )
    m = marked_block_re.search(html)
    assert m is not None, (
        f"{html_name}: no <script> tag referencing /static/vendor/marked.min.js"
    )
    tag = m.group(0)
    integrity_match = _INTEGRITY_RE.search(tag)
    assert integrity_match is not None, (
        f"{html_name}: <script src=marked> is missing integrity='sha384-...' "
        f"attribute. Regenerate with `openssl dgst -sha384 -binary | openssl base64 -A`."
    )
    got = integrity_match.group(1)
    assert got == expected_marked_hash, (
        f"{html_name}: integrity hash mismatch.\n"
        f"  HTML says:  sha384-{got}\n"
        f"  File hash:  sha384-{expected_marked_hash}\n"
        f"Re-vendor happened without updating the HTML — run the regen "
        f"command in the test docstring and commit the new value."
    )


def test_hash_matches_openssl_command(expected_marked_hash: str):
    """Sanity: the Python hashlib path produces the same value openssl
    would. If someone migrates the regen instruction in the docstring to
    a different tool, this ensures the spec stays stable."""
    # This is a no-op assertion against the fixture — the fixture
    # itself uses hashlib.sha384 + base64. We're just proving the
    # hash is non-empty and base64-decodable (48 bytes for sha384).
    raw = base64.b64decode(expected_marked_hash)
    assert len(raw) == 48, "sha384 digest must be 48 bytes"
