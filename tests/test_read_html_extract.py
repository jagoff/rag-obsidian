"""Tests for `_read_extract` — focus on hostile payloads that the
pre-2026-04-20 regex-based strip handled partially.

Post-fix uses `html.parser.HTMLParser` which parses structure rather
than string-matching, so attribute content never reaches the output.
These tests are the regression guard against someone reverting to a
regex approach.
"""
from __future__ import annotations


import rag


# ── Baseline ──────────────────────────────────────────────────────────────


def test_extract_basic_title_and_text():
    html = "<html><head><title>My Note</title></head><body><p>Hello world.</p></body></html>"
    title, text = rag._read_extract(html)
    assert title == "My Note"
    assert "Hello world." in text


def test_extract_preserves_paragraph_breaks():
    html = "<body><p>Para one.</p><p>Para two.</p></body>"
    _, text = rag._read_extract(html)
    # Paragraph closes insert newlines; double newline between paragraphs survives.
    assert "Para one." in text
    assert "Para two." in text
    # The two paragraphs should NOT be smushed onto the same line.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 2


# ── Hostile payloads (the point of the fix) ─────────────────────────────


def test_script_content_is_stripped():
    """<script> content must not leak into the extracted text."""
    html = "<body><p>Real content.</p><script>alert('xss')</script></body>"
    _, text = rag._read_extract(html)
    assert "Real content." in text
    assert "alert" not in text
    assert "xss" not in text


def test_style_content_is_stripped():
    html = "<head><style>body { background: url(javascript:alert(1)); }</style></head><body><p>OK.</p></body>"
    _, text = rag._read_extract(html)
    assert "OK." in text
    assert "javascript" not in text
    assert "background" not in text


def test_onerror_attribute_never_leaks_to_text():
    """The classic payload: <img src=x onerror="alert(1)">.
    Pre-fix, a regex strip could leave the attribute value visible
    as text content if the tag was malformed. Post-fix: the parser
    sees a structured attribute, never emits it via handle_data."""
    html = '<body><p>Before.</p><img src=x onerror="alert(\'xss\')"><p>After.</p></body>'
    _, text = rag._read_extract(html)
    assert "Before." in text
    assert "After." in text
    assert "alert" not in text
    assert "xss" not in text
    assert "onerror" not in text


def test_malformed_nested_script_still_fully_skipped():
    """<script> with a nested '<script>' tag inside (appears in some
    anti-XSS filter-bypass payloads). The stdlib parser treats this
    as script content verbatim up to the first </script>, which is
    all we care about — the inner garbage never leaks to handle_data."""
    html = "<body><p>Safe.</p><script>var x = '<script>alert(1)</script>'</script><p>Also safe.</p></body>"
    _, text = rag._read_extract(html)
    assert "Safe." in text
    assert "Also safe." in text
    assert "alert" not in text


def test_iframe_content_is_stripped():
    html = '<body><p>Text.</p><iframe src="https://evil.example/phish"><p>Inner junk.</p></iframe></body>'
    _, text = rag._read_extract(html)
    assert "Text." in text
    assert "Inner junk." not in text
    assert "evil.example" not in text


def test_svg_content_is_stripped():
    """SVG can carry inline scripts + event handlers. Skip entirely."""
    html = "<body><p>X.</p><svg><script>alert(2)</script><g onclick='evil()'>Y</g></svg></body>"
    _, text = rag._read_extract(html)
    assert "X." in text
    assert "alert" not in text
    assert "evil()" not in text


# ── Graceful degradation ────────────────────────────────────────────────


def test_empty_input_returns_empty_pair():
    title, text = rag._read_extract("")
    assert title == ""
    assert text == ""


def test_no_title_tag_returns_empty_title():
    html = "<body><p>No head section.</p></body>"
    title, text = rag._read_extract(html)
    assert title == ""
    assert "No head section." in text


def test_malformed_html_does_not_raise():
    """Unclosed tags, nested quotes, broken markup — must not raise.
    Whatever the parser manages to extract is acceptable as long as
    no exception escapes _read_extract."""
    html = "<body><p>Start<div class='x<y'><unclosed>text<<<>><b>end"
    title, text = rag._read_extract(html)
    assert isinstance(title, str)
    assert isinstance(text, str)
    # Must have extracted at least the visible words.
    assert "Start" in text or "text" in text or "end" in text


def test_html_entities_are_decoded():
    """&amp; &#39; &quot; etc. should come through as their char form."""
    html = "<body><p>Tom &amp; Jerry &#39;quoted&#39; &quot;text&quot;</p></body>"
    _, text = rag._read_extract(html)
    assert "Tom & Jerry 'quoted' \"text\"" in text


# ── Truncation ──────────────────────────────────────────────────────────


def test_text_truncated_to_max_chars():
    """Long pages get capped at _READ_MAX_CHARS + ellipsis marker."""
    long_paragraph = "Lorem ipsum dolor sit amet. " * 2000
    html = f"<body><p>{long_paragraph}</p></body>"
    _, text = rag._read_extract(html)
    assert len(text) <= rag._READ_MAX_CHARS + 1  # + "…"
    assert text.endswith("…")
