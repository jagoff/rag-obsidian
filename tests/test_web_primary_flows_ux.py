"""Static guards for the main web UX flows.

The browser behavior is covered manually in Playwright during the audit pass;
these tests pin the concrete regressions found there so they do not creep back.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"


def _read(rel: str) -> str:
    return (STATIC / rel).read_text(encoding="utf-8")


def test_chat_deep_link_uses_window_history_not_prompt_history() -> None:
    app_js = _read("app.js")
    assert "window.history.replaceState({}, \"\", window.location.pathname)" in app_js
    assert "history.replaceState({}, \"\", window.location.pathname)" not in app_js.replace(
        "window.history.replaceState({}, \"\", window.location.pathname)",
        "",
    )


def test_wzp_mobile_starts_on_chat_list_and_hides_closed_drawers() -> None:
    wa_html = _read("wa.html")
    wa_css = _read("wa.css")

    assert '<body data-pane="sidebar">' in wa_html
    for selector in (
        ".wa-anticipate-drawer[hidden]",
        ".wa-memory-drawer[hidden]",
        ".wa-promises-drawer[hidden]",
    ):
        assert selector in wa_css
    assert 'body[data-pane="thread"] .wa-thread' in wa_css


def test_finance_mobile_tables_scroll_and_cards_start_compact() -> None:
    finance_html = _read("finance.html")
    finance_js = _read("finance.js")

    assert "min-width: 520px" in finance_html
    assert "max-height: 480px; overflow: auto;" in finance_html
    assert ".cc-detail-group" in finance_html
    assert '<a href="/chat">chat</a>' in finance_html
    assert '<a href="/wzp">wzp</a>' in finance_html
    assert '<a href="/logs">logs</a>' in finance_html

    assert "const analysisGroup = analysisSections" in finance_js
    assert "const consumosGroup = consumosSection" in finance_js
    assert '<details class="cc-detail-group">' in finance_js
    assert 'summary>Consumos del ciclo</summary>' in finance_js


def test_mobile_page_navigation_stays_single_row_scrollable() -> None:
    home_css = _read("home.v2.css")
    status_html = _read("status.html")
    logs_html = _read("logs.html")

    assert ".topbar .topnav" in home_css
    assert "overflow-x: auto" in home_css
    assert ".topbar .topnav-link { flex: 0 0 auto; }" in home_css

    for html in (status_html, logs_html):
        assert ".header-meta {" in html
        assert "flex-wrap: nowrap" in html
        assert "overflow-x: auto" in html
        assert '<a href="/wzp">wzp</a>' in html
