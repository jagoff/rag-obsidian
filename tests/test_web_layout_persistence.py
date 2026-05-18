"""Static guards for browser-local layout persistence.

These tests keep the configurable web layouts from regressing to ephemeral
DOM-only state. The actual behavior runs in the browser; here we pin the
storage keys, controls, and asset wiring used by those modules.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"
JS = STATIC / "js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_shared_layout_persistence_helper_exists() -> None:
    js = _read(JS / "layout-persistence.mjs")
    for symbol in (
        "export async function hydrateServerLayout",
        "export function clearServerLayout",
        "export function readString",
        "export function readJSON",
        "export function readObject",
        "export function writeJSON",
        "export function writeString",
        "export function removeKeys",
        "export function readSizeOverrides",
        "export function writeSizeOverrides",
        "export function applySizeDataset",
    ):
        assert symbol in js
    assert "/snapshot" in js
    assert "LOCAL_UPDATED_AT_KEY" in js
    assert "payload?.updated_at" in js
    assert "localWins" in js
    assert "_flushServerLayoutSnapshot" in js
    assert "_snapshotLocalLayout(keys)" in js


def test_home_layout_uses_shared_persistence_for_saved_layout_state() -> None:
    app = _read(JS / "home" / "app.mjs")
    autosizer = _read(JS / "home" / "autosizer.mjs")
    layout = _read(JS / "home" / "layout.mjs")

    assert 'from "./layout.mjs?v=' in app
    assert 'from "./panel-today.mjs?v=' in app
    assert 'from "../layout-persistence.mjs?v=' in autosizer
    assert "readSizeOverrides(LS_PANEL_SIZES" in autosizer
    assert "writeSizeOverrides(LS_PANEL_SIZES" in autosizer
    assert 'LS_PANEL_SIZES = "home.v2.panel-sizes.v1"' in autosizer

    assert 'from "../layout-persistence.mjs?v=' in layout
    assert 'from "./panel-today.mjs?v=' in layout
    assert 'from "./autosizer.mjs?v=' in layout
    assert 'SERVER_LAYOUT_PAGE = "home.v2"' in layout
    assert 'HERO_BODY_ID = "today-hero-body"' in layout
    assert 'ORDER_ITEM_SELECTOR = ":scope > .panel, :scope > .kpi, :scope > .hero-section"' in layout
    assert "await hydrateServerLayout(SERVER_LAYOUT_PAGE, SERVER_LAYOUT_KEYS)" in layout
    assert "export function refreshLayoutControls()" in layout
    assert "function queryLayoutItems(selector)" in layout
    assert "function allKpis()" in layout
    assert 'return queryLayoutItems(".kpi")' in layout
    assert "function observeLayoutContainer(sec)" in layout
    assert "ORDER_CONTAINER_IDS.forEach((secId)" in layout
    assert "disableNativeChildDrag(kpi)" in layout
    assert 'el.setAttribute("draggable", "false")' in layout
    assert "removeKeys([" in layout
    assert "clearServerLayout(SERVER_LAYOUT_PAGE)" in layout
    assert "writeObjectOrRemove(LS_PANELS_COLLAPSED" in layout
    assert "readString(LS_HERO_COLLAPSED)" in layout
    assert "compactSectionCollapseMap" in layout
    assert "setSectionCollapsed(section, btn, shouldCollapse)" in layout
    assert "initCollapsibleSections();" in layout
    assert "function _dropPlacement(ev, el)" in layout
    assert "function _containerInsertionFromPoint(container, ev)" in layout
    assert '"drop-axis-x"' in layout
    assert '"drop-axis-y"' in layout
    assert 'panel.querySelector(".panel-head") || panel.querySelector(":scope > h3")' in layout
    assert 'head.querySelector(".panel-collapse-btn, .hero-collapse-btn")' in layout
    assert "function setupHeroSection(sec)" in layout
    assert "setupHeroSection(n)" in layout
    assert '".panel-size-chip, .panel-collapse-btn, .hero-collapse-btn, .kpi-size-btn, .panel-resize-handle"' in layout
    assert "!ev.target?.closest(\".kpi-drag-grip\")" not in layout
    assert 'typeof value === "boolean"' in layout
    assert "refreshLayoutControls();" in app


def test_home_hero_layout_state_uses_shared_persistence() -> None:
    panel_today = _read(JS / "home" / "panel-today.mjs")

    assert 'from "../layout-persistence.mjs?v=' in panel_today
    assert "readJSON(LS_HERO_ORDER" in panel_today
    assert "writeJSON(LS_HERO_ORDER" in panel_today
    assert "readObject(LS_HERO_SUB_COLLAPSED" in panel_today
    assert "writeObjectOrRemove(LS_HERO_SUB_COLLAPSED" in panel_today
    assert "readString(LS_HERO_COLLAPSED)" in panel_today
    assert "writeString(LS_HERO_COLLAPSED" in panel_today
    assert "localStorage.setItem(LS_HERO_COLLAPSED" not in panel_today
    assert "sec.addEventListener(\"dragstart\", _onHeroDragStart)" not in panel_today
    assert "body.addEventListener(\"drop\", _onHeroBodyDrop)" not in panel_today
    assert "drag/resize/orden lo maneja layout.mjs" in panel_today
    assert "#home-cmdbar > .hero-section, .section-body > .hero-section" in panel_today
    assert "preservedHeroBodyItems" in panel_today
    assert 'el.classList?.contains("panel") || el.classList?.contains("kpi")' in panel_today


def test_home_kpis_can_live_in_section_grid_css() -> None:
    css = _read(STATIC / "home.v2.css")

    assert '.section-body > .kpi[data-w="half"]' in css
    assert '.section-body > .kpi[data-h="xl"]' in css
    assert ".section-body > .kpi:not([data-w])" in css
    assert '.section-body > .kpi[data-collapsed="true"]' in css
    assert ".panel.drop-axis-x.drop-before::before" in css
    assert ".kpi.drop-axis-x.drop-after::after" in css
    assert ".hero-section.drop-axis-x.drop-before::before" in css
    assert '.cmdbar > .hero-section[data-w="half"]' in css
    assert '.today-hero-body > .panel[data-w="half"]' in css
    assert '.hero-section[data-w="full"] .panel-size-chip-w' in css
    assert "grid-auto-flow: row;" in css
    assert ".today-hero-body { grid-template-columns: repeat(2, 1fr); }" not in css


def test_home_spotify_empty_signal_does_not_render_blank_panel() -> None:
    js = _read(JS / "home" / "panel-ambient.mjs")

    assert 'sp.state === "empty"' in js
    assert "panel.hidden = true" in js
    assert 'if (body) body.innerHTML = ""' in js


def test_home_page_versions_layout_boot_asset() -> None:
    html = _read(STATIC / "home.v2.html")

    version_match = re.search(r'__RAG_HOME_ASSET_VERSION__ = "([^"]+)"', html)
    assert version_match
    version = version_match.group(1)

    assert f'/static/js/home/app.mjs?v={version}' in html
    assert f'/static/pwa/register-sw.js?v={version}' in html


def test_finance_layout_persists_order_collapse_and_section_sizes() -> None:
    js = _read(JS / "finance-layout.mjs")

    assert 'from "./layout-persistence.mjs"' in js
    assert 'LS_SECTIONS_ORDER = "finance.sections.order.v1"' in js
    assert 'LS_SECTIONS_COLLAPSED = "finance.sections.collapsed.v1"' in js
    assert 'LS_SECTION_SIZES = "finance.sections.sizes.v1"' in js

    assert "readSizeOverrides(LS_SECTION_SIZES" in js
    assert "writeSizeOverrides(LS_SECTION_SIZES" in js
    assert "removeKeys([LS_SECTIONS_ORDER, LS_SECTIONS_COLLAPSED, LS_SECTION_SIZES])" in js
    assert "map[section.id] = collapsed" in js
    assert 'typeof value === "boolean"' in js

    assert "section-size-btn" in js
    assert "finance-resize-handle" in js
    assert "pointerdown" in js
    assert "applySavedSectionSize(section)" in js


def test_finance_page_has_resizable_section_css_and_versioned_layout_asset() -> None:
    html = _read(STATIC / "finance.html")

    for css in (
        '.learn-section[data-h="tall"] .chart-card',
        '.learn-section[data-h="xl"] .chart-card',
        '.learn-section[data-h="tall"] .tx-scroll',
        '.learn-section[data-h="xl"] .tx-scroll',
        ".section-size-btn",
        ".finance-resize-handle",
    ):
        assert css in html

    versions = re.findall(r"/static/js/finance-layout\.mjs\?v=([0-9a-z-]+)", html)
    assert len(versions) == 2
    assert len(set(versions)) == 1
