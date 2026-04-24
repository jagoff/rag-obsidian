"""Regression tests para que el popover de /wzp y /mail siga cumpliendo
mobile UX best practices (Apple HIG / WCAG 2.5.5 / iOS Safari quirks).

NO testeamos rendering real — eso requeriría Playwright con viewports
de iPhone, que ya verifico a mano cuando toco style.css. Acá nos
enfocamos en:

  1. El CSS contiene las reglas críticas (media queries + caps + tap
     target padding + touch-action). Si alguien las saca por
     refactor/dead-code-removal, el test rompe y se acuerda por qué
     estaban.
  2. La PWA wiring de la PWA sigue intacta (manifest, register-sw,
     viewport-fit=cover).

Cubre regresiones reales que tuvimos:
  - El popover overflow off-screen al top en landscape (popover.y=-149
    pre-fix, primeros 5 contactos invisibles). Fix: cap dinámico con
    `min(110px, calc(100dvh - 270px))` en orientation: landscape.
  - Tap targets <44px (rows de 31px, falla regular del pulgar). Fix:
    padding 12px vertical en .slash-item (mobile media query).
  - Font 12.5px ilegible en mobile. Fix: 14-15px en mobile.
  - Truncation horizontal en números argentinos largos. Fix: stack
    name+phone verticalmente en mobile.
"""
from __future__ import annotations
from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
_STYLE_CSS = (_STATIC_DIR / "style.css").read_text(encoding="utf-8")
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── CSS reglas críticas ─────────────────────────────────────────────────


def test_popover_has_dynamic_max_height():
    """El cap base usa `dvh` (dynamic viewport height) para excluir el
    chrome de iOS Safari. Con vh estático el address bar al desaparecer
    deja un gap raro. Con dvh el browser recomputa al toque."""
    assert "max-height: min(260px, 60dvh)" in _STYLE_CSS, (
        "popover base necesita cap dinámico con dvh — sin esto el "
        "popover en mobile no se adapta cuando el address bar de Safari "
        "aparece/desaparece"
    )


def test_popover_has_overscroll_contain():
    """Cuando scrolleás dentro del popover en iOS y llegás al final,
    NO debería propagar al body (que rebotaría toda la página). Esto
    es comportamiento iOS-like nativo."""
    assert "overscroll-behavior: contain" in _STYLE_CSS, (
        "sin overscroll-behavior:contain, scrollear el popover en iOS "
        "rebota la página entera al llegar al límite"
    )


def test_slash_item_has_touch_action_manipulation():
    """`touch-action: manipulation` saca el delay de 300ms del tap +
    el zoom de doble-tap en items. Crítico para que el feedback al
    seleccionar un contacto sea inmediato."""
    assert "touch-action: manipulation" in _STYLE_CSS, (
        "items necesitan touch-action:manipulation para que iOS no "
        "espere 300ms para distinguir tap de doble-tap zoom"
    )


def test_slash_item_disables_tap_highlight():
    """El highlight default de Safari (`-webkit-tap-highlight-color`)
    es un overlay azul gritón sobre tappable elements. Lo apagamos
    porque ya tenemos feedback con :hover/.active."""
    assert "-webkit-tap-highlight-color: transparent" in _STYLE_CSS


def test_mobile_media_query_exists():
    """El bloque de overrides para mobile debe usar un selector que
    matchee tanto viewport pequeño como pointer:coarse (touch
    screens en cualquier viewport — iPad, desktop con touch)."""
    assert "@media (max-width: 640px), (pointer: coarse)" in _STYLE_CSS, (
        "el media query mobile debe disparar tanto por viewport como "
        "por pointer:coarse — sino tablets con touch quedan con tap "
        "targets de 31px"
    )


def test_mobile_tap_target_at_least_44px_padding():
    """Apple HIG + WCAG 2.5.5 AAA piden tap targets >=44x44 CSS px.
    Con padding 12px vertical + line-height ~20px = ~44px.
    Verificamos que el padding sigue siendo >=12px en el media
    query mobile."""
    # Buscar el bloque mobile y el `.slash-item` dentro
    mobile_idx = _STYLE_CSS.find("@media (max-width: 640px), (pointer: coarse)")
    assert mobile_idx > 0
    # Limitar la búsqueda al bloque mobile (hasta el siguiente @media o end)
    end_idx = _STYLE_CSS.find("@media", mobile_idx + 50)
    block = _STYLE_CSS[mobile_idx:end_idx if end_idx > 0 else len(_STYLE_CSS)]
    assert ".slash-item" in block
    assert "padding: 12px 12px" in block, (
        "tap targets en mobile necesitan padding 12px vertical para "
        "alcanzar 44px de alto (WCAG 2.5.5 AAA)"
    )


def test_mobile_stacks_name_and_hint_vertically():
    """En mobile el row cambia a `grid-template-columns: 1fr` (single
    column) para evitar truncar números argentinos largos. Sin esto
    "+5491145238999" mostraba "..." al medio, inutilizable."""
    mobile_idx = _STYLE_CSS.find("@media (max-width: 640px), (pointer: coarse)")
    end_idx = _STYLE_CSS.find("@media", mobile_idx + 50)
    block = _STYLE_CSS[mobile_idx:end_idx if end_idx > 0 else len(_STYLE_CSS)]
    assert "grid-template-columns: 1fr" in block, (
        "mobile necesita layout stacked (1 columna) para que números "
        "largos no se trunquen con ellipsis"
    )


def test_mobile_font_size_at_least_15px_for_name():
    """Body text en mobile debe ser >=15px (Apple HIG mínimo). El
    name del contacto (slash-cmd) sube a 15px en mobile."""
    mobile_idx = _STYLE_CSS.find("@media (max-width: 640px), (pointer: coarse)")
    end_idx = _STYLE_CSS.find("@media", mobile_idx + 50)
    block = _STYLE_CSS[mobile_idx:end_idx if end_idx > 0 else len(_STYLE_CSS)]
    # Buscar slash-cmd block con font-size 15px
    assert ".slash-item .slash-cmd" in block
    # En el bloque mobile hay font-size: 15px asociado al slash-cmd
    cmd_idx = block.find(".slash-item .slash-cmd")
    cmd_section = block[cmd_idx:cmd_idx + 200]
    assert "font-size: 15px" in cmd_section, (
        "el name del contacto debe ser >=15px en mobile para legibilidad"
    )


def test_landscape_orientation_query_caps_height():
    """En landscape mobile (iPhone 14 = 844x390), el espacio arriba
    del input es ~128px. El popover en orientation:landscape debe
    estar capado a ~110px max — sin esto los items se renderean
    fuera de pantalla al top."""
    assert "@media (orientation: landscape) and (max-height: 500px)" in _STYLE_CSS, (
        "necesitamos un media query específico para landscape compacto "
        "porque el cap de portrait (280px) no cabe en 128px disponibles"
    )
    land_idx = _STYLE_CSS.find("@media (orientation: landscape) and (max-height: 500px)")
    block = _STYLE_CSS[land_idx:land_idx + 1000]
    # El cap en landscape es 110px (o calc adaptado)
    assert "max-height: min(110px" in block, (
        "landscape compacto necesita cap a 110px para no overflow off-screen"
    )


def test_safe_area_inset_top_respected():
    """El popover acepta `env(safe-area-inset-top)` por si el iPhone
    está en landscape y el popover se acerca al notch / Dynamic
    Island. Sin esto las primeras filas se solapan con el notch."""
    assert "env(safe-area-inset-top" in _STYLE_CSS, (
        "popover debe respetar safe-area para no tapar el notch/DI en "
        "landscape — sino el primer item queda parcialmente oculto"
    )


# ── PWA + viewport wiring (lo que evita el zoom de iOS al focusear input) ─


def test_input_font_size_is_16px_to_prevent_ios_zoom():
    """iOS hace auto-zoom del viewport cuando el user focusea un input
    con font-size <16px. Esto es uno de los gotchas más conocidos de
    iOS Safari. El input del chat debe estar a 16px en mobile widths
    (puede estar a otro tamaño en desktop, pero los media queries
    para mobile deben subirlo a 16). Buscamos cualquier regla que
    matchee `#input` con font-size:16px en el archivo."""
    # Por simplicidad chequeamos que exista AL MENOS una regla
    # `#input { ... font-size: 16px ... }` en el CSS (puede estar
    # adentro de un @media query — eso es OK).
    import re
    # Match `#input` followed by `{` y dentro `font-size: 16px` antes del `}`.
    pattern = re.compile(r'#input\s*\{[^}]*font-size:\s*16px[^}]*\}', re.DOTALL)
    has_16 = bool(pattern.search(_STYLE_CSS))
    assert has_16, (
        "#input necesita font-size:16px para evitar el auto-zoom de "
        "iOS Safari al hacer focus — gotcha documentado en MDN/WebKit. "
        "Aunque el #input base esté en 14px, debe haber un @media para "
        "mobile (<=640px o pointer:coarse) que lo suba a 16px."
    )


def test_viewport_meta_includes_viewport_fit_cover():
    """Para que la PWA respete safe-area-inset (notch + home indicator),
    el meta viewport debe incluir `viewport-fit=cover`. Sino el OS
    deja barras negras alrededor en standalone mode."""
    assert 'viewport-fit=cover' in _INDEX_HTML, (
        "meta viewport sin viewport-fit=cover deja barras negras en "
        "iPhone con notch al lanzar la PWA en standalone"
    )


# ── Help modal mobile UX (audit 2026-04-24) ────────────────────────────────


def test_close_button_has_44px_tap_target():
    """Close `×` button del help modal debe ser 44x44 mínimo (Apple HIG /
    WCAG 2.5.5). Pre-fix era 23.6875px de ancho — falla regular del pulgar
    al tocar el "x" en mobile."""
    assert "min-width: 44px" in _STYLE_CSS
    assert "min-height: 44px" in _STYLE_CSS, (
        ".modal-close necesita min-width + min-height de 44px (Apple HIG / "
        "WCAG 2.5.5 AAA). Sin esto el botón es 23x44 y falla el tap mobile."
    )


def test_modal_card_uses_dvh_for_max_height():
    """Modal usa `dvh` (dynamic viewport) en lugar de `vh` (static) para
    que el address bar de iOS Safari no deje al modal con overflow al
    aparecer/desaparecer."""
    assert "max-height: min(80dvh," in _STYLE_CSS, (
        "modal-card max-height debe usar dvh — sin esto en iOS el modal "
        "queda con altura fija y se sale por debajo cuando el address "
        "bar reaparece"
    )


def test_modal_respects_safe_area_inset():
    """El padding del modal en mobile debe respetar safe-area-inset-top
    y bottom para no superponerse con notch / Dynamic Island / home
    indicator."""
    assert "env(safe-area-inset-top" in _STYLE_CSS
    assert "env(safe-area-inset-bottom" in _STYLE_CSS
    # En el bloque @media (max-width: 480px) del .modal hay max() con
    # safe-area-inset.
    idx = _STYLE_CSS.find("@media (max-width: 480px)")
    assert idx > 0
    # Buscar `.modal {` en alguno de los bloques @media (max-width: 480px).
    found = False
    while idx > 0:
        end = _STYLE_CSS.find("@media", idx + 50)
        block = _STYLE_CSS[idx:end if end > 0 else len(_STYLE_CSS)]
        if "padding-top: max(" in block and "safe-area-inset-top" in block:
            found = True
            break
        idx = _STYLE_CSS.find("@media (max-width: 480px)", idx + 50)
    assert found, (
        ".modal en mobile (<=480px) debe usar max(16px, env(safe-area-inset-top)) "
        "en padding-top — sino el modal se solapa con el notch en iPhone X+"
    )


def test_kb_list_stacks_vertically_on_small_screens():
    """En mobile (<=480px) el `.kb-list` cambia a 1 columna (stack
    dt sobre dd) en lugar de la grilla 2-col del desktop. La grilla
    angosta deja whitespace incómodo y el dd wrappea a 1-2 palabras
    por línea."""
    # Buscar el @media (max-width: 480px) que tenga kb-list con 1fr.
    needle = ".kb-list {\n    grid-template-columns: 1fr"
    assert needle in _STYLE_CSS, (
        "@media (max-width: 480px) debe forzar `grid-template-columns: 1fr` "
        "en .kb-list para stack vertical en mobile"
    )


def test_body_has_overflow_x_hidden_for_layout_safety():
    """Defensa en profundidad: si algún child se overflow horizontal
    (ej. #quick-chips con max-width hard-coded), el body lo clipea
    en lugar de hacer al doc más ancho que el viewport.

    Bug histórico: pre-fix, #quick-chips forzaba body a 760px en mobile
    y el modal de /help renderaba off-screen porque `position: fixed`
    cubría el body wider, no el viewport real."""
    assert "overflow-x: hidden" in _STYLE_CSS, (
        "html/body necesitan `overflow-x: hidden` como safety net para "
        "que un child overflow no rompa position:fixed en otros lados"
    )


def test_quick_chips_max_width_responsive():
    """`#quick-chips { max-width: 760px }` rompía el layout mobile —
    el container se forzaba a 760px en viewports de 375px. Fix:
    `min(760px, 100%)` para que en mobile fitee el viewport."""
    assert "max-width: min(760px, 100%)" in _STYLE_CSS, (
        "#quick-chips max-width debe ser responsive (min(760px, 100%)) "
        "en lugar de hard-coded 760px"
    )
