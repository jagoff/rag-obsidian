"""Genera los assets de la PWA (icons + splash screens) para iOS/Android.

Output: `web/static/pwa/*.png`

Uso:
    .venv/bin/python scripts/gen_pwa_assets.py

Cuándo regenerar:
    - Cambió el branding / color del logo.
    - Apple agregó un tamaño de splash nuevo (raro, suele pasar cada ~1 año).
    - Se agregó un icon alternativo (ej. maskable para Android 12+).

Por qué no hacerlo en build time / on-the-fly:
    - Estos PNGs son estáticos y no cambian entre releases. Regenerarlos en
      cada deploy es puro waste (son ~300 KB totales y tardan ~1s).
    - La alternativa sería vendorar un paquete npm (`pwa-asset-generator`)
      pero ya tenemos Pillow en el stack (viene como transitivo de las deps
      de imagen de ocrmac / faster-whisper), así que una tool más es mejor
      que un nodejs más.

Diseño:
    - Icon: fondo #1a1a1f (matches --bg del theme dark), texto "rag" en
      serif tono cyan (#79c0ff, matches el accent). El punto "·" del
      topbar lo emulamos como un circle cyan al costado.
    - Maskable variant: el safe-zone es 80% del canvas (spec W3C) — el
      texto se escala al 60% del canvas así sobra margen contra el crop
      circular / squircle del launcher Android.
    - Splash screens: mismo fondo, logo al centro, relación de tamaños
      escala a ~18% del lado corto del dispositivo.

Devices cubiertos (splash screens):
    Sólo iPhone (portrait). El user dijo "iphone se sienta nativa" —
    iPads + landscape se pueden agregar después sin romper nada (iOS usa
    el primer `apple-touch-startup-image` que matchea el media query).

    Tamaños actualizados 2026-04 (cover all iPhones since iPhone X):
        - iPhone X / XS / 11 Pro:            1125 × 2436  @3x
        - iPhone XR / 11:                     828 × 1792  @2x
        - iPhone XS Max / 11 Pro Max:        1242 × 2688  @3x
        - iPhone 12 mini / 13 mini:          1080 × 2340  @3x
        - iPhone 12 / 12 Pro / 13 / 13 Pro / 14 :  1170 × 2532  @3x
        - iPhone 12 Pro Max / 13 Pro Max / 14 Plus:  1284 × 2778  @3x
        - iPhone 14 Pro / 15 / 15 Pro / 16 : 1179 × 2556  @3x
        - iPhone 14 Pro Max / 15 Pro Max / 16 Plus:  1290 × 2796  @3x
        - iPhone 16 Pro:                     1206 × 2622  @3x
        - iPhone 16 Pro Max:                 1320 × 2868  @3x
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "web" / "static" / "pwa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── brand ──────────────────────────────────────────────────────────────
BG = (26, 26, 31)          # #1a1a1f — matches --bg
FG = (236, 236, 237)       # #ececed — matches --text
ACCENT = (121, 192, 255)   # #79c0ff — matches --cyan
DIM = (160, 160, 166)      # #a0a0a6 — matches --text-dim

TEXT = "rag"


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Carga una font serif del sistema (macOS). Fallback al default PIL."""
    # Orden de preferencia: SF Mono (monoespaciada, el vibe del CLI) →
    # Menlo (macOS default mono) → default bitmap si no hay nada.
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo-Bold.ttf",
        "/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    # Último recurso: bitmap font (sin antialiasing, feo pero existe).
    return ImageFont.load_default()


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Devuelve bbox (x0, y0, x1, y1) del texto con la font dada."""
    return draw.textbbox((0, 0), text, font=font)


def render_logo(size: int, *, safe_zone: float = 1.0) -> Image.Image:
    """Render del logo `rag·` centrado, con fondo sólido.

    safe_zone: fracción del canvas donde el contenido debe caber (spec
    maskable dice 0.8). Para el icon normal pasamos 1.0 (full bleed del texto
    dentro del canvas). Para la variante maskable pasamos 0.8 así los
    launchers Android pueden croppear el borde sin comerse la "g".

    El diseño: texto centrado vertical + horizontalmente + un dot cyan
    pegado al costado derecho (como el "rag · chat" del topbar).
    """
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)

    # Target ocupación del combo completo (texto + gap + dot) en el canvas.
    # safe_zone=1.0 → combo ocupa ~60% del lado; safe_zone=0.8 → ~48%.
    # (Antes usábamos la ALTURA como target y el texto se iba por ancho en
    # SF Mono — SF Mono "rag" es ~2.0× más ancho que alto, así que la h
    # era el constraint equivocado; la w gana siempre.)
    target_combo_width = int(size * 0.60 * safe_zone)

    # Proporciones fijas del combo: dot-radius ~ 5.5% del canvas, gap ~ 5%.
    # Las calculamos sobre el canvas (no sobre el combo) para que escalen
    # linealmente con el tamaño final.
    dot_r = max(int(size * 0.055), 3)
    dot_gap = max(int(size * 0.05), 2)
    dot_combo_w = dot_gap + dot_r * 2  # cuánto le roba el dot al combo
    target_text_width = target_combo_width - dot_combo_w

    # Bisección sobre font size hasta que el TEXT ancho coincida con target.
    lo, hi = 10, size * 2
    font = _font(target_text_width)
    for _ in range(14):
        mid = (lo + hi) // 2
        f = _font(mid)
        bbox = _measure(draw, TEXT, f)
        w = bbox[2] - bbox[0]
        if w > target_text_width:
            hi = mid
        else:
            lo = mid
            font = f
        if hi - lo <= 1:
            break

    bbox = _measure(draw, TEXT, font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    total_w = text_w + dot_gap + dot_r * 2
    # Centrado horizontal del combo (texto + gap + dot).
    x_text = (size - total_w) // 2 - bbox[0]
    # Centrado vertical del texto (usando baseline correction).
    y_text = (size - text_h) // 2 - bbox[1]

    # Texto principal.
    draw.text((x_text, y_text), TEXT, fill=FG, font=font)

    # Dot cyan — eje vertical lo alineamos al centro visual del texto (no
    # al centro geométrico del canvas), para que se sienta parte del logo.
    dot_cx = x_text + bbox[0] + text_w + dot_gap + dot_r
    dot_cy = y_text + bbox[1] + text_h // 2
    draw.ellipse(
        [(dot_cx - dot_r, dot_cy - dot_r), (dot_cx + dot_r, dot_cy + dot_r)],
        fill=ACCENT,
    )

    return img


def render_splash(w: int, h: int) -> Image.Image:
    """Render del splash screen: fondo sólido + logo centrado."""
    img = Image.new("RGB", (w, h), BG)
    # El logo ocupa ~40% del lado corto del device. Lo rendereamos
    # standalone y lo paste centrado.
    side = int(min(w, h) * 0.40)
    logo = render_logo(side)
    x = (w - side) // 2
    y = (h - side) // 2
    img.paste(logo, (x, y))

    # Sub-label debajo del logo ("obsidian-rag") en tono dim.
    draw = ImageDraw.Draw(img)
    sub_font = _font(int(min(w, h) * 0.028))
    sub_text = "obsidian-rag"
    bbox = _measure(draw, sub_text, sub_font)
    sub_w = bbox[2] - bbox[0]
    sub_h = bbox[3] - bbox[1]
    draw.text(
        ((w - sub_w) // 2 - bbox[0], y + side + int(min(w, h) * 0.02) - bbox[1]),
        sub_text,
        fill=DIM,
        font=sub_font,
    )
    return img


# ── iPhone splash sizes (portrait) ─────────────────────────────────────
# Mapeo: (portrait_width, portrait_height, device_width_pt, device_height_pt,
# pixel_ratio, "human_name"). El media query de Apple usa los `pt` (css px)
# + device-pixel-ratio, NO el tamaño en `px`. El `px` es sólo para rendear
# el PNG en la resolución correcta.
IPHONE_SPLASHES: list[tuple[int, int, int, int, int, str]] = [
    (1125, 2436, 375, 812, 3, "iphone-x"),       # iPhone X, XS, 11 Pro, 12 mini (ojo: 12 mini es 1080 real)
    (828, 1792, 414, 896, 2, "iphone-xr"),       # iPhone XR, 11
    (1242, 2688, 414, 896, 3, "iphone-xs-max"),  # iPhone XS Max, 11 Pro Max
    (1080, 2340, 360, 780, 3, "iphone-12-mini"), # iPhone 12/13 mini
    (1170, 2532, 390, 844, 3, "iphone-12"),      # iPhone 12/12 Pro/13/13 Pro/14
    (1284, 2778, 428, 926, 3, "iphone-12-pro-max"), # iPhone 12/13 Pro Max, 14 Plus
    (1179, 2556, 393, 852, 3, "iphone-14-pro"),  # iPhone 14 Pro, 15, 15 Pro, 16
    (1290, 2796, 430, 932, 3, "iphone-14-pro-max"), # iPhone 14 Pro Max, 15 Plus/Pro Max, 16 Plus
    (1206, 2622, 402, 874, 3, "iphone-16-pro"),  # iPhone 16 Pro
    (1320, 2868, 440, 956, 3, "iphone-16-pro-max"), # iPhone 16 Pro Max
]


def gen_all() -> None:
    # ── PWA / Android icons (any-purpose) ─────────────────────────────
    for size in (192, 512):
        logo = render_logo(size)
        logo.save(OUT_DIR / f"icon-{size}.png", optimize=True)

    # ── Maskable icons (Android 12+, safe-zone spec = 80%) ────────────
    for size in (192, 512):
        logo = render_logo(size, safe_zone=0.8)
        logo.save(OUT_DIR / f"icon-{size}-maskable.png", optimize=True)

    # ── Apple touch icon (180×180, iOS home screen) ───────────────────
    # iOS no lee `icons` del manifest hasta iOS 16.4+. Para compat con
    # versiones viejas + consistencia, usamos el <link rel="apple-touch-icon">
    # clásico (180px es el nominal de iPhone 6 Plus+, funciona en toda la línea).
    render_logo(180).save(OUT_DIR / "apple-touch-icon.png", optimize=True)

    # ── Favicon (32×32, tab del browser desktop) ──────────────────────
    render_logo(32).save(OUT_DIR / "favicon-32.png", optimize=True)
    render_logo(16).save(OUT_DIR / "favicon-16.png", optimize=True)

    # ── iPhone splashes ───────────────────────────────────────────────
    for w, h, _, _, _, name in IPHONE_SPLASHES:
        splash = render_splash(w, h)
        splash.save(OUT_DIR / f"splash-{name}.png", optimize=True)

    print(f"[ok] Generados {len(list(OUT_DIR.glob('*.png')))} PNGs en {OUT_DIR}")


def gen_html_snippet() -> str:
    """Devuelve el snippet de <link rel=\"apple-touch-startup-image\">
    para cada device, con el media query correcto. Lo imprimimos al final
    para que el user lo pueda copiar al <head> o inyectar programáticamente.
    """
    lines = []
    for w, h, pt_w, pt_h, ratio, name in IPHONE_SPLASHES:
        media = (
            f"(device-width: {pt_w}px) and (device-height: {pt_h}px) "
            f"and (-webkit-device-pixel-ratio: {ratio}) and (orientation: portrait)"
        )
        lines.append(
            f'<link rel="apple-touch-startup-image" '
            f'href="/static/pwa/splash-{name}.png" '
            f'media="{media}">'
        )
    return "\n".join(lines)


if __name__ == "__main__":
    gen_all()
    if "--print-html" in sys.argv:
        print()
        print("=== HTML snippet (pegá en <head>) ===")
        print(gen_html_snippet())
