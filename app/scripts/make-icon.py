#!/usr/bin/env python3
"""Turn icon.png into a transparent logo and a menu bar template image.

The source is a flat two-colour mark (maroon glyph on cream). Rather than
threshold it — which would leave the curves jagged — each pixel is projected
onto the line between the two colours, and that position becomes its alpha.
Anti-aliased edge pixels are partly along that line, so they come out partly
transparent and the curves stay smooth.

Outputs:
  app/Resources/logo.png            full-colour, transparent background
  app/Resources/MenuBarIcon.png     template mask, 18 px tall  (@1x)
  app/Resources/MenuBarIcon@2x.png  template mask, 36 px tall  (@2x)

Menu bar icons are template images: macOS ignores their colour and re-tints
them from the alpha channel, so they follow the menu bar in light and dark mode
exactly as the old "SM" text did.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "icon.png"
RESOURCES = ROOT / "app" / "Resources"

# Menu bar glyphs sit around 18 pt tall inside a 22 pt bar.
MENU_BAR_HEIGHT = 18
PADDING_RATIO = 0.04


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def dominant_colour(pixels: np.ndarray) -> np.ndarray:
    colours, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    return colours[counts.argmax()].astype(np.float32)


def separate_colours(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The background colour and the ink colour, as the two dominant tones.

    Both are taken by frequency rather than by extremes: the source has JPEG
    ringing that leaves a scatter of near-black speckles inside the glyph, and
    picking "the colour farthest from the background" lands on those instead of
    the actual ink.
    """
    background = dominant_colour(rgb[:8, :8])
    distance = np.linalg.norm(rgb - background, axis=-1)
    ink = dominant_colour(rgb[distance >= distance.max() * 0.5])
    return background, ink


def alpha_from_projection(rgb: np.ndarray, background: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """How far each pixel lies along background → ink, as 0..1."""
    axis = ink - background
    length_squared = float(axis @ axis)
    if length_squared == 0:
        raise SystemExit("icon.png looks like a single flat colour")
    t = ((rgb - background) @ axis) / length_squared
    t = np.clip(t, 0.0, 1.0)
    # The background is not perfectly flat — it dithers between two near-identical
    # creams — so bare background lands a hair above zero. Left alone that is an
    # invisible film of alpha across the whole canvas, which defeats the crop and
    # leaves a faint wash behind the mark.
    t[t < 0.03] = 0.0
    return t


def crop_to_ink(image: Image.Image, padding_ratio: float = PADDING_RATIO) -> Image.Image:
    """Trim the empty margin so the glyph fills the space it is given."""
    box = image.getchannel("A").getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    pad = int(round(max(right - left, bottom - top) * padding_ratio))
    return image.crop((
        max(0, left - pad),
        max(0, top - pad),
        min(image.width, right + pad),
        min(image.height, bottom + pad),
    ))


def scaled_to_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.LANCZOS)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")

    rgb = load_rgb(SOURCE)
    background, ink = separate_colours(rgb)
    alpha = alpha_from_projection(rgb, background, ink)

    height, width = alpha.shape
    alpha_bytes = (alpha * 255).round().astype(np.uint8)

    # Full-colour logo: solid ink everywhere, shaped by the alpha mask. Painting
    # the ink colour flat (rather than keeping the original pixels) stops the
    # cream background bleeding into the edge pixels as a pale halo.
    colour = np.zeros((height, width, 4), dtype=np.uint8)
    colour[..., :3] = ink.round().astype(np.uint8)
    colour[..., 3] = alpha_bytes
    logo = crop_to_ink(Image.fromarray(colour, mode="RGBA"))

    RESOURCES.mkdir(parents=True, exist_ok=True)
    logo.save(RESOURCES / "logo.png")

    # Template mask: black ink, same alpha. macOS re-tints from alpha alone.
    template_pixels = np.zeros((height, width, 4), dtype=np.uint8)
    template_pixels[..., 3] = alpha_bytes
    template = crop_to_ink(Image.fromarray(template_pixels, mode="RGBA"))

    scaled_to_height(template, MENU_BAR_HEIGHT).save(RESOURCES / "MenuBarIcon.png")
    scaled_to_height(template, MENU_BAR_HEIGHT * 2).save(RESOURCES / "MenuBarIcon@2x.png")

    print(f"background {background.astype(int).tolist()}  ink {ink.round().astype(int).tolist()}")
    print(f"logo.png            {logo.width}x{logo.height}")
    for name in ("MenuBarIcon.png", "MenuBarIcon@2x.png"):
        with Image.open(RESOURCES / name) as img:
            print(f"{name:<20}{img.width}x{img.height}")


if __name__ == "__main__":
    main()
