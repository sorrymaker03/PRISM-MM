#!/usr/bin/env python3
"""Patch revised Figure 3D/E panels into the existing combined Figure 3 PNG."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "bulk_pre_sc" / "main_figure_panels"
TMP_DIR = ROOT / "bulk_pre_sc" / "main_figure_source_data" / "_figure3_revised_de_rendered"
FIGURE = ROOT / "PRISM-MM" / "figures" / "figure3_combined.png"
BACKUP = ROOT / "PRISM-MM" / "figures" / "figure3_combined_before_study_panel_update.png"


def render_pdf(pdf_name: str, stem: str) -> Image.Image:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_DIR / stem
    out = TMP_DIR / f"{stem}-1.png"
    if out.exists():
        out.unlink()
    subprocess.run(
        ["pdftoppm", "-png", "-r", "280", str(PANEL_DIR / pdf_name), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Image.open(out).convert("RGB")


def crop_white(image: Image.Image, margin: int = 24) -> Image.Image:
    gray = image.convert("L")
    px = gray.load()
    w, h = gray.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(h):
        for x in range(w):
            if px[x, y] < 248:
                xs.append(x)
                ys.append(y)
    if not xs:
        return image
    return image.crop(
        (
            max(min(xs) - margin, 0),
            max(min(ys) - margin, 0),
            min(max(xs) + margin, w),
            min(max(ys) + margin, h),
        )
    )


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, int(round(image.height * ratio))), Image.Resampling.LANCZOS)


def crop_bottom_whitespace(image: Image.Image, margin: int = 55) -> Image.Image:
    gray = image.convert("L")
    w, h = gray.size
    last = h - 1
    for y in range(h - 1, -1, -1):
        nonwhite = 0
        for x in range(0, w, 10):
            if gray.getpixel((x, y)) < 248:
                nonwhite += 1
        if nonwhite > 5:
            last = y
            break
    return image.crop((0, 0, w, min(h, last + margin)))


def main() -> None:
    if FIGURE.exists() and not BACKUP.exists():
        shutil.copy2(FIGURE, BACKUP)

    base = Image.open(FIGURE).convert("RGB")
    d_panel = resize_to_width(crop_white(render_pdf("figure3_D_source_program_shift.pdf", "figure3_D")), 1120)
    e_panel = resize_to_width(crop_white(render_pdf("figure3_E_study_program_shift.pdf", "figure3_E")), 1120)

    # Coordinates match the existing Figure 3 layout. This keeps panels A-C
    # unchanged and only clears/replaces the D/E region.
    base.paste(Image.new("RGB", (2400, base.height - 4300), "white"), (900, 4300))

    row_top = 4418
    row_height = max(d_panel.height, e_panel.height)
    d_y = row_top + (row_height - d_panel.height) // 2
    e_y = row_top + (row_height - e_panel.height) // 2
    base.paste(d_panel, (955, d_y))
    base.paste(e_panel, (2170, e_y))

    out = crop_bottom_whitespace(base)
    out.save(FIGURE, dpi=(300, 300))
    print(FIGURE)


if __name__ == "__main__":
    main()
