#!/usr/bin/env python3
"""Patch revised Figure 5E into the existing combined Figure 5 PNG."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "bulk_pre_sc" / "main_figure_panels"
TMP_DIR = ROOT / "bulk_pre_sc" / "main_figure_source_data" / "_figure5e_revised_rendered"
FIGURE = ROOT / "PRISM-MM" / "figures" / "figure5_combined.png"
BACKUP = ROOT / "PRISM-MM" / "figures" / "figure5_combined_before_yaxis_update.png"


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


def main() -> None:
    if FIGURE.exists() and not BACKUP.exists():
        shutil.copy2(FIGURE, BACKUP)

    base = Image.open(FIGURE).convert("RGB")
    e_panel = crop_white(render_pdf("figure5_E_external_pair_heatmap.pdf", "figure5_E"))
    e_panel = resize_to_width(e_panel, 910)

    # Clear only the original panel E footprint plus a small margin for the new
    # y-axis label; keep panels D and F unchanged.
    clear_x, clear_y = 1115, 1025
    clear_w, clear_h = 1000, 920
    base.paste(Image.new("RGB", (clear_w, clear_h), "white"), (clear_x, clear_y))
    base.paste(e_panel, (1138, 1035))
    base.save(FIGURE, dpi=(300, 300))
    print(FIGURE)


if __name__ == "__main__":
    main()
