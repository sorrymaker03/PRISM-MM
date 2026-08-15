#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "bulk_pre_sc" / "main_figure_panels"
OUT_DIR = ROOT / "PRISM-MM" / "figures"
TMP_DIR = ROOT / "bulk_pre_sc" / "main_figure_source_data" / "_figure2_rendered_png"

PANELS = {
    "A": "figure2_A_cosine.pdf",
    "B": "figure2_B_pearson.pdf",
    "C": "figure2_C_spearman.pdf",
    "D": "figure2_D_top50_overlap.pdf",
    "E": "figure2_E_top100_overlap.pdf",
    "F": "figure2_F_top100_sign_agreement.pdf",
    "G": "figure2_G_magnitude_fidelity.pdf",
    "H": "figure2_H_program_recovery_score.pdf",
}


def render_pdf(label: str, pdf_name: str) -> Image.Image:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_DIR / f"figure2_{label}_revised"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "280", str(PANEL_DIR / pdf_name), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Image.open(TMP_DIR / f"figure2_{label}_revised-1.png").convert("RGB")


def crop_white(image: Image.Image, margin: int = 24) -> Image.Image:
    gray = image.convert("L")
    px = gray.load()
    width, height = gray.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            if px[x, y] < 248:
                xs.append(x)
                ys.append(y)
    if not xs:
        return image
    return image.crop(
        (
            max(min(xs) - margin, 0),
            max(min(ys) - margin, 0),
            min(max(xs) + margin, width),
            min(max(ys) + margin, height),
        )
    )


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, int(round(image.height * ratio))), Image.Resampling.LANCZOS)


def row_canvas(images: list[Image.Image], widths: list[int], gap: int) -> Image.Image:
    resized = [resize_to_width(image, width) for image, width in zip(images, widths)]
    row_width = sum(image.width for image in resized) + gap * (len(resized) - 1)
    row_height = max(image.height for image in resized)
    row = Image.new("RGB", (row_width, row_height), "white")
    x = 0
    for image in resized:
        row.paste(image, (x, (row_height - image.height) // 2))
        x += image.width + gap
    return row


def center_on_width(image: Image.Image, width: int) -> Image.Image:
    canvas = Image.new("RGB", (width, image.height), "white")
    canvas.paste(image, ((width - image.width) // 2, 0))
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = {label: crop_white(render_pdf(label, pdf)) for label, pdf in PANELS.items()}

    full_width = 3300
    row_gap = 120
    row1 = center_on_width(row_canvas([rendered["A"], rendered["B"], rendered["C"]], [980, 980, 980], gap=120), full_width)
    row2 = center_on_width(row_canvas([rendered["D"], rendered["E"], rendered["F"]], [980, 980, 980], gap=120), full_width)
    row3 = center_on_width(row_canvas([rendered["G"], rendered["H"]], [980, 980], gap=240), full_width)

    height = row1.height + row2.height + row3.height + row_gap * 2 + 120
    canvas = Image.new("RGB", (full_width, height), "white")
    y = 60
    for row in (row1, row2, row3):
        canvas.paste(row, (0, y))
        y += row.height + row_gap

    out = OUT_DIR / "figure2_combined.png"
    canvas.save(out, dpi=(300, 300))
    print(out)


if __name__ == "__main__":
    main()
