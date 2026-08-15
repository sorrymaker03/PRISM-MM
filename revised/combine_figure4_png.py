#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = ROOT / "bulk_pre_sc" / "main_figure_panels"
OUT_DIR = ROOT / "PRISM-MM" / "figures"
TMP_DIR = ROOT / "bulk_pre_sc" / "main_figure_source_data" / "_figure4_rendered_png"

PANELS = {
    "A": "figure4_A_heldout_drug_counts.pdf",
    "B": "figure4_B_heldout_program_shift.pdf",
    "C": "figure4_C_concordant_program_distributions.pdf",
    "D": "figure4_D_program_score_classifier.pdf",
    "E": "figure4_E_drug_program_shift.pdf",
    "F": "figure4_F_model_heldout_concordance.pdf",
}


def render_pdf(label: str, pdf_name: str) -> Image.Image:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_DIR / f"figure4_{label}_revised"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "280", str(PANEL_DIR / pdf_name), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Image.open(TMP_DIR / f"figure4_{label}_revised-1.png").convert("RGB")


def crop_white(image: Image.Image, margin: int = 28) -> Image.Image:
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
    if image.width == width:
        return image
    canvas = Image.new("RGB", (width, image.height), "white")
    canvas.paste(image, ((width - image.width) // 2, 0))
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = {label: crop_white(render_pdf(label, pdf)) for label, pdf in PANELS.items()}

    full_width = 3540
    row_gap = 120
    row_top = center_on_width(
        row_canvas([rendered["A"], rendered["B"], rendered["C"]], [1010, 1010, 1080], gap=115),
        full_width,
    )
    row_bottom = center_on_width(
        row_canvas([rendered["D"], rendered["E"], rendered["F"]], [1050, 1050, 1050], gap=95),
        full_width,
    )

    height = row_top.height + row_bottom.height + row_gap + 110
    canvas = Image.new("RGB", (full_width, height), "white")
    y = 55
    canvas.paste(row_top, (0, y))
    y += row_top.height + row_gap
    canvas.paste(row_bottom, (0, y))

    out = OUT_DIR / "figure4_combined.png"
    canvas.save(out, dpi=(300, 300))
    print(out)


if __name__ == "__main__":
    main()
