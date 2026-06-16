#!/usr/bin/env python3
# tool_draw_boxes2.py
#
# Minimal bbox drawing utility (no role logic).
#
# - draw_boxes2(): draw one or more boxes on an image with a caller-specified color.
# - concat_images_horiz(): same behavior as before (kept for convenience).
#
# English comments only.

from __future__ import annotations

import os
import ast
import argparse
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


RGB = List[int]


def _normalize_boxes(boxes: Any) -> List[List[float]]:
    """
    Normalize input `boxes` to a list of [x1, y1, x2, y2] float boxes.

    Accepts:
      - None
      - [x1,y1,x2,y2]
      - [[...], [...]]
      - string like "[x1,y1,x2,y2]" or list containing such strings
    """
    if boxes is None:
        return []

    raw = boxes

    # Single box: [x1, y1, x2, y2]
    if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(
        not isinstance(v, (list, tuple)) for v in raw
    ):
        raw = [raw]

    # Single string: "[x1, y1, x2, y2]"
    if isinstance(raw, (str, bytes)):
        try:
            parsed = ast.literal_eval(raw)
            raw = parsed
            if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(
                not isinstance(v, (list, tuple)) for v in raw
            ):
                raw = [raw]
        except Exception:
            return []

    norm: List[List[float]] = []
    if isinstance(raw, (list, tuple)):
        for b in raw:
            if b is None:
                continue
            if isinstance(b, (str, bytes)):
                try:
                    b = ast.literal_eval(b)
                except Exception:
                    continue
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in b]
                norm.append([x1, y1, x2, y2])
            except Exception:
                continue

    return norm


def _to_rgb(color: Any) -> Tuple[int, int, int]:
    """Convert color input to (R,G,B) tuple."""
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r = int(color[0])
            g = int(color[1])
            b = int(color[2])
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return (r, g, b)
        except Exception:
            pass
    return (255, 0, 0)


def _draw_rect_solid(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for t in range(thickness):
        draw.rectangle(
            [x1 - t, y1 - t, x2 + t, y2 + t],
            outline=color,
        )


def draw_boxes2(
    image_path: str,
    boxes: Any,
    filename: Optional[str] = None,
    color: RGB = [255, 0, 0],
) -> str:
    """
    Draw one or more boxes on an image.

    Args:
      image_path: input image path
      boxes: bbox input (see _normalize_boxes)
      filename: output image path. If None, save under ./debug/
      color: RGB list, e.g. [255,0,0] for red

    Returns:
      output filename
    """
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    norm_boxes = _normalize_boxes(boxes)
    col = _to_rgb(color)

    # Fixed thickness
    thickness = max(5, round(min(im.size) * 0.01))

    for b in norm_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        _draw_rect_solid(draw, x1, y1, x2, y2, col, thickness)

    if filename is None:
        outdir = "./debug"
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, "box_" + os.path.basename(image_path))

    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    im.save(filename)
    return filename


def concat_images_horiz(image_paths: Sequence[str], out_path: str) -> None:
    """
    Concatenate images horizontally and save to out_path.
    All images are resized to have the same height (the maximum height).
    """
    imgs = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
        except Exception as e:
            print(f"[WARN] failed to open {p}: {e}")

    if not imgs:
        print("[WARN] concat_images_horiz: no valid images, skip.")
        return

    heights = [im.size[1] for im in imgs]
    max_h = max(heights)

    resized = []
    for im in imgs:
        w, h = im.size
        if h != max_h:
            new_w = int(round(w * (max_h / h)))
            im = im.resize((new_w, max_h))
        resized.append(im)

    total_w = sum(im.size[0] for im in resized)
    canvas = Image.new("RGB", (total_w, max_h), (0, 0, 0))

    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    outdir = os.path.dirname(out_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    canvas.save(out_path)


def _parse_color_csv(s: str) -> List[int]:
    """Parse 'R,G,B' into [R,G,B]."""
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) < 3:
        raise ValueError("color must be like '255,0,0'")
    return [int(parts[0]), int(parts[1]), int(parts[2])]


if __name__ == "__main__":
    # Minimal CLI demo
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to an input image")
    parser.add_argument(
        "--boxes",
        type=str,
        required=True,
        help="BBox spec, e.g. '[x1,y1,x2,y2]' or '[[...],[...]]'",
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    parser.add_argument("--color", type=str, default="255,0,0", help="RGB like '255,0,0'")
    args = parser.parse_args()

    color = _parse_color_csv(args.color)
    out_path = draw_boxes2(
        image_path=args.image_path,
        boxes=args.boxes,
        filename=args.out,
        color=color,
    )
    print(out_path)
