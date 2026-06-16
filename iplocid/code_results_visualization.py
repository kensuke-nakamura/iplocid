#!/usr/bin/env python3
# code_results_visualization.py
#
# Summarize detected bboxes from:
#   results/<data_name>/generated_texts/<result_name>.json
#
# Raw output naming:
# - reference:
#     raw/data-<sample_id>_reference.png
# - query:
#     raw/data-<sample_id>_<image_id>_<role>_<result_name>.png
#
# Concatenated output naming:
# - legacy single concat:
#     data-<sample_id>_<result_name>.png
# - reference:
#     data-<sample_id>_reference.png
# - target:
#     data-<sample_id>_target_<result_name>.png
#
# Optional concatenation frame:
# - --frame-width controls the outer frame and internal separator width.
# - --frame-color controls the frame/separator color.
#
# English comments only.

import os
import re
import json
import ast
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageDraw

from tool_draw_boxes2 import concat_images_horiz


TARGET_HEIGHT_DEFAULT = 360
LINEWIDTH_DEFAULT = 12
LINEWIDTH_BASE_WIDTH = 640

_COLOR_PRESETS = {
    "red": (255, 0, 0),
    "yellowgreen": (154, 205, 50),
    "blue": (0, 0, 255),
    "orange": (255, 165, 0),
    "green": (0, 255, 0),
    "lime": (0, 255, 0),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "magenta": (255, 0, 255),
}


def _safe_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "model"


def _parse_image_id(x) -> int:
    try:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        return int(x)
    except Exception:
        return -1


def _normalize_role(role: str) -> str:
    r = str(role).strip().lower()
    if r in ("reference", "positive-image", "inclass-image", "outclass-image"):
        return r
    return r


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dirs(results_data_dir: Path) -> Tuple[Path, Path]:
    img2_dir = results_data_dir / "img2"
    raw_dir = img2_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    img2_dir.mkdir(parents=True, exist_ok=True)
    return img2_dir, raw_dir


def _pick_one_by_smallest_image_id(paths: List[Tuple[int, str]]) -> Optional[str]:
    if not paths:
        return None
    return sorted(paths, key=lambda t: (t[0], t[1]))[0][1]


def _role_order_queries_legacy() -> List[str]:
    return ["positive-image", "inclass-image", "outclass-image"]


def _bbox_to_iploc_str(v: Any) -> Optional[str]:
    if v is None:
        return None

    if isinstance(v, (list, tuple)) and len(v) >= 4:
        try:
            x1, y1, x2, y2 = v[:4]
            x1 = int(round(float(x1)))
            y1 = int(round(float(y1)))
            x2 = int(round(float(x2)))
            y2 = int(round(float(y2)))
            return f"[{x1}, {y1}, {x2}, {y2}]"
        except Exception:
            return None

    s = str(v).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return None
    if s.startswith("[") and s.endswith("]"):
        return s
    return f"[{s}]"


def _parse_color(s: str) -> Tuple[int, int, int]:
    if s is None:
        return _COLOR_PRESETS["red"]
    ss = str(s).strip().lower()
    if ss in _COLOR_PRESETS:
        return _COLOR_PRESETS[ss]

    if "," in ss:
        parts = [p.strip() for p in ss.split(",")]
        if len(parts) >= 3:
            try:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                return (r, g, b)
            except Exception:
                pass

    raise ValueError(f"Unknown color: {s}")


def _parse_box_str(box_str: str) -> Optional[List[int]]:
    if not box_str:
        return None
    try:
        v = ast.literal_eval(str(box_str))
        if not (isinstance(v, (list, tuple)) and len(v) == 4):
            return None
        x1, y1, x2, y2 = [int(round(float(x))) for x in v]
        return [x1, y1, x2, y2]
    except Exception:
        return None


def _compute_scaled_linewidth_from_image(image_path: str, base_linewidth: int) -> int:
    try:
        img = Image.open(image_path)
        w, _h = img.size
        if w <= 0:
            return max(1, int(base_linewidth))
        lw = int(round(float(base_linewidth) * (float(w) / float(LINEWIDTH_BASE_WIDTH))))
        return max(1, lw)
    except Exception:
        return max(1, int(base_linewidth))


def resize_png_inplace(path: str, target_h: int) -> None:
    if target_h <= 0:
        return
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return
    if h == target_h:
        img.save(path)
        return
    scale = float(target_h) / float(h)
    new_h = int(target_h)
    new_w = max(1, int(round(w * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    img.save(path)


def save_image_no_boxes(src_path: str, dst_path: str, target_h: int) -> None:
    img = Image.open(src_path).convert("RGB")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path)
    resize_png_inplace(dst_path, target_h=target_h)


def _draw_one_box(
    draw: ImageDraw.ImageDraw,
    box: List[int],
    color: Tuple[int, int, int],
    linewidth: int,
    image_size: Tuple[int, int],
) -> None:
    """
    Draw one solid bbox.
    Expand the bbox outward according to linewidth so that
    the stroke is less likely to occlude the object itself.
    Clamp to image bounds.
    """
    x1, y1, x2, y2 = box
    w, h = image_size
    lw = max(1, int(linewidth))

    # Expand bbox outward according to linewidth.
    # This follows the current visual policy: expand by one linewidth.
    expand = max(1, int(round(lw)))
    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    for t in range(lw):
        xx1 = max(0, x1 - t)
        yy1 = max(0, y1 - t)
        xx2 = min(w - 1, x2 + t)
        yy2 = min(h - 1, y2 + t)
        draw.rectangle([xx1, yy1, xx2, yy2], outline=color)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    linewidth: int,
    dash_len: int,
    gap_len: int,
) -> None:
    """Draw a horizontal or vertical dashed line."""
    x1, y1 = start
    x2, y2 = end
    lw = max(1, int(linewidth))

    if y1 == y2:
        if x2 < x1:
            x1, x2 = x2, x1
        x = x1
        step = dash_len + gap_len
        while x <= x2:
            xe = min(x + dash_len, x2)
            draw.line([(x, y1), (xe, y2)], fill=color, width=lw)
            x += step
    elif x1 == x2:
        if y2 < y1:
            y1, y2 = y2, y1
        y = y1
        step = dash_len + gap_len
        while y <= y2:
            ye = min(y + dash_len, y2)
            draw.line([(x1, y), (x2, ye)], fill=color, width=lw)
            y += step
    else:
        # This helper is intended for rectangle edges only.
        draw.line([start, end], fill=color, width=lw)


def _draw_dashed_box(
    draw: ImageDraw.ImageDraw,
    box: List[int],
    color: Tuple[int, int, int],
    linewidth: int,
    image_size: Tuple[int, int],
    dash_len: Optional[int] = None,
    gap_len: Optional[int] = None,
) -> None:
    """
    Draw one dashed bbox.
    Expand the bbox outward according to linewidth so that
    the stroke is less likely to occlude the object itself.
    Clamp to image bounds.
    """
    x1, y1, x2, y2 = box
    w, h = image_size
    lw = max(1, int(linewidth))

    expand = max(1, int(round(lw)))
    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if dash_len is None:
        dash_len = max(8, lw * 3)
    if gap_len is None:
        gap_len = max(5, lw * 2)

    _draw_dashed_line(draw, (x1, y1), (x2, y1), color, lw, dash_len, gap_len)
    _draw_dashed_line(draw, (x1, y2), (x2, y2), color, lw, dash_len, gap_len)
    _draw_dashed_line(draw, (x1, y1), (x1, y2), color, lw, dash_len, gap_len)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), color, lw, dash_len, gap_len)


def save_image_with_boxes(
    src_path: str,
    dst_path: str,
    boxes: List[str],
    color: Tuple[int, int, int],
    base_linewidth: int,
    target_h: int,
    dashed_boxes: Optional[List[str]] = None,
    dashed_color: Tuple[int, int, int] = (0, 0, 255),
) -> None:
    """
    Draw solid boxes first, optionally draw dashed boxes on top,
    then resize the saved image to target_h.
    """
    im = Image.open(src_path).convert("RGB")
    lw_eff = _compute_scaled_linewidth_from_image(src_path, base_linewidth=base_linewidth)
    draw = ImageDraw.Draw(im)

    for b in boxes:
        bb = _parse_box_str(b)
        if bb is None:
            continue
        _draw_one_box(draw, bb, color=color, linewidth=lw_eff, image_size=im.size)

    if dashed_boxes:
        for b in dashed_boxes:
            bb = _parse_box_str(b)
            if bb is None:
                continue
            _draw_dashed_box(
                draw,
                bb,
                color=dashed_color,
                linewidth=lw_eff,
                image_size=im.size,
            )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    im.save(dst_path)
    resize_png_inplace(dst_path, target_h=target_h)



def concat_images_horiz_with_frame(
    paths: List[str],
    out_path: str,
    frame_width: int = 0,
    frame_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Concatenate images horizontally.
    If frame_width > 0, draw an outer frame and internal separators with the same width.
    """
    if int(frame_width) <= 0:
        concat_images_horiz(paths, out_path)
        return

    if not paths:
        raise ValueError("paths must not be empty")

    fw = max(1, int(frame_width))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]

    content_h = max(heights)
    total_w = sum(widths) + fw * (len(imgs) + 1)
    total_h = content_h + fw * 2

    canvas = Image.new("RGB", (total_w, total_h), frame_color)

    x = fw
    for im in imgs:
        y = fw + (content_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + fw

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

def concat_images_grid(
    paths: List[str],
    nrows: int,
    ncols: int,
    out_path: str,
    frame_width: int = 0,
    frame_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """
    Concatenate images into a grid.
    If frame_width > 0, draw an outer frame and internal separators with the same width.
    """
    if len(paths) != nrows * ncols:
        raise ValueError(f"grid expects {nrows*ncols} images, got {len(paths)}")

    imgs: List[Image.Image] = [Image.open(p).convert("RGB") for p in paths]

    col_w = [0] * ncols
    row_h = [0] * nrows
    for r in range(nrows):
        for c in range(ncols):
            im = imgs[r * ncols + c]
            w, h = im.size
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)

    fw = max(0, int(frame_width))
    if fw > 0:
        total_w = sum(col_w) + fw * (ncols + 1)
        total_h = sum(row_h) + fw * (nrows + 1)
        canvas = Image.new("RGB", (total_w, total_h), frame_color)

        y = fw
        for r in range(nrows):
            x = fw
            for c in range(ncols):
                im = imgs[r * ncols + c]
                ox = (col_w[c] - im.size[0]) // 2
                oy = (row_h[r] - im.size[1]) // 2
                canvas.paste(im, (x + ox, y + oy))
                x += col_w[c] + fw
            y += row_h[r] + fw
    else:
        total_w = sum(col_w)
        total_h = sum(row_h)
        canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))

        y = 0
        for r in range(nrows):
            x = 0
            for c in range(ncols):
                im = imgs[r * ncols + c]
                ox = (col_w[c] - im.size[0]) // 2
                oy = (row_h[r] - im.size[1]) // 2
                canvas.paste(im, (x + ox, y + oy))
                x += col_w[c]
            y += row_h[r]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def _reference_png_path(raw_dir: Path, sample_id: int) -> Path:
    return raw_dir / f"data-{sample_id}_reference.png"


def _query_png_path(raw_dir: Path, result_name: str, sample_id: int, image_id: int, role: str) -> Path:
    return raw_dir / f"data-{sample_id}_{image_id}_{role}_{result_name}.png"


def process_one_result_json(
    results_data_dir: Path,
    result_json_path: Path,
    num_samples: int,
    target_h: int,
    base_linewidth: int,
    color_reference: Tuple[int, int, int],
    color_positive: Tuple[int, int, int],
    color_inclass: Tuple[int, int, int],
    color_outclass: Tuple[int, int, int],
    overwrite: bool,
    show_positive_gt: bool,
    frame_width: int,
    frame_color: Tuple[int, int, int],
) -> None:
    img2_dir, raw_dir = _ensure_dirs(results_data_dir)
    result_name = _safe_name(result_json_path.stem)

    payload = _read_json(result_json_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Result JSON is not a dict: {result_json_path}")

    outputs = payload.get("outputs", None)
    if not isinstance(outputs, list):
        raise ValueError(f"Invalid result json format (missing outputs list): {result_json_path}")

    by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for e in outputs:
        if not isinstance(e, dict):
            continue
        try:
            sid = int(e.get("sample"))
        except Exception:
            continue
        if sid < 0:
            continue
        by_sample.setdefault(sid, []).append(e)

    sample_ids = sorted(by_sample.keys())
    if not sample_ids:
        return

    max_samples = (max(sample_ids) + 1) if int(num_samples) <= 0 else int(num_samples)

    for sample_id in range(max_samples):
        entries = by_sample.get(sample_id, [])
        if not entries:
            continue

        trigger = None
        for e in entries:
            if _parse_image_id(e.get("image_id", -1)) == 0:
                trigger = e
                break
        if trigger is None:
            trigger = sorted(entries, key=lambda x: _parse_image_id(x.get("image_id", -1)))[0]

        refs = trigger.get("references", [])
        if isinstance(refs, list):
            for r in refs:
                if not isinstance(r, dict):
                    continue

                ref_img_path = str(r.get("image_path", ""))
                ref_gt_bbox = _bbox_to_iploc_str(r.get("gt_bbox_pixel_format", None))
                if not ref_img_path:
                    continue

                out_png = _reference_png_path(raw_dir, sample_id)
                if out_png.exists() and (not overwrite):
                    continue

                try:
                    if ref_gt_bbox is None:
                        save_image_no_boxes(ref_img_path, str(out_png), target_h=target_h)
                    else:
                        save_image_with_boxes(
                            ref_img_path,
                            str(out_png),
                            boxes=[ref_gt_bbox],
                            color=color_reference,
                            base_linewidth=base_linewidth,
                            target_h=target_h,
                        )
                except Exception as ex:
                    print(f"[WARN] reference save failed: sample={sample_id} err={ex}")
                    save_image_no_boxes(ref_img_path, str(out_png), target_h=target_h)

        for e in entries:
            role = _normalize_role(e.get("role", ""))
            if role not in ("positive-image", "inclass-image", "outclass-image"):
                continue

            img_path = str(e.get("image_path", ""))
            if not img_path:
                continue

            img_id = _parse_image_id(e.get("image_id", -1))
            pn_label = str(e.get("pn_label", "")).strip().lower()
            pred_bbox = _bbox_to_iploc_str(e.get("pred_bbox_pixel_format", None))

            out_png = _query_png_path(raw_dir, result_name, sample_id, img_id, role)
            if out_png.exists() and (not overwrite):
                continue

            role_color = color_positive
            if role == "inclass-image":
                role_color = color_inclass
            elif role == "outclass-image":
                role_color = color_outclass

            try:
                if pn_label == "positive" and pred_bbox is not None:
                    gt_bbox = _bbox_to_iploc_str(e.get("gt_bbox_pixel_format", None))
                    dashed_boxes: List[str] = []
                    if show_positive_gt and role == "positive-image" and gt_bbox is not None:
                        dashed_boxes = [gt_bbox]

                    save_image_with_boxes(
                        img_path,
                        str(out_png),
                        boxes=[pred_bbox],
                        color=role_color,
                        base_linewidth=base_linewidth,
                        target_h=target_h,
                        dashed_boxes=dashed_boxes,
                        dashed_color=_COLOR_PRESETS["blue"],
                    )
                else:
                    save_image_no_boxes(img_path, str(out_png), target_h=target_h)
            except Exception as ex:
                print(f"[WARN] query save failed: sample={sample_id} image_id={img_id} role={role} err={ex}")
                save_image_no_boxes(img_path, str(out_png), target_h=target_h)

    for sample_id in range(max_samples):
        entries = by_sample.get(sample_id, [])
        if not entries:
            continue

        ref_path = _reference_png_path(raw_dir, sample_id)
        if not ref_path.exists():
            continue
        ref_list_sorted = [str(ref_path)]

        q_entries_inorder = [
            e for e in entries
            if _normalize_role(e.get("role", "")) in ("positive-image", "inclass-image", "outclass-image")
        ]
        q_entries_sorted = sorted(q_entries_inorder, key=lambda x: _parse_image_id(x.get("image_id", -1)))
        T = len(q_entries_inorder)

        if T == 5:
            q_paths: List[str] = []
            for e in q_entries_inorder:
                role = _normalize_role(e.get("role", ""))
                img_id = _parse_image_id(e.get("image_id", -1))
                p = _query_png_path(raw_dir, result_name, sample_id, img_id, role)
                if p.exists():
                    q_paths.append(str(p))
                else:
                    print(
                        f"[ERROR] Missing raw query image for T=5: "
                        f"sample={sample_id} image_id={img_id} role={role} path={p}"
                    )
                    q_paths = []
                    break

            if len(q_paths) != 5:
                print(f"[ERROR] T=5 expects 5 query images but got {len(q_paths)} at sample={sample_id}. Skip.")
                continue

            concat_list = ref_list_sorted + q_paths
            out_concat = img2_dir / f"data-{sample_id}_{result_name}.png"
            if overwrite or (not out_concat.exists()):
                concat_images_horiz_with_frame(concat_list, str(out_concat), frame_width=frame_width, frame_color=frame_color)
            continue

        if T in (1, 2, 4):
            query_candidates = [
                p for p in raw_dir.iterdir()
                if p.is_file()
                and p.name.startswith(f"data-{sample_id}_")
                and p.suffix.lower() == ".png"
                and (not p.name.endswith("_reference.png"))
                and p.name.endswith(f"_{result_name}.png")
            ]

            per_role: Dict[str, List[Tuple[int, str]]] = {}
            prefix = f"data-{sample_id}_"
            suffix = f"_{result_name}.png"

            for p in query_candidates:
                name = p.name
                core = name[len(prefix):-len(suffix)]
                first_us = core.find("_")
                if first_us < 0:
                    continue

                img_id_str = core[:first_us]
                role_str = core[first_us + 1:]
                role = _normalize_role(role_str)
                if role not in ("positive-image", "inclass-image", "outclass-image"):
                    continue

                img_id = _parse_image_id(img_id_str)
                per_role.setdefault(role, []).append((img_id, str(p)))

            query_list: List[str] = []
            for role in _role_order_queries_legacy():
                picked = _pick_one_by_smallest_image_id(per_role.get(role, []))
                if picked is not None:
                    query_list.append(picked)

            concat_list = ref_list_sorted + query_list
            if not concat_list:
                continue

            out_concat = img2_dir / f"data-{sample_id}_{result_name}.png"
            if overwrite or (not out_concat.exists()):
                concat_images_horiz_with_frame(concat_list, str(out_concat), frame_width=frame_width, frame_color=frame_color)
            continue

        if T in (6, 9):
            out_ref = img2_dir / f"data-{sample_id}_reference.png"
            if overwrite or (not out_ref.exists()):
                concat_images_horiz_with_frame(ref_list_sorted, str(out_ref), frame_width=frame_width, frame_color=frame_color)

            q_paths: List[str] = []
            for e in q_entries_sorted:
                role = _normalize_role(e.get("role", ""))
                img_id = _parse_image_id(e.get("image_id", -1))
                p = _query_png_path(raw_dir, result_name, sample_id, img_id, role)
                if p.exists():
                    q_paths.append(str(p))
                else:
                    print(
                        f"[ERROR] Missing raw query image for T={T}: "
                        f"sample={sample_id} image_id={img_id} role={role} path={p}"
                    )
                    q_paths = []
                    break

            if len(q_paths) != T:
                print(f"[ERROR] T={T} expects {T} query images but got {len(q_paths)} at sample={sample_id}. Skip.")
                continue

            out_q = img2_dir / f"data-{sample_id}_target_{result_name}.png"
            if overwrite or (not out_q.exists()):
                concat_images_horiz_with_frame(q_paths, str(out_q), frame_width=frame_width, frame_color=frame_color)
            continue

        if T == 18:
            out_ref = img2_dir / f"data-{sample_id}_reference.png"
            if overwrite or (not out_ref.exists()):
                concat_images_horiz_with_frame(ref_list_sorted, str(out_ref), frame_width=frame_width, frame_color=frame_color)

            in_entries = [e for e in q_entries_sorted if _normalize_role(e.get("role", "")) == "inclass-image"]
            out_entries = [e for e in q_entries_sorted if _normalize_role(e.get("role", "")) == "outclass-image"]
            po_entries = [e for e in q_entries_sorted if _normalize_role(e.get("role", "")) == "positive-image"]

            if len(po_entries) != 1 or len(in_entries) != 8 or len(out_entries) != 9:
                print(
                    f"[ERROR] T=18 role counts mismatch at sample={sample_id}: "
                    f"positive-image={len(po_entries)} inclass-image={len(in_entries)} outclass-image={len(out_entries)}."
                )
                continue

            def _imgid(e: Dict[str, Any]) -> int:
                return _parse_image_id(e.get("image_id", -1))

            in_paths = [_query_png_path(raw_dir, result_name, sample_id, _imgid(e), "inclass-image") for e in in_entries]
            out_paths = [_query_png_path(raw_dir, result_name, sample_id, _imgid(e), "outclass-image") for e in out_entries]
            po_path = _query_png_path(raw_dir, result_name, sample_id, _imgid(po_entries[0]), "positive-image")

            missing = [str(p) for p in (in_paths + out_paths + [po_path]) if not p.exists()]
            if missing:
                print(f"[ERROR] Missing raw query images for sample={sample_id}. Skip grid. Missing: {missing[:3]}...")
                continue

            grid_paths = [
                str(in_paths[0]), str(in_paths[1]), str(in_paths[2]), str(out_paths[0]), str(out_paths[1]), str(out_paths[2]),
                str(in_paths[3]), str(po_path), str(in_paths[4]), str(out_paths[3]), str(out_paths[4]), str(out_paths[5]),
                str(in_paths[5]), str(in_paths[6]), str(in_paths[7]), str(out_paths[6]), str(out_paths[7]), str(out_paths[8]),
            ]

            out_grid = img2_dir / f"data-{sample_id}_target_{result_name}.png"
            if overwrite or (not out_grid.exists()):
                concat_images_grid(grid_paths, nrows=3, ncols=6, out_path=str(out_grid), frame_width=frame_width, frame_color=frame_color)
            continue

        print(f"[ERROR] Unsupported T={T} at sample={sample_id}. Skip final concatenation.")
        continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_data_dir",
        type=str,
        help="Path to results/<data_name> directory (contains generated_texts/).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Only process samples with sample_id < num_samples. Use <=0 to process all.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=TARGET_HEIGHT_DEFAULT,
        help="Resize all saved PNGs to this fixed height (aspect ratio preserved).",
    )
    parser.add_argument(
        "--linewidth",
        type=int,
        default=LINEWIDTH_DEFAULT,
        help=f"BBox linewidth at base width {LINEWIDTH_BASE_WIDTH} (scaled by original image width).",
    )
    parser.add_argument("--color_reference", type=str, default="red")
    parser.add_argument("--color_target", type=str, default="YellowGreen")
    parser.add_argument("--color_inclass", type=str, default="magenta")
    parser.add_argument("--color_outclass", type=str, default="magenta")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images instead of skipping them.",
    )
    parser.add_argument(
        "--show_positive_gt",
        action="store_true",
        help="If set, draw GT bbox for positive-image as a blue dashed box on top of predicted bbox.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=6,
        help="Frame width for concatenated output images. Use 0 to disable framing.",
    )
    parser.add_argument(
        "--frame-color",
        type=str,
        default="black",
        help="Frame color for concatenated output images. Preset name or 'R,G,B'.",
    )

    args = parser.parse_args()

    results_data_dir = Path(args.results_data_dir).expanduser().resolve()
    gen_dir = results_data_dir / "generated_texts"
    if not gen_dir.is_dir():
        raise SystemExit(f"[ERROR] generated_texts dir not found: {gen_dir}")

    json_files = sorted([p for p in gen_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    if not json_files:
        raise SystemExit(f"[ERROR] No result json found under: {gen_dir}")

    target_h = int(args.target_height)
    base_linewidth = int(args.linewidth)

    c_ref = _parse_color(args.color_reference)
    c_pos = _parse_color(args.color_target)
    c_in = _parse_color(args.color_inclass)
    c_out = _parse_color(args.color_outclass)
    c_frame = _parse_color(args.frame_color)
    frame_width = max(0, int(args.frame_width))

    for p in json_files:
        print(f"[INFO] Processing: {p}")
        process_one_result_json(
            results_data_dir=results_data_dir,
            result_json_path=p,
            num_samples=int(args.num_samples),
            target_h=target_h,
            base_linewidth=base_linewidth,
            color_reference=c_ref,
            color_positive=c_pos,
            color_inclass=c_in,
            color_outclass=c_out,
            overwrite=bool(args.overwrite),
            show_positive_gt=bool(args.show_positive_gt),
            frame_width=frame_width,
            frame_color=c_frame,
        )

    print(f"[INFO] Done. Images saved under: {results_data_dir / 'img2'}")


if __name__ == "__main__":
    main()
