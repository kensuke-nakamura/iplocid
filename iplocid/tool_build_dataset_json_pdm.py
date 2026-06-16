#!/usr/bin/env python3
# tool_build_dataset_json_pdm.py
#
# Build PDM (TAO/BURST-based) IPLoc-style JSONs from:
# - Teacher IPLoc JSON (explicitly N=2, T=1): provides (sequence, frames) and teacher bboxes/elements
# - BURST annotations JSON: provides per-track masks (RLE) and categories
# - BURST frames root: provides image files
#
# Outputs (2 configs):
# 1) N=2, T=2: [reference, reference, positive-image, outclass-image]
# 2) N=1, T=2: [reference, positive-image, outclass-image]
#
# For both configs, outclass-image is sampled from a RANDOM OTHER
# sequence's LAST annotated frame.
#
# IMPORTANT POLICY (explicit):
# - The sequence (BURST split/dataset/seq + the 3 teacher frames) follows the TEACHER JSON.
# - The object (track) inside the sequence is re-decided by these criteria:
#   (1) Prefer tracks whose category name matches the teacher "element" (if possible).
#   (2) Among tracks that appear in BOTH the first and the last teacher frames,
#       choose the track that maximizes IoU(first_bbox, teacher_first_bbox) + IoU(last_bbox, teacher_last_bbox).
#       If no category-matching candidate exists, fall back to all candidates.
#   (3) For the mid frame:
#       - If the chosen primary track exists in the mid frame, use it.
#       - Otherwise, choose a mid-frame track with the same category as the chosen primary track
#         that maximizes IoU(mid_bbox, teacher_mid_bbox). If none exists, the sample is skipped.
#
# Outclass policy:
# - For T=2, outclass-image is sampled from the LAST annotated frame of a RANDOM OTHER sequence.
# - Teacher sequences are excluded from outclass sampling (by default).
# - The sampled outclass object must have a category different from the teacher element.
#
# English comments only.

from __future__ import annotations

import argparse
import json
import random
import re
import os
import shutil
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_utils

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# -----------------------------
# Basic parsing / geometry utils
# -----------------------------

def parse_seq_from_image_path(p: str) -> Tuple[str, str, str, str]:
    # English comments only
    p = str(p).replace("\\", "/")
    m = re.search(r"/frames/(train|val|test)/([^/]+)/([^/]+)/([^/]+)$", p)
    if not m:
        raise ValueError(f"Cannot parse BURST-like frame path: {p}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


def rle_str_to_mask(rle_str: str, h: int, w: int) -> np.ndarray:
    rle_obj = {"size": [h, w], "counts": rle_str}
    m = mask_utils.decode(rle_obj)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def mask_to_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return (-1, -1, -1, -1)
    # Inclusive-style max corner based on mask pixels.
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_to_str(b: Tuple[int, int, int, int]) -> str:
    return f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"


def parse_bbox_str_xyxy(b: str) -> Optional[Tuple[float, float, float, float]]:
    if b is None:
        return None
    nums = _FLOAT_RE.findall(str(b))
    if len(nums) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(nums[i]) for i in range(4)]
    except Exception:
        return None
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)

    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def track_id_int(t: str) -> int:
    try:
        return int(t)
    except Exception:
        return 10 ** 18


# -----------------------------
# BURST access helpers
# -----------------------------

def build_seq_index(burst_ann: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    split = str(burst_ann.get("split", ""))
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for seq in burst_ann.get("sequences", []):
        ds = str(seq.get("dataset"))
        sn = str(seq.get("seq_name"))
        index[(split, ds, sn)] = seq
    return index


def get_bbox_xyxy_for_track_at_frame(
    segs: List[Dict[str, Any]],
    fi: int,
    track_id: str,
    h: int,
    w: int,
) -> Optional[Tuple[int, int, int, int]]:
    if fi < 0 or fi >= len(segs):
        return None
    frame_seg = segs[fi]
    if not isinstance(frame_seg, dict) or track_id not in frame_seg:
        return None
    rec = frame_seg[track_id]
    if not isinstance(rec, dict) or "rle" not in rec:
        return None
    mask = rle_str_to_mask(str(rec["rle"]), h, w)
    bbox = mask_to_bbox_xyxy(mask)
    if bbox[0] < 0:
        return None
    return bbox


def _track_category_name(
    track_to_cat: Dict[str, Any],
    cat_id_to_name: Dict[int, str],
    tid: str,
) -> Optional[str]:
    if not isinstance(track_to_cat, dict) or tid not in track_to_cat:
        return None
    try:
        cid = int(track_to_cat[tid])
    except Exception:
        return None
    return cat_id_to_name.get(cid, str(cid))


# -----------------------------
# Teacher-guided track selection
# -----------------------------

def choose_primary_track_by_teacher_iou(
    segs: List[Dict[str, Any]],
    fi0: int,
    fi2: int,
    h: int,
    w: int,
    track_to_cat: Dict[str, Any],
    cat_id_to_name: Dict[int, str],
    teacher_element: str,
    teacher_bbox0: Tuple[float, float, float, float],
    teacher_bbox2: Tuple[float, float, float, float],
) -> Optional[str]:
    # Candidates must exist in BOTH first and last frames.
    keys0 = set(segs[fi0].keys()) if (0 <= fi0 < len(segs) and isinstance(segs[fi0], dict)) else set()
    keys2 = set(segs[fi2].keys()) if (0 <= fi2 < len(segs) and isinstance(segs[fi2], dict)) else set()
    candidates = sorted(list(keys0 & keys2), key=track_id_int)
    if not candidates:
        return None

    cat_matched = [
        tid for tid in candidates
        if (_track_category_name(track_to_cat, cat_id_to_name, tid) == teacher_element)
    ]
    pool = cat_matched if len(cat_matched) > 0 else candidates

    best_tid = None
    best_score = -1.0
    for tid in pool:
        b0 = get_bbox_xyxy_for_track_at_frame(segs, fi0, tid, h, w)
        b2 = get_bbox_xyxy_for_track_at_frame(segs, fi2, tid, h, w)
        if b0 is None or b2 is None:
            continue
        score = iou_xyxy(tuple(map(float, b0)), teacher_bbox0) + iou_xyxy(tuple(map(float, b2)), teacher_bbox2)
        if score > best_score:
            best_score = score
            best_tid = tid

    return best_tid


def choose_mid_track_by_teacher_iou(
    segs: List[Dict[str, Any]],
    fi1: int,
    h: int,
    w: int,
    track_to_cat: Dict[str, Any],
    cat_id_to_name: Dict[int, str],
    primary_tid: str,
    teacher_bbox1: Tuple[float, float, float, float],
) -> Optional[str]:
    # If primary exists in mid, use it.
    keys1 = set(segs[fi1].keys()) if (0 <= fi1 < len(segs) and isinstance(segs[fi1], dict)) else set()
    if primary_tid in keys1:
        return primary_tid

    primary_cat_name = _track_category_name(track_to_cat, cat_id_to_name, primary_tid)
    if primary_cat_name is None:
        return None

    # Same-category candidates that exist in mid.
    candidates: List[str] = []
    for tid in keys1:
        if _track_category_name(track_to_cat, cat_id_to_name, tid) == primary_cat_name:
            candidates.append(tid)
    if not candidates:
        return None

    best_tid = None
    best_iou = -1.0
    for tid in sorted(candidates, key=track_id_int):
        b1 = get_bbox_xyxy_for_track_at_frame(segs, fi1, tid, h, w)
        if b1 is None:
            continue
        score = iou_xyxy(tuple(map(float, b1)), teacher_bbox1)
        if score > best_iou:
            best_iou = score
            best_tid = tid

    return best_tid


# -----------------------------
# Outclass sampling (T=2)
# -----------------------------

@dataclass(frozen=True)
class SeqKey:
    split: str
    dataset: str
    seq_name: str


def _collect_teacher_seq_keys(teacher: List[Dict[str, Any]]) -> set[Tuple[str, str, str]]:
    keys: set[Tuple[str, str, str]] = set()
    for item in teacher:
        img_paths = item.get("image_path")
        if not isinstance(img_paths, list) or len(img_paths) == 0:
            continue
        try:
            sp, ds, sn, _fr = parse_seq_from_image_path(img_paths[0])
        except Exception:
            continue
        keys.add((sp, ds, sn))
    return keys


def _pick_outclass_from_sequence_last_frame(
    seq_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    frames_base: Path,
    burst_split: str,
    cat_id_to_name: Dict[int, str],
    exclude_teacher_keys: set[Tuple[str, str, str]],
    exclude_cat_name: str,
    rng: random.Random,
    stop_on_missing_outclass_image: bool,
    max_tries: int = 500,
) -> Optional[Tuple[str, str]]:
    """Return (out_image_path_str, out_bbox_str) or None."""
    # English comments only
    all_keys = [k for k in seq_index.keys() if k[0] == burst_split]
    if not all_keys:
        return None

    for _ in range(max_tries):
        k = rng.choice(all_keys)
        if k in exclude_teacher_keys:
            continue

        seq = seq_index.get(k)
        if seq is None:
            continue

        ann_paths = seq.get("annotated_image_paths", [])
        segs = seq.get("segmentations", [])
        track_to_cat = seq.get("track_category_ids", {})
        if not isinstance(ann_paths, list) or not isinstance(segs, list) or len(ann_paths) == 0:
            continue

        # Last annotated frame
        frame_file = str(ann_paths[-1])
        fi = len(ann_paths) - 1
        if fi < 0 or fi >= len(segs):
            continue

        frame_seg = segs[fi]
        if not isinstance(frame_seg, dict) or len(frame_seg) == 0:
            continue

        h = int(seq.get("height", 0))
        w = int(seq.get("width", 0))
        if h <= 0 or w <= 0:
            continue

        # Filter tracks by category != exclude_cat_name
        candidate_tids: List[str] = []
        for tid in frame_seg.keys():
            cat_name = _track_category_name(track_to_cat, cat_id_to_name, str(tid))
            if cat_name is None:
                continue
            if str(cat_name) == str(exclude_cat_name):
                continue
            candidate_tids.append(str(tid))

        if not candidate_tids:
            continue

        tid = rng.choice(candidate_tids)
        bbox_xyxy = get_bbox_xyxy_for_track_at_frame(segs, fi, tid, h, w)
        if bbox_xyxy is None:
            continue

        out_img_path = frames_base / burst_split / str(k[1]) / str(k[2]) / frame_file

        if not out_img_path.exists():
            if stop_on_missing_outclass_image:
                raise FileNotFoundError(f"Missing outclass image: {out_img_path}")
            continue

        return str(out_img_path), bbox_to_str(bbox_xyxy)

    return None


# -----------------------------
# Dataset build (T=2 only)
# -----------------------------

def _append_item(
    out_list: List[Dict[str, Any]],
    element: str,
    image_paths: List[str],
    bboxes: List[str],
    roles: List[str],
) -> None:
    out_list.append(
        {
            "element": str(element),
            "image_path": list(image_paths),
            "bbox": list(bboxes),
            "image_id": list(range(len(image_paths))),
            "role": list(roles),
        }
    )


def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")



def _ensure_export_root(export_root: Path, backup_if_exists: bool) -> None:
    # English comments only
    if export_root.exists() and any(export_root.rglob("*")):
        if not backup_if_exists:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = export_root.parent / f"{export_root.name}_backup_{ts}"
        export_root.rename(backup)
    export_root.mkdir(parents=True, exist_ok=True)


def _relpath_under_frames_base(src_path: Path, frames_base: Path) -> Optional[Path]:
    # English comments only
    try:
        rel = src_path.resolve().relative_to(frames_base.resolve())
        return rel
    except Exception:
        return None


def _export_one_image(
    src_path: Path,
    frames_base: Path,
    export_root: Path,
    overwrite: bool,
) -> Optional[Path]:
    # English comments only
    rel = _relpath_under_frames_base(src_path, frames_base)
    if rel is None:
        return None
    dst_path = export_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and (not overwrite):
        return dst_path
    try:
        shutil.copy2(src_path, dst_path)
    except Exception:
        return None
    return dst_path


def _write_export_readme(export_root: Path, frames_base: Path) -> None:
    # English comments only
    export_root.mkdir(parents=True, exist_ok=True)
    p = export_root / "readme.txt"
    if p.exists():
        return
    lines = [
        "This is a minimized BURST frames dataset exported by tool_build_dataset_json_pdm.py.",
        f"Original frames_base: {frames_base}",
        "Directory structure is preserved relative to frames_base (e.g., test/<dataset>/<seq_name>/<frame>.jpg).",
        "BBox format in JSON: [x1, y1, x2, y2] in pixel coordinates (xyxy).",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--original_iploc_json", type=str, required=True, help="Teacher IPLoc JSON (N=2,T=1).")
    ap.add_argument("--burst_annotations_json", type=str, required=True)
    ap.add_argument("--burst_frames_base_dir", type=str, required=True)


    ap.add_argument(
        "--export_root",
        type=str,
        default=None,
        help="If set, copy all referenced images under this directory (minimized dataset). Paths in output JSONs will be rewritten to this root.",
    )
    ap.add_argument(
        "--export_overwrite",
        action="store_true",
        help="If set, overwrite existing exported images.",
    )
    ap.add_argument(
        "--export_backup_if_exists",
        action="store_true",
        help="If set and export_root already exists with contents, rename it to *_backup_<timestamp> before exporting.",
    )
    ap.add_argument("--out_dir", type=str, default="./data", help="Output directory for the 2 JSONs.")
    ap.add_argument("--out_prefix", type=str, default="pdm", help="Prefix for output filenames.")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--stop_on_missing_image", action="store_true")
    ap.add_argument("--stop_on_missing_outclass_image", action="store_true")

    ap.add_argument("--exclude_teacher_sequences_for_outclass", action="store_true", default=True)
    ap.add_argument("--allow_teacher_sequences_for_outclass", action="store_true", help="If set, do NOT exclude teacher sequences.")

    args = ap.parse_args()

    teacher = json.loads(Path(args.original_iploc_json).read_text(encoding="utf-8"))
    burst_ann = json.loads(Path(args.burst_annotations_json).read_text(encoding="utf-8"))

    frames_base = Path(args.burst_frames_base_dir)

    export_root = None
    if getattr(args, "export_root", None):
        export_root = Path(str(args.export_root))
        _ensure_export_root(export_root, backup_if_exists=bool(args.export_backup_if_exists))
        _write_export_readme(export_root, frames_base=frames_base)
    out_dir = Path(args.out_dir)

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in burst_ann.get("categories", [])}
    seq_index = build_seq_index(burst_ann)
    burst_split = str(burst_ann.get("split", ""))

    rng = random.Random(int(args.seed))

    teacher_seq_keys = _collect_teacher_seq_keys(teacher)
    if args.allow_teacher_sequences_for_outclass:
        exclude_teacher = set()
    else:
        exclude_teacher = teacher_seq_keys

    out_N2T2: List[Dict[str, Any]] = []
    out_N1T2: List[Dict[str, Any]] = []

    n = len(teacher)
    if args.max_items is not None and int(args.max_items) > 0:
        n = min(n, int(args.max_items))

    kept = 0
    for i in range(n):
        item = teacher[i]
        img_paths_teacher = item.get("image_path")
        bbox_teacher = item.get("bbox", item.get("bboX", item.get("bboxes", [])))
        teacher_element = str(item.get("element", "")).strip()

        if not isinstance(img_paths_teacher, list) or len(img_paths_teacher) != 3:
            continue
        if not isinstance(bbox_teacher, list) or len(bbox_teacher) != 3:
            continue
        if teacher_element == "":
            continue

        tb0 = parse_bbox_str_xyxy(bbox_teacher[0])
        tb1 = parse_bbox_str_xyxy(bbox_teacher[1])
        tb2 = parse_bbox_str_xyxy(bbox_teacher[2])
        if tb0 is None or tb1 is None or tb2 is None:
            continue

        try:
            split0, dataset0, seq_name0, frame0 = parse_seq_from_image_path(img_paths_teacher[0])
            split1, dataset1, seq_name1, frame1 = parse_seq_from_image_path(img_paths_teacher[1])
            split2, dataset2, seq_name2, frame2 = parse_seq_from_image_path(img_paths_teacher[2])
        except Exception:
            continue

        if not (split0 == split1 == split2 and dataset0 == dataset1 == dataset2 and seq_name0 == seq_name1 == seq_name2):
            continue
        if split0 != burst_split:
            continue

        key = (burst_split, dataset0, seq_name0)
        seq = seq_index.get(key)
        if seq is None:
            continue

        ann_paths = seq.get("annotated_image_paths", [])
        segs = seq.get("segmentations", [])
        track_to_cat = seq.get("track_category_ids", {})

        if not isinstance(ann_paths, list) or not isinstance(segs, list):
            continue

        # Teacher frames must be annotated
        try:
            fi0 = ann_paths.index(frame0)
            fi1 = ann_paths.index(frame1)
            fi2 = ann_paths.index(frame2)
        except ValueError:
            continue

        img0 = frames_base / burst_split / dataset0 / seq_name0 / frame0
        img1 = frames_base / burst_split / dataset0 / seq_name0 / frame1
        img2 = frames_base / burst_split / dataset0 / seq_name0 / frame2

        if args.stop_on_missing_image:
            if not img0.exists():
                raise FileNotFoundError(f"Missing image: {img0}")
            if not img1.exists():
                raise FileNotFoundError(f"Missing image: {img1}")
            if not img2.exists():
                raise FileNotFoundError(f"Missing image: {img2}")
        # Optional export: copy images to a minimized dataset root and rewrite JSON paths.
        if export_root is not None:
            dst0 = _export_one_image(img0, frames_base=frames_base, export_root=export_root, overwrite=bool(args.export_overwrite))
            dst1 = _export_one_image(img1, frames_base=frames_base, export_root=export_root, overwrite=bool(args.export_overwrite))
            dst2 = _export_one_image(img2, frames_base=frames_base, export_root=export_root, overwrite=bool(args.export_overwrite))
            if dst0 is None or dst1 is None or dst2 is None:
                if bool(args.stop_on_missing_image):
                    raise RuntimeError(f"Export failed for base images at sample index {i}.")
                continue
            img0 = dst0
            img1 = dst1
            img2 = dst2



        h = int(seq.get("height", 0))
        w = int(seq.get("width", 0))
        if h <= 0 or w <= 0:
            continue

        primary_track = choose_primary_track_by_teacher_iou(
            segs=segs,
            fi0=fi0,
            fi2=fi2,
            h=h,
            w=w,
            track_to_cat=track_to_cat,
            cat_id_to_name=cat_id_to_name,
            teacher_element=teacher_element,
            teacher_bbox0=tb0,
            teacher_bbox2=tb2,
        )
        if primary_track is None:
            continue

        mid_track = choose_mid_track_by_teacher_iou(
            segs=segs,
            fi1=fi1,
            h=h,
            w=w,
            track_to_cat=track_to_cat,
            cat_id_to_name=cat_id_to_name,
            primary_tid=primary_track,
            teacher_bbox1=tb1,
        )
        if mid_track is None:
            continue

        b0 = get_bbox_xyxy_for_track_at_frame(segs, fi0, primary_track, h, w)
        b1 = get_bbox_xyxy_for_track_at_frame(segs, fi1, mid_track, h, w)
        b2 = get_bbox_xyxy_for_track_at_frame(segs, fi2, primary_track, h, w)
        if b0 is None or b1 is None or b2 is None:
            continue

        # Base parts: three frames (ref0, ref1, target)
        base_paths = [str(img0), str(img1), str(img2)]
        base_bboxes = [bbox_to_str(b0), bbox_to_str(b1), bbox_to_str(b2)]

        # Shared outclass selection for both T=2 variants
        picked = _pick_outclass_from_sequence_last_frame(
            seq_index=seq_index,
            frames_base=frames_base,
            burst_split=burst_split,
            cat_id_to_name=cat_id_to_name,
            exclude_teacher_keys=exclude_teacher,
            exclude_cat_name=teacher_element,
            rng=rng,
            stop_on_missing_outclass_image=bool(args.stop_on_missing_outclass_image),
        )
        if picked is None:
            # If outclass is required but not found, skip T=2 for this sample.
            kept += 1
            continue

        out_img_path, out_bbox_str = picked

        if export_root is not None:
            dst_out = _export_one_image(Path(out_img_path), frames_base=frames_base, export_root=export_root, overwrite=bool(args.export_overwrite))
            if dst_out is None:
                if bool(args.stop_on_missing_outclass_image):
                    raise RuntimeError(f"Export failed for outclass image at sample index {i}: {out_img_path}")
                continue
            out_img_path = str(dst_out)


        # N=2, T=2
        _append_item(
            out_N2T2,
            element=teacher_element,
            image_paths=base_paths + [out_img_path],
            bboxes=base_bboxes + [out_bbox_str],
            roles=["reference", "reference", "positive-image", "outclass-image"],
        )

        # N=1, T=2
        _append_item(
            out_N1T2,
            element=teacher_element,
            image_paths=[base_paths[0], base_paths[2], out_img_path],
            bboxes=[base_bboxes[0], base_bboxes[2], out_bbox_str],
            roles=["reference", "positive-image", "outclass-image"],
        )

        kept += 1

    # Write T=2 outputs
    prefix = str(args.out_prefix).strip() or "pdm"
    p_N2T2 = out_dir / f"{prefix}_2shot_T2.json"
    p_N1T2 = out_dir / f"{prefix}_1shot_T2.json"

    _write_json(p_N2T2, out_N2T2)
    _write_json(p_N1T2, out_N1T2)

    print(f"[DONE] wrote: {p_N2T2} (entries={len(out_N2T2)})")
    print(f"[DONE] wrote: {p_N1T2} (entries={len(out_N1T2)})")


if __name__ == "__main__":
    main()
