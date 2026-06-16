#!/usr/bin/env python3
# tool_build_dataset_json_vasttrack.py
#
# Build VastTrack into IPLoc v22-style JSONs (schema matches user's LASOT JSON),
# and export only selected images to a minimized dataset root.
#
# User policy (VastTrack):
# - Root: <Class>/<SubClass>/{imgs/*.jpg, Groundtruth.txt, ...}
# - Only use classes that have >= 2 subclasses.
# - For each eligible class:
#     1) Randomly choose one subclass subA.
#     2) From subA, select 8 references + 1 positive (total 9) with valid bbox (w>0,h>0),
#        roughly evenly spaced among valid frames.
#        If not enough valid frames, skip this class (no retry within class).
#     3) Randomly choose another subclass subB != subA.
#     4) From subB, randomly sample 1 valid frame as "inclass-image" (retry within subB).
#        If subB has no valid frames -> skip this class (no retry with another subB).
#     5) Once a sample is built for a class, move to next class (at most 1 sample per class).
# - Export selected images only, preserving folder structure under export_root:
#     <export_root>/<Class>/<SubClass>/imgs/00001.jpg ...
# - Generate JSONs for N in {1,2,4,8} and T in {1,2}:
#     VastTrack_test_<N>shot_T<T>.json
#
# JSON schema (matches user's LASOT sample):
# [
#   {
#     "element": "<class>",
#     "image_path": [abs_path0, abs_path1, ...],
#     "bbox": ["[x1, y1, x2, y2]", ...],
#     "image_id": [0,1,2,...],
#     "role": ["reference", ..., "inclass-image/positive-image (random order when T==2)"]
#   },
#   ...
# ]
#
# Latest change (user requested):
# - When T==2, randomize the order of ("inclass-image", "positive-image") in JSON output.
#   References stay first: reference x N, then the two target roles in random order.
#
# Change requested earlier:
# - Do NOT write summary JSON into output_dir; print it to stdout instead.

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class FrameAnno:
    rel_path: str  # Relative path from vasttrack_root, e.g. "Aardwolf/Aardwolf-1/imgs/00001.jpg"
    bbox_xywh: Tuple[float, float, float, float]  # (x, y, w, h)


@dataclass
class VastTrackSampleBase:
    class_name: str
    subA: str
    subB: str
    refs8: List[FrameAnno]  # length 8
    pos1: FrameAnno         # 1 (from subA)
    inc1: FrameAnno         # 1 (from subB)


def parse_bbox_line(line: str) -> Tuple[float, float, float, float]:
    parts = line.strip().split(",")
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox line: {line!r}")
    x, y, w, h = [float(p) for p in parts]
    return x, y, w, h


def read_groundtruth(gt_path: Path) -> List[Tuple[float, float, float, float]]:
    text = gt_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = text.splitlines()
    return [parse_bbox_line(ln) for ln in lines]


def list_sorted_images(img_dir: Path) -> List[Path]:
    return sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )


def valid_indices_from_bboxes(bboxes: List[Tuple[float, float, float, float]]) -> List[int]:
    valid: List[int] = []
    for i, (_, _, w, h) in enumerate(bboxes):
        if w > 0 and h > 0:
            valid.append(i)
    return valid


def pick_evenly_spaced_unique(valid_indices: List[int], k: int) -> Optional[List[int]]:
    """
    Pick k original frame indices from valid_indices, roughly evenly spaced.
    Returns None if impossible.
    """
    L = len(valid_indices)
    if L < k:
        return None
    if k == 1:
        return [valid_indices[L // 2]]

    raw_pos = [int(round(j * (L - 1) / (k - 1))) for j in range(k)]

    used = set()
    chosen: List[int] = []
    for p in raw_pos:
        q = p
        if q in used:
            found = False
            for delta in range(1, L):
                qf = p + delta
                qb = p - delta
                if qf < L and qf not in used:
                    q = qf
                    found = True
                    break
                if qb >= 0 and qb not in used:
                    q = qb
                    found = True
                    break
            if not found:
                return None
        used.add(q)
        chosen.append(valid_indices[q])

    if len(set(chosen)) != k:
        return None
    return chosen


def rel_from_root(root: Path, abs_path: Path) -> str:
    return str(abs_path.relative_to(root)).replace("\\", "/")


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def xywh_to_xyxy_str(b: Tuple[float, float, float, float]) -> str:
    """
    Convert (x, y, w, h) to a string "[x1, y1, x2, y2]" using x2=x+w, y2=y+h.
    """
    x, y, w, h = b
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return f"[{x1}, {y1}, {x2}, {y2}]"


def build_one_sample_for_class(vast_root: Path, class_dir: Path, rng: random.Random) -> Optional[VastTrackSampleBase]:
    class_name = class_dir.name
    subs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
    if len(subs) < 2:
        return None

    # Step 1: pick subA
    subA_dir = rng.choice(subs)
    subA = subA_dir.name
    imgsA_dir = subA_dir / "imgs"
    gtA_path = subA_dir / "Groundtruth.txt"
    if not imgsA_dir.is_dir() or not gtA_path.is_file():
        return None

    imgsA = list_sorted_images(imgsA_dir)
    bboxesA = read_groundtruth(gtA_path)
    if len(imgsA) != len(bboxesA) or len(imgsA) == 0:
        return None

    validA = valid_indices_from_bboxes(bboxesA)

    # Need 9 valid frames (8 refs + 1 positive)
    chosenA = pick_evenly_spaced_unique(validA, k=9)
    if chosenA is None:
        return None

    def fa_A(i: int) -> FrameAnno:
        return FrameAnno(
            rel_path=rel_from_root(vast_root, imgsA[i]),
            bbox_xywh=bboxesA[i],
        )

    chosenA_ann = [fa_A(i) for i in chosenA]
    refs8 = chosenA_ann[:8]
    pos1 = chosenA_ann[8]

    # Step 3: pick subB != subA
    candB = [p for p in subs if p.name != subA]
    if not candB:
        return None
    subB_dir = rng.choice(candB)
    subB = subB_dir.name
    imgsB_dir = subB_dir / "imgs"
    gtB_path = subB_dir / "Groundtruth.txt"
    if not imgsB_dir.is_dir() or not gtB_path.is_file():
        return None

    imgsB = list_sorted_images(imgsB_dir)
    bboxesB = read_groundtruth(gtB_path)
    if len(imgsB) != len(bboxesB) or len(imgsB) == 0:
        return None

    validB = valid_indices_from_bboxes(bboxesB)
    if len(validB) == 0:
        return None

    # Step 4: pick 1 valid frame from subB as inclass
    pickB = rng.choice(validB)
    inc1 = FrameAnno(
        rel_path=rel_from_root(vast_root, imgsB[pickB]),
        bbox_xywh=bboxesB[pickB],
    )

    return VastTrackSampleBase(
        class_name=class_name,
        subA=subA,
        subB=subB,
        refs8=refs8,
        pos1=pos1,
        inc1=inc1,
    )


def export_images(vast_root: Path, export_root: Path, samples: List[VastTrackSampleBase]) -> None:
    """
    Export all selected images (max: 8 refs + 1 pos + 1 inclass per sample)
    to export_root, preserving relative paths.
    """
    export_root.mkdir(parents=True, exist_ok=True)

    def copy_rel(rel_path: str) -> None:
        src = vast_root / rel_path
        dst = export_root / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"Missing source image: {src}")
        ensure_parent(dst)
        if not dst.exists():
            shutil.copy2(src, dst)

    for s in samples:
        for f in s.refs8:
            copy_rel(f.rel_path)
        copy_rel(s.pos1.rel_path)
        copy_rel(s.inc1.rel_path)


def build_json_list(
    export_root: Path,
    samples: List[VastTrackSampleBase],
    N: int,
    T: int,
    seed: int,
) -> List[dict]:
    """
    Build JSON list matching user's IPLoc v22 schema.

    Order:
      - reference x N
      - if T==1: positive-image
      - if T==2: (positive-image, inclass-image) randomized per-sample
    """
    assert N in [1, 2, 4, 8]
    assert T in [1, 2]

    out: List[dict] = []

    # Use a config-specific RNG base so that different (N,T) produce deterministic-yet-different shuffles.
    base_rng = random.Random(seed + 1000 * N + 10 * T)

    for sample_idx, s in enumerate(samples):
        refsN = s.refs8[:N]

        image_paths: List[str] = []
        bboxes: List[str] = []
        roles: List[str] = []

        # reference x N
        for f in refsN:
            image_paths.append(str((export_root / f.rel_path).resolve()))
            bboxes.append(xywh_to_xyxy_str(f.bbox_xywh))
            roles.append("reference")

        if T == 1:
            image_paths.append(str((export_root / s.pos1.rel_path).resolve()))
            bboxes.append(xywh_to_xyxy_str(s.pos1.bbox_xywh))
            roles.append("positive-image")
        else:
            # Create two target entries then shuffle their order.
            entries: List[Dict[str, Any]] = [
                {
                    "image_path": str((export_root / s.pos1.rel_path).resolve()),
                    "bbox": xywh_to_xyxy_str(s.pos1.bbox_xywh),
                    "role": "positive-image",
                },
                {
                    "image_path": str((export_root / s.inc1.rel_path).resolve()),
                    "bbox": xywh_to_xyxy_str(s.inc1.bbox_xywh),
                    "role": "inclass-image",
                },
            ]

            # Derive per-sample RNG from base to avoid global-state dependency.
            # This keeps reproducibility while giving a different shuffle per sample.
            per_rng = random.Random(base_rng.randint(0, 2**31 - 1) + sample_idx)
            per_rng.shuffle(entries)

            for e in entries:
                image_paths.append(e["image_path"])
                bboxes.append(e["bbox"])
                roles.append(e["role"])

        out.append(
            {
                "element": s.class_name,
                "image_path": image_paths,
                "bbox": bboxes,
                "image_id": list(range(len(image_paths))),
                "role": roles,
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vasttrack_root", type=str, required=True, help="Extracted VastTrack root (contains class dirs).")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to write JSON files.")
    ap.add_argument("--export_root", type=str, required=True, help="Minimized dataset root (e.g., ICL_tracking_minimized/video/VastTrack).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--max_classes", type=int, default=-1, help="Optional cap on eligible classes (-1 = no cap).")
    ap.add_argument("--N_set", type=str, default="1,2,4,8", help="Comma-separated N values.")
    ap.add_argument("--T_set", type=str, default="2", help="Comma-separated T values.")
    args = ap.parse_args()

    vast_root = Path(args.vasttrack_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    export_root = Path(args.export_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    N_set = [int(x.strip()) for x in args.N_set.split(",") if x.strip()]
    T_set = [int(x.strip()) for x in args.T_set.split(",") if x.strip()]

    rng = random.Random(args.seed)

    # List class directories
    class_dirs = sorted([p for p in vast_root.iterdir() if p.is_dir()])

    # Eligible classes: those with >=2 subclasses
    eligible: List[Path] = []
    for cd in class_dirs:
        subs = [p for p in cd.iterdir() if p.is_dir()]
        if len(subs) >= 2:
            eligible.append(cd)

    if args.max_classes > 0:
        eligible = eligible[: args.max_classes]

    samples: List[VastTrackSampleBase] = []
    skipped = 0

    for cd in eligible:
        s = build_one_sample_for_class(vast_root, cd, rng)
        if s is None:
            skipped += 1
            continue
        samples.append(s)

    # Export images needed for the largest config; smaller configs are subsets
    export_images(vast_root, export_root, samples)

    # Write JSONs
    for N in N_set:
        for T in T_set:
            js_list = build_json_list(export_root, samples, N=N, T=T, seed=args.seed)
            out_path = out_dir / f"VastTrack_test_{N}shot_T{T}.json"
            out_path.write_text(json.dumps(js_list, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary (do not write a summary file)
    summary = {
        "vasttrack_root": str(vast_root),
        "export_root": str(export_root),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "eligible_classes": len(eligible),
        "built_samples": len(samples),
        "skipped_classes": skipped,
        "N_set": N_set,
        "T_set": T_set,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
