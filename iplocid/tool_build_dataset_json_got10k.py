#!/usr/bin/env python3
# tool_build_dataset_json_got10k_export.py
#
# Build GOT-10k split into IPLoc-style JSONs for multiple (N, T) configs.
#
# POLICY (user-shared):
# - Use ALL sequences in the selected split (typically val=180).
# - For each sequence, construct a base sample for N=8, T=2:
#     - Pick N=8 reference frames + 1 positive frame (total 9) roughly evenly across the sequence.
#     - If a chosen frame is absent (absence.label == 1):
#         - For references: move forward to the next available (present) frame.
#         - For the final positive: move backward to the previous available (present) frame.
#     - If we cannot obtain 9 DISTINCT present frames, drop the sequence.
#     - Outclass-image:
#         - Randomly pick another sequence (different from the current),
#           then randomly pick a present frame in it.
#         - If chosen outclass frame is absent, resample.
#         - Prefer outclass element != current element (resample with a cap).
# - From the base N=8,T=2 samples, derive and write:
#     - N in {1,2,4,8}
#     - T in {1,2} (T=1 means no outclass-image)
#   Output paths:
#     <out_dir>/got10k-<split>-<N>shot-T<T>.json
#
# Export/minimized policy (default ON):
# - By default, selected images are copied under a minimized root (export_root),
#   preserving relative paths from --got_root.
# - By default, JSON "image_path" points to the MINIMIZED paths (under export_root),
#   not the original paths under got_root.
#
# Notes:
# - GOT-10k bboxes are [xmin, ymin, width, height] in groundtruth.txt.
# - We output [x1, y1, x2, y2] (xyxy) as strings, consistent with prior scripts.
#
# English comments only.

from __future__ import annotations

import argparse
import configparser
import datetime
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# GOT-10k helpers
# -----------------------------


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _parse_absence_label(path: Path) -> List[int]:
    # 0 = present, 1 = absent
    lines = _read_lines(path)
    out: List[int] = []
    for ln in lines:
        s = ln.strip()
        if s == "":
            continue
        try:
            out.append(int(float(s)))
        except Exception:
            out.append(0)
    return out


def _parse_groundtruth_xywh(path: Path) -> List[Tuple[float, float, float, float]]:
    lines = _read_lines(path)
    out: List[Tuple[float, float, float, float]] = []
    for ln in lines:
        s = ln.strip()
        if s == "":
            continue
        parts = [p for p in re.split(r"[,\s]+", s) if p != ""]
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = [float(parts[i]) for i in range(4)]
        except Exception:
            continue
        out.append((x, y, w, h))
    return out


def _xywh_to_xyxy_str(b: Tuple[float, float, float, float]) -> str:
    x, y, w, h = b
    x2 = x + w
    y2 = y + h
    return f"[{x}, {y}, {x2}, {y2}]"


def _load_meta_object_class(meta_ini: Path) -> str:
    # meta_info.ini varies; try multiple keys.
    if not meta_ini.exists():
        return "object"

    cp = configparser.ConfigParser()
    try:
        cp.read(meta_ini, encoding="utf-8")
    except Exception:
        txt = meta_ini.read_text(encoding="utf-8", errors="ignore")
        for key in ["object_class", "object", "class", "target"]:
            m = re.search(rf"{key}\s*[:=]\s*(.+)", txt)
            if m:
                return str(m.group(1)).strip()
        return "object"

    cand_sections = ["META", "meta", "Meta", "Sequence", "sequence", "DEFAULT"]
    for sec in cand_sections:
        if sec in cp:
            for key in ["object_class", "object", "class", "target"]:
                if key in cp[sec]:
                    return str(cp[sec][key]).strip()

    for sec in cp.sections():
        for key in ["object_class", "object", "class", "target"]:
            if key in cp[sec]:
                return str(cp[sec][key]).strip()

    for key in ["object_class", "object", "class", "target"]:
        if key in cp.defaults():
            return str(cp.defaults()[key]).strip()

    return "object"


def _list_sequence_dirs(got_root: Path, split: str) -> List[Path]:
    base = got_root / split
    if not base.exists():
        raise FileNotFoundError(f"Split directory not found: {base}")

    list_txt = base / "list.txt"
    if list_txt.exists():
        names = [ln.strip() for ln in _read_lines(list_txt) if ln.strip()]
        return [base / nm for nm in names if (base / nm).is_dir()]

    return sorted([p for p in base.iterdir() if p.is_dir()])


def _resolve_image_dir(seq_dir: Path) -> Path:
    for cand in ["img", "imgs", "images", "color"]:
        p = seq_dir / cand
        if p.is_dir():
            return p
    return seq_dir


def _frame_path(img_dir: Path, frame_index_0based: int) -> Optional[Path]:
    idx1 = frame_index_0based + 1
    patterns = [
        f"{idx1:08d}.jpg",
        f"{idx1:08d}.png",
        f"{idx1:08d}.jpeg",
        f"{idx1:06d}.jpg",
        f"{idx1:06d}.png",
        f"{idx1:06d}.jpeg",
        f"{idx1:04d}.jpg",
        f"{idx1:04d}.png",
        f"{idx1:04d}.jpeg",
    ]
    for fn in patterns:
        p = img_dir / fn
        if p.exists():
            return p

    files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]])
    if 0 <= frame_index_0based < len(files):
        return files[frame_index_0based]
    return None


# -----------------------------
# Sampling logic
# -----------------------------


def _present_indices(absence: List[int]) -> List[int]:
    return [i for i, a in enumerate(absence) if int(a) == 0]


def _pick_even_targets(num_frames: int, k: int) -> List[int]:
    if num_frames <= 0 or k <= 0:
        return []
    if k == 1:
        return [num_frames // 2]
    out: List[int] = []
    for i in range(k):
        t = int(round(i * (num_frames - 1) / (k - 1)))
        out.append(max(0, min(num_frames - 1, t)))
    return out


def _shift_to_present_unique_forward(idx: int, present_set: set[int], used: set[int], num_frames: int) -> Optional[int]:
    for j in range(idx, num_frames):
        if j in present_set and j not in used:
            return j
    return None


def _shift_to_present_unique_backward(idx: int, present_set: set[int], used: set[int]) -> Optional[int]:
    for j in range(idx, -1, -1):
        if j in present_set and j not in used:
            return j
    return None


def _select_frames_with_absence_policy(absence: List[int], num_frames: int, n_refs: int) -> Optional[List[int]]:
    # Returns indices: n_refs references + 1 positive (last), total n_refs+1.
    need = n_refs + 1
    present_set = set(_present_indices(absence))
    if len(present_set) < need:
        return None

    targets = _pick_even_targets(num_frames, need)
    used: set[int] = set()
    picked: List[int] = []

    for t in targets[:-1]:
        j = _shift_to_present_unique_forward(t, present_set, used, num_frames)
        if j is None:
            return None
        used.add(j)
        picked.append(j)

    tpos = targets[-1]
    jpos = _shift_to_present_unique_backward(tpos, present_set, used)
    if jpos is None:
        return None
    used.add(jpos)
    picked.append(jpos)

    return picked if len(picked) == need else None


@dataclass(frozen=True)
class SeqInfo:
    name: str
    seq_dir: Path
    img_dir: Path
    element: str
    gt_xywh: List[Tuple[float, float, float, float]]
    absence: List[int]
    num_frames: int


def _load_seq_info(seq_dir: Path) -> Optional[SeqInfo]:
    img_dir = _resolve_image_dir(seq_dir)

    gt_path = seq_dir / "groundtruth.txt"
    abs_path = seq_dir / "absence.label"
    meta_path = seq_dir / "meta_info.ini"

    if not gt_path.exists() or not abs_path.exists():
        return None

    gt = _parse_groundtruth_xywh(gt_path)
    absence = _parse_absence_label(abs_path)
    if len(gt) == 0 or len(absence) == 0:
        return None

    n = min(len(gt), len(absence))
    gt = gt[:n]
    absence = absence[:n]
    if n <= 0:
        return None

    element = _load_meta_object_class(meta_path).strip() or "object"

    return SeqInfo(
        name=seq_dir.name,
        seq_dir=seq_dir,
        img_dir=img_dir,
        element=str(element),
        gt_xywh=gt,
        absence=absence,
        num_frames=n,
    )


def _pick_outclass(
    all_seqs: List[SeqInfo],
    exclude_name: str,
    exclude_element: str,
    rng: random.Random,
    max_tries: int = 500,
) -> Optional[Tuple[str, str, str]]:
    # Returns (rel_image_path_from_got_root, bbox_xyxy_str, element)
    if not all_seqs:
        return None

    def _try(prefer_diff: bool) -> Optional[Tuple[str, str, str]]:
        for _ in range(max_tries):
            s = rng.choice(all_seqs)
            if s.name == exclude_name:
                continue
            if prefer_diff and (str(s.element) == str(exclude_element)):
                continue

            present = _present_indices(s.absence)
            if not present:
                continue
            fi = rng.choice(present)

            imgp = _frame_path(s.img_dir, fi)
            if imgp is None or (not imgp.exists()):
                continue

            bbox = _xywh_to_xyxy_str(s.gt_xywh[fi])
            rel = str(imgp)
            return rel, bbox, str(s.element)
        return None

    out = _try(prefer_diff=True)
    if out is not None:
        return out
    return _try(prefer_diff=False)


# -----------------------------
# Export helpers (minimized)
# -----------------------------


def _infer_export_root(got_root: Path) -> Optional[Path]:
    # English comments only
    parts = list(got_root.parts)
    if "ICL_tracking" in parts and "ICL_tracking_minimized" not in parts:
        parts = ["ICL_tracking_minimized" if p == "ICL_tracking" else p for p in parts]
        return Path(*parts)
    return None


def _maybe_backup_export_root(export_root: Path, backup_if_exists: bool) -> None:
    # English comments only
    if not backup_if_exists:
        return
    if not export_root.exists():
        return
    try:
        if export_root.is_dir() and any(export_root.iterdir()):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = export_root.parent / f"{export_root.name}.bak_{ts}"
            shutil.move(str(export_root), str(backup_path))
            print(f"[INFO] export_root backed up to: {backup_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to backup export_root '{export_root}': {e}")


def _to_rel_under_root(p: Path, root: Path) -> Path:
    # English comments only
    try:
        return p.relative_to(root)
    except Exception:
        # If the image is outside root (unexpected), store under _external
        return Path("_external") / p.name


def _export_paths_and_rewrite(
    records: List[Dict[str, object]],
    got_root: Path,
    export_root: Path,
) -> None:
    # English comments only
    export_root.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    copied = 0
    skipped = 0
    missing = 0

    for rec in records:
        paths = rec.get("image_path")
        if not isinstance(paths, list):
            continue

        new_paths: List[str] = []
        for sp in paths:
            src = Path(str(sp))
            rel = _to_rel_under_root(src, got_root)
            dst = export_root / rel

            new_paths.append(str(dst))

            key = str(src)
            if key in seen:
                continue
            seen.add(key)

            if not src.exists():
                missing += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

        rec["image_path"] = new_paths
        rec["image_id"] = list(range(len(new_paths)))

    print(f"[INFO] export: copied={copied}, skipped_existing={skipped}, missing_src={missing}")


def _write_json(path: Path, data: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--got_root", type=str, required=True, help="Path to GOT-10k root directory (contains train/val/test).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--max_seqs", type=int, default=-1)

    ap.add_argument(
        "--export_root",
        type=str,
        default="__AUTO__",
        help="Minimized root to export images under. Default '__AUTO__' tries to infer from --got_root.",
    )
    ap.add_argument(
        "--no_export",
        action="store_true",
        help="If set, do not export/copy images. JSON paths will still be rewritten if --write_paths_root=export.",
    )
    ap.add_argument(
        "--backup_if_exists",
        action="store_true",
        help="If set and export_root exists and is non-empty, move it to a timestamped backup before exporting.",
    )
    ap.add_argument(
        "--write_paths_root",
        type=str,
        default="export",
        choices=["export", "original"],
        help="Which root to use for JSON image_path: export (minimized) or original.",
    )
    args = ap.parse_args()

    got_root = Path(args.got_root)
    split = str(args.split)
    out_dir = Path(args.out_dir)
    rng = random.Random(int(args.seed))

    # Resolve export_root
    export_root: Optional[Path]
    if str(args.export_root) == "__AUTO__":
        export_root = _infer_export_root(got_root)
        if export_root is None and args.write_paths_root == "export":
            raise SystemExit("[ERROR] Cannot infer export_root from --got_root. Please pass --export_root explicitly.")
    else:
        export_root = Path(str(args.export_root))

    seq_dirs = _list_sequence_dirs(got_root, split)
    if args.max_seqs is not None and int(args.max_seqs) > 0:
        seq_dirs = seq_dirs[: int(args.max_seqs)]

    seq_infos: List[SeqInfo] = []
    for sd in seq_dirs:
        info = _load_seq_info(sd)
        if info is not None:
            seq_infos.append(info)

    print(f"[INFO] loaded sequences: {len(seq_infos)} (split={split})")

    # Master N=8,T=2 (paths initially point to ORIGINAL images)
    master: List[Dict[str, object]] = []
    dropped_no9 = 0
    dropped_noout = 0

    for s in seq_infos:
        picked = _select_frames_with_absence_policy(s.absence, s.num_frames, n_refs=8)
        if picked is None:
            dropped_no9 += 1
            continue

        ref_pos_paths: List[str] = []
        ref_pos_bboxes: List[str] = []

        ok = True
        for fi in picked:
            imgp = _frame_path(s.img_dir, fi)
            if imgp is None or (not imgp.exists()):
                ok = False
                break
            ref_pos_paths.append(str(imgp))
            ref_pos_bboxes.append(_xywh_to_xyxy_str(s.gt_xywh[fi]))
        if not ok:
            dropped_no9 += 1
            continue

        outc = _pick_outclass(seq_infos, exclude_name=s.name, exclude_element=s.element, rng=rng)
        if outc is None:
            dropped_noout += 1
            continue

        out_img, out_bbox, _out_elem = outc

        paths_all = ref_pos_paths + [out_img]
        bboxes_all = ref_pos_bboxes + [out_bbox]
        roles_all = (["reference"] * 8) + ["positive-image"] + ["outclass-image"]

        master.append(
            {
                "element": str(s.element),
                "image_path": paths_all,
                "bbox": bboxes_all,
                "image_id": list(range(len(paths_all))),
                "role": roles_all,
            }
        )

    print(f"[INFO] master N=8,T=2 entries: {len(master)}")
    print(f"[INFO] dropped_no9={dropped_no9}, dropped_noout={dropped_noout}")

    # Export + rewrite to minimized paths (default)
    if export_root is not None and args.write_paths_root == "export":
        if not bool(args.no_export):
            _maybe_backup_export_root(export_root, bool(args.backup_if_exists))
            _export_paths_and_rewrite(records=master, got_root=got_root, export_root=export_root)
        else:
            # Even if no_export is set, rewrite paths deterministically to the export root.
            for rec in master:
                paths = rec.get("image_path")
                if not isinstance(paths, list):
                    continue
                new_paths = []
                for sp in paths:
                    rel = _to_rel_under_root(Path(str(sp)), got_root)
                    new_paths.append(str(export_root / rel))
                rec["image_path"] = new_paths
                rec["image_id"] = list(range(len(new_paths)))

    # Derive other (N,T) configs from master
    Ns = [1, 2, 4, 8]
    Ts = [2]
    outputs: Dict[Tuple[int, int], List[Dict[str, object]]] = {(N, T): [] for N in Ns for T in Ts}

    for rec in master:
        element = str(rec["element"])
        paths_all: List[str] = list(rec["image_path"])  # 8 refs + pos + out => 10
        bboxes_all: List[str] = list(rec["bbox"])

        if len(paths_all) != 10 or len(bboxes_all) != 10:
            continue

        ref_paths = paths_all[:8]
        ref_bboxes = bboxes_all[:8]
        pos_path = paths_all[8]
        pos_bbox = bboxes_all[8]
        out_path = paths_all[9]
        out_bbox = bboxes_all[9]

        for N in Ns:
            rp = ref_paths[:N]
            rb = ref_bboxes[:N]

            outputs[(N, 1)].append(
                {
                    "element": element,
                    "image_path": rp + [pos_path],
                    "bbox": rb + [pos_bbox],
                    "image_id": list(range(N + 1)),
                    "role": (["reference"] * N) + ["positive-image"],
                }
            )
            outputs[(N, 2)].append(
                {
                    "element": element,
                    "image_path": rp + [pos_path, out_path],
                    "bbox": rb + [pos_bbox, out_bbox],
                    "image_id": list(range(N + 2)),
                    "role": (["reference"] * N) + ["positive-image", "outclass-image"],
                }
            )

    for N in Ns:
        for T in Ts:
            out_path = out_dir / f"got10k-{split}_{N}shot_T{T}.json"
            _write_json(out_path, outputs[(N, T)])
            print(f"[DONE] wrote: {out_path} (entries={len(outputs[(N, T)])})")


if __name__ == "__main__":
    main()
