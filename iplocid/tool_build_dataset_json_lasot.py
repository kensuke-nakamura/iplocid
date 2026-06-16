#!/usr/bin/env python3
# tool_build_dataset_json_lasot.py
#
# Ver 10.3 (adds T=9 for TEST only)
# - Keeps Ver 10.2 behavior for T in {1,2,3,18}.
# - Adds T=9:
#   * TRAIN: do not generate (skip), same policy as T=18.
#   * TEST : for each class, generate only ONE sample (same selection policy as T=18: i_seq==0).
#            sample contains:
#              inclass(3 frames from seqA) + target(3 frames from target seq) + inclass(3 frames from seqB)
#            where frames are picked near the end with stride p=50 and shifted backward to valid bboxes.
#            If 3 valid frames cannot be found for any required sequence, print error and skip that sample.
#
# English comments only.

import os
import re
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set


# -------------------------
# Helpers
# -------------------------
def natural_key(s: str):
    """Natural sort key helper."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def list_lasot_classes(lasot_root: Path) -> List[str]:
    classes: List[str] = []
    for cls_name in os.listdir(lasot_root):
        p = lasot_root / cls_name
        if p.is_dir():
            classes.append(cls_name)
    return sorted(classes, key=natural_key)


def list_lasot_sequences_for_class(lasot_root: Path, cls_name: str) -> List[str]:
    seqs: List[str] = []
    cls_dir = lasot_root / cls_name
    if not cls_dir.is_dir():
        return seqs

    for seq_name in os.listdir(cls_dir):
        seq_dir = cls_dir / seq_name
        if not seq_dir.is_dir():
            continue
        img_dir = seq_dir / "img"
        gt_file = seq_dir / "groundtruth.txt"
        if img_dir.is_dir() and gt_file.is_file():
            seqs.append(seq_name)
    return sorted(seqs, key=natural_key)


def load_groundtruth_xyxy(gt_file: Path) -> List[List[float]]:
    """Load LASOT gt (xywh) and convert to xyxy float."""
    boxes: List[List[float]] = []
    with gt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                continue
            parts = line.split(",")
            if len(parts) < 4:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                continue
            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            boxes.append([x, y, x + w, y + h])
    return boxes


def is_valid_xyxy(box: List[float]) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    return (x2 > x1) and (y2 > y1)


def find_last_valid_idx(gt_boxes: List[List[float]], start_idx: int) -> Optional[int]:
    if not gt_boxes:
        return None
    i = min(start_idx, len(gt_boxes) - 1)
    while i >= 0:
        if is_valid_xyxy(gt_boxes[i]):
            return i
        i -= 1
    return None


def find_next_valid_idx(gt_boxes: List[List[float]], start_idx: int, used: Optional[Set[int]] = None) -> Optional[int]:
    """Find next valid bbox index at or after start_idx, optionally skipping used indices."""
    used = used or set()
    n = len(gt_boxes)
    i = max(0, start_idx)
    while i < n:
        if (i not in used) and is_valid_xyxy(gt_boxes[i]):
            return i
        i += 1
    return None


def find_prev_valid_idx(gt_boxes: List[List[float]], start_idx: int, used: Optional[Set[int]] = None) -> Optional[int]:
    """Find previous valid bbox index at or before start_idx, optionally skipping used indices."""
    used = used or set()
    i = min(start_idx, len(gt_boxes) - 1)
    while i >= 0:
        if (i not in used) and is_valid_xyxy(gt_boxes[i]):
            return i
        i -= 1
    return None


def backup_if_exists(_path: Path) -> None:
    """Disabled intentionally (user asked to remove backup behavior)."""
    return


def export_copy_file(src: Path, dst: Path) -> None:
    """Copy file if missing."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def export_asset(
    lasot_root: Path,
    export_root: Optional[Path],
    cls_name: str,
    seq_name: str,
    img_name: str,
    mode: str,
) -> None:
    if export_root is None:
        return
    src = lasot_root / cls_name / seq_name / "img" / img_name
    dst = export_root / cls_name / seq_name / "img" / img_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing source image: {src}")
    if mode == "copy":
        export_copy_file(src, dst)
    elif mode == "off":
        return
    else:
        raise ValueError(f"Unknown export_mode: {mode}")


def parse_comma_separated_ints(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def pick_random_frame_from_sequence_valid(
    lasot_root: Path,
    cls_name: str,
    seq_name: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
    max_tries: int = 200,
) -> Tuple[str, str]:
    """
    Pick a random valid-bbox frame from a specific sequence.
    Returns (img_path_str, bbox_str).
    """
    seq_dir = lasot_root / cls_name / seq_name
    img_dir = seq_dir / "img"
    gt_file = seq_dir / "groundtruth.txt"

    all_imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda p: natural_key(p.name),
    )
    if not all_imgs:
        raise RuntimeError(f"No images in {cls_name}/{seq_name}")

    gt_boxes = load_groundtruth_xyxy(gt_file)
    n = min(len(all_imgs), len(gt_boxes))
    valid_indices = [i for i in range(n) if is_valid_xyxy(gt_boxes[i])]
    if not valid_indices:
        raise RuntimeError(f"No valid bbox in {cls_name}/{seq_name}")

    for _ in range(max_tries):
        idx = rng.choice(valid_indices)
        img_name = all_imgs[idx].name

        export_asset(
            lasot_root=lasot_root,
            export_root=export_root,
            cls_name=cls_name,
            seq_name=seq_name,
            img_name=img_name,
            mode=export_mode,
        )

        x1, y1, x2, y2 = gt_boxes[idx]
        img_path_str = f"{image_root_prefix}/{cls_name}/{seq_name}/img/{img_name}"
        bbox_str = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"
        return img_path_str, bbox_str

    raise RuntimeError(f"Failed to sample valid frame: {cls_name}/{seq_name}")


def pick_random_inclass_from_pool_meta(
    lasot_root: Path,
    pool_seq_dict: Dict[str, List[str]],
    cls_name: str,
    exclude_seq: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
) -> Tuple[str, str, str, str]:
    """
    Inclass anchor from FULL pool:
    returns (cls_name, chosen_seq, img_path, bbox_str)
    """
    all_seqs = pool_seq_dict.get(cls_name, [])
    candidates = [s for s in all_seqs if s != exclude_seq]
    if not candidates:
        candidates = [exclude_seq]

    in_seq = rng.choice(candidates)
    img_path, bbox_str = pick_random_frame_from_sequence_valid(
        lasot_root=lasot_root,
        cls_name=cls_name,
        seq_name=in_seq,
        image_root_prefix=image_root_prefix,
        rng=rng,
        export_root=export_root,
        export_mode=export_mode,
    )
    return cls_name, in_seq, img_path, bbox_str


def pick_random_outclass_from_pool_meta(
    lasot_root: Path,
    pool_seq_dict: Dict[str, List[str]],
    exclude_cls: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
) -> Tuple[str, str, str, str]:
    """
    Outclass anchor from FULL pool:
    returns (neg_cls, neg_seq, img_path, bbox_str)
    """
    candidate_classes = [c for c, seqs in pool_seq_dict.items() if c != exclude_cls and len(seqs) > 0]
    if not candidate_classes:
        raise RuntimeError("No candidate classes for outclass sampling.")

    neg_cls = rng.choice(candidate_classes)
    neg_seq = rng.choice(pool_seq_dict[neg_cls])

    img_path, bbox_str = pick_random_frame_from_sequence_valid(
        lasot_root=lasot_root,
        cls_name=neg_cls,
        seq_name=neg_seq,
        image_root_prefix=image_root_prefix,
        rng=rng,
        export_root=export_root,
        export_mode=export_mode,
    )
    return neg_cls, neg_seq, img_path, bbox_str


def sample_reference_indices(num_frames: int, shots: int) -> List[int]:
    if shots <= 0:
        return []
    if num_frames <= shots:
        return list(range(num_frames))
    step = max(1, num_frames // shots)
    ref_indices = [min(i * step, num_frames - 1) for i in range(shots)]
    return sorted(set(ref_indices))


def build_sample_for_sequence(
    lasot_root: Path,
    export_root: Optional[Path],
    export_mode: str,
    cls_name: str,
    seq_name_target: str,
    image_root_prefix: str,
    ref_shots: int,
    target_slots: List[Tuple[str, str, str]],  # (role, img_path, bbox_str) in order
) -> Dict[str, Any]:
    """
    Build one JSON sample with:
      references (N shots) + target_slots (T slots).

    Reference sampling is bbox-validity-aware:
      - If sampled reference has invalid bbox, shift to nearest valid bbox frame.
      - Avoid duplicates as much as possible.
    """
    seq_dir = lasot_root / cls_name / seq_name_target
    img_dir = seq_dir / "img"
    gt_file = seq_dir / "groundtruth.txt"

    all_imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda p: natural_key(p.name),
    )
    num_frames = len(all_imgs)
    gt_boxes = load_groundtruth_xyxy(gt_file)

    raw_ref_indices = sample_reference_indices(num_frames, ref_shots)

    used_ref: Set[int] = set()
    ref_indices: List[int] = []

    for idx in raw_ref_indices:
        idx = max(0, min(idx, num_frames - 1))
        chosen: Optional[int] = None

        if idx < len(gt_boxes) and is_valid_xyxy(gt_boxes[idx]) and idx not in used_ref:
            chosen = idx
        else:
            chosen = find_next_valid_idx(gt_boxes, start_idx=idx, used=used_ref)
            if chosen is None:
                chosen = find_prev_valid_idx(gt_boxes, start_idx=idx, used=used_ref)

        if chosen is None:
            continue

        used_ref.add(chosen)
        ref_indices.append(chosen)

    # Fill remaining refs if still short
    if ref_shots > 0 and len(ref_indices) < min(ref_shots, num_frames):
        while len(ref_indices) < min(ref_shots, num_frames):
            seed_idx = max(0, num_frames - 1 - (len(ref_indices) * 3))
            chosen = find_prev_valid_idx(gt_boxes, start_idx=seed_idx, used=used_ref)
            if chosen is None:
                break
            used_ref.add(chosen)
            ref_indices.append(chosen)

    image_paths: List[str] = []
    bboxes: List[str] = []
    image_ids: List[int] = []
    roles: List[str] = []

    lid = 0
    for idx in ref_indices:
        idx = max(0, min(idx, num_frames - 1))
        img_name = all_imgs[idx].name

        export_asset(
            lasot_root=lasot_root,
            export_root=export_root,
            cls_name=cls_name,
            seq_name=seq_name_target,
            img_name=img_name,
            mode=export_mode,
        )

        img_path_str = f"{image_root_prefix}/{cls_name}/{seq_name_target}/img/{img_name}"
        x1, y1, x2, y2 = gt_boxes[idx]
        bbox_str = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"

        image_paths.append(img_path_str)
        bboxes.append(bbox_str)
        image_ids.append(lid)
        roles.append("reference")
        lid += 1

    for role, img_path, bbox_str in target_slots:
        image_paths.append(img_path)
        bboxes.append(bbox_str)
        image_ids.append(lid)
        roles.append(role)
        lid += 1

    return {
        "element": cls_name,
        "image_path": image_paths,
        "bbox": bboxes,
        "image_id": image_ids,
        "role": roles,
    }


def pick_k_inclass_slots_unique(
    lasot_root: Path,
    pool_seq_dict: Dict[str, List[str]],
    cls_name: str,
    exclude_seq: str,
    k: int,
    anchor_seq: str,
    anchor_img_path: str,
    anchor_bbox_str: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
) -> List[Tuple[str, str, str]]:
    """
    k inclass slots from same class using unique sequences.
    The first slot is the anchor.
    """
    all_seqs = pool_seq_dict.get(cls_name, [])
    candidates = [s for s in all_seqs if s != exclude_seq]
    if anchor_seq not in candidates and anchor_seq != exclude_seq:
        raise RuntimeError(f"[T18][INCLASS] Anchor not in pool: {cls_name}/{anchor_seq}")

    pool = [s for s in candidates if s != anchor_seq]
    needed = k - 1
    if len(pool) < needed:
        raise RuntimeError(
            f"[T18][INCLASS] Not enough unique sequences in class '{cls_name}' "
            f"to build k={k} inclass slots (exclude_seq={exclude_seq}). "
            f"Available (excluding target & anchor): {len(pool)}"
        )

    rng.shuffle(pool)
    chosen_seqs = [anchor_seq] + pool[:needed]

    out: List[Tuple[str, str, str]] = [("inclass-image", anchor_img_path, anchor_bbox_str)]
    for seq in chosen_seqs[1:]:
        img_path, bbox_str = pick_random_frame_from_sequence_valid(
            lasot_root=lasot_root,
            cls_name=cls_name,
            seq_name=seq,
            image_root_prefix=image_root_prefix,
            rng=rng,
            export_root=export_root,
            export_mode=export_mode,
        )
        out.append(("inclass-image", img_path, bbox_str))
    return out


def pick_k_outclass_slots_same_neg_class_unique(
    lasot_root: Path,
    pool_seq_dict: Dict[str, List[str]],
    neg_cls: str,
    k: int,
    anchor_seq: str,
    anchor_img_path: str,
    anchor_bbox_str: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
) -> List[Tuple[str, str, str]]:
    """
    k outclass slots from the same negative class using unique sequences.
    The first slot is the anchor.
    """
    all_seqs = pool_seq_dict.get(neg_cls, [])
    if not all_seqs:
        raise RuntimeError(f"[T18][OUTCLASS] Negative class '{neg_cls}' has no sequences in pool.")

    pool = [s for s in all_seqs if s != anchor_seq]
    needed = k - 1
    if len(pool) < needed:
        raise RuntimeError(
            f"[T18][OUTCLASS] Not enough unique sequences in negative class '{neg_cls}' "
            f"to build k={k} outclass slots (excluding anchor). Available: {len(pool)}"
        )

    rng.shuffle(pool)
    chosen_seqs = [anchor_seq] + pool[:needed]

    out: List[Tuple[str, str, str]] = [("outclass-image", anchor_img_path, anchor_bbox_str)]
    for seq in chosen_seqs[1:]:
        img_path, bbox_str = pick_random_frame_from_sequence_valid(
            lasot_root=lasot_root,
            cls_name=neg_cls,
            seq_name=seq,
            image_root_prefix=image_root_prefix,
            rng=rng,
            export_root=export_root,
            export_mode=export_mode,
        )
        out.append(("outclass-image", img_path, bbox_str))
    return out


# -------------------------
# T=9 helpers
# -------------------------
T9_STRIDE_P = 50
T9_K = 3


def _load_seq_imgs_and_gt(lasot_root: Path, cls_name: str, seq_name: str) -> Tuple[List[Path], List[List[float]]]:
    seq_dir = lasot_root / cls_name / seq_name
    img_dir = seq_dir / "img"
    gt_file = seq_dir / "groundtruth.txt"

    all_imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=lambda p: natural_key(p.name),
    )
    gt_boxes = load_groundtruth_xyxy(gt_file)
    n = min(len(all_imgs), len(gt_boxes))
    return all_imgs[:n], gt_boxes[:n]


def pick_last_k_indices_stride_prev_valid(
    gt_boxes: List[List[float]],
    k: int,
    stride: int,
    used: Optional[Set[int]] = None,
) -> Optional[List[int]]:
    """
    Pick k indices near the end:
      bases = [end - (k-1)*stride, ..., end - stride, end]
    For each base, shift backward to the nearest valid bbox index (prev valid),
    while avoiding duplicates via `used`.
    """
    used = used or set()
    n = len(gt_boxes)
    if n <= 0:
        return None

    end = n - 1
    bases = [max(0, end - stride * (k - 1 - i)) for i in range(k)]  # increasing bases
    chosen: List[int] = []
    for b in bases:
        idx = find_prev_valid_idx(gt_boxes, start_idx=b, used=used)
        if idx is None:
            return None
        used.add(idx)
        chosen.append(idx)
    return chosen


def make_slots_t9_test_only(
    lasot_root: Path,
    pool_seq_dict: Dict[str, List[str]],
    cls_name: str,
    target_seq: str,
    image_root_prefix: str,
    rng: random.Random,
    export_root: Optional[Path],
    export_mode: str,
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Build 9 slots for T=9:
      inclass(seqA) x3 + positive(target_seq) x3 + inclass(seqB) x3
    If any required 3 frames cannot be collected (valid bbox), return None.
    """
    all_seqs = pool_seq_dict.get(cls_name, [])
    candidates = [s for s in all_seqs if s != target_seq]
    if len(candidates) < 2:
        print(f"[ERROR][T9] Not enough inclass sequences in class={cls_name} excluding target_seq={target_seq}.")
        return None

    in_seq1, in_seq2 = rng.sample(candidates, 2)

    # Target seq
    tgt_imgs, tgt_gt = _load_seq_imgs_and_gt(lasot_root, cls_name, target_seq)
    used_tgt: Set[int] = set()
    tgt_idx = pick_last_k_indices_stride_prev_valid(tgt_gt, k=T9_K, stride=T9_STRIDE_P, used=used_tgt)
    if tgt_idx is None or len(tgt_idx) != T9_K:
        print(f"[ERROR][T9] Cannot pick target 3 frames (valid) for {cls_name}/{target_seq}. Skip.")
        return None

    # Inclass seq1
    in1_imgs, in1_gt = _load_seq_imgs_and_gt(lasot_root, cls_name, in_seq1)
    used_in1: Set[int] = set()
    in1_idx = pick_last_k_indices_stride_prev_valid(in1_gt, k=T9_K, stride=T9_STRIDE_P, used=used_in1)
    if in1_idx is None or len(in1_idx) != T9_K:
        print(f"[ERROR][T9] Cannot pick inclass(1) 3 frames (valid) for {cls_name}/{in_seq1}. Skip.")
        return None

    # Inclass seq2
    in2_imgs, in2_gt = _load_seq_imgs_and_gt(lasot_root, cls_name, in_seq2)
    used_in2: Set[int] = set()
    in2_idx = pick_last_k_indices_stride_prev_valid(in2_gt, k=T9_K, stride=T9_STRIDE_P, used=used_in2)
    if in2_idx is None or len(in2_idx) != T9_K:
        print(f"[ERROR][T9] Cannot pick inclass(2) 3 frames (valid) for {cls_name}/{in_seq2}. Skip.")
        return None

    def _mk(cls: str, seq: str, all_imgs: List[Path], gt: List[List[float]], idx: int, role: str) -> Tuple[str, str, str]:
        img_name = all_imgs[idx].name
        export_asset(lasot_root, export_root, cls, seq, img_name, export_mode)
        x1, y1, x2, y2 = gt[idx]
        bbox = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"
        img_path = f"{image_root_prefix}/{cls}/{seq}/img/{img_name}"
        return (role, img_path, bbox)

    slots: List[Tuple[str, str, str]] = []

    # inclass seq1 (3)
    for idx in in1_idx:
        slots.append(_mk(cls_name, in_seq1, in1_imgs, in1_gt, idx, "inclass-image"))

    # positive seq (3)
    for idx in tgt_idx:
        slots.append(_mk(cls_name, target_seq, tgt_imgs, tgt_gt, idx, "positive-image"))

    # inclass seq2 (3)
    for idx in in2_idx:
        slots.append(_mk(cls_name, in_seq2, in2_imgs, in2_gt, idx, "inclass-image"))

    if len(slots) != 9:
        print(f"[ERROR][T9] Internal error: slots length != 9 for class={cls_name}.")
        return None

    return slots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lasot_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_root_prefix", type=str, default=None)
    parser.add_argument("--test_sample_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--N_set", type=str, default="1,2,4")
    parser.add_argument("--T_set", type=str, default="1,2,3,18")
    parser.add_argument("--export_root", type=str, default=None)
    parser.add_argument("--export_mode", type=str, default="copy", choices=["copy", "off"])
    args = parser.parse_args()

    N_list = parse_comma_separated_ints(args.N_set)
    T_list = parse_comma_separated_ints(args.T_set)

    for T in T_list:
        if T not in (1, 2, 3, 9, 18):
            print(f"[ERROR] T must be in {{1,2,3,9,18}}. Got {T}")
            return

    lasot_root = Path(args.lasot_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # image_root_prefix
    if args.image_root_prefix is None or args.image_root_prefix == "":
        image_root_prefix = str(lasot_root)
    else:
        image_root_prefix = args.image_root_prefix.rstrip("/")

    export_root: Optional[Path] = None
    if args.export_root is not None and args.export_root != "":
        export_root = Path(args.export_root).resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        image_root_prefix = str(export_root)  # override to exported tree

    # 1) Split classes
    all_classes = list_lasot_classes(lasot_root)
    n_cls = len(all_classes)
    if n_cls < 2:
        print("[ERROR] Need at least 2 classes.")
        return

    n_test_cls = max(1, n_cls // 2)
    n_test_cls = min(n_cls - 1, n_test_cls)

    test_classes = all_classes[:n_test_cls]
    train_classes = all_classes[n_test_cls:]

    # 2) Build sequence dicts
    train_pool_seq_dict: Dict[str, List[str]] = {}
    test_pool_seq_dict: Dict[str, List[str]] = {}
    test_target_seq_dict: Dict[str, List[str]] = {}

    total_train = 0
    total_test_targets = 0
    total_test_unused = 0

    # Train pool = ALL sequences in train classes, and they are also targets
    for cls_name in train_classes:
        seqs = list_lasot_sequences_for_class(lasot_root, cls_name)
        if not seqs:
            continue
        train_pool_seq_dict[cls_name] = seqs
        total_train += len(seqs)

    # Test: pool = ALL sequences, targets = sampled subset
    for cls_name in test_classes:
        seqs = list_lasot_sequences_for_class(lasot_root, cls_name)
        if not seqs:
            continue

        test_pool_seq_dict[cls_name] = seqs[:]  # FULL pool

        seqs_shuf = seqs[:]
        rng.shuffle(seqs_shuf)

        n_take = int(round(len(seqs) * args.test_sample_ratio))
        n_take = max(1, min(len(seqs), n_take))

        test_take = seqs_shuf[:n_take]
        unused = seqs_shuf[n_take:]

        test_target_seq_dict[cls_name] = test_take
        total_test_targets += len(test_take)
        total_test_unused += len(unused)

    print(f"[INFO] classes: total={n_cls}, test={len(test_classes)}, train={len(train_classes)}")
    print(f"[INFO] train sequences (targets/pool): {total_train}")
    print(f"[INFO] test  sequences (targets):      {total_test_targets}")
    print(f"[INFO] test  sequences (unused):       {total_test_unused}")
    print(f"[INFO] N_set={N_list}, T_set={T_list}")
    print(f"[INFO] export_root={export_root}")
    print(f"[INFO] image_root_prefix={image_root_prefix}")
    print(f"[INFO] T9 stride p={T9_STRIDE_P}")

    # 3) Containers
    train_data: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    test_data: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for N in N_list:
        for T in T_list:
            train_data[(N, T)] = []
            test_data[(N, T)] = []

    # 4) Process TRAIN
    for cls_name, seq_list in train_pool_seq_dict.items():
        for seq_name_target in seq_list:
            seq_dir = lasot_root / cls_name / seq_name_target
            gt_boxes = load_groundtruth_xyxy(seq_dir / "groundtruth.txt")

            pos_idx = find_last_valid_idx(gt_boxes, start_idx=len(gt_boxes) - 1)
            if pos_idx is None:
                continue

            # NOTE: Kept as-is (legacy): assumes .jpg naming.
            img_name_pos = f"{pos_idx + 1:08d}.jpg"
            export_asset(lasot_root, export_root, cls_name, seq_name_target, img_name_pos, args.export_mode)

            x1, y1, x2, y2 = gt_boxes[pos_idx]
            if not is_valid_xyxy([x1, y1, x2, y2]):
                continue

            img_path_pos = f"{image_root_prefix}/{cls_name}/{seq_name_target}/img/{img_name_pos}"
            bbox_pos = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"

            in_anchor = pick_random_inclass_from_pool_meta(
                lasot_root=lasot_root,
                pool_seq_dict=train_pool_seq_dict,
                cls_name=cls_name,
                exclude_seq=seq_name_target,
                image_root_prefix=image_root_prefix,
                rng=rng,
                export_root=export_root,
                export_mode=args.export_mode,
            )
            out_anchor = pick_random_outclass_from_pool_meta(
                lasot_root=lasot_root,
                pool_seq_dict=train_pool_seq_dict,
                exclude_cls=cls_name,
                image_root_prefix=image_root_prefix,
                rng=rng,
                export_root=export_root,
                export_mode=args.export_mode,
            )

            for T in T_list:
                # Do not generate TRAIN for T=18 or T=9
                if T in (18, 9):
                    continue

                _, in_seq, in_img, in_bbox = in_anchor
                neg_cls, neg_seq, out_img, out_bbox = out_anchor

                if T == 1:
                    slots = [("positive-image", img_path_pos, bbox_pos)]
                elif T == 2:
                    slots = [("positive-image", img_path_pos, bbox_pos), ("inclass-image", in_img, in_bbox)]
                    rng.shuffle(slots)
                elif T == 3:
                    slots = [
                        ("positive-image", img_path_pos, bbox_pos),
                        ("inclass-image", in_img, in_bbox),
                        ("outclass-image", out_img, out_bbox),
                    ]
                    rng.shuffle(slots)
                else:
                    raise ValueError("Unexpected T")

                for N in N_list:
                    train_data[(N, T)].append(
                        build_sample_for_sequence(
                            lasot_root=lasot_root,
                            export_root=export_root,
                            export_mode=args.export_mode,
                            cls_name=cls_name,
                            seq_name_target=seq_name_target,
                            image_root_prefix=image_root_prefix,
                            ref_shots=N,
                            target_slots=slots,
                        )
                    )

    # 5) Process TEST
    for cls_name, target_list in test_target_seq_dict.items():
        pool_for_sampling = test_pool_seq_dict
        for i_seq, seq_name_target in enumerate(target_list):
            seq_dir = lasot_root / cls_name / seq_name_target
            gt_boxes = load_groundtruth_xyxy(seq_dir / "groundtruth.txt")

            pos_idx = find_last_valid_idx(gt_boxes, start_idx=len(gt_boxes) - 1)
            if pos_idx is None:
                continue

            # NOTE: Kept as-is (legacy): assumes .jpg naming.
            img_name_pos = f"{pos_idx + 1:08d}.jpg"
            export_asset(lasot_root, export_root, cls_name, seq_name_target, img_name_pos, args.export_mode)

            x1, y1, x2, y2 = gt_boxes[pos_idx]
            if not is_valid_xyxy([x1, y1, x2, y2]):
                continue

            img_path_pos = f"{image_root_prefix}/{cls_name}/{seq_name_target}/img/{img_name_pos}"
            bbox_pos = f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"

            in_anchor = pick_random_inclass_from_pool_meta(
                lasot_root=lasot_root,
                pool_seq_dict=pool_for_sampling,
                cls_name=cls_name,
                exclude_seq=seq_name_target,
                image_root_prefix=image_root_prefix,
                rng=rng,
                export_root=export_root,
                export_mode=args.export_mode,
            )
            out_anchor = pick_random_outclass_from_pool_meta(
                lasot_root=lasot_root,
                pool_seq_dict=pool_for_sampling,
                exclude_cls=cls_name,
                image_root_prefix=image_root_prefix,
                rng=rng,
                export_root=export_root,
                export_mode=args.export_mode,
            )

            _, in_seq, in_img, in_bbox = in_anchor
            neg_cls, neg_seq, out_img, out_bbox = out_anchor

            for T in T_list:
                # Keep original policy for T=18: only one sample per class
                if T == 18 and i_seq != 0:
                    continue
                # New policy for T=9: only one sample per class (same as T=18)
                if T == 9 and i_seq != 0:
                    continue

                if T == 1:
                    slots = [("positive-image", img_path_pos, bbox_pos)]
                elif T == 2:
                    slots = [("positive-image", img_path_pos, bbox_pos), ("inclass-image", in_img, in_bbox)]
                    rng.shuffle(slots)
                elif T == 3:
                    slots = [
                        ("positive-image", img_path_pos, bbox_pos),
                        ("inclass-image", in_img, in_bbox),
                        ("outclass-image", out_img, out_bbox),
                    ]
                    rng.shuffle(slots)
                elif T == 18:
                    in_slots_8 = pick_k_inclass_slots_unique(
                        lasot_root=lasot_root,
                        pool_seq_dict=pool_for_sampling,
                        cls_name=cls_name,
                        exclude_seq=seq_name_target,
                        k=8,
                        anchor_seq=in_seq,
                        anchor_img_path=in_img,
                        anchor_bbox_str=in_bbox,
                        image_root_prefix=image_root_prefix,
                        rng=rng,
                        export_root=export_root,
                        export_mode=args.export_mode,
                    )
                    out_slots_9 = pick_k_outclass_slots_same_neg_class_unique(
                        lasot_root=lasot_root,
                        pool_seq_dict=pool_for_sampling,
                        neg_cls=neg_cls,
                        k=9,
                        anchor_seq=neg_seq,
                        anchor_img_path=out_img,
                        anchor_bbox_str=out_bbox,
                        image_root_prefix=image_root_prefix,
                        rng=rng,
                        export_root=export_root,
                        export_mode=args.export_mode,
                    )
                    slots = [("positive-image", img_path_pos, bbox_pos)] + in_slots_8 + out_slots_9
                elif T == 9:
                    slots_t9 = make_slots_t9_test_only(
                        lasot_root=lasot_root,
                        pool_seq_dict=pool_for_sampling,
                        cls_name=cls_name,
                        target_seq=seq_name_target,
                        image_root_prefix=image_root_prefix,
                        rng=rng,
                        export_root=export_root,
                        export_mode=args.export_mode,
                    )
                    if slots_t9 is None:
                        # Error already printed; skip this test sample for T=9
                        continue
                    slots = slots_t9
                else:
                    raise ValueError("Unexpected T")

                for N in N_list:
                    test_data[(N, T)].append(
                        build_sample_for_sequence(
                            lasot_root=lasot_root,
                            export_root=export_root,
                            export_mode=args.export_mode,
                            cls_name=cls_name,
                            seq_name_target=seq_name_target,
                            image_root_prefix=image_root_prefix,
                            ref_shots=N,
                            target_slots=slots,
                        )
                    )

    # 6) Save JSONs
    for N in N_list:
        for T in T_list:
            test_json_path = output_dir / f"LASOT_{N}shot_T{T}_classwise-split_test.json"
            with test_json_path.open("w", encoding="utf-8") as f:
                json.dump(test_data[(N, T)], f, indent=2)
            print(f"[SAVE] TEST  N={N}, T={T}: {len(test_data[(N, T)])} -> {test_json_path}")

            # Do not save TRAIN for T=18 or T=9
            if T in (18, 9):
                continue

            train_json_path = output_dir / f"LASOT_{N}shot_T{T}_classwise-split_train.json"
            with train_json_path.open("w", encoding="utf-8") as f:
                json.dump(train_data[(N, T)], f, indent=2)
            print(f"[SAVE] TRAIN N={N}, T={T}: {len(train_data[(N, T)])} -> {train_json_path}")


if __name__ == "__main__":
    main()
