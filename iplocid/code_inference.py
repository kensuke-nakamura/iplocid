#!/usr/bin/env python3
# code_inference.py
#
# Inference script for IPLoc-style localization with Qwen2/Qwen3/Gemma3.
#
# Outputs:
# 1) Metrics JSON:
#    ./results/<data_name>/metrics/<safe_label>.json
# 2) Raw generated texts JSON:
#    ./results/<data_name>/generated_texts/<safe_label>.json
#    - stores raw generation + raw IoU per evaluated target entry
#    - also stores reference image paths + GT bboxes for the sample
#
# Notes:
# - Chunk arguments are accepted for compatibility but ignored for saving paths.
# - Raw outputs are made JSON-safe (no torch.Tensor objects).
#
# English comments only.

import os
import sys
import json
import random
import argparse
import time
import re
import gc

import numpy as np

os.environ["MKL_THREADING_LAYER"] = "INTEL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch
from peft import PeftModel
from PIL  import Image
import torchvision.ops as ops

from loc_dataset import get_dataloader
from utils_qwen  import eval_bbox

# local
from vlm_loader              import load_model_and_processor
from vlm_build_messages      import build_messages, ensure_alternating_roles
from vlm_coord_utils         import pixel_to_vlm_format, vlm_to_pixel_format, _to_four_floats
from vlm_external_query_set  import EXTERNAL_QUERY_SET

#DEVICE = "cuda:0"

RESET = "\033[0m"
GREEN = "\033[32m"

random.seed(42)

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# External instruction templates for --external_query.
# NOTE: --external_query accepts either:
# - a numeric key ("1".."4") to select from EXTERNAL_QUERY_SET, or
# - a raw instruction string to inject directly.
EXTERNAL_QUERY_SET = EXTERNAL_QUERY_SET


def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _should_stop_after_n_samples(args, samples_seen_1based: int) -> bool:
    if getattr(args, "num_samples", None) is None:
        return False
    n = int(args.num_samples)
    if n <= 0:
        return False
    return samples_seen_1based >= n


def _print_lap(iii: int, total: int, t_start: float) -> float:
    _cuda_sync_if_available()
    t_end = time.perf_counter()
    sec_per_data = t_end - t_start
    remaining = (total - (iii + 1))
    eta_sec = sec_per_data * remaining
    print(
        f"data {iii}/{total} done | "
        f"time {sec_per_data:.2f}s/data | ETA {eta_sec/60:.1f} min. \n"
    )
    return sec_per_data


def _safe_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "model"


def _default_model_label(args) -> str:
    if getattr(args, "name", None) and str(args.name).strip():
        return str(args.name).strip()
    mid = str(getattr(args, "model_id", "model"))
    return mid.split("/")[-1] if "/" in mid else mid


def _extract_bbox_text_first4floats(gen: str):
    if not gen:
        return None
    nums = _FLOAT_RE.findall(gen)
    if len(nums) < 4:
        return None
    try:
        vals = [float(nums[i]) for i in range(4)]
    except Exception:
        return None
    return f"[{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}]"


def _is_nonzero_area_box(b, eps: float = 0.0) -> bool:
    # English comments only
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return False
    w = x2 - x1
    h = y2 - y1
    return (w > eps) and (h > eps)


def PN_interpreter(gen: str):
    meta = {"matched_rule": None, "raw": gen if gen is not None else ""}

    # If empty, treat as negative (no decision / no bbox)
    if not gen or not str(gen).strip():
        meta["matched_rule"] = "negative:empty_generation"
        return "negative", meta

    s = gen.strip().lower()

    # Explicit yes/no tokens
    if re.search(r"\byes\b", s):
        meta["matched_rule"] = "positive:yes_token"
        return "positive", meta
    if re.search(r"\bno\b", s):
        meta["matched_rule"] = "negative:no_token"
        return "negative", meta

    # Text-level patterns
    positive_patterns = [r"\bsame\s+object\b", r"\bidentical\b"]
    negative_patterns = [
        r"\bnot\s+the\s+same\b",
        r"\bdifferent\b",
        r"\bnot\s+found\b",
        r"\bnot\s+present\b",
        r"\bno\s+match\b",
        r"\bno\b",  # conservative: any standalone 'no' not caught by token rule
    ]

    for p in negative_patterns:
        if re.search(p, s):
            meta["matched_rule"] = f"negative:text:{p}"
            return "negative", meta

    for p in positive_patterns:
        if re.search(p, s):
            meta["matched_rule"] = f"positive:text:{p}"
            return "positive", meta

    # BBox-based fallback:
    # - If we cannot extract 4 floats -> negative
    # - If extracted but zero-area -> negative
    bbox_text = _extract_bbox_text_first4floats(gen)
    if bbox_text is None:
        meta["matched_rule"] = "negative:no_bbox_4floats"
        return "negative", meta

    tmp = _to_four_floats(bbox_text)
    if tmp is None or (not _is_nonzero_area_box(tmp, eps=0.0)):
        meta["matched_rule"] = "negative:bbox_null_or_zero_area"
        return "negative", meta

    # Otherwise, default positive
    meta["matched_rule"] = "positive:default_bbox_present"
    return "positive", meta


def load_and_resize_image(path: str, max_side: int):
    """
    Load an image, and if max(width, height) > max_side, downscale keeping aspect ratio.

    Returns:
      resized_img (PIL.Image)
      (orig_w, orig_h)
      scale (float)  # new / old (same for x and y)
    """
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    m = max(orig_w, orig_h)
    if max_side is None or max_side <= 0 or m <= max_side:
        return img, (orig_w, orig_h), 1.0

    scale = float(max_side) / float(m)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (orig_w, orig_h), scale


def _parse_bbox_4ints_from_str(bbox_str: str):
    if not bbox_str:
        return None
    nums = _FLOAT_RE.findall(bbox_str)
    if len(nums) < 4:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]


def _scale_bbox_str_keep_axis(bbox_str: str, scale: float) -> str:
    vals = _parse_bbox_4ints_from_str(bbox_str)
    if vals is None:
        return bbox_str
    scaled = [v * scale for v in vals]
    return f"[{scaled[0]}, {scaled[1]}, {scaled[2]}, {scaled[3]}]"


def _cleanup_cuda_tensors(*objs) -> None:
    try:
        for o in objs:
            if o is None:
                continue
            del o
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_image_id_value(image_id, idx: int):
    try:
        v = image_id[idx]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return v[0]
        return v
    except Exception:
        return idx


def _json_safe_scalar(x):
    """Convert common scalar-like objects (torch/numpy) to JSON-serializable Python types."""
    try:
        if torch.is_tensor(x):
            return x.item()
    except Exception:
        pass
    try:
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x


def _is_qwen2_or_qwen3(model_id: str) -> bool:
    s = str(model_id)
    return ("Qwen/Qwen2" in s) or ("Qwen/Qwen3" in s)


def _resolve_external_query_arg(raw_value: str):
    """Resolve --external_query.

    If raw_value is a known numeric key in EXTERNAL_QUERY_SET, return (key, template_string).
    Otherwise return (None, raw_value_stripped).
    """
    if raw_value is None:
        return None, None
    s = str(raw_value).strip()
    if s == "":
        return None, None
    if s in EXTERNAL_QUERY_SET:
        return s, EXTERNAL_QUERY_SET[s]
    return None, s


def _inject_external_query_into_messages(messages, external_query: str):
    # English comments only
    if external_query is None:
        return messages
    external_query = str(external_query).strip()
    if external_query == "":
        return messages

    # Try to append to an existing system message; otherwise insert a new one.
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        # Find first system role
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "system":
                continue

            c = m.get("content")
            if isinstance(c, str):
                m["content"] = c.rstrip() + "\n\n" + external_query
                return messages
            if isinstance(c, list):
                # Append as a text chunk
                c.append({"type": "text", "text": external_query})
                return messages
            # Unknown content type -> overwrite conservatively
            m["content"] = external_query
            return messages

        # No system message found -> insert
        messages = list(messages)
        messages.insert(0, {"role": "system", "content": external_query})
        return messages

    # If messages are not in expected format, fall back to returning as-is.
    return messages


def eval_model(args):
    # Basic sanity check for JSON
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            print(f"[ERROR] JSON '{args.data_path}' is empty or not a list.")
            sys.exit(1)
        first = raw_data[0]
        if not isinstance(first, dict):
            print(f"[ERROR] JSON '{args.data_path}' first item is not a dict.")
            sys.exit(1)
        if "role" not in first:
            print(
                f"[WARN] JSON '{args.data_path}' has no 'role' field. "
                f"Assuming (n-1) references + last target for each sample."
            )
    except Exception as e:
        print(f"[ERROR] Failed to inspect JSON '{args.data_path}': {e}")
        sys.exit(1)

    dataloader = get_dataloader(args)

    # Three IoU lists (metrics; bbox-only subset, prior-work compatible)
    iou_bbox_all = []
    iou_bbox_T = []
    iou_bbox_TP = []

    # Full IoU for all GT-positive targets (role == "positive-image"):
    # if bbox is missing or invalid -> IoU=0
    full_iou_for_all_targets = []

    # Raw outputs for post-hoc evaluation
    outputs_records = []

    # Confusion counts
    conf_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    num_data = 0

    model, processor = load_model_and_processor(args.model_id)
    if args.lora_weights_path:
        model = PeftModel.from_pretrained(model, args.lora_weights_path)

    model.eval()
    input_device = next(model.parameters()).device

    model_id = args.model_id
    is_gemma = args.model_id.startswith("google/gemma-3-")

    external_query_key, external_query_text = _resolve_external_query_arg(getattr(args, "external_query", None))
    has_external_query = (external_query_text is not None)

    # Temporarily allow external_query for all models during compatibility testing.
    # if has_external_query and (not _is_qwen2_or_qwen3(model_id)):
    #     raise ValueError(f"--external_query is supported only for Qwen2/Qwen3. Got model_id={model_id}")
    if has_external_query:
        print(f"[INFO] Temporarily allowing --external_query for all models: model_id={model_id}")

    model_label = _safe_name(_default_model_label(args))

    total = dataloader.__len__()
    print(f"{total} samples")
    if args.num_samples is not None:
        print(f"[INFO] --num_samples={args.num_samples}")
    print(f"[INFO] --max_side={args.max_side}")
    print(f"[INFO] model_label={model_label}")

    if has_external_query:
        print("[INFO] external_query is enabled.")
        if external_query_key is not None:
            print(f"[INFO] external_query key: {external_query_key}")
        print("[INFO] external_query (resolved text):")
        print(external_query_text)

    pred_bbox_note = "A bounding box generated in the model coordinate system and converted to pixel coordinates."

    last_iii = -1

    for iii, data_item in enumerate(dataloader):
        last_iii = iii
        _cuda_sync_if_available()
        t_data_start = time.perf_counter()

        print(f"data index: {iii}")

        element, bbox, image_path, image_id, data = data_item
        element = element[0]

        # Extract role
        data_role = None
        if isinstance(data, dict) and "role" in data:
            data_role = data["role"]
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            if isinstance(data[0], dict) and "role" in data[0]:
                data_role = data[0]["role"]

        if data_role is None:
            print(f"[ERROR] 'role' is missing in data item #{iii}.")
            sys.exit(1)

        roles = []
        for r in data_role:
            if isinstance(r, str):
                roles.append(r)
            elif isinstance(r, (list, tuple)) and len(r) > 0:
                roles.append(str(r[0]))
            else:
                roles.append(str(r))

        ref_indices = [idx for idx, r in enumerate(roles) if r == "reference"]
        target_indices = [idx for idx, r in enumerate(roles) if r != "reference"]

        if len(target_indices) == 0:
            print(f"[WARN] No query in data item #{iii}, skipping.")
            _print_lap(iii, total, t_data_start)
            if _should_stop_after_n_samples(args, samples_seen_1based=(iii + 1)):
                print(f"[INFO] Reached num_samples={args.num_samples}. Stopping.")
                break
            continue

        # Build reference info for this sample (JSON-serializable)
        references_info = []
        for ref_idx in ref_indices:
            ref_img_path = image_path[ref_idx][0]
            ref_gt_bbox = bbox[ref_idx][0]
            ref_img_id = _extract_image_id_value(image_id, ref_idx)
            ref_img_id = _json_safe_scalar(ref_img_id)
            references_info.append(
                {
                    "image_id": ref_img_id,
                    "image_path": str(ref_img_path),
                    "gt_bbox_pixel_format": str(ref_gt_bbox),
                }
            )

        print("\telement", element)
        print("\tbbox", bbox)
        print("\timage_path", image_path)
        print("\timage_id", image_id)
        print("\tdata_role", roles)
        print(f"\tnum_reference={len(ref_indices)}, num_target={len(target_indices)}")

        for local_t, target_frame_idx in enumerate(target_indices):
            print(f"\t  -> target {local_t} (frame_idx={target_frame_idx})")

            messages, _last_box_norm = build_messages(
                element=element,
                bbox=bbox,
                image_path=image_path,
                data_role=data_role,
                target_index=target_frame_idx,
                pixel_to_vlm_format_fn=lambda a, b, c, d: pixel_to_vlm_format(a, b, c, d, model_id=model_id),
                args=args,
            )

            if has_external_query:
                messages = _inject_external_query_into_messages(messages, external_query_text)

            if is_gemma:
                messages = ensure_alternating_roles(messages)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            images_for_model = []
            for msg in messages:
                content = msg.get("content", None)

                # Qwen-style messages typically store multimodal chunks in a list.
                # Some templates may store plain text as a string.
                if not isinstance(content, list):
                    continue

                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "image":
                        p = c["image"]
                        img_resized, (_ow, _oh), _sc = load_and_resize_image(
                            p, max_side=args.max_side
                        )
                        images_for_model.append(img_resized)

            inputs = None
            try:
                if is_gemma:
                    inputs = processor(
                        text=[text],
                        images=images_for_model,
                        padding=True,
                        return_tensors="pt",
                    ).to(input_device)
                else:
                    inputs = processor(
                        text=[text],
                        images=images_for_model,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    ).to(input_device)
            except Exception as e:
                print(f"[WARN] processor(...) failed at data {iii}, target {local_t}: {e}")
                _cleanup_cuda_tensors(inputs)
                continue

            only_gen = ""
            generated_ids = None

            with torch.no_grad():
                try:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    only_gen = output_text[0]
                    print(f"{GREEN}output_text:\n{only_gen}{RESET}")
                except Exception as e:
                    print(f"[WARN] model.generate failed at data {iii}, target {local_t}: {e}")

            # Parse prediction (bbox + PN)
            pn_label, pn_meta = PN_interpreter(only_gen)
            bbox_text = _extract_bbox_text_first4floats(only_gen)

            pred_box_norm = None
            if bbox_text:
                strict_ok = eval_bbox(bbox_text)
                if strict_ok:
                    tmp = _to_four_floats(bbox_text)
                    if _is_nonzero_area_box(tmp, eps=0.0):
                        pred_box_norm = tmp

            query_path = image_path[target_frame_idx][0]
            gt_box_str_orig = bbox[target_frame_idx][0]
            img_id_val = _extract_image_id_value(image_id, target_frame_idx)
            img_id_val = _json_safe_scalar(img_id_val)
            sample_idx_val = int(iii)

            # Prepare pred bbox in pixel format (independent of IoU)
            pred_bbox_pixel_format = None
            pred_box = None
            if pred_box_norm is not None:
                pred_box = [
                    pred_box_norm[0],
                    pred_box_norm[1],
                    pred_box_norm[2],
                    pred_box_norm[3],
                ]
                try:
                    # Use PIL to get original size (no image writing)
                    sz = Image.open(query_path).size
                    pred_bbox_pixel_format = vlm_to_pixel_format(
                        args,
                        pred_box,
                        sz,
                        "NotGT",
                        model_id=model_id,
                    )
                except Exception:
                    pred_bbox_pixel_format = None

            # Record per-entry data (order matters)
            out_rec = {
                "sample": sample_idx_val,
                "image_id": img_id_val,
                "role": str(roles[target_frame_idx]),
                "image_path": str(query_path),

                "references": references_info,

                "text": str(only_gen),

                "external_query": (str(external_query_text) if has_external_query else None),
                "external_query_key": (str(external_query_key) if external_query_key is not None else None),

                "pred_bbox_pixel_format": pred_bbox_pixel_format,
                "gt_bbox_pixel_format": str(gt_box_str_orig),
                "pred_bbox_pixel_format_note": pred_bbox_note,

                "pn_label": str(pn_label),
                "pn_meta": pn_meta,
                "confusion": None,

                "iou_raw": None,
                "iou_status": "not_computed",
                "iou_error": None,

                # Full IoU definition for targets:
                # - positive-image role only
                # - bbox missing / invalid / conversion failed -> 0.0
                "full_iou_for_all_targets": None,
            }

            # Compute IoU only if pred bbox exists and GT can be converted
            iou_value = None
            if pred_box is not None:
                try:
                    query_img_resized, (_ow, _oh), sc = load_and_resize_image(
                        query_path, max_side=args.max_side
                    )
                    gt_box_str_scaled = _scale_bbox_str_keep_axis(gt_box_str_orig, sc)

                    GT = pixel_to_vlm_format(
                        args,
                        gt_box_str_scaled,
                        query_img_resized.size,
                        "GT",
                        model_id=model_id,
                    )
                    boxes_labels = torch.tensor(GT).unsqueeze(0)
                    boxes_preds = torch.tensor(pred_box).unsqueeze(0)

                    iou = ops.box_iou(boxes_preds, boxes_labels)
                    iou_value = float(iou.item())

                    out_rec["iou_raw"] = iou_value
                    out_rec["iou_status"] = "ok"
                except Exception as e:
                    out_rec["iou_raw"] = None
                    out_rec["iou_status"] = "failed"
                    out_rec["iou_error"] = str(e)
                    iou_value = None
            else:
                out_rec["iou_raw"] = None
                out_rec["iou_status"] = "no_strict_bbox"
                iou_value = None

            # Full IoU for ALL GT-positive targets (role == "positive-image")
            if roles[target_frame_idx] == "positive-image":
                full_iou = float(iou_value) if (iou_value is not None) else 0.0
                full_iou_for_all_targets.append(full_iou)
                out_rec["full_iou_for_all_targets"] = full_iou

            # Update IoU metrics (bbox-only, prior-work compatible)
            if iou_value is not None:
                iou_bbox_all.append(iou_value)
                if roles[target_frame_idx] == "positive-image":
                    iou_bbox_T.append(iou_value)
                    if pn_label == "positive":
                        iou_bbox_TP.append(iou_value)

            # Confusion label
            if roles[target_frame_idx] == "positive-image":
                confusion_metric = "TP" if pn_label == "positive" else "FN"
            else:
                confusion_metric = "TN" if pn_label == "negative" else "FP"

            out_rec["confusion"] = confusion_metric

            # Update confusion counts
            num_data += 1
            conf_counts[confusion_metric] = conf_counts.get(confusion_metric, 0) + 1

            outputs_records.append(out_rec)

            _cleanup_cuda_tensors(inputs, generated_ids)

        _print_lap(iii, total, t_data_start)

        if _should_stop_after_n_samples(args, samples_seen_1based=(iii + 1)):
            print(f"[INFO] Reached num_samples={args.num_samples}. Stopping.")
            break

    samples = (last_iii + 1) if last_iii >= 0 else 0
    return (
        iou_bbox_all,
        iou_bbox_T,
        iou_bbox_TP,
        full_iou_for_all_targets,
        outputs_records,
        samples,
        num_data,
        conf_counts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Chunk options (compat only)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--curr_chunk", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    # IO options
    parser.add_argument("--output_file", type=str, default="./outputs")
    parser.add_argument("--data_path", type=str, default="./Loc/data/path_to_test.json")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--lora_weights_path", type=str, default=None)

    # Name used in output filenames (safe-processed internally)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    # Run control
    parser.add_argument("--num_samples", type=int, default=None)

    # Overwrite behavior
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, run inference/evaluation even when output JSONs already exist.",
    )

    # Image downscale + generation length
    parser.add_argument("--max_side", type=int, default=640)
    parser.add_argument("--max_new_tokens", type=int, default=150)

    # External query (optional; Qwen2/Qwen3 only)
    parser.add_argument(
        "--external_query",
        type=str,
        default=None,
        help="Optional external query injected as a SYSTEM message (Qwen2/Qwen3 only).",
    )

    args = parser.parse_args()

    # data_name = JSON filename without extension
    args.data_name = (args.data_path.split("/")[-1]).split(".")[0]

    # Resolve external query here as well (needed for saving JSON payload)
    external_query_key, external_query_text = _resolve_external_query_arg(
        getattr(args, "external_query", None)
    )

    # Determine output label/path BEFORE running inference
    safe_label = _safe_name(_default_model_label(args))

    gen_dir = os.path.join("./results", args.data_name, "generated_texts")
    os.makedirs(gen_dir, exist_ok=True)
    gen_out_path = os.path.join(gen_dir, f"{safe_label}.json")

    # Metrics path is also needed before running inference for the skip check
    metrics_dir = os.path.join("./results", args.data_name, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_out_path = os.path.join(metrics_dir, f"{safe_label}.json")

    # Early-exit if both outputs exist and overwrite is not requested
    if (not args.overwrite) and os.path.exists(gen_out_path) and os.path.exists(metrics_out_path):
        print("[INFO] Output files already exist. Skipping inference/evaluation.")
        print(f"[INFO] generated_texts: {gen_out_path}")
        print(f"[INFO] metrics:         {metrics_out_path}")
        print("[INFO] Use --overwrite to recompute.")
        sys.exit(0)

    # Run inference
    (
        iou_bbox_all,
        iou_bbox_T,
        iou_bbox_TP,
        full_iou_for_all_targets,
        outputs_records,
        samples,
        num_data,
        conf_counts,
    ) = eval_model(args)

    def _mean(xs) -> float:
        return float(sum(xs) / max(1, len(xs))) if xs else 0.0

    # ---- Save metrics JSON ----
    metrics_payload = {
        "result_id": safe_label,
        "data_name": args.data_name,
        "data_path": args.data_path,
        "model_id": args.model_id,
        "image_resize": args.max_side,
        "samples": int(samples),
        "num_data": int(num_data),

        # BBox-only IoU stats (prior-work compatible)
        "num_iou_bbox": int(len(iou_bbox_all)),
        "num_iou_bbox_T": int(len(iou_bbox_T)),
        "num_iou_bbox_TP": int(len(iou_bbox_TP)),
        "miou_bbox": _mean(iou_bbox_all),
        "miou_bbox_T": _mean(iou_bbox_T),
        "miou_bbox_TP": _mean(iou_bbox_TP),
        "mIoU": _mean(iou_bbox_TP),

        # Full IoU over all GT-positive targets (bbox-missing -> 0.0)
        "num_full_iou_for_all_targets": int(len(full_iou_for_all_targets)),
        "mIoU_full_for_all_targets": _mean(full_iou_for_all_targets),

        # Confusion counts
        "TP": int(conf_counts.get("TP", 0)),
        "TN": int(conf_counts.get("TN", 0)),
        "FP": int(conf_counts.get("FP", 0)),
        "FN": int(conf_counts.get("FN", 0)),
    }

    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved metrics: {metrics_out_path}")

    # ---- Save raw generated texts JSON ----
    gen_payload = {
        "data_path": args.data_path,
        "model_id": args.model_id,
        "image_resize": args.max_side,
        "samples": int(samples),
        "external_query": (str(external_query_text) if external_query_text is not None else None),
        "external_query_key": (str(external_query_key) if external_query_key is not None else None),
        #"external_query_set": EXTERNAL_QUERY_SET,
        "outputs": outputs_records,
    }

    with open(gen_out_path, "w", encoding="utf-8") as f:
        json.dump(gen_payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved generated texts: {gen_out_path}")
