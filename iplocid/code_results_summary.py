#!/usr/bin/env python3
# code_results_summary.py
#
# Corrected summary + ranking tool (with inference_data column)
# - Accepts: python3 code_results_summary.py <results_dir> --no-plot --min_step 1500
# - Writes:
#     results/<data_name>/summary.txt
#     results/<data_name>/ranking_by_full_IoU.txt
#
# Ranking output format (spreadsheet-friendly):
#   Columns: inference_data, training_data, backbone, label, best_full_iou, best_step, best_algo, iou_tp, iou_tpfn, TP, FP, TN, FN
#
# English comments only.

import os
import re
import glob
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _stem_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def parse_first_col(s: str) -> Tuple[str, int]:
    """Parse '<series>_step####' -> (series_id, step). If no step suffix, treat as (s, 0)."""
    s = str(s)
    m = re.search(r"_step(\d+)$", s)
    if not m:
        return s, 0
    return s[: s.rfind("_step")], int(m.group(1))


def _basename_dir(p: str) -> str:
    """Return the last directory name (data_name) from a results_dir path."""
    s = os.path.normpath(str(p))
    return os.path.basename(s)


def normalize_label_with_trial_id(label: str) -> str:
    """Ensure labels without explicit '.<int>' suffix get '.1' appended."""
    s = str(label).strip()
    if re.search(r"\.\d+$", s):
        return s
    return s + ".1"


def _extract_NT_from_data_name(data_name: str) -> Tuple[str, str]:
    """Extract N and T from strings like 'LASOT_1shot_T2_classwise-split_test'."""
    s = str(data_name)
    m = re.search(r"_(\d+)shot_T(\d+)", s)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _has_explicit_NT_token(tokens: List[str]) -> bool:
    """Return True if any token already looks like 'N4T2'."""
    for t in tokens:
        if re.fullmatch(r"N\d+T\d+", str(t)):
            return True
    return False


def _extract_query_tag(tokens: List[str]) -> str:
    """Extract label-like query tag such as 'query1', 'query2', ..."""
    for t in tokens:
        if re.fullmatch(r"query\d+(\.\d+)?", str(t)):
            return str(t)
    return ""


def _extract_model_token(tokens: List[str]) -> str:
    """Extract backbone/model token from split algo tokens."""
    for t in tokens:
        s = str(t)
        sl = s.lower()
        if (
            "qwen" in sl
            or "gemma" in sl
            or "llava" in sl
            or "internvl" in sl
        ):
            return s
    return ""


def summarize_metrics_jsons(metrics_paths: List[str], out_path: str):
    """Write a compact summary CSV (kept minimal)."""
    rows = []
    for p in metrics_paths:
        js = _read_json(p)
        if not isinstance(js, dict):
            continue
        algo = _stem_no_ext(p)

        rows.append(
            {
                "Algorithm": algo,
                "TP": _safe_int(js.get("TP")) or 0,
                "FP": _safe_int(js.get("FP")) or 0,
                "TN": _safe_int(js.get("TN")) or 0,
                "FN": _safe_int(js.get("FN")) or 0,
                "IoU_TP": _safe_float(js.get("miou_bbox_TP")) or 0.0,
                "IoU_TPFN": _safe_float(js.get("miou_bbox_T")) or 0.0,
                "FullIoU": _safe_float(js.get("mIoU_full_for_all_targets")) or 0.0,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Algorithm,TP,FP,TN,FN,IoU_TP,IoU_TPFN,FullIoU\n")
        for r in rows:
            f.write(
                f"{r['Algorithm']},{r['TP']},{r['FP']},{r['TN']},{r['FN']},"
                f"{r['IoU_TP']:.6f},{r['IoU_TPFN']:.6f},{r['FullIoU']:.6f}\n"
            )


def _pick_best_for_series(
    entries: List[Dict[str, object]],
    min_step: int,
) -> Optional[Dict[str, object]]:
    """
    Pick one entry for a series.

    Rule:
    - If this series has step-based entries (any step > 0):
        * Pick the best FullIoU among entries with step >= min_step.
          Tie-break: larger step.
        * If there is no entry with step >= min_step, pick the last step (max step).
    - Otherwise (non-step series): pick best FullIoU across all entries (tie-break: larger step).
    """
    if not entries:
        return None

    has_step_series = any(int(e.get("step", 0)) > 0 for e in entries)

    if has_step_series:
        eligible = [e for e in entries if int(e.get("step", 0)) >= int(min_step)]
        if eligible:
            eligible_sorted = sorted(
                eligible,
                key=lambda e: (-float(e["full_iou"]), -int(e["step"])),
            )
            return eligible_sorted[0]

        last = max(entries, key=lambda e: int(e.get("step", 0)))
        return last

    best = sorted(entries, key=lambda e: (-float(e["full_iou"]), -int(e["step"])))[0]
    return best


def rank_by_best_full_iou(metrics_paths: List[str], results_dir: str, min_step: int):
    """Write ranking_by_full_IoU.txt with inference_data (data_name) as the first column."""
    data_name = _basename_dir(results_dir)

    data_name_from_json = ""
    if metrics_paths:
        js0 = _read_json(metrics_paths[0])
        if isinstance(js0, dict):
            data_name_from_json = str(js0.get("data_name", "")).strip()

    if data_name_from_json:
        data_name = data_name_from_json

    data_N, data_T = _extract_NT_from_data_name(data_name)

    series_entries: Dict[str, List[Dict[str, object]]] = {}

    for p in metrics_paths:
        algo = _stem_no_ext(p)
        series_id, step = parse_first_col(algo)

        js = _read_json(p)
        if not isinstance(js, dict):
            continue

        full_iou = _safe_float(js.get("mIoU_full_for_all_targets"))
        if full_iou is None:
            continue

        entry = {
            "series_id": series_id,
            "step": int(step),
            "algo": algo,
            "full_iou": float(full_iou),
            "TP": _safe_int(js.get("TP")) or 0,
            "FP": _safe_int(js.get("FP")) or 0,
            "TN": _safe_int(js.get("TN")) or 0,
            "FN": _safe_int(js.get("FN")) or 0,
            "iou_tp": _safe_float(js.get("miou_bbox_TP")) or 0.0,
            "iou_tpfn": _safe_float(js.get("miou_bbox_T")) or 0.0,
        }

        series_entries.setdefault(series_id, []).append(entry)

    best_by_series: Dict[str, Dict[str, object]] = {}
    for sid, entries in series_entries.items():
        chosen = _pick_best_for_series(entries, min_step=min_step)
        if chosen is None:
            continue
        best_by_series[sid] = {
            "best_full": float(chosen["full_iou"]),
            "best_step": int(chosen["step"]),
            "best_algo": str(chosen["algo"]),
            "TP": int(chosen["TP"]),
            "FP": int(chosen["FP"]),
            "TN": int(chosen["TN"]),
            "FN": int(chosen["FN"]),
            "iou_tp": float(chosen["iou_tp"]),
            "iou_tpfn": float(chosen["iou_tpfn"]),
        }

    ranked = sorted(
        best_by_series.items(),
        key=lambda kv: (-float(kv[1]["best_full"]), -int(kv[1]["best_step"]), kv[0]),
    )

    out = os.path.join(results_dir, "ranking_by_full_IoU.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Ranking by best FullTargetIoU_mean over steps\n")
        f.write(f"# results_dir: {results_dir}\n")
        f.write(
            "inference_data, training_data, backbone, label, best_full_iou, best_step, best_algo, iou_tp, iou_tpfn, TP, FP, TN, FN\n"
        )

        for _sid, v in ranked:
            toks = str(v["best_algo"]).split("_")

            nt_token = ""
            for t in toks:
                if re.fullmatch(r"N\d+T\d+", str(t)):
                    nt_token = str(t)
                    break

            has_explicit_nt = bool(nt_token)

            if not has_explicit_nt and data_N and data_T:
                nt_token = f"N{data_N}T{data_T}"

            model_token = _extract_model_token(toks)
            query_tag = _extract_query_tag(toks)

            # Reorder only when NT is missing:
            #   <data_name> _ <label> _ <model> _ ...
            # -> <data_name> _ <NT> _ <model> _ <label>
            if not has_explicit_nt:
                training_data = nt_token
                backbone = model_token if model_token else ""
                label = normalize_label_with_trial_id(query_tag) if query_tag else ""
            else:
                # Keep existing behavior for labels that already contain explicit NT
                label = normalize_label_with_trial_id(toks[0]) if len(toks) > 0 else ""
                training_data = toks[1] if len(toks) > 1 else ""
                backbone = toks[2] if len(toks) > 2 else ""

            f.write(
                f"{data_name},"
                f"{training_data},{backbone},{label},"
                f"{float(v['best_full']):.6f},step{int(v['best_step'])},{v['best_algo']},"
                f"{float(v['iou_tp']):.6f},{float(v['iou_tpfn']):.6f},"
                f"{int(v['TP'])},{int(v['FP'])},{int(v['TN'])},{int(v['FN'])}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--min_step",
        type=int,
        default=1500,
        help="For step-series metrics (algo ending with _step####), pick best only among step>=min_step; "
             "if none exists, pick the last step. Non-step metrics are kept as-is.",
    )
    args = ap.parse_args()

    results_dir = os.path.normpath(args.results_dir)
    metrics_glob = os.path.join(results_dir, "metrics", "*.json")
    metrics = sorted([p for p in glob.glob(metrics_glob) if os.path.isfile(p)], key=natural_key)

    if not metrics:
        raise RuntimeError(f"No metrics json found under: {metrics_glob}")

    summarize_metrics_jsons(metrics, os.path.join(results_dir, "summary.txt"))
    rank_by_best_full_iou(metrics, results_dir, min_step=int(args.min_step))


if __name__ == "__main__":
    main()
