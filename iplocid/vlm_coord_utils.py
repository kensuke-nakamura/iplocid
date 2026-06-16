# vlm_coord_utils.py
"""
Coordinate helpers for different VLM families (Qwen, Gemma, ...).

Right now both Qwen and Gemma 3 use the same textual coordinate
representation for LASOT-style data (normalized 0..1000 on each axis),
so we simply reuse the Qwen implementations.

If in the future we discover a model that needs a different mapping,
we only have to change the branches in this file.
"""

import ast
import re
from typing import Any, List, Union, Tuple

from utils_qwen import pixel_to_qwen_format  # forward: pixel -> normalized text


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _to_four_floats(box_norm: Any) -> Union[List[float], None]:
    """
    Try to convert the input into a list [x1, y1, x2, y2] in float.

    Accepts:
      - list/tuple of 4 numbers
      - list/tuple of 2 points: [[x1, y1], [x2, y2]]
      - string form like "[x1, y1, x2, y2]" or "(x1, y1, x2, y2)"
      - any string that contains at least four numbers (we use the first four)

    Returns:
        list[4 floats] if successful,
        None otherwise.
    """
    if box_norm is None:
        return None

    try:
        # String-like input
        if isinstance(box_norm, (str, bytes)):
            if isinstance(box_norm, bytes):
                s = box_norm.decode("utf-8", errors="ignore")
            else:
                s = box_norm

            # First, try literal_eval for forms like "[x1, y1, x2, y2]"
            try:
                parsed = ast.literal_eval(s)
                box_norm = parsed
            except Exception:
                # Fallback: extract first 4 numeric tokens from the string
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                if len(nums) >= 4:
                    vals = [float(v) for v in nums[:4]]
                    return vals
                else:
                    return None

        # Now handle list/tuple forms
        if isinstance(box_norm, (list, tuple)):
            # Case 1: [x1, y1, x2, y2]
            if len(box_norm) == 4 and all(not isinstance(v, (list, tuple)) for v in box_norm):
                x1, y1, x2, y2 = [float(v) for v in box_norm]
                return [x1, y1, x2, y2]

            # Case 2: [[x1, y1], [x2, y2]]
            if len(box_norm) == 2 and all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in box_norm):
                x1 = float(box_norm[0][0])
                y1 = float(box_norm[0][1])
                x2 = float(box_norm[1][0])
                y2 = float(box_norm[1][1])
                return [x1, y1, x2, y2]

        return None
    except Exception:
        # Any parsing/typing error is treated as invalid
        return None


def _qwen_like_to_pixel_format(
    args,
    box_norm: Any,
    img_size: Tuple[int, int],
    state: str,
) -> Union[List[int], None]:
    """
    Inverse of pixel_to_qwen_format for xyxy normalized coords (0..1000)
    or pixel coords for Qwen-style models (Qwen, Gemma 3 in our setting).

    Args:
        args:      same object passed to pixel_to_qwen_format (needs args.data_path).
        box_norm:  [x1n, y1n, x2n, y2n] where each is in [0,1000] for LASOT/frames,
                   or already pixel coordinates when "perseg" and state != "GT".
                   It may also be a string representation like "[x1, y1, x2, y2]",
                   or a string that contains four numbers.
        img_size:  (W, H).
        state:     e.g., "NotGT" or "GT".

    Returns:
        [x1, y1, x2, y2] in integer pixel coordinates, or
        None if we cannot reliably obtain 4 valid coordinates.
    """
    W, H = img_size

    # Convert input to 4 floats; if this fails, return None
    vals = _to_four_floats(box_norm)
    if vals is None:
        return None

    x1n, y1n, x2n, y2n = vals

    # Case 1: "perseg" file and not GT -> treat as direct pixel coordinates.
    # In this case pixel_to_qwen_format keeps them as pixels.
    if state != "GT" and "perseg" in args.data_path.split("/")[-1]:
        x1 = int(round(x1n))
        y1 = int(round(y1n))
        x2 = int(round(x2n))
        y2 = int(round(y2n))
    else:
        # Case 2: usual LASOT/frames style, where coordinates are normalized 0..1000
        x1 = int(round((x1n / 1000.0) * W))
        y1 = int(round((y1n / 1000.0) * H))
        x2 = int(round((x2n / 1000.0) * W))
        y2 = int(round((y2n / 1000.0) * H))

    # Clamp and order
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


# ------------------------------------------------------------
# Public API (forward / inverse) used by training & inference
# ------------------------------------------------------------

def pixel_to_vlm_format(
    args,
    box_str: str,
    img_size: Tuple[int, int],
    state: str,
    model_id: str,
):
    """
    Convert a pixel-space bbox string into a model-friendly normalized form.

    Args:
        args:      same args object passed around in IPLoc.
        box_str:   string like "[x1, y1, x2, y2]" in pixel coordinates.
        img_size:  (W, H) from PIL.Image.size.
        state:     "GT" or "NotGT" (mirrors the original API).
        model_id:  HF model id (used to branch per-model if needed).

    Returns:
        Normalized bbox representation compatible with the chosen VLM.
    """
    # For now Gemma 3 and Qwen share the same mapping.
    return pixel_to_qwen_format(args, box_str, img_size, state)


def vlm_to_pixel_format(
    args,
    box_norm: Any,
    img_size: Tuple[int, int],
    state: str,
    model_id: str,
):
    """
    Convert a model-friendly bbox (normalized) back into pixel coordinates.

    Args:
        args:      same args object passed around in IPLoc.
        box_norm:  list or string "[x1n, y1n, x2n, y2n]" (0..1000 scale
                   or pixel coordinates for certain datasets), or any string
                   that contains four numbers describing a box.
        img_size:  (W, H) from PIL.Image.size.
        state:     "GT" or "NotGT" (mirrors the original API).
        model_id:  HF model id (used to branch per-model if needed).

    Returns:
        [x1, y1, x2, y2] in integer pixel coordinates,
        or None if the box cannot be parsed.
    """
    # Same mapping for Qwen and Gemma for now.
    return _qwen_like_to_pixel_format(args, box_norm, img_size, state)
