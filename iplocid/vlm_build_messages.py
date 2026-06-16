# vlm_build_messages.py
# Common builders for IPLoc-style chat messages
#
# - ensure_alternating_roles: enforce user/assistant/user/... alternation
#   for models whose chat templates require strict alternation (e.g., Gemma 3).
# - build_messages: unified builder for both training and inference:
#     * Supports N-shot (multiple reference frames) + T-target (multiple targets).
#     * Supports optional query_box_text (used for identification phase).
#     * In dataset mode, query_box_text (if provided) is ALWAYS normalized by
#       pixel_to_vlm_format_fn to match the reference bbox format.
#
# Backward-compat:
# - build_messages_train -> calls build_messages(...) with N=1 style args
# - build_messages_eval  -> calls build_messages(...) with dataset args

from typing import List, Tuple, Any, Dict, Optional


def ensure_alternating_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure conversation roles alternate in the pattern:
        user / assistant / user / assistant / ...

    If two consecutive turns have the same role, an empty opposite-role turn
    is inserted in between.

    This is required by some chat templates (for example, Gemma 3).
    For other models like Qwen2-VL/Qwen3, inserting empty assistant turns
    is also safe.
    """
    fixed: List[Dict[str, Any]] = []
    last_role = None

    for m in messages:
        role = m.get("role", "user")

        # If the current role is the same as the previous role, insert
        # an empty turn with the opposite role.
        if last_role == role:
            if role == "user":
                fixed.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                    }
                )
                last_role = "assistant"
            else:
                fixed.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": ""}],
                    }
                )
                last_role = "user"

        fixed.append(m)
        last_role = role

    return fixed


def _to_pixel_bbox_str_for_vlm(s: str) -> str:
    """
    Convert various bbox text formats into a pixel-bbox string "[x1,y1,x2,y2]".
    Supported:
      - "(x1,y1),(x2,y2)"
      - "[x1,y1,x2,y2]"
      - "x1,y1,x2,y2"
    """
    # English comments only
    t = str(s).strip()

    # "(x1,y1),(x2,y2)" -> "[x1,y1,x2,y2]"
    if t.startswith("(") and "),(" in t:
        left, right = t.split("),(")
        left = left.strip().lstrip("(")
        right = right.strip().rstrip(")")
        x1, y1 = [float(v) for v in left.split(",")]
        x2, y2 = [float(v) for v in right.split(",")]
        return f"[{x1},{y1},{x2},{y2}]"

    # Already bracket list -> keep
    if t.startswith("[") and t.endswith("]"):
        return t

    # Plain "x1,y1,x2,y2" -> wrap
    parts = [p.strip() for p in t.split(",")]
    if len(parts) == 4:
        x1, y1, x2, y2 = [float(v) for v in parts]
        return f"[{x1},{y1},{x2},{y2}]"

    # Fallback: return original (may fail later)
    return t


def build_messages(
    *,
    element: str,
    bbox: Optional[List[Tuple[str]]] = None,
    image_path: Optional[List[Tuple[str]]] = None,
    data_role: Optional[List[Any]] = None,
    target_index: Optional[int] = None,
    pixel_to_vlm_format_fn=None,
    args=None,
    query_box_text: Optional[str] = None,
    # N=1-style direct inputs (for backward compatibility / simple training)
    reference_img_path: Optional[str] = None,
    ref_box_text: Optional[str] = None,
    query_img_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Unified IPLoc-style message builder.

    Supports two input modes:

    Mode A) Dataset mode (N-shot, T-target; recommended)
      - Provide: bbox, image_path, data_role, target_index, pixel_to_vlm_format_fn, args
      - Builds multi-shot prompt using all frames with role == "reference"
      - Uses frame at target_index as the query image
      - If N=0 (no reference), builds single-turn prompt with query image + label

    Mode B) Simple mode (N=1 style; mainly for legacy training usage)
      - Provide: reference_img_path, ref_box_text, query_img_path
      - Builds: [ref image + label] -> [ref_box_text] -> [query image + label]

    Optional:
      - query_box_text:
          If provided, an extra final user text turn is appended.
          In dataset mode, it is ALWAYS normalized using pixel_to_vlm_format_fn,
          so it matches the reference bbox format.

    Returns:
      (messages, last_box_norm)
        - last_box_norm is meaningful in dataset mode (normalized bbox of the last reference).
        - in simple mode, last_box_norm is "".
    """
    question = f"<ref>{element}</ref>"

    # ------------------------------------------------------------
    # Mode B: Simple mode (N=1 style)
    # ------------------------------------------------------------
    if reference_img_path is not None or query_img_path is not None or ref_box_text is not None:
        if reference_img_path is None or query_img_path is None or ref_box_text is None:
            raise ValueError(
                "Simple mode requires: reference_img_path, ref_box_text, query_img_path."
            )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": reference_img_path},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ref_box_text},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        if query_box_text is not None:
            # Simple mode: keep as-is (caller decides its format)
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(query_box_text)}],
                }
            )

        return messages, ""

    # ------------------------------------------------------------
    # Mode A: Dataset mode (N-shot, T-target)
    # ------------------------------------------------------------
    if bbox is None or image_path is None or data_role is None or target_index is None:
        raise ValueError(
            "Dataset mode requires: bbox, image_path, data_role, target_index."
        )

    if pixel_to_vlm_format_fn is None or args is None:
        raise ValueError(
            "Dataset mode requires: pixel_to_vlm_format_fn and args."
        )

    from PIL import Image

    num_frames = len(bbox)
    if num_frames < 1 or num_frames != len(image_path):
        raise ValueError("bbox and image_path must have the same positive length.")

    if not (0 <= int(target_index) < num_frames):
        raise ValueError(f"target_index {target_index} out of range [0, {num_frames-1}]")

    # Normalize role list to list[str]
    roles: List[str] = []
    for r in data_role:
        if isinstance(r, str):
            roles.append(r)
        elif isinstance(r, (list, tuple)) and len(r) > 0:
            roles.append(str(r[0]))
        else:
            roles.append(str(r))

    ref_indices = [i for i, r in enumerate(roles) if r == "reference"]
    num_reference = len(ref_indices)
    query_idx = int(target_index)

    # Case 1: N = 0 (no reference frames)
    if num_reference == 0:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path[query_idx][0]},
                    {"type": "text", "text": question},
                ],
            }
        ]

        if query_box_text is not None:
            img_q = Image.open(image_path[query_idx][0])
            img_size_q = img_q.size
            pixel_bbox_str = _to_pixel_bbox_str_for_vlm(query_box_text)
            query_box_norm = pixel_to_vlm_format_fn(args, pixel_bbox_str, img_size_q, "NotGT")

            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{query_box_norm}"}],
                }
            )

        return messages, ""

    # Case 2: N >= 1 (multi-shot prompt)
    first_ref_idx = ref_indices[0]
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path[first_ref_idx][0]},
                {"type": "text", "text": question},
            ],
        }
    ]

    last_box_norm = ""

    for j, idx in enumerate(ref_indices):
        img = Image.open(image_path[idx][0])
        img_size = img.size

        box_norm = pixel_to_vlm_format_fn(args, bbox[idx][0], img_size, "NotGT")
        last_box_norm = box_norm

        # Reference bbox text (always normalized)
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{box_norm}"}],
            }
        )

        # Next image: next reference, or the query (last)
        if j < num_reference - 1:
            next_idx = ref_indices[j + 1]
        else:
            next_idx = query_idx

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path[next_idx][0]},
                    {"type": "text", "text": question},
                ],
            }
        )

    # Append query bbox text (ALWAYS normalized in dataset mode)
    if query_box_text is not None:
        img_q = Image.open(image_path[query_idx][0])
        img_size_q = img_q.size
        pixel_bbox_str = _to_pixel_bbox_str_for_vlm(query_box_text)
        query_box_norm = pixel_to_vlm_format_fn(args, pixel_bbox_str, img_size_q, "NotGT")

        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{query_box_norm}"}],
            }
        )

    return messages, last_box_norm


# -------------------------------------------------------------------------
# Backward compatibility wrappers
# -------------------------------------------------------------------------

def build_messages_train(
    reference_img_path: str,
    element: str,
    ref_box_text: str,
    query_img_path: str,
    query_box_text: str = None,
) -> List[Dict[str, Any]]:
    """
    Backward-compat wrapper. Prefer build_messages(...).
    """
    messages, _ = build_messages(
        element=element,
        reference_img_path=reference_img_path,
        ref_box_text=ref_box_text,
        query_img_path=query_img_path,
        query_box_text=query_box_text,
    )
    return messages


def build_messages_eval(
    element: str,
    bbox: List[Tuple[str]],
    image_path: List[Tuple[str]],
    data_role: List[Any],
    target_index: int,
    pixel_to_vlm_format_fn,
    args,
    query_box_text: str = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Backward-compat wrapper. Prefer build_messages(...).
    """
    return build_messages(
        element=element,
        bbox=bbox,
        image_path=image_path,
        data_role=data_role,
        target_index=target_index,
        pixel_to_vlm_format_fn=pixel_to_vlm_format_fn,
        args=args,
        query_box_text=query_box_text,
    )
