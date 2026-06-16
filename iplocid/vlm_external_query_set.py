# vlm_external_query_set.py
#

# We evaluated four types of external instructions:
#
# (1) Two-line explicit format:
#     A strictly constrained two-line output specifying bbox and identity.
#
# (2) Single-line compact format:
#     A highly compact one-line output: [x1,y1,x2,y2], YES/NO.
#
# (3) Structured two-step reasoning format:
#     A two-stage instruction guiding localization and identity verification.
#
# (4) Minimal constraint format:
#     A weakly constrained instruction relying on the model’s intrinsic reasoning ability.


EXTERNAL_QUERY_SET = {
    "1": (
        "SYSTEM: You are performing visual localization and identity verification.\n"
        "Output exactly two lines.\n"
        "Line 1: bbox=[x1,y1,x2,y2] in pixels for the LAST (target) image, inferred from the reference images/labels/bboxes.\n"
        "Line 2: same_object=YES or same_object=NO indicating whether ALL boxes refer to the same object identity.\n"
        "Do not output any other text."
    ),
    "2": (
        "SYSTEM: Perform visual localization on the LAST (target) image using the reference images/labels/bboxes.\n"
        "Return EXACTLY ONE LINE and NOTHING ELSE.\n"
        "Output format must be: [x1, y1, x2, y2], YES_or_NO\n"
        "- [x1, y1, x2, y2] are pixel coordinates for the target bbox.\n"
        "- YES_or_NO is either YES or NO, indicating whether ALL boxes refer to the same object identity.\n"
        "Do not include words like bbox=, same_object=, in pixels, or any explanation."
    ),
    "3": (
        "SYSTEM: Use the given reference images + labels + bboxes to localize the object in the last (target) image.\n"
        "Then verify identity consistency across references and the target.\n"
        "Finalize with exactly two lines (no extra text):\n"
        "bbox=[x1,y1,x2,y2]\n"
        "same_object=YES/NO"
    ),
    "4": (
        "SYSTEM: First output a bbox for the target as [x1,y1,x2,y2]. Then answer the final identity question with only YES or NO.\n"
        "Do not include explanations."
    ),
}
