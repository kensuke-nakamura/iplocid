# loc_dataset.py
#
# Dataset / dataloader for IPLoc-style localization.
#
# Supported JSON formats:
#   (A) Custom JSON with explicit 'role'
#   (B) Legacy JSON without 'role'
#       -> interpreted as (n-1) references + last positive-image
#
# NOTE for (B):
#   Legacy JSON often uses relative paths like:
#       data/ICL_tracking/...
#   Make sure the following symlink exists before running inference:
#
#       mkdir -p data
#       ln -s /ssd1/dataset/ICL_tracking data/ICL_tracking
#
# This is a temporary workaround for research purposes.

import json
from torch.utils.data import Dataset, DataLoader


def _ensure_role_field(sample: dict) -> dict:
    """
    If 'role' is missing, define roles assuming:
      - first (n-1) images are 'reference'
      - last image is 'positive-image'
    Also ensures image_id length consistency (auto-fills if missing/broken).
    """
    if not isinstance(sample, dict):
        return sample

    # If role already exists, keep it as-is
    if "role" in sample and sample["role"] is not None:
        return sample

    # Cannot infer without image_path
    if "image_path" not in sample or sample["image_path"] is None:
        return sample

    n = len(sample["image_path"])
    if n <= 0:
        return sample

    # Define role: (n-1) references + last positive-image
    sample["role"] = ["reference"] * max(0, n - 1) + ["positive-image"]

    # Ensure image_id exists and length matches
    if "image_id" not in sample or sample["image_id"] is None:
        sample["image_id"] = list(range(n))
    else:
        try:
            if len(sample["image_id"]) != n:
                sample["image_id"] = list(range(n))
        except Exception:
            sample["image_id"] = list(range(n))

    return sample


class LocDataset(Dataset):
    def __init__(self, args):
        with open(args.data_path, "r", encoding="utf-8") as f:
            data_all = json.load(f)

        if not isinstance(data_all, list):
            raise ValueError(f"[ERROR] JSON must be a list. Got: {type(data_all)}")

        # Chunking (keep same behavior as your current code)
        if getattr(args, "chunks", 1) > 1:
            # Keep your original slicing semantics
            data_all = data_all[args.curr_chunk:-1:args.chunks]

        # Auto-fill role for legacy json
        fixed = []
        for s in data_all:
            if isinstance(s, dict):
                fixed.append(_ensure_role_field(s))
            else:
                fixed.append(s)
        self.data_all = fixed

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        data = self.data_all[idx]

        element = data["element"]
        bbox = data["bbox"]
        image_path = data["image_path"]
        image_id = data.get("image_id", list(range(len(image_path))))

        # IMPORTANT: return 'data' as the last item so inference can read role
        return element, bbox, image_path, image_id, data


def get_dataloader(args, shuffle=False):
    loc_dataset = LocDataset(args)
    loc_dataloader = DataLoader(loc_dataset, batch_size=args.bs, shuffle=shuffle)
    return loc_dataloader
