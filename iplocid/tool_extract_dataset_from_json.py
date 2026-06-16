#!/usr/bin/env python3
# tool_extract_dataset_from_json.py
#
# Usage:
#   python3 tool_extract_dataset_from_json.py ./data/LASOT_8shot_T2_classwise-split_test.json /ssd1/dataset/ICL_tracking_minimized
#
# Optional:
#   python3 tool_extract_dataset_from_json.py ./data/LASOT_8shot_T2_classwise-split_test.json /ssd1/dataset/ICL_tracking_minimized --src_root /ssd1/dataset/ICL_tracking
#   python3 tool_extract_dataset_from_json.py ./data/LASOT_8shot_T2_classwise-split_test.json /ssd1/dataset/ICL_tracking_minimized --dry_run
#
# English comments only.

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SRC_ROOT = "/ssd1/dataset/ICL_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images referenced by an IPLoc-style JSON file while preserving directory structure."
    )

    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the dataset JSON file.",
    )

    parser.add_argument(
        "dst_root",
        type=str,
        help="Destination root directory. Example: /ssd1/dataset/ICL_tracking_minimized",
    )

    parser.add_argument(
        "--src_root",
        type=str,
        default=DEFAULT_SRC_ROOT,
        help=f"Source root directory. Default: {DEFAULT_SRC_ROOT}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print copy plan without copying files.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if any source image is missing.",
    )

    return parser.parse_args()


def load_json(json_path: Path) -> Any:
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_root(path_str: str) -> Path:
    """
    Normalize a root path.
    This also accepts a path like 'ssd1/dataset/...' and converts it to '/ssd1/dataset/...'
    when it looks like an absolute Linux path without the leading slash.
    """
    s = str(path_str).strip()

    if s.startswith("ssd1/"):
        s = "/" + s

    return Path(s).expanduser().resolve()


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def convert_path(
    image_path_str: str,
    src_root: Path,
    dst_root: Path,
) -> Tuple[Path, Path]:
    """
    Convert an image_path entry to a source path and destination path.

    Cases:
      1. image_path starts with src_root:
           src = image_path
           dst = dst_root / relative_path_from_src_root

      2. image_path already starts with dst_root:
           src = src_root / relative_path_from_dst_root
           dst = image_path

      3. otherwise:
           src = image_path
           dst = dst_root / image_path.name
    """
    p = Path(str(image_path_str)).expanduser()

    if not p.is_absolute():
        p = p.resolve()

    if path_is_relative_to(p, src_root):
        rel = p.relative_to(src_root)
        src_path = p
        dst_path = dst_root / rel
        return src_path, dst_path

    if path_is_relative_to(p, dst_root):
        rel = p.relative_to(dst_root)
        src_path = src_root / rel
        dst_path = p
        return src_path, dst_path

    src_path = p
    dst_path = dst_root / p.name
    return src_path, dst_path


def iter_image_paths(data: Any) -> Iterable[str]:
    """
    Yield image_path strings from an IPLoc-style JSON.

    Expected main format:
      [
        {
          "element": "...",
          "image_path": ["...", "..."],
          "bbox": [...],
          "image_id": [...],
          "role": [...]
        },
        ...
      ]
    """
    if isinstance(data, list):
        for sample in data:
            if not isinstance(sample, dict):
                continue

            paths = sample.get("image_path", None)
            if isinstance(paths, list):
                for path_str in paths:
                    if isinstance(path_str, str):
                        yield path_str
            elif isinstance(paths, str):
                yield paths

    elif isinstance(data, dict):
        paths = data.get("image_path", None)
        if isinstance(paths, list):
            for path_str in paths:
                if isinstance(path_str, str):
                    yield path_str
        elif isinstance(paths, str):
            yield paths

    else:
        raise ValueError(f"Unsupported JSON root type: {type(data)}")


def collect_copy_items(
    data: Any,
    src_root: Path,
    dst_root: Path,
) -> List[Tuple[Path, Path]]:
    """
    Collect unique copy pairs.
    """
    copy_map: Dict[Path, Path] = {}

    for image_path_str in iter_image_paths(data):
        src_path, dst_path = convert_path(
            image_path_str=image_path_str,
            src_root=src_root,
            dst_root=dst_root,
        )

        copy_map[src_path] = dst_path

    return sorted(copy_map.items(), key=lambda x: str(x[0]))


def copy_one_file(
    src_path: Path,
    dst_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """
    Copy one file and return status.
    """
    if not src_path.is_file():
        return "missing"

    if dst_path.exists() and not overwrite:
        return "skipped_exists"

    if dry_run:
        return "dry_run"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return "copied"


def main() -> None:
    args = parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    src_root = normalize_root(args.src_root)
    dst_root = normalize_root(args.dst_root)

    print(f"[INFO] JSON:     {json_path}")
    print(f"[INFO] src_root: {src_root}")
    print(f"[INFO] dst_root: {dst_root}")

    data = load_json(json_path)

    copy_items = collect_copy_items(
        data=data,
        src_root=src_root,
        dst_root=dst_root,
    )

    print(f"[INFO] Unique image files referenced in JSON: {len(copy_items)}")

    counts = {
        "copied": 0,
        "skipped_exists": 0,
        "missing": 0,
        "dry_run": 0,
    }

    missing_files: List[Path] = []

    for i, (src_path, dst_path) in enumerate(copy_items, start=1):
        status = copy_one_file(
            src_path=src_path,
            dst_path=dst_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        counts[status] += 1

        if status == "missing":
            missing_files.append(src_path)

        if i <= 10 or i % 100 == 0 or status == "missing":
            print(f"[{i}/{len(copy_items)}] {status}: {src_path} -> {dst_path}")

    print("------------------------------------------------------------")
    print("[SUMMARY]")
    print(f"copied:         {counts['copied']}")
    print(f"skipped_exists: {counts['skipped_exists']}")
    print(f"missing:        {counts['missing']}")
    print(f"dry_run:        {counts['dry_run']}")

    if missing_files:
        print("------------------------------------------------------------")
        print("[WARNING] Missing source files:")
        for p in missing_files[:50]:
            print(f"  {p}")
        if len(missing_files) > 50:
            print(f"  ... and {len(missing_files) - 50} more")

        if args.strict:
            raise FileNotFoundError(f"{len(missing_files)} source files are missing.")


if __name__ == "__main__":
    main()
