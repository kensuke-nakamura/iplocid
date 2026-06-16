#!/usr/bin/env python3
# tool_concatenate_ranking_results.py
#
# Concatenate:
#   ./results/<data_name>/ranking_by_full_IoU.txt
# into:
#   ./summary_ranking.txt
#
# New simplified rules:
# - For the first ranking file:
#   - drop line 1 and 2
#   - keep line 3 as-is (the "# Columns: ..." line)
#   - keep all lines from line 4 onward as-is
# - For all other ranking files:
#   - drop line 1, 2, 3
#   - keep all lines from line 4 onward as-is
# - Concatenate the kept lines in file order.
#
# English comments only.

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def natural_key(s: str):
    """Natural sort key helper (e.g., 'T2' < 'T10')."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def find_ranking_files(results_root: Path) -> List[Tuple[str, Path]]:
    """Return list of (data_name, ranking_file_path). data_name is used only for sorting/reporting."""
    out: List[Tuple[str, Path]] = []
    if not results_root.is_dir():
        return out
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        data_name = p.name
        f = p / "ranking_by_full_IoU.txt"
        if f.is_file():
            out.append((data_name, f))
    out.sort(key=lambda t: natural_key(t[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="./results", help="Root directory containing <data_name>/ folders.")
    ap.add_argument("--output", type=str, default="./results/summary_ranking.txt", help="Output path for concatenated ranking.")
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    out_path = Path(args.output).resolve()

    files = find_ranking_files(results_root)
    if not files:
        print(f"[WARN] No ranking_by_full_IoU.txt found under: {results_root}")
        out_path.write_text("# (No ranking_by_full_IoU.txt files found.)\n", encoding="utf-8")
        print(f"[INFO] Wrote: {out_path}")
        return

    out_lines: List[str] = []

    for i, (data_name, fpath) in enumerate(files):
        lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()

        if len(lines) < 3:
            print(f"[WARN] Skip too-short ranking file: {fpath}")
            continue

        if i == 0:
            # Keep line 3 and beyond
            kept = lines[2:]
        else:
            # Keep line 4 and beyond
            kept = lines[3:]

        # Keep as-is (no modifications)
        out_lines.extend([ln for ln in kept if ln.strip() != ""])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Found {len(files)} ranking files.")
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
