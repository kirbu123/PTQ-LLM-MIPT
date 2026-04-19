#!/usr/bin/env python3
"""
Compare lm-eval metrics across two experiment directories, split by k_next parameter.

Each root should contain subfolders (e.g. per seed) with names containing "k_next=X";
each subfolder may have evaluation_results.txt with STUDENT EVALUATION RESULTS markdown table.

Usage:
  python compare_exps.py DIR_A DIR_B [--name-a A --name-b B] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- parsing -----------------------------------------------------------------

def extract_k_next(folder_name: str) -> Optional[int]:
    """Extract k_next value from folder name."""
    match = re.search(r'k_next=(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def _parse_student_table(text: str) -> Dict[Tuple[str, str], float]:
    """
    Parse STUDENT EVALUATION RESULTS markdown table.
    Returns mapping (task, metric) -> value (float).
    """
    marker = "STUDENT EVALUATION RESULTS:"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("STUDENT EVALUATION RESULTS section not found")

    block = text[idx:]
    lines = block.splitlines()
    # Find header row starting with |  Tasks
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "Tasks" in line and "Metric" in line:
            start = i + 2  # skip header + separator
            break
    if start is None:
        raise ValueError("Could not locate student results table header")

    out: Dict[Tuple[str, str], float] = {}
    last_task = ""

    for line in lines[start:]:
        s = line.strip()
        if not s.startswith("|"):
            break
        if re.match(r"^\|[\s\-:|]+\|$", s):  # separator-ish
            continue

        parts = [p.strip() for p in s.split("|")]
        # parts[0] and parts[-1] are often empty
        inner = [p for p in parts[1:-1] if p != ""] if len(parts) > 2 else parts
        # Expected columns: Tasks, Version, Filter, n-shot, Metric, arrow, Value, ±, Stderr
        # More robust: split all cells
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 8:
            continue

        task_cell = cells[0]
        if task_cell:
            last_task = task_cell
        metric = cells[4].strip()
        value_str = cells[6].strip()

        if not last_task or not metric:
            continue
        try:
            val = float(value_str)
        except ValueError:
            continue

        out[(last_task, metric)] = val

    if not out:
        raise ValueError("No student metric rows parsed")
    return out


def load_eval_file(path: Path) -> Dict[Tuple[str, str], float]:
    return _parse_student_table(path.read_text(encoding="utf-8", errors="replace"))


def find_eval_files_by_knext(root: Path) -> Dict[int, List[Path]]:
    """
    Find all evaluation_results.txt under root, grouped by k_next value.
    Returns dict: k_next -> list of file paths
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    
    # Find all evaluation_results.txt files
    all_files = sorted(root.rglob("evaluation_results.txt"))
    
    # Group by k_next
    grouped: Dict[int, List[Path]] = defaultdict(list)
    ungrouped: List[Path] = []
    
    for fp in all_files:
        # Get the parent folder name (the one containing evaluation_results.txt)
        parent_folder = fp.parent.name
        k_next = extract_k_next(parent_folder)
        
        if k_next is not None:
            grouped[k_next].append(fp)
        else:
            ungrouped.append(fp)

    # Sort the lists for consistency
    for k in grouped:
        grouped[k].sort()
    
    if ungrouped:
        print(f"WARNING: Found {len(ungrouped)} files without k_next in folder name", file=sys.stderr)
        for fp in ungrouped:
            print(f"  - {fp.parent.name}", file=sys.stderr)
    
    return grouped


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)


def aggregate_dir_by_knext(root: Path) -> Dict[int, Dict[Tuple[str, str], Tuple[float, float, int]]]:
    """
    For each k_next value, return per (task, metric): (mean, std, n_files).
    """
    grouped_files = find_eval_files_by_knext(root)
    
    result: Dict[int, Dict[Tuple[str, str], Tuple[float, float, int]]] = {}
    
    for k_next, files in grouped_files.items():
        buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        for fp in files:
            try:
                m = load_eval_file(fp)
            except Exception as e:
                print(f"WARN: skip {fp} (k_next={k_next}): {e}", file=sys.stderr)
                continue
            for k, v in m.items():
                buckets[k].append(v)
        
        per_knext: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
        for k, vs in buckets.items():
            mu, sd = mean_std(vs)
            per_knext[k] = (mu, sd, len(vs))
        
        result[k_next] = per_knext
    
    return result


def format_ms(mu: float, sd: float) -> str:
    if math.isnan(mu):
        return "nan"
    return f"{mu:.4f} ± {sd:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare evaluation_results.txt across two experiment dirs, split by k_next.")
    p.add_argument("dir_a", type=Path, help="First method root (contains subfolders with evaluation_results.txt)")
    p.add_argument("dir_b", type=Path, help="Second method root")
    p.add_argument("--name-a", default="method_A", help="Label for dir_a")
    p.add_argument("--name-b", default="method_B", help="Label for dir_b")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = p.parse_args()

    # Aggregate by k_next
    agg_a_by_knext = aggregate_dir_by_knext(args.dir_a)
    agg_b_by_knext = aggregate_dir_by_knext(args.dir_b)
    
    # Get all unique k_next values from both directories
    all_knext = sorted(set(agg_a_by_knext.keys()) | set(agg_b_by_knext.keys()))
    
    if not all_knext:
        print("No k_next values found in either directory!", file=sys.stderr)
        sys.exit(1)
    
    # Get all unique metric keys across all k_next values
    all_metric_keys = set()
    for k in all_knext:
        if k in agg_a_by_knext:
            all_metric_keys.update(agg_a_by_knext[k].keys())
        if k in agg_b_by_knext:
            all_metric_keys.update(agg_b_by_knext[k].keys())
    
    all_metric_keys = sorted(all_metric_keys, key=lambda x: (x[0], x[1]))
    
    # Prepare for CSV output
    all_rows = []
    
    # Print header
    print(f"\n{'='*120}")
    print(f"Comparing: {args.name_a} vs {args.name_b}")
    print(f"Directory A: {args.dir_a}")
    print(f"Directory B: {args.dir_b}")
    print(f"{'='*120}\n")
    
    # Print summary of files found
    print("Files found per k_next:")
    for k in all_knext:
        files_a = len(find_eval_files_by_knext(args.dir_a).get(k, []))
        files_b = len(find_eval_files_by_knext(args.dir_b).get(k, []))
        print(f"  k_next={k}: {args.name_a}: {files_a} files, {args.name_b}: {files_b} files")
    print()
    
    # For each k_next, print a table
    for k_next in all_knext:
        print(f"\n{'='*120}")
        print(f"k_next = {k_next}")
        print(f"{'='*120}")
        
        agg_a = agg_a_by_knext.get(k_next, {})
        agg_b = agg_b_by_knext.get(k_next, {})
        
        # Prepare columns
        colw = [12, 18, 28, 28, 14]
        header = ["task", "metric", args.name_a, args.name_b, "delta (A-B)"]
        
        def row_fmt(cols):
            return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cols, colw))
        
        print(row_fmt(header))
        print("-" * (sum(colw) + 2 * (len(colw) - 1)))
        
        for task, metric in all_metric_keys:
            ma, sa, na = agg_a.get((task, metric), (float("nan"), float("nan"), 0))
            mb, sb, nb = agg_b.get((task, metric), (float("nan"), float("nan"), 0))
            
            if not math.isnan(ma) and not math.isnan(mb):
                delta = ma - mb
            else:
                delta = float("nan")
            
            # Store for CSV
            all_rows.append({
                "k_next": k_next,
                "task": task,
                "metric": metric,
                f"{args.name_a}_mean": ma,
                f"{args.name_a}_std": sa,
                f"{args.name_a}_n": na,
                f"{args.name_b}_mean": mb,
                f"{args.name_b}_std": sb,
                f"{args.name_b}_n": nb,
                "delta_A_minus_B": delta,
            })
            
            print(
                row_fmt(
                    [
                        task,
                        metric,
                        format_ms(ma, sa) + (f" (n={na})" if na else ""),
                        format_ms(mb, sb) + (f" (n={nb})" if nb else ""),
                        f"{delta:+.4f}" if not math.isnan(delta) else "nan",
                    ]
                )
            )
    
    # Write CSV if requested
    if args.csv and all_rows:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["k_next", "task", "metric", 
                         f"{args.name_a}_mean", f"{args.name_a}_std", f"{args.name_a}_n",
                         f"{args.name_b}_mean", f"{args.name_b}_std", f"{args.name_b}_n",
                         "delta_A_minus_B"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {args.csv}")
    
    # Print summary statistics
    print(f"\n{'='*120}")
    print("SUMMARY: Number of k_next values compared")
    print(f"{'='*120}")
    for k_next in all_knext:
        metrics_a = len(agg_a_by_knext.get(k_next, {}))
        metrics_b = len(agg_b_by_knext.get(k_next, {}))
        print(f"k_next={k_next}: {args.name_a} has {metrics_a} metrics, {args.name_b} has {metrics_b} metrics")


if __name__ == "__main__":
    main()
