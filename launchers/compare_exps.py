#!/usr/bin/env python3
"""
Compare lm-eval metrics across two experiment directories.

Each root should contain subfolders (e.g. per seed); each subfolder may have
evaluation_results.txt with STUDENT EVALUATION RESULTS markdown table.

Usage:
  python compare_eval_dirs.py DIR_A DIR_B [--name-a A --name-b B] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- parsing -----------------------------------------------------------------

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


def find_eval_files(root: Path) -> List[Path]:
    """All evaluation_results.txt under root (one level or recursive)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    direct = sorted(root.glob("*/evaluation_results.txt"))
    if direct:
        return direct
    return sorted(root.rglob("evaluation_results.txt"))


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)


def aggregate_dir(root: Path) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """
    Per (task, metric): (mean, std, n_files).
    """
    files = find_eval_files(root)
    buckets: Dict[Tuple[str, str], List[float]] = {}

    for fp in files:
        try:
            m = load_eval_file(fp)
        except Exception as e:
            print(f"WARN: skip {fp}: {e}", file=sys.stderr)
            continue
        for k, v in m.items():
            buckets.setdefault(k, []).append(v)

    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for k, vs in buckets.items():
        mu, sd = mean_std(vs)
        out[k] = (mu, sd, len(vs))
    return out


def format_ms(mu: float, sd: float) -> str:
    if math.isnan(mu):
        return "nan"
    return f"{mu:.4f} ± {sd:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare evaluation_results.txt across two experiment dirs.")
    p.add_argument("dir_a", type=Path, help="First method root (contains subfolders with evaluation_results.txt)")
    p.add_argument("dir_b", type=Path, help="Second method root")
    p.add_argument("--name-a", default="method_A", help="Label for dir_a")
    p.add_argument("--name-b", default="method_B", help="Label for dir_b")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = p.parse_args()

    agg_a = aggregate_dir(args.dir_a)
    agg_b = aggregate_dir(args.dir_b)

    keys = sorted(set(agg_a.keys()) | set(agg_b.keys()), key=lambda x: (x[0], x[1]))

    rows = []
    print(f"Files: {args.dir_a.name}: n_files with metrics = {len(find_eval_files(args.dir_a))}")
    print(f"Files: {args.dir_b.name}: n_files with metrics = {len(find_eval_files(args.dir_b))}")
    print()

    header = ["task", "metric", args.name_a, args.name_b, "delta (A-B)"]
    colw = [12, 18, 28, 28, 14]

    def row_fmt(cols):
        return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cols, colw))

    print(row_fmt(header))
    print("-" * (sum(colw) + 2 * (len(colw) - 1)))

    for task, metric in keys:
        ma, sa, na = agg_a.get((task, metric), (float("nan"), float("nan"), 0))
        mb, sb, nb = agg_b.get((task, metric), (float("nan"), float("nan"), 0))
        if not math.isnan(ma) and not math.isnan(mb):
            delta = ma - mb
        else:
            delta = float("nan")

        rows.append(
            {
                "task": task,
                "metric": metric,
                f"{args.name_a}_mean": ma,
                f"{args.name_a}_std": sa,
                f"{args.name_a}_n": na,
                f"{args.name_b}_mean": mb,
                f"{args.name_b}_std": sb,
                f"{args.name_b}_n": nb,
                "delta_A_minus_B": delta,
            }
        )

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

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()