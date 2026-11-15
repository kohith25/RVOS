#!/usr/bin/env python3
"""
scripts/compare_runs.py - Side-by-Side Evaluation Comparator

Compares two pipeline evaluation outputs (e.g., Original TGS vs. RL GRPO).
Outputs:
  - Formatted Markdown / Terminal comparison table
  - Summary metrics JSON (comparison_summary.json)
  - Detailed per-video CSV with diffs & win/loss analysis (comparison_per_video.csv)
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two TGS-Agent pipeline evaluation runs")
    parser.add_argument("--run1", type=str, required=True, help="Directory of run 1 (typically use_rl=false)")
    parser.add_argument("--run2", type=str, required=True, help="Directory of run 2 (typically use_rl=true)")
    parser.add_argument("--label1", type=str, default="Original TGS", help="Label for run 1")
    parser.add_argument("--label2", type=str, default="RL GRPO", help="Label for run 2")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory to save summary files (defaults to run2)")
    return parser.parse_args()


def load_results(dir_path):
    results_path = os.path.join(dir_path, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in directory: {dir_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # FIX 6(b): Reject empty or placeholder runs
    per_video = data.get("per_video", {})
    metrics = data.get("metrics", {})
    if not per_video:
        raise ValueError(f"results.json in {dir_path} has no 'per_video' entries (empty or placeholder run rejected).")

    all_zero_or_none = all(v in (0, 0.0, None) for v in metrics.values())
    if all_zero_or_none and len(per_video) == 0:
        raise ValueError(f"results.json in {dir_path} contains all-zero metrics and empty per_video (placeholder rejected).")

    return data


def format_delta(val1, val2):
    diff = val2 - val1
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.4f}"


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir else args.run2
    os.makedirs(out_dir, exist_ok=True)

    data1 = load_results(args.run1)
    data2 = load_results(args.run2)

    m1 = data1.get("metrics", {})
    m2 = data2.get("metrics", {})

    j1 = float(m1.get("J", 0.0))
    j2 = float(m2.get("J", 0.0))
    f1 = float(m1.get("F", 0.0))
    f2 = float(m2.get("F", 0.0))
    jf1 = float(m1.get("J_and_F", 0.0))
    jf2 = float(m2.get("J_and_F", 0.0))
    s1 = float(m1.get("S_null", 0.0))
    s2 = float(m2.get("S_null", 0.0))

    # Print markdown summary table
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPARISON: {args.label1} vs. {args.label2}")
    print("=" * 70)
    print(f"Benchmark: {data1.get('benchmark', 'N/A')} | Split: {data1.get('test_name', 'N/A')}\n")

    table_header = f"| Metric | {args.label1:^15} | {args.label2:^15} | {'Delta':^12} |"
    table_divider = f"|:---|:---:|:---:|:---:|"
    print(table_header)
    print(table_divider)
    print(f"| $\\mathcal{{J}}$ (IoU) | {j1:^15.4f} | {j2:^15.4f} | {format_delta(j1, j2):^12} |")
    print(f"| $\\mathcal{{F}}$ (F-score) | {f1:^15.4f} | {f2:^15.4f} | {format_delta(f1, f2):^12} |")
    print(f"| $\\mathcal{{J}}\\&\\mathcal{{F}}$ | {jf1:^15.4f} | {jf2:^15.4f} | {format_delta(jf1, jf2):^12} |")
    if data1.get("test_name") == "test_n" or data2.get("test_name") == "test_n":
        print(f"| $\\mathcal{{S}}$ (Null) | {s1:^15.4f} | {s2:^15.4f} | {format_delta(s1, s2):^12} |")

    # RL Stats if present
    rl_stats = data2.get("rl_stats")
    if rl_stats:
        print("\n### RL Policy Operational Statistics")
        print(f"- Avg Turns: {rl_stats.get('avg_turns', 'N/A')}")
        print(f"- Avg Tool Calls: {rl_stats.get('avg_tool_calls', 'N/A')}")
        print(f"- Avg Tokens per Episode: {rl_stats.get('avg_tokens', 'N/A')}")
        print(f"- Avg Latency per Video: {rl_stats.get('avg_latency_sec', 'N/A')} s")

    # Per-video comparisons
    pv1 = data1.get("per_video", {})
    pv2 = data2.get("per_video", {})

    all_uids = sorted(list(set(pv1.keys()) | set(pv2.keys())))
    csv_rows = []
    wins_r2 = 0
    wins_r1 = 0
    ties = 0

    for uid in all_uids:
        item1 = pv1.get(uid, {})
        item2 = pv2.get(uid, {})

        vid = item1.get("vid") or item2.get("vid", "N/A")
        vid_j1 = float(item1.get("s_mean_iou", item1.get("J", 0.0)))
        vid_j2 = float(item2.get("s_mean_iou", item2.get("J", 0.0)))
        vid_f1 = float(item1.get("s_fscore", item1.get("F", 0.0)))
        vid_f2 = float(item2.get("s_fscore", item2.get("F", 0.0)))

        diff_j = vid_j2 - vid_j1
        diff_f = vid_f2 - vid_f1

        if diff_j > 1e-4 or diff_f > 1e-4:
            outcome = f"{args.label2} Win"
            wins_r2 += 1
        elif diff_j < -1e-4 or diff_f < -1e-4:
            outcome = f"{args.label1} Win"
            wins_r1 += 1
        else:
            outcome = "Tie"
            ties += 1

        csv_rows.append({
            "uid": uid,
            "vid": vid,
            f"{args.label1}_J": round(vid_j1, 4),
            f"{args.label2}_J": round(vid_j2, 4),
            "delta_J": round(diff_j, 4),
            f"{args.label1}_F": round(vid_f1, 4),
            f"{args.label2}_F": round(vid_f2, 4),
            "delta_F": round(diff_f, 4),
            "outcome": outcome
        })

    # Save CSV
    csv_out_path = os.path.join(out_dir, "comparison_per_video.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_out_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[Saved per-video analysis to: {csv_out_path}]")
        print(f"Outcome Summary: {args.label2} Wins: {wins_r2} | {args.label1} Wins: {wins_r1} | Ties: {ties}")

    # Save summary JSON
    summary_data = {
        "label1": args.label1,
        "label2": args.label2,
        "metrics": {
            "J": {"run1": j1, "run2": j2, "delta": j2 - j1},
            "F": {"run1": f1, "run2": f2, "delta": f2 - f1},
            "J_and_F": {"run1": jf1, "run2": jf2, "delta": jf2 - jf1},
        },
        "counts": {
            "total_videos": len(all_uids),
            "run2_wins": wins_r2,
            "run1_wins": wins_r1,
            "ties": ties
        }
    }
    summary_out_path = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_out_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)
    print(f"[Saved comparison summary to: {summary_out_path}]\n" + "=" * 70)


if __name__ == "__main__":
    main()
