"""Generate summary report from benchmark JSONL results.

Usage:
    python benchmarks/report.py                              # latest result
    python benchmarks/report.py --plot                       # with plots
    python benchmarks/report.py result1.jsonl result2.jsonl  # compare runs
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path):
    """Load JSONL file, return list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def summarize(results):
    """Group by condition and compute mean/std RMSD."""
    groups = defaultdict(list)
    for r in results:
        key = (r["pdb_id"], r["tip_label"], r["noise_type"], str(r["noise_sigma"]))
        groups[key].append(r["rmsd"])
    return groups


def print_table(groups, title=""):
    """Print formatted table of results."""
    if title:
        print(f"\n{title}")
    print(f"\n{'Condition':<45} {'RMSD (mean +/- std)':>20} {'N':>4}")
    print("-" * 72)

    all_rmsds = []
    for key in sorted(groups.keys()):
        rmsds = groups[key]
        all_rmsds.extend(rmsds)
        label = " / ".join(key)
        print(f"{label:<45} {np.mean(rmsds):>8.4f} +/- {np.std(rmsds):<8.4f} {len(rmsds):>4}")

    if all_rmsds:
        print("-" * 72)
        print(f"{'Overall':<45} {np.mean(all_rmsds):>8.4f} +/- {np.std(all_rmsds):<8.4f} {len(all_rmsds):>4}")

    return all_rmsds


def compare_runs(results_list, labels):
    """Compare multiple runs side by side."""
    summaries = [summarize(r) for r in results_list]
    all_keys = sorted(set().union(*[s.keys() for s in summaries]))

    header = f"{'Condition':<40}"
    for label in labels:
        header += f" {label:>15}"
    print(f"\n{header}")
    print("-" * (40 + 16 * len(labels)))

    for key in all_keys:
        row = f"{' / '.join(key):<40}"
        for s in summaries:
            if key in s:
                row += f" {np.mean(s[key]):>8.4f}+/-{np.std(s[key]):<5.4f}"
            else:
                row += f" {'N/A':>15}"
        print(row)


def make_plots(results, output_dir):
    """Generate summary plots."""
    import matplotlib.pyplot as plt

    groups = summarize(results)

    # Bar plot: mean RMSD by condition
    keys = sorted(groups.keys())
    labels = [" / ".join(k) for k in keys]
    means = [np.mean(groups[k]) for k in keys]
    stds = [np.std(groups[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(12, max(4, len(keys) * 0.4)))
    y_pos = range(len(keys))
    ax.barh(y_pos, means, xerr=stds, align="center", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("RMSD (nm)")
    ax.set_title("BTR Benchmark: RMSD by Condition")
    ax.invert_yaxis()
    plt.tight_layout()

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "benchmark_rmsd.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark report")
    parser.add_argument("files", nargs="*", help="JSONL result files to summarize")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    if not args.files:
        # Find latest result file
        result_dir = Path(args.output_dir)
        jsonl_files = sorted(result_dir.glob("*.jsonl"))
        if not jsonl_files:
            print("No result files found. Run 'python benchmarks/evaluate.py' first.")
            return
        args.files = [str(jsonl_files[-1])]

    if len(args.files) == 1:
        results = load_results(args.files[0])
        git_hash = results[0].get("git_commit", "unknown") if results else "unknown"
        groups = summarize(results)
        print_table(groups, title=f"Results: {args.files[0]} (commit: {git_hash})")

        if args.plot:
            make_plots(results, args.output_dir)
    else:
        results_list = [load_results(f) for f in args.files]
        labels = []
        for f, rl in zip(args.files, results_list):
            h = rl[0].get("git_commit", "unknown") if rl else "?"
            labels.append(h)
        compare_runs(results_list, labels)


if __name__ == "__main__":
    main()
