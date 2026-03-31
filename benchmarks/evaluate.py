"""Evaluate BTR benchmark. This file is FIXED — do not modify."""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

from colabbtr.morphology import pixel_rmsd
from benchmarks.train import reconstruct_tip


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_single(data_path):
    """Run BTR on a single prepared dataset and return result dict."""
    data = torch.load(data_path, weights_only=False)
    images = data["images"]
    surfaces = data["surfaces"]
    tip_gt = data["tip_gt"]
    cfg = data["config"]

    tip_size = (cfg["tip_size"], cfg["tip_size"])

    t0 = time.time()
    tip_est, loss = reconstruct_tip(images, tip_size)
    elapsed = time.time() - t0

    # Dynamic cutoff: tip can only be probed to molecule height depth
    cutoff = -surfaces.max().item()
    rmsd = pixel_rmsd(tip_est, tip_gt.to(tip_est.dtype), tip_gt.to(tip_est.dtype),
                      cutoff=cutoff)

    return {
        "pdb_id": cfg["pdb_id"],
        "tip_label": cfg["tip_label"],
        "probe_radius": cfg["probe_radius"],
        "probe_angle": cfg["probe_angle"],
        "noise_type": cfg["noise_type"],
        "noise_sigma": cfg["noise_sigma"],
        "seed": cfg["seed"],
        "nframe": cfg["nframe"],
        "rmsd": round(rmsd, 4),
        "rmsd_cutoff": round(cutoff, 4),
        "final_loss": round(loss[-1], 6),
        "elapsed_sec": round(elapsed, 1),
        "git_commit": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BTR benchmark")
    parser.add_argument("--data-dir", default="benchmark_results/data",
                        help="Directory with prepared .pt files")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Directory for result JSONL files")
    parser.add_argument("--pdb", help="Filter by PDB ID")
    parser.add_argument("--tip", help="Filter by tip label (sharp/blunt)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only one condition (smoke test)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run 'python benchmarks/prepare.py' first.")
        return

    pt_files = sorted(data_dir.glob("*.pt"))
    if args.pdb:
        pt_files = [f for f in pt_files if f.name.startswith(args.pdb)]
    if args.tip:
        pt_files = [f for f in pt_files if f"_{args.tip}_" in f.name]
    if args.quick:
        pt_files = pt_files[:1]

    if not pt_files:
        print("No matching data files found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    git_hash = get_git_hash()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"{git_hash}_{ts}.jsonl"

    print(f"Running {len(pt_files)} conditions → {result_path}")
    results = []

    for i, pt_file in enumerate(pt_files):
        print(f"[{i+1}/{len(pt_files)}] {pt_file.name} ...", end=" ", flush=True)
        result = run_single(pt_file)
        results.append(result)
        print(f"RMSD={result['rmsd']:.4f}  loss={result['final_loss']:.6f}  "
              f"({result['elapsed_sec']:.1f}s)")

        with open(result_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"Results saved to {result_path}")
    print(f"{'='*70}")

    # Group by (pdb_id, tip_label, noise_type, noise_sigma) and show mean/std
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["pdb_id"], r["tip_label"], r["noise_type"], str(r["noise_sigma"]))
        groups[key].append(r["rmsd"])

    print(f"\n{'Condition':<45} {'RMSD (mean +/- std)':>20} {'N':>4}")
    print("-" * 72)
    all_rmsds = []
    for key in sorted(groups.keys()):
        rmsds = groups[key]
        all_rmsds.extend(rmsds)
        import numpy as np
        mean = np.mean(rmsds)
        std = np.std(rmsds)
        label = " / ".join(key)
        print(f"{label:<45} {mean:>8.4f} +/- {std:<8.4f} {len(rmsds):>4}")

    if all_rmsds:
        import numpy as np
        print("-" * 72)
        print(f"{'Overall':<45} {np.mean(all_rmsds):>8.4f} +/- {np.std(all_rmsds):<8.4f} {len(all_rmsds):>4}")


if __name__ == "__main__":
    main()
