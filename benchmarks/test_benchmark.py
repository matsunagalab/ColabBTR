"""Parametrized BTR benchmark tests.

Run with: pytest benchmarks/ -m benchmark
Subset:   pytest benchmarks/ -m benchmark -k "3A5I and sharp"
"""

import time
from pathlib import Path

import pytest
import torch

from colabbtr.morphology import pixel_rmsd
from benchmarks.train import reconstruct_tip

SHARP = {"probe_radius": 2.0, "probe_angle": 0.3, "tip_size": 15, "label": "sharp"}
BLUNT = {"probe_radius": 5.0, "probe_angle": 0.5, "tip_size": 15, "label": "blunt"}


@pytest.mark.benchmark
@pytest.mark.parametrize("pdb_id", ["3A5I", "1GGG", "1AON"])
@pytest.mark.parametrize("noise_type,noise_sigma",
                         [("none", 0.0), ("gaussian", 0.3), ("gaussian", 1.0), ("poisson", 0.5)],
                         ids=["none", "gauss03", "gauss10", "poisson05"])
@pytest.mark.parametrize("tip_cfg", [SHARP, BLUNT], ids=["sharp", "blunt"])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_btr_benchmark(pdb_id, noise_type, noise_sigma, tip_cfg, seed,
                       benchmark_data_dir, save_result):
    fname = f"{pdb_id}_{tip_cfg['label']}_{noise_type}_{noise_sigma}_{seed}.pt"
    data_path = benchmark_data_dir / fname

    if not data_path.exists():
        pytest.skip(f"Data file not found: {fname}")

    data = torch.load(data_path, weights_only=False)
    images = data["images"]
    surfaces = data["surfaces"]
    tip_gt = data["tip_gt"]
    tip_size = (tip_cfg["tip_size"], tip_cfg["tip_size"])

    t0 = time.time()
    tip_est, loss = reconstruct_tip(images, tip_size)
    elapsed = time.time() - t0

    cutoff = -surfaces.max().item()
    rmsd = pixel_rmsd(tip_est, tip_gt.to(tip_est.dtype), tip_gt.to(tip_est.dtype),
                      cutoff=cutoff)

    save_result({
        "pdb_id": pdb_id,
        "tip_label": tip_cfg["label"],
        "probe_radius": tip_cfg["probe_radius"],
        "probe_angle": tip_cfg["probe_angle"],
        "noise_type": noise_type,
        "noise_sigma": noise_sigma,
        "seed": seed,
        "nframe": data["config"]["nframe"],
        "rmsd": round(rmsd, 4),
        "rmsd_cutoff": round(cutoff, 4),
        "final_loss": round(loss[-1], 6),
        "elapsed_sec": round(elapsed, 1),
    })

    # Soft assertion: catch catastrophic failures only
    assert rmsd < 200.0, f"RMSD unexpectedly large: {rmsd:.2f}"
