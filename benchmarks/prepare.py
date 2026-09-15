"""Prepare synthetic AFM benchmark data. This file is FIXED — do not modify.

Rendering change (2026-08-13): images are now generated with
afmize_supersampled() instead of surfing() + idilation() on the coarse grid.
The previous point-sampled rendering dropped ~34% of the molecular pixels and
truncated the peaks, and it made every image an exact coarse dilation of
tip_gt — precisely the forward model BTR inverts, so the benchmark could not
see any model mismatch. RMSD numbers recorded before this change are NOT
comparable with numbers recorded after it; regenerate with --force.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# "python benchmarks/prepare.py" puts benchmarks/ on sys.path, not the repo root,
# so an unrelated colabbtr installed in site-packages would shadow this checkout.
# Redundant under "uv run" (the editable install already puts the repo root on
# the path) but needed when the script is run with a bare interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.spatial.transform import Rotation

from colabbtr.morphology import (
    add_noise,
    afmize_supersampled,
    define_tip,
    load_pdb_ca,
)

PDB_IDS = ["3A5I", "1GGG", "1SMP"]
PDB_CACHE = Path("/tmp/pdb_cache")

TIP_CONFIGS = {
    "sharp": {"probe_radius": 2.0, "probe_angle": 0.3, "tip_size": 15},
    "blunt": {"probe_radius": 5.0, "probe_angle": 0.5, "tip_size": 15},
}

NOISE_CONFIGS = [
    ("none", 0.0),
    ("gaussian", 0.3),
    ("gaussian", 1.0),
    ("poisson", 0.5),
]

SEEDS = [42, 123, 456]
NFRAME = 20

AFM_CONFIG = {
    "min_x": -20.0, "max_x": 20.0,
    "min_y": -20.0, "max_y": 20.0,
    "resolution_x": 1.0, "resolution_y": 1.0,
}

# Every CA bead in Atom2Radius (0.225-0.340 nm) is smaller than half the 1 nm
# pitch, so sampling the surface at pixel centers alone drops ~34% of the
# molecular pixels and truncates the peaks. Render on a 5x finer grid and sample
# last: against a factor=15 reference that cuts the RMS over molecule-covered
# pixels from 0.63-0.96 nm to 0.06-0.12 nm, well under the 0.3 and 1.0 nm noise
# levels benchmarked here, at ~0.75 s/frame. Raising it costs roughly linearly.
# Changing it invalidates comparisons with results generated at another value.
SUPERSAMPLE = 5


def download_pdb(pdb_id):
    """Download PDB file from RCSB, return local path."""
    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDB_CACHE / f"{pdb_id}.pdb"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def generate_images(xyz, radii, tip_cfg, seed):
    """Generate synthetic AFM images with random molecule rotations."""
    rotations = Rotation.random(NFRAME, random_state=seed)
    rot_matrices = torch.tensor(rotations.as_matrix(), dtype=torch.float32)

    surfaces, images = [], []
    for i in range(NFRAME):
        xyz_rot = xyz @ rot_matrices[i].T
        image, surface = afmize_supersampled(
            xyz_rot, radii, AFM_CONFIG,
            tip_cfg["probe_radius"], tip_cfg["probe_angle"], tip_cfg["tip_size"],
            factor=SUPERSAMPLE,
        )
        surfaces.append(surface)
        images.append(image)

    return torch.stack(surfaces), torch.stack(images)


def _is_current(fpath):
    """True if fpath was written by the renderer this script currently uses."""
    try:
        cfg = torch.load(fpath, weights_only=False)["config"]
    except Exception:
        return False
    return cfg.get("supersample") == SUPERSAMPLE


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data")
    parser.add_argument("--force", action="store_true", help="Regenerate all data")
    parser.add_argument("--output-dir", default="benchmark_results/data",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(PDB_IDS) * len(TIP_CONFIGS) * len(NOISE_CONFIGS) * len(SEEDS)
    count = 0

    for pdb_id in PDB_IDS:
        pdb_path = download_pdb(pdb_id)
        xyz, radii = load_pdb_ca(pdb_path)
        xyz = xyz - xyz.mean(dim=0, keepdim=True)

        for tip_label, tip_cfg in TIP_CONFIGS.items():
            tip_gt = define_tip(
                torch.zeros(tip_cfg["tip_size"], tip_cfg["tip_size"]),
                1.0, 1.0, tip_cfg["probe_radius"], tip_cfg["probe_angle"],
            )

            for seed in SEEDS:
                surfaces, images_clean = generate_images(xyz, radii, tip_cfg, seed)

                for noise_type, noise_sigma in NOISE_CONFIGS:
                    fname = f"{pdb_id}_{tip_label}_{noise_type}_{noise_sigma}_{seed}.pt"
                    fpath = output_dir / fname

                    # Regenerate silently stale files: a directory left over from
                    # the point-sampled renderer would otherwise be reused, or
                    # worse, mixed with freshly written ones.
                    if fpath.exists() and not args.force and _is_current(fpath):
                        count += 1
                        continue

                    if noise_type == "none":
                        images = images_clean.clone()
                    else:
                        images = add_noise(images_clean, noise_type, noise_sigma,
                                           seed=seed)

                    torch.save({
                        "images": images,
                        "surfaces": surfaces,
                        "tip_gt": tip_gt,
                        "config": {
                            "pdb_id": pdb_id,
                            "tip_label": tip_label,
                            "probe_radius": tip_cfg["probe_radius"],
                            "probe_angle": tip_cfg["probe_angle"],
                            "tip_size": tip_cfg["tip_size"],
                            "noise_type": noise_type,
                            "noise_sigma": noise_sigma,
                            "seed": seed,
                            "nframe": NFRAME,
                            "supersample": SUPERSAMPLE,
                        },
                    }, fpath)

                    count += 1
                    print(f"[{count}/{total}] {fname}")

    print(f"Done. {total} datasets in {output_dir}")


if __name__ == "__main__":
    main()
