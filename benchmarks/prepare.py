"""Prepare synthetic AFM benchmark data. This file is FIXED — do not modify."""

import argparse
import os
import urllib.request
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation

from colabbtr.morphology import (
    add_noise,
    define_tip,
    idilation,
    load_pdb_ca,
    surfing,
)

PDB_IDS = ["3A5I", "1GGG", "1AON"]
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


def download_pdb(pdb_id):
    """Download PDB file from RCSB, return local path."""
    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDB_CACHE / f"{pdb_id}.pdb"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return str(path)


def generate_images(xyz, radii, tip_gt, seed):
    """Generate synthetic AFM images with random molecule rotations."""
    rotations = Rotation.random(NFRAME, random_state=seed)
    rot_matrices = torch.tensor(rotations.as_matrix(), dtype=torch.float32)

    surfaces, images = [], []
    for i in range(NFRAME):
        xyz_rot = xyz @ rot_matrices[i].T
        surface = surfing(xyz_rot, radii, AFM_CONFIG)
        image = idilation(surface.double(), tip_gt.double())
        surfaces.append(surface)
        images.append(image)

    return torch.stack(surfaces), torch.stack(images)


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
                surfaces, images_clean = generate_images(xyz, radii, tip_gt, seed)

                for noise_type, noise_sigma in NOISE_CONFIGS:
                    fname = f"{pdb_id}_{tip_label}_{noise_type}_{noise_sigma}_{seed}.pt"
                    fpath = output_dir / fname

                    if fpath.exists() and not args.force:
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
                        },
                    }, fpath)

                    count += 1
                    print(f"[{count}/{total}] {fname}")

    print(f"Done. {total} datasets in {output_dir}")


if __name__ == "__main__":
    main()
