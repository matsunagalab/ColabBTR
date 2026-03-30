"""Visualization script for BTR roundtrip test.

Generates synthetic AFM images from PDB 3A5I, runs BTR, and saves comparison plots.
Usage: uv run python scripts/test_btr.py
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from colabbtr.morphology import (
    define_tip,
    differentiable_btr,
    idilation,
    ierosion,
    load_pdb_ca,
    pixel_rmsd,
    surfing,
)

# --- Config ---
PDB_URL = "https://files.rcsb.org/download/3A5I.pdb"
PDB_CACHE = "/tmp/3A5I.pdb"
OUTPUT_DIR = "results"
NFRAME = 20
NEPOCH = 200
TIP_SIZE = (10, 10)
RESOLUTION = 0.5
PROBE_RADIUS = 5.0
PROBE_ANGLE = 0.3
SEED = 42


def download_pdb():
    if not os.path.exists(PDB_CACHE):
        print(f"Downloading PDB from {PDB_URL} ...")
        urllib.request.urlretrieve(PDB_URL, PDB_CACHE)
    return PDB_CACHE


def generate_data():
    """Generate synthetic AFM images from PDB structure."""
    pdb_path = download_pdb()
    xyz, radii = load_pdb_ca(pdb_path)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)

    # Create ground truth tip
    tip_gt = define_tip(torch.zeros(*TIP_SIZE), RESOLUTION, RESOLUTION, PROBE_RADIUS, PROBE_ANGLE)

    # Random rotations
    rotations = Rotation.random(NFRAME, random_state=SEED)
    rot_matrices = torch.tensor(rotations.as_matrix(), dtype=torch.float32)

    config = {
        "min_x": -5.0, "max_x": 5.0,
        "min_y": -5.0, "max_y": 5.0,
        "resolution_x": RESOLUTION, "resolution_y": RESOLUTION,
    }

    images, surfaces = [], []
    for i in range(NFRAME):
        xyz_rot = xyz @ rot_matrices[i].T
        surf = surfing(xyz_rot, radii, config)
        image = idilation(surf.to(torch.float64), tip_gt.to(torch.float64))
        surfaces.append(surf)
        images.append(image)

    images = torch.stack(images)
    surfaces = torch.stack(surfaces)
    return images, surfaces, tip_gt


def run_btr(images, tip_gt):
    """Run differentiable BTR."""
    print(f"Running BTR: {NFRAME} frames, {NEPOCH} epochs ...")
    tip_est, loss_train = differentiable_btr(
        images,
        tip_size=tip_gt.shape,
        nepoch=NEPOCH,
        lr=0.1,
        weight_decay=0.001,
    )
    return tip_est, loss_train


def plot_results(tip_gt, tip_est, loss_train, images, surfaces):
    """Generate and save all plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tip_gt_np = tip_gt.numpy()
    tip_est_np = tip_est.detach().numpy()

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_train)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("BTR Training Loss")
    ax.set_yscale("log")
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Tip comparison: 2D heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(tip_gt_np.min(), tip_est_np.min())
    vmax = 0
    im0 = axes[0].imshow(tip_gt_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth Tip")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tip_est_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed Tip")
    plt.colorbar(im1, ax=axes[1])
    diff = tip_est_np - tip_gt_np
    im2 = axes[2].imshow(diff, cmap="RdBu_r")
    axes[2].set_title("Difference")
    plt.colorbar(im2, ax=axes[2])
    fig.savefig(os.path.join(OUTPUT_DIR, "tip_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Tip 1D slices
    mid = tip_gt_np.shape[0] // 2
    x_axis = np.arange(tip_gt_np.shape[1]) * RESOLUTION
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_axis, tip_gt_np[mid, :], "b-", label="Ground Truth")
    axes[0].plot(x_axis, tip_est_np[mid, :], "r--", label="Reconstructed")
    axes[0].set_title(f"X-slice (row={mid})")
    axes[0].set_xlabel("Position (nm)")
    axes[0].set_ylabel("Height")
    axes[0].legend()
    axes[1].plot(x_axis, tip_gt_np[:, mid], "b-", label="Ground Truth")
    axes[1].plot(x_axis, tip_est_np[:, mid], "r--", label="Reconstructed")
    axes[1].set_title(f"Y-slice (col={mid})")
    axes[1].set_xlabel("Position (nm)")
    axes[1].set_ylabel("Height")
    axes[1].legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "tip_slices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Tip 3D surface plots
    fig = plt.figure(figsize=(14, 5))
    ny, nx = tip_gt_np.shape
    X, Y = np.meshgrid(np.arange(nx) * RESOLUTION, np.arange(ny) * RESOLUTION)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, tip_gt_np, cmap="viridis", alpha=0.8)
    ax1.set_title("Ground Truth Tip")
    ax1.set_zlabel("Height")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, tip_est_np, cmap="viridis", alpha=0.8)
    ax2.set_title("Reconstructed Tip")
    ax2.set_zlabel("Height")
    fig.savefig(os.path.join(OUTPUT_DIR, "tip_3d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. AFM image / erosion / ground truth surface comparison (frame 0)
    tip_est_64 = tip_est.to(torch.float64)
    eroded = ierosion(images[0], tip_est_64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(images[0].numpy(), cmap="hot")
    axes[0].set_title("Synthetic AFM Image")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(eroded.numpy(), cmap="hot")
    axes[1].set_title("Eroded (Deconvolved)")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(surfaces[0].numpy(), cmap="hot")
    axes[2].set_title("Ground Truth Surface")
    plt.colorbar(im2, ax=axes[2])
    fig.savefig(os.path.join(OUTPUT_DIR, "surface_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    images, surfaces, tip_gt = generate_data()
    print(f"Generated {images.shape[0]} synthetic AFM images: {images.shape}")

    tip_est, loss_train = run_btr(images, tip_gt)

    rmsd = pixel_rmsd(tip_est, tip_gt.to(tip_est.dtype), tip_gt.to(tip_est.dtype))
    print(f"\nResults:")
    print(f"  Final loss: {loss_train[-1]:.2f}")
    print(f"  Tip RMSD:   {rmsd:.2f}")

    plot_results(tip_gt, tip_est, loss_train, images, surfaces)
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
