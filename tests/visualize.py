"""Visualization script for test results.

Run: python tests/visualize.py
Generates PNG files in tests/ showing tip shapes, surfaces, and BTR results.
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
    surfing,
)

CMAP = "afmhot"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDB_URL = "https://files.rcsb.org/download/3A5I.pdb"
PDB_CACHE = "/tmp/3A5I.pdb"


def get_pdb():
    if not os.path.exists(PDB_CACHE):
        urllib.request.urlretrieve(PDB_URL, PDB_CACHE)
    return PDB_CACHE


def visualize_tip():
    """Visualize ground truth tip shape."""
    tip = torch.zeros(15, 15)
    tip = define_tip(tip, resolution_x=1.0, resolution_y=1.0, probeRadius=2.0, probeAngle=0.3)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(tip.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    ax.set_title("Tip shape (15x15, R=2.0 nm, 1.0 nm/pixel)")
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    plt.colorbar(im, ax=ax, label="height (nm)")
    path = os.path.join(OUTPUT_DIR, "vis_tip.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  edge value: {tip[0,0]:.1f} nm, center: {tip[7,7]:.1f} nm")


def visualize_surface_and_image():
    """Visualize molecular surface and dilated AFM image."""
    xyz, radii = load_pdb_ca(get_pdb())
    xyz = xyz - xyz.mean(dim=0, keepdim=True)

    tip = torch.zeros(15, 15)
    tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)

    config = {
        "min_x": -15.0, "max_x": 15.0,
        "min_y": -15.0, "max_y": 15.0,
        "resolution_x": 1.0, "resolution_y": 1.0,
    }

    surface = surfing(xyz, radii, config)
    image = idilation(surface.to(torch.float64), tip.to(torch.float64))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(surface.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    axes[0].set_title("Molecular surface")
    plt.colorbar(im0, ax=axes[0], label="nm")

    im1 = axes[1].imshow(image.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    axes[1].set_title("AFM image (after dilation)")
    plt.colorbar(im1, ax=axes[1], label="nm")

    eroded = ierosion(image, tip.to(torch.float64))
    im2 = axes[2].imshow(eroded.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    axes[2].set_title("Erosion (tip deconvolution)")
    plt.colorbar(im2, ax=axes[2], label="nm")

    for ax in axes:
        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")

    path = os.path.join(OUTPUT_DIR, "vis_surface_image.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def visualize_btr():
    """Visualize BTR tip reconstruction from synthetic images."""
    xyz, radii = load_pdb_ca(get_pdb())
    xyz = xyz - xyz.mean(dim=0, keepdim=True)

    tip_gt = torch.zeros(15, 15)
    tip_gt = define_tip(tip_gt, 1.0, 1.0, 2.0, 0.3)

    config = {
        "min_x": -15.0, "max_x": 15.0,
        "min_y": -15.0, "max_y": 15.0,
        "resolution_x": 1.0, "resolution_y": 1.0,
    }

    nframe = 20
    rotations = Rotation.random(nframe, random_state=42)
    rot_matrices = torch.tensor(rotations.as_matrix(), dtype=torch.float32)

    images = []
    for i in range(nframe):
        xyz_rot = xyz @ rot_matrices[i].T
        surf = surfing(xyz_rot, radii, config)
        image = idilation(surf.to(torch.float64), tip_gt.to(torch.float64))
        images.append(image)
    images = torch.stack(images)

    print("Running BTR (50 epochs)...")
    tip_est, loss_train = differentiable_btr(
        images, tip_size=tip_gt.shape, nepoch=200, lr=0.1, weight_decay=0.001, is_tqdm=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(tip_gt.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    axes[0].set_title("Ground truth tip")
    plt.colorbar(im0, ax=axes[0], label="nm")

    im1 = axes[1].imshow(tip_est.numpy(), cmap=CMAP, aspect="equal", interpolation="none")
    axes[1].set_title("Reconstructed tip (BTR)")
    plt.colorbar(im1, ax=axes[1], label="nm")

    axes[2].plot(loss_train)
    axes[2].set_title("Training loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE")
    axes[2].set_yscale("log")

    path = os.path.join(OUTPUT_DIR, "vis_btr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    visualize_tip()
    visualize_surface_and_image()
    visualize_btr()
