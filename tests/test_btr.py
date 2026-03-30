"""End-to-end roundtrip test for Blind Tip Reconstruction (BTR).

Workflow: define_tip → surfing + idilation → differentiable_btr → pixel_rmsd evaluation
Uses PDB 3A5I with random rotations to generate synthetic AFM images.

Test parameters are chosen so that:
  - Resolution is 1.0 nm/pixel (typical for high-speed AFM).
  - The AFM image is large enough for the molecule to sit well inside
    (molecule ~±5 nm, image ±15 nm → ~10 nm margin on each side).
  - The tip is narrow relative to its pixel array so that edge values
    are deeply negative (~-28), avoiding boundary artifacts.
"""

import os
import urllib.request

import numpy as np
import pytest
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

PDB_URL = "https://files.rcsb.org/download/3A5I.pdb"
PDB_CACHE = "/tmp/3A5I.pdb"


@pytest.fixture(scope="module")
def pdb_path():
    if not os.path.exists(PDB_CACHE):
        urllib.request.urlretrieve(PDB_URL, PDB_CACHE)
    return PDB_CACHE


@pytest.fixture(scope="module")
def molecular_data(pdb_path):
    xyz, radii = load_pdb_ca(pdb_path)
    # Center the molecule
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    return xyz, radii


@pytest.fixture(scope="module")
def ground_truth_tip():
    # 15x15 pixels at 1.0 nm/pixel → 7 nm physical half-width
    # probeRadius=2.0 nm: tip is narrow, edge values ≈ -28
    tip = torch.zeros(15, 15)
    tip = define_tip(tip, resolution_x=1.0, resolution_y=1.0, probeRadius=2.0, probeAngle=0.3)
    return tip


@pytest.fixture(scope="module")
def synthetic_images(molecular_data, ground_truth_tip):
    """Generate synthetic AFM images by rotating molecule and applying dilation."""
    xyz, radii = molecular_data
    tip = ground_truth_tip

    nframe = 20
    rotations = Rotation.random(nframe, random_state=42)
    rot_matrices = torch.tensor(rotations.as_matrix(), dtype=torch.float32)

    # 1.0 nm/pixel, image covers ±15 nm → 30x30 pixels
    # Molecule extent ~±5 nm → ~10 nm margin on each side
    config = {
        "min_x": -15.0, "max_x": 15.0,
        "min_y": -15.0, "max_y": 15.0,
        "resolution_x": 1.0, "resolution_y": 1.0,
    }

    images = []
    surfaces = []
    for i in range(nframe):
        xyz_rot = xyz @ rot_matrices[i].T
        surf = surfing(xyz_rot, radii, config)
        image = idilation(surf.to(torch.float64), tip.to(torch.float64))
        surfaces.append(surf)
        images.append(image)

    images = torch.stack(images)
    surfaces = torch.stack(surfaces)
    return images, surfaces


def test_btr_roundtrip(synthetic_images, ground_truth_tip):
    """Test that BTR can reconstruct a known tip shape from synthetic images."""
    images, _ = synthetic_images
    tip_gt = ground_truth_tip

    tip_est, loss_train = differentiable_btr(
        images,
        tip_size=tip_gt.shape,
        nepoch=200,
        lr=0.1,
        weight_decay=0.001,
        is_tqdm=False,
    )

    # Loss should decrease
    assert loss_train[-1] < loss_train[0], "Loss did not decrease during training"

    # RMSD between reconstructed and ground truth tip
    rmsd = pixel_rmsd(tip_est, tip_gt.to(tip_est.dtype), tip_gt.to(tip_est.dtype))
    print(f"\nTip RMSD: {rmsd:.2f}")
    print(f"Final loss: {loss_train[-1]:.2f}")
    assert rmsd < 50.0, f"Tip RMSD too large: {rmsd:.2f}"


def test_erosion_improves_surface(synthetic_images, ground_truth_tip):
    """Test that erosion with ground truth tip recovers surface better than raw images."""
    images, surfaces = synthetic_images
    tip_gt = ground_truth_tip.to(torch.float64)

    # Erode first image with ground truth tip
    eroded = ierosion(images[0], tip_gt)

    # Eroded image should be closer to the original surface
    raw_diff = (images[0] - surfaces[0].to(torch.float64)).abs().mean().item()
    eroded_diff = (eroded - surfaces[0].to(torch.float64)).abs().mean().item()
    print(f"\nRaw image diff from surface: {raw_diff:.2f}")
    print(f"Eroded image diff from surface: {eroded_diff:.2f}")
    assert eroded_diff < raw_diff, "Erosion did not improve surface reconstruction"
