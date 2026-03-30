# ColabBTR

End-to-end differentiable **Blind Tip Reconstruction (BTR)** for Atomic Force Microscopy (AFM) images, implemented in PyTorch.

Given a set of AFM images, BTR reconstructs the probe tip shape and removes tip convolution artifacts (dilation) to recover the original sample surface.

## Quick start on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagalab/ColabBTR/blob/main/ColabBTR.ipynb)

Open the notebook and follow the interactive steps -- no local setup required.

## Local installation

Requires Python >= 3.9.

### With uv (recommended)

```bash
git clone https://github.com/matsunagalab/ColabBTR.git
cd ColabBTR
uv sync            # install core package (torch, tqdm, libasd)
```

### With pip

```bash
pip install git+https://github.com/matsunagalab/ColabBTR
```

## Usage

### As a library

```python
import torch
from scipy.spatial.transform import Rotation
from colabbtr.morphology import (
    load_pdb_ca, define_tip, surfing, idilation, ierosion, differentiable_btr,
)

# 1. Load protein structure (CA atoms) from PDB file
xyz, radii = load_pdb_ca("data/3A5I.pdb")  # xyz: (N, 3) in nm, radii: (N,) in nm
xyz = xyz - xyz.mean(dim=0, keepdim=True)   # center the molecule

# 2. Define a conical probe tip (15x15 pixels, 1.0 nm/pixel)
tip = define_tip(torch.zeros(15, 15), resolution_x=1.0, resolution_y=1.0,
                 probeRadius=2.0, probeAngle=0.3)

# 3. Compute molecular surfaces and simulate AFM images (20 frames)
config = {
    "min_x": -15.0, "max_x": 15.0,
    "min_y": -15.0, "max_y": 15.0,
    "resolution_x": 1.0, "resolution_y": 1.0,
}
nframe = 20
surfaces, images = [], []
for _ in range(nframe):
    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
    xyz_rot = xyz @ R.T
    surface = surfing(xyz_rot, radii, config)        # (H, W) molecular surface
    image = idilation(surface.double(), tip.double()) # (H, W) simulated AFM image
    surfaces.append(surface)
    images.append(image)
surfaces = torch.stack(surfaces)  # (nframe, H, W)
images = torch.stack(images)      # (nframe, H, W)

# 4. Reconstruct tip shape from AFM images via BTR (200 epochs)
tip_est, loss = differentiable_btr(
    images, tip_size=(15, 15), nepoch=200, lr=0.1, weight_decay=0.001,
)

# 5. Deconvolve: remove tip artifacts to recover the sample surfaces
surfaces_recovered = torch.stack([ierosion(images[i], tip_est) for i in range(nframe)])
```

### Save results as ASD files

```python
import libasd
import numpy as np

# Scanning range (nm) = pixel count × resolution (nm/pixel)
x_range = images.shape[2]  # 30 pixels × 1.0 nm/pixel = 30 nm
y_range = images.shape[1]  # 30 pixels × 1.0 nm/pixel = 30 nm

libasd.write_asd("images.asd", images.numpy(), x_scanning_range=x_range, y_scanning_range=y_range)
libasd.write_asd("surfaces.asd", surfaces.numpy(), x_scanning_range=x_range, y_scanning_range=y_range)
libasd.write_asd("surfaces_recovered.asd", surfaces_recovered.numpy(), x_scanning_range=x_range, y_scanning_range=y_range)

# Tip shape (single frame)
tip_range = tip.shape[1]  # 15 pixels × 1.0 nm/pixel = 15 nm
libasd.write_asd("tip.asd", tip.numpy(), x_scanning_range=tip_range, y_scanning_range=tip_range)
libasd.write_asd("tip_est.asd", tip_est.numpy(), x_scanning_range=tip_range, y_scanning_range=tip_range)
```

### Run the Colab notebook locally

```bash
uv sync --group notebook
uv run jupyter notebook ColabBTR.ipynb
```

## How it works

AFM images are **dilated** versions of the true sample surface -- the recorded height at each pixel is the maximum of (surface + tip) over the tip footprint. BTR reverses this by optimizing a tip shape that minimizes the reconstruction error:

1. Start with a flat (zero) tip estimate
2. For each image: erode by the tip, then dilate back, and compute the MSE with the original
3. Update the tip via gradient descent (AdamW)
4. Clamp the tip to non-positive values and re-center after each step

The key morphological operations (dilation/erosion) are differentiable and JIT-compiled for performance.

## Tests

```bash
uv sync --group dev
uv run pytest tests/ -v
```

The test suite:

- **test_btr.py** -- End-to-end roundtrip: generates synthetic AFM images from PDB [3A5I](https://www.rcsb.org/structure/3A5I) with a known tip, runs BTR, and checks reconstruction quality
- **test_morphology.py** -- Unit tests for dilation, erosion, tip definition, surfing, translate_tip_mean, Atom2Radius consistency, and PINN models

### Visualization

```bash
uv run python tests/visualize.py
```

Generates `afmhot`-colored comparison plots (tip shape, surface/image/erosion pipeline, BTR reconstruction) in `tests/`.

## Notebooks

| Notebook | Description |
|----------|-------------|
| **ColabBTR.ipynb** | Main pipeline: data upload, tilt correction, denoising, BTR, erosion, download |
| **dilation.ipynb** | PDB structure to molecular surface to simulated AFM image |
| **simulation_on_stage.ipynb** | MD simulation (OpenMM) of molecules on an AFM stage |
| **development.ipynb** | Development/testing of morphological operations |

## Example data

`data/` contains synthetic test cases (`.npy` arrays):

| Directory | Contents |
|-----------|----------|
| `single_tip/` | Single conical tip |
| `double_tip/` | Bifurcated double tip |
| `mixed_tip/` | Mixed tip (subset of frames) |

Each includes `images.npy` (AFM images), `surfs.npy` (ground truth surfaces), and `tip.npy` (ground truth tip).

## Citation

```
Y. Matsunaga, S. Fuchigami, T. Ogane, and S. Takada.
End-to-end differentiable blind tip reconstruction for noisy atomic force microscopy images.
Sci. Rep. 13, 129 (2023).
https://doi.org/10.1038/s41598-022-27057-2
```

## Contact

Questions or issues? Open a [GitHub issue](https://github.com/matsunagalab/ColabBTR/issues) or email Yasuhiro Matsunaga (ymatsunaga@mail.saitama-u.ac.jp).
