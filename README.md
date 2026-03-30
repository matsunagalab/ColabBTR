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
from colabbtr.morphology import define_tip, idilation, ierosion, differentiable_btr

# Define a conical probe tip (15x15 pixels, 1.0 nm/pixel)
tip = define_tip(torch.zeros(15, 15), resolution_x=1.0, resolution_y=1.0,
                 probeRadius=2.0, probeAngle=0.3)

# Reconstruct tip shape from AFM images (nframe x H x W tensor)
tip_est, loss = differentiable_btr(images, tip_size=(15, 15), nepoch=200, lr=0.1)

# Deconvolve: remove tip artifacts to recover the sample surface
surface = ierosion(afm_image, tip_est)
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
