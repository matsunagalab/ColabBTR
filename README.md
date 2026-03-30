# ColabBTR

End-to-end differentiable blind tip reconstruction (BTR) for Atomic Force Microscopy (AFM) images, implemented in PyTorch.

BTR reconstructs AFM probe tip shapes and removes tip convolution artifacts from noisy AFM images.

## Quick start on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagalab/ColabBTR/blob/main/ColabBTR.ipynb)

Open the notebook and follow the interactive steps — no local setup required.

## Local installation

Requires Python >= 3.9 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/matsunagalab/ColabBTR.git
cd ColabBTR
```

### Install the core package only

```bash
uv sync
```

This installs the `colabbtr` module with its minimal dependencies (PyTorch, tqdm).

### Run the notebook locally

```bash
uv sync --group notebook
uv run jupyter notebook ColabBTR.ipynb
```

### Run the test suite

```bash
uv sync --group dev
uv run pytest tests/ -v
```

This downloads PDB [3A5I](https://www.rcsb.org/structure/3A5I), generates synthetic AFM images with a known tip shape, runs BTR to reconstruct the tip, and checks the reconstruction quality.

### Run the visualization script

```bash
uv run python scripts/test_btr.py
```

Generates comparison plots (ground truth vs. reconstructed tip, loss curve, surface deconvolution) in the `results/` directory.

## Using colabbtr as a library

```bash
pip install git+https://github.com/matsunagalab/ColabBTR
```

```python
import torch
from colabbtr.morphology import define_tip, surfing, idilation, ierosion, differentiable_btr

# Create a probe tip shape
tip = define_tip(torch.zeros(10, 10), resolution_x=1.0, resolution_y=1.0,
                 probeRadius=5.0, probeAngle=0.3)

# Reconstruct tip from AFM images
tip_est, loss = differentiable_btr(images, tip_size=(10, 10), nepoch=200, lr=0.1)

# Deconvolve (erosion) to recover the original surface
surface = ierosion(image, tip_est)
```

## Example data

`data/` contains synthetic test cases with `.npy` arrays:

- `single_tip/` — single conical tip (576 frames, 30x30 pixels)
- `double_tip/` — bifurcated double tip
- `mixed_tip/` — mixed tip (subset of frames)

Each directory includes `images.npy` (synthetic AFM images), `surfs.npy` (ground truth surfaces), and `tip.npy` (ground truth tip shape).

## Citation

```
Y. Matsunaga, S. Fuchigami, T. Ogane, and S. Takada.
End-to-end differentiable blind tip reconstruction for noisy atomic force microscopy images.
Sci. Rep. 13, 129 (2023).
https://doi.org/10.1038/s41598-022-27057-2
```

## Contact

If you have any questions or troubles, feel free to create GitHub issues, or send email to us.

Yasuhiro Matsunaga — ymatsunaga@mail.saitama-u.ac.jp
