# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ColabBTR implements end-to-end differentiable **Blind Tip Reconstruction (BTR)** for Atomic Force Microscopy (AFM) images using PyTorch. It reconstructs AFM probe tip shapes and removes tip convolution artifacts from noisy AFM images.

Based on: Matsunaga et al., "End-to-end differentiable blind tip reconstruction for noisy atomic force microscopy images," Scientific Reports 13, 129 (2023).

## Installation

```bash
pip install git+https://github.com/matsunagalab/ColabBTR
# Dependencies: torch, tqdm (Python >= 3.9)
```

No test suite, CI, linting, or Makefile exists. Development is notebook-driven.

## Architecture

The entire computational core lives in `colabbtr/morphology.py`. Key layers:

### Morphological Operations (JIT-compiled)
- `idilation()` / `ierosion()` — dilation and erosion of surfaces/images by tip shape, decorated with `@torch.jit.script` for performance
- `fixed_padding()` — padding helper for dilation

### BTR Optimization
- `differentiable_btr()` — main BTR algorithm using AdamW optimizer with cross-validation support
- `compute_xc_yc()` / `translate_tip_mean()` — tip centering utilities

### PINN Models (Physics-Informed Neural Networks)
- `TipShapeMLP` — learns tip shape as f(x, y, t)
- `SurfaceMLP` — learns surface as f(x, y, t)
- `BTRLoss` / `SurfaceLoss` — composite loss functions combining reconstruction error, boundary/height constraints, centroid constraints, temporal smoothness, and regularization

### AFM Simulation
- `define_tip()` — creates conical probe geometry
- `surfing()` — simulates tip-surface interaction; `shift_z=True` (default) places the molecule on the stage by shifting z so min(z)=0. Use `shift_z=False` for MD simulation data where molecules are already on the AFM stage.
- `afmize()` — generates synthetic AFM images from molecular coordinates (PDB-style)

## Notebooks

- **ColabBTR.ipynb** — Main user-facing pipeline: data upload → tilt correction → denoising → cross-validation for lambda → BTR → erosion → download results
- **development.ipynb** — Development/testing of morphological operations
- **dilation.ipynb** — Educational: PDB molecule → surface calculation → AFM simulation via dilation
- **simulation_on_stage.ipynb** — MD simulation (OpenMM) of molecules on AFM stage

## Data

`data/` contains synthetic test cases (`single_tip/`, `double_tip/`, `mixed_tip/`) with `.npy` arrays for images, surfaces, and ground-truth tips.
