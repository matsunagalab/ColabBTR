# Blind Tip Reconstruction Optimization

You are optimizing the BTR (Blind Tip Reconstruction) algorithm for AFM images.

## Goal

Minimize the average RMSD between estimated and ground truth tip shapes
across all benchmark conditions (3 PDBs [3A5I, 1GGG, 1SMP] x 2 tips x 4 noise levels x 3 seeds = 72 conditions).

## Rules

- Modify ONLY `benchmarks/train.py`
- Run `python benchmarks/evaluate.py` to measure RMSD
- Lower average RMSD is better
- If a change worsens RMSD, revert via git
- You may import anything from `colabbtr.morphology`
- You may NOT modify prepare.py, evaluate.py, or colabbtr/
- Work continuously without asking for permission
- Log your reasoning in git commit messages

## Evaluation

```bash
python benchmarks/evaluate.py --quick   # fast smoke test (~15 sec)
python benchmarks/evaluate.py           # full sweep (~15 min)
python benchmarks/report.py             # summary table
```

## Current Baseline

The baseline calls `differentiable_btr` with default parameters (nepoch=200, lr=0.1, weight_decay=0.001).

## Background

BTR reconstructs the AFM probe tip shape from convolved AFM images using
differentiable morphological operations (erosion + dilation). The key function is
`differentiable_btr()` in `colabbtr/morphology.py`. It initializes a tip as zeros,
then optimizes via AdamW to minimize the reconstruction error
`mean((idilation(ierosion(image, tip), tip) - image)^2)`.

Things you might try:
- Hyperparameter tuning (learning rate schedule, weight_decay, epochs)
- Tip initialization strategies
- Image preprocessing (denoising before BTR)
- Multi-scale or coarse-to-fine approaches
- Ensemble methods across frames
- Loss function modifications
