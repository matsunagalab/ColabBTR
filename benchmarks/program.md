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

## Experiment Loop (Autoresearch Mode)

Work continuously in this cycle:

### 1. Setup (first time only)
- Create branch: `git checkout -b autoresearch/<short-tag>`
- Run baseline: `PYTHONPATH=. python benchmarks/evaluate.py --quick --output-dir benchmark_results`
- Note the baseline RMSD from output
- Initialize `benchmark_results/results.tsv` with header and baseline row:
  ```
  commit  rmsd_mean  rmsd_std  status  description
  (baseline)  X.XXXX  Y.YYYY  baseline  Initial RMSD
  ```

### 2. Iterate continuously
a. Examine `benchmarks/train.py` and current `results.tsv`
b. Plan one change to `train.py` to improve RMSD
c. Apply the change and commit: `git commit -m "reason: <explanation of why this should help>"`
d. Run quick evaluation: `PYTHONPATH=. python benchmarks/evaluate.py --quick --output-dir benchmark_results`
e. Extract the RMSD from the latest JSONL file (report will show mean/std)
f. **Decision:**
   - **If RMSD improves (or unchanged):** Append row to `results.tsv` with `status=keep`
   - **If RMSD worsens:** Revert with `git checkout benchmarks/train.py` and append row with `status=discard`
g. Continue from step (a) with next idea

### 3. Full evaluation (every 5-10 iterations or when done)
- Run full benchmark: `PYTHONPATH=. python benchmarks/evaluate.py --output-dir benchmark_results`
- Record full RMSD in `results.tsv`

### 4. Stop Conditions
- User interrupts (press Ctrl+C)
- If no ideas remain, review "Things you might try" section and keep exploring
- Maintain careful reasoning in commit messages for future analysis
