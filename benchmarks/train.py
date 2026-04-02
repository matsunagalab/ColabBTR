"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def laplacian_smoothing(tip, weight=0.01):
    """Laplacian smoothing penalty: encourage smooth, physically plausible tips."""
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    roughness = torch.mean(lap_x) + torch.mean(lap_y)
    return weight * roughness


def estimate_high_freq_energy(images):
    """Measure high-frequency energy in images as a noise proxy."""
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def estimate_variance_ratio(images):
    """Bright/dark variance ratio to distinguish noise types.

    Gaussian noise: ratio ~1.5-6.5 (constant variance)
    Poisson/none: ratio >> 100 (signal-dependent variance)
    """
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-adaptive BTR with physical preprocessing and extended compute.

    Four regimes based on noise detection:

    1. Clean data (HF < 0.2):
       Depth regularizer to break the flat-tip degeneracy.

    2. Moderate noise (HF 0.2–0.5):
       Standard baseline.

    3. High additive noise (HF > 0.5, variance_ratio < 100 → Gaussian):
       Preprocessing: clamp negative pixels to 0 (physical constraint).
       Extended compute: 200 + 100 epochs.
       Stronger smoothing.

    4. High signal-dependent noise (HF > 0.5, variance_ratio > 100 → Poisson):
       Standard baseline (clamp and extended epochs hurt Poisson).
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_additive_noise = (hf_energy > 0.5) and (var_ratio < 100)

    # Depth regularizer for clean data only
    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    # Physical preprocessing: clamp negatives for high additive noise only
    if is_high_additive_noise:
        images = torch.clamp(images, min=0.0)

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # Noise-adaptive compute budget and smoothing
    if is_high_additive_noise:
        nepoch_stage1 = 200  # more time to converge through noise
        nepoch_stage2 = 100
        smooth_base = 0.008  # stronger smoothing for noisy data
    else:
        nepoch_stage1 = 140
        nepoch_stage2 = 60
        smooth_base = 0.005

    # STAGE 1: Optimization on all frames
    for epoch in range(nepoch_stage1):
        progress = epoch / nepoch_stage1
        if progress < 0.3:
            lr_factor = 0.6 + (progress / 0.3) * 0.4
            wd_factor = 1.0
            smooth_weight = smooth_base * 0.2
        elif progress < 0.85:
            lr_factor = 1.0
            wd_factor = 1.0
            smooth_weight = smooth_base
        else:
            decay = (progress - 0.85) / 0.15
            lr_factor = 0.1 ** decay
            wd_factor = max(0.05, 1.0 - decay * 0.95)
            smooth_weight = smooth_base * 2.0

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
            depth_loss = depth_alpha * torch.mean(tip)
            loss = recon_loss + smooth_loss + depth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # Evaluate frame errors for hard-frame selection
    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            error = torch.mean((image_reconstructed - images[iframe]) ** 2)
            frame_errors.append(error.item())

    hard_count = max(1, (nframe + 1) // 2)
    hard_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_count, largest=True
    ).indices.tolist()

    # STAGE 2: Hard-frame refinement
    for epoch in range(nepoch_stage2):
        decay = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay
        smooth_weight = smooth_base * 2.0 + smooth_base * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer.zero_grad()
                image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
                recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
                smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
                depth_loss = depth_weight * torch.mean(tip)
                loss = recon_loss + smooth_loss + depth_loss
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)

                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
