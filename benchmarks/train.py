"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
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
    """Measure high-frequency energy in images as a noise proxy.

    For clean images (no noise): ~0.13 (structural edges only)
    For Gaussian σ=0.3: ~0.36
    For Poisson: ~0.55
    For Gaussian σ=1.0: ~1.13
    """
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def estimate_variance_ratio(images):
    """Estimate bright/dark variance ratio to distinguish noise types.

    Gaussian noise: ratio ~1.5-6.5 (constant variance across intensities)
    Poisson noise: ratio >> 100 (variance scales with signal intensity)
    No noise: ratio >> 100 (dark regions have zero variance)
    """
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright_mask = pixel_mean > median_val
    dark_mask = ~bright_mask
    var_bright = pixel_var[bright_mask].mean().item()
    var_dark = pixel_var[dark_mask].mean().item()
    return var_bright / (var_dark + 1e-8)


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with noise-adaptive depth regularizer and gradient smoothing.

    Two noise-adaptive architectural components:

    1. Depth regularizer (for clean data):
       The morphological opening's anti-extensive property creates a
       bluntness bias. For CLEAN images, the flat tip is nearly degenerate.
       Depth penalty breaks this. For noisy data, noise already breaks it.

    2. Spatial gradient smoothing (for high additive noise):
       For high Gaussian noise, the per-pixel gradient is noisy. Spatial
       averaging of the gradient (3×3) during early epochs provides a
       cleaner descent direction. Disabled for Poisson noise (which has
       different spatial structure) and clean data.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Detect noise characteristics
    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_additive_noise = (var_ratio < 100)  # Gaussian: ~1.5-6.5, Poisson/none: >> 100

    # Depth penalty: only for clean data (flat-tip degeneracy)
    if hf_energy < 0.2:
        depth_alpha = 0.005
    else:
        depth_alpha = 0.0

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch_stage1 = 140
    nepoch_stage2 = 60
    loss_train = []

    # Gradient smoothing: only for high ADDITIVE noise (Gaussian)
    # Poisson noise has signal-dependent spatial structure that
    # should not be smoothed. Detected via variance ratio.
    use_grad_smooth = (hf_energy > 0.5) and is_additive_noise

    # STAGE 1: Coarse optimization on all frames
    for epoch in range(nepoch_stage1):
        if epoch < 40:
            lr_factor = 0.6 + (epoch / 40) * 0.4
            wd_factor = 1.0
            smooth_weight = 0.001
        elif epoch < 120:
            lr_factor = 1.0
            wd_factor = 1.0
            smooth_weight = 0.005
        else:
            decay_progress = (epoch - 120) / 20
            lr_factor = 0.1 ** decay_progress
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)
            smooth_weight = 0.01

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
            # Depth regularizer: alpha * mean(tip). Since tip <= 0,
            # this is negative for deep tips → rewards depth
            depth_loss = depth_alpha * torch.mean(tip)
            loss = recon_loss + smooth_loss + depth_loss
            loss.backward()

            # Spatial gradient smoothing for noisy conditions (early epochs only)
            # Removes high-frequency noise from the gradient without modifying
            # the forward pass. Helps the optimizer find the correct basin.
            if use_grad_smooth and epoch < 20:
                with torch.no_grad():
                    g = tip.grad.unsqueeze(0).unsqueeze(0)
                    g_smooth = F.avg_pool2d(g, kernel_size=3, stride=1, padding=1)
                    tip.grad.data = g_smooth.squeeze(0).squeeze(0)

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

    hard_frame_count = max(1, (nframe + 1) // 2)
    hard_frame_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_frame_count, largest=True
    ).indices.tolist()

    # STAGE 2: Hard-frame refinement (reduce depth penalty for fine-tuning)
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        # Decay depth penalty in Stage 2 — shape is already established
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay_progress)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_frame_indices:
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
