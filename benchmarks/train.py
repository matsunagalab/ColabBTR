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
    """Bright/dark variance ratio to distinguish noise types."""
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


def _quick_btr(images, tip_size, weight_decay, nepoch, smooth_weight=0.005):
    """Run a short BTR for lambda cross-validation."""
    device, dtype, nframe = images.device, images.dtype, images.shape[0]
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    opt = optim.AdamW([tip], lr=0.1, weight_decay=weight_decay)
    for epoch in range(nepoch):
        for iframe in range(nframe):
            opt.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, weight=smooth_weight)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
    return tip.detach()


def select_lambda(images, tip_size, candidates, nepoch_cv=50):
    """Select optimal weight_decay via 2-fold cross-validation.

    Split frames into two halves. For each lambda candidate, train on
    one half and evaluate reconstruction loss on the other. Select the
    lambda with the lowest mean validation loss.

    Cost: len(candidates) * 2 folds * nepoch_cv * (nframe/2) forward passes.
    With 4 candidates, 50 epochs, 10 frames: ~4000 forward passes (~12s).
    """
    nframe = images.shape[0]
    half = nframe // 2

    best_lambda = candidates[len(candidates) // 2]  # default: middle value
    best_val_loss = float('inf')

    for lam in candidates:
        total_val_loss = 0.0

        for fold in range(2):
            if fold == 0:
                train_imgs = images[:half]
                val_imgs = images[half:]
            else:
                train_imgs = images[half:]
                val_imgs = images[:half]

            tip = _quick_btr(train_imgs, tip_size, weight_decay=lam, nepoch=nepoch_cv)

            with torch.no_grad():
                for i in range(val_imgs.shape[0]):
                    recon = idilation(ierosion(val_imgs[i], tip), tip)
                    total_val_loss += torch.mean((recon - val_imgs[i]) ** 2).item()

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_lambda = lam

    return best_lambda


def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-adaptive BTR with auto-selected lambda (weight_decay).

    Architecture:
    1. Noise detection (HF energy + variance ratio)
    2. Lambda auto-selection via 2-fold cross-validation
    3. Full BTR with selected lambda + noise-specific adaptations

    Lambda candidates are chosen based on noise regime:
    - Clean/moderate: [0.001, 0.003, 0.01, 0.03]
    - High Gaussian: [0.0003, 0.001, 0.003, 0.01]
      (lower range because clamping already regularizes)
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Noise detection
    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)

    # Depth regularizer for clean data only
    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    # Physical preprocessing for high Gaussian noise
    if is_high_gaussian:
        images_for_btr = torch.clamp(images, min=0.0)
    else:
        images_for_btr = images

    # Auto-select lambda via cross-validation
    if is_high_gaussian:
        lambda_candidates = [0.0003, 0.001, 0.003, 0.01]
    else:
        lambda_candidates = [0.001, 0.003, 0.01, 0.03]

    optimal_lambda = select_lambda(images_for_btr, tip_size, lambda_candidates,
                                   nepoch_cv=50)

    # Main BTR with selected lambda
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=optimal_lambda)
    loss_train = []

    # Noise-adaptive epoch counts
    if is_high_gaussian:
        nepoch_stage1 = 200
        nepoch_stage2 = 100
    else:
        nepoch_stage1 = 140
        nepoch_stage2 = 60

    # STAGE 1
    for epoch in range(nepoch_stage1):
        epoch_40 = int(nepoch_stage1 * 40 / 140)
        epoch_120 = int(nepoch_stage1 * 120 / 140)
        epoch_cooldown = nepoch_stage1 - epoch_120

        if epoch < epoch_40:
            lr_factor = 0.6 + (epoch / epoch_40) * 0.4
            wd_factor = 1.0
            smooth_weight = 0.001 if not is_high_gaussian else 0.002
        elif epoch < epoch_120:
            lr_factor = 1.0
            wd_factor = 1.0
            smooth_weight = 0.005 if not is_high_gaussian else 0.008
        else:
            decay_progress = (epoch - epoch_120) / max(1, epoch_cooldown)
            lr_factor = 0.1 ** decay_progress
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)
            smooth_weight = 0.01 if not is_high_gaussian else 0.016

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = optimal_lambda * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images_for_btr[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images_for_btr[iframe]) ** 2)
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

    # Frame error evaluation for hard-frame selection
    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            image_reconstructed = idilation(ierosion(images_for_btr[iframe], tip), tip)
            error = torch.mean((image_reconstructed - images_for_btr[iframe]) ** 2)
            frame_errors.append(error.item())

    hard_count = max(1, (nframe + 1) // 2)
    hard_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_count, largest=True
    ).indices.tolist()

    # STAGE 2: Hard-frame refinement
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = optimal_lambda * max(0.02, 1.0 - decay_progress)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer.zero_grad()
                image_reconstructed = idilation(ierosion(images_for_btr[iframe], tip), tip)
                recon_loss = torch.mean((image_reconstructed - images_for_btr[iframe]) ** 2)
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
