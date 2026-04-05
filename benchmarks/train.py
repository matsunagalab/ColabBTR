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
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def estimate_variance_ratio(images):
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


def _single_frame_tip(image, tip_size, nepoch=30):
    """Estimate tip from a single frame (short optimization)."""
    device, dtype = image.device, image.dtype
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    opt = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    for epoch in range(nepoch):
        opt.zero_grad()
        recon = idilation(ierosion(image, tip), tip)
        loss = torch.mean((recon - image) ** 2) + laplacian_smoothing(tip, 0.01)
        loss.backward()
        opt.step()
        with torch.no_grad():
            tip.data = torch.clamp(tip, max=0.0)
            tip.data = translate_tip_mean(tip)
    return tip.detach()


def reconstruct_tip(images, tip_size, **kwargs):
    """Cross-frame consensus BTR.

    Novel architecture:
    Stage 0 (new): Estimate tip from EACH frame independently (30 epochs).
      Take pixel-wise MEDIAN across all per-frame tips → consensus tip.
      Median is robust to outlier frames (noisy or uninformative).
      Uses this as INITIALIZATION instead of flat zeros.

    Stage 1: Standard BTR with consensus initialization (100 epochs).
    Stage 2: Hard-frame refinement (60 epochs).

    The consensus initialization provides a better starting point than
    flat zeros, especially for sharp tips where the optimizer needs to
    travel far in parameter space.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)

    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    if is_high_gaussian:
        images = torch.clamp(images, min=0.0)

    # ── Stage 0: Per-frame tip estimation + median consensus ──
    per_frame_tips = []
    for iframe in range(nframe):
        tip_i = _single_frame_tip(images[iframe], tip_size, nepoch=30)
        per_frame_tips.append(tip_i)

    # Consensus: pixel-wise median (robust to outlier frames)
    tip_stack = torch.stack(per_frame_tips)  # (nframe, H, W)
    consensus_tip = tip_stack.median(dim=0).values

    # Center the consensus tip
    with torch.no_grad():
        consensus_tip = torch.clamp(consensus_tip, max=0.0)
        consensus_tip = translate_tip_mean(consensus_tip)

    # Use consensus as initialization
    tip = consensus_tip.clone().requires_grad_(True)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # ── Stage 1: Full multi-frame BTR (from consensus init) ──
    if is_high_gaussian:
        nepoch_stage1 = 160
        nepoch_stage2 = 100
    else:
        nepoch_stage1 = 100
        nepoch_stage2 = 60

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

    # ── Stage 2: Hard-frame refinement ──
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

    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay_progress)

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
