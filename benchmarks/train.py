"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def laplacian_smoothing(tip, weight=0.01):
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    return weight * (torch.mean(lap_x) + torch.mean(lap_y))


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


def _optimize_tip_stage(images, tip, nepoch, is_high_gaussian, depth_alpha):
    """Run one stage of per-frame BTR optimization."""
    device = images.device
    nframe = images.shape[0]
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    loss_train = []
    for epoch in range(nepoch):
        epoch_40 = int(nepoch * 40 / 140)
        epoch_120 = int(nepoch * 120 / 140)
        epoch_cooldown = nepoch - epoch_120

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

    return loss_train


def _upsample_tip(tip, new_size):
    """Upsample tip to a larger size using bilinear interpolation."""
    tip_4d = tip.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    upsampled = F.interpolate(tip_4d, size=new_size, mode='bilinear',
                               align_corners=True)
    result = upsampled.squeeze(0).squeeze(0)
    # Ensure tip properties
    result = torch.clamp(result, max=0.0)
    return result


def reconstruct_tip(images, tip_size, **kwargs):
    """Progressive tip resolution BTR.

    Novel architecture: optimize the tip at progressively higher resolutions.

    Phase 1: 5×5 tip (25 params) — few parameters, well-conditioned
      optimization, captures coarse tip shape robustly.
    Phase 2: Upsample to 9×9, refine (81 params) — add medium detail.
    Phase 3: Upsample to target size, refine (225 params for 15×15).
    Phase 4: Standard hard-frame refinement at full resolution.

    Why this works:
    - Small tip → fewer local minima → finds correct basin
    - Coarse shape is preserved through upsampling
    - Each phase starts from a good initialization (previous phase)
    - Images stay at full resolution throughout

    Unlike multi-scale on images (which loses morphological info),
    multi-resolution on the TIP is well-defined: a smaller tip kernel
    simply captures a smaller neighborhood in erosion/dilation.
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

    target_h, target_w = tip_size
    loss_train = []

    # Resolution schedule: small → medium → target
    # Ensure odd sizes for centered tips
    sizes = [(5, 5), (9, 9), tip_size]
    epochs = [40, 40, 60] if not is_high_gaussian else [50, 50, 100]

    tip = None
    for i, ((h, w), nepoch) in enumerate(zip(sizes, epochs)):
        if tip is None:
            # First phase: init from zeros
            tip = torch.zeros((h, w), dtype=dtype, requires_grad=True, device=device)
        else:
            # Upsample from previous phase
            with torch.no_grad():
                tip_upsampled = _upsample_tip(tip.detach(), (h, w))
                tip_upsampled = translate_tip_mean(tip_upsampled)
            tip = tip_upsampled.clone().requires_grad_(True)

        stage_loss = _optimize_tip_stage(
            images, tip, nepoch, is_high_gaussian, depth_alpha)
        loss_train.extend(stage_loss)

    # Stage 4: Hard-frame refinement at full resolution
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

    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    nepoch_s2 = 60 if not is_high_gaussian else 100

    for epoch in range(nepoch_s2):
        decay_progress = epoch / nepoch_s2
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
