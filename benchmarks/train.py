"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def laplacian_smoothing(tip, weight=0.01):
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    return weight * (torch.mean(lap_x) + torch.mean(lap_y))


def laplacian_smoothing_2d(image, weight=0.001):
    """Laplacian smoothing for a single 2D image (surface)."""
    lap_x = (image[2:, 1:-1] - 2 * image[1:-1, 1:-1] + image[0:-2, 1:-1]) ** 2
    lap_y = (image[1:-1, 2:] - 2 * image[1:-1, 1:-1] + image[1:-1, 0:-2]) ** 2
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


def reconstruct_tip(images, tip_size, **kwargs):
    """Joint surface-tip BTR with consistency anchoring.

    Stage 0: Standard BTR for initial tip (60 epochs)
    Stage 1: Joint optimization of surfaces + tip
      - Surfaces: per-frame free variables
      - Anchored via consistency loss: S_i ≈ erosion(image_i, tip).detach()
      - Tip updated via dilation loss: dilation(S_i, tip) ≈ image_i
      - Surfaces have flexibility to deviate from exact erosion (denoise)
      - Tip identifiability preserved through anchoring
    Stage 2: Standard hard-frame refinement
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

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # ── Stage 0: Standard BTR for initial tip (60 epochs) ──
    for epoch in range(60):
        lr_factor = 0.6 + (epoch / 60) * 0.4 if epoch < 20 else 1.0
        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, 0.003) + depth_alpha * torch.mean(tip)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 1: Joint surface-tip optimization with anchoring ──
    # Initialize surfaces from erosion
    surfaces = []
    with torch.no_grad():
        for iframe in range(nframe):
            s = ierosion(images[iframe], tip)
            surfaces.append(s.clone())
    surfaces_param = torch.stack(surfaces).requires_grad_(True)

    # Joint optimizer
    optimizer_joint = optim.AdamW([
        {'params': surfaces_param, 'lr': 0.05, 'weight_decay': 0.0},
        {'params': [tip], 'lr': 0.05, 'weight_decay': 0.01},
    ])

    nepoch_joint = 80 if not is_high_gaussian else 120
    consistency_alpha = 0.5  # weight for surface anchoring

    for epoch in range(nepoch_joint):
        progress = epoch / nepoch_joint
        lr_decay = 1.0 - 0.7 * progress
        for pg in optimizer_joint.param_groups:
            pg['lr'] = 0.05 * lr_decay

        # Re-anchor surfaces every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                for iframe in range(nframe):
                    surfaces_param.data[iframe] = ierosion(images[iframe], tip)

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer_joint.zero_grad()

            S = surfaces_param[iframe]
            # Reconstruction: dilation(S, tip) ≈ image
            recon = idilation(S, tip)
            recon_loss = torch.mean((recon - images[iframe]) ** 2)

            # Consistency anchor: S ≈ erosion(image, tip).detach()
            with torch.no_grad():
                S_anchor = ierosion(images[iframe], tip)
            cons_loss = consistency_alpha * torch.mean((S - S_anchor) ** 2)

            # Surface smoothness
            surf_smooth = laplacian_smoothing_2d(S, 0.001)

            # Tip regularization
            tip_smooth = laplacian_smoothing(tip, 0.005)
            depth_loss = depth_alpha * torch.mean(tip)

            loss = recon_loss + cons_loss + surf_smooth + tip_smooth + depth_loss
            loss.backward()
            optimizer_joint.step()

            with torch.no_grad():
                # Surface non-negative
                surfaces_param.data[iframe] = torch.clamp(surfaces_param.data[iframe], min=0.0)
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 2: Standard hard-frame refinement ──
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            error = torch.mean((recon - images[iframe]) ** 2)
            frame_errors.append(error.item())

    hard_count = max(1, (nframe + 1) // 2)
    hard_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_count, largest=True
    ).indices.tolist()

    nepoch_s2 = 60 if not is_high_gaussian else 100
    for epoch in range(nepoch_s2):
        decay = epoch / nepoch_s2
        lr_factor = 0.1 ** decay
        smooth_weight = 0.01 + 0.01 * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer.zero_grad()
                recon = idilation(ierosion(images[iframe], tip), tip)
                loss = torch.mean((recon - images[iframe]) ** 2)
                loss = loss + laplacian_smoothing(tip, smooth_weight)
                loss = loss + depth_weight * torch.mean(tip)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)
                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
