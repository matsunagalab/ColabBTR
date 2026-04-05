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


def reconstruct_tip(images, tip_size, **kwargs):
    """Hybrid L-BFGS + AdamW BTR with noise-adaptive features.

    Architecture:
    - Stage 0: AdamW warm-up (40 epochs) — initialize tip from flat
    - Stage 1: L-BFGS full-batch optimization (30 steps) — exploit
      curvature information for faster, more accurate convergence.
      Each L-BFGS step uses ALL frames simultaneously.
    - Stage 2: AdamW hard-frame refinement (60 epochs) — fine-tune

    L-BFGS advantages over AdamW for BTR:
    - Uses Hessian approximation → better step direction for deep tips
    - Full-batch → clean gradient, no per-frame oscillation
    - Faster convergence in smooth regions of the loss landscape
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
    loss_train = []

    # ── Stage 0: AdamW warm-up (escape flat initialization) ──
    optimizer_adam = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    for epoch in range(40):
        lr_factor = 0.6 + (epoch / 40) * 0.4
        for pg in optimizer_adam.param_groups:
            pg['lr'] = 0.1 * lr_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer_adam.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, 0.003) + depth_alpha * torch.mean(tip)
            loss.backward()
            optimizer_adam.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 1: L-BFGS full-batch optimization ──
    n_lbfgs_steps = 40 if not is_high_gaussian else 60

    optimizer_lbfgs = optim.LBFGS(
        [tip], lr=0.5, max_iter=10, history_size=10,
        line_search_fn='strong_wolfe',
    )

    for step in range(n_lbfgs_steps):
        progress = step / n_lbfgs_steps
        smooth_weight = 0.005 + 0.005 * progress

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
            for iframe in range(nframe):
                recon = idilation(ierosion(images[iframe], tip), tip)
                total_loss = total_loss + torch.mean((recon - images[iframe]) ** 2)
            total_loss = total_loss / nframe
            total_loss = total_loss + laplacian_smoothing(tip, smooth_weight)
            total_loss = total_loss + depth_alpha * torch.mean(tip)
            total_loss.backward()
            return total_loss

        loss_val = optimizer_lbfgs.step(closure)
        with torch.no_grad():
            tip.data = torch.clamp(tip, max=0.0)
            tip.data = translate_tip_mean(tip)

        if loss_val is not None:
            loss_train.append(loss_val.item())

    # ── Stage 2: AdamW hard-frame refinement ──
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

    optimizer_adam2 = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    nepoch_s2 = 60 if not is_high_gaussian else 100

    for epoch in range(nepoch_s2):
        decay = epoch / nepoch_s2
        lr_factor = 0.1 ** decay
        smooth_weight = 0.01 + 0.01 * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer_adam2.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer_adam2.zero_grad()
                recon = idilation(ierosion(images[iframe], tip), tip)
                recon_loss = torch.mean((recon - images[iframe]) ** 2)
                smooth_loss = laplacian_smoothing(tip, smooth_weight)
                depth_loss = depth_weight * torch.mean(tip)
                loss = recon_loss + smooth_loss + depth_loss
                loss.backward()
                optimizer_adam2.step()

                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)

                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
