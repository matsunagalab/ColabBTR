"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import math
import torch
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


def reconstruct_tip(images, tip_size, **kwargs):
    """Bayesian BTR via Stochastic Gradient Langevin Dynamics (SGLD).

    Treats the tip as a random variable with:
    - Prior: Laplacian smoothness + non-positivity constraint
    - Likelihood: Gaussian noise model on reconstruction error

    Architecture:
    Stage 1: AdamW warmup (80 epochs) — find a good basin
    Stage 2: SGLD sampling (80 epochs) — explore posterior around the mode
      θ_{t+1} = θ_t - lr * ∇L(θ) + √(2·lr·T) · ε,  ε ~ N(0,I)
      Temperature anneals from T_start to T_end
    Stage 3: Bayesian model averaging — average the last N tip samples
      This is more robust than a point estimate: local minima are
      averaged out, giving a smoother, more reliable tip estimate.
    Stage 4: AdamW hard-frame refinement from the averaged tip.

    The SGLD noise helps escape local minima during Stage 2, and the
    Bayesian averaging in Stage 3 provides a tip estimate that reflects
    the POSTERIOR MODE rather than a single optimization trajectory.
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

    # ── Stage 1: AdamW warmup ──
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    nepoch_warmup = 80 if not is_high_gaussian else 120

    for epoch in range(nepoch_warmup):
        if epoch < 30:
            lr_factor = 0.6 + (epoch / 30) * 0.4
            smooth_weight = 0.001
        elif epoch < 60:
            lr_factor = 1.0
            smooth_weight = 0.005
        else:
            d = (epoch - 60) / (nepoch_warmup - 60)
            lr_factor = 0.3 ** d
            smooth_weight = 0.008

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, smooth_weight)
            loss = loss + depth_alpha * torch.mean(tip)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 2: SGLD sampling ──
    nepoch_sgld = 60 if not is_high_gaussian else 80
    n_collect = 30  # collect last 30 samples for averaging
    collected_tips = []

    # Temperature schedule: start warm, anneal to cold
    T_start = 0.01
    T_end = 0.001
    sgld_lr = 0.01

    for epoch in range(nepoch_sgld):
        progress = epoch / nepoch_sgld
        temperature = T_start * (T_end / T_start) ** progress
        current_lr = sgld_lr * (1.0 - 0.5 * progress)

        loss_tmp = 0.0
        for iframe in range(nframe):
            # Compute gradient
            if tip.grad is not None:
                tip.grad.zero_()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, 0.005)
            loss = loss + depth_alpha * torch.mean(tip)
            loss.backward()

            # SGLD update: gradient descent + Langevin noise
            with torch.no_grad():
                noise = torch.randn_like(tip) * math.sqrt(2 * current_lr * temperature)
                tip.data -= current_lr * tip.grad + noise

                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

        # Collect samples for averaging (last n_collect epochs)
        if epoch >= nepoch_sgld - n_collect:
            collected_tips.append(tip.data.clone())

    # ── Stage 3: Bayesian model averaging ──
    with torch.no_grad():
        tip_stack = torch.stack(collected_tips)
        # Use MEDIAN for robustness (vs mean which can be pulled by outliers)
        tip_averaged = tip_stack.median(dim=0).values
        tip_averaged = torch.clamp(tip_averaged, max=0.0)
        tip_averaged = translate_tip_mean(tip_averaged)

    tip = tip_averaged.clone().requires_grad_(True)

    # ── Stage 4: Hard-frame refinement ──
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

    optimizer2 = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    nepoch_s2 = 60 if not is_high_gaussian else 80

    for epoch in range(nepoch_s2):
        decay = epoch / nepoch_s2
        lr_factor = 0.1 ** decay
        smooth_weight = 0.01 + 0.01 * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer2.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer2.zero_grad()
                recon = idilation(ierosion(images[iframe], tip), tip)
                recon_loss = torch.mean((recon - images[iframe]) ** 2)
                smooth_loss = laplacian_smoothing(tip, smooth_weight)
                depth_loss = depth_weight * torch.mean(tip)
                loss = recon_loss + smooth_loss + depth_loss
                loss.backward()
                optimizer2.step()

                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)

                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
