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

    # SGLD only for clean/moderate data (noisy data has natural exploration)
    use_sgld = hf_energy < 0.5

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    loss_train = []

    # ── Stage 1: AdamW optimization ──
    # If SGLD will follow: shorter warmup (SGLD continues exploration)
    # If no SGLD: full baseline-equivalent optimization
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    if use_sgld:
        nepoch_warmup = 80 if not is_high_gaussian else 120
    else:
        nepoch_warmup = 140 if not is_high_gaussian else 200

    for epoch in range(nepoch_warmup):
        # Use epoch-ratio breakpoints scaled to nepoch_warmup
        e40 = int(nepoch_warmup * 40 / 140)
        e120 = int(nepoch_warmup * 120 / 140)
        ecd = nepoch_warmup - e120
        if epoch < e40:
            lr_factor = 0.6 + (epoch / e40) * 0.4
            wd_factor = 1.0
            smooth_weight = 0.001 if not is_high_gaussian else 0.002
        elif epoch < e120:
            lr_factor = 1.0
            wd_factor = 1.0
            smooth_weight = 0.005 if not is_high_gaussian else 0.008
        else:
            d = (epoch - e120) / max(1, ecd)
            lr_factor = 0.1 ** d
            wd_factor = max(0.05, 1.0 - d * 0.95)
            smooth_weight = 0.01 if not is_high_gaussian else 0.016

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

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

    # ── Stage 2: SGLD sampling (only for clean/moderate data) ──
    nepoch_sgld = (60 if not is_high_gaussian else 80) if use_sgld else 0
    n_collect = 40  # collect last 40 samples for averaging
    collected_tips = []

    # Noise-adaptive temperature:
    # Clean data needs MORE exploration (flat landscape, local minima)
    # Noisy data needs LESS (gradient noise already provides exploration)
    if hf_energy < 0.2:
        T_start, T_end = 0.01, 0.001    # clean: explore aggressively
    elif hf_energy < 0.5:
        T_start, T_end = 0.003, 0.0005  # moderate: mild exploration
    else:
        T_start, T_end = 0.001, 0.0001  # noisy: minimal (gradient noise suffices)
    sgld_lr = 0.01

    # Fixed generator for reproducibility
    sgld_rng = torch.Generator(device=device)
    sgld_rng.manual_seed(12345)

    for epoch in range(nepoch_sgld):
        progress = epoch / nepoch_sgld
        temperature = T_start * (T_end / T_start) ** progress
        current_lr = sgld_lr * (1.0 - 0.5 * progress)

        loss_tmp = 0.0
        for iframe in range(nframe):
            if tip.grad is not None:
                tip.grad.zero_()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, 0.005)
            loss = loss + depth_alpha * torch.mean(tip)
            loss.backward()

            # SGLD update: gradient descent + Langevin noise
            with torch.no_grad():
                noise_scale = math.sqrt(2 * current_lr * temperature)
                noise = torch.randn(tip.shape, generator=sgld_rng,
                                    device=device, dtype=dtype) * noise_scale
                tip.data -= current_lr * tip.grad + noise

                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

        # Collect samples for averaging (last n_collect epochs)
        if epoch >= nepoch_sgld - n_collect:
            collected_tips.append(tip.data.clone())

    # ── Stage 3: Bayesian model averaging (if SGLD was used) ──
    if collected_tips:
        with torch.no_grad():
            tip_stack = torch.stack(collected_tips)
            tip_averaged = tip_stack.mean(dim=0)
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
    nepoch_s2 = 60 if not is_high_gaussian else 100

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
