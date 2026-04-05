"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean, fixed_padding
)


# ──────────────────────────────────────────────────────────────────────
# Soft morphological operations with entropic regularization
# ──────────────────────────────────────────────────────────────────────

def soft_dilation(image, tip, temperature):
    """Soft dilation: replace hard max with tempered log-sum-exp.

    Standard: dilation(x,y) = max_k [patch_k + tip_k]
    Soft:     dilation_T    = T · log Σ_k exp((patch_k + tip_k) / T)

    At T→0: exact morphological dilation
    At T>0: smooth approximation, gradient flows to ALL kernel elements
            proportionally (softmax weighting instead of argmax)
    """
    H, W = image.shape
    kernel_size = tip.shape[0]
    x = image.unsqueeze(0).unsqueeze(0)
    x = fixed_padding(x, torch.tensor(kernel_size), dilation=torch.tensor(1),
                       pad_value=float('-inf'))
    x = F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)
    x = x.unsqueeze(1)

    weight = tip.view(1, 1, -1, 1)
    x = weight + x  # (1, 1, k*k, L)

    # Soft max via log-sum-exp
    x = temperature * torch.logsumexp(x / temperature, dim=2, keepdim=False)
    x = x.view(-1, 1, H, W)
    return x.squeeze(0).squeeze(0)


def soft_erosion(surface, tip, temperature):
    """Soft erosion: replace hard min with tempered smooth-min.

    Standard: erosion(x,y) = min_k [patch_k - tip_k] = -max_k [tip_k - patch_k]
    Soft:     erosion_T    = -T · log Σ_k exp((tip_k - patch_k) / T)

    At T→0: exact morphological erosion
    At T>0: smooth approximation, less sensitive to single noisy pixels
    """
    kernel_size = tip.shape[0]
    H, W = surface.shape
    x = surface.unsqueeze(0).unsqueeze(0)
    x = fixed_padding(x, torch.tensor(kernel_size), dilation=torch.tensor(1),
                       pad_value=float('inf'))
    x = F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)
    x = x.unsqueeze(1)

    weight = tip.view(1, 1, -1, 1)
    x = weight - x  # (1, 1, k*k, L)

    # Soft max via log-sum-exp, then negate for soft min
    x = temperature * torch.logsumexp(x / temperature, dim=2, keepdim=False)
    x = -x
    x = x.view(-1, 1, H, W)
    return x.squeeze(0).squeeze(0)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def laplacian_smoothing(tip, weight=0.01):
    """Laplacian smoothing penalty: encourage smooth, physically plausible tips."""
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


# ──────────────────────────────────────────────────────────────────────
# Main reconstruction
# ──────────────────────────────────────────────────────────────────────

def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with entropic-regularized soft morphology + temperature annealing.

    Key innovation: Replace hard min/max in erosion/dilation with
    temperature-controlled log-sum-exp (soft morphology).

    Benefits:
    1. Gradient flows to ALL kernel elements (not just argmax/argmin)
       → smoother optimization landscape, fewer local minima
    2. At high temperature: robust to noise (soft operations average
       over local noise fluctuations)
    3. Temperature annealing: start warm (smooth, find correct basin)
       → anneal to cold (sharp, recover exact morphology)

    This is entropic regularization applied to morphological operators,
    analogous to Sinkhorn regularization in optimal transport.

    Stage 1: Soft morphology with temperature annealing (warm → cold)
    Stage 2: Standard (hard) morphology for hard-frame refinement
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

    if is_high_gaussian:
        nepoch_stage1 = 200
        nepoch_stage2 = 100
    else:
        nepoch_stage1 = 140
        nepoch_stage2 = 60

    # Temperature schedule: anneal from warm to cold
    # Warm (T=1.0): smooth, noise-robust, good gradient flow
    # Cold (T=0.01): approaches exact morphological operations
    T_start = 0.1
    T_end = 0.01

    # STAGE 1: Soft morphology with temperature annealing
    for epoch in range(nepoch_stage1):
        epoch_40 = int(nepoch_stage1 * 40 / 140)
        epoch_120 = int(nepoch_stage1 * 120 / 140)
        epoch_cooldown = nepoch_stage1 - epoch_120

        # Temperature: exponential decay from T_start to T_end
        progress = epoch / nepoch_stage1
        temperature = T_start * (T_end / T_start) ** progress

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

            # Soft erosion (noise-robust) + hard dilation (exact forward model)
            surface_est = soft_erosion(images[iframe], tip, temperature)
            image_reconstructed = idilation(surface_est, tip)

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

    # STAGE 2: Hard morphology for fine-tuning (exact operations)
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
