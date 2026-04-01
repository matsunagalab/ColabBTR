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


def estimate_noise_level(images):
    """Estimate noise level from images using MAD of the discrete Laplacian."""
    estimates = []
    for i in range(images.shape[0]):
        img = images[i]
        lap = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]
               - 4 * img[1:-1, 1:-1])
        mad = torch.median(torch.abs(lap - torch.median(lap))).item()
        estimates.append(mad)
    return float(torch.median(torch.tensor(estimates)).item())


def lorentzian_loss(residuals, sigma):
    """Lorentzian (Cauchy) robust loss function.

    L(x) = log(1 + (x/σ)²)

    Properties:
    - Near x=0: ≈ x²/σ² (quadratic, like MSE — preserves sensitivity)
    - Far from x=0: ≈ 2·log(|x|/σ) (logarithmic — bounds outlier influence)
    - Unlike Tukey bisquare, gradient never reaches zero (always provides signal)
    - σ controls the transition: residuals << σ → quadratic, >> σ → robust

    This directly addresses the root cause of high-noise failure: noisy pixels
    produce large residuals that dominate the MSE gradient, pulling the tip
    toward wrong local minima.
    """
    # Multiply by σ² so that for small residuals: σ²·log(1+(r/σ)²) ≈ r² (matches MSE)
    return sigma ** 2 * torch.mean(torch.log1p((residuals / sigma) ** 2))


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with Lorentzian robust loss and noise-adaptive scale.

    Architecture change from baseline:
    - Replace MSE with Lorentzian loss for the reconstruction term
    - Scale parameter σ is set from estimated noise level
    - For clean data: σ is small → Lorentzian ≈ MSE (no change)
    - For noisy data: σ is large → outlier pixels are downweighted

    Everything else (Laplacian smoothing, two-stage optimization,
    hard-frame selection) remains identical to baseline.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Estimate noise and set Lorentzian scale
    noise_level = estimate_noise_level(images)
    # For clean data (noise≈0): sigma small → Lorentzian ≈ MSE
    # For noisy data (noise≈3): sigma large → robust to outlier pixels
    sigma = max(0.1, noise_level * 0.5)

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch_stage1 = 140
    nepoch_stage2 = 60
    loss_train = []

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
            residuals = image_reconstructed - images[iframe]
            recon_loss = lorentzian_loss(residuals, sigma)
            smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
            loss = recon_loss + smooth_loss
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

    hard_frame_count = max(1, (nframe + 1) // 2)
    hard_frame_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_frame_count, largest=True
    ).indices.tolist()

    # STAGE 2: Hard-frame refinement
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay_progress)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_frame_indices:
                optimizer.zero_grad()
                image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
                residuals = image_reconstructed - images[iframe]
                recon_loss = lorentzian_loss(residuals, sigma)
                smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
                loss = recon_loss + smooth_loss
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)

                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
