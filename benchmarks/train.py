"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from tqdm.auto import tqdm
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
    """Estimate noise level from images using MAD of the discrete Laplacian.

    Returns a scalar estimate of noise amplitude. Zero for clean images,
    ~1 for sigma=0.3 Gaussian, ~3 for sigma=1.0 Gaussian.
    """
    estimates = []
    for i in range(images.shape[0]):
        img = images[i]
        lap = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]
               - 4 * img[1:-1, 1:-1])
        mad = torch.median(torch.abs(lap - torch.median(lap))).item()
        estimates.append(mad)
    return float(torch.median(torch.tensor(estimates)).item())


def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-Adaptive Graduated Regularization BTR.

    Architecture: Apply the Graduated Non-Convexity (GNC) principle to
    Laplacian smoothing regularization, with automatic noise adaptation.

    For high-noise conditions:
      - Start with very strong smoothing (tip forced smooth → noise-immune)
      - Gradually relax to recover fine detail
      This prevents the optimizer from getting trapped in noisy local minima.

    For low-noise conditions:
      - Standard moderate smoothing throughout (minimal change from baseline)
      - Preserves the good performance on easy conditions.

    The noise level is estimated from the data (MAD of Laplacian), making
    this fully automatic with no manual tuning.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Estimate noise level from the data
    noise_level = estimate_noise_level(images)
    # Noise factor: 1.0 for clean data, scales up with noise
    noise_factor = max(1.0, noise_level / 1.0)

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch_stage1 = 140
    nepoch_stage2 = 60
    loss_train = []

    # STAGE 1: GNC schedule — strong regularization first, then relax
    # For high noise: smooth_weight goes 0.05 → 0.005 (10x decrease)
    # For low noise: smooth_weight goes 0.005 → 0.003 (gentle decrease)
    for epoch in range(nepoch_stage1):
        # GNC smoothing schedule: exponential decay from strong to weak
        progress = epoch / nepoch_stage1
        if noise_factor > 1.5:
            # High noise: start very strong, decay significantly
            smooth_weight = 0.05 * noise_factor * (0.1 ** progress)
            smooth_weight = max(smooth_weight, 0.003)
        else:
            # Low noise: standard baseline schedule (proven to work well)
            if epoch < 40:
                smooth_weight = 0.001
            elif epoch < 120:
                smooth_weight = 0.005
            else:
                smooth_weight = 0.01

        # LR schedule: warm-up then constant then decay
        if epoch < 40:
            lr_factor = 0.6 + (epoch / 40) * 0.4
            wd_factor = 1.0
        elif epoch < 120:
            lr_factor = 1.0
            wd_factor = 1.0
        else:
            decay_progress = (epoch - 120) / 20
            lr_factor = 0.1 ** decay_progress
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
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
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            error = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            frame_errors.append(error.item())

    hard_frame_count = max(1, (nframe + 1) // 2)
    hard_frame_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_frame_count, largest=True
    ).indices.tolist()

    # STAGE 2: Hard-frame refinement (same as baseline)
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        wd_factor = max(0.02, 1.0 - decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_frame_indices:
                optimizer.zero_grad()
                image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
                recon_loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
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
