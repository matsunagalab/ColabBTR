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


def _single_run_btr(images, tip_size, depth_alpha, is_high_gaussian,
                    nepoch_s1, nepoch_s2, shuffle_seed=None):
    """Single BTR run with optional frame shuffling."""
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # Optional: create a frame order generator for shuffling
    if shuffle_seed is not None:
        rng = torch.Generator()
        rng.manual_seed(shuffle_seed)
    else:
        rng = None

    # STAGE 1
    for epoch in range(nepoch_s1):
        epoch_40 = int(nepoch_s1 * 40 / 140)
        epoch_120 = int(nepoch_s1 * 120 / 140)
        epoch_cooldown = nepoch_s1 - epoch_120

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

        # Frame order: deterministic or shuffled
        if rng is not None:
            frame_order = torch.randperm(nframe, generator=rng).tolist()
        else:
            frame_order = list(range(nframe))

        loss_tmp = 0.0
        for iframe in frame_order:
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

    # Hard-frame selection
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

    # STAGE 2
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

    # Final reconstruction error (for selection between runs)
    total_error = 0.0
    with torch.no_grad():
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            total_error += torch.mean((recon - images[iframe]) ** 2).item()

    return tip.detach(), loss_train, total_error


def reconstruct_tip(images, tip_size, **kwargs):
    """Stochastic multi-restart BTR with noise-adaptive features.

    Runs BTR twice: once with deterministic frame order, once with
    shuffled frame order. Selects the run with lower reconstruction error.

    For conditions where the optimization landscape has multiple local
    minima (e.g., sharp tips with Gaussian noise), the shuffled run
    may find a better minimum.

    The deterministic run ensures at least baseline performance.
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

    if is_high_gaussian:
        nepoch_s1, nepoch_s2 = 200, 100
    else:
        nepoch_s1, nepoch_s2 = 140, 60

    # Run 1: deterministic (baseline behavior)
    tip1, loss1, error1 = _single_run_btr(
        images, tip_size, depth_alpha, is_high_gaussian,
        nepoch_s1, nepoch_s2, shuffle_seed=None)

    # Run 2: shuffled frame order (explore different basin)
    tip2, loss2, error2 = _single_run_btr(
        images, tip_size, depth_alpha, is_high_gaussian,
        nepoch_s1, nepoch_s2, shuffle_seed=42)

    # Select the better run
    if error2 < error1:
        return tip2, loss2
    else:
        return tip1, loss1
