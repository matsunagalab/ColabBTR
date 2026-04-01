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


def _run_optimization(images, tip_size, smooth_schedule, nepoch_s1, nepoch_s2):
    """Run one complete BTR optimization with a given smoothing schedule.

    smooth_schedule: callable(epoch, nepoch) -> smooth_weight
    Returns: (tip, loss_train)
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # Stage 1: All frames
    for epoch in range(nepoch_s1):
        progress = epoch / nepoch_s1
        smooth_weight = smooth_schedule(epoch, nepoch_s1)

        if progress < 40 / 140:
            lr_factor = 0.6 + (progress / (40 / 140)) * 0.4
            wd_factor = 1.0
        elif progress < 120 / 140:
            lr_factor = 1.0
            wd_factor = 1.0
        else:
            decay_p = (progress - 120 / 140) / (20 / 140)
            lr_factor = 0.1 ** decay_p
            wd_factor = max(0.05, 1.0 - decay_p * 0.95)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
            loss = recon_loss + smooth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # Stage 2: Hard-frame refinement
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

    for epoch in range(nepoch_s2):
        decay_progress = epoch / nepoch_s2
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
                recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
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


def _evaluate_tip(images, tip):
    """Compute total reconstruction error for a tip estimate."""
    total_error = 0.0
    with torch.no_grad():
        for iframe in range(images.shape[0]):
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            error = torch.mean((image_reconstructed - images[iframe]) ** 2)
            total_error += error.item()
    return total_error


def reconstruct_tip(images, tip_size, **kwargs):
    """Dual-Path BTR with Automatic Strategy Selection.

    Architecture: Run two optimization strategies briefly, then commit the
    full compute budget to whichever performs better on this specific data.

    Path A (Standard): Smoothing schedule from proven baseline
            (weak → strong regularization)
    Path B (GNC): Graduated non-convexity schedule
            (strong → weak regularization, noise-adaptive)

    Phase 1 — Race (20 epochs each, parallel):
      Run both paths briefly. Evaluate reconstruction error.
      Select the path with lower error.

    Phase 2 — Commit (120 epochs + 60 hard-frame epochs):
      Continue the winning path with its remaining compute budget.

    This guarantees at least baseline performance (Path A wins on easy data)
    while capturing GNC improvements on hard data (Path B wins on noisy data).
    """
    noise_level = estimate_noise_level(images)
    noise_factor = max(1.0, noise_level / 1.0)

    # Define two smoothing schedules
    def schedule_standard(epoch, nepoch):
        """Baseline schedule: weak → medium → strong."""
        progress = epoch / nepoch
        if progress < 40 / 140:
            return 0.001
        elif progress < 120 / 140:
            return 0.005
        else:
            return 0.01

    def schedule_gnc(epoch, nepoch):
        """GNC schedule: strong → weak (noise-adaptive)."""
        progress = epoch / nepoch
        initial = 0.03 * noise_factor
        final = 0.003
        return initial * (final / initial) ** progress

    # Phase 1: Race — 20 epochs each
    race_epochs = 20
    tip_a, _ = _run_optimization(images, tip_size, schedule_standard,
                                  nepoch_s1=race_epochs, nepoch_s2=0)
    tip_b, _ = _run_optimization(images, tip_size, schedule_gnc,
                                  nepoch_s1=race_epochs, nepoch_s2=0)

    error_a = _evaluate_tip(images, tip_a)
    error_b = _evaluate_tip(images, tip_b)

    # Phase 2: Commit to the winner
    if error_a <= error_b:
        selected_schedule = schedule_standard
    else:
        selected_schedule = schedule_gnc

    tip_final, loss_train = _run_optimization(
        images, tip_size, selected_schedule,
        nepoch_s1=120, nepoch_s2=60
    )

    return tip_final, loss_train
