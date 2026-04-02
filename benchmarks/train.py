"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
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
    """Measure high-frequency energy in images as a noise proxy."""
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def gaussian_blur(images, kernel_size=3):
    """Apply Gaussian blur to a batch of images."""
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    k1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(images.unsqueeze(1), k2d, padding=kernel_size // 2).squeeze(1)


def _run_stage(images, tip, optimizer, nepoch, lr_schedule, smooth_schedule,
               depth_alpha, loss_train):
    """Run one optimization stage with per-frame updates."""
    nframe = images.shape[0]
    for epoch in range(nepoch):
        lr, wd, smooth_weight = lr_schedule(epoch, nepoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['weight_decay'] = wd

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


def _run_hard_frame_stage(images, tip, optimizer, nepoch, hard_indices,
                          depth_alpha, loss_train):
    """Run hard-frame refinement stage."""
    n_hard = len(hard_indices)
    for epoch in range(nepoch):
        decay = epoch / nepoch
        lr = 0.1 * (0.1 ** decay)
        smooth_weight = 0.01 + 0.01 * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

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


def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-adaptive BTR with preprocessing and extended compute.

    Three regimes based on image noise level:

    1. Clean data (HF < 0.2):
       Depth regularizer to break the flat-tip degeneracy.
       Standard compute budget.

    2. Moderate noise (HF 0.2–0.5):
       Standard baseline. No preprocessing needed.

    3. High noise (HF > 0.5):
       Preprocessing: clamp negatives (physical: image ≥ 0) + Gaussian blur.
       Phase 0: BTR on preprocessed images (smooth landscape → correct basin).
       Phase 1: BTR on original images (recover fine detail).
       Phase 2: Extended hard-frame refinement.
       Larger compute budget.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    hf_energy = estimate_high_freq_energy(images)
    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # Standard LR schedule
    def lr_schedule_warmup(epoch, nepoch):
        progress = epoch / nepoch
        if progress < 40 / 140:
            lr_factor = 0.6 + (progress / (40 / 140)) * 0.4
            return 0.1 * lr_factor, 0.01, 0.001
        elif progress < 120 / 140:
            return 0.1, 0.01, 0.005
        else:
            decay = (progress - 120 / 140) / (20 / 140)
            return 0.1 * (0.1 ** decay), 0.01 * max(0.05, 1.0 - decay * 0.95), 0.01

    def lr_schedule_steady(epoch, nepoch):
        return 0.1, 0.01, 0.005

    def lr_schedule_decay(epoch, nepoch):
        decay = epoch / nepoch
        return 0.1 * (0.5 ** decay), 0.01 * max(0.1, 1.0 - decay), 0.008

    if hf_energy > 0.5:
        # ─── HIGH NOISE REGIME ──────────────────────────────────────
        # Preprocessing: physical constraint + Gaussian blur
        images_denoised = torch.clamp(images, min=0.0)  # image ≥ 0 (physical)
        images_denoised = gaussian_blur(images_denoised, kernel_size=3)

        # Phase 0: BTR on denoised images → find correct basin
        _run_stage(images_denoised, tip, optimizer, nepoch=100,
                   lr_schedule=lr_schedule_warmup,
                   smooth_schedule=None, depth_alpha=0.0, loss_train=loss_train)

        # Phase 1: Refine on original images → recover fine detail
        _run_stage(images, tip, optimizer, nepoch=100,
                   lr_schedule=lr_schedule_decay,
                   smooth_schedule=None, depth_alpha=0.0, loss_train=loss_train)

        # Phase 2: Extended hard-frame refinement
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

        _run_hard_frame_stage(images, tip, optimizer, nepoch=80,
                              hard_indices=hard_indices, depth_alpha=0.0,
                              loss_train=loss_train)
    else:
        # ─── CLEAN / MODERATE NOISE REGIME ──────────────────────────
        _run_stage(images, tip, optimizer, nepoch=140,
                   lr_schedule=lr_schedule_warmup,
                   smooth_schedule=None, depth_alpha=depth_alpha,
                   loss_train=loss_train)

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

        _run_hard_frame_stage(images, tip, optimizer, nepoch=60,
                              hard_indices=hard_indices, depth_alpha=depth_alpha,
                              loss_train=loss_train)

    return tip.detach(), loss_train
