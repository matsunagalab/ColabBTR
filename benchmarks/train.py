"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
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


def gaussian_blur(images, kernel_size):
    """Apply Gaussian blur to a batch of images for noise reduction.

    Args:
        images: (N, H, W) tensor
        kernel_size: int, must be odd
    Returns:
        Blurred images: (N, H, W)
    """
    sigma = kernel_size / 3.0
    x = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    batched = images.unsqueeze(1)  # (N, 1, H, W)
    blurred = F.conv2d(batched, kernel_2d, padding=padding)
    return blurred.squeeze(1)  # (N, H, W)


def _optimize_stage(images, tip, optimizer, nepoch, lr_schedule_fn,
                    smooth_schedule_fn, loss_train):
    """Run one stage of per-frame optimization."""
    nframe = images.shape[0]
    for epoch in range(nepoch):
        lr, wd, smooth_weight = lr_schedule_fn(epoch, nepoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['weight_decay'] = wd

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


def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-Annealed Coarse-to-Fine BTR.

    Architecture: Progressively reduce image smoothing during optimization.
    Analogous to simulated annealing / graduated non-convexity.

    Phase 1: Optimize on heavily blurred images (smooth loss landscape,
             find correct basin even with high noise)
    Phase 2: Refine on lightly blurred images (recover medium-scale features)
    Phase 3: Refine on original images (recover fine detail)
    Phase 4: Hard-frame refinement (original baseline Stage 2)

    This addresses the root cause of high-noise failure: the noisy loss
    landscape has many local minima. Smoothing the images smooths the
    landscape, enabling convergence to the correct basin.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Create progressively less smoothed images
    images_heavy = gaussian_blur(images, kernel_size=5)
    images_light = gaussian_blur(images, kernel_size=3)

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    # Phase 1: Heavy blur — find correct basin (40 epochs)
    for epoch in range(40):
        lr_factor = 0.6 + (epoch / 40) * 0.4
        lr = 0.1 * lr_factor
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['weight_decay'] = 0.01

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images_heavy[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images_heavy[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.001)
            loss = recon_loss + smooth_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # Phase 2: Light blur — recover medium-scale features (40 epochs)
    for epoch in range(40):
        for pg in optimizer.param_groups:
            pg['lr'] = 0.1
            pg['weight_decay'] = 0.01

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images_light[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images_light[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.003)
            loss = recon_loss + smooth_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # Phase 3: Original images — recover fine detail (60 epochs)
    for epoch in range(60):
        if epoch < 40:
            smooth_weight = 0.005
            lr_factor = 1.0
        else:
            decay_progress = (epoch - 40) / 20
            lr_factor = 0.1 ** decay_progress
            smooth_weight = 0.01

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01

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

    # Phase 4: Hard-frame refinement (same as baseline Stage 2)
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

    for epoch in range(60):
        decay_progress = epoch / 60
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
