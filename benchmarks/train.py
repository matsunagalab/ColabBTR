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


def reconstruct_tip(images, tip_size, **kwargs):
    """Full-batch BTR: accumulate gradients from all frames before each update.

    Architectural change from baseline:
    - Baseline: update tip after EACH frame (20 noisy updates + 20 clamp/center per epoch)
    - This: accumulate gradients from ALL frames, one clean update + one clamp/center per epoch

    Benefits:
    - Cleaner gradient direction (averaged over all frames)
    - Less disruption from repeated clamp+center operations
    - Each update reflects the consensus of all frames simultaneously
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch_stage1 = 200
    nepoch_stage2 = 60
    loss_train = []

    # STAGE 1: Full-batch optimization on all frames
    for epoch in range(nepoch_stage1):
        if epoch < 40:
            lr_factor = 0.6 + (epoch / 40) * 0.4
            smooth_weight = 0.001
        elif epoch < 160:
            lr_factor = 1.0
            smooth_weight = 0.005
        else:
            decay_progress = (epoch - 160) / 40
            lr_factor = 0.1 ** decay_progress
            smooth_weight = 0.01

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01

        optimizer.zero_grad()
        total_loss = 0.0
        for iframe in range(nframe):
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            (recon_loss / nframe).backward()
            total_loss += recon_loss.item()

        smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
        smooth_loss.backward()
        total_loss += smooth_loss.item()

        optimizer.step()
        with torch.no_grad():
            tip.data = torch.clamp(tip, max=0.0)
            tip.data = translate_tip_mean(tip)

        loss_train.append(total_loss)

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

    # STAGE 2: Full-batch refinement on hard frames
    n_hard = len(hard_frame_indices)
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay_progress)

        optimizer.zero_grad()
        total_loss = 0.0
        for iframe in hard_frame_indices:
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            (recon_loss / n_hard).backward()
            total_loss += recon_loss.item()

        smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
        smooth_loss.backward()
        total_loss += smooth_loss.item()

        optimizer.step()
        with torch.no_grad():
            tip.data = torch.clamp(tip, max=0.0)
            tip.data = translate_tip_mean(tip)

        loss_train.append(total_loss)

    return tip.detach(), loss_train
