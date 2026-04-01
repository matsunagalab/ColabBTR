"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from tqdm.auto import tqdm
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def laplacian_smoothing(tip, weight=0.01):
    """Laplacian smoothing penalty: encourage smooth, physically plausible tips.

    Penalizes local roughness: sum of squared second differences.
    """
    # 2nd order differences in x and y directions
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    roughness = torch.mean(lap_x) + torch.mean(lap_y)
    return weight * roughness


def reconstruct_tip(images, tip_size, **kwargs):
    """Two-stage BTR with Laplacian smoothing for physical plausibility.

    Uses smoothness regularization to encourage physically realistic tip shapes
    while maintaining reconstruction quality.

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Initialize tip
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch_stage1 = 140
    nepoch_stage2 = 60
    loss_train = []

    # STAGE 1: Coarse optimization on all frames
    for epoch in tqdm(range(nepoch_stage1), disable=True, desc="Stage 1: Coarse"):
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
            lr_factor = 1.0 * (0.1 ** decay_progress)
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)
            smooth_weight = 0.01  # Stronger smoothing in cool-down

        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * lr_factor
            param_group['weight_decay'] = 0.01 * wd_factor

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

    # STAGE 2: Evaluate frame difficulties
    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            error = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            frame_errors.append(error.item())

    # Select hardest frames (top 50% or at least 1)
    hard_frame_count = max(1, (nframe + 1) // 2)
    hard_frame_indices = torch.topk(
        torch.tensor(frame_errors),
        k=hard_frame_count,
        largest=True
    ).indices.tolist()

    # STAGE 2: Intensive refinement on hard frames
    for epoch in tqdm(range(nepoch_stage2), disable=True, desc="Stage 2: Intensive"):
        # Cool-down schedule for final refinement
        decay_progress = epoch / nepoch_stage2
        lr_factor = 1.0 * (0.1 ** decay_progress)
        wd_factor = max(0.02, 1.0 - decay_progress)
        smooth_weight = 0.01 + 0.01 * decay_progress  # Increase smoothing toward end

        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * lr_factor
            param_group['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        # Only optimize on hard frames, multiple passes per epoch
        for _ in range(3):  # 3x more iterations on hard frames
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

    # STAGE 3: Stochastic Weight Averaging (SWA)
    # Average tip snapshots from continued optimization at constant low LR.
    # SWA finds wider optima that generalize better (Izmailov et al., 2018).
    # No shape assumptions — purely optimization-level improvement.
    swa_tip = tip.data.clone()
    swa_count = 1
    swa_lr = 0.005
    nepoch_swa = 30

    for param_group in optimizer.param_groups:
        param_group['lr'] = swa_lr
        param_group['weight_decay'] = 0.001  # light regularization

    for epoch in range(nepoch_swa):
        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.01)
            loss = recon_loss + smooth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()

        # Collect snapshot at end of each SWA epoch
        with torch.no_grad():
            swa_tip += tip.data
            swa_count += 1

        loss_train.append(loss_tmp)

    # Apply SWA average
    with torch.no_grad():
        tip.data = swa_tip / swa_count
        tip.data = torch.clamp(tip, max=0.0)
        tip.data = translate_tip_mean(tip)

    tip_estimate = tip.detach()
    return tip_estimate, loss_train
