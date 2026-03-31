"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with robust Huber loss for noise resilience (novel).

    Theory: L2 loss is sensitive to outliers (noisy pixels). Huber loss smoothly
    transitions from L2 (near zero) to L1 (far from zero), providing robustness
    to noise while maintaining precise fitting of well-explained pixels.

    Innovation: Adaptive delta parameter for Huber loss that increases during
    refinement, progressively reducing emphasis on noise-contaminated pixels.

    Benefit: Better handles noisy AFM data by focusing on structural content.

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    device = images.device
    dtype = images.dtype

    # Initialize tip
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch = 200
    loss_train = []

    for epoch in tqdm(range(nepoch), disable=True):
        # Three-phase learning rate and weight decay schedule
        if epoch < 50:
            lr_factor = 0.6 + (epoch / 50) * 0.4
            wd_factor = 1.0
            # Tight loss threshold in warm-up (emphasize L2, fit everything)
            huber_delta = 0.1
        elif epoch < 150:
            lr_factor = 1.0
            wd_factor = 1.0
            # Moderate threshold in main phase
            huber_delta = 0.2
        else:
            decay_progress = (epoch - 150) / 50
            lr_factor = 1.0 * (0.1 ** decay_progress)
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)
            # Relax threshold in cool-down (progressively ignore outliers)
            huber_delta = 0.2 + decay_progress * 0.3  # 0.2 → 0.5

        # Update optimizer parameters
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * lr_factor
            param_group['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(images.shape[0]):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)

            # Huber loss: robust to noise, less sensitive to outliers than L2
            residual = image_reconstructed - images[iframe, :, :]
            loss = torch.mean(F.huber_loss(residual, torch.zeros_like(residual), delta=huber_delta, reduction='mean'))

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            residual = image_reconstructed - images[iframe, :, :]
            loss = torch.mean(F.huber_loss(residual, torch.zeros_like(residual), delta=huber_delta, reduction='mean'))
            loss_tmp += loss.item()

        loss_train.append(loss_tmp)

    tip_estimate = tip.detach()
    return tip_estimate, loss_train
