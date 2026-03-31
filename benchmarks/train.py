"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from tqdm.auto import tqdm
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with adaptive learning rate and weight decay scheduling.

    Multi-phase optimization:
    - Warm-up: moderate regularization, increasing learning rate
    - Main: full learning, stable heavy regularization
    - Cool-down: reduced regularization, decaying learning rate for fine-tuning

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    device = images.device
    dtype = images.dtype

    # Initialize tip
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)

    # Initial optimizer
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    nepoch = 200
    loss_train = []

    for epoch in tqdm(range(nepoch), disable=True):
        # Three-phase learning rate and weight decay schedule
        if epoch < 50:
            # Warm-up: moderate regularization, increasing LR
            lr_factor = 0.6 + (epoch / 50) * 0.4
            wd_factor = 1.0
        elif epoch < 150:
            # Main phase: full learning with maximum regularization
            lr_factor = 1.0
            wd_factor = 1.0
        else:
            # Cool-down: reduce regularization, decay LR for fine-tuning
            decay_progress = (epoch - 150) / 50
            lr_factor = 1.0 * (0.2 ** decay_progress)
            wd_factor = 0.1 * (1 + decay_progress)  # Reduce regularization

        # Update optimizer parameters
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * lr_factor
            param_group['weight_decay'] = 0.01 * wd_factor

        loss_tmp = 0.0
        for iframe in range(images.shape[0]):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            loss_tmp += loss.item()

        loss_train.append(loss_tmp)

    tip_estimate = tip.detach()
    return tip_estimate, loss_train
