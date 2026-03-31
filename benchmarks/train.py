"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from tqdm.auto import tqdm
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with adaptive learning rate schedule and optimized weight decay.

    Progressive learning rate schedule: start with moderate lr, gradually
    increase during early epochs for faster convergence, then anneal to fine-tune.
    Combines benefits of warm-start and refinement phases.

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    device = images.device
    dtype = images.dtype

    # Initialize tip
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)

    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

    # Learning rate schedule with scheduler
    # Warm-up then decay pattern
    nepoch = 200
    loss_train = []

    for epoch in tqdm(range(nepoch), disable=True):
        # Adaptive learning rate schedule
        # Warm-up: first 30 epochs, increase learning rate
        # Main: epochs 30-150, stable or slight decay
        # Cool-down: last 50 epochs, exponential decay to fine-tune

        if epoch < 30:
            # Warm-up: linearly increase from 0.05 to 0.15
            lr_factor = 0.5 + (epoch / 30) * 0.5
        elif epoch < 150:
            # Stable phase
            lr_factor = 1.0
        else:
            # Cool-down: exponential decay
            decay_progress = (epoch - 150) / 50
            lr_factor = 1.0 * (0.1 ** decay_progress)

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * lr_factor

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
