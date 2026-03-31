"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from tqdm.auto import tqdm
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def _smooth_tip_loss(tip):
    """Compute smoothness regularization: penalize sharp gradients."""
    # Laplacian: second derivatives measure curvature
    # For a smooth conical tip, penalize excessive local variation
    grad_x = tip[:-1, :] - tip[1:, :]  # Vertical gradient
    grad_y = tip[:, :-1] - tip[:, 1:]  # Horizontal gradient

    # Second derivatives (curvature)
    grad2_x = grad_x[:-1, :] - grad_x[1:, :]
    grad2_y = grad_y[:, :-1] - grad_y[:, 1:]

    smoothness = torch.mean(grad2_x ** 2) + torch.mean(grad2_y ** 2)
    return smoothness


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with learning rate scheduling and smoothness constraints.

    Two-stage optimization with adaptive learning rates and explicit
    tip smoothness regularization for physically plausible solutions.

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    device = images.device
    dtype = images.dtype

    # Initialize tip with small random perturbation
    # This breaks symmetry while starting from a reasonable baseline
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    tip.data = tip.data + torch.randn_like(tip) * 0.001

    loss_train = []

    # Stage 1: Aggressive optimization (higher lr, higher smoothness weight)
    # This finds the general structure quickly
    print("Stage 1: Aggressive phase (100 epochs)")
    optimizer1 = optim.AdamW([tip], lr=0.15, weight_decay=0.005)
    nepoch1 = 100
    smooth_weight1 = 0.0001

    for epoch in tqdm(range(nepoch1), disable=True, desc="Stage 1"):
        loss_tmp = 0.0
        for iframe in range(images.shape[0]):
            optimizer1.zero_grad()

            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            smooth_loss = _smooth_tip_loss(tip)

            # Combined loss: reconstruction + smoothness
            total_loss = recon_loss + smooth_weight1 * smooth_loss
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([tip], max_norm=0.5)
            optimizer1.step()

            # Apply hard constraints after step
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += recon_loss.item()

        loss_train.append(loss_tmp)

    # Stage 2: Refinement with annealing schedule
    # Gradually reduce learning rate and smoothness weight
    print("Stage 2: Refinement phase (100 epochs with annealing)")
    nepoch2 = 100
    smooth_weight2_start = 0.00005
    smooth_weight2_end = 0.000001

    for epoch in tqdm(range(nepoch2), disable=True, desc="Stage 2"):
        # Cosine annealing for learning rate
        progress = epoch / nepoch2
        lr_current = 0.1 * (1 + torch.cos(torch.tensor(3.14159 * progress))) / 2

        # Linear decay for smoothness weight
        smooth_weight = smooth_weight2_start + (smooth_weight2_end - smooth_weight2_start) * progress

        optimizer2 = optim.AdamW([tip], lr=float(lr_current), weight_decay=0.01)

        loss_tmp = 0.0
        for iframe in range(images.shape[0]):
            optimizer2.zero_grad()

            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe, :, :]) ** 2)
            smooth_loss = _smooth_tip_loss(tip)

            total_loss = recon_loss + smooth_weight * smooth_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([tip], max_norm=0.5)
            optimizer2.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += recon_loss.item()

        loss_train.append(loss_tmp)

    tip_estimate = tip.detach()
    return tip_estimate, loss_train
