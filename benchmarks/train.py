"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.nn as nn
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


def estimate_variance_ratio(images):
    """Bright/dark variance ratio to distinguish noise types."""
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


class SurfaceNet(nn.Module):
    """Small CNN that learns to estimate surfaces from AFM images.

    Replaces the morphological erosion (which takes a hard min over a
    local window and is noise-sensitive) with a learned inverse that
    can implicitly denoise while inverting the dilation forward model.

    The CNN is trained jointly with the tip for each input dataset.
    """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        # Match input dtype
        self.to(dtype)

    def forward(self, image):
        """Input: (H, W) image. Output: (H, W) estimated surface."""
        x = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Surface should be non-negative
        x = F.relu(x)
        return x.squeeze(0).squeeze(0)  # (H, W)


def reconstruct_tip(images, tip_size, **kwargs):
    """Hybrid CNN-BTR: learned surface estimation + morphological dilation.

    Architecture:
    - Stage 0: Standard erosion-based BTR to get initial tip estimate
    - Stage 1: CNN replaces erosion. The CNN learns surface estimation
      that is robust to noise. Dilation (forward model) is unchanged.
      CNN and tip are jointly optimized.
    - Stage 2: Hard-frame refinement with standard erosion (fine-tuning)

    The CNN acts as a learned denoising inverse operator.
    Standard erosion: surface = min_over_window(image - tip) → noise-sensitive
    CNN: surface = f_theta(image) → can learn noise-robust inverse
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Noise detection
    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)

    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    if is_high_gaussian:
        images = torch.clamp(images, min=0.0)

    # ── Stage 0: Standard BTR for initial tip (60 epochs) ──
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    for epoch in range(60):
        lr_factor = 0.6 + (epoch / 60) * 0.4 if epoch < 20 else 1.0
        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.005)
            depth_loss = depth_alpha * torch.mean(tip)
            loss = recon_loss + smooth_loss + depth_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 1: CNN-based surface estimation + tip optimization ──
    surface_net = SurfaceNet(dtype=dtype).to(device)

    # Joint optimizer: CNN weights + tip
    optimizer = optim.AdamW(
        [{'params': surface_net.parameters(), 'lr': 0.001},
         {'params': [tip], 'lr': 0.05, 'weight_decay': 0.01}],
    )

    nepoch_cnn = 100 if not is_high_gaussian else 150
    for epoch in range(nepoch_cnn):
        decay = epoch / nepoch_cnn
        lr_scale = 1.0 - 0.5 * decay  # gentle LR decay

        for pg in optimizer.param_groups:
            pg['lr'] = pg['lr'] * lr_scale / (lr_scale + 1e-8) if epoch > 0 else pg['lr']

        # Reset LR each epoch (simpler than tracking)
        optimizer.param_groups[0]['lr'] = 0.001 * (1.0 - 0.5 * decay)
        optimizer.param_groups[1]['lr'] = 0.05 * (1.0 - 0.5 * decay)

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()

            # CNN predicts surface (replaces erosion)
            surface_est = surface_net(images[iframe])

            # Dilation with current tip (physical forward model)
            image_recon = idilation(surface_est, tip)

            recon_loss = torch.mean((image_recon - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.005)
            depth_loss = depth_alpha * torch.mean(tip)

            loss = recon_loss + smooth_loss + depth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 2: Standard erosion refinement (fine-tune tip) ──
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=0.01)

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

    nepoch_s2 = 60 if not is_high_gaussian else 80
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

    return tip.detach(), loss_train
