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


class DenoiseNet(nn.Module):
    """Residual CNN denoiser: learns to clean images before erosion.

    The CNN predicts a RESIDUAL correction to the input image.
    denoised = image + scale * CNN(image)

    The scale parameter controls how much denoising is applied.
    The residual architecture ensures the CNN starts near identity
    (output ≈ 0 at initialization), avoiding disrupting early training.
    """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 3, padding=1)
        # Initialize last layer near zero for stable residual learning
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.to(dtype)

    def forward(self, image):
        """Input: (H, W) image. Output: (H, W) denoised image."""
        x = image.unsqueeze(0).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        residual = self.conv3(x)
        # Residual denoising: original + learned correction
        denoised = image.unsqueeze(0).unsqueeze(0) + residual
        return denoised.squeeze(0).squeeze(0)


def reconstruct_tip(images, tip_size, **kwargs):
    """Hybrid Denoise-BTR: learned denoising + standard morphological BTR.

    Architecture:
    - Stage 0: Standard BTR for initial tip estimate (60 epochs)
    - Stage 1: Joint optimization of DenoiseNet + tip.
      The CNN denoises images BEFORE erosion, making the erosion
      more accurate. Tip identifiability is preserved because
      erosion+dilation still use the same tip parameter.
    - Stage 2: Standard hard-frame refinement (with denoised images)

    Key insight: erosion(image, tip) = min over local window.
    A single noisy pixel can corrupt the min. The CNN smooths noise
    so the erosion sees cleaner data → better surface estimates →
    better tip reconstruction.
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
    opt_tip = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    loss_train = []

    for epoch in range(60):
        lr_f = 0.6 + (epoch / 60) * 0.4 if epoch < 20 else 1.0
        for pg in opt_tip.param_groups:
            pg['lr'] = 0.1 * lr_f

        loss_tmp = 0.0
        for iframe in range(nframe):
            opt_tip.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, 0.005) + depth_alpha * torch.mean(tip)
            loss.backward()
            opt_tip.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 1: Joint DenoiseNet + tip optimization ──
    denoiser = DenoiseNet(dtype=dtype).to(device)

    # Separate optimizers with different LR
    opt_joint = optim.AdamW([
        {'params': denoiser.parameters(), 'lr': 0.001},
        {'params': [tip], 'lr': 0.05, 'weight_decay': 0.01},
    ])

    if is_high_gaussian:
        nepoch_s1 = 140
    else:
        nepoch_s1 = 80

    for epoch in range(nepoch_s1):
        decay = epoch / nepoch_s1
        opt_joint.param_groups[0]['lr'] = 0.001 * (1.0 - 0.7 * decay)
        opt_joint.param_groups[1]['lr'] = 0.05 * (1.0 - 0.7 * decay)

        loss_tmp = 0.0
        for iframe in range(nframe):
            opt_joint.zero_grad()

            # Denoise then standard erosion+dilation
            denoised = denoiser(images[iframe])
            surface_est = ierosion(denoised, tip)
            image_recon = idilation(surface_est, tip)

            recon_loss = torch.mean((image_recon - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=0.005)
            depth_loss = depth_alpha * torch.mean(tip)

            # Regularize denoiser: penalize large corrections
            denoise_reg = 0.01 * torch.mean((denoised - images[iframe]) ** 2)

            loss = recon_loss + smooth_loss + depth_loss + denoise_reg
            loss.backward()
            opt_joint.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 2: Hard-frame refinement with denoised images ──
    # Use denoiser for final refinement too
    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            denoised = denoiser(images[iframe])
            recon = idilation(ierosion(denoised, tip), tip)
            error = torch.mean((recon - images[iframe]) ** 2)
            frame_errors.append(error.item())

    hard_count = max(1, (nframe + 1) // 2)
    hard_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_count, largest=True
    ).indices.tolist()

    # Freeze denoiser, fine-tune tip only
    for param in denoiser.parameters():
        param.requires_grad = False

    opt_tip2 = optim.AdamW([tip], lr=0.1, weight_decay=0.01)
    nepoch_s2 = 60 if not is_high_gaussian else 80

    for epoch in range(nepoch_s2):
        decay_progress = epoch / nepoch_s2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in opt_tip2.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay_progress)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                opt_tip2.zero_grad()
                with torch.no_grad():
                    denoised = denoiser(images[iframe])
                recon = idilation(ierosion(denoised, tip), tip)
                recon_loss = torch.mean((recon - images[iframe]) ** 2)
                smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
                depth_loss = depth_weight * torch.mean(tip)
                loss = recon_loss + smooth_loss + depth_loss
                loss.backward()
                opt_tip2.step()

                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)

                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
