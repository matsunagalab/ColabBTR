"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import torch
import torch.optim as optim
from colabbtr.morphology import (
    idilation, ierosion, translate_tip_mean
)


def laplacian_smoothing(tip, weight=0.01):
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    return weight * (torch.mean(lap_x) + torch.mean(lap_y))


def estimate_high_freq_energy(images):
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def estimate_variance_ratio(images):
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


def freq_to_tip(freq_real, freq_imag, tip_size, K):
    """Convert low-frequency FFT coefficients to spatial tip.

    Only the K×K low-frequency block is optimized; all higher
    frequencies are zero. Inverse FFT gives a smooth tip.
    """
    H, W = tip_size
    # Build full Hermitian-symmetric frequency tensor
    # Use rfft layout: only positive frequencies in last dim
    Wr = W // 2 + 1
    full_freq = torch.zeros(H, Wr, dtype=torch.complex128, device=freq_real.device)
    full_freq[:K, :K] = torch.complex(freq_real, freq_imag)
    # Inverse rfft → real spatial tip
    tip = torch.fft.irfft2(full_freq, s=(H, W))
    # Subtract max so apex = 0 (tip <= 0)
    tip = tip - tip.max()
    return tip


def reconstruct_tip(images, tip_size, **kwargs):
    """BTR with frequency-domain tip parameterization.

    Tip is parameterized by K×K low-frequency FFT coefficients (~32 params)
    instead of full pixel grid (225 params). The spatial tip is generated
    via inverse FFT, automatically smooth.

    Stage 1: Frequency-domain optimization (low-dim, well-conditioned)
    Stage 2: Spatial-domain refinement (full 225 params, fine details)
    Stage 3: Hard-frame refinement
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)

    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    if is_high_gaussian:
        images = torch.clamp(images, min=0.0)

    H, W = tip_size
    K = 5  # 5x5 low-frequency block = ~50 parameters
    loss_train = []

    # ── Stage 1: Frequency-domain optimization ──
    freq_real = torch.zeros(K, K, dtype=torch.float64, requires_grad=True, device=device)
    freq_imag = torch.zeros(K, K, dtype=torch.float64, requires_grad=True, device=device)
    optimizer = optim.AdamW([freq_real, freq_imag], lr=0.1, weight_decay=0.001)

    nepoch_freq = 80 if not is_high_gaussian else 120

    for epoch in range(nepoch_freq):
        progress = epoch / nepoch_freq
        if progress < 0.3:
            lr_factor = 0.6 + (progress / 0.3) * 0.4
        elif progress < 0.85:
            lr_factor = 1.0
        else:
            d = (progress - 0.85) / 0.15
            lr_factor = 0.1 ** d

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            tip = freq_to_tip(freq_real, freq_imag, tip_size, K)
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + depth_alpha * torch.mean(tip)
            loss.backward()
            optimizer.step()
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 2: Convert to spatial tip and refine ──
    with torch.no_grad():
        tip_init = freq_to_tip(freq_real, freq_imag, tip_size, K)
        tip_init = torch.clamp(tip_init, max=0.0)
        tip_init = translate_tip_mean(tip_init)

    tip = tip_init.clone().requires_grad_(True)
    optimizer = optim.AdamW([tip], lr=0.05, weight_decay=0.01)

    nepoch_spatial = 80 if not is_high_gaussian else 120
    for epoch in range(nepoch_spatial):
        progress = epoch / nepoch_spatial
        lr_factor = (1.0 - 0.7 * progress)
        smooth_weight = 0.005 if not is_high_gaussian else 0.01

        for pg in optimizer.param_groups:
            pg['lr'] = 0.05 * lr_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, smooth_weight)
            loss = loss + depth_alpha * torch.mean(tip)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # ── Stage 3: Hard-frame refinement ──
    frame_errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            error = torch.mean((recon - images[iframe]) ** 2)
            frame_errors.append(error.item())

    hard_count = max(1, (nframe + 1) // 2)
    hard_indices = torch.topk(
        torch.tensor(frame_errors), k=hard_count, largest=True
    ).indices.tolist()

    nepoch_s3 = 60 if not is_high_gaussian else 100
    for epoch in range(nepoch_s3):
        decay = epoch / nepoch_s3
        lr_factor = 0.1 ** decay
        smooth_weight = 0.01 + 0.01 * decay
        depth_weight = depth_alpha * (1.0 - 0.5 * decay)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.05 * lr_factor
            pg['weight_decay'] = 0.01 * max(0.02, 1.0 - decay)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer.zero_grad()
                recon = idilation(ierosion(images[iframe], tip), tip)
                loss = torch.mean((recon - images[iframe]) ** 2)
                loss = loss + laplacian_smoothing(tip, smooth_weight)
                loss = loss + depth_weight * torch.mean(tip)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)
                loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    return tip.detach(), loss_train
