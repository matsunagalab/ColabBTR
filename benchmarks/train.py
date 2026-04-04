"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm
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


# ──────────────────────────────────────────────────────────────────────
# BO + GP-based 1SE for automatic lambda selection
# ──────────────────────────────────────────────────────────────────────

def _quick_btr(images, tip_size, weight_decay, depth_alpha, nepoch_s1, nepoch_s2):
    """Short BTR pipeline for lambda CV evaluation."""
    device, dtype, nframe = images.device, images.dtype, images.shape[0]
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    opt = optim.AdamW([tip], lr=0.1, weight_decay=weight_decay)

    for epoch in range(nepoch_s1):
        e40 = int(nepoch_s1 * 40 / 140)
        e120 = int(nepoch_s1 * 120 / 140)
        ecd = nepoch_s1 - e120
        if epoch < e40:
            lr_f = 0.6 + (epoch / e40) * 0.4; wdf = 1.0; sw = 0.001
        elif epoch < e120:
            lr_f = 1.0; wdf = 1.0; sw = 0.005
        else:
            d = (epoch - e120) / max(1, ecd)
            lr_f = 0.1 ** d; wdf = max(0.05, 1.0 - d * 0.95); sw = 0.01
        for pg in opt.param_groups:
            pg['lr'] = 0.1 * lr_f; pg['weight_decay'] = weight_decay * wdf
        for iframe in range(nframe):
            opt.zero_grad()
            recon = idilation(ierosion(images[iframe], tip), tip)
            loss = torch.mean((recon - images[iframe]) ** 2)
            loss = loss + laplacian_smoothing(tip, sw) + depth_alpha * torch.mean(tip)
            loss.backward(); opt.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

    errors = []
    with torch.no_grad():
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            errors.append(torch.mean((recon - images[iframe]) ** 2).item())
    hc = max(1, (nframe + 1) // 2)
    hi = torch.topk(torch.tensor(errors), k=hc, largest=True).indices.tolist()

    for epoch in range(nepoch_s2):
        d = epoch / nepoch_s2
        lr_f = 0.1 ** d; sw = 0.01 + 0.01 * d
        dw = depth_alpha * (1.0 - 0.5 * d)
        for pg in opt.param_groups:
            pg['lr'] = 0.1 * lr_f
            pg['weight_decay'] = weight_decay * max(0.02, 1.0 - d)
        for _ in range(3):
            for iframe in hi:
                opt.zero_grad()
                recon = idilation(ierosion(images[iframe], tip), tip)
                loss = torch.mean((recon - images[iframe]) ** 2)
                loss = loss + laplacian_smoothing(tip, sw) + dw * torch.mean(tip)
                loss.backward(); opt.step()
                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)
    return tip.detach()


def _cv_loss_2fold(images, tip_size, lam, depth_alpha, nepoch_s1, nepoch_s2):
    """2-fold CV, returns per-fold losses."""
    nframe = images.shape[0]
    half = nframe // 2
    fold_losses = []
    for fold in range(2):
        train = images[:half] if fold == 0 else images[half:]
        val = images[half:] if fold == 0 else images[:half]
        tip = _quick_btr(train, tip_size, lam, depth_alpha, nepoch_s1, nepoch_s2)
        val_loss = 0.0
        with torch.no_grad():
            for i in range(val.shape[0]):
                recon = idilation(ierosion(val[i], tip), tip)
                val_loss += torch.mean((recon - val[i]) ** 2).item()
        fold_losses.append(val_loss)
    return fold_losses


def _gp_fit_predict(X_obs, y_obs, y_se_obs, X_pred, length_scale=1.0):
    """GP regression with heteroscedastic noise. Returns predicted mean and std."""
    X_obs, y_obs, y_se_obs = np.asarray(X_obs), np.asarray(y_obs), np.asarray(y_se_obs)
    X_pred = np.asarray(X_pred)
    signal_var = np.var(y_obs) + 1e-6
    noise_var = y_se_obs ** 2 + 1e-8

    def rbf(X1, X2):
        return signal_var * np.exp(-0.5 * (X1[:, None] - X2[None, :]) ** 2 / length_scale ** 2)

    K = rbf(X_obs, X_obs) + np.diag(noise_var)
    K_star = rbf(X_pred, X_obs)
    K_ss = rbf(X_pred, X_pred)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    mu = K_star @ alpha
    v = np.linalg.solve(L, K_star.T)
    var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.maximum(var, 1e-10))


def select_lambda_bo_1se(images, tip_size, depth_alpha,
                          nepoch_s1, nepoch_s2,
                          n_eval=6, log_min=-4.0, log_max=-0.5):
    """Bayesian Optimization with GP-based 1SE rule for lambda selection.

    1. Evaluate n_eval lambda candidates with 2-fold CV (BO-guided)
    2. Fit GP to (log_lambda, loss_mean, loss_se) observations
    3. Apply 1SE rule on the smooth GP-predicted loss curve

    Returns the 1SE-selected lambda.
    """
    # Initial 3 points spanning the range
    log_lambdas = list(np.linspace(log_min, log_max, 3))
    fold_losses_list = []

    for ll in log_lambdas:
        fl = _cv_loss_2fold(images, tip_size, 10 ** ll, depth_alpha,
                             nepoch_s1, nepoch_s2)
        fold_losses_list.append(fl)

    # BO iterations: select next point via Expected Improvement
    candidate_grid = np.linspace(log_min, log_max, 200)
    for _ in range(n_eval - 3):
        means = [np.mean(fl) for fl in fold_losses_list]
        ses = [np.std(fl) / np.sqrt(len(fl)) for fl in fold_losses_list]
        ls = (log_max - log_min) / 3.0
        mu_pred, std_pred = _gp_fit_predict(log_lambdas, means, ses,
                                             candidate_grid, length_scale=ls)
        # Expected Improvement (minimize)
        y_best = min(means)
        z = (y_best - mu_pred) / std_pred
        ei = (y_best - mu_pred) * norm.cdf(z) + std_pred * norm.pdf(z)

        for idx in np.argsort(-ei):
            c = candidate_grid[idx]
            if all(abs(c - ll) > 0.3 for ll in log_lambdas):
                break
        else:
            c = candidate_grid[np.argmax(ei)]

        fl = _cv_loss_2fold(images, tip_size, 10 ** c, depth_alpha,
                             nepoch_s1, nepoch_s2)
        log_lambdas.append(c)
        fold_losses_list.append(fl)

    # Fit final GP and apply 1SE rule on the smooth GP surface
    means = [np.mean(fl) for fl in fold_losses_list]
    ses = [np.std(fl) / np.sqrt(len(fl)) for fl in fold_losses_list]
    ls = (log_max - log_min) / 3.0
    dense_grid = np.linspace(log_min, log_max, 500)
    mu_pred, std_pred = _gp_fit_predict(log_lambdas, means, ses,
                                         dense_grid, length_scale=ls)

    # 1SE rule: largest lambda whose predicted loss ≤ min + GP_std_at_min
    imin = np.argmin(mu_pred)
    threshold = mu_pred[imin] + std_pred[imin]
    candidates = np.where(mu_pred <= threshold)[0]
    i_1se = int(np.max(candidates))

    return 10 ** dense_grid[i_1se]


# ──────────────────────────────────────────────────────────────────────
# Main reconstruction function
# ──────────────────────────────────────────────────────────────────────

def reconstruct_tip(images, tip_size, **kwargs):
    """Noise-adaptive BTR with BO GP-1SE automatic lambda selection.

    Architecture:
    1. Noise detection (HF energy + variance ratio)
    2. Lambda auto-selection via Bayesian Optimization with GP-based 1SE rule
       (6 evaluations × 2-fold CV using the improved pipeline as objective)
    3. Full BTR with selected lambda + noise-specific adaptations

    The GP-based 1SE rule selects the most regularized lambda whose
    predicted loss is within 1 GP-std of the minimum — automatically
    balancing fit quality and regularization strength.
    """
    device = images.device
    dtype = images.dtype
    nframe = images.shape[0]

    # Noise detection
    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)

    # Depth regularizer for clean data
    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    # Physical preprocessing for high Gaussian noise
    if is_high_gaussian:
        images = torch.clamp(images, min=0.0)

    # Noise-adaptive epoch counts
    if is_high_gaussian:
        nepoch_stage1 = 200
        nepoch_stage2 = 100
    else:
        nepoch_stage1 = 140
        nepoch_stage2 = 60

    # Auto-select lambda via BO + GP-1SE
    optimal_wd = select_lambda_bo_1se(
        images, tip_size, depth_alpha,
        nepoch_s1=nepoch_stage1, nepoch_s2=nepoch_stage2,
        n_eval=6, log_min=-4.0, log_max=-0.5,
    )

    # Full BTR with selected lambda on all frames
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=0.1, weight_decay=optimal_wd)
    loss_train = []

    # STAGE 1
    for epoch in range(nepoch_stage1):
        epoch_40 = int(nepoch_stage1 * 40 / 140)
        epoch_120 = int(nepoch_stage1 * 120 / 140)
        epoch_cooldown = nepoch_stage1 - epoch_120

        if epoch < epoch_40:
            lr_factor = 0.6 + (epoch / epoch_40) * 0.4
            wd_factor = 1.0
            smooth_weight = 0.001 if not is_high_gaussian else 0.002
        elif epoch < epoch_120:
            lr_factor = 1.0
            wd_factor = 1.0
            smooth_weight = 0.005 if not is_high_gaussian else 0.008
        else:
            decay_progress = (epoch - epoch_120) / max(1, epoch_cooldown)
            lr_factor = 0.1 ** decay_progress
            wd_factor = max(0.05, 1.0 - decay_progress * 0.95)
            smooth_weight = 0.01 if not is_high_gaussian else 0.016

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = optimal_wd * wd_factor

        loss_tmp = 0.0
        for iframe in range(nframe):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
            recon_loss = torch.mean((image_reconstructed - images[iframe]) ** 2)
            smooth_loss = laplacian_smoothing(tip, weight=smooth_weight)
            depth_loss = depth_alpha * torch.mean(tip)
            loss = recon_loss + smooth_loss + depth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)

            loss_tmp += loss.item()
        loss_train.append(loss_tmp)

    # Frame error evaluation
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

    # STAGE 2: Hard-frame refinement
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = 0.1 * lr_factor
            pg['weight_decay'] = optimal_wd * max(0.02, 1.0 - decay_progress)

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
