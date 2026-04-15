import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from tqdm.auto import tqdm
from scipy.stats import norm

def compute_xc_yc(tip):
    """
    Compute the center position of the tip
        Input: tip (tensor of size (tip_height, tip_width))
        Output: xc, yc (int)
    """
    tip_xsiz, tip_ysiz = tip.size()
    xc = round((tip_xsiz - 1) / 2)
    yc = round((tip_ysiz - 1) / 2)
    return xc, yc

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
@torch.jit.script
def fixed_padding(inputs, kernel_size, dilation, pad_value: float = 0.0):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg:int = int(pad_total // 2)
    pad_end:int = int(pad_total - pad_beg)
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end), value=pad_value)
    return padded_inputs

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
@torch.jit.script
def idilation(image, tip):
    """
    Compute the dilation of surface by tip
        Input: surface (tensor of size (surface_height, surface_width)
               tip (tensor of size (kernel_size, kernel_size)
        Output: r (tensor of size (image_height, image_width)
                where image_heigh is equal to surface_height
                      image_width is equal to surface_width
    """
    H, W = image.shape
    kernel_size, _ = tip.shape
    x = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x = fixed_padding(x, torch.tensor(kernel_size), dilation=torch.tensor(1), pad_value=float('-inf'))
    x = F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)  # (B, Cin*kH*kW, L), where L is the numbers of patches
    x = x.unsqueeze(1) # (B, 1, Cin*kH*kW, L)
    L = x.size(-1)

    weight = tip.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    weight = weight.view(1, -1) # (Cout, Cin*kH*kW)
    weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
    x = weight + x # (B, Cout, Cin*kH*kW, L)
    x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
    x = x.view(-1, 1, H, W)  # (B, Cout, H, W)
    return x.squeeze(0).squeeze(0)

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
@torch.jit.script
def ierosion(surface, tip):
    """
    Compute the erosion of image by tip
        Input: image (tensor of size (image_height, image_width)
               tip (tensor of size (kernel_size, kernel_size)
        Output: r (tensor of size (image_height, image_width)
    """
    kernel_size, _ = tip.shape
    H, W = surface.shape
    x = surface.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x = fixed_padding(x, torch.tensor(kernel_size), dilation=torch.tensor(1), pad_value=float('inf'))
    x = F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)  # (B, Cin*kH*kW, L), where L is the numbers of patches
    x = x.unsqueeze(1) # (B, 1, Cin*kH*kW, L)
    L = x.size(-1)

    weight = tip.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    weight = weight.view(1, -1) # (Cout, Cin*kH*kW)
    weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
    x = weight - x # (B, Cout, Cin*kH*kW, L)
    x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
    x = -1 * x
    x = x.view(-1, 1, H, W)  # (B, Cout, H, W)
    return x.squeeze(0).squeeze(0)

def idilation_old(surface, tip):
    """
    Compute the dilation of surface by tip
        Input: surface (tensor of size (surface_height, surface_width)
               tip (tensor of size (tip_height, tip_width)
        Output: r (tensor of size (image_height, image_width)
                where image_heigh is equal to surface_height
                      image_width is equal to surface_width
    """
    xc, yc = compute_xc_yc(tip)
    tip_xsiz, tip_ysiz = tip.size()

    r = torch.full_like(surface, -float('inf'))
    for px in range(-xc, tip_xsiz - xc):
        for py in range(-yc, tip_ysiz - yc):
            temp = torch.roll(surface, shifts=(-px, -py), dims=(0, 1))
            temp = temp + tip[xc + px, yc + py]
            r = torch.maximum(r, temp)
    return r

def ierosion_old(image, tip):
    """
    Compute the erosion of image by tip
        Input: image (tensor of size (image_height, image_width)
               tip (tensor of size (tip_height, tip_width)
        Output: r (tensor of size (image_height, image_width)
    """
    xc, yc = compute_xc_yc(tip)
    tip_xsiz, tip_ysiz = tip.size()

    r = torch.full_like(image, float('inf'))
    for px in range(-xc, tip_xsiz - xc):
        for py in range(-yc, tip_ysiz - yc):
            temp = torch.roll(image, shifts=(px, py), dims=(0, 1))
            temp = temp - tip[xc + px, yc + py]
            r = torch.minimum(r, temp)
    return r

def translate_tip_mean(P, cutoff=10**(-8)):
    """
    Translate the tip to the center of mass
        Input: P (tensor of size (tip_height, tip_width))
        Output: P_new (tensor of size (tip_height, tip_width)
    """
    tip_xsiz, tip_ysiz = P.size()
    xc, yc = compute_xc_yc(P)

    #p_max = torch.min(P)
    #P = P - p_max

    p_min = torch.min(P)
    weight = P - p_min

    id = weight < cutoff
    weight[id] = 0.0

    if torch.all(weight < 10**(-10)):
        weight.fill_(1.0)

    ix = torch.ones(tip_xsiz, tip_ysiz, dtype=P.dtype, device=P.device)
    iy = torch.ones(tip_xsiz, tip_ysiz, dtype=P.dtype, device=P.device)
    ix = torch.cumsum(ix, dim=0) - 1.0
    iy = torch.cumsum(iy, dim=1) - 1.0
    com_x = torch.sum(ix * weight / torch.sum(weight))
    com_y = torch.sum(iy * weight / torch.sum(weight))
    id_x = round(com_x.item())
    id_y = round(com_y.item())

    pxmin = max(- xc, - id_x)
    pymin = max(- yc, - id_y)
    pxmax = min(tip_xsiz - xc, tip_xsiz - id_x)
    pymax = min(tip_ysiz - yc, tip_ysiz - id_y)

    P_new = torch.full_like(P, p_min.item())
    P_new[(xc + pxmin):(xc + pxmax), (yc + pymin):(yc + pymax)] = P[(id_x + pxmin):(id_x + pxmax), (id_y + pymin):(id_y + pymax)]

    return P_new

def differentiable_btr(images, tip_size, nepoch=100, lr=0.1, weight_decay=0.0, is_tqdm=True):
    """
    Reconstruct tip shape from given AFM images by differentiable blind tip reconstruction (BTR)
        Input: images (tensor of size (nframe, image_height, image_width)
               tip_size (2d tuple)
               nepoch (int)
               lr (float) for AdamW
               weight_decay (float) for AdamW
        Output: tip_estimate (tensor of tip_size), loss_train (list)
    """
    # Initialize tip with zeros (match input dtype for MPS compatibility)
    device = images.device
    dtype = images.dtype
    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)

    # Optimization settings
    optimizer = optim.AdamW([tip], lr=lr, weight_decay=weight_decay)

    loss_train = []
    for _ in tqdm(range(nepoch), disable=not is_tqdm):
        loss_tmp = 0.0
        for iframe in range(images.shape[0]):
            optimizer.zero_grad()
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            loss = torch.mean((image_reconstructed - images[iframe, :, :])**2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tip.data = torch.clamp(tip, max=0.0)
                tip.data = translate_tip_mean(tip)
            image_reconstructed = idilation(ierosion(images[iframe, :, :], tip), tip)
            loss = torch.mean((image_reconstructed - images[iframe, :, :])**2)
            loss_tmp += loss.item()
        #if epoch % 1 == 0:
        #    print(f"Epoch: {epoch}, Loss: {loss_tmp}")
        loss_train.append(loss_tmp)

    tip_estimate = tip.detach()
    return tip_estimate, loss_train

def _select_lambda_1se(lambdas, loss_mean, loss_std):
    """
    Apply the one-standard-error rule with linear interpolation.
    Selects the largest lambda whose mean CV loss is within one standard error
    of the minimum, refined by linear interpolation between adjacent grid points.

        Input:
            lambdas (array-like) — lambda values (sorted ascending)
            loss_mean (array-like) — mean CV loss for each lambda
            loss_std (array-like) — std of CV loss for each lambda
        Output:
            lambda_opt (float)
    """
    lambdas = np.asarray(lambdas, dtype=float)
    loss_mean = np.asarray(loss_mean, dtype=float)
    loss_std = np.asarray(loss_std, dtype=float)
    n = len(lambdas)

    if n == 1:
        return float(lambdas[0])

    imin = int(np.argmin(loss_mean))
    threshold = loss_mean[imin] + loss_std[imin]

    candidates = np.where(loss_mean <= threshold)[0]
    imin_1se = int(np.max(candidates))

    # Edge case: 1SE index is the last lambda — no interpolation possible
    if imin_1se >= n - 1:
        return float(lambdas[-1])

    # Linear interpolation between imin_1se and imin_1se+1
    denom = loss_mean[imin_1se + 1] - loss_mean[imin_1se]
    if abs(denom) < 1e-30:
        return float(lambdas[imin_1se])

    frac = (threshold - loss_mean[imin_1se]) / denom
    return float(lambdas[imin_1se] + (lambdas[imin_1se + 1] - lambdas[imin_1se]) * frac)

def cross_validate_lambda(images, tip_size, lambda_min=1e-4, lambda_max=0.1, lambda_num=5,
                          n_folds=5, nepoch=100, lr=0.1, is_tqdm=True):
    """
    Find the optimal regularization parameter (weight_decay) for differentiable BTR
    using k-fold cross-validation with the one-standard-error rule.

        Input:
            images (tensor of size (nframe, image_height, image_width))
            tip_size (2d tuple) — tip dimensions (tip_height, tip_width)
            lambda_min (float) — minimum lambda
            lambda_max (float) — maximum lambda
            lambda_num (int) — number of lambda values (log-spaced)
            n_folds (int) — number of CV folds (default 5)
            nepoch (int) — training epochs per fold
            lr (float) — learning rate for AdamW
            is_tqdm (bool) — show progress bar for lambda loop
        Output:
            optimal_lambda (float) — selected lambda by one-standard-error rule
            cv_result (dict) — keys: lambdas, loss_mean, loss_std, lambda_min_idx, lambda_1se_idx
    """
    n_samples = images.shape[0]
    if n_samples < n_folds:
        raise ValueError(f"Need at least {n_folds} images for {n_folds}-fold CV, got {n_samples}")
    if lambda_num < 1:
        raise ValueError("lambda_num must be >= 1")
    if lambda_min <= 0 or lambda_max <= 0:
        raise ValueError("lambda_min and lambda_max must be positive")
    if lambda_min >= lambda_max:
        raise ValueError("lambda_min must be less than lambda_max")

    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), lambda_num)
    fold_size = n_samples // n_folds

    loss_means = []
    loss_stds = []
    for lam in tqdm(lambdas, desc='Evaluating lambdas', disable=not is_tqdm):
        cv_losses = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

            train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
            train_data = images[train_indices]
            val_data = images[val_start:val_end]

            tip, _ = differentiable_btr(train_data, tip_size,
                                        nepoch=nepoch, lr=lr, weight_decay=lam, is_tqdm=False)

            with torch.no_grad():
                val_loss = 0.0
                for i in range(val_data.shape[0]):
                    image = val_data[i, :, :]
                    image_reconstructed = idilation(ierosion(image, tip), tip)
                    val_loss += torch.mean((image_reconstructed - image) ** 2).item()
            cv_losses.append(val_loss)

        loss_means.append(float(np.mean(cv_losses)))
        loss_stds.append(float(np.std(cv_losses)))

    loss_mean_arr = np.array(loss_means)
    loss_std_arr = np.array(loss_stds)

    imin = int(np.argmin(loss_mean_arr))
    threshold = loss_mean_arr[imin] + loss_std_arr[imin]
    candidates = np.where(loss_mean_arr <= threshold)[0]
    imin_1se = int(np.max(candidates))

    optimal_lambda = _select_lambda_1se(lambdas, loss_means, loss_stds)

    cv_result = {
        'lambdas': lambdas,
        'loss_mean': loss_means,
        'loss_std': loss_stds,
        'lambda_min_idx': imin,
        'lambda_1se_idx': imin_1se,
    }
    return optimal_lambda, cv_result


# ──────────────────────────────────────────────────────────────────────
# Improved BTR with BO GP-1SE automatic lambda selection
# ──────────────────────────────────────────────────────────────────────

def laplacian_smoothing(tip, weight=0.01):
    """Laplacian smoothing penalty: encourage smooth, physically plausible tips.

    Penalizes local roughness via sum of squared second differences.

        Input: tip (tensor of size (tip_height, tip_width))
               weight (float) — regularization strength
        Output: scalar penalty (tensor)
    """
    lap_x = (tip[2:, 1:-1] - 2 * tip[1:-1, 1:-1] + tip[0:-2, 1:-1]) ** 2
    lap_y = (tip[1:-1, 2:] - 2 * tip[1:-1, 1:-1] + tip[1:-1, 0:-2]) ** 2
    roughness = torch.mean(lap_x) + torch.mean(lap_y)
    return weight * roughness


def estimate_high_freq_energy(images):
    """Measure high-frequency energy in images as a noise proxy.

    Computes the standard deviation of the difference between each pixel
    and its 4-neighbor average. Higher values indicate more noise.

        Input: images (tensor of size (nframe, H, W))
        Output: float — mean HF energy across frames
    """
    energies = []
    for i in range(images.shape[0]):
        img = images[i]
        neighbors = (img[2:, 1:-1] + img[:-2, 1:-1] + img[1:-1, 2:] + img[1:-1, :-2]) / 4
        diff = img[1:-1, 1:-1] - neighbors
        energies.append(diff.std().item())
    return sum(energies) / len(energies)


def estimate_variance_ratio(images):
    """Bright/dark variance ratio to distinguish noise types.

    Gaussian noise: ratio ~1.5-6.5 (constant variance across intensities).
    Poisson/none: ratio >> 100 (signal-dependent or zero variance in dark regions).

        Input: images (tensor of size (nframe, H, W))
        Output: float — variance ratio
    """
    pixel_var = torch.var(images, dim=0)
    pixel_mean = torch.mean(images, dim=0)
    median_val = pixel_mean.median()
    bright = pixel_var[pixel_mean > median_val].mean().item()
    dark = pixel_var[pixel_mean <= median_val].mean().item()
    return bright / (dark + 1e-8)


def _adaptive_btr(images, tip_size, weight_decay, depth_alpha, nepoch_s1, nepoch_s2):
    """Two-stage BTR with Laplacian smoothing for lambda CV evaluation.

    Internal function used by _cv_loss_2fold during BO lambda search.
    Same architecture as improved_btr but without BO (lambda is given).
    """
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

    # Pixel importance from cross-frame mean residual
    with torch.no_grad():
        residuals = []
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            residuals.append(recon - images[iframe])
        mean_res = torch.stack(residuals).mean(dim=0)
        pix_imp = mean_res.abs()
        pix_imp = pix_imp / (pix_imp.mean() + 1e-8)
        pix_imp = torch.clamp(pix_imp, min=0.2, max=3.0)

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
                loss = torch.mean(pix_imp * (recon - images[iframe]) ** 2)
                loss = loss + laplacian_smoothing(tip, sw) + dw * torch.mean(tip)
                loss.backward(); opt.step()
                with torch.no_grad():
                    tip.data = torch.clamp(tip, max=0.0)
                    tip.data = translate_tip_mean(tip)
    return tip.detach()


def _cv_loss_2fold(images, tip_size, lam, depth_alpha, nepoch_s1, nepoch_s2):
    """2-fold CV loss for a given lambda. Returns list of per-fold losses."""
    nframe = images.shape[0]
    half = nframe // 2
    fold_losses = []
    for fold in range(2):
        train = images[:half] if fold == 0 else images[half:]
        val = images[half:] if fold == 0 else images[:half]
        tip = _adaptive_btr(train, tip_size, lam, depth_alpha, nepoch_s1, nepoch_s2)
        val_loss = 0.0
        with torch.no_grad():
            for i in range(val.shape[0]):
                recon = idilation(ierosion(val[i], tip), tip)
                val_loss += torch.mean((recon - val[i]) ** 2).item()
        fold_losses.append(val_loss)
    return fold_losses


def _gp_fit_predict(X_obs, y_obs, y_se_obs, X_pred, length_scale=1.0):
    """GP regression with heteroscedastic noise.

    Fits a Gaussian Process with RBF kernel to observed data with
    per-observation noise (from CV standard error), and predicts
    mean and std at new points.

        Input:
            X_obs — observed log-lambda values
            y_obs — observed mean CV losses
            y_se_obs — observed standard errors
            X_pred — prediction points
            length_scale — RBF kernel length scale
        Output:
            mu (ndarray) — predicted mean
            std (ndarray) — predicted standard deviation
    """
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


def select_lambda_bo_gp1se(images, tip_size, n_eval=6,
                            log_lambda_min=-4.0, log_lambda_max=-0.5,
                            is_tqdm=True):
    """Select optimal regularization lambda via Bayesian Optimization
    with GP-based one-standard-error rule.

    Uses the improved BTR pipeline (with Laplacian smoothing and depth
    regularizer) as the CV objective. Noise level is auto-detected to
    set depth_alpha and epoch counts.

    Steps:
    1. Evaluate n_eval lambda candidates with 2-fold CV (BO-guided)
    2. Fit GP to (log_lambda, loss_mean, loss_se) observations
    3. Apply 1SE rule on the smooth GP-predicted loss curve

    The 1SE rule selects the most regularized (largest) lambda whose
    predicted loss is within 1 GP-std of the predicted minimum.

        Input:
            images (tensor of size (nframe, H, W))
            tip_size (2d tuple) — tip dimensions
            n_eval (int) — total number of lambda evaluations (default 6)
            log_lambda_min (float) — log10 of minimum lambda to search
            log_lambda_max (float) — log10 of maximum lambda to search
            is_tqdm (bool) — show progress bar
        Output:
            optimal_lambda (float) — 1SE-selected lambda
            bo_result (dict) — visualization data:
                'log_lambdas_eval': evaluated log10(lambda) values
                'loss_mean_eval': mean CV loss at each point
                'loss_se_eval': standard error at each point
                'gp_log_lambdas': dense grid for GP predictions
                'gp_mu': GP predicted mean
                'gp_std': GP predicted std
                'gp_min_idx': index of GP minimum
                'gp_1se_idx': index of 1SE selection
    """
    # Auto-detect noise for depth_alpha and epoch counts
    hf_energy = estimate_high_freq_energy(images)
    var_ratio = estimate_variance_ratio(images)
    is_high_gaussian = (hf_energy > 0.5) and (var_ratio < 100)
    depth_alpha = 0.005 if hf_energy < 0.2 else 0.0

    if is_high_gaussian:
        nepoch_s1, nepoch_s2 = 200, 100
    else:
        nepoch_s1, nepoch_s2 = 140, 60

    # Physical preprocessing for high Gaussian noise
    if is_high_gaussian:
        images = torch.clamp(images, min=0.0)

    # Initial 3 points spanning the range
    log_lambdas = list(np.linspace(log_lambda_min, log_lambda_max, 3))
    fold_losses_list = []

    pbar = tqdm(total=n_eval, desc='BO lambda search', disable=not is_tqdm)
    for ll in log_lambdas:
        fl = _cv_loss_2fold(images, tip_size, 10 ** ll, depth_alpha,
                             nepoch_s1, nepoch_s2)
        fold_losses_list.append(fl)
        pbar.update(1)

    # BO iterations: select next point via Expected Improvement
    candidate_grid = np.linspace(log_lambda_min, log_lambda_max, 200)
    for _ in range(n_eval - 3):
        means = [np.mean(fl) for fl in fold_losses_list]
        ses = [np.std(fl) / np.sqrt(len(fl)) for fl in fold_losses_list]
        ls = (log_lambda_max - log_lambda_min) / 3.0
        mu_pred, std_pred = _gp_fit_predict(log_lambdas, means, ses,
                                             candidate_grid, length_scale=ls)
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
        pbar.update(1)
    pbar.close()

    # Fit final GP and apply 1SE rule on the smooth GP surface
    means = [np.mean(fl) for fl in fold_losses_list]
    ses = [np.std(fl) / np.sqrt(len(fl)) for fl in fold_losses_list]
    ls = (log_lambda_max - log_lambda_min) / 3.0
    dense_grid = np.linspace(log_lambda_min, log_lambda_max, 500)
    mu_pred, std_pred = _gp_fit_predict(log_lambdas, means, ses,
                                         dense_grid, length_scale=ls)

    imin = int(np.argmin(mu_pred))
    threshold = mu_pred[imin] + std_pred[imin]
    candidates = np.where(mu_pred <= threshold)[0]
    i_1se = int(np.max(candidates))

    optimal_lambda = float(10 ** dense_grid[i_1se])

    bo_result = {
        'log_lambdas_eval': list(log_lambdas),
        'loss_mean_eval': means,
        'loss_se_eval': ses,
        'gp_log_lambdas': dense_grid,
        'gp_mu': mu_pred,
        'gp_std': std_pred,
        'gp_min_idx': imin,
        'gp_1se_idx': i_1se,
    }
    return optimal_lambda, bo_result


def improved_btr(images, tip_size, nepoch=None, lr=0.1,
                 weight_decay=None, is_tqdm=True):
    """Noise-adaptive BTR with Laplacian smoothing, depth regularization,
    and two-stage hard-frame refinement.

    Improvements over differentiable_btr:
    - Laplacian smoothing for physically plausible tip shapes
    - Depth regularizer to counteract the opening bluntness bias (clean data)
    - Noise-adaptive scheduling: extended epochs and clamping for high noise
    - Hard-frame refinement stage (Stage 2) with curriculum weighting

    If weight_decay is None, uses BO GP-1SE to auto-select lambda.
    If weight_decay is provided (e.g., from select_lambda_bo_gp1se), uses it directly.

    If nepoch is None, epoch counts are set automatically based on noise level
    (140+60 for clean/moderate, 200+100 for high Gaussian noise).
    If nepoch is provided, it is split ~70/30 between Stage 1 and Stage 2.

        Input:
            images (tensor of size (nframe, H, W))
            tip_size (2d tuple) — tip dimensions (tip_height, tip_width)
            nepoch (int or None) — total epochs (auto if None)
            lr (float) — base learning rate (default 0.1)
            weight_decay (float or None) — regularization lambda (auto if None)
            is_tqdm (bool) — show progress bar
        Output:
            tip_estimate (tensor of tip_size)
            loss_train (list of floats, one per epoch)
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

    # Epoch counts
    if nepoch is not None:
        nepoch_stage1 = int(nepoch * 0.7)
        nepoch_stage2 = nepoch - nepoch_stage1
    elif is_high_gaussian:
        nepoch_stage1, nepoch_stage2 = 200, 100
    else:
        nepoch_stage1, nepoch_stage2 = 140, 60

    # Auto-select lambda if not provided
    if weight_decay is None:
        weight_decay, _ = select_lambda_bo_gp1se(
            images, tip_size, is_tqdm=is_tqdm)

    tip = torch.zeros(tip_size, dtype=dtype, requires_grad=True, device=device)
    optimizer = optim.AdamW([tip], lr=lr, weight_decay=weight_decay)
    loss_train = []

    total_epochs = nepoch_stage1 + nepoch_stage2
    pbar = tqdm(total=total_epochs, desc='improved_btr', disable=not is_tqdm)

    # STAGE 1: All frames
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
            pg['lr'] = lr * lr_factor
            pg['weight_decay'] = weight_decay * wd_factor

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
        pbar.update(1)

    # Hard-frame selection
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

    # Cross-frame mean residual as pixel-wise importance for Stage 2
    # Mean of 20 frames cancels zero-mean noise → pure structural residual.
    # |mean_residual| is large where tip consistently removes features.
    with torch.no_grad():
        residuals = []
        for iframe in range(nframe):
            recon = idilation(ierosion(images[iframe], tip), tip)
            residual = recon - images[iframe]
            residuals.append(residual)
        mean_residual = torch.stack(residuals).mean(dim=0)
        pixel_importance = mean_residual.abs()
        pixel_importance = pixel_importance / (pixel_importance.mean() + 1e-8)
        pixel_importance = torch.clamp(pixel_importance, min=0.2, max=3.0)

    # STAGE 2: Hard-frame refinement with pixel importance weighting
    for epoch in range(nepoch_stage2):
        decay_progress = epoch / nepoch_stage2
        lr_factor = 0.1 ** decay_progress
        smooth_weight = 0.01 + 0.01 * decay_progress
        depth_weight = depth_alpha * (1.0 - 0.5 * decay_progress)

        for pg in optimizer.param_groups:
            pg['lr'] = lr * lr_factor
            pg['weight_decay'] = weight_decay * max(0.02, 1.0 - decay_progress)

        loss_tmp = 0.0
        for _ in range(3):
            for iframe in hard_indices:
                optimizer.zero_grad()
                image_reconstructed = idilation(ierosion(images[iframe], tip), tip)
                recon_loss = torch.mean(pixel_importance * (image_reconstructed - images[iframe]) ** 2)
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
        pbar.update(1)

    pbar.close()
    return tip.detach(), loss_train


@torch.jit.script
def surfing(xyz, radius, config:dict[str, float], shift_z: bool = True):
    """
    Compute the maximum height (z-value) of molecular surface at grid points on AFM stage (where z=0)
        Input: xyz (tensor of size (*, N, 3))
                radius (tensor of size (N,))
                config (dict)
                shift_z (bool) — if True (default), shift z so that min(z) = 0.
                    Set to False for MD simulation data where molecules are already on the AFM stage.
        Output: z_stage (tensor of size (*, len(y_stage), len(x_stage))
    """
    xyz = xyz.clone()
    if shift_z:
        # Place molecule on the AFM stage: shift z so that min(z) = 0
        z_min = xyz[..., 2].min(dim=-1, keepdim=True)[0]  # (*, 1)
        xyz[..., 2] = xyz[..., 2] - z_min

    radius2 = radius**2
    x_stage = torch.arange(config["min_x"], config["max_x"], config["resolution_x"], dtype=xyz.dtype, device=xyz.device) + 0.5*config["resolution_x"] #(W,)
    y_stage = torch.arange(config["min_y"], config["max_y"], config["resolution_y"], dtype=xyz.dtype, device=xyz.device) + 0.5*config["resolution_y"] #(H,)

    dx = xyz[...,0,None] - x_stage #(*,N,W)
    dx2 = dx**2 #(*,N,W)
    dy = xyz[...,1,None] - y_stage #(*,N,H)
    dy2 = dy**2 #(*,N,H)
    r2 = dx2.unsqueeze(-2) + dy2[...,None] #(*,N,H,W)
    index_within_radius = r2 < radius2[...,None,None] #(*,N,H,W)
    diff = radius2[...,None,None] - r2
    diff = torch.where(index_within_radius, diff, 1) #(*,N,H,W)
    temp = torch.where(index_within_radius, xyz[...,2,None,None] + torch.sqrt(diff), -torch.inf) #(*,N,H,W)
    temp_max = temp.max(dim=-3)[0] #(*,H,W)
    z_stage = torch.where(index_within_radius.any(dim=-3), temp_max, torch.zeros_like(temp_max, dtype=xyz.dtype, device=xyz.device)) #(H,W)
    return z_stage.flip([-2])

def surfing_old(xyz, radius, config, shift_z=True):
    """
    Compute the maximum height (z-value) of molecular surface at grid points on AFM stage (where z=0)
        Input: xyz (tensor of size (N, 3))
                radius (tensor of size (N,))
                config (dict)
                shift_z (bool) — if True (default), shift z so that min(z) = 0.
                    Set to False for MD simulation data where molecules are already on the AFM stage.
        Output: z_stage (tensor of size (len(y_stage), len(x_stage))
    """
    device = xyz.device
    xyz = xyz.clone()
    if shift_z:
        # Place molecule on the AFM stage: shift z so that min(z) = 0
        xyz[:, 2] = xyz[:, 2] - xyz[:, 2].min()
    radius2 = radius**2
    x_stage = torch.arange(config["min_x"], config["max_x"], config["resolution_x"]) + 0.5*config["resolution_x"]
    y_stage = torch.arange(config["min_y"], config["max_y"], config["resolution_y"]) + 0.5*config["resolution_y"]
    #z_stage = torch.full((len(y_stage), len(x_stage)), xyz[:, 2].min())
    z_stage = torch.full((len(y_stage), len(x_stage)), 0.0, dtype=torch.float32, device=device)
    for i in range(len(x_stage)):
        x = x_stage[i]
        dx = xyz[:, 0] - x
        dx2 = dx**2
        for j in range(len(y_stage)):
            y = y_stage[j]
            dy = xyz[:, 1] - y
            dy2 = dy**2
            r2 = dx2 + dy2
            index_within_radius = r2 < radius2
            #print(r2[:3], radius[:3])
            if any(index_within_radius):
                z_stage[-j-1, i] = torch.max(xyz[index_within_radius, 2] + torch.sqrt(radius2[index_within_radius] - r2[index_within_radius]))
    return z_stage

def afmize(xyz, tip, radius, config):
    """
    Compute AFM image from xyz coordinates and atomic radii
        Input: xyz (tensor of size (N, 3))
                tip (tensor of size (tip_height, tip_width))
                radius (tensor of size (N,))
                config (dict)
        Output: image (tensor of size (len(y_stage), len(x_stage))
    """
    surface = surfing(xyz, radius, config)
    image = idilation(surface, tip)
    return image

# mapping atom name to radius in nanometer
Atom2Radius = {
    "H": 0.120,
    "HE": 0.140,
    "B": 0.192,
    "C": 0.170,
    "CA": 0.170,
    "CB": 0.170,
    "CG": 0.170,
    "CG1": 0.170,
    "CG2": 0.170,
    "CG3": 0.170,
    "CD": 0.170,
    "CD1": 0.170,
    "CD2": 0.170,
    "CD3": 0.170,
    "CZ": 0.170,
    "CZ1": 0.170,
    "CZ2": 0.170,
    "CZ3": 0.170,
    "CE": 0.170,
    "CE1": 0.170,
    "CE2": 0.170,
    "CE3": 0.170,
    "CH": 0.170,
    "CH1": 0.170,
    "CH2": 0.170,
    "CH3": 0.170,
    "N": 0.155,
    "NE": 0.155,
    "NZ": 0.155,
    "ND1": 0.155,
    "ND2": 0.155,
    "NE1": 0.155,
    "NE2": 0.155,
    "NH1": 0.155,
    "NH2": 0.155,
    "O": 0.152,
    "OH": 0.152,
    "OG": 0.152,
    "OE1": 0.152,
    "OE2": 0.152,
    "OG1": 0.152,
    "OG2": 0.152,
    "OD1": 0.152,
    "OD2": 0.152,
    "OXT": 0.152,
    "F": 0.147,
    "MG": 0.173,
    "AL": 0.184,
    "SI": 0.210,
    "P": 0.180,
    "S": 0.180,
    "SD": 0.180,
    "SG": 0.180,
    "CL": 0.175,
    "AR": 0.188,
    "K": 0.275,
    "CYS": 0.275,
    "PHE": 0.32,
    "LEU": 0.31,
    "TRP": 0.34,
    "VAL": 0.295,
    "ILE": 0.31,
    "MET": 0.31,
    "HIS": 0.305,
    "HSD": 0.305,
    "TYR": 0.325,
    "ALA": 0.25,
    "GLY": 0.225,
    "PRO": 0.28,
    "ASN": 0.285,
    "THR": 0.28,
    "SER": 0.26,
    "ARG": 0.33,
    "GLN": 0.30,
    "ASP": 0.28,
    "LYS": 0.32,
    "GLU": 0.295
}

def define_tip(tip, resolution_x, resolution_y, probeRadius, probeAngle):
    """
    Define the tip shape by the probe radius and half-cone angle
        Input: tip (tensor of size (tip_height, tip_width))
               resolution_x (float) — pixel size in nm
               resolution_y (float) — pixel size in nm
               probeRadius (float) — tip apex radius in nm
               probeAngle (float) — half-cone angle in radians
        Output: tip (tensor of size (tip_height, tip_width))
    """
    tip_xsiz, tip_ysiz = tip.shape
    xc, yc = compute_xc_yc(tip)
    for ix in range(tip_xsiz):
        for iy in range(tip_ysiz):
            x = resolution_x * abs(ix - xc)
            y = resolution_y * abs(iy - yc)
            d = math.sqrt(x**2 + y**2)
            if d <= probeRadius:
                z = math.sqrt(probeRadius**2 - d**2)
            else:
                theta = (0.5 * math.pi) - probeAngle
                z = -math.tan(theta) * (d - probeRadius)
            tip[ix, iy] = z
    tip -= tip.max()
    return tip

######################################################################################
# PINN
######################################################################################

class TipShapeMLP(nn.Module):
    def __init__(self,n_size,n_hidden_layers,n_nodes):
        super().__init__()
        #n_input = 2*(n_size**2)
        #n_output = n_size**2
        n_input = 3
        n_output = 1

        self.l_in = nn.Sequential(
            nn.Linear(n_input,n_nodes),
            nn.ReLU()
        )

        layers=[]
        for i in range(0,n_hidden_layers):
            layers.extend(nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU()
            ))
        self.l_hidden = nn.Sequential(*layers)

        self.l_out = nn.Linear(n_nodes,n_output)

        self.n_hidden = n_hidden_layers
        

    def forward(self, x, y, t):

        xyt = torch.stack((y,x,t), dim=1).to(x.device)
        xyt2 = self.l_in(xyt)
        xyt3 = self.l_hidden(xyt2)
        xyt4 = self.l_out(xyt3)
        tip = xyt4
        return -tip

def generate_tip_from_mlp(tip_mlp, kernel_size, t, xc, yc, device):
    """
    Generate a 2D tensor tip shape from TipShapeMLP
    
    Args:
    tip_mlp (TipShapeMLP): The MLP model for tip shape
    kernel_size (int): The size of the kernel (assumed to be square)
    device (torch.device, optional): The device to put the tensor on
    
    Returns:
    torch.Tensor: A 2D tensor representing the tip shape
    """
    if device is None:
        device = next(tip_mlp.parameters()).device

    x = torch.linspace(-kernel_size/2, kernel_size/2, kernel_size, device=device)
    y = torch.linspace(-kernel_size/2, kernel_size/2, kernel_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X + xc
    Y = Y + yc

    with torch.set_grad_enabled(tip_mlp.training):
        tip = tip_mlp(X.flatten(), Y.flatten(), t).view(kernel_size, kernel_size)

    return tip

def idilation_mlp(image, tip_mlp, kernel_size, t, xc, yc):
    """
    Compute the dilation of surface by tip represented as MLP
    
    Args:
    image (torch.Tensor): Input image of size (surface_height, surface_width)
    tip_mlp (TipShapeMLP): The MLP model for tip shape
    kernel_size (int): The size of the kernel for the tip

    Returns:
    torch.Tensor: Dilated image of size (image_height, image_width)
    """
    tip = generate_tip_from_mlp(tip_mlp, kernel_size, t, xc, yc, device=image.device)
    return idilation(image, tip)

def ierosion_mlp(surface, tip_mlp, kernel_size, t , xc, yc):
    """
    Compute the erosion of image by tip represented as MLP
    
    Args:
    surface (torch.Tensor): Input surface of size (surface_height, surface_width)
    tip_mlp (TipShapeMLP): The MLP model for tip shape
    kernel_size (int): The size of the kernel for the tip

    Returns:
    torch.Tensor: Eroded surface of size (surface_height, surface_width)
    """
    tip = generate_tip_from_mlp(tip_mlp, kernel_size, t, xc, yc, device=surface.device)
    return ierosion(surface, tip)

import torch
import torch.nn as nn

class BTRLoss(nn.Module):
    def __init__(self, tip_mlp, kernel_size, boundary_weight, weight_decay, height_constraint_weight, average_weight, centroid_weight, time_weight):
        super().__init__()
        self.tip_mlp = tip_mlp
        self.kernel_size = kernel_size
        self.boundary_weight = boundary_weight
        self.height_constraint_weight = height_constraint_weight
        self.weight_decay = weight_decay
        self.average_weight = average_weight
        self.centroid_weight = centroid_weight
        self.time_weight = time_weight
        
    def forward(self, images,n,xc,yc):
        batch_size = images.shape[0]
        total_loss = 0.0

        # Generate input
        x = torch.linspace(-self.kernel_size/2, self.kernel_size/2, self.kernel_size, device=images.device,requires_grad=True)
        y = torch.linspace(-self.kernel_size/2, self.kernel_size/2, self.kernel_size, device=images.device,requires_grad=True)
        
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X = X + xc
        Y = Y + yc

        for i in range(batch_size):
            image = images[i]
            t_scalar = torch.tensor(n*batch_size+i, dtype=torch.float32, requires_grad=True).to(images.device)
            t = t_scalar.expand_as(X.flatten())
            tip_shape = self.tip_mlp(X.flatten(), Y.flatten(),t).view(self.kernel_size, self.kernel_size)
            # Erosion followed by dilation
            eroded = ierosion_mlp(image, self.tip_mlp, self.kernel_size, t,xc,yc)
            reconstructed = idilation_mlp(eroded, self.tip_mlp, self.kernel_size, t,xc,yc)

            # Reconstruction loss (MSE)
            recon_loss = torch.mean((reconstructed - image) ** 2)
            
                   
            # Calculate the differential of the tip
            dt = torch.autograd.grad(tip_shape.sum(), t, create_graph=True)[0]**2
            dx = torch.autograd.grad(tip_shape.sum(), X, create_graph=True)[0]**2
            dy = torch.autograd.grad(tip_shape.sum(), Y, create_graph=True)[0]**2
            ddx = torch.autograd.grad(dx.sum(), X, create_graph=True)[0]**2
            ddy = torch.autograd.grad(dy.sum(), Y, create_graph=True)[0]**2

            dt_loss = torch.sum(dt**2)
            dx_loss = torch.sum(dx**2)
            dy_loss = torch.sum(dy**2)
            dd_loss = torch.sum(ddx**2+ddy**2)


            # Boundary condition loss
            boundary_heights = self.tip_mlp(X.flatten(), Y.flatten() ,t)
            boundary_loss = torch.mean((boundary_heights + 100) ** 2)
            
            #boundary_heights_reshaped = boundary_heights.view(self.kernel_size, self.kernel_size)
            #edge_mask = torch.zeros_like(boundary_heights_reshaped, dtype=torch.bool)
            #edge_mask[0, :] = edge_mask[self.kernel_size-1, :] = edge_mask[:, 0] = edge_mask[:, self.kernel_size-1] = True
            #edge_values = boundary_heights_reshaped[edge_mask]
            #boundary_loss = torch.sum((edge_values + 100) ** 2)

            regularization_loss = torch.sum((boundary_heights)**2)
            centroid_loss = torch.dot(torch.abs(boundary_heights.flatten()), X.flatten())**2 + torch.dot(torch.abs(boundary_heights.flatten()), Y.flatten())**2
            average_loss = (torch.mean(boundary_heights))**2
        
            # Height constraint loss
            height_loss = torch.mean(torch.relu(tip_shape)**2)  + torch.mean(tip_shape.max()  ** 2) 

            # Combine losses
            total_loss += (
                recon_loss
                + self.boundary_weight * boundary_loss
                + self.height_constraint_weight * height_loss
                + self.weight_decay * regularization_loss
                + self.average_weight * average_loss
                + self.centroid_weight * centroid_loss
                + self.time_weight * dt_loss
            )

        return total_loss / batch_size
          
# Usage example
def Tip_mlp(
        dataloader,
        num_epochs,
        lr,
        kernel_size,
        boundary_weight,
        height_constraint_weight,
        weight_decay,
        average_weight,
        centroid_weight,
        n_hidden_layers,
        n_nodes,
        num_frame,
        time_weight,
        device
):
    # Initialize the TipShapeMLP model
    tip_mlp = TipShapeMLP(
        n_size=kernel_size,
        n_hidden_layers=n_hidden_layers,
        n_nodes=n_nodes
    ).to(device)

    # Initialize the BTRLoss criterion
    criterion = BTRLoss(
        tip_mlp,
        kernel_size=kernel_size,
        boundary_weight=boundary_weight,
        weight_decay=weight_decay,
        height_constraint_weight=height_constraint_weight,
        average_weight=average_weight,
        centroid_weight=centroid_weight,
        time_weight=time_weight
    ).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(tip_mlp.parameters(), lr=lr)
   
    #Generate input
    x = torch.linspace(-kernel_size/2, kernel_size/2, kernel_size,device=device)
    y = torch.linspace(-kernel_size/2, kernel_size/2, kernel_size,device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xc = torch.tensor(0.0).to(device) 
    yc = torch.tensor(0.0).to(device)
    t_scalar= torch.tensor(0.0, dtype=torch.float32).to(device)
    t = t_scalar.expand(kernel_size*kernel_size)

# Training loop
    loss_train = []
    for epoch in range(num_epochs):
            n = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                loss = criterion(batch,n,xc,yc)
                loss.backward()
                optimizer.step()
                n += 1.0
                with torch.no_grad():
                    if epoch%25==0 and loss.item()>100:
                        tip = generate_tip_from_mlp(tip_mlp, kernel_size=kernel_size,xc=xc, yc=yc,t=t ,device=device)
                        weight = tip - tip.min()  # 最小値を0にシフト 
                        xc = torch.dot(weight.flatten(), (X+xc).flatten())/torch.sum(weight.flatten())
                        yc = torch.dot(weight.flatten(), (Y+yc).flatten())/torch.sum(weight.flatten())
            loss_train.append(loss.item())
  
    
    tip = generate_tip_from_mlp(tip_mlp, kernel_size=kernel_size, t=t,xc=xc, yc=yc, device=device)
    tip_estimate = tip.detach()

    tip_animation = torch.empty(0,kernel_size, kernel_size).to(device)

    for frame in range(num_frame):
        t_scalar = torch.tensor(frame, dtype=torch.float32).to(device)
        t = t_scalar.expand(kernel_size*kernel_size)
        tip = generate_tip_from_mlp(tip_mlp, kernel_size=kernel_size, t=t,xc=xc, yc=yc, device=device)
        tip_animation = torch.cat([tip_animation, tip.unsqueeze(0)], dim=0)
    
    tip_animation = tip_animation.detach()

    return tip_estimate, loss_train, tip_animation ,xc , yc
######################################################################################
#molecular surface
######################################################################################
class SurfaceMLP(nn.Module):
    def __init__(self,n_hidden_layers,n_nodes):
        super().__init__()
        n_input = 3
        n_output = 1

        self.l_in = nn.Sequential(
            nn.Linear(n_input,n_nodes),
            nn.ReLU()
        )

        layers=[]
        for i in range(0,n_hidden_layers):
            layers.extend(nn.Sequential(
            nn.Linear(n_nodes,n_nodes),
            nn.ReLU()
            ))
        self.l_hidden = nn.Sequential(*layers)

        self.l_out = nn.Linear(n_nodes,n_output)

        self.n_hidden = n_hidden_layers
        

    def forward(self, x, y, t):

        xyt = torch.stack((x, y, t), dim=1).to(x.device)
        xyt2 = self.l_in(xyt)
        xyt3 = self.l_hidden(xyt2)
        xyt4 = self.l_out(xyt3)
        surface = xyt4
        return surface

def generate_surface_from_mlp(surface_mlp, x, y, t, device):
    
    if device is None:
        device = next(surface_mlp.parameters()).device

    X, Y = torch.meshgrid(x, y, indexing='ij')

    with torch.set_grad_enabled(surface_mlp.training):
        surface = surface_mlp(X.flatten(), Y.flatten(), t.flatten()).view(x.shape[0], y.shape[0])
    return surface


class SurfaceLoss(nn.Module):
    def __init__(self, surface_mlp,diffusion_weight):
        super().__init__()
        self.surface_mlp = surface_mlp
        self.diffusion_weight = diffusion_weight

    def forward(self, images,n):
        batch_size = images.shape[0]
        y_size  =  images.shape[1]
        x_size  =  images.shape[2]
        total_loss = 0.0

        # Generate surface_mlp input
        x = torch.linspace(-x_size/2, x_size/2, x_size, device=images.device,requires_grad=True)
        y = torch.linspace(-y_size/2, y_size/2, y_size, device=images.device,requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        t = torch.empty(0, x_size).to(device=images.device)
        for i in range(y_size):
            t = torch.cat([t, torch.linspace(i*x_size, (i+1)*x_size-1, x_size, device=images.device,requires_grad=True).unsqueeze(0)], dim=0)
        t = t/x_size*y_size

        t_recon = torch.zeros(y_size,x_size,requires_grad=True).to(images.device)

        for i in range(batch_size):
            image = images[i]
            T = i + t + n * batch_size
            T_recon = i + t_recon + n * batch_size

            surface = self.surface_mlp(X.flatten(), Y.flatten(), T.flatten())
            surface_recon = self.surface_mlp(X.flatten(), Y.flatten(), T_recon.flatten())

            dt = torch.autograd.grad(surface.sum(), T, create_graph=True)[0]
            dx = torch.autograd.grad(surface.sum(), X, create_graph=True)[0]
            dy = torch.autograd.grad(surface.sum(), Y, create_graph=True)[0]
            ddx = torch.autograd.grad(dx.sum(), X, create_graph=True)[0]
            ddy = torch.autograd.grad(dy.sum(), Y, create_graph=True)[0]

            dt_recon = torch.autograd.grad(surface_recon.sum(), T_recon, create_graph=True)[0]
            dx_recon = torch.autograd.grad(surface_recon.sum(), X, create_graph=True)[0]
            dy_recon = torch.autograd.grad(surface_recon.sum(), Y, create_graph=True)[0]
            ddx_recon = torch.autograd.grad(dx_recon.sum(), X, create_graph=True)[0]
            ddy_recon = torch.autograd.grad(dy_recon.sum(), Y, create_graph=True)[0]

            diffusion_loss = torch.mean( (dt_recon - self.diffusion_weight*(ddx_recon+ddy_recon))**2)
            boundary_loss =  torch.mean((image - surface_recon.view(y.shape[0], x.shape[0]))**2)

            # Combine losses
            total_loss += boundary_loss 
            #+ diffusion_loss

        return total_loss/batch_size

def surface_reconstruct(dataloader,n_hidden_layers,n_nodes,lr,num_epochs,num_frame,diffusion_weight,device):
    surface_mlp = SurfaceMLP(n_hidden_layers=n_hidden_layers,n_nodes=n_nodes).to(device)

    criterion = SurfaceLoss(surface_mlp,diffusion_weight).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(surface_mlp.parameters(), lr=lr)

    # Training loop
    loss_train = []
    for epoch in range(num_epochs):
            n = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                loss = criterion(batch,n)
                loss.backward()
                optimizer.step()
                n += 1.0
            loss_train.append(loss.item())

    x_size = batch.shape[2]
    y_size  = batch.shape[1]
    x = torch.linspace(-x_size/2, x_size/2, x_size, device=device)
    y = torch.linspace(-y_size/2, y_size/2, y_size, device=device)

    surface_estimate = torch.empty(0, y_size, x_size).to(device)

    for frame in range(num_frame):
        t = torch.zeros(y_size,x_size,device=device) + frame 
        surface = generate_surface_from_mlp(surface_mlp, x, y, t, device)
        surface_estimate = torch.cat([surface_estimate, surface.unsqueeze(0)], dim=0)

    surface_estimate = surface_estimate.detach()

    return surface_estimate ,loss_train


def load_pdb_ca(filepath):
    """
    Load CA (alpha carbon) atoms from a PDB file.
    Returns coordinates and residue radii using Atom2Radius.
        Input: filepath (str) - path to PDB file
        Output: xyz (tensor of size (N, 3)) in nm,
                radii (tensor of size (N,))  in nm
    """
    coords = []
    radii = []
    with open(filepath) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resname = line[17:20].strip()
            if resname in Atom2Radius:
                r = Atom2Radius[resname]
            else:
                r = 0.17  # default C radius
            coords.append([x / 10.0, y / 10.0, z / 10.0])  # Angstrom -> nm
            radii.append(r)
    xyz = torch.tensor(coords, dtype=torch.float32)
    radii = torch.tensor(radii, dtype=torch.float32)
    return xyz, radii


def add_noise(images, noise_type="gaussian", sigma=0.3, seed=None):
    """
    Add noise to AFM images.
        Input: images (tensor of size (nframe, H, W) or (H, W))
               noise_type ("gaussian" or "poisson")
               sigma (float) — noise level in nm (default 0.3)
               seed (int or None) — random seed for reproducibility
        Output: noisy_images (tensor, same shape as images)
    """
    gen = torch.Generator(device=images.device)
    if seed is not None:
        gen.manual_seed(seed)
    if noise_type == "gaussian":
        noise = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen) * sigma
        return images + noise
    elif noise_type == "poisson":
        shifted = torch.clamp(images, min=0.0) / sigma
        noisy = torch.poisson(shifted, generator=gen) * sigma
        return noisy + (images - torch.clamp(images, min=0.0))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'poisson'.")


def pixel_rmsd(tip1, tip2, ref, cutoff=-70.0, max_shift=5):
    """
    Compute pixel RMSD between two tip shapes with alignment.
    Tries all shifts in [-max_shift, max_shift] and returns the minimum RMSD.
    Only pixels where ref > cutoff are considered.
        Input: tip1, tip2, ref (tensors of same shape)
               cutoff (float) - threshold for ref mask
               max_shift (int) - maximum shift in pixels
        Output: rmsd (float)
    """
    rmsd_min = float('inf')
    for du in range(-max_shift, max_shift + 1):
        for dv in range(-max_shift, max_shift + 1):
            shifted = torch.roll(tip1, shifts=(du, dv), dims=(0, 1))
            mask = ref > cutoff
            if mask.sum() == 0:
                continue
            diff = (shifted[mask] - tip2[mask]) ** 2
            rmsd = torch.sqrt(diff.mean()).item()
            if rmsd < rmsd_min:
                rmsd_min = rmsd
    return rmsd_min
