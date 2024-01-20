import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from tqdm.notebook import tqdm

def compute_xc_yc(tip):
    tip_xsiz, tip_ysiz = tip.size()
    xc = round((tip_xsiz - 1) / 2)
    yc = round((tip_ysiz - 1) / 2)
    return xc, yc

def idilation(surface, tip):
    xc, yc = compute_xc_yc(tip)
    tip_xsiz, tip_ysiz = tip.size()

    r = torch.full_like(surface, -float('inf'))
    for px in range(-xc, tip_xsiz - xc):
        for py in range(-yc, tip_ysiz - yc):
            temp = torch.roll(surface, shifts=(-px, -py), dims=(0, 1))
            temp = temp + tip[xc + px, yc + py]
            r = torch.maximum(r, temp)
    return r

def ierosion(image, tip):
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
    tip_xsiz, tip_ysiz = P.size()
    xc, yc = compute_xc_yc(P)

    #p_max = torch.min(P)
    #P = P - p_max

    p_min = torch.min(P)
    weight = P - p_min
    weight = weight

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

def differentiable_btr(images, tip_size, nepoch=100, lr=0.1, weight_decay=0.0, device='cpu'):
    # Initialize tip with zeros
    tip = torch.zeros(tip_size, dtype=torch.float64, requires_grad=True, device=device)

    # Optimization settings
    optimizer = optim.AdamW([tip], lr=lr, weight_decay=weight_decay)

    loss_train = []
    for epoch in tqdm(range(nepoch)):
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
