import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from tqdm.notebook import tqdm

#     Please write docstring here to explain what this function does
#     and what are the inputs and outputs
def compute_xc_yc(tip):
    """
    Compute the center position of the tip
        Input: tip (tensor of size (tip_height, tip_width)
        Output: xc, yc (int)
    """
    tip_xsiz, tip_ysiz = tip.size()
    xc = round((tip_xsiz - 1) / 2)
    yc = round((tip_ysiz - 1) / 2)
    return xc, yc

#     Please write docstring here to explain what this function does
#     and what are the inputs and outputs
def idilation(surface, tip):
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

def ierosion(image, tip):
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
    """
    Reconstruct tip shape from given AFM images by differentiable blind tip reconstruction (BTR)
        Input: images (tensor of size (nframe, image_height, image_width)
               tip_size (2d tuple)
               nepoch (int)
               lr (float) for AdamW
               weight_decay (float) for AdamW
               device (str) ''cpu'' or ''cuda''
        Output: tip_estimate (tensor of tip_size), loss_train (list)
    """
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

def surfing(xyz, radius, config):
    """
    Compute the maximum height (z-value) of molecular surface at grid points on AFM stage (where z=0)
        Input: xyz (tensor of size (N, 3))
                radius (tensor of size (N,))
                config (dict)
        Output: z_stage (tensor of size (len(y_stage), len(x_stage))
    """
    radius2 = radius**2
    x_stage = torch.arange(config["min_x"], config["max_x"], config["resolution_x"]) + 0.5*config["resolution_x"]
    y_stage = torch.arange(config["min_y"], config["max_y"], config["resolution_y"]) + 0.5*config["resolution_y"]
    z_stage = torch.full((len(y_stage), len(x_stage)), xyz[:, 2].min())
    for i in range(len(x_stage)):
        for j in range(len(y_stage)):
            x = x_stage[i]
            y = y_stage[j]
            dx = xyz[:, 0] - x
            dy = xyz[:, 1] - y
            r2 = dx**2 + dy**2
            index_within_radius = r2 < radius2
            #print(r2[:3], radius[:3])
            if any(index_within_radius):
                z_stage[-j-1, i] = torch.max(xyz[index_within_radius, 2] + torch.sqrt(radius2[index_within_radius] - r2[index_within_radius]))
    return z_stage

# mapping atom name to radius in Angstrom
Atom2Radius = {
    "H": 1.20,
    "HE": 1.40,
    "B": 1.92,
    "C": 1.70,
    "CA": 1.70,
    "CB": 1.70,
    "CG": 1.70,
    "CG1": 1.70,
    "CG2": 1.70,
    "CG3": 1.70,
    "CD": 1.70,
    "CD1": 1.70,
    "CD2": 1.70,
    "CD3": 1.70,
    "CZ": 1.70,
    "CZ1": 1.70,
    "CZ2": 1.70,
    "CZ3": 1.70,
    "CE": 1.70,
    "CE1": 1.70,
    "CE2": 1.70,
    "CE3": 1.70,
    "CH": 1.70,
    "CH1": 1.70,
    "CH2": 1.70,
    "CH3": 1.70,
    "N": 1.55,
    "NE": 1.55,
    "NZ": 1.55,
    "ND1": 1.55,
    "ND2": 1.55,
    "NE1": 1.55,
    "NE2": 1.55,
    "NH1": 1.55,
    "NH2": 1.55,
    "O": 1.52,
    "OH": 1.52,
    "OG": 1.52,
    "OE1": 1.52,
    "OE2": 1.52,
    "OG1": 1.52,
    "OG2": 1.52,
    "OD1": 1.52,
    "OD2": 1.52,
    "OXT": 1.52,
    "F": 1.47,
    "MG": 1.73,
    "AL": 1.84,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "SD": 1.80,
    "SG": 1.80,
    "CL": 1.75,
    "AR": 1.88,
    "K": 2.75,
    "CYS": 2.75,
    "PHE": 3.2,
    "LEU": 3.1,
    "TRP": 3.4,
    "VAL": 2.95,
    "ILE": 3.1,
    "MET": 3.1,
    "HIS": 3.05,
    "HSD": 3.05,
    "TYR": 3.25,
    "ALA": 2.5,
    "GLY": 2.25,
    "PRO": 2.8,
    "ASN": 2.85,
    "THR": 2.8,
    "SER": 2.6,
    "ARG": 3.3,
    "GLN": 3.0,
    "ASP": 2.8,
    "LYS": 3.2,
    "GLU": 2.95
}
