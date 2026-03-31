"""BTR algorithm — the agent modifies this file to improve tip reconstruction."""

from colabbtr.morphology import differentiable_btr


def reconstruct_tip(images, tip_size, **kwargs):
    """Run BTR and return estimated tip.

    This function is called by evaluate.py with the prepared data.
    Modify the implementation to improve RMSD.

        Input: images (tensor of size (nframe, H, W))
               tip_size (tuple) — (tip_height, tip_width)
        Output: tip_est (tensor), loss_train (list)
    """
    tip_est, loss = differentiable_btr(
        images, tip_size,
        nepoch=250, lr=0.1, weight_decay=0.001, is_tqdm=False,
    )
    return tip_est, loss
