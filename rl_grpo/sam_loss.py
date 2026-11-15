"""
sam_loss.py  –  Segmentation losses for TGS-Agent RL integration.
Copied from VideoSEG-O3/projects/open_r1/trainer/sam_loss.py and kept
identical so that the two codebases stay in sync.
"""

import torch
import torch.nn.functional as F


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    DICE loss, similar to generalised IoU for masks.

    Args:
        inputs:  float tensor of arbitrary shape (predictions).
        targets: float tensor with the same shape as inputs (binary GT).
        num_objects: number of objects in the batch.
        loss_on_multimask: True when multimask prediction is enabled.
    Returns:
        Scalar dice loss tensor.
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs / targets: [N, M, H, W]  (M = multiple masks)
        assert inputs.dim() == 4 and targets.dim() == 4
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_bce_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask: bool = False,
):
    """
    Focal / sigmoid BCE loss used in RetinaNet (arXiv:1708.02002).

    Args:
        inputs:  float tensor (predictions, raw logits).
        targets: float tensor (binary GT, same shape as inputs).
        num_objects: number of objects in the batch.
        alpha: weighting factor in [0,1] for positive vs negative examples.
        gamma: exponent of the modulating factor (1 - p_t).
        loss_on_multimask: True when multimask prediction is enabled.
    Returns:
        Scalar focal loss tensor.
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if loss_on_multimask:
        # loss: [N, M, H, W]
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False):
    """
    IoU prediction loss (MSE or L1 between predicted and actual IoU).

    Args:
        inputs:   float tensor [N, M, H, W].
        targets:  float tensor [N, M, H, W].
        pred_ious: float tensor of predicted IoU scores per mask.
        num_objects: number of objects.
        loss_on_multimask: True for multimask mode.
        use_l1_loss: use L1 instead of MSE.
    Returns:
        Scalar IoU loss tensor.
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious.float(), actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious.float(), actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def calculate_boundary_iou(mask1, mask2):
    """Boundary IoU between two binary mask tensors [N,1,H,W]."""
    assert mask1.dim() == 4 and mask2.dim() == 4
    assert mask1.shape == mask2.shape

    mask2 = mask2.to(mask1.device)
    boundary1 = _mask2boundary(mask1)
    boundary2 = _mask2boundary(mask2)

    boundary1 = (boundary1 > 0).squeeze(0)
    boundary2 = (boundary2 > 0).squeeze(0)

    inter = (boundary1 & boundary2).sum().float()
    union = (boundary1 | boundary2).sum().float()

    if union == 0:
        biou = torch.tensor(1.0 if inter == 0 else 0.0, device=mask1.device)
        return biou, boundary1, boundary2

    biou = inter / (union + 1e-6)
    return biou, boundary1, boundary2


def _mask2boundary(bin_mask):
    assert bin_mask.max() <= 1 and bin_mask.min() >= 0
    bin_mask = bin_mask.float()
    bw = min(bin_mask.shape[-2:]) // 20
    if bw % 2 == 0:
        bw += 1
    inner = 1 - F.max_pool2d(1 - bin_mask, bw, 1, bw // 2)
    return (bin_mask - inner) > 0.5
