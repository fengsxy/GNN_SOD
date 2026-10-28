"""BCE + Gaussian-window SSIM loss + soft IoU, with no legacy loss packages."""
import torch
from torch.nn import functional as F


def mixed_loss(prediction, target):
    if prediction.ndim != 4 or prediction.shape[1] != 1 or prediction.shape != target.shape:
        raise ValueError("Loss inputs must have matching [B, 1, H, W] shapes")
    width = min(11, *prediction.shape[-2:])
    width -= (width + 1) % 2
    axis = torch.arange(width, device=prediction.device, dtype=prediction.dtype) - width // 2
    gaussian = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    smooth = lambda x: F.conv2d(x, window, padding=width // 2)
    mu_p, mu_t = smooth(prediction), smooth(target)
    var_p = smooth(prediction.square()) - mu_p.square()
    var_t = smooth(target.square()) - mu_t.square()
    cov = smooth(prediction * target) - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + .01 ** 2) * (2 * cov + .03 ** 2) /
            ((mu_p.square() + mu_t.square() + .01 ** 2) * (var_p + var_t + .03 ** 2))).mean()
    axes = (1, 2, 3)
    intersection = (prediction * target).sum(axes)
    union = prediction.sum(axes) + target.sum(axes) - intersection
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
    return F.binary_cross_entropy(prediction, target) + (1 - ssim) + (1 - iou)
