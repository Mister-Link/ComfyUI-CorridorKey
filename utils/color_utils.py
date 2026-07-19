# Color/despill/despeckle math, ported from
# https://huggingface.co/spaces/Luminia/CorridorKey (app.py).

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def composite_straight(fg, bg, alpha):
    return fg * alpha + bg * (1.0 - alpha)


def premultiply(fg, alpha):
    return fg * alpha


def create_checkerboard(w, h, checker_size=64, color1=0.15, color2=0.55):
    xg, yg = np.meshgrid(np.arange(w) // checker_size, np.arange(h) // checker_size)
    bg = np.where(((xg + yg) % 2) == 0, color1, color2).astype(np.float32)
    return np.stack([bg, bg, bg], axis=-1)


def clean_matte(alpha_np, area_threshold=300, dilation=15, blur_size=5):
    """CPU/numpy despeckle: keep the largest component + anything above area_threshold."""
    is_3d = alpha_np.ndim == 3
    if is_3d:
        alpha_np = alpha_np[:, :, 0]
    mask_8u = (alpha_np > 0.02).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)
    valid = np.zeros(num_labels, dtype=bool)
    valid[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_threshold
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        valid[largest] = True
    cleaned = (valid[labels].astype(np.uint8) * 255)
    if dilation > 0:
        k = int(dilation * 2 + 1)
        cleaned = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if blur_size > 0:
        b = int(blur_size * 2 + 1)
        cleaned = cv2.GaussianBlur(cleaned, (b, b), 0)
    result = alpha_np * (cleaned.astype(np.float32) / 255.0)
    return result[:, :, np.newaxis] if is_3d else result


def despill(image, strength=1.0, screen_channel=1):
    """CPU/numpy despill on an [H, W, 3] sRGB frame."""
    if strength <= 0.0:
        return image
    other_a, other_b = [i for i in (0, 1, 2) if i != screen_channel]
    screen = image[..., screen_channel]
    a, b = image[..., other_a], image[..., other_b]
    spill = np.maximum(screen - (a + b) / 2.0, 0.0)
    out = np.empty_like(image)
    out[..., screen_channel] = screen - spill
    out[..., other_a] = a + spill * 0.5
    out[..., other_b] = b + spill * 0.5
    return image * (1.0 - strength) + out * strength if strength < 1.0 else out


def despill_torch(image, strength, screen_channel=1):
    """GPU despill on a [B, 3, H, W] tensor."""
    if strength <= 0.0:
        return image
    other_a, other_b = [i for i in (0, 1, 2) if i != screen_channel]
    screen, a, b = image[:, screen_channel], image[:, other_a], image[:, other_b]
    spill = torch.clamp(screen - (a + b) / 2.0, min=0.0)
    out = [None, None, None]
    out[screen_channel] = screen - spill
    out[other_a] = a + spill * 0.5
    out[other_b] = b + spill * 0.5
    despilled = torch.stack(out, dim=1)
    return image * (1.0 - strength) + despilled * strength if strength < 1.0 else despilled


def _clean_matte_single_gpu(alpha_single, area_threshold, dilation=25, blur_size=5, max_iter=20):
    """GPU clean matte on a single [1, 1, H, W] frame. Per-frame to avoid randperm overflow."""
    _, _, H, W = alpha_single.shape
    mask = (alpha_single > 0.02).float()
    comp = (torch.randperm(W * H, device=mask.device).float() + 1.0).view(1, 1, H, W)
    comp[mask != 1] = 0
    for _ in range(max_iter):
        comp[mask == 1] = F.max_pool2d(comp, 9, stride=1, padding=4)[mask == 1]
    _, comp = torch.unique(comp, return_inverse=True)
    comp = comp.view(1, 1, H, W)
    sizes = torch.bincount(comp.flatten())
    big = torch.nonzero(sizes >= area_threshold).squeeze(-1)
    big = big[big > 0]
    largest = sizes[1:].argmax() + 1 if sizes.shape[0] > 1 else None
    if largest is not None and largest not in big:
        big = torch.cat([big, largest.unsqueeze(0)])
    cleaned = torch.zeros_like(mask)
    if big.numel() > 0:
        cleaned[torch.isin(comp, big)] = 1.0
    if dilation > 0:
        for _ in range(dilation // 2):
            cleaned = F.max_pool2d(cleaned, 5, stride=1, padding=2)
    if blur_size > 0:
        cleaned = TF.gaussian_blur(cleaned, [blur_size * 2 + 1, blur_size * 2 + 1])
    return alpha_single * cleaned


def clean_matte_torch(alpha, area_threshold, dilation=25, blur_size=5):
    """GPU clean matte on [B, 1, H, W] tensor. Processes per-frame (avoids randperm 2^24 limit)."""
    max_iter = max(area_threshold // 20, 5)
    return torch.cat([
        _clean_matte_single_gpu(alpha[i:i + 1], area_threshold, dilation, blur_size, max_iter)
        for i in range(alpha.shape[0])
    ], dim=0)
