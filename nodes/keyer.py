import logging
import time

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF

from ..utils.mask_utils import generate_mask
from ..utils.color_utils import (
    srgb_to_linear,
    linear_to_srgb,
    composite_straight,
    premultiply,
    create_checkerboard,
    despill_torch,
    clean_matte_torch,
)

logger = logging.getLogger("CorridorKeyLuminia")

# Only "green" screens are supported until a blue GreenFormer checkpoint is installed
# alongside CorridorKey_v1.0.pth (see CorridorKeyBlue_1.0 on Hugging Face).
SCREEN_CHANNEL = {"Green": 1}


class CKL_Keyer:
    """
    CorridorKey (Luminia) Keyer — self-mask-generating green screen keyer.

    Ported from https://huggingface.co/spaces/Luminia/CorridorKey. Unlike the earlier
    CorridorKey ComfyUI node, this one generates its own alpha hint internally
    (Fast classical HSV / AI BiRefNet / Hybrid auto-fallback) instead of requiring
    a pre-made mask input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CKL_MODEL",),
                "image": ("IMAGE",),
                "screen_color": (["Green"], {"default": "Green"}),
                "mask_mode": (
                    ["Hybrid (auto)", "AI (BiRefNet)", "Fast (classical)"],
                    {"default": "Hybrid (auto)"},
                ),
                "despill_strength": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "auto_despeckle": ("BOOLEAN", {"default": True}),
                "despeckle_size": ("INT", {"default": 400, "min": 10, "max": 5000, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("composite", "alpha", "foreground", "processed")
    FUNCTION = "key"
    CATEGORY = "CorridorKey"

    def key(self, model, image, screen_color, mask_mode,
            despill_strength, auto_despeckle, despeckle_size):
        ckl_model = model.model
        img_size = model.img_size
        device = model.device
        screen_channel = SCREEN_CHANNEL[screen_color]
        despill_norm = despill_strength / 10.0

        batch = image.shape[0]
        images_np = image.cpu().numpy().astype(np.float32)  # [B, H, W, 3] sRGB [0,1]

        # --- Phase 1: mask generation (CPU/numpy, per-frame) ---
        t_mask = time.time()
        masks_np = []
        fast_n, biref_n = 0, 0
        for i in range(batch):
            mask, method = generate_mask(images_np[i], mask_mode, screen_color="green")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            masks_np.append(mask)
            if method == "fast":
                fast_n += 1
            else:
                biref_n += 1
        logger.info(
            "[CorridorKeyLuminia] Mask phase: %d frames in %.1fs (fast=%d, birefnet=%d)",
            batch, time.time() - t_mask, fast_n, biref_n,
        )

        # --- Phase 2: batched inference (GPU if available) ---
        model_dtype = next(ckl_model.parameters()).dtype

        batch_imgs = torch.from_numpy(
            np.stack([img.transpose(2, 0, 1) for img in images_np])
        ).to(device)
        batch_masks = torch.from_numpy(np.stack(masks_np)).unsqueeze(1).to(device)

        batch_imgs = TF.resize(batch_imgs, [img_size, img_size], antialias=False)
        batch_masks = TF.resize(batch_masks, [img_size, img_size], antialias=False)
        batch_imgs = TF.normalize(batch_imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        inp = torch.cat([batch_imgs, batch_masks], dim=1).to(model_dtype)
        del batch_imgs, batch_masks

        with torch.inference_mode():
            out = ckl_model(inp)
        del inp

        alpha = out["alpha"].float()
        fg = out["fg"].float()

        if auto_despeckle:
            alpha = clean_matte_torch(alpha, area_threshold=int(despeckle_size), dilation=25, blur_size=5)
        fg = despill_torch(fg, despill_norm, screen_channel=screen_channel)

        alpha_np = (alpha.clamp(0, 1) * 255).byte().cpu().numpy()  # [B, 1, S, S]
        fg_np = (fg.clamp(0, 1) * 255).byte().cpu().numpy()        # [B, 3, S, S]
        del alpha, fg

        # --- Phase 3: resize back to original resolution + build outputs ---
        comp_list, alpha_list, fg_list, processed_list = [], [], [], []
        for i in range(batch):
            orig_h, orig_w = images_np[i].shape[:2]

            alpha_r = cv2.resize(alpha_np[i, 0], (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            fg_r = cv2.resize(fg_np[i].transpose(1, 2, 0), (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            alpha_f = alpha_r.astype(np.float32) / 255.0
            fg_f = fg_r.astype(np.float32) / 255.0
            alpha_3 = alpha_f[:, :, np.newaxis]

            bg_lin = srgb_to_linear(create_checkerboard(orig_w, orig_h))
            fg_lin = srgb_to_linear(fg_f)

            comp = linear_to_srgb(composite_straight(fg_lin, bg_lin, alpha_3)).clip(0, 1)
            processed = linear_to_srgb(premultiply(fg_lin, alpha_3)).clip(0, 1)

            comp_list.append(comp.astype(np.float32))
            alpha_list.append(alpha_f.astype(np.float32))
            fg_list.append(fg_f.astype(np.float32))
            processed_list.append(processed.astype(np.float32))

        return (
            torch.from_numpy(np.stack(comp_list)),
            torch.from_numpy(np.stack(alpha_list)),
            torch.from_numpy(np.stack(fg_list)),
            torch.from_numpy(np.stack(processed_list)),
        )


NODE_CLASS_MAPPINGS = {"CKL_Keyer": CKL_Keyer}
NODE_DISPLAY_NAME_MAPPINGS = {"CKL_Keyer": "CorridorKey (Luminia) Keyer"}
