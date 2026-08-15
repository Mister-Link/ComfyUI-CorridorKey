import os
import math
import logging

import torch
import torch.nn.functional as F
import folder_paths

from ..corridorkey_core.model_transformer import GreenFormer
from ..utils.device_utils import get_device

logger = logging.getLogger("CorridorKeyLuminia")


class CKLModel:
    """Wrapper carrying the loaded GreenFormer model + the resolution it was built for."""

    def __init__(self, model, img_size, device):
        self.model = model
        self.img_size = img_size
        self.device = device


def _load_state_dict(checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(checkpoint_path, device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


class CKL_ModelLoader:
    """Load the CorridorKey (GreenFormer) checkpoint for a given inference resolution."""

    @classmethod
    def INPUT_TYPES(s):
        model_dir = os.path.join(folder_paths.models_dir, "corridorkey")
        os.makedirs(model_dir, exist_ok=True)
        models = [f for f in os.listdir(model_dir) if f.endswith((".pth", ".safetensors"))]
        if not models:
            models = ["CorridorKey_v1.0.pth"]
        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "resolution": (["1024", "2048"], {"default": "1024"}),
            },
        }

    RETURN_TYPES = ("CKL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "CorridorKey"

    def load(self, model_name, resolution):
        model_path = os.path.join(folder_paths.models_dir, "corridorkey", model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download a checkpoint (.pth or .safetensors) from:\n"
                f"  green: https://huggingface.co/nikopueringer/CorridorKey_v1.0\n"
                f"  blue:  https://huggingface.co/nikopueringer/CorridorKeyBlue_1.0\n"
                f"and place it in: {os.path.dirname(model_path)}\n"
                f"Pick the checkpoint matching the Keyer node's screen_color."
            )

        img_size = int(resolution)
        device = get_device()

        logger.info("Initializing GreenFormer (img_size=%d)...", img_size)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=img_size,
            use_refiner=True,
        )

        state_dict = _load_state_dict(model_path)

        # Fix compiled-model prefix & resize PosEmbed if the checkpoint was trained at a
        # different resolution than the one requested here.
        new_state_dict = {}
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]
            if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
                logger.info("Resizing %s from %s to %s", k, v.shape, model_state[k].shape)
                N_src, C = v.shape[1], v.shape[2]
                grid_src = int(math.sqrt(N_src))
                grid_dst = int(math.sqrt(model_state[k].shape[1]))
                v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
                v = v_resized.flatten(2).transpose(1, 2)
            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)

        model.eval()
        model = model.to(device)
        if device.type == "cuda":
            model = model.half()
            logger.info("[CorridorKeyLuminia] Model loaded as FP16 on %s", device)
        else:
            logger.info("[CorridorKeyLuminia] Model loaded as FP32 on %s", device)

        return (CKLModel(model, img_size, device),)


NODE_CLASS_MAPPINGS = {"CKL_ModelLoader": CKL_ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CKL_ModelLoader": "CorridorKey (Luminia) Model Loader"}
