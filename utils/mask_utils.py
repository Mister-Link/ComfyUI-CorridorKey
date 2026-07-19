# Screen-color detection + mask generation (fast classical HSV / BiRefNet), ported from
# https://huggingface.co/spaces/Luminia/CorridorKey (app.py).

import logging
import os
import glob
import ctypes

import numpy as np
import cv2


def _preload_cuda_runtime():
    """onnxruntime-gpu's CUDAExecutionProvider needs CUDA 13 runtime libs (cudart,
    cublas, curand, cufft, nvrtc, ...) plus cuDNN 9 — all present as pip dependencies
    (nvidia-cuda-runtime, nvidia-cudnn-cu13) but shipped under nvidia/cu13/lib and
    nvidia/cudnn/lib in site-packages rather than a system library path. The dynamic
    linker only consults LD_LIBRARY_PATH at process startup, so setting the env var
    here (mid-process, inside an already-running ComfyUI) has no effect — preload
    each .so directly by absolute path instead, before onnxruntime imports them.
    Coexists fine with torch's own (separately-versioned) CUDA/cuDNN libs since
    sonames differ (e.g. libcudart.so.13 vs torch's libcudart.so.12).
    """
    try:
        import nvidia
        nvidia_dir = os.path.dirname(nvidia.__file__)
    except ImportError:
        return
    for subdir in ("cu13/lib", "cudnn/lib"):
        libs = sorted(glob.glob(os.path.join(nvidia_dir, subdir, "*.so*")))
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        for lib in failed:  # retry once — handles simple load-order dependencies
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # e.g. libcufile_rdma.so.1 needs librdmacm, unused here


_preload_cuda_runtime()

import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger("CorridorKeyLuminia")

BIREFNET_REPO = "onnx-community/BiRefNet_lite-ONNX"
BIREFNET_FILE = "onnx/model.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

_birefnet_session = None


def _get_providers():
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_birefnet():
    """Lazily download + load the BiRefNet ONNX session (module-level singleton)."""
    global _birefnet_session
    if _birefnet_session is None:
        path = hf_hub_download(repo_id=BIREFNET_REPO, filename=BIREFNET_FILE)
        providers = _get_providers()
        logger.info("Loading BiRefNet ONNX: %s (providers: %s)", path, providers)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _birefnet_session = ort.InferenceSession(path, opts, providers=providers)
    return _birefnet_session


def birefnet_frame(session, image_rgb_uint8):
    """Run BiRefNet on a single [H, W, 3] uint8 RGB frame. Returns [H, W] float32 mask."""
    h, w = image_rgb_uint8.shape[:2]
    inp = session.get_inputs()[0]
    res = (inp.shape[2], inp.shape[3])
    img = cv2.resize(image_rgb_uint8, res).astype(np.float32) / 255.0
    img = ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    pred = 1.0 / (1.0 + np.exp(-session.run(None, {inp.name: img})[-1]))
    return (cv2.resize(pred[0, 0], (w, h)) > 0.04).astype(np.float32)


def estimate_screen_color(frame_f32, alpha_hint=None):
    """Detect green vs blue screen from background pixels. Returns 'green' or 'blue'."""
    if alpha_hint is not None:
        if alpha_hint.ndim == 3:
            alpha_hint = alpha_hint[:, :, 0]
        bg_mask = alpha_hint < 0.3
    else:
        h, w = frame_f32.shape[:2]
        ph, pw = max(int(h * 0.05), 4), max(int(w * 0.05), 4)
        bg_mask = np.zeros((h, w), dtype=bool)
        bg_mask[:ph, :pw] = bg_mask[:ph, -pw:] = bg_mask[-ph:, :pw] = bg_mask[-ph:, -pw:] = True
    if bg_mask.mean() < 0.01:
        return "green"
    bg = frame_f32[bg_mask]
    mean_g, mean_b = float(bg[:, 1].mean()), float(bg[:, 2].mean())
    if abs(mean_g - mean_b) < 0.05:
        return "green"
    return "blue" if mean_b > mean_g else "green"


def fast_chromascreen_mask(frame_rgb_f32, screen_color="green"):
    """Fast classical mask for green or blue screens. Returns (mask, confidence, detected_color)."""
    h, w = frame_rgb_f32.shape[:2]
    ph, pw = max(int(h * 0.05), 4), max(int(w * 0.05), 4)
    corners = np.concatenate([
        frame_rgb_f32[:ph, :pw].reshape(-1, 3),
        frame_rgb_f32[:ph, -pw:].reshape(-1, 3),
        frame_rgb_f32[-ph:, :pw].reshape(-1, 3),
        frame_rgb_f32[-ph:, -pw:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)
    is_green = bg_color[1] > bg_color[0] + 0.05 and bg_color[1] > bg_color[2] + 0.05
    is_blue = bg_color[2] > bg_color[0] + 0.05 and bg_color[2] > bg_color[1] + 0.05
    if screen_color == "green" and not is_green:
        return None, 0.0, "green"
    if screen_color == "blue" and not is_blue:
        return None, 0.0, "blue"
    if screen_color == "auto" and not is_green and not is_blue:
        return None, 0.0, "green"
    detected = screen_color if screen_color != "auto" else ("blue" if is_blue and not is_green else "green")
    frame_u8 = (np.clip(frame_rgb_f32, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)
    if detected == "blue":
        screen_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
    else:
        screen_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    fg_mask = cv2.bitwise_not(screen_mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    mask_f32 = fg_mask.astype(np.float32) / 255.0
    confidence = 1.0 - 2.0 * np.mean(np.minimum(mask_f32, 1.0 - mask_f32))
    return mask_f32, confidence, detected


def generate_mask(frame_f32, mask_mode, screen_color="green", confidence_threshold=0.7):
    """Generate an alpha hint mask for one frame per the selected mask_mode.

    mask_mode: "Fast (classical)" | "AI (BiRefNet)" | "Hybrid (auto)"

    Returns (mask, method) where method is "fast" or "birefnet", so callers can
    log/diagnose how often the slower BiRefNet fallback is triggered.
    """
    if mask_mode == "Fast (classical)":
        mask, conf, _ = fast_chromascreen_mask(frame_f32, screen_color)
        if mask is None:
            raise RuntimeError("Fast (classical) mask failed to detect a green screen in this frame. Try 'Hybrid (auto)' or 'AI (BiRefNet)'.")
        return mask, "fast"
    if mask_mode == "Hybrid (auto)":
        mask, conf, _ = fast_chromascreen_mask(frame_f32, screen_color)
        if mask is None or conf < confidence_threshold:
            frame_u8 = (np.clip(frame_f32, 0, 1) * 255).astype(np.uint8)
            mask = birefnet_frame(get_birefnet(), frame_u8)
            return mask, "birefnet"
        return mask, "fast"
    # "AI (BiRefNet)"
    frame_u8 = (np.clip(frame_f32, 0, 1) * 255).astype(np.uint8)
    return birefnet_frame(get_birefnet(), frame_u8), "birefnet"
