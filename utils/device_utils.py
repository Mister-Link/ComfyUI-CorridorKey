import torch


def get_device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
