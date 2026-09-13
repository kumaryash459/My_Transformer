import torch


def create_grad_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")

    return None