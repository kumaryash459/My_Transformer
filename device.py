import torch


def get_device() -> torch.device:
    """
    Select the best available computation device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info() -> None:
    """
    Print information about the selected device.
    """

    device = get_device()

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA version: {torch.version.cuda}"
        )