import os
import torch


def get_device() -> tuple[torch.device, str]:
    """
    Resolve inference device based on DEVICE env variable.

    DEVICE values:
      - "CUDA": force CUDA (error if unavailable)
      - "CPU": force CPU
      - "AUTO": CUDA if available, else CPU (default)

    Returns:
        (device, device_mode)
    """
    device_mode = os.getenv("DEVICE", "AUTO").upper()

    if device_mode == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=CUDA requested but CUDA is not available")
        device = torch.device("cuda")

    elif device_mode == "CPU":
        device = torch.device("cpu")

    elif device_mode == "AUTO":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        raise ValueError("DEVICE must be one of: CUDA, CPU, AUTO")

    return device
