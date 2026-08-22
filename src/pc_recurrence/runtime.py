from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def select_device() -> tuple[torch.device, str]:
    """Select the best PyTorch device available in the current environment."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device, torch.cuda.get_device_name(device)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Metal Performance Shaders"
    return torch.device("cpu"), "CPU"


def device_memory_info(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def inference_autocast(device: torch.device) -> AbstractContextManager[object]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
