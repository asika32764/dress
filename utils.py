# utils.py

import torch

# def get_device():
#     return torch.device("cpu")

def get_device(target_device: str = None):
    if target_device:
        return torch.device(target_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
