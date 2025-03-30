# utils.py

import torch

def get_device():
    return torch.device("cpu")

# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")
