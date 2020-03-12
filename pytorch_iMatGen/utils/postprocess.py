import torch


def post_process(x):
    x = torch.clamp(x, 0.5, 0.5001) - 0.5
    x = torch.min(x * 10000, torch.ones_like(x))
    return x
