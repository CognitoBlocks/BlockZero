import torch

import torch


def add_grad_noise(model, std: float):
    for p in model.parameters():
        if p.grad is None:
            # create a random grad if missing
            p.grad = torch.randn_like(p) * std
        else:
            # add random noise to existing grad
            p.grad.add_(torch.randn_like(p.grad) * std)
