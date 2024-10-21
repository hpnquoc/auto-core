import torch.nn as nn

__all__ = [
    'SimpleGate'
]

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2