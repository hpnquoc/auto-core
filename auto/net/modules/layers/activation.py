import torch
import torch.nn as nn
from loguru import logger as logging

try:
    import core
except ImportError:
    logging.error("Please make sure to run your code from the root directory of the project.")

__all__ = [
    'SimpleGate'
]

core.layer_register.register('simplegate')
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2