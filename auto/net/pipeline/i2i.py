import math
import torch
import abc
from tqdm import tqdm
import os

__all__ = [
    ''
]

class I2I(abc.ABC):
    def __init__(self, enc='', device=None):
        self.enc = 
        self.device = device

    @abc.abstractmethod
    def encode(self, x):
        pass

