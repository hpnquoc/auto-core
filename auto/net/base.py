import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.is_train = opt["is_train"]
        self.schedulers = []
        self.optimizers = []

    def feed_data():
        pass
    
    def optimize_parameters():
        pass

    def get_current_visuals():
        pass

    def get_current_losses():
        pass

    def print_network():
        pass

    def save(self, iter):
        pass

    def load(self):
        pass

    def save_network(self, network, network_label, iter_label):
        _filename = f"{iter_label}_{network_label}.pth"
        _save_path = os.path.join(self.opt["path"][network.__name__], _filename)