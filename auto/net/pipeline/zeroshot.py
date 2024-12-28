import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os

from net.modules.backbone.skip import Skip
from net.modules.loss import StdLoss

__all__ = [
    'ZeroShot',
    'ZeroshotDH',
]


class ZeroShot(abc.ABC):
    def __init__(self, device=None):
        self.device = device

    @abc.abstractmethod
    def _init_networks(self):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    @abc.abstractmethod
    def forward_step(self, x):
        pass

    ################################################################################
    def forward(self, x):
        x = self.forward_step(x)
        return x
    

#############################################################################


class ZeroshotDH(ZeroShot):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(
            self,
            optimizer:torch.optim.Optimizer,
            device=None
        ):
        super().__init__(device)

        self.optimizer = optimizer
        self.input_depth = 3
    
    def _init_inet(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        self.inet = Skip(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            filter_size_down=3,
            filter_size_up=3,
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            upsample_mode='bilinear',
            downsample_mode='stride',
            act_fun='LeakyReLU',
            need1x1_up=True
        )
        self.inet = self.inet.type(data_type)


    def _init_tnet(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        self.tnet = Skip(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            filter_size_down=3,
            filter_size_up=3,
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            upsample_mode='bilinear',
            downsample_mode='stride',
            act_fun='LeakyReLU',
            need1x1_up=True
        )
        self.tnet = self.inet.type(data_type)

    def _init_anet(self):
        pass

    def _init_networks(self):
        self._init_inet()
        self._init_tnet()
        self._init_anet()

    def _init_loss(self):
        data_type = torch.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def fit(
            self,
            num_iter,
        ):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        for i in range(num_iter):
            self.optimizer.zero_grad()
            self._opimization_step()
            self.optimizer.step()
        pass

    def _opimization_step(self, x):
        self.out_i = self.inet(x)
        self.out_t = self.tnet(x)


        pass

    def load(self, path):
        pass

    def save(self, path):
        pass

    def score_fn(self, x, t):
        pass

    def forward_step(self, x):
        pass
