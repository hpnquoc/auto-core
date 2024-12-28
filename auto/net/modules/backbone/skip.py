import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger as logging

try:
    import core
except ImportError:
    logging.error("Please make sure to run your code from the root directory of the project.")

__all__ = [
    'Skip',
]

class Skip(nn.Module):
    def __init__(
        self,
        num_input_channels=3,
        num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='zero',
        upsample_mode='bilinear',
        downsample_mode='stride',
        act_fun='LeakyReLU',
        need1x1_up=True
    ):
        super(Skip, self).__init__()

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.need_sigmoid = need_sigmoid
        self.pad = pad
        self.act_fun = getattr(nn, act_fun)() if isinstance(act_fun, str) else act_fun

        n_scales = len(num_channels_down)

        if not isinstance(upsample_mode, (list, tuple)):
            upsample_mode = [upsample_mode] * n_scales

        if not isinstance(downsample_mode, (list, tuple)):
            downsample_mode = [downsample_mode] * n_scales

        if not isinstance(filter_size_down, (list, tuple)):
            filter_size_down = [filter_size_down] * n_scales

        if not isinstance(filter_size_up, (list, tuple)):
            filter_size_up = [filter_size_up] * n_scales

        self.model = nn.Sequential()
        model_tmp = self.model

        input_depth = num_input_channels
        for i in range(n_scales):
            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add_module(f"concat_{i}", Concat(1, skip, deeper))
            else:
                model_tmp.add_module(f"deeper_{i}", deeper)

            model_tmp.add_module(f"bn_{i}", nn.BatchNorm2d(num_channels_skip[i] + (num_channels_up[i + 1] if i < n_scales - 1 else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add_module(f"conv_skip_{i}", nn.Conv2d(input_depth, num_channels_skip[i], filter_skip_size, padding=1, bias=need_bias))
                skip.add_module(f"bn_skip_{i}", nn.BatchNorm2d(num_channels_skip[i]))
                skip.add_module(f"act_skip_{i}", self.act_fun)

            deeper.add_module(f"conv_down_1_{i}", nn.Conv2d(input_depth, num_channels_down[i], filter_size_down[i], stride=2, padding=1, bias=need_bias))
            deeper.add_module(f"bn_down_1_{i}", nn.BatchNorm2d(num_channels_down[i]))
            deeper.add_module(f"act_down_1_{i}", self.act_fun)

            deeper.add_module(f"conv_down_2_{i}", nn.Conv2d(num_channels_down[i], num_channels_down[i], filter_size_down[i], padding=1, bias=need_bias))
            deeper.add_module(f"bn_down_2_{i}", nn.BatchNorm2d(num_channels_down[i]))
            deeper.add_module(f"act_down_2_{i}", self.act_fun)

            deeper_main = nn.Sequential()

            if i == n_scales - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add_module(f"deeper_main_{i}", deeper_main)
                k = num_channels_up[i + 1]

            deeper.add_module(f"upsample_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

            model_tmp.add_module(f"conv_up_{i}", nn.Conv2d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], stride=1, bias=need_bias, padding=1))
            model_tmp.add_module(f"bn_up_{i}", nn.BatchNorm2d(num_channels_up[i]))
            model_tmp.add_module(f"act_up_{i}", self.act_fun)

            if need1x1_up:
                model_tmp.add_module(f"conv_1x1_up_{i}", nn.Conv2d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias))
                model_tmp.add_module(f"bn_1x1_up_{i}", nn.BatchNorm2d(num_channels_up[i]))
                model_tmp.add_module(f"act_1x1_up_{i}", self.act_fun)

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

            input_depth = num_channels_down[i]
            model_tmp = deeper

        self.model.add_module("final_conv", nn.Conv2d(num_channels_up[0], num_output_channels, 1, bias=need_bias))
        if need_sigmoid:
            self.model.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

    def _get_padding(self, filter_size):
        if self.pad == 'zero':
            return filter_size // 2
        elif self.pad == 'reflection':
            return nn.ReflectionPad2d(filter_size // 2)
        else:
            raise ValueError("Unsupported padding type")

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module_ in enumerate(args):
            self.add_module(str(idx), module_)

    def forward(self, input_):
        inputs = []
        for module_ in self._modules.values():
            inputs.append(module_(input_))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                        np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)
