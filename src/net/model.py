import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from net.DGRN import DGRN
from net.encoder import DDE

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet

class DinoDecode(nn.Module):
    def __init__(self, in_shape, out_shape, groups=1, expand=False):
        super(DinoDecode, self).__init__()
        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        if len(in_shape) >= 4:
            out_shape4 = out_shape

        if expand:
            out_shape1 = out_shape
            out_shape2 = out_shape * 2
            out_shape3 = out_shape * 4
            if len(in_shape) >= 4:
                out_shape4 = out_shape * 8

        # channels compression
        self.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        if len(in_shape) >= 4:
            self.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

        # feature fusion
        self.refinenet1 = FeatureFusionBlock(
            out_shape,
            nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet2 = FeatureFusionBlock(
            out_shape,
            nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet3 = FeatureFusionBlock(
            out_shape,
            nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet4 = FeatureFusionBlock(
            out_shape,
            nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )

        head_features_1 = out_shape
        head_features_2 = 32

        self.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, out_shape, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, x, patch_h, patch_w):
        layer_1, layer_2, layer_3, layer_4 = x
        
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        
        path_4 = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1_rn)
        
        out = self.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.output_conv2(out)
        
        return out


class Fusion(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(Fusion, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = DinoDecode(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        out = self.scratch(out, patch_h, patch_w)
        
        return out


class AWIR(nn.Module):
    def __init__(
            self,
            opt,
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False
        ):
        super(AWIR, self).__init__()

        # Restorer
        self.R = DGRN(opt)

        # Fuser
        self.F = Fusion(DINOv2(model_name=encoder).embed_dim,features=features, out_channels=out_channels, use_bn=use_bn, use_clstoken=use_clstoken)

        # Encoder
        self.E = DDE(opt, encoder=encoder)

    def forward(self, x_query, x_key):
        patch_h, patch_w = x_query.shape[-2] // 14, x_query.shape[-1] // 14
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            inter = self.F(inter, patch_h, patch_w)

            restored = self.R(x_query, inter)


            return restored, logits, labels
        
        else:
            fea, inter = self.E(x_query, x_query)
            
            inter = self.F(inter, patch_h, patch_w)

            restored = self.R(x_query, inter)

            # return enhanced_image
            return restored
        
    def infer_image(self, x_query, x_key, input_size = 512):
        self.training = False
        restored = self.forward(x_query, x_key)
        return restored