from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy


from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.output_tokens = True
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)
        
    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, tokens, hiddens = self.visual_control(image, output_hiddens=True)
            tokens = tokens.permute(0, 2, 1)
            tokens = tokens.reshape(tokens.size(0), tokens.size(1), 7, 7).squeeze(0)
            print(tokens.shape)
            # tokens = tokens.reshape(192, 14, 14)
            print(tokens[0].shape)
            import cv2
            for i in range(tokens.size(0)):
                t = tokens[i].cpu().detach().numpy()
                t = cv2.resize(t, (224, 224), interpolation=cv2.INTER_NEAREST)
                t = np.array([t, t, t])
                t = t.transpose(1, 2, 0)

                i = image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
                # temp = image[:, :, 0]
                # image[:, :, 0] = image[:, :, 2]
                # image[:, :, 2] = temp
                print(i.shape)
                t = np.concatenate([i, t], axis=1)
                print(t.shape)
                cv2.imshow('tokens', t)
                cv2.waitKey(0)
            image_features = self.clip.visual(image, control=hiddens)
            
            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }


