from torch import nn
from net.moco import MoCo
from add_on.dinov2.dinov2 import DINOv2

class DDE(nn.Module):
    def __init__(
            self,
            opt,
            encoder='vitl'
        ):
        super(DDE, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        dim = 1024

        # Encoder
        self.E = MoCo(base_encoder=DINOv2(model_name=encoder), dim=dim, K=opt.batch_size * dim, intermediate_layers=self.intermediate_layer_idx[encoder])

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            return fea, logits, labels, inter
        else:
            fea, inter = self.E(x_query, x_query)
            return fea, inter