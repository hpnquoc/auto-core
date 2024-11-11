import torch
from torch import nn
from einops import rearrange
import traceback

try:
    from net.modules.layers.activation import SimpleGate
    from net.modules.layers.attention import SimpleChannelAttention, Residual, PreNorm, LayerNorm, SpatialTransformer
except ImportError:
    traceback.print_exc()
    pass


class NAFBlock(nn.Module):
    def __init__(self, c, emb_dim=None, att_type='sca', context_dim=512, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(emb_dim // 2, c * 4)
        ) if emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        dim = dw_channel // 2
        dim_head = 32
        # Attention
        if att_type == 'simple':
            self.att = SimpleChannelAttention(dim)
        elif att_type == 'cross':
            num_heads = dim // dim_head
            self.att = Residual(PreNorm(dim, SpatialTransformer(dim, num_heads, dim_head, depth=1, context_dim=context_dim)))

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def degra_forward(self, degra, mlp):
        degra_emb = mlp(degra)  
        degra_emb = rearrange(degra_emb, 'b c -> b c 1 1')
        return degra_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, degra, context = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.degra_forward(degra, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.att(x, context)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, degra, context