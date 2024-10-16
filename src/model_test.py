import torch
from net.model import AirNet
import argparse

opt = argparse.Namespace()
opt.batch_size = 5
checkpoint = torch.load('./ckpt/Denoise/epoch_500.pth')
model = AirNet(opt).cuda()
model.load_state_dict(checkpoint['net'])
input = torch.randn(1, 3, 518, 518).cuda()

fea, logits, labels, inter = model.E(input, input)
print(inter.shape)
print(fea.shape)