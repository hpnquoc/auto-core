import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import AWR

from option import options as opt

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    # Network Construction
    net = AWR(encoder='vits', features=64, out_channels=[48, 96, 192, 384], use_bn=False, use_clstoken=False).cuda()
    net.train()

    # Load pre-trained model
    net.load_state_dict(torch.load('ckpt/depth_anything/Depth-Anything-V2-small-hf/depth_anything_v2_vits.pth'), strict=False)
    net.pretrained.requires_grad_(False)
    net.depth_head.requires_grad_(False)



    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    l1 = nn.L1Loss().cuda()

    if opt.resume:
        checkpoint = torch.load(opt.ckpt_path + 'last.pth')
        net.load_state_dict(checkpoint['net'])
        net.light_enhancer.requires_grad_(False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    l1_best = 1000
    # Start training
    print('Start training...')
    for epoch in range(start_epoch, opt.epochs):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            optimizer.zero_grad()

            restored = net(x=degrad_patch_1)
            l1_loss = l1(restored, clean_patch_1)
            loss = l1_loss
            # backward
            loss.backward()
            optimizer.step()

        print(
                'Epoch (%d)  Loss: l1_loss:%0.4f\n' % (
                    epoch, l1_loss.item()
                ), '\r', end='')

        GPUS = 1
        if (epoch + 1) % 50 == 0:
            
            if GPUS == 1:
                checkpoint = {
                    "net": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
                torch.save(checkpoint, opt.ckpt_path + 'last.pth')
            else:
                checkpoint = {
                    "net": net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
                torch.save(checkpoint, opt.ckpt_path + 'last.pth')
        if l1_loss.item() < l1_best:
            l1_best = l1_loss.item()
            if GPUS == 1:
                checkpoint = {
                    "net": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, opt.ckpt_path + 'best.pth')
            else:
                checkpoint = {
                    "net": net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, opt.ckpt_path + 'best.pth')

        if epoch <= opt.epochs_encoder:
            lr = opt.lr * (0.1 ** (epoch // 60))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
