import logging

import torch

from net import modules
logger = logging.getLogger("base")

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    netG = getattr(modules.generator, which_model)(**setting)
    logger.info("Generator [{:s}] is created.".format(netG.__class__.__name__))
    return netG


# Discriminator
def define_D(opt):
    opt_net = opt["network_D"]
    which_model = opt_net["which_model_D"]
    setting = opt_net["setting"]
    netD = getattr(modules, which_model)(**setting)
    logger.info("Discriminator [{:s}] is created.".format(netD.__class__.__name__))
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = modules.feature_extractor.vision.VGG(
        depth=19, with_bn=use_bn, out_indices=feature_layer, device=device
    )
    netF.eval()  # No need to train
    return netF