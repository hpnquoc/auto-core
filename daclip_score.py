import sys, os
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

sys.path.append(f"{os.getcwd()}/auto")
sys.path.append(f"{os.getcwd()}/auto/add_on/")

import options as option
from net import create_model, pipeline

from add_on import open_clip
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default='auto/configs/test.yml', help="Path to options YMAL file.")
parser.add_argument("--img1", type=str, default='/home/hpnquoc/auto-core/dummy/test/suwon#54_02_01_frame_1.jpg', help="Path to image file.")
parser.add_argument("--img2", type=str, default='/home/hpnquoc/auto-core/dummy/test/suwon#54_01_16_frame_1.jpg', help="Path to image file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

def daclip_score(image1, image2):
    image1 = image1 / 255.
    image2 = image2 / 255.

    # Encode image
    img4clip1 = clip_transform(image1).unsqueeze(0).to(device)
    img4clip2 = clip_transform(image2).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_context1, degra_context1 = clip_model.encode_image(img4clip1, control=True)
        image_context2, degra_context2 = clip_model.encode_image(img4clip2, control=True)
        image_context1 = image_context1.float()
        degra_context1 = degra_context1.float()
        image_context2 = image_context2.float()
        degra_context2 = degra_context2.float()

    print("DaCLIP context score:", clip_model.logit_scale.exp()*torch.functional.F.cosine_similarity(image_context1, image_context2, dim=-1))
    print("DaCLIP degradation score:", clip_model.logit_scale.exp()*torch.functional.F.cosine_similarity(degra_context1, degra_context2, dim=-1))

def daclip_img_text(img, text):
    img = img / 255.

    # Encode image
    img4clip = clip_transform(img).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize(text).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        text_context = clip_model.encode_text(text_tokens)
    # calculate the cosine similarity between the image and text features
    print("DaCLIP context similarity:", clip_model.logit_scale.exp()*torch.functional.F.cosine_similarity(degra_context, text_context, dim=-1))
    # calculate score
    score = clip_model.logit_scale.exp() * (degra_context @ text_context.T).mean()
    print("DaCLIP score:", score)

if __name__ == "__main__":
    import cv2
    import numpy as np
    
    img1 = cv2.imread(parser.parse_args().img1)
    img2 = cv2.imread(parser.parse_args().img2)
    print(img1.shape)
    daclip_img_text(img1, "haze, degadation, foggy, dusty, dust")
    # daclip_score(img1, img2)


