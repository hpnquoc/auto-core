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
parser.add_argument("--dir_image", type=str, default='dummy/images/', help="Path to image file.")
parser.add_argument("--dir_output", type=str, default='output/', help="Path to output file.")
parser.add_argument("--ext", type=str, default='jpg', help="Extension of image file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = create_model(opt)
# device = model.device

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

# sde = pipeline.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
# sde.set_model(model.model)

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

import math
from torchvision.transforms import functional as F

def resize(image, size):
    #resize keeping aspect ratio
    h, w, c = image.shape
    aspect_ratio = float(h) / float(w)
    new_w = math.ceil(size / aspect_ratio)
    return F.resize(torch.tensor(image, dtype=torch.float32).permute(2,0,1), (size, new_w)).permute(1,2,0)

def infer(image):
    image = image / 255.

    # Encode image
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()

    # Resize image to fixed height
    image = resize(image, 256)
    print(image_context.shape)
    print(degra_context.shape)
    LQ_tensor = image.permute(2, 0, 1).unsqueeze(0) # B, C, H, W
    # noisy_tensor = sde.noise_state(LQ_tensor)
    # model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    # model.test(sde)
    # visuals = model.get_current_visuals(need_GT=False)
    # output = util.utils_image.tensor2img(visuals["Output"].squeeze())
    # image = (image * 255)
    # return output[:, :, [2, 1, 0]], np.array(image, dtype=np.uint8)

if __name__ == "__main__":
    import cv2
    import numpy as np
    import glob
    img_list = sorted(glob.glob(f"{parser.parse_args().dir_image}/*.{parser.parse_args().ext}"))
    dest = f"{parser.parse_args().dir_output}"
    dir_resized = f"{dest}resized/"
    dir_output = f"{dest}output/"

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dir_resized, exist_ok=True)
    os.makedirs(dir_output, exist_ok=True)
    print(f"Processing {len(img_list)} images...")
    for img in img_list:
        image = cv2.imread(img)
        infer(image)
        # cv2.imshow("output", output)
        # cv2.imshow("input", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(f"{dir_resized}{os.path.basename(img)}", image)
        # cv2.imwrite(f"{dir_output}{os.path.basename(img)}", output)


