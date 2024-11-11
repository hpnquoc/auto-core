import gradio as gr
import argparse
import sys, os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import torchvision.utils as tvutils
import traceback

sys.path.append(os.path.join(os.getcwd(), 'auto'))
sys.path.append(os.path.join(os.getcwd(), 'auto', 'add_on'))

try:
    from net import create_model, pipeline
    from add_on import open_clip
    import utils as util
    import options as option
except ImportError:
    traceback.print_exc()
    pass

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='src/configs/test.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)
device = model.device

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

sde = pipeline.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

import math
from torchvision.transforms import functional as F
class FixedHeightResize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        h, w, c = img.shape
        aspect_ratio = float(h) / float(w)
        new_w = math.ceil(self.size / aspect_ratio)
        return F.resize(torch.tensor(img, dtype=torch.float32).permute(2,0,1), (self.size, new_w)).permute(1,2,0)

class FixedWidthResize:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        h, w, c = img.shape
        aspect_ratio = float(w) / float(h)
        new_h = math.ceil(self.size / aspect_ratio)
        return F.resize(torch.tensor(img, dtype=torch.float32).permute(2,0,1), (new_h, self.size)).permute(1,2,0)

def restore(image):
    image = image / 255.

    # Encode image
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()

    # Resize image to fixed height
    h, w, c = image.shape
    if h > 256:
        transform = Compose([
            FixedHeightResize(256)
        ])
        image = transform(image)
    # if w > 640:
    #     transform = Compose([
    #         FixedWidthResize(640)
    #     ])
    #     image = transform(image)
    else:
        image = torch.tensor(image, dtype=torch.float32)

    LQ_tensor = image.permute(2, 0, 1).unsqueeze(0)
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.utils_image.tensor2img(visuals["Output"].squeeze())
    return output[:, :, [2, 1, 0]]

examples=[os.path.join(os.path.dirname(__file__), f"dummy/images/{i}.jpg") for i in range(1, 11)]
interface = gr.Interface(fn=restore, inputs="image", outputs="image", title="Image Restoration with DA-CLIP", examples=examples)
interface.launch()

