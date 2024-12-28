import os
import sys
import numpy as np
import torch
from PIL import Image
import traceback
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

sys.path.append(f"{os.getcwd()}/auto")
sys.path.append(f"{os.getcwd()}/auto/add_on/")

try:
    from add_on import open_clip
except ImportError:
    traceback.print_exc()
    pass

pretrained_path = "zoo/da-clip/daclip_ViT-B-32.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=pretrained_path)
clip_model = clip_model.to(device)

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

image = Image.open("dummy/images/2.jpg")
image = np.array(image)
image = image / 255.
image = clip_transform(image)
print('in: ', image.shape)
image = image.unsqueeze(0).to(device)
image_context, degra_context = clip_model.encode_image(image, control=True)
image_context = image_context.float()
degra_context = degra_context.float()

print('out: ',image_context.shape, degra_context.shape)