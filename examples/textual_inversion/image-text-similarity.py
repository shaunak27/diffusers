import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import os
from tqdm import tqdm
import random
from operator import itemgetter
import json

model_id = "CompVis/stable-diffusion-v1-4"
model_path = "./textual_inversion_imr_train_init/"
#model_path = model_id
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]
data_root = "/home/storage/ssd1/datasets/imagenet-r-test/"
folders = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
prompt_list = [file_path for file_path in os.listdir(data_root)]
image_paths = []
for dir in folders:
        image_paths.extend([os.path.join(dir,file_name) for file_name in os.listdir(dir)])

device = torch.device("cuda:1")

text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder", use_auth_token=True
).to(device)

tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer", use_auth_token=True
)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

template = "A photo of a {}"
accuracy = 0
pbar = tqdm(enumerate(image_paths),total=len(image_paths))

prompts = [template.format('<'+p+'*>') for p in prompt_list]
inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
pooled_output = text_encoder(**inputs)[1]
text_features = model.text_projection(pooled_output)
text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
logit_scale = model.logit_scale.exp()

for i, img in pbar:
    image = Image.open(img)
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    img_features = model.get_image_features(**img_inputs) 
    image_embeds = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    
    idx, ele = max(enumerate(logits_per_text), key=itemgetter(1))

    if prompt_list[idx] == img.split('/')[-2]:
        accuracy += 1
    pbar.set_postfix({'Accuracy':accuracy/(i+1)})

print("Total Accuracy : ", accuracy/6000.0)