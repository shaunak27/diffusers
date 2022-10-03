import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import os
import random
from tqdm import tqdm
model_id = "CompVis/stable-diffusion-v1-4"
model_path = "./textual_inversion_imr_train/"

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
data_root = "/home/storage/ssd1/datasets/imagenet-r-train/"
target_path = "/home/storage/ssd1/datasets/imagenet-r-synthetic/"
folders = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
image_paths = []
for dir in folders:
        image_paths.extend([os.path.join(dir,file_name) for file_name in os.listdir(dir)])
text_encoder = CLIPTextModel.from_pretrained(
    model_path, subfolder="text_encoder", use_auth_token=True
)
vae = AutoencoderKL.from_pretrained(
    model_path, subfolder="vae", use_auth_token=True
)
unet = UNet2DConditionModel.from_pretrained(
    model_path, subfolder="unet", use_auth_token=True
)
tokenizer = CLIPTokenizer.from_pretrained(
    model_path, subfolder="tokenizer", use_auth_token=True
)
pipe = StableDiffusionPipeline(text_encoder=text_encoder,vae=vae,unet=unet,tokenizer=tokenizer, 
scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),).to("cuda")

for i,img in tqdm(enumerate(image_paths),total=len(image_paths)):
    img_name = '/'.join(img.split('/')[-2:])
    prompt = random.choice(imagenet_templates_small).format('<' + img.split('/')[-2] + '*>')
    with autocast("cuda"):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]  
    if not os.path.exists(target_path+img.split('/')[-2]):
        os.makedirs(target_path+img.split('/')[-2])
    image.save(target_path+img_name)


