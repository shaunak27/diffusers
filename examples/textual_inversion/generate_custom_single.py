import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import os
import random
from tqdm import tqdm

model_id = "CompVis/stable-diffusion-v1-4"
model_path = "./textual_inversion_imr_test/"

target_path = './generated_images/new.jpg'

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
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),).to("cuda:1")

prompt = "A photo of a <n04552348*>"

with autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]  

image.save(target_path)


