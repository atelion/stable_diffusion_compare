import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)
import ImageReward as RM
from PIL import Image
import time


scoring_model = RM.load("ImageReward-v1.0")
# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "dataautogpt3/ProteusV0.2", 
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# Define prompts and generate image
prompt = "A playful octopus is joyfully juggling colorful coral pieces, fully submerged in the gleaming emerald waters of a tranquil lake."
negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

print(prompt)

start_time = time.time()
top_score = -3
top_images = None
for idx in range(1):
    # Note: Full size
    images = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images

    # Note: Half size
    # images = pipe(prompt, num_inference_steps=50, height=512, width=512).images
    # images[0] = upscale_image(images[0], 2)

    # images[0].save(f"{idx}-{prompt}.png")
    score = scoring_model.score(prompt, images)
    if top_score < score:
        top_score = score
        top_images = images    
    print(score)
end_time = time.time()
print(f"{end_time - start_time} seconds")
print(top_score)
top_images[0].save(f"{top_score}-{prompt}.png")


current_score = scoring_model.score(prompt, 'img.png')
print("Current Score is : ")
print(current_score)