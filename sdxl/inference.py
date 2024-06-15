from diffusers import DiffusionPipeline
import torch
import ImageReward as RM
from PIL import Image
import time

def upscale_image(input_image, scale_factor):
    """
    Increases the resolution of an image by the given scale factor.
    
    Args:
        input_image (numpy.ndarray): The input image.
        scale_factor (float): The scale factor to use for upscaling.
    
    Returns:
        numpy.ndarray: The upscaled image.
    """
    width = int(input_image.width * scale_factor)
    height = int(input_image.height * scale_factor)
    upscaled_image = input_image.resize((width, height), resample=Image.BICUBIC)
    return upscaled_image

scoring_model = RM.load("ImageReward-v1.0")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# prompt = "A medieval castle surrounded by a moat with dragons flying overhead."
# prompt = "A serene forest with ancient trees and a carpet of bluebells."
# prompt = "A bustling 1950s diner scene with Elvis Presley ordering a burger."
prompt = "A futuristic cityscape bathed in the multicolored lights of neon signs and holograms."
print(prompt)

start_time = time.time()
top_score = -3
top_images = None
for idx in range(8):
    # Note: Full size
    # images = pipe(prompt, num_inference_steps=35).images

    # Note: Half size
    images = pipe(prompt, num_inference_steps=50, height=512, width=512).images
    images[0] = upscale_image(images[0], 2)

    # images[0].save(f"{idx}-{prompt}.png")
    score = scoring_model.score(prompt, images)
    if top_score < score:
        top_score = score
        top_images = images    
    print(score)
end_time = time.time()
print(f"{end_time - start_time} seconds")
print(top_score)
top_images[0].save(f"{top_score}.png")