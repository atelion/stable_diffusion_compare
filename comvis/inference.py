import time
from diffusers import StableDiffusionPipeline
import torch
import ImageReward as RM
from PIL import Image
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
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "A medieval castle surrounded by a moat with dragons flying overhead."
# prompt = "A serene forest with ancient trees and a carpet of bluebells."
# prompt = "A bustling 1950s diner scene with Elvis Presley ordering a burger."
prompt = "A futuristic cityscape bathed in the multicolored lights of neon signs and holograms."

start_time = time.time()
for idx in range(8):
    images = pipe(prompt, num_inference_steps=35, height=1024, width=1024).images
    # images = pipe(prompt, num_inference_steps=35).images
    # images[0] = upscale_image(images[0], 2)
    # images[0].save(f"{idx}-{prompt}.png")
    score = scoring_model.score(prompt, images)
    print(score)
end_time = time.time()
print(f"{end_time-start_time}")