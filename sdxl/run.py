import asyncio
from diffusers import DiffusionPipeline
import torch
import ImageReward as RM
from PIL import Image
import time
from diffusers import (
    DiffusionPipeline,
)
import Tasks
import base64
import io
import os
import redis

def base64_to_pil_image(base64_image):
    image = base64.b64decode(base64_image)
    image = io.BytesIO(image)
    image = Image.open(image)
    image = image.convert("RGB")
    return image


def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    if format not in ["JPEG", "PNG"]:
        format = "JPEG"
    image_stream = io.BytesIO()
    image = image.convert("RGB")
    image.save(image_stream, format=format)
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return base64_image

scoring_model = RM.load("ImageReward-v1.0")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe.load_lora_weights("checkpoint-5000", weight_name="pytorch_lora_weights.safetensors", adapter_name="dpo-lora")
pipe.set_adapters(["dpo-lora"], adapter_weights=[0.9])

prompt = "A mischievous goblin finds a shiny, magical stone on a farm, causing the crops to grow disproportionately large overnight."
print(prompt)
async def main():
    while True:
        start_time = time.time()
        top_score = -3
        top_images = None
        tasks = []
        results = []
        num_images = 2
        for i in range(num_images):
            # guidance_scale = random.uniform(7.5, 10)
            guidance_scale = 7.5
            task = await Tasks.generate_image.kiq(pipe, prompt)
            tasks.append(task)
            
        
        generated_images_number = 0
        for task in tasks:                    
            tmp_time = time.time()
            
            result = await task.wait_result()
            
            results.append(result.return_value)
            generated_images_number += 1
            print(f"{generated_images_number-1}-{result.return_value['score']}")

        
        top_score = max(results, key=lambda x: x["score"])["score"]
        top_image = max(results, key=lambda x: x["score"])["image"]

        decoded_image = base64_to_pil_image(top_image)

        # images = model(**local_args).images
        images = []
        images.append(decoded_image)
        
        end_time = time.time()
        os.makedirs("output", exist_ok=True)
        decoded_image.save(f"output/{scoring_model.score(prompt, top_image)}-{top_score}.png")

        end_time = time.time()
        print(f"{end_time - start_time} seconds")
        print(top_score)
        top_images[0].save(f"{top_score}-DPO-{prompt}.png")
        asyncio.sleep(5)
if __name__ == "__main__":
    main()