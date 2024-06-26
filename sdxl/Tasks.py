from diffusers import DiffusionPipeline
import torch
import ImageReward as RM
from PIL import Image
import time
from diffusers import (
    DiffusionPipeline,
)
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import base64
import io

def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    if format not in ["JPEG", "PNG"]:
        format = "JPEG"
    image_stream = io.BytesIO()
    image = image.convert("RGB")
    image.save(image_stream, format=format)
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return base64_image

scoring_model = RM.load("ImageReward-v1.0")

redis_async_result = RedisAsyncResultBackend(
    redis_url="redis://localhost:6379",
)

# Or you can use PubSubBroker if you need broadcasting
broker = ListQueueBroker(
    url="redis://localhost:6379",
    result_backend=redis_async_result,
)

@broker.task
async def generate_image(t2i_model, prompt: str):
    t2i_model.to("cuda")
    print(prompt)

    start_time = time.time()
    images = t2i_model(prompt, num_inference_steps=35).images
    score = scoring_model.score(prompt, images)
    print(score)
    end_time = time.time()
    print(f"{end_time - start_time} seconds")
    # Note: encode <class 'PIL.Image.Image'>
    base64_image = pil_image_to_base64(images[0])
    print("All problems are solved!")
    # return images, score
    return {"prompt": prompt, "score": score, "image": base64_image}
