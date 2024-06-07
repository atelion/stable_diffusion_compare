from diffusers import DiffusionPipeline
import torch
import ImageReward as RM
import time
import redis
import json
import base64

redis_client = redis.Redis(host="localhost", port="6379", db=0)
first = 1
second = "2"
redis_client.set("35", str([first,second]))
result = eval(redis_client.get("35").decode())
print(result)
# exit()
scoring_model = RM.load("ImageReward-v1.0")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A busy morning at a bustling farmer's market on a sunny day"
# prompt = "A fairy dancing on a dew-covered mushroom in a mystical forest during sunrise"
num_images = 4
# images = pipe(prompt=prompt).images[0]
initial_time = time.time()
start_time = time.time()
images = None
score = -3
num_inference_steps = 35
for idx in range(num_images):
    tmp_images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=10, width=1024, height=1024).images

    # image.save(f"{prompt}png")

    tmp_score = scoring_model.score(prompt, tmp_images)
    if score < tmp_score:
        score = tmp_score
        images = tmp_images
image_base64 = [
    base64.b64encode(img_obj.tobytes()).decode('utf-8')
    for img_obj in images
]

redis_client.set("35", json.dumps([score, image_base64]))
print("-----------------------35------------------------")
# images[0].save(f"{prompt}.png")
end_time = time.time()
print(f"{end_time-start_time} : {num_inference_steps} : {score}")

start_time = time.time()
num_inference_steps = 50
images = None
score = -3
for idx in range(num_images):
    tmp_images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5, width=1024, height=1024).images

    # image.save(f"{prompt}png")

    tmp_score = scoring_model.score(prompt, tmp_images)
    if score < tmp_score:
        score = tmp_score
        images = tmp_images

image_base64 = [
    base64.b64encode(img_obj.tobytes()).decode('utf-8')
    for img_obj in images
]
redis_client.set("50", json.dumps([score, image_base64]))
print("-----------------------50------------------------")
end_time = time.time()
print(f"{end_time-start_time} - {num_inference_steps} : {score}")

images_35 = json.loads(redis_client.get("35").decode())
images_50 = json.loads(redis_client.get("50").decode())
if images_35[0] > images_50[0]:
    print("35 is better")
else:
    print("50 is better")
final_time = time.time()
print(f"{final_time-initial_time} seconds")