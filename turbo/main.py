from diffusers import AutoPipelineForText2Image
import torch
import ImageReward as RM
import time
scoring_model = RM.load("ImageReward-v1.0")

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# prompt = "A bustling 1920s New York City street with vintage cars and people dressed in classic flapper fashion."
prompt = "A bustling farm market overflowing with fresh fall produce on a sunny day"
start_time = time.time()
images = None
score = -3
num_images = 16
for idx in range(num_images):
    tmp_images = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=0.0, width=1024, height=1024).images
    tmp_image = tmp_images[0]

    # image.save(f"{prompt}png")

    tmp_score = scoring_model.score(prompt, tmp_images)
    if score < tmp_score:
        images = tmp_images
        score = tmp_score
    print(tmp_score)
images[0].save(f"{prompt}.png")

end_time = time.time()
print(f"{num_images} images are generated in {end_time - start_time} seconds and final score is {score}")